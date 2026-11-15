#!/usr/bin/env python3
"""R1の関節軌道をオフラインで検査し、100 Hzのプレビューを作る。

このモジュールにはDDS、SDK、モーター送信、制御権の切り替えを実装しない。
機体ごとの関節限界・速度限界は推測せず、出典を確認したprofileを入力する。
crc_valid等は入力スナップショット作成側の検証結果であり、ここでDDSのCRCを
再計算したという意味ではない。プレビューの成功は実機実行の許可ではない。

JSONスキーマ（位置rad・速度rad/s・時刻秒）:
  profile: schema_version=1, profile_id, model_id, mode_machine,
    max_state_age_s(0,1], seed_tolerance_rad[0,.05],
    source={verified:true,joint_layout_verified:true,reference,firmware},
    joints=[{name,index,lower_rad,upper_rad,max_velocity_rad_s}, ...]
  state: profile_id,model_id,mode_machine,source="rt/lowstate",crc_valid=true,
    capture_time(Unix時刻),joints=[{name,index,q_rad}, ...]
  trajectory: schema_version=1,profile_id,
    waypoints=[{time_s:0,positions_rad:{関節名:現在角}}, ...]

stateはprofileの全関節を含める。軌道の対象関節は一部でもよく、残りの関節は
測定角を保持したプレビューとなる。各waypointは同じ対象関節を含める。
実行例: python r1_motion/joint_control.py --profile p.json --state s.json
        --trajectory t.json --output preview.json
実機向けrt/lowcmdは別の制御方式であり、この100 Hzプレビューを直接送信しない。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time


SAMPLE_RATE_HZ = 100
MAX_DURATION_S = 30.0
MAX_WAYPOINTS = 2001
MAX_JSON_BYTES = 2_000_000
OFFICIAL_SOURCES = (
    "https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/dds_wrapper/robots/r1/defines.h",
    "https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/r1/low_level/r1_A5_low_level_example.py",
    "https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/r1/low_level/r1_A7_low_level_example.py",
)


class ValidationError(ValueError):
    """実機の命令へ変換する前に解消すべき入力の不整合。"""


@dataclass(frozen=True)
class JointLimit:
    name: str
    index: int
    lower: float
    upper: float
    max_velocity: float


def _object(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} はJSONオブジェクトにしてください")
    return value


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValidationError(f"{label} は空でない文字列にしてください")
    if value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValidationError(f"{label} に空白や制御文字を含めないでください")
    return value


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} は有限の数値にしてください")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValidationError(f"{label} は有限の数値にしてください") from None
    if not math.isfinite(number):
        raise ValidationError(f"{label} は有限の数値にしてください")
    return number


def _integer(value, label: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
        raise ValidationError(f"{label} は0から{upper}の整数にしてください")
    return value


def _same(value, expected, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValidationError(f"{label} がprofileと一致しません")


def _position(q, joint: JointLimit, label: str) -> float:
    q = _number(q, label)
    if not joint.lower <= q <= joint.upper:
        raise ValidationError(f"{joint.name}: {label} が関節限界を超えています")
    return q


def _profile(profile: dict):
    profile = _object(profile, "profile")
    _same(profile.get("schema_version"), 1, "profile.schema_version")
    profile_id = _text(profile.get("profile_id"), "profile_id")
    model_id = _text(profile.get("model_id"), "model_id")
    machine = _integer(profile.get("mode_machine"), "mode_machine", 255)
    source = _object(profile.get("source"), "profile.source")
    if source.get("verified") is not True or source.get("joint_layout_verified") is not True:
        raise ValidationError("機体型式・関節配列の確認済みprofileが必要です")
    _text(source.get("reference"), "profile.source.reference")
    _text(source.get("firmware"), "profile.source.firmware")
    max_age = _number(profile.get("max_state_age_s"), "max_state_age_s")
    tolerance = _number(profile.get("seed_tolerance_rad"), "seed_tolerance_rad")
    if not 0 < max_age <= 1.0 or not 0 <= tolerance <= 0.05:
        raise ValidationError("状態の最大経過時間は1秒以内、初期角の許容差は0.05rad以内にしてください")
    rows = profile.get("joints")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 35:
        raise ValidationError("profile.joints は1〜35関節の配列にしてください")
    joints, indices = {}, set()
    for row in rows:
        row = _object(row, "profile.joints[]")
        name = _text(row.get("name"), "joint.name")
        index = _integer(row.get("index"), f"{name}.index", 34)
        if name in joints or index in indices:
            raise ValidationError("関節名・モーターindexの重複があります")
        lower = _number(row.get("lower_rad"), f"{name}.lower_rad")
        upper = _number(row.get("upper_rad"), f"{name}.upper_rad")
        velocity = _number(row.get("max_velocity_rad_s"), f"{name}.max_velocity_rad_s")
        if lower >= upper or velocity <= 0:
            raise ValidationError(f"{name}: 関節限界または最大速度が不正です")
        joints[name] = JointLimit(name, index, lower, upper, velocity)
        indices.add(index)
    return profile_id, model_id, machine, max_age, tolerance, joints


def _state(state, profile_id, model_id, machine, max_age, joints, now):
    state = _object(state, "state")
    for key, expected in (("profile_id", profile_id), ("model_id", model_id), ("mode_machine", machine)):
        _same(state.get(key), expected, f"state.{key}")
    if state.get("source") != "rt/lowstate" or state.get("crc_valid") is not True:
        raise ValidationError("CRC検証済みrt/lowstate由来の測定状態が必要です")
    capture_time = _number(state.get("capture_time"), "state.capture_time")
    age = now - capture_time
    if capture_time < 0 or not math.isfinite(age) or age < 0 or age > max_age:
        raise ValidationError("測定状態が古いか、未来の時刻になっています")
    rows = state.get("joints")
    if not isinstance(rows, list):
        raise ValidationError("state.joints は配列にしてください")
    measured, indices = {}, set()
    for row in rows:
        row = _object(row, "state.joints[]")
        name = row.get("name")
        if not isinstance(name, str) or name not in joints:
            raise ValidationError("測定状態に未知の関節があります")
        index = _integer(row.get("index"), f"state.{name}.index", 34)
        if name in measured or index in indices:
            raise ValidationError("測定状態の関節名・indexが重複しています")
        if index != joints[name].index:
            raise ValidationError(f"{name}: 測定indexがprofileと一致しません")
        measured[name] = _position(row.get("q_rad"), joints[name], "測定角")
        indices.add(index)
    if set(measured) != set(joints):
        raise ValidationError("profileの全関節の測定角が必要です")
    return capture_time, measured


def _waypoints(trajectory, profile_id, joints, measured, tolerance):
    trajectory = _object(trajectory, "trajectory")
    if set(trajectory) != {"schema_version", "profile_id", "waypoints"}:
        raise ValidationError("trajectoryにはschema_version・profile_id・waypointsだけを指定してください")
    _same(trajectory.get("schema_version"), 1, "trajectory.schema_version")
    _same(trajectory.get("profile_id"), profile_id, "trajectory.profile_id")
    rows = trajectory.get("waypoints")
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_WAYPOINTS:
        raise ValidationError(f"waypointsは2〜{MAX_WAYPOINTS}点にしてください")
    targets, points = None, []
    for number, row in enumerate(rows):
        row = _object(row, "waypoint")
        if set(row) != {"time_s", "positions_rad"}:
            raise ValidationError("waypointにはtime_s・positions_radだけを指定してください")
        stamp = _number(row.get("time_s"), "waypoint.time_s")
        if (number == 0 and stamp != 0) or (points and stamp <= points[-1][0]):
            raise ValidationError("軌道の時刻は0秒から始め、単調増加にしてください")
        if not 0 <= stamp <= MAX_DURATION_S:
            raise ValidationError(f"プレビューの長さは{MAX_DURATION_S:g}秒以内にしてください")
        values = _object(row.get("positions_rad"), "waypoint.positions_rad")
        if not values or any(name not in joints for name in values):
            raise ValidationError("軌道に未知の関節があるか、対象関節が空です")
        if targets is None:
            targets = set(values)
        elif set(values) != targets:
            raise ValidationError("全waypointで同じ対象関節を指定してください")
        positions = {name: _position(value, joints[name], "目標角") for name, value in values.items()}
        if number == 0:
            if any(abs(positions[name] - measured[name]) > tolerance for name in targets):
                raise ValidationError("最初の目標角が測定角と連続していません")
            # 許容差内であっても、プレビューの初期フレームは実測角から開始する。
            positions = {name: measured[name] for name in targets}
        if points:
            previous_time, previous_q = points[-1]
            dt = stamp - previous_time
            for name in targets:
                speed = abs(positions[name] - previous_q[name]) / dt
                if not math.isfinite(speed) or speed > joints[name].max_velocity * (1 + 1e-9):
                    raise ValidationError(f"{name}: 区間の速度が最大速度を超えています")
        points.append((stamp, positions))
    return targets, points


def validate_and_preview(profile: dict, state: dict, trajectory: dict, *, now: float | None = None) -> dict:
    """入力を検査して位置・速度のプレビューを返す。実機には一切送信しない。"""
    now = _number(time.time() if now is None else now, "now")
    profile_id, model_id, machine, max_age, tolerance, joints = _profile(profile)
    capture_time, measured = _state(state, profile_id, model_id, machine, max_age, joints, now)
    targets, points = _waypoints(trajectory, profile_id, joints, measured, tolerance)
    ordered_names = [joint.name for joint in sorted(joints.values(), key=lambda joint: joint.index)]
    duration = points[-1][0]
    stamps = [index / SAMPLE_RATE_HZ for index in range(math.floor(duration * SAMPLE_RATE_HZ) + 1)]
    if stamps[-1] < duration:
        stamps.append(duration)
    segment, frames = 0, []
    for stamp in stamps:
        while segment < len(points) - 2 and stamp > points[segment + 1][0]:
            segment += 1
        first_t, first_q = points[segment]
        last_t, last_q = points[segment + 1]
        ratio = min(1.0, max(0.0, (stamp - first_t) / (last_t - first_t)))
        positions, velocities = dict(measured), {name: 0.0 for name in ordered_names}
        for name in targets:
            positions[name] = _position(first_q[name] * (1 - ratio) + last_q[name] * ratio,
                                        joints[name], "補間角")
        if frames:
            dt = stamp - frames[-1]["time_s"]
            if dt <= 0:
                raise ValidationError("補間時刻が重複しています")
            for name in ordered_names:
                velocity = (positions[name] - frames[-1]["positions_rad"][name]) / dt
                if not math.isfinite(velocity) or abs(velocity) > joints[name].max_velocity * (1 + 1e-9) + 1e-12:
                    raise ValidationError(f"{name}: 補間フレームの速度が上限を超えています")
                velocities[name] = velocity
        frames.append({"time_s": stamp, "positions_rad": {name: positions[name] for name in ordered_names},
                       "velocities_rad_s": velocities})
    return {
        "schema_version": 1, "artifact_kind": "joint_trajectory_preview", "executable": False,
        "profile_id": profile_id, "model_id": model_id, "mode_machine": machine,
        "sample_rate_hz": SAMPLE_RATE_HZ, "duration_s": duration,
        "measured_at": capture_time, "validated_at": now, "frame_count": len(frames),
        "controlled_joints": [name for name in ordered_names if name in targets],
        "held_joints": [name for name in ordered_names if name not in targets],
        "joint_indices": {name: joints[name].index for name in ordered_names},
        "validation": {"position_limits": True, "velocity_limits": True, "seed_continuity": True,
                       "state_freshness": True, "source_crc_attestation": True, "crc_computed_here": False},
        "limitations": ["プレビュー専用。DDS・モーター・制御権切り替えへの送信機能はありません。",
                        "衝突・バランス・加速度・トルク・PDゲイン・非常停止経路は検証していません。",
                        "profileと状態の検証フラグは入力側の記録です。実機の確認済み構成から作成してください。",
                        "公式低レベル例の2ms制御とは異なり、この100Hzプレビューは直接送信できません。"],
        "sources": list(OFFICIAL_SOURCES), "frames": frames,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSONの項目が重複しています: {key}")
        result[key] = value
    return result


def load_json(path: str | Path) -> dict:
    """重複キー・巨大入力を拒否してJSONファイルを読む。"""
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValidationError("JSONファイルが大きすぎます")
    return _object(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object), str(path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="関節軌道の検査と100Hzプレビュー（実機送信機能なし）")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", help="プレビューJSONの保存先。省略すると標準出力")
    parser.add_argument("--now", type=float, help="オフライン記録を検査する際のUnix基準時刻")
    args = parser.parse_args(argv)
    try:
        preview = validate_and_preview(load_json(args.profile), load_json(args.state),
                                       load_json(args.trajectory), now=args.now)
        text = json.dumps(preview, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "executable": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
