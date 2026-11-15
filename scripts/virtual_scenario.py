#!/usr/bin/env python3
"""研修台本のプレビューと、Loopback上の仮想R1だけを使うリハーサル。

既定は標準ライブラリだけで順序を表示する。--run-virtual の場合も接続先は
Loopback Pseudo-Interface 1 に固定し、実機用interface引数は提供しない。
仮想サービスは別途起動する。このファイルは起立・LowCmd・Strataを呼ばない。
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOPBACK_INTERFACE = "Loopback Pseudo-Interface 1"
MAX_FILE_BYTES = 128 * 1024
ARM_ACTIONS = frozenset({"wave", "emphasize", "hand_on_heart", "clap", "point_forward"})
ACTIONS = ARM_ACTIONS | {"none", "move_forward", "stop_move", "hat_tip"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _text(value, label, limit, *, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()) or "\0" in value:
        raise ValueError(f"{label}は{limit}文字以内の文字列にしてください")
    return value


def _number(value, label, lower, upper, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not lower <= value <= upper or not math.isfinite(value) or (positive and value == lower)):
        raise ValueError(f"{label}が範囲外です")
    return float(value)


def validate_scenario(data):
    """全段を検証してから独立したコピーを返す。依存パッケージをimportしない。"""
    top_fields = {"schema_version", "mode", "id", "title", "notes", "steps"}
    if not isinstance(data, dict) or set(data) - top_fields:
        raise ValueError("台本の形式または未対応の項目があります")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("schema_versionは1が必要です")
    if data.get("mode") != "virtual_only":
        raise ValueError("modeはvirtual_onlyだけを受け付けます")
    if not isinstance(data.get("id"), str) or not _ID.fullmatch(data["id"]):
        raise ValueError("台本idは英数字・ハイフン・下線・ピリオドの64文字以内です")
    _text(data.get("title"), "title", 160)
    notes = data.get("notes", "")
    if isinstance(notes, list):
        if len(notes) > 20:
            raise ValueError("notesは20項目以内です")
        for note in notes:
            _text(note, "notes", 2000)
    else:
        _text(notes, "notes", 10000, empty=True)
    steps = data.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
        raise ValueError("stepsは1〜64段です")
    ids, total_say = set(), 0
    fields = {"id", "title", "role", "say", "say_ref", "action", "duration_s", "pause_after_s"}
    for step in steps:
        if not isinstance(step, dict) or set(step) - fields:
            raise ValueError("段の形式または未対応の項目があります")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not _ID.fullmatch(step_id) or step_id in ids:
            raise ValueError("段idが不正または重複しています")
        ids.add(step_id)
        _text(step.get("title"), "段title", 160)
        role = step.get("role")
        if not isinstance(role, str) or role not in {"r1", "learner", "stage"}:
            raise ValueError("roleはr1/learner/stageです")
        if "say" in step and "say_ref" in step:
            raise ValueError("sayとsay_refは同時に指定できません")
        if "say" in step:
            total_say += len(_text(step["say"], "say", 1000, empty=True))
        if "say_ref" in step and (step["say_ref"] != "self_introduction" or role != "r1"):
            raise ValueError("say_refはR1のself_introductionだけを使えます")
        action = step.get("action", "none")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError("未対応の動作です")
        if role != "r1" and action != "none":
            raise ValueError("learner/stageにはロボット動作を指定できません")
        if action == "move_forward":
            _number(step.get("duration_s"), "duration_s", 0, 2, positive=True)
        elif "duration_s" in step:
            raise ValueError("duration_sはmove_forwardだけで指定できます")
        _number(step.get("pause_after_s", 0), "pause_after_s", 0, 5)
    if total_say > 20000:
        raise ValueError("発話の合計は20,000文字以内です")
    return copy.deepcopy(data)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSONの項目が重複しています: {key}")
        result[key] = value
    return result


def load_scenario(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("台本ファイルは128KiB以内です")
    def invalid_constant(value):
        raise ValueError(f"JSONの有限でない値は使えません: {value}")
    return validate_scenario(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                                        parse_constant=invalid_constant))


def select_step_range(data, start_step=None, end_step=None):
    """全台本を検証してから、指定idを両端に含む範囲を元の順序で返す。"""
    scenario = validate_scenario(data)
    step_ids = [step["id"] for step in scenario["steps"]]
    for option, value in (("--start-step", start_step), ("--end-step", end_step)):
        if value is not None and value not in step_ids:
            raise ValueError(f"{option}の段idが見つかりません: {value}")
    first = step_ids.index(start_step) if start_step is not None else 0
    last = step_ids.index(end_step) if end_step is not None else len(step_ids) - 1
    if first > last:
        raise ValueError("--start-stepは--end-stepと同じ段か、それより前の段を指定してください")
    selection = {"start_step": step_ids[first], "end_step": step_ids[last],
                 "source_start_index": first + 1, "source_end_index": last + 1,
                 "source_step_count": len(step_ids), "selected_step_count": last - first + 1,
                 "explicit": start_step is not None or end_step is not None}
    scenario["steps"] = scenario["steps"][first:last + 1]
    return scenario, selection


def speech_for(step):
    if step.get("say_ref"):
        # identity.pyは標準ライブラリだけで構成され、DDS/TTSを初期化しない。
        sys.path.insert(0, str(REPO_ROOT / "r1_chat"))
        from identity import SELF_INTRODUCTION
        return SELF_INTRODUCTION
    return step.get("say", "")


def preview_scenario(data, emit=print):
    scenario = validate_scenario(data)
    emit(f"仮想専用プレビュー: {scenario['title']}（DDS・音声・実機への送信なし）")
    notes = scenario.get("notes", [])
    for note in notes if isinstance(notes, list) else [notes]:
        if note:
            emit(f"注記: {note}")
    for index, step in enumerate(scenario["steps"], 1):
        action, say = step.get("action", "none"), speech_for(step)
        emit(f"{index:02d} [{step['role']}] {step['title']} ({step['id']})")
        if say:
            emit(f"   発話: {say}")
        if action == "hat_tip":
            emit("   帽子の身ぶり: 未実装の演出予定。関節指令は送らない")
        elif action == "move_forward":
            emit(f"   同時: R1発話 + 前進指令0.3m/s・{step['duration_s']:g}秒 → 自動ゼロ速度 → 終了を待つ")
        elif action == "point_forward":
            emit("   同時: R1発話 + 決めた配置の前方を示す → 終了・保持姿勢の解除を待つ（対象位置の自動認識なし）")
        elif action in ARM_ACTIONS:
            emit(f"   同時: R1発話 + {action} → 終了・保持姿勢の解除を待つ")
        elif action == "stop_move":
            emit("   ゼロ速度指令（仮想専用。実機での停止の証明ではない）")
        if step.get("pause_after_s", 0):
            emit(f"   次の段まで {step['pause_after_s']:g}秒")
    emit("各段の発話・動作終了後に次へ進む。仮想結果は実機の動作確認を意味しない。")


class VirtualBackend:
    """新しいCLIプロセスからLoopbackにだけ接続する実装。"""
    def __init__(self, *, pc_audio=False):
        # SDKのsingletonが以前のEthernet接続を再利用する余地をなくす。
        if "unitree_sdk2py.core.channel" in sys.modules:
            raise RuntimeError("新しいCLIプロセスで実行してください。既存のDDS接続は再利用しません")
        sys.path.insert(0, str(REPO_ROOT / "r1_chat"))
        sys.path.insert(0, str(REPO_ROOT / "r1_motion"))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        Path("/tmp").mkdir(exist_ok=True)
        ChannelFactoryInitialize(0, LOOPBACK_INTERFACE)
        import r1_chat as rc
        from body_control import BodyController
        self.gesturer = rc.Gesturer(dry=False, hat_fallback=False)
        self.body = BodyController(self.gesturer.loco, dry=False, forward_speed=0.3,
                                   motion_lock=self.gesturer.lock, max_duration=2.0,
                                   velocity_127_compatible=True, require_feedback=False)
        self.tts = self.speaker = None
        self.pc_audio = pc_audio
        self.require_running()
        if pc_audio:
            self.tts, self.speaker = rc.SapiTTS("Microsoft Ichiro", 0), rc.LocalSpeaker()

    def require_running(self):
        fsm = self.gesturer.loco.get_fsm()
        if fsm != 811:
            raise RuntimeError(f"仮想R1がRunning(811)ではありません: {fsm}")
        return fsm

    def prepare_speech(self, say):
        return self.tts.synth(say) if self.tts and say else None

    def speak(self, prepared, cancel):
        if prepared is not None:
            self.speaker.play(prepared, should_continue=lambda: not cancel.is_set())

    def start_action(self, action, duration=None):
        if action in ARM_ACTIONS:
            status = self.gesturer.start(action)
            if not status.startswith("sent"):
                raise RuntimeError(f"仮想の腕動作を開始できません: {status}")
            return {"status": status, "action": action}
        if action == "move_forward":
            ok, message = self.body.move("forward", duration)
            if not ok:
                raise RuntimeError(message)
            return {"accepted": ok, "message": message, "velocity": [0.3, 0.0, 0.0]}
        if action == "stop_move":
            ok, message = self.body.stop()
            state = self.body.state()
            if not ok:
                raise RuntimeError(message)
            return {"accepted": ok, "message": message, "body": state}
        raise ValueError("この動作はバックエンドへ送信できません")

    def finish_action(self, action):
        if action in ARM_ACTIONS:
            code = self.gesturer.finish()
            result = dict(self.gesturer.result)
            result["finish_code"] = code
            if code != 0 or self.gesturer.thread is not None:
                raise RuntimeError(f"仮想の腕動作を終了できません: {result}")
            # Arm Action(816)の解除を確認してから歩行などの次段へ進む。
            deadline = time.monotonic() + 3.0
            while self.gesturer.loco.get_fsm() != 811:
                if time.monotonic() >= deadline:
                    raise RuntimeError("腕動作後のRunning復帰を確認できません")
                time.sleep(0.05)
            return result
        if action == "move_forward":
            if not self.body.wait_until_idle(5.0):
                raise RuntimeError("仮想の前進処理が終了しません")
            state = self.body.state()
            if state["error"] or state["stop_required"] or state["status"] not in {"finished", "finished_unconfirmed"}:
                raise RuntimeError(f"仮想前進の処理失敗: {state}")
            return state
        return self.body.state()

    def abort(self):
        """開始済みの動作だけ後始末する。起立やモード変更は行わない。"""
        result = {}
        if self.speaker:
            import sounddevice as sd
            sd.stop()
        state = self.body.state()
        if state["busy"] or state["stop_required"]:
            result["body_stop"] = self.body.stop()
            result["body_idle"] = self.body.wait_until_idle(4.0)
            result["body"] = self.body.state()
        if self.gesturer.thread is not None:
            result["arm_finish_code"] = self.gesturer.finish()
            result["arm"] = dict(self.gesturer.result)
        return result

    def close(self):
        if self.tts and self.tts.proc:
            proc = self.tts.proc
            if proc.poll() is None:
                proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=3.0)


def run_scenario(data, backend, *, emit=print, clock=time.monotonic, sleep=time.sleep, cancel=None,
                 before_step=None):
    """注入可能なbackendで、発話と動作を重ねつつ段同士は直列に実行する。"""
    scenario = validate_scenario(data)
    cancel = cancel if cancel is not None else threading.Event()
    started = clock()
    report = {"schema_version": 1, "mode": "virtual_only", "physical_result": "not_tested",
              "scenario_id": scenario["id"], "interface": LOOPBACK_INTERFACE,
              "notes": scenario.get("notes", ""), "manual_cues": before_step is not None,
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "steps": []}
    try:
        for step in scenario["steps"]:
            if cancel.is_set():
                raise KeyboardInterrupt
            action, say = step.get("action", "none"), speech_for(step)
            item = {"id": step["id"], "title": step["title"], "role": step["role"],
                    "action": action, "say": say, "started_s": round(clock() - started, 4), "events": []}
            report["steps"].append(item)
            emit(f"[{step['id']}] {step['title']}: {say}")
            # 人の台詞や舞台の準備を終えてから進める。待機中は合成・発話・動作を始めない。
            if before_step is not None:
                item["events"].append({"event": "cue_wait", "t_s": round(clock() - started, 4)})
                before_step(copy.deepcopy(step))
                if cancel.is_set():
                    raise KeyboardInterrupt
                item["events"].append({"event": "cue_accepted", "t_s": round(clock() - started, 4)})
                item["started_s"] = round(clock() - started, 4)
            # 先に合成し、音声合成の待ち時間に動作が終わらないようにする。
            prepared = backend.prepare_speech(say) if step["role"] == "r1" and say else None
            if cancel.is_set():
                raise KeyboardInterrupt
            if step["role"] == "r1":
                item["fsm_before"] = backend.require_running()
            active_action = action not in {"none", "hat_tip"}
            if action == "hat_tip":
                item.update(status="placeholder", implemented=False,
                            note="帽子の身ぶりは演出予定。DDS動作・関節指令は送信しない")
                emit("  帽子の身ぶり: 未実装の演出予定（送信なし）")
            if active_action:
                item["events"].append({"event": "action_start", "t_s": round(clock() - started, 4)})
                item["start_result"] = backend.start_action(action, step.get("duration_s"))
            if step["role"] == "r1" and say:
                item["events"].append({"event": "speech_start", "t_s": round(clock() - started, 4)})
                backend.speak(prepared, cancel)
                item["events"].append({"event": "speech_finish", "t_s": round(clock() - started, 4)})
            if cancel.is_set():
                raise KeyboardInterrupt
            if active_action:
                item["result"] = backend.finish_action(action)
                item["events"].append({"event": "action_finish", "t_s": round(clock() - started, 4)})
            item.setdefault("status", "done")
            item["finished_s"] = round(clock() - started, 4)
            remaining = float(step.get("pause_after_s", 0))
            while remaining > 0:
                if cancel.is_set():
                    raise KeyboardInterrupt
                interval = min(0.05, remaining)
                sleep(interval)
                remaining -= interval
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        cancel.set()
        report["status"] = "cancelled" if isinstance(error, (KeyboardInterrupt, EOFError)) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if report["steps"]:
            report["steps"][-1]["status"] = report["status"]
        try:
            report["cleanup"] = backend.abort()
        except Exception as cleanup_error:
            report["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
    finally:
        try:
            backend.close()
        except Exception as close_error:
            report["close_error"] = f"{type(close_error).__name__}: {close_error}"
            if report["status"] == "completed":
                report["status"] = "failed"
        report["elapsed_s"] = round(clock() - started, 4)
    return report


def wait_for_step(step):
    """現在の場面を表示した後、人の合図を待つ。入力が終われば中止する。"""
    input("  この段を実行するには Enter、中止するには Ctrl+C > ")


def main(argv=None):
    parser = argparse.ArgumentParser(description="仮想R1専用の研修シナリオ（既定は送信なしのプレビュー）")
    parser.add_argument("scenario", nargs="?", default="scenarios/training_rehearsal.json")
    parser.add_argument("--run-virtual", action="store_true", help=f"{LOOPBACK_INTERFACE}に限って実行する")
    parser.add_argument("--pc-audio", action="store_true", help="仮想実行時、R1役の台詞をPCで読み上げる")
    parser.add_argument("--interactive", action="store_true", help="仮想実行時、各段を始める前にEnterの合図を待つ")
    parser.add_argument("--start-step", metavar="ID", help="この段idから開始する（その段を含む）")
    parser.add_argument("--end-step", metavar="ID", help="この段idで終了する（その段を含む）")
    args = parser.parse_args(argv)
    if args.pc_audio and not args.run_virtual:
        parser.error("--pc-audioには--run-virtualが必要です")
    if args.interactive and not args.run_virtual:
        parser.error("--interactiveには--run-virtualが必要です")
    try:
        scenario = load_scenario(args.scenario)
        scenario, selected_range = select_step_range(scenario, args.start_step, args.end_step)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    if selected_range["explicit"]:
        print(f"選択範囲: {selected_range['start_step']} から {selected_range['end_step']}（{selected_range['selected_step_count']}段）", flush=True)
    if not args.run_virtual:
        preview_scenario(scenario)
        return 0
    print(f"仮想専用: {LOOPBACK_INTERFACE}。PC音声={'あり' if args.pc_audio else 'なし'}。実機のテスト結果にはなりません。", flush=True)
    try:
        backend = VirtualBackend(pc_audio=args.pc_audio)
    except Exception as error:
        report = {"schema_version": 1, "mode": "virtual_only", "physical_result": "not_tested",
                  "scenario_id": scenario["id"], "interface": LOOPBACK_INTERFACE,
                  "notes": scenario.get("notes", ""),
                  "status": "initialization_failed", "error": f"{type(error).__name__}: {error}", "steps": []}
    else:
        report = run_scenario(scenario, backend, before_step=wait_for_step if args.interactive else None)
    report["pc_audio"] = args.pc_audio
    report["interactive"] = args.interactive
    report["selected_range"] = selected_range
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"virtual_scenario_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"結果: {report['status']} / {path}", flush=True)
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report["status"] == "completed" else 130 if report["status"] == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
