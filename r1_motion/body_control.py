#!/usr/bin/env python3
"""R1の速度指令を短時間だけ送る。既定は実機へ送らないプレビュー。

高レベルsport API7105を使い、脚のLowCmdやFSM変更は行わない。
実行にはRunning(811)を要求する。速度×時間は移動距離の保証ではなく、
障害物・人・床面を検出する機能もない。stop()は非常停止の代わりではない。
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
import re
import threading
import time
import unicodedata


def requested_body_actions(text: str) -> set[str]:
    """説明文や教材中の語句では動かず、利用者の短い直接依頼だけ許可する。"""
    if not isinstance(text, str):
        return set()
    text = unicodedata.normalize("NFKC", text).lower()
    allowed = set()
    for clause in re.split(r"[。！？!?\n]", text):
        if any(mark in clause for mark in '「」『』"“”'):
            continue
        clause = re.sub(r"\s+", "", clause)
        clause = re.sub(r"^(?:r1|アールワン)(?:さん)?[、,:：]*", "", clause)
        if re.search(r"方法|仕組み|原理|意味|説明|教えて|コード|プログラム|台本|指示書|という|って何|できますか|できる|英訳|翻訳|引用|添削|(?:と|って)(?:言|話|読|頼)|覚えて|したら|なら|とき|場合|あとで|後で|秒後|分後|てもいい|でもいい|いいですか", clause):
            continue
        if re.search(r"止まって|停止して|ストップ|歩行停止|歩くのをやめて|移動をやめて", clause):
            allowed.add("stop_move")
            continue
        if re.search(r"ないで|しない|するな|しなくて|禁止|(?:歩|動|進)かない", clause):
            continue
        if re.search(r"前(?:へ|に)(?:少し|ちょっと|一歩|ゆっくり)?(?:進んで|歩いて|移動して|動いて)|前進して", clause) or re.fullmatch(r"(?:少し|ちょっと|ゆっくり)?歩いて(?:みて)?(?:ください|くれ|お願い)?[、,]*", clause):
            allowed.add("move_forward")
        if re.search(r"後ろ(?:へ|に)(?:少し|ちょっと|一歩|ゆっくり)?(?:下がって|進んで|歩いて|移動して|動いて)|後退して", clause):
            allowed.add("move_backward")
        for word, action in (("左", "turn_left"), ("右", "turn_right")):
            if re.search(word + r"(?:へ|に|を)(?:少し|ちょっと|ゆっくり)?(?:向いて|回って|旋回して)", clause):
                allowed.add(action)
            elif re.search(word + r"(?:へ|に)(?:少し|ちょっと|一歩|ゆっくり)?(?:進んで|歩いて|移動して|動いて)", clause):
                allowed.add("move_left" if word == "左" else "move_right")
    return allowed


@dataclass
class _Move:
    direction: str
    velocity: tuple[float, float, float]
    duration: float
    started_at: float
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    sent: bool = False
    stopped: bool = False
    status: str = "queued"
    code: int | None = None
    stop_code: int | None = None
    error: str = ""
    joint_motion_observed: bool = False
    joint_motion_span: float = 0.0
    max_leg_dq: float = 0.0
    q_min: list[float] | None = None
    q_max: list[float] | None = None
    command_sent_at: float | None = None
    hold: bool = False
    token: str | None = None
    lease_deadline: float = 0.0
    refresh_count: int = 0


class BodyController:
    """手動の短い歩行。motion_lockは腕・関節制御と共用できるLock/RLock。

    move()は非同期で受理結果を返す。実行結果はstate()で確認する。
    共有ロックはワーカースレッドで取得・解放し、占有中なら待たずに見送る。
    injected waitのシグネチャはwait(cancel_event, seconds)。
    """

    ALIASES = {"turnleft": "turn_left", "turnright": "turn_right"}
    HOLD_LEASE = 0.75
    HOLD_REFRESH = 0.3
    HOLD_DURATION = 2.0

    def __init__(self, loco=None, *, dry: bool = True, linear_speed: float = 0.1,
                 forward_speed: float | None = None,
                 yaw_speed: float = 0.2, max_duration: float = 2.0, motion_lock=None,
                 monotonic=time.monotonic, wait=None, on_event=None,
                 velocity_127_compatible: bool = False, feedback=None, require_feedback: bool = False):
        forward_speed = linear_speed if forward_speed is None else forward_speed
        for name, value, maximum in (("linear_speed", linear_speed, 0.3),
                                     ("forward_speed", forward_speed, 0.3),
                                     ("yaw_speed", yaw_speed, 0.5),
                                     ("max_duration", max_duration, 2.0)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name}は0より大きく{maximum}以下にしてください")
        if not dry and loco is None:
            raise ValueError("実機制御にはR1用locoクライアントが必要です")
        if not isinstance(velocity_127_compatible, bool):
            raise ValueError("velocity_127_compatibleは真偽値で指定してください")
        if require_feedback and not callable(feedback):
            raise ValueError("実機の関節フィードバックが必要です")
        self.loco, self.dry = loco, bool(dry)
        self.linear_speed, self.yaw_speed, self.max_duration = linear_speed, yaw_speed, max_duration
        self.forward_speed = forward_speed
        self.motion_lock = motion_lock if motion_lock is not None else threading.Lock()
        self._clock, self._wait = monotonic, wait or (lambda event, seconds: event.wait(seconds))
        self.on_event = on_event
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._active: _Move | None = None
        self._last: _Move | None = None
        self._stop_failed = False
        self._stop_inflight = 0
        self.velocity_127_compatible = velocity_127_compatible
        self._last_stop_code = None
        self.feedback, self.require_feedback = feedback, bool(require_feedback)
        self._feedback_status = "waiting" if self.require_feedback else "not_required"
        self._joint_stationary_observed = False
        self._stop_sent_at = -math.inf
        self._stop_verified_at = -math.inf
        self._quiet_wait_event = threading.Event()
        self._stationary_q = None
        self._retired_hold_tokens = set()
        self._retired_hold_order = deque()

    @staticmethod
    def _valid_hold_token(token):
        return isinstance(token, str) and re.fullmatch(r"[A-Za-z0-9_-]{16,128}", token) is not None

    def _retire_hold(self, token):
        """_lockを保持して呼ぶ。遅延した開始・更新が解除済みの押下を復活させない。"""
        if token and token not in self._retired_hold_tokens:
            if len(self._retired_hold_order) >= 256:
                self._retired_hold_tokens.remove(self._retired_hold_order.popleft())
            self._retired_hold_order.append(token)
            self._retired_hold_tokens.add(token)

    def _hold_current(self, session):
        with self._lock:
            if session.cancel.is_set() or self._active is not session:
                return False
            if self._clock() >= session.lease_deadline:
                session.status = "lease_expired"
                session.cancel.set()
                self._retire_hold(session.token)
                return False
            return True

    def start_hold(self, direction: str, token: str, *, linear_speed=0.9, yaw_speed=1.8):
        """明示押下の間だけ更新する歩行。会話の単発move()の値は変更しない。"""
        direction = self.ALIASES.get(direction, direction) if isinstance(direction, str) else ""
        if not self._valid_hold_token(token):
            return False, "押下の識別子が不正です"
        for name, value, maximum in (("linear_speed", linear_speed, 0.9), ("yaw_speed", yaw_speed, 1.8)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
                return False, f"{name}は0より大きく{maximum:g}以下にしてください"
        vectors = {"forward": (linear_speed, 0.0, 0.0), "backward": (-linear_speed, 0.0, 0.0),
                   "left": (0.0, linear_speed, 0.0), "right": (0.0, -linear_speed, 0.0),
                   "turn_left": (0.0, 0.0, yaw_speed), "turn_right": (0.0, 0.0, -yaw_speed)}
        if direction not in vectors:
            return False, "不明な移動方向です"
        _, feedback_error = self._read_feedback()
        if feedback_error:
            return False, f"関節状態を確認できないため移動しません: {feedback_error}"
        with self._lock:
            if token in self._retired_hold_tokens:
                return False, "この押下は解除済みです。押し直してください"
            if self._stop_inflight:
                return False, "停止処理中です"
            if self._stop_failed:
                return False, "前の停止指令が確認できていません。歩行停止を再試行してください"
            if self._active is not None:
                return False, "前の動作が終了していません"
            now = self._clock()
            session = _Move(direction, vectors[direction], min(self.HOLD_DURATION, self.max_duration), now,
                            hold=True, token=token, lease_deadline=now + self.HOLD_LEASE)
            self._joint_stationary_observed = False
            self._stop_verified_at = -math.inf
            self._active = self._last = session
            session.thread = threading.Thread(target=self._run, args=(session,), daemon=False, name="r1-body-hold")
            try:
                session.thread.start()
            except Exception:
                self._active = None
                self._retire_hold(token)
                raise
        return True, "押下歩行を受け付けました。離すか更新が途切れると停止指令を送ります"

    def heartbeat_hold(self, token: str):
        if not self._valid_hold_token(token):
            return False, "押下の識別子が不正です"
        with self._lock:
            session = self._active
            if session is None or not session.hold or session.token != token:
                return False, "更新対象の押下歩行はありません"
            if not self._hold_current(session):
                return False, "押下歩行は終了しています。押し直してください"
            session.lease_deadline = self._clock() + self.HOLD_LEASE
        return True, "ok"

    def release_hold(self, token: str):
        if not self._valid_hold_token(token):
            return False, "押下の識別子が不正です"
        with self._lock:
            self._retire_hold(token)
            session = self._active
            if session is None or not session.hold or session.token != token:
                return True, "この押下は解除済みか、実行されていません"
            session.cancel.set()
            self._stop_inflight += 1
        try:
            with self._io_lock:
                # 古い解除要求が、新しい押下の速度をゼロにしない。
                with self._lock:
                    if self._active is not session:
                        return True, "押下歩行は終了しています"
                code, confirmed = self._finish_stop(session, force=True)
            if not self._accepted_velocity_code(code):
                return False, f"停止指令に失敗しました: code={code}"
            if not confirmed:
                return False, "ゼロ速度を送りましたが脚の静止は未確認です。目視で確認してください"
            return True, "ゼロ速度を送信しました。実際の停止は目視でも確認してください"
        except Exception as error:
            with self._lock:
                self._stop_failed = True
                session.status, session.error = "stop_error", f"停止指令: {type(error).__name__}: {error}"
            return False, session.error
        finally:
            with self._lock:
                self._stop_inflight -= 1

    def _read_feedback(self, *, update_status=True):
        """証跡のCRC・鮮度・機体プロファイル・モーター異常を毎回確認する。"""
        if not self.require_feedback:
            return None, None
        data, reason = None, None
        try:
            data = self.feedback()
            if not isinstance(data, dict):
                reason = "unavailable"
            elif data.get("crc_valid") is not True or data.get("error"):
                reason = "crc_or_data_invalid"
            elif not isinstance(data.get("t"), (int, float)) or isinstance(data.get("t"), bool) or not math.isfinite(data["t"]) or not 0 <= self._clock() - data["t"] < 0.5:
                reason = "stale"
            elif data.get("mode_machine") != 1:
                reason = "profile_mismatch"
            elif data.get("mode_pr") != 0:
                reason = "mode_pr_mismatch"
            elif not isinstance(data.get("motor_fault"), (list, tuple)) or data["motor_fault"]:
                reason = "motor_fault"
            else:
                for name in ("qlegs", "dqlegs"):
                    values = data.get(name)
                    if (not isinstance(values, (list, tuple)) or len(values) != 12 or
                            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)):
                        reason = "invalid_joints"
                        break
        except Exception:
            reason = "observer_error"
        if update_status:
            with self._lock:
                self._feedback_status = reason or "fresh"
        return data, reason

    def _observe_motion(self, session, data):
        if data is None:
            return
        q = list(data["qlegs"])
        with self._lock:
            if session.q_min is None:
                session.q_min, session.q_max = q[:], q[:]
            else:
                session.q_min = [min(a, b) for a, b in zip(session.q_min, q)]
                session.q_max = [max(a, b) for a, b in zip(session.q_max, q)]
            session.joint_motion_span = max(b - a for a, b in zip(session.q_min, session.q_max))
            if session.command_sent_at is not None and data["t"] > session.command_sent_at:
                session.max_leg_dq = max(session.max_leg_dq, max(abs(x) for x in data["dqlegs"]))
                session.joint_motion_observed = session.joint_motion_span >= 0.01 or session.max_leg_dq >= 0.12

    def _verify_stationary(self):
        """新しい停止後の観測だけで脚の静止を確認する。位置や安全は保証しない。"""
        if not self.require_feedback:
            return True
        data, reason = self._read_feedback()
        if self._joint_stationary_observed and not reason and self._clock() - self._stop_verified_at < 0.1:
            return True
        with self._lock:
            self._joint_stationary_observed = False
        deadline = self._clock() + 2.0
        window, last_t = [], -math.inf
        while self._clock() <= deadline:
            data, reason = self._read_feedback()
            if reason:
                window = []
            elif data["t"] > self._stop_sent_at and data["t"] > last_t:
                last_t = data["t"]
                if max(abs(x) for x in data["dqlegs"]) > 0.12:
                    window = []
                else:
                    window.append((data["t"], list(data["qlegs"])))
                    span = max(max(row[1][i] for row in window) - min(row[1][i] for row in window) for i in range(12))
                    if span > 0.01:
                        window = [window[-1]]
                    elif window[-1][0] - window[0][0] >= 0.5:
                        with self._lock:
                            self._joint_stationary_observed = True
                            self._stop_verified_at = self._clock()
                            self._stationary_q = list(data["qlegs"])
                            self._feedback_status = "joint_stationary_observed"
                        return True
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            # 停止操作でcancelがセット済みでも、待機を省略して空回りしない。
            self._wait(self._quiet_wait_event, min(0.05, remaining))
        with self._lock:
            self._feedback_status = "stop_not_observed:" + (reason or "joints_not_quiet")
        return False

    def _finish_stop(self, session, *, force=False):
        """呼び出し側が_io_lockを保持する。まずゼロ指令、その後に最長2秒確認する。"""
        code = self._zero(session, force=force)
        confirmed = self._accepted_velocity_code(code) and self._verify_stationary()
        with self._lock:
            self._stop_failed = not confirmed
            if not confirmed and session is not None:
                session.status = "stop_unconfirmed"
                session.error = f"停止を確認できません: code={code}, feedback={self._feedback_status}"
        return code, confirmed

    def _accepted_velocity_code(self, code):
        # 127は一部R1ファームウェアの実機報告にある値。物理成功の証拠にはしない。
        return code == 0 or (self.velocity_127_compatible and code == 127)

    def _event(self, kind: str, **data):
        if self.on_event:
            try:
                self.on_event({"event": kind, "dry": self.dry, **data})
            except Exception:
                # ログ出力の失敗で、停止処理を飛ばさない。
                pass

    def _velocity(self, velocity, duration):
        if self.dry:
            self._event("velocity", velocity=list(velocity), duration=duration)
            return 0
        code, _ = self.loco._Call(7105, json.dumps({"velocity": list(velocity), "duration": duration}))
        self._event("velocity", velocity=list(velocity), duration=duration, code=code)
        return code

    def _zero(self, session: _Move | None, *, force=False):
        """_io_lockを取得してから呼ぶ。停止RPC成功済みなら重ねて送らない。"""
        if session is not None and session.stopped and not force:
            return session.stop_code
        code = self._velocity((0.0, 0.0, 0.0), 1.0)
        with self._lock:
            self._last_stop_code = code
            self._stop_sent_at = self._clock()
            self._joint_stationary_observed = False
        if session is not None:
            with self._lock:
                session.stop_code = code
                session.stopped = self._accepted_velocity_code(code)
        return code

    def move(self, direction: str, duration: float = 1.0) -> tuple[bool, str]:
        if direction == "stop":
            return self.stop()
        direction = self.ALIASES.get(direction, direction) if isinstance(direction, str) else ""
        vectors = {"forward": (self.forward_speed, 0.0, 0.0),
                   "backward": (-self.linear_speed, 0.0, 0.0),
                   "left": (0.0, self.linear_speed, 0.0),
                   "right": (0.0, -self.linear_speed, 0.0),
                   "turn_left": (0.0, 0.0, self.yaw_speed),
                   "turn_right": (0.0, 0.0, -self.yaw_speed)}
        if direction not in vectors:
            return False, "不明な移動方向です"
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= self.max_duration:
            return False, f"移動時間は0より大きく{self.max_duration:g}秒以下にしてください"
        _, feedback_error = self._read_feedback()
        if feedback_error:
            return False, f"関節状態を確認できないため移動しません: {feedback_error}"
        with self._lock:
            if self._stop_inflight:
                return False, "停止処理中です"
            if self._stop_failed:
                return False, "前の停止指令が確認できていません。歩行停止を再試行してください"
            if self._active is not None:
                return False, "前の移動が終了していません"
            session = _Move(direction, vectors[direction], float(duration), self._clock())
            self._joint_stationary_observed = False
            self._stop_verified_at = -math.inf
            self._active = self._last = session
            session.thread = threading.Thread(target=self._run, args=(session,), daemon=False,
                                              name="r1-body-move")
            try:
                session.thread.start()
            except Exception:
                self._active = None
                raise
        return True, "移動要求を受け付けました。実行結果は状態を確認してください"

    def _run(self, session: _Move):
        acquired = self.motion_lock.acquire(blocking=False)
        if not acquired:
            with self._lock:
                session.status, session.error = "busy", "別の腕・関節動作が実行中です"
                if session.hold:
                    self._retire_hold(session.token)
                self._active = None
            self._event("skipped", reason=session.error)
            return
        try:
            with self._io_lock:
                if session.cancel.is_set():
                    with self._lock:
                        session.status = "cancelled"
                    return
                if session.hold and not self._hold_current(session):
                    return
                data, feedback_error = self._read_feedback()
                if feedback_error:
                    with self._lock:
                        session.status, session.error = "feedback_invalid", f"関節状態が不正です: {feedback_error}"
                    return
                fsm = 811 if self.dry else self.loco.get_fsm()
                if fsm != 811:
                    with self._lock:
                        session.status, session.error = "rejected_fsm", f"Running(811)ではありません: {fsm}"
                    return
                # 停止要求がFSM読み出し中に届いていたら、速度を送らない。
                if session.cancel.is_set():
                    with self._lock:
                        session.status = "cancelled"
                    return
                if session.hold and not self._hold_current(session):
                    return
                data, feedback_error = self._read_feedback()
                if feedback_error:
                    with self._lock:
                        session.status, session.error = "feedback_invalid", f"送信直前の関節状態が不正です: {feedback_error}"
                    return
                self._observe_motion(session, data)
                if session.hold and not self._hold_current(session):
                    return
                command_time = self._clock()
                session.command_sent_at = command_time
                code = self._velocity(session.velocity, session.duration)
                accepted = self._accepted_velocity_code(code)
                with self._lock:
                    session.code, session.sent = code, accepted
                    session.refresh_count += 1
                    session.status = "moving" if accepted else "rpc_error"
                    if not accepted:
                        session.error = f"速度指令に失敗しました: code={code}"
                if not accepted:
                    return
            if session.hold:
                self._run_hold_updates(session)
                return
            deadline = command_time + session.duration
            while not session.cancel.is_set():
                data, feedback_error = self._read_feedback()
                if feedback_error:
                    with self._lock:
                        session.status, session.error = "feedback_lost", f"動作中の関節状態が不正です: {feedback_error}"
                    session.cancel.set()
                    break
                self._observe_motion(session, data)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                self._wait(session.cancel, min(0.05, remaining))
            with self._lock:
                if session.status == "moving":
                    session.status = "cancelled" if session.cancel.is_set() else "finished"
        except Exception as error:
            with self._lock:
                session.status, session.error = "error", f"{type(error).__name__}: {error}"
        finally:
            try:
                with self._io_lock:
                    code, confirmed = self._finish_stop(session)
                    if not self._accepted_velocity_code(code):
                        with self._lock:
                            self._stop_failed = True
                            session.status = "stop_error"
                            session.error = f"停止指令に失敗しました: code={code}"
                    elif confirmed and 127 in (session.code, session.stop_code):
                        with self._lock:
                            if session.status in ("finished", "cancelled"):
                                session.status += "_unconfirmed"
            except Exception as error:
                with self._lock:
                    self._stop_failed = True
                    session.status, session.error = "stop_error", f"停止指令: {type(error).__name__}: {error}"
            finally:
                self._event("completed", **self.state())
                self.motion_lock.release()
                with self._lock:
                    if self._active is session:
                        if session.hold:
                            self._retire_hold(session.token)
                        self._active = None

    def _run_hold_updates(self, session):
        next_refresh = self._clock() + self.HOLD_REFRESH
        while self._hold_current(session):
            data, feedback_error = self._read_feedback()
            if feedback_error:
                with self._lock:
                    session.status, session.error = "feedback_lost", f"動作中の関節状態が不正です: {feedback_error}"
                    session.cancel.set()
                break
            self._observe_motion(session, data)
            now = self._clock()
            if now >= next_refresh:
                with self._io_lock:
                    if not self._hold_current(session):
                        break
                    fsm = 811 if self.dry else self.loco.get_fsm()
                    if fsm != 811:
                        with self._lock:
                            session.status, session.error = "rejected_fsm", f"Running(811)ではありません: {fsm}"
                            session.cancel.set()
                        break
                    # 読取RPCの間に離された場合や期限切れは速度を再送しない。
                    if not self._hold_current(session):
                        break
                    data, feedback_error = self._read_feedback()
                    if feedback_error:
                        with self._lock:
                            session.status, session.error = "feedback_lost", f"送信直前の関節状態が不正です: {feedback_error}"
                            session.cancel.set()
                        break
                    if not self._hold_current(session):
                        break
                    self._observe_motion(session, data)
                    code = self._velocity(session.velocity, session.duration)
                    with self._lock:
                        session.code = code
                        session.refresh_count += 1
                        if not self._accepted_velocity_code(code):
                            session.status, session.error = "rpc_error", f"速度更新に失敗しました: code={code}"
                            session.cancel.set()
                            break
                next_refresh = self._clock() + self.HOLD_REFRESH
            with self._lock:
                remaining = session.lease_deadline - self._clock()
            if remaining > 0:
                self._wait(session.cancel, min(0.05, remaining, max(0.0, next_refresh - self._clock())))
        with self._lock:
            if session.status == "moving":
                session.status = "cancelled"

    def stop(self) -> tuple[bool, str]:
        """モード確認なしにゼロ速度を送る。コントローラーの非常停止とは異なる。"""
        with self._lock:
            self._stop_inflight += 1
            session = self._active or (self._last if self._stop_failed else None)
            if session:
                session.cancel.set()
                if session.hold:
                    self._retire_hold(session.token)
        try:
            with self._io_lock:
                code, confirmed = self._finish_stop(session, force=True)
            with self._lock:
                if not self._stop_failed and session is not None and session.status in ("stop_error", "stop_unconfirmed"):
                    session.status, session.error = ("stopped_unconfirmed" if code == 127 else "stopped"), ""
            if not self._accepted_velocity_code(code):
                return False, f"停止指令に失敗しました: code={code}"
            if not confirmed:
                return False, "ゼロ速度を送りましたが、脚の静止を確認できません。オペレーターが状態を確認してください"
            if code == 127:
                return True, "ゼロ速度を送信しました。応答127：実際の停止を確認してください"
            return True, "ゼロ速度の停止指令を送りました" if not self.dry else "停止指令のプレビューです"
        except Exception as error:
            with self._lock:
                self._stop_failed = True
            return False, f"停止指令: {type(error).__name__}: {error}"
        finally:
            with self._lock:
                self._stop_inflight -= 1

    def state(self) -> dict:
        data, current_error = self._read_feedback(update_status=False)
        with self._lock:
            session = self._active or self._last
            stop_code = session.stop_code if session and session.stop_code is not None else self._last_stop_code
            native_127 = 127 in ((session.code if session else None), stop_code)
            currently_quiet = bool(not current_error and data and max(abs(x) for x in data["dqlegs"]) <= .12 and
                                   self._stationary_q is not None and max(abs(x-y) for x, y in zip(data["qlegs"], self._stationary_q)) <= .01)
            stationary_reported = self._joint_stationary_observed and (currently_quiet or not self.require_feedback)
            motion_unconfirmed = bool(self.require_feedback and session and session.sent and not session.joint_motion_observed)
            physical_confirmation_required = native_127 or (self.require_feedback and not stationary_reported) or motion_unconfirmed
            return {"busy": self._active is not None, "dry": self.dry,
                    "stopping": bool(self._stop_inflight),
                    "hold": bool(session and session.hold),
                    "hold_active": bool(self._active and self._active.hold and not self._active.cancel.is_set()
                                        and self._clock() < self._active.lease_deadline),
                    "hold_lease_remaining": max(0.0, self._active.lease_deadline - self._clock())
                                             if self._active and self._active.hold else 0.0,
                    "hold_refresh_count": session.refresh_count if session and session.hold else 0,
                    "linear_speed": self.linear_speed, "forward_speed": self.forward_speed,
                    "yaw_speed": self.yaw_speed,
                    "stop_required": self._stop_failed,
                    "velocity_127_compatible": self.velocity_127_compatible,
                    "physical_confirmation_required": physical_confirmation_required,
                    "require_feedback": self.require_feedback,
                    "feedback_status": current_error or ("current_joints_moving" if self._joint_stationary_observed and not currently_quiet and self.require_feedback else self._feedback_status),
                    "feedback_valid": bool(self.require_feedback and not current_error),
                    "feedback_t": data.get("t") if isinstance(data, dict) else None,
                    "feedback_age": max(0.0, self._clock()-data["t"]) if isinstance(data, dict) and isinstance(data.get("t"), (int,float)) and math.isfinite(data["t"]) else None,
                    "joint_stationary_observed": stationary_reported,
                    "joint_stationary_observed_at": self._stop_verified_at if math.isfinite(self._stop_verified_at) else None,
                    "joint_motion_observed": session.joint_motion_observed if session else False,
                    "joint_motion_span": session.joint_motion_span if session else 0.0,
                    "max_leg_dq": session.max_leg_dq if session else 0.0,
                    "response_note": "応答127：実際の動作・停止を確認" if native_127 else "脚の静止は未確認" if self.require_feedback and not stationary_reported else "脚の動作は未確認" if motion_unconfirmed else "",
                    "direction": session.direction if session else "",
                    "velocity": list(session.velocity) if session else [0.0, 0.0, 0.0],
                    "duration": session.duration if session else 0.0,
                    "elapsed": round(max(0.0, self._clock() - session.started_at), 3) if session else 0.0,
                    "status": session.status if session else "idle",
                    "code": session.code if session else None,
                    "stop_code": stop_code,
                    "error": session.error if session else ""}

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        with self._lock:
            session = self._active
        if session and session.thread:
            session.thread.join(timeout)
            return not session.thread.is_alive()
        return True


def main():
    p = argparse.ArgumentParser(description="R1の短時間の手動速度指令（既定はプレビュー）")
    p.add_argument("direction", choices=("forward", "backward", "left", "right", "turn_left", "turn_right", "stop"))
    group = p.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true", help="実機へ送らずプレビューする（既定）")
    group.add_argument("-i", "--interface", help="実機へ送る有線LANアダプタ名")
    p.add_argument("--duration", type=float, default=1.0, help="0より大きく2秒以下")
    p.add_argument("--velocity-127-compatible", action="store_true", help="7105の応答127を互換応答として扱う。物理成功は別途確認する")
    args = p.parse_args()
    dry, loco, monitor = not bool(args.interface), None, None
    if not dry:
        if not args.interface.isascii():
            p.error("アダプタ名には英数字を使ってください")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from stand import R1Loco
        ChannelFactoryInitialize(0, args.interface)
        loco = R1Loco()
        loco.SetTimeout(3.0)
        loco.Init()
        time.sleep(1.0)
        from body_feedback import MotorFeedbackMonitor
        monitor = MotorFeedbackMonitor(sample_hz=10).start()
        read_deadline = time.monotonic() + 1.0
        while monitor.snapshot() is None and time.monotonic() < read_deadline:
            time.sleep(0.02)
    controller = BodyController(loco, dry=dry, velocity_127_compatible=args.velocity_127_compatible,
                                feedback=monitor.snapshot if monitor else None, require_feedback=not dry,
                                on_event=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    try:
        ok, message = controller.move(args.direction, args.duration)
        print(message, flush=True)
        if not ok:
            raise SystemExit(1)
        controller.wait_until_idle()
        print(json.dumps(controller.state(), ensure_ascii=False), flush=True)
        if controller.state()["error"]:
            raise SystemExit(1)
    except KeyboardInterrupt:
        print(controller.stop()[1], flush=True)
        controller.wait_until_idle(4.0)
    finally:
        if monitor:
            monitor.close()


if __name__ == "__main__":
    main()
