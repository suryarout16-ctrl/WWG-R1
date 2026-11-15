"""仰向けからの高レベル起立。取り消しは後続指令だけを止め、機体の非常停止はしない。"""
from __future__ import annotations

import math
import threading
import time


class _Cancelled(Exception):
    pass


class _Failure(Exception):
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


class StandController:
    """run()を呼んだスレッドで実行する。FSMだけから仰向けと推測しない。

    clientはget_fsm()/set_fsm(target)を持つ既存のR1用クライアント。
    waitの差替えはwait(cancel_event, seconds)、時計はmonotonic()。
    キャンセル／失敗時に脱力・寝る指令などへ自動で戻さない。
    """

    def __init__(self, client=None, *, dry=False, motion_lock=None, on_event=None,
                 monotonic=time.monotonic, wait=None, poll_interval=0.2):
        if not isinstance(dry, bool):
            raise ValueError("dryは真偽値で指定してください")
        if client is None and not dry:
            raise ValueError("起立にはR1用クライアントが必要です")
        if (isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float))
                or not math.isfinite(poll_interval) or not 0 < poll_interval <= 1):
            raise ValueError("状態読取間隔が不正です")
        self.client, self.dry = client, dry
        self.motion_lock = motion_lock if motion_lock is not None else threading.Lock()
        self.on_event = on_event
        self._clock = monotonic
        self._wait = wait or (lambda event, seconds: event.wait(seconds))
        self._poll = float(poll_interval)
        self._run_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._cancel = threading.Event()
        self._state = self._new_state()

    def _new_state(self):
        return {"busy": False, "status": "idle", "phase": "idle", "fsm": None,
                "error": "", "dry": self.dry, "cancel_requested": False,
                "commands": [], "last_code": None}

    def state(self) -> dict:
        with self._state_lock:
            return {**self._state, "commands": [dict(row) for row in self._state["commands"]]}

    def _update(self, **fields):
        with self._state_lock:
            self._state.update(fields)

    def _event(self, kind, **fields):
        if self.on_event:
            try:
                self.on_event({"event": kind, "dry": self.dry, **fields})
            except Exception:
                pass

    def cancel(self):
        """後続の送信を取り消す。既に始まったファームウェアの起立は停止しない。"""
        with self._state_lock:
            self._cancel.set()
            self._state["cancel_requested"] = True
        # cancel()が戻った後に、確認済みの古い送信を始めない。
        with self._io_lock:
            pass

    def _current(self, should_continue):
        if self._cancel.is_set() or (should_continue is not None and not should_continue()):
            raise _Cancelled()

    def _read(self, should_continue):
        # 状態読取の一時的なSDK例外だけを最大3回まで再試行する。
        # 指令の再送はせず、次の送信には成功した新しい読取を必須とする。
        for attempt in range(1, 4):
            with self._io_lock:
                self._current(should_continue)
                try:
                    fsm = self.client.get_fsm()
                except Exception as error:
                    self._current(should_continue)
                    error_type = type(error).__name__
                else:
                    self._current(should_continue)
                    if isinstance(fsm, bool) or not isinstance(fsm, int):
                        raise _Failure("rejected_fsm", "モードを確認できないため起立指令を送りません")
                    self._update(fsm=fsm)
                    return fsm
            if attempt == 3:
                raise _Failure("read_error", f"モードを読み取れません ({error_type})") from None
            self._event("read_retry", attempt=attempt, error_type=error_type)
            self._pause(self._poll, should_continue)

    def _pause(self, seconds, should_continue):
        deadline = self._clock() + seconds
        while True:
            self._current(should_continue)
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._wait(self._cancel, min(self._poll, remaining))

    def _wait_for(self, target, timeout, allowed, should_continue):
        deadline = self._clock() + timeout
        while True:
            fsm = self._read(should_continue)
            if fsm == target:
                return
            if fsm not in allowed:
                raise _Failure("rejected_fsm", f"予期しないモード{fsm}へ変わったため後続の起立指令を送りません")
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise _Failure("timeout", f"モード{target}への移行を{timeout:g}秒以内に確認できませんでした")
            self._wait(self._cancel, min(self._poll, remaining))

    def _set_and_wait(self, source, target, expect, timeout, allowed, should_continue):
        with self._io_lock:
            fsm = self._read(should_continue)
            if fsm != source:
                raise _Failure("rejected_fsm", f"送信前にモードが{fsm}へ変わったため後続の起立指令を送りません")
            self._current(should_continue)
            try:
                code = self.client.set_fsm(target)
            except Exception as error:
                self._current(should_continue)
                raise _Failure("rpc_error", f"モード{target}への指令に失敗しました ({type(error).__name__})") from None
            recorded_code = code if isinstance(code, int) and not isinstance(code, bool) else None
            with self._state_lock:
                self._state["last_code"] = recorded_code
                self._state["commands"].append({"target": target, "code": recorded_code})
            self._event("set_fsm", target=target, code=recorded_code)
            self._current(should_continue)
            if isinstance(code, bool) or not isinstance(code, int) or code != 0:
                raise _Failure("rpc_error", f"モード{target}への指令が受理されませんでした (code={recorded_code})")
        self._wait_for(expect, timeout, allowed, should_continue)

    def _stand(self, should_continue):
        fsm = self._read(should_continue)
        if fsm == 811:
            self._update(status="already_running", phase="completed")
            return
        if fsm not in (0, 1, 4, 701):
            raise _Failure("rejected_fsm", f"モード{fsm}からの起立は開始しません")
        if fsm == 701:
            self._update(phase="rising")
            self._wait_for(811, 25, {701, 811}, should_continue)
        else:
            if fsm == 0:
                self._update(phase="damping")
                self._set_and_wait(0, 1, 1, 6, {0, 1}, should_continue)
                fsm = 1
            if fsm == 1:
                self._update(phase="locked_stand")
                self._set_and_wait(1, 4, 4, 8, {1, 4}, should_continue)
            self._update(phase="settling_locked")
            self._pause(3, should_continue)
            self._update(phase="rising")
            self._set_and_wait(4, 701, 811, 25, {4, 701, 811}, should_continue)
        self._update(phase="settling_running")
        self._pause(4, should_continue)
        if self._read(should_continue) != 811:
            raise _Failure("rejected_fsm", "起立後のRunning状態を確認できませんでした")
        self._update(status="finished", phase="completed")

    def run(self, posture: str, ready: bool, should_continue=None) -> dict:
        if not self._run_lock.acquire(blocking=False):
            return {**self.state(), "busy": True, "status": "busy", "error": "別の起立処理が進行中です"}
        motion_acquired = False
        try:
            with self._state_lock:
                self._cancel = threading.Event()
                self._state = self._new_state()
            if posture != "supine" or not isinstance(posture, str) or ready is not True:
                raise _Failure("rejected_input", "仰向けの姿勢と起立の準備完了を明示してください")
            if should_continue is not None and not callable(should_continue):
                raise _Failure("rejected_input", "起立の継続条件が不正です")
            self._current(should_continue)
            motion_acquired = self.motion_lock.acquire(blocking=False)
            if not motion_acquired:
                raise _Failure("motion_busy", "別の動作中のため起立を開始しません")
            self._update(busy=True, status="running", phase="checking")
            self._current(should_continue)
            if self.dry:
                self._update(status="preview", phase="completed",
                             note="ローカルの起立プレビュー。指令は送らず、実際の起立を確認した意味ではありません")
                self._event("preview", posture="supine", targets=[1, 4, 701], expected_fsm=811)
            else:
                self._stand(should_continue)
        except _Cancelled:
            self._update(status="cancelled", phase="cancelled",
                         error="後続の起立指令を取り消しました。開始済みの起立動作の停止は保証しません")
        except _Failure as error:
            self._update(status=error.status, error=str(error))
        except Exception as error:
            self._update(status="error", error=f"起立処理に失敗しました ({type(error).__name__})")
        finally:
            self._update(busy=False)
            if motion_acquired:
                self.motion_lock.release()
            self._run_lock.release()
        result = self.state()
        self._event("completed", **result)
        return result
