"""R1のLowStateを読むだけのモニター。DDSへの指令送信は行わない。"""
from __future__ import annotations

import copy
import math
import threading
import time


class _PollingSubscriber:
    """DDS listenerを使わず、最新1件を専用Pythonスレッドで読む。"""

    def __init__(self, channel, qos, interval, *, channel_factory=None,
                 on_reconnect=None, monotonic=time.monotonic, recovery_after=1.0):
        self._channel, self._qos, self._interval = channel, qos, interval
        self._channel_factory, self._on_reconnect = channel_factory, on_reconnect
        self._clock, self._recovery_after = monotonic, recovery_after
        if (isinstance(recovery_after, bool) or not isinstance(recovery_after, (int, float))
                or not math.isfinite(recovery_after) or recovery_after <= 0):
            raise ValueError("再接続間隔は0より大きい有限値にしてください")
        self._stop = threading.Event()
        self._thread = None

    def Init(self, handler, queue_len):
        self._channel.SetReader(qos=self._qos, handler=None, queueLen=0)

        def run():
            last_accepted = self._clock()
            next_recovery = last_accepted + self._recovery_after
            reason = "LowStateが届きません"
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    sample = (self._channel.Read(timeout=min(0.05, self._interval))
                              if self._channel is not None else None)
                    if sample is not None and not self._stop.is_set():
                        if handler(sample) is not False:
                            last_accepted = self._clock()
                        else:
                            reason = "LowStateのtickが更新されません"
                    elif sample is None:
                        reason = "LowStateが届きません（読取エラーを含む）"
                except Exception as error:
                    # 不正な観測を古い正常値のままにしない。feed側で無効化する。
                    if not self._stop.is_set():
                        handler(None)
                    reason = f"LowState読取失敗: {type(error).__name__}"
                now = self._clock()
                if (self._channel_factory is not None and not self._stop.is_set()
                        and now - last_accepted >= self._recovery_after and now >= next_recovery):
                    # 読取が戻った同じスレッド上だけで破棄・再生成する。
                    # Close()はこのスレッドの終了を待つため、Read中のreaderを破棄しない。
                    try:
                        if self._channel is not None:
                            self._channel.CloseReader()
                            self._channel = None
                        if not self._stop.is_set():
                            if self._on_reconnect is not None and self._on_reconnect(reason) is False:
                                break
                            channel = self._channel_factory()
                            self._channel = channel
                            if not self._stop.is_set():
                                channel.SetReader(qos=self._qos, handler=None, queueLen=0)
                    except Exception:
                        # 次の試行も同じ間隔を空ける。破棄に失敗したreaderへ重ねて作らない。
                        if not self._stop.is_set():
                            handler(None)
                    next_recovery = self._clock() + self._recovery_after
                self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))

        self._thread = threading.Thread(target=run, name="r1-lowstate-poll", daemon=True)
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            raise

    def Close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                # Read中のreaderを破棄しない。終了未確認を呼び出し元へ返す。
                raise RuntimeError("LowState読み取りスレッドが終了していません")
        if self._channel is not None:
            self._channel.CloseReader()
            self._channel = None


class MotorFeedbackMonitor:
    """既に初期化されたDDSからrt/lowstateを最大50Hzで読む。

    start()までSDKをimportしない。テストではcrc_validator(msg)->boolと
    subscriber_factory(topic, message_type)->subscriberを注入できる。
    注入したfactoryにはmessage_type=Noneを渡す。subscriberはInit(callback,10)
    とClose()を実装する。tは新しいtickを受けた単調時計の時刻である。
    """

    def __init__(self, sample_hz=50, monotonic=time.monotonic, crc_validator=None,
                 subscriber_factory=None):
        if (isinstance(sample_hz, bool) or not isinstance(sample_hz, (int, float))
                or not math.isfinite(sample_hz) or not 0 < sample_hz <= 50):
            raise ValueError("sample_hzは0より大きく50以下にしてください")
        self._interval = 1.0 / sample_hz
        self._clock = monotonic
        self._crc_validator = crc_validator
        self._subscriber_factory = subscriber_factory
        self._subscriber = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._latest = None
        self._last_sample_t = None
        self._last_tick = None
        self._closed = False
        self._generation = 0
        self._stream_epoch = 0
        self._recovery_count = 0
        self._recovering = False
        self._recovery_reason = ""
        self._candidate_tick = None
        self._candidate_started = None
        self._candidate_count = 0
        self._last_receive_t = self._last_accepted_t = None
        self._last_receive_tick = None
        self._received_count = self._accepted_count = self._rejected_tick_count = 0
        self._last_error = ""

    def start(self):
        with self._lifecycle_lock:
            if self._subscriber is not None:
                if self._closed:
                    raise RuntimeError("以前のLowState読取が終了していないため再開できません")
                return self
            message_type = None
            factory = self._subscriber_factory
            if factory is None:
                # ChannelFactoryInitializeは呼ばない。既存のR1接続を利用する。
                from unitree_sdk2py.core.channel import ChannelFactory
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
                from unitree_sdk2py.utils.crc import CRC
                from cyclonedds.qos import Qos, Policy
                message_type = LowState_
                # 高頻度LowStateのPythonデコードをDDS callbackから外す。
                # History1で古い観測をためず、RPC reader/QoSには触らない。
                qos = Qos(Policy.History.KeepLast(1), Policy.Durability.Volatile)
                def factory(topic, datatype):
                    create = lambda: ChannelFactory().CreateChannel(topic, datatype)
                    return _PollingSubscriber(
                        create(), qos, self._interval, channel_factory=create,
                        on_reconnect=lambda reason: self._begin_recovery(reason, generation),
                        monotonic=self._clock)
                if self._crc_validator is None:
                    crc = CRC()
                    self._crc_validator = lambda msg: int(msg.crc) == crc.Crc(msg)
            elif self._crc_validator is None:
                raise ValueError("注入したsubscriberにはCRC検証関数も指定してください")
            with self._lock:
                self._closed = False
                self._generation += 1
                generation = self._generation
                self._latest = self._last_sample_t = self._last_tick = None
                self._stream_epoch += 1
                self._recovering = False
                self._recovery_reason = ""
                self._candidate_tick = self._candidate_started = None
                self._candidate_count = 0
                self._last_receive_t = self._last_accepted_t = self._last_receive_tick = None
                self._received_count = self._accepted_count = self._rejected_tick_count = 0
                self._last_error = ""
            subscriber = factory("rt/lowstate", message_type)
            try:
                subscriber.Init(lambda msg: self._receive(msg, generation), 10)
            except Exception:
                try:
                    subscriber.Close()
                finally:
                    with self._lock:
                        self._closed = True
                        self._latest = None
                raise
            self._subscriber = subscriber
            return self

    def _receive(self, msg, generation):
        changed = self.feed(msg, _generation=generation)
        with self._lock:
            return bool(changed and not self._closed and generation == self._generation and
                        self._latest and self._latest["crc_valid"] and not self._latest["error"])

    def _begin_recovery(self, reason, generation):
        """新readerの観測を確認するまで以前の正常値とtickを再利用しない。"""
        now = self._clock()
        with self._lock:
            if self._closed or generation != self._generation:
                return False
            self._stream_epoch += 1
            self._recovery_count += 1
            self._recovering = True
            self._recovery_reason = str(reason)
            self._last_error = "LowState再接続後の更新を確認中"
            self._latest = self._empty(now, "LowState再接続後の更新を確認中")
            self._last_sample_t = self._last_tick = None
            self._candidate_tick = self._candidate_started = None
            self._candidate_count = 0
            return True

    @staticmethod
    def _empty(now, error, crc_valid=False):
        return {"t": now, "tick": None, "crc_valid": bool(crc_valid),
                "mode_machine": None, "mode_pr": None, "qlegs": [], "dqlegs": [],
                "motor_fault": [], "error": error}

    def feed(self, msg, _generation=None):
        """DDS callback。サンプルをコピーし、遅延・重複tickは鮮度を更新しない。"""
        now = self._clock()
        with self._lock:
            if self._closed or (_generation is not None and _generation != self._generation):
                return False
            if self._last_sample_t is not None and now - self._last_sample_t < self._interval:
                return False
            self._last_sample_t = now
            observed_epoch = self._stream_epoch
        verified = False
        try:
            if self._crc_validator is None:
                raise ValueError("CRC検証関数が準備できていません")
            verified = bool(self._crc_validator(msg))
            if not verified:
                raise ValueError("CRC不一致")
            tick = msg.tick
            if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick < 2**32:
                raise ValueError("tickが不正です")
            motors = msg.motor_state
            if len(motors) != 35:
                raise ValueError("35関節の状態が必要です")
            faults = []
            for index, motor in enumerate(motors):
                fault = motor.motorstate
                if isinstance(fault, bool) or not isinstance(fault, int) or not 0 <= fault < 2**32:
                    raise ValueError("motorstateが不正です")
                if fault:
                    faults.append({"joint": index, "motorstate": fault})
            qlegs = [float(motor.q) for motor in motors[:12]]
            dqlegs = [float(motor.dq) for motor in motors[:12]]
            if not all(math.isfinite(value) for value in qlegs + dqlegs):
                raise ValueError("脚の角度・速度が有限値ではありません")
            mode_machine, mode_pr = msg.mode_machine, msg.mode_pr
            for value in (mode_machine, mode_pr):
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
                    raise ValueError("modeフィールドが不正です")
            sample = {"t": now, "tick": tick, "crc_valid": True,
                      "mode_machine": mode_machine, "mode_pr": mode_pr,
                      "qlegs": qlegs, "dqlegs": dqlegs, "motor_fault": faults, "error": ""}
            with self._lock:
                if (self._closed or observed_epoch != self._stream_epoch or
                        (_generation is not None and _generation != self._generation)):
                    return False
                if self._latest is not None and self._latest["t"] > now:
                    return False
                self._last_receive_t, self._last_receive_tick = now, tick
                self._received_count += 1
                if self._recovering:
                    # 再生成直後の1件や同じキャッシュだけで鮮度を復活させない。
                    # CRC検証済みのtickが3件以上、0.2秒以上進むことを確認する。
                    if self._candidate_tick is None:
                        self._candidate_tick, self._candidate_started = tick, now
                        self._candidate_count = 1
                        return False
                    candidate_delta = (tick - self._candidate_tick) & 0xffffffff
                    if not 0 < candidate_delta < 2**31:
                        self._rejected_tick_count += 1
                        if candidate_delta != 0:
                            self._candidate_tick, self._candidate_started = tick, now
                            self._candidate_count = 1
                        return False
                    self._candidate_tick = tick
                    self._candidate_count += 1
                    if self._candidate_count < 3 or now - self._candidate_started < .2:
                        return False
                    self._recovering = False
                if self._last_tick is not None:
                    delta = (tick - self._last_tick) & 0xffffffff
                    if not 0 < delta < 2**31:
                        self._rejected_tick_count += 1
                        self._last_error = "LowStateのtickが更新されません"
                        return False
                self._last_tick = tick
                self._latest = sample
                self._last_accepted_t = now
                self._accepted_count += 1
                self._last_error = ""
            return True
        except Exception as error:
            # 不正フレームを受けた後、以前の正常値をそのまま使わせない。
            with self._lock:
                if (self._closed or observed_epoch != self._stream_epoch or
                        (_generation is not None and _generation != self._generation)):
                    return False
                if self._latest is not None and self._latest["t"] > now:
                    return False
                if self._recovering:
                    self._candidate_tick = self._candidate_started = None
                    self._candidate_count = 0
                self._last_error = f"{type(error).__name__}: {error}"
                self._latest = self._empty(now, self._last_error, verified)
            return True

    def snapshot(self):
        with self._lock:
            if self._latest is None:
                return None
            result = copy.deepcopy(self._latest)
            result.update(stream_epoch=self._stream_epoch, recovering=self._recovering,
                          recovery_count=self._recovery_count, recovery_reason=self._recovery_reason)
            return result

    def diagnostics(self):
        """通信の到着と、制御に使える更新の採用を区別する診断値。"""
        now = self._clock()
        with self._lock:
            subscriber = self._subscriber
            thread = getattr(subscriber, "_thread", None)
            return {"stream_epoch": self._stream_epoch, "recovering": self._recovering,
                    "recovery_count": self._recovery_count, "recovery_reason": self._recovery_reason,
                    "last_receive_age": max(0.0, now - self._last_receive_t) if self._last_receive_t is not None else None,
                    "last_accepted_age": max(0.0, now - self._last_accepted_t) if self._last_accepted_t is not None else None,
                    "last_receive_tick": self._last_receive_tick, "accepted_tick": self._last_tick,
                    "received_samples": self._received_count, "accepted_samples": self._accepted_count,
                    "rejected_ticks": self._rejected_tick_count, "last_error": self._last_error,
                    "reader_alive": thread.is_alive() if thread is not None else None,
                    "closed": self._closed}

    def close(self):
        with self._lifecycle_lock:
            subscriber = self._subscriber
            with self._lock:
                self._closed = True
                self._generation += 1
                self._latest = None
                self._recovering = False
            if subscriber is not None:
                try:
                    subscriber.Close()
                except Exception:
                    return False
            self._subscriber = None
            return True
