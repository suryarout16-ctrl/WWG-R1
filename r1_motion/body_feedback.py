"""R1のLowStateを読むだけのモニター。DDSへの指令送信は行わない。"""
from __future__ import annotations

import copy
import math
import threading
import time


class _PollingSubscriber:
    """DDS listenerを使わず、最新1件を専用Pythonスレッドで読む。"""

    def __init__(self, channel, qos, interval):
        self._channel, self._qos, self._interval = channel, qos, interval
        self._stop = threading.Event()
        self._thread = None

    def Init(self, handler, queue_len):
        self._channel.SetReader(qos=self._qos, handler=None, queueLen=0)

        def run():
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    sample = self._channel.Read(timeout=min(0.05, self._interval))
                    if sample is not None and not self._stop.is_set():
                        handler(sample)
                except Exception:
                    # 不正な観測を古い正常値のままにしない。feed側で無効化する。
                    if not self._stop.is_set():
                        handler(None)
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
        self._channel.CloseReader()


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

    def start(self):
        with self._lifecycle_lock:
            if self._subscriber is not None:
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
                qos = Qos(Policy.History.KeepLast(1))
                factory = lambda topic, datatype: _PollingSubscriber(
                    ChannelFactory().CreateChannel(topic, datatype), qos, self._interval)
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
            subscriber = factory("rt/lowstate", message_type)
            try:
                subscriber.Init(lambda msg: self.feed(msg, _generation=generation), 10)
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
                if self._closed or (_generation is not None and _generation != self._generation):
                    return False
                if self._latest is not None and self._latest["t"] > now:
                    return False
                if self._last_tick is not None:
                    delta = (tick - self._last_tick) & 0xffffffff
                    if not 0 < delta < 2**31:
                        return False
                self._last_tick = tick
                self._latest = sample
            return True
        except Exception as error:
            # 不正フレームを受けた後、以前の正常値をそのまま使わせない。
            with self._lock:
                if self._closed or (_generation is not None and _generation != self._generation):
                    return False
                if self._latest is not None and self._latest["t"] > now:
                    return False
                self._latest = self._empty(now, f"{type(error).__name__}: {error}", verified)
            return True

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._latest)

    def close(self):
        with self._lifecycle_lock:
            subscriber, self._subscriber = self._subscriber, None
            with self._lock:
                self._closed = True
                self._generation += 1
                self._latest = None
            if subscriber is not None:
                try:
                    subscriber.Close()
                except Exception:
                    return False
            return True
