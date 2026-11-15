"""通話形式の音声入力。無音で区切り、返答中はマイク入力を捨てる。

これは半二重の通話: 相手の話を聞く → on_utterance → 返答が終わる → 再び聞く。
音声コールバックでは録音データのコピーとキュー投入だけを行う。
認識・LLM・再生はワーカー上の on_utterance(samples, is_current) で同期実行する。
is_current() は通話終了・かけ直しで False になる。呼び出し側も認識や生成の後、
発話の直前に確認し、終了した通話の返答を再生しないこと。
"""
from __future__ import annotations

import collections
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from audio_input import resolve_input_device


def _pcm_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples)
    original_dtype = data.dtype
    if data.ndim == 2:
        data = data[:, 0] if data.shape[1] == 1 else data.astype(np.float32).mean(axis=1)
    if data.ndim != 1:
        raise ValueError("音声は mono または samples × channels の配列を指定してください")
    if data.dtype == np.int16:
        return data.copy()
    # sounddevice の float32 は -1〜1。テスト・別入力でも同じ振幅で判定する。
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        data = data.astype(np.float32) / max(abs(limits.min), limits.max)
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)


class UtteranceDetector:
    """サンプル数で発話を区切るRMS検出器。マイクや時計に依存しない。"""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.012,
                 silence_seconds: float = 0.8, min_speech_seconds: float = 0.25,
                 max_seconds: float = 15.0, pre_roll_seconds: float = 0.2):
        if sample_rate <= 0 or not 0 < threshold <= 1:
            raise ValueError("sample_rate と threshold が範囲外です")
        if silence_seconds <= 0 or min_speech_seconds <= 0 or max_seconds < min_speech_seconds:
            raise ValueError("発話検出の時間設定が範囲外です")
        if pre_roll_seconds < 0 or pre_roll_seconds + min_speech_seconds > max_seconds:
            raise ValueError("pre_roll_seconds が範囲外です")
        self.sample_rate, self.threshold = sample_rate, threshold
        self.silence_samples = max(1, round(sample_rate * silence_seconds))
        self.min_speech_samples = max(1, round(sample_rate * min_speech_seconds))
        self.max_samples = max(1, round(sample_rate * max_seconds))
        self.pre_roll_samples = round(sample_rate * pre_roll_seconds)
        self.level = 0.0
        self.reset()

    def reset(self) -> None:
        self._pre: collections.deque[np.ndarray] = collections.deque()
        self._pre_count = 0
        self._frames: list[np.ndarray] = []
        self._total = self._voiced = self._silent = 0

    def feed(self, samples: np.ndarray) -> np.ndarray | None:
        pcm = _pcm_mono(samples)
        if not len(pcm):
            return None
        signal = pcm.astype(np.float32) / 32768.0
        self.level = float(np.sqrt(np.mean(signal * signal)))
        voiced = self.level >= self.threshold
        if not self._frames:
            # pre-roll は開始直前の音を保持し、語頭が切れるのを抑える。
            if not voiced:
                self._pre.append(pcm)
                self._pre_count += len(pcm)
                while self._pre_count > self.pre_roll_samples and self._pre:
                    excess = self._pre_count - self.pre_roll_samples
                    first = self._pre.popleft()
                    if len(first) > excess:
                        self._pre.appendleft(first[excess:])
                        self._pre_count -= excess
                    else:
                        self._pre_count -= len(first)
                return None
            self._frames = [*self._pre, pcm]
            self._total = self._pre_count + len(pcm)
            self._voiced = len(pcm)
            self._pre.clear()
            self._pre_count = 0
        else:
            self._frames.append(pcm)
            self._total += len(pcm)
            if voiced:
                self._voiced += len(pcm)
                self._silent = 0
            else:
                self._silent += len(pcm)
        if self._silent < self.silence_samples and self._total < self.max_samples:
            return None
        result = np.concatenate(self._frames)[:self.max_samples] if self._voiced >= self.min_speech_samples else None
        self.reset()
        return result


@dataclass
class _Session:
    generation: int
    stop: threading.Event = field(default_factory=threading.Event)
    audio: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=20))
    io_lock: threading.Lock = field(default_factory=threading.Lock)
    stream: object | None = None
    thread: threading.Thread | None = None
    listening: bool = False
    processing: bool = False
    suspended: bool = False
    level: float = 0.0
    resume_at: float = 0.0
    error: str = ""
    input_ready: bool = True
    transport_epoch: int = 0
    manager: threading.Thread | None = None
    input_lost: threading.Event = field(default_factory=threading.Event)
    opened_at: float = 0.0
    last_callback_at: float = 0.0
    pinned_device: str | None = None
    device_detail: str = ""


class VoiceCall:
    """1回の開始操作で継続する音声会話。sounddeviceはstart時だけ読み込む。"""

    def __init__(self, on_utterance: Callable[[np.ndarray, Callable[[], bool]], None], *,
                 can_listen: Callable[[], bool] = lambda: True,
                 on_error: Callable[[str], None] | None = None,
                 stream_factory=None, sample_rate: int = 16000, threshold: float = 0.012,
                 silence_seconds: float = 0.8, min_speech_seconds: float = 0.25,
                 max_seconds: float = 15.0, device=None, cooldown_seconds: float = 0.35,
                 auto_reconnect: bool = False, reconnect_interval: float = 1.0,
                 liveness_timeout: float = 3.0, device_refresher=None):
        self.on_utterance, self.can_listen, self.on_error = on_utterance, can_listen, on_error
        self.stream_factory, self.device = stream_factory, device
        self.detector_options = dict(sample_rate=sample_rate, threshold=threshold,
                                     silence_seconds=silence_seconds, min_speech_seconds=min_speech_seconds,
                                     max_seconds=max_seconds)
        UtteranceDetector(**self.detector_options)  # 設定誤りはマイクを開く前に通知する。
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds は0以上にしてください")
        self.cooldown_seconds, self.sample_rate = cooldown_seconds, sample_rate
        self.threshold = threshold
        if not isinstance(auto_reconnect, bool):
            raise ValueError("auto_reconnectは真偽値で指定してください")
        for value in (reconnect_interval, liveness_timeout):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("マイク再接続の待機時間は0より大きい有限値にしてください")
        if auto_reconnect and device is None and stream_factory is None:
            raise ValueError("自動再接続には同じ入力機器の名前を指定してください")
        if device_refresher is not None and not callable(device_refresher):
            raise ValueError("入力機器の再列挙処理が不正です")
        self.auto_reconnect = auto_reconnect
        self.device_refresher = device_refresher
        self.reconnect_interval, self.liveness_timeout = float(reconnect_interval), float(liveness_timeout)
        self._lock = threading.Lock()
        self._delivery_lock = threading.Lock()
        self._input_lifecycle_lock = threading.Lock()
        self._session: _Session | None = None
        self._generation = 0
        self._last_error = ""
        self._input_metadata = {}

    def _current(self, session: _Session) -> bool:
        with self._lock:
            return self._session is session and not session.stop.is_set()

    def _input_current(self, session: _Session, transport: int) -> bool:
        return self._current(session) and (not self.auto_reconnect or
                (session.input_ready and session.transport_epoch == transport))

    def _report(self, session: _Session, error: Exception | str) -> None:
        message = str(error)
        session.error = message
        if self._current(session):
            self._last_error = message
            if self.on_error:
                try:
                    self.on_error(message)
                except Exception:
                    pass

    def start(self) -> bool:
        with self._lock:
            if self._session is not None:
                return False
            self._generation += 1
            session = _Session(self._generation)
            self._session = session
            self._last_error = ""
        try:
            if self.auto_reconnect:
                session.input_ready = False
                session.thread = threading.Thread(target=self._run, args=(session,), daemon=True,
                                                  name=f"R1-voice-call-{session.generation}")
                session.manager = threading.Thread(target=self._manage_input, args=(session,), daemon=True,
                                                   name=f"R1-microphone-{session.generation}")
                session.thread.start()
                session.manager.start()
                return self._current(session)
            factory = self.stream_factory
            device = self.device
            if factory is None:
                import sounddevice as sd
                if device is not None:
                    device, metadata = resolve_input_device(device, sd)
                    with self._lock:
                        self._input_metadata = metadata
                factory = sd.InputStream
            with session.io_lock:
                stream = factory(samplerate=self.sample_rate, channels=1, dtype="int16",
                                 blocksize=max(1, self.sample_rate // 50), device=device,
                                 callback=lambda data, frames, timing, status: self._capture(session, data, status))
                session.stream = stream
                if session.stop.is_set():
                    stream.close()
                    return False
                session.thread = threading.Thread(target=self._run, args=(session,), daemon=True,
                                                  name=f"R1-voice-call-{session.generation}")
                session.thread.start()
                stream.start()
            return self._current(session)
        except Exception as error:
            self._report(session, error)
            self._close(session)
            with self._lock:
                if self._session is session:
                    self._session = None
            raise

    def _close_input_unlocked(self, session: _Session) -> None:
        stream, session.stream = session.stream, None
        if stream is not None:
            for operation in (stream.stop, stream.close):
                try:
                    operation()
                except Exception:
                    pass

    def _disconnect_input(self, session: _Session) -> None:
        session.input_ready = session.listening = False
        session.level = 0.0
        session.transport_epoch += 1
        self._drain(session)
        with session.io_lock:
            self._close_input_unlocked(session)

    def _waiting_input(self, session: _Session, detail) -> None:
        session.input_ready = session.listening = False
        session.transport_epoch += 1
        session.level = 0.0
        self._drain(session)
        session.device_detail = str(detail)[:500]
        message = "ヘッドセットの接続待ち。装着し直してください"
        if session.error != message and self._current(session):
            self._report(session, message)

    def _manage_input(self, session: _Session) -> None:
        acquired = False
        try:
            while self._current(session) and not acquired:
                acquired = self._input_lifecycle_lock.acquire(timeout=0.05)
            if not acquired:
                return
            while self._current(session):
                failed = False
                try:
                    factory, device = self.stream_factory, session.pinned_device or self.device
                    if factory is None:
                        import sounddevice as sd
                        device, metadata = resolve_input_device(device, sd)
                        session.pinned_device = metadata["name"]
                        with self._lock:
                            if self._session is session and not session.stop.is_set():
                                self._input_metadata = metadata
                        factory = sd.InputStream
                    with session.io_lock:
                        if not self._current(session):
                            break
                        session.transport_epoch += 1
                        transport = session.transport_epoch
                        session.input_lost.clear()
                        stream = factory(samplerate=self.sample_rate, channels=1, dtype="int16",
                                         blocksize=max(1, self.sample_rate // 50), device=device,
                                         callback=lambda data, frames, timing, status, epoch=transport:
                                             self._capture(session, data, status, epoch))
                        session.stream = stream
                        if not self._current(session):
                            break
                        stream.start()
                        if not self._current(session):
                            break
                        if session.input_lost.is_set():
                            raise RuntimeError(session.device_detail or "マイク入力が途切れました")
                        session.opened_at = session.last_callback_at = time.monotonic()
                        session.resume_at = session.opened_at + self.cooldown_seconds
                        session.error = session.device_detail = ""
                        with self._lock:
                            if self._session is session:
                                self._last_error = ""
                        session.input_ready = True
                    while self._current(session):
                        if session.input_lost.is_set():
                            raise RuntimeError(session.device_detail or "マイク入力が途切れました")
                        if not getattr(stream, "active", True):
                            raise RuntimeError("マイクの入力ストリームが停止しました")
                        if time.monotonic() - session.last_callback_at > self.liveness_timeout:
                            raise RuntimeError("マイクからデータが届きません")
                        session.stop.wait(min(0.1, self.reconnect_interval))
                except Exception as error:
                    failed = True
                    self._waiting_input(session, error)
                finally:
                    self._disconnect_input(session)
                if failed and self._current(session) and self.device_refresher is not None:
                    try:
                        self.device_refresher()
                    except Exception as error:
                        original = session.device_detail[:300]
                        self._waiting_input(session, f"{original}; device refresh: {type(error).__name__}: {str(error)[:120]}")
                if self._current(session):
                    session.stop.wait(self.reconnect_interval)
        finally:
            if acquired:
                try:
                    self._disconnect_input(session)
                finally:
                    self._input_lifecycle_lock.release()

    def _close(self, session: _Session) -> None:
        session.stop.set()
        session.listening = False
        with session.io_lock:
            if session.stream is not None:
                errors = []
                try:
                    session.stream.stop()
                except Exception as error:
                    errors.append(str(error))
                try:
                    session.stream.close()
                except Exception as error:
                    errors.append(str(error))
                session.stream = None
                if errors:
                    session.error = " / ".join(errors)
                    with self._lock:
                        if self._session is session or self._session is None:
                            self._last_error = session.error

    def stop(self) -> bool:
        # 先に世代を無効化する。STTやLLMが戻るのを待たずに終了操作を返す。
        with self._lock:
            session, self._session = self._session, None
            if self.auto_reconnect:
                self._last_error = ""
        if session is None:
            return False
        if self.auto_reconnect:
            session.stop.set()
            session.input_ready = session.listening = False
            session.transport_epoch += 1
            self._drain(session)
            # OS open/close may be blocked. The manager owns cleanup; the next manager waits for it.
            return True
        self._close(session)
        return True

    def suspend(self) -> None:
        with self._lock:
            if self._session:
                self._session.suspended = True
                self._session.listening = False

    def resume(self) -> None:
        with self._lock:
            if self._session:
                self._session.suspended = False
                self._session.resume_at = time.monotonic() + self.cooldown_seconds

    def set_threshold(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("マイクのしきい値は0より大きく1以下にしてください")
        self.threshold = value

    def state(self) -> dict:
        with self._lock:
            session = self._session
            return {"active": session is not None, "listening": bool(session and session.listening),
                    "processing": bool(session and session.processing),
                    "level": round(session.level, 4) if session else 0.0,
                    "threshold": self.threshold, "error": session.error if session else self._last_error,
                    "microphone": dict(self._input_metadata),
                    "input_ready": bool(session and (not self.auto_reconnect or session.input_ready)),
                    "waiting_for_device": bool(self.auto_reconnect and session and not session.input_ready),
                    "reconnecting": bool(self.auto_reconnect and session and not session.input_ready),
                    "input_closing": bool(self.auto_reconnect and session is None and self._input_lifecycle_lock.locked()),
                    "device_detail": session.device_detail if session else ""}

    def _capture(self, session: _Session, data: np.ndarray, status, transport=None) -> None:
        if self.auto_reconnect:
            if session.stop.is_set() or session.transport_epoch != transport:
                return
            session.last_callback_at = time.monotonic()
            if status:
                session.device_detail = str(status)
                session.input_ready = session.listening = False
                session.input_lost.set()
                return
        if session.stop.is_set() or not session.listening or session.processing:
            return
        # キューが詰まった場合は一部だけを落として単語をつなげない。
        if status:
            session.error = str(status)
            session.listening = False
            return
        try:
            pcm = data.copy()
            session.audio.put_nowait((transport, pcm) if self.auto_reconnect else pcm)
        except queue.Full:
            session.listening = False
            session.error = "録音処理が遅れたため、この発話を破棄しました"

    @staticmethod
    def _drain(session: _Session) -> None:
        while True:
            try:
                session.audio.get_nowait()
            except queue.Empty:
                return

    def _run(self, session: _Session) -> None:
        detector = UtteranceDetector(**self.detector_options)
        was_blocked = False
        while self._current(session):
            try:
                transport = session.transport_epoch
                allowed = self._input_current(session, transport) and not session.suspended and bool(self.can_listen())
                if not allowed:
                    session.listening = False
                    detector.reset()
                    self._drain(session)
                    was_blocked = True
                    session.stop.wait(0.05)
                    continue
                if was_blocked:
                    session.resume_at = time.monotonic() + self.cooldown_seconds
                    was_blocked = False
                if not session.listening:
                    detector.reset()
                    self._drain(session)
                    if time.monotonic() < session.resume_at:
                        session.stop.wait(0.05)
                        continue
                    session.listening = True
                try:
                    samples = session.audio.get(timeout=0.05)
                except queue.Empty:
                    continue
                if self.auto_reconnect:
                    packet_transport, samples = samples
                    if packet_transport != transport:
                        detector.reset()
                        continue
                # suspend/外部の発話が入力を待っている間に始まる場合も破棄する。
                if (not self._input_current(session, transport) or not session.listening
                        or session.suspended or not self.can_listen()):
                    session.listening = False
                    continue
                detector.threshold = self.threshold
                utterance = detector.feed(samples)
                session.level = detector.level
                if utterance is None:
                    continue
                # feed中に入力オーバーフローが通知された場合も、途中の音を認識に回さない。
                if not session.listening:
                    detector.reset()
                    self._drain(session)
                    continue
                session.processing, session.listening = True, False
                self._drain(session)
                # かけ直しても、終了した通話のブロック中処理と並列にLLMを呼ばない。
                acquired = False
                try:
                    while self._input_current(session, transport) and not acquired:
                        acquired = self._delivery_lock.acquire(timeout=0.05)
                    if acquired and self._input_current(session, transport):
                        self.on_utterance(utterance, lambda epoch=transport: self._input_current(session, epoch))
                finally:
                    if acquired:
                        self._delivery_lock.release()
                    session.processing = False
                    session.resume_at = time.monotonic() + self.cooldown_seconds
                    detector.reset()
                    self._drain(session)
            except Exception as error:
                self._report(session, error)
                session.listening = False
                detector.reset()
                self._drain(session)
                session.resume_at = time.monotonic() + self.cooldown_seconds
                session.stop.wait(0.05)
