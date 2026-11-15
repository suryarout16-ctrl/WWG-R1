"""録音デバイスなしで、通話の区切り・エコー抑制・終了を検証する。"""
import pathlib
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r1_chat"))
from voice_call import UtteranceDetector, VoiceCall


RATE = 16000


def frame(seconds=0.1, amplitude=0.08, dtype=np.int16):
    samples = np.full(round(RATE * seconds), amplitude, dtype=np.float32)
    return (samples * 32767).astype(dtype) if dtype == np.int16 else samples.astype(dtype)


def eventually(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class DetectorTests(unittest.TestCase):
    def test_silence_does_not_trigger_and_noise_preroll_is_bounded(self):
        detector = UtteranceDetector()
        for _ in range(100):
            self.assertIsNone(detector.feed(frame(amplitude=0.001)))
        for _ in range(3):
            self.assertIsNone(detector.feed(frame()))
        result = None
        for _ in range(8):
            result = detector.feed(frame(amplitude=0))
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, np.int16)
        self.assertEqual(len(result), round(1.3 * RATE))

    def test_short_noise_click_is_not_a_turn(self):
        detector = UtteranceDetector()
        self.assertIsNone(detector.feed(frame(seconds=0.04)))
        self.assertIsNone(detector.feed(frame(seconds=0.8, amplitude=0)))

    def test_short_pauses_preserve_one_turn_and_count_only_voiced_time(self):
        detector = UtteranceDetector()
        for chunk in [frame(seconds=.15), frame(seconds=.3, amplitude=0), frame(seconds=.15)]:
            self.assertIsNone(detector.feed(chunk))
        result = detector.feed(frame(seconds=.8, amplitude=0))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), round(1.4 * RATE))

    def test_silence_does_not_make_a_too_short_voice_valid(self):
        detector = UtteranceDetector()
        self.assertIsNone(detector.feed(frame(seconds=.1)))
        self.assertIsNone(detector.feed(frame(seconds=.8, amplitude=0)))

    def test_max_duration_bounds_continuous_speech(self):
        detector = UtteranceDetector(max_seconds=1.0)
        result = None
        for _ in range(4):
            result = detector.feed(frame(seconds=.3))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), RATE)

    def test_float_and_int16_have_same_detection_scale(self):
        for dtype in (np.int16, np.float32):
            detector = UtteranceDetector()
            self.assertIsNone(detector.feed(frame(seconds=.3, dtype=dtype)))
            self.assertIsNotNone(detector.feed(frame(seconds=.8, amplitude=0, dtype=dtype)))

    def test_stereo_int16_does_not_saturate_during_downmix(self):
        detector = UtteranceDetector(threshold=.1)
        quiet = np.column_stack([frame(seconds=.3, amplitude=.02)] * 2)
        self.assertIsNone(detector.feed(quiet))
        self.assertLess(detector.level, .03)

    def test_reset_discards_partial_utterance(self):
        detector = UtteranceDetector()
        detector.feed(frame(seconds=.3))
        detector.reset()
        self.assertIsNone(detector.feed(frame(seconds=.8, amplitude=0)))

    def test_invalid_configuration_is_rejected_before_recording(self):
        for values in ({"threshold": 0}, {"threshold": float("nan")}, {"silence_seconds": 0},
                       {"max_seconds": .1}, {"sample_rate": 0}, {"pre_roll_seconds": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                UtteranceDetector(**values)


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def feed(self, samples):
        self.callback(samples[:, None], len(samples), None, None)


class CallTests(unittest.TestCase):
    def setUp(self):
        self.streams = []
        self.calls = []

    def make_call(self, callback, **kwargs):
        def factory(**options):
            stream = FakeStream(**options)
            self.streams.append(stream)
            return stream
        call = VoiceCall(callback, stream_factory=factory, cooldown_seconds=0, **kwargs)
        self.calls.append(call)
        return call

    def tearDown(self):
        for call in self.calls:
            call.stop()

    def listen(self, call):
        self.assertTrue(eventually(lambda: call.state()["listening"]))

    def utter(self, stream):
        stream.feed(frame(seconds=.3))
        stream.feed(frame(seconds=.8, amplitude=0))

    def test_multiple_turns_need_only_one_start(self):
        received = []
        call = self.make_call(lambda audio, valid: received.append((len(audio), valid())))
        self.assertTrue(call.start())
        self.assertFalse(call.start())
        self.listen(call)
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: len(received) == 1))
        self.listen(call)
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: len(received) == 2))
        self.assertTrue(all(current for _, current in received))
        call.stop()
        self.assertTrue(self.streams[0].closed)
        self.assertFalse(call.state()["active"])

    def test_own_voice_is_discarded_while_reply_is_processing(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        def reply(audio, current):
            received.append(audio)
            entered.set()
            release.wait(2)

        call = self.make_call(reply)
        call.start()
        self.listen(call)
        self.utter(self.streams[-1])
        self.assertTrue(entered.wait(2))
        try:
            for _ in range(5):
                self.utter(self.streams[-1])
            self.assertFalse(call.state()["listening"])
        finally:
            release.set()
        self.listen(call)
        time.sleep(.05)
        self.assertEqual(len(received), 1)

    def test_external_playback_discards_partial_utterance(self):
        allowed = threading.Event()
        allowed.set()
        received = []
        call = self.make_call(lambda audio, valid: received.append(audio), can_listen=allowed.is_set)
        call.start()
        self.listen(call)
        self.streams[-1].feed(frame(seconds=.3))
        allowed.clear()
        self.assertTrue(eventually(lambda: not call.state()["listening"]))
        self.utter(self.streams[-1])
        allowed.set()
        self.listen(call)
        self.streams[-1].feed(frame(seconds=.8, amplitude=0))
        time.sleep(.05)
        self.assertEqual(received, [])
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: len(received) == 1))

    def test_stop_invalidates_inflight_reply_and_restart_keeps_sessions_separate(self):
        entered, release = threading.Event(), threading.Event()
        statuses, spoken = [], []

        def reply(audio, current):
            is_first = not statuses
            statuses.append(current)
            if is_first:
                entered.set()
                release.wait(2)
            if current():
                spoken.append("reply")

        call = self.make_call(reply)
        call.start()
        self.listen(call)
        old_stream = self.streams[-1]
        self.utter(old_stream)
        self.assertTrue(entered.wait(2))
        try:
            call.stop()
            self.assertFalse(statuses[0]())
            call.start()
            self.listen(call)
            self.utter(old_stream)  # 過去のコールバックを呼んでも新通話に混ざらない。
            self.utter(self.streams[-1])
            self.assertTrue(eventually(lambda: call.state()["processing"]))
            self.assertEqual(len(statuses), 1)
        finally:
            release.set()
        self.assertTrue(eventually(lambda: len(spoken) == 1))
        self.assertEqual(len(statuses), 2)

    def test_suspend_resume_never_reuses_old_audio(self):
        received = []
        call = self.make_call(lambda audio, valid: received.append(audio))
        call.start()
        self.listen(call)
        call.suspend()
        self.utter(self.streams[-1])
        self.assertFalse(call.state()["listening"])
        call.resume()
        self.listen(call)
        self.streams[-1].feed(frame(seconds=.8, amplitude=0))
        time.sleep(.05)
        self.assertEqual(received, [])

    def test_threshold_can_be_tuned_without_reopening_the_microphone(self):
        received = []
        call = self.make_call(lambda audio, valid: received.append(audio), threshold=.2)
        call.start()
        self.listen(call)
        self.utter(self.streams[-1])
        time.sleep(.05)
        self.assertEqual(received, [])
        call.set_threshold(.02)
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: len(received) == 1))
        self.assertEqual(len(self.streams), 1)

    def test_processing_error_is_reported_and_call_can_listen_again(self):
        errors = []

        def fail(audio, current):
            raise RuntimeError("認識テストエラー")

        call = self.make_call(fail, on_error=errors.append)
        call.start()
        self.listen(call)
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: bool(errors)))
        self.listen(call)
        self.assertIn("認識テストエラー", call.state()["error"])

    def test_failed_stream_start_does_not_leave_call_active(self):
        class FailingStream(FakeStream):
            def start(self):
                raise OSError("マイクを開けません")

        call = VoiceCall(lambda *_: None, stream_factory=lambda **kwargs: FailingStream(**kwargs))
        self.calls.append(call)
        with self.assertRaises(OSError):
            call.start()
        self.assertFalse(call.state()["active"])
        self.assertIn("マイクを開けません", call.state()["error"])

    def test_device_overflow_discards_the_incomplete_turn(self):
        received = []
        call = self.make_call(lambda audio, valid: received.append(audio))
        call.start()
        self.listen(call)
        stream = self.streams[-1]
        stream.feed(frame(seconds=.3))
        self.assertTrue(eventually(lambda: call.state()["level"] > 0))
        stream.callback(frame()[:, None], 1600, None, "input overflow")
        self.listen(call)
        stream.feed(frame(seconds=.8, amplitude=0))
        time.sleep(.05)
        self.assertEqual(received, [])
        self.assertIn("input overflow", call.state()["error"])

    def test_overflow_during_detection_invalidates_dequeued_utterance(self):
        received = []
        detected, release = threading.Event(), threading.Event()
        original_feed = UtteranceDetector.feed

        def interrupted_feed(detector, samples):
            utterance = original_feed(detector, samples)
            if utterance is not None:
                detected.set()
                release.wait(2)
            return utterance

        call = self.make_call(lambda audio, valid: received.append(audio))
        with mock.patch.object(UtteranceDetector, "feed", interrupted_feed):
            call.start()
            self.listen(call)
            stream = self.streams[-1]
            self.utter(stream)
            self.assertTrue(detected.wait(2))
            try:
                stream.callback(frame()[:, None], 1600, None, "input overflow")
            finally:
                release.set()
            self.listen(call)
        self.assertEqual(received, [])
        # 次に正常に録れた発話から通話を再開できる。
        self.utter(stream)
        self.assertTrue(eventually(lambda: len(received) == 1))

    def test_stream_is_closed_even_if_stop_reports_a_device_error(self):
        class StopFailure(FakeStream):
            def stop(self):
                raise OSError("マイク切断")

        stream = StopFailure(callback=lambda *_: None)
        call = VoiceCall(lambda *_: None, stream_factory=lambda **_: stream)
        self.calls.append(call)
        call.start()
        self.assertTrue(call.stop())
        self.assertTrue(stream.closed)
        self.assertFalse(call.state()["active"])
        self.assertIn("マイク切断", call.state()["error"])


if __name__ == "__main__":
    unittest.main()
