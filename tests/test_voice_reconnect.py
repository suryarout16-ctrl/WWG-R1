"""ヘッドセット切断時の待機と同一機器の再接続を、偽の入力だけで検証する。"""
import pathlib
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r1_chat"))
import voice_call


def eventually(condition, timeout=1.5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(.005)
    return bool(condition())


class Stream:
    def __init__(self, callback, **kwargs):
        self.callback = callback
        self.active = False
        self.closed = False
        self.started = False

    def start(self):
        self.active = self.started = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True

    def feed(self, amplitude=0, seconds=.1, status=None):
        pcm = np.full((round(16000*seconds), 1), round(amplitude*32767), dtype=np.int16)
        self.callback(pcm, len(pcm), None, status)


class ReconnectTests(unittest.TestCase):
    def setUp(self):
        self.available = threading.Event()
        self.available.set()
        self.streams = []
        self.attempts = 0
        self.received = []
        self.calls = []

    def tearDown(self):
        for call in self.calls:
            call.stop()

    def factory(self, **kwargs):
        self.attempts += 1
        if not self.available.is_set():
            raise OSError("WdmSyncIoctl test failure")
        stream = Stream(**kwargs)
        self.streams.append(stream)
        return stream

    def make(self, **kwargs):
        call = voice_call.VoiceCall(kwargs.pop("on_utterance", lambda audio, current: self.received.append(current())),
                                   stream_factory=self.factory, auto_reconnect=True,
                                   device="Nothing Ear (3a)", reconnect_interval=.01,
                                   liveness_timeout=kwargs.pop("liveness_timeout", 5), cooldown_seconds=0,
                                   **kwargs)
        self.calls.append(call)
        return call

    def ready(self, call):
        self.assertTrue(eventually(lambda: call.state()["input_ready"] and call.state()["listening"]))

    def utter(self, stream):
        stream.feed(.1, .3)
        stream.feed(0, .8)

    def test_unavailable_start_waits_without_recognition_then_resumes_same_session(self):
        self.available.clear()
        call = self.make()
        self.assertTrue(call.start())
        session = call._session
        self.assertTrue(eventually(lambda: call.state()["waiting_for_device"] and self.attempts >= 2))
        self.assertTrue(call.state()["active"])
        self.assertFalse(call.state()["listening"])
        self.assertEqual(self.received, [])
        self.assertIn("接続待ち", call.state()["error"])
        self.available.set()
        self.ready(call)
        self.assertIs(call._session, session)
        self.assertFalse(call.state()["waiting_for_device"])
        self.assertEqual(call.state()["error"], "")
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: self.received == [True]))

    def test_loss_closes_old_stream_discards_partial_audio_and_resumes(self):
        call = self.make()
        call.start(); self.ready(call)
        old = self.streams[-1]
        old.feed(.1, .3)
        self.available.clear()
        old.feed(status="device gone")
        self.assertFalse(call.state()["input_ready"])
        self.assertTrue(eventually(lambda: old.closed and call.state()["waiting_for_device"]))
        old.feed(0, .8)
        self.assertEqual(self.received, [])
        self.available.set(); self.ready(call)
        self.assertIsNot(self.streams[-1], old)
        self.streams[-1].feed(0, .8)
        time.sleep(.03)
        self.assertEqual(self.received, [])
        self.utter(old)
        time.sleep(.02)
        self.assertEqual(self.received, [])
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: self.received == [True]))

    def test_processing_continuation_never_revives_after_rapid_reconnect(self):
        entered, release = threading.Event(), threading.Event()
        predicates = []
        def callback(audio, current):
            predicates.append(current)
            entered.set()
            release.wait(1.5)
        call = self.make(on_utterance=callback)
        call.start(); self.ready(call)
        old = self.streams[-1]
        self.utter(old)
        self.assertTrue(entered.wait(1))
        old.feed(status="headset removed")
        self.assertFalse(predicates[0]())
        self.assertTrue(eventually(lambda: len(self.streams) >= 2 and call.state()["input_ready"]))
        self.assertFalse(predicates[0]())
        release.set(); self.ready(call)
        self.assertFalse(predicates[0]())
        self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: len(predicates) == 2))
        self.assertTrue(predicates[1]())
        self.assertFalse(predicates[0]())

    def test_manual_stop_cancels_waiting_and_does_not_reopen(self):
        self.available.clear()
        call = self.make()
        call.start()
        self.assertTrue(eventually(lambda: self.attempts >= 2))
        call.stop()
        attempts = self.attempts
        self.available.set()
        time.sleep(.06)
        self.assertEqual(self.attempts, attempts)
        self.assertFalse(call.state()["active"])
        self.assertFalse(call.state()["waiting_for_device"])

    def test_stop_during_blocked_open_closes_result_without_start_or_reopen(self):
        entered, release = threading.Event(), threading.Event()
        def factory(**kwargs):
            self.attempts += 1
            entered.set(); release.wait(1)
            stream = Stream(**kwargs); self.streams.append(stream)
            return stream
        call = self.make()
        call.stream_factory = factory
        call.start(); self.assertTrue(entered.wait(1))
        started = time.monotonic()
        self.assertTrue(call.stop())
        self.assertLess(time.monotonic()-started, .1)
        release.set()
        self.assertTrue(eventually(lambda: bool(self.streams) and self.streams[0].closed))
        self.assertFalse(self.streams[0].started)
        self.assertEqual(self.attempts, 1)

    def test_liveness_counts_callbacks_even_when_listening_is_suspended(self):
        call = self.make(can_listen=lambda: False, liveness_timeout=.09)
        call.start()
        self.assertTrue(eventually(lambda: call.state()["input_ready"]))
        call.suspend()
        first = self.streams[-1]
        for _ in range(8):
            first.feed(0, .02); time.sleep(.02)
        self.assertEqual(len(self.streams), 1)
        self.assertFalse(call.state()["listening"])
        self.assertTrue(eventually(lambda: len(self.streams) >= 2))
        self.assertTrue(first.closed)

    def test_inactive_stream_is_reopened(self):
        call = self.make()
        call.start(); self.ready(call)
        first = self.streams[-1]; first.active = False
        self.assertTrue(eventually(lambda: len(self.streams) >= 2 and call.state()["input_ready"]))
        self.assertTrue(first.closed)

    def test_repeated_retry_reports_friendly_waiting_once(self):
        self.available.clear()
        errors = []
        call = self.make(on_error=errors.append)
        call.start()
        self.assertTrue(eventually(lambda: self.attempts >= 4))
        self.assertEqual(errors, ["ヘッドセットの接続待ち。装着し直してください"])
        self.assertIn("WdmSyncIoctl", call.state()["device_detail"])

    def test_resolve_pins_same_physical_name_across_index_changes(self):
        sd = types.SimpleNamespace(InputStream=self.factory)
        metadata = {"index": 15, "name": "Headset (Nothing Ear (3a))", "hostapi": "Windows WASAPI", "rate": 16000}
        call = voice_call.VoiceCall(mock.Mock(), device=15, auto_reconnect=True,
                                   reconnect_interval=.01, liveness_timeout=5)
        self.calls.append(call)
        with mock.patch.dict(sys.modules, {"sounddevice": sd}), \
                mock.patch.object(voice_call, "resolve_input_device", return_value=(15, metadata)) as resolve:
            call.start(); self.ready(call)
            self.streams[-1].active = False
            self.assertTrue(eventually(lambda: len(self.streams) >= 2))
            self.assertEqual(resolve.call_args_list[0].args[0], 15)
            self.assertEqual(resolve.call_args_list[1].args[0], metadata["name"])
            self.assertEqual(call.state()["microphone"], metadata)
            call.stop()

    def test_late_old_callback_cannot_enqueue_audio_into_new_transport(self):
        call = self.make()
        call.start(); self.ready(call)
        old = self.streams[-1]
        copied, release = threading.Event(), threading.Event()
        class LateCopy:
            def copy(self):
                copied.set(); release.wait(1)
                return np.full((4800, 1), 3277, dtype=np.int16)
        copier = threading.Thread(target=lambda: old.callback(LateCopy(), 4800, None, None))
        copier.start(); self.assertTrue(copied.wait(1))
        old.active = False
        self.assertTrue(eventually(lambda: len(self.streams) >= 2 and call.state()["input_ready"]))
        release.set(); copier.join(1)
        self.streams[-1].feed(0, .8)
        time.sleep(.04)
        self.assertEqual(self.received, [])
        self.ready(call); self.utter(self.streams[-1])
        self.assertTrue(eventually(lambda: self.received == [True]))

    def test_manual_stop_returns_while_close_blocks_and_new_open_waits(self):
        closing, release = threading.Event(), threading.Event()
        call = self.make()
        call.start(); self.ready(call)
        old = self.streams[-1]
        def close():
            closing.set(); release.wait(1)
            old.closed = True
        old.close = close
        began = time.monotonic()
        self.assertTrue(call.stop())
        self.assertLess(time.monotonic()-began, .1)
        self.assertTrue(closing.wait(1))
        self.assertTrue(call.state()["input_closing"])
        self.assertTrue(call.start())
        time.sleep(.04)
        self.assertEqual(len(self.streams), 1)
        self.assertTrue(call.state()["waiting_for_device"])
        release.set(); self.ready(call)
        self.assertEqual(len(self.streams), 2)
        self.assertTrue(old.closed)

    def test_invalid_reconnect_configuration_rejected_before_open(self):
        for kwargs in ({"auto_reconnect": 1}, {"auto_reconnect": True},
                       {"reconnect_interval": 0}, {"liveness_timeout": float("nan")},
                       {"device_refresher": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                voice_call.VoiceCall(mock.Mock(), **kwargs)

    def test_device_refresh_occurs_after_failed_stream_is_closed_before_retry(self):
        order = []
        attempts = []
        class FailedStart(Stream):
            def start(self):
                order.append("start failed")
                raise OSError("original device failure")
            def stop(self):
                order.append("stop")
            def close(self):
                order.append("close")
                super().close()
        def factory(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                return FailedStart(**kwargs)
            order.append("next open")
            stream = Stream(**kwargs); self.streams.append(stream)
            return stream
        call = self.make(device_refresher=lambda: order.append("refresh"))
        call.stream_factory = factory
        call.start(); self.ready(call)
        self.assertEqual(order, ["start failed", "stop", "close", "refresh", "next open"])
        time.sleep(.03)
        self.assertEqual(order.count("refresh"), 1)
        call.stop()
        time.sleep(.03)
        self.assertEqual(order.count("refresh"), 1)

    def test_stop_during_failed_close_skips_refresh_and_reopen(self):
        closing, release = threading.Event(), threading.Event()
        refresher = mock.Mock()
        call = self.make(device_refresher=refresher)
        call.start(); self.ready(call)
        stream = self.streams[-1]
        def close():
            closing.set(); release.wait(1)
            stream.closed = True
        stream.close = close
        stream.active = False
        self.assertTrue(closing.wait(1))
        call.stop()
        release.set()
        time.sleep(.06)
        refresher.assert_not_called()
        self.assertEqual(len(self.streams), 1)

    def test_refresh_failure_keeps_waiting_and_retains_original_failure_detail(self):
        self.available.clear()
        refresher = mock.Mock(side_effect=RuntimeError("refresh unavailable"))
        call = self.make(device_refresher=refresher)
        call.start()
        self.assertTrue(eventually(lambda: refresher.call_count >= 2))
        state = call.state()
        self.assertTrue(state["active"])
        self.assertTrue(state["waiting_for_device"])
        self.assertIn("接続待ち", state["error"])
        self.assertIn("WdmSyncIoctl", state["device_detail"])
        self.assertIn("refresh unavailable", state["device_detail"])


if __name__ == "__main__":
    unittest.main()
