"""通話・単発録音の入力選択を、実際の音声デバイスなしで検証する。"""
import sys
import types
import unittest
from unittest import mock

import test_engine_integration as fixtures
import voice_call


def audio_stub():
    stream = types.SimpleNamespace(start=mock.Mock(), stop=mock.Mock(), close=mock.Mock())
    sd = types.SimpleNamespace(InputStream=mock.Mock(return_value=stream))
    return sd, stream


class CallInputSelectionTests(unittest.TestCase):
    def test_explicit_device_uses_resolved_index_and_reports_selected_microphone(self):
        sd, stream = audio_stub()
        metadata = {"index": 21, "name": "Headset (Nothing Ear (3a))", "hostapi": "Windows WASAPI"}
        call = voice_call.VoiceCall(mock.Mock(), device="Nothing Ear (3a)")
        self.addCleanup(call.stop)
        with mock.patch.dict(sys.modules, {"sounddevice": sd}), \
                mock.patch.object(voice_call, "resolve_input_device", return_value=(21, metadata)) as resolve:
            self.assertTrue(call.start())
        resolve.assert_called_once_with("Nothing Ear (3a)", sd)
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], 21)
        stream.start.assert_called_once()
        self.assertEqual(call.state()["microphone"], metadata)
        exposed = call.state()["microphone"]
        exposed["index"] = -1
        self.assertEqual(call.state()["microphone"]["index"], 21)
        self.assertTrue(call.stop())
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_default_device_and_injected_factory_skip_selection(self):
        for injected in (False, True):
            with self.subTest(injected=injected):
                sd, stream = audio_stub()
                factory = mock.Mock(return_value=stream) if injected else None
                requested = "legacy fake device" if injected else None
                call = voice_call.VoiceCall(mock.Mock(), device=requested, stream_factory=factory)
                self.addCleanup(call.stop)
                with mock.patch.dict(sys.modules, {"sounddevice": sd}), \
                        mock.patch.object(voice_call, "resolve_input_device", side_effect=AssertionError("must skip")) as resolve:
                    self.assertTrue(call.start())
                resolve.assert_not_called()
                used_factory = factory if injected else sd.InputStream
                self.assertEqual(used_factory.call_args.kwargs["device"], requested)
                self.assertEqual(call.state()["microphone"], {})
                if injected:
                    sd.InputStream.assert_not_called()
                call.stop()

    def test_missing_requested_microphone_fails_without_default_fallback(self):
        sd, _ = audio_stub()
        call = voice_call.VoiceCall(mock.Mock(), device="Nothing Ear (3a)")
        self.addCleanup(call.stop)
        with mock.patch.dict(sys.modules, {"sounddevice": sd}), \
                mock.patch.object(voice_call, "resolve_input_device", side_effect=ValueError("requested microphone unavailable")):
            with self.assertRaisesRegex(ValueError, "requested microphone unavailable"):
                call.start()
        sd.InputStream.assert_not_called()
        self.assertFalse(call.state()["active"])
        self.assertIn("requested microphone unavailable", call.state()["error"])


class RecorderInputSelectionTests(unittest.TestCase):
    def setUp(self):
        fixtures.EngineTests.setUp(self)
        self.engine.args.mic_device = "Nothing Ear (3a)"

    def test_recorder_uses_resolved_input_instead_of_matching_wrong_endpoint(self):
        sd, stream = audio_stub()
        metadata = {"index": 24, "name": "Headset (Nothing Ear (3a))"}
        with mock.patch.dict(sys.modules, {"sounddevice": sd}), \
                mock.patch.object(fixtures.ui, "resolve_input_device", return_value=(24, metadata)) as resolve:
            self.assertTrue(self.engine.record_start()[0])
        resolve.assert_called_once_with("Nothing Ear (3a)", sd)
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], 24)
        self.assertIs(self.engine.rec_stream, stream)
        stream.start.assert_called_once()

    def test_recorder_selection_failure_does_not_open_default_input(self):
        sd, _ = audio_stub()
        with mock.patch.dict(sys.modules, {"sounddevice": sd}), \
                mock.patch.object(fixtures.ui, "resolve_input_device", side_effect=ValueError("requested microphone unavailable")):
            with self.assertRaisesRegex(ValueError, "requested microphone unavailable"):
                self.engine.record_start()
        sd.InputStream.assert_not_called()
        self.assertIsNone(self.engine.rec_stream)


if __name__ == "__main__":
    unittest.main()
