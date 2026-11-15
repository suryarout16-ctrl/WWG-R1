"""入力の列挙結果だけを使い、マイクを開かずに同じ機器のWASAPIを選ぶ。"""
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_chat"))
from audio_input import resolve_input_device, refresh_audio_devices


NAME = "ヘッドセット (Nothing Ear (3a))"
RAW = "ヘッドセット (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0\r\n;(Nothing Ear (3a)))"


class Devices:
    def __init__(self):
        self.hosts = [{"name": name} for name in ("MME", "Windows DirectSound", "Windows WASAPI", "Windows WDM-KS")]
        self.devices = [
            {"index": 2, "name": NAME, "hostapi": 0, "max_input_channels": 1, "default_samplerate": 44100},
            {"index": 8, "name": NAME, "hostapi": 1, "max_input_channels": 1, "default_samplerate": 44100},
            {"index": 15, "name": NAME, "hostapi": 2, "max_input_channels": 1, "default_samplerate": 16000},
            {"index": 31, "name": RAW, "hostapi": 3, "max_input_channels": 1, "default_samplerate": 16000},
            {"index": 1, "name": "Microphone (Realtek)", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
            {"index": 19, "name": "Headphones (Nothing Ear (3a))", "hostapi": 2, "max_input_channels": 0, "default_samplerate": 48000},
        ]
        self.default = 1
        self.InputStream = mock.Mock(side_effect=AssertionError("ストリームを開かない"))

    def query_devices(self, **kwargs):
        if kwargs:
            return next(row for row in self.devices if row["index"] == self.default)
        return self.devices

    def query_hostapis(self):
        return self.hosts


class AudioInputTests(unittest.TestCase):
    def setUp(self):
        self.sd = Devices()
        self.platform = mock.patch("audio_input.sys.platform", "win32")
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def resolve(self, request):
        return resolve_input_device(request, self.sd)

    def test_stable_name_selects_same_device_wasapi_metadata(self):
        index, info = self.resolve("Nothing Ear (3a)")
        self.assertEqual(index, 15)
        self.assertEqual(info, {"index": 15, "name": NAME, "hostapi": "Windows WASAPI", "rate": 16000})
        self.sd.InputStream.assert_not_called()

    def test_numeric_mme_and_directsound_select_same_friendly_name_wasapi(self):
        for index in (2, 8, "8", 15):
            with self.subTest(index=index):
                self.assertEqual(self.resolve(index)[0], 15)

    def test_bluetooth_resource_name_maps_to_same_model_not_another_mic(self):
        self.assertEqual(self.resolve(31)[0], 15)
        self.assertEqual(self.resolve(RAW)[0], 15)
        self.assertEqual(self.resolve(NAME)[0], 15)

    def test_default_keeps_its_physical_mic_instead_of_picking_headset(self):
        index, info = self.resolve(None)
        self.assertEqual(index, 1)
        self.assertEqual(info["name"], "Microphone (Realtek)")
        self.sd.default = 31
        self.assertEqual(self.resolve(None)[0], 15)

    def test_no_matching_wasapi_keeps_explicit_input_backend(self):
        self.sd.devices = [row for row in self.sd.devices if row["index"] != 15]
        self.assertEqual(self.resolve(31)[0], 31)
        self.assertEqual(self.resolve(2)[0], 2)

    def test_name_without_wasapi_prefers_directsound_for_the_same_device(self):
        self.sd.devices = [row for row in self.sd.devices if row["index"] != 15]
        self.assertEqual(self.resolve("Nothing Ear (3a)")[0], 8)

    def test_ambiguous_different_physical_names_are_rejected(self):
        self.sd.devices.append({"index": 22, "name": "ヘッドセット (Nothing Ear Other)", "hostapi": 2,
                                "max_input_channels": 1, "default_samplerate": 16000})
        with self.assertRaises(ValueError):
            self.resolve("Nothing Ear")
        self.assertEqual(self.resolve("Nothing Ear (3a)")[0], 15)

    def test_duplicate_wasapi_friendly_names_require_explicit_number(self):
        self.sd.devices.append({**self.sd.devices[2], "index": 23})
        for value in ("Nothing Ear (3a)", 31):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)
        self.assertEqual(self.resolve(15)[0], 15)

    def test_missing_device_and_output_only_are_rejected(self):
        for value in (404, "Missing microphone", 19, "Headphones"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)

    def test_invalid_inputs_do_not_even_enumerate(self):
        self.sd.query_devices = mock.Mock(side_effect=AssertionError("列挙しない"))
        for value in (True, [], {}, -1, ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)

    def test_non_windows_numeric_selection_does_not_rewrite_host_backend(self):
        with mock.patch("audio_input.sys.platform", "linux"):
            self.assertEqual(self.resolve(31)[0], 31)

    def test_absent_default_input_is_rejected(self):
        self.sd.query_devices = mock.Mock(side_effect=[self.sd.devices, {"index": -1}])
        with self.assertRaises(ValueError):
            self.resolve(None)


class AudioRefreshTests(unittest.TestCase):
    def module(self, count=1):
        module = types.SimpleNamespace(_initialized=count, default=object())
        module.InputStream = mock.Mock(side_effect=AssertionError("入力を開かない"))
        module.OutputStream = mock.Mock(side_effect=AssertionError("出力を開かない"))
        module._terminate = mock.Mock(side_effect=lambda: setattr(module, "_initialized", module._initialized - 1))
        module._initialize = mock.Mock(side_effect=lambda: setattr(module, "_initialized", module._initialized + 1))
        return module

    def test_refresh_balances_portaudio_and_preserves_default_without_opening_streams(self):
        module = self.module()
        original_default = module.default
        refresh_audio_devices(module)
        self.assertEqual(module._initialized, 1)
        module._terminate.assert_called_once_with()
        module._initialize.assert_called_once_with()
        self.assertIs(module.default, original_default)
        module.InputStream.assert_not_called()
        module.OutputStream.assert_not_called()

    def test_uninitialized_module_is_recovered_without_an_extra_terminate(self):
        module = self.module(0)
        refresh_audio_devices(module)
        self.assertEqual(module._initialized, 1)
        module._terminate.assert_not_called()
        module._initialize.assert_called_once_with()

    def test_multiple_or_unknown_references_are_rejected_without_mutation(self):
        for count in (None, False, True, -1, 2, 3, "1"):
            with self.subTest(count=count):
                module = self.module(count)
                with self.assertRaises(RuntimeError):
                    refresh_audio_devices(module)
                self.assertEqual(module._initialized, count)
                module._terminate.assert_not_called()
                module._initialize.assert_not_called()

    def test_unsupported_library_is_rejected_without_any_device_call(self):
        for missing in ("_initialize", "_terminate"):
            with self.subTest(missing=missing):
                module = self.module()
                setattr(module, missing, None)
                with self.assertRaises(RuntimeError):
                    refresh_audio_devices(module)
                module.InputStream.assert_not_called()
                module.OutputStream.assert_not_called()

    def test_active_convenience_stream_prevents_refresh(self):
        module = self.module()
        module._last_callback = types.SimpleNamespace(stream=types.SimpleNamespace(active=True))
        with self.assertRaises(RuntimeError):
            refresh_audio_devices(module)
        module._terminate.assert_not_called()
        module._initialize.assert_not_called()

    def test_termination_failure_does_not_add_another_initialization_reference(self):
        module = self.module()
        module._terminate.side_effect = RuntimeError("driver failure")
        with self.assertRaises(RuntimeError):
            refresh_audio_devices(module)
        self.assertEqual(module._initialized, 1)
        module._initialize.assert_not_called()

    def test_initialization_failure_attempts_one_recovery_and_preserves_the_error(self):
        module = self.module()
        calls = []

        def initialize():
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("temporary driver failure")
            module._initialized += 1

        module._initialize.side_effect = initialize
        with self.assertRaises(RuntimeError):
            refresh_audio_devices(module)
        self.assertEqual(module._initialized, 1)
        self.assertEqual(module._initialize.call_count, 2)
        module.InputStream.assert_not_called()

    def test_persistent_initialization_failure_stays_zero_and_can_recover_next_time(self):
        module = self.module()
        module._initialize.side_effect = RuntimeError("not available")
        with self.assertRaises(RuntimeError):
            refresh_audio_devices(module)
        self.assertEqual(module._initialized, 0)
        self.assertEqual(module._initialize.call_count, 2)
        module._initialize.side_effect = lambda: setattr(module, "_initialized", module._initialized + 1)
        refresh_audio_devices(module)
        self.assertEqual(module._initialized, 1)
        self.assertEqual(module._terminate.call_count, 1)

    def test_refreshes_are_serialized_within_the_process(self):
        module = self.module()
        entered, release = threading.Event(), threading.Event()
        sequence = []

        def terminate():
            sequence.append("terminate")
            module._initialized -= 1
            if len(sequence) == 1:
                entered.set()
                release.wait(2)

        def initialize():
            sequence.append("initialize")
            module._initialized += 1

        module._terminate.side_effect, module._initialize.side_effect = terminate, initialize
        errors = []

        def refresh():
            try:
                refresh_audio_devices(module)
            except Exception as error:
                errors.append(error)

        first = threading.Thread(target=refresh)
        second = threading.Thread(target=refresh)
        first.start()
        self.assertTrue(entered.wait(1))
        second.start()
        try:
            self.assertEqual(sequence, ["terminate"])
        finally:
            release.set()
            first.join(2)
            second.join(2)
        self.assertFalse(errors)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(sequence, ["terminate", "initialize", "terminate", "initialize"])
        self.assertEqual(module._initialized, 1)


if __name__ == "__main__":
    unittest.main()
