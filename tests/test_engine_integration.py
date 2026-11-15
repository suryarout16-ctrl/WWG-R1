"""パネルの音声・人格・教材連携を、実デバイスとネット接続なしで検証する。"""
import argparse
import contextlib
import email.message
import io
import json
import pathlib
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r1_chat"))
import web_ui as ui


class FakeCall:
    def __init__(self, callback, **kwargs):
        self.callback = callback
        self.active = False
        self.starts = 0
        self.stops = 0

    def state(self):
        return {"active": self.active, "listening": self.active, "processing": False,
                "level": 0, "threshold": .012, "error": ""}

    def start(self):
        self.starts += 1
        self.active = True
        return True

    def stop(self):
        self.stops += 1
        self.active = False


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="r1-engine-test-")
        self.addCleanup(self.folder.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(ui, "log"))
        self.stack.enter_context(mock.patch.object(ui.rc, "log_turn"))
        args = argparse.Namespace(
            motion=True, materials_file=str(pathlib.Path(self.folder.name) / "materials.json"),
            voice="test-voice", rate=0, local=True, interface=None, volume=70, gain=0,
            hat_action="hiyari_hat_v1", hat_fallback=False,
            strata_url="http://no-network.invalid/v1", strata_model="test-local",
            z840_host="", z840_user="", z840_key_dir="", z840_hostkey_alias="",
            persona="mentor", allow_persona_switch=False, model="test-unused", reasoning_effort="minimal",
            no_stt=False, stt_python=sys.executable, stt_model="small", stt_device="cpu",
            stt_compute_type="int8", cuda_dll_dir="", mic_device=None,
            call_threshold=.012, call_silence=.8,
        )
        self.tts = types.SimpleNamespace(synth=mock.Mock(return_value=b"\0\0" * 100), proc=None)
        self.speaker = types.SimpleNamespace(play=mock.Mock())
        self.stt = types.SimpleNamespace(transcribe=mock.Mock(return_value="手順を教えて"), close=mock.Mock())
        self.gesturer = mock.Mock()
        self.gesturer.start.return_value = "sent"
        self.gesturer.finish.return_value = 0
        self.gesturer.manual.return_value = 0
        self.gesturer.perform.return_value = 0
        # Thread.startも抑え、接続確認のバックグラウンド処理を実行しない。
        with contextlib.ExitStack() as constructors:
            constructors.enter_context(mock.patch.object(ui.rc, "SapiTTS", return_value=self.tts))
            constructors.enter_context(mock.patch.object(ui.rc, "LocalSpeaker", return_value=self.speaker))
            constructors.enter_context(mock.patch.object(ui.rc, "Gesturer", return_value=self.gesturer))
            constructors.enter_context(mock.patch.object(ui.rc, "WhisperSTT", return_value=self.stt))
            constructors.enter_context(mock.patch.object(ui, "VoiceCall", FakeCall))
            constructors.enter_context(mock.patch.object(ui.threading, "Thread"))
            self.secret = constructors.enter_context(mock.patch("credentials.get_secret",
                                                               side_effect=AssertionError("資格情報を読まない")))
            self.openai = mock.Mock()
            constructors.enter_context(mock.patch.dict(sys.modules, {"openai": self.openai}))
            self.engine = ui.Engine(args)
        self.addCleanup(self.engine.close)

    def test_default_mentor_needs_no_openai_key_or_client(self):
        self.assertEqual(self.engine.persona, "mentor")
        self.assertIsNone(self.engine.client)
        self.assertEqual(set(self.engine.brains), {"mentor"})
        self.secret.assert_not_called()
        self.openai.OpenAI.assert_not_called()
        self.assertTrue(self.engine.state()["persona_locked"])
        self.assertFalse(self.engine.call.active)
        self.assertEqual(self.engine.call.starts, 0)
        self.assertEqual(self.engine.body.forward_speed, .1)
        self.assertEqual(self.engine.forward_duration, 1.0)

    def test_forward_settings_reject_invalid_values_before_device_initialization(self):
        for name, value in (("forward_speed", .30001), ("forward_speed", 0), ("forward_speed", True),
                            ("forward_speed", float("nan")), ("forward_duration", 2.01),
                            ("forward_duration", -1), ("forward_duration", float("inf"))):
            with self.subTest(name=name, value=value):
                args = argparse.Namespace(**vars(self.engine.args), **{name: value})
                with mock.patch.object(ui.rc, "SapiTTS") as tts:
                    with self.assertRaises(ValueError):
                        ui.Engine(args)
                    tts.assert_not_called()

    def test_explicit_forward_settings_reach_controller_and_state(self):
        args = argparse.Namespace(**vars(self.engine.args), forward_speed=.3, forward_duration=2.0)
        with contextlib.ExitStack() as constructors:
            constructors.enter_context(mock.patch.object(ui.rc, "SapiTTS", return_value=self.tts))
            constructors.enter_context(mock.patch.object(ui.rc, "LocalSpeaker", return_value=self.speaker))
            constructors.enter_context(mock.patch.object(ui.rc, "Gesturer", return_value=self.gesturer))
            constructors.enter_context(mock.patch.object(ui.rc, "WhisperSTT", return_value=self.stt))
            constructors.enter_context(mock.patch.object(ui, "VoiceCall", FakeCall))
            constructors.enter_context(mock.patch.object(ui.threading, "Thread"))
            configured = ui.Engine(args)
        try:
            self.assertEqual(configured.body.forward_speed, .3)
            self.assertEqual(configured.body.linear_speed, .1)
            self.assertEqual(configured.body.yaw_speed, .2)
            self.assertEqual(configured.state()["body"]["forward_duration"], 2.0)
            self.assertEqual(configured._body_duration("move_forward"), 2.0)
            self.assertEqual(configured._body_duration("move_backward"), 1.0)
        finally:
            configured.close()

    def test_reset_keeps_fixed_persona_and_training_documents(self):
        self.assertTrue(self.engine.material_add("模擬作業指示書", "工具Aは使用前に点検する。")[0])
        brain = self.engine.brain
        context = brain.training_context
        instructions = brain.instructions
        brain.history = [{"role": "user", "content": "前の会話"}]
        self.assertTrue(self.engine.reset()[0])
        self.assertEqual(brain.history, [])
        self.assertEqual(brain.instructions, instructions)
        self.assertEqual(brain.training_context, context)
        self.assertEqual(self.engine.persona, "mentor")
        self.assertEqual(self.engine.materials.documents()[0]["name"], "模擬作業指示書")

    def test_fixed_persona_rejects_openai_and_accepts_legacy_mentor_alias(self):
        with mock.patch.object(self.engine, "_openai_brain", side_effect=AssertionError("OpenAI禁止")):
            self.assertFalse(self.engine.set_persona("r1")[0])
            self.assertTrue(self.engine.set_persona("omoshiro")[0])
        self.assertEqual(self.engine.persona, "mentor")

    def test_call_and_busy_job_block_persona_reset_and_material_changes(self):
        self.engine.material_add("台本", "模擬研修を始める。")
        document_id = self.engine.materials.documents()[0]["id"]
        original = self.engine.materials.context()
        self.engine.args.allow_persona_switch = True
        with mock.patch.object(self.engine, "_openai_brain", side_effect=AssertionError("OpenAI禁止")):
            for mode in ("call", "job"):
                with self.subTest(mode=mode):
                    self.engine.call.active = mode == "call"
                    if mode == "job":
                        self.engine.job_lock.acquire()
                    try:
                        self.assertFalse(self.engine.material_add("新資料", "変更")[0])
                        self.assertFalse(self.engine.material_remove(document_id)[0])
                        self.assertFalse(self.engine.set_persona("r1")[0])
                        self.assertFalse(self.engine.reset()[0])
                    finally:
                        self.engine.call.active = False
                        if mode == "job":
                            self.engine.job_lock.release()
        self.assertEqual(self.engine.materials.context(), original)
        self.assertEqual(self.engine.persona, "mentor")

    def test_call_cancelled_before_stt_does_no_work(self):
        self.engine.brain.reply = mock.Mock()
        self.engine._call_utterance(np.zeros(16000, dtype=np.int16), lambda: False)
        self.stt.transcribe.assert_not_called()
        self.engine.brain.reply.assert_not_called()
        self.tts.synth.assert_not_called()
        self.gesturer.start.assert_not_called()
        self.assertFalse(self.engine.job_lock.locked())

    def test_motion_speech_override_keeps_model_evidence_in_turn_log(self):
        original = "周囲の安全は俺が確認しているから、安心して見ていてくれ。"
        self.engine.brain._stream = mock.Mock(return_value=json.dumps(
            {"say": original, "gesture": "move_forward"}, ensure_ascii=False))
        with mock.patch.object(self.engine, "_speak") as speak:
            self.engine._chat("前へ少し進んでください。", "文字")
        spoken = speak.call_args.args[0]
        self.assertNotEqual(spoken, original)
        self.assertIn("実行できるか", spoken)
        evidence = ui.rc.log_turn.call_args.kwargs["model_reply"]
        self.assertEqual(evidence["model_say"], original)
        self.assertEqual(evidence["say"], spoken)
        self.assertEqual(evidence["gesture"], "move_forward")
        self.assertEqual(self.engine.history[-1]["text"], spoken)

    def test_call_cancelled_during_stt_does_not_generate_or_move(self):
        alive = threading.Event()
        alive.set()

        def transcribe(_):
            alive.clear()
            return "手順を教えて"

        self.stt.transcribe.side_effect = transcribe
        self.engine.brain.reply = mock.Mock()
        self.engine._call_utterance(np.zeros(16000, dtype=np.int16), alive.is_set)
        self.engine.brain.reply.assert_not_called()
        self.tts.synth.assert_not_called()
        self.gesturer.start.assert_not_called()
        self.assertFalse(self.engine.job_lock.locked())
        self.assertEqual(self.engine.busy, "")

    def test_call_cancelled_during_llm_does_not_speak_or_move(self):
        alive = threading.Event()
        alive.set()

        def reply(_):
            alive.clear()
            return "まず工具を確認しよう。", "emphasize"

        self.engine.brain.reply = mock.Mock(side_effect=reply)
        self.engine._call_utterance(np.zeros(16000, dtype=np.int16), alive.is_set)
        self.engine.brain.reply.assert_called_once()
        self.tts.synth.assert_not_called()
        self.gesturer.start.assert_not_called()
        self.speaker.play.assert_not_called()
        self.assertFalse(self.engine.job_lock.locked())

    def test_call_cancelled_during_tts_does_not_start_gesture_or_playback(self):
        alive = threading.Event()
        alive.set()
        self.engine.brain.reply = mock.Mock(return_value=("確認してくれて助かったよ。", "clap"))

        def synth(_):
            alive.clear()
            return b"\0\0" * 100

        self.tts.synth.side_effect = synth
        self.engine._call_utterance(np.zeros(16000, dtype=np.int16), alive.is_set)
        self.tts.synth.assert_called_once()
        self.gesturer.start.assert_not_called()
        self.speaker.play.assert_not_called()

    def test_shutdown_cancels_text_jobs_without_a_call_predicate(self):
        self.engine.closed = True
        self.engine.brain.reply = mock.Mock()
        self.engine._chat("続けて", "文字")
        self.engine._speak("停止後の返事", "wave")
        self.engine.brain.reply.assert_not_called()
        self.tts.synth.assert_not_called()
        self.gesturer.start.assert_not_called()

    def test_failed_manual_mic_start_closes_stream(self):
        stream = mock.Mock()
        stream.start.side_effect = OSError("マイク開始エラー")
        sounddevice = types.SimpleNamespace(InputStream=mock.Mock(return_value=stream))
        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            with self.assertRaises(OSError):
                self.engine.record_start()
        stream.close.assert_called_once()
        self.assertIsNone(self.engine.rec_stream)

    def test_voice_job_excludes_manual_gesture_and_text_job(self):
        entered, release = threading.Event(), threading.Event()

        def listen(*_, **__):
            entered.set()
            release.wait(2)

        with mock.patch.object(self.engine, "_listen", side_effect=listen):
            worker = threading.Thread(target=self.engine._call_utterance,
                                      args=(np.zeros(10, dtype=np.int16), lambda: True))
            worker.start()
            self.assertTrue(entered.wait(2))
            try:
                self.assertFalse(self.engine.gesture("wave")[0])
                self.assertFalse(self.engine.chat("割り込み")[0])
                self.gesturer.manual.assert_not_called()
            finally:
                release.set()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.engine.job_lock.locked())

    def test_manual_job_excludes_a_racing_voice_utterance(self):
        entered, release = threading.Event(), threading.Event()

        def manual_job():
            entered.set()
            release.wait(2)

        self.assertTrue(self.engine.submit("手動操作中", manual_job)[0])
        self.assertTrue(entered.wait(2))
        try:
            self.engine._call_utterance(np.zeros(10, dtype=np.int16), lambda: True)
            self.stt.transcribe.assert_not_called()
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while self.engine.job_lock.locked() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertFalse(self.engine.job_lock.locked())

    def test_manual_recording_waits_for_call_device_to_close(self):
        entered, release, recording_done = threading.Event(), threading.Event(), threading.Event()
        self.engine.call.active = True

        def stop():
            self.engine.call.active = False
            entered.set()
            release.wait(2)

        stream = mock.Mock()
        sounddevice = types.SimpleNamespace(InputStream=mock.Mock(return_value=stream))
        self.engine.call.stop = stop
        stopper = threading.Thread(target=self.engine.call_stop)
        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            stopper.start()
            self.assertTrue(entered.wait(2))
            recorder = threading.Thread(target=lambda: (self.engine.record_start(), recording_done.set()))
            recorder.start()
            try:
                self.assertFalse(recording_done.wait(.05))
                sounddevice.InputStream.assert_not_called()
            finally:
                release.set()
                stopper.join(2)
                recorder.join(2)
        self.assertTrue(recording_done.is_set())
        stream.start.assert_called_once()


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.engine = mock.Mock(spec=ui.Engine)
        for method in ("chat", "say", "greet", "gesture", "record_start", "record_stop", "reset",
                       "set_motion", "set_persona", "call_start", "call_stop", "material_add", "material_remove"):
            getattr(self.engine, method).return_value = (True, "ok")
        self.handler_class = ui.make_handler(self.engine)

    def post(self, path, body=None, *, host="127.0.0.1:8765", origin="http://127.0.0.1:8765",
             content_type="application/json", declared_length=None, raw=None):
        payload = raw if raw is not None else json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
        handler = self.handler_class.__new__(self.handler_class)
        handler.server = types.SimpleNamespace(server_port=8765)
        handler.path = path
        handler.headers = email.message.Message()
        handler.headers["Host"] = host
        if origin is not None:
            handler.headers["Origin"] = origin
        handler.headers["Content-Type"] = content_type
        handler.headers["Content-Length"] = str(len(payload) if declared_length is None else declared_length)
        handler.rfile = io.BytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.do_POST()
        return handler.send_response.call_args.args[0], json.loads(handler.wfile.getvalue()), handler

    def test_call_routes_dispatch_without_starting_real_devices(self):
        for path, method in (("/api/call/start", "call_start"), ("/api/call/stop", "call_stop")):
            status, data, _ = self.post(path)
            self.assertEqual(status, 200)
            self.assertTrue(data["ok"])
            getattr(self.engine, method).assert_called_once_with()

    def test_material_route_preserves_full_document_not_chat_limit(self):
        text = "研修本文" * 1000
        status, _, _ = self.post("/api/materials/add", {"name": "台本", "text": text})
        self.assertEqual(status, 200)
        self.engine.material_add.assert_called_once_with("台本", text)

    def test_foreign_host_or_origin_cannot_trigger_operations(self):
        for headers in ({"host": "foreign.invalid:8765"}, {"origin": "https://foreign.invalid"}):
            with self.subTest(headers=headers):
                status, _, handler = self.post("/api/call/start", **headers)
                self.assertEqual(status, 403)
                self.assertEqual(handler.rfile.tell(), 0)
        self.engine.call_start.assert_not_called()

    def test_non_json_request_is_rejected_before_reading_body(self):
        status, _, handler = self.post("/api/chat", {"text": "hello"}, content_type="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(handler.rfile.tell(), 0)
        self.engine.chat.assert_not_called()

    def test_payload_cap_rejects_empty_negative_and_excessive_lengths(self):
        for length in (0, -1, 131073):
            with self.subTest(length=length):
                status, _, handler = self.post("/api/materials/add", declared_length=length)
                self.assertEqual(status, 413)
                self.assertEqual(handler.rfile.tell(), 0)
        self.engine.material_add.assert_not_called()

    def test_non_object_or_malformed_json_is_rejected(self):
        for raw in (b"[]", b"not json", b"\xff"):
            with self.subTest(raw=raw):
                status, _, _ = self.post("/api/chat", raw=raw)
                self.assertEqual(status, 400)
        self.engine.chat.assert_not_called()

    def test_chat_text_cap_and_missing_route(self):
        status, _, _ = self.post("/api/chat", {"text": "a" * 400})
        self.assertEqual(status, 200)
        self.engine.chat.assert_called_once_with("a" * 300)
        status, _, _ = self.post("/api/unknown")
        self.assertEqual(status, 404)

    def test_rejected_engine_operation_returns_conflict(self):
        self.engine.call_start.return_value = (False, "処理中です")
        status, data, _ = self.post("/api/call/start")
        self.assertEqual(status, 409)
        self.assertEqual(data["message"], "処理中です")


if __name__ == "__main__":
    unittest.main()
