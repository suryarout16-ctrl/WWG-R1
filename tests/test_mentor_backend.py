"""API接続・資格情報・ロボットは偽物だけで、人格を保つ推論先切替を検証。"""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_chat"))
import personas
from mentor_backend import OpenAIMentorBrain, MentorAPIError
import test_engine_integration as engine_tests


class FakeStream:
    def __init__(self, text):
        self.closed = False
        self.chunks = [types.SimpleNamespace(choices=[
            types.SimpleNamespace(delta=types.SimpleNamespace(content=part))])
            for part in (text[:5], text[5:])]

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class OpenAIMentorTests(unittest.TestCase):
    def brain(self, text, model="gpt-5-mini"):
        stream = FakeStream(text)
        client = Mock()
        client.chat.completions.create.return_value = stream
        return OpenAIMentorBrain(client, model), client, stream

    def test_chat_arguments_and_same_mentor_materials_memory_and_guard(self):
        brain, client, stream = self.brain(json.dumps(
            {"say": "俺は販売員だ。", "gesture": "wave"}, ensure_ascii=False))
        brain.set_training_context("模擬機械：Aは移動、Bは停止、Cは誤操作。")
        brain.set_runtime_context("ローカルプレビュー。人や障害物の自動検知はない。")
        brain.set_body_enabled(True)
        brain.fixed_lines = True   # 固定自己紹介の回帰試験として確認する
        brain.history = [{"role": "user", "content": "前の発言"},
                         {"role": "assistant", "content": '{"say":"覚えたぞ。","gesture":"none"}'}]
        with patch.object(personas.rc, "log"):
            self.assertEqual(brain.reply("自己紹介して"), (personas.SELF_INTRODUCTION, "wave"))
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5-mini")
        self.assertEqual(kwargs["reasoning_effort"], "minimal")
        self.assertGreaterEqual(kwargs["max_completion_tokens"], 1000)
        self.assertFalse(kwargs["store"])
        for name in ("chat_template_kwargs", "temperature", "max_tokens", "api_key"):
            self.assertNotIn(name, kwargs)
        self.assertIn("熟練工", kwargs["messages"][0]["content"])
        self.assertIn("ローカルプレビュー", kwargs["messages"][0]["content"])
        self.assertIn("Cは誤操作", kwargs["messages"][1]["content"])
        self.assertIn({"role": "user", "content": "前の発言"}, kwargs["messages"])
        self.assertEqual(kwargs["response_format"]["json_schema"]["strict"], True)
        self.assertTrue(stream.closed)
        self.assertEqual(brain.last_reply["model_say"], "俺は販売員だ。")

    def test_body_candidate_still_gets_fact_checked_announcement(self):
        brain, _, _ = self.brain('{"say":"安全を確認して移動したぞ。","gesture":"move_forward"}')
        brain.set_body_enabled(True)
        with patch.object(personas.rc, "log"):
            say, gesture = brain.reply("前へ進んで")
        self.assertEqual(gesture, "move_forward")
        self.assertEqual(say, personas.BODY_ANNOUNCEMENTS[gesture])
        self.assertNotIn("安全を確認", say)

    def test_english_technical_turn_uses_measured_gpt_provider(self):
        brain, client, _ = self.brain('{"say":"The current AI is OpenAI GPT.","gesture":"emphasize"}')
        brain.set_runtime_context("PC内の仮想R1へ接続。現在の応答生成はGPT API。")
        brain.reply("How do you choose your words and gestures?")
        system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn(brain.BACKEND_LABEL, system)
        self.assertIn("モデル名 gpt-5-mini", system)
        self.assertIn("仮想R1", system)
        # Regression: the final English instruction previously forced Strata even on GPT.
        self.assertNotIn("the Strata AI on the Z840 research PC", system)
        self.assertIn("measured current-backend information", system)

    def test_non_reasoning_model_does_not_receive_reasoning_option(self):
        brain, client, _ = self.brain('{"say":"一緒に確認しよう。","gesture":"none"}', model="gpt-4.1-mini")
        brain.reply("確認したい")
        self.assertNotIn("reasoning_effort", client.chat.completions.create.call_args.kwargs)

    def test_error_does_not_leak_key_or_retain_failed_turn(self):
        brain, client, _ = self.brain("")
        error = RuntimeError("Incorrect API key: sk-dummy-must-not-appear")
        error.status_code = 429
        client.chat.completions.create.side_effect = error
        with self.assertRaises(MentorAPIError) as found:
            brain.reply("こんにちは")
        self.assertEqual(str(found.exception), "GPT APIへの問い合わせに失敗しました (HTTP 429)")
        self.assertEqual(brain.history, [])
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_model_availability_is_read_only_and_sanitizes_errors(self):
        brain, client, _ = self.brain("")
        self.assertEqual(brain.available(), (True, "GPT API gpt-5-mini"))
        client.models.retrieve.assert_called_once_with("gpt-5-mini", timeout=2.0)
        client.chat.completions.create.assert_not_called()
        client.models.retrieve.side_effect = RuntimeError("sk-dummy-secret")
        ok, detail = brain.available()
        self.assertFalse(ok)
        self.assertNotIn("sk-dummy", detail)


class MentorSwitchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = engine_tests.EngineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.engine = self.fixture.engine
        self.source = self.engine.brains["mentor"]
        self.target = OpenAIMentorBrain(Mock(), "gpt-5-mini")
        self.gpt_patch = patch.object(self.engine, "_gpt_mentor", return_value=self.target)
        self.gpt_patch.start()
        self.addCleanup(self.gpt_patch.stop)

    def test_switch_preserves_fixed_persona_and_copies_context_and_memory(self):
        self.engine.material_add("発表台本", "模擬機械のボタンAは移動、Bは停止。")
        self.source.set_runtime_context("FSM811。障害物検知なし。")
        self.source.history = [{"role": "user", "content": "Aの役割は？"},
                               {"role": "assistant", "content": '{"say":"移動だ。","gesture":"none"}'}]
        self.assertTrue(self.engine.set_mentor_backend("openai")[0])
        self.assertIs(self.engine.brain, self.target)
        self.assertEqual(self.engine.persona, "mentor")
        self.assertEqual(self.engine.brain.prompt, personas.MENTOR_PROMPT)
        self.assertEqual(self.target.history, self.source.history)
        self.assertIsNot(self.target.history, self.source.history)
        self.assertEqual(self.target.training_context, self.source.training_context)
        self.assertEqual(self.target.runtime_context, self.source.runtime_context)
        self.assertEqual(self.target.body_enabled, self.source.body_enabled)
        self.assertEqual(self.engine.state()["mentor_backend"], "openai")
        self.assertIn("GPT API", self.engine.state()["personas"][1]["label"])
        self.target.history.append({"role": "user", "content": "続き"})
        self.assertTrue(self.engine.set_mentor_backend("strata")[0])
        self.assertEqual(self.engine.brain.history[-1]["content"], "続き")

    def test_reset_and_material_change_do_not_resurrect_old_provider_history(self):
        self.source.history = [{"role": "user", "content": "古い資料の内容"}]
        self.engine.set_mentor_backend("openai")
        self.engine.material_add("新版台本", "Cを押したのは模擬設定。")
        self.engine.set_mentor_backend("strata")
        self.assertEqual(self.source.history, [])
        self.assertIn("新版台本", self.source.training_context)
        self.source.history = [{"role": "user", "content": "リセット前"}]
        self.engine.reset()
        self.engine.set_mentor_backend("openai")
        self.assertEqual(self.target.history, [])

    def test_switch_rejected_while_call_recording_or_work_is_active(self):
        for condition in ("call", "recording", "job"):
            with self.subTest(condition=condition):
                self.engine.call.active = condition == "call"
                self.engine.rec_stream = object() if condition == "recording" else None
                if condition == "job":
                    self.engine.job_lock.acquire()
                try:
                    self.assertFalse(self.engine.set_mentor_backend("openai")[0])
                    self.assertIs(self.engine.brain, self.source)
                finally:
                    self.engine.call.active = False
                    self.engine.rec_stream = None
                    if condition == "job":
                        self.engine.job_lock.release()

    def test_failed_gpt_setup_keeps_strata_and_no_secrets_in_message(self):
        self.engine._gpt_mentor.side_effect = RuntimeError("sk-dummy-private")
        ok, message = self.engine.set_mentor_backend("openai")
        self.assertFalse(ok)
        self.assertNotIn("sk-dummy", message)
        self.assertIs(self.engine.brain, self.source)
        self.assertEqual(self.engine.mentor_backend, "strata")

    def test_old_strata_poll_cannot_overwrite_new_provider_status(self):
        def old_available():
            self.engine.set_mentor_backend("openai")
            self.engine.closed = True
            return True, "old Strata result"
        self.source.available = Mock(side_effect=old_available)
        with patch.object(engine_tests.ui.time, "sleep"):
            self.engine._poll_personas()
        self.assertEqual(self.engine.persona_status["mentor"], (False, "接続確認中"))
        # Restore cleanup behavior after using closed as a finite polling-loop sentinel.
        self.engine.closed = False

    def test_chat_runtime_and_turn_log_identify_actual_provider(self):
        self.target.reply = Mock(return_value=("仕組みを説明するぞ。", "none"))
        self.engine.set_mentor_backend("openai")
        with patch.object(self.engine, "_speak"):
            self.engine._chat("どの仕組みで返事をする？", "文字")
        self.assertIn("現在の応答生成はGPT API", self.target.runtime_context)
        turn = engine_tests.ui.rc.log_turn.call_args.kwargs
        self.assertEqual(turn["backend"], "openai")
        self.assertEqual(turn["model"], "gpt-5-mini")
        self.assertIn("GPT API", self.engine.history[-1]["persona"])
        self.source.reply = Mock(return_value=("仕組みを説明するぞ。", "none"))
        self.engine.set_mentor_backend("strata")
        with patch.object(self.engine, "_speak"):
            self.engine._chat("どの仕組みで返事をする？", "文字")
        self.assertIn("現在の応答生成はZ840のStrata", self.source.runtime_context)

    def test_lazy_gpt_client_uses_existing_secret_source_and_no_completion_call(self):
        self.gpt_patch.stop()
        client = Mock()
        module = types.SimpleNamespace(OpenAI=Mock(return_value=client))
        with patch.dict("os.environ", {}, clear=True), patch("credentials.get_secret", return_value="fake-test-key") as secret, \
                patch.dict(sys.modules, {"openai": module}):
            self.assertTrue(self.engine.set_mentor_backend("openai")[0])
        secret.assert_called_once_with()
        module.OpenAI.assert_called_once_with(api_key="fake-test-key", base_url="https://api.openai.com/v1",
                                             timeout=20.0, max_retries=0)
        client.chat.completions.create.assert_not_called()
        self.assertIsInstance(self.engine.brain, OpenAIMentorBrain)

    def test_starting_with_gpt_and_missing_key_fails_before_device_initialization(self):
        args = types.SimpleNamespace(**vars(self.engine.args), mentor_backend="openai")
        with patch.dict("os.environ", {}, clear=True), patch("credentials.get_secret", return_value=None), \
                patch.object(engine_tests.ui.rc, "SapiTTS") as tts, \
                patch.object(engine_tests.ui.rc, "RobotSpeaker") as robot:
            with self.assertRaisesRegex(RuntimeError, "APIキー"):
                engine_tests.ui.Engine(args)
        tts.assert_not_called()
        robot.assert_not_called()

    def test_strata_timeout_default_and_explicit_setting_reach_only_strata(self):
        self.assertEqual(self.engine.strata_timeout, 10.0)
        self.assertEqual(self.source.timeout, 10.0)
        self.assertEqual(self.engine.state()["strata_timeout"], 10.0)
        args = types.SimpleNamespace(**vars(self.engine.args), strata_timeout=4.5)
        with patch.object(engine_tests.ui.rc, "SapiTTS", return_value=self.fixture.tts), \
                patch.object(engine_tests.ui.rc, "LocalSpeaker", return_value=self.fixture.speaker), \
                patch.object(engine_tests.ui.rc, "Gesturer", return_value=self.fixture.gesturer), \
                patch.object(engine_tests.ui.rc, "WhisperSTT", return_value=self.fixture.stt), \
                patch.object(engine_tests.ui, "VoiceCall", engine_tests.FakeCall), \
                patch.object(engine_tests.ui.threading, "Thread"):
            configured = engine_tests.ui.Engine(args)
        try:
            self.assertEqual(configured.brain.timeout, 4.5)
            self.assertEqual(configured.state()["strata_timeout"], 4.5)
        finally:
            configured.close()
        self.engine.set_mentor_backend("openai")
        self.assertEqual(self.target.timeout, 20.0)

    def test_invalid_strata_timeout_is_rejected_before_devices(self):
        for value in (0, .99, 120.01, True, float("nan"), float("inf"), "10"):
            with self.subTest(value=value):
                args = types.SimpleNamespace(**vars(self.engine.args), strata_timeout=value)
                with patch.object(engine_tests.ui.rc, "SapiTTS") as tts:
                    with self.assertRaisesRegex(ValueError, "strata-timeout"):
                        engine_tests.ui.Engine(args)
                tts.assert_not_called()


class MentorBackendHandlerTests(unittest.TestCase):
    def test_manual_backend_endpoint_dispatches(self):
        fixture = engine_tests.HandlerTests()
        fixture.setUp()
        fixture.engine.set_mentor_backend.return_value = (True, "ok")
        status, data, _ = fixture.post("/api/mentor/backend", {"key": "openai"})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        fixture.engine.set_mentor_backend.assert_called_once_with("openai")


if __name__ == "__main__":
    unittest.main()
