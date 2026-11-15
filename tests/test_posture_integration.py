"""起立の明示依頼・準備条件・取り消しを、機体や外部通信なしで検証する。"""
import json
import unittest
from unittest import mock

import test_engine_integration as fixtures


class PostureIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.EngineTests.setUp(self)
        self.posture = mock.Mock()
        self.posture.state.return_value = {"busy": False, "status": "idle", "error": ""}
        self.posture.run.return_value = {"status": "finished", "fsm": 811, "error": ""}
        self.engine.posture = self.posture
        self.engine.tts_en = self.tts  # 英語の案内も既存の音声合成の偽物だけを使う。
        self.engine.body = mock.Mock()
        self.engine.body.state.return_value = {"busy": False, "stop_required": False, "error": ""}
        self.engine.body.stop.return_value = (True, "stopped")
        self.gesturer.loco.reset_mock()
        self.engine.brain.reply = mock.Mock(return_value=("姿勢と手順を確認しよう。", "none"))

    def assert_no_posture_execution(self):
        self.posture.run.assert_not_called()
        self.gesturer.loco.get_fsm.assert_not_called()
        self.gesturer.loco.set_fsm.assert_not_called()

    def test_button_rejects_missing_or_wrong_posture_and_non_boolean_readiness(self):
        cases = ((None, True), ("", True), ("prone", True), ("hanging", True),
                 ("supine", False), ("supine", None), ("supine", 1), ("supine", "true"))
        with mock.patch.object(self.engine, "submit") as submit:
            for posture, ready in cases:
                with self.subTest(posture=posture, ready=ready):
                    ok, message = self.engine.stand_up(posture, ready)
                    self.assertFalse(ok)
                    self.assertTrue(message)
            submit.assert_not_called()
        self.assert_no_posture_execution()

    def test_button_rejects_disabled_body_control_before_submitting(self):
        self.engine.body_enabled = False
        with mock.patch.object(self.engine, "submit") as submit:
            ok, message = self.engine.stand_up("supine", True)
        self.assertFalse(ok)
        self.assertIn("全身制御", message)
        submit.assert_not_called()
        self.assert_no_posture_execution()

    def test_complete_button_request_queues_only_the_dedicated_stand_job(self):
        with mock.patch.object(self.engine, "submit", return_value=(True, "ok")) as submit:
            self.assertEqual(self.engine.stand_up("supine", True), (True, "ok"))
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[:4], ("起立中", self.engine._run_stand, "supine", True))
        self.assert_no_posture_execution()
        self.engine.brain.reply.assert_not_called()

    def test_explicit_chat_confirmation_runs_once_without_llm_or_followup(self):
        self.engine.training.start()
        self.engine.training.waiting()
        result = self.engine._chat("仰向けで準備確認済み、起立して", "文字")
        self.assertEqual(result["status"], "finished")
        self.posture.run.assert_called_once()
        self.assertEqual(self.posture.run.call_args.args, ("supine", True))
        self.assertTrue(self.posture.run.call_args.kwargs["should_continue"]())
        self.engine.brain.reply.assert_not_called()
        self.tts.synth.assert_not_called()
        self.gesturer.start.assert_not_called()
        self.engine.body.move.assert_not_called()
        self.assertFalse(self.engine.training.active)
        self.assertEqual(len([row for row in self.engine.history if row["role"] == "user"]), 1)

    def test_cancel_before_queued_job_starts_does_not_revive_stand_request(self):
        with mock.patch.object(self.engine, "submit", return_value=(True, "ok")) as submit:
            self.assertTrue(self.engine.stand_up("supine", True)[0])
        _, queued_job, *job_args = submit.call_args.args
        self.engine.stand_cancel()
        result = queued_job(*job_args)
        self.assertEqual(result["status"], "cancelled")
        self.assert_no_posture_execution()
        self.engine.brain.reply.assert_not_called()

    def test_incomplete_or_low_confidence_chat_guides_without_llm_or_motion(self):
        cases = (("起立して", False), ("仰向けから起立して", False),
                 ("仰向けで準備確認済み、起立して", True), ("Stand up", False))
        for text, low in cases:
            with self.subTest(text=text, low=low):
                self.tts.synth.reset_mock()
                result = self.engine._chat(text, "声", low_confidence=low)
                self.assertTrue(result["posture_rejected"])
                self.tts.synth.assert_called_once()
                self.assert_no_posture_execution()
        self.engine.brain.reply.assert_not_called()
        self.gesturer.start.assert_not_called()
        self.engine.body.move.assert_not_called()

    def test_full_chat_request_with_body_disabled_does_not_run_or_ask_llm(self):
        self.engine.body_enabled = False
        result = self.engine._chat("仰向けで準備確認済み、起立して", "文字")
        self.assertEqual(result["error"], "body_disabled")
        self.assert_no_posture_execution()
        self.engine.brain.reply.assert_not_called()

    def test_questions_and_quoted_examples_remain_conversation_without_standing(self):
        for text in ("どうやって起立するの？", "How can you stand up?", "「起立して」という文を説明して"):
            with self.subTest(text=text):
                self.engine.brain.reply.reset_mock()
                self.engine._chat(text, "文字")
                self.engine.brain.reply.assert_called_once()
                self.assert_no_posture_execution()
                self.engine.body.move.assert_not_called()

    def test_cancel_controls_cancel_without_querying_fsm_or_starting_stand(self):
        for name in ("stand_cancel", "body_stop", "call_stop"):
            with self.subTest(name=name):
                self.posture.cancel.reset_mock()
                ok, message = getattr(self.engine, name)()
                self.assertTrue(ok)
                self.posture.cancel.assert_called_once()
                self.assert_no_posture_execution()
        self.engine.brain.reply.assert_not_called()

    def test_close_cancels_posture_without_querying_fsm(self):
        self.engine.close()
        self.posture.cancel.assert_called()
        self.assert_no_posture_execution()

    def test_state_exposes_json_safe_posture_and_enabled_flag(self):
        state = self.engine.state()
        self.assertEqual(state["posture"], {"busy": False, "status": "idle", "error": "", "enabled": True})
        json.dumps(state)
        self.assert_no_posture_execution()


class PostureRouteTests(unittest.TestCase):
    post = fixtures.HandlerTests.post

    def setUp(self):
        fixtures.HandlerTests.setUp(self)
        self.engine.stand_up.side_effect = lambda posture, ready: (
            posture == "supine" and ready is True, "checked")
        self.engine.stand_cancel.return_value = (True, "ok")

    def test_stand_route_preserves_explicit_posture_and_boolean(self):
        status, data, _ = self.post("/api/posture/stand", {"posture": "supine", "ready": True})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.engine.stand_up.assert_called_once_with("supine", True)

    def test_route_does_not_coerce_truthy_or_missing_readiness(self):
        for raw in (1, "true", False, None):
            with self.subTest(raw=raw):
                self.engine.stand_up.reset_mock()
                status, data, _ = self.post("/api/posture/stand", {"posture": "supine", "ready": raw})
                self.assertEqual(status, 409)
                self.assertFalse(data["ok"])
                actual = self.engine.stand_up.call_args.args
                self.assertEqual(actual[0], "supine")
                self.assertIs(type(actual[1]), type(raw))
                self.assertEqual(actual[1], raw)
        status, _, _ = self.post("/api/posture/stand", {"ready": True})
        self.assertEqual(status, 409)
        self.engine.stand_up.assert_called_with(None, True)

    def test_cancel_route_only_dispatches_cancel(self):
        status, data, _ = self.post("/api/posture/cancel")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.engine.stand_cancel.assert_called_once_with()
        self.engine.stand_up.assert_not_called()


if __name__ == "__main__":
    unittest.main()
