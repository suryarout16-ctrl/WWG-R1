"""研修の実演・返答待ち・中断を、実デバイスや通信なしで検証する。"""
import unittest
from unittest import mock

import test_engine_integration as fixtures
from mentor_backend import OpenAIMentorBrain


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.EngineTests.setUp(self)
        # 動作の完了はここで模擬する。BodyControllerのスレッドも起動しない。
        self.body_state = {"busy": False, "stop_required": False, "error": ""}
        self.body = mock.Mock()
        self.body.state.side_effect = lambda: dict(self.body_state)
        self.body.move.return_value = (True, "accepted")
        self.body.stop.return_value = (True, "stopped")
        self.body.wait_until_idle.return_value = True
        self.engine.body = self.body
        self.gesturer.thread = None
        self.contexts = []

    def replies(self, responses):
        iterator = iter(responses)

        def reply(text, **kwargs):
            self.contexts.append(self.engine.brain.runtime_context)
            return next(iterator)

        self.engine.brain.reply = mock.Mock(side_effect=reply)

    def demo_replies(self):
        return [
            ("荷物を運ぶときはA、止めるときはBだ。まず見本を見てくれ。", "point_forward"),
            ("荷物を運ぶ動きを、俺が少し前へ出て表してみるぞ。", "move_forward"),
            ("荷物を止めたいとき、どのボタンを選ぶかな？", "none"),
        ]

    def test_demo_explains_moves_and_asks_once_then_waits(self):
        replies = self.demo_replies()
        self.replies(replies)
        with mock.patch.object(self.engine, "_speak", wraps=self.engine._speak) as speak:
            self.engine._run_training("研修を始めて", "文字")
        self.assertEqual(self.engine.brain.reply.call_count, 3)
        self.assertEqual(self.engine.training.phase, "await_reply")
        self.assertTrue(self.engine.training.state()["awaiting_reply"])
        self.assertEqual(self.engine.training.learner_turns, 0)
        self.assertEqual([row["text"] for row in self.engine.history if row["role"] == "user"],
                         ["研修を始めて"])
        self.assertEqual([row["text"] for row in self.engine.history if row["role"] == "r1"],
                         [words for words, gesture in replies])
        self.assertEqual([call.kwargs["allowed_body"] for call in speak.call_args_list],
                         [set(), {"move_forward"}, set()])
        self.body.move.assert_called_once_with("forward", self.engine.forward_duration)
        self.body.wait_until_idle.assert_called_once()
        self.gesturer.start.assert_called_once_with("point_forward")
        self.assertIn("理解確認の質問は一つだけ", self.contexts[-1])

    def test_learner_answer_uses_hearing_context_without_demo_walk_permission(self):
        self.replies(self.demo_replies())
        self.engine._run_training("研修を始めて", "文字")
        move_count = self.body.move.call_count
        self.replies([("止めるならBだ。Aは荷物を運ぶ方だから、ここを確かめよう。", "move_forward")])
        with mock.patch.object(self.engine, "_speak", wraps=self.engine._speak) as speak:
            record = self.engine._chat("止めたいからAを押すよ", "文字")
        self.assertIn("実際に届いた新人の返答", self.contexts[-1])
        self.assertIn("勘違い", self.contexts[-1])
        self.assertEqual(speak.call_args.kwargs["allowed_body"], set())
        self.assertEqual(record["gesture_status"], "body_not_requested")
        self.assertEqual(self.body.move.call_count, move_count)
        self.assertEqual(self.engine.training.learner_turns, 1)
        self.assertEqual(self.engine.training.phase, "await_reply")

    def test_local_preset_preview_is_a_valid_demonstration(self):
        # The real dry Gesturer returns "dry", whereas the shared fake defaults to "sent".
        self.gesturer.start.return_value = "dry"
        self.replies(self.demo_replies())
        self.engine._run_training("研修を始めて", "文字")
        self.assertEqual(self.engine.brain.reply.call_count, 3)
        self.assertTrue(self.engine.training.state()["awaiting_reply"])
        self.body.move.assert_called_once()

    def test_motion_off_rejects_button_start_before_submitting_a_job(self):
        self.engine.auto_motion = False
        with mock.patch.object(self.engine, "submit") as submit:
            ok, message = self.engine.training_start()
        self.assertFalse(ok)
        self.assertIn("オン", message)
        submit.assert_not_called()
        self.assertFalse(self.engine.training.active)

    def test_motion_off_voice_or_text_start_does_not_pretend_to_demonstrate(self):
        self.engine.auto_motion = False
        self.engine.brain.reply = mock.Mock()
        self.engine._chat("研修を始めて", "声")
        self.engine.brain.reply.assert_not_called()
        self.tts.synth.assert_not_called()
        self.body.move.assert_not_called()
        self.assertFalse(self.engine.training.active)
        self.assertEqual(self.engine.history[-1]["role"], "system")
        self.assertIn("オン", self.engine.history[-1]["text"])

    def test_stop_during_llm_discards_reply_and_never_reaches_demo(self):
        def reply(*args, **kwargs):
            self.engine.training_stop()
            return "見本を見てくれ。", "point_forward"

        self.engine.brain.reply = mock.Mock(side_effect=reply)
        self.engine._run_training("研修を始めて", "文字")
        self.engine.brain.reply.assert_called_once()
        self.tts.synth.assert_not_called()
        self.gesturer.start.assert_not_called()
        self.body.move.assert_not_called()
        self.assertEqual(self.engine.training.phase, "stopped")

    def test_stop_during_tts_never_starts_preset_or_followup_demo(self):
        self.replies(self.demo_replies())

        def synth(text):
            self.engine.training_stop()
            return b"\0\0" * 100

        self.tts.synth.side_effect = synth
        self.engine._run_training("研修を始めて", "文字")
        self.engine.brain.reply.assert_called_once()
        self.gesturer.start.assert_not_called()
        self.speaker.play.assert_not_called()
        self.body.move.assert_not_called()
        self.assertEqual(self.engine.training.phase, "stopped")

    def test_stop_during_body_start_sends_stop_and_skips_question(self):
        self.replies(self.demo_replies())

        def stop(*args, **kwargs):
            self.body_state["busy"] = False
            return True, "stopped"

        def move(*args, **kwargs):
            self.body_state["busy"] = True
            self.engine.training_stop()
            return True, "accepted"

        self.body.stop.side_effect = stop
        self.body.move.side_effect = move
        self.engine._run_training("研修を始めて", "文字")
        self.assertEqual(self.engine.brain.reply.call_count, 2)
        self.body.stop.assert_called_once()
        self.assertEqual(self.engine.training.phase, "stopped")
        self.assertFalse(self.engine.training.state()["awaiting_reply"])

    def test_llm_failure_ends_flow_instead_of_demonstrating(self):
        self.engine.brain.reply = mock.Mock(side_effect=RuntimeError("test timeout"))
        record = self.engine._run_training("研修を始めて", "文字")
        self.engine.brain.reply.assert_called_once()
        self.body.move.assert_not_called()
        self.assertIn("llm:", record["error"])
        self.assertEqual(self.engine.training.phase, "error")
        self.assertFalse(self.engine.training.active)

    def test_demonstration_without_an_action_stops_before_asking_the_learner(self):
        self.replies([
            self.demo_replies()[0],
            ("荷物を運ぶ見本の説明をするぞ。", "none"),
        ])
        self.engine._run_training("研修を始めて", "文字")
        self.assertEqual(self.engine.brain.reply.call_count, 2)
        self.body.move.assert_not_called()
        self.assertEqual(self.engine.training.phase, "error")
        self.assertFalse(self.engine.training.state()["awaiting_reply"])
        self.assertEqual(len([row for row in self.engine.history if row["role"] == "r1"]), 2)

    def test_physical_connection_uses_stationary_demo_without_auto_walk_permission(self):
        self.engine.args.local = False
        self.engine.args.interface = "Ethernet"
        replies = self.demo_replies()
        replies[1] = ("ここでは腕の見本で荷物を運ぶ方を示すぞ。", "point_forward")
        self.replies(replies)
        with mock.patch.object(self.engine, "_speak", wraps=self.engine._speak) as speak:
            self.engine._run_training("研修を始めて", "文字")
        self.assertTrue(self.engine.training.state()["awaiting_reply"])
        self.assertTrue(all(not call.kwargs["allowed_body"] for call in speak.call_args_list))
        self.body.move.assert_not_called()
        self.assertNotIn("仮想の見本歩行許可がある", self.contexts[1])

    def test_physical_connection_rejects_unrequested_demo_walking(self):
        self.engine.args.local = False
        self.engine.args.interface = "Ethernet"
        self.replies(self.demo_replies())
        record = self.engine._run_training("研修を始めて", "文字")
        self.assertEqual(self.engine.brain.reply.call_count, 2)
        self.assertEqual(record["training_action_blocked"], "move_forward")
        self.body.move.assert_not_called()
        self.assertEqual(self.engine.training.phase, "error")

    def test_reset_and_material_changes_end_waiting_session(self):
        self.engine.training.start()
        self.engine.training.waiting()
        self.assertTrue(self.engine.reset()[0])
        self.assertEqual(self.engine.training.phase, "idle")
        self.engine.training.start()
        self.engine.training.waiting()
        self.assertTrue(self.engine.material_add("作業", "Aは荷物の移動、Bは停止。")[0])
        self.assertFalse(self.engine.training.active)
        document_id = self.engine.materials.documents()[0]["id"]
        self.engine.training.start()
        self.engine.training.waiting()
        self.assertTrue(self.engine.material_remove(document_id)[0])
        self.assertFalse(self.engine.training.active)

    def test_backend_switch_preserves_engine_waiting_session(self):
        target = OpenAIMentorBrain(mock.Mock(), "gpt-5-mini")
        self.engine.training.start()
        self.engine.training.waiting()
        self.engine.training.learner_replied()
        before = self.engine.training.state()
        generation = self.engine.training.generation
        with mock.patch.object(self.engine, "_gpt_mentor", return_value=target):
            self.assertTrue(self.engine.set_mentor_backend("openai")[0])
        self.assertEqual(self.engine.training.generation, generation)
        self.assertEqual(self.engine.training.state(), before)


if __name__ == "__main__":
    unittest.main()
