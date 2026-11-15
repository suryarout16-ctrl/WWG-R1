"""研修の段階・中断・開始依頼を、LLMやデバイスなしで検証する。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r1_chat"))
from training_flow import TrainingFlow, is_training_start


class TrainingFlowTests(unittest.TestCase):
    def test_start_steps_and_wait_for_a_real_reply(self):
        flow = TrainingFlow()
        self.assertEqual(flow.state()["phase"], "idle")
        generation = flow.start()
        self.assertTrue(flow.active)
        self.assertEqual(flow.phase, "explain")
        for step in flow.STEPS:
            flow.enter(step)
            self.assertEqual(flow.phase, step)
            self.assertFalse(flow.state()["awaiting_reply"])
            self.assertEqual(flow.generation, generation)
        flow.waiting()
        self.assertTrue(flow.state()["awaiting_reply"])
        self.assertEqual(flow.learner_turns, 0)
        flow.learner_replied()
        self.assertEqual(flow.learner_turns, 1)
        self.assertEqual(flow.phase, "await_reply")

    def test_stop_invalidates_inflight_generation_and_restart_clears_turns(self):
        flow = TrainingFlow()
        first = flow.start()
        flow.waiting()
        flow.learner_replied()
        flow.stop()
        self.assertGreater(flow.generation, first)
        self.assertFalse(flow.active)
        self.assertFalse(flow.state()["awaiting_reply"])
        flow.learner_replied()
        self.assertEqual(flow.phase, "stopped")
        self.assertEqual(flow.learner_turns, 1)
        second = flow.start()
        self.assertGreater(second, first)
        self.assertEqual(flow.learner_turns, 0)
        flow.stop("error")
        self.assertEqual(flow.state()["phase"], "error")

    def test_invalid_step_cannot_start_or_corrupt_the_flow(self):
        flow = TrainingFlow()
        with self.assertRaises(ValueError):
            flow.enter("demonstrate")
        flow.start()
        with self.assertRaises(ValueError):
            flow.enter("pretend_learner_answer")
        self.assertEqual(flow.phase, "explain")

    def test_context_keeps_question_and_answer_separate_and_supports_other_topics(self):
        flow = TrainingFlow()
        self.assertEqual(flow.context(), "")
        ask = flow.context("ask")
        self.assertIn("一つだけ", ask)
        self.assertIn("自分でその答えを続けて言わず", ask)
        self.assertIn("返答を待つ", ask)
        hearing = flow.context("await_reply")
        self.assertIn("実際に届いた", hearing)
        self.assertIn("勘違い", hearing)
        self.assertIn("再実演", hearing)
        self.assertIn("存在意義", hearing)
        self.assertIn("無理に小テストへ戻さない", hearing)
        self.assertIn("台詞ではない", hearing)

    def test_state_has_no_mutable_internal_objects_or_generation_authority(self):
        flow = TrainingFlow()
        snapshot = flow.state()
        self.assertEqual(set(snapshot), {"active", "phase", "label", "awaiting_reply", "learner_turns"})
        snapshot["active"] = True
        self.assertFalse(flow.active)

    def test_stage_requests_are_distinct_goals_not_repeated_start_or_dialogue(self):
        flow = TrainingFlow()
        requests = [flow.request(step) for step in flow.STEPS]
        self.assertEqual(len(set(requests)), 3)
        for request in requests:
            self.assertTrue(request.strip())
            self.assertLessEqual(len(request), 250)
            self.assertFalse(is_training_start(request))
            self.assertNotIn('"say"', request)
            self.assertNotIn("俺は", request)
        self.assertIn("一文", requests[0])
        self.assertIn("始める前", requests[1])
        self.assertIn("質問を一つだけ", requests[2])
        self.assertIn("自分で言わず", requests[2])
        self.assertEqual(flow.phase, "idle")
        with self.assertRaises(ValueError):
            flow.request("await_reply")


class TrainingStartIntentTests(unittest.TestCase):
    def test_direct_start_requests_in_japanese_and_english(self):
        for text in (
            "研修を始めて", "新人研修を始めてください。", "研修を始めて下さい！",
            "作業の説明と実演をして", "作業の説明と実演をしてください。",
            "Start training", "Please start training.", "Please start the training demo!",
            "  PLEASE  START THE TRAINING DEMO  ",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_training_start(text))

    def test_pasted_material_quotes_negations_and_questions_do_not_start(self):
        for text in (
            "研修を始めないで", "研修を始めていいですか", "研修を始めてから前へ進んで",
            "「研修を始めて」", "台本: 研修を始めて", "研修を始めて\nAは移動",
            'Say "Start training"', "Don't start training", "Do not start training",
            "How do I start training?", "If I say start training, walk forward",
            "Start training and walk forward", "Start training\n", "", None, 123,
        ):
            with self.subTest(text=text):
                self.assertFalse(is_training_start(text))


if __name__ == "__main__":
    unittest.main()
