"""LLMの移動候補は、未確認の知覚・移動結果として読み上げない。実機は呼ばない。"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_chat"))
import personas


class MotionSpeechTests(unittest.TestCase):
    def brain(self):
        brain = personas.LocalBrain("http://unused.invalid/v1", personas.MENTOR_PROMPT,
                                    model="mock")
        brain.set_body_enabled(True)
        return brain

    def test_observed_false_perception_is_replaced_without_losing_direction_or_evidence(self):
        brain = self.brain()
        original = "了解した。ゆっくりと前へ進むぞ。周囲の安全は俺が確認しているから、安心して見ていてくれ。"
        brain._stream = Mock(return_value=json.dumps({"say": original, "gesture": "move_forward"},
                                                     ensure_ascii=False))
        with patch.object(personas.rc, "log") as log:
            say, action = brain.reply("前へ少し進んでください。")
        self.assertEqual(action, "move_forward")
        self.assertIn("前", say)
        self.assertIn("実行できるか", say)
        self.assertNotIn("安全", say)
        self.assertNotIn("安心", say)
        self.assertNotIn("確認している", say)
        self.assertEqual(brain.last_reply["model_say"], original)
        self.assertEqual(brain.last_reply["model_gesture"], action)
        self.assertIn(original, log.call_args.args[0])
        stored = json.loads(brain.history[-1]["content"])
        self.assertEqual(stored, {"say": say, "gesture": action})
        self.assertNotIn(original, brain._messages("次へ")[2]["content"])

    def test_all_body_candidates_use_pending_rather_than_completed_announcements(self):
        for action, direction in (("move_forward", "前"), ("move_backward", "後ろ"),
                                  ("move_left", "左"), ("move_right", "右"),
                                  ("turn_left", "左"), ("turn_right", "右"),
                                  ("stop_move", "停止")):
            with self.subTest(action=action), patch.object(personas.rc, "log"):
                brain = self.brain()
                brain._stream = Mock(return_value=json.dumps({"say": "動いて無事に停止したぞ。障害物はない。",
                                                             "gesture": action}, ensure_ascii=False))
                say, selected = brain.reply("動作の依頼")
                self.assertEqual(selected, action)
                self.assertIn(direction, say)
                self.assertNotIn("停止した", say)
                self.assertNotIn("障害物はない", say)
                if action == "stop_move":
                    self.assertIn("止まったか", say)
                    self.assertIn("目視", say)
                else:
                    self.assertIn("実行できるか", say)

    def test_no_motion_response_is_unchanged_and_new_turn_cannot_reuse_old_override(self):
        brain = self.brain()
        brain._stream = Mock(side_effect=[
            '{"say":"周囲は確認済みだ。","gesture":"move_left"}',
            '{"say":"まず指示書を確認しよう。","gesture":"emphasize"}',
            OSError("offline")])
        with patch.object(personas.rc, "log"):
            brain.reply("左へ動いて")
            self.assertEqual(brain.reply("手順を教えて"), ("まず指示書を確認しよう。", "emphasize"))
            self.assertEqual(brain.last_reply["model_say"], "まず指示書を確認しよう。")
            with self.assertRaises(OSError):
                brain.reply("もう一度")
        self.assertIsNone(brain.last_reply)

    def test_self_introduction_still_takes_precedence_over_motion(self):
        brain = self.brain()
        brain._stream = Mock(return_value='{"say":"前に動いたぞ。","gesture":"move_forward"}')
        with patch.object(personas.rc, "log"):
            say, gesture = brain.reply("自己紹介して")
        self.assertEqual(say, personas.SELF_INTRODUCTION)
        self.assertEqual(gesture, "none")
        self.assertEqual(brain.last_reply["model_say"], "前に動いたぞ。")


if __name__ == "__main__":
    unittest.main()
