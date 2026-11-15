"""研修の各ターンで許可した動作・発話量を守る。LLMと通信はすべて偽物。"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_chat"))
import personas


class TurnConstraintTests(unittest.TestCase):
    def brain(self, *, motion=True, model="test"):
        brain = personas.LocalBrain("http://unused.invalid/v1", personas.MENTOR_PROMPT,
                                    motion=motion, model=model)
        brain.set_body_enabled(True)
        brain.available = Mock(return_value=(True, "test"))
        brain._stream = Mock()
        return brain

    @staticmethod
    def structured(say="説明するぞ。", gesture="none"):
        return json.dumps({"say": say, "gesture": gesture}, ensure_ascii=False)

    @staticmethod
    def properties(brain, call_index=-1):
        body = brain._stream.call_args_list[call_index].args[0]
        return body["response_format"]["json_schema"]["schema"]["properties"]

    def test_turn_schema_uses_the_requested_actions_and_speech_limit(self):
        brain = self.brain()
        brain._stream.return_value = self.structured("対象を示すぞ。", "point_forward")
        self.assertEqual(brain.reply("操作を説明して", allowed_gestures=["none", "point_forward"],
                                     max_chars=30), ("対象を示すぞ。", "point_forward"))
        props = self.properties(brain)
        self.assertEqual(props["gesture"]["enum"], ["none", "point_forward"])
        self.assertEqual(props["say"]["maxLength"], 30)

    def test_constraints_do_not_persist_into_the_next_turn(self):
        brain = self.brain()
        brain._stream.side_effect = [self.structured("確認するぞ。", "wave"),
                                     self.structured("少し前へ出よう。", "move_forward")]
        brain.reply("説明して", allowed_gestures=["wave"], max_chars=12)
        self.assertEqual(brain.reply("前へ進んで"), ("少し前へ出よう。", "move_forward"))
        first, following = self.properties(brain, 0), self.properties(brain, 1)
        self.assertEqual(first["gesture"]["enum"], ["wave"])
        self.assertEqual(following["gesture"]["enum"], ["none", *brain.action_names])
        self.assertEqual(following["say"]["maxLength"], brain.MAX_SAY_CHARS)
        self.assertEqual(brain.MAX_SAY_CHARS, 240)

    def test_supported_action_collections_preserve_the_requested_set(self):
        for names in (["none", "wave"], ("none", "wave"), {"none", "wave"},
                      frozenset({"none", "wave"})):
            with self.subTest(collection=type(names).__name__):
                brain = self.brain()
                brain._stream.return_value = self.structured("ようこそ。", "wave")
                self.assertEqual(brain.reply("こんにちは", allowed_gestures=names)[1], "wave")
                self.assertEqual(set(self.properties(brain)["gesture"]["enum"]), set(names))

    def test_invalid_action_constraints_fail_before_model_lookup_or_stream(self):
        bad_values = ([], (), set(), frozenset(), "wave", {"wave": True}, 5, True,
                      ["not_a_real_action"], ["wave", "not_a_real_action"],
                      ["wave", 1], [None])
        for value in bad_values:
            with self.subTest(value=repr(value)):
                brain = self.brain(model="")
                with self.assertRaises((ValueError, TypeError)):
                    brain.reply("説明して", allowed_gestures=value)
                brain.available.assert_not_called()
                brain._stream.assert_not_called()
                self.assertEqual(brain.history, [])

    def test_body_action_cannot_be_enabled_by_turn_constraints(self):
        brain = self.brain(model="")
        brain.set_body_enabled(False)
        with self.assertRaises((ValueError, TypeError)):
            brain.reply("前へ進んで", allowed_gestures=["move_forward"])
        brain.available.assert_not_called()
        brain._stream.assert_not_called()

    def test_invalid_character_limits_fail_before_model_lookup_or_stream(self):
        for language, upper in (("ja", 240), ("en", 400)):
            for value in (True, False, 0, -1, upper + 1, 10.0, "10", [], {}):
                with self.subTest(language=language, value=repr(value)):
                    brain = self.brain(model="")
                    with self.assertRaises((ValueError, TypeError)):
                        brain.reply("説明して", language=language, max_chars=value)
                    brain.available.assert_not_called()
                    brain._stream.assert_not_called()
                    self.assertEqual(brain.history, [])

    def test_character_limit_bounds_are_inclusive_for_each_language(self):
        for language, upper, char in (("ja", 240, "あ"), ("en", 400, "a")):
            for limit in (1, upper):
                with self.subTest(language=language, limit=limit):
                    brain = self.brain()
                    text = char * limit
                    brain._stream.return_value = self.structured(text)
                    self.assertEqual(brain.reply("説明して", language=language,
                                                max_chars=limit), (text, "none"))
                    self.assertEqual(self.properties(brain)["say"]["maxLength"], limit)

    def test_out_of_turn_action_is_rejected_even_when_generally_available(self):
        brain = self.brain()
        brain._stream.side_effect = [self.structured("少し前へ出よう。", "move_forward"),
                                     "対象を示して説明するぞ。"]
        with patch.object(personas.rc, "log"):
            self.assertEqual(brain.reply("説明して", allowed_gestures=["point_forward"],
                                         max_chars=30), ("対象を示して説明するぞ。", "none"))
        self.assertEqual(brain._stream.call_count, 2)
        self.assertNotIn("response_format", brain._stream.call_args.args[0])
        self.assertEqual(json.loads(brain.history[-1]["content"])["gesture"], "none")
        self.assertNotIn("move_forward", brain.history[-1]["content"])

    def test_overlong_structured_speech_gets_only_one_plain_motionless_retry(self):
        brain = self.brain()
        brain._stream.side_effect = [self.structured("あ" * 11, "wave"), "短く説明する。"]
        with patch.object(personas.rc, "log"):
            self.assertEqual(brain.reply("説明して", allowed_gestures=["wave"], max_chars=10),
                             ("短く説明する。", "none"))
        self.assertEqual(brain._stream.call_count, 2)
        self.assertNotIn("response_format", brain._stream.call_args.args[0])
        self.assertEqual(brain.last_reply["gesture"], "none")

    def test_plain_retry_must_also_obey_the_turn_character_limit(self):
        brain = self.brain()
        brain._stream.side_effect = [self.structured("あ" * 11, "wave"), "い" * 11]
        with patch.object(personas.rc, "log"), self.assertRaises(RuntimeError):
            brain.reply("説明して", allowed_gestures=["wave"], max_chars=10)
        self.assertEqual(brain._stream.call_count, 2)
        self.assertEqual(brain.history, [])

    def test_plain_mode_keeps_motion_off_and_applies_the_turn_limit(self):
        brain = self.brain(motion=False)
        brain._stream.return_value = "作業を説明する。"
        self.assertEqual(brain.reply("説明して", allowed_gestures=["wave"], max_chars=12),
                         ("作業を説明する。", "none"))
        self.assertNotIn("response_format", brain._stream.call_args.args[0])

    def test_default_limits_and_actions_remain_unchanged(self):
        for language, limit in (("ja", 240), ("en", 400)):
            with self.subTest(language=language):
                brain = self.brain()
                brain._stream.return_value = self.structured("Okay.", "wave")
                self.assertEqual(brain.reply("質問", language=language), ("Okay.", "wave"))
                props = self.properties(brain)
                self.assertEqual(props["say"]["maxLength"], limit)
                self.assertEqual(props["gesture"]["enum"], ["none", *brain.action_names])


if __name__ == "__main__":
    unittest.main()
