"""起立依頼の明示的な姿勢・準備申告を、LLMや機体なしで検証する。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r1_chat"))
from posture_intent import parse_stand_request


class PostureIntentTests(unittest.TestCase):
    def test_bare_commands_expose_missing_conditions_without_granting_readiness(self):
        for text in ("起立して", "起立してください", "立ち上がって", "立ち上がってください", "Stand up", "Please stand up."):
            with self.subTest(text=text):
                self.assertEqual(parse_stand_request(text), {"posture": None, "ready": False})

    def test_supine_commands_still_require_preparation(self):
        for text in ("仰向けから起立して", "仰向けで起立してください", "仰向けから立ち上がってください", "Supine, stand up"):
            with self.subTest(text=text):
                self.assertEqual(parse_stand_request(text), {"posture": "supine", "ready": False})

    def test_explicit_supine_and_preparation_confirmation(self):
        for text in (
            "仰向けで準備確認済み、起立して",
            "仰向けで準備確認済み、起立してください。",
            "仰向けで準備確認済み、立ち上がってください",
            "仰向けに寝ています。起立準備確認済み。起立してください",
            "Supine, preparation confirmed, stand up",
            "SUPINE, PREPARATION CONFIRMED, PLEASE STAND UP!",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_stand_request(text), {"posture": "supine", "ready": True})

    def test_normalizes_width_spacing_and_deliberate_punctuation(self):
        for text in (
            "　仰向けで 準備確認済み ， 起立してください！　",
            "Ｓｕｐｉｎｅ，　ｐｒｅｐａｒａｔｉｏｎ　ｃｏｎｆｉｒｍｅｄ，　ｓｔａｎｄ　ｕｐ．",
            "  supine ,  preparation confirmed , stand\tup  ",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_stand_request(text), {"posture": "supine", "ready": True})

    def test_questions_negations_and_conditions_do_not_become_commands(self):
        for text in (
            "どうやって起立している？", "起立してください？", "起立できますか", "起立しないで", "立ち上がってはいけない",
            "仰向けなら起立して", "準備できたら起立して", "仰向けで準備確認済みではない、起立して",
            "仰向けに寝ています。起立準備確認済みではありません。起立してください",
            "How can you stand up?", "Can you stand up", "Stand up?", "Do not stand up",
            "If supine, preparation confirmed, stand up", "Supine, preparation not confirmed, stand up",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_stand_request(text))

    def test_quoted_embedded_or_compound_requests_are_rejected(self):
        for text in (
            "「起立して」", "『仰向けで準備確認済み、起立して』", '"Stand up"', "`Stand up`", "‘Stand up’",
            "先生が起立してと言った", "起立してから前へ歩いて", "起立して。手を振って", "Stand up and walk forward",
            "Supine, preparation confirmed, stand up; wave", "仰向けで準備確認済み、起立して、歩いて",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_stand_request(text))

    def test_other_postures_or_confirmation_without_posture_are_rejected(self):
        for text in (
            "うつ伏せから起立して", "横向きで準備確認済み、起立して", "吊り下げで起立して", "準備確認済み、起立して",
            "仰向けで起立準備確認済み、起立して", "Prone, preparation confirmed, stand up",
            "Hanging, preparation confirmed, stand up", "Preparation confirmed, stand up",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_stand_request(text))

    def test_non_text_long_or_multiline_input_is_rejected(self):
        for text in (None, False, 1, {}, [], "", " " * 121, "起立して" + " " * 120,
                     "起立して\n", "仰向けに寝ています。\n起立準備確認済み。起立してください", "Stand\rup", "Stand\u2028up"):
            with self.subTest(text=text):
                self.assertIsNone(parse_stand_request(text))

    def test_results_do_not_share_mutable_state(self):
        first = parse_stand_request("起立して")
        first["ready"] = True
        self.assertFalse(parse_stand_request("起立して")["ready"])


if __name__ == "__main__":
    unittest.main()
