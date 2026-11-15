"""日英の聞き分け・英語応答・技術Q&Aの事実集を、通信・デバイスなしで検証する。"""
import json
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1_chat"))
sys.path.insert(0, str(HERE))
import bilingual
import identity
import personas
import r1_chat as rc
import stt_worker
import tech_facts
import test_engine_integration as engine_base   # setUp だけ借りる（クラスは取り込まない）
import web_ui as ui


class SttLanguageTests(unittest.TestCase):
    def test_parse_languages(self):
        self.assertEqual(stt_worker.parse_languages("auto"), ["ja", "en"])
        self.assertEqual(stt_worker.parse_languages("ja"), ["ja"])
        self.assertEqual(stt_worker.parse_languages("en,ja,en"), ["en", "ja"])
        for bad in ("ja;rm", "japanese", "j1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stt_worker.parse_languages(bad)

    def test_accented_english_is_not_turned_into_hindi(self):
        # Whisper の全言語判定ではヒンディー語が最上位でも、日英の中では英語が圧倒的
        probs = [("hi", .55), ("en", .40), ("ja", .02), ("ur", .03)]
        self.assertEqual(stt_worker.choose_language(probs, ["ja", "en"])[0], "en")

    def test_japanese_is_preferred_when_unsure(self):
        lang, p = stt_worker.choose_language([("ja", .45), ("en", .55)], ["ja", "en"])
        self.assertEqual(lang, "ja")   # 英語 0.55 < しきい値 0.65
        self.assertEqual(stt_worker.choose_language([("ja", .2), ("en", .8)], ["ja", "en"])[0], "en")
        self.assertEqual(stt_worker.choose_language([("fr", 1.0)], ["ja", "en"]), ("ja", 0.0))

    def test_text_language(self):
        for text, lang in (("Please introduce yourself", "en"), ("Aボタンは何？", "ja"),
                           ("start を押す", "ja"), ("", "ja"), ("123 !?", "ja"), ("OK", "en")):
            with self.subTest(text=text):
                self.assertEqual(rc.text_language(text), lang)


class EnglishIntentTests(unittest.TestCase):
    def test_english_introduction_requests(self):
        for text in ("Please introduce yourself", "Hello sir, please introduce yourself.",
                     "Who are you?", "R1, what is your name", "Can you introduce yourself please"):
            with self.subTest(text=text):
                self.assertTrue(identity.is_self_introduction_request(text))
        for text in ("Don't introduce yourself", "Who are you working with in the factory today?",
                     'He said "introduce yourself"', "introduce yourself and then walk forward"):
            with self.subTest(text=text):
                self.assertFalse(identity.is_self_introduction_request(text))

    def test_english_body_requests_are_strict(self):
        cases = {
            "Please walk forward": {"move_forward"}, "R1, turn left a little": {"turn_left"},
            "Stop!": {"stop_move"}, "okay, step back slowly": {"move_backward"},
            "How do you walk forward?": set(), "If the machine stops, walk forward": set(),
            "Don't walk forward": set(), "Press button A to move forward": set(),
            "walk forward after 5 minutes": set(), "前へ進んで": set(),
            "Stop. How do you stop?": {"stop_move"},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(bilingual.requested_body_actions_en(text), expected)

    def test_technical_question_detection(self):
        # R1自身の本当の仕組み → 事実集を付ける（作り話を防ぐ）
        for text in ("どういう仕組みで動いてるの？", "AIは何を使ってる？", "How do you work?",
                     "Which AI model you are using?", "Is your code public on GitHub?",
                     "実際のR1の関節トルクは？", "本当にLoRAとカメラ自動認識が実装されている？",
                     "R1はカメラで見てるの？", "You are seeing us with camera for safety, no?",
                     "この機械はStrataで動いてるの？"):
            with self.subTest(text=text):
                self.assertTrue(tech_facts.is_technical_question(text))
        # 架空の機械・模擬の作業 → 創作に任せる
        for text in ("この架空の機械はどうやって荷物を運ぶの？", "How does this imaginary machine move boxes?",
                     "Can you train me on this pretend machine?", "この機械のモーターは何馬力？",
                     "Aボタンを押して", "おはよう", "Thank you", ""):
            with self.subTest(text=text):
                self.assertFalse(tech_facts.is_technical_question(text))


class BrainLanguageTests(unittest.TestCase):
    def brain(self):
        return personas.LocalBrain("http://unused.invalid/v1", personas.MENTOR_PROMPT, motion=True, model="test")

    @staticmethod
    def reply(say, gesture="none"):
        return json.dumps({"say": say, "gesture": gesture}, ensure_ascii=False)

    def system(self, brain):
        return brain._stream.call_args.args[0]["messages"][0]["content"]

    def test_english_rule_is_last_and_tech_facts_only_for_technical_questions(self):
        brain = self.brain()
        brain._stream = mock.Mock(return_value=self.reply("I use speech recognition on the PC."))
        brain.reply("How does it work?")
        system = self.system(brain)
        self.assertTrue(system.rstrip().endswith(bilingual.ENGLISH_REPLY_RULE.strip()))
        self.assertIn(tech_facts.TECH_FACTS.strip(), system)
        self.assertIn("公開状況", system)
        brain._stream = mock.Mock(return_value=self.reply("I am glad to help."))
        brain.reply("Thank you")
        self.assertNotIn(tech_facts.TECH_FACTS.strip(), self.system(brain))
        self.assertEqual(brain.last_reply["language"], "en")

    def test_japanese_turn_has_confirm_rule_but_no_english_rule(self):
        brain = self.brain()
        brain._stream = mock.Mock(return_value=self.reply("どのボタンのことかな？"))
        brain.reply("あれ押せばいいんだっけ")
        system = self.system(brain)
        self.assertIn(bilingual.CONFIRM_RULE.strip(), system)
        self.assertNotIn(bilingual.ENGLISH_REPLY_RULE.strip(), system)
        self.assertNotIn(tech_facts.TECH_FACTS.strip(), system)

    def test_low_confidence_rule_by_language(self):
        brain = self.brain()
        brain._stream = mock.Mock(return_value=self.reply("Did you mean button B?"))
        brain.reply("I press bee button?", language="en", low_confidence=True)
        self.assertIn(bilingual.LOW_CONFIDENCE_RULE["en"].strip(), self.system(brain))

    def test_fixed_lines_mode_keeps_canonical_introduction_for_regression(self):
        brain = self.brain()
        brain.fixed_lines = True
        brain._stream = mock.Mock(return_value=self.reply("I am a sales robot.", "wave"))
        self.assertEqual(brain.reply("Please introduce yourself"), (identity.SELF_INTRODUCTION_EN, "wave"))
        brain._stream = mock.Mock(return_value=self.reply("俺は販売員だ。"))
        self.assertEqual(brain.reply("自己紹介して")[0], identity.SELF_INTRODUCTION)
        brain._stream = mock.Mock(return_value=self.reply("Okay, a short step forward.", "move_forward"))
        brain.set_body_enabled(True)
        self.assertEqual(brain.reply("Please walk forward")[0], bilingual.BODY_ANNOUNCEMENTS_EN["move_forward"])

    def test_english_body_announcement_replaces_model_text(self):
        brain = self.brain()
        brain.set_body_enabled(True)
        brain._stream = mock.Mock(return_value=self.reply("The area is safe, trust me.", "move_forward"))
        say, gesture = brain.reply("Please walk forward")
        self.assertEqual((say, gesture), (bilingual.BODY_ANNOUNCEMENTS_EN["move_forward"], "move_forward"))

    def test_english_reply_may_be_longer_than_japanese_limit(self):
        brain = self.brain()
        text = "Button A moves the load and button B stops it. " * 6   # 約 290 文字
        self.assertGreater(len(text.strip()), brain.MAX_SAY_CHARS)
        brain._stream = mock.Mock(return_value=self.reply(text.strip(), "emphasize"))
        self.assertEqual(brain.reply("What do A and B do?")[0], text.strip())


class LiveConversationTests(unittest.TestCase):
    """本番の即興対話（fixed_lines=False）。言い回しは LLM、事実と実行可否はプログラム。"""

    def brain(self):
        brain = personas.LocalBrain("http://unused.invalid/v1", personas.MENTOR_PROMPT, motion=True, model="test")
        brain.set_body_enabled(True)
        return brain

    @staticmethod
    def reply(say, gesture="none"):
        return json.dumps({"say": say, "gesture": gesture}, ensure_ascii=False)

    def test_prompt_has_three_domains_and_no_fixed_script(self):
        self.assertIn("会話の三つの領域", personas.MENTOR_PROMPT)
        self.assertIn("架空の作業", personas.MENTOR_PROMPT)
        self.assertNotIn("固定文を全文", personas.MENTOR_PROMPT)
        self.assertNotIn(identity.SELF_INTRODUCTION, personas.MENTOR_PROMPT)
        for word in ("新人", "事故", "AIは何をするか決める"):   # 存在意義は意味として持つ
            self.assertIn(word, personas.MENTOR_PROMPT)
        self.assertFalse(self.brain().fixed_lines)

    def test_current_backend_is_given_as_measured_fact(self):
        from mentor_backend import OpenAIMentorBrain
        strata = self.brain()
        strata._stream = mock.Mock(return_value=self.reply("Strataで考えている。"))
        strata.reply("今どのAIで返事を作ってる？")
        system = strata._stream.call_args.args[0]["messages"][0]["content"]
        self.assertIn("【今この返事を作っているAI（実測。仕組みの説明より優先）】Z840という研究用PCのStrata（モデル名 test）", system)
        gpt = OpenAIMentorBrain(mock.Mock(), "gpt-5-mini")
        gpt._stream = mock.Mock(return_value=self.reply("GPT APIで考えている。"))
        gpt.reply("実際のあなたは返事の言葉と腕の動作をどう決める？")
        system = gpt._stream.call_args.args[0]["messages"][0]["content"]
        self.assertIn("】OpenAIのGPT API", system)
        self.assertIn("gpt-5-mini", system)
        self.assertIn(tech_facts.TECH_FACTS.strip(), system)   # 一般説明もあるが、実測行が優先と明記
        self.assertLess(system.index(tech_facts.TECH_FACTS.strip()), system.index("【今この返事を作っているAI"))

    def test_self_introduction_is_the_models_own_words(self):
        brain = self.brain()
        for text, said in (("自己紹介して", "俺は研修担当のR1。新人さんが事故なく機械を扱えるよう、一緒に手順を確かめる係だ。"),
                           ("Please introduce yourself", "I am R1. I help new workers learn machines safely.")):
            with self.subTest(text=text):
                brain._stream = mock.Mock(return_value=self.reply(said, "wave"))
                self.assertEqual(brain.reply(text), (said, "wave"))

    def test_introduction_never_moves_and_false_motion_claim_falls_back(self):
        brain = self.brain()
        brain._stream = mock.Mock(return_value=self.reply("俺はR1だ。よろしく。", "move_forward"))
        self.assertEqual(brain.reply("自己紹介して"), ("俺はR1だ。よろしく。", "none"))
        brain._stream = mock.Mock(return_value=self.reply("前に動いたぞ。", "move_forward"))
        self.assertEqual(brain.reply("自己紹介して"), (identity.SELF_INTRODUCTION, "none"))

    def test_natural_motion_speech_is_kept_unless_it_claims_results(self):
        brain = self.brain()
        kept = (("よし、少しだけ前へ出てみよう。", "move_forward", "前へ進んで"),
                ("Okay, a short step forward. Watch me.", "move_forward", "Please walk forward"),
                ("止まるぞ。", "stop_move", "止まって"))
        for say, gesture, text in kept:
            with self.subTest(say=say), mock.patch.object(personas.rc, "log"):
                brain._stream = mock.Mock(return_value=self.reply(say, gesture))
                self.assertEqual(brain.reply(text), (say, gesture))
        replaced = (("周りの安全は確認したから、1メートル進むぞ。", "move_forward", "前へ進んで", personas.BODY_ANNOUNCEMENTS),
                    ("I moved forward safely.", "move_forward", "Please walk forward", bilingual.BODY_ANNOUNCEMENTS_EN),
                    ("Trust me, there are no obstacles.", "turn_left", "Turn left", bilingual.BODY_ANNOUNCEMENTS_EN))
        for say, gesture, text, table in replaced:
            with self.subTest(say=say), mock.patch.object(personas.rc, "log"):
                brain._stream = mock.Mock(return_value=self.reply(say, gesture))
                self.assertEqual(brain.reply(text), (table[gesture], gesture))

    def test_motion_claim_detector(self):
        for text in ("動いたぞ", "安全を確認した", "2メートル進む", "障害物はない", "It is safe now", "I walked 3 m"):
            with self.subTest(text=text):
                self.assertTrue(personas.motion_claim(text))
        for text in ("少し前へ出てみよう", "状態を確かめてから動く", "A short step forward", "Here I go"):
            with self.subTest(text=text):
                self.assertFalse(personas.motion_claim(text))


class EngineLanguageTests(unittest.TestCase):
    setUp = engine_base.EngineTests.setUp

    def test_english_voice_is_created_once_and_closed(self):
        english = types.SimpleNamespace(synth=mock.Mock(return_value=b"\0\0" * 100),
                                        proc=mock.Mock(**{"poll.return_value": None}))
        with mock.patch.object(ui.rc, "SapiTTS", return_value=english) as tts:
            self.assertIs(self.engine._tts_for("Hello, I am R1."), english)
            self.assertIs(self.engine._tts_for("Press button A."), english)
            self.assertIs(self.engine._tts_for("こんにちは"), self.tts)
        tts.assert_called_once_with("Microsoft David", 0)
        self.engine.close()
        english.proc.terminate.assert_called_once()

    def test_missing_english_voice_falls_back_to_japanese_voice(self):
        with mock.patch.object(ui.rc, "SapiTTS", side_effect=RuntimeError("no voice")):
            self.assertIs(self.engine._tts_for("Hello"), self.tts)

    def test_voice_turn_passes_language_and_confidence_to_brain(self):
        self.stt.transcribe_detail = mock.Mock(return_value={
            "text": "I press bee button?", "language": "en", "language_prob": .9, "avg_logprob": -1.4})
        calls = []

        def reply(text, *, language=None, low_confidence=False):   # LocalBrain.reply と同じ形
            calls.append((text, language, low_confidence))
            return "Did you mean button B?", "none"

        self.engine.brain.reply = reply
        with mock.patch.object(ui.rc, "SapiTTS", return_value=self.tts):
            self.engine._call_utterance(np.zeros(16000, dtype=np.int16), lambda: True)
        self.assertEqual(calls, [("I press bee button?", "en", True)])
        self.stt.transcribe.assert_not_called()

    def test_old_style_brain_and_stt_still_work(self):
        self.engine.brain.reply = mock.Mock(side_effect=lambda text: ("まず工具を確認しよう。", "none"))
        self.engine._call_utterance(np.zeros(16000, dtype=np.int16), lambda: True)
        self.engine.brain.reply.assert_called_once_with("手順を教えて")
        self.tts.synth.assert_called()

    def test_errors_and_not_heard_follow_language(self):
        self.engine.brain.reply = mock.Mock(side_effect=RuntimeError("timeout"))
        with mock.patch.object(self.engine, "_speak") as speak:
            self.engine._chat("What does button A do?", "文字")
        self.assertEqual(speak.call_args.args[0], bilingual.ERROR_REPLY["en"])
        self.stt.transcribe_detail = mock.Mock(return_value={"text": "", "language": "en"})
        with mock.patch.object(self.engine, "add") as add:
            self.engine._call_utterance(np.zeros(16000, dtype=np.int16), lambda: True)
        add.assert_called_with("system", bilingual.NOT_HEARD["en"])

    def test_english_walk_request_is_allowed_only_when_direct(self):
        self.engine.brain.reply = mock.Mock(return_value=("A short step forward.", "move_forward"))
        for text, allowed in (("Please walk forward", {"move_forward"}), ("How do you walk forward?", set())):
            with self.subTest(text=text), mock.patch.object(self.engine, "_speak") as speak:
                self.engine._chat(text, "文字")
                self.assertEqual(speak.call_args.kwargs["allowed_body"], allowed)

    def test_low_confidence_thresholds(self):
        low = self.engine._low_confidence
        self.assertTrue(low({"avg_logprob": -1.2}))
        self.assertTrue(low({"language_prob": .55}))
        self.assertFalse(low({"avg_logprob": -.4, "language_prob": .95}))
        self.assertFalse(low({}))


class QaCasesFileTests(unittest.TestCase):
    def test_cases_are_well_formed_and_cover_both_languages(self):
        data = json.loads((HERE / "qa" / "technical_qa_cases.json").read_text(encoding="utf-8"))
        ids = [case["id"] for case in data["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        known = set(data["checks"])
        for case in data["cases"]:
            with self.subTest(id=case["id"]):
                self.assertTrue(set(case["expect"]) <= known)
                self.assertEqual(rc.text_language(case["input"]), case["expect"]["language"])
        self.assertGreaterEqual(sum(i.startswith("en-") for i in ids), 8)

    def test_scorer(self):
        import qa_eval
        case = {"expect": {"language": "en", "include_any": [["stop"]], "confirm": True, "gesture_not": ["move_forward"]}}
        self.assertEqual(qa_eval.score(case, "Did you mean the stop button B?", "none"), [])
        # 言語違い・日本語混入・stop なし・聞き返しなし・禁止の身ぶり
        self.assertEqual(len(qa_eval.score(case, "B は停止です。", "move_forward")), 5)
        self.assertEqual(qa_eval.score({"expect": {"no_refusal": True}}, "Long ago this machine ate my lunch box!", "none"), [])
        self.assertEqual(len(qa_eval.score({"expect": {"no_refusal": True}}, "それは架空なので分からない。", "none")), 1)


if __name__ == "__main__":
    unittest.main()
