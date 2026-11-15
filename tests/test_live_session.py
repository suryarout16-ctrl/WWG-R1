"""Live-session source and greeting behavior without devices or network."""
import unittest
from unittest import mock

import test_engine_integration as fixtures


class LiveSessionTests(unittest.TestCase):
    def setUp(self):
        fixtures.EngineTests.setUp(self)

    def test_greeting_uses_llm_words_and_selected_gesture(self):
        words = "新人さんが操作の理由まで分かるよう、一緒に確かめるのが俺の役目だ。"
        brain = mock.Mock()
        brain.reply.return_value = (words, "hand_on_heart")
        self.engine.brains["mentor"] = brain
        self.engine._greet()
        self.assertEqual(brain.reply.call_args.args[0], "自己紹介して")
        self.tts.synth.assert_called_once_with(words)
        self.gesturer.start.assert_called_once_with("hand_on_heart")
        self.assertEqual(self.engine.history[-1]["text"], words)

    def test_state_distinguishes_local_virtual_and_robot(self):
        for local, interface, kind in ((True, None, "local"),
                                        (False, "Loopback Pseudo-Interface 1", "virtual"),
                                        (False, "Ethernet", "robot")):
            with self.subTest(kind=kind):
                self.engine.args.local, self.engine.args.interface = local, interface
                state = self.engine.state()
                self.assertEqual(state["connection_kind"], kind)
                self.assertEqual(state["virtual"], kind == "virtual")

    def test_virtual_runtime_context_does_not_claim_physical_robot(self):
        self.engine.args.local = False
        self.engine.args.interface = "Loopback Pseudo-Interface 1"
        brain = mock.Mock()
        brain.reply.return_value = ("作業の話をしよう。", "none")
        self.engine.brains["mentor"] = brain
        self.engine._chat("こんにちは", "文字")
        context = brain.set_runtime_context.call_args.args[0]
        self.assertIn("仮想R1", context)
        self.assertNotIn("実機R1接続モード", context)

    def test_explicit_dds_address_takes_precedence_over_interface_label(self):
        self.engine.args.local = False
        self.engine.args.interface = "Loopback Pseudo-Interface 1"
        self.engine.dds_address = "192.168.123.164"
        self.assertEqual(self.engine.connection_kind, "robot")
        self.assertFalse(self.engine.state()["virtual"])
        self.engine.args.interface = "Ethernet"
        self.engine.dds_address = "127.0.0.1"
        self.assertEqual(self.engine.connection_kind, "virtual")


if __name__ == "__main__":
    unittest.main()
