"""仮想台本の検証と実行順を偽物のbackendで検証する。DDSを初期化しない。"""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import virtual_scenario as scenario


def document(steps=None):
    return {"schema_version": 1, "mode": "virtual_only", "id": "demo", "title": "模擬研修",
            "notes": ["架空の指示書。実機の検証結果ではない。"], "steps": steps or [
                {"id": "intro", "title": "あいさつ", "role": "r1", "say": "よろしくな。", "action": "wave"},
                {"id": "walk", "title": "前へ", "role": "r1", "say": "少し前へ進むぞ。",
                 "action": "move_forward", "duration_s": 2},
                {"id": "learner", "title": "新人の返事", "role": "learner", "say": "よろしくお願いします。"},
                {"id": "hat", "title": "ヒヤリハット", "role": "r1", "say": "ヒヤリハットだ。", "action": "hat_tip"}]}


class FakeBackend:
    def __init__(self, *, fail_speech=False, fsm=811, cancel_at_speech=None):
        self.calls, self.active = [], None
        self.fail_speech, self.fsm, self.cancel_at_speech = fail_speech, fsm, cancel_at_speech

    def require_running(self):
        self.calls.append(("fsm", self.fsm))
        if self.fsm != 811:
            raise RuntimeError("not Running")
        return self.fsm

    def prepare_speech(self, say):
        self.calls.append(("prepare", say))
        return say

    def speak(self, prepared, cancel):
        self.calls.append(("speak", prepared, self.active))
        if self.cancel_at_speech is not None:
            self.cancel_at_speech.set()
        if self.fail_speech:
            raise OSError("speaker failed")

    def start_action(self, action, duration=None):
        if self.active is not None:
            raise AssertionError("前の動作がまだ占有中")
        self.active = action
        self.calls.append(("start", action, duration))
        return {"code": 127 if action == "move_forward" else 0}

    def finish_action(self, action):
        if self.active != action:
            raise AssertionError("動作を開始せず終了した")
        if action == "move_forward":
            self.calls.append(("zero",))
        elif action in scenario.ARM_ACTIONS:
            self.calls.append(("release", action))
        self.calls.append(("finish", action))
        self.active = None
        return {"code": 127 if action == "move_forward" else 0, "stop_code": 127 if action == "move_forward" else None}

    def abort(self):
        self.calls.append(("abort",))
        if self.active == "move_forward":
            self.calls.append(("zero",))
        elif self.active in scenario.ARM_ACTIONS:
            self.calls.append(("release", self.active))
        self.active = None
        return {"cleaned": True}

    def close(self):
        self.calls.append(("close",))


class ScenarioTests(unittest.TestCase):
    def test_step_ranges_are_inclusive_ordered_and_keep_original_story(self):
        original = document()
        for start, end, expected in ((None, None, ["intro", "walk", "learner", "hat"]),
                                     ("walk", "learner", ["walk", "learner"]),
                                     (None, "walk", ["intro", "walk"]),
                                     ("learner", None, ["learner", "hat"]),
                                     ("walk", "walk", ["walk"])):
            with self.subTest(start=start, end=end):
                selected, metadata = scenario.select_step_range(original, start, end)
                self.assertEqual([step["id"] for step in selected["steps"]], expected)
                self.assertEqual(selected["id"], original["id"])
                self.assertEqual(selected["notes"], original["notes"])
                self.assertEqual(metadata["start_step"], expected[0])
                self.assertEqual(metadata["end_step"], expected[-1])
                self.assertEqual(metadata["selected_step_count"], len(expected))
                self.assertEqual(metadata["source_step_count"], len(original["steps"]))
                self.assertEqual(metadata["explicit"], start is not None or end is not None)
                scenario.validate_scenario(selected)
                selected["steps"][0]["title"] = "変更"
        self.assertEqual(original, document())

    def test_invalid_range_or_invalid_excluded_step_never_constructs_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.json"
            for options, data in ((["--start-step", "missing"], document()),
                                  (["--end-step", "missing"], document()),
                                  (["--start-step", "hat", "--end-step", "intro"], document()),
                                  (["--end-step", "intro"], document())):
                if options == ["--end-step", "intro"]:
                    data["steps"][-1]["action"] = "lowcmd"
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                with self.subTest(options=options), patch.object(scenario, "VirtualBackend") as factory, \
                        patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                    scenario.main([str(path), "--run-virtual", *options])
                self.assertEqual(error.exception.code, 2)
                factory.assert_not_called()

    def test_selected_cli_run_only_dispatches_selected_steps_and_records_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.json"
            path.write_text(json.dumps(document(), ensure_ascii=False), encoding="utf-8")
            backend = FakeBackend()
            with patch.object(scenario, "REPO_ROOT", Path(folder)), patch.object(scenario, "VirtualBackend", return_value=backend), \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(scenario.main([str(path), "--run-virtual", "--start-step", "walk", "--end-step", "learner"]), 0)
            reports = list((Path(folder) / "logs").glob("virtual_scenario_*.json"))
            report = json.loads(reports[0].read_text("utf-8"))
            self.assertEqual([step["id"] for step in report["steps"]], ["walk", "learner"])
            self.assertEqual(report["selected_range"], {"start_step": "walk", "end_step": "learner",
                                                       "source_start_index": 2, "source_end_index": 3,
                                                       "source_step_count": 4, "selected_step_count": 2, "explicit": True})
            self.assertIn(("start", "move_forward", 2), backend.calls)
            self.assertNotIn(("start", "wave", None), backend.calls)

    def test_manual_cues_precede_prepare_and_actions_for_every_role(self):
        steps = [
            {"id": "narrator", "title": "進行役の説明", "role": "stage", "say": "研修を始めます。"},
            {"id": "point", "title": "ボタン役の配置を示す", "role": "r1", "say": "こちらだ。", "action": "point_forward"},
            document()["steps"][1], document()["steps"][2],
        ]
        backend = FakeBackend()

        def before_step(step):
            self.assertIsNone(backend.active)
            backend.calls.append(("cue", step["id"]))
            # 合図側へ渡した値を変更しても、検証済みの実行内容は変わらない。
            step["action"] = "lowcmd"

        result = scenario.run_scenario(document(steps), backend, emit=lambda _: None, before_step=before_step)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["manual_cues"])
        self.assertEqual([call[1] for call in backend.calls if call[0] == "cue"], [step["id"] for step in steps])
        cue_point = backend.calls.index(("cue", "point"))
        prepare_point = backend.calls.index(("prepare", "こちらだ。"))
        release_point = backend.calls.index(("release", "point_forward"))
        cue_walk = backend.calls.index(("cue", "walk"))
        self.assertLess(cue_point, prepare_point)
        self.assertLess(release_point, cue_walk)
        self.assertIn(("start", "point_forward", None), backend.calls)
        self.assertIn(("start", "move_forward", 2), backend.calls)
        self.assertFalse(any(call[0] == "prepare" and call[1] in {"研修を始めます。", "よろしくお願いします。"} for call in backend.calls))
        for item in result["steps"]:
            self.assertEqual([event["event"] for event in item["events"][:2]], ["cue_wait", "cue_accepted"])

    def test_interrupted_cue_never_dispatches_the_queued_walk(self):
        backend = FakeBackend()

        def before_step(step):
            if step["id"] == "walk":
                raise KeyboardInterrupt

        result = scenario.run_scenario(document(), backend, emit=lambda _: None, before_step=before_step)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["steps"][-1]["id"], "walk")
        self.assertIn(("release", "wave"), backend.calls)
        self.assertNotIn(("start", "move_forward", 2), backend.calls)
        self.assertNotIn(("zero",), backend.calls)
        self.assertEqual(backend.calls[-2:], [("abort",), ("close",)])

    def test_eof_or_cancelled_cue_does_not_prepare_speech_or_start(self):
        for mode in ("eof", "cancel_event"):
            with self.subTest(mode=mode):
                cancel, backend = threading.Event(), FakeBackend()

                def before_step(step):
                    if mode == "eof":
                        raise EOFError("input closed")
                    cancel.set()

                result = scenario.run_scenario(document(), backend, emit=lambda _: None,
                                               before_step=before_step, cancel=cancel)
                self.assertEqual(result["status"], "cancelled")
                self.assertEqual(backend.calls, [("abort",), ("close",)])

    def test_interactive_cli_prompts_and_records_cues_without_dds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.json"
            path.write_text(json.dumps(document(), ensure_ascii=False), encoding="utf-8")
            backend = FakeBackend()
            with patch.object(scenario, "REPO_ROOT", Path(folder)), patch.object(scenario, "VirtualBackend", return_value=backend) as factory, \
                    patch("builtins.input", return_value="") as cue, patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(scenario.main([str(path), "--run-virtual", "--interactive"]), 0)
            factory.assert_called_once_with(pc_audio=False)
            self.assertEqual(cue.call_count, len(document()["steps"]))
            reports = list((Path(folder) / "logs").glob("virtual_scenario_*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text("utf-8"))
            self.assertTrue(report["interactive"])
            self.assertTrue(report["manual_cues"])
            self.assertFalse(report["pc_audio"])

    def test_speech_overlaps_action_and_release_zero_precede_next_step(self):
        backend = FakeBackend()
        result = scenario.run_scenario(document(), backend, emit=lambda _: None)
        self.assertEqual(result["status"], "completed")
        calls = backend.calls
        start_wave, speak_wave, finish_wave = (next(i for i, call in enumerate(calls) if call[:2] == key)
                                              for key in (("start", "wave"), ("speak", "よろしくな。"), ("finish", "wave")))
        start_walk = next(i for i, call in enumerate(calls) if call[:2] == ("start", "move_forward"))
        speak_walk = next(call for call in calls if call[:2] == ("speak", "少し前へ進むぞ。"))
        self.assertLess(start_wave, speak_wave)
        self.assertLess(speak_wave, finish_wave)
        self.assertLess(finish_wave, start_walk)
        self.assertEqual(speak_walk[2], "move_forward")
        self.assertIn(("zero",), calls)
        self.assertIn(("release", "wave"), calls)
        self.assertEqual(result["steps"][1]["result"]["stop_code"], 127)
        self.assertEqual(result["mode"], "virtual_only")
        self.assertEqual(result["physical_result"], "not_tested")
        self.assertEqual(result["notes"], document()["notes"])
        self.assertFalse(any(call[0] == "speak" and call[1] == "よろしくお願いします。" for call in calls))
        self.assertFalse(any(call[:2] == ("start", "hat_tip") for call in calls))
        self.assertEqual(result["steps"][-1]["status"], "placeholder")
        self.assertFalse(result["steps"][-1]["implemented"])

    def test_error_during_speech_stops_started_forward_and_skips_later_steps(self):
        steps = document()["steps"][1:]
        backend = FakeBackend(fail_speech=True)
        result = scenario.run_scenario(document(steps), backend, emit=lambda _: None)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["steps"]), 1)
        self.assertIn(("zero",), backend.calls)
        self.assertIsNone(backend.active)
        self.assertEqual(backend.calls[-1], ("close",))

    def test_cancellation_during_arm_speech_releases_and_skips_walk(self):
        cancel = threading.Event()
        backend = FakeBackend(cancel_at_speech=cancel)
        result = scenario.run_scenario(document(), backend, emit=lambda _: None, cancel=cancel)
        self.assertEqual(result["status"], "cancelled")
        self.assertIn(("release", "wave"), backend.calls)
        self.assertNotIn(("start", "move_forward", 2), backend.calls)
        self.assertNotIn(("zero",), backend.calls)

    def test_non_running_never_starts_action_or_speech(self):
        backend = FakeBackend(fsm=0)
        result = scenario.run_scenario(document(), backend, emit=lambda _: None)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(any(call[0] in {"start", "speak", "zero", "release"} for call in backend.calls))

    def test_entire_scenario_validated_before_any_backend_call(self):
        data = document()
        data["steps"][-1]["action"] = "lowcmd"
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            scenario.run_scenario(data, backend, emit=lambda _: None)
        self.assertEqual(backend.calls, [])

    def test_strict_schema_rejects_unsafe_or_nonfinite_arguments(self):
        cases = []
        for field, value in (("mode", "hardware"), ("schema_version", True), ("interface", "Ethernet")):
            data = document(); data[field] = value; cases.append(data)
        for field, value in (("duration_s", float("nan")), ("duration_s", float("inf")),
                             ("duration_s", True), ("duration_s", 2.1), ("duration_s", 0),
                             ("pause_after_s", -1), ("pause_after_s", 5.1), ("pause_after_s", True),
                             ("action", "move_left"), ("velocity", [1, 0, 0])):
            data = document(); data["steps"][1][field] = value; cases.append(data)
        data = document(); data["steps"][2]["action"] = "wave"; cases.append(data)
        data = document(); data["steps"][0]["duration_s"] = 1; cases.append(data)
        data = document(); data["steps"][0]["say_ref"] = "self_introduction"; cases.append(data)
        data = document(); data["steps"][1]["id"] = data["steps"][0]["id"]; cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                scenario.validate_scenario(data)

    def test_preview_uses_fixed_intro_notes_and_never_constructs_backend(self):
        data = document([{ "id": "intro", "title": "自己紹介", "role": "r1", "say_ref": "self_introduction"}])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with patch.object(scenario, "VirtualBackend", side_effect=AssertionError("送信しない")), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(scenario.main([str(path)]), 0)
        text = output.getvalue()
        self.assertIn("工場で新人に機械の操作を教える", text)
        self.assertIn(data["notes"][0], text)
        self.assertIn("送信なし", text)

    def test_loader_rejects_duplicates_file_size_and_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.json"
            for raw in ('{"mode":"virtual_only","mode":"hardware"}', '{"x":NaN}', ' ' * (scenario.MAX_FILE_BYTES + 1)):
                with self.subTest(raw=raw[:60]):
                    path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        scenario.load_scenario(path)

    def test_hardware_interface_cli_and_pc_audio_preview_are_rejected(self):
        for argv in (["-i", "Ethernet"], ["--pc-audio"], ["--interactive"]):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                scenario.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_preexisting_dds_module_is_not_reused(self):
        with patch.dict(sys.modules, {"unitree_sdk2py.core.channel": object()}):
            with self.assertRaisesRegex(RuntimeError, "既存のDDS接続"):
                scenario.VirtualBackend()


if __name__ == "__main__":
    unittest.main()
