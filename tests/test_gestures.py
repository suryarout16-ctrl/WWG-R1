"""Motion lifecycle tests use fake RPC clients; no DDS or robot connection."""
import builtins
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gesture_chat", ROOT / "r1_chat" / "r1_chat.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


class GestureTests(unittest.TestCase):
    def setUp(self):
        self.log_patch = patch.object(rc, "log")
        self.log = self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.arm = types.SimpleNamespace(
            execute=Mock(return_value=0), release=Mock(return_value=0),
            execute_custom=Mock(return_value=0), stop_custom=Mock(return_value=0),
            action_list=Mock(return_value=(0, [[], []])),
        )
        self.loco = types.SimpleNamespace(get_fsm=Mock(return_value=811), SetTimeout=Mock(), Init=Mock())

    def make(self, **kwargs):
        arm_module = types.SimpleNamespace(
            ACTIONS={99: "release", 25: "wave", 26: "overhead wave", 29: "fist", 31: "extend forward"},
            connect=Mock(return_value=self.arm), describe=lambda code: str(code),
        )
        stand_module = types.SimpleNamespace(R1Loco=Mock(return_value=self.loco))
        with patch.dict(sys.modules, {"arm_action": arm_module, "stand": stand_module}):
            return rc.Gesturer(**kwargs)

    def custom(self, duration=0.02, name="hiyari_hat_v1"):
        self.arm.action_list.return_value = (0, [[], [{"name": name, "time": duration}]])

    def test_dry_hat_names_exact_action_without_hardware_imports(self):
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name in {"arm_action", "stand"} or name.startswith("unitree"):
                self.fail("dry mode imported hardware")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked_import):
            gesture = rc.Gesturer(dry=True, hat_action="hat_v2")
            self.assertEqual(gesture.start("hat_tip"), "dry")
            self.assertEqual(gesture.result["action_name"], "hat_v2")
            self.assertEqual(gesture.perform("hat_tip"), 0)
        self.assertIn("hat_v2", " ".join(str(call) for call in self.log.call_args_list))

    def test_missing_custom_skips_without_silent_substitution(self):
        gesture = self.make()
        self.assertEqual(gesture.start("hat_tip"), "unavailable_hat_action")
        self.arm.execute.assert_not_called()
        self.arm.execute_custom.assert_not_called()
        self.assertEqual(gesture.result["code"], 7402)

    def test_custom_name_match_is_exact(self):
        self.custom(name="hiyari_hat_v1_extra")
        gesture = self.make()
        self.assertEqual(gesture.start("hat_tip"), "unavailable_hat_action")
        self.arm.execute_custom.assert_not_called()

    def test_fallback_is_explicit_and_labelled_as_approximation(self):
        gesture = self.make(hat_fallback=True)
        self.assertEqual(gesture.start("hat_tip"), "sent_fallback")
        self.assertEqual(gesture.finish(), 0)
        self.arm.execute.assert_called_once_with(26)
        self.arm.execute_custom.assert_not_called()
        self.assertIn("代替", gesture.result["label"])

    def test_custom_waits_then_stops_and_returns_home(self):
        self.custom()
        gesture = self.make()
        entered, release = threading.Event(), threading.Event()
        completed = []

        def gated_sleep(duration):
            self.assertEqual(duration, 0.02)
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release playback gate")

        with patch.object(rc.time, "sleep", side_effect=gated_sleep):
            self.assertEqual(gesture.start("hat_tip"), "sent_custom")
            self.assertTrue(entered.wait(1))
            self.assertEqual(gesture.start("wave"), "busy")
            self.assertEqual(gesture.manual(25), 7400)
            self.arm.stop_custom.assert_not_called()
            finisher = threading.Thread(target=lambda: completed.append(gesture.finish()))
            finisher.start()
            try:
                self.arm.stop_custom.assert_not_called()
            finally:
                release.set()
                finisher.join(1)
            self.assertFalse(finisher.is_alive())
            self.assertEqual(completed, [0])
        self.arm.execute_custom.assert_called_once_with("hiyari_hat_v1")
        self.arm.stop_custom.assert_called_once_with()
        self.arm.execute.assert_not_called()

    def test_missing_invalid_or_excessive_duration_is_rejected(self):
        for duration in (None, False, 0, -1, 10.01, float("nan"), "2"):
            with self.subTest(duration=duration):
                self.custom(duration)
                gesture = self.make(hat_fallback=True)
                self.assertEqual(gesture.start("hat_tip"), "invalid_hat_duration")
        self.arm.execute_custom.assert_not_called()
        self.arm.execute.assert_not_called()

    def test_discovery_error_does_not_fall_back(self):
        self.arm.action_list.side_effect = RuntimeError("offline")
        gesture = self.make(hat_fallback=True)
        self.assertEqual(gesture.start("hat_tip"), "list_error")
        self.arm.execute.assert_not_called()
        self.assertIn("offline", gesture.result["error"])

    def test_timeout_keeps_live_worker_and_refuses_overlap(self):
        unblock = threading.Event()
        self.arm.execute.side_effect = lambda _action: 0 if unblock.wait(1) else 3104
        gesture = self.make()
        gesture.FINISH_TIMEOUT = 0.005
        gesture.start("wave")
        try:
            self.assertEqual(gesture.finish(), 3104)
            self.assertTrue(gesture.thread.is_alive())
            self.assertEqual(gesture.start("wave_big"), "busy")
            self.assertEqual(gesture.manual(25), 7400)
            self.assertEqual(gesture.manual(99), 7400)
        finally:
            unblock.set()
            gesture.FINISH_TIMEOUT = 1
            self.assertEqual(gesture.finish(), 0)
        self.assertIsNone(gesture.thread)

    def test_failed_release_keeps_lifecycle_busy_until_retry(self):
        self.custom()
        self.arm.stop_custom.side_effect = [3104, 0]
        gesture = self.make()
        gesture.start("hat_tip")
        self.assertEqual(gesture.finish(), 3104)
        self.assertIsNotNone(gesture.thread)
        self.assertEqual(gesture.start("wave"), "busy")
        self.assertEqual(gesture.finish(), 0)
        self.assertIsNone(gesture.thread)

    def test_worker_exception_is_recorded(self):
        self.arm.execute.side_effect = RuntimeError("RPC failed")
        gesture = self.make()
        gesture.start("wave")
        self.assertEqual(gesture.finish(), 7399)
        self.assertIn("RPC failed", gesture.result["error"])

    def test_release_button_retries_completed_custom_cleanup(self):
        self.custom()
        self.arm.stop_custom.side_effect = [3104, 3104, 0]
        gesture = self.make()
        gesture.start("hat_tip")
        self.assertEqual(gesture.finish(), 3104)
        self.assertEqual(gesture.manual(99), 3104)
        self.assertIsNotNone(gesture.thread)
        self.assertEqual(gesture.manual(99), 0)
        self.assertIsNone(gesture.thread)
        self.assertEqual(self.arm.stop_custom.call_count, 3)
        self.arm.execute.assert_not_called()

    def test_release_button_cleans_completed_hold_without_duplicate_release(self):
        gesture = self.make()
        gesture.start("fist_pump")
        gesture.thread.join(1)
        self.assertEqual(gesture.manual(99), 0)
        self.arm.execute.assert_called_once_with(29)
        self.arm.release.assert_called_once()
        self.assertIsNone(gesture.thread)

    def test_forward_point_uses_existing_preset_and_releases_before_next_action(self):
        gesture = self.make()
        self.assertEqual(gesture.start("point_forward"), "sent")
        self.assertEqual(gesture.finish(), 0)
        self.arm.execute.assert_called_once_with(31)
        self.arm.release.assert_called_once_with()
        self.arm.execute_custom.assert_not_called()
        self.assertIsNone(gesture.thread)
        self.assertEqual(gesture.result["release_code"], 0)
        self.assertEqual(gesture.start("wave"), "sent")
        self.assertEqual(gesture.finish(), 0)
        self.assertEqual([call.args for call in self.arm.execute.call_args_list], [(31,), (25,)])

    def test_release_button_can_recover_after_completed_execution_error(self):
        self.arm.execute.side_effect = [RuntimeError("RPC failed"), 0]
        gesture = self.make()
        gesture.start("wave")
        gesture.thread.join(1)
        self.assertEqual(gesture.manual(99), 0)
        self.assertEqual([call.args for call in self.arm.execute.call_args_list], [(25,), (99,)])
        self.assertIsNone(gesture.thread)

    def test_manual_retry_requires_successful_release(self):
        self.arm.execute.return_value = 7401
        self.arm.release.return_value = 7407
        gesture = self.make()
        self.assertEqual(gesture.manual(25), 7407)
        self.arm.execute.assert_called_once_with(25)

    def test_manual_retries_held_action_once(self):
        self.arm.execute.side_effect = [7401, 0]
        gesture = self.make()
        self.assertEqual(gesture.manual(25), 0)
        self.assertEqual(self.arm.execute.call_count, 2)
        self.arm.release.assert_called_once()

    def test_manual_and_start_share_motion_lock(self):
        entered, unblock = threading.Event(), threading.Event()

        def execute(_action):
            entered.set()
            return 0 if unblock.wait(1) else 3104

        self.arm.execute.side_effect = execute
        gesture = self.make()
        manual = threading.Thread(target=gesture.manual, args=(25,))
        manual.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(gesture.lock.acquire(blocking=False))
        unblock.set()
        manual.join(1)
        self.assertFalse(manual.is_alive())
        self.assertEqual(gesture.start("wave"), "sent")
        self.assertEqual(gesture.finish(), 0)

    def test_bad_fsm_prevents_custom_discovery_or_motion(self):
        self.loco.get_fsm.return_value = 1
        gesture = self.make()
        self.assertEqual(gesture.start("hat_tip"), "skip_fsm_1")
        self.arm.action_list.assert_not_called()
        self.arm.execute_custom.assert_not_called()
        self.assertEqual(gesture.manual(25), 7404)


class SpeakerCancellationTests(unittest.TestCase):
    def robot(self):
        speaker = object.__new__(rc.RobotSpeaker)
        speaker.client = types.SimpleNamespace(PlayStream=Mock(return_value=(0, None)), PlayStop=Mock())
        return speaker

    def test_robot_cancel_before_send_does_not_touch_service(self):
        speaker = self.robot()
        speaker.play(b"\0\0", should_continue=lambda: False)
        speaker.client.PlayStream.assert_not_called()
        speaker.client.PlayStop.assert_not_called()

    def test_robot_cancel_between_chunks_stops_stream(self):
        speaker = self.robot()
        active = iter((True, True, False))
        speaker.play(b"\0" * (rc.CHUNK_BYTES + 2), should_continue=lambda: next(active))
        self.assertEqual(speaker.client.PlayStream.call_count, 1)
        speaker.client.PlayStop.assert_called_once()

    def test_robot_cancel_during_playback_wait(self):
        speaker = self.robot()
        active = iter((True, True, False))
        with patch.object(rc.time, "sleep") as sleep:
            speaker.play(b"\0\0", should_continue=lambda: next(active))
        sleep.assert_not_called()
        speaker.client.PlayStop.assert_called_once()

    def test_robot_default_wait_remains_compatible(self):
        speaker = self.robot()
        with patch.object(rc.time, "sleep") as sleep:
            speaker.play(b"\0\0")
        sleep.assert_called_once()
        speaker.client.PlayStop.assert_called_once()

    def test_robot_rpc_exception_still_stops_attempted_stream(self):
        speaker = self.robot()
        speaker.client.PlayStream.side_effect = RuntimeError("lost response")
        with self.assertRaises(RuntimeError):
            speaker.play(b"\0\0", should_continue=lambda: True)
        speaker.client.PlayStop.assert_called_once()

    def test_local_cancel_stops_only_started_playback(self):
        sd = types.SimpleNamespace(play=Mock(), get_stream=Mock(return_value=types.SimpleNamespace(active=True)), stop=Mock(), wait=Mock())
        active = iter((True, False))
        with patch.dict(sys.modules, {"sounddevice": sd}):
            rc.LocalSpeaker().play(b"\0\0", should_continue=lambda: next(active))
        sd.play.assert_called_once()
        sd.stop.assert_called_once()
        sd.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
