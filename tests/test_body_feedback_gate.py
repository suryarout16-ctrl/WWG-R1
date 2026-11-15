"""実機を使わず、観測鮮度・異常時停止・停止後の静止窓を検証する。"""
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_motion"))
from body_control import BodyController


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def wait(self, _event, seconds):
        self.t += seconds


class FeedbackGateTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stop_t = None
        self.fault = None
        self.loco = Mock()
        self.loco.get_fsm.return_value = 811
        self.loco._Call.side_effect = self.rpc
        self.native_code = 0

    def rpc(self, _api, payload):
        if json.loads(payload)["velocity"] == [0, 0, 0]:
            self.stop_t = self.clock()
        return self.native_code, ""

    def feedback(self):
        now = self.clock()
        moving = self.stop_t is None
        q = (now if moving else self.stop_t) * .15
        data = {"t": now, "crc_valid": True, "tick": int(now * 1000), "mode_machine": 1,
                "mode_pr": 0, "qlegs": [q] * 12, "dqlegs": [.2 if moving else 0] * 12,
                "motor_fault": [], "error": ""}
        if self.fault:
            data.update(self.fault(now, moving))
        return data

    def make(self, feedback=None, **kwargs):
        return BodyController(self.loco, dry=False, require_feedback=True,
                              feedback=feedback or self.feedback, monotonic=self.clock,
                              wait=self.clock.wait, **kwargs)

    def payloads(self):
        return [json.loads(call.args[1]) for call in self.loco._Call.call_args_list]

    def run_move(self, controller):
        self.assertTrue(controller.move("forward", .2)[0])
        self.assertTrue(controller.wait_until_idle(1))
        return controller.state()

    def test_stop_needs_unique_postzero_quiet_samples_for_half_second(self):
        controller = self.make()
        state = self.run_move(controller)
        self.assertTrue(state["joint_motion_observed"])
        self.assertTrue(state["joint_stationary_observed"])
        self.assertFalse(state["stop_required"])
        self.assertFalse(state["physical_confirmation_required"])
        self.assertGreaterEqual(self.clock() - self.stop_t, .5)
        self.assertEqual(state["feedback_status"], "joint_stationary_observed")

    def test_bad_initial_feedback_never_sends_a_command(self):
        for bad in ({"crc_valid": False}, {"t": -1}, {"mode_machine": 2}, {"mode_pr": 100},
                    {"motor_fault": [{"joint": 25, "state": 1}]}, {"dqlegs": [float("nan")] * 12},
                    {"qlegs": [0] * 11}, {"motor_fault": None}):
            with self.subTest(bad=bad):
                self.fault = lambda _now, _moving, changes=bad: changes
                controller = self.make()
                self.assertFalse(controller.move("forward")[0])
        self.loco._Call.assert_not_called()

    def test_observer_exception_before_motion_is_not_healthy(self):
        controller = self.make(feedback=Mock(side_effect=OSError("observer offline")))
        self.assertFalse(controller.move("forward")[0])
        self.loco._Call.assert_not_called()

    def test_feedback_is_rechecked_after_blocking_fsm_read(self):
        snapshot = self.feedback()

        def fsm():
            self.clock.t += .6
            return 811

        self.loco.get_fsm.side_effect = fsm
        controller = self.make(feedback=lambda: snapshot)
        state = self.run_move(controller)
        self.assertTrue(all(p["velocity"] == [0, 0, 0] for p in self.payloads()))
        self.assertTrue(state["stop_required"])
        self.assertFalse(state["joint_stationary_observed"])

    def test_fault_during_motion_cancels_and_zeros(self):
        self.fault = lambda now, moving: {"motor_fault": [1]} if moving and now >= .1 else {}
        controller = self.make()
        state = self.run_move(controller)
        self.assertEqual(state["status"], "feedback_lost")
        self.assertLessEqual(self.stop_t, .11)
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])
        self.assertTrue(state["joint_stationary_observed"])

    def test_observer_loss_after_send_stops_and_latches(self):
        def feedback():
            if self.clock() >= .1:
                raise OSError("lost")
            return self.feedback()

        controller = self.make(feedback=feedback)
        state = self.run_move(controller)
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])
        self.assertTrue(state["stop_required"])
        self.assertFalse(state["joint_stationary_observed"])
        self.assertTrue(state["physical_confirmation_required"])
        self.assertNotIn("127", state["response_note"])

    def test_stale_or_pre_stop_repeated_samples_cannot_prove_stop(self):
        for reuse in ("old", "zero_time"):
            with self.subTest(reuse=reuse):
                self.stop_t = None
                self.clock.t = 0
                self.fault = lambda _now, moving: {} if moving else {"t": -1 if reuse == "old" else self.stop_t}
                controller = self.make()
                state = self.run_move(controller)
                self.assertTrue(state["stop_required"])
                self.assertFalse(state["joint_stationary_observed"])
                self.assertFalse(controller.move("forward")[0])
                self.assertLessEqual(self.clock() - self.stop_t, 2.001)

    def test_nonquiet_velocity_or_position_span_blocks_next_move(self):
        for kind in ("velocity", "position"):
            with self.subTest(kind=kind):
                self.stop_t = None
                self.clock.t = 0

                def changes(now, moving):
                    if moving:
                        return {}
                    return {"dqlegs": [.13] * 12} if kind == "velocity" else {"qlegs": [now * .1] * 12, "dqlegs": [0] * 12}

                self.fault = changes
                controller = self.make()
                state = self.run_move(controller)
                self.assertTrue(state["stop_required"])
                self.assertFalse(state["joint_stationary_observed"])

    def test_idle_stop_sends_zero_before_feedback_check_and_can_retry_latch(self):
        self.fault = lambda _now, _moving: {"crc_valid": False}
        controller = self.make()
        self.assertFalse(controller.stop()[0])
        self.assertEqual(self.payloads()[0]["velocity"], [0, 0, 0])
        self.assertTrue(controller.state()["stop_required"])
        self.fault = None
        self.assertTrue(controller.stop()[0])
        self.assertFalse(controller.state()["stop_required"])
        self.assertTrue(controller.state()["joint_stationary_observed"])

    def test_rpc_127_preserved_separately_from_joint_observations(self):
        self.native_code = 127
        controller = self.make(velocity_127_compatible=True)
        state = self.run_move(controller)
        self.assertEqual((state["code"], state["stop_code"]), (127, 127))
        self.assertTrue(state["joint_stationary_observed"])
        self.assertTrue(state["joint_motion_observed"])
        self.assertTrue(state["physical_confirmation_required"])
        self.assertFalse(state["stop_required"])

    def test_prior_quiet_is_cleared_as_soon_as_next_move_is_accepted(self):
        controller = self.make()
        self.run_move(controller)
        entered, release = threading.Event(), threading.Event()

        def fsm():
            entered.set()
            release.wait(1)
            return 811

        self.loco.get_fsm.side_effect = fsm
        self.stop_t = None
        self.assertTrue(controller.move("forward", .2)[0])
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(controller.state()["joint_stationary_observed"])
            self.assertTrue(controller.state()["physical_confirmation_required"])
        finally:
            release.set()
        self.assertTrue(controller.wait_until_idle(1))

    def test_precommand_sway_is_not_counted_as_command_response(self):
        def feedback():
            data = self.feedback()
            data.update(qlegs=[0] * 12, dqlegs=[.2 if self.clock() == 0 else 0] * 12)
            return data

        controller = self.make(feedback=feedback)
        state = self.run_move(controller)
        self.assertFalse(state["joint_motion_observed"])
        self.assertTrue(state["joint_stationary_observed"])
        self.assertTrue(state["physical_confirmation_required"])
        self.assertEqual(state["response_note"], "脚の動作は未確認")

    def test_cached_quiet_does_not_hide_new_stale_fault_or_motion(self):
        controller = self.make()
        self.run_move(controller)
        observed_at = controller.state()["joint_stationary_observed_at"]
        for change in ({"t": -1}, {"crc_valid": False}, {"motor_fault": [1]}, {"dqlegs": [.2] * 12}):
            with self.subTest(change=change):
                self.fault = lambda _now, _moving, values=change: values
                state = controller.state()
                self.assertFalse(state["joint_stationary_observed"])
                self.assertTrue(state["physical_confirmation_required"])
                self.assertEqual(state["joint_stationary_observed_at"], observed_at)


if __name__ == "__main__":
    unittest.main()
