"""押下歩行の継続・解除・期限切れを、偽クライアントだけで検証する。"""
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_motion"))
from body_control import BodyController


TOKEN = "a" * 32
OTHER = "b" * 32


class Clock:
    def __init__(self):
        self.t = 0.0
        self.hook = None

    def __call__(self):
        return self.t

    def wait(self, event, seconds):
        self.t += seconds
        if self.hook:
            self.hook()
        return event.is_set()


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.loco = Mock()
        self.loco.get_fsm.return_value = 811
        self.loco._Call.return_value = (0, "")

    def make(self, **kwargs):
        return BodyController(self.loco, dry=False, monotonic=self.clock,
                              wait=self.clock.wait, **kwargs)

    def payloads(self):
        return [json.loads(call.args[1]) for call in self.loco._Call.call_args_list]

    def test_held_input_refreshes_beyond_two_seconds_then_release_zeros(self):
        controller = self.make()
        next_heartbeat = [0.2]

        def held():
            if self.clock() >= 3:
                controller.release_hold(TOKEN)
            elif self.clock() >= next_heartbeat[0]:
                self.assertTrue(controller.heartbeat_hold(TOKEN)[0])
                next_heartbeat[0] += .2

        self.clock.hook = held
        self.assertTrue(controller.start_hold("forward", TOKEN)[0])
        self.assertTrue(controller.wait_until_idle(1))
        payloads = self.payloads()
        moving = [row for row in payloads if row["velocity"] != [0, 0, 0]]
        self.assertGreaterEqual(len(moving), 8)
        self.assertTrue(all(row == {"velocity": [.9, 0, 0], "duration": 2} for row in moving))
        self.assertEqual(payloads[-1]["velocity"], [0, 0, 0])
        self.assertFalse(controller.state()["busy"])
        self.assertFalse(controller.state()["hold_active"])
        self.assertFalse(controller.heartbeat_hold(TOKEN)[0])
        self.assertFalse(controller.start_hold("forward", TOKEN)[0])

    def test_missing_heartbeat_expires_without_reviving(self):
        controller = self.make()
        self.assertTrue(controller.start_hold("forward", TOKEN)[0])
        self.assertTrue(controller.wait_until_idle(1))
        self.assertLessEqual(self.clock(), .751)
        self.assertEqual(controller.state()["status"], "lease_expired")
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])
        before = len(self.payloads())
        self.assertFalse(controller.heartbeat_hold(TOKEN)[0])
        self.assertFalse(controller.start_hold("forward", TOKEN)[0])
        self.assertEqual(len(self.payloads()), before)

    def test_wrong_heartbeat_cannot_extend_lease(self):
        controller = self.make()
        self.clock.hook = lambda: controller.heartbeat_hold(OTHER)
        controller.start_hold("forward", TOKEN)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(controller.state()["status"], "lease_expired")
        self.assertLessEqual(self.clock(), .751)

    def test_release_before_start_retires_token_without_any_rpc(self):
        controller = self.make()
        self.assertTrue(controller.release_hold(TOKEN)[0])
        self.assertFalse(controller.start_hold("forward", TOKEN)[0])
        self.loco.get_fsm.assert_not_called()
        self.loco._Call.assert_not_called()

    def test_stale_release_does_not_stop_a_new_hold(self):
        controller = self.make()
        controller.release_hold(TOKEN)
        next_heartbeat = [.2]
        stale_releases = []

        def held():
            if .35 <= self.clock() < .45:
                stale_releases.append(controller.release_hold(TOKEN)[0])
            if self.clock() >= 1.2:
                controller.release_hold(OTHER)
            elif self.clock() >= next_heartbeat[0]:
                controller.heartbeat_hold(OTHER)
                next_heartbeat[0] += .2

        self.clock.hook = held
        controller.start_hold("backward", OTHER)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertTrue(stale_releases)
        payloads = self.payloads()
        self.assertGreaterEqual(sum(row["velocity"] == [-.9, 0, 0] for row in payloads), 3)
        self.assertEqual(sum(row["velocity"] == [0, 0, 0] for row in payloads), 1)

    def test_hold_profile_applies_to_each_direction_without_changing_single_move(self):
        for direction, expected in (("forward", [.9, 0, 0]), ("backward", [-.9, 0, 0]),
                                    ("left", [0, .9, 0]), ("right", [0, -.9, 0]),
                                    ("turn_left", [0, 0, 1.8]), ("turn_right", [0, 0, -1.8])):
            with self.subTest(direction=direction):
                self.loco._Call.reset_mock()
                controller = self.make()
                controller.start_hold(direction, TOKEN)
                self.assertTrue(controller.wait_until_idle(1))
                self.assertEqual(self.payloads()[0]["velocity"], expected)
                self.assertEqual(controller.linear_speed, .1)
                self.assertEqual(controller.forward_speed, .1)
                self.assertEqual(controller.yaw_speed, .2)
                self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_higher_hold_speed_does_not_raise_single_move_limit(self):
        with self.assertRaises(ValueError):
            self.make(linear_speed=.9)
        with self.assertRaises(ValueError):
            self.make(forward_speed=.9)
        with self.assertRaises(ValueError):
            self.make(yaw_speed=.9)
        controller = self.make(forward_speed=.3)
        self.assertTrue(controller.move("forward", 2)[0])
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(self.payloads(), [{"velocity": [.3, 0, 0], "duration": 2},
                                          {"velocity": [0, 0, 0], "duration": 1}])

    def test_explicit_old_hold_speed_remains_available(self):
        controller = self.make()
        self.assertTrue(controller.start_hold("forward", TOKEN, linear_speed=.3)[0])
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(self.payloads()[0]["velocity"], [.3, 0, 0])
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_explicit_previous_hold_yaw_remains_available(self):
        for direction, expected in (("turn_left", .9), ("turn_right", -.9)):
            with self.subTest(direction=direction):
                self.loco._Call.reset_mock()
                controller = self.make()
                self.assertTrue(controller.start_hold(direction, TOKEN, yaw_speed=.9)[0])
                self.assertTrue(controller.wait_until_idle(1))
                self.assertEqual(self.payloads()[0]["velocity"], [0, 0, expected])
                self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_single_move_yaw_boundary_remains_half_a_radian(self):
        with self.assertRaises(ValueError):
            self.make(yaw_speed=.50001)
        controller = self.make(yaw_speed=.5)
        self.assertTrue(controller.move("turn_left", .2)[0])
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(self.payloads(), [{"velocity": [0, 0, .5], "duration": .2},
                                          {"velocity": [0, 0, 0], "duration": 1}])

    def test_invalid_tokens_and_speeds_never_start(self):
        controller = self.make()
        for token in (None, True, 4, "short", "a" * 129, "日本語" * 20, "a b" * 10):
            self.assertFalse(controller.start_hold("forward", token)[0])
            self.assertFalse(controller.heartbeat_hold(token)[0])
            self.assertFalse(controller.release_hold(token)[0])
        for kwargs in ({"linear_speed": .90001}, {"linear_speed": 0}, {"linear_speed": -1},
                       {"linear_speed": True}, {"linear_speed": "0.9"},
                       {"linear_speed": float("nan")}, {"linear_speed": float("inf")},
                       {"yaw_speed": 1.80001}, {"yaw_speed": float("nan")}, {"yaw_speed": float("inf")}):
            self.assertFalse(controller.start_hold("forward", TOKEN, **kwargs)[0])
        self.assertFalse(controller.start_hold("stop", TOKEN)[0])
        self.loco.get_fsm.assert_not_called()
        self.loco._Call.assert_not_called()

    def test_wrong_fsm_during_refresh_cancels_then_zeros(self):
        self.loco.get_fsm.side_effect = [811, 816]
        controller = self.make()
        controller.start_hold("forward", TOKEN)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(controller.state()["status"], "rejected_fsm")
        self.assertEqual([row["velocity"] for row in self.payloads()], [[.9, 0, 0], [0, 0, 0]])

    def test_expired_while_initial_fsm_blocks_sends_no_nonzero(self):
        def fsm():
            self.clock.t = 1
            return 811

        self.loco.get_fsm.side_effect = fsm
        controller = self.make()
        controller.start_hold("forward", TOKEN)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(controller.state()["status"], "lease_expired")
        self.assertTrue(all(row["velocity"] == [0, 0, 0] for row in self.payloads()))

    def test_shared_motion_lock_does_not_queue_a_hold(self):
        lock = threading.Lock()
        lock.acquire()
        controller = self.make(motion_lock=lock)
        try:
            controller.start_hold("forward", TOKEN)
            self.assertTrue(controller.wait_until_idle(1))
            self.assertEqual(controller.state()["status"], "busy")
            self.loco._Call.assert_not_called()
            self.assertFalse(controller.start_hold("forward", TOKEN)[0])
        finally:
            lock.release()

    def test_stop_failure_blocks_new_hold_until_stop_retry(self):
        fail_zero = [True]

        def rpc(_api, payload):
            zero = json.loads(payload)["velocity"] == [0, 0, 0]
            return (3104 if zero and fail_zero[0] else 0), ""

        self.loco._Call.side_effect = rpc
        controller = self.make()
        controller.start_hold("forward", TOKEN)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertTrue(controller.state()["stop_required"])
        self.assertFalse(controller.start_hold("forward", OTHER)[0])
        fail_zero[0] = False
        self.assertTrue(controller.stop()[0])
        self.assertFalse(controller.state()["stop_required"])
        self.assertTrue(controller.start_hold("forward", OTHER)[0])
        self.assertTrue(controller.wait_until_idle(1))

    def test_code127_is_not_physical_confirmation(self):
        self.loco._Call.return_value = (127, "")
        controller = self.make(velocity_127_compatible=True)
        controller.start_hold("forward", TOKEN)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertTrue(controller.state()["physical_confirmation_required"])
        self.assertEqual(controller.state()["stop_code"], 127)

    def test_stop_blocks_renewal_and_overlap_while_worker_is_still_running(self):
        entered, release = threading.Event(), threading.Event()

        def wait(cancel, seconds):
            entered.set()
            release.wait(2)
            self.clock.t += seconds
            return cancel.is_set()

        controller = BodyController(self.loco, dry=False, monotonic=self.clock, wait=wait)
        try:
            controller.start_hold("forward", TOKEN)
            self.assertTrue(entered.wait(1))
            self.assertFalse(controller.move("forward")[0])
            self.assertFalse(controller.start_hold("left", OTHER)[0])
            self.assertTrue(controller.stop()[0])
            self.assertFalse(controller.heartbeat_hold(TOKEN)[0])
            self.assertFalse(controller.start_hold("left", OTHER)[0])
        finally:
            release.set()
            self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_release_during_fsm_read_cannot_be_followed_by_movement(self):
        entered, unblock = threading.Event(), threading.Event()

        def fsm():
            entered.set()
            unblock.wait(2)
            return 811

        self.loco.get_fsm.side_effect = fsm
        controller = self.make()
        controller.start_hold("forward", TOKEN)
        self.assertTrue(entered.wait(1))
        releaser = threading.Thread(target=controller.release_hold, args=(TOKEN,))
        releaser.start()
        self.assertTrue(controller._active.cancel.wait(1))
        unblock.set()
        releaser.join(1)
        self.assertFalse(releaser.is_alive())
        self.assertTrue(controller.wait_until_idle(1))
        self.assertTrue(all(row["velocity"] == [0, 0, 0] for row in self.payloads()))

    def test_idle_stop_in_flight_does_not_queue_a_new_hold_or_move(self):
        entered, release = threading.Event(), threading.Event()

        def rpc(_api, payload):
            if json.loads(payload)["velocity"] == [0, 0, 0]:
                entered.set()
                release.wait(2)
            return 0, ""

        self.loco._Call.side_effect = rpc
        controller = self.make()
        stopper = threading.Thread(target=controller.stop)
        stopper.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(controller.state()["stopping"])
            self.assertFalse(controller.start_hold("forward", TOKEN)[0])
            self.assertFalse(controller.move("forward")[0])
        finally:
            release.set()
            stopper.join(1)
        self.assertFalse(stopper.is_alive())
        self.assertFalse(controller.state()["stopping"])
        self.assertEqual([row["velocity"] for row in self.payloads()], [[0, 0, 0]])

    def test_feedback_loss_during_hold_sends_zero_and_verifies_quiet_window(self):
        stopped_at = [None]

        def rpc(_api, payload):
            if json.loads(payload)["velocity"] == [0, 0, 0]:
                stopped_at[0] = self.clock()
            return 0, ""

        def feedback():
            now = self.clock()
            moving = stopped_at[0] is None
            q = now * .15 if moving else stopped_at[0] * .15
            return {"t": now, "crc_valid": not (moving and now >= .4),
                    "mode_machine": 1, "mode_pr": 0, "motor_fault": [], "error": "",
                    "qlegs": [q] * 12, "dqlegs": [.2 if moving else 0] * 12}

        self.loco._Call.side_effect = rpc
        controller = self.make(feedback=feedback, require_feedback=True)
        controller.start_hold("forward", TOKEN)
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(controller.state()["status"], "feedback_lost")
        self.assertLessEqual(stopped_at[0], .451)
        self.assertTrue(controller.state()["joint_stationary_observed"])
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
