"""押下歩行の排他・世代番号・HTTP入口を、機体やネットワークなしで確認する。"""
import threading
import time
import unittest
from unittest import mock

import numpy as np

import test_engine_integration as fixtures


TOKEN_A = "hold-test-token-a-00000001"
TOKEN_B = "hold-test-token-b-00000002"


class HoldIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.EngineTests.setUp(self)
        self.finish = threading.Event()
        self.wait_entered = threading.Event()
        self.body_state = {"busy": False, "stop_required": False, "error": "",
                           "require_feedback": True, "feedback_valid": True, "hold": False}
        self.body = mock.Mock()
        self.body.state.side_effect = lambda: dict(self.body_state)
        self.body.start_hold.side_effect = self.start_body
        self.body.wait_until_idle.side_effect = self.wait_body
        self.body.stop.side_effect = self.stop_body
        self.body.heartbeat_hold.return_value = (True, "renewed")
        self.body.release_hold.side_effect = self.release_body
        self.engine.body = self.body
        self.engine.fsm, self.engine.fsm_ok = 811, True
        self.gesturer.thread = None
        self.addCleanup(self.finish_job)

    def start_body(self, direction, token, **kwargs):
        self.body_state["busy"] = True
        self.body_state["token"] = token
        self.finish.clear()
        return True, "accepted"

    def wait_body(self, timeout=None):
        self.wait_entered.set()
        done = self.finish.wait(3)
        if done:
            self.body_state["busy"] = False
        return done

    def stop_body(self):
        self.body_state["busy"] = False
        self.finish.set()
        return True, "stopped"

    def release_body(self, token):
        if token != self.body_state.get("token"):
            return False, "not this hold"
        return self.stop_body()

    def finish_job(self):
        self.finish.set()
        deadline = time.monotonic() + 1
        while self.engine.job_lock.locked() and time.monotonic() < deadline:
            time.sleep(.005)

    def start(self, action="move_forward", token=TOKEN_A):
        accepted = self.engine.body_hold_start(action, token, self.engine.hold_epoch)
        self.assertTrue(accepted[0], accepted)
        self.assertTrue(self.wait_entered.wait(1))
        return accepted

    def test_start_reserves_job_through_motion_and_passes_manual_speed_profile(self):
        self.assertEqual(self.engine.hold_linear_speed, .9)
        self.assertEqual(self.engine.hold_yaw_speed, 1.8)
        self.start()
        self.assertTrue(self.engine.job_lock.locked())
        self.assertTrue(self.engine.busy)
        self.assertFalse(self.engine._can_listen())
        self.body.start_hold.assert_called_once_with(
            "forward", TOKEN_A, linear_speed=self.engine.hold_linear_speed,
            yaw_speed=self.engine.hold_yaw_speed)
        self.finish_job()
        self.assertFalse(self.engine.job_lock.locked())
        self.assertEqual(self.engine.busy, "")

    def test_hold_blocks_second_tab_arm_stand_and_voice_jobs(self):
        self.start()
        self.assertFalse(self.engine.body_hold_start("move_left", TOKEN_B, self.engine.hold_epoch)[0])
        self.assertFalse(self.engine.gesture("wave")[0])
        self.assertFalse(self.engine.stand_up("supine", True)[0])
        self.engine._call_utterance(np.zeros(16000, dtype=np.int16), lambda: True)
        self.body.start_hold.assert_called_once()
        self.stt.transcribe.assert_not_called()
        self.gesturer.manual.assert_not_called()
        self.tts.synth.assert_not_called()

    def test_old_or_malformed_epoch_cannot_start_a_hold(self):
        for epoch in (self.engine.hold_epoch - 1, self.engine.hold_epoch + 1,
                      None, True, "0", 0.0):
            with self.subTest(epoch=epoch):
                self.assertFalse(self.engine.body_hold_start("move_forward", TOKEN_A, epoch)[0])
        self.body.start_hold.assert_not_called()
        self.assertFalse(self.engine.job_lock.locked())

    def test_global_stop_invalidates_a_start_that_arrives_late(self):
        old_epoch = self.engine.hold_epoch
        self.assertTrue(self.engine.body_stop()[0])
        self.assertGreater(self.engine.hold_epoch, old_epoch)
        self.assertFalse(self.engine.body_hold_start("move_forward", TOKEN_A, old_epoch)[0])
        self.body.start_hold.assert_not_called()
        self.body.stop.assert_called_once()

    def test_global_stop_invalidates_requests_before_waiting_for_stop_rpc(self):
        self.start()
        old_epoch = self.engine.hold_epoch
        entered, allow_stop = threading.Event(), threading.Event()

        def slow_stop():
            entered.set()
            allow_stop.wait(2)
            return self.stop_body()

        self.body.stop.side_effect = slow_stop
        stopper = threading.Thread(target=self.engine.body_stop)
        stopper.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertGreater(self.engine.hold_epoch, old_epoch)
            # A slow robot RPC must not hold the panel's control lock.
            acquired = self.engine.control_lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                self.engine.control_lock.release()
            self.assertFalse(self.engine.body_hold_start("move_forward", TOKEN_B, old_epoch)[0])
            self.body.start_hold.assert_called_once()
        finally:
            allow_stop.set()
            stopper.join(1)
        self.assertFalse(stopper.is_alive())
        self.finish_job()
        self.assertFalse(self.engine.job_lock.locked())

    def test_heartbeat_and_release_bypass_reserved_job_and_preserve_token(self):
        self.start()
        self.assertTrue(self.engine.body_hold_heartbeat(TOKEN_A)[0])
        self.body.heartbeat_hold.assert_called_once_with(TOKEN_A)
        self.assertTrue(self.engine.job_lock.locked())
        self.assertTrue(self.engine.body_hold_release(TOKEN_A)[0])
        self.body.release_hold.assert_called_once_with(TOKEN_A)
        self.finish_job()
        self.assertFalse(self.engine.job_lock.locked())

    def test_another_tabs_release_does_not_end_the_owned_hold(self):
        self.start()
        self.assertFalse(self.engine.body_hold_release(TOKEN_B)[0])
        self.body.release_hold.assert_called_once_with(TOKEN_B)
        self.assertTrue(self.engine.job_lock.locked())
        self.assertTrue(self.body_state["busy"])
        self.body.stop.assert_not_called()

    def test_rejected_controller_start_releases_reservation(self):
        self.body.start_hold.side_effect = None
        self.body.start_hold.return_value = (False, "stop is unconfirmed")
        accepted = self.engine.body_hold_start("move_forward", TOKEN_A, self.engine.hold_epoch)
        self.assertFalse(accepted[0])
        self.assertFalse(self.engine.job_lock.locked())
        self.assertEqual(self.engine.busy, "")
        self.body.wait_until_idle.assert_not_called()

    def test_controller_start_exception_does_not_leave_panel_locked(self):
        self.body.start_hold.side_effect = RuntimeError("test startup failure")
        with self.assertRaisesRegex(RuntimeError, "test startup failure"):
            self.engine.body_hold_start("move_forward", TOKEN_A, self.engine.hold_epoch)
        self.assertFalse(self.engine.job_lock.locked())
        self.assertEqual(self.engine.busy, "")

    def test_failed_watcher_start_releases_motion_before_unlocking_panel(self):
        watcher = mock.Mock()
        watcher.start.side_effect = RuntimeError("watcher unavailable")
        with mock.patch.object(fixtures.ui.threading, "Thread", return_value=watcher):
            with self.assertRaisesRegex(RuntimeError, "watcher unavailable"):
                self.engine.body_hold_start("move_forward", TOKEN_A, self.engine.hold_epoch)
        self.body.release_hold.assert_called_once_with(TOKEN_A)
        self.body.wait_until_idle.assert_called_once()
        self.assertFalse(self.body_state["busy"])
        self.assertFalse(self.engine.job_lock.locked())
        self.assertEqual(self.engine.busy, "")

    def test_non_running_or_unknown_physical_fsm_does_not_start(self):
        self.engine.args.local = False
        for fsm, known in ((0, True), (4, True), (816, True), (None, False), (811, False)):
            with self.subTest(fsm=fsm, known=known):
                self.engine.fsm, self.engine.fsm_ok = fsm, known
                self.assertFalse(self.engine.body_hold_start("move_forward", TOKEN_A, self.engine.hold_epoch)[0])
        self.body.start_hold.assert_not_called()

    def test_closed_disabled_recording_or_invalid_direction_does_not_start(self):
        for attribute, value in (("closed", True), ("body_enabled", False), ("rec_stream", object())):
            original = getattr(self.engine, attribute)
            try:
                setattr(self.engine, attribute, value)
                self.assertFalse(self.engine.body_hold_start("move_forward", TOKEN_A, self.engine.hold_epoch)[0])
            finally:
                setattr(self.engine, attribute, original)
        for action in ("", "fly", "stop_move", None):
            with self.subTest(action=action):
                self.assertFalse(self.engine.body_hold_start(action, TOKEN_A, self.engine.hold_epoch)[0])
        self.body.start_hold.assert_not_called()
        self.assertFalse(self.engine.job_lock.locked())

    def test_state_exposes_epoch_and_manual_profile(self):
        body = self.engine.state()["body"]
        self.assertEqual(body["hold_epoch"], self.engine.hold_epoch)
        self.assertEqual(body["hold_profile"], {"linear_speed": .9, "yaw_speed": 1.8,
                                                "heartbeat_ms": 200, "lease_ms": 750})

    def test_hold_speed_limits_reject_invalid_settings_before_devices(self):
        for value in (.90001, 0, -1, True, float("nan"), float("inf"), "0.9"):
            args = mock.Mock(hold_linear_speed=value, hold_yaw_speed=.3)
            with self.subTest(value=value), self.assertRaises(ValueError):
                fixtures.ui._hold_settings(args)
        for value in (1.80001, 0, -1, True, float("nan"), float("inf"), "1.8"):
            args = mock.Mock(hold_linear_speed=.9, hold_yaw_speed=value)
            with self.subTest(yaw=value), self.assertRaises(ValueError):
                fixtures.ui._hold_settings(args)


class HoldHandlerTests(unittest.TestCase):
    post = fixtures.HandlerTests.post

    def setUp(self):
        self.engine = mock.Mock(spec=fixtures.ui.Engine)
        for name in ("body_hold_start", "body_hold_heartbeat", "body_hold_release"):
            getattr(self.engine, name).return_value = (True, "ok")
        self.handler_class = fixtures.ui.make_handler(self.engine)

    def test_start_route_preserves_action_token_and_epoch(self):
        status, data, _ = self.post("/api/body/hold/start", {
            "action": "move_forward", "token": TOKEN_A, "epoch": 3})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.engine.body_hold_start.assert_called_once_with("move_forward", TOKEN_A, 3)

    def test_routes_do_not_coerce_epoch_or_token_types(self):
        for epoch in (True, "0", None):
            with self.subTest(epoch=epoch):
                self.engine.body_hold_start.reset_mock()
                self.post("/api/body/hold/start", {"action": "move_forward", "token": TOKEN_A, "epoch": epoch})
                actual = self.engine.body_hold_start.call_args.args
                self.assertIs(type(actual[2]), type(epoch))
                self.assertEqual(actual[2], epoch)
        token = {"not": "a token"}
        self.post("/api/body/hold/release", {"token": token})
        self.engine.body_hold_release.assert_called_once_with(token)

    def test_keepalive_and_release_routes_dispatch_only_matching_operation(self):
        for path, method in (("/api/body/hold/heartbeat", "body_hold_heartbeat"),
                             ("/api/body/hold/release", "body_hold_release")):
            with self.subTest(path=path):
                status, data, _ = self.post(path, {"token": TOKEN_A})
                self.assertEqual(status, 200)
                self.assertTrue(data["ok"])
                getattr(self.engine, method).assert_called_once_with(TOKEN_A)
        self.engine.body_hold_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
