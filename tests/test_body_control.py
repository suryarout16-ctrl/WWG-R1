"""速度指令は偽クライアントだけで検証し、DDSや実機へ接続しない。"""
import json
import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_motion"))
from body_control import BodyController, requested_body_actions
import test_engine_integration as engine_tests


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def wait(self, event, seconds):
        self.value += seconds
        return event.is_set()


class BodyTests(unittest.TestCase):
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

    def test_bounded_velocity_and_final_zero_use_official_r1_payload(self):
        for direction, vector in (("forward", [.1, 0, 0]), ("backward", [-.1, 0, 0]),
                                  ("left", [0, .1, 0]), ("right", [0, -.1, 0]),
                                  ("turn_left", [0, 0, .2]), ("turnright", [0, 0, -.2])):
            with self.subTest(direction=direction):
                self.loco._Call.reset_mock()
                controller = self.make()
                self.assertTrue(controller.move(direction, .7)[0])
                self.assertTrue(controller.wait_until_idle(1))
                self.assertEqual(self.payloads(), [{"velocity": vector, "duration": .7},
                                                  {"velocity": [0, 0, 0], "duration": 1}])
                self.assertTrue(all(call.args[0] == 7105 for call in self.loco._Call.call_args_list))
                self.assertEqual(controller.state()["status"], "finished")
                self.assertFalse(controller.state()["busy"])

    def test_invalid_and_nonfinite_inputs_never_start_or_send(self):
        controller = self.make()
        for duration in (True, None, 0, -1, 2.1, float("nan"), float("inf"), "1"):
            self.assertFalse(controller.move("forward", duration)[0])
        self.assertFalse(controller.move("run", 1)[0])
        self.assertFalse(controller.move(None, 1)[0])
        self.loco.get_fsm.assert_not_called()
        self.loco._Call.assert_not_called()

    def test_speed_limits_are_validated_before_any_hardware_use(self):
        for kwargs in ({"linear_speed": .30001}, {"yaw_speed": 1}, {"max_duration": 3},
                       {"linear_speed": -1}, {"yaw_speed": float("nan")}, {"max_duration": True},
                       {"forward_speed": .30001}, {"forward_speed": 0}, {"forward_speed": True},
                       {"forward_speed": float("nan")}, {"forward_speed": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)
        with self.assertRaises(ValueError):
            BodyController(dry=False)

    def test_forward_profile_does_not_raise_other_direction_speeds(self):
        controller = self.make(forward_speed=.3)
        for direction, vector in (("forward", [.3, 0, 0]), ("backward", [-.1, 0, 0]),
                                  ("left", [0, .1, 0]), ("right", [0, -.1, 0]),
                                  ("turn_left", [0, 0, .2]), ("turn_right", [0, 0, -.2])):
            with self.subTest(direction=direction):
                self.loco._Call.reset_mock()
                duration = 2.0 if direction == "forward" else 1.0
                self.assertTrue(controller.move(direction, duration)[0])
                self.assertTrue(controller.wait_until_idle(1))
                self.assertEqual(self.payloads(), [{"velocity": vector, "duration": duration},
                                                  {"velocity": [0, 0, 0], "duration": 1.0}])

    def test_explicit_diagnostic_speed_keeps_time_limit_and_final_zero(self):
        controller = self.make(linear_speed=.3)
        self.assertFalse(controller.move("forward", 2.01)[0])
        self.assertTrue(controller.move("forward", 2.0)[0])
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(self.payloads(), [{"velocity": [.3, 0, 0], "duration": 2.0},
                                         {"velocity": [0, 0, 0], "duration": 1.0}])

    def test_wrong_or_unreadable_fsm_never_sends_nonzero_velocity(self):
        for fsm in (0, 4, 816, None):
            with self.subTest(fsm=fsm):
                self.loco.get_fsm.return_value = fsm
                self.loco._Call.reset_mock()
                controller = self.make()
                controller.move("forward")
                controller.wait_until_idle(1)
                self.assertEqual(controller.state()["status"], "rejected_fsm")
                self.assertTrue(all(p["velocity"] == [0, 0, 0] for p in self.payloads()))
        self.loco.get_fsm.side_effect = RuntimeError("offline")
        controller = self.make()
        controller.move("forward")
        controller.wait_until_idle(1)
        self.assertIn("offline", controller.state()["error"])
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_stop_bypasses_fsm_even_when_idle(self):
        self.loco.get_fsm.side_effect = AssertionError("停止時にFSMを読まない")
        controller = self.make()
        self.assertTrue(controller.stop()[0])
        self.assertEqual(self.payloads(), [{"velocity": [0, 0, 0], "duration": 1}])

    def test_stop_during_fsm_read_cannot_be_followed_by_a_move(self):
        entered, release = threading.Event(), threading.Event()

        def fsm():
            entered.set()
            release.wait(2)
            return 811

        self.loco.get_fsm.side_effect = fsm
        controller = self.make()
        controller.move("forward")
        self.assertTrue(entered.wait(1))
        stopper = threading.Thread(target=controller.stop)
        stopper.start()
        # stop() sets cancellation before waiting for the in-progress RPC lock.
        self.assertTrue(controller._active.cancel.wait(1))
        release.set()
        stopper.join(1)
        self.assertFalse(stopper.is_alive())
        self.assertTrue(controller.wait_until_idle(1))
        self.assertTrue(all(p["velocity"] == [0, 0, 0] for p in self.payloads()))
        self.assertEqual(controller.state()["status"], "cancelled")

    def test_cancel_running_move_sends_zero_and_rejects_overlap(self):
        entered, release = threading.Event(), threading.Event()

        def wait(cancel, _seconds):
            entered.set()
            release.wait(2)
            return cancel.is_set()

        controller = BodyController(self.loco, dry=False, wait=wait)
        controller.move("forward", 1)
        self.assertTrue(entered.wait(1))
        self.assertFalse(controller.move("backward")[0])
        self.assertTrue(controller.stop()[0])
        release.set()
        self.assertTrue(controller.wait_until_idle(1))
        self.assertEqual(controller.state()["status"], "cancelled")
        self.assertEqual(self.payloads()[0]["velocity"], [.1, 0, 0])
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_shared_motion_lock_never_queues_a_surprise_move(self):
        motion_lock = threading.Lock()
        motion_lock.acquire()
        controller = self.make(motion_lock=motion_lock)
        try:
            controller.move("forward")
            self.assertTrue(controller.wait_until_idle(1))
            self.assertEqual(controller.state()["status"], "busy")
            self.loco._Call.assert_not_called()
        finally:
            motion_lock.release()
        self.assertFalse(controller.state()["busy"])

    def test_failed_velocity_still_sends_stop(self):
        self.loco._Call.side_effect = [(3104, ""), (0, "")]
        controller = self.make()
        controller.move("forward")
        controller.wait_until_idle(1)
        self.assertEqual(controller.state()["status"], "rpc_error")
        self.assertEqual(self.payloads()[-1]["velocity"], [0, 0, 0])

    def test_stop_failure_blocks_new_movement_until_stop_retry_succeeds(self):
        self.loco._Call.side_effect = [(0, ""), (3104, ""), (0, ""), (0, ""), (0, "")]
        controller = self.make()
        controller.move("forward")
        controller.wait_until_idle(1)
        self.assertTrue(controller.state()["stop_required"])
        self.assertFalse(controller.move("forward")[0])
        self.assertTrue(controller.stop()[0])
        self.assertFalse(controller.state()["stop_required"])
        self.assertEqual(controller.state()["error"], "")
        self.assertTrue(controller.move("forward")[0])
        self.assertTrue(controller.wait_until_idle(1))

    def test_rpc_exception_attempts_stop_and_releases_shared_lock(self):
        self.loco._Call.side_effect = [OSError("timeout"), (0, "")]
        motion_lock = threading.Lock()
        controller = self.make(motion_lock=motion_lock)
        controller.move("forward")
        self.assertTrue(controller.wait_until_idle(1))
        self.assertIn("timeout", controller.state()["error"])
        self.assertEqual(len(self.payloads()), 2)
        self.assertTrue(motion_lock.acquire(blocking=False))
        motion_lock.release()

    def test_dry_preview_does_not_import_or_use_dds(self):
        events = []
        controller = BodyController(monotonic=self.clock, wait=self.clock.wait, on_event=events.append)
        controller.move("forward")
        self.assertTrue(controller.wait_until_idle(1))
        self.assertTrue(controller.state()["dry"])
        self.assertEqual(events[0]["event"], "velocity")
        self.loco._Call.assert_not_called()

    def test_native_127_is_rejected_by_strict_default_and_preserved(self):
        self.loco._Call.return_value = (127, "")
        controller = self.make()
        controller.move("forward", .5)
        self.assertTrue(controller.wait_until_idle(1))
        state = controller.state()
        self.assertEqual(state["code"], 127)
        self.assertEqual(state["stop_code"], 127)
        self.assertTrue(state["stop_required"])
        self.assertFalse(state["velocity_127_compatible"])
        self.assertFalse(controller.move("forward")[0])

    def test_compatibility_127_is_provisional_and_still_sends_final_zero(self):
        self.loco._Call.return_value = (127, "")
        controller = self.make(velocity_127_compatible=True)
        controller.move("forward", .5)
        self.assertTrue(controller.wait_until_idle(1))
        state = controller.state()
        self.assertEqual(state["code"], 127)
        self.assertEqual(state["stop_code"], 127)
        self.assertEqual(state["status"], "finished_unconfirmed")
        self.assertTrue(state["physical_confirmation_required"])
        self.assertIn("実際の動作・停止", state["response_note"])
        self.assertFalse(state["stop_required"])
        self.assertEqual(self.payloads(), [{"velocity": [.1, 0, 0], "duration": .5},
                                          {"velocity": [0, 0, 0], "duration": 1}])

    def test_compatibility_never_accepts_other_codes_or_fsm_127(self):
        self.loco._Call.return_value = (3104, "")
        controller = self.make(velocity_127_compatible=True)
        controller.move("forward", .5)
        controller.wait_until_idle(1)
        self.assertTrue(controller.state()["stop_required"])
        self.assertFalse(controller.move("forward")[0])
        self.loco._Call.reset_mock()
        self.loco._Call.return_value = (127, "")
        self.loco.get_fsm.return_value = 127
        controller = self.make(velocity_127_compatible=True)
        controller.move("forward", .5)
        controller.wait_until_idle(1)
        self.assertEqual(controller.state()["status"], "rejected_fsm")
        self.assertTrue(all(p["velocity"] == [0, 0, 0] for p in self.payloads()))

    def test_idle_stop_127_keeps_raw_response_and_uncertainty(self):
        self.loco._Call.return_value = (127, "")
        controller = self.make(velocity_127_compatible=True)
        ok, message = controller.stop()
        self.assertTrue(ok)
        self.assertIn("実際の停止", message)
        self.assertEqual(controller.state()["stop_code"], 127)
        self.assertTrue(controller.state()["physical_confirmation_required"])


class IntentTests(unittest.TestCase):
    def test_direct_requests_only_enable_requested_direction(self):
        for text, action in (("R1、少し前に進んで", "move_forward"), ("後ろへ下がってください", "move_backward"),
                             ("左へ少し移動して", "move_left"), ("右に動いて", "move_right"),
                             ("左を向いて", "turn_left"), ("右に旋回して", "turn_right"),
                             ("歩いて", "move_forward"), ("止まって", "stop_move")):
            with self.subTest(text=text):
                self.assertEqual(requested_body_actions(text), {action})

    def test_greeting_explanation_quotes_and_negation_do_not_authorize_walking(self):
        for text in ("こんにちは", "歩行の仕組みを教えて", "前へ歩いてという台本です", "右に動かないで",
                     "「前へ進んで」", "『左へ移動して』", '"右に動いて"',
                     "前に進まないで", "歩かないで", "旋回できますか", "自己紹介して",
                     "前に進んでを英訳して", "新人に前に進んでと言って", "合図したら前へ進んで",
                     "3秒後に前へ進んで", "私が前に進んでもいいですか"):
            with self.subTest(text=text):
                self.assertEqual(requested_body_actions(text), set())


class BodyEngineTests(unittest.TestCase):
    setUp = engine_tests.EngineTests.setUp

    def fake_body(self):
        self.gesturer.thread = None
        body = Mock()
        body.move.return_value = (True, "preview")
        body.stop.return_value = (True, "stopped")
        body.state.return_value = {"busy": False, "stop_required": False, "error": ""}
        body.wait_until_idle.return_value = True
        self.engine.body = body
        return body

    def test_unrequested_model_walk_keeps_speech_but_never_moves(self):
        body = self.fake_body()
        self.engine.brain.reply = Mock(return_value=("こんにちは。", "move_forward"))
        self.engine._chat("こんにちは", "文字")
        body.move.assert_not_called()
        self.speaker.play.assert_called_once()

    def test_direction_mismatch_cannot_move_opposite_to_request(self):
        body = self.fake_body()
        self.engine._speak("動くぞ。", "move_forward", allowed_body={"move_backward"})
        body.move.assert_not_called()

    def test_explicit_request_runs_only_brief_selected_move(self):
        body = self.fake_body()
        self.engine.brain.reply = Mock(return_value=("前へ少し進むぞ。", "move_forward"))
        self.engine._chat("前に進んで", "文字")
        body.move.assert_called_once_with("forward", 1.0)
        body.wait_until_idle.assert_called_once()
        self.gesturer.start.assert_not_called()

    def test_forward_profile_is_shared_by_speech_and_button_actual_payloads(self):
        self.gesturer.thread = None
        self.engine.forward_speed, self.engine.forward_duration = .3, 2.0
        loco, clock = Mock(), Clock()
        loco.get_fsm.return_value, loco._Call.return_value = 811, (0, "")
        self.engine.body = BodyController(loco, dry=False, forward_speed=self.engine.forward_speed,
                                          monotonic=clock, wait=clock.wait)
        # 音声が終わる前に動作完了を待ち、キャンセル経路に入らない。
        self.speaker.play.side_effect = lambda *a, **kw: self.engine.body.wait_until_idle(1)
        for action, direction, vector, duration in (
                ("move_forward", "forward", [.3, 0, 0], 2.0),
                ("move_backward", "backward", [-.1, 0, 0], 1.0),
                ("move_left", "left", [0, .1, 0], 1.0),
                ("turn_left", "turn_left", [0, 0, .2], 1.0)):
            for route in ("speech", "button"):
                with self.subTest(action=action, route=route):
                    loco._Call.reset_mock()
                    if route == "speech":
                        self.engine._speak("移動の依頼だな。", action, allowed_body={action})
                    else:
                        self.assertTrue(self.engine.body_move(action)[0])
                        self.assertTrue(self.engine.job_lock.acquire(timeout=1))
                        self.engine.job_lock.release()
                    self.assertEqual([json.loads(c.args[1]) for c in loco._Call.call_args_list],
                                     [{"velocity": vector, "duration": duration},
                                      {"velocity": [0, 0, 0], "duration": 1.0}])
                    self.assertEqual(self.engine.body.state()["direction"], direction)

    def test_body_hardware_disabled_and_unknown_fsm_fail_closed(self):
        body = self.fake_body()
        self.engine.args.local = False
        self.engine.body_enabled = False
        self.assertFalse(self.engine.body_move("move_forward")[0])
        self.engine.body_enabled = True
        self.engine.fsm_ok, self.engine.fsm = False, 811
        self.assertFalse(self.engine.body_move("move_forward")[0])
        self.engine.fsm_ok, self.engine.fsm = True, 816
        self.assertFalse(self.engine.body_move("move_forward")[0])
        body.move.assert_not_called()

    def test_pending_arm_cleanup_prevents_walking(self):
        body = self.fake_body()
        self.gesturer.thread = object()
        self.engine._speak("前へ。", "move_forward", allowed_body={"move_forward"})
        body.move.assert_not_called()

    def test_call_cancellation_stops_body_and_joins_before_job_release(self):
        body = self.fake_body()
        body.state.return_value = {"busy": True, "stop_required": False, "error": ""}
        current = threading.Event()
        current.set()
        self.speaker.play.side_effect = lambda *_args, **_kwargs: current.clear()
        self.engine._speak("前へ。", "move_forward", should_continue=current.is_set,
                           allowed_body={"move_forward"})
        body.stop.assert_called_once()
        body.wait_until_idle.assert_called_once()
        body.state.return_value["busy"] = False  # fixture cleanup must not retry mocked activity.

    def test_manual_move_keeps_job_lock_but_stop_bypasses_it(self):
        body = self.fake_body()
        entered, release = threading.Event(), threading.Event()
        body.wait_until_idle.side_effect = lambda *_: (entered.set(), release.wait(2))[-1]
        self.assertTrue(self.engine.body_move("move_forward")[0])
        self.assertTrue(entered.wait(1))
        try:
            self.assertTrue(self.engine.job_lock.locked())
            self.assertFalse(self.engine.gesture("wave")[0])
            self.assertTrue(self.engine.body_stop()[0])
            body.stop.assert_called_once()
            self.gesturer.manual.assert_not_called()
        finally:
            release.set()
        self.assertTrue(self.engine.job_lock.acquire(timeout=1))
        self.engine.job_lock.release()

    def test_typed_stop_bypasses_llm_and_an_existing_job(self):
        body = self.fake_body()
        self.engine.brain.reply = Mock()
        self.engine.job_lock.acquire()
        try:
            self.assertTrue(self.engine.chat("止まって")[0])
        finally:
            self.engine.job_lock.release()
        body.stop.assert_called_once()
        self.engine.brain.reply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
