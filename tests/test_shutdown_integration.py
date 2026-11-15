"""Graceful shutdown uses fake devices and a fake HTTP server only."""
import email.message
import io
import json
import os
import subprocess
import threading
import unittest
from unittest import mock

import test_engine_integration as fixtures

ui = fixtures.ui


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        fixtures.EngineTests.setUp(self)
        self.gesturer.thread = None

    def physical(self):
        self.engine.args.local = False
        self.engine.args.interface = "Ethernet"
        self.engine.dds_address = "192.168.123.164"
        self.engine.fsm, self.engine.fsm_ok = 811, True
        self.feedback = mock.Mock()
        self.engine.body_feedback = self.feedback
        state = {"busy": False, "stopping": False, "hold_active": False,
                 "stop_required": False, "feedback_valid": True,
                 "joint_stationary_observed": False}
        body = mock.Mock(spec=ui.BodyController)
        body.require_feedback = True
        body.state.side_effect = lambda: dict(state)
        body.require_stop_verification.side_effect = lambda reason: state.update(stop_required=True)

        def zero_and_verify():
            state.update(stop_required=False, joint_stationary_observed=True)
            return True, "Zero sent; fresh quiet observations verified"
        body.stop.side_effect = zero_and_verify
        self.engine.body = body
        return body, state

    def test_state_advertises_capability_and_actual_process(self):
        state = self.engine.state()
        self.assertEqual(state["process_id"], os.getpid())
        self.assertTrue(state["shutdown"]["supported"])
        self.assertFalse(state["shutdown"]["requested"])

    def test_local_shutdown_quiesces_voice_training_and_reserves_job_slot(self):
        self.engine.call.start()
        self.engine.training.start()
        self.engine.training.phase = "await_reply"
        with mock.patch.object(self.engine.body, "stop") as stop:
            ok, _ = self.engine.prepare_shutdown()
        self.assertTrue(ok)
        self.assertFalse(self.engine.closed)
        self.assertTrue(self.engine.shutdown_ready)
        self.assertTrue(self.engine.shutdown_requested)
        self.assertFalse(self.engine.training.active)
        self.assertFalse(self.engine.call.active)
        self.assertFalse(self.engine.job_lock.acquire(blocking=False))
        stop.assert_not_called()
        self.speaker.play.assert_not_called()
        self.gesturer.manual.assert_not_called()

    def test_physical_shutdown_requires_post_zero_observations(self):
        body, _ = self.physical()
        ok, _ = self.engine.prepare_shutdown()
        self.assertTrue(ok)
        body.stop.assert_called_once_with()
        body.require_stop_verification.assert_called_once()
        body.move.assert_not_called()
        body.start_hold.assert_not_called()
        self.feedback.close.assert_not_called()
        self.assertFalse(self.engine.closed)

    def test_repeated_preparation_does_not_repeat_zero(self):
        body, _ = self.physical()
        self.assertTrue(self.engine.prepare_shutdown()[0])
        self.assertTrue(self.engine.prepare_shutdown()[0])
        body.stop.assert_called_once()

    def test_127_or_success_without_quiet_proof_keeps_monitor_and_interlock(self):
        body, state = self.physical()
        body.stop.side_effect = lambda: (True, "127 response only")
        ok, _ = self.engine.prepare_shutdown()
        self.assertFalse(ok)
        self.assertFalse(self.engine.shutdown_ready)
        self.assertTrue(self.engine.shutdown_requested)
        self.assertTrue(state["stop_required"])
        self.assertFalse(self.engine.closed)
        self.feedback.close.assert_not_called()
        self.assertFalse(self.engine.submit("new work", mock.Mock())[0])

    def test_stale_feedback_refuses_shutdown_then_allows_explicit_retry(self):
        body, state = self.physical()
        verify = body.stop.side_effect
        state["feedback_valid"] = False
        body.stop.side_effect = lambda: (False, "stop_not_observed:stale")
        self.assertFalse(self.engine.prepare_shutdown()[0])
        self.assertFalse(self.engine.closed)
        self.assertFalse(self.engine.job_lock.locked())
        self.feedback.close.assert_not_called()
        state["feedback_valid"] = True
        body.stop.side_effect = verify
        self.assertTrue(self.engine.prepare_shutdown()[0])
        self.assertTrue(self.engine.job_lock.locked())

    def test_active_posture_and_arm_modes_refuse_before_cancellation(self):
        body, _ = self.physical()
        for fsm in (701, 702, 816):
            with self.subTest(fsm=fsm):
                self.engine.fsm = fsm
                self.assertFalse(self.engine.prepare_shutdown()[0])
                self.assertFalse(self.engine.shutdown_requested)
        body.stop.assert_not_called()
        body.require_stop_verification.assert_not_called()
        self.gesturer.manual.assert_not_called()

    def test_missing_fsm_or_feedback_monitor_cannot_produce_shutdown_success(self):
        body, _ = self.physical()
        self.engine.fsm_ok = False
        self.assertFalse(self.engine.prepare_shutdown()[0])
        body.stop.assert_not_called()
        self.assertFalse(self.engine.closed)
        self.feedback.close.assert_not_called()

    def test_recording_is_discarded_without_transcription_or_new_reply(self):
        stream = mock.Mock()
        self.engine.rec_stream = stream
        self.engine.rec_frames = [fixtures.np.zeros((16000, 1), dtype=fixtures.np.int16)]
        with mock.patch.object(self.engine, "record_stop") as submit_recording:
            self.assertTrue(self.engine.prepare_shutdown()[0])
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        submit_recording.assert_not_called()
        self.stt.transcribe.assert_not_called()
        self.assertIsNone(self.engine.rec_stream)
        self.assertEqual(self.engine.rec_frames, [])
        self.speaker.play.assert_not_called()

    def test_blocked_record_close_keeps_pending_stream_and_can_retry(self):
        release = threading.Event()
        stream = mock.Mock()
        stream.close.side_effect = lambda: release.wait(1)
        self.engine.rec_stream = stream
        try:
            self.assertFalse(self.engine.prepare_shutdown(timeout=.03)[0])
            self.assertIs(self.engine._shutdown_record_stream, stream)
            self.assertFalse(self.engine.closed)
            self.assertFalse(self.engine.shutdown_ready)
        finally:
            release.set()
            self.assertTrue(self.engine._shutdown_record_event.wait(1))
        self.assertTrue(self.engine.prepare_shutdown()[0])

    def test_inflight_text_response_cannot_resume_after_refused_shutdown(self):
        entered, release = threading.Event(), threading.Event()
        brain = mock.Mock()
        brain.model = "test-model"
        def reply(*args, **kwargs):
            entered.set()
            release.wait(1)
            return "This reply must not be played", "wave"
        brain.reply.side_effect = reply
        self.engine.brains["mentor"] = brain
        self.assertTrue(self.engine.chat("Explain this task")[0])
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(self.engine.prepare_shutdown(timeout=.03)[0])
            self.assertFalse(self.engine.closed)
            self.assertFalse(self.engine.call_start()[0])
            self.assertFalse(self.engine.record_start()[0])
            self.assertFalse(self.engine.body_hold_start("move_forward", "a" * 16,
                                                        self.engine.hold_epoch)[0])
        finally:
            release.set()
            self.assertTrue(self.engine.job_lock.acquire(timeout=1))
            self.engine.job_lock.release()
        self.speaker.play.assert_not_called()
        self.gesturer.start.assert_not_called()

    def test_record_close_worker_start_failure_can_retry_without_losing_stream(self):
        stream = mock.Mock()
        self.engine.rec_stream = stream
        with mock.patch.object(ui.threading, "Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("thread unavailable")
            self.assertFalse(self.engine.prepare_shutdown()[0])
        self.assertIs(self.engine._shutdown_record_stream, stream)
        self.assertTrue(self.engine._shutdown_record_event.is_set())
        stream.close.assert_not_called()
        self.assertTrue(self.engine.prepare_shutdown()[0])
        stream.close.assert_called_once()

    def test_input_closing_prevents_shutdown_success(self):
        self.engine.call.state = mock.Mock(return_value={"active": False, "input_closing": True})
        self.assertFalse(self.engine.prepare_shutdown(timeout=.02)[0])
        self.assertFalse(self.engine.closed)
        self.assertFalse(self.engine.shutdown_ready)

    def test_invalid_timeouts_do_not_begin_quiescence(self):
        for timeout in (True, 0, -1, float("nan"), float("inf"), 31, "1"):
            self.assertFalse(self.engine.prepare_shutdown(timeout)[0])
        self.assertFalse(self.engine.shutdown_requested)

    def test_owned_voice_worker_is_waited_and_reaped(self):
        proc = mock.Mock()
        proc.wait.side_effect = [subprocess.TimeoutExpired("voice worker", 3), 0]
        ui.Engine._wait_owned_worker(proc)
        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_args_list, [mock.call(timeout=3), mock.call(timeout=2)])


class ShutdownHandlerTests(unittest.TestCase):
    def setUp(self):
        self.engine = mock.Mock(spec=ui.Engine)
        self.engine.shutdown_requested = False
        self.engine.shutdown_ready = False
        self.engine.prepare_shutdown.return_value = (True, "verified")
        self.engine.body_stop.return_value = (True, "stop")
        self.shutdown_called = threading.Event()

    def post(self, path, *, origin="http://127.0.0.1:8765"):
        handler = ui.make_handler(self.engine).__new__(ui.make_handler(self.engine))
        handler.path = path
        handler.headers = email.message.Message()
        handler.headers["Host"] = "127.0.0.1:8765"
        handler.headers["Origin"] = origin
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = "2"
        handler.rfile, handler.wfile = io.BytesIO(b"{}"), io.BytesIO()
        self.sent_before_shutdown = []
        def stop_server():
            self.sent_before_shutdown.append(json.loads(handler.wfile.getvalue())["ok"])
            self.shutdown_called.set()
        handler.server = fixtures.types.SimpleNamespace(server_port=8765, shutdown=mock.Mock(side_effect=stop_server))
        handler.send_response, handler.send_header, handler.end_headers = mock.Mock(), mock.Mock(), mock.Mock()
        handler.do_POST()
        return handler.send_response.call_args.args[0], json.loads(handler.wfile.getvalue()), handler

    def test_success_responds_before_stopping_server_in_background(self):
        status, data, handler = self.post("/api/shutdown")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertTrue(self.shutdown_called.wait(1))
        self.assertEqual(self.sent_before_shutdown, [True])
        self.engine.prepare_shutdown.assert_called_once_with()

    def test_refused_confirmation_keeps_http_server_alive(self):
        self.engine.prepare_shutdown.return_value = (False, "stale")
        status, data, handler = self.post("/api/shutdown")
        self.assertEqual(status, 409)
        self.assertFalse(data["ok"])
        handler.server.shutdown.assert_not_called()

    def test_pending_shutdown_rejects_new_work_but_permits_stop(self):
        self.engine.shutdown_requested = True
        status, _, _ = self.post("/api/greet")
        self.assertEqual(status, 409)
        self.engine.greet.assert_not_called()
        self.assertEqual(self.post("/api/body/stop")[0], 200)
        self.engine.body_stop.assert_called_once()

    def test_shutdown_ready_rejects_additional_stop_work(self):
        self.engine.shutdown_requested = self.engine.shutdown_ready = True
        self.assertEqual(self.post("/api/body/stop")[0], 409)
        self.engine.body_stop.assert_not_called()

    def test_foreign_origin_cannot_shutdown(self):
        self.assertEqual(self.post("/api/shutdown", origin="https://other.invalid")[0], 403)
        self.engine.prepare_shutdown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
