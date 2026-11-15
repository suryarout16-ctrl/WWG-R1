"""起立は偽FSMと差替え時計だけで検証し、DDS・機体・実時間の待機を使わない。"""
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_motion"))
from posture_control import StandController


class Clock:
    def __init__(self):
        self.value = 0.0
        self.hook = None

    def __call__(self):
        return self.value

    def wait(self, event, seconds):
        self.value += seconds
        if self.hook:
            self.hook()
        return event.is_set()


class Loco:
    def __init__(self, clock, fsm=0):
        self.clock, self.fsm = clock, fsm
        self.commands = []
        self.codes = {}
        self.ignore = set()
        self.read_hook = None
        self.reads = 0
        self.rise_after = 3.5
        self.rises_at = 3.5 if fsm == 701 else None

    def get_fsm(self):
        self.reads += 1
        if self.read_hook:
            self.read_hook()
        if self.fsm == 701 and self.rises_at is not None and self.clock() >= self.rises_at:
            self.fsm = 811
        return self.fsm

    def set_fsm(self, target):
        self.commands.append(target)
        code = self.codes.get(target, 0)
        if type(code) is int and code == 0 and target not in self.ignore:
            self.fsm = target
            if target == 701:
                self.rises_at = None if self.rise_after is None else self.clock() + self.rise_after
        return code


class StandControllerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.loco = Loco(self.clock)

    def make(self, **kwargs):
        return StandController(self.loco, monotonic=self.clock, wait=self.clock.wait, **kwargs)

    def test_supine_sequence_requires_each_observed_mode_and_settles(self):
        controller = self.make()
        result = controller.run("supine", True)
        self.assertEqual(self.loco.commands, [1, 4, 701])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["fsm"], 811)
        self.assertFalse(result["busy"])
        self.assertGreaterEqual(self.clock(), 3 + 3.5 + 4)
        self.assertEqual(controller.state(), result)

    def test_resume_only_from_known_modes_and_running_is_noop(self):
        for fsm, commands, status in ((1, [4, 701], "finished"), (4, [701], "finished"),
                                       (811, [], "already_running"), (701, [], "finished")):
            with self.subTest(fsm=fsm):
                self.clock = Clock()
                self.loco = Loco(self.clock, fsm)
                result = self.make().run("supine", True)
                self.assertEqual(self.loco.commands, commands)
                self.assertEqual(result["status"], status)
                self.assertFalse(result["busy"])

    def test_unknown_posture_and_truthy_readiness_never_touch_client(self):
        controller = self.make()
        for posture, ready in ((None, True), ("unknown", True), ("hanging", True),
                                ("supine", False), ("supine", 1), ("supine", "yes")):
            with self.subTest(posture=posture, ready=ready):
                self.assertEqual(controller.run(posture, ready)["status"], "rejected_input")
        self.assertEqual(self.loco.reads, 0)
        self.assertEqual(self.loco.commands, [])

    def test_wrong_or_unreadable_modes_never_send_commands(self):
        for fsm in (2, 702, 816, None, "0", True, 0.0):
            with self.subTest(fsm=fsm):
                self.loco = Loco(self.clock, fsm)
                result = self.make().run("supine", True)
                self.assertEqual(result["status"], "rejected_fsm")
                self.assertEqual(self.loco.commands, [])
                self.assertEqual(self.loco.reads, 1)

    def test_read_exception_stops_without_recovery_command(self):
        events = []
        self.loco.read_hook = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        result = self.make(on_event=events.append).run("supine", True)
        self.assertEqual(result["status"], "read_error")
        self.assertEqual(self.loco.commands, [])
        self.assertEqual(self.loco.reads, 3)
        self.assertAlmostEqual(self.clock(), .4)
        self.assertEqual([row for row in events if row["event"] == "read_retry"], [
            {"event": "read_retry", "dry": False, "attempt": 1, "error_type": "RuntimeError"},
            {"event": "read_retry", "dry": False, "attempt": 2, "error_type": "RuntimeError"},
        ])

    def test_transient_read_failure_after_locked_command_requires_fresh_4_before_701(self):
        self.loco = Loco(self.clock, 1)
        failed = False
        audit, events = [], []

        def fail_once():
            nonlocal failed
            if self.loco.commands == [4] and not failed:
                failed = True
                raise RuntimeError("transient SDK details must not be logged")

        self.loco.read_hook = fail_once
        original_get, original_set = self.loco.get_fsm, self.loco.set_fsm

        def read():
            try:
                fsm = original_get()
            except RuntimeError:
                audit.append(("read_error", None))
                raise
            audit.append(("read", fsm))
            return fsm

        def write(target):
            audit.append(("write", target))
            return original_set(target)

        self.loco.get_fsm, self.loco.set_fsm = read, write
        result = self.make(on_event=events.append).run("supine", True)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(self.loco.commands, [4, 701])
        self.assertLess(audit.index(("write", 4)), audit.index(("read_error", None)))
        self.assertLess(audit.index(("read_error", None)), audit.index(("read", 4)))
        self.assertLess(audit.index(("read", 4)), audit.index(("write", 701)))
        retries = [row for row in events if row["event"] == "read_retry"]
        self.assertEqual(retries, [{"event": "read_retry", "dry": False,
                                    "attempt": 1, "error_type": "RuntimeError"}])
        self.assertNotIn("transient SDK details", repr(events))

    def test_persistent_read_failure_after_4_never_sends_701(self):
        self.loco = Loco(self.clock, 1)
        failures = 0

        def fail_after_locked():
            nonlocal failures
            if self.loco.commands == [4]:
                failures += 1
                raise RuntimeError("offline")

        self.loco.read_hook = fail_after_locked
        result = self.make().run("supine", True)
        self.assertEqual(result["status"], "read_error")
        self.assertEqual(failures, 3)
        self.assertEqual(self.loco.commands, [4])

    def test_cancel_between_read_retries_prevents_more_sdk_calls(self):
        controller = self.make()
        self.loco.read_hook = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        self.clock.hook = controller.cancel
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.reads, 1)
        self.assertEqual(self.loco.commands, [])

    def test_cancelled_exception_does_not_begin_a_retry(self):
        events = []
        controller = self.make(on_event=events.append)

        def cancel_then_fail():
            controller.cancel()
            raise RuntimeError("offline")

        self.loco.read_hook = cancel_then_fail
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.reads, 1)
        self.assertEqual(self.loco.commands, [])
        self.assertFalse(any(row["event"] == "read_retry" for row in events))

    def test_transition_deadline_is_not_restarted_by_transient_read_errors(self):
        self.loco = Loco(self.clock, 701)
        self.loco.rises_at = None

        def alternate_failure():
            if self.loco.reads % 2:
                raise RuntimeError("transient")

        self.loco.read_hook = alternate_failure
        result = self.make().run("supine", True)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(self.loco.commands, [])
        self.assertGreaterEqual(self.clock(), 25)
        self.assertLess(self.clock(), 26)

    def test_all_nonzero_or_noninteger_rpc_codes_stop_without_later_steps(self):
        for code in (127, 3104, None, False, "0"):
            with self.subTest(code=code):
                self.loco = Loco(self.clock)
                self.loco.codes[1] = code
                result = self.make().run("supine", True)
                self.assertEqual(result["status"], "rpc_error")
                self.assertEqual(self.loco.commands, [1])

    def test_acknowledgment_without_observed_target_times_out(self):
        self.loco.ignore.add(1)
        result = self.make().run("supine", True)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(self.loco.commands, [1])
        self.assertGreaterEqual(self.clock(), 6)

    def test_stalled_existing_rise_only_times_out_without_resending_701(self):
        self.loco = Loco(self.clock, 701)
        self.loco.rises_at = None
        result = self.make().run("supine", True)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(self.loco.commands, [])
        self.assertGreaterEqual(self.clock(), 25)

    def test_mode_change_before_rise_prevents_701(self):
        controller = self.make()
        def change():
            if controller.state()["phase"] == "settling_locked":
                self.loco.fsm = 702
        self.clock.hook = change
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "rejected_fsm")
        self.assertEqual(self.loco.commands, [1, 4])

    def test_cancel_during_initial_read_prevents_first_write(self):
        controller = self.make()
        self.loco.read_hook = controller.cancel
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.commands, [])

    def test_cancel_after_damping_never_sends_locked_or_recovery(self):
        def event(row):
            if row["event"] == "set_fsm" and row["target"] == 1:
                controller.cancel()
        controller = self.make(on_event=event)
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.commands, [1])

    def test_cancel_during_settling_never_sends_701(self):
        controller = self.make()
        def cancel():
            if controller.state()["phase"] == "settling_locked":
                controller.cancel()
        self.clock.hook = cancel
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.commands, [1, 4])

    def test_cancel_after_rise_command_does_not_deenergize_or_lie_down(self):
        controller = self.make()
        def cancel():
            if self.loco.fsm == 701:
                controller.cancel()
        self.clock.hook = cancel
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.commands, [1, 4, 701])
        self.assertEqual(self.loco.fsm, 701)

    def test_inactive_generation_rejects_before_reading(self):
        result = self.make().run("supine", True, should_continue=lambda: False)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.loco.reads, 0)

    def test_shared_motion_lock_rejects_without_queuing_then_releases(self):
        lock = threading.Lock()
        controller = self.make(motion_lock=lock)
        lock.acquire()
        try:
            self.assertEqual(controller.run("supine", True)["status"], "motion_busy")
            self.assertEqual(self.loco.reads, 0)
        finally:
            lock.release()
        self.assertEqual(controller.run("supine", True)["status"], "finished")
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()

    def test_overlapping_run_does_not_replace_current_state(self):
        rejected = []
        def event(row):
            if row["event"] == "set_fsm" and row["target"] == 1:
                rejected.append(controller.run("supine", True))
        controller = self.make(on_event=event)
        result = controller.run("supine", True)
        self.assertEqual(rejected[0]["status"], "busy")
        self.assertTrue(rejected[0]["busy"])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(self.loco.commands, [1, 4, 701])

    def test_preview_is_explicit_and_never_reads_or_sends(self):
        result = self.make(dry=True).run("supine", True)
        self.assertEqual(result["status"], "preview")
        self.assertIn("プレビュー", result["note"])
        self.assertFalse(result["busy"])
        self.assertEqual(self.loco.reads, 0)
        self.assertEqual(self.loco.commands, [])
        self.assertEqual(StandController(dry=True).run("supine", True)["status"], "preview")

    def test_logging_failure_does_not_change_command_sequence(self):
        def broken(row):
            raise RuntimeError("logger failed")
        result = self.make(on_event=broken).run("supine", True)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(self.loco.commands, [1, 4, 701])

    def test_running_mode_must_still_be_observed_after_settling(self):
        controller = self.make()
        def change():
            if controller.state()["phase"] == "settling_running":
                self.loco.fsm = 0
        self.clock.hook = change
        result = controller.run("supine", True)
        self.assertEqual(result["status"], "rejected_fsm")
        self.assertEqual(self.loco.commands, [1, 4, 701])


if __name__ == "__main__":
    unittest.main()
