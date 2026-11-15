"""LowStateは偽メッセージと偽購読者だけで検証する。実機には接続しない。"""
import builtins
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_motion"))
from body_feedback import MotorFeedbackMonitor, _PollingSubscriber


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds=.03):
        self.t += seconds


def message(tick=1):
    return SimpleNamespace(tick=tick, crc_ok=True, mode_machine=1, mode_pr=0,
                           motor_state=[SimpleNamespace(q=i * .01, dq=0., motorstate=0) for i in range(35)])


class FakeSubscriber:
    def __init__(self):
        self.handler = None
        self.closed = False

    def Init(self, handler, queue_len):
        self.handler = handler
        self.queue_len = queue_len

    def Close(self):
        self.closed = True


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.validator = Mock(side_effect=lambda msg: msg.crc_ok)
        self.monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=self.validator)

    def test_no_snapshot_until_data_and_no_sdk_import_for_injected_path(self):
        self.assertIsNone(self.monitor.snapshot())
        original = builtins.__import__

        def no_sdk(name, *args, **kwargs):
            if name.startswith("unitree"):
                self.fail("テストでSDKをimportしない")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_sdk):
            self.assertTrue(self.monitor.feed(message()))
        snapshot = self.monitor.snapshot()
        self.assertEqual(snapshot["tick"], 1)
        self.assertTrue(snapshot["crc_valid"])
        self.assertEqual(snapshot["mode_machine"], 1)
        self.assertEqual(len(snapshot["qlegs"]), 12)
        self.assertEqual(snapshot["dqlegs"], [0.] * 12)

    def test_callback_is_throttled_to_requested_rate(self):
        self.monitor.feed(message(1))
        self.clock.advance(.001)
        self.assertFalse(self.monitor.feed(message(2)))
        self.validator.assert_called_once()
        self.clock.advance(.02)
        self.assertTrue(self.monitor.feed(message(3)))
        self.assertEqual(self.monitor.snapshot()["tick"], 3)
        self.assertEqual(self.validator.call_count, 2)

    def test_crc_failure_replaces_old_healthy_snapshot(self):
        self.monitor.feed(message(1))
        self.clock.advance()
        bad = message(2)
        bad.crc_ok = False
        self.monitor.feed(bad)
        snapshot = self.monitor.snapshot()
        self.assertFalse(snapshot["crc_valid"])
        self.assertEqual(snapshot["qlegs"], [])
        self.assertIsNone(snapshot["mode_machine"])
        self.assertIn("CRC", snapshot["error"])

    def test_all_joint_faults_are_retained_without_defaulting_missing_fields(self):
        msg = message()
        msg.motor_state[34].motorstate = 4
        self.monitor.feed(msg)
        self.assertEqual(self.monitor.snapshot()["motor_fault"], [{"joint": 34, "motorstate": 4}])
        self.clock.advance()
        broken = message(2)
        del broken.motor_state[34].motorstate
        self.monitor.feed(broken)
        snapshot = self.monitor.snapshot()
        self.assertTrue(snapshot["error"])
        self.assertEqual(snapshot["qlegs"], [])

    def test_snapshot_has_no_shared_mutable_fields(self):
        msg = message()
        msg.motor_state[34].motorstate = 4
        self.monitor.feed(msg)
        read = self.monitor.snapshot()
        read["qlegs"][0] = 999
        read["motor_fault"][0]["motorstate"] = 0
        msg.motor_state[1].q = 999
        fresh = self.monitor.snapshot()
        self.assertEqual(fresh["qlegs"][0], 0)
        self.assertEqual(fresh["qlegs"][1], .01)
        self.assertEqual(fresh["motor_fault"][0]["motorstate"], 4)

    def test_malformed_and_nonfinite_data_cannot_leave_a_healthy_sample(self):
        broken = []
        msg = message()
        msg.motor_state.pop()
        broken.append(msg)
        for field, value in (("q", float("nan")), ("dq", float("inf"))):
            msg = message()
            setattr(msg.motor_state[0], field, value)
            broken.append(msg)
        msg = message()
        msg.mode_machine = None
        broken.append(msg)
        for msg in broken:
            with self.subTest(msg=msg):
                self.clock.advance()
                self.monitor.feed(msg)
                snapshot = self.monitor.snapshot()
                self.assertTrue(snapshot["error"])
                self.assertEqual(snapshot["qlegs"], [])

    def test_duplicate_and_older_ticks_do_not_refresh_receive_time(self):
        self.monitor.feed(message(10))
        saved = self.monitor.snapshot()
        for tick in (10, 9, 8):
            self.clock.advance()
            self.assertFalse(self.monitor.feed(message(tick)))
            self.assertEqual(self.monitor.snapshot(), saved)
        self.clock.advance()
        self.assertTrue(self.monitor.feed(message(11)))
        self.assertGreater(self.monitor.snapshot()["t"], saved["t"])

    def test_uint32_tick_wrap_is_accepted_but_reverse_wrap_is_not(self):
        self.monitor.feed(message(2**32 - 2))
        self.clock.advance()
        self.assertTrue(self.monitor.feed(message(1)))
        saved = self.monitor.snapshot()
        self.clock.advance()
        self.assertFalse(self.monitor.feed(message(2**32 - 1)))
        self.assertEqual(self.monitor.snapshot(), saved)

    def test_missing_and_noninteger_tick_invalidate_snapshot(self):
        for tick in (None, True, 1.5, -1, 2**32):
            with self.subTest(tick=tick):
                self.clock.advance()
                self.monitor.feed(message(tick))
                self.assertTrue(self.monitor.snapshot()["error"])
                self.assertEqual(self.monitor.snapshot()["qlegs"], [])
        self.clock.advance()
        msg = message()
        del msg.tick
        self.monitor.feed(msg)
        self.assertTrue(self.monitor.snapshot()["error"])

    def test_start_and_close_use_only_injected_read_only_subscriber(self):
        subscriptions = []

        def factory(topic, message_type):
            self.assertEqual(topic, "rt/lowstate")
            self.assertIsNone(message_type)
            sub = FakeSubscriber()
            subscriptions.append(sub)
            return sub

        monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=self.validator,
                                       subscriber_factory=factory)
        self.assertIs(monitor.start(), monitor)
        self.assertIs(monitor.start(), monitor)
        self.assertEqual(len(subscriptions), 1)
        self.assertEqual(subscriptions[0].queue_len, 10)
        subscriptions[0].handler(message())
        self.assertIsNotNone(monitor.snapshot())
        self.assertTrue(monitor.close())
        self.assertTrue(subscriptions[0].closed)
        self.assertIsNone(monitor.snapshot())
        self.clock.advance()
        self.assertFalse(monitor.feed(message(2)))
        monitor.start()
        self.clock.advance()
        self.assertFalse(subscriptions[0].handler(message(3)))
        self.assertIsNone(monitor.snapshot())
        subscriptions[1].handler(message(4))
        self.assertEqual(monitor.snapshot()["tick"], 4)
        monitor.close()

    def test_failed_start_closes_subscriber_and_invalidates_state(self):
        sub = FakeSubscriber()
        sub.Init = Mock(side_effect=OSError("failed"))
        monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=self.validator,
                                       subscriber_factory=lambda *_: sub)
        with self.assertRaises(OSError):
            monitor.start()
        self.assertTrue(sub.closed)
        self.assertIsNone(monitor.snapshot())
        self.assertFalse(monitor.feed(message()))

    def test_configuration_rejects_unsafe_sampling_values(self):
        for value in (0, -1, 51, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MotorFeedbackMonitor(sample_hz=value)

    def test_native_path_uses_latest_one_polling_without_dds_listener(self):
        class FakeKeepLast:
            def __init__(self, depth):
                self.depth = depth

        class FakeQos:
            def __init__(self, *policies):
                self.policies = policies

        channel = Mock()
        channel.Read.return_value = None
        factory = Mock()
        factory.CreateChannel.return_value = channel
        datatype = object()
        modules = {
            "unitree_sdk2py.core.channel": SimpleNamespace(ChannelFactory=Mock(return_value=factory)),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": SimpleNamespace(LowState_=datatype),
            "unitree_sdk2py.utils.crc": SimpleNamespace(CRC=Mock()),
            "cyclonedds.qos": SimpleNamespace(Qos=FakeQos, Policy=SimpleNamespace(
                History=SimpleNamespace(KeepLast=FakeKeepLast),
                Durability=SimpleNamespace(Volatile="volatile"))),
        }
        with patch.dict(sys.modules, modules):
            monitor = MotorFeedbackMonitor(sample_hz=10, crc_validator=self.validator)
            monitor.start()
            factory.CreateChannel.assert_called_once_with("rt/lowstate", datatype)
            factory.Init.assert_not_called()
            qos = channel.SetReader.call_args.kwargs["qos"]
            self.assertEqual(qos.policies[0].depth, 1)
            self.assertEqual(qos.policies[1], "volatile")
            self.assertEqual(channel.SetReader.call_args.kwargs["queueLen"], 0)
            self.assertIsNone(channel.SetReader.call_args.kwargs["handler"])
            channel.SetWriter.assert_not_called()
            monitor.close()
            channel.CloseReader.assert_called_once()

    def test_polling_closes_reader_only_after_blocked_read_has_finished(self):
        entered, release, exited, received = (threading.Event() for _ in range(4))
        sample = message()
        channel = Mock()

        def read(*, timeout):
            self.assertLessEqual(timeout, .05)
            entered.set()
            release.wait(1)
            exited.set()
            return sample

        channel.Read.side_effect = read
        channel.CloseReader.side_effect = lambda: self.assertTrue(exited.is_set())
        reader = _PollingSubscriber(channel, object(), .1)
        reader.Init(lambda msg: received.set(), 10)
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=reader.Close)
        closer.start()
        release.set()
        closer.join(1)
        self.assertFalse(closer.is_alive())
        channel.CloseReader.assert_called_once()

    def test_polling_delivers_samples_from_own_worker_and_invalidates_exceptions(self):
        received = threading.Event()
        values = []
        channel = Mock()
        channel.Read.side_effect = [message(), OSError("read failed")]

        def handler(sample):
            values.append(sample)
            if len(values) == 2:
                received.set()

        reader = _PollingSubscriber(channel, object(), .01)
        reader.Init(handler, 10)
        try:
            self.assertTrue(received.wait(1))
            self.assertEqual(values[0].tick, 1)
            self.assertIsNone(values[1])
        finally:
            reader.Close()

    def test_recovery_quarantines_tick_reset_until_progress_over_time(self):
        self.monitor.feed(message(5000))
        original_epoch = self.monitor.snapshot()["stream_epoch"]
        self.clock.advance(2)
        self.assertTrue(self.monitor._begin_recovery("no data", 0))
        pending = self.monitor.snapshot()
        self.assertFalse(pending["crc_valid"])
        self.assertTrue(pending["recovering"])
        self.assertGreater(pending["stream_epoch"], original_epoch)
        self.assertFalse(self.monitor.feed(message(1)))
        self.clock.advance(.1)
        self.assertFalse(self.monitor.feed(message(2)))
        self.clock.advance(.11)
        self.assertTrue(self.monitor.feed(message(3)))
        fresh = self.monitor.snapshot()
        self.assertTrue(fresh["crc_valid"])
        self.assertFalse(fresh["recovering"])
        self.assertEqual(fresh["tick"], 3)
        self.assertEqual(fresh["recovery_count"], 1)

    def test_repeated_cached_tick_cannot_become_fresh_across_recoveries(self):
        self.monitor.feed(message(5000))
        for recovery in range(3):
            self.clock.advance(1)
            self.monitor._begin_recovery("no tick progress", 0)
            pending = self.monitor.snapshot()
            for _ in range(5):
                self.clock.advance(.11)
                self.assertFalse(self.monitor.feed(message(1)))
                self.assertFalse(self.monitor.snapshot()["crc_valid"])
                self.assertEqual(self.monitor.snapshot()["t"], pending["t"])
            self.assertEqual(self.monitor.snapshot()["recovery_count"], recovery + 1)
        diagnostic = self.monitor.diagnostics()
        self.assertEqual(diagnostic["accepted_samples"], 1)
        self.assertEqual(diagnostic["last_receive_age"], 0)
        self.assertGreater(diagnostic["last_accepted_age"], 4)

    def test_crc_failure_restarts_recovery_confirmation_window(self):
        self.monitor._begin_recovery("read failure", 0)
        self.assertFalse(self.monitor.feed(message(10)))
        self.clock.advance(.11)
        self.assertFalse(self.monitor.feed(message(11)))
        self.clock.advance(.11)
        broken = message(12)
        broken.crc_ok = False
        self.monitor.feed(broken)
        self.assertFalse(self.monitor.snapshot()["crc_valid"])
        for tick in (13, 14):
            self.clock.advance(.11)
            self.assertFalse(self.monitor.feed(message(tick)))
        self.clock.advance(.11)
        self.assertTrue(self.monitor.feed(message(15)))
        self.assertEqual(self.monitor.snapshot()["tick"], 15)

    def test_recovery_rejects_reverse_sequence_and_preserves_motor_faults(self):
        self.monitor._begin_recovery("reset", 0)
        for tick in (8, 7, 6, 5):
            self.clock.advance(.11)
            self.assertFalse(self.monitor.feed(message(tick)))
        for tick in (6, 7):
            self.clock.advance(.11)
            candidate = message(tick)
            candidate.motor_state[3].motorstate = 8
            accepted = self.monitor.feed(candidate)
        self.assertTrue(accepted)
        self.assertEqual(self.monitor.snapshot()["motor_fault"], [{"joint": 3, "motorstate": 8}])

    def test_old_generation_cannot_reset_a_restarted_monitor(self):
        subscriptions = []
        def factory(*_):
            sub = FakeSubscriber()
            subscriptions.append(sub)
            return sub
        monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=self.validator,
                                       subscriber_factory=factory)
        monitor.start()
        old_generation = monitor._generation
        monitor.close()
        monitor.start()
        subscriptions[-1].handler(message(100))
        before = monitor.snapshot()
        self.assertFalse(monitor._begin_recovery("late reconnect", old_generation))
        self.assertEqual(monitor.snapshot(), before)
        monitor.close()

    def test_validation_started_before_recovery_cannot_enter_new_epoch(self):
        entered, release = threading.Event(), threading.Event()
        def validate(_msg):
            entered.set()
            release.wait(1)
            return True
        monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=validate)
        results = []
        worker = threading.Thread(target=lambda: results.append(monitor.feed(message(9000))))
        worker.start()
        self.assertTrue(entered.wait(1))
        monitor._begin_recovery("reader changed", 0)
        release.set()
        worker.join(1)
        self.assertEqual(results, [False])
        self.assertIsNone(monitor._candidate_tick)
        self.assertFalse(monitor.snapshot()["crc_valid"])

    def test_invalid_received_frames_are_not_accepted_by_reader_watchdog(self):
        sub = FakeSubscriber()
        monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=self.validator,
                                       subscriber_factory=lambda *_: sub)
        monitor.start()
        self.assertTrue(sub.handler(message(100)))
        self.clock.advance(.11)
        broken = message(101)
        broken.crc_ok = False
        self.assertFalse(sub.handler(broken))
        self.assertFalse(monitor.snapshot()["crc_valid"])
        monitor.close()

    def test_polling_rebuilds_after_rejected_ticks_then_recovers_reset_stream(self):
        self.monitor.feed(message(5000))
        self.clock.advance(.11)
        old_channel, new_channel = Mock(), Mock()
        completed = threading.Event()
        old_channel.Read.side_effect = lambda **_: (self.clock.advance(.11), message(5000))[1]
        ticks = iter((1, 1, 1, 2, 3, 4))
        def read_new(**_):
            self.clock.advance(.11)
            return message(next(ticks, 5))
        new_channel.Read.side_effect = read_new
        created = []
        def create():
            self.assertEqual(old_channel.CloseReader.call_count, 1)
            created.append(new_channel)
            return new_channel
        def receive(msg):
            result = self.monitor._receive(msg, 0)
            if result and msg.tick < 5000:
                completed.set()
            return result
        reader = _PollingSubscriber(old_channel, object(), .001, channel_factory=create,
                                    on_reconnect=lambda reason: self.monitor._begin_recovery(reason, 0),
                                    monotonic=self.clock, recovery_after=.5)
        reader.Init(receive, 10)
        try:
            self.assertTrue(completed.wait(1))
            self.assertEqual(len(created), 1)
            self.assertTrue(self.monitor.snapshot()["crc_valid"])
            self.assertEqual(self.monitor.snapshot()["stream_epoch"], 1)
            old_channel.SetWriter.assert_not_called()
            new_channel.SetWriter.assert_not_called()
        finally:
            reader.Close()

    def test_polling_retries_factory_failure_with_bounded_interval(self):
        channel = Mock()
        channel.Read.return_value = None
        attempted = threading.Event()
        times = []
        def create():
            times.append(time.monotonic())
            if len(times) >= 2:
                attempted.set()
            raise OSError("reader unavailable")
        reader = _PollingSubscriber(channel, object(), .002, channel_factory=create,
                                    on_reconnect=lambda _: True, recovery_after=.03)
        reader.Init(lambda _: False, 10)
        try:
            self.assertTrue(attempted.wait(1))
            self.assertGreaterEqual(times[1] - times[0], .03)
            channel.CloseReader.assert_called_once()
        finally:
            reader.Close()

    def test_close_during_factory_wait_does_not_start_new_reader(self):
        entered, release = threading.Event(), threading.Event()
        channel, replacement = Mock(), Mock()
        channel.Read.return_value = None
        def create():
            entered.set()
            release.wait(1)
            return replacement
        reader = _PollingSubscriber(channel, object(), .002, channel_factory=create,
                                    on_reconnect=lambda _: True, recovery_after=.01)
        reader.Init(lambda _: False, 10)
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=reader.Close)
        closer.start()
        self.assertTrue(reader._stop.wait(1))
        release.set()
        closer.join(1)
        self.assertFalse(closer.is_alive())
        replacement.SetReader.assert_not_called()
        replacement.CloseReader.assert_called_once()

    def test_refused_generation_recovery_does_not_create_reader(self):
        channel = Mock()
        channel.Read.return_value = None
        create = Mock()
        reader = _PollingSubscriber(channel, object(), .002, channel_factory=create,
                                    on_reconnect=lambda _: False, recovery_after=.01)
        reader.Init(lambda _: False, 10)
        reader._thread.join(1)
        self.assertFalse(reader._thread.is_alive())
        create.assert_not_called()
        reader.Close()

    def test_failed_reader_close_blocks_restart_until_close_is_confirmed(self):
        subscriptions = []
        def factory(*_):
            sub = FakeSubscriber()
            subscriptions.append(sub)
            return sub
        monitor = MotorFeedbackMonitor(monotonic=self.clock, crc_validator=self.validator,
                                       subscriber_factory=factory)
        monitor.start()
        subscriptions[0].Close = Mock(side_effect=RuntimeError("read still active"))
        self.assertFalse(monitor.close())
        with self.assertRaisesRegex(RuntimeError, "終了していない"):
            monitor.start()
        self.assertEqual(len(subscriptions), 1)
        self.assertFalse(subscriptions[0].handler(message()))
        subscriptions[0].Close = Mock()
        self.assertTrue(monitor.close())
        monitor.start()
        self.assertEqual(len(subscriptions), 2)
        monitor.close()


if __name__ == "__main__":
    unittest.main()
