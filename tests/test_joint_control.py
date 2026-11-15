"""Pure offline trajectory validation; these tests never import DDS or the SDK."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "r1_motion"))
import joint_control as jc


class JointPreviewTests(unittest.TestCase):
    def setUp(self):
        # Synthetic test bounds; these are not an R1 hardware profile.
        self.profile = {
            "schema_version": 1, "profile_id": "synthetic-test-only", "model_id": "mock-robot",
            "mode_machine": 7, "max_state_age_s": 0.5, "seed_tolerance_rad": 0.005,
            "source": {"verified": True, "joint_layout_verified": True,
                       "reference": "synthetic-unit-test", "firmware": "mock"},
            "joints": [
                {"name": "test_knee", "index": 3, "lower_rad": -1.0, "upper_rad": 1.0, "max_velocity_rad_s": 0.5},
                {"name": "test_elbow", "index": 18, "lower_rad": -2.0, "upper_rad": 2.0, "max_velocity_rad_s": 1.0},
            ],
        }
        self.state = {"profile_id": "synthetic-test-only", "model_id": "mock-robot", "mode_machine": 7,
                      "source": "rt/lowstate", "crc_valid": True, "capture_time": 1000.0,
                      "joints": [{"name": "test_knee", "index": 3, "q_rad": 0.0},
                                 {"name": "test_elbow", "index": 18, "q_rad": 0.2}]}
        self.trajectory = {"schema_version": 1, "profile_id": "synthetic-test-only", "waypoints": [
            {"time_s": 0.0, "positions_rad": {"test_knee": 0.0}},
            {"time_s": 1.0, "positions_rad": {"test_knee": 0.4}},
        ]}

    def preview(self):
        return jc.validate_and_preview(self.profile, self.state, self.trajectory, now=1000.1)

    def test_interpolates_at_100hz_and_holds_other_measured_joints(self):
        result = self.preview()
        self.assertFalse(result["executable"])
        self.assertEqual(result["sample_rate_hz"], 100)
        self.assertEqual(result["frame_count"], 101)
        self.assertEqual(result["controlled_joints"], ["test_knee"])
        self.assertEqual(result["held_joints"], ["test_elbow"])
        self.assertAlmostEqual(result["frames"][50]["positions_rad"]["test_knee"], .2)
        self.assertTrue(all(frame["positions_rad"]["test_elbow"] == .2 for frame in result["frames"]))
        self.assertTrue(all(abs(frame["velocities_rad_s"]["test_knee"]) <= .5 for frame in result["frames"]))
        self.assertFalse(result["validation"]["crc_computed_here"])

    def test_multisegment_preview_includes_exact_final_endpoint(self):
        self.trajectory["waypoints"].append({"time_s": 1.015, "positions_rad": {"test_knee": .403}})
        result = self.preview()
        self.assertEqual(result["frames"][-1]["time_s"], 1.015)
        self.assertAlmostEqual(result["frames"][-1]["positions_rad"]["test_knee"], .403)
        self.assertEqual(len({f["time_s"] for f in result["frames"]}), result["frame_count"])

    def test_first_frame_is_measured_even_within_seed_tolerance(self):
        self.trajectory["waypoints"][0]["positions_rad"]["test_knee"] = .004
        result = self.preview()
        self.assertEqual(result["frames"][0]["positions_rad"]["test_knee"], 0)
        self.assertEqual(self.trajectory["waypoints"][0]["positions_rad"]["test_knee"], .004)

    def test_seed_discontinuity_is_rejected(self):
        self.trajectory["waypoints"][0]["positions_rad"]["test_knee"] = .1
        with self.assertRaisesRegex(jc.ValidationError, "連続"):
            self.preview()

    def test_seed_adjustment_cannot_hide_velocity_limit_violation(self):
        self.trajectory["waypoints"][0]["positions_rad"]["test_knee"] = .004
        self.trajectory["waypoints"][1] = {"time_s": .1, "positions_rad": {"test_knee": .053}}
        with self.assertRaisesRegex(jc.ValidationError, "速度"):
            self.preview()

    def test_stale_future_or_nonfinite_state_timestamp_is_rejected(self):
        for timestamp in (999.0, 1001.0, float("nan"), float("inf"), True):
            with self.subTest(timestamp=timestamp):
                self.state["capture_time"] = timestamp
                with self.assertRaises(jc.ValidationError):
                    self.preview()

    def test_bad_crc_or_wrong_source_is_rejected(self):
        for field, value in (("crc_valid", False), ("crc_valid", 1), ("source", "simulated")):
            with self.subTest(field=field, value=value):
                state = dict(self.state, **{field: value})
                with self.assertRaisesRegex(jc.ValidationError, "CRC"):
                    jc.validate_and_preview(self.profile, state, self.trajectory, now=1000.1)

    def test_profile_model_and_mode_machine_must_match(self):
        for field, value in (("profile_id", "other"), ("model_id", "other"), ("mode_machine", 1), ("mode_machine", True)):
            with self.subTest(field=field):
                state = dict(self.state, **{field: value})
                with self.assertRaises(jc.ValidationError):
                    jc.validate_and_preview(self.profile, state, self.trajectory, now=1000.1)

    def test_profile_requires_verified_layout_and_explicit_source_limits(self):
        for field in ("verified", "joint_layout_verified", "reference", "firmware"):
            with self.subTest(field=field):
                profile = copy.deepcopy(self.profile)
                del profile["source"][field]
                with self.assertRaises(jc.ValidationError):
                    jc.validate_and_preview(profile, self.state, self.trajectory, now=1000.1)
        for field in ("lower_rad", "upper_rad", "max_velocity_rad_s"):
            with self.subTest(field=field):
                profile = copy.deepcopy(self.profile)
                del profile["joints"][0][field]
                with self.assertRaises(jc.ValidationError):
                    jc.validate_and_preview(profile, self.state, self.trajectory, now=1000.1)

    def test_duplicate_profile_names_or_indices_are_rejected(self):
        for field, value in (("name", "test_knee"), ("index", 3)):
            with self.subTest(field=field):
                profile = copy.deepcopy(self.profile)
                profile["joints"][1][field] = value
                with self.assertRaisesRegex(jc.ValidationError, "重複"):
                    jc.validate_and_preview(profile, self.state, self.trajectory, now=1000.1)

    def test_unknown_missing_duplicate_or_mismatched_state_channels_are_rejected(self):
        variants = []
        state = copy.deepcopy(self.state); state["joints"][0]["name"] = "unknown"; variants.append(state)
        state = copy.deepcopy(self.state); state["joints"].pop(); variants.append(state)
        state = copy.deepcopy(self.state); state["joints"].append(dict(state["joints"][0])); variants.append(state)
        state = copy.deepcopy(self.state); state["joints"][0]["index"] = 4; variants.append(state)
        for state in variants:
            with self.subTest(state=state), self.assertRaises(jc.ValidationError):
                jc.validate_and_preview(self.profile, state, self.trajectory, now=1000.1)

    def test_unknown_changed_or_empty_trajectory_channels_are_rejected(self):
        for values in ({}, {"unknown": 0}, {"test_elbow": .2}):
            with self.subTest(values=values):
                trajectory = copy.deepcopy(self.trajectory)
                trajectory["waypoints"][1]["positions_rad"] = values
                with self.assertRaises(jc.ValidationError):
                    jc.validate_and_preview(self.profile, self.state, trajectory, now=1000.1)

    def test_target_and_measured_position_limits_are_enforced(self):
        self.trajectory["waypoints"][1]["positions_rad"]["test_knee"] = 1.1
        with self.assertRaisesRegex(jc.ValidationError, "関節限界"):
            self.preview()
        self.trajectory["waypoints"][1]["positions_rad"]["test_knee"] = .4
        self.state["joints"][1]["q_rad"] = -2.1
        with self.assertRaisesRegex(jc.ValidationError, "関節限界"):
            self.preview()

    def test_velocity_limits_are_enforced_before_sampling(self):
        self.trajectory["waypoints"][1]["time_s"] = .5
        with self.assertRaisesRegex(jc.ValidationError, "速度"):
            self.preview()

    def test_nonfinite_targets_and_boolean_positions_are_rejected(self):
        for value in (float("nan"), float("inf"), True, "0.2"):
            with self.subTest(value=value):
                self.trajectory["waypoints"][1]["positions_rad"]["test_knee"] = value
                with self.assertRaises(jc.ValidationError):
                    self.preview()

    def test_invalid_times_and_unbounded_duration_are_rejected(self):
        for value in (0, -1, 31, float("inf"), True):
            with self.subTest(value=value):
                self.trajectory["waypoints"][1]["time_s"] = value
                with self.assertRaises(jc.ValidationError):
                    self.preview()

    def test_unsupported_command_fields_are_not_silently_ignored(self):
        self.trajectory["waypoints"][1]["torque"] = 2
        with self.assertRaisesRegex(jc.ValidationError, "だけ"):
            self.preview()

    def test_json_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.json"
            path.write_text('{"positions_rad":{"knee":0,"knee":1}}', encoding="utf-8")
            with self.assertRaisesRegex(jc.ValidationError, "重複"):
                jc.load_json(path)

    def test_cli_is_preview_only_and_writes_no_output_on_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value in (("profile", self.profile), ("state", self.state), ("trajectory", self.trajectory)):
                (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
            output = root / "preview.json"
            args = ["--profile", str(root / "profile.json"), "--state", str(root / "state.json"),
                    "--trajectory", str(root / "trajectory.json"), "--now", "1000.1", "--output", str(output)]
            self.assertEqual(jc.main(args), 0)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["executable"])
            (root / "state.json").write_text("{}", encoding="utf-8")
            output.unlink()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(jc.main(args), 2)
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(error.getvalue())["executable"])


if __name__ == "__main__":
    unittest.main()
