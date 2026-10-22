"""Offline regression checks: required videos cannot disappear from an episode.

No hardware, torch import, or real dataset directory is used.
"""
from contextlib import ExitStack, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import pack_episode as pe

KEY = "observation.images.headband_left_rgb"
SLOTS = {"ego_headband": [("ego_headband.npz", "left_timestamps",
                           "left.mp4", "headband_left")]}


class _Dataset:
    roots = []

    @classmethod
    def create(cls, *, root, **kwargs):
        root = Path(root)
        root.mkdir(parents=True)
        cls.roots.append(root)
        return types.SimpleNamespace(
            root=root, meta=types.SimpleNamespace(total_episodes=0, total_frames=3))


class VideoFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test-pack-video-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "2026-10-21-18-01-25"
        ego = self.source / "ego_headband"
        ego.mkdir(parents=True)
        np.savez(ego / "ego_headband.npz", left_timestamps=np.arange(3) / 30,
                 imu0_ts=np.arange(3) / 200,
                 imu0_accel=np.zeros((3, 3)), imu0_gyro=np.zeros((3, 3)))
        (ego / "left.mp4").write_bytes(b"source placeholder")
        self.target = self.base / "upload-root" / "episode"
        _Dataset.roots = []

    def _run_main(self, *, worker=None, validator=None, specs=None, force=False):
        mf = types.ModuleType("mf_lerobot")
        mf.MultiFrequencyLeRobotDataset = _Dataset
        mf.utils = types.SimpleNamespace(DEFAULT_VIDEO_PATH=pe._VIDEO_PATH_FMT,
                                         DEFAULT_DATA_PATH=pe._DATA_PATH_FMT)
        info = {"dir": self.source, "win": (0.0, 0.1),
                "master": np.arange(3) / 30, "present": {KEY}}

        def good_worker(*args):
            for index in (4, 5):
                path = Path(args[index])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"complete mocked video or timestamp parquet")
            return pe._VideoResult("ego_headband", KEY, 3, "")

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"mf_lerobot": mf}))
            overrides = {
                "ProcessPoolExecutor": ThreadPoolExecutor,
                "discover_video_slots": Mock(return_value=SLOTS),
                "_probe_session_index": Mock(return_value=info),
                "_build_specs": Mock(return_value=(specs if specs is not None
                                                   else {KEY: {"dtype": "video"}})),
                "load_task_label": Mock(return_value="test task"),
                "load_parquet_streams": Mock(return_value=[]),
                "_cut_video_job": worker or good_worker,
                "_write_episode": Mock(),
                "_cleanup_images": Mock(),
                "write_qc_meta": Mock(),
                "write_collect_meta": Mock(),
                "write_calibrations_meta": Mock(),
                "validate_video_outputs": validator or Mock(),
            }
            for name, value in overrides.items():
                stack.enter_context(patch.object(pe, name, value))
            output = io.StringIO()
            args = [str(self.source), "--out", str(self.target), "--workers", "1"]
            if force:
                args.append("--force")
            with redirect_stdout(output):
                rc = pe.main(args, gated=False)
            return rc, output.getvalue()

    def test_npz_declared_video_missing_is_error(self):
        (self.source / "ego_headband" / "left.mp4").unlink()
        with self.assertRaisesRegex(RuntimeError, "left.mp4"):
            pe._source_video_keys(self.source, {})

    def test_single_imu_and_no_calibration_do_not_hide_video(self):
        self.assertEqual(pe._source_video_keys(self.source, SLOTS), {KEY})

    def test_missing_result_is_error(self):
        with self.assertRaises(RuntimeError):
            pe._require_video_results({}, {KEY})

    def test_zero_result_is_error(self):
        res = pe._VideoResult("ego_headband", KEY, 0, "")
        with self.assertRaises(RuntimeError):
            pe._require_video_results({"headband_left": res}, {KEY})

    def test_writer_does_not_save_when_required_video_failed(self):
        ds = Mock()
        res = pe._VideoResult("ego_headband", KEY, 0, "")
        with self.assertRaises(RuntimeError):
            pe._write_episode(ds, self.source, "task", np.arange(3) / 30,
                              [], (0.0, 0.1), {"headband_left": res}, {KEY})
        ds.save_episode.assert_not_called()
        ds.add_frame.assert_not_called()

    def test_worker_zero_cut_is_error(self):
        ego = self.source / "ego_headband"
        with patch.object(pe, "cut_video_aligned", return_value=0):
            with self.assertRaisesRegex(RuntimeError, KEY):
                pe._cut_video_job(str(ego), "ego_headband.npz", "left_timestamps",
                                  "left.mp4", str(self.base / "out.mp4"),
                                  str(self.base / "out.parquet"), 0,
                                  (0.0, 0.1), 0.0, KEY, 30)

    def test_worker_missing_source_is_error(self):
        ego = self.source / "ego_headband"
        (ego / "left.mp4").unlink()
        with self.assertRaises(FileNotFoundError):
            pe._cut_video_job(str(ego), "ego_headband.npz", "left_timestamps",
                              "left.mp4", str(self.base / "out.mp4"),
                              str(self.base / "out.parquet"), 0,
                              (0.0, 0.1), 0.0, KEY, 30)

    def test_main_missing_video_spec_returns_failure(self):
        rc, output = self._run_main(specs={"other": {"dtype": "float32"}})
        self.assertEqual(rc, 1)
        self.assertFalse(self.target.exists())
        self.assertNotIn("[done]", output)

    def test_main_zero_result_returns_failure_without_published_directory(self):
        worker = lambda *a: pe._VideoResult("ego_headband", KEY, 0, "")
        rc, output = self._run_main(worker=worker)
        self.assertEqual(rc, 1)
        self.assertFalse(self.target.exists())
        self.assertNotIn("[done]", output)
        self.assertTrue(_Dataset.roots)
        self.assertTrue(all(not p.exists() for p in _Dataset.roots))

    def test_main_missing_artifact_returns_failure(self):
        worker = lambda *a: pe._VideoResult("ego_headband", KEY, 3, "")
        rc, output = self._run_main(worker=worker)
        self.assertEqual(rc, 1)
        self.assertFalse(self.target.exists())
        self.assertNotIn("[done]", output)

    def test_final_validation_failure_cannot_publish(self):
        validator = Mock(side_effect=RuntimeError("video/parquet counts differ"))
        rc, output = self._run_main(validator=validator)
        self.assertEqual(rc, 1)
        self.assertFalse(self.target.exists())
        validator.assert_called_once()
        self.assertNotIn("[done]", output)
        self.assertTrue(all(not p.exists() for p in _Dataset.roots))

    def test_success_validates_system_temp_before_publication(self):
        def validate(ds, keys, ep_idx):
            self.assertEqual(keys, {KEY})
            self.assertEqual(ep_idx, 0)
            self.assertTrue(ds.root.is_dir())
            self.assertFalse(self.target.exists())
            self.assertFalse(ds.root.is_relative_to(self.target.parent))
            self.assertEqual(len(list(ds.root.rglob("*.mp4"))), 1)
            self.assertEqual(len(list(ds.root.rglob("*.parquet"))), 1)
        rc, output = self._run_main(validator=validate)
        self.assertEqual(rc, 0)
        self.assertTrue(self.target.is_dir())
        self.assertIn("[done]", output)
        self.assertEqual(len(list(self.target.rglob("*.mp4"))), 1)

    def test_force_validation_failure_preserves_old_package(self):
        self.target.mkdir(parents=True)
        (self.target / "old.txt").write_text("old valid package")
        validator = Mock(side_effect=RuntimeError("bad video"))
        rc, _ = self._run_main(validator=validator, force=True)
        self.assertEqual(rc, 1)
        self.assertEqual((self.target / "old.txt").read_text(), "old valid package")

    def test_worker_frame_count_mismatch_is_error(self):
        ego = self.source / "ego_headband"
        with patch.object(pe, "cut_video_aligned", return_value=2):
            with self.assertRaisesRegex(RuntimeError, KEY):
                pe._cut_video_job(str(ego), "ego_headband.npz", "left_timestamps",
                                  "left.mp4", str(self.base / "out.mp4"),
                                  str(self.base / "out.parquet"), 0,
                                  (0.0, 0.1), 0.0, KEY, 30)

    def test_video_metadata_failure_is_not_suppressed(self):
        ds = Mock()
        ds._features = {KEY: Mock()}
        ds.meta.update_video_info.side_effect = RuntimeError("missing video metadata")
        res = pe._VideoResult("ego_headband", KEY, 3, "")
        with patch.object(pe, "load_microphone_index", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "missing video metadata"):
                pe._write_episode(ds, self.source, "task", np.arange(3) / 30,
                                  [], (0.0, 0.1), {"headband_left": res}, {KEY})

    def test_force_success_only_replaces_old_package_after_validation(self):
        self.target.mkdir(parents=True)
        (self.target / "old.txt").write_text("old valid package")
        def validate(ds, keys, ep_idx):
            self.assertEqual((self.target / "old.txt").read_text(), "old valid package")
        rc, output = self._run_main(validator=validate, force=True)
        self.assertEqual(rc, 0)
        self.assertFalse((self.target / "old.txt").exists())
        self.assertEqual(len(list(self.target.rglob("*.mp4"))), 1)
        self.assertIn("[done]", output)

    def test_publication_failure_restores_forced_old_package(self):
        self.target.mkdir(parents=True)
        (self.target / "old.txt").write_text("old valid package")
        staged = self.base / "staging" / "new"
        staged.mkdir(parents=True)
        (staged / "new.txt").write_text("new valid package")
        real_move = pe.shutil.move
        calls = []
        def move(src, dst):
            calls.append((src, dst))
            if len(calls) == 2:
                raise OSError("simulated publication error")
            return real_move(src, dst)
        with patch.object(pe.shutil, "move", side_effect=move):
            with self.assertRaises(OSError):
                pe._publish_output(staged, self.target, True)
        self.assertEqual((self.target / "old.txt").read_text(), "old valid package")
        self.assertFalse((self.target / "new.txt").exists())


if __name__ == "__main__":
    unittest.main()
