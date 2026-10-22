"""Regression coverage for required-video failures in the fast packaging entrypoint."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pack_daily_fast as fast


KEY = "observation.images.ego_headband_right"


class FastPackagingVideoGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def _cut(self):
        return fast._cut_video_job(
            str(self.root), "stream.npz", "right_timestamp", "right.mp4",
            str(self.root / "out.mp4"), str(self.root), 0, (None, None),
            1.0, KEY, 30)

    def _inputs(self):
        np.savez(self.root / "stream.npz", right_timestamp=np.array([1.0, 1.04]))
        (self.root / "right.mp4").write_bytes(b"test fixture")

    def _dataset(self):
        return SimpleNamespace(
            features={KEY: {"dtype": "video"}}, _features={KEY: Mock()},
            meta=SimpleNamespace(total_episodes=7, update_video_info=Mock()),
            add_frame=Mock(), save_episode=Mock())

    def _write(self, ds, results):
        fast._write_episode(ds, self.root, "task", np.array([1.0, 1.04]),
                            [], (None, None), results)

    def test_missing_source_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._cut()

    def test_unreadable_timestamps_raise(self):
        self._inputs()
        (self.root / "stream.npz").write_bytes(b"corrupt npz")
        with self.assertRaisesRegex(RuntimeError, "Cannot read required video timestamps"):
            self._cut()

    def test_zero_frame_worker_result_raises(self):
        self._inputs()
        with patch.object(fast, "cut_video_aligned", return_value=0), \
                patch.object(fast, "_write_timestamps_parquet") as write:
            with self.assertRaisesRegex(RuntimeError, "produced no frames"):
                self._cut()
            write.assert_not_called()

    def test_worker_count_mismatch_raises(self):
        self._inputs()
        with patch.object(fast, "cut_video_aligned", return_value=1), \
                patch.object(fast, "_write_timestamps_parquet") as write:
            with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
                self._cut()
            write.assert_not_called()

    def test_missing_video_result_stops_before_save(self):
        ds = self._dataset()
        with self.assertRaisesRegex(RuntimeError, "outputs incomplete"):
            self._write(ds, {})
        ds.add_frame.assert_not_called()
        ds.save_episode.assert_not_called()

    def test_zero_frame_result_stops_before_save(self):
        ds = self._dataset()
        with self.assertRaisesRegex(RuntimeError, "outputs incomplete"):
            self._write(ds, {"right": fast._VideoResult("ego_headband", KEY, 0, "")})
        ds.add_frame.assert_not_called()
        ds.save_episode.assert_not_called()

    def test_output_validation_failure_stops_before_save(self):
        ds = self._dataset()
        with patch.object(fast.base, "validate_video_outputs", side_effect=RuntimeError("bad mp4")):
            with self.assertRaisesRegex(RuntimeError, "bad mp4"):
                self._write(ds, {"right": fast._VideoResult("ego_headband", KEY, 2, "")})
        ds.add_frame.assert_not_called()
        ds.save_episode.assert_not_called()

    def test_success_validates_before_save(self):
        ds = self._dataset()
        calls = []
        ds.save_episode.side_effect = lambda: calls.append("save")
        with patch.object(fast.base, "validate_video_outputs",
                          side_effect=lambda *args: calls.append("validate")) as validate:
            self._write(ds, {"right": fast._VideoResult("ego_headband", KEY, 2, "")})
        validate.assert_called_once_with(ds, [KEY], 7)
        self.assertEqual(calls, ["validate", "save"])
        ds.meta.update_video_info.assert_called_once_with()

    def test_video_metadata_error_is_not_swallowed(self):
        ds = self._dataset()
        ds.meta.update_video_info.side_effect = RuntimeError("video metadata failed")
        with patch.object(fast.base, "validate_video_outputs"):
            with self.assertRaisesRegex(RuntimeError, "video metadata failed"):
                self._write(ds, {"right": fast._VideoResult("ego_headband", KEY, 2, "")})

    def test_missing_required_video_spec_raises(self):
        slot = self.root / "ego_headband"
        slot.mkdir()
        suffix = "ego_headband_right"
        key = fast.video_feature_key(suffix)
        slots = {"ego_headband": [("stream.npz", "right_timestamp", "right.mp4", suffix)]}
        sessions = [{"dir": self.root, "stream_meta": {}}]
        with self.assertRaisesRegex(RuntimeError, "Required video specs missing"):
            fast._build_specs(sessions, slots, {key})


if __name__ == "__main__":
    unittest.main()
