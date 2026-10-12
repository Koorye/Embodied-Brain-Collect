"""Receive a fresh, verified headband calibration snapshot for each connection."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


CAPABILITY = "headband_calibration_v1"
FILENAMES = (
    "camera-camchain.yaml",
    "camera_imu-camchain-imucam.yaml",
    "camera_imu-imu.yaml",
)
MAX_FILE_BYTES = 1024 * 1024


def _manifest_matches(value, entries):
    return (isinstance(value, list) and len(value) == len(entries)
            and all(isinstance(item, dict) and set(item) == {"name", "size", "sha256"}
                    and type(item["size"]) is int and item == expected
                    for item, expected in zip(value, entries)))


class CalibrationReceiver:
    """Bounded raw-byte transfer; never deserialize or rewrite calibration YAML."""

    def __init__(self):
        self._files = {}
        self._entries = {}
        self.complete = False

    def add_file(self, meta, payload):
        if self.complete:
            raise ValueError("calibration file received after calibration_end")
        name = meta.get("name")
        if not isinstance(name, str) or name not in FILENAMES:
            raise ValueError("unexpected calibration filename")
        if name in self._files:
            raise ValueError(f"duplicate calibration file: {name}")
        size = meta.get("size")
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES or len(payload) != size:
            raise ValueError(f"invalid calibration size: {name}")
        digest = hashlib.sha256(payload).hexdigest()
        if meta.get("sha256") != digest:
            raise ValueError(f"calibration SHA-256 mismatch: {name}")
        self._files[name] = bytes(payload)
        self._entries[name] = {"name": name, "size": size, "sha256": digest}

    def finish(self, meta, payload=b"", output_dir=None):
        """Validate the complete manifest and publish the directory before ACK.

        With no session directory (preview mode),
        keep a verified in-memory snapshot. Recorded sessions always persist it.
        """
        if self.complete or payload or type(meta.get("version")) is not int or meta["version"] != 1:
            raise ValueError("invalid or duplicate calibration_end")
        if set(self._files) != set(FILENAMES):
            missing = sorted(set(FILENAMES) - set(self._files))
            raise ValueError(f"incomplete headband calibrations; missing {missing}")
        entries = [self._entries[name] for name in FILENAMES]
        if not _manifest_matches(meta.get("files"), entries):
            raise ValueError("calibration manifest mismatch")
        if output_dir is not None:
            self._save(Path(output_dir), entries)
        self.complete = True
        return entries

    def verify_saved(self, output_dir):
        """Keep preview-only connections from becoming an uncalibrated recording."""
        if not self.complete:
            raise ValueError("headband calibrations are not complete")
        entries = [self._entries[name] for name in FILENAMES]
        self._verify_existing(Path(output_dir) / "calibrations", entries)

    def _save(self, output_dir, entries):
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "calibrations"
        # A session must not silently acquire a different calibration on retry.
        # Even an identical old directory is not a substitute for this transfer:
        # the freshly received bytes are validated above before checking it.
        if target.exists() or target.is_symlink():
            self._verify_existing(target, entries)
            return
        staging = Path(tempfile.mkdtemp(prefix=".calibrations-", dir=output_dir))
        try:
            for name in FILENAMES:
                self._write(staging / name, self._files[name])
            manifest = json.dumps({"version": 1, "files": entries}, indent=2).encode("utf-8") + b"\n"
            self._write(staging / "manifest.json", manifest)
            self._verify_existing(staging, entries)
            # Publish the complete set together; no partial final directory.
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @staticmethod
    def _write(path, data):
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def _verify_existing(self, target, entries):
        if target.is_symlink() or not target.is_dir():
            raise ValueError("calibration destination must be a real directory")
        if {p.name for p in target.iterdir()} != set(FILENAMES) | {"manifest.json"}:
            raise ValueError("existing calibration directory is incomplete or contains unexpected files")
        for name in (*FILENAMES, "manifest.json"):
            if (target / name).is_symlink() or not (target / name).is_file():
                raise ValueError(f"invalid calibration output file: {name}")
        for name in FILENAMES:
            if (target / name).read_bytes() != self._files[name]:
                raise ValueError(f"existing session calibration differs: {name}")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest != {"version": 1, "files": entries}:
            raise ValueError("saved calibration manifest differs from this connection")
