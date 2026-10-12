"""Offline calibration transfer tests: no headband, ROS, audio or GPU access.

Run with pytest, or with ``PYTHONPATH=src python -m unittest discover
-s tests/ego_headband -p test_calibration_transfer.py``.
"""

import hashlib
import json
import pickle
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from embodied_brain_collect.recorders.ego_headband.calibration_receiver import (
    CAPABILITY, FILENAMES, MAX_FILE_BYTES, CalibrationReceiver,
)
from embodied_brain_collect.recorders.ego_headband.ego_headband_recorder_config import (
    EgoHeadbandRecorderConfig,
)
from embodied_brain_collect.recorders.ego_headband import net_ego_headband_recorder as net


HEADER = struct.Struct("!II")


def snapshot(marker=b"first"):
    # Preserve comments, CRLF and UTF-8 exactly, rather than parsing YAML.
    return {name: b"# " + marker + b" \xe6\xa0\x87\xe5\xae\x9a\r\nvalue: "
            + str(index).encode() + b"\r\n"
            for index, name in enumerate(FILENAMES)}


def entries_for(files):
    return [{"name": name, "size": len(files[name]),
             "sha256": hashlib.sha256(files[name]).hexdigest()}
            for name in FILENAMES]


def add_snapshot(receiver, files):
    for entry in entries_for(files):
        receiver.add_file({"kind": "calibration_file", **entry}, files[entry["name"]])


def end_for(files):
    return {"kind": "calibration_end", "version": 1, "files": entries_for(files)}


def send_frame(peer, meta, payload=b""):
    raw = json.dumps(meta).encode("utf-8")
    peer.sendall(HEADER.pack(len(raw), len(payload)) + raw + payload)


def recv_frame(peer):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = peer.recv(size - len(result))
            if not chunk:
                raise EOFError("peer closed before frame completed")
            result.extend(chunk)
        return bytes(result)
    meta_size, payload_size = HEADER.unpack(exact(HEADER.size))
    return json.loads(exact(meta_size)), exact(payload_size)


class SocketPairClient:
    """Use real framing/read/write with a local Unix socket pair.

    TCP_NODELAY cannot apply to AF_UNIX; everything else delegates to the real
    socket. No production network endpoint is contacted.
    """

    def __init__(self, sock):
        self.sock = sock

    def setsockopt(self, level, option, value):
        if level != socket.IPPROTO_TCP:
            self.sock.setsockopt(level, option, value)

    def __getattr__(self, name):
        return getattr(self.sock, name)


class CalibrationReceiverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "ego_headband"
        self.files = snapshot()

    def receiver(self):
        receiver = CalibrationReceiver()
        add_snapshot(receiver, self.files)
        return receiver

    def assert_saved(self, files=None):
        files = self.files if files is None else files
        target = self.output / "calibrations"
        self.assertEqual({p.name for p in target.iterdir()}, set(FILENAMES) | {"manifest.json"})
        for name, contents in files.items():
            self.assertEqual((target / name).read_bytes(), contents)
        self.assertEqual(json.loads((target / "manifest.json").read_text()),
                         {"version": 1, "files": entries_for(files)})

    def test_complete_snapshot_is_published_byte_for_byte(self):
        receiver = self.receiver()
        self.assertEqual(receiver.finish(end_for(self.files), output_dir=self.output),
                         entries_for(self.files))
        self.assertTrue(receiver.complete)
        self.assert_saved()
        self.assertFalse(list(self.output.glob(".calibrations-*")))

    def test_missing_file_never_publishes(self):
        receiver = CalibrationReceiver()
        first = entries_for(self.files)[0]
        receiver.add_file(first, self.files[first["name"]])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            receiver.finish(end_for(self.files), output_dir=self.output)
        self.assertFalse(receiver.complete)
        self.assertFalse((self.output / "calibrations").exists())

    def test_rejects_duplicate_and_non_allowlisted_paths(self):
        entry = entries_for(self.files)[0]
        for name in ("../camera-camchain.yaml", "/tmp/camera-camchain.yaml",
                     "calibrations/camera-camchain.yaml", "camera.yaml", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CalibrationReceiver().add_file({**entry, "name": name}, self.files[FILENAMES[0]])
        receiver = self.receiver()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            receiver.add_file(entry, self.files[FILENAMES[0]])

    def test_rejects_bad_size_or_digest(self):
        entry = entries_for(self.files)[0]
        for change in ({"size": 0}, {"size": -1}, {"size": True},
                       {"size": MAX_FILE_BYTES + 1}, {"size": entry["size"] + 1},
                       {"sha256": "0" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                CalibrationReceiver().add_file({**entry, **change}, self.files[FILENAMES[0]])

    def test_rejects_bad_end_manifest_or_version(self):
        end = end_for(self.files)
        for change, payload in (({"version": True}, b""), ({"version": 2}, b""),
                                ({"files": end["files"][:-1]}, b""),
                                ({"files": list(reversed(end["files"]))}, b""),
                                ({}, b"unexpected payload")):
            with self.subTest(change=change, payload=payload), self.assertRaises(ValueError):
                self.receiver().finish({**end, **change}, payload, self.output)
        self.assertFalse((self.output / "calibrations").exists())

    def test_disk_error_leaves_no_partial_final_directory(self):
        receiver = self.receiver()
        actual_write = receiver._write
        calls = []

        def failing_write(path, data):
            calls.append(path.name)
            if len(calls) == 2:
                raise OSError("simulated disk full")
            actual_write(path, data)

        with patch.object(receiver, "_write", side_effect=failing_write):
            with self.assertRaisesRegex(OSError, "disk full"):
                receiver.finish(end_for(self.files), output_dir=self.output)
        self.assertFalse(receiver.complete)
        self.assertFalse((self.output / "calibrations").exists())
        self.assertFalse(list(self.output.glob(".calibrations-*")))

    def test_manifest_boolean_cannot_impersonate_one_byte_file_size(self):
        files = {name: b"x" for name in FILENAMES}
        receiver = CalibrationReceiver()
        add_snapshot(receiver, files)
        end = end_for(files)
        end["files"][0]["size"] = True
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            receiver.finish(end, output_dir=self.output)
        self.assertFalse(receiver.complete)

    def test_same_session_accepts_identical_fresh_transfer_but_never_overwrites(self):
        self.receiver().finish(end_for(self.files), output_dir=self.output)
        self.receiver().finish(end_for(self.files), output_dir=self.output)
        changed = snapshot(b"second")
        other = CalibrationReceiver()
        add_snapshot(other, changed)
        with self.assertRaisesRegex(ValueError, "differs"):
            other.finish(end_for(changed), output_dir=self.output)
        self.assert_saved()
        self.assertFalse(other.complete)

    def test_corrupted_existing_manifest_is_not_trusted(self):
        self.receiver().finish(end_for(self.files), output_dir=self.output)
        (self.output / "calibrations" / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.receiver().finish(end_for(self.files), output_dir=self.output)

    def test_preview_verifies_without_writing(self):
        receiver = self.receiver()
        receiver.finish(end_for(self.files))
        self.assertTrue(receiver.complete)
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            receiver.finish(end_for(self.files))
        with self.assertRaises(ValueError):
            receiver.add_file(entries_for(self.files)[0], self.files[FILENAMES[0]])


class CalibrationHandshakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_recorder(self, session="session"):
        config = EgoHeadbandRecorderConfig(
            session_dir=str(self.root / session) if session else "",
            connect_timeout=0.5, open_timeout=2.5, link_timeout=0,
            audio_enabled=False,
        )
        recorder = net.NetEgoHeadbandRecorder(config)
        self.addCleanup(self.clean_recorder, recorder)
        return recorder

    @staticmethod
    def clean_recorder(recorder):
        recorder._close_sock()
        if recorder._log_sink_id is not None:
            recorder.logger.remove(recorder._log_sink_id)
            recorder._log_sink_id = None

    def exchange(self, recorder, files=None, *, missing=False, early_stream=False):
        files = snapshot() if files is None else files
        client, server = socket.socketpair()
        server.settimeout(3.0)
        observed = {"acks": [], "errors": []}

        def serve():
            try:
                hello, payload = recv_frame(server)
                self.assertEqual(hello["kind"], "hello")
                self.assertIn(CAPABILITY, hello["capabilities"])
                self.assertEqual(payload, b"")
                if early_stream:
                    send_frame(server, {"kind": "stream_start", "topics": {}})
                else:
                    entries = entries_for(files)
                    for entry in entries[:-1] if missing else entries:
                        send_frame(server, {"kind": "calibration_file", **entry}, files[entry["name"]])
                    send_frame(server, end_for(files))
                try:
                    ack, payload = recv_frame(server)
                except EOFError:
                    return
                observed["acks"].append(ack)
                self.assertEqual(ack, {"kind": "calibration_ack", "version": 1,
                                       "files": entries_for(files)})
                self.assertEqual(payload, b"")
                if recorder.config.session_dir:
                    target = Path(recorder.config.session_dir) / recorder.output_dir / "calibrations"
                    for name, contents in files.items():
                        self.assertEqual((target / name).read_bytes(), contents,
                                         "ACK must follow all three durable file writes")
                    self.assertEqual(json.loads((target / "manifest.json").read_text()),
                                     {"version": 1, "files": entries_for(files)})
                send_frame(server, {"kind": "stream_start", "topics": {}})
            except Exception as exc:
                observed["errors"].append(exc)
            finally:
                server.close()

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            with patch.object(net.socket, "create_connection", return_value=SocketPairClient(client)), \
                    patch.object(net, "_probe_gpu_transcode"), \
                    patch.object(recorder, "_start_drain"):
                result = recorder._open()
        finally:
            recorder._close_sock()
            worker.join(timeout=4)
            server.close()
        self.assertFalse(worker.is_alive(), "fake server did not stop")
        if observed["errors"]:
            raise observed["errors"][0]
        return result, observed["acks"]

    def test_two_sessions_receive_fresh_bytes_and_ack_after_persistence(self):
        for session, marker in (("one", b"one"), ("two", b"two")):
            recorder = self.make_recorder(session)
            result, acks = self.exchange(recorder, snapshot(marker))
            self.assertTrue(result, recorder._open_error)
            self.assertEqual(len(acks), 1)
        for session, marker in (("one", b"one"), ("two", b"two")):
            for name, contents in snapshot(marker).items():
                self.assertEqual((self.root / session / "ego_headband" / "calibrations" / name).read_bytes(),
                                 contents)

    def test_old_server_cannot_start_recording_without_calibrations(self):
        recorder = self.make_recorder()
        result, acks = self.exchange(recorder, early_stream=True)
        self.assertFalse(result)
        self.assertEqual(acks, [])
        self.assertIn("calibrations missing", recorder._open_error)
        self.assertFalse((self.root / "session" / "ego_headband" / "calibrations").exists())

    def test_missing_file_fails_open_without_ack(self):
        recorder = self.make_recorder()
        result, acks = self.exchange(recorder, missing=True)
        self.assertFalse(result)
        self.assertEqual(acks, [])
        self.assertIn("incomplete", recorder._open_error)

    def test_disk_error_fails_open_without_ack(self):
        recorder = self.make_recorder()
        with patch.object(CalibrationReceiver, "_write", side_effect=OSError("disk full")):
            result, acks = self.exchange(recorder)
        self.assertFalse(result)
        self.assertEqual(acks, [])
        self.assertIn("disk full", recorder._open_error)
        self.assertFalse((self.root / "session" / "ego_headband" / "calibrations").exists())

    def test_commit_drops_preroll_but_preserves_calibration_fingerprints(self):
        recorder = self.make_recorder()
        self.assertTrue(self.exchange(recorder)[0])
        recorder._acc("imu0_ts", 1.0)
        recorder.request_commit()
        self.assertTrue(recorder._maybe_commit())
        self.assertFalse(recorder._buf)
        output = recorder._build_output()
        self.assertEqual(output["calibration_filenames"].tolist(), list(FILENAMES))
        self.assertEqual(output["calibration_sha256"].tolist(),
                         [entry["sha256"] for entry in entries_for(snapshot())])

    def test_empty_session_preview_uses_memory_only(self):
        recorder = self.make_recorder("")
        result, acks = self.exchange(recorder)
        self.assertTrue(result)
        self.assertEqual(len(acks), 1)
        self.assertTrue(recorder._calibration_receiver.complete)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_preview_cannot_turn_into_recording_without_new_saved_calibrations(self):
        recorder = self.make_recorder("")
        self.assertTrue(self.exchange(recorder)[0])
        with patch.object(net.BaseEgoHeadbandRecorder, "_record") as record:
            with self.assertRaisesRegex(ValueError, "session directory"):
                recorder._record()
            recorder.config.session_dir = str(self.root / "after-preview")
            with self.assertRaises(ValueError):
                recorder._record()
            record.assert_not_called()
        self.assertFalse((self.root / "after-preview").exists())

    def test_missing_saved_file_blocks_record_before_any_samples(self):
        recorder = self.make_recorder()
        self.assertTrue(self.exchange(recorder)[0])
        target = self.root / "session" / "ego_headband" / "calibrations"
        (target / FILENAMES[-1]).unlink()
        with patch.object(net.BaseEgoHeadbandRecorder, "_record") as record:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                recorder._record()
            record.assert_not_called()

    def test_old_pickled_parent_state_and_reopen_require_new_transfer(self):
        recorder = self.make_recorder()
        del recorder._calibration_receiver
        restored = pickle.loads(pickle.dumps(recorder))
        self.addCleanup(self.clean_recorder, restored)
        self.assertTrue(self.exchange(restored)[0])
        self.assertTrue(restored._calibration_receiver.complete)
        # A prior successful transfer and its saved directory cannot permit an
        # old server (or a reconnect missing files) to bypass this handshake.
        result, acks = self.exchange(restored, early_stream=True)
        self.assertFalse(result)
        self.assertEqual(acks, [])
        self.assertFalse(restored._calibration_receiver.complete)
        self.assertNotIn("calibration_sha256", restored._static)

    def test_incomplete_handshake_frame_obeys_timeout(self):
        recorder = self.make_recorder("")
        client, server = socket.socketpair()
        recorder._sock = client
        self.addCleanup(server.close)
        # Partial metadata must not keep the handshake alive indefinitely.
        server.sendall(HEADER.pack(100, 0) + b'{"kind":')
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            recorder._recv_frame(0.1)
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
