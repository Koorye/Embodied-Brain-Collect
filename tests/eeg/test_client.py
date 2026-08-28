from __future__ import annotations

import socket
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import yaml

EVENT_STRUCT_BYTES = 536


@dataclass
class CurryPacket:
    code: int
    request: int
    start_sample: int
    packet_size: int
    body: bytes


class CurryProtocol:
    REQUEST_CHANNEL_INFO = 3
    REQUEST_BASIC_INFO = 6
    REQUEST_STREAMING_START = 8
    REQUEST_STREAMING_STOP = 9

    @staticmethod
    def build_header(request: int, samples: int = 0, body_size: int = 0) -> bytes:
        return (
            b"CTRL"
            + (2).to_bytes(2, "big")
            + int(request).to_bytes(2, "big")
            + int(samples).to_bytes(4, "big")
            + int(body_size).to_bytes(4, "big")
            + int(body_size).to_bytes(4, "big")
        )


class Curry9TrialStreamer:
    def __init__(self, config: Dict[str, Any]) -> None:
        device = config.get("device", {})
        stream = config.get("stream", {})

        self.host: str = device.get("host", "127.0.0.1")
        self.port: int = device.get("port", 4455)
        self.channels: int = device.get("channels", 65)
        self.timeout: float = device.get("timeout_seconds", 5.0)
        self.reconnect_interval: float = device.get("reconnect_interval_seconds", 1.0)
        self.eeg_channel_indices: List[int] = device.get("eeg_channel_indices", list(range(64)))
        self.trial_prefix_seconds: float = stream.get("trial_prefix_seconds", 1.0)
        self.ring_buffer_seconds: int = int(stream.get("ring_buffer_seconds", 20))

        self.socket: Optional[socket.socket] = None
        self.info: Dict[str, Any] = {}
        self.channel_labels: List[str] = []
        self.event_queue: Deque[Dict[str, Any]] = deque()
        self.ring_buffer = np.zeros((self.channels, 1), dtype=np.float32)
        self.latest_frame: Optional[np.ndarray] = None
        self.last_trial_signature: Optional[tuple[int, int]] = None
        self.packet_counter = 0

    def log(self, message: str) -> None:
        print(f"[acq_curry9] {message}", flush=True)

    def connect(self) -> None:
        self.disconnect()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        sock.setblocking(True)
        self.socket = sock
        self.log(f"connected to Curry9 host {self.host}:{self.port}")

    def disconnect(self) -> None:
        if self.socket is None:
            return
        try:
            self.socket.sendall(CurryProtocol.build_header(CurryProtocol.REQUEST_STREAMING_STOP))
        except OSError:
            pass
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self.socket = None
        self.log("socket disconnected")

    def recv_exact(self, size: int) -> bytes:
        if self.socket is None:
            raise ConnectionError("socket is not connected")
        buffer = bytearray()
        while len(buffer) < size:
            chunk = self.socket.recv(size - len(buffer))
            if not chunk:
                raise ConnectionError("connection closed by peer")
            buffer.extend(chunk)
        return bytes(buffer)

    def read_packet(self, request: int, accepted_codes: List[int], accepted_requests: Optional[List[int]] = None) -> CurryPacket:
        if self.socket is None:
            raise ConnectionError("socket is not connected")
        self.socket.sendall(CurryProtocol.build_header(request))
        header = self.recv_exact(20)
        packet = CurryPacket(
            code=int.from_bytes(header[4:6], "big"),
            request=int.from_bytes(header[6:8], "big"),
            start_sample=int.from_bytes(header[8:12], "big"),
            packet_size=int.from_bytes(header[12:16], "big"),
            body=b"",
        )
        packet.body = self.recv_exact(packet.packet_size)
        if packet.code not in accepted_codes:
            raise ValueError(f"unexpected code={packet.code}")
        if accepted_requests is not None and packet.request not in accepted_requests:
            raise ValueError(f"unexpected request={packet.request}")
        return packet

    def request_basic_info(self) -> None:
        packet = self.read_packet(CurryProtocol.REQUEST_BASIC_INFO, accepted_codes=[1], accepted_requests=[2])
        self.info = {
            "eeg_channels": int(np.frombuffer(packet.body[4:8], dtype="<u4")[0]),
            "sample_rate": int(np.frombuffer(packet.body[8:12], dtype="<u4")[0]),
            "data_size": int(np.frombuffer(packet.body[12:16], dtype="<u4")[0]),
        }
        self.log(f"basic info: {self.info}")

    def request_channel_info(self) -> None:
        packet = self.read_packet(CurryProtocol.REQUEST_CHANNEL_INFO, accepted_codes=[1], accepted_requests=[4])
        labels: List[str] = []
        block_size = 120
        for index in range(self.info["eeg_channels"]):
            label = packet.body[index * block_size + 4: index * block_size + 10]
            labels.append(label.decode("ascii", errors="ignore").replace("\x00", ""))
        self.channel_labels = labels
        self.log(f"channel info: count={len(labels)}, labels={labels}")

    def parse_eeg_block(self, packet: CurryPacket) -> Optional[np.ndarray]:
        data_size = self.info.get("data_size", 0)
        channels = self.info.get("eeg_channels", 0)
        if data_size not in (2, 4) or channels <= 0:
            raise ValueError("invalid device info")

        valid_body_size = len(packet.body) - (len(packet.body) % data_size)
        if valid_body_size <= 0:
            return None
        values = np.frombuffer(packet.body[:valid_body_size], dtype=np.int16 if data_size == 2 else np.float32)
        valid_value_count = len(values) - (len(values) % channels)
        if valid_value_count <= 0:
            return None

        values = values[:valid_value_count]
        sample_count = valid_value_count // channels
        eeg = np.reshape(values, (channels, sample_count), order="F").astype(np.float32, copy=False)
        self.latest_frame = eeg
        self._log_eeg_block(packet.start_sample, eeg)
        return eeg

    def parse_event_block(self, packet: CurryPacket) -> None:
        if packet.packet_size % EVENT_STRUCT_BYTES != 0:
            return
        for offset in range(0, packet.packet_size, EVENT_STRUCT_BYTES):
            body = packet.body[offset: offset + EVENT_STRUCT_BYTES]
            event = {
                "type": int(np.frombuffer(body[0:4], dtype="<i4")[0]),
                "latency": int(np.frombuffer(body[4:8], dtype="<i4")[0]),
                "start": int(np.frombuffer(body[8:12], dtype="<i4")[0]),
                "end": int(np.frombuffer(body[12:16], dtype="<i4")[0]),
            }
            self.event_queue.append(event)
            self.log(f"event: {event}")

    def append_to_ring_buffer(self, eeg: np.ndarray) -> None:
        if eeg.size == 0:
            return
        self.ring_buffer = eeg.copy() if self.ring_buffer.shape[1] == 1 else np.hstack((self.ring_buffer, eeg))
        max_points = int(self.info.get("sample_rate", 1000) * self.ring_buffer_seconds)
        if self.ring_buffer.shape[1] > max_points:
            self.ring_buffer = self.ring_buffer[:, -max_points:]

    def try_build_trial(self) -> Optional[np.ndarray]:
        sample_rate = int(self.info.get("sample_rate", 1000))
        if self.latest_frame is None or self.ring_buffer.shape[1] < sample_rate:
            return None

        start_event: Optional[Dict[str, Any]] = None
        end_event: Optional[Dict[str, Any]] = None
        for event in self.event_queue:
            if event["type"] == 16:
                start_event = event
            elif event["type"] == 20 and start_event is not None and event["latency"] > start_event["latency"]:
                end_event = event
                break
        if start_event is None or end_event is None:
            return None

        signature = (start_event["latency"], end_event["latency"])
        if signature == self.last_trial_signature:
            return None

        prefix_points = int(self.trial_prefix_seconds * sample_rate)
        total_points = prefix_points + max(1, end_event["latency"] - start_event["latency"] + 1)
        if self.ring_buffer.shape[1] < total_points:
            return None

        trial_eeg = self.ring_buffer[self.eeg_channel_indices, -total_points:].copy()
        trigger = np.zeros((1, total_points), dtype=np.float32)
        for event in list(self.event_queue):
            if start_event["latency"] <= event["latency"] <= end_event["latency"]:
                trigger_index = prefix_points + event["latency"] - start_event["latency"]
                if 0 <= trigger_index < total_points:
                    trigger[0, trigger_index] = event["type"]

        self.last_trial_signature = signature
        self.event_queue = deque([event for event in self.event_queue if event["latency"] > end_event["latency"]])
        trial = np.concatenate((trial_eeg, trigger), axis=0)
        self._log_trial(trial, start_event, end_event)
        return trial

    def on_trial(self, trial: np.ndarray) -> None:
        return None

    def _log_eeg_block(self, start_sample: int, eeg: np.ndarray) -> None:
        preview_channels = min(2, eeg.shape[0])
        preview_samples = min(8, eeg.shape[1])
        preview = eeg[:preview_channels, :preview_samples].tolist()
        self.log(
            f"EEG block: start_sample={start_sample}, shape={eeg.shape}, "
            f"mean={float(eeg.mean()):.3f}, preview={preview}"
        )

    def _log_trial(self, trial: np.ndarray, start_event: Dict[str, Any], end_event: Dict[str, Any]) -> None:
        trigger_nonzero = trial[-1][trial[-1] != 0].tolist()
        preview_channels = min(2, trial.shape[0] - 1)
        preview_samples = min(8, trial.shape[1])
        eeg_preview = trial[:preview_channels, :preview_samples].tolist()
        self.log(
            "trial built: "
            f"shape={trial.shape}, start_event={start_event}, end_event={end_event}, "
            f"trigger_nonzero={trigger_nonzero}, eeg_preview={eeg_preview}"
        )

    def run(self) -> None:
        last_wait_log_time = 0.0
        try:
            while True:
                try:
                    if self.socket is None:
                        now = time.time()
                        if now - last_wait_log_time >= 2.0:
                            self.log(f"trying to connect to {self.host}:{self.port} ...")
                            last_wait_log_time = now
                        self.connect()
                        self.request_basic_info()
                        self.request_channel_info()
                    packet = self.read_packet(CurryProtocol.REQUEST_STREAMING_START, accepted_codes=[2, 3, 4])
                    self.packet_counter += 1
                    self.log(
                        f"packet #{self.packet_counter} received: code={packet.code}, request={packet.request}, "
                        f"start_sample={packet.start_sample}, packet_size={packet.packet_size}"
                    )
                    if packet.code == 2:
                        eeg = self.parse_eeg_block(packet)
                        if eeg is not None:
                            self.append_to_ring_buffer(eeg)
                            trial = self.try_build_trial()
                            if trial is not None:
                                self.on_trial(trial)
                    elif packet.code == 3:
                        self.parse_event_block(packet)
                    else:
                        self.log("keepalive packet received")
                except Exception as exc:
                    self.log(f"stream read failed: {exc}")
                    self.disconnect()
                    time.sleep(self.reconnect_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.disconnect()


def main() -> None:
    Curry9TrialStreamer({}).run()


if __name__ == "__main__":
    main()
