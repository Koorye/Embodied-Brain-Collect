"""Physiological wristband recorder over BLE (protocol V1.5).

Bleak is asyncio-based while the framework expects synchronous
``_open/_poll/_close`` methods.  Instead of a background thread, the
recorder drives Bleak's event loop directly from the record thread:
``_pump`` runs the loop for a few milliseconds, notification callbacks
fire synchronously into the decoder, and frames are parsed and stored in
the same thread.  No queue, no cross-thread state.

Scope is deliberately minimal: connect, sync the device clock (0x20),
store data frames while recording.  Setup (discover → connect → sync) is
retried as a whole up to ``open_attempts`` times within ``open_timeout`` —
the band often sits at marginal RSSI, and a link dropped during setup is
usually re-advertising seconds later.  Once open succeeds, every failure —
unexpected disconnect, rejected sync — lands in ``_error`` and is raised by
``_poll`` for the caller (``run()`` / launcher) to manage; recording never
auto-reconnects.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from ..base import BaseRecorder
from .protocol import (
    DataFrame,
    FrameStreamDecoder,
    TimeResponseFrame,
    build_time_sync_frame,
    parse_frame,
)
from .wristband_recorder_config import WristbandRecorderConfig


class WristbandRecorder(BaseRecorder):
    """Connect, UTC-sync, and store one physiological wristband."""

    name = "wristband"
    output_dir = "wristband"
    config: WristbandRecorderConfig

    def __init__(self, config: WristbandRecorderConfig):
        super().__init__(config)

        self._decoder = FrameStreamDecoder()
        # asyncio 资源在 _open() 里创建(普通属性可跨进程 pickle,
        # event loop 不行);_close() 关闭。
        self._ble_loop: asyncio.AbstractEventLoop | None = None
        self._session: asyncio.Task | None = None
        self._client: Any = None
        self._closing = False
        # 全部异常(发现/连接失败、意外掉线、同步被拒)都收敛到这一个
        # 字符串;open 阶段据此判定本次尝试失败并整体重试,录制开始后
        # _poll 统一 raise,不再做任何恢复。
        self._error = ""

        self._time_sync_ack = False
        self._time_sync_command: bytes | None = None
        self._time_sync_sent_unix_s: float | None = None
        self._time_sync_ack_unix_s: float | None = None
        self._record_end_unix_s: float | None = None

        self._selected_device_name = ""
        self._selected_device_address = ""

        self._data_frames_saved = 0

    # ------------------------------------------------------------------
    # Framework lifecycle: open / poll / close
    # ------------------------------------------------------------------

    def _open(self) -> bool:
        """Start BLE, connect, and complete the 0x20 UTC sync handshake.

        Each attempt runs the full discover → connect → sync sequence; a
        link that dies during setup gets up to ``open_attempts`` tries
        within the shared ``open_timeout`` budget.
        """

        attempts = max(1, self.config.open_attempts)
        deadline = time.monotonic() + max(1.0, self.config.open_timeout)
        self._ble_loop = asyncio.new_event_loop()

        for attempt in range(1, attempts + 1):
            self._reset_link_state()
            self._log(
                f"[wristband] scanning for BLE device "
                f"(service={self.config.service_uuid}, "
                f"attempt {attempt}/{attempts}) ..."
            )
            self._session = self._ble_loop.create_task(self._ble_session())

            while time.monotonic() < deadline:
                self._pump(0.02)
                if self._time_sync_ack or self._error:
                    break

            if self._time_sync_ack:
                self._log("[wristband] UTC sync acknowledged — ready")
                return True

            reason = self._error or (
                f"device was not ready within {self.config.open_timeout:g}s"
            )
            if attempt >= attempts or time.monotonic() >= deadline:
                self._open_error = reason
                break

            self._log(
                f"[wristband] open attempt {attempt}/{attempts} failed — "
                f"{reason}; retrying ..."
            )
            self._teardown_link()
            # 刚掉线的设备固件通常一两秒后才重新广播;稍等再扫。
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))

        self._log(f"[wristband] open failed — {self._open_error}")
        self._close()
        return False

    def _poll(self, ts: float) -> None:
        # ``ts`` is deliberately not used as the scientific sample time.  BLE
        # delivers one-second bursts, so device UTC in each 0x15 frame is the
        # authoritative time source.
        del ts
        self._pump(0.005)
        if self._error:
            raise RuntimeError(self._error)

    def _loop(self) -> None:
        super()._loop()
        # The device transmits the previous second's 50 frames as one 1 Hz
        # burst.  Keep listening briefly so the burst covering the session's
        # final second actually arrives before disconnect; frames sampled
        # after the stop boundary are dropped in _store_data_frame.
        self._record_end_unix_s = time.time()
        if self.config.tail_wait > 0:
            self._log(
                f"[wristband] draining final BLE burst for up to "
                f"{self.config.tail_wait:g}s"
            )
            deadline = time.monotonic() + self.config.tail_wait
            while time.monotonic() < deadline:
                self._pump(0.05)
                if self._error:
                    self._log(f"[wristband] tail drain ended early — {self._error}")
                    break

    def _close(self) -> None:
        self._closing = True
        if self._ble_loop is not None and not self._ble_loop.is_closed():
            self._teardown_link()
            self._ble_loop.close()
        self._session = None
        self._ble_loop = None
        self._log(f"[wristband] stopped (frames={self._data_frames_saved})")

    def _reset_link_state(self) -> None:
        """Clear per-attempt state so a stale echo/frame can't leak into a retry."""

        self._error = ""
        self._time_sync_ack = False
        self._time_sync_command = None
        self._time_sync_sent_unix_s = None
        self._time_sync_ack_unix_s = None
        self._decoder.reset()

    def _teardown_link(self) -> None:
        """Drop the failed attempt's link but keep the loop alive for a retry."""

        if self._ble_loop is None or self._ble_loop.is_closed():
            return
        if self._session is not None and not self._session.done():
            self._session.cancel()
            try:
                self._ble_loop.run_until_complete(self._session)
            except (asyncio.CancelledError, Exception):
                pass
        self._session = None
        if self._client is not None:
            try:
                self._ble_loop.run_until_complete(self._client.disconnect())
            except Exception:
                pass
            self._client = None

    def _recording_signal(self) -> int:
        # 只有 0x15 数据帧真正入库才算"数据在流动"——launcher 的录制
        # 确认以此为准,设备只连不发数据时不会误放行。
        return self._data_frames_saved

    # ------------------------------------------------------------------
    # asyncio plumbing (pumped from the record thread)
    # ------------------------------------------------------------------

    def _pump(self, seconds: float) -> None:
        """Run Bleak's event loop briefly; notification callbacks fire here."""
        if self._ble_loop is not None and not self._ble_loop.is_closed():
            self._ble_loop.run_until_complete(asyncio.sleep(seconds))

    async def _ble_session(self) -> None:
        """Discover, connect, enable notifications, send the UTC sync.

        Returns once the link is up — the connection itself lives on
        ``self._client`` and is serviced by every ``_pump`` call.
        """
        from bleak import BleakClient

        try:
            device = await self._discover_device()
            self._selected_device_name = getattr(device, "name", None) or ""
            self._selected_device_address = getattr(device, "address", "") or ""
            self._log(
                f"[wristband] connecting to "
                f"{self._selected_device_name or '(unnamed)'} "
                f"[{self._selected_device_address}] ..."
            )

            def on_disconnect(_client: Any) -> None:
                if not self._closing:
                    self._error = "BLE device disconnected"

            client = BleakClient(
                device,
                disconnected_callback=on_disconnect,
                timeout=self.config.connect_timeout,
            )
            self._client = client
            await client.connect()
            if not client.is_connected:
                raise ConnectionError("BLE connection did not become active")

            notify_char = client.services.get_characteristic(
                self.config.notify_uuid
            )
            write_char = client.services.get_characteristic(
                self.config.write_uuid
            )
            if notify_char is None:
                raise RuntimeError(
                    f"notify characteristic not found: {self.config.notify_uuid}"
                )
            if write_char is None:
                raise RuntimeError(
                    f"write characteristic not found: {self.config.write_uuid}"
                )

            await client.start_notify(notify_char, self._on_notification)

            # Build immediately before the write so the millisecond field is
            # as close as practical to the time observed by the firmware.
            command = build_time_sync_frame(sequence=0)
            self._time_sync_command = command
            self._time_sync_sent_unix_s = time.time()

            properties = {
                str(p).lower().replace("_", "-")
                for p in getattr(write_char, "properties", [])
            }
            await client.write_gatt_char(
                write_char,
                command,
                response=(
                    "write" in properties
                    and "write-without-response" not in properties
                ),
            )
            self._log("[wristband] BLE connected; 0x20 UTC sync sent")
        except Exception as exc:
            if not self._closing:
                self._error = f"{type(exc).__name__}: {exc}"

    async def _discover_device(self):
        from bleak import BleakScanner

        found = await BleakScanner.discover(
            timeout=self.config.scan_timeout,
            return_adv=True,
        )
        if isinstance(found, dict):
            entries = list(found.values())
        else:  # Compatibility fallback for older/custom Bleak backends.
            entries = [(device, None) for device in found]

        wanted_address = self.config.address.strip().lower()
        wanted_name = self.config.device_name.strip().lower()
        wanted_service = self.config.service_uuid.strip().lower()
        matches: list[tuple[Any, Any]] = []

        for device, advertisement in entries:
            address = (getattr(device, "address", "") or "").lower()
            local_name = (
                getattr(advertisement, "local_name", None)
                or getattr(device, "name", None)
                or ""
            )
            services = {
                str(uuid).lower()
                for uuid in (getattr(advertisement, "service_uuids", None) or [])
            }

            if wanted_address:
                matched = address == wanted_address
            elif wanted_name:
                matched = wanted_name in local_name.lower()
            else:
                matched = wanted_service in services
            if matched:
                matches.append((device, advertisement))

        if not matches:
            visible = []
            for device, advertisement in entries[:20]:
                local_name = (
                    getattr(advertisement, "local_name", None)
                    or getattr(device, "name", None)
                    or "(unnamed)"
                )
                visible.append(f"{local_name}[{getattr(device, 'address', '?')}]")
            selector = (
                f"address={self.config.address!r}"
                if wanted_address
                else f"name contains {self.config.device_name!r}"
                if wanted_name
                else f"advertised service={self.config.service_uuid}"
            )
            suffix = "; visible: " + ", ".join(visible) if visible else ""
            raise RuntimeError(f"no wristband matched {selector}{suffix}")

        # If several identical devices are present and no explicit selector was
        # supplied, choose the strongest advertisement.  For formal collection
        # the printed address should then be copied into the config.
        def signal_strength(item: tuple[Any, Any]) -> int:
            rssi = getattr(item[1], "rssi", -999) if item[1] is not None else -999
            return int(rssi) if rssi is not None else -999

        matches.sort(key=signal_strength, reverse=True)
        return matches[0][0]

    def _on_notification(self, _sender: Any, data: bytearray) -> None:
        # Runs on the record thread inside _pump: feed the decoder directly,
        # no queue needed.
        for raw_frame in self._decoder.feed(bytes(data)):
            self._handle_complete_frame(raw_frame, time.time())

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------

    def _handle_complete_frame(self, raw_frame: bytes, host_received_s: float) -> None:
        try:
            parsed = parse_frame(raw_frame)
        except ValueError:
            return  # 字节流损坏:帧直接丢弃

        if isinstance(parsed, TimeResponseFrame):
            if parsed.raw == self._time_sync_command:
                if not self._time_sync_ack:
                    self._time_sync_ack_unix_s = host_received_s
                    self._time_sync_ack = True
                    self._log("[wristband] 0x20 UTC sync acknowledged")
            elif parsed.is_failure:
                self._error = "device rejected the 0x20 UTC sync command"
            return

        if not isinstance(parsed, DataFrame):
            return  # 0x17 电量/存储帧,不存
        if not parsed.timestamp_valid:
            return  # 设备时钟未同步(0x20 未生效)的帧不可用,直接丢弃
        end = self._record_end_unix_s
        if end is not None and parsed.timestamp_ms / 1000.0 > end:
            return  # tail drain 期间到达的、停止边界之后采样的帧

        self._data_frames_saved += 1
        t0 = parsed.timestamp_ms / 1000.0
        self._acc("frame_timestamp_unix_ms", parsed.timestamp_ms)
        self._acc("frame_timestamps", t0)
        self._acc("host_receive_timestamps", host_received_s)

        pressure_ts = (t0, t0 + 1.0 / 150.0, t0 + 2.0 / 150.0)
        for sample_ts, value in zip(pressure_ts, parsed.pressure_raw):
            self._acc("pressure_timestamps", sample_ts)
            self._acc("pressure_raw", value)

        ppg_ts = (t0, t0 + 1.0 / 100.0)
        for sample_ts, values in zip(ppg_ts, parsed.ppg_raw):
            self._acc("ppg_timestamps", sample_ts)
            self._acc_arr("ppg_raw", np.asarray(values, dtype=np.int32))

        self._acc("imu_timestamps", t0)
        self._acc_arr("accel_raw", np.asarray(parsed.accel_raw, dtype=np.int16))
        self._acc_arr("accel_m_s2", np.asarray(parsed.accel_m_s2, dtype=np.float32))
        self._acc_arr("gyro_raw", np.asarray(parsed.gyro_raw, dtype=np.int16))
        self._acc_arr("gyro_rad_s", np.asarray(parsed.gyro_rad_s, dtype=np.float32))

        # These values are intentionally stored for every 0x15 frame.  The
        # firmware updates them at 1 Hz and holds/repeats the most recent
        # value in the other 49 frames; no "new value" flag exists.
        self._acc("temperature_timestamps", t0)
        self._acc("temperature_raw", parsed.temperature_raw)
        self._acc(
            "temperature_c",
            np.nan if parsed.temperature_c is None else parsed.temperature_c,
        )
        self._acc("spo2_timestamps", t0)
        self._acc("spo2_raw", parsed.spo2_raw)
        self._acc(
            "spo2_percent",
            np.nan if parsed.spo2_percent is None else parsed.spo2_percent,
        )

    # ------------------------------------------------------------------
    # Diagnostics and output
    # ------------------------------------------------------------------

    def _heartbeat_stats(self, elapsed: float) -> str:
        del elapsed
        return (
            f"frames={self._data_frames_saved} "
            f"pressure={len(self._buf.get('pressure_raw', []))}"
        )
