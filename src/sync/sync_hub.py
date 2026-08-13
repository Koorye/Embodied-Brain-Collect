"""sync_hub.py -- master clock translator + marker bus.

A single Python process that runs alongside the recorders.  It is the
*only* component that talks to E-Prime.

Inbound  : UDP datagrams on :9999 from E-Prime InLine code.
           Packet format (ASCII, pipe-separated):

                EVT|trial=<int>|tag=<NAME>|code=<int>|t_eprime_ms=<int>

           Extra ``key=val`` fields are accepted and stashed verbatim.

Outbound : ZMQ PUB on tcp://127.0.0.1:9998 with topic ``"marker"`` (the
           recorders subscribe to this topic and dump events into their
           own .npz alongside their data).

Side-effects:
   - Stamps each event with the authoritative PC clock
     (``time.time()``  -> ``t_pc_recv``;  ``time.time_ns()``  -> ``t_neon_unix_ns``).
   - If a Pupil Labs Neon ``--neon-ip`` is given, also forwards the event
     into the Neon device via ``simple.Device.send_event(...)`` so that
     it ends up in the Neon recording with native ns-precision timestamps.
   - On Ctrl+C, persists every event to ``<out>/markers.npz`` with the
     fields documented in record.tools.inspect_markers.

CLI:
    python -m record.sync.sync_hub --out C:\\sessions\\s01_run1
    python -m record.sync.sync_hub --out ./out --neon-ip 172.16.20.10

The two ports (UDP 9999, ZMQ 9998) are also configurable.

Bound to ``127.0.0.1`` by default (single-PC setup).  Pass
``--bind 0.0.0.0`` to accept E-Prime packets from a separate stim PC.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from . import marker_codes


# ---------------------------------------------------------------------------
# Packet parsing
# ---------------------------------------------------------------------------

def parse_packet(data: bytes) -> dict[str, Any] | None:
    """Parse the E-Prime ASCII pipe format into a dict.

    Returns ``None`` if the packet is not a valid ``EVT|...`` event.
    """
    try:
        text = data.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    if not text.startswith("EVT|"):
        return None

    fields: dict[str, str] = {}
    for tok in text.split("|")[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        fields[k.strip()] = v.strip()

    try:
        out: dict[str, Any] = {
            "trial": int(fields.get("trial", -1)),
            "tag":   fields.get("tag", "?"),
            "code":  int(fields.get("code", -1)),
            "t_eprime_ms": int(fields.get("t_eprime_ms", -1)),
            "raw":   text,
        }
    except ValueError:
        return None

    for k, v in fields.items():
        if k not in {"trial", "tag", "code", "t_eprime_ms"}:
            out[f"extra_{k}"] = v
    return out


# ---------------------------------------------------------------------------
# Optional Pupil Labs Neon forwarder
# ---------------------------------------------------------------------------

class NeonForwarder:
    """Lazy + tolerant.  Failures never block the main UDP loop."""

    def __init__(self, ip: str, port: int = 8080, timeout: float = 5.0):
        self.ip = ip
        self.port = port
        self._dev: Any = None
        self._lock = threading.Lock()
        self._failed = False
        try:
            from pupil_labs.realtime_api.simple import Device  # noqa
            self._Device = Device
        except Exception as exc:
            print(f"[neon] disabled: cannot import pupil_labs.realtime_api ({exc})")
            self._failed = True
            return
        try:
            self._dev = self._Device(address=ip, port=port)
            print(f"[neon] connected: {ip}:{port}")
        except Exception as exc:
            print(f"[neon] connect failed: {exc}; will retry per-event")
            self._dev = None

    def send(self, evt: dict[str, Any], t_pc_ns: int) -> None:
        if self._failed:
            return
        with self._lock:
            if self._dev is None:
                try:
                    self._dev = self._Device(address=self.ip, port=self.port)
                except Exception:
                    return
            name = f"{evt['tag']}|0x{int(evt['code']):02X}|trial={int(evt['trial'])}"
            try:
                self._dev.send_event(name, event_timestamp_unix_ns=t_pc_ns)
            except Exception as exc:
                print(f"[neon] send_event failed ({exc}); reconnecting next event")
                try:
                    self._dev.close()
                except Exception:
                    pass
                self._dev = None

    def close(self) -> None:
        try:
            if self._dev is not None:
                self._dev.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SyncHub main loop
# ---------------------------------------------------------------------------

class SyncHub:
    def __init__(
        self,
        out_dir: Path,
        bind: str = "127.0.0.1",
        udp_port: int = 9999,
        zmq_port: int = 9998,
        neon: NeonForwarder | None = None,
        verbose: bool = True,
        flush_every_events: int = 50,
        flush_every_seconds: float = 5.0,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.markers_path = self.out_dir / "markers.npz"
        self.bind = bind
        self.udp_port = udp_port
        self.zmq_port = zmq_port
        self.neon = neon
        self.verbose = verbose
        self.flush_every_events = flush_every_events
        self.flush_every_seconds = flush_every_seconds
        self.rows: list[dict[str, Any]] = []
        self._last_flush_n = 0
        self._last_flush_t = 0.0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind((self.bind, self.udp_port))
        sock.settimeout(0.5)

        ctx = zmq.Context.instance()
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://{self.bind}:{self.zmq_port}")

        print(f"[sync_hub] UDP   listen  -> {self.bind}:{self.udp_port}")
        print(f"[sync_hub] ZMQ   publish -> tcp://{self.bind}:{self.zmq_port}")
        print(f"[sync_hub] markers       -> {self.markers_path}")
        print(f"[sync_hub] Neon          -> {'on' if self.neon else 'off'}")
        print(f"[sync_hub] checkpoint    -> every {self.flush_every_events} events "
              f"or {self.flush_every_seconds:.1f}s")
        print(f"[sync_hub] ready. Ctrl+C to stop.\n")

        self._last_flush_t = time.time()

        try:
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(2048)
                except socket.timeout:
                    self._maybe_checkpoint()
                    continue
                except OSError:
                    break

                t_pc = time.time()
                t_pc_ns = time.time_ns()
                evt = parse_packet(data)
                if evt is None:
                    if self.verbose:
                        print(f"[sync_hub] bad packet from {addr}: {data!r}")
                    continue

                evt["t_pc_recv"] = t_pc
                evt["t_neon_unix_ns"] = 0

                if self.neon is not None:
                    self.neon.send(evt, t_pc_ns)
                    evt["t_neon_unix_ns"] = t_pc_ns

                self.rows.append(evt)

                try:
                    pub.send_string("marker " + json.dumps(evt))
                except zmq.ZMQError as exc:
                    print(f"[sync_hub] zmq publish failed: {exc}")

                if self.verbose:
                    code = int(evt["code"])
                    resolved = marker_codes.name_of(code)
                    flag = " " if marker_codes.is_known(code) else "!"
                    print(f"[sync_hub]{flag} t={t_pc:.4f}  trial={int(evt['trial']):>4}  "
                          f"0x{code:02X}({code:3d})  {resolved:<14s} "
                          f"raw_tag={evt['tag']}")

                self._maybe_checkpoint()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                sock.close()
            except Exception:
                pass
            try:
                pub.close(0)
            except Exception:
                pass
            if self.neon is not None:
                self.neon.close()
            self._save(final=True)
        return 0

    def _maybe_checkpoint(self) -> None:
        """Atomic checkpoint write so a hard kill never loses more than
        ``flush_every_events`` markers (or ``flush_every_seconds`` of idle)."""
        n = len(self.rows)
        now = time.time()
        if (n - self._last_flush_n >= self.flush_every_events
            or (n != self._last_flush_n
                and now - self._last_flush_t >= self.flush_every_seconds)):
            self._save(final=False)
            self._last_flush_n = n
            self._last_flush_t = now

    def _save(self, final: bool) -> None:
        n = len(self.rows)
        if n == 0:
            if final:
                print("[sync_hub] no events received; skipping save.")
            return

        # write to a tmp file then atomic rename so a power loss mid-write
        # can't corrupt the checkpoint.  np.savez auto-appends .npz if the
        # path doesn't end in one, so we keep a .npz suffix on tmp too.
        tmp = self.markers_path.with_name(self.markers_path.stem + ".tmp.npz")
        np.savez(
            tmp,
            trial=np.array([r["trial"] for r in self.rows], dtype=np.int32),
            tag=np.array([r["tag"]   for r in self.rows]),
            code=np.array([r["code"] for r in self.rows], dtype=np.int32),
            t_eprime_ms=np.array([r["t_eprime_ms"] for r in self.rows], dtype=np.int64),
            t_pc_recv=np.array([r["t_pc_recv"]   for r in self.rows], dtype=np.float64),
            t_neon_unix_ns=np.array([r["t_neon_unix_ns"] for r in self.rows], dtype=np.int64),
            raw=np.array([r["raw"] for r in self.rows]),
        )
        os.replace(tmp, self.markers_path)
        if final:
            print(f"\n[sync_hub] wrote {n} events -> {self.markers_path}")
        elif self.verbose:
            print(f"[sync_hub] checkpoint: {n} events -> {self.markers_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="Session output directory (markers.npz goes here).")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="UDP/ZMQ bind address. 127.0.0.1 = single-PC; "
                         "0.0.0.0 = accept E-Prime from another machine.")
    ap.add_argument("--udp-port", type=int, default=9999)
    ap.add_argument("--zmq-port", type=int, default=9998)
    ap.add_argument("--neon-ip", default=None,
                    help="If given, forward each event to this Pupil Labs "
                         "Neon Companion phone (so Neon recording also has "
                         "the marker with native ns timestamp).")
    ap.add_argument("--quiet", action="store_true",
                    help="Don't print every event.")
    ap.add_argument("--flush-every-events", type=int, default=50,
                    help="Atomic checkpoint markers.npz every N events "
                         "(against hard kills). Default 50.")
    ap.add_argument("--flush-every-seconds", type=float, default=5.0,
                    help="Atomic checkpoint after N idle seconds with new "
                         "events buffered. Default 5.")
    args = ap.parse_args(argv)

    neon = NeonForwarder(args.neon_ip) if args.neon_ip else None

    hub = SyncHub(
        out_dir=Path(args.out),
        bind=args.bind,
        udp_port=args.udp_port,
        zmq_port=args.zmq_port,
        neon=neon,
        verbose=not args.quiet,
        flush_every_events=args.flush_every_events,
        flush_every_seconds=args.flush_every_seconds,
    )

    def _handler(signum: int, _frame: Any) -> None:
        print(f"\n[sync_hub] signal {signum}; flushing.")
        hub.stop()

    signal.signal(signal.SIGINT, _handler)
    # SIGBREAK is the signal Windows launcher sends via CTRL_BREAK_EVENT.
    # Without this, the launcher's stop signal would terminate sync_hub
    # without giving it a chance to write the final markers.npz.
    for _name in ("SIGBREAK", "SIGTERM", "SIGHUP"):
        _sig = getattr(signal, _name, None)
        if _sig is not None:
            try:
                signal.signal(_sig, _handler)
            except (ValueError, OSError):
                pass

    return hub.run()


if __name__ == "__main__":
    sys.exit(main())
