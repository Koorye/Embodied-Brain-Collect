"""Preflight checks for the multimodal recording rig.

Run before data collection to quickly identify missing devices without starting a
full session. The checks are intentionally short and conservative: they verify
ports, processes, network reachability, camera opens, disk space, and optionally
run the VIVE/OpenVR list-only check in the interactive desktop session.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS = ROOT / "sessions"
DEFAULT_CONFIG = ROOT / "session" / "preflight_config.json"

DEFAULT_ENABLED = {
    "disk": True,
    "serial_ports": True,
    "parallelbox": True,
    "emg": True,
    "eye": True,
    "manus": True,
    "tactile": True,
    "wrist_cameras": True,
    "vive": True,
}


def load_enabled_config(path: Path) -> dict[str, bool]:
    enabled = dict(DEFAULT_ENABLED)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(enabled, indent=2), encoding="utf-8")
        return enabled
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "enabled" in data and isinstance(data["enabled"], dict):
            data = data["enabled"]
        for key, value in data.items():
            if key in enabled:
                enabled[key] = bool(value)
    except Exception as exc:
        print(f"[preflight] WARN: failed to read {path}: {exc}; using defaults")
    return enabled


def skipped(name: str, key: str) -> CheckResult:
    return result(name, "SKIP", f"disabled by preflight config: {key}", config_key=key)


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any]


def result(name: str, status: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status=status, message=message, details=details)


def run_cmd(cmd: list[str], timeout: float = 10.0, cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return 124, out + f"\n[TIMEOUT after {timeout}s]"
    except Exception as exc:
        return 125, f"{type(exc).__name__}: {exc}"


def check_python() -> CheckResult:
    return result("python_env", "PASS", f"Python {sys.version.split()[0]}", executable=sys.executable)


def check_disk(path: Path, min_free_gb: float) -> CheckResult:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    status = "PASS" if free_gb >= min_free_gb else "WARN"
    return result("disk", status, f"free {free_gb:.1f} GB at {path}", path=str(path), free_gb=free_gb)


def list_serial_ports() -> list[dict[str, str]]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    return [{"device": p.device, "description": p.description or "", "hwid": p.hwid or ""}
            for p in list_ports.comports()]


def check_serial_presence(name: str, port: str, ports: list[dict[str, str]]) -> CheckResult:
    hit = next((p for p in ports if p["device"].upper() == port.upper()), None)
    if hit:
        return result(name, "PASS", f"{port} present", **hit)
    return result(name, "FAIL", f"{port} not found", available=[p["device"] for p in ports])


def check_parallelbox(port: str, baud: int, do_write: bool) -> CheckResult:
    try:
        import serial
        ser = serial.Serial(port, baud, timeout=0.5)
        try:
            if do_write:
                ser.write(bytes([0]))
                ser.flush()
            return result("parallelbox", "PASS", f"opened {port} @ {baud}", port=port, wrote_idle=do_write)
        finally:
            ser.close()
    except Exception as exc:
        return result("parallelbox", "FAIL", f"cannot open {port}: {exc}", port=port, error=str(exc))


def check_emg(port: str, baud: int, seconds: float) -> CheckResult:
    try:
        import serial
        header = b"\xD2\xD2\xD2"
        buf = bytearray()
        with serial.Serial(port, baud, timeout=0.05) as ser:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            t0 = time.time()
            while time.time() - t0 < seconds:
                chunk = ser.read(4096)
                if chunk:
                    buf.extend(chunk)
        headers = aa = bb = 0
        idx = 0
        while True:
            idx = buf.find(header, idx)
            if idx < 0:
                break
            headers += 1
            if idx + 3 < len(buf):
                typ = buf[idx + 3]
                aa += int(typ == 0xAA)
                bb += int(typ == 0xBB)
            idx += 1
        if headers >= 5 and aa + bb >= 5:
            return result("emg_stream", "PASS", f"EMG frames on {port}: AA={aa}, BB={bb}", port=port, bytes=len(buf), headers=headers, aa=aa, bb=bb)
        status = "WARN" if len(buf) else "FAIL"
        return result("emg_stream", status, f"no clear EMG protocol on {port}", port=port, bytes=len(buf), headers=headers, aa=aa, bb=bb)
    except Exception as exc:
        return result("emg_stream", "FAIL", f"cannot probe {port}: {exc}", port=port, error=str(exc))


def check_tcp(name: str, host: str, port: int, timeout: float = 2.0) -> CheckResult:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        t0 = time.time()
        s.connect((host, port))
        dt_ms = (time.time() - t0) * 1000.0
        return result(name, "PASS", f"{host}:{port} reachable", host=host, port=port, connect_ms=dt_ms)
    except Exception as exc:
        return result(name, "FAIL", f"{host}:{port} unreachable: {exc}", host=host, port=port, error=str(exc))
    finally:
        s.close()


def check_neon(ip: str, port: int, deep: bool) -> CheckResult:
    tcp = check_tcp("eye_neon", ip, port)
    if tcp.status != "PASS" or not deep:
        return tcp
    try:
        from pupil_labs.realtime_api.simple import Device
        dev = Device(address=ip, port=port)
        return result("eye_neon", "PASS", f"Neon API reachable: {dev.phone_name}", ip=ip, port=port,
                      phone_name=str(dev.phone_name), battery=getattr(dev, "battery_level_percent", None),
                      free_gb=(getattr(dev, "memory_num_free_bytes", 0) or 0) / 1e9)
    except Exception as exc:
        return result("eye_neon", "WARN", f"TCP ok but API check failed: {exc}", ip=ip, port=port, error=str(exc))


def check_process(name: str, patterns: list[str]) -> CheckResult:
    rc, out = run_cmd(["tasklist"], timeout=5)
    if rc != 0:
        return result(name, "WARN", "tasklist failed", output=out[-1000:])
    lower = out.lower()
    hits = [p for p in patterns if p.lower() in lower]
    if hits:
        return result(name, "PASS", f"process found: {', '.join(hits)}", hits=hits)
    return result(name, "FAIL", f"process not found: {', '.join(patterns)}", patterns=patterns)


def check_manus(deep: bool) -> CheckResult:
    proc = check_process("manus_core", ["ManusCore.exe", "ManusCore"])
    if proc.status != "PASS" or not deep:
        return proc
    env = os.environ.copy()
    # _manus_sdk.py is now in record/recorders/; add that to PYTHONPATH
    sdk_dir = str(ROOT / "recorders")
    env["PYTHONPATH"] = sdk_dir + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(ROOT / "tests" / "manus" / "discover.py"), "--lookup-seconds", "2", "--landscape-timeout", "5"]
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=12)
        status = "PASS" if p.returncode == 0 else ("WARN" if p.returncode == 1 else "FAIL")
        lines = [line for line in p.stdout.splitlines() if line.strip()]
        msg = "Manus SDK ok" if status == "PASS" else "Manus SDK check not fully healthy"
        return result("manus", status, msg, returncode=p.returncode, tail="\n".join(lines[-12:]))
    except Exception as exc:
        return result("manus", "WARN", f"process ok but SDK check failed: {exc}", error=str(exc))


def check_tactile_serial(port: str, baud: int) -> CheckResult:
    """Verify the tactile glove's USB-serial port both exists *and* is openable.

    Why both: the 2026-05-28 incident exposed a CH343 driver state where
    the port name is still in ``list_ports.comports()`` (so a "presence"
    check passes) but ``serial.Serial(...)`` raises
    ``OSError(22, ERROR_INVALID_FUNCTION=1)`` because the driver got stuck
    after a previous process holding the port crashed.  Only an actual
    open call catches this, and the recovery requires physically
    unplugging + replugging the glove's USB.
    """
    try:
        import serial
    except Exception as exc:
        return result("tactile_serial", "WARN",
                      f"pyserial import failed: {exc}", port=port, error=str(exc))
    try:
        with serial.Serial(port, baud, timeout=0.1) as ser:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
        return result("tactile_serial", "PASS",
                      f"{port} @ {baud} opened cleanly", port=port, baud=baud)
    except Exception as exc:
        msg = str(exc)
        # Translate the common Windows driver-stuck error into an
        # actionable hint for the operator.
        if "OSError(22" in msg or "ERROR_INVALID_FUNCTION" in msg or "WinError 1" in msg:
            hint = (
                f"{port} is visible to Windows but its USB-serial driver "
                f"is stuck (likely after a previous tactile process crashed). "
                f"ACTION: physically unplug the tactile glove USB, wait 3 s, "
                f"replug, then rerun preflight."
            )
        elif "PermissionError" in msg or "WinError 5" in msg or "Access is denied" in msg:
            hint = (
                f"{port} is held by another process.  Close any other "
                f"program that might be using the glove (TouchTronix demo, "
                f"a leftover Python recorder, etc.) and rerun preflight."
            )
        elif "FileNotFoundError" in msg or "WinError 2" in msg \
                or "cannot find the file specified" in msg.lower():
            hint = (
                f"{port} is not present.  Plug the tactile glove USB in "
                f"and verify Device Manager shows the CH343 adapter."
            )
        else:
            hint = f"cannot open {port}: {exc}.  Try unplug/replug the glove USB."
        return result("tactile_serial", "FAIL", hint,
                      port=port, baud=baud, error=str(exc))


def check_cameras(indices: list[int], width: int, height: int) -> CheckResult:
    try:
        import cv2
    except Exception as exc:
        return result("wrist_cameras", "WARN", f"cv2 import failed: {exc}", error=str(exc))
    rows = []
    ok = 0
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        opened = bool(cap.isOpened())
        got = False
        shape = None
        if opened:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(5):
                got, frame = cap.read()
                if got:
                    shape = tuple(frame.shape)
                    break
        cap.release()
        rows.append({"index": idx, "opened": opened, "frame": bool(got), "shape": shape})
        ok += int(opened and got)
    status = "PASS" if ok == len(indices) else ("WARN" if ok else "FAIL")
    return result("wrist_cameras", status, f"{ok}/{len(indices)} cameras returned frames", cameras=rows)


def write_vive_payload(payload: Path, log: Path) -> None:
    lines = [
        "@echo off",
        r"call C:\Users\31454\miniconda3\Scripts\activate.bat record",
        fr"cd /d {ROOT}",
        fr"python tests\vive\test_vive_openvr_capture.py --list-only --classes tracker,hmd,controller > {log} 2>&1",
    ]
    payload.write_text("\n".join(lines) + "\n", encoding="ascii")


def _vive_hint_from_log(text: str) -> str:
    """Translate OpenVR init error variants into a single actionable line."""
    if "InitError_Init_NoServerForBackgroundApp" in text:
        return ("ACTION: SteamVR is not running.  Open SteamVR from the "
                "Windows tray (or launch via Steam) and rerun preflight.")
    if "InitError_Init_Internal" in text:
        return ("ACTION: SteamVR is in a bad state.  Right-click the SteamVR "
                "tray icon -> Exit, wait 5 s, restart, then rerun preflight.")
    if "InitError_Init_VRClientDLLNotFound" in text:
        return ("ACTION: OpenVR runtime is missing.  Install/repair SteamVR "
                "via Steam and rerun preflight.")
    if "InitError_Init_HmdNotFound" in text:
        return ("ACTION: HMD/base stations are powered off or out of range.  "
                "Power them on, wait for SteamVR to detect them, then rerun.")
    if "OpenVR error" in text:
        return ("ACTION: OpenVR initialization failed -- check SteamVR is "
                "running on the interactive desktop and rerun.")
    if "Missing dependency" in text:
        return ("ACTION: the `openvr` Python package is not installed in the "
                "record conda env.  Run `pip install openvr` and rerun.")
    return ("ACTION: SteamVR appears not to see any trackers -- check that "
            "SteamVR is running AND has at least one base station + tracker "
            "online.")


def check_vive(timeout: float) -> CheckResult:
    vive_dir = ROOT / "tests" / "vive"
    log = vive_dir / "preflight_vive_list.log"
    payload = vive_dir / "preflight_vive_list_payload.bat"
    task = "RecordPreflightViveList"
    try:
        write_vive_payload(payload, log)
        create_cmd = ["schtasks", "/Create", "/TN", task, "/SC", "ONCE", "/ST", "23:59", "/F", "/IT", "/TR", f"cmd.exe /c {payload}"]
        rc, out = run_cmd(create_cmd, timeout=10)
        if rc != 0:
            return result("vive_openvr", "WARN", "could not create interactive task", output=out[-1500:])
        rc, out = run_cmd(["schtasks", "/Run", "/TN", task], timeout=10)
        if rc != 0:
            return result("vive_openvr", "WARN", "could not run interactive task", output=out[-1500:])
        deadline = time.time() + timeout
        text = ""
        while time.time() < deadline:
            if log.exists():
                text = log.read_text(encoding="utf-8", errors="replace")
                if "Connected OpenVR devices:" in text or "OpenVR error" in text or "Missing dependency" in text:
                    break
            time.sleep(0.5)
        if not text and log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
        trackers = []
        for line in text.splitlines():
            if " tracker " in line or "] tracker" in line:
                parts = line.strip().split("serial=")
                serial = parts[1].split()[0] if len(parts) > 1 else "?"
                trackers.append(serial)
        if len(trackers) >= 3:
            return result("vive_openvr", "PASS", f"{len(trackers)} trackers visible", trackers=trackers, log=str(log))
        if trackers:
            return result(
                "vive_openvr", "WARN",
                f"only {len(trackers)} tracker(s) visible -- {_vive_hint_from_log(text)}",
                trackers=trackers, log_tail=text[-1500:],
            )
        return result(
            "vive_openvr", "FAIL",
            f"no tracker visible via interactive OpenVR task -- "
            f"{_vive_hint_from_log(text)}",
            log_tail=text[-1500:], log=str(log),
        )
    except Exception as exc:
        return result("vive_openvr", "WARN", f"Vive check failed: {exc}", error=str(exc))


def print_summary(rows: list[CheckResult]) -> None:
    print("\n== Preflight summary ==")
    for r in rows:
        print(f"[{r.status:<4}] {r.name:<18} {r.message}")
    counts = {s: sum(1 for r in rows if r.status == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print(f"\nTotals: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neon-ip", default="172.16.19.213")
    ap.add_argument("--neon-port", type=int, default=8080)
    ap.add_argument("--emg-port", default="COM10")
    ap.add_argument("--parallelbox-port", default="COM14")
    ap.add_argument("--parallelbox-baud", type=int, default=115200)
    ap.add_argument("--emg-baud", type=int, default=921600)
    ap.add_argument("--emg-probe-seconds", type=float, default=0.6)
    ap.add_argument("--tactile-port", default="COM3",
                    help="COM port of the JR tactile glove (CH343 USB-serial).")
    ap.add_argument("--tactile-baud", type=int, default=921600)
    ap.add_argument("--camera-indices", default="0,1")
    ap.add_argument("--camera-width", type=int, default=320)
    ap.add_argument("--camera-height", type=int, default=240)
    ap.add_argument("--min-free-gb", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=ROOT / "sessions" / "preflight_latest.json")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON file with per-device enable switches")
    ap.add_argument("--deep-neon", action="store_true")
    ap.add_argument("--deep-manus", action="store_true")
    ap.add_argument("--skip-vive", action="store_true")
    ap.add_argument("--skip-cameras", action="store_true")
    ap.add_argument("--write-parallelbox-idle", action="store_true", help="write byte 0 to COM14 after opening")
    ap.add_argument("--exit-zero", action="store_true", help="always exit 0 after writing the report, even if checks fail")
    args = ap.parse_args(argv)

    enabled = load_enabled_config(args.config)
    rows: list[CheckResult] = []
    rows.append(check_python())

    if enabled.get("disk", True):
        rows.append(check_disk(DEFAULT_SESSIONS, args.min_free_gb))
    else:
        rows.append(skipped("disk", "disk"))

    ports = list_serial_ports()
    if enabled.get("serial_ports", True):
        rows.append(result("serial_ports", "PASS" if ports else "WARN", f"{len(ports)} serial port(s) visible", ports=ports))
    else:
        rows.append(skipped("serial_ports", "serial_ports"))

    if enabled.get("emg", True):
        rows.append(check_serial_presence("emg_port", args.emg_port, ports))
        rows.append(check_emg(args.emg_port, args.emg_baud, args.emg_probe_seconds))
    else:
        rows.append(skipped("emg", "emg"))

    if enabled.get("parallelbox", True):
        rows.append(check_serial_presence("parallelbox_port", args.parallelbox_port, ports))
        rows.append(check_parallelbox(args.parallelbox_port, args.parallelbox_baud, args.write_parallelbox_idle))
    else:
        rows.append(skipped("parallelbox", "parallelbox"))

    if enabled.get("eye", True):
        rows.append(check_neon(args.neon_ip, args.neon_port, args.deep_neon))
    else:
        rows.append(skipped("eye_neon", "eye"))

    if enabled.get("manus", True):
        rows.append(check_process("manus_core_process", ["ManusCore.exe", "ManusCore"]))
        if args.deep_manus:
            rows.append(check_manus(deep=True))
    else:
        rows.append(skipped("manus", "manus"))

    if enabled.get("tactile", True):
        rows.append(check_serial_presence(
            "tactile_port", args.tactile_port, ports
        ))
        rows.append(check_tactile_serial(args.tactile_port, args.tactile_baud))
    else:
        rows.append(skipped("tactile_serial", "tactile"))

    if enabled.get("wrist_cameras", True) and not args.skip_cameras:
        indices = [int(x.strip()) for x in args.camera_indices.split(",") if x.strip()]
        rows.append(check_cameras(indices, args.camera_width, args.camera_height))
    else:
        rows.append(skipped("wrist_cameras", "wrist_cameras"))

    if enabled.get("vive", True) and not args.skip_vive:
        rows.append(check_vive(timeout=12.0))
    else:
        rows.append(skipped("vive_openvr", "vive"))

    print(f"config: {args.config}")
    print_summary(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_unix_s": time.time(), "root": str(ROOT), "config": str(args.config), "enabled": enabled, "checks": [asdict(r) for r in rows]}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    if args.exit_zero:
        return 0
    return 1 if any(r.status == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
