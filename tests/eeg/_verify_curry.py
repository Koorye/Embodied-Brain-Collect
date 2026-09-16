"""Headless smoke check: handshake + a few seconds of streaming + save.

End-to-end: open, poll ~3 s, close (runs alignment), save eeg.npz, then
reload it and report the timestamp array.
"""

import time

import numpy as np

from embodied_brain_collect.recorders.eeg import CurryEegRecorder, EegRecorderConfig

cfg = EegRecorderConfig(session_dir="tests/sessions/eeg")
rec = CurryEegRecorder(cfg)

assert rec._open(), "open failed"
t0 = time.time()
while time.time() - t0 < 3.0:
    rec._poll(time.time() - t0)
    time.sleep(0.05)
print(f"samples={rec._total_samples}  blocks={len(rec._blocks)}")

rec._close()
rec._save()

d = np.load("tests/sessions/eeg/eeg/eeg.npz")
print("npz keys:", d.files)
ts = d["eeg_timestamps_pc"]
print(f"eeg_timestamps_pc: shape={ts.shape}  first3={ts[:3]}  last3={ts[-3:]}")
print(f"monotonic={bool(np.all(np.diff(ts) > 0))}  "
      f"median_dt={float(np.median(np.diff(ts)))}")
