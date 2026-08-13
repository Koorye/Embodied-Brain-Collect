"""Multimodal data acquisition + sync project for the embodied-cognition lab.

See `record/README.md` for usage and `record/ARCHITECTURE.md` for the design.

Sub-packages
------------
    record.sync          UDP/ZMQ marker bus + E-Prime contract (marker codes).
    record.recorders     One module per modality (eye, tactile, wrist_cam, emg, ...).
    record.session       Session-level launcher + offline aligner.
    record.tools         Stand-alone helpers (markers inspector, etc.).
"""

__version__ = "0.1.0"
