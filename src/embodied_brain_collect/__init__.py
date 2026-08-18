"""embodied_brain_collect — multi-modal data collection framework.

Sub-packages
------------
    embodied_brain_collect.recorders     One sub-package per modality
        (eye, camera, emg, hand_pose, position, marker, tactile) on top of
        the shared base + FFmpegWriter.
    embodied_brain_collect.session       Process-parallel launcher + recorder
        preset factories.
    embodied_brain_collect.stim          Stimulus programs (paradigm1) + marker
        sender.
    embodied_brain_collect.config        Collection config (task library).
    embodied_brain_collect.checkers      Data validation / audit tools.
    embodied_brain_collect.visualizers   Session visualization tools.

Run:
    python -m embodied_brain_collect.session.launcher --session-dir <dir>
"""

__version__ = "0.1.0"
