"""EGO headband recorders — network device with 4 cameras + 2 IMUs.

Usage::

    from embodied_brain_collect.recorders.ego_headband import (
        EgoHeadbandRecorderConfig, DummyEgoHeadbandRecorder, NetEgoHeadbandRecorder,
    )

    cfg = EgoHeadbandRecorderConfig(host="0.0.0.0", port=5555, duration=10)
    rec = NetEgoHeadbandRecorder(cfg)
    rec.run()
"""

from .ego_headband_recorder_config import EgoHeadbandRecorderConfig
from .base_ego_headband_recorder import BaseEgoHeadbandRecorder
from .dummy_ego_headband_recorder import DummyEgoHeadbandRecorder
from .net_ego_headband_recorder import NetEgoHeadbandRecorder

__all__ = [
    "EgoHeadbandRecorderConfig",
    "BaseEgoHeadbandRecorder",
    "DummyEgoHeadbandRecorder",
    "NetEgoHeadbandRecorder",
]
