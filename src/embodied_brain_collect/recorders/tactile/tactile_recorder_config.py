"""Tactile glove config.

The TouchTronix implementation is a stub (open always fails) — no device
parameters yet.  Re-add port/baud/hand/bundle_path/license_path here when
the real SDK integration lands.
"""
from dataclasses import dataclass
from ..base import BaseRecorderConfig


@dataclass
class TactileRecorderConfig(BaseRecorderConfig):
    pass
