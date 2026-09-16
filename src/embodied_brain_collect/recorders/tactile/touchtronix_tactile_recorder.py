"""TouchTronix tactile glove — stub."""
from .base_tactile_recorder import BaseTactileRecorder
from .tactile_recorder_config import TactileRecorderConfig


class TouchtronixTactileRecorder(BaseTactileRecorder):
    config: TactileRecorderConfig

    def _open(self) -> bool:
        self._open_error = "TouchTronix recorder not implemented (stub)"
        self._log(f"[tactile:touchtronix] open failed — {self._open_error}")
        return False

    def _close(self) -> None:
        pass
