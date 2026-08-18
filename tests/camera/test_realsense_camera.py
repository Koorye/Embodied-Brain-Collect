"""Test Intel RealSense depth camera — real hardware."""

from matplotlib.gridspec import GridSpec
from tests.camera.test_dummy_camera import TestDummyCamera
from tests.base import SESSION_DIR
from src.recorders.camera import RealsenseCameraRecorder, RealsenseCameraConfig


class TestRealsenseCamera(TestDummyCamera):
    name = "RealSense Camera"

    def _build_layout(self, fig):
        gs = GridSpec(1, 2, figure=fig)
        self.ax = fig.add_subplot(gs[0, 0])        # color (RGB)
        self.ax_depth = fig.add_subplot(gs[0, 1])  # aligned depth (mm)

    def _update(self, rec, elapsed):
        frames = rec._arr_buf.get("frames", [])
        if frames:
            self.ax.clear()
            self.ax.imshow(frames[-1])
            self.ax.set_title(f"color ({len(frames)} frames) [Q to stop]")
            self.ax.axis("off")
        depths = rec._arr_buf.get("depth_frames", [])
        if depths:
            self.ax_depth.clear()
            self.ax_depth.imshow(depths[-1], cmap="viridis")
            self.ax_depth.set_title(f"depth ({len(depths)}) mm")
            self.ax_depth.axis("off")


def main():
    cfg = RealsenseCameraConfig(session_dir=f"{SESSION_DIR}/camera")
    rec = RealsenseCameraRecorder(cfg)
    TestRealsenseCamera(rec).run()


if __name__ == "__main__":
    main()
