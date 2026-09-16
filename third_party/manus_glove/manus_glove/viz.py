import numpy as np
import open3d as o3d


class GloveViz:
    """Open3D visualization for a single glove."""

    def __init__(self, glove_id: int):
        self.glove_id = glove_id
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(window_name=f"Manus Glove {glove_id}", width=800, height=600)

        self.node_meshes: dict[int, o3d.geometry.TriangleMesh] = {}
        self.node_positions: dict[int, np.ndarray] = {}

        self.frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.line_set = o3d.geometry.LineSet()

        self.viz.add_geometry(self.frame_mesh)
        self.viz.add_geometry(self.line_set)

    def update_lines(self, glove_data: dict) -> None:
        """Update lines connecting child and parent nodes."""
        line_points = []
        line_indices = []

        for node in glove_data["raw_nodes"]:
            node_id = node["id"]
            parent_id = node.get("parentId")

            if parent_id is None:
                continue

            if parent_id in self.node_positions and node_id in self.node_positions:
                child_pos = self.node_positions[node_id]
                parent_pos = self.node_positions[parent_id]

                start_idx = len(line_points)
                line_points.append(parent_pos)
                line_points.append(child_pos)
                line_indices.append([start_idx, start_idx + 1])

        if line_points:
            self.line_set.points = o3d.utility.Vector3dVector(line_points)
            self.line_set.lines = o3d.utility.Vector2iVector(line_indices)
            self.line_set.paint_uniform_color([0, 0, 0])
            self.viz.update_geometry(self.line_set)
