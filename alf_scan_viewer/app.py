from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import sys

from .loaders import (
    GEOMETRY_EXTENSIONS,
    GRIDMAP_EXTENSIONS,
    load_gridmap_surface,
    require_supported_path,
    supported_file_filter,
)


def _import_open3d() -> tuple[Any, Any, Any]:
    try:
        import open3d as o3d  # type: ignore
        import open3d.visualization.gui as gui  # type: ignore
        import open3d.visualization.rendering as rendering  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Open3D is required. Run: powershell -ExecutionPolicy Bypass "
            "-File .\\scripts\\install_runtime.ps1"
        ) from exc
    return o3d, gui, rendering


class ViewerApp:
    def __init__(self, initial_path: Path | None = None) -> None:
        self.o3d, self.gui, self.rendering = _import_open3d()
        self.window = self.gui.Application.instance.create_window(
            "ALF Scan Viewer", 1200, 800
        )
        self.scene = self.gui.SceneWidget()
        self.scene.scene = self.rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.16, 0.17, 0.18, 1.0])
        self.scene.scene.show_axes(False)
        self.scene.set_view_controls(self.gui.SceneWidget.Controls.ROTATE_CAMERA)

        self.open_button = self.gui.Button("Open...")
        self.open_button.set_on_clicked(self._show_open_dialog)
        self.status = self.gui.Label("Open an ALF gridmap or Open3D geometry file.")

        self.material = self.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 4.0
        self.mesh_material = self.rendering.MaterialRecord()
        self.mesh_material.shader = "defaultUnlit"
        self._initial_path = initial_path
        self._initial_load_done = False

        self._build_layout()

    def _build_layout(self) -> None:
        em = self.window.theme.font_size
        self.toolbar = self.gui.Horiz(0.5 * em)
        self.toolbar.add_child(self.open_button)
        self.toolbar.add_child(self.status)

        self.window.add_child(self.toolbar)
        self.window.add_child(self.scene)

        def on_layout(_ctx: Any) -> None:
            content = self.window.content_rect
            toolbar_height = int(2.8 * em)
            self.toolbar.frame = self.gui.Rect(
                content.x, content.y, content.width, toolbar_height
            )
            self.scene.frame = self.gui.Rect(
                content.x,
                content.y + toolbar_height,
                content.width,
                content.height - toolbar_height,
            )
            if self._initial_path and not self._initial_load_done:
                self._initial_load_done = True
                self.load_file(self._initial_path)

        self.window.set_on_layout(on_layout)

    def _show_open_dialog(self) -> None:
        dialog = self.gui.FileDialog(
            self.gui.FileDialog.OPEN,
            "Open scan file",
            self.window.theme,
        )
        dialog.add_filter(supported_file_filter(), "Supported scan files")
        dialog.add_filter(".npz", "ALF gridmap (.npz)")
        dialog.add_filter(".ply", "PLY geometry (.ply)")
        dialog.add_filter(
            " ".join(sorted(GEOMETRY_EXTENSIONS - {".ply"})),
            "Point cloud files",
        )
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        dialog.set_on_done(self._open_dialog_done)
        self.window.show_dialog(dialog)

    def _open_dialog_done(self, filename: str) -> None:
        self.window.close_dialog()
        self.load_file(Path(filename).expanduser().resolve())

    def load_file(self, path: Path) -> None:
        try:
            path = require_supported_path(path)
            if path.suffix.lower() in GRIDMAP_EXTENSIONS:
                geometry = self._load_gridmap(path)
                material = self.material
                label = f"{path.name} loaded ({len(geometry.points)} points)"
            else:
                geometry = self._load_open3d_geometry(path)
                material = self.mesh_material if self._is_mesh(geometry) else self.material
                label = self._geometry_label(path, geometry)

            self.scene.scene.clear_geometry()
            self.scene.scene.add_geometry("scan", geometry, material)
            self._frame_geometry(geometry)
            self.scene.force_redraw()
            self.status.text = label
        except Exception as exc:
            self.status.text = f"Could not open {Path(path).name}: {exc}"
            self._show_error("Could not open file", str(exc))

    def _load_gridmap(self, path: Path) -> Any:
        surface = load_gridmap_surface(path)
        point_cloud = self.o3d.geometry.PointCloud()
        point_cloud.points = self.o3d.utility.Vector3dVector(surface.vertices)
        point_cloud.colors = self.o3d.utility.Vector3dVector(surface.colors)
        return point_cloud

    def _load_open3d_geometry(self, path: Path) -> Any:
        suffix = path.suffix.lower()
        if suffix == ".ply":
            mesh = self.o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
            if mesh is not None and len(mesh.triangles) > 0:
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([0.78, 0.78, 0.78])
                mesh.compute_vertex_normals()
                return mesh

        point_cloud = self.o3d.io.read_point_cloud(str(path))
        if point_cloud is not None and len(point_cloud.points) > 0:
            if len(point_cloud.colors) == 0:
                point_cloud.paint_uniform_color([0.78, 0.78, 0.78])
            return point_cloud

        if suffix in GEOMETRY_EXTENSIONS:
            mesh = self.o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
            if mesh is not None and len(mesh.vertices) > 0:
                return self._mesh_vertices_as_point_cloud(mesh)

        raise ValueError(f"No supported geometry could be read from: {path}")

    def _mesh_vertices_as_point_cloud(self, mesh: Any) -> Any:
        point_cloud = self.o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        if len(mesh.vertex_colors) == len(mesh.vertices):
            point_cloud.colors = mesh.vertex_colors
        else:
            point_cloud.paint_uniform_color([0.78, 0.78, 0.78])
        return point_cloud

    @staticmethod
    def _is_mesh(geometry: Any) -> bool:
        return hasattr(geometry, "triangles") and len(geometry.triangles) > 0

    def _geometry_label(self, path: Path, geometry: Any) -> str:
        if self._is_mesh(geometry):
            return (
                f"{path.name} loaded "
                f"({len(geometry.vertices)} vertices, {len(geometry.triangles)} triangles)"
            )
        return f"{path.name} loaded ({len(geometry.points)} points)"

    def _frame_geometry(self, geometry: Any) -> None:
        bounds = geometry.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        max_extent = max(float(extent[0]), float(extent[1]), float(extent[2]), 1.0)
        distance = max_extent * 2.5
        eye = [
            float(center[0]) + distance * 0.65,
            float(center[1]) - distance,
            float(center[2]) + distance * 0.65,
        ]
        target = [float(center[0]), float(center[1]), float(center[2])]

        self.scene.setup_camera(60.0, bounds, center)
        self.scene.look_at(target, eye, [0.0, 0.0, 1.0])

    def _show_error(self, title: str, message: str) -> None:
        dialog = self.gui.Dialog(title)
        em = self.window.theme.font_size
        layout = self.gui.Vert(0.8 * em, self.gui.Margins(em, em, em, em))
        layout.add_child(self.gui.Label(message))
        ok = self.gui.Button("OK")
        ok.set_on_clicked(lambda: self.window.close_dialog())
        layout.add_child(ok)
        dialog.add_child(layout)
        self.window.show_dialog(dialog)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an ALF scan file.")
    parser.add_argument("path", nargs="?", help="Optional scan file to open.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _o3d, gui, _rendering = _import_open3d()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    gui.Application.instance.initialize()
    initial_path = Path(args.path).expanduser().resolve() if args.path else None
    ViewerApp(initial_path)
    gui.Application.instance.run()
    return 0
