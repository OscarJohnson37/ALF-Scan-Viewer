from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import re
import sys

from .loaders import (
    GEOMETRY_EXTENSIONS,
    GRIDMAP_EXTENSIONS,
    load_gridmap_surface,
    require_supported_path,
)
from .project import (
    MilestoneScan,
    SURFACE_VIEW_GRAPH_TYPE,
    default_baseline_scan,
    discover_milestone_scans,
    resolve_project_dir,
    resolve_surface_scanner_dir,
    surface_scanner_graph_dir,
)


MAX_STACK_VERTICES = 1_500_000
MAX_STACK_TRIANGLES = 5_000_000


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

        self.open_button = self.gui.Button("Open Project...")
        self.open_button.set_on_clicked(self._show_open_project_dialog)
        self.milestone_combo = self.gui.Combobox()
        self.milestone_combo.enabled = False
        self.milestone_combo.set_on_selection_changed(self._milestone_selected)
        self.status = self.gui.Label("Open an ALF project or Surface Scanner folder.")
        self.details = self.gui.Label("")

        self.view_mode_combo = self.gui.Combobox()
        self.view_mode_combo.add_item("Selected Milestone")
        self.view_mode_combo.add_item("Stack Milestones")
        self.view_mode_combo.selected_index = 0
        self.view_mode_combo.set_on_selection_changed(self._view_mode_changed)

        self.downsample_slider = self.gui.Slider(self.gui.Slider.INT)
        self.downsample_slider.set_limits(1, 25)
        self.downsample_slider.int_value = 1
        self.downsample_slider.set_on_value_changed(self._downsample_slider_changed)

        self.downsample_input = self.gui.NumberEdit(self.gui.NumberEdit.INT)
        self.downsample_input.set_limits(1, 100)
        self.downsample_input.set_value(1)
        self.downsample_input.set_preferred_width(80)
        self.downsample_input.set_on_value_changed(self._downsample_input_changed)

        self.color_combo = self.gui.Combobox()
        self.color_combo.add_item("RGB")
        self.color_combo.add_item("Height")
        self.color_combo.add_item("Deformation From Baseline")
        self.color_combo.add_item("Absolute Deformation From Baseline")
        self.color_combo.selected_index = 0
        self.color_combo.set_on_selection_changed(self._surface_option_changed)

        self.color_binary_checkbox = self.gui.Checkbox("Binary red/green")
        self.color_binary_checkbox.set_on_checked(self._color_binary_changed)

        self.color_threshold_slider = self.gui.Slider(self.gui.Slider.DOUBLE)
        self.color_threshold_slider.set_limits(-20.0, 100.0)
        self.color_threshold_slider.double_value = 0.0
        self.color_threshold_slider.set_on_value_changed(
            self._color_threshold_slider_changed
        )

        self.color_threshold_input = self.gui.NumberEdit(self.gui.NumberEdit.DOUBLE)
        self.color_threshold_input.set_limits(-20.0, 100.0)
        self.color_threshold_input.set_value(0.0)
        self.color_threshold_input.set_preferred_width(90)
        self.color_threshold_input.set_on_value_changed(
            self._color_threshold_input_changed
        )

        self.color_invert_checkbox = self.gui.Checkbox("Invert colours")
        self.color_invert_checkbox.set_on_checked(self._color_binary_changed)

        self.z_combo = self.gui.Combobox()
        self.z_combo.add_item("Mean Height")
        self.z_combo.add_item("Deformation From Baseline")
        self.z_combo.add_item("Absolute Deformation From Baseline")
        self.z_combo.add_item("Flat")
        self.z_combo.selected_index = 0
        self.z_combo.set_on_selection_changed(self._surface_option_changed)

        self.baseline_combo = self.gui.Combobox()
        self.baseline_combo.enabled = False
        self.baseline_combo.set_on_selection_changed(self._baseline_selected)

        self.z_scale_slider = self.gui.Slider(self.gui.Slider.DOUBLE)
        self.z_scale_slider.set_limits(0.0, 50.0)
        self.z_scale_slider.double_value = 1.0
        self.z_scale_slider.set_on_value_changed(self._z_scale_slider_changed)

        self.z_scale_input = self.gui.NumberEdit(self.gui.NumberEdit.DOUBLE)
        self.z_scale_input.set_limits(0.0, 1000.0)
        self.z_scale_input.set_value(1.0)
        self.z_scale_input.set_preferred_width(90)
        self.z_scale_input.set_on_value_changed(self._z_scale_input_changed)

        self.limit_x_checkbox = self.gui.Checkbox("Limit X")
        self.limit_x_checkbox.set_on_checked(self._section_option_changed)
        self.x_min_input = self._make_range_input(0.0)
        self.x_max_input = self._make_range_input(1.0)

        self.limit_y_checkbox = self.gui.Checkbox("Limit Y")
        self.limit_y_checkbox.set_on_checked(self._section_option_changed)
        self.y_min_input = self._make_range_input(0.0)
        self.y_max_input = self._make_range_input(1.0)

        self.stack_separation_slider = self.gui.Slider(self.gui.Slider.DOUBLE)
        self.stack_separation_slider.set_limits(0.0, 3.0)
        self.stack_separation_slider.double_value = 0.5
        self.stack_separation_slider.set_on_value_changed(
            self._stack_separation_slider_changed
        )

        self.stack_separation_input = self.gui.NumberEdit(self.gui.NumberEdit.DOUBLE)
        self.stack_separation_input.set_limits(0.0, 3.0)
        self.stack_separation_input.set_value(0.5)
        self.stack_separation_input.set_preferred_width(90)
        self.stack_separation_input.set_on_value_changed(
            self._stack_separation_input_changed
        )

        self.stack_selection_summary = self.gui.Label("No stack milestones loaded.")
        self.stack_select_all_button = self.gui.Button("Select all")
        self.stack_select_all_button.set_on_clicked(self._select_all_stack_milestones)
        self.stack_clear_button = self.gui.Button("Clear")
        self.stack_clear_button.set_on_clicked(self._clear_stack_milestone_selection)

        self.camera_combo = self.gui.Combobox()
        for preset in (
            "Current / Mouse",
            "Isometric",
            "Top",
            "Front",
            "Rear",
            "Left",
            "Right",
        ):
            self.camera_combo.add_item(preset)
        self.camera_combo.selected_index = 0
        self.camera_combo.set_on_selection_changed(self._camera_preset_selected)
        self.camera_apply_button = self.gui.Button("Apply view")
        self.camera_apply_button.set_on_clicked(self._apply_selected_camera_preset)
        self.camera_export_button = self.gui.Button("Export current view")
        self.camera_export_button.set_on_clicked(self._export_current_view)

        self.generate_button = self.gui.Button("Refresh View")
        self.generate_button.enabled = False
        self.generate_button.set_on_clicked(self._render_requested_view)

        self.material = self.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 4.0
        self.mesh_material = self.rendering.MaterialRecord()
        self.mesh_material.shader = "defaultUnlit"
        self._initial_path = initial_path
        self._initial_load_done = False
        self._milestone_scans: list[MilestoneScan] = []
        self._current_scan: MilestoneScan | None = None
        self._loaded_project_dir: Path | None = None
        self._loaded_surface_scanner_dir: Path | None = None
        self._syncing_downsample = False
        self._syncing_color_threshold = False
        self._syncing_z_scale = False
        self._syncing_stack_separation = False
        self._syncing_baseline = False
        self._syncing_stack_selection = False
        self._stack_scan_checkboxes: list[tuple[MilestoneScan, Any]] = []
        self._stack_surface_cache: list[tuple[MilestoneScan, Any]] = []
        self._stack_cache_signature: tuple[Any, ...] | None = None
        self._current_scene_bounds: Any | None = None
        self._scene_has_geometry = False

        self._build_layout()
        self._update_view_controls()

    def _build_layout(self) -> None:
        em = self.window.theme.font_size
        self.toolbar = self.gui.Horiz(0.5 * em)
        self.toolbar.add_child(self.open_button)
        self.toolbar.add_child(self.status)

        self.side_panel = self.gui.ScrollableVert(
            0.35 * em,
            self.gui.Margins(0.8 * em, 0.8 * em, 0.8 * em, 0.8 * em),
        )

        project_section = self._make_section("Project", em, is_open=True)
        self._add_labeled_widget(project_section, "View", self.view_mode_combo, em)
        self._add_labeled_widget(project_section, "Milestone", self.milestone_combo, em)
        project_section.add_child(self.generate_button)
        self.side_panel.add_child(project_section)

        surface_section = self._make_section("Surface", em, is_open=True)
        self._add_labeled_widget(surface_section, "Colour Source", self.color_combo, em)
        surface_section.add_child(self.color_binary_checkbox)
        surface_section.add_child(self.gui.Label("Threshold (mm)"))
        threshold_row = self.gui.Horiz(0.4 * em)
        threshold_row.add_child(self.color_threshold_slider)
        threshold_row.add_child(self.color_threshold_input)
        surface_section.add_child(threshold_row)
        surface_section.add_child(self.color_invert_checkbox)
        surface_section.add_fixed(0.45 * em)
        surface_section.add_child(self.gui.Label("Downsample Step"))
        downsample_row = self.gui.Horiz(0.4 * em)
        downsample_row.add_child(self.downsample_slider)
        downsample_row.add_child(self.downsample_input)
        surface_section.add_child(downsample_row)
        self.side_panel.add_child(surface_section)

        z_section = self._make_section("Z Axis", em, is_open=True)
        self._add_labeled_widget(z_section, "Mode", self.z_combo, em)
        self._add_labeled_widget(z_section, "Baseline", self.baseline_combo, em)
        z_section.add_child(self.gui.Label("Amplification"))
        z_scale_row = self.gui.Horiz(0.4 * em)
        z_scale_row.add_child(self.z_scale_slider)
        z_scale_row.add_child(self.z_scale_input)
        z_section.add_child(z_scale_row)
        self.side_panel.add_child(z_section)

        section = self._make_section("Section", em, is_open=False)
        section.add_child(self.limit_x_checkbox)
        x_row = self.gui.Horiz(0.4 * em)
        x_row.add_child(self.x_min_input)
        x_row.add_child(self.x_max_input)
        section.add_child(x_row)
        section.add_fixed(0.45 * em)
        section.add_child(self.limit_y_checkbox)
        y_row = self.gui.Horiz(0.4 * em)
        y_row.add_child(self.y_min_input)
        y_row.add_child(self.y_max_input)
        section.add_child(y_row)
        self.side_panel.add_child(section)

        self.stack_section = self._make_section("Stack", em, is_open=False)
        self.stack_section.add_child(self.gui.Label("Separation"))
        separation_row = self.gui.Horiz(0.4 * em)
        separation_row.add_child(self.stack_separation_slider)
        separation_row.add_child(self.stack_separation_input)
        self.stack_section.add_child(separation_row)
        self.stack_section.add_fixed(0.45 * em)
        self.stack_section.add_child(self.gui.Label("Milestones"))
        self.stack_section.add_child(self.stack_selection_summary)
        stack_button_row = self.gui.Horiz(0.4 * em)
        stack_button_row.add_child(self.stack_select_all_button)
        stack_button_row.add_child(self.stack_clear_button)
        self.stack_section.add_child(stack_button_row)
        self.stack_selection_list = self.gui.Vert(0.25 * em)
        self.stack_section.add_child(self.stack_selection_list)
        self.side_panel.add_child(self.stack_section)

        camera_section = self._make_section("Camera", em, is_open=False)
        self._add_labeled_widget(camera_section, "Preset", self.camera_combo, em)
        camera_button_row = self.gui.Horiz(0.4 * em)
        camera_button_row.add_child(self.camera_apply_button)
        camera_button_row.add_child(self.camera_export_button)
        camera_section.add_child(camera_button_row)
        camera_section.add_child(
            self.gui.Label("Mouse rotation is always exported as the current view.")
        )
        self.side_panel.add_child(camera_section)

        self.info_section = self._make_section("Info", em, is_open=False)
        self.info_section.add_child(self.details)
        self.side_panel.add_child(self.info_section)
        self.side_panel.add_stretch()

        self.window.add_child(self.toolbar)
        self.window.add_child(self.side_panel)
        self.window.add_child(self.scene)

        def on_layout(_ctx: Any) -> None:
            content = self.window.content_rect
            toolbar_height = int(2.8 * em)
            side_width = int(max(340, 24 * em))
            self.toolbar.frame = self.gui.Rect(
                content.x, content.y, content.width, toolbar_height
            )
            self.side_panel.frame = self.gui.Rect(
                content.get_right() - side_width,
                content.y + toolbar_height,
                side_width,
                content.height - toolbar_height,
            )
            self.scene.frame = self.gui.Rect(
                content.x,
                content.y + toolbar_height,
                content.width - side_width,
                content.height - toolbar_height,
            )
            if self._initial_path and not self._initial_load_done:
                self._initial_load_done = True
                self.load_path(self._initial_path)

        self.window.set_on_layout(on_layout)

    def _make_section(self, title: str, em: int, *, is_open: bool) -> Any:
        section = self.gui.CollapsableVert(
            title,
            0.35 * em,
            self.gui.Margins(0.35 * em, 0.35 * em, 0.35 * em, 0.45 * em),
        )
        section.set_is_open(is_open)
        return section

    def _add_labeled_widget(self, parent: Any, label: str, widget: Any, em: int) -> None:
        parent.add_child(self.gui.Label(label))
        parent.add_child(widget)
        parent.add_fixed(0.45 * em)

    def _make_range_input(self, value: float) -> Any:
        input_box = self.gui.NumberEdit(self.gui.NumberEdit.DOUBLE)
        input_box.set_limits(-100000.0, 100000.0)
        input_box.set_value(float(value))
        input_box.set_preferred_width(110)
        input_box.set_on_value_changed(self._section_value_changed)
        return input_box

    def _show_open_project_dialog(self) -> None:
        dialog = self.gui.FileDialog(
            self.gui.FileDialog.OPEN_DIR,
            "Open ALF project or Surface Scanner folder",
            self.window.theme,
        )
        dialog.set_on_cancel(lambda: self.window.close_dialog())
        dialog.set_on_done(self._open_project_dialog_done)
        self.window.show_dialog(dialog)

    def _open_project_dialog_done(self, directory: str) -> None:
        self.window.close_dialog()
        self.load_project(Path(directory).expanduser().resolve())

    def _milestone_selected(self, _text: str, index: int) -> None:
        if 0 <= index < len(self._milestone_scans):
            self.load_milestone(self._milestone_scans[index])

    def _view_mode_changed(self, _text: str, _index: int) -> None:
        self._update_view_controls()
        if self._milestone_scans:
            self._render_requested_view()

    def _surface_option_changed(self, _text: str, _index: int) -> None:
        self._update_view_controls()
        self._clear_stack_cache()
        self._auto_render_view()

    def _surface_value_changed(self, _value: float) -> None:
        self._clear_stack_cache()
        self._auto_render_view()

    def _color_binary_changed(self, _checked: bool) -> None:
        self._update_view_controls()
        self._clear_stack_cache()
        self._auto_render_view()

    def _color_threshold_slider_changed(self, value: float) -> None:
        if self._syncing_color_threshold:
            return
        self._syncing_color_threshold = True
        try:
            self.color_threshold_input.set_value(float(value))
        finally:
            self._syncing_color_threshold = False
        self._clear_stack_cache()
        self._auto_render_view()

    def _color_threshold_input_changed(self, value: float) -> None:
        if self._syncing_color_threshold:
            return
        value = float(value)
        self._syncing_color_threshold = True
        try:
            self.color_threshold_slider.double_value = min(max(value, -20.0), 100.0)
        finally:
            self._syncing_color_threshold = False
        self._clear_stack_cache()
        self._auto_render_view()

    def _z_scale_slider_changed(self, value: float) -> None:
        if self._syncing_z_scale:
            return
        value = max(0.0, float(value))
        self._syncing_z_scale = True
        try:
            self.z_scale_input.set_value(value)
        finally:
            self._syncing_z_scale = False
        self._clear_stack_cache()
        self._auto_render_view()

    def _z_scale_input_changed(self, value: float) -> None:
        if self._syncing_z_scale:
            return
        value = max(0.0, float(value))
        self._syncing_z_scale = True
        try:
            self.z_scale_slider.double_value = min(value, 50.0)
        finally:
            self._syncing_z_scale = False
        self._clear_stack_cache()
        self._auto_render_view()

    def _baseline_selected(self, _text: str, _index: int) -> None:
        if self._syncing_baseline:
            return
        self._clear_stack_cache()
        self._auto_render_view()

    def _section_option_changed(self, _checked: bool) -> None:
        self._update_view_controls()
        self._clear_stack_cache()
        self._auto_render_view()

    def _section_value_changed(self, _value: float) -> None:
        self._clear_stack_cache()
        self._auto_render_view()

    def _downsample_slider_changed(self, value: int) -> None:
        if self._syncing_downsample:
            return
        self._clear_stack_cache()
        self._syncing_downsample = True
        try:
            self.downsample_input.set_value(max(1, int(value)))
        finally:
            self._syncing_downsample = False
        self._auto_render_view()

    def _downsample_input_changed(self, value: int) -> None:
        if self._syncing_downsample:
            return
        self._clear_stack_cache()
        value = max(1, int(value))
        self._syncing_downsample = True
        try:
            self.downsample_slider.int_value = min(value, 25)
        finally:
            self._syncing_downsample = False
        self._auto_render_view()

    def _stack_separation_slider_changed(self, value: float) -> None:
        if self._syncing_stack_separation:
            return
        self._syncing_stack_separation = True
        try:
            self.stack_separation_input.set_value(max(0.0, float(value)))
        finally:
            self._syncing_stack_separation = False
        self._rerender_cached_stack()

    def _stack_separation_input_changed(self, value: float) -> None:
        if self._syncing_stack_separation:
            return
        value = max(0.0, float(value))
        self._syncing_stack_separation = True
        try:
            self.stack_separation_slider.double_value = min(value, 3.0)
        finally:
            self._syncing_stack_separation = False
        self._rerender_cached_stack()

    def _stack_milestone_selection_changed(self, _checked: bool) -> None:
        if self._syncing_stack_selection:
            return
        self._clear_stack_cache()
        self._update_view_controls()
        if self._is_stack_mode():
            self._generate_stack_surface()

    def _select_all_stack_milestones(self) -> None:
        self._set_all_stack_milestones_selected(True)

    def _clear_stack_milestone_selection(self) -> None:
        self._set_all_stack_milestones_selected(False)

    def _set_all_stack_milestones_selected(self, selected: bool) -> None:
        self._syncing_stack_selection = True
        try:
            for _scan, checkbox in self._stack_scan_checkboxes:
                checkbox.checked = bool(selected)
        finally:
            self._syncing_stack_selection = False
        self._clear_stack_cache()
        self._update_view_controls()
        if self._is_stack_mode():
            self._generate_stack_surface()

    def _camera_preset_selected(self, text: str, _index: int) -> None:
        if self._camera_preset_key(text) != "current":
            self._apply_selected_camera_preset()

    def _apply_selected_camera_preset(self) -> None:
        preset = self._camera_preset_key()
        if preset == "current":
            self.status.text = "Rotate with the mouse to choose the current view."
            return
        self._apply_camera_preset(preset)

    def load_path(self, path: Path) -> None:
        path = Path(path)
        if path.is_dir():
            self.load_project(path)
        else:
            self.load_file(path)

    def load_project(self, path: Path) -> None:
        try:
            self._set_loaded_project_context(path)
            scans = discover_milestone_scans(path)
            if not scans:
                raise FileNotFoundError(
                    "No Surface Scanner milestones with a processed surface or raw gridmap were found."
                )
            self._milestone_scans = scans
            self._clear_stack_cache()
            self.milestone_combo.clear_items()
            for scan in scans:
                self.milestone_combo.add_item(f"{scan.label} ({scan.source_kind})")
            self._populate_baseline_combo(scans)
            self._populate_stack_milestone_controls(scans)
            self.milestone_combo.enabled = True
            self._update_view_controls()
            self.milestone_combo.selected_index = 0
            self.load_milestone(scans[0])
        except Exception as exc:
            self._milestone_scans = []
            self._clear_stack_cache()
            self._loaded_project_dir = None
            self._loaded_surface_scanner_dir = None
            self._scene_has_geometry = False
            self._current_scene_bounds = None
            self.milestone_combo.clear_items()
            self.baseline_combo.clear_items()
            self._populate_stack_milestone_controls([])
            self.milestone_combo.enabled = False
            self._update_view_controls()
            self.status.text = f"Could not open project: {exc}"
            self._show_error("Could not open project", str(exc))

    def _set_loaded_project_context(self, path: Path) -> None:
        start = Path(path)
        search_root = start if start.is_dir() else start.parent
        for candidate in (search_root, *search_root.parents):
            try:
                self._loaded_surface_scanner_dir = resolve_surface_scanner_dir(candidate)
                self._loaded_project_dir = resolve_project_dir(candidate)
                return
            except FileNotFoundError:
                continue
        self._loaded_project_dir = None
        self._loaded_surface_scanner_dir = None

    def _populate_baseline_combo(self, scans: list[MilestoneScan]) -> None:
        baseline_scans = [scan for scan in scans if scan.raw_gridmap_path is not None]
        default_scan = default_baseline_scan(scans)

        self._syncing_baseline = True
        try:
            self.baseline_combo.clear_items()
            for scan in baseline_scans:
                label = scan.label
                if scan.post_bedding_baseline:
                    label = f"{label} (default)"
                self.baseline_combo.add_item(label)
            self.baseline_combo.enabled = bool(baseline_scans)
            if baseline_scans:
                default_index = 0
                if default_scan is not None:
                    for index, scan in enumerate(baseline_scans):
                        if scan.identity == default_scan.identity:
                            default_index = index
                            break
                self.baseline_combo.selected_index = default_index
        finally:
            self._syncing_baseline = False

    def _populate_stack_milestone_controls(self, scans: list[MilestoneScan]) -> None:
        self._syncing_stack_selection = True
        try:
            for _scan, checkbox in self._stack_scan_checkboxes:
                checkbox.visible = False
                checkbox.enabled = False
            self._stack_scan_checkboxes = []

            stackable_scans = [
                scan for scan in scans if scan.raw_gridmap_path is not None
            ]
            for scan in stackable_scans:
                checkbox = self.gui.Checkbox(scan.label)
                checkbox.checked = True
                checkbox.set_on_checked(self._stack_milestone_selection_changed)
                self.stack_selection_list.add_child(checkbox)
                self._stack_scan_checkboxes.append((scan, checkbox))
        finally:
            self._syncing_stack_selection = False
        self._update_stack_selection_summary()

    def load_milestone(self, scan: MilestoneScan) -> None:
        self._current_scan = scan
        self._update_view_controls()
        if self._is_stack_mode():
            self._generate_stack_surface()
            return

        if scan.raw_gridmap_path is not None:
            self._generate_current_surface()
            return

        if scan.processed_surface_path is not None:
            self.load_file(
                scan.processed_surface_path,
                label_prefix=f"{scan.label} | processed surface",
            )
            self.details.text = self._scan_details(scan)
            return

        self.status.text = f"No viewable scan found for {scan.label}"

    def _render_requested_view(self) -> None:
        if self._is_stack_mode():
            self._generate_stack_surface()
        elif self._current_scan and self._current_scan.raw_gridmap_path is not None:
            self._generate_current_surface()
        elif self._current_scan and self._current_scan.processed_surface_path is not None:
            self.load_file(
                self._current_scan.processed_surface_path,
                label_prefix=f"{self._current_scan.label} | processed surface",
            )

    def _auto_render_view(self) -> None:
        if not self._milestone_scans and self._current_scan is None:
            return
        if self._is_stack_mode():
            if self._selected_stack_scans():
                self._generate_stack_surface()
            return
        if self._current_scan and self._current_scan.raw_gridmap_path is not None:
            self._generate_current_surface()

    def _generate_current_surface(self) -> None:
        scan = self._current_scan
        if scan is None or scan.raw_gridmap_path is None:
            self.status.text = "No raw gridmap is available for this milestone."
            return

        try:
            surface = load_gridmap_surface(
                scan.raw_gridmap_path,
                downsample_step=self._downsample_step(),
                color_by=self._color_by(),
                color_binary=self._color_binary(),
                color_threshold=self._color_threshold(),
                color_invert=self._color_invert(),
                z_by=self._z_by(),
                z_scale=self._z_scale(),
                baseline_path=self._baseline_gridmap_path(scan),
                x_range=self._x_range(),
                y_range=self._y_range(),
            )
            geometry = self._gridmap_surface_to_mesh(surface)
            material = self.mesh_material if self._is_mesh(geometry) else self.material
            self._render_geometry(geometry, material)
            self.status.text = (
                f"{scan.label} | generated from raw gridmap | "
                f"downsample {surface.downsample_step}, "
                f"colour {self._display_mode_label(surface.color_by)}, "
                f"z {self._display_mode_label(surface.z_by)} x{surface.z_scale:g}"
            )
            self.details.text = self._scan_details(scan)
        except Exception as exc:
            self.status.text = f"Could not generate surface: {exc}"
            self._show_error("Could not generate surface", str(exc))

    def _generate_stack_surface(self) -> None:
        scans = self._selected_stack_scans()
        if not scans:
            self.scene.scene.clear_geometry()
            self.scene.force_redraw()
            self._scene_has_geometry = False
            self._current_scene_bounds = None
            self.details.text = self._stack_details()
            self.status.text = "No milestones selected for stacking."
            self._update_view_controls()
            return

        cache_signature = self._stack_surface_cache_signature(scans)
        if self._stack_cache_signature == cache_signature and self._stack_surface_cache:
            self._render_stack_surfaces()
            return

        baseline_path = self._baseline_gridmap_path()
        surfaces: list[tuple[MilestoneScan, Any]] = []
        failures: list[str] = []

        for scan in scans:
            if scan.raw_gridmap_path is None:
                continue
            try:
                surface = load_gridmap_surface(
                    scan.raw_gridmap_path,
                    downsample_step=self._downsample_step(),
                    color_by=self._color_by(),
                    color_binary=self._color_binary(),
                    color_threshold=self._color_threshold(),
                    color_invert=self._color_invert(),
                    z_by=self._z_by(),
                    z_scale=self._z_scale(),
                    baseline_path=baseline_path,
                    x_range=self._x_range(),
                    y_range=self._y_range(),
                )
                surfaces.append((scan, surface))
            except Exception as exc:
                failures.append(f"{scan.label}: {exc}")

        if not surfaces:
            message = "Could not render any stack surfaces."
            if failures:
                message = f"{message}\n" + "\n".join(failures[:5])
            self.status.text = "Could not render stack."
            self._show_error("Could not render stack", message)
            return

        self._stack_surface_cache = surfaces
        self._stack_cache_signature = cache_signature
        self._render_stack_surfaces(failures=failures)

    def _render_stack_surfaces(self, *, failures: list[str] | None = None) -> None:
        if not self._stack_surface_cache:
            return

        total_vertices = sum(len(surface.vertices) for _scan, surface in self._stack_surface_cache)
        total_triangles = sum(len(surface.triangles) for _scan, surface in self._stack_surface_cache)
        if total_vertices > MAX_STACK_VERTICES or total_triangles > MAX_STACK_TRIANGLES:
            self.status.text = "Stack is too large to render safely."
            self._show_error(
                "Stack too large",
                "This stack is too large to render safely at the current "
                "downsample step.\n\n"
                f"Vertices: {total_vertices:,}\n"
                f"Triangles: {total_triangles:,}\n\n"
                "Increase Downsample Step and render again.",
            )
            return

        self.scene.scene.clear_geometry()
        geometries: list[Any] = []
        separation = self._stack_separation()

        for index, (_scan, surface) in enumerate(self._stack_surface_cache):
            geometry = self._gridmap_surface_to_mesh(
                surface,
                z_offset=float(index) * separation,
            )
            material = self.mesh_material if self._is_mesh(geometry) else self.material
            self.scene.scene.add_geometry(f"stack_{index}", geometry, material)
            geometries.append(geometry)

        if geometries:
            self._scene_has_geometry = True
            self._frame_geometries(geometries)
            self.scene.force_redraw()
            self._update_view_controls()

        count = len(self._stack_surface_cache)
        first_surface = self._stack_surface_cache[0][1]
        self.status.text = (
            f"Stacked {count} milestones | separation {separation:g} | "
            f"z {self._display_mode_label(first_surface.z_by)} "
            f"x{first_surface.z_scale:g}"
        )
        self.details.text = self._stack_details(failures=failures)

    def _rerender_cached_stack(self) -> None:
        if not self._is_stack_mode():
            return
        if self._stack_surface_cache:
            self._render_stack_surfaces()
        elif self._selected_stack_scans():
            self._generate_stack_surface()

    def load_file(self, path: Path, *, label_prefix: str | None = None) -> None:
        try:
            self._set_loaded_project_context(path)
            path = require_supported_path(path)
            if path.suffix.lower() in GRIDMAP_EXTENSIONS:
                geometry = self._load_gridmap(path)
                material = self.mesh_material if self._is_mesh(geometry) else self.material
                label = self._geometry_label(path, geometry)
            else:
                geometry = self._load_open3d_geometry(path)
                material = self.mesh_material if self._is_mesh(geometry) else self.material
                label = self._geometry_label(path, geometry)
            if label_prefix:
                label = f"{label_prefix} | {label}"

            self._render_geometry(geometry, material)
            self.status.text = label
        except Exception as exc:
            self.status.text = f"Could not open {Path(path).name}: {exc}"
            self._show_error("Could not open file", str(exc))

    def _load_gridmap(self, path: Path) -> Any:
        surface = load_gridmap_surface(
            path,
            downsample_step=self._downsample_step(),
            color_by=self._color_by(),
            color_binary=self._color_binary(),
            color_threshold=self._color_threshold(),
            color_invert=self._color_invert(),
            z_by=self._z_by(),
            z_scale=self._z_scale(),
            baseline_path=path if self._uses_baseline() else None,
            x_range=self._x_range(),
            y_range=self._y_range(),
        )
        return self._gridmap_surface_to_mesh(surface)

    def _gridmap_surface_to_mesh(self, surface: Any, *, z_offset: float = 0.0) -> Any:
        vertices = surface.vertices
        if z_offset:
            vertices = vertices.copy()
            vertices[:, 2] += z_offset

        if len(surface.triangles) > 0:
            mesh = self.o3d.geometry.TriangleMesh()
            mesh.vertices = self.o3d.utility.Vector3dVector(vertices)
            mesh.triangles = self.o3d.utility.Vector3iVector(surface.triangles)
            mesh.vertex_colors = self.o3d.utility.Vector3dVector(surface.colors)
            mesh.compute_vertex_normals()
            return mesh

        point_cloud = self.o3d.geometry.PointCloud()
        point_cloud.points = self.o3d.utility.Vector3dVector(vertices)
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

    def _render_geometry(self, geometry: Any, material: Any) -> None:
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("scan", geometry, material)
        self._scene_has_geometry = True
        self._frame_geometry(geometry)
        self.scene.force_redraw()
        self._update_view_controls()

    def _camera_preset_key(self, text: str | None = None) -> str:
        raw = str(text or self.camera_combo.selected_text or "Current / Mouse").lower()
        if "iso" in raw:
            return "isometric"
        if "top" in raw:
            return "top"
        if "front" in raw:
            return "front"
        if "rear" in raw:
            return "rear"
        if "left" in raw:
            return "left"
        if "right" in raw:
            return "right"
        return "current"

    def _apply_camera_preset(self, preset: str) -> None:
        if self._current_scene_bounds is None or not self._scene_has_geometry:
            self.status.text = "Load a surface before applying a camera preset."
            return

        center = self._current_scene_bounds.get_center()
        extent = self._current_scene_bounds.get_extent()
        max_extent = max(float(extent[0]), float(extent[1]), float(extent[2]), 1.0)
        distance = max_extent * 2.5
        target = [float(center[0]), float(center[1]), float(center[2])]

        offsets = {
            "isometric": ([0.65, -1.0, 0.65], [0.0, 0.0, 1.0]),
            "top": ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            "front": ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            "rear": ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            "left": ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            "right": ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        }
        offset, up = offsets.get(preset, offsets["isometric"])
        eye = [
            target[0] + distance * offset[0],
            target[1] + distance * offset[1],
            target[2] + distance * offset[2],
        ]
        self.scene.look_at(target, eye, up)
        self.scene.force_redraw()
        self.status.text = f"Camera preset applied: {preset.replace('_', ' ').title()}"

    def _export_current_view(self) -> None:
        if not self._scene_has_geometry:
            self._show_error("Nothing to export", "Load or render a surface first.")
            return

        graph_dir = self._view_export_dir()
        if graph_dir is None:
            self._show_error(
                "No project loaded",
                "Open an ALF project or Surface Scanner folder before exporting a graph view.",
            )
            return

        try:
            graph_dir.mkdir(parents=True, exist_ok=True)
            width, height = self._export_image_size()
            image = self.gui.Application.instance.render_to_image(
                self.scene.scene,
                width,
                height,
            )
            out_path = self._next_view_export_path(graph_dir)
            ok = self.o3d.io.write_image(str(out_path), image, 9)
            if not ok:
                raise RuntimeError(f"Open3D failed to write image: {out_path}")
            self._write_view_export_metadata(out_path, width=width, height=height)
            self.status.text = f"Exported view graph: {out_path.name}"
            if self.details.text:
                self.details.text = f"{self.details.text}\nExported: {out_path}"
            else:
                self.details.text = f"Exported: {out_path}"
        except Exception as exc:
            self._show_error("Could not export view", str(exc))

    def _view_export_dir(self) -> Path | None:
        if self._loaded_surface_scanner_dir is not None:
            return self._loaded_surface_scanner_dir / "graphs" / SURFACE_VIEW_GRAPH_TYPE
        if self._loaded_project_dir is not None:
            return surface_scanner_graph_dir(self._loaded_project_dir)
        return None

    def _export_image_size(self) -> tuple[int, int]:
        frame = self.scene.frame
        width = max(1, int(getattr(frame, "width", 0) or 0))
        height = max(1, int(getattr(frame, "height", 0) or 0))
        if width <= 1 or height <= 1:
            return 1600, 1000
        return width, height

    def _next_view_export_path(self, graph_dir: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = self._view_export_stem(stamp)
        out_path = graph_dir / f"{stem}.png"
        counter = 2
        while out_path.exists():
            out_path = graph_dir / f"{stem}_{counter}.png"
            counter += 1
        return out_path

    def _view_export_stem(self, stamp: str) -> str:
        parts = ["surface_view", stamp, self._view_mode_slug()]
        if self._is_stack_mode():
            parts.append(f"{len(self._selected_stack_scans())}_milestones")
        elif self._current_scan is not None:
            parts.extend([self._current_scan.phase, self._current_scan.milestone])
        preset = self._camera_preset_key()
        if preset != "current":
            parts.append(preset)
        return "__".join(self._safe_filename_part(part) for part in parts if part)

    def _view_mode_slug(self) -> str:
        return str(self.view_mode_combo.selected_text or "view").strip().lower()

    @staticmethod
    def _safe_filename_part(value: object) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", text)
        return text.strip("._") or "view"

    def _write_view_export_metadata(
        self,
        out_path: Path,
        *,
        width: int,
        height: int,
    ) -> None:
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "ALF Scan Viewer",
            "image": out_path.name,
            "width": width,
            "height": height,
            "camera_preset": str(self.camera_combo.selected_text or "Current / Mouse"),
            "view_mode": str(self.view_mode_combo.selected_text or ""),
            "colour": self._display_mode_label(self._color_by()),
            "z_axis": self._display_mode_label(self._z_by()),
            "z_scale": self._z_scale(),
            "downsample_step": self._downsample_step(),
            "section": self._section_label(),
            "baseline": str(self._baseline_gridmap_path() or ""),
            "milestone": self._current_scan.label if self._current_scan else "",
            "stack_milestones": [scan.label for scan in self._selected_stack_scans()],
        }
        out_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=4),
            encoding="utf-8",
        )

    def _is_stack_mode(self) -> bool:
        return str(self.view_mode_combo.selected_text or "").strip().lower() == (
            "stack milestones"
        )

    def _update_view_controls(self) -> None:
        stack_mode = self._is_stack_mode()
        current = self._current_scan
        color_by = self._color_by()
        scalar_colour = color_by in {
            "height",
            "deformation",
            "absolute_deformation",
        }
        binary_colour = bool(self.color_binary_checkbox.checked)
        signed_deformation_gradient = color_by == "deformation" and not binary_colour
        self.milestone_combo.enabled = bool(self._milestone_scans)
        self.baseline_combo.enabled = bool(self._baseline_scans())
        self.color_binary_checkbox.enabled = scalar_colour
        threshold_enabled = scalar_colour and binary_colour
        self.color_threshold_slider.enabled = threshold_enabled
        self.color_threshold_input.enabled = threshold_enabled
        self.color_invert_checkbox.enabled = threshold_enabled or signed_deformation_gradient
        self.x_min_input.enabled = bool(self.limit_x_checkbox.checked)
        self.x_max_input.enabled = bool(self.limit_x_checkbox.checked)
        self.y_min_input.enabled = bool(self.limit_y_checkbox.checked)
        self.y_max_input.enabled = bool(self.limit_y_checkbox.checked)
        self.stack_separation_slider.enabled = stack_mode
        self.stack_separation_input.enabled = stack_mode
        self.camera_apply_button.enabled = self._scene_has_geometry
        self.camera_export_button.enabled = self._scene_has_geometry and (
            self._loaded_surface_scanner_dir is not None
            or self._loaded_project_dir is not None
        )
        self.stack_select_all_button.enabled = stack_mode and bool(
            self._stack_scan_checkboxes
        )
        self.stack_clear_button.enabled = stack_mode and bool(self._stack_scan_checkboxes)
        for _scan, checkbox in self._stack_scan_checkboxes:
            checkbox.enabled = stack_mode
        if hasattr(self, "stack_section"):
            self.stack_section.enabled = stack_mode
            self.stack_section.set_is_open(stack_mode)
        self._update_stack_selection_summary()
        if stack_mode:
            self.generate_button.enabled = bool(self._selected_stack_scans())
        else:
            self.generate_button.enabled = current.can_generate_surface if current else False

    def _clear_stack_cache(self) -> None:
        self._stack_surface_cache = []
        self._stack_cache_signature = None

    def _stackable_scans(self) -> list[MilestoneScan]:
        return [scan for scan in self._milestone_scans if scan.raw_gridmap_path is not None]

    def _selected_stack_scans(self) -> list[MilestoneScan]:
        if not self._stack_scan_checkboxes:
            return []
        return [
            scan
            for scan, checkbox in self._stack_scan_checkboxes
            if checkbox.checked and scan.raw_gridmap_path is not None
        ]

    def _update_stack_selection_summary(self) -> None:
        if not hasattr(self, "stack_selection_summary"):
            return
        total = len(self._stack_scan_checkboxes)
        selected = len(self._selected_stack_scans())
        if total:
            self.stack_selection_summary.text = f"{selected} of {total} selected"
        else:
            self.stack_selection_summary.text = "No raw gridmap milestones available."

    def _baseline_scans(self) -> list[MilestoneScan]:
        return [scan for scan in self._milestone_scans if scan.raw_gridmap_path is not None]

    def _stack_surface_cache_signature(self, scans: list[MilestoneScan]) -> tuple[Any, ...]:
        baseline_path = self._baseline_gridmap_path()
        return (
            tuple(str(scan.raw_gridmap_path) for scan in scans),
            self._downsample_step(),
            self._color_by(),
            self._color_binary(),
            self._color_threshold(),
            self._color_invert(),
            self._z_by(),
            self._z_scale(),
            str(baseline_path) if baseline_path is not None else "",
            self._x_range(),
            self._y_range(),
        )

    def _downsample_step(self) -> int:
        try:
            return max(1, int(self.downsample_input.int_value))
        except Exception:
            return 1

    def _color_by(self) -> str:
        value = str(self.color_combo.selected_text or "RGB").strip().lower()
        if value == "height":
            return "height"
        if value == "deformation from baseline":
            return "deformation"
        if value == "absolute deformation from baseline":
            return "absolute_deformation"
        return "rgb"

    def _color_binary(self) -> bool:
        return self._color_by() in {
            "height",
            "deformation",
            "absolute_deformation",
        } and bool(self.color_binary_checkbox.checked)

    def _color_threshold(self) -> float:
        try:
            return float(self.color_threshold_input.double_value) / 1000.0
        except Exception:
            return 0.0

    def _color_invert(self) -> bool:
        if not bool(self.color_invert_checkbox.checked):
            return False
        if self._color_binary():
            return True
        return self._color_by() == "deformation"

    def _z_by(self) -> str:
        value = str(self.z_combo.selected_text or "Mean Height").strip().lower()
        if value in {"deformation from baseline", "deformation from first scan"}:
            return "deformation"
        if value in {
            "absolute deformation from baseline",
            "absolute deformation from first scan",
        }:
            return "absolute_deformation"
        if value == "flat":
            return "flat"
        return "mean"

    def _z_scale(self) -> float:
        try:
            return float(self.z_scale_input.double_value)
        except Exception:
            return 1.0

    def _stack_separation(self) -> float:
        try:
            return max(0.0, float(self.stack_separation_input.double_value))
        except Exception:
            return 0.0

    def _x_range(self) -> tuple[float | None, float | None] | None:
        if not bool(self.limit_x_checkbox.checked):
            return None
        return self._input_range(self.x_min_input, self.x_max_input)

    def _y_range(self) -> tuple[float | None, float | None] | None:
        if not bool(self.limit_y_checkbox.checked):
            return None
        return self._input_range(self.y_min_input, self.y_max_input)

    @staticmethod
    def _input_range(start_input: Any, end_input: Any) -> tuple[float, float]:
        start = float(start_input.double_value)
        end = float(end_input.double_value)
        return (start, end) if start <= end else (end, start)

    def _selected_baseline_scan(self) -> MilestoneScan | None:
        baseline_scans = self._baseline_scans()
        index = int(getattr(self.baseline_combo, "selected_index", -1))
        if 0 <= index < len(baseline_scans):
            return baseline_scans[index]
        return default_baseline_scan(self._milestone_scans)

    def _uses_baseline(self) -> bool:
        deformation_modes = {"deformation", "absolute_deformation"}
        return self._z_by() in deformation_modes or self._color_by() in deformation_modes

    @staticmethod
    def _display_mode_label(mode: str) -> str:
        labels = {
            "rgb": "RGB",
            "height": "height",
            "mean": "mean height",
            "deformation": "signed deformation",
            "absolute_deformation": "absolute deformation",
            "flat": "flat",
        }
        return labels.get(mode, mode)

    def _baseline_gridmap_path(self, scan: MilestoneScan | None = None) -> Path | None:
        if not self._uses_baseline():
            return None
        selected = self._selected_baseline_scan()
        if selected is not None and selected.raw_gridmap_path is not None:
            return selected.raw_gridmap_path
        return scan.raw_gridmap_path if scan is not None else None

    def _scan_details(self, scan: MilestoneScan) -> str:
        processed = scan.processed_surface_path.name if scan.processed_surface_path else "None"
        raw = scan.raw_gridmap_path.name if scan.raw_gridmap_path else "None"
        baseline = self._baseline_gridmap_path(scan)
        baseline_name = baseline.name if baseline is not None else "None"
        return (
            f"Processed: {processed}\n"
            f"Raw gridmap: {raw}\n"
            f"Baseline: {baseline_name}\n"
            f"Section: {self._section_label()}"
        )

    def _stack_details(self, *, failures: list[str] | None = None) -> str:
        rendered_labels = [scan.label for scan, _surface in self._stack_surface_cache]
        selected_labels = [scan.label for scan in self._selected_stack_scans()]
        labels = rendered_labels or selected_labels
        baseline = self._baseline_gridmap_path()
        baseline_name = baseline.name if baseline is not None else "None"
        details = [
            f"Milestones: {len(selected_labels)} of {len(self._stack_scan_checkboxes)} selected",
            f"Baseline: {baseline_name}",
            f"Section: {self._section_label()}",
            f"Separation: {self._stack_separation():g}",
        ]
        if labels:
            details.extend(["Stack order:", *labels])
        if failures:
            details.append("Skipped:")
            details.extend(failures[:5])
        return "\n".join(details)

    def _section_label(self) -> str:
        parts: list[str] = []
        x_range = self._x_range()
        y_range = self._y_range()
        if x_range is not None:
            parts.append(f"X {x_range[0]:g} to {x_range[1]:g}")
        if y_range is not None:
            parts.append(f"Y {y_range[0]:g} to {y_range[1]:g}")
        return ", ".join(parts) if parts else "Full scan"

    def _frame_geometry(self, geometry: Any) -> None:
        self._frame_bounds(geometry.get_axis_aligned_bounding_box())

    def _frame_geometries(self, geometries: list[Any]) -> None:
        bounds_list = [geometry.get_axis_aligned_bounding_box() for geometry in geometries]
        min_bound = [
            min(float(bounds.get_min_bound()[axis]) for bounds in bounds_list)
            for axis in range(3)
        ]
        max_bound = [
            max(float(bounds.get_max_bound()[axis]) for bounds in bounds_list)
            for axis in range(3)
        ]
        self._frame_bounds(self.o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

    def _frame_bounds(self, bounds: Any) -> None:
        self._current_scene_bounds = bounds
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
    parser = argparse.ArgumentParser(description="Open an ALF Surface Scanner project.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional ALF project, Surface Scanner folder, or scan file to open.",
    )
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
