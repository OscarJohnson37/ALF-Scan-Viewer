from __future__ import annotations

from types import SimpleNamespace
import unittest

from alf_scan_viewer.app import ViewerApp


class _DummySection:
    def __init__(self) -> None:
        self.enabled = False
        self.is_open = False

    def set_is_open(self, value: bool) -> None:
        self.is_open = bool(value)


class ViewerAppLogicTests(unittest.TestCase):
    def _build_app_stub(self) -> ViewerApp:
        app = ViewerApp.__new__(ViewerApp)
        app._current_scan = None
        app._milestone_scans = []
        app._scene_has_geometry = False
        app._loaded_surface_scanner_dir = None
        app._loaded_project_dir = None
        app._stack_scan_checkboxes = []
        app.stack_section = _DummySection()
        app._update_stack_selection_summary = lambda: None
        app._baseline_scans = lambda: []
        app._color_by = lambda: "height"
        app._color_binary = lambda: False
        app._z_by = lambda: "height"
        app._is_stack_mode = lambda: False

        app.milestone_combo = SimpleNamespace(enabled=False)
        app.baseline_combo = SimpleNamespace(enabled=False)
        app.color_source_combo = SimpleNamespace(enabled=False)
        app.z_source_combo = SimpleNamespace(enabled=False)
        app.color_binary_checkbox = SimpleNamespace(enabled=False, checked=False)
        app.color_threshold_slider = SimpleNamespace(enabled=False)
        app.color_threshold_input = SimpleNamespace(enabled=False)
        app.color_threshold_label = SimpleNamespace(text="")
        app.color_invert_checkbox = SimpleNamespace(enabled=False, checked=False)
        app.limit_x_checkbox = SimpleNamespace(checked=False)
        app.x_min_input = SimpleNamespace(enabled=False)
        app.x_max_input = SimpleNamespace(enabled=False)
        app.limit_y_checkbox = SimpleNamespace(checked=False)
        app.y_min_input = SimpleNamespace(enabled=False)
        app.y_max_input = SimpleNamespace(enabled=False)
        app.stack_separation_slider = SimpleNamespace(enabled=False)
        app.stack_separation_input = SimpleNamespace(enabled=False)
        app.camera_apply_button = SimpleNamespace(enabled=False)
        app.camera_export_button = SimpleNamespace(enabled=False)
        app.stack_select_all_button = SimpleNamespace(enabled=False)
        app.stack_clear_button = SimpleNamespace(enabled=False)
        app.generate_button = SimpleNamespace(enabled=False)
        return app

    def test_scalar_red_point_mode_enables_invert_checkbox(self) -> None:
        app = self._build_app_stub()

        ViewerApp._update_view_controls(app)

        self.assertTrue(app.color_invert_checkbox.enabled)
        self.assertEqual(app.color_threshold_label.text, "Red Point (mm)")

    def test_scalar_red_point_mode_respects_checked_invert_state(self) -> None:
        app = ViewerApp.__new__(ViewerApp)
        app.color_invert_checkbox = SimpleNamespace(checked=True)
        app._color_by = lambda: "height"

        self.assertTrue(ViewerApp._color_invert(app))
