from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from alf_scan_viewer.loaders import (
    load_gridmap_surface,
    require_supported_path,
    stack_gridmap_surfaces,
    supported_file_filter,
)


class LoaderTests(unittest.TestCase):
    def test_supported_file_filter_is_narrow(self) -> None:
        file_filter = supported_file_filter()

        self.assertIn(".npz", file_filter)
        self.assertIn(".ply", file_filter)
        self.assertNotIn(".json", file_filter)
        self.assertNotIn(".obj", file_filter)

    def test_rejects_unsupported_files(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "scan.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                require_supported_path(path)

    def test_loads_gridmap_surface_from_mean_and_rgb(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(
                npz_path,
                mean=np.array([[1.0, 2.0], [3.0, np.nan]], dtype=float),
                rgb=np.array(
                    [
                        [[255, 0, 0], [0, 255, 0]],
                        [[0, 0, 255], [255, 255, 255]],
                    ],
                    dtype=np.uint8,
                ),
            )
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 2, "y_min": 10, "y_max": 12}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(npz_path)

            self.assertEqual(surface.vertices.shape, (3, 3))
            self.assertEqual(surface.triangles.shape, (2, 3))
            self.assertEqual(surface.colors.shape, (3, 3))
            self.assertEqual(surface.vertices[0].tolist(), [0.5, 10.5, 1.0])
            self.assertEqual(surface.colors[0].tolist(), [1.0, 0.0, 0.0])

    def test_downsamples_and_can_colour_by_height(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(
                npz_path,
                mean=np.arange(16, dtype=float).reshape((4, 4)),
                rgb=np.full((4, 4, 3), 128, dtype=np.uint8),
            )
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 4}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                npz_path,
                downsample_step=2,
                color_by="height",
            )

            self.assertEqual(surface.downsample_step, 2)
            self.assertEqual(surface.color_by, "height")
            self.assertEqual(surface.vertices.shape, (4, 3))
            self.assertEqual(surface.triangles.shape, (4, 3))
            self.assertNotEqual(surface.colors[0].tolist(), surface.colors[-1].tolist())
            self.assertGreater(surface.colors[0, 2], surface.colors[0, 0])
            self.assertGreater(surface.colors[-1, 0], surface.colors[-1, 2])

    def test_can_use_deformation_as_z_axis(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            np.savez(
                baseline_path,
                mean=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            )
            np.savez(
                current_path,
                mean=np.array([[1.5, 1.5], [4.0, 6.0]], dtype=float),
            )
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                current_path,
                z_by="deformation",
                z_scale=10.0,
                baseline_path=baseline_path,
            )

            self.assertEqual(surface.z_by, "deformation")
            self.assertEqual(surface.z_scale, 10.0)
            self.assertEqual(surface.vertices[:, 2].tolist(), [5.0, -5.0, 10.0, 20.0])

    def test_can_use_absolute_deformation_as_z_axis(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            np.savez(
                baseline_path,
                mean=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            )
            np.savez(
                current_path,
                mean=np.array([[1.5, 1.5], [4.0, 6.0]], dtype=float),
            )
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                current_path,
                z_by="absolute deformation from baseline",
                z_scale=10.0,
                baseline_path=baseline_path,
            )

            self.assertEqual(surface.z_by, "absolute_deformation")
            self.assertEqual(surface.vertices[:, 2].tolist(), [5.0, 5.0, 10.0, 20.0])

    def test_can_colour_by_deformation_from_baseline(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            np.savez(baseline_path, mean=np.zeros((2, 2), dtype=float))
            np.savez(
                current_path,
                mean=np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=float),
            )
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                current_path,
                color_by="deformation",
                color_binary=True,
                color_threshold=0.0,
                baseline_path=baseline_path,
            )

            self.assertEqual(surface.color_by, "deformation")
            self.assertTrue(surface.color_binary)
            self.assertEqual(surface.colors[0].tolist(), [0.08, 0.66, 0.18])
            self.assertEqual(surface.colors[1].tolist(), [0.9, 0.08, 0.06])

    def test_can_colour_by_absolute_deformation_from_baseline(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            np.savez(baseline_path, mean=np.zeros((2, 2), dtype=float))
            np.savez(
                current_path,
                mean=np.array([[0.0, 1.0], [2.0, -2.0]], dtype=float),
            )
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                current_path,
                color_by="absolute deformation from baseline",
                baseline_path=baseline_path,
            )

            self.assertEqual(surface.color_by, "absolute_deformation")
            self.assertGreater(surface.colors[0, 2], surface.colors[0, 0])
            self.assertGreater(surface.colors[-1, 0], surface.colors[-1, 2])

    def test_zero_absolute_deformation_colours_as_low_end_blue(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            mean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
            np.savez(baseline_path, mean=mean)
            np.savez(current_path, mean=mean)
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                current_path,
                color_by="absolute deformation from baseline",
                baseline_path=baseline_path,
            )

            self.assertTrue(np.all(surface.colors[:, 2] > surface.colors[:, 0]))
            self.assertTrue(np.all(surface.colors[:, 2] > surface.colors[:, 1]))

    def test_can_invert_signed_deformation_gradient(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1}
            np.savez(baseline_path, mean=np.zeros((1, 2), dtype=float))
            np.savez(current_path, mean=np.array([[-1.0, 1.0]], dtype=float))
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            normal = load_gridmap_surface(
                current_path,
                color_by="deformation",
                baseline_path=baseline_path,
            )
            inverted = load_gridmap_surface(
                current_path,
                color_by="deformation",
                color_invert=True,
                baseline_path=baseline_path,
            )

            self.assertGreater(normal.colors[0, 2], normal.colors[0, 0])
            self.assertGreater(normal.colors[1, 0], normal.colors[1, 2])
            self.assertGreater(inverted.colors[0, 0], inverted.colors[0, 2])
            self.assertGreater(inverted.colors[1, 2], inverted.colors[1, 0])

    def test_can_invert_binary_colour_threshold(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(npz_path, mean=np.array([[-1.0, 1.0]], dtype=float))
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                npz_path,
                color_by="height",
                color_binary=True,
                color_threshold=0.0,
                color_invert=True,
            )

            self.assertTrue(surface.color_invert)
            self.assertEqual(surface.colors[0].tolist(), [0.9, 0.08, 0.06])
            self.assertEqual(surface.colors[1].tolist(), [0.08, 0.66, 0.18])

    def test_can_flatten_z_axis(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(npz_path, mean=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(npz_path, z_by="flat", z_scale=100.0)

            self.assertEqual(surface.z_by, "flat")
            self.assertEqual(surface.vertices[:, 2].tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_can_clip_to_selected_section(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(npz_path, mean=np.arange(16, dtype=float).reshape((4, 4)))
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 4}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(npz_path, x_range=(0.0, 1.0))

            self.assertEqual(surface.x_range, (0.0, 1.0))
            self.assertTrue(np.all(surface.vertices[:, 0] >= 0.0))
            self.assertTrue(np.all(surface.vertices[:, 0] <= 1.0))
            self.assertEqual(surface.vertices.shape, (4, 3))
            self.assertEqual(surface.triangles.shape, (0, 3))

    def test_stacks_surfaces_with_z_separation(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            first_path = root / "first.npz"
            second_path = root / "second.npz"
            np.savez(first_path, mean=np.full((2, 2), 1.0, dtype=float))
            np.savez(second_path, mean=np.full((2, 2), 2.0, dtype=float))
            first_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            second_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            first = load_gridmap_surface(first_path)
            second = load_gridmap_surface(second_path)
            stacked = stack_gridmap_surfaces([first, second], separation=10.0)

            self.assertEqual(stacked.vertices.shape, (8, 3))
            self.assertEqual(stacked.triangles.shape, (8, 3))
            self.assertEqual(stacked.vertices[:4, 2].tolist(), [1.0, 1.0, 1.0, 1.0])
            self.assertEqual(stacked.vertices[4:, 2].tolist(), [12.0, 12.0, 12.0, 12.0])
            self.assertTrue(np.all(stacked.triangles[4:] >= 4))


if __name__ == "__main__":
    unittest.main()
