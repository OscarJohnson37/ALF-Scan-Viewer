from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from alf_scan_viewer.loaders import (
    gridmap_scalar_array_keys,
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

    def test_scalar_grid_arrays_can_drive_colour_and_z_axis(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            np.savez(
                npz_path,
                mean=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                median=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
                variance=np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
                count=np.array([[1, 2], [3, 4]], dtype=int),
                rgb=np.zeros((2, 2, 3), dtype=float),
            )
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="array:median",
                color_by="height",
                color_source="array:variance",
            )

            self.assertEqual(
                gridmap_scalar_array_keys(npz_path),
                ["mean", "median", "variance", "count"],
            )
            self.assertEqual(surface.z_by, "height")
            self.assertEqual(surface.z_source, "array:median")
            self.assertEqual(surface.color_by, "height")
            self.assertEqual(surface.color_source, "array:variance")
            self.assertEqual(surface.vertices[:, 2].tolist(), [10.0, 20.0, 30.0, 40.0])
            self.assertGreater(surface.colors[0, 2], surface.colors[0, 0])
            self.assertGreater(surface.colors[-1, 0], surface.colors[-1, 2])

    def test_visual_smoothed_mean_suppresses_high_variance_spike(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            mean = np.zeros((3, 3), dtype=float)
            mean[1, 1] = 10.0
            variance = np.full((3, 3), 0.01, dtype=float)
            variance[1, 1] = 100.0
            np.savez(npz_path, mean=mean, variance=variance, count=np.ones((3, 3)))
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 3}),
                encoding="utf-8",
            )

            raw = load_gridmap_surface(npz_path)
            smoothed = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="mean (visual smoothed)",
            )

            self.assertEqual(raw.vertices[4, 2], 10.0)
            self.assertLess(smoothed.vertices[4, 2], 0.01)
            self.assertEqual(smoothed.z_by, "height")
            self.assertEqual(smoothed.z_source, "visual_mean")

    def test_visual_smoothed_mean_repairs_high_variance_patch(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            mean = np.zeros((25, 25), dtype=float)
            mean[8:17, 8:17] = 10.0
            variance = np.full((25, 25), 0.01, dtype=float)
            variance[8:17, 8:17] = 100.0
            np.savez(npz_path, mean=mean, variance=variance)
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 25, "y_min": 0, "y_max": 25}),
                encoding="utf-8",
            )

            corrected = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="mean (visual smoothed)",
            )

            self.assertLess(corrected.vertices[312, 2], 0.1)

    def test_visual_smoothed_mean_does_not_move_trusted_neighbour(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            mean = np.zeros((5, 5), dtype=float)
            mean[2, 1] = 3.0
            mean[2, 2] = 10.0
            variance = np.full((5, 5), 0.01, dtype=float)
            variance[2, 2] = 100.0
            np.savez(npz_path, mean=mean, variance=variance)
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5}),
                encoding="utf-8",
            )

            corrected = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="mean (visual smoothed)",
            )

            self.assertEqual(corrected.vertices[11, 2], 3.0)
            self.assertLess(corrected.vertices[12, 2], 3.0)

    def test_visual_smoothed_mean_uses_point_count_for_confidence(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            mean = np.zeros((7, 7), dtype=float)
            mean[2, 2] = 10.0
            mean[4, 4] = 10.0
            variance = np.ones((7, 7), dtype=float)
            count = np.full((7, 7), 100, dtype=int)
            count[2, 2] = 1
            np.savez(npz_path, mean=mean, variance=variance, count=count)
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 7, "y_min": 0, "y_max": 7}),
                encoding="utf-8",
            )

            smoothed = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="mean (visual smoothed)",
            )

            self.assertLess(smoothed.vertices[16, 2], 0.01)
            self.assertEqual(smoothed.vertices[32, 2], 10.0)

    def test_processed_measurement_confidence_drives_visual_smoothing(self) -> None:
        with TemporaryDirectory() as td:
            milestone = (
                Path(td)
                / "data"
                / "tools"
                / "surface_scanner"
                / "milestones"
                / "post_bedding_in"
                / "milestone_1000"
            )
            raw_dir = milestone / "raw"
            processed_dir = milestone / "processed"
            raw_dir.mkdir(parents=True)
            processed_dir.mkdir()
            npz_path = raw_dir / "scan.npz"
            mean = np.zeros((3, 3), dtype=float)
            mean[1, 1] = 10.0
            np.savez(
                npz_path,
                mean=mean,
                variance=np.full((3, 3), 0.01, dtype=float),
                count=np.full((3, 3), 100, dtype=float),
            )
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 3}),
                encoding="utf-8",
            )

            processed_confidence = np.ones((3, 3), dtype=float)
            processed_confidence[1, 1] = 0.0
            np.savez(
                processed_dir
                / "project_post_bedding_in_1000_measurement_confidence.npz",
                confidence=processed_confidence,
                standard_error=np.full((3, 3), 0.25, dtype=float),
            )

            smoothed = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="mean (visual smoothed)",
            )

            self.assertIn("confidence", gridmap_scalar_array_keys(npz_path))
            self.assertIn("standard_error", gridmap_scalar_array_keys(npz_path))
            self.assertLess(smoothed.vertices[4, 2], 0.01)

    def test_processed_measurement_confidence_can_drive_colour_and_z_axis(
        self,
    ) -> None:
        with TemporaryDirectory() as td:
            milestone = (
                Path(td)
                / "data"
                / "tools"
                / "surface_scanner"
                / "milestones"
                / "post_bedding_in"
                / "milestone_1000"
            )
            raw_dir = milestone / "raw"
            processed_dir = milestone / "processed"
            raw_dir.mkdir(parents=True)
            processed_dir.mkdir()
            npz_path = raw_dir / "scan.npz"
            np.savez(npz_path, mean=np.zeros((2, 2), dtype=float))
            npz_path.with_suffix(".json").write_text(
                json.dumps({"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}),
                encoding="utf-8",
            )
            np.savez(
                processed_dir
                / "project_post_bedding_in_1000_measurement_confidence.npz",
                confidence=np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float),
                standard_error=np.array([[4.0, 3.0], [2.0, 1.0]], dtype=float),
            )

            surface = load_gridmap_surface(
                npz_path,
                z_by="height",
                z_source="array:standard_error",
                color_by="height",
                color_source="array:confidence",
            )
            available = gridmap_scalar_array_keys(npz_path)
            self.assertIn("confidence", available)
            self.assertIn("standard_error", available)
            self.assertEqual(surface.z_source, "array:standard_error")
            self.assertEqual(surface.color_source, "array:confidence")
            self.assertEqual(surface.vertices[:, 2].tolist(), [4.0, 3.0, 2.0, 1.0])
            self.assertGreater(surface.colors[-1, 0], surface.colors[0, 0])

    def test_scalar_grid_source_can_drive_deformation_modes(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            np.savez(
                baseline_path,
                mean=np.zeros((2, 2), dtype=float),
                median=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            )
            np.savez(
                current_path,
                mean=np.zeros((2, 2), dtype=float),
                median=np.array([[1.5, 1.5], [4.0, 6.0]], dtype=float),
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
                z_source="array:median",
                color_by="deformation from baseline",
                color_source="array:median",
                baseline_path=baseline_path,
            )

            self.assertEqual(surface.z_by, "absolute_deformation")
            self.assertEqual(surface.z_source, "array:median")
            self.assertEqual(surface.color_by, "deformation")
            self.assertEqual(surface.color_source, "array:median")
            self.assertEqual(surface.vertices[:, 2].tolist(), [0.5, 0.5, 1.0, 2.0])

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

    def test_visual_smoothed_deformation_suppresses_noisy_current_spike(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 3}
            clean_variance = np.full((3, 3), 0.01, dtype=float)
            current = np.zeros((3, 3), dtype=float)
            current[1, 1] = 10.0
            current_variance = clean_variance.copy()
            current_variance[1, 1] = 100.0
            np.savez(
                baseline_path,
                mean=np.zeros((3, 3), dtype=float),
                variance=clean_variance,
                count=np.ones((3, 3), dtype=float),
            )
            np.savez(
                current_path,
                mean=current,
                variance=current_variance,
                count=np.ones((3, 3), dtype=float),
            )
            baseline_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            current_path.with_suffix(".json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            z_surface = load_gridmap_surface(
                current_path,
                z_by="absolute deformation from baseline",
                z_source="mean (visual smoothed)",
                baseline_path=baseline_path,
            )
            color_surface = load_gridmap_surface(
                current_path,
                color_by="deformation from baseline",
                color_source="mean (visual smoothed)",
                color_binary=True,
                color_threshold=1.0,
                baseline_path=baseline_path,
            )

            self.assertEqual(z_surface.z_by, "absolute_deformation")
            self.assertEqual(z_surface.z_source, "visual_mean")
            self.assertLess(z_surface.vertices[4, 2], 0.01)
            self.assertEqual(color_surface.color_by, "deformation")
            self.assertEqual(color_surface.color_source, "visual_mean")
            self.assertEqual(color_surface.colors[4].tolist(), [0.08, 0.66, 0.18])

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

    def test_continuous_scalar_threshold_sets_first_red_point(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(npz_path, mean=np.array([[0.0, 1.0, 3.0]], dtype=float))
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 1}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                npz_path,
                color_by="height",
                color_threshold=1.0,
            )

            self.assertEqual(surface.color_value_min, 0.0)
            self.assertEqual(surface.color_value_max, 3.0)
            self.assertEqual(surface.colors[1].tolist(), surface.colors[2].tolist())
            self.assertGreater(surface.colors[1, 0], surface.colors[1, 2])
            self.assertGreater(surface.colors[0, 2], surface.colors[0, 0])

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

    def test_can_colour_by_mean_deformation_by_chainage(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 2}
            np.savez(baseline_path, mean=np.zeros((2, 3), dtype=float))
            np.savez(
                current_path,
                mean=np.array([[0.0, 1.0, 5.0], [0.0, 3.0, 7.0]], dtype=float),
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
                color_by="mean deformation by chainage",
                baseline_path=baseline_path,
                z_by="flat",
            )

            self.assertEqual(surface.color_by, "mean_deformation_by_chainage")
            self.assertEqual(surface.colors[0].tolist(), surface.colors[3].tolist())
            self.assertEqual(surface.colors[1].tolist(), surface.colors[4].tolist())
            self.assertEqual(surface.colors[2].tolist(), surface.colors[5].tolist())
            self.assertGreater(surface.colors[0, 2], surface.colors[0, 0])
            self.assertGreater(surface.colors[2, 0], surface.colors[2, 2])

    def test_can_colour_by_max_deformation_by_chainage(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2}
            np.savez(baseline_path, mean=np.zeros((2, 2), dtype=float))
            np.savez(
                current_path,
                mean=np.array([[0.0, 1.0], [0.0, 4.0]], dtype=float),
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
                color_by="max deformation by chainage",
                baseline_path=baseline_path,
                z_by="flat",
            )

            self.assertEqual(surface.color_by, "max_deformation_by_chainage")
            self.assertEqual(surface.colors[0].tolist(), surface.colors[2].tolist())
            self.assertEqual(surface.colors[1].tolist(), surface.colors[3].tolist())
            self.assertGreater(surface.colors[0, 2], surface.colors[0, 0])
            self.assertGreater(surface.colors[1, 0], surface.colors[1, 2])

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

    def test_inverted_continuous_threshold_keeps_low_values_red(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            baseline_path = root / "baseline.npz"
            current_path = root / "current.npz"
            metadata = {"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 1}
            np.savez(baseline_path, mean=np.zeros((1, 3), dtype=float))
            np.savez(current_path, mean=np.array([[-1.0, 0.0, 2.0]], dtype=float))
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
                color_threshold=0.0,
                color_invert=True,
                baseline_path=baseline_path,
            )

            self.assertEqual(surface.colors[0].tolist(), surface.colors[1].tolist())
            self.assertGreater(surface.colors[0, 0], surface.colors[0, 2])
            self.assertGreater(surface.colors[2, 2], surface.colors[2, 0])

    def test_can_invert_continuous_scalar_red_point(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            npz_path = root / "scan.npz"
            json_path = root / "scan.json"
            np.savez(npz_path, mean=np.array([[0.0, 1.0, 3.0]], dtype=float))
            json_path.write_text(
                json.dumps({"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 1}),
                encoding="utf-8",
            )

            surface = load_gridmap_surface(
                npz_path,
                color_by="height",
                color_threshold=1.0,
                color_invert=True,
            )

            self.assertTrue(surface.color_invert)
            self.assertEqual(surface.colors[0].tolist(), surface.colors[1].tolist())
            self.assertGreater(surface.colors[0, 0], surface.colors[0, 2])
            self.assertGreater(surface.colors[2, 2], surface.colors[2, 0])

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
