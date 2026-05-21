from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from alf_scan_viewer.loaders import (
    load_gridmap_surface,
    require_supported_path,
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


if __name__ == "__main__":
    unittest.main()

