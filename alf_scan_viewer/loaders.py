from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np


GRIDMAP_EXTENSIONS = {".npz"}
GEOMETRY_EXTENSIONS = {".ply", ".pcd", ".pts", ".xyz", ".xyzn", ".xyzrgb"}
SUPPORTED_EXTENSIONS = GRIDMAP_EXTENSIONS | GEOMETRY_EXTENSIONS


@dataclass(frozen=True)
class GridmapSurface:
    vertices: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray
    source_path: Path


def supported_file_filter() -> str:
    return " ".join(sorted(SUPPORTED_EXTENSIONS))


def is_supported_path(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def require_supported_path(path: Path) -> Path:
    path = Path(path)
    if not is_supported_path(path):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {supported}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    return path


def load_gridmap_surface(path: Path) -> GridmapSurface:
    path = require_supported_path(path)
    if path.suffix.lower() not in GRIDMAP_EXTENSIONS:
        raise ValueError(f"Not an ALF gridmap file: {path}")

    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Matching gridmap JSON sidecar not found: {sidecar_path}")

    with np.load(path) as data:
        if "mean" not in data:
            available = ", ".join(data.files)
            raise KeyError(f"Gridmap requires a 'mean' array. Available: {available}")
        mean_grid = np.asarray(data["mean"], dtype=float)
        rgb_grid = np.asarray(data["rgb"], dtype=float) if "rgb" in data else None

    metadata = _normalise_gridmap_metadata(
        json.loads(sidecar_path.read_text(encoding="utf-8")),
        grid_shape=mean_grid.shape,
    )
    valid = np.isfinite(mean_grid)
    if not np.any(valid):
        raise ValueError(f"Gridmap has no finite 'mean' values: {path}")

    x_centres, y_centres = _grid_centres(metadata, grid_shape=mean_grid.shape)
    yy, xx = np.meshgrid(y_centres, x_centres, indexing="ij")
    vertices = np.column_stack((xx[valid], yy[valid], mean_grid[valid])).astype(float)
    colors = _grid_colors(rgb_grid, valid_mask=valid, expected_shape=mean_grid.shape)
    triangles = _grid_triangles(valid)

    return GridmapSurface(
        vertices=vertices,
        triangles=triangles,
        colors=colors,
        source_path=path,
    )


def _normalise_gridmap_metadata(
    metadata: dict[str, Any], *, grid_shape: tuple[int, int]
) -> dict[str, Any]:
    out = dict(metadata or {})
    missing = [key for key in ("x_min", "x_max", "y_min", "y_max") if key not in out]
    if missing:
        raise KeyError(f"Gridmap JSON missing: {', '.join(missing)}")

    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    out["cell_size_x"] = float(
        out.get("cell_size_x", (float(out["x_max"]) - float(out["x_min"])) / nx)
    )
    out["cell_size_y"] = float(
        out.get("cell_size_y", (float(out["y_max"]) - float(out["y_min"])) / ny)
    )
    return out


def _grid_centres(
    metadata: dict[str, Any], *, grid_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    x = float(metadata["x_min"]) + (np.arange(nx, dtype=float) + 0.5) * float(
        metadata["cell_size_x"]
    )
    y = float(metadata["y_min"]) + (np.arange(ny, dtype=float) + 0.5) * float(
        metadata["cell_size_y"]
    )
    return x, y


def _grid_colors(
    rgb_grid: np.ndarray | None,
    *,
    valid_mask: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    default = np.full((int(valid_mask.sum()), 3), 0.78, dtype=float)
    if rgb_grid is None:
        return default

    rgb = np.asarray(rgb_grid, dtype=float)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    elif rgb.ndim == 3 and rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.ndim == 3 and rgb.shape[2] >= 3:
        rgb = rgb[:, :, :3]
    else:
        return default

    if rgb.shape[:2] != expected_shape:
        return default

    finite_values = rgb[np.isfinite(rgb)]
    if finite_values.size == 0:
        return default

    scale = 255.0 if float(finite_values.max()) > 1.0 + 1e-6 else 1.0
    fill = 255.0 if scale == 255.0 else 1.0
    rgb = np.where(np.isfinite(rgb), rgb, fill)
    return np.clip(rgb / scale, 0.0, 1.0)[valid_mask].astype(float)


def _grid_triangles(valid_mask: np.ndarray) -> np.ndarray:
    index_map = np.full(valid_mask.shape, -1, dtype=np.int32)
    index_map[valid_mask] = np.arange(int(valid_mask.sum()), dtype=np.int32)
    rows, cols = valid_mask.shape
    triangles: list[tuple[int, int, int]] = []

    for row in range(rows - 1):
        for col in range(cols - 1):
            top_left = int(index_map[row, col])
            top_right = int(index_map[row, col + 1])
            bottom_left = int(index_map[row + 1, col])
            bottom_right = int(index_map[row + 1, col + 1])
            if top_left >= 0 and bottom_left >= 0 and top_right >= 0:
                triangles.append((top_left, bottom_left, top_right))
            if top_right >= 0 and bottom_left >= 0 and bottom_right >= 0:
                triangles.append((top_right, bottom_left, bottom_right))

    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    triangles_array = np.asarray(triangles, dtype=np.int32)
    return np.vstack((triangles_array, triangles_array[:, ::-1]))

