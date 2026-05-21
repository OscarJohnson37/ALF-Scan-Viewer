from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np


GRIDMAP_EXTENSIONS = {".npz"}
GEOMETRY_EXTENSIONS = {".ply", ".pcd", ".pts", ".xyz", ".xyzn", ".xyzrgb"}
SUPPORTED_EXTENSIONS = GRIDMAP_EXTENSIONS | GEOMETRY_EXTENSIONS


@dataclass(frozen=True)
class GridmapData:
    mean_grid: np.ndarray
    rgb_grid: np.ndarray | None
    x_centres: np.ndarray
    y_centres: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class GridmapSurface:
    vertices: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray
    source_path: Path
    downsample_step: int
    color_by: str
    z_by: str
    z_scale: float
    color_binary: bool = False
    color_threshold: float = 0.0
    color_invert: bool = False
    x_range: tuple[float | None, float | None] | None = None
    y_range: tuple[float | None, float | None] | None = None


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


def load_gridmap_data(path: Path) -> GridmapData:
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
    x_centres, y_centres = _grid_centres(metadata, grid_shape=mean_grid.shape)
    return GridmapData(
        mean_grid=mean_grid,
        rgb_grid=rgb_grid,
        x_centres=x_centres,
        y_centres=y_centres,
        source_path=path,
    )


def load_gridmap_surface(
    path: Path,
    *,
    downsample_step: int = 1,
    color_by: str = "rgb",
    color_binary: bool = False,
    color_threshold: float = 0.0,
    color_invert: bool = False,
    z_by: str = "mean",
    z_scale: float = 1.0,
    baseline_path: Path | None = None,
    baseline_data: GridmapData | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
) -> GridmapSurface:
    gridmap = load_gridmap_data(path)
    if baseline_data is None and baseline_path is not None:
        baseline_data = load_gridmap_data(baseline_path)
    return build_gridmap_surface(
        gridmap,
        downsample_step=downsample_step,
        color_by=color_by,
        color_binary=color_binary,
        color_threshold=color_threshold,
        color_invert=color_invert,
        z_by=z_by,
        z_scale=z_scale,
        baseline_data=baseline_data,
        x_range=x_range,
        y_range=y_range,
    )


def build_gridmap_surface(
    gridmap: GridmapData,
    *,
    downsample_step: int = 1,
    color_by: str = "rgb",
    color_binary: bool = False,
    color_threshold: float = 0.0,
    color_invert: bool = False,
    z_by: str = "mean",
    z_scale: float = 1.0,
    baseline_data: GridmapData | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
) -> GridmapSurface:
    downsample_step = _normalise_downsample_step(downsample_step)
    color_by = _normalise_color_by(color_by)
    color_threshold = _normalise_z_scale(color_threshold)
    z_by = _normalise_z_by(z_by)
    z_scale = _normalise_z_scale(z_scale)
    x_range = _normalise_axis_range(x_range)
    y_range = _normalise_axis_range(y_range)

    mean_grid = gridmap.mean_grid
    rgb_grid = gridmap.rgb_grid
    x_centres = gridmap.x_centres
    y_centres = gridmap.y_centres

    mean_grid = mean_grid[::downsample_step, ::downsample_step]
    x_centres = x_centres[::downsample_step]
    y_centres = y_centres[::downsample_step]
    if rgb_grid is not None:
        rgb_grid = rgb_grid[::downsample_step, ::downsample_step]

    z_grid = _z_grid(
        mean_grid,
        x_centres=x_centres,
        y_centres=y_centres,
        z_by=z_by,
        z_scale=z_scale,
        baseline_data=baseline_data,
    )
    yy, xx = np.meshgrid(y_centres, x_centres, indexing="ij")
    valid = (
        np.isfinite(mean_grid)
        & np.isfinite(z_grid)
        & _axis_range_mask(xx, x_range)
        & _axis_range_mask(yy, y_range)
    )
    if not np.any(valid):
        raise ValueError(f"Gridmap has no finite values for this view: {gridmap.source_path}")

    vertices = np.column_stack((xx[valid], yy[valid], z_grid[valid])).astype(float)
    colors = _grid_colors(
        rgb_grid,
        mean_grid=mean_grid,
        valid_mask=valid,
        expected_shape=mean_grid.shape,
        color_by=color_by,
        color_binary=color_binary,
        color_threshold=color_threshold,
        color_invert=color_invert,
        x_centres=x_centres,
        y_centres=y_centres,
        baseline_data=baseline_data,
    )
    triangles = _grid_triangles(valid)

    return GridmapSurface(
        vertices=vertices,
        triangles=triangles,
        colors=colors,
        source_path=gridmap.source_path,
        downsample_step=downsample_step,
        color_by=color_by,
        color_binary=bool(color_binary),
        color_threshold=color_threshold,
        color_invert=bool(color_invert),
        z_by=z_by,
        z_scale=z_scale,
        x_range=x_range,
        y_range=y_range,
    )


def stack_gridmap_surfaces(
    surfaces: Sequence[GridmapSurface], *, separation: float
) -> GridmapSurface:
    if not surfaces:
        raise ValueError("At least one surface is required for stacking.")

    separation = _normalise_z_scale(separation)
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    vertex_offset = 0

    for index, surface in enumerate(surfaces):
        shifted_vertices = np.asarray(surface.vertices, dtype=float).copy()
        shifted_vertices[:, 2] += float(index) * separation
        vertices.append(shifted_vertices)
        colors.append(np.asarray(surface.colors, dtype=float))

        surface_triangles = np.asarray(surface.triangles, dtype=np.int32)
        if surface_triangles.size:
            triangles.append(surface_triangles + vertex_offset)
        vertex_offset += len(shifted_vertices)

    first = surfaces[0]
    return GridmapSurface(
        vertices=np.vstack(vertices),
        triangles=np.vstack(triangles) if triangles else np.empty((0, 3), dtype=np.int32),
        colors=np.vstack(colors),
        source_path=first.source_path,
        downsample_step=first.downsample_step,
        color_by=first.color_by,
        color_binary=first.color_binary,
        color_threshold=first.color_threshold,
        color_invert=first.color_invert,
        z_by=first.z_by,
        z_scale=first.z_scale,
        x_range=first.x_range,
        y_range=first.y_range,
    )


def _normalise_downsample_step(value: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _normalise_color_by(value: str) -> str:
    normalised = str(value or "rgb").strip().lower()
    aliases = {
        "rgb": "rgb",
        "height": "height",
        "mean": "height",
        "mean height": "height",
        "deformation": "deformation",
        "deformation from baseline": "deformation",
        "deformation from first scan": "deformation",
        "absolute deformation": "absolute_deformation",
        "absolute deformation from baseline": "absolute_deformation",
        "absolute deformation from first scan": "absolute_deformation",
        "absolute_deformation": "absolute_deformation",
    }
    if normalised in aliases:
        return aliases[normalised]
    return "rgb"


def _normalise_z_by(value: str) -> str:
    normalised = str(value or "mean").strip().lower()
    aliases = {
        "height": "mean",
        "mean height": "mean",
        "actual mean": "mean",
        "mean": "mean",
        "deformation": "deformation",
        "deformation from baseline": "deformation",
        "deformation from first scan": "deformation",
        "relative to first scan": "deformation",
        "absolute deformation": "absolute_deformation",
        "absolute deformation from baseline": "absolute_deformation",
        "absolute deformation from first scan": "absolute_deformation",
        "absolute_deformation": "absolute_deformation",
        "flat": "flat",
        "none": "flat",
    }
    return aliases.get(normalised, "mean")


def _normalise_z_scale(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return 1.0
    return out if np.isfinite(out) else 1.0


def _normalise_axis_range(
    axis_range: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None] | None:
    if axis_range is None:
        return None

    low_raw, high_raw = axis_range
    low = _finite_float_or_none(low_raw)
    high = _finite_float_or_none(high_raw)
    if low is None and high is None:
        return None
    if low is not None and high is not None and low > high:
        low, high = high, low
    return low, high


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _axis_range_mask(
    values: np.ndarray,
    axis_range: tuple[float | None, float | None] | None,
) -> np.ndarray:
    if axis_range is None:
        return np.ones(values.shape, dtype=bool)
    low, high = axis_range
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values >= low
    if high is not None:
        mask &= values <= high
    return mask


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


def _z_grid(
    mean_grid: np.ndarray,
    *,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    z_by: str,
    z_scale: float,
    baseline_data: GridmapData | None,
) -> np.ndarray:
    if z_by == "flat":
        return np.zeros_like(mean_grid, dtype=float)

    if z_by in {"deformation", "absolute_deformation"}:
        if baseline_data is None:
            raise ValueError("Deformation Z mode needs a baseline gridmap.")
        baseline_grid = _sample_reference_mean(
            baseline_data,
            target_x_centres=x_centres,
            target_y_centres=y_centres,
        )
        deformation = np.asarray(mean_grid, dtype=float) - baseline_grid
        if z_by == "absolute_deformation":
            deformation = np.abs(deformation)
        return deformation * z_scale

    return np.asarray(mean_grid, dtype=float) * z_scale


def _sample_reference_mean(
    reference: GridmapData,
    *,
    target_x_centres: np.ndarray,
    target_y_centres: np.ndarray,
) -> np.ndarray:
    if (
        reference.mean_grid.shape
        == (len(target_y_centres), len(target_x_centres))
        and np.allclose(reference.x_centres, target_x_centres)
        and np.allclose(reference.y_centres, target_y_centres)
    ):
        return reference.mean_grid.astype(float)

    x_index = _nearest_indices(reference.x_centres, target_x_centres)
    y_index = _nearest_indices(reference.y_centres, target_y_centres)
    sampled = reference.mean_grid[np.ix_(y_index, x_index)].astype(float)

    outside_x = (
        (target_x_centres < float(reference.x_centres.min()))
        | (target_x_centres > float(reference.x_centres.max()))
    )
    outside_y = (
        (target_y_centres < float(reference.y_centres.min()))
        | (target_y_centres > float(reference.y_centres.max()))
    )
    if np.any(outside_x):
        sampled[:, outside_x] = np.nan
    if np.any(outside_y):
        sampled[outside_y, :] = np.nan
    return sampled


def _nearest_indices(source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    source = np.asarray(source_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    if source.size == 0:
        raise ValueError("Cannot sample from an empty reference axis.")

    positions = np.searchsorted(source, target)
    right = np.clip(positions, 0, source.size - 1)
    left = np.clip(positions - 1, 0, source.size - 1)
    use_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    return np.where(use_right, right, left).astype(np.intp)


def _grid_colors(
    rgb_grid: np.ndarray | None,
    *,
    mean_grid: np.ndarray,
    valid_mask: np.ndarray,
    expected_shape: tuple[int, int],
    color_by: str,
    color_binary: bool,
    color_threshold: float,
    color_invert: bool,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    baseline_data: GridmapData | None,
) -> np.ndarray:
    if color_by == "height":
        return _scalar_colors(
            mean_grid,
            valid_mask=valid_mask,
            binary=color_binary,
            threshold=color_threshold,
            invert=color_invert,
        )

    if color_by in {"deformation", "absolute_deformation"}:
        if baseline_data is None:
            raise ValueError("Deformation colour mode needs a baseline gridmap.")
        baseline_grid = _sample_reference_mean(
            baseline_data,
            target_x_centres=x_centres,
            target_y_centres=y_centres,
        )
        deformation = np.asarray(mean_grid, dtype=float) - baseline_grid
        if color_by == "absolute_deformation":
            deformation = np.abs(deformation)
        return _scalar_colors(
            deformation,
            valid_mask=valid_mask,
            binary=color_binary,
            threshold=color_threshold,
            invert=color_invert,
        )

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


def _scalar_colors(
    scalar_grid: np.ndarray,
    *,
    valid_mask: np.ndarray,
    binary: bool,
    threshold: float,
    invert: bool,
) -> np.ndarray:
    values = np.asarray(scalar_grid, dtype=float)[valid_mask]
    if values.size == 0:
        return np.empty((0, 3), dtype=float)

    finite = np.isfinite(values)
    colors = np.full((values.size, 3), 0.55, dtype=float)
    if not np.any(finite):
        return colors

    finite_values = values[finite]
    if binary:
        above = finite_values > float(threshold)
        if invert:
            above = ~above
        colors[finite] = np.where(
            above[:, None],
            np.array([0.90, 0.08, 0.06], dtype=float),
            np.array([0.08, 0.66, 0.18], dtype=float),
        )
        return colors

    low = float(np.nanmin(finite_values))
    high = float(np.nanmax(finite_values))
    if np.isclose(high, low):
        t = np.zeros(finite_values.shape, dtype=float)
    else:
        t = np.clip((finite_values - low) / (high - low), 0.0, 1.0)
    if invert:
        t = 1.0 - t

    colors[finite] = _jet_colors(t)
    return colors.astype(float)


def _jet_colors(t: np.ndarray) -> np.ndarray:
    anchors = np.array(
        [
            [0.05, 0.08, 0.55],
            [0.00, 0.45, 1.00],
            [0.00, 0.85, 0.85],
            [0.18, 0.85, 0.18],
            [1.00, 0.92, 0.05],
            [1.00, 0.45, 0.00],
            [0.75, 0.00, 0.00],
        ],
        dtype=float,
    )
    positions = np.linspace(0.0, 1.0, len(anchors))
    channels = [np.interp(t, positions, anchors[:, index]) for index in range(3)]
    return np.column_stack(channels).astype(float)


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
