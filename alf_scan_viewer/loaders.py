from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import warnings

import numpy as np


GRIDMAP_EXTENSIONS = {".npz"}
GEOMETRY_EXTENSIONS = {".ply", ".pcd", ".pts", ".xyz", ".xyzn", ".xyzrgb"}
SUPPORTED_EXTENSIONS = GRIDMAP_EXTENSIONS | GEOMETRY_EXTENSIONS
GRID_ARRAY_MODE_PREFIX = "array:"
_VARIANCE_ARRAY_KEYS = ("variance", "var")
_COUNT_ARRAY_KEYS = ("count", "counts", "sample_count", "n")
_CONFIDENCE_ARRAY_KEYS = ("confidence", "mean_confidence", "measurement_confidence")
_VISUAL_VARIANCE_REFERENCE_PERCENTILE = 75.0
_VISUAL_COUNT_REFERENCE_PERCENTILE = 50.0
_VISUAL_VARIANCE_CONFIDENCE_POWER = 2.0
_VISUAL_COUNT_CONFIDENCE_POWER = 0.5
_VISUAL_LOW_CONFIDENCE = 0.4
_VISUAL_VERY_LOW_CONFIDENCE = 0.15
_VISUAL_OUTLIER_MAD_FACTOR = 3.0
_VISUAL_OUTLIER_RADIUS = 2
_VISUAL_INPAINT_ITERATIONS = 20
_VISUAL_FALLBACK_RADII = (3, 5, 7, 10, 15, 20)
_VISUAL_DONOR_WEIGHT_POWER = 4.0


@dataclass(frozen=True)
class GridmapData:
    mean_grid: np.ndarray
    rgb_grid: np.ndarray | None
    scalar_grids: dict[str, np.ndarray]
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
    color_source: str
    z_by: str
    z_source: str
    z_scale: float
    color_binary: bool = False
    color_threshold: float | None = None
    color_invert: bool = False
    color_value_min: float | None = None
    color_value_max: float | None = None
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


def gridmap_scalar_array_keys(path: Path) -> list[str]:
    return list(load_gridmap_data(path).scalar_grids)


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
        scalar_grids = _scalar_grids_from_npz(data, mean_grid=mean_grid)
    scalar_grids.update(
        _processed_measurement_scalar_grids(path, expected_shape=mean_grid.shape)
    )

    metadata = _normalise_gridmap_metadata(
        json.loads(sidecar_path.read_text(encoding="utf-8")),
        grid_shape=mean_grid.shape,
    )
    x_centres, y_centres = _grid_centres(metadata, grid_shape=mean_grid.shape)
    return GridmapData(
        mean_grid=mean_grid,
        rgb_grid=rgb_grid,
        scalar_grids=scalar_grids,
        x_centres=x_centres,
        y_centres=y_centres,
        source_path=path,
    )


def load_gridmap_surface(
    path: Path,
    *,
    downsample_step: int = 1,
    color_by: str = "rgb",
    color_source: str | None = None,
    color_binary: bool = False,
    color_threshold: float | None = None,
    color_invert: bool = False,
    z_by: str = "height",
    z_source: str | None = None,
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
        color_source=color_source,
        color_binary=color_binary,
        color_threshold=color_threshold,
        color_invert=color_invert,
        z_by=z_by,
        z_source=z_source,
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
    color_source: str | None = None,
    color_binary: bool = False,
    color_threshold: float | None = None,
    color_invert: bool = False,
    z_by: str = "height",
    z_source: str | None = None,
    z_scale: float = 1.0,
    baseline_data: GridmapData | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
) -> GridmapSurface:
    downsample_step = _normalise_downsample_step(downsample_step)
    raw_color_by = color_by
    raw_z_by = z_by
    color_by = _normalise_color_by(raw_color_by)
    color_source = _normalise_scalar_source(
        color_source,
        fallback=_legacy_color_source(raw_color_by),
    )
    color_threshold = _normalise_color_threshold(color_threshold)
    z_by = _normalise_z_by(raw_z_by)
    z_source = _normalise_scalar_source(
        z_source,
        fallback=_legacy_z_source(raw_z_by),
    )
    z_scale = _normalise_z_scale(z_scale)
    x_range = _normalise_axis_range(x_range)
    y_range = _normalise_axis_range(y_range)

    mean_grid = gridmap.mean_grid[::downsample_step, ::downsample_step]
    rgb_grid = gridmap.rgb_grid
    x_centres = gridmap.x_centres[::downsample_step]
    y_centres = gridmap.y_centres[::downsample_step]

    if rgb_grid is not None:
        rgb_grid = rgb_grid[::downsample_step, ::downsample_step]

    z_grid = _z_grid(
        gridmap,
        downsample_step=downsample_step,
        x_centres=x_centres,
        y_centres=y_centres,
        z_by=z_by,
        z_source=z_source,
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
    colors, color_value_range = _grid_colors(
        rgb_grid,
        gridmap=gridmap,
        downsample_step=downsample_step,
        valid_mask=valid,
        expected_shape=mean_grid.shape,
        color_by=color_by,
        color_source=color_source,
        color_binary=color_binary,
        color_threshold=color_threshold,
        color_invert=color_invert,
        x_centres=x_centres,
        y_centres=y_centres,
        baseline_data=baseline_data,
    )
    triangles = _grid_triangles(valid)
    color_value_min, color_value_max = (None, None)
    if color_value_range is not None:
        color_value_min, color_value_max = color_value_range

    return GridmapSurface(
        vertices=vertices,
        triangles=triangles,
        colors=colors,
        source_path=gridmap.source_path,
        downsample_step=downsample_step,
        color_by=color_by,
        color_source=color_source,
        color_binary=bool(color_binary),
        color_threshold=color_threshold,
        color_invert=bool(color_invert),
        color_value_min=color_value_min,
        color_value_max=color_value_max,
        z_by=z_by,
        z_source=z_source,
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
        color_source=first.color_source,
        color_binary=first.color_binary,
        color_threshold=first.color_threshold,
        color_invert=first.color_invert,
        color_value_min=min(
            (
                float(surface.color_value_min)
                for surface in surfaces
                if surface.color_value_min is not None
            ),
            default=None,
        ),
        color_value_max=max(
            (
                float(surface.color_value_max)
                for surface in surfaces
                if surface.color_value_max is not None
            ),
            default=None,
        ),
        z_by=first.z_by,
        z_source=first.z_source,
        z_scale=first.z_scale,
        x_range=first.x_range,
        y_range=first.y_range,
    )


def _normalise_downsample_step(value: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _normalise_color_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    return _normalise_z_scale(value)


def _normalise_color_by(value: str) -> str:
    raw = str(value or "rgb").strip()
    if _normalise_grid_array_mode(raw) is not None:
        return "height"

    normalised = raw.lower()
    aliases = {
        "rgb": "rgb",
        "height": "height",
        "mean": "height",
        "mean height": "height",
        "denoised height": "height",
        "denoised mean": "height",
        "denoised mean height": "height",
        "denoised_height": "height",
        "deformation": "deformation",
        "deformation from baseline": "deformation",
        "deformation from first scan": "deformation",
        "denoised deformation": "deformation",
        "denoised deformation from baseline": "deformation",
        "denoised deformation from first scan": "deformation",
        "denoised_deformation": "deformation",
        "absolute deformation": "absolute_deformation",
        "absolute deformation from baseline": "absolute_deformation",
        "absolute deformation from first scan": "absolute_deformation",
        "absolute_deformation": "absolute_deformation",
        "denoised absolute deformation": "absolute_deformation",
        "denoised absolute deformation from baseline": "absolute_deformation",
        "denoised absolute deformation from first scan": "absolute_deformation",
        "denoised_absolute_deformation": "absolute_deformation",
        "mean deformation by chainage": "mean_deformation_by_chainage",
        "mean absolute deformation by chainage": "mean_deformation_by_chainage",
        "chainage mean deformation": "mean_deformation_by_chainage",
        "chainage mean absolute deformation": "mean_deformation_by_chainage",
        "max deformation by chainage": "max_deformation_by_chainage",
        "maximum deformation by chainage": "max_deformation_by_chainage",
        "max absolute deformation by chainage": "max_deformation_by_chainage",
        "maximum absolute deformation by chainage": "max_deformation_by_chainage",
        "chainage max deformation": "max_deformation_by_chainage",
        "chainage maximum deformation": "max_deformation_by_chainage",
        "chainage max absolute deformation": "max_deformation_by_chainage",
        "chainage maximum absolute deformation": "max_deformation_by_chainage",
    }
    if normalised in aliases:
        return aliases[normalised]
    return "rgb"


def _normalise_z_by(value: str) -> str:
    raw = str(value or "height").strip()
    if _normalise_grid_array_mode(raw) is not None:
        return "height"

    normalised = raw.lower()
    aliases = {
        "height": "height",
        "mean height": "height",
        "actual mean": "height",
        "mean": "height",
        "denoised height": "height",
        "denoised mean height": "height",
        "denoised mean": "height",
        "denoised_mean": "height",
        "deformation": "deformation",
        "deformation from baseline": "deformation",
        "deformation from first scan": "deformation",
        "relative to first scan": "deformation",
        "denoised deformation": "deformation",
        "denoised deformation from baseline": "deformation",
        "denoised deformation from first scan": "deformation",
        "denoised_deformation": "deformation",
        "absolute deformation": "absolute_deformation",
        "absolute deformation from baseline": "absolute_deformation",
        "absolute deformation from first scan": "absolute_deformation",
        "absolute_deformation": "absolute_deformation",
        "denoised absolute deformation": "absolute_deformation",
        "denoised absolute deformation from baseline": "absolute_deformation",
        "denoised absolute deformation from first scan": "absolute_deformation",
        "denoised_absolute_deformation": "absolute_deformation",
        "flat": "flat",
        "none": "flat",
    }
    return aliases.get(normalised, "height")


def _normalise_grid_array_mode(value: str) -> str | None:
    raw = str(value or "").strip()
    if raw.lower().startswith(GRID_ARRAY_MODE_PREFIX):
        key = raw.split(":", 1)[1].strip()
        return f"{GRID_ARRAY_MODE_PREFIX}{key}" if key else None
    return None


def _normalise_scalar_source(value: str | None, *, fallback: str) -> str:
    raw = str(value or fallback).strip()
    array_mode = _normalise_grid_array_mode(raw)
    if array_mode is not None:
        return array_mode

    aliases = {
        "mean": "mean",
        "height": "mean",
        "mean height": "mean",
        "visual_mean": "visual_mean",
        "visual mean": "visual_mean",
        "mean (visual smoothed)": "visual_mean",
        "visual smoothed mean": "visual_mean",
        "denoised_mean": "visual_mean",
        "denoised mean": "visual_mean",
        "denoised mean height": "visual_mean",
        "mean (variance corrected)": "visual_mean",
        "variance corrected mean": "visual_mean",
    }
    return aliases.get(raw.lower(), "mean")


def _legacy_color_source(value: str) -> str:
    raw = str(value or "rgb").strip()
    array_mode = _normalise_grid_array_mode(raw)
    if array_mode is not None:
        return array_mode
    if raw.lower() in {
        "denoised height",
        "denoised mean",
        "denoised mean height",
        "denoised_height",
        "denoised deformation",
        "denoised deformation from baseline",
        "denoised deformation from first scan",
        "denoised_deformation",
        "denoised absolute deformation",
        "denoised absolute deformation from baseline",
        "denoised absolute deformation from first scan",
        "denoised_absolute_deformation",
    }:
        return "visual_mean"
    return "mean"


def _legacy_z_source(value: str) -> str:
    raw = str(value or "height").strip()
    array_mode = _normalise_grid_array_mode(raw)
    if array_mode is not None:
        return array_mode
    if raw.lower() in {
        "denoised height",
        "denoised mean",
        "denoised mean height",
        "denoised_mean",
        "denoised deformation",
        "denoised deformation from baseline",
        "denoised deformation from first scan",
        "denoised_deformation",
        "denoised absolute deformation",
        "denoised absolute deformation from baseline",
        "denoised absolute deformation from first scan",
        "denoised_absolute_deformation",
    }:
        return "visual_mean"
    return "mean"


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


def _scalar_grids_from_npz(data: Any, *, mean_grid: np.ndarray) -> dict[str, np.ndarray]:
    scalar_grids: dict[str, np.ndarray] = {}
    for key in data.files:
        raw = np.asarray(data[key])
        if raw.ndim != 2 or raw.shape != mean_grid.shape:
            continue
        if not np.issubdtype(raw.dtype, np.number):
            continue
        scalar_grids[str(key)] = np.asarray(raw, dtype=float)
    scalar_grids.setdefault("mean", np.asarray(mean_grid, dtype=float))
    return scalar_grids


def _processed_measurement_scalar_grids(
    raw_gridmap_path: Path, *, expected_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    confidence_path = _processed_grid_artifact_path(
        raw_gridmap_path,
        pattern="*_measurement_confidence.npz",
    )
    if confidence_path is None:
        return {}

    scalar_grids: dict[str, np.ndarray] = {}
    try:
        with np.load(confidence_path) as data:
            for key in data.files:
                raw = np.asarray(data[key])
                if raw.ndim != 2 or raw.shape != expected_shape:
                    continue
                if not np.issubdtype(raw.dtype, np.number):
                    continue
                scalar_grids[str(key)] = np.asarray(raw, dtype=float)
    except Exception as exc:
        warnings.warn(
            f"Could not read processed measurement confidence '{confidence_path}': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    return scalar_grids


def _same_shape_numeric_grid(
    data: Any, *, key: str, expected_shape: tuple[int, int]
) -> np.ndarray | None:
    if key not in data:
        return None
    raw = np.asarray(data[key])
    if (
        raw.ndim != 2
        or raw.shape != expected_shape
        or not np.issubdtype(raw.dtype, np.number)
    ):
        return None
    return np.asarray(raw, dtype=float)


def _processed_grid_artifact_path(
    raw_gridmap_path: Path, *, pattern: str
) -> Path | None:
    for parent in Path(raw_gridmap_path).parents:
        if parent.name.lower() != "raw":
            continue
        processed_dir = parent.parent / "processed"
        if not processed_dir.is_dir():
            return None
        matches = sorted(processed_dir.glob(pattern))
        return matches[0] if matches else None
    return None


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
    gridmap: GridmapData,
    *,
    downsample_step: int,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    z_by: str,
    z_source: str,
    z_scale: float,
    baseline_data: GridmapData | None,
) -> np.ndarray:
    if z_by == "flat":
        return np.zeros_like(
            gridmap.mean_grid[::downsample_step, ::downsample_step], dtype=float
        )

    if _is_deformation_mode(z_by):
        return (
            _deformation_grid(
                gridmap,
                mode=z_by,
                source=z_source,
                downsample_step=downsample_step,
                x_centres=x_centres,
                y_centres=y_centres,
                baseline_data=baseline_data,
                error_prefix="Deformation Z mode",
            )
            * z_scale
        )

    return (
        _surface_scalar_grid(
            gridmap,
            source=z_source,
            downsample_step=downsample_step,
        )
        * z_scale
    )


def _sample_reference_grid(
    reference: GridmapData,
    reference_grid: np.ndarray,
    *,
    target_x_centres: np.ndarray,
    target_y_centres: np.ndarray,
) -> np.ndarray:
    if (
        reference_grid.shape
        == (len(target_y_centres), len(target_x_centres))
        and np.allclose(reference.x_centres, target_x_centres)
        and np.allclose(reference.y_centres, target_y_centres)
    ):
        return reference_grid.astype(float)

    x_index = _nearest_indices(reference.x_centres, target_x_centres)
    y_index = _nearest_indices(reference.y_centres, target_y_centres)
    sampled = reference_grid[np.ix_(y_index, x_index)].astype(float)

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


def _deformation_grid(
    gridmap: GridmapData,
    *,
    mode: str,
    source: str,
    downsample_step: int,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    baseline_data: GridmapData | None,
    error_prefix: str,
) -> np.ndarray:
    if baseline_data is None:
        raise ValueError(f"{error_prefix} needs a baseline gridmap.")

    if source == "visual_mean":
        deformation = _visual_smoothed_deformation_grid(
            gridmap,
            downsample_step=downsample_step,
            x_centres=x_centres,
            y_centres=y_centres,
            baseline_data=baseline_data,
        )
        return np.abs(deformation) if mode == "absolute_deformation" else deformation

    current_grid = _surface_scalar_grid(
        gridmap,
        source=source,
        downsample_step=downsample_step,
    )
    baseline_grid = _sample_reference_grid(
        baseline_data,
        _source_scalar_grid(baseline_data, source=source),
        target_x_centres=x_centres,
        target_y_centres=y_centres,
    )
    deformation = np.asarray(current_grid, dtype=float) - baseline_grid
    if mode == "absolute_deformation":
        deformation = np.abs(deformation)
    return deformation


def _visual_smoothed_deformation_grid(
    gridmap: GridmapData,
    *,
    downsample_step: int,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    baseline_data: GridmapData,
) -> np.ndarray:
    current_grid = _surface_scalar_grid(
        gridmap,
        source="mean",
        downsample_step=downsample_step,
    )
    baseline_grid = _sample_reference_grid(
        baseline_data,
        _source_scalar_grid(baseline_data, source="mean"),
        target_x_centres=x_centres,
        target_y_centres=y_centres,
    )
    deformation = np.asarray(current_grid, dtype=float) - baseline_grid

    current_confidence = _mean_visual_confidence(gridmap)
    baseline_confidence = _mean_visual_confidence(baseline_data)
    if current_confidence is None or baseline_confidence is None:
        return deformation

    current_confidence = np.asarray(current_confidence, dtype=float)[
        ::downsample_step, ::downsample_step
    ]
    sampled_baseline_confidence = _sample_reference_grid(
        baseline_data,
        np.asarray(baseline_confidence, dtype=float),
        target_x_centres=x_centres,
        target_y_centres=y_centres,
    )
    confidence = np.minimum(current_confidence, sampled_baseline_confidence)
    confidence[~np.isfinite(deformation)] = 0.0
    return _visual_smooth_grid(deformation, confidence=confidence)


def _surface_scalar_grid(
    gridmap: GridmapData,
    *,
    source: str,
    downsample_step: int,
) -> np.ndarray:
    return _source_scalar_grid(gridmap, source=source)[
        ::downsample_step, ::downsample_step
    ]


def _source_scalar_grid(gridmap: GridmapData, *, source: str) -> np.ndarray:
    if source == "mean":
        grid = gridmap.mean_grid
    elif source == "visual_mean":
        grid = _visual_mean_grid(gridmap)
    elif source.startswith(GRID_ARRAY_MODE_PREFIX):
        grid = _gridmap_array(gridmap, source.split(":", 1)[1])
    else:
        grid = gridmap.mean_grid
    return np.asarray(grid, dtype=float)


def _is_deformation_mode(mode: str) -> bool:
    return mode in {"deformation", "absolute_deformation"}


def _is_chainage_deformation_color_mode(mode: str) -> bool:
    return mode in {
        "mean_deformation_by_chainage",
        "max_deformation_by_chainage",
    }


def _gridmap_array(gridmap: GridmapData, key: str) -> np.ndarray:
    grid = gridmap.scalar_grids.get(str(key))
    if grid is None:
        available = ", ".join(gridmap.scalar_grids) or "(none)"
        raise KeyError(
            f"Gridmap '{gridmap.source_path.name}' has no scalar array '{key}'. "
            f"Available: {available}"
        )
    return np.asarray(grid, dtype=float)


def _visual_mean_grid(gridmap: GridmapData) -> np.ndarray:
    mean = np.asarray(gridmap.mean_grid, dtype=float)
    confidence = _mean_visual_confidence(gridmap)
    if confidence is None:
        return mean
    return _visual_smooth_grid(mean, confidence=confidence)


def _mean_visual_confidence(gridmap: GridmapData) -> np.ndarray | None:
    confidence = _first_scalar_array(gridmap, _CONFIDENCE_ARRAY_KEYS)
    if confidence is not None:
        stored_confidence = np.asarray(confidence, dtype=float)
        valid = np.isfinite(gridmap.mean_grid) & np.isfinite(stored_confidence)
        if np.any(valid & (stored_confidence > 0.0)):
            bounded_confidence = np.zeros(stored_confidence.shape, dtype=float)
            bounded_confidence[valid] = np.clip(stored_confidence[valid], 0.0, 1.0)
            return bounded_confidence

    variance = _first_scalar_array(gridmap, _VARIANCE_ARRAY_KEYS)
    if variance is None:
        return None
    count = _first_scalar_array(gridmap, _COUNT_ARRAY_KEYS)
    return _visual_confidence(
        variance=np.asarray(variance, dtype=float),
        count=None if count is None else np.asarray(count, dtype=float),
        valid=np.isfinite(gridmap.mean_grid),
    )


def _first_scalar_array(
    gridmap: GridmapData, keys: tuple[str, ...]
) -> np.ndarray | None:
    keys_lc = {key.lower(): key for key in gridmap.scalar_grids}
    for candidate in keys:
        actual = keys_lc.get(candidate.lower())
        if actual is not None:
            return gridmap.scalar_grids[actual]
    return None


def _visual_confidence(
    *,
    variance: np.ndarray,
    count: np.ndarray | None,
    valid: np.ndarray,
) -> np.ndarray | None:
    variance_grid = np.asarray(variance, dtype=float)
    use = valid & np.isfinite(variance_grid) & (variance_grid >= 0.0)
    positive = variance_grid[use & (variance_grid > 0.0)]
    if not positive.size:
        return None

    reference = max(
        float(np.nanpercentile(positive, _VISUAL_VARIANCE_REFERENCE_PERCENTILE)),
        np.finfo(float).eps,
    )
    ratio = np.zeros(variance_grid.shape, dtype=float)
    ratio[use] = variance_grid[use] / reference
    confidence = np.ones(variance_grid.shape, dtype=float)
    confidence[use] = 1.0 / (
        1.0 + ratio[use] ** _VISUAL_VARIANCE_CONFIDENCE_POWER
    )
    if count is not None and count.shape == variance_grid.shape:
        count_grid = np.asarray(count, dtype=float)
        count_use = valid & np.isfinite(count_grid) & (count_grid > 0.0)
        positive_counts = count_grid[count_use]
        if positive_counts.size:
            count_reference = max(
                float(
                    np.nanpercentile(
                        positive_counts,
                        _VISUAL_COUNT_REFERENCE_PERCENTILE,
                    )
                ),
                1.0,
            )
            count_ratio = np.zeros(count_grid.shape, dtype=float)
            count_ratio[count_use] = np.clip(
                count_grid[count_use] / count_reference,
                0.0,
                1.0,
            )
            confidence[count_use] *= (
                count_ratio[count_use] ** _VISUAL_COUNT_CONFIDENCE_POWER
            )
            confidence[valid & ~count_use] = 0.0
    confidence[~valid] = 0.0
    return confidence


def _visual_smooth_grid(
    values: np.ndarray, *, confidence: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    finite = np.isfinite(values)
    local_median = _local_median_grid(values, radius=_VISUAL_OUTLIER_RADIUS)
    local_mad = _local_median_grid(
        np.abs(values - local_median),
        radius=_VISUAL_OUTLIER_RADIUS,
    )
    local_scale = np.maximum(1.4826 * local_mad, np.finfo(float).eps)
    local_outlier = (
        finite
        & np.isfinite(local_median)
        & (np.abs(values - local_median) > _VISUAL_OUTLIER_MAD_FACTOR * local_scale)
    )
    repair = finite & (
        (confidence < _VISUAL_VERY_LOW_CONFIDENCE)
        | ((confidence < _VISUAL_LOW_CONFIDENCE) & local_outlier)
    )
    return _inpaint_visual_outliers(values, confidence=confidence, repair=repair)


def _inpaint_visual_outliers(
    values: np.ndarray,
    *,
    confidence: np.ndarray,
    repair: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    finite = np.isfinite(values)
    repair = np.asarray(repair, dtype=bool) & finite
    if not np.any(repair):
        return values

    # Outliers do not get to donate to their own visual repair.
    repaired_values = values.copy()
    repaired_values[repair] = np.nan
    missing = repair.copy()
    for _ in range(_VISUAL_INPAINT_ITERATIONS):
        local_estimate = _local_median_grid(repaired_values)
        fill = missing & np.isfinite(local_estimate)
        if not np.any(fill):
            break
        repaired_values[fill] = local_estimate[fill]
        missing[fill] = False
        if not np.any(missing):
            break

    if np.any(missing):
        donors = finite & ~repair
        for radius in _VISUAL_FALLBACK_RADII:
            fallback = _confidence_weighted_mean_grid(
                values,
                confidence=confidence,
                donor_mask=donors,
                radius=radius,
            )
            fill = missing & np.isfinite(fallback)
            repaired_values[fill] = fallback[fill]
            missing[fill] = False
            if not np.any(missing):
                break

    out = values.copy()
    use = repair & np.isfinite(repaired_values)
    out[use] = repaired_values[use]
    return out


def _local_median_grid(values: np.ndarray, *, radius: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rows, cols = values.shape

    radius = max(1, int(radius))
    padded_values = np.pad(values, radius, mode="constant", constant_values=np.nan)
    neighbours = np.stack(
        [
            padded_values[
                radius + row_offset : radius + row_offset + rows,
                radius + col_offset : radius + col_offset + cols,
            ]
            for row_offset in range(-radius, radius + 1)
            for col_offset in range(-radius, radius + 1)
        ],
        axis=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(neighbours, axis=0)


def _confidence_weighted_mean_grid(
    values: np.ndarray,
    *,
    confidence: np.ndarray,
    donor_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    donors = np.asarray(donor_mask, dtype=bool)
    use = (
        donors
        & np.isfinite(values)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )

    weights = np.zeros(values.shape, dtype=float)
    weights[use] = confidence[use] ** _VISUAL_DONOR_WEIGHT_POWER
    weighted_values = np.zeros(values.shape, dtype=float)
    weighted_values[use] = values[use] * weights[use]

    total = _box_sum_grid(weighted_values, radius=radius)
    weight_total = _box_sum_grid(weights, radius=radius)
    out = np.full(values.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = total / weight_total
    out[weight_total > 0.0] = averaged[weight_total > 0.0]
    return out


def _box_sum_grid(values: np.ndarray, *, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rows, cols = values.shape
    integral = np.pad(values, ((1, 0), (1, 0)), constant_values=0.0)
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    row_low = np.maximum(np.arange(rows) - int(radius), 0)
    row_high = np.minimum(np.arange(rows) + int(radius) + 1, rows)
    col_low = np.maximum(np.arange(cols) - int(radius), 0)
    col_high = np.minimum(np.arange(cols) + int(radius) + 1, cols)
    return (
        integral[row_high[:, None], col_high[None, :]]
        - integral[row_low[:, None], col_high[None, :]]
        - integral[row_high[:, None], col_low[None, :]]
        + integral[row_low[:, None], col_low[None, :]]
    )


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
    gridmap: GridmapData,
    downsample_step: int,
    valid_mask: np.ndarray,
    expected_shape: tuple[int, int],
    color_by: str,
    color_source: str,
    color_binary: bool,
    color_threshold: float | None,
    color_invert: bool,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    baseline_data: GridmapData | None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    scalar_grid, zero_floor = _color_scalar_grid(
        gridmap,
        downsample_step=downsample_step,
        color_by=color_by,
        color_source=color_source,
        valid_mask=valid_mask,
        x_centres=x_centres,
        y_centres=y_centres,
        baseline_data=baseline_data,
    )
    if scalar_grid is not None:
        return (
            _scalar_colors(
                scalar_grid,
                valid_mask=valid_mask,
                binary=color_binary,
                threshold=color_threshold,
                invert=color_invert,
                zero_floor=zero_floor,
            ),
            _scalar_value_range(scalar_grid, valid_mask=valid_mask),
        )

    return (
        _rgb_colors(
            rgb_grid,
            valid_mask=valid_mask,
            expected_shape=expected_shape,
        ),
        None,
    )


def _color_scalar_grid(
    gridmap: GridmapData,
    *,
    downsample_step: int,
    color_by: str,
    color_source: str,
    valid_mask: np.ndarray,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    baseline_data: GridmapData | None,
) -> tuple[np.ndarray | None, bool]:
    if color_by == "height":
        return (
            _surface_scalar_grid(
                gridmap,
                source=color_source,
                downsample_step=downsample_step,
            ),
            False,
        )

    if _is_chainage_deformation_color_mode(color_by):
        return (
            _chainage_deformation_color_grid(
                gridmap,
                statistic="max"
                if color_by == "max_deformation_by_chainage"
                else "mean",
                source=color_source,
                downsample_step=downsample_step,
                x_centres=x_centres,
                y_centres=y_centres,
                baseline_data=baseline_data,
                valid_mask=valid_mask,
            ),
            True,
        )

    if _is_deformation_mode(color_by):
        return (
            _deformation_grid(
                gridmap,
                mode=color_by,
                source=color_source,
                downsample_step=downsample_step,
                x_centres=x_centres,
                y_centres=y_centres,
                baseline_data=baseline_data,
                error_prefix="Deformation colour mode",
            ),
            color_by == "absolute_deformation",
        )

    return None, False


def _rgb_colors(
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


def _chainage_deformation_color_grid(
    gridmap: GridmapData,
    *,
    statistic: str,
    source: str,
    downsample_step: int,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    baseline_data: GridmapData | None,
    valid_mask: np.ndarray,
) -> np.ndarray:
    deformation = _deformation_grid(
        gridmap,
        mode="absolute_deformation",
        source=source,
        downsample_step=downsample_step,
        x_centres=x_centres,
        y_centres=y_centres,
        baseline_data=baseline_data,
        error_prefix="Chainage deformation colour mode",
    )
    if deformation.shape != valid_mask.shape:
        raise ValueError(
            "Chainage deformation colour mode produced an unexpected grid shape."
        )

    finite = np.isfinite(deformation) & np.asarray(valid_mask, dtype=bool)
    column_values = np.full(deformation.shape[1], np.nan, dtype=float)
    for column_index in range(deformation.shape[1]):
        values = deformation[:, column_index][finite[:, column_index]]
        if not values.size:
            continue
        if statistic == "max":
            column_values[column_index] = float(np.nanmax(values))
        else:
            column_values[column_index] = float(np.nanmean(values))
    return np.broadcast_to(column_values[None, :], deformation.shape).astype(float)


def _scalar_colors(
    scalar_grid: np.ndarray,
    *,
    valid_mask: np.ndarray,
    binary: bool,
    threshold: float | None,
    invert: bool,
    zero_floor: bool = False,
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
        above = finite_values > float(0.0 if threshold is None else threshold)
        if invert:
            above = ~above
        colors[finite] = np.where(
            above[:, None],
            np.array([0.90, 0.08, 0.06], dtype=float),
            np.array([0.08, 0.66, 0.18], dtype=float),
        )
        return colors

    low, high = _scalar_color_bounds(
        finite_values,
        zero_floor=zero_floor,
    )
    if threshold is None:
        if np.isclose(high, low):
            t = np.zeros(finite_values.shape, dtype=float)
        else:
            t = np.clip((finite_values - low) / (high - low), 0.0, 1.0)
        if invert:
            t = 1.0 - t
        colors[finite] = _jet_colors(t)
        return colors.astype(float)

    threshold = float(np.clip(threshold, low, high))
    if invert:
        if np.isclose(high, threshold):
            t = np.ones(finite_values.shape, dtype=float)
        else:
            clipped = np.maximum(finite_values, threshold)
            t = np.clip((clipped - threshold) / (high - threshold), 0.0, 1.0)
            t = 1.0 - t
    elif np.isclose(threshold, low):
        t = np.ones(finite_values.shape, dtype=float)
    else:
        clipped = np.minimum(finite_values, threshold)
        t = np.clip((clipped - low) / (threshold - low), 0.0, 1.0)
    colors[finite] = _jet_colors(t)
    return colors.astype(float)


def _scalar_value_range(
    scalar_grid: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> tuple[float, float] | None:
    values = np.asarray(scalar_grid, dtype=float)[valid_mask]
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return None
    return float(np.nanmin(finite_values)), float(np.nanmax(finite_values))


def _scalar_color_bounds(
    finite_values: np.ndarray,
    *,
    zero_floor: bool,
) -> tuple[float, float]:
    low = float(np.nanmin(finite_values))
    high = float(np.nanmax(finite_values))
    if zero_floor and high >= 0.0:
        low = min(0.0, low)
    return low, high


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
