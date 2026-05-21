from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re


SURFACE_VIEW_GRAPH_TYPE = "3d_surface_views"


@dataclass(frozen=True)
class MilestoneScan:
    phase: str
    milestone: str
    milestone_dir: Path
    processed_surface_path: Path | None
    raw_gridmap_path: Path | None
    source_kind: str
    shared_metadata_path: Path | None = None
    post_bedding_baseline: bool = False

    @property
    def label(self) -> str:
        suffix = " (ALF baseline)" if self.post_bedding_baseline else ""
        return f"{self.phase} / {self.milestone}{suffix}"

    @property
    def identity(self) -> tuple[str, str]:
        return self.phase, self.milestone

    @property
    def scan_path(self) -> Path:
        if self.processed_surface_path is not None:
            return self.processed_surface_path
        if self.raw_gridmap_path is not None:
            return self.raw_gridmap_path
        raise FileNotFoundError(f"No scan file is available for {self.label}")

    @property
    def can_generate_surface(self) -> bool:
        return self.raw_gridmap_path is not None


def resolve_surface_scanner_dir(path: Path) -> Path:
    path = Path(path)
    candidates = [
        path,
        path / "data" / "tools" / "surface_scanner",
        path / "tools" / "surface_scanner",
        path / "surface_scanner",
    ]

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "milestones").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find an ALF Surface Scanner directory. Select either the "
        "project folder or data/tools/surface_scanner."
    )


def resolve_project_dir(path: Path) -> Path | None:
    path = Path(path)
    if (path / "data" / "tools" / "surface_scanner").is_dir():
        return path

    try:
        surface_dir = resolve_surface_scanner_dir(path)
    except FileNotFoundError:
        surface_dir = path

    parts = surface_dir.parts
    if len(parts) >= 3 and parts[-3:] == ("data", "tools", "surface_scanner"):
        return surface_dir.parents[2]
    if (surface_dir.parent.parent / "milestones").is_dir():
        return surface_dir.parent.parent.parent
    return None


def surface_scanner_graph_dir(
    path: Path, graph_type: str = SURFACE_VIEW_GRAPH_TYPE
) -> Path:
    surface_dir = resolve_surface_scanner_dir(path)
    safe_graph_type = _safe_path_segment(graph_type) or SURFACE_VIEW_GRAPH_TYPE
    return surface_dir / "graphs" / safe_graph_type


def discover_milestone_scans(path: Path) -> list[MilestoneScan]:
    surface_dir = resolve_surface_scanner_dir(path)
    project_dir = resolve_project_dir(path)
    milestones_root = surface_dir / "milestones"
    scans: list[MilestoneScan] = []

    for phase_dir in sorted(p for p in milestones_root.iterdir() if p.is_dir()):
        for milestone_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
            if not milestone_dir.name.startswith("milestone_"):
                continue
            processed_surface_path = _processed_surface_for_milestone(milestone_dir)
            raw_gridmap_path = _raw_gridmap_for_milestone(milestone_dir)
            if processed_surface_path is None and raw_gridmap_path is None:
                continue
            source_kind = (
                "processed surface"
                if processed_surface_path is not None
                else "raw gridmap"
            )
            shared_metadata_path = _shared_metadata_for_milestone(
                project_dir,
                phase_dir.name,
                milestone_dir.name.removeprefix("milestone_"),
            )
            shared_meta = _load_json(shared_metadata_path)
            scans.append(
                MilestoneScan(
                    phase=phase_dir.name,
                    milestone=milestone_dir.name.removeprefix("milestone_"),
                    milestone_dir=milestone_dir,
                    processed_surface_path=processed_surface_path,
                    raw_gridmap_path=raw_gridmap_path,
                    source_kind=source_kind,
                    shared_metadata_path=shared_metadata_path,
                    post_bedding_baseline=_is_post_bedding_baseline(
                        phase_dir.name,
                        milestone_dir.name.removeprefix("milestone_"),
                        shared_meta,
                    ),
                )
            )

    return sorted(
        scans,
        key=lambda item: (_phase_sort_key(item.phase), _milestone_sort_key(item.milestone)),
    )


def default_baseline_scan(scans: list[MilestoneScan]) -> MilestoneScan | None:
    raw_scans = [scan for scan in scans if scan.raw_gridmap_path is not None]
    if not raw_scans:
        return None

    for scan in raw_scans:
        if scan.post_bedding_baseline:
            return scan
    for scan in raw_scans:
        if scan.phase == "post_bedding_in" and scan.milestone == "0":
            return scan
    return raw_scans[0]


def _processed_surface_for_milestone(milestone_dir: Path) -> Path | None:
    processed_dir = milestone_dir / "processed"

    for pattern in ("*_surface_mesh.ply", "*.ply"):
        matches = sorted(processed_dir.glob(pattern)) if processed_dir.is_dir() else []
        if matches:
            return matches[0]
    return None


def _shared_metadata_for_milestone(
    project_dir: Path | None,
    phase: str,
    milestone: str,
) -> Path | None:
    if project_dir is None:
        return None
    candidate = (
        project_dir
        / "data"
        / "milestones"
        / phase
        / f"milestone_{milestone}"
        / "milestone_metadata.json"
    )
    if candidate.is_file():
        return candidate
    return None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_post_bedding_baseline(
    phase: str,
    milestone: str,
    shared_meta: dict[str, Any],
) -> bool:
    if bool(shared_meta.get("post_bedding_baseline")):
        return True
    return phase == "post_bedding_in" and milestone == "0"


def _raw_gridmap_for_milestone(milestone_dir: Path) -> Path | None:
    raw_dir = milestone_dir / "raw"
    raw_matches = sorted(raw_dir.glob("*.npz")) if raw_dir.is_dir() else []
    if raw_matches:
        return raw_matches[0]
    return None


def _safe_path_segment(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", text).strip("._")


def _phase_sort_key(phase: str) -> tuple[int, str]:
    order = {
        "bedding_in": 0,
        "post_bedding_in": 1,
    }
    return order.get(phase, 99), phase


def _milestone_sort_key(milestone: str) -> tuple[int, float | str]:
    try:
        return 0, float(milestone)
    except ValueError:
        return 1, milestone
