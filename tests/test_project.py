from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from alf_scan_viewer.project import (
    SURFACE_VIEW_GRAPH_TYPE,
    default_baseline_scan,
    discover_milestone_scans,
    resolve_surface_scanner_dir,
    surface_scanner_graph_dir,
)


class ProjectDiscoveryTests(unittest.TestCase):
    def test_resolves_project_root_to_surface_scanner_dir(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            surface_dir = project / "data" / "tools" / "surface_scanner"
            (surface_dir / "milestones").mkdir(parents=True)

            self.assertEqual(resolve_surface_scanner_dir(project), surface_dir)
            self.assertEqual(resolve_surface_scanner_dir(surface_dir), surface_dir)

    def test_resolves_surface_scanner_graph_export_dir(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            surface_dir = project / "data" / "tools" / "surface_scanner"
            (surface_dir / "milestones").mkdir(parents=True)

            self.assertEqual(
                surface_scanner_graph_dir(project),
                surface_dir / "graphs" / SURFACE_VIEW_GRAPH_TYPE,
            )

    def test_discovers_processed_surface_mesh_for_milestone(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            processed = (
                project
                / "data"
                / "tools"
                / "surface_scanner"
                / "milestones"
                / "post_bedding_in"
                / "milestone_1000"
                / "processed"
            )
            processed.mkdir(parents=True)
            mesh_path = processed / "project_post_bedding_in_1000_surface_mesh.ply"
            mesh_path.write_text("ply\n", encoding="utf-8")

            scans = discover_milestone_scans(project)

            self.assertEqual(len(scans), 1)
            self.assertEqual(scans[0].phase, "post_bedding_in")
            self.assertEqual(scans[0].milestone, "1000")
            self.assertEqual(scans[0].scan_path, mesh_path)
            self.assertEqual(scans[0].source_kind, "processed surface")

    def test_falls_back_to_raw_gridmap_when_processed_surface_is_missing(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            raw = (
                project
                / "data"
                / "tools"
                / "surface_scanner"
                / "milestones"
                / "bedding_in"
                / "milestone_0"
                / "raw"
            )
            raw.mkdir(parents=True)
            gridmap_path = raw / "scan.npz"
            gridmap_path.write_bytes(b"placeholder")

            scans = discover_milestone_scans(project)

            self.assertEqual(len(scans), 1)
            self.assertEqual(scans[0].scan_path, gridmap_path)
            self.assertEqual(scans[0].source_kind, "raw gridmap")

    def test_sorts_numeric_milestones_numerically(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            milestones_root = (
                project / "data" / "tools" / "surface_scanner" / "milestones"
            )
            for milestone in ("10", "2"):
                processed = (
                    milestones_root
                    / "post_bedding_in"
                    / f"milestone_{milestone}"
                    / "processed"
                )
                processed.mkdir(parents=True)
                (processed / f"scan_{milestone}_surface_mesh.ply").write_text(
                    "ply\n",
                    encoding="utf-8",
                )

            scans = discover_milestone_scans(project)

            self.assertEqual([scan.milestone for scan in scans], ["2", "10"])

    def test_detects_default_baseline_from_shared_milestone_metadata(self) -> None:
        with TemporaryDirectory() as td:
            project = Path(td)
            raw = (
                project
                / "data"
                / "tools"
                / "surface_scanner"
                / "milestones"
                / "bedding_in"
                / "milestone_500"
                / "raw"
            )
            raw.mkdir(parents=True)
            (raw / "scan.npz").write_bytes(b"placeholder")

            shared = (
                project
                / "data"
                / "milestones"
                / "bedding_in"
                / "milestone_500"
            )
            shared.mkdir(parents=True)
            (shared / "milestone_metadata.json").write_text(
                json.dumps(
                    {
                        "phase": "bedding_in",
                        "milestone_cycle_count": "500",
                        "post_bedding_baseline": True,
                    }
                ),
                encoding="utf-8",
            )

            scans = discover_milestone_scans(project)
            baseline = default_baseline_scan(scans)

            self.assertEqual(len(scans), 1)
            self.assertTrue(scans[0].post_bedding_baseline)
            self.assertIsNotNone(baseline)
            self.assertEqual(baseline.milestone, "500")


if __name__ == "__main__":
    unittest.main()
