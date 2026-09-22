from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xrd_workbench.io import read_xrdml
from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.models.project import CIF, POLE_DATA, SCAN, ProjectStore
from xrd_workbench.models.radiation import RadiationSettings, validate_radiation_lines
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.project_files import ProjectFileService
from xrd_workbench.version import APP_VERSION


ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = (
    ROOT / "xrd_workbench" / "models",
    ROOT / "xrd_workbench" / "services",
    ROOT / "xrd_workbench" / "io",
    ROOT / "xrd_workbench" / "localization",
)
FORBIDDEN_CORE_IMPORTS = ("tkinter", "PySide6", "matplotlib.backends")
REMOVED_TK_MODULES = (
    "calculated_pattern.py",
    "controls.py",
    "correction.py",
    "experimental_pole.py",
    "i18n.py",
    "main.py",
    "project_panel.py",
    "radiation.py",
    "reflection_table.py",
    "structure_render.py",
    "structure_view.py",
    "substrate_compare.py",
    "theoretical_pole.py",
    "twotheta.py",
)


def imported_names(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append((node.lineno, node.module))
    return result


class ArchitectureTests(unittest.TestCase):
    def test_application_version(self) -> None:
        self.assertEqual(APP_VERSION, "3.0.0b6.1")

    def test_project_objects_can_be_renamed_without_changing_source(self) -> None:
        store = ProjectStore()
        scan = Scan1D(
            "original",
            np.array([10.0, 11.0]),
            np.array([2.0, 3.0]),
            Path("original.xy"),
            axis_name="2Theta",
        )
        document = store.add_scan(scan)
        source = document.source
        events = []
        store.subscribe(lambda event, current, workspace: events.append(event))

        renamed = store.rename(document.uid, "renamed")

        self.assertIs(renamed, document)
        self.assertEqual(document.name, "renamed")
        self.assertEqual(scan.name, "renamed")
        self.assertEqual(document.source, source)
        self.assertEqual(events, ["renamed"])
        with self.assertRaises(ValueError):
            store.rename(document.uid, "   ")

    def test_qt_package_entry_point_is_lazy(self) -> None:
        code = (
            "import sys; import xrd_workbench.ui_qt; "
            "loaded = ('tkinter' in sys.modules or "
            "any(name.startswith('PySide6') for name in sys.modules)); "
            "raise SystemExit(1 if loaded else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0)

    def test_tkinter_interface_is_not_shipped(self) -> None:
        package = ROOT / "xrd_workbench"
        violations = [
            f"{path.relative_to(ROOT)}:{line}: {name}"
            for path in package.rglob("*.py")
            for line, name in imported_names(path)
            if name.startswith("tkinter")
        ]
        self.assertEqual(violations, [])
        self.assertFalse(any((package / "ui_tk").glob("*.py")))
        self.assertTrue(all(not (package / name).exists() for name in REMOVED_TK_MODULES))

    def test_pole_figures_use_the_shared_bruker_raw_parser(self) -> None:
        source = (
            ROOT / "xrd_workbench" / "services" / "experimental_pole.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from ..bruker_raw import", source)
        self.assertNotIn("def read_bruker_raw3", source)
        self.assertNotIn("BrukerRawPoleData", source)

    def test_only_canonical_launcher_is_shipped(self) -> None:
        launcher = (ROOT / "run_xrd_combine.py").read_text(encoding="utf-8")
        self.assertIn("from xrd_workbench.ui_qt import main", launcher)
        self.assertFalse((ROOT / "run_xrd_combine_qt.py").exists())

    def test_core_packages_do_not_import_gui_toolkits(self) -> None:
        violations = []
        for package in CORE_PACKAGES:
            for path in package.rglob("*.py"):
                for line, name in imported_names(path):
                    if name.startswith(FORBIDDEN_CORE_IMPORTS):
                        violations.append(f"{path.name}:{line}: {name}")
        self.assertEqual(violations, [])

    def test_importing_core_packages_does_not_load_gui_toolkits(self) -> None:
        code = (
            "import sys; import xrd_workbench.models; import xrd_workbench.services; "
            "import xrd_workbench.io; import xrd_workbench.localization; "
            "loaded = ('tkinter' in sys.modules or "
            "any(name.startswith('PySide6') for name in sys.modules) or "
            "any(name.startswith('matplotlib') for name in sys.modules)); "
            "raise SystemExit(1 if loaded else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0)

    def test_file_service_uses_injected_readers_and_pure_store(self) -> None:
        cif_payload = SimpleNamespace(name="phase")
        pole_payload = SimpleNamespace(is_pole_figure=True)
        cif_loads: list[Path] = []

        def load_cif(path: Path):
            cif_loads.append(path)
            return cif_payload

        scan = SimpleNamespace(
            name="scan",
            source=Path("scan.raw"),
            metadata={"range_index": 0},
        )
        service = ProjectFileService(
            load_cif_document=load_cif,
            read_bruker_raw=lambda _path: pole_payload,
            read_raw_scans=lambda _path, raw: [scan],
            read_scan_file=lambda _path: [],
        )
        store = ProjectStore()

        cif_document = service.load_path(store, "phase.cif")[0]
        same_cif_document = service.load_path(store, "phase.cif")[0]
        raw_documents = service.load_path(store, "scan.raw")

        self.assertEqual(cif_document.kind, CIF)
        self.assertIs(same_cif_document, cif_document)
        self.assertEqual(len(cif_loads), 1)
        self.assertEqual([item.kind for item in raw_documents], [POLE_DATA, SCAN])
        self.assertEqual(raw_documents[1].parent_uid, raw_documents[0].uid)

    def test_radiation_model_validates_without_localisation(self) -> None:
        settings = RadiationSettings()
        self.assertEqual(settings.lines(), [("Cu Kα1", 1.54056, 1.0)])
        settings.select_custom([("line", 1.0, 0.5)])
        self.assertEqual(settings.lines(), [("line", 1.0, 0.5)])
        with self.assertRaises(ValueError):
            validate_radiation_lines([])
        with self.assertRaises(ValueError):
            validate_radiation_lines([("line", -1.0, 1.0)])

    def test_scan_model_uses_structured_errors(self) -> None:
        with self.assertRaises(XRDDataError) as caught:
            Scan1D("short", np.array([1.0]), np.array([2.0]), Path("short.xy"))
        self.assertEqual(caught.exception.code, "scan_too_short")

    def test_core_xrdml_reader_returns_gui_independent_scan(self) -> None:
        xml = """\
<xrdMeasurements>
  <xrdMeasurement><scan><dataPoints>
    <positions axis="2Theta"><startPosition>20</startPosition><endPosition>22</endPosition></positions>
    <counts>1 2 3</counts>
  </dataPoints></scan></xrdMeasurement>
</xrdMeasurements>
"""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "architecture_test.xrdml"
            source.write_text(xml, encoding="utf-8")
            scan = read_xrdml(source)[0]
        self.assertIs(type(scan), Scan1D)
        np.testing.assert_allclose(scan.x, [20.0, 21.0, 22.0])


if __name__ == "__main__":
    unittest.main()
