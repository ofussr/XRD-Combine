from __future__ import annotations

import importlib.util
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xrd_workbench.models.diffraction import DiffractionStructure
from xrd_workbench.models.project import SCAN, STRUCTURES, VIEWER, ProjectStore
from xrd_workbench.models.scan import Scan1D


ROOT = Path(__file__).resolve().parents[1]
PYSIDE_INSTALLED = importlib.util.find_spec("PySide6") is not None
PYSIDE_AVAILABLE = PYSIDE_INSTALLED
QApplication = None
if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except (ImportError, OSError):
        PYSIDE_AVAILABLE = False


class QtBootstrapTests(unittest.TestCase):
    def test_unit_cell_gui_configures_opengl_before_qapplication(self) -> None:
        source = (ROOT / "xrd_workbench" / "ui_qt" / "app.py").read_text(
            encoding="utf-8"
        )
        configure_call = source.index("configure_qt_opengl()")
        application_call = source.index("QApplication.instance()")
        self.assertLess(configure_call, application_call)

    def test_project_tree_refresh_is_deferred_outside_item_changed(self) -> None:
        source = (ROOT / "xrd_workbench" / "ui_qt" / "project_panel.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self._refresh_timer = QTimer(self)", source)
        self.assertIn("self._refresh_timer.timeout.connect(self.refresh)", source)
        self.assertIn("def _schedule_refresh(self)", source)
        self.assertIn("self._refresh_timer.start(0)", source)

        store_event = source[source.index("def _store_event"):source.index("def open_files")]
        self.assertIn("self._schedule_refresh()", store_event)
        self.assertNotIn("self.refresh()", store_event)

    def test_viewer_page_uses_qt_canvas_and_shared_state(self) -> None:
        source = (ROOT / "xrd_workbench" / "ui_qt" / "viewer_page.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FigureCanvasQTAgg", source)
        self.assertIn("ViewerState", source)
        self.assertIn("clone_scan", source)
        self.assertIn("RadiationSelector", source)
        self.assertIn("calculate_reflections", source)
        self.assertIn("CorrectionRequest", source)
        self.assertIn("apply_correction", source)
        self.assertIn("fit_gaussian_peak", source)
        self.assertIn("write_processed_scan", source)
        self.assertIn("# display_form.addRow(self.offset_label, self.offset_spin)", source)
        self.assertIn("DISABLED_PLACEHOLDER_STYLE", source)
        self.assertIn("self.phase_axis.set_navigate(separate)", source)
        self.assertIn("PyQtGraphViewerPlot", source)
        self.assertIn("def _full_hkl", source)
        self.assertIn("SubstrateCorrectionDialog", source)
        self.assertIn("fit_substrate_calibration", source)
        self.assertIn("two_theta_to_d", source)
        self.assertIn("display_wavelength_combo", source)
        self.assertIn("ScrollBarAlwaysOff", source)
        self.assertNotIn("phase_x_min_edit", source)
        self.assertNotIn("phase_x_max_edit", source)
        self.assertNotIn("backend_tkagg", source)

    def test_comparison_page_uses_qt_canvas_and_shared_services(self) -> None:
        source = (ROOT / "xrd_workbench" / "ui_qt" / "comparison.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FigureCanvasQTAgg", source)
        self.assertIn("PyQtGraphViewerPlot", source)
        self.assertIn("ComparisonWorkspace", source)
        self.assertIn("prepare_comparison_plot", source)
        self.assertIn("plot_renderer_controller", source)
        self.assertIn("toggle_y_mode", source)
        self.assertIn("ApplicationModal", source)
        self.assertIn("ScrollBarAlwaysOff", source)
        self.assertNotIn("backend_tkagg", source)
        self.assertNotIn("tkinter", source)

    def test_structures_page_uses_qt_canvas_and_shared_diffraction_service(self) -> None:
        source = (
            ROOT / "xrd_workbench" / "ui_qt" / "structures_page.py"
        ).read_text(encoding="utf-8")
        self.assertIn("FigureCanvasQTAgg", source)
        self.assertIn("PyQtGraphViewerPlot", source)
        self.assertIn("plot_renderer_controller", source)
        self.assertIn("calculate_reflections", source)
        self.assertIn("gaussian_powder_profile", source)
        self.assertIn("write_reflection_csv", source)
        self.assertIn("RadiationSelector", source)
        self.assertIn("def _full_hkl", source)
        self.assertNotIn("backend_tkagg", source)
        self.assertNotIn("tkinter", source)

    def test_structure_viewer_uses_unit_cell_gui_and_shared_scene_geometry(self) -> None:
        viewer = (ROOT / "xrd_workbench" / "ui_qt" / "structure_viewer.py").read_text(
            encoding="utf-8"
        )
        adapter = (ROOT / "xrd_workbench" / "ui_qt" / "unit_cell_adapter.py").read_text(
            encoding="utf-8"
        )
        geometry = (
            ROOT / "xrd_workbench" / "services" / "structure_scene.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from unit_cell_gui import CrystalCanvas", viewer)
        self.assertIn("build_structure_scene", viewer)
        self.assertIn("unit_cell_scene", viewer)
        self.assertIn("CollapsibleSection", viewer)
        self.assertIn("ScrollBarAlwaysOff", viewer)
        self.assertIn("DisplayOptions", viewer)
        self.assertIn("StyleOverrides", viewer)
        self.assertIn("Scene", adapter)
        self.assertIn("AtomComponent", adapter)
        self.assertIn("basis_vectors=source.crystal.direct", adapter)
        self.assertIn("atom_ball_radius", adapter)
        self.assertIn("default_polyhedron_elements", geometry)
        self.assertIn("crystallographic_sites", geometry)
        self.assertIn("SceneAtom", geometry)
        self.assertNotIn("propagated_hatch_directions", geometry)
        self.assertNotIn("face_hatch_segments", geometry)
        self.assertFalse(
            (ROOT / "xrd_workbench" / "ui_qt" / "structure_canvas.py").exists()
        )
        self.assertIn(
            "display_section.content_layout.addWidget(self.polyhedra_section)",
            viewer,
        )
        self.assertIn(
            "display_section.content_layout.addWidget(self.atoms_section)",
            viewer,
        )
        self.assertNotIn("tkinter", viewer + adapter + geometry)
        self.assertNotIn("PySide6", geometry)
        structures = (
            ROOT / "xrd_workbench" / "ui_qt" / "structures_page.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("structure_placeholder", structures)

    def test_qt_modules_import_every_used_layout(self) -> None:
        for filename in (
            "main_window.py",
            "viewer_page.py",
            "radiation.py",
            "comparison.py",
            "structures_page.py",
            "structure_viewer.py",
        ):
            with self.subTest(filename=filename):
                source = (ROOT / "xrd_workbench" / "ui_qt" / filename).read_text(
                    encoding="utf-8"
                )
                tree = ast.parse(source)
                imported = {
                    alias.asname or alias.name
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom)
                    for alias in node.names
                }
                used_layouts = {
                    node.func.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id.endswith("Layout")
                }
                self.assertLessEqual(used_layouts, imported)

    def test_version_is_available_without_pyside6(self) -> None:
        result = subprocess.run(
            [sys.executable, "run_xrd_combine.py", "--version"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3.0.0b6.1")

    def test_command_line_accepts_paths_with_spaces(self) -> None:
        from xrd_workbench.ui_qt.app import _parser

        arguments = _parser().parse_args(
            [r"C:\XRD data\measurement.xrdml", r"C:\XRD data\phase.cif"]
        )
        self.assertEqual(
            arguments.paths,
            [r"C:\XRD data\measurement.xrdml", r"C:\XRD data\phase.cif"],
        )


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 QtWidgets runtime is unavailable")
class QtWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_project_drawer_and_workspace_assignments(self) -> None:
        from xrd_workbench.ui_qt.main_window import MainWindow

        store = ProjectStore()
        window = MainWindow(store=store)
        window.show()
        self.application.processEvents()

        scan = Scan1D(
            name="test scan",
            x=[10.0, 11.0],
            y=[20.0, 21.0],
            source=Path("test.xy"),
            axis_name="2Theta",
        )
        document = store.add_scan(scan)
        store.assign(document.uid, VIEWER, True)
        self.application.processEvents()
        self.assertEqual(window.pages[VIEWER].documents.count(), 1)

        phase_payload = SimpleNamespace(
            name="test cell",
            source=Path("test-cell.cell"),
            diffraction=DiffractionStructure(
                name="test cell",
                cell=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
                atoms=[],
                symmetry_operations=["x,y,z"],
                cell_only=True,
                space_group="P 1",
            ),
        )
        phase = store.add_cell_phase(phase_payload)
        store.assign(phase.uid, VIEWER, True)
        self.application.processEvents()
        viewer = window.pages[VIEWER]
        self.assertEqual(viewer.documents.count(), 2)
        self.assertEqual(viewer.plot_renderer, "pyqtgraph")
        self.assertEqual(viewer.viewer_state.plot.intensity_scale, "log")
        self.assertEqual(viewer.viewer_state.plot.phase_layout, "overlay")
        self.assertFalse(viewer.phase_section.isHidden())
        self.assertFalse(viewer.pyqtgraph_plot.phase_visible)

        window.toggle_project_panel()
        self.application.processEvents()
        self.assertTrue(window.project_panel.isHidden())
        self.assertFalse(window.project_rail.isHidden())
        window.toggle_project_panel()
        self.assertFalse(window.project_panel.isHidden())
        self.assertTrue(window.project_rail.isHidden())

        window.sections.setCurrentIndex(1)
        self.application.processEvents()
        self.assertEqual(window.current_workspace(), STRUCTURES)
        self.assertFalse(store.compatible(SCAN, STRUCTURES))
        window.close()

    def test_viewer_correction_adds_a_shifted_scan_and_preserves_view(self) -> None:
        from xrd_workbench.ui_qt.main_window import MainWindow

        store = ProjectStore()
        window = MainWindow(store=store)
        scan = Scan1D(
            name="correction test",
            x=[10.0, 11.0, 12.0],
            y=[20.0, 50.0, 25.0],
            source=Path("correction-test.xy"),
            axis_name="2Theta",
        )
        document = store.add_scan(scan)
        store.assign(document.uid, VIEWER, True)
        self.application.processEvents()

        viewer = window.pages[VIEWER]
        viewer.documents.setCurrentItem(viewer.documents.topLevelItem(0))
        self.application.processEvents()
        viewer.processing_x_spin.setValue(0.25)
        viewer.processing_y_spin.setValue(-2.0)
        viewer.processing_factor_spin.setValue(2.0)
        self.application.processEvents()
        viewer.pyqtgraph_plot.set_scan_range((10.4, 11.4), (25.0, 80.0))
        viewer.apply_processing_result()
        self.application.processEvents()

        self.assertEqual(len(store.documents), 2)
        derived = list(store.documents.values())[-1]
        self.assertEqual(derived.parent_uid, document.uid)
        self.assertTrue(derived.name.endswith(" shifted"))
        self.assertAlmostEqual(float(derived.payload.x[0]), 10.25)
        self.assertAlmostEqual(float(derived.payload.y[0]), 38.0)
        self.assertEqual(viewer.items[document.uid].x_shift, 0.0)
        self.assertAlmostEqual(viewer.pyqtgraph_plot.scan_limits()[0][0], 10.4)
        self.assertAlmostEqual(viewer.pyqtgraph_plot.scan_limits()[0][1], 11.4)
        window.close()

    def test_viewer_commands_and_single_modal_comparison(self) -> None:
        from xrd_workbench.ui_qt.main_window import MainWindow

        store = ProjectStore()
        window = MainWindow(store=store)
        documents = []
        for index in range(2):
            scan = Scan1D(
                name=f"comparison {index + 1}",
                x=[10.0, 11.0, 12.0],
                y=[20.0, 30.0, 25.0],
                source=Path(f"comparison-{index + 1}.xy"),
                axis_name="2Theta",
            )
            document = store.add_scan(scan)
            store.assign(document.uid, VIEWER, True)
            documents.append(document)
        self.application.processEvents()

        viewer = window.pages[VIEWER]
        viewer.viewer_state.items[documents[0].uid].visible = False
        viewer.show_all()
        self.assertTrue(
            all(item.visible for item in viewer.viewer_state.items.values())
        )

        viewer.select_uid(documents[0].uid)
        with patch(
            "xrd_workbench.ui_qt.viewer_page.QInputDialog.getText",
            return_value=("renamed scan", True),
        ):
            viewer.rename_selected()
        self.assertEqual(store.documents[documents[0].uid].name, "renamed scan")
        self.assertEqual(store.documents[documents[0].uid].payload.name, "renamed scan")

        viewer.remove_selected()
        self.assertFalse(store.is_assigned(documents[0].uid, VIEWER))
        self.assertIn(documents[0].uid, store.documents)

        self.assertTrue(viewer.comparison_button.isEnabled())
        window.open_comparison()
        self.application.processEvents()
        comparison = window.comparison_dialog
        self.assertIsNotNone(comparison)
        self.assertTrue(comparison.isModal())
        self.assertEqual(len(comparison.page.items), 1)
        window.open_comparison()
        self.assertIs(window.comparison_dialog, comparison)
        comparison.close()
        self.application.processEvents()
        self.assertIsNone(window.comparison_dialog)

        viewer.clear_all()
        self.assertFalse(store.assigned_documents(VIEWER))
        self.assertEqual(len(store.documents), 2)
        window.close()

    def test_structures_calculation_table_and_viewer_shortcut(self) -> None:
        from xrd_workbench.ui_qt.main_window import MainWindow

        store = ProjectStore()
        window = MainWindow(store=store)
        payload = SimpleNamespace(
            name="cell phase",
            source=Path("cell-phase.cell"),
            diffraction=DiffractionStructure(
                name="cell phase",
                cell=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
                atoms=[],
                symmetry_operations=["x,y,z"],
                cell_only=True,
                space_group="P 1",
            ),
        )
        document = store.add_cell_phase(payload)
        store.assign(document.uid, STRUCTURES, True)
        self.application.processEvents()

        structures = window.pages[STRUCTURES]
        self.assertEqual(structures.active_uid, document.uid)
        self.assertTrue(structures.calculated_pattern.rows)
        self.assertTrue(structures.calculated_pattern.sticks_radio.isChecked())
        self.assertEqual(
            structures.reflection_table.table.topLevelItemCount(),
            len(structures.reflection_table.rows),
        )
        self.assertTrue(
            all(row.intensity is None for row in structures.reflection_table.rows)
        )
        first = structures.reflection_table.table.topLevelItem(0)
        structures.reflection_table.table.setCurrentItem(first)
        self.application.processEvents()
        selected_row = structures.reflection_table.rows[
            int(first.data(0, Qt.ItemDataRole.UserRole))
        ]
        for h, k, l in selected_row.equivalents:
            self.assertIn(
                f"({h} {k} {l})",
                structures.reflection_table.details.toPlainText(),
            )

        viewer = window.pages[VIEWER]
        radiation_index = viewer.radiation_selector.combo.findData("cu_ka12")
        viewer.radiation_selector.combo.setCurrentIndex(radiation_index)
        self.application.processEvents()
        self.assertEqual(
            structures.radiation_selector.combo.currentData(),
            "cu_ka12",
        )
        self.assertEqual(
            {row.radiation for row in structures.reflection_table.rows},
            {"Cu Kα1", "Cu Kα2"},
        )

        store.assign(document.uid, VIEWER, True)
        self.application.processEvents()
        viewer.select_uid(document.uid)
        self.assertTrue(viewer.table_button.isEnabled())
        viewer.table_button.click()
        self.application.processEvents()
        self.assertEqual(window.current_workspace(), STRUCTURES)
        self.assertEqual(structures.tabs.currentIndex(), 2)
        window.close()

    def test_native_structure_viewer_loads_sites_polyhedra_and_controls(self) -> None:
        from xrd_workbench.cif_document import load_cif_document
        from xrd_workbench.ui_qt.main_window import MainWindow

        cif = """data_qt_structure
_chemical_formula_sum 'Nb0.4 Ta0.6 O3'
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Nb1 Nb 0.5 0.5 0.5 0.4
Ta1 Ta 0.5 0.5 0.5 0.6
O1 O 0.25 0.5 0.5 1
O2 O 0.75 0.5 0.5 1
O3 O 0.5 0.25 0.5 1
O4 O 0.5 0.75 0.5 1
O5 O 0.5 0.5 0.25 1
O6 O 0.5 0.5 0.75 1
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mixed.cif"
            path.write_text(cif, encoding="utf-8")
            payload = load_cif_document(path)
            store = ProjectStore()
            document = store.add_cif_document(path, payload)
            window = MainWindow(store=store)
            store.assign(document.uid, STRUCTURES, True)
            window.sections.setCurrentIndex(1)
            self.application.processEvents()

            page = window.pages[STRUCTURES].structure_viewer
            self.assertIs(page.crystal, payload.crystal)
            self.assertIs(page.canvas.scene, page.scene)
            self.assertFalse(page.orientation_section.content.isVisible())
            self.assertFalse(page.display_section.content.isVisible())
            self.assertFalse(page.atoms_section.content.isVisible())
            self.assertFalse(page.polyhedra_section.content.isVisible())
            self.assertEqual(page.polyhedron_tree.topLevelItemCount(), 1)
            centre_group = page.polyhedron_tree.topLevelItem(0)
            self.assertEqual(centre_group.text(0), "Nb/Ta")
            self.assertEqual(centre_group.childCount(), 1)
            self.assertEqual(centre_group.child(0).text(0), "Nb1/Ta1")
            self.assertEqual(
                {page.atom_tree.topLevelItem(index).text(0) for index in range(3)},
                {"Nb", "O", "Ta"},
            )
            self.assertEqual(len(page.scene.polyhedra), 1)
            self.assertEqual(page.scene.polyhedra[0].coordination_number, 6)

            page.absolute_inputs[0].setValue(25.0)
            page.apply_exact_rotation()
            self.assertAlmostEqual(page.absolute_inputs[0].value(), 25.0)
            self.assertFalse(page.polyhedra_check.isChecked())
            self.assertFalse(page.canvas.show_polyhedra)
            page.polyhedra_check.setChecked(True)
            page.style_combo.setCurrentIndex(1)
            self.assertTrue(page.hatching_check.isEnabled())
            self.assertTrue(page.grip_slider.isEnabled())
            polyhedron_row = next(iter(page.polyhedron_site_rows.values()))
            self.assertFalse(polyhedron_row[1].isEnabled())
            self.assertFalse(polyhedron_row[2].isEnabled())
            page.grip_slider.setValue(125)
            self.assertEqual(page.canvas.hatch_grip, 125)
            page.atom_size_slider.setValue(150)
            self.assertAlmostEqual(page.canvas.atom_scale, 1.5)
            self.assertTrue(page.external_atoms_check.isChecked())
            page.atoms_check.setChecked(False)
            self.assertFalse(page.external_atoms_check.isEnabled())
            self.assertTrue(page.basis_check.isEnabled())
            window.close()


if __name__ == "__main__":
    unittest.main()
