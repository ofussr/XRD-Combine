"""RSM geometry, missing samples, project integration, and Qt rendering."""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from xrd_workbench.bruker_raw import BrukerRawFile, RawRange, ScanPath
from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.models.project import RSM, RSM_DATA, SCAN
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.rsm import (
    angle_series, angles_to_q, build_raw_map, build_scan_map,
    display_mesh, make_orientation, map_view_bounds, measured_reflections,
    q_to_angles, reflection_point,
)


def raw_range(index, outer, values):
    data = np.asarray(values, dtype=float).reshape(-1, 1)
    return RawRange(
        index=index, scan_type="Theta", axis_name="Theta", start_angle=21.,
        step_size=.1, time_per_step=1.,
        drives={"Theta": 21., "2Theta": outer}, data=data,
        channel_names=["intensity"],
        scan_path=ScanPath("Theta", {"Theta": .1}, "explicit"),
    )


def raw_file(ranges):
    return BrukerRawFile(Path("rsm.raw"), 3, "RAW1.01", "", "", {}, ranges)


class RSMGeometryTests(unittest.TestCase):
    def test_interrupted_map_preserves_empty_and_partial_ranges(self):
        data = build_raw_map([raw_file([
            raw_range(0, 44., [10., 20., 30.]),
            raw_range(1, 45., [40., 50.]),
            raw_range(2, 46., []),
        ])])
        np.testing.assert_allclose(data.two_theta, [44., 45., 46.])
        np.testing.assert_allclose(data.omega, [21., 21.1, 21.2])
        self.assertEqual(data.intensity.shape, (3, 3))
        self.assertTrue(np.isnan(data.intensity[2, 1]))
        self.assertTrue(np.isnan(data.intensity[:, 2]).all())
        x, y, z, _, _ = display_mesh(data, "q", 1.5406)
        self.assertEqual(x.shape, (4, 4))
        self.assertEqual(y.shape, (4, 4))
        self.assertTrue(np.isnan(z[:, 2]).all())
        full = map_view_bounds(data, "angular", 1.5406, False)
        real = map_view_bounds(data, "angular", 1.5406, True)
        self.assertLess(real[3], full[3])
        q_full = map_view_bounds(data, "q", 1.5406, False)
        q_real = map_view_bounds(data, "q", 1.5406, True)
        self.assertLess(q_real[3], q_full[3])

    def test_map_from_two_theta_scans_uses_fixed_omega(self):
        first = raw_range(0, 23., [1., 2., 3.])
        second = raw_range(1, 23., [4., 5., 6.])
        for index, item in enumerate((first, second)):
            item.drives["Omega"] = 12. + index
            item.drives["2Theta"] = 23.
            item.scan_path = ScanPath("2Theta", {"2Theta": .1}, "explicit")
        data = build_raw_map([raw_file([first, second])])
        np.testing.assert_allclose(data.two_theta, [23., 23.1, 23.2])
        np.testing.assert_allclose(data.omega, [12., 13.])
        np.testing.assert_allclose(data.intensity[1], [4., 5., 6.])

    def test_scan_angles_and_q_round_trip(self):
        np.testing.assert_allclose(angle_series(3, 20., None, 22.), [20., 21., 22.])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            angle_series(3, 20., 1., 25.)
        qx, qz = angles_to_q(22., 45., 1.5406)
        omega, two_theta = q_to_angles(qx, qz, 1.5406)
        self.assertAlmostEqual(omega, 22.)
        self.assertAlmostEqual(two_theta, 45.)

    def test_xy_series_declares_x_as_two_theta_without_mutating_shared_scans(self):
        scans = [Scan1D(str(i), np.array([40., 40.1, 40.2]),
                        np.array([i + 1., i + 2., i + 3.]),
                        Path(f"{i}.xy"), axis_name="X", metadata={"format": "XY"})
                 for i in range(3)]
        data = build_scan_map(scans, 20., .1, None)
        np.testing.assert_allclose(data.omega, [20., 20.1, 20.2])
        np.testing.assert_allclose(data.intensity[2], [3., 4., 5.])
        self.assertTrue(all(scan.axis_name == "X" for scan in scans))

    def test_cif_uses_application_crystal_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cif = Path(directory) / "cubic.cif"
            cif.write_text(
                "data_cubic\n_cell_length_a 4\n_cell_length_b 4\n"
                "_cell_length_c 4\n_cell_angle_alpha 90\n"
                "_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                "_space_group_name_H-M_alt 'P 1'\n", encoding="utf-8")
            crystal = load_cif_document(cif).crystal
            orientation = make_orientation(crystal, (0, 0, 1), (1, 0, 0))
            point = reflection_point(crystal, orientation, (0, 0, 2), 1.5406)
            self.assertLess(abs(point.qy), 1e-10)
            self.assertTrue(math.isfinite(point.two_theta))
            measured = build_raw_map([raw_file([
                raw_range(0, 45.2, [10., 20., 30.]),
                raw_range(1, 45.3, [40., 50., 60.]),
                raw_range(2, 45.4, []),
            ])])
            # The synthetic grid is centered at omega 21 degrees, outside (002).
            self.assertEqual(measured_reflections(measured, [point]), [])
            measured.omega[:] = [22.6, 22.7, 22.8]
            self.assertEqual(measured_reflections(measured, [point]), [point])


class RSMQtTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("PySide6") is not None, "PySide6 unavailable")
    def test_project_raw_import_and_both_rsm_views(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from xrd_workbench.ui_qt.main_window import MainWindow
        from test_bruker_raw_geometry import make_v3_range, write_v3

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "map.raw"
            write_v3(source, [
                make_v3_range(axis_code=3, theta=22.6, two_theta=45.2 + .1 * index,
                              values=(1., 2., 3.) if index != 2 else ())
                for index in range(3)
            ])
            window = MainWindow()
            try:
                window.sections.setCurrentIndex(3)
                documents = window.import_paths([str(source)])
                self.assertEqual(sum(item.kind == RSM_DATA for item in documents), 1)
                self.assertTrue(window.project.assigned_documents(RSM, kind=RSM_DATA))
                page = window.pages[RSM]
                self.assertEqual(page.experimental.scale.currentData(), "linear")
                self.assertEqual(page.experimental.extent.currentData(), "real")
                self.assertEqual(page.experimental.data.intensity.shape, (3, 3))
                self.assertTrue(np.isnan(page.experimental.data.intensity[-1, -1]))
                self.assertEqual(page.experimental.first.text(), "22.6")
                self.assertEqual(page.experimental.step.text(), "0.1")
                self.assertEqual(page.experimental.last.text(), "22.8")
                self.assertTrue(page.experimental.first.isReadOnly())
                self.assertEqual(len(page.experimental.plot.getPlotItem().items), 1)
                plot_range = page.experimental.plot.getViewBox().viewRange()
                event = Mock()
                page.experimental.plot.getViewBox().wheelEvent(event)
                event.accept.assert_called_once()
                np.testing.assert_allclose(page.experimental.plot.getViewBox().viewRange(), plot_range)
                page.calculated.plot.getViewBox().wheelEvent(event)
                self.assertEqual(event.accept.call_count, 2)
                left_drag = Mock()
                left_drag.button.return_value = Qt.MouseButton.LeftButton
                for surface in (page.experimental.plot, page.calculated.plot):
                    surface.getViewBox().mouseDragEvent(left_drag)
                self.assertEqual(left_drag.accept.call_count, 2)
                np.testing.assert_allclose(page.experimental.plot.getViewBox().viewRange(), plot_range)
                real_range = page.experimental.plot.getViewBox().viewRange()
                page.experimental.extent.setCurrentIndex(1)
                full_range = page.experimental.plot.getViewBox().viewRange()
                self.assertGreater(full_range[1][1], real_range[1][1])
                page.experimental.view.setCurrentIndex(1)
                app.processEvents()
                self.assertEqual(len(page.experimental.plot.getPlotItem().items), 1)
                self.assertIn("3 ranges", page.experimental.status.text())

                cif = Path(directory) / "cubic.cif"
                cif.write_text(
                    "data_cubic\n_cell_length_a 4\n_cell_length_b 4\n"
                    "_cell_length_c 4\n_cell_angle_alpha 90\n"
                    "_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                    "_space_group_name_H-M_alt 'P 1'\n", encoding="utf-8")
                crystal_doc = window.file_service.load_cif(window.project, cif)
                window.project.assign(crystal_doc.uid, RSM, True)
                self.assertEqual(page.experimental.cif_combo.currentData(), crystal_doc.uid)
                page.experimental.add_cif_points()
                self.assertIn("CIF reflections", page.experimental.status.text())
                self.assertGreater(len(page.experimental.plot.getPlotItem().items), 1)
                page.experimental.view.setCurrentIndex(0)
                self.assertGreater(len(page.experimental.plot.getPlotItem().items), 1)
                other_cif = Path(directory) / "other.cif"
                other_cif.write_text(cif.read_text(encoding="utf-8").replace(
                    "_cell_length_a 4", "_cell_length_a 5").replace(
                    "_cell_length_b 4", "_cell_length_b 5").replace(
                    "_cell_length_c 4", "_cell_length_c 5"), encoding="utf-8")
                other_doc = window.file_service.load_cif(window.project, other_cif)
                page.experimental.cif_combo.setCurrentIndex(
                    page.experimental.cif_combo.findData(other_doc.uid))
                self.assertIn("0 CIF reflections", page.experimental.status.text())
                page.experimental.cif_combo.setCurrentIndex(
                    page.experimental.cif_combo.findData(crystal_doc.uid))
                self.assertGreater(len(page.experimental.plot.getPlotItem().items), 1)
                page.experimental.remove_cif_points()
                self.assertEqual(len(page.experimental.plot.getPlotItem().items), 1)
                page.calculated.draw()
                self.assertIn("Omega=", page.calculated.status.text())
                page.calculated.target.setCurrentIndex(1)
                page.calculated.target_a.setText("22,5")
                page.calculated.retranslate()
                self.assertEqual(page.calculated.target_a.text(), "22,5")
                page.calculated.view.setCurrentIndex(1)
                self.assertEqual(page.calculated.span_x.text(), "0.10")
                for document in list(window.project.assigned_documents(RSM)):
                    if document.kind in (RSM_DATA, SCAN):
                        window.project.assign(document.uid, RSM, False)
                self.assertFalse(page.experimental.first.isReadOnly())
                page.experimental.first.setText("20")
                page.experimental.step.setText("0.1")
                for index in range(3):
                    scan = Scan1D(f"xy_{index}", np.array([45.2, 45.3, 45.4]),
                                  np.array([1., 2., 3.]), Path(directory) / f"{index}.xy",
                                  axis_name="X", metadata={"format": "XY"})
                    document = window.project.add_scan(scan)
                    window.project.assign(document.uid, RSM, True)
                self.assertEqual(page.experimental.last.text(), "20.2")
                self.assertEqual(page.experimental.data.intensity.shape, (3, 3))
                page.experimental.first.setText("21")
                page.experimental.first.textEdited.emit("21")
                self.assertEqual(page.experimental.last.text(), "")
                page.experimental.draw()
                self.assertEqual(page.experimental.last.text(), "21.2")
                page.experimental.add_cif_points()
                self.assertTrue(page.experimental._overlay_requested)
                window.project.remove(crystal_doc.uid)
                self.assertFalse(page.experimental._overlay_requested)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
