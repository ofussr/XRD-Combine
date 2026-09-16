from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backend_bases import MouseEvent

from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.cell_phase import create_cell_phase_document
from xrd_workbench.space_groups import SETTINGS
from xrd_workbench.models.project import POLES, POLE_DATA, ProjectStore
from xrd_workbench.models.radiation import RadiationSettings
from xrd_workbench.services.project_files import ProjectFileService
from xrd_workbench.services.experimental_pole import load_experimental_pole, split_continuous_segments
from xrd_workbench.services.pole_figure import euler_matrix, project_reflections, pole_display_position
from xrd_workbench.localization import set_language, tr
from pole_controller_harness import Calculated, Experimental, workspace

ROOT = Path(__file__).resolve().parents[1]
CIF_TEXT = """data_Si
_chemical_formula_sum 'Si'
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_it_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
"""


def phase(name='phase', cell=(4., 5., 6., 90., 90., 90.)):
    return create_cell_phase_document(name, SETTINGS[0], cell)


def raw_fixture():
    return SimpleNamespace(is_pole_figure=True, ranges=[
        SimpleNamespace(chi=40., phi=np.array([0., 1., 2., 10., 11., 12.]), intensity=np.array([11., 12., 13., 14., 15., 16.])),
        SimpleNamespace(chi=10., phi=np.array([360., 270., 180., 90., 0.]), intensity=np.array([101., 4., 3., 2., 1.])),
        SimpleNamespace(chi=24., phi=np.array([]), intensity=np.array([])),
    ])


class ExperimentalMigrationTests(unittest.TestCase):
    def setUp(self):
        set_language('en')
        self.page = Experimental()

    def test_raw_grids_keep_irregular_chi_empty_ranges_and_phi_gaps(self):
        measurement = load_experimental_pole('scan.raw', raw_fixture())
        self.page.load_measurement(measurement)
        np.testing.assert_array_equal(self.page.displayed_radii, [10., 24., 40.])
        np.testing.assert_array_equal(measurement.scans[0][0], [0., 90., 180., 270.])
        np.testing.assert_array_equal(measurement.scans[0][1], [1., 2., 3., 4.])
        self.assertEqual(len(split_continuous_segments(*measurement.scans[2])), 2)
        # One low-Chi mesh, no mesh at the empty ring, two separated high-Chi meshes.
        self.assertEqual(len(self.page.plot_axis.collections), 3)
        self.assertEqual(sum(mesh.get_array().size for mesh in self.page.plot_axis.collections), 10)
        self.assertEqual(self.page.angle_step.get(), tr('text.variable'))
        self.assertTrue(self.page.save_button.enabled)

    def test_click_reports_measured_intensity_and_leaves_gap_unmeasured(self):
        self.page.load_measurement(load_experimental_pole('scan.raw', raw_fixture()))
        for phi, expected in [(1., '12'), (6., None)]:
            x, y = self.page.plot_axis.transData.transform((math.radians(phi), 40.))
            event = MouseEvent('button_press_event', self.page.canvas, x, y, button=1)
            self.page.on_plot_click(event)
            message = self.page.cursor_text.get()
            if expected:
                self.assertIn('Intensity: ' + expected, message)
            else:
                self.assertIn('no measured', message.lower())

    def test_manual_angles_and_zoom_preserve_samples(self):
        self.page.load_measurement(load_experimental_pole('scan.raw', raw_fixture()))
        self.page.first_angle.set('5')
        self.page.angle_step.set('10')
        self.page.draw_pole_figure()
        np.testing.assert_array_equal(self.page.displayed_radii, [5., 15., 25.])
        before = self.page.displayed_radii.copy()
        limits = self.page.plot_axis.get_ylim()
        x, y = self.page.plot_axis.transData.transform((0.5, 15.))
        event = MouseEvent('scroll_event', self.page.canvas, x, y, button='up', step=1)
        self.page.on_plot_scroll(event)
        self.assertGreater(self.page.zoom_factor, 1)
        self.assertEqual(self.page.plot_axis.get_ylim(), limits)
        np.testing.assert_array_equal(self.page.displayed_radii, before)
        self.page.reset_zoom()
        self.assertEqual(self.page.zoom_factor, 1)

    def test_xy_fallback_keeps_number_order_and_requires_tilt_angles(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sample.raw'
            source.write_bytes(b'bad raw')
            for number in (1, 0):
                source.with_name(f'sample_exported_{number}.xy').write_text(f'0 {number+1}\n1 {number+2}\n9 {number+3}\n')
            measurement = load_experimental_pole(source)
            self.assertIsNone(measurement.radii)
            self.assertEqual(measurement.scans[0][1][0], 1)
            self.page.load_measurement(measurement)
            self.assertFalse(self.page.save_button.enabled)
            self.page.first_angle.set('10')
            self.page.last_angle.set('20')
            self.page.draw_pole_figure()
            np.testing.assert_array_equal(self.page.displayed_radii, [10., 20.])
            source.with_name('sample_exported_1.xy').rename(source.with_name('sample_exported_2.xy'))
            with self.assertRaisesRegex(ValueError, 'Missing.*1'):
                load_experimental_pole(source)

    def test_colour_modes_limits_and_clear(self):
        self.page.load_measurement(load_experimental_pole('scan.raw', raw_fixture()))
        for mode in ('linear', 'log', 'square'):
            self.page.scale_mode.set(mode)
            self.page.on_scale_mode_changed()
            self.page.draw_pole_figure()
            self.assertTrue(self.page.save_button.enabled)
            self.assertEqual(len(self.page.figure.axes), 2)
        self.page.display_mode.set('solid')
        self.page.on_display_mode_changed()
        self.page.lower_limit.set('12')
        self.page.upper_limit.set('14')
        self.page.draw_pole_figure()
        self.assertEqual(len(self.page.figure.axes), 1)
        self.assertEqual(sum(mesh.get_array().count() for mesh in self.page.plot_axis.collections), 3)
        self.page.clear_data()
        self.assertIsNone(self.page.plot_axis)
        self.assertFalse(self.page.save_button.enabled)

    def test_empty_and_duplicate_chi_fail_before_ui_state_changes(self):
        for raw in (SimpleNamespace(is_pole_figure=True, ranges=[]),
                    SimpleNamespace(is_pole_figure=True, ranges=[raw_fixture().ranges[0]] * 2)):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    load_experimental_pole(Path(folder) / 'bad.raw', raw)


class CalculatedMigrationTests(unittest.TestCase):
    def setUp(self):
        set_language('en')
        self.radiation = RadiationSettings()
        self.page = Calculated(self.radiation.lines)
        self.page.load_document(phase())

    def test_both_projections_match_shared_engine_after_numeric_rotation(self):
        for variable, value in zip(self.page.rotation_vars, (20., -30., 42.)):
            variable.set(str(value))
        self.page.apply_exact_rotation()
        for projection in ('stereographic', 'equal_area'):
            self.page.projection_var.set(projection)
            self.page.redraw()
            expected = project_reflections(self.page.crystal, self.page.reflections,
                euler_matrix(20., -30., 42.) @ self.page.base_rotation, projection)
            self.assertEqual([p.hkl for p in self.page.points], [p.hkl for p in expected])
            np.testing.assert_allclose([[p.x, p.y] for p in self.page.points], [[p.x, p.y] for p in expected], atol=1e-12)
            displayed = self.page.ax.collections[0].get_offsets()
            np.testing.assert_allclose(displayed, [pole_display_position(group[0]) for group in self.page.point_groups], atol=1e-12)

    def test_overlay_orientation_coupling_selection_and_promotion(self):
        second = phase('second', (5., 5., 5., 90., 90., 90.))
        self.page.load_overlay_document(second)
        layer = self.page.overlay_layer
        initial = layer.orientation.copy()
        self.page.relative_rotation_vars[2].set('25')
        self.page.apply_relative_rotation()
        np.testing.assert_allclose(layer.orientation, initial)
        self.page.joint_rotation_var.set(True)
        self.page.joint_rotation_changed()
        before = self.page.user_rotation @ self.page.base_rotation
        relationship = before.T @ layer.orientation
        self.page.relative_rotation_vars[0].set('17')
        self.page.apply_relative_rotation()
        after = self.page.user_rotation @ self.page.base_rotation
        np.testing.assert_allclose(after.T @ layer.orientation, relationship, atol=1e-12)
        self.page._select_pole_group(1, layer.point_groups[-1])
        self.assertIn('second', self.page.info_text.text)
        saved = layer.orientation.copy()
        self.page.remove_document(self.page.cif_document)
        self.assertIs(self.page.cif_document, second)
        self.assertIsNone(self.page.overlay_layer)
        np.testing.assert_allclose(self.page.user_rotation @ self.page.base_rotation, saved, atol=1e-12)

    def test_cell_only_disables_intensity_and_radiation_preserves_orientation(self):
        self.assertFalse(self.page.intensity_colour_radio.enabled)
        self.page.rotation_vars[1].set('23')
        self.page.apply_exact_rotation()
        orientation = self.page.user_rotation.copy()
        old_angles = {p.hkl: p.two_theta for p in self.page.points}
        self.radiation.select_custom([('custom', 0.8, 1.)])
        self.page.radiation_changed()
        np.testing.assert_array_equal(self.page.user_rotation, orientation)
        for point in self.page.points:
            self.assertLess(point.two_theta, old_angles[point.hkl])

    def test_mouse_rotation_and_disabled_overlay_drag(self):
        self.page.canvas.draw()
        def event(name, x, y):
            px, py = self.page.ax.transData.transform((x, y))
            return MouseEvent(name, self.page.canvas, px, py, button=1)
        original = self.page.user_rotation.copy()
        self.page.on_press(event('button_press_event', .2, .2))
        self.page.on_motion(event('motion_notify_event', .4, .3))
        self.page.on_release(event('button_release_event', .4, .3))
        self.assertFalse(np.allclose(self.page.user_rotation, original))
        self.page.load_overlay_document(phase('other'))
        original = self.page.user_rotation.copy()
        self.page.on_press(event('button_press_event', .2, .2))
        self.page.on_motion(event('motion_notify_event', .4, .3))
        self.page.on_release(event('button_release_event', .4, .3))
        np.testing.assert_array_equal(self.page.user_rotation, original)

    def test_translated_title_fits_canvas(self):
        set_language('ru')
        self.page.redraw()
        self.page.canvas.draw()
        box = self.page.ax.title.get_window_extent(self.page.canvas.get_renderer())
        self.assertGreaterEqual(box.x0, 0.)
        self.assertLessEqual(box.x1, self.page.figure.bbox.width)
        self.assertLessEqual(box.y1, self.page.figure.bbox.height)

    def test_labels_colour_intensity_and_optional_structure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'si.cif'
            source.write_text(CIF_TEXT)
            self.page.load_document(load_cif_document(source))
            self.page.labels_var.set(True)
            self.page.label_leaders_var.set(True)
            self.page.coincident_outlines_var.set(True)
            self.page.size_by_d_var.set(True)
            for mode in ('d', 'intensity', 'uniform'):
                self.page.color_mode_var.set(mode)
                self.page.change_colour_mode()
                self.page.canvas.draw()
                self.assertTrue(self.page.points)
            self.assertTrue(self.page.intensity_by_spacing)
            self.page.labels_var.set(False)
            self.page.show_structure_var.set(True)
            self.page.structure_visibility_changed()
            self.page.canvas.draw()
            self.assertIs(self.page.structure_viewer.document, self.page.cif_document)
            self.assertTrue(self.page.structure_viewer.visible)
            self.page.show_structure_var.set(False)
            self.page.structure_visibility_changed()
            self.assertFalse(self.page.structure_viewer.visible)


class PoleProjectTests(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore()
        self.radiation = RadiationSettings()
        self.files = ProjectFileService(load_cif_document=load_cif_document,
            read_bruker_raw=lambda path: raw_fixture(), read_raw_scans=lambda *a, **kw: [], read_scan_file=lambda path: [])
        self.page = workspace(self.store, self.radiation, self.files)

    def test_shared_project_promotion_same_phase_overlay_and_unrelated_events(self):
        first, second = [self.store.add_cell_phase(phase(name)) for name in ('one', 'two')]
        self.store.assign(first.uid, POLES)
        calc = self.page.calculated
        calc.rotation_vars[2].set('31')
        calc.apply_exact_rotation()
        orientation = calc.user_rotation.copy()
        self.page.add_overlay(second.uid)
        self.assertIs(calc.overlay_layer.document, second.payload)
        np.testing.assert_array_equal(calc.user_rotation, orientation)
        self.store.rename(first.uid, 'renamed')
        np.testing.assert_array_equal(calc.user_rotation, orientation)
        calc.overlay_rotation_vars[1].set('42')
        calc.apply_overlay_exact_rotation()
        saved = calc.overlay_layer.orientation.copy()
        self.store.assign(first.uid, POLES, False)
        self.assertIs(calc.cif_document, second.payload)
        np.testing.assert_allclose(calc.user_rotation @ calc.base_rotation, saved, atol=1e-12)
        self.page.add_overlay(second.uid)
        self.assertIs(calc.overlay_layer.document, calc.cif_document)
        calc.remove_overlay()
        self.assertTrue(self.store.is_assigned(second.uid, POLES))
        self.assertIs(calc.cif_document, second.payload)
        self.store.assign(second.uid, POLES, False)
        self.assertIsNone(calc.cif_document)

    def test_raw_assignment_and_other_project_events_keep_manual_angles(self):
        document = self.store.add_pole_document('scan.raw', raw_fixture())
        self.store.assign(document.uid, POLES)
        self.page.experimental.first_angle.set('5')
        self.page.experimental.angle_step.set('10')
        self.store.add_cell_phase(phase('unrelated'))
        np.testing.assert_array_equal(self.page.experimental._validated_radii(), [5., 15., 25.])
        self.store.assign(document.uid, POLES, False)
        self.assertIsNone(self.page.experimental.raw_path)

    def test_pure_experimental_loader_and_scene_builder_do_not_import_gui(self):
        code = (
            'import sys; import xrd_workbench.services.experimental_pole; '
            'import xrd_workbench.services.structure_scene; '
            'assert "tkinter" not in sys.modules; '
            'assert not any(name.startswith("PySide6") for name in sys.modules)'
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


PYSIDE_AVAILABLE = importlib.util.find_spec('PySide6') is not None
if PYSIDE_AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, OSError):
        PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, 'PySide6 QtWidgets runtime is unavailable')
class PoleQtIntegrationTests(unittest.TestCase):
    def test_native_controls_bind_phase_rotations_and_shared_radiation(self):
        from xrd_workbench.ui_qt.main_window import MainWindow
        from xrd_workbench.models.project import VIEWER, STRUCTURES
        from PySide6.QtWidgets import QLineEdit
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        try:
            store = window.project
            a, b = [store.add_cell_phase(phase(name)) for name in ('one', 'two')]
            store.assign(a.uid, POLES)
            pole_page = window.pages[POLES]
            pole_page.add_overlay(b.uid)
            calc = pole_page.calculated
            fields = calc.rotation_section.findChildren(QLineEdit)
            fields[0].setText('27')
            fields[0].returnPressed.emit()
            np.testing.assert_allclose(calc.user_rotation, euler_matrix(27., 0., 0.), atol=1e-12)
            calc.joint_rotation_check.click()
            self.assertTrue(calc.overlay_layer.coupled_to_primary)
            window.radiation_settings.select_custom([('custom', .8, 1.)])
            pole_page.radiation_selector.radiation_changed.emit()
            app.processEvents()
            self.assertEqual(window.pages[VIEWER].radiation_selector.combo.currentData(), 'other')
            self.assertEqual(window.pages[STRUCTURES].radiation_selector.combo.currentData(), 'other')
            self.assertAlmostEqual(calc.get_wavelength(), .8)
            store.assign(a.uid, POLES, False)
            self.assertIs(calc.cif_document, b.payload)
        finally:
            window.close()

    def test_native_experimental_angle_edits_and_all_languages(self):
        from xrd_workbench.ui_qt.experimental_pole import ExperimentalPolePage
        app = QApplication.instance() or QApplication([])
        page = ExperimentalPolePage()
        page.show()
        try:
            page.load_measurement(load_experimental_pole('scan.raw', raw_fixture()))
            page.angle_entries['first'].setText('5')
            page.angle_entries['step'].setText('10')
            page.draw_pole_figure()
            np.testing.assert_array_equal(page.displayed_radii, [5., 15., 25.])
            for language in ('ru', 'fr', 'en'):
                set_language(language)
                page.retranslate()
                app.processEvents()
                self.assertTrue(page.save_button.isEnabled())
        finally:
            page.close()
            set_language('en')

    def test_native_experimental_pyqtgraph_cells_click_gap_and_zoom(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtTest import QTest
        from xrd_workbench.ui_qt.experimental_pole import ExperimentalPolePage

        app = QApplication.instance() or QApplication([])
        original_palette = app.palette()
        page = ExperimentalPolePage()
        page.resize(1100, 760)
        page.show()
        try:
            page.set_plot_renderer('pyqtgraph')
            page.load_measurement(load_experimental_pole('scan.raw', raw_fixture()))
            app.processEvents()
            plot = page.pyqtgraph_plot
            self.assertIs(page.plot_stack.currentWidget(), plot)
            self.assertEqual(len(plot.mesh_items), 3)
            self.assertEqual(plot.cell_count, 10)
            np.testing.assert_array_equal(page.displayed_radii, [10., 24., 40.])

            for phi, expected in ((1., 'Intensity: 12'), (6., 'no measured')):
                angle = math.radians(phi)
                scene_point = plot.view_box.mapViewToScene(QPointF(
                    40. * math.sin(angle),
                    40. * math.cos(angle),
                ))
                viewport_point = plot.plot_widget.mapFromScene(scene_point)
                QTest.mouseClick(
                    plot.plot_widget.viewport(),
                    Qt.MouseButton.LeftButton,
                    pos=viewport_point,
                )
                app.processEvents()
                self.assertIn(expected.lower(), page.cursor_text.get().lower())

            viewport = plot.plot_widget.viewport()
            drag_start = viewport.rect().center()
            ranges_before = plot.view_box.viewRange()
            QTest.mousePress(
                viewport,
                Qt.MouseButton.RightButton,
                pos=drag_start,
            )
            QTest.mouseMove(viewport, drag_start + QPoint(32, 18))
            QTest.mouseRelease(
                viewport,
                Qt.MouseButton.RightButton,
                pos=drag_start + QPoint(32, 18),
            )
            app.processEvents()
            ranges_after_right = plot.view_box.viewRange()
            self.assertFalse(np.allclose(ranges_before[0], ranges_after_right[0]))
            self.assertTrue(page.reset_zoom_button.isEnabled())
            QTest.mousePress(
                viewport,
                Qt.MouseButton.MiddleButton,
                pos=drag_start,
            )
            QTest.mouseMove(viewport, drag_start + QPoint(32, 18))
            QTest.mouseRelease(
                viewport,
                Qt.MouseButton.MiddleButton,
                pos=drag_start + QPoint(32, 18),
            )
            app.processEvents()
            ranges_after_middle = plot.view_box.viewRange()
            np.testing.assert_allclose(ranges_after_middle, ranges_after_right)
            plot.reset_view()
            self.assertFalse(page.reset_zoom_button.isEnabled())

            before = page.displayed_radii.copy()
            wheel_position = viewport.mapTo(page, viewport.rect().center())
            QTest.wheelEvent(
                page.windowHandle(),
                wheel_position,
                QPoint(0, 120),
            )
            app.processEvents()
            self.assertGreater(page.zoom_factor, 1.0)
            self.assertTrue(page.reset_zoom_button.isEnabled())
            np.testing.assert_array_equal(page.displayed_radii, before)
            page.reset_zoom()
            self.assertAlmostEqual(page.zoom_factor, 1.0)
            self.assertFalse(page.reset_zoom_button.isEnabled())

            for mode in ('log', 'square'):
                page.scale_mode.set(mode)
                page.draw_pole_figure()
                self.assertEqual(plot.cell_count, 10)
                self.assertTrue(plot.colour_scale.isVisible())

            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'experimental.png'
                with patch(
                    'xrd_workbench.ui_qt.pyqtgraph_experimental.'
                    'QFileDialog.getSaveFileName',
                    return_value=(str(target), 'PNG (*.png)'),
                ):
                    page.save_figure()
                self.assertTrue(target.read_bytes().startswith(b'\x89PNG'))

            dark_palette = QPalette(original_palette)
            dark_palette.setColor(QPalette.ColorRole.Base, QColor('#171b20'))
            dark_palette.setColor(QPalette.ColorRole.Text, QColor('#f0f0f0'))
            app.setPalette(dark_palette)
            app.processEvents()
            app.processEvents()
            self.assertEqual(
                plot.plot_widget.backgroundBrush().color().name(),
                '#171b20',
            )

            page.display_mode.set('solid')
            page.lower_limit.set('12')
            page.upper_limit.set('14')
            page.draw_pole_figure()
            self.assertEqual(plot.cell_count, 3)
        finally:
            page.close()
            app.setPalette(original_palette)
            set_language('en')
