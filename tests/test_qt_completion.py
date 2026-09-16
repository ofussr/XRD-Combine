"""Storage guarantees and native Qt checks for the final migration controls."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.localization import set_language, tr
from xrd_workbench.models.project import POLES, STRUCTURES, VIEWER, ProjectStore
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.pole_figure import pole_display_orientation
from xrd_workbench.services.reference_peaks import (
    ReferencePeakError, read_reference_peaks, validate_reference_peaks,
    write_reference_peaks,
)
from test_qt_shell import PYSIDE_AVAILABLE, QApplication


MIXED_CIF = """data_mixed
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


class ReferenceStorageTests(unittest.TestCase):
    def test_decimal_comma_unicode_and_empty_database_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pivo.json'
            write_reference_peaks(path, {'  Кремний (111)  ': '28,442'})
            self.assertEqual(read_reference_peaks(path), {'Кремний (111)': 28.442})
            self.assertIn('Кремний', path.read_text(encoding='utf-8'))
            write_reference_peaks(path, {})
            self.assertEqual(read_reference_peaks(path), {})

    def test_invalid_edits_cannot_replace_previous_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pivo.json'
            write_reference_peaks(path, {'Si': 28.44})
            previous = path.read_bytes()
            for edits in ({'Si': 'nan'}, {'Si': float('inf')}, {' ': 30},
                          {'Si': 28, ' Si ': 30}, {'Si': 'bad'}):
                with self.subTest(edits=edits), self.assertRaises(ReferencePeakError):
                    write_reference_peaks(path, edits)
                self.assertEqual(path.read_bytes(), previous)
            with self.assertRaises(ReferencePeakError) as caught:
                validate_reference_peaks([('Si', 28), ('Si', 30)])
            self.assertEqual((caught.exception.code, caught.exception.row), ('duplicate', 2))

    def test_failed_atomic_replace_preserves_original_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pivo.json'
            write_reference_peaks(path, {'Si': 28.44})
            previous = path.read_bytes()
            with patch('xrd_workbench.services.reference_peaks.os.replace',
                       side_effect=PermissionError('read only')):
                with self.assertRaises(PermissionError):
                    write_reference_peaks(path, {'New': 42})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_malformed_database_returns_no_partial_reference_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pivo.json'
            for source in ('[1, 2]', '{"Si": 28, "bad": "NaN"}', '{broken'):
                path.write_text(source, encoding='utf-8')
                self.assertEqual(read_reference_peaks(path), {})


@unittest.skipUnless(PYSIDE_AVAILABLE, 'PySide6 QtWidgets runtime is unavailable')
class NativeCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtCore import QSettings
        from xrd_workbench.ui_qt.main_window import MainWindow
        from xrd_workbench.ui_qt.theme import ThemeController

        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.settings_path = self.root / 'appearance.ini'
        self.settings = QSettings(str(self.settings_path), QSettings.Format.IniFormat)
        self.theme = ThemeController(self.application, self.settings)
        self.store = ProjectStore()
        self.window = MainWindow(store=self.store, theme_controller=self.theme)
        set_language('en')
        self.window.retranslate()
        self.window.show()
        self.application.processEvents()

    def tearDown(self):
        from PySide6.QtCore import QEvent

        self.window.close()
        self.window.deleteLater()
        self.theme.set_mode('system', persist=False)
        self.theme.deleteLater()
        self.application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()
        set_language('en')
        self.folder.cleanup()

    def add_structure(self, name='mixed'):
        path = self.root / (name + '.cif')
        path.write_text(MIXED_CIF.replace('data_mixed', 'data_' + name), encoding='utf-8')
        payload = load_cif_document(path)
        return self.store.add_cif_document(path, payload)

    def open_preview(self):
        document = self.add_structure()
        self.window.open_calculated_poles(document.uid)
        page = self.window.pages[POLES].calculated
        page.show_structure_var.set(True)
        page.structure_visibility_changed()
        self.application.processEvents()
        return document, page, page.structure_viewer

    def drag(self, canvas, button, delta):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        start = QPoint(canvas.width() // 2, canvas.height() // 2)
        QTest.mousePress(canvas, button, pos=start)
        QTest.mouseMove(canvas, start + QPoint(*delta))
        QTest.mouseRelease(canvas, button, pos=start + QPoint(*delta))
        self.application.processEvents()

    def test_theme_menu_applies_to_existing_windows_and_survives_recreation(self):
        from PySide6.QtCore import QSettings
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QMenu
        from xrd_workbench.ui_qt.reference_peaks import ReferencePeaksDialog
        from xrd_workbench.ui_qt.theme import SETTINGS_KEY, ThemeController

        document, page, preview = self.open_preview()
        orientation = preview.canvas.orientation.copy()
        scene = preview.scene
        dialog = ReferencePeaksDialog(self.window, path=self.root / 'pivo.json')
        dialog.show()
        scan = self.store.add_scan(Scan1D('scan', [10, 11], [2, 3], Path('scan.xy'), '2Theta'))
        self.store.assign(scan.uid, VIEWER, True)
        self.window.open_comparison()
        comparison = self.window.comparison_dialog
        original = self.application.palette().color(QPalette.Window)
        for mode, dark in (('dark', True), ('light', False)):
            self.window.theme_actions[mode].trigger()
            self.application.processEvents()
            for widget in (self.window, dialog, dialog.table, preview.canvas, comparison):
                self.assertEqual(widget.palette().color(QPalette.Base).lightnessF() < .5, dark)
                self.assertNotEqual(widget.palette().color(QPalette.Base),
                                    widget.palette().color(QPalette.Text))
            for section in (page.center_section, preview.display_section, preview.atoms_section):
                self.assertEqual(section.toggle.palette().color(QPalette.Button).lightnessF() < .5, dark)
                self.assertEqual(section.toggle.palette().color(QPalette.ButtonText).lightnessF() > .5, dark)
            icon = page.toolbar._actions['home'].icon().pixmap(24, 24).toImage()
            pixels = [icon.pixelColor(x, y).lightnessF() for x in range(icon.width())
                      for y in range(icon.height()) if icon.pixelColor(x, y).alpha() > 200]
            self.assertTrue(pixels)
            self.assertEqual(float(np.mean(pixels)) > .5, dark)
            self.assertEqual(self.settings.value(SETTINGS_KEY), mode)
            self.assertIs(preview.scene, scene)
            self.assertIs(page.cif_document, document.payload)
            np.testing.assert_array_equal(preview.canvas.orientation, orientation)
        self.window.theme_actions['dark'].trigger()
        for language in ('ru', 'fr', 'en'):
            set_language(language)
            self.window.retranslate()
            self.assertTrue(any(menu.title() == tr('qt.application_theme')
                                for menu in self.window.menuBar().findChildren(QMenu)))
            self.assertTrue(self.window.theme_actions['dark'].isChecked())
            self.assertEqual(sum(action.isChecked() for action in self.window.theme_actions.values()), 1)
        restored_settings = QSettings(str(self.settings_path), QSettings.Format.IniFormat)
        restored = ThemeController(self.application, restored_settings)
        self.assertEqual(restored.mode, 'dark')
        restored.deleteLater()
        self.window.theme_actions['system'].trigger()
        self.application.processEvents()
        self.assertEqual(self.application.palette().color(QPalette.Window), original)
        comparison.close()
        dialog.close()

    def test_failed_theme_save_keeps_previous_mode_and_menu_selection(self):
        self.window.theme_actions['light'].trigger()
        before = self.application.palette()
        with patch.object(self.theme, 'set_mode', side_effect=OSError('read only')):
            with patch('xrd_workbench.ui_qt.main_window.QMessageBox.warning') as warning:
                self.window.theme_actions['dark'].trigger()
                warning.assert_called_once()
        self.assertEqual(self.theme.mode, 'light')
        self.assertTrue(self.window.theme_actions['light'].isChecked())
        self.assertFalse(self.window.theme_actions['dark'].isChecked())
        self.assertEqual(self.application.palette(), before)

    def test_screen_change_repairs_stale_main_window_client_geometry(self):
        from PySide6.QtCore import QEvent, QRect

        self.window.resize(1440, 900)
        self.application.processEvents()
        central = self.window.centralWidget()
        expected = QRect(central.geometry())
        expected_menu = QRect(self.window.menuBar().geometry())
        expected_status = QRect(self.window.statusBar().geometry())
        expected_panel = QRect(self.window.project_panel.geometry())
        expected_sections = QRect(self.window.sections.geometry())
        page_ids = {
            workspace: id(page)
            for workspace, page in self.window.pages.items()
        }

        # Reproduce the Windows failure: top-level bands retain the previous
        # window's screen coordinates and the central layout inherits the
        # same horizontal displacement after maximising on another monitor.
        self.window.menuBar().setGeometry(297, 35, 1044, expected_menu.height())
        central.setGeometry(297, 35, 1044, 786)
        self.window.statusBar().setGeometry(297, 799, 1044, expected_status.height())
        self.window.project_panel.setGeometry(-248, 0, 300, 786)
        self.window.sections.setGeometry(60, 0, 980, 786)
        self.assertNotEqual(central.geometry(), expected)
        self.application.sendEvent(
            self.window,
            QEvent(QEvent.Type.ScreenChangeInternal),
        )
        self.application.processEvents()

        self.assertEqual(central.geometry(), expected)
        self.assertEqual(self.window.menuBar().geometry(), expected_menu)
        self.assertEqual(self.window.statusBar().geometry(), expected_status)
        self.assertEqual(self.window.project_panel.geometry(), expected_panel)
        self.assertEqual(self.window.sections.geometry(), expected_sections)
        self.assertEqual(
            {workspace: id(page) for workspace, page in self.window.pages.items()},
            page_ids,
        )

    def test_resize_event_normalises_all_main_window_bands(self):
        from PySide6.QtCore import QRect

        self.window.resize(1320, 820)
        self.application.processEvents()
        menu_height = self.window.menuBar().sizeHint().height()
        status_height = self.window.statusBar().sizeHint().height()
        self.assertEqual(
            self.window.menuBar().geometry(),
            QRect(0, 0, 1320, menu_height),
        )
        self.assertEqual(
            self.window.centralWidget().geometry(),
            QRect(0, menu_height, 1320, 820 - menu_height - status_height),
        )
        self.assertEqual(
            self.window.statusBar().geometry(),
            QRect(0, 820 - status_height, 1320, status_height),
        )

    def test_test_renderer_menu_switches_calculated_poles_and_persists(self):
        from PySide6.QtWidgets import QMenu
        from xrd_workbench.ui_qt.plot_renderer import SETTINGS_KEY

        document, page, _preview = self.open_preview()
        orientation = page.user_rotation.copy()
        self.window.plot_renderer_actions['pyqtgraph'].trigger()
        self.application.processEvents()
        self.assertEqual(page.plot_renderer, 'pyqtgraph')
        self.assertEqual(
            self.window.pages[POLES].experimental.plot_renderer,
            'pyqtgraph',
        )
        viewer = self.window.pages[VIEWER]
        self.assertEqual(viewer.plot_renderer, 'pyqtgraph')
        calculated_pattern = self.window.pages[STRUCTURES].calculated_pattern
        self.assertEqual(calculated_pattern.plot_renderer, 'pyqtgraph')
        self.assertEqual(type(viewer.plot_stack.currentWidget()).__name__,
                         'PyQtGraphViewerPlot')
        self.assertIs(
            calculated_pattern.plot_stack.currentWidget(),
            calculated_pattern.pyqtgraph_plot,
        )
        self.store.assign(document.uid, STRUCTURES, True)
        self.application.processEvents()
        self.assertTrue(calculated_pattern.rows)
        self.assertEqual(calculated_pattern.pyqtgraph_plot.scale_mode, 'linear')
        self.assertGreater(
            len(calculated_pattern.pyqtgraph_plot.scan_plot_item.items),
            0,
        )
        self.assertEqual(type(page.plot_stack.currentWidget()).__name__,
                         'PyQtGraphPolePlot')
        self.assertIs(page.cif_document, document.payload)
        np.testing.assert_array_equal(page.user_rotation, orientation)
        self.assertTrue(page.point_groups)
        self.assertGreater(len(page.pyqtgraph_plot.plot_item.items), 10)
        self.assertEqual(self.settings.value(SETTINGS_KEY), 'pyqtgraph')

        for language in ('ru', 'fr', 'en'):
            set_language(language)
            self.window.retranslate()
            self.assertTrue(any(menu.title() == tr('qt.plot_renderer_test')
                                for menu in self.window.menuBar().findChildren(QMenu)))
            self.assertTrue(self.window.plot_renderer_actions['pyqtgraph'].isChecked())

        self.window.plot_renderer_actions['matplotlib'].trigger()
        self.application.processEvents()
        self.assertEqual(page.plot_renderer, 'matplotlib')
        self.assertEqual(
            self.window.pages[POLES].experimental.plot_renderer,
            'matplotlib',
        )
        self.assertEqual(viewer.plot_renderer, 'matplotlib')
        self.assertEqual(calculated_pattern.plot_renderer, 'matplotlib')
        self.assertIs(viewer.plot_stack.currentWidget(), viewer.matplotlib_plot)
        self.assertIs(
            calculated_pattern.plot_stack.currentWidget(),
            calculated_pattern.matplotlib_plot,
        )
        self.assertIs(page.plot_stack.currentWidget(), page.matplotlib_plot)
        self.assertEqual(self.settings.value(SETTINGS_KEY), 'matplotlib')

    def test_comparison_reuses_pyqtgraph_and_keeps_all_y_modes(self):
        measurement_scan = Scan1D(
            'measurement',
            np.array([20.0, 30.0, 40.0]),
            np.array([2.0, 12.0, 3.0]),
            self.root / 'measurement.xy',
            '2Theta',
        )
        document = self.store.add_scan(measurement_scan)
        self.store.assign(document.uid, VIEWER, True)
        self.window.plot_renderer_actions['pyqtgraph'].trigger()
        self.window.open_comparison()
        self.application.processEvents()

        comparison = self.window.comparison_dialog
        page = comparison.page
        self.assertEqual(page.plot_renderer, 'pyqtgraph')
        measurement = page.items[0]
        substrate = page.add_scan(
            Scan1D(
                'substrate',
                np.array([20.0, 30.0, 40.0]),
                np.array([1.0, 5.0, 1.0]),
                self.root / 'substrate.xy',
                '2Theta',
            ),
            'substrate',
        )
        group_id = page.create_group(measurement, substrate)
        page.group_views[group_id] = {'ymode': 'square'}
        page.tabs.setCurrentIndex(1)
        page.refresh_gallery()
        self.application.processEvents()

        preview = page.gallery_plots[group_id]
        self.assertEqual(preview.scale_mode, 'linear')
        self.assertEqual(preview.height(), 180)
        self.assertFalse(preview.phase_visible)
        self.assertEqual(len(preview.scan_plot_item.items), 2)
        self.assertTrue(all(
            not button.isVisible()
            for button in (
                preview.home_button,
                preview.zoom_button,
                preview.settings_button,
                preview.save_button,
            )
        ))
        measurement_curve = next(
            item for item in preview.scan_plot_item.items
            if item.name() == 'measurement'
        )
        np.testing.assert_allclose(measurement_curve.yData, [4.0, 144.0, 9.0])

        assembly = page.comparison_state.active_assemblies()[0]
        page.open_viewer(assembly)
        self.application.processEvents()
        detail = next(iter(page.viewer_dialogs))
        self.assertIsNotNone(detail.plot)
        self.assertIsNone(detail.figure)
        self.assertTrue(hasattr(detail, 'mode_button'))
        self.assertEqual(detail.y_mode, 'square')
        self.assertEqual(detail.plot.scale_mode, 'linear')
        self.assertAlmostEqual(
            detail.plot.scan_plot_item.items[0].opts['pen'].widthF(),
            2.0,
        )
        detail.toggle_y_mode()
        self.assertEqual(detail.y_mode, 'linear')
        detail.toggle_y_mode()
        self.assertEqual(detail.y_mode, 'log')
        self.assertEqual(detail.plot.scale_mode, 'log')
        detail.toggle_y_mode()
        self.assertEqual(detail.y_mode, 'exp')
        self.assertEqual(detail.plot.scale_mode, 'linear')
        detail.toggle_y_mode()
        self.assertEqual(detail.y_mode, 'square')
        detail.x_min.setText('22')
        detail.x_max.setText('38')
        detail.y_min.setText('0')
        detail.y_max.setText('15')
        detail.apply_limits()
        np.testing.assert_allclose(detail.plot.scan_limits()[0], (22.0, 38.0))
        np.testing.assert_allclose(detail.plot.scan_limits()[1], (0.0, 15.0))
        target = self.root / 'comparison-pyqtgraph.png'
        self.assertTrue(detail.plot.save_png(target))
        self.assertEqual(target.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        page.copy_to_clipboard(assembly, preview)
        self.assertFalse(self.application.clipboard().image().isNull())

        detail.close()
        comparison.close()
        self.application.processEvents()

    def test_pyqtgraph_viewer_curves_selection_right_pan_modes_and_export(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtTest import QTest
        from xrd_workbench.ui_qt.pyqtgraph_viewer import (
            SparseAxisItem,
            ViewerAxisItem,
            ViewerPlotSettingsDialog,
        )

        x_values = np.linspace(10.0, 80.0, 4000)
        y_values = 8.0 + 120.0 * np.exp(-0.5 * ((x_values - 34.0) / 1.2) ** 2)
        scan = self.store.add_scan(
            Scan1D('fast scan', x_values, y_values, self.root / 'fast.xy', '2Theta')
        )
        self.store.assign(scan.uid, VIEWER, True)
        phase = self.add_structure('viewer-phase')
        self.store.assign(phase.uid, VIEWER, True)
        self.application.processEvents()

        viewer = self.window.pages[VIEWER]
        viewer.scan_axis.set_xlim(20.0, 60.0)
        viewer.scan_axis.set_ylim(5.0, 140.0)
        self.window.plot_renderer_actions['pyqtgraph'].trigger()
        self.application.processEvents()
        plot = viewer.pyqtgraph_plot
        self.assertIsNotNone(plot)
        self.assertEqual(len(plot.scan_plot_item.items), 1)
        np.testing.assert_allclose(plot.scan_limits()[0], (20.0, 60.0))
        np.testing.assert_allclose(plot.scan_limits()[1], (5.0, 140.0))
        self.assertGreater(len(plot.overlay_view_box.addedItems), 0)
        self.assertAlmostEqual(plot.line_width, 2.0)
        self.assertAlmostEqual(
            plot.scan_plot_item.items[0].opts['pen'].widthF(),
            2.0,
        )
        self.assertIsInstance(
            plot.scan_plot_item.getAxis('bottom'),
            SparseAxisItem,
        )
        self.assertEqual(
            len(plot.scan_plot_item.getAxis('bottom').tickValues(0.0, 100.0, 800)),
            1,
        )
        self.assertTrue(plot.scan_plot_item.getAxis('top').isVisible())
        self.assertTrue(plot.scan_plot_item.getAxis('right').isVisible())
        self.assertEqual(plot.scan_plot_item.getAxis('top').fixedHeight, 7)
        for button in (
            plot.home_button,
            plot.zoom_button,
            plot.settings_button,
            plot.save_button,
        ):
            self.assertEqual(button.text(), '')
            self.assertFalse(button.icon().isNull())
            self.assertTrue(button.toolTip())
        self.assertAlmostEqual(plot.grid_alpha, 0.10)

        scale_axis = ViewerAxisItem(orientation='left')
        scale_axis.configure(
            scale_mode='sqrt',
            major_ticks=True,
            minor_ticks=True,
        )
        sqrt_levels = scale_axis.tickValues(0.0, 10.0, 800)
        sqrt_physical = np.square(sqrt_levels[0][1])
        np.testing.assert_allclose(
            np.diff(sqrt_physical),
            sqrt_levels[0][0],
            atol=1e-10,
        )
        self.assertFalse(np.allclose(
            np.diff(sqrt_levels[0][1]),
            np.diff(sqrt_levels[0][1])[0],
        ))
        scale_axis.configure(scale_mode='square')
        square_levels = scale_axis.tickValues(0.0, 10000.0, 800)
        square_physical = np.sqrt(square_levels[0][1])
        np.testing.assert_allclose(
            np.diff(square_physical),
            square_levels[0][0],
            atol=1e-10,
        )
        scale_axis.setLogMode(True)
        scale_axis.configure(scale_mode='log')
        log_levels = scale_axis.tickValues(2.0, 6.5, 700)
        self.assertGreaterEqual(len(log_levels), 2)
        np.testing.assert_allclose(log_levels[0][1], [2.0, 3.0, 4.0, 5.0, 6.0])
        expected_log_minor = [
            decade + np.log10(multiplier)
            for decade in range(2, 7)
            for multiplier in range(2, 10)
            if 2.0 < decade + np.log10(multiplier) < 6.5
        ]
        np.testing.assert_allclose(log_levels[1][1], expected_log_minor)
        self.assertFalse(any(
            np.isclose(value, round(value))
            for value in log_levels[1][1]
        ))
        self.assertEqual(
            scale_axis.tickStrings(log_levels[0][1], 1.0, 1.0),
            ['100', '1000', '10000', '1e+05', '1e+06'],
        )
        self.assertEqual(
            scale_axis.tickStrings([-3.0, 0.0, 3.0, 6.0], 1.0, 2.0),
            ['0.001', '1', '1000', '1e+06'],
        )

        point_index = int(np.argmin(np.abs(x_values - 34.0)))
        scene_point = plot.scan_view_box.mapViewToScene(
            QPointF(
                float(x_values[point_index]),
                plot.data_to_view_y(float(y_values[point_index])),
            )
        )
        viewport_point = plot.scan_plot_widget.mapFromScene(scene_point)
        QTest.mouseClick(
            plot.scan_plot_widget.viewport(),
            Qt.MouseButton.LeftButton,
            pos=viewport_point,
        )
        self.application.processEvents()
        self.assertEqual(viewer._selected_point[:2], (scan.uid, 'scan'))
        self.assertLessEqual(abs(viewer._selected_point[2] - point_index), 1)

        viewport = plot.scan_plot_widget.viewport()
        before = plot.scan_limits()
        self.drag(viewport, Qt.MouseButton.RightButton, (36, 18))
        after_right = plot.scan_limits()
        self.assertFalse(np.allclose(before[0], after_right[0]))
        self.assertFalse(np.allclose(before[1], after_right[1]))
        self.drag(viewport, Qt.MouseButton.MiddleButton, (36, 18))
        after_middle = plot.scan_limits()
        np.testing.assert_allclose(after_middle[0], after_right[0])
        np.testing.assert_allclose(after_middle[1], after_right[1])

        wheel_position = viewport.rect().center()
        wheel = QWheelEvent(
            QPointF(wheel_position),
            QPointF(viewport.mapToGlobal(wheel_position)),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.application.sendEvent(viewport, wheel)
        self.application.processEvents()
        after_wheel = plot.scan_limits()
        np.testing.assert_allclose(after_wheel[0], after_middle[0])
        np.testing.assert_allclose(after_wheel[1], after_middle[1])

        viewer.reset_limits()
        self.application.processEvents()
        y_limits = plot.scan_view_box.viewRange()[1]
        plot.zoom_button.click()
        self.assertTrue(plot.zoom_button.isChecked())
        zoom_start = plot.scan_plot_widget.mapFromScene(
            plot.scan_view_box.mapViewToScene(
                QPointF(25.0, y_limits[0] + 0.2 * (y_limits[1] - y_limits[0]))
            )
        )
        zoom_end = plot.scan_plot_widget.mapFromScene(
            plot.scan_view_box.mapViewToScene(
                QPointF(45.0, y_limits[0] + 0.8 * (y_limits[1] - y_limits[0]))
            )
        )
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=zoom_start)
        QTest.mouseMove(viewport, zoom_end)
        self.application.processEvents()
        self.assertIsNotNone(plot.scan_view_box._selection_item)
        self.assertTrue(plot.scan_view_box._selection_item.pen().isCosmetic())
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=zoom_end)
        self.application.processEvents()
        self.assertFalse(plot.zoom_button.isChecked())
        np.testing.assert_allclose(plot.scan_limits()[0], (25.0, 45.0), atol=0.15)

        settings = ViewerPlotSettingsDialog(plot)
        settings.line_width.setValue(2.25)
        settings.grid_opacity.setValue(6)
        settings.x_label.setText('Diffraction angle')
        settings.y_label.setText('Counts')
        settings.legend_enabled.setChecked(False)
        settings.x_minor_ticks.setChecked(True)
        settings.y_minor_ticks.setChecked(False)
        settings.apply_and_accept()
        self.application.processEvents()
        self.assertAlmostEqual(plot.line_width, 2.25)
        self.assertAlmostEqual(plot.grid_alpha, 0.06)
        self.assertEqual(plot.custom_x_label, 'Diffraction angle')
        self.assertEqual(plot.custom_y_label, 'Counts')
        self.assertEqual(
            plot.scan_plot_item.getAxis('bottom').labelText,
            'Diffraction angle',
        )
        self.assertEqual(plot.scan_plot_item.getAxis('left').labelText, 'Counts')
        self.assertIsNone(plot._legend)
        self.assertEqual(
            len(plot.scan_plot_item.getAxis('bottom').tickValues(10.0, 80.0, 800)),
            2,
        )
        self.assertEqual(
            len(plot.scan_plot_item.getAxis('left').tickValues(0.0, 140.0, 800)),
            1,
        )
        self.assertAlmostEqual(
            plot.scan_plot_item.items[0].opts['pen'].widthF(),
            2.25,
        )

        for mode in ('log', 'sqrt', 'square', 'linear'):
            viewer.scale_combo.setCurrentIndex(viewer.scale_combo.findData(mode))
            self.application.processEvents()
            self.assertEqual(viewer.viewer_state.plot.intensity_scale, mode)
            self.assertEqual(plot._logarithmic, mode == 'log')
            if mode == 'sqrt':
                viewer.y_min_edit.setText('100')
                viewer.y_max_edit.setText('10000')
                viewer.apply_limits()
                self.application.processEvents()
                np.testing.assert_allclose(
                    plot.physical_scan_limits()[1],
                    (100.0, 10000.0),
                )

        from PySide6.QtWidgets import QDialog
        viewer.reset_limits()
        self.application.processEvents()
        viewer.select_uid(scan.uid)
        viewer.activate_peak_fit()
        self.assertTrue(plot.scan_view_box.selection_enabled)
        region_start = plot.scan_plot_widget.mapFromScene(
            plot.scan_view_box.mapViewToScene(QPointF(31.0, 5.0))
        )
        region_end = plot.scan_plot_widget.mapFromScene(
            plot.scan_view_box.mapViewToScene(QPointF(37.0, 130.0))
        )
        with patch(
            'xrd_workbench.ui_qt.viewer_page.PeakTargetDialog.exec',
            return_value=QDialog.DialogCode.Accepted,
        ), patch(
            'xrd_workbench.ui_qt.viewer_page.PeakTargetDialog.target_text',
            return_value='35.0',
        ):
            QTest.mousePress(
                plot.scan_plot_widget.viewport(),
                Qt.MouseButton.LeftButton,
                pos=region_start,
            )
            QTest.mouseMove(plot.scan_plot_widget.viewport(), region_end)
            QTest.mouseRelease(
                plot.scan_plot_widget.viewport(),
                Qt.MouseButton.LeftButton,
                pos=region_end,
            )
            self.application.processEvents()
        self.assertAlmostEqual(viewer.items[scan.uid].x_shift, 1.0, places=3)

        viewer.phase_layout_combo.setCurrentIndex(
            viewer.phase_layout_combo.findData('separate')
        )
        self.application.processEvents()
        self.assertTrue(plot.phase_visible)
        self.assertGreater(len(plot.phase_plot_item.items), 0)
        phase_item = viewer.items[phase.uid]
        rows = viewer._rows_for(
            phase_item,
            viewer._phase_limits(viewer.viewer_state.visible_scans()),
        )
        self.assertTrue(rows)
        phase_point = plot.phase_view_box.mapViewToScene(
            QPointF(rows[0].two_theta, 0.45)
        )
        QTest.mouseClick(
            plot.phase_plot_widget.viewport(),
            Qt.MouseButton.LeftButton,
            pos=plot.phase_plot_widget.mapFromScene(phase_point),
        )
        self.application.processEvents()
        self.assertEqual(viewer._selected_point[:2], (phase.uid, 'phase'))

        target = self.root / 'viewer-pyqtgraph.png'
        self.assertTrue(plot.save_png(target))
        self.assertEqual(target.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

        from PySide6.QtGui import QPalette
        self.window.theme_actions['dark'].trigger()
        self.application.processEvents()
        self.application.processEvents()
        self.assertEqual(
            plot.scan_plot_widget.backgroundBrush().color(),
            self.application.palette().color(QPalette.ColorRole.Base),
        )

        expected_x, expected_y = plot.scan_limits()
        self.window.plot_renderer_actions['matplotlib'].trigger()
        self.application.processEvents()
        np.testing.assert_allclose(viewer.scan_axis.get_xlim(), expected_x)
        np.testing.assert_allclose(viewer.scan_axis.get_ylim(), expected_y)

    def test_pyqtgraph_marker_receives_a_real_mouse_selection(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        from xrd_workbench.services.pole_figure import pole_display_position

        _document, page, _preview = self.open_preview()
        self.window.plot_renderer_actions['pyqtgraph'].trigger()
        self.application.processEvents()
        page.selected_hkl = None
        group = next(
            group for group in page.point_groups
            if np.hypot(*pole_display_position(group[0])) < 0.95
        )
        x, y = pole_display_position(group[0])
        plot = page.pyqtgraph_plot
        scene_point = plot.view_box.mapViewToScene(QPointF(x, y))
        viewport_point = plot.plot_widget.mapFromScene(scene_point)
        QTest.mouseClick(
            plot.plot_widget.viewport(),
            Qt.MouseButton.LeftButton,
            pos=viewport_point,
        )
        self.application.processEvents()

        self.assertIsNotNone(page.selected_hkl)
        self.assertIn(page.selected_hkl, [point.hkl for point in group])

        # The selection ring is another ScatterPlotItem drawn above the pole.
        # It must forward the next physical click as well.
        page.selected_hkl = None
        scene_point = plot.view_box.mapViewToScene(QPointF(x, y))
        viewport_point = plot.plot_widget.mapFromScene(scene_point)
        QTest.mouseClick(
            plot.plot_widget.viewport(),
            Qt.MouseButton.LeftButton,
            pos=viewport_point,
        )
        self.application.processEvents()
        self.assertIsNotNone(page.selected_hkl)
        self.assertIn(page.selected_hkl, [point.hkl for point in group])

        ranges_before = plot.view_box.viewRange()
        self.drag(plot.plot_widget.viewport(), Qt.MouseButton.RightButton, (30, 16))
        ranges_after_right = plot.view_box.viewRange()
        self.assertFalse(np.allclose(ranges_before[0], ranges_after_right[0]))
        self.drag(plot.plot_widget.viewport(), Qt.MouseButton.MiddleButton, (30, 16))
        np.testing.assert_allclose(plot.view_box.viewRange(), ranges_after_right)

    def test_reference_editor_cancel_add_edit_remove_save_and_correction_lookup(self):
        from PySide6.QtWidgets import QDialog
        from xrd_workbench.ui_qt.reference_peaks import ReferencePeaksDialog
        from xrd_workbench.ui_qt.viewer_page import PeakTargetDialog

        path = self.root / 'pivo.json'
        write_reference_peaks(path, {'Si': 28.44, 'Delete': 40})
        previous = path.read_bytes()
        cancelled = ReferencePeaksDialog(self.window, path=path)
        cancelled.table.topLevelItem(0).setText(1, '100')
        cancelled.reject()
        self.assertEqual(path.read_bytes(), previous)

        dialog = ReferencePeaksDialog(self.window, path=path)
        dialog.table.topLevelItem(0).setText(0, 'Si (111)')
        dialog.table.topLevelItem(0).setText(1, '28,442')
        dialog.table.topLevelItem(1).setSelected(True)
        dialog.remove_button.click()
        dialog.name_edit.setText('Al2O3')
        dialog.value_edit.setText('35,15')
        dialog.add_button.click()
        self.assertEqual(dialog.table.topLevelItemCount(), 2)
        self.assertEqual(path.read_bytes(), previous)
        dialog.save_and_close()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(read_reference_peaks(path), {'Si (111)': 28.442, 'Al2O3': 35.15})
        target = PeakTargetDialog(35, 100, read_reference_peaks(path), self.window)
        target.reference_combo.setCurrentIndex(target.reference_combo.findText('Al2O3'))
        self.assertAlmostEqual(float(target.target_edit.text()), 35.15)
        with patch('xrd_workbench.ui_qt.main_window.ReferencePeaksDialog') as editor:
            self.window.reference_peaks_action.trigger()
            editor.return_value.exec.assert_called_once()

    def test_invalid_editor_save_stays_open_and_preserves_file(self):
        from PySide6.QtWidgets import QDialog
        from xrd_workbench.ui_qt.reference_peaks import ReferencePeaksDialog

        path = self.root / 'pivo.json'
        write_reference_peaks(path, {'Si': 28.44, 'Al': 38})
        previous = path.read_bytes()
        dialog = ReferencePeaksDialog(self.window, path=path)
        dialog.show()
        for name, value in (('Si', '38'), ('Al', 'NaN'), ('', '38')):
            dialog.table.topLevelItem(1).setText(0, name)
            dialog.table.topLevelItem(1).setText(1, value)
            with patch('xrd_workbench.ui_qt.reference_peaks.QMessageBox.warning') as warning:
                dialog.save_and_close()
                warning.assert_called_once()
            self.assertTrue(dialog.isVisible())
            self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(path.read_bytes(), previous)
        dialog.reject()

    def test_viewer_commands_open_the_selected_phase_and_disable_for_scans(self):
        document = self.add_structure()
        self.store.assign(document.uid, VIEWER, True)
        viewer = self.window.pages[VIEWER]
        viewer.select_uid(document.uid)
        self.assertTrue(viewer.structure_button.isEnabled())
        viewer.structure_button.click()
        self.assertEqual(self.window.current_workspace(), STRUCTURES)
        self.assertIs(self.window.pages[STRUCTURES].structure_viewer.document, document.payload)
        self.assertEqual(self.window.pages[STRUCTURES].tabs.currentIndex(), 0)
        self.window.sections.setCurrentIndex(0)
        viewer.pole_button.click()
        self.assertEqual(self.window.current_workspace(), POLES)
        self.assertIs(self.window.pages[POLES].calculated.cif_document, document.payload)
        self.assertEqual(self.window.pages[POLES].tabs.currentIndex(), 1)
        scan = self.store.add_scan(Scan1D('scan', [10, 11], [2, 3], Path('scan.xy'), '2Theta'))
        self.store.assign(scan.uid, VIEWER, True)
        viewer.select_uid(scan.uid)
        self.assertFalse(viewer.structure_button.isEnabled())
        self.assertFalse(viewer.pole_button.isEnabled())

    def test_native_preview_controls_rotation_pan_zoom_and_visibility(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from unit_cell_gui import CrystalCanvas

        document, page, preview = self.open_preview()
        self.assertIs(type(preview.canvas), CrystalCanvas)
        self.assertIs(preview.document, document.payload)
        self.assertEqual(len(preview.scene.polyhedra), 1)
        self.assertEqual(preview.polyhedron_tree.topLevelItem(0).text(0), 'Nb/Ta')
        self.assertIs(preview.atoms_section.parent(), preview.display_section.content)
        self.assertTrue(preview.polyhedra_section.isHidden())
        self.assertFalse(preview.canvas.show_polyhedra)
        original = page.user_rotation.copy()
        self.drag(preview.canvas, Qt.MouseButton.LeftButton, (20, 15))
        self.assertFalse(np.allclose(page.user_rotation, original))
        expected = pole_display_orientation(page.user_rotation @ page.base_rotation)
        np.testing.assert_allclose(preview.canvas.orientation, expected)
        preview.style_combo.setCurrentIndex(1)
        preview.atom_size_slider.setValue(150)
        preview.basis_check.setChecked(False)
        self.assertTrue(preview.canvas.engraving)
        self.assertEqual(preview.canvas.atom_scale, 1.5)
        self.assertFalse(page.basis_visible.get())
        np.testing.assert_allclose(preview.canvas.orientation, expected)
        self.drag(preview.canvas, Qt.MouseButton.RightButton, (17, 9))
        self.assertEqual((preview.canvas.pan_x, preview.canvas.pan_y), (17, 9))
        wheel = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        self.application.sendEvent(preview.canvas, wheel)
        self.assertGreater(preview.canvas.zoom, 1)
        scene, zoom = preview.scene, preview.canvas.zoom
        page.show_structure_var.set(False)
        page.structure_visibility_changed()
        self.assertTrue(preview.isHidden())
        page.show_structure_var.set(True)
        page.structure_visibility_changed()
        self.assertFalse(preview.isHidden())
        self.assertIs(preview.scene, scene)
        self.assertEqual(preview.canvas.zoom, zoom)
        np.testing.assert_allclose(preview.canvas.orientation, expected)
        self.assertFalse(preview.canvas.show_basis)

    def test_native_preview_two_phase_coupling_and_primary_removal(self):
        from PySide6.QtCore import Qt

        document, page, preview = self.open_preview()
        second = self.add_structure('second')
        self.window.pages[POLES].add_overlay(second.uid)
        layer = page.overlay_layer
        original, overlay = page.user_rotation.copy(), layer.orientation.copy()
        self.drag(preview.canvas, Qt.MouseButton.LeftButton, (14, 8))
        np.testing.assert_array_equal(page.user_rotation, original)
        np.testing.assert_array_equal(layer.orientation, overlay)
        page.joint_rotation_var.set(True)
        page.joint_rotation_changed()
        relationship = (page.user_rotation @ page.base_rotation).T @ layer.orientation
        self.drag(preview.canvas, Qt.MouseButton.LeftButton, (14, 8))
        self.assertFalse(np.allclose(page.user_rotation, original))
        np.testing.assert_allclose((page.user_rotation @ page.base_rotation).T @ layer.orientation,
                                   relationship, atol=1e-12)
        surviving_orientation = layer.orientation.copy()
        self.store.assign(document.uid, POLES, False)
        self.assertIs(page.cif_document, second.payload)
        self.assertIs(preview.document, second.payload)
        np.testing.assert_allclose(preview.canvas.orientation,
                                   pole_display_orientation(surviving_orientation), atol=1e-12)
        self.store.assign(second.uid, POLES, False)
        self.assertIsNone(preview.scene)
        self.assertIsNone(preview.canvas.scene)

    def test_reflections_have_no_second_wavelength_editor_and_use_shared_radiation(self):
        from PySide6.QtWidgets import QLabel, QLineEdit

        _document, page, _preview = self.open_preview()
        ranges = next(section for section in page.control_panel.findChildren(type(page.center_section))
                      if getattr(section, 'title_source', None) == 'text.displayed_reflections')
        self.assertEqual(len(ranges.findChildren(QLineEdit)), 2)
        self.assertFalse(any('λ' in label.text() for label in ranges.findChildren(QLabel)))
        angles = {point.hkl: point.two_theta for point in page.points}
        self.window.radiation_settings.select_custom([('custom', .8, 1.)])
        self.window.pages[POLES].radiation_selector.radiation_changed.emit()
        self.assertEqual(page.get_wavelength(), .8)
        for point in page.points:
            self.assertLess(point.two_theta, angles[point.hkl])

    def test_narrow_preview_layout_and_export_include_native_structure(self):
        from PySide6.QtCore import Qt
        from PIL import Image

        _document, page, preview = self.open_preview()
        self.window.toggle_project_panel()
        self.window.resize(900, 600)
        self.application.processEvents()
        self.assertEqual(page.plot_splitter.orientation(), Qt.Orientation.Vertical)
        self.assertGreater(page.canvas.width(), 240)
        self.window.resize(1440, 900)
        self.application.processEvents()
        self.assertEqual(page.plot_splitter.orientation(), Qt.Orientation.Horizontal)
        context_method = getattr(preview.canvas, 'context', None)
        if context_method is not None:
            context = context_method()
            if context is None or not context.isValid():
                self.skipTest('Qt offscreen platform has no valid OpenGL context')
        preview.style_combo.setCurrentIndex(1)
        before = [axes.get_position().bounds for axes in page.figure.axes]
        size = page.figure.get_size_inches().copy()
        orientation = preview.canvas.orientation.copy()
        exported = preview.figure_for_export(page.figure)
        self.assertEqual(len(exported.axes), len(page.figure.axes) + 1)
        np.testing.assert_allclose(page.figure.get_size_inches(), size)
        self.assertEqual([axes.get_position().bounds for axes in page.figure.axes], before)
        np.testing.assert_array_equal(preview.canvas.orientation, orientation)
        self.assertTrue(exported.axes[-1].images)
        pixels = exported.axes[-1].images[0].get_array()
        self.assertGreater(float(np.std(pixels[:, :, :3])), 5)
        for extension in ('png', 'svg'):
            path = self.root / ('pole-with-structure.' + extension)
            with patch('xrd_workbench.ui_qt.plot_toolbar.QFileDialog.getSaveFileName',
                       return_value=(str(path), '')):
                page.toolbar.save_figure()
            self.assertGreater(path.stat().st_size, 1000)
            if extension == 'png':
                with Image.open(path) as image:
                    self.assertAlmostEqual(image.width / image.height, 2 * size[0] / size[1], delta=.01)
            else:
                self.assertIn('<image ', path.read_text(encoding='utf-8'))
        page.show_structure_var.set(False)
        page.structure_visibility_changed()
        self.assertIs(preview.figure_for_export(page.figure), page.figure)


if __name__ == '__main__':
    unittest.main()
