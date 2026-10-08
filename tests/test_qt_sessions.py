"""Native save/open commands reconstruct real widgets from portable files."""
from time import monotonic
import json

import numpy as np
import pytest

from qt_test_support import PYSIDE_AVAILABLE, QApplication
from test_sessions import project
from xrd_workbench.atom_styles import apply_session_colours, palette, custom_colours
from xrd_workbench.io.session_codec import encode
from xrd_workbench.localization import set_language
from xrd_workbench.models.project import VIEWER, STRUCTURES, POLES, RSM
from xrd_workbench.services.sessions import SessionCancelled, write_session

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason='PySide6 is unavailable')


@pytest.fixture
def window(project, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings, QEvent
    from PySide6.QtWidgets import QMessageBox
    from xrd_workbench.ui_qt.main_window import MainWindow
    from xrd_workbench.ui_qt.theme import ThemeController
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path/'ui.ini'), QSettings.Format.IniFormat)
    theme = ThemeController(app, settings)
    errors = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args[1:]))
    before_colours = palette(), custom_colours()
    current = MainWindow(store=project[0], file_service=project[1], theme_controller=theme)
    current._confirm_exit = lambda: True
    current.test_errors = errors
    current.show()
    app.processEvents()
    yield current
    current.close()
    current.pages[VIEWER]._phase_thread_pool.waitForDone(5000)
    current.deleteLater()
    theme.set_mode('system', persist=False)
    theme.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    apply_session_colours(*before_colours)
    set_language('en')


def ready(window):
    from PySide6.QtTest import QTest
    deadline = monotonic() + 10
    while (window.structure_preparation.busy or window.pages[VIEWER]._pending_row_jobs) and monotonic() < deadline:
        QTest.qWait(10)
    assert not window.structure_preparation.busy
    window.pages[VIEWER]._phase_thread_pool.waitForDone(5000)
    QApplication.instance().processEvents()


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_native_project_reopens_scientific_state_appearance_and_views(window, project, tmp_path, renderer):
    store, service, scan, phase, cell, derived = project
    window.plot_renderer_controller.set_mode(renderer, persist=False)
    window.theme_controller.set_mode('dark', persist=False)
    window.sections.setCurrentIndex(2)
    window.toggle_project_panel()
    ready(window)
    pole = window.pages[POLES].calculated
    pole.state.selected_hkl = (2, 2, 0)
    pole.state.selected_layer_index = 1
    pole.rotation_vars[2].set('27')
    pole.apply_exact_rotation()
    structure = window.pages[STRUCTURES].structure_viewer
    structure.canvas.zoom_by(1)
    saved_camera = structure.state.camera.zoom
    saved_rotation = pole.state.user_rotation.copy()
    if renderer == 'pyqtgraph':
        plot = window.pages[VIEWER].pyqtgraph_plot
        plot.apply_settings(line_width=3.5, grid_enabled=False, grid_alpha=.2,
            x_limits=(22, 35), y_limits=(1, 90), custom_x_label='Measured angle',
            custom_y_label='Counts', legend_enabled=False, x_major_ticks=True,
            x_minor_ticks=True, y_major_ticks=False, y_minor_ticks=True)
        pattern = window.pages[STRUCTURES].calculated_pattern.pyqtgraph_plot
        pattern.custom_x_label, pattern.line_width = 'Calculated angle', 4.0
    path = tmp_path/'session.xrdproject'
    assert window.save_project_as(path), window.test_errors
    assert set(window.session_actions) == {'open_project', 'save_project', 'save_project_as', 'load_tab', 'save_tab'}
    previous = window.project
    window.project.assign(derived.uid, VIEWER, False)
    window.theme_controller.set_mode('light', persist=False)
    assert window.open_project(path), window.test_errors
    ready(window)
    fresh = window.project
    assert fresh is not previous
    assert window.current_workspace() == POLES
    assert window.theme_controller.mode == 'dark'
    assert window.plot_renderer_controller.mode == renderer
    assert not window._drawer_visible
    assert len(fresh.analysis.peaks) == 2
    assert fresh.viewer.items[derived.uid].visible is False
    assert fresh.viewer.items[derived.uid].x_shift == .11
    assert fresh.viewer.phase_scan.orientations[phase.uid].phi_zero == 12.3
    assert window.pages[STRUCTURES].structure_viewer.canvas.zoom == saved_camera
    assert window.pages[POLES].calculated.state.selected_hkl == (2, 2, 0)
    assert window.pages[POLES].calculated.state.selected_layer_index == 1
    np.testing.assert_allclose(window.pages[POLES].calculated.state.user_rotation, saved_rotation)
    assert fresh.rsm.calculated.viewports['angular'] == ((30, 31), (60, 61))
    assert window._project_path == path
    if renderer == 'pyqtgraph':
        plot = window.pages[VIEWER].pyqtgraph_plot
        assert plot.appearance is fresh.viewer.plot.appearance
        assert (plot.line_width, plot.grid_enabled, plot.legend_enabled) == (3.5, False, False)
        assert plot.custom_x_label == 'Measured angle' and plot.custom_y_label == 'Counts'
        assert plot.x_minor_ticks and not plot.y_major_ticks and plot.y_minor_ticks
        pattern = window.pages[STRUCTURES].calculated_pattern.pyqtgraph_plot
        assert pattern.custom_x_label == 'Calculated angle' and pattern.line_width == 4.0
    assert window.save_project()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    calls = []
    window._save_session = lambda path, tab: calls.append((path, tab)) or True
    window.activateWindow()
    window.setFocus()
    QApplication.instance().processEvents()
    QTest.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert calls == [(path, False)] # Retranslation/reloading must not duplicate shortcuts.
    assert not window.test_errors
    assert previous._listeners == [] # Old widgets and preparer cannot react to later edits.


def test_native_tab_import_keeps_other_workspace_analysis_and_comparison(window, project, tmp_path):
    store, service, scan, phase, cell, derived = project
    ready(window)
    window.open_comparison()
    comparison = window.comparison_dialog
    assert comparison is not None
    window.sections.setCurrentIndex(1)
    path = tmp_path/'structures.xrdtab'
    assert window.save_tab(path)
    original_peaks = tuple(store.analysis.peaks)
    viewer_before = encode(store.viewer)
    store.structures.viewer.atom_scale_percent = 170
    assert window.load_tab(path), window.test_errors
    ready(window)
    assert window.current_workspace() == STRUCTURES
    assert window.project.structures.viewer.atom_scale_percent != 170
    assert encode(window.project.viewer) == viewer_before
    assert tuple(window.project.analysis.peaks) == original_peaks
    assert window.comparison_dialog is comparison
    assert comparison.page is not None
    assert not window.test_errors


def test_failed_open_and_cancel_leave_live_project_and_widgets_intact(window, project, tmp_path):
    path = tmp_path/'broken.xrdproject'
    assert window.save_project_as(path)
    original_store, original_pages = window.project, window.pages
    data = json.loads(path.read_text())
    data['schema_version'] = 99
    path.write_text(json.dumps(data))
    assert not window.open_project(path)
    assert window.project is original_store and window.pages is original_pages
    assert window.test_errors
    window.test_errors.clear()
    assert window.save_project_as(path)
    project[2].source.unlink()
    def cancel(issue):
        raise SessionCancelled()
    assert not window.open_project(path, resolver=cancel)
    assert window.project is original_store and window.pages is original_pages
    assert not window.test_errors
