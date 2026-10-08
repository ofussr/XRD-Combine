"""Shared native Qt workbench fixture and angular scan helpers."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from qt_test_support import PYSIDE_AVAILABLE, QApplication
from test_phase_scan import phase
from xrd_workbench.localization import set_language
from xrd_workbench.models.project import ProjectStore, VIEWER
from xrd_workbench.models.scan import Scan1D

@pytest.fixture
def ui(tmp_path):
    from PySide6.QtCore import QEvent, QSettings
    from xrd_workbench.ui_qt.main_window import MainWindow
    from xrd_workbench.ui_qt.theme import ThemeController
    app = QApplication.instance() or QApplication([])
    set_language('en')
    settings = QSettings(str(tmp_path / 'appearance.ini'), QSettings.Format.IniFormat)
    theme = ThemeController(app, settings)
    store = ProjectStore()
    window = MainWindow(store=store, theme_controller=theme)
    window._confirm_exit = lambda: True
    window.show()
    app.processEvents()
    viewer = window.pages[VIEWER]
    viewer.phase_scan_controls.max_index_spin.setValue(2)
    yield SimpleNamespace(app=app, store=store, window=window, viewer=viewer, root=tmp_path)
    window.close()
    viewer._phase_thread_pool.waitForDone(5000)
    window.deleteLater()
    theme.set_mode('system', persist=False)
    theme.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    set_language('en')


def settle(ui):
    for _ in range(5):
        ui.viewer._phase_thread_pool.waitForDone(5000)
        ui.app.processEvents()
        if not ui.viewer._pending_row_jobs:
            break
    assert not ui.viewer._pending_row_jobs


def add_phase(ui, name='cell'):
    crystal, structure = phase()
    doc = ui.store.add_cell_phase(SimpleNamespace(name=name, source=ui.root/f'{name}.cell',
                                                 diffraction=structure, crystal=crystal))
    ui.store.assign(doc.uid, VIEWER, True)
    return doc


def add_scan(ui, axis='Phi', metadata=True, name='scan'):
    wavelength = ui.viewer.radiation_settings.lines()[0][1]
    tt = math.degrees(2*math.asin(wavelength*math.sqrt(2)/8))
    data = {'axes': {'Phi': np.full(31,0), '2Theta':np.full(31,tt),
                     'Omega':np.full(31,tt/2), 'Chi':np.full(31,45)}} if metadata else {}
    if metadata:
        data['axes'][axis] = np.linspace(-10,350,31)
    doc = ui.store.add_scan(Scan1D(name,np.linspace(-10,350,31),np.ones(31),ui.root/f'{name}.xy',
                                 axis_name=axis,metadata=data))
    ui.store.assign(doc.uid, VIEWER, True)
    return doc


def set_angle(controls,name,low,high=None):
    edits = controls.angle_edits[name]
    for edit,value in zip(edits,(low,low if high is None else high)):
        edit.setText(str(value))
    edits[1].editingFinished.emit()


def displayed_rows(ui,doc):
    return ui.viewer._rows_for_display(ui.viewer.items[doc.uid],
        ui.viewer._phase_limits(ui.viewer.viewer_state.visible_scans()))


