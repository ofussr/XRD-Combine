"""Recreate structure and RSM adapters from accepted project state."""
from threading import Event
from time import monotonic

import numpy as np
import pytest

from pole_fixtures import MIXED_CIF, phase
from qt_test_support import PYSIDE_AVAILABLE
from qt_workbench_harness import ui
from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.models.project import POLES, RSM, STRUCTURES
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.pole_figure import euler_matrix, pole_display_orientation

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason='PySide6 is unavailable')


def until(predicate):
    from PySide6.QtTest import QTest
    deadline = monotonic() + 5
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), 'Timed out waiting for a structure result'


def dispose(ui, page):
    from PySide6.QtCore import QEvent
    page.close()
    page.deleteLater()
    ui.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    ui.app.processEvents()


def add_cif(ui):
    path = ui.root / 'mixed.cif'
    path.write_text(MIXED_CIF, encoding='utf-8')
    return ui.store.add_cif_document(path, load_cif_document(path))


def structures_page(ui):
    from xrd_workbench.ui_qt.structures_page import StructuresPage
    original = ui.window.pages[STRUCTURES]
    return StructuresPage(ui.store, original.radiation_settings,
        plot_renderer_controller=ui.window.plot_renderer_controller,
        scene_preparer=ui.window.structure_preparation)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_structure_calculation_table_and_navigation_survive_recreation(ui, renderer):
    doc = add_cif(ui)
    ui.store.assign(doc.uid, STRUCTURES)
    ui.window.change_plot_renderer(renderer)
    workspace = ui.window.pages[STRUCTURES]
    pattern, table = workspace.calculated_pattern, workspace.reflection_table
    pattern.minimum_edit.setText('12'); pattern.maximum_edit.setText('75')
    pattern.intensity_edit.setText('0.3'); pattern.fwhm_edit.setText('0.27')
    pattern.calculate()
    pattern.profile_radio.setChecked(True)
    if renderer == 'matplotlib':
        pattern.axis.set_xlim(22, 36); pattern.axis.set_ylim(0, 83)
    else:
        pattern.pyqtgraph_plot.set_scan_range((22, 36), (0, 83))
        pattern.pyqtgraph_plot.range_changed.emit('scan')
    expected_view = pattern.state.viewport
    table.minimum_edit.setText('16'); table.maximum_edit.setText('65')
    table.calculate()
    table.sort_table(3); table.sort_table(3)
    table.table.setCurrentItem(table.table.topLevelItem(2))
    table.table.header().resizeSection(0, 240)
    selected = table.state.selected_reflection
    workspace.tabs.setCurrentIndex(2)
    pattern.fwhm_edit.setText('invalid')
    pattern.calculate(show_errors=False)
    assert pattern.state.fwhm == .27
    fresh = structures_page(ui)
    try:
        restored, reflected = fresh.calculated_pattern, fresh.reflection_table
        assert restored.profile_radio.isChecked()
        assert float(restored.fwhm_edit.text()) == .27
        assert (restored.state.minimum, restored.state.maximum, restored.state.threshold) == (12, 75, .3)
        actual = ((restored.axis.get_xlim(), restored.axis.get_ylim()) if renderer == 'matplotlib'
                  else restored.pyqtgraph_plot.scan_limits())
        np.testing.assert_allclose(actual, expected_view)
        assert reflected.state.selected_reflection == selected
        assert reflected.table.selectedItems()[0].text(0) == selected[0]
        assert reflected.table.columnWidth(0) == 240
        assert reflected.state.sort_column == 3 and reflected.state.sort_descending
        assert fresh.tabs.currentIndex() == 2
        if renderer == 'pyqtgraph':
            restored.pyqtgraph_plot.reset_requested.emit()
            assert restored.state.viewport is None
            assert restored.pyqtgraph_plot.scan_limits()[0] == pytest.approx((12, 75))
    finally:
        dispose(ui, fresh)


def test_structure_and_pole_preview_restore_independent_styles_and_camera(ui):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from xrd_workbench.ui_qt.pole_figures import PolesPage
    doc = add_cif(ui)
    ui.store.assign(doc.uid, STRUCTURES)
    ui.store.assign(doc.uid, POLES)
    viewer = ui.window.pages[STRUCTURES].structure_viewer
    pole = ui.window.pages[POLES].calculated
    pole.show_structure_var.set(True)
    pole.redraw()
    preview = pole.structure_viewer
    until(lambda: viewer.scene is not None and preview.scene is not None)
    atom = next(iter(viewer.atom_component_rows))
    site = next(iter(viewer.polyhedron_site_rows))
    viewer._set_atom_component_visible(atom, False)
    viewer.atom_component_colours[atom] = '#123456'
    viewer._set_polyhedron_visible(site, False)
    viewer._set_polyhedron_opaque(site, True)
    viewer.polyhedron_site_colours[site] = '#aa3377'
    viewer.polyhedra_check.setChecked(True)
    viewer.style_combo.setCurrentIndex(viewer.style_combo.findData('engraving'))
    viewer.density_slider.setValue(31); viewer.grip_slider.setValue(75)
    viewer.atom_size_slider.setValue(140)
    viewer.absolute_inputs[2].setValue(27); viewer.apply_exact_rotation()
    viewer.canvas.resize(800, 500)
    viewer.canvas.zoom_by(2)
    viewer.canvas.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress,
        QPointF(10, 10), QPointF(10, 10), Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier))
    viewer.canvas.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove,
        QPointF(50, 35), QPointF(50, 35), Qt.MouseButton.NoButton,
        Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier))
    camera = (viewer.state.camera.zoom, viewer.state.camera.pan)
    expected = viewer.canvas.orientation.copy()
    preview.atom_size_slider.setValue(165)
    preview._set_atom_component_visible(atom, False)
    preview.canvas.zoom_by(-1)
    preview_camera = preview.state.camera.zoom
    pole.rotation_vars[2].set('18'); pole.apply_exact_rotation()
    fresh = structures_page(ui)
    original = ui.window.pages[POLES]
    poles = PolesPage(ui.store, original.radiation_settings, original.file_service,
                     scene_preparer=ui.window.structure_preparation)
    try:
        restored = fresh.structure_viewer
        assert not restored.atom_component_rows[atom][2].isChecked()
        assert restored.atom_component_colours[atom] == '#123456'
        assert not restored.polyhedron_site_visibility[site]
        assert site in restored.polyhedron_opaque_sites
        assert restored.polyhedron_site_colours[site] == '#aa3377'
        assert restored.style_combo.currentData() == 'engraving'
        assert restored.density_slider.value() == 31 and restored.grip_slider.value() == 75
        assert restored.atom_size_slider.value() == 140
        np.testing.assert_allclose(restored.canvas.orientation, expected)
        restored.canvas.resize(400, 300)
        restored.canvas.restore_camera()
        assert restored.canvas.zoom == camera[0]
        assert restored.state.camera.pan == camera[1]
        assert (restored.canvas.pan_x, restored.canvas.pan_y) == pytest.approx(
            (camera[1][0] * restored.canvas.width(), camera[1][1] * restored.canvas.height()))
        compact = poles.calculated.structure_viewer
        assert compact.atom_size_slider.value() == 165
        assert not compact.atom_component_rows[atom][2].isChecked()
        assert compact.canvas.zoom == preview_camera
        np.testing.assert_allclose(compact.canvas.orientation, pole_display_orientation(
            pole.user_rotation @ pole.base_rotation))
    finally:
        dispose(ui, fresh)
        dispose(ui, poles)


def test_pending_structure_result_uses_current_project_settings(ui, monkeypatch):
    from xrd_workbench.services.structure_scene import build_structure_scene
    from xrd_workbench.ui_qt import structure_preparation
    release, started = Event(), Event()
    def delayed(crystal):
        started.set()
        assert release.wait(5)
        return build_structure_scene(crystal)
    monkeypatch.setattr(structure_preparation, 'build_structure_scene', delayed)
    fresh = None
    try:
        doc = add_cif(ui)
        ui.store.assign(doc.uid, STRUCTURES)
        until(started.is_set)
        state = ui.store.structures.viewer
        state.user_rotation = euler_matrix(15, 25, 35)
        state.camera.zoom = 1.7
        state.atom_scale_percent = 135
        fresh = structures_page(ui)
        release.set()
        until(lambda: fresh.structure_viewer.scene is not None)
        old = ui.window.pages[STRUCTURES].structure_viewer
        for page in (old, fresh.structure_viewer):
            assert page.atom_size_slider.value() == 135
            assert page.canvas.zoom == 1.7
            np.testing.assert_allclose(page.canvas.orientation, pole_display_orientation(
                euler_matrix(15, 25, 35) @ state.base_rotation))
    finally:
        release.set()
        if fresh is not None:
            dispose(ui, fresh)


def test_rsm_recreation_restores_accepted_maps_and_rejects_invalid_drafts(ui):
    from xrd_workbench.ui_qt.rsm_page import RSMPage
    doc = add_cif(ui)
    ui.store.assign(doc.uid, RSM)
    workspace = ui.window.pages[RSM]
    experiment, calculated = workspace.experimental, workspace.calculated
    experiment.first.setText('20'); experiment.step.setText('0.1')
    for i in range(3):
        scan = ui.store.add_scan(Scan1D(str(i), np.array([44, 45, 46]),
            np.array([i + 1, i + 2, i + 3]), ui.root / f'{i}.xy'))
        ui.store.assign(scan.uid, RSM)
    experiment.view.setCurrentIndex(experiment.view.findData('q'))
    experiment.scale.setCurrentIndex(experiment.scale.findData('log'))
    experiment.cmap.setCurrentText('plasma')
    experiment.max_index.setText('2'); experiment.add_cif_points()
    experiment.plot.getViewBox().setRange(xRange=(-.2, .2), yRange=(2.8, 3.3), padding=0)
    experiment.plot.getViewBox().sigRangeChangedManually.emit([True, True])
    experimental_view = experiment.state.viewports['q']
    calculated.target.setCurrentIndex(calculated.target.findData('angles'))
    calculated.target_a.setText('22.5'); calculated.target_b.setText('45')
    calculated.max_index.setText('2')
    calculated.view.setCurrentIndex(calculated.view.findData('q'))
    calculated.span_x.setText('0.2'); calculated.span_y.setText('0.3')
    calculated.labels.setChecked(False); calculated.target_marker.setChecked(False)
    calculated.draw()
    assert calculated.state.has_calculated
    calculated.plot.getViewBox().setRange(xRange=(-.1, .1), yRange=(2.9, 3.2), padding=0)
    calculated.plot.getViewBox().sigRangeChangedManually.emit([True, True])
    calculated_view = calculated.state.viewports['q']
    workspace.tabs.setCurrentIndex(1)
    experiment.first.setText('invalid'); experiment.draw()
    calculated.target_a.setText('nan'); calculated.draw()
    assert experiment.state.manual_angles == pytest.approx((20, .1, 20.2))
    assert calculated.state.targets['angles'] == (22.5, 45)
    fresh = RSMPage(ui.store, workspace.radiation_settings, workspace.file_service)
    try:
        restored = fresh.experimental
        assert restored.view.currentData() == 'q'
        assert restored.scale.currentData() == 'log' and restored.cmap.currentText() == 'plasma'
        assert restored.state.overlay_enabled and restored.cif_combo.currentData() == doc.uid
        np.testing.assert_allclose(restored.plot.getViewBox().viewRange(), experimental_view)
        restored.first.setText('21'); restored.first.textEdited.emit('21')
        assert not restored.last.text()
        restored.draw()
        assert restored.state.manual_angles == pytest.approx((21, .1, 21.2))
        calc = fresh.calculated
        assert calc.target.currentData() == 'angles' and float(calc.target_a.text()) == 22.5
        assert float(calc.span_x.text()) == .2 and float(calc.span_y.text()) == .3
        assert not calc.labels.isChecked() and not calc.target_marker.isChecked()
        np.testing.assert_allclose(calc.plot.getViewBox().viewRange(), calculated_view)
        assert fresh.tabs.currentIndex() == 1
        overlay = ui.store.add_cell_phase(phase('overlay'))
        fresh.refresh_documents()
        restored.cif_combo.setCurrentIndex(restored.cif_combo.findData(overlay.uid))
        restored.inplane.setText('0 1 0')
        restored.add_cif_points()
        assert restored.state.overlay_enabled
        ui.store.replace_cell_phase(overlay.uid, phase('replacement'))
        fresh.refresh_documents()
        assert not restored.state.overlay_enabled and not restored._overlay_requested
        assert restored.inplane.text() == '1 0 0'
        assert len(restored.plot.getPlotItem().items) == 1
    finally:
        dispose(ui, fresh)
