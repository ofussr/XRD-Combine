"""Recreate Viewer adapters from accepted project state, including viewports."""

import pytest

from qt_workbench_harness import ui, add_scan, add_phase, settle, set_angle, displayed_rows
from qt_test_support import PYSIDE_AVAILABLE
from xrd_workbench.models.phase_scan import PhaseOrientation
pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason="PySide6 is unavailable")
if PYSIDE_AVAILABLE:
    from PySide6.QtCore import QEvent
    from xrd_workbench.ui_qt.viewer_page import ViewerPage


def recreate(ui, renderer):
    page = ViewerPage(ui.store, ui.viewer.radiation_settings,
                      plot_renderer_controller=ui.window.plot_renderer_controller)
    page.set_plot_renderer(renderer)
    page.show()
    ui.app.processEvents()
    return page


def dispose(ui, page):
    page._phase_thread_pool.waitForDone(5000)
    ui.app.processEvents()
    page.close()
    page.deleteLater()
    ui.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    ui.app.processEvents()


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_recreated_viewer_restores_order_axis_selection_and_controls(ui, renderer):
    scale = "linear"
    a = add_scan(ui, name='first')
    b = add_scan(ui, name='second')
    viewer = ui.viewer
    viewer.set_plot_renderer(renderer)
    state = ui.store.viewer
    state.items[a.uid].visible = False
    state.items[a.uid].colour = '#aa3366'
    state.items[b.uid].scan.use_axis('Chi')
    state.items[b.uid].x_shift = .5
    state.items[b.uid].show_peak_sum = False
    state.move_to_target(b.uid, a.uid)
    state.selected_uid = b.uid
    viewer._refresh_tree()
    viewer.scale_combo.setCurrentIndex(viewer.scale_combo.findData(scale))
    viewer.fwhm_spin.setValue(.37)
    viewer.background_spacing_spin.setValue(.42)
    viewer.replace_result_radio.setChecked(True)
    viewer._set_limits((44, 47), (2, 50))
    expected = state.plot.viewport
    fresh = recreate(ui, renderer)
    try:
        assert fresh.viewer_state is state
        assert list(fresh.items) == [b.uid, a.uid]
        assert fresh._selected_uid() == b.uid
        assert fresh.axis_combo.currentData() == 'Chi'
        assert fresh.items[b.uid].x_shift == .5
        assert not fresh.items[b.uid].show_peak_sum
        assert not fresh.items[a.uid].visible
        assert fresh.items[a.uid].colour == '#aa3366'
        assert fresh.scale_combo.currentData() == scale
        assert fresh.fwhm_spin.value() == .37
        assert fresh.background_spacing_spin.value() == .42
        assert fresh.replace_result_radio.isChecked()
        x, y = fresh._active_scan_limits()
        assert x == pytest.approx(expected.x)
        assert y == pytest.approx(expected.y)
    finally:
        dispose(ui, fresh)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
@pytest.mark.parametrize('scale', ['linear', 'log', 'sqrt', 'square'])
def test_recreated_viewport_preserves_display_coordinates_in_each_scale(ui, renderer, scale):
    add_scan(ui)
    viewer = ui.viewer
    viewer.set_plot_renderer(renderer)
    viewer.scale_combo.setCurrentIndex(viewer.scale_combo.findData(scale))
    viewer._set_limits((44, 47), (2, 50))
    expected = ui.store.viewer.plot.viewport
    fresh = recreate(ui, renderer)
    try:
        assert fresh.scale_combo.currentData() == scale
        x, y = fresh._active_scan_limits()
        assert x == pytest.approx(expected.x)
        assert y == pytest.approx(expected.y)
    finally:
        dispose(ui, fresh)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_recreated_phase_controls_use_model_ranges_orientation_and_reference(ui, renderer):
    first = add_scan(ui, name='first')
    second = add_scan(ui, name='second')
    doc = add_phase(ui)
    settle(ui)
    viewer = ui.viewer
    viewer.set_plot_renderer(renderer)
    controls = viewer.phase_scan_controls
    controls.reference_combo.setCurrentIndex(controls.reference_combo.findData(second.uid))
    controls.mode_combo.setCurrentIndex(controls.mode_combo.findData('phi'))
    set_angle(controls, 'chi', 44.5, 45.5)
    controls.surface_edit.setText('0 0 2')
    controls.surface_edit.editingFinished.emit()
    controls.inplane_edit.setText('2 0 0')
    controls.inplane_edit.editingFinished.emit()
    controls.phi_zero_spin.setValue(17.025)
    settle(ui)
    # Keep the tree selection different from the geometry reference.
    ui.store.viewer.selected_uid = first.uid
    viewer._restoring_state = True
    viewer._refresh_tree()
    viewer._restoring_state = False
    viewer.x_min_edit.setText('0')
    viewer.x_max_edit.setText('180')
    viewer.apply_limits()
    expected = ui.store.viewer.plot.viewport
    fresh = recreate(ui, renderer)
    try:
        geometry = fresh.phase_scan_controls.geometry()
        assert geometry.bounds('chi') == (44.5, 45.5)
        assert fresh.phase_scan_controls.reference_combo.currentData() == second.uid
        assert fresh.phase_scan_controls.mode_combo.currentData() == 'phi'
        assert fresh.phase_scan_controls.orientation(doc.uid) == PhaseOrientation((0, 0, 2), (2, 0, 0), 17.025)
        assert fresh.phase_scan_controls.max_index_spin.value() == 2
        assert fresh.x_min_edit.text() == '0'
        assert fresh.x_max_edit.text() == '180'
        assert fresh._active_scan_limits()[0] == pytest.approx(expected.x)
        assert fresh.phase_scan_controls.phi_zero_slider.value() == 170
    finally:
        dispose(ui, fresh)


def test_unfinished_and_invalid_inputs_do_not_overwrite_model(ui):
    doc = add_scan(ui)
    phase = add_phase(ui)
    settle(ui)
    controls = ui.viewer.phase_scan_controls
    key = (doc.uid, 'Phi')
    accepted = ui.store.viewer.phase_scan.angles[key]['chi']
    controls.surface_edit.setText('1 1')
    controls.surface_edit.editingFinished.emit()
    set_angle(controls, 'chi', 46, 44)
    assert ui.store.viewer.phase_scan.angles[key]['chi'] == accepted
    assert ui.store.viewer.phase_scan.orientation(phase.uid) == PhaseOrientation()
    assert not displayed_rows(ui, phase)
    fresh = recreate(ui, ui.viewer.plot_renderer)
    try:
        assert fresh.phase_scan_controls.geometry().bounds('chi') == (accepted.low, accepted.high)
        assert fresh.phase_scan_controls.surface_edit.text() == '0 0 1'
    finally:
        dispose(ui, fresh)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_independent_phase_panel_viewport_survives_recreation(ui, renderer):
    add_scan(ui, axis='2Theta', metadata=False)
    add_phase(ui)
    controls = ui.viewer.phase_scan_controls
    controls.mode_combo.setCurrentIndex(controls.mode_combo.findData('phi'))
    for key, value in (('two_theta', 30), ('omega', 15), ('chi', 45)):
        set_angle(controls, key, value - .1, value + .1)
    settle(ui)
    viewer = ui.viewer
    viewer.set_plot_renderer(renderer)
    assert not viewer._axes_linked
    viewer._set_limits((20, 40), (1, 10))
    viewer._set_phase_x_limits((50, 150))
    expected = ui.store.viewer.plot.phase_viewport
    fresh = recreate(ui, renderer)
    try:
        assert fresh._active_scan_limits()[0] == pytest.approx((20, 40))
        assert fresh._active_phase_limits()[0] == pytest.approx(expected.x)
        assert expected.x == pytest.approx((50, 150))
    finally:
        dispose(ui, fresh)


def test_replaced_inactive_measurement_does_not_keep_old_invalid_draft(ui):
    first = add_scan(ui, name='first')
    second = add_scan(ui, name='second')
    controls = ui.viewer.phase_scan_controls
    controls.reference_combo.setCurrentIndex(controls.reference_combo.findData(first.uid))
    set_angle(controls, 'chi', 46, 44)
    controls.reference_combo.setCurrentIndex(controls.reference_combo.findData(second.uid))
    from xrd_workbench.models.scan import clone_scan
    replacement = clone_scan(first.payload)
    replacement.metadata['axes']['Chi'][:] = 60
    ui.store.replace_scan(first.uid, replacement)
    ui.app.processEvents()
    controls.reference_combo.setCurrentIndex(controls.reference_combo.findData(first.uid))
    assert controls.geometry().bounds('chi') == pytest.approx((59.9, 60.1))


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_d_spacing_and_mouse_navigation_use_project_state(ui, renderer):
    add_scan(ui, axis='2Theta', metadata=False)
    viewer = ui.viewer
    viewer.set_plot_renderer(renderer)
    viewer.x_display_combo.setCurrentIndex(viewer.x_display_combo.findData('d'))
    viewer.display_wavelength_combo.setCurrentIndex(2)
    wavelength = viewer.viewer_state.plot.display_wavelength
    # Direct range changes exercise the same callbacks as dragging/zooming.
    if renderer == 'matplotlib':
        viewer.scan_axis.set_xlim(1, 3)
        viewer.scan_axis.set_ylim(1, 10)
    else:
        viewer.pyqtgraph_plot.set_scan_range((1, 3), (1, 10))
    assert viewer.viewer_state.plot.viewport.x == pytest.approx((1, 3))
    fresh = recreate(ui, renderer)
    try:
        assert fresh.x_display_combo.currentData() == 'd'
        assert fresh.viewer_state.plot.display_wavelength == wavelength
        assert fresh.display_wavelength_combo.currentData() == wavelength
        assert fresh._active_scan_limits()[0] == pytest.approx((1, 3))
    finally:
        dispose(ui, fresh)


@pytest.mark.parametrize('bounds', [('nan', 30), (180, 181), (-1, 1), (1, 800)])
def test_invalid_physical_angle_inputs_block_calculation_and_preserve_accepted_bounds(ui, bounds):
    doc = add_scan(ui)
    controls = ui.viewer.phase_scan_controls
    accepted = ui.store.viewer.phase_scan.angles[(doc.uid, 'Phi')]['two_theta']
    set_angle(controls, 'two_theta', *bounds)
    with pytest.raises(ValueError):
        controls.geometry()
    assert ui.store.viewer.phase_scan.angles[(doc.uid, 'Phi')]['two_theta'] == accepted
