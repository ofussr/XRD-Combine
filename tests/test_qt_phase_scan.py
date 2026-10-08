"""Angular phases in both renderers, fixed-angle editing and asynchronous updates."""

import math
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from qt_test_support import PYSIDE_AVAILABLE
from qt_workbench_harness import ui, settle, add_phase, add_scan, set_angle, displayed_rows
from test_phase_scan import phase
from xrd_workbench.models.project import VIEWER

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason='PySide6 is unavailable')


@pytest.mark.parametrize('renderer', ['pyqtgraph', 'matplotlib'])
def test_phi_autofill_overlay_and_selection_preserve_real_bragg_angle(ui, renderer):
    add_scan(ui)
    doc = add_phase(ui)
    ui.window.change_plot_renderer(renderer)
    settle(ui)
    viewer = ui.viewer
    controls = viewer.phase_scan_controls
    assert controls.mode == 'phi'
    assert [edit.text() for edit in controls.angle_edits['chi']] == ['44.9','45.1']
    assert all(not edit.text() for edit in controls.angle_edits['phi'])
    assert viewer.viewer_state.plot.phase_layout == 'overlay'
    assert viewer._axes_linked
    rows = displayed_rows(ui,doc)
    # Kalpha2 may also be selected: all spectral lines share four angular positions.
    assert sorted({round(r.coordinate,6) for r in rows}) == [0,90,180,270]
    viewer._selected_point = (doc.uid, 'phase', rows[0])
    viewer._refresh_selection_info()
    assert 'φ =' in viewer.selection_info.text()
    assert f'{rows[0].two_theta:.5f}' in viewer.selection_info.text()
    viewer.viewer_state.plot.phase_style = 'profile'
    viewer._draw(preserve_view=False)
    assert not viewer._phase_row_error
    assert not viewer.x_display_combo.isEnabled()


def test_missing_metadata_requires_manual_angles_without_default_zero(ui):
    add_scan(ui,metadata=False)
    add_phase(ui)
    settle(ui)
    controls = ui.viewer.phase_scan_controls
    assert all(not edit.text() for name in ('two_theta','omega','chi') for edit in controls.angle_edits[name])
    assert 'Enter fixed angles' in controls.message.text()
    assert not ui.viewer._row_cache


@pytest.mark.parametrize('mode,axis,angles,hkl', [
    ('chi','Chi',{'phi':90},(1,0,1)),
    ('omega','Omega',{'phi':0,'chi':0},(1,0,1)),
    ('coupled','2Theta',{'phi':0,'chi':0},(0,0,1)),
    ('detector','2Theta',{'phi':0,'chi':0,'omega':'tilt'},(1,0,1)),
])
def test_other_angular_modes_use_their_own_coordinate(ui,mode,axis,angles,hkl):
    scan = add_scan(ui,axis=axis)
    doc = add_phase(ui)
    controls = ui.viewer.phase_scan_controls
    controls.mode_combo.setCurrentIndex(controls.mode_combo.findData(mode))
    for key,value in angles.items():
        if value == 'tilt':
            # A zero-width window needs the exact angle, before UI formatting.
            value = float(scan.payload.metadata['axes']['Omega'][0])+45
        set_angle(controls,key,value)
    settle(ui)
    rows = displayed_rows(ui,doc)
    assert any(row.equivalents == (hkl,) for row in rows), (controls.geometry(),
        ui.viewer._phase_limits(ui.viewer.viewer_state.visible_scans()), rows)
    assert {row.scan_axis for row in rows} == {controls.axis}
    assert ui.viewer._axes_linked
    assert not ui.viewer._phase_row_error


@pytest.mark.parametrize('axis', ['X', 'YDrive', 'ZDrive'])
def test_translation_scan_hides_phase_plot_but_keeps_loaded_cif(ui, axis):
    add_scan(ui,axis=axis,metadata=False)
    phase_doc = add_phase(ui)
    settle(ui)
    assert phase_doc.uid in ui.viewer.items
    assert not ui.viewer._phase_plot_items()
    assert not ui.viewer.pyqtgraph_plot.phase_visible
    assert not ui.viewer.pyqtgraph_plot.overlay_view_box.addedItems
    assert not ui.viewer.phase_style_combo.isEnabled()
    assert not ui.viewer.phase_scan_controls.isEnabled()
    assert axis in ui.viewer.phase_scan_controls.message.text()
    assert not ui.viewer._pending_row_jobs


def test_manual_angles_survive_reference_changes_and_can_be_reset(ui):
    first = add_scan(ui,name='first')
    second = add_scan(ui,name='second')
    add_phase(ui)
    controls = ui.viewer.phase_scan_controls
    controls.reference_combo.setCurrentIndex(controls.reference_combo.findData(second.uid))
    set_angle(controls,'chi',39.9,40.1)
    controls.reference_combo.setCurrentIndex(controls.reference_combo.findData(first.uid))
    set_angle(controls,'chi',40.9,41.1)
    controls.reference_combo.setCurrentIndex(1)
    assert [edit.text() for edit in controls.angle_edits['chi']] == ['39.9','40.1']
    controls.reference_combo.setCurrentIndex(0)
    assert [edit.text() for edit in controls.angle_edits['chi']] == ['40.9','41.1']
    controls.fill_button.click()
    assert [edit.text() for edit in controls.angle_edits['chi']] == ['44.9','45.1']
    settle(ui)


def test_each_phase_has_its_own_orientation_and_partial_edits_survive_redraw(ui):
    add_scan(ui)
    first = add_phase(ui,'one')
    second = add_phase(ui,'two')
    controls = ui.viewer.phase_scan_controls
    controls.phase_combo.setCurrentIndex(controls.phase_combo.findData(first.uid))
    controls.phi_zero_spin.setValue(17)
    controls.phase_combo.setCurrentIndex(controls.phase_combo.findData(second.uid))
    assert controls.phi_zero_spin.value() == 0
    controls.phase_combo.setCurrentIndex(controls.phase_combo.findData(first.uid))
    assert controls.phi_zero_spin.value() == 17
    controls.surface_edit.setText('1 1')
    ui.viewer._draw(preserve_view=True)
    assert controls.surface_edit.text() == '1 1'
    settle(ui)
    assert sorted({round(r.coordinate,6) for r in displayed_rows(ui,first)}) == [17,107,197,287]


def test_replacing_cell_invalidates_reflections_from_previous_payload(ui):
    add_scan(ui)
    doc = add_phase(ui)
    settle(ui)
    assert any(r.equivalents == ((1,0,1),) for r in ui.viewer._row_cache[doc.uid][1])
    crystal, structure = phase((8,8,8,90,90,90))
    ui.store.replace_cell_phase(doc.uid, SimpleNamespace(name='larger', source=doc.source,
                                                        diffraction=structure, crystal=crystal))
    settle(ui)
    rows = ui.viewer._row_cache[doc.uid][1]
    assert any(r.equivalents == ((2,0,2),) for r in rows)
    assert not any(r.equivalents == ((1,0,1),) for r in rows)


@pytest.mark.parametrize('offset', [0,17])
def test_calculation_leaves_event_loop_live_and_discards_previous_geometry(ui, monkeypatch, offset):
    from PySide6.QtCore import QTimer
    import xrd_workbench.ui_qt.viewer_page as module
    original = module.calculate_scan_reflections
    started, release, timer_ran = Event(), Event(), Event()

    def held(*args, **kwargs):
        if not started.is_set():
            started.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'calculate_scan_reflections', held)
    add_scan(ui)
    doc = add_phase(ui)
    deadline = time.monotonic()+5
    while not started.is_set() and time.monotonic()<deadline:
        ui.app.processEvents()
        time.sleep(.005)
    assert started.is_set()
    controls = ui.viewer.phase_scan_controls
    set_angle(controls,'chi',45.9,46.1)
    set_angle(controls,'chi',44.9,45.1)
    controls.phi_zero_spin.setValue(offset)
    def finish():
        timer_ran.set()
        release.set()
    QTimer.singleShot(0, finish)
    ui.app.processEvents()
    assert timer_ran.is_set()
    settle(ui)
    assert sorted({round(r.coordinate,6) for r in displayed_rows(ui,doc)}) == [offset,90+offset,180+offset,270+offset]


def test_phi_slider_and_number_are_linked_and_reuse_reflections(ui, monkeypatch):
    import xrd_workbench.ui_qt.viewer_page as module
    add_scan(ui)
    doc = add_phase(ui)
    settle(ui)
    original = ui.viewer._row_cache[doc.uid]
    def unexpected(*_args,**_kwargs):
        pytest.fail('Changing phi shift must not recalculate reflections.')
    monkeypatch.setattr(module,'calculate_scan_reflections',unexpected)
    controls = ui.viewer.phase_scan_controls
    controls.phi_zero_slider.setValue(125)
    assert controls.phi_zero_spin.value() == 12.5
    assert [r.coordinate for r in displayed_rows(ui,doc)] == pytest.approx([12.5,102.5,192.5,282.5],abs=1e-6)
    controls.phi_zero_spin.setValue(-17.025)
    assert controls.phi_zero_slider.value() == -170
    assert controls.orientation(doc.uid).phi_zero == -17.025
    assert ui.viewer._row_cache[doc.uid] is original
    assert not ui.viewer._pending_row_jobs


def test_inverted_range_is_reported_and_missing_bounds_stay_blank(ui):
    add_scan(ui)
    doc = add_phase(ui)
    settle(ui)
    controls = ui.viewer.phase_scan_controls
    set_angle(controls,'chi',46,44)
    assert 'Range start must not exceed its end' in controls.message.text()
    assert not displayed_rows(ui,doc)
    set_angle(controls,'chi','','45')
    assert 'Enter fixed angles' in controls.message.text()
    controls.fill_button.click()
    settle(ui)
    assert displayed_rows(ui,doc)


@pytest.mark.parametrize('renderer',['pyqtgraph','matplotlib'])
def test_cursor_coordinates_keep_counts_and_match_log_axis(ui,renderer):
    add_scan(ui,axis='2Theta',metadata=False)
    ui.window.change_plot_renderer(renderer)
    viewer = ui.viewer
    viewer._set_limits((0,100),(1,1000))
    ui.app.processEvents()
    counts = viewer.status.text()
    if renderer == 'pyqtgraph':
        from PySide6.QtCore import QPointF
        scene = viewer.pyqtgraph_plot.scan_plot_widget.scene()
        point = viewer.pyqtgraph_plot.scan_view_box.mapViewToScene(QPointF(40,2))
        scene.sigMouseMoved.emit(point)
        viewer.pyqtgraph_plot.cursor_moved.emit('',None,None)
        assert not viewer.cursor_position.text()
        scene.sigMouseMoved.emit(point)
    else:
        from matplotlib.backend_bases import MouseEvent
        px,py = viewer.scan_axis.transData.transform((40,100))
        event = MouseEvent('motion_notify_event',viewer.canvas,px,py)
        viewer.canvas.callbacks.process('motion_notify_event',event)
    assert '2θ = 40.00000°' in viewer.cursor_position.text()
    assert 'Y = 100' in viewer.cursor_position.text()
    assert viewer.status.text() == counts
    assert viewer._selected_point is None
    if renderer == 'matplotlib':
        viewer.canvas.callbacks.process('figure_leave_event',SimpleNamespace())
        assert not viewer.cursor_position.text()


@pytest.mark.parametrize('scale,coordinate,expected',[('sqrt',10,'100'),('square',100,'10')])
def test_cursor_pyqtgraph_nonlinear_ticks(ui,scale,coordinate,expected):
    add_scan(ui,axis='2Theta',metadata=False)
    viewer = ui.viewer
    viewer.viewer_state.plot.intensity_scale = scale
    viewer._draw(preserve_view=False)
    viewer._pyqtgraph_cursor_moved('scan',40,coordinate)
    assert 'Y = '+expected in viewer.cursor_position.text()


def test_cursor_d_spacing_and_separate_phi_panel_use_their_own_axes(ui):
    add_scan(ui,axis='2Theta',metadata=False)
    doc = add_phase(ui)
    settle(ui)
    viewer = ui.viewer
    viewer.viewer_state.plot.x_display_mode = 'd'
    viewer._draw(preserve_view=False)
    viewer._pyqtgraph_cursor_moved('scan',2.5,0)
    assert 'd = 2.50000Å' in viewer.cursor_position.text()
    controls = viewer.phase_scan_controls
    controls.mode_combo.setCurrentIndex(controls.mode_combo.findData('phi'))
    wavelength = viewer.radiation_settings.lines()[0][1]
    tt = math.degrees(2*math.asin(wavelength*math.sqrt(2)/8))
    for key,value in (('two_theta',tt),('omega',tt/2),('chi',45)):
        set_angle(controls,key,value-.1,value+.1)
    settle(ui)
    assert viewer.pyqtgraph_plot.phase_visible
    viewer._pyqtgraph_cursor_moved('phase',90,.5)
    assert viewer.cursor_position.text() == 'φ = 90.00000°; Y = 0.5'
