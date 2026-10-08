"""Recreate native pole adapters from project state in either renderer."""
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from qt_test_support import PYSIDE_AVAILABLE, QApplication
from qt_workbench_harness import ui
from pole_fixtures import phase, raw_fixture
from xrd_workbench.models.project import POLES

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason='PySide6 is unavailable')


def assert_aspect_locked_view(actual, expected):
    # A different widget aspect ratio may expand one axis to keep circles round.
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual.mean(axis=1), expected.mean(axis=1), atol=1e-10)
    assert np.all(actual[:, 0] <= expected[:, 0] + 1e-10)
    assert np.all(actual[:, 1] >= expected[:, 1] - 1e-10)
    assert np.any(np.isclose(actual[:, 1] - actual[:, 0], expected[:, 1] - expected[:, 0]))


def add_phase(ui, name):
    doc = ui.store.add_cell_phase(phase(name))
    ui.store.assign(doc.uid, POLES, additive=True)
    return doc


def recreate(ui):
    from xrd_workbench.ui_qt.pole_figures import PolesPage
    original = ui.window.pages[POLES]
    fresh = PolesPage(ui.store, original.radiation_settings, original.file_service,
                      plot_renderer_controller=ui.window.plot_renderer_controller)
    fresh.resize(1180, 820)
    fresh.show()
    ui.app.processEvents()
    return fresh


def dispose(ui, page):
    from PySide6.QtCore import QEvent
    page.close()
    page.deleteLater()
    ui.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    ui.app.processEvents()


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_calculated_recreation_restores_geometry_styles_ranges_and_tab(ui, renderer):
    first = add_phase(ui, 'primary')
    second = add_phase(ui, 'overlay')
    ui.window.change_plot_renderer(renderer)
    workspace = ui.window.pages[POLES]
    page = workspace.calculated
    page.h_var.set('1'); page.k_var.set('1'); page.l_var.set('1')
    page.apply_center()
    page.rotation_vars[2].set('31.25')
    page.apply_exact_rotation()
    page.overlay_rotation_vars[0].set('19.5')
    page.apply_overlay_exact_rotation()
    page.primary_colour_var.set('#ff3377')
    page.overlay_colour_var.set('#228844')
    page.primary_opacity_var.set(57)
    page.overlay_size_var.set(145)
    page.global_size_var.set(130)
    page.projection_var.set('equal_area')
    page.labels_var.set(True)
    page.label_leaders_var.set(True)
    page.size_by_d_var.set(True)
    page.joint_rotation_var.set(True)
    page.d_lower_var.set('1.5'); page.d_upper_var.set('8')
    page.rebuild_reflections()
    page.range_mode_var.set('two_theta')
    page.change_range_mode()
    workspace.tabs.setCurrentIndex(1)
    expected = page.user_rotation @ page.base_rotation
    overlay_orientation = page.overlay_layer.orientation.copy()
    selected = page.selected_hkl
    fresh = recreate(ui)
    try:
        restored = fresh.calculated
        assert restored.state is ui.store.poles.calculated
        assert (restored.state.primary_uid, restored.state.overlay_uid) == (first.uid, second.uid)
        np.testing.assert_array_equal(restored.user_rotation @ restored.base_rotation, expected)
        np.testing.assert_array_equal(restored.overlay_layer.orientation, overlay_orientation)
        assert restored.center_hkl == (1, 1, 1)
        assert restored.selected_hkl == selected
        assert restored.primary_colour_var.get() == '#ff3377'
        assert restored.overlay_colour_var.get() == '#228844'
        assert restored.primary_opacity_var.get() == 57
        assert restored.overlay_size_var.get() == 145
        assert restored.global_size_var.get() == 130
        assert restored.projection_var.get() == 'equal_area'
        assert restored.labels_var.get() and restored.label_leaders_var.get()
        assert restored.size_by_d_var.get() and restored.joint_rotation_var.get()
        assert restored.get_d_range() == pytest.approx((1.5, 8))
        assert restored.range_mode_var.get() == 'two_theta'
        assert not restored.range_angle_controls.isHidden()
        assert fresh.tabs.currentIndex() == 1
        assert restored._hkl_highlight is None
    finally:
        dispose(ui, fresh)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_experimental_recreation_keeps_manual_angles_limits_samples_and_zoom(ui, renderer):
    raw = ui.store.add_pole_document('scan.raw', raw_fixture())
    ui.store.assign(raw.uid, POLES)
    ui.window.change_plot_renderer(renderer)
    page = ui.window.pages[POLES].experimental
    page.first_angle.set('5'); page.angle_step.set('10')
    page.lower_limit.set('2'); page.upper_limit.set('18')
    page.scale_mode.set('square')
    page.fill_colour.set('#cc5566')
    page.draw_pole_figure()
    measurement = page.measurement
    if renderer == 'matplotlib':
        x, y = page.plot_axis.transData.transform((0.3, 15))
        page.on_plot_scroll(MouseEvent('scroll_event', page.canvas, x, y, button='up'))
        expected_view = page.state.matplotlib_view
        assert expected_view is not None
    else:
        page.pyqtgraph_plot.view_box.setRange(xRange=(-12, 12), yRange=(-12, 12), padding=0)
        page.pyqtgraph_plot._report_zoom()
        expected_view = page.state.pyqtgraph_view
        assert expected_view is not None
    fresh = recreate(ui)
    try:
        restored = fresh.experimental
        assert restored.state is ui.store.poles.experimental
        assert restored.measurement is measurement
        assert restored.state.manual_angles == (5, 10, 25)
        assert restored.state.intensity_limits == (2, 18)
        assert restored.scale_mode.get() == 'square'
        assert restored.fill_colour.get() == '#cc5566'
        np.testing.assert_array_equal(restored.displayed_radii, [5, 15, 25])
        np.testing.assert_array_equal(restored.displayed_scans[0][1], [1, 2, 3, 4])
        if renderer == 'matplotlib':
            assert restored.state.matplotlib_view == expected_view
            base_x, base_y, base_w, base_h = restored.base_plot_position
            position = restored.plot_axis.get_position(original=True).bounds
            assert position == pytest.approx((base_x+expected_view[0]*base_w, base_y+expected_view[1]*base_h,
                                              expected_view[2]*base_w, expected_view[3]*base_h))
        else:
            assert_aspect_locked_view(restored.pyqtgraph_plot.view_box.viewRange(), expected_view)
        assert fresh.tabs.currentIndex() == 0
    finally:
        dispose(ui, fresh)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
def test_calculated_navigation_survives_recreation(ui, renderer):
    add_phase(ui, 'primary')
    ui.window.change_plot_renderer(renderer)
    page = ui.window.pages[POLES].calculated
    if renderer == 'matplotlib':
        page.ax.set_xlim(-.4, .6); page.ax.set_ylim(-.3, .7)
        expected = page.state.matplotlib_view
    else:
        page.pyqtgraph_plot.view_box.setRange(xRange=(-.4, .6), yRange=(-.3, .7), padding=0)
        page.pyqtgraph_plot.view_box.sigRangeChangedManually.emit([True, True])
        expected = page.state.pyqtgraph_view
    fresh = recreate(ui)
    try:
        restored = fresh.calculated
        actual = ((restored.ax.get_xlim(), restored.ax.get_ylim()) if renderer == 'matplotlib'
                  else restored.pyqtgraph_plot.view_box.viewRange())
        if renderer == "matplotlib":
            np.testing.assert_allclose(actual, expected)
        else:
            assert_aspect_locked_view(actual, expected)
    finally:
        dispose(ui, fresh)


def test_invalid_drafts_do_not_replace_accepted_calculated_or_raw_geometry(ui, monkeypatch):
    from xrd_workbench.ui_qt import calculated_pole
    errors = []
    monkeypatch.setattr(calculated_pole, 'show_error', lambda *args, **kwargs: errors.append(args))
    primary = add_phase(ui, 'primary')
    raw = ui.store.add_pole_document('scan.raw', raw_fixture())
    ui.store.assign(raw.uid, POLES)
    workspace = ui.window.pages[POLES]
    page = workspace.calculated
    accepted_d = page.state.d_range
    page.d_lower_var.set('9'); page.d_upper_var.set('2')
    page.rebuild_reflections()
    page.h_var.set('1.5'); page.apply_center()
    assert len(errors) == 2
    assert page.state.d_range == accepted_d
    assert page.state.center_hkl == (0, 1, 0)
    experimental = workspace.experimental
    accepted_radii = experimental.state.loaded_radii.copy()
    accepted_limits = experimental.state.intensity_limits
    experimental.first_angle.set('-10')
    experimental.lower_limit.set('nan')
    assert experimental.state.manual_angles is None
    np.testing.assert_array_equal(experimental.state.loaded_radii, accepted_radii)
    assert experimental.state.intensity_limits == accepted_limits
    fresh = recreate(ui)
    try:
        assert fresh.calculated.get_d_range() == accepted_d
        np.testing.assert_array_equal(fresh.experimental.displayed_radii, accepted_radii)
        assert float(fresh.experimental.lower_limit.get()) == accepted_limits[0]
    finally:
        dispose(ui, fresh)

    ui.store.assign(raw.uid, POLES, False)
    ui.store.assign(primary.uid, POLES, False)
    assert all(value.get() == '' for value in
               (experimental.first_angle, experimental.angle_step, experimental.last_angle))
    assert page.info_text.toPlainText() == ''
    assert page.status_var.get() == ''


def test_project_tree_keeps_live_item_during_assignment_and_refreshes_after_signal(ui):
    from PySide6.QtCore import Qt
    doc = ui.store.add_cell_phase(phase('primary'))
    ui.app.processEvents()
    panel = ui.window.project_panel
    from PySide6.QtWidgets import QTreeWidgetItemIterator
    iterator = QTreeWidgetItemIterator(panel.tree)
    while iterator.value() is not None and iterator.value().data(0, Qt.ItemDataRole.UserRole) != doc.uid:
        iterator += 1
    item = iterator.value()
    assert item is not None
    resets = []
    panel.tree.model().modelReset.connect(lambda: resets.append(True))
    item.setCheckState(1, Qt.CheckState.Checked)
    assert not resets
    assert item.data(0, Qt.ItemDataRole.UserRole) == doc.uid
    assert panel._refresh_timer.isActive()
    ui.app.processEvents()
    assert resets


def test_single_phase_color_mode_and_changed_radiation_restore_without_resetting_rotation(ui):
    from xrd_workbench.ui_qt.calculated_pole import CalculatedPolePage
    add_phase(ui, 'primary')
    page = ui.window.pages[POLES].calculated
    page.color_mode_var.set('d')
    page.rotation_vars[1].set('21')
    page.apply_exact_rotation()
    rotation = page.user_rotation.copy()
    fresh = CalculatedPolePage(lambda: [('test', .8, 1)], state=page.state)
    try:
        assert fresh.color_mode_var.get() == 'd'
        np.testing.assert_array_equal(fresh.user_rotation, rotation)
        assert fresh.state.wavelength == .8
        for reflection in fresh.reflections:
            assert reflection.two_theta == pytest.approx(fresh.crystal.two_theta(reflection.hkl, .8))
    finally:
        dispose(ui, fresh)
