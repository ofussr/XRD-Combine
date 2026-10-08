"""Transient hkl location and d/2theta range controls on native pole figures."""

import math

import numpy as np
import pytest

from qt_test_support import PYSIDE_AVAILABLE, QApplication
from pole_fixtures import phase
from xrd_workbench.localization import set_language, tr

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason='PySide6 is unavailable')


@pytest.fixture
def page():
    from PySide6.QtCore import QEvent
    from xrd_workbench.ui_qt.calculated_pole import CalculatedPolePage
    app = QApplication.instance() or QApplication([])
    set_language('en')
    widget = CalculatedPolePage()
    widget.resize(1180, 820)
    widget.show()
    widget.load_document(phase('cubic', (4,4,4,90,90,90)))
    app.processEvents()
    yield widget
    widget._hkl_highlight_timer.stop()
    widget.close()
    widget.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    set_language('en')


def locate(page, hkl):
    for edit, value in zip(page.find_hkl_edits, hkl):
        edit.setText(str(value))
    page.find_hkl_button.click()


def use_range_mode(page, mode):
    index = page.range_mode_combo.findData(mode)
    page.range_mode_combo.setCurrentIndex(index)
    page.range_mode_combo.activated.emit(index)


@pytest.mark.parametrize('renderer', ['matplotlib', 'pyqtgraph'])
@pytest.mark.parametrize('projection', ['stereographic', 'equal_area'])
def test_hkl_outside_range_has_transient_outline_in_current_orientation(page, renderer, projection):
    page.set_plot_renderer(renderer)
    page.projection_var.set(projection)
    page.d_lower_var.set('3.9')
    page.d_upper_var.set('4.1')
    page.rebuild_reflections()
    page.rotation_vars[2].set('27')
    page.apply_exact_rotation()
    before = list(page.reflections)
    selection = page.selected_hkl
    orientation = page.user_rotation.copy()
    assert all(row.hkl != (3,1,1) for row in before)
    locate(page, (3,1,1))
    points = page._highlighted_poles()
    assert len(points) == 1
    direction = page.user_rotation @ page.base_rotation @ page.crystal.reciprocal_vector((3,1,1))
    direction /= np.linalg.norm(direction)
    if direction[2] < 0:
        direction = -direction
    np.testing.assert_allclose(points[0].direction, direction, atol=1e-12)
    assert page.reflections == before
    assert page.selected_hkl == selection
    assert page.get_d_range() == (3.9,4.1)
    np.testing.assert_array_equal(page.user_rotation, orientation)
    if renderer == 'matplotlib':
        rings = [item for item in page.ax.collections if item.get_zorder() >= 8]
    else:
        rings = [item for item in page.pyqtgraph_plot.plot_item.items if item.zValue() >= 8]
    assert len(rings) == 2
    page._clear_hkl_highlight()
    assert not page._highlighted_poles()
    if renderer == 'matplotlib':
        assert not [item for item in page.ax.collections if item.get_zorder() >= 8]
    else:
        assert not [item for item in page.pyqtgraph_plot.plot_item.items if item.zValue() >= 8]


def test_locator_timer_restarts_on_new_hkl_and_expires(page):
    from PySide6.QtTest import QTest
    page.set_plot_renderer('pyqtgraph')
    locate(page, (1,1,1))
    QTest.qWait(1100)
    locate(page, (3,1,1))
    QTest.qWait(1100)
    assert page._hkl_highlight[2] == (3,1,1)
    assert page._hkl_highlight_timer.isActive()
    QTest.qWait(1100)
    assert page._hkl_highlight is None


def test_locator_uses_overlay_orientation_and_clears_when_layer_removed(page):
    page.load_overlay_document(phase('overlay', (5,5,5,90,90,90)))
    page.overlay_rotation_vars[0].set('39')
    page.apply_overlay_exact_rotation()
    page.find_phase_combo.setCurrentIndex(page.find_phase_combo.findData(1))
    locate(page, (1,1,2))
    point = page._highlighted_poles()[0]
    direction = page.overlay_layer.orientation @ page.overlay_layer.crystal.reciprocal_vector((1,1,2))
    direction /= np.linalg.norm(direction)
    if direction[2] < 0:
        direction = -direction
    np.testing.assert_allclose(point.direction, direction, atol=1e-12)
    page.remove_overlay()
    assert page._hkl_highlight is None
    assert page.find_phase_combo.count() == 1
    locate(page, (1,1,1))
    page.load_document(phase('replacement'))
    assert page._hkl_highlight is None


def test_invalid_hkl_preserves_plot_and_reports_error(page, monkeypatch):
    from xrd_workbench.ui_qt import calculated_pole
    errors = []
    monkeypatch.setattr(calculated_pole, 'show_error', lambda *args, **kw:errors.append(args))
    before = list(page.reflections)
    for hkl in ((0,0,0), ('1.5',0,1)):
        locate(page, hkl)
    assert len(errors) == 2
    assert page._hkl_highlight is None
    assert page.reflections == before


def test_range_mode_round_trip_preserves_reflections_and_orientation(page):
    before = {row.hkl for row in page.reflections}
    orientation = page.user_rotation.copy()
    use_range_mode(page, 'two_theta')
    assert not page.range_angle_controls.isHidden()
    assert page.range_d_controls.isHidden()
    assert '2θ =' in page._plot_title()
    assert {row.hkl for row in page.reflections} == before
    use_range_mode(page, 'd')
    assert page.get_d_range() == pytest.approx((1,13))
    assert {row.hkl for row in page.reflections} == before
    np.testing.assert_array_equal(page.user_rotation, orientation)


def test_angular_bounds_follow_wavelength_and_support_zero_lower_limit(page):
    use_range_mode(page, 'two_theta')
    page.two_theta_lower_var.set('20')
    page.two_theta_upper_var.set('80')
    page.rebuild_reflections()
    assert all(20 <= row.two_theta <= 80 for row in page.reflections)
    before = page.get_d_range()
    page.wavelength_var.set('.8')
    page.radiation_changed()
    assert page.two_theta_lower_var.get() == '20'
    assert page.two_theta_upper_var.get() == '80'
    assert page.get_d_range()[0] < before[0]
    assert all(20 <= row.two_theta <= 80 for row in page.reflections)
    page.two_theta_lower_var.set('0')
    page.rebuild_reflections()
    assert math.isinf(page.get_d_range()[1])
    use_range_mode(page, 'd')
    assert math.isinf(page.get_d_range()[1])


def test_invalid_ranges_and_switch_are_non_destructive(page, monkeypatch):
    from xrd_workbench.ui_qt import calculated_pole
    errors = []
    monkeypatch.setattr(calculated_pole, 'show_error', lambda *args, **kw:errors.append(args))
    before = list(page.reflections)
    page.d_lower_var.set('nan')
    use_range_mode(page, 'two_theta')
    assert page.range_mode_var.get() == 'd'
    assert page.reflections == before
    page.d_lower_var.set('.1')
    page.d_upper_var.set('.2')
    use_range_mode(page, 'two_theta')
    assert page.range_mode_var.get() == 'd'
    assert page.reflections == before
    page.d_lower_var.set('1')
    page.d_upper_var.set('13')
    use_range_mode(page, 'two_theta')
    for lower, upper in (('80','20'), ('0','181'), ('nan','80')):
        page.two_theta_lower_var.set(lower)
        page.two_theta_upper_var.set(upper)
        page.rebuild_reflections()
        assert page.reflections == before
    assert len(errors) == 5


def test_new_controls_retranslate_without_changing_search_or_ranges(page):
    use_range_mode(page, 'two_theta')
    locate(page, (3,1,1))
    bounds = page.get_d_range()
    highlight = page._hkl_highlight
    for language in ('ru','fr','en'):
        set_language(language)
        page.retranslate()
        assert page.find_hkl_section.toggle.text().endswith(tr('pole.find_hkl'))
        assert page.range_mode_var.get() == 'two_theta'
        assert page.get_d_range() == bounds
        assert page._hkl_highlight == highlight
