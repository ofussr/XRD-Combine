"""Pole workspace lifetime is governed by assignments without any GUI."""
import copy
import numpy as np
import pytest

from pole_fixtures import phase, raw_fixture
from xrd_workbench.models.project import POLES, VIEWER, ProjectStore
from xrd_workbench.services.pole_figure import euler_matrix


def two_phases():
    store = ProjectStore()
    # Assignment order, rather than project insertion order, defines the roles.
    overlay = store.add_cell_phase(phase('overlay'))
    primary = store.add_cell_phase(phase('primary'))
    store.assign(primary.uid, POLES)
    store.assign(overlay.uid, POLES, additive=True)
    return store, primary, overlay


@pytest.mark.parametrize('remove_from_project', [False, True])
def test_primary_removal_promotes_full_overlay_before_notifications(remove_from_project):
    store, primary, overlay = two_phases()
    state = store.poles.calculated
    layer = state.overlay_layer
    layer.user_rotation = euler_matrix(12, 25, 37)
    layer.center_hkl = (1, 2, 3)
    layer.selected_hkl = (2, 1, 0)
    layer.colour, layer.opacity_percent, layer.size_percent = '#aa3377', 43, 125
    layer.initialized = True
    orientation = layer.orientation.copy()
    events = []
    store.subscribe(lambda *_: events.append(store.poles.calculated.primary_uid))
    if remove_from_project:
        store.remove(primary.uid)
    else:
        store.assign(primary.uid, POLES, False)
    assert events and all(uid == overlay.uid for uid in events)
    assert state.primary_uid == overlay.uid
    assert state.overlay_layer is None
    assert state.center_hkl == (1, 2, 3)
    assert state.selected_hkl == (2, 1, 0)
    assert (state.primary_colour, state.primary_opacity, state.primary_size) == ('#aa3377', 43, 125)
    np.testing.assert_array_equal(state.user_rotation @ state.base_rotation, orientation)


def test_unassignment_discards_phase_orientation_and_other_workspaces_do_not_change_it():
    store, primary, overlay = two_phases()
    state = store.poles.calculated
    state.user_rotation = euler_matrix(5, 15, 25)
    before = state.user_rotation.copy()
    store.assign(primary.uid, VIEWER)
    store.rename(primary.uid, 'renamed')
    np.testing.assert_array_equal(state.user_rotation, before)
    store.assign(overlay.uid, POLES, False)
    store.assign(primary.uid, POLES, False)
    assert state.cif_document is None
    store.assign(primary.uid, POLES)
    np.testing.assert_array_equal(state.user_rotation, np.eye(3))


def test_replacement_resets_only_the_replaced_geometry_and_keeps_stable_uid():
    store, primary, overlay = two_phases()
    state = store.poles.calculated
    state.user_rotation = euler_matrix(1, 2, 3)
    layer = state.overlay_layer
    layer.user_rotation = euler_matrix(9, 8, 7)
    expected = layer.orientation.copy()
    replacement = phase('new cell', (8, 8, 8, 90, 90, 90))
    store.replace_cell_phase(primary.uid, replacement)
    assert state.primary_uid == primary.uid
    assert state.cif_document is replacement
    assert not state.initialized
    np.testing.assert_array_equal(state.user_rotation, np.eye(3))
    np.testing.assert_array_equal(state.overlay_layer.orientation, expected)


def test_raw_unassignment_and_replacement_discard_manual_geometry_limits_and_zoom():
    store = ProjectStore()
    raw = store.add_pole_document('first.raw', raw_fixture())
    other = store.add_pole_document('second.raw', raw_fixture())
    store.assign(raw.uid, POLES)
    state = store.poles.experimental
    state.manual_angles = (5, 10, 25)
    state.intensity_limits = (3, 18)
    state.pyqtgraph_view = ((-10, 10), (-10, 10))
    store.assign(other.uid, POLES)
    assert state.uid == other.uid
    assert state.manual_angles is state.intensity_limits is state.pyqtgraph_view is None
    store.assign(other.uid, POLES, False)
    assert state.uid is None and not state.scans


def test_state_copy_does_not_share_orientation_arrays_or_overlay_settings():
    store, _, _ = two_phases()
    clone = copy.deepcopy(store.poles)
    clone.calculated.user_rotation[0, 0] = 2
    clone.calculated.overlay_layer.user_rotation[1, 1] = 3
    clone.calculated.overlay_layer.colour = '#ffffff'
    assert store.poles.calculated.user_rotation[0, 0] == 1
    assert store.poles.calculated.overlay_layer.user_rotation[1, 1] == 1
    assert store.poles.calculated.overlay_layer.colour == '#d65f3c'
