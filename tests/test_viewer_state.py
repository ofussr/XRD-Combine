"""Project-owned Viewer state without constructing any GUI."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xrd_workbench.models.phase_scan import AngleRange, PhaseOrientation
from xrd_workbench.models.project import ProjectStore, VIEWER, RSM
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.models.viewer import ViewerViewport


def measurement(name='scan', chi=45):
    phi = np.linspace(-10, 350, 31)
    return Scan1D(name, phi, np.arange(31) + 1, Path(name + '.xy'), axis_name='Phi',
                  metadata={'axes': {'Phi': phi, 'Chi': np.full(31, chi),
                                     '2Theta': np.full(31, 30), 'Omega': np.full(31, 15)}})


def assigned_scan(store, **kwargs):
    doc = store.add_scan(measurement(**kwargs))
    store.assign(doc.uid, VIEWER)
    return doc


def test_assignment_creates_working_scan_and_geometry_without_a_viewer():
    store = ProjectStore()
    doc = assigned_scan(store)
    item = store.viewer.items[doc.uid]
    assert item.scan is not doc.payload
    geometry = store.viewer.phase_scan.geometry(doc.uid, item.scan)
    assert geometry.mode == 'phi'
    assert geometry.bounds('chi') == pytest.approx((44.9, 45.1))
    item.scan.use_axis('Chi')
    assert doc.payload.axis_name == 'Phi'
    item.scan.y[:] = 0
    assert np.all(doc.payload.y > 0)


def test_model_can_be_copied_without_widgets_and_keeps_independent_choices():
    store = ProjectStore()
    a, b = assigned_scan(store), assigned_scan(store, name='second')
    state = store.viewer
    state.items[a.uid].visible = False
    state.items[a.uid].colour = '#456789'
    state.items[b.uid].scan.use_axis('Chi')
    state.items[b.uid].x_shift = .5
    state.move_to_target(b.uid, a.uid)
    state.selected_uid = b.uid
    state.plot.viewport = ViewerViewport((10, 20), (2, 50))
    state.phase_scan.angles[(a.uid, 'Phi')]['chi'] = AngleRange(44, 46)
    copy = deepcopy(state)
    assert list(copy.items) == [b.uid, a.uid]
    assert copy.selected_uid == b.uid
    assert copy.plot.viewport == state.plot.viewport
    assert copy.items[b.uid].scan.axis_name == 'Chi'
    copy.items[a.uid].visible = True
    copy.phase_scan.angles[(a.uid, 'Phi')]['chi'] = AngleRange(43, 47)
    assert not state.items[a.uid].visible
    assert state.phase_scan.angles[(a.uid, 'Phi')]['chi'] == AngleRange(44, 46)


def test_source_replacement_resets_working_transform_and_all_axis_geometry():
    store = ProjectStore()
    doc = assigned_scan(store)
    state = store.viewer
    item = state.items[doc.uid]
    item.visible = False
    item.colour = '#abcdef'
    item.x_shift = 7
    state.phase_scan.angles[(doc.uid, 'Phi')]['chi'] = AngleRange(40, 50)
    item.scan.use_axis('Chi')
    state.phase_scan.ensure_angles(doc.uid, item.scan)
    store.replace_scan(doc.uid, measurement(chi=60))
    assert state.items[doc.uid] is item
    assert item.scan.axis_name == 'Phi'
    assert item.x_shift == 0
    assert not item.visible and item.colour == '#abcdef'
    assert (doc.uid, 'Chi') not in state.phase_scan.angles
    assert state.phase_scan.geometry(doc.uid, item.scan).bounds('chi') == pytest.approx((59.9, 60.1))


def test_unassign_discards_item_geometry_and_selection_but_keeps_project_scan():
    store = ProjectStore()
    doc = assigned_scan(store)
    store.assign(doc.uid, RSM)
    store.viewer.selected_uid = doc.uid
    store.viewer.phase_scan.reference_uid = doc.uid
    store.viewer.items[doc.uid].visible = False
    assert (doc.uid, 'Phi') in store.viewer.phase_scan.angles
    store.assign(doc.uid, VIEWER, False)
    assert doc.uid not in store.viewer.items
    assert not any(key[0] == doc.uid for key in store.viewer.phase_scan.angles)
    assert store.viewer.selected_uid is None
    assert store.viewer.phase_scan.reference_uid is None
    assert store.documents[doc.uid] is doc
    assert store.is_assigned(doc.uid, RSM)
    store.assign(doc.uid, VIEWER)
    assert store.viewer.items[doc.uid].visible
    assert store.viewer.items[doc.uid].scan.axis_name == 'Phi'


def test_phase_removal_discards_orientation_and_rename_preserves_it():
    store = ProjectStore()
    doc = store.add_cell_phase(SimpleNamespace(name='cell', source=Path('cell.cell'), diffraction=object()))
    store.assign(doc.uid, VIEWER)
    orientation = PhaseOrientation((1, 1, 1), (1, -1, 0), 17)
    store.viewer.phase_scan.set_orientation(doc.uid, orientation)
    store.rename(doc.uid, 'renamed')
    assert store.viewer.items[doc.uid].name == 'renamed'
    assert store.viewer.phase_scan.orientation(doc.uid) == orientation
    store.remove(doc.uid)
    assert doc.uid not in store.viewer.phase_scan.orientations
    assert doc.uid not in store.viewer.items


@pytest.mark.parametrize('bounds', [(5, 4), (float('nan'), 1), (0, float('inf'))])
def test_invalid_ranges_cannot_enter_accepted_model(bounds):
    with pytest.raises(ValueError):
        AngleRange(*bounds)
