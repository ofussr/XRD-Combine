"""RSM and structure source lifetimes are enforced without UI adapters."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pole_fixtures import phase
from xrd_workbench.models.project import POLES, RSM, STRUCTURES, ProjectStore
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.pole_figure import euler_matrix


@pytest.mark.parametrize('action', ['replace', 'unassign', 'remove'])
def test_phase_cleanup_precedes_notifications_and_keeps_workspace_options(action):
    store = ProjectStore()
    document = store.add_cell_phase(phase())
    for workspace in (STRUCTURES, POLES):
        store.assign(document.uid, workspace)
    structures = store.structures
    preview = store.poles.calculated.preview
    for view in (structures.viewer, preview):
        view.user_rotation = euler_matrix(15, 25, 35)
        view.camera.zoom = 2
        view.atom_component_visibility['site'] = False
        view.polyhedron_opaque_sites.add('octahedron')
        view.atom_scale_percent = 145
    structures.pattern.rows = ['cached']
    structures.pattern.viewport = ((10, 20), (0, 80))
    structures.table.selected_reflection = ('1 0 0', 'Cu', 1.54)
    structures.pattern.fwhm = .23
    observations = []
    store.subscribe(lambda *_: observations.append((
        structures.viewer.camera.zoom, list(structures.pattern.rows),
        structures.table.selected_reflection)))
    if action == 'replace':
        replacement = phase('replacement', (8, 8, 8, 90, 90, 90))
        store.replace_cell_phase(document.uid, replacement)
        assert structures.payload is preview.payload is replacement
    elif action == 'unassign':
        store.assign(document.uid, STRUCTURES, False)
        # Another workspace owns its own independent structure settings.
        assert preview.camera.zoom == 2
        store.assign(document.uid, POLES, False)
    else:
        store.remove(document.uid)
    assert observations and all(item == (1, [], None) for item in observations)
    for view in (structures.viewer, preview):
        np.testing.assert_array_equal(view.user_rotation, np.eye(3))
        assert not view.atom_component_visibility and not view.polyhedron_opaque_sites
        assert view.atom_scale_percent == 145
    assert structures.pattern.fwhm == .23
    assert structures.pattern.viewport is None


def test_rsm_effective_sources_and_overlay_cleanup():
    store = ProjectStore()
    scans = [store.add_scan(Scan1D(str(i), np.arange(3.), np.ones(3), Path(f'{i}.xy')))
             for i in range(2)]
    for document in scans:
        store.assign(document.uid, RSM)
    state = store.rsm.experimental
    state.manual_angles, state.derived_angle = (20, .1, 20.1), 'last'
    state.data = 'cache'
    state.viewports['angular'] = ((20, 21), (40, 45))
    store.replace_scan(scans[0].uid, Scan1D('new', np.arange(3.), np.ones(3), Path('new.xy')))
    assert state.manual_angles is state.derived_angle is state.data is None
    assert not state.viewports
    raw = store.add_rsm_document('map.raw', SimpleNamespace(ranges=[]))
    store.assign(raw.uid, RSM)
    assert state.selected_sources() == [raw]
    state.data = 'raw cache'
    state.colour_map = 'plasma'
    store.assign(scans[0].uid, RSM, False)
    assert state.data == 'raw cache'
    store.assign(raw.uid, RSM, False)
    assert state.selected_sources() == [scans[1]] and state.data is None
    assert state.colour_map == 'plasma'
    phase_doc = store.add_cell_phase(phase())
    state.phase_uid, state.overlay_enabled = phase_doc.uid, True
    store.rsm.calculated.bind(phase_doc.payload, phase_doc.uid)
    store.remove(phase_doc.uid)
    assert state.phase_uid is None and not state.overlay_enabled
    assert store.rsm.calculated.uid is None


def test_workspace_copy_has_independent_numeric_state():
    store = ProjectStore()
    store.structures.viewer.atom_component_colours['site'] = '#123456'
    store.rsm.calculated.targets['hkl'] = (1, 0, 2)
    clone = copy.deepcopy((store.structures, store.rsm))
    clone[0].viewer.user_rotation[0, 0] = 2
    clone[0].viewer.atom_component_colours['site'] = '#ffffff'
    clone[1].calculated.targets['hkl'] = (2, 0, 2)
    assert store.structures.viewer.user_rotation[0, 0] == 1
    assert store.structures.viewer.atom_component_colours['site'] == '#123456'
    assert store.rsm.calculated.targets['hkl'] == (1, 0, 2)
