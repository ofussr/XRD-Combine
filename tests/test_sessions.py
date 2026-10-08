"""Portable file sessions: scientific reconstruction and failure isolation."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pole_fixtures import CIF_TEXT
from test_bruker_raw_geometry import make_v3_range, write_v3
from xrd_workbench.bruker_raw import read_bruker_raw
from xrd_workbench.cell_phase import create_cell_phase_document
from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.io import read_raw_scans, read_scan_file
from xrd_workbench.io.session_codec import SessionFormatError, encode
from xrd_workbench.models.background import BackgroundAnchors
from xrd_workbench.models.correction import CorrectionRequest
from xrd_workbench.models.phase_scan import AngleRange, PhaseOrientation
from xrd_workbench.models.project import ProjectStore, VIEWER, STRUCTURES, POLES, RSM
from xrd_workbench.models.radiation import RadiationSettings
from xrd_workbench.models.scan import assign_text_axis
from xrd_workbench.models.viewer import ViewerViewport
from xrd_workbench.services.correction import apply_correction
from xrd_workbench.services.experimental_pole import load_experimental_pole
from xrd_workbench.services.project_files import ProjectFileService
from xrd_workbench.services.sessions import (write_session, load_session, read_manifest,
    merge_tab, snapshot, SessionCancelled)
from xrd_workbench.space_groups import BY_HALL_NUMBER


def files():
    return ProjectFileService(load_cif_document=load_cif_document, read_bruker_raw=read_bruker_raw,
        read_raw_scans=read_raw_scans, read_scan_file=read_scan_file)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / 'источники'
    path.mkdir()
    xy, cif = path/'scan.xy', path/'silicon.cif'
    np.savetxt(xy, np.c_[np.linspace(20, 60, 101), np.linspace(10, 40, 101)])
    cif.write_text(CIF_TEXT)
    store, service = ProjectStore(), files()
    scan = service.load_path(store, xy)[0]
    assign_text_axis(scan.payload, '2Theta', assumed=False)
    phase = service.load_path(store, cif)[0]
    cell = store.add_cell_phase(create_cell_phase_document('ручная ячейка', BY_HALL_NUMBER[1], (4, 4, 4, 90, 90, 90)))
    for doc in (scan, phase, cell):
        store.assign(doc.uid, VIEWER)
    store.assign(phase.uid, STRUCTURES)
    store.assign(phase.uid, POLES)
    store.assign(cell.uid, POLES, additive=True)
    store.assign(scan.uid, RSM)
    store.assign(phase.uid, RSM)
    # A derived curve and in-place corrections must replay exactly.
    derived = store.add_scan(apply_correction(scan.payload, CorrectionRequest(.2, 1.03, -2, 1.5)), derived=True, parent_uid=scan.uid)
    store.assign(derived.uid, VIEWER)
    store.replace_scan(derived.uid, apply_correction(derived.payload, CorrectionRequest(-.1, .97, 4, 2)))
    item = store.viewer.items[derived.uid]
    item.visible, item.colour, item.x_shift = False, '#123456', .11
    store.viewer.selected_uid = derived.uid
    store.viewer.plot.intensity_scale = 'linear'
    store.viewer.plot.viewport = ViewerViewport((22, 35), (1, 90))
    store.viewer.phase_scan.reference_uid = scan.uid
    store.viewer.phase_scan.phase_uid = phase.uid
    store.viewer.phase_scan.angles[(scan.uid, '2Theta')] = {'chi': AngleRange(44.8, 45.2)}
    store.viewer.phase_scan.set_orientation(phase.uid, PhaseOrientation((1, 1, 1), (1, -1, 0), 12.3))
    x = np.linspace(27, 29, 9)
    first = store.analysis.add_peak(derived.uid, '2Theta', source_center=28, source_x=x,
        source_background=x*.1, source_profile=np.exp(-(x-28)**2), source_height=8, source_area=12,
        source_fwhm=.3, hkl_assignments=((phase.uid, 1, 1, 1),))
    store.analysis.add_peak(derived.uid, '2Theta', source_center=28.5, source_x=x,
        source_background=x*.1, source_profile=np.exp(-(x-28.5)**2), source_height=4, source_area=6,
        source_fwhm=.3, kind='ka2', parent_number=first.number)
    store.analysis.set_background(derived.uid, BackgroundAnchors('2Theta', .5, x, x*.1).with_local_level(28, 3, .5))
    store.analysis.reserve_peak_numbers(51)
    store.structures.viewer.camera.zoom = 1.8
    store.structures.viewer.atom_component_visibility = {'Si1:Si': False}
    store.structures.viewer.view_name = 'a*'
    store.structures.pattern.viewport = ((21, 45), (0, 100))
    store.poles.calculated.selected_hkl = (2, 2, 0)
    store.poles.calculated.preview.camera.pan = (.1, -.2)
    store.rsm.experimental.manual_angles = (9, .1, 12)
    store.rsm.experimental.derived_angle = 'last'
    store.rsm.experimental.phase_uid = cell.uid
    store.rsm.experimental.overlay_enabled = True
    store.rsm.calculated.has_calculated = True
    store.rsm.calculated.targets['hkl'] = (2, 2, 0)
    store.rsm.calculated.viewports['angular'] = ((30, 31), (60, 61))
    return store, service, scan, phase, cell, derived


def test_project_round_trip_replays_data_and_analysis_without_sources(project, tmp_path):
    store, service, scan, phase, cell, derived = project
    target = tmp_path/'work.xrdproject'
    write_session(target, store, RadiationSettings(), application={'workspace': POLES})
    text = target.read_text()
    assert 'Si1 Si' not in text and 'raw_range_data' not in text and '_base_y' not in text
    restored = load_session(target, service).store
    assert restored.assignments == store.assignments
    assert list(restored.documents) == list(store.documents)
    np.testing.assert_allclose(restored.documents[derived.uid].payload.x, derived.payload.x)
    np.testing.assert_allclose(restored.documents[derived.uid].payload.y, derived.payload.y)
    assert restored.documents[derived.uid].parent_uid == scan.uid
    assert restored.source_document('scan', scan.source).uid == scan.uid
    assert restored.viewer.items[derived.uid].visible is False
    assert restored.viewer.items[derived.uid].x_shift == .11
    assert restored.viewer.plot.viewport == store.viewer.plot.viewport
    assert restored.viewer.phase_scan.angles == store.viewer.phase_scan.angles
    assert restored.viewer.phase_scan.orientations == store.viewer.phase_scan.orientations
    assert restored.structures.viewer.camera.zoom == 1.8
    assert restored.structures.viewer.view_name == 'a*'
    assert restored.poles.calculated.overlay_layer.document is restored.documents[cell.uid].payload
    assert restored.rsm.experimental.overlay_enabled
    assert restored.rsm.experimental.manual_angles == (9, .1, 12)
    assert restored.rsm.calculated.request_key is not None
    assert restored.analysis.next_peak_number == 51
    assert restored.analysis.peaks[1].parent_number == restored.analysis.peaks[0].number
    assert restored.analysis.peaks[0].hkl_assignments == ((phase.uid, 1, 1, 1),)
    assert not restored.analysis.peaks[0].source_profile.flags.writeable
    np.testing.assert_allclose(restored.analysis.background_for(derived.uid, '2Theta').values(np.array([28.])), [3.])


@pytest.mark.parametrize('scope', [VIEWER, STRUCTURES, POLES, RSM])
def test_tab_import_replaces_only_its_workspace_and_reuses_verified_sources(project, tmp_path, scope):
    store, service, *docs = project
    target = tmp_path/'tab.xrdtab'
    write_session(target, store, RadiationSettings(), scope=scope)
    incoming = load_session(target, service)
    before = {ws: encode(getattr(store, ws)) for ws in store.assignments}
    merged = merge_tab(store, incoming).store
    for ws in store.assignments:
        assert merged.assignments[ws] == store.assignments[ws]
        assert encode(getattr(merged, ws)) == before[ws]
    assert len(merged.documents) == len(store.documents)
    assert len(merged.analysis.peaks) == 2
    # Reassigning the merged workspace cannot change the original live model.
    uid = next(iter(merged.assignments[scope]))
    merged.assign(uid, scope, False)
    assert uid in store.assignments[scope]


def test_folder_move_relative_sources_work_with_originals_removed(project, tmp_path):
    store, service, *_ = project
    source = tmp_path/'work.xrdproject'
    write_session(source, store, RadiationSettings())
    moved = tmp_path/'moved'
    moved.mkdir()
    shutil.move(str(tmp_path/'источники'), moved/'источники')
    shutil.move(str(source), moved/source.name)
    loaded = load_session(moved/source.name, service)
    assert len(loaded.store.documents) == len(store.documents)
    assert all(doc.source.parent == moved/'источники' for doc in loaded.store.documents.values() if doc.kind != 'cell_phase')


def test_missing_source_skip_relink_and_cancel(project, tmp_path):
    store, service, scan, phase, cell, derived = project
    target = tmp_path/'work.xrdproject'
    write_session(target, store, RadiationSettings())
    relocated = tmp_path/'renamed.xy'
    scan.source.rename(relocated)
    loaded = load_session(target, service, resolver=lambda issue: None)
    assert scan.uid not in loaded.store.documents and derived.uid not in loaded.store.documents
    assert not loaded.store.analysis.peaks
    assert loaded.warnings
    linked = load_session(target, service, resolver=lambda issue: relocated)
    assert linked.store.documents[scan.uid].source == relocated
    assert len(linked.store.analysis.peaks) == 2
    def cancel(issue):
        raise SessionCancelled()
    with pytest.raises(SessionCancelled):
        load_session(target, service, resolver=cancel)
    assert len(store.analysis.peaks) == 2


def test_changed_source_requires_choice_and_discards_stale_results(project, tmp_path):
    store, service, scan, phase, cell, derived = project
    target = tmp_path/'work.xrdproject'
    write_session(target, store, RadiationSettings())
    np.savetxt(scan.source, np.c_[np.linspace(20, 60, 101), np.full(101, 100.)])
    issues = []
    def use_changed(issue):
        issues.append(issue.reason)
        return issue.path, True
    loaded = load_session(target, service, resolver=use_changed)
    assert issues == ['changed'] # Both original and derivative share one source.
    assert not loaded.store.analysis.peaks
    assert loaded.store.viewer.items[derived.uid].x_shift == 0
    assert (scan.uid, '2Theta') not in loaded.store.viewer.phase_scan.angles
    assert loaded.store.rsm.experimental.manual_angles is None
    assert len(store.analysis.peaks) == 2


@pytest.mark.parametrize('format', ['raw', 'xrdml'])
def test_multi_axis_correction_replays_per_axis(tmp_path, format):
    raw = tmp_path/f'map.{format}'
    if format == 'raw':
        write_v3(raw, [make_v3_range(axis_code=3, theta=20, two_theta=40), make_v3_range(axis_code=3, theta=20, two_theta=41)])
    else:
        from test_core import ROCKING_XRDML
        raw.write_text(ROCKING_XRDML)
    store, service = ProjectStore(), files()
    docs = service.load_path(store, raw)
    scan = next(doc for doc in docs if doc.kind == 'scan')
    corrected = apply_correction(scan.payload, CorrectionRequest(.3, 1.01, -1, 3))
    axes = corrected.available_axes
    if '2Theta' in axes:
        corrected.use_axis('2Theta')
    corrected = apply_correction(corrected, CorrectionRequest(-.2, 1, 2, .5))
    store.replace_scan(scan.uid, corrected)
    store.assign(scan.uid, VIEWER)
    for doc in docs:
        if doc.kind == 'rsm_data':
            store.assign(doc.uid, RSM)
    target = tmp_path/'raw.xrdproject'
    write_session(target, store, RadiationSettings())
    restored = load_session(target, service).store.documents[scan.uid].payload
    for axis, values in corrected.metadata['axes'].items():
        np.testing.assert_allclose(restored.metadata['axes'][axis], values)
    np.testing.assert_allclose(restored.y, corrected.y)


def test_exported_xy_pole_fallback_round_trip(tmp_path):
    raw = tmp_path/'missing.raw'
    for index in range(3):
        np.savetxt(tmp_path/f'missing_exported_{index}.xy', [[0, 1], [90, 3], [180, 2], [360, 1]])
    store, service = ProjectStore(), files()
    doc = store.add_pole_document(raw, load_experimental_pole(raw))
    store.assign(doc.uid, POLES)
    store.poles.experimental.manual_angles = (0, 10, 20)
    target = tmp_path/'pole.xrdproject'
    write_session(target, store, RadiationSettings())
    restored = load_session(target, service).store.poles.experimental
    assert restored.manual_angles == (0, 10, 20)
    assert restored.measurement.raw is None and len(restored.measurement.scans) == 3
    original = tmp_path/'missing_exported_0.xy'
    renamed = tmp_path/'renamed_export.xy'
    original.rename(renamed)
    loaded = load_session(target, service, resolver=lambda issue: renamed)
    assert loaded.store.poles.experimental.measurement.source_files[0] == renamed
    write_session(target, loaded.store, loaded.radiation)
    assert load_session(target, service).store.poles.experimental.manual_angles == (0, 10, 20)


@pytest.mark.parametrize('fault', ['schema', 'model', 'viewport', 'parent', 'assignment', 'application', 'recipe'])
def test_bad_manifest_never_replaces_existing_save(project, tmp_path, fault):
    store, service, scan, phase, cell, derived = project
    path = tmp_path/'bad.xrdproject'
    data = snapshot(store, RadiationSettings(), path)
    if fault == 'schema':
        data['schema_version'] = 999
    elif fault == 'model':
        data['states'][VIEWER]['model'] = 'os.system'
    elif fault == 'viewport':
        data['states'][RSM]['fields']['calculated']['fields']['viewports']['angular'] = encode(((3, 2), (0, 1)))
    elif fault == 'parent':
        data['documents'][0]['parent_uid'] = data['documents'][0]['uid']
    elif fault == 'assignment':
        data['assignments'][STRUCTURES] = [scan.uid]
    elif fault == 'application':
        data['application'] = {'language': 'unknown'}
    else:
        data['documents'][0]['recipe']['steps'] = [{'axis_name': 'X', 'y_factor': -1}]
    path.write_text(json.dumps(data))
    with pytest.raises(SessionFormatError):
        load_session(path, service)
    assert len(store.analysis.peaks) == 2
    path.write_text('previous save')
    store.viewer.items[scan.uid].x_scale = -1
    with pytest.raises(SessionFormatError):
        write_session(path, store, RadiationSettings())
    assert path.read_text() == 'previous save'
    assert not list(tmp_path.glob('*.tmp'))


def test_unbounded_angular_pole_range_and_unreadable_source(project, tmp_path):
    store, service, scan, phase, cell, derived = project
    store.poles.calculated.range_mode = 'two_theta'
    store.poles.calculated.two_theta_range = (0., 60.)
    store.poles.calculated.d_range = (1.54056, float('inf'))
    path = tmp_path/'angular.xrdproject'
    write_session(path, store, RadiationSettings())
    assert 'Infinity' not in path.read_text()
    restored = load_session(path, service)
    assert restored.store.poles.calculated.d_range[1] == float('inf')
    store.poles.calculated.d_range = (1.54056, 1.54056)
    store.poles.calculated.two_theta_range = (60., 60.)
    write_session(path, store, RadiationSettings())
    assert load_session(path, service).store.poles.calculated.d_range == (1.54056, 1.54056)
    scan.source.write_text('unreadable scan')
    reasons = []
    def skip_unreadable(issue):
        reasons.append(issue.reason)
        return (issue.path, True) if issue.reason == 'changed' else None
    loaded = load_session(path, service, resolver=skip_unreadable)
    assert reasons == ['changed', 'unreadable']
    assert scan.uid not in loaded.store.documents and derived.uid not in loaded.store.documents
