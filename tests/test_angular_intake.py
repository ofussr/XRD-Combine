"""Real readers, geometry and portable state for acquired angular trajectories."""
from pathlib import Path
import numpy as np
import pytest
from test_sessions import files
from test_bruker_raw_geometry import make_v3_range, write_v3
from xrd_workbench.models.project import ProjectStore, VIEWER, POLES, RSM
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.models.radiation import RadiationSettings
from xrd_workbench.models.scan_admission import scan_assignment_error
from xrd_workbench.services.experimental_pole import load_scan_poles
from xrd_workbench.services.rsm import build_rocking_traces, rocking_coordinates
from xrd_workbench.services.sessions import write_session, load_session, merge_tab


def scan(axis='Phi', **fixed):
    x = np.linspace(0, 90, 7)
    axes = {axis: x}
    axes.update({name: np.full(7, value) for name, value in fixed.items()})
    return Scan1D('angular', x, np.arange(7)+1, Path('angular.xrdml'), axis_name=axis, metadata={'axes': axes})


def write_xrdml(path, axis='Phi', fixed=None):
    fixed = fixed if fixed is not None else {'2Theta': 40, 'Omega': 20, 'Chi': 45}
    positions = f'<positions axis="{axis}"><startPosition>0</startPosition><endPosition>90</endPosition></positions>'
    positions += ''.join(f'<positions axis="{name}"><commonPosition>{value}</commonPosition></positions>' for name, value in fixed.items())
    path.write_text('<xrdMeasurements><xrdMeasurement><scan><dataPoints>'+positions+
        '<intensities>1 2 3 4 5 6 7</intensities></dataPoints></scan></xrdMeasurement></xrdMeasurements>')


@pytest.mark.parametrize('axis,workspace,fixed', [
    ('Phi', POLES, {'2Theta':40,'Omega':20,'Chi':0}),
    ('Chi', POLES, {'2Theta':40,'Omega':20,'Phi':0}),
    ('Theta', RSM, {'2Theta':40}),
    ('Omega', RSM, {'2Theta':40})])
def test_reader_accepts_recorded_zero_and_scan_geometry(tmp_path, axis, workspace, fixed):
    path=tmp_path/'scan.xrdml';write_xrdml(path,axis,fixed)
    store=ProjectStore();doc=files().load_path(store,path)[0]
    assert doc.payload.axis_name == axis
    assert store.can_assign(doc.uid, workspace)
    store.assign(doc.uid, workspace)
    assert store.is_assigned(doc.uid,workspace)


@pytest.mark.parametrize('axis,workspace,fixed', [
    ('Phi', POLES, {'2Theta':40,'Omega':20}),
    ('Chi', POLES, {'2Theta':40,'Omega':20}),
    ('Omega', RSM, {}),
    ('Z', POLES, {'2Theta':40,'Omega':20,'Chi':0}),
    ('Phi', RSM, {'2Theta':40,'Omega':20,'Chi':0})])
def test_missing_metadata_or_wrong_axis_stays_unassigned(tmp_path,axis,workspace,fixed):
    path=tmp_path/'scan.xrdml';write_xrdml(path,axis,fixed)
    store=ProjectStore();doc=files().load_path(store,path)[0]
    assert not store.can_assign(doc.uid,workspace)
    with pytest.raises(ValueError):store.assign(doc.uid,workspace)
    assert not store.is_assigned(doc.uid,workspace)
    store.assign(doc.uid,VIEWER)


@pytest.mark.parametrize('axis_code,axis,workspace', [(5,'Phi',POLES),(4,'Chi',POLES),(3,'Theta',RSM)])
def test_raw_reader_admits_single_angular_range(tmp_path,axis_code,axis,workspace):
    path=tmp_path/'single.raw'
    write_v3(path,[make_v3_range(axis_code=axis_code,theta=20,two_theta=40,chi=0,phi=0)])
    store=ProjectStore();documents=files().load_path(store,path)
    doc=next(d for d in documents if d.kind=='scan')
    assert doc.payload.axis_name==axis
    assert store.can_assign(doc.uid,workspace)


def test_phi_path_is_constant_tilt_and_rotates_azimuth_with_samples():
    source=scan('Phi',**{'2Theta':40,'Omega':20,'Chi':45})
    trace=load_scan_poles([source]).trajectories[0]
    np.testing.assert_allclose(trace.tilt,45)
    np.testing.assert_allclose((trace.azimuth-trace.azimuth[0])%360,source.x,atol=1e-10)
    np.testing.assert_array_equal(trace.intensity,source.y)
    assert len(trace.coordinate)==len(source.x)


def test_rocking_trace_uses_fixed_detector_and_constant_q_length():
    source=scan('Omega',**{'2Theta':40})
    omega,tt,values=build_rocking_traces([source]).traces[0]
    x,z,*_=rocking_coordinates(omega,tt,'q',1.5406)
    np.testing.assert_allclose(np.hypot(x,z),4*np.pi/1.5406*np.sin(np.deg2rad(20)))
    np.testing.assert_array_equal(values,source.y)


@pytest.mark.parametrize('scope',['project',POLES])
def test_measured_paths_roundtrip_and_tab_merge(tmp_path,scope):
    store=ProjectStore();service=files();ids=[]
    for index in range(2):
        path=tmp_path/f'scan{index}.xrdml';write_xrdml(path)
        doc=service.load_path(store,path)[0];ids.append(doc.uid);store.assign(doc.uid,POLES)
    store.poles.experimental.scale_mode='square'
    target=tmp_path/'paths.xrdproject';write_session(target,store,RadiationSettings(),scope=scope)
    restored=load_session(target,service)
    if scope==POLES:restored=merge_tab(ProjectStore(),restored)
    exp=restored.store.poles.experimental
    assert len(exp.source_uids)==2
    assert len(exp.measurement.trajectories)==2
    assert exp.scale_mode=='square'
    assert exp.uid==exp.source_uids[0]


def test_rocking_and_grid_modes_replace_only_assignments(tmp_path):
    store=ProjectStore()
    rocking=store.add_scan(scan('Omega',**{'2Theta':40}))
    xy=store.add_scan(Scan1D('xy',[20,21],[1,2],tmp_path/'xy.xy',metadata={'format':'XY'},axis_name='X'))
    store.assign(rocking.uid,RSM);store.assign(xy.uid,RSM)
    assert set(store.assignments[RSM])=={xy.uid}
    assert rocking.uid in store.documents
    store.assign(rocking.uid,RSM)
    assert set(store.assignments[RSM])=={rocking.uid}


def test_legacy_unsupported_rsm_assignment_keeps_document_and_loads_other_workspaces(tmp_path):
    import json
    path=tmp_path/'phi.xrdml';write_xrdml(path)
    store=ProjectStore();service=files();doc=service.load_path(store,path)[0]
    store.assign(doc.uid,VIEWER)
    saved=tmp_path/'legacy.xrdproject';write_session(saved,store,RadiationSettings())
    manifest=json.loads(saved.read_text())
    manifest['assignments'][RSM]=[doc.uid]  # Allowed by the 3.0.1 project tree.
    saved.write_text(json.dumps(manifest))
    loaded=load_session(saved,service)
    assert loaded.store.is_assigned(doc.uid,VIEWER)
    assert not loaded.store.is_assigned(doc.uid,RSM)
    assert doc.uid in loaded.store.documents
    assert loaded.warnings
