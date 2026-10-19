"""Native integration of the restored 3.0.2 interaction changes."""
from time import monotonic
import numpy as np
import pytest
from qt_test_support import PYSIDE_AVAILABLE, QApplication
from test_sessions import project
from test_qt_sessions import window, ready
from test_angular_intake import write_xrdml
from xrd_workbench.models.project import VIEWER, STRUCTURES, POLES, RSM
from xrd_workbench.models.pole_figure import CalculatedPoleLayer
from xrd_workbench.services.sessions import write_session, load_session
from xrd_workbench.services.rsm import RockingTrajectories

pytestmark=pytest.mark.skipif(not PYSIDE_AVAILABLE,reason='Native Qt unavailable')


def test_cell_phase_system_filters_preserve_group_and_parameters():
    from xrd_workbench.space_groups import SETTINGS, setting_from_user_text
    from xrd_workbench.ui_qt.project_panel import CellPhaseDialog
    app = QApplication.instance() or QApplication([])
    dialog = CellPhaseDialog(initial_cell=(4,5,6,80,90,100), initial_name='filtered')
    try:
        dialog.group_combo.setCurrentIndex(dialog.group_combo.findText(setting_from_user_text('75').label))
        before = tuple(field.value() for field in dialog.cell_inputs)
        text = dialog.group_combo.currentText()
        assert dialog.group_combo.count() == 530
        for systems in (('cubic',), ('cubic','orthorhombic'), ()):
            for system, check in dialog.system_checks.items():
                check.setChecked(system in systems)
            expected = [s.label for s in SETTINGS if not systems or s.crystal_system in systems]
            assert [dialog.group_combo.itemText(i) for i in range(dialog.group_combo.count())] == expected
            assert dialog.group_combo.currentText() == text
            assert tuple(field.value() for field in dialog.cell_inputs) == before
        # Choosing a listed group, rather than filtering, updates constraints.
        dialog.group_combo.setCurrentIndex(dialog.group_combo.findText(setting_from_user_text('195').label))
        assert [field.isEnabled() for field in dialog.cell_inputs] == [True,False,False,False,False,False]
        dialog.cell_inputs[0].setValue(7.25)
        assert tuple(field.value() for field in dialog.cell_inputs) == (7.25,7.25,7.25,90,90,90)
    finally:
        dialog.close()


def test_cell_phase_typed_groups_and_dependent_fields_round_trip(tmp_path):
    from PySide6.QtTest import QTest
    from xrd_workbench.localization import set_language
    from xrd_workbench.models.project import ProjectStore
    from xrd_workbench.models.radiation import RadiationSettings
    from xrd_workbench.space_groups import setting_from_user_text
    from xrd_workbench.ui_qt.project_panel import CellPhaseDialog
    from test_sessions import files
    app = QApplication.instance() or QApplication([])
    dialog = CellPhaseDialog(initial_name='constrained')
    try:
        dialog.show(); app.processEvents()
        for text, enabled in (('75',(0,2)), ('P2 [a]',(0,1,2,3)),
                              ('P2 [b]',(0,1,2,4)), ('P2 [c]',(0,1,2,5)),
                              ('R3c [H]',(0,2)), ('R3c [R]',(0,3))):
            edit = dialog.group_combo.lineEdit()
            edit.setFocus(); app.processEvents(); edit.selectAll(); QTest.keyClicks(edit, text)
            # Commit a typed symbol by moving focus, without accepting the dialog.
            dialog.name_edit.setFocus(); app.processEvents()
            assert tuple(i for i, field in enumerate(dialog.cell_inputs) if field.isEnabled()) == enabled
        dialog.cell_inputs[0].setValue(6.12)
        dialog.cell_inputs[3].setValue(78.5)
        assert tuple(field.value() for field in dialog.cell_inputs) == (6.12,6.12,6.12,78.5,78.5,78.5)
        dialog._accept()
        assert dialog.result_document.setting == setting_from_user_text('R3c [R]')
        store = ProjectStore()
        doc = store.add_cell_phase(dialog.result_document)
        path = tmp_path/'constrained.xrdproject'
        write_session(path, store, RadiationSettings())
        restored = load_session(path, files()).store.documents[doc.uid].payload
        assert restored.cell == dialog.result_document.cell
        assert restored.setting == dialog.result_document.setting
        # Invalid text remains a draft and cannot produce another document.
        dialog.group_combo.setEditText('not a group')
        dialog.result_document = None
        from unittest.mock import patch
        with patch('xrd_workbench.ui_qt.project_panel.QMessageBox.warning') as warning:
            dialog._accept()
        assert warning.called and dialog.result_document is None
        for language, expected in (('en','Cubic'),('fr','Cubique'),('ru','Кубическая')):
            set_language(language)
            localized = CellPhaseDialog()
            assert localized.system_checks['cubic'].text() == expected
            localized.close()
    finally:
        set_language('en')
        dialog.close()


def test_common_cif_layers_keep_independent_rotation_when_saved(window,tmp_path):
    ready(window)
    pole=window.pages[POLES].calculated
    state=pole.state
    uid=state.primary_uid
    doc=window.project.documents[uid]
    state.overlay_uid=uid
    state.overlay_layer=CalculatedPoleLayer(doc.payload,state.overlay_colour,state.overlay_opacity,state.overlay_size)
    state.overlay_layer.user_rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    path=tmp_path/'common.xrdproject'
    assert window.save_project_as(path),window.test_errors
    assert window.open_project(path),window.test_errors
    ready(window)
    restored=window.project.poles.calculated
    assert restored.primary_uid==restored.overlay_uid==uid
    np.testing.assert_allclose(restored.overlay_layer.user_rotation,[[0,-1,0],[1,0,0],[0,0,1]])
    assert restored.cif_document is restored.overlay_layer.document


@pytest.mark.parametrize('workspace',[VIEWER,STRUCTURES,POLES,RSM])
def test_reset_only_active_workspace_retains_sources_analysis_and_other_pages(window,workspace):
    ready(window)
    store=window.project
    identities={uid:id(doc.payload) for uid,doc in store.documents.items()}
    analysis=store.analysis
    old_pages=dict(window.pages)
    window.sections.setCurrentIndex(window.WORKSPACES.index(workspace))
    before=store._workspace_state(workspace)
    if workspace==POLES:before.calculated.labels=True
    elif workspace==VIEWER:before.plot.intensity_scale='log'
    elif workspace==STRUCTURES:before.viewer.camera.zoom=2
    elif workspace==RSM:before.calculated.labels=False
    assert window.reset_tab()
    assert window.pages[workspace] is not old_pages[workspace]
    assert store._workspace_state(workspace) is not before
    assert store.analysis is analysis
    assert {uid:id(doc.payload) for uid,doc in store.documents.items()}==identities
    for other,page in old_pages.items():
        if other!=workspace:assert window.pages[other] is page
    assert not window.test_errors


@pytest.mark.parametrize('joint',[False,True])
def test_shared_alignment_acts_on_selected_secondary_layer(window,joint):
    ready(window)
    page=window.pages[POLES].calculated
    page.joint_rotation_var.set(joint)
    page.joint_rotation_changed()
    group=next(g for g in page.overlay_layer.point_groups if np.linalg.norm(g[0].direction[:2])>.1)
    page._select_pole_group(1,group)
    before=page.user_rotation.copy()
    page.align_selected_pole(0.)
    selected=page._selected_group(1)[0]
    assert abs(((selected.phi+180)%360)-180)<1e-7
    assert np.allclose(before,page.user_rotation)==(not joint)


def test_resizable_dataset_columns_survive_session(window,tmp_path):
    ready(window)
    tree=window.pages[VIEWER].documents
    widths=(271,54,93,63)
    for i,width in enumerate(widths):tree.setColumnWidth(i,width)
    assert window.project.viewer.column_widths==widths
    path=tmp_path/'widths.xrdproject'
    assert window.save_project_as(path)
    assert window.open_project(path)
    assert tuple(window.pages[VIEWER].documents.columnWidth(i) for i in range(4))==widths


@pytest.mark.parametrize('renderer',['matplotlib','pyqtgraph'])
def test_measured_poles_draw_acquired_points_in_both_renderers(window,tmp_path,renderer):
    ready(window)
    window.plot_renderer_controller.set_mode(renderer,persist=False)
    path=tmp_path/'angular.xrdml';write_xrdml(path)
    document=window.file_service.load_path(window.project,path)[0]
    window.project.assign(document.uid,POLES)
    page=window.pages[POLES].experimental
    assert len(page.measurement.trajectories)==1
    assert not page.angle_section.isVisible()
    assert page.save_button.isEnabled()
    trace=page.measurement.trajectories[0]
    page._inspect_plot_position(np.deg2rad(trace.azimuth[0]),trace.tilt[0])
    assert 'angular' in page.cursor_text.get()
    page.scale_mode.set('log');page.draw_pole_figure()
    page.display_mode.set('solid');page.draw_pole_figure()
    assert not window.test_errors


def test_single_rocking_draws_in_angles_and_q_without_manual_series(window,tmp_path):
    ready(window)
    path=tmp_path/'rocking.xrdml';write_xrdml(path,'Omega',{'2Theta':40})
    document=window.file_service.load_path(window.project,path)[0]
    window.project.assign(document.uid,RSM)
    page=window.pages[RSM].experimental
    assert isinstance(page.data,RockingTrajectories)
    assert page.first.isHidden()
    page.view.setCurrentIndex(page.view.findData('q'));page.draw()
    assert isinstance(page.data,RockingTrajectories)
    assert not window.test_errors


def test_reset_returns_while_old_viewer_worker_is_still_running(window):
    from threading import Event
    from PySide6.QtCore import QRunnable
    ready(window)
    started, release=Event(),Event()
    class Waiting(QRunnable):
        def run(self):
            started.set();release.wait(5)
    old=window.pages[VIEWER]
    old._phase_thread_pool.start(Waiting())
    assert started.wait(2)
    window.sections.setCurrentIndex(0)
    try:
        begin=monotonic()
        assert window.reset_tab()
        assert monotonic()-begin<2
        assert old._phase_thread_pool.activeThreadCount()==1
    finally:
        release.set();old._phase_thread_pool.waitForDone(5000)
        QApplication.instance().processEvents()


def test_radiation_sync_reaches_replaced_rsm_page(window,monkeypatch):
    ready(window)
    window.sections.setCurrentIndex(window.WORKSPACES.index(RSM))
    assert window.reset_tab()
    replacement=window.pages[RSM]
    calls=[]
    monkeypatch.setattr(replacement,'radiation_changed',lambda:calls.append(True))
    window.pages[VIEWER].radiation_selector.radiation_changed.emit()
    assert calls==[True]


def test_rocking_mode_returns_to_editable_xy_series(window,tmp_path):
    ready(window)
    path=tmp_path/'rocking.xrdml';write_xrdml(path,'Omega',{'2Theta':40})
    doc=window.file_service.load_path(window.project,path)[0]
    window.project.assign(doc.uid,RSM)
    window.project.assign(doc.uid,RSM,False)
    documents=[]
    for index in range(2):
        path=tmp_path/f'pattern{index}.xy';np.savetxt(path,[[20,10],[21,20],[22,15]])
        documents.append(window.file_service.load_path(window.project,path)[0])
    window.project.assign(documents[0].uid,RSM)
    page=window.pages[RSM].experimental
    assert not page.first.isHidden()
    assert not page.first.isReadOnly()
    page.first.setText('9');page.step.setText('.1')
    window.project.assign(documents[1].uid,RSM)
    page.draw()
    assert not isinstance(page.data,RockingTrajectories)
    assert page.data.omega==pytest.approx([9,9.1])
    assert not window.test_errors


def test_full_raw_pole_import_keeps_parent_map_assignment(window,tmp_path):
    from test_bruker_raw_geometry import make_v3_range,write_v3
    path=tmp_path/'map.raw'
    write_v3(path,[make_v3_range(axis_code=5,theta=20,two_theta=40,chi=chi,phi=0)
                   for chi in (10,20)])
    window.import_paths([path],workspace=POLES)
    assigned=window.project.assigned_documents(POLES)
    measured=[doc for doc in assigned if doc.kind in {'scan','pole_data'}]
    assert len(measured)==1 and measured[0].kind=='pole_data'
    assert not window.test_errors


def test_alignment_footer_controls_fit_without_overlap(window):
    from PySide6.QtCore import QPoint,QRect
    ready(window)
    window.sections.setCurrentIndex(window.WORKSPACES.index(POLES))
    page=window.pages[POLES].calculated
    window.pages[POLES].tabs.setCurrentIndex(1)
    QApplication.instance().processEvents()
    footer=page.alignment_footer
    rectangles=[]
    for widget in [page.alignment_label,page.alignment_selection,*page.alignment_buttons,page.save_button]:
        rect=QRect(widget.mapTo(footer,QPoint(0,0)),widget.size())
        assert footer.rect().contains(rect)
        assert not any(rect.intersects(other) for other in rectangles)
        rectangles.append(rect)


def test_rsm_hkl_controls_fit_the_narrow_sidebar(window):
    from PySide6.QtCore import QPoint,QRect
    from PySide6.QtWidgets import QScrollArea
    ready(window)
    window.sections.setCurrentIndex(window.WORKSPACES.index(RSM))
    page=window.pages[RSM].experimental
    QApplication.instance().processEvents()
    panel=page.findChild(QScrollArea)
    viewport=panel.viewport()
    for editor in (page.surface,page.inplane):
        for field in editor.fields:
            rect=QRect(field.mapTo(viewport,QPoint(0,0)),field.size())
            assert rect.left()>=0 and rect.right()<viewport.width()
            assert field.width()>=30


@pytest.mark.parametrize('renderer',['matplotlib','pyqtgraph'])
@pytest.mark.parametrize('source_kind',['grid','trace'])
def test_experimental_poles_keep_full_hemisphere_and_unmeasured_area_empty(window,renderer,source_kind):
    from pole_fixtures import raw_fixture
    from test_angular_intake import scan
    from xrd_workbench.services.experimental_pole import load_experimental_pole,load_scan_poles
    ready(window)
    window.plot_renderer_controller.set_mode(renderer,persist=False)
    page=window.pages[POLES].experimental
    if source_kind=='grid':
        measurement=load_experimental_pole('partial.raw',raw_fixture())
        before=[values.copy() for _,values in measurement.scans]
    else:
        source=scan('Phi',**{'2Theta':40,'Omega':20,'Chi':45})
        before=[source.y.copy()]
        measurement=load_scan_poles([source])
    page.load_measurement(measurement)
    if renderer=='matplotlib':
        assert page.plot_axis.get_ylim()==pytest.approx((0,90))
    else:
        assert page.pyqtgraph_plot.radius_max==90
        for lower,upper in page.pyqtgraph_plot.view_box.viewRange():
            assert lower<=-90 and upper>=90
    after=([values for _,values in measurement.scans] if source_kind=='grid' else
           [trace.intensity for trace in measurement.trajectories])
    for old,new in zip(before,after):np.testing.assert_array_equal(old,new)
    page._inspect_plot_position(0.,85.)
    assert 'no measured point' in page.cursor_text.get().lower()
    page.reset_zoom()
    if renderer=='matplotlib':assert page.plot_axis.get_ylim()==pytest.approx((0,90))
    else:assert page.pyqtgraph_plot.radius_max==90


@pytest.mark.parametrize('extension',['raw','xrdml'])
@pytest.mark.parametrize('axis',['Phi','Chi'])
def test_pole_open_measurements_button_accepts_individual_scans(window,tmp_path,monkeypatch,extension,axis):
    from PySide6.QtWidgets import QFileDialog,QMessageBox
    from test_bruker_raw_geometry import make_v3_range,write_v3
    monkeypatch.setattr(QMessageBox,'critical',lambda *args:window.test_errors.append(args[1:]))
    path=tmp_path/f'individual.{extension}'
    if extension=='raw':
        write_v3(path,[make_v3_range(axis_code=5 if axis=='Phi' else 4,
                    theta=20,two_theta=40,chi=45,phi=0)])
    else:
        fixed={'2Theta':40,'Omega':20,'Chi' if axis=='Phi' else 'Phi':45}
        write_xrdml(path,axis,fixed)
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args,**kwargs:(str(path),''))
    workspace=window.pages[POLES]
    workspace.experimental.open_measurements_button.click()
    sources=window.project.poles.experimental.source_uids
    assert len(sources)==1,window.test_errors
    document=window.project.documents[sources[0]]
    assert document.source==path and window.project.is_assigned(document.uid,POLES)
    assert document.payload.axis_name==axis
    assert len(workspace.experimental.measurement.trajectories)==1
    assert workspace.experimental.save_button.isEnabled()
    assert workspace.tabs.currentIndex()==0
    assert not window.test_errors


@pytest.mark.parametrize('extension',['raw','xrdml'])
def test_rsm_open_measurements_button_accepts_individual_rocking(window,tmp_path,monkeypatch,extension):
    from PySide6.QtWidgets import QFileDialog
    from test_bruker_raw_geometry import make_v3_range,write_v3
    path=tmp_path/f'rocking.{extension}'
    if extension=='raw':
        write_v3(path,[make_v3_range(axis_code=3,theta=20,two_theta=40)])
    else:write_xrdml(path,'Omega',{'2Theta':40})
    monkeypatch.setattr(QFileDialog,'getOpenFileNames',lambda *args,**kwargs:([str(path)],''))
    page=window.pages[RSM].experimental
    page.open_raw_button.click()
    assert isinstance(page.data,RockingTrajectories),window.test_errors
    sources=window.project.assigned_documents(RSM,kind='scan')
    assert len(sources)==1 and sources[0].source==path
    assert not window.project.assigned_documents(RSM,kind='rsm_data')
    assert not window.test_errors


def test_direct_rsm_open_keeps_numbered_raw_detector_map_series(window,tmp_path,monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from test_bruker_raw_geometry import make_v3_range,write_v3
    from xrd_workbench.services.rsm import MapData
    paths=[]
    for index,omega in enumerate((10,11)):
        path=tmp_path/f'map__{index}.raw';paths.append(path)
        write_v3(path,[make_v3_range(axis_code=0,theta=omega,two_theta=20)])
    monkeypatch.setattr(QFileDialog,'getOpenFileNames',lambda *args,**kwargs:([str(paths[0])],''))
    page=window.pages[RSM].experimental
    page.open_raw_button.click()
    assert isinstance(page.data,MapData),window.test_errors
    np.testing.assert_array_equal(page.data.omega,[10,11])
    assert len(window.project.assigned_documents(RSM,kind='rsm_data'))==2
    assert not window.test_errors


def test_direct_pole_open_retains_numbered_xy_fallback(window,tmp_path,monkeypatch):
    from PySide6.QtWidgets import QFileDialog,QMessageBox
    monkeypatch.setattr(QMessageBox,'critical',lambda *args:window.test_errors.append(args[1:]))
    source=tmp_path/'legacy.raw';source.write_bytes(b'bad raw')
    for index in range(2):
        source.with_name(f'legacy_exported_{index}.xy').write_text('0 1\n90 2\n180 3\n')
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args,**kwargs:(str(source),''))
    page=window.pages[POLES].experimental
    page.open_measurements_button.click()
    assert len(window.project.assigned_documents(POLES,kind='pole_data'))==1
    assert len(page.measurement.scans)==2
    assert page.measurement.raw is None and not page.measurement.trajectories
    assert not window.test_errors


@pytest.mark.parametrize('workspace',[POLES,RSM])
def test_direct_open_rejects_missing_recorded_angles_without_discarding_current_assignments(window,tmp_path,monkeypatch,workspace):
    from PySide6.QtWidgets import QFileDialog,QMessageBox
    monkeypatch.setattr(QMessageBox,'critical',lambda *args:window.test_errors.append(args[1:]))
    path=tmp_path/'incomplete.xrdml'
    write_xrdml(path,'Phi' if workspace==POLES else 'Omega',{'2Theta':40} if workspace==POLES else {})
    assigned=set(window.project.assignments[workspace])
    if workspace==POLES:
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args,**kwargs:(str(path),''))
        window.pages[POLES].experimental.open_measurements_button.click()
    else:
        monkeypatch.setattr(QFileDialog,'getOpenFileNames',lambda *args,**kwargs:([str(path)],''))
        window.pages[RSM].experimental.open_raw_button.click()
    assert window.test_errors
    assert 'Missing recorded fixed angles' in str(window.test_errors)
    assert window.project.assignments[workspace]==assigned
    assert any(doc.source==path for doc in window.project.documents.values())
