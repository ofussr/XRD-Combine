"""Background loading, shared-cache and document-lifetime regressions."""

import os
from pathlib import Path
import tempfile
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QThreadPool, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from xrd_workbench import atom_styles
from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.localization import get_language, set_language
from xrd_workbench.models.project import STRUCTURES, VIEWER, ProjectStore
from xrd_workbench.services.pole_figure import rotation_z
from xrd_workbench.services.structure_scene import build_structure_scene
from xrd_workbench.ui_qt.main_window import MainWindow
from xrd_workbench.ui_qt.pole_structure import PoleStructureView
from xrd_workbench.ui_qt.structure_preparation import LoadingStructureDialog, StructurePreparation
from xrd_workbench.ui_qt.structure_viewer import StructureViewerPage


CIF_TEXT = """data_async
_chemical_formula_sum 'Nb O'
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Nb1 Nb 0.5 0.5 0.5
O1 O 0.25 0.5 0.5
"""


class StructurePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.release = Event()
        self.preparations = []
        self.addCleanup(self._finish_workers)
        self.old_language = get_language()
        set_language("en")
        self.addCleanup(set_language, self.old_language)

    def _finish_workers(self):
        self.release.set()
        for preparation in self.preparations:
            preparation._pool.shutdown(wait=True, cancel_futures=True)
        self.assertTrue(QThreadPool.globalInstance().waitForDone(5000))
        self.application.processEvents()

    def preparer(self, store=None):
        preparation = StructurePreparation(store)
        self.preparations.append(preparation)
        self.addCleanup(preparation.close)
        return preparation

    def dispose_widget(self, widget):
        widget.close()
        widget.deleteLater()
        self.application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()

    def payload(self, name="sample.cif"):
        path = Path(self.folder.name) / name
        path.write_text(CIF_TEXT, encoding="utf-8")
        return load_cif_document(path)

    def until(self, predicate):
        deadline = monotonic() + 5.0
        while not predicate() and monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), "Timed out waiting for a Qt worker result")

    def blocking_builder(self, calls, started):
        def build(crystal):
            calls.append((crystal, get_ident()))
            started.set()
            if not self.release.wait(5):
                raise RuntimeError("Test did not release the worker")
            return build_structure_scene(crystal)
        return build

    def test_viewer_import_prepares_in_background_and_tab_switch_reuses_job(self):
        payload = self.payload()
        window = MainWindow()
        self.preparations.append(window.structure_preparation)
        self.addCleanup(self.dispose_widget, window)
        window.show()
        self.application.processEvents()
        calls, started, ticks = [], Event(), []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(True))
        self.addCleanup(timer.stop)
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=self.blocking_builder(calls, started)):
            timer.start()
            document = window.import_paths([payload.source])[0]
            self.until(started.is_set)
            self.until(lambda: len(ticks) >= 3)
            self.assertEqual(window.current_workspace(), VIEWER)
            self.assertTrue(window.structure_loading_dialog.isVisible())
            self.assertEqual(window.structure_loading_dialog.label.text(), "Loading structure...")
            self.assertEqual(window.structure_loading_dialog.windowModality(), Qt.NonModal)
            self.assertNotEqual(calls[0][1], get_ident())
            window.open_structure(document.uid)
            page = window.pages[STRUCTURES].structure_viewer
            self.assertIsNone(page.scene)
            self.assertFalse(page.orientation_section.content.isEnabled())
            self.assertEqual(len(calls), 1)
            self.release.set()
            self.until(lambda: page.scene is not None)
            self.assertFalse(window.structure_loading_dialog.isVisible())
            if self.application.platformName() != "offscreen":
                self.until(lambda: page.canvas.context() is not None
                           and page.canvas.context().isValid())
                frame = page.canvas.grabFramebuffer()
                self.assertFalse(frame.isNull())
                self.assertIsNone(page.canvas.opengl_error)
                pixels = np.frombuffer(frame.constBits(), dtype=np.uint8)
                self.assertGreater(float(pixels.std()), 5.0)
            cached_scene = page.scene
            window.project.assign(document.uid, STRUCTURES, False)
            self.assertIsNone(page.scene)
            window.open_structure(document.uid)
            self.assertIs(page.scene, cached_scene)
            self.assertEqual(len(calls), 1)

    def test_multiple_requests_share_result_and_publish_on_gui_thread(self):
        document = self.payload()
        preparation = self.preparer()
        calls, started, ready = [], Event(), []
        preparation.ready.connect(lambda item: ready.append((item, get_ident())))
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=self.blocking_builder(calls, started)):
            self.assertIsNone(preparation.request(document))
            self.until(started.is_set)
            for _ in range(4):
                self.assertIsNone(preparation.request(document))
            self.release.set()
            self.until(lambda: bool(ready))
            result = preparation.request(document)
            self.assertIs(result, preparation.request(document))
            self.assertIs(result.geometry.crystal, document.crystal)
            self.assertEqual(ready, [(document, get_ident())])
            self.assertEqual(len(calls), 1)
            self.assertFalse(preparation.busy)

    def test_replaced_and_removed_payloads_cannot_publish_stale_results(self):
        store = ProjectStore()
        preparation = self.preparer(store)
        old, replacement, survivor = (self.payload(name) for name in
                                      ("old.cif", "replacement.cif", "survivor.cif"))
        calls, started, ready = [], Event(), []
        preparation.ready.connect(ready.append)
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=self.blocking_builder(calls, started)):
            phase = SimpleNamespace(source=old.source, name="Phase", crystal=old.crystal)
            document = store.add_cell_phase(phase)
            self.until(started.is_set)
            new_phase = SimpleNamespace(source=replacement.source, name="Replaced",
                                        crystal=replacement.crystal)
            store.replace_cell_phase(document.uid, new_phase)
            store.remove(document.uid)
            store.add_cif_document(survivor.source, survivor)
            self.release.set()
            self.until(lambda: not preparation.busy)
            self.assertEqual(ready, [survivor])
            self.assertEqual([item[0] for item in calls], [old.crystal, survivor.crystal])
            self.assertIsNone(preparation.error(phase))

    def test_pending_view_change_and_pole_rotation_survive_worker_completion(self):
        first, second = self.payload("first.cif"), self.payload("second.cif")
        preparation = self.preparer()
        viewer = StructureViewerPage(scene_preparer=preparation)
        pole = PoleStructureView(scene_preparer=preparation,
                                 on_orientation_changed=lambda *_: None,
                                 on_rotation_finished=lambda: None)
        self.addCleanup(self.dispose_widget, viewer)
        self.addCleanup(self.dispose_widget, pole)
        calls, started = [], Event()
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=self.blocking_builder(calls, started)):
            viewer.load_document(first)
            self.until(started.is_set)
            viewer.load_document(second)
            orientation = rotation_z(37)
            pole.sync_document(second, orientation)
            self.release.set()
            self.until(lambda: viewer.scene is not None and pole.scene is not None)
            self.assertIs(viewer.crystal, second.crystal)
            self.assertIs(viewer.scene, pole.scene)
            np.testing.assert_allclose(pole.canvas.orientation, orientation)
            self.assertEqual(len(calls), 2)

    def test_failed_geometry_closes_notice_and_next_document_still_loads(self):
        bad, good = self.payload("bad.cif"), self.payload("good.cif")
        preparation = self.preparer()
        viewer = StructureViewerPage(scene_preparer=preparation)
        self.addCleanup(self.dispose_widget, viewer)
        notice = LoadingStructureDialog()
        self.addCleanup(self.dispose_widget, notice)
        preparation.busy_changed.connect(notice.set_busy)
        failures = []
        preparation.failed.connect(lambda item, error: failures.append((item, error)))
        def build(crystal):
            if crystal is bad.crystal:
                raise ValueError("Invalid geometry")
            return build_structure_scene(crystal)
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=build) as builder:
            viewer.load_document(bad)
            self.assertTrue(notice.isVisible())
            self.until(lambda: bool(failures))
            self.assertEqual(failures, [(bad, "Invalid geometry")])
            self.assertFalse(preparation.busy)
            self.assertFalse(notice.isVisible())
            self.assertFalse(viewer._loading_structure)
            self.assertFalse(viewer.display_section.content.isEnabled())
            preparation.request(bad)
            self.assertEqual(builder.call_count, 1)
            viewer.load_document(good)
            self.until(lambda: viewer.scene is not None)
            self.assertIs(viewer.crystal, good.crystal)
            self.assertEqual(builder.call_count, 2)

    def test_close_during_geometry_returns_without_waiting_or_late_ui_updates(self):
        document = self.payload()
        window = MainWindow()
        self.preparations.append(window.structure_preparation)
        self.addCleanup(self.dispose_widget, window)
        window.show()
        self.application.processEvents()
        preparation = window.structure_preparation
        calls, started, ready = [], Event(), []
        preparation.ready.connect(ready.append)
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=self.blocking_builder(calls, started)):
            window.project.add_cif_document(document.source, document)
            self.until(started.is_set)
            window.close()
            # The worker remains blocked here: close must not wait for it.
            self.assertFalse(self.release.is_set())
            self.assertFalse(preparation.busy)
            self.assertFalse(window.structure_loading_dialog.isVisible())
            self.release.set()
            self.until(lambda: preparation._active_task[1].done())
            self.application.processEvents()
            self.assertEqual(ready, [])
            self.assertIsNone(preparation.request(document))

    def test_palette_changed_during_preparation_refreshes_cached_scene_only(self):
        document = self.payload()
        preparation = self.preparer()
        calls, started = [], Event()
        with patch("xrd_workbench.ui_qt.structure_preparation.build_structure_scene",
                   side_effect=self.blocking_builder(calls, started)):
            preparation.request(document)
            self.until(started.is_set)
            with patch.dict(atom_styles._custom, {"Nb": "#123456"}):
                self.release.set()
                self.until(lambda: not preparation.busy)
                prepared = preparation.request(document)
                colours = {component.colour for atom in prepared.scene.atoms
                           for component in atom.components if component.element == "Nb"}
                self.assertEqual(colours, {"#123456"})
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
