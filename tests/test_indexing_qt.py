import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import tempfile
from threading import Event, get_ident
from time import monotonic
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, QSettings, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from xrd_workbench.indexing.models import IndexingCancelled
from xrd_workbench.localization import set_language
from xrd_workbench.models.project import ProjectStore, VIEWER, CELL_PHASE
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.ui_qt.debug_dialog import DebugDialog
from xrd_workbench.ui_qt.debug_features import DebugFeatures
from xrd_workbench.ui_qt.main_window import MainWindow
from xrd_workbench.ui_qt.peak_table import SessionPeak
from xrd_workbench.ui_qt.project_panel import CellPhaseDialog
from xrd_workbench.ui_qt.theme import ThemeController
from indexing_fixtures import NIST_SI_ANGLES, NIST_SI_WAVELENGTH


class IndexingQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.settings = QSettings(str(Path(self.folder.name) / 'settings.ini'), QSettings.Format.IniFormat)
        self.theme = ThemeController(self.application, self.settings)
        self.store = ProjectStore()
        self.window = MainWindow(store=self.store, theme_controller=self.theme)
        self.window._confirm_exit = lambda: True
        self.viewer = self.window.pages[VIEWER]
        set_language('en')
        self.window.retranslate()
        self.window.show()
        x = np.linspace(20, 145, 800)
        self.document = self.store.add_scan(Scan1D('Si reference', x, np.ones_like(x), Path('reference.xy')))
        self.store.assign(self.document.uid, VIEWER, True)
        self.viewer.select_uid(self.document.uid)
        self.viewer._session_peaks = [SessionPeak(10 + i, self.document.uid, '2Theta', angle,
            np.array([angle]), np.zeros(1), np.ones(1), 1, 1, .1) for i, angle in enumerate(NIST_SI_ANGLES)]
        self.viewer.open_peak_table()
        self.table = self.viewer._peak_table_dialogs[self.document.uid]
        self.application.processEvents()

    def tearDown(self):
        self.window.close()
        self.viewer._phase_thread_pool.waitForDone(5000)
        self.window.deleteLater()
        self.theme.set_mode('system', persist=False)
        self.theme.deleteLater()
        self.application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()
        set_language('en')
        self.folder.cleanup()

    def wait_until(self, predicate, seconds=10):
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            self.application.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        self.fail('Timed out waiting for indexing')

    def open_indexing(self):
        self.window.debug_features.set_indexation_enabled(True)
        self.viewer.open_indexing(self.document.uid)
        self.application.processEvents()
        return self.viewer._indexing_dialogs[self.document.uid]

    def configure_silicon(self, dialog):
        for system, check in dialog.system_checks.items():
            check.setChecked(system == 'cubic')
        dialog.min_length.setValue(5.3)
        dialog.max_length.setValue(5.5)
        dialog.hkl_limit.setValue(8)
        dialog.position_error.setValue(.001)
        dialog.wavelength.setValue(NIST_SI_WAVELENGTH)

    def test_debug_opt_in_cancel_persistence_and_live_visibility(self):
        self.assertFalse(self.window.debug_features.indexation_enabled)
        self.assertTrue(self.table.index_button.isHidden())
        self.viewer.open_indexing(self.document.uid)
        self.assertFalse(self.viewer._indexing_dialogs)
        def rejected(dialog):
            dialog.indexation_check.setChecked(True)
            return QDialog.DialogCode.Rejected
        with patch.object(DebugDialog, 'exec', rejected):
            self.window.open_debug()
        self.assertFalse(self.window.debug_features.indexation_enabled)
        self.assertTrue(self.table.index_button.isHidden())
        def accepted(dialog):
            self.assertEqual(dialog.indexation_check.text(), 'Toggle indexation')
            dialog.indexation_check.setChecked(True)
            return QDialog.DialogCode.Accepted
        with patch.object(DebugDialog, 'exec', accepted):
            self.window.open_debug()
        self.assertFalse(self.table.index_button.isHidden())
        restored = DebugFeatures(self.settings)
        self.assertTrue(restored.indexation_enabled)
        restored.deleteLater()
        self.window.debug_features.set_indexation_enabled(False)
        self.assertTrue(self.table.index_button.isHidden())

    def test_snapshot_uses_source_coordinates_primary_peaks_and_explicit_error(self):
        item = self.viewer.items[self.document.uid]
        item.x_shift, item.x_scale, item.y_factor = 10, 2, 3
        self.viewer.viewer_state.plot.x_display_mode = 'd'
        self.viewer._session_peaks.append(replace(self.viewer._session_peaks[0], number=999, kind='ka2'))
        dialog = self.open_indexing()
        self.assertEqual(len(dialog.peaks), 11)
        self.assertEqual(dialog.peaks[0].position, NIST_SI_ANGLES[0])
        self.assertEqual(dialog.peaks[0].peak_id, 10)
        self.assertIsNone(dialog.peaks[0].uncertainty)
        self.assertEqual(dialog.peaks[0].intensity, 1)
        dialog.position_error.setValue(.001)
        self.assertEqual(dialog._request().peaks[0].uncertainty, .001)
        self.assertEqual(dialog.input_kind, 'two_theta')
        for language in ('ru', 'fr', 'en'):
            set_language(language)
            self.window.retranslate()
            self.assertNotIn('qt.', dialog.windowTitle())

    def test_real_background_indexing_create_phase_and_hkl_transfer(self):
        dialog = self.open_indexing()
        self.configure_silicon(dialog)
        dialog.start()
        self.wait_until(lambda: dialog._future is None)
        self.assertIsNotNone(dialog.result, dialog.status.text())
        self.assertEqual(len(dialog.result.candidates[0].indexed_lines), 11)
        self.assertTrue(dialog.create_button.isEnabled())
        # Exact label is checked through the setting, avoiding formatting coupling.
        def accept(editor):
            from xrd_workbench.space_groups import setting_from_user_text
            self.assertEqual(setting_from_user_text(editor.group_combo.currentText()).number, 1)
            self.assertAlmostEqual(editor.cell_inputs[0].value(), 5.431144, places=4)
            editor._accept()
            return QDialog.DialogCode.Accepted
        with patch.object(CellPhaseDialog, 'exec', accept):
            dialog._create_phase()
        phases = [p for p in self.store.documents.values() if p.kind == CELL_PHASE]
        self.assertEqual(len(phases), 1)
        self.assertIn(phases[0], self.store.assigned_documents(VIEWER))
        assignments = [p.hkl_assignments for p in self.viewer._session_peaks]
        self.assertTrue(all(len(values) == 1 and values[0][0] == phases[0].uid for values in assignments))

    def test_event_loop_cancel_disable_and_close_do_not_touch_worker_qt(self):
        dialog = self.open_indexing()
        worker_started, worker_stopped = Event(), Event()
        thread_ids = []
        def waiting(request, cancel):
            thread_ids.append(get_ident())
            worker_started.set()
            cancel.wait(5)
            worker_stopped.set()
            raise IndexingCancelled()
        ticks = []
        timer = QTimer(self.window)
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start()
        with patch('xrd_workbench.ui_qt.indexing_dialog.run_indexing', waiting):
            dialog.start()
            self.wait_until(worker_started.is_set)
            QTest.qWait(100)
            self.assertGreater(len(ticks), 3)
            self.assertNotEqual(thread_ids[0], get_ident())
            start = monotonic()
            self.window.debug_features.set_indexation_enabled(False)
            self.assertLess(monotonic() - start, 1)
            self.wait_until(worker_stopped.is_set)
        timer.stop()
        self.assertFalse(self.viewer._indexing_dialogs)
        self.assertTrue(self.table.index_button.isHidden())

    def test_changed_peaks_invalidate_results_before_phase_creation(self):
        dialog = self.open_indexing()
        self.configure_silicon(dialog)
        dialog.start()
        self.wait_until(lambda: dialog._future is None)
        self.viewer._session_peaks[0] = replace(self.viewer._session_peaks[0], source_center=29)
        self.viewer._refresh_peak_table()
        self.assertFalse(dialog.create_button.isEnabled())
        with patch.object(CellPhaseDialog, 'exec') as editor:
            dialog._create_phase()
            editor.assert_not_called()

    def test_edited_cell_does_not_receive_old_hkl(self):
        dialog = self.open_indexing()
        self.configure_silicon(dialog)
        dialog.start()
        self.wait_until(lambda: dialog._future is None)
        def accept(editor):
            editor.cell_inputs[0].setValue(6)
            editor._accept()
            return QDialog.DialogCode.Accepted
        with patch.object(CellPhaseDialog, 'exec', accept):
            dialog._create_phase()
        self.assertEqual(sum(p.kind == CELL_PHASE for p in self.store.documents.values()), 1)
        self.assertTrue(all(not p.hkl_assignments for p in self.viewer._session_peaks))
