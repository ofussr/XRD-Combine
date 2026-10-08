"""Measurement-result lifecycle and fitting without a GUI or workspace widgets."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import voigt_profile

from xrd_workbench.models.analysis import MeasurementAnalysis, SessionPeak
from xrd_workbench.models.background import BackgroundAnchors
from xrd_workbench.models.project import ProjectStore, RSM, VIEWER
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.analysis import recalculate_background, refit_analysis
from xrd_workbench.services.peak_fitting import refit_voigt_peaks


def peak_values(center=21.0):
    x = np.linspace(center - .5, center + .5, 201)
    profile = 80 * voigt_profile(x - center, .06, .02) / voigt_profile(0, .06, .02)
    return dict(source_center=center, source_x=x, source_background=np.full_like(x, 5),
                source_profile=profile, source_height=80, source_area=10,
                source_fwhm=.17, source_sigma=.06, source_gamma=.02)


def scan(name="measurement"):
    x = np.linspace(10, 60, 2001)
    return Scan1D(name, x, np.full_like(x, 5), Path(name + ".xy"))


class AnalysisModelTests(unittest.TestCase):
    def setUp(self):
        self.analysis = MeasurementAnalysis()

    def test_acceptance_redraw_and_deletion_keep_stable_ids(self):
        original = self.analysis.accept_peak("scan", "2Theta", center_tolerance=.01,
                                             **peak_values())
        self.analysis.assign_hkl("scan", original.number, ("phase", 1, 1, 0))
        self.analysis.update_peak(original.number, filled=False)
        self.assertIsNone(self.analysis.accept_peak("scan", "2Theta", center_tolerance=.01,
                                                   **peak_values(21.005)))
        redrawn = self.analysis.accept_peak("scan", "2Theta", center_tolerance=.01,
                                            manual=True, replace_number=original.number,
                                            **peak_values(21.02))
        self.assertEqual(redrawn.number, original.number)
        self.assertEqual(redrawn.hkl_assignments, (("phase", 1, 1, 0),))
        self.assertFalse(redrawn.filled)
        self.analysis.remove_peak(original.number)
        following = self.analysis.add_peak("scan", "2Theta", **peak_values())
        self.assertGreater(following.number, original.number)

    def test_measurements_and_axes_keep_separate_results(self):
        a = self.analysis.add_peak("a", "2Theta", **peak_values())
        b = self.analysis.add_peak("b", "2Theta", **peak_values())
        omega = self.analysis.add_peak("a", "Omega", **peak_values())
        for uid, axis in (("a", "2Theta"), ("a", "Omega"), ("b", "2Theta")):
            self.analysis.set_background(uid, BackgroundAnchors(axis, .2,
                                        np.array([10., 60.]), np.array([5., 5.])))
        self.analysis.clear_peaks("a", "2Theta")
        self.assertIsNone(self.analysis.peak(a.number))
        self.assertIs(self.analysis.peak(omega.number), omega)
        self.assertIs(self.analysis.peak(b.number), b)
        self.analysis.clear_background("a", "2Theta")
        self.assertIsNotNone(self.analysis.background_for("a", "Omega"))
        self.assertIsNotNone(self.analysis.background_for("b", "2Theta"))

    def test_removing_parent_promotes_companions_without_deleting_them(self):
        parent = self.analysis.add_peak("a", "2Theta", **peak_values())
        alpha = self.analysis.add_peak("a", "2Theta", **peak_values(21.1))
        beta = self.analysis.add_peak("a", "2Theta", **peak_values(19.5))
        self.analysis.classify_companions("a", "ka2", {alpha.number: parent.number})
        self.analysis.classify_companions("a", "kb", {beta.number: parent.number})
        self.analysis.remove_peak(parent.number, "b")
        self.assertIsNotNone(self.analysis.peak(parent.number))
        self.analysis.remove_peak(parent.number, "a")
        self.assertEqual({p.number for p in self.analysis.peaks}, {alpha.number, beta.number})
        self.assertTrue(all(p.kind == "primary" and p.parent_number is None
                            for p in self.analysis.peaks))

    def test_rejected_companion_batch_does_not_modify_valid_pair(self):
        a = self.analysis.add_peak("a", "2Theta", **peak_values())
        b = self.analysis.add_peak("a", "2Theta", **peak_values(21.1))
        foreign = self.analysis.add_peak("b", "2Theta", **peak_values())
        with self.assertRaises(ValueError):
            self.analysis.classify_companions("a", "ka2",
                                             {b.number: a.number, foreign.number: a.number})
        self.assertIs(self.analysis.peak(b.number), b)

    def test_peak_and_background_inputs_cannot_silently_modify_results(self):
        values = peak_values()
        peak = self.analysis.add_peak("a", "2Theta", **values)
        values["source_profile"][:] = 0
        self.assertGreater(peak.source_profile.max(), 70)
        with self.assertRaises(ValueError):
            peak.source_profile[:] = 0
        x, y = np.array([10., 60.]), np.array([5., 5.])
        background = BackgroundAnchors("2Theta", .2, x, y)
        self.analysis.set_background("a", background)
        y[:] = 500
        np.testing.assert_allclose(background.values(np.array([21.])), [5.])
        with self.assertRaises(ValueError):
            background.source_y[:] = 0
        self.assertIsInstance(self.analysis.peaks, tuple)

    def test_batch_publishes_one_complete_change(self):
        events = []
        self.analysis.subscribe(events.append)
        with self.analysis.batch():
            self.analysis.add_peak("a", "2Theta", **peak_values())
            with self.analysis.batch():
                self.analysis.add_peak("b", "2Theta", **peak_values())
            self.assertFalse(events)
        self.assertEqual(events, [frozenset({"a", "b"})])
        self.analysis.unsubscribe(events.append)
        self.analysis.clear_peaks()
        self.assertEqual(len(events), 1)

    def test_imported_peak_ids_are_not_reused(self):
        imported = SessionPeak(500, "a", "2Theta", **peak_values())
        self.analysis.put_peak(imported)
        self.analysis.clear_peaks()
        new = self.analysis.add_peak("b", "2Theta", **peak_values())
        self.assertEqual(new.number, 501)
        with self.assertRaises(ValueError):
            self.analysis.put_peak(replace(new, scan_uid="a"))

    def test_stale_joint_fit_is_rejected_before_any_commit(self):
        a = self.analysis.add_peak("a", "2Theta", **peak_values())
        b = self.analysis.add_peak("a", "2Theta", **peak_values(40))
        self.analysis.remove_peak(b.number)
        with self.assertRaises(ValueError):
            self.analysis.commit_fit([replace(a, source_height=200), b], scan_uid="a",
                                     axis_name="2Theta", background=None)
        self.assertIs(self.analysis.peak(a.number), a)


class AnalysisProjectTests(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore()
        self.a = self.store.add_scan(scan("a"))
        self.b = self.store.add_scan(scan("b"))
        for document in (self.a, self.b):
            self.store.analysis.add_peak(document.uid, "2Theta", **peak_values())
            recalculate_background(self.store.analysis, document.uid, document.payload, .5)

    def phase(self, name):
        return self.store.add_cell_phase(SimpleNamespace(name=name, source=Path(name + ".cell")))

    def test_viewer_unassignment_clears_all_axes_and_keeps_project_measurement(self):
        analysis = self.store.analysis
        peak = analysis.peaks_for(self.a.uid)[0]
        analysis.assign_hkl(self.a.uid, peak.number, ("phase", 1, 0, 0))
        companion = analysis.add_peak(self.a.uid, "2Theta", **peak_values(21.1))
        analysis.classify_companions(self.a.uid, "ka2", {companion.number: peak.number})
        analysis.add_peak(self.a.uid, "Omega", **peak_values())
        analysis.set_background(self.a.uid, BackgroundAnchors("Omega", .2,
                                np.array([10., 60.]), np.array([5., 5.])))
        other_peaks = analysis.peaks_for(self.b.uid)
        other_background = analysis.background_for(self.b.uid, "2Theta")
        payload = self.a.payload
        self.store.assign(self.a.uid, RSM)
        self.store.assign(self.a.uid, VIEWER)
        notifications = []
        self.store.subscribe(lambda event, document, workspace:
                             notifications.append((event, workspace, analysis.peaks_for(document.uid))))
        self.store.assign(self.a.uid, VIEWER, False)
        self.assertEqual(notifications, [("unassigned", VIEWER, ())])
        self.assertEqual(analysis.peaks_for(self.a.uid), ())
        self.assertIsNone(analysis.background_for(self.a.uid, "2Theta"))
        self.assertIsNone(analysis.background_for(self.a.uid, "Omega"))
        self.assertIs(self.store.documents[self.a.uid].payload, payload)
        self.assertTrue(self.store.is_assigned(self.a.uid, RSM))
        self.assertEqual(analysis.peaks_for(self.b.uid), other_peaks)
        self.assertIs(analysis.background_for(self.b.uid, "2Theta"), other_background)
        self.store.assign(self.a.uid, VIEWER)
        self.assertEqual(analysis.peaks_for(self.a.uid), ())

    def test_unassignment_from_another_workspace_keeps_viewer_analysis(self):
        analysis = self.store.analysis
        original = analysis.peaks_for(self.a.uid)
        background = analysis.background_for(self.a.uid, "2Theta")
        self.store.assign(self.a.uid, VIEWER)
        self.store.assign(self.a.uid, RSM)
        self.store.assign(self.a.uid, RSM, False)
        self.assertEqual(analysis.peaks_for(self.a.uid), original)
        self.assertIs(analysis.background_for(self.a.uid, "2Theta"), background)
        self.assertTrue(self.store.is_assigned(self.a.uid, VIEWER))

    def test_replace_invalidates_all_axes_before_project_notification_without_viewer(self):
        analysis = self.store.analysis
        analysis.add_peak(self.a.uid, "Omega", **peak_values())
        analysis.set_background(self.a.uid, BackgroundAnchors("Omega", .2,
                                np.array([10., 60.]), np.array([5., 5.])))
        retained = analysis.peaks_for(self.b.uid)
        notifications = []
        self.store.subscribe(lambda event, document, workspace:
                             notifications.append((event, analysis.peaks_for(document.uid))))
        self.store.replace_scan(self.a.uid, scan("corrected"))
        self.assertEqual(notifications, [("replaced", ())])
        self.assertIsNone(analysis.background_for(self.a.uid, "2Theta"))
        self.assertIsNone(analysis.background_for(self.a.uid, "Omega"))
        self.assertEqual(analysis.peaks_for(self.b.uid), retained)
        self.assertEqual(len(self.a.history), 1)

    def test_phase_edit_and_removal_only_clear_its_hkl(self):
        p, q = self.phase("p"), self.phase("q")
        analysis = self.store.analysis
        peak = analysis.peaks_for(self.a.uid)[0]
        analysis.assign_hkl(self.a.uid, peak.number, (p.uid, 1, 0, 0))
        analysis.assign_hkl(self.a.uid, peak.number, (q.uid, 1, 1, 0))
        analysis.assign_hkl(self.a.uid, peak.number, (q.uid, 1, 1, 0))
        self.store.assign(p.uid, VIEWER)
        self.store.assign(p.uid, VIEWER, False)
        self.assertEqual(len(analysis.peak(peak.number).hkl_assignments), 2)
        self.store.replace_cell_phase(p.uid, SimpleNamespace(name="edited", source=p.source))
        self.assertEqual(analysis.peak(peak.number).hkl_assignments, ((q.uid, 1, 1, 0),))
        self.store.remove(q.uid)
        self.assertEqual(analysis.peak(peak.number).hkl_assignments, ())
        self.assertEqual(len(analysis.peaks), 2)

    def test_removal_and_clear_discard_results_without_a_widget(self):
        analysis = self.store.analysis
        self.store.remove(self.a.uid)
        self.assertEqual(analysis.peaks_for(self.a.uid), ())
        self.assertIsNone(analysis.background_for(self.a.uid, "2Theta"))
        self.assertEqual(len(analysis.peaks_for(self.b.uid)), 1)
        self.store.clear()
        self.assertEqual(analysis.peaks, ())
        self.assertIsNone(analysis.background_for(self.b.uid, "2Theta"))


class AnalysisServiceTests(unittest.TestCase):
    def make_fit(self):
        x = np.linspace(10, 60, 4001)
        y = np.full_like(x, 5)
        analysis = MeasurementAnalysis()
        for center in (21., 45.):
            y += 80 * voigt_profile(x - center, .06, .02) / voigt_profile(0, .06, .02)
            values = peak_values(center + .01)
            values["source_height"] = 60
            peak = analysis.add_peak("scan", "2Theta", **values)
            analysis.update_peak(peak.number, filled=False, hkl_assignments=(("phase", 1, 0, 0),))
        scan = Scan1D("fit", x, y, Path("fit.xy"))
        background = BackgroundAnchors("2Theta", .5, x[[0, -1]], np.array([5., 5.]))
        analysis.set_background("scan", background)
        return analysis, scan, background

    def test_joint_fit_works_without_qt_and_retains_peak_metadata(self):
        analysis, scan, _background = self.make_fit()
        numbers = [peak.number for peak in analysis.peaks]
        events = []
        analysis.subscribe(events.append)
        self.assertTrue(refit_analysis(analysis, "scan", scan))
        np.testing.assert_allclose([p.source_center for p in analysis.peaks], [21., 45.], atol=.002)
        self.assertEqual([p.number for p in analysis.peaks], numbers)
        self.assertTrue(all(not p.filled and p.hkl_assignments == (("phase", 1, 0, 0),)
                            for p in analysis.peaks))
        self.assertEqual(events, [frozenset({"scan"})])

    def test_later_group_failure_does_not_commit_earlier_fit_or_background(self):
        analysis, scan, background = self.make_fit()
        originals = analysis.peaks
        events = []
        analysis.subscribe(events.append)
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("Second group failed")
            return refit_voigt_peaks(*args, **kwargs)

        with patch("xrd_workbench.services.analysis.refit_voigt_peaks", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "Second group"):
                refit_analysis(analysis, "scan", scan)
        self.assertEqual(calls, 2)
        self.assertTrue(all(a is b for a, b in zip(originals, analysis.peaks)))
        self.assertIs(analysis.background_for("scan", "2Theta"), background)
        self.assertFalse(events)

    def test_recalculation_preserves_background_edits_in_source_coordinates(self):
        analysis, scan, background = self.make_fit()
        modified = background.without_region(20, 22).with_local_level(30, 7, 3)
        analysis.set_background("scan", modified)
        recalculate_background(analysis, "scan", scan, .25)
        new = analysis.background_for("scan", "2Theta")
        self.assertEqual(new.excluded, modified.excluded)
        self.assertEqual(new.local_levels, modified.local_levels)
        self.assertEqual(new.spacing, .25)

    def test_core_imports_work_when_gui_imports_are_forbidden(self):
        code = '''
import importlib.abc
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("PySide6", "pyqtgraph", "xrd_workbench.ui_qt")):
            raise RuntimeError("Unexpected GUI import: " + fullname)
sys.meta_path.insert(0, NoGui())
from xrd_workbench.models.analysis import MeasurementAnalysis, SessionPeak
from xrd_workbench.models.background import BackgroundAnchors
from xrd_workbench.models.project import ProjectStore
from xrd_workbench.services.analysis import refit_analysis, recalculate_background
assert isinstance(ProjectStore().analysis, MeasurementAnalysis)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
