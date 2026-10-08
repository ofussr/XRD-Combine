"""Manual Voigt detection and the Viewer session peak table."""

from pathlib import Path
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import voigt_profile

from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.models.project import ProjectStore, VIEWER
from xrd_workbench.models.radiation import RadiationSettings
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.peak_fitting import (
    companion_two_theta, fit_voigt_region, refit_voigt_peaks,
)
from xrd_workbench.services.background import BackgroundAnchors


def sample_peaks(count=2):
    x = np.linspace(20.0, 24.0, 501)
    def component(centre, height):
        return height * voigt_profile(x - centre, 0.05, 0.025) / voigt_profile(
            0.0, 0.05, 0.025
        )
    y = 5.0 + 0.1 * (x - 22.0) + component(21.0, 120.0)
    if count == 2:
        y += component(21.42, 60.0)
    return x, y


class VoigtSearchTests(unittest.TestCase):
    def test_joint_refit_improves_sum_of_overlapping_accepted_peaks(self):
        x = np.linspace(20.0, 22.0, 501)
        base = 30.0 + 2.0 * (x - 20.0)
        actual = [(20.8, 170.0, .075, .025),
                  (20.97, 260.0, .045, .025),
                  (21.16, 100.0, .06, .03)]
        def profile(values, seed):
            centre, height, sigma, gamma = seed
            return height * voigt_profile(values - centre, sigma, gamma) / voigt_profile(
                0.0, sigma, gamma)
        y = base + sum((profile(x, seed) for seed in actual), np.zeros_like(x))
        seeds = [(20.78, 130.0, .05, .045),
                 (20.99, 190.0, .07, .035),
                 (21.19, 140.0, .045, .04)]
        before = base + sum((profile(x, seed) for seed in seeds), np.zeros_like(x))
        result = refit_voigt_peaks(x, y, base, seeds)
        self.assertLess(np.linalg.norm(result.fitted - y),
                        np.linalg.norm(before - y) * .01)
        np.testing.assert_allclose(result.fitted, result.background + sum(
            (peak.profile for peak in result.peaks), np.zeros_like(x)))
        np.testing.assert_allclose([peak.center for peak in result.peaks],
                                   [seed[0] for seed in actual], atol=.001)
        self.assertEqual(len(result.peaks[0].profile), len(x))

    def test_joint_refit_does_not_absorb_unaccepted_distant_peak(self):
        x, y = sample_peaks()
        base = 5.0 + .1 * (x - 22.0)
        result = refit_voigt_peaks(x, y, base, [(21.42, 60, .05, .025)])
        self.assertAlmostEqual(result.peaks[0].center, 21.42, delta=.005)
        self.assertAlmostEqual(result.peaks[0].height, 60, delta=3)

    def test_local_background_refit_uses_shoulders_and_preserves_distant_range(self):
        x = np.linspace(18, 24, 1201)
        correct_base = 20 + .25 * (x - 20)
        wrong_base = correct_base + 12 * np.exp(-.5 * ((x - 21) / 1.2) ** 2)
        y = correct_base.copy()
        seeds = []
        for centre, height in ((20.7, 120), (21.45, 90)):
            y += height * voigt_profile(x - centre, .06, .025) / voigt_profile(
                0, .06, .025)
            seeds.append((centre, .8 * height, .07, .03))
        fixed = refit_voigt_peaks(x, y, wrong_base, seeds)
        refined = refit_voigt_peaks(x, y, wrong_base, seeds,
                                    refine_background=True)
        self.assertLess(np.linalg.norm(refined.fitted - y),
                        np.linalg.norm(fixed.fitted - y) * .8)
        self.assertLess(float(np.interp(21, x, refined.background)),
                        float(np.interp(21, x, wrong_base)))
        np.testing.assert_allclose(refined.background[x < 19], wrong_base[x < 19])
        np.testing.assert_allclose(refined.background[x > 23], wrong_base[x > 23])
        self.assertIsNotNone(refined.background_delta)

    def test_search_bounds_select_centres_without_truncating_fitted_tails(self):
        x, y = sample_peaks(count=1)
        fitted = fit_voigt_region(
            x[(x >= 20.5) & (x <= 21.5)],
            y[(x >= 20.5) & (x <= 21.5)],
            search_bounds=(20.85, 21.15),
        )
        self.assertEqual(len(fitted.peaks), 1)
        self.assertAlmostEqual(fitted.peaks[0].center, 21.0, places=3)
        self.assertLess(fitted.x[0], 20.85)
        self.assertGreater(fitted.x[-1], 21.15)

    def test_one_and_two_peaks_with_background(self):
        for count in (1, 2):
            with self.subTest(count=count):
                x, y = sample_peaks(count)
                fit = fit_voigt_region(x, y)
                self.assertEqual(len(fit.peaks), count)
                self.assertAlmostEqual(fit.peaks[0].center, 21.0, places=3)
                if count == 2:
                    self.assertAlmostEqual(fit.peaks[1].center, 21.42, places=3)
                self.assertGreater(fit.peaks[0].fwhm, 0.0)
                np.testing.assert_allclose(
                    fit.fitted, fit.background + sum(
                        (peak.profile for peak in fit.peaks), np.zeros_like(fit.x)
                    ),
                )

    def test_no_peak_and_up_to_six_peaks(self):
        x, _ = sample_peaks()
        with self.assertRaises(XRDDataError):
            fit_voigt_region(x, np.full_like(x, 5.0))
        y = np.full_like(x, 4.0)
        for centre in (20.7, 21.7, 22.7):
            y += 50.0 * voigt_profile(x - centre, .045, .02) / voigt_profile(0, .045, .02)
        fit = fit_voigt_region(x, y)
        self.assertEqual(len(fit.peaks), 3)
        self.assertAlmostEqual(fit.peaks[-1].center, 22.7, places=3)

    def test_six_peaks_and_seven_peak_limit(self):
        x = np.linspace(20, 32, 1001)
        centers = (20.7, 22.1, 23.5, 25.2, 27.1, 29.3)
        y = np.full_like(x, 4.0)
        for center in centers:
            y += 80 * voigt_profile(x - center, .04, .022) / voigt_profile(0, .04, .022)
        fitted = fit_voigt_region(x, y)
        np.testing.assert_allclose([peak.center for peak in fitted.peaks], centers, atol=.001)
        y += 60 * voigt_profile(x - 30.6, .04, .022) / voigt_profile(0, .04, .022)
        with self.assertRaises(XRDDataError) as caught:
            fit_voigt_region(x, y)
        self.assertEqual(caught.exception.code, "peak_fit_many")

    def test_weak_peak_remains_visible_beside_intense_reflection(self):
        x = np.linspace(20, 22, 1001)
        background = 50 + 2 * (x - 20)
        y = background + np.random.default_rng(7).normal(0, .8, x.size)
        y += (x >= 20.9).astype(float)  # instrument intensity step
        for centre, height in ((20.4, 60), (21.2, 5000)):
            y += height * voigt_profile(x - centre, .035, .02) / voigt_profile(
                0, .035, .02)
        result = fit_voigt_region(x, y, background=background)
        self.assertEqual(len(result.peaks), 2)
        np.testing.assert_allclose([peak.center for peak in result.peaks],
                                   [20.4, 21.2], atol=.003)

    def test_negative_noise_maxima_do_not_hide_positive_peak(self):
        x = np.linspace(0, 4, 801)
        background = np.full_like(x, 100.0)
        y = background - 15 + 2 * np.sin(5 * x)
        y += 30 * voigt_profile(x - 2, .04, .02) / voigt_profile(0, .04, .02)
        result = fit_voigt_region(x, y, background=background)
        self.assertEqual(len(result.peaks), 1)
        self.assertAlmostEqual(result.peaks[0].center, 2, delta=.01)

    def test_small_positive_maximum_does_not_cancel_two_real_peaks(self):
        x = np.linspace(25.6, 27.5, 701)
        background = np.full_like(x, 100.0)
        y = background - 10 + np.random.default_rng(1).normal(0, .5, x.size)
        for center, height, width in ((26.1, 70, .022), (26.9, 90, .022),
                                      (26.55, 12, .006)):
            y += height * voigt_profile(x - center, width, .008) / voigt_profile(
                0, width, .008)
        result = fit_voigt_region(x, y, background=background)
        np.testing.assert_allclose([peak.center for peak in result.peaks],
                                   [26.1, 26.9], atol=.003)


try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from xrd_workbench.models.analysis import SessionPeak
    from xrd_workbench.ui_qt.viewer_page import ViewerPage
except ImportError:
    QApplication = None


@unittest.skipUnless(QApplication is not None, "Qt runtime unavailable")
class ViewerPeakTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.store = ProjectStore()
        x, y = sample_peaks()
        scan = Scan1D("Two peaks", x, y, Path("two-peaks.xy"))
        self.document = self.store.add_scan(scan)
        self.store.assign(self.document.uid, VIEWER, True)
        self.viewer = ViewerPage(self.store, RadiationSettings())
        self.viewer.select_uid(self.document.uid)
        self.viewer.set_plot_renderer("pyqtgraph")

    def tearDown(self):
        self.viewer.close()
        self.viewer.deleteLater()
        self.app.processEvents()

    def test_correction_number_fields_fit_narrow_panel(self):
        self.viewer.resize(960, 720)
        self.viewer.show()
        self.viewer.processing_section.set_expanded(True)
        self.viewer.controls_scroll.setFixedWidth(275)
        self.app.processEvents()
        viewport = self.viewer.controls_scroll.viewport()
        for spin in (self.viewer.processing_x_spin, self.viewer.processing_y_spin,
                     self.viewer.processing_factor_spin):
            self.assertGreaterEqual(spin.width(), 138)
            self.assertLess(spin.mapTo(viewport, spin.rect().topRight()).x(),
                            viewport.width())

    def test_choose_one_of_two_then_remove_and_clear(self):
        self.viewer.activate_peak_search()
        self.assertTrue(self.viewer._peak_search_active)
        self.viewer._fit_peak_bounds(20.5, 0, 22, 150)
        self.assertEqual(len(self.store.analysis.peaks), 0)
        self.assertEqual(len(self.viewer._pending_peak_fit.result.peaks), 2)
        self.assertTrue(self.viewer.peak_preview_bar.isVisibleTo(self.viewer))
        self.assertIn("2", self.viewer.peak_preview_count.text())
        self.assertTrue(self.viewer.pyqtgraph_plot._temporary_scan_items)
        self.viewer.peak_preview_checks[0].setChecked(False)
        self.viewer._confirm_peak_preview()
        self.assertIsNone(self.viewer._pending_peak_fit)
        self.assertFalse(self.viewer.peak_preview_bar.isVisibleTo(self.viewer))
        self.assertFalse(self.viewer.pyqtgraph_plot._temporary_scan_items)
        self.assertEqual(len(self.store.analysis.peaks), 1)
        self.assertAlmostEqual(self.store.analysis.peaks[0].source_center, 21.42, places=3)
        self.viewer.open_peak_table()
        self.assertEqual(self.viewer._peak_table_dialog.table.rowCount(), 1)
        self.viewer._remove_session_peak(self.store.analysis.peaks[0].number)
        self.assertEqual(self.viewer._peak_table_dialog.table.rowCount(), 0)

    def test_single_peak_confirm_highlight_and_d_spacing_selection(self):
        item = self.viewer.items[self.document.uid]
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.6, 0, 21.25, 500)
        self.assertEqual(len(self.store.analysis.peaks), 0)
        self.assertEqual(len(self.viewer._pending_peak_fit.result.peaks), 1)
        self.assertLess(self.viewer._pending_peak_fit.source_x[0], 20.6)
        self.assertGreater(self.viewer._pending_peak_fit.source_x[-1], 21.25)
        self.viewer._draw(preserve_view=True)
        self.assertTrue(self.viewer.pyqtgraph_plot._temporary_scan_items)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 1)
        fills = [graphic for graphic in self.viewer.pyqtgraph_plot.scan_plot_item.items
                 if "FillBetweenItem" in type(graphic).__name__]
        self.assertTrue(fills)
        y_view = self.viewer.pyqtgraph_plot.scan_view_box.viewRange()[1]
        self.assertLess(fills[0].boundingRect().top(), y_view[1])
        self.assertGreater(fills[0].boundingRect().bottom(), y_view[0])
        self.viewer.viewer_state.plot.x_display_mode = "d"
        self.viewer._draw(preserve_view=False)
        from xrd_workbench.models.viewer import two_theta_to_d
        limits = two_theta_to_d(np.asarray([20.6, 21.25]),
                                self.viewer.viewer_state.plot.display_wavelength)
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(float(limits[0]), 0, float(limits[1]), 500)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 1)  # repeated peak
        item.x_shift = 0.25
        self.viewer._draw(preserve_view=True)
        self.viewer._refresh_peak_table()
        self.viewer.open_peak_table()
        shown = float(self.viewer._peak_table_dialog.table.item(0, 2).text())
        expected_d = two_theta_to_d(
            np.asarray([self.store.analysis.peaks[0].source_center + 0.25]),
            self.viewer.viewer_state.plot.display_wavelength,
        )[0]
        self.assertAlmostEqual(shown, expected_d, places=4)

    def test_confirmed_sum_follows_scan_and_can_be_hidden_for_one_measurement(self):
        uid = self.document.uid
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.5, 0, 21.9, 500)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 2)
        item = self.viewer.items[uid]
        fit_x, fit_y = self.viewer._peak_sum_curve(item, 0)
        within = np.isfinite(fit_y) & (fit_x > 20.9) & (fit_x < 21.6)
        measured = np.interp(fit_x[within], item.scan.x, item.scan.y)
        np.testing.assert_allclose(fit_y[within], measured, atol=0.2)
        self.viewer.open_peak_table()
        check = self.viewer._peak_table_dialog.show_sum_check
        self.assertTrue(check.isChecked())
        check.click()
        self.assertIsNone(self.viewer._peak_sum_curve(item, 0))
        check.click()
        self.assertIsNotNone(self.viewer._peak_sum_curve(item, 0))
        self.viewer.set_plot_renderer("matplotlib")
        self.assertTrue(any(line.get_color() == "#8e44ad"
                            for line in self.viewer.scan_axis.lines))

    def test_acceptance_then_do_fit_updates_same_peaks_and_shared_sum(self):
        uid = self.document.uid
        item = self.viewer.items[uid]
        x = item.scan.x
        self.store.analysis.set_background(uid, BackgroundAnchors(
            item.scan.axis_name, .25, np.array([x[0], x[-1]]),
            np.array([4.8, 5.2]),
        ))
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.6, 0, 21.9, 500)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 2)
        self.assertTrue(all(len(peak.source_x) == len(x)
                            for peak in self.store.analysis.peaks))
        first = self.store.analysis.peaks[0]
        self.store.analysis.put_peak(replace(
            first, source_height=first.source_height * .5,
            source_profile=first.source_profile * .5,
            hkl_assignments=(("phase", 1, 1, 0),), filled=False,
        ))
        _axis, before = self.viewer._peak_sum_curve(item, 0)
        self.viewer.open_peak_table()
        self.viewer._peak_table_dialog.fit_button.click()
        _axis, after = self.viewer._peak_sum_curve(item, 0)
        self.assertLess(np.linalg.norm(after - item.scan.y),
                        np.linalg.norm(before - item.scan.y) * .1)
        self.assertEqual(self.store.analysis.peaks[0].number, first.number)
        self.assertEqual(self.store.analysis.peaks[0].hkl_assignments,
                         (("phase", 1, 1, 0),))
        self.assertFalse(self.store.analysis.peaks[0].filled)

    def test_second_region_refits_only_its_nearby_peak_group(self):
        x = np.linspace(10, 80, 25001)
        baseline = 30 + .05 * x
        y = baseline.copy()
        for centre in (25.0, 65.0):
            y += 300 * voigt_profile(x - centre, .07, .025) / voigt_profile(
                0, .07, .025)
        document = self.store.add_scan(Scan1D("Distant peaks", x, y,
                                             Path("distant-peaks.xy")))
        self.store.assign(document.uid, VIEWER, True)
        self.viewer.refresh_documents()
        self.viewer.select_uid(document.uid)
        item = self.viewer.items[document.uid]
        self.store.analysis.set_background(document.uid, BackgroundAnchors(
            item.scan.axis_name, .25, np.array([x[0], x[-1]]),
            np.array([baseline[0], baseline[-1]]),
        ))
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(24.75, 0, 25.25, 500)
        self.viewer._confirm_peak_preview()
        first = next(peak for peak in self.store.analysis.peaks
                     if peak.scan_uid == document.uid)
        prior_background = float(self.viewer._background_for(item).values(np.array([25.0]))[0])
        from xrd_workbench.services import analysis as analysis_service
        original = analysis_service.refit_voigt_peaks
        sizes = []
        def recorded(*args, **kwargs):
            sizes.append(len(args[3]))
            return original(*args, **kwargs)
        with patch.object(analysis_service, "refit_voigt_peaks", side_effect=recorded):
            self.viewer.activate_peak_search()
            self.viewer._fit_peak_bounds(64.75, 0, 65.25, 500)
            self.viewer._confirm_peak_preview()
        self.assertEqual(sizes, [1])
        self.assertEqual(len([p for p in self.store.analysis.peaks
                              if p.scan_uid == document.uid]), 2)
        current = next(peak for peak in self.store.analysis.peaks
                       if peak.number == first.number)
        self.assertIs(current, first)
        self.assertAlmostEqual(float(self.viewer._background_for(item).values(
            np.array([25.0]))[0]), prior_background, places=7)

    def test_second_region_search_subtracts_tails_of_accepted_peaks(self):
        x = np.linspace(28, 35, 3001)
        baseline = 500 + 40 * (x - 28)
        y = baseline.copy()
        for center, height, sigma, gamma in (
            (29.08, 120, .035, .012), (29.35, 130, .03, .015),
            (29.59, 2800, .021, .008),
            (31.7, 9000, .05, .035), (32.24, 45000, .06, .025),
            (32.55, 38000, .045, .03), (32.67, 7000, .016, .005),
            (32.82, 900000, .014, .005), (32.9, 350000, .013, .004),
        ):
            y += height * voigt_profile(x - center, sigma, gamma) / voigt_profile(
                0, sigma, gamma)
        document = self.store.add_scan(Scan1D("Nearby regions", x, y,
                                             Path("nearby-regions.xy")))
        self.store.assign(document.uid, VIEWER, True)
        self.viewer.refresh_documents()
        self.viewer.select_uid(document.uid)
        item = self.viewer.items[document.uid]
        self.store.analysis.set_background(document.uid, BackgroundAnchors(
            item.scan.axis_name, .25, np.array([x[0], x[-1]]),
            np.array([baseline[0], baseline[-1]]),
        ))
        with patch("xrd_workbench.ui_qt.viewer_page.QMessageBox.warning",
                   side_effect=AssertionError("Search must not open a warning")):
            for low, high in ((31, 34), (28.5, 30)):
                self.viewer.activate_peak_search()
                self.viewer._fit_peak_bounds(low, 0, high, 1e6)
                self.assertIsNotNone(self.viewer._pending_peak_fit)
                self.viewer._confirm_peak_preview()
        self.assertTrue(any(abs(p.source_center - 29.59) < .02
                            for p in self.store.analysis.peaks))

    def test_drawn_peak_keeps_user_shape_until_do_fit(self):
        uid = self.document.uid
        item = self.viewer.items[uid]
        x = item.scan.x
        baseline = 5.0 + .1 * (x - 22.0)
        first_profile = 120 * voigt_profile(x - 21, .05, .025) / voigt_profile(
            0, .05, .025)
        self.store.analysis.set_background(uid, BackgroundAnchors(
            item.scan.axis_name, .25, np.array([x[0], x[-1]]),
            np.array([baseline[0], baseline[-1]]),
        ))
        self.store.analysis.put_peak(SessionPeak(
            1, uid, item.scan.axis_name, 21.0, x, baseline, first_profile,
            120.0, 120.0 / voigt_profile(0, .05, .025), .14,
            source_sigma=.05, source_gamma=.025,
        ))

        self.viewer._fit_drawn_bounds(item, 21.42, 65.0, 21.50, 5.0)
        proposal = self.viewer._pending_peak_fit.result.peaks[0]
        self.assertEqual(proposal.center, 21.42)
        self.assertEqual(proposal.height, 60)
        self.assertAlmostEqual(proposal.fwhm, .16)
        with patch.object(self.viewer, "_refit_session_peaks") as refit:
            self.viewer._confirm_peak_preview()
            refit.assert_not_called()
        drawn = self.store.analysis.peaks[-1]
        self.assertEqual(drawn.source_center, 21.42)
        self.assertEqual(drawn.source_height, 60)
        self.assertAlmostEqual(drawn.source_fwhm, .16)
        self.viewer.open_peak_table()
        self.viewer._peak_table_dialog.fit_button.click()
        _axis, model = self.viewer._peak_sum_curve(item, 0)
        self.assertEqual(len(self.store.analysis.peaks), 2)
        self.assertLess(np.sqrt(np.mean((model - item.scan.y) ** 2)), 1.0)

    def test_main_do_fit_button_refits_drawn_peak(self):
        item = self.viewer.items[self.document.uid]
        self.viewer.activate_drawn_peak()
        self.viewer._fit_peak_bounds(21.0, 130.0, 21.08, 4.0)
        self.viewer._confirm_peak_preview()
        self.assertTrue(self.viewer.peak_do_fit_button.isEnabled())
        with patch.object(self.viewer, "_refit_session_peaks", return_value=True) as refit:
            self.viewer.peak_do_fit_button.click()
            refit.assert_called_once_with(item.uid, show_error=True)

    def test_drawn_peak_preview_cancel_and_confirm_background(self):
        item = self.viewer.items[self.document.uid]
        self.viewer.activate_drawn_peak()
        self.assertTrue(self.viewer.pyqtgraph_plot.scan_view_box.manual_enabled)
        self.viewer._fit_peak_bounds(21.0, 130.0, 21.08, 4.0)
        pending = self.viewer._pending_peak_fit
        self.assertIsNotNone(pending)
        self.assertTrue(pending.manual)
        self.assertEqual(len(pending.result.peaks), 1)
        self.assertIsNone(self.viewer._background_for(item))
        self.assertEqual(len(self.store.analysis.peaks), 0)
        self.viewer._discard_peak_preview()
        self.assertIsNone(self.viewer._background_for(item))
        self.assertEqual(len(self.store.analysis.peaks), 0)

        self.viewer.open_peak_table()
        self.viewer._peak_table_dialog.draw_button.click()
        self.assertTrue(self.viewer._manual_peak_active)
        self.viewer._fit_peak_bounds(21.0, 130.0, 21.08, 4.0)
        pending = self.viewer._pending_peak_fit
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 1)
        self.assertAlmostEqual(self.store.analysis.peaks[0].source_center, 21.0,
                               delta=.02)
        np.testing.assert_allclose(
            self.viewer._background_for(item).values(pending.source_x),
            np.interp(pending.source_x, self.store.analysis.peaks[0].source_x,
                      self.store.analysis.peaks[0].source_background),
            atol=.1,
        )
        self.assertIsNone(self.viewer._pending_peak_fit)

        # Re-drawing the same peak updates its fit and keeps its table number.
        number = self.store.analysis.peaks[0].number
        self.viewer.activate_drawn_peak()
        self.viewer._fit_peak_bounds(21.0, 130.0, 21.08, 4.0)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 1)
        self.assertEqual(self.store.analysis.peaks[0].number, number)

    def test_cancel_preview_then_confirm_on_default_plot(self):
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.6, 0, 21.25, 500)
        self.assertEqual(len(self.store.analysis.peaks), 0)
        self.assertTrue(self.viewer.pyqtgraph_plot._temporary_scan_items)
        self.viewer._discard_peak_preview()
        self.assertFalse(self.viewer.pyqtgraph_plot._temporary_scan_items)
        self.assertEqual(len(self.store.analysis.peaks), 0)
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.6, 0, 21.25, 500)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 1)
        self.assertFalse(self.viewer.pyqtgraph_plot._temporary_scan_items)

    def _insert_peak(self, uid, center, number):
        x = np.linspace(center - .2, center + .2, 100)
        peak = SessionPeak(number, uid, "2Theta", center, x,
                           np.full_like(x, 5.0), np.exp(-((x-center)/.03)**2),
                           100.0, 4.0, 0.10)
        self.store.analysis.put_peak(peak)
        return peak

    def test_unassigning_viewer_discards_analysis_and_reassignment_starts_empty(self):
        uid = self.document.uid
        peak = self._insert_peak(uid, 21.42, 1)
        self.store.analysis.assign_hkl(uid, peak.number, ("phase", 1, 0, 0))
        self.viewer.calculate_background()
        self.viewer.open_peak_table()
        self.store.assign(uid, VIEWER, False)
        self.viewer.refresh_documents()
        self.assertNotIn(uid, self.viewer.items)
        self.assertIn(uid, self.store.documents)
        self.assertIsNone(self.store.analysis.peak(peak.number))
        self.assertIsNone(self.store.analysis.background_for(uid, "2Theta"))
        self.store.assign(uid, VIEWER)
        self.viewer.refresh_documents()
        self.viewer.select_uid(uid)
        self.viewer.open_peak_table()
        self.assertEqual(self.viewer._peak_table_dialog.table.rowCount(), 0)
        self.assertIsNone(self.viewer._peak_sum_curve(self.viewer.items[uid], 0))

    def test_visibility_checkbox_and_show_all_preserve_analysis(self):
        uid = self.document.uid
        peak = self._insert_peak(uid, 21.42, 1)
        self.store.analysis.assign_hkl(uid, peak.number, ("phase", 1, 0, 0))
        retained = self.store.analysis.peak(peak.number)
        self.viewer.calculate_background()
        background = self.store.analysis.background_for(uid, "2Theta")
        row = self.viewer.documents.topLevelItem(0)
        row.setCheckState(1, Qt.CheckState.Unchecked)
        self.assertFalse(self.viewer.items[uid].visible)
        self.assertTrue(self.store.is_assigned(uid, VIEWER))
        self.assertIs(self.store.analysis.peak(peak.number), retained)
        self.assertIs(self.store.analysis.background_for(uid, "2Theta"), background)
        self.viewer.show_all()
        self.assertTrue(self.viewer.items[uid].visible)
        self.viewer.toggle_selected_visibility()
        self.viewer.toggle_selected_visibility()
        self.assertTrue(self.viewer.items[uid].visible)
        self.assertIs(self.store.analysis.peak(peak.number), retained)
        self.assertIs(self.store.analysis.background_for(uid, "2Theta"), background)

    def test_viewer_remove_and_clear_discard_analysis_but_keep_project_data(self):
        uid = self.document.uid
        self._insert_peak(uid, 21.42, 1)
        self.viewer.calculate_background()
        self.viewer.remove_selected()
        self.viewer.refresh_documents()
        self.assertIn(uid, self.store.documents)
        self.assertFalse(self.store.is_assigned(uid, VIEWER))
        self.assertEqual(self.store.analysis.peaks_for(uid), ())
        self.assertIsNone(self.store.analysis.background_for(uid, "2Theta"))
        self.store.assign(uid, VIEWER)
        self.viewer.refresh_documents()
        self.viewer.select_uid(uid)
        self._insert_peak(uid, 21.42, 2)
        self.viewer.calculate_background()
        self.viewer.clear_all()
        self.viewer.refresh_documents()
        self.assertIn(uid, self.store.documents)
        self.assertFalse(self.viewer.items)
        self.assertEqual(self.store.analysis.peaks_for(uid), ())
        self.assertIsNone(self.store.analysis.background_for(uid, "2Theta"))

    def test_recreating_viewer_uses_the_same_project_analysis(self):
        from PySide6.QtCore import QEvent
        uid = self.document.uid
        peak = self._insert_peak(uid, 21.42, 1)
        self.viewer.calculate_background()
        self.viewer.close()
        self.viewer.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.viewer = ViewerPage(self.store, RadiationSettings())
        self.viewer.select_uid(uid)
        self.viewer.set_plot_renderer("pyqtgraph")
        self.viewer.open_peak_table()
        self.assertIs(self.viewer.analysis, self.store.analysis)
        self.assertIs(self.store.analysis.peak(peak.number), peak)
        self.assertEqual(self.viewer._peak_table_dialog.table.rowCount(), 1)
        self.assertIsNotNone(self.viewer._background_for(self.viewer.items[uid]))
        # Notifications target the replacement page, with no deleted-widget access.
        self.store.analysis.remove_peak(peak.number)
        self.assertEqual(self.viewer._peak_table_dialog.table.rowCount(), 0)

    def test_external_model_update_refreshes_table_and_shading(self):
        uid = self.document.uid
        peak = self._insert_peak(uid, 21.42, 1)
        self.viewer.open_peak_table()
        self.store.analysis.update_peak(peak.number, source_center=21.5, filled=False)
        table = self.viewer._peak_table_dialog.table
        self.assertAlmostEqual(float(table.item(0, 2).text()), 21.5, places=4)
        self.assertEqual(list(self.viewer._peak_shades(self.viewer.items[uid], 0)), [])

    def test_model_update_keeps_active_region_selection_without_rebuilding_tree(self):
        peak = self._insert_peak(self.document.uid, 21.42, 1)
        self.viewer.calculate_background()
        self.viewer.activate_peak_search()
        self.store.analysis.update_peak(peak.number, filled=False)
        self.assertTrue(self.viewer._peak_search_active)
        self.assertTrue(self.viewer.pyqtgraph_plot.scan_view_box.selection_enabled)
        self.assertEqual(self.viewer._fit_uid, self.document.uid)
        self.viewer._fit_peak_bounds(20.6, 0, 21.25, 500)
        self.assertIsNotNone(self.viewer._pending_peak_fit)
        self.viewer._confirm_peak_preview()
        self.assertEqual(len(self.store.analysis.peaks), 2)
        # Region selection remains a one-shot action, as before this refactor.
        self.assertFalse(self.viewer._peak_search_active)

    def test_external_scan_replacement_cancels_preview_and_refreshes_payload(self):
        uid = self.document.uid
        self._insert_peak(uid, 21.42, 1)
        self.viewer.calculate_background()
        self.viewer.open_peak_table()
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.6, 0, 21.25, 500)
        self.assertIsNotNone(self.viewer._pending_peak_fit)
        x = np.linspace(30, 40, 500)
        replacement = Scan1D("Replacement", x, np.full_like(x, 7), Path("replacement.xy"))
        self.store.replace_scan(uid, replacement)
        self.assertIsNone(self.viewer._pending_peak_fit)
        self.assertEqual(self.store.analysis.peaks_for(uid), ())
        self.assertIsNone(self.viewer._background_for(self.viewer.items[uid]))
        self.assertEqual(self.viewer._peak_table_dialog.table.rowCount(), 0)
        np.testing.assert_allclose(self.viewer.items[uid].scan.x, x)
        self.assertEqual(self.viewer.items[uid].name, "Replacement")

    def test_one_measurement_auto_selected_and_tables_are_separate(self):
        self.assertEqual(self.viewer._selected_uid(), self.document.uid)
        self._insert_peak(self.document.uid, 21.42, 1)
        self._insert_peak(self.document.uid, 21.0, 2)
        self.viewer.open_peak_table()
        first = self.viewer._peak_table_dialog
        self.assertEqual(first.table.columnCount(), 9)
        self.assertNotIn("Measurement", [first.table.horizontalHeaderItem(n).text()
                                          for n in range(first.table.columnCount())])
        self.assertEqual([first.table.item(n, 2).text() for n in range(2)],
                         ["21.00000", "21.42000"])
        self.assertEqual([first.table.item(n, 0).text() for n in range(2)], ["1", "2"])
        first.table.setCurrentCell(0, 0)
        first._remove_selected()  # Remove by stable ID, even after sorting.
        self.assertEqual([p.source_center for p in self.store.analysis.peaks], [21.42])

        x, y = sample_peaks()
        second_doc = self.store.add_scan(Scan1D("Other", x, y, Path("other.xy")))
        self.store.assign(second_doc.uid, VIEWER, True)
        self.viewer.refresh_documents()
        self.assertEqual(self.viewer._selected_uid(), self.document.uid)
        self._insert_peak(second_doc.uid, 22.7, 3)
        self.viewer.select_uid(second_doc.uid)
        self.viewer.open_peak_table()
        second = self.viewer._peak_table_dialog
        self.assertIsNot(first, second)
        self.assertEqual(second.table.rowCount(), 1)
        self.assertEqual(first.table.rowCount(), 1)
        self.viewer._clear_session_peaks(second_doc.uid)
        self.assertEqual(second.table.rowCount(), 0)
        self.assertEqual(first.table.rowCount(), 1)

    def test_overlaid_preview_does_not_resize_plot(self):
        self.viewer.resize(1150, 750)
        self.viewer.show()
        self.app.processEvents()
        before = self.viewer.plot_stack.size()
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.5, 0, 22, 150)
        self.app.processEvents()
        self.assertEqual(self.viewer.plot_stack.size(), before)
        self.assertIs(self.viewer.peak_preview_bar.parentWidget(), self.viewer.plot_stack)
        self.assertEqual(len(self.viewer._pending_peak_fit.result.peaks), 2)
        self.assertTrue(self.viewer.peak_preview_bar.isVisible())

    def test_filling_and_hkl_are_per_peak_and_per_measurement(self):
        uid = self.document.uid
        self._insert_peak(uid, 21.0, 42)
        self.viewer.open_peak_table()
        table = self.viewer._peak_table_dialog.table
        self.assertEqual(table.item(0, 7).checkState(), Qt.CheckState.Checked)
        self.viewer._set_peak_fill(uid, 42, False)
        self.assertEqual(table.item(0, 7).checkState(), Qt.CheckState.Unchecked)
        self.assertEqual(list(self.viewer._peak_shades(self.viewer.items[uid], 0)), [])
        from types import SimpleNamespace
        structure_uid = self.store.add_cif_document(
            Path("phase.cif"), SimpleNamespace(name="First phase")).uid
        self.viewer._assign_peak_hkl(uid, 42, structure_uid, 1, 1, 1)
        self.assertIn("First phase (1 1 1)", table.item(0, 8).text())
        second_uid = self.store.add_cif_document(
            Path("second-phase.cif"), SimpleNamespace(name="Second phase")).uid
        self.viewer._assign_peak_hkl(uid, 42, second_uid, 1, 3, 1)
        self.assertIn("Second phase (1 3 1)", table.item(0, 8).text())
        self.assertEqual(self.store.analysis.peaks[0].hkl_assignments,
                         ((structure_uid, 1, 1, 1), (second_uid, 1, 3, 1)))

    def test_background_gap_and_local_fit_are_source_axis_based(self):
        uid = self.document.uid
        item = self.viewer.items[uid]
        self.viewer.background_spacing_spin.setValue(.1)
        self.viewer.calculate_background()
        background = self.viewer._background_for(item)
        self.assertIsNotNone(background)
        item.x_shift = .25
        self.viewer._exclude_background_region(item, 20.9 + .25, 21.1 + .25)
        active_x, _active_y = self.viewer._background_for(item).active_anchors()
        self.assertFalse(np.any((active_x >= 20.9) & (active_x <= 21.1)))
        self.viewer.activate_peak_search()
        self.viewer._fit_peak_bounds(20.6 + .25, 0, 21.25 + .25, 500)
        self.assertAlmostEqual(self.viewer._pending_peak_fit.result.peaks[0].center,
                               21.25, delta=.01)

    def test_mark_existing_alpha2_and_beta_without_fitting_or_adding(self):
        uid = self.document.uid
        center = 21.0
        alpha = companion_two_theta(center, 1.54056, 1.54443)
        beta = companion_two_theta(center, 1.54056, 1.39222)
        second_center = 22.0
        second_alpha = companion_two_theta(second_center, 1.54056, 1.54443)
        self._insert_peak(uid, beta, 11)
        self._insert_peak(uid, alpha, 12)
        self._insert_peak(uid, center, 13)
        self._insert_peak(uid, second_center, 14)
        self._insert_peak(uid, second_alpha, 15)
        self._insert_peak(uid, 23.0, 16)
        self.viewer.open_peak_table()
        table = self.viewer._peak_table_dialog
        self.assertTrue(table.alpha2_button.isEnabled())
        self.assertTrue(table.beta_button.isEnabled())
        with patch("xrd_workbench.ui_qt.viewer_page.CompanionReviewDialog.exec",
                   return_value=1), patch(
                   "xrd_workbench.ui_qt.viewer_page.CompanionReviewDialog.chosen_indices",
                   lambda dialog: list(range(len(dialog.checks)))), patch(
                   "xrd_workbench.ui_qt.viewer_page.fit_voigt_region",
                   side_effect=AssertionError("Companion search must not fit the scan")):
            self.viewer._find_companions(uid, "ka2")
            self.viewer._find_companions(uid, "kb")
        self.assertEqual(len(self.store.analysis.peaks), 6)
        tagged = {p.number: p for p in self.store.analysis.peaks}
        self.assertEqual((tagged[12].kind, tagged[12].parent_number), ("ka2", 13))
        self.assertEqual((tagged[11].kind, tagged[11].parent_number), ("kb", 13))
        self.assertEqual((tagged[15].kind, tagged[15].parent_number), ("ka2", 14))
        self.assertEqual({table.table.item(row, 6).text() for row in range(6)},
                         {"", "Kα2", "Kβ"})
        self.viewer._remove_session_peak(13, uid)
        tagged = {p.number: p for p in self.store.analysis.peaks}
        self.assertEqual((tagged[11].kind, tagged[12].kind), ("primary", "primary"))
        self.assertEqual(tagged[15].kind, "ka2")
        self.viewer.radiation_settings.select_preset("co_ka1")
        self.viewer._refresh_peak_table()
        self.assertTrue(table.alpha2_button.isEnabled())
        self.assertFalse(table.beta_button.isEnabled())
