"""Background continuity, editable anchors, and peak fitting integration."""

import unittest

import numpy as np
from scipy.special import voigt_profile

from xrd_workbench.services.background import BackgroundAnchors, estimate_background
from xrd_workbench.services.peak_fitting import draw_voigt_peak, fit_voigt_region


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(20, 30, 4001)
        self.true_background = 4 + 0.3 * (self.x - 20)
        self.y = self.true_background.copy()
        self.y += 100 * voigt_profile(self.x - 24, .05, .03) / voigt_profile(0, .05, .03)
        self.y += np.random.default_rng(21).normal(0, .3, len(self.x))

    def test_auto_background_does_not_track_the_narrow_peak(self):
        background = estimate_background(self.x, self.y, axis_name="2Theta", spacing=.12)
        self.assertLess(np.median(np.abs(background.values(self.x) - self.true_background)), 1)
        self.assertLess(background.values(np.array([24.]))[0], 8)
        self.assertEqual(background.axis_name, "2Theta")

    def test_excluded_support_points_bridge_a_region_without_losing_raw_data(self):
        background = estimate_background(self.x, self.y, axis_name="2Theta", spacing=.12)
        trimmed = background.without_region(23, 25)
        active_x, active_y = trimmed.active_anchors()
        self.assertTrue(np.all((active_x < 23) | (active_x > 25)))
        self.assertEqual(len(trimmed.source_x), len(background.source_x))
        mid = (active_x[-1] + active_x[0]) / 2
        self.assertAlmostEqual(trimmed.values(np.array([mid]))[0],
                               np.interp(mid, active_x, active_y))
        self.assertTrue(np.isfinite(trimmed.values(self.x)).all())
        self.assertEqual(self.y.size, 4001)

    def test_spacing_controls_anchor_count_and_preserves_exclusions(self):
        coarse = estimate_background(self.x, self.y, axis_name="2Theta", spacing=.5)
        finer = estimate_background(self.x, self.y, axis_name="2Theta", spacing=.1,
                                    excluded=coarse.without_region(23, 25).excluded)
        self.assertGreater(len(finer.source_x), len(coarse.source_x))
        self.assertFalse(np.any((finer.active_anchors()[0] >= 23) &
                                (finer.active_anchors()[0] <= 25)))

    def test_local_voigt_fit_accounts_for_global_background(self):
        background = estimate_background(self.x, self.y, axis_name="2Theta", spacing=.12)
        selected = (self.x > 23.7) & (self.x < 24.3)
        result = fit_voigt_region(self.x[selected], self.y[selected],
                                  background=background.values(self.x[selected]))
        self.assertEqual(len(result.peaks), 1)
        self.assertAlmostEqual(result.peaks[0].center, 24, delta=.005)
        self.assertAlmostEqual(result.peaks[0].height, 100, delta=2)
        np.testing.assert_allclose(result.fitted, result.background + result.peaks[0].profile)

    def test_curved_background_with_broad_peaks_and_scan_boundaries(self):
        x = np.linspace(10, 90, 10001)
        true_background = 20 + .12 * (x - 10) + 13 * np.exp(-((x - 50) / 20) ** 2)
        y = true_background + np.random.default_rng(311).normal(0, .6, x.size)
        for centre, width, height in ((14.8, .08, 400), (38, .12, 300),
                                      (73.2, .1, 270)):
            y += height * voigt_profile(x - centre, width, width * .4) / voigt_profile(
                0, width, width * .4)
        estimates = [estimate_background(x, y, axis_name="2Theta", spacing=spacing)
                     for spacing in (.2, .4)]
        for result in estimates:
            actual = result.values(x)
            self.assertLess(np.median(np.abs(actual - true_background)), 2.5)
            self.assertLess(abs(actual[0] - true_background[0]), 4)
            self.assertLess(abs(actual[-1] - true_background[-1]), 4)
        self.assertLess(np.median(np.abs(estimates[0].values(x) -
                                         estimates[1].values(x))), 1)

    def test_wide_bands_are_background_but_superimposed_narrow_peaks_are_not(self):
        x = np.linspace(10, 80, 7001)
        sigmoid = lambda values: 1.0 / (1.0 + np.exp(np.clip(-values, -100, 100)))
        true_background = (
            230
            + 430 * sigmoid((x - 15) / .4) * np.exp(-np.clip(x - 15, 0, None) / 9)
            + 1300 * sigmoid((x - 31) / .6) * np.exp(-np.clip(x - 33, 0, None) / 6)
            + 230 * sigmoid((x - 67) / 1.7)
        )
        y = true_background + np.random.default_rng(12).normal(0, 15, x.size)
        for centre, width, height in ((15.1, .1, 2200), (32.1, .55, 15000),
                                      (45, .09, 800), (62.1, .1, 2000),
                                      (69, .14, 16000)):
            y += height * voigt_profile(x - centre, width, .04) / voigt_profile(
                0, width, .04)
        result = estimate_background(x, y, axis_name="2Theta", spacing=.25)
        estimated = result.values(x)
        for location in (20, 26, 34, 40, 45, 55, 70):
            index = np.argmin(abs(x - location))
            self.assertLess(abs(estimated[index] - true_background[index]),
                            max(90, .16 * true_background[index]))
        for location in (15.1, 32.1, 45, 62.1, 69):
            index = np.argmin(abs(x - location))
            self.assertLess(estimated[index], y[index] * .65)

    def test_fitted_peaks_do_not_move_the_shared_background(self):
        x = np.linspace(23.5, 24.5, 501)
        background = 12 + 3 * (x - 24) + 2 * (x - 24) ** 2
        y = background + 120 * voigt_profile(x - 24, .05, .03) / voigt_profile(
            0, .05, .03)
        result = fit_voigt_region(x, y, background=background)
        np.testing.assert_allclose(result.background, np.interp(result.x, x, background))
        self.assertAlmostEqual(result.peaks[0].center, 24, delta=.002)

    def test_drawn_peak_corrects_only_local_common_background(self):
        x = np.linspace(21, 23, 1201)
        source_x = np.linspace(21, 23, 9)
        old = BackgroundAnchors("2Theta", .25, source_x,
                                np.full_like(source_x, 140.0))
        adjusted = old.with_local_level(22, 90, .4)
        self.assertAlmostEqual(adjusted.values(np.asarray([22]))[0], 90.0)
        for place in (21.0, 21.59, 22.41, 23.0):
            np.testing.assert_allclose(adjusted.values(np.asarray([place])),
                                       old.values(np.asarray([place])))
        # Both ends join the original line without a new slope discontinuity.
        self.assertLess(abs(adjusted.values(np.asarray([21.6001]))[0] -
                            adjusted.values(np.asarray([21.5999]))[0]), .001)
        self.assertEqual(len(old.local_levels), 0)  # Preview does not mutate it.

        y = 90 + 55 * voigt_profile(x - 22, .025, .015) / voigt_profile(
            0, .025, .015)
        selected = (x > 21.84) & (x < 22.16)
        fit = draw_voigt_peak(x[selected], adjusted.values(x[selected]),
                              center=22, half_width=.05, height=55)
        self.assertEqual(len(fit.peaks), 1)
        self.assertEqual(fit.peaks[0].center, 22)
        self.assertEqual(fit.peaks[0].height, 55)
        self.assertAlmostEqual(fit.peaks[0].fwhm, .10)
        self.assertAlmostEqual(float(np.interp(22.05, fit.x, fit.peaks[0].profile)),
                               27.5, delta=.1)

    def test_local_reconstruction_ignores_peak_contaminated_anchors(self):
        x = np.array([21.0, 21.5, 21.9, 22.0, 22.1, 22.5, 23.0])
        y = np.array([100., 100., 190., 240., 190., 100., 100.])
        old = BackgroundAnchors("2Theta", .1, x, y)
        updated = old.with_local_level(22, 100, .4, excluded_half_width=.11)
        self.assertAlmostEqual(updated.values(np.array([22]))[0], 100)
        self.assertLess(updated.values(np.array([22.1]))[0], 125)
        np.testing.assert_allclose(updated.values(np.array([21.5, 22.5])),
                                   old.values(np.array([21.5, 22.5])))
        self.assertEqual(old.source_y[3], 240)  # Original anchors are retained.


if __name__ == "__main__":
    unittest.main()
