from __future__ import annotations

import unittest

import numpy as np

from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.services.substrate_calibration import fit_substrate_calibration


class SubstrateCalibrationTests(unittest.TestCase):
    def test_known_first_order_reproduces_expected_affine_calibration(self) -> None:
        measured = np.array([23.708, 48.46493, 75.98642])

        result = fit_substrate_calibration(
            measured,
            [1, 2, 3],
            true_position=23.709,
            reference_order=1,
        )

        self.assertEqual(result.mode, "affine")
        self.assertAlmostEqual(result.x_scale, 1.001963333, places=9)
        self.assertAlmostEqual(result.x_shift, -0.044696352, places=9)
        np.testing.assert_allclose(
            result.expected,
            [23.709, 48.517001974, 76.090145365],
            atol=1.0e-8,
        )

    def test_unknown_reference_fits_one_common_angular_shift(self) -> None:
        true_positions = np.rad2deg(
            2.0 * np.arcsin(np.array([1.0, 2.0, 3.0]) * 0.19)
        )
        measured = true_positions - 0.275

        result = fit_substrate_calibration(measured, [1, 2, 3])

        self.assertEqual(result.mode, "common_shift")
        self.assertEqual(result.x_scale, 1.0)
        self.assertAlmostEqual(result.x_shift, 0.275, places=6)
        np.testing.assert_allclose(result.corrected(measured), true_positions, atol=3e-7)
        np.testing.assert_allclose(result.residuals, 0.0, atol=3e-7)

    def test_invalid_peak_families_are_rejected(self) -> None:
        with self.assertRaises(XRDDataError) as too_few:
            fit_substrate_calibration([20.0], [1])
        self.assertEqual(too_few.exception.code, "substrate_calibration_peaks")

        with self.assertRaises(XRDDataError) as duplicate:
            fit_substrate_calibration([20.0, 40.0], [1, 1])
        self.assertEqual(duplicate.exception.code, "substrate_calibration_orders")


if __name__ == "__main__":
    unittest.main()
