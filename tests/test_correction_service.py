from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from xrd_workbench.io.correction import (
    write_processed_scan,
    write_processed_xrdml,
    write_processed_xy,
)
from xrd_workbench.io.xrdml import read_xrdml
from xrd_workbench.models.correction import CorrectionRequest, validate_result_mode
from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.correction import apply_correction
from xrd_workbench.services.peak_fitting import fit_gaussian_peak
from xrd_workbench.services.reference_peaks import (
    read_reference_peaks,
    write_reference_peaks,
)


XRDML = """<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/2.2">
  <xrdMeasurement><sample><name>Coupled</name></sample><scan><dataPoints>
    <positions axis="2Theta"><startPosition>20</startPosition><endPosition>22</endPosition></positions>
    <positions axis="Omega"><startPosition>10</startPosition><endPosition>11</endPosition></positions>
    <counts>3 8 4</counts>
  </dataPoints></scan></xrdMeasurement>
</xrdMeasurements>
"""


class CorrectionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)

    def tearDown(self) -> None:
        self.folder.cleanup()

    def test_request_rejects_non_finite_values_and_non_positive_scale(self) -> None:
        with self.assertRaises(XRDDataError) as non_finite:
            CorrectionRequest(x_shift=float("nan"))
        self.assertEqual(non_finite.exception.code, "correction_values")
        with self.assertRaises(XRDDataError) as scale:
            CorrectionRequest(y_factor=0.0)
        self.assertEqual(scale.exception.code, "correction_y_factor")
        with self.assertRaises(XRDDataError) as x_scale:
            CorrectionRequest(x_scale=0.0)
        self.assertEqual(x_scale.exception.code, "correction_x_scale")
        self.assertEqual(validate_result_mode("add"), "add")
        with self.assertRaises(XRDDataError):
            validate_result_mode("overwrite")

    def test_apply_correction_preserves_source_and_updates_coupled_axes(self) -> None:
        source = Scan1D(
            "coupled",
            np.array([20.0, 21.0, 22.0]),
            np.array([3.0, 8.0, 4.0]),
            Path("coupled.xrdml"),
            axis_name="2Theta",
            metadata={
                "axes": {
                    "2Theta": np.array([20.0, 21.0, 22.0]),
                    "Omega": np.array([10.0, 10.5, 11.0]),
                }
            },
        )
        corrected = apply_correction(
            source,
            CorrectionRequest(x_shift=0.2, y_shift=1.0, y_factor=2.0),
        )

        self.assertEqual(corrected.name, "coupled shifted")
        np.testing.assert_allclose(corrected.x, [20.2, 21.2, 22.2])
        np.testing.assert_allclose(corrected.y, [7.0, 17.0, 9.0])
        np.testing.assert_allclose(
            corrected.metadata["axes"]["Omega"],
            [10.1, 10.6, 11.1],
        )
        np.testing.assert_allclose(source.x, [20.0, 21.0, 22.0])
        np.testing.assert_allclose(source.y, [3.0, 8.0, 4.0])

    def test_second_correction_composes_metadata_without_repeating_name(self) -> None:
        source = Scan1D(
            "scan",
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            Path("scan.xy"),
        )
        first = apply_correction(
            source,
            CorrectionRequest(x_shift=0.5, y_shift=1.0, y_factor=2.0),
        )
        second = apply_correction(
            first,
            CorrectionRequest(x_shift=-0.2, y_shift=3.0, y_factor=4.0),
        )

        self.assertEqual(second.name, "scan shifted")
        self.assertAlmostEqual(second.metadata["shift"], 0.3)
        self.assertAlmostEqual(second.metadata["y_factor"], 8.0)
        self.assertAlmostEqual(second.metadata["y_shift"], 7.0)
        np.testing.assert_allclose(second.y, [23.0, 39.0])

    def test_affine_x_correction_updates_active_and_coupled_axes(self) -> None:
        source = Scan1D(
            "coupled",
            np.array([20.0, 40.0]),
            np.array([3.0, 4.0]),
            Path("coupled.xrdml"),
            axis_name="2Theta",
            metadata={
                "axes": {
                    "2Theta": np.array([20.0, 40.0]),
                    "Omega": np.array([10.0, 20.0]),
                }
            },
        )

        corrected = apply_correction(
            source,
            CorrectionRequest(x_scale=1.01, x_shift=-0.1),
        )

        np.testing.assert_allclose(corrected.x, [20.1, 40.3])
        np.testing.assert_allclose(
            corrected.metadata["axes"]["Omega"],
            [10.05, 20.15],
        )
        self.assertAlmostEqual(corrected.metadata["x_scale"], 1.01)
        self.assertAlmostEqual(corrected.metadata["shift"], -0.1)

    def test_xy_export_uses_corrected_coordinates_and_intensities(self) -> None:
        scan = Scan1D(
            "scan shifted",
            np.array([20.25, 21.25]),
            np.array([6.0, 16.0]),
            Path("scan.raw"),
        )
        destination = self.path / "result.xy"
        write_processed_xy(scan, destination)
        np.testing.assert_allclose(
            np.loadtxt(destination),
            [[20.25, 6.0], [21.25, 16.0]],
        )

    def test_xrdml_export_updates_selected_range_and_preserves_source(self) -> None:
        source_path = self.path / "source.xrdml"
        source_path.write_text(XRDML, encoding="utf-8")
        original = read_xrdml(source_path)[0]
        corrected = apply_correction(
            original,
            CorrectionRequest(x_shift=0.2, y_shift=1.0, y_factor=2.0),
        )
        destination = self.path / "result.xrdml"

        write_processed_xrdml(corrected, destination)

        saved = read_xrdml(destination)[0]
        np.testing.assert_allclose(saved.metadata["axes"]["2Theta"], [20.2, 21.2, 22.2])
        np.testing.assert_allclose(saved.metadata["axes"]["Omega"], [10.1, 10.6, 11.1])
        np.testing.assert_allclose(saved.y, [7.0, 17.0, 9.0])
        unchanged = read_xrdml(source_path)[0]
        np.testing.assert_allclose(unchanged.x, [20.0, 21.0, 22.0])
        np.testing.assert_allclose(unchanged.y, [3.0, 8.0, 4.0])

    def test_generic_export_rejects_xrdml_for_non_xrdml_source(self) -> None:
        scan = Scan1D(
            "scan",
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            Path("scan.raw"),
        )
        with self.assertRaises(XRDDataError) as caught:
            write_processed_scan(scan, self.path / "invalid.xrdml")
        self.assertEqual(caught.exception.code, "correction_xrdml_source")

    def test_peak_fit_is_available_without_importing_tkinter(self) -> None:
        x = np.linspace(28.0, 32.0, 401)
        y = 20.0 + 0.7 * (x - 30.0) + 180.0 * np.exp(
            -0.5 * ((x - 30.35) / 0.16) ** 2
        )
        fit_x, fit_y, center, intensity = fit_gaussian_peak(x, y)
        self.assertEqual((fit_x.size, fit_y.size), (500, 500))
        self.assertAlmostEqual(center, 30.35, places=5)
        self.assertGreater(intensity, 195.0)
        with self.assertRaises(XRDDataError) as caught:
            fit_gaussian_peak(x[:5], y[:5])
        self.assertEqual(caught.exception.code, "peak_fit_points")

    def test_reference_peak_storage_normalizes_numeric_values(self) -> None:
        destination = self.path / "peaks.json"
        write_reference_peaks(destination, {"Si": 28, "Al2O3": 41.68})
        self.assertEqual(
            read_reference_peaks(destination),
            {"Si": 28.0, "Al2O3": 41.68},
        )
        destination.write_text("not json", encoding="utf-8")
        self.assertEqual(read_reference_peaks(destination), {})


if __name__ == "__main__":
    unittest.main()
