from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xrd_workbench.io.reflections import (
    read_scattering_factors,
    scattering_factor_path,
    write_reflection_csv,
)
from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.models.diffraction import (
    DiffractionAtom,
    DiffractionStructure,
    ReflectionRow,
)
from xrd_workbench.services.diffraction import (
    calculate_reflections,
    d_spacing_from_two_theta,
    gaussian_powder_profile,
)


class DiffractionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.structure = DiffractionStructure(
            "Si",
            (5.431, 5.431, 5.431, 90.0, 90.0, 90.0),
            [DiffractionAtom("Si", 0.0, 0.0, 0.0)],
            ["x,y,z"],
        )
        self.factors = read_scattering_factors(scattering_factor_path())

    def test_bragg_spacing_from_two_theta_rejects_nonphysical_coordinates(self) -> None:
        expected = 1.54056 / (2.0 * np.sin(np.radians(20.0)))
        self.assertAlmostEqual(d_spacing_from_two_theta(40.0, 1.54056), expected)
        self.assertIsNone(d_spacing_from_two_theta(0.0, 1.54056))
        self.assertIsNone(d_spacing_from_two_theta(181.0, 1.54056))
        self.assertIsNone(d_spacing_from_two_theta(40.0, -1.0))

    def test_cubic_reference_matches_pre_extraction_results(self) -> None:
        rows = calculate_reflections(
            self.structure,
            self.factors,
            [("Cu Kα1", 1.54056, 1.0)],
            5.0,
            120.0,
            0.1,
        )

        self.assertEqual(len(rows), 32)
        self.assertAlmostEqual(sum(row.two_theta for row in rows), 2349.1778451000123)
        self.assertAlmostEqual(sum(row.d for row in rows), 51.995957372309505)
        self.assertAlmostEqual(sum(row.intensity or 0.0 for row in rows), 491.76209744162156)
        self.assertEqual(rows[0].hkl, "(0 0 1)+(0 1 0)+(1 0 0)")
        self.assertAlmostEqual(rows[0].two_theta, 16.30753571485289)
        self.assertAlmostEqual(rows[0].intensity or 0.0, 100.0)

    def test_second_radiation_preserves_each_line_and_normalization(self) -> None:
        rows = calculate_reflections(
            self.structure,
            self.factors,
            [
                ("Cu Kα1", 1.54056, 1.0),
                ("Cu Kα2", 1.54443, 0.5),
            ],
        )

        self.assertEqual(len(rows), 64)
        self.assertEqual(rows[1].radiation, "Cu Kα2")
        self.assertAlmostEqual(rows[1].two_theta, 16.34878131512667)
        self.assertAlmostEqual(rows[1].intensity or 0.0, 49.742245565219335)

    def test_gaussian_profile_returns_total_and_radiation_components(self) -> None:
        rows = calculate_reflections(
            self.structure,
            self.factors,
            [
                ("Cu Kα1", 1.54056, 1.0),
                ("Cu Kα2", 1.54443, 0.5),
            ],
            5.0,
            40.0,
        )
        profile = gaussian_powder_profile(
            rows,
            5.0,
            40.0,
            0.15,
            point_count=5000,
        )

        self.assertEqual(profile.x.shape, (5000,))
        self.assertEqual(set(profile.components), {"Cu Kα1", "Cu Kα2"})
        np.testing.assert_allclose(
            profile.total,
            sum(profile.components.values(), np.zeros_like(profile.x)),
        )
        self.assertAlmostEqual(float(np.max(profile.total)), 100.0)

    def test_cell_only_reflections_keep_intensity_unavailable(self) -> None:
        structure = DiffractionStructure(
            "P cell",
            (4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
            [],
            ["x,y,z"],
            cell_only=True,
            space_group="P 1",
        )
        rows = calculate_reflections(
            structure,
            {},
            [("Cu Kα1", 1.54056, 1.0)],
            5.0,
            80.0,
        )
        self.assertTrue(rows)
        self.assertTrue(all(row.intensity is None for row in rows))
        self.assertTrue(all(row.f2_sum is None for row in rows))

    def test_errors_are_structured_and_toolkit_independent(self) -> None:
        with self.assertRaises(XRDDataError) as caught:
            calculate_reflections(self.structure, self.factors, [])
        self.assertEqual(caught.exception.code, "diffraction_no_radiation")

        with self.assertRaises(XRDDataError) as caught:
            gaussian_powder_profile([], 5.0, 4.0, 0.1)
        self.assertEqual(caught.exception.code, "diffraction_profile_limits")

    def test_csv_writer_uses_caller_headers_and_preserves_missing_values(self) -> None:
        row = ReflectionRow(
            "(1 0 0)",
            4.0,
            22.0,
            "Custom",
            1.5,
            1.0,
            2,
            None,
            None,
        )
        headers = [str(index) for index in range(9)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reflections.csv"
            write_reflection_csv(path, [row], headers)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                table = list(csv.reader(stream, delimiter=";"))
        self.assertEqual(table[0], headers)
        self.assertEqual(table[1][-2:], ["—", "—"])


if __name__ == "__main__":
    unittest.main()
