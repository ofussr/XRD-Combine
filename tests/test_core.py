from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xrd_workbench.bruker_raw import RawRange
from xrd_workbench.atom_styles import (
    ELEMENTS,
    FALLBACK_COLOUR,
    atom_ball_radius,
    atom_colour,
    parse_custom_colours,
)
from xrd_workbench.cell_phase import create_cell_phase_document
from xrd_workbench.io import (
    read_cif_data,
    read_raw_scans,
    read_scattering_factors,
    read_xrdml,
    scattering_factor_path,
)
from xrd_workbench.localization import choice_code, set_language, translate_text
from xrd_workbench.cif_document import load_cif_document
from xrd_workbench.models.correction import CorrectionRequest
from xrd_workbench.models.crystal import Crystal
from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.models.project import (
    CELL_PHASE,
    CIF,
    POLE_DATA,
    POLES,
    SCAN,
    STRUCTURES,
    VIEWER,
    ProjectStore,
)
from xrd_workbench.models.pole_figure import CalculatedPoleLayer, PolePoint, PoleReflection
from xrd_workbench.models.radiation import PRESETS, RadiationSettings
from xrd_workbench.models.scan import Scan1D, clone_scan
from xrd_workbench.models.viewer import positive_data_x_limits
from xrd_workbench.services.correction import apply_correction
from xrd_workbench.services.diffraction import calculate_reflections, format_hkl_family
from xrd_workbench.services.experimental_pole import scaled_axes_position
from xrd_workbench.services.peak_fitting import fit_gaussian_peak
from xrd_workbench.services.project_files import ProjectFileService
from xrd_workbench.space_groups import SETTINGS, setting_from_user_text
from xrd_workbench.services.pole_figure import (
    available_reflections,
    base_orientation,
    calculated_intensity_by_spacing,
    euler_matrix,
    follow_orientation_change,
    in_plane_alignment,
    pole_display_position,
    pole_display_orientation,
    pole_plot_coordinates,
    pole_plot_to_sphere,
    project_reflections,
    marker_sizes_by_d,
    place_label_boxes,
)


P1_CIF = """\
data_Si
_chemical_formula_sum 'Si'
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_it_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
"""

P1_CIF_WITH_ESCAPED_APOSTROPHE = """\
data_Si
loop_
_publ_author_name
'D\\'eputier, S.'
_chemical_formula_sum 'Si'
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_it_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
"""


NON_P1_WITHOUT_OPERATIONS = """\
data_bad
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P m -3 m'
_space_group_IT_number 221
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
X1 Si 0 0 0
"""


XRDML = """\
<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/2.2">
  <xrdMeasurement>
    <sample><name>Тест</name></sample>
    <scan>
      <dataPoints>
        <positions axis="2Theta">
          <startPosition>20</startPosition>
          <endPosition>24</endPosition>
        </positions>
        <counts>1 4 9 16 25</counts>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""

ROCKING_XRDML = """\
<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/2.2">
  <xrdMeasurement>
    <sample><name>Rocking</name></sample>
    <scan>
      <dataPoints>
        <positions axis="2Theta">
          <commonPosition>40</commonPosition>
        </positions>
        <positions axis="Omega">
          <startPosition>19</startPosition>
          <endPosition>21</endPosition>
        </positions>
        <counts>1 4 9 4 1</counts>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""


class CoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_language("en")

    def test_interface_languages_and_choice_codes(self) -> None:
        expected = {
            "en": "Correction",
            "fr": "Correction",
            "ru": "Коррекция",
        }
        for language, label in expected.items():
            set_language(language)
            self.assertEqual(translate_text("Коррекция"), label)
        self.assertEqual(choice_code("scale", "Logarithmique"), "log")
        self.assertEqual(choice_code("phase", "Raies"), "sticks")
        self.assertEqual(choice_code("projection", "Equal-area"), "equal_area")
        set_language("en")
        self.assertEqual(
            translate_text("В измерении нет конечных значений интенсивности."),
            "The measurement contains no finite intensity values.",
        )
        self.assertEqual(
            translate_text("Размер точек по d"),
            "Point size by d",
        )
        self.assertEqual(translate_text("Просмотр"), "Viewer")
        self.assertEqual(translate_text("Данные проекта"), "Project data")
        self.assertEqual(translate_text("Заменить исходный"), "Replace source")

    def test_compact_atom_styles_and_custom_colour_parser(self) -> None:
        self.assertEqual(len(ELEMENTS), 118)
        self.assertEqual(ELEMENTS["O"]["jmol"].upper(), "#FF0D0D")
        self.assertGreater(atom_ball_radius("K"), atom_ball_radius("O"))
        self.assertEqual(atom_colour("Xx"), FALLBACK_COLOUR)
        self.assertEqual(
            parse_custom_colours(("Fe #123abc", "O 1 2 3")),
            {"Fe": "#123ABC", "O": "#010203"},
        )
        with self.assertRaisesRegex(ValueError, "unknown element"):
            parse_custom_colours(("Xx #123456",))
        with self.assertRaisesRegex(ValueError, "duplicate element"):
            parse_custom_colours(("Fe #123456", "Fe #654321"))

    def test_cell_phase_settings_absences_and_unavailable_intensity(self) -> None:
        r3c_h = next(
            setting
            for setting in SETTINGS
            if setting.number == 161 and setting.choice == "H"
        )
        r3c_r = next(
            setting
            for setting in SETTINGS
            if setting.number == 161 and setting.choice == "R"
        )
        self.assertEqual(len(SETTINGS), 530)
        self.assertNotEqual(r3c_h.operations, r3c_r.operations)
        document = create_cell_phase_document(
            "R3c cell",
            r3c_h,
            (5.14829, 5.14829, 13.8631, 90.0, 90.0, 120.0),
        )
        self.assertTrue(document.crystal.is_systematically_absent((0, 0, 3)))
        self.assertFalse(document.crystal.is_systematically_absent((0, 0, 6)))
        rows = calculate_reflections(
            document.diffraction,
            {},
            [("Cu Kα1", 1.54056, 1.0)],
            5.0,
            80.0,
            99.0,
        )
        self.assertTrue(rows)
        self.assertTrue(all(row.f2_sum is None for row in rows))
        self.assertTrue(all(row.intensity is None for row in rows))

        store = ProjectStore()
        project_document = store.add_cell_phase(document)
        self.assertEqual(project_document.kind, CELL_PHASE)
        for workspace in (VIEWER, STRUCTURES, POLES):
            self.assertTrue(store.compatible(CELL_PHASE, workspace))
            store.assign(project_document.uid, workspace)
            self.assertTrue(store.is_assigned(project_document.uid, workspace))

    def test_space_group_manual_input_and_single_document_sections(self) -> None:
        self.assertEqual(setting_from_user_text("161").choice, "H")
        self.assertEqual(setting_from_user_text("R3c [R]").choice, "R")
        self.assertEqual(setting_from_user_text("Hall 452").choice, "H")
        self.assertEqual(setting_from_user_text("P 1 1 2_1/a").choice, "c1")
        with self.assertRaises(ValueError):
            setting_from_user_text("not a space group")

        p1 = SETTINGS[0]
        first = create_cell_phase_document(
            "first", p1, (4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
        )
        second = create_cell_phase_document(
            "second", p1, (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        )
        store = ProjectStore()
        first_document = store.add_cell_phase(first)
        second_document = store.add_cell_phase(second)

        for uid in (first_document.uid, second_document.uid):
            store.assign(uid, VIEWER)
        self.assertEqual(
            {item.uid for item in store.assigned_documents(VIEWER)},
            {first_document.uid, second_document.uid},
        )

        for workspace in (STRUCTURES, POLES):
            store.assign(first_document.uid, workspace)
            store.assign(second_document.uid, workspace)
            self.assertFalse(store.is_assigned(first_document.uid, workspace))
            self.assertTrue(store.is_assigned(second_document.uid, workspace))

    def test_calculated_poles_allow_one_explicit_overlay_only(self) -> None:
        p1 = SETTINGS[0]
        store = ProjectStore()
        documents = [
            store.add_cell_phase(
                create_cell_phase_document(
                    f"phase {index}",
                    p1,
                    (4.0 + index, 4.0, 4.0, 90.0, 90.0, 90.0),
                )
            )
            for index in range(3)
        ]
        store.assign(documents[0].uid, POLES)
        store.assign(documents[1].uid, POLES, additive=True)
        self.assertEqual(
            {document.uid for document in store.assigned_documents(POLES)},
            {documents[0].uid, documents[1].uid},
        )
        with self.assertRaises(ValueError):
            store.assign(documents[2].uid, POLES, additive=True)
        store.assign(documents[2].uid, POLES)
        self.assertEqual(store.assigned_documents(POLES), [documents[2]])

    def test_calculated_pole_layer_keeps_independent_style_and_orientation(self) -> None:
        document = SimpleNamespace(name="phase", crystal=object())
        layer = CalculatedPoleLayer(
            document,
            "#d65f3c",
            opacity_percent=70.0,
            size_percent=125.0,
        )
        layer.user_rotation = np.diag((1.0, -1.0, -1.0))
        np.testing.assert_allclose(layer.orientation, layer.user_rotation)
        self.assertAlmostEqual(layer.opacity, 0.7)
        self.assertAlmostEqual(layer.size_scale, 1.25)
        layer.opacity_percent = 500
        layer.size_percent = 1
        self.assertEqual(layer.opacity, 1.0)
        self.assertEqual(layer.size_scale, 0.1)
        self.assertFalse(layer.coupled_to_primary)

    def test_coupled_pole_rotation_preserves_relative_orientation(self) -> None:
        primary_before = euler_matrix(12.0, -7.0, 31.0)
        primary_after = euler_matrix(-18.0, 24.0, 9.0)
        follower_user = euler_matrix(4.0, 16.0, -11.0)
        follower_base = euler_matrix(22.0, -3.0, 8.0)
        follower_after = follow_orientation_change(
            follower_user,
            primary_before,
            primary_after,
        )
        np.testing.assert_allclose(
            primary_before.T @ follower_user @ follower_base,
            primary_after.T @ follower_after @ follower_base,
            atol=1e-12,
        )
        with self.assertRaises(ValueError):
            follow_orientation_change(
                np.eye(2),
                primary_before,
                primary_after,
            )

    def test_calculated_pole_intensity_uses_powder_engine(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "si.cif"
            path.write_text(P1_CIF, encoding="utf-8")
            document = load_cif_document(path)
            intensities = calculated_intensity_by_spacing(
                document.diffraction,
                [("Cu Kα1", 1.54056, 1.0)],
                read_scattering_factors(scattering_factor_path()),
            )
        self.assertTrue(intensities)
        self.assertAlmostEqual(max(intensities.values()), 100.0)
        self.assertTrue(all(0.0 <= value <= 100.0 for value in intensities.values()))

    def test_xrdml_2theta(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scan.xrdml"
            path.write_text(XRDML, encoding="utf-8")
            scans = read_xrdml(path)
        self.assertEqual(len(scans), 1)
        np.testing.assert_allclose(scans[0].x, [20, 21, 22, 23, 24])
        np.testing.assert_allclose(scans[0].y, [1, 4, 9, 16, 25])

    def test_xrdml_rocking_curve_uses_varying_axis(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rocking.xrdml"
            path.write_text(ROCKING_XRDML, encoding="utf-8")
            scan = read_xrdml(path)[0]
        self.assertEqual(scan.axis_name, "Omega")
        self.assertEqual(scan.available_axes, ("2Theta", "Omega"))
        np.testing.assert_allclose(scan.x, [19, 19.5, 20, 20.5, 21])
        scan.use_axis("2Theta")
        np.testing.assert_allclose(scan.x, [40, 40, 40, 40, 40])

    def test_scan_axis_switch_preserves_coordinate_intensity_pairs(self) -> None:
        scan = Scan1D(
            name="axes",
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([10.0, 20.0, 30.0]),
            source=Path("axes.xy"),
            axis_name="Theta",
            metadata={
                "axes": {
                    "Theta": np.array([1.0, 2.0, 3.0]),
                    "Phi": np.array([30.0, 20.0, 10.0]),
                }
            },
        )
        scan.use_axis("Phi")
        np.testing.assert_allclose(scan.x, [10, 20, 30])
        np.testing.assert_allclose(scan.y, [30, 20, 10])
        scan.use_axis("Theta")
        np.testing.assert_allclose(scan.y, [10, 20, 30])

    def test_scan_clone_is_independent_and_preserves_all_axes(self) -> None:
        original = Scan1D(
            name="source",
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([10.0, 20.0, 30.0]),
            source=Path("source.xrdml"),
            axis_name="Theta",
            metadata={
                "axes": {
                    "Theta": np.array([1.0, 2.0, 3.0]),
                    "Phi": np.array([30.0, 20.0, 10.0]),
                },
                "scan_index": 2,
            },
        )
        copied = clone_scan(original)
        copied.use_axis("Phi")
        copied.metadata["axes"]["Theta"][0] = 99.0

        self.assertEqual(copied.available_axes, ("Theta", "Phi"))
        self.assertEqual(copied.metadata["scan_index"], 2)
        np.testing.assert_allclose(copied.y, [30, 20, 10])
        np.testing.assert_allclose(original.x, [1, 2, 3])
        self.assertEqual(original.metadata["axes"]["Theta"][0], 1.0)

    def test_correction_transfer_shifts_only_the_active_axis(self) -> None:
        source = Scan1D(
            name="rocking",
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([10.0, 20.0, 30.0]),
            source=Path("rocking.xrdml"),
            axis_name="Omega",
            metadata={
                "axes": {
                    "Omega": np.array([1.0, 2.0, 3.0]),
                    "2Theta": np.array([40.0, 40.0, 40.0]),
                }
            },
        )
        corrected = apply_correction(source, CorrectionRequest(x_shift=0.25))

        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.name, "rocking shifted")
        self.assertEqual(corrected.axis_name, "Omega")
        np.testing.assert_allclose(corrected.x, [1.25, 2.25, 3.25])
        np.testing.assert_allclose(
            corrected.metadata["axes"]["2Theta"],
            [40.0, 40.0, 40.0],
        )
        np.testing.assert_allclose(corrected.y, [10.0, 20.0, 30.0])

    def test_correction_limits_ignore_isolated_x_outlier(self) -> None:
        limits = positive_data_x_limits(
            np.array([0.0, 20.0, 21.0, 22.0, 23.0]),
            np.ones(5),
        )
        self.assertIsNotNone(limits)
        assert limits is not None
        self.assertGreater(limits[0], 19.0)
        self.assertLess(limits[1], 24.0)

    def test_shared_peak_fit_recovers_center_on_linear_background(self) -> None:
        x = np.linspace(28.0, 32.0, 401)
        y = 20.0 + 0.7 * (x - 30.0) + 180.0 * np.exp(
            -0.5 * ((x - 30.35) / 0.16) ** 2
        )
        fit_x, fit_y, center, intensity = fit_gaussian_peak(x, y)
        self.assertAlmostEqual(center, 30.35, places=5)
        self.assertGreater(intensity, 195.0)
        self.assertEqual(fit_x.size, 500)
        self.assertEqual(fit_y.size, 500)

    def test_marker_sizes_increase_with_d_spacing(self) -> None:
        sizes = marker_sizes_by_d([1.0, 2.0, 3.0])
        self.assertTrue(np.all(np.diff(sizes) > 0))
        self.assertEqual(sizes[0], 22.0)
        self.assertEqual(sizes[-1], 250.0)
        np.testing.assert_allclose(
            marker_sizes_by_d([2.0, 2.0]),
            [58.0, 58.0],
        )

    def test_pole_label_layout_avoids_markers_and_other_labels(self) -> None:
        placements = place_label_boxes(
            [(50.0, 50.0), (50.0, 50.0), (94.0, 50.0)],
            [(30.0, 13.0), (30.0, 13.0), (30.0, 13.0)],
            (0.0, 0.0, 100.0, 100.0),
            protected_boxes=[(44.0, 44.0, 56.0, 56.0)],
            required_indices={0},
        )
        self.assertIsNotNone(placements[0])
        self.assertIsNotNone(placements[1])
        first = (*placements[0], placements[0][0] + 30.0, placements[0][1] + 13.0)
        second = (*placements[1], placements[1][0] + 30.0, placements[1][1] + 13.0)
        self.assertFalse(
            first[0] < second[2]
            and first[2] > second[0]
            and first[1] < second[3]
            and first[3] > second[1]
        )
        self.assertTrue(placements[2] is None or placements[2][0] <= 70.0)

    def test_compact_pole_label_layout_stays_close_to_marker(self) -> None:
        placements = place_label_boxes(
            [(50.0, 50.0), (50.0, 50.0)],
            [(30.0, 13.0), (30.0, 13.0)],
            (0.0, 0.0, 100.0, 100.0),
            protected_boxes=[(44.0, 44.0, 56.0, 56.0)],
            required_indices={0},
            anchor_clearances=[6.0, 6.0],
            candidate_gaps=(1.5, 4.0, 7.0),
        )
        self.assertIsNotNone(placements[0])
        assert placements[0] is not None
        left, bottom = placements[0]
        horizontal_gap = max(left - 50.0, 50.0 - (left + 30.0), 0.0)
        vertical_gap = max(bottom - 50.0, 50.0 - (bottom + 13.0), 0.0)
        self.assertLessEqual(math.hypot(horizontal_gap, vertical_gap), 13.1)

    def test_experimental_pole_zoom_scales_finished_axes_about_cursor(self) -> None:
        position = (0.1, 0.2, 0.6, 0.6)
        cursor = (0.4, 0.5)
        zoomed = scaled_axes_position(position, cursor, 1.25)
        self.assertEqual(zoomed[2:], (0.75, 0.75))
        self.assertAlmostEqual(
            (cursor[0] - zoomed[0]) / zoomed[2],
            (cursor[0] - position[0]) / position[2],
        )
        self.assertAlmostEqual(
            (cursor[1] - zoomed[1]) / zoomed[3],
            (cursor[1] - position[1]) / position[3],
        )

    def test_raw_pole_ranges_are_available_in_the_viewer(self) -> None:
        ranges = [
            RawRange(
                index=index,
                scan_type="Pole Figure",
                axis_name="Phi",
                start_angle=0.0,
                step_size=90.0,
                time_per_step=1.0,
                drives={
                    "Phi": 0.0,
                    "Chi": float(index * 10),
                    "Theta": 20.0,
                    "2Theta": 40.0,
                },
                data=np.array([[1.0], [2.0], [3.0], [4.0]]),
                channel_names=["intensity"],
            )
            for index in range(2)
        ]
        raw = SimpleNamespace(
            ranges=ranges,
            version=3,
            is_pole_figure=True,
        )
        scans = read_raw_scans("pole.raw", raw=raw)
        self.assertEqual(len(scans), 2)
        self.assertTrue(all(scan.axis_name == "Phi" for scan in scans))
        self.assertIn("Chi", scans[0].available_axes)

    def test_project_store_shares_pole_raw_and_keeps_each_viewer_range(self) -> None:
        raw = SimpleNamespace(is_pole_figure=True)
        scans = [
            Scan1D(
                name=f"pole – #{index + 1}",
                x=np.array([0.0, 90.0, 180.0]),
                y=np.array([1.0, 2.0, 3.0]),
                source=Path("pole.raw"),
                axis_name="Phi",
                metadata={"range_index": index},
            )
            for index in range(2)
        ]
        store = ProjectStore()
        service = ProjectFileService(
            load_cif_document=load_cif_document,
            read_bruker_raw=lambda _path: raw,
            read_raw_scans=lambda _path, *, raw: scans,
            read_scan_file=lambda _path: [],
        )
        documents = service.load_path(store, "pole.raw")

        self.assertEqual([document.kind for document in documents], [POLE_DATA, SCAN, SCAN])
        self.assertNotEqual(documents[1].uid, documents[2].uid)
        self.assertEqual(documents[1].parent_uid, documents[0].uid)
        self.assertTrue(store.compatible(POLE_DATA, POLES))
        self.assertTrue(store.compatible(SCAN, VIEWER))

    def test_raw_rocking_curve_is_not_rejected(self) -> None:
        rocking = RawRange(
            index=0,
            scan_type="Rocking Curve",
            axis_name="Theta",
            start_angle=19.8,
            step_size=0.1,
            time_per_step=1.0,
            drives={
                "Theta": 19.8,
                "2Theta": 40.0,
                "Chi": 0.0,
                "Phi": 0.0,
            },
            data=np.array([[1.0], [3.0], [1.0]]),
            channel_names=["intensity"],
        )
        raw = SimpleNamespace(
            ranges=[rocking],
            version=4,
            is_pole_figure=False,
        )
        scan = read_raw_scans("rocking.raw", raw=raw)[0]
        self.assertEqual(scan.axis_name, "Theta")
        np.testing.assert_allclose(scan.x, [19.8, 19.9, 20.0])

    def test_raw_reader_does_not_leak_russian_error_into_english_ui(self) -> None:
        set_language("en")
        with self.assertRaises(XRDDataError) as caught:
            read_raw_scans(
                "broken.raw",
                raw_reader=lambda _path: (_ for _ in ()).throw(
                    ValueError("Внутренняя диагностическая ошибка")
                ),
            )
        self.assertEqual(caught.exception.code, "raw_unreadable")

    def test_verified_cif_calculator_and_spaced_hkl(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "si.cif"
            path.write_text(P1_CIF, encoding="utf-8")
            structure = load_cif_document(path).diffraction
            rows = calculate_reflections(
                structure,
                read_scattering_factors(scattering_factor_path()),
                [("Cu Kα1", 1.54056, 1.0)],
                10,
                90,
            )
        self.assertTrue(rows)
        self.assertTrue(all("(" in row.hkl and " " in row.hkl for row in rows))
        self.assertTrue(all(row.equivalents for row in rows))

    def test_structure_radiation_presets_and_custom_lines(self) -> None:
        presets = {preset.key: preset.lines for preset in PRESETS}
        self.assertEqual(presets["cu_ka1"], (("Cu Kα1", 1.54056, 1.0),))
        self.assertEqual(
            presets["cu_ka12_kb"],
            (
                ("Cu Kα1", 1.54056, 1.0),
                ("Cu Kα2", 1.54443, 0.5),
                ("Cu Kβ", 1.39222, 0.15),
            ),
        )
        self.assertEqual(presets["co_ka1"], (("Co Kα1", 1.78897, 1.0),))
        settings = RadiationSettings()
        settings.profile_key = "other"
        settings.custom_lines = [(f"Custom {i}", 1.0 + i / 10, 1.0) for i in range(1, 6)]
        self.assertEqual(settings.lines(), settings.custom_lines)

    def test_hkl_overflow_marker_is_language_neutral(self) -> None:
        label = format_hkl_family({(index, 0, 1) for index in range(6)})
        self.assertIn("(0 0 1)", label)
        self.assertIn("+… 2", label)
        self.assertNotIn("++", label)
        self.assertNotRegex(label, r"[А-Яа-яA-Za-z]")

    def test_cod_style_escaped_apostrophe_is_valid_in_both_cif_readers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "escaped.cif"
            path.write_text(P1_CIF_WITH_ESCAPED_APOSTROPHE, encoding="utf-8")
            document = load_cif_document(path)
            structure = document.diffraction
            crystal = Crystal.from_cif(read_cif_data(path))
        self.assertEqual(structure.name, "Si")
        self.assertEqual(crystal.formula, "Si")

    def test_cif_lexer_error_is_localized(self) -> None:
        set_language("fr")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad_quote.cif"
            path.write_text("_tag 'sans fin", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_cif_document(path)
        self.assertIn("ligne CIF 1", str(caught.exception))
        self.assertNotRegex(str(caught.exception), r"[А-Яа-я]")

    def test_non_p1_without_operations_is_rejected_in_table(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.cif"
            path.write_text(NON_P1_WITHOUT_OPERATIONS, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cif_document(path)

    def test_non_p1_without_operations_is_rejected_in_pole_tool(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.cif"
            path.write_text(NON_P1_WITHOUT_OPERATIONS, encoding="utf-8")
            with self.assertRaises(ValueError):
                Crystal.from_cif(read_cif_data(path))

    def test_theoretical_pole_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "si.cif"
            path.write_text(P1_CIF, encoding="utf-8")
            crystal = Crystal.from_cif(read_cif_data(path))
            reflections = available_reflections(
                crystal,
                d_lower=1.0,
                d_upper=6.0,
                wavelength=1.54056,
            )
            orientation = base_orientation(crystal, (0, 1, 0))
            points = project_reflections(
                crystal,
                reflections,
                orientation,
                "Стереографическая",
            )
        self.assertGreater(len(reflections), 1)
        self.assertGreater(len(points), 1)

    def test_equatorial_friedel_pair_is_drawn_at_both_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "si.cif"
            path.write_text(P1_CIF, encoding="utf-8")
            crystal = Crystal.from_cif(read_cif_data(path))
            d_spacing = crystal.d_spacing((2, -1, 0))
            reflection = PoleReflection((2, -1, 0), d_spacing, 20.0)
            points = project_reflections(
                crystal,
                [reflection],
                np.eye(3),
                "Стереографическая",
            )
        self.assertEqual(len(points), 2)
        self.assertEqual({point.hkl for point in points}, {(2, -1, 0), (-2, 1, 0)})
        self.assertTrue(all(math.isclose(point.chi, 90.0) for point in points))
        np.testing.assert_allclose(
            [points[0].x + points[1].x, points[0].y + points[1].y],
            [0.0, 0.0],
            atol=1e-12,
        )

    def test_theoretical_pole_alignment_and_boundary_inset(self) -> None:
        direction = np.array(
            [
                math.cos(math.radians(37.0)),
                math.sin(math.radians(37.0)),
                0.0,
            ]
        )
        aligned = in_plane_alignment(37.0, -90.0) @ direction
        np.testing.assert_allclose(aligned[:2], [0.0, -1.0], atol=1e-12)

        point = PolePoint(
            (1, 0, 0),
            1.0,
            20.0,
            np.array([1.0, 0.0, 0.0]),
            90.0,
            0.0,
            1.0,
            0.0,
        )
        display_x, display_y = pole_display_position(point)
        self.assertLess(display_x, 1.0)
        self.assertEqual(display_y, 0.0)
        self.assertEqual(point.x, 1.0)

    def test_theoretical_pole_zero_is_top_without_mirroring(self) -> None:
        top = pole_plot_coordinates(1.0, 0.0)
        left = pole_plot_coordinates(1.0, math.radians(90.0))
        bottom = pole_plot_coordinates(1.0, math.radians(180.0))
        right = pole_plot_coordinates(1.0, math.radians(270.0))

        np.testing.assert_allclose(top, [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(left, [-1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(bottom, [0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(right, [1.0, 0.0], atol=1e-12)

        np.testing.assert_allclose(
            pole_display_orientation(np.eye(3)) @ np.array([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )
        arbitrary_phi = math.radians(37.0)
        pole_xy = pole_plot_coordinates(1.0, arbitrary_phi)
        structure_direction = pole_display_orientation(np.eye(3)) @ np.array(
            [math.cos(arbitrary_phi), math.sin(arbitrary_phi), 0.0]
        )
        np.testing.assert_allclose(
            structure_direction[:2],
            pole_xy,
            atol=1e-12,
        )

        chi = math.radians(40.0)
        phi = math.radians(70.0)
        expected = np.array(
            [
                math.sin(chi) * math.cos(phi),
                math.sin(chi) * math.sin(phi),
                math.cos(chi),
            ]
        )
        stereographic_radius = math.tan(chi / 2.0)
        stereo_x, stereo_y = pole_plot_coordinates(
            stereographic_radius,
            phi,
        )
        np.testing.assert_allclose(
            pole_plot_to_sphere(
                stereo_x,
                stereo_y,
                "Стереографическая",
            ),
            expected,
            atol=1e-12,
        )

        equal_area_radius = math.sqrt(2.0) * math.sin(chi / 2.0)
        equal_x, equal_y = pole_plot_coordinates(equal_area_radius, phi)
        np.testing.assert_allclose(
            pole_plot_to_sphere(equal_x, equal_y, "Равноплощадная"),
            expected,
            atol=1e-12,
        )

    def test_one_cif_document_drives_diffraction_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "phase.cif"
            path.write_text(P1_CIF, encoding="utf-8")
            document = load_cif_document(path)
        self.assertEqual(len(document.crystal.atoms), len(document.diffraction.atoms))
        self.assertEqual(document.crystal.formula, document.diffraction.name)
        self.assertEqual(document.diffraction.atoms[0].element, "Si")

    def test_project_store_assigns_without_duplicating_documents(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "phase.cif"
            path.write_text(P1_CIF, encoding="utf-8")
            store = ProjectStore()
            service = ProjectFileService(
                load_cif_document=load_cif_document,
                read_bruker_raw=lambda _path: None,
                read_raw_scans=lambda _path, **_kwargs: [],
                read_scan_file=lambda _path: [],
            )
            first = service.load_cif(store, path)
            second = service.load_cif(store, path)
            self.assertIs(first, second)
            self.assertEqual(first.kind, CIF)
            store.assign(first.uid, VIEWER)
            store.assign(first.uid, STRUCTURES)
            self.assertEqual(store.assigned_documents(VIEWER), [first])
            self.assertEqual(store.assigned_documents(STRUCTURES), [first])

    def test_project_store_replacement_keeps_identity_and_history(self) -> None:
        store = ProjectStore()
        original = Scan1D(
            "scan",
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            Path("scan.xy"),
        )
        document = store.add_scan(original)
        store.assign(document.uid, VIEWER)
        shifted = clone_scan(original, name="scan shifted")
        shifted.x = shifted.x + 0.1
        store.replace_scan(document.uid, shifted)
        self.assertEqual(document.kind, SCAN)
        self.assertEqual(document.name, "scan shifted")
        self.assertEqual(len(document.history), 1)
        self.assertEqual(store.assigned_documents(VIEWER), [document])


if __name__ == "__main__":
    unittest.main()
