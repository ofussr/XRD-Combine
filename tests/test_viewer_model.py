from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.models.viewer import (
    PlotItem,
    ViewerState,
    intensity_limits,
    overlay_phase_geometry,
    resolve_limits,
    scan_x_limits,
    scrolled_limits,
    scrollbar_window,
    transformed_intensity,
)


def scan_item(uid: str, *, axis: str = "2Theta", visible: bool = True) -> PlotItem:
    scan = Scan1D(
        uid,
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 4.0, 9.0]),
        Path(f"{uid}.xy"),
        axis_name=axis,
    )
    return PlotItem(uid, uid, "measurement", scan.source, "#123456", visible, scan=scan)


def phase_item(uid: str, *, visible: bool = True) -> PlotItem:
    return PlotItem(
        uid,
        uid,
        "CIF",
        Path(f"{uid}.cif"),
        "#654321",
        visible,
        structure=object(),
    )


class ViewerModelTests(unittest.TestCase):
    def test_state_keeps_measurement_and_phase_order_independent(self) -> None:
        state = ViewerState()
        for item in (
            scan_item("scan-a"),
            phase_item("phase-a"),
            scan_item("scan-b"),
            phase_item("phase-b"),
        ):
            state.add(item)

        order = state.move_to_target("scan-a", "scan-b")

        self.assertEqual(order, ["scan-b", "phase-a", "scan-a", "phase-b"])
        self.assertEqual(state.group_uids("scan-a"), ["scan-b", "scan-a"])
        self.assertEqual(state.group_uids("phase-a"), ["phase-a", "phase-b"])

    def test_visibility_and_axis_compatibility_are_viewer_local(self) -> None:
        state = ViewerState()
        state.add(scan_item("two-theta"))
        state.add(scan_item("rocking", axis="Omega"))
        state.add(phase_item("hidden-phase", visible=False))

        self.assertFalse(state.cif_axes_compatible())
        self.assertEqual([item.uid for item in state.visible_phases()], [])
        state.toggle("rocking")
        self.assertTrue(state.cif_axes_compatible())
        state.show_all()
        self.assertFalse(state.cif_axes_compatible())
        self.assertEqual([item.uid for item in state.visible_phases()], ["hidden-phase"])

    def test_display_transform_does_not_change_source_arrays(self) -> None:
        item = scan_item("scan")
        original_x = item.scan.x.copy()
        original_y = item.scan.y.copy()
        item.x_shift = 0.25
        item.y_shift = -2.0
        item.y_factor = 3.0

        x_values, y_values = item.display_arrays()

        np.testing.assert_allclose(x_values, [10.25, 20.25, 30.25])
        np.testing.assert_allclose(y_values, [1.0, 10.0, 25.0])
        np.testing.assert_array_equal(item.scan.x, original_x)
        np.testing.assert_array_equal(item.scan.y, original_y)
        item.reset_transform()
        self.assertEqual((item.x_shift, item.y_shift, item.y_factor), (0.0, 0.0, 1.0))
        self.assertTrue(item.shift_omega)

    def test_intensity_modes_keep_logarithmic_values_physical(self) -> None:
        values = np.array([-1.0, 0.0, 1.0, 4.0])
        linear = transformed_intensity(values, "linear")
        logarithmic = transformed_intensity(values, "log")

        np.testing.assert_array_equal(linear, values)
        self.assertTrue(np.isnan(logarithmic[:2]).all())
        np.testing.assert_array_equal(logarithmic[2:], [1.0, 4.0])
        np.testing.assert_allclose(transformed_intensity(values, "sqrt"), [0, 0, 1, 2])
        np.testing.assert_allclose(transformed_intensity(values, "square"), [1, 0, 1, 16])

    def test_intensity_limits_ignore_nonpositive_values_only_for_log_scale(self) -> None:
        values = [np.array([-5.0, 0.0, 2.0, 10.0, np.nan])]
        linear = intensity_limits(values)
        logarithmic = intensity_limits(values, logarithmic=True)

        self.assertLess(linear[0], -5.0)
        self.assertGreater(linear[1], 10.0)
        self.assertGreater(logarithmic[0], 0.0)
        self.assertLess(logarithmic[0], 2.0)
        self.assertGreater(logarithmic[1], 10.0)

    def test_shifted_data_limits_accept_valid_manual_overrides(self) -> None:
        first = scan_item("first")
        second = scan_item("second")
        first.x_shift = -1.0
        second.x_shift = 5.0
        automatic = scan_x_limits([first, second])

        self.assertEqual(automatic, (9.0, 35.0))
        self.assertEqual(resolve_limits(automatic, 12.0, None), (12.0, 35.0))
        with self.assertRaises(XRDDataError) as caught:
            resolve_limits(automatic, 40.0, 20.0)
        self.assertEqual(caught.exception.code, "viewer_limits_order")

    def test_phase_height_is_clamped_in_shared_plot_state(self) -> None:
        state = ViewerState()
        self.assertEqual(state.plot.phase_layout, "overlay")
        self.assertEqual(state.plot.intensity_scale, "log")
        self.assertTrue(state.cif_axes_compatible())
        self.assertTrue(state.plot.overlay_single_line)
        self.assertEqual(state.plot.overlay_height_percent, 10.0)
        self.assertEqual(state.plot.set_phase_height(5.0), 10.0)
        self.assertEqual(state.plot.set_phase_height(90.0), 85.0)
        self.assertEqual(state.plot.set_overlay_height(0.0), 1.0)
        self.assertEqual(state.plot.set_overlay_height(150.0), 100.0)
        with self.assertRaises(XRDDataError):
            state.plot.set_phase_height(float("nan"))

    def test_overlay_geometry_can_share_or_separate_phase_baselines(self) -> None:
        shared, shared_top = overlay_phase_geometry(
            3,
            single_line=True,
            height_percent=25.0,
        )
        rows, rows_top = overlay_phase_geometry(
            3,
            single_line=False,
            height_percent=25.0,
        )

        self.assertEqual(shared, [(0.0, 0.25)] * 3)
        self.assertEqual(shared_top, 0.25)
        self.assertEqual(len({baseline for baseline, _height in rows}), 3)
        self.assertGreater(rows_top, max(baseline + height for baseline, height in rows))

    def test_horizontal_scrollbar_moves_inside_full_bounds(self) -> None:
        fractions = scrollbar_window((0.0, 100.0), (20.0, 40.0))
        self.assertEqual(fractions, (0.2, 0.4, True))
        self.assertEqual(
            scrolled_limits((0.0, 100.0), (20.0, 40.0), "moveto", 0.5),
            (50.0, 70.0),
        )
        self.assertEqual(
            scrolled_limits((0.0, 100.0), (20.0, 40.0), "scroll", 1),
            (22.0, 42.0),
        )

    def test_vertical_scrollbar_uses_top_to_high_y_direction(self) -> None:
        fractions = scrollbar_window(
            (0.0, 100.0),
            (60.0, 80.0),
            vertical=True,
        )
        self.assertEqual(fractions, (0.2, 0.4, True))
        self.assertEqual(
            scrolled_limits(
                (0.0, 100.0),
                (60.0, 80.0),
                "moveto",
                0.5,
                vertical=True,
            ),
            (30.0, 50.0),
        )

    def test_colour_sequence_restarts_after_clear(self) -> None:
        state = ViewerState(colours=("red", "blue"))
        self.assertEqual([state.next_colour(), state.next_colour()], ["red", "blue"])
        state.clear()
        self.assertEqual(state.next_colour(), "red")


if __name__ == "__main__":
    unittest.main()
