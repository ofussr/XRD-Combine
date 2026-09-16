from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.models.substrate_compare import (
    ComparisonAssembly,
    ComparisonWorkspace,
)
from xrd_workbench.services.substrate_compare import (
    comparison_palette,
    comparison_preset_for,
    prepare_comparison_plot,
    transform_comparison_y,
)


def make_scan(name: str, axis_name: str = "2Theta") -> Scan1D:
    return Scan1D(
        name=name,
        x=np.array([20.0, 21.0, 22.0]),
        y=np.array([1.0, 4.0, 1.0]),
        source=Path(f"{name}.xy"),
        axis_name=axis_name,
    )


def add_item(
    state: ComparisonWorkspace,
    kind: str,
    name: str,
    x: int,
    y: int,
):
    return state.add_item(
        kind=kind,
        name=name,
        scan=make_scan(name),
        x=x,
        y=y,
        width=180,
        height=30,
    )


class SubstrateComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ComparisonWorkspace()

    def test_drag_group_state_is_toolkit_independent(self) -> None:
        measurement = add_item(self.state, "file", "sample", 400, 40)
        substrate = add_item(self.state, "substrate", "LN", 20, 40)
        group_id = self.state.create_group(substrate, measurement)

        self.assertEqual(group_id, 1)
        self.assertEqual(measurement.group, group_id)
        self.assertEqual(substrate.group, group_id)
        self.assertTrue(self.state.group_active[group_id])
        self.assertEqual(measurement.y, 40)
        self.assertEqual(substrate.y, 70)

        second = add_item(self.state, "file", "sample_2", 400, 80)
        self.state.add_to_group(group_id, second)
        self.assertEqual(len(self.state.groups[group_id]), 3)
        self.state.set_group_active(group_id, False)
        self.assertTrue(all(not item.active for item in self.state.groups[group_id]))

        self.state.remove_from_group(second)
        self.assertIsNone(second.group)
        self.assertEqual(len(self.state.groups[group_id]), 2)

    def test_overlap_reset_and_unique_names_do_not_need_widgets(self) -> None:
        first = add_item(self.state, "file", "sample", 10, 20)
        second = add_item(self.state, "substrate", "sample", 100, 35)
        self.assertEqual(self.state.unique_name("sample"), "sample_1")
        self.assertEqual(self.state.overlap_area(first, second), 1350)

        self.state.create_group(first, second)
        self.state.move_item(first, 500, 600)
        self.state.reset_layout()
        self.assertEqual((first.x, first.y), (10, 20))
        self.assertIsNone(first.group)
        self.assertFalse(first.active)
        self.assertEqual(self.state.groups, {})

    def test_active_assemblies_preserve_view_and_plot_order(self) -> None:
        substrate = add_item(self.state, "substrate", "LN", 20, 40)
        measurement = add_item(self.state, "file", "sample", 400, 40)
        group_id = self.state.create_group(substrate, measurement)
        self.state.group_views[group_id] = {"ymode": "linear"}

        assembly = self.state.active_assemblies()[0]
        self.assertEqual([item.kind for item in assembly.items], ["file", "substrate"])
        self.assertEqual(assembly.view, {"ymode": "linear"})
        self.state.set_group_active(group_id, False)
        self.assertEqual(self.state.active_assemblies(), [])

    def test_palette_y_modes_and_presets_are_pure(self) -> None:
        colours = comparison_palette(9, ("#000000", "#ffffff"))
        self.assertEqual(len(colours), 9)
        values = np.array([0.0, 1.0, 2.0])
        transformed = transform_comparison_y(values, "log")
        self.assertTrue(np.all(transformed > 0))
        np.testing.assert_allclose(values, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            transform_comparison_y(values, "square"),
            [0.0, 1.0, 4.0],
        )
        presets = {"LN": {"xlim": [10.0, 80.0], "ylim": [1.0, 100.0]}}
        self.assertIs(comparison_preset_for("LN_copy", presets), presets["LN"])
        self.assertIs(comparison_preset_for("LN_2", presets), presets["LN"])
        self.assertIsNone(comparison_preset_for("other", presets))

    def test_plot_service_prepares_series_axes_and_limits(self) -> None:
        substrate = add_item(self.state, "substrate", "LN_copy", 20, 40)
        measurement = add_item(self.state, "file", "sample", 400, 40)
        assembly = ComparisonAssembly(1, "sample + LN_copy", [measurement, substrate])
        presets = {"LN": {"xlim": [15.0, 75.0], "ylim": [0.5, 1000.0]}}

        plot = prepare_comparison_plot(assembly, presets)
        self.assertEqual([line.item.kind for line in plot.series], ["substrate", "file"])
        self.assertEqual(plot.series[1].colour, "#e31a1c")
        self.assertEqual(plot.x_label, "2θ, °")
        self.assertEqual(plot.y_mode, "log")
        self.assertEqual(plot.y_scale, "log")
        self.assertEqual(plot.x_limits, [15.0, 75.0])
        self.assertEqual(plot.y_limits, [0.5, 1000.0])
        self.assertFalse(plot.uses_automatic_limits)

        explicit = prepare_comparison_plot(
            assembly,
            presets,
            {"ymode": "square", "xlim": [20.0, 30.0], "ylim": [0.0, 20.0]},
            thumbnail=True,
        )
        self.assertEqual(explicit.y_scale, "linear")
        self.assertEqual(explicit.series[0].linewidth, 1.0)
        np.testing.assert_allclose(explicit.series[1].y, [1.0, 16.0, 1.0])

if __name__ == "__main__":
    unittest.main()
