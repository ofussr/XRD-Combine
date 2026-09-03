"""GUI-independent state and numeric transforms for the scan viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data_errors import XRDDataError
from .scan import Scan1D


DEFAULT_PLOT_COLOURS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)


def axis_key(name: str) -> str:
    """Return a stable comparison key for a scan-axis name."""

    return str(name).replace(" ", "").replace("-", "").lower()


def is_two_theta(name: str) -> bool:
    return axis_key(name) in {"2theta", "twotheta", "2θ"}


def axis_has_degree_units(name: str) -> bool:
    return axis_key(name) in {
        "2theta",
        "twotheta",
        "2θ",
        "theta",
        "omega",
        "chi",
        "phi",
    }


@dataclass
class PlotItem:
    """One measurement or phase and its viewer-local presentation state."""

    uid: str
    name: str
    kind: str
    source: Path
    colour: str
    visible: bool = True
    scan: Scan1D | None = None
    structure: Any | None = None
    x_shift: float = 0.0
    y_shift: float = 0.0
    y_factor: float = 1.0
    shift_omega: bool = True

    @property
    def is_measurement(self) -> bool:
        return self.scan is not None

    @property
    def is_phase(self) -> bool:
        return self.structure is not None

    def display_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scan is None:
            return np.empty(0), np.empty(0)
        return (
            np.asarray(self.scan.x, dtype=float) + self.x_shift,
            np.asarray(self.scan.y, dtype=float) * self.y_factor + self.y_shift,
        )

    def reset_transform(self) -> None:
        self.x_shift = 0.0
        self.y_shift = 0.0
        self.y_factor = 1.0
        self.shift_omega = True


@dataclass
class ViewerPlotState:
    """Matplotlib-independent state shared by present and future view adapters."""

    phase_layout: str = "separate"
    phase_style: str = "sticks"
    intensity_scale: str = "linear"
    vertical_offset: float = 0.0
    phase_height_percent: float = 25.0
    axes_linked: bool = True
    navigation_x_bounds: tuple[float, float] = (0.0, 1.0)
    navigation_y_bounds: tuple[float, float] = (0.0, 1.0)
    overlay_phase_top: float = 0.0

    def set_phase_height(self, value: float) -> float:
        if not np.isfinite(value):
            raise XRDDataError("viewer_phase_height")
        self.phase_height_percent = max(10.0, min(85.0, float(value)))
        return self.phase_height_percent


@dataclass
class ViewerState:
    """Ordered collection of objects connected to the Viewer workspace."""

    colours: Sequence[str] = DEFAULT_PLOT_COLOURS
    items: dict[str, PlotItem] = field(default_factory=dict)
    plot: ViewerPlotState = field(default_factory=ViewerPlotState)
    _colour_index: int = 0

    def next_colour(self) -> str:
        if not self.colours:
            raise XRDDataError("viewer_no_colours")
        colour = self.colours[self._colour_index % len(self.colours)]
        self._colour_index += 1
        return colour

    def add(self, item: PlotItem) -> bool:
        if item.uid in self.items:
            return False
        self.items[item.uid] = item
        return True

    def remove(self, uid: str) -> PlotItem | None:
        return self.items.pop(uid, None)

    def clear(self) -> None:
        self.items.clear()
        self._colour_index = 0
        self.plot = ViewerPlotState()

    def group_uids(self, uid: str) -> list[str]:
        item = self.items.get(uid)
        if item is None:
            return []
        return [
            key
            for key, value in self.items.items()
            if value.is_measurement == item.is_measurement
        ]

    def move_to_target(self, uid: str, target: str) -> list[str]:
        """Move within the measurement or phase group, preserving other items."""

        group = self.group_uids(uid)
        if uid == target or target not in group:
            return list(self.items)
        target_index = group.index(target)
        group.remove(uid)
        group.insert(target_index, uid)
        group_keys = set(group)
        iterator = iter(group)
        order = [next(iterator) if key in group_keys else key for key in self.items]
        reordered = {key: self.items[key] for key in order}
        self.items.clear()
        self.items.update(reordered)
        return order

    def visible_scans(self) -> list[PlotItem]:
        return [item for item in self.items.values() if item.visible and item.is_measurement]

    def visible_phases(self) -> list[PlotItem]:
        return [item for item in self.items.values() if item.visible and item.is_phase]

    def cif_axes_compatible(self) -> bool:
        return all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in self.visible_scans()
        )

    def toggle(self, uid: str) -> bool | None:
        item = self.items.get(uid)
        if item is None:
            return None
        item.visible = not item.visible
        return item.visible

    def show_all(self) -> None:
        for item in self.items.values():
            item.visible = True


def transformed_intensity(values: np.ndarray, mode: str) -> np.ndarray:
    """Apply a display-only intensity transform to physical values."""

    array = np.asarray(values, dtype=float)
    if mode == "log":
        return np.where(array > 0, array, np.nan)
    if mode == "sqrt":
        return np.sqrt(np.clip(array, 0, None))
    if mode == "square":
        return np.square(array)
    return array


def scan_x_limits(
    items: Iterable[PlotItem],
    *,
    default: tuple[float, float] = (5.0, 120.0),
) -> tuple[float, float]:
    """Find the complete shifted X range of visible measurements."""

    scans = [item for item in items if item.visible and item.scan is not None]
    if scans:
        minimum = min(float(np.min(item.scan.x)) + item.x_shift for item in scans)
        maximum = max(float(np.max(item.scan.x)) + item.x_shift for item in scans)
    else:
        minimum, maximum = map(float, default)
    if np.isclose(minimum, maximum, rtol=0.0, atol=1e-12):
        padding = max(0.5, abs(minimum) * 0.02)
        minimum -= padding
        maximum += padding
    return minimum, maximum


def resolve_limits(
    automatic: tuple[float, float],
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    """Apply optional manual bounds and validate the resulting interval."""

    low = automatic[0] if minimum is None else float(minimum)
    high = automatic[1] if maximum is None else float(maximum)
    if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
        raise XRDDataError("viewer_limits_order", minimum=low, maximum=high)
    return float(low), float(high)


def scrollbar_window(
    bounds: tuple[float, float],
    current: tuple[float, float],
    *,
    vertical: bool = False,
) -> tuple[float, float, bool]:
    """Return platform-neutral scrollbar fractions for the current view."""

    full_low, full_high = sorted(map(float, bounds))
    current_low, current_high = sorted(map(float, current))
    full_span = full_high - full_low
    current_span = current_high - current_low
    movable = (
        np.isfinite(full_span)
        and np.isfinite(current_span)
        and full_span > 0
        and current_span < full_span * (1.0 - 1e-9)
    )
    if not movable:
        return 0.0, 1.0, False
    if vertical:
        first = (full_high - current_high) / full_span
    else:
        first = (current_low - full_low) / full_span
    size = current_span / full_span
    first = max(0.0, min(1.0 - size, first))
    return first, first + size, True


def scrolled_limits(
    bounds: tuple[float, float],
    current: tuple[float, float],
    command: str,
    value: float,
    units: str = "units",
    *,
    vertical: bool = False,
) -> tuple[float, float]:
    """Calculate a new visible interval from a generic scrollbar command."""

    full_low, full_high = sorted(map(float, bounds))
    current_low, current_high = sorted(map(float, current))
    full_span = full_high - full_low
    view_span = min(current_high - current_low, full_span)
    if not np.isfinite(full_span) or full_span <= 0 or view_span >= full_span:
        return current_low, current_high

    if command == "moveto":
        fraction = float(value)
        if vertical:
            high = full_high - fraction * full_span
            low = high - view_span
        else:
            low = full_low + fraction * full_span
    elif command == "scroll":
        step = view_span * 0.8 if units == "pages" else full_span * 0.02
        direction = -1.0 if vertical else 1.0
        low = current_low + direction * float(value) * step
    else:
        return current_low, current_high

    low = max(full_low, min(full_high - view_span, low))
    return low, low + view_span


__all__ = [
    "DEFAULT_PLOT_COLOURS",
    "PlotItem",
    "ViewerPlotState",
    "ViewerState",
    "axis_has_degree_units",
    "axis_key",
    "is_two_theta",
    "resolve_limits",
    "scan_x_limits",
    "scrolled_limits",
    "scrollbar_window",
    "transformed_intensity",
]
