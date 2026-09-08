"""Prepare substrate-comparison plots without depending on a GUI toolkit."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..models.substrate_compare import ComparisonAssembly, ComparisonItem


FILE_COLOURS = ("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99")
SUBSTRATE_COLOURS = ("#000000", "#404040", "#7f7f7f")


def _rgb_hex(rgb: tuple[float, float, float]) -> str:
    channels = tuple(round(max(0.0, min(1.0, value)) * 255) for value in rgb)
    return "#{:02x}{:02x}{:02x}".format(*channels)


def comparison_palette(count: int, base: tuple[str, ...]) -> list[str]:
    if count <= len(base):
        return list(base[:count])
    extra = [
        _rgb_hex(
            colorsys.hsv_to_rgb(
                (index + 0.15) / (count - len(base) + 0.3),
                0.7,
                0.9,
            )
        )
        for index in range(count - len(base))
    ]
    return [*base, *extra]


def comparison_axis_label(name: str) -> str:
    key = name.replace(" ", "").replace("-", "").lower()
    labels = {
        "2theta": "2θ",
        "twotheta": "2θ",
        "theta": "θ",
        "omega": "ω",
        "chi": "χ",
        "phi": "φ",
        "xdrive": "X",
        "ydrive": "Y",
        "zdrive": "Z",
        "scanaxis": "X",
        "x": "X",
    }
    return labels.get(key, name)


def comparison_axis_uses_degrees(name: str) -> bool:
    return name.replace(" ", "").replace("-", "").lower() in {
        "2theta",
        "twotheta",
        "theta",
        "omega",
        "chi",
        "phi",
    }


def transform_comparison_y(values: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    if mode == "log":
        positive = y[np.isfinite(y) & (y > 0)]
        epsilon = (
            max(1e-12, float(np.min(positive)) * 1e-6)
            if positive.size
            else 1e-12
        )
        return np.clip(y, epsilon, None)
    if mode == "exp":
        maximum = float(np.nanmax(y)) if y.size else 0.0
        return np.exp(y / (maximum or 1.0))
    if mode == "square":
        return np.square(np.maximum(y, 0))
    return y


def comparison_preset_for(
    name: str,
    presets: Mapping[str, Mapping[str, Sequence[float]]],
) -> Mapping[str, Sequence[float]] | None:
    if name in presets:
        return presets[name]
    for key in sorted(presets, key=len, reverse=True):
        if not name.startswith(key):
            continue
        suffix = name[len(key) :]
        if suffix == "_copy" or (suffix.startswith("_") and suffix[1:].isdigit()):
            return presets[key]
    return None


@dataclass(frozen=True)
class ComparisonPlotSeries:
    item: ComparisonItem
    x: np.ndarray
    y: np.ndarray
    label: str
    colour: str
    linewidth: float
    zorder: int


@dataclass(frozen=True)
class ComparisonPlotData:
    series: tuple[ComparisonPlotSeries, ...]
    x_label: str | None
    y_mode: str
    y_scale: str
    x_limits: Sequence[float] | None
    y_limits: Sequence[float] | None

    @property
    def uses_automatic_limits(self) -> bool:
        return self.x_limits is None or self.y_limits is None


def prepare_comparison_plot(
    assembly: ComparisonAssembly,
    presets: Mapping[str, Mapping[str, Sequence[float]]],
    view: Mapping[str, object] | None = None,
    *,
    thumbnail: bool = False,
) -> ComparisonPlotData:
    """Return render-ready series and axes metadata for an assembly."""

    settings = view or assembly.view
    y_mode = str(settings.get("ymode", "log"))
    files = [item for item in assembly.items if item.kind == "file"]
    substrates = [item for item in assembly.items if item.kind == "substrate"]
    file_base = ("#e31a1c",) if len(files) == 1 else FILE_COLOURS
    colours = {
        **dict(zip(files, comparison_palette(len(files), file_base))),
        **dict(
            zip(
                substrates,
                comparison_palette(len(substrates), SUBSTRATE_COLOURS),
            )
        ),
    }
    series = tuple(
        ComparisonPlotSeries(
            item=item,
            x=np.asarray(item.scan.x, dtype=float),
            y=transform_comparison_y(item.scan.y, y_mode),
            label=item.name,
            colour=colours[item],
            linewidth=1.0 if thumbnail else 1.3,
            zorder=2 if item.kind == "substrate" else 3,
        )
        for item in sorted(
            assembly.items,
            key=lambda value: 0 if value.kind == "substrate" else 1,
        )
    )

    x_limits = settings.get("xlim")
    y_limits = settings.get("ylim")
    if not (x_limits and y_limits):
        preset = next(
            (
                candidate
                for item in substrates
                if (candidate := comparison_preset_for(item.name, presets))
            ),
            None,
        )
        if preset is not None:
            x_limits = preset["xlim"]
            y_limits = preset["ylim"]
        else:
            x_limits = None
            y_limits = None

    axes = {item.scan.axis_name for item in assembly.items}
    x_label: str | None = None
    if len(axes) == 1:
        axis_name = next(iter(axes))
        x_label = comparison_axis_label(axis_name)
        if comparison_axis_uses_degrees(axis_name):
            x_label = f"{x_label}, °"

    return ComparisonPlotData(
        series=series,
        x_label=x_label,
        y_mode=y_mode,
        y_scale="log" if y_mode == "log" else "linear",
        x_limits=x_limits,
        y_limits=y_limits,
    )
