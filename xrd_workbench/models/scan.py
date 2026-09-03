"""GUI-independent model and operations for one-dimensional measurements."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data_errors import XRDDataError


@dataclass
class Scan1D:
    """One measured curve with one or more interchangeable coordinate axes."""

    name: str
    x: np.ndarray
    y: np.ndarray
    source: Path
    kind: str = "измерение"
    axis_name: str = "2Theta"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial_x = np.asarray(self.x, dtype=float)
        initial_y = np.asarray(self.y, dtype=float)
        if (
            initial_x.ndim != 1
            or initial_y.ndim != 1
            or initial_x.size != initial_y.size
        ):
            raise XRDDataError("scan_arrays")

        supplied_axes = self.metadata.get("axes", {self.axis_name: initial_x})
        axes: dict[str, np.ndarray] = {}
        for name, values in supplied_axes.items():
            array = np.asarray(values, dtype=float)
            if array.ndim == 1 and array.size == initial_y.size:
                axes[str(name)] = array
        if self.axis_name not in axes:
            axes[self.axis_name] = initial_x

        valid = np.isfinite(initial_x) & np.isfinite(initial_y)
        if np.count_nonzero(valid) < 2:
            raise XRDDataError("scan_too_short")
        self.metadata["axes"] = {
            name: values[valid].copy() for name, values in axes.items()
        }
        self.metadata["_base_y"] = initial_y[valid].copy()
        self.use_axis(self.axis_name)

    @property
    def available_axes(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("axes", {}).keys())

    def use_axis(self, axis_name: str) -> None:
        axes = self.metadata.get("axes", {})
        if axis_name not in axes:
            raise XRDDataError("scan_axis_missing", axis=axis_name)
        values = np.asarray(axes[axis_name], dtype=float)
        base_y = np.asarray(self.metadata["_base_y"], dtype=float)
        valid = np.isfinite(values) & np.isfinite(base_y)
        if np.count_nonzero(valid) < 2:
            raise XRDDataError("scan_axis_too_short", axis=axis_name)
        order = np.argsort(values[valid], kind="stable")
        self.x = values[valid][order]
        self.y = base_y[valid][order]
        self.axis_name = axis_name


def clone_scan(scan: Scan1D, *, name: str | None = None) -> Scan1D:
    """Create an independent copy while preserving every coordinate axis."""

    metadata = copy.deepcopy(scan.metadata)
    base_y = np.asarray(metadata.pop("_base_y", scan.y), dtype=float).copy()
    axes = metadata.get("axes", {})
    active_x = np.asarray(axes.get(scan.axis_name, scan.x), dtype=float).copy()
    scan_type = type(scan)
    return scan_type(
        name=name or scan.name,
        x=active_x,
        y=base_y,
        source=Path(scan.source),
        kind=scan.kind,
        axis_name=scan.axis_name,
        metadata=metadata,
    )


def assign_text_axis(scan: Scan1D, axis_name: str, *, assumed: bool = False) -> None:
    """Assign a meaning to a text column without inventing extra coordinates."""

    if scan.metadata.get("format") != "XY":
        raise XRDDataError("scan_text_axis_only")
    old_axis = scan.metadata["axes"][scan.axis_name]
    scan.metadata["axes"] = {axis_name: old_axis.copy()}
    scan.metadata["axis_assumed"] = assumed
    scan.use_axis(axis_name)
