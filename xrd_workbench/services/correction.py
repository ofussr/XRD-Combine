"""Apply correction requests without depending on a graphical toolkit."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np

from ..models.correction import CorrectionRequest
from ..models.scan import Scan1D
from ..models.viewer import axis_key, is_two_theta


def corrected_name(name: str) -> str:
    return name if name.lower().endswith(" shifted") else f"{name} shifted"


def apply_correction(
    scan: Scan1D,
    request: CorrectionRequest,
    *,
    scan_factory: Callable[..., Scan1D] | None = None,
) -> Scan1D:
    """Return a corrected copy while leaving the source scan untouched."""

    metadata = copy.deepcopy(scan.metadata)
    base_y = np.asarray(metadata.pop("_base_y", scan.y), dtype=float).copy()
    axes = {
        str(name): np.asarray(values, dtype=float).copy()
        for name, values in metadata.get("axes", {}).items()
    }
    active_x = np.asarray(axes.get(scan.axis_name, scan.x), dtype=float).copy()
    axes[scan.axis_name] = active_x + request.x_shift
    if request.shift_omega_half and is_two_theta(scan.axis_name):
        for axis_name, values in tuple(axes.items()):
            if axis_key(axis_name) == "omega":
                axes[axis_name] = values + request.x_shift / 2.0

    metadata["axes"] = axes
    metadata["shift"] = float(metadata.get("shift", 0.0)) + request.x_shift
    metadata["y_shift"] = (
        float(metadata.get("y_shift", 0.0)) * request.y_factor
        + request.y_shift
    )
    metadata["y_factor"] = (
        float(metadata.get("y_factor", 1.0)) * request.y_factor
    )
    metadata["processing"] = {
        "x_shift": request.x_shift,
        "y_shift": request.y_shift,
        "y_factor": request.y_factor,
        "shift_omega_half": request.shift_omega_half,
    }

    factory = scan_factory or type(scan)
    return factory(
        name=corrected_name(scan.name),
        x=axes[scan.axis_name],
        y=base_y * request.y_factor + request.y_shift,
        source=Path(scan.source),
        kind=scan.kind,
        axis_name=scan.axis_name,
        metadata=metadata,
    )


__all__ = ["apply_correction", "corrected_name"]
