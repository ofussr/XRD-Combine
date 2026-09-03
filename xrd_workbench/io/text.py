"""Toolkit-independent reader for two-column text measurements."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Type

import numpy as np

from ..models.scan import Scan1D


def read_xy(
    path: str | Path,
    *,
    scan_factory: Type[Scan1D] = Scan1D,
) -> Scan1D:
    """Read the first two numeric columns of a text file."""

    source = Path(path)
    x_values: list[float] = []
    y_values: list[float] = []
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.replace(",", ".").strip().split()
            if len(parts) < 2:
                continue
            try:
                x_value, y_value = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if math.isfinite(x_value) and math.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)
    x_array = np.asarray(x_values)
    return scan_factory(
        name=source.stem,
        x=x_array,
        y=np.asarray(y_values),
        source=source,
        axis_name="X",
        metadata={"format": "XY", "axes": {"X": x_array}},
    )
