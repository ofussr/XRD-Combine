"""Toolkit-independent reader for two-column text measurements."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Type

import numpy as np

from ..models.scan import Scan1D


def _numeric_pair(line: str) -> tuple[float, float] | None:
    line = line.strip()
    if ";" in line:
        separators = (";",)
    else:
        # Whitespace XY files may mix spaces and tabs and use decimal commas.
        separators = ("\t", None, ",") if "\t" in line else (None, ",")
    for separator in separators:
        parts = line.split() if separator is None else next(csv.reader([line], delimiter=separator))
        if len(parts) < 2:
            continue
        try:
            return (float(parts[0].strip().replace(",", ".")),
                    float(parts[1].strip().replace(",", ".")))
        except ValueError:
            continue
    return None


def read_xy(
    path: str | Path,
    *,
    scan_factory: Type[Scan1D] = Scan1D,
) -> Scan1D:
    """Read whitespace, tab, comma or semicolon separated numeric columns.

    Decimal commas are accepted in whitespace/semicolon columns and quoted
    CSV fields. A comma between CSV fields is never changed into a decimal dot.
    """

    source = Path(path)
    x_values: list[float] = []
    y_values: list[float] = []
    with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                pair = _numeric_pair(line)
            except csv.Error:
                continue
            if pair is None:
                continue
            x_value, y_value = pair
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
