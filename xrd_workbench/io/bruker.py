"""Convert toolkit-independent Bruker RAW data into one-dimensional scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Type

import numpy as np

from ..bruker_raw import read_bruker_raw
from ..models.data_errors import XRDDataError
from ..models.scan import Scan1D


def read_raw_scans(
    path: str | Path,
    *,
    raw: Any = None,
    scan_factory: Type[Scan1D] = Scan1D,
    raw_reader: Callable[[Path], Any] | None = None,
) -> list[Scan1D]:
    """Read every one-dimensional range from a Bruker RAW v3/v4 file."""

    source = Path(path)
    if raw is None:
        try:
            raw = (raw_reader or read_bruker_raw)(source)
        except (OSError, ValueError) as exc:
            raise XRDDataError("raw_unreadable", source_name=source.name) from exc
    ranges = [item for item in raw.ranges if item.point_count >= 2]
    if not ranges:
        raise XRDDataError("raw_no_ranges")

    result: list[Scan1D] = []
    for item in ranges:
        axes: dict[str, np.ndarray] = {item.axis_name: item.axis}
        for drive_name in (
            "2Theta",
            "Theta",
            "Omega",
            "Chi",
            "Phi",
            "X-Drive",
            "Y-Drive",
            "Z-Drive",
        ):
            coordinate = item.coordinate(drive_name)
            if coordinate.size == item.point_count and np.any(np.isfinite(coordinate)):
                axes.setdefault(drive_name, coordinate)
        label = source.stem
        if len(ranges) > 1:
            label = f"{label} – #{item.index + 1}"
        result.append(
            scan_factory(
                name=label,
                x=item.axis,
                y=item.intensity,
                source=source,
                axis_name=item.axis_name,
                metadata={
                    "format": f"Bruker RAW v{raw.version}",
                    "range_index": item.index,
                    "wavelength": item.wavelength,
                    "scan_type": item.scan_type,
                    "is_pole_figure": raw.is_pole_figure,
                    "axes": axes,
                },
            )
        )
    return result
