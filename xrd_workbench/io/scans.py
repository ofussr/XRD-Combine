"""Format dispatch for GUI-independent one-dimensional readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Type

from ..models.data_errors import XRDDataError
from ..models.scan import Scan1D
from .bruker import read_raw_scans
from .text import read_xy
from .xrdml import read_xrdml


def read_scan_file(
    path: str | Path,
    *,
    scan_factory: Type[Scan1D] = Scan1D,
    raw_reader: Callable[[Path], Any] | None = None,
) -> list[Scan1D]:
    """Detect the format by extension and return one-dimensional scans."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xrdml", ".xml"}:
        return read_xrdml(source, scan_factory=scan_factory)
    if suffix == ".raw":
        return read_raw_scans(
            source,
            scan_factory=scan_factory,
            raw_reader=raw_reader,
        )
    if suffix in {".xy", ".txt", ".dat", ".csv"}:
        return [read_xy(source, scan_factory=scan_factory)]
    raise XRDDataError("unsupported_format", suffix=source.suffix or "no extension")
