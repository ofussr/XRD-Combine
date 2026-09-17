"""Locate application-owned files in source and frozen distributions."""

from __future__ import annotations

from pathlib import Path
import sys


def resource_path(filename: str) -> Path | None:
    """Return the first existing copy of an application resource."""

    package = Path(__file__).resolve().parent
    candidates = (
        package / "resources" / filename,
        Path(getattr(sys, "_MEIPASS", package)) / filename,
        Path(sys.executable).resolve().parent / filename,
        Path(sys.executable).resolve().parent / "xrd_workbench" / "resources" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


__all__ = ["resource_path"]
