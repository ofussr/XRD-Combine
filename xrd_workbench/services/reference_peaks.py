"""Read and write the editable reference-peak database."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping


def reference_peak_path() -> Path:
    """Use the same editable database in both interfaces and frozen builds."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "pivo.json"
    return Path(__file__).resolve().parents[1] / "pivo.json"


class ReferencePeakError(ValueError):
    def __init__(self, code: str, *, row: int, name: str = ""):
        self.code, self.row, self.name = code, row, name
        super().__init__(f"{code}: row {row}, {name!r}")


def validate_reference_peaks(rows) -> dict[str, float]:
    """Validate the complete edit before changing the stored database."""
    result = {}
    for index, (name, value) in enumerate(rows, start=1):
        name = str(name).strip()
        try:
            number = float(str(value).strip().replace(",", "."))
        except (TypeError, ValueError):
            number = math.nan
        if not name or not math.isfinite(number):
            raise ReferencePeakError("invalid", row=index, name=name)
        if name in result:
            raise ReferencePeakError("duplicate", row=index, name=name)
        result[name] = number
    return result


def read_reference_peaks(path: str | Path) -> dict[str, float]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            return {}
        return validate_reference_peaks(data.items())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def write_reference_peaks(
    path: str | Path,
    data: Mapping[str, float],
) -> None:
    normalized = validate_reference_peaks(data.items())
    destination = Path(path)
    # Keep the former file intact if writing or replacing fails.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=destination.parent, prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(normalized, stream, ensure_ascii=False, indent=4, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["reference_peak_path", "read_reference_peaks", "write_reference_peaks",
           "validate_reference_peaks", "ReferencePeakError"]
