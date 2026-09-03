"""Read and write the editable reference-peak database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def read_reference_peaks(path: str | Path) -> dict[str, float]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return {str(name): float(value) for name, value in data.items()}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def write_reference_peaks(
    path: str | Path,
    data: Mapping[str, float],
) -> None:
    normalized = {str(name): float(value) for name, value in data.items()}
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(normalized, stream, ensure_ascii=False, indent=4)


__all__ = ["read_reference_peaks", "write_reference_peaks"]
