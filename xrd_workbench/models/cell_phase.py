"""GUI-independent model of an atom-free crystallographic phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CellPhaseDocument:
    """A phase defined by its cell and space group, without atomic sites."""

    name: str
    setting: Any
    cell: tuple[float, float, float, float, float, float]
    source: Path
    data: Any
    crystal: Any
    diffraction: Any
    is_cell_only: bool = True
