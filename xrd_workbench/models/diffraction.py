"""GUI-independent data objects used by powder-diffraction calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RadiationLine = tuple[str, float, float]
ScatteringFactor = tuple[list[float], float, list[float]]
ScatteringFactors = dict[str, ScatteringFactor]


@dataclass(frozen=True)
class DiffractionAtom:
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float = 0.0


@dataclass
class DiffractionStructure:
    name: str
    cell: tuple[float, float, float, float, float, float]
    atoms: list[DiffractionAtom]
    symmetry_operations: list[str]
    cell_only: bool = False
    space_group: str = ""


@dataclass(frozen=True)
class ReflectionRow:
    hkl: str
    d: float
    two_theta: float
    radiation: str
    wavelength: float
    weight: float
    multiplicity: int
    f2_sum: float | None
    intensity: float | None
    equivalents: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True)
class PowderProfile:
    """A common grid, normalized total and per-radiation components."""

    x: np.ndarray
    total: np.ndarray
    components: dict[str, np.ndarray]


# Historical public names retained for source compatibility.
Atom = DiffractionAtom
Structure = DiffractionStructure


__all__ = [
    "Atom",
    "DiffractionAtom",
    "DiffractionStructure",
    "PowderProfile",
    "RadiationLine",
    "ReflectionRow",
    "ScatteringFactor",
    "ScatteringFactors",
    "Structure",
]
