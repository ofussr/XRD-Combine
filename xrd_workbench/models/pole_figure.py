"""Data objects produced by calculated pole-figure services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PolePoint:
    hkl: tuple[int, int, int]
    d_spacing: float
    two_theta: float
    direction: np.ndarray
    chi: float
    phi: float
    x: float
    y: float


@dataclass(frozen=True)
class PoleReflection:
    hkl: tuple[int, int, int]
    d_spacing: float
    two_theta: float


# Historical name used by the Tkinter module.
Reflection = PoleReflection


@dataclass(eq=False)
class CalculatedPoleLayer:
    """Toolkit-independent state of one phase in a calculated pole figure."""

    document: Any
    colour: str
    opacity_percent: float = 100.0
    size_percent: float = 100.0
    center_hkl: tuple[int, int, int] = (0, 1, 0)
    base_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    user_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    reflections: list[PoleReflection] = field(default_factory=list)
    points: list[PolePoint] = field(default_factory=list)
    point_groups: list[list[PolePoint]] = field(default_factory=list)
    selected_hkl: tuple[int, int, int] | None = None
    intensity_by_spacing: dict[float, float] | None = None
    coupled_to_primary: bool = False

    @property
    def crystal(self):
        return self.document.crystal

    @property
    def name(self) -> str:
        return self.document.name

    @property
    def orientation(self) -> np.ndarray:
        return self.user_rotation @ self.base_rotation

    @property
    def opacity(self) -> float:
        return min(100.0, max(0.0, float(self.opacity_percent))) / 100.0

    @property
    def size_scale(self) -> float:
        return min(300.0, max(10.0, float(self.size_percent))) / 100.0


__all__ = [
    "CalculatedPoleLayer",
    "PolePoint",
    "PoleReflection",
    "Reflection",
]
