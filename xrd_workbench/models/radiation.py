"""Radiation profiles without GUI or localisation dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


RadiationTuple = tuple[str, float, float]


def validate_radiation_lines(lines: Iterable[RadiationTuple]) -> list[RadiationTuple]:
    """Validate and normalise one to five wavelength/weight tuples."""

    result: list[RadiationTuple] = []
    for index, item in enumerate(lines, start=1):
        try:
            name, wavelength, weight = item
            wavelength = float(wavelength)
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("Radiation lines must contain a name, wavelength and weight") from exc
        if not math.isfinite(wavelength) or not math.isfinite(weight):
            raise ValueError("Radiation wavelength and weight must be finite")
        if wavelength <= 0 or weight <= 0:
            raise ValueError("Radiation wavelength and weight must be positive")
        result.append((str(name).strip() or f"Custom {index}", wavelength, weight))
    if not 1 <= len(result) <= 5:
        raise ValueError("Radiation must contain between one and five lines")
    return result


@dataclass(frozen=True)
class RadiationPreset:
    key: str
    lines: tuple[RadiationTuple, ...]

    @property
    def label(self) -> str:
        return " / ".join(
            f"{name} ({wavelength:.5f} Å)"
            for name, wavelength, _weight in self.lines
        )


PRESETS = (
    RadiationPreset("cu_ka1", (("Cu Kα1", 1.54056, 1.0),)),
    RadiationPreset(
        "cu_ka12",
        (("Cu Kα1", 1.54056, 1.0), ("Cu Kα2", 1.54443, 0.5)),
    ),
    RadiationPreset(
        "cu_ka12_kb",
        (
            ("Cu Kα1", 1.54056, 1.0),
            ("Cu Kα2", 1.54443, 0.5),
            ("Cu Kβ", 1.39222, 0.15),
        ),
    ),
    RadiationPreset("co_ka1", (("Co Kα1", 1.78897, 1.0),)),
    RadiationPreset(
        "co_ka12",
        (("Co Kα1", 1.78897, 1.0), ("Co Kα2", 1.79285, 0.5)),
    ),
)

_PRESETS_BY_KEY = {preset.key: preset for preset in PRESETS}


@dataclass
class RadiationSettings:
    """One source of truth shared by every diffraction calculation."""

    profile_key: str = "cu_ka12"
    custom_lines: list[RadiationTuple] = field(
        default_factory=lambda: [("Custom 1", 1.54056, 1.0)]
    )

    def lines(self) -> list[RadiationTuple]:
        if self.profile_key == "other":
            return validate_radiation_lines(self.custom_lines)
        try:
            return list(_PRESETS_BY_KEY[self.profile_key].lines)
        except KeyError as exc:
            raise ValueError(f"Unknown radiation profile: {self.profile_key}") from exc

    def label(self, *, other_label: str = "Other…") -> str:
        if self.profile_key == "other":
            return other_label
        try:
            return _PRESETS_BY_KEY[self.profile_key].label
        except KeyError as exc:
            raise ValueError(f"Unknown radiation profile: {self.profile_key}") from exc

    def select_preset(self, key: str) -> None:
        if key not in _PRESETS_BY_KEY:
            raise ValueError(f"Unknown radiation profile: {key}")
        self.profile_key = key

    def select_custom(self, lines: Iterable[RadiationTuple]) -> None:
        self.custom_lines = validate_radiation_lines(lines)
        self.profile_key = "other"
