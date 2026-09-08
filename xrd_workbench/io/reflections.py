"""File access for diffraction reference data and reflection tables."""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

from ..models.data_errors import XRDDataError
from ..models.diffraction import ReflectionRow, ScatteringFactors


def scattering_factor_path() -> Path:
    """Locate the bundled Waasmaier-Kirfel coefficient table."""

    package = Path(__file__).resolve().parents[1]
    candidates = [
        package / "resources" / "f0_WaasKirf.dat",
        package / "f0_WaasKirf.dat",
        Path(getattr(sys, "_MEIPASS", package)) / "f0_WaasKirf.dat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise XRDDataError("diffraction_factors_file_missing")


def read_scattering_factors(
    path: str | os.PathLike[str],
) -> ScatteringFactors:
    lines = Path(path).read_text(
        encoding="ascii", errors="replace"
    ).splitlines()
    result: ScatteringFactors = {}
    current: str | None = None
    for line in lines:
        if line.startswith("#S"):
            parts = line.split()
            current = parts[2] if len(parts) >= 3 else None
            continue
        if current and line.strip() and not line.startswith("#"):
            values = [float(value) for value in line.split()]
            if len(values) == 11 and re.fullmatch(r"[A-Z][a-z]?", current):
                result[current] = (values[:5], values[5], values[6:])
            current = None
    if not result:
        raise XRDDataError("diffraction_factors_empty")
    return result


def write_reflection_csv(
    path: str | os.PathLike[str],
    rows: Iterable[ReflectionRow],
    headers: Sequence[str],
) -> None:
    """Write reflection rows with caller-provided, already localized headers."""

    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(
                [
                    row.hkl,
                    f"{row.d:.6f}",
                    f"{row.two_theta:.5f}",
                    row.radiation,
                    f"{row.wavelength:.5f}",
                    f"{row.weight:.4g}",
                    row.multiplicity,
                    "—" if row.f2_sum is None else f"{row.f2_sum:.6g}",
                    "—" if row.intensity is None else f"{row.intensity:.4f}",
                ]
            )


__all__ = [
    "read_scattering_factors",
    "scattering_factor_path",
    "write_reflection_csv",
]
