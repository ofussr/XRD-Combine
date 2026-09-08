"""Toolkit-independent powder-diffraction and profile calculations."""

from __future__ import annotations

import ast
import cmath
import math
from collections import defaultdict
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from ..models.data_errors import XRDDataError
from ..models.diffraction import (
    DiffractionAtom,
    DiffractionStructure,
    PowderProfile,
    RadiationLine,
    ReflectionRow,
    ScatteringFactors,
)


def d_spacing_from_two_theta(
    two_theta: float,
    wavelength: float,
) -> float | None:
    """Return first-order Bragg spacing for a physical 2theta coordinate.

    ``None`` is returned for non-finite values, non-positive wavelengths, or
    coordinates outside the physical 0 < 2theta <= 180 degree interval.
    """

    try:
        two_theta = float(two_theta)
        wavelength = float(wavelength)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(two_theta)
        or not math.isfinite(wavelength)
        or wavelength <= 0.0
        or two_theta <= 0.0
        or two_theta > 180.0
    ):
        return None
    sine = math.sin(math.radians(two_theta / 2.0))
    if sine <= 0.0:
        return None
    return wavelength / (2.0 * sine)


def _inverse_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(determinant) < 1e-15:
        raise XRDDataError("diffraction_singular_metric")
    return [
        [
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ],
        [
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ],
        [
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ],
    ]


def _reciprocal_metric(
    cell: tuple[float, float, float, float, float, float],
) -> list[list[float]]:
    a, b, c, alpha, beta, gamma = cell
    ca, cb, cg = (
        math.cos(math.radians(value)) for value in (alpha, beta, gamma)
    )
    direct = [
        [a * a, a * b * cg, a * c * cb],
        [a * b * cg, b * b, b * c * ca],
        [a * c * cb, b * c * ca, c * c],
    ]
    return _inverse_3x3(direct)


def _q2(h: int, k: int, l: int, reciprocal: list[list[float]]) -> float:
    vector = (h, k, l)
    return sum(
        vector[i] * reciprocal[i][j] * vector[j]
        for i in range(3)
        for j in range(3)
    )


def _atomic_factor(
    element: str,
    s: float,
    factors: ScatteringFactors,
) -> float:
    try:
        a, c, b = factors[element]
    except KeyError as exc:
        raise XRDDataError("diffraction_factor_missing", element=element) from exc
    return c + sum(ai * math.exp(-bi * s * s) for ai, bi in zip(a, b))


def _structure_factor_squared(
    h: int,
    k: int,
    l: int,
    d: float,
    atoms: Iterable[DiffractionAtom],
    factors: ScatteringFactors,
) -> float:
    s = 1.0 / (2.0 * d)
    total = 0j
    for atom in atoms:
        f0 = _atomic_factor(atom.element, s, factors)
        debye_waller = math.exp(-max(0.0, atom.b_iso) * s * s)
        phase = 2.0 * math.pi * (h * atom.x + k * atom.y + l * atom.z)
        total += atom.occupancy * f0 * debye_waller * cmath.exp(1j * phase)
    return total.real * total.real + total.imag * total.imag


def _lorentz_polarization(two_theta: float) -> float:
    theta = math.radians(two_theta / 2.0)
    return (1.0 + math.cos(2.0 * theta) ** 2) / (
        math.sin(theta) ** 2 * max(math.cos(theta), 1e-12)
    )


def format_hkl_family(labels: set[tuple[int, int, int]]) -> str:
    ordered = sorted(labels, key=lambda item: (sum(item), item))
    shown = [f"({h} {k} {l})" for h, k, l in ordered[:4]]
    if len(ordered) > 4:
        shown.append(f"… {len(ordered) - 4}")
    return "+".join(shown)


def _affine_from_ast(node: ast.AST) -> tuple[np.ndarray, Fraction]:
    if isinstance(node, ast.Expression):
        return _affine_from_ast(node.body)
    if isinstance(node, ast.Name) and node.id in {"x", "y", "z"}:
        coefficients = np.array(
            [Fraction(0), Fraction(0), Fraction(0)], dtype=object
        )
        coefficients[{"x": 0, "y": 1, "z": 2}[node.id]] = Fraction(1)
        return coefficients, Fraction(0)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return (
            np.array([Fraction(0), Fraction(0), Fraction(0)], dtype=object),
            Fraction(str(node.value)),
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        coefficients, constant = _affine_from_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -coefficients, -constant
        return coefficients, constant
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left_coefficients, left_constant = _affine_from_ast(node.left)
        right_coefficients, right_constant = _affine_from_ast(node.right)
        sign = -1 if isinstance(node.op, ast.Sub) else 1
        return (
            left_coefficients + sign * right_coefficients,
            left_constant + sign * right_constant,
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_coefficients, left_constant = _affine_from_ast(node.left)
        right_coefficients, right_constant = _affine_from_ast(node.right)
        left_scalar = all(value == 0 for value in left_coefficients)
        right_scalar = all(value == 0 for value in right_coefficients)
        if left_scalar:
            return (
                right_coefficients * left_constant,
                right_constant * left_constant,
            )
        if right_scalar:
            return (
                left_coefficients * right_constant,
                left_constant * right_constant,
            )
        raise ValueError
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left_coefficients, left_constant = _affine_from_ast(node.left)
        right_coefficients, right_constant = _affine_from_ast(node.right)
        if any(value != 0 for value in right_coefficients) or right_constant == 0:
            raise ValueError
        return (
            left_coefficients / right_constant,
            left_constant / right_constant,
        )
    raise ValueError


def _parse_symmetry_operation(expression: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [part.strip().lower() for part in expression.split(",")]
    if len(parts) != 3:
        raise XRDDataError("diffraction_symmetry", expression=expression)
    matrix = np.zeros((3, 3), dtype=float)
    translation = np.zeros(3, dtype=float)
    try:
        for row, part in enumerate(parts):
            coefficients, constant = _affine_from_ast(ast.parse(part, mode="eval"))
            matrix[row] = [float(value) for value in coefficients]
            translation[row] = float(constant % 1)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        raise XRDDataError(
            "diffraction_symmetry", expression=expression
        ) from exc
    return matrix, translation


def _cell_reflection_absent(
    hkl: tuple[int, int, int],
    symmetry: Sequence[tuple[np.ndarray, np.ndarray]],
) -> bool:
    h = np.asarray(hkl, dtype=float)
    coefficients: dict[tuple[int, int, int], complex] = {}
    for matrix, translation in symmetry:
        q = matrix.T @ h
        q_int = tuple(int(value) for value in np.rint(q))
        phase = cmath.exp(2j * math.pi * float(h @ translation))
        coefficients[q_int] = coefficients.get(q_int, 0j) + phase
    return bool(coefficients) and all(
        abs(value) < 1e-7 for value in coefficients.values()
    )


def _calculate_cell_only_reflections(
    structure: DiffractionStructure,
    radiations: Sequence[RadiationLine],
    min_two_theta: float,
    max_two_theta: float,
) -> list[ReflectionRow]:
    reciprocal = _reciprocal_metric(structure.cell)
    shortest_wavelength = min(wavelength for _, wavelength, _ in radiations)
    d_min = shortest_wavelength / (
        2.0 * math.sin(math.radians(max_two_theta / 2.0))
    )
    limits = [math.ceil(length / d_min) + 1 for length in structure.cell[:3]]
    symmetry = [
        _parse_symmetry_operation(item) for item in structure.symmetry_operations
    ]
    grouped: dict[float, dict[str, object]] = {}
    for h in range(-limits[0], limits[0] + 1):
        for k in range(-limits[1], limits[1] + 1):
            for l in range(-limits[2], limits[2] + 1):
                if h == k == l == 0 or _cell_reflection_absent(
                    (h, k, l), symmetry
                ):
                    continue
                q2 = _q2(h, k, l, reciprocal)
                if q2 <= 0:
                    continue
                d = 1.0 / math.sqrt(q2)
                if d < d_min * (1.0 - 1e-10):
                    continue
                key = round(q2, 8)
                group = grouped.setdefault(
                    key,
                    {"d": d, "multiplicity": 0, "labels": set()},
                )
                group["multiplicity"] = int(group["multiplicity"]) + 1
                labels = group["labels"]
                assert isinstance(labels, set)
                labels.add((abs(h), abs(k), abs(l)))

    rows: list[ReflectionRow] = []
    for group in grouped.values():
        d = float(group["d"])
        labels = group["labels"]
        assert isinstance(labels, set)
        for radiation, wavelength, weight in radiations:
            argument = wavelength / (2.0 * d)
            if argument > 1.0:
                continue
            two_theta = math.degrees(2.0 * math.asin(argument))
            if min_two_theta <= two_theta <= max_two_theta:
                rows.append(
                    ReflectionRow(
                        format_hkl_family(labels),
                        d,
                        two_theta,
                        radiation,
                        wavelength,
                        weight,
                        int(group["multiplicity"]),
                        None,
                        None,
                        tuple(
                            sorted(labels, key=lambda item: (sum(item), item))
                        ),
                    )
                )
    rows.sort(key=lambda row: (row.two_theta, row.radiation))
    return rows


def calculate_reflections(
    structure: DiffractionStructure,
    factors: ScatteringFactors,
    radiations: Sequence[RadiationLine],
    min_two_theta: float = 5.0,
    max_two_theta: float = 120.0,
    min_intensity: float = 0.1,
) -> list[ReflectionRow]:
    """Calculate powder reflections without importing a UI toolkit."""

    if not radiations:
        raise XRDDataError("diffraction_no_radiation")
    if not 0.0 <= min_two_theta < max_two_theta < 180.0:
        raise XRDDataError("diffraction_limits")
    if any(
        not math.isfinite(wavelength)
        or not math.isfinite(weight)
        or wavelength <= 0
        or weight <= 0
        for _, wavelength, weight in radiations
    ):
        raise XRDDataError("diffraction_radiation_positive")

    if structure.cell_only:
        return _calculate_cell_only_reflections(
            structure,
            radiations,
            min_two_theta,
            max_two_theta,
        )

    reciprocal = _reciprocal_metric(structure.cell)
    shortest_wavelength = min(wavelength for _, wavelength, _ in radiations)
    d_min = shortest_wavelength / (
        2.0 * math.sin(math.radians(max_two_theta / 2.0))
    )
    a, b, c = structure.cell[:3]
    limits = [math.ceil(length / d_min) + 1 for length in (a, b, c)]

    grouped: dict[float, dict[str, object]] = {}
    max_single_f2 = 0.0
    candidates: list[tuple[int, int, int, float, float]] = []
    for h in range(-limits[0], limits[0] + 1):
        for k in range(-limits[1], limits[1] + 1):
            for l in range(-limits[2], limits[2] + 1):
                if h == k == l == 0:
                    continue
                q2 = _q2(h, k, l, reciprocal)
                if q2 <= 0:
                    continue
                d = 1.0 / math.sqrt(q2)
                if d < d_min * (1.0 - 1e-10):
                    continue
                f2 = _structure_factor_squared(
                    h, k, l, d, structure.atoms, factors
                )
                max_single_f2 = max(max_single_f2, f2)
                candidates.append((h, k, l, d, f2))

    extinction_limit = max(max_single_f2 * 1e-12, 1e-10)
    for h, k, l, d, f2 in candidates:
        if f2 < extinction_limit:
            continue
        key = round(1.0 / (d * d), 8)
        group = grouped.setdefault(
            key,
            {"d": d, "f2_sum": 0.0, "multiplicity": 0, "labels": set()},
        )
        group["f2_sum"] = float(group["f2_sum"]) + f2
        group["multiplicity"] = int(group["multiplicity"]) + 1
        labels = group["labels"]
        assert isinstance(labels, set)
        labels.add((abs(h), abs(k), abs(l)))

    raw_rows: list[ReflectionRow] = []
    for group in grouped.values():
        d = float(group["d"])
        for radiation, wavelength, weight in radiations:
            argument = wavelength / (2.0 * d)
            if argument > 1.0:
                continue
            two_theta = math.degrees(2.0 * math.asin(argument))
            if not min_two_theta <= two_theta <= max_two_theta:
                continue
            f2_sum = float(group["f2_sum"])
            raw = weight * f2_sum * _lorentz_polarization(two_theta)
            labels = group["labels"]
            assert isinstance(labels, set)
            raw_rows.append(
                ReflectionRow(
                    format_hkl_family(labels),
                    d,
                    two_theta,
                    radiation,
                    wavelength,
                    weight,
                    int(group["multiplicity"]),
                    f2_sum,
                    raw,
                    tuple(sorted(labels, key=lambda item: (sum(item), item))),
                )
            )

    if not raw_rows:
        return []
    maximum = max(float(row.intensity) for row in raw_rows)
    rows = [
        ReflectionRow(
            row.hkl,
            row.d,
            row.two_theta,
            row.radiation,
            row.wavelength,
            row.weight,
            row.multiplicity,
            row.f2_sum,
            100.0 * float(row.intensity) / maximum,
            row.equivalents,
        )
        for row in raw_rows
        if 100.0 * float(row.intensity) / maximum >= min_intensity
    ]
    rows.sort(key=lambda row: (row.two_theta, row.radiation))
    return rows


def gaussian_powder_profile(
    rows: Iterable[ReflectionRow],
    minimum: float,
    maximum: float,
    fwhm: float,
    *,
    point_count: int = 5000,
    normalize_to: float = 100.0,
    missing_intensity: float = 0.0,
) -> PowderProfile:
    """Build a Gaussian profile and its radiation components on one grid."""

    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum >= maximum
    ):
        raise XRDDataError("diffraction_profile_limits")
    if not math.isfinite(fwhm) or fwhm <= 0:
        raise XRDDataError("diffraction_profile_fwhm")
    if point_count < 2:
        raise XRDDataError("diffraction_profile_points")
    if not math.isfinite(normalize_to) or normalize_to <= 0:
        raise XRDDataError("diffraction_profile_scale")

    grid = np.linspace(minimum, maximum, int(point_count))
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    grouped: dict[str, list[ReflectionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.radiation].append(row)

    components: dict[str, np.ndarray] = {}
    for radiation, radiation_rows in grouped.items():
        profile = np.zeros_like(grid)
        for row in radiation_rows:
            intensity = (
                missing_intensity if row.intensity is None else row.intensity
            )
            profile += float(intensity) * np.exp(
                -0.5 * ((grid - row.two_theta) / sigma) ** 2
            )
        components[radiation] = profile

    total = sum(components.values(), np.zeros_like(grid))
    maximum_profile = float(np.max(total)) if total.size else 0.0
    if maximum_profile > 0:
        scale = normalize_to / maximum_profile
        total *= scale
        for radiation in components:
            components[radiation] *= scale
    return PowderProfile(grid, total, components)


__all__ = [
    "calculate_reflections",
    "format_hkl_family",
    "gaussian_powder_profile",
]
