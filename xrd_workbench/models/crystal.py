"""GUI-independent crystallographic structure and unit-cell geometry."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

from .data_errors import XRDDataError


FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?(?:\(\d+\))?$"
)


@dataclass
class CifLoop:
    tags: list[str]
    rows: list[list[str]]


@dataclass
class CifData:
    source: Path
    values: dict[str, str]
    loops: list[CifLoop]

    def get(self, *names: str, default: str | None = None) -> str | None:
        for name in names:
            value = self.values.get(name.lower())
            if value is not None:
                return value
        return default

    def loop_column(self, *names: str) -> list[str]:
        wanted = {name.lower() for name in names}
        for loop in self.loops:
            lower = [tag.lower() for tag in loop.tags]
            for column, tag in enumerate(lower):
                if tag in wanted:
                    return [row[column] for row in loop.rows]
        return []


def cif_number(value: str) -> float:
    """Parse a CIF number while discarding its parenthesized uncertainty."""

    value = value.strip()
    if value in {".", "?"}:
        raise XRDDataError("crystal_cif_number_undefined")
    value = re.sub(r"\(\d+\)$", "", value)
    if not FLOAT_RE.match(value):
        raise XRDDataError("crystal_cif_number_invalid", value=value)
    return float(value)


Affine = tuple[np.ndarray, Fraction]


def _affine_from_ast(node: ast.AST) -> Affine:
    if isinstance(node, ast.Expression):
        return _affine_from_ast(node.body)
    if isinstance(node, ast.Name) and node.id in {"x", "y", "z"}:
        coeff = np.array([Fraction(0), Fraction(0), Fraction(0)], dtype=object)
        coeff[{"x": 0, "y": 1, "z": 2}[node.id]] = Fraction(1)
        return coeff, Fraction(0)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return (
            np.array([Fraction(0), Fraction(0), Fraction(0)], dtype=object),
            Fraction(str(node.value)),
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        coeff, const = _affine_from_ast(node.operand)
        return (-coeff, -const) if isinstance(node.op, ast.USub) else (coeff, const)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left_coeff, left_const = _affine_from_ast(node.left)
        right_coeff, right_const = _affine_from_ast(node.right)
        sign = -1 if isinstance(node.op, ast.Sub) else 1
        return (
            left_coeff + sign * right_coeff,
            left_const + sign * right_const,
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_coeff, left_const = _affine_from_ast(node.left)
        right_coeff, right_const = _affine_from_ast(node.right)
        left_scalar = all(value == 0 for value in left_coeff)
        right_scalar = all(value == 0 for value in right_coeff)
        if left_scalar:
            return right_coeff * left_const, right_const * left_const
        if right_scalar:
            return left_coeff * right_const, left_const * right_const
        raise ValueError("Variables cannot be multiplied")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left_coeff, left_const = _affine_from_ast(node.left)
        right_coeff, right_const = _affine_from_ast(node.right)
        if any(value != 0 for value in right_coeff) or right_const == 0:
            raise ValueError("Invalid divisor")
        return left_coeff / right_const, left_const / right_const
    raise ValueError("Unsupported symmetry expression")


def parse_symmetry_operation(expression: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation of an affine CIF operation."""

    components = [item.strip() for item in expression.strip("'\"").split(",")]
    if len(components) != 3:
        raise XRDDataError("crystal_symmetry", expression=expression)
    matrix = np.zeros((3, 3), dtype=float)
    translation = np.zeros(3, dtype=float)
    for row, component in enumerate(components):
        try:
            tree = ast.parse(component.replace("^", "**"), mode="eval")
            coeff, const = _affine_from_ast(tree)
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            raise XRDDataError("crystal_symmetry", expression=expression) from exc
        matrix[row] = [float(value) for value in coeff]
        translation[row] = float(const % 1)
    return matrix, translation


def direct_basis(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    """Return direct-basis vectors as columns in a Cartesian frame."""

    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise XRDDataError("crystal_cell_degenerate_gamma")
    vector_a = np.array([a, 0.0, 0.0])
    vector_b = np.array([b * math.cos(gamma), b * sin_gamma, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= 0:
        raise XRDDataError("crystal_cell_degenerate")
    vector_c = np.array([cx, cy, math.sqrt(cz_squared)])
    return np.column_stack([vector_a, vector_b, vector_c])


@dataclass
class CrystalAtom:
    label: str
    element: str
    fractional: np.ndarray
    occupancy: float
    b_iso: float = 0.0


def _element_symbol(value: str) -> str:
    match = re.match(r"\s*([A-Z][a-z]?)", value)
    if not match:
        match = re.match(r"\s*([A-Za-z])", value)
    if not match:
        return "X"
    symbol = match.group(1)
    return symbol[0].upper() + symbol[1:].lower()


def expanded_atoms(
    cif: CifData,
    symmetry: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[CrystalAtom]:
    """Expand the asymmetric unit with explicit CIF symmetry operations."""

    asymmetric: list[CrystalAtom] = []
    for loop in cif.loops:
        tags = [tag.lower() for tag in loop.tags]
        required = (
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        )
        if not all(name in tags for name in required):
            continue
        x_column, y_column, z_column = (tags.index(name) for name in required)
        label_column = tags.index("_atom_site_label") if "_atom_site_label" in tags else None
        element_column = (
            tags.index("_atom_site_type_symbol")
            if "_atom_site_type_symbol" in tags
            else label_column
        )
        occupancy_column = (
            tags.index("_atom_site_occupancy")
            if "_atom_site_occupancy" in tags
            else None
        )
        b_iso_column = (
            tags.index("_atom_site_b_iso_or_equiv")
            if "_atom_site_b_iso_or_equiv" in tags
            else None
        )
        u_iso_column = (
            tags.index("_atom_site_u_iso_or_equiv")
            if "_atom_site_u_iso_or_equiv" in tags
            else None
        )
        for number, row in enumerate(loop.rows, start=1):
            label = row[label_column] if label_column is not None else f"A{number}"
            raw_element = row[element_column] if element_column is not None else label
            occupancy = (
                cif_number(row[occupancy_column])
                if occupancy_column is not None and row[occupancy_column] not in {".", "?"}
                else 1.0
            )
            b_iso = (
                cif_number(row[b_iso_column])
                if b_iso_column is not None and row[b_iso_column] not in {".", "?"}
                else 0.0
            )
            if (
                b_iso_column is None
                and u_iso_column is not None
                and row[u_iso_column] not in {".", "?"}
            ):
                b_iso = 8.0 * math.pi * math.pi * cif_number(row[u_iso_column])
            asymmetric.append(
                CrystalAtom(
                    label=label,
                    element=_element_symbol(raw_element),
                    fractional=np.array(
                        [
                            cif_number(row[x_column]),
                            cif_number(row[y_column]),
                            cif_number(row[z_column]),
                        ],
                        dtype=float,
                    ),
                    occupancy=occupancy,
                    b_iso=b_iso,
                )
            )
        break

    result: list[CrystalAtom] = []
    for atom in asymmetric:
        for matrix, translation in symmetry:
            fractional = matrix @ atom.fractional + translation
            fractional = fractional - np.floor(fractional)
            fractional[np.isclose(fractional, 1.0, atol=1e-9)] = 0.0
            duplicate = False
            for existing in result:
                if existing.element != atom.element:
                    continue
                difference = fractional - existing.fractional
                difference -= np.rint(difference)
                if np.linalg.norm(difference) < 1e-7:
                    duplicate = True
                    break
            if not duplicate:
                result.append(
                    CrystalAtom(
                        label=atom.label,
                        element=atom.element,
                        fractional=fractional,
                        occupancy=atom.occupancy,
                        b_iso=atom.b_iso,
                    )
                )
    return result


@dataclass
class CrystalStructure:
    cif: CifData
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    direct: np.ndarray
    reciprocal: np.ndarray
    symmetry: list[tuple[np.ndarray, np.ndarray]]
    atoms: list[CrystalAtom]
    space_group: str
    formula: str

    @classmethod
    def from_cif(cls, cif: CifData) -> "CrystalStructure":
        required = [
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        ]
        missing = [name for name in required if cif.get(name) is None]
        if missing:
            raise XRDDataError("crystal_cell_missing", fields=tuple(missing))
        a, b, c = (cif_number(cif.get(name) or "") for name in required[:3])
        alpha, beta, gamma = (
            cif_number(cif.get(name) or "") for name in required[3:]
        )
        direct = direct_basis(a, b, c, alpha, beta, gamma)
        reciprocal = np.linalg.inv(direct).T
        space_group = (
            cif.get(
                "_space_group_name_h-m_alt",
                "_symmetry_space_group_name_h-m",
                default="not specified",
            )
            or "not specified"
        )
        expressions = cif.loop_column(
            "_space_group_symop_operation_xyz",
            "_symmetry_equiv_pos_as_xyz",
        )
        if not expressions:
            group_key = space_group.strip("'\"").replace(" ", "").lower()
            number = cif.get(
                "_space_group_it_number",
                "_symmetry_int_tables_number",
                default="",
            )
            if group_key not in {"", "p1", "1", "notspecified"} or number not in {
                None,
                "",
                "1",
                "1.0",
                ".",
                "?",
            }:
                raise XRDDataError(
                    "crystal_non_p1_without_symmetry",
                    space_group=space_group,
                )
            expressions = ["x,y,z"]
        symmetry = [parse_symmetry_operation(item) for item in expressions]
        atoms = expanded_atoms(cif, symmetry)
        formula = cif.get("_chemical_formula_sum", default="not specified") or "not specified"
        return cls(
            cif,
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            direct,
            reciprocal,
            symmetry,
            atoms,
            space_group,
            formula,
        )

    def reciprocal_vector(self, hkl: Sequence[int]) -> np.ndarray:
        return self.reciprocal @ np.asarray(hkl, dtype=float)

    def d_spacing(self, hkl: Sequence[int]) -> float:
        length = float(np.linalg.norm(self.reciprocal_vector(hkl)))
        if length < 1e-14:
            raise XRDDataError("crystal_d_000")
        return 1.0 / length

    def two_theta(self, hkl: Sequence[int], wavelength: float) -> float | None:
        argument = wavelength / (2.0 * self.d_spacing(hkl))
        if argument > 1.0 + 1e-12:
            return None
        return math.degrees(2.0 * math.asin(min(1.0, argument)))

    def equivalent_reflections(self, hkl: Sequence[int]) -> list[tuple[int, int, int]]:
        source = np.asarray(hkl, dtype=float)
        result: set[tuple[int, int, int]] = set()
        for matrix, _translation in self.symmetry:
            transformed = np.linalg.solve(matrix.T, source)
            rounded = np.rint(transformed).astype(int)
            if not np.allclose(transformed, rounded, atol=1e-7):
                continue
            item = tuple(int(value) for value in rounded)
            result.add(item)
            result.add(tuple(-value for value in item))
        if not result:
            item = tuple(int(value) for value in hkl)
            result = {item, tuple(-value for value in item)}
        return sorted(result)

    def is_systematically_absent(self, hkl: Sequence[int]) -> bool:
        h = np.asarray(hkl, dtype=float)
        coefficients: dict[tuple[int, int, int], complex] = {}
        for matrix, translation in self.symmetry:
            q = matrix.T @ h
            q_int = tuple(int(value) for value in np.rint(q))
            phase = np.exp(2j * np.pi * float(np.dot(h, translation)))
            coefficients[q_int] = coefficients.get(q_int, 0j) + phase
        return bool(coefficients) and all(
            abs(value) < 1e-7 for value in coefficients.values()
        )


@dataclass(frozen=True)
class DisplayAtom:
    element: str
    fractional: np.ndarray
    cartesian: np.ndarray
    occupancy: float


def unit_cell_display_atoms(crystal: CrystalStructure) -> list[DisplayAtom]:
    """Include opposite-boundary copies of atoms on a unit-cell face."""

    result: list[DisplayAtom] = []
    for atom in crystal.atoms:
        choices: list[list[float]] = []
        for coordinate in atom.fractional:
            choices.append([0.0, 1.0] if abs(coordinate) < 1e-8 else [float(coordinate)])
        for x in choices[0]:
            for y in choices[1]:
                for z in choices[2]:
                    fractional = np.array([x, y, z], dtype=float)
                    result.append(
                        DisplayAtom(
                            atom.element,
                            fractional,
                            crystal.direct @ fractional,
                            atom.occupancy,
                        )
                    )
    return result


# Historical public names retained for compatibility.
Atom = CrystalAtom
Crystal = CrystalStructure


__all__ = [
    "Atom",
    "CifData",
    "CifLoop",
    "Crystal",
    "CrystalAtom",
    "CrystalStructure",
    "DisplayAtom",
    "cif_number",
    "direct_basis",
    "expanded_atoms",
    "parse_symmetry_operation",
    "unit_cell_display_atoms",
]
