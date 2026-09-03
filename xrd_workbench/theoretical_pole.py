#!/usr/bin/env python3
"""
Интерактивная теоретическая полюсная фигура по структурному файлу CIF.

Возможности:
- выбор полюса (hkl), задающего центрирование;
- одновременное отображение всех разрешённых отражений в диапазоне d;
- переключаемые размер и окраска точек по d;
- трёхмерная атомная ячейка с тем же поворотом, что у полюсной фигуры;
- стереографическая или равноплощадная проекция;
- точный поворот вокруг осей X, Y, Z в градусах;
- свободное вращение перетаскиванием мышью;
- выбор полюса щелчком и вывод его hkl, χ, φ и координат;
- учёт метрики элементарной ячейки и операций симметрии из CIF.

Зависимости: numpy, matplotlib. Tkinter входит в обычную установку Python.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .cif_lexer import CifLexError, tokenize_cif_text
    from .i18n import (
        LocalizedStringVar,
        apply_language,
        choice_code,
        filedialog,
        localised,
        messagebox,
        translate_text,
    )
except ImportError:
    from cif_lexer import CifLexError, tokenize_cif_text
    from i18n import (
        LocalizedStringVar,
        apply_language,
        choice_code,
        filedialog,
        localised,
        messagebox,
        translate_text,
    )


FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?(?:\(\d+\))?$"
)
POLE_AZIMUTH_GRID_STEP_DEG = 10


def cif_number(value: str) -> float:
    """Преобразует число CIF, удаляя скобочную неопределённость."""
    value = value.strip()
    if value in {".", "?"}:
        raise ValueError(
            localised(
                "Undefined numeric CIF value.",
                "Valeur numérique CIF indéfinie.",
                "Неопределённое числовое значение CIF.",
            )
        )
    value = re.sub(r"\(\d+\)$", "", value)
    if not FLOAT_RE.match(value):
        raise ValueError(
            localised(
                f"Invalid CIF number: {value!r}",
                f"Nombre CIF incorrect : {value!r}",
                f"Некорректное число CIF: {value!r}",
            )
        )
    return float(value)


def tokenize_cif(text: str) -> list[str]:
    """Минимальный токенизатор CIF 1.1 с поддержкой многострочных полей."""
    try:
        return tokenize_cif_text(text)
    except CifLexError as exc:
        reason = localised(
            "unclosed quotation mark"
            if exc.reason == "unclosed_quote"
            else "unclosed multiline field",
            "guillemet non fermé"
            if exc.reason == "unclosed_quote"
            else "champ multiligne non fermé",
            "незакрытая кавычка"
            if exc.reason == "unclosed_quote"
            else "незакрытое многострочное поле",
        )
        raise ValueError(
            localised(
                f"Could not parse CIF line {exc.line_number}: {reason}.",
                f"Impossible d’analyser la ligne CIF {exc.line_number} : {reason}.",
                f"Не удалось разобрать строку CIF {exc.line_number}: {reason}.",
            )
        ) from exc


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


def parse_cif(path: str | os.PathLike[str]) -> CifData:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    tokens = tokenize_cif(text)
    values: dict[str, str] = {}
    loops: list[CifLoop] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        low = token.lower()
        if low == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(tokens[index])
                index += 1
            if not tags:
                raise ValueError(
                    localised(
                        "No column names follow loop_.",
                        "Aucun nom de colonne ne suit loop_.",
                        "После loop_ не найдены имена столбцов.",
                    )
                )

            raw: list[str] = []
            while index < len(tokens):
                next_low = tokens[index].lower()
                if (
                    tokens[index].startswith("_")
                    or next_low == "loop_"
                    or next_low == "stop_"
                    or next_low.startswith("data_")
                    or next_low.startswith("save_")
                ):
                    break
                raw.append(tokens[index])
                index += 1

            if len(raw) % len(tags) != 0:
                raise ValueError(
                    localised(
                        f"The CIF loop value count is not divisible by the column "
                        f"count ({len(raw)} and {len(tags)}).",
                        f"Le nombre de valeurs de la boucle CIF n’est pas divisible "
                        f"par le nombre de colonnes ({len(raw)} et {len(tags)}).",
                        f"Число значений в цикле CIF не кратно числу столбцов "
                        f"({len(raw)} и {len(tags)}).",
                    )
                )
            rows = [
                raw[start : start + len(tags)]
                for start in range(0, len(raw), len(tags))
            ]
            loops.append(CifLoop(tags, rows))
            if index < len(tokens) and tokens[index].lower() == "stop_":
                index += 1
            continue

        if token.startswith("_"):
            if index + 1 >= len(tokens):
                raise ValueError(
                    localised(
                        f"CIF field {token} has no value.",
                        f"Le champ CIF {token} n’a pas de valeur.",
                        f"Для поля {token} отсутствует значение.",
                    )
                )
            values[token.lower()] = tokens[index + 1]
            index += 2
            continue

        index += 1

    return CifData(source, values, loops)


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
        if isinstance(node.op, ast.USub):
            return -coeff, -const
        return coeff, const
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
        raise ValueError(
            localised(
                "Variables cannot be multiplied in a symmetry operation.",
                "Les variables ne peuvent pas être multipliées dans une opération de symétrie.",
                "Произведение переменных в операции симметрии.",
            )
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left_coeff, left_const = _affine_from_ast(node.left)
        right_coeff, right_const = _affine_from_ast(node.right)
        if any(value != 0 for value in right_coeff) or right_const == 0:
            raise ValueError(
                localised(
                    "Invalid division in a symmetry operation.",
                    "Division incorrecte dans une opération de symétrie.",
                    "Недопустимое деление в операции симметрии.",
                )
            )
        return left_coeff / right_const, left_const / right_const
    raise ValueError(
        localised(
            "Unsupported expression in a symmetry operation.",
            "Expression non prise en charge dans une opération de symétrie.",
            "Неподдерживаемое выражение в операции симметрии.",
        )
    )


def parse_symmetry_operation(expression: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [part.strip().lower() for part in expression.split(",")]
    if len(parts) != 3:
        raise ValueError(
            localised(
                f"A symmetry operation must have three coordinates: {expression}",
                f"Une opération de symétrie doit avoir trois coordonnées : {expression}",
                f"Операция симметрии должна иметь три координаты: {expression}",
            )
        )
    matrix = np.zeros((3, 3), dtype=float)
    translation = np.zeros(3, dtype=float)
    for row, part in enumerate(parts):
        try:
            tree = ast.parse(part, mode="eval")
            coeff, const = _affine_from_ast(tree)
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                localised(
                    f"Could not parse symmetry operation {expression!r}.",
                    f"Impossible d’analyser l’opération de symétrie {expression!r}.",
                    f"Не удалось разобрать операцию симметрии {expression!r}.",
                )
            ) from exc
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
    """Матрица базисных векторов прямой ячейки в декартовой системе."""
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise ValueError(
            localised(
                "Degenerate unit cell: sin(γ) = 0.",
                "Maille dégénérée : sin(γ) = 0.",
                "Вырожденная ячейка: sin(γ) = 0.",
            )
        )

    vector_a = np.array([a, 0.0, 0.0])
    vector_b = np.array([b * math.cos(gamma), b * sin_gamma, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= 0:
        raise ValueError(
            localised(
                "The CIF parameters define a degenerate unit cell.",
                "Les paramètres CIF définissent une maille dégénérée.",
                "Параметры CIF задают вырожденную ячейку.",
            )
        )
    vector_c = np.array([cx, cy, math.sqrt(cz_squared)])
    return np.column_stack([vector_a, vector_b, vector_c])


@dataclass
class Atom:
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
) -> list[Atom]:
    """Разворачивает асимметричную часть CIF по операциям симметрии."""
    asymmetric: list[Atom] = []
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
        label_column = (
            tags.index("_atom_site_label")
            if "_atom_site_label" in tags
            else None
        )
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
            raw_element = (
                row[element_column] if element_column is not None else label
            )
            occupancy = (
                cif_number(row[occupancy_column])
                if occupancy_column is not None
                and row[occupancy_column] not in {".", "?"}
                else 1.0
            )
            b_iso = (
                cif_number(row[b_iso_column])
                if b_iso_column is not None and row[b_iso_column] not in {".", "?"}
                else 0.0
            )
            if b_iso_column is None and u_iso_column is not None and row[u_iso_column] not in {".", "?"}:
                b_iso = 8.0 * math.pi * math.pi * cif_number(row[u_iso_column])
            asymmetric.append(
                Atom(
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

    result: list[Atom] = []
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
                    Atom(
                        label=atom.label,
                        element=atom.element,
                        fractional=fractional,
                        occupancy=atom.occupancy,
                        b_iso=atom.b_iso,
                    )
                )
    return result


@dataclass
class Crystal:
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
    atoms: list[Atom]
    space_group: str
    formula: str

    @classmethod
    def from_cif(cls, cif: CifData) -> "Crystal":
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
            joined = ", ".join(missing)
            raise ValueError(
                localised(
                    f"Unit-cell parameters are missing from the CIF: {joined}",
                    f"Des paramètres de maille sont absents du CIF : {joined}",
                    f"В CIF отсутствуют параметры ячейки: {joined}",
                )
            )

        a, b, c = (cif_number(cif.get(name) or "") for name in required[:3])
        alpha, beta, gamma = (
            cif_number(cif.get(name) or "") for name in required[3:]
        )
        direct = direct_basis(a, b, c, alpha, beta, gamma)
        reciprocal = np.linalg.inv(direct).T

        unknown = localised("not specified", "non indiqué", "не указана")
        space_group = (
            cif.get(
                "_space_group_name_h-m_alt",
                "_symmetry_space_group_name_h-m",
                default=unknown,
            )
            or unknown
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
            declared_non_p1 = (
                group_key
                not in {"", "p1", "1", "notspecified", "nonindiqué", "неуказана"}
                or (number not in {None, "", "1", "1.0", ".", "?"})
            )
            if declared_non_p1:
                raise ValueError(
                    localised(
                        f"The CIF declares the non-P1 space group {space_group!r} "
                        "but contains no explicit symmetry operations. Calculating "
                        "it as P1 would be incorrect.",
                        f"Le CIF déclare le groupe d’espace non P1 {space_group!r}, "
                        "mais ne contient aucune opération de symétrie explicite. "
                        "Un calcul en P1 serait incorrect.",
                        f"В CIF указана непервичная пространственная группа "
                        f"{space_group!r}, но нет явных операций симметрии. "
                        "Расчёт как P1 был бы неверным.",
                    )
                )
            expressions = ["x,y,z"]
        symmetry = [parse_symmetry_operation(item) for item in expressions]
        atoms = expanded_atoms(cif, symmetry)
        formula = cif.get("_chemical_formula_sum", default=unknown) or unknown
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
            raise ValueError(
                localised(
                    "The interplanar spacing is undefined for (0 0 0).",
                    "La distance interréticulaire n’est pas définie pour (0 0 0).",
                    "Для (0 0 0) межплоскостное расстояние не определено.",
                )
            )
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
            result.add(tuple(-value for value in item))  # пара Фриделя
        if not result:
            item = tuple(int(value) for value in hkl)
            result = {item, tuple(-value for value in item)}
        return sorted(result)

    def is_systematically_absent(self, hkl: Sequence[int]) -> bool:
        """Проверка погасания по операциям общей позиции из CIF."""
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


def rotation_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_array))
    if norm < 1e-14 or abs(angle_rad) < 1e-14:
        return np.eye(3)
    x, y, z = axis_array / norm
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ]
    )


def rotation_x(angle_deg: float) -> np.ndarray:
    return rotation_axis_angle((1.0, 0.0, 0.0), math.radians(angle_deg))


def rotation_y(angle_deg: float) -> np.ndarray:
    return rotation_axis_angle((0.0, 1.0, 0.0), math.radians(angle_deg))


def rotation_z(angle_deg: float) -> np.ndarray:
    return rotation_axis_angle((0.0, 0.0, 1.0), math.radians(angle_deg))


def euler_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Повороты вокруг неподвижных экранных осей: Rz · Ry · Rx."""
    return rotation_z(z_deg) @ rotation_y(y_deg) @ rotation_x(x_deg)


def matrix_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    value = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    y = math.asin(value)
    if abs(math.cos(y)) > 1e-8:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        z = 0.0
    return tuple(math.degrees(value) for value in (x, y, z))


def align_to_z(vector: Sequence[float]) -> np.ndarray:
    source = np.asarray(vector, dtype=float)
    source /= np.linalg.norm(source)
    target = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        return rotation_axis_angle((1.0, 0.0, 0.0), math.pi)
    axis = np.cross(source, target)
    return rotation_axis_angle(axis, math.acos(dot))


def base_orientation(crystal: Crystal, hkl: Sequence[int]) -> np.ndarray:
    """
    Совмещает выбранный полюс с Z.

    Нулевая линия φ задаётся проекцией первого из a*, b*, c*, который не
    параллелен выбранному полюсу.
    """
    pole = crystal.reciprocal_vector(hkl)
    pole /= np.linalg.norm(pole)
    alignment = align_to_z(pole)

    candidates = [crystal.reciprocal[:, index] for index in range(3)]
    reference = max(
        candidates,
        key=lambda item: np.linalg.norm(item - np.dot(item, pole) * pole),
    )
    reference = reference - np.dot(reference, pole) * pole
    transformed = alignment @ reference
    azimuth = math.atan2(transformed[1], transformed[0])
    return rotation_z(-math.degrees(azimuth)) @ alignment


def rotation_between(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    first_array /= np.linalg.norm(first_array)
    second_array /= np.linalg.norm(second_array)
    dot = float(np.clip(np.dot(first_array, second_array), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(first_array[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        return rotation_axis_angle(np.cross(first_array, fallback), math.pi)
    return rotation_axis_angle(np.cross(first_array, second_array), math.acos(dot))


def format_hkl(hkl: Sequence[int], braces: bool = False) -> str:
    left, right = ("{", "}") if braces else ("(", ")")
    return left + " ".join(str(int(value)) for value in hkl) + right


def in_plane_alignment(current_phi: float, target_phi: float) -> np.ndarray:
    """Return the shortest rotation around sample Z to the target azimuth."""

    delta = (target_phi - current_phi + 180.0) % 360.0 - 180.0
    return rotation_z(delta)


def pole_plot_coordinates(radius: float, phi_rad: float) -> tuple[float, float]:
    """Rotate the usual pole projection by 90° without mirroring it.

    The original zero direction therefore moves from the right-hand boundary
    to the top without reversing the order of poles.
    """

    return -radius * math.sin(phi_rad), radius * math.cos(phi_rad)


def pole_display_orientation(orientation: np.ndarray) -> np.ndarray:
    """Apply the same fixed 90° display rotation to the crystal structure."""

    return rotation_z(90.0) @ orientation


def pole_plot_to_sphere(
    x: float,
    y: float,
    projection: str,
) -> np.ndarray:
    """Invert the displayed pole-figure projection for mouse rotation."""

    radius = math.hypot(x, y)
    if radius > 0.999999:
        x /= radius / 0.999999
        y /= radius / 0.999999
        radius = 0.999999
    if choice_code("projection", projection) == "equal_area":
        z = 1.0 - radius * radius
        factor = math.sqrt(max(0.0, 2.0 - radius * radius))
        return np.array([y * factor, -x * factor, z])
    denominator = 1.0 + radius * radius
    return np.array(
        [
            2.0 * y / denominator,
            -2.0 * x / denominator,
            (1.0 - radius * radius) / denominator,
        ]
    )


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


def pole_display_position(
    point: PolePoint,
    boundary_radius: float = 0.985,
) -> tuple[float, float]:
    """Keep equatorial markers just inside the primitive-circle outline.

    The scientific projection coordinates remain unchanged in ``PolePoint``;
    only the drawing and hit-testing position is inset.
    """

    radius = math.hypot(point.x, point.y)
    if radius <= boundary_radius or radius <= 1e-14:
        return point.x, point.y
    scale = boundary_radius / radius
    return point.x * scale, point.y * scale


@dataclass(frozen=True)
class Reflection:
    hkl: tuple[int, int, int]
    d_spacing: float
    two_theta: float


def _friedel_representative(hkl: Sequence[int]) -> bool:
    """Оставляет только один индекс из пары (hkl) и (-h-k-l)."""
    for value in hkl:
        if value:
            return value > 0
    return False


def available_reflections(
    crystal: Crystal,
    d_lower: float,
    d_upper: float,
    wavelength: float,
) -> list[Reflection]:
    """
    Возвращает все разрешённые отражения в диапазоне d.

    Пары Фриделя объединяются, но симметрически эквивалентные направления
    сохраняются: именно они образуют отдельные точки полюсной фигуры.
    """
    if d_lower <= 0 or d_upper <= 0:
        raise ValueError(
            localised(
                "The d limits must be positive.",
                "Les limites de d doivent être positives.",
                "Границы d должны быть положительными.",
            )
        )
    if d_lower > d_upper:
        raise ValueError(
            localised(
                "The lower d limit cannot exceed the upper limit.",
                "La limite inférieure de d ne peut pas dépasser la limite supérieure.",
                "Нижняя граница d не может быть больше верхней.",
            )
        )
    if wavelength <= 0:
        raise ValueError(
            localised(
                "The wavelength must be positive.",
                "La longueur d’onde doit être positive.",
                "Длина волны должна быть положительной.",
            )
        )

    # Если |g| <= 1/d_lower, то |h_i| не превосходит нормы соответствующей
    # строки B^-1, умноженной на |g|. Это даёт безопасные пределы перебора
    # для ячейки любой сингонии.
    inverse_reciprocal = np.linalg.inv(crystal.reciprocal)
    bounds = [
        max(1, int(math.ceil(np.linalg.norm(row) / d_lower + 1e-9)))
        for row in inverse_reciprocal
    ]
    candidates = (2 * bounds[0] + 1) * (2 * bounds[1] + 1) * (2 * bounds[2] + 1)
    if candidates > 2_000_000:
        raise ValueError(
            localised(
                "The selected lower d limit requires testing more than two million "
                "reciprocal-lattice nodes. Increase the lower d limit.",
                "La limite inférieure de d choisie nécessite de tester plus de deux "
                "millions de nœuds du réseau réciproque. Augmentez cette limite.",
                "Выбранная нижняя граница d требует перебора более двух миллионов "
                "узлов. Увеличьте нижнюю границу d.",
            )
        )

    result: list[Reflection] = []
    tolerance = 1e-10
    for h in range(-bounds[0], bounds[0] + 1):
        for k in range(-bounds[1], bounds[1] + 1):
            for l in range(-bounds[2], bounds[2] + 1):
                hkl = (h, k, l)
                if not _friedel_representative(hkl):
                    continue
                d_value = crystal.d_spacing(hkl)
                if d_value < d_lower - tolerance or d_value > d_upper + tolerance:
                    continue
                if crystal.is_systematically_absent(hkl):
                    continue
                two_theta = crystal.two_theta(hkl, wavelength)
                if two_theta is None:
                    continue
                result.append(Reflection(hkl, d_value, two_theta))

    result.sort(key=lambda item: (-item.d_spacing, item.hkl))
    return result


def project_reflections(
    crystal: Crystal,
    reflections: Sequence[Reflection],
    orientation: np.ndarray,
    projection: str,
) -> list[PolePoint]:
    points: list[PolePoint] = []

    def append_point(
        reflection: Reflection,
        direction: np.ndarray,
        label: np.ndarray,
    ) -> None:
        z = float(np.clip(direction[2], 0.0, 1.0))
        chi_rad = math.acos(z)
        phi_rad = (
            0.0
            if chi_rad < 1e-10
            else math.atan2(direction[1], direction[0])
        )
        if choice_code("projection", projection) == "equal_area":
            radius = math.sqrt(2.0) * math.sin(chi_rad / 2.0)
        else:
            radius = math.tan(chi_rad / 2.0)
        plot_x, plot_y = pole_plot_coordinates(radius, phi_rad)
        points.append(
            PolePoint(
                tuple(int(value) for value in label),
                reflection.d_spacing,
                reflection.two_theta,
                direction.copy(),
                math.degrees(chi_rad),
                math.degrees(phi_rad) % 360.0,
                plot_x,
                plot_y,
            )
        )

    for reflection in reflections:
        direction = orientation @ crystal.reciprocal_vector(reflection.hkl)
        direction /= np.linalg.norm(direction)
        label = np.asarray(reflection.hkl, dtype=int)

        if direction[2] < -1e-10:
            direction = -direction
            label = -label
        elif abs(direction[2]) <= 1e-10:
            direction = direction.copy()
            direction[2] = 0.0

        append_point(reflection, direction, label)
        if abs(direction[2]) <= 1e-10:
            # Friedel opposites coincide after projection everywhere except on
            # the equator.  At χ = 90° both ends of the plane normal belong to
            # the visible boundary and must be drawn at opposite positions.
            append_point(reflection, -direction, -label)
    points.sort(
        key=lambda item: (
            round(item.chi, 8),
            round(item.phi, 8),
            -item.d_spacing,
            item.hkl,
        )
    )
    return points


def group_coincident_poles(
    points: Sequence[PolePoint],
) -> list[list[PolePoint]]:
    """Группирует гармоники, лежащие в одной точке полюсной фигуры."""
    groups: dict[tuple[int, int], list[PolePoint]] = {}
    for point in points:
        key = (round(point.x * 1e8), round(point.y * 1e8))
        groups.setdefault(key, []).append(point)
    result = list(groups.values())
    for group in result:
        group.sort(key=lambda item: (-item.d_spacing, item.hkl))
    result.sort(
        key=lambda group: (
            round(group[0].chi, 8),
            round(group[0].phi, 8),
        )
    )
    return result


def marker_sizes_by_d(
    d_values: Sequence[float],
    minimum: float = 22.0,
    maximum: float = 250.0,
) -> np.ndarray:
    """Scale marker areas by d without implying calculated intensity."""

    values = np.asarray(d_values, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
        return np.full(values.shape, 58.0, dtype=float)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return minimum + (maximum - minimum) * np.square(normalized)


def calculated_intensity_by_spacing(
    diffraction_structure,
    radiations: list[tuple[str, float, float]],
) -> dict[float, float]:
    """Return normalized powder intensities keyed like the reflection engine."""
    if diffraction_structure is None or diffraction_structure.cell_only:
        return {}
    try:
        from .cif_xrd import _data_path, calculate_reflections, load_scattering_factors
    except ImportError:  # pragma: no cover
        from cif_xrd import _data_path, calculate_reflections, load_scattering_factors
    rows = calculate_reflections(
        diffraction_structure,
        load_scattering_factors(_data_path()),
        radiations,
        min_two_theta=0.01,
        max_two_theta=179.9,
        min_intensity=0.0,
    )
    totals: dict[float, float] = {}
    for row in rows:
        if row.intensity is None:
            continue
        key = round(1.0 / (row.d * row.d), 8)
        totals[key] = totals.get(key, 0.0) + row.intensity
    maximum = max(totals.values(), default=0.0)
    if maximum > 0:
        totals = {key: 100.0 * value / maximum for key, value in totals.items()}
    return totals


@dataclass
class DisplayAtom:
    element: str
    fractional: np.ndarray
    cartesian: np.ndarray
    occupancy: float


def unit_cell_display_atoms(crystal: Crystal) -> list[DisplayAtom]:
    """
    Возвращает атомы ячейки вместе с копиями на противоположных гранях.

    Это соответствует обычному изображению ячейки в кристаллографических
    программах, где атом на координате 0 также показывается на координате 1.
    """
    result: list[DisplayAtom] = []
    for atom in crystal.atoms:
        choices: list[list[float]] = []
        for coordinate in atom.fractional:
            if abs(coordinate) < 1e-8:
                choices.append([0.0, 1.0])
            else:
                choices.append([float(coordinate)])
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


def unit_cell_bonds(atoms: Sequence[DisplayAtom]) -> list[tuple[int, int]]:
    """Оценивает связи по ковалентным радиусам; для оксидов оставляет M–O."""
    try:
        from .atom_styles import covalent_radius
    except ImportError:  # pragma: no cover
        from atom_styles import covalent_radius
    result: list[tuple[int, int]] = []
    has_oxygen = any(atom.element == "O" for atom in atoms)
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            atom_a, atom_b = atoms[first], atoms[second]
            if atom_a.element == atom_b.element:
                continue
            pair = {atom_a.element, atom_b.element}
            if has_oxygen and "O" not in pair:
                continue
            radius_a = covalent_radius(atom_a.element)
            radius_b = covalent_radius(atom_b.element)
            cutoff = 1.25 * (radius_a + radius_b)
            distance = float(np.linalg.norm(atom_a.cartesian - atom_b.cartesian))
            if 0.35 < distance <= cutoff:
                result.append((first, second))
    return result


def draw_crystal_structure(axis, crystal: Crystal | None, orientation: np.ndarray,
                           *, preview=False, show_basis=True):
    try:
        from .structure_render import render_structure
    except ImportError:
        from structure_render import render_structure
    return render_structure(axis, crystal, orientation, preview=preview, show_basis=show_basis)


def _build_gui(
    initial_path: str | None,
    parent=None,
    auto_prompt: bool = True,
    on_open_cif=None,
    radiations_provider=None,
):
    try:
        from .controls import CollapsibleSection, ScrollableControls, FrameScheduler
        from .structure_render import screen_drag_rotation
    except ImportError:
        from controls import CollapsibleSection, ScrollableControls, FrameScheduler
        from structure_render import screen_drag_rotation
    import tkinter as tk
    from tkinter import ttk

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    class PoleFigureApp:
        def __init__(self, root, path: str | None):
            self.root = root
            if isinstance(self.root, (tk.Tk, tk.Toplevel)):
                self.root.title(
                    localised(
                        "Calculated pole figure from CIF",
                        "Figure de pôles calculée depuis un CIF",
                        "Теоретическая полюсная фигура по CIF",
                    )
                )
                self.root.geometry("1500x850")
                self.root.minsize(1100, 700)

            self.crystal: Crystal | None = None
            self.cif_document = None
            self.radiations_provider = radiations_provider
            self.center_hkl = (0, 1, 0)
            self.base_rotation = np.eye(3)
            self.user_rotation = np.eye(3)
            self.reflections: list[Reflection] = []
            self.points: list[PolePoint] = []
            self.point_groups: list[list[PolePoint]] = []
            self.center_map: dict[str, tuple[int, int, int]] = {}
            self.selected_hkl: tuple[int, int, int] | None = None
            self.press_event = None
            self.last_arcball: np.ndarray | None = None
            self.last_drag_pixel: tuple[float, float] | None = None
            self.drag_mode: str | None = None
            self.dragged = False

            self.path_var = tk.StringVar(value="CIF не открыт")
            self.structure_var = tk.StringVar(value="")
            self.h_var = tk.StringVar(value="0")
            self.k_var = tk.StringVar(value="1")
            self.l_var = tk.StringVar(value="0")
            self.max_index_var = tk.IntVar(value=4)
            self.wavelength_var = tk.StringVar(value="1.5406")
            self.d_lower_var = tk.StringVar(value="1.0")
            self.d_upper_var = tk.StringVar(value="13")
            self.projection_var = tk.StringVar(
                value=translate_text("Стереографическая")
            )
            self.labels_var = tk.BooleanVar(value=False)
            self.show_structure_var = tk.BooleanVar(value=False)
            self.basis_visible = tk.BooleanVar(value=True)
            self._colorbar = None
            self.size_by_d_var = tk.BooleanVar(value=False)
            self.color_mode_var = tk.StringVar(value="uniform")
            self.intensity_by_spacing: dict[float, float] | None = None
            self.rotation_vars = [
                tk.StringVar(value="0.0"),
                tk.StringVar(value="0.0"),
                tk.StringVar(value="0.0"),
            ]
            self.relative_rotation_vars = [
                tk.StringVar(value="0.0"),
                tk.StringVar(value="0.0"),
                tk.StringVar(value="0.0"),
            ]
            self.status_var = LocalizedStringVar(value="")

            self._build_layout()
            self._connect_canvas()
            self._frames = FrameScheduler(self.root, lambda: self.redraw(preview=True))
            apply_language(self.root)
            if path:
                self.load_cif(path)
            elif auto_prompt:
                self.root.after(100, self.ask_cif)

        def _build_layout(self) -> None:
            main = ttk.Frame(self.root, padding=8)
            main.grid(row=0, column=0, sticky="nsew")
            self.root.rowconfigure(0, weight=1)
            self.root.columnconfigure(0, weight=1)
            main.rowconfigure(0, weight=1)
            main.columnconfigure(1, weight=1)

            self.control_panel = ScrollableControls(main, width=325, padding=(0, 0, 10, 0))
            self.control_panel.grid(row=0, column=0, sticky="nsew")
            self.controls_canvas = self.control_panel.canvas
            controls = self.control_panel.body

            plot_frame = ttk.Frame(main)
            plot_frame.grid(row=0, column=1, sticky="nsew")
            plot_frame.rowconfigure(0, weight=1)
            plot_frame.columnconfigure(0, weight=1)

            information = ttk.Frame(main, padding=(10, 0, 0, 0), width=290)
            information.grid(row=0, column=2, sticky="ns")
            information.grid_propagate(False)

            file_box = CollapsibleSection(controls, text="Структура CIF", padding=8)
            file_box.pack(fill="x", pady=(0, 8))
            ttk.Button(file_box, text="Открыть CIF…", command=self.ask_cif).pack(
                fill="x"
            )
            ttk.Label(
                file_box,
                textvariable=self.path_var,
                wraplength=275,
                justify="left",
            ).pack(fill="x", pady=(7, 2))
            ttk.Label(
                file_box,
                textvariable=self.structure_var,
                wraplength=275,
                justify="left",
            ).pack(fill="x")

            center_box = CollapsibleSection(
                controls, text="Центрирование по полюсу", padding=8
            )
            center_box.pack(fill="x", pady=(0, 8))
            self.center_prompt = ttk.Label(center_box, text="Выбрать разрешённый полюс:")
            self.center_prompt.pack(anchor="w")
            self.center_combo = ttk.Combobox(center_box, state="readonly")
            self.center_combo.pack(fill="x", pady=(3, 7))
            self.center_combo.bind("<<ComboboxSelected>>", self.choose_center)

            hkl_row = ttk.Frame(center_box)
            hkl_row.pack(fill="x")
            for column, (label, variable) in enumerate(
                zip(("h", "k", "l"), (self.h_var, self.k_var, self.l_var))
            ):
                ttk.Label(hkl_row, text=label).grid(row=0, column=2 * column)
                entry = ttk.Entry(hkl_row, width=5, textvariable=variable)
                entry.grid(row=0, column=2 * column + 1, padx=(3, 9))
            ttk.Button(
                center_box,
                text="Поместить полюс (hkl) в центр",
                command=self.apply_center,
            ).pack(fill="x", pady=(7, 0))

            list_row = ttk.Frame(center_box)
            list_row.pack(fill="x", pady=(8, 0))
            ttk.Label(list_row, text="Макс. индекс списка").grid(row=0, column=0)
            ttk.Spinbox(
                list_row,
                from_=1,
                to=12,
                width=4,
                textvariable=self.max_index_var,
                command=self.refresh_center_list,
            ).grid(row=0, column=1, padx=(4, 0))

            range_box = CollapsibleSection(
                controls, text="Отображаемые отражения", padding=8
            )
            range_box.pack(fill="x", pady=(0, 8))
            range_row = ttk.Frame(range_box)
            range_row.pack(fill="x")
            ttk.Label(range_row, text="d от, Å").grid(row=0, column=0)
            ttk.Label(range_row, text="d до, Å").grid(row=0, column=1)
            ttk.Label(range_row, text="λ, Å").grid(row=0, column=2)
            for column, variable in enumerate(
                (self.d_lower_var, self.d_upper_var, self.wavelength_var)
            ):
                entry = ttk.Entry(range_row, width=8, textvariable=variable)
                entry.grid(row=1, column=column, padx=(0, 7), pady=(2, 0))
                entry.bind("<Return>", lambda _event: self.rebuild_reflections())
                if column == 2:
                    self.wavelength_entry = entry
            if self.radiations_provider is not None:
                self.wavelength_entry.configure(state="readonly")
            self.range_button = ttk.Button(
                range_box,
                text="Построить все разрешённые отражения",
                command=self.rebuild_reflections,
            )
            self.range_button.pack(fill="x", pady=(7, 0))

            view_box = CollapsibleSection(controls, text="Отображение", padding=8)
            view_box.pack(fill="x", pady=(0, 8))
            ttk.Label(view_box, text="Проекция:").pack(anchor="w")
            projection_combo = ttk.Combobox(
                view_box,
                state="readonly",
                textvariable=self.projection_var,
                values=("Стереографическая", "Равноплощадная"),
            )
            projection_combo.pack(fill="x", pady=(3, 5))
            projection_combo.bind("<<ComboboxSelected>>", lambda _event: self.redraw())
            ttk.Checkbutton(
                view_box,
                text="Подписывать полюса",
                variable=self.labels_var,
                command=self.redraw,
            ).pack(anchor="w")
            ttk.Checkbutton(
                view_box,
                text="Показать структуру рядом",
                variable=self.show_structure_var,
                command=self.redraw,
            ).pack(anchor="w")
            ttk.Checkbutton(
                view_box,
                text="Размер точек по d",
                variable=self.size_by_d_var,
                command=self.redraw,
            ).pack(anchor="w")
            colour_box = ttk.LabelFrame(view_box, text="Цвет точек", padding=5)
            colour_box.pack(fill="x", pady=(5, 0))
            for text_value, mode in (
                ("Один цвет", "uniform"),
                ("Цвет точек по d", "d"),
            ):
                ttk.Radiobutton(
                    colour_box,
                    text=text_value,
                    value=mode,
                    variable=self.color_mode_var,
                    command=self.change_colour_mode,
                ).pack(anchor="w")
            self.intensity_colour_radio = ttk.Radiobutton(
                colour_box,
                text="Цвет точек по расчётной интенсивности",
                value="intensity",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.intensity_colour_radio.pack(anchor="w")

            ttk.Checkbutton(view_box, text="Базисные векторы", variable=self.basis_visible,
                            command=self.redraw).pack(anchor="w")

            rotation_box = CollapsibleSection(
                controls, text="Абсолютный поворот кристалла", padding=8
            )
            rotation_box.pack(fill="x", pady=(0, 8))
            rotation_row = ttk.Frame(rotation_box)
            rotation_row.pack(fill="x")
            for column, (axis, variable) in enumerate(
                zip(("X, °", "Y, °", "Z, °"), self.rotation_vars)
            ):
                ttk.Label(rotation_row, text=axis).grid(row=0, column=column)
                entry = ttk.Entry(rotation_row, width=8, textvariable=variable)
                entry.grid(row=1, column=column, padx=(0, 7), pady=(2, 0))
                entry.bind("<Return>", lambda _event: self.apply_exact_rotation())
            ttk.Button(
                rotation_box,
                text="Установить абсолютные углы",
                command=self.apply_exact_rotation,
            ).pack(fill="x", pady=(7, 4))
            ttk.Button(
                rotation_box,
                text="Вернуть центрирующий полюс",
                command=self.reset_rotation,
            ).pack(fill="x")

            relative_rotation_box = CollapsibleSection(
                controls, text="Относительный поворот кристалла", padding=8
            )
            relative_rotation_box.pack(fill="x", pady=(0, 8))
            relative_rotation_row = ttk.Frame(relative_rotation_box)
            relative_rotation_row.pack(fill="x")
            for column, (axis, variable) in enumerate(
                zip(("ΔX, °", "ΔY, °", "ΔZ, °"), self.relative_rotation_vars)
            ):
                ttk.Label(relative_rotation_row, text=axis).grid(
                    row=0, column=column
                )
                entry = ttk.Entry(
                    relative_rotation_row,
                    width=8,
                    textvariable=variable,
                )
                entry.grid(row=1, column=column, padx=(0, 7), pady=(2, 0))
                entry.bind(
                    "<Return>",
                    lambda _event: self.apply_relative_rotation(),
                )
            ttk.Button(
                relative_rotation_box,
                text="Повернуть относительно текущего",
                command=self.apply_relative_rotation,
            ).pack(fill="x", pady=(7, 0))

            align_box = CollapsibleSection(
                controls, text="Совместить выбранный полюс", padding=8
            )
            align_box.pack(fill="x", pady=(0, 8))
            for column, (label, target) in enumerate(
                (("+X", 270.0), ("−X", 90.0), ("+Y", 0.0), ("−Y", 180.0))
            ):
                ttk.Button(
                    align_box,
                    text=label,
                    command=lambda angle=target: self.align_selected_pole(angle),
                ).grid(
                    row=0,
                    column=column,
                    sticky="ew",
                    padx=(0 if column == 0 else 3, 0),
                )
                align_box.columnconfigure(column, weight=1)

            ttk.Label(
                controls,
                text=(
                    "Перетаскивание внутри круга свободно вращает кристалл.\n"
                    "Щелчок по полюсу выводит его данные справа."
                ),
                wraplength=285,
                justify="left",
            ).pack(fill="x", pady=(2, 8))
            ttk.Label(
                controls,
                textvariable=self.status_var,
                foreground="#8a3f00",
                wraplength=285,
                justify="left",
            ).pack(fill="x")

            self.figure = Figure(figsize=(7.2, 7.2), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.structure_ax = None
            self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            ttk.Label(information, text="Выбранный полюс").pack(anchor="w")
            self.info_text = tk.Text(
                information,
                width=31,
                height=23,
                wrap="word",
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=8,
                state="disabled",
                font=("TkDefaultFont", 10),
            )
            self.info_text.pack(fill="x", pady=(5, 10))

        def _connect_canvas(self) -> None:
            self.canvas.mpl_connect("button_press_event", self.on_press)
            self.canvas.mpl_connect("motion_notify_event", self.on_motion)
            self.canvas.mpl_connect("button_release_event", self.on_release)
            self.canvas.mpl_connect("scroll_event", self.on_scroll)

        def ask_cif(self) -> None:
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Выберите структурный файл CIF",
                filetypes=[("CIF", "*.cif"), ("Все файлы", "*.*")],
            )
            if path:
                if on_open_cif is not None:
                    on_open_cif(path)
                    return
                self.load_cif(path)

        def load_cif(self, path: str) -> None:
            try:
                try:
                    from .cif_document import load_cif_document
                except ImportError:  # pragma: no cover
                    from cif_document import load_cif_document
                self.load_document(load_cif_document(path))
                return
            except Exception as exc:
                messagebox.showerror(
                    localised(
                        "Could not open CIF",
                        "Impossible d’ouvrir le CIF",
                        "Не удалось открыть CIF",
                    ),
                    localised(
                        "The CIF could not be read. Check the unit cell, atom sites "
                        "and explicit symmetry operations.\n\nDetails: ",
                        "Le CIF n’a pas pu être lu. Vérifiez la maille, les positions "
                        "atomiques et les opérations de symétrie explicites.\n\nDétails : ",
                        "Не удалось прочитать CIF. Проверьте параметры ячейки, "
                        "позиции атомов и явные операции симметрии.\n\nПодробности: ",
                    )
                    + str(exc),
                    parent=self.root,
                )
                return

        def load_document(self, document) -> None:
            self._frames.cancel()
            self.reflections = []
            self.points = []
            self.point_groups = []
            self.selected_hkl = None
            self.intensity_by_spacing = None
            self.cif_document = document
            crystal = document.crystal
            self.crystal = crystal
            self.path_var.set(document.source.name)
            self.structure_var.set(
                f"{crystal.formula}; {crystal.space_group}\n"
                f"a = {crystal.a:.4f} Å, b = {crystal.b:.4f} Å, "
                f"c = {crystal.c:.4f} Å"
            )
            cell_only = bool(getattr(document, "is_cell_only", False))
            self.intensity_colour_radio.configure(
                state="disabled" if cell_only else "normal"
            )
            if cell_only and self.color_mode_var.get() == "intensity":
                self.color_mode_var.set("uniform")
            self.center_prompt.configure(
                text=(
                    localised(
                        "Select a pole not systematically forbidden:",
                        "Choisir un pôle non interdit systématiquement :",
                        "Выберите полюс, не запрещённый систематически:",
                    )
                    if cell_only
                    else translate_text("Выбрать разрешённый полюс:")
                )
            )
            self.range_button.configure(
                text=(
                    localised(
                        "Plot reflections not systematically forbidden",
                        "Tracer les réflexions non interdites systématiquement",
                        "Построить отражения, не запрещённые систематически",
                    )
                    if cell_only
                    else translate_text("Построить все разрешённые отражения")
                )
            )
            self.user_rotation = np.eye(3)
            self.base_rotation = base_orientation(crystal, self.center_hkl)
            self.update_rotation_entries()
            self.radiation_changed(rebuild=False)
            self.refresh_center_list()
            self.rebuild_reflections()

        def clear_document(self) -> None:
            self._frames.cancel()
            self.cif_document = None
            self.crystal = None
            self.reflections = []
            self.points = []
            self.point_groups = []
            self.selected_hkl = None
            self.intensity_by_spacing = None
            self.path_var.set(translate_text("CIF не открыт"))
            self.structure_var.set("")
            self.center_combo.configure(values=())
            self.center_combo.set("")
            self.intensity_colour_radio.configure(state="normal")
            self.center_prompt.configure(text=translate_text("Выбрать разрешённый полюс:"))
            self.range_button.configure(
                text=translate_text("Построить все разрешённые отражения")
            )
            self.status_var.set("")
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            self.info_text.configure(state="disabled")
            self.redraw()

        def get_wavelength(self) -> float:
            if self.radiations_provider is not None:
                radiations = self.get_radiations()
                return min(wavelength for _name, wavelength, _weight in radiations)
            try:
                wavelength = float(self.wavelength_var.get().replace(",", "."))
            except ValueError as exc:
                raise ValueError(
                    localised(
                        "The wavelength must be numeric.",
                        "La longueur d’onde doit être numérique.",
                        "Длина волны должна быть числом.",
                    )
                ) from exc
            if wavelength <= 0:
                raise ValueError(
                    localised(
                        "The wavelength must be positive.",
                        "La longueur d’onde doit être positive.",
                        "Длина волны должна быть положительной.",
                    )
                )
            return wavelength

        def get_radiations(self) -> list[tuple[str, float, float]]:
            if self.radiations_provider is None:
                return [("Custom", self.get_wavelength(), 1.0)]
            radiations = list(self.radiations_provider())
            if not radiations:
                raise ValueError(
                    localised(
                        "No spectral line is selected.",
                        "Aucune raie spectrale n’est sélectionnée.",
                        "Не выбрана ни одна спектральная линия.",
                    )
                )
            return radiations

        def radiation_changed(self, rebuild: bool = True) -> None:
            self.intensity_by_spacing = None
            if self.radiations_provider is not None:
                try:
                    wavelength = min(item[1] for item in self.get_radiations())
                    self.wavelength_var.set(f"{wavelength:.5f}")
                except ValueError:
                    return
            if rebuild and self.crystal is not None:
                self.rebuild_reflections()

        def _ensure_intensities(self, show_errors: bool = True) -> bool:
            if self.crystal is None or self.cif_document is None:
                return False
            if getattr(self.cif_document, "is_cell_only", False):
                return False
            if self.intensity_by_spacing is not None:
                return True
            try:
                self.intensity_by_spacing = calculated_intensity_by_spacing(
                    self.cif_document.diffraction,
                    self.get_radiations(),
                )
            except Exception as exc:
                self.intensity_by_spacing = {}
                if show_errors:
                    messagebox.showerror(
                        localised(
                            "Intensity calculation error",
                            "Erreur de calcul de l’intensité",
                            "Ошибка расчёта интенсивности",
                        ),
                        str(exc),
                        parent=self.root,
                    )
                return False
            return bool(self.intensity_by_spacing)

        def change_colour_mode(self) -> None:
            if self.color_mode_var.get() == "intensity" and not self._ensure_intensities():
                self.color_mode_var.set("uniform")
            self.redraw()

        def _point_intensity(self, point: PolePoint) -> float | None:
            if self.intensity_by_spacing is None:
                return None
            return self.intensity_by_spacing.get(
                round(1.0 / (point.d_spacing * point.d_spacing), 8)
            )

        def get_d_range(self) -> tuple[float, float]:
            try:
                d_lower = float(self.d_lower_var.get().replace(",", "."))
                d_upper = float(self.d_upper_var.get().replace(",", "."))
            except ValueError as exc:
                raise ValueError(
                    localised(
                        "The d limits must be numeric.",
                        "Les limites de d doivent être numériques.",
                        "Границы d должны быть числами.",
                    )
                ) from exc
            if d_lower <= 0 or d_upper <= 0:
                raise ValueError(
                    localised(
                        "The d limits must be positive.",
                        "Les limites de d doivent être positives.",
                        "Границы d должны быть положительными.",
                    )
                )
            if d_lower > d_upper:
                raise ValueError(
                    localised(
                        "The lower d limit cannot exceed the upper limit.",
                        "La limite inférieure de d ne peut pas dépasser la limite supérieure.",
                        "Нижняя граница d не может быть больше верхней.",
                    )
                )
            return d_lower, d_upper

        def refresh_center_list(self) -> None:
            if self.crystal is None:
                return
            try:
                max_index = int(self.max_index_var.get())
                wavelength = self.get_wavelength()
                if not 1 <= max_index <= 20:
                    raise ValueError(
                        localised(
                            "The maximum index must be between 1 and 20.",
                            "L’indice maximal doit être compris entre 1 et 20.",
                            "Максимальный индекс должен быть от 1 до 20.",
                        )
                    )
            except (ValueError, tk.TclError) as exc:
                self.status_var.set(str(exc))
                return

            entries: list[tuple[float, tuple[int, int, int], str]] = []
            for h in range(max_index + 1):
                for k in range(max_index + 1):
                    for l in range(max_index + 1):
                        hkl = (h, k, l)
                        if hkl == (0, 0, 0):
                            continue
                        if self.crystal.is_systematically_absent(hkl):
                            continue
                        two_theta = self.crystal.two_theta(hkl, wavelength)
                        if two_theta is None:
                            continue
                        d_value = self.crystal.d_spacing(hkl)
                        display = (
                            f"{format_hkl(hkl)}   "
                            f"2θ={two_theta:.3f}°   d={d_value:.4f} Å"
                        )
                        entries.append((two_theta, hkl, display))
            entries.sort(key=lambda item: (item[0], item[1]))
            self.center_map = {display: hkl for _, hkl, display in entries}
            values = [display for _, _, display in entries]
            self.center_combo.configure(values=values)

            wanted = tuple(abs(value) for value in self.center_hkl)
            for display, hkl in self.center_map.items():
                if hkl == wanted:
                    self.center_combo.set(display)
                    break
            else:
                self.center_combo.set("")

        def choose_center(self, _event=None) -> None:
            display = self.center_combo.get()
            hkl = self.center_map.get(display)
            if hkl is None:
                return
            for variable, value in zip((self.h_var, self.k_var, self.l_var), hkl):
                variable.set(str(value))
            self.apply_center()

        def read_center(self) -> tuple[int, int, int]:
            values = []
            for variable in (self.h_var, self.k_var, self.l_var):
                text = variable.get().strip()
                if not re.fullmatch(r"[+-]?\d+", text):
                    raise ValueError(
                        localised(
                            "The h, k and l indices must be integers.",
                            "Les indices h, k et l doivent être des nombres entiers.",
                            "Индексы h, k, l должны быть целыми числами.",
                        )
                    )
                values.append(int(text))
            hkl = tuple(values)
            if hkl == (0, 0, 0):
                raise ValueError(
                    localised(
                        "The (0 0 0) reflection does not exist.",
                        "La réflexion (0 0 0) n’existe pas.",
                        "Отражение (0 0 0) не существует.",
                    )
                )
            return hkl

        def rebuild_reflections(self) -> None:
            if self.crystal is None:
                return
            try:
                d_lower, d_upper = self.get_d_range()
                wavelength = self.get_wavelength()
                reflections = available_reflections(
                    self.crystal,
                    d_lower,
                    d_upper,
                    wavelength,
                )
            except ValueError as exc:
                messagebox.showerror(
                    localised(
                        "Invalid reflection range",
                        "Intervalle de réflexions incorrect",
                        "Некорректный диапазон отражений",
                    ),
                    str(exc),
                    parent=self.root,
                )
                return
            self.reflections = reflections
            self.intensity_by_spacing = None
            self.selected_hkl = self.center_hkl
            self.refresh_center_list()
            self.redraw()

        def apply_center(self) -> None:
            if self.crystal is None:
                return
            try:
                center_hkl = self.read_center()
                base = base_orientation(self.crystal, center_hkl)
            except ValueError as exc:
                messagebox.showerror(
                    localised(
                        "Invalid h k l",
                        "h k l incorrects",
                        "Некорректные h k l",
                    ),
                    str(exc),
                    parent=self.root,
                )
                return
            self.center_hkl = center_hkl
            self.base_rotation = base
            self.user_rotation = np.eye(3)
            self.selected_hkl = center_hkl
            self.update_rotation_entries()
            if self.reflections:
                self.redraw()
            else:
                self.rebuild_reflections()

        def read_rotation_angles(
            self,
            variables: Sequence[tk.StringVar],
        ) -> list[float] | None:
            try:
                return [
                    float(variable.get().strip().replace(",", "."))
                    for variable in variables
                ]
            except ValueError:
                messagebox.showerror(
                    localised(
                        "Invalid angle",
                        "Angle incorrect",
                        "Некорректный угол",
                    ),
                    localised(
                        "X, Y and Z angles must be numeric.",
                        "Les angles X, Y et Z doivent être numériques.",
                        "Углы X, Y и Z должны быть числами.",
                    ),
                    parent=self.root,
                )
                return None

        def apply_exact_rotation(self) -> None:
            angles = self.read_rotation_angles(self.rotation_vars)
            if angles is None:
                return
            self.user_rotation = euler_matrix(*angles)
            self.update_rotation_entries()
            self.redraw()

        def apply_relative_rotation(self) -> None:
            angles = self.read_rotation_angles(self.relative_rotation_vars)
            if angles is None:
                return
            self.user_rotation = euler_matrix(*angles) @ self.user_rotation
            self.update_rotation_entries()
            for variable in self.relative_rotation_vars:
                variable.set("0.0")
            self.redraw()

        def update_rotation_entries(self) -> None:
            angles = matrix_to_euler(self.user_rotation)
            for variable, angle in zip(self.rotation_vars, angles):
                if abs(angle) < 5e-10:
                    angle = 0.0
                variable.set(f"{angle:.3f}")

        def reset_rotation(self) -> None:
            self.user_rotation = np.eye(3)
            self.selected_hkl = self.center_hkl
            self.update_rotation_entries()
            self.redraw()

        def align_selected_pole(self, target_phi: float) -> None:
            point = None
            if self.selected_hkl is not None:
                for group in self.point_groups:
                    for candidate in group:
                        if candidate.hkl == self.selected_hkl:
                            point = candidate
                            break
                    if point is not None:
                        break
            if point is None:
                messagebox.showinfo(
                    localised(
                        "No selected pole",
                        "Aucun pôle sélectionné",
                        "Полюс не выбран",
                    ),
                    localised(
                        "Click a displayed pole first.",
                        "Cliquez d’abord sur un pôle affiché.",
                        "Сначала щёлкните по отображаемому полюсу.",
                    ),
                    parent=self.root,
                )
                return
            if math.hypot(point.direction[0], point.direction[1]) < 1e-10:
                messagebox.showinfo(
                    localised(
                        "Azimuth is undefined",
                        "Azimut indéfini",
                        "Азимут не определён",
                    ),
                    localised(
                        "The selected pole is at the centre and has no unique azimuth.",
                        "Le pôle sélectionné est au centre et n’a pas d’azimut unique.",
                        "Выбранный полюс находится в центре и не имеет единственного азимута.",
                    ),
                    parent=self.root,
                )
                return
            self.user_rotation = (
                in_plane_alignment(point.phi, target_phi) @ self.user_rotation
            )
            self.update_rotation_entries()
            self.redraw()

        def prepare_plot_axes(self) -> None:
            if self._colorbar is not None:
                self._colorbar.remove()
                self._colorbar = None
            show = self.show_structure_var.get()
            if show == (self.structure_ax is not None):
                return  # Preserve the structure artists and camera limits.
            self.figure.clear()
            if show:
                self.ax = self.figure.add_subplot(1, 2, 1)
                self.structure_ax = self.figure.add_subplot(1, 2, 2, projection="3d")
            else:
                self.ax = self.figure.add_subplot(111)
                self.structure_ax = None

        def draw_grid(self) -> None:
            self.ax.clear()
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlim(-1.10, 1.10)
            self.ax.set_ylim(-1.10, 1.10)
            self.ax.axis("off")
            self.ax.add_patch(
                Circle((0, 0), 1.0, fill=False, linewidth=1.5, color="#303030")
            )

            for chi in (15, 30, 45, 60, 75):
                chi_rad = math.radians(chi)
                if choice_code("projection", self.projection_var.get()) == "equal_area":
                    radius = math.sqrt(2.0) * math.sin(chi_rad / 2.0)
                else:
                    radius = math.tan(chi_rad / 2.0)
                self.ax.add_patch(
                    Circle(
                        (0, 0),
                        radius,
                        fill=False,
                        linewidth=0.55,
                        color="#c2c2c2",
                    )
                )
                self.ax.text(
                    radius + 0.012,
                    0.012,
                    f"{chi}°",
                    color="#777777",
                    fontsize=8,
                    va="bottom",
                )

            for phi in range(0, 360, POLE_AZIMUTH_GRID_STEP_DEG):
                angle = math.radians(phi)
                self.ax.plot(
                    [0, math.cos(angle)],
                    [0, math.sin(angle)],
                    color="#d0d0d0",
                    linewidth=0.5,
                    zorder=0,
                )
            self.ax.text(1.045, 0.0, "X", ha="center", va="center")
            self.ax.text(0.0, 1.055, "Y", ha="center", va="center")

        def draw_structure(self, orientation: np.ndarray, *, preview=False) -> None:
            draw_crystal_structure(self.structure_ax, self.crystal, orientation,
                                   preview=preview, show_basis=self.basis_visible.get())

        def on_scroll(self, event) -> None:
            if self.structure_ax is None or event.inaxes is not self.structure_ax:
                return
            scene = getattr(self.structure_ax, "_crystal_scene", None)
            if scene is None:
                return
            steps = getattr(event, "step", 0) or (1 if event.button == "up" else -1)
            scene.zoom_by(float(steps))
            self.canvas.draw_idle()

        def refresh_atom_styles(self) -> None:
            if self.structure_ax is not None:
                try:
                    from .structure_render import discard_scene
                except ImportError:  # pragma: no cover
                    from structure_render import discard_scene
                discard_scene(self.structure_ax)
            self.redraw()

        def redraw(self, *, preview=False) -> None:
            self.prepare_plot_axes()
            self.draw_grid()
            if self.crystal is None:
                self.canvas.draw_idle()
                return

            orientation = self.user_rotation @ self.base_rotation
            self.points = project_reflections(
                self.crystal,
                self.reflections,
                orientation,
                self.projection_var.get(),
            )
            self.point_groups = group_coincident_poles(self.points)
            representatives = [group[0] for group in self.point_groups]
            display_positions = [
                pole_display_position(point) for point in representatives
            ]
            d_values = [point.d_spacing for point in representatives]
            marker_sizes = (
                marker_sizes_by_d(d_values)
                if self.size_by_d_var.get()
                else np.full(len(representatives), 58.0)
            )
            colour_mode = self.color_mode_var.get()
            if colour_mode == "intensity" and self.intensity_by_spacing is None:
                self._ensure_intensities(show_errors=False)
            if colour_mode in {"d", "intensity"} and representatives:
                from matplotlib.colors import LinearSegmentedColormap, Normalize

                if colour_mode == "intensity":
                    colour_values = [
                        max(
                            (self._point_intensity(point) or 0.0)
                            for point in group
                        )
                        for group in self.point_groups
                    ]
                    normalization = Normalize(vmin=0.0, vmax=100.0)
                    colour_map = "viridis"
                    colourbar_label = localised(
                        "Calculated powder intensity, %",
                        "Intensité calculée du diagramme de poudre, %",
                        "Расчётная интенсивность порошкового графика, %",
                    )
                else:
                    colour_values = d_values
                    d_min = min(d_values)
                    d_max = max(d_values)
                    if math.isclose(d_min, d_max):
                        d_min -= 0.5
                        d_max += 0.5
                    normalization = Normalize(vmin=d_min, vmax=d_max)
                    colour_map = LinearSegmentedColormap.from_list(
                        "d_spacing",
                        (
                            "#0000ff",
                            "#0075ff",
                            "#00d9ff",
                            "#00f06a",
                            "#76ff00",
                            "#ff0000",
                        ),
                    )
                    colourbar_label = "d, Å"
                colored_scatter = self.ax.scatter(
                    [position[0] for position in display_positions],
                    [position[1] for position in display_positions],
                    s=marker_sizes,
                    c=colour_values,
                    cmap=colour_map,
                    norm=normalization,
                    edgecolors="none",
                    clip_on=False,
                    zorder=3,
                )
                colorbar = self.figure.colorbar(
                    colored_scatter,
                    ax=self.ax,
                    fraction=0.045,
                    pad=0.025,
                    shrink=0.78,
                )
                colorbar.set_label(colourbar_label)
                self._colorbar = colorbar
            else:
                self.ax.scatter(
                    [position[0] for position in display_positions],
                    [position[1] for position in display_positions],
                    s=marker_sizes,
                    color="#000000" if self.size_by_d_var.get() else "#2d6da3",
                    edgecolors="none" if self.size_by_d_var.get() else "white",
                    linewidths=0.0 if self.size_by_d_var.get() else 0.8,
                    clip_on=False,
                    zorder=3,
                )

            selected_group = None
            if self.selected_hkl is not None:
                target = orientation @ self.crystal.reciprocal_vector(
                    self.selected_hkl
                )
                target /= np.linalg.norm(target)
                if target[2] < -1e-10:
                    target = -target
                elif abs(target[2]) <= 1e-10:
                    target[2] = 0.0
                for group in self.point_groups:
                    if float(np.dot(group[0].direction, target)) > 1.0 - 1e-8:
                        selected_group = group
                        break
                if selected_group is not None:
                    selected = selected_group[0]
                    selected_x, selected_y = pole_display_position(selected)
                    self.ax.scatter(
                        [selected_x],
                        [selected_y],
                        s=105 + 24 * len(selected_group),
                        facecolors="none",
                        edgecolors="#d23b2d",
                        linewidths=2.0,
                        clip_on=False,
                        zorder=4,
                    )
                    self.show_information(selected_group)

            if self.labels_var.get():
                for group in self.point_groups:
                    point = group[0]
                    display_x, display_y = pole_display_position(point)
                    suffix = f" +{len(group) - 1}" if len(group) > 1 else ""
                    self.ax.annotate(
                        format_hkl(point.hkl) + suffix,
                        (display_x, display_y),
                        xytext=(7, 6),
                        textcoords="offset points",
                        fontsize=9,
                        color="#202020",
                        zorder=5,
                    )

            try:
                d_lower, d_upper = self.get_d_range()
                wavelength = self.get_wavelength()
            except ValueError:
                d_lower, d_upper, wavelength = 0.0, 0.0, 0.0
            projection_name = localised(
                "equal-area" if choice_code("projection", self.projection_var.get()) == "equal_area" else "stereographic",
                "équivalente" if choice_code("projection", self.projection_var.get()) == "equal_area" else "stéréographique",
                "равноплощадная" if choice_code("projection", self.projection_var.get()) == "equal_area" else "стереографическая",
            )
            if getattr(self.cif_document, "is_cell_only", False):
                title = localised(
                    f"Reflections not systematically forbidden · d = {d_lower:g}–{d_upper:g} Å · {projection_name} projection",
                    f"Réflexions non interdites systématiquement · d = {d_lower:g}–{d_upper:g} Å · projection {projection_name}",
                    f"Отражения, не запрещённые систематически · d = {d_lower:g}–{d_upper:g} Å · {projection_name} проекция",
                )
            else:
                title = localised(
                    f"All allowed reflections · d = {d_lower:g}–{d_upper:g} Å · "
                    f"{projection_name} projection",
                    f"Toutes les réflexions autorisées · d = {d_lower:g}–{d_upper:g} Å · "
                    f"projection {projection_name}",
                    f"Все разрешённые отражения · d = {d_lower:g}–{d_upper:g} Å · "
                    f"{projection_name} проекция",
                )
            self.ax.set_title(title, pad=12, fontsize=12)
            status = localised(
                f"Centre: {format_hkl(self.center_hkl)}. "
                f"Reflections: {len(self.reflections)}; distinct positions: "
                f"{len(self.point_groups)}.",
                f"Centre : {format_hkl(self.center_hkl)}. "
                f"Réflexions : {len(self.reflections)} ; positions distinctes : "
                f"{len(self.point_groups)}.",
                f"Центр: {format_hkl(self.center_hkl)}. "
                f"Отражений: {len(self.reflections)}; различимых положений: "
                f"{len(self.point_groups)}.",
            )
            if wavelength and d_lower < wavelength / 2.0:
                status += localised(
                    f" Reflections with d < λ/2 = {wavelength / 2.0:.4f} Å "
                    "are excluded by Bragg’s law.",
                    f" Les réflexions avec d < λ/2 = {wavelength / 2.0:.4f} Å "
                    "sont exclues par la loi de Bragg.",
                    f" Отражения с d < λ/2 = {wavelength / 2.0:.4f} Å "
                    "исключены по условию Брэгга.",
                )
            if self.crystal.is_systematically_absent(self.center_hkl):
                status += localised(
                    " The centred pole is systematically absent and is used only "
                    "as a geometric axis.",
                    " Le pôle centré est systématiquement éteint et sert uniquement "
                    "d’axe géométrique.",
                    " Центрирующий полюс систематически погашен; "
                    "он используется только как геометрическая ось.",
                )
            if not self.reflections:
                if getattr(self.cif_document, "is_cell_only", False):
                    status += localised(
                        " There are no reflections not systematically forbidden in the selected range.",
                        " Il n’y a aucune réflexion non interdite systématiquement dans l’intervalle sélectionné.",
                        " В выбранном диапазоне нет отражений, не запрещённых систематически.",
                    )
                else:
                    status += localised(
                        " There are no allowed reflections in the selected range.",
                        " Aucune réflexion autorisée dans l’intervalle sélectionné.",
                        " В выбранном диапазоне разрешённых отражений нет.",
                    )
            self.status_var.set(status)
            self.draw_structure(pole_display_orientation(orientation), preview=preview)
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw_idle()

        def show_information(self, group: Sequence[PolePoint]) -> None:
            if self.crystal is None:
                return
            point = group[0]
            if self.selected_hkl is not None:
                for candidate in group:
                    if candidate.hkl == self.selected_hkl:
                        point = candidate
                        break
            wavelength = self.get_wavelength()
            lines = [
                f"hkl: {format_hkl(point.hkl)}",
                localised(
                    f"Centring: {format_hkl(self.center_hkl)}",
                    f"Centrage : {format_hkl(self.center_hkl)}",
                    f"Центрирование: {format_hkl(self.center_hkl)}",
                ),
                "",
                f"χ: {point.chi:.4f}°",
                f"φ: {point.phi:.4f}°",
                "",
                localised("Projection coordinates:", "Coordonnées de projection :", "Координаты проекции:"),
                f"X: {point.x:.6f}",
                f"Y: {point.y:.6f}",
                "",
                localised("Pole unit vector:", "Vecteur unitaire du pôle :", "Единичный вектор полюса:"),
                f"x: {point.direction[0]:.6f}",
                f"y: {point.direction[1]:.6f}",
                f"z: {point.direction[2]:.6f}",
                "",
                f"d(hkl): {point.d_spacing:.6f} Å",
                f"2θ: {point.two_theta:.5f}°",
                f"λ: {wavelength:.5f} Å",
                localised(
                    f"Calculated powder intensity: {self._point_intensity(point):.3f}%"
                    if self._point_intensity(point) is not None
                    else "Calculated powder intensity: —",
                    f"Intensité calculée du diagramme de poudre : {self._point_intensity(point):.3f} %"
                    if self._point_intensity(point) is not None
                    else "Intensité calculée du diagramme de poudre : —",
                    f"Расчётная интенсивность порошкового графика: {self._point_intensity(point):.3f}%"
                    if self._point_intensity(point) is not None
                    else "Расчётная интенсивность порошкового графика: —",
                ),
            ]
            if len(group) > 1:
                lines.extend(
                    [
                        "",
                        localised(
                            f"Reflections at this position: {len(group)}",
                            f"Réflexions à cette position : {len(group)}",
                            f"В этой позиции отражений: {len(group)}",
                        ),
                    ]
                )
                for candidate in group:
                    lines.append(
                        f"{format_hkl(candidate.hkl)}: "
                        f"d={candidate.d_spacing:.6f} Å; "
                        f"2θ={candidate.two_theta:.5f}°; "
                        + (
                            f"Irel={self._point_intensity(candidate):.3f}%"
                            if self._point_intensity(candidate) is not None
                            else "Irel=—"
                        )
                    )
            lines.extend(
                [
                    "",
                    localised(
                        "Systematic absence: no",
                        "Extinction systématique : non",
                        "Систематическое погасание: нет",
                    ),
                ]
            )
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            self.info_text.insert("1.0", "\n".join(lines))
            self.info_text.configure(state="disabled")

        def plot_to_sphere(self, x: float, y: float) -> np.ndarray:
            return pole_plot_to_sphere(x, y, self.projection_var.get())

        def on_press(self, event) -> None:
            if event.button != 1:
                return
            if self.structure_ax is not None and event.inaxes is self.structure_ax:
                self.press_event = (event.x, event.y)
                self.last_drag_pixel = (event.x, event.y)
                self.last_arcball = None
                self.drag_mode = "structure"
                self.dragged = False
                return
            if event.inaxes is not self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            if math.hypot(event.xdata, event.ydata) > 1.05:
                return
            self.press_event = (event.x, event.y)
            self.last_arcball = self.plot_to_sphere(event.xdata, event.ydata)
            self.last_drag_pixel = None
            self.drag_mode = "pole"
            self.dragged = False

        def on_motion(self, event) -> None:
            if self.press_event is None or self.drag_mode is None or event.x is None or event.y is None:
                return
            if math.hypot(event.x - self.press_event[0], event.y - self.press_event[1]) > 3:
                self.dragged = True
            if self.drag_mode == "structure":
                if self.last_drag_pixel is None:
                    self.last_drag_pixel = (event.x, event.y)
                    return
                dx = event.x - self.last_drag_pixel[0]
                dy = event.y - self.last_drag_pixel[1]
                if dx == 0 and dy == 0:
                    return
                delta = screen_drag_rotation(dx, dy)
                self.user_rotation = delta @ self.user_rotation
                self.last_drag_pixel = (event.x, event.y)
                self._frames.request()
                return
            if (
                self.last_arcball is None
                or event.inaxes is not self.ax
                or event.xdata is None
                or event.ydata is None
            ):
                return
            current = self.plot_to_sphere(event.xdata, event.ydata)
            delta = rotation_between(self.last_arcball, current)
            self.user_rotation = delta @ self.user_rotation
            self.last_arcball = current
            self._frames.request()

        def on_release(self, event) -> None:
            if self.press_event is None:
                return
            was_dragged = self.dragged
            drag_mode = self.drag_mode
            self.press_event = None
            self.last_arcball = None
            self.last_drag_pixel = None
            self.drag_mode = None
            self.dragged = False
            self.update_rotation_entries()
            self._frames.cancel()
            if was_dragged or drag_mode == "structure":
                self.redraw()
            if drag_mode == "structure":
                return
            if was_dragged or event.inaxes is not self.ax or not self.point_groups:
                return
            if event.x is None or event.y is None:
                return

            representatives = [group[0] for group in self.point_groups]
            screen_points = self.ax.transData.transform(
                np.array(
                    [pole_display_position(point) for point in representatives]
                )
            )
            distances = np.hypot(
                screen_points[:, 0] - event.x,
                screen_points[:, 1] - event.y,
            )
            index = int(np.argmin(distances))
            if distances[index] <= 14:
                self.selected_hkl = self.point_groups[index][0].hkl
                self.redraw()

    owns_window = parent is None
    if owns_window:
        root = tk.Tk()
    else:
        root = ttk.Frame(parent)
        root.pack(fill="both", expand=True)
    app = PoleFigureApp(root, initial_path)
    if owns_window:
        root.mainloop()
    return app


def self_test(path: str) -> None:
    cif = parse_cif(path)
    crystal = Crystal.from_cif(cif)
    center_hkl = (0, 1, 0)
    orientation = base_orientation(crystal, center_hkl)
    reflections = available_reflections(
        crystal,
        d_lower=1.0,
        d_upper=6.0,
        wavelength=1.5406,
    )
    points = project_reflections(
        crystal, reflections, orientation, "Стереографическая"
    )
    groups = group_coincident_poles(points)
    if len(points) <= 1 or len(groups) <= 1:
        raise RuntimeError("Не построен полный набор разрешённых отражений")
    if not any(
        point.chi < 1e-7
        and np.cross(point.hkl, center_hkl).tolist() == [0, 0, 0]
        for point in points
    ):
        raise RuntimeError("Центрирующий полюс не оказался в центре")
    determinant = float(np.linalg.det(orientation))
    if not math.isclose(determinant, 1.0, abs_tol=1e-8):
        raise RuntimeError("Некорректная матрица ориентации")
    display_atoms = unit_cell_display_atoms(crystal)
    bonds = unit_cell_bonds(display_atoms)
    if not crystal.atoms or not display_atoms:
        raise RuntimeError("Не удалось развернуть атомную структуру CIF")
    if not bonds:
        raise RuntimeError("Для трёхмерной структуры не найдены связи")
    print(f"Файл: {crystal.cif.source.name}")
    print(f"Формула: {crystal.formula}")
    print(f"Пространственная группа: {crystal.space_group}")
    print(
        f"Ячейка: a={crystal.a:.4f} Å, b={crystal.b:.4f} Å, "
        f"c={crystal.c:.4f} Å; α={crystal.alpha:.3f}°, "
        f"β={crystal.beta:.3f}°, γ={crystal.gamma:.3f}°"
    )
    print(f"Операций симметрии: {len(crystal.symmetry)}")
    print(
        f"Центр {format_hkl(center_hkl)}: "
        f"d={crystal.d_spacing(center_hkl):.6f} Å, "
        f"2θ(Cu Kα1)={crystal.two_theta(center_hkl, 1.5406):.6f}°"
    )
    print(f"Разрешённых отражений при 1 ≤ d ≤ 6 Å: {len(reflections)}")
    print(f"Различимых положений на верхней полусфере: {len(groups)}")
    print(
        f"Атомов в ячейке: {len(crystal.atoms)}; "
        f"отрисовываемых связей: {len(bonds)}"
    )
    print("Самопроверка завершена успешно.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Интерактивная теоретическая полюсная фигура по CIF"
    )
    parser.add_argument("cif", nargs="?", help="CIF, который нужно открыть")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить чтение CIF и расчёты без запуска окна",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        if not args.cif:
            parser.error("для --self-test требуется путь к CIF")
        self_test(args.cif)
        return
    _build_gui(args.cif)


if __name__ == "__main__":
    main()
