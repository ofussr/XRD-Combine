#!/usr/bin/env python3
"""
Расчёт таблицы порошковых рентгеновских отражений по CIF.

Рядом с программой должен находиться файл f0_WaasKirf.dat с коэффициентами
атомных факторов рассеяния.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    from .cif_lexer import CifLexError, tokenize_cif_text
    from .i18n import localised, tr, translate_text
    from .io.reflections import (
        read_scattering_factors as _read_scattering_factors,
        scattering_factor_path as _scattering_factor_path,
        write_reflection_csv,
    )
    from .models.data_errors import XRDDataError
    from .models.diffraction import Atom, ReflectionRow, Structure
    from .services.diffraction import (
        calculate_reflections as _calculate_reflections,
        format_hkl_family,
    )
except ImportError:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from xrd_workbench.cif_lexer import CifLexError, tokenize_cif_text
    from xrd_workbench.i18n import localised, tr, translate_text
    from xrd_workbench.io.reflections import (
        read_scattering_factors as _read_scattering_factors,
        scattering_factor_path as _scattering_factor_path,
        write_reflection_csv,
    )
    from xrd_workbench.models.data_errors import XRDDataError
    from xrd_workbench.models.diffraction import Atom, ReflectionRow, Structure
    from xrd_workbench.services.diffraction import (
        calculate_reflections as _calculate_reflections,
        format_hkl_family,
    )


RADIATIONS = [
    ("Cu Kα1", 1.54056, 1.0),
    ("Cu Kα2", 1.54443, 0.5),
]


class XRDCalculationError(Exception):
    pass


def _cif_tokens(text: str) -> list[str]:
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
        raise XRDCalculationError(
            localised(
                f"Could not parse CIF line {exc.line_number}: {reason}.",
                f"Impossible d’analyser la ligne CIF {exc.line_number} : {reason}.",
                f"Не удалось разобрать строку CIF {exc.line_number}: {reason}.",
            )
        ) from exc


def _parse_cif_items(text: str) -> tuple[str, dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    tokens = _cif_tokens(text)
    tags: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    block_name = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        low = token.lower()
        if low.startswith("data_"):
            if not block_name:
                block_name = token[5:] or "CIF"
            i += 1
        elif low == "loop_":
            i += 1
            headers: list[str] = []
            while i < len(tokens) and tokens[i].startswith("_"):
                headers.append(tokens[i].lower())
                i += 1
            if not headers:
                raise XRDCalculationError(
                    localised(
                        "A loop_ without column names was found in the CIF.",
                        "Un loop_ sans noms de colonnes a été trouvé dans le CIF.",
                        "В CIF найден loop_ без заголовков.",
                    )
                )
            values: list[str] = []
            while i < len(tokens):
                marker = tokens[i].lower()
                if marker == "loop_" or marker.startswith("data_") or marker.startswith("save_"):
                    break
                if tokens[i].startswith("_"):
                    break
                values.append(tokens[i])
                i += 1
            if len(values) % len(headers):
                raise XRDCalculationError(
                    localised(
                        "A CIF loop contains a value count that is not divisible "
                        "by its column count.",
                        "Le nombre de valeurs d’une boucle CIF n’est pas divisible "
                        "par son nombre de colonnes.",
                        "Число значений в одном из циклов CIF не кратно числу столбцов.",
                    )
                )
            rows = [
                values[j : j + len(headers)]
                for j in range(0, len(values), len(headers))
            ]
            loops.append((headers, rows))
        elif token.startswith("_"):
            if i + 1 >= len(tokens):
                raise XRDCalculationError(
                    localised(
                        f"CIF tag {token} has no value.",
                        f"La balise CIF {token} n’a pas de valeur.",
                        f"У тега {token} отсутствует значение.",
                    )
                )
            tags[low] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return block_name, tags, loops


def _number(value: str, default: float | None = None) -> float:
    value = value.strip()
    if value in {"?", "."}:
        if default is None:
            raise XRDCalculationError(
                localised(
                    "A required numeric CIF value is missing.",
                    "Une valeur numérique CIF obligatoire est absente.",
                    "В CIF отсутствует обязательное числовое значение.",
                )
            )
        return default
    value = re.sub(r"\(\d+\)$", "", value)
    try:
        return float(value)
    except ValueError as exc:
        raise XRDCalculationError(
            localised(
                f"Invalid number in CIF: {value!r}",
                f"Nombre incorrect dans le CIF : {value!r}",
                f"Некорректное число в CIF: {value!r}",
            )
        ) from exc


def _safe_fraction(expression: str, variables: dict[str, float]) -> float:
    try:
        root = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise XRDCalculationError(
            localised(
                f"Invalid symmetry operation: {expression}",
                f"Opération de symétrie incorrecte : {expression}",
                f"Некорректная операция симметрии: {expression}",
            )
        ) from exc

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise XRDCalculationError(
            localised(
                f"Unsupported expression in symmetry operation: {expression}",
                f"Expression non prise en charge dans l’opération de symétrie : {expression}",
                f"Недопустимое выражение в операции симметрии: {expression}",
            )
        )

    return evaluate(root)


def _apply_symmetry(operation: str, atom: Atom) -> tuple[float, float, float]:
    parts = [part.strip().lower() for part in operation.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise XRDCalculationError(
            localised(
                f"A symmetry operation must have three coordinates: {operation}",
                f"Une opération de symétrie doit avoir trois coordonnées : {operation}",
                f"Операция симметрии должна иметь три координаты: {operation}",
            )
        )
    variables = {"x": atom.x, "y": atom.y, "z": atom.z}
    result = tuple(_safe_fraction(part, variables) % 1.0 for part in parts)
    return result  # type: ignore[return-value]


def _element_from_label(label: str) -> str:
    match = re.match(r"([A-Z][a-z]?)", label.strip(), re.IGNORECASE)
    if not match:
        raise XRDCalculationError(
            localised(
                f"Could not determine the chemical element: {label!r}",
                f"Impossible de déterminer l’élément chimique : {label!r}",
                f"Не удалось определить химический элемент: {label!r}",
            )
        )
    raw = match.group(1)
    return raw[0].upper() + raw[1:].lower()


def read_structure(path: str | os.PathLike[str]) -> Structure:
    # Все страницы приложения должны получать структуру из одного разбора CIF.
    # Локальный импорт не создаёт цикл при загрузке ``cif_document``.
    try:
        from .cif_document import load_cif_document
    except ImportError:  # pragma: no cover - прямой запуск модуля
        from cif_document import load_cif_document

    try:
        return load_cif_document(path).diffraction
    except XRDCalculationError:
        raise
    except Exception as exc:
        # Табличный расчёт исторически публикует собственный тип ошибки.
        # Не связываем исключение с замороженным объектом лексера: unittest
        # очищает traceback причин и не может изменять frozen dataclass.
        raise XRDCalculationError(str(exc)) from None


def _read_structure_legacy(path: str | os.PathLike[str]) -> Structure:
    """Прежний независимый разборщик, оставленный только для сверочных тестов."""
    cif_path = Path(path)
    try:
        text = cif_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = cif_path.read_text(encoding="latin-1")
    block_name, tags, loops = _parse_cif_items(text)

    required_cell = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ]
    missing = [tag for tag in required_cell if tag not in tags]
    if missing:
        joined = ", ".join(missing)
        raise XRDCalculationError(
            localised(
                f"Unit-cell parameters are missing from the CIF: {joined}",
                f"Des paramètres de maille sont absents du CIF : {joined}",
                f"В CIF отсутствуют параметры ячейки: {joined}",
            )
        )
    cell = tuple(_number(tags[tag]) for tag in required_cell)

    symmetry_headers = {
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    }
    symmetry_operations: list[str] = []
    for headers, rows in loops:
        for sym_header in symmetry_headers.intersection(headers):
            col = headers.index(sym_header)
            symmetry_operations.extend(row[col] for row in rows)
    if not symmetry_operations:
        group_name = tags.get(
            "_space_group_name_h-m_alt",
            tags.get("_symmetry_space_group_name_h-m", ""),
        ).strip("'\"")
        group_key = group_name.replace(" ", "").lower()
        group_number = int(
            _number(
                tags.get(
                    "_space_group_it_number",
                    tags.get("_symmetry_int_tables_number", "1"),
                ),
                1.0,
            )
        )
        if group_number != 1 or group_key not in {"", "p1", "1", ".", "?"}:
            raise XRDCalculationError(
                localised(
                    "The CIF declares a space group but contains no explicit "
                    "symmetry-operation list. This version does not reconstruct "
                    "operations from the group number because silently calculating "
                    "as P1 would give incorrect intensities.",
                    "Le CIF déclare un groupe d’espace mais ne contient aucune liste "
                    "explicite d’opérations de symétrie. Cette version ne reconstruit "
                    "pas les opérations à partir du numéro du groupe, car un calcul "
                    "silencieux en P1 donnerait des intensités incorrectes.",
                    "В CIF указана пространственная группа, но нет явного списка операций "
                    "симметрии. Эта версия программы не восстанавливает операции только по "
                    "номеру группы, поскольку молчаливый расчёт как P1 дал бы неверные "
                    "интенсивности.",
                )
            )
        symmetry_operations = ["x,y,z"]

    atom_loop = None
    for headers, rows in loops:
        if {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }.issubset(headers):
            atom_loop = (headers, rows)
            break
    if atom_loop is None:
        raise XRDCalculationError(
            localised(
                "No fractional atom coordinates were found in the CIF.",
                "Aucune coordonnée atomique fractionnaire n’a été trouvée dans le CIF.",
                "В CIF не найдены дробные координаты атомов.",
            )
        )

    headers, rows = atom_loop
    ix = headers.index("_atom_site_fract_x")
    iy = headers.index("_atom_site_fract_y")
    iz = headers.index("_atom_site_fract_z")
    ielement = (
        headers.index("_atom_site_type_symbol")
        if "_atom_site_type_symbol" in headers
        else headers.index("_atom_site_label")
    )
    iocc = headers.index("_atom_site_occupancy") if "_atom_site_occupancy" in headers else None
    ib = headers.index("_atom_site_b_iso_or_equiv") if "_atom_site_b_iso_or_equiv" in headers else None
    iu = headers.index("_atom_site_u_iso_or_equiv") if "_atom_site_u_iso_or_equiv" in headers else None

    asymmetric_atoms: list[Atom] = []
    for row in rows:
        b_iso = _number(row[ib], 0.0) if ib is not None else 0.0
        if iu is not None and ib is None:
            b_iso = 8.0 * math.pi * math.pi * _number(row[iu], 0.0)
        asymmetric_atoms.append(
            Atom(
                _element_from_label(row[ielement]),
                _number(row[ix]) % 1.0,
                _number(row[iy]) % 1.0,
                _number(row[iz]) % 1.0,
                _number(row[iocc], 1.0) if iocc is not None else 1.0,
                b_iso,
            )
        )

    expanded: list[Atom] = []
    for atom in asymmetric_atoms:
        positions: list[tuple[float, float, float]] = []
        for operation in symmetry_operations:
            position = _apply_symmetry(operation, atom)
            if not any(
                all(abs(((a - b + 0.5) % 1.0) - 0.5) < 1e-6 for a, b in zip(position, old))
                for old in positions
            ):
                positions.append(position)
        expanded.extend(
            Atom(atom.element, x, y, z, atom.occupancy, atom.b_iso)
            for x, y, z in positions
        )

    name = tags.get("_chemical_formula_sum", block_name or cif_path.stem)
    return Structure(name, cell, expanded, symmetry_operations)


def _diffraction_error_text(exc: XRDDataError) -> str:
    element = exc.context.get("element", "")
    expression = exc.context.get("expression", "")
    messages = {
        "diffraction_no_radiation": localised(
            "No spectral line is selected.",
            "Aucune raie spectrale n’est sélectionnée.",
            "Не выбрана ни одна спектральная линия.",
        ),
        "diffraction_limits": localised(
            "The limits must satisfy 0 ≤ minimum < maximum < 180°.",
            "Les limites doivent respecter 0 ≤ minimum < maximum < 180°.",
            "Требуется 0 ≤ минимум < максимум < 180°.",
        ),
        "diffraction_radiation_positive": localised(
            "Wavelengths and relative weights must be positive.",
            "Les longueurs d’onde et les poids relatifs doivent être positifs.",
            "Длины волн и относительные веса должны быть положительными.",
        ),
        "diffraction_factors_empty": localised(
            "Could not read the atomic scattering factors.",
            "Impossible de lire les facteurs de diffusion atomique.",
            "Не удалось прочитать коэффициенты атомного рассеяния.",
        ),
        "diffraction_factors_file_missing": localised(
            "f0_WaasKirf.dat was not found. Place it beside the application.",
            "f0_WaasKirf.dat est introuvable. Placez-le à côté de l’application.",
            "Не найден f0_WaasKirf.dat. Положите его в одну папку с программой.",
        ),
        "diffraction_factor_missing": localised(
            f"No atomic scattering factors are available for {element}.",
            f"Aucun facteur de diffusion atomique n’est disponible pour {element}.",
            f"Нет коэффициентов атомного рассеяния для элемента {element}.",
        ),
        "diffraction_singular_metric": localised(
            "The metric matrix is singular.",
            "La matrice métrique est singulière.",
            "Матрица метрики вырождена.",
        ),
        "diffraction_symmetry": localised(
            f"Could not parse symmetry operation {expression!r}.",
            f"Impossible d’analyser l’opération de symétrie {expression!r}.",
            f"Не удалось разобрать операцию симметрии {expression!r}.",
        ),
    }
    return messages.get(exc.code, str(exc))


def load_scattering_factors(
    path: str | os.PathLike[str],
) -> dict[str, tuple[list[float], float, list[float]]]:
    try:
        return _read_scattering_factors(path)
    except XRDDataError as exc:
        raise XRDCalculationError(_diffraction_error_text(exc)) from exc


def calculate_reflections(
    structure: Structure,
    factors: dict[str, tuple[list[float], float, list[float]]],
    radiations: list[tuple[str, float, float]],
    min_two_theta: float = 5.0,
    max_two_theta: float = 120.0,
    min_intensity: float = 0.1,
) -> list[ReflectionRow]:
    try:
        return _calculate_reflections(
            structure,
            factors,
            radiations,
            min_two_theta,
            max_two_theta,
            min_intensity,
        )
    except XRDDataError as exc:
        raise XRDCalculationError(_diffraction_error_text(exc)) from exc


def _hkl_label(labels: set[tuple[int, int, int]]) -> str:
    return format_hkl_family(labels)

CSV_COLUMNS = [
    "hkl",
    "d, Å",
    "2θ, °",
    "линия",
    "λ, Å",
    "вес",
    "кратность",
    "Σ|F|^2",
    "Iотн, %",
]


def export_csv(path: str | os.PathLike[str], rows: Iterable[ReflectionRow]) -> None:
    write_reflection_csv(
        path,
        rows,
        [translate_text(column) for column in CSV_COLUMNS],
    )


def _data_path() -> Path:
    try:
        return _scattering_factor_path()
    except XRDDataError as exc:
        raise XRDCalculationError(_diffraction_error_text(exc)) from exc


def run_gui(initial_file: str | None = None) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class Application(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Таблица отражений из CIF")
            self.geometry("1180x720")
            self.minsize(900, 560)
            self.structure: Structure | None = None
            self.rows: list[ReflectionRow] = []
            self.path_var = tk.StringVar()
            self.min_angle = tk.StringVar(value="5")
            self.max_angle = tk.StringVar(value="120")
            self.min_intensity = tk.StringVar(value="0.1")
            self.status = tk.StringVar(value="Откройте CIF-файл.")
            self.line_vars: list[tuple[tk.BooleanVar, tk.StringVar, tk.StringVar, str]] = []
            self._build()
            if initial_file:
                self.after(100, lambda: self.open_file(initial_file))

        def _build(self) -> None:
            outer = ttk.Frame(self, padding=10)
            outer.pack(fill="both", expand=True)

            file_frame = ttk.Frame(outer)
            file_frame.pack(fill="x")
            ttk.Entry(file_frame, textvariable=self.path_var, state="readonly").pack(
                side="left", fill="x", expand=True
            )
            ttk.Button(file_frame, text=tr("text.open_cif"), command=self.open_dialog).pack(
                side="left", padx=(8, 0)
            )

            settings = ttk.LabelFrame(outer, text=tr("text.calculation"), padding=8)
            settings.pack(fill="x", pady=(10, 8))
            ttk.Label(settings, text=tr("text.two_theta_from")).grid(row=0, column=0, sticky="w")
            ttk.Entry(settings, textvariable=self.min_angle, width=8).grid(
                row=0, column=1, padx=(5, 12)
            )
            ttk.Label(settings, text=tr("text.to")).grid(row=0, column=2, sticky="w")
            ttk.Entry(settings, textvariable=self.max_angle, width=8).grid(
                row=0, column=3, padx=(5, 12)
            )
            ttk.Label(settings, text=tr("text.irel_2")).grid(row=0, column=4, sticky="w")
            ttk.Entry(settings, textvariable=self.min_intensity, width=8).grid(
                row=0, column=5, padx=(5, 18)
            )

            start_col = 6
            for offset, (name, wavelength, weight) in enumerate(RADIATIONS):
                enabled = tk.BooleanVar(value=True)
                wave_var = tk.StringVar(value=f"{wavelength:.5f}")
                weight_var = tk.StringVar(value=f"{weight:g}")
                self.line_vars.append((enabled, wave_var, weight_var, name))
                col = start_col + offset * 5
                ttk.Checkbutton(settings, text=name, variable=enabled).grid(
                    row=0, column=col, padx=(0, 4)
                )
                ttk.Label(settings, text="λ").grid(row=0, column=col + 1)
                ttk.Entry(settings, textvariable=wave_var, width=8).grid(
                    row=0, column=col + 2, padx=(3, 3)
                )
                ttk.Label(settings, text=tr("text.weight_2")).grid(row=0, column=col + 3)
                ttk.Entry(settings, textvariable=weight_var, width=5).grid(
                    row=0, column=col + 4, padx=(3, 10)
                )

            ttk.Button(settings, text=tr("text.calculate"), command=self.calculate).grid(
                row=0, column=start_col + 10, padx=(4, 0)
            )

            table_frame = ttk.Frame(outer)
            table_frame.pack(fill="both", expand=True)
            columns = ("hkl", "d", "tt", "line", "wave", "weight", "mult", "f2", "int")
            self.table = ttk.Treeview(table_frame, columns=columns, show="headings")
            headings = dict(zip(columns, CSV_COLUMNS))
            widths = {
                "hkl": 170,
                "d": 90,
                "tt": 90,
                "line": 85,
                "wave": 75,
                "weight": 60,
                "mult": 75,
                "f2": 100,
                "int": 90,
            }
            for column in columns:
                self.table.heading(
                    column,
                    text=headings[column],
                    command=lambda c=column: self.sort_table(c, False),
                )
                self.table.column(column, width=widths[column], anchor="center")
            yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
            xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
            self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
            self.table.grid(row=0, column=0, sticky="nsew")
            yscroll.grid(row=0, column=1, sticky="ns")
            xscroll.grid(row=1, column=0, sticky="ew")
            table_frame.rowconfigure(0, weight=1)
            table_frame.columnconfigure(0, weight=1)

            bottom = ttk.Frame(outer)
            bottom.pack(fill="x", pady=(8, 0))
            ttk.Label(bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
            ttk.Button(bottom, text=tr("text.save_csv"), command=self.save_csv).pack(side="right")

        def open_dialog(self) -> None:
            path = filedialog.askopenfilename(
                title="Открыть CIF",
                filetypes=[("Crystallographic Information File", "*.cif"), ("Все файлы", "*.*")],
            )
            if path:
                self.open_file(path)

        def open_file(self, path: str) -> None:
            try:
                self.structure = read_structure(path)
                self.path_var.set(path)
                self.calculate()
            except Exception as exc:
                messagebox.showerror("Ошибка чтения CIF", str(exc))

        def selected_radiations(self) -> list[tuple[str, float, float]]:
            selected = []
            for enabled, wave_var, weight_var, name in self.line_vars:
                if enabled.get():
                    selected.append((name, float(wave_var.get()), float(weight_var.get())))
            return selected

        def calculate(self) -> None:
            if self.structure is None:
                messagebox.showinfo("Нет структуры", "Сначала откройте CIF-файл.")
                return
            try:
                factors = load_scattering_factors(_data_path())
                self.rows = calculate_reflections(
                    self.structure,
                    factors,
                    self.selected_radiations(),
                    float(self.min_angle.get()),
                    float(self.max_angle.get()),
                    float(self.min_intensity.get()),
                )
            except Exception as exc:
                messagebox.showerror("Ошибка расчёта", str(exc))
                return
            self.table.delete(*self.table.get_children())
            for row in self.rows:
                self.table.insert(
                    "",
                    "end",
                    values=(
                        row.hkl,
                        f"{row.d:.6f}",
                        f"{row.two_theta:.5f}",
                        row.radiation,
                        f"{row.wavelength:.5f}",
                        f"{row.weight:.4g}",
                        row.multiplicity,
                        f"{row.f2_sum:.6g}",
                        f"{row.intensity:.4f}",
                    ),
                )
            self.status.set(
                f"{self.structure.name}: {len(self.structure.atoms)} атомов после симметрии; "
                f"строк в таблице: {len(self.rows)}. Интенсивности расчётные, порошковые."
            )

        def save_csv(self) -> None:
            if not self.rows:
                messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
                return
            default = Path(self.path_var.get()).stem + "_calculated_xrd.csv"
            path = filedialog.asksaveasfilename(
                title="Сохранить таблицу",
                defaultextension=".csv",
                initialfile=default,
                filetypes=[("CSV с разделителем «;»", "*.csv"), ("Все файлы", "*.*")],
            )
            if path:
                export_csv(path, self.rows)
                self.status.set(f"Таблица сохранена: {path}")

        def sort_table(self, column: str, reverse: bool) -> None:
            items = [(self.table.set(item, column), item) for item in self.table.get_children("")]

            def key(item: tuple[str, str]) -> object:
                try:
                    return float(item[0])
                except ValueError:
                    return item[0]

            items.sort(key=key, reverse=reverse)
            for index, (_, item) in enumerate(items):
                self.table.move(item, "", index)
            self.table.heading(
                column,
                command=lambda: self.sort_table(column, not reverse),
            )

    Application().mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Расчёт таблицы порошковых рентгеновских отражений по CIF."
    )
    parser.add_argument("cif", nargs="?", help="CIF-файл; без параметров открывается окно")
    parser.add_argument("--csv", help="сохранить таблицу без запуска окна")
    parser.add_argument("--min-angle", type=float, default=5.0)
    parser.add_argument("--max-angle", type=float, default=120.0)
    parser.add_argument("--min-intensity", type=float, default=0.1)
    args = parser.parse_args()

    if args.csv:
        if not args.cif:
            parser.error("для --csv требуется путь к CIF")
        structure = read_structure(args.cif)
        rows = calculate_reflections(
            structure,
            load_scattering_factors(_data_path()),
            RADIATIONS,
            args.min_angle,
            args.max_angle,
            args.min_intensity,
        )
        export_csv(args.csv, rows)
        print(f"Сохранено строк: {len(rows)}; файл: {args.csv}")
    else:
        run_gui(args.cif)


if __name__ == "__main__":
    main()
