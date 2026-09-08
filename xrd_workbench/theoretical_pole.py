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
import math
import os
import re
import sys
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
    from .io.reflections import read_scattering_factors, scattering_factor_path
    from .models.crystal import (
        Atom,
        CifData,
        CifLoop,
        Crystal,
        DisplayAtom,
        cif_number,
        direct_basis,
        expanded_atoms,
        parse_symmetry_operation,
        unit_cell_display_atoms,
    )
    from .models.data_errors import XRDDataError
    from .models.pole_figure import CalculatedPoleLayer, PolePoint, PoleReflection
    from .services.pole_figure import (
        align_to_z,
        available_reflections as _available_reflections,
        base_orientation,
        calculated_intensity_by_spacing as _calculated_intensity_by_spacing,
        euler_matrix,
        format_hkl,
        group_coincident_poles,
        in_plane_alignment,
        marker_sizes_by_d,
        matrix_to_euler,
        pole_display_orientation,
        pole_display_position,
        pole_plot_coordinates,
        pole_plot_to_sphere,
        project_reflections,
        projection_code,
        rotation_axis_angle,
        rotation_between,
        rotation_x,
        rotation_y,
        rotation_z,
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
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from xrd_workbench.io.reflections import (
        read_scattering_factors,
        scattering_factor_path,
    )
    from xrd_workbench.models.crystal import (
        Atom,
        CifData,
        CifLoop,
        Crystal,
        DisplayAtom,
        cif_number,
        direct_basis,
        expanded_atoms,
        parse_symmetry_operation,
        unit_cell_display_atoms,
    )
    from xrd_workbench.models.data_errors import XRDDataError
    from xrd_workbench.models.pole_figure import (
        CalculatedPoleLayer,
        PolePoint,
        PoleReflection,
    )
    from xrd_workbench.services.pole_figure import (
        align_to_z,
        available_reflections as _available_reflections,
        base_orientation,
        calculated_intensity_by_spacing as _calculated_intensity_by_spacing,
        euler_matrix,
        format_hkl,
        group_coincident_poles,
        in_plane_alignment,
        marker_sizes_by_d,
        matrix_to_euler,
        pole_display_orientation,
        pole_display_position,
        pole_plot_coordinates,
        pole_plot_to_sphere,
        project_reflections,
        projection_code,
        rotation_axis_angle,
        rotation_between,
        rotation_x,
        rotation_y,
        rotation_z,
    )


POLE_AZIMUTH_GRID_STEP_DEG = 10


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


Reflection = PoleReflection


def available_reflections(
    crystal: Crystal,
    d_lower: float,
    d_upper: float,
    wavelength: float,
) -> list[Reflection]:
    """Localized compatibility adapter for the GUI-independent pole service."""

    try:
        return _available_reflections(crystal, d_lower, d_upper, wavelength)
    except XRDDataError as exc:
        messages = {
            "pole_d_positive": localised(
                "The d limits must be positive.",
                "Les limites de d doivent être positives.",
                "Границы d должны быть положительными.",
            ),
            "pole_d_order": localised(
                "The lower d limit cannot exceed the upper limit.",
                "La limite inférieure de d ne peut pas dépasser la limite supérieure.",
                "Нижняя граница d не может быть больше верхней.",
            ),
            "pole_wavelength_positive": localised(
                "The wavelength must be positive.",
                "La longueur d’onde doit être positive.",
                "Длина волны должна быть положительной.",
            ),
            "pole_reflection_limit": localised(
                "The selected lower d limit requires testing more than two million "
                "reciprocal-lattice nodes. Increase the lower d limit.",
                "La limite inférieure de d choisie nécessite de tester plus de deux "
                "millions de nœuds du réseau réciproque. Augmentez cette limite.",
                "Выбранная нижняя граница d требует перебора более двух миллионов "
                "узлов. Увеличьте нижнюю границу d.",
            ),
        }
        raise ValueError(messages.get(exc.code, str(exc))) from exc


def calculated_intensity_by_spacing(
    diffraction_structure,
    radiations: list[tuple[str, float, float]],
) -> dict[float, float]:
    """Compatibility adapter that supplies bundled scattering-factor data."""

    factors = read_scattering_factors(scattering_factor_path())
    return _calculated_intensity_by_spacing(
        diffraction_structure, radiations, factors
    )


def unit_cell_bonds(atoms: Sequence[DisplayAtom]) -> list[tuple[int, int]]:
    try:
        from .structure_render import unit_cell_bonds as calculate_display_bonds
    except ImportError:  # pragma: no cover - direct module launch
        from xrd_workbench.structure_render import (
            unit_cell_bonds as calculate_display_bonds,
        )

    return calculate_display_bonds(list(atoms))


def draw_crystal_structure(
    axis,
    crystal: Crystal | None,
    orientation: np.ndarray,
    *,
    preview=False,
    show_basis=True,
):
    try:
        from .structure_render import render_structure
    except ImportError:  # pragma: no cover - direct module launch
        from xrd_workbench.structure_render import render_structure

    return render_structure(
        axis, crystal, orientation, preview=preview, show_basis=show_basis
    )


def _build_gui(
    initial_path: str | None,
    parent=None,
    auto_prompt: bool = True,
    on_open_cif=None,
    radiations_provider=None,
    on_add_overlay=None,
    on_remove_overlay=None,
    overlay_documents_provider=None,
):
    try:
        from .controls import CollapsibleSection, ScrollableControls, FrameScheduler
        from .structure_render import screen_drag_rotation
    except ImportError:
        from controls import CollapsibleSection, ScrollableControls, FrameScheduler
        from structure_render import screen_drag_rotation
    import tkinter as tk
    from tkinter import colorchooser, ttk

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
            self.overlay_layer: CalculatedPoleLayer | None = None
            self.on_add_overlay = on_add_overlay
            self.on_remove_overlay = on_remove_overlay
            self.overlay_documents_provider = overlay_documents_provider
            self.overlay_document_map = {}
            self._single_colour_mode: str | None = None

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
            self.primary_colour_var = tk.StringVar(value="#2d6da3")
            self.primary_opacity_var = tk.DoubleVar(value=100.0)
            self.primary_size_var = tk.DoubleVar(value=100.0)
            self.overlay_choice_var = tk.StringVar(value="")
            self.overlay_name_var = tk.StringVar(value="")
            self.overlay_colour_var = tk.StringVar(value="#d65f3c")
            self.overlay_opacity_var = tk.DoubleVar(value=70.0)
            self.overlay_size_var = tk.DoubleVar(value=100.0)
            self.overlay_hkl_vars = [
                tk.StringVar(value="0"),
                tk.StringVar(value="1"),
                tk.StringVar(value="0"),
            ]
            self.overlay_rotation_vars = [
                tk.StringVar(value="0.0") for _index in range(3)
            ]
            self.overlay_relative_rotation_vars = [
                tk.StringVar(value="0.0") for _index in range(3)
            ]
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
            self.refresh_overlay_choices()
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

            self.overlay_section = CollapsibleSection(
                controls, text="Вторая фаза поверх", padding=8
            )
            self.overlay_section.pack(fill="x", pady=(0, 8))
            ttk.Label(
                self.overlay_section,
                text="Фаза из данных проекта:",
            ).pack(anchor="w")
            self.overlay_combo = ttk.Combobox(
                self.overlay_section,
                state="readonly",
                textvariable=self.overlay_choice_var,
                postcommand=self.refresh_overlay_choices,
            )
            self.overlay_combo.pack(fill="x", pady=(3, 5))
            self.add_overlay_button = ttk.Button(
                self.overlay_section,
                text="Добавить поверх…",
                command=self.add_selected_overlay,
            )
            self.add_overlay_button.pack(fill="x")
            self.open_overlay_button = ttk.Button(
                self.overlay_section,
                text="Открыть CIF поверх…",
                command=self.ask_overlay_cif,
            )
            self.open_overlay_button.pack(fill="x", pady=(5, 0))
            ttk.Label(
                self.overlay_section,
                textvariable=self.overlay_name_var,
                wraplength=275,
                justify="left",
            ).pack(fill="x", pady=(7, 3))
            self.remove_overlay_button = ttk.Button(
                self.overlay_section,
                text="Убрать вторую фазу",
                command=self.remove_overlay,
                state="disabled",
            )
            self.remove_overlay_button.pack(fill="x")

            overlay_style = ttk.LabelFrame(
                self.overlay_section, text="Отображение второй фазы", padding=5
            )
            overlay_style.pack(fill="x", pady=(8, 0))
            overlay_colour_row = ttk.Frame(overlay_style)
            overlay_colour_row.pack(fill="x")
            ttk.Button(
                overlay_colour_row,
                text="Цвет…",
                command=lambda: self.choose_layer_colour(False),
            ).pack(side="left")
            ttk.Label(
                overlay_colour_row, textvariable=self.overlay_colour_var
            ).pack(side="left", padx=(6, 0))
            for label, variable, upper in (
                ("Прозрачность, %", self.overlay_opacity_var, 100),
                ("Размер точек, %", self.overlay_size_var, 300),
            ):
                row = ttk.Frame(overlay_style)
                row.pack(fill="x", pady=(5, 0))
                ttk.Label(row, text=label).pack(side="left")
                spin = ttk.Spinbox(
                    row,
                    from_=10,
                    to=upper,
                    increment=5,
                    width=7,
                    textvariable=variable,
                    command=self.redraw,
                )
                spin.pack(side="right")
                spin.bind("<Return>", lambda _event: self.redraw())

            overlay_center = ttk.LabelFrame(
                self.overlay_section, text="Центрирование второй фазы", padding=5
            )
            overlay_center.pack(fill="x", pady=(8, 0))
            overlay_hkl_row = ttk.Frame(overlay_center)
            overlay_hkl_row.pack(fill="x")
            for column, (label, variable) in enumerate(
                zip(("h", "k", "l"), self.overlay_hkl_vars)
            ):
                ttk.Label(overlay_hkl_row, text=label).grid(row=0, column=column)
                ttk.Entry(
                    overlay_hkl_row, width=8, textvariable=variable
                ).grid(row=1, column=column, padx=(0, 6))
            ttk.Button(
                overlay_center,
                text="Поместить полюс (hkl) в центр",
                command=self.apply_overlay_center,
            ).pack(fill="x", pady=(6, 0))

            for title, variables, command, button_text in (
                (
                    "Абсолютный поворот второй фазы",
                    self.overlay_rotation_vars,
                    self.apply_overlay_exact_rotation,
                    "Установить абсолютные углы",
                ),
                (
                    "Относительный поворот второй фазы",
                    self.overlay_relative_rotation_vars,
                    self.apply_overlay_relative_rotation,
                    "Повернуть относительно текущего",
                ),
            ):
                box = ttk.LabelFrame(self.overlay_section, text=title, padding=5)
                box.pack(fill="x", pady=(8, 0))
                row = ttk.Frame(box)
                row.pack(fill="x")
                for column, (axis, variable) in enumerate(
                    zip(("X, °", "Y, °", "Z, °"), variables)
                ):
                    ttk.Label(row, text=axis).grid(row=0, column=column)
                    entry = ttk.Entry(row, width=8, textvariable=variable)
                    entry.grid(row=1, column=column, padx=(0, 6))
                    entry.bind("<Return>", lambda _event, action=command: action())
                ttk.Button(box, text=button_text, command=command).pack(
                    fill="x", pady=(6, 0)
                )

            center_box = self.center_section = CollapsibleSection(
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
            self.uniform_colour_radio = ttk.Radiobutton(
                colour_box,
                text="Один цвет",
                value="uniform",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.uniform_colour_radio.pack(anchor="w")
            self.d_colour_radio = ttk.Radiobutton(
                colour_box,
                text="Цвет точек по d",
                value="d",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.d_colour_radio.pack(anchor="w")
            self.intensity_colour_radio = ttk.Radiobutton(
                colour_box,
                text="Цвет точек по расчётной интенсивности",
                value="intensity",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.intensity_colour_radio.pack(anchor="w")

            primary_style = ttk.LabelFrame(
                view_box, text="Отображение первой фазы", padding=5
            )
            primary_style.pack(fill="x", pady=(5, 0))
            primary_colour_row = ttk.Frame(primary_style)
            primary_colour_row.pack(fill="x")
            ttk.Button(
                primary_colour_row,
                text="Цвет…",
                command=lambda: self.choose_layer_colour(True),
            ).pack(side="left")
            ttk.Label(
                primary_colour_row, textvariable=self.primary_colour_var
            ).pack(side="left", padx=(6, 0))
            for label, variable, upper in (
                ("Прозрачность, %", self.primary_opacity_var, 100),
                ("Размер точек, %", self.primary_size_var, 300),
            ):
                row = ttk.Frame(primary_style)
                row.pack(fill="x", pady=(5, 0))
                ttk.Label(row, text=label).pack(side="left")
                spin = ttk.Spinbox(
                    row,
                    from_=10,
                    to=upper,
                    increment=5,
                    width=7,
                    textvariable=variable,
                    command=self.redraw,
                )
                spin.pack(side="right")
                spin.bind("<Return>", lambda _event: self.redraw())

            ttk.Checkbutton(view_box, text="Базисные векторы", variable=self.basis_visible,
                            command=self.redraw).pack(anchor="w")

            rotation_box = self.rotation_section = CollapsibleSection(
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

            relative_rotation_box = self.relative_rotation_section = CollapsibleSection(
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

            self.drag_help_label = ttk.Label(
                controls,
                text=(
                    "Перетаскивание внутри круга свободно вращает кристалл.\n"
                    "Щелчок по полюсу выводит его данные справа."
                ),
                wraplength=285,
                justify="left",
            )
            self.drag_help_label.pack(fill="x", pady=(2, 8))
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

        def refresh_overlay_choices(self) -> None:
            self.overlay_document_map = {}
            if self.overlay_layer is not None:
                self.overlay_combo.configure(values=(), state="disabled")
                self.add_overlay_button.configure(state="disabled")
                self.open_overlay_button.configure(state="disabled")
                return
            self.open_overlay_button.configure(state="normal")
            documents = (
                list(self.overlay_documents_provider())
                if self.overlay_documents_provider is not None
                else []
            )
            candidates = [
                document
                for document in documents
                if getattr(document, "payload", None) is not self.cif_document
                and getattr(document, "kind", None) in {"cif", "cell_phase"}
            ]
            counts: dict[str, int] = {}
            for document in candidates:
                counts[document.name] = counts.get(document.name, 0) + 1
            for document in candidates:
                label = document.name
                if counts[label] > 1:
                    label = f"{label} — {document.source.name}"
                self.overlay_document_map[label] = document
            values = tuple(self.overlay_document_map)
            self.overlay_combo.configure(
                values=values,
                state="readonly" if values else "disabled",
            )
            self.add_overlay_button.configure(
                state="normal" if values else "disabled"
            )
            if self.overlay_choice_var.get() not in self.overlay_document_map:
                self.overlay_choice_var.set(values[0] if values else "")

        def add_selected_overlay(self) -> None:
            document = self.overlay_document_map.get(self.overlay_choice_var.get())
            if document is None:
                self.refresh_overlay_choices()
                document = self.overlay_document_map.get(
                    self.overlay_choice_var.get()
                )
            if document is None:
                return
            if self.on_add_overlay is not None:
                self.on_add_overlay(document.uid)
            else:
                self.load_overlay_document(document.payload)

        def ask_overlay_cif(self) -> None:
            path = filedialog.askopenfilename(
                parent=self.root,
                title=localised(
                    "Open overlaid CIF",
                    "Ouvrir un CIF superposé",
                    "Открыть CIF поверх",
                ),
                filetypes=[("CIF", "*.cif"), ("All files", "*.*")],
            )
            if not path:
                return
            if self.on_add_overlay is not None:
                self.on_add_overlay(path)
                return
            try:
                try:
                    from .cif_document import load_cif_document
                except ImportError:  # pragma: no cover
                    from cif_document import load_cif_document
                self.load_overlay_document(load_cif_document(path))
            except Exception as exc:
                messagebox.showerror(
                    localised(
                        "Could not open CIF",
                        "Impossible d’ouvrir le CIF",
                        "Не удалось открыть CIF",
                    ),
                    str(exc),
                    parent=self.root,
                )

        def choose_layer_colour(self, primary: bool) -> None:
            variable = self.primary_colour_var if primary else self.overlay_colour_var
            _rgb, colour = colorchooser.askcolor(
                color=variable.get(),
                parent=self.root,
                title=localised(
                    "Phase colour",
                    "Couleur de la phase",
                    "Цвет фазы",
                ),
            )
            if colour:
                variable.set(colour)
                self.redraw()

        @staticmethod
        def _percentage(variable, default: float, lower: float, upper: float) -> float:
            try:
                value = float(variable.get())
            except (ValueError, tk.TclError):
                return default
            return min(upper, max(lower, value))

        def _set_multiphase_controls(self, enabled: bool) -> None:
            if enabled:
                if self._single_colour_mode is None:
                    self._single_colour_mode = self.color_mode_var.get()
                self.color_mode_var.set("uniform")
                self.d_colour_radio.configure(state="disabled")
                self.intensity_colour_radio.configure(state="disabled")
                self.drag_help_label.configure(
                    text=localised(
                        "Mouse rotation is disabled while two phases are overlaid.\n"
                        "Use the separate numerical rotations for each phase.",
                        "La rotation à la souris est désactivée lorsque deux phases "
                        "sont superposées.\nUtilisez les rotations numériques séparées.",
                        "При наложении двух фаз вращение мышью отключено.\n"
                        "Используйте отдельные числовые повороты каждой фазы.",
                    )
                )
                section_titles = (
                    (self.center_section, "Центрирование первой фазы"),
                    (self.rotation_section, "Абсолютный поворот первой фазы"),
                    (
                        self.relative_rotation_section,
                        "Относительный поворот первой фазы",
                    ),
                )
            else:
                self.d_colour_radio.configure(state="normal")
                cell_only = bool(
                    self.cif_document is not None
                    and getattr(self.cif_document, "is_cell_only", False)
                )
                self.intensity_colour_radio.configure(
                    state="disabled" if cell_only else "normal"
                )
                restored = self._single_colour_mode or "uniform"
                if restored == "intensity" and cell_only:
                    restored = "uniform"
                self.color_mode_var.set(restored)
                self._single_colour_mode = None
                self.drag_help_label.configure(
                    text=translate_text(
                        "Перетаскивание внутри круга свободно вращает кристалл.\n"
                        "Щелчок по полюсу выводит его данные справа."
                    )
                )
                section_titles = (
                    (self.center_section, "Центрирование по полюсу"),
                    (self.rotation_section, "Абсолютный поворот кристалла"),
                    (
                        self.relative_rotation_section,
                        "Относительный поворот кристалла",
                    ),
                )
            for section, title in section_titles:
                section.title_source = title
                section.localize_heading()

        def load_overlay_document(self, document) -> None:
            if self.cif_document is None:
                self.load_document(document)
                return
            if document is self.cif_document:
                self.status_var.set(
                    localised(
                        "The same phase cannot be overlaid with itself.",
                        "Une phase ne peut pas être superposée à elle-même.",
                        "Нельзя наложить фазу саму на себя.",
                    )
                )
                return
            try:
                d_lower, d_upper = self.get_d_range()
                wavelength = self.get_wavelength()
                centre = (0, 1, 0)
                layer = CalculatedPoleLayer(
                    document=document,
                    colour=self.overlay_colour_var.get(),
                    opacity_percent=self._percentage(
                        self.overlay_opacity_var, 70.0, 10.0, 100.0
                    ),
                    size_percent=self._percentage(
                        self.overlay_size_var, 100.0, 10.0, 300.0
                    ),
                    center_hkl=centre,
                    base_rotation=base_orientation(document.crystal, centre),
                    selected_hkl=centre,
                )
                layer.reflections = available_reflections(
                    document.crystal, d_lower, d_upper, wavelength
                )
            except ValueError as exc:
                messagebox.showerror(
                    localised(
                        "Could not add phase",
                        "Impossible d’ajouter la phase",
                        "Не удалось добавить фазу",
                    ),
                    str(exc),
                    parent=self.root,
                )
                return
            self.overlay_layer = layer
            self.overlay_name_var.set(document.name)
            for variable, value in zip(self.overlay_hkl_vars, centre):
                variable.set(str(value))
            self.update_overlay_rotation_entries()
            self.remove_overlay_button.configure(state="normal")
            self._set_multiphase_controls(True)
            self.refresh_overlay_choices()
            self.overlay_section.expand()
            self.redraw()

        def remove_overlay(self, *, notify: bool = True) -> None:
            layer = self.overlay_layer
            if layer is None:
                return
            self.overlay_layer = None
            self.overlay_name_var.set("")
            self.remove_overlay_button.configure(state="disabled")
            self._set_multiphase_controls(False)
            self.refresh_overlay_choices()
            self.redraw()
            if notify and self.on_remove_overlay is not None:
                self.on_remove_overlay(layer.document)

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
            if self.overlay_layer is not None:
                self.remove_overlay(notify=False)
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
            self.refresh_overlay_choices()

        def clear_document(self) -> None:
            self._frames.cancel()
            if self.overlay_layer is not None:
                self.remove_overlay(notify=False)
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
            self.refresh_overlay_choices()
            self.redraw()

        def remove_document(self, document) -> None:
            """Remove one assigned phase, promoting the overlay when necessary."""

            if self.overlay_layer is not None and self.overlay_layer.document is document:
                self.remove_overlay(notify=False)
                return
            if self.cif_document is not document:
                return
            if self.overlay_layer is None:
                self.clear_document()
                return
            promoted = self.overlay_layer
            self.overlay_layer = None
            self.overlay_name_var.set("")
            self.remove_overlay_button.configure(state="disabled")
            self._set_multiphase_controls(False)
            self.load_document(promoted.document)
            self.center_hkl = promoted.center_hkl
            self.base_rotation = promoted.base_rotation
            self.user_rotation = promoted.user_rotation
            self.reflections = promoted.reflections
            self.selected_hkl = promoted.selected_hkl
            self.primary_colour_var.set(promoted.colour)
            self.primary_opacity_var.set(promoted.opacity_percent)
            self.primary_size_var.set(promoted.size_percent)
            for variable, value in zip(
                (self.h_var, self.k_var, self.l_var), self.center_hkl
            ):
                variable.set(str(value))
            self.update_rotation_entries()
            self.refresh_overlay_choices()
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
            if self.overlay_layer is not None:
                self.overlay_layer.intensity_by_spacing = None
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
            if self.overlay_layer is not None:
                self.color_mode_var.set("uniform")
                self.redraw()
                return
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

        def _read_hkl_variables(self, variables) -> tuple[int, int, int]:
            values = []
            for variable in variables:
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

        def read_center(self) -> tuple[int, int, int]:
            return self._read_hkl_variables((self.h_var, self.k_var, self.l_var))

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
                overlay_reflections = (
                    available_reflections(
                        self.overlay_layer.crystal,
                        d_lower,
                        d_upper,
                        wavelength,
                    )
                    if self.overlay_layer is not None
                    else None
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
            if self.overlay_layer is not None and overlay_reflections is not None:
                self.overlay_layer.reflections = overlay_reflections
                self.overlay_layer.intensity_by_spacing = None
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

        def apply_overlay_center(self) -> None:
            layer = self.overlay_layer
            if layer is None:
                return
            try:
                centre = self._read_hkl_variables(self.overlay_hkl_vars)
                layer.base_rotation = base_orientation(layer.crystal, centre)
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
            layer.center_hkl = centre
            layer.user_rotation = np.eye(3)
            layer.selected_hkl = centre
            self.update_overlay_rotation_entries()
            self.redraw()

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

        def apply_overlay_exact_rotation(self) -> None:
            layer = self.overlay_layer
            if layer is None:
                return
            angles = self.read_rotation_angles(self.overlay_rotation_vars)
            if angles is None:
                return
            layer.user_rotation = euler_matrix(*angles)
            self.update_overlay_rotation_entries()
            self.redraw()

        def apply_overlay_relative_rotation(self) -> None:
            layer = self.overlay_layer
            if layer is None:
                return
            angles = self.read_rotation_angles(self.overlay_relative_rotation_vars)
            if angles is None:
                return
            layer.user_rotation = euler_matrix(*angles) @ layer.user_rotation
            self.update_overlay_rotation_entries()
            for variable in self.overlay_relative_rotation_vars:
                variable.set("0.0")
            self.redraw()

        def update_rotation_entries(self) -> None:
            angles = matrix_to_euler(self.user_rotation)
            for variable, angle in zip(self.rotation_vars, angles):
                if abs(angle) < 5e-10:
                    angle = 0.0
                variable.set(f"{angle:.3f}")

        def update_overlay_rotation_entries(self) -> None:
            layer = self.overlay_layer
            if layer is None:
                angles = (0.0, 0.0, 0.0)
            else:
                angles = matrix_to_euler(layer.user_rotation)
            for variable, angle in zip(self.overlay_rotation_vars, angles):
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
            marker_sizes *= self._percentage(
                self.primary_size_var, 100.0, 10.0, 300.0
            ) / 100.0
            primary_alpha = self._percentage(
                self.primary_opacity_var, 100.0, 10.0, 100.0
            ) / 100.0
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
                    alpha=primary_alpha,
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
                    color=self.primary_colour_var.get(),
                    alpha=primary_alpha,
                    edgecolors="none" if self.size_by_d_var.get() else "white",
                    linewidths=0.0 if self.size_by_d_var.get() else 0.8,
                    clip_on=False,
                    zorder=3,
                )

            if self.overlay_layer is not None:
                layer = self.overlay_layer
                layer.colour = self.overlay_colour_var.get()
                layer.opacity_percent = self._percentage(
                    self.overlay_opacity_var, 70.0, 10.0, 100.0
                )
                layer.size_percent = self._percentage(
                    self.overlay_size_var, 100.0, 10.0, 300.0
                )
                layer.points = project_reflections(
                    layer.crystal,
                    layer.reflections,
                    layer.orientation,
                    self.projection_var.get(),
                )
                layer.point_groups = group_coincident_poles(layer.points)
                overlay_representatives = [
                    group[0] for group in layer.point_groups
                ]
                overlay_positions = [
                    pole_display_position(point)
                    for point in overlay_representatives
                ]
                overlay_d = [
                    point.d_spacing for point in overlay_representatives
                ]
                overlay_sizes = (
                    marker_sizes_by_d(overlay_d)
                    if self.size_by_d_var.get()
                    else np.full(len(overlay_representatives), 58.0)
                )
                overlay_sizes *= layer.size_scale
                self.ax.scatter(
                    [position[0] for position in overlay_positions],
                    [position[1] for position in overlay_positions],
                    s=overlay_sizes,
                    color=layer.colour,
                    alpha=layer.opacity,
                    edgecolors="none" if self.size_by_d_var.get() else "white",
                    linewidths=0.0 if self.size_by_d_var.get() else 0.8,
                    clip_on=False,
                    zorder=3.1,
                )
                from matplotlib.lines import Line2D

                primary_legend_name = self.cif_document.name
                overlay_legend_name = layer.name
                if primary_legend_name == overlay_legend_name:
                    primary_legend_name = self.cif_document.source.name
                    overlay_legend_name = layer.document.source.name
                legend_handles = [
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        markersize=8,
                        markerfacecolor=self.primary_colour_var.get(),
                        markeredgecolor="white",
                        alpha=primary_alpha,
                        label=primary_legend_name,
                    ),
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        markersize=8,
                        markerfacecolor=layer.colour,
                        markeredgecolor="white",
                        alpha=layer.opacity,
                        label=overlay_legend_name,
                    ),
                ]
                self.ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    framealpha=0.9,
                    fontsize=9,
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
                if self.overlay_layer is not None:
                    for group in self.overlay_layer.point_groups:
                        point = group[0]
                        display_x, display_y = pole_display_position(point)
                        suffix = f" +{len(group) - 1}" if len(group) > 1 else ""
                        self.ax.annotate(
                            format_hkl(point.hkl) + suffix,
                            (display_x, display_y),
                            xytext=(7, -11),
                            textcoords="offset points",
                            fontsize=9,
                            color=self.overlay_layer.colour,
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
            if self.overlay_layer is not None:
                layer = self.overlay_layer
                status += localised(
                    f" Overlay {layer.name}: {len(layer.reflections)} reflections; "
                    f"{len(layer.point_groups)} distinct positions.",
                    f" Superposition {layer.name} : {len(layer.reflections)} réflexions ; "
                    f"{len(layer.point_groups)} positions distinctes.",
                    f" Наложение {layer.name}: отражений {len(layer.reflections)}; "
                    f"различимых положений {len(layer.point_groups)}.",
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
            if self.overlay_layer is not None:
                if event.inaxes is not self.ax:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                if math.hypot(event.xdata, event.ydata) > 1.05:
                    return
                self.press_event = (event.x, event.y)
                self.last_arcball = None
                self.last_drag_pixel = None
                self.drag_mode = "selection"
                self.dragged = False
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
