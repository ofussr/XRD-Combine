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
    from .io.cif import read_cif_data, tokenize_cif as _tokenize_cif
    from .i18n import (
        LocalizedStringVar,
        apply_language,
        bind_widget_text,
        choice_code,
        filedialog,
        localised,
        messagebox,
        tr,
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
        follow_orientation_change,
        format_hkl,
        group_coincident_poles,
        in_plane_alignment,
        marker_sizes_by_d,
        matrix_to_euler,
        place_label_boxes,
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
    from i18n import (
        LocalizedStringVar,
        apply_language,
        bind_widget_text,
        choice_code,
        filedialog,
        localised,
        messagebox,
        tr,
        translate_text,
    )
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from xrd_workbench.io.reflections import (
        read_scattering_factors,
        scattering_factor_path,
    )
    from xrd_workbench.io.cif import read_cif_data, tokenize_cif as _tokenize_cif
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
        follow_orientation_change,
        format_hkl,
        group_coincident_poles,
        in_plane_alignment,
        marker_sizes_by_d,
        matrix_to_euler,
        place_label_boxes,
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
        return _tokenize_cif(text)
    except XRDDataError as exc:
        if exc.code != "cif_lex":
            raise
        reason_code = exc.context.get("reason", "")
        reason = localised(
            "unclosed quotation mark"
            if reason_code == "unclosed_quote"
            else "unclosed multiline field",
            "guillemet non fermé"
            if reason_code == "unclosed_quote"
            else "champ multiligne non fermé",
            "незакрытая кавычка"
            if reason_code == "unclosed_quote"
            else "незакрытое многострочное поле",
        )
        line_number = exc.context.get("line_number", "")
        raise ValueError(
            localised(
                f"Could not parse CIF line {line_number}: {reason}.",
                f"Impossible d’analyser la ligne CIF {line_number} : {reason}.",
                f"Не удалось разобрать строку CIF {line_number}: {reason}.",
            )
        ) from exc


def parse_cif(path: str | os.PathLike[str]) -> CifData:
    try:
        return read_cif_data(path)
    except XRDDataError as exc:
        if exc.code == "cif_lex":
            source = Path(path).expanduser().resolve()
            # Re-run only the lightweight compatibility wrapper so the 2.x
            # interface receives its historical localized ValueError.
            tokenize_cif(source.read_text(encoding="utf-8-sig", errors="replace"))
            raise AssertionError("The malformed CIF unexpectedly tokenized")
        if exc.code == "cif_loop_no_columns":
            message = localised(
                "No column names follow loop_.",
                "Aucun nom de colonne ne suit loop_.",
                "После loop_ не найдены имена столбцов.",
            )
        elif exc.code == "cif_loop_width":
            values = exc.context.get("value_count", "")
            columns = exc.context.get("column_count", "")
            message = localised(
                f"The CIF loop value count is not divisible by the column count ({values} and {columns}).",
                f"Le nombre de valeurs de la boucle CIF n’est pas divisible par le nombre de colonnes ({values} et {columns}).",
                f"Число значений в цикле CIF не кратно числу столбцов ({values} и {columns}).",
            )
        elif exc.code == "cif_field_missing":
            field = exc.context.get("field", "")
            message = localised(
                f"CIF field {field} has no value.",
                f"Le champ CIF {field} n’a pas de valeur.",
                f"Для поля {field} отсутствует значение.",
            )
        else:
            raise
        raise ValueError(message) from exc


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
                    tr("text.calculated_pole_figure_from_cif")
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
            self.selected_layer_index = 0
            self._selection_menu = None
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

            self.path_var = LocalizedStringVar(
                translation_key="text.no_cif_loaded"
            )
            self.structure_var = tk.StringVar(value="")
            self.h_var = tk.StringVar(value="0")
            self.k_var = tk.StringVar(value="1")
            self.l_var = tk.StringVar(value="0")
            self.max_index_var = tk.IntVar(value=4)
            self.wavelength_var = tk.StringVar(value="1.5406")
            self.d_lower_var = tk.StringVar(value="1.0")
            self.d_upper_var = tk.StringVar(value="13")
            self.projection_var = tk.StringVar(
                value=tr("text.stereographic")
            )
            self.labels_var = tk.BooleanVar(value=False)
            self.label_leaders_var = tk.BooleanVar(value=False)
            self.coincident_outlines_var = tk.BooleanVar(value=False)
            self.angle_labels_var = tk.BooleanVar(value=True)
            self.show_structure_var = tk.BooleanVar(value=False)
            self.basis_visible = tk.BooleanVar(value=True)
            self._colorbar = None
            self.size_by_d_var = tk.BooleanVar(value=False)
            self.color_mode_var = tk.StringVar(value="uniform")
            self.intensity_by_spacing: dict[float, float] | None = None
            self.primary_colour_var = tk.StringVar(value="#2d6da3")
            self.primary_opacity_var = tk.DoubleVar(value=100.0)
            self.primary_size_var = tk.DoubleVar(value=100.0)
            self.global_size_var = tk.DoubleVar(value=100.0)
            self.global_size_label_var = tk.StringVar(value="100%")
            self.overlay_choice_var = tk.StringVar(value="")
            self.overlay_name_var = tk.StringVar(value="")
            self.overlay_colour_var = tk.StringVar(value="#d65f3c")
            self.overlay_opacity_var = tk.DoubleVar(value=70.0)
            self.overlay_size_var = tk.DoubleVar(value=100.0)
            self.joint_rotation_var = tk.BooleanVar(value=False)
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
            self._display_frames = FrameScheduler(self.root, self.redraw, interval=40)
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

            file_box = CollapsibleSection(controls, text=tr("text.cif_structure_2"), padding=8)
            file_box.pack(fill="x", pady=(0, 8))
            ttk.Button(file_box, text=tr("text.open_cif"), command=self.ask_cif).pack(
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
                controls, text=tr("text.overlaid_second_phase"), padding=8
            )
            ttk.Label(
                self.overlay_section,
                text=tr("text.phase_from_project_data"),
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
                text=tr("text.add_overlay"),
                command=self.add_selected_overlay,
            )
            self.add_overlay_button.pack(fill="x")
            self.open_overlay_button = ttk.Button(
                self.overlay_section,
                text=tr("text.open_cif_as_overlay"),
                command=self.ask_overlay_cif,
            )
            self.open_overlay_button.pack(fill="x", pady=(5, 0))

            self.overlay_settings = ttk.Frame(self.overlay_section)
            ttk.Label(
                self.overlay_settings,
                textvariable=self.overlay_name_var,
                wraplength=275,
                justify="left",
            ).pack(fill="x", pady=(7, 3))
            self.remove_overlay_button = ttk.Button(
                self.overlay_settings,
                text=tr("text.remove_second_phase"),
                command=self.remove_overlay,
                state="disabled",
            )
            self.remove_overlay_button.pack(fill="x")
            self.joint_rotation_check = ttk.Checkbutton(
                self.overlay_settings,
                variable=self.joint_rotation_var,
                command=self.joint_rotation_changed,
            )
            bind_widget_text(
                self.joint_rotation_check,
                "pole.rotate_phases_together",
            )
            self.joint_rotation_check.pack(anchor="w", pady=(7, 0))

            overlay_style = ttk.LabelFrame(
                self.overlay_settings, text=tr("text.second_phase_display"), padding=5
            )
            overlay_style.pack(fill="x", pady=(8, 0))
            overlay_colour_row = ttk.Frame(overlay_style)
            overlay_colour_row.pack(fill="x")
            ttk.Button(
                overlay_colour_row,
                text=tr("text.colour_2"),
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
                self.overlay_settings, text=tr("text.second_phase_centring"), padding=5
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
                text=tr("text.place_pole_h_k_l_at_centre"),
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
                box = ttk.LabelFrame(self.overlay_settings, text=title, padding=5)
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

            overlay_align = CollapsibleSection(
                self.overlay_settings,
                text=tr("text.align_selected_pole_of_second_phase"),
                padding=5,
            )
            overlay_align.pack(fill="x", pady=(8, 0))
            for column, (label, target) in enumerate(
                (("+X", 270.0), ("−X", 90.0), ("+Y", 0.0), ("−Y", 180.0))
            ):
                ttk.Button(
                    overlay_align,
                    text=label,
                    command=lambda angle=target: self.align_selected_pole(
                        angle, layer_index=1
                    ),
                ).grid(
                    row=0,
                    column=column,
                    sticky="ew",
                    padx=(0 if column == 0 else 3, 0),
                )
                overlay_align.columnconfigure(column, weight=1)
            self.overlay_settings.pack(fill="x")
            self.overlay_settings.pack_forget()

            center_box = self.center_section = CollapsibleSection(
                controls, text=tr("text.pole_centring"), padding=8
            )
            center_box.pack(fill="x", pady=(0, 8))
            self.center_prompt = ttk.Label(center_box, text=tr("text.select_an_allowed_pole"))
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
                text=tr("text.place_pole_h_k_l_at_centre"),
                command=self.apply_center,
            ).pack(fill="x", pady=(7, 0))

            list_row = ttk.Frame(center_box)
            list_row.pack(fill="x", pady=(8, 0))
            ttk.Label(list_row, text=tr("text.maximum_list_index")).grid(row=0, column=0)
            ttk.Spinbox(
                list_row,
                from_=1,
                to=12,
                width=4,
                textvariable=self.max_index_var,
                command=self.refresh_center_list,
            ).grid(row=0, column=1, padx=(4, 0))

            range_box = CollapsibleSection(
                controls, text=tr("text.displayed_reflections"), padding=8
            )
            range_box.pack(fill="x", pady=(0, 8))
            range_row = ttk.Frame(range_box)
            range_row.pack(fill="x")
            ttk.Label(range_row, text=tr("text.d_from_angstrom")).grid(row=0, column=0)
            ttk.Label(range_row, text=tr("text.d_to_angstrom")).grid(row=0, column=1)
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
                text=tr("text.plot_all_allowed_reflections"),
                command=self.rebuild_reflections,
            )
            self.range_button.pack(fill="x", pady=(7, 0))

            view_box = CollapsibleSection(controls, text=tr("text.display"), padding=8)
            view_box.pack(fill="x", pady=(0, 8))
            ttk.Label(view_box, text=tr("text.projection")).pack(anchor="w")
            projection_combo = ttk.Combobox(
                view_box,
                state="readonly",
                textvariable=self.projection_var,
                values=("Стереографическая", "Равноплощадная"),
            )
            projection_combo.pack(fill="x", pady=(3, 5))
            projection_combo.bind("<<ComboboxSelected>>", lambda _event: self.redraw())
            self.labels_check = ttk.Checkbutton(
                view_box,
                text=tr("text.label_poles"),
                variable=self.labels_var,
                command=self.labels_visibility_changed,
            )
            self.labels_check.pack(anchor="w")
            self.label_leaders_check = ttk.Checkbutton(
                view_box,
                variable=self.label_leaders_var,
                command=self.redraw,
                state="disabled",
            )
            bind_widget_text(
                self.label_leaders_check,
                "pole.show_hkl_label_lines",
            )
            self.label_leaders_check.pack(anchor="w")
            self.coincident_outlines_check = ttk.Checkbutton(
                view_box,
                variable=self.coincident_outlines_var,
                command=self.redraw,
            )
            bind_widget_text(
                self.coincident_outlines_check,
                "pole.outline_coincident_poles",
            )
            self.coincident_outlines_check.pack(anchor="w")
            ttk.Checkbutton(
                view_box,
                text=tr("text.show_angle_labels"),
                variable=self.angle_labels_var,
                command=self.redraw,
            ).pack(anchor="w")
            self.show_structure_check = ttk.Checkbutton(
                view_box,
                text=tr("text.show_structure_alongside"),
                variable=self.show_structure_var,
                command=self.structure_visibility_changed,
            )
            self.show_structure_check.pack(anchor="w")
            self.basis_check = ttk.Checkbutton(
                view_box,
                text=tr("text.basis_vectors"),
                variable=self.basis_visible,
                command=self.redraw,
                state="disabled",
            )
            self.basis_check.pack(anchor="w")
            ttk.Checkbutton(
                view_box,
                text=tr("text.point_size_by_d"),
                variable=self.size_by_d_var,
                command=self.redraw,
            ).pack(anchor="w")
            colour_box = ttk.LabelFrame(view_box, text=tr("text.point_colour"), padding=5)
            colour_box.pack(fill="x", pady=(5, 0))
            self.uniform_colour_radio = ttk.Radiobutton(
                colour_box,
                text=tr("text.uniform_colour"),
                value="uniform",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.uniform_colour_radio.pack(anchor="w")
            self.d_colour_radio = ttk.Radiobutton(
                colour_box,
                text=tr("text.point_colour_by_d"),
                value="d",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.d_colour_radio.pack(anchor="w")
            self.intensity_colour_radio = ttk.Radiobutton(
                colour_box,
                text=tr("text.point_colour_by_calculated_intensity"),
                value="intensity",
                variable=self.color_mode_var,
                command=self.change_colour_mode,
            )
            self.intensity_colour_radio.pack(anchor="w")

            primary_style = ttk.LabelFrame(
                view_box, text=tr("text.first_phase_display"), padding=5
            )
            primary_style.pack(fill="x", pady=(5, 0))
            primary_colour_row = ttk.Frame(primary_style)
            primary_colour_row.pack(fill="x")
            ttk.Button(
                primary_colour_row,
                text=tr("text.colour_2"),
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

            global_size_row = ttk.Frame(view_box)
            global_size_row.pack(fill="x", pady=(7, 0))
            ttk.Label(global_size_row, text=tr("text.overall_point_scale")).pack(
                anchor="w"
            )
            global_size_scale_row = ttk.Frame(global_size_row)
            global_size_scale_row.pack(fill="x", pady=(2, 0))
            ttk.Scale(
                global_size_scale_row,
                from_=25,
                to=300,
                orient="horizontal",
                variable=self.global_size_var,
                command=self.global_point_scale_changed,
            ).pack(side="left", fill="x", expand=True)
            ttk.Label(
                global_size_scale_row,
                textvariable=self.global_size_label_var,
                width=6,
                anchor="e",
            ).pack(side="right", padx=(5, 0))

            rotation_box = self.rotation_section = CollapsibleSection(
                controls, text=tr("text.absolute_crystal_rotation"), padding=8
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
                text=tr("text.set_absolute_angles"),
                command=self.apply_exact_rotation,
            ).pack(fill="x", pady=(7, 4))
            ttk.Button(
                rotation_box,
                text=tr("text.restore_centred_pole"),
                command=self.reset_rotation,
            ).pack(fill="x")

            relative_rotation_box = self.relative_rotation_section = CollapsibleSection(
                controls, text=tr("text.relative_crystal_rotation"), padding=8
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
                text=tr("text.rotate_relative_to_current"),
                command=self.apply_relative_rotation,
            ).pack(fill="x", pady=(7, 0))

            align_box = CollapsibleSection(
                controls, text=tr("text.align_selected_pole"), padding=8
            )
            align_box.pack(fill="x", pady=(0, 8))
            for column, (label, target) in enumerate(
                (("+X", 270.0), ("−X", 90.0), ("+Y", 0.0), ("−Y", 180.0))
            ):
                ttk.Button(
                    align_box,
                    text=label,
                    command=lambda angle=target: self.align_selected_pole(
                        angle, layer_index=0
                    ),
                ).grid(
                    row=0,
                    column=column,
                    sticky="ew",
                    padx=(0 if column == 0 else 3, 0),
                )
                align_box.columnconfigure(column, weight=1)

            self.overlay_section.pack(fill="x", pady=(0, 8))

            self.drag_help_label = ttk.Label(
                controls,
                text=(
                    tr("text.dragging_inside_the_circle_freely_rotates_the_crystal_clicking_a_pole_shows_its_data_on_the_right")
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

            ttk.Label(information, text=tr("text.selected_pole")).pack(anchor="w")
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
                if getattr(document, "kind", None) in {"cif", "cell_phase"}
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
                    tr("text.could_not_open_cif"),
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

        def global_point_scale_changed(self, value) -> None:
            try:
                percentage = min(300.0, max(25.0, float(value)))
            except (TypeError, ValueError, tk.TclError):
                percentage = 100.0
            self.global_size_label_var.set(f"{percentage:.0f}%")
            if hasattr(self, "_display_frames"):
                self._display_frames.request()

        def labels_visibility_changed(self) -> None:
            self.label_leaders_check.configure(
                state="normal" if self.labels_var.get() else "disabled"
            )
            self.redraw()

        def _update_basis_control_state(self) -> None:
            enabled = self.show_structure_var.get() and self.crystal is not None
            self.basis_check.configure(state="normal" if enabled else "disabled")

        def structure_visibility_changed(self) -> None:
            self._update_basis_control_state()
            self.redraw()

        def _set_overlay_settings_visible(self, visible: bool) -> None:
            # Expanding first makes the conditional child independent of the
            # section's saved collapsed layout.
            self.overlay_section.expand()
            if visible:
                if not self.overlay_settings.winfo_manager():
                    self.overlay_settings.pack(fill="x")
            else:
                self.overlay_settings.pack_forget()
            self.overlay_section.after_idle(self.overlay_section._notify_layout)

        def _joint_rotation_enabled(self) -> bool:
            return bool(
                self.overlay_layer is not None
                and self.overlay_layer.coupled_to_primary
            )

        def _update_drag_help(self) -> None:
            if self.overlay_layer is None:
                key = (
                    "text.dragging_inside_the_circle_freely_rotates_the_crystal_"
                    "clicking_a_pole_shows_its_data_on_the_right"
                )
            elif self._joint_rotation_enabled():
                key = "pole.mouse_rotation_moves_both_phases"
            else:
                key = (
                    "text.mouse_rotation_is_disabled_while_two_phases_are_"
                    "overlaid_use_the_separate_numerical_rotations_for_each_phase"
                )
            self.drag_help_label.configure(text=tr(key))

        def joint_rotation_changed(self) -> None:
            if self.overlay_layer is not None:
                self.overlay_layer.coupled_to_primary = bool(
                    self.joint_rotation_var.get()
                )
            self._update_drag_help()

        def _set_primary_orientation(
            self,
            *,
            base_rotation: np.ndarray | None = None,
            user_rotation: np.ndarray | None = None,
            update_overlay_entries: bool = True,
        ) -> None:
            before = self.user_rotation @ self.base_rotation
            if base_rotation is not None:
                self.base_rotation = np.asarray(base_rotation, dtype=float)
            if user_rotation is not None:
                self.user_rotation = np.asarray(user_rotation, dtype=float)
            after = self.user_rotation @ self.base_rotation
            if self._joint_rotation_enabled():
                layer = self.overlay_layer
                layer.user_rotation = follow_orientation_change(
                    layer.user_rotation,
                    before,
                    after,
                )
                if update_overlay_entries:
                    self.update_overlay_rotation_entries()

        def _set_multiphase_controls(self, enabled: bool) -> None:
            if enabled:
                if self._single_colour_mode is None:
                    self._single_colour_mode = self.color_mode_var.get()
                self.color_mode_var.set("uniform")
                self.d_colour_radio.configure(state="disabled")
                self.intensity_colour_radio.configure(state="disabled")
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
            self._update_drag_help()

        def load_overlay_document(self, document) -> None:
            if self.cif_document is None:
                self.load_document(document)
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
                    coupled_to_primary=False,
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
            self.joint_rotation_var.set(False)
            self.overlay_name_var.set(document.name)
            for variable, value in zip(self.overlay_hkl_vars, centre):
                variable.set(str(value))
            self.update_overlay_rotation_entries()
            self.remove_overlay_button.configure(state="normal")
            self._set_overlay_settings_visible(True)
            self._set_multiphase_controls(True)
            self.refresh_overlay_choices()
            self.redraw()

        def remove_overlay(self, *, notify: bool = True) -> None:
            layer = self.overlay_layer
            if layer is None:
                return
            self.overlay_layer = None
            self.selected_layer_index = 0
            self.joint_rotation_var.set(False)
            self.overlay_name_var.set("")
            self.remove_overlay_button.configure(state="disabled")
            self._set_overlay_settings_visible(False)
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
                    tr("text.could_not_open_cif"),
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
            self._display_frames.cancel()
            if self.overlay_layer is not None:
                self.remove_overlay(notify=False)
            self.reflections = []
            self.points = []
            self.point_groups = []
            self.selected_hkl = None
            self.selected_layer_index = 0
            self.intensity_by_spacing = None
            self.cif_document = document
            crystal = document.crystal
            self.crystal = crystal
            self._update_basis_control_state()
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
                    tr("text.select_a_pole_not_systematically_forbidden")
                    if cell_only
                    else tr("text.select_an_allowed_pole")
                )
            )
            self.range_button.configure(
                text=(
                    tr("text.plot_reflections_not_systematically_forbidden")
                    if cell_only
                    else tr("text.plot_all_allowed_reflections")
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
            self._display_frames.cancel()
            if self.overlay_layer is not None:
                self.remove_overlay(notify=False)
            self.cif_document = None
            self.crystal = None
            self._update_basis_control_state()
            self.reflections = []
            self.points = []
            self.point_groups = []
            self.selected_hkl = None
            self.selected_layer_index = 0
            self.intensity_by_spacing = None
            self.path_var.set_key("text.no_cif_loaded")
            self.structure_var.set("")
            self.center_combo.configure(values=())
            self.center_combo.set("")
            self.intensity_colour_radio.configure(state="normal")
            self.center_prompt.configure(text=tr("text.select_an_allowed_pole"))
            self.range_button.configure(
                text=tr("text.plot_all_allowed_reflections")
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
                if self.cif_document is document:
                    self.remove_overlay(notify=False)
                    self.clear_document()
                    return
                self.remove_overlay(notify=False)
                return
            if self.cif_document is not document:
                return
            if self.overlay_layer is None:
                self.clear_document()
                return
            promoted = self.overlay_layer
            self.overlay_layer = None
            self.joint_rotation_var.set(False)
            self.overlay_name_var.set("")
            self.remove_overlay_button.configure(state="disabled")
            self._set_overlay_settings_visible(False)
            self._set_multiphase_controls(False)
            self.load_document(promoted.document)
            self.center_hkl = promoted.center_hkl
            self.base_rotation = promoted.base_rotation
            self.user_rotation = promoted.user_rotation
            self.reflections = promoted.reflections
            self.selected_hkl = promoted.selected_hkl
            self.selected_layer_index = 0
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
            return self._ensure_layer_intensities(0, show_errors=show_errors)

        def _ensure_layer_intensities(
            self,
            layer_index: int,
            *,
            show_errors: bool = True,
        ) -> bool:
            if layer_index == 0:
                document = self.cif_document
                cached = self.intensity_by_spacing
            elif layer_index == 1 and self.overlay_layer is not None:
                document = self.overlay_layer.document
                cached = self.overlay_layer.intensity_by_spacing
            else:
                return False
            if document is None or getattr(document, "is_cell_only", False):
                return False
            if cached is not None:
                return True
            try:
                calculated = calculated_intensity_by_spacing(
                    document.diffraction,
                    self.get_radiations(),
                )
            except Exception as exc:
                calculated = {}
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
            if layer_index == 0:
                self.intensity_by_spacing = calculated
            elif self.overlay_layer is not None:
                self.overlay_layer.intensity_by_spacing = calculated
            return bool(calculated)

        def change_colour_mode(self) -> None:
            if self.overlay_layer is not None:
                self.color_mode_var.set("uniform")
                self.redraw()
                return
            if self.color_mode_var.get() == "intensity" and not self._ensure_intensities():
                self.color_mode_var.set("uniform")
            self.redraw()

        def _point_intensity(
            self,
            point: PolePoint,
            layer_index: int = 0,
        ) -> float | None:
            if layer_index == 0:
                intensity_by_spacing = self.intensity_by_spacing
            elif layer_index == 1 and self.overlay_layer is not None:
                intensity_by_spacing = self.overlay_layer.intensity_by_spacing
            else:
                intensity_by_spacing = None
            if intensity_by_spacing is None:
                return None
            return intensity_by_spacing.get(
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
                    tr("text.invalid_reflection_range"),
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
            self.selected_layer_index = 0
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
            self._set_primary_orientation(
                base_rotation=base,
                user_rotation=np.eye(3),
            )
            self.selected_hkl = center_hkl
            self.selected_layer_index = 0
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
            self.selected_layer_index = 1
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
                    tr("text.invalid_angle"),
                    tr("text.x_y_and_z_angles_must_be_numeric"),
                    parent=self.root,
                )
                return None

        def apply_exact_rotation(self) -> None:
            angles = self.read_rotation_angles(self.rotation_vars)
            if angles is None:
                return
            self._set_primary_orientation(user_rotation=euler_matrix(*angles))
            self.update_rotation_entries()
            self.redraw()

        def apply_relative_rotation(self) -> None:
            angles = self.read_rotation_angles(self.relative_rotation_vars)
            if angles is None:
                return
            self._set_primary_orientation(
                user_rotation=euler_matrix(*angles) @ self.user_rotation
            )
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
            self._set_primary_orientation(user_rotation=np.eye(3))
            self.selected_hkl = self.center_hkl
            self.selected_layer_index = 0
            self.update_rotation_entries()
            self.redraw()

        def align_selected_pole(
            self,
            target_phi: float,
            *,
            layer_index: int = 0,
        ) -> None:
            if layer_index == 0:
                selected_hkl = self.selected_hkl
                point_groups = self.point_groups
            elif layer_index == 1 and self.overlay_layer is not None:
                selected_hkl = self.overlay_layer.selected_hkl
                point_groups = self.overlay_layer.point_groups
            else:
                selected_hkl = None
                point_groups = ()
            point = None
            if selected_hkl is not None:
                for group in point_groups:
                    for candidate in group:
                        if candidate.hkl == selected_hkl:
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
            alignment = in_plane_alignment(point.phi, target_phi)
            if layer_index == 0:
                self._set_primary_orientation(
                    user_rotation=alignment @ self.user_rotation
                )
                self.update_rotation_entries()
            else:
                self.overlay_layer.user_rotation = (
                    alignment @ self.overlay_layer.user_rotation
                )
                self.update_overlay_rotation_entries()
            self.selected_layer_index = layer_index
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
                if self.angle_labels_var.get():
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

        def _layer_context(self, layer_index: int):
            if layer_index == 0 and self.crystal is not None:
                return (
                    self.cif_document,
                    self.crystal,
                    self.center_hkl,
                    self.user_rotation @ self.base_rotation,
                    self.point_groups,
                    self.selected_hkl,
                )
            if layer_index == 1 and self.overlay_layer is not None:
                layer = self.overlay_layer
                return (
                    layer.document,
                    layer.crystal,
                    layer.center_hkl,
                    layer.orientation,
                    layer.point_groups,
                    layer.selected_hkl,
                )
            return None

        def _selected_group(self, layer_index: int):
            context = self._layer_context(layer_index)
            if context is None:
                return None
            _document, crystal, _centre, orientation, groups, selected_hkl = context
            if selected_hkl is None:
                return None
            target = orientation @ crystal.reciprocal_vector(selected_hkl)
            target /= np.linalg.norm(target)
            if target[2] < -1e-10:
                target = -target
            elif abs(target[2]) <= 1e-10:
                target[2] = 0.0
            for group in groups:
                if float(np.dot(group[0].direction, target)) > 1.0 - 1e-8:
                    return group
            return None

        def _draw_coincident_group_markers(
            self,
            groups: Sequence[Sequence[PolePoint]],
            marker_sizes: Sequence[float],
        ) -> None:
            indices = [index for index, group in enumerate(groups) if len(group) > 1]
            if not indices:
                return
            positions = [pole_display_position(groups[index][0]) for index in indices]
            ring_sizes = [
                (math.sqrt(float(marker_sizes[index])) + 4.0) ** 2
                for index in indices
            ]
            self.ax.scatter(
                [position[0] for position in positions],
                [position[1] for position in positions],
                s=ring_sizes,
                facecolors="none",
                edgecolors="#555555",
                linewidths=0.75,
                clip_on=False,
                zorder=3.3,
            )

        def _draw_pole_labels(
            self,
            primary_marker_sizes: Sequence[float],
            overlay_marker_sizes: Sequence[float],
        ) -> None:
            entries = []
            selected_group = self._selected_group(self.selected_layer_index)
            for layer_index, groups, colour, marker_sizes in (
                (0, self.point_groups, "#202020", primary_marker_sizes),
                (
                    1,
                    self.overlay_layer.point_groups
                    if self.overlay_layer is not None
                    else (),
                    self.overlay_layer.colour
                    if self.overlay_layer is not None
                    else "#202020",
                    overlay_marker_sizes,
                ),
            ):
                for group_index, group in enumerate(groups):
                    point = group[0]
                    entries.append(
                        {
                            "layer_index": layer_index,
                            "group": group,
                            "group_index": group_index,
                            "text": format_hkl(point.hkl),
                            "colour": colour,
                            "position": pole_display_position(point),
                            "marker_size": float(marker_sizes[group_index]),
                            "selected": group is selected_group,
                        }
                    )
            entries.sort(
                key=lambda entry: (
                    not entry["selected"],
                    entry["layer_index"],
                    -entry["group"][0].d_spacing,
                    entry["group"][0].hkl,
                )
            )
            if not entries:
                return

            anchors = self.ax.transData.transform(
                np.asarray([entry["position"] for entry in entries], dtype=float)
            )
            dpi_scale = self.figure.dpi / 72.0
            protected_boxes = []
            for anchor, entry in zip(anchors, entries):
                radius = math.sqrt(entry["marker_size"]) * dpi_scale / 2.0 + 2.0
                protected_boxes.append(
                    (
                        anchor[0] - radius,
                        anchor[1] - radius,
                        anchor[0] + radius,
                        anchor[1] + radius,
                    )
                )
            renderer = self.canvas.get_renderer()
            legend = self.ax.get_legend()
            if legend is not None:
                legend_box = legend.get_window_extent(renderer)
                protected_boxes.append(
                    (legend_box.x0, legend_box.y0, legend_box.x1, legend_box.y1)
                )
            box_sizes = [
                (max(24.0, len(entry["text"]) * 6.2), 13.0)
                for entry in entries
            ]
            axes_box = self.ax.bbox
            show_leaders = self.label_leaders_var.get()
            placements = place_label_boxes(
                [tuple(anchor) for anchor in anchors],
                box_sizes,
                (axes_box.x0 + 3, axes_box.y0 + 3, axes_box.x1 - 3, axes_box.y1 - 3),
                protected_boxes=protected_boxes,
                required_indices={
                    index for index, entry in enumerate(entries) if entry["selected"]
                },
                anchor_clearances=(
                    None
                    if show_leaders
                    else [
                        math.sqrt(entry["marker_size"]) * dpi_scale / 2.0 + 2.0
                        for entry in entries
                    ]
                ),
                candidate_gaps=(
                    (6.0, 16.0, 28.0, 44.0, 64.0)
                    if show_leaders
                    else (1.5, 4.0, 7.0)
                ),
            )
            inverse = self.ax.transData.inverted()
            for anchor, entry, box_size, placement in zip(
                anchors, entries, box_sizes, placements
            ):
                if placement is None:
                    continue
                text_position = inverse.transform(placement)
                width, height = box_size
                horizontal_gap = max(
                    placement[0] - anchor[0],
                    anchor[0] - (placement[0] + width),
                    0.0,
                )
                vertical_gap = max(
                    placement[1] - anchor[1],
                    anchor[1] - (placement[1] + height),
                    0.0,
                )
                leader = math.hypot(horizontal_gap, vertical_gap) > 10.0
                self.ax.annotate(
                    entry["text"],
                    entry["position"],
                    xytext=text_position,
                    textcoords="data",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color=entry["colour"],
                    arrowprops=(
                        {
                            "arrowstyle": "-",
                            "color": entry["colour"],
                            "linewidth": 0.55,
                            "shrinkA": 1.0,
                            "shrinkB": 2.0,
                        }
                        if leader and show_leaders
                        else None
                    ),
                    annotation_clip=False,
                    zorder=5,
                )

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
            global_size_scale = self._percentage(
                self.global_size_var, 100.0, 25.0, 300.0
            ) / 100.0
            marker_sizes *= self._percentage(
                self.primary_size_var, 100.0, 10.0, 300.0
            ) / 100.0 * global_size_scale
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
            if self.coincident_outlines_var.get():
                self._draw_coincident_group_markers(
                    self.point_groups,
                    marker_sizes,
                )

            overlay_sizes = np.empty(0, dtype=float)
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
                overlay_sizes *= layer.size_scale * global_size_scale
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
                if self.coincident_outlines_var.get():
                    self._draw_coincident_group_markers(
                        layer.point_groups,
                        overlay_sizes,
                    )
                from matplotlib.lines import Line2D

                primary_legend_name = self.cif_document.name
                overlay_legend_name = layer.name
                if layer.document is self.cif_document:
                    primary_legend_name = localised(
                        f"{primary_legend_name} — primary",
                        f"{primary_legend_name} — principale",
                        f"{primary_legend_name} — основная",
                    )
                    overlay_legend_name = localised(
                        f"{overlay_legend_name} — second",
                        f"{overlay_legend_name} — seconde",
                        f"{overlay_legend_name} — вторая",
                    )
                elif primary_legend_name == overlay_legend_name:
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

            selected_group = self._selected_group(self.selected_layer_index)
            if selected_group is not None:
                selected = selected_group[0]
                selected_x, selected_y = pole_display_position(selected)
                selected_sizes = (
                    marker_sizes
                    if self.selected_layer_index == 0
                    else overlay_sizes
                )
                context = self._layer_context(self.selected_layer_index)
                group_index = context[4].index(selected_group)
                marker_size = float(selected_sizes[group_index])
                self.ax.scatter(
                    [selected_x],
                    [selected_y],
                    s=(math.sqrt(marker_size) + 8.0) ** 2,
                    facecolors="none",
                    edgecolors="#d23b2d",
                    linewidths=2.0,
                    clip_on=False,
                    zorder=4,
                )
                self.show_information(
                    selected_group,
                    layer_index=self.selected_layer_index,
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
            if self.labels_var.get():
                self._draw_pole_labels(marker_sizes, overlay_sizes)
            self.canvas.draw_idle()

        def show_information(
            self,
            group: Sequence[PolePoint],
            *,
            layer_index: int = 0,
        ) -> None:
            context = self._layer_context(layer_index)
            if context is None:
                return
            document, _crystal, centre, _orientation, _groups, selected_hkl = context
            point = group[0]
            if selected_hkl is not None:
                for candidate in group:
                    if candidate.hkl == selected_hkl:
                        point = candidate
                        break
            wavelength = self.get_wavelength()
            self._ensure_layer_intensities(layer_index, show_errors=False)
            intensity = self._point_intensity(point, layer_index)
            phase_name = document.name
            if (
                self.overlay_layer is not None
                and self.overlay_layer.document is self.cif_document
            ):
                phase_name = localised(
                    f"{phase_name} — {'primary' if layer_index == 0 else 'second'}",
                    f"{phase_name} — {'principale' if layer_index == 0 else 'seconde'}",
                    f"{phase_name} — {'основная' if layer_index == 0 else 'вторая'}",
                )
            lines = [
                localised(
                    f"Phase: {phase_name}",
                    f"Phase : {phase_name}",
                    f"Фаза: {phase_name}",
                ),
                f"hkl: {format_hkl(point.hkl)}",
                localised(
                    f"Centring: {format_hkl(centre)}",
                    f"Centrage : {format_hkl(centre)}",
                    f"Центрирование: {format_hkl(centre)}",
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
                    f"Calculated powder intensity: {intensity:.3f}%"
                    if intensity is not None
                    else "Calculated powder intensity: —",
                    f"Intensité calculée du diagramme de poudre : {intensity:.3f} %"
                    if intensity is not None
                    else "Intensité calculée du diagramme de poudre : —",
                    f"Расчётная интенсивность порошкового графика: {intensity:.3f}%"
                    if intensity is not None
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
                    candidate_intensity = self._point_intensity(
                        candidate, layer_index
                    )
                    lines.append(
                        f"{format_hkl(candidate.hkl)}: "
                        f"d={candidate.d_spacing:.6f} Å; "
                        f"2θ={candidate.two_theta:.5f}°; "
                        + (
                            f"Irel={candidate_intensity:.3f}%"
                            if candidate_intensity is not None
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

        def _select_pole_group(
            self,
            layer_index: int,
            group: Sequence[PolePoint],
        ) -> None:
            if not group:
                return
            if layer_index == 0:
                self.selected_hkl = group[0].hkl
            elif layer_index == 1 and self.overlay_layer is not None:
                self.overlay_layer.selected_hkl = group[0].hkl
            else:
                return
            self.selected_layer_index = layer_index
            self.redraw()

        def _show_pole_candidate_menu(self, candidates, event) -> None:
            if self._selection_menu is not None:
                try:
                    self._selection_menu.destroy()
                except tk.TclError:
                    pass
            menu = tk.Menu(self.root, tearoff=False)
            self._selection_menu = menu
            for _distance, layer_index, group in candidates:
                context = self._layer_context(layer_index)
                if context is None:
                    continue
                document = context[0]
                count = len(group)
                reflection_text = localised(
                    f"{count} reflections",
                    f"{count} réflexions",
                    f"отражений: {count}",
                )
                label = f"{document.name} — {format_hkl(group[0].hkl)}"
                if count > 1:
                    label += f" — {reflection_text}"
                menu.add_command(
                    label=label,
                    command=lambda index=layer_index, selected=group: (
                        self._select_pole_group(index, selected)
                    ),
                )
            gui_event = getattr(event, "guiEvent", None)
            x_root = getattr(gui_event, "x_root", None)
            y_root = getattr(gui_event, "y_root", None)
            if x_root is None or y_root is None:
                x_root = self.root.winfo_pointerx()
                y_root = self.root.winfo_pointery()
            try:
                menu.tk_popup(int(x_root), int(y_root))
            finally:
                menu.grab_release()

        def on_press(self, event) -> None:
            if event.button != 1:
                return
            if self.overlay_layer is not None and not self._joint_rotation_enabled():
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
                self._set_primary_orientation(
                    user_rotation=delta @ self.user_rotation,
                    update_overlay_entries=False,
                )
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
            self._set_primary_orientation(
                user_rotation=delta @ self.user_rotation,
                update_overlay_entries=False,
            )
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
            if self._joint_rotation_enabled():
                self.update_overlay_rotation_entries()
            self._frames.cancel()
            if was_dragged or drag_mode == "structure":
                self.redraw()
            if drag_mode == "structure":
                return
            if was_dragged or event.inaxes is not self.ax:
                return
            if event.x is None or event.y is None:
                return

            candidates = []
            for layer_index in (0, 1):
                context = self._layer_context(layer_index)
                if context is None:
                    continue
                groups = context[4]
                if not groups:
                    continue
                screen_points = self.ax.transData.transform(
                    np.asarray(
                        [
                            pole_display_position(group[0])
                            for group in groups
                        ],
                        dtype=float,
                    )
                )
                distances = np.hypot(
                    screen_points[:, 0] - event.x,
                    screen_points[:, 1] - event.y,
                )
                candidates.extend(
                    (float(distance), layer_index, groups[index])
                    for index, distance in enumerate(distances)
                    if distance <= 14.0
                )
            candidates.sort(key=lambda item: (item[0], item[1]))
            if not candidates:
                return
            nearest_distance = candidates[0][0]
            ambiguous = [
                candidate
                for candidate in candidates
                if candidate[0] - nearest_distance <= 3.0
            ]
            if len(ambiguous) > 1:
                self._show_pole_candidate_menu(ambiguous, event)
            else:
                _distance, layer_index, group = candidates[0]
                self._select_pole_group(layer_index, group)

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
