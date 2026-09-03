"""Главная страница 2θ: измерения, выбор осей и CIF-фазы."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from tkinter import colorchooser, simpledialog, ttk

try:
    from .controls import CollapsibleSection, ScrollableControls
    from .correction import (
        CustomInputDialog,
        fit_gaussian_peak,
        load_reference_peaks,
        save_reference_peaks,
    )
    from .cif_xrd import (
        ReflectionRow,
        Structure,
        _data_path,
        calculate_reflections,
        export_csv,
        load_scattering_factors,
        read_structure,
    )
    from .i18n import (
        LocalizedStringVar,
        apply_language,
        choice_code,
        filedialog,
        localised,
        messagebox,
        translate_text,
    )
    from .io.correction import write_processed_scan, write_processed_xrdml
    from .models.correction import CorrectionRequest, validate_result_mode
    from .models.data_errors import XRDDataError
    from .models.viewer import (
        DEFAULT_PLOT_COLOURS,
        PlotItem,
        ViewerState,
        axis_has_degree_units as _axis_has_degree_units,
        axis_key as _axis_key,
        is_two_theta as _is_two_theta,
        overlay_phase_geometry,
        resolve_limits,
        scan_x_limits,
        scrolled_limits,
        scrollbar_window,
        transformed_intensity,
    )
    from .services.correction import apply_correction
    from .xrd_io import Scan1D, assign_text_axis, read_scan_file
except ImportError:
    from controls import CollapsibleSection, ScrollableControls
    from correction import (
        CustomInputDialog,
        fit_gaussian_peak,
        load_reference_peaks,
        save_reference_peaks,
    )
    from cif_xrd import (
        ReflectionRow,
        Structure,
        _data_path,
        calculate_reflections,
        export_csv,
        load_scattering_factors,
        read_structure,
    )
    from i18n import (
        LocalizedStringVar,
        apply_language,
        choice_code,
        filedialog,
        localised,
        messagebox,
        translate_text,
    )
    from io.correction import write_processed_scan, write_processed_xrdml
    from models.correction import CorrectionRequest, validate_result_mode
    from models.data_errors import XRDDataError
    from models.viewer import (
        DEFAULT_PLOT_COLOURS,
        PlotItem,
        ViewerState,
        axis_has_degree_units as _axis_has_degree_units,
        axis_key as _axis_key,
        is_two_theta as _is_two_theta,
        overlay_phase_geometry,
        resolve_limits,
        scan_x_limits,
        scrolled_limits,
        scrollbar_window,
        transformed_intensity,
    )
    from services.correction import apply_correction
    from xrd_io import Scan1D, assign_text_axis, read_scan_file


COLOURS = DEFAULT_PLOT_COLOURS
MAX_LEGEND_ITEMS = 12


def _reflection_intensity_text(row: ReflectionRow) -> str:
    if row.intensity is None:
        return localised("Irel = —", "Irel = —", "Iотн = —")
    return localised(
        f"Irel = {row.intensity:.2f}%",
        f"Irel = {row.intensity:.2f} %",
        f"Iотн = {row.intensity:.2f}%",
    )


def _axis_label(name: str) -> str:
    labels = {
        "2theta": "2θ",
        "twotheta": "2θ",
        "2θ": "2θ",
        "theta": "θ",
        "omega": "ω",
        "chi": "χ",
        "phi": "φ",
        "xdrive": "X",
        "ydrive": "Y",
        "zdrive": "Z",
        "scanaxis": "X",
        "x": "X",
    }
    return labels.get(_axis_key(name), name)
class ReflectionTableWindow(tk.Toplevel):
    """Таблица отражений выбранной CIF-фазы."""

    def __init__(self, parent: tk.Misc, title: str, rows: list[ReflectionRow]) -> None:
        super().__init__(parent)
        self.title(
            localised(
                f"Reflections – {title}",
                f"Réflexions – {title}",
                f"Отражения – {title}",
            )
        )
        self.geometry("1080x650")
        self.minsize(820, 480)
        self.rows = rows

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        columns = ("hkl", "d", "two_theta", "line", "wave", "weight", "mult", "f2", "i")
        self.table = ttk.Treeview(outer, columns=columns, show="headings")
        headings = (
            ("hkl", "(h k l)", 170),
            ("d", "d, Å", 90),
            ("two_theta", "2θ, °", 90),
            ("line", "Линия", 90),
            ("wave", "λ, Å", 80),
            ("weight", "Вес", 65),
            ("mult", "Кратность", 75),
            ("f2", "Σ|F|^2", 100),
            ("i", "Iотн, %", 85),
        )
        for key, label, width in headings:
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor="center")

        y_scroll = ttk.Scrollbar(outer, orient="vertical", command=self.table.yview)
        x_scroll = ttk.Scrollbar(outer, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        for row in rows:
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
                    "—" if row.f2_sum is None else f"{row.f2_sum:.6g}",
                    "—" if row.intensity is None else f"{row.intensity:.4f}",
                ),
            )

        footer = ttk.Frame(outer)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(9, 0))
        ttk.Label(
            footer,
            text=localised(
                f"Rows: {len(rows)}",
                f"Lignes : {len(rows)}",
                f"Строк: {len(rows)}",
            ),
        ).pack(side="left")
        ttk.Button(footer, text="Сохранить CSV…", command=self.save_csv).pack(side="right")
        apply_language(self)

    def save_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить таблицу",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"), ("Все файлы", "*.*")),
        )
        if not path:
            return
        try:
            export_csv(path, self.rows)
        except OSError as exc:
            messagebox.showerror("Ошибка сохранения", str(exc), parent=self)


class TwoThetaPage(ttk.Frame):
    """Рабочая область, которая открывается при запуске приложения."""

    def __init__(
        self,
        parent: tk.Misc,
        on_open_theoretical: Callable[[str], None] | None = None,
        on_open_reflection_table: Callable[[str], None] | None = None,
        on_commit_scan: Callable[[str, Scan1D, str], None] | None = None,
        on_open_structure: Callable[[str], None] | None = None,
        on_import_paths: Callable[[list[str]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_open_theoretical = on_open_theoretical
        self.on_open_reflection_table = on_open_reflection_table
        self.on_commit_scan = on_commit_scan
        self.on_open_structure = on_open_structure
        self.on_import_paths = on_import_paths
        self.viewer_state = ViewerState(colours=COLOURS)
        self.items = self.viewer_state.items
        self._row_cache: dict[str, tuple[tuple, list[ReflectionRow]]] = {}
        self._axis_name_by_label: dict[str, str] = {}
        self._cif_widgets: list[tk.Misc] = []
        self._syncing_limits = False
        self._updating_scrollbars = False
        self._split_drag = None
        self._tree_drag_uid = None
        self._syncing_processing = False
        self.rect_selector = None
        self.fit_active = False
        self._fit_uid: str | None = None

        self.status = LocalizedStringVar(
            value=localised(
                "Open an XRDML, RAW, XY or CIF file.",
                "Ouvrez un fichier XRDML, RAW, XY ou CIF.",
                "Откройте XRDML, RAW, XY или CIF.",
            )
        )
        self.selection_info = LocalizedStringVar(value="Выберите точку или отражение на графике.")
        self._plot_selection = None
        self.y_scale = tk.StringVar(value=translate_text("Линейная"))
        self.axis_var = tk.StringVar()
        self.axis_hint = LocalizedStringVar(value="")
        self.phase_height = tk.StringVar(value="25")
        self.phase_x_min = tk.StringVar(value="5")
        self.phase_x_max = tk.StringVar(value="120")
        self.phase_layout = tk.StringVar(value=translate_text("Наложение"))
        self.phase_style = tk.StringVar(value=translate_text("Штрихи"))
        self.fwhm = tk.StringVar(value="0.12")
        self.overlay_single_line = tk.BooleanVar(value=True)
        self.overlay_height = tk.DoubleVar(value=10.0)
        self.overlay_height_text = tk.StringVar(value="10%")
        self.offset = tk.StringVar(value="0")
        self.processing_name = LocalizedStringVar(value="Выберите измерение в списке.")
        self.processing_x = tk.DoubleVar(value=0.0)
        self.processing_y = tk.DoubleVar(value=0.0)
        self.processing_factor = tk.DoubleVar(value=1.0)
        self.processing_shift_omega = tk.BooleanVar(value=True)
        self.result_mode = tk.StringVar(value="add")
        self.x_min = tk.StringVar()
        self.x_max = tk.StringVar()
        self.y_min = tk.StringVar()
        self.y_max = tk.StringVar()
        self.ka1_enabled = tk.BooleanVar(value=True)
        self.ka2_enabled = tk.BooleanVar(value=False)
        self.ka1_wave = tk.StringVar(value="1.54056")
        self.ka2_wave = tk.StringVar(value="1.54443")
        self.ka1_weight = tk.StringVar(value="1.0")
        self.ka2_weight = tk.StringVar(value="0.5")

        self.factors = load_scattering_factors(_data_path())
        self._build()
        apply_language(self)
        self._draw()

    @property
    def _axes_linked(self) -> bool:
        return self.viewer_state.plot.axes_linked

    @_axes_linked.setter
    def _axes_linked(self, value: bool) -> None:
        self.viewer_state.plot.axes_linked = bool(value)

    @property
    def _navigation_x_bounds(self) -> tuple[float, float]:
        return self.viewer_state.plot.navigation_x_bounds

    @_navigation_x_bounds.setter
    def _navigation_x_bounds(self, value: tuple[float, float]) -> None:
        self.viewer_state.plot.navigation_x_bounds = tuple(map(float, value))

    @property
    def _navigation_y_bounds(self) -> tuple[float, float]:
        return self.viewer_state.plot.navigation_y_bounds

    @_navigation_y_bounds.setter
    def _navigation_y_bounds(self, value: tuple[float, float]) -> None:
        self.viewer_state.plot.navigation_y_bounds = tuple(map(float, value))

    @property
    def _overlay_phase_top(self) -> float:
        return self.viewer_state.plot.overlay_phase_top

    @_overlay_phase_top.setter
    def _overlay_phase_top(self, value: float) -> None:
        self.viewer_state.plot.overlay_phase_top = float(value)

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.control_panel = ScrollableControls(self, width=390)
        self.control_panel.grid(row=0, column=0, sticky="nsew")
        self.sidebar_canvas = self.control_panel.canvas
        sidebar = self.control_panel.body
        sidebar.columnconfigure(0, weight=1)

        files = CollapsibleSection(sidebar, text="Данные", padding=7)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure((0, 1), weight=1)
        ttk.Button(files, text="Открыть файлы…", command=self.open_files).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(files, text="Открыть папку…", command=self.open_folder).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )
        self.add_cif_button = ttk.Button(
            files, text="Добавить CIF…", command=self.open_cif
        )
        self.add_cif_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        radiation = CollapsibleSection(sidebar, text="Излучение для CIF", padding=7)
        radiation.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(radiation, text="Линия").grid(row=0, column=0, sticky="w")
        ttk.Label(radiation, text="λ, Å").grid(row=0, column=1)
        ttk.Label(radiation, text="Вес").grid(row=0, column=2)
        self.ka1_check = ttk.Checkbutton(
            radiation, text="Kα1", variable=self.ka1_enabled, command=self.recalculate
        )
        self.ka1_check.grid(row=1, column=0, sticky="w")
        self.ka1_wave_entry = ttk.Entry(
            radiation, textvariable=self.ka1_wave, width=9
        )
        self.ka1_wave_entry.grid(row=1, column=1)
        self.ka1_weight_entry = ttk.Entry(
            radiation, textvariable=self.ka1_weight, width=7
        )
        self.ka1_weight_entry.grid(row=1, column=2)
        self.ka2_check = ttk.Checkbutton(
            radiation, text="Kα2", variable=self.ka2_enabled, command=self.recalculate
        )
        self.ka2_check.grid(row=2, column=0, sticky="w")
        self.ka2_wave_entry = ttk.Entry(
            radiation, textvariable=self.ka2_wave, width=9
        )
        self.ka2_wave_entry.grid(row=2, column=1)
        self.ka2_weight_entry = ttk.Entry(
            radiation, textvariable=self.ka2_weight, width=7
        )
        self.ka2_weight_entry.grid(row=2, column=2)
        for variable in (self.ka1_wave, self.ka2_wave, self.ka1_weight, self.ka2_weight):
            variable.trace_add("write", lambda *_: self._schedule_redraw())

        list_frame = CollapsibleSection(sidebar, text="Загруженные наборы", padding=5)
        list_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            list_frame,
            columns=("visible", "kind", "colour"),
            show="tree headings",
            selectmode="browse",
            height=11,
        )
        self.tree.heading("#0", text="Название")
        self.tree.heading("visible", text="Вид.")
        self.tree.heading("kind", text="Тип")
        self.tree.heading("colour", text="Цвет")
        self.tree.column("#0", width=170)
        self.tree.column("visible", width=42, anchor="center")
        self.tree.column("kind", width=72, anchor="center")
        self.tree.column("colour", width=62, anchor="center")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<Double-1>", self.toggle_selected)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._update_buttons())
        self.tree.bind("<ButtonPress-1>", self._tree_drag_start, add="+")
        self.tree.bind("<B1-Motion>", self._tree_drag_motion, add="+")
        self.tree.bind("<ButtonRelease-1>", self._tree_drag_end, add="+")

        row = ttk.Frame(list_frame)
        row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        for column in range(4):
            row.columnconfigure(column, weight=1)
        self.visible_button = ttk.Button(row, text="Скрыть", command=self.toggle_selected)
        self.visible_button.grid(row=0, column=0, sticky="ew")
        self.colour_button = ttk.Button(row, text="Цвет", command=self.choose_colour)
        self.colour_button.grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(row, text="Удалить", command=self.remove_selected).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Button(row, text="Все", command=self.show_all).grid(
            row=0, column=3, sticky="ew", padx=(3, 0)
        )
        self.rename_button = ttk.Button(
            list_frame, text="Переименовать…", command=self.rename_selected
        )
        self.rename_button.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        self.clear_button = ttk.Button(
            list_frame,
            text="Очистить",
            command=self.clear_all,
        )
        self.clear_button.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(5, 0),
        )
        self.table_button = ttk.Button(
            list_frame, text="Таблица отражений…", command=self.open_reflection_table
        )
        self.table_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.pole_button = ttk.Button(
            list_frame,
            text="Расчётная полюсная фигура…",
            command=self.open_theoretical,
        )
        self.pole_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ordering = ttk.Frame(list_frame)
        ordering.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ordering.columnconfigure((0, 1), weight=1)
        self.up_button = ttk.Button(ordering, text="Выше", command=lambda: self.move_selected(-1))
        self.down_button = ttk.Button(ordering, text="Ниже", command=lambda: self.move_selected(1))
        self.up_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.down_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.structure_button = ttk.Button(
            list_frame, text="Структура CIF…", command=self.open_structure
        )
        self.structure_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        processing = CollapsibleSection(
            sidebar, text="Коррекция выбранного измерения", padding=7
        )
        processing.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        processing.columnconfigure(1, weight=1)
        self.processing_section = processing
        ttk.Label(
            processing,
            textvariable=self.processing_name,
            wraplength=345,
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))

        self.processing_controls: list[tk.Misc] = []
        transform_rows = (
            ("Сдвиг X", self.processing_x, self._processing_x_changed),
            ("Сдвиг нуля Y", self.processing_y, self._processing_y_changed),
            ("Масштаб Y", self.processing_factor, self._processing_factor_changed),
        )
        for row_index, (label, variable, command) in enumerate(transform_rows, start=1):
            ttk.Label(processing, text=label).grid(row=row_index, column=0, sticky="w")
            slider = ttk.Scale(processing, variable=variable, command=command)
            slider.grid(row=row_index, column=1, sticky="ew", padx=5)
            entry = ttk.Entry(processing, textvariable=variable, width=10)
            entry.grid(row=row_index, column=2, sticky="ew")
            entry.bind("<Return>", self._processing_entry_changed)
            entry.bind("<FocusOut>", self._processing_entry_changed)
            self.processing_controls.extend((slider, entry))
            if row_index == 1:
                self.processing_x_slider = slider
            elif row_index == 2:
                self.processing_y_slider = slider
            else:
                self.processing_factor_slider = slider
        self.processing_factor_slider.configure(from_=0.05, to=5.0)

        tool_row = ttk.Frame(processing)
        tool_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        tool_row.columnconfigure(0, weight=1)
        self.fit_button = ttk.Button(
            tool_row, text="Коррекция по пику…", command=self.activate_peak_fit
        )
        self.fit_button.grid(row=0, column=0, sticky="ew")

        mode_row = ttk.Frame(processing)
        mode_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        self.add_result_radio = ttk.Radiobutton(
            mode_row, text="Добавить новый", variable=self.result_mode, value="add"
        )
        self.add_result_radio.pack(side="left")
        self.replace_result_radio = ttk.Radiobutton(
            mode_row, text="Заменить исходный", variable=self.result_mode, value="replace"
        )
        self.replace_result_radio.pack(side="left", padx=(8, 0))

        self.shift_omega_check = ttk.Checkbutton(
            processing,
            text="Сдвигать Omega на 1/2",
            variable=self.processing_shift_omega,
            command=self._processing_values_changed,
        )
        self.shift_omega_check.grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

        action_row = ttk.Frame(processing)
        action_row.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        action_row.columnconfigure((0, 1), weight=1)
        self.apply_processing_button = ttk.Button(
            action_row,
            text="Применить результат",
            command=self.apply_processing_result,
        )
        self.apply_processing_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.reset_processing_button = ttk.Button(
            action_row, text="Сбросить преобразования", command=self.reset_processing
        )
        self.reset_processing_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.save_processing_button = ttk.Button(
            action_row, text="Сохранить результат…", command=self.save_processing_result
        )
        self.save_processing_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        self.processing_controls.extend(
            (
                self.fit_button,
                self.add_result_radio,
                self.replace_result_radio,
                self.shift_omega_check,
                self.apply_processing_button,
                self.save_processing_button,
                self.reset_processing_button,
            )
        )

        view = CollapsibleSection(sidebar, text="Отображение", padding=7)
        view.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        view.columnconfigure(1, weight=1)
        ttk.Label(view, text="Ось X").grid(row=0, column=0, sticky="w")
        self.axis_combo = ttk.Combobox(
            view,
            textvariable=self.axis_var,
            state="disabled",
            width=18,
        )
        self.axis_combo.grid(row=0, column=1, sticky="ew")
        self.axis_combo.bind("<<ComboboxSelected>>", self.change_selected_axis)
        ttk.Label(view, text="Шкала Y").grid(row=1, column=0, sticky="w")
        scale = ttk.Combobox(
            view,
            textvariable=self.y_scale,
            values=("Линейная", "Логарифмическая", "Квадратный корень", "Квадрат"),
            state="readonly",
            width=18,
        )
        scale.grid(row=1, column=1, sticky="ew")
        scale.bind("<<ComboboxSelected>>", lambda _event: self._draw())
        ttk.Label(view, text="Сдвиг кривых").grid(row=2, column=0, sticky="w")
        ttk.Entry(view, textvariable=self.offset, width=10).grid(row=2, column=1, sticky="ew")
        ttk.Label(view, text="Режим CIF").grid(row=3, column=0, sticky="w")
        self.phase_layout_combo = ttk.Combobox(
            view,
            textvariable=self.phase_layout,
            values=("Отдельно", "Наложение"),
            state="readonly",
            width=18,
        )
        self.phase_layout_combo.grid(row=3, column=1, sticky="ew")
        self.phase_layout_combo.bind(
            "<<ComboboxSelected>>", self._change_phase_layout
        )
        ttk.Label(view, text="Фазы").grid(row=4, column=0, sticky="w")
        self.phase_style_combo = ttk.Combobox(
            view,
            textvariable=self.phase_style,
            values=("Штрихи", "Профиль"),
            state="readonly",
            width=18,
        )
        self.phase_style_combo.grid(row=4, column=1, sticky="ew")
        self.phase_style_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._draw(preserve_view=True),
        )
        ttk.Label(view, text="FWHM, °").grid(row=5, column=0, sticky="w")
        self.fwhm_entry = ttk.Entry(view, textvariable=self.fwhm, width=10)
        self.fwhm_entry.grid(row=5, column=1, sticky="ew")
        ttk.Button(view, text="Перестроить", command=self._draw).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        self.phase_height_label = ttk.Label(view, text="Высота CIF, %")
        self.phase_height_label.grid(row=7, column=0, sticky="w")
        self.phase_height_entry = ttk.Spinbox(
            view, from_=10, to=85, increment=5, width=8,
            textvariable=self.phase_height, command=self._apply_phase_height,
        )
        self.phase_height_entry.grid(row=7, column=1, sticky="ew", pady=(5, 0))
        self.phase_height_entry.bind("<Return>", self._apply_phase_height)
        self.phase_height_entry.bind("<FocusOut>", self._apply_phase_height)
        self.overlay_single_check = ttk.Checkbutton(
            view,
            text="CIF в одну линию",
            variable=self.overlay_single_line,
            command=self._change_overlay_arrangement,
        )
        self.overlay_single_check.grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )
        ttk.Label(view, text="Высота линий CIF, %").grid(
            row=9, column=0, sticky="w"
        )
        overlay_height_frame = ttk.Frame(view)
        overlay_height_frame.grid(row=9, column=1, sticky="ew")
        overlay_height_frame.columnconfigure(0, weight=1)
        self.overlay_height_scale = ttk.Scale(
            overlay_height_frame,
            from_=1,
            to=100,
            variable=self.overlay_height,
            command=self._overlay_height_changed,
        )
        self.overlay_height_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            overlay_height_frame,
            textvariable=self.overlay_height_text,
            width=5,
            anchor="e",
        ).grid(row=0, column=1, sticky="e", padx=(5, 0))
        ttk.Label(view, textvariable=self.axis_hint, wraplength=345).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        self._cif_widgets = [
            self.add_cif_button,
            self.ka1_check,
            self.ka1_wave_entry,
            self.ka1_weight_entry,
            self.ka2_check,
            self.ka2_wave_entry,
            self.ka2_weight_entry,
            self.phase_layout_combo,
            self.phase_style_combo,
            self.fwhm_entry,
        ]

        limits = CollapsibleSection(sidebar, text="Границы графика", padding=7)
        limits.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        for column in (1, 3):
            limits.columnconfigure(column, weight=1)
        for row_index, (label_a, var_a, label_b, var_b) in enumerate(
            (("X min", self.x_min, "X max", self.x_max), ("Y min", self.y_min, "Y max", self.y_max))
        ):
            ttk.Label(limits, text=label_a).grid(row=row_index, column=0, sticky="w")
            ttk.Entry(limits, textvariable=var_a, width=8).grid(row=row_index, column=1, sticky="ew")
            ttk.Label(limits, text=label_b).grid(row=row_index, column=2, sticky="w", padx=(5, 0))
            ttk.Entry(limits, textvariable=var_b, width=8).grid(row=row_index, column=3, sticky="ew")
        ttk.Button(limits, text="Применить", command=self._draw).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(limits, text="Авто", command=self.reset_limits).grid(
            row=2, column=2, columnspan=2, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        ttk.Label(limits, text="Отдельная ось CIF, 2θ").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )
        ttk.Label(limits, text="min").grid(row=4, column=0)
        self.phase_min_entry = ttk.Entry(limits, textvariable=self.phase_x_min, width=8)
        self.phase_min_entry.grid(row=4, column=1, sticky="ew")
        ttk.Label(limits, text="max").grid(row=4, column=2)
        self.phase_max_entry = ttk.Entry(limits, textvariable=self.phase_x_max, width=8)
        self.phase_max_entry.grid(row=4, column=3, sticky="ew")

        plot = ttk.Frame(self, padding=(0, 8, 8, 4))
        plot.grid(row=0, column=1, sticky="nsew")
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.plot_grid = self.figure.add_gridspec(3, 1, height_ratios=(0.75, 0.025, 0.25))
        self.scan_axis = self.figure.add_subplot(self.plot_grid[0, 0])
        self.phase_axis = self.figure.add_subplot(self.plot_grid[2, 0])
        self.split_axis = self.figure.add_subplot(self.plot_grid[1, 0])
        self.split_axis.axhline(0.5, color="#b8b8b8", lw=2)
        self.split_axis.text(0.5, 0.5, "↕", ha="center", va="center", color="#555555")
        self.split_axis.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.y_scrollbar = ttk.Scrollbar(
            plot,
            orient="vertical",
            command=lambda *args: self._scroll_view("y", *args),
        )
        self.y_scrollbar.grid(row=0, column=1, sticky="ns")
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.x_scrollbar = ttk.Scrollbar(
            plot,
            orient="horizontal",
            command=lambda *args: self._scroll_view("x", *args),
        )
        self.x_scrollbar.grid(row=2, column=0, sticky="ew")
        selected = ttk.LabelFrame(plot, text="Выбранная точка / отражение", padding=7)
        selected.grid(row=3, column=0, columnspan=2, sticky="ew")
        selected.columnconfigure(0, weight=1)
        info = ttk.Label(selected, textvariable=self.selection_info, anchor="w", justify="left")
        info.grid(row=0, column=0, sticky="ew")
        notice = ttk.Label(plot, textvariable=self.status, anchor="w", justify="left", padding=(7, 4))
        notice.grid(row=4, column=0, columnspan=2, sticky="ew")
        plot.bind("<Configure>", lambda e: (
            info.configure(wraplength=max(150, e.width-35)),
            notice.configure(wraplength=max(150, e.width-35))))
        self.canvas.mpl_connect("button_press_event", self._plot_press)
        self.canvas.mpl_connect("motion_notify_event", self._plot_motion)
        self.canvas.mpl_connect("button_release_event", self._plot_release)

        self._update_buttons()

    def _next_colour(self) -> str:
        return self.viewer_state.next_colour()

    def _group_uids(self, uid: str) -> list[str]:
        return self.viewer_state.group_uids(uid)

    def _move_item(self, uid: str, target: str) -> None:
        group = self._group_uids(uid)
        if uid == target or target not in group:
            return
        order = self.viewer_state.move_to_target(uid, target)
        for index, key in enumerate(order):
            self.tree.move(key, "", index)
        self.tree.selection_set(uid)
        self.tree.see(uid)
        self._update_buttons()
        self._draw(preserve_view=True)

    def move_selected(self, direction: int) -> None:
        uid = self._selected_uid()
        group = self._group_uids(uid) if uid else []
        if uid in group:
            index = group.index(uid) + direction
            if 0 <= index < len(group):
                self._move_item(uid, group[index])

    def _tree_drag_start(self, event) -> None:
        self._tree_drag_uid = (
            self.tree.identify_row(event.y)
            if self.tree.identify_region(event.x, event.y) in ("tree", "cell") else None
        )

    def _tree_drag_motion(self, event) -> None:
        if self._tree_drag_uid:
            target = self.tree.identify_row(event.y)
            if target:
                self._move_item(self._tree_drag_uid, target)

    def _tree_drag_end(self, _event) -> None:
        self._tree_drag_uid = None

    def _apply_phase_height(self, _event=None) -> None:
        if self._phase_layout_code() == "overlay":
            return
        try:
            percent = float(self.phase_height.get().replace(",", "."))
            percent = self.viewer_state.plot.set_phase_height(percent)
        except (ValueError, XRDDataError):
            self.status.set(translate_text("Высота CIF должна быть числом от 10 до 85%."))
            return
        self.phase_height.set(f"{percent:.1f}")
        fraction = percent / 100.0
        self.plot_grid.set_height_ratios((1.0-fraction, 0.025, fraction))
        self.canvas.draw_idle()

    def _change_overlay_arrangement(self) -> None:
        self.viewer_state.plot.overlay_single_line = bool(
            self.overlay_single_line.get()
        )
        self._update_cif_controls()
        self._draw(preserve_view=True)

    def _overlay_height_changed(self, value=None) -> None:
        try:
            height = float(
                self.overlay_height.get() if value is None else value
            )
            height = self.viewer_state.plot.set_overlay_height(height)
        except (ValueError, XRDDataError, tk.TclError):
            self.status.set(
                localised(
                    "CIF line height must be a number from 1 to 100%.",
                    "La hauteur des raies CIF doit être comprise entre 1 et 100 %.",
                    "Высота линий CIF должна быть числом от 1 до 100%.",
                )
            )
            return
        self.overlay_height.set(height)
        self.overlay_height_text.set(f"{height:.0f}%")
        if (
            self._phase_layout_code() == "overlay"
            and self.overlay_single_line.get()
        ):
            self._draw(preserve_view=True)

    def _update_phase_panel(self) -> None:
        """Keep the user ratio, but give scans the entire plot without CIFs."""
        shown = (
            self._phase_layout_code() == "separate"
            and any(
                item.visible and item.structure is not None
                for item in self.items.values()
            )
        )
        self.phase_axis.set_visible(shown)
        self.phase_axis.set_in_layout(shown)
        self.phase_axis.set_navigate(shown)
        self.split_axis.set_visible(shown)
        self.split_axis.set_in_layout(shown)
        # The divider is draggable by our own handler, but it must never
        # participate in Matplotlib Pan/Zoom. Hidden axes otherwise still pass
        # NavigationToolbar2's in_axes() filter and can capture the event.
        self.split_axis.set_navigate(False)
        self.scan_axis.set_navigate(True)
        self.scan_axis.set_subplotspec(self.plot_grid[0, 0] if shown else self.plot_grid[:, 0])

    def _plot_press(self, event) -> None:
        if self.fit_active:
            return
        if event.button == 1 and event.inaxes is self.split_axis and not self.toolbar.mode:
            height = self.figure.bbox.height * (
                self.scan_axis.get_position().height + self.phase_axis.get_position().height
            )
            ratios = self.plot_grid.get_height_ratios()
            percent = 100.0 * ratios[2] / (ratios[0] + ratios[2])
            self._split_drag = (event.y, percent, max(height, 1.0))
            return
        if not self.toolbar.mode:
            self._plot_click(event)

    def _plot_motion(self, event) -> None:
        if self._split_drag is None or event.y is None:
            return
        start_y, start_percent, height = self._split_drag
        self.phase_height.set(str(start_percent + 100.0*(event.y-start_y)/height))
        self._apply_phase_height()

    def _plot_release(self, _event) -> None:
        self._split_drag = None

    def _sync_x_limits(self, axis) -> None:
        if self._syncing_limits or not self._axes_linked:
            return
        self._syncing_limits = True
        try:
            other = self.phase_axis if axis is self.scan_axis else self.scan_axis
            other.set_xlim(axis.get_xlim(), emit=False)
        finally:
            self._syncing_limits = False

    @staticmethod
    def _set_scrollbar_state(scrollbar, enabled: bool) -> None:
        try:
            scrollbar.state(["!disabled"] if enabled else ["disabled"])
        except (AttributeError, tk.TclError):
            scrollbar.configure(state="normal" if enabled else "disabled")

    def _update_navigation_scrollbars(self) -> None:
        if self._updating_scrollbars:
            return
        self._updating_scrollbars = True
        try:
            for dimension, axis, bounds, scrollbar in (
                ("x", self.scan_axis, self._navigation_x_bounds, self.x_scrollbar),
                ("y", self.scan_axis, self._navigation_y_bounds, self.y_scrollbar),
            ):
                current = axis.get_xlim() if dimension == "x" else axis.get_ylim()
                first, last, movable = scrollbar_window(
                    bounds,
                    current,
                    vertical=dimension == "y",
                )
                scrollbar.set(first, last)
                self._set_scrollbar_state(scrollbar, movable)
        finally:
            self._updating_scrollbars = False

    def _scroll_view(self, dimension: str, *args) -> None:
        if not args:
            return
        axis = self.scan_axis
        bounds = (
            self._navigation_x_bounds
            if dimension == "x"
            else self._navigation_y_bounds
        )
        command = str(args[0])
        if command == "moveto" and len(args) >= 2:
            value = float(args[1])
            units = "units"
        elif command == "scroll" and len(args) >= 3:
            value = int(args[1])
            units = str(args[2])
        else:
            return
        current = axis.get_xlim() if dimension == "x" else axis.get_ylim()
        low, high = scrolled_limits(
            bounds,
            current,
            command,
            value,
            units,
            vertical=dimension == "y",
        )
        if dimension == "x":
            axis.set_xlim(low, high)
        else:
            axis.set_ylim(low, high)
        self._update_navigation_scrollbars()
        self.canvas.draw_idle()

    def _axis_limits_changed(self, axis, dimension: str) -> None:
        if dimension == "x":
            self._sync_x_limits(axis)
        self._update_navigation_scrollbars()

    def _selected_uid(self) -> str | None:
        selection = self.tree.selection()
        return selection[0] if selection else None

    def _radiations(self) -> list[tuple[str, float, float]]:
        result: list[tuple[str, float, float]] = []
        try:
            if self.ka1_enabled.get():
                result.append(("Cu Kα1", float(self.ka1_wave.get().replace(",", ".")), float(self.ka1_weight.get().replace(",", "."))))
            if self.ka2_enabled.get():
                result.append(("Cu Kα2", float(self.ka2_wave.get().replace(",", ".")), float(self.ka2_weight.get().replace(",", "."))))
        except ValueError as exc:
            raise ValueError(
                localised(
                    "Radiation wavelengths and weights must be numeric.",
                    "Les longueurs d’onde et les poids doivent être numériques.",
                    "Для спектральных линий нужны числовые λ и веса.",
                )
            ) from exc
        if not result:
            raise ValueError(
                localised(
                    "Select at least one radiation line.",
                    "Sélectionnez au moins une raie.",
                    "Выберите хотя бы одну спектральную линию.",
                )
            )
        return result

    def _insert_item(self, item: PlotItem) -> None:
        self.viewer_state.add(item)
        self.tree.insert(
            "",
            "end",
            iid=item.uid,
            text=item.name,
            values=(
                "●" if item.visible else "○",
                translate_text(item.kind),
                item.colour,
            ),
        )

    def localize_content(self) -> None:
        for uid, item in self.items.items():
            self.tree.set(uid, "kind", translate_text(item.kind))
        self._update_buttons()
        self._refresh_selection_info()

    def _visible_scans(self) -> list[PlotItem]:
        return self.viewer_state.visible_scans()

    def _cif_axes_compatible(self) -> bool:
        return self.viewer_state.cif_axes_compatible()

    def _phase_layout_code(self) -> str:
        return choice_code("phase_layout", self.phase_layout.get())

    def _change_phase_layout(self, _event=None) -> None:
        if self._phase_layout_code() == "overlay" and not self._cif_axes_compatible():
            self.phase_layout.set(translate_text("Отдельно"))
            self.status.set(
                localised(
                    "CIF overlay is available only for measurements on the 2θ axis.",
                    "La superposition CIF est disponible uniquement pour les mesures sur l’axe 2θ.",
                    "Наложение CIF доступно только для измерений по оси 2θ.",
                )
            )
        self._update_cif_controls()
        self._draw(preserve_view=True)

    def _update_cif_controls(self) -> None:
        # Import, radiation and reflection tables do not depend on scan coordinates.
        for widget in self._cif_widgets:
            widget.configure(
                state=(
                    "readonly"
                    if widget in (self.phase_layout_combo, self.phase_style_combo)
                    else "normal"
                )
            )
        separate = self._phase_layout_code() == "separate"
        self.phase_height_entry.configure(state="normal" if separate else "disabled")
        self.phase_height_label.configure(state="normal" if separate else "disabled")
        overlay = not separate
        self.overlay_single_check.configure(
            state="normal" if overlay else "disabled"
        )
        self.overlay_height_scale.configure(
            state=(
                "normal"
                if overlay and self.overlay_single_line.get()
                else "disabled"
            )
        )
        state = "disabled" if self._cif_axes_compatible() else "normal"
        self.phase_min_entry.configure(state=state)
        self.phase_max_entry.configure(state=state)

    @staticmethod
    def _cif_error_text(exc: Exception) -> str:
        return localised(
            "The CIF could not be read or calculated. Check the unit cell, "
            "atom sites and explicit symmetry operations.",
            "Le CIF n’a pas pu être lu ou calculé. Vérifiez la maille, "
            "les positions atomiques et les opérations de symétrie explicites.",
            str(exc),
        )

    def _x_axis_title(self, visible_scans: list[PlotItem]) -> str:
        axes = {
            item.scan.axis_name
            for item in visible_scans
            if item.scan is not None
        }
        if len(axes) == 1:
            axis_name = next(iter(axes))
            label = _axis_label(axis_name)
            return f"{label}, °" if _axis_has_degree_units(axis_name) else label
        if len(axes) > 1:
            label = localised(
                "Scan coordinate",
                "Coordonnée du balayage",
                "Координата скана",
            )
            if all(_axis_has_degree_units(name) for name in axes):
                return f"{label}, °"
            return label
        return "2θ, °"

    def _refresh_axis_control(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        self._axis_name_by_label = {}
        if item is None or item.scan is None:
            self.axis_var.set("")
            self.axis_hint.set("")
            self.axis_combo.configure(values=(), state="disabled")
            return

        is_text = item.scan.metadata.get("format") == "XY"
        axes = ("2Theta", "Omega", "Theta", "Chi", "Phi", "X") if is_text else item.scan.available_axes
        self.axis_hint.set(
            "XY: ось X принята за 2θ; при необходимости выберите другую."
            if is_text and item.scan.metadata.get("axis_assumed") else ""
        )
        for axis_name in axes:
            label = _axis_label(axis_name)
            if label in self._axis_name_by_label:
                label = f"{label} ({axis_name})"
            self._axis_name_by_label[label] = axis_name
        labels = tuple(self._axis_name_by_label)
        current_label = next(
            (
                label
                for label, axis_name in self._axis_name_by_label.items()
                if axis_name == item.scan.axis_name
            ),
            labels[0] if labels else "",
        )
        self.axis_combo.configure(
            values=labels,
            state="readonly" if labels else "disabled",
        )
        self.axis_var.set(current_label)

    def change_selected_axis(self, _event=None) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        axis_name = self._axis_name_by_label.get(self.axis_var.get())
        if item is None or item.scan is None or axis_name is None:
            return
        try:
            if item.scan.metadata.get("format") == "XY":
                assign_text_axis(item.scan, axis_name)
            else:
                item.scan.use_axis(axis_name)
        except ValueError as exc:
            messagebox.showerror(
                localised("Axis error", "Erreur d’axe", "Ошибка оси"),
                str(exc),
                parent=self,
            )
            self._refresh_axis_control()
            return
        self.x_min.set("")
        self.x_max.set("")
        self._row_cache.clear()
        self._update_cif_controls()
        self._update_buttons()
        self._draw()
        cif_note = (
            ""
            if self._cif_axes_compatible()
            else localised(
                " CIF uses a separate 2θ axis.",
                " Le CIF utilise un axe 2θ séparé.",
                " CIF отображается по отдельной оси 2θ.",
            )
        )
        self.status.set(
            localised(
                f"{item.name}: X axis changed to {_axis_label(axis_name)}.{cif_note}",
                f"{item.name} : axe X défini sur {_axis_label(axis_name)}.{cif_note}",
                f"{item.name}: выбрана ось X {_axis_label(axis_name)}.{cif_note}",
            )
        )

    def add_scan(
        self,
        scan: Scan1D,
        kind: str = "измерение",
        redraw: bool = True,
        uid: str | None = None,
    ) -> str:
        uid = uid or uuid.uuid4().hex
        if uid in self.items:
            return uid
        scan.kind = kind
        if (scan.metadata.get("format") == "XY" and scan.axis_name == "X"
                and "axis_assumed" not in scan.metadata):
            assign_text_axis(scan, "2Theta", assumed=True)
        self._insert_item(
            PlotItem(uid, scan.name, kind, scan.source, self._next_colour(), scan=scan)
        )
        if redraw:
            self._update_buttons()
            self._draw()
        return uid

    def add_path(self, path: str | Path) -> None:
        source = Path(path)
        if source.suffix.lower() == ".cif":
            self.add_cif(source)
            return
        scans = read_scan_file(source)
        for scan in scans:
            uid = self.add_scan(scan, redraw=False)
            self.tree.selection_set(uid)
        self._update_buttons()
        self._draw()
        self.status.set(
            localised(
                f"Loaded {source.name}: {len(scans)} dataset(s).",
                f"{source.name} chargé : {len(scans)} jeu(x) de données.",
                f"Загружен файл {source.name}: наборов {len(scans)}.",
            )
        )

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Открыть рентгенограммы",
            filetypes=(
                ("Поддерживаемые файлы", "*.xrdml *.xml *.raw *.xy *.txt *.dat *.csv"),
                ("XRDML", "*.xrdml *.xml"),
                ("Bruker RAW", "*.raw"),
                ("Текстовые данные", "*.xy *.txt *.dat *.csv"),
                ("Все файлы", "*.*"),
            ),
        )
        if paths:
            if self.on_import_paths is not None:
                self.on_import_paths(list(paths))
                return
            self._load_many(paths)

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Открыть папку с измерениями")
        if not folder:
            return
        suffixes = {".xrdml", ".xml", ".raw", ".xy", ".txt", ".dat", ".csv"}
        paths = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in suffixes)
        if not paths:
            messagebox.showinfo(
                localised("Empty folder", "Dossier vide", "Папка пуста"),
                localised(
                    "No supported files were found.",
                    "Aucun fichier pris en charge n’a été trouvé.",
                    "Поддерживаемые файлы не найдены.",
                ),
                parent=self,
            )
            return
        if self.on_import_paths is not None:
            self.on_import_paths([str(path) for path in paths])
            return
        self._load_many(paths)

    def _load_many(self, paths) -> None:
        errors: list[str] = []
        loaded = 0
        for path in paths:
            try:
                source = Path(path)
                for scan in read_scan_file(source):
                    uid = self.add_scan(scan, redraw=False)
                    self.tree.selection_set(uid)
                    loaded += 1
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        self._update_buttons()
        self._draw()
        self.status.set(
            localised(
                f"Datasets added: {loaded}.",
                f"Jeux de données ajoutés : {loaded}.",
                f"Добавлено наборов: {loaded}.",
            )
        )
        if errors:
            messagebox.showwarning(
                localised(
                    "Some files could not be read",
                    "Certains fichiers n’ont pas pu être lus",
                    "Не все файлы прочитаны",
                ),
                "\n\n".join(errors[:12]),
                parent=self,
            )

    def open_cif(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Добавить CIF",
            filetypes=(("CIF", "*.cif"), ("Все файлы", "*.*")),
        )
        if path:
            if self.on_import_paths is not None:
                self.on_import_paths([path])
                return
            try:
                self.add_cif(path)
            except Exception as exc:
                messagebox.showerror(
                    "Ошибка CIF", self._cif_error_text(exc), parent=self
                )

    def add_cif(self, path: str | Path) -> str:
        try:
            from .cif_document import load_cif_document
        except ImportError:  # pragma: no cover
            from cif_document import load_cif_document
        return self.add_cif_document(load_cif_document(path))

    def add_cif_document(self, document, uid: str | None = None) -> str:
        return self.add_phase_document(document, uid=uid, kind="CIF")

    def add_phase_document(
        self,
        document,
        uid: str | None = None,
        kind: str | None = None,
    ) -> str:
        source = Path(document.source)
        structure = document.diffraction
        uid = uid or uuid.uuid4().hex
        if uid in self.items:
            return uid
        self._insert_item(
            PlotItem(
                uid=uid,
                name=structure.name or source.stem,
                kind=kind or ("Ячейка" if getattr(document, "is_cell_only", False) else "CIF"),
                source=source,
                colour=self._next_colour(),
                structure=structure,
            )
        )
        self._row_cache.pop(uid, None)
        self.tree.selection_set(uid)
        self._update_buttons()
        self._draw()
        phase_name = structure.name or source.stem
        self.status.set(
            localised(
                f"Added phase {phase_name}.",
                f"Phase {phase_name} ajoutée.",
                f"Добавлена фаза {phase_name}.",
            )
        )
        return uid

    def remove_uid(self, uid: str, *, redraw: bool = True) -> None:
        if uid not in self.items:
            return
        self.viewer_state.remove(uid)
        self._row_cache.pop(uid, None)
        try:
            self.tree.delete(uid)
        except tk.TclError:
            pass
        if self._plot_selection and self._plot_selection[0] == uid:
            self._plot_selection = None
        if redraw:
            self._update_buttons()
            self._draw(preserve_view=True)

    def toggle_selected(self, _event=None) -> None:
        uid = self._selected_uid()
        if uid is None:
            return
        item = self.items[uid]
        visible = self.viewer_state.toggle(uid)
        self.tree.set(uid, "visible", "●" if visible else "○")
        self._update_buttons()
        self._draw(preserve_view=True)

    def choose_colour(self) -> None:
        uid = self._selected_uid()
        if uid is None:
            return
        item = self.items[uid]
        result = colorchooser.askcolor(item.colour, parent=self)
        if result[1]:
            item.colour = result[1]
            self.tree.set(uid, "colour", item.colour)
            self._draw(preserve_view=True)

    def remove_selected(self) -> None:
        uid = self._selected_uid()
        if uid is None:
            return
        self.remove_uid(uid)

    def clear_all(self) -> None:
        """Remove every loaded measurement and CIF phase from the viewer."""

        redraw_job = getattr(self, "_redraw_job", None)
        if redraw_job:
            try:
                self.after_cancel(redraw_job)
            except tk.TclError:
                pass
            self._redraw_job = None
        self.viewer_state.clear()
        self._row_cache.clear()
        children = self.tree.get_children()
        if children:
            self.tree.delete(*children)
        self.axis_var.set("")
        self.offset.set("0")
        for variable in (self.x_min, self.x_max, self.y_min, self.y_max):
            variable.set("")
        self._update_buttons()
        self._draw()
        self.status.set(
            localised(
                "The viewer has been cleared.",
                "La visualisation a été effacée.",
                "Просмотрщик очищен.",
            )
        )

    def rename_selected(self) -> None:
        uid = self._selected_uid()
        if uid is None:
            return
        item = self.items[uid]
        name = simpledialog.askstring(
            localised("Rename dataset", "Renommer le jeu de données", "Переименовать набор"),
            localised("New name:", "Nouveau nom :", "Новое название:"),
            initialvalue=item.name,
            parent=self,
        )
        if name and name.strip():
            item.name = name.strip()
            if item.scan is not None:
                item.scan.name = item.name
            self.tree.item(uid, text=item.name)
            self._draw(preserve_view=True)

    def show_all(self) -> None:
        self.viewer_state.show_all()
        for uid in self.items:
            self.tree.set(uid, "visible", "●")
        self._update_buttons()
        self._draw(preserve_view=True)

    @staticmethod
    def _display_arrays(item: PlotItem) -> tuple[np.ndarray, np.ndarray]:
        return item.display_arrays()

    def _load_processing_controls(self, item: PlotItem | None) -> None:
        scan_item = item if item is not None and item.scan is not None else None
        self._syncing_processing = True
        try:
            if scan_item is None:
                self.processing_name.set("Выберите измерение в списке.")
                values = (0.0, 0.0, 1.0)
                x_limit = y_limit = 1.0
            else:
                assert scan_item.scan is not None
                self.processing_name.set(
                    localised(
                        f"Selected: {scan_item.name}",
                        f"Sélectionné : {scan_item.name}",
                        f"Выбрано: {scan_item.name}",
                    )
                )
                values = (scan_item.x_shift, scan_item.y_shift, scan_item.y_factor)
                x_span = float(np.ptp(scan_item.scan.x))
                y_values = np.asarray(scan_item.scan.y, dtype=float)
                y_span = float(np.ptp(y_values))
                y_size = float(np.max(np.abs(y_values))) if y_values.size else 0.0
                x_limit = max(1.0, 0.1 * x_span, 1.2 * abs(scan_item.x_shift))
                y_limit = max(1.0, y_span, 0.25 * y_size, 1.2 * abs(scan_item.y_shift))
            self.processing_x_slider.configure(from_=-x_limit, to=x_limit)
            self.processing_y_slider.configure(from_=-y_limit, to=y_limit)
            self.processing_x.set(values[0])
            self.processing_y.set(values[1])
            self.processing_factor.set(values[2])
            self.processing_shift_omega.set(
                scan_item.shift_omega if scan_item is not None else True
            )
        finally:
            self._syncing_processing = False

        state = "normal" if scan_item is not None else "disabled"
        for control in self.processing_controls:
            control.configure(state=state)
        self.fit_button.configure(
            state="normal" if scan_item is not None and scan_item.visible else "disabled"
        )
        if scan_item is None or self.on_commit_scan is None:
            self.apply_processing_button.configure(state="disabled")

    def _processing_values_changed(self) -> None:
        if self._syncing_processing:
            return
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.scan is None:
            return
        try:
            x_shift = float(self.processing_x.get())
            y_shift = float(self.processing_y.get())
            y_factor = float(self.processing_factor.get())
            request = CorrectionRequest(
                x_shift=x_shift,
                y_shift=y_shift,
                y_factor=y_factor,
                shift_omega_half=bool(self.processing_shift_omega.get()),
            )
        except (ValueError, XRDDataError, tk.TclError):
            self.status.set(
                localised(
                    "X and Y shifts must be finite numbers; the Y scale must be positive.",
                    "Les décalages X et Y doivent être finis ; l’échelle Y doit être positive.",
                    "Сдвиги X и Y должны быть конечными числами, а масштаб Y — положительным.",
                )
            )
            self._load_processing_controls(item)
            return
        item.x_shift = request.x_shift
        item.y_shift = request.y_shift
        item.y_factor = request.y_factor
        item.shift_omega = request.shift_omega_half
        self._draw(preserve_view=True)

    def _processing_x_changed(self, _value=None) -> None:
        self._processing_values_changed()

    def _processing_y_changed(self, _value=None) -> None:
        self._processing_values_changed()

    def _processing_factor_changed(self, _value=None) -> None:
        self._processing_values_changed()

    def _processing_entry_changed(self, _event=None) -> None:
        self._processing_values_changed()

    def reset_processing(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.scan is None:
            return
        item.reset_transform()
        self._load_processing_controls(item)
        self._draw(preserve_view=True)

    def open_processing_for(self, uid: str) -> None:
        item = self.items.get(uid)
        if item is None or item.scan is None:
            return
        self.tree.selection_set(uid)
        self.tree.see(uid)
        self._update_buttons()
        self.processing_section.expand()
        self.control_panel.after_idle(self.control_panel._resize_content)

    def build_processed_scan(self, uid: str | None = None) -> Scan1D | None:
        uid = uid or self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.scan is None:
            return None
        return apply_correction(
            item.scan,
            CorrectionRequest(
                x_shift=item.x_shift,
                y_shift=item.y_shift,
                y_factor=item.y_factor,
                shift_omega_half=item.shift_omega,
            ),
        )

    @staticmethod
    def _write_processed_xrdml(scan: Scan1D, path: str | Path) -> None:
        """Compatibility entry point for the shared XRDML exporter."""
        write_processed_xrdml(scan, path)

    def _choose_processing_save_path(self, scan: Scan1D) -> str:
        source = Path(scan.source)
        is_xrdml = source.suffix.lower() in {".xrdml", ".xml"}
        extension = source.suffix if is_xrdml else ".xy"
        filetypes = (
            (("XRDML", "*.xrdml *.xml"), ("XY", "*.xy"), ("All files", "*.*"))
            if is_xrdml
            else (("XY", "*.xy"), ("All files", "*.*"))
        )
        return filedialog.asksaveasfilename(
            parent=self,
            title=localised(
                "Save shifted data",
                "Enregistrer les données décalées",
                "Сохранить сдвинутые данные",
            ),
            initialfile=f"{source.stem} shifted{extension}",
            defaultextension=extension,
            filetypes=filetypes,
        )

    @staticmethod
    def _processing_export_error(exc: Exception) -> str:
        if not isinstance(exc, XRDDataError):
            return str(exc)
        messages = {
            "correction_xrdml_source": localised(
                "XRDML export requires an XRDML source file.",
                "L’export XRDML nécessite un fichier source XRDML.",
                "Для экспорта XRDML нужен исходный файл XRDML.",
            ),
            "correction_xrdml_range": localised(
                "The selected XRDML range was not found in the source file.",
                "La plage XRDML sélectionnée est introuvable dans le fichier source.",
                "Выбранный диапазон XRDML не найден в исходном файле.",
            ),
            "correction_xrdml_intensity": localised(
                "The XRDML intensity array was not found.",
                "Le tableau d’intensité XRDML est introuvable.",
                "В XRDML не найден массив интенсивностей.",
            ),
            "correction_xrdml_array_length": localised(
                "The processed and source XRDML arrays have different lengths.",
                "Les tableaux XRDML traité et source ont des longueurs différentes.",
                "Массивы обработанного и исходного XRDML имеют разную длину.",
            ),
            "correction_xrdml_axis_length": localised(
                "An XRDML coordinate axis has an unexpected length.",
                "Un axe de coordonnées XRDML a une longueur inattendue.",
                "Одна из координатных осей XRDML имеет неверную длину.",
            ),
        }
        return messages.get(exc.code, str(exc))

    def save_processing_result(self) -> None:
        scan = self.build_processed_scan()
        if scan is None:
            return
        path_text = self._choose_processing_save_path(scan)
        if not path_text:
            return
        path = Path(path_text)
        try:
            same_as_source = path.resolve() == Path(scan.source).resolve()
        except OSError:
            same_as_source = False
        if same_as_source and not messagebox.askyesno(
            localised("Overwrite source", "Écraser la source", "Перезапись исходника"),
            localised(
                "This is the original measurement file. Overwrite it explicitly?",
                "Il s’agit du fichier de mesure d’origine. Voulez-vous vraiment l’écraser ?",
                "Это исходный файл измерения. Действительно перезаписать его?",
            ),
            parent=self,
        ):
            return
        try:
            write_processed_scan(scan, path)
        except Exception as exc:
            messagebox.showerror(
                localised("Save error", "Erreur d’enregistrement", "Ошибка сохранения"),
                self._processing_export_error(exc),
                parent=self,
            )
            return
        self.status.set(
            localised(
                f"Shifted data saved as {path.name}.",
                f"Les données décalées ont été enregistrées sous {path.name}.",
                f"Сдвинутые данные сохранены в {path.name}.",
            )
        )

    def commit_processing_result(self, mode: str) -> None:
        validate_result_mode(mode)
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        scan = self.build_processed_scan(uid)
        if uid is None or item is None or scan is None or self.on_commit_scan is None:
            return
        self.on_commit_scan(uid, scan, mode)
        # The source returns to its raw display state. The committed object
        # already contains the transformation and must not receive it twice.
        item.reset_transform()
        self._update_buttons()
        self._draw(preserve_view=True)

    def apply_processing_result(self) -> None:
        self.commit_processing_result(self.result_mode.get())

    def _cancel_peak_fit(self) -> None:
        selector = self.rect_selector
        if selector is not None:
            try:
                selector.set_active(False)
            except Exception:
                pass
        self.rect_selector = None
        self.fit_active = False
        self._fit_uid = None

    def activate_peak_fit(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.scan is None:
            return
        if self.toolbar.mode:
            self.status.set(
                localised(
                    "Turn off Pan or Zoom before selecting a peak.",
                    "Désactivez le déplacement ou le zoom avant de sélectionner un pic.",
                    "Отключите перемещение или масштабирование перед выбором пика.",
                )
            )
            return
        self._cancel_peak_fit()
        self._fit_uid = uid
        self.fit_active = True
        self.rect_selector = RectangleSelector(
            self.scan_axis,
            self._fit_peak_rectangle,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords="data",
            interactive=False,
        )
        self.status.set(
            localised(
                "Drag a rectangle around one peak on the main plot.",
                "Tracez un rectangle autour d’un pic sur le graphique principal.",
                "Выделите прямоугольником один пик на основном графике.",
            )
        )

    def _fit_peak_rectangle(self, click, release) -> None:
        uid = self._fit_uid
        item = self.items.get(uid) if uid else None
        self._cancel_peak_fit()
        if (
            item is None
            or item.scan is None
            or click.xdata is None
            or click.ydata is None
            or release.xdata is None
            or release.ydata is None
        ):
            return
        x_low, x_high = sorted((float(click.xdata), float(release.xdata)))
        y_low, y_high = sorted((float(click.ydata), float(release.ydata)))
        x_values, physical_y = self._display_arrays(item)
        plotted_y = self._transformed_y(physical_y, self.y_scale.get())
        visible_scans = self._visible_scans()
        try:
            index = visible_scans.index(item)
            vertical_offset = float(self.offset.get().replace(",", ".") or "0")
            stacking_offset = (len(visible_scans) - index - 1) * vertical_offset
        except (ValueError, tk.TclError):
            stacking_offset = 0.0
        plotted_y = plotted_y + stacking_offset
        mask = (
            np.isfinite(x_values)
            & np.isfinite(plotted_y)
            & (x_values >= x_low)
            & (x_values <= x_high)
            & (plotted_y >= y_low)
            & (plotted_y <= y_high)
        )
        old_x = self.scan_axis.get_xlim()
        old_y = self.scan_axis.get_ylim()
        try:
            fit_x, fit_y, center, intensity = fit_gaussian_peak(
                x_values[mask], physical_y[mask]
            )
        except Exception as exc:
            messagebox.showerror(
                localised("Peak fitting", "Ajustement du pic", "Аппроксимация пика"),
                str(exc),
                parent=self,
            )
            return

        self.scan_axis.scatter(
            x_values[mask],
            plotted_y[mask],
            color="green",
            s=18,
            zorder=7,
        )
        self.scan_axis.plot(
            fit_x,
            self._transformed_y(fit_y, self.y_scale.get()) + stacking_offset,
            color="orange",
            linewidth=2,
            zorder=7,
        )
        self.scan_axis.axvline(center, color="red", linestyle="--", linewidth=1)
        self.scan_axis.scatter(
            [center],
            [
                self._transformed_y(
                    np.asarray([intensity]), self.y_scale.get()
                )[0]
                + stacking_offset
            ],
            color="red",
            s=55,
            zorder=8,
        )
        self.scan_axis.set_xlim(old_x)
        self.scan_axis.set_ylim(old_y)
        self.canvas.draw_idle()

        dialog = CustomInputDialog(self, center, intensity, load_reference_peaks())
        target_text = dialog.result
        if target_text is None or not target_text.strip():
            return
        try:
            target = float(target_text.replace(",", "."))
            if not math.isfinite(target):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                localised("Invalid value", "Valeur incorrecte", "Некорректное значение"),
                localised(
                    "Enter a finite reference coordinate.",
                    "Saisissez une coordonnée de référence finie.",
                    "Введите конечную координату опорного пика.",
                ),
                parent=self,
            )
            return
        shift = target - center
        item.x_shift += shift
        self._load_processing_controls(item)
        self._draw(preserve_view=True)
        self.status.set(
            localised(
                f"Peak centre {center:.6f}° aligned to {target:.6f}°; X shift {item.x_shift:+.6f}°.",
                f"Centre du pic {center:.6f}° aligné sur {target:.6f}° ; décalage X {item.x_shift:+.6f}°.",
                f"Центр пика {center:.6f}° совмещён с {target:.6f}°; сдвиг X {item.x_shift:+.6f}°.",
            )
        )

    def open_peak_database(self) -> None:
        window = tk.Toplevel(self)
        window.title(localised(
            "Reference peaks", "Pics de référence", "Опорные пики"
        ))
        window.geometry("520x420")
        window.transient(self.winfo_toplevel())
        window.grab_set()
        columns = ("name", "value")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        tree.heading("name", text=localised("Name", "Nom", "Название"))
        tree.heading("value", text="2θ")
        tree.column("name", width=300)
        tree.column("value", width=130, anchor="center")
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        for name, value in load_reference_peaks().items():
            tree.insert("", "end", values=(name, value))

        inputs = ttk.Frame(window)
        inputs.pack(fill="x", padx=10)
        name_entry = ttk.Entry(inputs)
        value_entry = ttk.Entry(inputs, width=12)
        name_entry.pack(side="left", fill="x", expand=True)
        value_entry.pack(side="left", padx=5)

        def add_entry() -> None:
            name = name_entry.get().strip()
            try:
                value = float(value_entry.get().replace(",", "."))
                if not name or not math.isfinite(value):
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    localised("Invalid value", "Valeur incorrecte", "Некорректное значение"),
                    localised(
                        "Enter a name and a finite 2θ value.",
                        "Saisissez un nom et une valeur 2θ finie.",
                        "Введите название и конечное значение 2θ.",
                    ),
                    parent=window,
                )
                return
            tree.insert("", "end", values=(name, value))
            name_entry.delete(0, "end")
            value_entry.delete(0, "end")

        ttk.Button(inputs, text="Добавить", command=add_entry).pack(side="left")
        buttons = ttk.Frame(window)
        buttons.pack(fill="x", padx=10, pady=10)
        ttk.Button(
            buttons,
            text="Удалить выбранное",
            command=lambda: [tree.delete(uid) for uid in tree.selection()],
        ).pack(side="left")

        def save_and_close() -> None:
            data: dict[str, float] = {}
            try:
                for uid in tree.get_children():
                    name, value = tree.item(uid, "values")
                    data[str(name)] = float(value)
                save_reference_peaks(data)
            except (OSError, ValueError) as exc:
                messagebox.showerror(
                    localised("Save error", "Erreur d’enregistrement", "Ошибка сохранения"),
                    str(exc),
                    parent=window,
                )
                return
            window.destroy()

        ttk.Button(buttons, text="Сохранить", command=save_and_close).pack(side="right")
        apply_language(window)

    def _update_buttons(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        cif_state = (
            "normal"
            if item and item.structure is not None
            else "disabled"
        )
        phase_state = "normal" if item and item.structure is not None else "disabled"
        self.table_button.configure(state=cif_state)
        self.pole_button.configure(
            state=phase_state if self.on_open_theoretical is not None else "disabled"
        )
        self.clear_button.configure(state="normal" if self.items else "disabled")
        self.visible_button.configure(
            text=translate_text("Скрыть" if item is None or item.visible else "Показать"),
            state="normal" if item else "disabled",
        )
        self.colour_button.configure(state="normal" if item else "disabled")
        self.structure_button.configure(
            state=phase_state if self.on_open_structure is not None else "disabled"
        )
        group = self._group_uids(uid) if uid else []
        index = group.index(uid) if uid in group else -1
        self.up_button.configure(state="normal" if index > 0 else "disabled")
        self.down_button.configure(state="normal" if 0 <= index < len(group)-1 else "disabled")
        self._refresh_axis_control()
        self._update_cif_controls()
        self._load_processing_controls(item)

    def reset_limits(self) -> None:
        for variable in (self.x_min, self.x_max, self.y_min, self.y_max):
            variable.set("")
        self.phase_x_min.set("5")
        self.phase_x_max.set("120")
        self._draw()

    def recalculate(self) -> None:
        self._redraw_job = None
        self._row_cache.clear()
        self._draw(preserve_view=True)

    def _schedule_redraw(self) -> None:
        if hasattr(self, "_redraw_job") and self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(450, self.recalculate)

    def _angle_limits(self) -> tuple[float, float]:
        automatic = scan_x_limits(self.items.values())
        try:
            minimum = (
                float(self.x_min.get().replace(",", "."))
                if self.x_min.get().strip()
                else None
            )
            maximum = (
                float(self.x_max.get().replace(",", "."))
                if self.x_max.get().strip()
                else None
            )
        except ValueError as exc:
            raise ValueError(
                localised(
                    "X limits must be numeric.",
                    "Les limites X doivent être numériques.",
                    "Границы X должны быть числами.",
                )
            ) from exc
        try:
            return resolve_limits(automatic, minimum, maximum)
        except XRDDataError as exc:
            raise ValueError(
                localised(
                    "X min must be lower than X max.",
                    "X min doit être inférieur à X max.",
                    "X min должен быть меньше X max.",
                )
            ) from exc

    def _phase_limits(self) -> tuple[float, float]:
        if self._cif_axes_compatible():
            return self._angle_limits()
        try:
            minimum = float(self.phase_x_min.get().replace(",", "."))
            maximum = float(self.phase_x_max.get().replace(",", "."))
            if not (math.isfinite(minimum) and math.isfinite(maximum) and minimum < maximum):
                raise ValueError
        except ValueError as exc:
            raise ValueError(translate_text("Для CIF нужны конечные числовые границы: min < max.")) from exc
        return minimum, maximum

    def _rows_for(self, item: PlotItem) -> list[ReflectionRow]:
        if item.structure is None:
            return []
        minimum, maximum = self._phase_limits()
        minimum, maximum = max(0.0, minimum), min(179.9, maximum)
        if minimum >= maximum:
            raise ValueError(translate_text("Диапазон CIF должен пересекаться с 0–180°."))
        radiations = self._radiations()
        key = (minimum, maximum, tuple(radiations))
        cached = self._row_cache.get(item.uid)
        if cached is not None and cached[0] == key:
            return cached[1]
        rows = calculate_reflections(
            item.structure,
            self.factors,
            radiations,
            min_two_theta=minimum,
            max_two_theta=maximum,
            min_intensity=0.1,
        )
        self._row_cache[item.uid] = (key, rows)
        return rows

    @staticmethod
    def _transformed_y(y: np.ndarray, mode: str) -> np.ndarray:
        mode_code = choice_code("scale", mode)
        return transformed_intensity(y, mode_code)

    def _phase_profile(
        self,
        rows: list[ReflectionRow],
        minimum: float,
        maximum: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        width = float(self.fwhm.get().replace(",", "."))
        if not math.isfinite(width) or width <= 0:
            raise ValueError(
                localised(
                    "FWHM must be positive.",
                    "La FWHM doit être positive.",
                    "FWHM должен быть положительным.",
                )
            )
        count = max(1200, min(100000, int((maximum - minimum) * 30)))
        grid = np.linspace(minimum, maximum, count)
        sigma = width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        profile = np.zeros_like(grid)
        for row in rows:
            profile += (100.0 if row.intensity is None else row.intensity) * np.exp(
                -0.5 * ((grid - row.two_theta) / sigma) ** 2
            )
        if profile.size and np.nanmax(profile) > 0:
            profile = profile / np.nanmax(profile)
        return grid, profile

    def _draw_separate_phases(
        self,
        phases: list[PlotItem],
        minimum: float,
        maximum: float,
    ) -> None:
        self.phase_axis.set_visible(True)
        self.phase_axis.set_xlabel("2θ, °")
        self.phase_axis.set_ylabel(localised("Phases", "Phases", "Фазы"))
        self.phase_axis.grid(True, axis="x", alpha=0.15)
        self.phase_axis.set_yticks([])
        self.phase_axis.set_xlim(minimum, maximum)
        for index, item in enumerate(phases):
            rows = self._rows_for(item)
            baseline = float(len(phases) - index - 1)
            if choice_code("phase", self.phase_style.get()) == "profile" and not item.structure.cell_only:
                grid, profile = self._phase_profile(rows, minimum, maximum)
                profile = profile * 0.75
                self.phase_axis.plot(
                    grid, baseline + profile, color=item.colour, lw=1
                )
                self.phase_axis.fill_between(
                    grid,
                    baseline,
                    baseline + profile,
                    color=item.colour,
                    alpha=0.16,
                )
            else:
                for row in rows:
                    height = 0.85 if row.intensity is None else 0.15 + 0.7 * row.intensity / 100.0
                    self.phase_axis.vlines(
                        row.two_theta,
                        baseline,
                        baseline + height,
                        color=item.colour,
                        lw=1,
                    )
            self.phase_axis.text(
                0.01,
                baseline + 0.87,
                item.name,
                color=item.colour,
                fontsize=8,
                ha="left",
                va="top",
                transform=self.phase_axis.get_yaxis_transform(),
                clip_on=True,
            )
        self.phase_axis.set_ylim(-0.1, len(phases) + 0.05)

    def _draw_overlay_phases(
        self,
        phases: list[PlotItem],
        minimum: float,
        maximum: float,
    ) -> None:
        single_line = self.viewer_state.plot.overlay_single_line
        bands, occupied_height = overlay_phase_geometry(
            len(phases),
            single_line=single_line,
            height_percent=self.viewer_state.plot.overlay_height_percent,
        )
        transform = self.scan_axis.get_xaxis_transform()
        for index, (item, geometry) in enumerate(zip(phases, bands)):
            rows = self._rows_for(item)
            baseline, amplitude = geometry
            if choice_code("phase", self.phase_style.get()) == "profile" and not item.structure.cell_only:
                grid, profile = self._phase_profile(rows, minimum, maximum)
                profile = profile * amplitude
                self.scan_axis.plot(
                    grid,
                    baseline + profile,
                    color=item.colour,
                    lw=1,
                    transform=transform,
                    clip_on=True,
                )
                self.scan_axis.fill_between(
                    grid,
                    baseline,
                    baseline + profile,
                    color=item.colour,
                    alpha=0.16,
                    transform=transform,
                    clip_on=True,
                )
            else:
                for row in rows:
                    height = amplitude if row.intensity is None else amplitude * (0.18 + 0.82 * row.intensity / 100.0)
                    self.scan_axis.vlines(
                        row.two_theta,
                        baseline,
                        baseline + height,
                        color=item.colour,
                        lw=1,
                        transform=transform,
                        clip_on=True,
                    )
            if single_line:
                label_x = (index + 0.5) / len(phases)
                if occupied_height <= 0.9:
                    label_y = occupied_height + 0.01
                    vertical_alignment = "bottom"
                else:
                    label_y = 0.99
                    vertical_alignment = "top"
                horizontal_alignment = "center"
            else:
                band_height = amplitude / 0.75
                label_x = 0.01
                label_y = baseline + band_height * 0.9
                vertical_alignment = "top"
                horizontal_alignment = "left"
            self.scan_axis.text(
                label_x,
                label_y,
                item.name,
                color=item.colour,
                fontsize=8,
                ha=horizontal_alignment,
                va=vertical_alignment,
                transform=self.scan_axis.transAxes,
                clip_on=True,
            )
        self._overlay_phase_top = occupied_height

    def _draw(self, *, preserve_view: bool = False) -> None:
        if self.fit_active:
            self._cancel_peak_fit()
        old_limits = (self.scan_axis.get_xlim(), self.scan_axis.get_ylim(), self.phase_axis.get_xlim())
        self._axes_linked = self._cif_axes_compatible()
        self.viewer_state.plot.phase_layout = self._phase_layout_code()
        self.viewer_state.plot.phase_style = choice_code("phase", self.phase_style.get())
        self.viewer_state.plot.intensity_scale = choice_code("scale", self.y_scale.get())
        self.viewer_state.plot.overlay_single_line = bool(
            self.overlay_single_line.get()
        )
        self.viewer_state.plot.set_overlay_height(float(self.overlay_height.get()))
        if self._phase_layout_code() == "overlay" and not self._axes_linked:
            self.phase_layout.set(translate_text("Отдельно"))
            self.status.set(
                localised(
                    "CIF overlay is available only for measurements on the 2θ axis.",
                    "La superposition CIF est disponible uniquement pour les mesures sur l’axe 2θ.",
                    "Наложение CIF доступно только для измерений по оси 2θ.",
                )
            )
            self._update_cif_controls()
            self.viewer_state.plot.phase_layout = "separate"
        self.scan_axis.clear()
        self.phase_axis.clear()
        self._overlay_phase_top = 0.0
        self._update_phase_panel()
        try:
            vertical_offset = float(self.offset.get().replace(",", ".") or "0")
            self.viewer_state.plot.vertical_offset = vertical_offset
            x_min, x_max = self._angle_limits()
        except ValueError as exc:
            self.status.set(str(exc))
            self.canvas.draw_idle()
            return

        visible_scans = self._visible_scans()
        x_axis_title = self._x_axis_title(visible_scans)
        scale_code = choice_code("scale", self.y_scale.get())
        self.scan_axis.set_yscale("log" if scale_code == "log" else "linear")
        for index, item in enumerate(visible_scans):
            assert item.scan is not None
            x_values, physical_y = self._display_arrays(item)
            y_values = self._transformed_y(physical_y, self.y_scale.get())
            self.scan_axis.plot(
                x_values,
                y_values + (len(visible_scans) - index - 1) * vertical_offset,
                color=item.colour,
                linewidth=1.15,
                label=item.name,
                picker=5,
            )
        y_titles = {
            "log": localised(
                "Intensity – logarithmic scale",
                "Intensité – échelle logarithmique",
                "Интенсивность — логарифмическая шкала",
            ),
            "sqrt": localised(
                "Square root of intensity",
                "Racine carrée de l’intensité",
                "Квадратный корень интенсивности",
            ),
            "square": localised(
                "Intensity^2",
                "Intensité^2",
                "Интенсивность^2",
            ),
        }
        self.scan_axis.set_ylabel(
            y_titles.get(scale_code, localised("Intensity", "Intensité", "Интенсивность"))
        )
        self.scan_axis.grid(True, alpha=0.2)
        self.scan_axis.set_xlim(x_min, x_max)
        if 0 < len(visible_scans) <= MAX_LEGEND_ITEMS:
            self.scan_axis.legend(loc="best", fontsize=8)
        elif visible_scans:
            self.scan_axis.text(
                0.995,
                0.99,
                localised(
                    f"{len(visible_scans)} visible datasets – legend hidden",
                    f"{len(visible_scans)} jeux visibles – légende masquée",
                    f"Видимых наборов: {len(visible_scans)} – легенда скрыта",
                ),
                transform=self.scan_axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#666666",
            )
        else:
            self.scan_axis.text(
                0.5,
                0.5,
                localised(
                    "Open a measurement file",
                    "Ouvrez un fichier de mesure",
                    "Откройте файл измерения",
                ),
                transform=self.scan_axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )

        try:
            if self.y_min.get().strip() or self.y_max.get().strip():
                bottom, top = self.scan_axis.get_ylim()
                if self.y_min.get().strip():
                    bottom = float(self.y_min.get().replace(",", "."))
                if self.y_max.get().strip():
                    top = float(self.y_max.get().replace(",", "."))
                self.scan_axis.set_ylim(bottom, top)
        except ValueError:
            self.status.set(
                localised(
                    "Y limits must be numeric.",
                    "Les limites Y doivent être numériques.",
                    "Границы Y должны быть числами.",
                )
            )

        self._navigation_x_bounds = (x_min, x_max)
        self._navigation_y_bounds = tuple(sorted(self.scan_axis.get_ylim()))

        phases = self.viewer_state.visible_phases()
        if phases:
            try:
                phase_min, phase_max = self._phase_limits()
                if self._phase_layout_code() == "overlay":
                    self._draw_overlay_phases(phases, phase_min, phase_max)
                    self.scan_axis.set_xlabel(x_axis_title)
                else:
                    self._draw_separate_phases(phases, phase_min, phase_max)
            except Exception as exc:
                self.status.set(
                    localised(
                        f"Could not calculate the phase: {self._cif_error_text(exc)}",
                        f"Impossible de calculer la phase : {self._cif_error_text(exc)}",
                        f"Не удалось рассчитать фазу: {exc}",
                    )
                )
        else:
            self.phase_axis.set_visible(False)
            self.scan_axis.set_xlabel(x_axis_title)
        if phases and self._phase_layout_code() == "separate" and not self._axes_linked:
            self.scan_axis.set_xlabel(x_axis_title)
            self.phase_axis.set_title(translate_text("CIF: отдельная ось 2θ"), fontsize=9)
        if preserve_view:
            self.scan_axis.set_xlim(old_limits[0])
            self.scan_axis.set_ylim(old_limits[1])
            self.phase_axis.set_xlim(
                old_limits[0] if self._axes_linked else old_limits[2]
            )
        self._refresh_selection_info()
        self._update_phase_panel()
        # clear() recreates the callbacks registry, so reconnect after drawing.
        self.scan_axis.callbacks.connect(
            "xlim_changed", lambda axis: self._axis_limits_changed(axis, "x")
        )
        self.scan_axis.callbacks.connect(
            "ylim_changed", lambda axis: self._axis_limits_changed(axis, "y")
        )
        self.phase_axis.callbacks.connect(
            "xlim_changed", lambda axis: self._axis_limits_changed(axis, "x")
        )
        self._update_navigation_scrollbars()
        self.canvas.draw_idle()

    def _nearest_phase_reflection(
        self, x_value: float, x_span: float
    ) -> tuple[PlotItem, ReflectionRow] | None:
        nearest: tuple[float, PlotItem, ReflectionRow] | None = None
        for item in self.items.values():
            if not item.visible or item.structure is None:
                continue
            for row in self._rows_for(item):
                distance = abs(row.two_theta - x_value)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, item, row)
        if nearest and nearest[0] <= max(0.15, x_span / 150.0):
            return nearest[1], nearest[2]
        return None

    def _select_phase_reflection(self, item: PlotItem, row: ReflectionRow) -> None:
        self._plot_selection = (item.uid, "cif", (row.hkl, row.d, row.radiation))
        intensity = _reflection_intensity_text(row)
        self.selection_info.set(
            f"{item.name}: {row.hkl}, d = {row.d:.5f} Å, "
            f"2θ = {row.two_theta:.5f}°, {intensity}"
        )

    def _event_y_fraction(self, event) -> float | None:
        if getattr(event, "y", None) is not None:
            return float(
                self.scan_axis.transAxes.inverted().transform(
                    (self.scan_axis.bbox.x0, event.y)
                )[1]
            )
        if getattr(event, "ydata", None) is not None:
            bottom, top = self.scan_axis.get_ylim()
            if not math.isclose(bottom, top):
                return float((event.ydata - bottom) / (top - bottom))
        return None

    def _plot_click(self, event) -> None:
        if event.xdata is None:
            return
        if event.inaxes is self.scan_axis:
            if self._phase_layout_code() == "overlay" and self._overlay_phase_top > 0:
                fraction = self._event_y_fraction(event)
                span = abs(self.scan_axis.get_xlim()[1] - self.scan_axis.get_xlim()[0])
                reflection = (
                    self._nearest_phase_reflection(event.xdata, span)
                    if fraction is not None and fraction <= self._overlay_phase_top
                    else None
                )
                if reflection is not None:
                    self._select_phase_reflection(*reflection)
                    return
            candidates: list[tuple[float, PlotItem, int]] = []
            for item in self.items.values():
                if not item.visible or item.scan is None:
                    continue
                x_values, _physical_y = self._display_arrays(item)
                index = int(np.argmin(np.abs(x_values - event.xdata)))
                candidates.append((abs(float(x_values[index]) - event.xdata), item, index))
            if candidates:
                _distance, item, index = min(candidates, key=lambda value: value[0])
                assert item.scan is not None
                x_values, physical_y = self._display_arrays(item)
                axis_label = _axis_label(item.scan.axis_name)
                unit = "°" if _axis_has_degree_units(item.scan.axis_name) else ""
                self._plot_selection = (item.uid, "scan", index)
                self.selection_info.set(
                    f"{item.name}: {axis_label} = {x_values[index]:.5f}{unit}, "
                    f"I = {physical_y[index]:.6g}"
                )
        elif event.inaxes is self.phase_axis:
            span = abs(self.phase_axis.get_xlim()[1] - self.phase_axis.get_xlim()[0])
            reflection = self._nearest_phase_reflection(event.xdata, span)
            if reflection is not None:
                self._select_phase_reflection(*reflection)

    def _refresh_selection_info(self) -> None:
        selection = self._plot_selection
        item = self.items.get(selection[0]) if selection else None
        if item is None or not item.visible:
            self._plot_selection = None
            self.selection_info.set("Выберите точку или отражение на графике.")
            return
        _uid, kind, identity = selection
        if kind == "scan" and item.scan is not None:
            index = identity
            x_values, physical_y = self._display_arrays(item)
            unit = "°" if _axis_has_degree_units(item.scan.axis_name) else ""
            self.selection_info.set(
                f"{item.name}: {_axis_label(item.scan.axis_name)} = {x_values[index]:.5f}{unit}, "
                f"I = {physical_y[index]:.6g}")
            return
        if kind == "cif" and item.structure is not None:
            hkl, spacing, radiation = identity
            try:
                row = next((r for r in self._rows_for(item)
                            if r.hkl == hkl and r.radiation == radiation
                            and math.isclose(r.d, spacing, rel_tol=1e-7)), None)
            except Exception:
                row = None  # Never keep stale numeric values after an invalid edit.
            if row is not None:
                self.selection_info.set(
                    f"{item.name}: {row.hkl}, d = {row.d:.5f} Å, "
                    f"2θ = {row.two_theta:.5f}°, {_reflection_intensity_text(row)}"
                )
                return
            self.selection_info.set("Выбранное отражение недоступно при текущих параметрах.")

    def open_reflection_table(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.structure is None:
            return
        if self.on_open_reflection_table is not None:
            self.on_open_reflection_table(
                item.uid if item.structure.cell_only else str(item.source)
            )
            return
        try:
            rows = self._rows_for(item)
        except Exception as exc:
            messagebox.showerror(
                "Ошибка расчёта", self._cif_error_text(exc), parent=self
            )
            return
        ReflectionTableWindow(self, item.name, rows)

    def open_theoretical(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item and item.structure is not None and self.on_open_theoretical is not None:
            self.on_open_theoretical(
                item.uid if item.structure.cell_only else str(item.source)
            )

    def open_structure(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item and item.structure is not None and self.on_open_structure is not None:
            self.on_open_structure(
                item.uid if item.structure.cell_only else str(item.source)
            )
