"""Embedded CIF reflection-angle table."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

try:
    from .cif_xrd import (
        CSV_COLUMNS,
        RADIATIONS,
        ReflectionRow,
        Structure,
        _data_path,
        calculate_reflections,
        export_csv,
        load_scattering_factors,
        read_structure,
    )
    from .i18n import apply_language, filedialog, localised, messagebox
except ImportError:
    from cif_xrd import (
        CSV_COLUMNS,
        RADIATIONS,
        ReflectionRow,
        Structure,
        _data_path,
        calculate_reflections,
        export_csv,
        load_scattering_factors,
        read_structure,
    )
    from i18n import apply_language, filedialog, localised, messagebox


class ReflectionTablePage(ttk.Frame):
    """Calculate and display powder reflection angles independently of the viewer."""

    COLUMNS = ("hkl", "d", "tt", "line", "wave", "weight", "mult", "f2", "int")

    def __init__(
        self,
        parent: tk.Misc,
        on_open_cif=None,
        radiations_provider: Callable[[], list[tuple[str, float, float]]] | None = None,
    ) -> None:
        super().__init__(parent, padding=10)
        self.on_open_cif = on_open_cif
        self.radiations_provider = radiations_provider or (lambda: list(RADIATIONS))
        self.cif_document = None
        self.structure: Structure | None = None
        self.rows: list[ReflectionRow] = []
        self.factors = load_scattering_factors(_data_path())
        self.path_var = tk.StringVar()
        self.min_angle = tk.StringVar(value="5")
        self.max_angle = tk.StringVar(value="120")
        self.min_intensity = tk.StringVar(value="0.1")
        self.status = tk.StringVar(
            value=localised(
                "Open a CIF file.",
                "Ouvrez un fichier CIF.",
                "Откройте CIF-файл.",
            )
        )
        self.summary_text = self.status.get()
        self._build()
        apply_language(self)

    def _build(self) -> None:
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        file_frame = ttk.Frame(self)
        file_frame.grid(row=0, column=0, sticky="ew")
        file_frame.columnconfigure(0, weight=1)
        ttk.Entry(
            file_frame,
            textvariable=self.path_var,
            state="readonly",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            file_frame,
            text="Открыть CIF…",
            command=self.open_dialog,
        ).grid(row=0, column=1, padx=(8, 0))

        settings = ttk.LabelFrame(self, text="Расчёт", padding=8)
        settings.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        ttk.Label(settings, text="2θ от").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.min_angle, width=8).grid(
            row=0, column=1, padx=(5, 12)
        )
        ttk.Label(settings, text="до").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.max_angle, width=8).grid(
            row=0, column=3, padx=(5, 12)
        )
        ttk.Label(settings, text="Iотн ≥, %").grid(row=0, column=4, sticky="w")
        self.min_intensity_entry = ttk.Entry(settings, textvariable=self.min_intensity, width=8)
        self.min_intensity_entry.grid(
            row=0, column=5, padx=(5, 18)
        )

        ttk.Button(
            settings,
            text="Рассчитать",
            command=self.calculate,
        ).grid(row=0, column=6, padx=(4, 0))

        table_frame = ttk.Frame(self)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table = ttk.Treeview(
            table_frame,
            columns=self.COLUMNS,
            show="headings",
        )
        widths = {
            "hkl": 190,
            "d": 90,
            "tt": 90,
            "line": 85,
            "wave": 75,
            "weight": 60,
            "mult": 80,
            "f2": 110,
            "int": 90,
        }
        for column, heading in zip(self.COLUMNS, CSV_COLUMNS):
            self.table.heading(
                column,
                text=heading,
                command=lambda selected=column: self.sort_table(selected, False),
            )
            self.table.column(column, width=widths[column], anchor="center")
        y_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.table.yview,
        )
        x_scroll = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.table.xview,
        )
        self.table.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        self.table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.table.bind("<<TreeviewSelect>>", self._selection_changed)

        bottom = ttk.Frame(self)
        bottom.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        details_frame = ttk.Frame(bottom)
        details_frame.grid(row=0, column=0, sticky="ew")
        details_frame.columnconfigure(0, weight=1)
        self.details = tk.Text(
            details_frame,
            height=3,
            wrap="word",
            relief="flat",
            borderwidth=0,
            padx=4,
            pady=3,
            state="disabled",
        )
        details_scroll = ttk.Scrollbar(
            details_frame, orient="vertical", command=self.details.yview
        )
        self.details.configure(yscrollcommand=details_scroll.set)
        self.details.grid(row=0, column=0, sticky="ew")
        details_scroll.grid(row=0, column=1, sticky="ns")
        self.details.bind("<Button-1>", lambda _event: self.details.focus_set())
        self.save_button = ttk.Button(
            bottom,
            text="Сохранить CSV…",
            command=self.save_csv,
            state="disabled",
        )
        self.save_button.grid(row=0, column=1, sticky="ne", padx=(8, 0))
        self._set_details(self.summary_text)

    def localize_content(self) -> None:
        apply_language(self)
        if self.structure is not None:
            self.calculate()
        else:
            self._set_summary(
                localised("Open a CIF file.", "Ouvrez un fichier CIF.", "Откройте CIF-файл.")
            )

    def open_dialog(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Открыть CIF",
            filetypes=(
                ("Crystallographic Information File", "*.cif"),
                ("Все файлы", "*.*"),
            ),
        )
        if path:
            if self.on_open_cif is not None:
                self.on_open_cif(path)
                return
            self.load_cif(path)

    def load_cif(self, path: str | Path) -> None:
        try:
            try:
                from .cif_document import load_cif_document
            except ImportError:  # pragma: no cover
                from cif_document import load_cif_document
            self.load_document(load_cif_document(path))
        except Exception as exc:
            messagebox.showerror(
                localised(
                    "CIF read error",
                    "Erreur de lecture du CIF",
                    "Ошибка чтения CIF",
                ),
                str(exc),
                parent=self,
            )

    def load_document(self, document) -> None:
        self.cif_document = document
        self.structure = document.diffraction
        self.path_var.set(str(document.source))
        self.min_intensity_entry.configure(
            state="disabled" if self.structure.cell_only else "normal"
        )
        self.calculate()

    def clear_document(self) -> None:
        self.cif_document = None
        self.structure = None
        self.rows = []
        self.path_var.set("")
        self.min_intensity_entry.configure(state="normal")
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        self.save_button.configure(state="disabled")
        self._set_summary(
            localised("Open a CIF file.", "Ouvrez un fichier CIF.", "Откройте CIF-файл.")
        )

    def selected_radiations(self) -> list[tuple[str, float, float]]:
        return list(self.radiations_provider())

    def radiation_changed(self) -> None:
        if self.structure is not None:
            self.calculate()

    def calculate(self) -> None:
        if self.structure is None:
            messagebox.showinfo(
                localised("No structure", "Aucune structure", "Нет структуры"),
                localised(
                    "Open a CIF file first.",
                    "Ouvrez d’abord un fichier CIF.",
                    "Сначала откройте CIF-файл.",
                ),
                parent=self,
            )
            return
        try:
            try:
                minimum = float(self.min_angle.get().replace(",", "."))
                maximum = float(self.max_angle.get().replace(",", "."))
                threshold = float(self.min_intensity.get().replace(",", "."))
            except ValueError as exc:
                raise ValueError(
                    localised(
                        "The angle limits and intensity threshold must be numeric.",
                        "Les limites angulaires et le seuil d’intensité doivent être numériques.",
                        "Границы углов и порог интенсивности должны быть числами.",
                    )
                ) from exc
            self.rows = calculate_reflections(
                self.structure,
                self.factors,
                self.selected_radiations(),
                minimum,
                maximum,
                threshold,
            )
        except Exception as exc:
            messagebox.showerror(
                localised(
                    "Calculation error",
                    "Erreur de calcul",
                    "Ошибка расчёта",
                ),
                str(exc),
                parent=self,
            )
            return

        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        for index, row in enumerate(self.rows):
            self.table.insert(
                "",
                "end",
                iid=f"row:{index}",
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
        self.save_button.configure(state="normal" if self.rows else "disabled")
        if self.structure.cell_only:
            summary = localised(
                f"{self.structure.name}: atom coordinates are not specified; "
                f"{len(self.rows)} reflections are not systematically forbidden "
                "by the space group. F^2 and intensity are unavailable.",
                f"{self.structure.name} : les coordonnées atomiques ne sont pas "
                f"indiquées ; {len(self.rows)} réflexions ne sont pas interdites "
                "systématiquement par le groupe d’espace. F^2 et l’intensité sont indisponibles.",
                f"{self.structure.name}: координаты атомов не заданы; отражений, "
                f"не запрещённых систематически пространственной группой, — {len(self.rows)}. "
                "F^2 и интенсивность недоступны.",
            )
        else:
            summary = localised(
                f"{self.structure.name}: {len(self.structure.atoms)} atoms after "
                f"symmetry expansion; {len(self.rows)} table rows. Intensities "
                "are calculated for a powder pattern.",
                f"{self.structure.name} : {len(self.structure.atoms)} atomes après "
                f"développement de la symétrie ; {len(self.rows)} lignes. Les "
                "intensités sont calculées pour un diagramme de poudre.",
                f"{self.structure.name}: атомов после размножения симметрией – "
                f"{len(self.structure.atoms)}; строк в таблице – {len(self.rows)}. "
                "Интенсивности расчётные, для порошковой дифрактограммы.",
            )
        self._set_summary(summary)

    def save_csv(self) -> None:
        if not self.rows:
            return
        default_name = Path(self.path_var.get()).stem + "_calculated_xrd.csv"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить таблицу",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=(("CSV", "*.csv"), ("Все файлы", "*.*")),
        )
        if not path:
            return
        try:
            export_csv(path, self.rows)
            self._set_summary(
                localised(
                    f"Table saved: {path}",
                    f"Table enregistrée : {path}",
                    f"Таблица сохранена: {path}",
                )
            )
        except OSError as exc:
            messagebox.showerror(
                localised(
                    "Save error",
                    "Erreur d’enregistrement",
                    "Ошибка сохранения",
                ),
                str(exc),
                parent=self,
            )

    def _set_details(self, text: str) -> None:
        self.status.set(text)
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def _set_summary(self, text: str) -> None:
        self.summary_text = text
        self._set_details(text)

    def _selection_changed(self, _event=None) -> None:
        selection = self.table.selection()
        if not selection:
            self._set_details(self.summary_text)
            return
        try:
            index = int(selection[0].split(":", 1)[1])
            row = self.rows[index]
        except (IndexError, ValueError):
            self._set_details(self.summary_text)
            return
        if row.equivalents:
            full_list = "+".join(f"({h} {k} {l})" for h, k, l in row.equivalents)
        else:
            full_list = row.hkl
        prefix = localised(
            "Equivalent hkl",
            "hkl équivalents",
            "Эквивалентные hkl",
        )
        self._set_details(f"{prefix}: {full_list}")

    def sort_table(self, column: str, reverse: bool) -> None:
        items = [
            (self.table.set(item, column), item)
            for item in self.table.get_children("")
        ]

        def key(item: tuple[str, str]) -> object:
            try:
                return float(item[0])
            except ValueError:
                return item[0]

        items.sort(key=key, reverse=reverse)
        for index, (_value, item) in enumerate(items):
            self.table.move(item, "", index)
        self.table.heading(
            column,
            command=lambda: self.sort_table(column, not reverse),
        )
