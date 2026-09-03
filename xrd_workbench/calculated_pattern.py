"""Calculated powder diffraction pattern for a shared CIF document."""

from __future__ import annotations

import math
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from .cif_xrd import (
        ReflectionRow,
        Structure,
        _data_path,
        calculate_reflections,
        load_scattering_factors,
    )
    from .i18n import LocalizedStringVar, apply_language, localised, messagebox
except ImportError:  # pragma: no cover
    from cif_xrd import (
        ReflectionRow,
        Structure,
        _data_path,
        calculate_reflections,
        load_scattering_factors,
    )
    from i18n import LocalizedStringVar, apply_language, localised, messagebox


class CalculatedPatternPage(ttk.Frame):
    """Plot sticks or a Gaussian profile from the existing CIF reflection engine."""

    def __init__(self, parent: tk.Misc, radiations_provider) -> None:
        super().__init__(parent, padding=10)
        self.radiations_provider = radiations_provider
        self.cif_document = None
        self.structure: Structure | None = None
        self.rows: list[ReflectionRow] = []
        self.factors = load_scattering_factors(_data_path())
        self.path_var = LocalizedStringVar(value="CIF не открыт")
        self.min_angle = tk.StringVar(value="5")
        self.max_angle = tk.StringVar(value="120")
        self.min_intensity = tk.StringVar(value="0.1")
        self.fwhm = tk.StringVar(value="0.15")
        self.style = tk.StringVar(value="profile")
        self.status = tk.StringVar(value="")
        self._build()
        self.localize_content()

    def _build(self) -> None:
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        file_row = ttk.Frame(self)
        file_row.grid(row=0, column=0, sticky="ew")
        file_row.columnconfigure(0, weight=1)
        ttk.Label(file_row, textvariable=self.path_var).grid(row=0, column=0, sticky="w")

        settings = ttk.LabelFrame(self, text="Расчёт", padding=8)
        settings.grid(row=1, column=0, sticky="ew", pady=(8, 8))
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
            row=0, column=5, padx=(5, 12)
        )
        ttk.Label(settings, text="FWHM, °").grid(row=0, column=6, sticky="w")
        self.fwhm_entry = ttk.Entry(settings, textvariable=self.fwhm, width=8)
        self.fwhm_entry.grid(row=0, column=7, padx=(5, 12))
        self.sticks_radio = ttk.Radiobutton(
            settings,
            text="Штрихи",
            value="sticks",
            variable=self.style,
            command=self.redraw,
        )
        self.sticks_radio.grid(row=0, column=8, padx=(0, 6))
        self.profile_radio = ttk.Radiobutton(
            settings,
            text="Профиль",
            value="profile",
            variable=self.style,
            command=self.redraw,
        )
        self.profile_radio.grid(row=0, column=9, padx=(0, 8))
        ttk.Button(settings, text="Рассчитать", command=self.calculate).grid(
            row=0, column=10
        )

        plot = ttk.Frame(self)
        plot.grid(row=2, column=0, sticky="nsew")
        plot.rowconfigure(0, weight=1)
        plot.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(9, 5), dpi=100, constrained_layout=True)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_host = ttk.Frame(plot)
        toolbar_host.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_host, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left")
        ttk.Label(toolbar_host, textvariable=self.status).pack(
            side="left", fill="x", expand=True, padx=8
        )

    def load_document(self, document) -> None:
        self.cif_document = document
        self.structure = document.diffraction
        self.path_var.set(Path(document.source).name)
        cell_only = self.structure.cell_only
        if cell_only:
            self.style.set("sticks")
        state = "disabled" if cell_only else "normal"
        self.min_intensity_entry.configure(state=state)
        self.fwhm_entry.configure(state=state)
        self.profile_radio.configure(state=state)
        self.calculate()

    def clear_document(self) -> None:
        self.cif_document = None
        self.structure = None
        self.rows = []
        self.path_var.set(localised("No CIF loaded", "Aucun CIF chargé", "CIF не открыт"))
        for widget in (self.min_intensity_entry, self.fwhm_entry, self.profile_radio):
            widget.configure(state="normal")
        self.status.set("")
        self.redraw()

    def radiation_changed(self) -> None:
        if self.structure is not None:
            self.calculate(show_errors=True)

    def _parameters(self) -> tuple[float, float, float, float]:
        try:
            minimum = float(self.min_angle.get().replace(",", "."))
            maximum = float(self.max_angle.get().replace(",", "."))
            threshold = float(self.min_intensity.get().replace(",", "."))
            fwhm = float(self.fwhm.get().replace(",", "."))
        except ValueError as exc:
            raise ValueError(
                localised(
                    "The angle limits, intensity threshold and FWHM must be numeric.",
                    "Les limites angulaires, le seuil d’intensité et la FWHM doivent être numériques.",
                    "Границы углов, порог интенсивности и FWHM должны быть числами.",
                )
            ) from exc
        if not math.isfinite(fwhm) or fwhm <= 0:
            raise ValueError(
                localised(
                    "FWHM must be a positive finite number.",
                    "La FWHM doit être un nombre fini positif.",
                    "FWHM должен быть положительным конечным числом.",
                )
            )
        return minimum, maximum, threshold, fwhm

    def calculate(self, show_errors: bool = True) -> None:
        if self.structure is None:
            if show_errors:
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
            minimum, maximum, threshold, _fwhm = self._parameters()
            self.rows = calculate_reflections(
                self.structure,
                self.factors,
                self.radiations_provider(),
                minimum,
                maximum,
                threshold,
            )
        except Exception as exc:
            if show_errors:
                messagebox.showerror(
                    localised("Calculation error", "Erreur de calcul", "Ошибка расчёта"),
                    str(exc),
                    parent=self,
                )
            return
        if self.structure.cell_only:
            self.status.set(
                localised(
                    f"Reference positions: {len(self.rows)}; intensity is unavailable.",
                    f"Positions de référence : {len(self.rows)} ; intensité indisponible.",
                    f"Положений отражений: {len(self.rows)}; интенсивность недоступна.",
                )
            )
        else:
            self.status.set(
                localised(
                    f"Calculated reflections: {len(self.rows)}",
                    f"Réflexions calculées : {len(self.rows)}",
                    f"Рассчитано отражений: {len(self.rows)}",
                )
            )
        self.redraw()

    def redraw(self) -> None:
        self.axis.clear()
        self.axis.set_xlabel("2θ, °")
        cell_only = bool(self.structure is not None and self.structure.cell_only)
        self.axis.set_ylabel(
            localised("Reference positions", "Positions de référence", "Положения отражений")
            if cell_only
            else localised("Relative intensity, %", "Intensité relative, %", "Относительная интенсивность, %")
        )
        self.axis.grid(True, alpha=0.2)
        if self.structure is None:
            self.axis.text(
                0.5,
                0.5,
                localised("Open a CIF file.", "Ouvrez un fichier CIF.", "Откройте CIF-файл."),
                transform=self.axis.transAxes,
                ha="center",
                va="center",
            )
            self.canvas.draw_idle()
            return
        try:
            minimum, maximum, _threshold, fwhm = self._parameters()
        except ValueError:
            minimum, maximum, fwhm = 5.0, 120.0, 0.15
        self.axis.set_xlim(minimum, maximum)
        self.axis.set_ylim(0, 105)
        if self.rows:
            grouped: dict[str, list[ReflectionRow]] = defaultdict(list)
            for row in self.rows:
                grouped[row.radiation].append(row)
            if self.style.get() == "sticks" or cell_only:
                for radiation, rows in grouped.items():
                    x = [row.two_theta for row in rows]
                    y = [100.0 if row.intensity is None else row.intensity for row in rows]
                    self.axis.vlines(x, 0, y, linewidth=1.2, label=radiation)
            else:
                grid = np.linspace(minimum, maximum, 5000)
                sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
                components: dict[str, np.ndarray] = {}
                for radiation, rows in grouped.items():
                    profile = np.zeros_like(grid)
                    for row in rows:
                        profile += float(row.intensity or 0.0) * np.exp(
                            -0.5 * ((grid - row.two_theta) / sigma) ** 2
                        )
                    components[radiation] = profile
                total = sum(components.values(), np.zeros_like(grid))
                maximum_profile = float(np.max(total)) if total.size else 0.0
                if maximum_profile > 0:
                    scale = 100.0 / maximum_profile
                    total *= scale
                    for radiation in components:
                        components[radiation] *= scale
                if len(components) > 1:
                    for radiation, profile in components.items():
                        self.axis.plot(
                            grid,
                            profile,
                            linewidth=0.8,
                            linestyle="--",
                            alpha=0.55,
                            label=radiation,
                        )
                    self.axis.plot(
                        grid,
                        total,
                        color="#202020",
                        linewidth=1.4,
                        label=localised("Total", "Somme", "Сумма"),
                    )
                else:
                    self.axis.plot(grid, total, linewidth=1.3)
            if len(grouped) > 1:
                self.axis.legend(loc="upper right")
        self.canvas.draw_idle()

    def localize_content(self) -> None:
        apply_language(self)
        self.redraw()
