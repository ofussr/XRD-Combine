"""Tkinter controls for the shared GUI-independent radiation model."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

try:
    from ..i18n import apply_language, get_language, localised, messagebox
    from ..models.radiation import (
        PRESETS,
        RadiationSettings,
        RadiationTuple,
        validate_radiation_lines,
    )
except ImportError:  # pragma: no cover - direct module execution
    from i18n import apply_language, get_language, localised, messagebox
    from models.radiation import (
        PRESETS,
        RadiationSettings,
        RadiationTuple,
        validate_radiation_lines,
    )


def _other_label() -> str:
    return localised("Other…", "Autre…", "Другое…")


class CustomRadiationDialog(tk.Toplevel):
    """Modal editor for one to five custom wavelengths and relative weights."""

    def __init__(self, parent: tk.Misc, initial: list[RadiationTuple]) -> None:
        super().__init__(parent)
        self.title(localised("Custom radiation", "Rayonnement personnalisé", "Своё излучение"))
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self.result: list[RadiationTuple] | None = None
        self.rows: list[tuple[ttk.Frame, tk.StringVar, tk.StringVar]] = []

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="Введите от одной до пяти спектральных линий.",
        ).pack(anchor="w", pady=(0, 8))

        headings = ttk.Frame(body)
        headings.pack(fill="x")
        ttk.Label(headings, text="Линия", width=12).grid(row=0, column=0, sticky="w")
        ttk.Label(headings, text="λ, Å", width=14).grid(row=0, column=1, sticky="w")
        ttk.Label(headings, text="Относительный вес", width=18).grid(
            row=0, column=2, sticky="w"
        )
        self.rows_host = ttk.Frame(body)
        self.rows_host.pack(fill="x")

        row_buttons = ttk.Frame(body)
        row_buttons.pack(fill="x", pady=(8, 0))
        self.add_button = ttk.Button(
            row_buttons, text="Добавить линию", command=self._add_default_row
        )
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(
            row_buttons, text="Удалить последнюю", command=self._remove_row
        )
        self.remove_button.pack(side="left", padx=(6, 0))

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="Отмена", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Применить", command=self._accept).pack(
            side="right", padx=(0, 6)
        )

        for _name, wavelength, weight in initial[:5] or [("", 1.54056, 1.0)]:
            self._add_row(wavelength, weight)
        self._update_buttons()
        apply_language(self, get_language())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _add_default_row(self) -> None:
        self._add_row(1.54056, 1.0)
        self._update_buttons()

    def _add_row(self, wavelength: float, weight: float) -> None:
        if len(self.rows) >= 5:
            return
        frame = ttk.Frame(self.rows_host)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=f"{len(self.rows) + 1}", width=12).grid(
            row=0, column=0, sticky="w"
        )
        wavelength_var = tk.StringVar(value=f"{wavelength:.5f}")
        weight_var = tk.StringVar(value=f"{weight:g}")
        ttk.Entry(frame, textvariable=wavelength_var, width=14).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Entry(frame, textvariable=weight_var, width=18).grid(
            row=0, column=2, sticky="w", padx=(6, 0)
        )
        self.rows.append((frame, wavelength_var, weight_var))

    def _remove_row(self) -> None:
        if len(self.rows) <= 1:
            return
        frame, _wavelength, _weight = self.rows.pop()
        frame.destroy()
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.add_button.configure(state="disabled" if len(self.rows) >= 5 else "normal")
        self.remove_button.configure(state="disabled" if len(self.rows) <= 1 else "normal")

    def _accept(self) -> None:
        try:
            result = validate_radiation_lines(
                (
                    f"Custom {index}",
                    wavelength_var.get().strip().replace(",", "."),
                    weight_var.get().strip().replace(",", "."),
                )
                for index, (_frame, wavelength_var, weight_var) in enumerate(
                    self.rows, start=1
                )
            )
        except ValueError:
            messagebox.showerror(
                localised("Invalid radiation", "Rayonnement incorrect", "Некорректное излучение"),
                localised(
                    "Every wavelength and relative weight must be a positive finite number.",
                    "Chaque longueur d’onde et chaque poids relatif doit être un nombre fini positif.",
                    "Каждая длина волны и относительный вес должны быть положительными конечными числами.",
                ),
                parent=self,
            )
            return
        self.result = result
        self.destroy()


class RadiationSelector(ttk.Frame):
    """One selector controlling every CIF calculation in Structures."""

    def __init__(
        self,
        parent: tk.Misc,
        settings: RadiationSettings,
        on_change: Callable[[list[RadiationTuple]], None],
    ) -> None:
        super().__init__(parent, padding=(10, 7))
        self.settings = settings
        self.on_change = on_change
        self.value = tk.StringVar()
        ttk.Label(self, text="Излучение").pack(side="left")
        self.combo = ttk.Combobox(
            self,
            textvariable=self.value,
            state="readonly",
            width=73,
        )
        self.combo.pack(side="left", padx=(8, 0), fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self._selected)
        self.localize_content()

    def _values(self) -> list[str]:
        return [preset.label for preset in PRESETS] + [_other_label()]

    def localize_content(self) -> None:
        apply_language(self, get_language())
        self.combo.configure(values=self._values())
        self.value.set(self.settings.label(other_label=_other_label()))

    def _selected(self, _event=None) -> None:
        selected = self.value.get()
        if selected == _other_label():
            dialog = CustomRadiationDialog(self, self.settings.custom_lines)
            self.wait_window(dialog)
            if dialog.result is None:
                self.value.set(self.settings.label(other_label=_other_label()))
                return
            self.settings.select_custom(dialog.result)
        else:
            preset = next((item for item in PRESETS if item.label == selected), None)
            if preset is None:
                self.value.set(self.settings.label(other_label=_other_label()))
                return
            self.settings.select_preset(preset.key)
        self.value.set(self.settings.label(other_label=_other_label()))
        self.on_change(self.settings.lines())
