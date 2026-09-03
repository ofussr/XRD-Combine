"""Standalone CIF structure page using the existing pole-tool geometry."""

from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from .controls import CollapsibleSection, ScrollableControls, FrameScheduler
    from .structure_render import discard_scene, direction_orientation, screen_drag_rotation
    from .i18n import LocalizedStringVar, apply_language, filedialog, localised, messagebox
    from .theoretical_pole import (
        Crystal, base_orientation, draw_crystal_structure, euler_matrix,
        matrix_to_euler, parse_cif, pole_display_orientation, rotation_x, rotation_y,
    )
except ImportError:
    from controls import CollapsibleSection, ScrollableControls, FrameScheduler
    from structure_render import discard_scene, direction_orientation, screen_drag_rotation
    from i18n import LocalizedStringVar, apply_language, filedialog, localised, messagebox
    from theoretical_pole import (
        Crystal, base_orientation, draw_crystal_structure, euler_matrix,
        matrix_to_euler, parse_cif, pole_display_orientation, rotation_x, rotation_y,
    )


class StructurePage(ttk.Frame):
    """Existing atoms, cell, approximate bonds and orientation controls only."""

    def __init__(self, parent: tk.Misc, on_open_cif=None) -> None:
        super().__init__(parent)
        self.on_open_cif = on_open_cif
        self.cif_document = None
        self.crystal: Crystal | None = None
        self.base_rotation = np.eye(3)
        self.user_rotation = np.eye(3)
        self.center_hkl = (0, 1, 0)
        self._drag_position = None
        self.view_name = "hkl"
        self.orientation_info = tk.StringVar(value="")
        self.basis_visible = tk.BooleanVar(value=True)
        self.path_var = LocalizedStringVar(value="CIF не открыт")
        self.info_var = tk.StringVar(value="")
        self.hkl_vars = [tk.StringVar(value=str(v)) for v in self.center_hkl]
        self.rotation_vars = [tk.StringVar(value="0.0") for _ in range(3)]
        self.relative_rotation_vars = [tk.StringVar(value="0.0") for _ in range(3)]
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.control_panel = ScrollableControls(self, width=355)
        self.control_panel.grid(row=0, column=0, sticky="nsew")
        self.controls_canvas = self.control_panel.canvas
        controls = self.control_panel.body
        file_box = CollapsibleSection(controls, text="Структура CIF", padding=8)
        file_box.pack(fill="x", pady=(0, 8))
        ttk.Button(file_box, text="Открыть CIF…", command=self.open_cif).pack(fill="x")
        ttk.Label(file_box, textvariable=self.path_var, wraplength=295).pack(fill="x", pady=6)
        ttk.Label(file_box, textvariable=self.info_var, wraplength=295).pack(fill="x")

        orientation = CollapsibleSection(controls, text="Ориентация и вращение", padding=8)
        orientation.pack(fill="x", pady=(0, 8))
        directions = ttk.Frame(orientation)
        directions.pack(fill="x")
        for i, name in enumerate(("a", "b", "c", "a*", "b*", "c*")):
            directions.columnconfigure(i, weight=1)
            ttk.Button(directions, text=name, width=4,
                       command=lambda key=name: self.apply_direction(key)).grid(row=0, column=i, sticky="ew")
        ttk.Button(orientation, text="Стандартная ориентация",
                   command=lambda: self.apply_direction("standard")).pack(fill="x", pady=(5, 0))
        ttk.Label(orientation, textvariable=self.orientation_info, wraplength=310).pack(fill="x", pady=5)
        center = ttk.LabelFrame(orientation, text="Ориентация по (h k l)", padding=6)
        center.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(center)
        row.pack(fill="x")
        for i, (name, var) in enumerate(zip(("h", "k", "l"), self.hkl_vars)):
            ttk.Label(row, text=name).grid(row=0, column=i)
            entry = ttk.Entry(row, textvariable=var, width=9)
            entry.grid(row=1, column=i, padx=2)
            entry.bind("<Return>", lambda _e: self.apply_center())
        ttk.Button(center, text="Вид вдоль нормали к (h k l)", command=self.apply_center).pack(fill="x", pady=(6, 0))

        for title, variables, button_text, command in (
            ("Абсолютный поворот", self.rotation_vars,
             "Установить углы", self.apply_exact_rotation),
            ("Относительный поворот", self.relative_rotation_vars,
             "Повернуть на Δ", self.apply_relative_rotation),
        ):
            box = ttk.LabelFrame(orientation, text=title, padding=6)
            box.pack(fill="x", pady=(0, 8))
            row = ttk.Frame(box)
            row.pack(fill="x")
            names = ("X, °", "Y, °", "Z, °") if variables is self.rotation_vars else ("ΔX, °", "ΔY, °", "ΔZ, °")
            for i, (name, var) in enumerate(zip(names, variables)):
                ttk.Label(row, text=name).grid(row=0, column=i)
                entry = ttk.Entry(row, textvariable=var, width=9)
                entry.grid(row=1, column=i, padx=2)
                entry.bind("<Return>", lambda _e, action=command: action())
            ttk.Button(box, text=button_text, command=command).pack(fill="x", pady=(6, 0))
        ttk.Button(orientation, text="Сбросить поворот", command=self.reset_rotation).pack(fill="x")
        view = CollapsibleSection(controls, text="Отображение", padding=8)
        view.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(view, text="Базисные векторы", variable=self.basis_visible,
                        command=self.redraw).pack(anchor="w")
        ttk.Label(view, text="Связи показаны приближённо, по ковалентным радиусам.", wraplength=295).pack(fill="x", pady=8)

        self.figure = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self._frames = FrameScheduler(self, lambda: self.redraw(preview=True))
        self.localize_content()

    def open_cif(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Открыть CIF…", filetypes=(("CIF", "*.cif"),))
        if path:
            if self.on_open_cif is not None:
                self.on_open_cif(path)
                return
            self.load_cif(path)

    def load_cif(self, path: str) -> bool:
        try:
            try:
                from .cif_document import load_cif_document
            except ImportError:  # pragma: no cover
                from cif_document import load_cif_document
            return self.load_document(load_cif_document(path))
        except Exception as exc:
            messagebox.showerror(localised("Could not open CIF", "Impossible d’ouvrir le CIF", "Не удалось открыть CIF"), str(exc), parent=self)
            return False

    def load_document(self, document) -> bool:
        try:
            crystal = document.crystal
            base = base_orientation(crystal, self.center_hkl)
        except Exception as exc:
            messagebox.showerror(localised("Could not open CIF", "Impossible d’ouvrir le CIF", "Не удалось открыть CIF"), str(exc), parent=self)
            return False
        self.cif_document = document
        self.crystal = crystal
        self.view_name = "hkl"
        self.base_rotation = base
        self.path_var.set(Path(document.source).name)
        for var, value in zip(self.hkl_vars, self.center_hkl):
            var.set(str(value))
        self.reset_rotation()
        self.localize_content()
        return True

    def clear_document(self) -> None:
        self._frames.cancel()
        self.cif_document = None
        self.crystal = None
        discard_scene(self.axis)
        self.path_var.set(localised("No CIF loaded", "Aucun CIF chargé", "CIF не открыт"))
        self.info_var.set("")
        self.orientation_info.set("")
        self.redraw()

    def localize_content(self) -> None:
        apply_language(self)
        if self.crystal is not None:
            c = self.crystal
            self.info_var.set(
                f"{c.formula}; {c.space_group}\n"
                f"a = {c.a:.4f} Å, b = {c.b:.4f} Å, c = {c.c:.4f} Å\n"
                f"α = {c.alpha:.3f}°, β = {c.beta:.3f}°, γ = {c.gamma:.3f}°"
            )
        self._update_orientation_info()
        self.redraw()

    def apply_center(self) -> None:
        if self.crystal is None:
            return
        try:
            hkl = tuple(int(var.get().strip()) for var in self.hkl_vars)
            if hkl == (0, 0, 0):
                raise ValueError
            rotation = base_orientation(self.crystal, hkl)
        except ValueError:
            messagebox.showerror("Ошибка", localised(
                "h, k, l must be integers and cannot all be zero.",
                "h, k, l doivent être entiers et non tous nuls.",
                "h, k, l должны быть целыми и не могут одновременно равняться нулю."), parent=self)
            return
        self.center_hkl = hkl
        self.view_name = "hkl"
        self.base_rotation = rotation
        self.reset_rotation()

    def apply_direction(self, name: str) -> None:
        if self.crystal is None:
            return
        self.base_rotation = direction_orientation(self.crystal, name)
        self.view_name = name
        if name.endswith("*"):
            self.center_hkl = tuple(int(i == "abc".index(name[0])) for i in range(3))
            for var, value in zip(self.hkl_vars, self.center_hkl):
                var.set(str(value))
        # Direct directions do not masquerade as Miller plane normals.
        self.reset_rotation()

    def _update_orientation_info(self) -> None:
        name = self.view_name
        if name == "standard":
            description = localised("Base: standard orientation", "Base : orientation standard", "Основа: стандартная ориентация")
        elif name == "hkl":
            indices = " ".join(str(v) for v in self.center_hkl)
            description = localised(f"Base: normal to ({indices})", f"Base : normale à ({indices})", f"Основа: нормаль к ({indices})")
        else:
            description = localised(f"Base: view along {name}", f"Base : vue suivant {name}", f"Основа: вид вдоль {name}")
        self.orientation_info.set(description)

    def _read_angles(self, variables) -> list[float] | None:
        try:
            angles = [float(var.get().strip().replace(",", ".")) for var in variables]
            if not all(math.isfinite(value) for value in angles):
                raise ValueError
            return angles
        except ValueError:
            messagebox.showerror("Ошибка", localised(
                "Angles must be finite numbers.", "Les angles doivent être des nombres finis.",
                "Углы должны быть конечными числами."), parent=self)
            return None

    def apply_exact_rotation(self) -> None:
        angles = self._read_angles(self.rotation_vars)
        if self.crystal is not None and angles is not None:
            self.user_rotation = euler_matrix(*angles)
            self._update_angles()
            self.redraw()

    def apply_relative_rotation(self) -> None:
        angles = self._read_angles(self.relative_rotation_vars)
        if self.crystal is not None and angles is not None:
            self.user_rotation = euler_matrix(*angles) @ self.user_rotation
            for var in self.relative_rotation_vars:
                var.set("0.0")
            self._update_angles()
            self.redraw()

    def _update_angles(self) -> None:
        for var, angle in zip(self.rotation_vars, matrix_to_euler(self.user_rotation)):
            var.set(f"{0.0 if abs(angle) < 5e-10 else angle:.3f}")

    def reset_rotation(self) -> None:
        self._update_orientation_info()
        self.user_rotation = np.eye(3)
        for var in self.relative_rotation_vars:
            var.set("0.0")
        self._update_angles()
        self.redraw()

    def redraw(self, *, preview=False) -> None:
        if self.crystal is None:
            self.axis.clear()
            self.axis.set_axis_off()
            self.axis.text2D(0.5, 0.5, localised("Open a CIF file", "Ouvrez un fichier CIF", "Откройте CIF"),
                             transform=self.axis.transAxes, ha="center")
        else:
            draw_crystal_structure(self.axis, self.crystal,
                                   pole_display_orientation(self.user_rotation @ self.base_rotation),
                                   preview=preview, show_basis=self.basis_visible.get())
        self.canvas.draw_idle()

    def on_press(self, event) -> None:
        if event.button == 1 and event.inaxes is self.axis and self.crystal is not None:
            self._drag_position = (event.x, event.y)

    def on_motion(self, event) -> None:
        if self._drag_position is None or event.x is None or event.y is None:
            return
        dx, dy = event.x-self._drag_position[0], event.y-self._drag_position[1]
        self.user_rotation = screen_drag_rotation(dx, dy) @ self.user_rotation
        self._drag_position = (event.x, event.y)
        self._frames.request()

    def on_release(self, _event) -> None:
        if self._drag_position is not None:
            self._drag_position = None
            self._frames.cancel()
            self._update_angles()
            self.redraw()

    def on_scroll(self, event) -> None:
        if event.inaxes is not self.axis:
            return
        scene = getattr(self.axis, "_crystal_scene", None)
        if scene is None:
            return
        steps = getattr(event, "step", 0) or (1 if event.button == "up" else -1)
        scene.zoom_by(float(steps))
        self.canvas.draw_idle()

    def refresh_atom_styles(self) -> None:
        discard_scene(self.axis)
        self.redraw()
