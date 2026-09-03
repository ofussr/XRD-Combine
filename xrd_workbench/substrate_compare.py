"""Свободное рабочее поле и галерея сравнения измерений с подложками."""

from __future__ import annotations

import colorsys
import ctypes
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from typing import Callable

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from .i18n import apply_language, filedialog, localised, messagebox
    from .xrd_io import Scan1D, clone_scan, read_scan_file
except ImportError:
    from i18n import apply_language, filedialog, localised, messagebox
    from xrd_io import Scan1D, clone_scan, read_scan_file


SUPPORTED_SUFFIXES = {".xrdml", ".xml", ".raw", ".xy", ".txt", ".dat", ".csv"}
FILE_COLOURS = ("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99")
SUBSTRATE_COLOURS = ("#000000", "#404040", "#7f7f7f")


def _palette(count: int, base: tuple[str, ...]) -> list[str]:
    if count <= len(base):
        return list(base[:count])
    extra = [
        matplotlib.colors.to_hex(
            colorsys.hsv_to_rgb(
                (index + 0.15) / (count - len(base) + 0.3),
                0.7,
                0.9,
            )
        )
        for index in range(count - len(base))
    ]
    return [*base, *extra]


def _axis_label(name: str) -> str:
    key = name.replace(" ", "").replace("-", "").lower()
    labels = {
        "2theta": "2θ",
        "twotheta": "2θ",
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
    return labels.get(key, name)


def _axis_uses_degrees(name: str) -> bool:
    return name.replace(" ", "").replace("-", "").lower() in {
        "2theta",
        "twotheta",
        "theta",
        "omega",
        "chi",
        "phi",
    }


def _base_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


@dataclass(eq=False)
class WorkspaceItem:
    kind: str
    name: str
    scan: Scan1D
    button: tk.Button
    x: int
    y: int
    home_x: int
    home_y: int
    group: int | None = None
    active: bool = False


@dataclass
class Assembly:
    group_id: int
    label: str
    items: list[WorkspaceItem]
    view: dict[str, object] = field(default_factory=dict)


class SubstrateComparisonPage(ttk.Frame):
    """Встроенная версия рабочего поля из исходного ``новапор.py``."""

    def __init__(
        self,
        parent: tk.Misc,
        on_send_viewer: Callable[[Scan1D], None] | None = None,
        on_send_correction: Callable[[Scan1D], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_send_viewer = on_send_viewer
        self.on_send_correction = on_send_correction
        self.items: list[WorkspaceItem] = []
        self.item_by_button: dict[tk.Button, WorkspaceItem] = {}
        self.groups: dict[int, set[WorkspaceItem]] = {}
        self.group_active: dict[int, bool] = {}
        self.group_views: dict[int, dict[str, object]] = {}
        self.next_group_id = 1
        self.drag_data: dict[str, object] = {
            "button": None,
            "sx": 0,
            "sy": 0,
            "moved": False,
            "px": 0,
            "py": 0,
        }
        self.context_item: WorkspaceItem | None = None
        self.next_substrate_y = 46
        self.next_file_y = 46
        self.substrate_x = 20
        self.file_x = 400
        self.reset_counter = 0
        self._gallery_figures: list[Figure] = []
        self._viewer_figures: set[Figure] = set()

        base = _base_directory()
        self.substrate_directory = base / "substrates"
        self.preset_path = base / "substrate_view_presets.json"
        self.presets: dict[str, dict[str, list[float]]] = {}
        try:
            self.presets = json.loads(self.preset_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self.presets = {}

        self._build()
        self._load_default_substrates()
        apply_language(self)

    def _build(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.workspace_tab = ttk.Frame(self.notebook)
        self.gallery_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.workspace_tab, text="Рабочее поле")
        self.notebook.add(self.gallery_tab, text="Галерея и результаты")
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)

        controls = ttk.Frame(self.workspace_tab, padding=7, relief="raised")
        controls.pack(side="left", fill="y")
        ttk.Button(
            controls,
            text="Загрузить папку подложек…",
            command=self.load_substrate_folder,
        ).pack(fill="x", pady=2)
        ttk.Separator(controls, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(
            controls,
            text="Открыть файлы…",
            command=self.load_files,
        ).pack(fill="x", pady=2)
        ttk.Button(
            controls,
            text="Открыть папку…",
            command=self.load_folder,
        ).pack(fill="x", pady=2)
        ttk.Separator(controls, orient="horizontal").pack(fill="x", pady=8)
        ttk.Button(
            controls,
            text="Сбросить расположение",
            command=self.reset_layout,
        ).pack(fill="x", pady=2)

        self.workspace = tk.Frame(self.workspace_tab, bg="#f0f0f0")
        self.workspace.pack(side="right", fill="both", expand=True)
        self.substrate_heading = tk.Label(
            self.workspace,
            text="Подложки – правый щелчок создаёт копию",
            bg="#f0f0f0",
            font=("Arial", 10, "bold"),
        )
        self.substrate_heading.place(x=self.substrate_x, y=12)
        self.file_heading = tk.Label(
            self.workspace,
            text="Измерения – перетащите для объединения",
            bg="#f0f0f0",
            font=("Arial", 10, "bold"),
        )
        self.file_heading.place(x=self.file_x, y=12)

        self.context_menu = tk.Menu(self, tearoff=False)
        self.context_menu.add_command(
            label="Отправить в просмотр",
            command=self.send_context_to_viewer,
        )
        self.context_menu.add_command(
            label="Отправить в коррекцию",
            command=self.send_context_to_correction,
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="Создать копию",
            command=self.duplicate_substrate,
        )

        self.gallery_tab.rowconfigure(0, weight=1)
        self.gallery_tab.columnconfigure(0, weight=1)
        self.gallery_canvas = tk.Canvas(self.gallery_tab, highlightthickness=0)
        self.gallery_scroll = ttk.Scrollbar(
            self.gallery_tab,
            orient="vertical",
            command=self.gallery_canvas.yview,
        )
        self.gallery_frame = ttk.Frame(self.gallery_canvas)
        self.gallery_window = self.gallery_canvas.create_window(
            (0, 0),
            window=self.gallery_frame,
            anchor="nw",
        )
        self.gallery_frame.bind(
            "<Configure>",
            lambda _event: self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all")
            ),
        )
        self.gallery_canvas.bind(
            "<Configure>",
            lambda event: self.gallery_canvas.itemconfigure(
                self.gallery_window,
                width=event.width,
            ),
        )
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scroll.set)
        self.gallery_canvas.grid(row=0, column=0, sticky="nsew")
        self.gallery_scroll.grid(row=0, column=1, sticky="ns")

        gallery_controls = ttk.Frame(self.gallery_tab, padding=6)
        gallery_controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(
            gallery_controls,
            text="Загрузить настройки…",
            command=self.load_presets,
        ).pack(side="left", padx=4)
        ttk.Button(
            gallery_controls,
            text="Сохранить настройки…",
            command=self.save_presets,
        ).pack(side="left", padx=4)
        ttk.Button(
            gallery_controls,
            text="Обновить галерею",
            command=self.refresh_gallery,
        ).pack(side="right", padx=4)

    def localize_content(self) -> None:
        apply_language(self)
        try:
            self.context_menu.entryconfigure(
                0,
                label=localised(
                    "Send to viewer",
                    "Envoyer vers la visualisation",
                    "Отправить в просмотр",
                ),
            )
            self.context_menu.entryconfigure(
                1,
                label=localised(
                    "Send to correction",
                    "Envoyer vers la correction",
                    "Отправить в коррекцию",
                ),
            )
            self.context_menu.entryconfigure(
                3,
                label=localised(
                    "Duplicate substrate",
                    "Dupliquer le substrat",
                    "Создать копию подложки",
                ),
            )
        except tk.TclError:
            pass
        if self.notebook.index("current") == 1:
            self.refresh_gallery()

    def _load_default_substrates(self) -> None:
        if not self.substrate_directory.exists():
            try:
                self.substrate_directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                return
        paths = [
            path
            for path in sorted(self.substrate_directory.iterdir())
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if paths:
            self._load_paths(paths, "substrate")

    def _unique_name(self, wanted: str) -> str:
        existing = {item.name for item in self.items}
        if wanted not in existing:
            return wanted
        index = 1
        while f"{wanted}_{index}" in existing:
            index += 1
        return f"{wanted}_{index}"

    def _make_card(self, name: str, scan: Scan1D, kind: str) -> WorkspaceItem:
        x = self.substrate_x if kind == "substrate" else self.file_x
        y = self.next_substrate_y if kind == "substrate" else self.next_file_y
        button = tk.Button(
            self.workspace,
            text=name,
            width=31,
            pady=6,
            bg="#d0d0d0",
            relief="raised",
        )
        button.place(x=x, y=y)
        item = WorkspaceItem(
            kind=kind,
            name=name,
            scan=scan,
            button=button,
            x=x,
            y=y,
            home_x=x,
            home_y=y,
        )
        self.items.append(item)
        self.item_by_button[button] = item
        button.bind("<Button-1>", lambda event, target=button: self._press(event, target))
        button.bind(
            "<B1-Motion>",
            lambda event, target=button: self._motion(event, target),
        )
        button.bind(
            "<ButtonRelease-1>",
            lambda event, target=button: self._release(event, target),
        )
        button.bind(
            "<Button-3>",
            lambda event, target=item: self._show_context(event, target),
        )
        if kind == "substrate":
            self.next_substrate_y += 38
        else:
            self.next_file_y += 38
        return item

    def add_scan(self, scan: Scan1D, kind: str = "file") -> WorkspaceItem:
        """Add a curve received from another application tab."""

        name = self._unique_name(scan.name)
        return self._make_card(name, scan, kind)

    def _load_paths(self, paths, kind: str) -> None:
        errors: list[str] = []
        for value in paths:
            path = Path(value)
            try:
                scans = read_scan_file(path)
                for scan in scans:
                    name = self._unique_name(scan.name)
                    self._make_card(name, scan, kind)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        if errors:
            messagebox.showwarning(
                localised(
                    "Some files were skipped",
                    "Certains fichiers ont été ignorés",
                    "Часть файлов пропущена",
                ),
                "\n\n".join(errors[:12]),
                parent=self,
            )

    def _clear_kind(self, kind: str) -> None:
        self.reset_layout(count=False)
        remaining: list[WorkspaceItem] = []
        for item in self.items:
            if item.kind == kind:
                self.item_by_button.pop(item.button, None)
                item.button.destroy()
            else:
                remaining.append(item)
        self.items = remaining
        if kind == "substrate":
            self.next_substrate_y = 46
        else:
            self.next_file_y = 46

    def load_substrate_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self,
            title="Папка с подложками",
        )
        if not folder:
            return
        paths = [
            path
            for path in sorted(Path(folder).iterdir())
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        self._clear_kind("substrate")
        self._load_paths(paths, "substrate")

    def load_files(self) -> None:
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
            self._load_paths(paths, "file")

    def load_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self,
            title="Открыть папку с измерениями",
        )
        if not folder:
            return
        paths = [
            path
            for path in sorted(Path(folder).iterdir())
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        self._load_paths(paths, "file")

    def _show_context(self, event, item: WorkspaceItem) -> None:
        self.context_item = item
        self.context_menu.entryconfigure(
            0,
            state="normal" if self.on_send_viewer is not None else "disabled",
        )
        self.context_menu.entryconfigure(
            1,
            state="normal" if self.on_send_correction is not None else "disabled",
        )
        self.context_menu.entryconfigure(
            3,
            label=localised(
                f"Duplicate “{item.name}”",
                f"Dupliquer « {item.name} »",
                f"Создать копию «{item.name}»",
            ),
            state="normal" if item.kind == "substrate" else "disabled",
        )
        self.context_menu.post(event.x_root, event.y_root)

    def send_context_to_viewer(self) -> None:
        source = self.context_item
        if source is not None and self.on_send_viewer is not None:
            self.on_send_viewer(source.scan)

    def send_context_to_correction(self) -> None:
        source = self.context_item
        if source is not None and self.on_send_correction is not None:
            self.on_send_correction(source.scan)

    def duplicate_substrate(self) -> None:
        source = self.context_item
        if source is None or source.kind != "substrate":
            return
        name = self._unique_name(source.name)
        self._make_card(name, clone_scan(source.scan), "substrate")

    def reset_layout(self, count: bool = True) -> None:
        self.groups.clear()
        self.group_active.clear()
        self.group_views.clear()
        for item in self.items:
            item.group = None
            item.active = False
            item.button.configure(bg="#d0d0d0", relief="raised", bd=1)
            item.x, item.y = item.home_x, item.home_y
            item.button.place(x=item.x, y=item.y)
        if count:
            self.reset_counter += 1
            if self.reset_counter == 10:
                self._show_about_author()
                self.reset_counter = 0

    def _show_about_author(self) -> None:
        top = tk.Toplevel(self)
        top.title("About Author")
        top.geometry("350x200")
        top.resizable(False, False)
        top.configure(bg="#202020")
        tk.Label(
            top,
            text="CREATED BY",
            font=("Arial", 8, "bold"),
            fg="#808080",
            bg="#202020",
        ).pack(pady=(20, 5))
        tk.Label(
            top,
            text="Mikhail D. Mirus(h)chenko",
            font=("Arial", 14, "bold"),
            fg="#ffffff",
            bg="#202020",
        ).pack()
        tk.Label(
            top,
            text="ofussr",
            font=("Consolas", 11),
            fg="#00ff00",
            bg="#202020",
        ).pack(pady=5)
        tk.Label(
            top,
            text="miruschenko98@gmail.com",
            font=("Arial", 10),
            fg="#a0a0a0",
            bg="#202020",
        ).pack(pady=10)
        tk.Button(
            top,
            text="Close",
            command=top.destroy,
            bg="#404040",
            fg="white",
            relief="flat",
        ).pack(pady=10)

    def _press(self, event, button: tk.Button) -> None:
        button.lift()
        self.drag_data.update(
            {
                "button": button,
                "sx": button.winfo_x() - event.x_root,
                "sy": button.winfo_y() - event.y_root,
                "moved": False,
                "px": button.winfo_x(),
                "py": button.winfo_y(),
            }
        )

    def _motion(self, event, button: tk.Button) -> None:
        if self.drag_data["button"] is not button:
            return
        x = int(self.drag_data["sx"]) + event.x_root
        y = int(self.drag_data["sy"]) + event.y_root
        if (
            abs(x - int(self.drag_data["px"])) > 3
            or abs(y - int(self.drag_data["py"])) > 3
        ):
            self.drag_data["moved"] = True
        button.place(x=x, y=y)
        item = self.item_by_button[button]
        item.x, item.y = x, y

    @staticmethod
    def _overlap(first: WorkspaceItem, second: WorkspaceItem) -> int:
        ax1, ay1 = first.button.winfo_x(), first.button.winfo_y()
        ax2 = ax1 + first.button.winfo_width()
        ay2 = ay1 + first.button.winfo_height()
        bx1, by1 = second.button.winfo_x(), second.button.winfo_y()
        bx2 = bx1 + second.button.winfo_width()
        by2 = by1 + second.button.winfo_height()
        width = max(0, min(ax2, bx2) - max(ax1, bx1))
        height = max(0, min(ay2, by2) - max(ay1, by1))
        return width * height

    def _release(self, _event, button: tk.Button) -> None:
        source = self.item_by_button[button]
        if not bool(self.drag_data["moved"]):
            if source.group is not None:
                self.set_group_active(
                    source.group,
                    not self.group_active[source.group],
                )
            else:
                self.set_item_active(source, not source.active)
            self.drag_data["button"] = None
            return

        hits = [
            (item, self._overlap(source, item))
            for item in self.items
            if item is not source and self._overlap(source, item) > 0
        ]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        if hits:
            target = hits[0][0]
            if source.group is None and target.group is None:
                group_id = self.create_group(source, target)
            elif source.group is not None and target.group is None:
                group_id = source.group
                self.add_to_group(group_id, target)
            elif source.group is None and target.group is not None:
                group_id = target.group
                self.add_to_group(group_id, source)
            else:
                assert source.group is not None and target.group is not None
                group_id = self.merge_groups(source.group, target.group)
            self.layout_group(
                group_id,
                target.button.winfo_x(),
                min(source.button.winfo_y(), target.button.winfo_y()),
            )
        elif source.group is not None:
            overlaps_group = any(
                self._overlap(source, other) > 0
                for other in self.groups[source.group]
                if other is not source
            )
            if not overlaps_group:
                self.remove_from_group(source)
        self.drag_data["button"] = None

    @staticmethod
    def set_item_active(item: WorkspaceItem, active: bool) -> None:
        item.active = active
        item.button.configure(
            bg="#2ecc71" if active else "#d0d0d0",
            relief="solid" if active else "raised",
            bd=2 if active else 1,
        )

    def set_group_active(self, group_id: int, active: bool) -> None:
        self.group_active[group_id] = active
        for item in self.groups[group_id]:
            self.set_item_active(item, active)

    def layout_group(self, group_id: int, x: int, y: int) -> None:
        members = sorted(
            self.groups[group_id],
            key=lambda item: (0 if item.kind == "file" else 1, item.name),
        )
        for item in members:
            item.x, item.y = x, y
            item.button.place(x=x, y=y)
            y += item.button.winfo_height()

    def create_group(self, *members: WorkspaceItem) -> int:
        group_id = self.next_group_id
        self.next_group_id += 1
        self.groups[group_id] = set(members)
        self.group_active[group_id] = True
        self.group_views[group_id] = {}
        for item in members:
            item.group = group_id
            self.set_item_active(item, True)
        self.layout_group(
            group_id,
            members[0].button.winfo_x(),
            members[0].button.winfo_y(),
        )
        return group_id

    def add_to_group(self, group_id: int, item: WorkspaceItem) -> None:
        self.groups[group_id].add(item)
        item.group = group_id
        self.set_item_active(item, self.group_active[group_id])
        anchor = next(iter(self.groups[group_id]))
        self.layout_group(
            group_id,
            anchor.button.winfo_x(),
            anchor.button.winfo_y(),
        )

    def merge_groups(self, first: int, second: int) -> int:
        if first == second:
            return first
        for item in list(self.groups[second]):
            item.group = first
            self.groups[first].add(item)
        self.group_active[first] = (
            self.group_active[first] or self.group_active[second]
        )
        self.group_views[first] = (
            self.group_views.get(first) or self.group_views.get(second) or {}
        )
        del self.groups[second]
        del self.group_active[second]
        self.group_views.pop(second, None)
        anchor = next(iter(self.groups[first]))
        self.layout_group(first, anchor.button.winfo_x(), anchor.button.winfo_y())
        return first

    def remove_from_group(self, item: WorkspaceItem) -> None:
        group_id = item.group
        if group_id is None:
            return
        self.groups[group_id].discard(item)
        item.group = None
        self.set_item_active(item, False)
        if len(self.groups[group_id]) < 2:
            if self.groups[group_id]:
                remaining = next(iter(self.groups[group_id]))
                remaining.group = None
                self.set_item_active(remaining, False)
            del self.groups[group_id]
            del self.group_active[group_id]
            self.group_views.pop(group_id, None)
            return
        anchor = next(iter(self.groups[group_id]))
        self.layout_group(
            group_id,
            anchor.button.winfo_x(),
            anchor.button.winfo_y(),
        )

    def _tab_changed(self, _event=None) -> None:
        if self.notebook.index("current") == 1:
            self.refresh_gallery()

    def _assemblies(self) -> list[Assembly]:
        result: list[Assembly] = []
        for group_id, members in self.groups.items():
            if not self.group_active[group_id]:
                continue
            ordered = sorted(
                members,
                key=lambda item: (0 if item.kind == "file" else 1, item.name),
            )
            result.append(
                Assembly(
                    group_id=group_id,
                    label=" + ".join(item.name for item in ordered),
                    items=ordered,
                    view=dict(self.group_views.get(group_id, {})),
                )
            )
        return result

    def _close_gallery_figures(self) -> None:
        for figure in self._gallery_figures:
            try:
                figure.clear()
            except Exception:
                pass
        self._gallery_figures.clear()

    def refresh_gallery(self) -> None:
        self._close_gallery_figures()
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        assemblies = self._assemblies()
        if not assemblies:
            ttk.Label(
                self.gallery_frame,
                text="На рабочем поле нет активных групп.",
            ).pack(pady=25)
            apply_language(self.gallery_frame)
            return

        columns = 2
        for index, assembly in enumerate(assemblies):
            row, column = divmod(index, columns)
            frame = ttk.Frame(self.gallery_frame, relief="groove", padding=5)
            frame.grid(row=row, column=column, padx=10, pady=10, sticky="nsew")
            ttk.Label(
                frame,
                text=assembly.label[:65],
                font=("Arial", 9, "bold"),
            ).pack(anchor="w")
            figure = Figure(figsize=(4, 2.2), dpi=80)
            self._gallery_figures.append(figure)
            axis = figure.add_subplot(111)
            figure.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.18)
            self.plot_assembly(axis, assembly, thumbnail=True)
            canvas = FigureCanvasTkAgg(figure, frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.get_tk_widget().bind(
                "<Button-1>",
                lambda _event, value=assembly: self.open_viewer(value),
            )
            buttons = ttk.Frame(frame)
            buttons.pack(fill="x", pady=2)
            ttk.Button(
                buttons,
                text="Открыть",
                command=lambda value=assembly: self.open_viewer(value),
            ).pack(side="right")
            ttk.Button(
                buttons,
                text="Копировать",
                command=lambda value=assembly: self.copy_to_clipboard(value),
            ).pack(side="right", padx=5)
        for column in range(columns):
            self.gallery_frame.columnconfigure(column, weight=1)
        apply_language(self.gallery_frame)

    def _preset_for(self, name: str):
        if name in self.presets:
            return self.presets[name]
        for key in sorted(self.presets, key=len, reverse=True):
            if not name.startswith(key):
                continue
            suffix = name[len(key) :]
            if suffix == "_copy" or (
                suffix.startswith("_") and suffix[1:].isdigit()
            ):
                return self.presets[key]
        return None

    @staticmethod
    def _transformed_y(values: np.ndarray, mode: str) -> np.ndarray:
        y = np.asarray(values, dtype=float).copy()
        if mode == "log":
            positive = y[np.isfinite(y) & (y > 0)]
            epsilon = (
                max(1e-12, float(np.min(positive)) * 1e-6)
                if positive.size
                else 1e-12
            )
            return np.clip(y, epsilon, None)
        if mode == "exp":
            maximum = float(np.nanmax(y)) if y.size else 0.0
            return np.exp(y / (maximum or 1.0))
        if mode == "square":
            return np.square(np.maximum(y, 0))
        return y

    def plot_assembly(
        self,
        axis,
        assembly: Assembly,
        view: dict[str, object] | None = None,
        thumbnail: bool = False,
    ) -> None:
        settings = view or assembly.view
        y_mode = str(settings.get("ymode", "log"))
        axis.clear()
        files = [item for item in assembly.items if item.kind == "file"]
        substrates = [
            item for item in assembly.items if item.kind == "substrate"
        ]
        file_base = ("#e31a1c",) if len(files) == 1 else FILE_COLOURS
        colours = {
            **dict(zip(files, _palette(len(files), file_base))),
            **dict(
                zip(
                    substrates,
                    _palette(len(substrates), SUBSTRATE_COLOURS),
                )
            ),
        }
        for item in sorted(
            assembly.items,
            key=lambda value: 0 if value.kind == "substrate" else 1,
        ):
            y = self._transformed_y(item.scan.y, y_mode)
            axis.plot(
                item.scan.x,
                y,
                label=item.name,
                color=colours[item],
                linewidth=1.0 if thumbnail else 1.3,
                zorder=2 if item.kind == "substrate" else 3,
            )

        if settings.get("xlim") and settings.get("ylim"):
            axis.set_xlim(settings["xlim"])
            axis.set_ylim(settings["ylim"])
        else:
            preset = next(
                (
                    self._preset_for(item.name)
                    for item in substrates
                    if self._preset_for(item.name)
                ),
                None,
            )
            if preset:
                axis.set_xlim(preset["xlim"])
                axis.set_ylim(preset["ylim"])
            else:
                axis.autoscale(True, "both")

        if y_mode == "log":
            axis.set_yscale("log")
        else:
            axis.set_yscale("linear")
        axes = {item.scan.axis_name for item in assembly.items}
        if len(axes) == 1:
            axis_name = next(iter(axes))
            title = _axis_label(axis_name)
            if _axis_uses_degrees(axis_name):
                title = f"{title}, °"
        else:
            title = localised(
                "Scan coordinate",
                "Coordonnée du balayage",
                "Координата скана",
            )
        axis.set_xlabel(title)
        axis.set_ylabel(localised("Intensity", "Intensité", "Интенсивность"))
        axis.grid(True, alpha=0.2)
        if not thumbnail:
            axis.legend(loc="upper left")

    def open_viewer(self, assembly: Assembly) -> None:
        top = tk.Toplevel(self)
        top.title(assembly.label)
        top.geometry("900x650")
        figure = Figure(figsize=(6, 4), dpi=100)
        self._viewer_figures.add(figure)
        axis = figure.add_subplot(111)
        figure.subplots_adjust(left=0.1, right=0.97, top=0.96, bottom=0.12)
        canvas = FigureCanvasTkAgg(figure, top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        toolbar.pack(fill="x")

        controls = ttk.Frame(top, padding=5)
        controls.pack(fill="x")
        x_min, x_max = tk.StringVar(), tk.StringVar()
        y_min, y_max = tk.StringVar(), tk.StringVar()
        ttk.Label(controls, text="X min").pack(side="left")
        ttk.Entry(controls, width=9, textvariable=x_min).pack(side="left")
        ttk.Label(controls, text="X max").pack(side="left")
        ttk.Entry(controls, width=9, textvariable=x_max).pack(side="left")
        ttk.Label(controls, text="Y min").pack(side="left", padx=(8, 0))
        ttk.Entry(controls, width=9, textvariable=y_min).pack(side="left")
        ttk.Label(controls, text="Y max").pack(side="left")
        ttk.Entry(controls, width=9, textvariable=y_max).pack(side="left")

        y_mode = tk.StringVar(value=str(assembly.view.get("ymode", "log")))

        def apply_limits() -> None:
            try:
                axis.set_xlim(
                    float(x_min.get().replace(",", ".")),
                    float(x_max.get().replace(",", ".")),
                )
                axis.set_ylim(
                    float(y_min.get().replace(",", ".")),
                    float(y_max.get().replace(",", ".")),
                )
                canvas.draw_idle()
            except ValueError:
                messagebox.showerror(
                    localised("Invalid limits", "Limites incorrectes", "Некорректные границы"),
                    localised(
                        "All four limits must be numeric.",
                        "Les quatre limites doivent être numériques.",
                        "Все четыре границы должны быть числами.",
                    ),
                    parent=top,
                )

        def toggle_y() -> None:
            modes = ("linear", "log", "exp", "square")
            current = y_mode.get()
            y_mode.set(modes[(modes.index(current) + 1) % len(modes)])
            self.plot_assembly(
                axis,
                assembly,
                {
                    **assembly.view,
                    "ymode": y_mode.get(),
                    "xlim": axis.get_xlim(),
                    "ylim": axis.get_ylim(),
                },
            )
            canvas.draw_idle()

        def close_viewer() -> None:
            self._viewer_figures.discard(figure)
            figure.clear()
            top.destroy()

        def save_and_close() -> None:
            assembly.view = {
                "xlim": list(axis.get_xlim()),
                "ylim": list(axis.get_ylim()),
                "ymode": y_mode.get(),
            }
            self.group_views[assembly.group_id] = dict(assembly.view)
            for item in assembly.items:
                if item.kind == "substrate":
                    self.presets[item.name] = {
                        "xlim": list(axis.get_xlim()),
                        "ylim": list(axis.get_ylim()),
                    }
            try:
                self.preset_path.write_text(
                    json.dumps(self.presets, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass
            close_viewer()
            self.refresh_gallery()

        ttk.Button(
            controls,
            text="Применить",
            command=apply_limits,
        ).pack(side="left", padx=5)
        ttk.Button(
            controls,
            text="Режим Y",
            command=toggle_y,
        ).pack(side="left")
        ttk.Button(
            controls,
            text="Сохранить и закрыть",
            command=save_and_close,
        ).pack(side="right", padx=5)

        self.plot_assembly(axis, assembly, assembly.view)
        current_x = axis.get_xlim()
        current_y = axis.get_ylim()
        x_min.set(f"{current_x[0]:.6g}")
        x_max.set(f"{current_x[1]:.6g}")
        y_min.set(f"{current_y[0]:.6g}")
        y_max.set(f"{current_y[1]:.6g}")
        top.protocol("WM_DELETE_WINDOW", close_viewer)
        apply_language(top)
        canvas.draw_idle()

    def copy_to_clipboard(self, assembly: Assembly) -> None:
        buffer = io.BytesIO()
        figure = Figure(figsize=(7, 4), dpi=150)
        axis = figure.add_subplot(111)
        figure.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.15)
        self.plot_assembly(axis, assembly, assembly.view)
        figure.savefig(buffer, format="png", dpi=150)
        figure.clear()
        buffer.seek(0)
        try:
            system = platform.system()
            if system == "Windows":
                if Image is None:
                    raise RuntimeError(
                        localised(
                            "Pillow is required to copy images on Windows.",
                            "Pillow est requis pour copier des images sous Windows.",
                            "Для копирования изображения в Windows нужен Pillow.",
                        )
                    )
                image = Image.open(buffer)
                output = io.BytesIO()
                image.save(output, "BMP")
                data = output.getvalue()[14:]
                kernel32 = ctypes.windll.kernel32
                user32 = ctypes.windll.user32
                kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
                kernel32.GlobalAlloc.restype = ctypes.c_void_p
                kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
                kernel32.GlobalLock.restype = ctypes.c_void_p
                kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
                user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
                user32.OpenClipboard.argtypes = [ctypes.c_void_p]
                if not user32.OpenClipboard(None):
                    raise RuntimeError("OpenClipboard failed")
                try:
                    user32.EmptyClipboard()
                    handle = kernel32.GlobalAlloc(0x0002, len(data))
                    if not handle:
                        raise RuntimeError("GlobalAlloc failed")
                    pointer = kernel32.GlobalLock(handle)
                    ctypes.memmove(pointer, data, len(data))
                    kernel32.GlobalUnlock(handle)
                    user32.SetClipboardData(8, handle)
                finally:
                    user32.CloseClipboard()
            elif system == "Darwin":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as stream:
                    stream.write(buffer.getvalue())
                    temporary = stream.name
                try:
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            f'set the clipboard to (read (POSIX file "{temporary}") as «class PNGf»)',
                        ],
                        check=True,
                    )
                finally:
                    os.unlink(temporary)
            else:
                if shutil.which("xclip"):
                    command = ["xclip", "-sel", "clip", "-t", "image/png"]
                elif shutil.which("wl-copy"):
                    command = ["wl-copy", "-t", "image/png"]
                else:
                    raise RuntimeError(
                        localised(
                            "Neither xclip nor wl-copy is installed.",
                            "Ni xclip ni wl-copy n’est installé.",
                            "Не установлены ни xclip, ни wl-copy.",
                        )
                    )
                subprocess.run(command, input=buffer.getvalue(), check=True)
        except Exception as exc:
            messagebox.showerror(
                localised(
                    "Clipboard error",
                    "Erreur du presse-papiers",
                    "Ошибка буфера обмена",
                ),
                str(exc),
                parent=self,
            )

    def load_presets(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=(("JSON", "*.json"), ("Все файлы", "*.*")),
        )
        if not path:
            return
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("JSON root must be an object")
            self.presets.update(loaded)
            self.refresh_gallery()
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror(
                localised(
                    "Preset error",
                    "Erreur de réglages",
                    "Ошибка настроек",
                ),
                str(exc),
                parent=self,
            )

    def save_presets(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("Все файлы", "*.*")),
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self.presets, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            messagebox.showerror(
                localised(
                    "Preset error",
                    "Erreur de réglages",
                    "Ошибка настроек",
                ),
                str(exc),
                parent=self,
            )

    def close_figures(self) -> None:
        self._close_gallery_figures()
        for figure in list(self._viewer_figures):
            try:
                figure.clear()
            except Exception:
                pass
        self._viewer_figures.clear()
