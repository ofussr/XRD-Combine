"""Интерактивная полюсная фигура для Bruker RAW v3/v4 и нумерованных XY.

Сначала программа пытается прочитать выбранный RAW напрямую. При неудаче она
ищет рядом ``name_exported_0.xy``, ``name_exported_1.xy`` и далее.
Интерполяция не выполняется ни в одном из режимов.

Requirements:
    Python 3.10+
    numpy
    matplotlib
    tkinter (normally included with Python on Windows)
"""

from __future__ import annotations

import math
import re
import struct
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import colorchooser, ttk

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, PowerNorm
from matplotlib.figure import Figure

try:
    from .controls import CollapsibleSection, ScrollableControls
    from .bruker_raw import read_bruker_raw
    from .i18n import (
        LocalizedStringVar,
        apply_language,
        filedialog,
        load_language,
        localised,
        messagebox,
        set_language,
        translate_text,
    )
except ImportError:
    from controls import CollapsibleSection, ScrollableControls
    from bruker_raw import read_bruker_raw
    from i18n import (
        LocalizedStringVar,
        apply_language,
        filedialog,
        load_language,
        localised,
        messagebox,
        set_language,
        translate_text,
    )


WINDOW_TITLE = "Экспериментальная полюсная фигура"
POLE_COLOURS = LinearSegmentedColormap.from_list(
    "white_viridis",
    [(1.0, 1.0, 1.0, 1.0), *colormaps["viridis"](np.linspace(0.0, 1.0, 256))],
    N=256,
)
POLE_COLOURS.set_bad("white")
POLE_COLOURS.set_under("white")

RAW3_FILE_HEADER_SIZE = 712
RAW3_RANGE_HEADER_SIZE = 304
RAW3_MAGIC = b"RAW1.01\x00"


@dataclass
class BrukerRawRange:
    index: int
    theta: float
    two_theta: float
    chi: float
    phi_start: float
    phi_step: float
    time_per_step: float
    intensity: np.ndarray

    @property
    def phi(self) -> np.ndarray:
        return self.phi_start + self.phi_step * np.arange(self.intensity.size)


@dataclass
class BrukerRawPoleData:
    declared_ranges: int
    status_code: int
    ranges: list[BrukerRawRange]


def _raw_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _raw_f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _raw_f64(data: bytes, offset: int) -> float:
    return struct.unpack_from("<d", data, offset)[0]


def read_bruker_raw3(path: Path) -> BrukerRawPoleData:
    """Read a Siemens/Bruker RAW version 3 pole measurement (RAW1.01)."""
    data = path.read_bytes()
    if len(data) < RAW3_FILE_HEADER_SIZE:
        raise ValueError("The file is shorter than the RAW1.01 file header.")
    if data[:8] != RAW3_MAGIC:
        signature = data[:8].rstrip(b"\x00")
        raise ValueError(f"Unsupported RAW signature: {signature!r}.")

    status_code = _raw_u32(data, 8)
    declared_ranges = _raw_u32(data, 12)
    if declared_ranges == 0:
        raise ValueError("The RAW file declares no measurement ranges.")
    if declared_ranges > 100_000:
        raise ValueError(f"Implausible RAW range count: {declared_ranges}.")

    ranges: list[BrukerRawRange] = []
    offset = RAW3_FILE_HEADER_SIZE

    for index in range(declared_ranges):
        if offset + RAW3_RANGE_HEADER_SIZE > len(data):
            raise ValueError(
                f"Range {index}: its header extends beyond the end of the RAW file."
            )

        header_size = _raw_u32(data, offset)
        points = _raw_u32(data, offset + 4)
        supplementary_size = _raw_u32(data, offset + 256)
        if header_size != RAW3_RANGE_HEADER_SIZE:
            raise ValueError(
                f"Range {index}: expected a {RAW3_RANGE_HEADER_SIZE}-byte header, "
                f"found {header_size} bytes."
            )

        data_offset = offset + header_size + supplementary_size
        data_end = data_offset + 4 * points
        if data_offset > len(data) or data_end > len(data):
            available = max(0, (len(data) - data_offset) // 4)
            raise ValueError(
                f"Range {index}: {points} points are declared, but only "
                f"{available} are available."
            )

        intensity = np.frombuffer(
            data,
            dtype="<f4",
            count=points,
            offset=data_offset,
        ).astype(float, copy=True)
        theta = _raw_f64(data, offset + 8)
        two_theta = _raw_f64(data, offset + 16)
        chi = _raw_f64(data, offset + 24)
        phi_start = _raw_f64(data, offset + 32)
        phi_step = _raw_f64(data, offset + 176)
        time_per_step = _raw_f32(data, offset + 192)

        numeric_header = (theta, two_theta, chi, phi_start, phi_step, time_per_step)
        if not all(math.isfinite(value) for value in numeric_header):
            raise ValueError(f"Range {index}: its angular header contains invalid values.")
        if points > 1 and math.isclose(phi_step, 0.0, abs_tol=1e-15):
            raise ValueError(f"Range {index}: Phi step is zero for {points} points.")

        ranges.append(
            BrukerRawRange(
                index=index,
                theta=theta,
                two_theta=two_theta,
                chi=chi,
                phi_start=phi_start,
                phi_step=phi_step,
                time_per_step=time_per_step,
                intensity=intensity,
            )
        )
        offset = data_end

    trailing = data[offset:]
    if trailing and any(byte != 0 for byte in trailing):
        raise ValueError(f"The RAW file contains {len(trailing)} unexpected trailing bytes.")

    return BrukerRawPoleData(
        declared_ranges=declared_ranges,
        status_code=status_code,
        ranges=ranges,
    )


def prepare_scan(
    phi: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove only a duplicated 360° endpoint, preserving all measured gaps."""
    if phi.size > 1 and phi[0] > phi[-1]:
        phi = phi[::-1]
        intensity = intensity[::-1]
    if phi.size > 2 and math.isclose(phi[-1] - phi[0], 360.0, abs_tol=0.05):
        return phi[:-1], intensity[:-1]
    return phi, intensity


def scans_from_raw(
    raw: BrukerRawPoleData,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Return per-range original Phi grids and their measured Chi positions."""
    ordered = sorted(raw.ranges, key=lambda scan: scan.chi)
    radii = np.asarray([scan.chi for scan in ordered], dtype=float)
    if radii.size == 0:
        raise ValueError("The RAW file contains no ranges.")
    if np.any(radii < 0):
        raise ValueError("The RAW file contains negative Chi values.")
    if radii.size > 1 and np.any(np.diff(radii) <= 0):
        raise ValueError("The RAW file contains duplicate Chi values.")

    scans = [prepare_scan(scan.phi, scan.intensity) for scan in ordered]
    if not any(intensity.size for _phi, intensity in scans):
        raise ValueError("The RAW file contains no recorded intensity points.")
    if not any(np.any(np.isfinite(intensity)) for _phi, intensity in scans):
        raise ValueError("The RAW file contains no finite intensity values.")
    return scans, radii


def read_xy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two numeric columns while ignoring arbitrary headers."""
    angles: list[float] = []
    intensities: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                angle = float(parts[0])
                intensity = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(angle) and math.isfinite(intensity):
                angles.append(angle)
                intensities.append(intensity)

    if len(angles) < 2:
        raise ValueError(f"No usable two-column numeric data found in {path.name}.")

    angle_array = np.asarray(angles, dtype=float)
    intensity_array = np.asarray(intensities, dtype=float)
    order = np.argsort(angle_array)
    angle_array = angle_array[order]
    intensity_array = intensity_array[order]

    # Keep one value per angle so the plotting cells remain unambiguous.
    unique_angles, unique_indices = np.unique(angle_array, return_index=True)
    return unique_angles, intensity_array[unique_indices]


def find_exported_xy_files(raw_path: Path) -> list[Path]:
    """Return matching XY exports in their numeric order."""
    pattern = re.compile(
        rf"^{re.escape(raw_path.stem)}_exported_(\d+)\.xy$", re.IGNORECASE
    )
    matches: list[tuple[int, Path]] = []

    for candidate in raw_path.parent.iterdir():
        if not candidate.is_file():
            continue
        match = pattern.match(candidate.name)
        if match:
            matches.append((int(match.group(1)), candidate))

    if not matches:
        return []

    matches.sort(key=lambda item: item[0])
    indices = [item[0] for item in matches]
    expected = list(range(0, indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        missing_text = ", ".join(str(index) for index in missing)
        raise ValueError(f"Missing exported XY file number(s): {missing_text}.")

    return [item[1] for item in matches]


def load_xy_series(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load every XY file on its original Phi grid without interpolation."""
    scans: list[tuple[np.ndarray, np.ndarray]] = []

    for path in paths:
        phi, intensity = read_xy_file(path)
        scans.append(prepare_scan(phi, intensity))

    if not scans:
        raise ValueError("No XY data were loaded.")
    return scans


def split_continuous_segments(
    phi: np.ndarray, intensity: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a scan at missing angular intervals so gaps remain unpainted."""
    if phi.size == 0:
        return []
    if phi.size < 2:
        return [(phi, intensity)]

    differences = np.diff(phi)
    positive_differences = differences[differences > 0]
    if positive_differences.size == 0:
        return [(phi, intensity)]

    usual_step = float(np.median(positive_differences))
    break_points = np.flatnonzero(differences > usual_step * 1.5) + 1
    phi_parts = np.split(phi, break_points)
    intensity_parts = np.split(intensity, break_points)
    return [
        (phi_part, intensity_part)
        for phi_part, intensity_part in zip(phi_parts, intensity_parts)
        if phi_part.size > 0
    ]


def centres_to_edges(values: np.ndarray, single_width: float = 1.0) -> np.ndarray:
    """Convert monotonically increasing cell centres to pcolormesh edges."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        half_width = single_width / 2.0
        return np.array([values[0] - half_width, values[0] + half_width])

    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def format_number(value: float) -> str:
    return f"{value:.10g}"


def scaled_axes_position(
    position: tuple[float, float, float, float],
    cursor: tuple[float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    """Magnify an already drawn axes around a cursor in figure coordinates."""

    x0, y0, width, height = position
    cursor_x, cursor_y = cursor
    return (
        cursor_x - (cursor_x - x0) * scale,
        cursor_y - (cursor_y - y0) * scale,
        width * scale,
        height * scale,
    )


class PoleFigureApp:
    def __init__(self, root: tk.Misc, on_open_raw=None) -> None:
        self.root = root
        self.on_open_raw = on_open_raw
        if isinstance(self.root, (tk.Tk, tk.Toplevel)):
            self.root.title(WINDOW_TITLE)
            self.root.geometry("1200x820")
            self.root.minsize(920, 650)

        self.raw_path: Path | None = None
        self.scans: list[tuple[np.ndarray, np.ndarray]] = []
        self.loaded_radii: np.ndarray | None = None
        self.data_source_description = "data"
        self.displayed_radii: np.ndarray | None = None
        self.displayed_radius_edges: np.ndarray | None = None
        self.displayed_scans: list[tuple[np.ndarray, np.ndarray]] = []
        self.plot_axis = None
        self.base_plot_position: tuple[float, float, float, float] | None = None
        self.zoom_factor = 1.0
        self.redraw_job: str | None = None
        self.angle_update_guard = False
        self.limit_update_guard = False

        self.first_angle = tk.StringVar()
        self.angle_step = tk.StringVar()
        self.last_angle = tk.StringVar()
        self.display_mode = tk.StringVar(value="colour")
        self.scale_mode = tk.StringVar(value="linear")
        self.fill_colour = tk.StringVar(value="#2a788e")
        self.lower_limit = tk.StringVar(value="0")
        self.upper_limit = tk.StringVar(value="1")
        self.file_text = LocalizedStringVar(
            value=localised("No RAW selected", "Aucun RAW sélectionné", "RAW не выбран")
        )
        self.status_text = LocalizedStringVar(
            value=localised("Select a RAW file.", "Sélectionnez un fichier RAW.", "Выберите RAW-файл.")
        )
        self.cursor_text = LocalizedStringVar(
            value=localised(
                "Click the figure to inspect a point.",
                "Cliquez sur la figure pour examiner un point.",
                "Щёлкните по фигуре для чтения точки.",
            )
        )

        self._build_interface()
        self._attach_traces()
        self._show_placeholder()
        apply_language(self.root)

    def _build_interface(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.control_panel = ScrollableControls(self.root, width=320, padding=12)
        self.control_panel.grid(row=0, column=0, sticky="nsew")
        controls = self.control_panel.body
        controls.columnconfigure(0, weight=1)

        files = CollapsibleSection(controls, text="Данные", padding=10)
        files.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(files, text="Открыть RAW…", command=self.select_raw).pack(fill="x")
        ttk.Label(files, textvariable=self.file_text, wraplength=260,
                  justify="left").pack(fill="x", pady=(6, 0))

        angles = CollapsibleSection(controls, text="Ряд углов наклона", padding=10)
        angles.grid(row=3, column=0, sticky="ew")
        angles.columnconfigure(1, weight=1)

        angle_fields = (
            ("Первый угол, °", self.first_angle, "first"),
            ("Шаг, °", self.angle_step, "step"),
            ("Последний угол, °", self.last_angle, "last"),
        )
        self.angle_entries: dict[str, ttk.Entry] = {}
        for row, (label, variable, name) in enumerate(angle_fields):
            ttk.Label(angles, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
            entry = ttk.Entry(angles, textvariable=variable, width=14)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.angle_entries[name] = entry

        ttk.Label(
            angles,
            text=(
                "Углы из RAW заполняются автоматически. Для резервных XY задайте "
                "любые два значения – третье будет вычислено."
            ),
            wraplength=240,
            foreground="#555555",
            justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(7, 0))

        display = CollapsibleSection(controls, text="Отображение", padding=10)
        display.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        display.columnconfigure(1, weight=1)
        ttk.Radiobutton(
            display,
            text="Цветовая шкала",
            value="colour",
            variable=self.display_mode,
            command=self.on_display_mode_changed,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            display,
            text="Сплошная заливка",
            value="solid",
            variable=self.display_mode,
            command=self.on_display_mode_changed,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.fill_colour_button = tk.Button(
            display,
            text="Цвет заливки…",
            command=self.choose_fill_colour,
            background=self.fill_colour.get(),
            activebackground=self.fill_colour.get(),
            disabledforeground="#777777",
            state="disabled",
        )
        self.fill_colour_button.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))

        scaling = CollapsibleSection(controls, text="Шкала интенсивности", padding=10)
        scaling.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        self.scale_buttons: list[ttk.Radiobutton] = []
        for column, (text, value) in enumerate(
            (("Линейная", "linear"), ("Лог.", "log"), ("Квадрат", "square"))
        ):
            button = ttk.Radiobutton(
                scaling,
                text=text,
                value=value,
                variable=self.scale_mode,
                command=self.on_scale_mode_changed,
            )
            button.grid(row=0, column=column, sticky="w", padx=(0, 7))
            self.scale_buttons.append(button)

        limits = CollapsibleSection(controls, text="Границы интенсивности", padding=10)
        limits.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        limits.columnconfigure(1, weight=1)

        ttk.Label(limits, text="Нижняя").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.lower_entry = ttk.Entry(limits, textvariable=self.lower_limit, width=16)
        self.lower_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(limits, text="Верхняя").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.upper_entry = ttk.Entry(limits, textvariable=self.upper_limit, width=16)
        self.upper_entry.grid(row=1, column=1, sticky="ew", pady=2)

        selected_point = CollapsibleSection(controls, text="Выбранная точка", padding=10)
        selected_point.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            selected_point,
            textvariable=self.cursor_text,
            wraplength=240,
            justify="left",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            selected_point,
            text="Колесо мыши масштабирует фигуру вокруг курсора.",
            wraplength=240,
            justify="left",
            foreground="#555555",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 4))
        self.reset_zoom_button = ttk.Button(
            selected_point,
            text="Сбросить масштаб",
            command=self.reset_zoom,
            state="disabled",
        )
        self.reset_zoom_button.grid(row=2, column=0, sticky="ew")

        self.save_button = ttk.Button(
            controls, text="Сохранить рисунок…", command=self.save_figure, state="disabled"
        )
        self.save_button.grid(row=8, column=0, sticky="ew", pady=(14, 0))

        ttk.Separator(controls).grid(row=9, column=0, sticky="ew", pady=(16, 10))
        ttk.Label(
            controls,
            textvariable=self.status_text,
            wraplength=270,
            justify="left",
        ).grid(row=10, column=0, sticky="ew")

        plot_frame = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.canvas.mpl_connect("scroll_event", self.on_plot_scroll)

    def _attach_traces(self) -> None:
        self.first_angle.trace_add("write", lambda *_: self.on_angle_changed("first"))
        self.angle_step.trace_add("write", lambda *_: self.on_angle_changed("step"))
        self.last_angle.trace_add("write", lambda *_: self.on_angle_changed("last"))
        self.lower_limit.trace_add("write", lambda *_: self.on_limit_changed())
        self.upper_limit.trace_add("write", lambda *_: self.on_limit_changed())

    def _show_placeholder(self) -> None:
        self.figure.clear()
        if hasattr(self.figure, "set_layout_engine"):
            self.figure.set_layout_engine("constrained")
        self.plot_axis = None
        self.base_plot_position = None
        self.zoom_factor = 1.0
        axis = self.figure.add_subplot(111)
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            localised(
                "Open a RAW file",
                "Ouvrez un fichier RAW",
                "Откройте RAW-файл",
            ),
            ha="center",
            va="center",
            color="#666666",
            fontsize=13,
            transform=axis.transAxes,
        )
        self.canvas.draw_idle()

    def clear_data(self) -> None:
        if self.redraw_job is not None:
            self.root.after_cancel(self.redraw_job)
            self.redraw_job = None
        self.raw_path = None
        self.scans = []
        self.loaded_radii = None
        self.displayed_radii = None
        self.displayed_radius_edges = None
        self.displayed_scans = []
        self.file_text.set(localised("No RAW selected", "Aucun RAW sélectionné", "RAW не выбран"))
        self.status_text.set(localised("Select a RAW file.", "Sélectionnez un fichier RAW.", "Выберите RAW-файл."))
        self.cursor_text.set(
            localised(
                "Click the figure to inspect a point.",
                "Cliquez sur la figure pour examiner un point.",
                "Щёлкните по фигуре для чтения точки.",
            )
        )
        self.save_button.configure(state="disabled")
        self.reset_zoom_button.configure(state="disabled")
        self._show_placeholder()

    def select_raw(self, selected=None, raw_data=None) -> None:
        if selected is None:
            selected = filedialog.askopenfilename(
                title="Открыть RAW",
                filetypes=(("RAW", "*.raw"), ("Все файлы", "*.*")),
            )
            if selected and self.on_open_raw is not None:
                self.on_open_raw(selected)
                return
        if not selected:
            return

        raw_path = Path(selected)
        raw_error: str | None = None
        loaded_radii: np.ndarray | None = None
        direct_raw = False

        try:
            if raw_data is None:
                raw_data = read_bruker_raw(raw_path)
            if not raw_data.is_pole_figure:
                raise ValueError(
                    "RAW прочитан, но не распознан как полюсная съёмка "
                    "(ожидаются диапазоны по φ при разных χ)."
                )
            ordered = sorted(raw_data.ranges, key=lambda scan: scan.chi)
            loaded_radii = np.asarray([scan.chi for scan in ordered], dtype=float)
            if loaded_radii.size > 1 and np.any(np.diff(loaded_radii) <= 0):
                raise ValueError("В RAW повторяются или нарушен порядок значений χ.")
            scans = [prepare_scan(scan.phi, scan.intensity) for scan in ordered]
            direct_raw = True
            self.data_source_description = localised(
                f"{len(scans)} RAW ranges",
                f"{len(scans)} plages RAW",
                f"{len(scans)} диапазонов RAW",
            )
        except (OSError, ValueError, struct.error) as error:
            raw_error = str(error)
            try:
                xy_paths = find_exported_xy_files(raw_path)
                if not xy_paths:
                    raise ValueError(
                        f"Рядом с RAW не найдены файлы "
                        f"{raw_path.stem}_exported_N.xy."
                    )
                scans = load_xy_series(xy_paths)
                self.data_source_description = localised(
                    f"{len(scans)} fallback XY files",
                    f"{len(scans)} fichiers XY de secours",
                    f"{len(scans)} резервных файлов XY",
                )
            except (OSError, ValueError) as xy_error:
                error_text = localised(
                    "The RAW file could not be read directly, and no usable "
                    "fallback XY series was found.",
                    "Le fichier RAW n’a pas pu être lu directement et aucune série "
                    "XY de secours exploitable n’a été trouvée.",
                    f"RAW не удалось прочитать напрямую:\n{raw_error}\n\n"
                    f"Резервная загрузка XY также не удалась:\n{xy_error}",
                )
                messagebox.showerror("Ошибка чтения измерения", error_text)
                self.status_text.set("RAW и резервные XY не прочитаны.")
                return

        self.raw_path = raw_path
        self.scans = scans
        self.loaded_radii = loaded_radii
        self.displayed_radii = None
        self.displayed_radius_edges = None
        self.displayed_scans = []
        self.base_plot_position = None
        self.zoom_factor = 1.0
        self.reset_zoom_button.configure(state="disabled")
        self.cursor_text.set("Щёлкните по фигуре для чтения точки.")
        self.file_text.set(raw_path.name)
        point_counts = [phi.size for phi, _intensity in scans]
        if min(point_counts) == max(point_counts):
            points_text = localised(
                f"{point_counts[0]} φ points per range",
                f"{point_counts[0]} points φ par plage",
                f"{point_counts[0]} точек φ в диапазоне",
            )
        else:
            points_text = localised(
                f"{min(point_counts)} to {max(point_counts)} φ points per range",
                f"de {min(point_counts)} à {max(point_counts)} points φ par plage",
                f"от {min(point_counts)} до {max(point_counts)} точек φ в диапазоне",
            )
        if direct_raw:
            empty_ranges = sum(count == 0 for count in point_counts)
            empty_text = (
                localised(
                    f"; empty ranges: {empty_ranges}",
                    f" ; plages vides : {empty_ranges}",
                    f"; пустых диапазонов: {empty_ranges}",
                )
                if empty_ranges
                else ""
            )
            self.status_text.set(localised(
                f"RAW ranges read directly: {len(scans)}; {points_text}{empty_text}.",
                f"Plages RAW lues directement : {len(scans)} ; {points_text}{empty_text}.",
                f"Напрямую прочитано диапазонов RAW: {len(scans)}; "
                f"{points_text}{empty_text}.",
            ))
        else:
            self.status_text.set(localised(
                f"Direct RAW reading failed. Loaded matching XY files: "
                f"{len(scans)}; {points_text}.",
                f"La lecture directe du RAW a échoué. Fichiers XY "
                f"correspondants chargés : {len(scans)} ; {points_text}.",
                f"Прямое чтение RAW не удалось ({raw_error}). Загружены подходящие "
                f"XY: {len(scans)}; {points_text}.",
            ))

        self.angle_update_guard = True
        try:
            if loaded_radii is None:
                self.first_angle.set("")
                self.angle_step.set("")
                self.last_angle.set("")
            else:
                self.first_angle.set(format_number(float(loaded_radii[0])))
                self.last_angle.set(format_number(float(loaded_radii[-1])))
                if loaded_radii.size == 1:
                    self.angle_step.set("0")
                else:
                    differences = np.diff(loaded_radii)
                    if np.allclose(differences, differences[0], rtol=0.0, atol=1e-8):
                        self.angle_step.set(format_number(float(differences[0])))
                    else:
                        self.angle_step.set("переменный")
        finally:
            self.angle_update_guard = False

        nonempty_intensities = [
            intensity for _phi, intensity in scans if intensity.size > 0
        ]
        finite = np.concatenate(nonempty_intensities)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            messagebox.showerror(
                "Ошибка чтения измерения",
                "В выбранном измерении нет конечных значений интенсивности.",
            )
            self.status_text.set("В измерении нет конечных значений интенсивности.")
            return
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        if math.isclose(data_min, data_max):
            data_max = data_min + 1.0
        self._configure_limits(data_min, data_max)
        self.save_button.configure(state="disabled")
        if direct_raw:
            self.draw_pole_figure()
        else:
            self._show_placeholder()
            self.angle_entries["first"].focus_set()

    @staticmethod
    def _parse(variable: tk.StringVar) -> float | None:
        text = variable.get().strip().replace(",", ".")
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    def on_angle_changed(self, changed: str) -> None:
        if self.angle_update_guard or not self.scans:
            return

        # A manual edit intentionally replaces the Chi values read from RAW
        # with a uniformly spaced user-defined series.
        self.loaded_radii = None
        first = self._parse(self.first_angle)
        step = self._parse(self.angle_step)
        last = self._parse(self.last_angle)
        count = len(self.scans)
        values = {"first": first, "step": step, "last": last}

        self.angle_update_guard = True
        try:
            if count == 1:
                if first is not None:
                    self.last_angle.set(format_number(first))
                    if step is None:
                        self.angle_step.set("0")
            elif sum(value is not None for value in values.values()) >= 2:
                if first is None and step is not None and last is not None:
                    self.first_angle.set(format_number(last - step * (count - 1)))
                elif step is None and first is not None and last is not None:
                    self.angle_step.set(format_number((last - first) / (count - 1)))
                elif last is None and first is not None and step is not None:
                    self.last_angle.set(format_number(first + step * (count - 1)))
                elif first is not None and step is not None and last is not None:
                    if changed == "last":
                        self.angle_step.set(format_number((last - first) / (count - 1)))
                    else:
                        self.last_angle.set(format_number(first + step * (count - 1)))
        finally:
            self.angle_update_guard = False

        self.schedule_redraw()

    def _validated_radii(self) -> np.ndarray | None:
        if not self.scans:
            return None
        if self.loaded_radii is not None:
            return self.loaded_radii.copy()
        first = self._parse(self.first_angle)
        step = self._parse(self.angle_step)
        last = self._parse(self.last_angle)
        if first is None or step is None or last is None:
            return None

        count = len(self.scans)
        radii = first + step * np.arange(count, dtype=float)
        if np.any(radii < 0):
            self.status_text.set("Углы наклона не могут быть отрицательными.")
            return None
        if count > 1 and math.isclose(step, 0.0, abs_tol=1e-15):
            self.status_text.set("Для нескольких диапазонов шаг угла не может быть нулевым.")
            return None
        return radii

    def _configure_limits(self, data_min: float, data_max: float) -> None:
        self.limit_update_guard = True
        try:
            self.lower_limit.set(format_number(data_min))
            self.upper_limit.set(format_number(data_max))
        finally:
            self.limit_update_guard = False

    def _minimum_positive(self) -> float | None:
        if not self.scans:
            return None
        positive_rows = [
            intensity[intensity > 0] for _phi, intensity in self.scans
        ]
        positive = np.concatenate(positive_rows)
        return float(np.min(positive)) if positive.size else None

    def on_display_mode_changed(self) -> None:
        solid = self.display_mode.get() == "solid"
        for button in self.scale_buttons:
            button.configure(state="disabled" if solid else "normal")
        self.fill_colour_button.configure(state="normal" if solid else "disabled")
        if not solid and self.scale_mode.get() == "log":
            self.on_scale_mode_changed()
        else:
            self.schedule_redraw()

    def choose_fill_colour(self) -> None:
        _rgb, selected = colorchooser.askcolor(
            color=self.fill_colour.get(),
            parent=self.root,
            title=translate_text("Выбрать цвет заливки"),
        )
        if not selected:
            return
        self.fill_colour.set(selected)
        self.fill_colour_button.configure(background=selected, activebackground=selected)
        self.schedule_redraw()

    def on_scale_mode_changed(self) -> None:
        if not self.scans:
            return
        low = self._parse(self.lower_limit)
        if self.scale_mode.get() == "log" and (low is None or low <= 0):
            minimum_positive = self._minimum_positive()
            if minimum_positive is None:
                messagebox.showerror(
                    "Логарифмическая шкала недоступна",
                    "В данных нет положительной интенсивности.",
                )
                self.scale_mode.set("linear")
            else:
                self.limit_update_guard = True
                try:
                    self.lower_limit.set(format_number(minimum_positive))
                finally:
                    self.limit_update_guard = False
        self.schedule_redraw()

    def on_limit_changed(self) -> None:
        if self.limit_update_guard or not self.scans:
            return

        low = self._parse(self.lower_limit)
        high = self._parse(self.upper_limit)
        if low is None or high is None:
            return
        if low >= high:
            self.status_text.set("Нижняя граница должна быть меньше верхней.")
            self.save_button.configure(state="disabled")
            return
        if (
            self.display_mode.get() == "colour"
            and self.scale_mode.get() == "log"
            and low <= 0
        ):
            self.status_text.set(
                "Для логарифмической шкалы нижняя граница должна быть положительной."
            )
            self.save_button.configure(state="disabled")
            return
        self.schedule_redraw(delay=180)

    def schedule_redraw(self, delay: int = 150) -> None:
        if self.redraw_job is not None:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(delay, self.draw_pole_figure)

    def draw_pole_figure(self) -> None:
        self.redraw_job = None
        if self.raw_path is None or not self.scans:
            return

        radii = self._validated_radii()
        if radii is None:
            self.save_button.configure(state="disabled")
            return

        scans = self.scans
        if radii.size > 1 and radii[0] > radii[-1]:
            radii = radii[::-1]
            scans = list(reversed(scans))

        low = self._parse(self.lower_limit)
        high = self._parse(self.upper_limit)
        if low is None or high is None or not low < high:
            return

        display_mode = self.display_mode.get()
        scale_mode = self.scale_mode.get()
        if display_mode == "solid":
            plot_cmap = LinearSegmentedColormap.from_list(
                "solid_fill",
                [self.fill_colour.get(), self.fill_colour.get()],
                N=2,
            )
            plot_cmap.set_bad("white")
            norm = Normalize(vmin=0.0, vmax=1.0)
            scale_title = None
        elif scale_mode == "log":
            minimum_positive = self._minimum_positive()
            if minimum_positive is None:
                return
            low = max(low, minimum_positive)
            plot_cmap = POLE_COLOURS
            norm = LogNorm(vmin=low, vmax=high, clip=True)
            scale_title = localised("logarithmic", "logarithmique", "логарифмическая")
        elif scale_mode == "square":
            plot_cmap = POLE_COLOURS
            norm = PowerNorm(gamma=2.0, vmin=low, vmax=high, clip=True)
            scale_title = localised("square", "quadratique", "квадратная")
        else:
            plot_cmap = POLE_COLOURS
            norm = Normalize(vmin=low, vmax=high, clip=True)
            scale_title = localised("linear", "linéaire", "линейная")

        radial_step = abs(float(self._parse(self.angle_step) or 1.0))
        radius_edges = centres_to_edges(radii, single_width=max(radial_step, 1.0))
        radius_edges[0] = max(0.0, radius_edges[0])

        self.figure.clear()
        if hasattr(self.figure, "set_layout_engine"):
            self.figure.set_layout_engine("constrained")
        axis = self.figure.add_subplot(111, projection="polar")
        self.plot_axis = axis
        mesh = None
        for row, (phi, intensity) in enumerate(scans):
            for phi_part, intensity_part in split_continuous_segments(phi, intensity):
                theta_edges = np.deg2rad(centres_to_edges(phi_part))
                if display_mode == "solid":
                    inside_limits = (intensity_part >= low) & (intensity_part <= high)
                    plot_data = np.ma.masked_where(
                        ~inside_limits[np.newaxis, :],
                        np.ones((1, intensity_part.size), dtype=float),
                    )
                else:
                    plot_data = intensity_part[np.newaxis, :]
                if display_mode == "colour" and scale_mode == "log":
                    plot_data = np.ma.masked_less_equal(plot_data, 0)
                mesh = axis.pcolormesh(
                    theta_edges,
                    radius_edges[row : row + 2],
                    plot_data,
                    cmap=plot_cmap,
                    norm=norm,
                    shading="flat",
                    rasterized=True,
                )

        if mesh is None:
            self.status_text.set("Не найдено данных, пригодных для построения.")
            self.save_button.configure(state="disabled")
            return
        self.displayed_radii = radii.copy()
        self.displayed_radius_edges = radius_edges.copy()
        self.displayed_scans = scans
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_ylim(0.0, max(float(radius_edges[-1]), 1e-9))
        axis.set_rlabel_position(135)
        axis.grid(True, color="#8a8a8a", alpha=0.70, linewidth=0.8)
        axis.set_axisbelow(False)
        axis.set_title(
            localised(
                f"{self.raw_path.stem}\nPole figure",
                f"{self.raw_path.stem}\nFigure de pôles",
                f"{self.raw_path.stem}\nПолюсная фигура",
            ),
            pad=20,
        )

        if display_mode == "colour":
            colour_bar = self.figure.colorbar(mesh, ax=axis, pad=0.10, shrink=0.84)
            colour_bar.set_label(
                localised(
                    f"Intensity ({scale_title})",
                    f"Intensité ({scale_title})",
                    f"Интенсивность ({scale_title})",
                )
            )
        self.canvas.draw()
        self.base_plot_position = tuple(
            float(value) for value in axis.get_position().bounds
        )
        self.zoom_factor = 1.0
        if hasattr(self.figure, "set_layout_engine"):
            self.figure.set_layout_engine("none")
        self.save_button.configure(state="normal")
        self.reset_zoom_button.configure(state="disabled")
        if display_mode == "solid":
            mode_text = localised(
                f"solid fill from {format_number(low)} to {format_number(high)}",
                f"remplissage uni de {format_number(low)} à {format_number(high)}",
                f"заливка от {format_number(low)} до {format_number(high)}",
            )
        else:
            mode_text = localised(
                f"{scale_title} colour scale",
                f"échelle de couleurs {scale_title}",
                f"{scale_title} цветовая шкала",
            )
        self.status_text.set(localised(
            f"Pole figure built from {self.data_source_description} without "
            f"interpolation. {mode_text}. χ range: "
            f"{format_number(radii[0])}–{format_number(radii[-1])}°.",
            f"Figure de pôles construite à partir de {self.data_source_description} "
            f"sans interpolation. {mode_text}. Intervalle χ : "
            f"{format_number(radii[0])}–{format_number(radii[-1])}°.",
            f"Полюсная фигура построена по {self.data_source_description} "
            f"без интерполяции. {mode_text}. Диапазон χ: "
            f"{format_number(radii[0])}–{format_number(radii[-1])}°.",
        ))

    def on_plot_scroll(self, event: object) -> None:
        """Magnify the finished pole-figure drawing without changing χ limits."""

        axis = getattr(event, "inaxes", None)
        button = getattr(event, "button", None)
        if (
            axis is None
            or axis is not self.plot_axis
            or getattr(axis, "name", "") != "polar"
            or button not in {"up", "down"}
            or self.base_plot_position is None
            or getattr(event, "x", None) is None
            or getattr(event, "y", None) is None
        ):
            return
        previous_factor = self.zoom_factor
        if button == "up":
            self.zoom_factor = min(8.0, self.zoom_factor * 1.25)
        else:
            self.zoom_factor = max(1.0, self.zoom_factor / 1.25)
        if math.isclose(self.zoom_factor, previous_factor):
            return
        if math.isclose(self.zoom_factor, 1.0):
            axis.set_position(self.base_plot_position)
        else:
            cursor_x, cursor_y = self.figure.transFigure.inverted().transform(
                (float(event.x), float(event.y))
            )
            position = tuple(float(value) for value in axis.get_position().bounds)
            axis.set_position(
                scaled_axes_position(
                    position,
                    (float(cursor_x), float(cursor_y)),
                    self.zoom_factor / previous_factor,
                )
            )
        self.reset_zoom_button.configure(
            state="normal" if self.zoom_factor > 1.0 else "disabled"
        )
        self.canvas.draw_idle()

    def reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        if self.plot_axis is not None and self.base_plot_position is not None:
            self.plot_axis.set_position(self.base_plot_position)
            self.canvas.draw_idle()
        self.reset_zoom_button.configure(state="disabled")

    def on_plot_click(self, event: object) -> None:
        """Show the measured point under the cursor without changing the figure."""
        if getattr(event, "button", None) != 1:
            return
        axis = getattr(event, "inaxes", None)
        if axis is None or getattr(axis, "name", "") != "polar":
            return
        theta = getattr(event, "xdata", None)
        radius = getattr(event, "ydata", None)
        if theta is None or radius is None:
            return
        if (
            self.displayed_radii is None
            or self.displayed_radius_edges is None
            or not self.displayed_scans
        ):
            return

        radius_edges = self.displayed_radius_edges
        if radius < radius_edges[0] or radius > radius_edges[-1]:
            self.cursor_text.set(
                localised(
                    "There is no measured point at this position.",
                    "Aucun point mesuré à cette position.",
                    "В этой позиции нет измеренной точки.",
                )
            )
            return

        row = int(np.searchsorted(radius_edges, radius, side="right") - 1)
        row = min(max(row, 0), len(self.displayed_scans) - 1)
        phi, intensity = self.displayed_scans[row]
        clicked_phi = math.degrees(theta) % 360.0

        for phi_part, intensity_part in split_continuous_segments(phi, intensity):
            angular_edges = centres_to_edges(phi_part)
            for candidate_phi in (
                clicked_phi - 360.0,
                clicked_phi,
                clicked_phi + 360.0,
            ):
                if not angular_edges[0] <= candidate_phi <= angular_edges[-1]:
                    continue
                column = int(
                    np.searchsorted(angular_edges, candidate_phi, side="right") - 1
                )
                column = min(max(column, 0), phi_part.size - 1)
                self.cursor_text.set(
                    localised(
                        f"φ: {format_number(float(phi_part[column]))}°\n"
                        f"χ: {format_number(float(self.displayed_radii[row]))}°\n"
                        f"Intensity: {format_number(float(intensity_part[column]))}",
                        f"φ : {format_number(float(phi_part[column]))}°\n"
                        f"χ : {format_number(float(self.displayed_radii[row]))}°\n"
                        f"Intensité : {format_number(float(intensity_part[column]))}",
                        f"φ: {format_number(float(phi_part[column]))}°\n"
                        f"χ: {format_number(float(self.displayed_radii[row]))}°\n"
                        f"Интенсивность: {format_number(float(intensity_part[column]))}",
                    )
                )
                return

        self.cursor_text.set(
            localised(
                "There is no measured point at this position.",
                "Aucun point mesuré à cette position.",
                "В этой позиции нет измеренной точки.",
            )
        )

    def save_figure(self) -> None:
        if self.raw_path is None or not self.figure.axes:
            return
        default_name = f"{self.raw_path.stem}_pole_figure.png"
        selected = filedialog.asksaveasfilename(
            title="Сохранить полюсную фигуру",
            initialdir=str(self.raw_path.parent),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=(
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("TIFF", "*.tif *.tiff"),
                ("Все файлы", "*.*"),
            ),
        )
        if not selected:
            return
        try:
            self.figure.savefig(selected, dpi=300, bbox_inches="tight")
        except (OSError, ValueError) as error:
            messagebox.showerror("Ошибка сохранения", str(error))
            return
        self.status_text.set(
            localised(
                f"Figure saved: {Path(selected).name}.",
                f"Figure enregistrée : {Path(selected).name}.",
                f"Рисунок сохранён: {Path(selected).name}.",
            )
        )


def main() -> None:
    set_language(load_language())
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    PoleFigureApp(root)
    root.title(localised(
        "Experimental pole figure",
        "Figure de pôles expérimentale",
        "Экспериментальная полюсная фигура",
    ))
    apply_language(root)
    root.mainloop()


if __name__ == "__main__":
    main()
