"""Native Qt experimental pole figures on the measured RAW/XY grids."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, PowerNorm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from ..localization import tr, localised, translate_text
from ..services.experimental_pole import (
    ExperimentalPoleMeasurement, load_experimental_pole, centres_to_edges,
    split_continuous_segments, format_number, scaled_axes_position,
)
from .pole_widgets import PoleUi, Value, configure, choose_colour, save_plot, show_error

POLE_COLOURS = LinearSegmentedColormap.from_list(
    "white_viridis", [(1., 1., 1., 1.), *colormaps["viridis"](np.linspace(0., 1., 256))], N=256)
POLE_COLOURS.set_bad("white")
POLE_COLOURS.set_under("white")


class ExperimentalPolePage(QWidget, PoleUi):
    def set_plot_renderer(self, mode: str) -> None:
        """Switch the drawing surface without replacing measured data."""
        if mode not in {"matplotlib", "pyqtgraph"}:
            raise ValueError(mode)
        if mode == "pyqtgraph" and self.pyqtgraph_plot is None:
            from .pyqtgraph_experimental import PyQtGraphExperimentalPlot

            self.pyqtgraph_plot = PyQtGraphExperimentalPlot(self)
            self.pyqtgraph_plot.point_clicked.connect(
                self._inspect_plot_position
            )
            self.pyqtgraph_plot.zoom_changed.connect(
                self._pyqtgraph_zoom_changed
            )
            self.pyqtgraph_plot.palette_changed.connect(
                self._pyqtgraph_palette_changed
            )
            self.plot_stack.addWidget(self.pyqtgraph_plot)
        self.plot_renderer = mode
        self.plot_stack.setCurrentWidget(
            self.matplotlib_plot
            if mode == "matplotlib"
            else self.pyqtgraph_plot
        )
        if self.raw_path is not None and self.scans:
            self.draw_pole_figure()
        else:
            self._show_placeholder()

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
        if self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.show_placeholder(
                localised(
                    "Open a RAW file",
                    "Ouvrez un fichier RAW",
                    "Откройте RAW-файл",
                )
            )

    @staticmethod
    def _parse(variable: Value) -> float | None:
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
            configure(button, state="disabled" if solid else "normal")
        configure(self.fill_colour_button, state="normal" if solid else "disabled")
        if not solid and self.scale_mode.get() == "log":
            self.on_scale_mode_changed()
        else:
            self.schedule_redraw()

    def on_scale_mode_changed(self) -> None:
        if not self.scans:
            return
        low = self._parse(self.lower_limit)
        if self.scale_mode.get() == "log" and (low is None or low <= 0):
            minimum_positive = self._minimum_positive()
            if minimum_positive is None:
                show_error(
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
            configure(self.save_button, state="disabled")
            return
        if (
            self.display_mode.get() == "colour"
            and self.scale_mode.get() == "log"
            and low <= 0
        ):
            self.status_text.set(
                "Для логарифмической шкалы нижняя граница должна быть положительной."
            )
            configure(self.save_button, state="disabled")
            return
        self.schedule_redraw(delay=180)

    def _plot_title(self) -> str:
        return localised(
            f"{self.raw_path.stem}\nPole figure",
            f"{self.raw_path.stem}\nFigure de pôles",
            f"{self.raw_path.stem}\nПолюсная фигура",
        )

    def _set_success_status(
        self,
        radii: np.ndarray,
        lower: float,
        upper: float,
        display_mode: str,
        scale_title: str | None,
    ) -> None:
        if display_mode == "solid":
            mode_text = localised(
                f"solid fill from {format_number(lower)} to {format_number(upper)}",
                f"remplissage uni de {format_number(lower)} à {format_number(upper)}",
                f"заливка от {format_number(lower)} до {format_number(upper)}",
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

    def _draw_pyqtgraph_pole_figure(self) -> None:
        from .pyqtgraph_experimental import (
            intensity_colours,
            normalised_intensity,
            solid_colours,
        )

        self.redraw_job = None
        if self.raw_path is None or not self.scans:
            return
        radii = self._validated_radii()
        if radii is None:
            configure(self.save_button, state="disabled")
            return
        scans = self.scans
        if radii.size > 1 and radii[0] > radii[-1]:
            radii = radii[::-1]
            scans = list(reversed(scans))

        low = self._parse(self.lower_limit)
        high = self._parse(self.upper_limit)
        if low is None or high is None or not low < high:
            return
        plot = self.pyqtgraph_plot
        display_mode = self.display_mode.get()
        scale_mode = self.scale_mode.get()
        if display_mode == "solid":
            colour_map = solid_colours(self.fill_colour.get())
            scale_title = None
        elif scale_mode == "log":
            minimum_positive = self._minimum_positive()
            if minimum_positive is None:
                return
            low = max(low, minimum_positive)
            colour_map = intensity_colours(plot.background_colour())
            scale_title = localised(
                "logarithmic", "logarithmique", "логарифмическая"
            )
        elif scale_mode == "square":
            colour_map = intensity_colours(plot.background_colour())
            scale_title = localised("square", "quadratique", "квадратная")
        else:
            colour_map = intensity_colours(plot.background_colour())
            scale_title = localised("linear", "linéaire", "линейная")

        radial_step = abs(float(self._parse(self.angle_step) or 1.0))
        radius_edges = centres_to_edges(
            radii, single_width=max(radial_step, 1.0)
        )
        radius_edges[0] = max(0.0, radius_edges[0])
        plot.begin_frame(
            max(float(radius_edges[-1]), 1e-9),
            self._plot_title(),
        )
        cell_groups = {}
        cell_entries = []
        for row, (phi, intensity) in enumerate(scans):
            for phi_part, intensity_part in split_continuous_segments(
                phi, intensity
            ):
                phi_edges = centres_to_edges(phi_part)
                if display_mode == "solid":
                    values = np.full(intensity_part.shape, np.nan, dtype=float)
                    inside = (
                        np.isfinite(intensity_part)
                        & (intensity_part >= low)
                        & (intensity_part <= high)
                    )
                    values[inside] = 0.5
                else:
                    values = normalised_intensity(
                        intensity_part, low, high, scale_mode
                    )
                key = tuple(np.round(phi_edges, decimals=12))
                cell_groups.setdefault(key, []).append((row, phi_edges, values))
                cell_entries.append((row, phi_edges, values))

        # Bruker ranges normally share one Phi grid.  Rendering those ranges as
        # one mesh avoids hundreds of separate QGraphicsItems; NaN rows retain
        # empty Chi ranges.  Highly irregular series use the bounded fallback.
        if len(cell_groups) <= max(8, len(scans) // 4):
            for entries in cell_groups.values():
                phi_edges = entries[0][1]
                values = np.full(
                    (len(scans), phi_edges.size - 1),
                    np.nan,
                    dtype=float,
                )
                for row, _edges, row_values in entries:
                    values[row, :] = row_values
                plot.add_cells(
                    phi_edges,
                    radius_edges,
                    values,
                    colour_map,
                )
        else:
            for row, phi_edges, values in cell_entries:
                plot.add_cells(
                    phi_edges,
                    radius_edges[row : row + 2],
                    values,
                    colour_map,
                )

        if not plot.mesh_items:
            self.status_text.set("Не найдено данных, пригодных для построения.")
            configure(self.save_button, state="disabled")
            return
        self.displayed_radii = radii.copy()
        self.displayed_radius_edges = radius_edges.copy()
        self.displayed_scans = scans
        if display_mode == "colour":
            plot.set_colour_scale(
                low,
                high,
                localised(
                    f"Intensity ({scale_title})",
                    f"Intensité ({scale_title})",
                    f"Интенсивность ({scale_title})",
                ),
            )
        configure(self.save_button, state="normal")
        configure(
            self.reset_zoom_button,
            state="normal" if self.zoom_factor > 1.0 else "disabled",
        )
        self._set_success_status(radii, low, high, display_mode, scale_title)

    def draw_pole_figure(self) -> None:
        if self.plot_renderer == "pyqtgraph":
            self._draw_pyqtgraph_pole_figure()
            return
        self.redraw_job = None
        if self.raw_path is None or not self.scans:
            return

        radii = self._validated_radii()
        if radii is None:
            configure(self.save_button, state="disabled")
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
            configure(self.save_button, state="disabled")
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
        axis.set_title(self._plot_title(), pad=20)

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
        configure(self.save_button, state="normal")
        configure(self.reset_zoom_button, state="disabled")
        self._set_success_status(radii, low, high, display_mode, scale_title)

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
        configure(self.reset_zoom_button, 
            state="normal" if self.zoom_factor > 1.0 else "disabled"
        )
        self.canvas.draw_idle()

    def reset_zoom(self) -> None:
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.reset_view()
            return
        self.zoom_factor = 1.0
        if self.plot_axis is not None and self.base_plot_position is not None:
            self.plot_axis.set_position(self.base_plot_position)
            self.canvas.draw_idle()
        configure(self.reset_zoom_button, state="disabled")

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
        self._inspect_plot_position(float(theta), float(radius))

    def _pyqtgraph_zoom_changed(self, factor: float) -> None:
        self.zoom_factor = float(factor)
        configure(
            self.reset_zoom_button,
            state="normal" if self.zoom_factor > 1.0 + 1e-9 else "disabled",
        )

    def _pyqtgraph_palette_changed(self) -> None:
        if self.plot_renderer == "pyqtgraph":
            self.draw_pole_figure()

    def _inspect_plot_position(self, theta: float, radius: float) -> None:
        """Show the original measured cell at a polar plot position."""
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

    def __init__(self, parent=None, *, on_open_raw=None):
        super().__init__(parent)
        self.root = self
        self.on_open_raw = on_open_raw
        self._init_ui_helpers()
        self.raw_path = None
        self.measurement = None
        self.scans = []
        self.loaded_radii = None
        self.data_source_description = ''
        self.displayed_radii = None
        self.displayed_radius_edges = None
        self.displayed_scans = []
        self.plot_axis = None
        self.base_plot_position = None
        self.zoom_factor = 1.0
        self.plot_renderer = "matplotlib"
        self.pyqtgraph_plot = None
        self.redraw_job = None
        self.angle_update_guard = False
        self.limit_update_guard = False
        self.first_angle, self.angle_step, self.last_angle = Value(), Value(), Value()
        self.display_mode, self.scale_mode = Value('colour'), Value('linear')
        self.fill_colour = Value('#2a788e')
        self.lower_limit, self.upper_limit = Value('0'), Value('1')
        self.file_text = Value(translation_key='text.no_raw_selected')
        self.status_text = Value(translation_key='text.select_a_raw_file')
        self.cursor_text = Value(translation_key='text.click_the_figure_to_inspect_a_point')
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self.draw_pole_figure)
        self._build_interface()
        self._attach_traces()
        self._show_placeholder()

    def _build_interface(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter()
        root.addWidget(splitter)
        self.control_panel, controls = self.control_scroll()
        splitter.addWidget(self.control_panel)
        files = self.section(controls, 'text.data')
        self.button(files.content_layout, 'text.open_raw', self.select_raw)
        self.label(files.content_layout, value=self.file_text)
        files.set_expanded(True)
        angles = self.section(controls, 'text.tilt_angle_series').content_layout
        self.angle_entries = {}
        for title, value, name in [('Первый угол, °', self.first_angle, 'first'),
                                    ('Шаг, °', self.angle_step, 'step'),
                                    ('Последний угол, °', self.last_angle, 'last')]:
            self.angle_entries[name] = self.field(angles, title, value)
        self.label(angles, 'text.raw_angles_are_filled_automatically_for_xy_fallback_enter_any_two_values_the_third_is_calculated')
        display = self.section(controls, 'text.display').content_layout
        self.radios(display, [('text.colour_scale', 'colour'), ('text.solid_fill', 'solid')],
                    self.display_mode, self.on_display_mode_changed)
        self.fill_colour_button = self.button(display, 'text.fill_colour', self.choose_fill_colour)
        configure(self.fill_colour_button, state='disabled', background=self.fill_colour.get())
        scaling = self.section(controls, 'text.intensity_scale').content_layout
        self.scale_buttons = self.radios(scaling, [('Линейная', 'linear'), ('Лог.', 'log'), ('Квадрат', 'square')],
                                         self.scale_mode, self.on_scale_mode_changed)
        limits = self.section(controls, 'text.intensity_limits').content_layout
        self.lower_entry = self.field(limits, 'text.lower', self.lower_limit)
        self.upper_entry = self.field(limits, 'text.upper', self.upper_limit)
        selected = self.section(controls, 'text.selected_point').content_layout
        self.label(selected, value=self.cursor_text)
        self.label(selected, 'text.use_the_mouse_wheel_to_zoom_around_the_cursor')
        self.reset_zoom_button = self.button(selected, 'text.reset_zoom', self.reset_zoom)
        self.reset_zoom_button.setEnabled(False)
        self.save_button = self.button(controls, 'text.save_figure', self.save_figure)
        self.save_button.setEnabled(False)
        self.label(controls, value=self.status_text)
        controls.addStretch(1)
        self.figure = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(0)
        self.matplotlib_plot = self.canvas
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.matplotlib_plot)
        splitter.addWidget(self.plot_stack)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.canvas.mpl_connect('scroll_event', self.on_plot_scroll)

    def select_raw(self, selected=None, raw_data=None):
        if selected is None:
            selected, _ = QFileDialog.getOpenFileName(self, tr('text.open_raw'), '', 'RAW (*.raw);;All files (*)')
            if selected and self.on_open_raw is not None:
                self.on_open_raw(selected)
                return
        if not selected:
            return
        try:
            measurement = load_experimental_pole(selected, raw_data)
        except (OSError, ValueError) as error:
            show_error('Ошибка чтения измерения', error, self)
            return
        self.load_measurement(measurement)

    def load_measurement(self, measurement):
        self._redraw_timer.stop()
        self.measurement = measurement
        self.raw_path = measurement.source
        self.scans = measurement.scans
        self.loaded_radii = None if measurement.radii is None else measurement.radii.copy()
        self.displayed_radii = self.displayed_radius_edges = None
        self.displayed_scans = []
        self.base_plot_position = None
        self.zoom_factor = 1.0
        self.reset_zoom_button.setEnabled(False)
        if self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.reset_view()
        self.cursor_text.set_key('text.click_the_figure_to_inspect_a_point')
        self.file_text.set(self.raw_path.name)
        self._update_source_description()
        self.angle_update_guard = True
        try:
            if self.loaded_radii is None:
                for value in (self.first_angle, self.angle_step, self.last_angle):
                    value.set('')
            else:
                radii = self.loaded_radii
                self.first_angle.set(format_number(float(radii[0])))
                self.last_angle.set(format_number(float(radii[-1])))
                differences = np.diff(radii)
                self.angle_step.set('0' if len(radii) == 1 else
                                    format_number(float(differences[0])) if np.allclose(differences, differences[0], rtol=0, atol=1e-8)
                                    else tr('text.variable'))
        finally:
            self.angle_update_guard = False
        finite = np.concatenate([row[np.isfinite(row)] for _phi, row in self.scans])
        low, high = float(finite.min()), float(finite.max())
        self._configure_limits(low, low + 1.0 if math.isclose(low, high) else high)
        self.save_button.setEnabled(False)
        if self.loaded_radii is not None:
            self.draw_pole_figure()
        else:
            self._show_placeholder()
            self.status_text.set(localised(
                f'Loaded {len(self.scans)} matching XY files. Enter two tilt-angle values.',
                f'{len(self.scans)} fichiers XY correspondants chargés. Saisissez deux valeurs d’angle.',
                f'Загружено подходящих XY-файлов: {len(self.scans)}. Укажите два значения серии углов наклона.'))
            self.angle_entries['first'].setFocus()

    def _update_source_description(self):
        count = len(self.scans)
        if self.measurement is not None and self.measurement.raw is not None:
            self.data_source_description = localised(f'{count} RAW ranges', f'{count} plages RAW', f'{count} диапазонов RAW')
        else:
            self.data_source_description = localised(f'{count} fallback XY files', f'{count} fichiers XY de secours', f'{count} резервных файлов XY')

    def clear_data(self):
        self._redraw_timer.stop()
        self.redraw_job = None
        self.measurement = self.raw_path = None
        self.scans = []
        self.loaded_radii = self.displayed_radii = self.displayed_radius_edges = None
        self.displayed_scans = []
        self.file_text.set_key('text.no_raw_selected')
        self.status_text.set_key('text.select_a_raw_file')
        self.cursor_text.set_key('text.click_the_figure_to_inspect_a_point')
        self.save_button.setEnabled(False)
        self.reset_zoom_button.setEnabled(False)
        self._show_placeholder()

    def schedule_redraw(self, delay=150):
        self._redraw_timer.start(delay)

    def choose_fill_colour(self):
        colour = choose_colour(self, self.fill_colour.get(), tr('text.select_fill_colour'))
        if colour:
            self.fill_colour.set(colour)
            configure(self.fill_colour_button, background=colour)
            self.schedule_redraw()

    def save_figure(self):
        if self.raw_path is None or self.displayed_radii is None:
            return
        default_path = self.raw_path.with_name(
            self.raw_path.stem + '_pole_figure.png'
        )
        if self.plot_renderer == 'pyqtgraph' and self.pyqtgraph_plot is not None:
            path = self.pyqtgraph_plot.save_image(default_path)
        else:
            path = save_plot(self, self.figure, default_path)
        if path:
            self.status_text.set(localised(f'Figure saved: {Path(path).name}.',
                                          f'Figure enregistrée : {Path(path).name}.',
                                          f'Рисунок сохранён: {Path(path).name}.'))

    def retranslate(self):
        self.retranslate_controls()
        if self.measurement is None:
            self._show_placeholder()
            return
        if self.loaded_radii is not None and len(self.loaded_radii) > 2:
            differences = np.diff(self.loaded_radii)
            if not np.allclose(differences, differences[0], rtol=0, atol=1e-8):
                self.angle_update_guard = True
                try:
                    self.angle_step.set(tr('text.variable'))
                finally:
                    self.angle_update_guard = False
        self._update_source_description()
        if self.plot_renderer == 'pyqtgraph':
            self.draw_pole_figure()
            return
        position = self.plot_axis.get_position().bounds if self.plot_axis is not None else None
        zoom = self.zoom_factor
        self.draw_pole_figure()
        if position is not None and self.plot_axis is not None and zoom != 1:
            self.plot_axis.set_position(position)
            self.zoom_factor = zoom
            self.reset_zoom_button.setEnabled(True)
            self.canvas.draw_idle()
