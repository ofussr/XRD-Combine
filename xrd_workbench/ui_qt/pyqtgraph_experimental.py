"""Experimental pole-figure surface rendered with PyQtGraph and Qt."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .pyqtgraph_pole import ColourScale
from .pyqtgraph_interaction import handle_navigation_drag


EXPERIMENTAL_COLOUR_STOPS = (
    (0.000, "#ffffff"),
    (0.004, "#440154"),
    (0.130, "#472d7b"),
    (0.250, "#3b528b"),
    (0.380, "#2c728e"),
    (0.500, "#21918c"),
    (0.630, "#28ae80"),
    (0.750, "#5ec962"),
    (0.880, "#addc30"),
    (1.000, "#fde725"),
)


def experimental_colour_stops(under_colour: str = "#ffffff"):
    return ((0.0, under_colour), *EXPERIMENTAL_COLOUR_STOPS[1:])


def intensity_colours(under_colour: str = "#ffffff") -> pg.ColorMap:
    """Return the background-to-viridis map used by the Matplotlib renderer."""
    stops = experimental_colour_stops(under_colour)
    return pg.ColorMap(
        np.asarray([position for position, _colour in stops]),
        [QColor(colour) for _position, colour in stops],
    )


def solid_colours(colour: str) -> pg.ColorMap:
    fill = QColor(colour)
    return pg.ColorMap(np.asarray([0.0, 1.0]), [fill, fill])


def normalised_intensity(
    values: Sequence[float],
    lower: float,
    upper: float,
    scale_mode: str,
) -> np.ndarray:
    """Apply the same linear, logarithmic or square mapping as Matplotlib."""
    source = np.asarray(values, dtype=float)
    result = np.full(source.shape, np.nan, dtype=float)
    finite = np.isfinite(source)
    if scale_mode == "log":
        finite &= source > 0.0
        if lower <= 0.0 or upper <= lower:
            return result
        result[finite] = (
            np.log(source[finite]) - math.log(lower)
        ) / (math.log(upper) - math.log(lower))
    else:
        if upper <= lower:
            return result
        result[finite] = (source[finite] - lower) / (upper - lower)
        if scale_mode == "square":
            result[finite] = np.square(np.clip(result[finite], 0.0, 1.0))
    result[finite] = np.clip(result[finite], 0.0, 1.0)
    return result


class ExperimentalViewBox(pg.ViewBox):
    """View box with point inspection on left click and native wheel zoom."""

    plot_clicked = Signal(float, float)
    wheel_used = Signal()

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToView(event.pos())
            self.plot_clicked.emit(float(point.x()), float(point.y()))
            event.accept()
            return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        if handle_navigation_drag(self, event, axis):
            if event.button() == Qt.MouseButton.RightButton:
                self.wheel_used.emit()
            return
        super().mouseDragEvent(event, axis=axis)

    def wheelEvent(self, event, axis=None):
        super().wheelEvent(event, axis=axis)
        self.wheel_used.emit()


class PyQtGraphExperimentalPlot(QWidget):
    """Qt-native polar cell plot retaining the original measured grid."""

    point_clicked = Signal(float, float)
    zoom_changed = Signal(float)
    palette_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view_box = ExperimentalViewBox(enableMenu=False)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, background=None)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("bottom")
        self.plot_item.setAspectLocked(True, ratio=1.0)
        self.plot_item.setMouseEnabled(x=True, y=True)
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.colour_scale = ColourScale()
        self.colour_scale.hide()

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        body.addWidget(self.plot_widget, 1)
        body.addWidget(self.colour_scale)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addLayout(body, 1)

        self.mesh_items: list[pg.PColorMeshItem] = []
        self.cell_count = 0
        self.radius_max = 1.0
        self._home_ranges = ((-1.0, 1.0), (-1.0, 1.0))
        self._resetting = False
        self.view_box.plot_clicked.connect(self._cartesian_clicked)
        self.view_box.wheel_used.connect(
            lambda: QTimer.singleShot(0, self._report_zoom)
        )
        self.show_placeholder("")

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            QTimer.singleShot(0, self.palette_changed.emit)

    def _foreground(self) -> QColor:
        return self.palette().color(QPalette.ColorRole.Text)

    def _background(self) -> QColor:
        return self.palette().color(QPalette.ColorRole.Base)

    def background_colour(self) -> str:
        return self._background().name()

    def _clear(self) -> None:
        self.plot_item.clear()
        self.mesh_items.clear()
        self.cell_count = 0
        self.colour_scale.hide()
        self.plot_widget.setBackground(self._background())

    def show_placeholder(self, text: str) -> None:
        self._clear()
        self.title_label.clear()
        label = pg.TextItem(
            text=str(text),
            color=self._foreground(),
            anchor=(0.5, 0.5),
        )
        label.setPos(0.0, 0.0)
        self.plot_item.addItem(label)
        self.radius_max = 1.0
        self.reset_view()

    @staticmethod
    def _nice_radial_ticks(radius_max: float) -> np.ndarray:
        if radius_max <= 0.0:
            return np.empty(0, dtype=float)
        rough_step = radius_max / 5.0
        magnitude = 10.0 ** math.floor(math.log10(rough_step))
        step = next(
            candidate * magnitude
            for candidate in (1.0, 2.0, 2.5, 5.0, 10.0)
            if candidate * magnitude >= rough_step
        )
        return np.arange(step, radius_max + step * 0.25, step)

    def begin_frame(self, radius_max: float, title: str) -> None:
        previous_radius = self.radius_max
        preserve_view = bool(self.mesh_items) and math.isclose(
            previous_radius, radius_max, rel_tol=1e-12, abs_tol=1e-12
        )
        previous_ranges = self.view_box.viewRange() if preserve_view else None
        self._clear()
        self.title_label.setText(title)
        self.radius_max = max(float(radius_max), 1e-9)
        self._draw_grid()
        if previous_ranges is None:
            self.reset_view()
        else:
            self.view_box.setRange(
                xRange=previous_ranges[0],
                yRange=previous_ranges[1],
                padding=0.0,
            )
            self._report_zoom()

    def _draw_grid(self) -> None:
        foreground = self._foreground()
        grid = QColor(foreground)
        grid.setAlpha(118)
        boundary = QColor(foreground)
        theta = np.linspace(0.0, 2.0 * math.pi, 361)

        for radius in self._nice_radial_ticks(self.radius_max):
            if radius >= self.radius_max * 0.999999:
                continue
            item = pg.PlotCurveItem(
                x=radius * np.sin(theta),
                y=radius * np.cos(theta),
                pen=pg.mkPen(grid, width=0.7),
            )
            item.setZValue(3.0)
            self.plot_item.addItem(item)
            angle = math.radians(135.0)
            self._add_text(
                radius * math.sin(angle),
                radius * math.cos(angle),
                f"{radius:g}°",
                grid,
                anchor=(0.0, 0.5),
            )

        for phi in range(0, 360, 45):
            angle = math.radians(phi)
            item = pg.PlotCurveItem(
                x=[0.0, self.radius_max * math.sin(angle)],
                y=[0.0, self.radius_max * math.cos(angle)],
                pen=pg.mkPen(grid, width=0.7),
            )
            item.setZValue(3.0)
            self.plot_item.addItem(item)
            label_radius = self.radius_max * 1.035
            self._add_text(
                label_radius * math.sin(angle),
                label_radius * math.cos(angle),
                f"{phi}°",
                foreground,
                anchor=(0.5, 0.5),
            )

        outer = pg.PlotCurveItem(
            x=self.radius_max * np.sin(theta),
            y=self.radius_max * np.cos(theta),
            pen=pg.mkPen(boundary, width=1.4),
        )
        outer.setZValue(4.0)
        self.plot_item.addItem(outer)

    def _add_text(self, x, y, text, colour, *, anchor) -> None:
        item = pg.TextItem(str(text), color=colour, anchor=anchor)
        item.setPos(float(x), float(y))
        item.setZValue(4.0)
        self.plot_item.addItem(item)

    def add_cells(
        self,
        phi_edges: Sequence[float],
        radius_edges: Sequence[float],
        values: Sequence[float],
        colour_map: pg.ColorMap,
    ) -> None:
        phi_edges = np.deg2rad(np.asarray(phi_edges, dtype=float))
        radius_edges = np.asarray(radius_edges, dtype=float)
        z_values = np.asarray(values, dtype=float)
        if z_values.ndim == 1:
            z_values = z_values[np.newaxis, :]
        if z_values.ndim != 2:
            raise ValueError("Polar cell values must be one- or two-dimensional.")
        if radius_edges.size != z_values.shape[0] + 1:
            raise ValueError("Radial edges do not match the polar cell rows.")
        if phi_edges.size != z_values.shape[1] + 1:
            raise ValueError("Angular edges do not match the polar cell columns.")
        x = radius_edges[:, np.newaxis] * np.sin(phi_edges)[np.newaxis, :]
        y = radius_edges[:, np.newaxis] * np.cos(phi_edges)[np.newaxis, :]
        mesh = pg.PColorMeshItem(
            x,
            y,
            z_values,
            colorMap=colour_map,
            levels=(0.0, 1.0),
            enableAutoLevels=False,
        )
        mesh.setZValue(1.0)
        self.plot_item.addItem(mesh)
        self.mesh_items.append(mesh)
        self.cell_count += int(np.count_nonzero(np.isfinite(z_values)))

    def set_colour_scale(
        self,
        lower: float,
        upper: float,
        label: str,
    ) -> None:
        self.colour_scale.set_scale(
            experimental_colour_stops(self.background_colour()),
            lower,
            upper,
            label,
        )
        self.colour_scale.show()

    def reset_view(self) -> None:
        half_span = self.radius_max * 1.13
        self._resetting = True
        self.view_box.setRange(
            xRange=(-half_span, half_span),
            yRange=(-half_span, half_span),
            padding=0.0,
        )
        ranges = self.view_box.viewRange()
        self._home_ranges = (
            (float(ranges[0][0]), float(ranges[0][1])),
            (float(ranges[1][0]), float(ranges[1][1])),
        )
        self._resetting = False
        self.zoom_changed.emit(1.0)

    def _report_zoom(self) -> None:
        if self._resetting:
            return
        ranges = self.view_box.viewRange()
        home_width = self._home_ranges[0][1] - self._home_ranges[0][0]
        width = float(ranges[0][1] - ranges[0][0])
        factor = max(1.0, home_width / max(width, 1e-12))
        home_center = (
            0.5 * sum(self._home_ranges[0]),
            0.5 * sum(self._home_ranges[1]),
        )
        center = (
            0.5 * sum(ranges[0]),
            0.5 * sum(ranges[1]),
        )
        if not np.allclose(center, home_center, rtol=0.0, atol=1e-9):
            factor = max(factor, 1.000001)
        self.zoom_changed.emit(factor)

    def _cartesian_clicked(self, x: float, y: float) -> None:
        radius = math.hypot(x, y)
        phi = math.atan2(x, y) % (2.0 * math.pi)
        self.point_clicked.emit(phi, radius)

    def save_image(self, default_path: Path) -> str | None:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "PNG",
            str(default_path),
            "PNG (*.png)",
        )
        if not path:
            return None
        if not Path(path).suffix:
            path += ".png"
        try:
            if not self.grab().save(path, "PNG"):
                raise OSError(path)
        except OSError as error:
            QMessageBox.critical(self, "PNG", str(error))
            return None
        return path
