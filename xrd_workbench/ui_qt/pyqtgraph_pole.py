"""Experimental PyQtGraph surface for calculated pole figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QLinearGradient, QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr
from .pyqtgraph_interaction import handle_navigation_drag


VIRIDIS_STOPS = (
    (0.00, "#440154"),
    (0.13, "#472d7b"),
    (0.25, "#3b528b"),
    (0.38, "#2c728e"),
    (0.50, "#21918c"),
    (0.63, "#28ae80"),
    (0.75, "#5ec962"),
    (0.88, "#addc30"),
    (1.00, "#fde725"),
)

D_SPACING_STOPS = (
    (0.00, "#0000ff"),
    (0.20, "#0075ff"),
    (0.40, "#00d9ff"),
    (0.55, "#00f06a"),
    (0.72, "#76ff00"),
    (1.00, "#ff0000"),
)


class PoleViewBox(pg.ViewBox):
    """View box that reserves the left mouse button for crystal rotation."""

    drag_started = Signal(float, float)
    drag_moved = Signal(float, float)
    drag_finished = Signal(float, float)
    plot_clicked = Signal(float, float)

    def _view_position(self, event):
        point = self.mapToView(event.pos())
        return float(point.x()), float(point.y())

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = self._view_position(event)
            if event.isStart():
                self.drag_started.emit(x, y)
            elif event.isFinish():
                self.drag_finished.emit(x, y)
            else:
                self.drag_moved.emit(x, y)
            event.accept()
            return
        if handle_navigation_drag(self, event, axis):
            return
        super().mouseDragEvent(event, axis=axis)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = self._view_position(event)
            self.plot_clicked.emit(x, y)
            event.accept()
            return
        super().mouseClickEvent(event)


class ColourScale(QWidget):
    """Small Qt-painted colour scale independent of Matplotlib."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(116)
        self.setMaximumWidth(160)
        self._stops = D_SPACING_STOPS
        self._minimum = 0.0
        self._maximum = 1.0
        self._label = ""

    def set_scale(self, stops, minimum: float, maximum: float, label: str) -> None:
        self._stops = tuple(stops)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._label = str(label)
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        text = self.palette().color(QPalette.ColorRole.Text)
        painter.setPen(text)
        margin = 8
        top = 58
        bottom = max(top + 30, self.height() - 30)
        bar_width = 18
        gradient = QLinearGradient(0, bottom, 0, top)
        for position, colour in self._stops:
            gradient.setColorAt(float(position), QColor(colour))
        painter.fillRect(margin, top, bar_width, bottom - top, gradient)
        painter.drawRect(margin, top, bar_width, bottom - top)
        painter.drawText(margin, 4, self.width() - 2 * margin, 48,
                         Qt.AlignmentFlag.AlignLeft
                         | Qt.AlignmentFlag.AlignVCenter
                         | Qt.TextFlag.TextWordWrap,
                         self._label)
        painter.drawText(margin + bar_width + 6, top - 8, self.width() - 34, 18,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"{self._maximum:g}")
        painter.drawText(margin + bar_width + 6, bottom - 9, self.width() - 34, 18,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"{self._minimum:g}")


def colour_values(
    values: Sequence[float],
    stops: Sequence[tuple[float, str]],
    minimum: float,
    maximum: float,
    opacity: float,
) -> list[QColor]:
    """Map numerical values to reusable QColor objects."""
    positions = np.asarray([item[0] for item in stops], dtype=float)
    channels = np.asarray(
        [
            (QColor(colour).red(), QColor(colour).green(), QColor(colour).blue())
            for _position, colour in stops
        ],
        dtype=float,
    )
    span = maximum - minimum
    normalized = np.zeros(len(values), dtype=float) if abs(span) < 1e-15 else (
        np.clip((np.asarray(values, dtype=float) - minimum) / span, 0.0, 1.0)
    )
    result = []
    alpha = round(255 * min(1.0, max(0.0, opacity)))
    for value in normalized:
        red, green, blue = (
            round(float(np.interp(value, positions, channels[:, index])))
            for index in range(3)
        )
        result.append(QColor(red, green, blue, alpha))
    return result


class PyQtGraphPolePlot(QWidget):
    """Qt-native interactive surface used by the test renderer."""

    drag_started = Signal(float, float)
    drag_moved = Signal(float, float)
    drag_finished = Signal(float, float)
    plot_clicked = Signal(float, float)
    palette_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view_box = PoleViewBox(enableMenu=False)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, background=None)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("bottom")
        self.plot_item.setAspectLocked(True, ratio=1.0)
        self.plot_item.setMouseEnabled(x=True, y=True)
        self._legend = None
        self._labels = []

        self.colour_scale = ColourScale()
        self.colour_scale.hide()
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        body.addWidget(self.plot_widget, 1)
        body.addWidget(self.colour_scale)

        self.home_button = QPushButton()
        self.home_button.clicked.connect(self.reset_view)
        self.save_button = QPushButton()
        self.save_button.clicked.connect(self.save_image)
        controls = QHBoxLayout()
        controls.setContentsMargins(2, 2, 2, 2)
        controls.addWidget(self.home_button)
        controls.addWidget(self.save_button)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addLayout(body, 1)
        layout.addLayout(controls)

        self.view_box.drag_started.connect(self.drag_started.emit)
        self.view_box.drag_moved.connect(self.drag_moved.emit)
        self.view_box.drag_finished.connect(self.drag_finished.emit)
        self.view_box.plot_clicked.connect(self.plot_clicked.emit)
        self.retranslate()
        self.reset_view()
        QTimer.singleShot(0, self.reset_view)

    def retranslate(self) -> None:
        self.home_button.setText(tr("qt.plot_reset_view"))
        self.save_button.setText(tr("text.save_figure"))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            QTimer.singleShot(0, self.palette_changed.emit)

    def reset_view(self) -> None:
        viewport = self.plot_widget.viewport()
        width = max(1.0, float(viewport.width()))
        height = max(1.0, float(viewport.height()))
        if width >= height:
            x_half = 1.10 * width / height
            y_half = 1.10
        else:
            x_half = 1.10
            y_half = 1.10 * height / width
        self.view_box.setRange(
            xRange=(-x_half, x_half),
            yRange=(-y_half, y_half),
            padding=0.0,
        )

    def save_image(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            tr("text.save_figure"),
            "pole_figure.png",
            "PNG (*.png)",
        )
        if not path:
            return
        try:
            if not Path(path).suffix:
                path += ".png"
            if not self.grab().save(path, "PNG"):
                raise OSError(path)
        except OSError as error:
            QMessageBox.critical(self, tr("text.save_error"), str(error))

    def begin_frame(self, *, projection: str, angle_labels: bool, title: str) -> None:
        if self._legend is not None:
            scene = self._legend.scene()
            if scene is not None:
                scene.removeItem(self._legend)
            self._legend = None
        self.plot_item.clear()
        self._labels.clear()
        self.colour_scale.hide()
        base = self.palette().color(QPalette.ColorRole.Base)
        foreground = self.palette().color(QPalette.ColorRole.Text)
        self.plot_widget.setBackground(base)
        self.plot_item.setTitle("")
        self.title_label.setText(title)

        theta = np.linspace(0.0, 2.0 * math.pi, 361)
        self.plot_item.plot(
            np.cos(theta), np.sin(theta),
            pen=pg.mkPen(foreground, width=1.5),
        )
        projection_code = str(projection)
        for chi in (15, 30, 45, 60, 75):
            chi_rad = math.radians(chi)
            radius = (
                math.sqrt(2.0) * math.sin(chi_rad / 2.0)
                if projection_code == "equal_area"
                else math.tan(chi_rad / 2.0)
            )
            grid_colour = QColor(foreground)
            grid_colour.setAlpha(72)
            self.plot_item.plot(
                radius * np.cos(theta),
                radius * np.sin(theta),
                pen=pg.mkPen(grid_colour, width=0.7),
            )
            if angle_labels:
                self.add_label((radius + 0.012, 0.012), f"{chi}°", grid_colour)
        radial_colour = QColor(foreground)
        radial_colour.setAlpha(58)
        for phi in range(0, 360, 10):
            angle = math.radians(phi)
            self.plot_item.plot(
                [0.0, math.cos(angle)],
                [0.0, math.sin(angle)],
                pen=pg.mkPen(radial_colour, width=0.6),
            )
        self.add_label((1.045, 0.0), "X", foreground, anchor=(0.5, 0.5))
        self.add_label((0.0, 1.055), "Y", foreground, anchor=(0.5, 0.5))

    def _scatter_clicked(self, _item, points, _event) -> None:
        """Forward clicks captured by a marker to the shared pole selector."""
        if not points:
            return
        position = points[0].pos()
        self.plot_clicked.emit(float(position.x()), float(position.y()))

    def add_scatter(
        self,
        positions: Sequence[tuple[float, float]],
        sizes: Sequence[float],
        *,
        colour: str | None = None,
        colours: Sequence[QColor] | None = None,
        opacity: float = 1.0,
        outlined: bool = True,
        z: float = 3.0,
    ):
        if not positions:
            return None
        x = np.asarray([item[0] for item in positions], dtype=float)
        y = np.asarray([item[1] for item in positions], dtype=float)
        diameters = np.sqrt(np.asarray(sizes, dtype=float))
        if colours is None:
            fill = QColor(colour or "#2d6da3")
            fill.setAlpha(round(255 * min(1.0, max(0.0, opacity))))
            brushes = pg.mkBrush(fill)
        else:
            brushes = [pg.mkBrush(item) for item in colours]
        pen = pg.mkPen("#ffffff", width=0.8) if outlined else None
        item = pg.ScatterPlotItem(
            x=x,
            y=y,
            size=diameters,
            pen=pen,
            brush=brushes,
            pxMode=True,
        )
        item.setZValue(z)
        item.sigClicked.connect(self._scatter_clicked)
        self.plot_item.addItem(item)
        return item

    def add_rings(
        self,
        positions: Sequence[tuple[float, float]],
        sizes: Sequence[float],
        *,
        colour: str,
        width: float,
        extra: float,
        z: float,
    ) -> None:
        if not positions:
            return
        item = pg.ScatterPlotItem(
            x=[point[0] for point in positions],
            y=[point[1] for point in positions],
            size=np.sqrt(np.asarray(sizes, dtype=float)) + extra,
            pen=pg.mkPen(colour, width=width),
            brush=None,
            pxMode=True,
        )
        item.setZValue(z)
        item.sigClicked.connect(self._scatter_clicked)
        self.plot_item.addItem(item)

    def add_label(self, position, text: str, colour, *, anchor=(0.0, 1.0)) -> None:
        item = pg.TextItem(text=str(text), color=colour, anchor=anchor)
        item.setPos(float(position[0]), float(position[1]))
        item.setZValue(5.0)
        self.plot_item.addItem(item)
        self._labels.append(item)

    def add_leader(self, start, end, colour) -> None:
        item = pg.PlotCurveItem(
            x=[float(start[0]), float(end[0])],
            y=[float(start[1]), float(end[1])],
            pen=pg.mkPen(colour, width=0.55),
        )
        item.setZValue(4.5)
        self.plot_item.addItem(item)

    def set_colour_scale(self, stops, minimum: float, maximum: float, label: str) -> None:
        self.colour_scale.set_scale(stops, minimum, maximum, label)
        self.colour_scale.show()

    def set_legend(self, entries: Sequence[tuple[str, str, float]]) -> None:
        if not entries:
            return
        legend = pg.LegendItem(offset=(-8, 8))
        legend.setParentItem(self.plot_item.graphicsItem())
        for label, colour, opacity in entries:
            fill = QColor(colour)
            fill.setAlpha(round(255 * min(1.0, max(0.0, opacity))))
            sample = pg.ScatterPlotItem(
                x=[0.0], y=[0.0], size=9,
                pen=pg.mkPen("#ffffff", width=0.8),
                brush=pg.mkBrush(fill),
            )
            legend.addItem(sample, label)
        self._legend = legend

    def pixel_size(self) -> tuple[float, float]:
        x_size, y_size = self.view_box.viewPixelSize()
        return abs(float(x_size)), abs(float(y_size))

    def data_to_scene(self, positions) -> np.ndarray:
        mapped = []
        for x, y in positions:
            point = self.view_box.mapViewToScene(QPointF(float(x), float(y)))
            mapped.append((float(point.x()), float(point.y())))
        return np.asarray(mapped, dtype=float)

    def scene_to_data(self, position) -> tuple[float, float]:
        point = self.view_box.mapSceneToView(
            QPointF(float(position[0]), float(position[1]))
        )
        return float(point.x()), float(point.y())

    def scene_bounds(self) -> tuple[float, float, float, float]:
        bounds = self.view_box.sceneBoundingRect()
        return (
            float(bounds.left()),
            float(bounds.top()),
            float(bounds.right()),
            float(bounds.bottom()),
        )
