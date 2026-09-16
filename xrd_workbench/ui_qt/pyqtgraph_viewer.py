"""PyQtGraph display adapter for the main one-dimensional viewer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr
from .pyqtgraph_interaction import handle_navigation_drag


class ViewerAxisItem(pg.AxisItem):
    """Axis with independently switchable, scale-aware tick levels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_mode = "linear"
        self.major_ticks = True
        self.minor_ticks = False
        self._log_label_step = 1
        self.enableAutoSIPrefix(False)
        self.setStyle(maxTickLevel=1, maxTextLevel=0)

    def configure(
        self,
        *,
        scale_mode: str | None = None,
        major_ticks: bool | None = None,
        minor_ticks: bool | None = None,
    ) -> None:
        if scale_mode is not None:
            self.scale_mode = str(scale_mode)
        if major_ticks is not None:
            self.major_ticks = bool(major_ticks)
        if minor_ticks is not None:
            self.minor_ticks = bool(minor_ticks)
        self.picture = None
        self.update()

    def _physical_value(self, position: float) -> float:
        value = float(position)
        if self.scale_mode == "log":
            return 10.0 ** max(-307.0, min(307.0, value))
        if self.scale_mode == "sqrt":
            return max(0.0, value) ** 2
        if self.scale_mode == "square":
            return math.sqrt(max(0.0, value))
        return value

    def _axis_position(self, value: float) -> float:
        value = float(value)
        if self.scale_mode == "sqrt":
            return math.sqrt(max(0.0, value))
        if self.scale_mode == "square":
            return max(0.0, value) ** 2
        return value

    def _log_tick_values(self, minVal: float, maxVal: float, size: float):
        low, high = sorted((float(minVal), float(maxVal)))
        span = high - low
        if span <= 0.0:
            return []
        if span < 1.0:
            self._log_label_step = 1
            physical_low = self._physical_value(low)
            physical_high = self._physical_value(high)
            previous_log_mode = self.logMode
            self.logMode = False
            try:
                levels = super().tickValues(physical_low, physical_high, size)
            finally:
                self.logMode = previous_log_mode
            return [
                (
                    spacing,
                    [math.log10(value) for value in values if value > 0.0],
                )
                for spacing, values in levels
            ]

        first_exponent = math.ceil(low)
        last_exponent = math.floor(high)
        major = [
            float(exponent)
            for exponent in range(first_exponent, last_exponent + 1)
        ]
        pixels_per_decade = size / span
        self._log_label_step = max(
            1,
            int(math.ceil(45.0 / max(pixels_per_decade, 1e-12))),
        )
        if pixels_per_decade >= 70.0:
            multipliers = range(2, 10)
        elif pixels_per_decade >= 25.0:
            multipliers = (2, 5)
        else:
            multipliers = ()
        minor = []
        for decade in range(math.floor(low), math.ceil(high)):
            for multiplier in multipliers:
                position = decade + math.log10(multiplier)
                if low < position < high:
                    minor.append(position)
        return [(1.0, major), (math.log10(2.0), sorted(set(minor)))]

    def tickValues(self, minVal, maxVal, size):
        if self.scale_mode == "log":
            levels = self._log_tick_values(minVal, maxVal, size)
        elif self.scale_mode in {"sqrt", "square"}:
            physical_limits = sorted(
                (self._physical_value(minVal), self._physical_value(maxVal))
            )
            levels = super().tickValues(*physical_limits, size)
            levels = [
                (spacing, [self._axis_position(value) for value in values])
                for spacing, values in levels
            ]
        else:
            levels = super().tickValues(minVal, maxVal, size)
        if not levels:
            return []
        selected = [
            (levels[0][0], levels[0][1] if self.major_ticks else [])
        ]
        if self.minor_ticks and len(levels) > 1:
            selected.append(levels[1])
        return selected

    @staticmethod
    def _format_physical(value: float, spacing: float | None) -> str:
        if not math.isfinite(value):
            return ""
        magnitude = abs(value)
        if magnitude and (magnitude < 0.001 or magnitude >= 10000.0):
            return f"{value:.5g}"
        if spacing is None or not math.isfinite(spacing) or spacing <= 0.0:
            return f"{value:.6g}"
        places = max(0, min(8, math.ceil(-math.log10(spacing))))
        return f"{value:.{places}f}"

    def tickStrings(self, values, scale, spacing):
        if self.scale_mode == "linear":
            return super().tickStrings(values, scale, spacing)
        physical = [self._physical_value(value) * scale for value in values]
        physical_spacing = None if self.scale_mode == "log" else spacing
        labels = []
        for position, value in zip(values, physical):
            if self.scale_mode == "log" and self._log_label_step > 1:
                exponent = round(float(position))
                if math.isclose(float(position), exponent, abs_tol=1e-10):
                    if exponent % self._log_label_step:
                        labels.append("")
                        continue
            labels.append(self._format_physical(value, physical_spacing))
        return labels


# Kept as a public alias for code written against 3.0.0a16.1.
SparseAxisItem = ViewerAxisItem


def _toolbar_icon(kind: str, colour: QColor) -> QIcon:
    """Draw a small palette-aware icon without depending on Matplotlib assets."""

    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(colour)
    pen.setWidthF(1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if kind == "home":
        painter.drawLine(QPointF(4.0, 11.0), QPointF(12.0, 4.0))
        painter.drawLine(QPointF(12.0, 4.0), QPointF(20.0, 11.0))
        painter.drawRect(QRectF(6.0, 10.0, 12.0, 10.0))
    elif kind == "zoom":
        painter.drawRect(QRectF(3.5, 3.5, 11.0, 11.0))
        painter.drawEllipse(QPointF(14.0, 14.0), 4.5, 4.5)
        painter.drawLine(QPointF(17.2, 17.2), QPointF(21.0, 21.0))
    elif kind == "settings":
        for y, knob in ((6.0, 9.0), (12.0, 16.0), (18.0, 11.0)):
            painter.drawLine(QPointF(3.0, y), QPointF(21.0, y))
            painter.setBrush(colour)
            painter.drawEllipse(QPointF(knob, y), 2.0, 2.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "save":
        painter.drawRect(QRectF(4.0, 3.0, 16.0, 18.0))
        painter.drawRect(QRectF(7.0, 4.0, 9.0, 6.0))
        painter.drawRect(QRectF(7.0, 14.0, 10.0, 7.0))
    painter.end()
    return QIcon(pixmap)


class ViewerPlotSettingsDialog(QDialog):
    """Small PyQtGraph counterpart of Matplotlib's plot-options dialog."""

    def __init__(self, plot, parent=None):
        super().__init__(parent or plot)
        self.plot = plot
        self.setModal(True)
        self.setWindowTitle(tr("qt.plot_settings"))
        x_limits, y_limits = plot.physical_scan_limits()

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.x_minimum = QLineEdit(f"{x_limits[0]:.12g}")
        self.x_maximum = QLineEdit(f"{x_limits[1]:.12g}")
        self.y_minimum = QLineEdit(f"{y_limits[0]:.12g}")
        self.y_maximum = QLineEdit(f"{y_limits[1]:.12g}")
        self.x_label = QLineEdit(plot.custom_x_label)
        self.y_label = QLineEdit(plot.custom_y_label)
        self.x_label.setPlaceholderText(tr("qt.plot_axis_label_automatic"))
        self.y_label.setPlaceholderText(tr("qt.plot_axis_label_automatic"))
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.25, 8.0)
        self.line_width.setDecimals(2)
        self.line_width.setSingleStep(0.25)
        self.line_width.setValue(plot.line_width)
        self.grid_enabled = QCheckBox()
        self.grid_enabled.setChecked(plot.grid_enabled)
        self.legend_enabled = QCheckBox()
        self.legend_enabled.setChecked(plot.legend_enabled)

        x_ticks = QWidget()
        x_tick_layout = QHBoxLayout(x_ticks)
        x_tick_layout.setContentsMargins(0, 0, 0, 0)
        self.x_major_ticks = QCheckBox(tr("qt.plot_major_ticks"))
        self.x_minor_ticks = QCheckBox(tr("qt.plot_minor_ticks"))
        self.x_major_ticks.setChecked(plot.x_major_ticks)
        self.x_minor_ticks.setChecked(plot.x_minor_ticks)
        x_tick_layout.addWidget(self.x_major_ticks)
        x_tick_layout.addWidget(self.x_minor_ticks)
        x_tick_layout.addStretch(1)

        y_ticks = QWidget()
        y_tick_layout = QHBoxLayout(y_ticks)
        y_tick_layout.setContentsMargins(0, 0, 0, 0)
        self.y_major_ticks = QCheckBox(tr("qt.plot_major_ticks"))
        self.y_minor_ticks = QCheckBox(tr("qt.plot_minor_ticks"))
        self.y_major_ticks.setChecked(plot.y_major_ticks)
        self.y_minor_ticks.setChecked(plot.y_minor_ticks)
        y_tick_layout.addWidget(self.y_major_ticks)
        y_tick_layout.addWidget(self.y_minor_ticks)
        y_tick_layout.addStretch(1)

        self.grid_opacity = QSlider(Qt.Orientation.Horizontal)
        self.grid_opacity.setRange(0, 40)
        self.grid_opacity.setValue(round(plot.grid_alpha * 100.0))
        self.grid_opacity_value = QLabel()
        self.grid_opacity.valueChanged.connect(
            lambda value: self.grid_opacity_value.setText(f"{value}%")
        )
        self.grid_opacity.valueChanged.emit(self.grid_opacity.value())
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.grid_opacity, 1)
        opacity_row.addWidget(self.grid_opacity_value)
        form.addRow(tr("qt.plot_x_minimum"), self.x_minimum)
        form.addRow(tr("qt.plot_x_maximum"), self.x_maximum)
        form.addRow(tr("qt.plot_y_minimum"), self.y_minimum)
        form.addRow(tr("qt.plot_y_maximum"), self.y_maximum)
        form.addRow(tr("qt.plot_x_label"), self.x_label)
        form.addRow(tr("qt.plot_y_label"), self.y_label)
        form.addRow(tr("qt.plot_x_ticks"), x_ticks)
        form.addRow(tr("qt.plot_y_ticks"), y_ticks)
        form.addRow(tr("qt.plot_line_width"), self.line_width)
        form.addRow(tr("qt.plot_legend"), self.legend_enabled)
        form.addRow(tr("qt.plot_grid"), self.grid_enabled)
        form.addRow(tr("qt.plot_grid_opacity"), opacity_row)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("text.ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            tr("text.cancel")
        )
        buttons.accepted.connect(self.apply_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _number(edit: QLineEdit) -> float:
        return float(edit.text().strip().replace(",", "."))

    def apply_and_accept(self) -> None:
        try:
            x_limits = (self._number(self.x_minimum), self._number(self.x_maximum))
            y_limits = (self._number(self.y_minimum), self._number(self.y_maximum))
            if (
                not all(np.isfinite((*x_limits, *y_limits)))
                or x_limits[0] >= x_limits[1]
                or y_limits[0] >= y_limits[1]
                or (self.plot.scale_mode == "log" and y_limits[0] <= 0.0)
                or (
                    self.plot.scale_mode in {"sqrt", "square"}
                    and y_limits[0] < 0.0
                )
            ):
                raise ValueError
        except (TypeError, ValueError, ArithmeticError):
            QMessageBox.warning(
                self,
                tr("qt.plot_settings"),
                tr("qt.plot_settings_invalid"),
            )
            return
        self.plot.apply_settings(
            line_width=self.line_width.value(),
            grid_enabled=self.grid_enabled.isChecked(),
            grid_alpha=self.grid_opacity.value() / 100.0,
            x_limits=x_limits,
            y_limits=y_limits,
            custom_x_label=self.x_label.text().strip(),
            custom_y_label=self.y_label.text().strip(),
            legend_enabled=self.legend_enabled.isChecked(),
            x_major_ticks=self.x_major_ticks.isChecked(),
            x_minor_ticks=self.x_minor_ticks.isChecked(),
            y_major_ticks=self.y_major_ticks.isChecked(),
            y_minor_ticks=self.y_minor_ticks.isChecked(),
        )
        self.accept()


class ViewerViewBox(pg.ViewBox):
    """View box with selection on the left and panning on the right."""

    point_clicked = Signal(float, float)
    selection_finished = Signal(float, float, float, float)
    zoom_finished = Signal(float, float, float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_enabled = False
        self.zoom_enabled = False
        self._selection_item: QGraphicsRectItem | None = None

    def set_selection_enabled(self, enabled: bool) -> None:
        self.selection_enabled = bool(enabled)
        if enabled:
            self.zoom_enabled = False
        if not enabled:
            self._remove_selection_item()

    def set_zoom_enabled(self, enabled: bool) -> None:
        self.zoom_enabled = bool(enabled)
        if enabled:
            self.selection_enabled = False
        if not enabled:
            self._remove_selection_item()

    def _remove_selection_item(self) -> None:
        if self._selection_item is not None:
            self.removeItem(self._selection_item)
            self._selection_item = None

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.selection_enabled and not self.zoom_enabled:
                point = self.mapToView(event.pos())
                self.point_clicked.emit(float(point.x()), float(point.y()))
            event.accept()
            return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.selection_enabled and not self.zoom_enabled:
                event.accept()
                return
            start = self.mapToView(
                event.buttonDownPos(Qt.MouseButton.LeftButton)
            )
            current = self.mapToView(event.pos())
            if event.isStart() or self._selection_item is None:
                self._remove_selection_item()
                item = QGraphicsRectItem()
                colour = "#d32f2f" if self.selection_enabled else "#1976d2"
                pen = QPen(QColor(colour))
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidthF(1.4)
                pen.setCosmetic(True)
                item.setPen(pen)
                fill = QColor(colour)
                fill.setAlpha(32)
                item.setBrush(fill)
                self.addItem(item, ignoreBounds=True)
                self._selection_item = item
            if self._selection_item is not None:
                self._selection_item.setRect(QRectF(start, current).normalized())
            if event.isFinish():
                self._remove_selection_item()
                signal = (
                    self.selection_finished
                    if self.selection_enabled
                    else self.zoom_finished
                )
                self.selection_enabled = False
                self.zoom_enabled = False
                signal.emit(
                    float(start.x()),
                    float(start.y()),
                    float(current.x()),
                    float(current.y()),
                )
            event.accept()
            return
        if handle_navigation_drag(self, event, axis):
            return
        super().mouseDragEvent(event, axis=axis)

    def wheelEvent(self, event, axis=None):
        del axis
        event.accept()


class PyQtGraphViewerPlot(QWidget):
    """Qt-native plot surface for measurements and calculated phases."""

    plot_clicked = Signal(str, float, float)
    peak_region_selected = Signal(float, float, float, float)
    range_changed = Signal(str)
    reset_requested = Signal()
    settings_changed = Signal()
    palette_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.save_filename = "viewer.png"
        self.line_width = 2.0
        self.grid_enabled = True
        self.grid_alpha = 0.10
        self.legend_enabled = True
        self.x_major_ticks = True
        self.x_minor_ticks = False
        self.y_major_ticks = True
        self.y_minor_ticks = False
        self.custom_x_label = ""
        self.custom_y_label = ""
        self._automatic_x_label = ""
        self._automatic_y_label = ""
        self.scale_mode = "linear"
        self.scan_view_box = ViewerViewBox(enableMenu=False)
        self.phase_view_box = ViewerViewBox(enableMenu=False)
        scan_axes = {
            name: ViewerAxisItem(orientation=name)
            for name in ("left", "bottom", "right", "top")
        }
        phase_axes = {
            name: ViewerAxisItem(orientation=name)
            for name in ("left", "bottom", "right", "top")
        }
        self.scan_plot_widget = pg.PlotWidget(
            viewBox=self.scan_view_box,
            background=None,
            axisItems=scan_axes,
        )
        self.phase_plot_widget = pg.PlotWidget(
            viewBox=self.phase_view_box,
            background=None,
            axisItems=phase_axes,
        )
        self.scan_plot_item = self.scan_plot_widget.getPlotItem()
        self.phase_plot_item = self.phase_plot_widget.getPlotItem()
        self.scan_plot_item.getAxis("left").setWidth(66)
        self.scan_plot_item.setMouseEnabled(x=True, y=True)
        self.phase_plot_item.setMouseEnabled(x=True, y=True)
        for item in (self.scan_plot_item, self.phase_plot_item):
            item.showAxis("top")
            item.showAxis("right")
            top_axis = item.getAxis("top")
            right_axis = item.getAxis("right")
            top_axis.setStyle(showValues=False, tickLength=0)
            right_axis.setStyle(showValues=False, tickLength=0)
            top_axis.setHeight(7)
            right_axis.setWidth(7)
            top_axis.configure(major_ticks=False, minor_ticks=False)
            right_axis.configure(major_ticks=False, minor_ticks=False)
        self.phase_plot_item.showAxis("left")
        phase_left_axis = self.phase_plot_item.getAxis("left")
        phase_left_axis.setStyle(
            showValues=False,
            tickLength=0,
        )
        phase_left_axis.setWidth(7)
        phase_left_axis.configure(major_ticks=False, minor_ticks=False)
        self._apply_ticks()
        self._apply_grid()

        self.overlay_view_box = pg.ViewBox(enableMenu=False)
        self.overlay_view_box.setMouseEnabled(x=False, y=False)
        self.overlay_view_box.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.overlay_view_box.setYRange(0.0, 1.0, padding=0.0)
        self.scan_plot_widget.scene().addItem(self.overlay_view_box)
        self.overlay_view_box.setXLink(self.scan_view_box)
        self.scan_view_box.sigResized.connect(self._sync_overlay_geometry)
        self.scan_view_box.sigXRangeChanged.connect(
            lambda *_args: self._position_overlay_labels()
        )

        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.separator.hide()
        self.phase_plot_widget.hide()

        self.graphs = QWidget()
        self.graph_layout = QVBoxLayout(self.graphs)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setSpacing(2)
        self.graph_layout.addWidget(self.scan_plot_widget, 75)
        self.graph_layout.addWidget(self.separator)
        self.graph_layout.addWidget(self.phase_plot_widget, 25)

        self.home_button = QToolButton()
        self.home_button.clicked.connect(self._reset_clicked)
        self.zoom_button = QToolButton()
        self.zoom_button.setCheckable(True)
        self.zoom_button.toggled.connect(self.set_zoom_enabled)
        self.settings_button = QToolButton()
        self.settings_button.clicked.connect(self.show_settings)
        self.save_button = QToolButton()
        self.save_button.clicked.connect(self.save_image)
        for button in (
            self.home_button,
            self.zoom_button,
            self.settings_button,
            self.save_button,
        ):
            button.setAutoRaise(True)
            button.setIconSize(QSize(22, 22))
            button.setFixedSize(30, 30)
        controls = QHBoxLayout()
        controls.setContentsMargins(2, 0, 2, 0)
        controls.setSpacing(2)
        controls.addWidget(self.home_button)
        controls.addWidget(self.zoom_button)
        controls.addWidget(self.settings_button)
        controls.addWidget(self.save_button)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.graphs, 1)
        layout.addLayout(controls)

        self._legend = None
        self._logarithmic = False
        self._separate = False
        self._axes_linked = False
        self._overlay_items: list[object] = []
        self._overlay_labels: list[tuple[object, float]] = []
        self._temporary_scan_items: list[object] = []

        self.scan_view_box.point_clicked.connect(
            lambda x, y: self.plot_clicked.emit("scan", x, y)
        )
        self.phase_view_box.point_clicked.connect(
            lambda x, y: self.plot_clicked.emit("phase", x, y)
        )
        self.scan_view_box.selection_finished.connect(
            self.peak_region_selected.emit
        )
        self.scan_view_box.zoom_finished.connect(
            lambda *bounds: self._zoom_region("scan", *bounds)
        )
        self.phase_view_box.zoom_finished.connect(
            lambda *bounds: self._zoom_region("phase", *bounds)
        )
        self.scan_view_box.sigRangeChanged.connect(
            lambda *_args: self.range_changed.emit("scan")
        )
        self.phase_view_box.sigRangeChanged.connect(
            lambda *_args: self.range_changed.emit("phase")
        )
        self.retranslate()
        self._apply_palette()
        QTimer.singleShot(0, self._sync_overlay_geometry)

    def retranslate(self) -> None:
        labels = (
            (self.home_button, "qt.plot_reset_view"),
            (self.zoom_button, "qt.plot_zoom_rectangle"),
            (self.settings_button, "qt.plot_settings"),
            (self.save_button, "text.save_figure"),
        )
        for button, key in labels:
            label = tr(key)
            button.setToolTip(label)
            button.setAccessibleName(label)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._apply_palette()
            QTimer.singleShot(0, self.palette_changed.emit)

    def _apply_palette(self) -> None:
        background = self.palette().color(QPalette.ColorRole.Base)
        foreground = self.palette().color(QPalette.ColorRole.Text)
        for widget in (self.scan_plot_widget, self.phase_plot_widget):
            widget.setBackground(background)
        for item in (self.scan_plot_item, self.phase_plot_item):
            for name in ("left", "bottom", "right", "top"):
                axis = item.getAxis(name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)
        icons = (
            (self.home_button, "home"),
            (self.zoom_button, "zoom"),
            (self.settings_button, "settings"),
            (self.save_button, "save"),
        )
        for button, kind in icons:
            button.setIcon(_toolbar_icon(kind, foreground))

    def _apply_ticks(self) -> None:
        self.scan_plot_item.getAxis("bottom").configure(
            scale_mode="linear",
            major_ticks=self.x_major_ticks,
            minor_ticks=self.x_minor_ticks,
        )
        self.scan_plot_item.getAxis("left").configure(
            scale_mode=self.scale_mode,
            major_ticks=self.y_major_ticks,
            minor_ticks=self.y_minor_ticks,
        )
        self.phase_plot_item.getAxis("bottom").configure(
            scale_mode="linear",
            major_ticks=self.x_major_ticks,
            minor_ticks=self.x_minor_ticks,
        )

    def _apply_grid(self) -> None:
        alpha = self.grid_alpha if self.grid_enabled else 0.0
        self.scan_plot_item.showGrid(
            x=self.grid_enabled,
            y=self.grid_enabled,
            alpha=alpha,
        )
        self.phase_plot_item.showGrid(
            x=self.grid_enabled,
            y=False,
            alpha=alpha,
        )

    def _reset_clicked(self) -> None:
        self.set_zoom_enabled(False)
        self.reset_requested.emit()

    def set_zoom_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.zoom_button.isChecked() != enabled:
            self.zoom_button.blockSignals(True)
            self.zoom_button.setChecked(enabled)
            self.zoom_button.blockSignals(False)
        self.scan_view_box.set_zoom_enabled(enabled)
        self.phase_view_box.set_zoom_enabled(enabled)

    def _zoom_region(
        self,
        source: str,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
    ) -> None:
        self.set_zoom_enabled(False)
        x_limits = tuple(sorted((float(x_start), float(x_end))))
        y_limits = tuple(sorted((float(y_start), float(y_end))))
        if (
            math.isclose(x_limits[0], x_limits[1], rel_tol=0.0, abs_tol=1e-12)
            or math.isclose(y_limits[0], y_limits[1], rel_tol=0.0, abs_tol=1e-12)
        ):
            return
        view_box = self.scan_view_box if source == "scan" else self.phase_view_box
        view_box.setRange(
            xRange=x_limits,
            yRange=y_limits,
            padding=0.0,
            disableAutoRange=True,
        )

    def show_settings(self) -> None:
        ViewerPlotSettingsDialog(self, self).exec()

    def apply_settings(
        self,
        *,
        line_width: float,
        grid_enabled: bool,
        grid_alpha: float,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        custom_x_label: str,
        custom_y_label: str,
        legend_enabled: bool,
        x_major_ticks: bool,
        x_minor_ticks: bool,
        y_major_ticks: bool,
        y_minor_ticks: bool,
    ) -> None:
        self.line_width = max(0.25, min(8.0, float(line_width)))
        self.grid_enabled = bool(grid_enabled)
        self.grid_alpha = max(0.0, min(0.4, float(grid_alpha)))
        self.custom_x_label = str(custom_x_label)
        self.custom_y_label = str(custom_y_label)
        self.legend_enabled = bool(legend_enabled)
        self.x_major_ticks = bool(x_major_ticks)
        self.x_minor_ticks = bool(x_minor_ticks)
        self.y_major_ticks = bool(y_major_ticks)
        self.y_minor_ticks = bool(y_minor_ticks)
        self._apply_ticks()
        self._apply_grid()
        self.settings_changed.emit()
        self.set_physical_scan_range(x_limits, y_limits)

    def _sync_overlay_geometry(self) -> None:
        self.overlay_view_box.setGeometry(self.scan_view_box.sceneBoundingRect())
        self.overlay_view_box.linkedViewChanged(
            self.scan_view_box,
            self.overlay_view_box.XAxis,
        )
        self._position_overlay_labels()

    def _position_overlay_labels(self) -> None:
        if not self._overlay_labels:
            return
        x_range = self.scan_view_box.viewRange()[0]
        span = x_range[1] - x_range[0]
        for item, fraction in self._overlay_labels:
            item.setX(float(x_range[0] + fraction * span))

    def begin_frame(
        self,
        *,
        scale_mode: str,
        x_label: str,
        y_label: str,
        separate: bool,
        phase_height_percent: float,
    ) -> None:
        self.scan_plot_item.clear()
        self.phase_plot_item.clear()
        self.overlay_view_box.clear()
        self._overlay_items.clear()
        self._overlay_labels.clear()
        self._temporary_scan_items.clear()
        if self._legend is not None:
            self._legend.clear()
            self.scan_plot_item.scene().removeItem(self._legend)
            self._legend = None
        if scale_mode not in {"linear", "log", "sqrt", "square"}:
            raise ValueError(scale_mode)
        logarithmic = scale_mode == "log"
        if logarithmic and not self._logarithmic:
            # AxisItem interprets the current ViewBox range as base-10
            # exponents as soon as log mode is enabled.  Reset the transient
            # range first so a preceding large linear range cannot overflow.
            self.scan_view_box.setYRange(0.0, 1.0, padding=0.0)
        self.scale_mode = scale_mode
        self._logarithmic = logarithmic
        self.scan_plot_item.setLogMode(x=False, y=self._logarithmic)
        self.phase_plot_item.setLogMode(x=False, y=False)
        self._automatic_x_label = str(x_label)
        self._automatic_y_label = str(y_label)
        self.scan_plot_item.setLabel(
            "bottom",
            self.custom_x_label or self._automatic_x_label,
        )
        self.scan_plot_item.setLabel(
            "left",
            self.custom_y_label or self._automatic_y_label,
        )
        self.phase_plot_item.setLabel("bottom", "2θ, °")
        self._apply_ticks()
        self.set_separate(separate, phase_height_percent)
        self._apply_palette()

    def set_separate(self, separate: bool, phase_height_percent: float) -> None:
        self._separate = bool(separate)
        self.separator.setVisible(self._separate)
        self.phase_plot_widget.setVisible(self._separate)
        fraction = max(10, min(85, round(float(phase_height_percent))))
        self.graph_layout.setStretch(0, 100 - fraction if self._separate else 1)
        self.graph_layout.setStretch(2, fraction if self._separate else 0)

    def set_axes_linked(self, linked: bool) -> None:
        linked = bool(linked and self._separate)
        self._axes_linked = linked

    @property
    def phase_visible(self) -> bool:
        return self._separate

    def add_scan_curve(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        colour: str,
        name: str,
        *,
        line_width: float | None = None,
        line_style: Qt.PenStyle | None = None,
        alpha: float = 1.0,
    ) -> object:
        width = self.line_width if line_width is None else float(line_width)
        pen_colour = QColor(colour)
        pen_colour.setAlphaF(max(0.0, min(1.0, float(alpha))))
        pen = pg.mkPen(pen_colour, width=width)
        if line_style is not None:
            pen.setStyle(line_style)
        item = pg.PlotDataItem(
            np.asarray(x_values, dtype=float),
            np.asarray(y_values, dtype=float),
            pen=pen,
            name=name,
            connect="finite",
            skipFiniteCheck=False,
        )
        self.scan_plot_item.addItem(item)
        return item

    def add_scan_sticks(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        colour: str,
        name: str,
        *,
        baseline: float = 0.0,
        line_width: float | None = None,
    ) -> object:
        """Add vertical sticks to the main scan axes."""

        values = np.asarray(y_values, dtype=float)
        starts = np.full(values.shape, float(baseline), dtype=float)
        x, y = self._segment_arrays(x_values, starts, values)
        width = self.line_width if line_width is None else float(line_width)
        item = pg.PlotCurveItem(
            x=x,
            y=y,
            pen=pg.mkPen(colour, width=width),
            name=name,
            connect="finite",
        )
        self.scan_plot_item.addItem(item)
        return item

    def set_toolbar_visible(self, visible: bool) -> None:
        """Show the full Viewer toolbar or collapse it for embedded previews."""

        for button in (
            self.home_button,
            self.zoom_button,
            self.settings_button,
            self.save_button,
        ):
            button.setVisible(bool(visible))

    def set_navigation_enabled(self, enabled: bool) -> None:
        """Enable right-button panning while preserving left-click signals."""

        enabled = bool(enabled)
        self.scan_plot_item.setMouseEnabled(x=enabled, y=enabled)
        self.phase_plot_item.setMouseEnabled(x=enabled, y=enabled)
        if not enabled:
            self.set_zoom_enabled(False)

    def add_legend(self, entries: Sequence[tuple[object, str]]) -> None:
        if not entries or not self.legend_enabled:
            return
        self._legend = pg.LegendItem(offset=(-8, 8))
        self._legend.setParentItem(self.scan_plot_item.vb)
        self._legend.anchor((1, 0), (1, 0))
        for item, name in entries:
            self._legend.addItem(item, name)

    @staticmethod
    def _segment_arrays(
        x_values: Sequence[float],
        starts: Sequence[float],
        ends: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        x_source = np.asarray(x_values, dtype=float)
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        x = np.full(x_source.size * 3, np.nan, dtype=float)
        y = np.full(x_source.size * 3, np.nan, dtype=float)
        x[0::3] = x_source
        x[1::3] = x_source
        y[0::3] = starts
        y[1::3] = ends
        return x, y

    def _add_filled_profile(
        self,
        target,
        x_values: Sequence[float],
        baseline: Sequence[float] | float,
        values: Sequence[float],
        colour: str,
    ) -> list[object]:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(values, dtype=float)
        base = (
            np.full_like(y, float(baseline))
            if np.isscalar(baseline)
            else np.asarray(baseline, dtype=float)
        )
        lower = pg.PlotCurveItem(x=x, y=base, pen=None, connect="finite")
        upper = pg.PlotCurveItem(
            x=x,
            y=y,
            pen=pg.mkPen(colour, width=max(0.5, self.line_width * 0.87)),
            connect="finite",
        )
        brush = QColor(colour)
        brush.setAlpha(42)
        fill = pg.FillBetweenItem(lower, upper, brush=brush)
        target.addItem(lower)
        target.addItem(upper)
        target.addItem(fill)
        return [lower, upper, fill]

    def add_overlay_sticks(
        self,
        x_values: Sequence[float],
        starts: Sequence[float],
        ends: Sequence[float],
        colour: str,
    ) -> None:
        x, y = self._segment_arrays(x_values, starts, ends)
        item = pg.PlotCurveItem(
            x=x,
            y=y,
            pen=pg.mkPen(colour, width=max(0.5, self.line_width * 0.87)),
            connect="finite",
        )
        self.overlay_view_box.addItem(item)
        self._overlay_items.append(item)

    def add_overlay_profile(
        self,
        x_values: Sequence[float],
        baseline: float,
        values: Sequence[float],
        colour: str,
    ) -> None:
        self._overlay_items.extend(
            self._add_filled_profile(
                self.overlay_view_box,
                x_values,
                baseline,
                values,
                colour,
            )
        )

    def add_overlay_label(
        self,
        text: str,
        x: float,
        y: float,
        colour: str,
        *,
        anchor: tuple[float, float],
        x_fraction: float | None = None,
    ) -> None:
        item = pg.TextItem(text=text, color=colour, anchor=anchor)
        item.setPos(float(x), float(y))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.overlay_view_box.addItem(item, ignoreBounds=True)
        self._overlay_items.append(item)
        if x_fraction is not None:
            self._overlay_labels.append((item, float(x_fraction)))
            self._position_overlay_labels()

    def add_separate_sticks(
        self,
        x_values: Sequence[float],
        starts: Sequence[float],
        ends: Sequence[float],
        colour: str,
    ) -> None:
        x, y = self._segment_arrays(x_values, starts, ends)
        self.phase_plot_item.addItem(
            pg.PlotCurveItem(
                x=x,
                y=y,
                pen=pg.mkPen(colour, width=max(0.5, self.line_width * 0.87)),
                connect="finite",
            )
        )

    def add_separate_profile(
        self,
        x_values: Sequence[float],
        baseline: float,
        values: Sequence[float],
        colour: str,
    ) -> None:
        self._add_filled_profile(
            self.phase_plot_item,
            x_values,
            baseline,
            values,
            colour,
        )

    def add_separate_label(
        self,
        text: str,
        x: float,
        y: float,
        colour: str,
    ) -> None:
        item = pg.TextItem(text=text, color=colour, anchor=(0.0, 0.0))
        item.setPos(float(x), float(y))
        self.phase_plot_item.addItem(item, ignoreBounds=True)

    def show_placeholder(
        self,
        text: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        x = 0.5 * (x_limits[0] + x_limits[1])
        if self._logarithmic:
            low = max(y_limits[0], np.finfo(float).tiny)
            high = max(y_limits[1], low * 1.0001)
            y = 0.5 * (math.log10(low) + math.log10(high))
        else:
            y = 0.5 * (y_limits[0] + y_limits[1])
        item = pg.TextItem(
            text=text,
            color=self.palette().color(QPalette.ColorRole.Text),
            anchor=(0.5, 0.5),
        )
        item.setPos(float(x), float(y))
        self.scan_plot_item.addItem(item, ignoreBounds=True)

    def scan_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_range, y_range = self.scan_view_box.viewRange()
        if self._logarithmic:
            y_range = [10.0 ** max(-307.0, min(307.0, value)) for value in y_range]
        return tuple(map(float, x_range)), tuple(map(float, y_range))

    def physical_scan_limits(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return limits as original intensity values for the settings UI."""

        x_range, view_y_range = self.scan_view_box.viewRange()
        return tuple(map(float, x_range)), self.physical_y_limits(view_y_range)

    def physical_y_limits(
        self,
        display_limits: Sequence[float],
    ) -> tuple[float, float]:
        axis = self.scan_plot_item.getAxis("left")
        values = [axis._physical_value(value) for value in display_limits]
        return tuple(sorted(map(float, values)))

    def display_y_limits(
        self,
        physical_limits: Sequence[float],
    ) -> tuple[float, float]:
        if self.scale_mode == "log":
            return tuple(map(float, physical_limits))
        axis = self.scan_plot_item.getAxis("left")
        values = [axis._axis_position(value) for value in physical_limits]
        return tuple(sorted(map(float, values)))

    def phase_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_range, y_range = self.phase_view_box.viewRange()
        return tuple(map(float, x_range)), tuple(map(float, y_range))

    def set_scan_range(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        y_range = y_limits
        if self._logarithmic:
            low = max(float(y_limits[0]), np.finfo(float).tiny)
            high = max(float(y_limits[1]), low * 1.0000001)
            y_range = (math.log10(low), math.log10(high))
        self.scan_view_box.setRange(
            xRange=tuple(map(float, x_limits)),
            yRange=tuple(map(float, y_range)),
            padding=0.0,
            disableAutoRange=True,
        )

    def set_physical_scan_range(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        """Apply settings limits expressed as original intensity values."""

        self.set_scan_range(x_limits, self.display_y_limits(y_limits))

    def set_phase_range(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        self.phase_view_box.setRange(
            xRange=tuple(map(float, x_limits)),
            yRange=tuple(map(float, y_limits)),
            padding=0.0,
            disableAutoRange=True,
        )

    def view_to_data_y(self, value: float) -> float:
        return 10.0 ** float(value) if self._logarithmic else float(value)

    def data_to_view_y(self, value: float) -> float:
        if not self._logarithmic:
            return float(value)
        return math.log10(value) if value > 0.0 else math.nan

    def scan_fraction_y(self, value: float) -> float:
        y_range = self.scan_view_box.viewRange()[1]
        span = y_range[1] - y_range[0]
        return 0.0 if math.isclose(span, 0.0) else (float(value) - y_range[0]) / span

    @staticmethod
    def _scene_distance(view_box, first, second) -> float:
        a = view_box.mapViewToScene(QPointF(float(first[0]), float(first[1])))
        b = view_box.mapViewToScene(QPointF(float(second[0]), float(second[1])))
        return math.hypot(float(a.x() - b.x()), float(a.y() - b.y()))

    def scan_distance(self, first, second) -> float:
        return self._scene_distance(self.scan_view_box, first, second)

    def phase_distance(self, first, second) -> float:
        return self._scene_distance(self.phase_view_box, first, second)

    def overlay_distance(self, first, second) -> float:
        return self._scene_distance(self.overlay_view_box, first, second)

    def set_peak_selection_enabled(self, enabled: bool) -> None:
        self.scan_view_box.set_selection_enabled(enabled)

    def show_peak_fit(
        self,
        selected_x: Sequence[float],
        selected_y: Sequence[float],
        fit_x: Sequence[float],
        fit_y: Sequence[float],
        centre: float,
        centre_y: float,
    ) -> None:
        items = [
            pg.PlotDataItem(
                x=np.asarray(selected_x, dtype=float),
                y=np.asarray(selected_y, dtype=float),
                pen=None,
                symbol="o",
                symbolPen=None,
                symbolBrush=pg.mkBrush("#2ca02c"),
                symbolSize=7,
            ),
            pg.PlotDataItem(
                x=np.asarray(fit_x, dtype=float),
                y=np.asarray(fit_y, dtype=float),
                pen=pg.mkPen("#ff9800", width=2.0),
                connect="finite",
            ),
            pg.InfiniteLine(
                pos=float(centre),
                angle=90,
                pen=pg.mkPen("#d32f2f", width=1.0, style=Qt.PenStyle.DashLine),
            ),
            pg.PlotDataItem(
                x=[float(centre)],
                y=[float(centre_y)],
                pen=None,
                symbol="o",
                symbolPen=pg.mkPen("#d32f2f"),
                symbolBrush=pg.mkBrush("#d32f2f"),
                symbolSize=10,
            ),
        ]
        for item in items:
            self.scan_plot_item.addItem(item)
        self._temporary_scan_items.extend(items)

    def save_png(self, path: str | Path) -> bool:
        return bool(self.graphs.grab().save(str(path), "PNG"))

    def save_image(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("text.save_figure"),
            self.save_filename,
            "PNG (*.png)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        if not self.save_png(target):
            QMessageBox.critical(self, tr("text.save_figure"), str(target))
