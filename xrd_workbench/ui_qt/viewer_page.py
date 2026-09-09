"""PySide6 page for one-dimensional experimental measurements."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..localization import tr
from ..models.data_errors import XRDDataError
from ..models.project import SCAN, VIEWER
from ..models.radiation import RadiationSettings
from ..models.scan import clone_scan
from ..models.viewer import (
    PlotItem,
    ViewerState,
    axis_has_degree_units,
    axis_key,
    intensity_limits,
    resolve_limits,
    scan_x_limits,
    scrolled_limits,
    scrollbar_window,
    transformed_intensity,
)
from ..services.diffraction import d_spacing_from_two_theta


MAX_LEGEND_ITEMS = 12
SCROLL_RESOLUTION = 10_000


def _axis_label(name: str) -> str:
    labels = {
        "2theta": "2θ",
        "twotheta": "2θ",
        "2θ": "2θ",
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
    return labels.get(axis_key(name), name)


class DocumentTree(QTreeWidget):
    """Tree with the small ``count`` compatibility used by the alpha shell tests."""

    def count(self) -> int:
        return self.topLevelItemCount()


class CollapsibleSection(QWidget):
    """Compact disclosure section whose contents start collapsed."""

    def __init__(self, title: str, parent=None, *, expanded: bool = False) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.toggle = QToolButton()
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.setStyleSheet("QToolButton { text-align: left; font-weight: 600; }")
        self.toggle.clicked.connect(self._toggle)
        layout.addWidget(self.toggle)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(14, 3, 2, 7)
        self.content_layout.setSpacing(6)
        layout.addWidget(self.content)

        self._title = title
        self.set_expanded(expanded)

    def set_title(self, title: str) -> None:
        self._title = title
        self.toggle.setText(title)

    def set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _toggle(self) -> None:
        self.set_expanded(not self.content.isVisible())


class ViewerPage(QWidget):
    """Experimental-scan viewer backed by the GUI-independent ViewerState."""

    title_key = "text.viewer"

    def __init__(
        self,
        store,
        radiation_settings: RadiationSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.workspace = VIEWER
        self.store = store
        self.radiation_settings = radiation_settings
        self.viewer_state = ViewerState()
        self.items = self.viewer_state.items

        self._refreshing_tree = False
        self._refreshing_axis = False
        self._syncing_scrollbars = False
        self._drawing = False
        self._has_drawn_data = False
        self._selected_point: tuple[str, int] | None = None
        self._navigation_x_bounds = (0.0, 1.0)
        self._navigation_y_bounds = (0.0, 1.0)

        self._build_ui()
        self._connect_plot_events()
        self.retranslate()
        self.refresh_documents()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setMinimumWidth(275)
        controls_scroll.setMaximumWidth(440)
        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        self.controls_layout.setContentsMargins(9, 9, 7, 9)
        self.controls_layout.setSpacing(7)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        self.data_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.data_section)
        self.documents = DocumentTree()
        self.documents.setColumnCount(3)
        self.documents.setRootIsDecorated(False)
        self.documents.setAlternatingRowColors(True)
        self.documents.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.documents.setMinimumHeight(150)
        self.documents.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.documents.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.documents.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.documents.itemChanged.connect(self._tree_item_changed)
        self.documents.itemSelectionChanged.connect(self._selection_changed)
        self.data_section.content_layout.addWidget(self.documents)

        data_buttons = QGridLayout()
        self.visibility_button = QPushButton()
        self.colour_button = QPushButton()
        self.up_button = QPushButton()
        self.down_button = QPushButton()
        self.visibility_button.clicked.connect(self.toggle_selected_visibility)
        self.colour_button.clicked.connect(self.choose_selected_colour)
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        self.down_button.clicked.connect(lambda: self.move_selected(1))
        data_buttons.addWidget(self.visibility_button, 0, 0)
        data_buttons.addWidget(self.colour_button, 0, 1)
        data_buttons.addWidget(self.up_button, 1, 0)
        data_buttons.addWidget(self.down_button, 1, 1)
        self.data_section.content_layout.addLayout(data_buttons)

        self.display_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.display_section)
        display_form = QFormLayout()
        self.axis_label = QLabel()
        self.axis_combo = QComboBox()
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        display_form.addRow(self.axis_label, self.axis_combo)

        self.scale_label = QLabel()
        self.scale_combo = QComboBox()
        self.scale_combo.currentIndexChanged.connect(self._scale_changed)
        display_form.addRow(self.scale_label, self.scale_combo)

        self.offset_label = QLabel()
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(6)
        self.offset_spin.setRange(-1.0e12, 1.0e12)
        self.offset_spin.setKeyboardTracking(False)
        self.offset_spin.valueChanged.connect(self._offset_changed)
        display_form.addRow(self.offset_label, self.offset_spin)
        self.display_section.content_layout.addLayout(display_form)

        self.redraw_button = QPushButton()
        self.redraw_button.clicked.connect(lambda: self._draw(preserve_view=True))
        self.display_section.content_layout.addWidget(self.redraw_button)

        self.limits_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.limits_section)
        limits_grid = QGridLayout()
        self.x_min_edit = QLineEdit()
        self.x_max_edit = QLineEdit()
        self.y_min_edit = QLineEdit()
        self.y_max_edit = QLineEdit()
        for edit in (self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit):
            edit.setPlaceholderText("auto")
        limits_grid.addWidget(QLabel("X min"), 0, 0)
        limits_grid.addWidget(self.x_min_edit, 0, 1)
        limits_grid.addWidget(QLabel("X max"), 0, 2)
        limits_grid.addWidget(self.x_max_edit, 0, 3)
        limits_grid.addWidget(QLabel("Y min"), 1, 0)
        limits_grid.addWidget(self.y_min_edit, 1, 1)
        limits_grid.addWidget(QLabel("Y max"), 1, 2)
        limits_grid.addWidget(self.y_max_edit, 1, 3)
        self.limits_section.content_layout.addLayout(limits_grid)

        limits_buttons = QHBoxLayout()
        self.apply_limits_button = QPushButton()
        self.auto_limits_button = QPushButton()
        self.apply_limits_button.clicked.connect(self.apply_limits)
        self.auto_limits_button.clicked.connect(self.reset_limits)
        limits_buttons.addWidget(self.apply_limits_button)
        limits_buttons.addWidget(self.auto_limits_button)
        self.limits_section.content_layout.addLayout(limits_buttons)
        self.controls_layout.addStretch(1)

        plot_widget = QWidget()
        plot_layout = QGridLayout(plot_widget)
        plot_layout.setContentsMargins(4, 6, 8, 6)
        plot_layout.setSpacing(3)

        self.figure = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()
        plot_layout.addWidget(self.canvas, 0, 0)

        self.y_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.y_scrollbar.valueChanged.connect(lambda value: self._scrollbar_changed("y", value))
        plot_layout.addWidget(self.y_scrollbar, 0, 1)

        self.toolbar = NavigationToolbar2QT(self.canvas, plot_widget)
        plot_layout.addWidget(self.toolbar, 1, 0, 1, 2)

        self.x_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.x_scrollbar.valueChanged.connect(lambda value: self._scrollbar_changed("x", value))
        plot_layout.addWidget(self.x_scrollbar, 2, 0)

        self.selected_group = QGroupBox()
        selected_layout = QVBoxLayout(self.selected_group)
        selected_layout.setContentsMargins(8, 7, 8, 7)
        self.selection_info = QLabel()
        self.selection_info.setWordWrap(True)
        self.selection_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        selected_layout.addWidget(self.selection_info)
        plot_layout.addWidget(self.selected_group, 3, 0, 1, 2)

        self.status = QLabel()
        self.status.setWordWrap(True)
        plot_layout.addWidget(self.status, 4, 0, 1, 2)
        plot_layout.setRowStretch(0, 1)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((315, 900))

    def _connect_plot_events(self) -> None:
        self.canvas.mpl_connect("button_press_event", self._plot_clicked)
        self.canvas.mpl_connect("scroll_event", self._plot_scrolled)

    def retranslate(self) -> None:
        self.data_section.set_title(tr("text.loaded_datasets"))
        self.display_section.set_title(tr("text.display"))
        self.limits_section.set_title(tr("text.plot_limits"))
        self.documents.setHeaderLabels(
            (tr("text.name"), tr("text.vis"), tr("text.colour"))
        )
        self.axis_label.setText(tr("text.x_axis"))
        self.scale_label.setText(tr("text.y_scale"))
        self.offset_label.setText(tr("text.curve_offset"))
        self.redraw_button.setText(tr("text.redraw"))
        self.colour_button.setText(tr("text.colour"))
        self.up_button.setText(tr("text.move_up"))
        self.down_button.setText(tr("text.move_down"))
        self.apply_limits_button.setText(tr("text.apply"))
        self.auto_limits_button.setText(tr("text.auto"))
        self.selected_group.setTitle(tr("text.selected_point_reflection"))

        current_scale = self.scale_combo.currentData() or self.viewer_state.plot.intensity_scale
        self.scale_combo.blockSignals(True)
        self.scale_combo.clear()
        for key, code in (
            ("text.linear", "linear"),
            ("text.logarithmic", "log"),
            ("text.square_root", "sqrt"),
            ("text.square", "square"),
        ):
            self.scale_combo.addItem(tr(key), code)
        index = self.scale_combo.findData(current_scale)
        self.scale_combo.setCurrentIndex(max(0, index))
        self.scale_combo.blockSignals(False)
        for edit in (self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit):
            edit.setPlaceholderText(tr("text.auto"))
        self._refresh_tree()
        self._update_buttons()
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def refresh_documents(self) -> None:
        assigned = {
            document.uid: document
            for document in self.store.assigned_documents(VIEWER, kind=SCAN)
        }
        removed = [uid for uid in self.items if uid not in assigned]
        added = []
        for uid in removed:
            self.viewer_state.remove(uid)
            if self._selected_point and self._selected_point[0] == uid:
                self._selected_point = None
        for uid, document in assigned.items():
            if uid in self.items:
                self.items[uid].name = document.name
                continue
            self.viewer_state.add(
                PlotItem(
                    uid=uid,
                    name=document.name,
                    kind=document.kind,
                    source=document.source,
                    colour=self.viewer_state.next_colour(),
                    scan=clone_scan(document.payload),
                )
            )
            added.append(uid)
        self._refresh_tree()
        if removed or added:
            self._draw(preserve_view=not bool(added))
        else:
            self._update_buttons()

    def _selected_uid(self) -> str | None:
        selected = self.documents.selectedItems()
        if not selected:
            return None
        return selected[0].data(0, Qt.ItemDataRole.UserRole)

    def _refresh_tree(self) -> None:
        if not hasattr(self, "documents"):
            return
        selected_uid = self._selected_uid()
        self._refreshing_tree = True
        self.documents.blockSignals(True)
        self.documents.clear()
        selected_item = None
        for item in self.items.values():
            row = QTreeWidgetItem((item.name, "", "■"))
            row.setData(0, Qt.ItemDataRole.UserRole, item.uid)
            row.setToolTip(0, str(item.source))
            row.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            row.setCheckState(
                1,
                Qt.CheckState.Checked if item.visible else Qt.CheckState.Unchecked,
            )
            row.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            row.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
            row.setForeground(2, QBrush(QColor(item.colour)))
            self.documents.addTopLevelItem(row)
            if item.uid == selected_uid:
                selected_item = row
        if selected_item is not None:
            self.documents.setCurrentItem(selected_item)
        self.documents.blockSignals(False)
        self._refreshing_tree = False
        self._selection_changed()

    def _tree_item_changed(self, row: QTreeWidgetItem, column: int) -> None:
        if self._refreshing_tree or column != 1:
            return
        uid = row.data(0, Qt.ItemDataRole.UserRole)
        item = self.items.get(uid)
        if item is None:
            return
        item.visible = row.checkState(1) == Qt.CheckState.Checked
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def _selection_changed(self) -> None:
        self._populate_axis_combo()
        self._update_buttons()

    def _update_buttons(self) -> None:
        uid = self._selected_uid() if hasattr(self, "documents") else None
        item = self.items.get(uid) if uid else None
        enabled = item is not None
        for button in (self.visibility_button, self.colour_button):
            button.setEnabled(enabled)
        self.visibility_button.setText(
            tr("text.hide") if item is not None and item.visible else tr("text.show")
        )
        group = self.viewer_state.group_uids(uid) if uid else []
        index = group.index(uid) if uid in group else -1
        self.up_button.setEnabled(index > 0)
        self.down_button.setEnabled(0 <= index < len(group) - 1)

    def _populate_axis_combo(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        self._refreshing_axis = True
        self.axis_combo.blockSignals(True)
        self.axis_combo.clear()
        if item is not None and item.scan is not None:
            for name in item.scan.available_axes:
                self.axis_combo.addItem(_axis_label(name), name)
            index = self.axis_combo.findData(item.scan.axis_name)
            self.axis_combo.setCurrentIndex(max(0, index))
        self.axis_combo.setEnabled(self.axis_combo.count() > 0)
        self.axis_combo.blockSignals(False)
        self._refreshing_axis = False

    def _axis_changed(self, _index: int) -> None:
        if self._refreshing_axis:
            return
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        axis_name = self.axis_combo.currentData()
        if item is None or item.scan is None or not axis_name:
            return
        item.scan.use_axis(str(axis_name))
        self._selected_point = None
        self._draw(preserve_view=False)

    def _scale_changed(self, _index: int) -> None:
        code = self.scale_combo.currentData()
        if not code:
            return
        old_code = self.viewer_state.plot.intensity_scale
        self.viewer_state.plot.intensity_scale = str(code)
        self._draw(preserve_view=True, preserve_y=old_code == code)

    def _offset_changed(self, value: float) -> None:
        self.viewer_state.plot.vertical_offset = float(value)
        self._draw(preserve_view=True)

    def toggle_selected_visibility(self) -> None:
        uid = self._selected_uid()
        if uid is None:
            return
        self.viewer_state.toggle(uid)
        self._refresh_tree()
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def choose_selected_colour(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None:
            return
        colour = QColorDialog.getColor(QColor(item.colour), self, tr("text.colour"))
        if not colour.isValid():
            return
        item.colour = colour.name()
        self._refresh_tree()
        self._draw(preserve_view=True)

    def move_selected(self, direction: int) -> None:
        uid = self._selected_uid()
        group = self.viewer_state.group_uids(uid) if uid else []
        if uid not in group:
            return
        index = group.index(uid) + int(direction)
        if not 0 <= index < len(group):
            return
        self.viewer_state.move_to_target(uid, group[index])
        self._refresh_tree()
        self._draw(preserve_view=True)

    @staticmethod
    def _optional_number(edit: QLineEdit) -> float | None:
        value = edit.text().strip().replace(",", ".")
        return None if not value else float(value)

    def apply_limits(self) -> None:
        try:
            x_minimum = self._optional_number(self.x_min_edit)
            x_maximum = self._optional_number(self.x_max_edit)
            y_minimum = self._optional_number(self.y_min_edit)
            y_maximum = self._optional_number(self.y_max_edit)
        except (TypeError, ValueError, ArithmeticError):
            QMessageBox.warning(
                self,
                tr("text.plot_limits"),
                tr("qt.viewer_limits_numeric"),
            )
            return
        try:
            x_limits = resolve_limits(
                self._navigation_x_bounds,
                x_minimum,
                x_maximum,
            )
            y_limits = resolve_limits(
                self._navigation_y_bounds,
                y_minimum,
                y_maximum,
            )
        except XRDDataError:
            QMessageBox.warning(
                self,
                tr("text.plot_limits"),
                tr("text.the_lower_limit_must_be_below_the_upper_limit"),
            )
            return
        if self.viewer_state.plot.intensity_scale == "log" and y_limits[0] <= 0:
            QMessageBox.warning(
                self,
                tr("text.plot_limits"),
                tr("text.the_lower_limit_must_be_positive_for_a_logarithmic_scale"),
            )
            return
        self._set_limits(x_limits, y_limits)

    def reset_limits(self) -> None:
        for edit in (self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit):
            edit.clear()
        self._draw(preserve_view=False)

    def _visible_plot_arrays(self) -> list[tuple[PlotItem, np.ndarray, np.ndarray]]:
        mode = self.viewer_state.plot.intensity_scale
        offset = self.viewer_state.plot.vertical_offset
        result = []
        for index, item in enumerate(self.viewer_state.visible_scans()):
            x_values, physical_y = item.display_arrays()
            plotted_y = transformed_intensity(physical_y, mode)
            if offset:
                plotted_y = plotted_y + index * offset
            result.append((item, x_values, plotted_y))
        return result

    def _draw(self, *, preserve_view: bool, preserve_y: bool = True) -> None:
        if not hasattr(self, "axis") or self._drawing:
            return
        self._drawing = True
        old_x = self.axis.get_xlim()
        old_y = self.axis.get_ylim()
        had_data = self._has_drawn_data
        mode = self.viewer_state.plot.intensity_scale
        arrays = self._visible_plot_arrays()
        try:
            self.axis.clear()
            self.axis.set_yscale("log" if mode == "log" else "linear")
            for item, x_values, plotted_y in arrays:
                self.axis.plot(x_values, plotted_y, color=item.colour, lw=1.15, label=item.name)

            if arrays:
                self._navigation_x_bounds = scan_x_limits(
                    [item for item, _x, _y in arrays]
                )
                self._navigation_y_bounds = intensity_limits(
                    [values for _item, _x, values in arrays],
                    logarithmic=mode == "log",
                )
                if len(arrays) <= MAX_LEGEND_ITEMS:
                    self.axis.legend(loc="best", fontsize=8)
                self.status.setText(tr("qt.viewer_measurement_count", count=len(arrays)))
                self._has_drawn_data = True
            else:
                self._navigation_x_bounds = (0.0, 1.0)
                self._navigation_y_bounds = (0.0, 1.0)
                empty_key = (
                    "qt.viewer_no_visible_measurements"
                    if self.items
                    else "qt.viewer_no_measurements"
                )
                self.axis.text(
                    0.5,
                    0.5,
                    tr(empty_key),
                    ha="center",
                    va="center",
                    transform=self.axis.transAxes,
                    color="#666666",
                )
                self.status.setText(tr(empty_key))
                self._has_drawn_data = False

            visible_axes = {
                item.scan.axis_name
                for item, _x, _y in arrays
                if item.scan is not None
            }
            x_label = _axis_label(next(iter(visible_axes))) if len(visible_axes) == 1 else tr("text.scan_coordinate")
            if len(visible_axes) == 1 and axis_has_degree_units(next(iter(visible_axes))):
                x_label += ", °"
            self.axis.set_xlabel(x_label)
            self.axis.set_ylabel(tr("qt.viewer_intensity"))
            self.axis.grid(True, alpha=0.22)

            x_limits = self._navigation_x_bounds
            y_limits = self._navigation_y_bounds
            if preserve_view and had_data and arrays:
                if all(np.isfinite(old_x)) and old_x[0] < old_x[1]:
                    x_limits = old_x
                if preserve_y and all(np.isfinite(old_y)) and old_y[0] < old_y[1]:
                    if mode != "log" or old_y[0] > 0:
                        y_limits = old_y
            self.axis.set_xlim(*x_limits)
            self.axis.set_ylim(*y_limits)
            self.axis.callbacks.connect(
                "xlim_changed", lambda _axis: self._axes_changed()
            )
            self.axis.callbacks.connect(
                "ylim_changed", lambda _axis: self._axes_changed()
            )
            if not preserve_view:
                self.toolbar.update()
            self._refresh_selection_info()
            self._update_navigation_scrollbars()
            self.canvas.draw_idle()
        finally:
            self._drawing = False

    def _set_limits(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        self._drawing = True
        try:
            self.axis.set_xlim(*x_limits)
            self.axis.set_ylim(*y_limits)
            self._update_navigation_scrollbars()
            self.canvas.draw_idle()
        finally:
            self._drawing = False

    def _axes_changed(self) -> None:
        if not self._drawing:
            self._update_navigation_scrollbars()

    def _configure_scrollbar(
        self,
        scrollbar: QScrollBar,
        bounds: tuple[float, float],
        current: tuple[float, float],
        *,
        vertical: bool,
    ) -> None:
        first, last, movable = scrollbar_window(bounds, current, vertical=vertical)
        scrollbar.blockSignals(True)
        if not movable:
            scrollbar.setRange(0, 0)
            scrollbar.setPageStep(SCROLL_RESOLUTION)
            scrollbar.setValue(0)
            scrollbar.setEnabled(False)
        else:
            size = last - first
            maximum = max(1, round((1.0 - size) * SCROLL_RESOLUTION))
            scrollbar.setRange(0, maximum)
            scrollbar.setPageStep(max(1, round(size * SCROLL_RESOLUTION)))
            scrollbar.setSingleStep(max(1, round(0.02 * SCROLL_RESOLUTION)))
            scrollbar.setValue(round(first * SCROLL_RESOLUTION))
            scrollbar.setEnabled(True)
        scrollbar.blockSignals(False)

    def _update_navigation_scrollbars(self) -> None:
        if self._syncing_scrollbars or not hasattr(self, "axis"):
            return
        self._syncing_scrollbars = True
        try:
            self._configure_scrollbar(
                self.x_scrollbar,
                self._navigation_x_bounds,
                self.axis.get_xlim(),
                vertical=False,
            )
            self._configure_scrollbar(
                self.y_scrollbar,
                self._navigation_y_bounds,
                self.axis.get_ylim(),
                vertical=True,
            )
        finally:
            self._syncing_scrollbars = False

    def _scrollbar_changed(self, dimension: str, value: int) -> None:
        if self._syncing_scrollbars or self._drawing:
            return
        bounds = self._navigation_x_bounds if dimension == "x" else self._navigation_y_bounds
        current = self.axis.get_xlim() if dimension == "x" else self.axis.get_ylim()
        low, high = scrolled_limits(
            bounds,
            current,
            "moveto",
            float(value) / SCROLL_RESOLUTION,
            vertical=dimension == "y",
        )
        self._drawing = True
        try:
            if dimension == "x":
                self.axis.set_xlim(low, high)
            else:
                self.axis.set_ylim(low, high)
            self.canvas.draw_idle()
        finally:
            self._drawing = False
        self._update_navigation_scrollbars()

    def _plot_scrolled(self, event) -> None:
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
            return
        factor = 0.80 if event.button == "up" else 1.25
        x_low, x_high = self.axis.get_xlim()
        x_span = (x_high - x_low) * factor
        x_fraction = (event.xdata - x_low) / (x_high - x_low)
        new_x = (
            event.xdata - x_fraction * x_span,
            event.xdata + (1.0 - x_fraction) * x_span,
        )

        y_low, y_high = self.axis.get_ylim()
        if self.viewer_state.plot.intensity_scale == "log" and y_low > 0 and event.ydata > 0:
            log_low, log_high, log_cursor = map(math.log10, (y_low, y_high, event.ydata))
            log_span = (log_high - log_low) * factor
            fraction = (log_cursor - log_low) / (log_high - log_low)
            new_y = (
                10.0 ** (log_cursor - fraction * log_span),
                10.0 ** (log_cursor + (1.0 - fraction) * log_span),
            )
        else:
            y_span = (y_high - y_low) * factor
            fraction = (event.ydata - y_low) / (y_high - y_low)
            new_y = (
                event.ydata - fraction * y_span,
                event.ydata + (1.0 - fraction) * y_span,
            )
        self._set_limits(new_x, new_y)

    def _plot_clicked(self, event) -> None:
        if (
            event.button != 1
            or event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
            or self.toolbar.mode
        ):
            return
        nearest: tuple[float, PlotItem, int] | None = None
        mode = self.viewer_state.plot.intensity_scale
        offset = self.viewer_state.plot.vertical_offset
        for curve_index, item in enumerate(self.viewer_state.visible_scans()):
            x_values, physical_y = item.display_arrays()
            plotted_y = transformed_intensity(physical_y, mode) + curve_index * offset
            insertion = int(np.searchsorted(x_values, event.xdata))
            for index in range(max(0, insertion - 2), min(len(x_values), insertion + 3)):
                if not np.isfinite(plotted_y[index]):
                    continue
                px, py = self.axis.transData.transform((x_values[index], plotted_y[index]))
                distance = math.hypot(px - event.x, py - event.y)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, item, index)
        if nearest is None:
            return
        _distance, item, index = nearest
        self._selected_point = (item.uid, index)
        self._refresh_selection_info()

    def _d_suffix(self, item: PlotItem, x_value: float) -> str:
        if item.scan is None or axis_key(item.scan.axis_name) not in {"2theta", "twotheta", "2θ"}:
            return ""
        values = []
        for name, wavelength, _weight in self.radiation_settings.lines():
            spacing = d_spacing_from_two_theta(x_value, wavelength)
            if spacing is not None:
                values.append((name, spacing))
        if len(values) == 1:
            return tr("viewer.scan_d_single", value=f"{values[0][1]:.5f}")
        if values:
            rendered = "; ".join(f"{name} = {spacing:.5f} Å" for name, spacing in values)
            return tr("viewer.scan_d_multiple", values=rendered)
        return ""

    def _refresh_selection_info(self) -> None:
        if not hasattr(self, "selection_info"):
            return
        selection = self._selected_point
        item = self.items.get(selection[0]) if selection else None
        if item is None or not item.visible or item.scan is None:
            self._selected_point = None
            self.selection_info.setText(tr("text.select_a_point_or_reflection_on_the_plot"))
            return
        index = int(selection[1])
        x_values, physical_y = item.display_arrays()
        if not 0 <= index < len(x_values):
            self._selected_point = None
            self.selection_info.setText(tr("text.select_a_point_or_reflection_on_the_plot"))
            return
        label = _axis_label(item.scan.axis_name)
        unit = "°" if axis_has_degree_units(item.scan.axis_name) else ""
        self.selection_info.setText(
            tr(
                "viewer.scan_selection",
                name=item.name,
                axis=label,
                value=f"{x_values[index]:.5f}",
                unit=unit,
                d_suffix=self._d_suffix(item, float(x_values[index])),
                intensity=f"{physical_y[index]:.6g}",
            )
        )


__all__ = ["CollapsibleSection", "ViewerPage"]
