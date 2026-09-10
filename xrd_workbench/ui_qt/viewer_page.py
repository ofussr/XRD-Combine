"""PySide6 viewer for one-dimensional measurements and calculated phases."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QSlider,
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
from matplotlib.widgets import RectangleSelector

from ..localization import localised, tr
from ..models.data_errors import XRDDataError
from ..io.correction import write_processed_scan
from ..io.reflections import read_scattering_factors, scattering_factor_path
from ..models.diffraction import ReflectionRow
from ..models.project import CELL_PHASE, CIF, SCAN, VIEWER
from ..models.correction import CorrectionRequest, validate_result_mode
from ..models.radiation import RadiationSettings
from ..models.scan import clone_scan
from ..models.viewer import (
    PlotItem,
    ViewerState,
    axis_has_degree_units,
    axis_key,
    intensity_limits,
    is_two_theta,
    overlay_phase_geometry,
    resolve_limits,
    scan_x_limits,
    scrolled_limits,
    scrollbar_window,
    transformed_intensity,
)
from ..services.diffraction import (
    calculate_reflections,
    d_spacing_from_two_theta,
    gaussian_powder_profile,
)
from ..services.correction import apply_correction
from ..services.peak_fitting import fit_gaussian_peak
from ..services.reference_peaks import read_reference_peaks
from .radiation import RadiationSelector


MAX_LEGEND_ITEMS = 12
SCROLL_RESOLUTION = 10_000
DEFAULT_PHASE_LIMITS = (5.0, 120.0)
DISABLED_PLACEHOLDER_STYLE = "QPushButton:disabled { color: #c62828; }"


def _reference_peak_path() -> Path:
    """Return the editable peak list used by the stable interface."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "pivo.json"
    return Path(__file__).resolve().parents[1] / "pivo.json"


class PeakTargetDialog(QDialog):
    """Qt counterpart of the existing peak-alignment input dialog."""

    def __init__(
        self,
        centre: float,
        intensity: float,
        references: dict[str, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("text.align_peak"))
        self.setModal(True)
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        selected = QLabel(
            tr(
                "qt.correction_selected_peak",
                centre=f"{centre:.6f}",
                intensity=f"{intensity:.2f}",
            )
        )
        selected.setWordWrap(True)
        layout.addWidget(selected)
        layout.addWidget(QLabel(tr("text.select_a_database_reference")))

        self.reference_combo = QComboBox()
        self.reference_combo.addItem("")
        for name, value in references.items():
            self.reference_combo.addItem(name, float(value))
        self.reference_combo.currentIndexChanged.connect(self._reference_selected)
        layout.addWidget(self.reference_combo)

        layout.addWidget(QLabel(tr("text.or_enter_the_true_two_theta_value")))
        self.target_edit = QLineEdit()
        self.target_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.target_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("text.ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            tr("text.cancel_2")
        )
        self.target_edit.setFocus()

    def _reference_selected(self, index: int) -> None:
        value = self.reference_combo.itemData(index)
        if value is not None:
            self.target_edit.setText(str(value))

    def target_text(self) -> str:
        return self.target_edit.text()


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


def _allow_horizontal_shrink(widget: QWidget) -> None:
    """Allow a control to follow the width of the scroll-area viewport."""

    widget.setMinimumWidth(0)
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    widget.setSizePolicy(policy)


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
        *,
        on_open_comparison=None,
        on_open_reflection_table=None,
    ) -> None:
        super().__init__(parent)
        self.workspace = VIEWER
        self.store = store
        self.radiation_settings = radiation_settings
        self.on_open_comparison = on_open_comparison
        self.on_open_reflection_table = on_open_reflection_table
        self.viewer_state = ViewerState()
        self.items = self.viewer_state.items

        self._refreshing_tree = False
        self._refreshing_axis = False
        self._syncing_scrollbars = False
        self._drawing = False
        self._has_drawn_data = False
        self._selected_point: tuple[str, str, object] | None = None
        self._navigation_x_bounds = (0.0, 1.0)
        self._navigation_y_bounds = (0.0, 1.0)
        self._row_cache: dict[str, tuple[tuple, list[ReflectionRow]]] = {}
        self._factors = None
        self._axes_linked = True
        self._syncing_x_limits = False
        self._overlay_phase_top = 0.0
        self._syncing_processing = False
        self._processing_x_limit = 1.0
        self._processing_y_limit = 1.0
        self._fit_selector: RectangleSelector | None = None
        self._fit_uid: str | None = None

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

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.controls_scroll.setMinimumWidth(275)
        self.controls_scroll.setMaximumWidth(440)
        controls = QWidget()
        controls.setMinimumWidth(0)
        controls.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.controls_layout = QVBoxLayout(controls)
        self.controls_layout.setContentsMargins(9, 9, 7, 9)
        self.controls_layout.setSpacing(7)
        self.controls_scroll.setWidget(controls)
        splitter.addWidget(self.controls_scroll)

        self.data_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.data_section)
        self.documents = DocumentTree()
        self.documents.setColumnCount(4)
        self.documents.setRootIsDecorated(False)
        self.documents.setAlternatingRowColors(True)
        self.documents.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.documents.setMinimumWidth(0)
        self.documents.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.documents.setMinimumHeight(150)
        self.documents.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.documents.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.documents.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.documents.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.documents.itemChanged.connect(self._tree_item_changed)
        self.documents.itemSelectionChanged.connect(self._selection_changed)
        self.data_section.content_layout.addWidget(self.documents)

        data_buttons = QGridLayout()
        self.visibility_button = QPushButton()
        self.colour_button = QPushButton()
        self.remove_button = QPushButton()
        self.show_all_button = QPushButton()
        self.rename_button = QPushButton()
        self.clear_button = QPushButton()
        self.table_button = QPushButton()
        self.pole_button = QPushButton()
        self.up_button = QPushButton()
        self.down_button = QPushButton()
        self.structure_button = QPushButton()
        for button in (
            self.visibility_button,
            self.colour_button,
            self.remove_button,
            self.show_all_button,
            self.rename_button,
            self.clear_button,
            self.table_button,
            self.pole_button,
            self.up_button,
            self.down_button,
            self.structure_button,
        ):
            _allow_horizontal_shrink(button)
        for button in (
            self.pole_button,
            self.structure_button,
        ):
            button.setEnabled(False)
            button.setStyleSheet(DISABLED_PLACEHOLDER_STYLE)
        self.visibility_button.clicked.connect(self.toggle_selected_visibility)
        self.colour_button.clicked.connect(self.choose_selected_colour)
        self.remove_button.clicked.connect(self.remove_selected)
        self.show_all_button.clicked.connect(self.show_all)
        self.rename_button.clicked.connect(self.rename_selected)
        self.clear_button.clicked.connect(self.clear_all)
        if self.on_open_reflection_table is not None:
            self.table_button.clicked.connect(self.open_selected_reflection_table)
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        self.down_button.clicked.connect(lambda: self.move_selected(1))
        data_buttons.addWidget(self.visibility_button, 0, 0)
        data_buttons.addWidget(self.colour_button, 0, 1)
        data_buttons.addWidget(self.remove_button, 0, 2)
        data_buttons.addWidget(self.show_all_button, 0, 3)
        data_buttons.addWidget(self.rename_button, 1, 0, 1, 2)
        data_buttons.addWidget(self.clear_button, 1, 2, 1, 2)
        data_buttons.addWidget(self.table_button, 2, 0, 1, 4)
        data_buttons.addWidget(self.pole_button, 3, 0, 1, 4)
        data_buttons.addWidget(self.up_button, 4, 0, 1, 2)
        data_buttons.addWidget(self.down_button, 4, 2, 1, 2)
        data_buttons.addWidget(self.structure_button, 5, 0, 1, 4)
        self.data_section.content_layout.addLayout(data_buttons)

        self.processing_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.processing_section)
        self.processing_name = QLabel()
        self.processing_name.setWordWrap(True)
        self.processing_section.content_layout.addWidget(self.processing_name)

        processing_grid = QGridLayout()
        self.processing_x_label = QLabel()
        self.processing_y_label = QLabel()
        self.processing_factor_label = QLabel()
        self.processing_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.processing_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.processing_factor_slider = QSlider(Qt.Orientation.Horizontal)
        self.processing_x_spin = QDoubleSpinBox()
        self.processing_y_spin = QDoubleSpinBox()
        self.processing_factor_spin = QDoubleSpinBox()
        for spin in (
            self.processing_x_spin,
            self.processing_y_spin,
            self.processing_factor_spin,
        ):
            _allow_horizontal_shrink(spin)
            spin.setDecimals(6)
            spin.setKeyboardTracking(False)
        self.processing_x_spin.setRange(-1.0, 1.0)
        self.processing_y_spin.setRange(-1.0, 1.0)
        self.processing_factor_spin.setRange(0.05, 5.0)
        for slider in (self.processing_x_slider, self.processing_y_slider):
            slider.setRange(-1000, 1000)
        self.processing_factor_slider.setRange(0, 1000)
        for row, (label, slider, spin) in enumerate(
            (
                (
                    self.processing_x_label,
                    self.processing_x_slider,
                    self.processing_x_spin,
                ),
                (
                    self.processing_y_label,
                    self.processing_y_slider,
                    self.processing_y_spin,
                ),
                (
                    self.processing_factor_label,
                    self.processing_factor_slider,
                    self.processing_factor_spin,
                ),
            )
        ):
            processing_grid.addWidget(label, row, 0)
            processing_grid.addWidget(slider, row, 1)
            processing_grid.addWidget(spin, row, 2)
            processing_grid.setColumnStretch(1, 1)
        self.processing_section.content_layout.addLayout(processing_grid)
        self.processing_x_slider.valueChanged.connect(
            lambda value: self._processing_slider_changed("x", value)
        )
        self.processing_y_slider.valueChanged.connect(
            lambda value: self._processing_slider_changed("y", value)
        )
        self.processing_factor_slider.valueChanged.connect(
            lambda value: self._processing_slider_changed("factor", value)
        )
        self.processing_x_spin.valueChanged.connect(
            lambda value: self._processing_spin_changed("x", value)
        )
        self.processing_y_spin.valueChanged.connect(
            lambda value: self._processing_spin_changed("y", value)
        )
        self.processing_factor_spin.valueChanged.connect(
            lambda value: self._processing_spin_changed("factor", value)
        )

        self.fit_button = QPushButton()
        _allow_horizontal_shrink(self.fit_button)
        self.fit_button.clicked.connect(self.activate_peak_fit)
        self.processing_section.content_layout.addWidget(self.fit_button)

        mode_row = QHBoxLayout()
        self.add_result_radio = QRadioButton()
        self.replace_result_radio = QRadioButton()
        self.add_result_radio.setChecked(True)
        mode_row.addWidget(self.add_result_radio)
        mode_row.addWidget(self.replace_result_radio)
        mode_row.addStretch(1)
        self.processing_section.content_layout.addLayout(mode_row)

        self.shift_omega_check = QCheckBox()
        self.shift_omega_check.setChecked(True)
        self.shift_omega_check.toggled.connect(self._processing_values_changed)
        self.processing_section.content_layout.addWidget(self.shift_omega_check)

        action_row = QGridLayout()
        self.apply_processing_button = QPushButton()
        self.reset_processing_button = QPushButton()
        self.save_processing_button = QPushButton()
        for button in (
            self.apply_processing_button,
            self.reset_processing_button,
            self.save_processing_button,
        ):
            _allow_horizontal_shrink(button)
        self.apply_processing_button.clicked.connect(self.apply_processing_result)
        self.reset_processing_button.clicked.connect(self.reset_processing)
        self.save_processing_button.clicked.connect(self.save_processing_result)
        action_row.addWidget(self.apply_processing_button, 0, 0)
        action_row.addWidget(self.reset_processing_button, 0, 1)
        action_row.addWidget(self.save_processing_button, 1, 0, 1, 2)
        self.processing_section.content_layout.addLayout(action_row)

        self.processing_controls = (
            self.processing_x_slider,
            self.processing_y_slider,
            self.processing_factor_slider,
            self.processing_x_spin,
            self.processing_y_spin,
            self.processing_factor_spin,
            self.fit_button,
            self.add_result_radio,
            self.replace_result_radio,
            self.shift_omega_check,
            self.apply_processing_button,
            self.reset_processing_button,
            self.save_processing_button,
        )

        self.display_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.display_section)
        display_form = QFormLayout()
        self.axis_label = QLabel()
        self.axis_combo = QComboBox()
        _allow_horizontal_shrink(self.axis_combo)
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        display_form.addRow(self.axis_label, self.axis_combo)

        self.scale_label = QLabel()
        self.scale_combo = QComboBox()
        _allow_horizontal_shrink(self.scale_combo)
        self.scale_combo.currentIndexChanged.connect(self._scale_changed)
        display_form.addRow(self.scale_label, self.scale_combo)

        self.offset_label = QLabel()
        self.offset_spin = QDoubleSpinBox()
        _allow_horizontal_shrink(self.offset_spin)
        self.offset_spin.setDecimals(6)
        self.offset_spin.setRange(-1.0e12, 1.0e12)
        self.offset_spin.setKeyboardTracking(False)
        self.offset_spin.valueChanged.connect(self._offset_changed)
        # Intentionally retained but hidden for this migration checkpoint, so
        # the experimental Curve offset control can be restored if it is useful.
        # display_form.addRow(self.offset_label, self.offset_spin)
        self.display_section.content_layout.addLayout(display_form)

        self.redraw_button = QPushButton()
        _allow_horizontal_shrink(self.redraw_button)
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
            _allow_horizontal_shrink(edit)
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
        _allow_horizontal_shrink(self.apply_limits_button)
        _allow_horizontal_shrink(self.auto_limits_button)
        self.apply_limits_button.clicked.connect(self.apply_limits)
        self.auto_limits_button.clicked.connect(self.reset_limits)
        limits_buttons.addWidget(self.apply_limits_button)
        limits_buttons.addWidget(self.auto_limits_button)
        self.limits_section.content_layout.addLayout(limits_buttons)

        self.phase_section = CollapsibleSection("")
        self.phase_section.hide()
        self.controls_layout.addWidget(self.phase_section)
        self.radiation_selector = RadiationSelector(self.radiation_settings)
        self.radiation_selector.radiation_changed.connect(self._radiation_changed)
        self.phase_section.content_layout.addWidget(self.radiation_selector)

        phase_form = QFormLayout()
        self.phase_layout_label = QLabel()
        self.phase_layout_combo = QComboBox()
        _allow_horizontal_shrink(self.phase_layout_combo)
        self.phase_layout_combo.currentIndexChanged.connect(self._phase_layout_changed)
        phase_form.addRow(self.phase_layout_label, self.phase_layout_combo)

        self.phase_style_label = QLabel()
        self.phase_style_combo = QComboBox()
        _allow_horizontal_shrink(self.phase_style_combo)
        self.phase_style_combo.currentIndexChanged.connect(self._phase_style_changed)
        phase_form.addRow(self.phase_style_label, self.phase_style_combo)

        self.fwhm_label = QLabel("FWHM, °")
        self.fwhm_spin = QDoubleSpinBox()
        _allow_horizontal_shrink(self.fwhm_spin)
        self.fwhm_spin.setDecimals(4)
        self.fwhm_spin.setRange(0.0001, 30.0)
        self.fwhm_spin.setValue(0.12)
        self.fwhm_spin.setKeyboardTracking(False)
        self.fwhm_spin.valueChanged.connect(lambda _value: self._draw(preserve_view=True))
        phase_form.addRow(self.fwhm_label, self.fwhm_spin)

        self.phase_height_label = QLabel()
        self.phase_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.phase_height_slider.setRange(10, 85)
        self.phase_height_slider.setValue(round(self.viewer_state.plot.phase_height_percent))
        self.phase_height_slider.valueChanged.connect(self._phase_height_changed)
        self.phase_height_value = QLabel()
        height_row = QHBoxLayout()
        height_row.addWidget(self.phase_height_slider, 1)
        height_row.addWidget(self.phase_height_value)
        phase_form.addRow(self.phase_height_label, height_row)
        self.phase_section.content_layout.addLayout(phase_form)

        self.overlay_single_check = QCheckBox()
        self.overlay_single_check.setChecked(self.viewer_state.plot.overlay_single_line)
        self.overlay_single_check.toggled.connect(self._overlay_arrangement_changed)
        self.phase_section.content_layout.addWidget(self.overlay_single_check)

        overlay_height_row = QHBoxLayout()
        self.overlay_height_label = QLabel()
        self.overlay_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_height_slider.setRange(1, 100)
        self.overlay_height_slider.setValue(
            round(self.viewer_state.plot.overlay_height_percent)
        )
        self.overlay_height_slider.valueChanged.connect(self._overlay_height_changed)
        self.overlay_height_value = QLabel()
        overlay_height_row.addWidget(self.overlay_height_label)
        overlay_height_row.addWidget(self.overlay_height_slider, 1)
        overlay_height_row.addWidget(self.overlay_height_value)
        self.phase_section.content_layout.addLayout(overlay_height_row)

        self.controls_layout.addStretch(1)

        plot_widget = QWidget()
        plot_layout = QGridLayout(plot_widget)
        plot_layout.setContentsMargins(4, 6, 8, 6)
        plot_layout.setSpacing(3)

        self.figure = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.plot_grid = self.figure.add_gridspec(
            3, 1, height_ratios=(0.75, 0.025, 0.25)
        )
        self.scan_axis = self.figure.add_subplot(self.plot_grid[0, 0])
        self.split_axis = self.figure.add_subplot(self.plot_grid[1, 0])
        self.phase_axis = self.figure.add_subplot(self.plot_grid[2, 0])
        self.axis = self.scan_axis
        self.split_axis.set_axis_off()
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

        comparison_bar = QHBoxLayout()
        comparison_bar.setContentsMargins(8, 4, 8, 8)
        comparison_bar.addStretch(1)
        self.comparison_button = QPushButton()
        if self.on_open_comparison is not None:
            self.comparison_button.clicked.connect(self.on_open_comparison)
        comparison_bar.addWidget(self.comparison_button)
        root.addLayout(comparison_bar)

    def _connect_plot_events(self) -> None:
        self.canvas.mpl_connect("button_press_event", self._plot_clicked)
        self.canvas.mpl_connect("scroll_event", self._plot_scrolled)

    def retranslate(self) -> None:
        self.data_section.set_title(tr("text.loaded_datasets"))
        self.processing_section.set_title(tr("text.selected_measurement_correction"))
        self.display_section.set_title(tr("text.display"))
        self.limits_section.set_title(tr("text.plot_limits"))
        self.documents.setHeaderLabels(
            (tr("text.name"), tr("text.vis"), tr("text.type"), tr("text.colour"))
        )
        self.axis_label.setText(tr("text.x_axis"))
        self.scale_label.setText(tr("text.y_scale"))
        self.offset_label.setText(tr("text.curve_offset"))
        self.redraw_button.setText(tr("text.redraw"))
        self.colour_button.setText(tr("text.colour"))
        self.remove_button.setText(tr("text.remove"))
        self.show_all_button.setText(tr("text.all"))
        self.rename_button.setText(tr("text.rename"))
        self.clear_button.setText(tr("text.clear"))
        self.table_button.setText(tr("text.reflection_table"))
        self.pole_button.setText(tr("text.calculated_pole_figure"))
        self.up_button.setText(tr("text.move_up"))
        self.down_button.setText(tr("text.move_down"))
        self.structure_button.setText(tr("text.cif_structure"))
        self.processing_x_label.setText(tr("text.x_shift"))
        self.processing_y_label.setText(tr("text.y_zero_shift"))
        self.processing_factor_label.setText(tr("text.y_multiplier"))
        self.fit_button.setText(tr("text.peak_correction"))
        self.add_result_radio.setText(tr("text.add_new"))
        self.replace_result_radio.setText(tr("text.replace_source"))
        self.shift_omega_check.setText(tr("text.shift_omega_by_1_2"))
        self.apply_processing_button.setText(tr("text.apply_result"))
        self.reset_processing_button.setText(tr("text.reset_transformations"))
        self.save_processing_button.setText(tr("text.save_result"))
        self.comparison_button.setText(
            tr("viewer.send_to_comparison_count", count=self._comparison_count())
        )
        self.apply_limits_button.setText(tr("text.apply"))
        self.auto_limits_button.setText(tr("text.auto"))
        self.selected_group.setTitle(tr("text.selected_point_reflection"))
        self.phase_section.set_title(tr("text.phases"))
        self.phase_layout_label.setText(tr("text.cif_mode"))
        self.phase_style_label.setText(tr("qt.viewer_phase_style"))
        self.phase_height_label.setText(tr("text.cif_height"))
        self.overlay_single_check.setText(tr("text.cif_phases_on_one_line"))
        self.overlay_height_label.setText(tr("text.cif_line_height"))
        self.radiation_selector.retranslate()

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
        current_layout = (
            self.phase_layout_combo.currentData()
            or self.viewer_state.plot.phase_layout
        )
        self.phase_layout_combo.blockSignals(True)
        self.phase_layout_combo.clear()
        self.phase_layout_combo.addItem(tr("text.overlay"), "overlay")
        self.phase_layout_combo.addItem(tr("text.separate"), "separate")
        self.phase_layout_combo.setCurrentIndex(
            max(0, self.phase_layout_combo.findData(current_layout))
        )
        self.phase_layout_combo.blockSignals(False)

        current_style = (
            self.phase_style_combo.currentData()
            or self.viewer_state.plot.phase_style
        )
        self.phase_style_combo.blockSignals(True)
        self.phase_style_combo.clear()
        self.phase_style_combo.addItem(tr("text.sticks"), "sticks")
        self.phase_style_combo.addItem(tr("text.profile"), "profile")
        self.phase_style_combo.setCurrentIndex(
            max(0, self.phase_style_combo.findData(current_style))
        )
        self.phase_style_combo.blockSignals(False)
        for edit in (self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit):
            edit.setPlaceholderText(tr("text.auto"))
        self._refresh_tree()
        self._update_phase_controls()
        self._update_buttons()
        self._load_processing_controls(self._selected_item())
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def refresh_documents(self) -> None:
        assigned = {
            document.uid: document
            for document in self.store.assigned_documents(VIEWER)
            if document.kind in {SCAN, CIF, CELL_PHASE}
        }
        removed = [uid for uid in self.items if uid not in assigned]
        added_scans = []
        added_phases = []
        for uid in removed:
            self.viewer_state.remove(uid)
            self._row_cache.pop(uid, None)
            if self._selected_point and self._selected_point[0] == uid:
                self._selected_point = None
        for uid, document in assigned.items():
            if uid in self.items:
                self.items[uid].name = document.name
                continue
            if document.kind == SCAN:
                item = PlotItem(
                    uid=uid,
                    name=document.name,
                    kind=document.kind,
                    source=document.source,
                    colour=self.viewer_state.next_colour(),
                    scan=clone_scan(document.payload),
                )
                added_scans.append(uid)
            else:
                item = PlotItem(
                    uid=uid,
                    name=document.name,
                    kind=document.kind,
                    source=document.source,
                    colour=self.viewer_state.next_colour(),
                    structure=document.payload.diffraction,
                )
                added_phases.append(uid)
            self.viewer_state.add(item)
        self._refresh_tree()
        self._update_phase_controls()
        if removed or added_scans or added_phases:
            self._draw(preserve_view=not bool(added_scans))
        else:
            self._update_buttons()

    def _selected_uid(self) -> str | None:
        selected = self.documents.selectedItems()
        if not selected:
            return None
        return selected[0].data(0, Qt.ItemDataRole.UserRole)

    def _selected_item(self) -> PlotItem | None:
        uid = self._selected_uid()
        return self.items.get(uid) if uid else None

    def _comparison_count(self) -> int:
        return sum(
            1
            for item in self.items.values()
            if item.scan is not None and is_two_theta(item.scan.axis_name)
        )

    def _refresh_tree(self) -> None:
        if not hasattr(self, "documents"):
            return
        selected_uid = self._selected_uid()
        self._refreshing_tree = True
        self.documents.blockSignals(True)
        self.documents.clear()
        selected_item = None
        for item in self.items.values():
            kind = tr("text.measurement") if item.is_measurement else (
                tr("text.cell") if item.kind == CELL_PHASE else "CIF"
            )
            row = QTreeWidgetItem((item.name, "", kind, "■"))
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
            row.setTextAlignment(3, Qt.AlignmentFlag.AlignCenter)
            row.setForeground(3, QBrush(QColor(item.colour)))
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
        self._update_buttons()
        self._load_processing_controls(item)
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def _selection_changed(self) -> None:
        self._cancel_peak_fit()
        self._populate_axis_combo()
        self._update_buttons()
        self._load_processing_controls(self._selected_item())
        self._load_processing_controls(self._selected_item())

    def _update_buttons(self) -> None:
        uid = self._selected_uid() if hasattr(self, "documents") else None
        item = self.items.get(uid) if uid else None
        enabled = item is not None
        for button in (
            self.visibility_button,
            self.colour_button,
            self.remove_button,
            self.rename_button,
        ):
            button.setEnabled(enabled)
        has_items = bool(self.items)
        self.show_all_button.setEnabled(has_items)
        self.clear_button.setEnabled(has_items)
        self.table_button.setEnabled(
            item is not None
            and item.is_phase
            and self.on_open_reflection_table is not None
        )
        self.visibility_button.setText(
            tr("text.hide") if item is not None and item.visible else tr("text.show")
        )
        group = self.viewer_state.group_uids(uid) if uid else []
        index = group.index(uid) if uid in group else -1
        self.up_button.setEnabled(index > 0)
        self.down_button.setEnabled(0 <= index < len(group) - 1)
        if hasattr(self, "comparison_button"):
            comparison_count = self._comparison_count()
            self.comparison_button.setText(
                tr("viewer.send_to_comparison_count", count=comparison_count)
            )
            self.comparison_button.setEnabled(
                comparison_count > 0 and self.on_open_comparison is not None
            )

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
        self._update_buttons()
        self._load_processing_controls(item)
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

    @staticmethod
    def _processing_slider_value(kind: str, position: int, limit: float) -> float:
        if kind == "factor":
            return 0.05 + (float(position) / 1000.0) * 4.95
        return float(position) / 1000.0 * limit

    @staticmethod
    def _processing_slider_position(kind: str, value: float, limit: float) -> int:
        if kind == "factor":
            return round((float(value) - 0.05) / 4.95 * 1000.0)
        return round(float(value) / max(limit, 1.0e-12) * 1000.0)

    def _load_processing_controls(self, item: PlotItem | None) -> None:
        scan_item = item if item is not None and item.scan is not None else None
        self._syncing_processing = True
        controls_to_block = (
            self.processing_x_slider,
            self.processing_y_slider,
            self.processing_factor_slider,
            self.processing_x_spin,
            self.processing_y_spin,
            self.processing_factor_spin,
            self.shift_omega_check,
        )
        for control in controls_to_block:
            control.blockSignals(True)
        try:
            if scan_item is None:
                self.processing_name.setText(
                    tr("text.select_a_measurement_in_the_list")
                )
                values = (0.0, 0.0, 1.0)
                self._processing_x_limit = 1.0
                self._processing_y_limit = 1.0
            else:
                assert scan_item.scan is not None
                self.processing_name.setText(
                    tr("qt.correction_selected_measurement", name=scan_item.name)
                )
                values = (
                    scan_item.x_shift,
                    scan_item.y_shift,
                    scan_item.y_factor,
                )
                x_values = np.asarray(scan_item.scan.x, dtype=float)
                y_values = np.asarray(scan_item.scan.y, dtype=float)
                x_span = float(np.ptp(x_values)) if x_values.size else 0.0
                y_span = float(np.ptp(y_values)) if y_values.size else 0.0
                y_size = float(np.max(np.abs(y_values))) if y_values.size else 0.0
                self._processing_x_limit = max(
                    1.0, 0.1 * x_span, 1.2 * abs(scan_item.x_shift)
                )
                self._processing_y_limit = max(
                    1.0,
                    y_span,
                    0.25 * y_size,
                    1.2 * abs(scan_item.y_shift),
                )
            self.processing_x_spin.setRange(
                -self._processing_x_limit, self._processing_x_limit
            )
            self.processing_y_spin.setRange(
                -self._processing_y_limit, self._processing_y_limit
            )
            self.processing_x_spin.setValue(values[0])
            self.processing_y_spin.setValue(values[1])
            self.processing_factor_spin.setValue(values[2])
            self.processing_x_slider.setValue(
                self._processing_slider_position(
                    "x", values[0], self._processing_x_limit
                )
            )
            self.processing_y_slider.setValue(
                self._processing_slider_position(
                    "y", values[1], self._processing_y_limit
                )
            )
            self.processing_factor_slider.setValue(
                self._processing_slider_position("factor", values[2], 1.0)
            )
            self.shift_omega_check.setChecked(
                scan_item.shift_omega if scan_item is not None else True
            )
        finally:
            for control in controls_to_block:
                control.blockSignals(False)
            self._syncing_processing = False

        enabled = scan_item is not None
        for control in self.processing_controls:
            control.setEnabled(enabled)
        self.fit_button.setEnabled(bool(scan_item is not None and scan_item.visible))

    def _processing_slider_changed(self, kind: str, position: int) -> None:
        if self._syncing_processing:
            return
        spin = {
            "x": self.processing_x_spin,
            "y": self.processing_y_spin,
            "factor": self.processing_factor_spin,
        }[kind]
        limit = {
            "x": self._processing_x_limit,
            "y": self._processing_y_limit,
            "factor": 1.0,
        }[kind]
        spin.blockSignals(True)
        spin.setValue(self._processing_slider_value(kind, position, limit))
        spin.blockSignals(False)
        self._processing_values_changed()

    def _processing_spin_changed(self, kind: str, value: float) -> None:
        if self._syncing_processing:
            return
        slider = {
            "x": self.processing_x_slider,
            "y": self.processing_y_slider,
            "factor": self.processing_factor_slider,
        }[kind]
        limit = {
            "x": self._processing_x_limit,
            "y": self._processing_y_limit,
            "factor": 1.0,
        }[kind]
        slider.blockSignals(True)
        slider.setValue(self._processing_slider_position(kind, value, limit))
        slider.blockSignals(False)
        self._processing_values_changed()

    def _processing_values_changed(self, _checked: bool | None = None) -> None:
        if self._syncing_processing:
            return
        item = self._selected_item()
        if item is None or item.scan is None:
            return
        try:
            request = CorrectionRequest(
                x_shift=self.processing_x_spin.value(),
                y_shift=self.processing_y_spin.value(),
                y_factor=self.processing_factor_spin.value(),
                shift_omega_half=self.shift_omega_check.isChecked(),
            )
        except (TypeError, ValueError, ArithmeticError, XRDDataError):
            self.status.setText(
                localised(
                    "X and Y shifts must be finite numbers; the Y scale must be positive.",
                    "Les décalages X et Y doivent être finis ; l’échelle Y doit être positive.",
                    "Сдвиги X и Y должны быть конечными числами, а масштаб Y — положительным.",
                )
            )
            self._load_processing_controls(item)
            return
        item.x_shift = request.x_shift
        item.y_shift = request.y_shift
        item.y_factor = request.y_factor
        item.shift_omega = request.shift_omega_half
        self._draw(preserve_view=True)

    def reset_processing(self) -> None:
        item = self._selected_item()
        if item is None or item.scan is None:
            return
        item.reset_transform()
        self._load_processing_controls(item)
        self._draw(preserve_view=True)

    def build_processed_scan(self, uid: str | None = None):
        uid = uid or self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.scan is None:
            return None
        return apply_correction(
            item.scan,
            CorrectionRequest(
                x_shift=item.x_shift,
                y_shift=item.y_shift,
                y_factor=item.y_factor,
                shift_omega_half=item.shift_omega,
            ),
        )

    def apply_processing_result(self) -> None:
        mode = "replace" if self.replace_result_radio.isChecked() else "add"
        validate_result_mode(mode)
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        scan = self.build_processed_scan(uid)
        if uid is None or item is None or scan is None:
            return
        old_x = self.scan_axis.get_xlim()
        old_y = self.scan_axis.get_ylim()
        if mode == "add":
            document = self.store.add_scan(scan, derived=True, parent_uid=uid)
            self.store.assign(document.uid, VIEWER, True)
        else:
            self.store.replace_scan(uid, scan)
            item.name = scan.name
            item.source = Path(scan.source)
            item.scan = clone_scan(scan)
        item.reset_transform()
        self._selected_point = None
        self._refresh_tree()
        self._draw(preserve_view=True)
        if all(np.isfinite(old_x)) and all(np.isfinite(old_y)):
            if old_x[0] < old_x[1] and old_y[0] < old_y[1]:
                self._set_limits(old_x, old_y)

    def _choose_processing_save_path(self, scan) -> Path | None:
        source = Path(scan.source)
        is_xrdml = source.suffix.lower() in {".xrdml", ".xml"}
        extension = source.suffix if is_xrdml else ".xy"
        suggested = source.with_name(f"{source.stem} shifted{extension}")
        file_filter = (
            "XRDML (*.xrdml *.xml);;XY (*.xy);;All files (*)"
            if is_xrdml
            else "XY (*.xy);;All files (*)"
        )
        path_text, _selected_filter = QFileDialog.getSaveFileName(
            self,
            localised(
                "Save shifted data",
                "Enregistrer les données décalées",
                "Сохранить сдвинутые данные",
            ),
            str(suggested),
            file_filter,
        )
        if not path_text:
            return None
        path = Path(path_text)
        if not path.suffix:
            path = path.with_suffix(extension)
        return path

    @staticmethod
    def _processing_export_error(exc: Exception) -> str:
        if not isinstance(exc, XRDDataError):
            return str(exc)
        messages = {
            "correction_xrdml_source": localised(
                "XRDML export requires an XRDML source file.",
                "L’export XRDML nécessite un fichier source XRDML.",
                "Для экспорта XRDML нужен исходный файл XRDML.",
            ),
            "correction_xrdml_range": localised(
                "The selected XRDML range was not found in the source file.",
                "La plage XRDML sélectionnée est introuvable dans le fichier source.",
                "Выбранный диапазон XRDML не найден в исходном файле.",
            ),
            "correction_xrdml_intensity": localised(
                "The XRDML intensity array was not found.",
                "Le tableau d’intensité XRDML est introuvable.",
                "В XRDML не найден массив интенсивностей.",
            ),
            "correction_xrdml_array_length": localised(
                "The processed and source XRDML arrays have different lengths.",
                "Les tableaux XRDML traité et source ont des longueurs différentes.",
                "Массивы обработанного и исходного XRDML имеют разную длину.",
            ),
            "correction_xrdml_axis_length": localised(
                "An XRDML coordinate axis has an unexpected length.",
                "Un axe de coordonnées XRDML a une longueur inattendue.",
                "Одна из координатных осей XRDML имеет неверную длину.",
            ),
        }
        return messages.get(exc.code, str(exc))

    def save_processing_result(self) -> None:
        scan = self.build_processed_scan()
        if scan is None:
            return
        path = self._choose_processing_save_path(scan)
        if path is None:
            return
        try:
            same_as_source = path.resolve() == Path(scan.source).resolve()
        except OSError:
            same_as_source = False
        if same_as_source:
            answer = QMessageBox.question(
                self,
                localised(
                    "Overwrite source",
                    "Écraser la source",
                    "Перезапись исходника",
                ),
                localised(
                    "This is the original measurement file. Overwrite it explicitly?",
                    "Il s’agit du fichier de mesure d’origine. Voulez-vous vraiment l’écraser ?",
                    "Это исходный файл измерения. Действительно перезаписать его?",
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            write_processed_scan(scan, path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                localised(
                    "Save error",
                    "Erreur d’enregistrement",
                    "Ошибка сохранения",
                ),
                self._processing_export_error(exc),
            )
            return
        self.status.setText(tr("qt.correction_saved", name=path.name))

    def _cancel_peak_fit(self) -> None:
        if self._fit_selector is not None:
            try:
                self._fit_selector.set_active(False)
                self._fit_selector.disconnect_events()
            except Exception:
                pass
        self._fit_selector = None
        self._fit_uid = None

    def activate_peak_fit(self) -> None:
        item = self._selected_item()
        uid = self._selected_uid()
        if item is None or item.scan is None or uid is None:
            return
        if self.toolbar.mode:
            self.status.setText(
                localised(
                    "Turn off Pan or Zoom before selecting a peak.",
                    "Désactivez le déplacement ou le zoom avant de sélectionner un pic.",
                    "Отключите перемещение или масштабирование перед выбором пика.",
                )
            )
            return
        self._cancel_peak_fit()
        self._fit_uid = uid
        self._fit_selector = RectangleSelector(
            self.scan_axis,
            self._fit_peak_rectangle,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords="data",
            interactive=False,
        )
        self.status.setText(
            localised(
                "Drag a rectangle around one peak on the main plot.",
                "Tracez un rectangle autour d’un pic sur le graphique principal.",
                "Выделите прямоугольником один пик на основном графике.",
            )
        )

    def _peak_fit_error(self, exc: Exception) -> str:
        if isinstance(exc, XRDDataError):
            messages = {
                "peak_fit_scipy": localised(
                    "SciPy is required for peak fitting.",
                    "SciPy est requis pour l’ajustement du pic.",
                    "Для аппроксимации пика требуется SciPy.",
                ),
                "peak_fit_points": localised(
                    "Select at least seven data points around the peak.",
                    "Sélectionnez au moins sept points autour du pic.",
                    "Выберите вокруг пика не менее семи точек.",
                ),
                "peak_fit_flat": localised(
                    "The selected region contains no measurable peak.",
                    "La zone sélectionnée ne contient aucun pic mesurable.",
                    "В выбранной области нет измеримого пика.",
                ),
            }
            if exc.code in messages:
                return messages[exc.code]
        return str(exc)

    def _fit_peak_rectangle(self, click, release) -> None:
        uid = self._fit_uid
        item = self.items.get(uid) if uid else None
        self._cancel_peak_fit()
        if (
            item is None
            or item.scan is None
            or click.xdata is None
            or click.ydata is None
            or release.xdata is None
            or release.ydata is None
        ):
            return
        x_low, x_high = sorted((float(click.xdata), float(release.xdata)))
        y_low, y_high = sorted((float(click.ydata), float(release.ydata)))
        x_values, physical_y = item.display_arrays()
        plotted_y = transformed_intensity(
            physical_y, self.viewer_state.plot.intensity_scale
        )
        mask = (
            np.isfinite(x_values)
            & np.isfinite(plotted_y)
            & (x_values >= x_low)
            & (x_values <= x_high)
            & (plotted_y >= y_low)
            & (plotted_y <= y_high)
        )
        old_x = self.scan_axis.get_xlim()
        old_y = self.scan_axis.get_ylim()
        try:
            fit_x, fit_y, centre, intensity = fit_gaussian_peak(
                x_values[mask], physical_y[mask]
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                localised(
                    "Peak fitting",
                    "Ajustement du pic",
                    "Аппроксимация пика",
                ),
                self._peak_fit_error(exc),
            )
            return

        self.scan_axis.scatter(x_values[mask], plotted_y[mask], color="green", s=18, zorder=7)
        self.scan_axis.plot(
            fit_x,
            transformed_intensity(fit_y, self.viewer_state.plot.intensity_scale),
            color="orange",
            linewidth=2,
            zorder=7,
        )
        self.scan_axis.axvline(centre, color="red", linestyle="--", linewidth=1)
        centre_y = transformed_intensity(
            np.asarray([intensity]), self.viewer_state.plot.intensity_scale
        )[0]
        self.scan_axis.scatter([centre], [centre_y], color="red", s=55, zorder=8)
        self.scan_axis.set_xlim(old_x)
        self.scan_axis.set_ylim(old_y)
        self.canvas.draw_idle()

        references = read_reference_peaks(_reference_peak_path())
        dialog = PeakTargetDialog(centre, intensity, references, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            target = float(dialog.target_text().strip().replace(",", "."))
            if not math.isfinite(target):
                raise ValueError
        except ValueError:
            QMessageBox.critical(
                self,
                localised(
                    "Invalid value",
                    "Valeur incorrecte",
                    "Некорректное значение",
                ),
                localised(
                    "Enter a finite reference coordinate.",
                    "Saisissez une coordonnée de référence finie.",
                    "Введите конечную координату опорного пика.",
                ),
            )
            return
        item.x_shift += target - centre
        self._load_processing_controls(item)
        self._draw(preserve_view=True)
        self.status.setText(
            tr(
                "qt.correction_peak_aligned",
                centre=f"{centre:.6f}",
                target=f"{target:.6f}",
                shift=f"{item.x_shift:+.6f}",
            )
        )

    def _phase_layout_changed(self, _index: int) -> None:
        code = self.phase_layout_combo.currentData()
        if not code:
            return
        if code == "overlay" and not self.viewer_state.cif_axes_compatible():
            code = "separate"
            self.phase_layout_combo.blockSignals(True)
            self.phase_layout_combo.setCurrentIndex(
                self.phase_layout_combo.findData("separate")
            )
            self.phase_layout_combo.blockSignals(False)
            self.status.setText(
                tr("text.cif_overlay_is_available_only_for_measurements_on_the_two_theta_axis")
            )
        self.viewer_state.plot.phase_layout = str(code)
        self._update_phase_controls()
        self._draw(preserve_view=True)

    def _phase_style_changed(self, _index: int) -> None:
        code = self.phase_style_combo.currentData()
        if code:
            self.viewer_state.plot.phase_style = str(code)
            self._draw(preserve_view=True)

    def _phase_height_changed(self, value: int) -> None:
        height = self.viewer_state.plot.set_phase_height(float(value))
        self.phase_height_value.setText(f"{height:.0f}%")
        self._draw(preserve_view=True)

    def _overlay_arrangement_changed(self, checked: bool) -> None:
        self.viewer_state.plot.overlay_single_line = bool(checked)
        self._update_phase_controls()
        self._draw(preserve_view=True)

    def _overlay_height_changed(self, value: int) -> None:
        height = self.viewer_state.plot.set_overlay_height(float(value))
        self.overlay_height_value.setText(f"{height:.0f}%")
        self._draw(preserve_view=True)

    def _radiation_changed(self) -> None:
        if self._selected_point and self._selected_point[1] == "phase":
            self._selected_point = None
        self._row_cache.clear()
        self._draw(preserve_view=True)
        self._refresh_selection_info()

    def _update_phase_controls(self) -> None:
        if not hasattr(self, "phase_section"):
            return
        has_phases = any(item.is_phase for item in self.items.values())
        self.phase_section.setVisible(has_phases)
        compatible = self.viewer_state.cif_axes_compatible()
        if not compatible and self.viewer_state.plot.phase_layout == "overlay":
            self.viewer_state.plot.phase_layout = "separate"
            self.phase_layout_combo.blockSignals(True)
            index = self.phase_layout_combo.findData("separate")
            if index >= 0:
                self.phase_layout_combo.setCurrentIndex(index)
            self.phase_layout_combo.blockSignals(False)
        layout = self.viewer_state.plot.phase_layout
        separate = layout == "separate"
        self.phase_height_slider.setEnabled(separate)
        self.overlay_single_check.setEnabled(not separate)
        self.overlay_height_slider.setEnabled(
            not separate and self.viewer_state.plot.overlay_single_line
        )
        self.phase_height_value.setText(
            f"{self.viewer_state.plot.phase_height_percent:.0f}%"
        )
        self.overlay_height_value.setText(
            f"{self.viewer_state.plot.overlay_height_percent:.0f}%"
        )

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

    def remove_selected(self) -> None:
        """Disconnect the selected project object from Viewer."""

        uid = self._selected_uid()
        if uid is not None and uid in self.store.documents:
            self.store.assign(uid, VIEWER, False)

    def show_all(self) -> None:
        self.viewer_state.show_all()
        self._refresh_tree()
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def rename_selected(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            tr("text.rename_dataset"),
            tr("text.new_name"),
            text=item.name,
        )
        if accepted and name.strip():
            self.store.rename(uid, name)

    def clear_all(self) -> None:
        """Disconnect every currently loaded object from Viewer."""

        for uid in tuple(self.items):
            if uid in self.store.documents:
                self.store.assign(uid, VIEWER, False)
        self.status.setText(
            localised(
                "The viewer has been cleared.",
                "La visualisation a été effacée.",
                "Просмотрщик очищен.",
            )
        )

    def select_uid(self, uid: str, *, open_processing: bool = False) -> None:
        """Select one loaded object and optionally reveal its correction block."""

        for index in range(self.documents.topLevelItemCount()):
            row = self.documents.topLevelItem(index)
            if row.data(0, Qt.ItemDataRole.UserRole) != uid:
                continue
            self.documents.setCurrentItem(row)
            self.documents.scrollToItem(row)
            if open_processing:
                item = self.items.get(uid)
                if item is not None and item.scan is not None:
                    self.processing_section.set_expanded(True)
                    self.controls_scroll.ensureWidgetVisible(self.processing_section)
            return

    def open_selected_reflection_table(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if (
            uid is not None
            and item is not None
            and item.is_phase
            and self.on_open_reflection_table is not None
        ):
            self.on_open_reflection_table(uid)

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

    def _phase_limits(self, scans: list[PlotItem]) -> tuple[float, float]:
        if scans and all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in scans
        ):
            return scan_x_limits(scans)
        return DEFAULT_PHASE_LIMITS

    def _ensure_factors(self):
        if self._factors is None:
            self._factors = read_scattering_factors(scattering_factor_path())
        return self._factors

    def _rows_for(
        self,
        item: PlotItem,
        limits: tuple[float, float],
    ) -> list[ReflectionRow]:
        if item.structure is None:
            return []
        radiations = tuple(self.radiation_settings.lines())
        key = (float(limits[0]), float(limits[1]), radiations)
        cached = self._row_cache.get(item.uid)
        if cached is not None and cached[0] == key:
            return cached[1]
        factors = (
            {}
            if bool(getattr(item.structure, "cell_only", False))
            else self._ensure_factors()
        )
        rows = calculate_reflections(
            item.structure,
            factors,
            radiations,
            min_two_theta=max(0.0, float(limits[0])),
            max_two_theta=min(179.9, float(limits[1])),
            min_intensity=0.1,
        )
        self._row_cache[item.uid] = (key, rows)
        return rows

    def _phase_profile(
        self,
        rows: list[ReflectionRow],
        limits: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        minimum, maximum = limits
        count = max(1200, min(100000, int((maximum - minimum) * 30)))
        profile = gaussian_powder_profile(
            rows,
            minimum,
            maximum,
            self.fwhm_spin.value(),
            point_count=count,
            normalize_to=1.0,
            missing_intensity=100.0,
        )
        return profile.x, profile.total

    def _draw_overlay_phases(
        self,
        phases: list[PlotItem],
        limits: tuple[float, float],
    ) -> None:
        bands, occupied_height = overlay_phase_geometry(
            len(phases),
            single_line=self.viewer_state.plot.overlay_single_line,
            height_percent=self.viewer_state.plot.overlay_height_percent,
        )
        transform = self.scan_axis.get_xaxis_transform()
        for index, (item, geometry) in enumerate(zip(phases, bands)):
            rows = self._rows_for(item, limits)
            baseline, amplitude = geometry
            profile_allowed = not bool(getattr(item.structure, "cell_only", False))
            if self.viewer_state.plot.phase_style == "profile" and profile_allowed:
                grid, profile = self._phase_profile(rows, limits)
                profile = profile * amplitude
                self.scan_axis.plot(
                    grid,
                    baseline + profile,
                    color=item.colour,
                    lw=1.0,
                    transform=transform,
                    clip_on=True,
                )
                self.scan_axis.fill_between(
                    grid,
                    baseline,
                    baseline + profile,
                    color=item.colour,
                    alpha=0.16,
                    transform=transform,
                    clip_on=True,
                )
            else:
                for row in rows:
                    height = (
                        amplitude
                        if row.intensity is None
                        else amplitude * (0.18 + 0.82 * row.intensity / 100.0)
                    )
                    self.scan_axis.vlines(
                        row.two_theta,
                        baseline,
                        baseline + height,
                        color=item.colour,
                        lw=1.0,
                        transform=transform,
                        clip_on=True,
                    )
            if self.viewer_state.plot.overlay_single_line:
                label_x = (index + 0.5) / len(phases)
                label_y = min(0.99, occupied_height + 0.01)
                vertical_alignment = "top" if occupied_height > 0.9 else "bottom"
                horizontal_alignment = "center"
            else:
                band_height = amplitude / 0.75
                label_x = 0.01
                label_y = baseline + band_height * 0.9
                vertical_alignment = "top"
                horizontal_alignment = "left"
            self.scan_axis.text(
                label_x,
                label_y,
                item.name,
                color=item.colour,
                fontsize=8,
                ha=horizontal_alignment,
                va=vertical_alignment,
                transform=self.scan_axis.transAxes,
                clip_on=True,
            )
        self._overlay_phase_top = occupied_height

    def _draw_separate_phases(
        self,
        phases: list[PlotItem],
        limits: tuple[float, float],
    ) -> None:
        self.phase_axis.set_xlabel("2θ, °")
        self.phase_axis.set_ylabel(tr("text.phases"))
        self.phase_axis.grid(True, axis="x", alpha=0.15)
        self.phase_axis.set_yticks([])
        for index, item in enumerate(phases):
            rows = self._rows_for(item, limits)
            baseline = float(len(phases) - index - 1)
            profile_allowed = not bool(getattr(item.structure, "cell_only", False))
            if self.viewer_state.plot.phase_style == "profile" and profile_allowed:
                grid, profile = self._phase_profile(rows, limits)
                profile = profile * 0.75
                self.phase_axis.plot(grid, baseline + profile, color=item.colour, lw=1.0)
                self.phase_axis.fill_between(
                    grid,
                    baseline,
                    baseline + profile,
                    color=item.colour,
                    alpha=0.16,
                )
            else:
                for row in rows:
                    height = (
                        0.85
                        if row.intensity is None
                        else 0.15 + 0.7 * row.intensity / 100.0
                    )
                    self.phase_axis.vlines(
                        row.two_theta,
                        baseline,
                        baseline + height,
                        color=item.colour,
                        lw=1.0,
                    )
            self.phase_axis.text(
                0.01,
                baseline + 0.87,
                item.name,
                color=item.colour,
                fontsize=8,
                ha="left",
                va="top",
                transform=self.phase_axis.get_yaxis_transform(),
                clip_on=True,
            )
        self.phase_axis.set_xlim(*limits)
        self.phase_axis.set_ylim(-0.1, len(phases) + 0.05)

    def _prepare_phase_axes(self, separate: bool) -> None:
        self.phase_axis.set_visible(separate)
        self.phase_axis.set_in_layout(separate)
        self.phase_axis.set_navigate(separate)
        self.split_axis.set_visible(separate)
        self.split_axis.set_in_layout(separate)
        self.split_axis.set_navigate(False)
        self.scan_axis.set_navigate(True)
        self.scan_axis.set_subplotspec(
            self.plot_grid[0, 0] if separate else self.plot_grid[:, 0]
        )
        if separate:
            fraction = self.viewer_state.plot.phase_height_percent / 100.0
            self.plot_grid.set_height_ratios((1.0 - fraction, 0.025, fraction))
            self.split_axis.axhline(0.5, color="#b8b8b8", lw=2.0)
            self.split_axis.set_axis_off()

    def _status_text(self, scan_count: int, phase_count: int) -> str:
        if scan_count or phase_count:
            return tr(
                "qt.viewer_visible_objects",
                scans=scan_count,
                phases=phase_count,
            )
        return (
            tr("qt.viewer_no_visible_measurements")
            if self.items
            else tr("qt.viewer_no_measurements")
        )

    def _draw(self, *, preserve_view: bool, preserve_y: bool = True) -> None:
        if not hasattr(self, "scan_axis") or self._drawing:
            return
        self._drawing = True
        old_x = self.scan_axis.get_xlim()
        old_y = self.scan_axis.get_ylim()
        old_phase_x = self.phase_axis.get_xlim()
        phase_was_visible = self.phase_axis.get_visible()
        had_data = self._has_drawn_data
        mode = self.viewer_state.plot.intensity_scale
        arrays = self._visible_plot_arrays()
        scans = [item for item, _x, _y in arrays]
        phases = self.viewer_state.visible_phases()
        self._axes_linked = bool(scans) and all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in scans
        )
        if phases and not self._axes_linked and self.viewer_state.plot.phase_layout == "overlay":
            self.viewer_state.plot.phase_layout = "separate"
            self.phase_layout_combo.blockSignals(True)
            index = self.phase_layout_combo.findData("separate")
            if index >= 0:
                self.phase_layout_combo.setCurrentIndex(index)
            self.phase_layout_combo.blockSignals(False)
        separate = bool(phases) and self.viewer_state.plot.phase_layout == "separate"
        try:
            self.scan_axis.clear()
            self.phase_axis.clear()
            self.split_axis.clear()
            self._prepare_phase_axes(separate)
            self.scan_axis.set_yscale("log" if mode == "log" else "linear")
            for item, x_values, plotted_y in arrays:
                self.scan_axis.plot(
                    x_values,
                    plotted_y,
                    color=item.colour,
                    lw=1.15,
                    label=item.name,
                )

            if arrays:
                self._navigation_x_bounds = scan_x_limits(scans)
                self._navigation_y_bounds = intensity_limits(
                    [values for _item, _x, values in arrays],
                    logarithmic=mode == "log",
                )
                if len(arrays) <= MAX_LEGEND_ITEMS:
                    self.scan_axis.legend(loc="best", fontsize=8)
            else:
                self._navigation_x_bounds = (0.0, 1.0)
                self._navigation_y_bounds = (0.0, 1.0)
                if not phases:
                    self.scan_axis.text(
                        0.5,
                        0.5,
                        self._status_text(0, 0),
                        ha="center",
                        va="center",
                        transform=self.scan_axis.transAxes,
                        color="#666666",
                    )

            phase_limits = self._phase_limits(scans) if phases else DEFAULT_PHASE_LIMITS
            if phases and not arrays:
                self._navigation_x_bounds = phase_limits
            self._overlay_phase_top = 0.0
            phase_error = ""
            if phases:
                try:
                    if separate:
                        self._draw_separate_phases(phases, phase_limits)
                    else:
                        self._draw_overlay_phases(phases, phase_limits)
                except (OSError, ValueError, XRDDataError) as exc:
                    phase_error = str(exc)

            self.status.setText(
                phase_error or self._status_text(len(arrays), len(phases))
            )
            self._has_drawn_data = bool(arrays or phases)

            visible_axes = {
                item.scan.axis_name
                for item, _x, _y in arrays
                if item.scan is not None
            }
            x_label = (
                _axis_label(next(iter(visible_axes)))
                if len(visible_axes) == 1
                else tr("text.scan_coordinate")
            )
            if len(visible_axes) == 1 and axis_has_degree_units(next(iter(visible_axes))):
                x_label += ", °"
            self.scan_axis.set_xlabel(x_label)
            self.scan_axis.set_ylabel(tr("qt.viewer_intensity"))
            self.scan_axis.grid(True, alpha=0.22)

            x_limits = self._navigation_x_bounds
            y_limits = self._navigation_y_bounds
            scan_axis_has_content = bool(arrays or (phases and not separate))
            if preserve_view and had_data and scan_axis_has_content:
                previous_x = (
                    old_phase_x
                    if phase_was_visible and not arrays and not separate
                    else old_x
                )
                if all(np.isfinite(previous_x)) and previous_x[0] < previous_x[1]:
                    x_limits = previous_x
                if (
                    arrays
                    and preserve_y
                    and all(np.isfinite(old_y))
                    and old_y[0] < old_y[1]
                ):
                    if mode != "log" or old_y[0] > 0:
                        y_limits = old_y
            self.scan_axis.set_xlim(*x_limits)
            self.scan_axis.set_ylim(*y_limits)
            if separate:
                if self._axes_linked:
                    self.phase_axis.set_xlim(*x_limits)
                elif preserve_view and had_data:
                    previous_phase_x = old_phase_x if phase_was_visible else old_x
                    if (
                        all(np.isfinite(previous_phase_x))
                        and previous_phase_x[0] < previous_phase_x[1]
                    ):
                        self.phase_axis.set_xlim(*previous_phase_x)
            # Axes.clear() replaces Matplotlib's callback registry, so these
            # connections must be restored after every redraw.
            self.scan_axis.callbacks.connect("xlim_changed", self._x_limits_changed)
            self.scan_axis.callbacks.connect(
                "ylim_changed", lambda _axis: self._update_navigation_scrollbars()
            )
            self.phase_axis.callbacks.connect("xlim_changed", self._x_limits_changed)
            if not preserve_view:
                self.toolbar.update()
            self._update_phase_controls()
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
            self.scan_axis.set_xlim(*x_limits)
            self.scan_axis.set_ylim(*y_limits)
            if self._axes_linked and self.phase_axis.get_visible():
                self.phase_axis.set_xlim(*x_limits)
            self._update_navigation_scrollbars()
            self.canvas.draw_idle()
        finally:
            self._drawing = False

    def _x_limits_changed(self, source_axis) -> None:
        if self._drawing or self._syncing_x_limits:
            return
        self._syncing_x_limits = True
        try:
            if self._axes_linked and self.phase_axis.get_visible():
                if source_axis is self.scan_axis:
                    self.phase_axis.set_xlim(*self.scan_axis.get_xlim())
                elif source_axis is self.phase_axis:
                    self.scan_axis.set_xlim(*self.phase_axis.get_xlim())
            self._update_navigation_scrollbars()
            self.canvas.draw_idle()
        finally:
            self._syncing_x_limits = False

    def _x_navigation_axis(self):
        if self.phase_axis.get_visible() and not self.viewer_state.visible_scans():
            return self.phase_axis
        return self.scan_axis

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
        if self._syncing_scrollbars or not hasattr(self, "scan_axis"):
            return
        self._syncing_scrollbars = True
        try:
            x_axis = self._x_navigation_axis()
            self._configure_scrollbar(
                self.x_scrollbar,
                self._navigation_x_bounds,
                x_axis.get_xlim(),
                vertical=False,
            )
            self._configure_scrollbar(
                self.y_scrollbar,
                self._navigation_y_bounds,
                self.scan_axis.get_ylim(),
                vertical=True,
            )
        finally:
            self._syncing_scrollbars = False

    def _scrollbar_changed(self, dimension: str, value: int) -> None:
        if self._syncing_scrollbars or self._drawing:
            return
        bounds = self._navigation_x_bounds if dimension == "x" else self._navigation_y_bounds
        target_axis = self._x_navigation_axis() if dimension == "x" else self.scan_axis
        current = target_axis.get_xlim() if dimension == "x" else target_axis.get_ylim()
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
                target_axis.set_xlim(low, high)
                if self._axes_linked and self.phase_axis.get_visible():
                    other_axis = (
                        self.phase_axis
                        if target_axis is self.scan_axis
                        else self.scan_axis
                    )
                    other_axis.set_xlim(low, high)
            else:
                target_axis.set_ylim(low, high)
            self.canvas.draw_idle()
        finally:
            self._drawing = False
        self._update_navigation_scrollbars()

    def _plot_scrolled(self, event) -> None:
        axis = event.inaxes
        if (
            axis not in {self.scan_axis, self.phase_axis}
            or not axis.get_visible()
            or event.xdata is None
            or event.ydata is None
        ):
            return
        factor = 0.80 if event.button == "up" else 1.25
        x_low, x_high = axis.get_xlim()
        if math.isclose(x_low, x_high):
            return
        x_span = (x_high - x_low) * factor
        x_fraction = (event.xdata - x_low) / (x_high - x_low)
        new_x = (
            event.xdata - x_fraction * x_span,
            event.xdata + (1.0 - x_fraction) * x_span,
        )

        y_low, y_high = axis.get_ylim()
        if math.isclose(y_low, y_high):
            return
        logarithmic = (
            axis is self.scan_axis
            and self.viewer_state.plot.intensity_scale == "log"
        )
        if logarithmic and y_low > 0 and event.ydata > 0:
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
        if axis is self.scan_axis:
            self._set_limits(new_x, new_y)
            return
        self._drawing = True
        try:
            self.phase_axis.set_xlim(*new_x)
            self.phase_axis.set_ylim(*new_y)
            if self._axes_linked:
                self.scan_axis.set_xlim(*new_x)
            self.canvas.draw_idle()
        finally:
            self._drawing = False
        self._update_navigation_scrollbars()

    def _plot_clicked(self, event) -> None:
        if (
            event.button != 1
            or event.inaxes not in {self.scan_axis, self.phase_axis}
            or not event.inaxes.get_visible()
            or event.xdata is None
            or event.ydata is None
            or self.toolbar.mode
        ):
            return
        if event.inaxes is self.phase_axis:
            self._select_nearest_phase_reflection(event)
            return
        if self.viewer_state.visible_phases() and self.viewer_state.plot.phase_layout == "overlay":
            axes_y = self.scan_axis.transAxes.inverted().transform(
                (event.x, event.y)
            )[1]
            if axes_y <= self._overlay_phase_top + 0.025:
                if self._select_nearest_phase_reflection(event):
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
                px, py = self.scan_axis.transData.transform(
                    (x_values[index], plotted_y[index])
                )
                distance = math.hypot(px - event.x, py - event.y)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, item, index)
        if nearest is None:
            return
        _distance, item, index = nearest
        self._selected_point = (item.uid, "scan", index)
        self._refresh_selection_info()

    @staticmethod
    def _full_hkl(row: ReflectionRow) -> str:
        if not row.equivalents:
            return row.hkl
        return "+".join(f"({h} {k} {l})" for h, k, l in row.equivalents)

    def _select_nearest_phase_reflection(self, event) -> bool:
        phases = self.viewer_state.visible_phases()
        if not phases:
            return False
        try:
            limits = self._phase_limits(self.viewer_state.visible_scans())
        except (TypeError, ValueError, ArithmeticError):
            return False
        nearest: tuple[float, PlotItem, ReflectionRow] | None = None
        if event.inaxes is self.phase_axis:
            for phase_index, item in enumerate(phases):
                baseline = float(len(phases) - phase_index - 1)
                for row in self._rows_for(item, limits):
                    px, py = self.phase_axis.transData.transform(
                        (row.two_theta, baseline + 0.45)
                    )
                    distance = math.hypot(px - event.x, py - event.y)
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, item, row)
        else:
            bands, _occupied = overlay_phase_geometry(
                len(phases),
                single_line=self.viewer_state.plot.overlay_single_line,
                height_percent=self.viewer_state.plot.overlay_height_percent,
            )
            transform = self.scan_axis.get_xaxis_transform()
            for item, (baseline, amplitude) in zip(phases, bands):
                for row in self._rows_for(item, limits):
                    px, py = transform.transform(
                        (row.two_theta, baseline + 0.5 * amplitude)
                    )
                    distance = math.hypot(px - event.x, py - event.y)
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, item, row)
        if nearest is None:
            return False
        _distance, item, row = nearest
        self._selected_point = (item.uid, "phase", row)
        self._refresh_selection_info()
        return True

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
        if item is None or not item.visible or selection is None:
            self._selected_point = None
            self.selection_info.setText(tr("text.select_a_point_or_reflection_on_the_plot"))
            return
        kind = selection[1]
        if kind == "phase":
            row = selection[2]
            if item.structure is None or not isinstance(row, ReflectionRow):
                self._selected_point = None
                self.selection_info.setText(
                    tr("text.select_a_point_or_reflection_on_the_plot")
                )
                return
            intensity = "—" if row.intensity is None else f"{row.intensity:.2f}%"
            self.selection_info.setText(
                tr(
                    "qt.viewer_phase_selection",
                    name=item.name,
                    hkl=self._full_hkl(row),
                    d=f"{row.d:.5f}",
                    two_theta=f"{row.two_theta:.5f}",
                    intensity=intensity,
                )
            )
            return
        if kind != "scan" or item.scan is None:
            self._selected_point = None
            self.selection_info.setText(tr("text.select_a_point_or_reflection_on_the_plot"))
            return
        index = int(selection[2])
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
