"""PySide6 viewer for one-dimensional measurements and calculated phases."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, QThreadPool, Qt, Signal
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
    QStackedWidget,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from .plot_toolbar import PlotToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from ..localization import localised, tr
from ..models.data_errors import XRDDataError
from ..io.correction import write_processed_scan
from ..io.reflections import read_scattering_factors, scattering_factor_path
from ..models.diffraction import ReflectionRow
from ..models.project import CELL_PHASE, CIF, SCAN, VIEWER
from ..models.correction import CorrectionRequest, validate_result_mode
from ..models.radiation import PRESETS, RadiationSettings
from ..models.scan import clone_scan
from ..models.viewer import (
    PlotItem,
    ViewerState,
    axis_has_degree_units,
    axis_key,
    d_to_two_theta,
    intensity_limits,
    is_two_theta,
    overlay_phase_geometry,
    resolve_limits,
    scrolled_limits,
    scrollbar_window,
    transformed_intensity,
    two_theta_to_d,
)
from ..services.diffraction import (
    calculate_reflections,
    d_spacing_from_two_theta,
    gaussian_powder_profile,
)
from ..services.correction import apply_correction
from ..services.peak_fitting import (
    VoigtRegionFit, companion_two_theta, draw_voigt_peak, drawn_voigt_profile,
    fit_gaussian_peak, fit_voigt_region, refit_voigt_peaks,
)
from ..services.background import BackgroundAnchors, estimate_background
from ..services.reference_peaks import read_reference_peaks, reference_peak_path
from ..services.substrate_calibration import (
    SubstrateCalibration,
    fit_substrate_calibration,
)
from .radiation import RadiationSelector
from .peak_table import (
    PEAK_COLOURS, CompanionProposal, CompanionReviewDialog, PeakTableDialog,
    SessionPeak,
)


MAX_LEGEND_ITEMS = 12
SCROLL_RESOLUTION = 10_000
DEFAULT_PHASE_LIMITS = (5.0, 120.0)
DISABLED_PLACEHOLDER_STYLE = "QPushButton:disabled { color: #c62828; }"


@dataclass(frozen=True)
class PendingPeakFit:
    uid: str
    axis_name: str
    result: VoigtRegionFit
    selected_source_x: np.ndarray
    selected_source_y: np.ndarray
    source_x: np.ndarray
    source_background: np.ndarray
    source_fitted: np.ndarray
    source_centers: tuple[float, ...]
    fit_x_scale: float
    fit_y_factor: float
    background_after: BackgroundAnchors | None = None
    manual: bool = False
    replace_number: int | None = None


class _PhaseReflectionSignals(QObject):
    finished = Signal(object, object, object, object)


class _PhaseReflectionTask(QRunnable):
    """Calculate one phase without blocking the Qt event loop."""

    def __init__(
        self,
        uid,
        key,
        structure,
        factors,
        radiations,
        limits,
    ) -> None:
        super().__init__()
        self.uid = uid
        self.key = key
        self.structure = structure
        self.factors = factors
        self.radiations = radiations
        self.limits = limits
        self.signals = _PhaseReflectionSignals()

    def run(self) -> None:
        try:
            rows = calculate_reflections(
                self.structure,
                self.factors,
                self.radiations,
                min_two_theta=max(0.0, float(self.limits[0])),
                max_two_theta=min(179.9, float(self.limits[1])),
                min_intensity=0.1,
            )
            error = None
        except Exception as exc:
            rows = []
            error = exc
        self.signals.finished.emit(self.uid, self.key, rows, error)


def _reference_peak_path() -> Path:
    """Return the editable peak list used by the stable interface."""

    return reference_peak_path()


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


class SubstrateCorrectionDialog(QDialog):
    """Modeless editor for calibration from one harmonic substrate family."""

    add_peak_requested = Signal()
    apply_requested = Signal(object)

    def __init__(
        self,
        measurement_name: str,
        references: dict[str, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(
            localised(
                "Substrate peak correction",
                "Correction par les pics du substrat",
                "Коррекция по пикам подложки",
            )
        )
        self.setMinimumSize(720, 430)
        self.calibration: SubstrateCalibration | None = None

        root = QVBoxLayout(self)
        note = QLabel(
            localised(
                f"Measurement: {measurement_name}. Add two or three successive orders of one substrate reflection family.",
                f"Mesure : {measurement_name}. Ajoutez deux ou trois ordres successifs d’une même famille de réflexions du substrat.",
                f"Измерение: {measurement_name}. Добавьте два или три последовательных порядка одной семьи отражений подложки.",
            )
        )
        note.setWordWrap(True)
        root.addWidget(note)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            (
                localised("Order", "Ordre", "Порядок"),
                localised("Measured 2θ, °", "2θ mesuré, °", "Измеренное 2θ, °"),
                localised("Expected 2θ, °", "2θ attendu, °", "Ожидаемое 2θ, °"),
                localised("Applied correction, °", "Correction appliquée, °", "Применяемая поправка, °"),
                localised("Residual, °", "Résidu, °", "Остаток, °"),
            )
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        root.addWidget(self.table, 1)

        peak_buttons = QHBoxLayout()
        self.add_button = QPushButton("+")
        self.add_button.setToolTip(
            localised(
                "Select another peak on the main plot",
                "Sélectionner un autre pic sur le graphique principal",
                "Выбрать следующий пик на основном графике",
            )
        )
        self.remove_button = QPushButton("−")
        self.remove_button.setToolTip(
            localised("Remove selected peak", "Supprimer le pic sélectionné", "Удалить выбранный пик")
        )
        self.add_button.clicked.connect(self.add_peak_requested.emit)
        self.remove_button.clicked.connect(self.remove_selected_peak)
        peak_buttons.addWidget(self.add_button)
        peak_buttons.addWidget(self.remove_button)
        peak_buttons.addStretch(1)
        root.addLayout(peak_buttons)

        reference_form = QFormLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.addItem(
            localised(
                "Unknown — fit a common shift",
                "Inconnue — ajuster un décalage commun",
                "Неизвестно — рассчитать общий сдвиг",
            ),
            None,
        )
        for name, value in references.items():
            self.reference_combo.addItem(name, float(value))
        self.reference_combo.currentIndexChanged.connect(self._reference_selected)
        reference_form.addRow(
            localised("Reference", "Référence", "Опорное положение"),
            self.reference_combo,
        )

        self.true_position_edit = QLineEdit()
        self.true_position_edit.setPlaceholderText(
            localised("optional", "facultatif", "необязательно")
        )
        self.true_position_edit.textChanged.connect(self.invalidate)
        reference_form.addRow(
            localised("True 2θ, °", "2θ vrai, °", "Истинное 2θ, °"),
            self.true_position_edit,
        )

        self.reference_order_combo = QComboBox()
        self.reference_order_combo.currentIndexChanged.connect(self.invalidate)
        reference_form.addRow(
            localised("Reference order", "Ordre de référence", "Порядок опорного пика"),
            self.reference_order_combo,
        )
        root.addLayout(reference_form)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        root.addWidget(self.result_label)

        actions = QHBoxLayout()
        self.calculate_button = QPushButton(tr("text.calculate"))
        self.apply_button = QPushButton(
            localised("Apply calibration", "Appliquer l’étalonnage", "Применить коррекцию")
        )
        self.close_button = QPushButton(tr("text.close"))
        self.calculate_button.clicked.connect(self.calculate)
        self.apply_button.clicked.connect(self._apply)
        self.close_button.clicked.connect(self.close)
        actions.addStretch(1)
        actions.addWidget(self.calculate_button)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.close_button)
        root.addLayout(actions)
        self._update_buttons()

    @staticmethod
    def _readonly_item(text: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    def add_peak(self, centre: float, _intensity: float | None = None) -> None:
        if self.table.rowCount() >= 3:
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        order = QSpinBox()
        order.setRange(1, 99)
        order.setValue(row + 1)
        order.valueChanged.connect(self._orders_changed)
        self.table.setCellWidget(row, 0, order)
        self.table.setItem(row, 1, self._readonly_item(f"{float(centre):.8f}"))
        for column in range(2, 5):
            self.table.setItem(row, column, self._readonly_item())
        self.table.selectRow(row)
        self._orders_changed()

    def remove_selected_peak(self) -> None:
        row = self.table.currentRow()
        if row < 0 and self.table.rowCount():
            row = self.table.rowCount() - 1
        if row >= 0:
            self.table.removeRow(row)
            self._orders_changed()

    def _orders(self) -> list[int]:
        result = []
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            result.append(int(widget.value()))
        return result

    def _measured(self) -> list[float]:
        return [
            float(self.table.item(row, 1).text())
            for row in range(self.table.rowCount())
        ]

    def _orders_changed(self, _value: int | None = None) -> None:
        previous = self.reference_order_combo.currentData()
        self.reference_order_combo.blockSignals(True)
        self.reference_order_combo.clear()
        for order in self._orders():
            self.reference_order_combo.addItem(str(order), order)
        index = self.reference_order_combo.findData(previous)
        self.reference_order_combo.setCurrentIndex(index if index >= 0 else 0)
        self.reference_order_combo.blockSignals(False)
        self.invalidate()

    def _reference_selected(self, index: int) -> None:
        value = self.reference_combo.itemData(index)
        if value is None:
            self.true_position_edit.clear()
        else:
            self.true_position_edit.setText(f"{float(value):.8f}")
        self.invalidate()

    def invalidate(self, *_args) -> None:
        self.calibration = None
        self.result_label.clear()
        for row in range(self.table.rowCount()):
            for column in range(2, 5):
                item = self.table.item(row, column)
                if item is not None:
                    item.setText("")
        self._update_buttons()

    def _update_buttons(self) -> None:
        count = self.table.rowCount()
        self.add_button.setEnabled(count < 3)
        self.remove_button.setEnabled(count > 0)
        self.calculate_button.setEnabled(count >= 2)
        self.apply_button.setEnabled(self.calibration is not None)
        self.reference_order_combo.setEnabled(count > 0)

    def _error_text(self, exc: Exception) -> str:
        if not isinstance(exc, XRDDataError):
            return str(exc)
        messages = {
            "substrate_calibration_peaks": localised(
                "Add at least two peaks.",
                "Ajoutez au moins deux pics.",
                "Добавьте не менее двух пиков.",
            ),
            "substrate_calibration_angles": localised(
                "Peak positions must lie between 0° and 180°.",
                "Les positions des pics doivent être comprises entre 0° et 180°.",
                "Положения пиков должны находиться между 0° и 180°.",
            ),
            "substrate_calibration_orders": localised(
                "Orders must be positive and must not repeat.",
                "Les ordres doivent être positifs et distincts.",
                "Порядки должны быть положительными и не должны повторяться.",
            ),
            "substrate_calibration_reference": localised(
                "Enter a true position between 0° and 180°.",
                "Saisissez une position vraie comprise entre 0° et 180°.",
                "Введите истинное положение между 0° и 180°.",
            ),
            "substrate_calibration_harmonic": localised(
                "The selected orders cannot belong to this harmonic series.",
                "Les ordres sélectionnés ne peuvent pas appartenir à cette série harmonique.",
                "Выбранные порядки не могут принадлежать этой серии отражений.",
            ),
            "substrate_calibration_reference_order": localised(
                "Select the order of the known reference peak.",
                "Sélectionnez l’ordre du pic de référence connu.",
                "Выберите порядок известного опорного пика.",
            ),
            "substrate_calibration_fit": localised(
                "The angular correction could not be fitted to these peaks.",
                "La correction angulaire ne peut pas être ajustée à ces pics.",
                "Не удалось рассчитать угловую коррекцию по этим пикам.",
            ),
            "substrate_calibration_scipy": localised(
                "SciPy is required for substrate peak correction.",
                "SciPy est requis pour la correction par les pics du substrat.",
                "Для коррекции по пикам подложки требуется SciPy.",
            ),
        }
        return messages.get(exc.code, str(exc))

    def calculate(self) -> None:
        target_text = self.true_position_edit.text().strip().replace(",", ".")
        try:
            true_position = float(target_text) if target_text else None
            reference_order = (
                int(self.reference_order_combo.currentData())
                if true_position is not None
                else None
            )
            calibration = fit_substrate_calibration(
                self._measured(),
                self._orders(),
                true_position=true_position,
                reference_order=reference_order,
            )
        except (TypeError, ValueError, ArithmeticError, XRDDataError) as exc:
            QMessageBox.critical(
                self,
                localised("Calibration error", "Erreur d’étalonnage", "Ошибка коррекции"),
                self._error_text(exc),
            )
            return
        self.calibration = calibration
        measured = np.asarray(self._measured(), dtype=float)
        applied = calibration.corrected(measured) - measured
        for row, (expected, correction, residual) in enumerate(
            zip(calibration.expected, applied, calibration.residuals)
        ):
            self.table.item(row, 2).setText(f"{expected:.8f}")
            self.table.item(row, 3).setText(f"{correction:+.8f}")
            self.table.item(row, 4).setText(f"{residual:+.8f}")
        if calibration.mode == "common_shift":
            equation = f"2θtrue = 2θmeas {calibration.x_shift:+.8f}°"
        else:
            equation = (
                f"2θtrue = {calibration.x_scale:.9f} · 2θmeas "
                f"{calibration.x_shift:+.8f}°"
            )
        self.result_label.setText(
            localised(
                f"{equation}; RMS residual = {calibration.rms:.6g}°",
                f"{equation} ; résidu RMS = {calibration.rms:.6g}°",
                f"{equation}; среднеквадратичный остаток = {calibration.rms:.6g}°",
            )
        )
        self._update_buttons()

    def _apply(self) -> None:
        if self.calibration is not None:
            self.apply_requested.emit(self.calibration)
            self.accept()


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
        on_open_structure=None,
        on_open_poles=None,
        plot_renderer_controller=None,
        debug_features=None,
    ) -> None:
        super().__init__(parent)
        self.workspace = VIEWER
        self.store = store
        self.radiation_settings = radiation_settings
        self.on_open_comparison = on_open_comparison
        self.on_open_reflection_table = on_open_reflection_table
        self.on_open_structure = on_open_structure
        self.on_open_poles = on_open_poles
        self.plot_renderer_controller = plot_renderer_controller
        self.debug_features = debug_features
        self.indexation_enabled = bool(debug_features and debug_features.indexation_enabled)
        self._indexing_dialogs = {}
        if debug_features is not None:
            debug_features.indexation_changed.connect(self.set_indexation_enabled)
        self.plot_renderer = "matplotlib"
        self.pyqtgraph_plot = None
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
        self._pending_row_jobs: set[tuple[str, tuple]] = set()
        self._phase_row_error = ""
        self._phase_thread_pool = QThreadPool(self)
        self._phase_thread_pool.setMaxThreadCount(1)
        self._factors = None
        self._axes_linked = True
        self._syncing_x_limits = False
        self._overlay_phase_top = 0.0
        self._syncing_processing = False
        self._processing_x_limit = 1.0
        self._processing_y_limit = 1.0
        self._fit_selector: RectangleSelector | None = None
        self._fit_uid: str | None = None
        self._fit_callback: Callable[[float, float], None] | None = None
        self._fit_artists: list[object] = []
        self._peak_search_active = False
        self._manual_peak_active = False
        self._manual_mpl_cids: list[int] = []
        self._manual_mpl_start: tuple[float, float] | None = None
        self._pending_peak_fit: PendingPeakFit | None = None
        self._session_peaks: list[SessionPeak] = []
        self._show_peak_sum: dict[str, bool] = {}
        self._next_peak_number = 1
        self._peak_table_dialog: PeakTableDialog | None = None
        self._peak_table_dialogs: dict[str, PeakTableDialog] = {}
        self._backgrounds: dict[tuple[str, str], BackgroundAnchors] = {}
        self._background_selection_uid: str | None = None
        self._substrate_dialog: SubstrateCorrectionDialog | None = None

        self._build_ui()
        self._connect_plot_events()
        if self.plot_renderer_controller is not None:
            self.set_plot_renderer(self.plot_renderer_controller.mode)
            self.plot_renderer_controller.changed.connect(self.set_plot_renderer)
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
        self.pole_button.clicked.connect(self.open_selected_poles)
        self.structure_button.clicked.connect(self.open_selected_structure)
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

        self.peak_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.peak_section)
        self.peak_search_button = QPushButton()
        self.peak_search_button.clicked.connect(self.activate_peak_search)
        self.peak_draw_button = QPushButton()
        self.peak_draw_button.clicked.connect(self.activate_drawn_peak)
        self.peak_do_fit_button = QPushButton("Do Fit")
        self.peak_do_fit_button.clicked.connect(self._do_selected_peak_fit)
        self.peak_table_button = QPushButton()
        self.peak_table_button.clicked.connect(self.open_peak_table)
        self.peak_section.content_layout.addWidget(self.peak_search_button)
        self.peak_section.content_layout.addWidget(self.peak_draw_button)
        self.peak_section.content_layout.addWidget(self.peak_do_fit_button)
        self.peak_section.content_layout.addWidget(self.peak_table_button)

        self.background_section = CollapsibleSection("")
        self.controls_layout.addWidget(self.background_section)
        background_row = QHBoxLayout()
        self.background_spacing_label = QLabel()
        self.background_spacing_spin = QDoubleSpinBox()
        self.background_spacing_spin.setDecimals(3)
        self.background_spacing_spin.setRange(0.001, 1000.0)
        self.background_spacing_spin.setValue(0.25)
        self.background_spacing_spin.setSingleStep(0.05)
        self.background_spacing_spin.valueChanged.connect(self._background_spacing_changed)
        background_row.addWidget(self.background_spacing_label)
        background_row.addWidget(self.background_spacing_spin)
        self.background_section.content_layout.addLayout(background_row)
        self.background_auto_button = QPushButton()
        self.background_auto_button.clicked.connect(self.calculate_background)
        self.background_section.content_layout.addWidget(self.background_auto_button)
        self.background_exclude_button = QPushButton()
        self.background_exclude_button.clicked.connect(self.select_background_gap)
        self.background_section.content_layout.addWidget(self.background_exclude_button)
        self.background_reset_button = QPushButton()
        self.background_reset_button.clicked.connect(self.clear_background)
        self.background_section.content_layout.addWidget(self.background_reset_button)

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
            # The control pane is only about 300 px wide. Letting a spin box
            # ignore its size hint in a three-column grid collapses it to
            # zero width, so the numeric entry is invisible to the user.
            spin.setMinimumWidth(138)
            spin.setMaximumWidth(160)
            spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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
            label.setWordWrap(True)
            processing_grid.addWidget(label, row * 2, 0)
            processing_grid.addWidget(spin, row * 2, 1)
            processing_grid.addWidget(slider, row * 2 + 1, 0, 1, 2)
        processing_grid.setColumnStretch(0, 1)
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

        self.substrate_correction_button = QPushButton()
        _allow_horizontal_shrink(self.substrate_correction_button)
        self.substrate_correction_button.clicked.connect(
            self.open_substrate_correction
        )
        self.processing_section.content_layout.addWidget(
            self.substrate_correction_button
        )

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
            self.substrate_correction_button,
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

        self.x_display_label = QLabel()
        self.x_display_combo = QComboBox()
        _allow_horizontal_shrink(self.x_display_combo)
        self.x_display_combo.currentIndexChanged.connect(
            self._x_display_mode_changed
        )
        display_form.addRow(self.x_display_label, self.x_display_combo)

        self.display_wavelength_label = QLabel()
        self.display_wavelength_combo = QComboBox()
        _allow_horizontal_shrink(self.display_wavelength_combo)
        self.display_wavelength_combo.currentIndexChanged.connect(
            self._display_wavelength_changed
        )
        display_form.addRow(
            self.display_wavelength_label,
            self.display_wavelength_combo,
        )

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
        self.matplotlib_plot = self.canvas
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.matplotlib_plot)
        plot_layout.addWidget(self.plot_stack, 0, 0)
        self.plot_stack.installEventFilter(self)

        self.peak_preview_bar = QFrame(self.plot_stack)
        self.peak_preview_bar.setFrameShape(QFrame.Shape.StyledPanel)
        self.peak_preview_bar.setAutoFillBackground(True)
        preview_layout = QHBoxLayout(self.peak_preview_bar)
        preview_layout.setContentsMargins(8, 4, 8, 4)
        self.peak_preview_count = QLabel()
        preview_layout.addWidget(self.peak_preview_count)
        self.peak_preview_checks: list[QCheckBox] = []
        self.peak_preview_checks_layout = QGridLayout()
        self.peak_preview_checks_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addLayout(self.peak_preview_checks_layout)
        preview_layout.addStretch(1)
        self.peak_preview_confirm = QToolButton()
        self.peak_preview_confirm.setText("✓")
        self.peak_preview_confirm.clicked.connect(self._confirm_peak_preview)
        preview_layout.addWidget(self.peak_preview_confirm)
        self.peak_preview_cancel = QToolButton()
        self.peak_preview_cancel.setText("✕")
        self.peak_preview_cancel.clicked.connect(self._discard_peak_preview)
        preview_layout.addWidget(self.peak_preview_cancel)
        self.peak_preview_bar.hide()

        self.y_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.y_scrollbar.valueChanged.connect(lambda value: self._scrollbar_changed("y", value))
        plot_layout.addWidget(self.y_scrollbar, 0, 1)

        self.toolbar = PlotToolbar(self.canvas, plot_widget)
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

    def eventFilter(self, watched, event):
        if watched is self.plot_stack and event.type() == QEvent.Type.Resize:
            self._position_peak_preview_bar()
        return super().eventFilter(watched, event)

    def _position_peak_preview_bar(self) -> None:
        if not hasattr(self, "peak_preview_bar"):
            return
        bar = self.peak_preview_bar
        size = bar.layout().sizeHint()
        bar.resize(min(size.width(), max(120, self.plot_stack.width() - 24)), size.height())
        bar.move(max(8, self.plot_stack.width() - bar.width() - 12), 12)
        bar.raise_()

    def set_plot_renderer(self, mode: str) -> None:
        """Switch the Viewer display adapter without changing its data state."""

        if mode not in {"matplotlib", "pyqtgraph"}:
            raise ValueError(mode)
        old_x, old_y = self._active_scan_limits()
        old_phase_x, _old_phase_y = self._active_phase_limits()
        had_data = self._has_drawn_data
        self._cancel_peak_fit()
        if mode == "pyqtgraph":
            if self.pyqtgraph_plot is None:
                from .pyqtgraph_viewer import PyQtGraphViewerPlot

                self.pyqtgraph_plot = PyQtGraphViewerPlot(self)
                self.pyqtgraph_plot.plot_clicked.connect(
                    self._pyqtgraph_plot_clicked
                )
                self.pyqtgraph_plot.peak_region_selected.connect(
                    self._pyqtgraph_peak_region
                )
                self.pyqtgraph_plot.manual_peak_drawn.connect(
                    self._pyqtgraph_peak_region
                )
                self.pyqtgraph_plot.range_changed.connect(
                    self._pyqtgraph_range_changed
                )
                self.pyqtgraph_plot.reset_requested.connect(self.reset_limits)
                self.pyqtgraph_plot.settings_changed.connect(
                    lambda: self._draw(preserve_view=True)
                )
                self.pyqtgraph_plot.palette_changed.connect(
                    self._pyqtgraph_palette_changed
                )
                self.plot_stack.addWidget(self.pyqtgraph_plot)
            self.plot_stack.setCurrentWidget(self.pyqtgraph_plot)
            self.toolbar.hide()
        else:
            if self.pyqtgraph_plot is not None:
                self.pyqtgraph_plot.set_zoom_enabled(False)
            self.plot_stack.setCurrentWidget(self.matplotlib_plot)
            self.toolbar.show()
        changed = mode != self.plot_renderer
        self.plot_renderer = mode
        if changed or not self._has_drawn_data:
            self._draw(preserve_view=False)
            if had_data and all(np.isfinite((*old_x, *old_y))):
                if old_x[0] < old_x[1] and old_y[0] < old_y[1]:
                    self._set_limits(old_x, old_y)
                    if self._phase_plot_visible() and not self._axes_linked:
                        self._set_phase_x_limits(old_phase_x)

    def _active_scan_limits(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            return self.pyqtgraph_plot.scan_limits()
        return tuple(self.scan_axis.get_xlim()), tuple(self.scan_axis.get_ylim())

    def _active_phase_limits(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            return self.pyqtgraph_plot.phase_limits()
        return tuple(self.phase_axis.get_xlim()), tuple(self.phase_axis.get_ylim())

    def _phase_plot_visible(self) -> bool:
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            return self.pyqtgraph_plot.phase_visible
        return bool(self.phase_axis.get_visible())

    def _set_phase_x_limits(self, limits: tuple[float, float]) -> None:
        if not all(np.isfinite(limits)) or limits[0] >= limits[1]:
            return
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            _old_x, old_y = self.pyqtgraph_plot.phase_limits()
            self.pyqtgraph_plot.set_phase_range(limits, old_y)
        else:
            self.phase_axis.set_xlim(*limits)

    def _pyqtgraph_range_changed(self, source: str) -> None:
        if self._drawing or self.plot_renderer != "pyqtgraph":
            return
        if (
            self.pyqtgraph_plot is not None
            and self._axes_linked
            and self.pyqtgraph_plot.phase_visible
            and not self._syncing_x_limits
        ):
            self._syncing_x_limits = True
            try:
                if source == "scan":
                    scan_x, _scan_y = self.pyqtgraph_plot.scan_limits()
                    _phase_x, phase_y = self.pyqtgraph_plot.phase_limits()
                    self.pyqtgraph_plot.set_phase_range(scan_x, phase_y)
                else:
                    phase_x, _phase_y = self.pyqtgraph_plot.phase_limits()
                    _scan_x, scan_y = self.pyqtgraph_plot.scan_limits()
                    self.pyqtgraph_plot.set_scan_range(phase_x, scan_y)
            finally:
                self._syncing_x_limits = False
        self._update_navigation_scrollbars()

    def _pyqtgraph_palette_changed(self) -> None:
        if self.plot_renderer == "pyqtgraph":
            self._draw(preserve_view=True)

    def retranslate(self) -> None:
        self.data_section.set_title(tr("text.loaded_datasets"))
        self.peak_section.set_title(localised("Peak search", "Recherche de pics", "Поиск пиков"))
        self.peak_search_button.setText(localised(
            "Select and fit peaks…", "Sélectionner et ajuster des pics…", "Выделить и найти пики…"
        ))
        self.peak_draw_button.setText(localised(
            "Draw peak…", "Dessiner un pic…", "Нарисовать пик…"
        ))
        self.peak_do_fit_button.setToolTip(localised(
            "Fit accepted peaks against the measured scan",
            "Ajuster les pics validés aux données mesurées",
            "Подогнать принятые пики к измеренному спектру",
        ))
        self.peak_table_button.setText(localised(
            "Detected peaks…", "Pics détectés…", "Найденные пики…"
        ))
        self.background_section.set_title(localised("Background", "Fond", "Фон"))
        self.background_spacing_label.setText(localised(
            "Anchor spacing, source axis", "Espacement des points, axe source",
            "Шаг точек по исходной оси"))
        self.background_auto_button.setText(localised(
            "Find background", "Estimer le fond", "Найти фон"))
        self.background_exclude_button.setText(localised(
            "Exclude anchors in region…", "Exclure des points…",
            "Убрать точки в области…"))
        self.background_reset_button.setText(localised(
            "Remove background", "Supprimer le fond", "Убрать фон"))
        self.peak_preview_confirm.setToolTip(localised(
            "Confirm selected peaks", "Confirmer les pics sélectionnés", "Добавить выбранные пики"
        ))
        self.peak_preview_cancel.setToolTip(localised(
            "Discard fit", "Ignorer l'ajustement", "Отменить подгонку"
        ))
        if self._pending_peak_fit is not None:
            self._update_peak_preview_bar()
        self.processing_section.set_title(tr("text.selected_measurement_correction"))
        self.display_section.set_title(tr("text.display"))
        self.limits_section.set_title(tr("text.plot_limits"))
        self.documents.setHeaderLabels(
            (tr("text.name"), tr("text.vis"), tr("text.type"), tr("text.colour"))
        )
        self.axis_label.setText(tr("text.x_axis"))
        self.x_display_label.setText(
            localised("Horizontal scale", "Échelle horizontale", "Горизонтальная шкала")
        )
        self.display_wavelength_label.setText("λ, Å")
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
        self.substrate_correction_button.setText(
            localised(
                "Substrate peak correction…",
                "Correction par les pics du substrat…",
                "Коррекция по пикам подложки…",
            )
        )
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
        for dialog in self._peak_table_dialogs.values():
            dialog.retranslate()
        for dialog in self._indexing_dialogs.values():
            dialog.retranslate()
        if self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.retranslate()

        current_x_display = (
            self.x_display_combo.currentData()
            or self.viewer_state.plot.x_display_mode
        )
        self.x_display_combo.blockSignals(True)
        self.x_display_combo.clear()
        self.x_display_combo.addItem("2θ", "angle")
        self.x_display_combo.addItem("d, Å", "d")
        self.x_display_combo.setCurrentIndex(
            max(0, self.x_display_combo.findData(current_x_display))
        )
        self.x_display_combo.blockSignals(False)
        self._populate_display_wavelengths()
        self._update_x_display_controls()

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
        self._update_x_display_controls()
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
        for uid in tuple(self._indexing_dialogs):
            if uid not in assigned:
                self._indexing_dialogs.pop(uid).close()
        added_scans = []
        added_phases = []
        for uid in removed:
            self.viewer_state.remove(uid)
            if self._background_selection_uid == uid:
                self._cancel_peak_fit()
            for key in tuple(self._backgrounds):
                if key[0] == uid:
                    del self._backgrounds[key]
            if uid not in self.store.documents:
                self._session_peaks = [replace(peak, hkl_assignments=tuple(
                    entry for entry in peak.hkl_assignments if entry[0] != uid))
                    for peak in self._session_peaks]
            self._session_peaks = [p for p in self._session_peaks if p.scan_uid != uid]
            self._show_peak_sum.pop(uid, None)
            dialog = self._peak_table_dialogs.pop(uid, None)
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()
                if self._peak_table_dialog is dialog:
                    self._peak_table_dialog = None
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
        self._refresh_peak_table()
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
        elif sum(item.scan is not None for item in self.items.values()) == 1:
            only_scan = next(item for item in self.items.values() if item.scan is not None)
            for index in range(self.documents.topLevelItemCount()):
                row = self.documents.topLevelItem(index)
                if row.data(0, Qt.ItemDataRole.UserRole) == only_scan.uid:
                    self.documents.setCurrentItem(row)
                    break
        self.documents.blockSignals(False)
        self._refreshing_tree = False
        self._selection_changed()
        self._refresh_peak_table()

    def _tree_item_changed(self, row: QTreeWidgetItem, column: int) -> None:
        if self._refreshing_tree or column != 1:
            return
        uid = row.data(0, Qt.ItemDataRole.UserRole)
        item = self.items.get(uid)
        if item is None:
            return
        item.visible = row.checkState(1) == Qt.CheckState.Checked
        self._update_x_display_controls()
        self._update_buttons()
        self._load_processing_controls(item)
        self._update_background_controls()
        self._refresh_selection_info()
        self._draw(preserve_view=True)

    def _selection_changed(self) -> None:
        self._cancel_peak_fit()
        self._populate_axis_combo()
        self._update_buttons()
        self._load_processing_controls(self._selected_item())
        self._update_background_controls()

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
        self.peak_search_button.setEnabled(item is not None and item.scan is not None)
        self.peak_draw_button.setEnabled(item is not None and item.scan is not None)
        self.peak_do_fit_button.setEnabled(bool(item is not None and item.scan is not None
            and any(p.scan_uid == item.uid and p.axis_name == item.scan.axis_name
                    for p in self._session_peaks)))
        self.peak_table_button.setEnabled(item is not None and item.scan is not None)
        self.pole_button.setEnabled(
            item is not None and item.is_phase and self.on_open_poles is not None
        )
        self.structure_button.setEnabled(
            item is not None and item.is_phase and self.on_open_structure is not None
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
        self._update_x_display_controls()

    def _populate_display_wavelengths(self) -> None:
        if not hasattr(self, "display_wavelength_combo"):
            return
        current = float(self.viewer_state.plot.display_wavelength)
        choices: list[tuple[str, float]] = []
        seen: set[float] = set()
        for preset in PRESETS:
            for name, wavelength, _weight in preset.lines:
                key = round(float(wavelength), 10)
                if key not in seen:
                    choices.append((name, float(wavelength)))
                    seen.add(key)
        try:
            active_lines = self.radiation_settings.lines()
        except ValueError:
            active_lines = []
        for name, wavelength, _weight in active_lines:
            key = round(float(wavelength), 10)
            if key not in seen:
                choices.append((name, float(wavelength)))
                seen.add(key)
        self.display_wavelength_combo.blockSignals(True)
        self.display_wavelength_combo.clear()
        best_index = 0
        best_distance = math.inf
        for index, (name, wavelength) in enumerate(choices):
            self.display_wavelength_combo.addItem(
                f"{name} ({wavelength:.5f} Å)",
                wavelength,
            )
            distance = abs(wavelength - current)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if choices:
            self.display_wavelength_combo.setCurrentIndex(best_index)
            self.viewer_state.plot.display_wavelength = float(
                self.display_wavelength_combo.currentData()
            )
        self.display_wavelength_combo.blockSignals(False)

    def _update_x_display_controls(self) -> None:
        if not hasattr(self, "x_display_combo"):
            return
        visible_scans = self.viewer_state.visible_scans()
        compatible = not visible_scans or all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in visible_scans
        )
        self.x_display_combo.setEnabled(compatible)
        d_mode = self.viewer_state.plot.x_display_mode == "d"
        if d_mode and not compatible:
            self.viewer_state.plot.x_display_mode = "angle"
            self.x_display_combo.blockSignals(True)
            self.x_display_combo.setCurrentIndex(
                self.x_display_combo.findData("angle")
            )
            self.x_display_combo.blockSignals(False)
            d_mode = False
        self.display_wavelength_label.setVisible(d_mode)
        self.display_wavelength_combo.setVisible(d_mode)

    def _x_display_mode_changed(self, _index: int) -> None:
        mode = self.x_display_combo.currentData()
        if mode not in {"angle", "d"}:
            return
        self._cancel_peak_fit()
        self.viewer_state.plot.x_display_mode = str(mode)
        self._selected_point = None
        self._row_cache.clear()
        self._update_x_display_controls()
        self._load_processing_controls(self._selected_item())
        self._draw(preserve_view=False)

    def _display_wavelength_changed(self, _index: int) -> None:
        value = self.display_wavelength_combo.currentData()
        if value is None:
            return
        self.viewer_state.plot.display_wavelength = float(value)
        self._selected_point = None
        self._row_cache.clear()
        if self.viewer_state.plot.x_display_mode == "d":
            self._draw(preserve_view=False)

    def _axis_changed(self, _index: int) -> None:
        if self._refreshing_axis:
            return
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        axis_name = self.axis_combo.currentData()
        if item is None or item.scan is None or not axis_name:
            return
        item.scan.use_axis(str(axis_name))
        if (
            self.viewer_state.plot.x_display_mode == "d"
            and not is_two_theta(item.scan.axis_name)
        ):
            self.viewer_state.plot.x_display_mode = "angle"
            self.x_display_combo.blockSignals(True)
            self.x_display_combo.setCurrentIndex(
                self.x_display_combo.findData("angle")
            )
            self.x_display_combo.blockSignals(False)
        self._selected_point = None
        self._update_buttons()
        self._load_processing_controls(item)
        self._update_background_controls()
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
        angular_view = self.viewer_state.plot.x_display_mode == "angle"
        self.fit_button.setEnabled(
            bool(scan_item is not None and scan_item.visible and angular_view)
        )
        self.substrate_correction_button.setEnabled(
            bool(
                scan_item is not None
                and scan_item.visible
                and scan_item.scan is not None
                and is_two_theta(scan_item.scan.axis_name)
                and angular_view
            )
        )

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
                x_scale=item.x_scale,
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
                x_scale=item.x_scale,
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
        old_x, old_y = self._active_scan_limits()
        if mode == "add":
            document = self.store.add_scan(scan, derived=True, parent_uid=uid)
            self.store.assign(document.uid, VIEWER, True)
        else:
            self.store.replace_scan(uid, scan)
            item.name = scan.name
            item.source = Path(scan.source)
            item.scan = clone_scan(scan)
            self._session_peaks = [p for p in self._session_peaks if p.scan_uid != uid]
            for key in tuple(self._backgrounds):
                if key[0] == uid:
                    del self._backgrounds[key]
            self._refresh_peak_table()
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
        self._discard_peak_preview()
        for connection in self._manual_mpl_cids:
            self.canvas.mpl_disconnect(connection)
        self._manual_mpl_cids.clear()
        self._manual_mpl_start = None
        if self._fit_selector is not None:
            try:
                self._fit_selector.set_active(False)
                self._fit_selector.set_visible(False)
                self._fit_selector.disconnect_events()
            except Exception:
                pass
        self._fit_selector = None
        if self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.set_peak_selection_enabled(False)
            self.pyqtgraph_plot.set_manual_peak_enabled(False)
        self._clear_peak_fit_artifacts()
        self._fit_uid = None
        self._fit_callback = None
        self._peak_search_active = False
        self._manual_peak_active = False
        self._background_selection_uid = None

    def _clear_peak_fit_artifacts(self) -> None:
        for artist in self._fit_artists:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        had_artists = bool(self._fit_artists)
        self._fit_artists.clear()
        if self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.clear_peak_fit()
        if had_artists and hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _discard_peak_preview(self) -> None:
        self._pending_peak_fit = None
        if hasattr(self, "peak_preview_bar"):
            self.peak_preview_bar.hide()
        self._clear_peak_fit_artifacts()

    def _update_peak_preview_bar(self) -> None:
        pending = self._pending_peak_fit
        if pending is None:
            return
        count = len(pending.result.peaks)
        self.peak_preview_count.setText(localised(
            f"Drawn peaks: {count}" if pending.manual else f"Found peaks: {count}",
            f"Pics dessinés : {count}" if pending.manual else f"Pics trouvés : {count}",
            f"Нарисовано пиков: {count}" if pending.manual else f"Найдено пиков: {count}"
        ))
        self.peak_preview_confirm.setToolTip(localised(
            "Insert drawn peak and local background" if pending.manual else "Confirm selected peaks",
            "Insérer le pic dessiné et le fond local" if pending.manual else "Confirmer les pics sélectionnés",
            "Вставить нарисованный пик и местный фон" if pending.manual else "Добавить выбранные пики",
        ))
        while len(self.peak_preview_checks) < count:
            check = QCheckBox(self.peak_preview_bar)
            check.setChecked(True)
            self.peak_preview_checks.append(check)
            index = len(self.peak_preview_checks) - 1
            self.peak_preview_checks_layout.addWidget(check, index // 3, index % 3)
        item = self.items.get(pending.uid)
        for index, check in enumerate(self.peak_preview_checks):
            check.setVisible(index < count and count > 1)
            if index < count and item is not None:
                display_x = self._display_x_values(
                    item, np.asarray([pending.source_centers[index] * item.x_scale + item.x_shift])
                )[0]
                check.setText(f"{display_x:.5f}")
        self.peak_preview_bar.show()
        self._position_peak_preview_bar()

    def _show_pending_peak_fit(self) -> None:
        """Draw the unconfirmed Voigt sum over the current displayed measurement."""

        pending = self._pending_peak_fit
        if pending is None:
            return
        item = self.items.get(pending.uid)
        if item is None or item.scan is None or item.scan.axis_name != pending.axis_name:
            self._discard_peak_preview()
            return
        if item.uid not in {scan.uid for scan in self.viewer_state.visible_scans()}:
            self._discard_peak_preview()
            return
        offset = next(index for index, scan in enumerate(self.viewer_state.visible_scans())
                      if scan.uid == item.uid) * self.viewer_state.plot.vertical_offset
        def display_x(source_x):
            return self._display_x_values(item, source_x * item.x_scale + item.x_shift)

        def display_y(source_y):
            return transformed_intensity(
                source_y * item.y_factor + item.y_shift,
                self.viewer_state.plot.intensity_scale,
            ) + offset

        selected_x = display_x(pending.selected_source_x)
        selected_y = display_y(pending.selected_source_y)
        fit_x = display_x(pending.source_x)
        fit_y = display_y(pending.source_fitted)
        baseline_y = display_y(pending.source_background) if pending.manual else None
        centres = [float(display_x(np.asarray([center]))[0])
                   for center in pending.source_centers]
        centre_y = [float(display_y(np.asarray([np.interp(center, pending.source_x,
                                                         pending.source_fitted)]))[0])
                    for center in pending.source_centers]
        self._clear_peak_fit_artifacts()
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.show_peak_region(
                selected_x, selected_y, fit_x, fit_y, centres, centre_y,
                baseline_y=baseline_y,
            )
        else:
            self._fit_artists.append(self.scan_axis.scatter(
                selected_x, selected_y, color="green", s=18, zorder=7
            ))
            self._fit_artists.extend(self.scan_axis.plot(
                fit_x, fit_y, color="orange", linewidth=2, zorder=7
            ))
            if baseline_y is not None:
                self._fit_artists.extend(self.scan_axis.plot(
                    fit_x, baseline_y, color="#8e44ad", linestyle="--",
                    linewidth=1.7, zorder=7,
                ))
            for center, height in zip(centres, centre_y):
                self._fit_artists.append(self.scan_axis.axvline(
                    center, color="red", linestyle="--", linewidth=1
                ))
                self._fit_artists.append(self.scan_axis.scatter(
                    [center], [height], color="red", s=55, zorder=8
                ))
            self.canvas.draw_idle()

    def _confirm_peak_preview(self) -> None:
        pending = self._pending_peak_fit
        if pending is None:
            return
        item = self.items.get(pending.uid)
        if item is None or item.scan is None or item.scan.axis_name != pending.axis_name:
            self._discard_peak_preview()
            return
        chosen = (list(range(len(pending.result.peaks))) if len(pending.result.peaks) == 1
                  else [i for i, check in enumerate(self.peak_preview_checks[:len(pending.result.peaks)])
                        if check.isChecked()])
        spacing = float(np.median(np.diff(pending.selected_source_x)))
        added = 0
        changed_numbers: set[int] = set()
        for index in chosen:
            peak = pending.result.peaks[index]
            source_center = pending.source_centers[index]
            existing = next((previous for previous in self._session_peaks
                             if (pending.replace_number is not None
                                 and previous.number == pending.replace_number)
                             or (pending.replace_number is None
                             and previous.scan_uid == item.uid
                             and previous.axis_name == pending.axis_name
                             and abs(previous.source_center - source_center) < abs(spacing) / 2)),
                            None)
            if existing is not None and not pending.manual:
                continue
            values = dict(
                source_center=source_center, source_x=pending.source_x,
                source_background=pending.source_background,
                source_profile=peak.profile / pending.fit_y_factor,
                source_height=peak.height / pending.fit_y_factor,
                source_area=peak.area / (pending.fit_y_factor * abs(pending.fit_x_scale)),
                source_fwhm=peak.fwhm / abs(pending.fit_x_scale),
                source_sigma=peak.sigma / abs(pending.fit_x_scale),
                source_gamma=peak.gamma / abs(pending.fit_x_scale),
            )
            if existing is None:
                self._session_peaks.append(SessionPeak(
                    number=self._next_peak_number, scan_uid=item.uid,
                    axis_name=pending.axis_name, **values,
                ))
                changed_numbers.add(self._next_peak_number)
                self._next_peak_number += 1
            else:
                position = next(i for i, previous in enumerate(self._session_peaks)
                                if previous is existing)
                self._session_peaks[position] = replace(existing, **values)
                changed_numbers.add(existing.number)
            added += 1
        if added and pending.background_after is not None:
            self._backgrounds[(item.uid, pending.axis_name)] = pending.background_after
            self._update_background_controls()
        self._discard_peak_preview()
        if added:
            fitted = (False if pending.manual else self._refit_session_peaks(
                item.uid, automatic=True, changed_numbers=changed_numbers))
            self._refresh_peak_table()
            self._draw(preserve_view=True)
            self.status.setText(localised(
                "Drawn peak inserted; press Do Fit to refine it.",
                "Pic dessiné inséré ; cliquez sur Do Fit pour l'ajuster.",
                "Нарисованный пик вставлен; нажмите Do Fit для подгонки.",
            ) if pending.manual else localised(
                "Accepted peaks jointly refined." if fitted else "Peaks added; joint fit unavailable.",
                "Pics validés réajustés ensemble." if fitted else "Pics ajoutés ; ajustement indisponible.",
                "Принятые пики совместно уточнены." if fitted else "Пики добавлены; совместная подгонка не удалась.",
            ))

    def activate_peak_fit(self) -> None:
        item = self._selected_item()
        uid = self._selected_uid()
        if item is None or item.scan is None or uid is None:
            return
        self._begin_peak_selection(uid)

    def activate_peak_search(self) -> None:
        """Select a region of the currently selected measurement for Voigt fitting."""

        item = self._selected_item()
        uid = self._selected_uid()
        if item is None or item.scan is None or uid is None:
            return
        self._begin_peak_selection(uid)
        if self._fit_uid == uid:
            self._peak_search_active = True
            self.status.setText(localised(
                "Select an x range around peaks on the main plot.",
                "Sélectionnez une plage en x autour des pics sur le graphique.",
                "Выделите по горизонтали область с пиками на основном графике.",
            ))

    def activate_drawn_peak(self) -> None:
        """Draw a peak's top, baseline and half-width on the selected scan."""
        item = self._selected_item()
        uid = self._selected_uid()
        if item is None or item.scan is None or uid is None:
            return
        self._begin_peak_selection(uid, manual=True)
        if self._fit_uid == uid:
            self.status.setText(localised(
                "Click the peak top, drag to its background level and half-width, then release.",
                "Cliquez sur le sommet du pic, glissez jusqu'au fond et à la demi-largeur, puis relâchez.",
                "Нажмите на вершину пика и проведите к уровню фона на расстояние полуширины.",
            ))

    def open_peak_table(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is None or item.scan is None:
            return
        if uid not in self._peak_table_dialogs:
            dialog = PeakTableDialog(uid, self)
            dialog.set_indexation_enabled(self.indexation_enabled)
            dialog.index_requested.connect(lambda scan_uid=uid: self.open_indexing(scan_uid))
            dialog.remove_requested.connect(
                lambda number, scan_uid=uid: self._remove_session_peak(number, scan_uid)
            )
            dialog.clear_requested.connect(
                lambda scan_uid=uid: self._clear_session_peaks(scan_uid)
            )
            dialog.add_requested.connect(
                lambda scan_uid=uid: self._activate_peak_search_for(scan_uid)
            )
            dialog.draw_requested.connect(
                lambda scan_uid=uid: self._activate_drawn_peak_for(scan_uid)
            )
            dialog.fit_requested.connect(
                lambda scan_uid=uid: self._do_peak_fit(scan_uid)
            )
            dialog.sum_visible_changed.connect(
                lambda visible, scan_uid=uid: self._set_peak_sum_visible(scan_uid, visible)
            )
            dialog.fill_changed.connect(
                lambda number, filled, scan_uid=uid:
                    self._set_peak_fill(scan_uid, number, filled))
            dialog.assign_hkl_requested.connect(
                lambda number, phase_uid, h, k, l, scan_uid=uid:
                    self._assign_peak_hkl(scan_uid, number, phase_uid, h, k, l))
            dialog.clear_hkl_requested.connect(
                lambda number, scan_uid=uid: self._clear_peak_hkl(scan_uid, number))
            dialog.find_alpha2_requested.connect(
                lambda scan_uid=uid: self._find_companions(scan_uid, "ka2")
            )
            dialog.find_beta_requested.connect(
                lambda scan_uid=uid: self._find_companions(scan_uid, "kb")
            )
            self._peak_table_dialogs[uid] = dialog
        self._peak_table_dialog = self._peak_table_dialogs[uid]
        self._refresh_peak_table()
        self._peak_table_dialog.show()
        self._peak_table_dialog.raise_()
        self._peak_table_dialog.activateWindow()

    def set_indexation_enabled(self, enabled: bool) -> None:
        self.indexation_enabled = bool(enabled)
        for dialog in self._peak_table_dialogs.values():
            dialog.set_indexation_enabled(self.indexation_enabled)
        if not self.indexation_enabled:
            self.close_indexing()

    def close_indexing(self) -> None:
        dialogs, self._indexing_dialogs = self._indexing_dialogs, {}
        for dialog in dialogs.values():
            dialog.close()

    def _indexing_signature(self, uid):
        item = self.items.get(uid)
        document = self.store.documents.get(uid)
        if item is None or item.scan is None or document is None:
            return None
        peaks = tuple((p.number, p.axis_name, p.source_center, p.source_height, p.kind)
                      for p in self._session_peaks if p.scan_uid == uid)
        return id(document.payload), item.scan.axis_name, peaks

    def open_indexing(self, uid: str) -> None:
        if not self.indexation_enabled:
            return
        item = self.items.get(uid)
        if item is None or item.scan is None:
            return
        if uid in self._indexing_dialogs:
            dialog = self._indexing_dialogs[uid]
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return
        from ..indexing.models import Peak
        from .indexing_dialog import IndexingDialog
        coordinate = axis_key(item.scan.axis_name)
        if is_two_theta(item.scan.axis_name):
            input_kind = "two_theta"
        elif coordinate in {"d", "dspacing"}:
            input_kind = "d"
        else:
            QMessageBox.warning(self, localised("Index cell", "Indexer la maille", "Индексация"),
                localised("Indexing requires a 2theta or d-spacing axis.",
                          "L’indexation nécessite un axe 2theta ou d.",
                          "Для индексации нужна ось 2theta или межплоскостных расстояний."))
            return
        # Session centres are in measurement coordinates, independent of display mode
        # and pending presentation/processing transforms. Applied corrections are
        # already in the working measurement, and require a fresh peak fit.
        peaks = [Peak(p.source_center, p.source_height, peak_id=p.number, label=str(p.number))
                 for p in self._session_peaks if p.scan_uid == uid
                 and p.axis_name == item.scan.axis_name and p.kind == "primary"]
        wavelength = self.radiation_settings.lines()[0][1]
        dialog = IndexingDialog(peaks, measurement_name=item.name, input_kind=input_kind,
                                wavelength=wavelength, source_signature=self._indexing_signature(uid),
                                parent=self)
        dialog.create_phase_requested.connect(
            lambda candidate, result, scan_uid=uid: self._create_indexed_phase(scan_uid, candidate, result))
        dialog.finished.connect(lambda _result, scan_uid=uid: self._indexing_dialogs.pop(scan_uid, None))
        self._indexing_dialogs[uid] = dialog
        dialog.show()

    def _create_indexed_phase(self, uid, candidate, result) -> None:
        if not self.indexation_enabled:
            return
        indexing = self._indexing_dialogs.get(uid)
        if indexing is None or indexing.source_signature != self._indexing_signature(uid):
            if indexing is not None:
                indexing.invalidate()
            return
        from .project_panel import CellPhaseDialog
        dialog = CellPhaseDialog(self, initial_cell=candidate.cell.as_tuple(),
                                 initial_name=f"{self.items[uid].name} — {result.method_name} #{candidate.rank}")
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_document is None:
            return
        if indexing.source_signature != self._indexing_signature(uid):
            indexing.invalidate()
            return
        phase = self.store.add_cell_phase(dialog.result_document)
        self.store.assign(phase.uid, VIEWER, True)
        if np.allclose(dialog.result_document.cell, candidate.cell.as_tuple(), rtol=1e-7, atol=1e-6):
            assignments = {line.peak_id: (phase.uid, line.h, line.k, line.l)
                           for line in candidate.indexed_lines}
            self._session_peaks = [replace(p, hkl_assignments=p.hkl_assignments + (assignments[p.number],))
                if p.scan_uid == uid and p.number in assignments else p for p in self._session_peaks]
            self._refresh_peak_table()
            self._draw(preserve_view=True)
        else:
            indexing.status.setText(localised(
                "Cell phase created. Cell parameters changed; hkl assignments were not transferred.",
                "Phase créée. La maille a changé ; les indices hkl n’ont pas été transférés.",
                "Cell Phase создана. Параметры ячейки изменены: назначения hkl не перенесены."))

    def _activate_peak_search_for(self, uid: str) -> None:
        if uid not in self.items:
            return
        self.select_uid(uid)
        self.activate_peak_search()

    def _activate_drawn_peak_for(self, uid: str) -> None:
        if uid not in self.items:
            return
        self.select_uid(uid)
        self.activate_drawn_peak()

    def _companion_lines(self) -> tuple[float, float, float | None] | None:
        """Use the element's known doublet even when only Kα1 is selected."""

        key = self.radiation_settings.profile_key
        if key.startswith("cu_"):
            lines = next(preset.lines for preset in PRESETS if preset.key == "cu_ka12_kb")
            return lines[0][1], lines[1][1], lines[2][1]
        if key.startswith("co_"):
            lines = next(preset.lines for preset in PRESETS if preset.key == "co_ka12")
            return lines[0][1], lines[1][1], None
        return None

    def _find_companions(self, scan_uid: str, kind: str) -> None:
        """Match existing table rows by the radiation's expected 2θ ratio."""

        item = self.items.get(scan_uid)
        lines = self._companion_lines()
        if (item is None or item.scan is None or lines is None
                or not is_two_theta(item.scan.axis_name)
                or (kind == "kb" and lines[2] is None)):
            return
        wavelength = lines[1] if kind == "ka2" else lines[2]
        table_peaks = [peak for peak in self._session_peaks
                       if peak.scan_uid == scan_uid
                       and peak.axis_name == item.scan.axis_name]
        primary = [peak for peak in table_peaks if peak.kind == "primary"]
        linked_parents = {peak.parent_number for peak in table_peaks if peak.kind == kind}
        possible: list[tuple[float, CompanionProposal]] = []
        for peak in primary:
            if peak.number in linked_parents:
                continue
            center = peak.source_center * item.x_scale + item.x_shift
            predicted = companion_two_theta(center, lines[0], wavelength)
            if not np.isfinite(predicted):
                continue
            distance = abs(predicted - center)
            for candidate in primary:
                if candidate.number == peak.number:
                    continue
                measured = candidate.source_center * item.x_scale + item.x_shift
                width = max(peak.source_fwhm, candidate.source_fwhm) * abs(item.x_scale)
                tolerance = min(max(0.02, 0.35 * width), 0.45 * distance)
                error = abs(measured - predicted)
                if error <= tolerance:
                    possible.append((error / tolerance, CompanionProposal(
                        peak.number, center, predicted, measured, candidate.number,
                    )))
        # The best positional match wins; a table row participates in at most
        # one new pair per search. No spectra are fitted or added here.
        possible.sort(key=lambda entry: entry[0])
        used: set[int] = set()
        proposals: list[CompanionProposal] = []
        for _score, proposal in possible:
            if proposal.parent_number in used or proposal.existing_number in used:
                continue
            proposals.append(proposal)
            used.update((proposal.parent_number, proposal.existing_number))
        if not proposals:
            label = "Kα2" if kind == "ka2" else "Kβ"
            QMessageBox.information(self, localised("Peak search", "Recherche de pics", "Поиск пиков"),
                                    localised(f"No matching {label} peaks in this table.",
                                              f"Aucun pic {label} correspondant dans ce tableau.",
                                              f"В этой таблице нет пиков, совпадающих с {label}."))
            return
        dialog = CompanionReviewDialog(kind, proposals, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = {proposals[index].existing_number: proposals[index].parent_number
                  for index in dialog.chosen_indices()}
        if not chosen:
            return
        self._session_peaks = [replace(
            peak, kind=kind, parent_number=chosen[peak.number]
        ) if peak.scan_uid == scan_uid and peak.number in chosen else peak
            for peak in self._session_peaks]
        self._refresh_peak_table()
        self._draw(preserve_view=True)

    def _refresh_peak_table(self) -> None:
        for uid, dialog in self._indexing_dialogs.items():
            if dialog.source_signature != self._indexing_signature(uid):
                dialog.invalidate()
        selected = self._selected_item()
        self.peak_do_fit_button.setEnabled(bool(
            selected is not None and selected.scan is not None
            and any(p.scan_uid == selected.uid and p.axis_name == selected.scan.axis_name
                    for p in self._session_peaks)))
        lines = self._companion_lines()
        for uid, dialog in self._peak_table_dialogs.items():
            item = self.items.get(uid)
            if item is not None and item.scan is not None:
                dialog.set_sum_visible(self._show_peak_sum.get(uid, True))
                dialog.refresh(
                    self._session_peaks, item,
                    wavelength=self.viewer_state.plot.display_wavelength,
                    display_mode=self.viewer_state.plot.x_display_mode,
                    copper=lines is not None and lines[2] is not None
                           and is_two_theta(item.scan.axis_name),
                    alpha2_available=lines is not None
                           and is_two_theta(item.scan.axis_name),
                    phases=[(phase.uid, phase.name) for phase in self.store.documents.values()
                            if phase.kind in {CIF, CELL_PHASE}],
                )

    def _set_peak_sum_visible(self, uid: str, visible: bool) -> None:
        self._show_peak_sum[uid] = bool(visible)
        self._draw(preserve_view=True)

    def _do_peak_fit(self, uid: str) -> None:
        """Refit the accepted peaks, including any manually drawn additions."""
        if self._refit_session_peaks(uid, show_error=True):
            self._refresh_peak_table()
            self._draw(preserve_view=True)
            self.status.setText(localised(
                "Joint peak fit updated.", "Ajustement conjoint des pics mis à jour.",
                "Совместная подгонка пиков обновлена.",
            ))

    def _do_selected_peak_fit(self) -> None:
        uid = self._selected_uid()
        if uid is not None:
            self._do_peak_fit(uid)

    def _refit_session_peaks(self, uid: str, *, show_error: bool = False,
                             automatic: bool = False,
                             changed_numbers: set[int] | None = None) -> bool:
        item = self.items.get(uid)
        if item is None or item.scan is None:
            return False
        peaks = [peak for peak in self._session_peaks
                 if peak.scan_uid == uid and peak.axis_name == item.scan.axis_name]
        if not peaks:
            return False
        x = np.asarray(item.scan.x, dtype=float)
        y = np.asarray(item.scan.y, dtype=float)
        background = self._background_for(item)
        if automatic and background is None:
            # Local baselines from separate previews are not a shared model.
            # Leave these accepted proposals untouched until Do Fit is invoked.
            return False
        if background is not None:
            base = background.values(x)
        else:
            # Retain the local baseline already accepted with each peak when
            # no measurement-wide background has been calculated yet.
            total = np.zeros_like(x)
            count = np.zeros_like(x)
            for peak in peaks:
                order = np.argsort(peak.source_x)
                local_x = np.asarray(peak.source_x)[order]
                inside = (x >= local_x[0]) & (x <= local_x[-1])
                total[inside] += np.interp(
                    x[inside], local_x, np.asarray(peak.source_background)[order])
                count[inside] += 1
            if not np.any(count):
                return False
            anchors = np.flatnonzero(count)
            base = np.interp(x, x[anchors], total[anchors] / count[anchors])
        # Peaks with overlapping neighbourhoods share a fit. A new group
        # does not force every earlier peak in a distant part of the scan to
        # be optimised again.
        step = float(np.median(np.diff(x)))
        groups: list[tuple[float, float, list[SessionPeak]]] = []
        for peak in sorted(peaks, key=lambda p: p.source_center):
            radius = max(2.5 * peak.source_fwhm, .2, 15.0 * step)
            left, right = peak.source_center - radius, peak.source_center + radius
            if groups and left <= groups[-1][1]:
                previous_left, previous_right, members = groups[-1]
                members.append(peak)
                groups[-1] = previous_left, max(previous_right, right), members
            else:
                groups.append((left, right, [peak]))
        if changed_numbers is not None:
            groups = [group for group in groups
                      if any(peak.number in changed_numbers for peak in group[2])]
        if not groups:
            return False

        from scipy.special import voigt_profile

        def component(values: np.ndarray, peak: SessionPeak) -> np.ndarray:
            if peak.source_sigma > 0 and peak.source_gamma > 0:
                return peak.source_height * voigt_profile(
                    values - peak.source_center, peak.source_sigma, peak.source_gamma,
                ) / voigt_profile(0.0, peak.source_sigma, peak.source_gamma)
            order = np.argsort(peak.source_x)
            local_x = np.asarray(peak.source_x)[order]
            return np.interp(values, local_x,
                             np.asarray(peak.source_profile)[order], left=0.0, right=0.0)

        updated: dict[int, SessionPeak] = {}
        working = {peak.number: peak for peak in peaks}
        for _left, _right, group in groups:
            group_ids = {peak.number for peak in group}
            other = sum((component(x, peak) for peak in working.values()
                         if peak.number not in group_ids), np.zeros_like(x))
            seeds = []
            for peak in group:
                width = max(float(peak.source_fwhm), step)
                seeds.append((peak.source_center, peak.source_height,
                              peak.source_sigma or width / 3.0,
                              peak.source_gamma or width / 6.0))
            try:
                result = refit_voigt_peaks(
                    x, y - other, base, seeds,
                    refine_background=background is not None)
            except (ValueError, XRDDataError) as exc:
                if show_error:
                    QMessageBox.warning(self, "Do Fit", self._peak_fit_error(exc))
                return False
            for peak, fit in zip(group, result.peaks):
                replacement = replace(
                    peak, source_center=fit.center, source_x=result.x,
                    source_background=result.background, source_profile=fit.profile,
                    source_height=fit.height, source_area=fit.area,
                    source_fwhm=fit.fwhm, source_sigma=fit.sigma,
                    source_gamma=fit.gamma,
                )
                updated[peak.number] = replacement
                working[peak.number] = replacement
            if background is not None and result.background_delta is not None:
                background = background.with_fit_correction(result.x, result.background_delta)
                base = background.values(x)
        if background is not None:
            self._backgrounds[(uid, item.scan.axis_name)] = background
        self._session_peaks = [updated.get(peak.number, peak) for peak in self._session_peaks]
        return True

    def _set_peak_fill(self, uid: str, number: int, filled: bool) -> None:
        self._session_peaks = [replace(peak, filled=filled)
                               if peak.scan_uid == uid and peak.number == number else peak
                               for peak in self._session_peaks]
        self._draw(preserve_view=True)

    def _assign_peak_hkl(self, uid: str, number: int, phase_uid: str,
                         h: int, k: int, l: int) -> None:
        phase = self.store.documents.get(phase_uid)
        if phase is None or phase.kind not in {CIF, CELL_PHASE}:
            return
        assignment = (phase_uid, h, k, l)
        self._session_peaks = [
            replace(peak, hkl_assignments=(*peak.hkl_assignments, assignment))
            if peak.scan_uid == uid and peak.number == number
            and assignment not in peak.hkl_assignments else peak
            for peak in self._session_peaks
        ]
        self._refresh_peak_table()

    def _clear_peak_hkl(self, uid: str, number: int) -> None:
        self._session_peaks = [replace(peak, hkl_assignments=())
                               if peak.scan_uid == uid and peak.number == number else peak
                               for peak in self._session_peaks]
        self._refresh_peak_table()

    def _background_for(self, item: PlotItem) -> BackgroundAnchors | None:
        if item.scan is None:
            return None
        return self._backgrounds.get((item.uid, item.scan.axis_name))

    def _update_background_controls(self) -> None:
        item = self._selected_item()
        valid = item is not None and item.scan is not None
        background = self._background_for(item) if valid else None
        self.background_spacing_spin.blockSignals(True)
        if background is not None:
            self.background_spacing_spin.setValue(background.spacing)
        self.background_spacing_spin.blockSignals(False)
        self.background_spacing_spin.setEnabled(valid)
        self.background_auto_button.setEnabled(valid)
        self.background_exclude_button.setEnabled(bool(background and item.visible))
        self.background_reset_button.setEnabled(background is not None)

    def calculate_background(self) -> None:
        item = self._selected_item()
        if item is None or item.scan is None:
            return
        key = item.uid, item.scan.axis_name
        old = self._backgrounds.get(key)
        try:
            estimate = estimate_background(
                item.scan.x, item.scan.y, axis_name=item.scan.axis_name,
                spacing=self.background_spacing_spin.value(),
                excluded=old.excluded if old is not None else (),
            )
            self._backgrounds[key] = (replace(estimate, local_levels=old.local_levels)
                                      if old is not None else estimate)
        except ValueError as exc:
            QMessageBox.warning(self, localised("Background", "Fond", "Фон"), str(exc))
            return
        if self._pending_peak_fit is not None and self._pending_peak_fit.uid == item.uid:
            self._discard_peak_preview()
        self._update_background_controls()
        self._draw(preserve_view=True)

    def _background_spacing_changed(self, _value: float) -> None:
        item = self._selected_item()
        if item is not None and self._background_for(item) is not None:
            self.calculate_background()

    def select_background_gap(self) -> None:
        item = self._selected_item()
        if item is None or not item.visible or self._background_for(item) is None:
            return
        self._begin_peak_selection(item.uid)
        if self._fit_uid == item.uid:
            self._background_selection_uid = item.uid
            self.status.setText(localised(
                "Select an X region whose background anchors should be removed.",
                "Sélectionnez la plage X des points de fond à retirer.",
                "Выделите область X, из которой нужно убрать опорные точки фона."))

    def _exclude_background_region(self, item: PlotItem, low: float, high: float) -> None:
        background = self._background_for(item)
        if background is None:
            return
        if self.viewer_state.plot.x_display_mode == "d" and is_two_theta(item.scan.axis_name):
            converted = d_to_two_theta(np.asarray([low, high]),
                                       self.viewer_state.plot.display_wavelength)
            if not np.all(np.isfinite(converted)):
                return
            low, high = sorted(map(float, converted))
        if item.x_scale == 0:
            return
        low, high = sorted(((low - item.x_shift) / item.x_scale,
                            (high - item.x_shift) / item.x_scale))
        key = item.uid, item.scan.axis_name
        self._backgrounds[key] = background.without_region(low, high)
        if self._pending_peak_fit is not None and self._pending_peak_fit.uid == item.uid:
            self._discard_peak_preview()
        self._draw(preserve_view=True)

    def clear_background(self) -> None:
        item = self._selected_item()
        if item is None or item.scan is None:
            return
        self._backgrounds.pop((item.uid, item.scan.axis_name), None)
        if self._pending_peak_fit is not None and self._pending_peak_fit.uid == item.uid:
            self._discard_peak_preview()
        self._update_background_controls()
        self._draw(preserve_view=True)

    def _background_curve(self, item: PlotItem, curve_index: int, *, anchors=False):
        background = self._background_for(item)
        if background is None or item.scan is None:
            return None
        if anchors:
            source_x, source_y = background.active_anchors()
        else:
            x = item.scan.x
            # Decimate the visible line to avoid doubling the plot workload.
            step = max(1, x.size // 5000)
            indices = np.unique(np.r_[np.arange(0, x.size, step), x.size - 1])
            source_x = x[indices]
            source_y = background.values(source_x)
        display_x = self._display_x_values(item, source_x * item.x_scale + item.x_shift)
        physical_y = source_y * item.y_factor + item.y_shift
        display_y = transformed_intensity(physical_y, self.viewer_state.plot.intensity_scale)
        display_y += curve_index * self.viewer_state.plot.vertical_offset
        return display_x, display_y

    def _remove_session_peak(self, number: int, scan_uid: str | None = None) -> None:
        self._session_peaks = [replace(p, kind="primary", parent_number=None)
                               if p.parent_number == number and
                               (scan_uid is None or p.scan_uid == scan_uid)
                               else p for p in self._session_peaks
                               if not (p.number == number and
                                       (scan_uid is None or p.scan_uid == scan_uid))]
        self._refresh_peak_table()
        self._draw(preserve_view=True)

    def _clear_session_peaks(self, scan_uid: str | None = None) -> None:
        if scan_uid is None:
            self._session_peaks.clear()
        else:
            self._session_peaks = [p for p in self._session_peaks if p.scan_uid != scan_uid]
        self._refresh_peak_table()
        self._draw(preserve_view=True)

    def _peak_shades(self, item: PlotItem, curve_index: int):
        """Return shading arrays transformed with the current scan and plot settings."""

        if item.scan is None:
            return
        mode = self.viewer_state.plot.intensity_scale
        offset = curve_index * self.viewer_state.plot.vertical_offset
        current_background = self._background_for(item)
        for peak in self._session_peaks:
            if (peak.scan_uid != item.uid or peak.axis_name != item.scan.axis_name
                    or not peak.filled):
                continue
            x = self._display_x_values(item, peak.source_x * item.x_scale + item.x_shift)
            baseline_source = (current_background.values(peak.source_x)
                               if current_background is not None else peak.source_background)
            baseline = baseline_source * item.y_factor + item.y_shift
            top = (baseline_source + peak.source_profile) * item.y_factor + item.y_shift
            low = transformed_intensity(baseline, mode) + offset
            high = transformed_intensity(top, mode) + offset
            valid = (np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
                     & (peak.source_profile > .001 * peak.source_height))
            if np.count_nonzero(valid) >= 3:
                yield x[valid], low[valid], high[valid], PEAK_COLOURS[peak.kind]

    def _accepted_peak_profiles(self, item: PlotItem, source_x: np.ndarray, *,
                                omit_number: int | None = None) -> np.ndarray:
        """Evaluate existing components when a newly drawn peak is fitted."""
        from scipy.special import voigt_profile

        total = np.zeros_like(source_x, dtype=float)
        for peak in self._session_peaks:
            if (item.scan is None or peak.scan_uid != item.uid
                    or peak.axis_name != item.scan.axis_name
                    or peak.number == omit_number):
                continue
            if peak.source_sigma > 0 and peak.source_gamma > 0:
                total += peak.source_height * voigt_profile(
                    source_x - peak.source_center, peak.source_sigma, peak.source_gamma,
                ) / voigt_profile(0.0, peak.source_sigma, peak.source_gamma)
            else:
                order = np.argsort(peak.source_x)
                support_x = np.asarray(peak.source_x)[order]
                inside = (source_x >= support_x[0]) & (source_x <= support_x[-1])
                total[inside] += np.interp(
                    source_x[inside], support_x, np.asarray(peak.source_profile)[order])
        return total

    def _peak_sum_curve(self, item: PlotItem, curve_index: int):
        """Return the shared baseline plus all confirmed components on scan X."""
        if (item.scan is None or not self._show_peak_sum.get(item.uid, True)):
            return None
        peaks = [peak for peak in self._session_peaks
                 if peak.scan_uid == item.uid and peak.axis_name == item.scan.axis_name]
        if not peaks:
            return None
        source_x = np.asarray(item.scan.x, dtype=float)
        valid = np.isfinite(source_x)
        support = np.zeros(source_x.shape, dtype=bool)
        profile_sum = np.zeros(source_x.shape, dtype=float)
        local_background = np.zeros(source_x.shape, dtype=float)
        background_count = np.zeros(source_x.shape, dtype=int)
        common = self._background_for(item)
        for peak in peaks:
            order = np.argsort(peak.source_x)
            fit_x = np.asarray(peak.source_x, dtype=float)[order]
            inside = valid & (source_x >= fit_x[0]) & (source_x <= fit_x[-1])
            if not np.any(inside):
                continue
            support |= inside
            profile_sum[inside] += np.interp(
                source_x[inside], fit_x, peak.source_profile[order]
            )
            if common is None:
                local_background[inside] += np.interp(
                    source_x[inside], fit_x, peak.source_background[order]
                )
                background_count[inside] += 1
        if not np.any(support):
            return None
        baseline = (common.values(source_x) if common is not None else
                    np.divide(local_background, background_count,
                              out=np.zeros_like(local_background),
                              where=background_count > 0))
        intensity = (baseline + profile_sum) * item.y_factor + item.y_shift
        y = transformed_intensity(intensity, self.viewer_state.plot.intensity_scale)
        y += curve_index * self.viewer_state.plot.vertical_offset
        y[~support] = np.nan
        x = self._display_x_values(item, source_x * item.x_scale + item.x_shift)
        return x, y

    def _fit_voigt_bounds(self, item: PlotItem, x_low: float, x_high: float) -> None:
        if self.viewer_state.plot.x_display_mode == "d" and is_two_theta(item.scan.axis_name):
            converted = d_to_two_theta(
                np.asarray([x_low, x_high], dtype=float),
                self.viewer_state.plot.display_wavelength,
            )
            if not np.all(np.isfinite(converted)):
                QMessageBox.warning(self, localised("Peak search", "Recherche de pics", "Поиск пиков"),
                                    localised("Choose a valid d-spacing range.",
                                              "Sélectionnez une plage d valide.",
                                              "Выберите допустимый интервал межплоскостных расстояний."))
                return
            x_low, x_high = sorted(map(float, converted))
        values_x, values_y = item.display_arrays()
        x_low, x_high = sorted((x_low, x_high))
        selection = np.isfinite(values_x) & (values_x >= x_low) & (values_x <= x_high)
        selected_x = values_x[selection]
        step = (float(np.median(np.diff(np.sort(selected_x))))
                if selected_x.size >= 3 else 0.0)
        # The dragged interval chooses centres. Include neighbouring data in
        # the actual fit so peak tails are not truncated at the drag boundary.
        margin = max(8.0 * step, 0.25 * (x_high - x_low))
        mask = (np.isfinite(values_x) & (values_x >= x_low - margin)
                & (values_x <= x_high + margin))
        try:
            background = self._background_for(item)
            fitted_background = None
            source_x = (values_x[mask] - item.x_shift) / item.x_scale
            accepted = self._accepted_peak_profiles(item, source_x) * item.y_factor
            if background is not None:
                fitted_background = background.values(source_x) * item.y_factor + item.y_shift
            result = fit_voigt_region(values_x[mask], values_y[mask] - accepted,
                                      background=fitted_background,
                                      search_bounds=(x_low, x_high))
        except Exception as exc:
            if (isinstance(exc, XRDDataError) and exc.code == "peak_fit_flat"
                    and any(peak.scan_uid == item.uid
                            and peak.axis_name == item.scan.axis_name
                            and x_low <= peak.source_center * item.x_scale + item.x_shift <= x_high
                            for peak in self._session_peaks)):
                self.status.setText(localised(
                    "No new peaks in this range.",
                    "Aucun nouveau pic dans cette plage.",
                    "В этой области новых пиков нет.",
                ))
                return
            QMessageBox.warning(
                self, localised("Peak search", "Recherche de pics", "Поиск пиков"),
                self._peak_fit_error(exc),
            )
            return
        source_x = (result.x - item.x_shift) / item.x_scale
        self._pending_peak_fit = PendingPeakFit(
            uid=item.uid,
            axis_name=item.scan.axis_name,
            result=result,
            selected_source_x=(values_x[selection] - item.x_shift) / item.x_scale,
            selected_source_y=(values_y[selection] - item.y_shift) / item.y_factor,
            source_x=source_x,
            source_background=(result.background - item.y_shift) / item.y_factor,
            source_fitted=(result.fitted + np.interp(result.x, values_x[mask], accepted)
                           - item.y_shift) / item.y_factor,
            source_centers=tuple((peak.center - item.x_shift) / item.x_scale
                                 for peak in result.peaks),
            fit_x_scale=item.x_scale,
            fit_y_factor=item.y_factor,
        )
        for check in self.peak_preview_checks:
            check.setChecked(True)
        self._update_peak_preview_bar()
        self._show_pending_peak_fit()
        self.status.setText(localised(
            "Inspect the fitted curve and confirm the peaks with ✓, or discard with ✕.",
            "Examinez la courbe ajustée ; confirmez avec ✓ ou annulez avec ✕.",
            "Проверьте кривую подгонки; подтвердите пики ✓ или отмените ✕.",
        ))

    def _fit_drawn_bounds(self, item: PlotItem, x_start: float, y_start: float,
                          x_end: float, y_end: float) -> None:
        """Use a drawn top-to-background gesture to seed one fixed-background fit."""
        if item.scan is None or item.x_scale == 0 or item.y_factor == 0:
            return
        try:
            x_gesture = np.asarray([x_start, x_end], dtype=float)
            if (self.viewer_state.plot.x_display_mode == "d"
                    and is_two_theta(item.scan.axis_name)):
                x_gesture = d_to_two_theta(
                    x_gesture, self.viewer_state.plot.display_wavelength)
            x_gesture = (x_gesture - item.x_shift) / item.x_scale
            visible = self.viewer_state.visible_scans()
            offset = next((index * self.viewer_state.plot.vertical_offset
                           for index, scan in enumerate(visible)
                           if scan.uid == item.uid), 0.0)
            display_y = np.asarray([y_start, y_end], dtype=float) - offset
            mode = self.viewer_state.plot.intensity_scale
            if mode == "sqrt":
                display_y = np.maximum(display_y, 0.0) ** 2
            elif mode == "square":
                display_y = np.sqrt(np.maximum(display_y, 0.0))
            source_y = (display_y - item.y_shift) / item.y_factor
            center, dragged_x = map(float, x_gesture)
            top, level = map(float, source_y)
            source_x = np.asarray(item.scan.x, dtype=float)
            source_intensity = np.asarray(item.scan.y, dtype=float)
            sorted_x = np.sort(source_x[np.isfinite(source_x)])
            step = float(np.median(np.diff(sorted_x)))
            half_width = abs(dragged_x - center)
            if (not np.all(np.isfinite((*x_gesture, *source_y)))
                    or step <= 0 or half_width < 2 * step
                    or center <= sorted_x[0] or center >= sorted_x[-1]
                    or top <= level):
                raise ValueError(localised(
                    "Draw the peak from its top toward a lower background level, "
                    "with a visible horizontal width.",
                    "Dessinez le pic du sommet vers un fond plus bas, "
                    "avec une largeur horizontale visible.",
                    "Проведите от вершины пика к более низкому уровню фона "
                    "и задайте его ширину по горизонтали.",
                ))
            background = self._background_for(item)
            if background is None:
                background = estimate_background(
                    source_x, source_intensity, axis_name=item.scan.axis_name,
                    spacing=self.background_spacing_spin.value(),
                )
            adjusted = background.with_local_level(
                center, level, radius=max(4 * half_width, 3 * step),
                excluded_half_width=max(1.5 * half_width, 2 * step),
            )
            window = max(4 * half_width, 12 * step)
            mask = (np.isfinite(source_x) & np.isfinite(source_intensity)
                    & (source_x >= center - window)
                    & (source_x <= center + window))
            replacing = next((peak for peak in self._session_peaks
                              if peak.scan_uid == item.uid
                              and peak.axis_name == item.scan.axis_name
                              and abs(peak.source_center - center) <= 2 * step), None)
            previous_profiles = self._accepted_peak_profiles(
                item, source_x[mask],
                omit_number=replacing.number if replacing is not None else None)
            result = draw_voigt_peak(
                source_x[mask], adjusted.values(source_x[mask]),
                center=center, half_width=half_width, height=top - level)
        except (ValueError, XRDDataError) as exc:
            QMessageBox.warning(
                self, localised("Draw peak", "Dessiner un pic", "Нарисовать пик"),
                self._peak_fit_error(exc),
            )
            return
        self._pending_peak_fit = PendingPeakFit(
            uid=item.uid, axis_name=item.scan.axis_name, result=result,
            selected_source_x=source_x[mask],
            selected_source_y=source_intensity[mask], source_x=result.x,
            source_background=adjusted.values(result.x),
            source_fitted=result.fitted + np.interp(
                result.x, source_x[mask], previous_profiles),
            source_centers=(result.peaks[0].center,), fit_x_scale=1.0,
            fit_y_factor=1.0, background_after=adjusted, manual=True,
            replace_number=replacing.number if replacing is not None else None,
        )
        for check in self.peak_preview_checks:
            check.setChecked(True)
        self._update_peak_preview_bar()
        self._show_pending_peak_fit()
        self.status.setText(localised(
            "Review the drawn peak and corrected background; confirm to insert them.",
            "Vérifiez le pic dessiné et le fond corrigé ; confirmez leur insertion.",
            "Проверьте нарисованный пик и исправленный фон; подтвердите их добавление.",
        ))

    def _begin_peak_selection(
        self,
        uid: str,
        callback: Callable[[float, float], None] | None = None,
        *,
        manual: bool = False,
    ) -> None:
        item = self.items.get(uid)
        if item is None or item.scan is None:
            return
        if self.plot_renderer == "matplotlib" and self.toolbar.mode:
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
        self._fit_callback = callback
        self._manual_peak_active = manual
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.set_zoom_enabled(False)
            self.pyqtgraph_plot.set_peak_selection_enabled(not manual)
            self.pyqtgraph_plot.set_manual_peak_enabled(manual)
            self.status.setText(
                localised(
                    "Drag a rectangle around one peak on the main plot.",
                    "Tracez un rectangle autour d’un pic sur le graphique principal.",
                    "Выделите прямоугольником один пик на основном графике.",
                )
            )
            return
        if manual:
            for signal, handler in (("button_press_event", self._manual_plot_press),
                                    ("motion_notify_event", self._manual_plot_move),
                                    ("button_release_event", self._manual_plot_release)):
                self._manual_mpl_cids.append(self.canvas.mpl_connect(signal, handler))
            return
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

    def _manual_plot_press(self, event) -> None:
        if (self._manual_peak_active and event.inaxes is self.scan_axis
                and event.button == 1 and event.xdata is not None
                and event.ydata is not None):
            self._manual_mpl_start = float(event.xdata), float(event.ydata)

    def _manual_plot_move(self, event) -> None:
        if (self._manual_mpl_start is None or event.inaxes is not self.scan_axis
                or event.xdata is None or event.ydata is None):
            return
        start_x, start_y = self._manual_mpl_start
        end_x, end_y = float(event.xdata), float(event.ydata)
        width = abs(end_x - start_x)
        if width <= 0 or end_y >= start_y:
            return
        curve_x = np.linspace(start_x - 3 * width, start_x + 3 * width, 121)
        profile, _sigma, _gamma = drawn_voigt_profile(
            curve_x, center=start_x, half_width=width, height=start_y - end_y)
        curve_y = end_y + profile
        if self._fit_artists:
            self._fit_artists[0].set_data(curve_x, curve_y)
        else:
            self._fit_artists.extend(self.scan_axis.plot(
                curve_x, curve_y, color="orange", linestyle="--", linewidth=1.5,
                zorder=8,
            ))
        self.canvas.draw_idle()

    def _manual_plot_release(self, event) -> None:
        start = self._manual_mpl_start
        self._manual_mpl_start = None
        if (start is not None and event.inaxes is self.scan_axis
                and event.xdata is not None and event.ydata is not None):
            self._fit_peak_bounds(start[0], start[1],
                                  float(event.xdata), float(event.ydata))
        elif start is not None:
            self._clear_peak_fit_artifacts()

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
                "peak_find_points": localised(
                    "Select at least twelve measured points around a peak.",
                    "Sélectionnez au moins douze points de mesure autour d'un pic.",
                    "Выделите вокруг пика не менее двенадцати измеренных точек.",
                ),
                "peak_fit_flat": localised(
                    "The selected region contains no measurable peak.",
                    "La zone sélectionnée ne contient aucun pic mesurable.",
                    "В выбранной области нет измеримого пика.",
                ),
                "peak_fit_axis": localised(
                    "The selected scan coordinates must be distinct.",
                    "Les coordonnées sélectionnées doivent être distinctes.",
                    "Координаты выбранных точек должны различаться.",
                ),
                "peak_fit_many": localised(
                    "More than six peaks were found. Select a narrower interval.",
                    "Plus de six pics ont été trouvés. Réduisez la plage.",
                    "Найдено больше шести пиков. Выделите более узкую область.",
                ),
            }
            if exc.code in messages:
                return messages[exc.code]
        return str(exc)

    def _fit_peak_rectangle(self, click, release) -> None:
        if (
            click.xdata is None
            or click.ydata is None
            or release.xdata is None
            or release.ydata is None
        ):
            return
        self._fit_peak_bounds(
            float(click.xdata),
            float(click.ydata),
            float(release.xdata),
            float(release.ydata),
        )

    def _pyqtgraph_peak_region(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
    ) -> None:
        if self.pyqtgraph_plot is None:
            return
        self._fit_peak_bounds(
            x_start,
            self.pyqtgraph_plot.view_to_data_y(y_start),
            x_end,
            self.pyqtgraph_plot.view_to_data_y(y_end),
        )

    def _fit_peak_bounds(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
    ) -> None:
        uid = self._fit_uid
        callback = self._fit_callback
        peak_search = self._peak_search_active
        manual_peak = self._manual_peak_active
        background_uid = self._background_selection_uid
        item = self.items.get(uid) if uid else None
        self._cancel_peak_fit()
        if item is None or item.scan is None:
            return
        x_low, x_high = sorted((float(x_start), float(x_end)))
        if background_uid == uid:
            self._exclude_background_region(item, x_low, x_high)
            return
        if peak_search:
            self._fit_voigt_bounds(item, x_low, x_high)
            return
        if manual_peak:
            self._fit_drawn_bounds(item, x_start, y_start, x_end, y_end)
            return
        y_low, y_high = sorted((float(y_start), float(y_end)))
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
        old_x, old_y = self._active_scan_limits()
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

        fitted_display_y = transformed_intensity(
            fit_y,
            self.viewer_state.plot.intensity_scale,
        )
        centre_y = transformed_intensity(
            np.asarray([intensity]), self.viewer_state.plot.intensity_scale
        )[0]
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.show_peak_fit(
                x_values[mask],
                plotted_y[mask],
                fit_x,
                fitted_display_y,
                centre,
                centre_y,
            )
        else:
            selected_artist = self.scan_axis.scatter(
                x_values[mask], plotted_y[mask], color="green", s=18, zorder=7
            )
            fit_artist = self.scan_axis.plot(
                fit_x,
                fitted_display_y,
                color="orange",
                linewidth=2,
                zorder=7,
            )[0]
            centre_line = self.scan_axis.axvline(
                centre, color="red", linestyle="--", linewidth=1
            )
            centre_artist = self.scan_axis.scatter(
                [centre], [centre_y], color="red", s=55, zorder=8
            )
            self._fit_artists.extend(
                (selected_artist, fit_artist, centre_line, centre_artist)
            )
            self.scan_axis.set_xlim(old_x)
            self.scan_axis.set_ylim(old_y)
            self.canvas.draw_idle()
        self._set_limits(old_x, old_y)

        if callback is not None:
            callback(float(centre), float(intensity))
            self._clear_peak_fit_artifacts()
            return

        references = read_reference_peaks(_reference_peak_path())
        dialog = PeakTargetDialog(centre, intensity, references, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._clear_peak_fit_artifacts()
        if not accepted:
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

    def open_substrate_correction(self) -> None:
        item = self._selected_item()
        uid = self._selected_uid()
        if (
            item is None
            or item.scan is None
            or uid is None
            or not is_two_theta(item.scan.axis_name)
        ):
            return
        if self._substrate_dialog is not None:
            self._substrate_dialog.raise_()
            self._substrate_dialog.activateWindow()
            return
        dialog = SubstrateCorrectionDialog(
            item.name,
            read_reference_peaks(_reference_peak_path()),
            self,
        )
        dialog.add_peak_requested.connect(
            lambda: self._begin_peak_selection(uid, dialog.add_peak)
        )
        dialog.apply_requested.connect(
            lambda calibration: self._apply_substrate_calibration(
                uid,
                calibration,
            )
        )
        dialog.finished.connect(self._substrate_dialog_closed)
        self._substrate_dialog = dialog
        dialog.show()

    def _substrate_dialog_closed(self, _result: int) -> None:
        self._cancel_peak_fit()
        dialog = self._substrate_dialog
        self._substrate_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def _apply_substrate_calibration(
        self,
        uid: str,
        calibration: SubstrateCalibration,
    ) -> None:
        item = self.items.get(uid)
        if item is None or item.scan is None:
            return
        item.x_shift = (
            calibration.x_scale * item.x_shift + calibration.x_shift
        )
        item.x_scale = calibration.x_scale * item.x_scale
        self._load_processing_controls(item)
        self._selected_point = None
        self._draw(preserve_view=True)
        self.status.setText(
            localised(
                f"Angular calibration applied: 2θtrue = {calibration.x_scale:.9f} · 2θmeas {calibration.x_shift:+.8f}°.",
                f"Étalonnage angulaire appliqué : 2θvrai = {calibration.x_scale:.9f} · 2θmesuré {calibration.x_shift:+.8f}°.",
                f"Угловая коррекция применена: 2θист = {calibration.x_scale:.9f} · 2θизм {calibration.x_shift:+.8f}°.",
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
        self._populate_display_wavelengths()
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

    def open_selected_structure(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is not None and item.is_phase and self.on_open_structure is not None:
            self.on_open_structure(uid)

    def open_selected_poles(self) -> None:
        uid = self._selected_uid()
        item = self.items.get(uid) if uid else None
        if item is not None and item.is_phase and self.on_open_poles is not None:
            self.on_open_poles(uid)

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
            y_bounds = self._navigation_y_bounds
            physical_nonlinear_limits = bool(
                self.plot_renderer == "pyqtgraph"
                and self.pyqtgraph_plot is not None
                and self.viewer_state.plot.intensity_scale in {"sqrt", "square"}
            )
            if physical_nonlinear_limits:
                y_bounds = self.pyqtgraph_plot.physical_y_limits(y_bounds)
            y_limits = resolve_limits(
                y_bounds,
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
        if physical_nonlinear_limits and y_limits[0] < 0:
            QMessageBox.warning(
                self,
                tr("text.plot_limits"),
                tr("qt.plot_settings_invalid"),
            )
            return
        if physical_nonlinear_limits:
            y_limits = self.pyqtgraph_plot.display_y_limits(y_limits)
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
            x_values = self._display_x_values(item, x_values)
            plotted_y = transformed_intensity(physical_y, mode)
            if offset:
                plotted_y = plotted_y + index * offset
            result.append((item, x_values, plotted_y))
        return result

    def _display_x_values(
        self,
        item: PlotItem,
        values: np.ndarray,
    ) -> np.ndarray:
        if (
            self.viewer_state.plot.x_display_mode == "d"
            and item.scan is not None
            and is_two_theta(item.scan.axis_name)
        ):
            return two_theta_to_d(
                values,
                self.viewer_state.plot.display_wavelength,
            )
        return np.asarray(values, dtype=float)

    @staticmethod
    def _finite_x_limits(parts: list[np.ndarray]) -> tuple[float, float] | None:
        finite_parts = [part[np.isfinite(part)] for part in parts]
        finite_parts = [part for part in finite_parts if part.size]
        if not finite_parts:
            return None
        merged = np.concatenate(finite_parts)
        low, high = float(np.min(merged)), float(np.max(merged))
        if math.isclose(low, high, rel_tol=0.0, abs_tol=1.0e-15):
            padding = max(abs(low) * 0.01, 1.0e-6)
            return low - padding, high + padding
        return low, high

    def _phase_limits(self, scans: list[PlotItem]) -> tuple[float, float]:
        if scans and all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in scans
        ):
            parts = []
            for item in scans:
                x_values, _physical_y = item.display_arrays()
                parts.append(self._display_x_values(item, x_values))
            limits = self._finite_x_limits(parts)
            if limits is not None:
                return limits
        if self.viewer_state.plot.x_display_mode == "d":
            converted = two_theta_to_d(
                np.asarray(DEFAULT_PHASE_LIMITS, dtype=float),
                self.viewer_state.plot.display_wavelength,
            )
            return tuple(sorted(map(float, converted)))
        return DEFAULT_PHASE_LIMITS

    def _phase_request(
        self,
        limits: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple, tuple]:
        if self.viewer_state.plot.x_display_mode == "d":
            wavelength = float(self.viewer_state.plot.display_wavelength)
            converted = d_to_two_theta(np.asarray(limits, dtype=float), wavelength)
            if not np.all(np.isfinite(converted)):
                raise XRDDataError("viewer_d_limits")
            angular_limits = tuple(sorted(map(float, converted)))
            radiations = (("Display", wavelength, 1.0),)
            key = ("d", *angular_limits, radiations)
            return angular_limits, radiations, key
        angular_limits = tuple(sorted(map(float, limits)))
        radiations = tuple(self.radiation_settings.lines())
        key = ("angle", *angular_limits, radiations)
        return angular_limits, radiations, key

    def _phase_x(self, row: ReflectionRow) -> float:
        return (
            float(row.d)
            if self.viewer_state.plot.x_display_mode == "d"
            else float(row.two_theta)
        )

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
        angular_limits, radiations, key = self._phase_request(limits)
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
            min_two_theta=max(0.0, float(angular_limits[0])),
            max_two_theta=min(179.9, float(angular_limits[1])),
            min_intensity=0.1,
        )
        self._row_cache[item.uid] = (key, rows)
        return rows

    def _rows_for_display(
        self,
        item: PlotItem,
        limits: tuple[float, float],
    ) -> list[ReflectionRow]:
        """Return cached rows and start a non-blocking first calculation."""

        if item.structure is None:
            return []
        angular_limits, radiations, key = self._phase_request(limits)
        cached = self._row_cache.get(item.uid)
        if cached is not None and cached[0] == key:
            return cached[1]
        job_key = (item.uid, key)
        if job_key not in self._pending_row_jobs:
            factors = (
                {}
                if bool(getattr(item.structure, "cell_only", False))
                else self._ensure_factors()
            )
            self._pending_row_jobs.add(job_key)
            self._phase_row_error = ""
            task = _PhaseReflectionTask(
                item.uid,
                key,
                item.structure,
                factors,
                radiations,
                angular_limits,
            )
            task.signals.finished.connect(self._phase_rows_ready)
            self._phase_thread_pool.start(task)
        return []

    def _phase_rows_ready(self, uid, key, rows, error) -> None:
        self._pending_row_jobs.discard((uid, key))
        item = self.items.get(uid)
        if item is None:
            return
        self._row_cache[uid] = (key, list(rows))
        self._phase_row_error = "" if error is None else str(error)
        self._draw(preserve_view=True)

    def _phase_profile(
        self,
        rows: list[ReflectionRow],
        limits: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        angular_limits, _radiations, _key = self._phase_request(limits)
        minimum, maximum = angular_limits
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
        x_values = np.asarray(profile.x, dtype=float)
        y_values = np.asarray(profile.total, dtype=float)
        if self.viewer_state.plot.x_display_mode == "d":
            x_values = two_theta_to_d(
                x_values,
                self.viewer_state.plot.display_wavelength,
            )
            order = np.argsort(x_values)
            x_values = x_values[order]
            y_values = y_values[order]
        return x_values, y_values

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
            rows = self._rows_for_display(item, limits)
            baseline, amplitude = geometry
            profile_allowed = not bool(getattr(item.structure, "cell_only", False))
            if (
                rows
                and self.viewer_state.plot.phase_style == "profile"
                and profile_allowed
            ):
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
                        self._phase_x(row),
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
        self.phase_axis.set_xlabel(self._phase_axis_label())
        self.phase_axis.set_ylabel(tr("text.phases"))
        self.phase_axis.grid(True, axis="x", alpha=0.15)
        self.phase_axis.set_yticks([])
        for index, item in enumerate(phases):
            rows = self._rows_for_display(item, limits)
            baseline = float(len(phases) - index - 1)
            profile_allowed = not bool(getattr(item.structure, "cell_only", False))
            if (
                rows
                and self.viewer_state.plot.phase_style == "profile"
                and profile_allowed
            ):
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
                        self._phase_x(row),
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

    def _draw_pyqtgraph_overlay_phases(
        self,
        phases: list[PlotItem],
        limits: tuple[float, float],
    ) -> None:
        plot = self.pyqtgraph_plot
        if plot is None:
            return
        bands, occupied_height = overlay_phase_geometry(
            len(phases),
            single_line=self.viewer_state.plot.overlay_single_line,
            height_percent=self.viewer_state.plot.overlay_height_percent,
        )
        span = max(float(limits[1] - limits[0]), 1e-12)
        for index, (item, geometry) in enumerate(zip(phases, bands)):
            rows = self._rows_for_display(item, limits)
            baseline, amplitude = geometry
            profile_allowed = not bool(getattr(item.structure, "cell_only", False))
            if (
                rows
                and self.viewer_state.plot.phase_style == "profile"
                and profile_allowed
            ):
                grid, profile = self._phase_profile(rows, limits)
                plot.add_overlay_profile(
                    grid,
                    baseline,
                    baseline + profile * amplitude,
                    item.colour,
                )
            else:
                positions = np.asarray([self._phase_x(row) for row in rows], dtype=float)
                heights = np.asarray(
                    [
                        amplitude
                        if row.intensity is None
                        else amplitude * (0.18 + 0.82 * row.intensity / 100.0)
                        for row in rows
                    ],
                    dtype=float,
                )
                plot.add_overlay_sticks(
                    positions,
                    np.full(positions.shape, baseline),
                    baseline + heights,
                    item.colour,
                )
            if self.viewer_state.plot.overlay_single_line:
                label_fraction = (index + 0.5) / len(phases)
                label_x = limits[0] + label_fraction * span
                label_y = min(0.99, occupied_height + 0.01)
                anchor = (0.5, 1.0 if occupied_height > 0.9 else 0.0)
            else:
                band_height = amplitude / 0.75
                label_fraction = 0.01
                label_x = limits[0] + label_fraction * span
                label_y = baseline + band_height * 0.9
                anchor = (0.0, 1.0)
            plot.add_overlay_label(
                item.name,
                label_x,
                label_y,
                item.colour,
                anchor=anchor,
                x_fraction=label_fraction,
            )
        self._overlay_phase_top = occupied_height

    def _draw_pyqtgraph_separate_phases(
        self,
        phases: list[PlotItem],
        limits: tuple[float, float],
    ) -> None:
        plot = self.pyqtgraph_plot
        if plot is None:
            return
        label_x = limits[0] + 0.01 * max(limits[1] - limits[0], 1e-12)
        for index, item in enumerate(phases):
            rows = self._rows_for_display(item, limits)
            baseline = float(len(phases) - index - 1)
            profile_allowed = not bool(getattr(item.structure, "cell_only", False))
            if (
                rows
                and self.viewer_state.plot.phase_style == "profile"
                and profile_allowed
            ):
                grid, profile = self._phase_profile(rows, limits)
                plot.add_separate_profile(
                    grid,
                    baseline,
                    baseline + profile * 0.75,
                    item.colour,
                )
            else:
                positions = np.asarray([self._phase_x(row) for row in rows], dtype=float)
                heights = np.asarray(
                    [
                        0.85
                        if row.intensity is None
                        else 0.15 + 0.7 * row.intensity / 100.0
                        for row in rows
                    ],
                    dtype=float,
                )
                plot.add_separate_sticks(
                    positions,
                    np.full(positions.shape, baseline),
                    baseline + heights,
                    item.colour,
                )
            plot.add_separate_label(
                item.name,
                label_x,
                baseline + 0.87,
                item.colour,
            )

    def _viewer_x_label(
        self,
        arrays: list[tuple[PlotItem, np.ndarray, np.ndarray]],
    ) -> str:
        if self.viewer_state.plot.x_display_mode == "d":
            return "d, Å"
        visible_axes = {
            item.scan.axis_name
            for item, _x, _y in arrays
            if item.scan is not None
        }
        label = (
            _axis_label(next(iter(visible_axes)))
            if len(visible_axes) == 1
            else tr("text.scan_coordinate")
        )
        if len(visible_axes) == 1 and axis_has_degree_units(next(iter(visible_axes))):
            label += ", °"
        return label

    def _phase_axis_label(self) -> str:
        return (
            "d, Å"
            if self.viewer_state.plot.x_display_mode == "d"
            else "2θ, °"
        )

    def _draw_pyqtgraph(
        self,
        *,
        preserve_view: bool,
        preserve_y: bool = True,
    ) -> None:
        plot = self.pyqtgraph_plot
        if plot is None or self._drawing:
            return
        self._drawing = True
        old_x, old_y = plot.scan_limits()
        old_phase_x, _old_phase_y = plot.phase_limits()
        phase_was_visible = plot.phase_visible
        had_data = self._has_drawn_data
        mode = self.viewer_state.plot.intensity_scale
        arrays = self._visible_plot_arrays()
        scans = [item for item, _x, _y in arrays]
        phases = self.viewer_state.visible_phases()
        overlay_compatible = all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in scans
        )
        self._axes_linked = bool(scans) and overlay_compatible
        if (
            phases
            and not overlay_compatible
            and self.viewer_state.plot.phase_layout == "overlay"
        ):
            self.viewer_state.plot.phase_layout = "separate"
            self.phase_layout_combo.blockSignals(True)
            index = self.phase_layout_combo.findData("separate")
            if index >= 0:
                self.phase_layout_combo.setCurrentIndex(index)
            self.phase_layout_combo.blockSignals(False)
        separate = bool(phases) and self.viewer_state.plot.phase_layout == "separate"
        try:
            plot.begin_frame(
                scale_mode=mode,
                x_label=self._viewer_x_label(arrays),
                y_label=tr("qt.viewer_intensity"),
                separate=separate,
                phase_height_percent=self.viewer_state.plot.phase_height_percent,
                phase_x_label=self._phase_axis_label(),
            )
            legend_entries = []
            for curve_index, (item, x_values, plotted_y) in enumerate(arrays):
                curve = plot.add_scan_curve(
                    x_values,
                    plotted_y,
                    item.colour,
                    item.name,
                )
                legend_entries.append((curve, item.name))
                for shade_x, shade_low, shade_high, colour in self._peak_shades(item, curve_index):
                    plot.add_peak_shade(shade_x, shade_low, shade_high, colour)
                peak_sum = self._peak_sum_curve(item, curve_index)
                if peak_sum is not None:
                    plot.add_scan_curve(
                        *peak_sum, "#8e44ad",
                        localised("Fit + background", "Ajustement + fond",
                                  "Подгонка + фон"),
                        line_width=1.8, line_style=Qt.PenStyle.DashLine,
                    )
                background_curve = self._background_curve(item, curve_index)
                if background_curve is not None:
                    plot.add_scan_curve(*background_curve, "#c05a28", "Background",
                                        line_width=1.4,
                                        line_style=Qt.PenStyle.DashLine, alpha=0.85)
                    plot.add_scan_curve(*self._background_curve(item, curve_index,
                                                                 anchors=True),
                                        "#c05a28", "Background anchors", symbols=True)
            if arrays:
                self._navigation_x_bounds = self._finite_x_limits(
                    [x_values for _item, x_values, _y in arrays]
                ) or (0.0, 1.0)
                self._navigation_y_bounds = intensity_limits(
                    [values for _item, _x, values in arrays],
                    logarithmic=mode == "log",
                )
                plot.add_legend(legend_entries)
            else:
                self._navigation_x_bounds = (0.0, 1.0)
                self._navigation_y_bounds = (
                    (1.0, 10.0) if mode == "log" else (0.0, 1.0)
                )

            phase_limits = self._phase_limits(scans) if phases else DEFAULT_PHASE_LIMITS
            if phases and not arrays:
                self._navigation_x_bounds = phase_limits
            self._overlay_phase_top = 0.0
            phase_error = ""
            if phases:
                try:
                    if separate:
                        self._draw_pyqtgraph_separate_phases(phases, phase_limits)
                    else:
                        self._draw_pyqtgraph_overlay_phases(phases, phase_limits)
                except (OSError, ValueError, XRDDataError) as exc:
                    phase_error = str(exc)

            self.status.setText(
                phase_error or self._status_text(len(arrays), len(phases))
            )
            self._has_drawn_data = bool(arrays or phases)

            x_limits = self._navigation_x_bounds
            y_limits = self._navigation_y_bounds
            scan_has_content = bool(arrays or (phases and not separate))
            if preserve_view and had_data and scan_has_content:
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

            plot.set_axes_linked(self._axes_linked)
            plot.set_scan_range(x_limits, y_limits)
            if separate:
                phase_x = phase_limits
                if self._axes_linked:
                    phase_x = x_limits
                elif preserve_view and had_data:
                    previous_phase_x = old_phase_x if phase_was_visible else old_x
                    if (
                        all(np.isfinite(previous_phase_x))
                        and previous_phase_x[0] < previous_phase_x[1]
                    ):
                        phase_x = previous_phase_x
                plot.set_phase_range(
                    phase_x,
                    (-0.1, len(phases) + 0.05),
                )
            if not arrays and not phases:
                plot.show_placeholder(
                    self._status_text(0, 0),
                    x_limits,
                    y_limits,
                )
            self._update_phase_controls()
            self._refresh_selection_info()
            self._update_navigation_scrollbars()
        finally:
            self._drawing = False
            if self._pending_peak_fit is not None:
                self._show_pending_peak_fit()

    def _status_text(self, scan_count: int, phase_count: int) -> str:
        if self._pending_row_jobs:
            return localised(
                "Calculating phase reflections…",
                "Calcul des réflexions de phase…",
                "Расчёт отражений фазы…",
            )
        if self._phase_row_error:
            return self._phase_row_error
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
        if self.plot_renderer == "pyqtgraph":
            self._draw_pyqtgraph(
                preserve_view=preserve_view,
                preserve_y=preserve_y,
            )
            self._refresh_peak_table()
            return
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
        overlay_compatible = all(
            item.scan is not None and is_two_theta(item.scan.axis_name)
            for item in scans
        )
        self._axes_linked = bool(scans) and overlay_compatible
        if (
            phases
            and not overlay_compatible
            and self.viewer_state.plot.phase_layout == "overlay"
        ):
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
            for curve_index, (item, x_values, plotted_y) in enumerate(arrays):
                self.scan_axis.plot(
                    x_values,
                    plotted_y,
                    color=item.colour,
                    lw=1.15,
                    label=item.name,
                )
                for shade_x, shade_low, shade_high, colour in self._peak_shades(item, curve_index):
                    self.scan_axis.fill_between(
                        shade_x, shade_low, shade_high, color=colour, alpha=0.09,
                        linewidth=0, zorder=1.5,
                    )
                peak_sum = self._peak_sum_curve(item, curve_index)
                if peak_sum is not None:
                    self.scan_axis.plot(
                        *peak_sum, color="#8e44ad", linestyle="--",
                        linewidth=1.8, zorder=3,
                        label=localised("Fit + background", "Ajustement + fond",
                                        "Подгонка + фон"),
                    )
                background_curve = self._background_curve(item, curve_index)
                if background_curve is not None:
                    self.scan_axis.plot(*background_curve, color="#c05a28",
                                        linestyle="--", linewidth=1.4, alpha=0.85)
                    self.scan_axis.scatter(*self._background_curve(item, curve_index,
                                                                    anchors=True),
                                           color="#c05a28", s=9, zorder=2)

            if arrays:
                self._navigation_x_bounds = self._finite_x_limits(
                    [x_values for _item, x_values, _y in arrays]
                ) or (0.0, 1.0)
                self._navigation_y_bounds = intensity_limits(
                    [values for _item, _x, values in arrays],
                    logarithmic=mode == "log",
                )
                if len(arrays) <= MAX_LEGEND_ITEMS:
                    self.scan_axis.legend(loc="best", fontsize=8)
            else:
                self._navigation_x_bounds = (0.0, 1.0)
                self._navigation_y_bounds = (
                    (1.0, 10.0) if mode == "log" else (0.0, 1.0)
                )
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

            self.scan_axis.set_xlabel(self._viewer_x_label(arrays))
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
            if self._pending_peak_fit is not None:
                self._show_pending_peak_fit()
            self._refresh_peak_table()

    def _set_limits(
        self,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        self._drawing = True
        try:
            if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
                self.pyqtgraph_plot.set_scan_range(x_limits, y_limits)
                if self._axes_linked and self.pyqtgraph_plot.phase_visible:
                    _phase_x, phase_y = self.pyqtgraph_plot.phase_limits()
                    self.pyqtgraph_plot.set_phase_range(x_limits, phase_y)
                self._update_navigation_scrollbars()
                return
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

    def _active_x_navigation_limits(self) -> tuple[float, float]:
        if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
            if self.pyqtgraph_plot.phase_visible and not self.viewer_state.visible_scans():
                return self.pyqtgraph_plot.phase_limits()[0]
            return self.pyqtgraph_plot.scan_limits()[0]
        return tuple(self._x_navigation_axis().get_xlim())

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
            self._configure_scrollbar(
                self.x_scrollbar,
                self._navigation_x_bounds,
                self._active_x_navigation_limits(),
                vertical=False,
            )
            _scan_x, scan_y = self._active_scan_limits()
            self._configure_scrollbar(
                self.y_scrollbar,
                self._navigation_y_bounds,
                scan_y,
                vertical=True,
            )
        finally:
            self._syncing_scrollbars = False

    def _scrollbar_changed(self, dimension: str, value: int) -> None:
        if self._syncing_scrollbars or self._drawing:
            return
        bounds = self._navigation_x_bounds if dimension == "x" else self._navigation_y_bounds
        if dimension == "x":
            current = self._active_x_navigation_limits()
        else:
            _scan_x, current = self._active_scan_limits()
        low, high = scrolled_limits(
            bounds,
            current,
            "moveto",
            float(value) / SCROLL_RESOLUTION,
            vertical=dimension == "y",
        )
        self._drawing = True
        try:
            if self.plot_renderer == "pyqtgraph" and self.pyqtgraph_plot is not None:
                if dimension == "x":
                    if self.pyqtgraph_plot.phase_visible and not self.viewer_state.visible_scans():
                        _phase_x, phase_y = self.pyqtgraph_plot.phase_limits()
                        self.pyqtgraph_plot.set_phase_range((low, high), phase_y)
                    else:
                        _scan_x, scan_y = self.pyqtgraph_plot.scan_limits()
                        self.pyqtgraph_plot.set_scan_range((low, high), scan_y)
                        if self._axes_linked and self.pyqtgraph_plot.phase_visible:
                            _phase_x, phase_y = self.pyqtgraph_plot.phase_limits()
                            self.pyqtgraph_plot.set_phase_range((low, high), phase_y)
                else:
                    scan_x, _scan_y = self.pyqtgraph_plot.scan_limits()
                    self.pyqtgraph_plot.set_scan_range(scan_x, (low, high))
                self._update_navigation_scrollbars()
                return
            target_axis = self._x_navigation_axis() if dimension == "x" else self.scan_axis
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
        nearest: tuple[float, PlotItem, int] | None = None
        mode = self.viewer_state.plot.intensity_scale
        offset = self.viewer_state.plot.vertical_offset
        for curve_index, item in enumerate(self.viewer_state.visible_scans()):
            x_values, physical_y = item.display_arrays()
            x_values = self._display_x_values(item, x_values)
            plotted_y = transformed_intensity(physical_y, mode) + curve_index * offset
            finite_x = np.flatnonzero(np.isfinite(x_values))
            if not finite_x.size:
                continue
            insertion = int(
                finite_x[np.argmin(np.abs(x_values[finite_x] - event.xdata))]
            )
            for index in range(max(0, insertion - 2), min(len(x_values), insertion + 3)):
                if not np.isfinite(plotted_y[index]):
                    continue
                px, py = self.scan_axis.transData.transform(
                    (x_values[index], plotted_y[index])
                )
                distance = math.hypot(px - event.x, py - event.y)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, item, index)
        phase_nearest = None
        if (
            self.viewer_state.visible_phases()
            and self.viewer_state.plot.phase_layout == "overlay"
        ):
            phase_nearest = self._nearest_phase_reflection(event)
        if phase_nearest is not None and (
            nearest is None or phase_nearest[0] < nearest[0]
        ):
            _distance, item, row = phase_nearest
            self._selected_point = (item.uid, "phase", row)
            self._refresh_selection_info()
            return
        if nearest is None:
            return
        _distance, item, index = nearest
        self._selected_point = (item.uid, "scan", index)
        self._refresh_selection_info()

    def _pyqtgraph_plot_clicked(self, source: str, x_value: float, y_value: float) -> None:
        plot = self.pyqtgraph_plot
        if plot is None or self.plot_renderer != "pyqtgraph":
            return
        if source == "phase":
            self._select_nearest_phase_reflection_pyqtgraph(
                x_value,
                y_value,
                separate=True,
            )
            return
        nearest: tuple[float, PlotItem, int] | None = None
        mode = self.viewer_state.plot.intensity_scale
        offset = self.viewer_state.plot.vertical_offset
        for curve_index, item in enumerate(self.viewer_state.visible_scans()):
            x_values, physical_y = item.display_arrays()
            x_values = self._display_x_values(item, x_values)
            plotted_y = transformed_intensity(physical_y, mode) + curve_index * offset
            finite_x = np.flatnonzero(np.isfinite(x_values))
            if not finite_x.size:
                continue
            insertion = int(
                finite_x[np.argmin(np.abs(x_values[finite_x] - x_value))]
            )
            for index in range(max(0, insertion - 2), min(len(x_values), insertion + 3)):
                candidate_y = plot.data_to_view_y(float(plotted_y[index]))
                if not math.isfinite(candidate_y):
                    continue
                distance = plot.scan_distance(
                    (float(x_values[index]), candidate_y),
                    (x_value, y_value),
                )
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, item, index)
        phase_nearest = None
        if (
            self.viewer_state.visible_phases()
            and self.viewer_state.plot.phase_layout == "overlay"
        ):
            phase_nearest = self._nearest_phase_reflection_pyqtgraph(
                x_value,
                plot.scan_fraction_y(y_value),
                separate=False,
            )
        if phase_nearest is not None and (
            nearest is None or phase_nearest[0] < nearest[0]
        ):
            _distance, item, row = phase_nearest
            self._selected_point = (item.uid, "phase", row)
            self._refresh_selection_info()
            return
        if nearest is None:
            return
        _distance, item, index = nearest
        self._selected_point = (item.uid, "scan", index)
        self._refresh_selection_info()

    def _select_nearest_phase_reflection_pyqtgraph(
        self,
        x_value: float,
        y_value: float,
        *,
        separate: bool,
    ) -> bool:
        nearest = self._nearest_phase_reflection_pyqtgraph(
            x_value,
            y_value,
            separate=separate,
        )
        if nearest is None:
            return False
        _distance, item, row = nearest
        self._selected_point = (item.uid, "phase", row)
        self._refresh_selection_info()
        return True

    def _nearest_phase_reflection_pyqtgraph(
        self,
        x_value: float,
        y_value: float,
        *,
        separate: bool,
    ) -> tuple[float, PlotItem, ReflectionRow] | None:
        plot = self.pyqtgraph_plot
        phases = self.viewer_state.visible_phases()
        if plot is None or not phases:
            return None
        try:
            limits = self._phase_limits(self.viewer_state.visible_scans())
        except (TypeError, ValueError, ArithmeticError):
            return None
        nearest: tuple[float, PlotItem, ReflectionRow] | None = None
        if separate:
            for phase_index, item in enumerate(phases):
                baseline = float(len(phases) - phase_index - 1)
                for row in self._rows_for(item, limits):
                    distance = plot.phase_distance(
                        (self._phase_x(row), baseline + 0.45),
                        (x_value, y_value),
                    )
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, item, row)
        else:
            bands, _occupied = overlay_phase_geometry(
                len(phases),
                single_line=self.viewer_state.plot.overlay_single_line,
                height_percent=self.viewer_state.plot.overlay_height_percent,
            )
            for item, (baseline, amplitude) in zip(phases, bands):
                for row in self._rows_for(item, limits):
                    distance = plot.overlay_distance(
                        (self._phase_x(row), baseline + 0.5 * amplitude),
                        (x_value, y_value),
                    )
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, item, row)
        return nearest

    @staticmethod
    def _full_hkl(row: ReflectionRow) -> str:
        if not row.equivalents:
            return row.hkl
        return "+".join(f"({h} {k} {l})" for h, k, l in row.equivalents)

    def _select_nearest_phase_reflection(self, event) -> bool:
        nearest = self._nearest_phase_reflection(event)
        if nearest is None:
            return False
        _distance, item, row = nearest
        self._selected_point = (item.uid, "phase", row)
        self._refresh_selection_info()
        return True

    def _nearest_phase_reflection(
        self,
        event,
    ) -> tuple[float, PlotItem, ReflectionRow] | None:
        phases = self.viewer_state.visible_phases()
        if not phases:
            return None
        try:
            limits = self._phase_limits(self.viewer_state.visible_scans())
        except (TypeError, ValueError, ArithmeticError):
            return None
        nearest: tuple[float, PlotItem, ReflectionRow] | None = None
        if event.inaxes is self.phase_axis:
            for phase_index, item in enumerate(phases):
                baseline = float(len(phases) - phase_index - 1)
                for row in self._rows_for(item, limits):
                    px, py = self.phase_axis.transData.transform(
                        (self._phase_x(row), baseline + 0.45)
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
                        (self._phase_x(row), baseline + 0.5 * amplitude)
                    )
                    distance = math.hypot(px - event.x, py - event.y)
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, item, row)
        return nearest

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
        displayed_x = self._display_x_values(item, x_values)
        value = displayed_x[index]
        d_suffix = self._d_suffix(item, float(x_values[index]))
        if self.viewer_state.plot.x_display_mode == "d":
            label = "d"
            unit = "Å"
            d_suffix = localised(
                f"; 2θ = {x_values[index]:.5f}°",
                f" ; 2θ = {x_values[index]:.5f}°",
                f"; 2θ = {x_values[index]:.5f}°",
            )
        self.selection_info.setText(
            tr(
                "viewer.scan_selection",
                name=item.name,
                axis=label,
                value=f"{value:.5f}",
                unit=unit,
                d_suffix=d_suffix,
                intensity=f"{physical_y[index]:.6g}",
            )
        )


__all__ = ["CollapsibleSection", "ViewerPage"]
