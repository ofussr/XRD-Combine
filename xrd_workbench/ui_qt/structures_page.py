"""Native PySide6 calculated-pattern and reflection-table pages."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..io.reflections import (
    read_scattering_factors,
    scattering_factor_path,
    write_reflection_csv,
)
from ..localization import localised, tr
from ..models.data_errors import XRDDataError
from ..models.diffraction import DiffractionStructure, ReflectionRow
from ..models.project import CELL_PHASE, CIF, STRUCTURES
from ..models.radiation import RadiationSettings
from ..services.diffraction import calculate_reflections, gaussian_powder_profile
from .radiation import RadiationSelector
from .structure_viewer import StructureViewerPage


ROW_INDEX_ROLE = int(Qt.ItemDataRole.UserRole)
SORT_VALUE_ROLE = ROW_INDEX_ROLE + 1


def _diffraction_error_text(error: XRDDataError) -> str:
    """Translate calculation errors at the Qt presentation boundary."""

    element = error.context.get("element", "")
    expression = error.context.get("expression", "")
    messages = {
        "diffraction_no_radiation": localised(
            "No spectral line is selected.",
            "Aucune raie spectrale n’est sélectionnée.",
            "Не выбрана ни одна спектральная линия.",
        ),
        "diffraction_limits": localised(
            "The limits must satisfy 0 ≤ minimum < maximum < 180°.",
            "Les limites doivent respecter 0 ≤ minimum < maximum < 180°.",
            "Требуется 0 ≤ минимум < максимум < 180°.",
        ),
        "diffraction_radiation_positive": localised(
            "Wavelengths and relative weights must be positive.",
            "Les longueurs d’onde et les poids relatifs doivent être positifs.",
            "Длины волн и относительные веса должны быть положительными.",
        ),
        "diffraction_factors_empty": localised(
            "Could not read the atomic scattering factors.",
            "Impossible de lire les facteurs de diffusion atomique.",
            "Не удалось прочитать коэффициенты атомного рассеяния.",
        ),
        "diffraction_factors_file_missing": localised(
            "f0_WaasKirf.dat was not found. Place it beside the application.",
            "f0_WaasKirf.dat est introuvable. Placez-le à côté de l’application.",
            "Не найден f0_WaasKirf.dat. Положите его в одну папку с программой.",
        ),
        "diffraction_factor_missing": localised(
            f"No atomic scattering factors are available for {element}.",
            f"Aucun facteur de diffusion atomique n’est disponible pour {element}.",
            f"Нет коэффициентов атомного рассеяния для элемента {element}.",
        ),
        "diffraction_singular_metric": localised(
            "The metric matrix is singular.",
            "La matrice métrique est singulière.",
            "Матрица метрики вырождена.",
        ),
        "diffraction_symmetry": localised(
            f"Could not parse symmetry operation {expression!r}.",
            f"Impossible d’analyser l’opération de symétrie {expression!r}.",
            f"Не удалось разобрать операцию симметрии {expression!r}.",
        ),
        "diffraction_profile_limits": localised(
            "The profile limits must be finite and increasing.",
            "Les limites du profil doivent être finies et croissantes.",
            "Границы профиля должны быть конечными и возрастать.",
        ),
        "diffraction_profile_fwhm": localised(
            "FWHM must be a positive finite number.",
            "La FWHM doit être un nombre fini positif.",
            "FWHM должен быть положительным конечным числом.",
        ),
    }
    return messages.get(error.code, str(error))


def _full_hkl(row: ReflectionRow) -> str:
    if row.equivalents:
        return "+".join(f"({h} {k} {l})" for h, k, l in row.equivalents)
    return row.hkl


def _reflection_headers() -> tuple[str, ...]:
    return (
        "hkl",
        "d, Å",
        "2θ, °",
        tr("text.line_2"),
        "λ, Å",
        tr("text.weight_2"),
        tr("text.multiplicity_2"),
        "Σ|F|^2",
        tr("text.irel"),
    )


class ReflectionTreeItem(QTreeWidgetItem):
    """Tree row whose numeric columns retain numeric sorting."""

    def __lt__(self, other) -> bool:
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        left = self.data(column, SORT_VALUE_ROLE)
        right = other.data(column, SORT_VALUE_ROLE)
        try:
            return left < right
        except TypeError:
            return str(left) < str(right)


class CalculationPage(QWidget):
    """Common document and scattering-factor state for Qt calculations."""

    def __init__(self, radiations_provider: Callable[[], list[tuple[str, float, float]]], parent=None) -> None:
        super().__init__(parent)
        self.radiations_provider = radiations_provider
        self.cif_document = None
        self.structure: DiffractionStructure | None = None
        self.rows: list[ReflectionRow] = []
        self._factors = None

    def _scattering_factors(self):
        if self.structure is not None and self.structure.cell_only:
            return {}
        if self._factors is None:
            self._factors = read_scattering_factors(scattering_factor_path())
        return self._factors

    @staticmethod
    def _number(field: QLineEdit) -> float:
        return float(field.text().strip().replace(",", "."))

    def _calculation_error(self, error: Exception) -> None:
        text = _diffraction_error_text(error) if isinstance(error, XRDDataError) else str(error)
        QMessageBox.critical(self, tr("text.calculation_error"), text)


class CalculatedPatternPage(CalculationPage):
    """Plot the established calculated sticks or Gaussian powder profile."""

    def __init__(self, radiations_provider, parent=None) -> None:
        super().__init__(radiations_provider, parent)
        self._build_ui()
        self.retranslate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        root.addWidget(self.path_label)

        self.settings_group = QGroupBox()
        settings = QGridLayout(self.settings_group)
        self.minimum_label = QLabel()
        self.maximum_label = QLabel()
        self.intensity_label = QLabel()
        self.fwhm_label = QLabel("FWHM, °")
        self.minimum_edit = QLineEdit("5")
        self.maximum_edit = QLineEdit("120")
        self.intensity_edit = QLineEdit("0.1")
        self.fwhm_edit = QLineEdit("0.15")
        for field in (
            self.minimum_edit,
            self.maximum_edit,
            self.intensity_edit,
            self.fwhm_edit,
        ):
            field.setMaximumWidth(80)
            field.setMinimumWidth(0)
        self.sticks_radio = QRadioButton()
        self.profile_radio = QRadioButton()
        self.profile_radio.setChecked(True)
        self.sticks_radio.toggled.connect(
            lambda checked: self.redraw() if checked else None
        )
        self.profile_radio.toggled.connect(
            lambda checked: self.redraw() if checked else None
        )
        self.calculate_button = QPushButton()
        self.calculate_button.clicked.connect(self.calculate)

        settings.addWidget(self.minimum_label, 0, 0)
        settings.addWidget(self.minimum_edit, 0, 1)
        settings.addWidget(self.maximum_label, 0, 2)
        settings.addWidget(self.maximum_edit, 0, 3)
        settings.addWidget(self.intensity_label, 0, 4)
        settings.addWidget(self.intensity_edit, 0, 5)
        settings.addWidget(self.fwhm_label, 0, 6)
        settings.addWidget(self.fwhm_edit, 0, 7)
        settings.addWidget(self.sticks_radio, 0, 8)
        settings.addWidget(self.profile_radio, 0, 9)
        settings.addWidget(self.calculate_button, 0, 10)
        settings.setColumnStretch(11, 1)
        root.addWidget(self.settings_group)

        self.figure = Figure(figsize=(9, 5), dpi=100, constrained_layout=True)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self.canvas, 1)
        toolbar_row = QHBoxLayout()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        toolbar_row.addWidget(self.toolbar)
        toolbar_row.addWidget(self.status_label, 1)
        root.addLayout(toolbar_row)

    def retranslate(self) -> None:
        self.settings_group.setTitle(tr("text.calculation"))
        self.minimum_label.setText(tr("text.two_theta_from"))
        self.maximum_label.setText(tr("text.to"))
        self.intensity_label.setText(tr("text.irel_2"))
        self.sticks_radio.setText(tr("text.sticks"))
        self.profile_radio.setText(tr("text.profile"))
        self.calculate_button.setText(tr("text.calculate"))
        if self.structure is None:
            self.path_label.setText(
                localised("No CIF loaded", "Aucun CIF chargé", "CIF не загружен")
            )
            self.status_label.clear()
            self.redraw()
        else:
            self.calculate(show_errors=False)

    def load_document(self, document) -> None:
        self.cif_document = document
        self.structure = document.diffraction
        self.path_label.setText(Path(document.source).name)
        if self.structure.cell_only:
            self.sticks_radio.setChecked(True)
        enabled = not self.structure.cell_only
        self.intensity_edit.setEnabled(enabled)
        self.fwhm_edit.setEnabled(enabled)
        self.profile_radio.setEnabled(enabled)
        self.calculate()

    def clear_document(self) -> None:
        self.cif_document = None
        self.structure = None
        self.rows = []
        self.path_label.setText(
            localised("No CIF loaded", "Aucun CIF chargé", "CIF не загружен")
        )
        self.intensity_edit.setEnabled(True)
        self.fwhm_edit.setEnabled(True)
        self.profile_radio.setEnabled(True)
        self.status_label.clear()
        self.redraw()

    def _parameters(self) -> tuple[float, float, float, float]:
        try:
            values = (
                self._number(self.minimum_edit),
                self._number(self.maximum_edit),
                self._number(self.intensity_edit),
                self._number(self.fwhm_edit),
            )
        except ValueError as error:
            raise ValueError(
                localised(
                    "The angle limits, intensity threshold and FWHM must be numeric.",
                    "Les limites angulaires, le seuil d’intensité et la FWHM doivent être numériques.",
                    "Границы углов, порог интенсивности и FWHM должны быть числами.",
                )
            ) from error
        if not math.isfinite(values[3]) or values[3] <= 0:
            raise ValueError(
                localised(
                    "FWHM must be a positive finite number.",
                    "La FWHM doit être un nombre fini positif.",
                    "FWHM должен быть положительным конечным числом.",
                )
            )
        return values

    def calculate(self, _checked: bool = False, *, show_errors: bool = True) -> None:
        if self.structure is None:
            if show_errors:
                QMessageBox.information(
                    self,
                    localised("No structure", "Aucune structure", "Нет структуры"),
                    localised(
                        "Open a CIF file first.",
                        "Ouvrez d’abord un fichier CIF.",
                        "Сначала откройте CIF-файл.",
                    ),
                )
            return
        try:
            minimum, maximum, threshold, _fwhm = self._parameters()
            self.rows = calculate_reflections(
                self.structure,
                self._scattering_factors(),
                self.radiations_provider(),
                minimum,
                maximum,
                threshold,
            )
        except Exception as error:
            if show_errors:
                self._calculation_error(error)
            return
        if self.structure.cell_only:
            self.status_label.setText(
                localised(
                    f"Reference positions: {len(self.rows)}; intensity is unavailable.",
                    f"Positions de référence : {len(self.rows)} ; intensité indisponible.",
                    f"Положений отражений: {len(self.rows)}; интенсивность недоступна.",
                )
            )
        else:
            self.status_label.setText(
                localised(
                    f"Calculated reflections: {len(self.rows)}",
                    f"Réflexions calculées : {len(self.rows)}",
                    f"Рассчитано отражений: {len(self.rows)}",
                )
            )
        self.redraw()

    def redraw(self) -> None:
        self.axis.clear()
        self.axis.set_xlabel("2θ, °")
        cell_only = bool(self.structure is not None and self.structure.cell_only)
        self.axis.set_ylabel(
            localised(
                "Reference positions",
                "Positions de référence",
                "Положения отражений",
            )
            if cell_only
            else localised(
                "Relative intensity, %",
                "Intensité relative, %",
                "Относительная интенсивность, %",
            )
        )
        self.axis.grid(True, alpha=0.2)
        if self.structure is None:
            self.axis.text(
                0.5,
                0.5,
                localised(
                    "Open a CIF file.",
                    "Ouvrez un fichier CIF.",
                    "Откройте CIF-файл.",
                ),
                transform=self.axis.transAxes,
                ha="center",
                va="center",
            )
            self.canvas.draw_idle()
            return
        try:
            minimum, maximum, _threshold, fwhm = self._parameters()
        except ValueError:
            minimum, maximum, fwhm = 5.0, 120.0, 0.15
        self.axis.set_xlim(minimum, maximum)
        self.axis.set_ylim(0, 105)
        grouped: dict[str, list[ReflectionRow]] = defaultdict(list)
        for row in self.rows:
            grouped[row.radiation].append(row)
        if self.rows and (self.sticks_radio.isChecked() or cell_only):
            for radiation, rows in grouped.items():
                x_values = [row.two_theta for row in rows]
                y_values = [
                    100.0 if row.intensity is None else row.intensity for row in rows
                ]
                self.axis.vlines(
                    x_values,
                    0,
                    y_values,
                    linewidth=1.2,
                    label=radiation,
                )
        elif self.rows:
            try:
                profile = gaussian_powder_profile(
                    self.rows,
                    minimum,
                    maximum,
                    fwhm,
                    point_count=5000,
                    normalize_to=100.0,
                )
            except XRDDataError:
                self.canvas.draw_idle()
                return
            if len(profile.components) > 1:
                for radiation, component in profile.components.items():
                    self.axis.plot(
                        profile.x,
                        component,
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.55,
                        label=radiation,
                    )
                self.axis.plot(
                    profile.x,
                    profile.total,
                    color="#202020",
                    linewidth=1.4,
                    label=localised("Total", "Somme", "Сумма"),
                )
            else:
                self.axis.plot(profile.x, profile.total, linewidth=1.3)
        if len(grouped) > 1:
            self.axis.legend(loc="upper right")
        self.canvas.draw_idle()


class ReflectionTablePage(CalculationPage):
    """Display and export the established calculated reflection rows."""

    def __init__(self, radiations_provider, on_import_paths=None, parent=None) -> None:
        super().__init__(radiations_provider, parent)
        self.on_import_paths = on_import_paths
        self.summary_text = ""
        self._sort_column = -1
        self._sort_descending = False
        self._build_ui()
        self.retranslate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.open_button = QPushButton()
        self.open_button.clicked.connect(self.open_dialog)
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(self.open_button)
        root.addLayout(file_row)

        self.settings_group = QGroupBox()
        settings = QGridLayout(self.settings_group)
        self.minimum_label = QLabel()
        self.maximum_label = QLabel()
        self.intensity_label = QLabel()
        self.minimum_edit = QLineEdit("5")
        self.maximum_edit = QLineEdit("120")
        self.intensity_edit = QLineEdit("0.1")
        for field in (self.minimum_edit, self.maximum_edit, self.intensity_edit):
            field.setMaximumWidth(80)
            field.setMinimumWidth(0)
        self.calculate_button = QPushButton()
        self.calculate_button.clicked.connect(self.calculate)
        settings.addWidget(self.minimum_label, 0, 0)
        settings.addWidget(self.minimum_edit, 0, 1)
        settings.addWidget(self.maximum_label, 0, 2)
        settings.addWidget(self.maximum_edit, 0, 3)
        settings.addWidget(self.intensity_label, 0, 4)
        settings.addWidget(self.intensity_edit, 0, 5)
        settings.addWidget(self.calculate_button, 0, 6)
        settings.setColumnStretch(7, 1)
        root.addWidget(self.settings_group)

        self.table = QTreeWidget()
        self.table.setColumnCount(9)
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        header = self.table.header()
        header.setSectionsClickable(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for index, width in enumerate((190, 90, 90, 85, 75, 60, 80, 110, 90)):
            header.resizeSection(index, width)
        header.sectionClicked.connect(self.sort_table)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        root.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(76)
        self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.save_button = QPushButton()
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_csv)
        bottom.addWidget(self.details, 1)
        bottom.addWidget(self.save_button, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(bottom)

    def retranslate(self) -> None:
        self.open_button.setText(tr("text.open_cif"))
        self.settings_group.setTitle(tr("text.calculation"))
        self.minimum_label.setText(tr("text.two_theta_from"))
        self.maximum_label.setText(tr("text.to"))
        self.intensity_label.setText(tr("text.irel_2"))
        self.calculate_button.setText(tr("text.calculate"))
        self.save_button.setText(tr("text.save_csv"))
        self.table.setHeaderLabels(_reflection_headers())
        if self.structure is None:
            self._set_summary(
                localised(
                    "Open a CIF file.",
                    "Ouvrez un fichier CIF.",
                    "Откройте CIF-файл.",
                )
            )
        else:
            self.calculate(show_errors=False)

    def open_dialog(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            tr("text.open_cif"),
            "",
            f"CIF (*.cif);;{tr('qt.all_files')} (*)",
        )
        if path and self.on_import_paths is not None:
            self.on_import_paths([path])

    def load_document(self, document) -> None:
        self.cif_document = document
        self.structure = document.diffraction
        self.path_edit.setText(str(document.source))
        self.intensity_edit.setEnabled(not self.structure.cell_only)
        self.calculate()

    def clear_document(self) -> None:
        self.cif_document = None
        self.structure = None
        self.rows = []
        self.path_edit.clear()
        self.intensity_edit.setEnabled(True)
        self.table.clear()
        self.save_button.setEnabled(False)
        self._set_summary(
            localised(
                "Open a CIF file.",
                "Ouvrez un fichier CIF.",
                "Откройте CIF-файл.",
            )
        )

    def _parameters(self) -> tuple[float, float, float]:
        try:
            return (
                self._number(self.minimum_edit),
                self._number(self.maximum_edit),
                self._number(self.intensity_edit),
            )
        except ValueError as error:
            raise ValueError(
                localised(
                    "The angle limits and intensity threshold must be numeric.",
                    "Les limites angulaires et le seuil d’intensité doivent être numériques.",
                    "Границы углов и порог интенсивности должны быть числами.",
                )
            ) from error

    def calculate(self, _checked: bool = False, *, show_errors: bool = True) -> None:
        if self.structure is None:
            if show_errors:
                QMessageBox.information(
                    self,
                    localised("No structure", "Aucune structure", "Нет структуры"),
                    localised(
                        "Open a CIF file first.",
                        "Ouvrez d’abord un fichier CIF.",
                        "Сначала откройте CIF-файл.",
                    ),
                )
            return
        try:
            minimum, maximum, threshold = self._parameters()
            self.rows = calculate_reflections(
                self.structure,
                self._scattering_factors(),
                self.radiations_provider(),
                minimum,
                maximum,
                threshold,
            )
        except Exception as error:
            if show_errors:
                self._calculation_error(error)
            return
        self._fill_table()
        self.save_button.setEnabled(bool(self.rows))
        if self.structure.cell_only:
            summary = localised(
                f"{self.structure.name}: atom coordinates are not specified; "
                f"{len(self.rows)} reflections are not systematically forbidden "
                "by the space group. F^2 and intensity are unavailable.",
                f"{self.structure.name} : les coordonnées atomiques ne sont pas "
                f"indiquées ; {len(self.rows)} réflexions ne sont pas interdites "
                "systématiquement par le groupe d’espace. F^2 et l’intensité sont indisponibles.",
                f"{self.structure.name}: координаты атомов не заданы; отражений, "
                f"не запрещённых систематически пространственной группой, — {len(self.rows)}. "
                "F^2 и интенсивность недоступны.",
            )
        else:
            summary = localised(
                f"{self.structure.name}: {len(self.structure.atoms)} atoms after "
                f"symmetry expansion; {len(self.rows)} table rows. Intensities "
                "are calculated for a powder pattern.",
                f"{self.structure.name} : {len(self.structure.atoms)} atomes après "
                f"développement de la symétrie ; {len(self.rows)} lignes. Les "
                "intensités sont calculées pour un diagramme de poudre.",
                f"{self.structure.name}: атомов после размножения симметрией — "
                f"{len(self.structure.atoms)}; строк в таблице — {len(self.rows)}. "
                "Интенсивности расчётные, для порошковой дифрактограммы.",
            )
        self._set_summary(summary)

    def _fill_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.clear()
        for index, row in enumerate(self.rows):
            values = (
                row.hkl,
                f"{row.d:.6f}",
                f"{row.two_theta:.5f}",
                row.radiation,
                f"{row.wavelength:.5f}",
                f"{row.weight:.4g}",
                str(row.multiplicity),
                "—" if row.f2_sum is None else f"{row.f2_sum:.6g}",
                "—" if row.intensity is None else f"{row.intensity:.4f}",
            )
            sort_values = (
                row.hkl,
                row.d,
                row.two_theta,
                row.radiation,
                row.wavelength,
                row.weight,
                row.multiplicity,
                "—" if row.f2_sum is None else row.f2_sum,
                "—" if row.intensity is None else row.intensity,
            )
            item = ReflectionTreeItem(values)
            item.setData(0, ROW_INDEX_ROLE, index)
            for column, sort_value in enumerate(sort_values):
                item.setData(column, SORT_VALUE_ROLE, sort_value)
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignCenter)
            self.table.addTopLevelItem(item)
        if self._sort_column >= 0:
            order = (
                Qt.SortOrder.DescendingOrder
                if self._sort_descending
                else Qt.SortOrder.AscendingOrder
            )
            self.table.sortItems(self._sort_column, order)

    def _set_summary(self, text: str) -> None:
        self.summary_text = text
        self.details.setPlainText(text)

    def _selection_changed(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.details.setPlainText(self.summary_text)
            return
        index = selected[0].data(0, ROW_INDEX_ROLE)
        try:
            row = self.rows[int(index)]
        except (IndexError, TypeError, ValueError):
            self.details.setPlainText(self.summary_text)
            return
        prefix = localised(
            "Equivalent hkl",
            "hkl équivalents",
            "Эквивалентные hkl",
        )
        self.details.setPlainText(f"{prefix}: {_full_hkl(row)}")

    def sort_table(self, column: int) -> None:
        if column == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = int(column)
            self._sort_descending = False
        order = (
            Qt.SortOrder.DescendingOrder
            if self._sort_descending
            else Qt.SortOrder.AscendingOrder
        )
        self.table.sortItems(column, order)

    def save_csv(self) -> None:
        if not self.rows:
            return
        source = Path(self.cif_document.source) if self.cif_document is not None else Path("structure.cif")
        path, _selected = QFileDialog.getSaveFileName(
            self,
            tr("text.save_table"),
            f"{source.stem}_calculated_xrd.csv",
            f"CSV (*.csv);;{tr('qt.all_files')} (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".csv")
        try:
            write_reflection_csv(target, self.rows, _reflection_headers())
        except OSError as error:
            QMessageBox.critical(self, tr("text.save_error"), str(error))
            return
        self._set_summary(
            localised(
                f"Table saved: {target}",
                f"Table enregistrée : {target}",
                f"Таблица сохранена: {target}",
            )
        )


class StructuresPage(QWidget):
    """Qt Structures section with one shared radiation selector."""

    title_key = "text.structures"

    def __init__(
        self,
        store,
        radiation_settings: RadiationSettings,
        parent=None,
        *,
        on_import_paths=None,
    ) -> None:
        super().__init__(parent)
        self.workspace = STRUCTURES
        self.store = store
        self.radiation_settings = radiation_settings
        self.active_uid: str | None = None
        self.active_payload = None
        self._radiation_signature = tuple(radiation_settings.lines())

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)
        self.radiation_selector = RadiationSelector(radiation_settings)
        self.radiation_selector.radiation_changed.connect(self.radiation_changed)
        root.addWidget(self.radiation_selector)

        self.tabs = QTabWidget()
        self.structure_viewer = StructureViewerPage(
            on_import_paths=on_import_paths,
        )
        self.calculated_pattern = CalculatedPatternPage(radiation_settings.lines)
        self.reflection_table = ReflectionTablePage(
            radiation_settings.lines,
            on_import_paths=on_import_paths,
        )
        self.tabs.addTab(self.structure_viewer, "")
        self.tabs.addTab(self.calculated_pattern, "")
        self.tabs.addTab(self.reflection_table, "")
        root.addWidget(self.tabs, 1)

        self.retranslate()
        self.refresh_documents()

    def retranslate(self) -> None:
        self.radiation_selector.retranslate()
        self.tabs.setTabText(0, tr("text.structure_viewer"))
        self.tabs.setTabText(1, tr("text.calculated_graph"))
        self.tabs.setTabText(2, tr("text.reflection_table_2"))
        self.structure_viewer.retranslate()
        self.calculated_pattern.retranslate()
        self.reflection_table.retranslate()

    def refresh_documents(self) -> None:
        self.radiation_selector.sync_from_settings()
        assigned = [
            document
            for document in self.store.assigned_documents(STRUCTURES)
            if document.kind in {CIF, CELL_PHASE}
        ]
        document = assigned[-1] if assigned else None
        radiation_signature = tuple(self.radiation_settings.lines())
        radiation_changed = radiation_signature != self._radiation_signature
        self._radiation_signature = radiation_signature
        if document is None:
            if self.active_uid is not None:
                self.active_uid = None
                self.active_payload = None
                self.structure_viewer.clear_document()
                self.calculated_pattern.clear_document()
                self.reflection_table.clear_document()
            return
        if document.uid != self.active_uid or document.payload is not self.active_payload:
            self.active_uid = document.uid
            self.active_payload = document.payload
            self.structure_viewer.load_document(document.payload)
            self.calculated_pattern.load_document(document.payload)
            self.reflection_table.load_document(document.payload)
        elif radiation_changed:
            self.radiation_changed()

    def radiation_changed(self) -> None:
        self._radiation_signature = tuple(self.radiation_settings.lines())
        self.calculated_pattern.calculate(show_errors=False)
        self.reflection_table.calculate(show_errors=False)

    def sync_radiation(self) -> None:
        self.radiation_selector.sync_from_settings()
        self.radiation_changed()

    def select_reflection_table(self) -> None:
        self.tabs.setCurrentIndex(2)

    def select_calculated_pattern(self) -> None:
        self.tabs.setCurrentIndex(1)

    def refresh_atom_styles(self) -> None:
        self.structure_viewer.refresh_atom_styles()


__all__ = [
    "CalculatedPatternPage",
    "ReflectionTablePage",
    "ReflectionTreeItem",
    "StructuresPage",
]
