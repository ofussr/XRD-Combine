"""Session-only per-measurement peak tables and companion review."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QInputDialog,
    QVBoxLayout,
)

from ..localization import localised, tr


PEAK_COLOURS = {"primary": "#888888", "ka2": "#bc7512", "kb": "#7c4ab2"}


@dataclass(frozen=True)
class SessionPeak:
    number: int  # Stable ID, independent of the table's sorted row number.
    scan_uid: str
    axis_name: str
    source_center: float
    source_x: np.ndarray
    source_background: np.ndarray
    source_profile: np.ndarray
    source_height: float
    source_area: float
    source_fwhm: float
    kind: str = "primary"
    parent_number: int | None = None
    filled: bool = True
    hkl_assignments: tuple[tuple[str, int, int, int], ...] = ()
    source_sigma: float = 0.0
    source_gamma: float = 0.0


@dataclass(frozen=True)
class CompanionProposal:
    parent_number: int
    parent_center: float
    predicted_center: float
    fitted_center: float
    existing_number: int


class CompanionReviewDialog(QDialog):
    """Review calculated matches between already stored peaks."""

    def __init__(self, kind: str, proposals: list[CompanionProposal], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(localised("Review companion peaks", "Vérifier les pics satellites",
                                      "Проверка пиков-спутников"))
        root = QVBoxLayout(self)
        label = "Kα2" if kind == "ka2" else "Kβ"
        root.addWidget(QLabel(localised(
            f"Matched {len(proposals)} existing peaks as possible {label}. Confirm the pairs:",
            f"{len(proposals)} pics existants correspondent à {label}. Confirmez les paires :",
            f"Найдено совпадений в таблице для {label}: {len(proposals)}. Подтвердите пары:",
        )))
        self.checks = []
        for proposal in proposals:
            check = QCheckBox(localised(
                f"{proposal.parent_center:.5f} → {proposal.fitted_center:.5f}"
                f" (expected {proposal.predicted_center:.5f})",
                f"{proposal.parent_center:.5f} → {proposal.fitted_center:.5f}"
                f" (attendu {proposal.predicted_center:.5f})",
                f"{proposal.parent_center:.5f} → {proposal.fitted_center:.5f}"
                f" (ожидалось {proposal.predicted_center:.5f})",
            ))
            check.setChecked(True)
            root.addWidget(check)
            self.checks.append(check)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(localised(
            "Mark selected", "Marquer la sélection", "Отметить выбранные"
        ))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def chosen_indices(self) -> list[int]:
        return [index for index, check in enumerate(self.checks) if check.isChecked()]


class PeakTableDialog(QDialog):
    """A non-modal table for one measurement, identified by its stable UID."""

    remove_requested = Signal(int)
    clear_requested = Signal()
    add_requested = Signal()
    draw_requested = Signal()
    fit_requested = Signal()
    sum_visible_changed = Signal(bool)
    find_alpha2_requested = Signal()
    find_beta_requested = Signal()
    fill_changed = Signal(int, bool)
    assign_hkl_requested = Signal(int, str, int, int, int)
    clear_hkl_requested = Signal(int)
    index_requested = Signal()

    def __init__(self, scan_uid: str, parent=None) -> None:
        super().__init__(parent)
        self.scan_uid = scan_uid
        self.measurement_name = ""
        self.setModal(False)
        self.resize(880, 360)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 9)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table)
        self.show_sum_check = QCheckBox()
        self.show_sum_check.setChecked(True)
        self.show_sum_check.toggled.connect(self.sum_visible_changed.emit)
        layout.addWidget(self.show_sum_check)
        self.phases: list[tuple[str, str]] = []
        self.index_button = QPushButton()
        self.index_button.clicked.connect(self.index_requested.emit)
        self.index_button.hide()
        index_actions = QHBoxLayout()
        index_actions.addWidget(self.index_button)
        index_actions.addStretch(1)
        layout.addLayout(index_actions)
        actions = QHBoxLayout()
        self.add_button = QPushButton()
        self.add_button.clicked.connect(self.add_requested.emit)
        actions.addWidget(self.add_button)
        self.draw_button = QPushButton()
        self.draw_button.clicked.connect(self.draw_requested.emit)
        actions.addWidget(self.draw_button)
        self.fit_button = QPushButton()
        self.fit_button.clicked.connect(self.fit_requested.emit)
        actions.addWidget(self.fit_button)
        self.alpha2_button = QPushButton()
        self.alpha2_button.clicked.connect(self.find_alpha2_requested.emit)
        actions.addWidget(self.alpha2_button)
        self.beta_button = QPushButton()
        self.beta_button.clicked.connect(self.find_beta_requested.emit)
        actions.addWidget(self.beta_button)
        self.assign_button = QPushButton()
        self.assign_button.clicked.connect(self._assign_hkl)
        actions.addWidget(self.assign_button)
        self.clear_hkl_button = QPushButton()
        self.clear_hkl_button.clicked.connect(self._clear_hkl)
        actions.addWidget(self.clear_hkl_button)
        actions.addStretch(1)
        self.remove_button = QPushButton()
        self.remove_button.clicked.connect(self._remove_selected)
        actions.addWidget(self.remove_button)
        self.clear_button = QPushButton()
        self.clear_button.clicked.connect(self._clear)
        actions.addWidget(self.clear_button)
        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        layout.addLayout(actions)
        self.retranslate()

    def retranslate(self) -> None:
        name = self.measurement_name
        self.setWindowTitle(localised(
            f"Detected peaks — {name}", f"Pics détectés — {name}", f"Найденные пики — {name}"
        ))
        self.table.setHorizontalHeaderLabels((
            localised("No.", "N°", "№"),
            localised("Axis", "Axe", "Ось"),
            localised("Position", "Position", "Положение"),
            localised("Height", "Hauteur", "Высота"),
            localised("FWHM", "L.M.H.", "ПШПВ"),
            localised("Area", "Aire", "Площадь"),
            tr("text.line"),
            localised("Fill", "Remplir", "Заливка"),
            "hkl",
        ))
        self.add_button.setText(localised("Add peaks", "Ajouter des pics", "Добавить пики"))
        self.draw_button.setText(localised("Draw peak", "Dessiner un pic", "Нарисовать пик"))
        self.fit_button.setText("Do Fit")
        self.index_button.setText(localised("Index cell…", "Indexer la maille…", "Индексировать ячейку…"))
        self.fit_button.setToolTip(localised(
            "Refine all accepted peaks together against the shared background",
            "Réajuster ensemble tous les pics validés sur le fond commun",
            "Совместно уточнить все принятые пики относительно общего фона",
        ))
        self.show_sum_check.setText(localised(
            "Show background + fitted peaks", "Afficher le fond + les pics ajustés",
            "Показать сумму фона и подогнанных пиков",
        ))
        self.alpha2_button.setText(localised("Find Kα2", "Trouver Kα2", "Найти Kα2"))
        self.beta_button.setText(localised("Find Kβ", "Trouver Kβ", "Найти Kβ"))
        self.assign_button.setText(localised("Assign hkl…", "Attribuer hkl…", "Назначить hkl…"))
        self.clear_hkl_button.setText(localised("Clear hkl", "Effacer hkl", "Очистить hkl"))
        self.beta_button.setToolTip(localised(
            "Available for copper radiation", "Disponible pour le rayonnement du cuivre",
            "Доступно для медного излучения",
        ))
        self.remove_button.setText(localised(
            "Remove selected", "Supprimer la sélection", "Удалить выбранный"
        ))
        self.clear_button.setText(localised("Clear table", "Vider le tableau", "Очистить таблицу"))
        self.close_button.setText(tr("text.close"))

    def set_indexation_enabled(self, enabled: bool) -> None:
        self.index_button.setVisible(enabled)

    def refresh(self, peaks: list[SessionPeak], item, *, wavelength: float,
                display_mode: str, copper: bool, alpha2_available: bool,
                phases: list[tuple[str, str]] | None = None) -> None:
        self.measurement_name = item.name
        self.phases = phases or []
        self.retranslate()
        from ..models.viewer import is_two_theta, two_theta_to_d

        def position(peak: SessionPeak) -> float:
            value = peak.source_center * item.x_scale + item.x_shift
            if display_mode == "d" and is_two_theta(peak.axis_name):
                value = float(two_theta_to_d(np.asarray([value]), wavelength)[0])
            return value

        own_peaks = [p for p in peaks if p.scan_uid == self.scan_uid]
        own_peaks.sort(key=lambda peak: (not np.isfinite(position(peak)), position(peak)))
        selected = self.table.item(self.table.currentRow(), 0)
        selected_id = selected.data(Qt.ItemDataRole.UserRole) if selected else None
        display_numbers = {peak.number: row + 1 for row, peak in enumerate(own_peaks)}
        self.table.blockSignals(True)
        self.table.setRowCount(len(own_peaks))
        for row, peak in enumerate(own_peaks):
            label = {"primary": "", "ka2": "Kα2", "kb": "Kβ"}[peak.kind]
            phase_names = dict(self.phases)
            assignments = "; ".join(
                f"{phase_names.get(uid, uid)} ({h} {k} {l})"
                for uid, h, k, l in peak.hkl_assignments
            )
            values = (
                str(row + 1), peak.axis_name, f"{position(peak):.5f}",
                f"{peak.source_height * item.y_factor:.5g}",
                f"{peak.source_fwhm * abs(item.x_scale):.5g}",
                f"{peak.source_area * item.y_factor * abs(item.x_scale):.5g}", label,
                "", assignments,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, peak.number)
                if column == 7:
                    cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    cell.setCheckState(Qt.CheckState.Checked if peak.filled
                                       else Qt.CheckState.Unchecked)
                if peak.kind != "primary":
                    colour = QColor(PEAK_COLOURS[peak.kind])
                    tint = QColor(colour)
                    tint.setAlpha(35)
                    cell.setBackground(QBrush(tint))
                    if column == 6:
                        cell.setForeground(QBrush(colour))
                        if peak.parent_number in display_numbers:
                            cell.setToolTip(localised(
                                f"Linked to peak {display_numbers[peak.parent_number]}",
                                f"Lié au pic {display_numbers[peak.parent_number]}",
                                f"Связан с пиком {display_numbers[peak.parent_number]}",
                            ))
                self.table.setItem(row, column, cell)
            if peak.number == selected_id:
                self.table.setCurrentCell(row, 0)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.alpha2_button.setEnabled(alpha2_available and any(
            peak.kind == "primary" for peak in own_peaks
        ))
        self.beta_button.setEnabled(copper and any(
            peak.kind == "primary" for peak in own_peaks
        ))
        self.clear_button.setEnabled(bool(own_peaks))
        self.remove_button.setEnabled(bool(own_peaks))
        self.assign_button.setEnabled(bool(own_peaks) and bool(self.phases))
        self.clear_hkl_button.setEnabled(bool(own_peaks))
        self.show_sum_check.setEnabled(bool(own_peaks))

    def set_sum_visible(self, visible: bool) -> None:
        self.show_sum_check.blockSignals(True)
        self.show_sum_check.setChecked(bool(visible))
        self.show_sum_check.blockSignals(False)

    def _selected_number(self) -> int | None:
        row = self.table.currentRow()
        cell = self.table.item(row, 0) if row >= 0 else None
        return int(cell.data(Qt.ItemDataRole.UserRole)) if cell is not None else None

    def _item_changed(self, cell: QTableWidgetItem) -> None:
        if cell.column() != 7:
            return
        identifier = self.table.item(cell.row(), 0)
        if identifier is not None:
            self.fill_changed.emit(int(identifier.data(Qt.ItemDataRole.UserRole)),
                                   cell.checkState() == Qt.CheckState.Checked)

    def _assign_hkl(self) -> None:
        number = self._selected_number()
        if number is None or not self.phases:
            return
        names = [name for _uid, name in self.phases]
        labels = [f"{name} [{uid[:8]}]" if names.count(name) > 1 else name
                  for uid, name in self.phases]
        name, accepted = QInputDialog.getItem(
            self, "hkl", localised("Structure", "Structure", "Структура"), labels, 0, False)
        if not accepted:
            return
        value, accepted = QInputDialog.getText(
            self, "hkl", localised("Enter h k l", "Saisir h k l", "Введите h k l"))
        if not accepted:
            return
        parts = re.fullmatch(r"\s*([+-]?\d+)[\s,;]+([+-]?\d+)[\s,;]+([+-]?\d+)\s*", value)
        if parts is None or all(int(part) == 0 for part in parts.groups()):
            QMessageBox.warning(self, "hkl", localised(
                "Enter three integers, not all zero.",
                "Saisissez trois entiers, non tous nuls.",
                "Введите три целых индекса, не все нулевые."))
            return
        uid = self.phases[labels.index(name)][0]
        self.assign_hkl_requested.emit(number, uid, *map(int, parts.groups()))

    def _clear_hkl(self) -> None:
        number = self._selected_number()
        if number is not None:
            self.clear_hkl_requested.emit(number)

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            number = self.table.item(row, 0)
            if number is not None:
                self.remove_requested.emit(int(number.data(Qt.ItemDataRole.UserRole)))

    def _clear(self) -> None:
        if not self.table.rowCount():
            return
        answer = QMessageBox.question(
            self,
            localised("Clear peaks", "Effacer les pics", "Очистить пики"),
            localised("Remove the peaks of this measurement?",
                      "Supprimer les pics de cette mesure ?",
                      "Удалить пики этого измерения?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.clear_requested.emit()
