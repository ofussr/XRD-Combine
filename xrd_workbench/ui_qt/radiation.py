"""PySide6 controls for the shared radiation-settings model."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSizePolicy,
)

from ..localization import tr
from ..models.radiation import PRESETS, RadiationSettings, RadiationTuple


class CustomRadiationDialog(QDialog):
    """Modal editor for one to five custom wavelengths and weights."""

    def __init__(
        self,
        initial: list[RadiationTuple],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.result_lines: list[RadiationTuple] | None = None
        self.rows: list[tuple[QLabel, QDoubleSpinBox, QDoubleSpinBox]] = []
        self.setModal(True)
        self.setWindowTitle(tr("text.custom_radiation"))

        root = QVBoxLayout(self)
        note = QLabel(tr("text.enter_between_one_and_five_spectral_lines"))
        note.setWordWrap(True)
        root.addWidget(note)

        self.grid = QGridLayout()
        self.grid.addWidget(QLabel(tr("text.line")), 0, 0)
        self.grid.addWidget(QLabel("λ, Å"), 0, 1)
        self.grid.addWidget(QLabel(tr("text.relative_weight")), 0, 2)
        root.addLayout(self.grid)

        row_buttons = QHBoxLayout()
        self.add_button = QPushButton(tr("text.add_line"))
        self.remove_button = QPushButton(tr("text.remove_last"))
        self.add_button.clicked.connect(self._add_default_row)
        self.remove_button.clicked.connect(self._remove_row)
        row_buttons.addWidget(self.add_button)
        row_buttons.addWidget(self.remove_button)
        row_buttons.addStretch(1)
        root.addLayout(row_buttons)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).setText(
            tr("text.apply")
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            tr("text.cancel")
        )
        self.buttons.clicked.connect(self._button_clicked)
        root.addWidget(self.buttons)

        source = initial[:5] or [("Custom 1", 1.54056, 1.0)]
        for _name, wavelength, weight in source:
            self._add_row(float(wavelength), float(weight))
        self._update_buttons()

    def _spin_box(self, value: float, *, wavelength: bool) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setKeyboardTracking(False)
        field.setDecimals(8 if wavelength else 6)
        field.setRange(0.00000001, 1000.0 if wavelength else 1_000_000.0)
        field.setValue(value)
        return field

    def _add_row(self, wavelength: float, weight: float) -> None:
        if len(self.rows) >= 5:
            return
        row_index = len(self.rows) + 1
        number = QLabel(str(row_index))
        wavelength_input = self._spin_box(wavelength, wavelength=True)
        weight_input = self._spin_box(weight, wavelength=False)
        self.grid.addWidget(number, row_index, 0)
        self.grid.addWidget(wavelength_input, row_index, 1)
        self.grid.addWidget(weight_input, row_index, 2)
        self.rows.append((number, wavelength_input, weight_input))

    def _add_default_row(self) -> None:
        self._add_row(1.54056, 1.0)
        self._update_buttons()

    def _remove_row(self) -> None:
        if len(self.rows) <= 1:
            return
        for widget in self.rows.pop():
            self.grid.removeWidget(widget)
            widget.deleteLater()
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.add_button.setEnabled(len(self.rows) < 5)
        self.remove_button.setEnabled(len(self.rows) > 1)

    def _button_clicked(self, button: QPushButton) -> None:
        role = self.buttons.buttonRole(button)
        if role == QDialogButtonBox.ButtonRole.ApplyRole:
            self.result_lines = [
                (f"Custom {index}", wavelength.value(), weight.value())
                for index, (_number, wavelength, weight) in enumerate(
                    self.rows, start=1
                )
            ]
            self.accept()
        elif role == QDialogButtonBox.ButtonRole.RejectRole:
            self.reject()


class RadiationSelector(QWidget):
    """Qt selector backed by the application-wide RadiationSettings object."""

    radiation_changed = Signal()

    def __init__(
        self,
        settings: RadiationSettings,
        parent=None,
        *,
        show_label: bool = True,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.show_label = show_label
        self._refreshing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self.label = QLabel()
        self.label.setVisible(show_label)
        layout.addWidget(self.label)
        self.combo = QComboBox()
        self.combo.setMinimumWidth(0)
        combo_policy = self.combo.sizePolicy()
        combo_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.combo.setSizePolicy(combo_policy)
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo.setMinimumContentsLength(8)
        self.combo.currentIndexChanged.connect(self._selected)
        layout.addWidget(self.combo, 1)
        self.retranslate()

    def retranslate(self) -> None:
        self.label.setText(tr("text.radiation"))
        self._refreshing = True
        self.combo.blockSignals(True)
        self.combo.clear()
        for preset in PRESETS:
            self.combo.addItem(preset.label, preset.key)
        self.combo.addItem(tr("qt.radiation_other"), "other")
        self.combo.blockSignals(False)
        self._refreshing = False
        self.sync_from_settings()

    def sync_from_settings(self) -> None:
        self._refreshing = True
        self.combo.blockSignals(True)
        index = self.combo.findData(self.settings.profile_key)
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        if self.settings.profile_key == "other":
            self.combo.setItemText(
                self.combo.currentIndex(),
                self.settings.label(other_label=tr("qt.radiation_other")),
            )
        self.combo.blockSignals(False)
        self._refreshing = False

    def _selected(self, _index: int) -> None:
        if self._refreshing:
            return
        key = self.combo.currentData()
        if key == "other":
            dialog = CustomRadiationDialog(self.settings.custom_lines, self)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_lines is None:
                self.sync_from_settings()
                return
            self.settings.select_custom(dialog.result_lines)
        elif key:
            self.settings.select_preset(str(key))
        else:
            self.sync_from_settings()
            return
        self.sync_from_settings()
        self.radiation_changed.emit()


__all__ = ["CustomRadiationDialog", "RadiationSelector"]
