"""Native editor for the reference peaks used by the correction workflow."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFormLayout, QHeaderView,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout,
)

from ..localization import tr
from ..services.reference_peaks import (
    ReferencePeakError, read_reference_peaks, reference_peak_path,
    validate_reference_peaks, write_reference_peaks,
)


class ReferencePeaksDialog(QDialog):
    def __init__(self, parent=None, *, path: str | Path | None = None):
        super().__init__(parent)
        self.path = Path(path) if path is not None else reference_peak_path()
        self.setWindowTitle(tr("text.reference_peaks_2"))
        self.resize(580, 460)
        layout = QVBoxLayout(self)
        self.table = QTreeWidget()
        self.table.setHeaderLabels([tr("text.name"), "2θ, °"])
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        note = QLabel(tr("qt.reference_edit_note"))
        note.setWordWrap(True)
        layout.addWidget(note)
        inputs = QFormLayout()
        self.name_edit = QLineEdit()
        self.value_edit = QLineEdit()
        inputs.addRow(tr("text.name"), self.name_edit)
        inputs.addRow("2θ, °", self.value_edit)
        layout.addLayout(inputs)
        actions = QHBoxLayout()
        self.add_button = QPushButton(tr("text.add"))
        self.add_button.setAutoDefault(False)
        self.add_button.clicked.connect(self.add_entry)
        self.remove_button = QPushButton(tr("text.remove_selected"))
        self.remove_button.setAutoDefault(False)
        self.remove_button.clicked.connect(self.remove_selected)
        self.remove_button.setEnabled(False)
        self.table.itemSelectionChanged.connect(
            lambda: self.remove_button.setEnabled(bool(self.table.selectedItems())))
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                       QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText(tr("text.save"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("text.cancel"))
        self.buttons.accepted.connect(self.save_and_close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for name, value in read_reference_peaks(self.path).items():
            self._append_row(name, value)

    def _append_row(self, name: str, value: float) -> None:
        row = QTreeWidgetItem([name, str(value)])
        row.setFlags(row.flags() | Qt.ItemFlag.ItemIsEditable)
        row.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.addTopLevelItem(row)

    def _rows(self):
        return [(self.table.topLevelItem(index).text(0),
                 self.table.topLevelItem(index).text(1))
                for index in range(self.table.topLevelItemCount())]

    def _validation_error(self, error: ReferencePeakError) -> None:
        key = "qt.reference_duplicate" if error.code == "duplicate" else "qt.reference_invalid"
        QMessageBox.warning(self, tr("text.reference_peaks_2"),
                            tr(key, row=error.row, name=error.name))

    def add_entry(self) -> None:
        rows = self._rows() + [(self.name_edit.text(), self.value_edit.text())]
        try:
            entries = validate_reference_peaks(rows)
        except ReferencePeakError as error:
            self._validation_error(error)
            return
        name = self.name_edit.text().strip()
        self._append_row(name, entries[name])
        self.name_edit.clear()
        self.value_edit.clear()
        self.name_edit.setFocus()

    def remove_selected(self) -> None:
        for row in self.table.selectedItems():
            self.table.takeTopLevelItem(self.table.indexOfTopLevelItem(row))

    def save_and_close(self) -> None:
        try:
            entries = validate_reference_peaks(self._rows())
            write_reference_peaks(self.path, entries)
        except ReferencePeakError as error:
            self._validation_error(error)
            return
        except OSError as error:
            QMessageBox.critical(self, tr("text.save_error"), str(error))
            return
        self.accept()
