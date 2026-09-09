"""Shared project-object drawer for the parallel PySide6 interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..cell_phase import create_cell_phase_document
from ..localization import tr
from ..models.cell_phase import CellPhaseDocument
from ..models.project import CELL_PHASE, CIF, POLE_DATA, SCAN, ProjectDocument
from ..space_groups import BY_HALL_NUMBER, SETTINGS, setting_from_user_text


SUPPORTED_SUFFIXES = {
    ".xrdml",
    ".xml",
    ".raw",
    ".xy",
    ".txt",
    ".dat",
    ".csv",
    ".cif",
}


class CellPhaseDialog(QDialog):
    """Qt editor for an atom-free phase already supported by the core."""

    def __init__(self, parent=None, initial: CellPhaseDocument | None = None) -> None:
        super().__init__(parent)
        self.result_document: CellPhaseDocument | None = None
        self.setWindowTitle(tr("text.cell_parameter_phase"))
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(initial.name if initial else "")
        form.addRow(tr("text.name_2"), self.name_edit)

        self.group_combo = QComboBox()
        self.group_combo.setEditable(True)
        self.group_combo.addItems([setting.label for setting in SETTINGS])
        setting = initial.setting if initial else BY_HALL_NUMBER[1]
        self.group_combo.setCurrentText(setting.label)
        form.addRow(tr("text.space_group"), self.group_combo)
        help_label = QLabel(tr("text.enter_a_number_symbol_or_hall_n"))
        help_label.setWordWrap(True)
        form.addRow("", help_label)
        layout.addLayout(form)

        cell_box = QGroupBox(tr("text.cell"))
        cell_layout = QGridLayout(cell_box)
        defaults = initial.cell if initial else (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        self.cell_inputs: list[QDoubleSpinBox] = []
        for column, (label, value) in enumerate(
            zip(("a, Å", "b, Å", "c, Å", "α, °", "β, °", "γ, °"), defaults)
        ):
            cell_layout.addWidget(QLabel(label), 0, column)
            field = QDoubleSpinBox()
            field.setDecimals(6)
            field.setKeyboardTracking(False)
            if column < 3:
                field.setRange(0.000001, 1_000_000.0)
            else:
                field.setRange(0.000001, 179.999999)
            field.setValue(float(value))
            self.cell_inputs.append(field)
            cell_layout.addWidget(field, 1, column)
        layout.addWidget(cell_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("text.apply"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("text.cancel"))
        layout.addWidget(buttons)

    def _accept(self) -> None:
        try:
            self.result_document = create_cell_phase_document(
                self.name_edit.text(),
                setting_from_user_text(self.group_combo.currentText()),
                tuple(field.value() for field in self.cell_inputs),
            )
        except ValueError as exc:
            QMessageBox.warning(self, tr("qt.invalid_phase"), str(exc))
            return
        self.accept()


class ProjectPanel(QWidget):
    """One document pool whose check marks target the active section."""

    collapse_requested = Signal()
    paths_requested = Signal(object)

    GROUPS = (
        (SCAN, "text.measurements"),
        (CIF, "text.cif_structures"),
        (CELL_PHASE, "text.cell_parameter_phases"),
        (POLE_DATA, "text.pole_figure_data"),
    )

    def __init__(
        self,
        store,
        current_workspace: Callable[[], str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.current_workspace = current_workspace
        self._refreshing = False
        self.setMinimumWidth(300)
        self.setMaximumWidth(430)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        header = QHBoxLayout()
        self.heading = QLabel()
        heading_font = self.heading.font()
        heading_font.setBold(True)
        self.heading.setFont(heading_font)
        header.addWidget(self.heading, 1)
        self.collapse_button = QPushButton("<")
        self.collapse_button.setFixedWidth(34)
        self.collapse_button.clicked.connect(self.collapse_requested)
        header.addWidget(self.collapse_button)
        layout.addLayout(header)

        actions = QGridLayout()
        self.open_files_button = QPushButton()
        self.open_folder_button = QPushButton()
        self.new_phase_button = QPushButton()
        self.open_files_button.clicked.connect(self.open_files)
        self.open_folder_button.clicked.connect(self.open_folder)
        self.new_phase_button.clicked.connect(self.new_cell_phase)
        actions.addWidget(self.open_files_button, 0, 0)
        actions.addWidget(self.open_folder_button, 0, 1)
        actions.addWidget(self.new_phase_button, 1, 0, 1, 2)
        layout.addLayout(actions)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.tree, 1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.remove_button = QPushButton()
        self.remove_button.clicked.connect(self.remove_selected)
        layout.addWidget(self.remove_button)

        self.store.subscribe(self._store_event)
        self.retranslate()

    def retranslate(self) -> None:
        self.heading.setText(tr("text.project_data"))
        self.open_files_button.setText(tr("text.open_files"))
        self.open_folder_button.setText(tr("text.open_folder"))
        self.new_phase_button.setText(tr("text.new_cell_phase"))
        self.remove_button.setText(tr("text.remove_from_project"))
        self.tree.setHeaderLabels(
            (tr("text.name"), tr("text.in_section"), tr("text.type"))
        )
        self.refresh()

    def selected_uid(self) -> str | None:
        selected = self.tree.selectedItems()
        if not selected:
            return None
        return selected[0].data(0, Qt.ItemDataRole.UserRole)

    def refresh(self) -> None:
        selected_uid = self.selected_uid()
        self._refreshing = True
        self.tree.blockSignals(True)
        self.tree.clear()
        workspace = self.current_workspace()
        selected_item = None
        for kind, title_key in self.GROUPS:
            group = QTreeWidgetItem((tr(title_key), "", ""))
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(group)
            group.setExpanded(True)
            for document in self.store.documents.values():
                if document.kind != kind:
                    continue
                item = QTreeWidgetItem((document.name, "", self._type_label(document)))
                item.setData(0, Qt.ItemDataRole.UserRole, document.uid)
                item.setToolTip(0, str(document.source))
                compatible = self.store.compatible(kind, workspace)
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if compatible:
                    flags |= Qt.ItemFlag.ItemIsUserCheckable
                    state = (
                        Qt.CheckState.Checked
                        if self.store.is_assigned(document.uid, workspace)
                        else Qt.CheckState.Unchecked
                    )
                    item.setCheckState(1, state)
                else:
                    item.setText(1, "—")
                item.setFlags(flags)
                group.addChild(item)
                if document.uid == selected_uid:
                    selected_item = item
        if selected_item is not None:
            self.tree.setCurrentItem(selected_item)
            self.tree.scrollToItem(selected_item)
        self.tree.blockSignals(False)
        self._refreshing = False
        self._selection_changed()

    @staticmethod
    def _type_label(document: ProjectDocument) -> str:
        if document.kind == SCAN:
            return document.source.suffix.lstrip(".").upper() or "XY"
        if document.kind == CIF:
            return "CIF"
        if document.kind == CELL_PHASE:
            return tr("text.cell")
        return tr("text.pole_figure")

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._refreshing or column != 1:
            return
        uid = item.data(0, Qt.ItemDataRole.UserRole)
        if uid not in self.store.documents:
            return
        enabled = item.checkState(1) == Qt.CheckState.Checked
        try:
            self.store.assign(uid, self.current_workspace(), enabled)
        except ValueError as exc:
            QMessageBox.warning(self, tr("text.project_data"), str(exc))
            self.refresh()

    def _selection_changed(self) -> None:
        uid = self.selected_uid()
        document = self.store.documents.get(uid) if uid else None
        if document is None:
            self.status.setText(
                tr("qt.project_objects_count", count=len(self.store.documents))
            )
            self.remove_button.setEnabled(False)
            return
        if document.kind == CELL_PHASE:
            phase = document.payload
            suffix = f" [{phase.setting.choice}]" if phase.setting.choice else ""
            self.status.setText(
                f"{phase.setting.number}: {phase.setting.international_short}{suffix}"
            )
        else:
            self.status.setText(str(document.source))
        self.remove_button.setEnabled(True)

    def _store_event(self, _event, _document, _workspace) -> None:
        if not self._refreshing:
            self.refresh()

    def open_files(self) -> None:
        supported = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
        paths, _filter = QFileDialog.getOpenFileNames(
            self,
            tr("text.open_files"),
            "",
            f"{tr('text.supported_files')} ({supported});;CIF (*.cif);;{tr('qt.all_files')} (*)",
        )
        if paths:
            self.paths_requested.emit(paths)

    def open_cif(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            tr("text.add_cif_2"),
            "",
            f"CIF (*.cif);;{tr('qt.all_files')} (*)",
        )
        if path:
            self.paths_requested.emit([path])

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("text.open_folder"))
        if not folder:
            return
        paths = [
            str(path)
            for path in sorted(Path(folder).iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not paths:
            QMessageBox.information(
                self,
                tr("qt.empty_folder"),
                tr("qt.no_supported_files"),
            )
            return
        self.paths_requested.emit(paths)

    def new_cell_phase(self) -> None:
        dialog = CellPhaseDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.result_document is None:
            return
        document = self.store.add_cell_phase(dialog.result_document)
        self.store.assign(document.uid, self.current_workspace(), True)
        self.refresh()

    def remove_selected(self) -> None:
        uid = self.selected_uid()
        document = self.store.documents.get(uid) if uid else None
        if document is None:
            return
        answer = QMessageBox.question(
            self,
            tr("text.remove_from_project"),
            tr("qt.remove_question", name=document.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.remove(uid)


__all__ = ["CellPhaseDialog", "ProjectPanel", "SUPPORTED_SUFFIXES"]
