"""Small Qt pages used while the feature pages are migrated incrementally."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr
from ..models.project import CELL_PHASE, CIF, POLE_DATA, SCAN
from ..version import APP_VERSION


class SectionPage(QWidget):
    """Visible migration boundary backed by real project assignments."""

    def __init__(self, workspace: str, title_key: str, store, parent=None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.title_key = title_key
        self.store = store

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(14)

        self.heading = QLabel()
        font = self.heading.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        self.heading.setFont(font)
        layout.addWidget(self.heading)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.message.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        layout.addWidget(self.message)

        self.assigned_group = QGroupBox()
        assigned_layout = QVBoxLayout(self.assigned_group)
        self.documents = QListWidget()
        self.documents.setAlternatingRowColors(True)
        assigned_layout.addWidget(self.documents)
        layout.addWidget(self.assigned_group, 1)

        self.retranslate()
        self.refresh_documents()

    def retranslate(self) -> None:
        self.heading.setText(tr(self.title_key))
        self.message.setText(
            f"<b>{tr('qt.preview_heading')}</b><br><br>"
            + tr("qt.preview_body", stable_version=APP_VERSION)
        )
        self.assigned_group.setTitle(tr("qt.assigned_objects"))
        self.refresh_documents()

    def refresh_documents(self) -> None:
        self.documents.clear()
        assigned = self.store.assigned_documents(self.workspace)
        if not assigned:
            item = QListWidgetItem(tr("qt.no_assigned_objects"))
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.documents.addItem(item)
            return
        for document in assigned:
            kind = {
                SCAN: tr("text.measurement"),
                CIF: "CIF",
                CELL_PHASE: tr("text.cell"),
                POLE_DATA: tr("text.pole_figure"),
            }.get(document.kind, document.kind)
            label = f"{document.name}   [{kind}]"
            item = QListWidgetItem(label)
            item.setToolTip(str(document.source))
            item.setData(Qt.ItemDataRole.UserRole, document.uid)
            self.documents.addItem(item)


__all__ = ["SectionPage"]
