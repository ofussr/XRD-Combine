"""Small diagnostic dialog for normally hidden compatibility controls."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..file_associations import ASSOCIATION_SUFFIXES, WindowsFileAssociations
from ..localization import tr
from .plot_renderer import PLOT_RENDERER_KEYS, pyqtgraph_available


class DebugDialog(QDialog):
    """Expose the retained Matplotlib fallback without cluttering the menus."""

    def __init__(
        self,
        renderer_controller,
        parent=None,
        *,
        file_associations: WindowsFileAssociations | None = None,
    ) -> None:
        super().__init__(parent)
        self.renderer_controller = renderer_controller
        self.setWindowTitle(tr("qt.debug"))
        self.setModal(True)
        self.setMinimumWidth(340)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.renderer_combo = QComboBox()
        for mode, key in PLOT_RENDERER_KEYS.items():
            self.renderer_combo.addItem(tr(key), mode)
            index = self.renderer_combo.count() - 1
            self.renderer_combo.model().item(index).setEnabled(
                mode != "pyqtgraph" or pyqtgraph_available()
            )
        current = self.renderer_combo.findData(renderer_controller.mode)
        self.renderer_combo.setCurrentIndex(max(0, current))
        form.addRow(tr("qt.debug_plot_renderer"), self.renderer_combo)
        root.addLayout(form)

        self.file_associations = file_associations or WindowsFileAssociations()
        association_box = QGroupBox(tr("qt.debug_file_associations"))
        association_layout = QVBoxLayout(association_box)
        explanation = QLabel(tr("qt.debug_file_associations_note"))
        explanation.setWordWrap(True)
        association_layout.addWidget(explanation)

        self.association_status_labels: dict[str, QLabel] = {}
        status_grid = QGridLayout()
        for row, suffix in enumerate(ASSOCIATION_SUFFIXES):
            status_grid.addWidget(QLabel(suffix), row, 0)
            label = QLabel()
            self.association_status_labels[suffix] = label
            status_grid.addWidget(label, row, 1)
        association_layout.addLayout(status_grid)

        association_buttons = QHBoxLayout()
        self.register_associations_button = QPushButton(
            tr("qt.debug_register_file_associations")
        )
        self.unregister_associations_button = QPushButton(
            tr("qt.debug_remove_file_associations")
        )
        self.register_associations_button.clicked.connect(
            self._register_file_associations
        )
        self.unregister_associations_button.clicked.connect(
            self._unregister_file_associations
        )
        association_buttons.addWidget(self.register_associations_button)
        association_buttons.addWidget(self.unregister_associations_button)
        association_layout.addLayout(association_buttons)

        if not self.file_associations.platform_supported:
            explanation.setText(tr("qt.debug_file_associations_windows_only"))
        elif not self.file_associations.available:
            explanation.setText(tr("qt.debug_file_associations_compiled_only"))
        enabled = self.file_associations.available
        self.register_associations_button.setEnabled(enabled)
        self.unregister_associations_button.setEnabled(enabled)
        self._refresh_file_association_status()
        root.addWidget(association_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("text.ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            tr("text.cancel")
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def selected_renderer(self) -> str:
        return str(self.renderer_combo.currentData())

    def _refresh_file_association_status(self) -> None:
        try:
            status = self.file_associations.status()
        except OSError:
            status = {suffix: False for suffix in ASSOCIATION_SUFFIXES}
        for suffix, label in self.association_status_labels.items():
            label.setText(
                tr(
                    "qt.debug_file_association_registered"
                    if status.get(suffix, False)
                    else "qt.debug_file_association_not_registered"
                )
            )

    def _register_file_associations(self) -> None:
        try:
            self.file_associations.register()
        except OSError as error:
            QMessageBox.warning(
                self,
                tr("qt.debug_file_associations"),
                str(error),
            )
            return
        self._refresh_file_association_status()

    def _unregister_file_associations(self) -> None:
        try:
            self.file_associations.unregister()
        except OSError as error:
            QMessageBox.warning(
                self,
                tr("qt.debug_file_associations"),
                str(error),
            )
            return
        self._refresh_file_association_status()


__all__ = ["DebugDialog"]
