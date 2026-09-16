"""Small diagnostic dialog for normally hidden compatibility controls."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
)

from ..localization import tr
from .plot_renderer import PLOT_RENDERER_KEYS, pyqtgraph_available


class DebugDialog(QDialog):
    """Expose the retained Matplotlib fallback without cluttering the menus."""

    def __init__(self, renderer_controller, parent=None) -> None:
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


__all__ = ["DebugDialog"]
