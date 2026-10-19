"""Shared labelled Miller-index fields for all structure controls."""
import re
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QSpinBox, QStackedWidget, QLineEdit, QSizePolicy


class HklEditor(QWidget):
    returnPressed = Signal()
    editingFinished = Signal()

    def __init__(self, text='0 0 1', parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        self.fields = []
        for column, name in enumerate(('h', 'k', 'l')):
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 0, column)
            field = QSpinBox()
            field.setRange(-999, 999)
            field.setKeyboardTracking(False)
            field.setMinimumWidth(field.fontMetrics().horizontalAdvance('-999') + 24)
            field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            field.lineEdit().returnPressed.connect(self.returnPressed)
            field.editingFinished.connect(self.editingFinished)
            layout.addWidget(field, 1, column)
            self.fields.append(field)
        self.setText(text)

    @staticmethod
    def set_index_text(field, value):
        try:
            number = int(str(value))
            if not field.minimum() <= number <= field.maximum():
                raise ValueError()
            field.setValue(number)
        except ValueError:
            field.lineEdit().setText(str(value))

    def text(self):
        return ' '.join(field.lineEdit().text() for field in self.fields).strip()

    def setText(self, text):
        parts = re.split(r'[\s,;]+', str(text).strip())
        if len(parts) > 3:
            parts = [str(text), '', '']
        for field, value in zip(self.fields, parts + [''] * (3-len(parts))):
            self.set_index_text(field, value)


class HklTargetEditor(QStackedWidget):
    editingFinished = Signal()
    returnPressed = Signal()

    def __init__(self, text='0 0 2', parent=None):
        super().__init__(parent)
        self.hkl = HklEditor(text)
        self.numeric = QLineEdit()
        self.addWidget(self.hkl)
        self.addWidget(self.numeric)
        for editor in (self.hkl, self.numeric):
            editor.editingFinished.connect(self.editingFinished)
            editor.returnPressed.connect(self.returnPressed)
        self.setMinimumWidth(0)

    def set_mode(self, mode):
        self.setCurrentIndex(0 if mode == 'hkl' else 1)

    def text(self):
        return self.currentWidget().text()

    def setText(self, value):
        self.currentWidget().setText(value)
