"""Native Qt bindings used by the migrated pole-figure controllers."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSlider,
    QSizePolicy, QVBoxLayout, QWidget,
)

from ..localization import tr, translate_text
from .viewer_page import CollapsibleSection


class Value:
    """Small observable value; keeps the established controller get/set API."""

    def __init__(self, value="", translation_key=None):
        self.key = translation_key
        self.value = tr(translation_key) if translation_key else value
        self.callbacks = []

    def get(self):
        return self.value

    def set(self, value):
        self.key = None
        if self.value == value:
            return
        self.value = value
        for callback in tuple(self.callbacks):
            callback()

    def set_key(self, key):
        self.set(tr(key))
        self.key = key

    def trace_add(self, _mode, callback):
        self.callbacks.append(callback)


class FrameScheduler:
    """Coalesce drawing requests on the Qt event loop."""

    def __init__(self, owner, callback, interval=16):
        self.timer = QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(callback)

    def request(self):
        if not self.timer.isActive():
            self.timer.start()

    def cancel(self):
        self.timer.stop()


def configure(widget, *, state=None, text=None, values=None, background=None, **_unused):
    """Apply controller presentation updates to native widgets."""
    if state is not None:
        if isinstance(widget, QPlainTextEdit):
            widget.setReadOnly(state == "disabled")
        else:
            widget.setEnabled(state != "disabled")
    if text is not None:
        widget.setText(translate_text(text))
    if values is not None:
        previous = widget.currentText()
        blocker = QSignalBlocker(widget)
        widget.clear()
        widget.addItems([str(value) for value in values])
        widget.setCurrentIndex(widget.findText(previous))
        del blocker
    if background is not None:
        colour = QColor(background)
        foreground = "#111111" if colour.lightnessF() > 0.5 else "#ffffff"
        widget.setStyleSheet(f"background-color: {background}; color: {foreground};")


def bind_value(value, widget, read, write, signal=None):
    def update():
        blocker = QSignalBlocker(widget)
        write(value.get())
        del blocker
    value.trace_add("write", update)
    update()
    if signal is not None:
        signal.connect(lambda *_: value.set(read()))


class PoleUi:
    """Builder for compact native controls, with live EN/FR/RU captions."""

    def _init_ui_helpers(self):
        self._captions = []
        self._value_labels = []
        self._choice_captions = []

    def caption(self, widget, source, setter=None):
        setter = setter or widget.setText
        self._captions.append((setter, source))
        setter(tr(source) if source.startswith(("text.", "pole.", "qt.")) else translate_text(source))
        return widget

    def label(self, layout, text=None, value=None):
        label = QLabel()
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if value is not None:
            bind_value(value, label, label.text, lambda v: label.setText(translate_text(str(v))))
            self._value_labels.append((value, label))
        elif text is not None:
            self.caption(label, text)
        layout.addWidget(label)
        return label

    def section(self, layout, title):
        section = CollapsibleSection("")
        section.title_source = title
        self.caption(section, title, section.set_title)
        layout.addWidget(section)
        return section

    def button(self, layout, title, callback):
        button = self.caption(QPushButton(), title)
        button.clicked.connect(lambda _checked=False: callback())
        layout.addWidget(button)
        return button

    def field(self, layout, title, value, callback=None, *, read_only=False):
        row = QHBoxLayout()
        label = self.caption(QLabel(), title)
        row.addWidget(label)
        field = QLineEdit()
        field.setMinimumWidth(0)
        field.setReadOnly(read_only)
        bind_value(value, field, field.text, lambda v: field.setText(str(v)), field.textChanged)
        if callback:
            field.returnPressed.connect(callback)
        row.addWidget(field, 1)
        layout.addLayout(row)
        return field

    def check(self, layout, title, value, callback):
        field = self.caption(QCheckBox(), title)
        bind_value(value, field, field.isChecked, field.setChecked, field.toggled)
        field.clicked.connect(lambda _checked=False: callback())
        layout.addWidget(field)
        return field

    def radios(self, layout, choices, value, callback):
        group = QButtonGroup(self)
        buttons = []
        for title, code in choices:
            field = self.caption(QRadioButton(), title)
            group.addButton(field)
            bind_value(value, field, lambda code=code: code,
                       lambda v, field=field, code=code: field.setChecked(v == code))
            field.clicked.connect(lambda checked, code=code: (value.set(code), callback()) if checked else None)
            layout.addWidget(field)
            buttons.append(field)
        return buttons

    def combo(self, layout, value=None, choices=None, callback=None):
        field = QComboBox()
        field.setMinimumWidth(0)
        field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        if choices:
            for title, code in choices:
                field.addItem(tr(title), code)
            self._choice_captions.append((field, choices))
        if value is not None:
            if choices:
                bind_value(value, field, field.currentData,
                           lambda v: field.setCurrentIndex(field.findData(v)), field.currentIndexChanged)
            else:
                bind_value(value, field, field.currentText, field.setCurrentText, field.currentTextChanged)
        if callback:
            field.activated.connect(lambda *_: callback())
        layout.addWidget(field)
        return field

    def percent(self, layout, title, value, callback, minimum=10, maximum=300):
        row = QHBoxLayout()
        row.addWidget(self.caption(QLabel(), title))
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(0)
        field.setSingleStep(5)
        field.setSuffix("%")
        bind_value(value, field, field.value, lambda v: field.setValue(float(v)), field.valueChanged)
        field.valueChanged.connect(lambda *_: callback())
        row.addWidget(field)
        layout.addLayout(row)
        return field

    def control_scroll(self, width=320):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(200)
        scroll.setMaximumWidth(460)
        body = QWidget()
        body.setMinimumWidth(0)
        body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(body)
        return scroll, layout

    def retranslate_controls(self):
        for setter, source in self._captions:
            setter(tr(source) if source.startswith(("text.", "pole.", "qt.")) else translate_text(source))
        for value, label in self._value_labels:
            if value.key:
                value.set_key(value.key)
            label.setText(translate_text(str(value.get())))
        for field, choices in self._choice_captions:
            for index, (title, _code) in enumerate(choices):
                field.setItemText(index, tr(title))


def show_error(title, message, parent=None):
    QMessageBox.critical(parent, translate_text(str(title)), translate_text(str(message)))


def show_info(title, message, parent=None):
    QMessageBox.information(parent, translate_text(str(title)), translate_text(str(message)))


def choose_colour(parent, current, title):
    colour = QColorDialog.getColor(QColor(current), parent, translate_text(title))
    return colour.name() if colour.isValid() else None


def save_plot(parent, figure, default_name):
    path, _ = QFileDialog.getSaveFileName(
        parent, tr("text.save_figure"), str(default_name),
        "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif *.tiff)",
    )
    if not path:
        return None
    try:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    except (OSError, ValueError) as error:
        show_error(tr("text.save_figure"), error, parent)
        return None
    return path
