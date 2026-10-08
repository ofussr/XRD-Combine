"""Source-file drops across nested widgets of the main workspace."""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from ..localization import tr
from ..services.project_files import SUPPORTED_SUFFIXES


def dropped_paths(mime):
    if not mime.hasUrls():
        return []
    from pathlib import Path
    paths = []
    for url in mime.urls():
        if not url.isLocalFile():
            return []
        path = Path(url.toLocalFile())
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            return []
        value = str(path.resolve())
        if value not in paths:
            paths.append(value)
    return paths


class SourceFileDrop(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.hint = QLabel(window)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet("background: rgba(25, 80, 135, 220); color: white; "
                               "border: 2px solid #84c7ff; border-radius: 10px; "
                               "font-size: 18px; padding: 18px;")
        self.hint.hide()
        window.setAcceptDrops(True)
        QApplication.instance().installEventFilter(self)

    def _show_hint(self):
        self.hint.setText(tr("qt.drop_source_files"))
        self.hint.adjustSize()
        self.hint.move((self.window.width() - self.hint.width()) // 2,
                       (self.window.height() - self.hint.height()) // 2)
        self.hint.raise_()
        self.hint.show()

    def eventFilter(self, watched, event):
        if not isinstance(watched, QWidget) or watched.window() is not self.window:
            return False
        kind = event.type()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            if not event.mimeData().hasUrls():
                return False  # Keep ordinary text and internal drags intact.
            paths = dropped_paths(event.mimeData())
            if not paths:
                self.hint.hide()
                event.ignore()
                return True  # Prevent an input field from consuming file URLs.
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            if kind == QEvent.Type.Drop:
                self.hint.hide()
                self.window.open_external_files(paths, activate=False)
            else:
                self._show_hint()
            return True
        if (kind == QEvent.Type.DragLeave or watched is self.window
                and kind in (QEvent.Type.Hide, QEvent.Type.Close)):
            self.hint.hide()
        return False

    def close(self):
        QApplication.instance().removeEventFilter(self)
        self.hint.hide()
