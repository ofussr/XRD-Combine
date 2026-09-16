"""Application-wide plot renderer with a hidden diagnostic override."""

from __future__ import annotations

from importlib.util import find_spec

from PySide6.QtCore import QObject, QSettings, Signal


PLOT_RENDERER_KEYS = {
    "matplotlib": "qt.plot_renderer_matplotlib",
    "pyqtgraph": "qt.plot_renderer_pyqtgraph",
}
SETTINGS_KEY = "debug/plot_renderer"
LEGACY_SETTINGS_KEY = "appearance/plot_renderer"


def pyqtgraph_available() -> bool:
    """Return whether the experimental renderer can be imported."""
    return find_spec("pyqtgraph") is not None


class PlotRendererController(QObject):
    """Use PyQtGraph normally and remember an explicit debug override."""

    changed = Signal(str)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = (
            settings
            if settings is not None
            else QSettings("XRD Combine", "XRD Combine")
        )
        saved = self.settings.value(SETTINGS_KEY, "pyqtgraph")
        self.mode = (
            saved
            if isinstance(saved, str)
            and saved in PLOT_RENDERER_KEYS
            and (saved != "pyqtgraph" or pyqtgraph_available())
            else "matplotlib"
        )

    def set_mode(self, mode: str, *, persist: bool = True) -> None:
        if mode not in PLOT_RENDERER_KEYS:
            raise ValueError(mode)
        if mode == "pyqtgraph" and not pyqtgraph_available():
            raise RuntimeError("PyQtGraph is not installed.")
        if persist:
            self.settings.setValue(SETTINGS_KEY, mode)
            self.settings.sync()
            if self.settings.status() != QSettings.Status.NoError:
                raise OSError("Could not save the plot-renderer setting.")
        if mode == self.mode:
            return
        self.mode = mode
        self.changed.emit(mode)
