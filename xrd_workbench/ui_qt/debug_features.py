"""Remember explicit opt-ins to experimental interface features."""

from PySide6.QtCore import QObject, QSettings, Signal

INDEXATION_KEY = "debug/indexation_enabled"


class DebugFeatures(QObject):
    indexation_changed = Signal(bool)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else QSettings("XRD Combine", "XRD Combine")
        saved = self.settings.value(INDEXATION_KEY, False)
        self.indexation_enabled = saved is True or str(saved).lower() == "true"

    def set_indexation_enabled(self, enabled: bool):
        enabled = bool(enabled)
        self.settings.setValue(INDEXATION_KEY, enabled)
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            raise OSError("Could not save the indexation setting.")
        if enabled != self.indexation_enabled:
            self.indexation_enabled = enabled
            self.indexation_changed.emit(enabled)
