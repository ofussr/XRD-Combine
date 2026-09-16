"""Application-wide colour scheme, independent of the desktop when selected."""

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


THEME_KEYS = {
    "system": "qt.theme_system",
    "light": "qt.theme_light",
    "dark": "qt.theme_dark",
}
SETTINGS_KEY = "appearance/colour_scheme"


def theme_palette(mode: str) -> QPalette:
    """Supply every UI role so forced themes work with any system appearance."""
    if mode not in {"light", "dark"}:
        raise ValueError(mode)
    dark = mode == "dark"
    colours = {
        "Window": ("#f5f6f8", "#242830"),
        "WindowText": ("#20242b", "#eef1f5"),
        "Base": ("#ffffff", "#181c22"),
        "AlternateBase": ("#eef1f5", "#222831"),
        "Text": ("#20242b", "#eef1f5"),
        "Button": ("#e9edf2", "#343b46"),
        "ButtonText": ("#20242b", "#eef1f5"),
        "BrightText": ("#ffffff", "#ffffff"),
        "ToolTipBase": ("#fffde8", "#303641"),
        "ToolTipText": ("#20242b", "#eef1f5"),
        "Highlight": ("#2367ba", "#376db0"),
        "HighlightedText": ("#ffffff", "#ffffff"),
        "Link": ("#175fb8", "#80baff"),
        "LinkVisited": ("#7045a0", "#c09df1"),
        "PlaceholderText": ("#687381", "#a2adba"),
        "Light": ("#ffffff", "#4d5867"),
        "Midlight": ("#e0e5eb", "#414b58"),
        "Mid": ("#a0aab7", "#647080"),
        "Dark": ("#7a8696", "#14171c"),
        "Shadow": ("#596472", "#080a0d"),
        "Accent": ("#2367ba", "#80baff"),
    }
    palette = QPalette()
    for role, pair in colours.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(pair[int(dark)]))
    disabled = QPalette.ColorGroup.Disabled
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText,
                 QPalette.PlaceholderText):
        palette.setColor(disabled, role, QColor("#87919f" if dark else "#798391"))
    palette.setColor(disabled, QPalette.Highlight, QColor("#414b58" if dark else "#d4dce6"))
    palette.setColor(disabled, QPalette.HighlightedText, QColor("#b8c1cd" if dark else "#596472"))
    return palette


class ThemeController(QObject):
    """Keep one theme for every application window and remember the choice."""

    changed = Signal(str)

    def __init__(self, application: QApplication, settings=None):
        super().__init__(application)
        self.application = application
        self.settings = settings if settings is not None else QSettings("XRD Combine", "XRD Combine")
        self.system_style = application.style().objectName()
        self.mode = "system"
        saved = self.settings.value(SETTINGS_KEY, "system")
        self.set_mode(saved if isinstance(saved, str) and saved in THEME_KEYS else "system", persist=False)

    def set_mode(self, mode: str, *, persist: bool = True) -> None:
        if mode not in THEME_KEYS:
            raise ValueError(mode)
        if persist:
            self.settings.setValue(SETTINGS_KEY, mode)
            self.settings.sync()
            if self.settings.status() != QSettings.Status.NoError:
                raise OSError("Could not save the application colour scheme.")
        hints = self.application.styleHints()
        if mode == "system":
            hints.unsetColorScheme()
            # A palette with no explicitly resolved roles restores platform
            # inheritance, including later system colour-scheme changes.
            palette = QPalette()
            palette.setResolveMask(0)
            self.application.setPalette(palette)
            self.application.setStyle(self.system_style)
        else:
            hints.setColorScheme(Qt.ColorScheme.Dark if mode == "dark" else Qt.ColorScheme.Light)
            # Fusion respects all palette roles; some platform-native styles
            # otherwise keep drawing selected controls in the desktop scheme.
            self.application.setPalette(theme_palette(mode))
            # Re-polish after applying the palette: stylesheet-backed controls
            # otherwise retain resolved colours from the previous theme.
            self.application.setStyle("Fusion")
        self.mode = mode
        self.changed.emit(mode)


def application_theme(application=None) -> ThemeController:
    application = application or QApplication.instance()
    if application is None:
        raise RuntimeError("A QApplication must exist before creating the theme controller")
    controller = getattr(application, "_xrd_theme_controller", None)
    if controller is None:
        controller = ThemeController(application)
        application._xrd_theme_controller = controller
    return controller
