"""Optional native Qt runtime shared by integration tests."""
import importlib.util
import os

Qt = None
PYSIDE_INSTALLED = importlib.util.find_spec("PySide6") is not None
PYSIDE_AVAILABLE = PYSIDE_INSTALLED
QApplication = None
if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except (ImportError, OSError):
        PYSIDE_AVAILABLE = False

