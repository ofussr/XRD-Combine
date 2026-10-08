"""Command-line bootstrap for XRD Combine."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from ..version import APP_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="XRD Combine",
        description="XRD Combine desktop application.",
    )
    parser.add_argument("paths", nargs="*", help="measurement and CIF files to open")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        # unit-cell-gui must configure shared OpenGL contexts before the first
        # QApplication is constructed.  Importing MainWindow is too late:
        # structure_viewer only reaches this helper while its modules load.
        from unit_cell_gui import configure_qt_opengl

        configure_qt_opengl()
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name.startswith("PySide6")
            or exc.name.startswith("unit_cell_gui")
        ):
            print(
                "PySide6 and unit-cell-gui are required for XRD Combine 3. "
                "Install dependencies with: py -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 2
        raise
    except ImportError as exc:
        print(f"The PySide6 runtime could not be loaded: {exc}", file=sys.stderr)
        return 2

    from ..application_resources import resource_path
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("XRD Combine")
    application.setApplicationVersion(APP_VERSION)
    from .single_instance import SingleInstance
    instance = SingleInstance()
    try:
        primary = instance.start_or_forward(arguments.paths)
    except OSError as exc:
        from PySide6.QtWidgets import QMessageBox
        _close_splash()
        QMessageBox.warning(None, "XRD Combine", str(exc))
        return 2
    if not primary:
        _close_splash()
        return 0
    try:
        # The IPC thread is already accepting requests while scientific modules
        # and workspace widgets load. No second MainWindow is constructed.
        from .main_window import MainWindow

        icon_path = resource_path("xrd_combine.ico")
        if icon_path is not None:
            application.setWindowIcon(QIcon(str(icon_path)))
        window = MainWindow(initial_paths=arguments.paths)
        window.show()
        instance.attach(window.open_external_files)
        application.processEvents()
        _close_splash()
        return application.exec()
    finally:
        instance.close()


def _close_splash():
    try:
        import pyi_splash
    except ImportError:
        pass  # Ordinary Python launches have no PyInstaller splash module.
    else:
        pyi_splash.close()


__all__ = ["main"]
