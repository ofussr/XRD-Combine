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
    parser.add_argument("paths", nargs="*", help="project files to open")
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

    from .main_window import MainWindow

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("XRD Combine")
    application.setApplicationVersion(APP_VERSION)
    window = MainWindow(initial_paths=arguments.paths)
    window.show()
    return application.exec()


__all__ = ["main"]
