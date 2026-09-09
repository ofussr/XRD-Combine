"""Command-line bootstrap for the PySide6 transition preview."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from ..version import QT_PREVIEW_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="XRD Combine (PySide6)",
        description="Parallel PySide6 transition preview for XRD Combine.",
    )
    parser.add_argument("paths", nargs="*", help="project files to open")
    parser.add_argument("--version", action="version", version=QT_PREVIEW_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            print(
                "PySide6 is required for the 3.0 preview. "
                "Install it with: py -m pip install -r requirements-qt.txt",
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
    application.setApplicationVersion(QT_PREVIEW_VERSION)
    window = MainWindow(initial_paths=arguments.paths)
    window.show()
    return application.exec()


__all__ = ["main"]
