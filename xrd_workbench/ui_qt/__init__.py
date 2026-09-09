"""Lazy entry point for the parallel PySide6 interface."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Qt interface without importing PySide6 at package import time."""

    from .app import main as run

    return run(argv)


__all__ = ["main"]
