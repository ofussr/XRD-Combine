"""Shared mouse interaction rules for the PyQtGraph surfaces."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt


def handle_navigation_drag(view_box, event, axis=None) -> bool:
    """Pan with the right button and reserve the middle button.

    PyQtGraph normally pans with the middle button and scales with the right
    button.  XRD Combine uses the same convention as ``unit-cell-gui``:
    right-drag pans, while left-button gestures remain surface-specific.
    """

    button = event.button()
    if button == Qt.MouseButton.MiddleButton:
        event.accept()
        return True
    if button != Qt.MouseButton.RightButton:
        return False

    enabled = np.asarray(view_box.state["mouseEnabled"], dtype=bool)
    if axis is not None:
        enabled[1 - int(axis)] = False
    current = view_box.mapToView(event.pos())
    previous = view_box.mapToView(event.lastPos())
    x = float(previous.x() - current.x()) if enabled[0] else None
    y = float(previous.y() - current.y()) if enabled[1] else None
    view_box._resetTarget()
    if x is not None or y is not None:
        view_box.translateBy(x=x, y=y)
    view_box.sigRangeChangedManually.emit(view_box.state["mouseEnabled"])
    event.accept()
    return True
