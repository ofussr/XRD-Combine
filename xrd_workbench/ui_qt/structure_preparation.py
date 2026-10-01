"""Prepare and cache structure scenes without running geometry in the GUI thread."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from ..atom_styles import ELEMENTS, atom_colour, custom_colours, palette
from ..localization import tr
from ..models.project import CIF, CELL_PHASE
from ..services.structure_scene import build_structure_scene
from .unit_cell_adapter import unit_cell_scene


def _style_signature() -> tuple:
    return palette(), tuple(sorted(custom_colours().items()))


@dataclass(frozen=True)
class PreparedStructure:
    geometry: object
    scene: object
    styles: tuple


@dataclass
class _Entry:
    document: object
    cancelled: Event = field(default_factory=Event)
    prepared: PreparedStructure | None = None
    error: str | None = None


class _StructureTask:
    def __init__(self, entry: _Entry) -> None:
        self.entry = entry
        # Capture GUI-owned colour settings before the worker starts.
        self.styles = _style_signature()
        self.colours = {element: atom_colour(element) for element in ELEMENTS}

    def run(self) -> tuple:
        prepared = None
        error = None
        try:
            if not self.entry.cancelled.is_set():
                geometry = build_structure_scene(self.entry.document.crystal)
                if not self.entry.cancelled.is_set():
                    prepared = PreparedStructure(
                        geometry,
                        unit_cell_scene(geometry, colours=self.colours),
                        self.styles,
                    )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        return prepared, error


class StructurePreparation(QObject):
    """One shared job and cached result per project payload.

    Workers calculate geometry and immutable renderer models only. All QObject,
    widget and OpenGL updates stay on the GUI thread. Jobs are serialized to
    avoid competing geometry calculations when several CIFs are imported.
    """

    ready = Signal(object)
    failed = Signal(object, str)
    busy_changed = Signal(bool)

    def __init__(self, store=None, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._entries: dict[int, _Entry] = {}
        self._queue: deque[_Entry] = deque()
        self._active_task = None
        self._busy = False
        self._closed = False
        # Workers own plain Python data only, including after a window closes.
        # Polling on the GUI thread avoids signals from destroyed Qt objects.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xrd-structure")
        self._result_timer = QTimer(self)
        self._result_timer.setInterval(20)
        self._result_timer.timeout.connect(self._collect_result)
        if store is not None:
            store.subscribe(self._project_event)

    @property
    def busy(self) -> bool:
        return self._busy

    def prepare_project(self) -> None:
        if self.store is not None:
            for document in self.store.documents.values():
                if document.kind in {CIF, CELL_PHASE}:
                    self.request(document.payload)

    def request(self, document) -> PreparedStructure | None:
        if self._closed or getattr(document, "crystal", None) is None:
            return None
        key = id(document)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(document)
            self._entries[key] = entry
            self._queue.append(entry)
            self._update_busy()
            self._start_next()
        elif entry.prepared is not None:
            if entry.prepared.styles != _style_signature():
                entry.prepared = PreparedStructure(
                    entry.prepared.geometry,
                    unit_cell_scene(entry.prepared.geometry),
                    _style_signature(),
                )
        return entry.prepared

    def error(self, document) -> str | None:
        entry = self._entries.get(id(document))
        return entry.error if entry is not None else None

    def _project_event(self, event, document, _workspace) -> None:
        if self._closed:
            return
        if event in {"removed", "replaced"}:
            live_payloads = {id(item.payload) for item in self.store.documents.values()}
            for key in tuple(self._entries):
                if key not in live_payloads:
                    self._entries.pop(key).cancelled.set()
            self._update_busy()
        if event in {"added", "replaced"} and document.kind in {CIF, CELL_PHASE}:
            self.request(document.payload)

    def _start_next(self) -> None:
        if self._closed or self._active_task is not None:
            return
        while self._queue:
            entry = self._queue.popleft()
            if entry.cancelled.is_set():
                continue
            task = _StructureTask(entry)
            self._active_task = task, self._pool.submit(task.run)
            self._result_timer.start()
            return

    @Slot()
    def _collect_result(self) -> None:
        if self._closed or self._active_task is None:
            return
        task, future = self._active_task
        if future.done():
            self._result_timer.stop()
            prepared, error = future.result()
            self._complete(task.entry, prepared, error)

    def _complete(self, entry, prepared, error) -> None:
        self._active_task = None
        if self._closed:
            return
        valid = self._entries.get(id(entry.document)) is entry
        if valid and not entry.cancelled.is_set():
            entry.prepared = prepared
            entry.error = error
        self._update_busy()
        self._start_next()
        if valid and not entry.cancelled.is_set():
            if error is not None:
                self.failed.emit(entry.document, error)
            elif prepared is not None:
                self.ready.emit(entry.document)

    def _update_busy(self) -> None:
        busy = any(entry.prepared is None and entry.error is None
                   for entry in self._entries.values())
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def close(self) -> None:
        """Discard queued work and results; never wait in a window close handler."""
        self._closed = True
        self._result_timer.stop()
        if self.store is not None:
            self.store.unsubscribe(self._project_event)
        for entry in self._entries.values():
            entry.cancelled.set()
        self._entries.clear()
        self._queue.clear()
        self._update_busy()
        self._pool.shutdown(wait=False, cancel_futures=True)


class LoadingStructureDialog(QDialog):
    """Small nonmodal notice with one static caption and no progress stages."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint
                         | Qt.WindowType.WindowTitleHint)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle("XRD Combine")
        self.label.setText(tr("qt.loading_structure"))
        self.setFixedSize(self.sizeHint())

    @Slot(bool)
    def set_busy(self, busy: bool) -> None:
        if busy:
            parent = self.parentWidget()
            if parent is not None:
                self.move(parent.frameGeometry().center() - self.rect().center())
            self.show()
        else:
            self.hide()

    def reject(self) -> None:
        # Escape must not dismiss the notice while preparation is still running.
        pass
