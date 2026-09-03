"""Общий пул загруженных документов и их привязки к разделам программы."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import uuid

try:
    from .bruker_raw import read_bruker_raw
    from .cif_document import CifDocument, load_cif_document
    from .cell_phase import CellPhaseDocument
    from .xrd_io import Scan1D, read_raw_scans, read_scan_file
except ImportError:  # pragma: no cover - прямой запуск модуля
    from bruker_raw import read_bruker_raw
    from cif_document import CifDocument, load_cif_document
    from cell_phase import CellPhaseDocument
    from xrd_io import Scan1D, read_raw_scans, read_scan_file


VIEWER = "viewer"
STRUCTURES = "structures"
POLES = "poles"
WORKSPACES = (VIEWER, STRUCTURES, POLES)

SCAN = "scan"
CIF = "cif"
POLE_DATA = "pole_data"
CELL_PHASE = "cell_phase"


@dataclass
class ProjectDocument:
    uid: str
    kind: str
    name: str
    source: Path
    payload: Any
    parent_uid: str | None = None
    history: list[Any] = field(default_factory=list)


ProjectListener = Callable[[str, ProjectDocument | None, str | None], None]


class ProjectStore:
    """Хранит документы один раз и отдельно помнит их использование в разделах."""

    def __init__(self) -> None:
        self.documents: OrderedDict[str, ProjectDocument] = OrderedDict()
        self.assignments: dict[str, set[str]] = {
            workspace: set() for workspace in WORKSPACES
        }
        self._source_keys: dict[tuple[str, str, int], str] = {}
        self._listeners: list[ProjectListener] = []

    def subscribe(self, listener: ProjectListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def _emit(
        self,
        event: str,
        document: ProjectDocument | None,
        workspace: str | None = None,
    ) -> None:
        for listener in tuple(self._listeners):
            listener(event, document, workspace)

    @staticmethod
    def compatible(kind: str, workspace: str) -> bool:
        return kind in {
            VIEWER: {SCAN, CIF, CELL_PHASE},
            STRUCTURES: {CIF, CELL_PHASE},
            POLES: {CIF, CELL_PHASE, POLE_DATA},
        }.get(workspace, set())

    def add_path(self, path: str | Path) -> list[ProjectDocument]:
        source = Path(path)
        if source.suffix.lower() == ".cif":
            return [self.add_cif(source)]
        if source.suffix.lower() == ".raw":
            try:
                raw = read_bruker_raw(source)
            except (OSError, ValueError):
                raw = None
            if raw is not None and raw.is_pole_figure:
                pole_document = self.add_pole_data(source, raw)
                scan_documents = [
                    self.add_scan(scan, parent_uid=pole_document.uid)
                    for scan in read_raw_scans(source, raw=raw)
                ]
                return [pole_document, *scan_documents]
        return [self.add_scan(scan) for scan in read_scan_file(source)]

    def add_pole_data(self, path: str | Path, payload=None) -> ProjectDocument:
        source = Path(path)
        key = (POLE_DATA, str(source.resolve()), 0)
        existing = self._source_keys.get(key)
        if existing in self.documents:
            return self.documents[existing]
        raw = payload if payload is not None else read_bruker_raw(source)
        if not raw.is_pole_figure:
            raise ValueError("RAW file is not a pole-figure measurement")
        document = ProjectDocument(
            uid=uuid.uuid4().hex,
            kind=POLE_DATA,
            name=source.stem,
            source=source,
            payload=raw,
        )
        self.documents[document.uid] = document
        self._source_keys[key] = document.uid
        self._emit("added", document)
        return document

    def add_cif(self, path: str | Path) -> ProjectDocument:
        source = Path(path)
        key = (CIF, str(source.resolve()), 0)
        existing = self._source_keys.get(key)
        if existing in self.documents:
            return self.documents[existing]
        payload = load_cif_document(source)
        document = ProjectDocument(
            uid=uuid.uuid4().hex,
            kind=CIF,
            name=payload.name,
            source=source,
            payload=payload,
        )
        self.documents[document.uid] = document
        self._source_keys[key] = document.uid
        self._emit("added", document)
        return document

    def add_scan(
        self,
        scan: Scan1D,
        *,
        derived: bool = False,
        parent_uid: str | None = None,
    ) -> ProjectDocument:
        index = int(
            scan.metadata.get(
                "scan_index",
                scan.metadata.get("range_index", 0),
            )
            or 0
        )
        key = (SCAN, str(Path(scan.source).resolve()), index)
        if not derived:
            existing = self._source_keys.get(key)
            if existing in self.documents:
                return self.documents[existing]
        document = ProjectDocument(
            uid=uuid.uuid4().hex,
            kind=SCAN,
            name=scan.name,
            source=Path(scan.source),
            payload=scan,
            parent_uid=parent_uid,
        )
        self.documents[document.uid] = document
        if not derived:
            self._source_keys[key] = document.uid
        self._emit("added", document)
        return document

    def add_cell_phase(self, payload: CellPhaseDocument) -> ProjectDocument:
        document = ProjectDocument(
            uid=uuid.uuid4().hex,
            kind=CELL_PHASE,
            name=payload.name,
            source=payload.source,
            payload=payload,
        )
        self.documents[document.uid] = document
        self._emit("added", document)
        return document

    def replace_cell_phase(
        self,
        uid: str,
        payload: CellPhaseDocument,
    ) -> ProjectDocument:
        document = self.documents[uid]
        if document.kind != CELL_PHASE:
            raise TypeError("Only cell-phase documents can be replaced")
        document.history.append(document.payload)
        document.name = payload.name
        document.source = payload.source
        document.payload = payload
        self._emit("replaced", document)
        return document

    def assign(self, uid: str, workspace: str, enabled: bool = True) -> None:
        document = self.documents[uid]
        if enabled and not self.compatible(document.kind, workspace):
            raise ValueError(f"{document.kind} is not compatible with {workspace}")
        assigned = self.assignments[workspace]
        changed = False
        if enabled:
            exclusive_kinds: set[str] = set()
            if workspace in {STRUCTURES, POLES} and document.kind in {CIF, CELL_PHASE}:
                exclusive_kinds = {CIF, CELL_PHASE}
            elif workspace == POLES and document.kind == POLE_DATA:
                exclusive_kinds = {POLE_DATA}
            for other_uid in tuple(assigned):
                other = self.documents.get(other_uid)
                if (
                    other_uid != uid
                    and other is not None
                    and other.kind in exclusive_kinds
                ):
                    assigned.remove(other_uid)
                    self._emit("unassigned", other, workspace)
            if uid not in assigned:
                assigned.add(uid)
                changed = True
        elif not enabled and uid in assigned:
            assigned.remove(uid)
            changed = True
        if changed:
            self._emit("assigned" if enabled else "unassigned", document, workspace)

    def is_assigned(self, uid: str, workspace: str) -> bool:
        return uid in self.assignments.get(workspace, set())

    def assigned_documents(
        self,
        workspace: str,
        *,
        kind: str | None = None,
    ) -> list[ProjectDocument]:
        assigned = self.assignments.get(workspace, set())
        return [
            document
            for uid, document in self.documents.items()
            if uid in assigned and (kind is None or document.kind == kind)
        ]

    def replace_scan(self, uid: str, scan: Scan1D) -> ProjectDocument:
        document = self.documents[uid]
        if document.kind != SCAN:
            raise TypeError("Only scan documents can be replaced")
        document.history.append(document.payload)
        document.payload = scan
        document.name = scan.name
        document.source = Path(scan.source)
        self._emit("replaced", document)
        return document

    def remove(self, uid: str) -> None:
        document = self.documents.pop(uid)
        for workspace, assigned in self.assignments.items():
            if uid in assigned:
                assigned.remove(uid)
                self._emit("unassigned", document, workspace)
        for key, value in tuple(self._source_keys.items()):
            if value == uid:
                self._source_keys.pop(key, None)
        self._emit("removed", document)

    def clear(self) -> None:
        for uid in tuple(self.documents):
            self.remove(uid)
