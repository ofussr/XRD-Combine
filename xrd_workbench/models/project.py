"""GUI-independent project documents, assignments and history."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import uuid


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
    """Store project objects once and track their workspace assignments."""

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

    def unsubscribe(self, listener: ProjectListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

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

    def _existing(self, key: tuple[str, str, int]) -> ProjectDocument | None:
        uid = self._source_keys.get(key)
        return self.documents.get(uid) if uid is not None else None

    def source_document(
        self,
        kind: str,
        source: str | Path,
        index: int = 0,
    ) -> ProjectDocument | None:
        """Return an already loaded source without invoking a format reader."""

        return self._existing((kind, str(Path(source).resolve()), index))

    def _add(
        self,
        *,
        kind: str,
        name: str,
        source: str | Path,
        payload: Any,
        parent_uid: str | None = None,
        source_key: tuple[str, str, int] | None = None,
    ) -> ProjectDocument:
        if source_key is not None:
            existing = self._existing(source_key)
            if existing is not None:
                return existing
        document = ProjectDocument(
            uid=uuid.uuid4().hex,
            kind=kind,
            name=name,
            source=Path(source),
            payload=payload,
            parent_uid=parent_uid,
        )
        self.documents[document.uid] = document
        if source_key is not None:
            self._source_keys[source_key] = document.uid
        self._emit("added", document)
        return document

    def add_cif_document(self, source: str | Path, payload: Any) -> ProjectDocument:
        source = Path(source)
        key = (CIF, str(source.resolve()), 0)
        return self._add(
            kind=CIF,
            name=payload.name,
            source=source,
            payload=payload,
            source_key=key,
        )

    def add_pole_document(self, source: str | Path, payload: Any) -> ProjectDocument:
        source = Path(source)
        if not getattr(payload, "is_pole_figure", False):
            raise ValueError("RAW file is not a pole-figure measurement")
        key = (POLE_DATA, str(source.resolve()), 0)
        return self._add(
            kind=POLE_DATA,
            name=source.stem,
            source=source,
            payload=payload,
            source_key=key,
        )

    def add_scan(
        self,
        scan: Any,
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
        source = Path(scan.source)
        key = (SCAN, str(source.resolve()), index)
        return self._add(
            kind=SCAN,
            name=scan.name,
            source=source,
            payload=scan,
            parent_uid=parent_uid,
            source_key=None if derived else key,
        )

    def add_cell_phase(self, payload: Any) -> ProjectDocument:
        return self._add(
            kind=CELL_PHASE,
            name=payload.name,
            source=payload.source,
            payload=payload,
        )

    def replace_cell_phase(self, uid: str, payload: Any) -> ProjectDocument:
        document = self.documents[uid]
        if document.kind != CELL_PHASE:
            raise TypeError("Only cell-phase documents can be replaced")
        document.history.append(document.payload)
        document.name = payload.name
        document.source = Path(payload.source)
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
        elif uid in assigned:
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

    def replace_scan(self, uid: str, scan: Any) -> ProjectDocument:
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
