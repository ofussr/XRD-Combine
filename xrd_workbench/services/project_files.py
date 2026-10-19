"""Load files into a project store without depending on a GUI toolkit."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Callable

from ..io.source_snapshot import source_snapshot


SUPPORTED_SUFFIXES = frozenset({".xrdml", ".xml", ".raw", ".xy", ".txt",
                              ".dat", ".csv", ".cif"})

try:
    from ..models.project import CIF, POLE_DATA, RSM_DATA
except ImportError:  # pragma: no cover - direct module execution
    from models.project import CIF, POLE_DATA, RSM_DATA


class ProjectFileService:
    """Coordinate format readers and the GUI-independent project model."""

    def __init__(
        self,
        *,
        load_cif_document: Callable[[Path], Any],
        read_bruker_raw: Callable[[Path], Any],
        read_raw_scans: Callable[..., list[Any]],
        read_scan_file: Callable[[Path], list[Any]],
    ) -> None:
        self._load_cif_document = load_cif_document
        self._read_bruker_raw = read_bruker_raw
        self._read_raw_scans = read_raw_scans
        self._read_scan_file = read_scan_file

    @staticmethod
    def _stamp(digest, documents, *, existing_uids=()):
        for document in documents:
            if document.uid in existing_uids:
                continue
            document.source_digest = digest
            metadata = getattr(document.payload, 'metadata', None)
            if isinstance(metadata, dict) and digest is not None:
                metadata['source_digest'] = digest
        return documents

    @staticmethod
    def _scan_source(scans, source, digest):
        for scan in scans:
            scan.source = source
            if digest is not None:
                scan.metadata['source_digest'] = digest
        return scans

    def load_path(self, store: Any, path: str | Path) -> list[Any]:
        source = Path(path)
        if source.suffix.lower() == '.cif':
            return [self.load_cif(store, source)]
        existing_uids = set(store.documents)
        raw, parent_kind = None, None
        with source_snapshot(source) as snapshot:
            if source.suffix.lower() == '.raw':
                try:
                    raw = self._read_bruker_raw(snapshot.path)
                except (OSError, ValueError):
                    raw = None
                if raw is not None and (raw.is_pole_figure or getattr(raw, 'is_rsm', False)):
                    parent_kind = POLE_DATA if raw.is_pole_figure else RSM_DATA
            scans = (self._read_raw_scans(snapshot.path, raw=raw) if parent_kind else
                     self._read_scan_file(snapshot.path))
            self._scan_source(scans, source, snapshot.digest)
        # The entire input is parsed and verified before the first 'added' event.
        documents = []
        if parent_kind:
            raw.path = source
            if isinstance(getattr(raw, 'metadata', None), dict) and snapshot.digest is not None:
                raw.metadata['source_digest'] = snapshot.digest
            parent = (store.add_pole_document(source, raw) if parent_kind == POLE_DATA else
                      store.add_rsm_document(source, raw))
            documents.append(parent)
            documents.extend(store.add_scan(scan, parent_uid=parent.uid) for scan in scans)
        else:
            documents.extend(store.add_scan(scan) for scan in scans)
        return self._stamp(snapshot.digest, documents, existing_uids=existing_uids)

    def load_cif(self, store: Any, path: str | Path) -> Any:
        source = Path(path)
        existing = store.source_document(CIF, source)
        if existing is not None:
            return existing
        with source_snapshot(source) as snapshot:
            payload = self._load_cif_document(snapshot.path)
        if snapshot.digest is not None:
            payload = replace(payload, source=source)
            payload.data.source = source
        return self._stamp(snapshot.digest, [store.add_cif_document(source, payload)])[0]

    def _load_map(self, store, source, kind, payload):
        source = Path(source)
        existing = store.source_document(kind, source)
        if existing is not None:
            return existing
        with source_snapshot(source) as snapshot:
            # Pre-parsed external payloads have no verifiable relationship to
            # the current bytes. Re-read real sources through the snapshot.
            raw = (self._read_bruker_raw(snapshot.path) if snapshot.digest is not None or payload is None
                   else payload)
        raw.path = source
        if isinstance(getattr(raw, 'metadata', None), dict) and snapshot.digest is not None:
            raw.metadata['source_digest'] = snapshot.digest
        document = (store.add_pole_document(source, raw) if kind == POLE_DATA else
                    store.add_rsm_document(source, raw))
        return self._stamp(snapshot.digest, [document])[0]

    def load_pole_data(self, store: Any, path: str | Path, payload: Any | None = None) -> Any:
        return self._load_map(store, path, POLE_DATA, payload)

    def load_rsm_data(self, store: Any, path: str | Path, payload: Any | None = None) -> Any:
        return self._load_map(store, path, RSM_DATA, payload)
