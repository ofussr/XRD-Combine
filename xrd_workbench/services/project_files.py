"""Load files into a project store without depending on a GUI toolkit."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any, Callable


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
        self._digests = {}

    def _stamp(self, source, documents):
        source = Path(source)
        if source.is_file():
            stat = source.stat()
            key = (str(source.resolve()), stat.st_mtime_ns, stat.st_size)
            if key not in self._digests:
                with source.open('rb') as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                    self._digests[key] = digest.hexdigest()
            for document in documents:
                document.source_digest = self._digests[key]
                metadata = getattr(document.payload, 'metadata', None)
                if isinstance(metadata, dict):
                    metadata['source_digest'] = document.source_digest
        return documents

    def load_path(self, store: Any, path: str | Path) -> list[Any]:
        source = Path(path)
        if source.suffix.lower() == ".cif":
            return [self.load_cif(store, source)]
        if source.suffix.lower() == ".raw":
            try:
                raw = self._read_bruker_raw(source)
            except (OSError, ValueError):
                raw = None
            if raw is not None and raw.is_pole_figure:
                pole_document = store.add_pole_document(source, raw)
                scan_documents = [
                    store.add_scan(scan, parent_uid=pole_document.uid)
                    for scan in self._read_raw_scans(source, raw=raw)
                ]
                return self._stamp(source, [pole_document, *scan_documents])
            if raw is not None and getattr(raw, "is_rsm", False):
                rsm_document = store.add_rsm_document(source, raw)
                scan_documents = [
                    store.add_scan(scan, parent_uid=rsm_document.uid)
                    for scan in self._read_raw_scans(source, raw=raw)
                ]
                return self._stamp(source, [rsm_document, *scan_documents])
        return self._stamp(source, [store.add_scan(scan) for scan in self._read_scan_file(source)])

    def load_cif(self, store: Any, path: str | Path) -> Any:
        source = Path(path)
        existing = store.source_document(CIF, source)
        if existing is not None:
            return existing
        return self._stamp(source, [store.add_cif_document(source, self._load_cif_document(source))])[0]

    def load_pole_data(
        self,
        store: Any,
        path: str | Path,
        payload: Any | None = None,
    ) -> Any:
        source = Path(path)
        existing = store.source_document(POLE_DATA, source)
        if existing is not None:
            return existing
        raw = payload if payload is not None else self._read_bruker_raw(source)
        return self._stamp(source, [store.add_pole_document(source, raw)])[0]

    def load_rsm_data(self, store: Any, path: str | Path, payload: Any | None = None) -> Any:
        source = Path(path)
        existing = store.source_document(RSM_DATA, source)
        if existing is not None:
            return existing
        raw = payload if payload is not None else self._read_bruker_raw(source)
        return self._stamp(source, [store.add_rsm_document(source, raw)])[0]
