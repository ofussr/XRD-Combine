"""Load files into a project store without depending on a GUI toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from ..models.project import CIF, POLE_DATA
except ImportError:  # pragma: no cover - direct module execution
    from models.project import CIF, POLE_DATA


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
                return [pole_document, *scan_documents]
        return [store.add_scan(scan) for scan in self._read_scan_file(source)]

    def load_cif(self, store: Any, path: str | Path) -> Any:
        source = Path(path)
        existing = store.source_document(CIF, source)
        if existing is not None:
            return existing
        return store.add_cif_document(source, self._load_cif_document(source))

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
        return store.add_pole_document(source, raw)
