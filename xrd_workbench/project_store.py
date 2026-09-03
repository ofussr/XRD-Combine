"""Compatibility facade for the shared project model and file service.

The GUI-independent store lives in :mod:`xrd_workbench.models.project`.  This
facade preserves the 2.x ``add_path`` API used by the Tkinter interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .bruker_raw import read_bruker_raw
    from .cif_document import load_cif_document
    from .models.project import (
        CELL_PHASE,
        CIF,
        POLE_DATA,
        POLES,
        SCAN,
        STRUCTURES,
        VIEWER,
        WORKSPACES,
        ProjectDocument,
        ProjectListener,
        ProjectStore as ProjectModel,
    )
    from .services.project_files import ProjectFileService
    from .xrd_io import read_raw_scans, read_scan_file
except ImportError:  # pragma: no cover - direct module execution
    from bruker_raw import read_bruker_raw
    from cif_document import load_cif_document
    from models.project import (
        CELL_PHASE,
        CIF,
        POLE_DATA,
        POLES,
        SCAN,
        STRUCTURES,
        VIEWER,
        WORKSPACES,
        ProjectDocument,
        ProjectListener,
        ProjectStore as ProjectModel,
    )
    from services.project_files import ProjectFileService
    from xrd_io import read_raw_scans, read_scan_file


def _file_service() -> ProjectFileService:
    """Build the adapter from current readers, including patched test readers."""

    return ProjectFileService(
        load_cif_document=load_cif_document,
        read_bruker_raw=read_bruker_raw,
        read_raw_scans=read_raw_scans,
        read_scan_file=read_scan_file,
    )


class ProjectStore(ProjectModel):
    """2.x-compatible project store backed by the separated model and service."""

    def add_path(self, path: str | Path) -> list[ProjectDocument]:
        return _file_service().load_path(self, path)

    def add_cif(self, path: str | Path) -> ProjectDocument:
        return _file_service().load_cif(self, path)

    def add_pole_data(
        self,
        path: str | Path,
        payload: Any | None = None,
    ) -> ProjectDocument:
        return _file_service().load_pole_data(self, path, payload)


__all__ = [
    "CELL_PHASE",
    "CIF",
    "POLE_DATA",
    "POLES",
    "SCAN",
    "STRUCTURES",
    "VIEWER",
    "WORKSPACES",
    "ProjectDocument",
    "ProjectListener",
    "ProjectStore",
]
