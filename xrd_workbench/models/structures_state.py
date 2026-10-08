"""Accepted structure workspace settings, without Qt or renderer objects."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .viewer import PlotAppearance


@dataclass
class StructureCamera:
    zoom: float = 1.0
    # Pan is a fraction of canvas width/height, independent of widget size.
    pan: tuple[float, float] = (0.0, 0.0)


@dataclass
class StructureViewState:
    uid: str | None = None
    payload: Any = None
    initialized: bool = False
    center_hkl: tuple[int, int, int] = (0, 1, 0)
    base_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    user_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    view_name: str = 'hkl'
    camera: StructureCamera = field(default_factory=StructureCamera)
    atom_component_visibility: dict[str, bool] = field(default_factory=dict)
    atom_component_colours: dict[str, str] = field(default_factory=dict)
    polyhedron_site_visibility: dict[str, bool] = field(default_factory=dict)
    polyhedron_site_colours: dict[str, str] = field(default_factory=dict)
    polyhedron_opaque_sites: set[str] = field(default_factory=set)
    show_atoms: bool = True
    show_external_atoms: bool = True
    show_bonds: bool = True
    show_cell: bool = True
    show_basis: bool = True
    show_polyhedra: bool = False
    engraving: bool = False
    hatching: bool = True
    hatch_density: int = 22
    hatch_grip: int = 50
    atom_scale_percent: int = 100

    def bind(self, payload: Any, uid: str | None = None) -> None:
        if self.payload is payload and self.uid == uid:
            return
        self.payload, self.uid = payload, uid
        self.initialized = False
        self.center_hkl = (0, 1, 0)
        self.base_rotation, self.user_rotation = np.eye(3), np.eye(3)
        self.view_name = 'hkl'
        self.camera = StructureCamera()
        self.atom_component_visibility.clear()
        self.atom_component_colours.clear()
        self.polyhedron_site_visibility.clear()
        self.polyhedron_site_colours.clear()
        self.polyhedron_opaque_sites.clear()


@dataclass
class StructureCalculationState:
    minimum: float = 5.0
    maximum: float = 120.0
    threshold: float = .1
    fwhm: float = .15
    style: str = 'sticks'
    sort_column: int = -1
    sort_descending: bool = False
    column_widths: tuple[int, ...] = (190, 90, 90, 90, 85, 75, 60, 80, 110, 90)
    selected_reflection: tuple[str, str, float] | None = None
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None
    appearance: PlotAppearance = field(default_factory=PlotAppearance)
    # Reflection rows are a recomputable runtime cache, not source-file data.
    rows: list = field(default_factory=list)

    def clear_results(self) -> None:
        self.rows = []
        self.selected_reflection = None
        self.viewport = None


@dataclass
class StructuresState:
    uid: str | None = None
    payload: Any = None
    active_tab: int = 0
    viewer: StructureViewState = field(default_factory=StructureViewState)
    pattern: StructureCalculationState = field(default_factory=StructureCalculationState)
    table: StructureCalculationState = field(default_factory=StructureCalculationState)

    def sync_document(self, document: Any) -> None:
        if self.uid == document.uid and self.payload is document.payload:
            return
        self.uid, self.payload = document.uid, document.payload
        self.viewer.bind(document.payload, document.uid)
        self.pattern.clear_results()
        self.table.clear_results()

    def remove(self, uid: str) -> None:
        if self.uid == uid:
            self.uid = self.payload = None
            self.viewer.bind(None)
            self.pattern.clear_results()
            self.table.clear_results()
