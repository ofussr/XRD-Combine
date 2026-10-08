"""Accepted pole-figure workspace state, independent of widgets and renderers.

Parsed measurements and calculated reflection lists are runtime caches. Source
UIDs, geometry and display settings are the state needed to reconstruct a view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .pole_figure import CalculatedPoleLayer
from .structures_state import StructureViewState


@dataclass
class CalculatedPoleState:
    preview: StructureViewState = field(default_factory=StructureViewState)
    primary_uid: str | None = None
    overlay_uid: str | None = None
    cif_document: Any = None
    crystal: Any = None
    center_hkl: tuple[int, int, int] = (0, 1, 0)
    base_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    user_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    reflections: list = field(default_factory=list)
    points: list = field(default_factory=list)
    point_groups: list = field(default_factory=list)
    selected_hkl: tuple[int, int, int] | None = None
    selected_layer_index: int = 0
    intensity_by_spacing: dict | None = None
    overlay_layer: CalculatedPoleLayer | None = None
    initialized: bool = False
    reflection_radiations: tuple = ()
    single_colour_mode: str | None = None
    max_index: int = 4
    wavelength: float = 1.5406
    d_range: tuple[float, float] = (1.0, 13.0)
    two_theta_range: tuple[float, float] | None = None
    range_mode: str = 'd'
    projection: str = 'stereographic'
    labels: bool = False
    label_leaders: bool = False
    coincident_outlines: bool = False
    angle_labels: bool = True
    show_structure: bool = False
    basis_visible: bool = True
    size_by_d: bool = False
    color_mode: str = 'uniform'
    primary_colour: str = '#2d6da3'
    primary_opacity: float = 100.0
    primary_size: float = 100.0
    global_size: float = 100.0
    overlay_colour: str = '#d65f3c'
    overlay_opacity: float = 70.0
    overlay_size: float = 100.0
    joint_rotation: bool = False
    matplotlib_view: tuple[tuple[float, float], tuple[float, float]] | None = None
    pyqtgraph_view: tuple[tuple[float, float], tuple[float, float]] | None = None

    def set_primary(self, payload: Any, uid: str | None = None) -> None:
        self.preview.bind(payload, uid)
        self.primary_uid = uid
        self.cif_document = payload
        self.crystal = getattr(payload, "crystal", None)
        self.center_hkl = (0, 1, 0)
        self.base_rotation = np.eye(3)
        self.user_rotation = np.eye(3)
        self.reflections, self.points, self.point_groups = [], [], []
        self.selected_hkl = None
        self.selected_layer_index = 0
        self.intensity_by_spacing = None
        self.initialized = False
        self.matplotlib_view = self.pyqtgraph_view = None

    def remove_overlay(self) -> None:
        self.overlay_layer = None
        self.overlay_uid = None
        self.selected_layer_index = 0
        self.joint_rotation = False
        if self.single_colour_mode is not None:
            self.color_mode = self.single_colour_mode

    def remove(self, uid: str) -> None:
        if uid == self.overlay_uid:
            self.remove_overlay()
        if uid == self.primary_uid:
            self._remove_primary()

    def remove_payload(self, payload: Any) -> None:
        """Support standalone adapters as well as project UID assignments."""
        if self.overlay_layer is not None and self.overlay_layer.document is payload:
            self.remove_overlay()
        if self.cif_document is payload:
            self._remove_primary()

    def _remove_primary(self) -> None:
        layer, promoted_uid = self.overlay_layer, self.overlay_uid
        matplotlib_view, pyqtgraph_view = self.matplotlib_view, self.pyqtgraph_view
        self.remove_overlay()
        self.set_primary(None)
        if layer is None:
            return
        self.primary_uid, self.cif_document, self.crystal = promoted_uid, layer.document, layer.crystal
        for name in ('center_hkl', 'base_rotation', 'user_rotation', 'reflections',
                     'points', 'point_groups', 'selected_hkl', 'intensity_by_spacing'):
            setattr(self, name, getattr(layer, name))
        self.primary_colour, self.primary_opacity, self.primary_size = (
            layer.colour, layer.opacity_percent, layer.size_percent)
        self.preview.bind(layer.document, promoted_uid)
        self.initialized = layer.initialized
        self.matplotlib_view, self.pyqtgraph_view = matplotlib_view, pyqtgraph_view
        if getattr(layer.document, 'is_cell_only', False) and self.color_mode == 'intensity':
            self.color_mode = 'uniform'

    def sync_document(self, document: Any) -> None:
        if document.uid == self.primary_uid:
            if document.payload is not self.cif_document:
                self.set_primary(document.payload, document.uid)
            return
        if document.uid == self.overlay_uid and self.overlay_layer is not None:
            if document.payload is not self.overlay_layer.document:
                self.overlay_layer = CalculatedPoleLayer(document.payload, self.overlay_colour,
                    self.overlay_opacity, self.overlay_size)
            return
        if self.primary_uid is None:
            self.set_primary(document.payload, document.uid)
        elif self.overlay_layer is None:
            self.overlay_uid = document.uid
            self.overlay_layer = CalculatedPoleLayer(document.payload, self.overlay_colour,
                self.overlay_opacity, self.overlay_size)


@dataclass
class ExperimentalPoleState:
    uid: str | None = None
    source_document: Any = None
    measurement: Any = None
    raw_path: Any = None
    scans: list = field(default_factory=list)
    loaded_radii: np.ndarray | None = None
    manual_angles: tuple[float, float, float] | None = None
    intensity_limits: tuple[float, float] | None = None
    display_mode: str = 'colour'
    scale_mode: str = 'linear'
    fill_colour: str = '#2a788e'
    zoom_factor: float = 1.0
    # Renderer-specific viewports; no widget objects or pixel coordinates.
    matplotlib_view: tuple[float, float, float, float] | None = None
    pyqtgraph_view: tuple[tuple[float, float], tuple[float, float]] | None = None

    def clear_data(self) -> None:
        self.uid = None
        self.source_document = None
        self.measurement = self.raw_path = self.loaded_radii = None
        self.scans = []
        self.manual_angles = self.intensity_limits = None
        self.zoom_factor = 1.0
        self.matplotlib_view = self.pyqtgraph_view = None

    def sync_document(self, document: Any) -> None:
        if self.uid == document.uid and self.source_document is document.payload:
            return
        self.clear_data()
        self.uid, self.source_document = document.uid, document.payload


@dataclass
class PolesState:
    calculated: CalculatedPoleState = field(default_factory=CalculatedPoleState)
    experimental: ExperimentalPoleState = field(default_factory=ExperimentalPoleState)
    active_tab: int = 0

    def sync_document(self, document: Any) -> None:
        if document.kind in {'cif', 'cell_phase'}:
            self.calculated.sync_document(document)
        elif document.kind == 'pole_data':
            self.experimental.sync_document(document)

    def remove(self, uid: str) -> None:
        self.calculated.remove(uid)
        if self.experimental.uid == uid:
            self.experimental.clear_data()
