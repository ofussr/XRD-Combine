"""Accepted reciprocal-space map configuration and source lifecycle."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RSMOrientation:
    surface: tuple[int, int, int] = (0, 0, 1)
    inplane: tuple[int, int, int] = (1, 0, 0)
    max_index: int = 8
    tolerance: float = .02


@dataclass
class ExperimentalRSMState:
    sources: OrderedDict = field(default_factory=OrderedDict)
    source_key: tuple = ()
    manual_angles: tuple[float, float, float] | None = None
    derived_angle: str | None = None
    coordinates: str = 'angular'
    extent: str = 'real'
    scale: str = 'linear'
    colour_map: str = 'viridis'
    phase_uid: str | None = None
    overlay_enabled: bool = False
    orientation: RSMOrientation = field(default_factory=RSMOrientation)
    viewports: dict = field(default_factory=dict)
    data: Any = None

    def selected_sources(self) -> list:
        raw = [doc for doc in self.sources.values() if doc.kind == 'rsm_data']
        return raw or [doc for doc in self.sources.values() if doc.kind == 'scan']

    def refresh_sources(self) -> None:
        key = tuple((doc.uid, id(doc.payload)) for doc in self.selected_sources())
        if key != self.source_key:
            self.source_key = key
            self.manual_angles = None
            self.derived_angle = None
            self.overlay_enabled = False
            self.viewports.clear()
            self.data = None


@dataclass
class CalculatedRSMState:
    uid: str | None = None
    payload: Any = None
    orientation: RSMOrientation = field(default_factory=RSMOrientation)
    target_mode: str = 'hkl'
    targets: dict = field(default_factory=lambda: {'hkl': (0, 0, 2), 'angles': (22.5, 45.0), 'q': (0.0, 3.0)})
    coordinates: str = 'angular'
    spans: dict = field(default_factory=lambda: {'angular': (1.0, 1.0), 'q': (.1, .1)})
    labels: bool = True
    target_marker: bool = True
    has_calculated: bool = False
    viewports: dict = field(default_factory=dict)
    request_key: tuple | None = None

    def bind(self, payload: Any, uid: str | None = None) -> None:
        if self.payload is payload and self.uid == uid:
            return
        self.payload, self.uid = payload, uid
        self.orientation = RSMOrientation()
        self.target_mode = 'hkl'
        self.targets = {'hkl': (0, 0, 2), 'angles': (22.5, 45.0), 'q': (0.0, 3.0)}
        self.has_calculated = False
        self.viewports.clear()
        self.request_key = None


@dataclass
class RSMState:
    experimental: ExperimentalRSMState = field(default_factory=ExperimentalRSMState)
    calculated: CalculatedRSMState = field(default_factory=CalculatedRSMState)
    active_tab: int = 0

    def sync_document(self, document: Any) -> None:
        if document.kind in {'cif', 'cell_phase'}:
            self.calculated.bind(document.payload, document.uid)
        else:
            self.experimental.sources[document.uid] = document
            self.experimental.refresh_sources()

    def remove(self, uid: str) -> None:
        self.experimental.sources.pop(uid, None)
        self.experimental.refresh_sources()
        if uid == self.calculated.uid:
            self.calculated.bind(None)

    def invalidate_phase(self, uid: str, *, removed=False) -> None:
        if uid == self.experimental.phase_uid:
            self.experimental.overlay_enabled = False
            self.experimental.orientation = RSMOrientation()
            if removed:
                self.experimental.phase_uid = None
