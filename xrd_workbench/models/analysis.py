"""Accepted measurement results, independent of widgets and plotting adapters."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator

import numpy as np

from .background import BackgroundAnchors


def readonly_array(values: np.ndarray) -> np.ndarray:
    """Take ownership of writable input without copying immutable fit snapshots."""
    array = np.asarray(values, dtype=float)
    if array.flags.writeable:
        array = array.copy()
        array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SessionPeak:
    number: int  # Stable project ID, independent of sorted table rows.
    scan_uid: str
    axis_name: str
    source_center: float
    source_x: np.ndarray
    source_background: np.ndarray
    source_profile: np.ndarray
    source_height: float
    source_area: float
    source_fwhm: float
    kind: str = "primary"
    parent_number: int | None = None
    filled: bool = True
    hkl_assignments: tuple[tuple[str, int, int, int], ...] = ()
    source_sigma: float = 0.0
    source_gamma: float = 0.0

    def __post_init__(self) -> None:
        for name in ("source_x", "source_background", "source_profile"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class CompanionProposal:
    parent_number: int
    parent_center: float
    predicted_center: float
    fitted_center: float
    existing_number: int


AnalysisListener = Callable[[frozenset[str]], None]


class MeasurementAnalysis:
    """Single owner of peaks, background estimates and their relationships.

    Results use native measurement coordinates and are keyed by document ID
    and axis. Display transforms do not invalidate them. ProjectStore resets
    them when data changes or a measurement leaves Viewer. Collections are
    exposed as snapshots; all changes use this API.
    """

    def __init__(self) -> None:
        self._peaks: dict[int, SessionPeak] = {}
        self._backgrounds: dict[tuple[str, str], BackgroundAnchors] = {}
        self._next_peak_number = 1
        self._listeners: list[AnalysisListener] = []
        self._batch_depth = 0
        self._changed_uids: set[str] = set()

    @property
    def peaks(self) -> tuple[SessionPeak, ...]:
        return tuple(self._peaks.values())

    @property
    def backgrounds(self) -> tuple[tuple[str, BackgroundAnchors], ...]:
        return tuple((uid, value) for (uid, _axis), value in self._backgrounds.items())

    @property
    def next_peak_number(self) -> int:
        return self._next_peak_number

    def reserve_peak_numbers(self, next_number: int) -> None:
        if not isinstance(next_number, int) or isinstance(next_number, bool) or next_number < 1:
            raise ValueError('The next peak number must be a positive integer')
        self._next_peak_number = max(self._next_peak_number, next_number)

    def peaks_for(self, scan_uid: str, axis_name: str | None = None) -> tuple[SessionPeak, ...]:
        return tuple(peak for peak in self._peaks.values()
                     if peak.scan_uid == scan_uid
                     and (axis_name is None or peak.axis_name == axis_name))

    def peak(self, number: int, scan_uid: str | None = None) -> SessionPeak | None:
        peak = self._peaks.get(number)
        return peak if peak is not None and (scan_uid is None or peak.scan_uid == scan_uid) else None

    def subscribe(self, listener: AnalysisListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: AnalysisListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Coalesce notifications; callers prepare fallible results before committing."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if not self._batch_depth:
                self._notify()

    def _changed(self, *uids: str) -> None:
        self._changed_uids.update(uids)
        if not self._batch_depth:
            self._notify()

    def _notify(self) -> None:
        if not self._changed_uids:
            return
        uids = frozenset(self._changed_uids)
        self._changed_uids.clear()
        for listener in tuple(self._listeners):
            listener(uids)

    def put_peak(self, peak: SessionPeak) -> SessionPeak:
        """Import or replace one peak while retaining its document/axis identity."""
        if peak.number < 1:
            raise ValueError("Accepted peaks require positive stable IDs")
        previous = self._peaks.get(peak.number)
        if previous is not None and (previous.scan_uid, previous.axis_name) != (peak.scan_uid, peak.axis_name):
            raise ValueError("A peak ID cannot move to another measurement or axis")
        self._peaks[peak.number] = peak
        self._next_peak_number = max(self._next_peak_number, peak.number + 1)
        self._changed(peak.scan_uid)
        return peak

    def add_peak(self, scan_uid: str, axis_name: str, **values) -> SessionPeak:
        return self.put_peak(SessionPeak(self._next_peak_number, scan_uid, axis_name, **values))

    def update_peak(self, number: int, *, scan_uid: str | None = None, **values) -> SessionPeak | None:
        if {"number", "scan_uid", "axis_name"} & values.keys():
            raise ValueError("Peak identity cannot be changed")
        peak = self.peak(number, scan_uid)
        return self.put_peak(replace(peak, **values)) if peak is not None else None

    def accept_peak(self, scan_uid: str, axis_name: str, *, center_tolerance: float,
                    manual: bool = False, replace_number: int | None = None,
                    **values) -> SessionPeak | None:
        """Accept a preview, suppressing duplicates and retaining IDs on redraw."""
        if replace_number is not None:
            existing = self.peak(replace_number, scan_uid)
            if existing is None or existing.axis_name != axis_name:
                raise ValueError("The peak being replaced no longer belongs to this scan axis")
        else:
            existing = next((peak for peak in self.peaks_for(scan_uid, axis_name)
                             if abs(peak.source_center - values["source_center"]) < abs(center_tolerance)), None)
        if existing is not None:
            return self.update_peak(existing.number, **values) if manual else None
        return self.add_peak(scan_uid, axis_name, **values)

    def commit_fit(self, peaks: Iterable[SessionPeak], *, scan_uid: str,
                   axis_name: str, background: BackgroundAnchors | None) -> None:
        """Commit a completed joint fit once, preserving unrelated measurements."""
        peaks = tuple(peaks)
        for peak in peaks:
            previous = self.peak(peak.number, scan_uid)
            if (previous is None or previous.axis_name != axis_name
                    or (peak.scan_uid, peak.axis_name) != (scan_uid, axis_name)):
                raise ValueError("Fit results no longer match the accepted peaks")
        if background is not None and background.axis_name != axis_name:
            raise ValueError("Fit background does not match the measurement axis")
        with self.batch():
            for peak in peaks:
                self.put_peak(peak)
            if background is not None:
                self.set_background(scan_uid, background)

    def remove_peak(self, number: int, scan_uid: str | None = None) -> None:
        peak = self.peak(number, scan_uid)
        if peak is None:
            return
        with self.batch():
            del self._peaks[number]
            self._changed(peak.scan_uid)
            for companion in self.peaks_for(peak.scan_uid, peak.axis_name):
                if companion.parent_number == number:
                    self.update_peak(companion.number, kind="primary", parent_number=None)

    def clear_peaks(self, scan_uid: str | None = None, axis_name: str | None = None) -> None:
        with self.batch():
            for peak in self.peaks:
                if ((scan_uid is None or peak.scan_uid == scan_uid)
                        and (axis_name is None or peak.axis_name == axis_name)):
                    del self._peaks[peak.number]
                    self._changed(peak.scan_uid)

    def classify_companions(self, scan_uid: str, kind: str, pairs: dict[int, int]) -> None:
        if kind not in {"ka2", "kb"}:
            raise ValueError("Unknown companion type")
        if len(set(pairs.values())) != len(pairs) or pairs.keys() & set(pairs.values()):
            raise ValueError("A peak can participate in only one new companion pair")
        replacements = []
        for number, parent_number in pairs.items():
            peak, parent = self.peak(number, scan_uid), self.peak(parent_number, scan_uid)
            if (peak is None or parent is None or number == parent_number
                    or peak.axis_name != parent.axis_name
                    or peak.kind != "primary" or parent.kind != "primary"
                    or any(p.kind == kind and p.parent_number == parent_number
                           for p in self.peaks_for(scan_uid, parent.axis_name))):
                raise ValueError("Companions require a primary parent on the same measurement axis")
            replacements.append(replace(peak, kind=kind, parent_number=parent_number))
        with self.batch():
            for peak in replacements:
                self.put_peak(peak)

    def assign_hkl(self, scan_uid: str, number: int, assignment: tuple[str, int, int, int]) -> None:
        peak = self.peak(number, scan_uid)
        if peak is not None and assignment not in peak.hkl_assignments:
            self.update_peak(number, hkl_assignments=(*peak.hkl_assignments, assignment))

    def clear_hkl(self, scan_uid: str, number: int) -> None:
        self.update_peak(number, scan_uid=scan_uid, hkl_assignments=())

    def invalidate_phase(self, phase_uid: str) -> None:
        with self.batch():
            for peak in self.peaks:
                retained = tuple(entry for entry in peak.hkl_assignments if entry[0] != phase_uid)
                if retained != peak.hkl_assignments:
                    self.update_peak(peak.number, hkl_assignments=retained)

    def background_for(self, scan_uid: str, axis_name: str) -> BackgroundAnchors | None:
        return self._backgrounds.get((scan_uid, axis_name))

    def set_background(self, scan_uid: str, background: BackgroundAnchors) -> None:
        self._backgrounds[(scan_uid, background.axis_name)] = background
        self._changed(scan_uid)

    def clear_background(self, scan_uid: str, axis_name: str) -> None:
        if self._backgrounds.pop((scan_uid, axis_name), None) is not None:
            self._changed(scan_uid)

    def exclude_background_region(self, scan_uid: str, axis_name: str, low: float, high: float) -> None:
        background = self.background_for(scan_uid, axis_name)
        if background is not None:
            self.set_background(scan_uid, background.without_region(low, high))

    def invalidate_measurement(self, scan_uid: str) -> None:
        with self.batch():
            self.clear_peaks(scan_uid)
            for uid, axis_name in tuple(self._backgrounds):
                if uid == scan_uid:
                    self.clear_background(uid, axis_name)

    def invalidate_document(self, uid: str) -> None:
        with self.batch():
            self.invalidate_measurement(uid)
            self.invalidate_phase(uid)
