"""Background estimates in native measurement coordinates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class BackgroundAnchors:
    axis_name: str
    spacing: float
    source_x: np.ndarray
    source_y: np.ndarray
    excluded: tuple[tuple[float, float], ...] = ()
    # center, correction radius, offset to the user's level, excluded half-width
    local_levels: tuple[tuple[float, float, float, float], ...] = ()
    fit_correction_x: np.ndarray | None = None
    fit_correction_y: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("source_x", "source_y", "fit_correction_x", "fit_correction_y"):
            values = getattr(self, name)
            if values is not None:
                array = np.asarray(values, dtype=float)
                if array.flags.writeable:
                    array = array.copy()
                    array.setflags(write=False)
                object.__setattr__(self, name, array)

    def values(self, coordinates: np.ndarray) -> np.ndarray:
        x, y = self._unmodified_anchors()
        coordinates = np.asarray(coordinates, dtype=float)
        result = self._apply_local_levels(coordinates, np.interp(coordinates, x, y))
        if self.fit_correction_x is not None and self.fit_correction_y is not None:
            result += np.interp(coordinates, self.fit_correction_x,
                                self.fit_correction_y, left=0.0, right=0.0)
        return result

    def with_fit_correction(self, coordinates: np.ndarray,
                            delta: np.ndarray) -> BackgroundAnchors:
        """Accumulate one smooth local fit correction in source coordinates."""
        x = np.asarray(coordinates, dtype=float)
        change = np.asarray(delta, dtype=float)
        if (x.ndim != 1 or x.size < 2 or x.shape != change.shape
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(change))
                or np.any(np.diff(x) <= 0)):
            raise ValueError("Fit correction must have a finite, ordered scan axis")
        previous = (np.interp(x, self.fit_correction_x, self.fit_correction_y,
                              left=0.0, right=0.0)
                    if self.fit_correction_x is not None else np.zeros_like(x))
        return replace(self, fit_correction_x=x.copy(),
                       fit_correction_y=previous + change)

    def _apply_local_levels(self, coordinates: np.ndarray,
                            values: np.ndarray) -> np.ndarray:
        result = np.array(values, copy=True)
        base_x, base_y = self._unmodified_anchors()
        for center, radius, delta, excluded_half_width in self.local_levels:
            distance = np.abs(coordinates - center) / radius
            taper = (1.0 + np.cos(np.pi * np.minimum(distance, 1.0))) / 2.0
            if excluded_half_width:
                keep = ((base_x < center - excluded_half_width)
                        | (base_x > center + excluded_half_width))
                keep[0] = keep[-1] = True
                bridged = np.interp(coordinates, base_x[keep], base_y[keep])
                original = np.interp(coordinates, base_x, base_y)
                result = result + (bridged - original + delta) * taper
            else:
                result = result + delta * taper
        return result

    def active_anchors(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._unmodified_anchors()
        return x, self._apply_local_levels(x, y)

    def _unmodified_anchors(self) -> tuple[np.ndarray, np.ndarray]:
        keep = np.ones(self.source_x.size, dtype=bool)
        for low, high in self.excluded:
            keep &= (self.source_x < low) | (self.source_x > high)
        # Retain the domain endpoints so interpolation covers the whole scan.
        keep[0] = keep[-1] = True
        return self.source_x[keep], self.source_y[keep]

    def without_region(self, low: float, high: float) -> BackgroundAnchors:
        low, high = sorted((float(low), float(high)))
        return replace(self, excluded=(*self.excluded, (low, high)))

    def with_local_level(self, center: float, level: float, radius: float,
                         *, excluded_half_width: float = 0.0) -> BackgroundAnchors:
        """Bridge peak-contaminated anchors and smoothly match the drawn level.

        The correction is limited to the drawn peak and its surroundings;
        outside that interval the original common background is unchanged.
        """
        if (not all(np.isfinite((center, level, radius, excluded_half_width)))
                or radius <= 0 or not 0 <= excluded_half_width < radius
                or center < self.source_x[0] or center > self.source_x[-1]):
            raise ValueError("Choose a finite background level inside the scan")
        current = float(self.values(np.asarray([center]))[0])
        base_x, base_y = self._unmodified_anchors()
        original = float(np.interp(center, base_x, base_y))
        bridge = original
        if excluded_half_width:
            keep = ((base_x < center - excluded_half_width)
                    | (base_x > center + excluded_half_width))
            keep[0] = keep[-1] = True
            bridge = float(np.interp(center, base_x[keep], base_y[keep]))
        return replace(self, local_levels=(*self.local_levels,
                                          (float(center), float(radius),
                                           float(level) - current - (bridge - original),
                                           float(excluded_half_width))))

