"""Editable, session-only background estimates in measured scan coordinates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter


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


def estimate_background(
    coordinates: np.ndarray,
    intensities: np.ndarray,
    *,
    axis_name: str,
    spacing: float,
    excluded: tuple[tuple[float, float], ...] = (),
) -> BackgroundAnchors:
    """Estimate a background that retains broad bands but excludes narrow peaks.

    The median of each X interval removes raw counting noise. A running local
    percentile ignores features narrower than its physical window, while wide
    scattering bands survive. Gentle symmetric smoothing then removes window
    steps without fitting another peak-rejecting baseline. Excluded anchors
    continue to bridge their interval by straight interpolation.
    """
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(intensities, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Background requires at least three points and positive spacing")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    if span <= 0:
        raise ValueError("Background axis has zero width")
    bins = max(3, min(2500, int(np.ceil(span / spacing))))
    edges = np.linspace(x[0], x[-1], bins + 1)
    groups = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    starts = np.searchsorted(groups, np.arange(bins), side="left")
    ends = np.searchsorted(groups, np.arange(bins), side="right")
    support_x, support_y = [], []
    for first, last in zip(starts, ends):
        if first < last:
            support_x.append(float(np.median(x[first:last])))
            support_y.append(float(np.median(y[first:last])))
    if len(support_x) < 3:
        # Sparse or irregular axis: retain one anchor per distinct coordinate.
        distinct = np.r_[True, np.diff(x) > 0]
        support_x = list(x[distinct])
        support_y = list(y[distinct])
    raw = np.asarray(support_y, dtype=float)
    n = raw.size
    if n < 3:
        raise ValueError("Background requires three distinct axis positions")
    median_step = float(np.median(np.diff(support_x)))
    # At the default 0.25-degree support spacing the window spans about 5
    # degrees: wide diffuse bands survive; isolated 1–2 degree peaks do not.
    # Cap the window for short or non-angular scans. Spacing also controls the
    # physical distinction between narrow and broad features.
    window = min(span / 6.0, max(20.0 * median_step, span / 16.0))
    window_bins = max(3, int(round(window / median_step)) | 1)
    window_bins = min(window_bins, n if n % 2 else n - 1)
    envelope = percentile_filter(raw, percentile=35, size=window_bins,
                                 mode="reflect")
    smooth = gaussian_filter1d(envelope, sigma=max(0.7, window_bins / 6.0),
                               mode="nearest")
    anchor_x = np.r_[x[0], support_x, x[-1]]
    anchor_y = np.r_[smooth[0], smooth, smooth[-1]]
    # Binning can place an anchor on an endpoint; avoid duplicate X values.
    unique = np.r_[True, np.diff(anchor_x) > 0]
    return BackgroundAnchors(axis_name, float(spacing), anchor_x[unique],
                             anchor_y[unique], excluded)
