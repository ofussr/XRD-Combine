"""Editable, session-only background estimates in measured scan coordinates."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter


from ..models.background import BackgroundAnchors


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
