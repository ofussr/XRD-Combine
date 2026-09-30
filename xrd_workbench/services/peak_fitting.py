"""Peak fitting calculations shared by graphical interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..models.data_errors import XRDDataError

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import brentq, curve_fit, least_squares
    from scipy.signal import find_peaks, peak_widths
    from scipy.special import voigt_profile
except ImportError:  # pragma: no cover - exercised only without optional SciPy
    curve_fit = None
    least_squares = None


def fit_gaussian_peak(
    coordinates: np.ndarray,
    intensities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit one positive Gaussian peak with a linear background."""

    if curve_fit is None:
        raise XRDDataError("peak_fit_scipy")
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(intensities, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 7:
        raise XRDDataError("peak_fit_points", count=int(x.size))
    if float(np.max(y)) <= float(np.min(y)):
        raise XRDDataError("peak_fit_flat")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_mid = 0.5 * (x_min + x_max)

    def model(values, c0, c1, amplitude, center, sigma):
        return (
            c0
            + c1 * (values - x_mid)
            + amplitude * np.exp(-0.5 * ((values - center) / sigma) ** 2)
        )

    spread = max(x_max - x_min, 1e-8)
    initial = [
        float(np.min(y)),
        0.0,
        float(np.max(y) - np.min(y)),
        float(x[np.argmax(y)]),
        max(spread / 6.0, 1e-6),
    ]
    parameters, _covariance = curve_fit(
        model,
        x,
        y,
        p0=initial,
        bounds=(
            [-np.inf, -np.inf, 0.0, x_min, 1e-8],
            [np.inf, np.inf, np.inf, x_max, spread],
        ),
        maxfev=20000,
    )
    center = float(parameters[3])
    intensity = float(model(np.asarray([center]), *parameters)[0])
    fit_x = np.linspace(x_min, x_max, 500)
    return fit_x, model(fit_x, *parameters), center, intensity


@dataclass(frozen=True)
class VoigtPeak:
    center: float
    height: float
    area: float
    sigma: float
    gamma: float
    fwhm: float
    profile: np.ndarray


@dataclass(frozen=True)
class VoigtRegionFit:
    x: np.ndarray
    fitted: np.ndarray
    background: np.ndarray
    peaks: tuple[VoigtPeak, ...]
    background_delta: np.ndarray | None = None


MAX_MANUAL_PEAKS = 6


def refit_voigt_peaks(coordinates: np.ndarray, intensities: np.ndarray,
                      background: np.ndarray,
                      seeds: list[tuple[float, float, float, float]], *,
                      refine_background: bool = False) -> VoigtRegionFit:
    """Refine accepted components and, optionally, a smooth local background.

    Seeds contain (center, height, sigma, gamma) in the scan's native units.
    No peak detection or model selection takes place here. Peak order is kept
    so callers can retain their table IDs and assignments.
    """
    if least_squares is None:
        raise XRDDataError("peak_fit_scipy")
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(intensities, dtype=float)
    base = np.asarray(background, dtype=float)
    if x.shape != y.shape or x.shape != base.shape or x.ndim != 1:
        raise ValueError("Scan and background must have matching one-dimensional axes")
    if len(seeds) == 0 or x.size < 12 or not np.all(np.isfinite(x + y + base)):
        raise XRDDataError("peak_fit_points")
    if np.any(np.diff(x) <= 0):
        raise XRDDataError("peak_fit_axis")
    if refine_background:
        preliminary = refit_voigt_peaks(x, y, base, seeds)
        step = float(np.median(np.diff(x)))
        def shoulder_correction(fit: VoigtRegionFit, current_base: np.ndarray,
                                *, downward_only: bool = False) -> np.ndarray:
            intervals = []
            for peak in sorted(fit.peaks, key=lambda p: p.center):
                radius = max(1.2, 10.0 * peak.fwhm)
                left = max(float(x[0]), peak.center - radius)
                right = min(float(x[-1]), peak.center + radius)
                if intervals and left <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], right))
                else:
                    intervals.append((left, right))
            excluded = np.zeros(x.size, dtype=bool)
            for peak in fit.peaks:
                excluded |= np.abs(x - peak.center) <= max(1.5 * peak.fwhm, 8.0 * step)
            correction = np.zeros_like(x)
            residual_y = y - fit.fitted
            for left, right in intervals:
                window = (x >= left) & (x <= right)
                eligible = window & ~excluded
                if np.count_nonzero(eligible) < 12:
                    continue
                bin_width = max(.20, (right - left) / 35.0, 8.0 * step)
                edges = np.arange(left, right + bin_width, bin_width)
                support_x, support_y = [], []
                for a, b in zip(edges[:-1], edges[1:]):
                    sample = eligible & (x >= a) & (x < b)
                    if np.count_nonzero(sample) >= 4:
                        support_x.append(float(np.median(x[sample])))
                        support_y.append(float(np.median(residual_y[sample])))
                if len(support_x) < 2:
                    continue
                raw = np.interp(x[window], support_x, support_y)
                smoothed = gaussian_filter1d(raw, sigma=max(1.0, .22 / step))
                taper = np.sin(np.pi * np.clip(
                    (x[window] - left) / (right - left), 0, 1)) ** 2
                proposed = smoothed * taper
                if downward_only:
                    proposed = np.minimum(proposed, 0.0)
                correction[window] += np.clip(
                    proposed, -.95 * current_base[window],
                    0.0 if downward_only else .35 * current_base[window])
            return correction

        correction = shoulder_correction(preliminary, base)
        if not np.any(np.abs(correction) > 1e-9):
            return preliminary
        provisional_seeds = [(p.center, p.height, p.sigma, p.gamma)
                             for p in preliminary.peaks]
        refined = refit_voigt_peaks(x, y, base + correction, provisional_seeds)
        # A second, downward-only pass removes excess background beneath the
        # long tails that remain after the peak widths are refined.
        remaining = shoulder_correction(refined, base + correction, downward_only=True)
        remaining[np.abs(correction) < 1e-9] = 0.0
        if np.any(remaining < -1e-9):
            correction += remaining
            seeds_after = [(p.center, p.height, p.sigma, p.gamma)
                           for p in refined.peaks]
            refined = refit_voigt_peaks(x, y, base + correction, seeds_after)
        return VoigtRegionFit(refined.x, refined.fitted, refined.background,
                              refined.peaks, correction)
    step = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    minimum = max(step / 5.0, span * 1e-8)
    order = sorted(range(len(seeds)), key=lambda index: seeds[index][0])
    start, lower, upper = [], [], []
    for position, index in enumerate(order):
        centre, height, sigma, gamma = map(float, seeds[index])
        if not np.all(np.isfinite([centre, height, sigma, gamma])) or not x[0] < centre < x[-1]:
            raise XRDDataError("peak_fit_axis")
        sigma = max(sigma, minimum * 1.01)
        gamma = max(gamma, minimum * 1.01)
        width = max(sigma + gamma, step)
        left = x[0] if position == 0 else (centre + seeds[order[position - 1]][0]) / 2.0
        right = x[-1] if position == len(order) - 1 else (centre + seeds[order[position + 1]][0]) / 2.0
        centre_low = max(float(left), centre - max(1.5 * width, 3.0 * step))
        centre_high = min(float(right), centre + max(1.5 * width, 3.0 * step))
        width_high = max(5.0 * width, 4.0 * step)
        start.extend([max(height, 1e-8), centre, sigma, gamma])
        lower.extend([0.0, centre_low, minimum, minimum])
        upper.extend([np.inf, centre_high, width_high, width_high])

    # The stored profiles extend across the full scan, but distant points
    # contain no useful information about a narrow peak. Restrict the
    # optimizer to the peak neighbourhoods to keep long scans responsive.
    fit_mask = np.zeros(x.size, dtype=bool)
    for centre, _height, sigma, gamma in seeds:
        radius = max(30.0 * (sigma + gamma), 30.0 * step)
        fit_mask |= np.abs(x - centre) <= radius
    fit_x, fit_y, fit_base = x[fit_mask], y[fit_mask], base[fit_mask]

    def component(values, height, centre, sigma, gamma):
        return height * voigt_profile(values - centre, sigma, gamma) / voigt_profile(
            0.0, sigma, gamma)

    # An accepted subset must not stretch a selected component across another
    # obvious, as-yet-unaccepted reflection. Keep each accepted core, but
    # exclude strong unexplained positive maxima farther from every core.
    seed_model = fit_base.copy()
    core = np.zeros(fit_x.size, dtype=bool)
    for centre, height, sigma, gamma in seeds:
        seed_model += component(fit_x, height, centre, sigma, gamma)
        core |= np.abs(fit_x - centre) <= max(4.0 * (sigma + gamma), 6.0 * step)
    unexplained = (~core & (fit_y - seed_model > .15 * max(seed[1] for seed in seeds)))
    fit_x, fit_y, fit_base = fit_x[~unexplained], fit_y[~unexplained], fit_base[~unexplained]

    floor = max(1.0, float(np.percentile(fit_y, 10)))
    weights = 1.0 / np.maximum(fit_y, floor) ** .7

    def residual(parameters):
        total = fit_base.copy()
        for offset in range(0, len(parameters), 4):
            total += component(fit_x, *parameters[offset:offset + 4])
        return (total - fit_y) * weights

    initial = np.asarray(start)
    initial_error = float(np.dot(residual(initial), residual(initial)))
    result = least_squares(residual, initial, bounds=(lower, upper),
                           x_scale="jac", max_nfev=700)
    fitted_error = float(np.dot(result.fun, result.fun))
    if not np.all(np.isfinite(result.x)) or not np.isfinite(fitted_error):
        raise XRDDataError("peak_fit_flat")
    parameters = result.x if fitted_error <= initial_error else initial
    by_original = [None] * len(seeds)
    for position, index in enumerate(order):
        height, centre, sigma, gamma = map(float, parameters[4 * position:4 * position + 4])
        profile = component(x, height, centre, sigma, gamma)
        half_maximum = voigt_profile(0.0, sigma, gamma) / 2.0
        half_span = brentq(
            lambda delta: voigt_profile(delta, sigma, gamma) - half_maximum,
            0.0, 10.0 * (sigma + gamma),
        )
        by_original[index] = VoigtPeak(
            centre, height, height / voigt_profile(0.0, sigma, gamma),
            sigma, gamma, 2.0 * half_span, profile,
        )
    peaks = tuple(by_original)
    return VoigtRegionFit(x, base + sum((peak.profile for peak in peaks),
                                        np.zeros_like(x)), base, peaks)


def fit_voigt_region(coordinates: np.ndarray, intensities: np.ndarray,
                     background: np.ndarray | None = None, *,
                     search_bounds: tuple[float, float] | None = None) -> VoigtRegionFit:
    """Fit up to six resolved maxima with Voigt profiles.

    Search bounds limit which peak centres may be proposed; all supplied
    coordinates, including the margins outside those bounds, enter the fit.
    The model count is selected by comparing residuals; ambiguous shoulders
    remain one fitted peak. The caller always confirms the proposed components.
    An explicitly supplied background is fixed across all peak regions; only
    without one is a local linear background fitted alongside the profiles.
    """
    if least_squares is None:
        raise XRDDataError("peak_fit_scipy")
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(intensities, dtype=float)
    fixed_background = background is not None
    if fixed_background:
        base = np.asarray(background, dtype=float)
        if x.shape != y.shape or base.shape != x.shape:
            raise ValueError("Background must match scan coordinates")
    else:
        base = np.zeros_like(y)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(base)
    x, y, base = x[valid], y[valid], base[valid]
    if x.size < 12:
        raise XRDDataError("peak_find_points", count=int(x.size))
    order = np.argsort(x)
    x, y, base = x[order], y[order], base[order]
    if np.any(np.diff(x) <= 0.0):
        raise XRDDataError("peak_fit_axis")
    if search_bounds is not None:
        search_low, search_high = sorted(map(float, search_bounds))
        if (not np.isfinite(search_low) or not np.isfinite(search_high)
                or search_low >= search_high):
            raise ValueError("Peak search bounds must be finite and increasing")
    spread = float(x[-1] - x[0])
    residual_y = y - base
    signal = float(np.max(residual_y) - np.min(residual_y))
    if not spread > 0 or not signal > 0:
        raise XRDDataError("peak_fit_flat")
    step = float(np.median(np.diff(x)))
    midpoint = (x[0] + x[-1]) / 2.0
    edge_count = max(2, x.size // 8)
    left = float(np.median(residual_y[:edge_count]))
    right = float(np.median(residual_y[-edge_count:]))
    baseline = (np.zeros_like(x) if fixed_background else
                left + (right - left) * (x - x[0]) / spread)
    smoothed = gaussian_filter1d(residual_y - baseline, 1.1)
    differences = np.diff(residual_y - baseline)
    noise = float(
        1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)
    )
    # A fixed fraction of the strongest peak hides real weak peaks in a region
    # containing both an intense substrate reflection and a film reflection.
    # Require statistical contrast and a positive height over the shared
    # background instead. A tiny scale-dependent floor handles noiseless data.
    threshold = max(4.0 * noise, 1e-6 * signal)
    maxima, _properties = find_peaks(
        smoothed, prominence=threshold,
        height=threshold, distance=3,
    )
    if search_bounds is not None:
        maxima = maxima[(x[maxima] >= search_low) & (x[maxima] <= search_high)]
    if maxima.size == 0:
        raise XRDDataError("peak_fit_flat")
    if maxima.size > MAX_MANUAL_PEAKS:
        raise XRDDataError("peak_fit_many")
    maxima = maxima[np.argsort(x[maxima])]
    half_widths = dict(zip(
        map(int, maxima), peak_widths(smoothed, maxima, rel_height=0.5)[0] * step
    ))
    min_width = max(step / 5.0, spread * 1e-7)
    max_width = spread / 2.0

    def component(values, height, center, sigma, gamma):
        return height * voigt_profile(values - center, sigma, gamma) / voigt_profile(
            0.0, sigma, gamma
        )

    def model(parameters, values):
        total = (np.zeros_like(values) if fixed_background else
                 parameters[0] + parameters[1] * (values - midpoint))
        for offset in range(0 if fixed_background else 2, len(parameters), 4):
            total = total + component(values, *parameters[offset : offset + 4])
        return total

    def fitted(indices):
        guess = [] if fixed_background else [left, (right - left) / spread]
        lower = [] if fixed_background else [-np.inf, -np.inf]
        upper = [] if fixed_background else [np.inf, np.inf]
        for index in indices:
            seed_width = max(step, half_widths[index] / 3.0)
            guess += [max(float(residual_y[index] - baseline[index]), signal * 0.02), float(x[index]),
                      min(seed_width, max_width * 0.8),
                      min(seed_width, max_width * 0.8)]
            lower += [0.0, float(x[0]) if search_bounds is None else search_low,
                      min_width, min_width]
            upper += [np.inf, float(x[-1]) if search_bounds is None else search_high,
                      max_width, max_width]
        result = least_squares(
            lambda parameters: model(parameters, x) - residual_y,
            guess,
            bounds=(lower, upper),
            x_scale="jac", max_nfev=120 if len(indices) <= 2 else 100,
        )
        # A wide interval with several peaks may reach the iteration limit
        # while fitting a deliberately incomplete one-peak trial. The next
        # model is still compared using the actual residual of that trial.
        if not np.all(np.isfinite(result.x)):
            raise XRDDataError("peak_fit_flat")
        return result.x, float(np.dot(result.fun, result.fun))

    strongest_first = sorted(maxima, key=lambda index: -smoothed[index])
    peak_offset = 0 if fixed_background else 2
    if len(strongest_first) > 2:
        # For a multi-peak interval, fitting every smaller subset in turn is
        # prohibitively slow and poorly conditioned. Fit all well-separated
        # detected maxima together and let the user inspect the preview.
        parameters, _rss = fitted([int(index) for index in strongest_first])
        if (min(parameters[peak_offset::4]) < max(3.0 * noise, 1e-6 * signal)
                or min(np.diff(np.sort(parameters[peak_offset + 1::4]))) < 2.0 * step):
            raise XRDDataError("peak_fit_flat")
    else:
        parameters, single_rss = fitted([int(strongest_first[0])])
        if len(strongest_first) == 2:
            double, double_rss = fitted([int(index) for index in strongest_first])
            n = x.size
            single_bic = n * np.log(max(single_rss / n, 1e-24)) + len(parameters) * np.log(n)
            double_bic = n * np.log(max(double_rss / n, 1e-24)) + len(double) * np.log(n)
            if (single_bic - double_bic > 10.0
                    and min(double[peak_offset::4]) >= max(3.0 * noise, 1e-6 * signal)
                    and abs(double[peak_offset + 1] - double[peak_offset + 5]) >= 2.0 * step):
                parameters = double

    fit_x = np.linspace(float(x[0]), float(x[-1]), 400)
    baseline_fit = (np.interp(fit_x, x, base) if fixed_background else
                    parameters[0] + parameters[1] * (fit_x - midpoint))
    peaks = []
    for offset in range(peak_offset, len(parameters), 4):
        height, center, sigma, gamma = map(float, parameters[offset : offset + 4])
        profile = component(fit_x, height, center, sigma, gamma)
        half_maximum = voigt_profile(0.0, sigma, gamma) / 2.0
        half_width = brentq(
            lambda displacement: voigt_profile(displacement, sigma, gamma) - half_maximum,
            0.0,
            10.0 * (sigma + gamma),
        )
        peaks.append(
            VoigtPeak(center, height, height / voigt_profile(0.0, sigma, gamma),
                      sigma, gamma, 2.0 * half_width, profile)
        )
    peaks.sort(key=lambda peak: peak.center)
    return VoigtRegionFit(fit_x, baseline_fit + sum((peak.profile for peak in peaks),
                                                  np.zeros_like(fit_x)),
                          baseline_fit, tuple(peaks))


@lru_cache(maxsize=1)
def _unit_voigt_half_width() -> float:
    return brentq(
        lambda distance: voigt_profile(distance, 1.0, 1.0)
        / voigt_profile(0.0, 1.0, 1.0) - 0.5,
        0.0, 10.0,
    )


def drawn_voigt_profile(values: np.ndarray, *, center: float,
                        half_width: float, height: float) -> tuple[np.ndarray, float, float]:
    """Return the peak drawn by a top, baseline and half-width gesture."""
    if least_squares is None:
        raise XRDDataError("peak_fit_scipy")
    if (not np.all(np.isfinite((center, half_width, height)))
            or half_width <= 0.0 or height <= 0.0):
        raise XRDDataError("peak_fit_axis")
    sigma = gamma = float(half_width / _unit_voigt_half_width())
    profile = height * voigt_profile(np.asarray(values) - center, sigma, gamma) / voigt_profile(
        0.0, sigma, gamma)
    return profile, sigma, gamma


def draw_voigt_peak(coordinates: np.ndarray, background: np.ndarray, *,
                    center: float, half_width: float, height: float) -> VoigtRegionFit:
    """Insert precisely the drawn Voigt shape without reading measured intensities."""
    x = np.asarray(coordinates, dtype=float)
    base = np.asarray(background, dtype=float)
    if (x.ndim != 1 or x.shape != base.shape or x.size < 12
            or not np.all(np.isfinite(x + base)) or np.any(np.diff(x) <= 0)
            or not x[0] < center < x[-1]):
        raise XRDDataError("peak_fit_axis")
    profile, sigma, gamma = drawn_voigt_profile(
        x, center=center, half_width=half_width, height=height)
    peak = VoigtPeak(center, height, height / voigt_profile(0.0, sigma, gamma),
                     sigma, gamma, 2.0 * half_width, profile)
    baseline_fit = base.copy()
    return VoigtRegionFit(x, baseline_fit + profile, baseline_fit, (peak,))


def companion_two_theta(primary_two_theta: float, primary_wavelength: float,
                        companion_wavelength: float) -> float:
    """Predict a second spectral line of the same lattice reflection."""

    argument = (companion_wavelength / primary_wavelength) * np.sin(
        np.deg2rad(primary_two_theta / 2.0)
    )
    if not 0.0 < argument < 1.0:
        return float("nan")
    return float(np.rad2deg(2.0 * np.arcsin(argument)))



__all__ = ["fit_gaussian_peak", "fit_voigt_region", "draw_voigt_peak",
           "drawn_voigt_profile",
           "refit_voigt_peaks",
           "VoigtPeak", "VoigtRegionFit",
           "MAX_MANUAL_PEAKS", "companion_two_theta"]
