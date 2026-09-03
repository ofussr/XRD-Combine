"""Peak fitting calculations shared by graphical interfaces."""

from __future__ import annotations

import numpy as np

from ..models.data_errors import XRDDataError

try:
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover - exercised only without optional SciPy
    curve_fit = None


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


__all__ = ["fit_gaussian_peak"]
