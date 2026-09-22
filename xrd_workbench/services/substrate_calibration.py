"""Angular-axis calibration from successive orders of one substrate family."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..models.data_errors import XRDDataError

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - SciPy is already used by peak fitting
    least_squares = None


@dataclass(frozen=True)
class SubstrateCalibration:
    """One affine conversion from measured to corrected 2-theta values."""

    mode: str
    x_scale: float
    x_shift: float
    expected: np.ndarray
    residuals: np.ndarray
    rms: float

    def corrected(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        return source * self.x_scale + self.x_shift


def _validated_peaks(
    measured: Sequence[float],
    orders: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(measured, dtype=float)
    n = np.asarray(orders, dtype=int)
    if x.ndim != 1 or n.ndim != 1 or x.size != n.size or x.size < 2:
        raise XRDDataError("substrate_calibration_peaks")
    if not np.all(np.isfinite(x)) or np.any((x <= 0.0) | (x >= 180.0)):
        raise XRDDataError("substrate_calibration_angles")
    if np.any(n <= 0) or np.unique(n).size != n.size:
        raise XRDDataError("substrate_calibration_orders")
    return x, n


def fit_substrate_calibration(
    measured: Sequence[float],
    orders: Sequence[int],
    *,
    true_position: float | None = None,
    reference_order: int | None = None,
) -> SubstrateCalibration:
    """Fit a common shift or an affine scale from one harmonic peak family.

    Without an external reference, all selected orders are assumed to share a
    single angular offset.  If one true peak position is supplied, Bragg's law
    generates the expected positions of the remaining orders and an affine
    measured-to-true conversion is fitted to all selected peaks.
    """

    x, n = _validated_peaks(measured, orders)
    if true_position is None:
        if least_squares is None:
            raise XRDDataError("substrate_calibration_scipy")
        maximum_order = int(np.max(n))
        initial_sine = float(np.median(np.sin(np.deg2rad(x / 2.0)) / n))
        upper_sine = (1.0 - 1.0e-12) / maximum_order
        initial_sine = max(1.0e-12, min(upper_sine * 0.999, initial_sine))

        def residual(parameters: np.ndarray) -> np.ndarray:
            shift, fundamental_sine = parameters
            corrected = x + shift
            return np.sin(np.deg2rad(corrected / 2.0)) - n * fundamental_sine

        result = least_squares(
            residual,
            x0=np.asarray([0.0, initial_sine]),
            bounds=(np.asarray([-20.0, 1.0e-12]), np.asarray([20.0, upper_sine])),
            max_nfev=20_000,
            ftol=1.0e-14,
            xtol=1.0e-14,
            gtol=1.0e-14,
        )
        if not result.success:
            raise XRDDataError("substrate_calibration_fit")
        shift = float(result.x[0])
        fundamental_sine = float(result.x[1])
        expected = np.rad2deg(2.0 * np.arcsin(n * fundamental_sine))
        scale = 1.0
        mode = "common_shift"
    else:
        target = float(true_position)
        order = int(reference_order or 0)
        if not math.isfinite(target) or not 0.0 < target < 180.0:
            raise XRDDataError("substrate_calibration_reference")
        if order <= 0:
            raise XRDDataError("substrate_calibration_reference_order")
        fundamental_sine = math.sin(math.radians(target / 2.0)) / order
        arguments = n * fundamental_sine
        if np.any(arguments <= 0.0) or np.any(arguments >= 1.0):
            raise XRDDataError("substrate_calibration_harmonic")
        expected = np.rad2deg(2.0 * np.arcsin(arguments))
        design = np.column_stack((x, np.ones(x.shape)))
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            design,
            expected,
            rcond=None,
        )
        if rank < 2:
            raise XRDDataError("substrate_calibration_fit")
        scale, shift = map(float, coefficients)
        if not math.isfinite(scale) or not math.isfinite(shift) or scale <= 0.0:
            raise XRDDataError("substrate_calibration_fit")
        mode = "affine"

    predicted = x * scale + shift
    residuals = predicted - expected
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return SubstrateCalibration(
        mode=mode,
        x_scale=float(scale),
        x_shift=float(shift),
        expected=np.asarray(expected, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        rms=rms,
    )


__all__ = ["SubstrateCalibration", "fit_substrate_calibration"]
