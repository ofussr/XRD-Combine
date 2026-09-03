"""Toolkit-independent correction request model."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .data_errors import XRDDataError


RESULT_MODES = ("add", "replace")


@dataclass(frozen=True)
class CorrectionRequest:
    """One non-destructive correction to a one-dimensional measurement."""

    x_shift: float = 0.0
    y_shift: float = 0.0
    y_factor: float = 1.0
    shift_omega_half: bool = True

    def __post_init__(self) -> None:
        values = (self.x_shift, self.y_shift, self.y_factor)
        if not all(math.isfinite(float(value)) for value in values):
            raise XRDDataError("correction_values")
        if self.y_factor <= 0:
            raise XRDDataError("correction_y_factor")


def validate_result_mode(mode: str) -> str:
    if mode not in RESULT_MODES:
        raise XRDDataError("correction_result_mode", mode=mode)
    return mode


__all__ = ["CorrectionRequest", "RESULT_MODES", "validate_result_mode"]
