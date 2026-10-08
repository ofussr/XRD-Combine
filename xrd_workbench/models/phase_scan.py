"""Geometry and display coordinates for oriented phase reflections."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .diffraction import ReflectionRow
from .scan import Scan1D


REQUIRED_ANGLES = {
    'phi': ('two_theta', 'omega', 'chi'),
    'chi': ('two_theta', 'omega', 'phi'),
    'omega': ('two_theta', 'chi', 'phi'),
    'coupled': ('chi', 'phi', 'omega_offset'),
    'detector': ('omega', 'chi', 'phi'),
}


def angular_axis(name: str) -> str | None:
    key = str(name).strip().lower().replace(' ', '').replace('-', '').replace('_', '')
    return {'2theta': 'two_theta', 'twotheta': 'two_theta', '2θ': 'two_theta',
            'phi': 'phi', 'φ': 'phi', 'chi': 'chi', 'χ': 'chi',
            'omega': 'omega', 'ω': 'omega', 'theta': 'omega', 'θ': 'omega'}.get(key)


@dataclass(frozen=True)
class PhaseOrientation:
    surface: tuple[int, int, int] = (0, 0, 1)
    inplane: tuple[int, int, int] = (1, 0, 0)
    phi_zero: float = 0.0

    def validate(self) -> None:
        for triple in (self.surface, self.inplane):
            if (len(triple) != 3 or not any(triple)
                    or any(not isinstance(v, int) or isinstance(v, bool) for v in triple)):
                raise ValueError('Orientation requires nonzero integer h k l triples.')
        h, k, l = self.surface
        a, b, c = self.inplane
        if (k*c-l*b, l*a-h*c, h*b-k*a) == (0, 0, 0):
            raise ValueError('The in-plane reference must not be parallel to the surface normal.')
        if not math.isfinite(self.phi_zero):
            raise ValueError('The phi offset must be finite.')


@dataclass(frozen=True)
class AngleRange:
    """Accepted numeric bounds; None represents a missing measured angle."""

    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if any(v is not None and not math.isfinite(v) for v in (self.low, self.high)):
            raise ValueError('Angles must be finite.')
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError('Range start must not exceed its end.')
        if self.complete and self.high - self.low > 720:
            raise ValueError('Angle ranges must span at most 720 degrees.')

    @property
    def complete(self) -> bool:
        return self.low is not None and self.high is not None


@dataclass
class PhaseScanState:
    """Accepted scan geometry and orientations, independent of control widgets.

    Geometry contexts use project IDs and axis names, never Python object IDs.
    Source replacement invalidates the affected contexts through ProjectStore.
    """

    mode: str = 'auto'
    max_index: int = 10
    reference_uid: str | None = None
    phase_uid: str | None = None
    angles: dict[tuple[str | None, str], dict[str, AngleRange]] = field(default_factory=dict)
    orientations: dict[str, PhaseOrientation] = field(default_factory=dict)

    @staticmethod
    def context(uid: str | None, scan: Scan1D | None) -> tuple[str | None, str]:
        return uid, scan.axis_name if scan is not None else ''

    def fill_angles(self, uid: str | None, scan: Scan1D | None) -> dict[str, AngleRange]:
        values = measurement_geometry(scan) if scan is not None else {}
        values.setdefault('omega_offset', 0.0)
        bounds = {name: AngleRange(value - .1, value + .1) for name, value in values.items()}
        self.angles[self.context(uid, scan)] = bounds
        return bounds

    def ensure_angles(self, uid: str | None, scan: Scan1D | None) -> dict[str, AngleRange]:
        key = self.context(uid, scan)
        if key not in self.angles:
            return self.fill_angles(uid, scan)
        return self.angles[key]

    @staticmethod
    def validate_angle(name: str, bounds: AngleRange) -> None:
        if name not in {'two_theta', 'omega', 'chi', 'phi', 'omega_offset'}:
            raise ValueError('Unknown fixed angle.')
        if name == 'two_theta' and any(
            v is not None and not 0 < v < 180 for v in (bounds.low, bounds.high)
        ):
            raise ValueError('Fixed 2theta must be between 0 and 180 degrees.')

    def set_angle(self, context: tuple[str | None, str], name: str, bounds: AngleRange) -> None:
        self.validate_angle(name, bounds)
        self.angles[context][name] = bounds

    def resolved_mode(self, scan: Scan1D | None) -> str:
        if self.mode != 'auto':
            if self.mode not in {'powder', *REQUIRED_ANGLES}:
                raise ValueError('Unsupported phase scan mode.')
            return self.mode
        axis = angular_axis(scan.axis_name) if scan is not None else None
        return 'powder' if axis in {None, 'two_theta'} else axis

    def geometry(self, uid: str | None, scan: Scan1D | None) -> PhaseScanGeometry:
        mode = self.resolved_mode(scan)
        if mode == 'powder':
            return PhaseScanGeometry('powder')
        bounds = self.ensure_angles(uid, scan)
        required = REQUIRED_ANGLES[mode]
        if any(not bounds.get(name, AngleRange()).complete for name in required):
            raise ValueError('Missing fixed angles.')
        ranges = tuple((name, bounds[name].low, bounds[name].high) for name in required)
        values = {name: (low + high) / 2 for name, low, high in ranges}
        geometry = PhaseScanGeometry(mode, max_index=self.max_index, ranges=ranges, **values)
        geometry.validate()
        return geometry

    def orientation(self, uid: str | None) -> PhaseOrientation:
        return self.orientations.get(uid, PhaseOrientation())

    def set_orientation(self, uid: str, orientation: PhaseOrientation) -> None:
        orientation.validate()
        self.orientations[uid] = orientation

    def set_phi_offset(self, uid: str, value: float) -> None:
        self.set_orientation(uid, replace(self.orientation(uid), phi_zero=float(value)))

    def invalidate_measurement(self, uid: str) -> None:
        for key in tuple(self.angles):
            if key[0] == uid:
                del self.angles[key]

    def remove(self, uid: str) -> None:
        self.invalidate_measurement(uid)
        self.orientations.pop(uid, None)
        if self.reference_uid == uid:
            self.reference_uid = None
        if self.phase_uid == uid:
            self.phase_uid = None


@dataclass(frozen=True)
class PhaseScanGeometry:
    mode: str
    two_theta: float | None = None
    omega: float | None = None
    chi: float | None = None
    phi: float | None = None
    omega_offset: float = 0.0
    tolerance: float = 0.2
    max_index: int = 10
    ranges: tuple[tuple[str, float, float], ...] = ()

    def bounds(self, name: str) -> tuple[float, float] | None:
        for key, low, high in self.ranges:
            if key == name:
                return low, high
        value = getattr(self, name)
        return None if value is None else (value, value)

    @property
    def axis(self) -> str:
        return 'two_theta' if self.mode in {'powder', 'coupled', 'detector'} else self.mode

    def validate(self) -> None:
        if self.mode not in {'phi', 'chi', 'omega', 'coupled', 'detector'}:
            raise ValueError('Unsupported oriented scan mode.')
        required = {'phi': ('two_theta', 'omega', 'chi'),
                    'chi': ('two_theta', 'omega', 'phi'),
                    'omega': ('two_theta', 'chi', 'phi'),
                    'coupled': ('chi', 'phi'),
                    'detector': ('omega', 'chi', 'phi')}[self.mode]
        missing = [key for key in required if self.bounds(key) is None]
        if missing:
            raise ValueError('Missing fixed angles: ' + ', '.join(missing))
        for value in (self.two_theta, self.omega, self.chi, self.phi, self.omega_offset):
            if value is not None and not math.isfinite(value):
                raise ValueError('Angles must be finite.')
        names = set()
        for name, low, high in self.ranges:
            if name not in {'two_theta', 'omega', 'chi', 'phi', 'omega_offset'} or name in names:
                raise ValueError('Unknown or duplicate fixed-angle range.')
            names.add(name)
            if not (math.isfinite(low) and math.isfinite(high) and low <= high and high-low <= 720):
                raise ValueError('Angle ranges must be finite, ordered and span at most 720 degrees.')
        tt = self.bounds('two_theta')
        if self.mode in {'phi', 'chi', 'omega'} and not 0 < tt[0] <= tt[1] < 180:
            raise ValueError('Fixed 2theta must be between 0 and 180 degrees.')
        if not math.isfinite(self.tolerance) or not 0 < self.tolerance <= 5:
            raise ValueError('Angular acceptance must be between 0 and 5 degrees.')
        if not isinstance(self.max_index, int) or isinstance(self.max_index, bool) or not 1 <= self.max_index <= 20:
            raise ValueError('Maximum hkl index must be between 1 and 20.')


@dataclass(frozen=True)
class ScanReflectionRow(ReflectionRow):
    coordinate: float = 0.0
    scan_axis: str = ''
    angular_mismatch: float = 0.0


def measurement_geometry(scan: Scan1D) -> dict[str, float]:
    """Read actual fixed positions without averaging a moving axis."""
    fixed = {}
    series = {}
    for name, values in scan.metadata.get('axes', {}).items():
        axis = angular_axis(name)
        if axis is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.size and np.all(np.isfinite(values)):
            series[axis] = values
            if np.ptp(values) <= 1e-6:
                fixed[axis] = float(values[0])
    active = angular_axis(scan.axis_name)
    for name, value in scan.metadata.get('raw_drive_starts', {}).items():
        axis = angular_axis(name)
        if axis is None or axis == active or axis in series:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            fixed[axis] = value
    tt, om = series.get('two_theta'), series.get('omega')
    if tt is not None and om is not None and tt.shape == om.shape:
        offset = om - tt / 2
        if np.ptp(offset) <= 1e-6:
            fixed['omega_offset'] = float(offset[0])
    if 'omega_offset' not in fixed and 'omega' in fixed and 'two_theta' in fixed:
        fixed['omega_offset'] = fixed['omega'] - fixed['two_theta'] / 2
    return fixed
