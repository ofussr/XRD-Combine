"""Position-only oriented scans using the shared crystal and RSM geometry.

At chi=phi=0 the sample scattering vector is the existing RSM (Qx, 0, Qz).
The inner sample rotations give q_sample = Rz(-phi) Rx(-chi) q_RSM.
Phi is the instrument rotation, not the plotted pole azimuth. All angles are
degrees; lengths and wavelength are angstroms. Intensities are deliberately
unspecified, and profile widths belong to the display, not this calculation.
"""

from __future__ import annotations

import math
from dataclasses import replace
from threading import Event
from typing import Sequence

import numpy as np

from ..models.crystal import CrystalStructure
from ..models.diffraction import DiffractionStructure, RadiationLine, ScatteringFactors
from ..models.phase_scan import (
    PhaseOrientation, PhaseScanGeometry, ScanReflectionRow, angular_axis, measurement_geometry,
)
from .diffraction import _structure_factor_squared
from .pole_figure import rotation_x, rotation_z
from .rsm import angles_to_q, make_orientation


def scattering_vector(two_theta, omega, chi, phi, wavelength):
    """Four-circle vector consistent with the application's existing RSM convention."""
    qx, qz = angles_to_q(omega, two_theta, wavelength)
    return rotation_z(-phi) @ rotation_x(-chi) @ np.array([qx, 0.0, qz])


def _path_vector(geometry, coordinate, wavelength, phi_zero):
    tt, om, chi, phi = geometry.two_theta, geometry.omega, geometry.chi, geometry.phi
    if geometry.mode == 'coupled':
        tt, om = coordinate, coordinate / 2 + geometry.omega_offset
    elif geometry.mode == 'detector':
        tt = coordinate
    elif geometry.mode == 'omega':
        om = coordinate
    elif geometry.mode == 'chi':
        chi = coordinate
    elif geometry.mode == 'phi':
        phi = coordinate
    return scattering_vector(tt, om, chi, phi - phi_zero, wavelength)


def _periodic_positions(centre, limits):
    low, high = limits
    first = math.ceil((low - centre - 1e-9) / 360)
    last = math.floor((high - centre + 1e-9) / 360)
    return [centre + 360 * turn for turn in range(first, last + 1)]


def _lift_angle(angle, bounds):
    """Pick the periodic copy nearest the middle of a bounded angular interval."""
    copies = _periodic_positions(angle, bounds)
    if not copies:
        return None
    middle = sum(bounds)/2
    return min(copies, key=lambda value: abs(value-middle))


def _range_positions(direction, tt, geometry, orientation, limits):
    """Intersect exact four-circle solutions with each fixed-angle interval.

    Parameterise the solution curve by phi. Chi and omega boundaries have
    analytic sine/cosine roots, so testing the intervals between those roots
    does not miss narrow or zero-width windows. A bounded numerical search
    selects one representative nearest the nominal fixed settings; it is not
    used to decide whether an intersection exists.
    """
    mode = geometry.mode
    phi_bounds = limits if mode == 'phi' else geometry.bounds('phi')
    chi_bounds = limits if mode == 'chi' else geometry.bounds('chi')
    if mode == 'coupled':
        omega_bounds = tuple(tt/2+v for v in geometry.bounds('omega_offset'))
    else:
        omega_bounds = limits if mode == 'omega' else geometry.bounds('omega')
    px, py, pz = map(float, direction)
    rho = math.hypot(px, py)
    if mode == 'phi' and rho < 1e-10:
        return []  # A pole on the rotation axis has no isolated phi peak.
    beta = math.degrees(math.atan2(py, px))
    phi_origin = orientation.phi_zero-beta
    cuts = {float(phi_bounds[0]), float(phi_bounds[1])}

    def add_root(value):
        cuts.update(_periodic_positions(value+phi_origin, phi_bounds))

    for value in (0,90,180,270):
        add_root(value)
    if mode != 'phi':
        cuts.add(sum(phi_bounds)/2)
    if rho > 1e-12:
        for chi in (*chi_bounds, sum(chi_bounds)/2):
            cosine = math.cos(math.radians(chi))
            if abs(cosine) < 1e-12:
                continue
            sine = pz*math.tan(math.radians(chi))/rho
            if abs(sine) <= 1+1e-12:
                root = math.degrees(math.asin(max(-1.,min(1.,sine))))
                add_root(root)
                add_root(180-root)
        for omega in (*omega_bounds, sum(omega_bounds)/2):
            cosine = math.sin(math.radians(omega-tt/2))/rho
            if abs(cosine) <= 1+1e-12:
                root = math.degrees(math.acos(max(-1.,min(1.,cosine))))
                add_root(root)
                add_root(-root)

    best = None
    for sign in (1,-1):
        def candidate(phi):
            v = math.radians(phi-phi_origin)
            sx, sy = rho*math.cos(v), rho*math.sin(v)
            length = math.hypot(sy,pz)
            if length < 1e-10:
                if mode == 'chi':
                    return None  # Chi is free: there is no isolated chi peak.
                chi = sum(chi_bounds)/2
            else:
                chi = math.degrees(math.atan2(sy,pz))+(180 if sign < 0 else 0)
            omega = tt/2+math.degrees(math.atan2(sx,sign*length))
            chi = _lift_angle(chi,chi_bounds)
            omega = _lift_angle(omega,omega_bounds)
            if chi is None or omega is None:
                return None
            values = {'phi':phi, 'chi':chi, 'omega':omega}
            score = 0.
            for name,bounds in (('phi',phi_bounds), ('chi',chi_bounds), ('omega',omega_bounds)):
                if name != mode:
                    scale = max((bounds[1]-bounds[0])/2, 1e-6)
                    score += ((values[name]-sum(bounds)/2)/scale)**2
            coordinate = tt if geometry.axis == 'two_theta' else values[mode]
            return score, coordinate

        ordered = sorted(cuts)
        probes = list(ordered)
        for low,high in zip(ordered,ordered[1:]):
            middle = (low+high)/2
            if high-low < 1e-10 or candidate(middle) is None:
                continue
            probes.append(middle)
            # Every point in this segment satisfies the angular windows.
            ratio = (math.sqrt(5)-1)/2
            a,b = low,high
            c,d = b-ratio*(b-a), a+ratio*(b-a)
            def score(value):
                result = candidate(value)
                return math.inf if result is None else result[0]
            fc,fd = score(c),score(d)
            for _ in range(42):
                if fc < fd:
                    b,d,fd = d,c,fc
                    c = b-ratio*(b-a)
                    fc = score(c)
                else:
                    a,c,fc = c,d,fd
                    d = a+ratio*(b-a)
                    fd = score(d)
            probes.append((a+b)/2)
        for phi in probes:
            result = candidate(phi)
            if result is not None and (best is None or result[0] < best[0]-1e-12):
                best = result
    if best is None:
        return []
    coordinate = best[1]
    return [coordinate] if geometry.axis == 'two_theta' else _periodic_positions(coordinate,limits)


def shift_phi_reflections(rows, offset, limits):
    """Shift a cached full turn and crop periodic copies to the displayed scan."""
    result, seen = [],set()
    for row in rows:
        centre = (row.coordinate+offset)%360
        for coordinate in _periodic_positions(centre,limits):
            key = row.hkl,row.radiation,round(coordinate,9)
            if key not in seen:
                seen.add(key)
                result.append(replace(row,coordinate=coordinate))
    return sorted(result,key=lambda row:(row.coordinate,row.hkl,row.radiation))


def calculate_scan_reflections(
    crystal: CrystalStructure,
    structure: DiffractionStructure,
    factors: ScatteringFactors,
    radiations: Sequence[RadiationLine],
    geometry: PhaseScanGeometry,
    orientation: PhaseOrientation,
    limits: tuple[float, float],
    cancel_event: Event | None = None,
) -> list[ScanReflectionRow]:
    """Enumerate signed hkl and find closest positions on the specified angular path.

    Fixed-2theta mismatch and angular direction mismatch must both lie inside
    the explicit acceptance. A reflection invariant over a whole scan has no
    isolated peak position and is omitted. No Friedel folding or powder-family
    multiplicity is used. Bounds in hkl are explicit and finite.
    """
    geometry.validate()
    low, high = map(float, limits)
    if not (math.isfinite(low) and math.isfinite(high) and low < high and high - low <= 720):
        raise ValueError('Scan bounds must be finite, ordered and span at most 720 degrees.')
    if not radiations or any(not math.isfinite(w) or w <= 0 or not math.isfinite(weight)
                             or weight <= 0 for _, w, weight in radiations):
        raise ValueError('Positive finite radiation wavelengths and weights are required.')
    if not math.isfinite(orientation.phi_zero):
        raise ValueError('Azimuth offset must be finite.')
    basis = make_orientation(crystal, orientation.surface, orientation.inplane)
    n = geometry.max_index
    indices = np.array(np.meshgrid(*([np.arange(-n, n + 1)] * 3), indexing='ij')).reshape(3, -1).T
    vectors = indices @ (2 * np.pi * crystal.reciprocal).T
    lengths = np.linalg.norm(vectors, axis=1)
    candidates = []
    max_f2 = 0.0
    for index, (hkl, vector, length) in enumerate(zip(indices, vectors, lengths)):
        if cancel_event is not None and index % 128 == 0 and cancel_event.is_set():
            return []
        if length < 1e-12 or crystal.is_systematically_absent(hkl):
            continue
        d = 2 * np.pi / length
        eligible = []
        for radiation, wavelength, weight in radiations:
            argument = wavelength / (2 * d)
            if argument >= 1:
                continue
            tt = math.degrees(2 * math.asin(argument))
            if geometry.axis == 'two_theta':
                if not low <= tt <= high:
                    continue
            elif geometry.ranges:
                low_tt,high_tt = geometry.bounds('two_theta')
                if not low_tt-1e-9 <= tt <= high_tt+1e-9:
                    continue
            elif abs(tt - geometry.two_theta) > geometry.tolerance:
                continue
            eligible.append((radiation, wavelength, weight, tt))
        if not eligible:
            continue
        f2 = None
        if structure.atoms and not structure.cell_only:
            f2 = _structure_factor_squared(*map(int, hkl), d, structure.atoms, factors)
            max_f2 = max(max_f2, f2)
        candidates.append((tuple(map(int, hkl)), basis.to_sample(vector), d, f2, eligible))
    result = []
    for hkl, vector, d, f2, radiations_for_line in candidates:
        if cancel_event is not None and cancel_event.is_set():
            return []
        if f2 is not None and f2 < max(max_f2 * 1e-12, 1e-10):
            continue
        direction = vector / np.linalg.norm(vector)
        for radiation, wavelength, weight, tt in radiations_for_line:
            if geometry.ranges:
                for coordinate in _range_positions(direction,tt,geometry,orientation,limits):
                    result.append(ScanReflectionRow(
                        hkl='('+' '.join(map(str,hkl))+')', d=d, two_theta=tt,
                        radiation=radiation,wavelength=wavelength,weight=weight,
                        multiplicity=1,f2_sum=f2,intensity=None,equivalents=(hkl,),
                        coordinate=float(coordinate),scan_axis=geometry.axis,angular_mismatch=0.))
                continue
            if geometry.axis == 'two_theta':
                positions = [tt]
            else:
                # One moving sample rotation is exactly C + A*cos(t) + B*sin(t).
                q0 = _path_vector(geometry, 0, wavelength, orientation.phi_zero)
                q180 = _path_vector(geometry, 180, wavelength, orientation.phi_zero)
                constant = .5 * (q0 + q180)
                cosine = .5 * (q0 - q180)
                sine = _path_vector(geometry, 90, wavelength, orientation.phi_zero) - constant
                a, b = float(direction @ cosine), float(direction @ sine)
                if math.hypot(a, b) < 1e-10 * np.linalg.norm(q0):
                    continue  # No isolated maximum along this path.
                centre = math.degrees(math.atan2(b, a))
                positions = _periodic_positions(centre, limits)
            for coordinate in positions:
                predicted = _path_vector(geometry, coordinate, wavelength, orientation.phi_zero)
                predicted /= np.linalg.norm(predicted)
                mismatch = math.degrees(math.acos(float(np.clip(predicted @ direction, -1, 1))))
                if mismatch <= geometry.tolerance + 1e-8:
                    result.append(ScanReflectionRow(
                        hkl='(' + ' '.join(map(str, hkl)) + ')', d=d, two_theta=tt,
                        radiation=radiation, wavelength=wavelength, weight=weight,
                        multiplicity=1, f2_sum=f2, intensity=None, equivalents=(hkl,),
                        coordinate=float(coordinate), scan_axis=geometry.axis,
                        angular_mismatch=mismatch))
    return sorted(result, key=lambda row: (row.coordinate, row.hkl, row.radiation))


def scan_profile(rows, limits, fwhm, point_count=5000):
    """Uniform-height illustrative Gaussian peaks; preserve physical 2theta in rows."""
    from .diffraction import gaussian_powder_profile

    display_rows = [replace(row, two_theta=row.coordinate, intensity=100.0) for row in rows]
    profile = gaussian_powder_profile(display_rows, *limits, fwhm, point_count=point_count,
                                      normalize_to=1.0)
    return profile.x, profile.total
