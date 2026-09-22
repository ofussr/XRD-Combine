"""Reciprocal-space maps from the application's shared RAW and crystal models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models.crystal import CrystalStructure


@dataclass(frozen=True)
class MapData:
    two_theta: np.ndarray
    omega: np.ndarray
    intensity: np.ndarray  # (omega, 2theta); missing acquisition points are NaN
    description: str


def _grid(slices, coordinate: str):
    """Return a common inner grid and a matrix preserving unfinished slices."""
    active = [item for item in slices if item.point_count]
    if not active:
        raise ValueError("The measurement has no recorded points.")
    reference = max(active, key=lambda item: item.point_count)
    result = np.asarray(reference.coordinate(coordinate), dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Missing {coordinate} scan coordinates.")
    data = np.full((len(slices), len(result)), np.nan, dtype=float)
    for row, item in enumerate(slices):
        if not item.point_count:
            continue
        values = np.asarray(item.coordinate(coordinate), dtype=float)
        if (item.point_count > result.size
                or not np.allclose(values, result[:len(values)], rtol=0, atol=1e-6)):
            raise ValueError(f"{coordinate} coordinates differ between RAW ranges.")
        data[row, :item.point_count] = item.intensity
    return result, data


def build_raw_map(raw_files) -> MapData:
    """Use pointwise scan paths; include empty and partial RAW ranges as gaps."""
    sources = list(raw_files)
    if not sources:
        raise ValueError("Select at least one RAW file.")
    ranges = [item for raw in sources for item in raw.ranges]
    recorded = [item for item in ranges if item.point_count]
    if len(recorded) < 2:
        raise ValueError("An RSM needs at least two measured ranges.")
    primary = {scan.scan_path.primary_drive for scan in recorded}
    if primary <= {"Theta", "Omega"}:
        drive = next(iter(primary)) if len(primary) == 1 else None
        if drive is None:
            raise ValueError("Theta and Omega ranges cannot be mixed in one map.")
        pairs = sorted(
            [(float(scan.drives.get("2Theta", np.nan)), scan) for scan in ranges],
            key=lambda pair: pair[0],
        )
        outer = np.array([pair[0] for pair in pairs])
        if not np.all(np.isfinite(outer)) or np.any(np.diff(outer) <= 0):
            raise ValueError("Fixed 2Theta values must be finite and distinct.")
        omega, rows = _grid([pair[1] for pair in pairs], drive)
        return MapData(outer, omega, rows.T, f"{drive} scan at fixed 2Theta; {len(ranges)} ranges")
    if primary == {"2Theta"}:
        pairs = sorted(
            [(float(scan.drives.get("Omega", scan.drives.get("Theta", np.nan))), scan)
             for scan in ranges],
            key=lambda pair: pair[0],
        )
        outer = np.array([pair[0] for pair in pairs])
        if not np.all(np.isfinite(outer)) or np.any(np.diff(outer) <= 0):
            raise ValueError("Fixed Omega/Theta values must be finite and distinct.")
        two_theta, rows = _grid([pair[1] for pair in pairs], "2Theta")
        return MapData(two_theta, outer, rows, f"2Theta scan at fixed Omega; {len(ranges)} ranges")
    raise ValueError("RAW geometry is ambiguous or incompatible with a two-axis RSM.")


def angle_series(count: int, first: float | None, step: float | None,
                 last: float | None) -> np.ndarray:
    if count < 2:
        raise ValueError("Select at least two scans.")
    if sum(value is not None for value in (first, step, last)) < 2:
        raise ValueError("Enter two of: first angle, step, last angle.")
    if first is None:
        first = last - step * (count - 1)
    elif step is None:
        step = (last - first) / (count - 1)
    elif last is None:
        last = first + step * (count - 1)
    elif not np.isclose(first + step * (count - 1), last, atol=1e-6, rtol=0):
        raise ValueError("The entered angular series is inconsistent.")
    return float(first) + float(step) * np.arange(count)


def build_scan_map(scans, first: float | None, step: float | None,
                   last: float | None) -> MapData:
    sources = list(scans)
    omega = angle_series(len(sources), first, step, last)
    if np.any(np.diff(omega) <= 0):
        raise ValueError("Omega angles must increase from one scan to the next.")
    two_theta = np.asarray(sources[0].x, dtype=float)
    # Choosing a two-column XY series for RSM explicitly declares X as 2Theta.
    # Leave the shared Scan1D axis label untouched for the other pages.
    if not all((item.axis_name == "2Theta" or
                (item.axis_name == "X" and item.metadata.get("format") == "XY"))
               and len(item.x) == len(two_theta)
               and np.allclose(item.x, two_theta, atol=1e-7, rtol=0) for item in sources):
        raise ValueError("All scans must have the same 2Theta grid.")
    return MapData(two_theta, omega, np.vstack([item.y for item in sources]),
                   f"{len(sources)} 2Theta scans with entered Omega angles")


def axis_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Map axis is empty.")
    if values.size == 1:
        return np.array([values[0] - .5, values[0] + .5])
    mid = .5 * (values[1:] + values[:-1])
    return np.r_[values[0] - .5 * (values[1] - values[0]),
                 mid, values[-1] + .5 * (values[-1] - values[-2])]


def display_mesh(data: MapData, mode: str, wavelength: float):
    """Return corner grids and point cells for exact PyQtGraph mesh drawing."""
    z = np.asarray(data.intensity, dtype=float)
    if z.shape != (len(data.omega), len(data.two_theta)):
        raise ValueError("RSM intensity shape does not match its coordinates.")
    omega = axis_edges(data.omega)
    two_theta = axis_edges(data.two_theta)
    if mode == "angular":
        x, y = np.meshgrid(omega, two_theta)
        return x, y, z.T, "Omega / Theta, °", "2Theta, °"
    if mode != "q" or not math.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("Select angular or q space and a positive wavelength.")
    tt, om = np.meshgrid(two_theta, omega)
    scale = 2 * np.pi / wavelength
    qx = scale * (np.cos(np.deg2rad(tt - om)) - np.cos(np.deg2rad(om)))
    qz = scale * (np.sin(np.deg2rad(tt - om)) + np.sin(np.deg2rad(om)))
    return qx, qz, z, "Qx, Å^-1", "Qz, Å^-1"


def map_view_bounds(data: MapData, mode: str, wavelength: float,
                    measured_only: bool) -> tuple[float, float, float, float]:
    """Bounds of all declared cells or the corners of recorded cells only."""
    x, y, _z, _xlabel, _ylabel = display_mesh(data, mode, wavelength)
    if measured_only:
        recorded = np.isfinite(data.intensity)
        if mode == "angular":
            recorded = recorded.T
        if not np.any(recorded):
            raise ValueError("The map has no recorded points.")
        vertices = np.zeros(x.shape, dtype=bool)
        vertices[:-1, :-1] |= recorded
        vertices[1:, :-1] |= recorded
        vertices[:-1, 1:] |= recorded
        vertices[1:, 1:] |= recorded
        x, y = x[vertices], y[vertices]
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def measured_reflections(data: MapData, points: list[ReflectionPoint]):
    """Keep only reflections inside a cell with an acquired intensity."""
    omega_edges = axis_edges(data.omega)
    two_theta_edges = axis_edges(data.two_theta)
    result = []
    for point in points:
        if point.omega is None or point.two_theta is None:
            continue
        row = int(np.searchsorted(omega_edges, point.omega, side="right") - 1)
        col = int(np.searchsorted(two_theta_edges, point.two_theta, side="right") - 1)
        if (0 <= row < len(data.omega) and 0 <= col < len(data.two_theta)
                and np.isfinite(data.intensity[row, col])):
            result.append(point)
    return result


@dataclass(frozen=True)
class Orientation:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def to_sample(self, vector):
        return np.array([np.dot(vector, axis) for axis in (self.x, self.y, self.z)])


def make_orientation(crystal: CrystalStructure, surface_hkl, inplane_hkl) -> Orientation:
    surface = np.asarray(surface_hkl, dtype=int)
    inplane = np.asarray(inplane_hkl, dtype=int)
    if not np.any(surface) or not np.any(inplane):
        raise ValueError("Surface and in-plane hkl must be nonzero.")
    z = np.asarray(crystal.reciprocal_vector(surface), dtype=float)
    z /= np.linalg.norm(z)
    x = np.asarray(crystal.reciprocal_vector(inplane), dtype=float)
    x -= np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-10:
        raise ValueError("In-plane reflection is parallel to the surface normal.")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Orientation(x, y, z)


def angles_to_q(omega: float, two_theta: float, wavelength: float):
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    k = 2 * np.pi / wavelength
    a = np.deg2rad(omega)
    b = np.deg2rad(two_theta - omega)
    return float(k * (np.cos(b) - np.cos(a))), float(k * (np.sin(b) + np.sin(a)))


def q_to_angles(qx: float, qz: float, wavelength: float):
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    argument = wavelength * math.hypot(qx, qz) / (4 * math.pi)
    if argument > 1 + 1e-12:
        raise ValueError("Target lies outside the Ewald sphere.")
    theta = math.asin(min(argument, 1))
    omega = theta + math.atan2(qx, qz)
    return math.degrees(omega), math.degrees(2 * theta)


@dataclass(frozen=True)
class ReflectionPoint:
    hkl: tuple[int, int, int]
    qx: float
    qy: float
    qz: float
    omega: float | None
    two_theta: float | None


def reflection_point(crystal: CrystalStructure, orientation: Orientation,
                     hkl, wavelength: float) -> ReflectionPoint:
    vector = orientation.to_sample(2 * np.pi * crystal.reciprocal_vector(hkl))
    qx, qy, qz = map(float, vector)
    try:
        omega, two_theta = q_to_angles(qx, qz, wavelength)
    except ValueError:
        omega = two_theta = None
    return ReflectionPoint(tuple(map(int, hkl)), qx, qy, qz, omega, two_theta)


def calculate_reflections(crystal: CrystalStructure, orientation: Orientation,
                          wavelength: float, max_index: int, qy_tolerance: float):
    if not 1 <= max_index <= 20 or qy_tolerance < 0:
        raise ValueError("Max index must be 1–20 and q-plane tolerance nonnegative.")
    result = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == k == l == 0 or crystal.is_systematically_absent((h, k, l)):
                    continue
                point = reflection_point(crystal, orientation, (h, k, l), wavelength)
                if abs(point.qy) <= qy_tolerance and point.omega is not None:
                    result.append(point)
    return result
