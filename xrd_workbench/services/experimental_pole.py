"""Experimental pole grids: original samples and gaps, with no interpolation."""
from __future__ import annotations
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..bruker_raw import read_bruker_raw

RAW3_FILE_HEADER_SIZE = 712
RAW3_RANGE_HEADER_SIZE = 304
RAW3_MAGIC = b"RAW1.01\x00"


@dataclass
class BrukerRawRange:
    index: int
    theta: float
    two_theta: float
    chi: float
    phi_start: float
    phi_step: float
    time_per_step: float
    intensity: np.ndarray

    @property
    def phi(self) -> np.ndarray:
        return self.phi_start + self.phi_step * np.arange(self.intensity.size)


@dataclass
class BrukerRawPoleData:
    declared_ranges: int
    status_code: int
    ranges: list[BrukerRawRange]


def _raw_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _raw_f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _raw_f64(data: bytes, offset: int) -> float:
    return struct.unpack_from("<d", data, offset)[0]


def read_bruker_raw3(path: Path) -> BrukerRawPoleData:
    """Read a Siemens/Bruker RAW version 3 pole measurement (RAW1.01)."""
    data = path.read_bytes()
    if len(data) < RAW3_FILE_HEADER_SIZE:
        raise ValueError("The file is shorter than the RAW1.01 file header.")
    if data[:8] != RAW3_MAGIC:
        signature = data[:8].rstrip(b"\x00")
        raise ValueError(f"Unsupported RAW signature: {signature!r}.")

    status_code = _raw_u32(data, 8)
    declared_ranges = _raw_u32(data, 12)
    if declared_ranges == 0:
        raise ValueError("The RAW file declares no measurement ranges.")
    if declared_ranges > 100_000:
        raise ValueError(f"Implausible RAW range count: {declared_ranges}.")

    ranges: list[BrukerRawRange] = []
    offset = RAW3_FILE_HEADER_SIZE

    for index in range(declared_ranges):
        if offset + RAW3_RANGE_HEADER_SIZE > len(data):
            raise ValueError(
                f"Range {index}: its header extends beyond the end of the RAW file."
            )

        header_size = _raw_u32(data, offset)
        points = _raw_u32(data, offset + 4)
        supplementary_size = _raw_u32(data, offset + 256)
        if header_size != RAW3_RANGE_HEADER_SIZE:
            raise ValueError(
                f"Range {index}: expected a {RAW3_RANGE_HEADER_SIZE}-byte header, "
                f"found {header_size} bytes."
            )

        data_offset = offset + header_size + supplementary_size
        data_end = data_offset + 4 * points
        if data_offset > len(data) or data_end > len(data):
            available = max(0, (len(data) - data_offset) // 4)
            raise ValueError(
                f"Range {index}: {points} points are declared, but only "
                f"{available} are available."
            )

        intensity = np.frombuffer(
            data,
            dtype="<f4",
            count=points,
            offset=data_offset,
        ).astype(float, copy=True)
        theta = _raw_f64(data, offset + 8)
        two_theta = _raw_f64(data, offset + 16)
        chi = _raw_f64(data, offset + 24)
        phi_start = _raw_f64(data, offset + 32)
        phi_step = _raw_f64(data, offset + 176)
        time_per_step = _raw_f32(data, offset + 192)

        numeric_header = (theta, two_theta, chi, phi_start, phi_step, time_per_step)
        if not all(math.isfinite(value) for value in numeric_header):
            raise ValueError(f"Range {index}: its angular header contains invalid values.")
        if points > 1 and math.isclose(phi_step, 0.0, abs_tol=1e-15):
            raise ValueError(f"Range {index}: Phi step is zero for {points} points.")

        ranges.append(
            BrukerRawRange(
                index=index,
                theta=theta,
                two_theta=two_theta,
                chi=chi,
                phi_start=phi_start,
                phi_step=phi_step,
                time_per_step=time_per_step,
                intensity=intensity,
            )
        )
        offset = data_end

    trailing = data[offset:]
    if trailing and any(byte != 0 for byte in trailing):
        raise ValueError(f"The RAW file contains {len(trailing)} unexpected trailing bytes.")

    return BrukerRawPoleData(
        declared_ranges=declared_ranges,
        status_code=status_code,
        ranges=ranges,
    )


def prepare_scan(
    phi: np.ndarray, intensity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove only a duplicated 360° endpoint, preserving all measured gaps."""
    if phi.size > 1 and phi[0] > phi[-1]:
        phi = phi[::-1]
        intensity = intensity[::-1]
    if phi.size > 2 and math.isclose(phi[-1] - phi[0], 360.0, abs_tol=0.05):
        return phi[:-1], intensity[:-1]
    return phi, intensity


def scans_from_raw(
    raw: BrukerRawPoleData,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Return per-range original Phi grids and their measured Chi positions."""
    ordered = sorted(raw.ranges, key=lambda scan: scan.chi)
    radii = np.asarray([scan.chi for scan in ordered], dtype=float)
    if radii.size == 0:
        raise ValueError("The RAW file contains no ranges.")
    if np.any(radii < 0):
        raise ValueError("The RAW file contains negative Chi values.")
    if radii.size > 1 and np.any(np.diff(radii) <= 0):
        raise ValueError("The RAW file contains duplicate Chi values.")

    scans = [prepare_scan(scan.phi, scan.intensity) for scan in ordered]
    if not any(intensity.size for _phi, intensity in scans):
        raise ValueError("The RAW file contains no recorded intensity points.")
    if not any(np.any(np.isfinite(intensity)) for _phi, intensity in scans):
        raise ValueError("The RAW file contains no finite intensity values.")
    return scans, radii


def read_xy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two numeric columns while ignoring arbitrary headers."""
    angles: list[float] = []
    intensities: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                angle = float(parts[0])
                intensity = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(angle) and math.isfinite(intensity):
                angles.append(angle)
                intensities.append(intensity)

    if len(angles) < 2:
        raise ValueError(f"No usable two-column numeric data found in {path.name}.")

    angle_array = np.asarray(angles, dtype=float)
    intensity_array = np.asarray(intensities, dtype=float)
    order = np.argsort(angle_array)
    angle_array = angle_array[order]
    intensity_array = intensity_array[order]

    # Keep one value per angle so the plotting cells remain unambiguous.
    unique_angles, unique_indices = np.unique(angle_array, return_index=True)
    return unique_angles, intensity_array[unique_indices]


def find_exported_xy_files(raw_path: Path) -> list[Path]:
    """Return matching XY exports in their numeric order."""
    pattern = re.compile(
        rf"^{re.escape(raw_path.stem)}_exported_(\d+)\.xy$", re.IGNORECASE
    )
    matches: list[tuple[int, Path]] = []

    for candidate in raw_path.parent.iterdir():
        if not candidate.is_file():
            continue
        match = pattern.match(candidate.name)
        if match:
            matches.append((int(match.group(1)), candidate))

    if not matches:
        return []

    matches.sort(key=lambda item: item[0])
    indices = [item[0] for item in matches]
    expected = list(range(0, indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        missing_text = ", ".join(str(index) for index in missing)
        raise ValueError(f"Missing exported XY file number(s): {missing_text}.")

    return [item[1] for item in matches]


def load_xy_series(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load every XY file on its original Phi grid without interpolation."""
    scans: list[tuple[np.ndarray, np.ndarray]] = []

    for path in paths:
        phi, intensity = read_xy_file(path)
        scans.append(prepare_scan(phi, intensity))

    if not scans:
        raise ValueError("No XY data were loaded.")
    return scans


def split_continuous_segments(
    phi: np.ndarray, intensity: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a scan at missing angular intervals so gaps remain unpainted."""
    if phi.size == 0:
        return []
    if phi.size < 2:
        return [(phi, intensity)]

    differences = np.diff(phi)
    positive_differences = differences[differences > 0]
    if positive_differences.size == 0:
        return [(phi, intensity)]

    usual_step = float(np.median(positive_differences))
    break_points = np.flatnonzero(differences > usual_step * 1.5) + 1
    phi_parts = np.split(phi, break_points)
    intensity_parts = np.split(intensity, break_points)
    return [
        (phi_part, intensity_part)
        for phi_part, intensity_part in zip(phi_parts, intensity_parts)
        if phi_part.size > 0
    ]


def centres_to_edges(values: np.ndarray, single_width: float = 1.0) -> np.ndarray:
    """Convert monotonically increasing cell centres to pcolormesh edges."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        half_width = single_width / 2.0
        return np.array([values[0] - half_width, values[0] + half_width])

    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def format_number(value: float) -> str:
    return f"{value:.10g}"


def scaled_axes_position(
    position: tuple[float, float, float, float],
    cursor: tuple[float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    """Magnify an already drawn axes around a cursor in figure coordinates."""

    x0, y0, width, height = position
    cursor_x, cursor_y = cursor
    return (
        cursor_x - (cursor_x - x0) * scale,
        cursor_y - (cursor_y - y0) * scale,
        width * scale,
        height * scale,
    )



@dataclass
class ExperimentalPoleMeasurement:
    source: Path
    scans: list[tuple[np.ndarray, np.ndarray]]
    radii: np.ndarray | None
    raw: object | None = None
    raw_error: str | None = None
    is_pole_figure: bool = True


def load_experimental_pole(path, raw_data=None) -> ExperimentalPoleMeasurement:
    """Read RAW or its numbered XY fallback, retaining every original grid."""
    if isinstance(raw_data, ExperimentalPoleMeasurement):
        return raw_data
    source = Path(path)
    raw_error = None
    raw = None
    radii = None
    try:
        raw = raw_data if raw_data is not None else read_bruker_raw(source)
        if not raw.is_pole_figure:
            raise ValueError("RAW is not a pole-figure measurement.")
        ordered = sorted(raw.ranges, key=lambda scan: scan.chi)
        radii = np.asarray([scan.chi for scan in ordered], dtype=float)
        if not radii.size or not np.all(np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("RAW contains no valid tilt angles.")
        if np.any(np.diff(radii) <= 0):
            raise ValueError("RAW contains duplicate tilt angles.")
        scans = [prepare_scan(scan.phi, scan.intensity) for scan in ordered]
    except (OSError, ValueError, struct.error) as error:
        raw_error = str(error)
        raw = None
        radii = None
        try:
            scans = load_xy_series(find_exported_xy_files(source))
        except (OSError, ValueError) as xy_error:
            raise ValueError(f"RAW: {raw_error}\nXY: {xy_error}") from xy_error
    if not any(np.isfinite(intensity).any() for _phi, intensity in scans):
        raise ValueError("The measurement contains no finite intensity values.")
    return ExperimentalPoleMeasurement(source, scans, radii, raw, raw_error)
