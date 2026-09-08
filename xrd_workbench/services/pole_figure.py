"""GUI-independent geometry and calculations for theoretical pole figures."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Sequence

import numpy as np

from ..models.crystal import CrystalStructure
from ..models.data_errors import XRDDataError
from ..models.diffraction import DiffractionStructure, ScatteringFactors
from ..models.pole_figure import PolePoint, PoleReflection
from .diffraction import calculate_reflections


def projection_code(value: str) -> str:
    choices = {
        "stereographic": {"stereographic", "stéréographique", "стереографическая"},
        "equal_area": {"equal_area", "equal-area", "équivalente", "равноплощадная"},
    }
    normalized = str(value).strip().lower()
    for code, aliases in choices.items():
        if normalized in aliases:
            return code
    return normalized


def rotation_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_array))
    if norm < 1e-14 or abs(angle_rad) < 1e-14:
        return np.eye(3)
    x, y, z = axis_array / norm
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ]
    )


def rotation_x(angle_deg: float) -> np.ndarray:
    return rotation_axis_angle((1.0, 0.0, 0.0), math.radians(angle_deg))


def rotation_y(angle_deg: float) -> np.ndarray:
    return rotation_axis_angle((0.0, 1.0, 0.0), math.radians(angle_deg))


def rotation_z(angle_deg: float) -> np.ndarray:
    return rotation_axis_angle((0.0, 0.0, 1.0), math.radians(angle_deg))


def euler_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    return rotation_z(z_deg) @ rotation_y(y_deg) @ rotation_x(x_deg)


def follow_orientation_change(
    follower_user_rotation: np.ndarray,
    primary_before: np.ndarray,
    primary_after: np.ndarray,
) -> np.ndarray:
    """Apply a primary layer's orientation delta to another layer.

    The follower keeps its own base orientation and relative offset.  All
    matrices are rotations, so the inverse of the previous primary
    orientation is its transpose.
    """

    follower = np.asarray(follower_user_rotation, dtype=float)
    before = np.asarray(primary_before, dtype=float)
    after = np.asarray(primary_after, dtype=float)
    if follower.shape != (3, 3) or before.shape != (3, 3) or after.shape != (3, 3):
        raise ValueError("orientation matrices must have shape (3, 3)")
    return (after @ before.T) @ follower


def matrix_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    value = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    y = math.asin(value)
    if abs(math.cos(y)) > 1e-8:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        z = 0.0
    return tuple(math.degrees(item) for item in (x, y, z))


def align_to_z(vector: Sequence[float]) -> np.ndarray:
    source = np.asarray(vector, dtype=float)
    source /= np.linalg.norm(source)
    target = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        return rotation_axis_angle((1.0, 0.0, 0.0), math.pi)
    return rotation_axis_angle(np.cross(source, target), math.acos(dot))


def base_orientation(crystal: CrystalStructure, hkl: Sequence[int]) -> np.ndarray:
    pole = crystal.reciprocal_vector(hkl)
    pole /= np.linalg.norm(pole)
    alignment = align_to_z(pole)
    candidates = [crystal.reciprocal[:, index] for index in range(3)]
    reference = max(
        candidates,
        key=lambda item: np.linalg.norm(item - np.dot(item, pole) * pole),
    )
    reference = reference - np.dot(reference, pole) * pole
    transformed = alignment @ reference
    azimuth = math.atan2(transformed[1], transformed[0])
    return rotation_z(-math.degrees(azimuth)) @ alignment


def rotation_between(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    first_array /= np.linalg.norm(first_array)
    second_array /= np.linalg.norm(second_array)
    dot = float(np.clip(np.dot(first_array, second_array), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(first_array[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        return rotation_axis_angle(np.cross(first_array, fallback), math.pi)
    return rotation_axis_angle(np.cross(first_array, second_array), math.acos(dot))


def format_hkl(hkl: Sequence[int], braces: bool = False) -> str:
    left, right = ("{", "}") if braces else ("(", ")")
    return left + " ".join(str(int(value)) for value in hkl) + right


def in_plane_alignment(current_phi: float, target_phi: float) -> np.ndarray:
    delta = (target_phi - current_phi + 180.0) % 360.0 - 180.0
    return rotation_z(delta)


def pole_plot_coordinates(radius: float, phi_rad: float) -> tuple[float, float]:
    return -radius * math.sin(phi_rad), radius * math.cos(phi_rad)


def pole_display_orientation(orientation: np.ndarray) -> np.ndarray:
    return rotation_z(90.0) @ orientation


def pole_plot_to_sphere(x: float, y: float, projection: str) -> np.ndarray:
    radius = math.hypot(x, y)
    if radius > 0.999999:
        x /= radius / 0.999999
        y /= radius / 0.999999
        radius = 0.999999
    if projection_code(projection) == "equal_area":
        z = 1.0 - radius * radius
        factor = math.sqrt(max(0.0, 2.0 - radius * radius))
        return np.array([y * factor, -x * factor, z])
    denominator = 1.0 + radius * radius
    return np.array(
        [
            2.0 * y / denominator,
            -2.0 * x / denominator,
            (1.0 - radius * radius) / denominator,
        ]
    )


def pole_display_position(
    point: PolePoint,
    boundary_radius: float = 0.985,
) -> tuple[float, float]:
    radius = math.hypot(point.x, point.y)
    if radius <= boundary_radius or radius <= 1e-14:
        return point.x, point.y
    scale = boundary_radius / radius
    return point.x * scale, point.y * scale


def _friedel_representative(hkl: Sequence[int]) -> bool:
    for value in hkl:
        if value:
            return value > 0
    return False


def available_reflections(
    crystal: CrystalStructure,
    d_lower: float,
    d_upper: float,
    wavelength: float,
) -> list[PoleReflection]:
    if d_lower <= 0 or d_upper <= 0:
        raise XRDDataError("pole_d_positive")
    if d_lower > d_upper:
        raise XRDDataError("pole_d_order")
    if wavelength <= 0:
        raise XRDDataError("pole_wavelength_positive")
    inverse_reciprocal = np.linalg.inv(crystal.reciprocal)
    bounds = [
        max(1, int(math.ceil(np.linalg.norm(row) / d_lower + 1e-9)))
        for row in inverse_reciprocal
    ]
    candidates = (2 * bounds[0] + 1) * (2 * bounds[1] + 1) * (2 * bounds[2] + 1)
    if candidates > 2_000_000:
        raise XRDDataError("pole_reflection_limit")
    result: list[PoleReflection] = []
    tolerance = 1e-10
    for h in range(-bounds[0], bounds[0] + 1):
        for k in range(-bounds[1], bounds[1] + 1):
            for l in range(-bounds[2], bounds[2] + 1):
                hkl = (h, k, l)
                if not _friedel_representative(hkl):
                    continue
                d_value = crystal.d_spacing(hkl)
                if d_value < d_lower - tolerance or d_value > d_upper + tolerance:
                    continue
                if crystal.is_systematically_absent(hkl):
                    continue
                two_theta = crystal.two_theta(hkl, wavelength)
                if two_theta is not None:
                    result.append(PoleReflection(hkl, d_value, two_theta))
    result.sort(key=lambda item: (-item.d_spacing, item.hkl))
    return result


def project_reflections(
    crystal: CrystalStructure,
    reflections: Sequence[PoleReflection],
    orientation: np.ndarray,
    projection: str,
) -> list[PolePoint]:
    points: list[PolePoint] = []

    def append_point(
        reflection: PoleReflection,
        direction: np.ndarray,
        label: np.ndarray,
    ) -> None:
        z = float(np.clip(direction[2], 0.0, 1.0))
        chi_rad = math.acos(z)
        phi_rad = 0.0 if chi_rad < 1e-10 else math.atan2(direction[1], direction[0])
        radius = (
            math.sqrt(2.0) * math.sin(chi_rad / 2.0)
            if projection_code(projection) == "equal_area"
            else math.tan(chi_rad / 2.0)
        )
        plot_x, plot_y = pole_plot_coordinates(radius, phi_rad)
        points.append(
            PolePoint(
                tuple(int(value) for value in label),
                reflection.d_spacing,
                reflection.two_theta,
                direction.copy(),
                math.degrees(chi_rad),
                math.degrees(phi_rad) % 360.0,
                plot_x,
                plot_y,
            )
        )

    for reflection in reflections:
        direction = orientation @ crystal.reciprocal_vector(reflection.hkl)
        direction /= np.linalg.norm(direction)
        label = np.asarray(reflection.hkl, dtype=int)
        if direction[2] < -1e-10:
            direction = -direction
            label = -label
        elif abs(direction[2]) <= 1e-10:
            direction = direction.copy()
            direction[2] = 0.0
        append_point(reflection, direction, label)
        if abs(direction[2]) <= 1e-10:
            append_point(reflection, -direction, -label)
    points.sort(
        key=lambda item: (
            round(item.chi, 8),
            round(item.phi, 8),
            -item.d_spacing,
            item.hkl,
        )
    )
    return points


def group_coincident_poles(points: Sequence[PolePoint]) -> list[list[PolePoint]]:
    groups: dict[tuple[int, int], list[PolePoint]] = {}
    for point in points:
        key = (round(point.x * 1e8), round(point.y * 1e8))
        groups.setdefault(key, []).append(point)
    result = list(groups.values())
    for group in result:
        group.sort(key=lambda item: (-item.d_spacing, item.hkl))
    result.sort(key=lambda group: (round(group[0].chi, 8), round(group[0].phi, 8)))
    return result


def marker_sizes_by_d(
    d_values: Sequence[float],
    minimum: float = 22.0,
    maximum: float = 250.0,
) -> np.ndarray:
    values = np.asarray(d_values, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
        return np.full(values.shape, 58.0, dtype=float)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return minimum + (maximum - minimum) * np.square(normalized)


def place_label_boxes(
    anchors: Sequence[tuple[float, float]],
    box_sizes: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    *,
    protected_boxes: Iterable[tuple[float, float, float, float]] = (),
    required_indices: Iterable[int] = (),
    anchor_clearances: Sequence[float] | None = None,
    candidate_gaps: Sequence[float] = (6.0, 16.0, 28.0, 44.0, 64.0),
) -> list[tuple[float, float] | None]:
    """Place screen-space labels without overlapping earlier labels.

    The function knows nothing about Matplotlib or a GUI toolkit.  Anchors,
    label sizes and bounds use the same arbitrary screen unit (normally
    pixels).  Results are lower-left label corners in input order; callers
    should put important labels first and may mark labels that must remain
    visible with ``required_indices``.
    """

    if len(anchors) != len(box_sizes):
        raise ValueError("anchors and box_sizes must have equal lengths")
    if anchor_clearances is not None and len(anchor_clearances) != len(anchors):
        raise ValueError("anchor_clearances and anchors must have equal lengths")
    gaps = tuple(max(0.0, float(value)) for value in candidate_gaps)
    if not gaps:
        raise ValueError("candidate_gaps must not be empty")
    x_min, y_min, x_max, y_max = (float(value) for value in bounds)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("label bounds must have positive width and height")

    required = set(required_indices)
    cell_size = 32.0
    occupied: list[tuple[float, float, float, float]] = []
    cells: dict[tuple[int, int], list[int]] = {}

    def cell_keys(box: tuple[float, float, float, float]):
        left, bottom, right, top = box
        first_x = math.floor(left / cell_size)
        last_x = math.floor(right / cell_size)
        first_y = math.floor(bottom / cell_size)
        last_y = math.floor(top / cell_size)
        for cell_x in range(first_x, last_x + 1):
            for cell_y in range(first_y, last_y + 1):
                yield cell_x, cell_y

    def add_box(box: tuple[float, float, float, float]) -> None:
        index = len(occupied)
        occupied.append(box)
        for key in cell_keys(box):
            cells.setdefault(key, []).append(index)

    def overlaps(box: tuple[float, float, float, float]) -> bool:
        possible: set[int] = set()
        for key in cell_keys(box):
            possible.update(cells.get(key, ()))
        left, bottom, right, top = box
        return any(
            left < occupied[index][2]
            and right > occupied[index][0]
            and bottom < occupied[index][3]
            and top > occupied[index][1]
            for index in possible
        )

    for box in protected_boxes:
        add_box(tuple(float(value) for value in box))

    centre_x = (x_min + x_max) / 2.0
    centre_y = (y_min + y_max) / 2.0
    results: list[tuple[float, float] | None] = []
    for item_index, ((anchor_x, anchor_y), (width, height)) in enumerate(
        zip(anchors, box_sizes)
    ):
        anchor_x = float(anchor_x)
        anchor_y = float(anchor_y)
        width = max(1.0, float(width))
        height = max(1.0, float(height))
        clearance = (
            0.0
            if anchor_clearances is None
            else max(0.0, float(anchor_clearances[item_index]))
        )
        radial_x = anchor_x - centre_x
        radial_y = anchor_y - centre_y
        radial_norm = math.hypot(radial_x, radial_y)
        if radial_norm < 1e-9:
            radial_x, radial_y = 1.0, 1.0
            radial_norm = math.sqrt(2.0)
        radial_x /= radial_norm
        radial_y /= radial_norm

        directions = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ]
        directions.sort(
            key=lambda direction: -(
                direction[0] * radial_x + direction[1] * radial_y
            )
            / math.hypot(*direction)
        )
        candidates: list[tuple[float, float]] = []
        for gap in gaps:
            offset = clearance + gap
            for direction_x, direction_y in directions:
                if direction_x > 0:
                    left = anchor_x + offset
                elif direction_x < 0:
                    left = anchor_x - offset - width
                else:
                    left = anchor_x - width / 2.0
                if direction_y > 0:
                    bottom = anchor_y + offset
                elif direction_y < 0:
                    bottom = anchor_y - offset - height
                else:
                    bottom = anchor_y - height / 2.0
                candidates.append((left, bottom))

        chosen: tuple[float, float] | None = None
        fallback: tuple[float, float] | None = None
        for left, bottom in candidates:
            box = (left, bottom, left + width, bottom + height)
            inside = (
                left >= x_min
                and bottom >= y_min
                and box[2] <= x_max
                and box[3] <= y_max
            )
            if not inside:
                continue
            if fallback is None:
                fallback = (left, bottom)
            if overlaps(box):
                continue
            chosen = (left, bottom)
            add_box(box)
            break
        if chosen is None and item_index in required and fallback is not None:
            chosen = fallback
            add_box((fallback[0], fallback[1], fallback[0] + width, fallback[1] + height))
        results.append(chosen)
    return results


def calculated_intensity_by_spacing(
    diffraction_structure: DiffractionStructure,
    radiations: list[tuple[str, float, float]],
    factors: ScatteringFactors,
) -> dict[float, float]:
    if diffraction_structure is None or diffraction_structure.cell_only:
        return {}
    rows = calculate_reflections(
        diffraction_structure,
        factors,
        radiations,
        min_two_theta=0.01,
        max_two_theta=179.9,
        min_intensity=0.0,
    )
    totals: dict[float, float] = {}
    for row in rows:
        if row.intensity is None:
            continue
        key = round(1.0 / (row.d * row.d), 8)
        totals[key] = totals.get(key, 0.0) + row.intensity
    maximum = max(totals.values(), default=0.0)
    if maximum > 0:
        totals = {key: 100.0 * value / maximum for key, value in totals.items()}
    return totals


# Historical public name retained for compatibility.
Reflection = PoleReflection


__all__ = [
    "Reflection",
    "align_to_z",
    "available_reflections",
    "base_orientation",
    "calculated_intensity_by_spacing",
    "euler_matrix",
    "follow_orientation_change",
    "format_hkl",
    "group_coincident_poles",
    "in_plane_alignment",
    "marker_sizes_by_d",
    "matrix_to_euler",
    "pole_display_orientation",
    "pole_display_position",
    "pole_plot_coordinates",
    "pole_plot_to_sphere",
    "project_reflections",
    "projection_code",
    "rotation_axis_angle",
    "rotation_between",
    "rotation_x",
    "rotation_y",
    "rotation_z",
]
