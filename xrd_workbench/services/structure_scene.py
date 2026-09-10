"""GUI-independent geometry for the Qt crystal-structure scene."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..atom_styles import atom_ball_radius, covalent_radius
from ..models.crystal import CrystalStructure, DisplayAtom, unit_cell_display_atoms
from .pole_figure import (
    align_to_z,
    pole_display_orientation,
    rotation_x,
    rotation_y,
    rotation_z,
)


CLINOGRAPHIC_HORIZONTAL_DEG = math.degrees(math.atan(1.0 / 3.0))
CLINOGRAPHIC_TILT_DEG = math.degrees(math.atan(1.0 / 6.0))


@dataclass(frozen=True)
class SiteComponent:
    """One chemical component occupying a crystallographic site."""

    element: str
    occupancy: float


@dataclass
class CrystalSite:
    """Symmetry-expanded site; coincident chemical components stay together."""

    fractional: np.ndarray
    components: tuple[SiteComponent, ...]

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(component.element for component in self.components)


@dataclass
class CoordinationPolyhedron:
    """A convex coordination polyhedron around one displayed site image."""

    center: np.ndarray
    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    external_atoms: tuple[DisplayAtom, ...]
    components: tuple[SiteComponent, ...]

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(component.element for component in self.components)

    @property
    def coordination_number(self) -> int:
        return int(len(self.vertices))


@dataclass
class StructureSceneData:
    """Cached world-space geometry consumed by a rendering backend."""

    crystal: CrystalStructure
    atoms: tuple[DisplayAtom, ...]
    atom_centers: np.ndarray
    bonds: np.ndarray
    cell_vertices: np.ndarray
    cell_edges: np.ndarray
    polyhedra: tuple[CoordinationPolyhedron, ...]
    base_radius: float
    elements: tuple[str, ...]


def hatch_division_count(
    far_factor: float,
    density_setting: int,
    depth_setting: int,
) -> int:
    """Return the original density with a separate GRIP depth coefficient."""

    density_norm = (float(density_setting) - 4.0) / (42.0 - 4.0)
    density_norm = max(0.0, min(1.0, density_norm))
    near_count = 4.0 + 8.0 * density_norm
    base_difference = 4.0 + 10.0 * density_norm
    far_count = near_count + base_difference * (float(depth_setting) / 50.0)
    depth = max(0.0, min(1.0, float(far_factor)))
    smooth_depth = depth * depth * (3.0 - 2.0 * depth)
    count = near_count + (far_count - near_count) * smooth_depth
    return max(1, int(round(count)))


def standard_clinographic_orientation(crystal: CrystalStructure) -> np.ndarray:
    """Return the classical mineralogical clinographic orientation."""

    display = pole_display_orientation(np.eye(3))
    up = crystal.direct[:, 2].copy()
    up /= np.linalg.norm(up)
    right = crystal.direct[:, 1] - np.dot(crystal.direct[:, 1], up) * up
    right /= np.linalg.norm(right)
    towards = np.cross(right, up)
    towards /= np.linalg.norm(towards)
    if np.dot(towards, crystal.direct[:, 0]) < 0:
        right = -right
        towards = -towards
    initial_screen = np.vstack((right, up, towards))
    clinographic_screen = (
        rotation_x(CLINOGRAPHIC_TILT_DEG)
        @ rotation_y(-CLINOGRAPHIC_HORIZONTAL_DEG)
        @ initial_screen
    )
    return display.T @ clinographic_screen


def direction_orientation(crystal: CrystalStructure, name: str) -> np.ndarray:
    """Return a stable proper rotation looking along a direct/reciprocal axis."""

    if name == "standard":
        return standard_clinographic_orientation(crystal)
    index = "abc".index(name[0])
    basis = crystal.reciprocal if name.endswith("*") else crystal.direct
    direction = basis[:, index].copy()
    direction /= np.linalg.norm(direction)
    alignment = align_to_z(direction)
    candidates = [
        basis[:, candidate] / np.linalg.norm(basis[:, candidate])
        for candidate in range(3)
        if candidate != index
    ]
    reference = max(
        candidates,
        key=lambda vector: np.linalg.norm(np.cross(direction, vector)),
    )
    projected = alignment @ reference
    return rotation_z(-math.degrees(math.atan2(projected[1], projected[0]))) @ alignment


def screen_drag_rotation(dx: float, dy: float) -> np.ndarray:
    """Map a screen drag to the shared non-reflected crystal frame."""

    display = pole_display_orientation(np.eye(3))
    return display.T @ (rotation_y(0.45 * dx) @ rotation_x(-0.45 * dy)) @ display


def crystallographic_sites(
    crystal: CrystalStructure,
    tolerance: float = 1e-7,
) -> list[CrystalSite]:
    """Group coincident symmetry-expanded atoms into crystallographic sites."""

    grouped: list[tuple[np.ndarray, dict[str, float]]] = []
    for atom in crystal.atoms:
        fractional = np.asarray(atom.fractional, dtype=float)
        target = None
        for position, components in grouped:
            difference = fractional - position
            difference -= np.rint(difference)
            if float(np.linalg.norm(difference)) < tolerance:
                target = components
                break
        if target is None:
            target = {}
            grouped.append((fractional.copy(), target))
        target[atom.element] = target.get(atom.element, 0.0) + float(atom.occupancy)

    result: list[CrystalSite] = []
    for position, components in grouped:
        result.append(
            CrystalSite(
                position,
                tuple(
                    SiteComponent(element, occupancy)
                    for element, occupancy in sorted(components.items())
                ),
            )
        )
    return result


def boundary_images(fractional: Iterable[float], eps: float = 1e-8) -> list[np.ndarray]:
    """Return display-only images for a site on opposite unit-cell boundaries."""

    choices: list[tuple[float, ...]] = []
    for value in np.asarray(tuple(fractional), dtype=float):
        choices.append((0.0, 1.0) if abs(float(value)) < eps else (float(value),))
    return [np.asarray(values, dtype=float) for values in itertools.product(*choices)]


def unit_cell_bonds(atoms: Iterable[DisplayAtom]) -> list[tuple[int, int]]:
    """Estimate display bonds using the same rule as the stable viewer."""

    atoms = list(atoms)
    result: list[tuple[int, int]] = []
    has_oxygen = any(atom.element == "O" for atom in atoms)
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            atom_a, atom_b = atoms[first], atoms[second]
            if atom_a.element == atom_b.element:
                continue
            pair = {atom_a.element, atom_b.element}
            if has_oxygen and "O" not in pair:
                continue
            cutoff = 1.25 * (
                covalent_radius(atom_a.element) + covalent_radius(atom_b.element)
            )
            distance = float(np.linalg.norm(atom_a.cartesian - atom_b.cartesian))
            if 0.35 < distance <= cutoff:
                result.append((first, second))
    return result


def _convex_hull_2d(points: np.ndarray, indices: list[int]) -> list[int]:
    """Return the outer polygon of coplanar points without interior vertices."""

    ordered = sorted(
        ((float(points[index, 0]), float(points[index, 1]), index) for index in indices),
        key=lambda item: (item[0], item[1], item[2]),
    )
    if len(ordered) <= 2:
        return [item[2] for item in ordered]

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-10:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-10:
            upper.pop()
        upper.append(point)
    return [item[2] for item in lower[:-1] + upper[:-1]]


def convex_polyhedron_faces(
    vertices: Iterable[Iterable[float]],
    eps: float = 1e-7,
) -> tuple[tuple[int, ...], ...]:
    """Build outward-oriented polygon faces of a small convex point cloud."""

    points = np.asarray(tuple(vertices), dtype=float).reshape(-1, 3)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
        return ()

    centre = points.mean(axis=0)
    plane_sets: set[frozenset[int]] = set()
    plane_normals: dict[frozenset[int], np.ndarray] = {}
    for first, second, third in itertools.combinations(range(len(points)), 3):
        a, b, c = points[first], points[second], points[third]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < eps:
            continue
        normal /= length
        signed = (points - a) @ normal
        if not (np.all(signed <= eps) or np.all(signed >= -eps)):
            continue
        on_plane = frozenset(
            int(index) for index, value in enumerate(signed) if abs(float(value)) <= eps
        )
        if len(on_plane) < 3:
            continue
        face_centre = points[list(on_plane)].mean(axis=0)
        if float(np.dot(normal, face_centre - centre)) < 0.0:
            normal = -normal
        plane_sets.add(on_plane)
        plane_normals[on_plane] = normal

    faces: list[tuple[int, ...]] = []
    for face_set in plane_sets:
        normal = plane_normals[face_set]
        face_indices = sorted(face_set)
        face_centre = points[face_indices].mean(axis=0)
        first_vector = points[face_indices[0]] - face_centre
        first_vector /= np.linalg.norm(first_vector)
        second_vector = np.cross(normal, first_vector)
        projected = np.column_stack(
            ((points - face_centre) @ first_vector, (points - face_centre) @ second_vector)
        )
        polygon = _convex_hull_2d(projected, face_indices)
        if len(polygon) >= 3:
            polygon_points = points[polygon]
            polygon_normal = np.cross(
                polygon_points[1] - polygon_points[0],
                polygon_points[2] - polygon_points[0],
            )
            if float(np.dot(polygon_normal, face_centre - centre)) < 0.0:
                polygon.reverse()
            faces.append(tuple(polygon))

    def face_key(face: tuple[int, ...]) -> tuple[float, float, float, int]:
        centre_value = points[list(face)].mean(axis=0)
        return (*tuple(float(value) for value in centre_value), len(face))

    return tuple(sorted(faces, key=face_key))


def _pair_cutoff(
    centre: CrystalSite,
    ligand: CrystalSite,
    oxygen_only: bool,
) -> float | None:
    pairs = []
    for central in centre.components:
        if oxygen_only and central.element == "O":
            continue
        for outer in ligand.components:
            if oxygen_only and outer.element != "O":
                continue
            if not oxygen_only and central.element == outer.element:
                continue
            pairs.append(
                1.25
                * (covalent_radius(central.element) + covalent_radius(outer.element))
            )
    return max(pairs) if pairs else None


def _coordination_vertices(
    crystal: CrystalStructure,
    sites: list[CrystalSite],
    centre_site: CrystalSite,
    centre_fractional: np.ndarray,
    oxygen_only: bool,
) -> tuple[np.ndarray, tuple[DisplayAtom, ...]]:
    candidates: list[
        tuple[float, np.ndarray, np.ndarray, tuple[SiteComponent, ...]]
    ] = []
    for ligand in sites:
        cutoff = _pair_cutoff(centre_site, ligand, oxygen_only)
        if cutoff is None:
            continue
        for translation in itertools.product((-1.0, 0.0, 1.0), repeat=3):
            image = ligand.fractional + np.asarray(translation, dtype=float)
            cartesian = crystal.direct @ image
            distance = float(
                np.linalg.norm(crystal.direct @ (image - centre_fractional))
            )
            if 0.35 < distance <= cutoff:
                candidates.append(
                    (distance, cartesian, image.copy(), ligand.components)
                )

    candidates.sort(key=lambda item: item[0])
    unique: list[
        tuple[np.ndarray, np.ndarray, tuple[SiteComponent, ...]]
    ] = []
    for _distance, cartesian, fractional, components in candidates:
        if any(
            float(np.linalg.norm(cartesian - other_cartesian)) < 1e-7
            for other_cartesian, _other_fractional, _other_components in unique
        ):
            continue
        unique.append((cartesian, fractional, components))

    external_atoms: list[DisplayAtom] = []
    for cartesian, fractional, components in unique:
        if not bool(np.any(fractional < -1e-8) or np.any(fractional > 1.0 + 1e-8)):
            continue
        external_atoms.extend(
            DisplayAtom(
                component.element,
                fractional.copy(),
                cartesian.copy(),
                component.occupancy,
            )
            for component in components
        )
    return (
        np.asarray([item[0] for item in unique], dtype=float).reshape(-1, 3),
        tuple(external_atoms),
    )


def coordination_polyhedra(crystal: CrystalStructure) -> list[CoordinationPolyhedron]:
    """Construct convex ligand polyhedra around crystallographic site images."""

    sites = crystallographic_sites(crystal)
    oxygen_only = any("O" in site.elements for site in sites)
    cell_centre = crystal.direct @ np.full(3, 0.5)
    result: list[CoordinationPolyhedron] = []
    for site in sites:
        if oxygen_only and "O" in site.elements:
            continue
        for centre_fractional in boundary_images(site.fractional):
            vertices, external_atoms = _coordination_vertices(
                crystal,
                sites,
                site,
                centre_fractional,
                oxygen_only,
            )
            faces = convex_polyhedron_faces(vertices)
            if not faces:
                continue
            result.append(
                CoordinationPolyhedron(
                    center=crystal.direct @ centre_fractional - cell_centre,
                    vertices=vertices - cell_centre,
                    faces=faces,
                    external_atoms=external_atoms,
                    components=site.components,
                )
            )
    return result


def default_polyhedron_elements(
    polyhedra: Iterable[CoordinationPolyhedron],
) -> set[str]:
    """Choose compact common coordination centres without hiding other choices."""

    coordination: dict[str, list[int]] = {}
    for polyhedron in polyhedra:
        for element in polyhedron.elements:
            coordination.setdefault(element, []).append(polyhedron.coordination_number)
    compact = {
        element
        for element, values in coordination.items()
        if values and float(np.median(values)) <= 8.0
    }
    return compact or set(coordination)


def build_structure_scene(crystal: CrystalStructure) -> StructureSceneData:
    """Build all world-space geometry once for fast interactive repainting."""

    atoms = tuple(unit_cell_display_atoms(crystal))
    cell_centre = crystal.direct @ np.full(3, 0.5)
    atom_centers = np.asarray(
        [atom.cartesian - cell_centre for atom in atoms], dtype=float
    ).reshape(-1, 3)
    bonds = np.asarray(unit_cell_bonds(atoms), dtype=int).reshape(-1, 2)
    fractional_corners = np.asarray(
        list(itertools.product((0.0, 1.0), repeat=3)), dtype=float
    )
    cell_vertices = fractional_corners @ crystal.direct.T - cell_centre
    cell_edges = np.asarray(
        [
            (first, second)
            for first in range(8)
            for second in range(first + 1, 8)
            if np.count_nonzero(fractional_corners[first] != fractional_corners[second])
            == 1
        ],
        dtype=int,
    ).reshape(-1, 2)
    polyhedra = tuple(coordination_polyhedra(crystal))

    extents = [float(np.linalg.norm(point)) for point in cell_vertices]
    extents.extend(
        float(np.linalg.norm(point)) + atom_ball_radius(atom.element)
        for point, atom in zip(atom_centers, atoms)
    )
    extents.extend(
        float(np.linalg.norm(vertex))
        for polyhedron in polyhedra
        for vertex in polyhedron.vertices
    )
    base_radius = max(extents, default=1.0) * 1.08
    return StructureSceneData(
        crystal=crystal,
        atoms=atoms,
        atom_centers=atom_centers,
        bonds=bonds,
        cell_vertices=cell_vertices,
        cell_edges=cell_edges,
        polyhedra=polyhedra,
        base_radius=max(base_radius, 1e-6),
        elements=tuple(sorted({atom.element for atom in atoms})),
    )


__all__ = [
    "CLINOGRAPHIC_HORIZONTAL_DEG",
    "CLINOGRAPHIC_TILT_DEG",
    "CoordinationPolyhedron",
    "CrystalSite",
    "SiteComponent",
    "StructureSceneData",
    "boundary_images",
    "build_structure_scene",
    "convex_polyhedron_faces",
    "coordination_polyhedra",
    "crystallographic_sites",
    "default_polyhedron_elements",
    "direction_orientation",
    "hatch_division_count",
    "screen_drag_rotation",
    "standard_clinographic_orientation",
    "unit_cell_bonds",
]
