"""Translate calculated XRD Combine geometry into unit-cell-gui input."""

from __future__ import annotations

import numpy as np
from unit_cell_gui import (
    Atom,
    AtomComponent,
    Polyhedron,
    Scene,
)

from ..atom_styles import atom_ball_radius, atom_colour
from ..services.pole_figure import pole_display_orientation
from ..services.structure_scene import StructureSceneData


def _component(component) -> AtomComponent:
    """Attach host-owned atom styles to one resolved site component."""

    return AtomComponent(
        key=component.key,
        label=component.label,
        element=component.element,
        occupancy=component.occupancy,
        colour=atom_colour(component.element),
        radius=atom_ball_radius(component.element),
    )


def unit_cell_scene(source: StructureSceneData) -> Scene:
    """Return the complete, immutable scene consumed by unit-cell-gui."""

    atoms = tuple(
        Atom(
            site_key=atom.site_key,
            site_label=atom.site_label,
            components=tuple(_component(component) for component in atom.components),
            external=atom.external,
        )
        for atom in source.atoms
    )
    polyhedra = tuple(
        Polyhedron(
            center=polyhedron.center,
            vertices=polyhedron.vertices,
            faces=polyhedron.faces,
            site_key=polyhedron.site_key,
            site_label=polyhedron.site_label,
            components=tuple(
                _component(component) for component in polyhedron.components
            ),
        )
        for polyhedron in source.polyhedra
    )
    return Scene(
        atoms=atoms,
        atom_centers=source.atom_centers,
        bonds=source.bonds,
        cell_vertices=source.cell_vertices,
        cell_edges=source.cell_edges,
        basis_vectors=source.crystal.direct,
        polyhedra=polyhedra,
        base_radius=source.base_radius,
        elements=source.elements,
    )


def user_rotation_from_display(
    display_orientation,
    base_rotation,
):
    """Convert the renderer's complete matrix back to host rotation state."""

    display_basis = pole_display_orientation(np.eye(3))
    world_orientation = display_basis.T @ np.asarray(
        display_orientation,
        dtype=float,
    ).reshape(3, 3)
    return world_orientation @ np.asarray(base_rotation, dtype=float).reshape(3, 3).T


__all__ = ["unit_cell_scene", "user_rotation_from_display"]
