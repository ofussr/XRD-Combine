"""Toolkit-independent structure-scene and polyhedron regressions."""

from pathlib import Path
import unittest

import numpy as np
from unit_cell_gui import (
    Scene,
    hatch_division_count,
    occupancy_fractions,
    quadrilateral_hatch_line_count,
)

from xrd_workbench.models.crystal import (
    CifData,
    CrystalAtom,
    CrystalStructure,
    direct_basis,
)
from xrd_workbench.services.structure_scene import (
    SiteComponent,
    boundary_images,
    build_structure_scene,
    convex_polyhedron_faces,
    crystallographic_sites,
)
from xrd_workbench.services.pole_figure import (
    pole_display_orientation,
    rotation_x,
    rotation_y,
)
from xrd_workbench.ui_qt.unit_cell_adapter import (
    unit_cell_scene,
    user_rotation_from_display,
)


def mixed_octahedron(
    centre_fractional: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> CrystalStructure:
    direct = direct_basis(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    centre = np.asarray(centre_fractional, dtype=float)
    atoms = [
        CrystalAtom("Nb1", "Nb", centre.copy(), 0.4),
        CrystalAtom("Ta1", "Ta", centre.copy(), 0.6),
    ]
    displacements = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            displacement = np.zeros(3)
            displacement[axis] = sign * 0.25
            displacements.append(displacement)
    for index, displacement in enumerate(displacements, start=1):
        fractional = np.mod(centre + displacement, 1.0)
        atoms.append(CrystalAtom(f"O{index}", "O", fractional, 1.0))
    return CrystalStructure(
        CifData(Path("mixed.cif"), {}, []),
        4.0,
        4.0,
        4.0,
        90.0,
        90.0,
        90.0,
        direct,
        np.linalg.inv(direct).T,
        [(np.eye(3), np.zeros(3))],
        atoms,
        "P 1",
        "Nb0.4 Ta0.6 O3",
    )


class StructureSceneTests(unittest.TestCase):
    def test_scene_adapter_supplies_only_ready_render_data(self) -> None:
        calculated = build_structure_scene(mixed_octahedron())
        rendered = unit_cell_scene(calculated)

        self.assertIsInstance(rendered, Scene)
        self.assertEqual(len(rendered.atoms), len(calculated.atoms))
        self.assertEqual(len(rendered.polyhedra), len(calculated.polyhedra))
        np.testing.assert_array_equal(rendered.basis_vectors, calculated.crystal.direct)
        self.assertFalse(hasattr(rendered, "crystal"))
        self.assertFalse(rendered.atom_centers.flags.writeable)
        component = rendered.atoms[0].components[0]
        self.assertTrue(component.colour.startswith("#"))
        self.assertGreater(component.radius, 0.0)

    def test_display_orientation_round_trip_preserves_host_rotation(self) -> None:
        base = rotation_y(31.0)
        user = rotation_x(-17.0)
        displayed = pole_display_orientation(user @ base)

        np.testing.assert_allclose(
            user_rotation_from_display(displayed, base),
            user,
            atol=1e-12,
        )

    def test_quadrilateral_hatching_uses_two_halves_and_a_centre_line(self) -> None:
        self.assertEqual(quadrilateral_hatch_line_count(0), 1)
        self.assertEqual(quadrilateral_hatch_line_count(4), 9)
        self.assertEqual(quadrilateral_hatch_line_count(12), 25)

    def test_occupancy_fractions_preserve_vacancies_and_normalise_overfill(self) -> None:
        partial = (
            SiteComponent("Nb1", "Nb", 0.106),
            SiteComponent("Ta1", "Ta", 0.141),
        )
        self.assertEqual(occupancy_fractions(partial), (0.106, 0.141))
        self.assertAlmostEqual(sum(occupancy_fractions(partial)), 0.247)

        overfilled = (
            SiteComponent("Nb1", "Nb", 0.8),
            SiteComponent("Ta1", "Ta", 0.7),
        )
        fractions = occupancy_fractions(overfilled)
        self.assertAlmostEqual(sum(fractions), 1.0)
        self.assertAlmostEqual(fractions[0] / fractions[1], 8.0 / 7.0)

    def test_grip_changes_only_the_depth_contrast(self) -> None:
        self.assertEqual(hatch_division_count(0.0, 22, 0), 8)
        self.assertEqual(hatch_division_count(0.0, 22, 50), 8)
        self.assertEqual(hatch_division_count(0.0, 22, 200), 8)
        self.assertEqual(hatch_division_count(1.0, 22, 0), 8)
        self.assertEqual(hatch_division_count(1.0, 22, 50), 17)
        self.assertEqual(hatch_division_count(1.0, 22, 200), 43)

    def test_boundary_images_cover_faces_edges_and_corners(self) -> None:
        self.assertEqual(len(boundary_images((0.0, 0.3, 0.4))), 2)
        self.assertEqual(len(boundary_images((0.0, 0.0, 0.4))), 4)
        self.assertEqual(len(boundary_images((0.0, 0.0, 0.0))), 8)

    def test_convex_hull_keeps_polygon_faces_without_fake_diagonals(self) -> None:
        cube = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        faces = convex_polyhedron_faces(cube)
        self.assertEqual(len(faces), 6)
        self.assertEqual({len(face) for face in faces}, {4})
        centre = cube.mean(axis=0)
        for face in faces:
            points = cube[list(face)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            self.assertGreater(float(np.dot(normal, points.mean(axis=0) - centre)), 0.0)

        tetrahedron = np.asarray(
            [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
            dtype=float,
        )
        tetrahedron_faces = convex_polyhedron_faces(tetrahedron)
        self.assertEqual(len(tetrahedron_faces), 4)
        self.assertEqual({len(face) for face in tetrahedron_faces}, {3})
        centre = tetrahedron.mean(axis=0)
        for face in tetrahedron_faces:
            points = tetrahedron[list(face)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            self.assertGreater(float(np.dot(normal, points.mean(axis=0) - centre)), 0.0)

    def test_mixed_occupancy_is_one_site_and_one_polyhedron(self) -> None:
        crystal = mixed_octahedron()
        sites = crystallographic_sites(crystal)
        centre = next(site for site in sites if set(site.elements) == {"Nb", "Ta"})
        self.assertEqual(
            {component.element: component.occupancy for component in centre.components},
            {"Nb": 0.4, "Ta": 0.6},
        )

        scene = build_structure_scene(crystal)
        mixed_atoms = [
            atom for atom in scene.atoms if set(atom.elements) == {"Nb", "Ta"}
        ]
        self.assertEqual(len(mixed_atoms), 1)
        self.assertEqual(mixed_atoms[0].site_label, "Nb1/Ta1")
        self.assertEqual(
            {component.key for component in mixed_atoms[0].components},
            {"Nb:Nb1", "Ta:Ta1"},
        )
        mixed = [
            polyhedron
            for polyhedron in scene.polyhedra
            if set(polyhedron.elements) == {"Nb", "Ta"}
        ]
        self.assertEqual(len(mixed), 1)
        self.assertEqual(mixed[0].coordination_number, 6)
        self.assertEqual(len(mixed[0].faces), 8)
        centre = mixed[0].vertices.mean(axis=0)
        for face in mixed[0].faces:
            points = mixed[0].vertices[list(face)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            self.assertGreater(float(np.dot(normal, points.mean(axis=0) - centre)), 0.0)

    def test_scene_uses_the_real_oblique_cell_matrix(self) -> None:
        crystal = mixed_octahedron()
        crystal.alpha = 80.0
        crystal.gamma = 70.0
        crystal.direct = direct_basis(4.0, 4.0, 4.0, 80.0, 90.0, 70.0)
        crystal.reciprocal = np.linalg.inv(crystal.direct).T
        scene = build_structure_scene(crystal)
        edges = [
            np.linalg.norm(scene.cell_vertices[second] - scene.cell_vertices[first])
            for first, second in scene.cell_edges
        ]
        self.assertEqual(len(edges), 12)
        np.testing.assert_allclose(edges, np.full(12, 4.0), atol=1e-10)

    def test_boundary_polyhedra_retain_only_their_external_ligand_images(self) -> None:
        scene = build_structure_scene(mixed_octahedron((0.0, 0.5, 0.5)))
        mixed = [
            polyhedron
            for polyhedron in scene.polyhedra
            if set(polyhedron.elements) == {"Nb", "Ta"}
        ]
        self.assertEqual(len(mixed), 2)
        external = [atom for polyhedron in mixed for atom in polyhedron.external_atoms]
        self.assertEqual({atom.element for atom in external}, {"O"})
        coordinates = {tuple(np.round(atom.fractional, 8)) for atom in external}
        self.assertIn((-0.25, 0.5, 0.5), coordinates)
        self.assertIn((1.25, 0.5, 0.5), coordinates)
        self.assertTrue(
            all(
                np.any(atom.fractional < 0.0) or np.any(atom.fractional > 1.0)
                for atom in external
            )
        )
        scene_external = [atom for atom in scene.atoms if atom.external]
        self.assertEqual(
            {tuple(np.round(atom.fractional, 8)) for atom in scene_external},
            coordinates,
        )
        self.assertTrue(
            any(
                scene.atoms[first].external or scene.atoms[second].external
                for first, second in scene.bonds
            )
        )


if __name__ == "__main__":
    unittest.main()
