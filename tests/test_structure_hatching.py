"""Geometric regressions for the hatching supplied by unit-cell-gui."""

from __future__ import annotations

import unittest

import numpy as np
from unit_cell_gui import (
    face_hatch_segments,
    propagated_hatch_directions,
    quadrilateral_hatch_line_count,
)

from xrd_workbench.services.pole_figure import rotation_x, rotation_y
from xrd_workbench.services.structure_scene import (
    convex_polyhedron_faces,
)


def on_perimeter(point, polygon, tolerance=1e-8):
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        fraction = np.dot(point - first, edge) / np.dot(edge, edge)
        if -tolerance <= fraction <= 1 + tolerance:
            if np.linalg.norm(point - first - fraction * edge) < tolerance:
                return True
    return False


class StructureHatchingTests(unittest.TestCase):
    def test_quadrilateral_strokes_are_complete_parallel_and_exactly_counted(self):
        shapes = [
            np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float),
            np.array([(-2, -1, 0), (1, -1, 0), (1.5, 2, 0), (-1, 1, 0)], float),
        ]
        rotation = rotation_y(28) @ rotation_x(17)
        for shape in shapes:
            points = shape @ rotation.T
            directions = propagated_hatch_directions(((0, 1, 2, 3),), points, (0,))
            direction = directions[0]
            apex = int(np.argmax(points[:, 2]))
            diagonal = points[(apex + 1) % 4] - points[(apex - 1) % 4]
            self.assertLess(np.linalg.norm(np.cross(direction, diagonal)), 1e-10)
            for base_count in (0, 4, 12):
                segments = face_hatch_segments(points, direction, 2 * base_count + 1)
                self.assertEqual(len(segments), 2 * base_count + 1)
                for first, second in segments:
                    self.assertTrue(on_perimeter(first, points))
                    self.assertTrue(on_perimeter(second, points))
                    self.assertLess(np.linalg.norm(np.cross(second - first, direction)), 1e-9)

    def test_far_triangle_direction_is_tangent_and_preserves_fold_angle(self):
        faces = ((0, 1, 2), (1, 3, 2), (2, 3, 4))
        vertices = np.array([(0, 1, 3), (-1, 0, 2), (1, 0, 2), (0, -1, 0), (-0.3, -2, -3)], float)
        directions = propagated_hatch_directions(faces, vertices, (0, 1, 2))
        for index, face in enumerate(faces):
            points = vertices[list(face)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal /= np.linalg.norm(normal)
            self.assertAlmostEqual(float(np.dot(directions[index], normal)), 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(directions[index])), 1.0)
        for parent, child in ((0, 1), (1, 2)):
            first, second = sorted(set(faces[parent]).intersection(faces[child]))
            edge = vertices[second] - vertices[first]
            edge /= np.linalg.norm(edge)
            self.assertAlmostEqual(float(directions[parent] @ edge), float(directions[child] @ edge))
        self.assertGreater(abs(float(directions[2][1])), 0.05)

    def test_direction_from_quad_is_transported_not_replaced_by_shared_edge(self):
        faces = ((3, 2, 1, 0), (2, 4, 1))
        vertices = np.array([(-1, 1, 2), (1, 1, 2), (1, -1, 2), (-1, -1, 2), (2, 0, 0)], float)
        vertices = vertices @ (rotation_y(10) @ rotation_x(10)).T
        directions = propagated_hatch_directions(faces, vertices, (0, 1))
        edge = vertices[2] - vertices[1]
        edge /= np.linalg.norm(edge)
        self.assertAlmostEqual(float(directions[0] @ edge), float(directions[1] @ edge))
        self.assertGreater(np.linalg.norm(np.cross(directions[1], edge)), 0.5)
        points = vertices[list(faces[1])]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        self.assertAlmostEqual(float(directions[1] @ normal), 0.0)

    def test_front_triangles_keep_separate_families_around_nearest_apex(self):
        vertices = np.array([(-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0), (0,0,2)], float)
        faces = convex_polyhedron_faces(vertices)
        sides = [index for index, face in enumerate(faces) if len(face) == 3]
        for angles in ((0, 0, 0), (17, 9, 0)):
            camera = vertices @ (rotation_y(angles[1]) @ rotation_x(angles[0])).T
            directions = propagated_hatch_directions(faces, camera, sides)
            for index in sides:
                first, second = [vertex for vertex in faces[index] if vertex != 4]
                edge = camera[second] - camera[first]
                self.assertLess(np.linalg.norm(np.cross(directions[index], edge)), 1e-9)

    def test_frontal_face_is_blank_and_neighbours_use_its_real_edges(self):
        vertices = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 2)], float)
        faces = convex_polyhedron_faces(vertices)
        blank = next(index for index, face in enumerate(faces) if len(face) == 4)
        directions = propagated_hatch_directions(faces, vertices, range(len(faces)), (blank,))
        self.assertNotIn(blank, directions)
        for index, direction in directions.items():
            first, second = sorted(set(faces[index]).intersection(faces[blank]))
            edge = vertices[second] - vertices[first]
            self.assertLess(np.linalg.norm(np.cross(direction, edge)), 1e-9)

    def test_seed_triangle_preserves_a9_opposite_edge_spacing(self):
        vertices = np.array([(0, 1, 2), (-1, -1, 0), (2, -1, 0)], float)
        direction = propagated_hatch_directions(((0, 1, 2),), vertices, (0,))[0]
        actual = face_hatch_segments(vertices, direction, 4)
        expected = [
            (vertices[0] + t * (vertices[1] - vertices[0]), vertices[0] + t * (vertices[2] - vertices[0]))
            for t in (0.2, 0.4, 0.6, 0.8)
        ]
        for first, second in expected:
            self.assertTrue(any(
                (np.allclose(first, a) and np.allclose(second, b))
                or (np.allclose(first, b) and np.allclose(second, a))
                for a, b in actual
            ))


if __name__ == "__main__":
    unittest.main()
