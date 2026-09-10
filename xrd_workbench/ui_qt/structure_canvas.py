"""Fast native Qt renderer for crystal structures and coordination polyhedra."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QPainter,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..atom_styles import atom_ball_radius, atom_colour
from ..localization import tr
from ..services.structure_scene import (
    CoordinationPolyhedron,
    StructureSceneData,
    hatch_division_count,
)


def _qpoint_array(point: QPointF) -> np.ndarray:
    return np.array([point.x(), point.y()], dtype=float)


def _line_direction(first: QPointF, second: QPointF) -> np.ndarray:
    direction = _qpoint_array(second) - _qpoint_array(first)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    direction /= length
    if direction[0] < 0.0 or (
        abs(float(direction[0])) < 1e-12 and direction[1] < 0.0
    ):
        direction = -direction
    return direction


def _clip_line_to_polygon(
    polygon: list[QPointF],
    direction: np.ndarray,
    line_constant: float,
    eps: float = 1e-8,
) -> tuple[QPointF, QPointF] | None:
    """Clip an infinite parallel line against one convex screen polygon."""

    normal = np.array([-direction[1], direction[0]], dtype=float)
    points: list[np.ndarray] = []
    vertices = [_qpoint_array(point) for point in polygon]
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        first_side = float(np.dot(first, normal) - line_constant)
        second_side = float(np.dot(second, normal) - line_constant)
        if abs(first_side) < eps:
            points.append(first.copy())
        if first_side * second_side < -(eps * eps):
            fraction = first_side / (first_side - second_side)
            points.append(first + fraction * (second - first))

    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) < 1e-6 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        first, second = max(
            (
                (a, b)
                for position, a in enumerate(unique)
                for b in unique[position + 1 :]
            ),
            key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])),
        )
    else:
        first, second = unique
    return QPointF(float(first[0]), float(first[1])), QPointF(
        float(second[0]), float(second[1])
    )


def _face_adjacency(
    faces: tuple[tuple[int, ...], ...],
) -> tuple[dict[int, set[int]], dict[tuple[int, int], list[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for position, first in enumerate(face):
            second = face[(position + 1) % len(face)]
            edge = tuple(sorted((int(first), int(second))))
            edge_faces.setdefault(edge, []).append(face_index)
    adjacency = {index: set() for index in range(len(faces))}
    for owners in edge_faces.values():
        for first in owners:
            adjacency[first].update(second for second in owners if second != first)
    return adjacency, edge_faces


def _longest_edge(face: tuple[int, ...], projected: list[QPointF]) -> tuple[int, int]:
    edges = [
        (int(face[index]), int(face[(index + 1) % len(face)]))
        for index in range(len(face))
    ]
    return max(
        edges,
        key=lambda edge: float(
            np.linalg.norm(_qpoint_array(projected[edge[0]]) - _qpoint_array(projected[edge[1]]))
        ),
    )


def _opposite_edge(
    face: tuple[int, ...],
    apex: int,
    projected: list[QPointF],
) -> tuple[int, int]:
    candidates = []
    for position, first in enumerate(face):
        second = face[(position + 1) % len(face)]
        if apex not in {first, second}:
            candidates.append((int(first), int(second)))
    if not candidates:
        return _longest_edge(face, projected)
    apex_point = _qpoint_array(projected[apex])
    return max(
        candidates,
        key=lambda edge: float(
            np.linalg.norm(
                0.5
                * (_qpoint_array(projected[edge[0]]) + _qpoint_array(projected[edge[1]]))
                - apex_point
            )
        ),
    )


def _is_octahedron(polyhedron: CoordinationPolyhedron) -> bool:
    return (
        polyhedron.coordination_number == 6
        and len(polyhedron.faces) == 8
        and all(len(face) == 3 for face in polyhedron.faces)
    )


def _octahedron_opposite_vertex(
    polyhedron: CoordinationPolyhedron,
    apex: int,
) -> int | None:
    neighbours: set[int] = set()
    for face in polyhedron.faces:
        if apex in face:
            neighbours.update(int(vertex) for vertex in face if vertex != apex)
    candidates = set(range(polyhedron.coordination_number)) - neighbours - {apex}
    return next(iter(candidates)) if len(candidates) == 1 else None


def _octahedron_face_families(
    polyhedron: CoordinationPolyhedron,
    visible_faces: list[int],
    anchor: int,
) -> tuple[set[int], dict[int, int], dict[int, tuple[int, int]]]:
    """Reproduce the original three continuous families around a blank face."""

    visible = set(visible_faces)
    adjacency, edge_faces = _face_adjacency(polyhedron.faces)
    blank = {anchor}
    families: dict[int, int] = {}
    references: dict[int, tuple[int, int]] = {}
    queue: deque[int] = deque()

    anchor_face = polyhedron.faces[anchor]
    for position, first in enumerate(anchor_face):
        second = anchor_face[(position + 1) % len(anchor_face)]
        edge = tuple(sorted((int(first), int(second))))
        for neighbour in edge_faces.get(edge, []):
            if neighbour == anchor or neighbour not in visible or neighbour in families:
                continue
            family = len(references)
            references[family] = (int(first), int(second))
            families[neighbour] = family
            queue.append(neighbour)

    while queue:
        face_index = queue.popleft()
        family = families[face_index]
        for neighbour in adjacency[face_index]:
            if neighbour not in visible or neighbour in blank or neighbour in families:
                continue
            families[neighbour] = family
            queue.append(neighbour)
    return blank, families, references


def _single_face_hatches(
    face: tuple[int, ...],
    projected: list[QPointF],
    reference_edge: tuple[int, int],
    divisions: int,
) -> list[tuple[QPointF, QPointF]]:
    """Fill one face with equally spaced lines parallel to a real face edge."""

    if len(face) == 3:
        edge_vertices = {int(reference_edge[0]), int(reference_edge[1])}
        apex_candidates = [
            int(vertex) for vertex in face if int(vertex) not in edge_vertices
        ]
        if len(apex_candidates) == 1:
            apex = projected[apex_candidates[0]]
            first = projected[int(reference_edge[0])]
            second = projected[int(reference_edge[1])]
            return [
                (
                    QPointF(
                        apex.x() + fraction * (first.x() - apex.x()),
                        apex.y() + fraction * (first.y() - apex.y()),
                    ),
                    QPointF(
                        apex.x() + fraction * (second.x() - apex.x()),
                        apex.y() + fraction * (second.y() - apex.y()),
                    ),
                )
                for step in range(1, divisions + 1)
                for fraction in (step / (divisions + 1.0),)
            ]

    first, second = (projected[index] for index in reference_edge)
    direction = _line_direction(first, second)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    polygon = [projected[index] for index in face]
    values = np.asarray([float(np.dot(_qpoint_array(point), normal)) for point in polygon])
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    if span <= 1e-9:
        return []
    constants = [
        minimum + step * span / (divisions + 1.0)
        for step in range(1, divisions + 1)
    ]
    return [
        segment
        for constant in constants
        if (segment := _clip_line_to_polygon(polygon, direction, constant)) is not None
    ]


def _polyhedron_hatches(
    polyhedron: CoordinationPolyhedron,
    vertices_camera: np.ndarray,
    projected: list[QPointF],
    visible_faces: list[int],
    normals: dict[int, np.ndarray],
    divisions: int,
) -> dict[int, list[tuple[QPointF, QPointF]]]:
    result = {face_index: [] for face_index in visible_faces}
    if not visible_faces:
        return result

    anchor = max(visible_faces, key=lambda index: float(normals[index][2]))
    face_mode = float(normals[anchor][2]) >= 0.999
    octahedron = _is_octahedron(polyhedron)

    if octahedron and face_mode:
        blank, families, references = _octahedron_face_families(
            polyhedron,
            visible_faces,
            anchor,
        )
        by_family: dict[int, list[int]] = {}
        for face_index, family in families.items():
            by_family.setdefault(family, []).append(face_index)
        for family, face_indices in by_family.items():
            edge = references[family]
            first, second = projected[edge[0]], projected[edge[1]]
            direction = _line_direction(first, second)
            normal = np.array([-direction[1], direction[0]], dtype=float)
            points = np.asarray(
                [
                    _qpoint_array(projected[index])
                    for face_index in face_indices
                    for index in polyhedron.faces[face_index]
                ],
                dtype=float,
            )
            values = points @ normal
            minimum, maximum = float(values.min()), float(values.max())
            span = maximum - minimum
            if span <= 1e-9:
                continue
            reference = 0.5 * (
                float(np.dot(_qpoint_array(first), normal))
                + float(np.dot(_qpoint_array(second), normal))
            )
            step = span / (divisions + 1.0)
            constants = [
                reference + offset * step
                for offset in range(-divisions - 2, divisions + 3)
            ]
            for face_index in face_indices:
                polygon = [projected[index] for index in polyhedron.faces[face_index]]
                for constant in constants:
                    segment = _clip_line_to_polygon(polygon, direction, constant)
                    if segment is not None:
                        result[face_index].append(segment)
        for face_index in blank:
            result[face_index] = []
        return result

    blank = {anchor} if face_mode else set()
    nearest = int(np.argmax(vertices_camera[:, 2])) if octahedron else None
    opposite = (
        _octahedron_opposite_vertex(polyhedron, nearest)
        if nearest is not None
        else None
    )
    anchor_face = polyhedron.faces[anchor]
    anchor_edges = {
        tuple(
            sorted(
                (
                    int(first),
                    int(anchor_face[(position + 1) % len(anchor_face)]),
                )
            )
        )
        for position, first in enumerate(anchor_face)
    }
    for face_index in visible_faces:
        if face_index in blank:
            continue
        face = polyhedron.faces[face_index]
        reference_edge = None
        if face_mode and not octahedron:
            for position, first in enumerate(face):
                second = face[(position + 1) % len(face)]
                if tuple(sorted((int(first), int(second)))) in anchor_edges:
                    reference_edge = (int(first), int(second))
                    break
        if reference_edge is None:
            if nearest is not None and nearest in face:
                apex = nearest
            elif opposite is not None and opposite in face:
                apex = opposite
            else:
                apex = max(face, key=lambda vertex: float(vertices_camera[vertex, 2]))
            reference_edge = _opposite_edge(face, int(apex), projected)
        result[face_index] = _single_face_hatches(
            face,
            projected,
            reference_edge,
            divisions,
        )
    return result


def _blend_components(components) -> QColor:
    weighted = np.zeros(3, dtype=float)
    total = 0.0
    for component in components:
        colour = QColor(atom_colour(component.element))
        weight = max(0.0, float(component.occupancy))
        weighted += weight * np.asarray((colour.red(), colour.green(), colour.blue()))
        total += weight
    if total <= 1e-12:
        return QColor("#b0b0b0")
    red, green, blue = np.clip(np.rint(weighted / total), 0, 255).astype(int)
    return QColor(int(red), int(green), int(blue))


def _engraving_colour(element: str) -> QColor:
    if element == "O":
        return QColor(248, 248, 248)
    colour = QColor(atom_colour(element))
    grey = int(round(0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()))
    grey = max(55, min(220, grey))
    return QColor(grey, grey, grey)


class CrystalCanvas(QWidget):
    """Native Qt viewport; crystal geometry is cached outside paintEvent."""

    rotation_dragged = Signal(float, float)
    rotation_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene: StructureSceneData | None = None
        self.orientation = np.eye(3)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_atoms = True
        self.show_external_atoms = True
        self.show_bonds = True
        self.show_cell = True
        self.show_basis = True
        self.show_polyhedra = True
        self.engraving = False
        self.hatching = True
        self.hatch_density = 22
        self.hatch_grip = 50
        self.polyhedron_elements: set[str] = set()
        self._last_mouse: QPointF | None = None
        self._drag_button = None
        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(760, 620)

    def set_scene(self, scene: StructureSceneData | None) -> None:
        self.scene = scene
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def set_orientation(self, orientation: np.ndarray) -> None:
        self.orientation = np.asarray(orientation, dtype=float).reshape(3, 3).copy()
        self.update()

    def zoom_by(self, steps: float) -> None:
        self.zoom = float(np.clip(self.zoom * (1.12 ** float(steps)), 0.25, 4.5))
        self.update()

    def _project(self, point: np.ndarray) -> tuple[QPointF, float]:
        width, height = max(1, self.width()), max(1, self.height())
        radius = self.scene.base_radius if self.scene is not None else 1.0
        scale = 0.88 * min(width, height) / (2.0 * radius) * self.zoom
        return (
            QPointF(
                width * 0.5 + float(point[0]) * scale + self.pan_x,
                height * 0.5 - float(point[1]) * scale + self.pan_y,
            ),
            scale,
        )

    def _atom_primitive(self, atom, camera: np.ndarray) -> dict:
        point, scale = self._project(camera)
        return {
            "kind": "atom",
            "depth": float(camera[2]),
            "position": point,
            "radius": max(2.5, atom_ball_radius(atom.element) * scale),
            "element": atom.element,
            "occupancy": max(0.0, min(1.0, float(atom.occupancy))),
        }

    def _external_atom_primitives(self) -> list[dict]:
        if (
            self.scene is None
            or not self.show_atoms
            or not self.show_external_atoms
            or not self.show_polyhedra
            or not self.polyhedron_elements
        ):
            return []
        cell_centre = self.scene.crystal.direct @ np.full(3, 0.5)
        primitives: list[dict] = []
        seen: set[tuple[str, float, float, float]] = set()
        for polyhedron in self.scene.polyhedra:
            if not set(polyhedron.elements).intersection(self.polyhedron_elements):
                continue
            for atom in polyhedron.external_atoms:
                key = (
                    atom.element,
                    *tuple(float(value) for value in np.round(atom.cartesian, 7)),
                )
                if key in seen:
                    continue
                seen.add(key)
                camera = self.orientation @ (atom.cartesian - cell_centre)
                primitives.append(self._atom_primitive(atom, camera))
        return primitives

    def _polyhedron_primitives(self) -> list[dict]:
        if (
            self.scene is None
            or not self.show_polyhedra
            or not self.polyhedron_elements
        ):
            return []
        selected = []
        for polyhedron in self.scene.polyhedra:
            if not set(polyhedron.elements).intersection(self.polyhedron_elements):
                continue
            center_camera = self.orientation @ polyhedron.center
            selected.append((polyhedron, center_camera))
        if not selected:
            return []
        depths = [float(center[2]) for _polyhedron, center in selected]
        minimum, maximum = min(depths), max(depths)
        primitives = []
        for polyhedron, center_camera in selected:
            vertices_camera = np.asarray(
                [self.orientation @ vertex for vertex in polyhedron.vertices], dtype=float
            )
            projected = [self._project(vertex)[0] for vertex in vertices_camera]
            normals: dict[int, np.ndarray] = {}
            visible_faces = []
            for face_index, face in enumerate(polyhedron.faces):
                vertices = vertices_camera[list(face)]
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                length = float(np.linalg.norm(normal))
                if length < 1e-12:
                    continue
                normal /= length
                normals[face_index] = normal
                if float(normal[2]) > 1e-9:
                    visible_faces.append(face_index)
            far = (
                0.5
                if abs(maximum - minimum) < 1e-12
                else (maximum - float(center_camera[2])) / (maximum - minimum)
            )
            hatches: dict[int, list[tuple[QPointF, QPointF]]] = {}
            if self.engraving and self.hatching and visible_faces:
                hatches = _polyhedron_hatches(
                    polyhedron,
                    vertices_camera,
                    projected,
                    visible_faces,
                    normals,
                    hatch_division_count(
                        far,
                        self.hatch_density,
                        self.hatch_grip,
                    ),
                )
            colour = _blend_components(polyhedron.components)
            occupancy = min(
                1.0,
                sum(max(0.0, component.occupancy) for component in polyhedron.components),
            )
            for face_index in visible_faces:
                face = polyhedron.faces[face_index]
                primitives.append(
                    {
                        "kind": "face",
                        "depth": float(vertices_camera[list(face), 2].mean()),
                        "polygon": [projected[index] for index in face],
                        "hatches": hatches.get(face_index, []),
                        "colour": colour,
                        "occupancy": occupancy,
                    }
                )
        return primitives

    def _primitives(self) -> list[dict]:
        if self.scene is None:
            return []
        primitives = self._polyhedron_primitives()
        camera_atoms = self.scene.atom_centers @ self.orientation.T
        if self.show_bonds:
            for first, second in self.scene.bonds:
                endpoints = camera_atoms[[first, second]]
                primitives.append(
                    {
                        "kind": "bond",
                        "depth": float(endpoints[:, 2].mean()),
                        "first": self._project(endpoints[0])[0],
                        "second": self._project(endpoints[1])[0],
                    }
                )
        if self.show_atoms:
            primitives.extend(
                self._atom_primitive(atom, camera)
                for atom, camera in zip(self.scene.atoms, camera_atoms)
            )
            primitives.extend(self._external_atom_primitives())
        primitives.sort(key=lambda primitive: primitive["depth"])
        return primitives

    def _draw_face(self, painter: QPainter, primitive: dict) -> None:
        polygon = QPolygonF(primitive["polygon"])
        if self.engraving:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(polygon)
            hatch_pen = QPen(QColor(30, 30, 30))
            hatch_pen.setWidthF(0.75)
            hatch_pen.setCosmetic(True)
            painter.setPen(hatch_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for first, second in primitive["hatches"]:
                painter.drawLine(first, second)
            edge = QPen(QColor(20, 20, 20))
        else:
            fill = QColor(primitive["colour"])
            fill.setAlpha(int(round(70 + 100 * primitive["occupancy"])))
            painter.setBrush(QBrush(fill))
            edge = QPen(fill.lighter(145))
        edge.setWidthF(1.05)
        edge.setCosmetic(True)
        painter.setPen(edge)
        painter.drawPolygon(polygon)

    def _draw_atom(self, painter: QPainter, primitive: dict) -> None:
        radius = primitive["radius"]
        point = primitive["position"]
        if self.engraving:
            fill = _engraving_colour(primitive["element"])
            fill.setAlpha(int(round(80 + 175 * primitive["occupancy"])))
            painter.setBrush(QBrush(fill))
            pen = QPen(QColor(30, 30, 30))
        else:
            fill = QColor(atom_colour(primitive["element"]))
            alpha = int(round(80 + 175 * primitive["occupancy"]))
            gradient = QRadialGradient(
                QPointF(point.x() - 0.28 * radius, point.y() - 0.28 * radius),
                1.35 * radius,
                QPointF(point.x() - 0.30 * radius, point.y() - 0.30 * radius),
            )
            highlight = fill.lighter(175)
            highlight.setAlpha(alpha)
            middle = QColor(fill)
            middle.setAlpha(alpha)
            shadow = fill.darker(135)
            shadow.setAlpha(alpha)
            gradient.setColorAt(0.0, QColor(255, 255, 255, alpha))
            gradient.setColorAt(0.25, highlight)
            gradient.setColorAt(0.72, middle)
            gradient.setColorAt(1.0, shadow)
            painter.setBrush(QBrush(gradient))
            pen = QPen(fill.darker(150))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(
            QRectF(
                point.x() - radius,
                point.y() - radius,
                2.0 * radius,
                2.0 * radius,
            )
        )

    def _draw_cell(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_cell:
            return
        vertices = self.scene.cell_vertices @ self.orientation.T
        pen = QPen(QColor(75, 75, 75))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for first, second in self.scene.cell_edges:
            painter.drawLine(self._project(vertices[first])[0], self._project(vertices[second])[0])

    def _draw_basis(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_basis:
            return
        directions = self.orientation @ (
            self.scene.crystal.direct
            / np.linalg.norm(self.scene.crystal.direct, axis=0)
        )
        origin = QPointF(56.0, self.height() - 52.0)
        colours = (QColor("#db2828"), QColor("#25a43a"), QColor("#2155dc"))
        for index, (name, colour) in enumerate(zip("abc", colours)):
            x, y = float(directions[0, index]), float(directions[1, index])
            length = math.hypot(x, y)
            pen = QPen(colour)
            pen.setWidthF(2.4)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(colour))
            if length <= 0.04:
                painter.drawEllipse(QRectF(origin.x() - 3, origin.y() - 3, 6, 6))
                painter.drawText(QPointF(origin.x() + 7, origin.y() - 7), name)
                continue
            target = QPointF(origin.x() + 43.0 * x, origin.y() - 43.0 * y)
            painter.drawLine(origin, target)
            vector = np.array([target.x() - origin.x(), target.y() - origin.y()])
            vector /= np.linalg.norm(vector)
            normal = np.array([-vector[1], vector[0]])
            tip = np.array([target.x(), target.y()])
            left = tip - 9.0 * vector + 4.0 * normal
            right = tip - 9.0 * vector - 4.0 * normal
            painter.drawPolygon(
                QPolygonF(
                    [target, QPointF(*left), QPointF(*right)]
                )
            )
            painter.drawText(
                QPointF(target.x() + 7.0 * x, target.y() - 7.0 * y), name
            )
        painter.setBrush(QBrush(QColor("#777777")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(origin.x() - 2.5, origin.y() - 2.5, 5, 5))

    def _draw_legend(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_atoms:
            return
        x, y = 12.0, 18.0
        metrics = QFontMetrics(painter.font())
        for element in self.scene.elements:
            colour = _engraving_colour(element) if self.engraving else QColor(atom_colour(element))
            painter.setPen(QPen(colour.darker(150)))
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(QRectF(x, y - 8.0, 10.0, 10.0))
            painter.setPen(QPen(self.palette().text().color()))
            painter.drawText(QPointF(x + 15.0, y), element)
            x += 22.0 + metrics.horizontalAdvance(element)
            if x > self.width() - 80.0:
                x = 12.0
                y += metrics.height() + 4.0

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            background = QColor(250, 250, 247) if self.engraving else self.palette().base().color()
            painter.fillRect(self.rect(), background)
            if self.scene is None:
                painter.setPen(QPen(self.palette().text().color()))
                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    tr("text.no_structure_is_loaded"),
                )
                return
            for primitive in self._primitives():
                if primitive["kind"] == "face":
                    self._draw_face(painter, primitive)
                elif primitive["kind"] == "bond":
                    pen = QPen(QColor("#8b8b8b"))
                    pen.setWidthF(2.0)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.drawLine(primitive["first"], primitive["second"])
                else:
                    self._draw_atom(painter, primitive)
            self._draw_cell(painter)
            self._draw_basis(painter)
            self._draw_legend(painter)
            painter.setPen(QPen(self.palette().text().color()))
            painter.drawText(
                QRectF(8.0, 7.0, self.width() - 16.0, 24.0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                tr("qt.structure_title", formula=self.scene.crystal.formula),
            )
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if (
            event.button()
            in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
            and self.scene is not None
        ):
            self._last_mouse = event.position()
            self._drag_button = event.button()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        dx = position.x() - self._last_mouse.x()
        dy = position.y() - self._last_mouse.y()
        self._last_mouse = position
        if self._drag_button == Qt.MouseButton.LeftButton:
            self.rotation_dragged.emit(float(dx), float(dy))
        elif self._drag_button == Qt.MouseButton.RightButton:
            self.pan_x += float(dx)
            self.pan_y += float(dy)
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == self._drag_button and self._last_mouse is not None:
            rotated = self._drag_button == Qt.MouseButton.LeftButton
            self._last_mouse = None
            self._drag_button = None
            if rotated:
                self.rotation_finished.emit()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if self.scene is None:
            super().wheelEvent(event)
            return
        self.zoom_by(event.angleDelta().y() / 120.0)
        event.accept()


__all__ = ["CrystalCanvas"]
