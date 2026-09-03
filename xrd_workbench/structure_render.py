"""Cached crystal geometry and persistent artists shared by both structure views."""
from __future__ import annotations

import itertools
import math
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    from .atom_styles import atom_ball_radius, atom_colour
    from .i18n import localised
    from .theoretical_pole import (
        align_to_z, pole_display_orientation, rotation_x, rotation_y, rotation_z,
        unit_cell_bonds, unit_cell_display_atoms,
    )
except ImportError:
    from atom_styles import atom_ball_radius, atom_colour
    from i18n import localised
    from theoretical_pole import (
        align_to_z, pole_display_orientation, rotation_x, rotation_y, rotation_z,
        unit_cell_bonds, unit_cell_display_atoms,
    )


CLINOGRAPHIC_HORIZONTAL_DEG = math.degrees(math.atan(1.0 / 3.0))
CLINOGRAPHIC_TILT_DEG = math.degrees(math.atan(1.0 / 6.0))


def standard_clinographic_orientation(crystal):
    """Return the classical mineralogical clinographic crystal orientation.

    The unrotated screen frame has c upward, the projection of b to the right,
    and the positive side of a towards the observer.  It is then yawed around
    the vertical by atan(1/3) and tilted forward by atan(1/6).
    """

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


def direction_orientation(crystal, name):
    """A repeatable proper rotation: chosen direct/reciprocal vector faces Z."""
    if name == "standard":
        return standard_clinographic_orientation(crystal)
    index = "abc".index(name[0])
    basis = crystal.reciprocal if name.endswith("*") else crystal.direct
    direction = basis[:, index].copy()
    direction /= np.linalg.norm(direction)
    alignment = align_to_z(direction)
    # The secondary axis with the largest projection fixes the roll. Internal
    # +X becomes screen +Y after the shared 90-degree display rotation.
    candidates = [basis[:, i]/np.linalg.norm(basis[:, i]) for i in range(3) if i != index]
    reference = max(candidates, key=lambda v: np.linalg.norm(np.cross(direction, v)))
    projected = alignment @ reference
    return rotation_z(-math.degrees(math.atan2(projected[1], projected[0]))) @ alignment


def screen_drag_rotation(dx, dy):
    display = pole_display_orientation(np.eye(3))
    return display.T @ (rotation_y(.45*dx) @ rotation_x(-.45*dy)) @ display


class BasisIndicator:
    """A direct-basis triad with fixed pixel size, outside data coordinates."""
    def __init__(self, parent):
        self.parent = parent
        self.axis = parent.inset_axes((0, 0, .1, .1), label=f"basis-{id(self)}")
        self.axis.set_axes_locator(self._locate)
        self.axis.set_in_layout(False)
        self.axis.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), aspect="equal")
        self.axis.set_axis_off()
        self.axis.patch.set_alpha(0)
        self.arrows, self.labels, self.dots = [], [], []
        for name, colour in zip("abc", ("#db2828", "#25a43a", "#2155dc")):
            arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=14,
                                    linewidth=2.5, color=colour)
            self.axis.add_patch(arrow)
            self.arrows.append(arrow)
            self.labels.append(self.axis.text(0, 0, name, color=colour, weight="bold",
                                              ha="center", va="center", fontsize=10))
            self.dots.append(self.axis.plot([0], [0], marker="o", color=colour, ms=5)[0])
        self.axis.plot([0], [0], marker="o", color="#777777", ms=4)

    def _locate(self, _axis, _renderer):
        box = self.parent.bbox
        size = min(125.0, .28*box.width, .28*box.height)
        pixels = Bbox.from_bounds(box.x0+8, box.y0+8, size, size)
        return pixels.transformed(self.parent.figure.transFigure.inverted())

    def update(self, direct, orientation, visible=True):
        self.axis.set_visible(visible)
        self.directions = orientation @ (direct / np.linalg.norm(direct, axis=0))
        for i, (arrow, label, dot) in enumerate(zip(self.arrows, self.labels, self.dots)):
            x, y = self.directions[:2, i]
            length = math.hypot(x, y)
            arrow.set_visible(length > .04)
            arrow.set_positions((0, 0), (x, y))
            dot.set_visible(length <= .04)
            # A direction straight towards the observer projects to a dot.
            label.set_position((1.22*x, 1.22*y) if length > .15 else (.14, .18))

    def remove(self):
        if self.axis in self.parent.child_axes:
            self.axis.remove()


class CrystalScene:
    def __init__(self, axis, crystal):
        self.axis, self.crystal = axis, crystal
        axis.clear()
        self.atoms = unit_cell_display_atoms(crystal)
        self.bonds = np.asarray(unit_cell_bonds(self.atoms), dtype=int).reshape(-1, 2)
        center = crystal.direct @ np.full(3, .5)
        self.centers = np.asarray(
            [a.cartesian-center for a in self.atoms], dtype=float
        ).reshape(-1, 3)
        fractional = np.asarray(list(itertools.product((0., 1.), repeat=3)))
        self.cell_vertices = fractional @ crystal.direct.T-center
        self.cell_edges = np.asarray([(a, b) for a in range(8) for b in range(a+1, 8)
                                      if np.count_nonzero(fractional[a] != fractional[b]) == 1])
        self.radii = np.asarray(
            [atom_ball_radius(a.element) for a in self.atoms], dtype=float
        )
        self.colors = np.asarray(
            [
                to_rgba(
                    atom_colour(a.element),
                    max(.25, min(1., a.occupancy)),
                )
                for a in self.atoms
            ],
            dtype=float,
        ).reshape(-1, 4)
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 13), np.linspace(0, np.pi, 9), indexing="ij")
        sphere = np.stack((np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)), axis=-1)
        quads = np.stack((sphere[:-1, :-1], sphere[1:, :-1], sphere[1:, 1:], sphere[:-1, 1:]), axis=2).reshape(-1, 4, 3)
        self.sphere_offsets = self.radii[:, None, None, None]*quads[None, :, :, :]
        normals = quads.mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        light = np.array([-.4, .6, 1.]); light /= np.linalg.norm(light)
        shade = .38+.62*np.maximum(0., normals @ light)
        facecolors = np.repeat(self.colors[:, None, :], len(quads), axis=1)
        facecolors[:, :, :3] *= shade[None, :, None]
        self.spheres = Poly3DCollection([], facecolors=facecolors.reshape(-1, 4),
                                        linewidths=0, antialiased=False, zsort="average")
        self.bond_artist = Line3DCollection([], colors="#8b8b8b", linewidths=2)
        self.cell_artist = Line3DCollection([], colors="#333333", linewidths=1.1)
        for artist in (self.bond_artist, self.cell_artist, self.spheres):
            axis.add_collection(artist, autolim=False)
        self.preview = axis.scatter(*self.centers.T, c=self.colors, s=35, depthshade=True)
        self.preview.set_visible(False)
        atom_extent = (
            float((np.linalg.norm(self.centers, axis=1) + self.radii).max())
            if len(self.centers)
            else 0.0
        )
        self.base_radius = max(
            float(np.linalg.norm(self.cell_vertices, axis=1).max()),
            atom_extent,
        ) * 1.07
        self.zoom = 1.0
        self._apply_zoom()
        axis.set_box_aspect((1., 1., 1.))
        axis.set_proj_type("ortho")
        axis.view_init(elev=90, azim=-90)
        axis.disable_mouse_rotation()
        axis.set_axis_off()
        legend = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                         markerfacecolor=atom_colour(el),
                         markeredgecolor="#555555", label=el)
                  for el in sorted({a.element for a in self.atoms})]
        if legend:
            axis.legend(
                handles=legend,
                loc="upper left",
                frameon=False,
                ncol=min(3, len(legend)),
            )
        self.no_atoms_note = axis.text2D(
            .5,
            .03,
            "",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color="#666666",
        )
        self.basis = BasisIndicator(axis)

    def _apply_zoom(self):
        radius = self.base_radius / self.zoom
        for setter in (self.axis.set_xlim, self.axis.set_ylim, self.axis.set_zlim):
            setter(-radius, radius)

    def zoom_by(self, steps: float) -> float:
        """Zoom the already built scene without rebuilding crystal geometry."""
        self.zoom = float(np.clip(self.zoom * (1.16 ** steps), 0.35, 5.0))
        self._apply_zoom()
        return self.zoom

    def update(self, orientation, *, preview=False, show_basis=True):
        self.orientation = np.array(orientation, copy=True)
        self.rotated_centers = self.centers @ orientation.T
        vertices = self.cell_vertices @ orientation.T
        self.cell_artist.set_segments(vertices[self.cell_edges])
        self.bond_artist.set_segments(self.rotated_centers[self.bonds])
        self.spheres.set_visible(not preview)
        self.preview.set_visible(preview)
        if preview:
            self.preview._offsets3d = tuple(self.rotated_centers.T)
            # Estimate projected ball diameter from the fixed orthographic view.
            span = abs(self.axis.get_xlim()[1]-self.axis.get_xlim()[0])
            diameter = 2*self.radii*self.axis.bbox.width/max(span, 1e-8)*.65
            self.preview.set_sizes((diameter*72/self.axis.figure.dpi)**2)
        else:
            faces = self.sphere_offsets+self.rotated_centers[:, None, None, :]
            self.spheres.set_verts(faces.reshape(-1, 4, 3))
        self.basis.update(self.crystal.direct, orientation, show_basis)
        self.no_atoms_note.set_visible(not self.atoms)
        self.no_atoms_note.set_text(
            localised(
                "No atom positions are available",
                "Aucune position atomique n’est disponible",
                "Координаты атомов не заданы",
            )
        )
        self.axis.set_title(localised(f"Structure {self.crystal.formula}",
                                      f"Structure {self.crystal.formula}",
                                      f"Структура {self.crystal.formula}"), fontsize=11, pad=8)


def render_structure(axis, crystal, orientation, *, preview=False, show_basis=True):
    if axis is None:
        return
    scene = getattr(axis, "_crystal_scene", None)
    if crystal is None:
        if scene is not None:
            scene.basis.remove()
            del axis._crystal_scene
        axis.clear()
        axis.set_axis_off()
        axis.text2D(.5, .5, localised("No structure is loaded", "Aucune structure n’est chargée",
                                    "Структура не загружена"),
                    transform=axis.transAxes, ha="center", va="center")
        return
    if scene is None or scene.crystal is not crystal or scene.spheres not in axis.collections:
        if scene is not None:
            scene.basis.remove()
        scene = CrystalScene(axis, crystal)
        axis._crystal_scene = scene
    scene.update(orientation, preview=preview, show_basis=show_basis)
    return scene


def discard_scene(axis):
    """Drop cached artists so palette changes are applied on the next draw."""
    if axis is None:
        return
    scene = getattr(axis, "_crystal_scene", None)
    if scene is not None:
        scene.basis.remove()
        del axis._crystal_scene
