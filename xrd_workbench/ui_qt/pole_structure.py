"""Embed the shared Qt structure viewer beside a calculated pole figure."""

from copy import deepcopy

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QSplitter

from ..localization import tr
from .structure_viewer import StructureViewerPage


class PolePlotSplitter(QSplitter):
    """Keep both plots readable when the available width is small."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        orientation = Qt.Orientation.Horizontal if self.width() >= 600 else Qt.Orientation.Vertical
        if self.orientation() != orientation:
            self.setOrientation(orientation)
            self.setSizes([500, 500])


class PoleStructureView(StructureViewerPage):
    """Reuse the viewer's geometry, painter, camera and display controls."""

    def __init__(self, parent=None, *, on_orientation_changed, on_rotation_finished):
        super().__init__(parent)
        # The pole page supplies the document and orientation controls. Its
        # Display section hosts this viewer's existing atom/polyhedron controls.
        self.controls_scroll.hide()
        self.canvas.setMinimumSize(240, 240)
        self.canvas.orientation_changed.disconnect(self.orientation_from_canvas)
        self.canvas.interaction_finished.disconnect(self.finish_mouse_rotation)
        self.canvas.orientation_changed.connect(on_orientation_changed)
        self.canvas.interaction_finished.connect(on_rotation_finished)
        # The compact pole preview intentionally contains no polyhedra.
        self.polyhedra_check.setChecked(False)
        self.polyhedra_section.hide()

    def sync_document(self, document, orientation):
        if document is None:
            if self.document is not None:
                self.clear_document()
            return
        if self.document is not document:
            self.load_document(document)
        self.canvas.set_orientation(orientation)

    def retranslate(self):
        super().retranslate()
        self.display_section.set_title(tr("qt.structure_display"))

    def figure_for_export(self, figure):
        """Include the visible native drawing without changing the live figure."""
        if self.isHidden() or self.canvas.scene is None:
            return figure
        result = deepcopy(figure)
        width, height = figure.get_size_inches()
        result.set_size_inches(2 * width, height)
        for axes in result.axes:
            left, bottom, w, h = axes.get_position(original=True).bounds
            axes.set_position([left / 2, bottom, w / 2, h])
        # Render the same painter at higher resolution, keeping pan, zoom,
        # occupancy sectors, hatching and all visibility settings intact.
        scale = 3
        image = QImage(self.canvas.width() * scale, self.canvas.height() * scale,
                       QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            painter.scale(scale, scale)
            self.canvas.render(painter, self.canvas.rect().topLeft())
        finally:
            painter.end()
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
            image.height(), image.width(), 4).copy()
        axes = result.add_axes([.5, 0, .5, 1])
        axes.imshow(pixels)
        axes.set_axis_off()
        return result
