"""Bridge unit-cell-gui camera interaction to project-owned numeric state."""
from unit_cell_gui import CrystalCanvas


class StatefulCrystalCanvas(CrystalCanvas):
    def __init__(self, state):
        self.project_state = state
        super().__init__()

    def remember_camera(self):
        self.project_state.camera.zoom = float(self.zoom)
        self.project_state.camera.pan = (float(self.pan_x) / max(1, self.width()),
                                         float(self.pan_y) / max(1, self.height()))

    def restore_camera(self):
        camera = self.project_state.camera
        self.zoom = camera.zoom
        self.pan_x = camera.pan[0] * self.width()
        self.pan_y = camera.pan[1] * self.height()
        self._hatch_dirty = True
        self._atom_outline_dirty = True
        self.update()

    def zoom_by(self, steps):
        super().zoom_by(steps)
        self.remember_camera()

    def mouseMoveEvent(self, event):
        dragging = self._last_mouse is not None
        super().mouseMoveEvent(event)
        if dragging:
            self.remember_camera()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.restore_camera()
