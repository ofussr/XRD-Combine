"""PySide6 crystal-structure viewer with the stable orientation controls."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr
from ..services.pole_figure import (
    base_orientation,
    euler_matrix,
    matrix_to_euler,
    pole_display_orientation,
)
from ..services.structure_scene import (
    build_structure_scene,
    default_polyhedron_elements,
    direction_orientation,
    screen_drag_rotation,
)
from .structure_canvas import CrystalCanvas
from .viewer_page import CollapsibleSection


def _compact(widget: QWidget) -> None:
    widget.setMinimumWidth(0)
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    widget.setSizePolicy(policy)


class StructureViewerPage(QWidget):
    """Structure tab backed by the shared CIF model and a native Qt canvas."""

    def __init__(self, parent=None, *, on_import_paths=None) -> None:
        super().__init__(parent)
        self.on_import_paths = on_import_paths
        self.document = None
        self.crystal = None
        self.scene = None
        self.center_hkl = (0, 1, 0)
        self.base_rotation = np.eye(3)
        self.user_rotation = np.eye(3)
        self.view_name = "hkl"
        self._build_ui()
        self.retranslate()
        self.clear_document()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.controls_scroll.setMinimumWidth(280)
        self.controls_scroll.setMaximumWidth(390)
        self.controls = QWidget()
        _compact(self.controls)
        self.controls_layout = QVBoxLayout(self.controls)
        self.controls_layout.setContentsMargins(7, 7, 7, 7)
        self.controls_layout.setSpacing(7)
        self.controls_scroll.setWidget(self.controls)
        splitter.addWidget(self.controls_scroll)

        self.file_section = CollapsibleSection("")
        self.open_button = QPushButton()
        self.open_button.clicked.connect(self.open_cif)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        for widget in (self.open_button, self.path_label, self.info_label):
            _compact(widget)
            self.file_section.content_layout.addWidget(widget)
        self.controls_layout.addWidget(self.file_section)

        self.orientation_section = CollapsibleSection("")
        directions = QGridLayout()
        directions.setContentsMargins(0, 0, 0, 0)
        directions.setHorizontalSpacing(3)
        self.direction_buttons: dict[str, QPushButton] = {}
        for column, name in enumerate(("a", "b", "c", "a*", "b*", "c*")):
            button = QPushButton(name)
            button.clicked.connect(
                lambda _checked=False, direction=name: self.apply_direction(direction)
            )
            directions.addWidget(button, 0, column)
            self.direction_buttons[name] = button
        self.orientation_section.content_layout.addLayout(directions)
        self.standard_button = QPushButton()
        self.standard_button.clicked.connect(lambda: self.apply_direction("standard"))
        self.orientation_section.content_layout.addWidget(self.standard_button)
        self.orientation_info = QLabel()
        self.orientation_info.setWordWrap(True)
        self.orientation_section.content_layout.addWidget(self.orientation_info)

        self.hkl_group = QGroupBox()
        hkl_layout = QGridLayout(self.hkl_group)
        hkl_layout.setContentsMargins(7, 7, 7, 7)
        hkl_layout.setHorizontalSpacing(4)
        self.hkl_inputs: list[QSpinBox] = []
        for column, (name, value) in enumerate(zip(("h", "k", "l"), self.center_hkl)):
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hkl_layout.addWidget(label, 0, column)
            field = QSpinBox()
            field.setRange(-999, 999)
            field.setValue(value)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            field.lineEdit().returnPressed.connect(self.apply_center)
            _compact(field)
            hkl_layout.addWidget(field, 1, column)
            self.hkl_inputs.append(field)
        self.hkl_button = QPushButton()
        self.hkl_button.clicked.connect(self.apply_center)
        hkl_layout.addWidget(self.hkl_button, 2, 0, 1, 3)
        self.orientation_section.content_layout.addWidget(self.hkl_group)

        self.absolute_group, self.absolute_inputs, self.absolute_button = (
            self._rotation_group(relative=False)
        )
        self.relative_group, self.relative_inputs, self.relative_button = (
            self._rotation_group(relative=True)
        )
        self.orientation_section.content_layout.addWidget(self.absolute_group)
        self.orientation_section.content_layout.addWidget(self.relative_group)
        self.reset_button = QPushButton()
        self.reset_button.clicked.connect(self.reset_rotation)
        self.orientation_section.content_layout.addWidget(self.reset_button)
        self.controls_layout.addWidget(self.orientation_section)

        self.display_section = CollapsibleSection("")
        self.atoms_check = QCheckBox()
        self.atoms_check.setChecked(True)
        self.external_atoms_check = QCheckBox()
        self.external_atoms_check.setChecked(True)
        self.bonds_check = QCheckBox()
        self.bonds_check.setChecked(True)
        self.cell_check = QCheckBox()
        self.cell_check.setChecked(True)
        self.basis_check = QCheckBox()
        self.basis_check.setChecked(True)
        for check in (
            self.atoms_check,
            self.external_atoms_check,
            self.bonds_check,
            self.cell_check,
            self.basis_check,
        ):
            check.toggled.connect(self.update_display)
            self.display_section.content_layout.addWidget(check)
        self.bond_note = QLabel()
        self.bond_note.setWordWrap(True)
        self.display_section.content_layout.addWidget(self.bond_note)
        self.controls_layout.addWidget(self.display_section)

        self.polyhedra_section = CollapsibleSection("")
        self.polyhedra_check = QCheckBox()
        self.polyhedra_check.setChecked(True)
        self.polyhedra_check.toggled.connect(self.update_polyhedron_controls)
        self.polyhedra_section.content_layout.addWidget(self.polyhedra_check)
        self.style_label = QLabel()
        self.style_combo = QComboBox()
        self.style_combo.addItem("", "colour")
        self.style_combo.addItem("", "engraving")
        self.style_combo.currentIndexChanged.connect(self.update_polyhedron_controls)
        self.polyhedra_section.content_layout.addWidget(self.style_label)
        self.polyhedra_section.content_layout.addWidget(self.style_combo)
        self.hatching_check = QCheckBox()
        self.hatching_check.setChecked(True)
        self.hatching_check.toggled.connect(self.update_polyhedron_controls)
        self.polyhedra_section.content_layout.addWidget(self.hatching_check)
        self.density_label = QLabel()
        self.density_label.setWordWrap(True)
        _compact(self.density_label)
        self.density_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_slider.setRange(4, 42)
        self.density_slider.setValue(22)
        self.density_slider.valueChanged.connect(self.update_display)
        self.polyhedra_section.content_layout.addWidget(self.density_label)
        self.polyhedra_section.content_layout.addWidget(self.density_slider)
        self.grip_label = QLabel()
        self.grip_slider = QSlider(Qt.Orientation.Horizontal)
        self.grip_slider.setRange(0, 200)
        self.grip_slider.setValue(50)
        self.grip_slider.valueChanged.connect(self.update_display)
        self.polyhedra_section.content_layout.addWidget(self.grip_label)
        self.polyhedra_section.content_layout.addWidget(self.grip_slider)
        self.centre_elements_label = QLabel()
        self.centre_elements_label.setWordWrap(True)
        self.polyhedra_section.content_layout.addWidget(self.centre_elements_label)
        self.centre_elements = QListWidget()
        self.centre_elements.setMaximumHeight(130)
        self.centre_elements.itemChanged.connect(self.update_display)
        self.polyhedra_section.content_layout.addWidget(self.centre_elements)
        self.polyhedra_note = QLabel()
        self.polyhedra_note.setWordWrap(True)
        self.polyhedra_section.content_layout.addWidget(self.polyhedra_note)
        self.controls_layout.addWidget(self.polyhedra_section)
        self.controls_layout.addStretch(1)

        self.canvas = CrystalCanvas()
        self.canvas.rotation_dragged.connect(self.rotate_from_mouse)
        self.canvas.rotation_finished.connect(self.finish_mouse_rotation)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])

    def _rotation_group(self, *, relative: bool):
        group = QGroupBox()
        layout = QGridLayout(group)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setHorizontalSpacing(4)
        fields = []
        names = ("ΔX", "ΔY", "ΔZ") if relative else ("X", "Y", "Z")
        for column, name in enumerate(names):
            label = QLabel(f"{name}, °")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 0, column)
            field = QDoubleSpinBox()
            field.setRange(-360000.0, 360000.0)
            field.setDecimals(3)
            field.setSingleStep(1.0)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            _compact(field)
            layout.addWidget(field, 1, column)
            fields.append(field)
        button = QPushButton()
        if relative:
            button.clicked.connect(self.apply_relative_rotation)
        else:
            button.clicked.connect(self.apply_exact_rotation)
        layout.addWidget(button, 2, 0, 1, 3)
        return group, fields, button

    def retranslate(self) -> None:
        self.file_section.set_title(tr("text.cif_structure_2"))
        self.open_button.setText(tr("text.open_cif"))
        self.orientation_section.set_title(tr("text.orientation_and_rotation"))
        self.standard_button.setText(tr("text.standard_orientation"))
        self.hkl_group.setTitle(tr("text.orientation_by_h_k_l"))
        self.hkl_button.setText(tr("text.view_along_h_k_l_normal"))
        self.absolute_group.setTitle(tr("text.absolute_rotation"))
        self.relative_group.setTitle(tr("text.relative_rotation"))
        self.absolute_button.setText(tr("text.set_angles"))
        self.relative_button.setText(tr("text.rotate_by_delta"))
        self.reset_button.setText(tr("text.reset_rotation"))
        self.display_section.set_title(tr("text.display"))
        self.atoms_check.setText(tr("qt.show_atoms"))
        self.external_atoms_check.setText(tr("qt.show_external_polyhedron_atoms"))
        self.bonds_check.setText(tr("qt.show_bonds"))
        self.cell_check.setText(tr("qt.show_unit_cell"))
        self.basis_check.setText(tr("text.basis_vectors"))
        self.bond_note.setText(tr("text.bonds_are_approximate_based_on_covalent_radii"))
        self.polyhedra_section.set_title(tr("qt.polyhedra"))
        self.polyhedra_check.setText(tr("qt.show_polyhedra"))
        self.style_label.setText(tr("qt.drawing_style"))
        self.style_combo.setItemText(0, tr("qt.colour_style"))
        self.style_combo.setItemText(1, tr("qt.engraving_style"))
        self.hatching_check.setText(tr("qt.hatching"))
        self._update_hatching_labels()
        self.centre_elements_label.setText(tr("qt.polyhedron_centre_elements"))
        self.polyhedra_note.setText(tr("qt.polyhedra_distance_note"))
        if self.document is None:
            self.path_label.setText(tr("text.no_cif_loaded"))
        self._update_information()
        self._update_orientation_info()
        self.canvas.update()

    def open_cif(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            tr("text.open_cif"),
            "",
            "CIF (*.cif);;" + tr("text.all_files") + " (*)",
        )
        if path and self.on_import_paths is not None:
            self.on_import_paths([path])

    def load_document(self, document) -> bool:
        self.document = document
        self.path_label.setText(Path(document.source).name)
        crystal = getattr(document, "crystal", None)
        if crystal is None:
            self.crystal = None
            self.scene = None
            self.canvas.set_scene(None)
            self._set_structure_controls_enabled(False)
            self._update_information()
            return False
        try:
            scene = build_structure_scene(crystal)
            base = base_orientation(crystal, self.center_hkl)
        except Exception as error:
            QMessageBox.warning(self, tr("text.could_not_open_cif"), str(error))
            self.crystal = None
            self.scene = None
            self.canvas.set_scene(None)
            self._set_structure_controls_enabled(False)
            return False
        self.crystal = crystal
        self.scene = scene
        self.base_rotation = base
        self.user_rotation = np.eye(3)
        self.view_name = "hkl"
        for field, value in zip(self.hkl_inputs, self.center_hkl):
            field.setValue(value)
        self.canvas.set_scene(scene)
        self._populate_centre_elements()
        self._set_structure_controls_enabled(True)
        self.reset_rotation()
        self._update_information()
        return True

    def clear_document(self) -> None:
        self.document = None
        self.crystal = None
        self.scene = None
        self.canvas.set_scene(None)
        self.path_label.setText(tr("text.no_cif_loaded"))
        self.info_label.clear()
        self.orientation_info.clear()
        self.centre_elements.clear()
        self._set_structure_controls_enabled(False)

    def _set_structure_controls_enabled(self, enabled: bool) -> None:
        self.orientation_section.content.setEnabled(enabled)
        self.display_section.content.setEnabled(enabled)
        self.polyhedra_section.content.setEnabled(
            enabled and self.scene is not None and bool(self.scene.polyhedra)
        )
        self.basis_check.setEnabled(enabled)

    def _populate_centre_elements(self) -> None:
        self.centre_elements.blockSignals(True)
        self.centre_elements.clear()
        available = sorted(
            {
                element
                for polyhedron in self.scene.polyhedra
                for element in polyhedron.elements
            }
        )
        selected = default_polyhedron_elements(self.scene.polyhedra)
        for element in available:
            item = QListWidgetItem(element)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if element in selected
                else Qt.CheckState.Unchecked
            )
            self.centre_elements.addItem(item)
        self.centre_elements.blockSignals(False)
        self.polyhedra_section.content.setEnabled(bool(available))
        self.update_polyhedron_controls()

    def selected_centre_elements(self) -> set[str]:
        return {
            self.centre_elements.item(index).text()
            for index in range(self.centre_elements.count())
            if self.centre_elements.item(index).checkState() == Qt.CheckState.Checked
        }

    def _update_information(self) -> None:
        if self.crystal is None:
            self.info_label.setText(
                tr("text.no_atom_positions_are_available")
                if self.document is not None
                else ""
            )
            return
        crystal = self.crystal
        self.info_label.setText(
            f"{crystal.formula}; {crystal.space_group}\n"
            f"a = {crystal.a:.4f} Å, b = {crystal.b:.4f} Å, c = {crystal.c:.4f} Å\n"
            f"α = {crystal.alpha:.3f}°, β = {crystal.beta:.3f}°, γ = {crystal.gamma:.3f}°"
        )

    def _update_orientation_info(self) -> None:
        if self.crystal is None:
            self.orientation_info.clear()
            return
        if self.view_name == "standard":
            text = tr("qt.orientation_standard_base")
        elif self.view_name == "hkl":
            text = tr(
                "qt.orientation_hkl_base",
                indices=" ".join(str(value) for value in self.center_hkl),
            )
        else:
            text = tr("qt.orientation_direction_base", direction=self.view_name)
        self.orientation_info.setText(text)

    def apply_center(self) -> None:
        if self.crystal is None:
            return
        hkl = tuple(field.value() for field in self.hkl_inputs)
        if hkl == (0, 0, 0):
            QMessageBox.warning(self, tr("text.error"), tr("qt.hkl_must_not_be_zero"))
            return
        try:
            self.base_rotation = base_orientation(self.crystal, hkl)
        except Exception as error:
            QMessageBox.warning(self, tr("text.error"), str(error))
            return
        self.center_hkl = hkl
        self.view_name = "hkl"
        self.reset_rotation()

    def apply_direction(self, name: str) -> None:
        if self.crystal is None:
            return
        self.base_rotation = direction_orientation(self.crystal, name)
        self.view_name = name
        if name.endswith("*"):
            self.center_hkl = tuple(
                int(index == "abc".index(name[0])) for index in range(3)
            )
            for field, value in zip(self.hkl_inputs, self.center_hkl):
                field.setValue(value)
        self.reset_rotation()

    def apply_exact_rotation(self) -> None:
        if self.crystal is None:
            return
        angles = [field.value() for field in self.absolute_inputs]
        if not all(math.isfinite(angle) for angle in angles):
            return
        self.user_rotation = euler_matrix(*angles)
        self._update_angles()
        self.redraw()

    def apply_relative_rotation(self) -> None:
        if self.crystal is None:
            return
        angles = [field.value() for field in self.relative_inputs]
        if not all(math.isfinite(angle) for angle in angles):
            return
        self.user_rotation = euler_matrix(*angles) @ self.user_rotation
        for field in self.relative_inputs:
            field.setValue(0.0)
        self._update_angles()
        self.redraw()

    def _update_angles(self) -> None:
        for field, angle in zip(self.absolute_inputs, matrix_to_euler(self.user_rotation)):
            field.blockSignals(True)
            field.setValue(0.0 if abs(angle) < 5e-10 else angle)
            field.blockSignals(False)

    def reset_rotation(self) -> None:
        self.user_rotation = np.eye(3)
        for field in self.relative_inputs:
            field.setValue(0.0)
        self._update_angles()
        self._update_orientation_info()
        self.redraw()

    def rotate_from_mouse(self, dx: float, dy: float) -> None:
        if self.crystal is None:
            return
        self.user_rotation = screen_drag_rotation(dx, dy) @ self.user_rotation
        self.redraw()

    def finish_mouse_rotation(self) -> None:
        self._update_angles()
        self.redraw()

    def update_polyhedron_controls(self, _value=None) -> None:
        enabled = self.polyhedra_check.isChecked() and self.scene is not None
        self.style_label.setEnabled(enabled)
        self.style_combo.setEnabled(enabled)
        engraving = self.style_combo.currentData() == "engraving"
        self.hatching_check.setEnabled(enabled and engraving)
        self.density_label.setEnabled(enabled and engraving and self.hatching_check.isChecked())
        self.density_slider.setEnabled(enabled and engraving and self.hatching_check.isChecked())
        self.grip_label.setEnabled(
            enabled and engraving and self.hatching_check.isChecked()
        )
        self.grip_slider.setEnabled(
            enabled and engraving and self.hatching_check.isChecked()
        )
        self.centre_elements_label.setEnabled(enabled)
        self.centre_elements.setEnabled(enabled)
        self.update_display()

    def _update_hatching_labels(self) -> None:
        self.density_label.setText(
            f"{tr('qt.hatching_density')}: {self.density_slider.value()}"
        )
        self.grip_label.setText(
            f"{tr('qt.hatching_grip')}: {self.grip_slider.value()}"
        )

    def update_display(self, _value=None) -> None:
        self._update_hatching_labels()
        self.external_atoms_check.setEnabled(
            self.scene is not None and self.atoms_check.isChecked()
        )
        self.canvas.show_atoms = self.atoms_check.isChecked()
        self.canvas.show_external_atoms = self.external_atoms_check.isChecked()
        self.canvas.show_bonds = self.bonds_check.isChecked()
        self.canvas.show_cell = self.cell_check.isChecked()
        self.canvas.show_basis = self.basis_check.isChecked()
        self.canvas.show_polyhedra = self.polyhedra_check.isChecked()
        self.canvas.engraving = self.style_combo.currentData() == "engraving"
        self.canvas.hatching = self.hatching_check.isChecked()
        self.canvas.hatch_density = self.density_slider.value()
        self.canvas.hatch_grip = self.grip_slider.value()
        self.canvas.polyhedron_elements = self.selected_centre_elements()
        self.canvas.update()

    def redraw(self) -> None:
        if self.crystal is None:
            self.canvas.update()
            return
        orientation = pole_display_orientation(self.user_rotation @ self.base_rotation)
        self.canvas.set_orientation(orientation)

    def refresh_atom_styles(self) -> None:
        self.canvas.update()


__all__ = ["StructureViewerPage"]
