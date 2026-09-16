"""PySide6 crystal-structure viewer with the stable orientation controls."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from unit_cell_gui import CrystalCanvas, DisplayOptions, StyleOverrides

from ..atom_styles import atom_colour
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
)
from .unit_cell_adapter import unit_cell_scene, user_rotation_from_display
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
        self.geometry_scene = None
        self.scene = None
        self.center_hkl = (0, 1, 0)
        self.base_rotation = np.eye(3)
        self.user_rotation = np.eye(3)
        self.view_name = "hkl"
        self.atom_component_visibility: dict[str, bool] = {}
        self.atom_component_colours: dict[str, str] = {}
        self.polyhedron_site_visibility: dict[str, bool] = {}
        self.polyhedron_site_colours: dict[str, str] = {}
        self.polyhedron_opaque_sites: set[str] = set()
        self._updating_style_trees = False
        self.atom_component_rows: dict[str, tuple] = {}
        self.atom_element_rows: dict[str, tuple] = {}
        self.polyhedron_site_rows: dict[str, tuple] = {}
        self.polyhedron_group_rows: dict[str, tuple] = {}
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
        self.style_label = QLabel()
        self.style_combo = QComboBox()
        self.style_combo.addItem("", "colour")
        self.style_combo.addItem("", "engraving")
        self.style_combo.currentIndexChanged.connect(self.update_polyhedron_controls)
        self.display_section.content_layout.addWidget(self.style_label)
        self.display_section.content_layout.addWidget(self.style_combo)

        self.atoms_section = CollapsibleSection("")
        self.atoms_check = QCheckBox()
        self.atoms_check.setChecked(True)
        self.atoms_check.toggled.connect(self.update_display)
        self.atoms_section.content_layout.addWidget(self.atoms_check)
        self.atom_tree = self._new_style_tree(3)
        self.atoms_section.content_layout.addWidget(self.atom_tree)
        self.atom_size_label = QLabel()
        self.atom_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.atom_size_slider.setRange(25, 200)
        self.atom_size_slider.setValue(100)
        self.atom_size_slider.valueChanged.connect(self.update_display)
        self.atoms_section.content_layout.addWidget(self.atom_size_label)
        self.atoms_section.content_layout.addWidget(self.atom_size_slider)
        self.external_atoms_check = QCheckBox()
        self.external_atoms_check.setChecked(True)
        self.bonds_check = QCheckBox()
        self.bonds_check.setChecked(True)
        self.cell_check = QCheckBox()
        self.cell_check.setChecked(True)
        self.basis_check = QCheckBox()
        self.basis_check.setChecked(True)
        for check in (self.external_atoms_check, self.bonds_check):
            check.toggled.connect(self.update_display)
            self.atoms_section.content_layout.addWidget(check)
        self.bond_note = QLabel()
        self.bond_note.setWordWrap(True)
        self.atoms_section.content_layout.addWidget(self.bond_note)

        self.polyhedra_section = CollapsibleSection("")
        self.polyhedra_check = QCheckBox()
        self.polyhedra_check.setChecked(False)
        self.polyhedra_check.toggled.connect(self.update_polyhedron_controls)
        self.polyhedra_section.content_layout.addWidget(self.polyhedra_check)
        self.polyhedron_tree = self._new_style_tree(4)
        self.polyhedra_section.content_layout.addWidget(self.polyhedron_tree)
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
        self.polyhedra_note = QLabel()
        self.polyhedra_note.setWordWrap(True)
        self.polyhedra_section.content_layout.addWidget(self.polyhedra_note)

        self.display_section.content_layout.addWidget(self.atoms_section)
        self.display_section.content_layout.addWidget(self.polyhedra_section)
        for check in (self.cell_check, self.basis_check):
            check.toggled.connect(self.update_display)
            self.display_section.content_layout.addWidget(check)
        self.controls_layout.addWidget(self.display_section)
        self.controls_layout.addStretch(1)

        self.canvas = CrystalCanvas()
        self.canvas.orientation_changed.connect(self.orientation_from_canvas)
        self.canvas.interaction_finished.connect(self.finish_mouse_rotation)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])

    @staticmethod
    def _new_style_tree(columns: int) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setColumnCount(columns)
        tree.setRootIsDecorated(True)
        tree.setAlternatingRowColors(True)
        tree.setMaximumHeight(185)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tree.header().setStretchLastSection(False)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, columns):
            tree.header().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        return tree

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
        self.style_label.setText(tr("qt.drawing_mode"))
        self.style_combo.setItemText(0, tr("qt.colour_style"))
        self.style_combo.setItemText(1, tr("qt.black_white_style"))
        self.atoms_section.set_title(tr("qt.atoms"))
        self.atoms_check.setText(tr("qt.show_atoms"))
        self.atom_tree.setHeaderLabels([tr("qt.site"), "C", "V"])
        self.atom_tree.headerItem().setToolTip(1, tr("text.colour"))
        self.atom_tree.headerItem().setToolTip(2, tr("qt.visibility"))
        self.external_atoms_check.setText(tr("qt.show_external_atoms"))
        self.bonds_check.setText(tr("qt.show_bonds"))
        self.cell_check.setText(tr("qt.show_unit_cell"))
        self.basis_check.setText(tr("text.basis_vectors"))
        self.bond_note.setText(tr("text.bonds_are_approximate_based_on_covalent_radii"))
        self.polyhedra_section.set_title(tr("qt.polyhedra"))
        self.polyhedra_check.setText(tr("qt.show_polyhedra"))
        self.polyhedron_tree.setHeaderLabels([tr("qt.polyhedra"), "C", "S", "V"])
        self.polyhedron_tree.headerItem().setToolTip(1, tr("text.colour"))
        self.polyhedron_tree.headerItem().setToolTip(2, tr("qt.opaque"))
        self.polyhedron_tree.headerItem().setToolTip(3, tr("qt.visibility"))
        self.hatching_check.setText(tr("qt.hatching"))
        self._update_hatching_labels()
        self._update_atom_size_label()
        self.polyhedra_note.setText(tr("qt.polyhedra_distance_note"))
        if self.document is None:
            self.path_label.setText(tr("text.no_cif_loaded"))
        self._update_information()
        self._update_orientation_info()
        self.canvas.set_labels(
            empty=tr("text.no_structure_is_loaded"),
            title=(
                tr("qt.structure_title", formula=self.crystal.formula)
                if self.crystal is not None
                else ""
            ),
        )

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
            self.geometry_scene = None
            self.scene = None
            self.canvas.set_scene(None)
            self._set_structure_controls_enabled(False)
            self._update_information()
            return False
        try:
            geometry_scene = build_structure_scene(crystal)
            scene = unit_cell_scene(geometry_scene)
            base = base_orientation(crystal, self.center_hkl)
        except Exception as error:
            QMessageBox.warning(self, tr("text.could_not_open_cif"), str(error))
            self.crystal = None
            self.geometry_scene = None
            self.scene = None
            self.canvas.set_scene(None)
            self._set_structure_controls_enabled(False)
            return False
        self.crystal = crystal
        self.geometry_scene = geometry_scene
        self.scene = scene
        self.base_rotation = base
        self.user_rotation = np.eye(3)
        self.view_name = "hkl"
        self.atom_component_visibility = {}
        self.atom_component_colours = {}
        self.polyhedron_site_visibility = {}
        self.polyhedron_site_colours = {}
        self.polyhedron_opaque_sites = set()
        for field, value in zip(self.hkl_inputs, self.center_hkl):
            field.setValue(value)
        self.canvas.set_scene(scene)
        self.canvas.set_labels(title=tr("qt.structure_title", formula=crystal.formula))
        self._populate_style_trees()
        self._set_structure_controls_enabled(True)
        self.reset_rotation()
        self._update_information()
        return True

    def clear_document(self) -> None:
        self.document = None
        self.crystal = None
        self.geometry_scene = None
        self.scene = None
        self.canvas.set_scene(None)
        self.canvas.set_labels(title="")
        self.path_label.setText(tr("text.no_cif_loaded"))
        self.info_label.clear()
        self.orientation_info.clear()
        self.atom_tree.clear()
        self.polyhedron_tree.clear()
        self.atom_component_rows.clear()
        self.atom_element_rows.clear()
        self.polyhedron_site_rows.clear()
        self.polyhedron_group_rows.clear()
        self._set_structure_controls_enabled(False)

    def _set_structure_controls_enabled(self, enabled: bool) -> None:
        self.orientation_section.content.setEnabled(enabled)
        self.display_section.content.setEnabled(enabled)
        self.atoms_section.content.setEnabled(enabled)
        self.polyhedra_section.content.setEnabled(
            enabled and self.scene is not None and bool(self.scene.polyhedra)
        )
        self.basis_check.setEnabled(enabled)

    @staticmethod
    def _set_colour_button(button: QPushButton, colour: str | QColor) -> None:
        value = QColor(colour).name().upper()
        button.setProperty("site_colour", value)
        button.setToolTip(value)
        button.setFixedSize(34, 19)
        button.setStyleSheet(
            "QPushButton {"
            f" background-color: {value};"
            " border: 1px solid palette(mid);"
            " border-radius: 2px;"
            "}"
        )

    @staticmethod
    def _default_site_colour(components) -> str:
        total = sum(max(0.0, float(item.occupancy)) for item in components)
        if total <= 1e-12:
            return "#B0B0B0"
        channels = [0.0, 0.0, 0.0]
        for component in components:
            colour = QColor(atom_colour(component.element))
            weight = max(0.0, float(component.occupancy))
            channels[0] += weight * colour.red()
            channels[1] += weight * colour.green()
            channels[2] += weight * colour.blue()
        return QColor(*(int(round(channel / total)) for channel in channels)).name().upper()

    @staticmethod
    def _sync_parent_checkbox(parent: QCheckBox, children: list[QCheckBox]) -> None:
        states = [child.isChecked() for child in children]
        parent.setTristate(True)
        if states and all(states):
            parent.setCheckState(Qt.CheckState.Checked)
        elif any(states):
            parent.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            parent.setCheckState(Qt.CheckState.Unchecked)

    def _populate_style_trees(self) -> None:
        self._updating_style_trees = True
        try:
            self._populate_atom_tree()
            self._populate_polyhedron_tree()
        finally:
            self._updating_style_trees = False
        self.update_polyhedron_controls()

    def _populate_atom_tree(self) -> None:
        self.atom_tree.clear()
        self.atom_component_rows.clear()
        self.atom_element_rows.clear()
        if self.scene is None:
            return
        by_element: dict[str, dict[str, object]] = defaultdict(dict)
        for atom in self.scene.atoms:
            for component in atom.components:
                by_element[component.element][component.key] = component
        for element in sorted(by_element):
            components = sorted(
                by_element[element].values(),
                key=lambda component: (component.label, component.key),
            )
            parent_item = QTreeWidgetItem([element, "", ""])
            self.atom_tree.addTopLevelItem(parent_item)
            parent_colour = QPushButton()
            self._set_colour_button(parent_colour, atom_colour(element))
            parent_colour.clicked.connect(
                lambda _checked=False, symbol=element: (
                    self._choose_atom_element_colour(symbol)
                )
            )
            parent_visible = QCheckBox()
            parent_visible.setTristate(True)
            parent_visible.setChecked(True)
            parent_visible.stateChanged.connect(
                lambda state, symbol=element: self._set_atom_element_visible(symbol, state)
            )
            self.atom_tree.setItemWidget(parent_item, 1, parent_colour)
            self.atom_tree.setItemWidget(parent_item, 2, parent_visible)
            component_keys: list[str] = []
            for component in components:
                component_keys.append(component.key)
                self.atom_component_visibility.setdefault(component.key, True)
                child = QTreeWidgetItem([component.label, "", ""])
                parent_item.addChild(child)
                colour_button = QPushButton()
                colour = self.atom_component_colours.get(
                    component.key,
                    atom_colour(component.element),
                )
                self._set_colour_button(colour_button, colour)
                colour_button.clicked.connect(
                    lambda _checked=False, key=component.key: self._choose_atom_colour(key)
                )
                visible = QCheckBox()
                visible.setChecked(self.atom_component_visibility[component.key])
                visible.toggled.connect(
                    lambda checked, key=component.key: self._set_atom_component_visible(
                        key,
                        checked,
                    )
                )
                self.atom_tree.setItemWidget(child, 1, colour_button)
                self.atom_tree.setItemWidget(child, 2, visible)
                self.atom_component_rows[component.key] = (
                    child,
                    colour_button,
                    visible,
                    component,
                )
            self.atom_element_rows[element] = (
                parent_item,
                parent_colour,
                parent_visible,
                tuple(component_keys),
            )
            parent_item.setExpanded(True)

    def _populate_polyhedron_tree(self) -> None:
        self.polyhedron_tree.clear()
        self.polyhedron_site_rows.clear()
        self.polyhedron_group_rows.clear()
        if self.scene is None:
            return
        sites = {}
        for polyhedron in self.scene.polyhedra:
            sites.setdefault(polyhedron.site_key, polyhedron)
        default_elements = default_polyhedron_elements(self.scene.polyhedra)
        grouped: dict[str, list] = defaultdict(list)
        for polyhedron in sites.values():
            group = "/".join(sorted(set(polyhedron.elements)))
            grouped[group].append(polyhedron)
            self.polyhedron_site_visibility.setdefault(
                polyhedron.site_key,
                bool(set(polyhedron.elements).intersection(default_elements)),
            )
        for group in sorted(grouped):
            polyhedra = sorted(
                grouped[group],
                key=lambda polyhedron: (polyhedron.site_label, polyhedron.site_key),
            )
            parent_item = QTreeWidgetItem([group, "", "", ""])
            self.polyhedron_tree.addTopLevelItem(parent_item)
            parent_colour = QPushButton()
            self._set_colour_button(
                parent_colour,
                self._default_site_colour(polyhedra[0].components),
            )
            parent_colour.clicked.connect(
                lambda _checked=False, name=group: (
                    self._choose_polyhedron_group_colour(name)
                )
            )
            parent_opaque = QCheckBox()
            parent_opaque.setTristate(True)
            parent_opaque.stateChanged.connect(
                lambda state, name=group: self._set_polyhedron_group_opaque(
                    name,
                    state,
                )
            )
            parent_visible = QCheckBox()
            parent_visible.setTristate(True)
            parent_visible.stateChanged.connect(
                lambda state, name=group: self._set_polyhedron_group_visible(name, state)
            )
            self.polyhedron_tree.setItemWidget(parent_item, 1, parent_colour)
            self.polyhedron_tree.setItemWidget(parent_item, 2, parent_opaque)
            self.polyhedron_tree.setItemWidget(parent_item, 3, parent_visible)
            site_keys: list[str] = []
            for polyhedron in polyhedra:
                site_keys.append(polyhedron.site_key)
                child = QTreeWidgetItem([polyhedron.site_label, "", "", ""])
                parent_item.addChild(child)
                colour_button = QPushButton()
                colour = self.polyhedron_site_colours.get(
                    polyhedron.site_key,
                    self._default_site_colour(polyhedron.components),
                )
                self._set_colour_button(colour_button, colour)
                colour_button.clicked.connect(
                    lambda _checked=False, key=polyhedron.site_key: self._choose_polyhedron_colour(key)
                )
                opaque = QCheckBox()
                opaque.setChecked(
                    polyhedron.site_key in self.polyhedron_opaque_sites
                )
                opaque.toggled.connect(
                    lambda checked, key=polyhedron.site_key: self._set_polyhedron_opaque(
                        key,
                        checked,
                    )
                )
                visible = QCheckBox()
                visible.setChecked(self.polyhedron_site_visibility[polyhedron.site_key])
                visible.toggled.connect(
                    lambda checked, key=polyhedron.site_key: self._set_polyhedron_visible(
                        key,
                        checked,
                    )
                )
                self.polyhedron_tree.setItemWidget(child, 1, colour_button)
                self.polyhedron_tree.setItemWidget(child, 2, opaque)
                self.polyhedron_tree.setItemWidget(child, 3, visible)
                self.polyhedron_site_rows[polyhedron.site_key] = (
                    child,
                    colour_button,
                    opaque,
                    visible,
                    polyhedron,
                )
            self.polyhedron_group_rows[group] = (
                parent_item,
                parent_colour,
                parent_opaque,
                parent_visible,
                tuple(site_keys),
            )
            parent_item.setExpanded(True)
        self._sync_all_parent_checks()

    def _choose_colour(self, initial: str) -> str | None:
        selected = QColorDialog.getColor(
            QColor(initial),
            self,
            tr("qt.choose_colour"),
        )
        return selected.name().upper() if selected.isValid() else None

    def _choose_atom_colour(self, key: str) -> None:
        row = self.atom_component_rows.get(key)
        if row is None or self.style_combo.currentData() == "engraving":
            return
        initial = self.atom_component_colours.get(key, row[1].property("site_colour"))
        colour = self._choose_colour(initial)
        if colour is None:
            return
        self.atom_component_colours[key] = colour
        self._set_colour_button(row[1], colour)
        self.update_display()

    def _choose_atom_element_colour(self, element: str) -> None:
        row = self.atom_element_rows.get(element)
        if row is None or self.style_combo.currentData() == "engraving":
            return
        colour = self._choose_colour(row[1].property("site_colour"))
        if colour is None:
            return
        self._set_colour_button(row[1], colour)
        for key in row[3]:
            self.atom_component_colours[key] = colour
            self._set_colour_button(self.atom_component_rows[key][1], colour)
        self.update_display()

    def _choose_polyhedron_colour(self, key: str) -> None:
        row = self.polyhedron_site_rows.get(key)
        if row is None or self.style_combo.currentData() == "engraving":
            return
        colour = self._choose_colour(row[1].property("site_colour"))
        if colour is None:
            return
        self.polyhedron_site_colours[key] = colour
        self._set_colour_button(row[1], colour)
        self.update_display()

    def _choose_polyhedron_group_colour(self, group: str) -> None:
        row = self.polyhedron_group_rows.get(group)
        if row is None or self.style_combo.currentData() == "engraving":
            return
        colour = self._choose_colour(row[1].property("site_colour"))
        if colour is None:
            return
        self._set_colour_button(row[1], colour)
        for key in row[4]:
            self.polyhedron_site_colours[key] = colour
            self._set_colour_button(self.polyhedron_site_rows[key][1], colour)
        self.update_display()

    def _set_atom_component_visible(self, key: str, checked: bool) -> None:
        if self._updating_style_trees:
            return
        self.atom_component_visibility[key] = bool(checked)
        self._sync_atom_parent(self.atom_component_rows[key][3].element)
        self.update_display()

    def _set_atom_element_visible(self, element: str, state: int) -> None:
        if self._updating_style_trees:
            return
        checked = int(state) != int(Qt.CheckState.Unchecked.value)
        self._updating_style_trees = True
        try:
            for key in self.atom_element_rows[element][3]:
                self.atom_component_visibility[key] = checked
                self.atom_component_rows[key][2].setChecked(checked)
        finally:
            self._updating_style_trees = False
        self._sync_atom_parent(element)
        self.update_display()

    def _sync_atom_parent(self, element: str) -> None:
        row = self.atom_element_rows.get(element)
        if row is None:
            return
        previous = self._updating_style_trees
        self._updating_style_trees = True
        try:
            self._sync_parent_checkbox(
                row[2],
                [self.atom_component_rows[key][2] for key in row[3]],
            )
        finally:
            self._updating_style_trees = previous

    def _set_polyhedron_visible(self, key: str, checked: bool) -> None:
        if self._updating_style_trees:
            return
        self.polyhedron_site_visibility[key] = bool(checked)
        self._sync_polyhedron_parent_for_key(key)
        self.update_display()

    def _set_polyhedron_group_visible(self, group: str, state: int) -> None:
        if self._updating_style_trees:
            return
        checked = int(state) != int(Qt.CheckState.Unchecked.value)
        self._updating_style_trees = True
        try:
            for key in self.polyhedron_group_rows[group][4]:
                self.polyhedron_site_visibility[key] = checked
                self.polyhedron_site_rows[key][3].setChecked(checked)
        finally:
            self._updating_style_trees = False
        self._sync_polyhedron_parent(group)
        self.update_display()

    def _set_polyhedron_opaque(self, key: str, checked: bool) -> None:
        if self._updating_style_trees:
            return
        if checked:
            self.polyhedron_opaque_sites.add(key)
        else:
            self.polyhedron_opaque_sites.discard(key)
        self._sync_polyhedron_parent_for_key(key)
        self.update_display()

    def _set_polyhedron_group_opaque(self, group: str, state: int) -> None:
        if self._updating_style_trees:
            return
        checked = int(state) != int(Qt.CheckState.Unchecked.value)
        self._updating_style_trees = True
        try:
            for key in self.polyhedron_group_rows[group][4]:
                if checked:
                    self.polyhedron_opaque_sites.add(key)
                else:
                    self.polyhedron_opaque_sites.discard(key)
                self.polyhedron_site_rows[key][2].setChecked(checked)
        finally:
            self._updating_style_trees = False
        self._sync_polyhedron_parent(group)
        self.update_display()

    def _sync_polyhedron_parent_for_key(self, key: str) -> None:
        for group, row in self.polyhedron_group_rows.items():
            if key in row[4]:
                self._sync_polyhedron_parent(group)
                return

    def _sync_polyhedron_parent(self, group: str) -> None:
        row = self.polyhedron_group_rows.get(group)
        if row is None:
            return
        previous = self._updating_style_trees
        self._updating_style_trees = True
        try:
            self._sync_parent_checkbox(
                row[2],
                [self.polyhedron_site_rows[key][2] for key in row[4]],
            )
            self._sync_parent_checkbox(
                row[3],
                [self.polyhedron_site_rows[key][3] for key in row[4]],
            )
        finally:
            self._updating_style_trees = previous

    def _sync_all_parent_checks(self) -> None:
        for element in self.atom_element_rows:
            self._sync_atom_parent(element)
        for group in self.polyhedron_group_rows:
            self._sync_polyhedron_parent(group)

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

    def orientation_from_canvas(self, orientation: np.ndarray) -> None:
        if self.crystal is None:
            return
        self.user_rotation = user_rotation_from_display(
            orientation,
            self.base_rotation,
        )

    def finish_mouse_rotation(self) -> None:
        self._update_angles()
        self.redraw()

    def update_polyhedron_controls(self, _value=None) -> None:
        enabled = self.polyhedra_check.isChecked() and self.scene is not None
        self.style_label.setEnabled(self.scene is not None)
        self.style_combo.setEnabled(self.scene is not None)
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
        self.polyhedron_tree.setEnabled(enabled)
        for row in self.polyhedron_group_rows.values():
            row[1].setEnabled(enabled and not engraving)
            row[2].setEnabled(enabled and not engraving)
        for row in self.polyhedron_site_rows.values():
            row[1].setEnabled(enabled and not engraving)
            row[2].setEnabled(enabled and not engraving)
        atoms_enabled = self.scene is not None and self.atoms_check.isChecked()
        self.atom_tree.setEnabled(atoms_enabled)
        for row in self.atom_element_rows.values():
            row[1].setEnabled(atoms_enabled and not engraving)
        for row in self.atom_component_rows.values():
            row[1].setEnabled(atoms_enabled and not engraving)
        self.update_display()

    def _update_hatching_labels(self) -> None:
        self.density_label.setText(
            f"{tr('qt.hatching_density')}: {self.density_slider.value()}"
        )
        self.grip_label.setText(
            f"{tr('qt.hatching_grip')}: {self.grip_slider.value()}"
        )

    def _update_atom_size_label(self) -> None:
        self.atom_size_label.setText(
            f"{tr('qt.atom_size')}: {self.atom_size_slider.value()}%"
        )

    def update_display(self, _value=None) -> None:
        self._update_hatching_labels()
        self._update_atom_size_label()
        atoms_enabled = self.scene is not None and self.atoms_check.isChecked()
        self.external_atoms_check.setEnabled(atoms_enabled)
        self.atom_size_label.setEnabled(atoms_enabled)
        self.atom_size_slider.setEnabled(atoms_enabled)
        self.canvas.set_display_options(
            DisplayOptions(
                show_atoms=self.atoms_check.isChecked(),
                show_external_atoms=self.external_atoms_check.isChecked(),
                show_bonds=self.bonds_check.isChecked(),
                show_cell=self.cell_check.isChecked(),
                show_basis=self.basis_check.isChecked(),
                show_polyhedra=self.polyhedra_check.isChecked(),
                engraving=self.style_combo.currentData() == "engraving",
                hatching=self.hatching_check.isChecked(),
                hatch_density=self.density_slider.value(),
                hatch_grip=self.grip_slider.value(),
                atom_scale=self.atom_size_slider.value() / 100.0,
                rotation_enabled=self.canvas.rotation_enabled,
            )
        )
        self.canvas.set_style_overrides(
            StyleOverrides(
                atom_component_visibility=self.atom_component_visibility,
                atom_component_colours=self.atom_component_colours,
                polyhedron_site_visibility=self.polyhedron_site_visibility,
                polyhedron_site_colours=self.polyhedron_site_colours,
                polyhedron_opaque_sites=self.polyhedron_opaque_sites,
            )
        )

    def redraw(self) -> None:
        if self.crystal is None:
            self.canvas.update()
            return
        orientation = pole_display_orientation(self.user_rotation @ self.base_rotation)
        self.canvas.set_orientation(orientation)

    def refresh_atom_styles(self) -> None:
        for key, row in self.atom_component_rows.items():
            if key not in self.atom_component_colours:
                self._set_colour_button(row[1], atom_colour(row[3].element))
        for element, row in self.atom_element_rows.items():
            if not any(key in self.atom_component_colours for key in row[3]):
                self._set_colour_button(row[1], atom_colour(element))
        for key, row in self.polyhedron_site_rows.items():
            if key not in self.polyhedron_site_colours:
                self._set_colour_button(
                    row[1],
                    self._default_site_colour(row[4].components),
                )
        for _group, row in self.polyhedron_group_rows.items():
            if not any(key in self.polyhedron_site_colours for key in row[4]):
                first_polyhedron = self.polyhedron_site_rows[row[4][0]][4]
                self._set_colour_button(
                    row[1],
                    self._default_site_colour(first_polyhedron.components),
                )
        if self.geometry_scene is not None:
            self.scene = unit_cell_scene(self.geometry_scene)
            self.canvas.set_scene(self.scene, reset_camera=False)
            self.canvas.set_labels(
                title=tr("qt.structure_title", formula=self.crystal.formula)
            )
            self.update_display()


__all__ = ["StructureViewerPage"]
