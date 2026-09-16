"""Native Qt calculated pole figures with independent or coupled phase layers."""
from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Sequence
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit, QFileDialog, QMenu, QSlider, QSpinBox, QStackedWidget
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from .plot_toolbar import PlotToolbar
from ..localization import tr, localised, translate_text, choice_code
from .pole_structure import PoleStructureView, PolePlotSplitter
from .unit_cell_adapter import user_rotation_from_display
from .pole_widgets import PoleUi, Value, FrameScheduler, configure, bind_value, choose_colour, save_plot, show_error, show_info
from ..io.reflections import read_scattering_factors, scattering_factor_path
from ..models.crystal import Atom, CifData, CifLoop, Crystal, DisplayAtom, cif_number, direct_basis, expanded_atoms, parse_symmetry_operation, unit_cell_display_atoms
from ..models.data_errors import XRDDataError
from ..models.pole_figure import CalculatedPoleLayer, PolePoint, PoleReflection
from ..services.pole_figure import align_to_z, available_reflections as _available_reflections, base_orientation, calculated_intensity_by_spacing as _calculated_intensity_by_spacing, euler_matrix, follow_orientation_change, format_hkl, group_coincident_poles, in_plane_alignment, marker_sizes_by_d, matrix_to_euler, place_label_boxes, pole_display_orientation, pole_display_position, pole_plot_coordinates, pole_plot_to_sphere, project_reflections, projection_code, rotation_axis_angle, rotation_between, rotation_x, rotation_y, rotation_z

Reflection = PoleReflection
POLE_AZIMUTH_GRID_STEP_DEG = 10

def available_reflections(crystal: Crystal, d_lower: float, d_upper: float, wavelength: float) -> list[Reflection]:
    """Localized compatibility adapter for the GUI-independent pole service."""
    try:
        return _available_reflections(crystal, d_lower, d_upper, wavelength)
    except XRDDataError as exc:
        messages = {'pole_d_positive': localised('The d limits must be positive.', 'Les limites de d doivent être positives.', 'Границы d должны быть положительными.'), 'pole_d_order': localised('The lower d limit cannot exceed the upper limit.', 'La limite inférieure de d ne peut pas dépasser la limite supérieure.', 'Нижняя граница d не может быть больше верхней.'), 'pole_wavelength_positive': localised('The wavelength must be positive.', 'La longueur d’onde doit être positive.', 'Длина волны должна быть положительной.'), 'pole_reflection_limit': localised('The selected lower d limit requires testing more than two million reciprocal-lattice nodes. Increase the lower d limit.', 'La limite inférieure de d choisie nécessite de tester plus de deux millions de nœuds du réseau réciproque. Augmentez cette limite.', 'Выбранная нижняя граница d требует перебора более двух миллионов узлов. Увеличьте нижнюю границу d.')}
        raise ValueError(messages.get(exc.code, str(exc))) from exc

def calculated_intensity_by_spacing(diffraction_structure, radiations: list[tuple[str, float, float]]) -> dict[float, float]:
    """Compatibility adapter that supplies bundled scattering-factor data."""
    factors = read_scattering_factors(scattering_factor_path())
    return _calculated_intensity_by_spacing(diffraction_structure, radiations, factors)

class CalculatedPolePage(QWidget, PoleUi):
    def __init__(self, radiations_provider=None, parent=None, *, on_open_cif=None,
                 on_add_overlay=None, on_remove_overlay=None, overlay_documents_provider=None):
        super().__init__(parent)
        self.root = self
        self.on_open_cif = on_open_cif
        self._init_ui_helpers()
        self.crystal: Crystal | None = None
        self.cif_document = None
        self.radiations_provider = radiations_provider
        self.center_hkl = (0, 1, 0)
        self.base_rotation = np.eye(3)
        self.user_rotation = np.eye(3)
        self.reflections: list[Reflection] = []
        self.points: list[PolePoint] = []
        self.point_groups: list[list[PolePoint]] = []
        self.center_map: dict[str, tuple[int, int, int]] = {}
        self.selected_hkl: tuple[int, int, int] | None = None
        self.selected_layer_index = 0
        self._selection_menu = None
        self.press_event = None
        self.last_arcball: np.ndarray | None = None
        self.last_drag_pixel: tuple[float, float] | None = None
        self.drag_mode: str | None = None
        self.dragged = False
        self.overlay_layer: CalculatedPoleLayer | None = None
        self.on_add_overlay = on_add_overlay
        self.on_remove_overlay = on_remove_overlay
        self.overlay_documents_provider = overlay_documents_provider
        self.overlay_document_map = {}
        self._single_colour_mode: str | None = None
        self.path_var = Value(translation_key='text.no_cif_loaded')
        self.structure_var = Value(value='')
        self.h_var = Value(value='0')
        self.k_var = Value(value='1')
        self.l_var = Value(value='0')
        self.max_index_var = Value(value=4)
        self.wavelength_var = Value(value='1.5406')
        self.d_lower_var = Value(value='1.0')
        self.d_upper_var = Value(value='13')
        self.projection_var = Value(value='stereographic')
        self.labels_var = Value(value=False)
        self.label_leaders_var = Value(value=False)
        self.coincident_outlines_var = Value(value=False)
        self.angle_labels_var = Value(value=True)
        self.show_structure_var = Value(value=False)
        self.basis_visible = Value(value=True)
        self._colorbar = None
        self.size_by_d_var = Value(value=False)
        self.color_mode_var = Value(value='uniform')
        self.intensity_by_spacing: dict[float, float] | None = None
        self.primary_colour_var = Value(value='#2d6da3')
        self.primary_opacity_var = Value(value=100.0)
        self.primary_size_var = Value(value=100.0)
        self.global_size_var = Value(value=100.0)
        self.global_size_label_var = Value(value='100%')
        self.overlay_choice_var = Value(value='')
        self.overlay_name_var = Value(value='')
        self.overlay_colour_var = Value(value='#d65f3c')
        self.overlay_opacity_var = Value(value=70.0)
        self.overlay_size_var = Value(value=100.0)
        self.joint_rotation_var = Value(value=False)
        self.overlay_hkl_vars = [Value(value='0'), Value(value='1'), Value(value='0')]
        self.overlay_rotation_vars = [Value(value='0.0') for _index in range(3)]
        self.overlay_relative_rotation_vars = [Value(value='0.0') for _index in range(3)]
        self.rotation_vars = [Value(value='0.0'), Value(value='0.0'), Value(value='0.0')]
        self.relative_rotation_vars = [Value(value='0.0'), Value(value='0.0'), Value(value='0.0')]
        self.status_var = Value(value='')
        self.plot_renderer = "matplotlib"
        self.pyqtgraph_plot = None
        self._frames = FrameScheduler(self, lambda: self.redraw(preview=True))
        self._display_frames = FrameScheduler(self, self.redraw, interval=40)
        self._build_layout()
        self._connect_canvas()
        self.refresh_overlay_choices()
        self.redraw()

    def _connect_canvas(self) -> None:
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("resize_event", lambda _event: self._display_frames.request())

    def set_plot_renderer(self, mode: str) -> None:
        """Switch only the drawing surface; retain all pole-figure state."""
        if mode not in {"matplotlib", "pyqtgraph"}:
            raise ValueError(mode)
        if mode == "pyqtgraph" and self.pyqtgraph_plot is None:
            from .pyqtgraph_pole import PyQtGraphPolePlot

            self.pyqtgraph_plot = PyQtGraphPolePlot(self)
            self.pyqtgraph_plot.drag_started.connect(self._pyqtgraph_drag_started)
            self.pyqtgraph_plot.drag_moved.connect(self._pyqtgraph_drag_moved)
            self.pyqtgraph_plot.drag_finished.connect(self._pyqtgraph_drag_finished)
            self.pyqtgraph_plot.plot_clicked.connect(self._pyqtgraph_clicked)
            self.pyqtgraph_plot.palette_changed.connect(
                self._pyqtgraph_palette_changed
            )
            self.plot_stack.addWidget(self.pyqtgraph_plot)
        self.plot_renderer = mode
        self.plot_stack.setCurrentWidget(
            self.matplotlib_plot
            if mode == "matplotlib"
            else self.pyqtgraph_plot
        )
        self.redraw()

    def _pyqtgraph_palette_changed(self) -> None:
        if self.plot_renderer == "pyqtgraph":
            self.redraw()

    def refresh_overlay_choices(self) -> None:
        self.overlay_document_map = {}
        if self.overlay_layer is not None:
            configure(self.overlay_combo, values=(), state="disabled")
            configure(self.add_overlay_button, state="disabled")
            configure(self.open_overlay_button, state="disabled")
            return
        configure(self.open_overlay_button, state="normal")
        documents = (
            list(self.overlay_documents_provider())
            if self.overlay_documents_provider is not None
            else []
        )
        candidates = [
            document
            for document in documents
            if getattr(document, "kind", None) in {"cif", "cell_phase"}
        ]
        counts: dict[str, int] = {}
        for document in candidates:
            counts[document.name] = counts.get(document.name, 0) + 1
        for document in candidates:
            label = document.name
            if counts[label] > 1:
                label = f"{label} — {document.source.name}"
            self.overlay_document_map[label] = document
        values = tuple(self.overlay_document_map)
        configure(self.overlay_combo, 
            values=values,
            state="readonly" if values else "disabled",
        )
        configure(self.add_overlay_button, 
            state="normal" if values else "disabled"
        )
        if self.overlay_choice_var.get() not in self.overlay_document_map:
            self.overlay_choice_var.set(values[0] if values else "")
        self.overlay_combo.setCurrentText(self.overlay_choice_var.get())

    def add_selected_overlay(self) -> None:
        document = self.overlay_document_map.get(self.overlay_choice_var.get())
        if document is None:
            self.refresh_overlay_choices()
            document = self.overlay_document_map.get(
                self.overlay_choice_var.get()
            )
        if document is None:
            return
        if self.on_add_overlay is not None:
            self.on_add_overlay(document.uid)
        else:
            self.load_overlay_document(document.payload)

    @staticmethod
    def _percentage(variable, default: float, lower: float, upper: float) -> float:
        try:
            value = float(variable.get())
        except (ValueError, TypeError):
            return default
        return min(upper, max(lower, value))

    def global_point_scale_changed(self, value) -> None:
        try:
            percentage = min(300.0, max(25.0, float(value)))
        except (TypeError, ValueError, TypeError):
            percentage = 100.0
        self.global_size_label_var.set(f"{percentage:.0f}%")
        if hasattr(self, "_display_frames"):
            self._display_frames.request()

    def labels_visibility_changed(self) -> None:
        configure(self.label_leaders_check, 
            state="normal" if self.labels_var.get() else "disabled"
        )
        self.redraw()

    def _update_basis_control_state(self) -> None:
        enabled = self.show_structure_var.get() and self.crystal is not None
        configure(self.basis_check, state="normal" if enabled else "disabled")

    def structure_visibility_changed(self) -> None:
        self._update_basis_control_state()
        self.redraw()

    def _joint_rotation_enabled(self) -> bool:
        return bool(
            self.overlay_layer is not None
            and self.overlay_layer.coupled_to_primary
        )

    def _update_drag_help(self) -> None:
        if self.overlay_layer is None:
            key = (
                "text.dragging_inside_the_circle_freely_rotates_the_crystal_"
                "clicking_a_pole_shows_its_data_on_the_right"
            )
        elif self._joint_rotation_enabled():
            key = "pole.mouse_rotation_moves_both_phases"
        else:
            key = (
                "text.mouse_rotation_is_disabled_while_two_phases_are_"
                "overlaid_use_the_separate_numerical_rotations_for_each_phase"
            )
        configure(self.drag_help_label, text=tr(key))

    def joint_rotation_changed(self) -> None:
        if self.overlay_layer is not None:
            self.overlay_layer.coupled_to_primary = bool(
                self.joint_rotation_var.get()
            )
        self._update_drag_help()
        self.redraw()

    def _set_primary_orientation(
        self,
        *,
        base_rotation: np.ndarray | None = None,
        user_rotation: np.ndarray | None = None,
        update_overlay_entries: bool = True,
    ) -> None:
        before = self.user_rotation @ self.base_rotation
        if base_rotation is not None:
            self.base_rotation = np.asarray(base_rotation, dtype=float)
        if user_rotation is not None:
            self.user_rotation = np.asarray(user_rotation, dtype=float)
        after = self.user_rotation @ self.base_rotation
        if self._joint_rotation_enabled():
            layer = self.overlay_layer
            layer.user_rotation = follow_orientation_change(
                layer.user_rotation,
                before,
                after,
            )
            if update_overlay_entries:
                self.update_overlay_rotation_entries()

    def _set_multiphase_controls(self, enabled: bool) -> None:
        if enabled:
            if self._single_colour_mode is None:
                self._single_colour_mode = self.color_mode_var.get()
            self.color_mode_var.set("uniform")
            configure(self.d_colour_radio, state="disabled")
            configure(self.intensity_colour_radio, state="disabled")
            section_titles = (
                (self.center_section, "Центрирование первой фазы"),
                (self.rotation_section, "Абсолютный поворот первой фазы"),
                (
                    self.relative_rotation_section,
                    "Относительный поворот первой фазы",
                ),
            )
        else:
            configure(self.d_colour_radio, state="normal")
            cell_only = bool(
                self.cif_document is not None
                and getattr(self.cif_document, "is_cell_only", False)
            )
            configure(self.intensity_colour_radio, 
                state="disabled" if cell_only else "normal"
            )
            restored = self._single_colour_mode or "uniform"
            if restored == "intensity" and cell_only:
                restored = "uniform"
            self.color_mode_var.set(restored)
            self._single_colour_mode = None
            section_titles = (
                (self.center_section, "Центрирование по полюсу"),
                (self.rotation_section, "Абсолютный поворот кристалла"),
                (
                    self.relative_rotation_section,
                    "Относительный поворот кристалла",
                ),
            )
        for section, title in section_titles:
            section.title_source = title
            section.set_title(translate_text(section.title_source))
        self._update_drag_help()

    def load_overlay_document(self, document) -> None:
        if self.cif_document is None:
            self.load_document(document)
            return
        try:
            d_lower, d_upper = self.get_d_range()
            wavelength = self.get_wavelength()
            centre = (0, 1, 0)
            layer = CalculatedPoleLayer(
                document=document,
                colour=self.overlay_colour_var.get(),
                opacity_percent=self._percentage(
                    self.overlay_opacity_var, 70.0, 10.0, 100.0
                ),
                size_percent=self._percentage(
                    self.overlay_size_var, 100.0, 10.0, 300.0
                ),
                center_hkl=centre,
                base_rotation=base_orientation(document.crystal, centre),
                selected_hkl=centre,
                coupled_to_primary=False,
            )
            layer.reflections = available_reflections(
                document.crystal, d_lower, d_upper, wavelength
            )
        except ValueError as exc:
            show_error(
                localised(
                    "Could not add phase",
                    "Impossible d’ajouter la phase",
                    "Не удалось добавить фазу",
                ),
                str(exc),
                parent=self.root,
            )
            return
        self.overlay_layer = layer
        self.joint_rotation_var.set(False)
        self.overlay_name_var.set(document.name)
        for variable, value in zip(self.overlay_hkl_vars, centre):
            variable.set(str(value))
        self.update_overlay_rotation_entries()
        configure(self.remove_overlay_button, state="normal")
        self._set_overlay_settings_visible(True)
        self._set_multiphase_controls(True)
        self.refresh_overlay_choices()
        self.redraw()

    def remove_overlay(self, *, notify: bool = True) -> None:
        layer = self.overlay_layer
        if layer is None:
            return
        self.overlay_layer = None
        self.selected_layer_index = 0
        self.joint_rotation_var.set(False)
        self.overlay_name_var.set("")
        configure(self.remove_overlay_button, state="disabled")
        self._set_overlay_settings_visible(False)
        self._set_multiphase_controls(False)
        self.refresh_overlay_choices()
        self.redraw()
        if notify and self.on_remove_overlay is not None:
            self.on_remove_overlay(layer.document)

    def load_cif(self, path: str) -> None:
        try:
            try:
                from ..cif_document import load_cif_document
            except ImportError:  # pragma: no cover
                from xrd_workbench.cif_document import load_cif_document
            self.load_document(load_cif_document(path))
            return
        except Exception as exc:
            show_error(
                tr("text.could_not_open_cif"),
                localised(
                    "The CIF could not be read. Check the unit cell, atom sites "
                    "and explicit symmetry operations.\n\nDetails: ",
                    "Le CIF n’a pas pu être lu. Vérifiez la maille, les positions "
                    "atomiques et les opérations de symétrie explicites.\n\nDétails : ",
                    "Не удалось прочитать CIF. Проверьте параметры ячейки, "
                    "позиции атомов и явные операции симметрии.\n\nПодробности: ",
                )
                + str(exc),
                parent=self.root,
            )
            return

    def load_document(self, document) -> None:
        self._frames.cancel()
        self._display_frames.cancel()
        if self.overlay_layer is not None:
            self.remove_overlay(notify=False)
        self.reflections = []
        self.points = []
        self.point_groups = []
        self.selected_hkl = None
        self.selected_layer_index = 0
        self.intensity_by_spacing = None
        self.cif_document = document
        crystal = document.crystal
        self.crystal = crystal
        self._update_basis_control_state()
        self.path_var.set(document.source.name)
        self.structure_var.set(
            f"{crystal.formula}; {crystal.space_group}\n"
            f"a = {crystal.a:.4f} Å, b = {crystal.b:.4f} Å, "
            f"c = {crystal.c:.4f} Å"
        )
        cell_only = bool(getattr(document, "is_cell_only", False))
        configure(self.intensity_colour_radio, 
            state="disabled" if cell_only else "normal"
        )
        if cell_only and self.color_mode_var.get() == "intensity":
            self.color_mode_var.set("uniform")
        configure(self.center_prompt, 
            text=(
                tr("text.select_a_pole_not_systematically_forbidden")
                if cell_only
                else tr("text.select_an_allowed_pole")
            )
        )
        configure(self.range_button, 
            text=(
                tr("text.plot_reflections_not_systematically_forbidden")
                if cell_only
                else tr("text.plot_all_allowed_reflections")
            )
        )
        self.user_rotation = np.eye(3)
        self.base_rotation = base_orientation(crystal, self.center_hkl)
        self.update_rotation_entries()
        self.radiation_changed(rebuild=False)
        self.refresh_center_list()
        self.rebuild_reflections()
        self.refresh_overlay_choices()

    def clear_document(self) -> None:
        self._frames.cancel()
        self._display_frames.cancel()
        if self.overlay_layer is not None:
            self.remove_overlay(notify=False)
        self.cif_document = None
        self.crystal = None
        self._update_basis_control_state()
        self.reflections = []
        self.points = []
        self.point_groups = []
        self.selected_hkl = None
        self.selected_layer_index = 0
        self.intensity_by_spacing = None
        self.path_var.set_key("text.no_cif_loaded")
        self.structure_var.set("")
        configure(self.center_combo, values=())
        self.center_combo.setCurrentText("")
        configure(self.intensity_colour_radio, state="normal")
        configure(self.center_prompt, text=tr("text.select_an_allowed_pole"))
        configure(self.range_button, 
            text=tr("text.plot_all_allowed_reflections")
        )
        self.status_var.set("")
        configure(self.info_text, state="normal")
        self.info_text.clear()
        configure(self.info_text, state="disabled")
        self.refresh_overlay_choices()
        self.redraw()

    def remove_document(self, document) -> None:
        """Remove one assigned phase, promoting the overlay when necessary."""

        if self.overlay_layer is not None and self.overlay_layer.document is document:
            if self.cif_document is document:
                self.remove_overlay(notify=False)
                self.clear_document()
                return
            self.remove_overlay(notify=False)
            return
        if self.cif_document is not document:
            return
        if self.overlay_layer is None:
            self.clear_document()
            return
        promoted = self.overlay_layer
        self.overlay_layer = None
        self.joint_rotation_var.set(False)
        self.overlay_name_var.set("")
        configure(self.remove_overlay_button, state="disabled")
        self._set_overlay_settings_visible(False)
        self._set_multiphase_controls(False)
        self.load_document(promoted.document)
        self.center_hkl = promoted.center_hkl
        self.base_rotation = promoted.base_rotation
        self.user_rotation = promoted.user_rotation
        self.reflections = promoted.reflections
        self.selected_hkl = promoted.selected_hkl
        self.selected_layer_index = 0
        self.primary_colour_var.set(promoted.colour)
        self.primary_opacity_var.set(promoted.opacity_percent)
        self.primary_size_var.set(promoted.size_percent)
        for variable, value in zip(
            (self.h_var, self.k_var, self.l_var), self.center_hkl
        ):
            variable.set(str(value))
        self.update_rotation_entries()
        self.refresh_overlay_choices()
        self.redraw()

    def get_wavelength(self) -> float:
        if self.radiations_provider is not None:
            radiations = self.get_radiations()
            return min(wavelength for _name, wavelength, _weight in radiations)
        try:
            wavelength = float(self.wavelength_var.get().replace(",", "."))
        except ValueError as exc:
            raise ValueError(
                localised(
                    "The wavelength must be numeric.",
                    "La longueur d’onde doit être numérique.",
                    "Длина волны должна быть числом.",
                )
            ) from exc
        if wavelength <= 0:
            raise ValueError(
                localised(
                    "The wavelength must be positive.",
                    "La longueur d’onde doit être positive.",
                    "Длина волны должна быть положительной.",
                )
            )
        return wavelength

    def get_radiations(self) -> list[tuple[str, float, float]]:
        if self.radiations_provider is None:
            return [("Custom", self.get_wavelength(), 1.0)]
        radiations = list(self.radiations_provider())
        if not radiations:
            raise ValueError(
                localised(
                    "No spectral line is selected.",
                    "Aucune raie spectrale n’est sélectionnée.",
                    "Не выбрана ни одна спектральная линия.",
                )
            )
        return radiations

    def radiation_changed(self, rebuild: bool = True) -> None:
        self.intensity_by_spacing = None
        if self.overlay_layer is not None:
            self.overlay_layer.intensity_by_spacing = None
        if self.radiations_provider is not None:
            try:
                wavelength = min(item[1] for item in self.get_radiations())
                self.wavelength_var.set(f"{wavelength:.5f}")
            except ValueError:
                return
        if rebuild and self.crystal is not None:
            self.rebuild_reflections()

    def _ensure_intensities(self, show_errors: bool = True) -> bool:
        return self._ensure_layer_intensities(0, show_errors=show_errors)

    def _ensure_layer_intensities(
        self,
        layer_index: int,
        *,
        show_errors: bool = True,
    ) -> bool:
        if layer_index == 0:
            document = self.cif_document
            cached = self.intensity_by_spacing
        elif layer_index == 1 and self.overlay_layer is not None:
            document = self.overlay_layer.document
            cached = self.overlay_layer.intensity_by_spacing
        else:
            return False
        if document is None or getattr(document, "is_cell_only", False):
            return False
        if cached is not None:
            return True
        try:
            calculated = calculated_intensity_by_spacing(
                document.diffraction,
                self.get_radiations(),
            )
        except Exception as exc:
            calculated = {}
            if show_errors:
                show_error(
                    localised(
                        "Intensity calculation error",
                        "Erreur de calcul de l’intensité",
                        "Ошибка расчёта интенсивности",
                    ),
                    str(exc),
                    parent=self.root,
                )
        if layer_index == 0:
            self.intensity_by_spacing = calculated
        elif self.overlay_layer is not None:
            self.overlay_layer.intensity_by_spacing = calculated
        return bool(calculated)

    def change_colour_mode(self) -> None:
        if self.overlay_layer is not None:
            self.color_mode_var.set("uniform")
            self.redraw()
            return
        if self.color_mode_var.get() == "intensity" and not self._ensure_intensities():
            self.color_mode_var.set("uniform")
        self.redraw()

    def _point_intensity(
        self,
        point: PolePoint,
        layer_index: int = 0,
    ) -> float | None:
        if layer_index == 0:
            intensity_by_spacing = self.intensity_by_spacing
        elif layer_index == 1 and self.overlay_layer is not None:
            intensity_by_spacing = self.overlay_layer.intensity_by_spacing
        else:
            intensity_by_spacing = None
        if intensity_by_spacing is None:
            return None
        return intensity_by_spacing.get(
            round(1.0 / (point.d_spacing * point.d_spacing), 8)
        )

    def get_d_range(self) -> tuple[float, float]:
        try:
            d_lower = float(self.d_lower_var.get().replace(",", "."))
            d_upper = float(self.d_upper_var.get().replace(",", "."))
        except ValueError as exc:
            raise ValueError(
                localised(
                    "The d limits must be numeric.",
                    "Les limites de d doivent être numériques.",
                    "Границы d должны быть числами.",
                )
            ) from exc
        if d_lower <= 0 or d_upper <= 0:
            raise ValueError(
                localised(
                    "The d limits must be positive.",
                    "Les limites de d doivent être positives.",
                    "Границы d должны быть положительными.",
                )
            )
        if d_lower > d_upper:
            raise ValueError(
                localised(
                    "The lower d limit cannot exceed the upper limit.",
                    "La limite inférieure de d ne peut pas dépasser la limite supérieure.",
                    "Нижняя граница d не может быть больше верхней.",
                )
            )
        return d_lower, d_upper

    def refresh_center_list(self) -> None:
        if self.crystal is None:
            return
        try:
            max_index = int(self.max_index_var.get())
            wavelength = self.get_wavelength()
            if not 1 <= max_index <= 20:
                raise ValueError(
                    localised(
                        "The maximum index must be between 1 and 20.",
                        "L’indice maximal doit être compris entre 1 et 20.",
                        "Максимальный индекс должен быть от 1 до 20.",
                    )
                )
        except (ValueError, TypeError) as exc:
            self.status_var.set(str(exc))
            return

        entries: list[tuple[float, tuple[int, int, int], str]] = []
        for h in range(max_index + 1):
            for k in range(max_index + 1):
                for l in range(max_index + 1):
                    hkl = (h, k, l)
                    if hkl == (0, 0, 0):
                        continue
                    if self.crystal.is_systematically_absent(hkl):
                        continue
                    two_theta = self.crystal.two_theta(hkl, wavelength)
                    if two_theta is None:
                        continue
                    d_value = self.crystal.d_spacing(hkl)
                    display = (
                        f"{format_hkl(hkl)}   "
                        f"2θ={two_theta:.3f}°   d={d_value:.4f} Å"
                    )
                    entries.append((two_theta, hkl, display))
        entries.sort(key=lambda item: (item[0], item[1]))
        self.center_map = {display: hkl for _, hkl, display in entries}
        values = [display for _, _, display in entries]
        configure(self.center_combo, values=values)

        wanted = tuple(abs(value) for value in self.center_hkl)
        for display, hkl in self.center_map.items():
            if hkl == wanted:
                self.center_combo.setCurrentText(display)
                break
        else:
            self.center_combo.setCurrentText("")

    def choose_center(self, _event=None) -> None:
        display = self.center_combo.currentText()
        hkl = self.center_map.get(display)
        if hkl is None:
            return
        for variable, value in zip((self.h_var, self.k_var, self.l_var), hkl):
            variable.set(str(value))
        self.apply_center()

    def _read_hkl_variables(self, variables) -> tuple[int, int, int]:
        values = []
        for variable in variables:
            text = variable.get().strip()
            if not re.fullmatch(r"[+-]?\d+", text):
                raise ValueError(
                    localised(
                        "The h, k and l indices must be integers.",
                        "Les indices h, k et l doivent être des nombres entiers.",
                        "Индексы h, k, l должны быть целыми числами.",
                    )
                )
            values.append(int(text))
        hkl = tuple(values)
        if hkl == (0, 0, 0):
            raise ValueError(
                localised(
                    "The (0 0 0) reflection does not exist.",
                    "La réflexion (0 0 0) n’existe pas.",
                    "Отражение (0 0 0) не существует.",
                )
            )
        return hkl

    def read_center(self) -> tuple[int, int, int]:
        return self._read_hkl_variables((self.h_var, self.k_var, self.l_var))

    def rebuild_reflections(self) -> None:
        if self.crystal is None:
            return
        try:
            d_lower, d_upper = self.get_d_range()
            wavelength = self.get_wavelength()
            reflections = available_reflections(
                self.crystal,
                d_lower,
                d_upper,
                wavelength,
            )
            overlay_reflections = (
                available_reflections(
                    self.overlay_layer.crystal,
                    d_lower,
                    d_upper,
                    wavelength,
                )
                if self.overlay_layer is not None
                else None
            )
        except ValueError as exc:
            show_error(
                tr("text.invalid_reflection_range"),
                str(exc),
                parent=self.root,
            )
            return
        self.reflections = reflections
        if self.overlay_layer is not None and overlay_reflections is not None:
            self.overlay_layer.reflections = overlay_reflections
            self.overlay_layer.intensity_by_spacing = None
        self.intensity_by_spacing = None
        self.selected_hkl = self.center_hkl
        self.selected_layer_index = 0
        self.refresh_center_list()
        self.redraw()

    def apply_center(self) -> None:
        if self.crystal is None:
            return
        try:
            center_hkl = self.read_center()
            base = base_orientation(self.crystal, center_hkl)
        except ValueError as exc:
            show_error(
                localised(
                    "Invalid h k l",
                    "h k l incorrects",
                    "Некорректные h k l",
                ),
                str(exc),
                parent=self.root,
            )
            return
        self.center_hkl = center_hkl
        self._set_primary_orientation(
            base_rotation=base,
            user_rotation=np.eye(3),
        )
        self.selected_hkl = center_hkl
        self.selected_layer_index = 0
        self.update_rotation_entries()
        if self.reflections:
            self.redraw()
        else:
            self.rebuild_reflections()

    def apply_overlay_center(self) -> None:
        layer = self.overlay_layer
        if layer is None:
            return
        try:
            centre = self._read_hkl_variables(self.overlay_hkl_vars)
            layer.base_rotation = base_orientation(layer.crystal, centre)
        except ValueError as exc:
            show_error(
                localised(
                    "Invalid h k l",
                    "h k l incorrects",
                    "Некорректные h k l",
                ),
                str(exc),
                parent=self.root,
            )
            return
        layer.center_hkl = centre
        layer.user_rotation = np.eye(3)
        layer.selected_hkl = centre
        self.selected_layer_index = 1
        self.update_overlay_rotation_entries()
        self.redraw()

    def read_rotation_angles(
        self,
        variables: Sequence[Value],
    ) -> list[float] | None:
        try:
            return [
                float(variable.get().strip().replace(",", "."))
                for variable in variables
            ]
        except ValueError:
            show_error(
                tr("text.invalid_angle"),
                tr("text.x_y_and_z_angles_must_be_numeric"),
                parent=self.root,
            )
            return None

    def apply_exact_rotation(self) -> None:
        angles = self.read_rotation_angles(self.rotation_vars)
        if angles is None:
            return
        self._set_primary_orientation(user_rotation=euler_matrix(*angles))
        self.update_rotation_entries()
        self.redraw()

    def apply_relative_rotation(self) -> None:
        angles = self.read_rotation_angles(self.relative_rotation_vars)
        if angles is None:
            return
        self._set_primary_orientation(
            user_rotation=euler_matrix(*angles) @ self.user_rotation
        )
        self.update_rotation_entries()
        for variable in self.relative_rotation_vars:
            variable.set("0.0")
        self.redraw()

    def apply_overlay_exact_rotation(self) -> None:
        layer = self.overlay_layer
        if layer is None:
            return
        angles = self.read_rotation_angles(self.overlay_rotation_vars)
        if angles is None:
            return
        layer.user_rotation = euler_matrix(*angles)
        self.update_overlay_rotation_entries()
        self.redraw()

    def apply_overlay_relative_rotation(self) -> None:
        layer = self.overlay_layer
        if layer is None:
            return
        angles = self.read_rotation_angles(self.overlay_relative_rotation_vars)
        if angles is None:
            return
        layer.user_rotation = euler_matrix(*angles) @ layer.user_rotation
        self.update_overlay_rotation_entries()
        for variable in self.overlay_relative_rotation_vars:
            variable.set("0.0")
        self.redraw()

    def update_rotation_entries(self) -> None:
        angles = matrix_to_euler(self.user_rotation)
        for variable, angle in zip(self.rotation_vars, angles):
            if abs(angle) < 5e-10:
                angle = 0.0
            variable.set(f"{angle:.3f}")

    def update_overlay_rotation_entries(self) -> None:
        layer = self.overlay_layer
        if layer is None:
            angles = (0.0, 0.0, 0.0)
        else:
            angles = matrix_to_euler(layer.user_rotation)
        for variable, angle in zip(self.overlay_rotation_vars, angles):
            if abs(angle) < 5e-10:
                angle = 0.0
            variable.set(f"{angle:.3f}")

    def reset_rotation(self) -> None:
        self._set_primary_orientation(user_rotation=np.eye(3))
        self.selected_hkl = self.center_hkl
        self.selected_layer_index = 0
        self.update_rotation_entries()
        self.redraw()

    def align_selected_pole(
        self,
        target_phi: float,
        *,
        layer_index: int = 0,
    ) -> None:
        if layer_index == 0:
            selected_hkl = self.selected_hkl
            point_groups = self.point_groups
        elif layer_index == 1 and self.overlay_layer is not None:
            selected_hkl = self.overlay_layer.selected_hkl
            point_groups = self.overlay_layer.point_groups
        else:
            selected_hkl = None
            point_groups = ()
        point = None
        if selected_hkl is not None:
            for group in point_groups:
                for candidate in group:
                    if candidate.hkl == selected_hkl:
                        point = candidate
                        break
                if point is not None:
                    break
        if point is None:
            show_info(
                localised(
                    "No selected pole",
                    "Aucun pôle sélectionné",
                    "Полюс не выбран",
                ),
                localised(
                    "Click a displayed pole first.",
                    "Cliquez d’abord sur un pôle affiché.",
                    "Сначала щёлкните по отображаемому полюсу.",
                ),
                parent=self.root,
            )
            return
        if math.hypot(point.direction[0], point.direction[1]) < 1e-10:
            show_info(
                localised(
                    "Azimuth is undefined",
                    "Azimut indéfini",
                    "Азимут не определён",
                ),
                localised(
                    "The selected pole is at the centre and has no unique azimuth.",
                    "Le pôle sélectionné est au centre et n’a pas d’azimut unique.",
                    "Выбранный полюс находится в центре и не имеет единственного азимута.",
                ),
                parent=self.root,
            )
            return
        alignment = in_plane_alignment(point.phi, target_phi)
        if layer_index == 0:
            self._set_primary_orientation(
                user_rotation=alignment @ self.user_rotation
            )
            self.update_rotation_entries()
        else:
            self.overlay_layer.user_rotation = (
                alignment @ self.overlay_layer.user_rotation
            )
            self.update_overlay_rotation_entries()
        self.selected_layer_index = layer_index
        self.redraw()

    def prepare_plot_axes(self) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        show = self.show_structure_var.get()
        self.structure_viewer.setVisible(show)
        self.structure_viewer.display_section.setVisible(show)
        if self.crystal is None:
            self.structure_viewer.sync_document(None, np.eye(3))

    def draw_grid(self) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-1.10, 1.10)
        self.ax.set_ylim(-1.10, 1.10)
        self.ax.axis("off")
        self.ax.add_patch(
            Circle((0, 0), 1.0, fill=False, linewidth=1.5, color="#303030")
        )

        for chi in (15, 30, 45, 60, 75):
            chi_rad = math.radians(chi)
            if choice_code("projection", self.projection_var.get()) == "equal_area":
                radius = math.sqrt(2.0) * math.sin(chi_rad / 2.0)
            else:
                radius = math.tan(chi_rad / 2.0)
            self.ax.add_patch(
                Circle(
                    (0, 0),
                    radius,
                    fill=False,
                    linewidth=0.55,
                    color="#c2c2c2",
                )
            )
            if self.angle_labels_var.get():
                self.ax.text(
                    radius + 0.012,
                    0.012,
                    f"{chi}°",
                    color="#777777",
                    fontsize=8,
                    va="bottom",
                )

        for phi in range(0, 360, POLE_AZIMUTH_GRID_STEP_DEG):
            angle = math.radians(phi)
            self.ax.plot(
                [0, math.cos(angle)],
                [0, math.sin(angle)],
                color="#d0d0d0",
                linewidth=0.5,
                zorder=0,
            )
        self.ax.text(1.045, 0.0, "X", ha="center", va="center")
        self.ax.text(0.0, 1.055, "Y", ha="center", va="center")

    def draw_structure(self, orientation: np.ndarray, *, preview=False) -> None:
        if not self.show_structure_var.get():
            return
        self.structure_viewer.sync_document(self.cif_document, orientation)
        self.structure_viewer.canvas.set_display_options(
            show_basis=self.basis_visible.get(),
            show_polyhedra=False,
            rotation_enabled=(
                self.crystal is not None
                and (
                    self.overlay_layer is None
                    or self._joint_rotation_enabled()
                )
            ),
        )
        self._update_basis_control_state()

    def rotate_structure_from_mouse(self, orientation: np.ndarray) -> None:
        if self.crystal is None or (self.overlay_layer is not None and not self._joint_rotation_enabled()):
            return
        self._set_primary_orientation(
            user_rotation=user_rotation_from_display(
                orientation,
                self.base_rotation,
            ),
            update_overlay_entries=False,
        )
        self._frames.request()

    def finish_structure_rotation(self) -> None:
        self._frames.cancel()
        self.update_rotation_entries()
        if self._joint_rotation_enabled():
            self.update_overlay_rotation_entries()
        self.redraw()

    def refresh_atom_styles(self) -> None:
        self.structure_viewer.refresh_atom_styles()
        self.redraw()

    def _layer_context(self, layer_index: int):
        if layer_index == 0 and self.crystal is not None:
            return (
                self.cif_document,
                self.crystal,
                self.center_hkl,
                self.user_rotation @ self.base_rotation,
                self.point_groups,
                self.selected_hkl,
            )
        if layer_index == 1 and self.overlay_layer is not None:
            layer = self.overlay_layer
            return (
                layer.document,
                layer.crystal,
                layer.center_hkl,
                layer.orientation,
                layer.point_groups,
                layer.selected_hkl,
            )
        return None

    def _selected_group(self, layer_index: int):
        context = self._layer_context(layer_index)
        if context is None:
            return None
        _document, crystal, _centre, orientation, groups, selected_hkl = context
        if selected_hkl is None:
            return None
        target = orientation @ crystal.reciprocal_vector(selected_hkl)
        target /= np.linalg.norm(target)
        if target[2] < -1e-10:
            target = -target
        elif abs(target[2]) <= 1e-10:
            target[2] = 0.0
        for group in groups:
            if float(np.dot(group[0].direction, target)) > 1.0 - 1e-8:
                return group
        return None

    def _draw_coincident_group_markers(
        self,
        groups: Sequence[Sequence[PolePoint]],
        marker_sizes: Sequence[float],
    ) -> None:
        indices = [index for index, group in enumerate(groups) if len(group) > 1]
        if not indices:
            return
        positions = [pole_display_position(groups[index][0]) for index in indices]
        ring_sizes = [
            (math.sqrt(float(marker_sizes[index])) + 4.0) ** 2
            for index in indices
        ]
        self.ax.scatter(
            [position[0] for position in positions],
            [position[1] for position in positions],
            s=ring_sizes,
            facecolors="none",
            edgecolors="#555555",
            linewidths=0.75,
            clip_on=False,
            zorder=3.3,
        )

    def _draw_pole_labels(
        self,
        primary_marker_sizes: Sequence[float],
        overlay_marker_sizes: Sequence[float],
    ) -> None:
        entries = []
        selected_group = self._selected_group(self.selected_layer_index)
        for layer_index, groups, colour, marker_sizes in (
            (0, self.point_groups, "#202020", primary_marker_sizes),
            (
                1,
                self.overlay_layer.point_groups
                if self.overlay_layer is not None
                else (),
                self.overlay_layer.colour
                if self.overlay_layer is not None
                else "#202020",
                overlay_marker_sizes,
            ),
        ):
            for group_index, group in enumerate(groups):
                point = group[0]
                entries.append(
                    {
                        "layer_index": layer_index,
                        "group": group,
                        "group_index": group_index,
                        "text": format_hkl(point.hkl),
                        "colour": colour,
                        "position": pole_display_position(point),
                        "marker_size": float(marker_sizes[group_index]),
                        "selected": group is selected_group,
                    }
                )
        entries.sort(
            key=lambda entry: (
                not entry["selected"],
                entry["layer_index"],
                -entry["group"][0].d_spacing,
                entry["group"][0].hkl,
            )
        )
        if not entries:
            return

        anchors = self.ax.transData.transform(
            np.asarray([entry["position"] for entry in entries], dtype=float)
        )
        dpi_scale = self.figure.dpi / 72.0
        protected_boxes = []
        for anchor, entry in zip(anchors, entries):
            radius = math.sqrt(entry["marker_size"]) * dpi_scale / 2.0 + 2.0
            protected_boxes.append(
                (
                    anchor[0] - radius,
                    anchor[1] - radius,
                    anchor[0] + radius,
                    anchor[1] + radius,
                )
            )
        renderer = self.canvas.get_renderer()
        legend = self.ax.get_legend()
        if legend is not None:
            legend_box = legend.get_window_extent(renderer)
            protected_boxes.append(
                (legend_box.x0, legend_box.y0, legend_box.x1, legend_box.y1)
            )
        box_sizes = [
            (max(24.0, len(entry["text"]) * 6.2), 13.0)
            for entry in entries
        ]
        axes_box = self.ax.bbox
        show_leaders = self.label_leaders_var.get()
        placements = place_label_boxes(
            [tuple(anchor) for anchor in anchors],
            box_sizes,
            (axes_box.x0 + 3, axes_box.y0 + 3, axes_box.x1 - 3, axes_box.y1 - 3),
            protected_boxes=protected_boxes,
            required_indices={
                index for index, entry in enumerate(entries) if entry["selected"]
            },
            anchor_clearances=(
                None
                if show_leaders
                else [
                    math.sqrt(entry["marker_size"]) * dpi_scale / 2.0 + 2.0
                    for entry in entries
                ]
            ),
            candidate_gaps=(
                (6.0, 16.0, 28.0, 44.0, 64.0)
                if show_leaders
                else (1.5, 4.0, 7.0)
            ),
        )
        inverse = self.ax.transData.inverted()
        for anchor, entry, box_size, placement in zip(
            anchors, entries, box_sizes, placements
        ):
            if placement is None:
                continue
            text_position = inverse.transform(placement)
            width, height = box_size
            horizontal_gap = max(
                placement[0] - anchor[0],
                anchor[0] - (placement[0] + width),
                0.0,
            )
            vertical_gap = max(
                placement[1] - anchor[1],
                anchor[1] - (placement[1] + height),
                0.0,
            )
            leader = math.hypot(horizontal_gap, vertical_gap) > 10.0
            self.ax.annotate(
                entry["text"],
                entry["position"],
                xytext=text_position,
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=9,
                color=entry["colour"],
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": entry["colour"],
                        "linewidth": 0.55,
                        "shrinkA": 1.0,
                        "shrinkB": 2.0,
                    }
                    if leader and show_leaders
                    else None
                ),
                annotation_clip=False,
                zorder=5,
            )

    def _plot_title(self) -> str:
        try:
            d_lower, d_upper = self.get_d_range()
        except ValueError:
            d_lower, d_upper = 0.0, 0.0
        projection_name = localised(
            "equal-area"
            if choice_code("projection", self.projection_var.get()) == "equal_area"
            else "stereographic",
            "équivalente"
            if choice_code("projection", self.projection_var.get()) == "equal_area"
            else "stéréographique",
            "равноплощадная"
            if choice_code("projection", self.projection_var.get()) == "equal_area"
            else "стереографическая",
        )
        if getattr(self.cif_document, "is_cell_only", False):
            return localised(
                f"Reflections not systematically forbidden · d = {d_lower:g}–{d_upper:g} Å · {projection_name} projection",
                f"Réflexions non interdites systématiquement · d = {d_lower:g}–{d_upper:g} Å · projection {projection_name}",
                f"Отражения, не запрещённые систематически · d = {d_lower:g}–{d_upper:g} Å · {projection_name} проекция",
            )
        return localised(
            f"All allowed reflections · d = {d_lower:g}–{d_upper:g} Å · {projection_name} projection",
            f"Toutes les réflexions autorisées · d = {d_lower:g}–{d_upper:g} Å · projection {projection_name}",
            f"Все разрешённые отражения · d = {d_lower:g}–{d_upper:g} Å · {projection_name} проекция",
        )

    def _plot_status(self) -> str:
        try:
            d_lower, _d_upper = self.get_d_range()
            wavelength = self.get_wavelength()
        except ValueError:
            d_lower, wavelength = 0.0, 0.0
        status = localised(
            f"Centre: {format_hkl(self.center_hkl)}. "
            f"Reflections: {len(self.reflections)}; distinct positions: "
            f"{len(self.point_groups)}.",
            f"Centre : {format_hkl(self.center_hkl)}. "
            f"Réflexions : {len(self.reflections)} ; positions distinctes : "
            f"{len(self.point_groups)}.",
            f"Центр: {format_hkl(self.center_hkl)}. "
            f"Отражений: {len(self.reflections)}; различимых положений: "
            f"{len(self.point_groups)}.",
        )
        if self.overlay_layer is not None:
            layer = self.overlay_layer
            status += localised(
                f" Overlay {layer.name}: {len(layer.reflections)} reflections; "
                f"{len(layer.point_groups)} distinct positions.",
                f" Superposition {layer.name} : {len(layer.reflections)} réflexions ; "
                f"{len(layer.point_groups)} positions distinctes.",
                f" Наложение {layer.name}: отражений {len(layer.reflections)}; "
                f"различимых положений {len(layer.point_groups)}.",
            )
        if wavelength and d_lower < wavelength / 2.0:
            status += localised(
                f" Reflections with d < λ/2 = {wavelength / 2.0:.4f} Å "
                "are excluded by Bragg’s law.",
                f" Les réflexions avec d < λ/2 = {wavelength / 2.0:.4f} Å "
                "sont exclues par la loi de Bragg.",
                f" Отражения с d < λ/2 = {wavelength / 2.0:.4f} Å "
                "исключены по условию Брэгга.",
            )
        if self.crystal.is_systematically_absent(self.center_hkl):
            status += localised(
                " The centred pole is systematically absent and is used only "
                "as a geometric axis.",
                " Le pôle centré est systématiquement éteint et sert uniquement "
                "d’axe géométrique.",
                " Центрирующий полюс систематически погашен; "
                "он используется только как геометрическая ось.",
            )
        if not self.reflections:
            if getattr(self.cif_document, "is_cell_only", False):
                status += localised(
                    " There are no reflections not systematically forbidden in the selected range.",
                    " Il n’y a aucune réflexion non interdite systématiquement dans l’intervalle sélectionné.",
                    " В выбранном диапазоне нет отражений, не запрещённых систематически.",
                )
            else:
                status += localised(
                    " There are no allowed reflections in the selected range.",
                    " Aucune réflexion autorisée dans l’intervalle sélectionné.",
                    " В выбранном диапазоне разрешённых отражений нет.",
                )
        return status

    @staticmethod
    def _coincident_marker_data(groups, marker_sizes):
        indices = [index for index, group in enumerate(groups) if len(group) > 1]
        return (
            [pole_display_position(groups[index][0]) for index in indices],
            [float(marker_sizes[index]) for index in indices],
        )

    def _pyqtgraph_labels(self, primary_sizes, overlay_sizes) -> None:
        if not self.labels_var.get() or self.pyqtgraph_plot is None:
            return
        primary_colour = self.palette().text().color().name()
        selected_group = self._selected_group(self.selected_layer_index)
        entries = []
        for layer_index, groups, colour, sizes in (
            (0, self.point_groups, primary_colour, primary_sizes),
            (
                1,
                self.overlay_layer.point_groups
                if self.overlay_layer is not None
                else (),
                self.overlay_layer.colour
                if self.overlay_layer is not None
                else primary_colour,
                overlay_sizes,
            ),
        ):
            for index, group in enumerate(groups):
                entries.append(
                    {
                        "layer_index": layer_index,
                        "group": group,
                        "text": format_hkl(group[0].hkl),
                        "colour": colour,
                        "position": pole_display_position(group[0]),
                        "marker_size": float(sizes[index]),
                        "selected": group is selected_group,
                    }
                )
        entries.sort(
            key=lambda entry: (
                not entry["selected"],
                entry["layer_index"],
                -entry["group"][0].d_spacing,
                entry["group"][0].hkl,
            )
        )
        if not entries:
            return
        plot = self.pyqtgraph_plot
        anchors = plot.data_to_scene([entry["position"] for entry in entries])
        protected = []
        for anchor, entry in zip(anchors, entries):
            radius = math.sqrt(entry["marker_size"]) / 2.0 + 2.0
            protected.append(
                (
                    anchor[0] - radius,
                    anchor[1] - radius,
                    anchor[0] + radius,
                    anchor[1] + radius,
                )
            )
        box_sizes = [
            (max(24.0, len(entry["text"]) * 6.2), 15.0)
            for entry in entries
        ]
        bounds = plot.scene_bounds()
        show_leaders = self.label_leaders_var.get()
        placements = place_label_boxes(
            [tuple(anchor) for anchor in anchors],
            box_sizes,
            (bounds[0] + 3.0, bounds[1] + 3.0, bounds[2] - 3.0, bounds[3] - 3.0),
            protected_boxes=protected,
            required_indices={
                index for index, entry in enumerate(entries) if entry["selected"]
            },
            anchor_clearances=(
                None
                if show_leaders
                else [
                    math.sqrt(entry["marker_size"]) / 2.0 + 2.0
                    for entry in entries
                ]
            ),
            candidate_gaps=(
                (6.0, 16.0, 28.0, 44.0, 64.0)
                if show_leaders
                else (1.5, 4.0, 7.0)
            ),
        )
        for anchor, entry, box_size, placement in zip(
            anchors, entries, box_sizes, placements
        ):
            if placement is None:
                continue
            label_position = plot.scene_to_data(placement)
            plot.add_label(
                label_position,
                entry["text"],
                entry["colour"],
                anchor=(0.0, 0.0),
            )
            width, height = box_size
            end_x = min(max(anchor[0], placement[0]), placement[0] + width)
            end_y = min(max(anchor[1], placement[1]), placement[1] + height)
            if show_leaders and math.hypot(end_x - anchor[0], end_y - anchor[1]) > 10.0:
                plot.add_leader(
                    entry["position"],
                    plot.scene_to_data((end_x, end_y)),
                    entry["colour"],
                )

    def _pyqtgraph_legend_entries(self, primary_alpha):
        if self.overlay_layer is None:
            return ()
        layer = self.overlay_layer
        primary_name = self.cif_document.name
        overlay_name = layer.name
        if layer.document is self.cif_document:
            primary_name = localised(
                f"{primary_name} — primary",
                f"{primary_name} — principale",
                f"{primary_name} — основная",
            )
            overlay_name = localised(
                f"{overlay_name} — second",
                f"{overlay_name} — seconde",
                f"{overlay_name} — вторая",
            )
        elif primary_name == overlay_name:
            primary_name = self.cif_document.source.name
            overlay_name = layer.document.source.name
        return (
            (primary_name, self.primary_colour_var.get(), primary_alpha),
            (overlay_name, layer.colour, layer.opacity),
        )

    def _redraw_pyqtgraph(self, *, preview=False) -> None:
        from .pyqtgraph_pole import (
            D_SPACING_STOPS,
            VIRIDIS_STOPS,
            colour_values,
        )

        self.prepare_plot_axes()
        plot = self.pyqtgraph_plot
        projection = choice_code("projection", self.projection_var.get())
        plot.begin_frame(
            projection=projection,
            angle_labels=self.angle_labels_var.get(),
            title=self._plot_title() if self.crystal is not None else "",
        )
        if self.crystal is None:
            return

        orientation = self.user_rotation @ self.base_rotation
        self.points = project_reflections(
            self.crystal,
            self.reflections,
            orientation,
            self.projection_var.get(),
        )
        self.point_groups = group_coincident_poles(self.points)
        representatives = [group[0] for group in self.point_groups]
        positions = [pole_display_position(point) for point in representatives]
        d_values = [point.d_spacing for point in representatives]
        marker_sizes = (
            marker_sizes_by_d(d_values)
            if self.size_by_d_var.get()
            else np.full(len(representatives), 58.0)
        )
        global_scale = self._percentage(
            self.global_size_var, 100.0, 25.0, 300.0
        ) / 100.0
        marker_sizes *= (
            self._percentage(self.primary_size_var, 100.0, 10.0, 300.0)
            / 100.0
            * global_scale
        )
        primary_alpha = self._percentage(
            self.primary_opacity_var, 100.0, 10.0, 100.0
        ) / 100.0
        mode = self.color_mode_var.get()
        mapped_colours = None
        if mode == "intensity" and self.intensity_by_spacing is None:
            self._ensure_intensities(show_errors=False)
        if mode == "intensity" and representatives:
            values = [
                max((self._point_intensity(point) or 0.0) for point in group)
                for group in self.point_groups
            ]
            mapped_colours = colour_values(
                values, VIRIDIS_STOPS, 0.0, 100.0, primary_alpha
            )
            plot.set_colour_scale(
                VIRIDIS_STOPS,
                0.0,
                100.0,
                localised(
                    "Calculated intensity, %",
                    "Intensité calculée, %",
                    "Расчётная интенсивность, %",
                ),
            )
        elif mode == "d" and representatives:
            d_min, d_max = min(d_values), max(d_values)
            if math.isclose(d_min, d_max):
                d_min -= 0.5
                d_max += 0.5
            mapped_colours = colour_values(
                d_values, D_SPACING_STOPS, d_min, d_max, primary_alpha
            )
            plot.set_colour_scale(D_SPACING_STOPS, d_min, d_max, "d, Å")
        plot.add_scatter(
            positions,
            marker_sizes,
            colour=self.primary_colour_var.get(),
            colours=mapped_colours,
            opacity=primary_alpha,
            outlined=not self.size_by_d_var.get(),
            z=3.0,
        )
        if self.coincident_outlines_var.get():
            ring_positions, ring_sizes = self._coincident_marker_data(
                self.point_groups, marker_sizes
            )
            plot.add_rings(
                ring_positions,
                ring_sizes,
                colour="#555555",
                width=0.75,
                extra=4.0,
                z=3.3,
            )

        overlay_sizes = np.empty(0, dtype=float)
        if self.overlay_layer is not None:
            layer = self.overlay_layer
            layer.colour = self.overlay_colour_var.get()
            layer.opacity_percent = self._percentage(
                self.overlay_opacity_var, 70.0, 10.0, 100.0
            )
            layer.size_percent = self._percentage(
                self.overlay_size_var, 100.0, 10.0, 300.0
            )
            layer.points = project_reflections(
                layer.crystal,
                layer.reflections,
                layer.orientation,
                self.projection_var.get(),
            )
            layer.point_groups = group_coincident_poles(layer.points)
            overlay_representatives = [group[0] for group in layer.point_groups]
            overlay_positions = [
                pole_display_position(point) for point in overlay_representatives
            ]
            overlay_d = [point.d_spacing for point in overlay_representatives]
            overlay_sizes = (
                marker_sizes_by_d(overlay_d)
                if self.size_by_d_var.get()
                else np.full(len(overlay_representatives), 58.0)
            )
            overlay_sizes *= layer.size_scale * global_scale
            plot.add_scatter(
                overlay_positions,
                overlay_sizes,
                colour=layer.colour,
                opacity=layer.opacity,
                outlined=not self.size_by_d_var.get(),
                z=3.1,
            )
            if self.coincident_outlines_var.get():
                ring_positions, ring_sizes = self._coincident_marker_data(
                    layer.point_groups, overlay_sizes
                )
                plot.add_rings(
                    ring_positions,
                    ring_sizes,
                    colour="#555555",
                    width=0.75,
                    extra=4.0,
                    z=3.3,
                )
            plot.set_legend(self._pyqtgraph_legend_entries(primary_alpha))

        selected_group = self._selected_group(self.selected_layer_index)
        if selected_group is not None:
            selected = selected_group[0]
            sizes = marker_sizes if self.selected_layer_index == 0 else overlay_sizes
            context = self._layer_context(self.selected_layer_index)
            group_index = context[4].index(selected_group)
            selected_size = float(sizes[group_index])
            plot.add_rings(
                [pole_display_position(selected)],
                [selected_size],
                colour="#d23b2d",
                width=2.0,
                extra=8.0,
                z=4.0,
            )
            self.show_information(
                selected_group,
                layer_index=self.selected_layer_index,
            )

        self._pyqtgraph_labels(marker_sizes, overlay_sizes)
        self.draw_structure(pole_display_orientation(orientation), preview=preview)
        self.status_var.set(self._plot_status())

    def _pyqtgraph_nearest_candidates(self, x: float, y: float):
        pixel_x, pixel_y = self.pyqtgraph_plot.pixel_size()
        pixel_x = max(pixel_x, 1e-12)
        pixel_y = max(pixel_y, 1e-12)
        candidates = []
        for layer_index in (0, 1):
            context = self._layer_context(layer_index)
            if context is None:
                continue
            for group in context[4]:
                px, py = pole_display_position(group[0])
                distance = math.hypot((px - x) / pixel_x, (py - y) / pixel_y)
                if distance <= 14.0:
                    candidates.append((distance, layer_index, group))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def _pyqtgraph_clicked(self, x: float, y: float) -> None:
        if math.hypot(x, y) > 1.05:
            return
        candidates = self._pyqtgraph_nearest_candidates(x, y)
        if not candidates:
            return
        nearest = candidates[0][0]
        ambiguous = [item for item in candidates if item[0] - nearest <= 3.0]
        if len(ambiguous) > 1:
            self._show_pole_candidate_menu(ambiguous, None)
        else:
            _distance, layer_index, group = candidates[0]
            self._select_pole_group(layer_index, group)

    def _pyqtgraph_drag_started(self, x: float, y: float) -> None:
        if math.hypot(x, y) > 1.05:
            return
        if self.overlay_layer is not None and not self._joint_rotation_enabled():
            self.last_arcball = None
            return
        self.last_arcball = self.plot_to_sphere(x, y)

    def _pyqtgraph_drag_moved(self, x: float, y: float) -> None:
        if self.last_arcball is None or math.hypot(x, y) > 1.05:
            return
        current = self.plot_to_sphere(x, y)
        delta = rotation_between(self.last_arcball, current)
        self._set_primary_orientation(
            user_rotation=delta @ self.user_rotation,
            update_overlay_entries=False,
        )
        self.last_arcball = current
        self._frames.request()

    def _pyqtgraph_drag_finished(self, x: float, y: float) -> None:
        del x, y
        if self.last_arcball is None:
            return
        self.last_arcball = None
        self._frames.cancel()
        self.update_rotation_entries()
        if self._joint_rotation_enabled():
            self.update_overlay_rotation_entries()
        self.redraw()

    def redraw(self, *, preview=False) -> None:
        if self.plot_renderer == "pyqtgraph":
            self._redraw_pyqtgraph(preview=preview)
            return
        self.prepare_plot_axes()
        self.draw_grid()
        if self.crystal is None:
            self.canvas.draw_idle()
            return

        orientation = self.user_rotation @ self.base_rotation
        self.points = project_reflections(
            self.crystal,
            self.reflections,
            orientation,
            self.projection_var.get(),
        )
        self.point_groups = group_coincident_poles(self.points)
        representatives = [group[0] for group in self.point_groups]
        display_positions = [
            pole_display_position(point) for point in representatives
        ]
        d_values = [point.d_spacing for point in representatives]
        marker_sizes = (
            marker_sizes_by_d(d_values)
            if self.size_by_d_var.get()
            else np.full(len(representatives), 58.0)
        )
        global_size_scale = self._percentage(
            self.global_size_var, 100.0, 25.0, 300.0
        ) / 100.0
        marker_sizes *= self._percentage(
            self.primary_size_var, 100.0, 10.0, 300.0
        ) / 100.0 * global_size_scale
        primary_alpha = self._percentage(
            self.primary_opacity_var, 100.0, 10.0, 100.0
        ) / 100.0
        colour_mode = self.color_mode_var.get()
        if colour_mode == "intensity" and self.intensity_by_spacing is None:
            self._ensure_intensities(show_errors=False)
        if colour_mode in {"d", "intensity"} and representatives:
            from matplotlib.colors import LinearSegmentedColormap, Normalize

            if colour_mode == "intensity":
                colour_values = [
                    max(
                        (self._point_intensity(point) or 0.0)
                        for point in group
                    )
                    for group in self.point_groups
                ]
                normalization = Normalize(vmin=0.0, vmax=100.0)
                colour_map = "viridis"
                colourbar_label = localised(
                    "Calculated powder intensity, %",
                    "Intensité calculée du diagramme de poudre, %",
                    "Расчётная интенсивность порошкового графика, %",
                )
            else:
                colour_values = d_values
                d_min = min(d_values)
                d_max = max(d_values)
                if math.isclose(d_min, d_max):
                    d_min -= 0.5
                    d_max += 0.5
                normalization = Normalize(vmin=d_min, vmax=d_max)
                colour_map = LinearSegmentedColormap.from_list(
                    "d_spacing",
                    (
                        "#0000ff",
                        "#0075ff",
                        "#00d9ff",
                        "#00f06a",
                        "#76ff00",
                        "#ff0000",
                    ),
                )
                colourbar_label = "d, Å"
            colored_scatter = self.ax.scatter(
                [position[0] for position in display_positions],
                [position[1] for position in display_positions],
                s=marker_sizes,
                c=colour_values,
                cmap=colour_map,
                norm=normalization,
                edgecolors="none",
                alpha=primary_alpha,
                clip_on=False,
                zorder=3,
            )
            colorbar = self.figure.colorbar(
                colored_scatter,
                ax=self.ax,
                fraction=0.045,
                pad=0.025,
                shrink=0.78,
            )
            colorbar.set_label(colourbar_label)
            self._colorbar = colorbar
        else:
            self.ax.scatter(
                [position[0] for position in display_positions],
                [position[1] for position in display_positions],
                s=marker_sizes,
                color=self.primary_colour_var.get(),
                alpha=primary_alpha,
                edgecolors="none" if self.size_by_d_var.get() else "white",
                linewidths=0.0 if self.size_by_d_var.get() else 0.8,
                clip_on=False,
                zorder=3,
            )
        if self.coincident_outlines_var.get():
            self._draw_coincident_group_markers(
                self.point_groups,
                marker_sizes,
            )

        overlay_sizes = np.empty(0, dtype=float)
        if self.overlay_layer is not None:
            layer = self.overlay_layer
            layer.colour = self.overlay_colour_var.get()
            layer.opacity_percent = self._percentage(
                self.overlay_opacity_var, 70.0, 10.0, 100.0
            )
            layer.size_percent = self._percentage(
                self.overlay_size_var, 100.0, 10.0, 300.0
            )
            layer.points = project_reflections(
                layer.crystal,
                layer.reflections,
                layer.orientation,
                self.projection_var.get(),
            )
            layer.point_groups = group_coincident_poles(layer.points)
            overlay_representatives = [
                group[0] for group in layer.point_groups
            ]
            overlay_positions = [
                pole_display_position(point)
                for point in overlay_representatives
            ]
            overlay_d = [
                point.d_spacing for point in overlay_representatives
            ]
            overlay_sizes = (
                marker_sizes_by_d(overlay_d)
                if self.size_by_d_var.get()
                else np.full(len(overlay_representatives), 58.0)
            )
            overlay_sizes *= layer.size_scale * global_size_scale
            self.ax.scatter(
                [position[0] for position in overlay_positions],
                [position[1] for position in overlay_positions],
                s=overlay_sizes,
                color=layer.colour,
                alpha=layer.opacity,
                edgecolors="none" if self.size_by_d_var.get() else "white",
                linewidths=0.0 if self.size_by_d_var.get() else 0.8,
                clip_on=False,
                zorder=3.1,
            )
            if self.coincident_outlines_var.get():
                self._draw_coincident_group_markers(
                    layer.point_groups,
                    overlay_sizes,
                )
            from matplotlib.lines import Line2D

            primary_legend_name = self.cif_document.name
            overlay_legend_name = layer.name
            if layer.document is self.cif_document:
                primary_legend_name = localised(
                    f"{primary_legend_name} — primary",
                    f"{primary_legend_name} — principale",
                    f"{primary_legend_name} — основная",
                )
                overlay_legend_name = localised(
                    f"{overlay_legend_name} — second",
                    f"{overlay_legend_name} — seconde",
                    f"{overlay_legend_name} — вторая",
                )
            elif primary_legend_name == overlay_legend_name:
                primary_legend_name = self.cif_document.source.name
                overlay_legend_name = layer.document.source.name
            legend_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=8,
                    markerfacecolor=self.primary_colour_var.get(),
                    markeredgecolor="white",
                    alpha=primary_alpha,
                    label=primary_legend_name,
                ),
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=8,
                    markerfacecolor=layer.colour,
                    markeredgecolor="white",
                    alpha=layer.opacity,
                    label=overlay_legend_name,
                ),
            ]
            self.ax.legend(
                handles=legend_handles,
                loc="upper right",
                framealpha=0.9,
                fontsize=9,
            )

        selected_group = self._selected_group(self.selected_layer_index)
        if selected_group is not None:
            selected = selected_group[0]
            selected_x, selected_y = pole_display_position(selected)
            selected_sizes = (
                marker_sizes
                if self.selected_layer_index == 0
                else overlay_sizes
            )
            context = self._layer_context(self.selected_layer_index)
            group_index = context[4].index(selected_group)
            marker_size = float(selected_sizes[group_index])
            self.ax.scatter(
                [selected_x],
                [selected_y],
                s=(math.sqrt(marker_size) + 8.0) ** 2,
                facecolors="none",
                edgecolors="#d23b2d",
                linewidths=2.0,
                clip_on=False,
                zorder=4,
            )
            self.show_information(
                selected_group,
                layer_index=self.selected_layer_index,
            )

        self._set_plot_title(self._plot_title())
        self.status_var.set(self._plot_status())
        self.draw_structure(pole_display_orientation(orientation), preview=preview)
        self.figure.tight_layout(pad=1.0)
        if self.labels_var.get():
            self._draw_pole_labels(marker_sizes, overlay_sizes)
        self.canvas.draw_idle()

    def show_information(
        self,
        group: Sequence[PolePoint],
        *,
        layer_index: int = 0,
    ) -> None:
        context = self._layer_context(layer_index)
        if context is None:
            return
        document, _crystal, centre, _orientation, _groups, selected_hkl = context
        point = group[0]
        if selected_hkl is not None:
            for candidate in group:
                if candidate.hkl == selected_hkl:
                    point = candidate
                    break
        wavelength = self.get_wavelength()
        self._ensure_layer_intensities(layer_index, show_errors=False)
        intensity = self._point_intensity(point, layer_index)
        phase_name = document.name
        if (
            self.overlay_layer is not None
            and self.overlay_layer.document is self.cif_document
        ):
            phase_name = localised(
                f"{phase_name} — {'primary' if layer_index == 0 else 'second'}",
                f"{phase_name} — {'principale' if layer_index == 0 else 'seconde'}",
                f"{phase_name} — {'основная' if layer_index == 0 else 'вторая'}",
            )
        lines = [
            localised(
                f"Phase: {phase_name}",
                f"Phase : {phase_name}",
                f"Фаза: {phase_name}",
            ),
            f"hkl: {format_hkl(point.hkl)}",
            localised(
                f"Centring: {format_hkl(centre)}",
                f"Centrage : {format_hkl(centre)}",
                f"Центрирование: {format_hkl(centre)}",
            ),
            "",
            f"χ: {point.chi:.4f}°",
            f"φ: {point.phi:.4f}°",
            "",
            localised("Projection coordinates:", "Coordonnées de projection :", "Координаты проекции:"),
            f"X: {point.x:.6f}",
            f"Y: {point.y:.6f}",
            "",
            localised("Pole unit vector:", "Vecteur unitaire du pôle :", "Единичный вектор полюса:"),
            f"x: {point.direction[0]:.6f}",
            f"y: {point.direction[1]:.6f}",
            f"z: {point.direction[2]:.6f}",
            "",
            f"d(hkl): {point.d_spacing:.6f} Å",
            f"2θ: {point.two_theta:.5f}°",
            f"λ: {wavelength:.5f} Å",
            localised(
                f"Calculated powder intensity: {intensity:.3f}%"
                if intensity is not None
                else "Calculated powder intensity: —",
                f"Intensité calculée du diagramme de poudre : {intensity:.3f} %"
                if intensity is not None
                else "Intensité calculée du diagramme de poudre : —",
                f"Расчётная интенсивность порошкового графика: {intensity:.3f}%"
                if intensity is not None
                else "Расчётная интенсивность порошкового графика: —",
            ),
        ]
        if len(group) > 1:
            lines.extend(
                [
                    "",
                    localised(
                        f"Reflections at this position: {len(group)}",
                        f"Réflexions à cette position : {len(group)}",
                        f"В этой позиции отражений: {len(group)}",
                    ),
                ]
            )
            for candidate in group:
                candidate_intensity = self._point_intensity(
                    candidate, layer_index
                )
                lines.append(
                    f"{format_hkl(candidate.hkl)}: "
                    f"d={candidate.d_spacing:.6f} Å; "
                    f"2θ={candidate.two_theta:.5f}°; "
                    + (
                        f"Irel={candidate_intensity:.3f}%"
                        if candidate_intensity is not None
                        else "Irel=—"
                    )
                )
        lines.extend(
            [
                "",
                localised(
                    "Systematic absence: no",
                    "Extinction systématique : non",
                    "Систематическое погасание: нет",
                ),
            ]
        )
        configure(self.info_text, state="normal")
        self.info_text.clear()
        self.info_text.setPlainText("\n".join(lines))
        configure(self.info_text, state="disabled")

    def plot_to_sphere(self, x: float, y: float) -> np.ndarray:
        return pole_plot_to_sphere(x, y, self.projection_var.get())

    def _select_pole_group(
        self,
        layer_index: int,
        group: Sequence[PolePoint],
    ) -> None:
        if not group:
            return
        if layer_index == 0:
            self.selected_hkl = group[0].hkl
        elif layer_index == 1 and self.overlay_layer is not None:
            self.overlay_layer.selected_hkl = group[0].hkl
        else:
            return
        self.selected_layer_index = layer_index
        self.redraw()

    def on_press(self, event) -> None:
        if self.toolbar.mode:
            return
        if event.button != 1:
            return
        if self.overlay_layer is not None and not self._joint_rotation_enabled():
            if event.inaxes is not self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            if math.hypot(event.xdata, event.ydata) > 1.05:
                return
            self.press_event = (event.x, event.y)
            self.last_arcball = None
            self.last_drag_pixel = None
            self.drag_mode = "selection"
            self.dragged = False
            return
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if math.hypot(event.xdata, event.ydata) > 1.05:
            return
        self.press_event = (event.x, event.y)
        self.last_arcball = self.plot_to_sphere(event.xdata, event.ydata)
        self.last_drag_pixel = None
        self.drag_mode = "pole"
        self.dragged = False

    def on_motion(self, event) -> None:
        if self.press_event is None or self.drag_mode is None or event.x is None or event.y is None:
            return
        if math.hypot(event.x - self.press_event[0], event.y - self.press_event[1]) > 3:
            self.dragged = True
        if (
            self.last_arcball is None
            or event.inaxes is not self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        current = self.plot_to_sphere(event.xdata, event.ydata)
        delta = rotation_between(self.last_arcball, current)
        self._set_primary_orientation(
            user_rotation=delta @ self.user_rotation,
            update_overlay_entries=False,
        )
        self.last_arcball = current
        self._frames.request()

    def on_release(self, event) -> None:
        if self.press_event is None:
            return
        was_dragged = self.dragged
        drag_mode = self.drag_mode
        self.press_event = None
        self.last_arcball = None
        self.last_drag_pixel = None
        self.drag_mode = None
        self.dragged = False
        self.update_rotation_entries()
        if self._joint_rotation_enabled():
            self.update_overlay_rotation_entries()
        self._frames.cancel()
        if was_dragged:
            self.redraw()
        if was_dragged or event.inaxes is not self.ax:
            return
        if event.x is None or event.y is None:
            return

        candidates = []
        for layer_index in (0, 1):
            context = self._layer_context(layer_index)
            if context is None:
                continue
            groups = context[4]
            if not groups:
                continue
            screen_points = self.ax.transData.transform(
                np.asarray(
                    [
                        pole_display_position(group[0])
                        for group in groups
                    ],
                    dtype=float,
                )
            )
            distances = np.hypot(
                screen_points[:, 0] - event.x,
                screen_points[:, 1] - event.y,
            )
            candidates.extend(
                (float(distance), layer_index, groups[index])
                for index, distance in enumerate(distances)
                if distance <= 14.0
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            return
        nearest_distance = candidates[0][0]
        ambiguous = [
            candidate
            for candidate in candidates
            if candidate[0] - nearest_distance <= 3.0
        ]
        if len(ambiguous) > 1:
            self._show_pole_candidate_menu(ambiguous, event)
        else:
            _distance, layer_index, group = candidates[0]
            self._select_pole_group(layer_index, group)

    def _set_plot_title(self, title):
        """Keep long translated captions within the plot's allotted width."""
        self.ax.title.set_fontsize(12)
        font = self.ax.title.get_fontproperties()
        renderer = self.canvas.get_renderer()
        available = max(120., self.ax.get_position().width * self.figure.bbox.width - 16.)
        lines = []
        for part in title.split(' · '):
            line = ''
            for word in part.split():
                candidate = f'{line} {word}'.strip()
                if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > available:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        self.ax.set_title('\n'.join(lines), pad=12, fontsize=12)

    def _build_layout(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.structure_viewer = PoleStructureView(
            self, on_orientation_changed=self.rotate_structure_from_mouse,
            on_rotation_finished=self.finish_structure_rotation,
        )
        splitter = QSplitter()
        root.addWidget(splitter)
        self.control_panel, controls = self.control_scroll()
        splitter.addWidget(self.control_panel)
        file_box = self.section(controls, 'text.cif_structure_2')
        self.button(file_box.content_layout, 'text.open_cif', self.ask_cif)
        self.label(file_box.content_layout, value=self.path_var)
        self.label(file_box.content_layout, value=self.structure_var)
        file_box.set_expanded(True)
        self.center_section = self.section(controls, 'text.pole_centring')
        center = self.center_section.content_layout
        self.center_prompt = self.label(center, 'text.select_an_allowed_pole')
        self.center_combo = self.combo(center, callback=self.choose_center)
        for title, value in zip(('h', 'k', 'l'), (self.h_var, self.k_var, self.l_var)):
            self.field(center, title, value, self.apply_center)
        self.button(center, 'text.place_pole_h_k_l_at_centre', self.apply_center)
        self.label(center, 'text.maximum_list_index')
        max_index = QSpinBox()
        max_index.setRange(1, 12)
        bind_value(self.max_index_var, max_index, max_index.value,
                   lambda value: max_index.setValue(int(value)), max_index.valueChanged)
        max_index.valueChanged.connect(self.refresh_center_list)
        center.addWidget(max_index)
        ranges = self.section(controls, 'text.displayed_reflections').content_layout
        self.field(ranges, 'text.d_from_angstrom', self.d_lower_var, self.rebuild_reflections)
        self.field(ranges, 'text.d_to_angstrom', self.d_upper_var, self.rebuild_reflections)
        self.range_button = self.button(ranges, 'text.plot_all_allowed_reflections', self.rebuild_reflections)
        display = self.section(controls, 'text.display').content_layout
        self.combo(display, self.projection_var,
                   [('text.stereographic', 'stereographic'), ('text.equal_area', 'equal_area')], self.redraw)
        self.check(display, 'text.label_poles', self.labels_var, self.labels_visibility_changed)
        self.label_leaders_check = self.check(display, 'pole.show_hkl_label_lines', self.label_leaders_var, self.redraw)
        self.label_leaders_check.setEnabled(self.labels_var.get())
        self.check(display, 'pole.outline_coincident_poles', self.coincident_outlines_var, self.redraw)
        self.check(display, 'text.show_angle_labels', self.angle_labels_var, self.redraw)
        self.check(display, 'text.show_structure_alongside', self.show_structure_var, self.structure_visibility_changed)
        display.addWidget(self.structure_viewer.display_section)
        self.basis_check = self.structure_viewer.basis_check
        bind_value(self.basis_visible, self.basis_check, self.basis_check.isChecked,
                   self.basis_check.setChecked, self.basis_check.toggled)
        self.basis_check.setEnabled(False)
        self.check(display, 'text.point_size_by_d', self.size_by_d_var, self.redraw)
        self.uniform_colour_radio, self.d_colour_radio, self.intensity_colour_radio = self.radios(
            display, [('text.uniform_colour', 'uniform'), ('text.point_colour_by_d', 'd'),
                      ('text.point_colour_by_calculated_intensity', 'intensity')], self.color_mode_var, self.change_colour_mode)
        self.label(display, 'text.first_phase_display')
        self.button(display, 'text.colour_2', lambda: self.choose_layer_colour(True))
        self.label(display, value=self.primary_colour_var)
        self.percent(display, 'Прозрачность, %', self.primary_opacity_var, self.redraw, maximum=100)
        self.percent(display, 'Размер точек, %', self.primary_size_var, self.redraw)
        self.label(display, 'text.overall_point_scale')
        scale = QSlider(Qt.Orientation.Horizontal)
        scale.setRange(25, 300)
        bind_value(self.global_size_var, scale, scale.value, lambda value: scale.setValue(round(value)), scale.valueChanged)
        scale.valueChanged.connect(self.global_point_scale_changed)
        display.addWidget(scale)
        self.label(display, value=self.global_size_label_var)
        self.rotation_section = self.section(controls, 'text.absolute_crystal_rotation')
        self._rotation_fields(self.rotation_section.content_layout, self.rotation_vars,
                              self.apply_exact_rotation, 'text.set_absolute_angles')
        self.button(self.rotation_section.content_layout, 'text.restore_centred_pole', self.reset_rotation)
        self.relative_rotation_section = self.section(controls, 'text.relative_crystal_rotation')
        self._rotation_fields(self.relative_rotation_section.content_layout, self.relative_rotation_vars,
                              self.apply_relative_rotation, 'text.rotate_relative_to_current', relative=True)
        align = self.section(controls, 'text.align_selected_pole').content_layout
        self._alignment_buttons(align, 0)
        self.overlay_section = self.section(controls, 'text.overlaid_second_phase')
        overlay = self.overlay_section.content_layout
        self.label(overlay, 'text.phase_from_project_data')
        self.overlay_combo = self.combo(overlay, self.overlay_choice_var)
        self.add_overlay_button = self.button(overlay, 'text.add_overlay', self.add_selected_overlay)
        self.open_overlay_button = self.button(overlay, 'text.open_cif_as_overlay', self.ask_overlay_cif)
        self.overlay_settings = QWidget()
        settings = QVBoxLayout(self.overlay_settings)
        settings.setContentsMargins(0, 0, 0, 0)
        overlay.addWidget(self.overlay_settings)
        self.label(settings, value=self.overlay_name_var)
        self.remove_overlay_button = self.button(settings, 'text.remove_second_phase', self.remove_overlay)
        self.remove_overlay_button.setEnabled(False)
        self.joint_rotation_check = self.check(settings, 'pole.rotate_phases_together', self.joint_rotation_var, self.joint_rotation_changed)
        self.label(settings, 'text.second_phase_display')
        self.button(settings, 'text.colour_2', lambda: self.choose_layer_colour(False))
        self.label(settings, value=self.overlay_colour_var)
        self.percent(settings, 'Прозрачность, %', self.overlay_opacity_var, self.redraw, maximum=100)
        self.percent(settings, 'Размер точек, %', self.overlay_size_var, self.redraw)
        self.label(settings, 'Центрирование второй фазы')
        for title, value in zip(('h', 'k', 'l'), self.overlay_hkl_vars):
            self.field(settings, title, value, self.apply_overlay_center)
        self.button(settings, 'text.place_pole_h_k_l_at_centre', self.apply_overlay_center)
        self.label(settings, 'Абсолютный поворот второй фазы')
        self._rotation_fields(settings, self.overlay_rotation_vars, self.apply_overlay_exact_rotation, 'text.set_absolute_angles')
        self.label(settings, 'Относительный поворот второй фазы')
        self._rotation_fields(settings, self.overlay_relative_rotation_vars, self.apply_overlay_relative_rotation,
                              'text.rotate_relative_to_current', relative=True)
        self.label(settings, 'text.align_selected_pole')
        self._alignment_buttons(settings, 1)
        self.overlay_settings.hide()
        self.drag_help_label = self.label(controls)
        self._update_drag_help()
        self.label(controls, value=self.status_var)
        controls.addStretch(1)
        self.matplotlib_plot = QWidget()
        plot_layout = QVBoxLayout(self.matplotlib_plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(7.2, 7.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(0)
        plot_layout.addWidget(self.canvas, 1)
        self.toolbar = PlotToolbar(self.canvas, self.matplotlib_plot,
            figure_provider=lambda: self.structure_viewer.figure_for_export(self.figure))
        plot_layout.addWidget(self.toolbar)
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.matplotlib_plot)
        self.plot_splitter = PolePlotSplitter()
        self.plot_splitter.addWidget(self.plot_stack)
        self.plot_splitter.addWidget(self.structure_viewer)
        self.plot_splitter.setSizes([500, 500])
        self.plot_splitter.setStretchFactor(0, 1)
        self.plot_splitter.setStretchFactor(1, 1)
        splitter.addWidget(self.plot_splitter)
        information = QWidget()
        info_layout = QVBoxLayout(information)
        info_layout.setContentsMargins(4, 8, 8, 8)
        self.label(info_layout, 'text.selected_pole')
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumWidth(150)
        info_layout.addWidget(self.info_text, 1)
        splitter.addWidget(information)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 710, 250])

    def _rotation_fields(self, layout, values, callback, title, *, relative=False):
        for axis, value in zip(('X', 'Y', 'Z'), values):
            self.field(layout, ('Δ' if relative else '') + axis + ', °', value, callback)
        self.button(layout, title, callback)

    def _alignment_buttons(self, layout, layer_index):
        row = QHBoxLayout()
        for title, angle in (('+X', 270.), ('−X', 90.), ('+Y', 0.), ('−Y', 180.)):
            self.button(row, title, lambda angle=angle: self.align_selected_pole(angle, layer_index=layer_index))
        layout.addLayout(row)

    def ask_cif(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('text.open_cif'), '', 'CIF (*.cif);;All files (*)')
        if path:
            if self.on_open_cif is not None:
                self.on_open_cif(path)
            else:
                self.load_cif(path)

    def ask_overlay_cif(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('text.open_cif_as_overlay'), '', 'CIF (*.cif);;All files (*)')
        if not path:
            return
        if self.on_add_overlay is not None:
            self.on_add_overlay(path)
            return
        try:
            from ..cif_document import load_cif_document
            self.load_overlay_document(load_cif_document(path))
        except (OSError, ValueError) as error:
            show_error(tr('text.could_not_open_cif'), error, self)

    def choose_layer_colour(self, primary):
        variable = self.primary_colour_var if primary else self.overlay_colour_var
        colour = choose_colour(self, variable.get(), localised('Phase colour', 'Couleur de la phase', 'Цвет фазы'))
        if colour:
            variable.set(colour)
            self.redraw()

    def _set_overlay_settings_visible(self, visible):
        self.overlay_section.set_expanded(True)
        self.overlay_settings.setVisible(visible)

    def _show_pole_candidate_menu(self, candidates, event):
        if self._selection_menu is not None:
            self._selection_menu.close()
            self._selection_menu.deleteLater()
        menu = QMenu(self)
        self._selection_menu = menu
        for _distance, layer_index, group in candidates:
            context = self._layer_context(layer_index)
            if context is None:
                continue
            label = f'{context[0].name} — {format_hkl(group[0].hkl)}'
            if len(group) > 1:
                count = len(group)
                label += ' — ' + localised(f'{count} reflections', f'{count} réflexions', f'отражений: {count}')
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, index=layer_index, selected=group: self._select_pole_group(index, selected))
        menu.popup(QCursor.pos())

    def retranslate(self):
        self.retranslate_controls()
        self.structure_viewer.retranslate()
        if self.pyqtgraph_plot is not None:
            self.pyqtgraph_plot.retranslate()
        # Preserve dynamic phase headings and all numerical/orientation state.
        for section in (self.center_section, self.rotation_section, self.relative_rotation_section):
            source = section.title_source
            section.set_title(tr(source) if source.startswith('text.') else translate_text(source))
        cell_only = self.cif_document is not None and getattr(self.cif_document, 'is_cell_only', False)
        configure(self.center_prompt, text=tr('text.select_a_pole_not_systematically_forbidden' if cell_only else 'text.select_an_allowed_pole'))
        configure(self.range_button, text=tr('text.plot_reflections_not_systematically_forbidden' if cell_only else 'text.plot_all_allowed_reflections'))
        self._update_drag_help()
        self.refresh_center_list()
        self.refresh_overlay_choices()
        self.redraw()
