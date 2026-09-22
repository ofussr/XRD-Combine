"""Project-backed experimental and calculated reciprocal-space maps."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..localization import localised, tr
from ..models.project import CIF, RSM, RSM_DATA, SCAN
from ..services.rsm import (
    angles_to_q, build_raw_map, build_scan_map, calculate_reflections,
    display_mesh, make_orientation, map_view_bounds, measured_reflections,
    q_to_angles, reflection_point,
)
from .pyqtgraph_interaction import handle_navigation_drag
from .radiation import RadiationSelector


def L(en: str, fr: str, ru: str) -> str:
    return localised(en, fr, ru)


def number(field: QLineEdit, *, optional=False) -> float | None:
    value = field.text().strip().replace(",", ".")
    if not value and optional:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Enter a finite number.")
    return result


def indices(field: QLineEdit) -> tuple[int, int, int]:
    parts = field.text().replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Enter three integer Miller indices separated by spaces.")
    return tuple(int(part) for part in parts)


def _plot_surface():
    class PanBox(pg.ViewBox):
        def mouseDragEvent(self, event, axis=None):
            if handle_navigation_drag(self, event, axis):
                return
            if event.button() == Qt.MouseButton.LeftButton:
                event.accept()
                return
            event.accept()

        def wheelEvent(self, event, axis=None):
            event.accept()

        def mouseDoubleClickEvent(self, event):
            event.accept()

    plot = pg.PlotWidget(viewBox=PanBox())
    plot.setBackground("w")
    for axis in ("left", "bottom"):
        plot.getPlotItem().getAxis(axis).setPen("#333333")
        plot.getPlotItem().getAxis(axis).setTextPen("#333333")
    plot.showGrid(x=True, y=True, alpha=.15)
    plot.getPlotItem().hideButtons()
    plot.getPlotItem().setMenuEnabled(False)
    return plot


def _controls_plot(parent: QWidget):
    root = QVBoxLayout(parent)
    splitter = QSplitter(Qt.Orientation.Horizontal)
    panel = QScrollArea()
    panel.setWidgetResizable(True)
    panel.setMinimumWidth(230)
    controls = QWidget()
    form = QFormLayout(controls)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    panel.setWidget(controls)
    splitter.addWidget(panel)
    plot = _plot_surface()
    splitter.addWidget(plot)
    splitter.setSizes([300, 800])
    root.addWidget(splitter)
    return form, plot


def _export(plot: pg.PlotWidget, parent: QWidget, stem: str):
    path, _filter = QFileDialog.getSaveFileName(
        parent, L("Save RSM map", "Enregistrer la carte RSM", "Сохранить карту RSM"),
        str(Path.cwd() / f"{stem}.png"), "PNG (*.png)",
    )
    if path:
        import pyqtgraph.exporters
        pyqtgraph.exporters.ImageExporter(plot.getPlotItem()).export(path)


def _numbered_files(selected: list[Path]) -> list[Path]:
    if len(selected) != 1:
        return sorted(selected, key=lambda path: (path.stem.lower().rsplit("__", 1)[0],
                                                 int(path.stem.rsplit("__", 1)[1])
                                                 if path.stem.rsplit("__", 1)[-1].isdigit()
                                                 else 0, path.name.lower()))
    source = selected[0]
    match = re.fullmatch(r"(.*?__(?:range_)?)(\d+)", source.stem, re.I)
    if match is None:
        return selected
    peers = []
    for candidate in source.parent.iterdir():
        found = re.fullmatch(r"(.*?__(?:range_)?)(\d+)", candidate.stem, re.I)
        if (candidate.suffix.lower() == source.suffix.lower()
                and found and found.group(1).lower() == match.group(1).lower()):
            peers.append((int(found.group(2)), candidate))
    return [path for _, path in sorted(peers)] or selected


class ExperimentalRSM(QWidget):
    def __init__(self, store, file_service, wavelength_provider, parent=None):
        super().__init__(parent)
        self.store = store
        self.file_service = file_service
        self.wavelength_provider = wavelength_provider
        self.data = None
        self._token = None
        self._source_is_raw = None
        self._overlay_requested = False
        self._overlay_cache_key = None
        self._overlay_cache = []
        form, self.plot = _controls_plot(self)
        self.open_raw_button = QPushButton()
        self.open_raw_button.clicked.connect(self.open_raw)
        self.open_xy_button = QPushButton()
        self.open_xy_button.clicked.connect(self.open_xy)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.open_raw_button)
        row.addWidget(self.open_xy_button)
        form.addRow(buttons)
        self.source = QLabel()
        self.source.setWordWrap(True)
        form.addRow(self.source)
        self.first = QLineEdit()
        self.step = QLineEdit()
        self.last = QLineEdit()
        self._filled_field = None
        for field in (self.first, self.step, self.last):
            field.textEdited.connect(
                lambda _text, current=field: self._angle_field_edited(current))
        self.angle_label = QLabel()
        self.first_label = QLabel()
        self.step_label = QLabel()
        self.last_label = QLabel()
        self.xy_note = QLabel()
        self.xy_note.setWordWrap(True)
        for label, field in ((self.first_label, self.first),
                             (self.step_label, self.step),
                             (self.last_label, self.last)):
            form.addRow(label, field)
        form.addRow(self.xy_note)
        self.view_label = QLabel()
        self.view = QComboBox()
        self.view.addItem("", "angular")
        self.view.addItem("", "q")
        form.addRow(self.view_label, self.view)
        self.extent_label = QLabel()
        self.extent = QComboBox()
        self.extent.addItem("", "real")
        self.extent.addItem("", "full")
        form.addRow(self.extent_label, self.extent)
        self.scale_label = QLabel()
        self.scale = QComboBox()
        self.scale.addItem("", "log")
        self.scale.addItem("", "linear")
        self.scale.setCurrentIndex(1)
        form.addRow(self.scale_label, self.scale)
        self.cmap_label = QLabel()
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "turbo", "plasma", "inferno", "magma"])
        form.addRow(self.cmap_label, self.cmap)
        self.cif_label = QLabel()
        self.cif_combo = QComboBox()
        form.addRow(self.cif_label, self.cif_combo)
        self.open_cif_button = QPushButton()
        self.open_cif_button.clicked.connect(self.open_cif)
        form.addRow(self.open_cif_button)
        self.surface_label = QLabel()
        self.surface = QLineEdit("0 0 1")
        self.inplane_label = QLabel()
        self.inplane = QLineEdit("1 0 0")
        self.max_label = QLabel()
        self.max_index = QLineEdit("8")
        self.tolerance_label = QLabel()
        self.tolerance = QLineEdit("0.02")
        for label, field in ((self.surface_label, self.surface),
                             (self.inplane_label, self.inplane),
                             (self.max_label, self.max_index),
                             (self.tolerance_label, self.tolerance)):
            form.addRow(label, field)
            field.editingFinished.connect(self._overlay_input_changed)
        self.add_cif_button = QPushButton()
        self.add_cif_button.clicked.connect(self.add_cif_points)
        self.remove_cif_button = QPushButton()
        self.remove_cif_button.clicked.connect(self.remove_cif_points)
        form.addRow(self.add_cif_button)
        form.addRow(self.remove_cif_button)
        self.draw_button = QPushButton()
        self.draw_button.clicked.connect(self.draw)
        self.save_button = QPushButton()
        self.save_button.clicked.connect(
            lambda: _export(self.plot, self, "experimental_rsm"))
        form.addRow(self.draw_button)
        form.addRow(self.save_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.view.currentIndexChanged.connect(self.draw)
        self.extent.currentIndexChanged.connect(self.draw)
        self.scale.currentIndexChanged.connect(self.draw)
        self.cmap.currentIndexChanged.connect(self.draw)
        self.cif_combo.currentIndexChanged.connect(self._overlay_input_changed)
        self.retranslate()

    def retranslate(self):
        self.open_raw_button.setText(L("Open RAW…", "Ouvrir RAW…", "Открыть RAW…"))
        self.open_xy_button.setText(L("Open XY series…", "Ouvrir une série XY…", "Открыть серию XY…"))
        self.first_label.setText(L("First omega, °", "Premier oméga, °", "Первый омега, °"))
        self.step_label.setText(L("Omega step, °", "Pas oméga, °", "Шаг омега, °"))
        self.last_label.setText(L("Last omega, °", "Dernier oméga, °", "Последний омега, °"))
        self.xy_note.setText(L(
            "For XY series, X is treated as 2Theta; enter any two omega values above. RAW coordinates are read from the file.",
            "Pour une série XY, X représente 2Theta ; indiquez deux valeurs oméga ci-dessus. Les coordonnées RAW viennent du fichier.",
            "Для серии XY координата X считается 2Theta; задайте любые два значения омега выше. Координаты RAW считываются из файла.",
        ))
        self.view_label.setText(L("Coordinates", "Coordonnées", "Координаты"))
        self.view.setItemText(0, L("Angles", "Angles", "Углы"))
        self.view.setItemText(1, "Qx / Qz")
        self.extent_label.setText(L("Map area", "Zone de la carte", "Область карты"))
        self.extent.setItemText(0, L("Real (measured)", "Réelle (mesurée)", "Real (измеренная)"))
        self.extent.setItemText(1, L("Full (declared)", "Complète (déclarée)", "Full (заявленная)"))
        self.scale_label.setText(L("Intensity scale", "Échelle d’intensité", "Шкала интенсивности"))
        self.scale.setItemText(0, L("Logarithmic", "Logarithmique", "Логарифмическая"))
        self.scale.setItemText(1, L("Linear", "Linéaire", "Линейная"))
        self.cmap_label.setText(L("Colour map", "Palette", "Цветовая схема"))
        self.cif_label.setText(L("CIF for reflections", "CIF des réflexions", "CIF для рефлексов"))
        self.open_cif_button.setText(L("Open CIF…", "Ouvrir CIF…", "Открыть CIF…"))
        self.surface_label.setText(L("Surface normal h k l", "Normale à la surface h k l", "Нормаль поверхности h k l"))
        self.inplane_label.setText(L("In-plane direction h k l", "Direction dans le plan h k l", "Направление в плоскости h k l"))
        self.max_label.setText(L("Maximum |h,k,l|", "Maximum |h,k,l|", "Максимум |h,k,l|"))
        self.tolerance_label.setText(L("|Qy| tolerance, Å^-1", "Tolérance |Qy|, Å^-1", "Допуск |Qy|, Å^-1"))
        self.add_cif_button.setText(L("Add CIF reflections", "Ajouter les réflexions CIF", "Добавить рефлексы CIF"))
        self.remove_cif_button.setText(L("Remove CIF reflections", "Retirer les réflexions CIF", "Убрать рефлексы CIF"))
        self.draw_button.setText(L("Draw map", "Tracer la carte", "Построить карту"))
        self.save_button.setText(L("Save PNG…", "Enregistrer PNG…", "Сохранить PNG…"))

    def open_raw(self):
        files, _ = QFileDialog.getOpenFileNames(self, self.open_raw_button.text(),
                                                "", "RAW (*.raw);;All files (*)")
        if not files:
            return
        try:
            for source in _numbered_files([Path(p) for p in files]):
                documents = self.file_service.load_path(self.store, source)
                document = next((item for item in documents if item.kind == RSM_DATA), None)
                if document is None:
                    document = self.file_service.load_rsm_data(self.store, source)
                self.store.assign(document.uid, RSM, True)
            self.refresh_documents()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "RSM", str(exc))

    def open_xy(self):
        files, _ = QFileDialog.getOpenFileNames(self, self.open_xy_button.text(),
                                                "", "XY (*.xy *.txt *.dat *.csv)")
        if not files:
            return
        try:
            for source in _numbered_files([Path(p) for p in files]):
                for item in self.file_service.load_path(self.store, source):
                    if item.kind == SCAN:
                        self.store.assign(item.uid, RSM, True)
            self.refresh_documents()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "RSM", str(exc))

    def open_cif(self):
        name, _ = QFileDialog.getOpenFileName(self, self.open_cif_button.text(), "", "CIF (*.cif)")
        if not name:
            return
        try:
            document = self.file_service.load_cif(self.store, name)
            self._refresh_cif_documents()
            self.cif_combo.setCurrentIndex(self.cif_combo.findData(document.uid))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "RSM", str(exc))

    def _refresh_cif_documents(self):
        selected = self.cif_combo.currentData()
        documents = [doc for doc in self.store.documents.values() if doc.kind == CIF]
        self.cif_combo.blockSignals(True)
        self.cif_combo.clear()
        for doc in documents:
            self.cif_combo.addItem(f"{doc.name} — {doc.source.name}", doc.uid)
        current = self.cif_combo.findData(selected)
        self.cif_combo.setCurrentIndex(current if current >= 0 else 0)
        self.cif_combo.blockSignals(False)
        self.add_cif_button.setEnabled(bool(documents))
        return selected != self.cif_combo.currentData()

    def _angle_field_edited(self, current):
        if self._filled_field is None:
            return
        if self._filled_field is current:
            (self.last if current is self.step else self.step).clear()
        else:
            self._filled_field.clear()
        self._filled_field = None

    def _set_raw_angles(self, data):
        steps = np.diff(data.omega)
        values = (data.omega[0], steps[0] if len(steps) and
                  np.allclose(steps, steps[0], atol=1e-8, rtol=1e-8)
                  else None, data.omega[-1])
        for field, value in zip((self.first, self.step, self.last), values):
            field.setText(f"{value:.12g}" if value is not None else "")
            field.setReadOnly(True)
        self._filled_field = None

    def _complete_xy_angles(self):
        fields = (self.first, self.step, self.last)
        missing = [field for field in fields if not field.text().strip()]
        if len(missing) != 1:
            return
        field = missing[0]
        value = (self.data.omega[0] if field is self.first else
                 self.data.omega[-1] if field is self.last else
                 self.data.omega[1] - self.data.omega[0])
        field.setText(f"{value:.12g}")
        self._filled_field = field

    def _overlay_input_changed(self, *_args):
        self._overlay_cache_key = None
        if self._overlay_requested and self.data is not None:
            self.draw()

    def add_cif_points(self):
        if self.data is None:
            self.status.setText(L("Open an RSM map first.", "Ouvrez d’abord une carte RSM.",
                                  "Сначала откройте карту RSM."))
            return
        if self.cif_combo.currentData() is None:
            self.status.setText(L("Select a CIF first.", "Sélectionnez d’abord un CIF.",
                                  "Сначала выберите CIF."))
            return
        self._overlay_requested = True
        self.draw()

    def remove_cif_points(self):
        self._overlay_requested = False
        self._overlay_cache_key = None
        if self.data is not None:
            self.draw()

    def _overlay_points(self):
        uid = self.cif_combo.currentData()
        doc = self.store.documents.get(uid)
        if doc is None or doc.kind != CIF:
            raise ValueError("Select a CIF for the experimental map.")
        wavelength = self.wavelength_provider()
        surface = indices(self.surface)
        inplane = indices(self.inplane)
        max_index = int(self.max_index.text())
        tolerance = number(self.tolerance)
        key = (uid, id(doc.payload), wavelength, surface, inplane, max_index, tolerance)
        if key != self._overlay_cache_key:
            orientation = make_orientation(doc.payload.crystal, surface, inplane)
            self._overlay_cache = calculate_reflections(
                doc.payload.crystal, orientation, wavelength, max_index, tolerance)
            self._overlay_cache_key = key
        return measured_reflections(self.data, self._overlay_cache)

    def _draw_overlay(self):
        points = self._overlay_points()
        in_q = self.view.currentData() == "q"
        if points:
            x = [point.qx if in_q else point.omega for point in points]
            y = [point.qz if in_q else point.two_theta for point in points]
            self.plot.addItem(pg.ScatterPlotItem(
                x=x, y=y, symbol="o", size=12,
                pen=pg.mkPen("#ffffff", width=2), brush=pg.mkBrush("#ed2048")))
            for point, px, py in zip(points, x, y):
                label = pg.TextItem("(%d %d %d)" % point.hkl,
                                    color="#b50032", anchor=(0, 1))
                label.setPos(px, py)
                self.plot.addItem(label)
        return len(points)

    def refresh_documents(self):
        selection_changed = self._refresh_cif_documents()
        raw = self.store.assigned_documents(RSM, kind=RSM_DATA)
        scans = self.store.assigned_documents(RSM, kind=SCAN)
        token = tuple((item.uid, id(item.payload)) for item in (raw or scans))
        if token == self._token:
            if self._overlay_requested:
                selected = self.cif_combo.currentData()
                doc = self.store.documents.get(selected)
                if selection_changed:
                    self.remove_cif_points()
                elif doc is not None and (self._overlay_cache_key is None or
                      self._overlay_cache_key[:2] != (selected, id(doc.payload))):
                    self.draw()
            return
        self._token = token
        self._overlay_requested = False
        self._overlay_cache_key = None
        self.data = None
        self.plot.clear()
        if not token:
            self.source.setText(L("Select RAW or XY scans in Project data.",
                                  "Sélectionnez des scans RAW ou XY dans Données du projet.",
                                  "Выберите RAW или XY в «Данных проекта»."))
            self.status.clear()
            return
        self.source.setText("\n".join(item.source.name for item in (raw or scans)[:6]))
        try:
            if raw:
                self.data = build_raw_map([item.payload for item in raw])
                self._set_raw_angles(self.data)
                self._source_is_raw = True
            else:
                if self._source_is_raw:
                    for field in (self.first, self.step, self.last):
                        field.clear()
                elif self._filled_field is not None:
                    self._filled_field.clear()
                for field in (self.first, self.step, self.last):
                    field.setReadOnly(False)
                self._filled_field = None
                self._source_is_raw = False
                self.data = build_scan_map(
                    [item.payload for item in scans], number(self.first, optional=True),
                    number(self.step, optional=True), number(self.last, optional=True))
                self._complete_xy_angles()
            self.draw()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))

    def draw(self, *_args):
        raw = self.store.assigned_documents(RSM, kind=RSM_DATA)
        if not raw:
            scans = self.store.assigned_documents(RSM, kind=SCAN)
            try:
                self.data = build_scan_map(
                    [item.payload for item in scans], number(self.first, optional=True),
                    number(self.step, optional=True), number(self.last, optional=True))
                self._complete_xy_angles()
            except (ValueError, TypeError) as exc:
                self.status.setText(str(exc))
                return
        if self.data is None:
            return
        try:
            x, y, z, xlabel, ylabel = display_mesh(
                self.data, self.view.currentData(), self.wavelength_provider())
            valid = np.isfinite(z)
            if self.scale.currentData() == "log":
                valid &= z > 0
                if not np.any(valid):
                    raise ValueError("No positive intensities for logarithmic scaling.")
                logged = np.full(z.shape, np.nan, dtype=float)
                logged[valid] = np.log10(z[valid])
                z = logged
            if not np.any(valid):
                raise ValueError("The map contains no finite intensities.")
            lower, upper = np.nanmin(z), np.nanmax(z)
            if upper <= lower:
                upper = lower + 1
            colourmap = pg.colormap.get(self.cmap.currentText())
            mesh = pg.PColorMeshItem(x, y, z, colorMap=colourmap,
                                     levels=(lower, upper), enableAutoLevels=False)
            self.plot.clear()
            self.plot.addItem(mesh)
            self.plot.setLabel("bottom", xlabel)
            self.plot.setLabel("left", ylabel)
            x_min, x_max, y_min, y_max = map_view_bounds(
                self.data, self.view.currentData(), self.wavelength_provider(),
                self.extent.currentData() == "real")
            self.plot.setXRange(x_min, x_max, padding=.01)
            self.plot.setYRange(y_min, y_max, padding=.01)
            overlay_count = self._draw_overlay() if self._overlay_requested else None
            if overlay_count is None:
                self.status.setText(self.data.description)
            else:
                self.status.setText(self.data.description + L(
                    f"; {overlay_count} CIF reflections on measured cells",
                    f" ; {overlay_count} réflexions CIF sur les cellules mesurées",
                    f"; рефлексов CIF в измеренной области: {overlay_count}"))
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))


class CalculatedRSM(QWidget):
    def __init__(self, store, file_service, wavelength_provider, parent=None):
        super().__init__(parent)
        self.store = store
        self.file_service = file_service
        self.wavelength_provider = wavelength_provider
        self._token = None
        self.crystal = None
        self._span_memory = {"angular": ("1.0", "1.0"), "q": ("0.10", "0.10")}
        self._previous_view_mode = "angular"
        form, self.plot = _controls_plot(self)
        self.open_button = QPushButton()
        self.open_button.clicked.connect(self.open_cif)
        form.addRow(self.open_button)
        self.source = QLabel()
        self.source.setWordWrap(True)
        form.addRow(self.source)
        self.surface_label = QLabel()
        self.surface = QLineEdit("0 0 1")
        self.inplane_label = QLabel()
        self.inplane = QLineEdit("1 0 0")
        form.addRow(self.surface_label, self.surface)
        form.addRow(self.inplane_label, self.inplane)
        self.target_label = QLabel()
        self.target = QComboBox()
        for mode in ("hkl", "angles", "q"):
            self.target.addItem("", mode)
        self._previous_target_mode = None
        form.addRow(self.target_label, self.target)
        self.target_a_label = QLabel()
        self.target_b_label = QLabel()
        self.target_a = QLineEdit("0 0 2")
        self.target_b = QLineEdit()
        form.addRow(self.target_a_label, self.target_a)
        form.addRow(self.target_b_label, self.target_b)
        self.target.currentIndexChanged.connect(self._target_changed)
        self.max_label = QLabel()
        self.max_index = QLineEdit("8")
        self.tolerance_label = QLabel()
        self.tolerance = QLineEdit("0.02")
        form.addRow(self.max_label, self.max_index)
        form.addRow(self.tolerance_label, self.tolerance)
        self.view_label = QLabel()
        self.view = QComboBox()
        self.view.addItem("", "angular")
        self.view.addItem("Qx / Qz", "q")
        form.addRow(self.view_label, self.view)
        self.span_x_label = QLabel()
        self.span_y_label = QLabel()
        self.span_x = QLineEdit("1.0")
        self.span_y = QLineEdit("1.0")
        form.addRow(self.span_x_label, self.span_x)
        form.addRow(self.span_y_label, self.span_y)
        self.view.currentIndexChanged.connect(self._view_changed)
        self.labels = QCheckBox()
        self.labels.setChecked(True)
        form.addRow(self.labels)
        self.target_marker = QCheckBox()
        self.target_marker.setChecked(True)
        form.addRow(self.target_marker)
        self.draw_button = QPushButton()
        self.draw_button.clicked.connect(self.draw)
        self.save_button = QPushButton()
        self.save_button.clicked.connect(
            lambda: _export(self.plot, self, "calculated_rsm"))
        form.addRow(self.draw_button)
        form.addRow(self.save_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self._target_changed()
        self.retranslate()

    def retranslate(self):
        self.open_button.setText(L("Open CIF…", "Ouvrir CIF…", "Открыть CIF…"))
        self.surface_label.setText(L("Surface normal h k l", "Normale à la surface h k l", "Нормаль поверхности h k l"))
        self.inplane_label.setText(L("In-plane direction h k l", "Direction dans le plan h k l", "Направление в плоскости h k l"))
        self.target_label.setText(L("Target by", "Cible définie par", "Цель задана через"))
        self.target.setItemText(0, "h k l")
        self.target.setItemText(1, L("Angles", "Angles", "Углы"))
        self.target.setItemText(2, "Qx / Qz")
        self.max_label.setText(L("Maximum |h,k,l|", "Maximum |h,k,l|", "Максимум |h,k,l|"))
        self.tolerance_label.setText(L("|Qy| tolerance, Å^-1", "Tolérance |Qy|, Å^-1", "Допуск |Qy|, Å^-1"))
        self.view_label.setText(L("Coordinates", "Coordonnées", "Координаты"))
        self.view.setItemText(0, L("Angles", "Angles", "Углы"))
        self._update_span_labels()
        self.labels.setText(L("Show hkl labels", "Afficher les indices hkl", "Показывать индексы hkl"))
        self.target_marker.setText(L("Show target marker", "Afficher le marqueur cible", "Показывать метку цели"))
        self.draw_button.setText(L("Calculate map", "Calculer la carte", "Рассчитать карту"))
        self.save_button.setText(L("Save PNG…", "Enregistrer PNG…", "Сохранить PNG…"))
        self._target_changed()

    def _target_changed(self, *_args):
        mode = self.target.currentData()
        self.target_a_label.setText({"hkl": "h k l", "angles": "Omega, °", "q": "Qx, Å^-1"}[mode])
        self.target_b_label.setText({"hkl": "", "angles": "2Theta, °", "q": "Qz, Å^-1"}[mode])
        self.target_b.setVisible(mode != "hkl")
        self.target_b_label.setVisible(mode != "hkl")
        if self._previous_target_mode != mode:
            self.target_a.setText({"hkl": "0 0 2", "angles": "22.5", "q": "0"}[mode])
            self.target_b.setText({"hkl": "", "angles": "45", "q": "3"}[mode])
            self._previous_target_mode = mode

    def _view_changed(self, *_args):
        mode = self.view.currentData()
        if mode == self._previous_view_mode:
            return
        self._span_memory[self._previous_view_mode] = (self.span_x.text(), self.span_y.text())
        horizontal, vertical = self._span_memory[mode]
        self.span_x.setText(horizontal)
        self.span_y.setText(vertical)
        self._previous_view_mode = mode
        self._update_span_labels()

    def _update_span_labels(self):
        if self.view.currentData() == "q":
            self.span_x_label.setText("Qx, Å^-1")
            self.span_y_label.setText("Qz, Å^-1")
        else:
            self.span_x_label.setText("Omega, °")
            self.span_y_label.setText("2Theta, °")

    def open_cif(self):
        name, _ = QFileDialog.getOpenFileName(self, self.open_button.text(), "", "CIF (*.cif)")
        if name:
            try:
                document = self.file_service.load_cif(self.store, name)
                self.store.assign(document.uid, RSM, True)
                self.refresh_documents()
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "RSM", str(exc))

    def refresh_documents(self):
        docs = self.store.assigned_documents(RSM, kind=CIF)
        doc = docs[-1] if docs else None
        token = (doc.uid, id(doc.payload)) if doc is not None else None
        if token == self._token:
            return
        self._token = token
        self.crystal = doc.payload.crystal if doc is not None else None
        self.source.setText(doc.name if doc is not None else L(
            "Select a CIF in Project data.", "Sélectionnez un CIF dans Données du projet.",
            "Выберите CIF в «Данных проекта»."))
        self.plot.clear()
        self.status.clear()

    def draw(self):
        if self.crystal is None:
            self.status.setText(L("Select a CIF first.", "Sélectionnez d’abord un CIF.",
                                  "Сначала выберите CIF."))
            return
        try:
            wavelength = self.wavelength_provider()
            crystal = self.crystal
            orientation = make_orientation(crystal, indices(self.surface), indices(self.inplane))
            tolerance = number(self.tolerance)
            mode = self.target.currentData()
            if mode == "hkl":
                hkl = indices(self.target_a)
                if crystal.is_systematically_absent(hkl):
                    raise ValueError("Target reflection is systematically absent.")
                target = reflection_point(crystal, orientation, hkl, wavelength)
                if abs(target.qy) > tolerance:
                    raise ValueError("Target reflection is outside the scattering plane.")
                if target.omega is None:
                    raise ValueError("Target reflection is outside the Ewald sphere.")
                omega, tt, qx, qz = target.omega, target.two_theta, target.qx, target.qz
            elif mode == "angles":
                omega, tt = number(self.target_a), number(self.target_b)
                qx, qz = angles_to_q(omega, tt, wavelength)
            else:
                qx, qz = number(self.target_a), number(self.target_b)
                omega, tt = q_to_angles(qx, qz, wavelength)
            span_x, span_y = number(self.span_x), number(self.span_y)
            if min(span_x, span_y) <= 0:
                raise ValueError("Both display half-ranges must be positive.")
            max_index = int(self.max_index.text())
            points = calculate_reflections(crystal, orientation, wavelength,
                                           max_index, tolerance)
            in_q = self.view.currentData() == "q"
            center_x, center_y = (qx, qz) if in_q else (omega, tt)
            shown = [item for item in points if
                     abs((item.qx if in_q else item.omega) - center_x) <= span_x
                     and abs((item.qz if in_q else item.two_theta) - center_y) <= span_y]
            self.plot.clear()
            xs = [item.qx if in_q else item.omega for item in shown]
            ys = [item.qz if in_q else item.two_theta for item in shown]
            self.plot.plot(xs, ys, pen=None, symbol="o", symbolSize=9,
                           symbolBrush=(53, 139, 220))
            if self.target_marker.isChecked():
                self.plot.plot([center_x], [center_y], pen=None, symbol="x",
                               symbolSize=15, symbolPen=(230, 80, 35))
            if self.labels.isChecked():
                for x, y, point in zip(xs, ys, shown):
                    label = pg.TextItem("(%d %d %d)" % point.hkl, anchor=(0, 1))
                    label.setPos(x, y)
                    self.plot.addItem(label)
            self.plot.setLabel("bottom", "Qx, Å^-1" if in_q else "Omega, °")
            self.plot.setLabel("left", "Qz, Å^-1" if in_q else "2Theta, °")
            self.plot.setXRange(center_x - span_x, center_x + span_x, padding=0)
            self.plot.setYRange(center_y - span_y, center_y + span_y, padding=0)
            self.status.setText(f"{len(shown)} / {len(points)}; Omega={omega:.5g}°, "
                                f"2Theta={tt:.5g}°; Qx={qx:.5g}, Qz={qz:.5g} Å^-1")
        except (ValueError, TypeError, OverflowError) as exc:
            self.status.setText(str(exc))


class RSMPage(QWidget):
    title_key = "qt.rsm_maps"

    def __init__(self, store, radiation_settings, file_service, parent=None):
        super().__init__(parent)
        self.store = store
        self.radiation_settings = radiation_settings
        self.file_service = file_service
        layout = QVBoxLayout(self)
        self.radiation_selector = RadiationSelector(radiation_settings)
        self.radiation_selector.radiation_changed.connect(self.radiation_changed)
        layout.addWidget(self.radiation_selector)
        self.tabs = QTabWidget()
        wavelength = lambda: float(self.radiation_settings.lines()[0][1])
        self.experimental = ExperimentalRSM(store, file_service, wavelength)
        self.calculated = CalculatedRSM(store, file_service, wavelength)
        self.tabs.addTab(self.experimental, "")
        self.tabs.addTab(self.calculated, "")
        layout.addWidget(self.tabs, 1)
        self.retranslate()
        self.refresh_documents()

    def radiation_changed(self):
        self.experimental.draw()
        if self.calculated.crystal is not None:
            self.calculated.draw()

    def refresh_documents(self):
        self.experimental.refresh_documents()
        self.calculated.refresh_documents()

    def retranslate(self):
        self.radiation_selector.retranslate()
        self.experimental.retranslate()
        self.calculated.retranslate()
        self.tabs.setTabText(0, tr("text.experimental_raw"))
        self.tabs.setTabText(1, tr("text.calculated"))
