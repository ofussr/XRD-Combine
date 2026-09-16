"""Pole-figure workspace connected to shared project objects and radiation."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from ..localization import tr
from ..models.project import POLES, POLE_DATA, CIF, CELL_PHASE
from ..services.experimental_pole import load_experimental_pole
from .radiation import RadiationSelector
from .experimental_pole import ExperimentalPolePage
from .calculated_pole import CalculatedPolePage
from .pole_widgets import show_error


class PolesPage(QWidget):
    title_key = "text.pole_figures"

    def __init__(
        self,
        store,
        radiation_settings,
        file_service,
        parent=None,
        *,
        plot_renderer_controller=None,
    ):
        super().__init__(parent)
        self.workspace = POLES
        self.store = store
        self.radiation_settings = radiation_settings
        self.file_service = file_service
        self.plot_renderer_controller = plot_renderer_controller
        self._raw_token = None
        self._radiation_signature = tuple(radiation_settings.lines())
        self._refreshing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.radiation_selector = RadiationSelector(radiation_settings)
        self.radiation_selector.radiation_changed.connect(self.sync_radiation)
        layout.addWidget(self.radiation_selector)
        self.tabs = QTabWidget()
        self.experimental = ExperimentalPolePage(on_open_raw=self.open_raw)
        self.calculated = CalculatedPolePage(
            radiation_settings.lines,
            on_open_cif=self.open_cif,
            on_add_overlay=self.add_overlay,
            on_remove_overlay=self.remove_overlay,
            overlay_documents_provider=self.structure_documents,
        )
        if self.plot_renderer_controller is not None:
            self.experimental.set_plot_renderer(
                self.plot_renderer_controller.mode
            )
            self.calculated.set_plot_renderer(
                self.plot_renderer_controller.mode
            )
            self.plot_renderer_controller.changed.connect(
                self.experimental.set_plot_renderer
            )
            self.plot_renderer_controller.changed.connect(
                self.calculated.set_plot_renderer
            )
        self.tabs.addTab(self.experimental, "")
        self.tabs.addTab(self.calculated, "")
        layout.addWidget(self.tabs, 1)
        self.retranslate()
        self.refresh_documents()

    def structure_documents(self):
        return [document for document in self.store.documents.values()
                if document.kind in {CIF, CELL_PHASE}]

    def refresh_documents(self):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.sync_radiation()
            assigned = self.store.assigned_documents(POLES)
            structures = [item for item in assigned if item.kind in {CIF, CELL_PHASE}]
            payloads = [item.payload for item in structures]
            page = self.calculated
            # Keep the surviving layer's full orientation and display settings.
            current = [page.cif_document]
            if page.overlay_layer is not None:
                current.append(page.overlay_layer.document)
            for payload in current:
                if payload is not None and not any(payload is target for target in payloads):
                    page.remove_document(payload)
            for document in structures:
                if page.cif_document is document.payload:
                    continue
                if page.overlay_layer is not None and page.overlay_layer.document is document.payload:
                    continue
                if page.cif_document is None:
                    page.load_document(document.payload)
                elif page.overlay_layer is None:
                    page.load_overlay_document(document.payload)
                self.tabs.setCurrentIndex(1)
            page.refresh_overlay_choices()
            raw_documents = [item for item in assigned if item.kind == POLE_DATA]
            raw_document = raw_documents[-1] if raw_documents else None
            token = (raw_document.uid, id(raw_document.payload)) if raw_document is not None else None
            if token != self._raw_token:
                self._raw_token = token
                if raw_document is None:
                    self.experimental.clear_data()
                else:
                    measurement = load_experimental_pole(raw_document.source, raw_document.payload)
                    self.experimental.load_measurement(measurement)
                    self.tabs.setCurrentIndex(0)
        except (OSError, ValueError) as error:
            show_error(tr("text.pole_figures"), error, self)
        finally:
            self._refreshing = False

    def open_raw(self, path):
        try:
            measurement = load_experimental_pole(path)
            if measurement.raw is not None:
                documents = self.file_service.load_path(self.store, path)
                document = next(item for item in documents if item.kind == POLE_DATA)
            else:
                document = self.store.add_pole_document(path, measurement)
            self.store.assign(document.uid, POLES, True)
            self.refresh_documents()
            self.tabs.setCurrentIndex(0)
        except (OSError, ValueError) as error:
            show_error(tr("text.open_raw"), error, self)

    def open_cif(self, path):
        try:
            document = self.store.documents.get(path)
            if document is None:
                document = self.file_service.load_cif(self.store, path)
            self.store.assign(document.uid, POLES, True)
            self.refresh_documents()
            self.tabs.setCurrentIndex(1)
        except (OSError, ValueError) as error:
            show_error(tr("text.could_not_open_cif"), error, self)

    def add_overlay(self, value):
        if self.calculated.overlay_layer is not None:
            return
        try:
            document = self.store.documents.get(value)
            if document is None:
                document = self.file_service.load_cif(self.store, value)
            if document.kind not in {CIF, CELL_PHASE}:
                return
            if self.calculated.cif_document is None:
                self.store.assign(document.uid, POLES, True)
            else:
                self.calculated.load_overlay_document(document.payload)
                if self.calculated.overlay_layer is None:
                    return
                self.store.assign(document.uid, POLES, True, additive=True)
            self.refresh_documents()
            self.tabs.setCurrentIndex(1)
        except (OSError, ValueError) as error:
            show_error(tr("text.add_overlay"), error, self)

    def remove_overlay(self, payload):
        if self.calculated.cif_document is payload:
            return
        for document in self.structure_documents():
            if document.payload is payload:
                self.store.assign(document.uid, POLES, False)
                break
        self.refresh_documents()

    def sync_radiation(self):
        self.radiation_selector.sync_from_settings()
        signature = tuple(self.radiation_settings.lines())
        if signature != self._radiation_signature:
            self._radiation_signature = signature
            self.calculated.radiation_changed()

    def refresh_atom_styles(self):
        self.calculated.refresh_atom_styles()

    def retranslate(self):
        self.radiation_selector.retranslate()
        self.tabs.setTabText(0, tr("text.experimental_raw"))
        self.tabs.setTabText(1, tr("text.calculated"))
        self.experimental.retranslate()
        self.calculated.retranslate()
