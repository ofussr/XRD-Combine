"""Main window of the parallel PySide6 transition preview."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..atom_styles import (
    custom_colours,
    export_custom_colours,
    import_custom_colours,
    palette as atom_palette,
    reset_custom_colours,
    set_palette as set_atom_palette,
)
from ..bruker_raw import read_bruker_raw
from ..cif_document import load_cif_document
from ..io import read_raw_scans, read_scan_file
from ..localization import (
    LANGUAGES,
    get_language,
    load_language,
    localised,
    set_language,
    tr,
)
from ..models.project import (
    CELL_PHASE,
    CIF,
    POLES,
    SCAN,
    STRUCTURES,
    VIEWER,
    ProjectStore,
)
from ..models.radiation import RadiationSettings
from ..models.scan import Scan1D, clone_scan
from ..models.viewer import is_two_theta
from ..services.project_files import ProjectFileService
from ..version import APP_VERSION, QT_PREVIEW_VERSION
from .pages import SectionPage
from .comparison import ComparisonDialog
from .project_panel import ProjectPanel
from .structures_page import StructuresPage
from .viewer_page import ViewerPage


class MainWindow(QMainWindow):
    """Qt main window that owns shared state and incrementally migrated pages."""

    WORKSPACES = (VIEWER, STRUCTURES, POLES)

    def __init__(
        self,
        initial_paths: list[str] | None = None,
        *,
        store: ProjectStore | None = None,
        file_service: ProjectFileService | None = None,
    ) -> None:
        super().__init__()
        set_language(load_language())
        self.project = store or ProjectStore()
        self.radiation_settings = RadiationSettings()
        self.file_service = file_service or ProjectFileService(
            load_cif_document=load_cif_document,
            read_bruker_raw=read_bruker_raw,
            read_raw_scans=read_raw_scans,
            read_scan_file=read_scan_file,
        )
        self._drawer_visible = True
        self.comparison_dialog: ComparisonDialog | None = None

        self.resize(1280, 800)
        self.setMinimumSize(900, 600)
        self._build_central_widget()
        self._build_menu()
        self.project.subscribe(self._project_event)
        self.retranslate()

        if initial_paths:
            QTimer.singleShot(0, lambda: self.import_paths(initial_paths))

    def _build_central_widget(self) -> None:
        central = QWidget()
        body = QHBoxLayout(central)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.project_panel = ProjectPanel(self.project, self.current_workspace)
        self.project_panel.collapse_requested.connect(self.toggle_project_panel)
        self.project_panel.paths_requested.connect(self.import_paths)
        body.addWidget(self.project_panel)

        self.project_rail = QWidget()
        self.project_rail.setFixedWidth(44)
        rail_layout = QVBoxLayout(self.project_rail)
        rail_layout.setContentsMargins(5, 8, 5, 8)
        rail_layout.setSpacing(6)
        self.expand_button = QPushButton(">")
        self.expand_button.setFixedSize(34, 34)
        self.expand_button.clicked.connect(self.toggle_project_panel)
        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(34, 34)
        self.add_button.clicked.connect(self.project_panel.open_files)
        rail_layout.addWidget(self.expand_button)
        rail_layout.addWidget(self.add_button)
        rail_layout.addStretch(1)
        self.project_rail.hide()
        body.addWidget(self.project_rail)

        self.sections = QTabWidget()
        self.pages = {
            VIEWER: ViewerPage(
                self.project,
                self.radiation_settings,
                on_open_comparison=self.open_comparison,
                on_open_reflection_table=self.open_reflection_table,
            ),
            STRUCTURES: StructuresPage(
                self.project,
                self.radiation_settings,
                on_import_paths=self.import_paths,
            ),
            POLES: SectionPage(POLES, "text.pole_figures", self.project),
        }
        for workspace in self.WORKSPACES:
            self.sections.addTab(self.pages[workspace], "")
        self.sections.currentChanged.connect(self._workspace_changed)
        self.pages[VIEWER].radiation_selector.radiation_changed.connect(
            self.pages[STRUCTURES].sync_radiation
        )
        self.pages[STRUCTURES].radiation_selector.radiation_changed.connect(
            self._sync_viewer_radiation
        )
        body.addWidget(self.sections, 1)

        self.setCentralWidget(central)
        self.statusBar()

    def _build_menu(self) -> None:
        self.menuBar().clear()

        file_menu = self.menuBar().addMenu(tr("text.file"))
        open_action = QAction(tr("text.open_measurements"), self)
        open_action.triggered.connect(self.project_panel.open_files)
        file_menu.addAction(open_action)
        cif_action = QAction(tr("text.add_cif"), self)
        cif_action.triggered.connect(self.project_panel.open_cif)
        file_menu.addAction(cif_action)
        phase_action = QAction(tr("text.new_cell_phase"), self)
        phase_action.triggered.connect(self.project_panel.new_cell_phase)
        file_menu.addAction(phase_action)
        file_menu.addSeparator()
        exit_action = QAction(tr("text.exit"), self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu(tr("text.edit"))
        language_menu = edit_menu.addMenu(tr("text.language"))
        language_group = QActionGroup(self)
        language_group.setExclusive(True)
        for code, label in LANGUAGES.items():
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == get_language())
            action.triggered.connect(
                lambda checked=False, language=code: self.change_language(language)
            )
            language_group.addAction(action)
            language_menu.addAction(action)
        self._language_group = language_group

        atom_menu = edit_menu.addMenu(tr("text.atom_colours"))
        atom_group = QActionGroup(self)
        atom_group.setExclusive(True)
        for code, label in (
            ("jmol", "Jmol"),
            ("cpk", "CPK"),
            ("molcas_gv", "MOLCAS GV"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == atom_palette())
            action.triggered.connect(
                lambda checked=False, value=code: self.change_atom_palette(value)
            )
            atom_group.addAction(action)
            atom_menu.addAction(action)
        atom_menu.addSeparator()
        import_colours = QAction(tr("text.import_custom_colours"), self)
        import_colours.triggered.connect(self.import_atom_colours)
        atom_menu.addAction(import_colours)
        export_colours = QAction(tr("text.export_custom_colours"), self)
        export_colours.triggered.connect(self.export_atom_colours)
        atom_menu.addAction(export_colours)
        reset_colours = QAction(tr("text.reset_custom_colours"), self)
        reset_colours.triggered.connect(self.reset_atom_colours)
        atom_menu.addAction(reset_colours)
        self._atom_group = atom_group

        section_menu = self.menuBar().addMenu(tr("text.section"))
        for index, key in enumerate(
            ("text.viewer", "text.structures", "text.pole_figures")
        ):
            action = QAction(tr(key), self)
            action.triggered.connect(
                lambda checked=False, target=index: self.sections.setCurrentIndex(target)
            )
            section_menu.addAction(action)

        help_menu = self.menuBar().addMenu(tr("text.help"))
        about_action = QAction(tr("text.about"), self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

    def current_workspace(self) -> str:
        index = self.sections.currentIndex() if hasattr(self, "sections") else 0
        return self.WORKSPACES[index] if 0 <= index < len(self.WORKSPACES) else VIEWER

    def toggle_project_panel(self) -> None:
        self._drawer_visible = not self._drawer_visible
        self.project_panel.setVisible(self._drawer_visible)
        self.project_rail.setVisible(not self._drawer_visible)

    def change_language(self, language: str) -> None:
        set_language(language, persist=True)
        self.retranslate()

    def _refresh_atom_styles(self) -> None:
        structures = self.pages.get(STRUCTURES)
        if structures is not None:
            structures.refresh_atom_styles()

    def change_atom_palette(self, value: str) -> None:
        try:
            set_atom_palette(value)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, tr("text.atom_colours"), str(error))
            return
        self._refresh_atom_styles()

    def import_atom_colours(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            tr("text.import_custom_colours"),
            "",
            "Text files (*.txt *.dat *.ini);;" + tr("text.all_files") + " (*)",
        )
        if not path:
            return
        try:
            colours = import_custom_colours(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, tr("text.atom_colours"), str(error))
            return
        self._refresh_atom_styles()
        QMessageBox.information(
            self,
            tr("text.atom_colours"),
            tr("qt.atom_colours_imported", count=len(colours)),
        )

    def export_atom_colours(self) -> None:
        if not custom_colours():
            QMessageBox.information(
                self,
                tr("text.atom_colours"),
                tr("qt.no_custom_atom_colours"),
            )
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            tr("text.export_custom_colours"),
            "xrd_combine_atom_colours.txt",
            "Text files (*.txt);;" + tr("text.all_files") + " (*)",
        )
        if not path:
            return
        try:
            export_custom_colours(path)
        except OSError as error:
            QMessageBox.warning(self, tr("text.save_error"), str(error))

    def reset_atom_colours(self) -> None:
        try:
            reset_custom_colours()
        except OSError as error:
            QMessageBox.warning(self, tr("text.atom_colours"), str(error))
            return
        self._refresh_atom_styles()

    def retranslate(self) -> None:
        self.setWindowTitle(f"XRD Combine — {QT_PREVIEW_VERSION}")
        self._build_menu()
        self.project_panel.retranslate()
        for index, workspace in enumerate(self.WORKSPACES):
            page = self.pages[workspace]
            page.retranslate()
            self.sections.setTabText(index, tr(page.title_key))
        if self.comparison_dialog is not None:
            self.comparison_dialog.setWindowTitle(tr("text.substrate_comparison"))
            self.comparison_dialog.page.retranslate()
        self._show_project_count()

    def _workspace_changed(self, _index: int) -> None:
        self.project_panel.refresh()
        self.pages[self.current_workspace()].refresh_documents()

    def _sync_viewer_radiation(self) -> None:
        viewer = self.pages[VIEWER]
        viewer.radiation_selector.sync_from_settings()
        viewer._radiation_changed()

    def import_paths(self, paths) -> list:
        workspace = self.current_workspace()
        loaded = []
        errors: list[str] = []
        for value in paths:
            try:
                documents = self.file_service.load_path(self.project, value)
                for document in documents:
                    loaded.append(document)
                    if self.project.compatible(document.kind, workspace):
                        self.project.assign(document.uid, workspace, True)
            except Exception as exc:
                errors.append(f"{Path(value).name}: {exc}")
        if errors:
            QMessageBox.warning(
                self,
                tr("qt.some_files_failed"),
                "\n\n".join(errors[:12]),
            )
        self._show_project_count()
        return loaded

    def _project_event(self, _event, _document, _workspace) -> None:
        for page in self.pages.values():
            page.refresh_documents()
        self._show_project_count()

    def _show_project_count(self) -> None:
        self.statusBar().showMessage(
            tr("qt.project_objects_count", count=len(self.project.documents))
        )

    def _comparison_documents(self):
        return [
            document
            for document in self.project.assigned_documents(VIEWER, kind=SCAN)
            if is_two_theta(document.payload.axis_name)
        ]

    def open_comparison(self) -> None:
        """Open the one application-modal substrate-comparison window."""

        if self.comparison_dialog is not None:
            self.comparison_dialog.show()
            self.comparison_dialog.raise_()
            self.comparison_dialog.activateWindow()
            return
        documents = self._comparison_documents()
        if not documents:
            QMessageBox.information(
                self,
                localised("Comparison", "Comparaison", "Сравнение"),
                localised(
                    "Assign at least one 2θ measurement to Viewer first.",
                    "Affectez d’abord au moins une mesure 2θ à la section Visualisation.",
                    "Сначала подключите к разделу «Просмотр» хотя бы одно измерение 2θ.",
                ),
            )
            return
        dialog = ComparisonDialog(
            [document.payload for document in documents],
            self,
            on_send_viewer=self.send_to_viewer,
            on_send_correction=self.send_to_correction,
        )
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.finished.connect(self._comparison_closed)
        self.comparison_dialog = dialog
        dialog.open()

    def _comparison_closed(self, _result: int = 0) -> None:
        self.comparison_dialog = None

    def _document_for_scan(self, scan: Scan1D):
        document = next(
            (
                current
                for current in self.project.documents.values()
                if current.kind == SCAN and current.payload is scan
            ),
            None,
        )
        if document is None:
            document = self.project.add_scan(clone_scan(scan), derived=True)
        return document

    def send_to_viewer(self, scan: Scan1D) -> None:
        document = self._document_for_scan(scan)
        self.project.assign(document.uid, VIEWER, True)
        self.sections.setCurrentIndex(0)
        viewer = self.pages[VIEWER]
        viewer.refresh_documents()
        viewer.select_uid(document.uid)

    def send_to_correction(self, scan: Scan1D) -> None:
        document = self._document_for_scan(scan)
        self.project.assign(document.uid, VIEWER, True)
        self.sections.setCurrentIndex(0)
        viewer = self.pages[VIEWER]
        viewer.refresh_documents()
        viewer.select_uid(document.uid, open_processing=True)

    def open_reflection_table(self, uid: str) -> None:
        document = self.project.documents.get(uid)
        if document is None or document.kind not in {CIF, CELL_PHASE}:
            return
        self.project.assign(uid, STRUCTURES, True)
        self.sections.setCurrentIndex(1)
        structures = self.pages[STRUCTURES]
        structures.refresh_documents()
        structures.select_reflection_table()

    def about(self) -> None:
        message = QMessageBox(self)
        message.setWindowTitle("XRD Combine")
        message.setIcon(QMessageBox.Icon.Information)
        message.setText(f"XRD Combine {QT_PREVIEW_VERSION}")
        message.setInformativeText(
            tr("qt.preview_about", stable_version=APP_VERSION)
            + "\n\nMikhail Mirushchenko\nmiruschenko98@gmail.com"
        )
        notices = Path(__file__).resolve().parents[2] / "THIRD_PARTY_NOTICES.txt"
        try:
            message.setDetailedText(notices.read_text(encoding="utf-8"))
        except OSError:
            pass
        message.exec()

    def closeEvent(self, event) -> None:
        if self.comparison_dialog is not None:
            self.comparison_dialog.close()
            self.comparison_dialog = None
        super().closeEvent(event)


__all__ = ["MainWindow"]
