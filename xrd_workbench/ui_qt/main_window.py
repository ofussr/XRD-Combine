"""Main window of the parallel PySide6 transition preview."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..bruker_raw import read_bruker_raw
from ..cif_document import load_cif_document
from ..io import read_raw_scans, read_scan_file
from ..localization import LANGUAGES, get_language, load_language, set_language, tr
from ..models.project import POLES, STRUCTURES, VIEWER, ProjectStore
from ..models.radiation import RadiationSettings
from ..services.project_files import ProjectFileService
from ..version import APP_VERSION, QT_PREVIEW_VERSION
from .pages import SectionPage
from .project_panel import ProjectPanel
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
            VIEWER: ViewerPage(self.project, self.radiation_settings),
            STRUCTURES: SectionPage(STRUCTURES, "text.structures", self.project),
            POLES: SectionPage(POLES, "text.pole_figures", self.project),
        }
        for workspace in self.WORKSPACES:
            self.sections.addTab(self.pages[workspace], "")
        self.sections.currentChanged.connect(self._workspace_changed)
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

    def retranslate(self) -> None:
        self.setWindowTitle(f"XRD Combine — {QT_PREVIEW_VERSION}")
        self._build_menu()
        self.project_panel.retranslate()
        for index, workspace in enumerate(self.WORKSPACES):
            page = self.pages[workspace]
            page.retranslate()
            self.sections.setTabText(index, tr(page.title_key))
        self._show_project_count()

    def _workspace_changed(self, _index: int) -> None:
        self.project_panel.refresh()
        self.pages[self.current_workspace()].refresh_documents()

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


__all__ = ["MainWindow"]
