"""Main window of XRD Combine."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QEvent, QRect, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
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
from ..application_resources import resource_path
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
    RSM,
    SCAN,
    STRUCTURES,
    VIEWER,
    ProjectStore,
)
from ..models.radiation import RadiationSettings
from ..models.scan import Scan1D, clone_scan
from ..models.viewer import is_two_theta
from ..services.project_files import ProjectFileService
from ..version import APP_VERSION
from .pole_figures import PolesPage
from .rsm_page import RSMPage
from .comparison import ComparisonDialog
from .project_panel import ProjectPanel
from .structures_page import StructuresPage
from .viewer_page import ViewerPage
from .reference_peaks import ReferencePeaksDialog
from .debug_dialog import DebugDialog
from .plot_renderer import PlotRendererController
from .theme import THEME_KEYS, application_theme


class MainWindow(QMainWindow):
    """Qt main window that owns the shared project and application pages."""

    WORKSPACES = (VIEWER, STRUCTURES, POLES, RSM)
    PROJECT_PANEL_WIDTH_KEY = "project_data/panel_width"

    def __init__(
        self,
        initial_paths: list[str] | None = None,
        *,
        store: ProjectStore | None = None,
        file_service: ProjectFileService | None = None,
        theme_controller=None,
        plot_renderer_controller=None,
    ) -> None:
        super().__init__()
        self._window_refresh_generation = 0
        self._screen_window_handle = None
        self._replaying_native_resize = False
        self._surface_refresh_timers = []
        for delay in (0, 80, 250):
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(delay)
            timer.timeout.connect(self._refresh_window_surface)
            self._surface_refresh_timers.append(timer)
        set_language(load_language())
        self.theme_controller = theme_controller or application_theme()
        self.plot_renderer_controller = (
            plot_renderer_controller
            or PlotRendererController(
                getattr(self.theme_controller, "settings", None),
                self,
            )
        )
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
        self.theme_controller.changed.connect(self._sync_theme_actions)

        if initial_paths:
            QTimer.singleShot(0, lambda: self.import_paths(initial_paths))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._bind_screen_change()
        self._normalise_managed_window_geometry()
        self._schedule_window_surface_refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "sections"):
            self._normalise_managed_window_geometry()
            self._schedule_window_surface_refresh()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        if hasattr(self, "_window_refresh_generation"):
            self._schedule_window_surface_refresh()

    def nativeEvent(self, event_type, message):
        """Replay Windows client-size changes that Qt occasionally loses.

        With mixed-DPI monitors Windows can resize the native frame while the
        QWidget backing store keeps the previous normal-window rectangle.  A
        delayed replay of the real client size makes Qt reconcile the native
        frame and its QWidget hierarchy without recreating the window.
        """

        if (
            sys.platform == "win32"
            and not getattr(self, "_replaying_native_resize", False)
            and hasattr(self, "_window_refresh_generation")
        ):
            try:
                from ctypes import wintypes

                native_message = wintypes.MSG.from_address(int(message))
                if native_message.message in {
                    0x0005,  # WM_SIZE
                    0x0047,  # WM_WINDOWPOSCHANGED
                    0x007E,  # WM_DISPLAYCHANGE
                    0x02E0,  # WM_DPICHANGED
                }:
                    self._schedule_window_surface_refresh()
            except (ImportError, OSError, TypeError, ValueError):
                # The normal Qt event path remains available if a particular
                # Python/Windows runtime does not expose MSG as an address.
                pass
        return super().nativeEvent(event_type, message)

    def event(self, event) -> bool:
        result = super().event(event)
        event_type = event.type()
        refresh_types = {
            QEvent.Type.ScreenChangeInternal,
            QEvent.Type.WindowStateChange,
        }
        device_pixel_ratio_change = getattr(
            QEvent.Type,
            "DevicePixelRatioChange",
            None,
        )
        if device_pixel_ratio_change is not None:
            refresh_types.add(device_pixel_ratio_change)
        if event_type in refresh_types and hasattr(
            self,
            "_window_refresh_generation",
        ):
            self._bind_screen_change()
            self._schedule_window_surface_refresh()
        return result

    def _bind_screen_change(self) -> None:
        handle = self.windowHandle()
        if handle is None or handle is self._screen_window_handle:
            return
        previous = self._screen_window_handle
        if previous is not None:
            for signal_name in (
                "screenChanged",
                "widthChanged",
                "heightChanged",
                "xChanged",
                "yChanged",
            ):
                try:
                    getattr(previous, signal_name).disconnect(
                        self._window_geometry_changed
                    )
                except (AttributeError, RuntimeError, TypeError):
                    pass
        for signal_name in (
            "screenChanged",
            "widthChanged",
            "heightChanged",
            "xChanged",
            "yChanged",
        ):
            getattr(handle, signal_name).connect(self._window_geometry_changed)
        self._screen_window_handle = handle

    def _window_geometry_changed(self, *_args) -> None:
        self._schedule_window_surface_refresh()

    def _schedule_window_surface_refresh(self) -> None:
        """Repair stale backing-store geometry after Windows screen changes."""

        self._window_refresh_generation += 1
        for timer in self._surface_refresh_timers:
            timer.start()

    def _normalise_managed_window_geometry(self) -> None:
        """Place QMainWindow bands in local, never screen, coordinates."""

        central = self.centralWidget()
        if central is None:
            return

        window_rect = self.rect()
        menu = self.menuBar()
        status = self.statusBar()
        menu_height = menu.sizeHint().height() if menu.isVisible() else 0
        status_height = status.sizeHint().height() if status.isVisible() else 0

        menu_geometry = QRect(0, 0, window_rect.width(), menu_height)
        central_geometry = QRect(
            0,
            menu_height,
            window_rect.width(),
            max(0, window_rect.height() - menu_height - status_height),
        )
        status_geometry = QRect(
            0,
            max(menu_height, window_rect.height() - status_height),
            window_rect.width(),
            status_height,
        )
        if menu.geometry() != menu_geometry:
            menu.setGeometry(menu_geometry)
        if central.geometry() != central_geometry:
            central.setGeometry(central_geometry)
        if status.geometry() != status_geometry:
            status.setGeometry(status_geometry)

        central_layout = central.layout()
        if central_layout is not None:
            central_layout.invalidate()
            central_layout.setGeometry(central.rect())
            central_layout.activate()
        for child in central.findChildren(QWidget):
            child_layout = child.layout()
            if child_layout is not None:
                child_layout.invalidate()
                child_layout.activate()

    def _replay_windows_client_size(self) -> None:
        """Send Qt the current Win32 client size without changing window state."""

        if sys.platform != "win32" or self.isMinimized():
            return
        try:
            from ctypes import byref, windll
            from ctypes import wintypes

            rectangle = wintypes.RECT()
            window_id = int(self.winId())
            if not windll.user32.GetClientRect(
                wintypes.HWND(window_id),
                byref(rectangle),
            ):
                return
            width = max(0, rectangle.right - rectangle.left)
            height = max(0, rectangle.bottom - rectangle.top)
            if not width or not height or width > 0xFFFF or height > 0xFFFF:
                return
            size_state = 2 if self.isMaximized() else 0
            size_parameter = width | (height << 16)
            self._replaying_native_resize = True
            try:
                windll.user32.SendMessageW(
                    wintypes.HWND(window_id),
                    0x0005,  # WM_SIZE
                    size_state,
                    size_parameter,
                )
            finally:
                self._replaying_native_resize = False
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            return

    def _refresh_window_surface(self, generation: int | None = None) -> None:
        if generation is not None and generation != self._window_refresh_generation:
            return
        if not self.isVisible():
            return
        self._replay_windows_client_size()
        updates_enabled = self.updatesEnabled()
        if updates_enabled:
            self.setUpdatesEnabled(False)
        self._normalise_managed_window_geometry()
        central = self.centralWidget()
        if central is not None:
            central.updateGeometry()
        if updates_enabled:
            self.setUpdatesEnabled(True)
        if central is not None:
            central.update()
            for child in central.findChildren(QWidget):
                if child.isVisible():
                    child.update()
        self.update()
        self.repaint()
        handle = self.windowHandle()
        if handle is not None:
            handle.requestUpdate()

    def _build_central_widget(self) -> None:
        central = QWidget()
        body = QHBoxLayout(central)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        panel_settings = getattr(self.theme_controller, "settings", None)
        self.project_panel = ProjectPanel(
            self.project,
            self.current_workspace,
            settings=panel_settings,
        )
        self.project_panel.collapse_requested.connect(self.toggle_project_panel)
        self.project_panel.paths_requested.connect(self.import_paths)

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
                on_open_structure=self.open_structure,
                on_open_poles=self.open_calculated_poles,
                plot_renderer_controller=self.plot_renderer_controller,
            ),
            STRUCTURES: StructuresPage(
                self.project,
                self.radiation_settings,
                on_import_paths=self.import_paths,
                plot_renderer_controller=self.plot_renderer_controller,
            ),
            POLES: PolesPage(
                self.project,
                self.radiation_settings,
                self.file_service,
                plot_renderer_controller=self.plot_renderer_controller,
            ),
            RSM: RSMPage(
                self.project,
                self.radiation_settings,
                self.file_service,
            ),
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
        for workspace in (VIEWER, STRUCTURES):
            self.pages[workspace].radiation_selector.radiation_changed.connect(
                self.pages[POLES].sync_radiation
            )
        self.pages[POLES].radiation_selector.radiation_changed.connect(
            self._sync_viewer_radiation
        )
        self.pages[POLES].radiation_selector.radiation_changed.connect(
            self.pages[STRUCTURES].sync_radiation
        )
        self.pages[RSM].radiation_selector.radiation_changed.connect(
            self._sync_viewer_radiation
        )
        self.pages[RSM].radiation_selector.radiation_changed.connect(
            self.pages[STRUCTURES].sync_radiation
        )
        self.pages[RSM].radiation_selector.radiation_changed.connect(
            self.pages[POLES].sync_radiation
        )
        for workspace in (VIEWER, STRUCTURES, POLES):
            self.pages[workspace].radiation_selector.radiation_changed.connect(
                self.pages[RSM].radiation_selector.sync_from_settings
            )
            self.pages[workspace].radiation_selector.radiation_changed.connect(
                self.pages[RSM].radiation_changed
            )
        self.project_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.project_splitter.setChildrenCollapsible(False)
        self.project_splitter.addWidget(self.project_panel)
        self.project_splitter.addWidget(self.sections)
        self.project_splitter.setStretchFactor(0, 0)
        self.project_splitter.setStretchFactor(1, 1)
        self._project_panel_width = self._saved_project_panel_width()
        self.project_splitter.setSizes([self._project_panel_width, 1000])
        self.project_splitter.splitterMoved.connect(
            self._project_splitter_moved
        )
        body.addWidget(self.project_splitter, 1)

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
        self.reference_peaks_action = QAction(tr("text.reference_peaks_for_correction"), self)
        self.reference_peaks_action.triggered.connect(self.open_reference_peaks)
        edit_menu.addAction(self.reference_peaks_action)
        theme_menu = edit_menu.addMenu(tr("qt.application_theme"))
        self.theme_actions = {}
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        for mode, key in THEME_KEYS.items():
            action = QAction(tr(key), self)
            action.setCheckable(True)
            action.setChecked(mode == self.theme_controller.mode)
            action.triggered.connect(lambda checked=False, mode=mode: self.change_theme(mode))
            self._theme_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[mode] = action
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
            ("text.viewer", "text.structures", "text.pole_figures", "qt.rsm_maps")
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

    def _saved_project_panel_width(self) -> int:
        settings = getattr(self.theme_controller, "settings", None)
        saved = settings.value(self.PROJECT_PANEL_WIDTH_KEY, 330) if settings else 330
        try:
            return max(220, min(900, int(saved)))
        except (TypeError, ValueError):
            return 330

    def _project_splitter_moved(self, *_args) -> None:
        if not self._drawer_visible:
            return
        sizes = self.project_splitter.sizes()
        if not sizes or sizes[0] < self.project_panel.minimumWidth():
            return
        self._project_panel_width = int(sizes[0])
        settings = getattr(self.theme_controller, "settings", None)
        if settings is not None:
            settings.setValue(
                self.PROJECT_PANEL_WIDTH_KEY,
                self._project_panel_width,
            )

    def toggle_project_panel(self) -> None:
        if self._drawer_visible:
            sizes = self.project_splitter.sizes()
            if sizes and sizes[0] >= self.project_panel.minimumWidth():
                self._project_panel_width = int(sizes[0])
        self._drawer_visible = not self._drawer_visible
        self.project_panel.setVisible(self._drawer_visible)
        self.project_rail.setVisible(not self._drawer_visible)
        if self._drawer_visible:
            total = max(sum(self.project_splitter.sizes()), self.width() - 44)
            self.project_splitter.setSizes(
                [self._project_panel_width, max(1, total - self._project_panel_width)]
            )

    def change_language(self, language: str) -> None:
        set_language(language, persist=True)
        self.retranslate()

    def change_theme(self, mode: str) -> None:
        try:
            self.theme_controller.set_mode(mode)
        except OSError:
            QMessageBox.warning(self, tr("qt.application_theme"), tr("qt.theme_save_error"))
            self._sync_theme_actions()

    def _sync_theme_actions(self, _mode=None) -> None:
        for mode, action in self.theme_actions.items():
            action.setChecked(mode == self.theme_controller.mode)

    def change_plot_renderer(self, mode: str) -> None:
        try:
            self.plot_renderer_controller.set_mode(mode)
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, tr("qt.debug"), str(error))

    def open_debug(self) -> None:
        dialog = DebugDialog(self.plot_renderer_controller, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.change_plot_renderer(dialog.selected_renderer())

    def open_reference_peaks(self) -> None:
        ReferencePeaksDialog(self).exec()

    def _refresh_atom_styles(self) -> None:
        structures = self.pages.get(STRUCTURES)
        if structures is not None:
            structures.refresh_atom_styles()
        poles = self.pages.get(POLES)
        if poles is not None:
            poles.refresh_atom_styles()

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
        self.setWindowTitle(f"XRD Combine — {APP_VERSION}")
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
            plot_renderer_controller=self.plot_renderer_controller,
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

    def open_structure(self, uid: str) -> None:
        document = self.project.documents.get(uid)
        if document is None or document.kind not in {CIF, CELL_PHASE}:
            return
        self.project.assign(uid, STRUCTURES, True)
        self.sections.setCurrentIndex(1)
        self.pages[STRUCTURES].refresh_documents()
        self.pages[STRUCTURES].tabs.setCurrentIndex(0)

    def open_calculated_poles(self, uid: str) -> None:
        document = self.project.documents.get(uid)
        if document is None or document.kind not in {CIF, CELL_PHASE}:
            return
        self.sections.setCurrentIndex(2)
        self.pages[POLES].open_cif(uid)

    def about(self) -> None:
        message = QMessageBox(self)
        message.setWindowTitle("XRD Combine")
        logo_path = resource_path("xrd_combine.png")
        logo_loaded = False
        if logo_path is not None:
            logo = QPixmap(str(logo_path))
            if not logo.isNull():
                message.setIconPixmap(
                    logo.scaled(
                        128,
                        128,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                logo_loaded = True
        if not logo_loaded:
            message.setIcon(QMessageBox.Icon.Information)
        message.setText(f"XRD Combine {APP_VERSION}")
        message.setInformativeText(
            "Mikhail Mirushchenko\nmiruschenko98@gmail.com"
        )
        notices = Path(__file__).resolve().parents[2] / "THIRD_PARTY_NOTICES.txt"
        try:
            message.setDetailedText(notices.read_text(encoding="utf-8"))
        except OSError:
            pass
        message.setStandardButtons(QMessageBox.StandardButton.Ok)
        debug_button = message.addButton(
            tr("qt.debug"),
            QMessageBox.ButtonRole.ActionRole,
        )
        message.exec()
        if message.clickedButton() is debug_button:
            self.open_debug()

    def closeEvent(self, event) -> None:
        if self.comparison_dialog is not None:
            self.comparison_dialog.close()
            self.comparison_dialog = None
        super().closeEvent(event)


__all__ = ["MainWindow"]
