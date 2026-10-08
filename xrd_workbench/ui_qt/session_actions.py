"""Qt commands for portable project and workspace files."""
from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..atom_styles import apply_session_colours, palette, custom_colours
from ..localization import get_language, set_language, localised, tr
from ..models.project import VIEWER
from ..services.sessions import (PROJECT_EXTENSION, TAB_EXTENSION, SessionFormatError,
    SessionCancelled, write_session, load_session, merge_tab, read_manifest)
from .plot_renderer import pyqtgraph_available
from .structure_preparation import StructurePreparation


class SessionActions:
    def add_session_actions(self, menu):
        for action in getattr(self, 'session_actions', {}).values():
            action.setShortcut(QKeySequence())
            action.deleteLater()
        self.session_actions = {}
        for key, method, shortcut in (
            ('open_project', self.open_project, 'Ctrl+O'),
            ('save_project', self.save_project, 'Ctrl+S'),
            ('save_project_as', self.save_project_as, 'Ctrl+Shift+S'),
            ('load_tab', self.load_tab, ''),
            ('save_tab', self.save_tab, ''),
        ):
            action = QAction(tr('session.' + key), self)
            action.triggered.connect(lambda checked=False, callback=method: callback())
            if shortcut:
                action.setShortcut(shortcut)
            menu.addAction(action)
            self.session_actions[key] = action

    def _session_filter(self, tab=False):
        return ('XRD Combine tab (*' + TAB_EXTENSION + ')' if tab else
                'XRD Combine project (*' + PROJECT_EXTENSION + ')')

    def _application_state(self):
        normal = self.normalGeometry() if self.isMaximized() else self.geometry()
        return {'language': get_language(), 'theme': self.theme_controller.mode,
            'renderer': self.plot_renderer_controller.mode, 'atom_palette': palette(),
            'atom_colours': custom_colours(), 'workspace': self.current_workspace(),
            'window_size': [normal.width(), normal.height()], 'maximized': self.isMaximized(),
            'drawer_visible': self._drawer_visible,
            'panel_width': max(220, min(900, self._project_panel_width)),
            'tree_widths': [max(1, self.project_panel.tree.columnWidth(i)) for i in range(3)]}

    def save_project(self, path=None):
        return self._save_session(path or self._project_path, tab=False)

    def save_project_as(self, path=None):
        return self._save_session(path, tab=False)

    def save_tab(self, path=None):
        return self._save_session(path, tab=True)

    def _save_session(self, path, *, tab):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, tr('session.save_tab' if tab else 'session.save_project'),
                self.current_workspace() + TAB_EXTENSION if tab else str(self._project_path or 'project' + PROJECT_EXTENSION),
                self._session_filter(tab))
        if not path:
            return False
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(TAB_EXTENSION if tab else PROJECT_EXTENSION)
        try:
            write_session(path, self.project, self.radiation_settings,
                scope=self.current_workspace() if tab else 'project',
                application=None if tab else self._application_state())
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, tr('text.save_error'), str(error))
            return False
        if not tab:
            self._project_path = path.resolve()
        self.statusBar().showMessage(tr('session.saved', path=str(path)), 8000)
        return True

    def open_project(self, path=None, *, resolver=None):
        return self._load_session(path, tab=False, resolver=resolver)

    def load_tab(self, path=None, *, resolver=None):
        return self._load_session(path, tab=True, resolver=resolver)

    def _load_session(self, path, *, tab, resolver):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr('session.load_tab' if tab else 'session.open_project'),
                '', self._session_filter(tab))
        if not path:
            return False
        try:
            manifest = read_manifest(path)
            if (manifest['scope'] == 'project') == tab:
                raise SessionFormatError(localised(
                    'Choose a single-tab file.' if tab else 'Choose a complete project file.',
                    'Choisissez un fichier de section.' if tab else 'Choisissez un projet complet.',
                    'Выберите файл отдельной вкладки.' if tab else 'Выберите файл всего проекта.'))
            loaded = load_session(path, self.file_service, resolver=resolver or self._resolve_session_source)
            if tab:
                loaded = merge_tab(self.project, loaded)
            self._install_session(loaded)
        except SessionCancelled:
            return False
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, tr('session.load_error'), str(error))
            return False
        if not tab:
            self._project_path = Path(path).resolve()
        self.statusBar().showMessage(tr('session.loaded', path=str(path)), 8000)
        if loaded.warnings:
            QMessageBox.warning(self, tr('session.source_warning'), '\n\n'.join(loaded.warnings[:12]))
        return True

    def _resolve_session_source(self, issue):
        dialog = QMessageBox(self)
        dialog.setWindowTitle(tr('session.source_warning'))
        message = {
            'missing': localised('Source file not found:', 'Fichier source introuvable :', 'Исходный файл не найден:'),
            'changed': localised('Source file has changed since saving:', 'Le fichier source a changé :', 'Исходный файл изменился после сохранения:'),
            'unreadable': localised('Could not read the source file:', 'Impossible de lire le fichier source :', 'Не удалось прочитать исходный файл:'),
        }[issue.reason]
        dialog.setText(message + '\n' + str(issue.path))
        if issue.reason == 'changed':
            dialog.setInformativeText(localised(
                'Loading the changed file resets source-dependent analysis, orientation and view limits.',
                'Le chargement du fichier modifié réinitialise les résultats, orientations et limites liés à cette source.',
                'При загрузке изменённого файла связанные результаты анализа, ориентация и границы просмотра будут сброшены.'))
        if issue.message:
            dialog.setDetailedText(issue.message)
        locate = dialog.addButton(localised('Locate file…', 'Localiser le fichier…', 'Указать файл…'), QMessageBox.ButtonRole.ActionRole)
        skip = dialog.addButton(localised('Skip source', 'Ignorer la source', 'Пропустить источник'), QMessageBox.ButtonRole.DestructiveRole)
        changed = dialog.addButton(localised('Use changed file', 'Utiliser le fichier modifié', 'Загрузить изменённый файл'), QMessageBox.ButtonRole.AcceptRole) if issue.reason == 'changed' else None
        cancel = dialog.addButton(tr('text.cancel'), QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is skip:
            return None
        if changed is not None and clicked is changed:
            return issue.path, True
        if clicked is locate:
            replacement, _ = QFileDialog.getOpenFileName(self, localised('Locate source file', 'Localiser le fichier source', 'Указать исходный файл'), str(issue.path.parent))
            if replacement:
                return Path(replacement)
            return self._resolve_session_source(issue)
        raise SessionCancelled()

    def _install_session(self, loaded):
        # Build replacement widgets before disposing of the old model/view pair.
        names = ('project', 'radiation_settings', 'structure_preparation', 'pages', 'sections',
            'project_panel', 'project_rail', 'expand_button', 'add_button', 'project_splitter', '_project_panel_width')
        previous = {name: getattr(self, name) for name in names}
        appearance = self._application_state()
        previous_workspace = self.current_workspace()
        old_central = self.centralWidget()
        new_central = None
        self.setUpdatesEnabled(False)
        try:
            self._apply_session_appearance(loaded.application)
            self.project, self.radiation_settings = loaded.store, loaded.radiation
            self.structure_preparation = StructurePreparation(self.project, self)
            self.structure_preparation.busy_changed.connect(self.structure_loading_dialog.set_busy)
            self.structure_preparation.failed.connect(self._structure_preparation_failed)
            new_central = self._build_central_widget(install=False)
        except Exception:
            if self.structure_preparation is not previous['structure_preparation']:
                self.structure_preparation.close()
                self.structure_preparation.deleteLater()
            if new_central is not None:
                new_central.deleteLater()
            for name, value in previous.items():
                setattr(self, name, value)
            self._apply_session_appearance(appearance)
            raise
        finally:
            self.setUpdatesEnabled(True)
        old_viewer = previous['pages'][VIEWER]
        old_viewer.close_indexing()
        old_viewer.close_phase_calculations()
        for dialog in tuple(old_viewer._peak_table_dialogs.values()):
            dialog.close()
        substrate = getattr(old_viewer, '_substrate_dialog', None)
        if substrate is not None:
            substrate.close()
        previous['project'].unsubscribe(self._project_event)
        previous['project'].unsubscribe(previous['project_panel']._store_event)
        previous['structure_preparation'].close()
        previous['structure_preparation'].deleteLater()
        self.takeCentralWidget()
        self.setCentralWidget(new_central)
        old_central.deleteLater()
        self.project.subscribe(self._project_event)
        application = loaded.application
        workspace = application.get('workspace', loaded.scope if loaded.scope in self.WORKSPACES else previous_workspace)
        self.sections.setCurrentIndex(self.WORKSPACES.index(workspace))
        self._drawer_visible = application.get('drawer_visible', self._drawer_visible)
        self.project_panel.setVisible(self._drawer_visible)
        self.project_rail.setVisible(not self._drawer_visible)
        self._project_panel_width = application.get('panel_width', previous['_project_panel_width'])
        self.project_splitter.setSizes([self._project_panel_width, max(1, self.width() - self._project_panel_width)])
        for i, width in enumerate(application.get('tree_widths', appearance['tree_widths'])):
            self.project_panel.tree.setColumnWidth(i, width)
        if application:
            self.resize(*application.get('window_size', appearance['window_size']))
            self.showMaximized() if application.get('maximized', False) else self.showNormal()
        self.retranslate()
        self.structure_preparation.prepare_project()
        self.structure_loading_dialog.set_busy(self.structure_preparation.busy)

    def _apply_session_appearance(self, application):
        if not application:
            return
        set_language(application.get('language', get_language()), persist=False)
        self.theme_controller.set_mode(application.get('theme', self.theme_controller.mode), persist=False)
        renderer = application.get('renderer', self.plot_renderer_controller.mode)
        self.plot_renderer_controller.set_mode('matplotlib' if renderer == 'pyqtgraph' and not pyqtgraph_available() else renderer, persist=False)
        apply_session_colours(application.get('atom_palette', palette()), application.get('atom_colours', custom_colours()))
