"""Точка входа единого приложения XRD Combine."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("TkAgg")

try:
    from .atom_styles import (
        custom_colours,
        export_custom_colours,
        import_custom_colours,
        palette as atom_palette,
        reset_custom_colours,
        set_palette as set_atom_palette,
    )
    from .experimental_pole import PoleFigureApp as ExperimentalPoleApp
    from .calculated_pattern import CalculatedPatternPage
    from .controls import CollapsibleSection
    from .i18n import (
        LANGUAGES,
        apply_language,
        filedialog,
        get_language,
        load_language,
        localised,
        messagebox,
        set_language,
    )
    from .reflection_table import ReflectionTablePage
    from .models.project import CELL_PHASE, CIF, SCAN, POLES, STRUCTURES, VIEWER
    from .models.radiation import RadiationSettings
    from .models.scan import Scan1D, clone_scan
    from .project_panel import ProjectPanel
    from .project_store import ProjectStore
    from .structure_view import StructurePage
    from .substrate_compare import SubstrateComparisonPage
    from .theoretical_pole import _build_gui as build_theoretical_pole
    from .twotheta import TwoThetaPage
    from .ui_tk.radiation import RadiationSelector
    from .version import APP_VERSION
except ImportError:
    from atom_styles import (
        custom_colours,
        export_custom_colours,
        import_custom_colours,
        palette as atom_palette,
        reset_custom_colours,
        set_palette as set_atom_palette,
    )
    from experimental_pole import PoleFigureApp as ExperimentalPoleApp
    from calculated_pattern import CalculatedPatternPage
    from controls import CollapsibleSection
    from i18n import (
        LANGUAGES,
        apply_language,
        filedialog,
        get_language,
        load_language,
        localised,
        messagebox,
        set_language,
    )
    from reflection_table import ReflectionTablePage
    from models.project import CELL_PHASE, CIF, SCAN, POLES, STRUCTURES, VIEWER
    from models.radiation import RadiationSettings
    from models.scan import Scan1D, clone_scan
    from project_panel import ProjectPanel
    from project_store import ProjectStore
    from structure_view import StructurePage
    from substrate_compare import SubstrateComparisonPage
    from theoretical_pole import _build_gui as build_theoretical_pole
    from twotheta import TwoThetaPage
    from ui_tk.radiation import RadiationSelector
    from version import APP_VERSION


class XRDCombine(tk.Tk):
    def __init__(self, initial_paths: list[str] | None = None) -> None:
        super().__init__()
        self._closing = False
        set_language(load_language())
        self.title("XRD Combine")
        self.geometry("1500x900")
        self.minsize(1080, 700)
        self.project = ProjectStore()
        self.active_documents: dict[str, str | None] = {
            VIEWER: None,
            STRUCTURES: None,
            POLES: None,
        }
        self.comparison_window: tk.Toplevel | None = None
        self.about_window: tk.Toplevel | None = None
        self.comparison_page: SubstrateComparisonPage | None = None
        self._drawer_visible = True
        self.project.subscribe(self._project_event)

        self._configure_style()
        self._build_menu()
        self._build_pages()
        self.after_idle(lambda: apply_language(self, get_language()))

        self.protocol("WM_DELETE_WINDOW", self.close)
        for index, path in enumerate(initial_paths or []):
            self.after(120 + 50 * index, lambda value=path: self.open_path(value))

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        available = style.theme_names()
        for preferred in ("vista", "clam"):
            if preferred in available:
                style.theme_use(preferred)
                break
        style.configure("TNotebook.Tab", padding=(14, 7))

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Открыть измерения…", command=self._menu_open)
        file_menu.add_command(label="Добавить CIF…", command=self._menu_cif)
        file_menu.add_command(
            label="Новая фаза по ячейке…",
            command=lambda: self.project_panel.new_cell_phase(),
        )
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.close)
        menu.add_cascade(label="Файл", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(
            label="Опорные пики для коррекции…",
            command=self.open_reference_peaks,
        )
        language_menu = tk.Menu(edit_menu, tearoff=False)
        self.language_var = tk.StringVar(value=get_language())
        for code, label in LANGUAGES.items():
            language_menu.add_radiobutton(
                label=label,
                value=code,
                variable=self.language_var,
                command=self.change_language,
            )
        edit_menu.add_cascade(label="Язык", menu=language_menu)
        atom_menu = tk.Menu(edit_menu, tearoff=False)
        self.atom_palette_var = tk.StringVar(value=atom_palette())
        for value, label in (
            ("jmol", "Jmol"),
            ("cpk", "CPK"),
            ("molcas_gv", "MOLCAS GV"),
        ):
            atom_menu.add_radiobutton(
                label=label,
                value=value,
                variable=self.atom_palette_var,
                command=self.change_atom_palette,
            )
        atom_menu.add_separator()
        atom_menu.add_command(
            label="Импортировать свои цвета…",
            command=self.import_atom_colours,
        )
        atom_menu.add_command(
            label="Экспортировать свои цвета…",
            command=self.export_atom_colours,
        )
        atom_menu.add_command(
            label="Сбросить свои цвета",
            command=self.reset_atom_colours,
        )
        edit_menu.add_cascade(label="Цвета атомов", menu=atom_menu)
        menu.add_cascade(label="Правка", menu=edit_menu)

        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_command(label="Просмотр", command=lambda: self.sections.select(self.viewer_tab))
        view_menu.add_command(
            label="Структуры", command=lambda: self.sections.select(self.structures_tab)
        )
        view_menu.add_command(
            label="Полюсные фигуры", command=lambda: self.sections.select(self.pole_tab)
        )
        menu.add_cascade(label="Раздел", menu=view_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="О программе", command=self.about)
        menu.add_cascade(label="Справка", menu=help_menu)

        self.configure(menu=menu)

    def _build_pages(self) -> None:
        shell = ttk.Frame(self)
        shell.pack(fill="both", expand=True)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        body = ttk.Frame(shell)
        body.grid(row=0, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        self.sections = ttk.Notebook(body)
        self.sections.grid(row=0, column=1, sticky="nsew")

        self.viewer_tab = ttk.Frame(self.sections)
        self.structures_tab = ttk.Frame(self.sections)
        self.pole_tab = ttk.Frame(self.sections)
        self.sections.add(self.viewer_tab, text="Просмотр")
        self.sections.add(self.structures_tab, text="Структуры")
        self.sections.add(self.pole_tab, text="Полюсные фигуры")
        self.sections.bind("<<NotebookTabChanged>>", self._workspace_changed)

        self.project_panel = ProjectPanel(
            body,
            self.project,
            self.current_workspace,
            on_activate=self.activate_project_document,
            on_collapse=self.toggle_project_panel,
        )
        self.project_panel.grid(row=0, column=0, sticky="nsew")
        self.project_rail = ttk.Frame(body, width=42, padding=(4, 8))
        self.project_rail.grid_propagate(False)
        self.project_rail.rowconfigure(2, weight=1)
        ttk.Button(
            self.project_rail,
            text=">",
            width=3,
            command=self.toggle_project_panel,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            self.project_rail,
            text="+",
            width=3,
            command=self.project_panel.open_files,
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.viewer_tab.rowconfigure(0, weight=1)
        self.viewer_tab.columnconfigure(0, weight=1)
        self.twotheta = TwoThetaPage(
            self.viewer_tab,
            on_open_theoretical=self.open_theoretical_cif,
            on_open_reflection_table=self.open_reflection_cif,
            on_commit_scan=self.commit_corrected_scan,
            on_open_structure=self.open_structure_cif,
            on_import_paths=lambda paths: self.import_paths(paths, VIEWER),
        )
        self.twotheta.grid(row=0, column=0, sticky="nsew")
        compare_bar = ttk.Frame(self.viewer_tab, padding=(8, 4, 8, 8))
        compare_bar.grid(row=1, column=0, sticky="ew")
        compare_bar.columnconfigure(0, weight=1)
        self.compare_button = ttk.Button(
            compare_bar,
            text="Отправить в сравнение (0)",
            command=self.open_comparison,
            state="disabled",
        )
        self.compare_button.grid(row=0, column=0, sticky="e")

        self.structures_tab.rowconfigure(1, weight=1)
        self.structures_tab.columnconfigure(0, weight=1)
        self.radiation_settings = RadiationSettings()
        self.radiation_selector = RadiationSelector(
            self.structures_tab,
            self.radiation_settings,
            self._radiation_changed,
        )
        self.radiation_selector.grid(row=0, column=0, sticky="ew")
        self.structure_modes = ttk.Notebook(self.structures_tab)
        self.structure_modes.grid(row=1, column=0, sticky="nsew")
        self.structure_tab = ttk.Frame(self.structure_modes)
        self.calculated_pattern_tab = ttk.Frame(self.structure_modes)
        self.reflection_tab = ttk.Frame(self.structure_modes)
        self.structure_modes.add(self.structure_tab, text="Просмотр структуры")
        self.structure_modes.add(self.calculated_pattern_tab, text="Расчётный график")
        self.structure_modes.add(self.reflection_tab, text="Таблица отражений")

        self.structure_tab.rowconfigure(0, weight=1)
        self.structure_tab.columnconfigure(0, weight=1)
        self.structure_view = StructurePage(
            self.structure_tab,
            on_open_cif=lambda path: self.import_paths([path], STRUCTURES),
        )
        self.structure_view.grid(row=0, column=0, sticky="nsew")

        self.calculated_pattern_tab.rowconfigure(0, weight=1)
        self.calculated_pattern_tab.columnconfigure(0, weight=1)
        self.calculated_pattern = CalculatedPatternPage(
            self.calculated_pattern_tab,
            self.radiation_settings.lines,
        )
        self.calculated_pattern.grid(row=0, column=0, sticky="nsew")

        self.reflection_tab.rowconfigure(0, weight=1)
        self.reflection_tab.columnconfigure(0, weight=1)
        self.reflection_table = ReflectionTablePage(
            self.reflection_tab,
            on_open_cif=lambda path: self.import_paths([path], STRUCTURES),
            radiations_provider=self.radiation_settings.lines,
        )
        self.reflection_table.grid(row=0, column=0, sticky="nsew")

        self.pole_modes = ttk.Notebook(self.pole_tab)
        self.pole_modes.pack(fill="both", expand=True)
        self.experimental_tab = ttk.Frame(self.pole_modes)
        self.theoretical_tab = ttk.Frame(self.pole_modes)
        self.pole_modes.add(self.experimental_tab, text="Экспериментальная RAW")
        self.pole_modes.add(self.theoretical_tab, text="Расчётная")

        self.experimental_tab.rowconfigure(0, weight=1)
        self.experimental_tab.columnconfigure(0, weight=1)
        experimental_host = ttk.Frame(self.experimental_tab)
        experimental_host.grid(row=0, column=0, sticky="nsew")
        self.experimental = ExperimentalPoleApp(
            experimental_host,
            on_open_raw=lambda path: self.import_paths([path], POLES),
        )

        self.theoretical_tab.rowconfigure(1, weight=1)
        self.theoretical_tab.columnconfigure(0, weight=1)
        self.pole_radiation_selector = RadiationSelector(
            self.theoretical_tab,
            self.radiation_settings,
            self._radiation_changed,
        )
        self.pole_radiation_selector.grid(row=0, column=0, sticky="ew")
        theoretical_host = ttk.Frame(self.theoretical_tab)
        theoretical_host.grid(row=1, column=0, sticky="nsew")
        self.theoretical = build_theoretical_pole(
            None,
            parent=theoretical_host,
            auto_prompt=False,
            on_open_cif=lambda path: self.import_paths([path], POLES),
            radiations_provider=self.radiation_settings.lines,
        )

        self.sections.select(self.viewer_tab)
        self._update_compare_button()

    def _menu_open(self) -> None:
        self.sections.select(self.viewer_tab)
        self.project_panel.open_files()

    def _menu_cif(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title=localised("Add CIF", "Ajouter un CIF", "Добавить CIF"),
            filetypes=(("CIF", "*.cif"), ("All files", "*.*")),
        )
        if path:
            self.import_paths([path], self.current_workspace())

    def current_workspace(self) -> str:
        selected = self.sections.select() if hasattr(self, "sections") else ""
        if selected == str(getattr(self, "structures_tab", "")):
            return STRUCTURES
        if selected == str(getattr(self, "pole_tab", "")):
            return POLES
        return VIEWER

    def _select_workspace(self, workspace: str) -> None:
        tab = {
            VIEWER: self.viewer_tab,
            STRUCTURES: self.structures_tab,
            POLES: self.pole_tab,
        }.get(workspace, self.viewer_tab)
        self.sections.select(tab)

    def toggle_project_panel(self) -> None:
        self._drawer_visible = not self._drawer_visible
        if self._drawer_visible:
            self.project_panel.grid()
            self.project_rail.grid_remove()
        else:
            self.project_panel.grid_remove()
            self.project_rail.grid(row=0, column=0, sticky="ns")

    def _radiation_changed(self, _radiations=None) -> None:
        for name in ("radiation_selector", "pole_radiation_selector"):
            selector = getattr(self, name, None)
            if selector is not None:
                selector.value.set(self.radiation_settings.label())
        if hasattr(self, "calculated_pattern"):
            self.calculated_pattern.radiation_changed()
        if hasattr(self, "reflection_table"):
            self.reflection_table.radiation_changed()
        if hasattr(self, "theoretical"):
            self.theoretical.radiation_changed()

    def open_reference_peaks(self) -> None:
        if hasattr(self, "twotheta"):
            self.twotheta.open_peak_database()

    def _refresh_atom_styles(self) -> None:
        if hasattr(self, "structure_view"):
            self.structure_view.refresh_atom_styles()
        if hasattr(self, "theoretical"):
            self.theoretical.refresh_atom_styles()

    def change_atom_palette(self) -> None:
        try:
            set_atom_palette(self.atom_palette_var.get())
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                localised("Atom colours", "Couleurs des atomes", "Цвета атомов"),
                str(exc),
                parent=self,
            )
            self.atom_palette_var.set(atom_palette())
            return
        self._refresh_atom_styles()

    def import_atom_colours(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title=localised(
                "Import custom atom colours",
                "Importer des couleurs d’atomes personnalisées",
                "Импортировать свои цвета атомов",
            ),
            filetypes=(("Text files", "*.txt *.dat *.ini"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            colours = import_custom_colours(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                localised("Invalid colour file", "Fichier de couleurs incorrect", "Некорректный файл цветов"),
                str(exc),
                parent=self,
            )
            return
        self._refresh_atom_styles()
        messagebox.showinfo(
            localised("Atom colours", "Couleurs des atomes", "Цвета атомов"),
            localised(
                f"Imported colours: {len(colours)}.",
                f"Couleurs importées : {len(colours)}.",
                f"Импортировано цветов: {len(colours)}.",
            ),
            parent=self,
        )

    def export_atom_colours(self) -> None:
        if not custom_colours():
            messagebox.showinfo(
                localised("Atom colours", "Couleurs des atomes", "Цвета атомов"),
                localised(
                    "No custom colours are defined.",
                    "Aucune couleur personnalisée n’est définie.",
                    "Пользовательские цвета не заданы.",
                ),
                parent=self,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title=localised(
                "Export custom atom colours",
                "Exporter les couleurs d’atomes personnalisées",
                "Экспортировать свои цвета атомов",
            ),
            defaultextension=".txt",
            initialfile="xrd_combine_atom_colours.txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if path:
            try:
                export_custom_colours(path)
            except OSError as exc:
                messagebox.showerror(
                    localised("Save error", "Erreur d’enregistrement", "Ошибка сохранения"),
                    str(exc),
                    parent=self,
                )

    def reset_atom_colours(self) -> None:
        try:
            reset_custom_colours()
        except OSError as exc:
            messagebox.showerror(
                localised("Atom colours", "Couleurs des atomes", "Цвета атомов"),
                str(exc),
                parent=self,
            )
            return
        self._refresh_atom_styles()

    def _workspace_changed(self, _event=None) -> None:
        if hasattr(self, "project_panel"):
            self.project_panel.refresh()
        self._update_compare_button()

    def import_paths(self, paths, workspace: str | None = None):
        workspace = workspace or self.current_workspace()
        self._select_workspace(workspace)
        return self.project_panel.import_paths(paths)

    def _project_event(self, event, document, workspace) -> None:
        if document is None or not hasattr(self, "twotheta"):
            return
        if event == "assigned" and workspace == VIEWER:
            if document.kind == SCAN:
                self.twotheta.add_scan(document.payload, uid=document.uid)
            elif document.kind in {CIF, CELL_PHASE}:
                self.twotheta.add_phase_document(document.payload, uid=document.uid)
        elif event == "unassigned" and workspace == VIEWER:
            self.twotheta.remove_uid(document.uid)
        elif event == "unassigned" and workspace in {STRUCTURES, POLES}:
            was_active = self.active_documents.get(workspace) == document.uid
            if workspace == STRUCTURES and (
                self.structure_view.cif_document is document.payload
                or self.calculated_pattern.cif_document is document.payload
                or self.reflection_table.cif_document is document.payload
            ):
                self.structure_view.clear_document()
                self.calculated_pattern.clear_document()
                self.reflection_table.clear_document()
            elif (
                workspace == POLES
                and document.kind in {CIF, CELL_PHASE}
                and self.theoretical.cif_document is document.payload
            ):
                self.theoretical.clear_document()
            elif (
                workspace == POLES
                and document.kind == POLE_DATA
                and self.experimental.raw_path == document.source
            ):
                self.experimental.clear_data()
            if was_active:
                self.active_documents[workspace] = None
                remaining = self.project.assigned_documents(workspace)
                if remaining:
                    self.activate_project_document(remaining[-1].uid, workspace)
        elif event == "removed":
            self.twotheta.remove_uid(document.uid)
            for active_workspace, uid in tuple(self.active_documents.items()):
                if uid == document.uid:
                    self.active_documents[active_workspace] = None
        elif event == "replaced" and document.kind == SCAN:
            item = self.twotheta.items.get(document.uid)
            if item is not None:
                item.scan = document.payload
                item.name = document.name
                item.source = document.source
                item.reset_transform()
                self.twotheta.tree.item(document.uid, text=document.name)
                self.twotheta._update_buttons()
                self.twotheta._draw(preserve_view=True)
        elif event == "replaced" and document.kind == CELL_PHASE:
            self.twotheta.remove_uid(document.uid, redraw=False)
            if self.project.is_assigned(document.uid, VIEWER):
                self.twotheta.add_phase_document(document.payload, uid=document.uid)
            for target in (STRUCTURES, POLES):
                if self.active_documents.get(target) == document.uid:
                    self.activate_project_document(document.uid, target)
        if hasattr(self, "compare_button"):
            self.after_idle(self._update_compare_button)

    def activate_project_document(self, uid: str, workspace: str) -> None:
        document = self.project.documents.get(uid)
        if document is None:
            return
        self.active_documents[workspace] = uid
        if workspace == VIEWER:
            if uid in self.twotheta.items:
                self.twotheta.tree.selection_set(uid)
                self.twotheta.tree.see(uid)
                self.twotheta._update_buttons()
            return
        if workspace == STRUCTURES and document.kind in {CIF, CELL_PHASE}:
            self.structure_view.load_document(document.payload)
            self.calculated_pattern.load_document(document.payload)
            self.reflection_table.load_document(document.payload)
            return
        if workspace == POLES and document.kind in {CIF, CELL_PHASE}:
            self.pole_modes.select(self.theoretical_tab)
            self.theoretical.load_document(document.payload)
        elif workspace == POLES and document.kind == POLE_DATA:
            self.pole_modes.select(self.experimental_tab)
            self.experimental.select_raw(document.source, document.payload)

    @staticmethod
    def _comparison_compatible(scan: Scan1D) -> bool:
        key = scan.axis_name.lower().replace(" ", "").replace("_", "")
        key = key.replace("θ", "theta")
        return key in {"2theta", "twotheta"}

    def _comparison_documents(self):
        return [
            document
            for document in self.project.assigned_documents(VIEWER, kind=SCAN)
            if self._comparison_compatible(document.payload)
        ]

    def _update_compare_button(self) -> None:
        if not hasattr(self, "compare_button"):
            return
        count = len(self._comparison_documents())
        self.compare_button.configure(
            text=localised(
                f"Send to comparison ({count})",
                f"Envoyer vers la comparaison ({count})",
                f"Отправить в сравнение ({count})",
            ),
            state="normal" if count else "disabled",
        )

    def open_comparison(self, _scan: Scan1D | None = None) -> None:
        if self.comparison_window is not None and self.comparison_window.winfo_exists():
            self.comparison_window.lift()
            self.comparison_window.focus_force()
            return
        documents = self._comparison_documents()
        if not documents:
            messagebox.showinfo(
                localised("Comparison", "Comparaison", "Сравнение"),
                localised(
                    "Assign at least one 2θ measurement to Viewer first.",
                    "Affectez d’abord au moins une mesure 2θ à la section Visualisation.",
                    "Сначала подключите к разделу «Просмотр» хотя бы одно измерение 2θ.",
                ),
                parent=self,
            )
            return

        window = tk.Toplevel(self)
        window.title(localised("Substrate comparison", "Comparaison avec des substrats", "Сравнение с подложками"))
        window.geometry("1450x850")
        window.minsize(980, 650)
        window.transient(self)
        page = SubstrateComparisonPage(
            window,
            on_send_viewer=self.send_to_viewer,
            on_send_correction=self.send_to_correction,
        )
        page.pack(fill="both", expand=True)
        for document in documents:
            page.add_scan(document.payload)
        self.comparison_window = window
        self.comparison_page = page
        window.protocol("WM_DELETE_WINDOW", self._close_comparison)
        window.grab_set()
        window.focus_set()

    def _close_comparison(self) -> None:
        window = self.comparison_window
        if window is None:
            return
        try:
            if self.comparison_page is not None:
                self.comparison_page.close_figures()
        finally:
            try:
                window.grab_release()
            except tk.TclError:
                pass
            try:
                window.destroy()
            except tk.TclError:
                pass
            self.comparison_window = None
            self.comparison_page = None

    def change_language(self) -> None:
        language = self.language_var.get()
        set_language(language, persist=True)
        apply_language(self, language)
        self.project_panel.localize_content()
        self.twotheta.localize_content()
        self.twotheta._draw(preserve_view=True)
        self.radiation_selector.localize_content()
        self.pole_radiation_selector.localize_content()
        self.calculated_pattern.localize_content()
        self.reflection_table.localize_content()
        self.structure_view.localize_content()
        if self.comparison_page is not None:
            self.comparison_page.localize_content()
            self.comparison_window.title(
                localised(
                    "Substrate comparison",
                    "Comparaison avec des substrats",
                    "Сравнение с подложками",
                )
            )
        self._update_compare_button()
        if self.experimental.scans:
            self.experimental.draw_pole_figure()
        if self.theoretical.crystal is not None:
            self.theoretical.redraw()

    def open_path(self, path: str) -> None:
        self.import_paths([path], VIEWER)

    def add_corrected_scan(self, path: str) -> None:
        """После сохранения коррекции добавить производный скан в 2θ."""

        try:
            documents = self.project.add_path(path)
            for document in documents:
                self.project.assign(document.uid, VIEWER, True)
                self.activate_project_document(document.uid, VIEWER)
            self._select_workspace(VIEWER)
            self.twotheta.status.set(
                localised(
                    f"Corrected file {Path(path).name} was added to the plot.",
                    f"Le fichier corrigé {Path(path).name} a été ajouté au graphique.",
                    f"Исправленный файл {Path(path).name} добавлен на график.",
                )
            )
        except Exception as exc:
            messagebox.showwarning(
                localised("File saved", "Fichier enregistré", "Файл сохранён"),
                localised(
                    f"The file was saved but could not be added to the plot:\n{exc}",
                    f"Le fichier a été enregistré, mais n’a pas pu être ajouté au graphique :\n{exc}",
                    f"Файл сохранён, но не добавлен на график:\n{exc}",
                ),
                parent=self,
            )

    def send_to_correction(self, scan: Scan1D) -> None:
        uid = next(
            (
                document.uid
                for document in self.project.documents.values()
                if document.kind == SCAN and document.payload is scan
            ),
            None,
        )
        if uid is None:
            document = self.project.add_scan(clone_scan(scan), derived=True)
            self.project.assign(document.uid, VIEWER, True)
            uid = document.uid
        self._select_workspace(VIEWER)
        self.activate_project_document(uid, VIEWER)
        self.twotheta.open_processing_for(uid)

    def send_to_viewer(self, scan: Scan1D) -> None:
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
        self.project.assign(document.uid, VIEWER, True)
        self._select_workspace(VIEWER)
        self.activate_project_document(document.uid, VIEWER)

    def commit_corrected_scan(self, source_uid: str, scan: Scan1D, mode: str) -> None:
        if mode == "replace" and source_uid in self.project.documents:
            document = self.project.replace_scan(source_uid, scan)
            action = localised("replaced", "remplacé", "заменён")
        else:
            document = self.project.add_scan(
                scan,
                derived=True,
                parent_uid=source_uid,
            )
            self.project.assign(document.uid, VIEWER, True)
            action = localised("added", "ajouté", "добавлен")
        self.activate_project_document(document.uid, VIEWER)
        self.twotheta.status.set(
            localised(
                f"Shifted dataset {document.name} was {action} in the project.",
                f"Le jeu de données décalé {document.name} a été {action} dans le projet.",
                f"Сдвинутый набор {document.name} {action} в проекте.",
            )
        )

    def open_theoretical_cif(self, path: str) -> None:
        if path in self.project.documents:
            self.project.assign(path, POLES, True)
            self._select_workspace(POLES)
            self.pole_modes.select(self.theoretical_tab)
            self.activate_project_document(path, POLES)
            return
        documents = self.import_paths([path], POLES)
        self.pole_modes.select(self.theoretical_tab)
        if documents:
            self.activate_project_document(documents[-1].uid, POLES)

    def open_reflection_cif(self, path: str) -> None:
        if path in self.project.documents:
            self.project.assign(path, STRUCTURES, True)
            self._select_workspace(STRUCTURES)
            self.structure_modes.select(self.reflection_tab)
            self.activate_project_document(path, STRUCTURES)
            return
        documents = self.import_paths([path], STRUCTURES)
        self.structure_modes.select(self.reflection_tab)
        if documents:
            self.activate_project_document(documents[-1].uid, STRUCTURES)

    def open_structure_cif(self, path: str) -> None:
        if path in self.project.documents:
            self.project.assign(path, STRUCTURES, True)
            self._select_workspace(STRUCTURES)
            self.structure_modes.select(self.structure_tab)
            self.activate_project_document(path, STRUCTURES)
            return
        documents = self.import_paths([path], STRUCTURES)
        self.structure_modes.select(self.structure_tab)
        if documents:
            self.activate_project_document(documents[-1].uid, STRUCTURES)

    def about(self) -> None:
        if self.about_window is not None and self.about_window.winfo_exists():
            self.about_window.lift()
            self.about_window.focus_set()
            return
        window = tk.Toplevel(self)
        self.about_window = window
        window.title(localised("About XRD Combine", "À propos de XRD Combine", "О программе XRD Combine"))
        window.transient(self)
        window.resizable(True, True)
        window.minsize(600, 390)
        body = ttk.Frame(window, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="XRD Combine", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(
            body,
            text=localised(
                "Three connected workspaces for viewing measurements, working with CIF "
                "structures and building pole figures. Files are loaded once into a common "
                "project pool; every workspace keeps its own display settings. Substrate "
                "comparison opens in one modal window, and corrected datasets can be added "
                "or replaced without overwriting the source file.\n\n"
                f"Version {APP_VERSION}",
                "Trois espaces reliés pour visualiser les mesures, travailler avec les "
                "structures CIF et construire des figures de pôles. Les fichiers sont chargés "
                "une fois dans un pool commun ; chaque espace conserve ses propres réglages "
                "d’affichage. La comparaison aux substrats s’ouvre dans une seule fenêtre "
                "modale et les données corrigées peuvent être ajoutées ou remplacées sans "
                "écraser le fichier source.\n\n"
                f"Version {APP_VERSION}",
                "Три связанных раздела для просмотра измерений, работы со структурами CIF "
                "и построения полюсных фигур. Файлы один раз загружаются в общий пул проекта; "
                "каждый раздел сохраняет собственные настройки отображения. Сравнение с "
                "подложками открывается в одном модальном окне, а исправленный набор можно "
                "добавить или заменить без перезаписи исходного файла.\n\n"
                f"Версия {APP_VERSION}",
            ),
            wraplength=660,
            justify="left",
        ).pack(fill="x", pady=(9, 10))
        ttk.Label(
            body,
            text="Mikhail Miruschenko\nmiruschenko98@gmail.com",
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        details = CollapsibleSection(
            body,
            text="Научные данные и сторонние компоненты",
            padding=8,
            expanded=False,
        )
        details.pack(fill="x")
        ttk.Label(
            details,
            text=localised(
                "Atomic scattering factors: Waasmaier–Kirfel coefficients from DABAX.\n"
                "Atom colours and radii: compact data derived from mendeleev.\n"
                "Space-group operations: compact data derived from spglib.\n"
                "Runtime: Python/Tcl/Tk, NumPy, SciPy, Matplotlib and Pillow.",
                "Facteurs de diffusion atomique : coefficients de Waasmaier–Kirfel issus de DABAX.\n"
                "Couleurs et rayons atomiques : données compactes dérivées de mendeleev.\n"
                "Opérations des groupes d’espace : données compactes dérivées de spglib.\n"
                "Exécution : Python/Tcl/Tk, NumPy, SciPy, Matplotlib et Pillow.",
                "Атомные факторы рассеяния: коэффициенты Ваасмайера—Кирфеля из DABAX.\n"
                "Цвета и радиусы атомов: компактные данные на основе mendeleev.\n"
                "Операции пространственных групп: компактные данные на основе spglib.\n"
                "Среда выполнения: Python/Tcl/Tk, NumPy, SciPy, Matplotlib и Pillow.",
            ),
            wraplength=640,
            justify="left",
        ).pack(fill="x")
        ttk.Button(
            details,
            text="Полные уведомления…",
            command=lambda: self._show_third_party_notices(window),
        ).pack(anchor="w", pady=(8, 0))
        def closed() -> None:
            self.about_window = None
            window.destroy()

        ttk.Button(body, text="Закрыть", command=closed).pack(anchor="e", pady=(12, 0))
        window.protocol("WM_DELETE_WINDOW", closed)
        apply_language(window, get_language())
        window.focus_set()

    @staticmethod
    def _third_party_notices_path() -> Path | None:
        candidates = (
            Path(__file__).resolve().parent.parent / "THIRD_PARTY_NOTICES.txt",
            Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
            / "THIRD_PARTY_NOTICES.txt",
            Path(sys.executable).resolve().parent / "THIRD_PARTY_NOTICES.txt",
        )
        return next((path for path in candidates if path.exists()), None)

    def _show_third_party_notices(self, parent) -> None:
        path = self._third_party_notices_path()
        if path is None:
            messagebox.showerror(
                localised("Notices not found", "Mentions introuvables", "Уведомления не найдены"),
                "THIRD_PARTY_NOTICES.txt",
                parent=parent,
            )
            return
        window = tk.Toplevel(parent)
        window.title("THIRD_PARTY_NOTICES.txt")
        window.geometry("820x650")
        window.transient(parent)
        frame = ttk.Frame(window, padding=8)
        frame.pack(fill="both", expand=True)
        text_widget = tk.Text(frame, wrap="word", padx=8, pady=8)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scroll.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text_widget.insert("1.0", path.read_text(encoding="utf-8"))
        text_widget.configure(state="disabled")

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True

        try:
            self.twotheta._cancel_peak_fit()
        except Exception:
            pass

        for owner, attribute in (
            (self.twotheta, "_redraw_job"),
            (self.twotheta, "_processing_redraw_job"),
            (self.experimental, "redraw_job"),
        ):
            try:
                job = getattr(owner, attribute, None)
                if job:
                    owner.after_cancel(job)
                    setattr(owner, attribute, None)
            except Exception:
                pass

        if self.comparison_window is not None:
            self._close_comparison()

        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass

        try:
            self.quit()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass
        raise SystemExit(0)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    app = XRDCombine(arguments)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
