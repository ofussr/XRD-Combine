"""Общая сворачиваемая панель документов проекта."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

try:
    from .cell_phase import CellPhaseDocument, create_cell_phase_document
    from .i18n import apply_language, filedialog, localised, messagebox, translate_text
    from .project_store import CELL_PHASE, CIF, POLE_DATA, SCAN, ProjectDocument, ProjectStore
    from .space_groups import BY_HALL_NUMBER, SETTINGS, setting_from_user_text
except ImportError:  # pragma: no cover
    from cell_phase import CellPhaseDocument, create_cell_phase_document
    from i18n import apply_language, filedialog, localised, messagebox, translate_text
    from project_store import CELL_PHASE, CIF, POLE_DATA, SCAN, ProjectDocument, ProjectStore
    from space_groups import BY_HALL_NUMBER, SETTINGS, setting_from_user_text


SUPPORTED_SUFFIXES = {
    ".xrdml", ".xml", ".raw", ".xy", ".txt", ".dat", ".csv", ".cif"
}


class CellPhaseDialog(tk.Toplevel):
    """Create or edit an atom-free phase from cell and space-group data."""

    def __init__(self, parent, initial: CellPhaseDocument | None = None):
        super().__init__(parent)
        self.title(localised("Cell-parameter phase", "Phase définie par la maille", "Фаза по параметрам ячейки"))
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self.result: CellPhaseDocument | None = None
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        self.name_var = tk.StringVar(value=initial.name if initial else "")
        ttk.Label(body, text="Название:").grid(row=0, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.name_var, width=46).grid(
            row=0, column=1, columnspan=5, sticky="ew", pady=(0, 8)
        )
        ttk.Label(body, text="Пространственная группа").grid(row=1, column=0, sticky="w")
        self.group_var = tk.StringVar()
        self.group_combo = ttk.Combobox(
            body,
            textvariable=self.group_var,
            values=tuple(item.label for item in SETTINGS),
            state="normal",
            width=39,
        )
        self.group_combo.grid(row=1, column=1, columnspan=5, sticky="ew", pady=(0, 8))
        ttk.Label(
            body,
            text="Введите номер, символ или Hall N.",
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(0, 8))
        setting = initial.setting if initial else BY_HALL_NUMBER[1]
        self.group_var.set(setting.label)
        values = initial.cell if initial else (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        self.cell_vars = [tk.StringVar(value=f"{value:g}") for value in values]
        for column, (label, variable) in enumerate(
            zip(("a, Å", "b, Å", "c, Å", "α, °", "β, °", "γ, °"), self.cell_vars)
        ):
            ttk.Label(body, text=label).grid(row=3, column=column, sticky="w")
            ttk.Entry(body, textvariable=variable, width=9).grid(
                row=4, column=column, sticky="ew", padx=(0, 5)
            )
        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="Отмена", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Применить", command=self._accept).pack(
            side="right", padx=(0, 6)
        )
        apply_language(self)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _accept(self) -> None:
        try:
            cell = tuple(
                float(variable.get().strip().replace(",", "."))
                for variable in self.cell_vars
            )
            self.result = create_cell_phase_document(
                self.name_var.get(),
                setting_from_user_text(self.group_var.get()),
                cell,
            )
        except ValueError as exc:
            messagebox.showerror(
                localised("Invalid phase", "Phase incorrecte", "Некорректная фаза"),
                str(exc),
                parent=self,
            )
            return
        self.destroy()


class ProjectPanel(ttk.Frame):
    """Один пул файлов; галочка означает привязку к текущему разделу."""

    GROUPS = (
        (SCAN, "Измерения"),
        (CIF, "Структуры CIF"),
        (CELL_PHASE, "Фазы по параметрам ячейки"),
        (POLE_DATA, "Данные полюсных фигур"),
    )

    def __init__(
        self,
        parent,
        store: ProjectStore,
        get_workspace,
        on_activate=None,
        on_collapse=None,
        width: int = 330,
    ) -> None:
        super().__init__(parent, padding=8, width=width)
        self.grid_propagate(False)
        self.store = store
        self.get_workspace = get_workspace
        self.on_activate = on_activate
        self.on_collapse = on_collapse
        self.status = tk.StringVar(value="")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Данные проекта").grid(row=0, column=0, sticky="w")
        self.collapse_button = ttk.Button(
            header,
            text="<",
            width=3,
            command=self.on_collapse,
        )
        self.collapse_button.grid(row=0, column=1, sticky="e")

        actions = ttk.Frame(self)
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        actions.columnconfigure((0, 1), weight=1)
        ttk.Button(actions, text="Открыть файлы…", command=self.open_files).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(actions, text="Открыть папку…", command=self.open_folder).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )
        ttk.Button(
            actions,
            text="Новая фаза по ячейке…",
            command=self.new_cell_phase,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        tree_frame = ttk.Frame(self)
        tree_frame.grid(row=2, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("use", "type"),
            show="tree headings",
            selectmode="browse",
        )
        self.tree.heading("#0", text="Название")
        self.tree.heading("use", text="В разделе")
        self.tree.heading("type", text="Тип")
        self.tree.column("#0", width=165, stretch=True)
        self.tree.column("use", width=74, anchor="center", stretch=False)
        self.tree.column("type", width=70, anchor="center", stretch=False)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<ButtonRelease-1>", self._tree_click)
        self.tree.bind("<Double-1>", self._tree_double_click)
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)

        bottom = ttk.Frame(self)
        bottom.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.status, wraplength=245).grid(
            row=0, column=0, sticky="ew"
        )
        self.remove_button = ttk.Button(
            bottom, text="Удалить из проекта", command=self.remove_selected
        )
        self.remove_button.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.edit_button = ttk.Button(
            bottom,
            text="Редактировать параметры…",
            command=self.edit_selected_cell_phase,
        )
        self.edit_button.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        self.store.subscribe(self._store_event)
        self.refresh()
        apply_language(self)

    def current_workspace(self) -> str:
        return self.get_workspace()

    def refresh(self) -> None:
        selected = self.selected_uid()
        for item in self.tree.get_children(""):
            self.tree.delete(item)
        workspace = self.current_workspace()
        for kind, title in self.GROUPS:
            group_id = f"group:{kind}"
            self.tree.insert("", "end", iid=group_id, text=translate_text(title), open=True)
            for document in self.store.documents.values():
                if document.kind != kind:
                    continue
                compatible = self.store.compatible(kind, workspace)
                marker = (
                    "✓"
                    if self.store.is_assigned(document.uid, workspace)
                    else "□" if compatible else "—"
                )
                self.tree.insert(
                    group_id,
                    "end",
                    iid=document.uid,
                    text=document.name,
                    values=(marker, self._type_label(document)),
                )
        if selected in self.store.documents:
            self.tree.selection_set(selected)
            self.tree.see(selected)
        self._update_status()

    @staticmethod
    def _type_label(document: ProjectDocument) -> str:
        if document.kind == SCAN:
            suffix = document.source.suffix.lower().lstrip(".")
            return suffix.upper() or "XY"
        if document.kind == CIF:
            return "CIF"
        if document.kind == CELL_PHASE:
            return translate_text("Ячейка")
        return translate_text("Полюсная")

    def selected_uid(self) -> str | None:
        selection = self.tree.selection()
        if not selection or selection[0].startswith("group:"):
            return None
        return selection[0]

    def _tree_click(self, event) -> None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        if self.tree.identify_column(event.x) != "#1":
            return
        uid = self.tree.identify_row(event.y)
        if uid in self.store.documents:
            self.toggle(uid)

    def _tree_double_click(self, event) -> None:
        uid = self.tree.identify_row(event.y)
        if uid not in self.store.documents:
            return
        if not self.store.is_assigned(uid, self.current_workspace()):
            self.toggle(uid)
        self._activate(uid)

    def _selection_changed(self, _event=None) -> None:
        self._update_status()

    def _activate(self, uid: str) -> None:
        if self.on_activate is not None:
            self.on_activate(uid, self.current_workspace())

    def toggle(self, uid: str) -> None:
        workspace = self.current_workspace()
        document = self.store.documents[uid]
        if not self.store.compatible(document.kind, workspace):
            self.status.set(
                localised(
                    "This data type is not used in the current section.",
                    "Ce type de données n’est pas utilisé dans la section actuelle.",
                    "Этот тип данных не используется в текущем разделе.",
                )
            )
            return
        enabled = not self.store.is_assigned(uid, workspace)
        self.store.assign(uid, workspace, enabled)
        if enabled:
            self._activate(uid)

    def _store_event(self, _event, _document, _workspace) -> None:
        if self.winfo_exists():
            self.after_idle(self.refresh)

    def _update_status(self) -> None:
        uid = self.selected_uid()
        document = self.store.documents.get(uid) if uid else None
        if document is None:
            count = len(self.store.documents)
            self.status.set(
                localised(
                    f"Project objects: {count}",
                    f"Objets du projet : {count}",
                    f"Объектов в проекте: {count}",
                )
            )
            self.remove_button.configure(state="disabled")
            self.edit_button.configure(state="disabled")
            return
        if document.kind == CELL_PHASE:
            phase = document.payload
            self.status.set(
                f"{phase.setting.number}: {phase.setting.international_short}"
                + (f" [{phase.setting.choice}]" if phase.setting.choice else "")
            )
        else:
            self.status.set(str(document.source))
        self.remove_button.configure(state="normal")
        self.edit_button.configure(
            state="normal" if document.kind == CELL_PHASE else "disabled"
        )

    def new_cell_phase(self) -> None:
        dialog = CellPhaseDialog(self)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        document = self.store.add_cell_phase(dialog.result)
        workspace = self.current_workspace()
        self.store.assign(document.uid, workspace, True)
        self.refresh()
        self.tree.selection_set(document.uid)
        self._activate(document.uid)

    def edit_selected_cell_phase(self) -> None:
        uid = self.selected_uid()
        document = self.store.documents.get(uid) if uid else None
        if document is None or document.kind != CELL_PHASE:
            return
        dialog = CellPhaseDialog(self, document.payload)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self.store.replace_cell_phase(document.uid, dialog.result)
        self.refresh()
        self.tree.selection_set(document.uid)
        self._activate(document.uid)

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Открыть файлы…",
            filetypes=(
                ("Поддерживаемые файлы", "*.xrdml *.xml *.raw *.xy *.txt *.dat *.csv *.cif"),
                ("CIF", "*.cif"),
                ("Все файлы", "*.*"),
            ),
        )
        if paths:
            self.import_paths(paths)

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Открыть папку…")
        if not folder:
            return
        paths = [
            path for path in sorted(Path(folder).iterdir())
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not paths:
            messagebox.showinfo(
                localised("Empty folder", "Dossier vide", "Папка пуста"),
                localised(
                    "No supported files were found.",
                    "Aucun fichier pris en charge n’a été trouvé.",
                    "Поддерживаемые файлы не найдены.",
                ),
                parent=self,
            )
            return
        self.import_paths(paths)

    def import_paths(self, paths) -> list[ProjectDocument]:
        workspace = self.current_workspace()
        loaded: list[ProjectDocument] = []
        last_assigned_uid: str | None = None
        errors: list[str] = []
        for value in paths:
            try:
                documents = self.store.add_path(value)
                for document in documents:
                    if self.store.compatible(document.kind, workspace):
                        self.store.assign(document.uid, workspace, True)
                        self._activate(document.uid)
                        last_assigned_uid = document.uid
                    loaded.append(document)
            except Exception as exc:
                errors.append(f"{Path(value).name}: {exc}")
        selected_uid = last_assigned_uid or (loaded[-1].uid if loaded else None)
        self.refresh()
        if selected_uid:
            self.tree.selection_set(selected_uid)
            self.tree.see(selected_uid)
        if errors:
            messagebox.showwarning(
                localised(
                    "Some files could not be read",
                    "Certains fichiers n’ont pas pu être lus",
                    "Не все файлы прочитаны",
                ),
                "\n\n".join(errors[:12]),
                parent=self,
            )
        return loaded

    def remove_selected(self) -> None:
        uid = self.selected_uid()
        if uid is None:
            return
        document = self.store.documents[uid]
        if not messagebox.askyesno(
            localised("Remove from project", "Supprimer du projet", "Удалить из проекта"),
            localised(
                f"Remove {document.name} from every section?",
                f"Supprimer {document.name} de toutes les sections ?",
                f"Удалить {document.name} из всех разделов?",
            ),
            parent=self,
        ):
            return
        self.store.remove(uid)

    def localize_content(self) -> None:
        apply_language(self)
        self.refresh()
