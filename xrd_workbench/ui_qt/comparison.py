"""Native PySide6 interface for the existing substrate-comparison workflow."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..bruker_raw import read_bruker_raw
from ..io import read_scan_file
from ..localization import localised, tr
from ..models.scan import Scan1D, clone_scan
from ..models.substrate_compare import (
    ComparisonAssembly,
    ComparisonItem,
    ComparisonWorkspace,
)
from ..services.substrate_compare import prepare_comparison_plot


SUPPORTED_SUFFIXES = {".xrdml", ".xml", ".raw", ".xy", ".txt", ".dat", ".csv"}


def _base_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


class ComparisonCard(QPushButton):
    """One draggable card backed by a toolkit-independent comparison item."""

    def __init__(self, owner: "ComparisonPage", item: ComparisonItem) -> None:
        super().__init__(item.name, owner.workspace)
        self.owner = owner
        self.item = item
        self.setFixedSize(230, 34)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_requested)

    def _context_requested(self, position) -> None:
        self.owner.show_context_menu(self.item, self.mapToGlobal(position))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.owner.card_pressed(self, event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.owner.card_moved(self, event.globalPosition().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.owner.card_released(self)
        super().mouseReleaseEvent(event)


class ComparisonPlotDialog(QDialog):
    """Detailed Matplotlib view of one active comparison assembly."""

    def __init__(
        self,
        page: "ComparisonPage",
        assembly: ComparisonAssembly,
    ) -> None:
        super().__init__(page)
        self.page = page
        self.assembly = assembly
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(assembly.label)
        self.resize(900, 650)

        root = QVBoxLayout(self)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure.subplots_adjust(left=0.1, right=0.97, top=0.96, bottom=0.12)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas, 1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        root.addWidget(self.toolbar)

        controls = QHBoxLayout()
        self.x_min = QLineEdit()
        self.x_max = QLineEdit()
        self.y_min = QLineEdit()
        self.y_max = QLineEdit()
        for label, edit in (
            ("X min", self.x_min),
            ("X max", self.x_max),
            ("Y min", self.y_min),
            ("Y max", self.y_max),
        ):
            controls.addWidget(QLabel(label))
            edit.setMaximumWidth(90)
            controls.addWidget(edit)
        self.apply_button = QPushButton(tr("text.apply"))
        self.mode_button = QPushButton(tr("text.y_mode"))
        self.save_close_button = QPushButton(tr("text.save_close"))
        self.apply_button.clicked.connect(self.apply_limits)
        self.mode_button.clicked.connect(self.toggle_y_mode)
        self.save_close_button.clicked.connect(self.save_and_close)
        controls.addWidget(self.apply_button)
        controls.addWidget(self.mode_button)
        controls.addStretch(1)
        controls.addWidget(self.save_close_button)
        root.addLayout(controls)

        self.y_mode = str(assembly.view.get("ymode", "log"))
        self.page.plot_assembly(self.axis, assembly, assembly.view)
        current_x = self.axis.get_xlim()
        current_y = self.axis.get_ylim()
        self.x_min.setText(f"{current_x[0]:.6g}")
        self.x_max.setText(f"{current_x[1]:.6g}")
        self.y_min.setText(f"{current_y[0]:.6g}")
        self.y_max.setText(f"{current_y[1]:.6g}")
        self.canvas.draw_idle()

    def apply_limits(self) -> None:
        try:
            values = [
                float(edit.text().strip().replace(",", "."))
                for edit in (self.x_min, self.x_max, self.y_min, self.y_max)
            ]
            if values[0] >= values[1] or values[2] >= values[3]:
                raise ValueError
            if self.y_mode == "log" and values[2] <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(
                self,
                localised(
                    "Invalid limits",
                    "Limites incorrectes",
                    "Некорректные границы",
                ),
                localised(
                    "All four limits must be numeric and increasing; logarithmic Y must be positive.",
                    "Les quatre limites doivent être numériques et croissantes ; Y logarithmique doit être positif.",
                    "Все четыре границы должны быть числами и возрастать; нижняя граница логарифмической Y должна быть положительной.",
                ),
            )
            return
        self.axis.set_xlim(values[0], values[1])
        self.axis.set_ylim(values[2], values[3])
        self.canvas.draw_idle()

    def toggle_y_mode(self) -> None:
        modes = ("linear", "log", "exp", "square")
        self.y_mode = modes[(modes.index(self.y_mode) + 1) % len(modes)]
        self.page.plot_assembly(
            self.axis,
            self.assembly,
            {
                **self.assembly.view,
                "ymode": self.y_mode,
                "xlim": self.axis.get_xlim(),
                "ylim": self.axis.get_ylim(),
            },
        )
        self.canvas.draw_idle()

    def save_and_close(self) -> None:
        self.assembly.view = {
            "xlim": list(self.axis.get_xlim()),
            "ylim": list(self.axis.get_ylim()),
            "ymode": self.y_mode,
        }
        self.page.group_views[self.assembly.group_id] = dict(self.assembly.view)
        for item in self.assembly.items:
            if item.kind == "substrate":
                self.page.presets[item.name] = {
                    "xlim": list(self.axis.get_xlim()),
                    "ylim": list(self.axis.get_ylim()),
                }
        self.page.write_default_presets()
        self.close()
        self.page.refresh_gallery()

    def closeEvent(self, event) -> None:
        self.figure.clear()
        self.page.viewer_dialogs.discard(self)
        super().closeEvent(event)


class ComparisonPage(QWidget):
    """Qt presentation of the established draggable comparison workspace."""

    def __init__(
        self,
        parent=None,
        *,
        on_send_viewer: Callable[[Scan1D], None] | None = None,
        on_send_correction: Callable[[Scan1D], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_send_viewer = on_send_viewer
        self.on_send_correction = on_send_correction
        self.comparison_state = ComparisonWorkspace()
        self.items = self.comparison_state.items
        self.groups = self.comparison_state.groups
        self.group_active = self.comparison_state.group_active
        self.group_views = self.comparison_state.group_views
        self.card_by_item: dict[ComparisonItem, ComparisonCard] = {}
        self.next_substrate_y = 46
        self.next_file_y = 46
        self.substrate_x = 20
        self.file_x = 400
        self.drag_card: ComparisonCard | None = None
        self.drag_offset = None
        self.drag_start = None
        self.drag_moved = False
        self.reset_counter = 0
        self.gallery_figures: list[Figure] = []
        self.viewer_dialogs: set[ComparisonPlotDialog] = set()

        base = _base_directory()
        self.substrate_directory = base / "substrates"
        self.preset_path = base / "substrate_view_presets.json"
        try:
            loaded = json.loads(self.preset_path.read_text(encoding="utf-8"))
            self.presets = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError, TypeError):
            self.presets: dict[str, dict[str, list[float]]] = {}

        self._build_ui()
        self._load_default_substrates()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        root.addWidget(self.tabs)

        self.workspace_tab = QWidget()
        workspace_layout = QHBoxLayout(self.workspace_tab)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        controls = QFrame()
        controls.setFrameShape(QFrame.Shape.StyledPanel)
        controls.setFixedWidth(205)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(7, 7, 7, 7)
        self.substrate_folder_button = QPushButton()
        self.open_files_button = QPushButton()
        self.open_folder_button = QPushButton()
        self.reset_button = QPushButton()
        self.substrate_folder_button.clicked.connect(self.load_substrate_folder)
        self.open_files_button.clicked.connect(self.load_files)
        self.open_folder_button.clicked.connect(self.load_folder)
        self.reset_button.clicked.connect(self.reset_layout)
        controls_layout.addWidget(self.substrate_folder_button)
        substrate_separator = QFrame()
        substrate_separator.setFrameShape(QFrame.Shape.HLine)
        substrate_separator.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(substrate_separator)
        controls_layout.addWidget(self.open_files_button)
        controls_layout.addWidget(self.open_folder_button)
        measurement_separator = QFrame()
        measurement_separator.setFrameShape(QFrame.Shape.HLine)
        measurement_separator.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(measurement_separator)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addStretch(1)
        workspace_layout.addWidget(controls)

        self.workspace = QFrame()
        self.workspace.setMinimumSize(760, 540)
        self.workspace.setStyleSheet("QFrame { background: #f0f0f0; }")
        self.substrate_heading = QLabel(self.workspace)
        self.file_heading = QLabel(self.workspace)
        self.substrate_heading.setStyleSheet("font-weight: 600; background: transparent;")
        self.file_heading.setStyleSheet("font-weight: 600; background: transparent;")
        self.substrate_heading.move(self.substrate_x, 12)
        self.file_heading.move(self.file_x, 12)
        self.substrate_heading.adjustSize()
        self.file_heading.adjustSize()
        workspace_layout.addWidget(self.workspace, 1)
        self.tabs.addTab(self.workspace_tab, "")

        self.gallery_tab = QWidget()
        gallery_root = QVBoxLayout(self.gallery_tab)
        gallery_root.setContentsMargins(0, 0, 0, 0)
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.gallery_scroll.setWidget(self.gallery_widget)
        gallery_root.addWidget(self.gallery_scroll, 1)
        gallery_controls = QHBoxLayout()
        self.load_presets_button = QPushButton()
        self.save_presets_button = QPushButton()
        self.update_gallery_button = QPushButton()
        self.load_presets_button.clicked.connect(self.load_presets)
        self.save_presets_button.clicked.connect(self.save_presets)
        self.update_gallery_button.clicked.connect(self.refresh_gallery)
        gallery_controls.addWidget(self.load_presets_button)
        gallery_controls.addWidget(self.save_presets_button)
        gallery_controls.addStretch(1)
        gallery_controls.addWidget(self.update_gallery_button)
        gallery_root.addLayout(gallery_controls)
        self.tabs.addTab(self.gallery_tab, "")
        self.retranslate()

    def retranslate(self) -> None:
        self.tabs.setTabText(0, tr("text.workspace"))
        self.tabs.setTabText(1, tr("text.gallery_results"))
        self.substrate_folder_button.setText(tr("text.load_substrate_folder"))
        self.open_files_button.setText(tr("text.open_files"))
        self.open_folder_button.setText(tr("text.open_folder"))
        self.reset_button.setText(tr("text.reset_layout"))
        self.substrate_heading.setText(tr("text.substrates_right_click_to_clone"))
        self.file_heading.setText(tr("text.measurements_drag_to_connect"))
        self.substrate_heading.adjustSize()
        self.file_heading.adjustSize()
        self.load_presets_button.setText(tr("text.load_presets"))
        self.save_presets_button.setText(tr("text.save_presets"))
        self.update_gallery_button.setText(tr("text.update_gallery"))
        if self.tabs.currentIndex() == 1:
            self.refresh_gallery()

    def _load_default_substrates(self) -> None:
        if not self.substrate_directory.exists():
            try:
                self.substrate_directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                return
        paths = [
            path
            for path in sorted(self.substrate_directory.iterdir())
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if paths:
            self._load_paths(paths, "substrate")

    def add_scan(self, scan: Scan1D, kind: str = "file") -> ComparisonItem:
        name = self.comparison_state.unique_name(scan.name)
        x = self.substrate_x if kind == "substrate" else self.file_x
        y = self.next_substrate_y if kind == "substrate" else self.next_file_y
        item = self.comparison_state.add_item(
            kind=kind,
            name=name,
            scan=scan,
            x=x,
            y=y,
            width=230,
            height=34,
        )
        card = ComparisonCard(self, item)
        self.card_by_item[item] = card
        card.move(x, y)
        card.show()
        self._sync_item_widget(item)
        if kind == "substrate":
            self.next_substrate_y += 38
        else:
            self.next_file_y += 38
        return item

    def _load_paths(self, paths, kind: str) -> None:
        errors: list[str] = []
        for value in paths:
            path = Path(value)
            try:
                scans = read_scan_file(path, raw_reader=read_bruker_raw)
                for scan in scans:
                    self.add_scan(scan, kind)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        if errors:
            QMessageBox.warning(
                self,
                localised(
                    "Some files were skipped",
                    "Certains fichiers ont été ignorés",
                    "Часть файлов пропущена",
                ),
                "\n\n".join(errors[:12]),
            )

    @staticmethod
    def _supported_filter() -> str:
        return (
            "XRD (*.xrdml *.xml *.raw *.xy *.txt *.dat *.csv);;"
            "XRDML (*.xrdml *.xml);;Bruker RAW (*.raw);;"
            "XY (*.xy *.txt *.dat *.csv);;All files (*)"
        )

    def load_substrate_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            tr("text.substrate_folder"),
        )
        if not folder:
            return
        self._clear_kind("substrate")
        self._load_paths(self._folder_paths(folder), "substrate")

    def load_files(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(
            self,
            tr("text.open_diffraction_patterns"),
            "",
            self._supported_filter(),
        )
        if paths:
            self._load_paths(paths, "file")

    def load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            localised(
                "Open measurement folder",
                "Ouvrir le dossier des mesures",
                "Открыть папку с измерениями",
            ),
        )
        if folder:
            self._load_paths(self._folder_paths(folder), "file")

    @staticmethod
    def _folder_paths(folder: str | Path) -> list[Path]:
        return [
            path
            for path in sorted(Path(folder).iterdir())
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]

    def _clear_kind(self, kind: str) -> None:
        removed = self.comparison_state.remove_kind(kind)
        for item in removed:
            card = self.card_by_item.pop(item, None)
            if card is not None:
                card.deleteLater()
        if kind == "substrate":
            self.next_substrate_y = 46
        else:
            self.next_file_y = 46
        self._sync_widgets()

    def _sync_item_widget(self, item: ComparisonItem) -> None:
        card = self.card_by_item.get(item)
        if card is None:
            return
        card.move(item.x, item.y)
        if item.active:
            card.setStyleSheet(
                "QPushButton { background: #2ecc71; border: 2px solid #2b7a45; }"
            )
        else:
            card.setStyleSheet(
                "QPushButton { background: #d0d0d0; border: 1px solid #9a9a9a; }"
            )

    def _sync_widgets(self, items=None) -> None:
        for item in self.items if items is None else items:
            self._sync_item_widget(item)

    def card_pressed(self, card: ComparisonCard, global_position) -> None:
        card.raise_()
        self.drag_card = card
        local = card.mapFromGlobal(global_position)
        self.drag_offset = local
        self.drag_start = card.pos()
        self.drag_moved = False

    def card_moved(self, card: ComparisonCard, global_position) -> None:
        if self.drag_card is not card or self.drag_offset is None:
            return
        position = self.workspace.mapFromGlobal(global_position) - self.drag_offset
        if self.drag_start is not None:
            if (position - self.drag_start).manhattanLength() > 3:
                self.drag_moved = True
        card.move(position)
        self.comparison_state.move_item(card.item, position.x(), position.y())

    def card_released(self, card: ComparisonCard) -> None:
        if self.drag_card is not card:
            return
        source = card.item
        if not self.drag_moved:
            if source.group is not None:
                self.set_group_active(
                    source.group,
                    not self.group_active[source.group],
                )
            else:
                self.set_item_active(source, not source.active)
            self.drag_card = None
            return

        hits = [
            (item, self.comparison_state.overlap_area(source, item))
            for item in self.items
            if item is not source
            and self.comparison_state.overlap_area(source, item) > 0
        ]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        if hits:
            target = hits[0][0]
            if source.group is None and target.group is None:
                group_id = self.create_group(source, target)
            elif source.group is not None and target.group is None:
                group_id = source.group
                self.add_to_group(group_id, target)
            elif source.group is None and target.group is not None:
                group_id = target.group
                self.add_to_group(group_id, source)
            else:
                assert source.group is not None and target.group is not None
                group_id = self.merge_groups(source.group, target.group)
            self.layout_group(group_id, target.x, min(source.y, target.y))
        elif source.group is not None:
            overlaps_group = any(
                self.comparison_state.overlap_area(source, other) > 0
                for other in self.groups[source.group]
                if other is not source
            )
            if not overlaps_group:
                self.remove_from_group(source)
        self.drag_card = None

    def set_item_active(self, item: ComparisonItem, active: bool) -> None:
        self.comparison_state.set_item_active(item, active)
        self._sync_item_widget(item)

    def set_group_active(self, group_id: int, active: bool) -> None:
        self.comparison_state.set_group_active(group_id, active)
        self._sync_widgets(self.groups[group_id])

    def layout_group(self, group_id: int, x: int, y: int) -> None:
        self.comparison_state.layout_group(group_id, x, y)
        self._sync_widgets(self.groups[group_id])

    def create_group(self, *members: ComparisonItem) -> int:
        group_id = self.comparison_state.create_group(*members)
        self._sync_widgets(self.groups[group_id])
        return group_id

    def add_to_group(self, group_id: int, item: ComparisonItem) -> None:
        self.comparison_state.add_to_group(group_id, item)
        self._sync_widgets(self.groups[group_id])

    def merge_groups(self, first: int, second: int) -> int:
        group_id = self.comparison_state.merge_groups(first, second)
        self._sync_widgets(self.groups[group_id])
        return group_id

    def remove_from_group(self, item: ComparisonItem) -> None:
        old_group = item.group
        affected = set(self.groups.get(old_group, ()))
        self.comparison_state.remove_from_group(item)
        self._sync_widgets(affected)

    def reset_layout(self) -> None:
        self.comparison_state.reset_layout()
        self._sync_widgets()
        self.reset_counter += 1
        if self.reset_counter == 10:
            self._show_author()
            self.reset_counter = 0

    def _show_author(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("About Author")
        dialog.setFixedSize(350, 200)
        dialog.setStyleSheet("QDialog { background: #202020; } QLabel { color: white; }")
        layout = QVBoxLayout(dialog)
        created = QLabel("CREATED BY")
        created.setAlignment(Qt.AlignmentFlag.AlignCenter)
        created.setStyleSheet("color: #808080; font-weight: 600;")
        name = QLabel("Mikhail Mirushchenko")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("font-size: 16px; font-weight: 600;")
        email = QLabel("miruschenko98@gmail.com")
        email.setAlignment(Qt.AlignmentFlag.AlignCenter)
        email.setStyleSheet("color: #a0a0a0;")
        close_button = QPushButton(tr("text.close"))
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(created)
        layout.addWidget(name)
        layout.addWidget(email)
        layout.addWidget(close_button)
        dialog.exec()

    def show_context_menu(self, item: ComparisonItem, global_position) -> None:
        menu = QMenu(self)
        send_viewer = QAction(tr("text.send_to_viewer"), menu)
        send_viewer.setEnabled(self.on_send_viewer is not None)
        send_viewer.triggered.connect(lambda: self.send_to_viewer(item))
        menu.addAction(send_viewer)
        send_correction = QAction(tr("text.send_to_correction"), menu)
        send_correction.setEnabled(self.on_send_correction is not None)
        send_correction.triggered.connect(lambda: self.send_to_correction(item))
        menu.addAction(send_correction)
        menu.addSeparator()
        duplicate = QAction(
            localised(
                f'Duplicate “{item.name}”',
                f'Dupliquer « {item.name} »',
                f'Создать копию «{item.name}»',
            ),
            menu,
        )
        duplicate.setEnabled(item.kind == "substrate")
        duplicate.triggered.connect(lambda: self.duplicate_substrate(item))
        menu.addAction(duplicate)
        menu.exec(global_position)

    def send_to_viewer(self, item: ComparisonItem) -> None:
        if self.on_send_viewer is not None:
            self.on_send_viewer(item.scan)

    def send_to_correction(self, item: ComparisonItem) -> None:
        if self.on_send_correction is not None:
            self.on_send_correction(item.scan)

    def duplicate_substrate(self, item: ComparisonItem) -> None:
        if item.kind == "substrate":
            self.add_scan(clone_scan(item.scan), "substrate")

    def _tab_changed(self, index: int) -> None:
        if index == 1:
            self.refresh_gallery()

    def _clear_gallery(self) -> None:
        for figure in self.gallery_figures:
            figure.clear()
        self.gallery_figures.clear()
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()

    def refresh_gallery(self) -> None:
        self._clear_gallery()
        assemblies = self.comparison_state.active_assemblies()
        if not assemblies:
            label = QLabel(tr("text.no_active_groups_exist_in_the_workspace"))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gallery_layout.addWidget(label, 0, 0)
            return
        for index, assembly in enumerate(assemblies):
            row, column = divmod(index, 2)
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame_layout = QVBoxLayout(frame)
            title = QLabel(assembly.label[:65])
            title.setStyleSheet("font-weight: 600;")
            frame_layout.addWidget(title)
            figure = Figure(figsize=(4, 2.2), dpi=80)
            figure.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.18)
            axis = figure.add_subplot(111)
            self.plot_assembly(axis, assembly, thumbnail=True)
            canvas = FigureCanvasQTAgg(figure)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.mpl_connect(
                "button_press_event",
                lambda _event, value=assembly: self.open_viewer(value),
            )
            frame_layout.addWidget(canvas, 1)
            buttons = QHBoxLayout()
            buttons.addStretch(1)
            copy_button = QPushButton(tr("text.copy"))
            open_button = QPushButton(tr("text.open"))
            copy_button.clicked.connect(
                lambda _checked=False, value=assembly: self.copy_to_clipboard(value)
            )
            open_button.clicked.connect(
                lambda _checked=False, value=assembly: self.open_viewer(value)
            )
            buttons.addWidget(copy_button)
            buttons.addWidget(open_button)
            frame_layout.addLayout(buttons)
            self.gallery_figures.append(figure)
            self.gallery_layout.addWidget(frame, row, column)
        self.gallery_layout.setColumnStretch(0, 1)
        self.gallery_layout.setColumnStretch(1, 1)

    def plot_assembly(
        self,
        axis,
        assembly: ComparisonAssembly,
        view: dict[str, object] | None = None,
        thumbnail: bool = False,
    ) -> None:
        plot_data = prepare_comparison_plot(
            assembly,
            self.presets,
            view,
            thumbnail=thumbnail,
        )
        axis.clear()
        for series in plot_data.series:
            axis.plot(
                series.x,
                series.y,
                label=series.label,
                color=series.colour,
                linewidth=series.linewidth,
                zorder=series.zorder,
            )
        if plot_data.uses_automatic_limits:
            axis.autoscale(True, "both")
        else:
            axis.set_xlim(plot_data.x_limits)
            axis.set_ylim(plot_data.y_limits)
        axis.set_yscale(plot_data.y_scale)
        axis.set_xlabel(plot_data.x_label or tr("text.scan_coordinate"))
        axis.set_ylabel(tr("qt.viewer_intensity"))
        axis.grid(True, alpha=0.2)
        if not thumbnail:
            axis.legend(loc="upper left")

    def open_viewer(self, assembly: ComparisonAssembly) -> None:
        dialog = ComparisonPlotDialog(self, assembly)
        self.viewer_dialogs.add(dialog)
        dialog.show()

    def copy_to_clipboard(self, assembly: ComparisonAssembly) -> None:
        buffer = io.BytesIO()
        figure = Figure(figsize=(7, 4), dpi=150)
        axis = figure.add_subplot(111)
        figure.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.15)
        self.plot_assembly(axis, assembly, assembly.view)
        figure.savefig(buffer, format="png", dpi=150)
        figure.clear()
        image = QImage.fromData(buffer.getvalue(), "PNG")
        if image.isNull():
            QMessageBox.critical(
                self,
                localised(
                    "Clipboard error",
                    "Erreur du presse-papiers",
                    "Ошибка буфера обмена",
                ),
                localised(
                    "The plot image could not be created.",
                    "L’image du graphique n’a pas pu être créée.",
                    "Не удалось создать изображение графика.",
                ),
            )
            return
        QApplication.clipboard().setImage(image)

    def load_presets(self) -> None:
        path_text, _selected = QFileDialog.getOpenFileName(
            self, "", "", "JSON (*.json);;All files (*)"
        )
        if not path_text:
            return
        try:
            loaded = json.loads(Path(path_text).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("JSON root must be an object")
            self.presets.update(loaded)
            self.refresh_gallery()
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                localised(
                    "Preset error",
                    "Erreur de réglages",
                    "Ошибка настроек",
                ),
                str(exc),
            )

    def save_presets(self) -> None:
        path_text, _selected = QFileDialog.getSaveFileName(
            self, "", "substrate_view_presets.json", "JSON (*.json);;All files (*)"
        )
        if not path_text:
            return
        path = Path(path_text)
        if not path.suffix:
            path = path.with_suffix(".json")
        try:
            path.write_text(
                json.dumps(self.presets, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(
                self,
                localised(
                    "Preset error",
                    "Erreur de réglages",
                    "Ошибка настроек",
                ),
                str(exc),
            )

    def write_default_presets(self) -> None:
        try:
            self.preset_path.write_text(
                json.dumps(self.presets, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def close_figures(self) -> None:
        self._clear_gallery()
        for dialog in tuple(self.viewer_dialogs):
            dialog.close()
        self.viewer_dialogs.clear()


class ComparisonDialog(QDialog):
    """Single application-modal comparison window."""

    def __init__(
        self,
        scans: list[Scan1D],
        parent=None,
        *,
        on_send_viewer: Callable[[Scan1D], None] | None = None,
        on_send_correction: Callable[[Scan1D], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("text.substrate_comparison"))
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumSize(980, 650)
        self.resize(1450, 850)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.page = ComparisonPage(
            self,
            on_send_viewer=on_send_viewer,
            on_send_correction=on_send_correction,
        )
        layout.addWidget(self.page)
        for scan in scans:
            self.page.add_scan(scan)

    def closeEvent(self, event) -> None:
        self.page.close_figures()
        super().closeEvent(event)


__all__ = ["ComparisonCard", "ComparisonDialog", "ComparisonPage", "ComparisonPlotDialog"]
