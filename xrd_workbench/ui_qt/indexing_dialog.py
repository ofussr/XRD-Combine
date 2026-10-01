"""Opt-in indexing UI; workers receive only immutable scientific snapshots."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import html
import json
import math
from threading import Event

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from ..localization import localised, tr
from ..indexing.models import IndexingCancelled, IndexingError, IndexingRequest
from ..indexing.service import METHODS, run_indexing
from ..indexing.indexing_visser import VisserSettings
from ..indexing.indexing_boultif_louer import BoultifLouerSettings, CRYSTAL_SYSTEMS


def system_label(system):
    names = {
        "cubic": ("Cubic", "Cubique", "Кубическая"),
        "tetragonal": ("Tetragonal", "Tétragonale", "Тетрагональная"),
        "hexagonal": ("Hexagonal", "Hexagonale", "Гексагональная"),
        "orthorhombic": ("Orthorhombic", "Orthorhombique", "Ромбическая"),
        "monoclinic": ("Monoclinic", "Monoclinique", "Моноклинная"),
        "triclinic": ("Triclinic", "Triclinique", "Триклинная"),
    }
    return localised(*names[system])


def error_text(error):
    messages = {
        "too_few_lines": ("Too few selected peaks (required: {required}, selected: {actual}).",
            "Trop peu de pics sélectionnés (requis : {required}, sélectionnés : {actual}).",
            "Недостаточно выбранных пиков: нужно {required}, выбрано {actual}."),
        "invalid_position": ("Non-physical position for peak {peak}.",
            "Position non physique pour le pic {peak}.", "Нефизическое положение пика {peak}."),
        "invalid_uncertainty": ("The position error for peak {peak} must be positive and finite.",
            "L’erreur de position du pic {peak} doit être positive et finie.",
            "Погрешность положения пика {peak} должна быть положительным конечным числом."),
        "invalid_wavelength": ("Enter a positive wavelength.", "Entrez une longueur d’onde positive.",
            "Введите положительную длину волны."),
        "invalid_settings": ("Invalid search settings: {reason}", "Paramètres invalides : {reason}",
            "Некорректные настройки поиска: {reason}"),
        "reduction_failed": ("Cell reduction failed.", "Échec de la réduction de maille.",
            "Не удалось привести ячейку."),
    }
    if isinstance(error, IndexingError):
        return localised(*messages.get(error.code, (error.code,) * 3)).format(**error.details)
    return localised("Calculation failed: ", "Échec du calcul : ", "Ошибка расчёта: ") + str(error)


def _numeric_spin(value, low, high, decimals=6):
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(low, high)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


def _show_number(value):
    if value is None or math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    return f"{value:.7g}"


class IndexingDialog(QDialog):
    create_phase_requested = Signal(object, object)

    def __init__(self, peaks, *, measurement_name, input_kind, wavelength,
                 source_signature, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setModal(False)
        self.resize(1120, 760)
        self.peaks = tuple(peaks)
        self.measurement_name = measurement_name
        self.input_kind = input_kind
        self.source_signature = source_signature
        self.result = None
        self._stale = False
        self._closed = False
        self._future = None
        self._cancel = Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xrd-indexing")
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)
        self._build_ui(wavelength)
        self.retranslate()
        self._method_changed()

    def _build_ui(self, wavelength):
        root = QVBoxLayout(self)
        self.note = QLabel()
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        self.settings_box = QGroupBox()
        grid = QGridLayout(self.settings_box)
        form = QFormLayout()
        self.method_combo = QComboBox()
        for key, module in METHODS.items():
            self.method_combo.addItem(module.METHOD_NAME, key)
        self.method_combo.setCurrentIndex(self.method_combo.findData("boultif_louer"))
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.method_label = QLabel()
        form.addRow(self.method_label, self.method_combo)
        self.wavelength = _numeric_spin(wavelength, 0.000001, 100, 8)
        form.addRow("λ, Å", self.wavelength)
        self.error_label = QLabel()
        self.position_error = _numeric_spin(0.03 if self.input_kind == "two_theta" else 0.0001,
                                            0.000001, 10, 6)
        form.addRow(self.error_label, self.position_error)
        self.search_label = QLabel()
        self.search_count = QSpinBox()
        self.search_count.setRange(8, 80)
        self.search_count.setValue(20)
        form.addRow(self.search_label, self.search_count)
        grid.addLayout(form, 0, 0)
        bounds = QFormLayout()
        self.min_length = _numeric_spin(1.5, 0.01, 199, 3)
        self.max_length = _numeric_spin(25, 0.02, 200, 3)
        self.max_volume = _numeric_spin(2500, 0.01, 1e7, 2)
        self.min_label, self.max_label, self.volume_label = QLabel(), QLabel(), QLabel()
        bounds.addRow(self.min_label, self.min_length)
        bounds.addRow(self.max_label, self.max_length)
        bounds.addRow(self.volume_label, self.max_volume)
        self.hkl_label = QLabel()
        self.hkl_limit = QSpinBox()
        self.hkl_limit.setRange(2, 12)
        self.hkl_limit.setValue(5)
        bounds.addRow(self.hkl_label, self.hkl_limit)
        grid.addLayout(bounds, 0, 1)
        self.systems_box = QGroupBox()
        systems_layout = QHBoxLayout(self.systems_box)
        self.system_checks = {}
        for system in CRYSTAL_SYSTEMS:
            check = QCheckBox()
            check.setChecked(True)
            systems_layout.addWidget(check)
            self.system_checks[system] = check
        grid.addWidget(self.systems_box, 1, 0, 1, 2)
        self.spurious_label = QLabel()
        self.spurious = QSpinBox()
        self.spurious.setRange(0, 5)
        self.spurious.setValue(1)
        self.zero_check = QCheckBox()
        self.zero_range = _numeric_spin(0.2, 0.001, 2, 3)
        extras = QHBoxLayout()
        extras.addWidget(self.spurious_label)
        extras.addWidget(self.spurious)
        extras.addWidget(self.zero_check)
        extras.addWidget(self.zero_range)
        extras.addWidget(QLabel("± °2θ"))
        extras.addStretch(1)
        grid.addLayout(extras, 2, 0, 1, 2)
        root.addWidget(self.settings_box)
        self.reference = QLabel()
        self.reference.setWordWrap(True)
        self.reference.setOpenExternalLinks(True)
        root.addWidget(self.reference)
        self.tabs = QTabWidget()
        self.peak_table = QTableWidget(len(self.peaks), 5)
        for row, peak in enumerate(self.peaks):
            values = ("", str(peak.peak_id), f"{peak.position:.8g}",
                      "" if peak.uncertainty is None else str(peak.uncertainty),
                      "" if peak.intensity is None else f"{peak.intensity:.7g}")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 3:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column == 0:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
                self.peak_table.setItem(row, column, item)
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.peak_table, "")
        result_page = QWidget()
        results_layout = QVBoxLayout(result_page)
        self.candidate_table = QTableWidget(0, 13)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidate_table.horizontalHeader().setMinimumSectionSize(55)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_changed)
        results_layout.addWidget(self.candidate_table)
        self.line_table = QTableWidget(0, 7)
        self.line_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.line_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.line_table)
        self.tabs.addTab(result_page, "")
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.tabs.addTab(self.diagnostics, "")
        root.addWidget(self.tabs, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        actions = QHBoxLayout()
        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton()
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        self.create_button = QPushButton()
        self.create_button.setEnabled(False)
        self.create_button.clicked.connect(self._create_phase)
        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.close)
        for button in (self.run_button, self.cancel_button, self.create_button):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.close_button)
        root.addLayout(actions)

    def retranslate(self):
        self.setWindowTitle(localised("Index cell", "Indexer la maille", "Индексация ячейки")
                            + " — " + self.measurement_name)
        self.note.setText(localised(
            "Experimental indexing with limited search. Review alternate solutions. "
            "Positions use the measurement coordinates; apply corrections and refit peaks before indexing.",
            "Indexation expérimentale à recherche limitée. Vérifiez les solutions alternatives. "
            "Les positions utilisent les coordonnées de mesure ; appliquez les corrections puis réajustez les pics.",
            "Экспериментальная индексация с ограниченным поиском. Проверяйте альтернативные решения. "
            "Используются координаты измерения: перед индексацией примените коррекции и уточните пики."))
        self.settings_box.setTitle(localised("Search settings", "Paramètres de recherche", "Настройки поиска"))
        self.method_label.setText(localised("Method", "Méthode", "Метод"))
        unit = {"two_theta": "°2θ", "d": "Å", "q": "1/Å^2"}[self.input_kind]
        self.error_label.setText(localised("Allowed position error", "Erreur de position admise", "Допуск положения") + ", " + unit)
        self.position_error.setToolTip(localised(
            "Used for blank per-peak errors. Centre uncertainty is independent of FWHM.",
            "Utilisée si l’erreur du pic est vide. Elle est indépendante de la largeur du pic.",
            "Для пиков с пустой погрешностью. Погрешность центра не определяется через ПШПВ."))
        self.search_label.setText(localised("Low-angle search lines", "Raies de recherche", "Низкоугловых линий для поиска"))
        self.min_label.setText(localised("Minimum cell length, Å", "Longueur minimale, Å", "Минимальная длина, Å"))
        self.max_label.setText(localised("Maximum cell length, Å", "Longueur maximale, Å", "Максимальная длина, Å"))
        self.volume_label.setText(localised("Maximum volume, Å^3", "Volume maximal, Å^3", "Максимальный объём, Å^3"))
        self.hkl_label.setText(localised("Maximum |h|, |k|, |l|", "Maximum |h|, |k|, |l|", "Максимум |h|, |k|, |l|"))
        self.systems_box.setTitle(localised("Crystal systems", "Systèmes cristallins", "Кристаллические системы"))
        for system, check in self.system_checks.items():
            check.setText(system_label(system))
        self.spurious_label.setText(localised("Allowed unindexed search lines", "Raies non indexées admises", "Допустимо неиндексированных линий"))
        self.zero_check.setText(localised("Search zero shift", "Rechercher le décalage zéro", "Искать нулевой сдвиг"))
        self.peak_table.setHorizontalHeaderLabels([
            localised("Use", "Utiliser", "Использовать"), "ID", "Position, " + unit,
            localised("Centre error (optional)", "Erreur du centre (facultative)", "Погрешность центра (необяз.)"),
            tr("qt.viewer_intensity")])
        self.peak_table.resizeColumnsToContents()
        self.candidate_table.setHorizontalHeaderLabels([
            "#", localised("System", "Système", "Система"), "a, Å", "b, Å", "c, Å",
            "α, °", "β, °", "γ, °", "V, Å^3",
            localised("Indexed / used", "Indexés / utilisés", "Индексировано / использовано"),
            "M20", "F20", localised("Zero correction, °", "Correction zéro, °", "Нулевая коррекция, °")])
        self.line_table.setHorizontalHeaderLabels([
            "ID", "Qobs, 1/Å^2", "Qcalc, 1/Å^2", "ΔQ = Qobs − Qcalc", "hkl",
            "Δ2θ, °", localised("Status", "État", "Статус")])
        for index, texts in enumerate((('Peaks', 'Pics', 'Пики'), ('Candidates', 'Solutions', 'Кандидаты'),
                                      ('Diagnostics', 'Diagnostic', 'Диагностика'))):
            self.tabs.setTabText(index, localised(*texts))
        self.run_button.setText(localised("Run indexing", "Lancer l’indexation", "Запустить индексацию"))
        self.cancel_button.setText(tr("text.cancel"))
        self.create_button.setText(localised("Create cell phase…", "Créer une phase…", "Создать Cell Phase…"))
        self.close_button.setText(tr("text.close"))
        self._update_reference()
        if self.result is not None:
            self._display_result()

    def _update_reference(self):
        module = METHODS[self.method_combo.currentData()]
        publication = module.VISSER_PUBLICATION if module.METHOD_ID == "visser" else module.BOULTIF_LOUER_PUBLICATION
        self.reference.setText(html.escape(publication.citation) +
                               f' <a href="{publication.url}">DOI</a>')

    def _method_changed(self, *_args):
        is_bl = self.method_combo.currentData() == "boultif_louer"
        self.systems_box.setVisible(is_bl)
        for widget in (self.spurious, self.spurious_label, self.zero_check, self.zero_range,
                       self.max_volume, self.volume_label):
            widget.setEnabled(is_bl)
        if self.input_kind != "two_theta":
            self.zero_check.setEnabled(False)
            self.zero_check.setChecked(False)
        self.search_count.setMinimum(8 if is_bl else 20)
        self._update_reference()

    def _request(self):
        peaks = []
        for row, peak in enumerate(self.peaks):
            value = self.peak_table.item(row, 3).text().strip()
            try:
                error = float(value.replace(",", ".")) if value else self.position_error.value()
            except ValueError as exc:
                raise IndexingError("invalid_uncertainty", peak=row + 1) from exc
            peaks.append(replace(peak, uncertainty=error,
                enabled=self.peak_table.item(row, 0).checkState() == Qt.CheckState.Checked))
        common = dict(search_peak_count=self.search_count.value(),
                      min_cell_length=self.min_length.value(), max_cell_length=self.max_length.value(),
                      max_input_peaks=max(80, len(peaks)))
        if self.method_combo.currentData() == "visser":
            settings = VisserSettings(**common, trial_hkl_max=self.hkl_limit.value())
        else:
            systems = tuple(system for system, check in self.system_checks.items() if check.isChecked())
            settings = BoultifLouerSettings(**common, systems=systems,
                max_cell_volume=self.max_volume.value(), hkl_max=self.hkl_limit.value(),
                max_spurious_search_lines=self.spurious.value(), search_zero_shift=self.zero_check.isChecked(),
                zero_shift_half_range_deg=self.zero_range.value())
        try:
            settings.validate()
        except ValueError as exc:
            raise IndexingError("invalid_settings", reason=str(exc)) from exc
        return IndexingRequest(tuple(peaks), self.method_combo.currentData(), self.input_kind,
                               self.wavelength.value(), settings)

    def start(self):
        if self._closed or self._stale or self._future is not None:
            return
        try:
            request = self._request()
        except IndexingError as error:
            self.status.setText(error_text(error))
            return
        self.result = None
        self.candidate_table.setRowCount(0)
        self.line_table.setRowCount(0)
        self.diagnostics.clear()
        self.create_button.setEnabled(False)
        self.settings_box.setEnabled(False)
        self.peak_table.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText(localised("Indexing…", "Indexation…", "Индексация…"))
        self._cancel = Event()
        self._future = self._executor.submit(run_indexing, request, self._cancel)
        self._timer.start()

    def cancel(self):
        self._cancel.set()
        self.cancel_button.setEnabled(False)

    def invalidate(self):
        self._stale = True
        self.cancel()
        self.run_button.setEnabled(False)
        self.create_button.setEnabled(False)
        self.status.setText(localised(
            "Measurement or peaks changed. Reopen indexing to use the current data.",
            "La mesure ou les pics ont changé. Rouvrez l’indexation avec les données actuelles.",
            "Измерение или пики изменились. Откройте индексацию заново для актуальных данных."))

    def _poll(self):
        if self._future is None or not self._future.done():
            return
        self._timer.stop()
        future, self._future = self._future, None
        self.settings_box.setEnabled(True)
        self.peak_table.setEnabled(True)
        self.run_button.setEnabled(not self._stale)
        self.cancel_button.setEnabled(False)
        if self._stale:
            return
        try:
            result = future.result()
            if self._cancel.is_set():
                raise IndexingCancelled()
        except IndexingCancelled:
            self.status.setText(localised("Cancelled.", "Annulé.", "Отменено."))
        except Exception as error:
            self.status.setText(error_text(error))
        else:
            self.result = result
            self._display_result()
            self.tabs.setCurrentIndex(1)
            self.status.setText(localised(
                f"Candidates: {len(result.candidates)}. Search limits do not guarantee a solution.",
                f"Solutions : {len(result.candidates)}. Les limites ne garantissent pas de solution.",
                f"Кандидатов: {len(result.candidates)}. Ограничения поиска не гарантируют нахождение решения."))

    def _display_result(self):
        self.candidate_table.setRowCount(len(self.result.candidates))
        for row, candidate in enumerate(self.result.candidates):
            total = len(candidate.indexed_lines) + len(candidate.unindexed_lines)
            values = [str(candidate.rank), system_label(candidate.crystal_system),
                      *[_show_number(x) for x in candidate.cell.as_tuple()], _show_number(candidate.volume),
                      f"{len(candidate.indexed_lines)} / {total}", _show_number(candidate.m20),
                      _show_number(candidate.f20), _show_number(candidate.zero_shift_deg)]
            for column, value in enumerate(values):
                self.candidate_table.setItem(row, column, QTableWidgetItem(value))
        self.candidate_table.resizeColumnsToContents()
        if self.result.candidates:
            self.candidate_table.selectRow(0)
        else:
            self.diagnostics.setPlainText(json.dumps(self.result.diagnostics, indent=2, ensure_ascii=False))

    def _selected_candidate(self):
        row = self.candidate_table.currentRow()
        if self.result is not None and 0 <= row < len(self.result.candidates):
            return self.result.candidates[row]
        return None

    def _candidate_changed(self):
        candidate = self._selected_candidate()
        self.create_button.setEnabled(candidate is not None and not self._stale and self._future is None)
        if candidate is None:
            return
        lines = sorted([*candidate.indexed_lines, *candidate.unindexed_lines], key=lambda l: l.observed_q)
        self.line_table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            indexed = hasattr(line, "calculated_q")
            values = (str(line.peak_id), _show_number(line.observed_q),
                      _show_number(line.calculated_q) if indexed else "—",
                      _show_number(line.delta_q) if indexed else "—",
                      f"{line.h} {line.k} {line.l}" if indexed else "—",
                      _show_number(line.delta_two_theta) if indexed else "—",
                      localised("Indexed", "Indexé", "Индексирован") if indexed
                      else localised("Unindexed", "Non indexé", "Не индексирован"))
            for column, value in enumerate(values):
                self.line_table.setItem(row, column, QTableWidgetItem(value))
        self.line_table.resizeColumnsToContents()
        report = dict(self.result.diagnostics)
        report.update(candidate=asdict(candidate), publications=[asdict(p) | {"url": p.url,
            "citation": p.citation, "bibtex": p.bibtex} for p in self.result.publications])
        self.diagnostics.setPlainText(json.dumps(report, indent=2, ensure_ascii=False))

    def _create_phase(self):
        candidate = self._selected_candidate()
        if candidate is not None and not self._stale and self._future is None:
            self.create_phase_requested.emit(candidate, self.result)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._cancel.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def done(self, result):
        self.shutdown()
        super().done(result)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
