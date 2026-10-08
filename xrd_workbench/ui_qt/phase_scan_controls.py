"""Viewer controls for scan geometry and per-phase crystal orientation."""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..localization import localised, tr
from ..models.phase_scan import AngleRange, PhaseOrientation, PhaseScanState, REQUIRED_ANGLES


class PhaseScanControls(QWidget):
    changed = Signal()
    phi_shift_changed = Signal(object)

    def __init__(self, parent=None, *, state=None):
        super().__init__(parent)
        self.state = state if state is not None else PhaseScanState()
        self._syncing = False
        self._scans = {}
        self._phases = {}
        # Invalid committed drafts remain local. Calculations and recreated
        # controls use only the typed, accepted values in the project model.
        self._angle_drafts = {}
        self._draft_bounds = {}
        self._orientation_drafts = {}
        self._loaded_angles = None
        self._context = None
        self._orientation_uid = None
        self._loaded_orientation = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        outer.addLayout(form)
        self.mode_label = QLabel()
        self.mode_combo = QComboBox()
        form.addRow(self.mode_label, self.mode_combo)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.reference_label = QLabel()
        self.reference_combo = QComboBox()
        form.addRow(self.reference_label, self.reference_combo)
        self.reference_combo.currentIndexChanged.connect(self._reference_changed)
        self.geometry_group = QGroupBox()
        outer.addWidget(self.geometry_group)
        self.geometry_layout = QVBoxLayout(self.geometry_group)
        self.geometry_form = QFormLayout()
        self.geometry_layout.addLayout(self.geometry_form)
        self.geometry_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.angle_edits = {}
        self.angle_rows = {}
        self.angle_labels = {}
        for key, label in [('two_theta', '2θ, °'), ('omega', 'ω / θ, °'),
                           ('chi', 'χ, °'), ('phi', 'φ, °'), ('omega_offset', 'ω − θ, °')]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0,0,0,0)
            low,high = QLineEdit(),QLineEdit()
            low.setMinimumWidth(0)
            high.setMinimumWidth(0)
            row_layout.addWidget(low)
            row_layout.addWidget(QLabel('…'))
            row_layout.addWidget(high)
            self.angle_edits[key] = (low,high)
            self.angle_rows[key] = row
            self.angle_labels[key] = QLabel(label)
            self.geometry_form.addRow(self.angle_labels[key], row)
            low.editingFinished.connect(self._edited)
            high.editingFinished.connect(self._edited)
        self.fill_button = QPushButton()
        self.fill_button.clicked.connect(self._fill)
        self.geometry_layout.addWidget(self.fill_button)
        self.max_index_label = QLabel()
        self.max_index_spin = QSpinBox()
        self.max_index_spin.setRange(1, 20)
        self.max_index_spin.setValue(self.state.max_index)
        self.max_index_spin.setKeyboardTracking(False)
        index_layout = QHBoxLayout()
        index_layout.addWidget(self.max_index_label)
        index_layout.addWidget(self.max_index_spin)
        self.geometry_layout.addLayout(index_layout)
        self.max_index_spin.valueChanged.connect(self._index_changed)
        self.orientation_group = QGroupBox()
        outer.addWidget(self.orientation_group)
        self.orientation_form = QFormLayout(self.orientation_group)
        self.orientation_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.phase_label = QLabel()
        self.phase_combo = QComboBox()
        self.orientation_form.addRow(self.phase_label, self.phase_combo)
        self.phase_combo.currentIndexChanged.connect(self._phase_changed)
        self.surface_label = QLabel()
        self.surface_edit = QLineEdit('0 0 1')
        self.orientation_form.addRow(self.surface_label, self.surface_edit)
        self.inplane_label = QLabel()
        self.inplane_edit = QLineEdit('1 0 0')
        self.orientation_form.addRow(self.inplane_label, self.inplane_edit)
        self.phi_zero_label = QLabel()
        self.phi_zero_spin = QDoubleSpinBox()
        self.phi_zero_spin.setRange(-360,360)
        self.phi_zero_spin.setDecimals(3)
        self.phi_zero_spin.setSingleStep(.1)
        self.phi_zero_spin.setKeyboardTracking(False)
        self.phi_zero_slider = QSlider(Qt.Orientation.Horizontal)
        self.phi_zero_slider.setRange(-3600,3600)
        self.phi_zero_slider.setPageStep(100)
        self.orientation_form.addRow(self.phi_zero_label, self.phi_zero_spin)
        self.orientation_form.addRow(self.phi_zero_slider)
        self.phi_zero_slider.valueChanged.connect(lambda value:self.phi_zero_spin.setValue(value/10))
        self.phi_zero_spin.valueChanged.connect(self._phi_shift_edited)
        for edit in (self.surface_edit, self.inplane_edit):
            edit.editingFinished.connect(self._orientation_edited)
        self.message = QLabel()
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        self.retranslate()

    def retranslate(self):
        previous = self.state.mode
        self._syncing = True
        self.mode_combo.clear()
        for label, key in [(tr('text.auto'), 'auto'),
                           (localised('Powder 2θ', 'Poudre 2θ', 'Порошковый 2θ'), 'powder'),
                           ('2θ–ω', 'coupled'),
                           (localised('2θ, fixed ω', '2θ, ω fixe', '2θ, фиксированный ω'), 'detector'),
                           ('φ', 'phi'), ('χ', 'chi'), ('ω / θ', 'omega')]:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(previous)))
        self.mode_label.setText(localised('Phase scan', 'Balayage de phase', 'Скан фазы'))
        self.reference_label.setText(localised('Geometry from', 'Géométrie de', 'Геометрия измерения'))
        self.geometry_group.setTitle(localised('Fixed-angle ranges', 'Plages des angles fixes', 'Диапазоны фиксированных углов'))
        self.fill_button.setText(localised('Use measurement angles', 'Utiliser les angles mesurés', 'Подставить углы измерения'))
        self.max_index_label.setText(localised('Maximum |h|, |k|, |l|', 'Maximum |h|, |k|, |l|', 'Максимум |h|, |k|, |l|'))
        self.orientation_group.setTitle(localised('Phase orientation', 'Orientation de phase', 'Ориентация фазы'))
        self.phase_label.setText(tr('pole.search_phase'))
        self.surface_label.setText(localised('Surface normal h k l', 'Normale de surface h k l', 'Нормаль поверхности h k l'))
        self.inplane_label.setText(localised('Sample X direction h k l', 'Direction X h k l', 'Направление X образца h k l'))
        self.phi_zero_label.setText(localised('φ zero offset, °', 'Décalage zéro φ, °', 'Смещение нуля φ, °'))
        self.phi_zero_slider.setAccessibleName(self.phi_zero_label.text())
        for low,high in self.angle_edits.values():
            low.setPlaceholderText(localised('From', 'De', 'От'))
            high.setPlaceholderText(localised('To', 'À', 'До'))
        self._syncing = False
        self._update_visibility()

    @property
    def mode(self):
        item = self._scans.get(self.state.reference_uid)
        return self.state.resolved_mode(item.scan if item is not None else None)

    @property
    def axis(self):
        return 'two_theta' if self.mode in {'powder', 'coupled', 'detector'} else self.mode

    def sync(self, scans, phases, selected_uid=None):
        self._syncing = True
        old = self.state.reference_uid
        self._scans = {item.uid: item for item in scans}
        self._phases = {item.uid: item for item in phases}
        uid = selected_uid if selected_uid in self._scans else old
        if uid not in self._scans:
            uid = next(iter(self._scans), None)
        self.reference_combo.clear()
        for item in scans:
            self.reference_combo.addItem(item.name, item.uid)
        if uid is not None:
            self.reference_combo.setCurrentIndex(self.reference_combo.findData(uid))
        self.state.reference_uid = uid
        phase_uid = self.state.phase_uid
        self.phase_combo.clear()
        for item in phases:
            self.phase_combo.addItem(item.name, item.uid)
        if phase_uid in self._phases:
            self.phase_combo.setCurrentIndex(self.phase_combo.findData(phase_uid))
        self.state.phase_uid = self.phase_combo.currentData()
        for key in tuple(self._angle_drafts):
            if key[0] is not None and key[0] not in self._scans:
                del self._angle_drafts[key]
                self._draft_bounds.pop(key, None)
        for key in tuple(self._orientation_drafts):
            if key not in self._phases:
                del self._orientation_drafts[key]
        self._syncing = False
        self._load_context()
        self._load_orientation()
        self._update_visibility()

    def _load_context(self):
        item = self._scans.get(self.state.reference_uid)
        scan = item.scan if item is not None else None
        key = self.state.context(self.state.reference_uid, scan)
        bounds = self.state.ensure_angles(self.state.reference_uid, scan)
        if key == self._context and bounds is self._loaded_angles:
            return
        # A new bounds mapping for the same key means the source was replaced.
        if key == self._context or (key in self._draft_bounds and self._draft_bounds[key] is not bounds):
            self._angle_drafts.pop(key, None)
            self._draft_bounds.pop(key, None)
        self._context = key
        self._loaded_angles = bounds
        for name,edits in self.angle_edits.items():
            accepted = bounds.get(name, AngleRange())
            values = self._angle_drafts.get(key, {}).get(name, tuple(
                '' if value is None else format(value, '.10g')
                for value in (accepted.low, accepted.high)))
            for edit,value in zip(edits, values):
                edit.setText(value)
                edit.setCursorPosition(0)

    def _reference_changed(self, *_):
        if self._syncing:
            return
        self.state.reference_uid = self.reference_combo.currentData()
        self._load_context()
        self._update_visibility()
        self.changed.emit()

    def _mode_changed(self, *_):
        if self._syncing:
            return
        self.state.mode = self.mode_combo.currentData() or 'auto'
        self._update_visibility()
        self.changed.emit()

    def _fill(self):
        self.state.angles.pop(self._context, None)
        self._angle_drafts.pop(self._context, None)
        self._draft_bounds.pop(self._context, None)
        self._context = None
        self._load_context()
        self._notify()

    def _edited(self):
        if self._context is not None:
            bounds = self.state.angles[self._context]
            self._draft_bounds[self._context] = bounds
            drafts = self._angle_drafts.setdefault(self._context, {})
            for key, edits in self.angle_edits.items():
                text = tuple(edit.text() for edit in edits)
                try:
                    self.state.set_angle(self._context, key, AngleRange(*(self._number(value) for value in text)))
                    drafts.pop(key, None)
                except ValueError:
                    drafts[key] = text
        self._notify()

    def _index_changed(self, value):
        self.state.max_index = value
        self._notify()

    def _load_orientation(self):
        uid = self.state.phase_uid
        orientation = self.state.orientation(uid)
        if uid == self._orientation_uid and orientation == self._loaded_orientation:
            return
        if uid == self._orientation_uid:
            self._orientation_drafts.pop(uid, None)
        self._orientation_uid = uid
        self._loaded_orientation = orientation
        values = self._orientation_drafts.get(uid, (
            ' '.join(map(str, orientation.surface)), ' '.join(map(str, orientation.inplane))))
        for edit, value in zip((self.surface_edit, self.inplane_edit), values[:2]):
            edit.setText(value)
        self.phi_zero_spin.blockSignals(True)
        self.phi_zero_spin.setValue(orientation.phi_zero)
        self.phi_zero_spin.blockSignals(False)
        self._sync_phi_slider()

    def _phase_changed(self, *_):
        if not self._syncing:
            self.state.phase_uid = self.phase_combo.currentData()
            self._load_orientation()

    def _orientation_edited(self):
        uid = self.phase_combo.currentData()
        if uid is not None:
            values = self.surface_edit.text(), self.inplane_edit.text()
            try:
                orientation = PhaseOrientation(*(tuple(int(v) for v in text.split()) for text in values),
                                               self.state.orientation(uid).phi_zero)
                self.state.set_orientation(uid, orientation)
                self._loaded_orientation = orientation
                self._orientation_drafts.pop(uid, None)
            except ValueError:
                self._orientation_drafts[uid] = values
        self._notify()

    def _sync_phi_slider(self):
        self.phi_zero_slider.blockSignals(True)
        self.phi_zero_slider.setValue(round(self.phi_zero_spin.value()*10))
        self.phi_zero_slider.blockSignals(False)

    def _phi_shift_edited(self, *_):
        self._sync_phi_slider()
        uid = self.phase_combo.currentData()
        if uid is None or self._syncing:
            return
        self.state.set_phi_offset(uid, self.phi_zero_spin.value())
        self._loaded_orientation = self.state.orientation(uid)
        self.phi_shift_changed.emit(uid)

    def _notify(self, *_):
        if not self._syncing:
            self.changed.emit()

    def _update_visibility(self):
        oriented = self.mode != 'powder'
        self.geometry_group.setVisible(oriented)
        self.orientation_group.setVisible(oriented)
        self.reference_label.setVisible(oriented)
        self.reference_combo.setVisible(oriented)
        required = REQUIRED_ANGLES.get(self.mode, ())
        for key,row in self.angle_rows.items():
            self.angle_labels[key].setVisible(key in required)
            row.setVisible(key in required)
        self.fill_button.setEnabled(bool(self._scans))

    @staticmethod
    def _number(text):
        if not text.strip():
            return None
        value = float(text.strip().replace(',', '.'))
        if not math.isfinite(value):
            raise ValueError(localised('Enter finite angles.', 'Saisissez des angles finis.', 'Введите конечные значения углов.'))
        return value

    def geometry(self):
        item = self._scans.get(self.state.reference_uid)
        scan = item.scan if item is not None else None
        required = REQUIRED_ANGLES.get(self.mode, ())
        bounds = self.state.ensure_angles(self.state.reference_uid, scan)
        for key in required:
            draft = self._angle_drafts.get(self._context, {}).get(key)
            if draft is not None:
                low, high = (self._number(value) for value in draft)
                if low is not None and high is not None and low > high:
                    raise ValueError(localised('Range start must not exceed its end: ',
                        'Le début de plage ne doit pas dépasser sa fin : ',
                        'Начало диапазона не должно превышать конец: ') + self.angle_labels[key].text())
                self.state.validate_angle(key, AngleRange(low, high))
        missing = [self.angle_labels[key].text() for key in required
                   if not bounds.get(key, AngleRange()).complete]
        if missing:
            raise ValueError(localised('Enter fixed angles: ', 'Saisissez les angles fixes : ',
                                      'Введите фиксированные углы: ') + ', '.join(missing))
        return self.state.geometry(self.state.reference_uid, scan)

    def orientation(self, uid):
        if uid in self._orientation_drafts:
            raise ValueError(localised('Enter nonzero, nonparallel integer h k l triples and a finite φ offset.',
                                      'Saisissez des triplets h k l entiers non nuls et non parallèles, et un décalage φ fini.',
                                      'Введите ненулевые непараллельные целочисленные тройки h k l и конечное смещение φ.'))
        return self.state.orientation(uid)
