"""Save portable settings and reconstruct projects from referenced source files."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from ..cell_phase import create_cell_phase_document
from ..io.session_codec import encode, decode, expect, SessionFormatError
from ..models.analysis import SessionPeak
from ..models.correction import CorrectionRequest
from ..models.project import ProjectDocument, ProjectStore, VIEWER, STRUCTURES, POLES, RSM, SCAN, CIF, CELL_PHASE, POLE_DATA, RSM_DATA, WORKSPACES
from ..models.radiation import RadiationSettings
from ..models.scan import assign_text_axis
from ..models.viewer import ViewerState
from ..models.pole_state import PolesState
from ..models.structures_state import StructuresState
from ..models.rsm_state import RSMState
from ..space_groups import BY_HALL_NUMBER
from .correction import apply_correction
from .experimental_pole import ExperimentalPoleMeasurement, find_exported_xy_files, load_experimental_pole, load_xy_series
from .pole_figure import available_reflections

FORMAT = 'xrd-combine-session'
SCHEMA_VERSION = 1
PROJECT_EXTENSION = '.xrdproject'
TAB_EXTENSION = '.xrdtab'
_OWNERS = {VIEWER: ViewerState, STRUCTURES: StructuresState, POLES: PolesState, RSM: RSMState}
_ITEM_FIELDS = ('visible', 'colour', 'x_shift', 'x_scale', 'y_shift', 'y_factor', 'shift_omega', 'show_peak_sum')


class SessionCancelled(Exception):
    pass


@dataclass(frozen=True)
class SourceIssue:
    path: Path
    reason: str
    message: str = ''


@dataclass
class LoadedSession:
    store: ProjectStore
    radiation: RadiationSettings
    scope: str
    application: dict
    warnings: list[str]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source(path, destination, digest=None):
    absolute = Path(path).resolve()
    try:
        relative = os.path.relpath(absolute, Path(destination).resolve().parent).replace(os.sep, '/')
    except ValueError:
        relative = None
    return {'relative': relative, 'absolute': str(absolute),
        'sha256': digest or (file_digest(absolute) if absolute.is_file() else None)}


def _recipe(document):
    scan = document.payload
    if document.kind == CELL_PHASE:
        return {'hall_number': scan.setting.hall_number, 'cell': list(scan.cell), 'name': scan.name}
    if document.kind != SCAN:
        return {}
    metadata = scan.metadata
    steps = metadata.get('processing_steps', [])
    if metadata.get('processing') and not steps:
        raise SessionFormatError('This processed scan has no replayable operation history; export it and reopen the exported file first.')
    return {'index': int(metadata.get('scan_index', metadata.get('range_index', 0)) or 0),
        'axis_name': scan.axis_name, 'text_axis': scan.axis_name if metadata.get('format') == 'XY' else None,
        'axis_assumed': bool(metadata.get('axis_assumed', False)), 'steps': copy.deepcopy(steps)}


def snapshot(store, radiation, destination, *, scope='project', application=None):
    if scope not in {'project', *WORKSPACES}:
        raise SessionFormatError('Unknown session scope')
    workspaces = WORKSPACES if scope == 'project' else (scope,)
    selected = set(store.documents) if scope == 'project' else set(store.assignments[scope])
    if scope == RSM and store.rsm.experimental.phase_uid:
        selected.add(store.rsm.experimental.phase_uid)
    if scope == VIEWER:
        selected.update(assignment[0] for peak in store.analysis.peaks if peak.scan_uid in selected for assignment in peak.hkl_assignments)
    pending = list(selected)
    while pending:
        uid = pending.pop()
        document = store.documents.get(uid)
        if document and document.parent_uid and document.parent_uid not in selected:
            selected.add(document.parent_uid)
            pending.append(document.parent_uid)
    source_keys = set(store._source_keys.values())
    records = []
    for uid, doc in store.documents.items():
        if uid not in selected:
            continue
        record = {'uid': uid, 'kind': doc.kind, 'name': doc.name, 'parent_uid': doc.parent_uid,
            'source': _source(doc.source, destination, doc.source_digest) if doc.kind != CELL_PHASE else None,
            'recipe': _recipe(doc), 'derived': doc.kind == SCAN and uid not in source_keys}
        if doc.kind == POLE_DATA and isinstance(doc.payload, ExperimentalPoleMeasurement) and doc.payload.raw is None:
            record['pole_xy'] = [_source(path, destination) for path in
                (doc.payload.source_files or find_exported_xy_files(doc.source))]
        records.append(record)
    state = {workspace: getattr(store, workspace) for workspace in workspaces}
    items = [{'uid': item.uid, 'axis_name': item.scan.axis_name if item.scan else None,
        **{name: getattr(item, name) for name in _ITEM_FIELDS}} for item in store.viewer.items.values()] if VIEWER in workspaces else []
    return {'format': FORMAT, 'schema_version': SCHEMA_VERSION, 'application_version': _version(),
        'scope': scope, 'documents': records,
        'assignments': {ws: [uid for uid in store.documents if uid in store.assignments[ws]] for ws in workspaces},
        'states': encode(state), 'viewer_items': items, 'radiation': encode(radiation),
        'analysis': encode({'peaks': list(store.analysis.peaks) if VIEWER in workspaces else [],
            'backgrounds': list(store.analysis.backgrounds) if VIEWER in workspaces else [],
            'next_peak_number': store.analysis.next_peak_number}),
        'application': application or {}}


def _version():
    from ..version import APP_VERSION
    return APP_VERSION


def write_session(path, store, radiation, *, scope='project', application=None):
    destination = Path(path)
    sources = {doc.source.resolve() for doc in store.documents.values() if doc.kind != CELL_PHASE}
    for doc in store.documents.values():
        if isinstance(doc.payload, ExperimentalPoleMeasurement):
            sources.update(Path(source).resolve() for source in doc.payload.source_files)
    if destination.resolve() in sources:
        raise SessionFormatError('A session file cannot overwrite an original source file.')
    data = snapshot(store, radiation, destination, scope=scope, application=application)
    # Decode our own schema before writing: malformed accepted data cannot replace
    # a previous save. The JSON contains no pickle, source arrays or executable code.
    _checked(data)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if len(text.encode('utf-8')) > 64 * 1024 * 1024:
        raise SessionFormatError('Session file exceeds 64 MiB')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.resolve().parent,
                prefix=destination.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SessionFormatError('Duplicate JSON key')
        result[key] = value
    return result


def read_manifest(path):
    source = Path(path)
    if source.stat().st_size > 64 * 1024 * 1024:
        raise SessionFormatError('Session file exceeds 64 MiB')
    try:
        value = json.loads(source.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(SessionFormatError(f'Invalid JSON number: {item}')))
        _checked(value)
        return value
    except (TypeError, KeyError, AttributeError, RecursionError, json.JSONDecodeError) as exc:
        raise SessionFormatError(f'Invalid session file: {exc}') from exc


def _checked(data):
    try:
        _validate(data)
    except (TypeError, KeyError, AttributeError, IndexError, RecursionError, ValueError) as exc:
        if isinstance(exc, SessionFormatError):
            raise
        raise SessionFormatError(f'Invalid session file: {exc}') from exc


def _validate(data):
    if not isinstance(data, dict) or data.get('format') != FORMAT or type(data.get('schema_version')) is not int or data['schema_version'] != SCHEMA_VERSION:
        raise SessionFormatError('Unsupported session format or schema version')
    scope = data.get('scope')
    if scope not in {'project', *WORKSPACES}:
        raise SessionFormatError('Unknown session scope')
    workspaces = set(WORKSPACES if scope == 'project' else (scope,))
    documents = data.get('documents')
    if not isinstance(documents, list):
        raise SessionFormatError('Missing document list')
    known = {}
    for record in documents:
        if not isinstance(record, dict) or not isinstance(record.get('uid'), str) or not record['uid'] or record['uid'] in known:
            raise SessionFormatError('Invalid or duplicate document identity')
        if record.get('kind') not in {SCAN, CIF, CELL_PHASE, POLE_DATA, RSM_DATA} or not isinstance(record.get('name'), str) or not record['name'].strip():
            raise SessionFormatError('Invalid document kind or name')
        if type(record.get('derived')) is not bool or not isinstance(record.get('recipe'), dict):
            raise SessionFormatError('Invalid document recipe')
        if record['kind'] == CELL_PHASE:
            recipe = record['recipe']
            if type(recipe.get('hall_number')) is not int or recipe['hall_number'] not in BY_HALL_NUMBER or len(recipe.get('cell', [])) != 6:
                raise SessionFormatError('Invalid manually defined cell')
            create_cell_phase_document(recipe['name'], BY_HALL_NUMBER[recipe['hall_number']], tuple(recipe['cell']))
        else:
            _validate_source(record.get('source'))
        if record['kind'] == SCAN:
            recipe = record['recipe']
            if type(recipe.get('index')) is not int or recipe['index'] < 0 or not isinstance(recipe.get('axis_name'), str) or not isinstance(recipe.get('steps'), list):
                raise SessionFormatError('Invalid scan recipe')
            if recipe.get('text_axis') is not None and not isinstance(recipe['text_axis'], str):
                raise SessionFormatError('Invalid text scan axis')
            for step in recipe['steps']:
                if not isinstance(step, dict) or set(step) != {'axis_name', 'x_shift', 'x_scale', 'y_shift', 'y_factor', 'shift_omega_half'} or not isinstance(step['axis_name'], str) or type(step['shift_omega_half']) is not bool or any(type(step[key]) not in (int, float) for key in ('x_shift', 'x_scale', 'y_shift', 'y_factor')):
                    raise SessionFormatError('Invalid correction recipe')
                CorrectionRequest(**{key: value for key, value in step.items() if key != 'axis_name'})
        for ref in record.get('pole_xy', []):
            _validate_source(ref)
        known[record['uid']] = record
    for record in documents:
        if record.get('parent_uid') is not None and record['parent_uid'] not in known:
            raise SessionFormatError('Unknown parent document')
        parents, uid = set(), record['uid']
        while uid is not None:
            if uid in parents:
                raise SessionFormatError('Circular document parent references')
            parents.add(uid)
            uid = known[uid]['parent_uid']
    assignments = data.get('assignments')
    if not isinstance(assignments, dict) or set(assignments) != workspaces:
        raise SessionFormatError('Invalid workspace assignments')
    for ws, ids in assignments.items():
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(uid not in known or not ProjectStore.compatible(known[uid]['kind'], ws) for uid in ids):
            raise SessionFormatError('Invalid workspace source')
        structural = sum(known[uid]['kind'] in {CIF, CELL_PHASE} for uid in ids)
        if ws != VIEWER and structural > (2 if ws == POLES else 1):
            raise SessionFormatError('Too many phases in workspace')
        if ws == POLES and sum(known[uid]['kind'] == POLE_DATA for uid in ids) > 1:
            raise SessionFormatError('Too many experimental pole sources')
    states = decode(data.get('states'))
    if not isinstance(states, dict) or set(states) != workspaces:
        raise SessionFormatError('Invalid workspace settings')
    for ws in workspaces:
        expect(states[ws], _OWNERS[ws])
        _validate_ownership(ws, states[ws], assignments[ws], known)
    expect(decode(data.get('radiation')), RadiationSettings)
    analysis = decode(data.get('analysis'))
    if not isinstance(analysis, dict) or set(analysis) != {'peaks', 'backgrounds', 'next_peak_number'} or type(analysis['next_peak_number']) is not int or analysis['next_peak_number'] < 1:
        raise SessionFormatError('Invalid analysis section')
    numbers = set()
    for peak in analysis['peaks']:
        expect(peak, SessionPeak)
        if peak.scan_uid not in assignments.get(VIEWER, []) or known[peak.scan_uid]['kind'] != SCAN or peak.number in numbers:
            raise SessionFormatError('Invalid peak ownership or identity')
        numbers.add(peak.number)
        for phase, h, k, l in peak.hkl_assignments:
            if phase not in known or known[phase]['kind'] not in {CIF, CELL_PHASE} or (h, k, l) == (0, 0, 0):
                raise SessionFormatError('Invalid peak phase assignment')
    peaks = {peak.number: peak for peak in analysis['peaks']}
    for peak in peaks.values():
        if peak.parent_number is not None:
            parent = peaks.get(peak.parent_number)
            if parent is None or parent.number == peak.number or parent.parent_number is not None or (parent.scan_uid, parent.axis_name) != (peak.scan_uid, peak.axis_name):
                raise SessionFormatError('Invalid companion peak relationship')
    backgrounds = set()
    for uid, background in analysis['backgrounds']:
        from ..models.background import BackgroundAnchors
        expect(background, BackgroundAnchors)
        if uid not in assignments.get(VIEWER, []) or known[uid]['kind'] != SCAN or (uid, background.axis_name) in backgrounds:
            raise SessionFormatError('Invalid background ownership')
        backgrounds.add((uid, background.axis_name))
    _validate_application(data.get('application', {}))
    if not isinstance(data.get('viewer_items'), list) or len(data['viewer_items']) != len(assignments.get(VIEWER, [])) or {item.get('uid') for item in data['viewer_items']} != set(assignments.get(VIEWER, [])):
        raise SessionFormatError('Invalid Viewer item list')
    for item in data['viewer_items']:
        import re
        if set(item) != {'uid', 'axis_name', *_ITEM_FIELDS} or any(type(item[key]) is not bool for key in ('visible', 'shift_omega', 'show_peak_sum')) or not isinstance(item['colour'], str) or not re.fullmatch('#[0-9a-fA-F]{6}', item['colour']):
            raise SessionFormatError('Invalid Viewer settings')
        if (item['axis_name'] is not None and not isinstance(item['axis_name'], str)) or any(type(item[key]) not in (int, float) for key in ('x_shift', 'x_scale', 'y_shift', 'y_factor')):
            raise SessionFormatError('Invalid Viewer axis or transformation')
        CorrectionRequest(item['x_shift'], item['x_scale'], item['y_shift'], item['y_factor'], item['shift_omega'])


def _validate_ownership(ws, state, assigned, known):
    def owned(uid, kinds):
        if uid is not None and (uid not in assigned or known[uid]['kind'] not in kinds):
            raise SessionFormatError('Workspace settings reference an unassigned source')
    if ws == VIEWER:
        owned(state.selected_uid, {SCAN, CIF, CELL_PHASE})
        owned(state.phase_scan.reference_uid, {SCAN})
        owned(state.phase_scan.phase_uid, {CIF, CELL_PHASE})
        for uid in state.phase_scan.orientations:
            owned(uid, {CIF, CELL_PHASE})
        for uid, _axis in state.phase_scan.angles:
            owned(uid, {SCAN})
    elif ws == STRUCTURES:
        owned(state.uid, {CIF, CELL_PHASE})
        if state.viewer.uid != state.uid:
            raise SessionFormatError('Structure view source mismatch')
    elif ws == POLES:
        owned(state.calculated.primary_uid, {CIF, CELL_PHASE})
        owned(state.calculated.overlay_uid, {CIF, CELL_PHASE})
        owned(state.experimental.uid, {POLE_DATA})
        if state.calculated.overlay_uid == state.calculated.primary_uid and state.calculated.primary_uid is not None:
            raise SessionFormatError('Duplicate calculated pole source')
        if (state.calculated.overlay_uid is None) != (state.calculated.overlay_layer is None):
            raise SessionFormatError('Missing calculated pole layer')
        if state.calculated.preview.uid != state.calculated.primary_uid:
            raise SessionFormatError('Pole structure preview source mismatch')
    elif ws == RSM:
        owned(state.calculated.uid, {CIF, CELL_PHASE})
        uid = state.experimental.phase_uid
        if uid is not None and (uid not in known or known[uid]['kind'] not in {CIF, CELL_PHASE}):
            raise SessionFormatError('Invalid experimental RSM phase')


def _validate_application(settings):
    from ..atom_styles import PALETTES, ELEMENTS
    import re
    if not isinstance(settings, dict):
        raise SessionFormatError('Invalid application settings')
    choices = {'language': {'en', 'fr', 'ru'}, 'theme': {'system', 'light', 'dark'},
        'renderer': {'matplotlib', 'pyqtgraph'}, 'atom_palette': set(PALETTES), 'workspace': set(WORKSPACES)}
    allowed = {*choices, 'atom_colours', 'window_size', 'maximized', 'drawer_visible', 'panel_width', 'tree_widths'}
    if set(settings) - allowed or any(key in settings and settings[key] not in values for key, values in choices.items()):
        raise SessionFormatError('Invalid application appearance')
    colours = settings.get('atom_colours', {})
    if not isinstance(colours, dict) or any(symbol not in ELEMENTS or not isinstance(colour, str) or not re.fullmatch('#[0-9a-fA-F]{6}', colour) for symbol, colour in colours.items()):
        raise SessionFormatError('Invalid saved atom colours')
    for name in ('maximized', 'drawer_visible'):
        if name in settings and type(settings[name]) is not bool:
            raise SessionFormatError('Invalid saved window setting')
    for name, count in (('window_size', 2), ('tree_widths', 3)):
        if name in settings and (not isinstance(settings[name], list) or len(settings[name]) != count or any(type(v) is not int or not 1 <= v <= 32768 for v in settings[name])):
            raise SessionFormatError('Invalid saved window size')
    if 'panel_width' in settings and (type(settings['panel_width']) is not int or not 220 <= settings['panel_width'] <= 900):
        raise SessionFormatError('Invalid project panel width')


def _validate_source(value):
    import re
    if not isinstance(value, dict) or set(value) != {'relative', 'absolute', 'sha256'} or not isinstance(value['absolute'], str) or not value['absolute']:
        raise SessionFormatError('Invalid source path')
    if value['relative'] is not None and not isinstance(value['relative'], str):
        raise SessionFormatError('Invalid relative source path')
    if value['sha256'] is not None and (not isinstance(value['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', value['sha256'])):
        raise SessionFormatError('Invalid source checksum')


class _Sources:
    def __init__(self, base, resolver):
        self.base, self.resolver = base, resolver
        self.cache, self.digests = {}, {}
        self.warnings = []

    def resolve(self, reference, issue=None):
        key = json.dumps(reference, sort_keys=True)
        if issue is None and key in self.cache:
            return self.cache[key]
        candidates = ([self.base / reference['relative']] if reference['relative'] else []) + [Path(reference['absolute'])]
        chosen = next((path for path in candidates if path.is_file() and self.digest(path) == reference['sha256']), None) if reference['sha256'] else next((path for path in candidates if path.is_file()), None)
        if issue is None and chosen is not None:
            self.cache[key] = (chosen.resolve(), False)
            return self.cache[key]
        path = issue.path if issue is not None else next((path for path in candidates if path.is_file()), candidates[0])
        issue = issue or SourceIssue(path, 'changed' if path.is_file() else 'missing')
        while True:
            if self.resolver is None:
                raise SessionFormatError(f'{issue.reason}: {issue.path}')
            choice = self.resolver(issue)
            if choice is None:
                self.warnings.append(f'Skipped: {path}')
                self.cache[key] = (None, False)
                return self.cache[key]
            allow_changed = isinstance(choice, tuple) and choice[1]
            replacement = Path(choice[0] if isinstance(choice, tuple) else choice)
            if replacement.is_file():
                changed = bool(reference['sha256'] and self.digest(replacement) != reference['sha256'])
                if not changed or allow_changed:
                    self.cache[key] = (replacement.resolve(), changed)
                    if changed:
                        self.warnings.append(f'Changed source; temporary results reset: {replacement}')
                    return self.cache[key]
                issue = SourceIssue(replacement, 'changed')
            else:
                issue = SourceIssue(replacement, 'missing')

    def digest(self, path):
        key = str(Path(path).resolve())
        if key not in self.digests:
            self.digests[key] = file_digest(path)
        return self.digests[key]


def load_session(path, file_service, *, resolver=None):
    data = read_manifest(path)
    sources = _Sources(Path(path).resolve().parent, resolver)
    loaded, reading = ProjectStore(), ProjectStore()
    groups, changed = {}, set()
    for record in data['documents']:
        kind, uid, recipe = record['kind'], record['uid'], record['recipe']
        if kind == CELL_PHASE:
            payload = create_cell_phase_document(recipe['name'], BY_HALL_NUMBER[recipe['hall_number']], tuple(recipe['cell']))
            source, digest, modified = payload.source, None, False
        elif record.get('pole_xy'):
            resolved = [sources.resolve(ref) for ref in record['pole_xy']]
            if any(item[0] is None for item in resolved):
                continue
            source = sources.base / record['source']['relative'] if record['source']['relative'] else Path(record['source']['absolute'])
            xy_paths = tuple(item[0] for item in resolved)
            payload = ExperimentalPoleMeasurement(source, load_xy_series(xy_paths), None, source_files=xy_paths)
            digest, modified = None, any(item[1] for item in resolved)
        else:
            restored = _restore_source(record, sources, file_service, reading, groups)
            if restored is None:
                continue
            payload, source, digest, modified = restored
        document = ProjectDocument(uid, kind, record['name'], Path(source), payload, record['parent_uid'], source_digest=digest)
        loaded.restore_document(document, source_index=None if record['derived'] or kind == CELL_PHASE else recipe.get('index', 0))
        if modified:
            changed.add(uid)
    for document in loaded.documents.values():
        if document.parent_uid not in loaded.documents:
            document.parent_uid = None
    for ws, ids in data['assignments'].items():
        for uid in ids:
            if uid in loaded.documents:
                loaded.assign(uid, ws, additive=ws == POLES)
    states = decode(data['states'])
    for ws, state in states.items():
        setattr(loaded, ws, state)
    _attach(loaded, data, changed)
    radiation = decode(data['radiation'])
    _rebuild_poles(loaded, radiation)
    _rsm_key(loaded, radiation)
    return LoadedSession(loaded, radiation, data['scope'], data.get('application', {}), sources.warnings)


def _restore_source(record, sources, file_service, reading, groups):
    recipe, kind = record['recipe'], record['kind']
    source, modified = sources.resolve(record['source'])
    while source is not None:
        try:
            if kind == POLE_DATA:
                measurement = load_experimental_pole(source)
                payload = measurement.raw if measurement.raw is not None else measurement
            elif kind == RSM_DATA:
                payload = file_service.load_rsm_data(reading, source).payload
            else:
                if source not in groups:
                    groups[source] = file_service.load_path(reading, source)
                matches = [doc.payload for doc in groups[source] if doc.kind == kind and
                    (kind != SCAN or int(doc.payload.metadata.get('scan_index', doc.payload.metadata.get('range_index', 0)) or 0) == recipe['index'])]
                if len(matches) != 1:
                    raise SessionFormatError('The saved range is unavailable.')
                payload = copy.deepcopy(matches[0]) if kind == SCAN else matches[0]
            digest = sources.digest(source)
            if kind == SCAN:
                if recipe.get('text_axis'):
                    assign_text_axis(payload, recipe['text_axis'], assumed=recipe.get('axis_assumed', False))
                for step in recipe.get('steps', []):
                    payload.use_axis(step['axis_name'])
                    payload = apply_correction(payload, CorrectionRequest(**{k: v for k, v in step.items() if k != 'axis_name'}))
                payload.use_axis(recipe['axis_name'])
                payload.name = record['name']
                payload.metadata['source_digest'] = digest
            return payload, source, digest, modified
        except (OSError, ValueError) as exc:
            groups.pop(source, None)
            source, modified = sources.resolve(record['source'], SourceIssue(source, 'unreadable', str(exc)))
    return None


def _attach(store, data, changed):
    docs = store.documents
    if 'viewer' in data['assignments']:
        viewer = store.viewer
        phase_scan, colour_index = viewer.phase_scan, viewer._colour_index
        # Source attachment normally invalidates geometry. The saved geometry
        # is applied afterwards because its source has already been verified.
        viewer.phase_scan = type(phase_scan)()
        for entry in data['viewer_items']:
            uid = entry['uid']
            if uid not in docs:
                continue
            viewer.sync_document(docs[uid])
            item = viewer.items[uid]
            for name in _ITEM_FIELDS:
                setattr(item, name, entry[name])
            if item.scan is not None and entry['axis_name'] in item.scan.available_axes:
                item.scan.use_axis(entry['axis_name'])
            if uid in changed:
                item.reset_transform()
                phase_scan.invalidate_measurement(uid)
                phase_scan.orientations.pop(uid, None)
        viewer.phase_scan, viewer._colour_index = phase_scan, colour_index
        if any(entry['uid'] not in docs or entry['uid'] in changed for entry in data['viewer_items']):
            viewer.plot.viewport = viewer.plot.phase_viewport = None
            viewer.plot.manual_limits = (None,) * 4
        for uid in set(viewer.phase_scan.orientations) | {key[0] for key in viewer.phase_scan.angles}:
            if uid is not None and uid not in docs:
                viewer.phase_scan.remove(uid)
        if viewer.selected_uid not in viewer.items:
            viewer.selected_uid = None
        for name in ('reference_uid', 'phase_uid'):
            if getattr(viewer.phase_scan, name) not in docs:
                setattr(viewer.phase_scan, name, None)
    if 'structures' in data['assignments']:
        structure = store.structures
        if structure.uid in docs:
            structure.payload = docs[structure.uid].payload
            structure.viewer.payload = structure.payload
            if structure.uid in changed:
                structure.viewer.payload = None
                structure.viewer.bind(structure.payload, structure.uid)
                structure.pattern.clear_results(); structure.table.clear_results()
        elif structure.uid is not None:
            structure.remove(structure.uid)
    if 'poles' in data['assignments']:
        poles = store.poles
        calc = poles.calculated
        if calc.primary_uid in docs:
            calc.cif_document = docs[calc.primary_uid].payload
            calc.crystal = calc.cif_document.crystal
            calc.preview.payload = calc.cif_document
            if calc.primary_uid in changed:
                calc.preview.payload = None
                calc.set_primary(calc.cif_document, calc.primary_uid)
        if calc.overlay_uid in docs and calc.overlay_layer is not None:
            calc.overlay_layer.document = docs[calc.overlay_uid].payload
            if calc.overlay_uid in changed:
                from ..models.pole_figure import CalculatedPoleLayer
                old = calc.overlay_layer
                calc.overlay_layer = CalculatedPoleLayer(old.document, old.colour, old.opacity_percent, old.size_percent)
                calc.matplotlib_view = calc.pyqtgraph_view = None
        elif calc.overlay_uid is not None:
            calc.remove_overlay()
        if calc.primary_uid not in docs and calc.primary_uid is not None:
            calc._remove_primary()
        exp = poles.experimental
        if exp.uid in docs:
            exp.source_document = docs[exp.uid].payload
            exp.measurement = load_experimental_pole(docs[exp.uid].source, exp.source_document)
            if exp.uid in changed:
                uid = exp.uid
                exp.clear_data()
                exp.uid, exp.source_document = uid, docs[uid].payload
                exp.measurement = load_experimental_pole(docs[uid].source, exp.source_document)
        elif exp.uid is not None:
            exp.clear_data()
    if 'rsm' in data['assignments']:
        rsm = store.rsm
        experiment = rsm.experimental
        experiment.sources.update((uid, docs[uid]) for uid in data['assignments'].get(RSM, []) if uid in docs and docs[uid].kind in {SCAN, RSM_DATA})
        experiment.source_key = tuple((doc.uid, id(doc.payload)) for doc in experiment.selected_sources())
        if any(uid in changed or uid not in docs for uid in data['assignments'].get(RSM, []) if uid not in {rsm.calculated.uid}):
            experiment.source_key = None
            experiment.refresh_sources()
        if experiment.phase_uid not in docs or experiment.phase_uid in changed:
            rsm.invalidate_phase(experiment.phase_uid, removed=experiment.phase_uid not in docs)
        calculated = rsm.calculated
        if calculated.uid in docs:
            calculated.payload = docs[calculated.uid].payload
            if calculated.uid in changed:
                payload, uid = calculated.payload, calculated.uid
                calculated.payload = None
                calculated.bind(payload, uid)
        elif calculated.uid is not None:
            calculated.bind(None)
    if 'viewer' in data['assignments']:
        analysis = decode(data['analysis'])
        valid = set(store.assignments[VIEWER]) - changed
        peaks = [peak for peak in analysis['peaks'] if peak.scan_uid in valid]
        numbers = {peak.number for peak in peaks}
        for peak in peaks:
            if peak.axis_name not in docs[peak.scan_uid].payload.available_axes:
                raise SessionFormatError('A peak references an unavailable scan axis')
            if peak.parent_number not in numbers:
                peak = replace(peak, parent_number=None, kind='primary')
            peak = replace(peak, hkl_assignments=tuple(item for item in peak.hkl_assignments if item[0] in docs and item[0] not in changed))
            store.analysis.put_peak(peak)
        for uid, background in analysis['backgrounds']:
            if uid in valid:
                if background.axis_name not in docs[uid].payload.available_axes:
                    raise SessionFormatError('A background references an unavailable scan axis')
                store.analysis.set_background(uid, background)
        store.analysis.reserve_peak_numbers(analysis['next_peak_number'])


def _rebuild_poles(store, radiation):
    calc = store.poles.calculated
    wavelength = min(line[1] for line in radiation.lines())
    if calc.cif_document is not None and calc.initialized:
        calc.reflections = available_reflections(calc.crystal, *calc.d_range, wavelength)
        calc.reflection_radiations = tuple(radiation.lines())
    layer = calc.overlay_layer
    if layer is not None and layer.initialized:
        layer.reflections = available_reflections(layer.crystal, *calc.d_range, wavelength)


def _rsm_key(store, radiation):
    state = store.rsm.calculated
    if state.payload is not None and state.has_calculated:
        orientation = state.orientation
        state.request_key = (id(state.payload), radiation.lines()[0][1], orientation.surface, orientation.inplane,
            orientation.max_index, orientation.tolerance, state.target_mode, state.targets[state.target_mode],
            state.coordinates, *state.spans[state.coordinates])


def clone_store(original):
    result = ProjectStore()
    result.documents = original.documents.copy()
    result.assignments = {ws: ids.copy() for ws, ids in original.assignments.items()}
    result._source_keys = original._source_keys.copy()
    memo = {id(doc): doc for doc in original.documents.values()}
    memo.update((id(doc.payload), doc.payload) for doc in original.documents.values())
    for ws in WORKSPACES:
        setattr(result, ws, copy.deepcopy(getattr(original, ws), memo))
    for peak in original.analysis.peaks:
        result.analysis.put_peak(peak)
    for uid, background in original.analysis.backgrounds:
        result.analysis.set_background(uid, background)
    result.analysis.reserve_peak_numbers(original.analysis.next_peak_number)
    return result


def merge_tab(current, session):
    if session.scope not in WORKSPACES:
        raise SessionFormatError('Expected a single-tab file')
    scope, incoming = session.scope, session.store
    result = clone_store(current)
    for uid in tuple(result.assignments[scope]):
        result.assign(uid, scope, False)
    mapping = {}
    for uid, doc in incoming.documents.items():
        candidate = result.documents.get(uid)
        if candidate is None and doc.kind != CELL_PHASE:
            candidate = result.source_document(doc.kind, doc.source, _recipe(doc).get('index', 0))
        try:
            equal = (candidate is not None and candidate.kind == doc.kind and candidate.source.resolve() == doc.source.resolve()
                and candidate.source_digest == doc.source_digest and _recipe(candidate) == _recipe(doc))
        except SessionFormatError:
            equal = False
        if equal:
            mapping[uid] = candidate.uid
        else:
            new_uid = uuid.uuid4().hex if uid in result.documents else uid
            mapping[uid] = new_uid
            imported = replace(doc, uid=new_uid, history=[])
            result.restore_document(imported, source_index=_recipe(doc).get('index', 0) if doc.kind != CELL_PHASE and doc.uid in incoming._source_keys.values() else None)
    for uid, doc in incoming.documents.items():
        if mapping[uid] not in current.documents:
            result.documents[mapping[uid]].parent_uid = mapping.get(doc.parent_uid)
    ids = incoming.assignments[scope]
    for uid in incoming.documents:
        if uid in ids:
            result.assign(mapping[uid], scope, additive=scope == POLES)
    def rewrite(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {mapping.get(key, key): rewrite(item) for key, item in value.items()}
        return value
    # Encode settings to omit runtime payloads and caches, then remap references.
    manifest = snapshot(incoming, session.radiation, Path.cwd()/'unused.xrdtab', scope=scope)
    manifest['states'] = rewrite(manifest['states'])
    manifest['viewer_items'] = rewrite(manifest['viewer_items'])
    manifest['assignments'] = rewrite(manifest['assignments'])
    manifest['analysis'] = rewrite(manifest['analysis'])
    state = decode(manifest['states'])[scope]
    setattr(result, scope, state)
    _attach(result, manifest, set())
    _rebuild_poles(result, session.radiation)
    _rsm_key(result, session.radiation)
    return LoadedSession(result, session.radiation, scope, {}, session.warnings)
