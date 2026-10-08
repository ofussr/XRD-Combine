"""Versioned JSON settings codec with an explicit model/field allowlist.

Only accepted settings and analysis results are registered. Source payloads,
scientific caches, callbacks, widgets and Python import names are never encoded.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
import re

import numpy as np

from ..models.analysis import SessionPeak
from ..models.background import BackgroundAnchors
from ..models.phase_scan import AngleRange, PhaseOrientation, PhaseScanState
from ..models.pole_figure import CalculatedPoleLayer
from ..models.pole_state import CalculatedPoleState, ExperimentalPoleState, PolesState
from ..models.radiation import RadiationSettings
from ..models.rsm_state import RSMOrientation, CalculatedRSMState, ExperimentalRSMState, RSMState
from ..models.structures_state import StructureCamera, StructureViewState, StructureCalculationState, StructuresState
from ..models.viewer import PlotAppearance, ViewerViewport, ViewerPlotState, ViewerState


class SessionFormatError(ValueError):
    pass


_EXCLUDED = {
    ViewerState: {'items', 'colours', '_payloads'},
    ViewerPlotState: {'navigation_x_bounds', 'navigation_y_bounds', 'overlay_phase_top'},
    StructureViewState: {'payload'},
    StructureCalculationState: {'rows'},
    StructuresState: {'payload'},
    CalculatedPoleState: {'cif_document', 'crystal', 'reflections', 'points', 'point_groups',
        'intensity_by_spacing', 'reflection_radiations'},
    CalculatedPoleLayer: {'document', 'reflections', 'points', 'point_groups', 'intensity_by_spacing'},
    ExperimentalPoleState: {'source_document', 'measurement', 'raw_path', 'scans', 'loaded_radii'},
    ExperimentalRSMState: {'sources', 'source_key', 'data'},
    CalculatedRSMState: {'payload', 'request_key'},
}
_CLASSES = (ViewerState, ViewerPlotState, ViewerViewport, PlotAppearance, PhaseScanState, AngleRange,
    PhaseOrientation, StructureCamera, StructureViewState, StructureCalculationState,
    StructuresState, CalculatedPoleState, CalculatedPoleLayer, ExperimentalPoleState,
    PolesState, RSMOrientation, CalculatedRSMState, ExperimentalRSMState, RSMState,
    RadiationSettings, SessionPeak, BackgroundAnchors)
_REGISTRY = {cls.__name__: cls for cls in _CLASSES}
_FIELDS = {cls: {f.name for f in fields(cls)} - _EXCLUDED.get(cls, set()) for cls in _CLASSES}


def encode(value):
    if is_dataclass(value):
        cls = type(value)
        if cls not in _FIELDS:
            raise SessionFormatError(f'Unsupported saved model: {cls.__name__}')
        return {'model': cls.__name__, 'fields': {f.name: encode((value.d_range[0], None)
            if cls is CalculatedPoleState and f.name == 'd_range' and value.d_range[1] == math.inf else getattr(value, f.name))
            for f in fields(value) if f.name in _FIELDS[cls]}}
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise SessionFormatError('Saved arrays must be finite')
        return {'array': value.tolist()}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, tuple):
        return {'tuple': [encode(item) for item in value]}
    if isinstance(value, set):
        return {'set': [encode(item) for item in sorted(value)]}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: encode(item) for key, item in value.items()}
        return {'mapping': [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SessionFormatError(f'Unsupported saved value: {type(value).__name__}')


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        if value is None or isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value)):
            return value
        raise SessionFormatError('Saved values must be finite JSON data')
    if set(value) == {'model', 'fields'}:
        cls = _REGISTRY.get(value['model']) if isinstance(value['model'], str) else None
        supplied = value['fields']
        if cls is None or not isinstance(supplied, dict) or set(supplied) - _FIELDS[cls]:
            raise SessionFormatError('Unknown saved model or field')
        values = {key: decode(item) for key, item in supplied.items()}
        if cls is CalculatedPoleState and isinstance(values.get('d_range'), tuple) and len(values['d_range']) == 2 and values['d_range'][1] is None:
            values['d_range'] = (values['d_range'][0], math.inf)
        if cls is CalculatedPoleLayer:
            values['document'] = None
        try:
            result = cls(**values)
            _validate(result)
            return result
        except (ValueError, TypeError, IndexError, AttributeError, KeyError) as exc:
            raise SessionFormatError(f'Invalid {cls.__name__}: {exc}') from exc
    if set(value) == {'array'}:
        try:
            array = np.asarray(value['array'], dtype=float)
        except (ValueError, TypeError) as exc:
            raise SessionFormatError('Invalid numeric array') from exc
        if not np.all(np.isfinite(array)):
            raise SessionFormatError('Saved arrays must be finite')
        return array
    if set(value) in ({'tuple'}, {'set'}):
        key = next(iter(value))
        if not isinstance(value[key], list):
            raise SessionFormatError('Invalid saved collection')
        items = [decode(item) for item in value[key]]
        try:
            return tuple(items) if key == 'tuple' else set(items)
        except TypeError as exc:
            raise SessionFormatError('Invalid saved set') from exc
    if set(value) == {'mapping'}:
        result = {}
        try:
            for key, item in value['mapping']:
                key = decode(key)
                if key in result:
                    raise SessionFormatError('Duplicate saved mapping key')
                result[key] = decode(item)
            return result
        except (TypeError, ValueError) as exc:
            raise SessionFormatError('Invalid saved mapping') from exc
    return {key: decode(item) for key, item in value.items()}


def expect(value, cls):
    if type(value) is not cls:
        raise SessionFormatError(f'Expected {cls.__name__}')
    return value


def _validate(value):
    from types import UnionType
    from typing import Any, get_args, get_origin, get_type_hints, Union
    def matches(item, hint):
        origin, args = get_origin(hint), get_args(hint)
        if hint is Any:
            return True
        if origin in (Union, UnionType):
            return any(matches(item, option) for option in args)
        if origin in (tuple, list, set):
            if not isinstance(item, origin):
                return False
            if origin is tuple and args and args[-1] is not Ellipsis:
                return len(item) == len(args) and all(matches(v, h) for v, h in zip(item, args))
            return not args or all(matches(v, args[0]) for v in item)
        if origin is dict:
            return isinstance(item, dict) and (not args or all(matches(k, args[0]) and matches(v, args[1]) for k, v in item.items()))
        if hint is float:
            return type(item) in (float, int) and math.isfinite(item)
        if hint in (int, bool, str, type(None)):
            return type(item) is hint
        return isinstance(item, hint)
    for name, hint in get_type_hints(type(value)).items():
        if name == 'd_range' and isinstance(value, CalculatedPoleState):
            pair = value.d_range
            if not isinstance(pair, tuple) or len(pair) != 2 or any(type(v) not in (int, float) for v in pair) or not math.isfinite(pair[0]) or not (math.isfinite(pair[1]) or pair[1] == math.inf):
                raise SessionFormatError('Invalid d-spacing range')
            continue
        if name in _FIELDS[type(value)] and not matches(getattr(value, name), hint):
            raise SessionFormatError(f'Wrong type for {name}')
    for name in ('base_rotation', 'user_rotation'):
        if hasattr(value, name):
            matrix = getattr(value, name)
            if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
                raise SessionFormatError('Invalid orientation matrix')
    if hasattr(value, 'center_hkl') and not any(value.center_hkl):
        raise SessionFormatError('A centre reflection must have nonzero hkl')
    for name in ('colour', 'primary_colour', 'overlay_colour', 'fill_colour'):
        if hasattr(value, name) and not re.fullmatch(r'#[0-9a-fA-F]{6}', getattr(value, name)):
            raise SessionFormatError('Invalid colour')
    if isinstance(value, RadiationSettings):
        value.lines()
    if isinstance(value, PhaseOrientation):
        value.validate()
    if isinstance(value, StructureCamera) and not (.25 <= value.zoom <= 4.5):
        raise SessionFormatError('Invalid structure zoom')
    if isinstance(value, StructureViewState):
        if not 25 <= value.atom_scale_percent <= 200 or not 4 <= value.hatch_density <= 42 or not 0 <= value.hatch_grip <= 200:
            raise SessionFormatError('Invalid structure display setting')
        for mapping in (value.atom_component_colours, value.polyhedron_site_colours):
            if any(not re.fullmatch(r'#[0-9a-fA-F]{6}', item) for item in mapping.values()):
                raise SessionFormatError('Invalid site colour')
        if value.view_name not in {'hkl', 'standard', 'a', 'b', 'c', 'a*', 'b*', 'c*'} or not any(value.center_hkl):
            raise SessionFormatError('Invalid structure orientation')
    if isinstance(value, RSMOrientation):
        PhaseOrientation(value.surface, value.inplane).validate()
        if not 1 <= value.max_index <= 20 or value.tolerance < 0:
            raise SessionFormatError('Invalid RSM orientation limits')
    enums = {'intensity_scale': {'linear', 'log', 'sqrt', 'square'}, 'scale_mode': {'linear', 'log', 'square'},
        'phase_layout': {'overlay', 'separate'}, 'phase_style': {'sticks', 'profile'},
        'x_display_mode': {'angle', 'd'}, 'result_mode': {'add', 'replace'},
        'coordinates': {'angular', 'q'}, 'extent': {'real', 'full'}, 'scale': {'linear', 'log'},
        'target_mode': {'hkl', 'angles', 'q'}, 'range_mode': {'d', 'two_theta'},
        'projection': {'stereographic', 'equal_area'}, 'style': {'sticks', 'profile'},
        'display_mode': {'colour', 'solid'}, 'color_mode': {'uniform', 'd', 'intensity', 'phase'},
        'colour_map': {'viridis', 'turbo', 'plasma', 'inferno', 'magma'}, 'derived_angle': {None, 'first', 'step', 'last'}}
    for name, choices in enums.items():
        if hasattr(value, name) and getattr(value, name) not in choices:
            raise SessionFormatError(f'Invalid {name}')
    if isinstance(value, PhaseScanState):
        if value.mode not in {'auto', 'powder', 'phi', 'chi', 'omega', 'coupled', 'detector'} or not 1 <= value.max_index <= 20:
            raise SessionFormatError('Invalid phase scan mode or index limit')
        for angles in value.angles.values():
            for name, bounds in angles.items():
                value.validate_angle(name, bounds)
    if isinstance(value, ViewerState) and (value.background_spacing <= 0 or value._colour_index < 0):
        raise SessionFormatError('Invalid Viewer parameters')
    if isinstance(value, ViewerPlotState):
        if value.display_wavelength <= 0 or value.profile_fwhm <= 0 or not 10 <= value.phase_height_percent <= 85 or not 1 <= value.overlay_height_percent <= 100:
            raise SessionFormatError('Invalid Viewer plot parameters')
        for low, high in (value.manual_limits[:2], value.manual_limits[2:]):
            if low is not None and high is not None and low >= high:
                raise SessionFormatError('Invalid manual plot limits')
    if isinstance(value, PlotAppearance) and (not .25 <= value.line_width <= 8 or not 0 <= value.grid_alpha <= .4):
        raise SessionFormatError('Invalid plot formatting')
    if isinstance(value, (StructuresState, PolesState, RSMState)) and not 0 <= value.active_tab <= (2 if isinstance(value, StructuresState) else 1):
        raise SessionFormatError('Invalid active workspace tab')
    if isinstance(value, (ExperimentalRSMState, CalculatedRSMState)):
        if set(value.viewports) - {'angular', 'q'}:
            raise SessionFormatError('Invalid RSM viewport coordinates')
        for viewport in value.viewports.values():
            _viewport(viewport)
    if isinstance(value, CalculatedRSMState):
        if set(value.targets) != {'hkl', 'angles', 'q'} or set(value.spans) != {'angular', 'q'}:
            raise SessionFormatError('Incomplete RSM calculation parameters')
        for mode, target in value.targets.items():
            if not isinstance(target, tuple) or len(target) != (3 if mode == 'hkl' else 2) or any(type(v) not in ((int,) if mode == 'hkl' else (int, float)) or not math.isfinite(v) for v in target):
                raise SessionFormatError('Invalid RSM target')
        for span in value.spans.values():
            if not isinstance(span, tuple) or len(span) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in span):
                raise SessionFormatError('Invalid RSM display span')
    if isinstance(value, (StructureCalculationState, CalculatedPoleState, ExperimentalPoleState)):
        for name in ('viewport', 'pyqtgraph_view', 'matplotlib_view'):
            viewport = getattr(value, name, None)
            if viewport is not None and not (isinstance(value, ExperimentalPoleState) and name == 'matplotlib_view'):
                _viewport(viewport)
    if isinstance(value, ExperimentalPoleState) and (value.zoom_factor <= 0 or (value.intensity_limits is not None and value.intensity_limits[0] > value.intensity_limits[1])):
        raise SessionFormatError('Invalid experimental pole limits')
    if isinstance(value, CalculatedPoleState):
        if not 1 <= value.max_index <= 20 or value.wavelength <= 0 or not 0 < value.d_range[0] <= value.d_range[1] or value.selected_layer_index not in (0, 1):
            raise SessionFormatError('Invalid calculated pole parameters')
        if value.two_theta_range is not None and (not 0 <= value.two_theta_range[0] <= value.two_theta_range[1] <= 180 or value.two_theta_range[1] == 0):
            raise SessionFormatError('Invalid pole angular range')
    if isinstance(value, StructureCalculationState):
        if not 0 <= value.minimum < value.maximum <= 180 or value.threshold < 0 or value.fwhm <= 0 or not -1 <= value.sort_column < 10 or len(value.column_widths) != 10 or min(value.column_widths) < 1:
            raise SessionFormatError('Invalid structure calculation settings')
    if isinstance(value, SessionPeak):
        if value.number < 1 or value.kind not in {'primary', 'ka2', 'kb'} or min(value.source_height, value.source_area, value.source_fwhm, value.source_sigma, value.source_gamma) < 0:
            raise SessionFormatError('Invalid peak result')
        _arrays(value.source_x, value.source_background, value.source_profile)
    if isinstance(value, BackgroundAnchors):
        _arrays(value.source_x, value.source_y)
        if value.spacing <= 0 or np.any(np.diff(value.source_x) <= 0):
            raise SessionFormatError('Invalid background anchors')
        if (value.fit_correction_x is None) != (value.fit_correction_y is None):
            raise SessionFormatError('Incomplete background correction')
        if value.fit_correction_x is not None:
            _arrays(value.fit_correction_x, value.fit_correction_y)


def _arrays(*arrays):
    if any(item.ndim != 1 or item.size < 2 or item.shape != arrays[0].shape for item in arrays):
        raise SessionFormatError('Analysis arrays must be one-dimensional and equally sized')


def _viewport(value):
    if not isinstance(value, tuple) or len(value) != 2 or any(not isinstance(pair, tuple) or len(pair) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) for v in pair) or pair[0] >= pair[1] for pair in value):
        raise SessionFormatError('Viewport limits must be finite and increasing')
