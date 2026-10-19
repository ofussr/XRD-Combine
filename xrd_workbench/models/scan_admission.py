"""Admission of measured angular paths based only on recorded metadata."""
import numpy as np
from .phase_scan import angular_axis, measurement_geometry
from .data_errors import XRDDataError


def scan_assignment_error(scan, workspace):
    if workspace not in {'poles', 'rsm'}:
        return None
    axis = angular_axis(scan.axis_name)
    if workspace == 'rsm' and (axis == 'two_theta' or
            (scan.axis_name == 'X' and scan.metadata.get('format') == 'XY')):
        return None  # Existing manual series workflow.
    allowed = {'phi', 'chi'} if workspace == 'poles' else {'omega'}
    if axis not in allowed:
        return XRDDataError('unsupported_axis', axis=scan.axis_name)
    geometry = measurement_geometry(scan)
    required = {'phi': ('two_theta', 'omega', 'chi'),
                'chi': ('two_theta', 'omega', 'phi'),
                'omega': ('two_theta',)}[axis]
    missing = [name for name in required if name not in geometry]
    if missing:
        return XRDDataError('missing_fixed', angles=', '.join(missing))
    if not 0 < geometry['two_theta'] < 180:
        return XRDDataError('invalid_detector')
    if not (np.isfinite(scan.x).all() and np.isfinite(scan.y).all()):
        return XRDDataError('invalid_scan')
    return None
