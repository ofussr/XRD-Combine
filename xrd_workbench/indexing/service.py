"""Validated common API around the two experimental numerical methods."""

from dataclasses import asdict, replace
import math
import numpy as np

from . import indexing_visser as visser
from . import indexing_boultif_louer as dichotomy
from .models import Candidate, IndexingCancelled, IndexingError, IndexingRequest, IndexingResult
from .reduction import cell_from_metric, matrix_tuple, reduce_metric

METHODS = {"visser": visser, "boultif_louer": dichotomy}


def _angle(d, wavelength):
    if wavelength is None or wavelength / (2 * d) > 1:
        return None
    return math.degrees(2 * math.asin(wavelength / (2 * d)))


def _metric_system(direct):
    # Classification of this representation only, never a space-group inference.
    cell = cell_from_metric(direct)
    a, b, c, al, be, ga = cell.as_tuple()
    near = lambda x, y: math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-4)
    right = [near(angle, 90) for angle in (al, be, ga)]
    if all(right):
        if near(a, b) and near(b, c):
            return "cubic"
        if near(a, b) or near(a, c) or near(b, c):
            return "tetragonal"
        return "orthorhombic"
    if near(a, b) and near(al, 90) and near(be, 90) and near(ga, 120):
        return "hexagonal"
    if sum(right) == 2:
        return "monoclinic"
    return "triclinic"


def run_indexing(request: IndexingRequest, cancel=None) -> IndexingResult:
    def check():
        if cancel is not None and cancel.is_set():
            raise IndexingCancelled()
    check()
    if request.method not in METHODS:
        raise IndexingError("invalid_method")
    if request.input_kind not in {"two_theta", "d", "q"}:
        raise IndexingError("invalid_coordinate")
    wavelength = request.wavelength
    if (request.input_kind == "two_theta" and wavelength is None
            or wavelength is not None and (not math.isfinite(wavelength) or wavelength <= 0)):
        raise IndexingError("invalid_wavelength")
    enabled = tuple((i, p) for i, p in enumerate(request.peaks) if p.enabled)
    ids = [p.peak_id for _, p in enabled if p.peak_id is not None]
    if len(ids) != len(set(ids)):
        raise IndexingError("duplicate_peak_id")
    for i, peak in enabled:
        value = peak.position
        if not math.isfinite(value) or value <= 0 or request.input_kind == "two_theta" and value >= 180:
            raise IndexingError("invalid_position", peak=i + 1)
        if peak.uncertainty is not None and (not math.isfinite(peak.uncertainty) or peak.uncertainty <= 0):
            raise IndexingError("invalid_uncertainty", peak=i + 1)
    if request.method == "visser":
        settings = request.settings or visser.VisserSettings()
        if not isinstance(settings, visser.VisserSettings):
            raise IndexingError("invalid_settings", reason="VisserSettings required")
        required = settings.search_peak_count
    else:
        settings = request.settings or dichotomy.BoultifLouerSettings()
        if not isinstance(settings, dichotomy.BoultifLouerSettings):
            raise IndexingError("invalid_settings", reason="BoultifLouerSettings required")
        required = settings.min_required_peaks
    if len(enabled) < required:
        raise IndexingError("too_few_lines", required=required, actual=len(enabled))
    try:
        indexer = (visser.VisserIndexer if request.method == "visser"
                   else dichotomy.BoultifLouerIndexer)(settings, cancel_check=check)
        raw = indexer.index(tuple(p for _, p in enabled), input_kind=request.input_kind,
                            wavelength=wavelength)
    except ValueError as error:
        raise IndexingError("invalid_settings", reason=str(error)) from error
    candidates = []
    for candidate in raw.candidates:
        check()
        reciprocal = np.asarray(candidate.reciprocal_metric)
        direct = np.linalg.inv(reciprocal)
        reduced = reduce_metric(reciprocal)
        lines = []
        for line in candidate.indexed_lines:
            original_index, peak = enabled[line.observed_index]
            obs_angle = line.observed_two_theta
            if obs_angle is None:
                obs_angle = _angle(line.observed_d, wavelength)
            calc_angle = _angle(line.calculated_d, wavelength)
            lines.append(replace(line, observed_index=original_index, peak_id=peak.peak_id,
                                 observed_two_theta=obs_angle, calculated_two_theta=calc_angle,
                                 delta_two_theta=None if obs_angle is None or calc_angle is None
                                 else obs_angle - calc_angle))
        unindexed = tuple(replace(line, observed_index=enabled[line.observed_index][0],
                                 peak_id=enabled[line.observed_index][1].peak_id)
                          for line in candidate.unindexed_lines)
        candidates.append(Candidate(
            candidate.rank, getattr(candidate, "crystal_system", _metric_system(direct)),
            candidate.cell, matrix_tuple(direct), matrix_tuple(reciprocal), reduced,
            tuple(lines), unindexed, candidate.indexed_search_lines, candidate.m20,
            getattr(candidate, "f20", None), getattr(candidate, "n20", getattr(candidate, "n20_possible", None)),
            candidate.q20, getattr(candidate, "zero_shift_deg", 0.0),
            diagnostics={"rms_normalized_residual": candidate.rms_normalized_residual,
                         "mean_abs_delta_q": candidate.mean_abs_delta_q,
                         "spurious_search_lines": getattr(candidate, "spurious_search_lines", None)}))
    # Mark equivalent representations without concealing alternative candidates.
    candidates = [replace(c, equivalent_ranks=tuple(other.rank for other in candidates
                  if other.rank != c.rank and np.allclose(c.reduced.direct_metric,
                  other.reduced.direct_metric, rtol=1e-6, atol=1e-5))) for c in candidates]
    module = METHODS[request.method]
    references = ((module.VISSER_PUBLICATION, module.DE_WOLFF_PUBLICATION)
                  if request.method == "visser" else (*module.METHOD_PUBLICATIONS,
                  module.DE_WOLFF_PUBLICATION, module.SMITH_SNYDER_PUBLICATION))
    diagnostics = asdict(raw.diagnostics)
    diagnostics.update(settings=asdict(settings), experimental=True, exhaustive_search=False,
                       disabled_peak_ids=[p.peak_id for p in request.peaks if not p.enabled],
                       delta_convention="observed - calculated")
    return IndexingResult(raw.method_id, raw.method_name, references, request,
                          tuple(candidates), diagnostics)
