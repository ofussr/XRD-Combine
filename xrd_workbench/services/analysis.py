"""Qt-free operations on accepted measurement analysis results."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..models.analysis import CompanionProposal, MeasurementAnalysis, SessionPeak
from ..models.scan import Scan1D
from .background import estimate_background
from .peak_fitting import companion_two_theta, refit_voigt_peaks


def recalculate_background(analysis: MeasurementAnalysis, uid: str,
                           scan: Scan1D, spacing: float) -> None:
    old = analysis.background_for(uid, scan.axis_name)
    estimate = estimate_background(scan.x, scan.y, axis_name=scan.axis_name,
                                   spacing=spacing,
                                   excluded=old.excluded if old is not None else ())
    analysis.set_background(uid, replace(estimate, local_levels=old.local_levels)
                            if old is not None else estimate)


def refit_analysis(analysis: MeasurementAnalysis, uid: str, scan: Scan1D, *,
                   automatic: bool = False,
                   changed_numbers: set[int] | None = None) -> bool:
    """Fit nearby groups and commit only after every requested group succeeds."""
    peaks = analysis.peaks_for(uid, scan.axis_name)
    if not peaks:
        return False
    x = np.asarray(scan.x, dtype=float)
    y = np.asarray(scan.y, dtype=float)
    background = analysis.background_for(uid, scan.axis_name)
    if automatic and background is None:
        # Local baselines from separate previews are not a shared model.
        # Leave these accepted proposals untouched until Do Fit is invoked.
        return False
    if background is not None:
        base = background.values(x)
    else:
        # Retain the local baseline already accepted with each peak when
        # no measurement-wide background has been calculated yet.
        total = np.zeros_like(x)
        count = np.zeros_like(x)
        for peak in peaks:
            order = np.argsort(peak.source_x)
            local_x = np.asarray(peak.source_x)[order]
            inside = (x >= local_x[0]) & (x <= local_x[-1])
            total[inside] += np.interp(
                x[inside], local_x, np.asarray(peak.source_background)[order])
            count[inside] += 1
        if not np.any(count):
            return False
        anchors = np.flatnonzero(count)
        base = np.interp(x, x[anchors], total[anchors] / count[anchors])
    # Peaks with overlapping neighbourhoods share a fit. A new group
    # does not force every earlier peak in a distant part of the scan to
    # be optimised again.
    step = float(np.median(np.diff(x)))
    groups: list[tuple[float, float, list[SessionPeak]]] = []
    for peak in sorted(peaks, key=lambda p: p.source_center):
        radius = max(2.5 * peak.source_fwhm, .2, 15.0 * step)
        left, right = peak.source_center - radius, peak.source_center + radius
        if groups and left <= groups[-1][1]:
            previous_left, previous_right, members = groups[-1]
            members.append(peak)
            groups[-1] = previous_left, max(previous_right, right), members
        else:
            groups.append((left, right, [peak]))
    if changed_numbers is not None:
        groups = [group for group in groups
                  if any(peak.number in changed_numbers for peak in group[2])]
    if not groups:
        return False

    from scipy.special import voigt_profile

    def component(values: np.ndarray, peak: SessionPeak) -> np.ndarray:
        if peak.source_sigma > 0 and peak.source_gamma > 0:
            return peak.source_height * voigt_profile(
                values - peak.source_center, peak.source_sigma, peak.source_gamma,
            ) / voigt_profile(0.0, peak.source_sigma, peak.source_gamma)
        order = np.argsort(peak.source_x)
        local_x = np.asarray(peak.source_x)[order]
        return np.interp(values, local_x,
                         np.asarray(peak.source_profile)[order], left=0.0, right=0.0)

    updated: dict[int, SessionPeak] = {}
    working = {peak.number: peak for peak in peaks}
    for _left, _right, group in groups:
        group_ids = {peak.number for peak in group}
        other = sum((component(x, peak) for peak in working.values()
                     if peak.number not in group_ids), np.zeros_like(x))
        seeds = []
        for peak in group:
            width = max(float(peak.source_fwhm), step)
            seeds.append((peak.source_center, peak.source_height,
                          peak.source_sigma or width / 3.0,
                          peak.source_gamma or width / 6.0))
        result = refit_voigt_peaks(
            x, y - other, base, seeds,
            refine_background=background is not None)
        for peak, fit in zip(group, result.peaks):
            replacement = replace(
                peak, source_center=fit.center, source_x=result.x,
                source_background=result.background, source_profile=fit.profile,
                source_height=fit.height, source_area=fit.area,
                source_fwhm=fit.fwhm, source_sigma=fit.sigma,
                source_gamma=fit.gamma,
            )
            updated[peak.number] = replacement
            working[peak.number] = replacement
        if background is not None and result.background_delta is not None:
            background = background.with_fit_correction(result.x, result.background_delta)
            base = background.values(x)
    analysis.commit_fit(updated.values(), scan_uid=uid, axis_name=scan.axis_name,
                        background=background)
    return True


def find_companion_pairs(table_peaks: tuple[SessionPeak, ...], kind: str, *,
                         primary_wavelength: float, wavelength: float,
                         x_scale: float = 1.0, x_shift: float = 0.0) -> list[CompanionProposal]:
    """Suggest non-conflicting positional matches between already accepted peaks."""
    primary = [peak for peak in table_peaks if peak.kind == "primary"]
    linked_parents = {peak.parent_number for peak in table_peaks if peak.kind == kind}
    possible: list[tuple[float, CompanionProposal]] = []
    for peak in primary:
        if peak.number in linked_parents:
            continue
        center = peak.source_center * x_scale + x_shift
        predicted = companion_two_theta(center, primary_wavelength, wavelength)
        if not np.isfinite(predicted):
            continue
        distance = abs(predicted - center)
        for candidate in primary:
            if candidate.number == peak.number:
                continue
            measured = candidate.source_center * x_scale + x_shift
            width = max(peak.source_fwhm, candidate.source_fwhm) * abs(x_scale)
            tolerance = min(max(0.02, 0.35 * width), 0.45 * distance)
            error = abs(measured - predicted)
            if error <= tolerance:
                possible.append((error / tolerance, CompanionProposal(
                    peak.number, center, predicted, measured, candidate.number,
                )))
    # The best positional match wins; a table row participates in at most
    # one new pair per search. No spectra are fitted or added here.
    possible.sort(key=lambda entry: entry[0])
    used: set[int] = set()
    proposals: list[CompanionProposal] = []
    for _score, proposal in possible:
        if proposal.parent_number in used or proposal.existing_number in used:
            continue
        proposals.append(proposal)
        used.update((proposal.parent_number, proposal.existing_number))
    return proposals
