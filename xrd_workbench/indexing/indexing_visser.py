"""Independent Visser zone-indexing engine for powder diffraction.

This module implements a modern, self-contained interpretation of the
zone-indexing strategy described by J. W. Visser.  It does not execute, wrap,
translate, or depend on the historical ITO program.

Primary publication
-------------------
J. W. Visser, "A fully automatic program for finding the unit cell from
powder data", Journal of Applied Crystallography 2 (1969) 89-95.
DOI: 10.1107/S0021889869006649

The historical algorithm searches potential reciprocal-lattice zones from the
20 lowest-angle powder lines, combines the best zones into trial reciprocal
lattices, refines candidate metrics, and ranks the resulting cells.  The public
API below follows that workflow while using modern data structures and numerical
linear algebra.

The de Wolff M20 figure of merit is reported for candidates when at least 20
lines are indexed:
P. M. de Wolff, Journal of Applied Crystallography 1 (1968) 108-113.
DOI: 10.1107/S002188986800508X

Notes
-----
* The method needs at least 20 peak positions.  By default only the first 40
  input lines are retained and the first 20 are used to discover trial cells.
* Intensities are accepted as metadata but are not used by the Visser search.
* No space group is inferred.  The output is a metric unit cell plus hkl
  assignments.  Systematic absences and space-group choice belong to a later
  validation step.
* Equivalent primitive cells can be expressed by different basis choices.
  Candidate deduplication uses shared Niggli-reduced metrics, with the original
  candidate basis preserved. Zone discovery retains its bounded heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Sequence
import itertools
import math

import numpy as np

from .models import Publication, UnitCell, IndexedLine, UnindexedLine, Peak as VisserPeak
from .reduction import metric_signature


# ---------------------------------------------------------------------------
# Static bibliographic metadata.  The GUI can import these values directly.
# ---------------------------------------------------------------------------

METHOD_ID = "visser"
METHOD_NAME = "Visser"
METHOD_DESCRIPTION = "Visser zone indexing"

REFERENCE_AUTHOR = "J. W. Visser"
REFERENCE_TITLE = "A fully automatic program for finding the unit cell from powder data"
REFERENCE_JOURNAL = "Journal of Applied Crystallography"
REFERENCE_YEAR = 1969
REFERENCE_VOLUME = 2
REFERENCE_PAGES = "89-95"
REFERENCE_DOI = "10.1107/S0021889869006649"
REFERENCE_URL = f"https://doi.org/{REFERENCE_DOI}"
REFERENCE_CITATION = (
    f"{REFERENCE_AUTHOR}, \"{REFERENCE_TITLE}\", "
    f"{REFERENCE_JOURNAL} {REFERENCE_VOLUME} ({REFERENCE_YEAR}) {REFERENCE_PAGES}. "
    f"DOI: {REFERENCE_DOI}"
)
REFERENCE_BIBTEX = """@article{Visser1969,
  author  = {Visser, J. W.},
  title   = {A fully automatic program for finding the unit cell from powder data},
  journal = {Journal of Applied Crystallography},
  year    = {1969},
  volume  = {2},
  pages   = {89--95},
  doi     = {10.1107/S0021889869006649}
}"""

M20_REFERENCE_AUTHOR = "P. M. de Wolff"
M20_REFERENCE_TITLE = "A simplified criterion for the reliability of a powder pattern indexing"
M20_REFERENCE_JOURNAL = "Journal of Applied Crystallography"
M20_REFERENCE_YEAR = 1968
M20_REFERENCE_VOLUME = 1
M20_REFERENCE_PAGES = "108-113"
M20_REFERENCE_DOI = "10.1107/S002188986800508X"


InputKind = Literal["two_theta", "d", "q"]




VISSER_PUBLICATION = Publication(
    REFERENCE_AUTHOR,
    REFERENCE_TITLE,
    REFERENCE_JOURNAL,
    REFERENCE_YEAR,
    REFERENCE_VOLUME,
    REFERENCE_PAGES,
    REFERENCE_DOI,
)

DE_WOLFF_PUBLICATION = Publication(
    M20_REFERENCE_AUTHOR,
    M20_REFERENCE_TITLE,
    M20_REFERENCE_JOURNAL,
    M20_REFERENCE_YEAR,
    M20_REFERENCE_VOLUME,
    M20_REFERENCE_PAGES,
    M20_REFERENCE_DOI,
)




@dataclass(frozen=True)
class VisserSettings:
    """Numerical controls for the independent Visser implementation."""

    search_peak_count: int = 20
    max_input_peaks: int = 40
    best_zone_count: int = 6
    max_candidates: int = 4

    # Peak-matching tolerances used when no per-peak uncertainty is supplied.
    two_theta_tolerance_deg: float = 0.03
    q_absolute_tolerance: float = 2.0e-5  # A^-2
    q_relative_tolerance: float = 2.0e-3

    # Zone discovery.
    zone_seed_peak_count: int = 8
    zone_seed_index_max: int = 3
    zone_prediction_index_max: int = 5
    zone_min_matches: int = 5

    # Trial-cell search and refinement.
    trial_hkl_max: int = 6
    trial_keep_before_refine: int = 120
    refinement_iterations: int = 3
    min_indexed_search_lines: int = 15

    # Broad physical guards; these are not symmetry constraints.
    min_cell_length: float = 0.5
    max_cell_length: float = 200.0
    min_cell_angle_deg: float = 10.0
    max_cell_angle_deg: float = 170.0

    def validate(self) -> None:
        if self.search_peak_count < 20:
            raise ValueError("Visser zone indexing requires at least the first 20 lines")
        if self.max_input_peaks < self.search_peak_count:
            raise ValueError("max_input_peaks must be >= search_peak_count")
        if self.best_zone_count < 2:
            raise ValueError("best_zone_count must be at least 2")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.trial_hkl_max < 2:
            raise ValueError("trial_hkl_max must be at least 2")
        if self.refinement_iterations < 0:
            raise ValueError("refinement_iterations cannot be negative")
        if self.two_theta_tolerance_deg <= 0:
            raise ValueError("two_theta_tolerance_deg must be positive")
        if self.q_absolute_tolerance <= 0 or self.q_relative_tolerance <= 0:
            raise ValueError("Q tolerances must be positive")








@dataclass(frozen=True)
class VisserCandidate:
    rank: int
    cell: UnitCell
    reciprocal_metric: tuple[tuple[float, float, float], ...]
    indexed_lines: tuple[IndexedLine, ...]
    unindexed_lines: tuple[UnindexedLine, ...]
    indexed_search_lines: int
    indexed_total_lines: int
    mean_abs_delta_q: float
    rms_normalized_residual: float
    m20: float | None
    n20: int | None
    q20: float | None

    @property
    def volume(self) -> float:
        return self.cell.volume


@dataclass(frozen=True)
class VisserDiagnostics:
    input_line_count: int
    retained_line_count: int
    search_line_count: int
    zones_found: int
    raw_trial_cells: int
    refined_trial_cells: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisserResult:
    method_id: str
    method_name: str
    publication: Publication
    m20_publication: Publication
    input_kind: InputKind
    wavelength: float | None
    candidates: tuple[VisserCandidate, ...]
    diagnostics: VisserDiagnostics

    @property
    def best(self) -> VisserCandidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True)
class _PreparedPeak:
    original_index: int
    original_position: float
    d: float
    q: float
    q_tolerance: float
    intensity: float | None
    label: str | None


@dataclass(frozen=True)
class _Zone:
    metric: np.ndarray = field(compare=False, repr=False)
    matches: int
    rms: float
    signature: tuple[float, ...]


@dataclass(frozen=True)
class _Trial:
    metric: np.ndarray = field(compare=False, repr=False)
    search_matches: int
    rms: float


class VisserIndexer:
    """Independent implementation of the Visser zone-indexing workflow."""

    METHOD_ID: ClassVar[str] = METHOD_ID
    METHOD_NAME: ClassVar[str] = METHOD_NAME
    METHOD_DESCRIPTION: ClassVar[str] = METHOD_DESCRIPTION
    PUBLICATION: ClassVar[Publication] = VISSER_PUBLICATION
    M20_PUBLICATION: ClassVar[Publication] = DE_WOLFF_PUBLICATION
    REFERENCE_DOI: ClassVar[str] = REFERENCE_DOI
    REFERENCE_URL: ClassVar[str] = REFERENCE_URL
    REFERENCE_CITATION: ClassVar[str] = REFERENCE_CITATION
    REFERENCE_BIBTEX: ClassVar[str] = REFERENCE_BIBTEX

    def __init__(self, settings: VisserSettings | None = None, *, cancel_check=None) -> None:
        self._cancel_check = cancel_check or (lambda: None)
        self.settings = settings or VisserSettings()
        self.settings.validate()
        self._vector_cache: dict[int, np.ndarray] = {}

    def index(
        self,
        peaks: Sequence[VisserPeak],
        *,
        input_kind: InputKind = "two_theta",
        wavelength: float | None = None,
    ) -> VisserResult:
        prepared = self._prepare_peaks(peaks, input_kind=input_kind, wavelength=wavelength)
        s = self.settings
        warnings: list[str] = []
        input_count = len(prepared)
        if input_count < s.search_peak_count:
            raise ValueError(
                f"Visser indexing needs at least {s.search_peak_count} valid lines; "
                f"received {input_count}"
            )
        retained = prepared[: s.max_input_peaks]
        if input_count > s.max_input_peaks:
            warnings.append(
                f"Only the first {s.max_input_peaks} lines were retained for this run."
            )

        search = retained[: s.search_peak_count]
        q_search = np.asarray([peak.q for peak in search], dtype=float)
        tol_search = np.asarray([peak.q_tolerance for peak in search], dtype=float)

        zones = self._find_zones(q_search, tol_search)
        raw_trials = self._combine_zones(zones, q_search, tol_search)
        refined_trials = self._refine_trials(raw_trials, q_search, tol_search)
        candidates = self._build_candidates(refined_trials, retained, search_count=len(search))

        diagnostics = VisserDiagnostics(
            input_line_count=input_count,
            retained_line_count=len(retained),
            search_line_count=len(search),
            zones_found=len(zones),
            raw_trial_cells=len(raw_trials),
            refined_trial_cells=len(refined_trials),
            warnings=tuple(warnings),
        )
        return VisserResult(
            method_id=self.METHOD_ID,
            method_name=self.METHOD_NAME,
            publication=self.PUBLICATION,
            m20_publication=self.M20_PUBLICATION,
            input_kind=input_kind,
            wavelength=wavelength,
            candidates=tuple(candidates),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    def _prepare_peaks(
        self,
        peaks: Sequence[VisserPeak],
        *,
        input_kind: InputKind,
        wavelength: float | None,
    ) -> list[_PreparedPeak]:
        if input_kind not in {"two_theta", "d", "q"}:
            raise ValueError("input_kind must be 'two_theta', 'd', or 'q'")
        if input_kind == "two_theta":
            if wavelength is None or not math.isfinite(wavelength) or wavelength <= 0:
                raise ValueError("A positive wavelength is required for two_theta input")
        elif wavelength is not None and (not math.isfinite(wavelength) or wavelength <= 0):
            raise ValueError("wavelength must be positive when supplied")

        prepared: list[_PreparedPeak] = []
        for index, peak in enumerate(peaks):
            self._cancel_check()
            if not peak.enabled:
                continue
            value = float(peak.position)
            if not math.isfinite(value):
                continue
            explicit_uncertainty = peak.uncertainty is not None
            uncertainty = peak.uncertainty
            if uncertainty is not None:
                uncertainty = float(uncertainty)
                if not math.isfinite(uncertainty) or uncertainty <= 0:
                    raise ValueError("Peak uncertainty must be positive and finite")

            if input_kind == "two_theta":
                if not 0.0 < value < 180.0:
                    continue
                d = self._two_theta_to_d(value, float(wavelength))
                q = 1.0 / (d * d)
                if uncertainty is None:
                    uncertainty = self.settings.two_theta_tolerance_deg
                low = max(1e-9, value - uncertainty)
                high = min(179.999999, value + uncertainty)
                q_low = 1.0 / self._two_theta_to_d(low, float(wavelength)) ** 2
                q_high = 1.0 / self._two_theta_to_d(high, float(wavelength)) ** 2
                q_tol = max(abs(q - q_low), abs(q_high - q))
            elif input_kind == "d":
                if value <= 0:
                    continue
                d = value
                q = 1.0 / (d * d)
                if uncertainty is None:
                    q_tol = max(
                        self.settings.q_absolute_tolerance,
                        self.settings.q_relative_tolerance * q,
                    )
                else:
                    d_low = max(1e-12, d - uncertainty)
                    d_high = d + uncertainty
                    q_tol = max(abs(q - 1.0 / d_low**2), abs(q - 1.0 / d_high**2))
            else:
                if value <= 0:
                    continue
                q = value
                d = 1.0 / math.sqrt(q)
                q_tol = (
                    uncertainty
                    if uncertainty is not None
                    else max(
                        self.settings.q_absolute_tolerance,
                        self.settings.q_relative_tolerance * q,
                    )
                )

            q_tol = float(q_tol)
            if not explicit_uncertainty:
                q_tol = max(q_tol, self.settings.q_absolute_tolerance,
                            self.settings.q_relative_tolerance * q)
            prepared.append(
                _PreparedPeak(
                    original_index=index,
                    original_position=value,
                    d=d,
                    q=q,
                    q_tolerance=q_tol,
                    intensity=None if peak.intensity is None else float(peak.intensity),
                    label=peak.label,
                )
            )

        # Visser consumes lines from lowest Bragg angle, equivalently lowest Q.
        prepared.sort(key=lambda peak: peak.q)
        return prepared

    @staticmethod
    def _two_theta_to_d(two_theta: float, wavelength: float) -> float:
        sine = math.sin(math.radians(two_theta / 2.0))
        if sine <= 0:
            raise ValueError("Non-physical two_theta value")
        return wavelength / (2.0 * sine)

    # ------------------------------------------------------------------
    # Zone discovery
    # ------------------------------------------------------------------

    def _find_zones(self, qobs: np.ndarray, tolerances: np.ndarray) -> list[_Zone]:
        s = self.settings
        n_seed = min(s.zone_seed_peak_count, len(qobs))
        index_pairs = [
            (h, k)
            for h in range(-s.zone_seed_index_max, s.zone_seed_index_max + 1)
            for k in range(-s.zone_seed_index_max, s.zone_seed_index_max + 1)
            if h != 0 and k != 0 and math.gcd(abs(h), abs(k)) == 1
        ]
        best: dict[tuple[float, ...], _Zone] = {}

        # A zone has Q(h,k) = A h^2 + B k^2 + 2 D h k.
        # Two observed lines seed A and B; a third line plus a small integer
        # (h,k) pair determines D.  The trial zone is then judged against all
        # search lines, which is the deductive core of zone indexing.
        for ia in range(n_seed):
            self._cancel_check()
            A = qobs[ia]
            for ib in range(ia + 1, n_seed):
                self._cancel_check()
                B = qobs[ib]
                for ic in range(n_seed):
                    self._cancel_check()
                    if ic in {ia, ib}:
                        continue
                    q_c = qobs[ic]
                    for h, k in index_pairs:
                        self._cancel_check()
                        D = (q_c - h * h * A - k * k * B) / (2.0 * h * k)
                        if D * D >= 0.99 * A * B:
                            continue
                        metric = np.asarray(((A, D), (D, B)), dtype=float)
                        if float(np.linalg.det(metric)) <= 1e-14:
                            continue
                        groups = self._predicted_2d(metric, qobs[-1] + tolerances[-1])
                        matches, rms = self._score_groups(groups, qobs, tolerances)
                        if matches < s.zone_min_matches:
                            continue
                        signature = self._line_signature(groups, qobs[-1], count=12)
                        zone = _Zone(metric, matches, rms, signature)
                        previous = best.get(signature)
                        if previous is None or (-matches, rms) < (-previous.matches, previous.rms):
                            best[signature] = zone

        return sorted(best.values(), key=lambda z: (-z.matches, z.rms))[: s.best_zone_count]

    def _predicted_2d(
        self, metric: np.ndarray, qmax: float
    ) -> list[tuple[float, tuple[tuple[int, int], ...]]]:
        m = self.settings.zone_prediction_index_max
        vectors = np.asarray(
            [
                (h, k)
                for h in range(-m, m + 1)
                for k in range(-m, m + 1)
                if (h, k) != (0, 0)
            ],
            dtype=int,
        )
        q = np.einsum("ni,ij,nj->n", vectors.astype(float), metric, vectors.astype(float))
        mask = (q > 1e-14) & (q <= qmax * 1.03)
        order = np.argsort(q[mask])
        q = q[mask][order]
        vectors = vectors[mask][order]
        groups: list[list[object]] = []
        for value, vector in zip(q, vectors):
            self._cancel_check()
            if not groups or abs(float(value) - float(groups[-1][0])) > 1e-8 * max(1.0, abs(float(value))):
                groups.append([float(value), [tuple(map(int, vector))]])
            else:
                groups[-1][1].append(tuple(map(int, vector)))  # type: ignore[index]
        return [(float(value), tuple(vectors_)) for value, vectors_ in groups]

    # ------------------------------------------------------------------
    # Combining two zones into a 3D reciprocal metric
    # ------------------------------------------------------------------

    def _combine_zones(
        self,
        zones: Sequence[_Zone],
        qobs: np.ndarray,
        tolerances: np.ndarray,
    ) -> list[_Trial]:
        if len(zones) < 2:
            return []
        raw: list[_Trial] = []

        for index, first in enumerate(zones):
            self._cancel_check()
            rows_first = self._primitive_zone_rows(first.metric, qobs[-1] + tolerances[-1])
            for second in zones[index + 1 :]:
                self._cancel_check()
                rows_second = self._primitive_zone_rows(second.metric, qobs[-1] + tolerances[-1])
                common_rows: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
                for q1, row1 in rows_first:
                    self._cancel_check()
                    for q2, row2 in rows_second:
                        self._cancel_check()
                        tolerance = max(
                            self.settings.q_absolute_tolerance,
                            self.settings.q_relative_tolerance * max(q1, q2),
                        )
                        if abs(q1 - q2) <= tolerance:
                            common_rows.append((abs(q1 - q2), row1, row2))
                common_rows.sort(key=lambda item: item[0])

                # A few best common rows are enough; later scoring rejects false
                # coincidences against the full 20-line search set.
                for _error, row1, row2 in common_rows[:4]:
                    self._cancel_check()
                    zone1 = self._move_common_row_to_first_basis(first.metric, row1)
                    zone2 = self._move_common_row_to_first_basis(second.metric, row2)
                    if zone1 is None or zone2 is None:
                        continue

                    A = 0.5 * (zone1[0, 0] + zone2[0, 0])
                    B = zone1[1, 1]
                    C = zone2[1, 1]
                    D = zone1[0, 1]
                    E = zone2[0, 1]

                    # The two zones define five of the six independent terms of
                    # G*.  A low-index line involving both remaining basis
                    # vectors determines the final cross term F.
                    for q_trial in qobs[: min(12, len(qobs))]:
                        self._cancel_check()
                        for m, n in ((1, 1), (1, -1), (1, 2), (2, 1)):
                            self._cancel_check()
                            F = (q_trial - m * m * B - n * n * C) / (2.0 * m * n)
                            metric = np.asarray(
                                ((A, D, E), (D, B, F), (E, F, C)), dtype=float
                            )
                            eigenvalues = np.linalg.eigvalsh(metric)
                            if float(eigenvalues[0]) <= 1e-10:
                                continue
                            cell = self._metric_to_cell(metric)
                            if cell is None or not self._cell_is_plausible(cell):
                                continue
                            matches, rms, _pairs, _groups = self._score_3d(
                                metric, qobs, tolerances
                            )
                            if matches >= max(12, self.settings.min_indexed_search_lines - 3):
                                raw.append(_Trial(metric, matches, rms))

        raw.sort(key=lambda trial: (-trial.search_matches, trial.rms))
        return raw[: max(self.settings.trial_keep_before_refine * 4, 200)]

    def _primitive_zone_rows(
        self, metric: np.ndarray, qmax: float
    ) -> list[tuple[float, tuple[int, int]]]:
        groups = self._predicted_2d(metric, qmax)
        rows: list[tuple[float, tuple[int, int]]] = []
        for q, equivalents in groups[:20]:
            self._cancel_check()
            for h, k in equivalents:
                self._cancel_check()
                if math.gcd(abs(h), abs(k)) == 1:
                    rows.append((q, (h, k)))
        return rows

    @staticmethod
    def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
        old_r, r = abs(a), abs(b)
        old_s, s = 1, 0
        old_t, t = 0, 1
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        x = old_s * (1 if a >= 0 else -1)
        y = old_t * (1 if b >= 0 else -1)
        return old_r, x, y

    def _move_common_row_to_first_basis(
        self, metric: np.ndarray, row: tuple[int, int]
    ) -> np.ndarray | None:
        h, k = row
        gcd, x, y = self._extended_gcd(h, k)
        if gcd != 1:
            return None
        # h*x + k*y = 1, therefore [[h, -y], [k, x]] is unimodular.
        transform = np.asarray(((h, -y), (k, x)), dtype=int)
        if abs(round(float(np.linalg.det(transform)))) != 1:
            return None
        return transform.T @ metric @ transform

    # ------------------------------------------------------------------
    # Trial-cell scoring and least-squares metric refinement
    # ------------------------------------------------------------------

    def _refine_trials(
        self,
        trials: Sequence[_Trial],
        qobs: np.ndarray,
        tolerances: np.ndarray,
    ) -> list[_Trial]:
        refined: list[_Trial] = []
        for trial in trials[: self.settings.trial_keep_before_refine]:
            self._cancel_check()
            metric = np.array(trial.metric, dtype=float)
            for _ in range(self.settings.refinement_iterations):
                self._cancel_check()
                _matches, _rms, pairs, groups = self._score_3d(metric, qobs, tolerances)
                if len(pairs) < 6:
                    break
                design: list[list[float]] = []
                target: list[float] = []
                for observed_index, calc_index, _error in pairs:
                    self._cancel_check()
                    h, k, l = min(
                        groups[calc_index][1],
                        key=lambda item: (sum(abs(value) for value in item), item),
                    )
                    design.append(
                        [h * h, k * k, l * l, 2 * h * k, 2 * h * l, 2 * k * l]
                    )
                    target.append(float(qobs[observed_index]))
                parameters, *_ = np.linalg.lstsq(
                    np.asarray(design, dtype=float),
                    np.asarray(target, dtype=float),
                    rcond=None,
                )
                proposed = np.asarray(
                    (
                        (parameters[0], parameters[3], parameters[4]),
                        (parameters[3], parameters[1], parameters[5]),
                        (parameters[4], parameters[5], parameters[2]),
                    ),
                    dtype=float,
                )
                if float(np.min(np.linalg.eigvalsh(proposed))) <= 1e-10:
                    break
                metric = proposed

            metric = self._reduce_reciprocal_metric(metric)
            matches, rms, _pairs, groups = self._score_3d(metric, qobs, tolerances)
            if matches < self.settings.min_indexed_search_lines:
                continue
            cell = self._metric_to_cell(metric)
            if cell is None or not self._cell_is_plausible(cell):
                continue
            refined.append(_Trial(metric, matches, rms))

        # Faithful correction: compare full shared Niggli metrics, not partial spectra.
        deduplicated: dict[tuple[float, ...], _Trial] = {}
        for trial in refined:
            self._cancel_check()
            signature = metric_signature(trial.metric)
            previous = deduplicated.get(signature)
            if previous is None or (-trial.search_matches, trial.rms) < (
                -previous.search_matches,
                previous.rms,
            ):
                deduplicated[signature] = trial

        return sorted(
            deduplicated.values(),
            key=lambda trial: (-trial.search_matches, trial.rms),
        )

    def _score_3d(
        self,
        metric: np.ndarray,
        qobs: np.ndarray,
        tolerances: np.ndarray,
    ) -> tuple[
        int,
        float,
        list[tuple[int, int, float]],
        list[tuple[float, tuple[tuple[int, int, int], ...]]],
    ]:
        groups = self._predicted_3d(metric, qobs[-1] + tolerances[-1])
        pairs = self._match_groups(groups, qobs, tolerances)
        residuals = [error / tolerances[observed] for observed, _calc, error in pairs]
        rms = (
            float(np.sqrt(np.mean(np.square(residuals))))
            if residuals
            else math.inf
        )
        return len(pairs), rms, pairs, groups

    def _predicted_3d(
        self, metric: np.ndarray, qmax: float
    ) -> list[tuple[float, tuple[tuple[int, int, int], ...]]]:
        vectors = self._integer_vectors_3d(self.settings.trial_hkl_max)
        floating = vectors.astype(float)
        q = np.einsum("ni,ij,nj->n", floating, metric, floating)
        mask = (q > 1e-14) & (q <= qmax * 1.03)
        order = np.argsort(q[mask])
        q = q[mask][order]
        vectors = vectors[mask][order]
        groups: list[list[object]] = []
        for value, vector in zip(q, vectors):
            self._cancel_check()
            if not groups or abs(float(value) - float(groups[-1][0])) > 1e-8 * max(1.0, abs(float(value))):
                groups.append([float(value), [tuple(map(int, vector))]])
            else:
                groups[-1][1].append(tuple(map(int, vector)))  # type: ignore[index]
        return [(float(value), tuple(vectors_)) for value, vectors_ in groups]

    def _integer_vectors_3d(self, maximum: int) -> np.ndarray:
        cached = self._vector_cache.get(maximum)
        if cached is not None:
            return cached
        values = []
        for h in range(-maximum, maximum + 1):
            self._cancel_check()
            for k in range(-maximum, maximum + 1):
                self._cancel_check()
                for l in range(-maximum, maximum + 1):
                    self._cancel_check()
                    if h == k == l == 0:
                        continue
                    triplet = (h, k, l)
                    first_nonzero = next(value for value in triplet if value != 0)
                    if first_nonzero < 0:
                        continue  # Friedel pair -h,-k,-l has the same Q.
                    values.append(triplet)
        array = np.asarray(values, dtype=int)
        self._vector_cache[maximum] = array
        return array

    @staticmethod
    def _match_groups(
        groups: Sequence[tuple[float, Sequence[tuple[int, ...]]]],
        qobs: np.ndarray,
        tolerances: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        if not groups:
            return []
        qcalc = np.asarray([group[0] for group in groups], dtype=float)
        edges: list[tuple[float, float, int, int]] = []
        for observed_index, q in enumerate(qobs):
            insertion = int(np.searchsorted(qcalc, q))
            for calc_index in range(
                max(0, insertion - 2), min(len(qcalc), insertion + 3)
            ):
                error = abs(float(qcalc[calc_index] - q))
                if error <= tolerances[observed_index]:
                    edges.append(
                        (
                            error / tolerances[observed_index],
                            error,
                            observed_index,
                            calc_index,
                        )
                    )
        edges.sort()
        used_observed: set[int] = set()
        used_calculated: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for _normalised, error, observed_index, calc_index in edges:
            if observed_index in used_observed or calc_index in used_calculated:
                continue
            used_observed.add(observed_index)
            used_calculated.add(calc_index)
            pairs.append((observed_index, calc_index, error))
        pairs.sort(key=lambda item: item[0])
        return pairs

    @staticmethod
    def _score_groups(
        groups: Sequence[tuple[float, Sequence[tuple[int, ...]]]],
        qobs: np.ndarray,
        tolerances: np.ndarray,
    ) -> tuple[int, float]:
        pairs = VisserIndexer._match_groups(groups, qobs, tolerances)
        residuals = [error / tolerances[observed] for observed, _calc, error in pairs]
        return (
            len(pairs),
            float(np.sqrt(np.mean(np.square(residuals)))) if residuals else math.inf,
        )

    @staticmethod
    def _line_signature(
        groups: Sequence[tuple[float, Sequence[tuple[int, ...]]]],
        scale: float,
        *,
        count: int,
    ) -> tuple[float, ...]:
        if scale <= 0:
            return ()
        return tuple(round(float(group[0]) / scale, 5) for group in groups[:count])

    # ------------------------------------------------------------------
    # Public candidate construction and figures of merit
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        trials: Sequence[_Trial],
        peaks: Sequence[_PreparedPeak],
        *,
        search_count: int,
    ) -> list[VisserCandidate]:
        if not trials:
            return []
        q_all = np.asarray([peak.q for peak in peaks], dtype=float)
        tol_all = np.asarray([peak.q_tolerance for peak in peaks], dtype=float)
        built: list[VisserCandidate] = []

        for trial in trials:
            self._cancel_check()
            cell = self._metric_to_cell(trial.metric)
            if cell is None:
                continue
            groups = self._predicted_3d(trial.metric, q_all[-1] + tol_all[-1])
            pairs = self._match_groups(groups, q_all, tol_all)
            by_observed = {observed: (calc, error) for observed, calc, error in pairs}

            indexed: list[IndexedLine] = []
            unindexed: list[UnindexedLine] = []
            residuals: list[float] = []
            search_matches = 0
            for observed_index, peak in enumerate(peaks):
                self._cancel_check()
                pair = by_observed.get(observed_index)
                if pair is None:
                    unindexed.append(
                        UnindexedLine(
                            peak.original_index,
                            peak.original_position,
                            peak.d,
                            peak.q,
                            peak.intensity,
                            peak.label,
                        )
                    )
                    continue
                calc_index, error = pair
                q_calc, equivalents = groups[calc_index]
                h, k, l = min(
                    equivalents,
                    key=lambda item: (sum(abs(value) for value in item), item),
                )
                d_calc = 1.0 / math.sqrt(q_calc)
                norm = error / peak.q_tolerance
                residuals.append(norm)
                if observed_index < search_count:
                    search_matches += 1
                indexed.append(
                    IndexedLine(
                        observed_index=peak.original_index,
                        observed_position=peak.original_position,
                        observed_d=peak.d,
                        observed_q=peak.q,
                        h=h,
                        k=k,
                        l=l,
                        calculated_d=d_calc,
                        calculated_q=q_calc,
                        delta_q=peak.q - q_calc,
                        normalized_residual=norm,
                        intensity=peak.intensity,
                        label=peak.label,
                    )
                )

            mean_abs_delta_q = (
                float(np.mean([abs(line.delta_q) for line in indexed]))
                if indexed
                else math.inf
            )
            rms = (
                float(np.sqrt(np.mean(np.square(residuals))))
                if residuals
                else math.inf
            )
            m20, n20, q20 = self._m20(indexed, groups, peaks)
            built.append(
                VisserCandidate(
                    rank=0,
                    cell=cell,
                    reciprocal_metric=tuple(
                        tuple(float(value) for value in row) for row in trial.metric
                    ),
                    indexed_lines=tuple(indexed),
                    unindexed_lines=tuple(unindexed),
                    indexed_search_lines=search_matches,
                    indexed_total_lines=len(indexed),
                    mean_abs_delta_q=mean_abs_delta_q,
                    rms_normalized_residual=rms,
                    m20=m20,
                    n20=n20,
                    q20=q20,
                )
            )

        def merit_value(candidate: VisserCandidate) -> float:
            value = candidate.m20
            return value if value is not None and not math.isnan(value) else -math.inf

        built.sort(
            key=lambda candidate: (
                -candidate.indexed_search_lines,
                -candidate.indexed_total_lines,
                -merit_value(candidate),
                candidate.rms_normalized_residual,
                candidate.volume,
            )
        )
        return [
            VisserCandidate(
                rank=rank,
                cell=candidate.cell,
                reciprocal_metric=candidate.reciprocal_metric,
                indexed_lines=candidate.indexed_lines,
                unindexed_lines=candidate.unindexed_lines,
                indexed_search_lines=candidate.indexed_search_lines,
                indexed_total_lines=candidate.indexed_total_lines,
                mean_abs_delta_q=candidate.mean_abs_delta_q,
                rms_normalized_residual=candidate.rms_normalized_residual,
                m20=candidate.m20,
                n20=candidate.n20,
                q20=candidate.q20,
            )
            for rank, candidate in enumerate(built[: self.settings.max_candidates], start=1)
        ]

    @staticmethod
    def _m20(
        indexed: Sequence[IndexedLine],
        groups: Sequence[tuple[float, Sequence[tuple[int, ...]]]],
        peaks: Sequence[_PreparedPeak],
    ) -> tuple[float | None, int | None, float | None]:
        if len(peaks) < 20:
            return None, None, None
        first_ids = {p.original_index for p in peaks[:20]}
        ordered = sorted((line for line in indexed if line.observed_index in first_ids),
                         key=lambda line: line.observed_q)
        if len(ordered) < 20:
            return None, None, None
        first_twenty = ordered[:20]
        q20 = first_twenty[-1].observed_q
        mean_error = float(np.mean([abs(line.delta_q) for line in first_twenty]))
        n20 = sum(1 for q, _equivalents in groups if q <= q20 + 1e-12)
        if n20 == 0:
            return None, None, q20
        if mean_error <= 1e-15:
            return math.inf, n20, q20
        return q20 / (2.0 * mean_error * n20), n20, q20

    def _reduce_reciprocal_metric(self, metric: np.ndarray) -> np.ndarray:
        """Return a short primitive basis for an equivalent reciprocal metric.

        Visser's workflow includes a cell-reduction stage.  Here the same
        purpose is served by a bounded unimodular search in the direct lattice:
        short primitive direct-lattice vectors are combined into determinant-1
        bases and the shortest well-conditioned basis is selected.  This keeps
        equivalent sheared descriptions from being presented to the caller when
        a shorter primitive cell is available.

        This is intentionally self-contained; it does not attempt to assign a
        Bravais lattice or a conventional crystallographic setting.
        """

        try:
            direct = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            return metric

        vectors: list[tuple[float, np.ndarray]] = []
        span = 2
        for triplet in itertools.product(range(-span, span + 1), repeat=3):
            self._cancel_check()
            if triplet == (0, 0, 0):
                continue
            divisor = math.gcd(math.gcd(abs(triplet[0]), abs(triplet[1])), abs(triplet[2]))
            if divisor != 1:
                continue
            first_nonzero = next(value for value in triplet if value != 0)
            if first_nonzero < 0:
                continue
            vector = np.asarray(triplet, dtype=int)
            floating = vector.astype(float)
            length2 = float(floating @ direct @ floating)
            if length2 > 0 and math.isfinite(length2):
                vectors.append((length2, vector))
        vectors.sort(key=lambda item: item[0])
        short = [vector for _length, vector in vectors[:36]]

        best_transform: np.ndarray | None = None
        best_key: tuple[float, float, float] | None = None
        for first, second, third in itertools.combinations(short, 3):
            self._cancel_check()
            transform = np.column_stack((first, second, third))
            if abs(round(float(np.linalg.det(transform)))) != 1:
                continue
            proposed = transform.T @ direct @ transform
            lengths = np.sqrt(np.diag(proposed))
            if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
                continue
            angular = sum(
                abs(float(proposed[i, j]) / (lengths[i] * lengths[j]))
                for i, j in ((0, 1), (0, 2), (1, 2))
            )
            key = (float(np.max(lengths)), float(np.sum(lengths * lengths)), angular)
            if best_key is None or key < best_key:
                best_key = key
                best_transform = transform

        if best_transform is None:
            return metric

        # Canonicalise permutations and signs: lengths increase a <= b <= c;
        # where supplementary-angle choices are possible, prefer the setting
        # with fewer acute angles.  This is presentation canonicalisation, not
        # a space-group/conventional-cell assignment.
        canonical: tuple[tuple[object, ...], np.ndarray] | None = None
        for permutation in itertools.permutations(range(3)):
            self._cancel_check()
            permute = np.eye(3, dtype=int)[:, permutation]
            for signs in itertools.product((1, -1), repeat=3):
                self._cancel_check()
                if signs[0] < 0:
                    continue
                signed = np.diag(signs)
                transform = best_transform @ permute @ signed
                proposed = transform.T @ direct @ transform
                lengths = np.sqrt(np.diag(proposed))
                if not (
                    lengths[0] <= lengths[1] + 1e-10
                    and lengths[1] <= lengths[2] + 1e-10
                ):
                    continue
                angles = []
                for i, j, left, right in (
                    (1, 2, lengths[1], lengths[2]),
                    (0, 2, lengths[0], lengths[2]),
                    (0, 1, lengths[0], lengths[1]),
                ):
                    self._cancel_check()
                    cosine = float(proposed[i, j]) / float(left * right)
                    angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
                acute = sum(angle < 89.999999 for angle in angles)
                key: tuple[object, ...] = (
                    *(round(float(value), 10) for value in lengths),
                    acute,
                    *(round(abs(angle - 90.0), 10) for angle in angles),
                )
                if canonical is None or key < canonical[0]:
                    canonical = (key, proposed)

        if canonical is None:
            return metric
        try:
            reduced = np.linalg.inv(canonical[1])
        except np.linalg.LinAlgError:
            return metric
        return reduced if np.all(np.isfinite(reduced)) else metric

    @staticmethod
    def _metric_to_cell(metric: np.ndarray) -> UnitCell | None:
        try:
            direct = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            return None
        diagonal = np.diag(direct)
        if np.any(diagonal <= 0) or not np.all(np.isfinite(direct)):
            return None
        a, b, c = (math.sqrt(float(diagonal[index])) for index in range(3))

        def angle(i: int, j: int, left: float, right: float) -> float:
            cosine = float(direct[i, j]) / (left * right)
            return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

        return UnitCell(
            a,
            b,
            c,
            angle(1, 2, b, c),
            angle(0, 2, a, c),
            angle(0, 1, a, b),
        )

    def _cell_is_plausible(self, cell: UnitCell) -> bool:
        s = self.settings
        if not all(s.min_cell_length <= value <= s.max_cell_length for value in (cell.a, cell.b, cell.c)):
            return False
        if not all(
            s.min_cell_angle_deg <= value <= s.max_cell_angle_deg
            for value in (cell.alpha, cell.beta, cell.gamma)
        ):
            return False
        return math.isfinite(cell.volume) and cell.volume > 0


def index_visser(
    positions: Sequence[float],
    *,
    wavelength: float | None = None,
    input_kind: InputKind = "two_theta",
    intensities: Sequence[float] | None = None,
    uncertainties: Sequence[float] | None = None,
    labels: Sequence[str | None] | None = None,
    settings: VisserSettings | None = None,
) -> VisserResult:
    """Convenience function for callers that already have plain arrays.

    Parameters
    ----------
    positions:
        Peak positions.  For ``two_theta`` they are degrees; for ``d`` they are
        Angstrom; for ``q`` they are 1/d^2 in A^-2.
    wavelength:
        X-ray wavelength in Angstrom.  Required only for ``two_theta`` input.
    input_kind:
        ``"two_theta"``, ``"d"``, or ``"q"``.
    intensities, uncertainties, labels:
        Optional metadata aligned one-to-one with ``positions``.  Intensities
        do not influence the Visser search.
    settings:
        Optional numerical controls.  Classical defaults use 20 search lines,
        six best zones, and at most four returned cells.
    """

    count = len(positions)
    for optional, name in (
        (intensities, "intensities"),
        (uncertainties, "uncertainties"),
        (labels, "labels"),
    ):
        if optional is not None and len(optional) != count:
            raise ValueError(f"{name} must have the same length as positions")

    peaks = [
        VisserPeak(
            float(position),
            None if intensities is None else float(intensities[index]),
            None if uncertainties is None else float(uncertainties[index]),
            None if labels is None else labels[index],
        )
        for index, position in enumerate(positions)
    ]
    return VisserIndexer(settings).index(
        peaks,
        input_kind=input_kind,
        wavelength=wavelength,
    )


__all__ = [
    "DE_WOLFF_PUBLICATION",
    "IndexedLine",
    "METHOD_DESCRIPTION",
    "METHOD_ID",
    "METHOD_NAME",
    "Publication",
    "REFERENCE_AUTHOR",
    "REFERENCE_BIBTEX",
    "REFERENCE_CITATION",
    "REFERENCE_DOI",
    "REFERENCE_JOURNAL",
    "REFERENCE_PAGES",
    "REFERENCE_TITLE",
    "REFERENCE_URL",
    "REFERENCE_VOLUME",
    "REFERENCE_YEAR",
    "UnitCell",
    "UnindexedLine",
    "VISSER_PUBLICATION",
    "VisserCandidate",
    "VisserDiagnostics",
    "VisserIndexer",
    "VisserPeak",
    "VisserResult",
    "VisserSettings",
    "index_visser",
]
