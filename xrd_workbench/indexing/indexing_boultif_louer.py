"""Independent Boultif-Louër successive-dichotomy powder indexer.

This module implements a modern, self-contained interpretation of the
successive dichotomy strategy developed by D. Louër, M. Louër, A. Boultif and
collaborators for powder-pattern indexing.  It does not execute, wrap,
translate, or depend on DICVOL91/DICVOL04 source code or binaries.

Primary publication
-------------------
A. Boultif and D. Louër, "Powder pattern indexing with the dichotomy method",
Journal of Applied Crystallography 37 (2004) 724-731.
DOI: 10.1107/S0021889804014876

Low-symmetry theoretical basis
------------------------------
A. Boultif and D. Louër, "Indexing of powder diffraction patterns for
low-symmetry lattices by the successive dichotomy method",
Journal of Applied Crystallography 24 (1991) 987-993.
DOI: 10.1107/S0021889891006441

Historical origin
-----------------
D. Louër and M. Louër, "Méthode d'essais et erreurs pour l'indexation
automatique des diagrammes de poudre", Journal of Applied Crystallography 5
(1972) 271-275. DOI: 10.1107/S0021889872009483

Implementation notes
--------------------
* The search is performed separately for cubic, tetragonal, hexagonal,
  orthorhombic, monoclinic and triclinic metrics, matching the strategy
  emphasized in the 2004 paper.
* The numerical core uses interval boxes in reciprocal-metric parameters.
  Boxes are repeatedly bisected (successive dichotomy); interval bounds for
  Q(hkl) decide whether a box can still explain enough observed lines.
* The first ``search_peak_count`` lines (20 by default) discover cells.  Every
  retained input line is then reviewed against each surviving solution.
* A configurable number of spurious/inaccurate observed lines may be tolerated.
* Optional zero-point searching is available for 2theta input.  It is an
  independent numerical implementation, not a reproduction of DICVOL04's
  internal zero-point procedure.
* Equivalent trial cells are compared using shared Niggli-reduced metrics.
  This does not claim to reproduce historical reduced-cell source code.
* No space group or lattice centering is inferred.  Candidate cells are
  primitive metric cells with hkl assignments; systematic-absence analysis is
  deliberately left to a later crystallographic step.
* Search is heuristic and bounded by retained boxes, leaf/depth/index limits,
  seed caps and point matching. Raising one limit does not make it exhaustive.

The de Wolff M20 and Smith-Snyder F20 figures of merit are reported when the
first 20 observed lines are all indexed and the required coordinate conversion
is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal, Sequence
import itertools
import math

import numpy as np

from .models import Publication, UnitCell, IndexedLine, UnindexedLine, Peak as BoultifLouerPeak
from .reduction import metric_signature


# ---------------------------------------------------------------------------
# Public bibliographic metadata
# ---------------------------------------------------------------------------

METHOD_ID = "boultif_louer"
METHOD_NAME = "Boultif-Louër"
METHOD_DESCRIPTION = "Boultif-Louër successive dichotomy indexing"

REFERENCE_AUTHOR = "A. Boultif; D. Louër"
REFERENCE_TITLE = "Powder pattern indexing with the dichotomy method"
REFERENCE_JOURNAL = "Journal of Applied Crystallography"
REFERENCE_YEAR = 2004
REFERENCE_VOLUME = 37
REFERENCE_PAGES = "724-731"
REFERENCE_DOI = "10.1107/S0021889804014876"
REFERENCE_URL = f"https://doi.org/{REFERENCE_DOI}"
REFERENCE_CITATION = (
    f"A. Boultif and D. Louër, \"{REFERENCE_TITLE}\", "
    f"{REFERENCE_JOURNAL} {REFERENCE_VOLUME} ({REFERENCE_YEAR}) "
    f"{REFERENCE_PAGES}. DOI: {REFERENCE_DOI}"
)
REFERENCE_BIBTEX = """@article{BoultifLouer2004,
  author  = {Boultif, A. and Lou{\\\"e}r, D.},
  title   = {Powder pattern indexing with the dichotomy method},
  journal = {Journal of Applied Crystallography},
  year    = {2004},
  volume  = {37},
  pages   = {724--731},
  doi     = {10.1107/S0021889804014876}
}"""

THEORY_REFERENCE_AUTHOR = "A. Boultif; D. Louër"
THEORY_REFERENCE_TITLE = (
    "Indexing of powder diffraction patterns for low-symmetry lattices "
    "by the successive dichotomy method"
)
THEORY_REFERENCE_JOURNAL = "Journal of Applied Crystallography"
THEORY_REFERENCE_YEAR = 1991
THEORY_REFERENCE_VOLUME = 24
THEORY_REFERENCE_PAGES = "987-993"
THEORY_REFERENCE_DOI = "10.1107/S0021889891006441"

ORIGIN_REFERENCE_AUTHOR = "D. Louër; M. Louër"
ORIGIN_REFERENCE_TITLE = (
    "Méthode d'essais et erreurs pour l'indexation automatique des diagrammes de poudre"
)
ORIGIN_REFERENCE_JOURNAL = "Journal of Applied Crystallography"
ORIGIN_REFERENCE_YEAR = 1972
ORIGIN_REFERENCE_VOLUME = 5
ORIGIN_REFERENCE_PAGES = "271-275"
ORIGIN_REFERENCE_DOI = "10.1107/S0021889872009483"

M20_REFERENCE_AUTHOR = "P. M. de Wolff"
M20_REFERENCE_TITLE = "A simplified criterion for the reliability of a powder pattern indexing"
M20_REFERENCE_JOURNAL = "Journal of Applied Crystallography"
M20_REFERENCE_YEAR = 1968
M20_REFERENCE_VOLUME = 1
M20_REFERENCE_PAGES = "108-113"
M20_REFERENCE_DOI = "10.1107/S002188986800508X"

F20_REFERENCE_AUTHOR = "G. S. Smith; R. L. Snyder"
F20_REFERENCE_TITLE = (
    "F_N: A criterion for rating powder diffraction patterns and evaluating "
    "the reliability of powder-pattern indexing"
)
F20_REFERENCE_JOURNAL = "Journal of Applied Crystallography"
F20_REFERENCE_YEAR = 1979
F20_REFERENCE_VOLUME = 12
F20_REFERENCE_PAGES = "60-65"
F20_REFERENCE_DOI = "10.1107/S002188987901178X"


InputKind = Literal["two_theta", "d", "q"]
CrystalSystem = Literal[
    "cubic",
    "tetragonal",
    "hexagonal",
    "orthorhombic",
    "monoclinic",
    "triclinic",
]

CRYSTAL_SYSTEMS: tuple[CrystalSystem, ...] = (
    "cubic",
    "tetragonal",
    "hexagonal",
    "orthorhombic",
    "monoclinic",
    "triclinic",
)




BOULTIF_LOUER_PUBLICATION = Publication(
    REFERENCE_AUTHOR,
    REFERENCE_TITLE,
    REFERENCE_JOURNAL,
    REFERENCE_YEAR,
    REFERENCE_VOLUME,
    REFERENCE_PAGES,
    REFERENCE_DOI,
)

BOULTIF_LOUER_1991_PUBLICATION = Publication(
    THEORY_REFERENCE_AUTHOR,
    THEORY_REFERENCE_TITLE,
    THEORY_REFERENCE_JOURNAL,
    THEORY_REFERENCE_YEAR,
    THEORY_REFERENCE_VOLUME,
    THEORY_REFERENCE_PAGES,
    THEORY_REFERENCE_DOI,
)

LOUER_LOUER_1972_PUBLICATION = Publication(
    ORIGIN_REFERENCE_AUTHOR,
    ORIGIN_REFERENCE_TITLE,
    ORIGIN_REFERENCE_JOURNAL,
    ORIGIN_REFERENCE_YEAR,
    ORIGIN_REFERENCE_VOLUME,
    ORIGIN_REFERENCE_PAGES,
    ORIGIN_REFERENCE_DOI,
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

SMITH_SNYDER_PUBLICATION = Publication(
    F20_REFERENCE_AUTHOR,
    F20_REFERENCE_TITLE,
    F20_REFERENCE_JOURNAL,
    F20_REFERENCE_YEAR,
    F20_REFERENCE_VOLUME,
    F20_REFERENCE_PAGES,
    F20_REFERENCE_DOI,
)

METHOD_PUBLICATIONS = (
    LOUER_LOUER_1972_PUBLICATION,
    BOULTIF_LOUER_1991_PUBLICATION,
    BOULTIF_LOUER_PUBLICATION,
)


# ---------------------------------------------------------------------------
# Public data objects
# ---------------------------------------------------------------------------




@dataclass(frozen=True)
class BoultifLouerSettings:
    """Controls for the independent successive-dichotomy implementation."""

    search_peak_count: int = 20
    min_required_peaks: int = 8
    max_input_peaks: int = 80
    systems: tuple[CrystalSystem, ...] = CRYSTAL_SYSTEMS
    max_candidates: int = 12
    max_candidates_per_system: int = 5

    # DICVOL04 extended common default limits to 25 A and 2500 A^3.  We retain
    # those values as practical defaults but expose them as ordinary settings.
    min_cell_length: float = 1.5
    max_cell_length: float = 25.0
    max_cell_volume: float = 2500.0
    min_cell_angle_deg: float = 35.0
    max_cell_angle_deg: float = 145.0

    # Peak-position tolerance.  DICVOL documentation commonly uses 0.03 deg
    # 2theta as the default absolute error; q tolerances cover d/q inputs.
    two_theta_tolerance_deg: float = 0.03
    q_absolute_tolerance: float = 2.0e-5
    q_relative_tolerance: float = 2.0e-3
    max_spurious_search_lines: int = 1

    # Successive-dichotomy search controls.
    hkl_max: int = 5
    initial_diagonal_partitions: int = 3
    max_dichotomy_depth: int = 24
    max_boxes_per_level: int = 500
    max_leaf_boxes: int = 180
    reciprocal_q_upper_factor: float = 2.5
    reciprocal_diag_max_factor: float = 4.0
    off_diagonal_correlation_limit: float = 0.92
    min_parameter_relative_width: float = 2.0e-3

    # Low-symmetry bound-contraction stage.  The 1991 treatment emphasizes
    # optimized bound relations for monoclinic/triclinic searches.  This
    # implementation first constrains interval boxes with a few low-angle
    # lines and low-index reflection hypotheses before ordinary bisection.
    low_symmetry_seed_template_limit: int = 18
    low_symmetry_seed_state_cap: int = 3000
    low_symmetry_seed_refine_limit: int = 1500

    # Refinement and matching.
    refinement_iterations: int = 6
    relaxed_refinement_tolerance_factor: float = 5.0
    spectrum_dedup_decimals: int = 6  # Legacy API field; full reduced metrics now identify duplicates.

    # Optional zero-point search.  A correction is ADDED to observed 2theta.
    search_zero_shift: bool = False
    zero_shift_half_range_deg: float = 0.20
    zero_shift_steps: int = 5
    refine_zero_shift: bool = False
    zero_refine_iterations: int = 4

    def validate(self) -> None:
        if self.search_peak_count < self.min_required_peaks:
            raise ValueError("search_peak_count must be >= min_required_peaks")
        if self.min_required_peaks < 4:
            raise ValueError("min_required_peaks must be at least 4")
        if self.max_input_peaks < self.search_peak_count:
            raise ValueError("max_input_peaks must be >= search_peak_count")
        if self.max_candidates < 1 or self.max_candidates_per_system < 1:
            raise ValueError("candidate limits must be positive")
        if not self.systems:
            raise ValueError("At least one crystal system must be enabled")
        unknown = set(self.systems) - set(CRYSTAL_SYSTEMS)
        if unknown:
            raise ValueError(f"Unknown crystal system(s): {sorted(unknown)}")
        if self.min_cell_length <= 0 or self.max_cell_length <= self.min_cell_length:
            raise ValueError("Invalid cell-length limits")
        if self.max_cell_volume <= 0:
            raise ValueError("max_cell_volume must be positive")
        if not 0 < self.min_cell_angle_deg < self.max_cell_angle_deg < 180:
            raise ValueError("Invalid cell-angle limits")
        if self.two_theta_tolerance_deg <= 0:
            raise ValueError("two_theta_tolerance_deg must be positive")
        if self.q_absolute_tolerance <= 0 or self.q_relative_tolerance <= 0:
            raise ValueError("Q tolerances must be positive")
        if self.max_spurious_search_lines < 0:
            raise ValueError("max_spurious_search_lines cannot be negative")
        if self.hkl_max < 2:
            raise ValueError("hkl_max must be at least 2")
        if self.initial_diagonal_partitions < 1:
            raise ValueError("initial_diagonal_partitions must be positive")
        if self.max_dichotomy_depth < 1 or self.max_boxes_per_level < 2:
            raise ValueError("Dichotomy search limits are too small")
        if self.max_leaf_boxes < 1:
            raise ValueError("max_leaf_boxes must be positive")
        if self.low_symmetry_seed_template_limit < 6:
            raise ValueError("low_symmetry_seed_template_limit must be at least 6")
        if self.low_symmetry_seed_state_cap < 100:
            raise ValueError("low_symmetry_seed_state_cap must be at least 100")
        if self.low_symmetry_seed_refine_limit < 1:
            raise ValueError("low_symmetry_seed_refine_limit must be positive")
        if not 0 < self.off_diagonal_correlation_limit < 1:
            raise ValueError("off_diagonal_correlation_limit must lie in (0, 1)")
        if self.refinement_iterations < 0:
            raise ValueError("refinement_iterations cannot be negative")
        if self.search_zero_shift:
            if self.zero_shift_half_range_deg <= 0:
                raise ValueError("zero_shift_half_range_deg must be positive")
            if self.zero_shift_steps < 3:
                raise ValueError("zero_shift_steps must be at least 3")








@dataclass(frozen=True)
class BoultifLouerCandidate:
    rank: int
    crystal_system: CrystalSystem
    cell: UnitCell
    reciprocal_metric: tuple[tuple[float, float, float], ...]
    indexed_lines: tuple[IndexedLine, ...]
    unindexed_lines: tuple[UnindexedLine, ...]
    indexed_search_lines: int
    indexed_total_lines: int
    spurious_search_lines: int
    mean_abs_delta_q: float
    rms_normalized_residual: float
    zero_shift_deg: float
    m20: float | None
    f20: float | None
    n20_possible: int | None
    q20: float | None

    @property
    def volume(self) -> float:
        return self.cell.volume


@dataclass(frozen=True)
class SystemDiagnostics:
    crystal_system: CrystalSystem
    zero_shift_deg: float
    reflection_templates: int
    initial_boxes: int
    boxes_evaluated: int
    constraint_seed_states: int
    surviving_leaf_boxes: int
    candidate_cells: int


@dataclass(frozen=True)
class BoultifLouerDiagnostics:
    input_line_count: int
    retained_line_count: int
    search_line_count: int
    systems_searched: tuple[CrystalSystem, ...]
    zero_shifts_tested: tuple[float, ...]
    system_runs: tuple[SystemDiagnostics, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoultifLouerResult:
    method_id: str
    method_name: str
    publication: Publication
    theory_publication: Publication
    origin_publication: Publication
    m20_publication: Publication
    f20_publication: Publication
    input_kind: InputKind
    wavelength: float | None
    candidates: tuple[BoultifLouerCandidate, ...]
    diagnostics: BoultifLouerDiagnostics

    @property
    def best(self) -> BoultifLouerCandidate | None:
        return self.candidates[0] if self.candidates else None


# ---------------------------------------------------------------------------
# Internal representations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PreparedPeak:
    original_index: int
    original_position: float
    corrected_position: float
    d: float
    q: float
    q_tolerance: float
    two_theta: float | None
    intensity: float | None
    label: str | None


@dataclass(frozen=True)
class _ReflectionTemplate:
    coefficients: tuple[float, ...]
    hkls: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _Box:
    low: np.ndarray = field(compare=False, repr=False)
    high: np.ndarray = field(compare=False, repr=False)
    depth: int
    potential_matches: int = 0
    ambiguity: float = math.inf


@dataclass(frozen=True)
class _SeedState:
    box: _Box
    used_templates: frozenset[int]
    assignments: tuple[int | None, ...]
    skipped_lines: int = 0


@dataclass(frozen=True)
class _Match:
    peak_index: int
    template_index: int
    q_calculated: float
    delta_q: float


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class BoultifLouerIndexer:
    """Independent successive-dichotomy powder-indexing engine."""

    METHOD_ID: ClassVar[str] = METHOD_ID
    METHOD_NAME: ClassVar[str] = METHOD_NAME
    METHOD_DESCRIPTION: ClassVar[str] = METHOD_DESCRIPTION
    PUBLICATION: ClassVar[Publication] = BOULTIF_LOUER_PUBLICATION
    THEORY_PUBLICATION: ClassVar[Publication] = BOULTIF_LOUER_1991_PUBLICATION
    ORIGIN_PUBLICATION: ClassVar[Publication] = LOUER_LOUER_1972_PUBLICATION
    METHOD_PUBLICATIONS: ClassVar[tuple[Publication, ...]] = METHOD_PUBLICATIONS
    M20_PUBLICATION: ClassVar[Publication] = DE_WOLFF_PUBLICATION
    F20_PUBLICATION: ClassVar[Publication] = SMITH_SNYDER_PUBLICATION
    REFERENCE_DOI: ClassVar[str] = REFERENCE_DOI
    REFERENCE_URL: ClassVar[str] = REFERENCE_URL
    REFERENCE_CITATION: ClassVar[str] = REFERENCE_CITATION
    REFERENCE_BIBTEX: ClassVar[str] = REFERENCE_BIBTEX

    def __init__(self, settings: BoultifLouerSettings | None = None, *, cancel_check=None) -> None:
        self._cancel_check = cancel_check or (lambda: None)
        self.settings = settings or BoultifLouerSettings()
        self.settings.validate()
        self._template_cache: dict[tuple[CrystalSystem, int], tuple[_ReflectionTemplate, ...]] = {}

    def index(
        self,
        peaks: Sequence[BoultifLouerPeak],
        *,
        input_kind: InputKind = "two_theta",
        wavelength: float | None = None,
    ) -> BoultifLouerResult:
        if input_kind not in {"two_theta", "d", "q"}:
            raise ValueError("input_kind must be 'two_theta', 'd', or 'q'")
        if input_kind == "two_theta":
            if wavelength is None or not math.isfinite(wavelength) or wavelength <= 0:
                raise ValueError("A positive wavelength is required for two_theta input")
        elif wavelength is not None and (not math.isfinite(wavelength) or wavelength <= 0):
            raise ValueError("wavelength must be positive when supplied")
        if self.settings.search_zero_shift and input_kind != "two_theta":
            raise ValueError("zero-point search requires two_theta input")

        base = self._prepare_peaks(
            peaks, input_kind=input_kind, wavelength=wavelength, zero_shift_deg=0.0
        )
        if len(base) < self.settings.min_required_peaks:
            raise ValueError(
                f"Boultif-Louër indexing needs at least "
                f"{self.settings.min_required_peaks} valid lines; received {len(base)}"
            )

        warnings: list[str] = []
        if len(base) < self.settings.search_peak_count:
            warnings.append(
                f"Only {len(base)} valid lines are available; the search will use all of them "
                f"instead of the configured {self.settings.search_peak_count}."
            )
        if len(base) > self.settings.max_input_peaks:
            warnings.append(
                f"Only the first {self.settings.max_input_peaks} lines are retained for review."
            )

        zero_shifts = self._zero_shift_grid(input_kind)
        all_candidates: list[BoultifLouerCandidate] = []
        run_diagnostics: list[SystemDiagnostics] = []

        for zero_shift in zero_shifts:
            self._cancel_check()
            prepared = self._prepare_peaks(
                peaks,
                input_kind=input_kind,
                wavelength=wavelength,
                zero_shift_deg=zero_shift,
            )[: self.settings.max_input_peaks]
            search_count = min(self.settings.search_peak_count, len(prepared))
            search = prepared[:search_count]
            for system in self.settings.systems:
                self._cancel_check()
                candidates, diagnostics = self._search_system(
                    system,
                    search,
                    prepared,
                    input_kind=input_kind,
                    wavelength=wavelength,
                    zero_shift_deg=zero_shift,
                )
                all_candidates.extend(candidates)
                run_diagnostics.append(diagnostics)

        if (
            input_kind == "two_theta"
            and self.settings.refine_zero_shift
            and wavelength is not None
            and all_candidates
        ):
            refined_zero: list[BoultifLouerCandidate] = []
            # Refining every duplicate trial can be expensive; first retain a
            # small, diverse set from the coarse search, then alternate a local
            # zero-shift scan with linear metric refinement.
            provisional = self._deduplicate_and_rank(all_candidates)
            for candidate in provisional[: max(self.settings.max_candidates * 2, 8)]:
                self._cancel_check()
                refined_zero.append(
                    self._refine_zero_shift_candidate(candidate, peaks, wavelength)
                )
            all_candidates = refined_zero

        candidates = self._deduplicate_and_rank(all_candidates)
        candidates = candidates[: self.settings.max_candidates]
        candidates = [replace(candidate, rank=index + 1) for index, candidate in enumerate(candidates)]

        diagnostics = BoultifLouerDiagnostics(
            input_line_count=len(base),
            retained_line_count=min(len(base), self.settings.max_input_peaks),
            search_line_count=min(len(base), self.settings.search_peak_count),
            systems_searched=tuple(self.settings.systems),
            zero_shifts_tested=tuple(float(value) for value in zero_shifts),
            system_runs=tuple(run_diagnostics),
            warnings=tuple(warnings),
        )
        return BoultifLouerResult(
            method_id=self.METHOD_ID,
            method_name=self.METHOD_NAME,
            publication=self.PUBLICATION,
            theory_publication=self.THEORY_PUBLICATION,
            origin_publication=self.ORIGIN_PUBLICATION,
            m20_publication=self.M20_PUBLICATION,
            f20_publication=self.F20_PUBLICATION,
            input_kind=input_kind,
            wavelength=wavelength,
            candidates=tuple(candidates),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _zero_shift_grid(self, input_kind: InputKind) -> np.ndarray:
        if input_kind != "two_theta" or not self.settings.search_zero_shift:
            return np.asarray([0.0])
        return np.linspace(
            -self.settings.zero_shift_half_range_deg,
            self.settings.zero_shift_half_range_deg,
            self.settings.zero_shift_steps,
        )

    def _prepare_peaks(
        self,
        peaks: Sequence[BoultifLouerPeak],
        *,
        input_kind: InputKind,
        wavelength: float | None,
        zero_shift_deg: float,
    ) -> list[_PreparedPeak]:
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

            two_theta: float | None = None
            if input_kind == "two_theta":
                corrected = value + zero_shift_deg
                if not 0.0 < corrected < 180.0:
                    continue
                two_theta = corrected
                d = self._two_theta_to_d(corrected, float(wavelength))
                q = 1.0 / (d * d)
                angular_error = (
                    self.settings.two_theta_tolerance_deg
                    if uncertainty is None else uncertainty
                )
                low = max(1e-9, corrected - angular_error)
                high = min(179.999999, corrected + angular_error)
                q_low = 1.0 / self._two_theta_to_d(low, float(wavelength)) ** 2
                q_high = 1.0 / self._two_theta_to_d(high, float(wavelength)) ** 2
                q_tol = max(abs(q - q_low), abs(q_high - q))
                corrected_position = corrected
            elif input_kind == "d":
                if value <= 0:
                    continue
                corrected_position = value
                d = value
                q = 1.0 / (d * d)
                if uncertainty is None:
                    q_tol = max(
                        self.settings.q_absolute_tolerance,
                        self.settings.q_relative_tolerance * q,
                    )
                else:
                    dlo = max(1e-12, d - uncertainty)
                    dhi = d + uncertainty
                    q_tol = max(abs(q - 1.0 / dlo**2), abs(q - 1.0 / dhi**2))
                if wavelength is not None:
                    two_theta = self._d_to_two_theta(d, wavelength)
            else:
                if value <= 0:
                    continue
                corrected_position = value
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
                if wavelength is not None:
                    two_theta = self._d_to_two_theta(d, wavelength)

            q_tol = float(q_tol)
            if not explicit_uncertainty:
                q_tol = max(q_tol, self.settings.q_absolute_tolerance,
                            self.settings.q_relative_tolerance * q)
            prepared.append(
                _PreparedPeak(
                    original_index=index,
                    original_position=value,
                    corrected_position=corrected_position,
                    d=d,
                    q=q,
                    q_tolerance=q_tol,
                    two_theta=two_theta,
                    intensity=None if peak.intensity is None else float(peak.intensity),
                    label=peak.label,
                )
            )
        prepared.sort(key=lambda item: item.q)
        return prepared

    @staticmethod
    def _two_theta_to_d(two_theta: float, wavelength: float) -> float:
        sine = math.sin(math.radians(two_theta / 2.0))
        if sine <= 0:
            raise ValueError("Non-physical two_theta value")
        return wavelength / (2.0 * sine)

    @staticmethod
    def _d_to_two_theta(d: float, wavelength: float) -> float | None:
        argument = wavelength / (2.0 * d)
        if not 0.0 < argument <= 1.0:
            return None
        return math.degrees(2.0 * math.asin(argument))

    # ------------------------------------------------------------------
    # System search by successive dichotomy
    # ------------------------------------------------------------------

    def _search_system(
        self,
        system: CrystalSystem,
        search: Sequence[_PreparedPeak],
        retained: Sequence[_PreparedPeak],
        *,
        input_kind: InputKind,
        wavelength: float | None,
        zero_shift_deg: float,
    ) -> tuple[list[BoultifLouerCandidate], SystemDiagnostics]:
        templates = self._reflection_templates(system)
        coefficients = np.asarray([item.coefficients for item in templates], dtype=float)
        qobs = np.asarray([item.q for item in search], dtype=float)
        tolerances = np.asarray([item.q_tolerance for item in search], dtype=float)
        roots = self._initial_boxes(system, qobs)
        required = max(
            self.settings.min_required_peaks,
            len(search) - self.settings.max_spurious_search_lines,
        )

        evaluated = 0
        candidates: list[BoultifLouerCandidate] = []
        seed_states: list[_SeedState] = []

        # Boultif & Louër (1991) devote special attention to optimized bound
        # relations for low-symmetry metrics.  Before blind bisection, contract
        # the broad reciprocal-metric boxes against a few lowest-Q lines and
        # low-index hkl hypotheses.  This is still interval/dichotomy search:
        # each retained state is a parameter box, not a point estimate.
        if system in {"monoclinic", "triclinic"}:
            seed_states, seed_evaluated = self._low_symmetry_seed_states(
                roots, coefficients, qobs, tolerances, required
            )
            evaluated += seed_evaluated
            for state in seed_states[: self.settings.low_symmetry_seed_refine_limit]:
                self._cancel_check()
                candidate = self._candidate_from_seed_state(
                    system,
                    state,
                    templates,
                    search,
                    retained,
                    wavelength=wavelength,
                    zero_shift_deg=zero_shift_deg,
                )
                if candidate is not None:
                    candidates.append(candidate)

        leaves: list[_Box] = []
        # For low symmetry the constraint stage normally supplies the useful
        # leaves.  Fall back to ordinary level-wise dichotomy if it produced no
        # valid cell.  High-symmetry systems use ordinary dichotomy directly.
        if not candidates:
            active = [state.box for state in seed_states] if seed_states else roots
            # A contracted low-symmetry state already consumed several logical
            # levels; nevertheless depth is counted afresh here because the
            # stopping criterion is numerical box width.
            for depth in range(self.settings.max_dichotomy_depth + 1):
                self._cancel_check()
                scored: list[_Box] = []
                for box in active:
                    self._cancel_check()
                    evaluated += 1
                    potential, ambiguity = self._interval_score(
                        box.low, box.high, coefficients, qobs, tolerances
                    )
                    if potential < required:
                        continue
                    scored.append(replace(
                        box,
                        potential_matches=potential,
                        ambiguity=ambiguity,
                    ))

                if not scored:
                    break
                scored.sort(key=self._box_sort_key)
                scored = scored[: self.settings.max_boxes_per_level]

                next_active: list[_Box] = []
                for box in scored:
                    self._cancel_check()
                    if (
                        depth >= self.settings.max_dichotomy_depth
                        or self._box_narrow_enough(box)
                    ):
                        leaves.append(box)
                        continue
                    first, second = self._bisect_box(box)
                    next_active.extend((first, second))
                if not next_active:
                    leaves.extend(scored)
                    break
                active = next_active

            if not leaves and active:
                scored = []
                for box in active:
                    self._cancel_check()
                    evaluated += 1
                    potential, ambiguity = self._interval_score(
                        box.low, box.high, coefficients, qobs, tolerances
                    )
                    if potential >= required:
                        scored.append(replace(
                            box, potential_matches=potential, ambiguity=ambiguity
                        ))
                scored.sort(key=self._box_sort_key)
                leaves.extend(scored[: self.settings.max_leaf_boxes])

            leaves.sort(key=self._box_sort_key)
            leaves = leaves[: self.settings.max_leaf_boxes]
            for box in leaves:
                self._cancel_check()
                candidate = self._candidate_from_box(
                    system,
                    box,
                    templates,
                    search,
                    retained,
                    wavelength=wavelength,
                    zero_shift_deg=zero_shift_deg,
                )
                if candidate is not None:
                    candidates.append(candidate)

        candidates = self._deduplicate_and_rank(candidates)
        candidates = candidates[: self.settings.max_candidates_per_system]
        diagnostics = SystemDiagnostics(
            crystal_system=system,
            zero_shift_deg=float(zero_shift_deg),
            reflection_templates=len(templates),
            initial_boxes=len(roots),
            boxes_evaluated=evaluated,
            constraint_seed_states=len(seed_states),
            surviving_leaf_boxes=len(leaves) if leaves else len(seed_states),
            candidate_cells=len(candidates),
        )
        return candidates, diagnostics

    def _low_symmetry_seed_states(
        self,
        roots: Sequence[_Box],
        coefficients: np.ndarray,
        qobs: np.ndarray,
        tolerances: np.ndarray,
        required: int,
    ) -> tuple[list[_SeedState], int]:
        dimension = coefficients.shape[1]
        seed_line_count = min(
            len(qobs), dimension + self.settings.max_spurious_search_lines
        )
        template_limit = min(
            self.settings.low_symmetry_seed_template_limit, len(coefficients)
        )
        states = [
            _SeedState(box, frozenset(), tuple(), 0) for box in roots
        ]
        evaluated = 0

        for peak_index in range(seed_line_count):
            self._cancel_check()
            generated: list[tuple[tuple[object, ...], _SeedState]] = []
            observed = float(qobs[peak_index])
            tolerance = float(tolerances[peak_index])
            for state in states:
                self._cancel_check()
                box = state.box
                midpoint = 0.5 * (box.low + box.high)
                half_width = 0.5 * (box.high - box.low)
                qmid = coefficients[:template_limit] @ midpoint
                qrad = np.abs(coefficients[:template_limit]) @ half_width

                if state.skipped_lines < self.settings.max_spurious_search_lines:
                    skipped = _SeedState(
                        box, state.used_templates,
                        state.assignments + (None,), state.skipped_lines + 1
                    )
                    potential, ambiguity = self._interval_score(
                        box.low, box.high, coefficients, qobs, tolerances
                    )
                    evaluated += 1
                    if potential >= required:
                        scored_box = replace(
                            box, potential_matches=potential, ambiguity=ambiguity
                        )
                        skipped = replace(skipped, box=scored_box)
                        generated.append((self._seed_sort_key(skipped), skipped))

                overlaps = np.flatnonzero(
                    (qmid - qrad <= observed + tolerance)
                    & (qmid + qrad >= observed - tolerance)
                    & (qmid + qrad > 0.0)
                )
                for template_index_np in overlaps:
                    self._cancel_check()
                    template_index = int(template_index_np)
                    if template_index in state.used_templates:
                        continue
                    contracted = self._contract_box_for_reflection(
                        box, coefficients[template_index],
                        observed - tolerance, observed + tolerance
                    )
                    if contracted is None:
                        continue
                    potential, ambiguity = self._interval_score(
                        contracted.low, contracted.high,
                        coefficients, qobs, tolerances
                    )
                    evaluated += 1
                    if potential < required:
                        continue
                    contracted = replace(
                        contracted, potential_matches=potential, ambiguity=ambiguity
                    )
                    child = _SeedState(
                        contracted,
                        state.used_templates | {template_index},
                        state.assignments + (template_index,),
                        state.skipped_lines,
                    )
                    generated.append((self._seed_sort_key(child), child))

            if not generated:
                return [], evaluated

            # Preserve impurity hypotheses instead of letting the much more
            # numerous zero-skip paths crowd them out completely.
            grouped: dict[int, list[tuple[tuple[object, ...], _SeedState]]] = {}
            for item in generated:
                self._cancel_check()
                grouped.setdefault(item[1].skipped_lines, []).append(item)
            cap = self.settings.low_symmetry_seed_state_cap
            groups = max(1, self.settings.max_spurious_search_lines + 1)
            quota = max(1, cap // groups)
            selected: list[tuple[tuple[object, ...], _SeedState]] = []
            leftovers: list[tuple[tuple[object, ...], _SeedState]] = []
            for skip_count in sorted(grouped):
                self._cancel_check()
                items = grouped[skip_count]
                items.sort(key=lambda item: item[0])
                selected.extend(items[:quota])
                leftovers.extend(items[quota:])
            if len(selected) < cap and leftovers:
                leftovers.sort(key=lambda item: item[0])
                selected.extend(leftovers[: cap - len(selected)])
            selected.sort(key=lambda item: item[0])
            states = [item[1] for item in selected[:cap]]

        states.sort(key=self._seed_sort_key)
        return states, evaluated

    def _seed_sort_key(self, state: _SeedState) -> tuple[object, ...]:
        assigned = [value for value in state.assignments if value is not None]
        # Low-angle powder lines normally carry low-index reflections.  The
        # low-index term is a deterministic tie-break/search heuristic; box
        # feasibility remains controlled by the interval bounds themselves.
        index_sum = sum(assigned)
        index_max = max(assigned, default=0)
        return (
            state.skipped_lines,
            index_sum,
            index_max,
            self._box_sort_key(state.box),
        )

    @staticmethod
    def _contract_box_for_reflection(
        box: _Box,
        coefficients: np.ndarray,
        q_low: float,
        q_high: float,
    ) -> _Box | None:
        low = box.low.copy()
        high = box.high.copy()
        widths = high - low
        nonzero = np.flatnonzero(np.abs(coefficients) > 1e-14)
        if nonzero.size == 0:
            return None
        leverage = np.abs(coefficients[nonzero]) * widths[nonzero]
        dimension = int(nonzero[int(np.argmax(leverage))])
        coefficient = float(coefficients[dimension])

        other_low = 0.0
        other_high = 0.0
        for index, value in enumerate(coefficients):
            if index == dimension:
                continue
            value = float(value)
            if value >= 0:
                other_low += value * low[index]
                other_high += value * high[index]
            else:
                other_low += value * high[index]
                other_high += value * low[index]

        if coefficient > 0:
            allowed_low = (q_low - other_high) / coefficient
            allowed_high = (q_high - other_low) / coefficient
        else:
            allowed_low = (q_high - other_low) / coefficient
            allowed_high = (q_low - other_high) / coefficient
        if allowed_low > allowed_high:
            allowed_low, allowed_high = allowed_high, allowed_low
        low[dimension] = max(low[dimension], allowed_low)
        high[dimension] = min(high[dimension], allowed_high)
        if low[dimension] > high[dimension]:
            return None
        return _Box(low, high, box.depth + 1)

    def _initial_boxes(self, system: CrystalSystem, qobs: np.ndarray) -> list[_Box]:
        s = self.settings
        qmax = float(qobs[-1])
        base_low = 1.0 / (s.max_cell_length * s.max_cell_length)
        base_high = min(
            s.reciprocal_diag_max_factor / (s.min_cell_length * s.min_cell_length),
            max(base_low * 1.01, qmax * s.reciprocal_q_upper_factor),
        )

        if system == "hexagonal":
            diagonal_ranges = [
                (4.0 / (3.0 * s.max_cell_length**2),
                 min(4.0 * s.reciprocal_diag_max_factor / (3.0 * s.min_cell_length**2),
                     max(4.0 / (3.0 * s.max_cell_length**2) * 1.01,
                         qmax * s.reciprocal_q_upper_factor))),
                (base_low, base_high),
            ]
        elif system == "cubic":
            diagonal_ranges = [(base_low, base_high)]
        elif system == "tetragonal":
            diagonal_ranges = [(base_low, base_high), (base_low, base_high)]
        elif system == "orthorhombic":
            diagonal_ranges = [(base_low, base_high)] * 3
        elif system == "monoclinic":
            diagonal_ranges = [(base_low, base_high)] * 3
        else:
            diagonal_ranges = [(base_low, base_high)] * 3

        partitioned = [self._log_partitions(lo, hi) for lo, hi in diagonal_ranges]
        boxes: list[_Box] = []
        for diag_intervals in itertools.product(*partitioned):
            self._cancel_check()
            diag_low = [item[0] for item in diag_intervals]
            diag_high = [item[1] for item in diag_intervals]
            if system in {"cubic", "tetragonal", "hexagonal", "orthorhombic"}:
                low = np.asarray(diag_low, dtype=float)
                high = np.asarray(diag_high, dtype=float)
            elif system == "monoclinic":
                correlation = s.off_diagonal_correlation_limit
                cross = correlation * math.sqrt(diag_high[0] * diag_high[2])
                low = np.asarray([*diag_low, -cross], dtype=float)
                high = np.asarray([*diag_high, cross], dtype=float)
            else:
                correlation = s.off_diagonal_correlation_limit
                d23 = correlation * math.sqrt(diag_high[1] * diag_high[2])
                d13 = correlation * math.sqrt(diag_high[0] * diag_high[2])
                d12 = correlation * math.sqrt(diag_high[0] * diag_high[1])
                low = np.asarray([*diag_low, -d23, -d13, -d12], dtype=float)
                high = np.asarray([*diag_high, d23, d13, d12], dtype=float)
            boxes.append(_Box(low=low, high=high, depth=0))
        return boxes

    def _log_partitions(self, low: float, high: float) -> list[tuple[float, float]]:
        if self.settings.initial_diagonal_partitions == 1:
            return [(low, high)]
        edges = np.geomspace(low, high, self.settings.initial_diagonal_partitions + 1)
        return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]

    @staticmethod
    def _interval_score(
        low: np.ndarray,
        high: np.ndarray,
        coefficients: np.ndarray,
        qobs: np.ndarray,
        tolerances: np.ndarray,
    ) -> tuple[int, float]:
        midpoint = 0.5 * (low + high)
        half_width = 0.5 * (high - low)
        qmid = coefficients @ midpoint
        qrad = np.abs(coefficients) @ half_width
        qlow = qmid - qrad
        qhigh = qmid + qrad
        useful = qhigh > 0.0
        if not np.any(useful):
            return 0, math.inf
        qlow = qlow[useful]
        qhigh = qhigh[useful]

        matched = 0
        ambiguity = 0.0
        for observed, tolerance in zip(qobs, tolerances):
            overlaps = (qlow <= observed + tolerance) & (qhigh >= observed - tolerance)
            count = int(np.count_nonzero(overlaps))
            if count:
                matched += 1
                ambiguity += math.log1p(count)
            else:
                ambiguity += 50.0
        half_width = 0.5 * (high - low)
        scale = np.maximum(np.maximum(np.abs(midpoint), half_width), 1e-12)
        width_penalty = float(np.mean((high - low) / scale))
        return matched, ambiguity + 0.05 * width_penalty

    @staticmethod
    def _box_sort_key(box: _Box) -> tuple[float, float, float]:
        midpoint = 0.5 * (box.high + box.low)
        half_width = 0.5 * (box.high - box.low)
        scale = np.maximum(np.maximum(np.abs(midpoint), half_width), 1e-12)
        width = float(np.mean((box.high - box.low) / scale))
        return (-box.potential_matches, box.ambiguity, width)

    def _box_narrow_enough(self, box: _Box) -> bool:
        midpoint = 0.5 * (box.low + box.high)
        relative = (box.high - box.low) / np.maximum(
            np.abs(midpoint), self.settings.q_absolute_tolerance
        )
        return bool(np.max(relative) <= self.settings.min_parameter_relative_width)

    def _bisect_box(self, box: _Box) -> tuple[_Box, _Box]:
        midpoint = 0.5 * (box.low + box.high)
        widths = box.high - box.low
        scale = np.maximum(np.maximum(np.abs(box.low), np.abs(box.high)), self.settings.q_absolute_tolerance)
        dimension = int(np.argmax(widths / scale))
        split = midpoint[dimension]
        high1 = box.high.copy()
        high1[dimension] = split
        low2 = box.low.copy()
        low2[dimension] = split
        return (
            _Box(box.low.copy(), high1, box.depth + 1),
            _Box(low2, box.high.copy(), box.depth + 1),
        )

    # ------------------------------------------------------------------
    # Reciprocal metric parameterizations and reflection templates
    # ------------------------------------------------------------------

    def _reflection_templates(
        self, system: CrystalSystem
    ) -> tuple[_ReflectionTemplate, ...]:
        key = (system, self.settings.hkl_max)
        cached = self._template_cache.get(key)
        if cached is not None:
            return cached

        grouped: dict[tuple[float, ...], list[tuple[int, int, int]]] = {}
        m = self.settings.hkl_max
        for h in range(-m, m + 1):
            self._cancel_check()
            for k in range(-m, m + 1):
                self._cancel_check()
                for l in range(-m, m + 1):
                    self._cancel_check()
                    if h == k == l == 0:
                        continue
                    # Friedel mates have identical Q; retain one representative.
                    first = next(value for value in (h, k, l) if value != 0)
                    if first < 0:
                        continue
                    coefficients = self._coefficients(system, h, k, l)
                    key_coeff = tuple(float(value) for value in coefficients)
                    grouped.setdefault(key_coeff, []).append((h, k, l))

        templates = tuple(
            _ReflectionTemplate(coeff, tuple(sorted(hkls)))
            for coeff, hkls in grouped.items()
        )
        # A stable low-index ordering helps deterministic matching.
        templates = tuple(sorted(
            templates,
            key=lambda item: (
                sum(abs(value) for value in item.hkls[0]),
                sum(value * value for value in item.hkls[0]),
                item.hkls[0],
            ),
        ))
        self._template_cache[key] = templates
        return templates

    @staticmethod
    def _coefficients(
        system: CrystalSystem, h: int, k: int, l: int
    ) -> tuple[float, ...]:
        if system == "cubic":
            return (float(h * h + k * k + l * l),)
        if system == "tetragonal":
            return (float(h * h + k * k), float(l * l))
        if system == "hexagonal":
            return (float(h * h + h * k + k * k), float(l * l))
        if system == "orthorhombic":
            return (float(h * h), float(k * k), float(l * l))
        if system == "monoclinic":
            return (float(h * h), float(k * k), float(l * l), float(2 * h * l))
        return (
            float(h * h),
            float(k * k),
            float(l * l),
            float(2 * k * l),
            float(2 * h * l),
            float(2 * h * k),
        )

    @staticmethod
    def _metric_from_parameters(system: CrystalSystem, parameters: np.ndarray) -> np.ndarray:
        p = np.asarray(parameters, dtype=float)
        if system == "cubic":
            return np.diag([p[0], p[0], p[0]])
        if system == "tetragonal":
            return np.diag([p[0], p[0], p[1]])
        if system == "hexagonal":
            return np.asarray(
                ((p[0], 0.5 * p[0], 0.0),
                 (0.5 * p[0], p[0], 0.0),
                 (0.0, 0.0, p[1])),
                dtype=float,
            )
        if system == "orthorhombic":
            return np.diag(p[:3])
        if system == "monoclinic":
            return np.asarray(
                ((p[0], 0.0, p[3]),
                 (0.0, p[1], 0.0),
                 (p[3], 0.0, p[2])),
                dtype=float,
            )
        return np.asarray(
            ((p[0], p[5], p[4]),
             (p[5], p[1], p[3]),
             (p[4], p[3], p[2])),
            dtype=float,
        )

    def _metric_to_cell(self, metric: np.ndarray) -> UnitCell | None:
        try:
            eigenvalues = np.linalg.eigvalsh(metric)
            if float(eigenvalues[0]) <= 1e-12:
                return None
            direct = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            return None
        lengths = np.sqrt(np.diag(direct))
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
            return None

        def angle(i: int, j: int) -> float:
            cosine = float(direct[i, j] / (lengths[i] * lengths[j]))
            cosine = max(-1.0, min(1.0, cosine))
            return math.degrees(math.acos(cosine))

        return UnitCell(
            float(lengths[0]),
            float(lengths[1]),
            float(lengths[2]),
            angle(1, 2),
            angle(0, 2),
            angle(0, 1),
        )

    def _cell_is_plausible(self, system: CrystalSystem, cell: UnitCell) -> bool:
        s = self.settings
        lengths = (cell.a, cell.b, cell.c)
        if any(value < 0.98 * s.min_cell_length or value > 1.02 * s.max_cell_length for value in lengths):
            return False
        if cell.volume <= 0 or cell.volume > 1.02 * s.max_cell_volume:
            return False
        if any(not s.min_cell_angle_deg <= value <= s.max_cell_angle_deg
               for value in (cell.alpha, cell.beta, cell.gamma)):
            return False
        atol = 0.08
        if system == "cubic":
            return (
                abs(cell.a - cell.b) <= atol * max(cell.a, cell.b)
                and abs(cell.b - cell.c) <= atol * max(cell.b, cell.c)
                and all(abs(value - 90.0) < 0.2 for value in (cell.alpha, cell.beta, cell.gamma))
            )
        if system == "tetragonal":
            return (
                abs(cell.a - cell.b) <= atol * max(cell.a, cell.b)
                and all(abs(value - 90.0) < 0.2 for value in (cell.alpha, cell.beta, cell.gamma))
            )
        if system == "hexagonal":
            return (
                abs(cell.a - cell.b) <= atol * max(cell.a, cell.b)
                and abs(cell.alpha - 90.0) < 0.2
                and abs(cell.beta - 90.0) < 0.2
                and min(abs(cell.gamma - 120.0), abs(cell.gamma - 60.0)) < 0.2
            )
        if system == "orthorhombic":
            return all(abs(value - 90.0) < 0.2 for value in (cell.alpha, cell.beta, cell.gamma))
        if system == "monoclinic":
            return abs(cell.alpha - 90.0) < 0.2 and abs(cell.gamma - 90.0) < 0.2
        return True

    # ------------------------------------------------------------------
    # Leaf refinement, matching and figures of merit
    # ------------------------------------------------------------------

    def _candidate_from_box(
        self,
        system: CrystalSystem,
        box: _Box,
        templates: Sequence[_ReflectionTemplate],
        search: Sequence[_PreparedPeak],
        retained: Sequence[_PreparedPeak],
        *,
        wavelength: float | None,
        zero_shift_deg: float,
    ) -> BoultifLouerCandidate | None:
        coefficients = np.asarray([item.coefficients for item in templates], dtype=float)
        midpoint = 0.5 * (box.low + box.high)
        half_width = 0.5 * (box.high - box.low)
        qmid = coefficients @ midpoint
        qrad = np.abs(coefficients) @ half_width

        initial_rows: list[np.ndarray] = []
        initial_values: list[float] = []
        used: set[int] = set()
        for peak in search:
            self._cancel_check()
            overlaps = np.flatnonzero(
                (qmid - qrad <= peak.q + peak.q_tolerance)
                & (qmid + qrad >= peak.q - peak.q_tolerance)
                & (qmid + qrad > 0)
            )
            if overlaps.size == 0:
                continue
            ordered = overlaps[np.argsort(np.abs(qmid[overlaps] - peak.q))]
            chosen = next((int(value) for value in ordered if int(value) not in used), None)
            if chosen is None:
                continue
            used.add(chosen)
            initial_rows.append(coefficients[chosen])
            initial_values.append(peak.q)

        dimension = len(midpoint)
        if len(initial_rows) < max(dimension, self.settings.min_required_peaks):
            return None
        matrix = np.asarray(initial_rows, dtype=float)
        values = np.asarray(initial_values, dtype=float)
        if np.linalg.matrix_rank(matrix) < dimension:
            return None
        try:
            parameters, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return self._candidate_from_parameters(
            system, np.asarray(parameters, dtype=float), templates, search, retained,
            wavelength=wavelength, zero_shift_deg=zero_shift_deg
        )

    def _candidate_from_seed_state(
        self,
        system: CrystalSystem,
        state: _SeedState,
        templates: Sequence[_ReflectionTemplate],
        search: Sequence[_PreparedPeak],
        retained: Sequence[_PreparedPeak],
        *,
        wavelength: float | None,
        zero_shift_deg: float,
    ) -> BoultifLouerCandidate | None:
        rows: list[tuple[float, ...]] = []
        values: list[float] = []
        for peak_index, template_index in enumerate(state.assignments):
            self._cancel_check()
            if template_index is None:
                continue
            rows.append(templates[template_index].coefficients)
            values.append(search[peak_index].q)
        dimension = len(state.box.low)
        if len(rows) < dimension:
            return None
        matrix = np.asarray(rows, dtype=float)
        if np.linalg.matrix_rank(matrix) < dimension:
            return None
        try:
            parameters, *_ = np.linalg.lstsq(
                matrix, np.asarray(values, dtype=float), rcond=None
            )
        except np.linalg.LinAlgError:
            return None
        # Reject a point solution that escaped its interval hypothesis because
        # of an ill-conditioned assignment.  A tiny numerical margin is enough.
        margin = np.maximum(1e-10, 1e-7 * np.maximum(np.abs(parameters), 1.0))
        if np.any(parameters < state.box.low - margin) or np.any(parameters > state.box.high + margin):
            return None
        return self._candidate_from_parameters(
            system, np.asarray(parameters, dtype=float), templates, search, retained,
            wavelength=wavelength, zero_shift_deg=zero_shift_deg
        )

    def _candidate_from_parameters(
        self,
        system: CrystalSystem,
        parameters: np.ndarray,
        templates: Sequence[_ReflectionTemplate],
        search: Sequence[_PreparedPeak],
        retained: Sequence[_PreparedPeak],
        *,
        wavelength: float | None,
        zero_shift_deg: float,
    ) -> BoultifLouerCandidate | None:
        refined = self._refine_parameters(
            system, np.asarray(parameters, dtype=float), templates, search
        )
        if refined is None:
            return None
        parameters, search_matches = refined
        required = max(
            self.settings.min_required_peaks,
            len(search) - self.settings.max_spurious_search_lines,
        )
        if len(search_matches) < required:
            return None

        metric = self._metric_from_parameters(system, parameters)
        cell = self._metric_to_cell(metric)
        if cell is None or not self._cell_is_plausible(system, cell):
            return None

        all_matches = self._match_peaks(parameters, templates, retained, tolerance_factor=1.0)
        indexed_lines, unindexed_lines = self._public_line_results(
            parameters, templates, retained, all_matches, wavelength=wavelength
        )
        search_ids = {peak.original_index for peak in search}
        indexed_search = sum(line.observed_index in search_ids for line in indexed_lines)
        if indexed_search < required:
            return None

        residuals = [line.normalized_residual for line in indexed_lines]
        mean_abs_delta_q = (
            float(np.mean([abs(line.delta_q) for line in indexed_lines]))
            if indexed_lines else math.inf
        )
        rms = float(math.sqrt(np.mean(np.square(residuals)))) if residuals else math.inf
        m20, f20, n20, q20 = self._figures_of_merit(
            parameters, templates, retained, all_matches, wavelength=wavelength
        )
        return BoultifLouerCandidate(
            rank=0,
            crystal_system=system,
            cell=cell,
            reciprocal_metric=tuple(tuple(float(value) for value in row) for row in metric),
            indexed_lines=tuple(indexed_lines),
            unindexed_lines=tuple(unindexed_lines),
            indexed_search_lines=indexed_search,
            indexed_total_lines=len(indexed_lines),
            spurious_search_lines=len(search) - indexed_search,
            mean_abs_delta_q=mean_abs_delta_q,
            rms_normalized_residual=rms,
            zero_shift_deg=float(zero_shift_deg),
            m20=m20,
            f20=f20,
            n20_possible=n20,
            q20=q20,
        )

    def _refine_parameters(
        self,
        system: CrystalSystem,
        initial: np.ndarray,
        templates: Sequence[_ReflectionTemplate],
        peaks: Sequence[_PreparedPeak],
    ) -> tuple[np.ndarray, list[_Match]] | None:
        parameters = np.asarray(initial, dtype=float)
        matches: list[_Match] = []
        for iteration in range(self.settings.refinement_iterations + 1):
            self._cancel_check()
            tolerance_factor = (
                self.settings.relaxed_refinement_tolerance_factor
                if iteration == 0 else max(1.0, 2.5 / iteration)
            )
            matches = self._match_peaks(
                parameters,
                templates,
                peaks,
                tolerance_factor=tolerance_factor,
            )
            if len(matches) < max(len(parameters), self.settings.min_required_peaks):
                return None
            rows = np.asarray(
                [templates[match.template_index].coefficients for match in matches],
                dtype=float,
            )
            values = np.asarray([peaks[match.peak_index].q for match in matches], dtype=float)
            weights = np.asarray(
                [1.0 / max(peaks[match.peak_index].q_tolerance, 1e-12) for match in matches],
                dtype=float,
            )
            weighted_rows = rows * weights[:, None]
            weighted_values = values * weights
            if np.linalg.matrix_rank(weighted_rows) < len(parameters):
                return None
            try:
                updated, *_ = np.linalg.lstsq(weighted_rows, weighted_values, rcond=None)
            except np.linalg.LinAlgError:
                return None
            metric = self._metric_from_parameters(system, updated)
            try:
                if float(np.linalg.eigvalsh(metric)[0]) <= 1e-12:
                    return None
            except np.linalg.LinAlgError:
                return None
            if np.linalg.norm(updated - parameters) <= 1e-10 * max(1.0, np.linalg.norm(parameters)):
                parameters = updated
                break
            parameters = updated

        strict = self._match_peaks(parameters, templates, peaks, tolerance_factor=1.0)
        return parameters, strict

    def _match_peaks(
        self,
        parameters: np.ndarray,
        templates: Sequence[_ReflectionTemplate],
        peaks: Sequence[_PreparedPeak],
        *,
        tolerance_factor: float,
    ) -> list[_Match]:
        coefficients = np.asarray([item.coefficients for item in templates], dtype=float)
        qcalc = coefficients @ parameters
        valid = np.isfinite(qcalc) & (qcalc > 1e-12)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return []
        order = indices[np.argsort(qcalc[indices])]
        sorted_q = qcalc[order]
        used: set[int] = set()
        matches: list[_Match] = []

        # Match the most precise observed lines first, then restore peak order.
        peak_order = sorted(
            range(len(peaks)),
            key=lambda i: (peaks[i].q_tolerance, peaks[i].q),
        )
        for peak_index in peak_order:
            self._cancel_check()
            peak = peaks[peak_index]
            insertion = int(np.searchsorted(sorted_q, peak.q))
            neighborhood = range(max(0, insertion - 4), min(len(order), insertion + 5))
            best: tuple[float, int] | None = None
            for local in neighborhood:
                self._cancel_check()
                template_index = int(order[local])
                if template_index in used:
                    continue
                delta = float(qcalc[template_index] - peak.q)
                tolerance = tolerance_factor * peak.q_tolerance
                if abs(delta) <= tolerance:
                    score = abs(delta) / max(peak.q_tolerance, 1e-12)
                    if best is None or score < best[0]:
                        best = (score, template_index)
            if best is not None:
                template_index = best[1]
                used.add(template_index)
                matches.append(_Match(
                    peak_index=peak_index,
                    template_index=template_index,
                    q_calculated=float(qcalc[template_index]),
                    delta_q=float(qcalc[template_index] - peak.q),
                ))
        matches.sort(key=lambda item: item.peak_index)
        return matches

    def _public_line_results(
        self,
        parameters: np.ndarray,
        templates: Sequence[_ReflectionTemplate],
        peaks: Sequence[_PreparedPeak],
        matches: Sequence[_Match],
        *,
        wavelength: float | None,
    ) -> tuple[list[IndexedLine], list[UnindexedLine]]:
        by_peak = {match.peak_index: match for match in matches}
        indexed: list[IndexedLine] = []
        unindexed: list[UnindexedLine] = []
        for peak_index, peak in enumerate(peaks):
            self._cancel_check()
            match = by_peak.get(peak_index)
            if match is None:
                unindexed.append(UnindexedLine(
                    observed_index=peak.original_index,
                    observed_position=peak.original_position,
                    observed_d=peak.d,
                    observed_q=peak.q,
                    intensity=peak.intensity,
                    label=peak.label,
                ))
                continue
            template = templates[match.template_index]
            h, k, l = template.hkls[0]
            dcalc = 1.0 / math.sqrt(match.q_calculated)
            calculated_two_theta = (
                self._d_to_two_theta(dcalc, wavelength) if wavelength is not None else None
            )
            delta_two_theta = (
                None
                if peak.two_theta is None or calculated_two_theta is None
                else peak.two_theta - calculated_two_theta
            )
            indexed.append(IndexedLine(
                observed_index=peak.original_index,
                observed_position=peak.original_position,
                observed_d=peak.d,
                observed_q=peak.q,
                h=h,
                k=k,
                l=l,
                equivalent_hkls=template.hkls,
                calculated_d=dcalc,
                calculated_q=match.q_calculated,
                delta_q=-match.delta_q,
                normalized_residual=abs(match.delta_q) / max(peak.q_tolerance, 1e-12),
                observed_two_theta=peak.two_theta,
                calculated_two_theta=calculated_two_theta,
                delta_two_theta=delta_two_theta,
                intensity=peak.intensity,
                label=peak.label,
            ))
        return indexed, unindexed

    def _figures_of_merit(
        self,
        parameters: np.ndarray,
        templates: Sequence[_ReflectionTemplate],
        peaks: Sequence[_PreparedPeak],
        matches: Sequence[_Match],
        *,
        wavelength: float | None,
    ) -> tuple[float | None, float | None, int | None, float | None]:
        n = min(20, len(peaks))
        if n < 20:
            return None, None, None, None
        first = peaks[:n]
        by_peak = {match.peak_index: match for match in matches}
        if any(index not in by_peak for index in range(n)):
            return None, None, None, float(first[-1].q)
        q20 = float(first[-1].q)
        q20_limit = q20 + float(first[-1].q_tolerance)
        calculated = self._predicted_groups(parameters, templates, q20_limit)
        n_possible = sum(value <= q20_limit for value, _hkls in calculated)
        if n_possible <= 0:
            return None, None, None, q20

        delta_q = [abs(by_peak[index].delta_q) for index in range(n)]
        mean_delta_q = float(np.mean(delta_q))
        m20 = (
            math.inf if mean_delta_q <= 1e-16
            else q20 / (2.0 * mean_delta_q * n_possible)
        )

        f20: float | None = None
        if wavelength is not None and all(peak.two_theta is not None for peak in first):
            delta_tt: list[float] = []
            for index, peak in enumerate(first):
                self._cancel_check()
                qcalc = by_peak[index].q_calculated
                dcalc = 1.0 / math.sqrt(qcalc)
                ttcalc = self._d_to_two_theta(dcalc, wavelength)
                if ttcalc is None or peak.two_theta is None:
                    delta_tt = []
                    break
                delta_tt.append(abs(ttcalc - peak.two_theta))
            if delta_tt:
                mean_delta_tt = float(np.mean(delta_tt))
                f20 = (
                    math.inf if mean_delta_tt <= 1e-16
                    else (1.0 / mean_delta_tt) * (20.0 / n_possible)
                )
        return float(m20), None if f20 is None else float(f20), int(n_possible), q20

    def _predicted_groups(
        self,
        parameters: np.ndarray,
        templates: Sequence[_ReflectionTemplate],
        qmax: float,
    ) -> list[tuple[float, tuple[tuple[int, int, int], ...]]]:
        coefficients = np.asarray([item.coefficients for item in templates], dtype=float)
        qvalues = coefficients @ parameters
        rows = [
            (float(q), templates[index].hkls)
            for index, q in enumerate(qvalues)
            if math.isfinite(float(q)) and 1e-12 < float(q) <= qmax
        ]
        rows.sort(key=lambda item: item[0])
        groups: list[list[object]] = []
        for q, hkls in rows:
            self._cancel_check()
            tolerance = max(1e-10, 1e-8 * q)
            if not groups or abs(q - float(groups[-1][0])) > tolerance:
                groups.append([q, list(hkls)])
            else:
                groups[-1][1].extend(hkls)  # type: ignore[index]
        return [
            (float(q), tuple(sorted(set(hkls))))
            for q, hkls in groups
        ]

    def _refine_zero_shift_candidate(
        self,
        candidate: BoultifLouerCandidate,
        raw_peaks: Sequence[BoultifLouerPeak],
        wavelength: float,
    ) -> BoultifLouerCandidate:
        """Alternate local zero-point and metric refinement.

        The correction is added to observed 2theta.  This is an independent
        one-dimensional refinement around a dichotomy solution; it implements
        the published *capability* to refine zero point without claiming to
        reproduce DICVOL04's private numerical procedure.
        """

        current = candidate
        metric = np.asarray(candidate.reciprocal_metric, dtype=float)
        parameters = self._parameters_from_metric(candidate.crystal_system, metric)
        if self.settings.search_zero_shift and self.settings.zero_shift_steps > 1:
            coarse_step = (
                2.0 * self.settings.zero_shift_half_range_deg
                / (self.settings.zero_shift_steps - 1)
            )
            step = max(coarse_step * 0.5, 0.005)
        else:
            step = max(self.settings.two_theta_tolerance_deg, 0.01)

        for _iteration in range(self.settings.zero_refine_iterations):
            self._cancel_check()
            trials: list[BoultifLouerCandidate] = []
            for shift in (current.zero_shift_deg - step, current.zero_shift_deg,
                          current.zero_shift_deg + step):
                self._cancel_check()
                prepared = self._prepare_peaks(
                    raw_peaks, input_kind="two_theta", wavelength=wavelength,
                    zero_shift_deg=float(shift)
                )[: self.settings.max_input_peaks]
                if len(prepared) < self.settings.min_required_peaks:
                    continue
                search_count = min(self.settings.search_peak_count, len(prepared))
                search = prepared[:search_count]
                templates = self._reflection_templates(candidate.crystal_system)
                trial = self._candidate_from_parameters(
                    candidate.crystal_system, parameters, templates, search, prepared,
                    wavelength=wavelength, zero_shift_deg=float(shift)
                )
                if trial is not None:
                    trials.append(trial)
            if not trials:
                break
            trials.sort(key=self._candidate_sort_key)
            # Within an already selected cell family, residual is the useful
            # discriminator for the zero correction once match count/N20 tie.
            best_key = lambda item: (
                -item.indexed_search_lines, item.spurious_search_lines,
                item.n20_possible if item.n20_possible is not None else 10**9,
                item.rms_normalized_residual, abs(item.zero_shift_deg),
            )
            current = min(trials, key=best_key)
            parameters = self._parameters_from_metric(
                current.crystal_system, np.asarray(current.reciprocal_metric, dtype=float)
            )
            step *= 0.35
        return current

    # ------------------------------------------------------------------
    # Result ranking / deduplication
    # ------------------------------------------------------------------

    def _deduplicate_and_rank(
        self, candidates: Sequence[BoultifLouerCandidate]
    ) -> list[BoultifLouerCandidate]:
        ordered = sorted(candidates, key=self._candidate_sort_key)
        kept: list[BoultifLouerCandidate] = []
        signatures: set[tuple[float, ...]] = set()
        for candidate in ordered:
            self._cancel_check()
            metric = np.asarray(candidate.reciprocal_metric, dtype=float)
            signature = metric_signature(metric)
            if signature and signature in signatures:
                continue
            if signature:
                signatures.add(signature)
            kept.append(candidate)
        return kept

    @staticmethod
    def _candidate_sort_key(candidate: BoultifLouerCandidate) -> tuple[float, ...]:
        system_rank = CRYSTAL_SYSTEMS.index(candidate.crystal_system)
        if candidate.m20 is None:
            m20_key = math.inf
        elif math.isinf(candidate.m20):
            m20_key = -math.inf
        else:
            m20_key = -candidate.m20
        if candidate.f20 is None:
            f20_key = math.inf
        elif math.isinf(candidate.f20):
            f20_key = -math.inf
        else:
            f20_key = -candidate.f20
        # Primitive/simple metrics normally generate fewer possible powder
        # lines before Q20 than supercells that merely index the observations
        # as a subset.  Use N20 before the numerical FOM tie-breakers.
        n20_key = float(candidate.n20_possible if candidate.n20_possible is not None else 10**9)
        return (
            -candidate.indexed_search_lines,
            candidate.spurious_search_lines,
            n20_key,
            m20_key,
            f20_key,
            candidate.rms_normalized_residual,
            float(system_rank),
            candidate.volume,
        )

    @staticmethod
    def _parameters_from_metric(system: CrystalSystem, metric: np.ndarray) -> np.ndarray:
        if system == "cubic":
            return np.asarray([metric[0, 0]])
        if system == "tetragonal":
            return np.asarray([metric[0, 0], metric[2, 2]])
        if system == "hexagonal":
            return np.asarray([metric[0, 0], metric[2, 2]])
        if system == "orthorhombic":
            return np.asarray([metric[0, 0], metric[1, 1], metric[2, 2]])
        if system == "monoclinic":
            return np.asarray([metric[0, 0], metric[1, 1], metric[2, 2], metric[0, 2]])
        return np.asarray([
            metric[0, 0], metric[1, 1], metric[2, 2],
            metric[1, 2], metric[0, 2], metric[0, 1],
        ])


# Friendly aliases for callers that prefer the generic method name.
DichotomyPeak = BoultifLouerPeak
DichotomySettings = BoultifLouerSettings
DichotomyCandidate = BoultifLouerCandidate
DichotomyDiagnostics = BoultifLouerDiagnostics
DichotomyResult = BoultifLouerResult
DichotomyIndexer = BoultifLouerIndexer


def index_boultif_louer(
    positions: Sequence[float],
    *,
    wavelength: float | None = None,
    input_kind: InputKind = "two_theta",
    intensities: Sequence[float] | None = None,
    uncertainties: Sequence[float] | None = None,
    labels: Sequence[str | None] | None = None,
    settings: BoultifLouerSettings | None = None,
) -> BoultifLouerResult:
    """Convenience function for array-like peak lists.

    Parameters
    ----------
    positions:
        Peak positions.  2theta is in degrees; d in Angstrom; q is 1/d² A⁻².
    wavelength:
        Wavelength in Angstrom.  Required for 2theta input and useful for F20
        when d/q positions are supplied.
    input_kind:
        ``"two_theta"``, ``"d"`` or ``"q"``.
    intensities, uncertainties, labels:
        Optional metadata aligned one-to-one with ``positions``.
    settings:
        Search limits and dichotomy controls.
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
        BoultifLouerPeak(
            position=float(position),
            intensity=None if intensities is None else float(intensities[index]),
            uncertainty=None if uncertainties is None else float(uncertainties[index]),
            label=None if labels is None else labels[index],
        )
        for index, position in enumerate(positions)
    ]
    return BoultifLouerIndexer(settings).index(
        peaks,
        input_kind=input_kind,
        wavelength=wavelength,
    )


def index_dichotomy(*args, **kwargs) -> BoultifLouerResult:
    """Alias for :func:`index_boultif_louer`."""

    return index_boultif_louer(*args, **kwargs)


__all__ = [
    "BOULTIF_LOUER_1991_PUBLICATION",
    "BOULTIF_LOUER_PUBLICATION",
    "BoultifLouerCandidate",
    "BoultifLouerDiagnostics",
    "BoultifLouerIndexer",
    "BoultifLouerPeak",
    "BoultifLouerResult",
    "BoultifLouerSettings",
    "CRYSTAL_SYSTEMS",
    "CrystalSystem",
    "DE_WOLFF_PUBLICATION",
    "DichotomyCandidate",
    "DichotomyDiagnostics",
    "DichotomyIndexer",
    "DichotomyPeak",
    "DichotomyResult",
    "DichotomySettings",
    "F20_REFERENCE_DOI",
    "IndexedLine",
    "InputKind",
    "LOUER_LOUER_1972_PUBLICATION",
    "METHOD_DESCRIPTION",
    "METHOD_ID",
    "METHOD_NAME",
    "METHOD_PUBLICATIONS",
    "M20_REFERENCE_DOI",
    "ORIGIN_REFERENCE_DOI",
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
    "SMITH_SNYDER_PUBLICATION",
    "SystemDiagnostics",
    "THEORY_REFERENCE_DOI",
    "UnitCell",
    "UnindexedLine",
    "index_boultif_louer",
    "index_dichotomy",
]
