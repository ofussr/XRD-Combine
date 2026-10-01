"""Shared scientific input, bibliography and result objects (no GUI imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math

InputKind = Literal["two_theta", "d", "q"]
Matrix = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Publication:
    author: str
    title: str
    journal: str
    year: int
    volume: int
    pages: str
    doi: str

    @property
    def authors(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.author.split(";"))

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}"

    @property
    def citation(self) -> str:
        return (f'{self.author}, "{self.title}", {self.journal} '
                f'{self.volume} ({self.year}) {self.pages}. DOI: {self.doi}')

    @property
    def bibtex(self) -> str:
        return (f"@article{{indexing{self.year}{self.doi.replace('/', '_')},\n"
                f"  author = {{{' and '.join(self.authors)}}},\n"
                f"  title = {{{self.title}}},\n  journal = {{{self.journal}}},\n"
                f"  year = {{{self.year}}},\n  volume = {{{self.volume}}},\n"
                f"  pages = {{{self.pages}}},\n  doi = {{{self.doi}}}\n}}")


@dataclass(frozen=True)
class Peak:
    position: float
    intensity: float | None = None
    uncertainty: float | None = None  # Centre error, in the input coordinate; never FWHM.
    label: str | None = None
    peak_id: int | str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class UnitCell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @property
    def volume(self) -> float:
        ar, br, gr = map(math.radians, (self.alpha, self.beta, self.gamma))
        factor = (1 + 2 * math.cos(ar) * math.cos(br) * math.cos(gr)
                  - math.cos(ar)**2 - math.cos(br)**2 - math.cos(gr)**2)
        return self.a * self.b * self.c * math.sqrt(max(0.0, factor))

    def as_tuple(self) -> tuple[float, ...]:
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma


@dataclass(frozen=True)
class IndexedLine:
    observed_index: int
    observed_position: float
    observed_d: float
    observed_q: float
    h: int
    k: int
    l: int
    calculated_d: float
    calculated_q: float
    delta_q: float  # Observed minus calculated, in both methods.
    normalized_residual: float
    intensity: float | None = None
    label: str | None = None
    equivalent_hkls: tuple[tuple[int, int, int], ...] = ()
    observed_two_theta: float | None = None
    calculated_two_theta: float | None = None
    delta_two_theta: float | None = None
    peak_id: int | str | None = None


@dataclass(frozen=True)
class UnindexedLine:
    observed_index: int
    observed_position: float
    observed_d: float
    observed_q: float
    intensity: float | None = None
    label: str | None = None
    peak_id: int | str | None = None


@dataclass(frozen=True)
class ReducedCell:
    cell: UnitCell
    direct_metric: Matrix
    reciprocal_metric: Matrix
    basis_transform: tuple[tuple[int, int, int], ...]

    def transform_hkl(self, hkl: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(sum(row[j] * hkl[j] for j in range(3))
                     for row in self.basis_transform)


@dataclass(frozen=True)
class Candidate:
    rank: int
    crystal_system: str
    cell: UnitCell  # Original basis, matching the supplied hkl and Cell Phase.
    direct_metric: Matrix
    reciprocal_metric: Matrix
    reduced: ReducedCell
    indexed_lines: tuple[IndexedLine, ...]
    unindexed_lines: tuple[UnindexedLine, ...]
    indexed_search_lines: int
    m20: float | None
    f20: float | None
    n20: int | None
    q20: float | None
    zero_shift_deg: float
    equivalent_ranks: tuple[int, ...] = ()
    diagnostics: dict = field(default_factory=dict, compare=False)

    @property
    def volume(self) -> float:
        return self.cell.volume


@dataclass(frozen=True)
class IndexingRequest:
    peaks: tuple[Peak, ...]
    method: str
    input_kind: InputKind = "two_theta"
    wavelength: float | None = None
    settings: object | None = None


@dataclass(frozen=True)
class IndexingResult:
    method_id: str
    method_name: str
    publications: tuple[Publication, ...]
    request: IndexingRequest
    candidates: tuple[Candidate, ...]
    diagnostics: dict


class IndexingError(ValueError):
    def __init__(self, code: str, **details):
        super().__init__(code)
        self.code = code
        self.details = details


class IndexingCancelled(Exception):
    """Raised at a numerical loop boundary on a cancellation request."""
