from dataclasses import replace
import math

import numpy as np
import pytest

from xrd_workbench.indexing.models import IndexedLine, Peak, IndexingRequest
from xrd_workbench.indexing.service import run_indexing
from xrd_workbench.indexing.indexing_visser import VisserIndexer
from indexing_fixtures import CELLS, q_lines


def test_perfect_cubic_candidate_and_infinite_m20_rank_first():
    peaks = tuple(Peak(q) for q in q_lines(CELLS["cubic"]))
    result = run_indexing(IndexingRequest(peaks, "visser", "q"))
    assert result.candidates
    assert result.candidates[0].m20 == math.inf
    np.testing.assert_allclose(result.candidates[0].cell.as_tuple(), CELLS["cubic"], atol=1e-7)
    assert len(result.candidates[0].indexed_lines) == 25
    assert result.publications[0].bibtex


def test_m20_uses_first_twenty_supplied_lines_and_all_calculated_lines():
    peaks = VisserIndexer()._prepare_peaks([Peak(q) for q in np.arange(1, 26) / 10], input_kind="q", wavelength=None)
    lines = [IndexedLine(i, p.original_position, p.d, p.q, 1, 0, 0, p.d, p.q - .0001,
                         .0001, 0) for i, p in enumerate(peaks)]
    groups = [(q, [(1, 0, 0)]) for q in np.arange(1, 41) / 20]
    m20, n20, q20 = VisserIndexer._m20(lines, groups, peaks)
    assert q20 == 2
    assert n20 == 40  # Includes calculated lines not represented by an observed peak.
    assert m20 == pytest.approx(2 / (2 * .0001 * 40))
    assert VisserIndexer._m20(lines[1:], groups, peaks) == (None, None, None)


@pytest.mark.parametrize("system", CELLS)
def test_six_system_line_review_and_signed_residuals(system):
    # Verify the final metric/line review independently of heuristic discovery.
    from xrd_workbench.indexing.indexing_visser import _Trial
    from indexing_fixtures import reciprocal
    indexer = VisserIndexer()
    values = q_lines(CELLS[system])
    peaks = indexer._prepare_peaks([Peak(x) for x in values], input_kind="q", wavelength=None)
    candidate = indexer._build_candidates([_Trial(reciprocal(CELLS[system]), 20, 0)], peaks, search_count=20)[0]
    assert len(candidate.indexed_lines) == 25
    for line in candidate.indexed_lines:
        assert line.delta_q == pytest.approx(line.observed_q - line.calculated_q)
