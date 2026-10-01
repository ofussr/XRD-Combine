from dataclasses import replace
import math

import numpy as np
import pytest

from xrd_workbench.indexing.models import Peak, IndexingRequest
from xrd_workbench.indexing.service import run_indexing
from xrd_workbench.indexing.reduction import reduce_metric
from xrd_workbench.indexing.indexing_boultif_louer import BoultifLouerSettings
from indexing_fixtures import CELLS, q_lines, reciprocal, NIST_SI_ANGLES, NIST_SI_WAVELENGTH, NIST_SI_LATTICE


@pytest.mark.parametrize("system", CELLS)
def test_independent_six_systems(system):
    cell = CELLS[system]
    settings = BoultifLouerSettings(systems=(system,), min_cell_length=3.8,
        max_cell_length=max(cell[:3]) + .2, min_cell_angle_deg=70, max_cell_angle_deg=125,
        max_spurious_search_lines=0, hkl_max=4)
    result = run_indexing(IndexingRequest(tuple(Peak(q) for q in q_lines(cell)),
                          "boultif_louer", "q", settings=settings))
    expected = np.array(reduce_metric(reciprocal(cell)).direct_metric)
    matching = [c for c in result.candidates if np.allclose(c.reduced.direct_metric, expected, atol=1e-5)]
    assert matching
    assert len(matching[0].indexed_lines) == 25
    assert result.diagnostics["exhaustive_search"] is False


def test_independent_nist_silicon_published_positions():
    settings = BoultifLouerSettings(systems=("cubic",), min_cell_length=5.3, max_cell_length=5.5, hkl_max=8)
    peaks = tuple(Peak(x, uncertainty=.001) for x in NIST_SI_ANGLES)
    result = run_indexing(IndexingRequest(peaks, "boultif_louer", wavelength=NIST_SI_WAVELENGTH, settings=settings))
    best = result.candidates[0]
    assert best.cell.a == pytest.approx(NIST_SI_LATTICE, abs=1e-5)
    assert len(best.indexed_lines) == 11
    assert best.m20 is None
    assert max(abs(l.delta_two_theta) for l in best.indexed_lines) < .001


def test_noise_missing_spurious_and_residual_convention():
    random = np.random.default_rng(1969)
    original = q_lines(CELLS["cubic"], count=30)
    qs = [q + noise for q, noise in zip(original, random.normal(0, 1e-6, len(original)))]
    del qs[6]
    qs.insert(3, .21123)
    settings = BoultifLouerSettings(systems=("cubic",), min_cell_length=3.8, max_cell_length=4.2)
    result = run_indexing(IndexingRequest(tuple(Peak(q, uncertainty=5e-6) for q in qs),
                         "boultif_louer", "q", settings=settings))
    best = result.candidates[0]
    assert best.cell.a == pytest.approx(4, abs=1e-4)
    assert len(best.unindexed_lines) == 1
    for line in best.indexed_lines:
        assert line.delta_q == pytest.approx(line.observed_q - line.calculated_q)
    assert best.m20 is None  # The first twenty supplied lines include an impurity.


def test_zero_shift_is_added_to_observed_angles():
    qs = q_lines(CELLS["cubic"])
    angles = [math.degrees(2 * math.asin(math.sqrt(q) / 2)) + .1 for q in qs]
    settings = BoultifLouerSettings(systems=("cubic",), min_cell_length=3.8, max_cell_length=4.2,
                                  search_zero_shift=True, zero_shift_half_range_deg=.2)
    result = run_indexing(IndexingRequest(tuple(Peak(x, uncertainty=.001) for x in angles),
                         "boultif_louer", wavelength=1.0, settings=settings))
    best = result.candidates[0]
    assert best.zero_shift_deg == pytest.approx(-.1)
    assert best.cell.a == pytest.approx(4, abs=1e-6)
