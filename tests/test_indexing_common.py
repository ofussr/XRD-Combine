import math
import subprocess
import sys
from threading import Event

import numpy as np
import pytest

from xrd_workbench.indexing.models import IndexingCancelled, IndexingError, IndexingRequest, Peak
from xrd_workbench.indexing.reduction import reduce_metric
from xrd_workbench.indexing.service import run_indexing
from xrd_workbench.indexing.indexing_visser import VisserIndexer
from xrd_workbench.indexing.indexing_boultif_louer import BoultifLouerIndexer, BoultifLouerSettings
from indexing_fixtures import CELLS, reciprocal


@pytest.mark.parametrize("engine", [VisserIndexer, BoultifLouerIndexer])
@pytest.mark.parametrize("kind,position,error", [("two_theta", 90, .001), ("d", 2, 1e-6), ("q", .25, 1e-8)])
def test_explicit_uncertainty_is_not_inflated(engine, kind, position, error):
    indexer = engine()
    options = {"zero_shift_deg": 0} if engine is BoultifLouerIndexer else {}
    precise = indexer._prepare_peaks([Peak(position, uncertainty=error)],
                                    input_kind=kind, wavelength=1.5406, **options)[0]
    fallback = indexer._prepare_peaks([Peak(position)], input_kind=kind, wavelength=1.5406, **options)[0]
    assert precise.q_tolerance < fallback.q_tolerance / 10
    if kind == "two_theta":
        q = lambda angle: (2 * math.sin(math.radians(angle / 2)) / 1.5406)**2
        assert precise.q_tolerance == pytest.approx(max(q(90.001) - q(90), q(90) - q(89.999)))
    elif kind == "q":
        assert precise.q_tolerance == error
    else:
        assert precise.q_tolerance == pytest.approx(1 / (2 - error)**2 - .25)


@pytest.mark.parametrize("error", [0, -1, math.nan, math.inf])
def test_invalid_uncertainty_is_reported(error):
    with pytest.raises(IndexingError, match="invalid_uncertainty"):
        run_indexing(IndexingRequest((Peak(30, uncertainty=error),), "visser", wavelength=1.54))


@pytest.mark.parametrize("system", CELLS)
def test_reduction_preserves_lattice_and_hkl_under_change_of_basis(system):
    original = reciprocal(CELLS[system])
    direct = np.linalg.inv(original)
    basis = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    changed = np.linalg.inv(basis @ direct @ basis.T)
    a, b = reduce_metric(original), reduce_metric(changed)
    np.testing.assert_allclose(a.direct_metric, b.direct_metric, atol=1e-7)
    for hkl in [(1, 2, 3), (-2, 0, 1), (0, 0, 1)]:
        new_hkl = b.transform_hkl(hkl)
        assert np.array(new_hkl) @ np.array(b.reciprocal_metric) @ np.array(new_hkl) == pytest.approx(
            np.array(hkl) @ changed @ np.array(hkl))


def test_invalid_coordinates_and_cancel_are_structured():
    with pytest.raises(IndexingError, match="invalid_position"):
        run_indexing(IndexingRequest((Peak(-1),), "visser", wavelength=1.54))
    with pytest.raises(IndexingError, match="invalid_wavelength"):
        run_indexing(IndexingRequest((Peak(30),), "visser"))
    event = Event()
    event.set()
    with pytest.raises(IndexingCancelled):
        run_indexing(IndexingRequest((), "visser"), event)


def test_indexing_imports_without_gui():
    result = subprocess.run([sys.executable, "-c", "import sys; import xrd_workbench.indexing.service; "
                             "assert not any(x.startswith('PySide6') for x in sys.modules)"], check=False)
    assert result.returncode == 0


def test_enabled_peaks_keep_original_indices_and_stable_ids():
    positions = [n / 16 for n in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22)]
    peaks = tuple(Peak(value, peak_id=100 + i, enabled=i != 3) for i, value in enumerate(reversed(positions)))
    settings = BoultifLouerSettings(systems=("cubic",), min_cell_length=3.8, max_cell_length=4.2)
    result = run_indexing(IndexingRequest(peaks, "boultif_louer", "q", settings=settings))
    assert result.candidates
    for line in result.candidates[0].indexed_lines:
        assert line.peak_id == 100 + line.observed_index
        assert line.observed_position == peaks[line.observed_index].position
        assert line.observed_index != 3
    assert result.diagnostics["disabled_peak_ids"] == [103]
