"""Shared Niggli reduction and verified integral changes of lattice basis."""

import numpy as np
from spglib import niggli_reduce

from .models import IndexingError, ReducedCell, UnitCell


def matrix_tuple(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix)


def cell_from_metric(direct):
    lengths = np.sqrt(np.diag(direct))
    angles = [float(np.degrees(np.arccos(np.clip(
        direct[i, j] / (lengths[i] * lengths[j]), -1, 1))))
        for i, j in ((1, 2), (0, 2), (0, 1))]
    return UnitCell(*map(float, lengths), *angles)


def reduce_metric(reciprocal) -> ReducedCell:
    direct = np.linalg.inv(np.asarray(reciprocal, dtype=float))
    lattice = np.linalg.cholesky(direct)  # Rows are direct lattice vectors.
    reduced = niggli_reduce(lattice, eps=1e-7)
    if reduced is None:
        raise IndexingError("reduction_failed")
    transform_float = reduced @ np.linalg.inv(lattice)
    transform = np.rint(transform_float).astype(int)
    if (not np.allclose(transform_float, transform, atol=1e-6, rtol=0)
            or abs(round(np.linalg.det(transform))) != 1):
        raise IndexingError("reduction_failed")
    reduced_metric = transform @ direct @ transform.T
    if not np.allclose(reduced @ reduced.T, reduced_metric, atol=1e-6, rtol=1e-8):
        raise IndexingError("reduction_failed")
    return ReducedCell(cell_from_metric(reduced_metric), matrix_tuple(reduced_metric),
                       matrix_tuple(np.linalg.inv(reduced_metric)),
                       tuple(tuple(int(x) for x in row) for row in transform))


def metric_signature(reciprocal):
    # A numerical signature of a full reduced metric, never a truncated spectrum.
    return tuple(round(x, 7) for row in reduce_metric(reciprocal).direct_metric for x in row)
