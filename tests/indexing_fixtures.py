"""Independent analytic metrics and a published NIST line-position table.

No XRD Combine reflection calculator or indexing conversion is used here.
"""

import itertools
import math
import numpy as np

CELLS = {
    "cubic": (4, 4, 4, 90, 90, 90),
    "tetragonal": (4, 4, 6, 90, 90, 90),
    "hexagonal": (4, 4, 6, 90, 90, 120),
    "orthorhombic": (4, 5, 6, 90, 90, 90),
    "monoclinic": (4, 5, 6, 90, 110, 90),
    "triclinic": (4, 4.3, 4.7, 80, 95, 105),
}


def reciprocal(cell):
    a, b, c, alpha, beta, gamma = cell
    direct = np.diag([a*a, b*b, c*c]).astype(float)
    for i, j, left, right, angle in ((0, 1, a, b, gamma), (0, 2, a, c, beta), (1, 2, b, c, alpha)):
        direct[i, j] = direct[j, i] = left * right * math.cos(math.radians(angle))
    return np.linalg.inv(direct)


def q_lines(cell, count=25):
    metric = reciprocal(cell)
    values = {round(float(np.array(hkl) @ metric @ np.array(hkl)), 10)
              for hkl in itertools.product(range(-4, 5), repeat=3) if any(hkl)}
    return sorted(values)[:count]


# NIST SRM 640f certificate, 26 February 2019, Table 1, page 3.
# Published informational positions, not raw measured diffraction data.
# https://tsapps.nist.gov/srmext/certificates/archives/640f.pdf
NIST_SI_ANGLES = (28.441, 47.301, 56.120, 69.127, 76.373, 88.026,
                  94.947, 106.702, 114.085, 127.535, 136.882)
NIST_SI_WAVELENGTH = 1.5405929  # Certificate uses 0.15405929 nm.
NIST_SI_LATTICE = 5.431144  # 0.5431144 nm, certified at 22.5 °C.
