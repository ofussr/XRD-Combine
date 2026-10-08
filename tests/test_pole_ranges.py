"""Physical reflection intervals expressed in d or 2theta."""

import math

import pytest

from pole_fixtures import phase
from xrd_workbench.models.data_errors import XRDDataError
from xrd_workbench.services.pole_figure import (
    available_reflections, d_range_to_two_theta, two_theta_range_to_d,
)


def test_known_bragg_interval_and_zero_angle_limit():
    assert d_range_to_two_theta(1, 2, 1) == pytest.approx((28.9550243719, 60))
    assert two_theta_range_to_d(28.9550243719, 60, 1) == pytest.approx((1, 2))
    assert two_theta_range_to_d(0, 180, 1) == (.5, math.inf)
    assert d_range_to_two_theta(.5, math.inf, 1) == (0, 180)
    # An entirely unreachable d interval must not become a new 180-degree pole.
    with pytest.raises(XRDDataError, match='no physical 2theta interval'):
        d_range_to_two_theta(.1, .2, 1)


@pytest.mark.parametrize('wavelength', [.8, 1.5406])
def test_angular_interval_selects_actual_bragg_angles(wavelength):
    crystal = phase().crystal
    all_rows = available_reflections(crystal, wavelength / 2, math.inf, wavelength)
    expected = {row.hkl for row in all_rows if 20 <= row.two_theta <= 80}
    actual = available_reflections(crystal, *two_theta_range_to_d(20, 80, wavelength), wavelength)
    assert {row.hkl for row in actual} == expected
    assert expected


@pytest.mark.parametrize('lower,upper', [(-1,30), (40,30), (0,0), (0,181),
                                       (math.nan,30), (20,math.inf)])
def test_invalid_angular_intervals_are_rejected(lower, upper):
    with pytest.raises(XRDDataError, match='2theta limits'):
        two_theta_range_to_d(lower, upper, 1)


@pytest.mark.parametrize('bounds', [(0,2), (2,1), (math.nan,2), (1,math.nan), (math.inf,math.inf)])
def test_invalid_spacing_intervals_are_rejected(bounds):
    with pytest.raises(XRDDataError):
        d_range_to_two_theta(*bounds, 1)


@pytest.mark.parametrize('wavelength', [0, -1, math.nan, math.inf])
def test_nonphysical_wavelengths_are_rejected(wavelength):
    for convert, bounds in ((d_range_to_two_theta, (1,2)), (two_theta_range_to_d, (20,80))):
        with pytest.raises(XRDDataError, match='wavelength'):
            convert(*bounds, wavelength)
