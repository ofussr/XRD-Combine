"""Analytic angular positions, signed reflections and measured fixed angles."""

import math
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from xrd_workbench.models.crystal import CifData, CrystalStructure
from xrd_workbench.models.diffraction import DiffractionAtom, DiffractionStructure
from xrd_workbench.models.phase_scan import PhaseOrientation, PhaseScanGeometry
from xrd_workbench.models.scan import Scan1D
from xrd_workbench.services.phase_scan import (
    calculate_scan_reflections, measurement_geometry, scan_profile, scattering_vector, shift_phi_reflections,
)
from xrd_workbench.services.rsm import angles_to_q


def phase(cell=(4, 4, 4, 90, 90, 90)):
    tags = ('_cell_length_a', '_cell_length_b', '_cell_length_c',
            '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
    data = CifData(Path('cell.cif'), dict(zip(tags, map(str, cell))), [])
    crystal = CrystalStructure.from_cif(data)
    structure = DiffractionStructure('cell', cell, [], ['x,y,z'], cell_only=True)
    return crystal, structure


TT = math.degrees(2 * math.asin(math.sqrt(2) / 8))
THETA = TT / 2


def calculate(geometry, limits, orientation=PhaseOrientation(), cell=None, radiations=None):
    crystal, structure = phase() if cell is None else phase(cell)
    return calculate_scan_reflections(crystal, structure, {}, radiations or [('test', 1., 1.)],
                                      geometry, orientation, limits)


def test_phi_fourfold_positions_and_signed_indices():
    rows = calculate(PhaseScanGeometry('phi', two_theta=TT, omega=THETA, chi=45, max_index=2), (-10, 350))
    assert [r.coordinate for r in rows] == pytest.approx([0, 90, 180, 270])
    assert [r.equivalents[0] for r in rows] == [(0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1)]
    assert all(r.two_theta == pytest.approx(TT) and r.intensity is None for r in rows)


def test_phi_periodic_endpoints_and_zero_offset():
    geometry = PhaseScanGeometry('phi', two_theta=TT, omega=THETA, chi=45, max_index=2)
    assert [r.coordinate for r in calculate(geometry, (0, 360))] == pytest.approx([0, 90, 180, 270, 360])
    assert [r.coordinate for r in calculate(geometry, (-10, 350), PhaseOrientation(phi_zero=17))] == pytest.approx([17, 107, 197, 287])


def test_chi_positive_and_negative_positions():
    rows = calculate(PhaseScanGeometry('chi', two_theta=TT, omega=THETA, phi=90, max_index=2), (-85, 85))
    assert [r.coordinate for r in rows] == pytest.approx([-45, 45])
    assert [r.equivalents[0] for r in rows] == [(-1, 0, 1), (1, 0, 1)]
    x, y = scan_profile(rows, (-85, 85), .2, 17001)
    assert sorted(x[np.argpartition(y, -2)[-2:]]) == pytest.approx([-45, 45])
    assert all(r.two_theta == pytest.approx(TT) for r in rows)


def test_rocking_curve_agrees_with_analytic_theta_plus_minus_tilt():
    rows = calculate(PhaseScanGeometry('omega', two_theta=TT, chi=0, phi=0, max_index=2), (-80, 90))
    assert [r.coordinate for r in rows] == pytest.approx([THETA-45, THETA+45])
    assert [r.equivalents[0] for r in rows] == [(-1, 0, 1), (1, 0, 1)]


def test_coupled_selects_only_surface_normal_reflections():
    rows = calculate(PhaseScanGeometry('coupled', chi=0, phi=0, max_index=3), (1, 50))
    assert [r.equivalents[0] for r in rows] == [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
    assert [r.coordinate for r in rows] == pytest.approx([math.degrees(2*math.asin(l/8)) for l in (1,2,3)])


def test_fixed_omega_detector_scan_finds_oblique_reflection():
    rows = calculate(PhaseScanGeometry('detector', omega=THETA+45, chi=0, phi=0, max_index=2), (1, 45))
    assert len(rows) == 1
    assert rows[0].equivalents == ((1, 0, 1),)
    assert rows[0].coordinate == pytest.approx(TT)


def test_rotated_surface_orientation():
    rows = calculate(PhaseScanGeometry('coupled', chi=0, phi=0, max_index=2), (1,45),
                     PhaseOrientation(surface=(1,1,0), inplane=(0,0,1)))
    assert [r.equivalents[0] for r in rows] == [(1,1,0), (2,2,0)]


def test_oblique_metric_retains_actual_signed_hkl():
    cell = (4, 5, 6, 90, 105, 90)
    crystal, _ = phase(cell)
    d = 1 / np.linalg.norm(crystal.reciprocal_vector((-1,0,1)))
    tt = math.degrees(2*math.asin(1/(2*d)))
    rows = calculate(PhaseScanGeometry('coupled', chi=0, phi=0, max_index=2), (tt-.1,tt+.1),
                     PhaseOrientation(surface=(-1,0,1), inplane=(0,1,0)), cell)
    assert len(rows) == 1
    assert rows[0].equivalents == ((-1,0,1),)
    assert rows[0].d == pytest.approx(d)


def test_radial_acceptance_and_radiation_filter():
    geometry = PhaseScanGeometry('phi', two_theta=TT+.1, omega=(TT+.1)/2, chi=45, tolerance=.01, max_index=2)
    assert not calculate(geometry, (-10,350))
    assert len(calculate(replace(geometry, tolerance=.2), (-10,350))) == 4
    lines = [('one',1.,1.), ('two',2.,.5)]
    assert {r.radiation for r in calculate(replace(geometry, two_theta=TT, omega=THETA), (-10,350), radiations=lines)} == {'one'}


def test_continuous_invariant_phi_has_no_artificial_peak():
    tt = math.degrees(2*math.asin(1/8))
    assert not calculate(PhaseScanGeometry('phi', two_theta=tt, omega=tt/2, chi=0, max_index=2), (0,360))


def test_structure_factor_extinctions_and_pre_cancelled_job():
    crystal, structure = phase()
    structure.cell_only = False
    structure.atoms = [DiffractionAtom('C',0,0,0), DiffractionAtom('C',.5,.5,.5)]
    factors = {'C': ([1.,0.,0.,0.], 0., [0.,0.,0.,0.])}
    geometry = PhaseScanGeometry('coupled', chi=0, phi=0, max_index=2)
    rows = calculate_scan_reflections(crystal, structure, factors, [('t',1.,1.)], geometry, PhaseOrientation(), (1,45))
    assert [r.equivalents[0] for r in rows] == [(0,0,2)]
    event = Event()
    event.set()
    assert not calculate_scan_reflections(crystal, structure, factors, [('t',1.,1.)], geometry, PhaseOrientation(), (1,45), event)


def test_systematic_absences_apply_to_signed_oriented_reflections():
    crystal, structure = phase()
    crystal.symmetry = [(np.eye(3), np.zeros(3)), (np.eye(3), np.array([.5,.5,.5]))]
    rows = calculate_scan_reflections(crystal, structure, {}, [('t',1.,1.)],
        PhaseScanGeometry('coupled',chi=0,phi=0,max_index=3), PhaseOrientation(), (1,50))
    assert [r.equivalents[0] for r in rows] == [(0,0,2)]


def test_zero_rotations_match_existing_rsm_vector():
    qx,qz = angles_to_q(17, 40, 1.54)
    np.testing.assert_allclose(scattering_vector(40,17,0,0,1.54), (qx,0,qz))


@pytest.mark.parametrize('geometry', [PhaseScanGeometry('phi'), PhaseScanGeometry('omega',two_theta=0,chi=0,phi=0),
    PhaseScanGeometry('coupled',chi=0,phi=0,tolerance=0), PhaseScanGeometry('coupled',chi=0,phi=float('nan'))])
def test_invalid_geometry_is_rejected(geometry):
    with pytest.raises(ValueError):
        calculate(geometry, (0,90))


def test_measurement_autofill_never_averages_a_moving_angle():
    scan = Scan1D('phi', [0,90,180], [1,2,1], Path('phi.xrdml'), axis_name='Phi', metadata={
        'axes': {'Phi': [0,90,180], '2Theta': [40,40,40], 'Omega': [20,20,20], 'Chi': [45,45,45]},
        'raw_drive_starts': {'Phi':0, 'Chi':42, 'X':100}})
    assert measurement_geometry(scan) == {'two_theta':40, 'omega':20, 'chi':45, 'omega_offset':0}
    scan.axis_name = '2Theta'
    scan.metadata = {'axes': {'2Theta':[20,30,40], 'Omega':[11,16,21]}, 'raw_drive_starts':{'Omega':11,'Phi':0}}
    assert measurement_geometry(scan) == {'phi':0, 'omega_offset':1}


def ranged(mode,**bounds):
    return PhaseScanGeometry(mode,max_index=2,
        ranges=tuple((key,*pair) for key,pair in bounds.items()))


def test_range_phi_fourfold_and_periodic_crop():
    geometry = ranged('phi',two_theta=(TT-.1,TT+.1),omega=(THETA-.1,THETA+.1),chi=(44.9,45.1))
    rows = calculate(geometry,(0,360))
    assert [r.coordinate for r in rows] == pytest.approx([0,90,180,270,360],abs=1e-6)
    shifted = shift_phi_reflections(rows,17,(-10,350))
    assert [r.coordinate for r in shifted] == pytest.approx([17,107,197,287],abs=1e-6)
    cropped = shift_phi_reflections(rows,-5,(-10,10))
    assert [r.coordinate for r in cropped] == pytest.approx([-5],abs=1e-6)
    assert all(r.two_theta == pytest.approx(TT) for r in shifted)


def test_range_phi_position_matches_analytic_omega_offset():
    geometry = ranged('phi',two_theta=(TT,TT),omega=(THETA+.1,THETA+.1),chi=(44.9,45.1))
    rows = calculate(geometry,(-10,350))
    row = next(r for r in rows if r.equivalents == ((1,0,1),))
    expected = math.degrees(math.acos(math.sin(math.radians(.1))*math.sqrt(2)))
    assert row.coordinate == pytest.approx(expected,abs=1e-7)


def test_each_fixed_range_is_enforced_jointly_without_global_tolerance():
    geometry = ranged('phi',two_theta=(TT,TT),omega=(THETA+.1,THETA+.1),chi=(45,45))
    assert not calculate(replace(geometry,tolerance=5),(-10,350))
    broad = replace(geometry,ranges=(('two_theta',TT,TT),('omega',THETA+.1,THETA+.1),('chi',44.9,45.1)))
    assert len(calculate(broad,(-10,350))) == 4
    assert not calculate(ranged('phi',two_theta=(TT+.01,TT+.02),omega=(THETA-1,THETA+1),chi=(44,46)),(-10,350))


def test_range_chi_and_omega_exact_and_zero_width_windows():
    rows = calculate(ranged('chi',two_theta=(TT,TT),omega=(THETA,THETA),phi=(90,90)),(-85,85))
    assert [r.coordinate for r in rows] == pytest.approx([-45,45],abs=1e-6)
    rows = calculate(ranged('omega',two_theta=(TT,TT),chi=(0,0),phi=(359.9,360.1)),(-80,90))
    assert [r.coordinate for r in rows] == pytest.approx([THETA-45,THETA+45],abs=1e-6)


def test_range_coupled_and_fixed_detector():
    rows = calculate(ranged('coupled',chi=(-.1,.1),phi=(-.1,.1),omega_offset=(-.1,.1)),(1,50))
    assert [r.equivalents[0] for r in rows] == [(0,0,1),(0,0,2)]
    rows = calculate(ranged('detector',chi=(-.1,.1),phi=(-.1,.1),omega=(THETA+44.9,THETA+45.1)),(1,45))
    assert [r.equivalents[0] for r in rows] == [(1,0,1)]
    assert rows[0].coordinate == pytest.approx(TT)


@pytest.mark.parametrize('bounds', [((45,44),), ((float('nan'),45),), ((-1000,1000),)])
def test_invalid_angle_ranges_are_rejected(bounds):
    geometry = ranged('coupled',chi=bounds[0],phi=(0,0),omega_offset=(0,0))
    with pytest.raises(ValueError):
        calculate(geometry,(1,50))


def test_range_scan_omits_poles_invariant_under_the_scanned_rotation():
    tt = math.degrees(2*math.asin(1/8))
    assert not calculate(ranged('phi',two_theta=(tt,tt),omega=(tt/2,tt/2),chi=(0,0)),(0,360))
