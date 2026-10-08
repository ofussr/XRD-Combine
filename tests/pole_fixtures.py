"""Scientific pole-figure fixtures shared by model and native UI tests."""
from types import SimpleNamespace
import numpy as np
from xrd_workbench.cell_phase import create_cell_phase_document
from xrd_workbench.space_groups import SETTINGS

CIF_TEXT = """data_Si
_chemical_formula_sum 'Si'
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_it_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
"""


def phase(name='phase', cell=(4., 5., 6., 90., 90., 90.)):
    return create_cell_phase_document(name, SETTINGS[0], cell)


def raw_fixture():
    return SimpleNamespace(is_pole_figure=True, ranges=[
        SimpleNamespace(chi=40., phi=np.array([0., 1., 2., 10., 11., 12.]), intensity=np.array([11., 12., 13., 14., 15., 16.])),
        SimpleNamespace(chi=10., phi=np.array([360., 270., 180., 90., 0.]), intensity=np.array([101., 4., 3., 2., 1.])),
        SimpleNamespace(chi=24., phi=np.array([]), intensity=np.array([])),
    ])


MIXED_CIF = """data_mixed
_chemical_formula_sum 'Nb0.4 Ta0.6 O3'
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Nb1 Nb 0.5 0.5 0.5 0.4
Ta1 Ta 0.5 0.5 0.5 0.6
O1 O 0.25 0.5 0.5 1
O2 O 0.75 0.5 0.5 1
O3 O 0.5 0.25 0.5 1
O4 O 0.5 0.75 0.5 1
O5 O 0.5 0.5 0.25 1
O6 O 0.5 0.5 0.75 1
"""
