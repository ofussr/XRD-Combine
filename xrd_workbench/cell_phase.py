"""Atom-free crystallographic phase defined by a cell and a space group."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .cif_xrd import Structure
    from .i18n import localised
    from .space_groups import SpaceGroupSetting
    from .theoretical_pole import CifData, Crystal, direct_basis, parse_symmetry_operation
except ImportError:  # pragma: no cover
    from cif_xrd import Structure
    from i18n import localised
    from space_groups import SpaceGroupSetting
    from theoretical_pole import CifData, Crystal, direct_basis, parse_symmetry_operation


@dataclass(frozen=True)
class CellPhaseDocument:
    name: str
    setting: SpaceGroupSetting
    cell: tuple[float, float, float, float, float, float]
    source: Path
    data: CifData
    crystal: Crystal
    diffraction: Structure
    is_cell_only: bool = True


def create_cell_phase_document(
    name: str,
    setting: SpaceGroupSetting,
    cell: tuple[float, float, float, float, float, float],
) -> CellPhaseDocument:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError(
            localised(
                "Phase name cannot be empty.",
                "Le nom de la phase ne peut pas être vide.",
                "Название фазы не может быть пустым.",
            )
        )
    if any(not np.isfinite(value) for value in cell):
        raise ValueError(
            localised(
                "Cell parameters must be finite numbers.",
                "Les paramètres de maille doivent être des nombres finis.",
                "Параметры ячейки должны быть конечными числами.",
            )
        )
    if any(value <= 0 for value in cell[:3]):
        raise ValueError(
            localised(
                "Cell lengths must be positive.",
                "Les longueurs de maille doivent être positives.",
                "Длины рёбер ячейки должны быть положительными.",
            )
        )
    if any(value <= 0 or value >= 180 for value in cell[3:]):
        raise ValueError(
            localised(
                "Cell angles must be between 0 and 180 degrees.",
                "Les angles de maille doivent être compris entre 0 et 180 degrés.",
                "Углы ячейки должны находиться между 0 и 180 градусами.",
            )
        )

    a, b, c, alpha, beta, gamma = cell
    try:
        direct = direct_basis(a, b, c, alpha, beta, gamma)
        reciprocal = np.linalg.inv(direct).T
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            localised(
                "These parameters do not define a valid unit cell.",
                "Ces paramètres ne définissent pas une maille valide.",
                "Эти параметры не задают допустимую элементарную ячейку.",
            )
        ) from exc
    symmetry = [parse_symmetry_operation(item) for item in setting.operations]
    source = Path(f"{clean_name}.cell")
    values = {
        "_cell_length_a": str(a),
        "_cell_length_b": str(b),
        "_cell_length_c": str(c),
        "_cell_angle_alpha": str(alpha),
        "_cell_angle_beta": str(beta),
        "_cell_angle_gamma": str(gamma),
        "_space_group_it_number": str(setting.number),
        "_space_group_name_h-m_alt": setting.international_short,
        "_chemical_formula_sum": clean_name,
    }
    data = CifData(source, values, [])
    space_group = setting.international_short
    if setting.choice:
        space_group += f" ({setting.choice})"
    crystal = Crystal(
        data,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        direct,
        reciprocal,
        symmetry,
        [],
        space_group,
        clean_name,
    )
    diffraction = Structure(
        clean_name,
        cell,
        [],
        list(setting.operations),
        cell_only=True,
        space_group=space_group,
    )
    return CellPhaseDocument(
        clean_name,
        setting,
        cell,
        source,
        data,
        crystal,
        diffraction,
    )
