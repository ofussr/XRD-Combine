"""Единый загруженный CIF для всех расчётных и визуальных страниц."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .cif_xrd import Atom as DiffractionAtom
    from .cif_xrd import Structure as DiffractionStructure
    from .theoretical_pole import CifData, Crystal, parse_cif
except ImportError:  # pragma: no cover - прямой запуск модуля
    from cif_xrd import Atom as DiffractionAtom
    from cif_xrd import Structure as DiffractionStructure
    from theoretical_pole import CifData, Crystal, parse_cif


@dataclass(frozen=True)
class CifDocument:
    """Один раз разобранный CIF и представления для существующих расчётов."""

    source: Path
    data: CifData
    crystal: Crystal
    diffraction: DiffractionStructure

    @property
    def name(self) -> str:
        return self.crystal.formula or self.source.stem


def load_cif_document(path: str | Path) -> CifDocument:
    """Разобрать CIF один раз и построить обе совместимые модели из него."""

    source = Path(path)
    data = parse_cif(source)
    crystal = Crystal.from_cif(data)
    symmetry_operations = data.loop_column(
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    ) or ["x,y,z"]
    diffraction = DiffractionStructure(
        name=crystal.formula or source.stem,
        cell=(
            crystal.a,
            crystal.b,
            crystal.c,
            crystal.alpha,
            crystal.beta,
            crystal.gamma,
        ),
        atoms=[
            DiffractionAtom(
                atom.element,
                float(atom.fractional[0]),
                float(atom.fractional[1]),
                float(atom.fractional[2]),
                atom.occupancy,
                atom.b_iso,
            )
            for atom in crystal.atoms
        ],
        symmetry_operations=list(symmetry_operations),
    )
    return CifDocument(source, data, crystal, diffraction)
