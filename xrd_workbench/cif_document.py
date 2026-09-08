"""Единый загруженный CIF для всех расчётных и визуальных страниц."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from .i18n import localised
    from .models.crystal import CifData, Crystal
    from .models.data_errors import XRDDataError
    from .models.diffraction import DiffractionAtom, DiffractionStructure
    from .theoretical_pole import parse_cif
except ImportError:  # pragma: no cover - прямой запуск модуля
    from i18n import localised
    from models.crystal import CifData, Crystal
    from models.data_errors import XRDDataError
    from models.diffraction import DiffractionAtom, DiffractionStructure
    from theoretical_pole import parse_cif


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


def _localized_crystal_error(error: XRDDataError) -> str:
    if error.code == "crystal_cif_number_undefined":
        return localised(
            "Undefined numeric CIF value.",
            "Valeur numérique CIF indéfinie.",
            "Неопределённое числовое значение CIF.",
        )
    if error.code == "crystal_cif_number_invalid":
        value = error.context.get("value", "")
        return localised(
            f"Invalid CIF number: {value!r}",
            f"Nombre CIF incorrect : {value!r}",
            f"Некорректное число CIF: {value!r}",
        )
    if error.code == "crystal_symmetry":
        expression = error.context.get("expression", "")
        return localised(
            f"Could not parse symmetry operation {expression!r}.",
            f"Impossible d’analyser l’opération de symétrie {expression!r}.",
            f"Не удалось разобрать операцию симметрии {expression!r}.",
        )
    if error.code == "crystal_cell_degenerate_gamma":
        return localised(
            "Degenerate unit cell: sin(gamma) = 0.",
            "Maille dégénérée : sin(gamma) = 0.",
            "Вырожденная ячейка: sin(gamma) = 0.",
        )
    if error.code == "crystal_cell_degenerate":
        return localised(
            "The CIF parameters define a degenerate unit cell.",
            "Les paramètres CIF définissent une maille dégénérée.",
            "Параметры CIF задают вырожденную ячейку.",
        )
    if error.code == "crystal_cell_missing":
        joined = ", ".join(error.context.get("fields", ()))
        return localised(
            f"Unit-cell parameters are missing from the CIF: {joined}",
            f"Des paramètres de maille sont absents du CIF : {joined}",
            f"В CIF отсутствуют параметры ячейки: {joined}",
        )
    if error.code == "crystal_non_p1_without_symmetry":
        group = error.context.get("space_group", "")
        return localised(
            f"The CIF declares the non-P1 space group {group!r} but contains no "
            "explicit symmetry operations. Calculating it as P1 would be incorrect.",
            f"Le CIF déclare le groupe d’espace non P1 {group!r}, mais ne contient "
            "aucune opération de symétrie explicite. Un calcul en P1 serait incorrect.",
            f"В CIF указана непримитивная группа {group!r}, но отсутствуют явные "
            "операции симметрии. Расчёт как P1 был бы некорректен.",
        )
    return str(error)


def load_cif_document(path: str | Path) -> CifDocument:
    """Разобрать CIF один раз и построить обе совместимые модели из него."""

    source = Path(path)
    data = parse_cif(source)
    try:
        crystal = Crystal.from_cif(data)
    except XRDDataError as exc:
        raise ValueError(_localized_crystal_error(exc)) from exc
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
