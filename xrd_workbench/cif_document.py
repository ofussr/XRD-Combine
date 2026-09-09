"""Единый загруженный CIF для всех расчётных и визуальных страниц."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

try:
    from .io.cif import read_cif_data
    from .localization import localised
    from .models.crystal import CifData, Crystal
    from .models.data_errors import XRDDataError
    from .models.diffraction import DiffractionAtom, DiffractionStructure
except ImportError:  # pragma: no cover - прямой запуск модуля
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from xrd_workbench.io.cif import read_cif_data
    from xrd_workbench.localization import localised
    from xrd_workbench.models.crystal import CifData, Crystal
    from xrd_workbench.models.data_errors import XRDDataError
    from xrd_workbench.models.diffraction import DiffractionAtom, DiffractionStructure


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
    if error.code == "cif_lex":
        line = error.context.get("line_number", "")
        reason_code = error.context.get("reason", "")
        reason = localised(
            "unclosed quotation mark"
            if reason_code == "unclosed_quote"
            else "unclosed multiline field",
            "guillemet non fermé"
            if reason_code == "unclosed_quote"
            else "champ multiligne non fermé",
            "незакрытая кавычка"
            if reason_code == "unclosed_quote"
            else "незакрытое многострочное поле",
        )
        return localised(
            f"Could not parse CIF line {line}: {reason}.",
            f"Impossible d’analyser la ligne CIF {line} : {reason}.",
            f"Не удалось разобрать строку CIF {line}: {reason}.",
        )
    if error.code == "cif_loop_no_columns":
        return localised(
            "No column names follow loop_.",
            "Aucun nom de colonne ne suit loop_.",
            "После loop_ не найдены имена столбцов.",
        )
    if error.code == "cif_loop_width":
        values = error.context.get("value_count", "")
        columns = error.context.get("column_count", "")
        return localised(
            f"The CIF loop value count is not divisible by the column count ({values} and {columns}).",
            f"Le nombre de valeurs de la boucle CIF n’est pas divisible par le nombre de colonnes ({values} et {columns}).",
            f"Число значений в цикле CIF не кратно числу столбцов ({values} и {columns}).",
        )
    if error.code == "cif_field_missing":
        field = error.context.get("field", "")
        return localised(
            f"CIF field {field} has no value.",
            f"Le champ CIF {field} n’a pas de valeur.",
            f"Для поля {field} отсутствует значение.",
        )
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
    try:
        data = read_cif_data(source)
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
