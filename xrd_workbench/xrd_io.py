"""Compatibility facade for scan models, readers and translated errors.

New toolkit-independent code should import from :mod:`xrd_workbench.models.scan`
and :mod:`xrd_workbench.io`. The 2.x Tkinter interface continues to use this
module so its public API and localised error messages remain unchanged.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .bruker_raw import read_bruker_raw
    from .i18n import localised
    from .io.bruker import read_raw_scans as _read_raw_scans
    from .io.text import read_xy as _read_xy
    from .io.xrdml import read_xrdml as _read_xrdml
    from .models.data_errors import XRDDataError
    from .models.scan import (
        Scan1D as ScanModel,
        assign_text_axis as _assign_text_axis,
        clone_scan,
    )
except ImportError:  # pragma: no cover - direct module execution
    from bruker_raw import read_bruker_raw
    from i18n import localised
    from io.bruker import read_raw_scans as _read_raw_scans
    from io.text import read_xy as _read_xy
    from io.xrdml import read_xrdml as _read_xrdml
    from models.data_errors import XRDDataError
    from models.scan import (
        Scan1D as ScanModel,
        assign_text_axis as _assign_text_axis,
        clone_scan,
    )


def _localised_error(error: XRDDataError) -> str:
    code = error.code
    context = error.context
    axis = context.get("axis", "")
    index = context.get("index", "")
    source_name = context.get("source_name", "")

    if code == "scan_arrays":
        return localised(
            "Coordinates and intensities must be one-dimensional arrays.",
            "Les coordonnées et les intensités doivent être des tableaux unidimensionnels.",
            "Координаты и интенсивности должны быть одномерными массивами.",
        )
    if code == "scan_too_short":
        return localised(
            "The dataset contains fewer than two valid points.",
            "Le jeu de données contient moins de deux points valides.",
            "В наборе данных меньше двух корректных точек.",
        )
    if code == "scan_axis_missing":
        return localised(
            f"Axis {axis!r} is not available for this dataset.",
            f"L’axe {axis!r} n’est pas disponible pour ce jeu de données.",
            f"Ось {axis!r} недоступна для этого набора данных.",
        )
    if code == "scan_axis_too_short":
        return localised(
            f"Axis {axis!r} contains fewer than two valid points.",
            f"L’axe {axis!r} contient moins de deux points valides.",
            f"Ось {axis!r} содержит меньше двух корректных точек.",
        )
    if code == "scan_text_axis_only":
        return localised(
            "Only text datasets support axis assignment.",
            "Seules les données textuelles permettent de redéfinir l’axe.",
            "Назначить смысл оси можно только для текстовых данных.",
        )
    if code == "xrdml_invalid_xml":
        detail = context.get("detail", "")
        return localised(
            f"Invalid XML: {detail}",
            f"XML incorrect : {detail}",
            f"Некорректный XML: {detail}",
        )
    if code == "xrdml_no_data":
        return localised(
            "No dataPoints block was found in the XRDML file.",
            "Aucun bloc dataPoints n’a été trouvé dans le fichier XRDML.",
            "В XRDML не найден блок dataPoints.",
        )
    if code == "xrdml_no_axis":
        return localised(
            "The XRDML scan contains no readable coordinate axis.",
            "Le balayage XRDML ne contient aucun axe de coordonnées lisible.",
            "В скане XRDML нет читаемой координатной оси.",
        )
    if code == "xrdml_scan_no_intensity":
        return localised(
            f"scan {index}: no intensities/counts",
            f"balayage {index} : aucune intensité",
            f"скан {index}: нет intensities/counts",
        )
    if code == "xrdml_scan_too_short":
        return localised(
            f"scan {index}: fewer than two points",
            f"balayage {index} : moins de deux points",
            f"скан {index}: меньше двух точек",
        )
    if code == "xrdml_scan_no_axis":
        detail = _localised_error(XRDDataError("xrdml_no_axis"))
        return localised(
            f"scan {index}: {detail}",
            f"balayage {index} : {detail}",
            f"скан {index}: {detail}",
        )
    if code == "xrdml_no_valid_scan":
        detail = "; ".join(_localised_error(item) for item in context.get("issues", ()))
        return localised(
            f"Could not read a one-dimensional scan from XRDML: {detail}",
            f"Impossible de lire un balayage unidimensionnel depuis le XRDML : {detail}",
            f"Не удалось прочитать одномерный скан из XRDML: {detail}",
        )
    if code == "raw_unreadable":
        return localised(
            f"Could not read {source_name} as Bruker RAW v3/v4. "
            "The file may be damaged or use an unsupported RAW variant.",
            f"Impossible de lire {source_name} comme fichier Bruker RAW v3/v4. "
            "Le fichier est peut-être endommagé ou utilise une variante RAW non prise en charge.",
            f"Не удалось прочитать {source_name} как Bruker RAW v3/v4. "
            "Файл может быть повреждён либо использовать неподдерживаемый вариант RAW.",
        )
    if code == "raw_no_ranges":
        return localised(
            "The RAW file contains no one-dimensional range with at least two points.",
            "Le fichier RAW ne contient aucune plage unidimensionnelle d’au moins deux points.",
            "В RAW нет одномерных диапазонов как минимум с двумя точками.",
        )
    if code == "unsupported_format":
        suffix = context.get("suffix", "")
        if suffix == "no extension":
            suffix = localised("no extension", "sans extension", "без расширения")
        return localised(
            f"Unsupported format: {suffix}.",
            f"Format non pris en charge : {suffix}.",
            f"Неподдерживаемый формат: {suffix}.",
        )
    return str(error)


class Scan1D(ScanModel):
    """Tkinter-era compatible scan model with localised validation errors."""

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except XRDDataError as exc:
            raise ValueError(_localised_error(exc)) from exc

    def use_axis(self, axis_name: str) -> None:
        try:
            super().use_axis(axis_name)
        except XRDDataError as exc:
            raise ValueError(_localised_error(exc)) from exc


def assign_text_axis(scan: ScanModel, axis_name: str, *, assumed: bool = False) -> None:
    try:
        _assign_text_axis(scan, axis_name, assumed=assumed)
    except XRDDataError as exc:
        raise ValueError(_localised_error(exc)) from exc


def read_xrdml(path: str | Path) -> list[Scan1D]:
    try:
        return _read_xrdml(path, scan_factory=Scan1D)
    except XRDDataError as exc:
        raise ValueError(_localised_error(exc)) from exc


def read_xy(path: str | Path) -> Scan1D:
    try:
        return _read_xy(path, scan_factory=Scan1D)
    except XRDDataError as exc:
        raise ValueError(_localised_error(exc)) from exc


def read_raw_scans(path: str | Path, *, raw=None) -> list[Scan1D]:
    try:
        return _read_raw_scans(
            path,
            raw=raw,
            scan_factory=Scan1D,
            raw_reader=read_bruker_raw,
        )
    except XRDDataError as exc:
        raise ValueError(_localised_error(exc)) from exc


def read_scan_file(path: str | Path) -> list[Scan1D]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xrdml", ".xml"}:
        return read_xrdml(source)
    if suffix == ".raw":
        return read_raw_scans(source)
    if suffix in {".xy", ".txt", ".dat", ".csv"}:
        return [read_xy(source)]
    error = XRDDataError(
        "unsupported_format",
        suffix=source.suffix or "no extension",
    )
    raise ValueError(_localised_error(error)) from error


__all__ = [
    "Scan1D",
    "assign_text_axis",
    "clone_scan",
    "read_raw_scans",
    "read_scan_file",
    "read_xrdml",
    "read_xy",
]
