"""Общие модели и функции чтения одномерных рентгенограмм."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .bruker_raw import read_bruker_raw
    from .i18n import localised
except ImportError:  # запуск модулей напрямую
    from bruker_raw import read_bruker_raw
    from i18n import localised


@dataclass
class Scan1D:
    """Один отображаемый одномерный набор данных."""

    name: str
    x: np.ndarray
    y: np.ndarray
    source: Path
    kind: str = "измерение"
    axis_name: str = "2Theta"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial_x = np.asarray(self.x, dtype=float)
        initial_y = np.asarray(self.y, dtype=float)
        if (
            initial_x.ndim != 1
            or initial_y.ndim != 1
            or initial_x.size != initial_y.size
        ):
            raise ValueError(
                localised(
                    "Coordinates and intensities must be one-dimensional arrays.",
                    "Les coordonnées et les intensités doivent être des tableaux unidimensionnels.",
                    "Координаты и интенсивности должны быть одномерными массивами.",
                )
            )

        supplied_axes = self.metadata.get("axes", {self.axis_name: initial_x})
        axes: dict[str, np.ndarray] = {}
        for name, values in supplied_axes.items():
            array = np.asarray(values, dtype=float)
            if array.ndim == 1 and array.size == initial_y.size:
                axes[str(name)] = array
        if self.axis_name not in axes:
            axes[self.axis_name] = initial_x

        valid = np.isfinite(initial_x) & np.isfinite(initial_y)
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                localised(
                    "The dataset contains fewer than two valid points.",
                    "Le jeu de données contient moins de deux points valides.",
                    "В наборе данных меньше двух корректных точек.",
                )
            )
        self.metadata["axes"] = {
            name: values[valid].copy() for name, values in axes.items()
        }
        self.metadata["_base_y"] = initial_y[valid].copy()
        self.use_axis(self.axis_name)

    @property
    def available_axes(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("axes", {}).keys())

    def use_axis(self, axis_name: str) -> None:
        axes = self.metadata.get("axes", {})
        if axis_name not in axes:
            raise ValueError(
                localised(
                    f"Axis {axis_name!r} is not available for this dataset.",
                    f"L’axe {axis_name!r} n’est pas disponible pour ce jeu de données.",
                    f"Ось {axis_name!r} недоступна для этого набора данных.",
                )
            )
        values = np.asarray(axes[axis_name], dtype=float)
        base_y = np.asarray(self.metadata["_base_y"], dtype=float)
        valid = np.isfinite(values) & np.isfinite(base_y)
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                localised(
                    f"Axis {axis_name!r} contains fewer than two valid points.",
                    f"L’axe {axis_name!r} contient moins de deux points valides.",
                    f"Ось {axis_name!r} содержит меньше двух корректных точек.",
                )
            )
        order = np.argsort(values[valid], kind="stable")
        self.x = values[valid][order]
        self.y = base_y[valid][order]
        self.axis_name = axis_name


def clone_scan(scan: Scan1D, *, name: str | None = None) -> Scan1D:
    """Create an independent scan copy while preserving every coordinate axis."""

    metadata = copy.deepcopy(scan.metadata)
    base_y = np.asarray(metadata.pop("_base_y", scan.y), dtype=float).copy()
    axes = metadata.get("axes", {})
    active_x = np.asarray(
        axes.get(scan.axis_name, scan.x),
        dtype=float,
    ).copy()
    return Scan1D(
        name=name or scan.name,
        x=active_x,
        y=base_y,
        source=Path(scan.source),
        kind=scan.kind,
        axis_name=scan.axis_name,
        metadata=metadata,
    )


def assign_text_axis(scan: Scan1D, axis_name: str, *, assumed: bool = False) -> None:
    """Assign a meaning to a text column, without inventing extra coordinates."""
    if scan.metadata.get("format") != "XY":
        raise ValueError(localised(
            "Only text datasets support axis assignment.",
            "Seules les données textuelles permettent de redéfinir l’axe.",
            "Назначить смысл оси можно только для текстовых данных.",
        ))
    old_axis = scan.metadata["axes"][scan.axis_name]
    scan.metadata["axes"] = {axis_name: old_axis.copy()}
    scan.metadata["axis_assumed"] = assumed
    scan.use_axis(axis_name)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element.iter() if _local_name(child.tag) == name)


def _first(element: ET.Element, name: str) -> ET.Element | None:
    return next(_children(element, name), None)


def _numbers(text: str | None) -> np.ndarray:
    if not text:
        return np.empty(0, dtype=float)
    return np.fromstring(text.replace(",", "."), sep=" ", dtype=float)


def _position_axes(
    data_points: ET.Element, point_count: int
) -> tuple[dict[str, np.ndarray], str]:
    axes: dict[str, np.ndarray] = {}
    for positions in _children(data_points, "positions"):
        axis_name = positions.attrib.get("axis", "").strip()
        if not axis_name:
            continue
        listed = _first(positions, "listPositions")
        common_node = _first(positions, "commonPosition")
        start_node = _first(positions, "startPosition")
        end_node = _first(positions, "endPosition")
        try:
            if listed is not None:
                values = _numbers(listed.text)
                if values.size != point_count:
                    continue
            elif common_node is not None:
                value = float((common_node.text or "").replace(",", "."))
                values = np.full(point_count, value)
            elif start_node is not None and end_node is not None:
                start = float((start_node.text or "").replace(",", "."))
                end = float((end_node.text or "").replace(",", "."))
                values = np.linspace(start, end, point_count)
            else:
                continue
        except ValueError:
            continue
        axes[axis_name] = values

    if not axes:
        raise ValueError(
            localised(
                "The XRDML scan contains no readable coordinate axis.",
                "Le balayage XRDML ne contient aucun axe de coordonnées lisible.",
                "В скане XRDML нет читаемой координатной оси.",
            )
        )
    varying_axes = [
        name
        for name, values in axes.items()
        if np.nanmax(values) - np.nanmin(values) > 1e-12
    ]
    two_theta = next(
        (name for name in axes if name.replace(" ", "").lower() == "2theta"),
        None,
    )
    if two_theta is not None and two_theta in varying_axes:
        default_axis = two_theta
    elif varying_axes:
        default_axis = varying_axes[0]
    else:
        default_axis = two_theta or next(iter(axes))
    return axes, default_axis


def read_xrdml(path: str | Path) -> list[Scan1D]:
    """Прочитать все одномерные сканы и их координатные оси из XRDML."""

    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise ValueError(
            localised(
                f"Invalid XML: {exc}",
                f"XML incorrect : {exc}",
                f"Некорректный XML: {exc}",
            )
        ) from exc

    sample_node = _first(root, "sampleName")
    if sample_node is None:
        sample_node = _first(root, "name")
    sample_name = (sample_node.text or "").strip() if sample_node is not None else ""
    data_blocks = list(_children(root, "dataPoints"))
    if not data_blocks:
        raise ValueError(
            localised(
                "No dataPoints block was found in the XRDML file.",
                "Aucun bloc dataPoints n’a été trouvé dans le fichier XRDML.",
                "В XRDML не найден блок dataPoints.",
            )
        )

    result: list[Scan1D] = []
    errors: list[str] = []
    for index, block in enumerate(data_blocks, start=1):
        intensity_node = _first(block, "intensities")
        if intensity_node is None:
            intensity_node = _first(block, "counts")
        if intensity_node is None:
            errors.append(
                localised(
                    f"scan {index}: no intensities/counts",
                    f"balayage {index} : aucune intensité",
                    f"скан {index}: нет intensities/counts",
                )
            )
            continue
        intensity = _numbers(intensity_node.text)
        if intensity.size < 2:
            errors.append(
                localised(
                    f"scan {index}: fewer than two points",
                    f"balayage {index} : moins de deux points",
                    f"скан {index}: меньше двух точек",
                )
            )
            continue
        try:
            axes, axis_name = _position_axes(block, intensity.size)
        except ValueError as exc:
            errors.append(
                localised(
                    f"scan {index}: {exc}",
                    f"balayage {index} : {exc}",
                    f"скан {index}: {exc}",
                )
            )
            continue

        label = sample_name or source.stem
        if len(data_blocks) > 1:
            label = f"{label} – #{index}"
        result.append(
            Scan1D(
                name=label,
                x=axes[axis_name],
                y=intensity,
                source=source,
                axis_name=axis_name,
                metadata={
                    "format": "XRDML",
                    "scan_index": index,
                    "axes": axes,
                },
            )
        )

    if not result:
        detail = "; ".join(errors)
        raise ValueError(
            localised(
                f"Could not read a one-dimensional scan from XRDML: {detail}",
                f"Impossible de lire un balayage unidimensionnel depuis le XRDML : {detail}",
                f"Не удалось прочитать одномерный скан из XRDML: {detail}",
            )
        )
    return result


def read_xy(path: str | Path) -> Scan1D:
    """Прочитать первые два числовых столбца текстового файла."""

    source = Path(path)
    x_values: list[float] = []
    y_values: list[float] = []
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.replace(",", ".").strip().split()
            if len(parts) < 2:
                continue
            try:
                x_value, y_value = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if math.isfinite(x_value) and math.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)
    return Scan1D(
        name=source.stem,
        x=np.asarray(x_values),
        y=np.asarray(y_values),
        source=source,
        axis_name="X",
        metadata={"format": "XY", "axes": {"X": np.asarray(x_values)}},
    )


def read_raw_scans(path: str | Path, *, raw=None) -> list[Scan1D]:
    """Прочитать все одномерные диапазоны Bruker RAW v3/v4."""

    source = Path(path)
    if raw is None:
        try:
            raw = read_bruker_raw(source)
        except (OSError, ValueError) as exc:
            raise ValueError(
                localised(
                    f"Could not read {source.name} as Bruker RAW v3/v4. "
                    f"The file may be damaged or use an unsupported RAW variant.",
                    f"Impossible de lire {source.name} comme fichier Bruker RAW v3/v4. "
                    f"Le fichier est peut-être endommagé ou utilise une variante RAW non prise en charge.",
                    f"Не удалось прочитать {source.name} как Bruker RAW v3/v4. "
                    f"Файл может быть повреждён либо использовать неподдерживаемый вариант RAW.",
                )
            ) from exc
    ranges = [item for item in raw.ranges if item.point_count >= 2]
    if not ranges:
        raise ValueError(
            localised(
                "The RAW file contains no one-dimensional range with at least two points.",
                "Le fichier RAW ne contient aucune plage unidimensionnelle d’au moins deux points.",
                "В RAW нет одномерных диапазонов как минимум с двумя точками.",
            )
        )

    result: list[Scan1D] = []
    for item in ranges:
        axes: dict[str, np.ndarray] = {item.axis_name: item.axis}
        for drive_name in (
            "2Theta",
            "Theta",
            "Omega",
            "Chi",
            "Phi",
            "X-Drive",
            "Y-Drive",
            "Z-Drive",
        ):
            coordinate = item.coordinate(drive_name)
            if coordinate.size == item.point_count and np.any(np.isfinite(coordinate)):
                axes.setdefault(drive_name, coordinate)
        label = source.stem
        if len(ranges) > 1:
            label = f"{label} – #{item.index + 1}"
        result.append(
            Scan1D(
                name=label,
                x=item.axis,
                y=item.intensity,
                source=source,
                axis_name=item.axis_name,
                metadata={
                    "format": f"Bruker RAW v{raw.version}",
                    "range_index": item.index,
                    "wavelength": item.wavelength,
                    "scan_type": item.scan_type,
                    "is_pole_figure": raw.is_pole_figure,
                    "axes": axes,
                },
            )
        )
    return result


def read_scan_file(path: str | Path) -> list[Scan1D]:
    """Определить формат по расширению и прочитать одномерные данные."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xrdml", ".xml"}:
        return read_xrdml(source)
    if suffix == ".raw":
        return read_raw_scans(source)
    if suffix in {".xy", ".txt", ".dat", ".csv"}:
        return [read_xy(source)]
    suffix_text = source.suffix or localised(
        "no extension", "sans extension", "без расширения"
    )
    raise ValueError(
        localised(
            f"Unsupported format: {suffix_text}.",
            f"Format non pris en charge : {suffix_text}.",
            f"Неподдерживаемый формат: {suffix_text}.",
        )
    )
