"""Toolkit-independent XRDML reader."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Type

import numpy as np

from ..models.data_errors import XRDDataError
from ..models.scan import Scan1D


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
    data_points: ET.Element,
    point_count: int,
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
        raise XRDDataError("xrdml_no_axis")
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


def read_xrdml(
    path: str | Path,
    *,
    scan_factory: Type[Scan1D] = Scan1D,
) -> list[Scan1D]:
    """Read all one-dimensional scans and coordinate axes from XRDML."""

    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise XRDDataError("xrdml_invalid_xml", detail=str(exc)) from exc

    sample_node = _first(root, "sampleName")
    if sample_node is None:
        sample_node = _first(root, "name")
    sample_name = (sample_node.text or "").strip() if sample_node is not None else ""
    data_blocks = list(_children(root, "dataPoints"))
    if not data_blocks:
        raise XRDDataError("xrdml_no_data")

    result: list[Scan1D] = []
    errors: list[XRDDataError] = []
    for index, block in enumerate(data_blocks, start=1):
        intensity_node = _first(block, "intensities")
        if intensity_node is None:
            intensity_node = _first(block, "counts")
        if intensity_node is None:
            errors.append(XRDDataError("xrdml_scan_no_intensity", index=index))
            continue
        intensity = _numbers(intensity_node.text)
        if intensity.size < 2:
            errors.append(XRDDataError("xrdml_scan_too_short", index=index))
            continue
        try:
            axes, axis_name = _position_axes(block, intensity.size)
        except XRDDataError:
            errors.append(XRDDataError("xrdml_scan_no_axis", index=index))
            continue

        label = sample_name or source.stem
        if len(data_blocks) > 1:
            label = f"{label} – #{index}"
        result.append(
            scan_factory(
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
        raise XRDDataError("xrdml_no_valid_scan", issues=tuple(errors))
    return result
