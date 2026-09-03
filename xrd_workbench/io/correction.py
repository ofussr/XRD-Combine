"""Export corrected scans without depending on file dialogs or GUI toolkits."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..models.data_errors import XRDDataError
from ..models.scan import Scan1D
from ..models.viewer import axis_key


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def write_processed_xrdml(scan: Scan1D, path: str | Path) -> None:
    """Write one processed range into a copy of its source XRDML file."""

    source = Path(scan.source)
    if source.suffix.lower() not in {".xrdml", ".xml"}:
        raise XRDDataError("correction_xrdml_source")
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag.startswith("{"):
        ET.register_namespace("", root.tag[1:].split("}", 1)[0])
    blocks = [
        node for node in root.iter() if _xml_local_name(node.tag) == "dataPoints"
    ]
    scan_index = int(scan.metadata.get("scan_index", 1)) - 1
    if not 0 <= scan_index < len(blocks):
        raise XRDDataError("correction_xrdml_range")
    data_points = blocks[scan_index]
    intensity_node = next(
        (
            node
            for node in data_points.iter()
            if _xml_local_name(node.tag) in {"intensities", "counts"}
        ),
        None,
    )
    if intensity_node is None:
        raise XRDDataError("correction_xrdml_intensity")
    y_values = np.asarray(scan.metadata.get("_base_y", scan.y), dtype=float)
    original_count = len((intensity_node.text or "").split())
    if y_values.size != original_count:
        raise XRDDataError("correction_xrdml_array_length")
    intensity_node.text = " ".join(f"{value:.10g}" for value in y_values)

    axes = {
        axis_key(name): np.asarray(values, dtype=float)
        for name, values in scan.metadata.get("axes", {}).items()
    }
    for positions in (
        node
        for node in data_points.iter()
        if _xml_local_name(node.tag) == "positions"
    ):
        values = axes.get(axis_key(positions.attrib.get("axis", "")))
        if values is None:
            continue
        if values.size != y_values.size:
            raise XRDDataError("correction_xrdml_axis_length")
        nodes = {_xml_local_name(node.tag): node for node in positions.iter()}
        listed = nodes.get("listPositions")
        common = nodes.get("commonPosition")
        start = nodes.get("startPosition")
        end = nodes.get("endPosition")
        if listed is not None:
            listed.text = " ".join(f"{value:.10g}" for value in values)
        elif common is not None:
            common.text = f"{float(values[0]):.10g}"
        elif start is not None and end is not None:
            start.text = f"{float(values[0]):.10g}"
            end.text = f"{float(values[-1]):.10g}"
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_processed_xy(scan: Scan1D, path: str | Path) -> None:
    np.savetxt(path, np.column_stack((scan.x, scan.y)), fmt="%.10g")


def write_processed_scan(scan: Scan1D, path: str | Path) -> None:
    destination = Path(path)
    if destination.suffix.lower() in {".xrdml", ".xml"}:
        write_processed_xrdml(scan, destination)
    else:
        write_processed_xy(scan, destination)


__all__ = ["write_processed_scan", "write_processed_xrdml", "write_processed_xy"]
