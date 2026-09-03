"""Structured data errors that can be translated by a user-interface adapter."""

from __future__ import annotations

from typing import Any


class XRDDataError(ValueError):
    """A toolkit-independent error identified by a stable machine code."""

    def __init__(self, code: str, **context: Any) -> None:
        self.code = code
        self.context = context
        super().__init__(self._english_message())

    def _english_message(self) -> str:
        axis = self.context.get("axis", "")
        source_name = self.context.get("source_name", "")
        suffix = self.context.get("suffix", "no extension")
        index = self.context.get("index", "")
        messages = {
            "scan_arrays": "Coordinates and intensities must be one-dimensional arrays.",
            "scan_too_short": "The dataset contains fewer than two valid points.",
            "scan_axis_missing": f"Axis {axis!r} is not available for this dataset.",
            "scan_axis_too_short": f"Axis {axis!r} contains fewer than two valid points.",
            "scan_text_axis_only": "Only text datasets support axis assignment.",
            "xrdml_invalid_xml": f"Invalid XML: {self.context.get('detail', '')}",
            "xrdml_no_data": "No dataPoints block was found in the XRDML file.",
            "xrdml_no_axis": "The XRDML scan contains no readable coordinate axis.",
            "xrdml_scan_no_intensity": f"scan {index}: no intensities/counts",
            "xrdml_scan_too_short": f"scan {index}: fewer than two points",
            "xrdml_scan_no_axis": (
                f"scan {index}: The XRDML scan contains no readable coordinate axis."
            ),
            "raw_unreadable": (
                f"Could not read {source_name} as Bruker RAW v3/v4. "
                "The file may be damaged or use an unsupported RAW variant."
            ),
            "raw_no_ranges": (
                "The RAW file contains no one-dimensional range with at least two points."
            ),
            "unsupported_format": f"Unsupported format: {suffix}.",
            "viewer_phase_height": "The phase-panel height must be finite.",
            "viewer_overlay_height": "The overlaid phase height must be finite.",
            "viewer_no_colours": "The viewer colour palette is empty.",
            "viewer_limits_order": "The lower limit must be below the upper limit.",
            "correction_values": "Correction values must be finite.",
            "correction_y_factor": "The Y scale must be positive.",
            "correction_result_mode": "Unsupported correction result mode.",
            "correction_xrdml_source": "XRDML export requires an XRDML source file.",
            "correction_xrdml_range": "The selected XRDML range was not found in the source file.",
            "correction_xrdml_intensity": "The XRDML intensity array was not found.",
            "correction_xrdml_array_length": "The processed and source XRDML arrays have different lengths.",
            "correction_xrdml_axis_length": "An XRDML coordinate axis has an unexpected length.",
            "peak_fit_scipy": "SciPy is required for peak fitting.",
            "peak_fit_points": "Select at least seven data points around the peak.",
            "peak_fit_flat": "The selected region contains no measurable peak.",
        }
        if self.code == "xrdml_no_valid_scan":
            issues = self.context.get("issues", ())
            detail = "; ".join(str(issue) for issue in issues)
            return f"Could not read a one-dimensional scan from XRDML: {detail}"
        return messages.get(self.code, self.code)
