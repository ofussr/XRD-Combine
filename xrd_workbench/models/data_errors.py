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
            "cif_lex": (
                f"Could not parse CIF line {self.context.get('line_number', '')}: "
                f"{self.context.get('reason', '')}."
            ),
            "cif_loop_no_columns": "No column names follow loop_.",
            "cif_loop_width": (
                "The CIF loop value count is not divisible by the column count "
                f"({self.context.get('value_count', '')} and "
                f"{self.context.get('column_count', '')})."
            ),
            "cif_field_missing": (
                f"CIF field {self.context.get('field', '')} has no value."
            ),
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
            "diffraction_no_radiation": "No spectral line is selected.",
            "diffraction_limits": "The limits must satisfy 0 <= minimum < maximum < 180 degrees.",
            "diffraction_radiation_positive": "Wavelengths and relative weights must be positive.",
            "diffraction_factors_empty": "Could not read the atomic scattering factors.",
            "diffraction_factors_file_missing": (
                "f0_WaasKirf.dat was not found. Place it beside the application."
            ),
            "diffraction_factor_missing": (
                "No atomic scattering factors are available for "
                f"{self.context.get('element', '')}."
            ),
            "diffraction_singular_metric": "The metric matrix is singular.",
            "diffraction_symmetry": (
                "Could not parse symmetry operation "
                f"{self.context.get('expression', '')!r}."
            ),
            "diffraction_profile_limits": "The profile limits must be finite and increasing.",
            "diffraction_profile_fwhm": "FWHM must be a positive finite number.",
            "diffraction_profile_points": "A powder profile requires at least two grid points.",
            "diffraction_profile_scale": "The profile normalization must be positive and finite.",
            "crystal_cif_number_undefined": "The CIF contains an undefined numeric value.",
            "crystal_cif_number_invalid": (
                f"Invalid CIF number: {self.context.get('value', '')!r}."
            ),
            "crystal_symmetry": (
                "Could not parse symmetry operation "
                f"{self.context.get('expression', '')!r}."
            ),
            "crystal_cell_degenerate_gamma": "Degenerate unit cell: sin(gamma) = 0.",
            "crystal_cell_degenerate": "The unit-cell parameters define a degenerate cell.",
            "crystal_cell_missing": (
                "Unit-cell parameters are missing from the CIF: "
                + ", ".join(self.context.get("fields", ()))
            ),
            "crystal_non_p1_without_symmetry": (
                "The CIF declares a non-P1 space group but contains no explicit "
                "symmetry operations. Calculating it as P1 would be incorrect."
            ),
            "crystal_d_000": "The interplanar spacing is undefined for (0 0 0).",
            "pole_d_positive": "The d limits must be positive.",
            "pole_d_order": "The lower d limit cannot exceed the upper limit.",
            "pole_wavelength_positive": "The wavelength must be positive.",
            "pole_reflection_limit": (
                "The lower d limit requires testing more than two million "
                "reciprocal-lattice nodes. Increase the lower d limit."
            ),
        }
        if self.code == "xrdml_no_valid_scan":
            issues = self.context.get("issues", ())
            detail = "; ".join(str(issue) for issue in issues)
            return f"Could not read a one-dimensional scan from XRDML: {detail}"
        return messages.get(self.code, self.code)
