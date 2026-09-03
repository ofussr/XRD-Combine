"""Compatibility exports for radiation models and the Tkinter selector.

New code should import the data model from :mod:`xrd_workbench.models.radiation`
and the current widget from :mod:`xrd_workbench.ui_tk.radiation` explicitly.
"""

try:
    from .models.radiation import (
        PRESETS,
        RadiationPreset,
        RadiationSettings,
        RadiationTuple,
        validate_radiation_lines,
    )
    from .ui_tk.radiation import CustomRadiationDialog, RadiationSelector
except ImportError:  # pragma: no cover - direct module execution
    from models.radiation import (
        PRESETS,
        RadiationPreset,
        RadiationSettings,
        RadiationTuple,
        validate_radiation_lines,
    )
    from ui_tk.radiation import CustomRadiationDialog, RadiationSelector

__all__ = [
    "PRESETS",
    "CustomRadiationDialog",
    "RadiationPreset",
    "RadiationSelector",
    "RadiationSettings",
    "RadiationTuple",
    "validate_radiation_lines",
]
