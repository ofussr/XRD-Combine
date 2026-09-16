# XRD Combine 3.0.0b1 Release Notes

Version 3.0.0b1 is the first beta release of XRD Combine 3. It completes the
transition of the migrated plots to PyQtGraph as the standard rendering engine
and introduces new defaults for the main Viewer and calculated structure
patterns.

## Changes in 3.0.0b1

- PyQtGraph is now the default renderer in the main Viewer, Comparison,
  calculated diffraction patterns, and pole figures.
- The temporary **Edit > Renderer (test)** menu item has been removed.
- Matplotlib remains available only as a diagnostic fallback. Its selector is
  located under **About > Debug**, and the selected renderer is retained
  between application sessions.
- The main Viewer now uses logarithmic intensity scaling by default.
- A CIF phase opened in the Viewer now starts in Overlay mode even when no
  measured scan has been loaded.
- Adding a compatible 2theta scan preserves Overlay mode. Adding a scan with
  an incompatible coordinate axis, such as omega or chi, automatically moves
  the phase to Separate mode.
- Calculated diffraction patterns on the Structures page now use reflection
  sticks rather than a Gaussian profile by default.
- Cell Phase numeric fields now accept either a decimal point or a decimal
  comma, independently of the operating-system locale.

## Compatibility

- Python 3.10 or newer;
- PySide6 and PyQtGraph are required;
- `unit-cell-gui>=0.2.1,<0.3`;
- a renderer choice saved by an earlier test build is intentionally ignored,
  so the first 3.0.0b1 launch uses PyQtGraph.

## Validation

The source tree contains 153 automated tests. In the release environment, 152
tests passed and one OpenGL screenshot test was skipped because the Qt
offscreen platform did not provide a graphics context. The same test suite was
also run against the unpacked release contents.

See `VALIDATION.md` for the complete validation record.

## Installation

Install the dependencies and start the application from the release directory:

```bash
python -m pip install -r requirements.txt
python run_xrd_combine.py
```

For a description of the application, supported data, and normal workflow, see
`README.md`.
