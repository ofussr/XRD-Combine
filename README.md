<p align="center">
  <img src="xrd_workbench/resources/xrd_combine.png" width="220" alt="XRD Combine">
</p>

<h1 align="center">XRD Combine</h1>

XRD Combine is a desktop application for viewing, comparing, correcting, and
interpreting X-ray diffraction data. It brings measured scans, crystal
structures, calculated diffraction patterns, reflection tables, and pole
figures together in one project-oriented interface.

The application is intended for day-to-day work with individual measurements,
sample series, reference phases, and crystallographic models. Original input
files remain unchanged unless an exported result is explicitly saved.

## Main workspace

Files are opened in the **Project data** panel and can then be assigned to the
appropriate page with their checkboxes. The workspace is divided into four
main pages. The panel width and all three table-column widths can be adjusted
with the mouse and are preserved between sessions. Multiple measurements read
from one source file are collected under an expandable source-file node.

### Viewer

The Viewer is used to inspect and compare measured scans and calculated phase
patterns.

- display several measurements and phases on the same plot;
- select the coordinate and intensity axes contained in a measurement;
- overlay compatible phases on a measured 2theta scan or display them
  separately;
- show a calculated phase as reflection sticks or as a broadened profile;
- use linear, logarithmic, square-root, or squared intensity scaling;
- change curve visibility, color, order, and vertical arrangement;
- apply manual coordinate and intensity corrections;
- align data using selected peaks or stored reference reflections;
- calibrate the angular axis from two or three orders of one substrate
  reflection family, with or without a known true peak position;
- switch compatible 2theta data and calculated phase positions to d-spacing
  using one selected wavelength;
- add a corrected result to the project or replace the current working object;
- send selected data to the Comparison window;
- configure axis labels, major and minor ticks, legend, line width, and grid;
- save the current plot as a PNG image.

### Structures

The Structures page combines crystal-structure inspection with diffraction
calculations.

**Structure view**

- display atoms, bonds, the unit-cell box, basis vectors, and polyhedra;
- control visibility, color, and opacity by element or crystallographic site;
- display mixed and partially occupied positions;
- include neighbouring atoms and bonds outside the selected unit cell;
- use color or black-and-white presentation modes;
- apply engraving and face hatching to polyhedra;
- orient the structure along direct or reciprocal lattice directions, or an
  `hkl` plane normal;
- rotate, pan, and zoom the model interactively.

**Calculated pattern**

- calculate a powder diffraction pattern from a CIF structure;
- display individual reflection sticks or a broadened profile;
- set the angular range, minimum intensity, peak width, and radiation lines;
- save the calculated plot as a PNG image.

**Reflection table**

- inspect `hkl`, d-spacing, diffraction angle, wavelength, line weight,
  multiplicity, structure factor, and calculated intensity;
- sort the table by any displayed quantity;
- export the table to CSV.

A **Cell Phase** can also be created manually from a space group and unit-cell
parameters when an atom-free phase is sufficient for the intended operation.

### Pole figures

The Pole figures page supports both measured and calculated pole data.

- open experimental pole-figure data from supported RAW and text formats;
- define angular coordinates manually when they are not stored in the source;
- inspect individual points and their values;
- choose the color map, scale, interpolation, and grid presentation;
- calculate crystallographic poles from a CIF structure or Cell Phase;
- combine first- and second-phase pole sets;
- configure orientations, pole labels, and marker sizes;
- view the corresponding crystal structure beside a calculated pole figure;
- rotate the pole figure and structure synchronously in either direction.

### Reciprocal-space maps (RSM)

The RSM page has separate **Experimental** and **Calculated** tabs.

- Open an RSM RAW file, numbered RAW files, or a series of two-column XY files;
- select the data and a CIF through Project data, or open them from the RSM page;
- show an experimental intensity map in angular coordinates or Qx/Qz,
  in linear or logarithmic scale with selectable color maps;
- keep unfinished RAW ranges as gaps instead of filling unmeasured points;
- switch the plot between **Real** (measured area) and **Full** (all declared
  RAW ranges), without changing the saved data or interpolating gaps;
- for an XY series, treat X as 2Theta and enter two of the first, step, and
  last Omega values; the third value is filled in automatically, and the
  application does not change the underlying scan axes;
- read the first, step, and last Omega positions automatically from RAW data;
- choose a loaded CIF or open another one and add its indexed reflection
  markers to measured cells in the experimental map. Enter the surface normal
  and in-plane reference for that overlay; change the wavelength in the
  common radiation selector;
- calculate reciprocal-lattice points from the already loaded CIF and its
  symmetry, with a surface normal, an in-plane direction, and a target given
  as hkl, angular coordinates, or Qx/Qz;
- configure the scattering-plane tolerance, limits, point labels, and target
  marker, then save a PNG of the map.

Mouse wheel and mouse-drag zoom are disabled on both RSM maps. Right-button
drag still pans; display bounds come from Real/Full or the calculated map's
range fields.

RSM uses the same radiation setting as the other pages. The calculated map
shows allowed reciprocal-lattice points; it does not synthesize measured
intensities or model an instrument resolution function.

### Comparison

The Comparison window is designed for presenting related scans as a set.

- assemble measured curves and reference or substrate scans;
- switch between overlaid and vertically offset presentation;
- control curve colors, order, and intensity scaling;
- browse the result as a thumbnail gallery and open a detailed plot;
- copy or save the resulting figures.

## Supported files

| Format | Typical use |
| --- | --- |
| XRDML / XML | Measured scans with instrument axes and metadata |
| Bruker RAW v3 / v4 | Measured scans, supported pole figures, and reciprocal-space maps |
| XY / TXT / DAT / CSV | Two-column diffraction data and supported pole data |
| CIF | Crystal structures, calculated patterns, reflections, and poles |

The exact information available after import depends on the contents of the
source file. For example, an XRDML file may provide several coordinate or
intensity axes, while a simple two-column file normally contains only one of
each.

Bruker RAW v3 and v4 files are read by one shared parser for Viewer, reciprocal
space maps, and experimental pole figures. Each range retains a `ScanPath`
describing every known moving drive, including coupled motion, while the file
retains a `MeasurementGeometry` describing its inner and outer scan axes. For
RAW v3, confirmed range-header codes identify 2Theta, Theta, and Phi motion
directly. This covers ordinary scans, rocking curves, azimuth scans, pole
figures, and Theta/2Theta reciprocal-space maps in the validated Bruker D8
files. Unknown codes are retained without guessing a physical axis.

To export decoded RAW metadata and complete source headers without intensity
arrays, run:

```bash
python -m xrd_workbench.bruker_raw measurement.raw --json report.json
```

## Installation

Python 3.10 or newer is recommended. Using a dedicated virtual environment is
strongly advised.

```bash
python -m pip install -r requirements.txt
```

Start the application with:

```bash
python run_xrd_combine.py
```

The package entry point is also available:

```bash
python -m xrd_workbench
```

Files may be supplied on the command line:

```bash
python run_xrd_combine.py measurement.xrdml structure.cif
```

To print the installed application version without opening a window:

```bash
python run_xrd_combine.py --version
```

## Quick start

1. Open one or more measurement, text, RAW, or CIF files from the Project data
   panel.
2. Select **Viewer**, **Structures**, **Pole figures**, or **RSM**.
3. Enable the required objects with their checkboxes.
4. Expand the controls on the left to choose axes, display modes, limits, and
   calculation parameters.
5. Save plots, reflection tables, or corrected data explicitly when needed.

## Mouse controls

For two-dimensional plots:

- left button: select a point or pole where selection is available;
- right-button drag: pan the plot;
- rectangle-zoom tool: enlarge a selected region;
- Home button: restore the full view.

For the three-dimensional structure view:

- left-button drag: rotate the structure;
- right-button drag: pan the structure;
- mouse wheel: zoom.

## Interface settings

XRD Combine includes Russian, English, and French interface languages, along
with system, light, and dark application themes. Radiation settings and stored
reference peaks can be configured from the application menus.

Compiled Windows builds also provide experimental per-user file associations
under **About > Debug**. XRDML, RAW, XY, and CIF associations can be registered
or removed without administrator rights. The application does not claim the
generic XML, TXT, DAT, or CSV extensions.

## Packaging the application icon

For PyInstaller, use the ICO as the executable icon and include both runtime
images:

```text
--icon "xrd_workbench\resources\xrd_combine.ico" --add-data "xrd_workbench\resources\xrd_combine.ico:." --add-data "xrd_workbench\resources\xrd_combine.png:."
```

For Nuitka, use:

```text
--windows-icon-from-ico=xrd_workbench/resources/xrd_combine.ico --include-data-file=xrd_workbench/resources/xrd_combine.ico=xrd_combine.ico --include-data-file=xrd_workbench/resources/xrd_combine.png=xrd_combine.png
```

## Tests

Run the automated test suite from the project directory:

```bash
python -m unittest discover -s tests -v
```

See `VALIDATION.md` for the validation record supplied with this release.

## Documentation

- `RELEASE_NOTES_3.0.0b6.1.md` — changes introduced in version 3.0.0b6.1;
- `RELEASE_NOTES_3.0.0b6.md` — initial RSM integration;
- `VALIDATION.md` — automated and manual validation record;
- `ARCHITECTURE.md` — internal module boundaries and design notes;
- `THIRD_PARTY_NOTICES.txt` — notices for bundled third-party components.

## License

XRD Combine is distributed under the MIT License. See `LICENSE` for the full
license text.

Copyright (c) 2026 Mikhail Mirushchenko.

## How to cite

If you use `XRD Combine` in academic work, please cite the Zenodo record:

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22809711-blue.svg)](https://doi.org/10.5281/zenodo.22809711)

You can also use the metadata provided in [`CITATION.cff`](CITATION.cff).
