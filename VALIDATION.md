# XRD Combine 3.0.0b6.1 Validation

Validation date: 22 September 2026.

## Environment

Linux, Python 3.12, PySide6 6.11.2, PyQtGraph 0.14.0,
unit-cell-gui 0.2.1, NumPy 2.5.3, and Qt offscreen platform.

## Automated tests

Run from the project directory:

```bash
python -m unittest discover -s tests -q
```

The suite contains 184 tests, including six new RSM tests. One existing
OpenGL screenshot test is skipped on the offscreen Qt platform, which does
not provide a compatible graphics context. The new tests cover incomplete
RAW grids, both scan geometries, XY angular series, angle/Q conversion,
shared CIF geometry, Project data import, both measured RSM coordinate
views, a calculated map, and persistence of numeric target inputs on
translation refresh.

## RSM improvements in 3.0.0b6.1

The GUI test checks selection of two CIF files, a reflection marker inside
an acquired map cell, removal of the marker, persistence when switching
between angular and Qx/Qz coordinates, and the absence of a false marker
for a CIF whose allowed reflections lie outside the recorded map. It also
checks that wheel and left-drag zoom events are consumed on both plots.

RAW angle fields are read-only and filled from the source scan geometry;
XY angle fields remain editable and fill the third value after two are
provided. The third value is recalculated when the XY series grows.

## Supplied RSM sample

The supplied interrupted Bruker RAW file was opened in the fourth RSM page.
The parser identified 101 Theta ranges with 2Theta as the outer drive and
61 ranges with recorded points. The experimental grid is 101 by 101 and
contains 4,120 missing points (40 empty ranges and 80 absent samples from
the interrupted final range). One PyQtGraph mesh rendered successfully in
each of the angular and Qx/Qz coordinate views.

The supplied RAW reports First omega = 21.725 degrees, Omega step = 0.02
degrees, and Last omega = 23.725 degrees. Real ends near 45.84 degrees
2Theta; Full includes declared, unmeasured ranges to approximately 46.65
degrees. No missing range is filled with synthesized intensity.

The supplied KNbO3 CIF was loaded through the application CIF reader and
its assigned calculated RSM produced reciprocal-lattice points and angular
and Q target coordinates. These checks do not assert absolute sample
orientation without measured instrument alignment information.

## Remaining platform checks

The offscreen environment cannot validate visual appearance and OpenGL on
Windows. A final manual review should include pan and zoom, export to PNG,
large RSM files, and shifting the main window between displays with
different scaling. A calculated RSM shows allowed reciprocal-lattice point
positions and does not predict measured intensity or peak broadening.
