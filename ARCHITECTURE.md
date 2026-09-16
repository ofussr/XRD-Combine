# XRD Combine architecture

Version 3 has one PySide6 interface. Tkinter and the historical Matplotlib
structure renderer are not part of the application package.

## Dependency direction

```text
ui_qt ────────────────┐
  │                   │
  ▼                   ▼
services ────────► unit-cell-gui
  │                 (rendering only)
  ▼
models ◄──────────── io
  ▲
  └──── localization
```

The intended rules are:

- `models`, `services`, `io`, and `localization` import no GUI toolkit and no
  GUI-specific Matplotlib backend;
- `ui_qt` owns widgets, dialogs, the Qt event loop, and both temporary plot
  adapters: Matplotlib and PyQtGraph;
- `unit-cell-gui` receives immutable display input and owns unit-cell painting,
  hatching, camera interaction, and orientation events;
- neither the core calculation layer nor `unit-cell-gui` owns project files or
  application navigation.

These rules and the absence of Tkinter imports are checked by
`tests/test_architecture.py`.

## Main components

- `xrd_workbench.models.project` owns documents, assignments, identity,
  replacement history, and project events.
- `xrd_workbench.services.project_files` coordinates injected CIF, RAW, and
  scan readers with the project model.
- `xrd_workbench.models.scan` and `xrd_workbench.io` own one-dimensional scan
  data, axis selection, cloning, and format readers/writers.
- `xrd_workbench.models.viewer` owns toolkit-independent viewer presentation
  state and numeric navigation geometry.
- `xrd_workbench.models.diffraction` and
  `xrd_workbench.services.diffraction` own powder-reflection and calculated
  profile data and calculations.
- `xrd_workbench.models.crystal` owns the parsed crystal, symmetry expansion,
  bases, d spacings, and display atoms.
- `xrd_workbench.services.structure_scene` constructs cached bonds and
  coordination polyhedra without drawing them.
- `xrd_workbench.models.pole_figure` and
  `xrd_workbench.services.pole_figure` own pole data, projections, orientation
  matrices, rotations, marker scaling, and label placement.
- `xrd_workbench.services.experimental_pole` reads and prepares measured pole
  grids without interpolation.
- `xrd_workbench.models.substrate_compare` and
  `xrd_workbench.services.substrate_compare` own comparison groups and plot
  preparation.
- `xrd_workbench.localization` owns stable translation keys and catalogues for
  English, French, and Russian.
- `xrd_workbench.ui_qt` owns the main window, Viewer, Structures, pole figures,
  comparison, reference-peak editor, application themes, and Qt controls.
- `xrd_workbench.ui_qt.plot_renderer` owns the temporary persisted renderer
  selection. `pyqtgraph_pole` is the experimental calculated-pole surface;
  `pyqtgraph_experimental` draws the original measured polar cells;
  `pyqtgraph_viewer` renders the main one-dimensional Viewer. The shared
  `pyqtgraph_interaction` module defines right-button panning for all three
  surfaces. Calculations, Viewer state, phase reflections, and RAW/XY grid
  preparation remain in `models` and `services` and are shared with the
  Matplotlib surfaces.

## Unit-cell display boundary

`xrd_workbench.ui_qt.unit_cell_adapter` converts the scene calculated by
`services.structure_scene` into a `unit_cell_gui.Scene`. The external package
receives:

- ready Cartesian atom positions and occupancy components;
- ready bonds and polygonal polyhedron faces;
- basis vectors, labels, style overrides, and display options;
- the current 3×3 orientation matrix.

It does not receive a CIF path, a parser, a symmetry engine, diffraction state,
or a project store. Orientation changes emitted by the widget are sent back to
the calculated pole-figure controller, which provides bidirectional synchronized
rotation.

## Entry points and versioning

`run_xrd_combine.py` and `python -m xrd_workbench` both call the lazy
`xrd_workbench.ui_qt` entry point. There is one application version in
`xrd_workbench.version.APP_VERSION` and one dependency file,
`requirements.txt`.
