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
- `ui_qt` owns widgets, dialogs, the Qt event loop, the primary PyQtGraph plot
  adapters, and the retained Matplotlib compatibility adapters;
- `unit-cell-gui` receives immutable display input and owns unit-cell painting,
  hatching, camera interaction, and orientation events;
- neither the core calculation layer nor `unit-cell-gui` owns project files or
  application navigation.

These rules and the absence of Tkinter imports are checked by
`tests/test_architecture.py`.

## Source-file intake and startup

`ui_qt.app` parses `--version` before importing GUI dependencies and configures
shared OpenGL before constructing QApplication. `ui_qt.single_instance` elects
one owner with a lifetime QLockFile whose identity is independent of version and
portable-copy location. A private local mailbox uses atomic JSON publication,
request IDs, generation IDs and acknowledgements; its worker thread continues
accepting paths while heavy imports and window construction block the main thread.
It transports only paths and activation requests, never project objects. A stale
lock is recovered only after Qt detects that the owning process is dead. Failed
handoff reports an error rather than creating a duplicate main window.

The secondary process grants Windows foreground permission before publishing its
request and exits before importing MainWindow. The primary attaches its receiver
after constructing its window. The receiver restores/activates the window and
queues imports on the UI thread, deferring them while a modal dialog is active.
Each queued batch captures its destination workspace when received.

`ui_qt.file_drop` filters file-URL drag events across the main window's nested
widgets, shows an accepted-drop hint and rejects unsupported batches, directories,
remote URLs and saved sessions. Ordinary text/internal drags retain normal handling.
The shared source-extension policy lives in `services.project_files`; dropped,
forwarded and file-dialog source paths all enter MainWindow.import_paths and the
existing ProjectFileService. Saved projects/tabs retain SessionActions routing.

## Main components

- `xrd_workbench.models.project` owns documents, assignments, identity,
  replacement history, and project events.
- `xrd_workbench.models.analysis.MeasurementAnalysis`, owned by `ProjectStore`,
  is the single owner of accepted peaks, stable peak IDs, spectral relationships,
  hkl assignments and per-measurement/per-axis backgrounds. `SessionPeak` and
  `BackgroundAnchors` live in `models`; numeric arrays are read-only snapshots.
  Collections are read through snapshots and changed through the model API.
  Model events coalesce compound edits before updating the Viewer and tables.
  `services.analysis` owns joint peak-fit orchestration, background recalculation
  and companion suggestions. Failed group fits commit no partial result.
  Project replacement/removal invalidates measurement results or phase-linked
  hkl before project notifications, even when no Viewer exists. Workspace
  unassignment of a scan from Viewer clears its analysis on all axes while
  retaining the project document and assignments to other workspaces. This
  cleanup belongs to ProjectStore and also runs without a Viewer widget.
  Plot visibility changes preserve results. Viewer tracks source payload identity
  so replacements also discard provisional previews. Cursor/drag state,
  provisional fits and dialogs remain in Qt. Accepted analysis is included in project/Viewer-tab serialization.
  Comparison is outside workflow/tab persistence;
  its existing display-bound preset files remain sufficient.
- `xrd_workbench.services.project_files` coordinates injected CIF, RAW, and
  scan readers with the project model and records source checksums at import.
  `services.correction` appends replayable per-axis operations to scan metadata;
  cumulative scalar metadata alone is not used to reconstruct corrected scans.
- `models.viewer.PlotAppearance` holds accepted PyQtGraph custom labels,
  line width, grid/legend and tick visibility. Viewer and structure calculation
  state own separate instances; the plot adapter reads and writes those fields.
- `xrd_workbench.io.session_codec` registers explicit model classes and accepted
  fields, with typed validation and JSON tags for tuples, mappings, numeric
  analysis arrays and sets. Runtime payloads, caches and callbacks are excluded.
  The unbounded upper d limit for a zero-degree pole range is encoded as `null`,
  never a nonstandard JSON infinity. No pickle or dynamic imports are used.
- `xrd_workbench.services.sessions` owns the versioned project/tab manifest,
  relative/absolute source references, checksums, source selectors, manual-cell
  recipes, analysis and ordered correction replay. It validates identities and
  relationships before reading sources, stages a fresh store, restores payload
  bindings, and rebuilds derived cache keys and pole reflections. A source
  resolver supplies locate/skip/cancel/use-changed decisions without a Qt import.
  Changed or missing sources invalidate dependent results before rendering.
  Tab import stages a copy, replaces only its workspace, remaps identities and
  reuses matching source/recipe objects without replacing other workspaces.
  Saves use a temporary file in the destination directory and atomic replacement;
  source files cannot be used as a save destination.
- `ui_qt.session_actions` adapts File commands to these services, resolves source
  issues and installs replacement pages only after successful model restoration.
  Old store listeners, auxiliary analysis windows and workers are detached.
  Whole-project files include validated application appearance; tab files keep
  current appearance and restore shared radiation. Appearance restoration does
  not overwrite global language/theme/renderer/atom-colour preferences.
- `xrd_workbench.models.scan` and `xrd_workbench.io` own one-dimensional scan
  data, axis selection, cloning, and format readers/writers.
- `xrd_workbench.bruker_raw` is the single Bruker RAW v3/v4 parser used by
  ordinary scans, reciprocal-space maps, and experimental pole figures. Its
  `ScanPath` records per-point drive motion and `MeasurementGeometry` records
  the inner and outer dimensions of a complete measurement.
- `xrd_workbench.models.viewer` owns toolkit-independent viewer presentation
  state and numeric navigation geometry. `ProjectStore.viewer` owns its lifetime,
  independently of `ViewerPage`. Assignment creates working scan clones before
  notifications; replacement refreshes clones and angle contexts in the model.
  Unassignment/removal clears object-local presentation and geometry settings.
  Visibility, colours, order, active axes, correction previews, peak-sum display,
  selection, processing result mode and default background spacing live here.
  Plot state contains scales, wavelength, phase rendering parameters, accepted
  manual limit entries and the current measurement/phase viewports. Viewports
  use display coordinates shared by both renderers: log limits remain physical
  counts, while sqrt/square limits use transformed Y coordinates. Qt adapters
  record pan/zoom changes and restore views after their initial draw. Rendering
  caches, source-identity bookkeeping and worker requests are runtime data, excluded from the project-file format.
- `xrd_workbench.models.pole_state` defines accepted calculated and experimental
  pole-figure state. `ProjectStore.poles` owns phase/source UIDs, layer roles,
  centring, orientation matrices, display settings, accepted ranges and manual
  experimental geometry. Unassignment, replacement and overlay promotion happen
  before notifications, including when no pole page exists. Removing a primary
  phase promotes the surviving layer's geometry, selection and styling. Runtime
  payloads, measurement grids, projected points and reflection/intensity caches
  are excluded from the settings codec and rebuilt from referenced sources.
  `ui_qt.pole_state_binding` exposes these fields to the controllers and commits validated display values. Numeric drafts,
  transient hkl outlines, drag state, timers and widgets stay in Qt. Experimental
  and calculated plot adapters record accepted navigation and restore it on
  recreation. Circular plots retain equal aspect; a different widget shape may
  expand one axis to fit the saved view. The compact structure preview owns a
  separate `StructureViewState` in the calculated pole state; its orientation
  still follows the pole controller.
- `xrd_workbench.models.structures_state`, owned by `ProjectStore.structures`,
  holds the assigned phase, structure orientation, component/site visibility and
  styling, display options and numeric camera state. Pan is normalized to canvas
  width/height so a different widget size preserves relative displacement.
  `ui_qt.structure_canvas_state` records unit-cell-gui camera interactions.
  Both structure previews restore accepted settings before applying background
  geometry results. Calculated patterns and reflection tables retain accepted
  parameters, plot style and viewport, sorting, column widths, selection by
  `(hkl, radiation, wavelength)`, and active subtab. Replacement/unassignment
  clears source-local geometry, styles, navigation, selection and reflection
  caches before notifications. General display/calculation options remain.
- `xrd_workbench.models.rsm_state`, owned by `ProjectStore.rsm`, holds effective
  experimental sources, accepted XY angle series and the derived field, coordinate
  mode, intensity scale, colour map, extent, overlay phase and sample orientation.
  Calculated state holds the source, accepted orientation, target, per-coordinate
  display ranges, labels and target marker. Both maps retain numeric navigation
  and the workspace retains its active subtab. RAW sources take priority over
  assigned XY scans. A changed effective source set clears map-specific geometry,
  navigation and caches; replacing/removing an overlay phase invalidates its
  geometry independently. Qt owns invalid or unfinished numeric drafts, plot
  markers, renderer objects and widget bookkeeping. Payload references, map data,
  reflection rows and request/cache identities are runtime data, not a file schema.
- `xrd_workbench.models.diffraction` and
  `xrd_workbench.services.diffraction` own powder-reflection and calculated
  profile data and calculations.
- `xrd_workbench.models.crystal` owns the parsed crystal, symmetry expansion,
  bases, d spacings, and display atoms.
- `xrd_workbench.models.phase_scan` defines oriented scan geometry, per-phase
  orientation, and reflection rows with a separate plotted angular coordinate.
  `ViewerState.phase_scan` owns the accepted mode, reference measurement, selected
  phase, maximum index, numeric angle bounds and orientations. Angle contexts
  use stable `(measurement_uid, axis_name)` keys. Reading fixed measurement
  metadata and seeding the default ±0.1-degree bounds requires no GUI.
  `xrd_workbench.services.phase_scan` enumerates signed hkl, checks systematic
  and structure-factor extinctions, and intersects their directions with a
  four-circle scan path using the existing RSM sample basis. Powder families
  are not reused because their merged labels lose signed reciprocal vectors.
  `ui_qt.phase_scan_controls` adapts this model to controls and owns unfinished or
  invalid input drafts. Invalid committed drafts block the corresponding live
  calculation without replacing accepted model values. Recreated controls use
  accepted state. Viewer runs calculations in its worker pool; requests include the
  geometry, orientation, radiation, limits, and payload identity. Superseded
  requests are cancelled and their results are ignored on the GUI thread.
  Fixed-angle windows are intersected using exact four-circle solutions
  parameterised by phi. Analytic boundary roots divide the solution curve
  into segments; feasible segments and boundary points are checked before
  selecting one representative position near the nominal settings. A phi
  scan caches a full turn at zero offset; offset changes shift and crop its
  periodic copies on the GUI thread without recalculating structure factors.
  Viewer cursor updates use separate labels beside the status counters and
  the coordinate mapping of each rendering adapter, with no plot redraw.
- `xrd_workbench.services.structure_scene` constructs cached bonds and
  coordination polyhedra without drawing them.
- `xrd_workbench.ui_qt.structure_preparation` observes project imports and
  prepares geometry and immutable renderer scenes in a Python worker pool.
  A GUI-thread timer collects results; widgets and OpenGL are never accessed
  by the worker. Structures and pole previews share the per-payload cache.
  Removed/replaced payloads invalidate cached or pending results. Closing the
  main window stops result polling and cancels queued work without waiting.
- `xrd_workbench.models.pole_figure` and
  `xrd_workbench.services.pole_figure` own pole data, projections, orientation
  matrices, rotations, marker scaling, and label placement.
  The service also converts reflection intervals between d and 2theta using
  the same wavelength as calculated poles. The Qt controller owns a separate
  hkl locator and a restartable two-second timer. Its geometric poles are
  projected independently of the filtered reflection list in both renderers;
  redraws follow the current phase orientation. Locating a pole does not modify
  the persistent selection, centring or range. Removing its phase clears the
  transient locator.
- `xrd_workbench.services.experimental_pole` prepares pole grids from the
  shared RAW model or numbered XY exports without interpolation.
- `xrd_workbench.services.rsm` constructs measured RSM grids from shared
  RAW/scan models, preserves incomplete ranges, and computes reciprocal-lattice
  point coordinates from the shared crystal model without a second CIF parser.
  It also calculates bounds of measured or declared cells and selects only
  reflections that intersect an acquired cell; the UI owns plot markers.
- `xrd_workbench.models.substrate_compare` and
  `xrd_workbench.services.substrate_compare` own comparison groups and plot
  preparation. Comparison remains outside workflow/tab persistence and retains
  its existing JSON display-bound preset files.
- `xrd_workbench.localization` owns stable translation keys and catalogues for
  English, French, and Russian.
- `xrd_workbench.ui_qt` owns the main window, Viewer, Structures, pole figures,
  measured and calculated RSM pages,
  comparison, reference-peak editor, application themes, and Qt controls.
- `xrd_workbench.ui_qt.plot_renderer` selects PyQtGraph by default and owns the
  persisted diagnostic Matplotlib override exposed from the About window.
- `xrd_workbench.indexing` contains GUI-independent, experimental Visser and
  Boultif–Louër numerical methods, shared scientific models, service validation,
  bibliography, and Niggli reduction. Its public residual convention is
  observed minus calculated. Original candidate bases and hkl are retained;
  reduced metrics and verified integer basis transforms identify equivalents.
- `ui_qt.debug_features` owns the persisted, initially disabled indexation
  opt-in. `ui_qt.indexing_dialog` snapshots primary session peaks through the
  Viewer adapter and uses a Python worker plus GUI timer polling. Workers
  receive no Qt objects; cancellation checks occur in numerical loops.
  Viewer invalidates changed sources and transfers accepted assignments by
  stable peak ID after the existing Cell Phase editor confirms the phase.
  `pyqtgraph_pole` is the calculated-pole surface;
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
