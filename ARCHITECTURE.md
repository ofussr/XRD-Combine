# XRD Combine architecture

The 2.9.x series prepares the scientific code for a PySide6 interface while
keeping the released Tkinter application operational.

## Dependency direction

```text
    ui_tk / ui_qt
        |
        v
    services
        |
        v
      models
```

`models` and `services` must not import Tkinter, PySide6, or a GUI-specific
Matplotlib backend. This rule is checked by `tests/test_architecture.py`.

## Current separated components

- `xrd_workbench.models.project` owns project documents, assignments, identity,
  replacement history, and listener events.
- `xrd_workbench.services.project_files` coordinates injected CIF, RAW, and
  one-dimensional scan readers with the project model.
- `xrd_workbench.models.radiation` owns radiation presets, custom lines, and
  their validation. The application keeps one `RadiationSettings` instance for
  the viewer, calculated pattern, reflection table, and calculated pole figure.
- `xrd_workbench.models.cell_phase` defines the shared atom-free phase document.
- `xrd_workbench.models.scan` owns one-dimensional measurements, coordinate-axis
  selection, and cloning.
- `xrd_workbench.models.viewer` owns ordered viewer items, visibility, colours,
  pending X/Y transforms, plot-layout state, numeric limits, and scrollbar
  navigation. It also defines toolkit-independent shared-line and stacked-row
  geometry for overlaid calculated phases.
- `xrd_workbench.models.correction` defines validated correction requests and
  result modes.
- `xrd_workbench.models.diffraction` defines diffraction atoms, structures,
  reflection rows, scattering-factor types, and calculated profile data.
- `xrd_workbench.models.crystal` defines the shared CIF crystal, direct and
  reciprocal bases, symmetry operations, expanded atoms, d spacings, and the
  display-atom geometry used by both structure views.
- `xrd_workbench.models.pole_figure` defines calculated reflection and projected
  pole data plus the independent state of each overlaid calculated phase,
  without any plotting or widget objects.
- `xrd_workbench.models.substrate_compare` owns draggable comparison items,
  overlap geometry, groups, active state, layout, and saved group views without
  storing Tkinter widgets.
- `xrd_workbench.services.correction` creates corrected scans without changing
  their source objects; `services.peak_fitting` and `services.reference_peaks`
  provide the remaining shared correction calculations and data storage.
- `xrd_workbench.services.diffraction` calculates powder reflections,
  structure factors, cell-only systematic absences, and Gaussian powder
  profiles without importing a GUI or localisation layer.
- `xrd_workbench.services.pole_figure` owns reflection enumeration, pole
  projection, coincident-point grouping, orientation matrices, rotations,
  marker scaling, non-overlapping screen-label placement, and powder-intensity
  mapping. Both the current Tk page and a future Qt page can call the same
  service.
- `xrd_workbench.services.substrate_compare` prepares comparison series,
  palettes, Y transformations, axis metadata, and saved or preset limits. The
  Tkinter page only creates widgets and renders this prepared result.
- `xrd_workbench.services.structure_scene` groups coincident chemical
  components into crystallographic sites, creates display-only periodic
  boundary images, estimates bonds, builds polygonal convex coordination
  polyhedra and caches the complete world-space scene. It has no Qt,
  Matplotlib or Tkinter objects.
- `xrd_workbench.io` reads XRDML, two-column text files, and Bruker RAW ranges
  and exports corrected XY/XRDML data. Its reflection module reads scattering
  factors and writes reflection tables with caller-provided headers. It reports
  structured, toolkit-independent errors.
- `xrd_workbench.localization` owns the current language, stable translation
  keys, named interpolation, and separate English, French, and Russian Python
  catalogues. It imports no GUI toolkit. Catalogue keys and interpolation fields
  are validated when the package is imported.
- `xrd_workbench.ui_tk.radiation` contains the reusable Tkinter selector and
  custom one-to-five-line editor backed by that shared model.
- `xrd_workbench.ui_qt` contains the parallel 3.0.0a8.3 interface: the main
  window, three top-level sections, shared project drawer, the native Qt Viewer
  page, the modal Qt comparison window, the calculated-pattern and
  reflection-table pages, the native `QPainter` structure canvas, its Qt
  control page, a Qt radiation selector, and Qt-only widgets. Its package entry
  point is lazy, so importing it does not load PySide6.
- `xrd_workbench.io.cif` now owns CIF tokenisation and basic loop/scalar parsing.
  The shared `CifDocument` therefore no longer imports the historical Tkinter
  pole-figure module, and a Qt process can load CIF files without loading Tk.

The old `xrd_workbench.project_store`, `xrd_workbench.radiation`,
`xrd_workbench.xrd_io`, and `xrd_workbench.cif_xrd` modules are compatibility
facades. `xrd_workbench.theoretical_pole` is also a compatibility and Tk
adapter for the extracted structure and pole engine. `xrd_workbench.i18n`
retains the legacy text-based API and provides Tkinter widget retranslations on
top of `xrd_workbench.localization`. Existing 2.x imports remain valid while
new code imports the explicit layer.

The project model keeps single-selection semantics for calculated structures by
default. Its explicit additive assignment path is used only by the calculated
pole overlay and enforces a maximum of two structural phases.

## 3.0 migration checkpoints

Version 2.9.13 remains the stable, complete Tkinter application.

Version 3.0.0a1 established a separate PySide6 shell and consumed the existing
project, file, radiation and localisation layers without creating a mixed
Tkinter/Qt event loop.

Version 3.0.0a2 replaces the Viewer placeholder with a native Qt page for
experimental one-dimensional scans. It reuses `ViewerState`, cloned `Scan1D`
objects, shared radiation settings and Matplotlib's Qt canvas. Visibility,
colour, order, axes, physical intensity transforms, limits and navigation
remain presentation state and do not mutate project documents.

Version 3.0.0a3 adds CIF and cell-parameter phases to that page. Both overlaid
and separate-axis layouts call the same diffraction and Gaussian-profile
services as Tkinter. One shared `RadiationSettings` instance drives presets and
one-to-five custom lines. Phase layout, height, style, visibility, colour,
order and cached reflection rows remain Qt presentation state. Matplotlib
navigation is connected only to visible axes, and linked 2θ axes synchronize
their X limits without changing project data.

Version 3.0.0a4 removes an unnecessary phase-range control from the Qt
presentation. Independent non-2θ phase axes use the existing automatic
5–120-degree calculation range and remain navigable through the plot. The
Viewer control column now treats its viewport width as authoritative and never
creates an outer horizontal scrollbar.

Version 3.0.0a5 restores the stable Viewer control order and connects its
existing correction workflow to the shared correction, peak-fitting and export
services. Preview transforms stay local until the user explicitly adds or
replaces a project scan; the source is not overwritten without a separate save
action and confirmation. Unmigrated Viewer commands remain visible as disabled
red placeholders at their former positions. The experimental curve-offset
widget is retained in source but omitted from the layout.

Version 3.0.0a6 enables the remaining Viewer list commands and ports the
existing comparison presentation to Qt. The comparison continues to use the
shared `ComparisonWorkspace` and `prepare_comparison_plot` layers; only its
widgets and Matplotlib canvas backend change. One application-modal dialog
receives all Viewer-assigned 2θ scans, preserves substrate presets and can send
a scan back to Viewer or its correction block. Removing or clearing within
Viewer changes workspace assignments without deleting project documents.

Version 3.0.0a7 replaces the Structures placeholder with the established
calculated powder-pattern and reflection-table workflows. Both presentations
consume the same assigned `CifDocument` or `CellPhaseDocument`, shared
`RadiationSettings`, `services.diffraction` calculations and reflection CSV
writer. The shortened hkl family stays in the table cell while the selection
area exposes every equivalent index. The first internal tab remains a marked
placeholder until the three-dimensional structure renderer is transferred.

Version 3.0.0a8.3 replaces that final Structures placeholder with a native Qt
canvas. The established orientation matrices, direct and reciprocal axes,
standard clinographic view, hkl alignment and absolute/relative rotations are
shared calculations rather than reimplemented widget logic. Atom colours and
radii come from the existing style resource and persistent custom overrides.
The scene service groups coincident mixed-occupancy atoms into sites before
constructing coordination polyhedra, uses the full oblique cell matrix for
periodic neighbours and keeps boundary images presentation-only. Convex faces
support arbitrary coordination polyhedra. Geometry is cached at document load;
mouse rotation and right-button panning only project that cache, while hatching
remains visible throughout the interaction. The old scrollable, collapsible control order
is retained without a horizontal scrollbar. Periodic ligand images outside the
unit cell are stored only as presentation data for the selected polyhedra and
can be hidden independently. Hatching uses the original density setting and a
separate GRIP depth coefficient. The
octahedron-specific hatch plan is retained, while larger coordination
polyhedra use face-local opposite edges instead of propagating an arbitrary
family across unrelated faces. Right-button dragging restores the original
screen-space pan.

The pole figures follow as the next separately verified page. Only after
feature parity will the Tkinter launcher be retired.
