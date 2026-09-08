# XRD Combine architecture

The 2.9.x series prepares the scientific code for a PySide6 interface while
keeping the released Tkinter application operational.

## Dependency direction

```text
ui_tk / future ui_qt
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

## Remaining 2.9.x work

The shared scientific, project, viewer, correction, and localisation layers
are now separated from the Tkinter application. Version 2.9.13 completes the
planned Tkinter stabilisation pass: compact and leader-based pole-label layout
remain GUI-independent, Bragg spacing for selected experimental points is a
shared diffraction service, and catalogued static Tk text uses stable language
keys. The next planned architectural step is the first PySide6 shell in the
3.0.x series; it should consume these models and services without moving
scientific calculations back into widgets.
