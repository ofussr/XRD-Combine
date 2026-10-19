# XRD Combine 3.0.2

Changes since **3.0.1**.

This release improves manual cell entry, unifies Miller-index controls, adds
workspace reset, and extends measured angular scans to Pole figures and RSM.
It also fixes scientific-data handling, imports and project saving.

## Cell Phase editor

- Filter the space-group list with seven crystal-system checkboxes. Select one
  or several systems to show their groups; clear all checkboxes to show all
  530 settings again. Filtering preserves the selected group and cell values.
- Edit only the independent cell parameters for the selected space-group
  setting. Dependent fields are greyed out and filled automatically: a cubic
  cell needs **a**; a tetragonal cell needs **a** and **c**; an orthorhombic cell
  needs **a**, **b** and **c**. Triclinic cells retain all six editable parameters.
- Respect monoclinic unique-axis choices and trigonal **H/R** settings. In the
  rhombohedral setting, lengths and angles are linked separately; in hexagonal
  axes, **b = a**, **α = β = 90°** and **γ = 120°**.
- Keep manual input by group number, symbol or Hall setting. Resolve typed
  entries when committed and validate the complete cell before applying it.

## Miller indices and Viewer controls

- Use consistent labelled **h**, **k** and **l** fields in Structures, pole-figure
  centring and Locate hkl, Viewer phase orientation, RSM orientation and targets,
  and peak assignments. Signed keyboard input, spin buttons and applicable
  Enter actions remain available.
- Resize every **Loaded datasets** column by dragging its header boundary.
  Save accepted widths in project and Viewer-tab files; use horizontal scrolling
  when the table is wider than the sidebar.

## Pole figures and measured angular scans

- Move **Align selected pole** below the calculated plot, beside **Save figure**.
  Align points from either phase; the footer identifies the selected phase and
  hkl. **Rotate phases together** applies the same rotation to both layers.
- Save projects and tabs with the same CIF or Cell Phase in both calculated
  layers. Store the shared source once while retaining each layer's orientation
  and appearance.
- Show experimental pole figures over the full **χ = 0–90°** hemisphere in both
  renderers, even when the measurement covers a smaller tilt interval.
- Open individual angular scans directly through **Open measurements…** in
  Pole figures and Experimental RSM, or assign them through **Project data**.
  Supported RAW, XRDML and XML sources must provide the recorded geometry below.

| Destination | Moving axis | Required fixed positions |
| --- | --- | --- |
| Pole figures | φ | 2θ, ω/θ and χ |
| Pole figures | χ | 2θ, ω/θ and φ |
| Experimental RSM | ω or θ | 2θ |

- Retain acquired coordinates, intensities and angular gaps. Draw φ/χ paths in
  Pole figures and rocking trajectories in angular or Qx/Qz RSM coordinates.
  Display applicable CIF markers at acquired rocking positions. Unmeasured
  regions remain blank; individual scans do not become interpolated map grids.
- Treat recorded zero angles as valid and report missing geometry explicitly.
  Unsupported sources remain available in Project data and Viewer without
  replacing the current Pole figures or RSM assignments.
- Preserve complete RAW maps, numbered RAW map assembly and existing manual
  XY-series workflows, including the numbered-XY fallback for unreadable pole
  RAW files. Keep a full map assigned when its child ranges are imported.
- Restore angular-path sources and their accepted display settings from project
  and tab files.

## Workspace reset and project reliability

- Add **File > Reset tab** for Viewer, Structures, Pole figures and RSM. Restore
  the active workspace's default calculation/display settings, orientation and
  navigation while retaining assigned sources and the active subtab.
- Retain accepted peaks, backgrounds, spectral classifications, hkl assignments
  and corrections already applied to source objects. Other workspaces keep
  their state. Resetting Viewer does not wait on its running phase worker.
- Keep accepted data and their original checksum together on repeated import.
  Detect changed sources during session loading; explicit acceptance clears
  temporary results tied to the old data.
- Parse verified private snapshots for source imports. Reject a concurrently
  changed source before adding its objects, and retain the original file paths.
- Keep child curves as independent, saveable documents after deleting their
  parent measurement or RAW map.
- Reject invalid intensity thresholds before replacing accepted calculation
  settings. Negative, NaN or infinite drafts cannot make a project unsaveable.
- Report oversized session numbers as controlled format errors. Validate local
  background radii, excluded widths and correction-axis order consistently.
- Write corrected XY and XRDML exports atomically, preserving an existing output
  if writing the replacement fails.

## Scientific data and import fixes

- Preserve independent coincident atom components and their individual
  occupancies and isotropic displacement parameters. Deduplicate only symmetry
  images of the same source component, using explicit coordinate tolerances.
- Retain quoting, multiline text and source locations in CIF tokens. Strings
  resembling CIF commands remain values when quoted. Reject multiple `data_`
  blocks explicitly instead of mixing their structures.
- Share symmetry parsing between structural geometry and powder calculations.
  Reject singular, scaling, noncrystallographic and cell-incompatible operations;
  validate finite positive lengths, physical angles and nondegenerate metrics.
- Preserve relative hkl signs in powder tables and exports. Use a common
  representative for Friedel pairs while retaining distinct signed reflections
  and their correct spacings in oblique cells.
- Read comma-, semicolon-, tab- and whitespace-separated text measurements,
  including quoted CSV fields and decimal commas in supported fields.
- Keep valid ranges from partially readable XRDML files and report skipped
  range numbers. Check numbered XY sequences without allocating space for absent
  indices, and keep missing-index diagnostics bounded.
- Fall back to default atom colours and language settings when their JSON root
  has an invalid type, with a diagnostic in the log.
- Keep local file handoff running after deeply nested or malformed JSON. Enforce
  the same 4096-path limit on both sides and report excessive requests immediately.
- Restore the missing experimental-indexing workflow documentation.

## Compatibility

Existing **3.0.1** project and tab files remain readable. Original measurement
and CIF files remain external and must be available or relinked. Comparison
continues to use its existing display-bound presets. Experimental indexing
retains its opt-in status. Windows packaging remains **onedir**.

See [VALIDATION.md](VALIDATION.md) for automated checks and platform limitations.
