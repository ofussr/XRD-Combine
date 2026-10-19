# Changelog

## 3.0.2 — 2026-10-19

Cell Phase editing:

- Filter the space-group list by any combination of the seven crystal systems;
  no checked filters shows all settings. Preserve the current group and cell.
- Enable independent cell parameters only and fill dependent greyed-out fields
  automatically, respecting monoclinic unique axes and trigonal H/R settings.
- Retain typed group numbers, symbols and Hall settings. Consolidate the English
  release notes against published 3.0.1; the application version remains 3.0.2.

Further import/scientific-data fixes (version unchanged):

- Preserve coincident atom components and their occupancy/B_iso contributions;
  deduplicate symmetry images per source component and make boundary tolerance
  absolute. Retain CIF quoting, multiline values and token source locations.
- Share and validate symmetry parsing, including finite-order unimodular
  rotations and compatibility with the cell metric.
- Parse verified private file snapshots and reject concurrently changed sources
  before project mutation; retain original paths and source fingerprints.
- Keep valid XRDML ranges with skipped-range warnings. Check sparse numbered XY
  series in memory proportional to the number of files, with bounded diagnostics.
- Validate finite nonnegative intensity thresholds before accepting GUI state,
  normalize oversized session numbers, and atomically export corrected XY/XRDML.
- Restore the missing experimental-indexing workflow documentation.

Data-integrity follow-up:

- Preserve loaded payloads and their original checksums on repeated import;
  changed-source acceptance resets stale analysis during session restoration.
- Detach and retain child curves when their parent document is deleted, keeping
  projects and tabs saveable.
- Reject multi-block CIFs explicitly; validate physical cell parameters through
  the shared geometry path used by CIF and manual Cell Phase input.
- Retain relative hkl signs in powder tables and exports, canonicalizing only
  whole-index Friedel inversions.
- Read comma/semicolon CSV alongside tab/whitespace text, handling decimal commas
  within fields and skipping invalid rows.
- Ignore nonobject atom/language settings with diagnostics; reject invalid local
  background corrections and unordered interpolation axes in models and sessions.
- Reject deeply nested file-handoff JSON without terminating the mailbox worker;
  enforce a shared 4096-path limit before forwarding or claiming a new instance.

Follow-up fixes: experimental pole figures now show the full χ = 0–90° circle
with unmeasured regions blank. The in-page **Open measurements…** commands in
Pole figures and Experimental RSM now accept individual supported angular scans
from RAW, XRDML and XML, using the same geometry checks as Project data.
Complete and numbered RAW maps and the existing XY fallbacks are preserved.

Changes since **3.0.1**.

## Consistent Miller-index controls

- Use the same labelled **h**, **k** and **l** fields in Structures, calculated
  pole-figure centring, second-phase centring, Locate hkl, Viewer phase
  orientation, RSM orientation and targets, and peak hkl assignment.
- Enter signed integer indices with the keyboard or spin buttons. Existing
  validation still rejects a zero direction or an invalid orientation where
  appropriate. Press Enter to apply controls that previously supported it.

## Calculated pole figures

- Move **Align selected pole** beneath the plot, beside **Save figure**.
- Apply alignment to the selected point in either the primary or the second
  phase. The footer identifies the selected phase and hkl.
- With **Rotate phases together** enabled, aligning either layer rotates both
  layers by the same transformation and preserves their relative orientation.
- Fix project and tab saving when the same CIF or Cell Phase is used for both
  calculated layers. Each layer retains its own orientation and appearance;
  the shared source is stored once.

## Viewer dataset table

- Resize every **Loaded datasets** column by dragging its header boundary.
- Retain accepted column widths in project and Viewer-tab files. Horizontal
  scrolling makes wider columns accessible in a narrow sidebar.

## Reset the active workspace

- Add **File > Reset tab** for Viewer, Structures, Pole figures and RSM.
- Restore the active workspace's default calculation and display settings,
  including orientation and navigation, while retaining its assigned sources.
- Preserve project documents, accepted peaks, backgrounds, spectral
  classifications, hkl assignments and corrections already applied to source
  objects. Other workspaces keep their state. The active Experimental/Calculated
  subtab remains selected.
- Reset a Viewer with a running phase calculation without waiting for the
  cancelled worker to finish on the interface thread.

## Measured angular scans

Individual scans can now be assigned through **Project data** when their
recorded metadata provide the required geometry:

| Destination | Scan axis | Required fixed positions |
| --- | --- | --- |
| Pole figures | φ | 2θ, ω/θ and χ |
| Pole figures | χ | 2θ, ω/θ and φ |
| Experimental RSM | ω or θ | 2θ |

- Recognize fixed positions from supported XRDML and Bruker RAW scan metadata.
  A recorded zero is valid; missing angles are not replaced with zeros.
- Draw φ/χ scans at their acquired pole coordinates with either plot renderer.
  Draw individual rocking curves as angular or Qx/Qz trajectories in RSM.
- Keep the original sample intensities and angular gaps. These paths do not
  fill the unmeasured area or interpolate a complete pole figure or RSM grid.
- Retain intensity scaling, image export and applicable plot navigation. Display
  CIF reflection markers only at acquired rocking-curve positions.
- Keep full RAW map import and the existing manually defined XY-series workflow.
  A complete RAW map stays assigned when its child ranges are imported.
- Reject incomplete or unsupported scan geometry for these workspaces, with a
  reason in the assignment tooltip. The source remains available in Project data
  and Viewer. Rocking trajectories and regular RSM grids use separate assignment
  modes in the existing Experimental subtab.
- Reconstruct angular-path sources and accepted display settings when loading
  project or tab files. Existing 3.0.1 session files remain readable.


## 3.0.1 — 2026-10-08

Changes since **3.0.1b**, including the changes developed in **3.0.2b–3.0.7b**.
The application version is now **3.0.1**, without a beta suffix.

## Opening source files

- Drag one or several measurement or CIF files into the main window, including
  its plots, dataset tables and input controls. A drop hint identifies an accepted
  batch. Supported extensions are `.xrdml`, `.xml`, `.raw`, `.xy`, `.txt`, `.dat`,
  `.csv` and `.cif`; uppercase extensions and paths with spaces or Unicode work.
- Add imported objects to **Project data** and assign compatible objects to the
  active workspace, using the same import route as the existing file commands.
- Reuse the running window when opening files with **Open with…** in Windows
  Explorer or launching another copy. Restore a minimized window and request
  foreground activation; starting again without files brings the window forward.
- Queue requests while the first window is starting or a modal dialog is open.
  Parallel launches elect one owner; local acknowledgements and request identities
  prevent duplicate delivery on retries. A dead owner's lock is recovered.
- Keep saved `.xrdproject` and `.xrdtab` files in their explicit File-menu commands.
  Drag-and-drop accepts source files only; directories and remote URLs are rejected.
  A batch containing an unsupported item is rejected as a whole.

## Oriented phase scans and cursor coordinates

- Display oriented CIF and Cell Phase reflection positions in φ, χ and ω/θ rocking
  scans, coupled 2θ–ω scans and detector-only 2θ scans at fixed ω.
- **Phases > Phase scan > Auto** follows angular measurements. Standard 2θ scans
  retain the powder calculation by default; oriented modes can be selected manually.
- Import fixed geometry into editable **From … To** ranges, initially **±0.1°**
  around available measurement angles. Missing bounds require manual input.
  Each reflection must satisfy all fixed-angle windows together.
- Set a surface-normal hkl, in-plane reference and linked **φ zero offset** slider
  and numeric field per phase. φ shifts use cached reflections immediately.
- Retain manual geometry between measurements; **Use measurement angles** restores
  imported defaults. Calculate oriented reflections in the background and discard
  stale results after geometry or source changes.
- Keep phases loaded while hiding their plots for X, Y, Z and unsupported coordinates.
- Show current cursor X/Y coordinates beside the visible-measurement/phase counters,
  following the panel's units and nonlinear scales; clear the readout on leaving.

Oriented scans predict **positions**, not measured intensity, mosaic spread or
instrumental broadening. Equal-height sticks and profile widths are illustrative.
A fixed-angle window may admit several positions; one representative position near
the nominal geometry is shown within the finite maximum-hkl search range.

## Calculated pole figures

- Add **Locate hkl**: enter h, k and l for either phase and press OK or Enter to
  outline its reciprocal-lattice direction for **two seconds**. Locate directions
  outside the displayed interval without changing filters, orientation or selection.
- Select reflection bounds in **d, Å** or **2θ, °**, converting with the pole-figure
  wavelength and retaining angular limits when radiation changes.
- Support the primary and overlaid phase with either PyQtGraph or Matplotlib.
- Preserve the surviving phase's orientation, selection and styling when an overlay
  becomes primary after removal of the original primary phase.

## Accepted analysis and workspace state

- Move accepted peaks, backgrounds, spectral classifications and hkl assignments
  into GUI-independent project models shared by plots and peak tables. Preserve
  stable peak IDs, companion links and fill choices through refinement/redrawing.
- Commit complete fit groups together; a failed group leaves accepted results intact.
  Batch compound notifications and expose read-only result arrays.
- Preserve analysis when hiding a Viewer curve, closing its peak table or recreating
  the page. Removing a measurement from Viewer clears its analysis on every axis,
  even when it stays in Project data. Replacing/deleting data also clears analysis.
  Deleting/editing a phase invalidates only hkl assignments linked to that phase.
- Keep Viewer dataset order, selected object, source axes, visibility, colours,
  correction previews, scales, plot navigation, phase layout/profile settings and
  per-measurement/per-axis geometry in the project rather than the widgets.
- Keep calculated/experimental pole phase order, rotations, centring, projection,
  labels, marker styling, reflection bounds, manual angle series, intensity settings,
  navigation and active subtab in the project.
- Keep Structures atom/polyhedron visibility, component/site colours, opacity,
  display mode, hatching, size, orientation and camera pan/zoom in the project.
  Retain calculated-pattern/reflection-table parameters, plot navigation, table
  sorting/widths, selected reflection and active subtab. Compact pole-structure
  previews keep independent display/camera settings while following pole orientation.
- Keep experimental/calculated RSM geometry, XY angle series, coordinate mode,
  intensity scale, colour map, extent, overlay phase/orientation, calculation target,
  ranges, labels, markers, navigation and active subtab in the project. Adding XY
  files retains the two entered angles and recalculates the derived angle.
- Restore accepted settings when pages are recreated or prepared structures arrive.
  Clear source-dependent geometry, selections and caches after removal/replacement;
  keep unrelated objects and general options. Invalid/unfinished input remains a
  local draft and does not overwrite accepted configurations.

## Save and reopen workflows

- Add **Save project** (Ctrl+S), **Save project as…** (Ctrl+Shift+S) and
  **Open project…** (Ctrl+O) for `.xrdproject` files, plus **Save current tab…**
  and **Load tab…** for `.xrdtab` files covering Viewer, Structures, Pole figures
  and RSM.
- Restore documents, assignments, accepted analysis, workspace presentation,
  geometry, cameras, navigation and subtabs. Replay ordered correction operations
  from original source axes and preserve derived-curve relationships.
- Restore complete-project radiation, language, theme, renderer, atom colours,
  active workspace, window size and project-panel layout. Tab loading retains
  current application appearance; its radiation profile remains shared globally.
- Load only the workspace named in a tab file, preserving other workspaces and
  their analysis. Reuse matching verified sources/correction recipes and remap
  references when importing a different saved version.
- Save source paths and SHA-256 checksums instead of embedding original scan or
  CIF contents. Store manual Cell Phases as cell parameters and a space-group setting.
  Support multi-range RAW/XRDML and numbered XY experimental-pole exports.
- Relocate, skip or cancel missing sources. Explicitly accepting changed sources
  clears dependent stale analysis, orientation and navigation while replaying saved
  corrections. Validate the versioned JSON and relationships before replacing the
  live project; invalid or cancelled loads leave it open.
- Write saves atomically and prevent overwriting original source files. Preserve
  PyQtGraph line width, custom labels, grids, legend and tick settings in Viewer
  and calculated patterns. Keep save shortcuts unique after menu recreation.

Input files must remain available or be relinked. Comparison retains its existing
bound-preset files and stays outside workflow persistence. Saves exclude calculated
caches, unfinished inputs, open dialogs, running jobs, direct Matplotlib toolbar
edits and old undo/replacement snapshots. Autosave and automatic project reopening
are not included. Experimental indexing remains opt-in and experimental.

## Maintenance and packaging

- Consolidate shared scientific/native Qt fixtures and remove redundant or obsolete
  source-string tests; add model lifecycle, restoration, session and file-handoff checks.
- Keep core models/services independent of Qt and retain lazy `--version` output.
- Provide the new drop/source-file messages in English, French and Russian.
- Retain Windows **onedir** PyInstaller packaging and the existing splash/About assets.

See [VALIDATION.md](VALIDATION.md) for the executed checks and platform limits.
