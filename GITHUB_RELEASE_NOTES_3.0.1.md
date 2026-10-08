# XRD Combine 3.0.1

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
