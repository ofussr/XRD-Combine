# XRD Combine 3.0.1b

Changes since **3.0.0b6.17**.

This beta introduces background structure preparation and opt-in experimental
powder indexing. It also adopts the shorter **3.0.1b** version number.

## Faster structure loading

- Opening a CIF now starts preparing its atom positions, bonds and coordination
  polyhedra in the background, including when it is opened in **Viewer**.
- A small, nonmodal **Loading structure...** window appears while preparation
  is pending. Its static caption is available in English, French and Russian.
- **Structures** and calculated pole previews share the prepared scene. Switching
  tabs reuses an active calculation or its cached result.
- Multiple CIFs are prepared in sequence. Results for removed or replaced
  documents are discarded, and closing the application cancels queued work.
- Changes to atom colors and pole orientations made during preparation are
  preserved when the structure becomes ready.
- Updated `unit-cell-gui` to retain its improved atom and polyhedron visibility
  performance.

The first display still creates OpenGL buffers in the GUI thread. CIF parsing
and diffraction calculations retain their existing loading paths.

## Experimental powder indexing

Indexing is **hidden by default**. Enable **Toggle indexation** in
**About > Debug**, confirm with **OK**, then open
**Detected peaks... > Index cell...**.

- Choose between independent **Visser** and **Boultif–Louër** methods.
- Review the primary peaks, their center-position errors, wavelength and search
  constraints before calculation. Known Kα2/Kβ companions are excluded.
- Calculations run in the background and can be cancelled. Disabling indexing
  closes its windows; changed, removed or replaced source measurements
  invalidate pending results.
- Inspect candidate cell parameters, indexed-line counts, **M20/F20**, zero
  corrections, individual **hkl** assignments and signed residuals.
- Use **Create cell phase...** to review a candidate in the existing phase
  editor. Space group selection remains a user decision. Accepted hkl return
  to the original experimental peaks; changing the candidate cell prevents
  transfer of stale assignments.

### Numerical behavior

- Explicit positive, finite peak-position errors are respected without being
  enlarged to default Q tolerances. Invalid explicit errors are rejected.
- Perfect-fit infinite Visser M20 ranks correctly. Its first-twenty-line
  evaluation requires the first twenty supplied lines to be indexed.
- Both methods use shared scientific models and residuals defined as
  **observed minus calculated**.
- Shared **Niggli reduction** through spglib replaces partial-spectrum candidate
  deduplication. Reduced metrics and integral hkl basis transformations remain
  available alongside the original candidate representation.

Both methods remain experimental and use bounded searches. They do not promise
exhaustive discovery, compatibility with historical programs, a unique lattice
or space group determination. Broad low-symmetry validation is still incomplete.
See [INDEXING.md](INDEXING.md) for method references and limitations.

## Dependencies and packaging

- Updated `unit-cell-gui` to **>=0.2.2,<0.3**.
- Added **spglib>=2.7,<3**, including its runtime files in the PyInstaller spec.
- Install the updated `requirements.txt` before running or building from source.
- Windows packaging remains **onedir** with the existing static startup splash.

Automated validation is recorded in [VALIDATION.md](VALIDATION.md). A Windows
executable was not built or launched in this environment.
