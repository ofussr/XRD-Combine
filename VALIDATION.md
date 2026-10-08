# XRD Combine Validation

## 3.0.1

The complete Linux Qt offscreen suite passed **462 tests and 19 subtests**, with
one existing OpenGL export test skipped (263.97 seconds; 356 dependency warnings).
The initial affected import/session/shell/architecture/localization run passed
**61 tests**. The final native file-opening, handoff, session, shell, architecture
and localization run passed **45 tests** after the drop-hint refresh adjustment.
The final focused multi-process handoff run passed **6 tests**. `compileall`,
`git diff --check` and lazy command-line `--version` passed; the version is **3.0.1**.

Native drag/drop checks exercise the main window, project-tree viewport, input
controls and plot widgets with multiple actual scans, uppercase extensions,
Unicode/spaced paths and repeated paths. Sessions, unsupported extensions,
directories and remote URLs are rejected. A real CIF drop assigns the Structures
workspace and uses background preparation. Queued requests wait for modal dialogs
and retain their destination workspace; unsupported command-line/session paths
cannot enter source readers. Visibility/minimized-window restoration is checked.

Local handoff tests use actual subprocesses, private atomic mailboxes and Qt
lifetime locks. They cover acknowledgement while the main thread is blocked,
five concurrent clients without lost requests, simultaneous cold-launch owner
election, invalid messages, duplicate-request acknowledgement, and recovery after
an owner exits without cleanup. A secondary call through the real app bootstrap
is checked with MainWindow imports explicitly forbidden. The private mailbox
requires no network sockets or additional dependency.

A further real-application smoke check starts with a Unicode/spaced source path,
minimizes the main window and launches the unchanged Python launcher with a second
source. Both sources appear in one window, the window is restored and the accepted
drop hint remains visible after an event-loop update. The resulting native window
preview was inspected. The source-only intake shares existing ProjectFileService
routing; project/tab save commands retain their explicit JSON-session route.

README, the consolidated CHANGELOG and English 3.0.1 release notes cover all
changes since **3.0.1b**, including the intervening beta iterations. The launcher,
PyInstaller spec, startup splash, About image and all Comparison modules are
byte-identical to delivered **3.0.7b**. Windows packaging remains **onedir**.
This is a source release: no Windows executable was built or launched here.
Windows Explorer integration, foreground permissions and real mouse drag/drop
still need a check in the packaged Windows application before publication.
Experimental indexing retains its existing opt-in status and scientific limits.

The source ZIP was unpacked and checked independently: **37 file-intake, local
handoff and session reconstruction tests** passed (15.14 seconds), and its CLI
reported **3.0.1**. Every archived member was compared byte-for-byte with the
source tree; archive CRC validation passed.

## 3.0.7b

The Linux Qt offscreen suite passed **444 tests and 19 subtests**, with one
existing OpenGL export test skipped (323.38 seconds; 356 dependency warnings).
After the final display-state and format checks, **103 affected model, session,
Viewer, Structures, native Qt, architecture and localization tests**, including
nine subtests, passed again; the same OpenGL test was skipped. The final native
session/shortcut checks passed **4 tests**, and final format/reconstruction
checks passed **19 tests**. `compileall`, `git diff --check` and the lazy CLI
`--version` check passed; the version is **3.0.7b**.

Service round trips use actual CIF, XY, RAW and XRDML files, manual cells,
derived curves and multiple correction operations, including changing the
processed source axis. Checks verify document identities, source/range selectors,
parent links, working transformations, hidden curves, accepted phase geometry,
background/local adjustments, peak classifications and hkl relationships,
immutable result arrays and retained peak-number allocation. Original source
arrays and CIF text are absent from session manifests. Model checks also cover
relative-path relocation, source relinking/skipping/cancellation, checksum
changes, stale-result cleanup, RAW map sources, numbered XY pole exports with
renamed paths, unbounded and exact-angle pole ranges, tab isolation and reuse
of matching source objects. Invalid schema versions, models, viewports, parent
cycles, assignments, application settings and correction recipes are rejected;
failed serialization preserves an existing save.

Native Qt tests save and reopen complete projects with both Matplotlib and
PyQtGraph. They restore structure cameras, calculated pole orientation and
selection, RSM navigation, Viewer analysis/visibility/geometry, theme, renderer,
active section and project-panel visibility. PyQtGraph custom labels, line width,
grid, legend and tick settings are owned by the display model and restore in
Viewer and calculated patterns. Ctrl+S triggers once after menus are rebuilt.
A single-tab import retains other sections' analysis and an existing Comparison
window. Failed and cancelled loads preserve the live project and its widgets;
old project listeners and workers are detached on successful replacement.
A native File-menu preview and saved-project restoration were also inspected.

Test definitions increased from **346 to 358**. Parameterized session cases
share actual-file fixtures and the existing scientific RAW/CIF builders.
The launcher, PyInstaller spec, startup splash, About image and all Comparison
modules are byte-identical to the delivered **3.0.6b** archive. This is a source
archive; no Windows executable was built or launched. Source files remain
external and must be available or relinked. Direct Matplotlib toolbar edits,
unfinished inputs, transient windows/jobs and old replacement/undo snapshots
are not persisted. Autosave and automatic project reopening are not included.

## 3.0.6b

The Linux Qt offscreen suite passed **421 tests and 19 subtests**, with one
existing OpenGL export test skipped. `compileall`, `git diff --check` and the
command-line `--version` check passed; the displayed version is **3.0.6b**.
After the final overlay-phase lifecycle adjustment, all **16 affected RSM,
workspace-model and native restoration tests** passed again, including replacing
an overlay phase without reviving its invalidated geometry from the controls.

New model checks cover structure replacement, unassignment and removal before
notifications, independent workspace settings, source-local cleanup, effective
RSM source changes with RAW priority, overlay-phase removal, and deep-copied
numeric state. Native checks recreate the structure workspace with both
calculated-pattern renderers and verify accepted parameters, sticks/profile
selection, plot navigation and reset, reflection sorting, column widths, selected
reflection and active subtab. Structure and compact pole previews restore
independent atom visibility, site colours, polyhedron visibility/opacity,
engraving, hatching, atom size, orientation and normalized camera pan/zoom.
A delayed background result restores current project settings in both adapters.

Experimental and calculated RSM checks cover XY assembly and derived-angle
editing after recreation, coordinate modes, intensity scale, colour map,
reflection-overlay phase and orientation, calculated target and ranges, labels,
target marker, navigation, active subtab and rejected invalid numeric drafts.
Scientific geometry checks still cover missing RAW ranges, Real/Full bounds,
map coordinates and reciprocal-space transformations.

The shared mixed-occupancy CIF fixture is reused by the existing completion
checks and the new restoration checks. Test definitions changed from **339 to
346**; existing native checks now accept the camera-state adapter as a subclass
of unit-cell-gui's canvas and compare numeric span values independently of text
formatting.

The launcher, PyInstaller spec, startup splash, About image, and all Comparison
modules are byte-identical to the delivered 3.0.5b archive. This is a source
archive; no Windows executable was built or launched. Project/tab save/load and
applied correction-operation history remain future work. Comparison is excluded
from planned workflow/tab persistence and keeps its existing display-bound
preset files.

## 3.0.5b

The Linux Qt offscreen suite passed **411 tests and 19 subtests**, with one
existing OpenGL export test skipped. After the final removal-label cleanup and
Matplotlib viewport repaint adjustment, all **38 affected pole-controller and
native pole tests** passed again. `compileall`, `git diff --check` and the
command-line `--version` check passed; the displayed version is **3.0.5b**.

Model checks cover assignment order, overlay promotion before notifications,
phase unassignment/removal, replacement of one cell without altering the other
layer, RAW source cleanup, and independent copied orientation arrays. Native
checks recreate calculated and experimental pole pages with both Matplotlib
and PyQtGraph, exercising rotations, joint rotation, colours, opacity, sizes,
labels, projection, d/2theta ranges, manual tilt series, intensity limits,
measured samples, plot navigation and active subtab. Invalid drafts retain
accepted geometry. A changed radiation rebuilds cached reflections while
preserving orientation. Removing sources clears stale fields and pole details.
Circular plots keep equal aspect; a changed widget shape may expand one axis
when restoring the saved view.

Test maintenance removed ten duplicate or obsolete methods, including the
repeated Gaussian-fit scenario and source-string migration checks. Common Viewer
settings now run once per renderer; viewport checks still cover all eight
renderer/scale combinations. CLI version validation blocks GUI imports and
compares against APP_VERSION. Shared Qt runtime/setup and scientific pole
fixtures no longer live in collected Qt/pole test modules. Test definitions
changed from **337 to 339**, including the new pole-state coverage. The former
project-tree source check is replaced by a live Qt test that verifies the tree
item remains valid during assignment and refresh occurs after the signal.

A reconstructed two-phase calculated pole page was also visually inspected in
both renderers. The launcher, PyInstaller spec, startup splash and About image
are byte-identical to the delivered 3.0.4b archive. This is a source archive;
no Windows executable was built or launched. Project/tab save/load, RSM,
Structures and Comparison state extraction, the compact structure preview's
atom/camera settings, and applied correction-operation history remain future
work.


## 3.0.4b

The Linux Qt offscreen suite excluding the About-window module passed **399
tests and 25 subtests**, with one existing OpenGL export test skipped. The
five About-window tests passed separately after recovering a truncated local
logo PNG from the previous delivered archive. Every byte of the truncated PNG
matched the complete file's prefix; the recovered image is identical to that
archive. The combined result is **404 tests and 25 subtests passed, one skipped**.
Existing spglib deprecation warnings do not fail the suite. The displayed
version and command-line `--version` output are **3.0.4b**.

New model checks cover assignment and angle autofill before creating any GUI;
independent working scan arrays; copying state without widgets; active axes,
ordering and visibility; replacement of all axis-specific angle contexts;
removal of object-local settings; and preservation of the source document and
assignments to other workspaces after Viewer unassignment.

Native adapter checks recreate Viewer in PyQtGraph and Matplotlib under linear,
logarithmic, square-root and squared intensity scales. They verify restored
selection, active source axes, order, colours, visibility, correction previews,
profile width, background spacing, result mode, peak-sum visibility and view
limits. Additional checks cover d-spacing and wavelength, mouse-range callback
updates, independent phase-panel X limits, a geometry reference different from
the tree selection, per-phase orientation and linked phi offset controls.
Invalid ranges, nonfinite angles and out-of-domain 2theta inputs block live
calculation while preserving accepted model values. Replacing an inactive
measurement clears its old invalid input draft. Existing asynchronous phase
calculation, cached phi shifting, peak/background lifecycle, fitting, indexing
and other workspace checks continue to pass.

A synthetic phi scan was also reopened from project-owned state and visually
checked in both renderers. Fixed-angle bounds, shifted phase markers and the
measurement zoom matched after recreation.

The startup splash, entry-point launcher and PyInstaller spec remain byte-for-byte
identical to the source checkout, retaining onedir packaging. No Windows
executable was built or launched. State restoration is in-memory only;
project/tab save/load, other workspace state extraction and correction-operation
history remain future work.

## 3.0.3b

The original full Linux Qt offscreen refactor suite passed **373 tests and
25 subtests**, with one existing OpenGL export test skipped. After correcting
Viewer-unassignment cleanup, all **83 targeted tests and 8 subtests** passed
across analysis lifecycle, peak search, Qt shell, indexing UI and architecture.
The displayed version and command-line `--version` output are **3.0.3b**.
Existing spglib deprecation warnings do not fail the suite.

New core checks cover toolkit-free imports with GUI imports actively blocked;
stable IDs after redraw, deletion and import; measurement/axis isolation;
companion promotion after parent removal; hkl invalidation after phase edits
and removal; immutable result arrays; batched notifications; and replacement
or removal without a Viewer. A two-group synthetic fit verifies that failure
of the later group commits neither earlier peak updates nor background edits.
Successful fits retain IDs, hkl assignments and fill choices.

Native Viewer checks cover discarding analysis when a measurement leaves
Viewer and starting empty on reassignment, retaining results when merely hiding
its graph or recreating the widget, updating tables and shading from external
model edits, replacing source data while a preview is open, and keeping active
region selection intact during analysis-only notifications. Cleanup checks
include Viewer's Remove/Clear actions, all measurement axes, hkl and companion
metadata, unchanged project data, unaffected measurements, and preserved
assignments to other workspaces. Unassigning from another workspace preserves
Viewer analysis. Existing peak search, manual drawing, background, companion and
indexing integration checks also pass. A two-peak smoke run of the initial
refactor was visually inspected in PyQtGraph, Matplotlib and the peak table.

The startup splash, entry-point launcher and PyInstaller spec remain byte-for-byte
identical to the source checkout, retaining onedir packaging. No Windows
executable was built or launched. This release extracts accepted measurement
analysis only; other persistent tab settings and project/tab save/load remain
future work.

## 3.0.2b

The full Linux Qt offscreen suite passed **352 tests and 25 subtests**, with
one existing OpenGL export test skipped. After the final guard for a d interval
entirely below the Bragg limit, all **34 targeted range, locator and localization
tests** passed again. The displayed version and command-line `--version`
output are **3.0.2b**.

Oriented-scan coverage includes joint fixed-angle windows, missing and reversed
bounds, metadata autofill, phi periodicity, rocking/coupled/detector geometry,
asynchronous stale-result rejection, and phi shifts using cached reflections.
Native Viewer checks cover both renderers, cursor coordinates, d-spacing and
nonlinear intensity ticks. An atom-bearing CIF smoke check produced the four
expected phi positions with positive structure factors.

Pole-figure checks cover both renderers and projections; out-of-range hkl
outlines in the current phase orientation; restart and expiry of the two-second
timer; independent overlay orientation and removal; invalid hkl; d/2theta
round trips; wavelength changes; zero-angle bounds; and non-destructive range
validation. The existing shared-radiation check now accounts for both pairs
of range fields. EN/FR/RU controls and transient outlines were visually inspected.

The startup splash, entry-point launcher and PyInstaller spec are byte-for-byte
unchanged from the source checkout, retaining onedir packaging. No Windows
executable was built or launched. The existing spglib deprecation warnings do
not fail the suite.

## 3.0.1b

The full Linux Qt offscreen suite passed **274 tests and 25 subtests**, with
one existing OpenGL export test skipped. A separate Xvfb session with a real
software OpenGL context passed **18 selected Qt integration tests**, including
all seven background structure preparation tests and all six indexing UI tests.

The displayed version and command-line `--version` output are **3.0.1b**.
All Python application modules passed syntax checks. The startup splash image,
entry-point launcher and PyInstaller spec are byte-for-byte unchanged from
3.0.0b6.19, retaining onedir packaging and the static image-only splash.

The numerical indexing coverage and limitations recorded below remain
applicable. spglib 2.7.0 emits upstream deprecation warnings which do not fail
the suite. No Windows executable was built or launched in this environment.

## 3.0.0b6.19

The full Linux Qt offscreen suite passed **269 tests and 25 subtests**, with
one existing OpenGL export test skipped. All 36 new numerical indexing tests
and six new Qt indexing tests passed. The six Qt tests also passed in a
separate Xvfb session with a real software OpenGL context.

Numerical coverage includes authoritative explicit centre errors for 2theta,
d and Q in both methods; invalid input; toolkit-free imports; stable IDs and
disabled input mapping; six independent analytic system fixtures; shared
Niggli reduction under integral basis changes and invariant Q/hkl; perfect
cubic Visser discovery with infinite M20; first-twenty-line M20 semantics;
Boultif–Louër discovery across six systems; noise, missing/spurious lines and
zero correction. Visser's six-system tests cover final line review, not full
discovery in all six systems. Broad low-symmetry validation remains incomplete.

The independent NIST SRM 640f check indexes all 11 published informational
positions at 0.001-degree explicit error with the specified cubic bounds and
hkl limit. The resulting silicon length is 5.4311426806 Å versus the
certificate value 5.431144 Å. These are published calculated positions, not a
new measured-profile validation. An exploratory triclinic case returned no
candidate with bounded settings; this limitation is recorded in INDEXING.md.

Qt coverage verifies initially hidden controls, Debug Cancel/OK semantics,
persisted opt-in and immediate visibility changes, source-coordinate input,
companion exclusion, background calculation and event-loop responsiveness,
cancellation on disable/close, stale-source invalidation, Cell Phase creation,
stable-ID hkl transfer and suppression of transfer after cell edits. Screenshots
of the Debug, peak-selection and results windows were inspected.

spglib 2.7.0 emits upstream deprecation warnings about its legacy error handling;
these do not fail the tests. The PyInstaller spec was executed with recording
build targets: onedir EXE/COLLECT separation, static splash configuration and
collection of unit_cell_gui, pyqtgraph and spglib passed. The launcher and
startup image are unchanged. No Windows executable was built or launched.

## 3.0.0b6.18

The full Linux Qt offscreen suite passed 227 tests and 25 subtests, with one
OpenGL export test skipped. Seven new regressions cover background preparation
after Viewer imports, GUI event-loop responsiveness, shared pending jobs and
cached results, removed/replaced payloads, pending view changes and pole
orientation, geometry errors, closing during calculation, and changed colors.

The seven background-loading regressions also passed twice in separate Xvfb
sessions with a real software OpenGL context. The Viewer-to-Structures test
checks a nonempty framebuffer and successful OpenGL initialization after an
early tab switch. The final implementation uses a Python worker pool and
GUI-thread result polling; workers never own or access Qt objects.

A separate legacy native pole-export check encountered a figure-size assertion
while Matplotlib processed a deferred resize under Xvfb. This export check is
not counted as passed and is outside the structure-loading changes.

The PyInstaller spec and entry-point launcher are unchanged from b6.17; onedir
packaging and the static startup splash are retained. The current repository
README, citation metadata, and splash image are included. No Windows executable
was built or launched, so packaged startup requires a native Windows check.

## 3.0.0b6.17

The corrected spec was compiled and executed with recording build targets.
Checks passed for the onedir EXE/COLLECT split, `exclude_binaries=True`,
`contents_directory="."`, console suppression, collection of `unit_cell_gui`
and `pyqtgraph`, all eight source data files, and their command-line paths.
The static splash remains attached to EXE while its runtime binaries go to
COLLECT; `text_pos=None` and `max_img_size=(650, 434)` were checked.

These are configuration checks, not a real PyInstaller build. No Windows EXE
was built or launched. The application test suite was not rerun: analysis and
GUI code are unchanged apart from the displayed version number. The b6.16
test results below remain historical results.

## 3.0.0b6.16

The project contains a PyInstaller one-file spec with the static PNG splash,
650-pixel width limit, and an explicit close after the main Qt window is
shown. The offscreen Qt suite passed 221 tests, with one OpenGL test skipped.
An offscreen startup check injected a mock splash module and confirmed that
the window is shown and the splash is closed once. A Windows executable could
not be built in the Linux workspace; the splash timing and rendering need a
native Windows build check.

## 3.0.0b6.15

The offscreen Qt suite passed 221 tests, with one OpenGL test skipped. New
regressions cover successive nearby peak regions, a deterministic drawn Voigt
profile (its stated height and width are preserved until Do Fit), and the
Do Fit button in the Viewer controls. Existing tests cover preview cancellation,
shared background changes, and manual peak refinement.

On the supplied LB067 XRDML scan, selecting 31–34° found six peaks in 1.56 s
and accepted them in 3.58 s. Selecting 28.5–30° immediately afterward found
three more in 0.02 s and accepted them in 0.07 s, without opening a warning
dialog. These are local offscreen measurements. The second background pass
reduced the model overestimate at 30° from about 100 counts in b6.14 to near
zero in this run; near 30.5° the remaining excess is about 36 counts. The
profile around 31° still has a visible residual, so this is not an exact fit.

## 3.0.0b6.14

The offscreen Qt suite passed 219 tests with one OpenGL test skipped. A
25,001-point two-region regression verifies that accepting the second distant
peak only refits its own group, keeps the first peak object and its local
background unchanged, and accepts both peaks. The preview cancel/confirm test
now covers the default PyQtGraph path.

The supplied LB067 XRDML scan (25,767 points) was opened in Viewer. Detection
and acceptance of the 31.3–33.5° region found six peaks in 0.33 s and accepted
them in 2.34 s; 68.4–69.4° found two more in 0.03 s and accepted them in
0.13 s. These are local offscreen timings, not guaranteed UI latency.

Matplotlib remains in the other plotting screens and their export paths. The
dependency must remain until those screens are migrated and their behavior
verified.

## 3.0.0b6.13

The final offscreen Qt suite passed 218 tests, with one OpenGL test skipped.
From the extracted archive, 43 targeted peak, localization, and shell tests
passed, and the main application remained open during a six-second startup
smoke check.

Regression checks cover the local background correction and its unchanged
values beyond the peak group, joint peak refinement, and cancelling/accepting
the main window close event. The supplied LB067 scan was inspected numerically
and visually over its 29–35° peak group. The corrected model better follows
the low-intensity left flank; the isolated peak near 29.5° and asymmetry near
33° remain visible residuals unless represented by appropriate components.

## 3.0.0b6.12

With the offscreen Qt runtime, `python -m unittest discover -s tests -q`
completed 216 tests successfully (one OpenGL test skipped). New regression
tests cover joint refinement of overlapping accepted Voigt peaks, preservation
of an unaccepted distant reflection, the Do Fit button and table metadata,
and drawing a new peak on the residual of an existing fit.

On the supplied LB067 XRDML scan (25,767 points), a five-component refinement
completed in approximately 0.5 seconds. With manually seeded components and
the existing automatic background, the root-mean-square residual in the
32–34° interval fell from about 389,000 to 10,900 counts; the 68.4–69.4°
interval fell from about 68,700 to 2,300 counts. These are checks of the
joint optimizer against supplied measured values, not a claim of phase or
peak identification.

## 3.0.0b6.11

## Search boundaries, extended fit, and summed curve

Automatic selection now limits only peak-centre discovery. The fitted interval
includes a margin of 25% of the selected width on either side (at least eight
measurement steps). The optimiser uses the measured intensities in the wider
interval; the fitted Voigt profiles and the preview extend through that data.
The summed line evaluates confirmed components on the original measurement X
coordinates. It uses the common background if present, or the fitted local
background in each saved region. Separate, unfit regions are not joined by
invented model segments. Both display adapters use the same computed curve.

The supplied XRDML scan was checked around the crowded 31.7–32.9 degree group.
Selecting 30.6–33.6 degrees produced a fit over approximately 29.85–34.35
degrees while keeping detected centres within the selection. The median
absolute model deviation divided by experimental intensity over 400 preview
coordinates was about 4.1%. The proposed peak list still requires review:
asymmetric diffraction features and intensity steps need not follow a sum of
symmetric Voigt components.

In an offscreen Qt environment, 18 peak-related tests, 12 architecture tests,
and 14 shell tests passed. The peak tests include a numerical comparison of
the summed curve to a known synthetic measurement and confirmation of its
per-measurement checkbox on both renderers. `compileall` passed. The window
can start headlessly from the repacked archive; native Windows rendering and
mouse interaction still require local confirmation.

## Previous release (3.0.0b6.10)

## Drawn peak and local background

One selected measurement can receive a drawn peak. The user starts at the peak
top and drags to the desired background level and approximate half-width. A
Voigt component is fit without the automatic peak-detection threshold. Support
points inside the marked peak are bridged and the manual level is blended into
the pre-existing common background with a cosine taper. Outside the adjusted
interval the background is unchanged. Preview, cancellation, and confirmation
share the existing session-only peak table and work with both plot adapters.

The supplied XRDML scan was used to recheck the missing 26.41-degree peak with
0.5-degree background anchor spacing: a manually supplied local background
level of 630 counts (from the default-spacing estimate) recovers a fitted peak
at 26.41143 degrees with height 233 counts and leaves the 27.21-degree
background unchanged. A separate synthetic regression includes peak-
contaminated background anchors and verifies the local splice. GUI tests for
preview, cancellation, confirmation and re-drawing a table peak are included,
but native PySide6 is unavailable in this environment.

`PYTHONPATH=<local unit-cell-gui source> QT_QPA_PLATFORM=offscreen python -m
unittest discover -s tests -q` completed 210 tests: 168 passed and 42 skipped.
`python -m compileall -q xrd_workbench tests` completed successfully.

## Previous release (3.0.0b6.9)

## Manual peak detection with uneven intensities

The supplied `LB067_500C-90H-NaNO3_128YLN-6H-700C.xrdml` was parsed with
the bundled XRDML reader. With the default background spacing of 0.25 degrees,
the original search proposed only one peak at 28.5–30.2 degrees, and rejected
all peaks at 25.6–27.5 degrees because of a subthreshold local maximum. The
new search proposes three and two peaks in those intervals respectively.
The same candidates were recovered with 0.1-degree anchor spacing. With 0.5
degrees, the background absorbs the first peak at 26.41 degrees and only one
peak is proposed in the second interval; choosing an appropriate background
remains necessary.
Those are candidates for manual confirmation; intensity steps are not
automatically identified as crystalline reflections.

Service regressions also cover a weak peak next to an intense peak, a detector
step, and an ineligible local maximum. Native Qt visual interaction is not
available in the build environment.

`PYTHONPATH=<local unit-cell-gui source> QT_QPA_PLATFORM=offscreen python -m
unittest discover -s tests -q` completed 207 tests: 166 passed and 41 skipped.
`python -m compileall -q xrd_workbench tests` completed successfully.

## Previous release (3.0.0b6.8)

## Width-aware background regression

The regression creates broad bands with independently superimposed narrow
Voigt peaks, including an intense peak wider than one degree. It checks that
the broad bands remain in the calculated background while those peaks remain
above it. An approximate extraction of the user's plotted screenshot was used
for diagnostic comparison, but no original numerical scan was available.

`PYTHONPATH=<local unit-cell-gui source> QT_QPA_PLATFORM=offscreen python -m
unittest discover -s tests -q` completed 204 tests: 163 passed, 41 skipped.
PySide6 was unavailable locally; verify the background visually on Windows
against the original measurement.

## Previous release (3.0.0b6.7)

## Background and angle display update

The automatic background is now fitted once to median scan samples with arPLS
and constrained endpoints. Synthetic tests cover narrow and broad Voigt peaks,
a curved known background, endpoint accuracy, changing support-point density,
and exact reuse of a fixed global background in the Voigt result. Reflections
export theta and 2-theta in the same column order as the Structures table.

`PYTHONPATH=<local unit-cell-gui source> QT_QPA_PLATFORM=offscreen python -m
unittest discover -s tests -q` completed 203 tests: 162 passed, 41 skipped.
GUI tests, including the narrow correction-pane geometry check, were skipped
because PySide6 is unavailable locally. The background and angle display still
require an interactive Windows check with representative measurements.

## Previous release (3.0.0b6.6)

Environment for this release: Linux, NumPy and SciPy. The local PySide6 and
PyQtGraph installations are unavailable; GUI tests are therefore skipped.
The local unit-cell-gui 0.2.1 source was supplied on `PYTHONPATH` for the
independent geometry tests.

## Background and peak annotations

Four service-level tests cover a synthetic Voigt peak over a sloped noisy
background, baseline accuracy, continuous interpolation across an excluded
interval, anchor density in physical source-axis units, and Voigt fitting with
a global baseline plus a local adjustment. Eight GUI peak tests (including two
new interaction tests) were skipped because PySide6 is unavailable.

`PYTHONPATH=<local unit-cell-gui source> QT_QPA_PLATFORM=offscreen python -m
unittest discover -s tests -q` completed 200 tests: 160 passed and 40 skipped.
`python -m compileall -q xrd_workbench tests` completed successfully. The
interactive table checkboxes, hkl dialog and PyQtGraph/Matplotlib background
markers need a Windows visual check.

## Previous release (3.0.0b6.5)

## RAW1.01 primary-axis update

The 16 supplied instrument RAW1.01 files were loaded with both the binary
reader and the Viewer adapter. The recognized primary coordinates and endpoints
were checked against their recorded starts and steps. All intensity values and
acquisition statuses were retained; the aborted Z scan has 137 valid points.
Codes 9999 and 129 deliberately remain point-indexed. Nine RAW-geometry tests
and 37 core tests passed, including a regression check that fixed drives are
kept as metadata while known moving drives remain selectable.

## New behavior

Nine peak-search tests passed. They cover automatic selection of a single
measurement, independent peak tables after loading a second measurement,
numeric ordering and removal after sorting, a preview overlay without changing
plot dimensions, confirmation and cancellation with PyQtGraph and Matplotlib,
three and six-peak Voigt fitting, seven-peak limit, d-spacing display, and
batch Kα2/Kβ matching entirely from existing table rows. The matching test
asserts that no spectral fitting is called and no table rows are added. Kβ is
disabled under a cobalt preset. A four-peak fit and overlay were also inspected
in an offscreen screenshot. Six separated synthetic peaks fit in less than a
second in the development environment; this is not a performance guarantee for
arbitrary measurements.

## Complete suite

`QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -q` ran 194
tests for this version: 193 passed and one existing Matplotlib structure-export
test failed. The same failure was previously reproduced in the unmodified
3.0.0b6.1 source on this offscreen platform: the figure size changes after
export. It is outside this RAW reader change.

## Platform check

The Qt offscreen screenshot verifies that an overlay showing four peaks sits
inside the graph widget. Packaged Windows mouse interaction and appearance
still need a manual check. The peak tables remain session-only.
