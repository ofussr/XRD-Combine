# XRD Combine Validation

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
