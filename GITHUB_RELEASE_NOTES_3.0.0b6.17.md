# XRD Combine 3.0.0b6.17

Changes since the last published release, 3.0.0b6.1.

## Peak search and fitting

- Added range-based peak search in the Viewer with up to six distinguishable Voigt components per selection. Fits use measured intensities independently of the current display scale.
- Review fitted profiles and peak positions before adding them. Each proposed component has its own confirmation checkbox; cancelling preserves the existing peaks and background. Preview controls float over the plot without changing its dimensions.
- The selected interval restricts candidate peak centres, while fitting and profile tails include measured points beyond its boundaries.
- Improved detection of weak peaks beside much stronger reflections by assessing candidate prominence against local noise. Repeated searches subtract already accepted profiles before looking for new candidates, reducing duplicate detections from their tails.
- Accepting a new region jointly refines its peaks and nearby overlapping accepted components. Distant groups retain their existing parameters. Scaled fit parameters and bounded optimiser evaluations improve repeated searches in neighbouring regions.

## Manual peaks and refinement

- Added **Draw peak** to define a Voigt component by its centre, height and full width at half maximum (FWHM). Confirmation stores the drawn parameters and chosen local background without running an optimiser.
- Manually drawn components can represent visible peaks missed by candidate detection. Redrawing an existing peak updates it while preserving its identity and assignments.
- Added **Do Fit** below Draw peak in the Viewer and in each peak table. It refines accepted and manually drawn components against the measurement. Overlapping components are refined jointly within each group; separate groups are fitted independently.

## Background

- Added a **Background** section with a separate background for each measurement and coordinate axis. Calculate a baseline, set anchor spacing in source-axis units, exclude anchors within a selected interval, or remove the background. The plot shows both the continuous curve and its support points.
- Background estimation preserves broad scattering bands while rejecting narrower superimposed peaks. Anchor spacing controls support-point density and influences the distinction between broad background features and narrow peaks.
- Confirming detected groups and running Do Fit can refine the shared background locally around accepted peaks. Corrections fade smoothly at group boundaries, preserve the background outside their support, and keep the automatically refined background non-negative. A further local downward correction addresses model overestimates on measured shoulders.
- Added **Show background + fitted peaks** to compare the summed model with the experiment. The curve follows the selected coordinate view, intensity scale and measurement transformations. Background estimation and peak fitting preserve the original measured data.

## Peak tables and indexing

- Added a separate non-modal peak table for each measurement, showing position, height, FWHM, integrated profile area, spectral classification and hkl assignments. Tables provide actions to add, draw, refine and remove peaks.
- Rows are sorted by displayed position, while stable internal identifiers preserve peak identity through sorting, deletion, redrawing and refinement. Each peak has an independent **Fill** checkbox controlling its shaded profile on the plot.
- Assign one or more hkl triples to a peak using loaded CIF or Cell Phase structures, or remove assignments with **Clear hkl**.
- Added **Find Kα2** and **Find Kβ** to propose matches between existing peaks and expected spectral-line positions. Users review the proposed pairs before classification; confirmed matches receive matching colours in the table and plot. These actions classify existing entries without adding or fitting new peaks.
- Kα2 matching supports Cu and Co radiation presets; Kβ matching is available for Cu. Custom radiation is not classified automatically. Removing a primary peak clears the spectral classifications linked to it.

## Bruker RAW import

- Extended RAW1.01 primary-axis recognition to locked coupled scans (code 0), detector scans (2), Chi (4), X-Drive (6) and Z-Drive (8). The Viewer uses their recorded starting coordinates and steps; support for previously recognised codes 1, 3 and 5 is retained.
- Fixed motor positions no longer appear as selectable per-point axes. They remain in the RAW metadata, while confirmed coordinates that vary during the scan remain selectable.
- Retained additional file and range metadata, original channel data, drive starting positions, acquisition timing, generator settings and measurement status. Technical scans with codes 9999 and 129 remain indexed by point until their physical coordinates can be established.

## Viewer and reflection details

- A single loaded measurement is selected automatically. Adding another measurement preserves the current selection.
- Moved the numeric **X shift**, **Y zero shift** and **Y multiplier** fields onto separate rows from their sliders, keeping them accessible in a narrow correction panel.
- Added θ alongside 2θ in the Structures reflection table and CSV export, calculated pole-figure centring choices, and reflection details.
- Closing the main window, including through the File menu, now asks for confirmation. **Cancel** leaves the application open.

## Startup and Windows packaging

- Added a static startup image displayed by the PyInstaller bootloader before Qt loads. It is limited to approximately 650 × 433 pixels and closes after the main window is shown. Direct Python launches continue to work without the splash module.
- Added a Windows build specification that retains the directory-based distribution, with the executable and supporting files together in `dist\XRD Combine`. It includes the runtime resources, reference database, required packages and licence notices.

## Validation and limitations

The latest recorded full suite, for 3.0.0b6.16, passed 221 automated tests in a Linux Qt offscreen environment, with one OpenGL test skipped. For 3.0.0b6.17, checks covered the directory-build configuration, resource paths and splash-component placement; the full application suite was not rerun. Packaged Windows startup and visual behaviour still require a native Windows check.

Peak tables, backgrounds and hkl assignments are retained only for the current session. Fits use symmetric Voigt profiles, so background errors, asymmetric reflections and missing components can leave visible residuals. Background refinement improves the model but does not guarantee an exact match. Matplotlib remains required by other plotting and export paths.
