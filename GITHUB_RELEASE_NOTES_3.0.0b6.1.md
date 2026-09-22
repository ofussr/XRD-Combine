# XRD Combine 3.0.0b6.1

Changes since the last published release, 3.0.0b2.

## Reciprocal-space maps

- Added an RSM workspace for experimental and calculated maps. Import Bruker RAW maps, numbered RAW files, or numbered XY series. Incomplete and unmeasured ranges remain empty rather than receiving invented intensities.
- View experimental maps in angular or Qx/Qz coordinates, with linear or logarithmic intensity and a choice of color maps. Export maps to PNG.
- Added **Real** bounds for acquired map cells and **Full** bounds for the complete range declared in a RAW file. Omega start, step, and end are read from RAW geometry; for XY series, any two values determine the third.
- Overlay indexed reflections from a loaded or newly opened CIF on acquired cells of an experimental map. The overlay follows the selected wavelength, orientation, scattering-plane tolerance, Miller-index limit, and coordinate view; it can be removed without changing the measured data.
- Calculate reciprocal-lattice positions from a CIF using the application's existing structure and symmetry model. Set the surface normal, in-plane reference, scattering-plane tolerance, index bounds, and target reflection or coordinates. Calculated maps show reflection positions, not simulated peak intensities or instrument broadening.
- Disabled wheel and drag zoom on both RSM plots, including their axes. Right-button drag still pans; display bounds are controlled through **Real/Full** and the calculated-map range fields.

## Bruker RAW import and Project data

- Corrected RAW v3 scan axes using confirmed numeric drive codes: 1 for 2Theta, 3 for Theta, and 5 for Phi. Ordinary scans, rocking curves, pole figures, and reciprocal-space maps now use the same RAW v3/v4 geometry parser. Coupled scans retain their individual moving-drive coordinates.
- Unrecognized scan modes remain explicitly ambiguous instead of receiving a guessed physical axis. A JSON diagnostic export exposes retained RAW header and range fields for further instrument analysis.
- The Project data panel and its columns are mouse-resizable and remember their widths. Measurements imported from one file are grouped under an expandable source-file node.

## Viewer, correction, and gallery

- Added substrate peak correction using two or three Gaussian-located orders of one reflection family. Without a known true position, it fits their common angular shift. With a known true position, it derives the remaining orders from Bragg's law and fits an affine measured-to-true angle correction. Residuals are shown before application.
- Added a d-spacing display axis for compatible scans and phase reflections using a selected wavelength. Clicking an overlaid measurement or phase selects the nearest visible item.
- Canceling peak alignment now clears its temporary markers and selection graphics. On logarithmic plots, finite zero and negative intensities use a display floor without modifying the source data; missing values remain gaps.
- Restored compact, fixed-size thumbnails in the Comparison gallery, wrapping to fit the available width.

## Validation

The 3.0.0b6.1 source passed 183 of 184 automated tests in a Linux Qt offscreen environment. One OpenGL screenshot test was skipped because that environment has no compatible graphics context. The parser and RSM views were also checked with an interrupted Bruker RAW map; its missing cells remained empty. Windows visual behavior and packaged builds still need a local check.
