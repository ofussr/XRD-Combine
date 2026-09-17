# XRD Combine 3.0.0b2 Release Notes

Version 3.0.0b2 adds the first application branding and an experimental,
reversible Windows file-association controller. The diffraction, plotting, and
structure-calculation code is unchanged from 3.0.0b1.

## Changes in 3.0.0b2

- XRD Combine now ships its application icon as both a multi-resolution ICO
  file and a full-size PNG image.
- The icon is assigned to the Qt application and its windows at runtime.
- The About dialog now displays the XRD Combine logo.
- The hidden **About > Debug** dialog now contains a Windows file-association
  section.
- XRDML, Bruker RAW, XY, and CIF associations can be registered or removed for
  the current Windows user without administrator rights.
- The dialog reports the current registration state of every supported
  extension.
- Registration is available only from a compiled executable, preventing a
  source launch from accidentally associating files with `python.exe`.
- Files opened through Windows are passed to the existing command-line import
  path. Quoted paths containing spaces are supported.

Generic XML, TXT, DAT, and CSV extensions remain openable from inside XRD
Combine but are intentionally not registered with Windows because these
extensions are widely used by unrelated applications.

## Packaging

The executable icon and the two runtime image resources are separate. A
PyInstaller build must use the ICO as its executable icon and include both
files as data:

```text
--icon "xrd_workbench\resources\xrd_combine.ico" --add-data "xrd_workbench\resources\xrd_combine.ico:." --add-data "xrd_workbench\resources\xrd_combine.png:."
```

The equivalent Nuitka options are:

```text
--windows-icon-from-ico=xrd_workbench/resources/xrd_combine.ico --include-data-file=xrd_workbench/resources/xrd_combine.ico=xrd_combine.ico --include-data-file=xrd_workbench/resources/xrd_combine.png=xrd_combine.png
```

## Compatibility

- Python 3.10 or newer;
- PySide6 and PyQtGraph are required;
- `unit-cell-gui>=0.2.1,<0.3`;
- file-association controls are enabled only in compiled Windows builds.

## Validation

The source tree contains 161 automated tests. In the release environment, 160
tests passed and one OpenGL screenshot test was skipped because the Qt
offscreen platform did not provide a graphics context. Actual Windows registry
integration still requires a manual check from the compiled executable.

See `VALIDATION.md` for the complete validation record.
