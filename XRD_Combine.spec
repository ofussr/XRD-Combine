# -*- mode: python ; coding: utf-8 -*-
"""Windows one-directory build with an early, image-only startup splash."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all


project = Path(SPECPATH).resolve()
resources = project / "xrd_workbench" / "resources"
runtime_files = (
    "xrd_combine.ico",
    "xrd_combine.png",
    "atom_styles.json",
    "space_groups.json",
    "f0_WaasKirf.dat",
)
datas = [(str(resources / name), ".") for name in runtime_files]
datas += [
    (str(project / "xrd_workbench" / "pivo.json"), "."),
    (str(project / "LICENSE"), "."),
    (str(project / "THIRD_PARTY_NOTICES.txt"), "."),
]
binaries = []
hiddenimports = []
for package in ("unit_cell_gui", "pyqtgraph", "spglib"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(project / "run_xrd_combine.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
splash = Splash(
    str(resources / "startup_splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    max_img_size=(650, 434),
    text_pos=None,
    center="active",
)
exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name="XRD Combine",
    console=False,
    contents_directory=".",
    icon=str(resources / "xrd_combine.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    name="XRD Combine",
)
