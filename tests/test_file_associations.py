from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from xrd_workbench.application_resources import resource_path
from xrd_workbench.file_associations import (
    ASSOCIATION_SUFFIXES,
    CLASSES_ROOT,
    PROG_ID,
    WindowsFileAssociations,
    open_command,
    runtime_executable,
)


class FileAssociationTests(unittest.TestCase):
    def test_only_unambiguous_supported_formats_are_registered(self) -> None:
        self.assertEqual(ASSOCIATION_SUFFIXES, (".xrdml", ".raw", ".xy", ".cif"))

    def test_shell_command_quotes_executable_and_file(self) -> None:
        executable = Path(r"C:\Program Files\XRD Combine\XRD Combine.exe")
        self.assertEqual(
            open_command(executable),
            r'"C:\Program Files\XRD Combine\XRD Combine.exe" "%1"',
        )

    def test_source_interpreter_is_never_registered(self) -> None:
        with patch("xrd_workbench.file_associations.sys.platform", "win32"), patch.object(
            sys,
            "frozen",
            False,
            create=True,
        ):
            self.assertIsNone(runtime_executable())

    def test_registration_writes_only_per_user_application_keys(self) -> None:
        manager = WindowsFileAssociations(r"C:\Apps\XRD Combine.exe")
        writes: list[tuple[str, str, str]] = []
        empty_writes: list[tuple[str, str]] = []
        with patch.object(
            manager,
            "_set_value",
            side_effect=lambda path, name, value: writes.append((path, name, value)),
        ), patch.object(
            manager,
            "_set_empty_value",
            side_effect=lambda path, name: empty_writes.append((path, name)),
        ), patch.object(manager, "_notify_shell"):
            manager.register()

        self.assertTrue(all(path.startswith(CLASSES_ROOT) for path, _name, _value in writes))
        self.assertIn(
            (manager.application_key, "FriendlyAppName", "XRD Combine"),
            writes,
        )
        for suffix in ASSOCIATION_SUFFIXES:
            self.assertIn(
                (manager.application_key + r"\SupportedTypes", suffix, ""),
                writes,
            )
            self.assertIn(
                (rf"{CLASSES_ROOT}\{suffix}\OpenWithProgids", PROG_ID),
                empty_writes,
            )

    def test_application_icon_resources_are_shipped(self) -> None:
        ico = resource_path("xrd_combine.ico")
        png = resource_path("xrd_combine.png")
        self.assertIsNotNone(ico)
        self.assertIsNotNone(png)
        self.assertGreater(ico.stat().st_size, 0)
        self.assertGreater(png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
