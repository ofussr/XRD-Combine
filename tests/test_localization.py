from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

from xrd_workbench.localization import (
    CATALOGS,
    catalogue_keys,
    key_for_text,
    localised,
    set_language,
    tr,
    translate_text,
)


ROOT = Path(__file__).resolve().parents[1]


class LocalizationTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_language("en")

    def test_language_files_have_identical_stable_keys(self) -> None:
        expected = set(CATALOGS["en"])
        self.assertEqual(len(catalogue_keys()), 430)
        self.assertEqual(set(CATALOGS), {"en", "fr", "ru"})
        for catalogue in CATALOGS.values():
            self.assertEqual(set(catalogue), expected)

    def test_key_translation_and_named_interpolation(self) -> None:
        self.assertEqual(tr("text.open_files", "en"), "Open files…")
        self.assertEqual(tr("text.open_files", "fr"), "Ouvrir des fichiers…")
        self.assertEqual(tr("text.open_files", "ru"), "Открыть файлы…")
        self.assertEqual(
            tr("viewer.send_to_comparison_count", "ru", count=4),
            "Отправить в сравнение (4)",
        )
        self.assertEqual(
            tr("pole.rotate_phases_together", "fr"),
            "Faire tourner les phases ensemble",
        )
        self.assertEqual(tr("unknown.key", "en"), "unknown.key")

    def test_legacy_text_lookup_and_localised_calls_remain_compatible(self) -> None:
        set_language("fr")
        self.assertEqual(translate_text("Открыть файлы…"), "Ouvrir des fichiers…")
        self.assertEqual(key_for_text("Open files…"), "text.open_files")
        self.assertEqual(localised("One", "Deux", "Три"), "Deux")
        set_language("ru")
        self.assertEqual(translate_text("Cancel"), "Отмена")

    def test_core_localization_import_does_not_load_gui_toolkits(self) -> None:
        code = (
            "import sys; import xrd_workbench.localization; "
            "raise SystemExit(1 if ('tkinter' in sys.modules or "
            "any(name.startswith('matplotlib') for name in sys.modules)) else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0)

    def test_catalogued_static_text_uses_stable_keys_at_call_sites(self) -> None:
        violations: list[str] = []
        for path in (ROOT / "xrd_workbench").rglob("*.py"):
            if "localization" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "translate_text"
                    and node.args
                ):
                    try:
                        value = ast.literal_eval(node.args[0])
                    except (ValueError, TypeError):
                        value = None
                    if key_for_text(value) is not None:
                        violations.append(f"{path.name}:{node.lineno}: translate_text")
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "localised"
                    and len(node.args) >= 3
                ):
                    try:
                        keys = [
                            key_for_text(ast.literal_eval(argument))
                            for argument in node.args[:3]
                        ]
                    except (ValueError, TypeError):
                        keys = []
                    if keys and keys[0] is not None and len(set(keys)) == 1:
                        violations.append(f"{path.name}:{node.lineno}: localised")
                for keyword in node.keywords:
                    if (
                        keyword.arg == "text"
                        and isinstance(keyword.value, ast.Constant)
                        and key_for_text(keyword.value.value) is not None
                    ):
                        violations.append(f"{path.name}:{node.lineno}: text literal")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
