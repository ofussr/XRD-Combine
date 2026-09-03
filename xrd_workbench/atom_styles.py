"""Compact atom drawing styles and user colour overrides."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    from .i18n import localised
except ImportError:  # pragma: no cover
    from i18n import localised


PALETTES = ("jmol", "cpk", "molcas_gv")
FALLBACK_COLOUR = "#b0b0b0"
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _resource_path() -> Path:
    candidates = (
        Path(__file__).resolve().parent / "resources" / "atom_styles.json",
        Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        / "atom_styles.json",
        Path(sys.executable).resolve().parent / "atom_styles.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("atom_styles.json was not found")


def _settings_path() -> Path:
    return Path.home() / ".xrd_combine_atom_colours.json"


def _load_reference() -> dict:
    return json.loads(_resource_path().read_text(encoding="utf-8"))


_REFERENCE = _load_reference()
ELEMENTS: dict[str, dict] = _REFERENCE["elements"]
_palette = "jmol"
_custom: dict[str, str] = {}


def _load_settings() -> None:
    global _palette, _custom
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    palette = data.get("palette")
    custom = data.get("custom")
    if palette in PALETTES:
        _palette = palette
    if isinstance(custom, dict):
        _custom = {
            symbol: colour.upper()
            for symbol, colour in custom.items()
            if symbol in ELEMENTS and isinstance(colour, str) and _HEX.fullmatch(colour)
        }


def _save_settings() -> None:
    path = _settings_path()
    path.write_text(
        json.dumps({"palette": _palette, "custom": _custom}, indent=2) + "\n",
        encoding="utf-8",
    )


def palette() -> str:
    return _palette


def set_palette(name: str) -> None:
    global _palette
    if name not in PALETTES:
        raise ValueError(
            localised(
                f"Unknown atom colour palette: {name}",
                f"Palette de couleurs atomiques inconnue : {name}",
                f"Неизвестная палитра цветов атомов: {name}",
            )
        )
    _palette = name
    _save_settings()


def custom_colours() -> dict[str, str]:
    return dict(_custom)


def atom_colour(element: str) -> str:
    if element in _custom:
        return _custom[element]
    value = ELEMENTS.get(element, {}).get(_palette)
    return value.upper() if isinstance(value, str) and _HEX.fullmatch(value) else FALLBACK_COLOUR


def atom_ball_radius(element: str) -> float:
    """Return a readable sphere radius derived from the Pyykko radius."""
    radius_pm = ELEMENTS.get(element, {}).get("covalent_radius_pyykko")
    if not isinstance(radius_pm, (int, float)) or not math.isfinite(radius_pm):
        return 0.22
    radius_angstrom = float(radius_pm) / 100.0
    return min(0.38, max(0.13, 0.11 + 0.12 * radius_angstrom))


def covalent_radius(element: str) -> float:
    data = ELEMENTS.get(element, {})
    radius_pm = data.get("covalent_radius_cordero")
    if not isinstance(radius_pm, (int, float)) or not math.isfinite(radius_pm):
        radius_pm = data.get("covalent_radius_pyykko")
    if not isinstance(radius_pm, (int, float)) or not math.isfinite(radius_pm):
        return 1.0
    return float(radius_pm) / 100.0


def parse_custom_colours(lines: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        symbol = parts[0][:1].upper() + parts[0][1:].lower()
        if symbol not in ELEMENTS:
            raise ValueError(
                localised(
                    f"Line {line_number}: unknown element {parts[0]!r}.",
                    f"Ligne {line_number} : élément inconnu {parts[0]!r}.",
                    f"Строка {line_number}: неизвестный элемент {parts[0]!r}.",
                )
            )
        if symbol in result:
            raise ValueError(
                localised(
                    f"Line {line_number}: duplicate element {symbol}.",
                    f"Ligne {line_number} : élément {symbol} répété.",
                    f"Строка {line_number}: элемент {symbol} указан повторно.",
                )
            )
        if len(parts) == 2 and _HEX.fullmatch(parts[1]):
            colour = parts[1].upper()
        elif len(parts) == 4:
            try:
                channels = tuple(int(value) for value in parts[1:])
            except ValueError as exc:
                raise ValueError(localised(
                    f"Line {line_number}: RGB channels must be integers.",
                    f"Ligne {line_number} : les composantes RGB doivent être entières.",
                    f"Строка {line_number}: компоненты RGB должны быть целыми числами.",
                )) from exc
            if any(value < 0 or value > 255 for value in channels):
                raise ValueError(localised(
                    f"Line {line_number}: RGB channels must be in 0–255.",
                    f"Ligne {line_number} : les composantes RGB doivent être comprises entre 0 et 255.",
                    f"Строка {line_number}: компоненты RGB должны находиться в диапазоне от 0 до 255.",
                ))
            colour = "#" + "".join(f"{value:02X}" for value in channels)
        else:
            raise ValueError(localised(
                f"Line {line_number}: use ELEMENT #RRGGBB or ELEMENT R G B.",
                f"Ligne {line_number} : utilisez ELEMENT #RRGGBB ou ELEMENT R G B.",
                f"Строка {line_number}: используйте ELEMENT #RRGGBB или ELEMENT R G B.",
            ))
        result[symbol] = colour
    if not result:
        raise ValueError(localised(
            "The file contains no atom colours.",
            "Le fichier ne contient aucune couleur atomique.",
            "В файле нет цветов атомов.",
        ))
    return result


def import_custom_colours(path: str | Path) -> dict[str, str]:
    global _custom
    parsed = parse_custom_colours(
        Path(path).read_text(encoding="utf-8-sig").splitlines()
    )
    _custom = parsed
    _save_settings()
    return custom_colours()


def reset_custom_colours() -> None:
    global _custom
    _custom = {}
    _save_settings()


def export_custom_colours(path: str | Path) -> None:
    lines = [
        "; XRD Combine custom atom colours",
        "; Format: ELEMENT #RRGGBB",
    ]
    lines.extend(f"{symbol} {_custom[symbol]}" for symbol in sorted(_custom))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


_load_settings()
