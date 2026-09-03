"""Runtime access to the compact spglib-derived space-group table."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

try:
    from .i18n import localised
except ImportError:  # pragma: no cover
    from i18n import localised


@dataclass(frozen=True)
class SpaceGroupSetting:
    hall_number: int
    number: int
    international_short: str
    international_full: str
    choice: str
    hall_symbol: str
    operations: tuple[str, ...]

    @property
    def label(self) -> str:
        suffix = f" [{self.choice}]" if self.choice else ""
        return f"{self.number}: {self.international_short}{suffix}"


def _data_path() -> Path:
    candidates = (
        Path(__file__).resolve().parent / "resources" / "space_groups.json",
        Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        / "space_groups.json",
        Path(sys.executable).resolve().parent / "space_groups.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("space_groups.json was not found")


def _load() -> tuple[SpaceGroupSetting, ...]:
    data = json.loads(_data_path().read_text(encoding="utf-8"))
    return tuple(
        SpaceGroupSetting(
            hall_number=int(item["hall_number"]),
            number=int(item["number"]),
            international_short=item["international_short"],
            international_full=item["international_full"],
            choice=item["choice"],
            hall_symbol=item["hall_symbol"],
            operations=tuple(item["operations"]),
        )
        for item in data["settings"]
    )


SETTINGS = _load()
BY_HALL_NUMBER = {item.hall_number: item for item in SETTINGS}
BY_LABEL = {item.label: item for item in SETTINGS}


def setting_from_label(label: str) -> SpaceGroupSetting:
    try:
        return BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Unknown space-group setting: {label}") from exc


def _normalise_setting_text(value: str) -> str:
    return re.sub(r"[\s_]", "", value).casefold()


def setting_from_user_text(value: str) -> SpaceGroupSetting:
    """Resolve a typed IT number, Hermann-Mauguin symbol, or Hall setting.

    When a short Hermann-Mauguin symbol has several axis/origin choices, the
    first spglib setting is the conventional default.  A choice in brackets or
    a full symbol selects the requested non-default setting explicitly.
    """
    text = value.strip()
    if text in BY_LABEL:
        return BY_LABEL[text]
    if not text:
        raise ValueError(localised(
            "Enter a space group.",
            "Saisissez un groupe d’espace.",
            "Введите пространственную группу.",
        ))

    hall_match = re.fullmatch(r"hall\s*(?:no\.?|number|№|#)?\s*[:=]?\s*(\d+)", text, re.I)
    if hall_match:
        setting = BY_HALL_NUMBER.get(int(hall_match.group(1)))
        if setting is not None:
            return setting

    if text.isdecimal():
        number = int(text)
        match = next((item for item in SETTINGS if item.number == number), None)
        if match is not None:
            return match

    needle = _normalise_setting_text(text)
    matches: list[SpaceGroupSetting] = []
    for item in SETTINGS:
        suffix = f"[{item.choice}]" if item.choice else ""
        aliases = {
            item.label,
            item.international_short,
            item.international_full,
            item.hall_symbol,
            f"{item.number}:{item.international_short}",
            f"{item.international_short}{suffix}",
            f"{item.number}:{item.international_short}{suffix}",
        }
        if needle in {_normalise_setting_text(alias) for alias in aliases}:
            matches.append(item)
    if matches:
        return matches[0]

    raise ValueError(localised(
        f"Unknown space group or setting: {value!r}.",
        f"Groupe d’espace ou choix inconnu : {value!r}.",
        f"Неизвестная пространственная группа или установка: {value!r}.",
    ))
