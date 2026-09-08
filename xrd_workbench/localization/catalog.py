"""GUI-independent language selection and translation catalogue access."""

from __future__ import annotations

import json
from pathlib import Path
from string import Formatter

from .locales.en import STRINGS as ENGLISH
from .locales.fr import STRINGS as FRENCH
from .locales.ru import STRINGS as RUSSIAN


LANGUAGES = {"en": "English", "fr": "Français", "ru": "Русский"}
DEFAULT_LANGUAGE = "en"
CATALOGS: dict[str, dict[str, str]] = {
    "en": ENGLISH,
    "fr": FRENCH,
    "ru": RUSSIAN,
}

_language = DEFAULT_LANGUAGE
_settings_path = Path.home() / ".xrd_combine.json"
_legacy_settings_path = Path.home() / ".xrd_workbench.json"


def _validate_catalogues() -> None:
    expected = set(ENGLISH)
    for language, catalogue in CATALOGS.items():
        missing = expected - set(catalogue)
        extra = set(catalogue) - expected
        if missing or extra:
            raise RuntimeError(
                f"Localisation catalogue {language!r} differs from English: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
        for key, source in ENGLISH.items():
            source_fields = {
                field_name
                for _, field_name, _, _ in Formatter().parse(source)
                if field_name is not None
            }
            translated_fields = {
                field_name
                for _, field_name, _, _ in Formatter().parse(catalogue[key])
                if field_name is not None
            }
            if source_fields != translated_fields:
                raise RuntimeError(
                    f"Localisation placeholders differ for {language}:{key}"
                )


_validate_catalogues()

_ALIASES: dict[str, str] = {}
for _key in ENGLISH:
    for _code in ("en", "fr", "ru"):
        _ALIASES[CATALOGS[_code][_key]] = _key


def catalogue_keys() -> tuple[str, ...]:
    return tuple(ENGLISH)


def get_language() -> str:
    return _language


def load_language() -> str:
    for path in (_settings_path, _legacy_settings_path):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("language")
        except (OSError, ValueError, TypeError):
            continue
        if value in LANGUAGES:
            return value
    return DEFAULT_LANGUAGE


def set_language(language: str, persist: bool = False) -> None:
    global _language
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    _language = language
    if persist:
        try:
            _settings_path.write_text(
                json.dumps({"language": language}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass


def tr(key: str, language: str | None = None, **values: object) -> str:
    """Translate a stable key and interpolate named values."""

    target = language if language in CATALOGS else _language
    template = CATALOGS[target].get(key, ENGLISH.get(key, key))
    return template.format(**values) if values else template


def key_for_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _ALIASES.get(value)


def translate_text(value: object, language: str | None = None) -> object:
    """Translate a legacy visible string through the keyed catalogues."""

    key = key_for_text(value)
    if key is None:
        return value
    return tr(key, language)


def localised(en: str, fr: str, ru: str) -> str:
    """Compatibility helper for call sites not yet converted to stable keys."""

    return {"en": en, "fr": fr, "ru": ru}[_language]


_CHOICE_KEYS = {
    "scale": {
        "linear": "text.linear",
        "log": "text.logarithmic",
        "sqrt": "text.square_root",
        "square": "text.square",
    },
    "phase": {
        "sticks": "text.sticks",
        "profile": "text.profile",
    },
    "phase_layout": {
        "separate": "text.separate",
        "overlay": "text.overlay",
    },
    "projection": {
        "stereographic": "text.stereographic",
        "equal_area": "text.equal_area",
    },
}


def choice_code(group: str, value: str) -> str:
    for code, key in _CHOICE_KEYS[group].items():
        if value in {CATALOGS[language][key] for language in CATALOGS}:
            return code
    return value


__all__ = [
    "CATALOGS",
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "catalogue_keys",
    "choice_code",
    "get_language",
    "key_for_text",
    "load_language",
    "localised",
    "set_language",
    "tr",
    "translate_text",
]
