"""Shared localisation API for current and future interfaces."""

from .catalog import (
    CATALOGS,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    catalogue_keys,
    choice_code,
    get_language,
    key_for_text,
    load_language,
    localised,
    set_language,
    tr,
    translate_text,
)

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
