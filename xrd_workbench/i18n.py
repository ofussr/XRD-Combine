"""Tkinter localisation adapter and compatibility facade."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog as _filedialog
from tkinter import messagebox as _messagebox
from tkinter import ttk

try:
    from .localization import (
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
except ImportError:
    from localization import (
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


class LocalizedStringVar(tk.StringVar):
    """A StringVar supporting both legacy text and stable catalogue keys."""

    def __init__(
        self,
        *args,
        translation_key: str | None = None,
        translation_values: dict[str, object] | None = None,
        **kwargs,
    ) -> None:
        self.translation_key = translation_key
        self.translation_values = dict(translation_values or {})
        if translation_key is not None:
            kwargs["value"] = tr(translation_key, **self.translation_values)
        super().__init__(*args, **kwargs)

    def set(self, value) -> None:
        self.translation_key = None
        self.translation_values = {}
        super().set(translate_text(value))

    def set_key(self, key: str, **values: object) -> None:
        self.translation_key = key
        self.translation_values = dict(values)
        super().set(tr(key, **values))

    def retranslate(self, language: str | None = None) -> None:
        if self.translation_key is not None:
            super().set(
                tr(self.translation_key, language, **self.translation_values)
            )
        else:
            super().set(translate_text(self.get(), language))


def bind_widget_text(widget: tk.Misc, key: str, **values: object) -> None:
    """Attach a stable translation key to a widget and set its current text."""

    widget._translation_key = key
    widget._translation_values = dict(values)
    widget.configure(text=tr(key, **values))


def _translate_widget(widget: tk.Misc, language: str) -> None:
    if hasattr(widget, "localize_heading"):
        widget.localize_heading(language)
    try:
        key = widget._translation_key
    except AttributeError:
        key = None
    if key is not None:
        try:
            widget.configure(
                text=tr(
                    key,
                    language,
                    **getattr(widget, "_translation_values", {}),
                )
            )
        except (tk.TclError, AttributeError):
            pass
    else:
        try:
            text = widget.cget("text")
            matched_key = key_for_text(text)
            if matched_key is not None:
                widget._translation_key = matched_key
                widget._translation_values = {}
                widget.configure(text=tr(matched_key, language))
        except (tk.TclError, AttributeError):
            pass

    try:
        variable_name = widget.cget("textvariable")
        if variable_name:
            value = widget.getvar(variable_name)
            translated = translate_text(value, language)
            if translated != value:
                widget.setvar(variable_name, translated)
    except (tk.TclError, AttributeError):
        pass

    if isinstance(widget, ttk.Combobox):
        values = tuple(widget.cget("values"))
        translated = tuple(translate_text(value, language) for value in values)
        if translated != values:
            widget.configure(values=translated)

    if isinstance(widget, ttk.Notebook):
        for tab_id in widget.tabs():
            text = widget.tab(tab_id, "text")
            widget.tab(tab_id, text=translate_text(text, language))

    if isinstance(widget, ttk.Treeview):
        for column in ("#0", *widget.cget("columns")):
            try:
                text = widget.heading(column, "text")
                widget.heading(column, text=translate_text(text, language))
            except tk.TclError:
                pass

    if isinstance(widget, tk.Menu):
        end = widget.index("end")
        if end is not None:
            for index in range(end + 1):
                try:
                    label = widget.entrycget(index, "label")
                    widget.entryconfigure(index, label=translate_text(label, language))
                except tk.TclError:
                    pass


def apply_language(root: tk.Misc, language: str | None = None) -> None:
    target = language or get_language()
    _translate_widget(root, target)
    if isinstance(root, (tk.Tk, tk.Toplevel)):
        try:
            menu_name = root.cget("menu")
            if menu_name:
                apply_language(root.nametowidget(menu_name), target)
        except (tk.TclError, KeyError):
            pass
    for child in root.winfo_children():
        apply_language(child, target)


class _MessageboxProxy:
    def __getattr__(self, name):
        function = getattr(_messagebox, name)

        def call(title=None, message=None, **kwargs):
            if title is not None:
                title = translate_text(title)
            if message is not None:
                message = translate_text(message)
            return function(title=title, message=message, **kwargs)

        return call


class _FileDialogProxy:
    def __getattr__(self, name):
        function = getattr(_filedialog, name)

        def call(**kwargs):
            if "title" in kwargs:
                kwargs["title"] = translate_text(kwargs["title"])
            if "filetypes" in kwargs:
                kwargs["filetypes"] = tuple(
                    (translate_text(label), pattern)
                    for label, pattern in kwargs["filetypes"]
                )
            return function(**kwargs)

        return call


messagebox = _MessageboxProxy()
filedialog = _FileDialogProxy()


__all__ = [
    "CATALOGS",
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LocalizedStringVar",
    "apply_language",
    "bind_widget_text",
    "catalogue_keys",
    "choice_code",
    "filedialog",
    "get_language",
    "key_for_text",
    "load_language",
    "localised",
    "messagebox",
    "set_language",
    "tr",
    "translate_text",
]
