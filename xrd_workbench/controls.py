"""Shared, local wheel handling and non-destructive collapsible controls."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from .i18n import translate_text
except ImportError:
    from i18n import translate_text


class CollapsibleSection(ttk.LabelFrame):
    """A LabelFrame whose heading hides its children, keeping their state."""

    def __init__(self, parent, *, text, expanded=False, **kwargs):
        super().__init__(parent, **kwargs)
        self.title_source = text
        self.expanded = True
        self._saved_layout = []
        self._expanded_padding = self.cget("padding")
        self.heading = ttk.Button(self, command=self.toggle, takefocus=True)
        self.configure(labelwidget=self.heading)
        self.localize_heading()
        # Widgets are added by the caller after this constructor returns, so
        # the initial collapse has to run once Tk has finished building them.
        if not expanded:
            self.after_idle(self.collapse)

    def localize_heading(self, language=None):
        marker = "▼" if self.expanded else "▶"
        self.heading.configure(text=f"{marker} {translate_text(self.title_source, language)}")

    def toggle(self):
        if self.expanded:
            self._saved_layout = []
            # pack_slaves preserves packing order; grid remembers cell positions.
            for widget in self.pack_slaves():
                self._saved_layout.append((widget, "pack", widget.pack_info()))
                widget.pack_forget()
            for widget in self.grid_slaves():
                self._saved_layout.append((widget, "grid", widget.grid_info()))
                widget.grid_remove()
            self.configure(padding=0)
            # With no managed children Tk otherwise retains the previous size.
            self.configure(height=self.heading.winfo_reqheight()+6)
        else:
            self.configure(padding=self._expanded_padding, height=0)
            for widget, manager, options in self._saved_layout:
                getattr(widget, manager)(**options)
            self._saved_layout = []
        self.expanded = not self.expanded
        self.localize_heading()
        self.after_idle(self._notify_layout)

    def expand(self):
        if not self.expanded:
            self.toggle()

    def collapse(self):
        if self.expanded:
            self.toggle()

    def _notify_layout(self):
        parent = self.master
        while parent is not None:
            if isinstance(parent, ScrollableControls):
                # Nested grid/pack requests reach the canvas on a later idle
                # pass. Settle those requests before measuring the new height.
                parent.body.update_idletasks()
                parent._resize_content()
                break
            parent = parent.master


class ScrollableControls(ttk.Frame):
    def __init__(self, parent, *, width=335, padding=8):
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        background = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0,
                                background=background, yscrollincrement=20)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.body = ttk.Frame(self.canvas, padding=padding)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._resize_content)
        self.canvas.bind("<Configure>", self._resize_canvas)
        # Instance-local bindings: never bind_all, so figures keep their wheel.
        self.after_idle(self.bind_wheel)

    def _resize_canvas(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)
        self._resize_content()

    def _resize_content(self, _event=None):
        requested = self.body.winfo_reqheight()
        self.canvas.itemconfigure(self._window, height=requested)
        height = max(requested, self.canvas.winfo_height())
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), height))
        # Shrinking sections must not leave the viewport below the new bottom.
        top = self.canvas.canvasy(0)
        maximum = max(0, height-self.canvas.winfo_height())
        if top > maximum:
            self.canvas.yview_moveto(maximum/max(height, 1))
        elif top < 0:
            self.canvas.yview_moveto(0)

    def bind_wheel(self):
        def visit(widget):
            if isinstance(widget, (ttk.Treeview, tk.Text, tk.Listbox)):
                return
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, self._wheel)
            for child in widget.winfo_children():
                visit(child)
        visit(self)

    def _wheel(self, event):
        delta, number = getattr(event, "delta", 0), getattr(event, "num", None)
        if number == 4 or delta > 0:
            self.canvas.yview_scroll(-max(1, abs(int(delta))//120), "units")
        elif number == 5 or delta < 0:
            self.canvas.yview_scroll(max(1, abs(int(delta))//120), "units")
        return "break"


class FrameScheduler:
    """At most one pending frame; the callback reads the latest orientation."""

    def __init__(self, owner, callback, interval=25):
        self.owner, self.callback, self.interval = owner, callback, interval
        self.job = None
        owner.bind("<Destroy>", self._destroy, add="+")

    def request(self):
        if self.job is None:
            self.job = self.owner.after(self.interval, self._run)

    def _run(self):
        self.job = None
        self.callback()

    def cancel(self):
        if self.job is not None:
            self.owner.after_cancel(self.job)
            self.job = None

    def _destroy(self, event):
        if event.widget is self.owner:
            self.cancel()
