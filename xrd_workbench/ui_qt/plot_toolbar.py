"""Matplotlib toolbar with icons that follow live Qt palette changes."""

import os

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..localization import tr


class PlotToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent=None, *, figure_provider=None):
        self.figure_provider = figure_provider
        super().__init__(canvas, parent)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            actions = getattr(self, '_actions', {})
            for _text, _tooltip, image, callback in self.toolitems:
                if image and callback in actions:
                    actions[callback].setIcon(self._icon(image + '.png'))

    def save_figure(self, *args):
        if self.figure_provider is None:
            return super().save_figure(*args)
        # Keep Matplotlib's available formats and save-directory preference.
        filetypes = self.canvas.get_supported_filetypes_grouped()
        default = self.canvas.get_default_filetype()
        startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        filters, selected = [], None
        for name, extensions in sorted(filetypes.items()):
            item = f"{name} ({' '.join('*.' + extension for extension in extensions)})"
            filters.append(item)
            if default in extensions:
                selected = item
        path, _filter = QFileDialog.getSaveFileName(
            self, tr('text.save'), os.path.join(startpath, self.canvas.get_default_filename()),
            ';;'.join(filters), selected or '')
        if path:
            if startpath:
                mpl.rcParams['savefig.directory'] = os.path.dirname(path)
            try:
                self.figure_provider().savefig(path)
            except Exception as error:
                QMessageBox.critical(self, tr('text.save_error'), str(error))
        return path
