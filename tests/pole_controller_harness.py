"""Run the real pole controllers with Agg, without claiming to test Qt widgets.

Only the Qt imports and widget builders are substituted. All controller,
calculation, projection, event and Matplotlib drawing methods come from the
shipped source. Native widget integration is tested separately when Qt exists.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from xrd_workbench.localization import tr

ROOT = Path(__file__).resolve().parents[1] / 'xrd_workbench' / 'ui_qt'


class Widget:
    def __init__(self, *args):
        self.enabled = True
        self.text = ''
        self.values = []
        self.title_source = ''
        self.visible = True

    def setText(self, text): self.text = text
    def setPlainText(self, text): self.text = text
    def setEnabled(self, value): self.enabled = value
    def setVisible(self, value): self.visible = value
    def set_expanded(self, value): self.visible = value
    def set_title(self, value): self.text = value
    def currentText(self): return self.text
    def setCurrentText(self, value): self.text = value
    def clear(self): self.text = ''
    def setFocus(self): pass


class Timer:
    def __init__(self, *args, **kwargs):
        self.timeout = SimpleNamespace(connect=lambda callback: None)
        self.active = False
    def setSingleShot(self, value): pass
    def start(self, *args): self.active = True
    def stop(self): self.active = False
    def request(self): self.active = True
    def cancel(self): self.active = False


class Ui:
    def _init_ui_helpers(self): pass
    def retranslate_controls(self): pass


class StructureView(Widget):
    """Presentation boundary only; native Qt rendering has separate tests."""
    def __init__(self):
        super().__init__()
        self.document = None
        self.display_section = Widget()
        self.canvas = SimpleNamespace(show_basis=True, show_polyhedra=False)
        self.canvas.set_display_options = self._set_display_options

    def _set_display_options(self, **changes):
        for name, value in changes.items():
            setattr(self.canvas, name, value)

    def sync_document(self, document, orientation):
        self.document = document
        self.canvas.orientation = orientation.copy()

    def refresh_atom_styles(self): pass
    def retranslate(self): pass


def configure(widget, **kwargs):
    if 'state' in kwargs: widget.enabled = kwargs['state'] != 'disabled'
    if 'text' in kwargs: widget.text = kwargs['text']
    if 'values' in kwargs: widget.values = kwargs['values']


def fail_dialog(title, message, **kwargs):
    raise AssertionError(f'{title}: {message}')


def load_controller(filename):
    namespace = {'__package__': 'xrd_workbench.ui_qt', '__name__': 'pole_harness_' + filename,
                 'QWidget': Widget, 'PoleUi': Ui, 'FrameScheduler': Timer, 'QTimer': Timer,
                 'configure': configure, 'show_error': fail_dialog, 'show_info': lambda *a, **kw: None}
    value_tree = ast.parse((ROOT / 'pole_widgets.py').read_text())
    value = next(node for node in value_tree.body if isinstance(node, ast.ClassDef) and node.name == 'Value')
    exec(compile(ast.Module(body=[value], type_ignores=[]), 'pole_widgets.py', 'exec'), {'tr': tr}, namespace)
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module.startswith('PySide6') or
        node.module in ('pole_widgets', 'pole_structure', 'plot_toolbar', 'radiation', 'experimental_pole', 'calculated_pole', 'matplotlib.backends.backend_qtagg')))]
    exec(compile(tree, str(ROOT / filename), 'exec'), namespace)
    return namespace


def _calculated_layout(self):
    for name in ('overlay_combo', 'add_overlay_button', 'open_overlay_button',
                 'remove_overlay_button', 'd_colour_radio', 'intensity_colour_radio',
                 'label_leaders_check', 'basis_check', 'drag_help_label', 'center_section',
                 'rotation_section', 'relative_rotation_section', 'overlay_section',
                 'overlay_settings', 'center_prompt', 'range_button', 'center_combo', 'info_text'):
        setattr(self, name, Widget())
    self.figure = Figure(figsize=(7.2, 7.2), dpi=100)
    self.canvas = FigureCanvasAgg(self.figure)
    self.ax = self.figure.add_subplot(111)
    self.structure_viewer = StructureView()
    self.toolbar = SimpleNamespace(mode='')


def _experimental_layout(self):
    for name in ('fill_colour_button', 'save_button', 'reset_zoom_button', 'lower_entry', 'upper_entry'):
        setattr(self, name, Widget())
    self.scale_buttons = [Widget() for _ in range(3)]
    self.angle_entries = {key: Widget() for key in ('first', 'step', 'last')}
    self.figure = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
    self.canvas = FigureCanvasAgg(self.figure)


_calculated = load_controller('calculated_pole.py')
Calculated = _calculated['CalculatedPolePage']
Calculated._build_layout = _calculated_layout
_experimental = load_controller('experimental_pole.py')
Experimental = _experimental['ExperimentalPolePage']
Experimental._build_interface = _experimental_layout


def workspace(store, radiation, file_service):
    namespace = load_controller('pole_figures.py')
    page = namespace['PolesPage'].__new__(namespace['PolesPage'])
    page.store, page.radiation_settings, page.file_service = store, radiation, file_service
    page._raw_token = None
    page._radiation_signature = tuple(radiation.lines())
    page._refreshing = False
    page.radiation_selector = SimpleNamespace(sync_from_settings=lambda: None)
    page.tabs = SimpleNamespace(setCurrentIndex=lambda index: None)
    page.experimental = Experimental(on_open_raw=page.open_raw)
    page.calculated = Calculated(radiation.lines, on_open_cif=page.open_cif,
                                on_add_overlay=page.add_overlay, on_remove_overlay=page.remove_overlay,
                                overlay_documents_provider=page.structure_documents)
    store.subscribe(lambda *_: page.refresh_documents())
    return page
