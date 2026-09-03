import tkinter as tk
from tkinter import ttk
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import json
import sys
import copy
from pathlib import Path

try:
    from .i18n import (
        apply_language,
        filedialog,
        localised,
        messagebox,
        translate_text,
    )
    from .models.data_errors import XRDDataError
    from .models.correction import CorrectionRequest
    from .models.viewer import positive_data_x_limits as _positive_data_x_limits
    from .services.peak_fitting import fit_gaussian_peak as _fit_gaussian_peak
    from .services.correction import apply_correction
    from .services.reference_peaks import read_reference_peaks, write_reference_peaks
    from .xrd_io import Scan1D, clone_scan, read_scan_file
except ImportError:
    from i18n import apply_language, filedialog, localised, messagebox, translate_text
    from models.data_errors import XRDDataError
    from models.correction import CorrectionRequest
    from models.viewer import positive_data_x_limits as _positive_data_x_limits
    from services.peak_fitting import fit_gaussian_peak as _fit_gaussian_peak
    from services.correction import apply_correction
    from services.reference_peaks import read_reference_peaks, write_reference_peaks
    from xrd_io import Scan1D, clone_scan, read_scan_file

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


NS = {'xrd': 'http://www.xrdml.com/XRDMeasurement/2.2'}
ET.register_namespace('', NS['xrd'])


def reference_peak_path() -> Path:
    """Return the shared editable peak database beside the app executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "pivo.json"
    return Path(__file__).resolve().parent / "pivo.json"


def load_reference_peaks() -> dict[str, float]:
    return read_reference_peaks(reference_peak_path())


def save_reference_peaks(data: dict[str, float]) -> None:
    write_reference_peaks(reference_peak_path(), data)


def fit_gaussian_peak(
    coordinates: np.ndarray,
    intensities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compatibility adapter translating shared peak-fit errors."""
    try:
        return _fit_gaussian_peak(coordinates, intensities)
    except XRDDataError as exc:
        messages = {
            "peak_fit_scipy": localised(
                "SciPy is required for peak fitting.",
                "SciPy est requis pour l’ajustement du pic.",
                "Для аппроксимации пика требуется SciPy.",
            ),
            "peak_fit_points": localised(
                "Select at least seven data points around the peak.",
                "Sélectionnez au moins sept points autour du pic.",
                "Выберите вокруг пика не менее семи точек.",
            ),
            "peak_fit_flat": localised(
                "The selected region contains no measurable peak.",
                "La zone sélectionnée ne contient aucun pic mesurable.",
                "В выбранной области нет измеримого пика.",
            ),
        }
        error_type = RuntimeError if exc.code == "peak_fit_scipy" else ValueError
        raise error_type(messages.get(exc.code, str(exc))) from exc


def _local_name(tag):
    return tag.rsplit('}', 1)[-1]


def _first_local(parent, name):
    return next((node for node in parent.iter() if _local_name(node.tag) == name), None)


def _all_local(parent, name):
    return [node for node in parent.iter() if _local_name(node.tag) == name]


class CustomInputDialog(tk.Toplevel):
    def __init__(self, parent, click_x, click_y, db_data):
        super().__init__(parent)
        self.title(localised("Align peak", "Aligner le pic", "Совмещение пика"))
        self.geometry("300x250")
        self.resizable(False, False)

        self.result = None
        self.db_data = db_data

        self.transient(parent.winfo_toplevel())
        self.grab_set()

        tk.Label(
            self,
            text=localised(
                f"Selected point: {click_x:.6f}°\nIntensity: {click_y:.2f}",
                f"Point sélectionné : {click_x:.6f}°\nIntensité : {click_y:.2f}",
                f"Выбранная точка: {click_x:.6f}°\n"
                f"Интенсивность: {click_y:.2f}",
            ),
            font=("Arial", 10, "bold")
        ).pack(pady=10)

        tk.Label(self, text="Select reference from DB:").pack()

        self.combo_var = tk.StringVar()
        self.combo = ttk.Combobox(
            self,
            textvariable=self.combo_var,
            state="readonly",
            width=25
        )
        self.combo['values'] = list(self.db_data.keys())
        self.combo.pack(pady=5)
        self.combo.bind("<<ComboboxSelected>>", self.on_combo_select)

        tk.Label(self, text="Or enter TRUE 2Theta value:").pack(pady=(10, 0))

        self.entry = tk.Entry(self, justify="center", width=15)
        self.entry.pack(pady=5)
        self.entry.focus_set()

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="OK",
            command=self.on_ok,
            width=8
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Cancel",
            command=self.on_cancel,
            width=8
        ).pack(side=tk.LEFT, padx=5)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.on_cancel())

        self.update_idletasks()

        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
        apply_language(self)

        self.wait_window(self)

    def on_combo_select(self, event):
        selection = self.combo_var.get()

        if selection in self.db_data:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, str(self.db_data[selection]))

    def on_ok(self):
        self.result = self.entry.get()
        self.destroy()

    def on_cancel(self):
        self.destroy()


class ScanSelectionDialog(tk.Toplevel):
    def __init__(self, parent, scans):
        super().__init__(parent)
        self.result = None
        self.scans = scans
        self.title(
            localised(
                "Select RAW range",
                "Sélectionner la plage RAW",
                "Выбор диапазона RAW",
            )
        )
        self.resizable(False, False)
        self.transient(parent.winfo_toplevel())
        self.grab_set()

        ttk.Label(
            self,
            text=localised(
                "The file contains several 2θ ranges:",
                "Le fichier contient plusieurs plages 2θ :",
                "Файл содержит несколько диапазонов 2θ:",
            ),
        ).pack(anchor="w", padx=12, pady=(12, 5))
        self.combo = ttk.Combobox(
            self,
            state="readonly",
            width=52,
            values=[scan.name for scan in scans],
        )
        self.combo.current(0)
        self.combo.pack(fill="x", padx=12)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=12, pady=12)
        ttk.Button(
            buttons,
            text=localised("Open", "Ouvrir", "Открыть"),
            command=self.accept,
        ).pack(side="right")
        ttk.Button(
            buttons,
            text=localised("Cancel", "Annuler", "Отмена"),
            command=self.destroy,
        ).pack(side="right", padx=(0, 6))
        apply_language(self)
        self.wait_window(self)

    def accept(self):
        index = self.combo.current()
        if index >= 0:
            self.result = self.scans[index]
        self.destroy()


class XRDShiftApp:
    def __init__(
        self,
        root,
        on_saved=None,
        on_send_scan=None,
        on_send_comparison=None,
        on_commit_scan=None,
        on_loaded=None,
    ):
        self.root = root
        self.on_saved = on_saved
        self.on_send_scan = on_send_scan
        self.on_send_comparison = on_send_comparison
        self.on_commit_scan = on_commit_scan
        self.on_loaded = on_loaded
        self.loaded_name = None
        self.loaded_axis_name = "2Theta"
        self.loaded_scan_template = None
        self.xrdml_scan_index = 1
        self.base_shift = 0.0
        if isinstance(self.root, (tk.Tk, tk.Toplevel)):
            self.root.title(
                localised(
                    "XRD zero-shift correction",
                    "Correction du décalage zéro XRD",
                    "Коррекция нулевого сдвига XRD",
                )
            )
            self.root.geometry("1100x750")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            base_dir = Path.cwd()

        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).parent

        self.db_file = base_dir / "pivo.json"
        self.db_data = self.load_pivo()

        self.filepath = None
        self.tree = None
        self.xml_root = None

        self.twotheta = None
        self.counts = None

        self.highlight_point = None

        self.press_xy = None

        self.rect_selector = None
        self.fit_active = False
        self.fit_line = None
        self.fit_center_line = None
        self.fit_selected_points = None

        self.shift_omega_var = tk.BooleanVar(value=True)
        self.result_mode_var = tk.StringVar(value="add")
        self.save_result_var = tk.BooleanVar(value=False)

        self.setup_ui()
        apply_language(self.root)

    def on_closing(self):
        try:
            if self.rect_selector is not None:
                self.rect_selector.set_active(False)
                self.rect_selector = None
        except Exception:
            pass

        try:
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        except Exception:
            pass

        if isinstance(self.root, (tk.Tk, tk.Toplevel)):
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    def load_pivo(self):
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}

        return {}

    def save_pivo(self, new_data):
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

            self.db_data = new_data

        except Exception as e:
            messagebox.showerror(
                "Database Error",
                f"Failed to save DB at {self.db_file}:\n{e}"
            )

    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = tk.Button(
            control_frame,
            text=localised(
                "1. Open data",
                "1. Ouvrir des données",
                "1. Открыть данные",
            ),
            command=self.load_file
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)

        tk.Label(control_frame, text="Сдвиг оси X, °:").pack(side=tk.LEFT, padx=5)

        self.shift_entry = tk.Entry(control_frame, width=12)
        self.shift_entry.pack(side=tk.LEFT, padx=5)
        self.shift_entry.insert(0, "0.0")

        self.btn_save_xrdml = tk.Button(
            control_frame,
            text="2а. Сохранить XRDML",
            command=self.save_xrdml,
            state=tk.DISABLED
        )
        self.btn_save_xrdml.pack(side=tk.LEFT, padx=5)

        self.btn_save_xy = tk.Button(
            control_frame,
            text="2б. Сохранить XY",
            command=self.save_xy,
            state=tk.DISABLED
        )
        self.btn_save_xy.pack(side=tk.LEFT, padx=5)

        action_frame = tk.Frame(self.root)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        result_frame = tk.Frame(action_frame)
        result_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_send_scan = tk.Button(
            result_frame,
            text="Применить результат",
            command=self.apply_result,
            state=tk.DISABLED,
        )
        self.btn_send_scan.pack(side=tk.LEFT, padx=5)

        self.add_result_radio = tk.Radiobutton(
            result_frame,
            text="Добавить новый",
            variable=self.result_mode_var,
            value="add",
        )
        self.add_result_radio.pack(side=tk.LEFT, padx=(8, 2))
        self.replace_result_radio = tk.Radiobutton(
            result_frame,
            text="Заменить исходный",
            variable=self.result_mode_var,
            value="replace",
        )
        self.replace_result_radio.pack(side=tk.LEFT, padx=2)
        self.save_result_check = tk.Checkbutton(
            result_frame,
            text="Сохранить в файл",
            variable=self.save_result_var,
        )
        self.save_result_check.pack(side=tk.LEFT, padx=(8, 2))

        self.btn_send_comparison = tk.Button(
            result_frame,
            text="Отправить в сравнение",
            command=self.send_to_comparison,
            state=tk.DISABLED,
        )
        if self.on_send_comparison is not None:
            self.btn_send_comparison.pack(side=tk.LEFT, padx=5)

        tools_frame = tk.Frame(action_frame)
        tools_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        self.btn_fit_rect = tk.Button(
            tools_frame,
            text="Аппроксимация",
            command=self.activate_rectangle_fit,
            state=tk.DISABLED
        )
        self.btn_fit_rect.pack(side=tk.LEFT, padx=5)

        self.shift_omega_check = tk.Checkbutton(
            tools_frame,
            text="Сдвигать Omega на 1/2",
            variable=self.shift_omega_var
        )
        self.shift_omega_check.pack(side=tk.LEFT, padx=5)

        self.btn_db = tk.Button(
            tools_frame,
            text="Опорные пики",
            command=self.open_db_window
        )
        self.btn_db.pack(side=tk.LEFT, padx=10)

        tk.Label(
            tools_frame,
            text="Увеличьте пик, нажмите «Аппроксимация» и выделите полезные точки.",
            fg="gray",
            wraplength=360,
            justify=tk.LEFT,
        ).pack(side=tk.RIGHT, padx=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def open_db_window(self):
        db_win = tk.Toplevel(self.root)
        db_win.title("База опорных пиков")
        db_win.geometry("500x400")
        db_win.transient(self.root.winfo_toplevel())
        db_win.grab_set()

        columns = ('Name', 'Value')

        tree = ttk.Treeview(db_win, columns=columns, show='headings')
        tree.heading('Name', text='Подложка / название пика')
        tree.heading('Value', text='2θ')
        tree.column('Name', width=250)
        tree.column('Value', width=150, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for name, val in self.db_data.items():
            tree.insert('', 'end', values=(name, val))

        input_frame = tk.Frame(db_win)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(input_frame, text="Название:").pack(side=tk.LEFT)

        name_entry = tk.Entry(input_frame, width=20)
        name_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(input_frame, text="2θ:").pack(side=tk.LEFT)

        val_entry = tk.Entry(input_frame, width=12)
        val_entry.pack(side=tk.LEFT, padx=5)

        def add_entry():
            name = name_entry.get().strip()
            val_str = val_entry.get().replace(',', '.').strip()

            if not name or not val_str:
                return

            try:
                val = float(val_str)
                tree.insert('', 'end', values=(name, val))

                name_entry.delete(0, tk.END)
                val_entry.delete(0, tk.END)

            except ValueError:
                messagebox.showerror("Error", "Invalid 2Theta value", parent=db_win)

        tk.Button(
            input_frame,
            text="Добавить",
            command=add_entry
        ).pack(side=tk.LEFT, padx=5)

        def delete_selected():
            for item in tree.selection():
                tree.delete(item)

        btn_frame = tk.Frame(db_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(
            btn_frame,
            text="Удалить выбранное",
            command=delete_selected
        ).pack(side=tk.LEFT)

        def save_and_close():
            new_data = {}

            for item in tree.get_children():
                name, val = tree.item(item, 'values')
                new_data[name] = float(val)

            self.save_pivo(new_data)
            db_win.destroy()

        tk.Button(
            btn_frame,
            text="Сохранить",
            command=save_and_close
        ).pack(side=tk.RIGHT)
        apply_language(db_win)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("XRD data", "*.xrdml *.xml *.raw *.xy *.txt *.dat *.csv"),
                ("XRDML", "*.xrdml *.xml"),
                ("Bruker RAW", "*.raw"),
                ("XY data", "*.xy *.txt *.dat *.csv"),
                ("Все файлы", "*.*")
            ]
        )

        if not filepath:
            return

        try:
            self.load_path(filepath)
        except Exception as e:
            messagebox.showerror(
                "Ошибка",
                localised(
                    f"Could not read the file:\n{e}",
                    f"Impossible de lire le fichier :\n{e}",
                    f"Не удалось прочитать файл:\n{e}",
                ),
            )

    def load_path(self, filepath):
        self.filepath = str(filepath)
        self.xrdml_scan_index = 1
        self.base_shift = 0.0
        self.loaded_axis_name = "2Theta"
        self.loaded_scan_template = None
        suffix = Path(self.filepath).suffix.lower()
        if suffix not in {".xrdml", ".xml"}:
            scans = read_scan_file(self.filepath)
            if len(scans) == 1:
                selected = scans[0]
            else:
                dialog = ScanSelectionDialog(self.root, scans)
                selected = dialog.result
                if selected is None:
                    return
            self.load_scan(selected)
            return
        try:
            self.tree = ET.parse(self.filepath)
            self.xml_root = self.tree.getroot()
            self.loaded_name = Path(self.filepath).stem
            self.plot_data()
            if self.twotheta is not None and self.counts is not None:
                self.loaded_scan_template = Scan1D(
                    name=self.loaded_name,
                    x=self.twotheta.copy(),
                    y=self.counts.copy(),
                    source=Path(self.filepath),
                    axis_name="2Theta",
                )
            self.btn_save_xrdml.config(state=tk.NORMAL)
            self.btn_save_xy.config(state=tk.NORMAL)
            self.btn_fit_rect.config(state=tk.NORMAL)
            self._update_transfer_buttons()
            if self.loaded_scan_template is not None and self.on_loaded is not None:
                self.on_loaded(self.loaded_scan_template)
        except Exception:
            self.tree = None
            self.xml_root = None
            raise

    def load_scan(self, scan: Scan1D):
        """Load a curve passed from the 2θ page without reopening its file."""
        self.filepath = str(scan.source)
        self.loaded_name = scan.name
        self.loaded_axis_name = scan.axis_name
        self.loaded_scan_template = clone_scan(scan)
        self.xrdml_scan_index = int(scan.metadata.get("scan_index", 1))
        self.base_shift = float(scan.metadata.get("shift", 0.0))
        self.twotheta = np.asarray(scan.x, dtype=float).copy()
        self.counts = np.asarray(scan.y, dtype=float).copy()
        self.tree = None
        self.xml_root = None

        if scan.source.suffix.lower() in {".xrdml", ".xml"}:
            try:
                self.tree = ET.parse(scan.source)
                self.xml_root = self.tree.getroot()
            except (OSError, ET.ParseError):
                self.tree = None
                self.xml_root = None

        self.shift_entry.delete(0, tk.END)
        self.shift_entry.insert(0, "0.0")
        self.plot_current_data()
        is_two_theta = (
            self.loaded_axis_name.replace(" ", "").replace("-", "").lower()
            in {"2theta", "twotheta", "2θ"}
        )
        self.btn_save_xrdml.config(
            state=tk.NORMAL if self.tree is not None and is_two_theta else tk.DISABLED
        )
        self.btn_save_xy.config(state=tk.NORMAL)
        self.btn_fit_rect.config(state=tk.NORMAL)
        self._update_transfer_buttons()
        if self.on_loaded is not None:
            self.on_loaded(self.loaded_scan_template)

    def get_data_from_xml(self):
        if self.xml_root is None:
            raise ValueError(
                localised(
                    "No XML file is loaded.",
                    "Aucun fichier XML n’est chargé.",
                    "XML-файл не загружен.",
                )
            )

        blocks = _all_local(self.xml_root, 'dataPoints')
        index = self.xrdml_scan_index - 1
        data_points = blocks[index] if 0 <= index < len(blocks) else None

        if data_points is None:
            raise ValueError(
                localised(
                    "The <dataPoints> tag was not found.",
                    "La balise <dataPoints> est introuvable.",
                    "Тег <dataPoints> не найден.",
                )
            )

        counts_node = _first_local(data_points, 'counts')
        if counts_node is None:
            counts_node = _first_local(data_points, 'intensities')

        if counts_node is None or counts_node.text is None:
            raise ValueError(
                localised(
                    "The <counts> tag was not found or is empty.",
                    "La balise <counts> est introuvable ou vide.",
                    "Тег <counts> не найден либо пуст.",
                )
            )

        counts = np.array([float(x) for x in counts_node.text.split()])

        positions = [
            node for node in data_points.iter()
            if _local_name(node.tag) == 'positions'
        ]

        start_theta = None
        end_theta = None

        for pos in positions:
            if pos.attrib.get('axis') == '2Theta':
                start_node = _first_local(pos, 'startPosition')
                end_node = _first_local(pos, 'endPosition')

                if start_node is None or end_node is None:
                    raise ValueError(
                        localised(
                            "2Theta positions were found, but the start or end "
                            "position is missing.",
                            "Les positions 2Theta ont été trouvées, mais la position "
                            "de début ou de fin manque.",
                            "Позиции 2Theta найдены, но отсутствует начальная либо "
                            "конечная позиция.",
                        )
                    )

                start_theta = float(start_node.text)
                end_theta = float(end_node.text)
                break

        if start_theta is None or end_theta is None:
            raise ValueError(
                localised(
                    "The 2Theta axis was not found.",
                    "L’axe 2Theta est introuvable.",
                    "Ось 2Theta не найдена.",
                )
            )

        twotheta = np.linspace(start_theta, end_theta, len(counts))

        return twotheta, counts

    def clear_fit_artists(self):
        for artist_name in [
            'fit_line',
            'fit_center_line',
            'fit_selected_points',
            'highlight_point'
        ]:
            artist = getattr(self, artist_name, None)

            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass

                setattr(self, artist_name, None)

    def plot_data(self):
        try:
            self.twotheta, self.counts = self.get_data_from_xml()
            self.plot_current_data()
        except Exception as e:
            messagebox.showerror("Parsing Error", str(e))

    def plot_current_data(self):
        self.ax.clear()
        self.clear_fit_artists()
        if self.twotheta is None or self.counts is None:
            return

        plot_counts = np.where(self.counts > 0, self.counts, np.nan)

        self.ax.plot(
            self.twotheta,
            plot_counts,
            color='b',
            linewidth=1,
            picker=True
        )

        self.ax.set_yscale('log')

        file_name = self.loaded_name or (
            Path(self.filepath).name if self.filepath else "Файл не открыт"
        )

        self.ax.set_title(
            localised(
                f"Diffraction pattern: {file_name}",
                f"Diffractogramme : {file_name}",
                f"Рентгенограмма: {file_name}",
            )
        )
        axis_labels = {
            "2theta": "2θ",
            "twotheta": "2θ",
            "theta": "θ",
            "omega": "ω",
            "chi": "χ",
            "phi": "φ",
        }
        axis_key = self.loaded_axis_name.replace(" ", "").replace("-", "").lower()
        axis_label = axis_labels.get(axis_key, self.loaded_axis_name)
        degree_axes = {"2theta", "twotheta", "theta", "omega", "chi", "phi"}
        self.ax.set_xlabel(
            f"{axis_label}, °" if axis_key in degree_axes else axis_label
        )
        self.ax.set_ylabel(
            localised(
                "Intensity, counts – logarithmic scale",
                "Intensité, coups – échelle logarithmique",
                "Интенсивность, отсчёты – логарифмическая шкала",
            )
        )
        self.ax.grid(True, which="both", ls="--", alpha=0.5)
        x_limits = _positive_data_x_limits(self.twotheta, self.counts)
        if x_limits is not None:
            self.ax.set_xlim(*x_limits)

        self.canvas.draw()

    def on_press(self, event):
        if self.fit_active:
            return

        if event.inaxes and event.button == 1:
            self.press_xy = (event.x, event.y)

    def on_release(self, event):
        if self.fit_active:
            return

        if (
            not event.inaxes or
            self.press_xy is None or
            self.twotheta is None or
            event.button != 1
        ):
            return

        dx = abs(event.x - self.press_xy[0])
        dy = abs(event.y - self.press_xy[1])

        self.press_xy = None

        if dx > 5 or dy > 5:
            return

        idx = (np.abs(self.twotheta - event.xdata)).argmin()

        click_x = self.twotheta[idx]
        click_y = self.counts[idx]
        view_xlim = self.ax.get_xlim()
        view_ylim = self.ax.get_ylim()

        self.clear_fit_artists()

        self.highlight_point = self.ax.scatter(
            click_x,
            click_y,
            color='red',
            s=50,
            zorder=5
        )
        self.ax.set_xlim(*view_xlim)
        self.ax.set_ylim(*view_ylim)

        self.canvas.draw()

        dialog = CustomInputDialog(self.root, click_x, click_y, self.db_data)
        target_str = dialog.result

        if target_str is not None and target_str.strip() != "":
            try:
                target_val = float(target_str.replace(',', '.'))
                shift = target_val - click_x

                self.shift_entry.delete(0, tk.END)
                self.shift_entry.insert(0, f"{shift:.6f}")

            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number!")

                if self.highlight_point is not None:
                    self.highlight_point.remove()
                    self.highlight_point = None

                self.canvas.draw()

        else:
            if self.highlight_point is not None:
                self.highlight_point.remove()
                self.highlight_point = None

            self.canvas.draw()

    def deactivate_toolbar_tools(self):
        try:
            mode = str(self.toolbar.mode).lower()

            if "zoom" in mode:
                self.toolbar.zoom()
            elif "pan" in mode:
                self.toolbar.pan()

        except Exception:
            pass

    def activate_rectangle_fit(self):
        if self.twotheta is None or self.counts is None:
            messagebox.showerror("Error", "Load XRDML file first!")
            return

        if curve_fit is None:
            messagebox.showerror(
                "Missing dependency",
                "scipy is required for Gaussian fitting.\nInstall it with:\npip install scipy"
            )
            return

        self.deactivate_toolbar_tools()

        self.fit_active = True
        self.press_xy = None

        self.btn_fit_rect.config(text=translate_text("Выделите область…"))

        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
            self.rect_selector = None

        self.rect_selector = RectangleSelector(
            self.ax,
            self.on_rectangle_selected,
            useblit=True,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords='pixels',
            interactive=False,
            props=dict(alpha=0.2, fill=True)
        )

        self.canvas.draw()

    def on_rectangle_selected(self, eclick, erelease):
        self.fit_active = False
        self.btn_fit_rect.config(text=translate_text("Аппроксимация"))

        if self.rect_selector is not None:
            self.rect_selector.set_active(False)

        if eclick.xdata is None or erelease.xdata is None:
            return

        if eclick.ydata is None or erelease.ydata is None:
            return

        xmin = min(eclick.xdata, erelease.xdata)
        xmax = max(eclick.xdata, erelease.xdata)
        ymin = min(eclick.ydata, erelease.ydata)
        ymax = max(eclick.ydata, erelease.ydata)
        view_xlim = self.ax.get_xlim()
        view_ylim = self.ax.get_ylim()

        mask = (
            (self.twotheta >= xmin) &
            (self.twotheta <= xmax) &
            (self.counts >= ymin) &
            (self.counts <= ymax)
        )

        x = self.twotheta[mask]
        y = self.counts[mask]

        if len(x) < 7:
            messagebox.showerror(
                "Fit Error",
                "Too few data points inside the selected rectangle."
            )
            return

        if np.max(y) <= 0:
            messagebox.showerror(
                "Fit Error",
                "Selected region has no positive intensity."
            )
            return

        x_min_data = float(np.min(x))
        x_max_data = float(np.max(x))
        x_mid = 0.5 * (x_min_data + x_max_data)

        def gaussian_with_linear_background(x_values, c0, c1, amplitude, center, sigma):
            return (
                c0 +
                c1 * (x_values - x_mid) +
                amplitude * np.exp(-0.5 * ((x_values - center) / sigma) ** 2)
            )

        try:
            c0_guess = float(np.min(y))
            c1_guess = 0.0
            amplitude_guess = float(np.max(y) - np.min(y))
            center_guess = float(x[np.argmax(y)])
            sigma_guess = max(float((x_max_data - x_min_data) / 6.0), 1e-6)

            p0 = [
                c0_guess,
                c1_guess,
                amplitude_guess,
                center_guess,
                sigma_guess
            ]

            lower_bounds = [
                -np.inf,
                -np.inf,
                0.0,
                x_min_data,
                1e-8
            ]

            upper_bounds = [
                np.inf,
                np.inf,
                np.inf,
                x_max_data,
                max(x_max_data - x_min_data, 1e-8)
            ]

            popt, pcov = curve_fit(
                gaussian_with_linear_background,
                x,
                y,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000
            )

            c0, c1, amplitude, center, sigma = popt

            fitted_center = float(center)

            fitted_intensity = float(
                gaussian_with_linear_background(
                    np.array([fitted_center]),
                    *popt
                )[0]
            )

            self.clear_fit_artists()

            x_fit = np.linspace(x_min_data, x_max_data, 500)
            y_fit = gaussian_with_linear_background(x_fit, *popt)
            y_fit_plot = np.where(y_fit > 0, y_fit, np.nan)

            self.fit_selected_points = self.ax.scatter(
                x,
                y,
                color='green',
                s=20,
                zorder=5
            )

            self.fit_line, = self.ax.plot(
                x_fit,
                y_fit_plot,
                color='orange',
                linewidth=2,
                label=localised(
                    "Gaussian fit",
                    "Ajustement gaussien",
                    "Гауссова аппроксимация",
                )
            )

            self.fit_center_line = self.ax.axvline(
                fitted_center,
                color='red',
                linestyle='--',
                linewidth=1
            )

            self.highlight_point = self.ax.scatter(
                fitted_center,
                fitted_intensity,
                color='red',
                s=60,
                zorder=6
            )
            self.ax.set_xlim(*view_xlim)
            self.ax.set_ylim(*view_ylim)

            self.canvas.draw()

            dialog = CustomInputDialog(
                self.root,
                fitted_center,
                fitted_intensity,
                self.db_data
            )

            target_str = dialog.result

            if target_str is not None and target_str.strip() != "":
                try:
                    target_val = float(target_str.replace(',', '.'))
                    shift = target_val - fitted_center

                    self.shift_entry.delete(0, tk.END)
                    self.shift_entry.insert(0, f"{shift:.6f}")

                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number!")

        except Exception as e:
            messagebox.showerror(
                "Ошибка аппроксимации",
                localised(
                    f"Gaussian fitting failed:\n{e}",
                    f"Échec de l’ajustement gaussien :\n{e}",
                    f"Не удалось выполнить гауссову аппроксимацию:\n{e}",
                ),
            )

    def _update_transfer_buttons(self):
        has_data = self.twotheta is not None and self.counts is not None
        self.btn_send_scan.config(
            state=(
                tk.NORMAL
                if has_data and (
                    self.on_commit_scan is not None or self.on_send_scan is not None
                )
                else tk.DISABLED
            )
        )
        self.btn_send_comparison.config(
            state=(
                tk.NORMAL
                if has_data and self.on_send_comparison is not None
                else tk.DISABLED
            )
        )

    def _corrected_scan(self):
        if self.twotheta is None or self.counts is None:
            return None
        try:
            shift_val = float(self.shift_entry.get().replace(',', '.'))
        except ValueError:
            messagebox.showerror(
                "Ошибка",
                localised(
                    "Enter a valid shift value.",
                    "Saisissez une valeur de décalage valide.",
                    "Введите корректное значение сдвига.",
                ),
            )
            return None

        source = Path(self.filepath) if self.filepath else Path("corrected.xy")
        base_name = self.loaded_name or source.stem
        if self.loaded_scan_template is not None:
            template = self.loaded_scan_template
        else:
            template = Scan1D(
                name=base_name,
                x=np.asarray(self.twotheta, dtype=float).copy(),
                y=np.asarray(self.counts, dtype=float).copy(),
                source=source,
                axis_name=self.loaded_axis_name,
                metadata={"shift": self.base_shift},
            )
        corrected = apply_correction(
            template,
            CorrectionRequest(x_shift=shift_val, shift_omega_half=False),
        )
        corrected.metadata["format"] = "correction preview"
        return corrected

    def send_to_twotheta(self):
        if self.on_send_scan is None:
            return
        scan = self._corrected_scan()
        if scan is not None:
            self.on_send_scan(scan)

    def apply_result(self):
        scan = self._corrected_scan()
        if scan is None:
            return
        if self.on_commit_scan is not None:
            self.on_commit_scan(scan, self.result_mode_var.get())
        elif self.on_send_scan is not None:
            self.on_send_scan(scan)
        if self.save_result_var.get():
            if self.tree is not None and self.xml_root is not None:
                self.save_xrdml(notify_saved=False)
            else:
                self.save_xy(notify_saved=False)

    def send_to_comparison(self):
        if self.on_send_comparison is None:
            return
        scan = self._corrected_scan()
        if scan is not None:
            self.on_send_comparison(scan)

    def save_xy(self, notify_saved=True):
        if self.twotheta is None or self.counts is None:
            messagebox.showerror("Error", "No data loaded!")
            return

        try:
            shift_text = self.shift_entry.get().replace(',', '.')
            shift_val = float(shift_text)

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the shift!")
            return

        default_name = (
            f"{Path(self.filepath).stem} shifted.xy"
            if self.filepath
            else "shifted.xy"
        )

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xy",
            filetypes=[
                ("XY Data files", "*.xy"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialfile=default_name
        )

        if not save_path:
            return

        try:
            shifted_twotheta = self.twotheta + shift_val
            data_to_save = np.column_stack((shifted_twotheta, self.counts))

            np.savetxt(
                save_path,
                data_to_save,
                fmt=['%.6f', '%g'],
                delimiter=' '
            )

            messagebox.showinfo("Готово", "Файл XY сохранён.")
            if notify_saved and self.on_saved is not None:
                self.on_saved(save_path)

        except Exception as e:
            messagebox.showerror(
                "Ошибка экспорта",
                localised(
                    f"Failed to save the XY file:\n{e}",
                    f"Impossible d’enregistrer le fichier XY :\n{e}",
                    f"Не удалось сохранить файл XY:\n{e}",
                ),
            )

    def save_xrdml(self, notify_saved=True):
        if self.tree is None or self.xml_root is None:
            messagebox.showerror("Error", "No XRDML file loaded!")
            return

        try:
            shift_text = self.shift_entry.get().replace(',', '.')
            shift_val = float(shift_text)

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the shift!")
            return

        initial_name = (
            f"{Path(self.filepath).stem} shifted{Path(self.filepath).suffix}"
            if self.filepath
            else "shifted.xrdml"
        )

        save_path = filedialog.asksaveasfilename(
            defaultextension=".xrdml",
            filetypes=[("XRDML files", "*.xrdml")],
            initialfile=initial_name
        )

        if not save_path:
            return

        try:
            total_shift = self.base_shift + shift_val
            tree_copy = copy.deepcopy(self.tree)
            root_copy = tree_copy.getroot()

            blocks = _all_local(root_copy, 'dataPoints')
            index = self.xrdml_scan_index - 1
            data_points = blocks[index] if 0 <= index < len(blocks) else None

            if data_points is None:
                raise ValueError(
                    localised(
                        "The <dataPoints> tag was not found.",
                        "La balise <dataPoints> est introuvable.",
                        "Тег <dataPoints> не найден.",
                    )
                )

            positions = [
                node for node in data_points.iter()
                if _local_name(node.tag) == 'positions'
            ]

            for pos in positions:
                axis = pos.attrib.get('axis')

                start_node = _first_local(pos, 'startPosition')
                end_node = _first_local(pos, 'endPosition')

                if start_node is None or end_node is None:
                    continue

                if axis == '2Theta':
                    start_node.text = f"{(float(start_node.text) + total_shift):.8f}"
                    end_node.text = f"{(float(end_node.text) + total_shift):.8f}"

                elif axis == 'Omega' and self.shift_omega_var.get():
                    start_node.text = f"{(float(start_node.text) + total_shift / 2.0):.8f}"
                    end_node.text = f"{(float(end_node.text) + total_shift / 2.0):.8f}"

            tree_copy.write(save_path, encoding="utf-8", xml_declaration=True)

            messagebox.showinfo("Готово", "Исправленный XRDML сохранён.")
            if notify_saved and self.on_saved is not None:
                self.on_saved(save_path)

            self.filepath = save_path
            self.base_shift = 0.0
            self.tree = ET.parse(save_path)
            self.xml_root = self.tree.getroot()

            self.shift_entry.delete(0, tk.END)
            self.shift_entry.insert(0, "0.0")

            self.plot_data()

        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = XRDShiftApp(root)
    root.mainloop()
