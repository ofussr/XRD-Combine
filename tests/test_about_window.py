from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_qt_shell import QApplication, PYSIDE_AVAILABLE


@unittest.skipUnless(PYSIDE_AVAILABLE, 'PySide6 QtWidgets runtime is unavailable')
class AboutWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtCore import QSettings
        from xrd_workbench.ui_qt.main_window import MainWindow
        from xrd_workbench.ui_qt.theme import ThemeController

        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        self.theme = ThemeController(self.application, self.settings)
        self.window = MainWindow(theme_controller=self.theme)
        self.window.show()
        self.application.processEvents()
        self.confirmation = patch.object(self.window, '_needs_close_confirmation',
                                         return_value=False)
        self.confirmation.start()

    def tearDown(self):
        from PySide6.QtCore import QEvent

        self.window.close()
        self.window.deleteLater()
        self.theme.set_mode('system', persist=False)
        self.theme.deleteLater()
        self.application.sendPostedEvents(None, QEvent.DeferredDelete)
        self.application.processEvents()
        self.confirmation.stop()
        self.folder.cleanup()

    def activate(self, check_other_clicks=False):
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QLabel, QMessageBox

        errors = []

        def click():
            dialog = self.application.activeModalWidget()
            try:
                self.assertIsInstance(dialog, QMessageBox)
                labels = dialog.findChildren(QLabel)
                logo = next(label for label in labels if not label.pixmap().isNull())
                if check_other_clicks:
                    self.assertEqual(logo.cursor().shape(), Qt.ArrowCursor)
                    self.assertEqual(logo.toolTip(), '')
                    QTest.mouseClick(logo, Qt.LeftButton)
                    QTest.mouseDClick(logo, Qt.RightButton)
                    text = next(label for label in labels if label.text().startswith('XRD Combine'))
                    QTest.mouseDClick(text, Qt.LeftButton)
                    self.assertTrue(dialog.isVisible())
                    self.assertIsNone(getattr(self.window, '_wc_window', None))
                QTest.mouseDClick(logo, Qt.LeftButton)
            except BaseException as error:
                errors.append(error)
                dialog.reject()

        QTimer.singleShot(0, click)
        self.window.about()
        if errors:
            raise errors[0]
        self.application.processEvents()
        self.assertIsNone(self.application.activeModalWidget())
        return self.window._wc_window

    def test_logo_activation_and_keyboard_after_about_closes(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        window = self.activate(check_other_clicks=True)
        self.assertTrue(window.isVisible())
        view = window.centralWidget()
        QTest.keyClick(view, Qt.Key_Space)
        self.assertGreater(view.s, 0)
        QTest.keyClick(view, Qt.Key_P)
        self.assertTrue(view.h)

    def test_repeated_activation_reuses_window_and_state(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        first = self.activate()
        QTest.keyClick(first.centralWidget(), Qt.Key_Space)
        score = first.centralWidget().s
        second = self.activate()
        self.assertIs(second, first)
        self.assertEqual(second.centralWidget().s, score)

    def test_close_stops_timer_and_delayed_deletion_preserves_new_window(self):
        from PySide6.QtCore import QEvent
        from xrd_workbench.ui_qt.WolfCat import open_wolfcat

        first = open_wolfcat(self.window)
        timer = first.centralWidget().u
        self.assertTrue(timer.isActive())
        first.close()
        self.assertFalse(timer.isActive())
        self.assertIsNone(self.window._wc_window)
        second = open_wolfcat(self.window)
        self.assertIsNot(second, first)
        self.application.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertIs(self.window._wc_window, second)
        self.assertTrue(second.centralWidget().u.isActive())

    def test_main_window_close_stops_child_timer(self):
        from xrd_workbench.ui_qt.WolfCat import open_wolfcat

        child = open_wolfcat(self.window)
        timer = child.centralWidget().u
        self.window.close()
        self.assertFalse(timer.isActive())
        self.assertIsNone(self.window._wc_window)

    def test_data_round_trip_uses_user_directory_and_handles_corruption(self):
        from xrd_workbench.ui_qt import WolfCat

        target = self.root / 'user' / 'data'
        with patch.object(WolfCat.QStandardPaths, 'writableLocation',
                          return_value=str(target)):
            WolfCat._write([('Михаил|test\nname', 250), ('Other', 100)])
            self.assertEqual(WolfCat._read(), [('Михаил/test name', 250), ('Other', 100)])
            data = target / 'WolfCat.dat'
            self.assertTrue(data.is_file())
            data.write_text('not valid hex', encoding='ascii')
            self.assertEqual(WolfCat._read(), [])
        with patch.object(WolfCat.QStandardPaths, 'writableLocation', return_value=str(data)):
            WolfCat._write([('Ignored', 1)])
            self.assertEqual(WolfCat._read(), [])
            self.assertEqual(data.read_text(encoding='ascii'), 'not valid hex')
