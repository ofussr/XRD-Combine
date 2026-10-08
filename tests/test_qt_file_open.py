"""Native source-file drops and queued requests share the project import route."""

import pytest

from qt_test_support import PYSIDE_AVAILABLE
from qt_workbench_harness import ui
from xrd_workbench.models.project import SCAN, STRUCTURES, VIEWER

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason="PySide6 is unavailable")


def source(root, name):
    path = root / name
    path.write_text("10 20\n11 30\n12 25\n", encoding="utf-8")
    return path


def mime(paths):
    from PySide6.QtCore import QMimeData, QUrl
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return data


@pytest.mark.parametrize("target_name", ["window", "tree", "input", "plot"])
def test_native_drop_over_nested_widgets_imports_multiple_sources(ui, target_name):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QLineEdit
    paths = [source(ui.root, "замер с пробелами.XY"), source(ui.root, "second.csv")]
    targets = {"window": ui.window, "tree": ui.window.project_panel.tree.viewport(),
               "input": ui.window.findChild(QLineEdit),
               "plot": ui.viewer.pyqtgraph_plot}
    target = targets[target_name]
    data = mime([*paths, paths[0]])
    enter = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, data,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    ui.app.sendEvent(target, enter)
    assert enter.isAccepted()
    ui.app.processEvents()
    assert ui.window.source_file_drop.hint.isVisible()
    drop = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, data,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    ui.app.sendEvent(target, drop)
    assert drop.isAccepted()
    for _ in range(5):
        ui.app.processEvents()
    documents = ui.store.assigned_documents(VIEWER, kind=SCAN)
    assert [doc.source for doc in documents] == paths
    assert not ui.window.source_file_drop.hint.isVisible()


@pytest.mark.parametrize("invalid", ["session.xrdproject", "tab.xrdtab", "image.png", "folder", "remote"])
def test_drop_rejects_sessions_directories_and_remote_urls(ui, invalid):
    from PySide6.QtCore import QPoint, QUrl, Qt
    from PySide6.QtGui import QDragEnterEvent
    from PySide6.QtWidgets import QLineEdit
    path = ui.root / invalid
    if invalid == "folder":
        path.mkdir()
    else:
        path.write_text("{}")
    data = mime([path])
    if invalid == "remote":
        data.setUrls([QUrl("https://example.com/measurement.xy")])
    target = ui.window.findChild(QLineEdit)
    old = target.text()
    event = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, data,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    ui.app.sendEvent(target, event)
    assert not event.isAccepted()
    assert target.text() == old
    assert not ui.store.documents


def test_external_queue_waits_for_dialog_and_keeps_destination(ui):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from unittest.mock import patch
    first = source(ui.root, "first.xy")
    second = source(ui.root, "second.xy")
    dialog = QDialog(ui.window)
    dialog.setModal(True)
    dialog.show()
    ui.app.processEvents()
    ui.window.open_external_files([str(first)], activate=False)
    ui.app.processEvents()
    assert not ui.store.documents
    ui.window.sections.setCurrentIndex(1)
    ui.window.open_external_files([str(second)], activate=False)
    dialog.close()
    ui.window._external_import_timer.stop()
    ui.window._import_external_files()
    ui.window._external_import_timer.stop()
    ui.window._import_external_files()
    assert ui.store.source_document(SCAN, first).uid in ui.store.assignments[VIEWER]
    assert ui.store.source_document(SCAN, second).uid not in ui.store.assignments[VIEWER]
    assert not ui.store.assigned_documents(STRUCTURES)
    with patch.object(QMessageBox, "warning") as warning:
        ui.window.import_paths([str(ui.root / "session.xrdproject")])
    assert warning.call_count == 1
    assert len(ui.store.documents) == 2


def test_forwarded_request_restores_minimized_window(ui):
    ui.window.showMinimized()
    ui.app.processEvents()
    assert ui.window.isMinimized()
    ui.window.open_external_files([])
    ui.app.processEvents()
    assert ui.window.isVisible() and not ui.window.isMinimized()


def test_cif_drop_assigns_structure_and_prepares_it_in_background(ui):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from pole_fixtures import CIF_TEXT
    from time import monotonic
    path = ui.root / "структура с пробелами.cif"
    path.write_text(CIF_TEXT, encoding="utf-8")
    ui.window.sections.setCurrentIndex(1)
    data = mime([path])
    enter = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, data,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    ui.app.sendEvent(ui.window, enter)
    drop = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, data,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    ui.app.sendEvent(ui.window, drop)
    ui.app.processEvents()
    documents = ui.store.assigned_documents(STRUCTURES)
    assert len(documents) == 1 and documents[0].source == path
    deadline = monotonic() + 10
    while ui.window.structure_preparation.busy and monotonic() < deadline:
        ui.app.processEvents()
    assert not ui.window.structure_preparation.busy
