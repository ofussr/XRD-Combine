"""Real local mailbox/processes, including requests during blocked startup."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from qt_test_support import PYSIDE_AVAILABLE, QApplication

pytestmark = pytest.mark.skipif(not PYSIDE_AVAILABLE, reason="PySide6 is unavailable")
ROOT = Path(__file__).resolve().parents[1]
CLIENT = """
import sys
from PySide6.QtCore import QCoreApplication
from xrd_workbench.ui_qt.single_instance import SingleInstance
app = QCoreApplication([])
instance = SingleInstance(name=sys.argv[1], lock_directory=sys.argv[2])
primary = instance.start_or_forward(sys.argv[3:])
assert not primary
assert 'xrd_workbench.ui_qt.main_window' not in sys.modules
print('forwarded')
"""


@pytest.fixture
def owner(tmp_path):
    from xrd_workbench.ui_qt.single_instance import SingleInstance
    app = QApplication.instance() or QApplication([])
    instance = SingleInstance(name="xrd-test-" + uuid.uuid4().hex, lock_directory=tmp_path)
    assert instance.start_or_forward([])
    yield app, instance, tmp_path
    instance.close()


def client(owner, paths):
    _app, instance, root = owner
    return subprocess.run([sys.executable, "-c", CLIENT, instance.name, str(root),
                           *map(str, paths)], cwd=ROOT, capture_output=True, text=True,
                          timeout=15)


def test_requests_acknowledged_while_main_thread_is_busy_then_delivered(owner):
    app, instance, root = owner
    paths = [root / "данные с пробелами.xy", root / "phase.cif"]
    # No main-thread processEvents: the IPC thread must still acknowledge.
    result = client(owner, paths)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "forwarded"
    app.processEvents()
    received = []
    instance.attach(received.append)
    assert received == [list(map(str, paths))]


def test_secondary_bootstrap_exits_before_main_window_import(owner):
    app, instance, root = owner
    code = """
import importlib.abc, sys
from xrd_workbench.ui_qt import single_instance
real = single_instance.SingleInstance
single_instance.SingleInstance = lambda: real(name=sys.argv[1], lock_directory=sys.argv[2])
class NoMainWindow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'xrd_workbench.ui_qt.main_window':
            raise ImportError('A secondary process must never import MainWindow')
sys.meta_path.insert(0, NoMainWindow())
from xrd_workbench.ui_qt.app import main
raise SystemExit(main(sys.argv[3:]))
"""
    path = root / "измерение.xy"
    result = subprocess.run([sys.executable, "-c", code, instance.name, str(root), str(path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    app.processEvents()
    received = []
    instance.attach(received.append)
    assert received == [[str(path)]]


def test_simultaneous_cold_launches_elect_exactly_one_owner(tmp_path):
    name = "xrd-test-" + uuid.uuid4().hex
    code = """
import json, sys, time
from pathlib import Path
from PySide6.QtCore import QCoreApplication
from xrd_workbench.ui_qt.single_instance import SingleInstance
app = QCoreApplication([])
root = Path(sys.argv[2])
instance = SingleInstance(name=sys.argv[1], lock_directory=root)
path = str(root / sys.argv[3])
primary = instance.start_or_forward([path])
if primary:
    deadline = time.monotonic() + 10
    while not (root / 'forwarded').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    received = []
    instance.attach(received.append)
    app.processEvents()
    print(json.dumps({'primary': True, 'paths': [path] + [p for batch in received for p in batch]}))
    instance.close()
else:
    (root / 'forwarded').write_text('ok')
    print(json.dumps({'primary': False}))
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, name, str(tmp_path),
                                    f"scan-{index}.xy"], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True) for index in range(2)]
    try:
        results = [process.communicate(timeout=15) for process in processes]
        assert all(process.returncode == 0 for process in processes), results
        states = [json.loads(stdout) for stdout, _stderr in results]
        assert sum(state["primary"] for state in states) == 1
        primary = next(state for state in states if state["primary"])
        assert sorted(primary["paths"]) == [str(tmp_path / f"scan-{i}.xy") for i in range(2)]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()


def test_concurrent_launches_keep_every_request_and_release_lock(owner):
    app, instance, root = owner
    requests = [[root / f"scan-{index}.xy"] for index in range(5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda paths: client(owner, paths), requests))
    assert all(result.returncode == 0 for result in results), [r.stderr for r in results]
    received = []
    instance.attach(received.append)
    app.processEvents()
    assert sorted(received) == sorted([list(map(str, paths)) for paths in requests])
    instance.close()
    from xrd_workbench.ui_qt.single_instance import SingleInstance
    successor = SingleInstance(name=instance.name, lock_directory=root)
    try:
        assert successor.start_or_forward([])
    finally:
        successor.close()


def test_invalid_request_and_retry_do_not_stop_or_duplicate_delivery(owner):
    from time import monotonic, sleep
    from xrd_workbench.ui_qt.single_instance import _write_json, _read_json
    app, instance, root = owner
    request_id = uuid.uuid4().hex
    path = instance.directory / f"{request_id}.request"
    for invalid in ({"protocol": 999}, [], {"protocol": 1, "id": request_id,
                    "generation": instance.generation, "paths": ["relative.xy"]}):
        _write_json(path, invalid)
        deadline = monotonic() + 2
        while path.exists() and monotonic() < deadline:
            sleep(0.01)
        assert not path.exists()
        assert not path.with_suffix(".ack").exists()
    request = {"protocol": 1, "id": request_id, "generation": instance.generation,
               "paths": [str(root / "scan.xy")]}
    for _ in range(2):
        path.with_suffix(".ack").unlink(missing_ok=True)
        _write_json(path, request)
        deadline = monotonic() + 2
        while not path.with_suffix(".ack").exists() and monotonic() < deadline:
            sleep(0.01)
        assert _read_json(path.with_suffix(".ack"))["accepted"] is True
    app.processEvents()
    received = []
    instance.attach(received.append)
    assert received == [[str(root / "scan.xy")]]


def test_crashed_owner_lock_and_mailbox_are_recovered(tmp_path):
    from xrd_workbench.ui_qt.single_instance import SingleInstance
    app = QApplication.instance() or QApplication([])
    name = "xrd-test-" + uuid.uuid4().hex
    code = """
import os, sys
from PySide6.QtCore import QCoreApplication
from xrd_workbench.ui_qt.single_instance import SingleInstance
app = QCoreApplication([])
owner = SingleInstance(name=sys.argv[1], lock_directory=sys.argv[2])
assert owner.start_or_forward([])
os._exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code, name, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    successor = SingleInstance(name=name, lock_directory=tmp_path)
    try:
        assert successor.start_or_forward([])
    finally:
        successor.close()
