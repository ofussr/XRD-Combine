"""Per-user local file handoff, available before the heavy GUI imports.

A lifetime Qt lock elects the owner; a private, atomic file mailbox transports
requests without network sockets. Its worker acknowledges even during startup.
"""

from __future__ import annotations

from collections import deque
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

from PySide6.QtCore import QLockFile, QObject, QThread, Signal, Slot


MAX_REQUEST_BYTES = 1024 * 1024


def instance_name() -> str:
    # Neither the executable location nor version separates portable copies.
    desktop = "" if sys.platform == "win32" else os.environ.get("DISPLAY", "")
    identity = f"{getpass.getuser()}:{os.environ.get('WAYLAND_DISPLAY', '')}:{desktop}"
    return "xrd-combine-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _write_json(path, value):
    """Publish a whole message; readers never observe a partially written file."""
    fd, temporary = tempfile.mkstemp(prefix=".message-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("File request is too large")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("File messages must be JSON objects")
    return value


class _MailboxThread(QThread):
    received = Signal(list)

    def __init__(self, directory, generation):
        super().__init__()
        self.directory = directory
        self.generation = generation
        self.stopping = threading.Event()
        self.seen = deque(maxlen=2048)

    def run(self):
        while not self.stopping.is_set():
            for path in sorted(self.directory.glob("*.request")):
                try:
                    request = _read_json(path)
                    request_id = request["id"]
                    paths = request["paths"]
                    if (request.get("protocol") != 1
                            or request.get("generation") != self.generation
                            or request_id != path.stem or len(request_id) != 32
                            or not isinstance(paths, list) or len(paths) > 4096
                            or not all(isinstance(value, str) and "\0" not in value
                                       and Path(value).is_absolute() for value in paths)):
                        raise ValueError("Invalid file request")
                    if request_id not in self.seen:
                        self.seen.append(request_id)
                        self.received.emit(paths)
                    _write_json(path.with_suffix(".ack"), {
                        "id": request_id, "generation": self.generation, "accepted": True})
                    path.unlink(missing_ok=True)
                except OSError:
                    # Retry a transient read/acknowledgement failure. Seen IDs
                    # keep a successfully queued request from being delivered twice.
                    pass
                except (KeyError, TypeError, ValueError, UnicodeError):
                    # A malformed message cannot stop the owner's mailbox.
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            self.stopping.wait(0.05)


def _allow_activation(pid: int):
    if sys.platform == "win32":
        import ctypes
        # Explorer's new process grants the running process foreground rights
        # before publishing the request that will bring its window forward.
        ctypes.windll.user32.AllowSetForegroundWindow(pid)


class SingleInstance(QObject):
    """Elect one owner and queue acknowledged requests until its UI is ready."""

    def __init__(self, *, name=None, lock_directory=None):
        super().__init__()
        self.name = name or instance_name()
        directory = Path(lock_directory or tempfile.gettempdir())
        self.lock = QLockFile(str(directory / f"{self.name}.lock"))
        self.lock.setStaleLockTime(0)  # A live application may run indefinitely.
        self.directory = directory / f"{self.name}.ipc"
        self.thread = None
        self.generation = None
        self.receiver = None
        self.pending = deque()

    def _become_owner(self):
        try:
            self.directory.mkdir(mode=0o700, exist_ok=True)
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise OSError("Invalid application mailbox directory")
            if os.name != "nt":
                stat = self.directory.stat()
                if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                    raise OSError("The application mailbox must be private to this user")
            for pattern in ("*.request", "*.ack", ".message-*.tmp"):
                for path in self.directory.glob(pattern):
                    path.unlink(missing_ok=True)
            self.generation = uuid.uuid4().hex
            _write_json(self.directory / "owner.json", {
                "pid": os.getpid(), "generation": self.generation})
            self.thread = _MailboxThread(self.directory, self.generation)
            self.thread.received.connect(self._receive)
            self.thread.start()
        except Exception:
            self.close()
            raise

    def start_or_forward(self, paths, *, timeout=10.0) -> bool:
        paths = [str(Path(path).resolve()) for path in paths]
        request_id = uuid.uuid4().hex
        request = {"protocol": 1, "id": request_id, "paths": paths}
        if len(json.dumps(request, ensure_ascii=False).encode("utf-8")) > MAX_REQUEST_BYTES - 128:
            raise OSError("Too many file paths in one launch request")
        deadline = time.monotonic() + timeout
        generation = None
        message = self.directory / f"{request_id}.request"
        acknowledgement = message.with_suffix(".ack")
        try:
            while True:
                if generation is not None and acknowledgement.exists():
                    try:
                        reply = _read_json(acknowledgement)
                        if (reply.get("accepted") is True and reply.get("id") == request_id
                                and reply.get("generation") == generation):
                            return False
                    except (OSError, TypeError, ValueError, UnicodeError):
                        pass
                if self.lock.tryLock(0):
                    self._become_owner()
                    return True
                if self.lock.error() != QLockFile.LockError.LockFailedError:
                    raise OSError("Cannot create the application instance lock")
                try:
                    owner = _read_json(self.directory / "owner.json")
                    if owner["generation"] != generation:
                        _allow_activation(int(owner["pid"]))
                        request["generation"] = owner["generation"]
                        _write_json(message, request)
                        generation = owner["generation"]
                except (OSError, KeyError, TypeError, ValueError, UnicodeError):
                    pass  # The owner may be between acquiring its lock and publishing its inbox.
                if time.monotonic() >= deadline:
                    # Never open a duplicate window because its owner is busy.
                    raise OSError("The running XRD Combine could not accept the files. Please retry.")
                time.sleep(0.025)
        finally:
            message.unlink(missing_ok=True)
            acknowledgement.unlink(missing_ok=True)

    @Slot(list)
    def _receive(self, paths):
        if self.receiver is None:
            self.pending.append(paths)
        else:
            self.receiver(paths)

    def attach(self, receiver):
        self.receiver = receiver
        while self.pending:
            receiver(self.pending.popleft())

    def close(self):
        if self.thread is not None:
            self.thread.stopping.set()
            self.thread.wait()
            self.thread = None
        self.receiver = None
        if self.generation is not None:
            (self.directory / "owner.json").unlink(missing_ok=True)
            self.generation = None
        self.lock.unlock()
