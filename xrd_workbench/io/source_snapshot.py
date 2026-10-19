"""Parse a private file snapshot and reject concurrent source changes."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile

from ..models.data_errors import XRDDataError


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    digest: str | None


@contextmanager
def source_snapshot(path):
    source = Path(path)
    if not source.is_file():
        # Also retain dependency-injected readers for virtual/test sources.
        yield SourceSnapshot(source, None)
        return
    identity = _identity(source)
    with tempfile.TemporaryDirectory(prefix='xrd-source-') as folder:
        copied = Path(folder) / source.name
        digest = hashlib.sha256()
        with source.open('rb') as original, copied.open('wb') as target:
            for block in iter(lambda: original.read(1024 * 1024), b''):
                digest.update(block)
                target.write(block)
        fingerprint = digest.hexdigest()
        if _identity(source) != identity:
            raise XRDDataError('source_changed', source_name=source.name)
        yield SourceSnapshot(copied, fingerprint)
        # No project mutation occurs until both the parsed bytes and the live
        # source have been checked. Readers never hash an unrelated later file.
        if (_identity(source) != identity or source_digest(source) != fingerprint
                or source_digest(copied) != fingerprint):
            raise XRDDataError('source_changed', source_name=source.name)
