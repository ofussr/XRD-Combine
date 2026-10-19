"""Replace one output file only after its complete contents are written."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_output(path):
    destination = Path(path)
    fd, name = tempfile.mkstemp(dir=destination.resolve().parent,
                                prefix=destination.name + '.', suffix='.tmp')
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb+') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
