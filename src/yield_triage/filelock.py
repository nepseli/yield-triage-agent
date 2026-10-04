"""Tiny cross-process exclusive file lock (POSIX ``fcntl``, Windows ``msvcrt``).

What: a context manager that holds an OS-level exclusive lock on a sidecar
``<path>.lock`` file.

Why: the MCP server and the human approval CLI are separate processes that
both append to the audit log, and the commit step must check-and-consume a
token nonce atomically. Without a lock, two writers could fork the hash chain
or both accept the same token.

Connects to: ``audit.py`` and ``approvals.py``.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_TIMEOUT_S = 10.0


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Block (up to ``LOCK_TIMEOUT_S``) until we own ``<path>.lock``."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _acquire(fd)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


if sys.platform == "win32":
    import msvcrt

    def _acquire(fd: int) -> None:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("could not acquire file lock") from None
                time.sleep(0.01)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
