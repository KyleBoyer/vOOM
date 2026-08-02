"""Small Darwin descriptor helpers for exact ephemeral tier files."""

from __future__ import annotations

import sys


def set_darwin_nocache(fd: int) -> bool:
    """Request ``F_NOCACHE`` without making Darwin a portability dependency."""
    if sys.platform != "darwin":
        return False
    try:
        import fcntl

        command = getattr(fcntl, "F_NOCACHE", None)
        if command is None:
            return False
        fcntl.fcntl(int(fd), command, 1)
        return True
    except (ImportError, OSError):
        return False
