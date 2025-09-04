import jax
import ctypes
import threading
from pathlib import Path
from typing import Dict

from ..ja_compat.config import get_shared_lib

_lock = threading.Lock()
_lib_handle: ctypes.CDLL | None = None


# (Ruturaj4): Add multiple targets later.
def _load_ja_library():
    """Load and cache the JA library, returning a persistent CDLL handle."""
    global _lib_handle
    with _lock:
        if _lib_handle is not None:
            return _lib_handle

        so_path: Path = get_shared_lib()
        if not so_path.is_file():
            raise FileNotFoundError(f"{so_path} not found – did you run `make`?")
        # (Ruturaj4): Mode is 1 is RTLD_LAZY.
        _lib_handle = ctypes.CDLL(str(so_path), mode=1)
        return _lib_handle


def get_lib_handle() -> ctypes.CDLL:
    """Public accessor to the CDLL handle (loads it if not already loaded)."""
    return _load_ja_library()
