import os
from pathlib import Path
from typing import Iterable, Optional

try:
    from importlib.resources import files as pkg_files
except ImportError:
    # Fallback for older Python versions.
    from importlib import resources

    pkg_files = resources.files

def get_packaged_lib_dir():
    """Returns a Traversable to jax_aiter/_lib inside the installed package."""
    return pkg_files("jax_aiter")


def get_repo_root() -> Path:
    # Prefer explicit env for local builds,
    # otherwise fall back to common repo layout.
    root = os.environ.get("JA_ROOT_DIR")
    if root:
        return Path(root).resolve() / "build" / "aiter_build"
    elif get_packaged_lib_dir():
        return get_packaged_lib_dir() / "_lib"

    raise FileNotFoundError(
        f"Can't find JAX Aiter library. Set JA_ROOT_DIR or install the package."
    )


def get_umbrella_lib() -> Path:
    """Get the path to the umbrella shared library."""
    return get_repo_root() / "libjax_aiter.so"


def get_local_lib_dirs() -> list[Path]:
    """Get directories where locally built .so files are expected."""
    root = get_repo_root()
    return [
        root / "build" / "bin",
        root / "build" / "aiter_build",
    ]
