import os
import logging
from pathlib import Path
from typing import Iterable, Optional

try:
    from importlib.resources import files as pkg_files
except ImportError:
    # Fallback for older Python versions.
    from importlib import resources

    pkg_files = resources.files

logger = logging.getLogger("JAX_AITER")


def get_packaged_lib_dir():
    """Returns a Traversable to jax_aiter/_lib inside the installed package."""
    return pkg_files("jax_aiter")


def get_package_install_dir() -> Path:
    """Get the actual filesystem path where jax_aiter is installed.

    Returns:
        Path to the jax_aiter package directory (e.g., /usr/local/lib/python3.10/dist-packages/jax_aiter)
    """
    import jax_aiter

    return Path(jax_aiter.__file__).parent


def get_lib_root() -> Path:
    # Prefer explicit env for local builds,
    # otherwise fall back to common repo layout.
    root = os.environ.get("JA_ROOT_DIR")
    if root:
        return Path(root).resolve() / "build"
    elif get_packaged_lib_dir():
        return get_packaged_lib_dir() / "_lib"

    raise FileNotFoundError(
        f"Can't find JAX Aiter library. Set JA_ROOT_DIR or install the package."
    )


def get_umbrella_lib() -> Path:
    """Get the path to the umbrella shared library."""
    return get_lib_root() / "jax_aiter_build" / "libjax_aiter.so"


def set_aiter_asm_dir():
    """
    Set AITER_ASM_DIR environment variable to the base HSA directory.

    NEW CONVENTION (AITER 3baf198+):
    AITER_ASM_DIR should point to the base hsa/ directory (no arch suffix).
    AITER's get_asm_dir() function will append the architecture dynamically.

    The path is set to:
    - For development mode: <repo_root>/third_party/aiter/hsa/
    - For installed packages: <package_location>/jax_aiter/_hsa/

    This should be called early during package initialization.
    """
    # Only set if not already set by user.
    if os.environ.get("AITER_ASM_DIR"):
        logger.info(f"AITER_ASM_DIR already set to: {os.environ['AITER_ASM_DIR']}")
        return

    try:
        from .chip_info import get_gfx

        # Get GPU architecture for verification (e.g., "gfx942", "gfx950").
        gfx_arch = get_gfx()

        # Check if we're in development mode (JA_ROOT_DIR set).
        ja_root = os.environ.get("JA_ROOT_DIR")

        if ja_root:
            # Development mode - use third_party/aiter/hsa (base directory).
            hsa_base = Path(ja_root) / "third_party" / "aiter" / "hsa"
        else:
            # Installed package mode - use jax_aiter/_hsa (base directory).
            pkg_root = get_package_install_dir()
            hsa_base = pkg_root / "_hsa"

        # NEW CONVENTION: Set to base hsa/ directory (AITER appends arch internally).
        if hsa_base.exists():
            # Verify arch-specific subdirectory exists as a sanity check.
            arch_dir = hsa_base / gfx_arch
            if not arch_dir.exists():
                logger.warning(
                    f"HSA arch directory not found: {arch_dir}. "
                    f"Assembly kernels may not be available for {gfx_arch}."
                )
            
            os.environ["AITER_ASM_DIR"] = str(hsa_base) + "/"
            logger.info(f"Set AITER_ASM_DIR to: {os.environ['AITER_ASM_DIR']}")
        else:
            logger.warning(
                f"HSA base directory not found: {hsa_base}. "
                f"Assembly kernels will not be available."
            )
    except Exception as e:
        logger.warning(f"Failed to set AITER_ASM_DIR: {e}")
