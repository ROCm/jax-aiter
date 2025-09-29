# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
# JAX-AITER FFI Registry
# This file manages the loading and registration of thin module shared libraries (.so)
# for JAX custom operations. It loads the umbrella library and all module_*.so files
# from build/aiter_build, resolves symbols, and registers FFI targets
# with JAX for use in custom GPU kernels.
"""
import jax
import ctypes
import logging
from typing import Dict, List, Optional

from ..ja_compat import config as ja_config

logger = logging.getLogger("JAX_AITER")

# Global state - simplified.
_umbrella_handle: Optional[ctypes.CDLL] = None
_module_handles: Dict[str, ctypes.CDLL] = {}
_registered_ffi_targets = set()
_jax_initialized = False

# Symbol to module mapping.
SYMBOL_TO_MODULE_MAP = {
    # module_custom.so symbols.
    "LLMM1JA": "module_custom.so",
    "LLZZJA": "module_custom.so",
    "MMCustomGPUJA": "module_custom.so",
    "WvSplitKJA": "module_custom.so",
    "WvSplitKQJA": "module_custom.so",
    "WvSplitKSmallJA": "module_custom.so",
    # Individual module symbols.
    "FmhaV3BwdJA": "module_fmha_v3_bwd.so",
    "FmhaV3FwdJA": "module_fmha_v3_fwd.so",
    "FmhaV3VarlenBwdJA": "module_fmha_v3_varlen_bwd.so",
    "FmhaV3VarlenFwdJA": "module_fmha_v3_varlen_fwd.so",
    "MhaBatchPrefillJA": "module_mha_batch_prefill.so",
    "MhaBwdJA": "module_mha_bwd.so",
    "MhaFwdJA": "module_mha_fwd.so",
    "MhaVarlenFwdJA": "module_mha_varlen_fwd.so",
    "MhaVarlenBwdJA": "module_mha_varlen_bwd.so",
}


def ensure_jax_backend_ready():
    """Ensure JAX backend is initialized before loading libraries."""
    global _jax_initialized
    if not _jax_initialized:
        try:
            devices = jax.devices()
            logger.info(f"JAX backend initialized with devices: {devices}")
            _jax_initialized = True
        except Exception as e:
            logger.warning(f"JAX backend initialization failed: {e}")
            _jax_initialized = True


def load_umbrella_library():
    """Load the umbrella library in main namespace to preserve HIP context."""
    global _umbrella_handle

    if _umbrella_handle is not None:
        return _umbrella_handle

    umbrella_path = ja_config.get_umbrella_lib()
    if not umbrella_path.exists():
        raise FileNotFoundError(
            f"Umbrella library not found: {umbrella_path}. Run `make` first."
        )

    logger.info("Loading umbrella library in main namespace to preserve HIP context...")
    try:
        _umbrella_handle = ctypes.CDLL(str(umbrella_path))
        logger.info("Successfully loaded umbrella library in main namespace")
        return _umbrella_handle
    except Exception as e:
        logger.error(f"Failed to load umbrella library: {e}")
        raise OSError(f"Failed to load umbrella library: {e}")


def load_thin_modules():
    """Load thin modules in main namespace."""
    global _module_handles

    if not _umbrella_handle:
        raise RuntimeError("Umbrella library must be loaded first")

    lib_dir = ja_config.get_repo_root()
    if not lib_dir.exists():
        logger.warning(f"Module directory not found: {lib_dir}")
        return

    # Search for *.so files in configured directories.
    for module_so in sorted(lib_dir.glob("*.so")):
        try:
            if module_so.name == "libjax_aiter.so":
                continue
            logger.info(f"Loading thin module: {module_so}")
            module_handle = ctypes.CDLL(str(module_so), mode=ctypes.RTLD_GLOBAL)
            _module_handles[module_so.name] = module_handle
            logger.info(f"Loaded thin module: {module_so}")
        except Exception as e:
            logger.warning(f"Could not load {module_so}: {e}")


def resolve_symbol(target_name: str) -> int:
    """Resolve a symbol from the appropriate module."""
    module_name = SYMBOL_TO_MODULE_MAP.get(target_name)
    if not module_name:
        raise ValueError(
            f"Unknown symbol: {target_name}. Available symbols: {list(SYMBOL_TO_MODULE_MAP.keys())}"
        )

    module_handle = _module_handles.get(module_name)
    if not module_handle:
        raise RuntimeError(
            f"Module not loaded: {module_name}. Available modules: {list(_module_handles.keys())}"
        )

    try:
        symbol_func = getattr(module_handle, target_name)
        symbol_ptr = ctypes.cast(symbol_func, ctypes.c_void_p).value
        if not symbol_ptr:
            raise RuntimeError(f"Symbol {target_name} resolved to NULL pointer")
        return symbol_ptr
    except AttributeError:
        raise RuntimeError(f"Symbol not found: {target_name} in {module_name}")


def register_ffi_target(target_name: str, platform: str = "ROCM"):
    """Register an FFI target with JAX."""
    # Return if already registered.
    if target_name in _registered_ffi_targets:
        return

    # Ensure libraries are loaded.
    if _umbrella_handle is None:
        load_umbrella_library()
        load_thin_modules()

    logger.info(f"Registering FFI target: {target_name}")

    try:
        # Resolve the symbol.
        symbol_ptr = resolve_symbol(target_name)

        jax.ffi.register_ffi_target(
            target_name,
            jax.ffi.pycapsule(symbol_ptr),
            platform=platform,
        )
        _registered_ffi_targets.add(target_name)
        module_name = SYMBOL_TO_MODULE_MAP[target_name]
        logger.info(
            f"FFI target '{target_name}' from {module_name} registered successfully with JAX."
        )

    except Exception as e:
        logger.error(f"Failed to register FFI target '{target_name}': {e}")
        raise


def get_available_symbols() -> List[str]:
    """Get list of all available symbols."""
    return list(SYMBOL_TO_MODULE_MAP.keys())


def get_loaded_modules() -> List[str]:
    """Get list of all loaded modules."""
    return list(_module_handles.keys())


def get_registry_status():
    """Get the current status of the registry system."""
    return {
        "jax_initialized": _jax_initialized,
        "umbrella_loaded": _umbrella_handle is not None,
        "loaded_modules": list(_module_handles.keys()),
        "registered_targets": list(_registered_ffi_targets),
        "available_symbols": len(SYMBOL_TO_MODULE_MAP),
    }


def print_registry_status():
    """Print a human-readable status of the registry system."""
    status = get_registry_status()

    logger.info("=== Registry Status ===")
    logger.info(f"JAX Initialized: {status['jax_initialized']}")
    logger.info(f"Umbrella Loaded: {status['umbrella_loaded']}")
    logger.info(
        f"Loaded Modules ({len(status['loaded_modules'])}): {status['loaded_modules']}"
    )
    logger.info(
        f"Registered Targets ({len(status['registered_targets'])}): {status['registered_targets']}"
    )
    logger.info(f"Available Symbols: {status['available_symbols']}")
    logger.info("=======================")
