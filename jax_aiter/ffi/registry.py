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
    # JA custom symbols -> custom_ja.so
    "LLMM1JA": "custom_ja.so",
    "LLZZJA": "custom_ja.so",
    "MMCustomGPUJA": "custom_ja.so",
    "WvSplitKJA": "custom_ja.so",
    "WvSplitKQJA": "custom_ja.so",
    "WvSplitKSmallJA": "custom_ja.so",
    # JA MHA symbols -> *_ja.so
    "FmhaV3BwdJA": "asm_mha_bwd_ja.so",
    "FmhaV3FwdJA": "asm_mha_fwd_ja.so",
    "FmhaV3VarlenBwdJA": "asm_mha_varlen_bwd_ja.so",
    "FmhaV3VarlenFwdJA": "asm_mha_varlen_fwd_ja.so",
    "MhaBatchPrefillJA": "ck_mha_batch_prefill_ja.so",
    "MhaBwdJA": "ck_fused_attn_bwd_ja.so",
    "MhaFwdJA": "ck_fused_attn_fwd_ja.so",
    "MhaVarlenFwdJA": "ck_mha_varlen_fwd_ja.so",
    "MhaVarlenBwdJA": "ck_mha_varlen_bwd_ja.so",
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
    """Load thin modules in main namespace from AITER and JA build directories."""
    global _module_handles

    if not _umbrella_handle:
        raise RuntimeError("Umbrella library must be loaded first")

    build_root = ja_config.get_lib_root()
    aiter_dir = build_root / "aiter_build"
    ja_dir = build_root / "jax_aiter_build"

    def _load_dir(dir_path, tag):
        """Load all .so files from a directory."""
        if not dir_path.exists():
            logger.info(f"{tag} directory not found: {dir_path}")
            return
        for module_so in sorted(dir_path.glob("*.so")):
            try:
                if module_so.name == "libjax_aiter.so":
                    continue
                module_handle = ctypes.CDLL(str(module_so), mode=ctypes.RTLD_GLOBAL)
                _module_handles[module_so.name] = module_handle
            except Exception as e:
                logger.warning(f"Could not load {module_so}: {e}")

    # Load AITER modules first, then JA modules
    _load_dir(aiter_dir, "aiter")
    _load_dir(ja_dir, "ja")


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
