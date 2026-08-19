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
    "MhaFwdUnifiedJA": "mha_fwd_ja.so",
    "MhaBwdUnifiedJA": "mha_bwd_ja.so",
    "RmsnormFwdJA": "rmsnorm_fwd_ja.so",
    "SiluAndMulJA": "silu_and_mul_ja.so",
    "GemmFwdJA": "gemm_fwd_ja.so",
    "GemmFp4FwdJA": "gemm_fp4_ja.so",
    "CastMxfp4JA": "cast_mxfp4_ja.so",
    "CastMxfp4KeyedSrJA": "cast_mxfp4_ja.so",
    "CastMxfp4DualJA": "cast_mxfp4_ja.so",
    "CastMxfp4DualKeyedSrJA": "cast_mxfp4_ja.so",
    "KvAliasProbeJA": "kv_alias_probe_ja.so",
    "AppendKvJA": "append_kv_ja.so",
    "PagedAttentionJA": "paged_attention_ja.so",
    "PagedPrefillJA": "paged_prefill_ja.so",
}

# Symbols whose module resolves no CK or umbrella symbols and so can be loaded
# on its own. Everything else needs libjax_aiter.so loaded first, both for the
# symbols it provides and to keep the HIP context alive.
#
# The paged-KV shims qualify because they compile the aiter kernels they need
# directly into the module rather than linking a JIT-built library.
STANDALONE_SYMBOLS = {
    "KvAliasProbeJA",
    "AppendKvJA",
    "PagedAttentionJA",
    "PagedPrefillJA",
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
    """Load thin modules from aiter_build and jax_aiter_build directories."""
    global _module_handles

    if not _umbrella_handle:
        raise RuntimeError("Umbrella library must be loaded first")

    # The default wheel packages thin FFI shims but downloads the multi-GB JIT
    # libraries into a writable user cache. These directories can therefore
    # have different roots.
    aiter_dir = ja_config.get_aiter_lib_dir()
    ja_dir = ja_config.get_jax_aiter_lib_dir()

    def _load_modules(dir_path, skip_names=frozenset()):
        loaded = []
        if not dir_path.exists():
            return loaded

        for module_so in sorted(dir_path.glob("*.so")):
            if module_so.name == "libjax_aiter.so":
                continue
            if module_so.name in skip_names or module_so.name in _module_handles:
                continue
            try:
                module_handle = ctypes.CDLL(str(module_so), mode=ctypes.RTLD_GLOBAL)
                _module_handles[module_so.name] = module_handle
                loaded.append(module_so.name)
                logger.debug(f"Loaded {module_so.name}")
            except Exception as e:
                logger.error(f"Failed to load {module_so.name}: {e}")
        return loaded

    # Load aiter_build modules first (libmha_bwd.so, etc)
    aiter_loaded = _load_modules(aiter_dir)
    if aiter_loaded:
        logger.info(f"Loaded aiter_build modules: {aiter_loaded}")

    # Then load jax_aiter_build modules. The default wheel contains the MHA
    # shims but downloads their backing JIT libraries later; do not dlopen the
    # shims until their symbols can resolve.
    aiter_available = set(_module_handles)
    skip_ja = set()
    if "libmha_fwd.so" not in aiter_available:
        skip_ja.add("mha_fwd_ja.so")
    if "libmha_bwd.so" not in aiter_available:
        skip_ja.add("mha_bwd_ja.so")
    ja_loaded = _load_modules(ja_dir, skip_ja)
    if ja_loaded:
        logger.info(f"Loaded jax_aiter_build modules: {ja_loaded}")


def _standalone_search_dirs():
    """Directories a standalone module may live in.

    The same resolution load_thin_modules uses, rather than joining onto the
    build root directly: the default wheel packages the thin shims and downloads
    the JIT libraries into a writable cache, so the two can have different roots.
    """
    return (
        ja_config.get_jax_aiter_lib_dir(),
        ja_config.get_aiter_lib_dir(),
    )


def find_standalone_module(module_name: str):
    """Return the path to a standalone module, or None if it is not built."""
    for directory in _standalone_search_dirs():
        candidate = directory / module_name
        if candidate.exists():
            return candidate
    return None


def standalone_symbol_available(target_name: str) -> bool:
    """Whether a standalone target's module is present and loadable.

    Lets callers and tests probe for an optional shim without triggering the
    FileNotFoundError that registration would raise. The paged-KV shims are
    built by a separate make target, so they can legitimately be absent from an
    otherwise complete tree.
    """
    if target_name not in STANDALONE_SYMBOLS:
        return False
    module_name = SYMBOL_TO_MODULE_MAP.get(target_name)
    if not module_name:
        return False
    return module_name in _module_handles or find_standalone_module(module_name) is not None


def load_standalone_module(module_name: str):
    """Load a single self-contained module without the umbrella library.

    Only valid for modules listed via STANDALONE_SYMBOLS, which link nothing
    from libjax_aiter.so. This lets self-contained shims be used in a tree where
    the heavy aiter sources have not been built.
    """
    if module_name in _module_handles:
        return

    path = find_standalone_module(module_name)
    if path is None:
        raise FileNotFoundError(
            f"Standalone module not found: {module_name} in "
            f"{[str(d) for d in _standalone_search_dirs()]}. "
            f"Run `make -f Makefile.kv ja_kv` first."
        )

    _module_handles[module_name] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    logger.info(f"Loaded standalone module: {module_name}")


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
        if target_name in STANDALONE_SYMBOLS:
            load_standalone_module(SYMBOL_TO_MODULE_MAP[target_name])
        else:
            load_umbrella_library()
            load_thin_modules()

    # A caller can fetch optional MHA libraries in the same Python process
    # after core ops initialized the registry. Retry loading when this target's
    # module was previously unavailable; already-loaded modules are skipped.
    # Standalone modules take their own path, since load_thin_modules requires
    # the umbrella library that they exist precisely to do without.
    module_name = SYMBOL_TO_MODULE_MAP[target_name]
    if module_name not in _module_handles:
        if target_name in STANDALONE_SYMBOLS:
            load_standalone_module(module_name)
        else:
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

        if target_name == "GemmFwdJA":
            _preload_gemm_kernels(module_name)

    except Exception as e:
        logger.error(f"Failed to register FFI target '{target_name}': {e}")
        raise


def _preload_gemm_kernels(module_name: str):
    """Pre-load BF16 GEMM kernels on all visible devices to avoid blocking collectives."""
    try:
        handle = _module_handles.get(module_name)
        if handle is None:
            return
        preload_fn = getattr(handle, "gemm_fwd_ja_preload_kernels", None)
        if preload_fn is not None:
            preload_fn()
    except Exception as e:
        logger.warning(f"Failed to preload GEMM kernels: {e}")


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
