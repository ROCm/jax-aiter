# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
JAX-AITER JIT Core Module
Reuses AITER's core compilation system with JAX FFI adaptations
"""

import os
import sys
import json
import subprocess
import functools
import importlib
from pathlib import Path
from typing import Optional, Callable, Any, List

import jax
import jax.numpy as jnp
import numpy as np
from functools import lru_cache


def setup_aiter_environment():
    """Setup environment and import AITER's core module"""

    # Get JA_ROOT_DIR from environment
    if "JA_ROOT_DIR" not in os.environ:
        raise RuntimeError("JA_ROOT_DIR environment variable not set")

    jax_aiter_root = Path(os.environ["JA_ROOT_DIR"])
    aiter_root = jax_aiter_root / "build" / "hipified_aiter"
    aiter_jit_dir = aiter_root / "aiter" / "jit"

    # Add AITER to Python path
    sys.path.insert(0, str(aiter_jit_dir))
    sys.path.insert(0, str(aiter_jit_dir / "utils"))

    try:
        # Import AITER's core module
        import core as aiter_core

        return jax_aiter_root, aiter_core
    except ImportError as e:
        raise ImportError(
            f"Error importing AITER core: {e}. Make sure AITER is properly built and hipified."
        )


# Initialize AITER environment
JAX_AITER_ROOT, AITER_CORE = setup_aiter_environment()

# Import AITER functions directly to avoid code duplication
from third_party.aiter.aiter.jit.core import (
    build_module,
    mp_lock,
    get_user_jit_dir,
    get_asm_dir,
    recopy_ck,
    clear_build,
    rm_module,
    check_numa,
    AITER_REBUILD,
    __mds,
)

# Set up JAX-AITER specific globals
AITER_CORE.JA_ROOT_DIR = JAX_AITER_ROOT
_REGISTERED_FFI_TARGETS = set()


def patch_aiter_get_args_of_build():
    """Patch AITER's get_args_of_build to use JAX-AITER configuration"""

    original_get_args_of_build = AITER_CORE.get_args_of_build

    def get_args_of_build_ja(ops_name: str, exclude=[]):
        # Load JAX-AITER configuration
        config_path = JAX_AITER_ROOT / "jax_aiter" / "jit" / "optCompilerConfig.json"
        with open(config_path, "r") as f:
            our_config = json.load(f)

        # Check if the operation is in JAX-AITER config
        if ops_name in our_config:
            config = our_config[ops_name]

            # Get JAX FFI include directory
            try:
                jax_ffi_include = subprocess.check_output(
                    ["python", "-c", "from jax import ffi; print(ffi.include_dir())"],
                    text=True,
                ).strip()
            except:
                jax_ffi_include = ""

            # Get PyTorch include directories (needed for compilation)
            torch_site = str(JAX_AITER_ROOT / "third_party" / "pytorch")
            pytorch_include_dirs = [
                f"{torch_site}/torch/csrc/api/include",
                f"{torch_site}/aten",
                f"{torch_site}/aten/src",
                f"{torch_site}/aten/src/ATen",
                f"{torch_site}/build_static",
                f"{torch_site}/build_static/aten/src",
                f"{torch_site}/build_static/install/include",
                f"{torch_site}/torch/csrc",
            ]

            # Convert string expressions to actual values
            def eval_config_value(value):
                if isinstance(value, str) and value.startswith("f'"):
                    eval_globals = {
                        "os": os,
                        "subprocess": subprocess,
                        "JA_ROOT_DIR": JAX_AITER_ROOT,
                        "AITER_CSRC_DIR": AITER_CORE.AITER_CSRC_DIR,
                        "CK_DIR": AITER_CORE.CK_DIR,
                        "get_asm_dir": get_asm_dir,
                        "jax_ffi_include_dir": jax_ffi_include,
                        "torch_site": torch_site,
                        "pytorch_include_dirs": pytorch_include_dirs,
                    }
                    try:
                        return eval(value, eval_globals)
                    except:
                        return value
                elif isinstance(value, list):
                    return [eval_config_value(v) for v in value]
                else:
                    return value

            # Process the config
            processed_config = {}
            for key, value in config.items():
                processed_config[key] = eval_config_value(value)

            # Set defaults for missing keys
            default_config = {
                "md_name": ops_name,
                "flags_extra_cc": [],
                "flags_extra_hip": [],
                "extra_ldflags": [],
                "extra_include": [],
                "verbose": False,
                "is_python_module": True,
                "is_standalone": False,
                "torch_exclude": True,
                "hip_clang_path": None,
                "blob_gen_cmd": "",
            }

            # Merge with defaults
            default_config.update(processed_config)

            # Add PyTorch include directories (needed for compilation)
            pytorch_includes = [f"-I{torch_site}"]
            pytorch_includes.extend(
                [f"-I{inc_dir}" for inc_dir in pytorch_include_dirs]
            )

            # Add JAX-AITER specific flags
            ja_includes = [f"-I{jax_ffi_include}", f"-I{JAX_AITER_ROOT}/csrc/common"]

            # Add includes to both CC and HIP flags
            if "flags_extra_cc" not in default_config:
                default_config["flags_extra_cc"] = []
            default_config["flags_extra_cc"].extend(pytorch_includes)
            default_config["flags_extra_cc"].extend(ja_includes)

            return default_config
        else:
            # Fall back to original AITER config
            return original_get_args_of_build(ops_name, exclude)

    # Replace the function in AITER core
    AITER_CORE.get_args_of_build = get_args_of_build_ja


# Apply the patch
patch_aiter_get_args_of_build()


def get_lib_handle():
    """Get the JAX-AITER library handle for FFI registration"""
    try:
        from ..ffi.registry import get_lib_handle

        return get_lib_handle()
    except ImportError:
        raise ImportError("Could not import JAX-AITER FFI registry")


def ensure_ffi_target_registered(target_name: str):
    """Lazily register a specific FFI target when first needed"""
    global _REGISTERED_FFI_TARGETS

    if target_name not in _REGISTERED_FFI_TARGETS:
        lib = get_lib_handle()

        jax.ffi.register_ffi_target(
            target_name,
            jax.ffi.pycapsule(getattr(lib, target_name)),
            platform="ROCM",
        )
        _REGISTERED_FFI_TARGETS.add(target_name)


@lru_cache(maxsize=1024)
def get_module_ja(md_name):
    """JAX-AITER version of get_module that handles FFI registration"""
    check_numa()

    global __mds
    if md_name not in __mds:
        if "AITER_JIT_DIR" in os.environ:
            __mds[md_name] = importlib.import_module(md_name)
        else:
            # Use AITER's JIT directory but import as JAX module
            jit_dir = get_user_jit_dir()
            sys.path.insert(0, jit_dir)
            __mds[md_name] = importlib.import_module(md_name)

    return __mds[md_name]


def ja_compile_ops(
    _md_name: str,
    fc_name: Optional[str] = None,
    gen_func: Optional[Callable[..., dict[str, Any]]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
):
    """
    JAX-AITER version of compile_ops decorator

    Args:
        _md_name: Module name in optCompilerConfig.json
        fc_name: FFI function name (defaults to decorated function name)
        gen_func: Function to generate additional build arguments
        gen_fake: Function to generate fake outputs for abstract evaluation
    """

    def decorator(func):
        func.arg_checked = False

        @functools.wraps(func)
        def wrapper(*args, custom_build_args={}, **kwargs):

            loadName = fc_name if fc_name is not None else func.__name__
            md_name = _md_name

            try:
                module = None
                if gen_func is not None:
                    custom_build_args.update(gen_func(*args, **kwargs))

                # Check if we need to rebuild
                if AITER_REBUILD and md_name not in AITER_CORE.rebuilded_list:
                    AITER_CORE.rebuilded_list.append(md_name)
                    raise ModuleNotFoundError("")

                if module is None:
                    md = custom_build_args.get("md_name", md_name)
                    module = get_module_ja(md)

            except ModuleNotFoundError:
                # Module doesn't exist, need to build it
                d_args = AITER_CORE.get_args_of_build(md_name)
                d_args.update(custom_build_args)

                # Update module name if we have custom build
                md_name = custom_build_args.get("md_name", md_name)

                # Extract build arguments
                srcs = d_args["srcs"]
                flags_extra_cc = d_args["flags_extra_cc"]
                flags_extra_hip = d_args["flags_extra_hip"]
                blob_gen_cmd = d_args["blob_gen_cmd"]
                extra_include = d_args["extra_include"]
                extra_ldflags = d_args["extra_ldflags"]
                verbose = d_args["verbose"]
                is_python_module = d_args["is_python_module"]
                is_standalone = d_args["is_standalone"]
                torch_exclude = d_args["torch_exclude"]
                hipify = d_args.get("hipify", True)
                hip_clang_path = d_args.get("hip_clang_path", None)

                # Set HIP_CLANG_PATH if specified
                prev_hip_clang_path = None
                if hip_clang_path is not None and os.path.exists(hip_clang_path):
                    prev_hip_clang_path = os.environ.get("HIP_CLANG_PATH", None)
                    os.environ["HIP_CLANG_PATH"] = hip_clang_path

                # Build the module using AITER's build_module
                build_module(
                    md_name,
                    srcs,
                    flags_extra_cc,
                    flags_extra_hip,
                    blob_gen_cmd,
                    extra_include,
                    extra_ldflags,
                    verbose,
                    is_python_module,
                    is_standalone,
                    torch_exclude,
                    hipify,
                )

                # Restore HIP_CLANG_PATH
                if hip_clang_path is not None:
                    if prev_hip_clang_path is not None:
                        os.environ["HIP_CLANG_PATH"] = prev_hip_clang_path
                    else:
                        os.environ.pop("HIP_CLANG_PATH", None)

                if is_python_module:
                    module = get_module_ja(md_name)

                if md_name not in __mds:
                    __mds[md_name] = module

            # Register FFI target and call the function
            ensure_ffi_target_registered(loadName)

            # For JAX, we don't need to do the complex PyTorch-style wrapping
            # Just return the original function - FFI registration is handled above
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Re-export useful AITER functions
from third_party.aiter.aiter.jit.core import (
    validate_and_update_archs,
    hip_flag_checker,
    check_and_set_ninja_worker,
)


# JAX-specific utilities
def create_jax_ffi_call(
    target_name: str,
    output_shapes_dtypes: List[jax.ShapeDtypeStruct],
    vmap_method: str = "broadcast_all",
):
    """
    Helper to create JAX FFI calls with proper shape/dtype specification

    Args:
        target_name: Name of the FFI target
        output_shapes_dtypes: List of output shape/dtype specifications
        vmap_method: Vectorization method for the call

    Returns:
        JAX FFI call function
    """
    return jax.ffi.ffi_call(
        target_name,
        output_shapes_dtypes,
        vmap_method=vmap_method,
    )


def ja_build_module_on_demand(module_name: str):
    """
    Build a single module on demand

    Args:
        module_name: Name of the module in optCompilerConfig.json
    """
    try:
        # Get build configuration
        build_args = AITER_CORE.get_args_of_build(module_name)

        # Build the module using AITER's build system
        build_module(
            build_args["md_name"],
            build_args["srcs"],
            build_args["flags_extra_cc"],
            build_args["flags_extra_hip"],
            build_args["blob_gen_cmd"],
            build_args["extra_include"],
            build_args["extra_ldflags"],
            build_args["verbose"],
            build_args.get("is_python_module", True),
            build_args.get("is_standalone", False),
            build_args.get("torch_exclude", False),
            build_args.get("hipify", True),
        )

        return True

    except Exception as e:
        print(f"Failed to build {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


# Export the main decorator for use by JAX-AITER modules
__all__ = [
    "ja_compile_ops",
    "ensure_ffi_target_registered",
    "create_jax_ffi_call",
    "ja_build_module_on_demand",
    "JAX_AITER_ROOT",
    "get_lib_handle",
]
