#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Build script for JAX-AITER using AITER's JIT system.

import os
import re
import sys
import shutil
import subprocess
import json
import time
import argparse
import functools
from pathlib import Path
import logging

logger = logging.getLogger("JAX_AITER")


def setup_environment():
    """Setup environment variables for JAX-AITER and AITER."""

    # Get JA_ROOT_DIR from environment.
    if "JA_ROOT_DIR" not in os.environ:
        logger.error("JA_ROOT_DIR environment variable not set")
        logger.error("Please set JA_ROOT_DIR to your JAX-AITER root directory")
        sys.exit(1)

    jax_aiter_root = Path(os.environ["JA_ROOT_DIR"])
    aiter_root = jax_aiter_root / "third_party" / "aiter"
    aiter_jit_dir = aiter_root / "aiter" / "jit"

    # Add AITER to Python path for imports.
    sys.path.insert(0, str(aiter_jit_dir))
    sys.path.insert(0, str(aiter_jit_dir / "utils"))

    logger.info("Environment Setup:")
    logger.info(f"  JA_ROOT_DIR: {os.environ['JA_ROOT_DIR']}")
    logger.info(f"  AITER JIT DIR: {aiter_jit_dir}")

    return jax_aiter_root, aiter_jit_dir


def patch_aiter_core(core_module, jax_aiter_root):
    """Patch AITER's core module to support JAX-AITER configuration."""

    # Add JA_ROOT_DIR to the core module's globals.
    core_module.JA_ROOT_DIR = jax_aiter_root

    # AITER_ASM_DIR convention: base hsa/ directory without arch suffix.
    orig_asm_dir = core_module.AITER_ASM_DIR
    if orig_asm_dir.rstrip(os.sep).endswith(('gfx942', 'gfx950', 'gfx1100', 'gfx1150', 'gfx1200')):
        gfx_dir = orig_asm_dir.rstrip(os.sep)
        hsa_dir = os.path.dirname(gfx_dir) + "/"
        core_module.AITER_ASM_DIR = hsa_dir
        os.environ["AITER_ASM_DIR"] = hsa_dir
        logger.info(f"Detected old AITER_ASM_DIR convention, updated to: {hsa_dir}")
    else:
        logger.info(f"AITER_ASM_DIR already using new convention: {orig_asm_dir}")

    # Propagate GPU_ARCHS to AITER_GPU_ARCHS for hsa/codegen.py.
    if "AITER_GPU_ARCHS" not in os.environ and "GPU_ARCHS" in os.environ:
        os.environ["AITER_GPU_ARCHS"] = os.environ["GPU_ARCHS"]

    # Override get_user_jit_dir to use JAX-AITER build directory.
    @functools.lru_cache(maxsize=1)
    def get_user_jit_dir_ja():
        """Return JAX-AITER JIT build directory instead of AITER default."""
        jit_dir = jax_aiter_root / "build" / "aiter_build"
        jit_dir.mkdir(parents=True, exist_ok=True)
        # Add to Python path if not already there.
        jit_dir_str = str(jit_dir)
        if jit_dir_str not in sys.path:
            sys.path.insert(0, jit_dir_str)
        return jit_dir_str

    # Replace the function in the core module.
    core_module.get_user_jit_dir = get_user_jit_dir_ja

    # Ensure bd_dir is correct after patching get_user_jit_dir.
    core_module.bd_dir = Path(get_user_jit_dir_ja()) / "build"
    core_module.bd_dir.mkdir(parents=True, exist_ok=True)
    # core_module.CK_DIR = f"{core_module.bd_dir}/ck"
    # Path(core_module.CK_DIR).mkdir(parents=True, exist_ok=True)
    # recopy_ck caches by function; after changing CK_DIR, clear cache:
    if hasattr(core_module, "recopy_ck"):
        core_module.recopy_ck.cache_clear()

    # Override the get_args_of_build function to use JA config.
    original_get_args_of_build = core_module.get_args_of_build

    def get_args_of_build_ja(ops_name, exclude=[]):
        """
        Override get_args_of_build to use JAX-AITER configuration.
        """
        # Set defaults for missing keys.
        d_opt_build_args = {
            "srcs": [],
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

        # Convert string expressions to actual values using eval.
        def eval_config_value(value):
            if isinstance(value, str):
                eval_globals = {
                    "os": os,
                    "subprocess": subprocess,
                    "JA_ROOT_DIR": core_module.JA_ROOT_DIR,
                    "AITER_CSRC_DIR": core_module.AITER_CSRC_DIR,
                    "CK_DIR": core_module.CK_DIR,
                    "get_asm_dir": core_module.get_asm_dir,
                    "jax_ffi_include_dir": jax_ffi_include,
                }
                try:
                    return eval(value, eval_globals)
                except:
                    return value
            elif isinstance(value, list):
                return [eval_config_value(v) for v in value]
            else:
                return value

        # Load JA configuration from JSON file.
        config_path = jax_aiter_root / "jax_aiter" / "jit" / "optCompilerConfig.json"
        with open(config_path, "r") as f:
            our_config = json.load(f)

        # Check if the operation is in JA config.
        if ops_name in our_config:
            # Use JA config for this operation.
            config = our_config[ops_name]

            # Get JAX FFI include directory for JAX integration.
            try:
                jax_ffi_include = subprocess.check_output(
                    ["python", "-c", "from jax import ffi; print(ffi.include_dir())"],
                    text=True,
                ).strip()
            except:
                jax_ffi_include = ""

            # Process the config values through eval.
            processed_config = {}
            for key, value in config.items():
                processed_config[key] = eval_config_value(value)

            # Merge with defaults.
            d_opt_build_args.update(processed_config)

            # Define JAX-aiter specific include flags.
            ja_includes = [
                f"-I{jax_ffi_include}",
                f"-I{core_module.JA_ROOT_DIR}/csrc/common",
            ]

            # All three configured modules use torch_exclude=True. The previous
            # PyTorch include list was dead weight: Ninja's dependency database
            # for every built object contains zero PyTorch headers.
            if "flags_extra_cc" not in d_opt_build_args:
                d_opt_build_args["flags_extra_cc"] = []
            d_opt_build_args["flags_extra_cc"].extend(ja_includes)

            # Add jax-aiter library linking flags.
            d_opt_build_args["extra_ldflags"].extend(
                [
                    "-Wl,--no-as-needed",
                    f"-L{core_module.JA_ROOT_DIR}/build/jax_aiter_build",
                    "-ljax_aiter",
                    "-Wl,--as-needed",
                    "-Wl,-rpath,'$ORIGIN'",
                    "-Wl,--enable-new-dtags",
                ]
            )

            return d_opt_build_args
        else:
            # Fall back to original AITER config.
            return original_get_args_of_build(ops_name, exclude)

    # Replace the function with our customized version.
    core_module.get_args_of_build = get_args_of_build_ja

    # Prevent ninja file regeneration for incremental builds.
    # Import and patch the cpp_extension module that AITER uses.
    # (Ruturaj4): Remove most of this after merging
    # https://github.com/ROCm/aiter/pull/1010.
    try:
        import cpp_extension

        def _prepare_ldflags_ja(
            extra_ldflags, with_cuda, verbose, is_standalone, torch_exclude
        ):
            extra_ldflags.append("-mcmodel=large")
            extra_ldflags.append("-ffunction-sections")
            extra_ldflags.append("-fdata-sections ")
            extra_ldflags.append("-Wl,--gc-sections")
            extra_ldflags.append("-Wl,--cref")
            if not torch_exclude:
                import torch

                _TORCH_PATH = os.path.join(os.path.dirname(torch.__file__))
                TORCH_LIB_PATH = os.path.join(_TORCH_PATH, "lib")
                extra_ldflags.append(f"-L{TORCH_LIB_PATH}")
                extra_ldflags.append("-lc10")
                if with_cuda:
                    extra_ldflags.append(
                        "-lc10_hip"
                        if cpp_extension.IS_HIP_EXTENSION
                        else "-lc10_cuda"
                    )
                extra_ldflags.append("-ltorch_cpu")
                if with_cuda:
                    extra_ldflags.append(
                        "-ltorch_hip"
                        if cpp_extension.IS_HIP_EXTENSION
                        else "-ltorch_cuda"
                    )
                extra_ldflags.append("-ltorch")
                if not is_standalone:
                    extra_ldflags.append("-ltorch_python")

                if is_standalone:
                    extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

            if with_cuda and cpp_extension.IS_HIP_EXTENSION:
                if verbose:
                    print("Detected CUDA files, patching ldflags", file=sys.stderr)

                extra_ldflags.append(f'-L{cpp_extension._join_rocm_home("lib")}')
                extra_ldflags.append("-lamdhip64")

            return extra_ldflags

        cpp_extension._prepare_ldflags = _prepare_ldflags_ja

        # Mock torch module for cpp_extension when torch_exclude=True.
        original_write_ninja_file_to_build_library = cpp_extension._write_ninja_file_to_build_library

        def _write_ninja_file_to_build_library_ja_inner(
            path,
            name,
            sources,
            extra_cflags,
            extra_cuda_cflags,
            extra_ldflags,
            extra_include_paths,
            with_cuda,
            is_python_module,
            is_standalone,
            torch_exclude,
            extra_cuda_cflags_per_source=None,
        ) -> None:
            """Wrapper to handle torch import when torch_exclude=True."""
            if torch_exclude:
                import sys
                import types

                mock_torch = types.ModuleType('torch')
                # cpp_extension only needs a module-shaped object while
                # torch_exclude=True. Point at an existing repository file;
                # no PyTorch source tree or runtime package is consumed.
                mock_torch.__file__ = str(
                    core_module.JA_ROOT_DIR / "jax_aiter" / "__init__.py"
                )
                sys.modules['torch'] = mock_torch
                
                try:
                    original_write_ninja_file_to_build_library(
                        path=path,
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags,
                        extra_cuda_cflags=extra_cuda_cflags,
                        extra_ldflags=extra_ldflags,
                        extra_include_paths=extra_include_paths,
                        with_cuda=with_cuda,
                        is_python_module=is_python_module,
                        is_standalone=is_standalone,
                        torch_exclude=torch_exclude,
                        extra_cuda_cflags_per_source=extra_cuda_cflags_per_source,
                    )
                finally:
                    if 'torch' in sys.modules and sys.modules['torch'] is mock_torch:
                        del sys.modules['torch']
            else:
                original_write_ninja_file_to_build_library(
                    path=path,
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    with_cuda=with_cuda,
                    is_python_module=is_python_module,
                    is_standalone=is_standalone,
                    torch_exclude=torch_exclude,
                    extra_cuda_cflags_per_source=extra_cuda_cflags_per_source,
                )
        
        cpp_extension._write_ninja_file_to_build_library = _write_ninja_file_to_build_library_ja_inner

        def _write_ninja_file_and_build_library_ja(
            name,
            sources,
            extra_cflags,
            extra_cuda_cflags,
            extra_ldflags,
            extra_include_paths,
            build_directory: str,
            verbose: bool,
            with_cuda,
            is_python_module: bool,
            is_standalone: bool = False,
            torch_exclude: bool = False,
            extra_cuda_cflags_per_source=None,
        ) -> None:
            cpp_extension.verify_ninja_availability()

            compiler = cpp_extension.get_cxx_compiler()
            cpp_extension.get_compiler_abi_compatibility_and_version(
                compiler, torch_exclude
            )
            if with_cuda is None:
                with_cuda = any(map(cpp_extension._is_cuda_file, sources))
            extra_ldflags = cpp_extension._prepare_ldflags(
                extra_ldflags or [],
                with_cuda,
                verbose,
                is_standalone,
                torch_exclude,
            )
            build_file_path = os.path.join(build_directory, "build.ninja")
            if verbose:
                print(
                    f"Emitting ninja build file {build_file_path}...", file=sys.stderr
                )
            # NOTE: Emitting a new ninja build file does not cause re-compilation if
            # the sources did not change, so it's ok to re-emit (and it's fast).
            cpp_extension._write_ninja_file_to_build_library(
                path=build_file_path,
                name=name,
                sources=sorted(set(sources)),
                extra_cflags=extra_cflags or [],
                extra_cuda_cflags=sorted(extra_cuda_cflags) or [],
                extra_ldflags=extra_ldflags or [],
                extra_include_paths=extra_include_paths or [],
                with_cuda=with_cuda,
                is_python_module=is_python_module,
                is_standalone=is_standalone,
                torch_exclude=torch_exclude,
                extra_cuda_cflags_per_source=extra_cuda_cflags_per_source,
            )

            if verbose:
                print(f"Building extension module {name}...", file=sys.stderr)
            _run_ninja_build(
                build_directory,
                verbose,
                error_prefix=f"Error building extension '{name}'",
            )

        cpp_extension._write_ninja_file_and_build_library = (
            _write_ninja_file_and_build_library_ja
        )

        _NINJA_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]")
        _NINJA_FAIL_RE = re.compile(r"^(FAILED:|ninja: build stopped)|\berror:")

        def _run_ninja_build(
            build_directory: str, verbose: bool, error_prefix: str
        ) -> None:
            """Run ninja, rendering one self-updating progress line.

            The JIT build compiles tens of thousands of generated sources. Ninja
            already counts them, but the previous implementation captured stdout
            into a pipe and threw it away unless the build failed, so a two-hour
            compile looked identical to a hung one. Here the stream is consumed
            live: progress collapses onto a single line, compiler errors are
            echoed the moment they appear, and the whole output is retained so a
            failure still reports the real reason.
            """
            command = ["ninja"]
            num_workers = cpp_extension._get_num_workers(verbose)
            if num_workers is not None:
                command.extend(["-j", str(num_workers)])

            # error_prefix looks like: Error building extension 'libmha_bwd'
            label = (m.group(1) if (m := re.search(r"'([^']+)'", error_prefix))
                     else "ninja")
            tty = sys.stdout.isatty()
            start = time.time()
            captured: list[str] = []
            last_tick = 0.0
            done = total = 0

            sys.stdout.flush()
            sys.stderr.flush()
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=build_directory,
                env=os.environ.copy(),
                text=True,
                bufsize=1,
            )

            def _status() -> str:
                pct = f"{100.0 * done / total:5.1f}%" if total else "  ?  "
                el = int(time.time() - start)
                return (f"[{label}] {done}/{total or '?'} {pct} "
                        f"{el // 60}m{el % 60:02d}s")

            assert proc.stdout is not None
            for line in proc.stdout:
                captured.append(line)
                if m := _NINJA_PROGRESS_RE.match(line):
                    done, total = int(m.group(1)), int(m.group(2))
                    now = time.time()
                    if tty:
                        sys.stdout.write("\r\033[K" + _status())
                        sys.stdout.flush()
                    elif now - last_tick >= 30:
                        # nohup / CI: periodic lines, since \r would be noise.
                        print(_status(), flush=True)
                        last_tick = now
                    continue
                if verbose or _NINJA_FAIL_RE.search(line):
                    if tty:
                        sys.stdout.write("\r\033[K")
                    sys.stdout.write(line)
                    sys.stdout.flush()

            rc = proc.wait()
            if tty:
                sys.stdout.write("\r\033[K" + _status() + "\n")
                sys.stdout.flush()

            if rc != 0:
                raise RuntimeError(f"{error_prefix}: {''.join(captured)}")

        cpp_extension._run_ninja_build = _run_ninja_build

        logger.info("Patched ninja file generation for incremental compilation")

    except ImportError:
        logger.warning("Could not patch cpp_extension for incremental compilation")

    logger.info("Patched AITER core to support JAX-AITER configuration")
    logger.info(f"Redirected JIT build directory to: {jax_aiter_root}/build/jit")


def import_aiter_core(jax_aiter_root):
    """Import AITER's core module, patch it, and display environment information."""
    try:
        # Import AITER's core module.
        import core

        # Patch AITER core to support JAX-AITER configuration.
        patch_aiter_core(core, jax_aiter_root)

        # Print AITER environment info.
        logger.info(f"\nAITER Environment:")
        logger.info(f"  AITER_ROOT_DIR: {core.AITER_ROOT_DIR}")
        logger.info(f"  AITER_CSRC_DIR: {core.AITER_CSRC_DIR}")
        logger.info(f"  CK_DIR: {core.CK_DIR}")
        logger.info(f"  AITER_ASM_DIR: {core.AITER_ASM_DIR}")

        return core

    except ImportError as e:
        logger.error(f"Error importing AITER core: {e}")
        logger.error("Make sure AITER is properly built and hipified.")
        sys.exit(1)


def build_module(core_module, module_name, verbose=False):
    """Build a single module using AITER's build system with incremental compilation support."""
    logger.info(f"=== Building {module_name} ===")
    try:
        build_args = core_module.get_args_of_build(module_name)
        logger.info(f"Source files: {len(build_args['srcs'])}")
        if verbose:
            for src in build_args["srcs"]:
                logger.info(f"  - {src}")
            if build_args["flags_extra_hip"]:
                logger.info(f"Extra HIP flags: {build_args['flags_extra_hip']}")
            if build_args["blob_gen_cmd"]:
                logger.info(f"Blob gen cmd: {build_args['blob_gen_cmd']}")
        is_python_module = (
            False
            if module_name.startswith("lib")
            else build_args.get("is_python_module", True)
        )
        # v0.1.14 added a positional `third_party` param to build_module
        # (before `hipify`). Pass the trailing params by keyword so they bind
        # correctly: `third_party` defaults to [] (JA configs don't clone
        # 3rdparty repos at JIT time) and `hipify` stays True for JA builds.
        core_module.build_module(
            build_args["md_name"],
            build_args["srcs"],
            build_args["flags_extra_cc"],
            build_args["flags_extra_hip"],
            build_args["blob_gen_cmd"],
            build_args["extra_include"],
            build_args["extra_ldflags"],
            build_args["verbose"] or verbose,
            is_python_module,
            build_args.get("is_standalone", False),
            build_args.get("torch_exclude", False),
            third_party=build_args.get("third_party", []),
            hipify=build_args.get("hipify", True),
        )
        logger.info(f"Successfully built {module_name}")

        # Automatically copy standalone lib .so files to the parent directory for proper linking.
        if module_name.startswith("lib"):
            jit_build_dir = Path(core_module.get_user_jit_dir())
            src_so = (
                jit_build_dir / "build" / module_name / "build" / f"{module_name}.so"
            )
            dst_so = jit_build_dir / f"{module_name}.so"

            if src_so.exists():
                shutil.copy2(src_so, dst_so)
                logger.info(f"Copied {module_name}.so to {jit_build_dir}")
            else:
                logger.warning(f"Expected .so file not found at {src_so}")

        return True
    except Exception as e:
        logger.error(f"Failed to build {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main build function that orchestrates the entire build process."""

    # Parse command line arguments for build options.
    parser = argparse.ArgumentParser(description="JAX-AITER JIT Build System")
    parser.add_argument(
        "--module", type=str, help="Build only specific module(s) (comma-separated)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    logger.info("JAX-AITER JIT Build System")
    logger.info("=" * 30)

    # Setup environment and import required modules.
    jax_aiter_root, aiter_jit_dir = setup_environment()

    # Import AITER core module (now also patches it).
    core_module = import_aiter_core(jax_aiter_root)

    # Load our module configuration from JSON.
    with open(
        jax_aiter_root / "jax_aiter" / "jit" / "optCompilerConfig.json", "r"
    ) as f:
        config = json.load(f)

    # Filter modules if specific ones requested.
    modules_to_build = list(config.keys())
    if args.module:
        requested_modules = [m.strip() for m in args.module.split(",")]
        modules_to_build = [m for m in requested_modules if m in config]
        if not modules_to_build:
            logger.error(
                f"None of the requested modules found: {requested_modules}. Available: {list(config.keys())}"
            )
            return 1

    if verbose := args.verbose:
        logger.info(
            f"Modules available: {len(config)}; to process: {len(modules_to_build)}"
        )
        for module_name in modules_to_build:
            logger.info(f"  - {module_name}")

    jit_build_dir = core_module.get_user_jit_dir()
    built_modules, failed_modules = [], []

    for module_name in modules_to_build:
        if build_module(core_module, module_name, verbose):
            built_modules.append(module_name)
        else:
            failed_modules.append(module_name)

    logger.info(f"Modules built successfully: {len(built_modules)}")
    if verbose:
        for module in built_modules:
            logger.info(f" -> {module}")

    if failed_modules:
        logger.error(f"Modules failed to build: {len(failed_modules)}")
        for module in failed_modules:
            logger.error(f" -> {module}")
        return 1

    so_files = list(Path(jit_build_dir).glob("*.so"))
    logger.info(
        f"=== Generated Files ===\nJIT build directory: {jit_build_dir}\nGenerated .so files: {len(so_files)}"
    )
    if verbose:
        for so_file in so_files:
            logger.info(f"  - {so_file.name}")

    logger.info("Build completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
