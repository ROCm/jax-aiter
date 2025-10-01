#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Build script for JAX-AITER using AITER's JIT system.

import os
import sys
import shutil
import subprocess
import json
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
    core_module.CK_DIR = f"{core_module.bd_dir}/ck"
    Path(core_module.CK_DIR).mkdir(parents=True, exist_ok=True)
    # recopy_ck caches by function; after changing CK_DIR, clear cache:
    if hasattr(core_module, "recopy_ck"):
        core_module.recopy_ck.cache_clear()

    # Patch AITER's rename_cpp_to_cu to only copy changed files.
    # This preserves file timestamps and enables ninja incremental compilation.
    def rename_cpp_to_cu_incremental(els, dst, recurisve=False):
        """Copy files only if source is newer than destination, preserving ninja incremental compilation."""

        def should_copy_file(src_path, dst_path):
            """Check if we should copy the file based on modification time."""
            if not os.path.exists(dst_path):
                return True

            src_mtime = os.path.getmtime(src_path)
            dst_mtime = os.path.getmtime(dst_path)
            return src_mtime > dst_mtime

        def do_rename_and_mv(name, src, dst, ret):
            """
            Rename .cpp files to .cu and copy only if the source is newer than the destination.
            """
            newName = name
            if name.endswith(".cpp") or name.endswith(".cu"):
                newName = name.replace(".cpp", ".cu")
                ret.append(f"{dst}/{newName}")

            src_path = f"{src}/{name}"
            dst_path = f"{dst}/{newName}"

            # Only copy if source is newer. This is the key fix.
            if should_copy_file(src_path, dst_path):
                shutil.copy2(src_path, dst_path)

        ret = []
        for el in els:
            if not os.path.exists(el):
                logger.warning(f"---> {el} not exists!!!!!!")
                continue
            if os.path.isdir(el):
                for entry in os.listdir(el):
                    if os.path.isdir(f"{el}/{entry}"):
                        if recurisve:
                            ret += rename_cpp_to_cu_incremental(
                                [f"{el}/{entry}"], dst, recurisve
                            )
                        continue
                    do_rename_and_mv(entry, el, dst, ret)
            else:
                do_rename_and_mv(os.path.basename(el), os.path.dirname(el), dst, ret)
        return ret

    # Replace AITER's function with our incremental version.
    core_module.rename_cpp_to_cu = rename_cpp_to_cu_incremental

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
            if isinstance(value, str) and value.startswith("f'"):
                eval_globals = {
                    "os": os,
                    "subprocess": subprocess,
                    "JA_ROOT_DIR": core_module.JA_ROOT_DIR,
                    "AITER_CSRC_DIR": core_module.AITER_CSRC_DIR,
                    "CK_DIR": core_module.CK_DIR,
                    "get_asm_dir": core_module.get_asm_dir,
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

            # Get PyTorch include directories (needed for compilation).
            torch_site = str(core_module.JA_ROOT_DIR / "third_party" / "pytorch")
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

            # Process the config values through eval.
            processed_config = {}
            for key, value in config.items():
                processed_config[key] = eval_config_value(value)

            # Merge with defaults.
            d_opt_build_args.update(processed_config)

            # Always add PyTorch include directories (needed for compilation).
            # Even with torch_exclude=True, we need headers for compilation.
            pytorch_includes = [f"-I{torch_site}"]
            pytorch_includes.extend(
                [f"-I{inc_dir}" for inc_dir in pytorch_include_dirs]
            )

            # Define JAX-aiter specific include flags.
            ja_includes = [
                f"-I{jax_ffi_include}",
                f"-I{core_module.JA_ROOT_DIR}/csrc/common",
            ]

            # Add PyTorch includes to both CC and HIP flags.
            if "flags_extra_cc" not in d_opt_build_args:
                d_opt_build_args["flags_extra_cc"] = []
            d_opt_build_args["flags_extra_cc"].extend(pytorch_includes)
            d_opt_build_args["flags_extra_cc"].extend(ja_includes)

            # Add jax-aiter library linking flags.
            d_opt_build_args["extra_ldflags"].extend(
                [
                    "-Wl,--no-as-needed",
                    f"-L{core_module.JA_ROOT_DIR}/build/aiter_build",
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
                        "-lc10_hip" if cpp_extension.IS_HIP_EXTENSION else "-lc10_cuda"
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

            return sorted(extra_ldflags)

        cpp_extension._prepare_ldflags = _prepare_ldflags_ja

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
        ) -> None:
            cpp_extension.verify_ninja_availability()

            compiler = cpp_extension.get_cxx_compiler()
            cpp_extension.get_compiler_abi_compatibility_and_version(
                compiler, torch_exclude
            )
            if with_cuda is None:
                with_cuda = any(map(_is_cuda_file, sources))
            extra_ldflags = cpp_extension._prepare_ldflags(
                extra_ldflags or [], with_cuda, verbose, is_standalone, torch_exclude
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

        def _run_ninja_build(
            build_directory: str, verbose: bool, error_prefix: str
        ) -> None:
            command = ["ninja"]
            num_workers = cpp_extension._get_num_workers(verbose)
            if num_workers is not None:
                command.extend(["-j", str(num_workers)])
            env = os.environ.copy()

            try:
                sys.stdout.flush()
                sys.stderr.flush()
                stdout_fileno = 1
                subprocess.run(
                    command,
                    stdout=stdout_fileno if verbose else subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=build_directory,
                    check=True,
                    env=env,
                )
            except subprocess.CalledProcessError as e:
                # Python 2 and 3 compatible way of getting the error object.
                _, error, _ = sys.exc_info()
                # error.output contains the stdout and stderr of the build attempt.
                message = error_prefix
                # `error` is a CalledProcessError (which has an `output`) attribute, but
                # mypy thinks it's Optional[BaseException] and doesn't narrow.
                if hasattr(error, "output") and error.output:
                    message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"
                raise RuntimeError(message) from e

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
        # libmha_ modules aren't python.
        is_python_module = (
            False
            if module_name in ("libmha_fwd", "libmha_bwd")
            else build_args.get("is_python_module", True)
        )
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
            build_args.get("hipify", True),
        )
        logger.info(f"Successfully built {module_name}")
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
