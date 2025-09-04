#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Build script for JAX-AITER using AITER's JIT system

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path


def setup_environment():
    """Setup environment variables for JAX-AITER and AITER"""

    # Get JA_ROOT_DIR from environment.
    if "JA_ROOT_DIR" not in os.environ:
        print("Error: JA_ROOT_DIR environment variable not set")
        print("Please set JA_ROOT_DIR to your JAX-AITER root directory")
        sys.exit(1)

    jax_aiter_root = Path(os.environ["JA_ROOT_DIR"])
    aiter_root = jax_aiter_root / "build" / "hipified_aiter"
    aiter_jit_dir = aiter_root / "aiter" / "jit"

    # Add AITER to Python path.
    sys.path.insert(0, str(aiter_jit_dir))
    sys.path.insert(0, str(aiter_jit_dir / "utils"))

    print("Environment Setup:")
    print(f"  JA_ROOT_DIR: {os.environ['JA_ROOT_DIR']}")
    print(f"  AITER JIT DIR: {aiter_jit_dir}")

    return jax_aiter_root, aiter_jit_dir


def import_aiter_core():
    """Import AITER's core module and setup AITER environment"""

    try:
        # Import AITER's core module.
        import core

        # Print AITER environment info.
        print(f"\nAITER Environment:")
        print(f"  AITER_ROOT_DIR: {core.AITER_ROOT_DIR}")
        print(f"  AITER_CSRC_DIR: {core.AITER_CSRC_DIR}")
        print(f"  CK_DIR: {core.CK_DIR}")
        print(f"  AITER_ASM_DIR: {core.AITER_ASM_DIR}")

        return core

    except ImportError as e:
        print(f"Error importing AITER core: {e}")
        print("Make sure AITER is properly built and hipified.")
        sys.exit(1)


def patch_aiter_core(core_module, jax_aiter_root):
    """Patch AITER's core module to support JA_ROOT_DIR"""

    # Add JA_ROOT_DIR to the core module's globals.
    core_module.JA_ROOT_DIR = jax_aiter_root

    # Override the get_args_of_build function to use JA config.
    original_get_args_of_build = core_module.get_args_of_build

    def get_args_of_build_ja(ops_name, exclude=[]):
        # Load JA configuration.
        config_path = jax_aiter_root / "jax_aiter" / "jit" / "optCompilerConfig.json"
        with open(config_path, "r") as f:
            our_config = json.load(f)

        # Check if the operation is in JA config.
        if ops_name in our_config:
            # Use JA config.
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

            # Convert string expressions to actual values.
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

            # Process the config.
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

            # Always add PyTorch include directories (needed for compilation)
            # Even with torch_exclude=True, we need headers for compilation.
            pytorch_includes = [f"-I{torch_site}"]
            pytorch_includes.extend(
                [f"-I{inc_dir}" for inc_dir in pytorch_include_dirs]
            )

            # Define JAX-aiter flags.
            ja_includes = [
                f"-I{jax_ffi_include}",
                f"-I{core_module.JA_ROOT_DIR}/csrc/common",
            ]

            # Add PyTorch includes to both CC and HIP flags.
            if "flags_extra_cc" not in default_config:
                default_config["flags_extra_cc"] = []
            default_config["flags_extra_cc"].extend(pytorch_includes)
            default_config["flags_extra_cc"].extend(ja_includes)

            return default_config
        else:
            # Fall back to original AITER config
            return original_get_args_of_build(ops_name, exclude)

    # Replace the function
    core_module.get_args_of_build = get_args_of_build_ja

    print("Patched AITER core to support JAX-AITER configuration")


def build_module(core_module, module_name):
    """Build a single module"""

    print(f"\n=== Building {module_name} ===")

    try:
        # Get build configuration
        build_args = core_module.get_args_of_build(module_name)

        print(f"Source files: {len(build_args['srcs'])} files")
        for src in build_args["srcs"]:
            print(f"  - {src}")

        if build_args["flags_extra_hip"]:
            print(f"Extra HIP flags: {build_args['flags_extra_hip']}")
        if build_args["blob_gen_cmd"]:
            print(f"Blob gen cmd: {build_args['blob_gen_cmd']}")

        # Build the module
        core_module.build_module(
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

        print(f"Successfully built {module_name}")
        return True

    except Exception as e:
        print(f"Failed to build {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main build function"""

    print("JAX-AITER JIT Build System")
    print("=" * 10)

    # Setup environment
    jax_aiter_root, aiter_jit_dir = setup_environment()

    # Import AITER core
    core_module = import_aiter_core()

    # Patch AITER core to support our configuration
    patch_aiter_core(core_module, jax_aiter_root)

    # Load our module configuration
    with open(
        jax_aiter_root / "jax_aiter" / "jit" / "optCompilerConfig.json", "r"
    ) as f:
        config = json.load(f)

    print(f"\nModules to build: {len(config)}")
    for module_name in config.keys():
        print(f"  - {module_name}")

    # Create build output directory
    build_dir = jax_aiter_root / "build" / "lib"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build each module
    built_modules = []
    failed_modules = []

    for module_name in config.keys():
        success = build_module(core_module, module_name)
        if success:
            built_modules.append(module_name)
        else:
            failed_modules.append(module_name)

    # Build summary
    print(f"\n=== Build Summary ===")
    print(f"Successfully built: {len(built_modules)}")
    for module in built_modules:
        print(f"* {module}")

    if failed_modules:
        print(f"\nFailed to build: {len(failed_modules)}")
        for module in failed_modules:
            print(f"* {module}")
        return 1

    # Find generated .so files
    jit_build_dir = core_module.get_user_jit_dir()
    so_files = list(Path(jit_build_dir).glob("*.so"))

    print(f"\n=== Generated Files ===")
    print(f"JIT build directory: {jit_build_dir}")
    print(f"Generated .so files: {len(so_files)}")
    for so_file in so_files:
        print(f"  - {so_file.name}")

    # Copy .so files to our build directory
    print(f"\nCopying .so files to {build_dir}")
    for so_file in so_files:
        dest = build_dir / so_file.name
        shutil.copy2(so_file, dest)

    print(f"All modules built successfully!")
    print(f"All .so files available in: {build_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
