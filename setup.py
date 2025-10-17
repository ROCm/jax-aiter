from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from pathlib import Path
import shutil
import subprocess
import os

THIS_DIR = Path(__file__).parent
README_FILE = THIS_DIR / "README.md"
README = README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""

PKG_BUILD_DIR = Path("jax_aiter/build")
PKG_HSA_DIR = Path("jax_aiter/_hsa")
SO_AITER_DIR = Path("build/aiter_build")
SO_JA_DIR = Path("build/jax_aiter_build")
HSA_DIR = Path("third_party/aiter/hsa")


def _build_libraries():
    """Build umbrella, AITER modules, and JA modules."""
    root = THIS_DIR
    env = os.environ.copy()
    
    # GPU arch selection: prefer GPU_ARCHS, fallback to GFX, else ROCM_ARCH.
    gpu_archs = env.get("GPU_ARCHS") or env.get("GFX") or env.get("ROCM_ARCH", "gfx942")
    
    print(f"Building libraries with GPU_ARCHS={gpu_archs}")
    
    # 1) Build umbrella library.
    print("Building umbrella library...")
    subprocess.run(["make"], cwd=root, check=True, env=env)
    
    # 2) Build AITER thin modules.
    print("Building AITER modules...")
    python_exe = env.get("PYTHON", "python3")
    subprocess.run([python_exe, "jax_aiter/jit/build_jit.py"], cwd=root, check=True, env=env)
    
    # 3) Build JA modules with architecture support.
    print(f"Building JA modules for architectures: {gpu_archs}")
    subprocess.run(["make", "ja_mods", f"GPU_ARCHS={gpu_archs}"], cwd=root, check=True, env=env)


def _copy_libs() -> int:
    """Copy prebuilt .so files into jax_aiter/build/{aiter_build,jax_aiter_build}."""
    # Create destination directories.
    pkg_aiter_dir = PKG_BUILD_DIR / "aiter_build"
    pkg_ja_dir = PKG_BUILD_DIR / "jax_aiter_build"
    pkg_aiter_dir.mkdir(parents=True, exist_ok=True)
    pkg_ja_dir.mkdir(parents=True, exist_ok=True)
    
    n = 0
    
    # Copy AITER modules (only top-level .so, skip build/** subdirectories).
    if SO_AITER_DIR.exists():
        for so in SO_AITER_DIR.glob("*.so"):
            shutil.copy2(so, pkg_aiter_dir / so.name)
            n += 1
    
    # Copy JA modules.
    if SO_JA_DIR.exists():
        for so in SO_JA_DIR.glob("*.so"):
            shutil.copy2(so, pkg_ja_dir / so.name)
            n += 1
    
    return n


def _copy_hsa_kernels() -> int:
    """Copy HSA kernel files (.co) from third_party/aiter/hsa into jax_aiter/_hsa."""
    if not HSA_DIR.exists():
        print(f"Warning: HSA directory not found at {HSA_DIR}, skipping kernel copy")
        return 0

    PKG_HSA_DIR.mkdir(parents=True, exist_ok=True)
    n = 0

    # Copy the entire hsa directory structure.
    for item in HSA_DIR.rglob("*"):
        if item.is_file():
            # Calculate relative path from HSA_DIR.
            rel_path = item.relative_to(HSA_DIR)
            dest_file = PKG_HSA_DIR / rel_path

            # Create parent directories if needed.
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file.
            shutil.copy2(item, dest_file)
            n += 1

    return n


class build_py(_build_py):
    def run(self):
        # Build all libraries before packaging
        self.announce("Building umbrella, AITER, and JA libraries...", level=3)
        _build_libraries()
        
        n_libs = _copy_libs()
        self.announce(f"Copied {n_libs} .so files into {PKG_BUILD_DIR}", level=3)

        n_hsa = _copy_hsa_kernels()
        self.announce(f"Copied {n_hsa} HSA kernel files into {PKG_HSA_DIR}", level=3)

        super().run()


class develop(_develop):
    """Ensure libs are copied for editable installs too."""

    def run(self):
        # Build all libraries for editable install.
        self.announce("(develop) Building umbrella, AITER, and JA libraries...", level=3)
        _build_libraries()
        
        n_libs = _copy_libs()
        self.announce(
            f"(develop) Copied {n_libs} .so files into {PKG_BUILD_DIR}", level=3
        )

        n_hsa = _copy_hsa_kernels()
        self.announce(
            f"(develop) Copied {n_hsa} HSA kernel files into {PKG_HSA_DIR}", level=3
        )

        super().run()


class bdist_wheel(_bdist_wheel):
    """Mark wheel as non-pure (platform specific) since we ship .so files."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


setup(
    name="jax-aiter",
    version="0.0.0",
    author="Ruturaj4",
    author_email="Ruturaj.Vaidya@amd.com",
    description="JAX FFI wrappers for AITER kernels",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ROCm/jax-aiter.git",
    license="MIT",
    packages=find_packages(include=["jax_aiter", "jax_aiter.*"]),
    include_package_data=True,  # picks up MANIFEST.in entries for sdist.
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "examples": ["torch"],
    },
    package_data={
        # Wheel inclusion: include the copied .so files and HSA kernels.
        "jax_aiter": [
            "build/aiter_build/*.so",
            "build/jax_aiter_build/*.so",
            "_hsa/**/*.co",
            "_hsa/**/*.csv",
            "_hsa/**/*.py",
        ],
    },
    cmdclass={
        "build_py": build_py,
        "develop": develop,
        "bdist_wheel": bdist_wheel,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
