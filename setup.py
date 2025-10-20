from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from pathlib import Path
import shutil

THIS_DIR = Path(__file__).parent
README_FILE = THIS_DIR / "README.md"
README = README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""

PKG_LIB_DIR = Path("jax_aiter/_lib")
PKG_HSA_DIR = Path("jax_aiter/_hsa")
# many .so files; we'll exclude aiter_build/build/**
SO_DIR = Path("build")
HSA_DIR = Path("third_party/aiter/hsa")


def _copy_libs() -> int:
    """Copy prebuilt .so files into jax_aiter/_lib/{aiter_build,jax_aiter_build}.

    Copies only top-level *.so from build/aiter_build and build/jax_aiter_build
    into jax_aiter/_lib preserving the directory names. This matches the runtime
    expectations in ja_compat.config and ffi.registry.
    """
    PKG_LIB_DIR.mkdir(parents=True, exist_ok=True)
    n = 0

    for name in ("aiter_build", "jax_aiter_build"):
        src_dir = SO_DIR / name
        if not src_dir.exists():
            print(f"Warning: source directory not found: {src_dir}, skipping")
            continue

        dst_dir = PKG_LIB_DIR / name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy only top-level .so files; registry uses dir.glob("*.so")
        for so in sorted(src_dir.glob("*.so")):
            shutil.copy2(so, dst_dir / so.name)
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
        n_libs = _copy_libs()
        self.announce(f"Copied {n_libs} .so files into {PKG_LIB_DIR}", level=3)

        n_hsa = _copy_hsa_kernels()
        self.announce(f"Copied {n_hsa} HSA kernel files into {PKG_HSA_DIR}", level=3)

        super().run()


class develop(_develop):
    """Ensure libs are copied for editable installs too."""

    def run(self):
        n_libs = _copy_libs()
        self.announce(
            f"(develop) Copied {n_libs} .so files into {PKG_LIB_DIR}", level=3
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
            "_lib/**/*.so",
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
