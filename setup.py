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
# many .so files; we'll exclude aiter_build/build/**
SO_DIR = Path("build/aiter_build")


def _should_skip(p: Path) -> bool:
    """Skip files in aiter_build/build/** subdirectory"""
    parts = p.parts
    if "aiter_build" in parts:
        i = parts.index("aiter_build")
        if "build" in parts[i + 1 :]:
            return True
    return False


def _copy_libs() -> int:
    """Copy prebuilt .so files into jax_aiter/_lib, excluding aiter_build/build/**."""
    PKG_LIB_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for so in SO_DIR.rglob("*.so"):
        if _should_skip(so):
            continue
        shutil.copy2(so, PKG_LIB_DIR / so.name)
        n += 1
    return n


class build_py(_build_py):
    def run(self):
        n = _copy_libs()
        self.announce(f"Copied {n} .so files into {PKG_LIB_DIR}", level=3)
        super().run()


class develop(_develop):
    """Ensure libs are copied for editable installs too."""

    def run(self):
        n = _copy_libs()
        self.announce(f"(develop) Copied {n} .so files into {PKG_LIB_DIR}", level=3)
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
    include_package_data=True,  # picks up MANIFEST.in entries for sdist
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "examples": ["torch"],
    },
    package_data={
        # Wheel inclusion: include the copied .so files
        "jax_aiter": ["_lib/*.so"],
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
