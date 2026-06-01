from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from pathlib import Path
import os
import re
import shutil

THIS_DIR = Path(__file__).parent
README_FILE = THIS_DIR / "README.md"
README = README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""

PKG_LIB_DIR = Path("jax_aiter/_lib")
PKG_HSA_DIR = Path("jax_aiter/_hsa")
# many .so files; we'll exclude aiter_build/build/**
SO_DIR = Path("build")
HSA_DIR = Path("third_party/aiter/hsa")


def _read_version() -> str:
    """Parse __version__ from jax_aiter/__version__.py without importing it.

    Importing the package at build time would pull in heavy runtime deps
    (jax, ctypes lib loading); a regex parse keeps the build hermetic.
    """
    version_file = THIS_DIR / "jax_aiter" / "__version__.py"
    text = version_file.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {version_file}")
    return match.group(1)


# Wheel variant: "full" (default -- preserves `pip install .` for CI/dev,
# ships everything) or "lite" (drops the MHA libs + thin shims).
JA_WHEEL_VARIANT = os.environ.get("JA_WHEEL_VARIANT", "full").strip().lower()
IS_LITE = JA_WHEEL_VARIANT == "lite"

# Stale shim with no source -- never ship it (it would otherwise leak in).
_ALWAYS_SKIP_LIBS = {"moe_fwd_ja.so"}
# MHA libs + thin FFI shims: dropped from the lite wheel.
_LITE_SKIP_LIBS = {
    "libmha_fwd.so",
    "libmha_bwd.so",
    "mha_fwd_ja.so",
    "mha_bwd_ja.so",
}

BASE_VERSION = _read_version()
# Lite wheels carry a `+lite` PEP 440 local-version tag; full carries none.
FULL_VERSION = BASE_VERSION + ("+lite" if IS_LITE else "")


def _copy_libs() -> int:
    """Copy prebuilt .so files into jax_aiter/_lib/{aiter_build,jax_aiter_build}.

    Copies only top-level *.so from build/aiter_build and build/jax_aiter_build
    into jax_aiter/_lib preserving the directory names. This matches the runtime
    expectations in ja_compat.config and ffi.registry.

    Variant-aware:
      * The staging dir is wiped first so a prior (e.g. full) build can never
        leak stale libs into a later (e.g. lite) wheel.
      * ``moe_fwd_ja.so`` (no source) is always skipped.
      * The lite variant additionally skips the MHA libs + thin shims.
    """
    # Clear the staging dir first to prevent stale-lib leak across variants.
    shutil.rmtree(PKG_LIB_DIR, ignore_errors=True)
    PKG_LIB_DIR.mkdir(parents=True, exist_ok=True)

    skip = set(_ALWAYS_SKIP_LIBS)
    if IS_LITE:
        skip |= _LITE_SKIP_LIBS

    copied, skipped = [], []
    for name in ("aiter_build", "jax_aiter_build"):
        src_dir = SO_DIR / name
        if not src_dir.exists():
            print(f"Warning: source directory not found: {src_dir}, skipping")
            continue

        dst_dir = PKG_LIB_DIR / name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy only top-level .so files; registry uses dir.glob("*.so")
        for so in sorted(src_dir.glob("*.so")):
            if so.name in skip:
                skipped.append(so.name)
                continue
            shutil.copy2(so, dst_dir / so.name)
            copied.append(so.name)

    print(
        f"[jax-aiter setup] variant={JA_WHEEL_VARIANT} "
        f"copied {len(copied)} libs, skipped {len(skipped)}"
    )
    if copied:
        print(f"[jax-aiter setup]   copied:  {sorted(copied)}")
    if skipped:
        print(f"[jax-aiter setup]   skipped: {sorted(skipped)}")
    return len(copied)


def _copy_hsa_kernels() -> int:
    """Copy HSA kernel files (.co) from third_party/aiter/hsa into jax_aiter/_hsa.

    By default the entire hsa/ tree is copied (full and lite alike). When the
    lite variant is built AND ``JA_LITE_DROP_FMHA_HSA=1`` is set, the fused
    MHA v3 kernel subtrees (``fmha_v3_fwd`` / ``fmha_v3_bwd``) are skipped to
    trim a further ~30-80 MiB. This drop is opt-in because it is irreversible
    at install time.
    """
    if not HSA_DIR.exists():
        print(f"Warning: HSA directory not found at {HSA_DIR}, skipping kernel copy")
        return 0

    # Clear the staging dir first to prevent a stale fmha subtree from a prior
    # full build leaking into a later lite wheel.
    shutil.rmtree(PKG_HSA_DIR, ignore_errors=True)
    PKG_HSA_DIR.mkdir(parents=True, exist_ok=True)

    drop_fmha = IS_LITE and os.environ.get("JA_LITE_DROP_FMHA_HSA") == "1"
    _FMHA_MARKERS = ("fmha_v3_fwd", "fmha_v3_bwd")

    n = 0
    n_skipped = 0
    # Copy the entire hsa directory structure.
    for item in HSA_DIR.rglob("*"):
        if item.is_file():
            # Calculate relative path from HSA_DIR.
            rel_path = item.relative_to(HSA_DIR)

            if drop_fmha and any(m in rel_path.as_posix() for m in _FMHA_MARKERS):
                n_skipped += 1
                continue

            dest_file = PKG_HSA_DIR / rel_path

            # Create parent directories if needed.
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file.
            shutil.copy2(item, dest_file)
            n += 1

    if drop_fmha:
        print(
            f"[jax-aiter setup] lite HSA trim: dropped {n_skipped} fmha_v3 "
            f"kernel files, copied {n}"
        )
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
    version=FULL_VERSION,
    author="Ruturaj4",
    author_email="Ruturaj.Vaidya@amd.com",
    description="JAX FFI wrappers for AITER kernels",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ROCm/jax-aiter.git",
    license="MIT",
    packages=find_packages(include=["jax_aiter", "jax_aiter.*"]),
    include_package_data=True,  # picks up MANIFEST.in entries for sdist.
    python_requires="~=3.12",
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
