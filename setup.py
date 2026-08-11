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


# Wheel variant: "lite" is the default/PyPI package. It includes every thin FFI
# shim but drops the two multi-GB MHA JIT libraries; users add those with
# `jax-aiter-fetch-mha`. "full" is a GitHub-release convenience artifact.
JA_WHEEL_VARIANT = os.environ.get("JA_WHEEL_VARIANT", "lite").strip().lower()
IS_LITE = JA_WHEEL_VARIANT == "lite"
if JA_WHEEL_VARIANT not in {"lite", "full"}:
    raise RuntimeError(
        f"JA_WHEEL_VARIANT must be 'lite' or 'full', got {JA_WHEEL_VARIANT!r}"
    )

# Stale shim with no source -- never ship it (it would otherwise leak in).
_ALWAYS_SKIP_LIBS = {"moe_fwd_ja.so"}
# Only the large JIT libs are downloaded separately. The thin MHA FFI shims
# must be in the default wheel or fetching the JIT libs would still leave
# `import jax_aiter.mha` unusable.
_LITE_SKIP_LIBS = {
    "libmha_fwd.so",
    "libmha_bwd.so",
}

BASE_VERSION = _read_version()
# The PyPI/default package gets the public version. The optional oversized full
# artifact gets a local tag and is published on GitHub, not PyPI.
FULL_VERSION = BASE_VERSION + ("" if IS_LITE else "+full")


def _copy_libs() -> int:
    """Copy prebuilt .so files into jax_aiter/_lib/{aiter_build,jax_aiter_build}.

    Copies only top-level *.so from build/aiter_build and build/jax_aiter_build
    into jax_aiter/_lib preserving the directory names. This matches the runtime
    expectations in ja_compat.config and ffi.registry.

    Variant-aware:
      * The staging dir is wiped first so a prior (e.g. full) build can never
        leak stale libs into a later (e.g. lite) wheel.
      * ``moe_fwd_ja.so`` (no source) is always skipped.
      * The lite variant additionally skips the MHA JIT libs, but keeps the
        thin MHA FFI shims so `jax-aiter-fetch-mha` completes the install.
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
    """Copy one architecture's HSA files into jax_aiter/_hsa.

    Alpha2 supports gfx950/MI355X only, matching the available CI hardware.
    ``JA_WHEEL_ARCH`` is parameterized for a future MI300 wheel, but accepting a
    semicolon-separated fat build here would silently double the artifact.

    When the
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

    wheel_arch = os.environ.get(
        "JA_WHEEL_ARCH", os.environ.get("GPU_ARCHS", "gfx950")
    ).strip()
    if ";" in wheel_arch or wheel_arch != "gfx950":
        raise RuntimeError(
            "Alpha2 wheels are gfx950-only; set JA_WHEEL_ARCH=gfx950 "
            f"(got {wheel_arch!r})"
        )

    drop_fmha = IS_LITE and os.environ.get("JA_LITE_DROP_FMHA_HSA") == "1"
    _FMHA_MARKERS = ("fmha_v3_fwd", "fmha_v3_bwd")

    n = 0
    n_arch_skipped = 0
    n_fmha_skipped = 0
    # Keep root metadata/helpers plus the selected architecture. Do not copy
    # other GPU subtrees into a wheel we cannot test on available hardware.
    for item in HSA_DIR.rglob("*"):
        if item.is_file():
            # Calculate relative path from HSA_DIR.
            rel_path = item.relative_to(HSA_DIR)
            if len(rel_path.parts) > 1 and rel_path.parts[0].startswith("gfx"):
                if rel_path.parts[0] != wheel_arch:
                    n_arch_skipped += 1
                    continue

            if drop_fmha and any(m in rel_path.as_posix() for m in _FMHA_MARKERS):
                n_fmha_skipped += 1
                continue

            dest_file = PKG_HSA_DIR / rel_path

            # Create parent directories if needed.
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file.
            shutil.copy2(item, dest_file)
            n += 1

    if drop_fmha:
        print(
            f"[jax-aiter setup] lite HSA trim: dropped {n_fmha_skipped} fmha_v3 "
            f"kernel files, copied {n}"
        )
    print(
        f"[jax-aiter setup] wheel architecture={wheel_arch}; "
        f"skipped {n_arch_skipped} files for other architectures"
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
        # Release builds run in the project's manylinux_2_28 image and set this
        # explicitly. Local developer builds retain wheel's normal linux tag.
        release_plat = os.environ.get("JA_WHEEL_PLAT_NAME", "").strip()
        if release_plat:
            self.plat_name = release_plat


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
    # jax is a genuine runtime dependency: every op registers an XLA FFI target
    # at import. The ROCm backend (jax-rocm7-plugin / -pjrt) is deliberately NOT
    # pinned here -- it must match the ROCm the machine actually has, and pinning
    # it would fight the container's own stack. The README states which pair was
    # validated.
    install_requires=["jax", "zstandard"],
    extras_require={
        "dev": ["pytest", "pytest-rerunfailures", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            # Pulls the ~2.6 GB of flash-attention libraries the default wheel
            # omits, instead of a 2-3 hour source build.
            "jax-aiter-fetch-mha = jax_aiter.fetch_mha:main",
        ],
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
