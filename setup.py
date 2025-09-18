from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme_file = this_dir / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="jax-aiter",
    version="0.0.0",
    author="Ruturaj4",
    author_email="Ruturaj.Vaidya@amd.com",
    description="JAX FFI wrappers for AITER kernels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ROCm/jax-aiter.git",
    license="MIT",

    packages=find_packages(include=["jax_aiter", "jax_aiter.*"]),
    include_package_data=True,

    python_requires=">=3.10",

    install_requires=[
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "examples": [
            "torch",
        ],
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
