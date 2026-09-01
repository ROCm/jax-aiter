# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Single source of truth for the jax_aiter package version.

`setup.py` parses ``__version__`` from this file (without importing the
package). The default/PyPI wheel uses this public version; the oversized
GitHub-only artifact appends ``+full``. Keep this as a plain string literal so
the parse stays trivial.
"""

__version__ = "0.1.0a2"
