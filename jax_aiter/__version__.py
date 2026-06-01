# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Single source of truth for the jax_aiter package version.

`setup.py` parses ``__version__`` from this file (without importing the
package) and appends a ``+lite`` local-version tag for the lite wheel
variant. Keep this as a plain string literal so the parse stays trivial.
"""

__version__ = "0.1.0a0"
