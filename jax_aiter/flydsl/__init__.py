# SPDX-License-Identifier: Apache-2.0
"""JAX integration for first-party FlyDSL kernels.

FlyDSL is optional. Importing :mod:`jax_aiter` does not import the compiler;
the bridge is initialized only when ``flydsl_call`` is used.
"""

from .call import flydsl_call

__all__ = ["flydsl_call"]
