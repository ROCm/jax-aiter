# SPDX-License-Identifier: Apache-2.0
"""JAX integration for AITER Triton kernels via HSACO FFI.

Importing :mod:`jax_aiter` does not import Triton. The compiler and HIP
trampoline initialize only when a Triton op is lowered.
"""

from .call import triton_call

__all__ = ["triton_call"]
