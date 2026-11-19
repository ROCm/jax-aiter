# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet via flash-linear-attention Triton kernels.

Default-off. JAX owns the VJP. Does not call Torch autograd wrappers.
"""

from .chunk import chunk_gated_delta_rule_fla

__all__ = ["chunk_gated_delta_rule_fla"]
