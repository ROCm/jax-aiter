# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's physical paged-KV kernel ABI.

This is the *vendor* half of a deliberate two-layer split. The neutral, semantic
layer -- pool geometry and per-step page tables -- is owned by the control plane
and knows nothing about packing, strides, or kernel shapes. This module owns the
exact physical contract AITER's kernels require, and the conversion between the
two happens in the execution layer.

Keeping them apart is what lets a vendor-neutral ``gpu_paged`` attention path
exist at all: a control plane that imported this file could never have FlashInfer
as a peer backend.

v1 scope, matching what the M1/M2 kernels were bridged against:
  * NHD only, K and V in separate pools, no x-packing
  * ``[num_pages, tokens_per_page, num_kv_heads, head_dim]``, which is both what
    ``reshape_and_cache_flash`` writes and what ``paged_attention_ragged`` reads
  * unquantised pools (``kv_cache_dtype="auto"``)

Anything beyond that -- x-packed K, the 5-D shuffled V form, MLA's fused tensor,
fp8 pools -- changes this class and nothing above it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Bumped whenever the physical contract changes in a way a caller must notice.
AITER_PAGED_ABI_VERSION = 1

_ITEMSIZE = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float8_e4m3fnuz": 1,
    "float8_e4m3fn": 1,
}


@dataclasses.dataclass(frozen=True)
class AiterPagedAttentionAbiV1:
    """Physical shapes, forms and strides for AITER's paged kernels.

    Attributes:
        k_form: ``"nhd"`` is the only v1 value. ``"x_packed"`` and ``"hnd"``
            exist in AITER and are deferred until numerics are settled.
        v_form: ``"nhd"`` in v1, so K and V share a shape and therefore a stride
            triple. AITER also accepts ``"hdb"`` and a 5-D shuffled form.
        packing_x: ``16 // itemsize`` once x-packing lands; 1 means unpacked.
        sentinel_page: page id reserved as a padding target and never allocated.
        kv_cache_dtype: AITER's dtype selector string, ``"auto"`` for unquantised.
    """

    version: int = AITER_PAGED_ABI_VERSION
    k_form: str = "nhd"
    v_form: str = "nhd"
    packing_x: int = 1
    sentinel_page: int = 0
    kv_cache_dtype: str = "auto"

    def __post_init__(self):
        if self.k_form != "nhd" or self.v_form != "nhd":
            raise ValueError(
                f"v1 supports NHD only, got k_form={self.k_form!r} "
                f"v_form={self.v_form!r}"
            )
        if self.packing_x != 1:
            raise ValueError(
                f"v1 is unpacked; packing_x must be 1, got {self.packing_x}"
            )

    # -- shapes ------------------------------------------------------------

    def pool_shape(self, layout) -> tuple[int, int, int, int]:
        """Per-shard shape of one pool (K or V), for one layer."""
        return (
            layout.num_pages,
            layout.tokens_per_page,
            layout.heads_per_shard(),
            layout.head_dim,
        )

    def new_kv_shape(self, layout, num_tokens: int) -> tuple[int, int, int]:
        """Shape of the incoming K or V for ``append_kv``."""
        return (num_tokens, layout.heads_per_shard(), layout.head_dim)

    def query_shape(self, layout, num_seqs: int, num_q_heads: int):
        """Shape of Q for paged decode: one token per sequence."""
        return (num_seqs, num_q_heads, layout.head_dim)

    # -- strides -----------------------------------------------------------

    def strides(self, layout) -> tuple[int, int, int]:
        """``(kv_block_stride, kv_head_stride, kv_seq_stride)`` in elements.

        The HIP kernels are layout-agnostic and driven purely by these, so they
        are computed from the pool shape rather than assumed. For NHD
        ``[num_pages, tokens_per_page, num_kv_heads, head_dim]``:

          * block stride advances one whole page,
          * seq stride advances one token within a page,
          * head stride advances one KV head.
        """
        heads = layout.heads_per_shard()
        head_dim = layout.head_dim
        kv_seq_stride = heads * head_dim
        kv_block_stride = layout.tokens_per_page * kv_seq_stride
        kv_head_stride = head_dim
        return (kv_block_stride, kv_head_stride, kv_seq_stride)

    def q_stride(self, num_q_heads: int, head_dim: int) -> int:
        return num_q_heads * head_dim

    # -- sizing ------------------------------------------------------------

    def bytes_per_page(self, layout) -> int:
        itemsize = _ITEMSIZE.get(layout.dtype)
        if itemsize is None:
            itemsize = np.dtype(layout.dtype).itemsize
        return layout.tokens_per_page * layout.heads_per_shard() * layout.head_dim * itemsize

    def validate(self, layout, num_q_heads: int) -> None:
        """Reject configurations the kernels cannot serve, at startup.

        Failing here is much cheaper than a wrong-numerics debug later.
        """
        if num_q_heads % layout.heads_per_shard() != 0:
            raise ValueError(
                f"num_q_heads {num_q_heads} must be a multiple of KV heads per "
                f"shard {layout.heads_per_shard()}: GQA ratio must be integral"
            )
        if layout.head_dim not in (64, 128, 256):
            raise ValueError(
                f"head_dim {layout.head_dim} is outside AITER's supported set. "
                f"Note the ASM paged-attention path asserts head_dim == 128."
            )
        if layout.tokens_per_page not in (16, 32):
            raise ValueError(
                f"tokens_per_page {layout.tokens_per_page} unsupported; "
                f"AITER's paged kernels are compiled for 16 and 32."
            )
        if layout.dtype not in ("bfloat16", "float16"):
            raise ValueError(
                f"v1 pools are bf16 or fp16, got {layout.dtype!r}"
            )
