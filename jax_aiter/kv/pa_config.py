# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Naming and sizing for aiter's ahead-of-time compiled paged-attention kernels.

Single source of truth shared by ``scripts/gen_pa_ragged.py``, which renders and
names the kernels, and ``jax_aiter.ops.paged_attention``, which calls them. If
these two ever disagreed the build would emit a symbol under a name the handler
never asks for, so they must derive the name from the same code.

Why we call the generated symbol directly rather than aiter's C++ wrapper:
``aiter::paged_attention_ragged`` forwards to the generated function through an
unchecked varargs call whose argument order has drifted from the template it is
supposed to match -- it passes 23 arguments in a different order than the 24 the
generated entry declares, so scalars land in pointer slots. Only aiter's Python
path is consistent with the template. Binding the generated symbol ourselves is
both simpler and the only correct option today.
"""

from __future__ import annotations

import hashlib

# aiter's C++ type spellings, as chosen in pa_ragged.py's torch wrapper.
DTYPE_BF16 = "__hip_bfloat16"
DTYPE_FP16 = "_Float16"

_JAX_DTYPE_TO_AITER = {
    "bfloat16": DTYPE_BF16,
    "float16": DTYPE_FP16,
}

# aiter splits a sequence into partitions of this size; it also sets how much
# workspace the kernel needs.
PARTITION_SIZE = 256
WARP_SIZE = 64  # gfx9

# The nine values pa_ragged.cpp joins, in order, to name a configuration. Kept
# here because the folder the prebuild writes and the folder the handler opens
# have to agree exactly.
_NAME_ARGS = (
    "gqa_ratio",
    "head_size",
    "npar_loops",
    "dtype",
    "kv_dtype",
    "kv_cache_dtype",
    "out_dtype",
    "block_size",
    "alibi_enabled",
)


def aiter_dtype_name(dtype) -> str:
    """Map a JAX/numpy dtype to aiter's C++ type spelling."""
    import numpy as np

    name = np.dtype(dtype).name
    try:
        return _JAX_DTYPE_TO_AITER[name]
    except KeyError:
        raise ValueError(
            f"paged attention supports bfloat16 and float16, got {name}"
        ) from None


def max_num_partitions(max_seq_len: int) -> int:
    return -(-int(max_seq_len) // PARTITION_SIZE)


def npar_loops(max_seq_len: int) -> int:
    """``ceil(max_num_partitions / warpSize)``, part of the configuration key."""
    return -(-max_num_partitions(max_seq_len) // WARP_SIZE)


def make_config(
    *,
    gqa_ratio: int,
    head_size: int,
    max_seq_len: int,
    dtype,
    block_size: int,
    kv_cache_dtype: str = "auto",
    alibi_enabled: bool = False,
) -> dict:
    """Build the configuration dict the name is derived from."""
    name = aiter_dtype_name(dtype)
    return {
        "gqa_ratio": int(gqa_ratio),
        "head_size": int(head_size),
        "npar_loops": npar_loops(max_seq_len),
        "dtype": name,
        "kv_dtype": name,
        "kv_cache_dtype": kv_cache_dtype,
        "out_dtype": name,
        "block_size": int(block_size),
        "alibi_enabled": "true" if alibi_enabled else "false",
    }


def func_name(config: dict) -> str:
    """Reproduce ``get_default_func_name("pa_ragged", args)`` from aiter's utils.h.

    md5 over the lowercased, underscore-joined nine values. Note aiter's Python
    ``compile_template_op`` would hash all thirteen template kwargs if allowed to
    pick a default, which is why the prebuild passes this name explicitly.
    """
    signature = "_".join(str(config[k]).lower() for k in _NAME_ARGS)
    digest = hashlib.md5(signature.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"pa_ragged_{digest}"


def workspace_bytes(num_seqs: int, num_heads: int, head_dim: int,
                    max_seq_len: int, dtype) -> int:
    """Bytes the kernel needs for exp_sums, max_logits and tmp_out.

    The generated entry carves the buffer into three consecutive regions: two
    float32 ``[num_seqs, num_heads, max_num_partitions]`` arrays followed by
    ``tmp_out`` in the compute dtype with a trailing ``head_dim``.
    """
    import numpy as np

    parts = max_num_partitions(max_seq_len)
    elems = num_seqs * num_heads * parts
    itemsize = np.dtype(dtype).itemsize
    return 2 * 4 * elems + itemsize * elems * head_dim
