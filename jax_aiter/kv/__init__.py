# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged-KV support for jax-aiter: kernel ABI and the in-place mutation contract."""

from .abi import (  # noqa: F401
    AITER_PAGED_ABI_VERSION,
    AiterPagedAttentionAbiV1,
)
from .aliasing import (  # noqa: F401
    AliasContractError,
    AliasEvidence,
    assert_flat_live_memory,
    assert_in_place_mutation,
    collect_alias_evidence,
    donation_warnings_as_errors,
    live_bytes,
    pool_shard_pointers,
)
