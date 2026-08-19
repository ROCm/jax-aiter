#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exploratory: dump lowered and compiled HLO for the M0 aliasing probe.

Used to pin down the exact textual markers the aliasing assertions key on, so
they are written against real compiler output rather than guessed spellings.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.ops.kv_alias_probe import kv_alias_probe

POOL_ROWS, ROW_ELEMS, N_ROWS = 64, 128, 8


def step(pool, row_idx, vals):
    return kv_alias_probe(pool, row_idx, vals)


def main():
    dev = jax.devices()[0]
    print("device:", dev)

    pool = jnp.zeros((POOL_ROWS, ROW_ELEMS), dtype=jnp.float32)
    row_idx = jnp.arange(N_ROWS, dtype=jnp.int32)
    vals = jnp.ones((N_ROWS, ROW_ELEMS), dtype=jnp.float32)

    jitted = jax.jit(step, donate_argnums=(0,))
    lowered = jitted.lower(pool, row_idx, vals)

    print("\n############ LOWERED (StableHLO) ############")
    print(lowered.as_text())

    compiled = lowered.compile()
    print("\n############ COMPILED (HLO) ############")
    print(compiled.as_text())

    print("\n############ MEMORY ANALYSIS ############")
    try:
        ma = compiled.memory_analysis()
        for f in ("argument_size_in_bytes", "output_size_in_bytes",
                  "temp_size_in_bytes", "alias_size_in_bytes",
                  "host_temp_size_in_bytes"):
            print(f"  {f}: {getattr(ma, f, None)}")
    except Exception as e:
        print("  memory_analysis unavailable:", e)

    print("\n############ RUN ############")
    out = jitted(pool, row_idx, vals)
    out.block_until_ready()
    got = np.asarray(out)
    print("  written rows all 1.0:", bool(np.all(got[:N_ROWS] == 1.0)))
    print("  untouched rows all 0.0:", bool(np.all(got[N_ROWS:] == 0.0)))

    print("\n############ MEMORY STATS KEYS ############")
    ms = dev.memory_stats()
    for k in sorted(ms):
        print(f"  {k}: {ms[k]}")


if __name__ == "__main__":
    main()
