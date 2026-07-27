#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lever B HLO probe: gate/up weight all-gather dtype (bf16 vs packed u8).

Builds a minimal FSDP=8 gate/up-style FP4 GEMM (a @ w^T, w = [mlp, embed] with
the contraction-K = embed axis sharded over the 'fsdp' mesh axis -- the same
sharding MaxText gives the gate/up DenseGeneral weight at pure FSDP=8) and dumps
the POST-OPTIMIZATION HLO so the FSDP weight all-gather is visible.

  flag-OFF (legacy): the rowwise weight cast forces K-replication => XLA all-
                     gathers the weight in **bf16** before the cast.
  flag-ON  (LeverB): the weight cast runs per-K-shard (unshuffled packed FP4) and
                     the GEMM partition all-gathers the **packed u8** operand.

Compile-only (jit.lower().compile()); no execution, so it fits in a tiny mem
fraction even while the box is busy. Prints + writes the all-gather lines.
"""
from __future__ import annotations

import argparse
import re
import sys


def run(args):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax_aiter.gemm_fp4 import gemm_fp4_bf16

    devs = jax.devices()
    print("jax devices:", [str(d) for d in devs])
    n = args.fsdp
    assert len(devs) >= n, f"need >={n} devices, have {len(devs)}"
    mesh = Mesh(np.asarray(devs[:n]).reshape(n), ("fsdp",))

    M, embed, mlp = args.m, args.embed, args.mlp
    a = jnp.zeros((M, embed), jnp.bfloat16)         # [tokens, embed]
    w = jnp.zeros((mlp, embed), jnp.bfloat16)        # [N=mlp, K=embed]
    a_sh = NamedSharding(mesh, P("fsdp", None))      # tokens sharded, embed full
    # wshard=k: K=embed(axis1) sharded (gate/up -> forward K-gather, Fix 1).
    # wshard=n: N=mlp(axis0) sharded  (down/O   -> dgrad N-gather,   Fix 2).
    w_sh = (NamedSharding(mesh, P("fsdp", None)) if args.wshard == "n"
            else NamedSharding(mesh, P(None, "fsdp")))

    def fwd(a, w):
        return gemm_fp4_bf16(a, w)

    def loss(a, w):
        return gemm_fp4_bf16(a, w).sum()

    fn = jax.value_and_grad(loss, argnums=1) if args.grad else fwd
    jfn = jax.jit(fn, in_shardings=(a_sh, w_sh))
    compiled = jfn.lower(a, w).compile()
    hlo = compiled.as_text()
    return hlo


def summarize(hlo, tag, out_dir):
    import os
    path = os.path.join(out_dir, f"hlo_{tag}.txt")
    with open(path, "w") as f:
        f.write(hlo)
    ag = [ln.strip() for ln in hlo.splitlines()
          if ("all-gather" in ln) and ("= " in ln)]
    print(f"\n==== {tag}: {len(ag)} all-gather lines (wrote {path}) ====")
    # bytes per dtype heuristic
    def dt_bytes(ln):
        m = re.search(r"=\s*\(?([a-z0-9]+)\[([0-9,]+)\]", ln)
        if not m:
            return None, 0
        dt, dims = m.group(1), m.group(2)
        sz = {"bf16": 2, "f32": 4, "u8": 1, "s8": 1, "f16": 2, "u32": 4}.get(dt, 0)
        n = 1
        for d in dims.split(","):
            n *= int(d)
        return dt, n * sz
    tot = {}
    for ln in ag:
        dt, b = dt_bytes(ln)
        tot[dt] = tot.get(dt, 0) + b
        print("  ", ln[:200])
    print(f"  -- all-gather bytes by dtype: {tot}")
    return ag, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fsdp", type=int, default=8)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--embed", type=int, default=4096)
    ap.add_argument("--mlp", type=int, default=14336)
    ap.add_argument("--grad", action="store_true", help="also trace backward (dgrad/wgrad)")
    ap.add_argument("--wshard", choices=["k", "n"], default="k",
                    help="k=K/embed sharded (fwd, Fix 1); n=N/mlp sharded (dgrad, Fix 2)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    hlo = run(args)
    summarize(hlo, "thisflag", args.out_dir)


if __name__ == "__main__":
    main()
