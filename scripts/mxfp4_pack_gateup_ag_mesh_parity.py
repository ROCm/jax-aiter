#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lever B LIVE multi-device parity: flag-ON == flag-OFF across an FSDP=8 mesh.

Runs a real gate/up-style FP4 GEMM (a @ w^T, w=[mlp,embed], K=embed sharded over
'fsdp') with value_and_grad over BOTH operands, so the forward, dgrad (dA), and
wgrad (dB) all execute on the 8-GPU mesh. Saves value + grads to an npz keyed by
the flag, so a separate compare step can prove flag-ON is byte-identical (or
cos=1.0 / max_abs=0) to flag-OFF -- the SHARDED (not just CPU) reconfirm.

One process per flag (the flag is read at import). Tiny tensors + a tiny,
non-preallocating mem fraction so it fits even on a shared box, but it executes
COLLECTIVES so it must hold the GPU lock (run when the box is idle).
"""
from __future__ import annotations
import argparse, os
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_aiter.gemm_fp4 import gemm_fp4_bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdp", type=int, default=8)
    ap.add_argument("--m", type=int, default=2048)      # M_shard = 2048/8 = 256
    ap.add_argument("--embed", type=int, default=4096)  # K_shard = 4096/8 = 512 (aligned)
    ap.add_argument("--mlp", type=int, default=14336)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = jax.devices()
    assert len(d) >= args.fsdp, f"need {args.fsdp} devices, have {len(d)}"
    mesh = Mesh(np.asarray(d[:args.fsdp]).reshape(args.fsdp), ("fsdp",))
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((args.m, args.embed)).astype(np.float32) * 0.1, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((args.mlp, args.embed)).astype(np.float32) * 0.1, jnp.bfloat16)
    ash = NamedSharding(mesh, P("fsdp", None)); wsh = NamedSharding(mesh, P(None, "fsdp"))
    a = jax.device_put(a, ash); w = jax.device_put(w, wsh)

    def loss(a, w):
        return gemm_fp4_bf16(a, w).sum()

    val, (ga, gw) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)),
                            in_shardings=(ash, wsh))(a, w)
    flag = os.environ.get("JA_FP4_PACK_GATEUP_AG", "0")
    np.savez(args.out,
             val=np.asarray(val).astype(np.float32),
             ga=np.asarray(ga).astype(np.float32),
             gw=np.asarray(gw).astype(np.float32))
    print(f"[mesh-parity] flag={flag} val={float(val):.6f} "
          f"ga.shape={tuple(ga.shape)} gw.shape={tuple(gw.shape)} "
          f"|ga|={float(jnp.linalg.norm(ga.astype(jnp.float32))):.5f} "
          f"|gw|={float(jnp.linalg.norm(gw.astype(jnp.float32))):.5f} -> wrote {args.out}")


if __name__ == "__main__":
    main()
