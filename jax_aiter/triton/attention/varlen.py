# SPDX-License-Identifier: Apache-2.0
"""Variable-length flash attention using AITER Triton kernels."""

from __future__ import annotations

from functools import partial
import os

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from ..call import triton_call
from ..compile import prepare_aiter_triton

_PTR_BF16 = (
    "q_ptr",
    "k_ptr",
    "v_ptr",
    "out_ptr",
    "alibi_slopes_ptr",
    "s_dmask_ptr",
    "dropout_mask_ptr",
    "sink_ptr",
    "descale_q_ptr",
    "descale_k_ptr",
    "descale_v_ptr",
)
_PTR_FP32 = ("softmax_lse_ptr",)
_PTR_I32 = ("cu_seqlens_q", "cu_seqlens_k")
_FP32 = ("sm_scale", "dropout_p")


def _cdiv(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _pointer_type(dtype) -> str:
    if dtype == jnp.bfloat16:
        return "*bf16"
    if dtype == jnp.float16:
        return "*fp16"
    if dtype == jnp.float32:
        return "*fp32"
    if dtype == jnp.int32:
        return "*i32"
    raise TypeError(f"Unsupported Triton buffer dtype {dtype}")


def _fwd_runtime_types(q_dtype) -> dict[str, str]:
    prepare_aiter_triton()
    from aiter.ops.triton._triton_kernels.attention.mha import _attn_fwd

    types: dict[str, str] = {}
    constexpr_idx = set(_attn_fwd.constexprs)
    none_tensors = {
        "descale_q_ptr",
        "descale_k_ptr",
        "descale_v_ptr",
        "alibi_slopes_ptr",
        "s_dmask_ptr",
        "dropout_mask_ptr",
        "sink_ptr",
    }
    for index, name in enumerate(_attn_fwd.arg_names):
        if index in constexpr_idx or name in none_tensors:
            continue
        if name in _PTR_I32:
            types[name] = "*i32"
        elif name in _PTR_FP32:
            types[name] = "*fp32"
        elif name in _PTR_BF16:
            types[name] = _pointer_type(q_dtype) if name in (
                "q_ptr",
                "k_ptr",
                "v_ptr",
                "out_ptr",
            ) else "*fp32"
        elif name in _FP32:
            types[name] = "fp32"
        else:
            types[name] = "i32"
    return types


def _kinds_for(runtime_types: dict[str, str], pointer_names: set[str]) -> list[str]:
    return [
        "buffer" if name in pointer_names or typ.startswith("*") else "scalar"
        for name, typ in runtime_types.items()
    ]


def _input_buffers(runtime_types, kinds, output_names, values_by_name):
    outputs = set(output_names)
    arrays = []
    for (name, _typ), kind in zip(runtime_types.items(), kinds):
        if kind != "buffer" or name in outputs:
            continue
        arrays.append(values_by_name[name])
    return arrays


def _thd_strides(tokens: int, heads: int, dim: int) -> tuple[int, int, int, int]:
    # JAX C-contiguous [T, H, D] → (batch=0, head, token, dim)
    return (0, dim, heads * dim, 1)


def _attn_fwd_config(*, dtype, has_pe: bool):
    prepare_aiter_triton()
    from aiter.ops.triton._triton_kernels.attention.mha import _get_config
    import torch

    torch_dtype = torch.bfloat16 if dtype == jnp.bfloat16 else torch.float16
    return _get_config(False, torch_dtype, has_pe=has_pe)


def _fwd(q, k, v, cu_q, cu_k, max_sq, max_sk, softmax_scale, causal):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Triton varlen MHA expects THD q/k/v")
    if not causal:
        raise NotImplementedError("v1 Triton varlen MHA is causal-only")
    total_q, hq, d_qk = q.shape
    _total_k, hk, _d_qk_k = k.shape
    dv = v.shape[-1]
    if d_qk != k.shape[-1]:
        raise ValueError("Q and K head dims must match")
    pe = d_qk - dv
    if pe < 0:
        raise ValueError("QK head dim must be >= V head dim")

    batch = int(cu_q.shape[0]) - 1
    cfg = _attn_fwd_config(dtype=q.dtype, has_pe=pe > 0)
    block_m = int(cfg["BLOCK_M"])
    block_n = int(cfg["BLOCK_N"])
    grid = (batch * hq * _cdiv(max_sq, block_m), 1, 1)
    q_strides = _thd_strides(total_q, hq, d_qk)
    k_strides = _thd_strides(k.shape[0], hk, k.shape[-1])
    v_strides = _thd_strides(v.shape[0], hk, dv)
    o_strides = _thd_strides(total_q, hq, dv)
    lse_h, lse_m = 1, hq

    from aiter.ops.triton._triton_kernels.attention.mha import _attn_fwd

    runtime_types = _fwd_runtime_types(q.dtype)
    kinds = _kinds_for(runtime_types, set())
    values = {
        "q_ptr": q,
        "k_ptr": k,
        "v_ptr": v,
        "out_ptr": None,
        "softmax_lse_ptr": None,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
    }
    scalars = {
        "stride_qz_in": q_strides[0],
        "stride_qh_in": q_strides[1],
        "stride_qm_in": q_strides[2],
        "stride_qk_in": q_strides[3],
        "stride_kz_in": k_strides[0],
        "stride_kh_in": k_strides[1],
        "stride_kn_in": k_strides[2],
        "stride_kk_in": k_strides[3],
        "stride_vz_in": v_strides[0],
        "stride_vh_in": v_strides[1],
        "stride_vn_in": v_strides[2],
        "stride_vk_in": v_strides[3],
        "stride_descale_q_z_in": 0,
        "stride_descale_k_z_in": 0,
        "stride_descale_v_z_in": 0,
        "stride_oz_in": o_strides[0],
        "stride_oh_in": o_strides[1],
        "stride_om_in": o_strides[2],
        "stride_on_in": o_strides[3],
        "stride_alibi_z_in": 0,
        "stride_alibi_h_in": 0,
        "stride_sd_z_in": 0,
        "stride_sd_h_in": 0,
        "stride_sd_m_in": 0,
        "stride_sd_n_in": 0,
        "stride_lse_z_in": 0,
        "stride_lse_h_in": lse_h,
        "stride_lse_m_in": lse_m,
        "sm_scale": float(softmax_scale),
        "dropout_p": 0.0,
        "philox_seed": 0,
        "philox_offset_base_in": 0,
        "SEQLEN_Q": int(max_sq),
        "SEQLEN_K": int(max_sk),
        "BATCH": batch,
    }
    constexprs = {
        "IS_CAUSAL": True,
        "NUM_Q_HEADS": int(hq),
        "NUM_K_HEADS": int(hk),
        "PRELOAD_V": bool(cfg["PRELOAD_V"]),
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_DMODEL": int(dv),
        "BLOCK_DMODEL_POW2": _block_pow2(dv),
        "BLOCK_DMODEL_PE": 0 if pe == 0 else _next_pow2(pe),
        "RETURN_SCORES": False,
        "ENABLE_DROPOUT": False,
        "IS_FP8": False,
        "FP8_MAX": 0,
        "VARLEN": True,
        "NUM_XCD": 8,
        "USE_INT64_STRIDES": True,
        "ENABLE_SINK": False,
        "SLIDING_WINDOW": 0,
        "HEAD_STRIDE_ALIGNED_8": (
            q_strides[1] % 8 == 0
            and k_strides[1] % 8 == 0
            and v_strides[1] % 8 == 0
        ),
        "descale_q_ptr": None,
        "descale_k_ptr": None,
        "descale_v_ptr": None,
        "alibi_slopes_ptr": None,
        "s_dmask_ptr": None,
        "dropout_mask_ptr": None,
        "sink_ptr": None,
    }
    if pe > 0 and (
        dv != constexprs["BLOCK_DMODEL_POW2"]
        or pe != constexprs["BLOCK_DMODEL_PE"]
    ):
        raise ValueError("MLA PE requires unpadded power-of-two NOPE and PE dims")

    out_shape = (
        jax.ShapeDtypeStruct(q.shape[:-1] + (dv,), q.dtype),
        jax.ShapeDtypeStruct((total_q, hq), jnp.float32),
    )
    arrays = _input_buffers(
        runtime_types, kinds, ("out_ptr", "softmax_lse_ptr"), values
    )
    out, lse = triton_call(
        *arrays,
        fn=_attn_fwd,
        runtime_types=runtime_types,
        constexprs=constexprs,
        kinds=kinds,
        scalars=scalars,
        out_shape=out_shape,
        output_names=("out_ptr", "softmax_lse_ptr"),
        grid=grid,
        options={
            "num_warps": int(cfg["num_warps"]),
            "num_stages": int(cfg["num_stages"]),
            "waves_per_eu": int(cfg["waves_per_eu"]),
        },
    )
    return out, lse


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _block_pow2(n: int) -> int:
    return max(_next_pow2(n), 16)


def _bwd_config():
    prepare_aiter_triton()
    from aiter.ops.triton._triton_kernels.attention.mha_onekernel_bwd import (
        _get_config,
    )

    return _get_config()


def _runtime_types_from_fn(fn, *, pointer_overrides: dict[str, str]):
    types: dict[str, str] = {}
    constexpr_idx = set(fn.constexprs)
    skip = set(fn.arg_names[i] for i in constexpr_idx)
    for index, name in enumerate(fn.arg_names):
        if name in skip:
            continue
        if name in pointer_overrides:
            types[name] = pointer_overrides[name]
        elif name in ("sm_scale", "dropout_p"):
            types[name] = "fp32"
        elif name[0].isupper() and name not in ("HQ", "HK"):
            # Optional tensor specialized to None at compile; skip.
            continue
        elif name.endswith("_ptr"):
            continue
        else:
            types[name] = "i32"
    return types


def _bwd(
    q, k, v, out, lse, cu_q, cu_k, dout, max_sq, max_sk, softmax_scale,
    *, lse_head_major=False,
):
    """One-kernel varlen backward.

    ``lse`` is ``[total_q, Hq]`` as written by this module's forward, or
    ``[Hq, total_q]`` (``lse_head_major=True``) as written by the AITER
    unified varlen forward. Delta is allocated in the same layout because the
    kernel addresses both through one set of strides.
    """
    hq = q.shape[1]
    hk = k.shape[1]
    dv = v.shape[-1]
    pe = q.shape[-1] - dv
    total_q = q.shape[0]
    if lse_head_major:
        if lse.shape != (hq, total_q):
            raise ValueError(f"head-major LSE must be {(hq, total_q)}, got {lse.shape}")
        delta_strides = (0, total_q, 1)
    else:
        if lse.shape != (total_q, hq):
            raise ValueError(f"token-major LSE must be {(total_q, hq)}, got {lse.shape}")
        delta_strides = (0, 1, hq)
    batch = int(cu_q.shape[0]) - 1
    cfg = _bwd_config()
    pre_block = int(cfg["preprocess_kernel"]["PRE_BLOCK"])
    one = cfg["onekernel"]
    q_dtype = q.dtype
    ptr = _pointer_type(q_dtype)

    prepare_aiter_triton()
    from aiter.ops.triton._triton_kernels.attention.mha_onekernel_bwd import (
        _bwd_preprocess,
        bwd_kernel_causal,
    )

    o_strides = _thd_strides(out.shape[0], hq, dv)
    do_strides = _thd_strides(dout.shape[0], hq, dv)

    pre_types = _runtime_types_from_fn(
        _bwd_preprocess,
        pointer_overrides={
            "o_ptr": ptr,
            "do_ptr": ptr,
            "delta_ptr": "*fp32",
            "cu_seqlens_q": "*i32",
        },
    )
    pre_kinds = _kinds_for(pre_types, set())
    pre_values = {
        "o_ptr": out,
        "do_ptr": dout,
        "delta_ptr": None,
        "cu_seqlens_q": cu_q,
    }
    pre_scalars = {
        "stride_o_b": o_strides[0],
        "stride_o_h": o_strides[1],
        "stride_o_m": o_strides[2],
        "stride_o_k": o_strides[3],
        "stride_do_b": do_strides[0],
        "stride_do_h": do_strides[1],
        "stride_do_m": do_strides[2],
        "stride_do_k": do_strides[3],
        "stride_delta_b": delta_strides[0],
        "stride_delta_h": delta_strides[1],
        "stride_delta_m": delta_strides[2],
        "stride_descale_do_z": 0,
        "max_seqlen_q": int(max_sq),
    }
    delta = triton_call(
        *_input_buffers(pre_types, pre_kinds, ("delta_ptr",), pre_values),
        fn=_bwd_preprocess,
        runtime_types=pre_types,
        constexprs={
            "BLOCK_M": pre_block,
            "BLOCK_D_MODEL": int(dv),
            "BLOCK_D_MODEL_POW2": _block_pow2(dv),
            "IS_VARLEN": True,
            "IS_FP8": False,
            "descale_do_ptr": None,
        },
        kinds=pre_kinds,
        scalars=pre_scalars,
        out_shape=jax.ShapeDtypeStruct(lse.shape, jnp.float32),
        output_names=("delta_ptr",),
        grid=(_cdiv(max_sq, pre_block), batch, hq),
        options={"num_warps": 4, "num_stages": 1},
    )

    q_strides = _thd_strides(q.shape[0], hq, q.shape[-1])
    k_strides = _thd_strides(k.shape[0], hk, k.shape[-1])
    v_strides = _thd_strides(v.shape[0], hk, dv)
    dq_strides = q_strides
    dk_strides = k_strides
    dv_strides = v_strides

    bwd_types = _runtime_types_from_fn(
        bwd_kernel_causal,
        pointer_overrides={
            "Q": ptr,
            "K": ptr,
            "V": ptr,
            "DO": ptr,
            "DQ": ptr,
            "DK": ptr,
            "DV": ptr,
            "M": "*fp32",
            "Delta": "*fp32",
            "cu_seqlens_q": "*i32",
            "cu_seqlens_k": "*i32",
        },
    )
    bwd_kinds = _kinds_for(bwd_types, set())
    bwd_values = {
        "Q": q,
        "K": k,
        "V": v,
        "DO": dout,
        "DQ": None,
        "DK": None,
        "DV": None,
        "M": lse,
        "Delta": delta,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
    }
    bwd_scalars = {
        "sm_scale": float(softmax_scale),
        "stride_qb_in": q_strides[0],
        "stride_qh_in": q_strides[1],
        "stride_qm_in": q_strides[2],
        "stride_qd_in": q_strides[3],
        "stride_kb_in": k_strides[0],
        "stride_kh_in": k_strides[1],
        "stride_kn_in": k_strides[2],
        "stride_kd_in": k_strides[3],
        "stride_vb_in": v_strides[0],
        "stride_vh_in": v_strides[1],
        "stride_vn_in": v_strides[2],
        "stride_vd_in": v_strides[3],
        "stride_dqb_in": dq_strides[0],
        "stride_dqh_in": dq_strides[1],
        "stride_dqm_in": dq_strides[2],
        "stride_dqd_in": dq_strides[3],
        "stride_dkb_in": dk_strides[0],
        "stride_dkh_in": dk_strides[1],
        "stride_dkn_in": dk_strides[2],
        "stride_dkd_in": dk_strides[3],
        "stride_dvb_in": dv_strides[0],
        "stride_dvh_in": dv_strides[1],
        "stride_dvn_in": dv_strides[2],
        "stride_dvd_in": dv_strides[3],
        "stride_deltab_in": delta_strides[0],
        "stride_deltah_in": delta_strides[1],
        "stride_deltam_in": delta_strides[2],
        "stride_dob_in": do_strides[0],
        "stride_doh_in": do_strides[1],
        "stride_dom_in": do_strides[2],
        "stride_dod_in": do_strides[3],
        "stride_dropoutb_in": 0,
        "stride_dropouth_in": 0,
        "stride_dropoutm_in": 0,
        "stride_dropoutn_in": 0,
        "stride_descale_q_z_in": 0,
        "stride_descale_k_z_in": 0,
        "stride_descale_v_z_in": 0,
        "stride_descale_do_z_in": 0,
        "stride_az_in": 0,
        "stride_ah_in": 0,
        "HQ": int(hq),
        "HK": int(hk),
        "max_seqlen_q": int(max_sq),
        "max_seqlen_k": int(max_sk),
        "dropout_p": 0.0,
        "philox_seed": 0,
        "philox_offset_base_in": 0,
    }
    seqlen = max(int(max_sq), int(max_sk))
    dq, dk, dv_grad = triton_call(
        *_input_buffers(
            bwd_types, bwd_kinds, ("DQ", "DK", "DV"), bwd_values
        ),
        fn=bwd_kernel_causal,
        runtime_types=bwd_types,
        constexprs={
            "BLOCK_M1": int(one["BLOCK_M1"]),
            "BLOCK_N1": int(one["BLOCK_N1"]),
            "BLOCK_M2": int(one["BLOCK_M2"]),
            "BLOCK_N2": int(one["BLOCK_N2"]),
            "BLK_SLICE_FACTOR": int(one["BLK_SLICE_FACTOR"]),
            "HEAD_DIM": _block_pow2(dv),
            "ACTUAL_HEAD_DIM": int(dv),
            "PE_HEAD_DIM": int(pe),
            "ENABLE_DROPOUT": False,
            "IS_VARLEN": True,
            "USE_ALIBI": False,
            "USE_EXP2": True,
            "IS_FP8": False,
            "FP8_MAX": 0,
            "DEBUG_TRITON": False,
            "DEBUG_TRITON_DETAIL": False,
            "USE_INT64_STRIDES": True,
            "ENABLE_SINK": False,
            "SLIDING_WINDOW": 0,
            "Sink": None,
            "DSink": None,
            "Dropout_mask": None,
            "Alibi_slopes": None,
            "Descale_q": None,
            "Descale_k": None,
            "Descale_v": None,
            "Descale_do": None,
        },
        kinds=bwd_kinds,
        scalars=bwd_scalars,
        out_shape=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ),
        output_names=("DQ", "DK", "DV"),
        grid=(hk, _cdiv(seqlen, int(one["BLOCK_N1"])), batch),
        options={
            "num_warps": int(one["num_warps"]),
            "num_stages": int(one["num_stages"]),
            "waves_per_eu": int(one["waves_per_eu"]),
            "matrix_instr_nonkdim": int(one["matrix_instr_nonkdim"]),
        },
    )
    return dq, dk, dv_grad


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def flash_attn_varlen_triton(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
):
    """Varlen THD flash attention (causal) using AITER Triton kernels.

    q,k: [total, Hq/Hk, Dqk], v: [total, Hk, Dv], cu_seqlens_*: [batch+1] int32.
    """
    out, _lse = _fwd(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, causal,
    )
    return out


def _favt_fwd(q, k, v, cu_q, cu_k, max_sq, max_sk, dropout_p, softmax_scale, causal):
    if dropout_p:
        raise NotImplementedError("Triton varlen v1 does not implement dropout")
    out, lse = _fwd(q, k, v, cu_q, cu_k, max_sq, max_sk, softmax_scale, causal)
    if os.environ.get("JA_MHA_REMAT_CONTEXT", "1") != "0":
        out = checkpoint_name(out, "context")
        lse = checkpoint_name(lse, "context")
    return out, (q, k, v, out, lse, cu_q, cu_k)


def _favt_bwd(max_sq, max_sk, dropout_p, softmax_scale, causal, res, dout):
    q, k, v, out, lse, cu_q, cu_k = res
    del dropout_p, causal
    dq, dk, dv = _bwd(
        q, k, v, out, lse, cu_q, cu_k, dout, max_sq, max_sk, softmax_scale
    )
    return dq, dk, dv, None, None


flash_attn_varlen_triton.defvjp(_favt_fwd, _favt_bwd)
