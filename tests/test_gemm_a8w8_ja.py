# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import random
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, perftest, benchmark
import pandas as pd
import argparse

# JAX imports
import jax
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

# JAX-side implementations
import jax_aiter
from jax_aiter.gemm_a8w8.asm_gemm_a8w8 import gemm_a8w8_ASM as jax_gemm_a8w8_ASM
from jax_aiter.wv_splitkq.wv_splitkq import wv_splitkq_fp8 as jax_wv_splitkq_fp8

TEST_NUM_ITERS = 100


def _to_jax(t: torch.Tensor) -> jnp.ndarray:
    # Handle FP8 tensors that DLPack doesn't support
    dtype_str = str(t.dtype)
    if "float8" in dtype_str or "fp8" in dtype_str:
        # Convert to float32 first, then to JAX, then cast to the correct FP8 variant
        t_f32 = t.to(torch.float32)
        jax_f32 = jax_dlpack.from_dlpack(t_f32)
        from jax_aiter.ja_compat import dtypes as jax_dtypes

        # Use the specific FP8 dtype that matches the hardware
        return jax_f32.astype(jax_dtypes.get_dtype_fp8())
    return jax_dlpack.from_dlpack(t)


def _to_torch(x: jnp.ndarray) -> torch.Tensor:
    return torch_dlpack.from_dlpack(x)


@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_ASM(x, weightshuffle, x_scale, w_scale, bias)


@perftest()
def run_jax_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    # Convert to JAX
    XQ = _to_jax(x)
    WQ = _to_jax(weightshuffle)
    XS = _to_jax(x_scale)
    WS = _to_jax(w_scale)
    B = _to_jax(bias) if bias is not None else None

    # Call JAX implementation
    yj = jax_gemm_a8w8_ASM(XQ, WQ, XS, WS, B, check=True)
    yj.block_until_ready()

    # Convert back to torch
    return _to_torch(yj)


@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_skinny(
    x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16, cu_count=1
):
    out = torch.empty(x.shape[0], weight.shape[0], dtype=dtype, device=x.device)
    aiter.wvSplitKQ(weight, x, out, w_scale, x_scale, cu_count)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def run_jax_gemm_skinny(
    x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16, cu_count=1
):
    # Convert to JAX
    XQ = _to_jax(x)
    WQ = _to_jax(weight)
    XS = _to_jax(x_scale)
    WS = _to_jax(w_scale)

    # Convert PyTorch dtype to JAX dtype
    from jax_aiter.ja_compat import dtypes as jax_dtypes

    if dtype == dtypes.bf16:
        jax_out_dtype = jax_dtypes.bf16
    elif dtype == dtypes.fp16:
        jax_out_dtype = jax_dtypes.fp16
    else:
        jax_out_dtype = jax_dtypes.bf16

    # AITER call: aiter.wvSplitKQ(weight[N,K], x[M,K], out[M,N], w_scale, x_scale, cu_count)
    # Our JAX call should match this exactly: wv_splitkq_fp8(weight, x, w_scale, x_scale, ...)
    yj = jax_wv_splitkq_fp8(
        WQ, XQ, WS, XS, cu_count=cu_count, dtype=jax_out_dtype, check=False
    )
    yj.block_until_ready()

    # Convert back to torch
    out = _to_torch(yj)

    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@benchmark()
def test_gemm(dtype, m, n, k, quantDtype=dtypes.i8):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quantDtype)
    weightshuffle = shuffle_weight(weight, layout=(16, 16))
    bias = torch.rand([1, n], dtype=dtype, device="cuda") * 10

    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)

    # Only test operations we support
    avg_b = None
    err_b = None
    avg_c = None
    err_c = None
    avg_d = None
    err_d = None

    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    print(f"CU count: {cu_num}")

    # Test ASM GEMM if supported
    if dtype == dtypes.bf16 and quantDtype == dtypes.i8 and bias is not None:
        weightshuffle_asm = shuffle_weight(weight, layout=(32, 16))
        bias_f32 = bias.to(dtypes.fp32)

        # PyTorch ASM
        d, avg_d = run_gemm_asm(x, weightshuffle_asm, x_scale, w_scale, bias_f32, dtype)
        if d is not None:
            err_d = checkAllclose(a, d, msg="asm: ", rtol=1e-2, atol=1e-2)
        else:
            avg_d = None

        # JAX ASM
        try:
            d_jax, avg_d_jax = run_jax_gemm_asm(
                x, weightshuffle_asm, x_scale, w_scale, bias_f32, dtype
            )
            if d_jax is not None:
                err_d_jax = checkAllclose(
                    a, d_jax, msg="jax asm: ", rtol=1e-2, atol=1e-2
                )
                err_d_vs_jax = checkAllclose(
                    d, d_jax, msg="asm vs jax asm: ", rtol=1e-2, atol=1e-2
                )
                print(
                    f"JAX ASM time: {avg_d_jax:.2f} us, PyTorch ASM time: {avg_d:.2f} us"
                )
        except Exception as e:
            print(f"JAX ASM failed: {e}")

    return {
        "ck us": avg_b,
        "ck err": err_b,
        "ck bpreshuffle us": avg_c,
        "ck bpreshuffle err": err_c,
        "asm us": avg_d,
        "asm err": err_d,
    }


def test_skinny_gemm(dtype, m, n, k, quantDtype=dtypes.fp8, cu_count=80):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.per_tensor_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.per_tensor_quant(weight, quant_dtype=quantDtype)
    bias = None

    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)

    if m <= 2:
        # Test WvSplitKQ
        b, avg_b = run_gemm_skinny(x, weight, x_scale, w_scale, None, dtype, cu_count)

        # Test JAX WvSplitKQ
        try:
            b_jax, avg_b_jax = run_jax_gemm_skinny(
                x, weight, x_scale, w_scale, None, dtype, cu_count
            )

            msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, quantDtype: {quantDtype}, torch avg: {avg_a:<8.2f} us, skinny_gemm avg: {avg_b:<8.2f} us, jax_skinny avg: {avg_b_jax:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
            checkAllclose(a, b, msg="torch vs aiter: " + msg, rtol=1e-2, atol=0.01)
            checkAllclose(a, b_jax, msg="torch vs jax: " + msg, rtol=1e-2, atol=0.01)
            checkAllclose(b, b_jax, msg="aiter vs jax: " + msg, rtol=1e-2, atol=0.01)

        except Exception as e:
            print(f"JAX WvSplitKQ failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        # For larger m, we don't have CK support, so skip
        print(f"Skipping large batch size {m} (no CK support)")


# Test cases from original AITER test
l_dtype = ["bf16", "fp16"]
l_quantDtype = ["i8", "fp8"]
l_mnk_nm = [
    # qkv_proj
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    (16384, 1280, 8192),
    # attn_out
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
]


def test_normal_gemm_a8w8_pertoken_quant(l_dtype, l_quantDtype, l_mnk):
    df = []
    for dtype in l_dtype:
        for quantDtype in l_quantDtype:
            for m, n, k in l_mnk:
                ret = test_gemm(dtype, m, n, k, quantDtype)
                df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")


def test_skinny_gemm_a8w8_pertoken_quant():
    random.seed(137)

    aligned_k = 16
    cu_count = torch.cuda.get_device_properties(device="cuda").multi_processor_count

    # Test cases for WvSplitKQ (small batch sizes)
    test_mnk_list = [
        [1, 4, 1264],
        [1, 12, 8720],
        [1, 17, 6192],
        [1, 40, 9024],
        [2, 27, 4544],
        [2, 28, 6544],
        [2, 57, 1952],
        [2, 60, 96],
        [2, 320, 32768],
        [2, 640, 32768],
        [2, 1280, 32768],
        [2, 320, 32768 + 1024],
        [2, 320, 2 * 32768],
    ]

    print(f"cu_count={cu_count}")
    print(f"total test case count: {2 * len(test_mnk_list)}")

    loop_count = 1
    for i in range(loop_count):
        for mnk in test_mnk_list:
            m, n, k = mnk
            for quant_dtype in [dtypes.fp8]:
                for dtype in [dtypes.fp16, dtypes.bf16]:
                    test_skinny_gemm(dtype, m, n, k, quant_dtype, cu_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=l_dtype,
        nargs="?",
        const=None,
        default=None,
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-q",
        "--quantDtype",
        type=str,
        choices=l_quantDtype,
        nargs="?",
        const=None,
        default=None,
        help="""Date type of quantization.
        e.g.: -q fp8""",
    )
    parser.add_argument(
        "-mnk",
        type=dtypes.str2tuple,
        nargs="?",
        const=None,
        default=None,
        help="""Shape of mnk.
        e.g. -mnk 1280,8192,1024""",
    )

    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.quantDtype is None:
        l_quantDtype = [dtypes.d_dtypes[key] for key in l_quantDtype]
    else:
        l_quantDtype = [dtypes.d_dtypes[args.quantDtype]]
    if args.mnk is not None:
        l_mnk_nm = [args.mnk]

    # Test normal GEMM (ASM with I8)
    # test_normal_gemm_a8w8_pertoken_quant(l_dtype, l_quantDtype, l_mnk_nm)

    # Test skinny GEMM (WvSplitKQ with FP8)
    test_skinny_gemm_a8w8_pertoken_quant()
