# SPDX-License-Identifier: MIT
"""Integration tests for gfx950 A6W6 MXFP6 GEMM.

The GPU checks follow AITER's ``op_tests/test_gemm_a6w6.py``: a quantize,
dequantize, and FP32 matmul reference; bitwise agreement between specialized
or direct-scale kernels and the safe kernel; manifest eligibility; and
rejection of an exact-shape kernel on the wrong shape. When PyTorch AITER is
importable, the same kernel is also compared directly with ``gemm_a6w6``.
"""

import csv
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_aiter.ops.gemm_mxfp6 import (
    default_mxfp6_gemm_kernel,
    gemm_mxfp6,
    load_mxfp6_gemm_config,
    load_mxfp6_gemm_configs,
    mxfp6_gemm_pack_size,
    quantize_mxfp6_gemm,
)


def test_mxfp6_gemm_accepts_aiter_tuned_csv(tmp_path):
    config = tmp_path / "tuned.csv"
    config.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        "gfx950,256,512,5120,5120,1,0,1.0,kernel_for_this_gpu,1,1,0\n"
        "gfx942,304,512,5120,5120,2,0,1.0,kernel_for_other_gpu,1,1,0\n"
        "gfx950,256,256,256,128,3,0,1.0,padded_kernel,1,1,0\n",
        encoding="utf-8",
    )
    loaded = load_mxfp6_gemm_config(config, device=("gfx950", 256))
    assert loaded == {
        (512, 5120, 5120): "kernel_for_this_gpu",
        (256, 256, 128): "padded_kernel",
    }


def test_customer_mxfp6_tuning_overrides_aiter_table(tmp_path):
    pretrained = tmp_path / "a6w6_blockscale_tuned_gemm.csv"
    customer = tmp_path / "mxfp6_gemm_tuned.csv"
    header = "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
    pretrained.write_text(
        header
        + "gfx950,256,512,5120,5120,1,0,1.0,pretrained_kernel,1,1,0\n"
        + "gfx950,256,256,256,128,2,0,1.0,untouched_kernel,1,1,0\n",
        encoding="utf-8",
    )
    customer.write_text(
        "M,N,K,kernel_name\n512,5120,5120,customer_kernel\n",
        encoding="utf-8",
    )
    loaded = load_mxfp6_gemm_configs([pretrained, customer], device=("gfx950", 256))
    assert loaded[(512, 5120, 5120)] == "customer_kernel"
    assert loaded[(256, 256, 128)] == "untouched_kernel"


def test_mxfp6_gemm_pack_sizes_include_abi_guard_tiles():
    # ceil(257 / 256) * (ceil(129 / 128) + 2) = 8 tiles.
    assert mxfp6_gemm_pack_size(257, 129) == (8 * 24576, 8 * 1024)


def test_mxfp6_gemm_selects_natural_ordering_outside_short_k_swizzle_bounds():
    assert default_mxfp6_gemm_kernel(576, 18432, 6144) == "_ZN5aiter34f6gemm_bf16_per1x32Fp6_dmabig_swz0E"
    assert default_mxfp6_gemm_kernel(576, 16384, 6144) == "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE"


@pytest.mark.parametrize("m,n,k", [(256, 256, 128), (257, 259, 129)])
def test_mxfp6_gemm_runs_and_jit_is_stable(m, n, k):
    key_a, key_b = jax.random.split(jax.random.key(m + n + k))
    a = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16)

    packed, scales = quantize_mxfp6_gemm(a)
    expected_packed, expected_scales = mxfp6_gemm_pack_size(m, k)
    assert packed.shape == (expected_packed,)
    assert scales.shape == (expected_scales,)

    eager = gemm_mxfp6(a, b).block_until_ready()
    compiled = jax.jit(gemm_mxfp6)(a, b).block_until_ready()
    assert eager.shape == (m, n)
    assert eager.dtype == jnp.bfloat16
    assert jnp.array_equal(eager, compiled)


def test_mxfp6_gemm_rejects_invalid_operands():
    with pytest.raises(ValueError, match="bfloat16"):
        gemm_mxfp6(jnp.ones((16, 16), jnp.float32), jnp.ones((16, 16), jnp.float32))
    with pytest.raises(ValueError, match="K mismatch"):
        gemm_mxfp6(jnp.ones((16, 16), jnp.bfloat16), jnp.ones((16, 8), jnp.bfloat16))


_SPECIALIZED_CASES = (
    ((9450, 5120, 5120), "_ZN5aiter44f6gemm_bf16_per1x32Fp6_m9472_n5120_k5120_m82E",
     "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE"),
    ((9450, 13824, 5120), "_ZN5aiter45f6gemm_bf16_per1x32Fp6_m9472_n13824_k5120_m82E",
     "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE"),
    ((9450, 5120, 13824), "_ZN5aiter45f6gemm_bf16_per1x32Fp6_m9472_n5120_k13824_m82E",
     "_ZN5aiter34f6gemm_bf16_per1x32Fp6_dmabig_allkE"),
)
_DIRECT_SCALE_CASES = (
    ((257, 513, 129), "_ZN5aiter34f6gemm_bf16_per1x32Fp6_dmabig_swz0E",
     "_ZN5aiter44f6gemm_bf16_per1x32Fp6_dmabig_m82_nat_dscaleE"),
    ((513, 784, 1152), "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE",
     "_ZN5aiter44f6gemm_bf16_per1x32Fp6_dmabig_m82_g32_dscaleE"),
    ((256, 256, 6273), "_ZN5aiter34f6gemm_bf16_per1x32Fp6_dmabig_swz0E",
     "_ZN5aiter44f6gemm_bf16_per1x32Fp6_dmabig_m82_nat_dscaleE"),
)
# The five exact-shape kernels added for the Ideogram DiT shapes. The other
# four DiT shapes keep an existing generic kernel, so they are not new cases.
_IDEOGRAM_EXACT_CASES = (
    ((10240, 4608, 4608), "_ZN5aiter48f6gemm_bf16_per1x32Fp6_m10240_n4608_k4608_x4m84wE"),
    ((10240, 12288, 4608), "_ZN5aiter49f6gemm_bf16_per1x32Fp6_m10240_n12288_k4608_x4m84wE"),
    ((25600, 4608, 4608), "_ZN5aiter47f6gemm_bf16_per1x32Fp6_m25600_n4608_k4608_x5m84E"),
    ((25600, 4608, 12288), "_ZN5aiter48f6gemm_bf16_per1x32Fp6_m25600_n4608_k12288_x5m84E"),
    ((33280, 12288, 4608), "_ZN5aiter48f6gemm_bf16_per1x32Fp6_m33280_n12288_k4608_x5m84E"),
)


def _operands(m, n, k):
    key_a, key_b = jax.random.split(jax.random.key(m + n + k))
    return (
        jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16),
        jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16),
    )


def _assert_allclose(actual, expected, *, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05):
    """Match AITER ``checkAllclose`` for the MXFP6 GEMM reference."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    mismatch = ~np.isclose(actual, expected, rtol=rtol, atol=atol)
    ratio = float(mismatch.mean()) if mismatch.size else 0.0
    assert ratio < tol_err_ratio, (
        f"MXFP6 reference mismatch ratio {ratio:.3%} exceeds {tol_err_ratio:.1%}"
    )


def _aiter_mxfp6():
    pytest.importorskip("torch")
    try:
        from aiter.ops import gemm_op_a6w6
    except ImportError:
        import sys

        root = Path(__file__).resolve().parents[1] / "third_party" / "aiter"
        sys.path.insert(0, str(root))
        from aiter.ops import gemm_op_a6w6
    return gemm_op_a6w6


def _reference_gemm(a, b):
    """Quantize, dequantize, then multiply, as AITER's ``run_torch`` does."""
    gemm_op = _aiter_mxfp6()
    import torch

    def convert(value):
        array = np.asarray(value, dtype=np.float32)
        return torch.tensor(array, device="cuda", dtype=torch.bfloat16)

    x, w = convert(a), convert(b)
    k = x.shape[1]
    padded_k = (k + 127) // 128 * 128
    if padded_k != k:
        x = torch.nn.functional.pad(x, (0, padded_k - k))
        w = torch.nn.functional.pad(w, (0, padded_k - w.shape[1]))
    x_codes, x_scales = gemm_op.quant_mxfp6_torch(x)
    w_codes, w_scales = gemm_op.quant_mxfp6_torch(w)
    product = torch.mm(
        gemm_op.dequant_mxfp6_torch(x_codes, x_scales),
        gemm_op.dequant_mxfp6_torch(w_codes, w_scales).T,
    )
    return product.to(torch.bfloat16).float().cpu().numpy()


def _manifest_csv():
    configured = os.environ.get("AITER_ASM_DIR", "").strip()
    if configured:
        matches = list(Path(configured).glob("**/f6gemm/f6gemm_bf16_per1x32Fp6.csv"))
        if matches:
            return matches[0]
    checkout = Path(__file__).resolve().parents[1]
    return checkout / "third_party/aiter/hsa/gfx950/f6gemm/f6gemm_bf16_per1x32Fp6.csv"


def _manifest_kernel_names(m, n, k):
    manifest = _manifest_csv()
    if not manifest.is_file():
        raise FileNotFoundError(f"MXFP6 manifest not found: {manifest}")
    padded = (
        (m + 255) // 256 * 256,
        (n + 255) // 256 * 256,
        (k + 127) // 128 * 128,
    )
    names = []
    with manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            exact = tuple(int(row[key]) for key in ("exact_M", "exact_N", "exact_K"))
            limits = tuple(int(row[key]) for key in ("swizzle_max_M", "swizzle_max_N", "swizzle_max_K"))
            exact_ok = exact[2] <= 0 or exact == padded
            swizzle_ok = limits[2] <= 0 or padded[2] > limits[2] or (
                padded[0] <= limits[0] and padded[1] <= limits[1]
            )
            if exact_ok and swizzle_ok:
                names.append(row["knl_name"])
    return names


@pytest.mark.parametrize("m,n,k", [(257, 513, 129), (2048, 2048, 2048)])
def test_mxfp6_gemm_matches_quantized_reference(m, n, k):
    _aiter_mxfp6()
    a, b = _operands(m, n, k)
    actual = gemm_mxfp6(a, b, kernel_name=default_mxfp6_gemm_kernel(m, n, k))
    _assert_allclose(actual, _reference_gemm(a, b))


@pytest.mark.parametrize("shape,specialized,safe", _SPECIALIZED_CASES, ids=("square", "ffn_up", "ffn_down"))
def test_mxfp6_specialized_kernel_matches_safe_kernel(shape, specialized, safe):
    a, b = _operands(*shape)
    baseline = gemm_mxfp6(a, b, kernel_name=safe)
    actual = gemm_mxfp6(a, b, kernel_name=specialized)
    assert jnp.array_equal(actual, baseline)


def test_mxfp6_manifest_follows_aiter_kernel_names():
    """AITER checks every manifest row, including the five renamed dmabig kernels."""
    manifest = _manifest_csv()
    with manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    exact = [
        row for row in rows
        if min(int(row[key]) for key in ("exact_M", "exact_N", "exact_K")) > 0
    ]
    non_wan = [row for row in exact if int(row["exact_M"]) != 9472]
    assert len(exact) == 35
    assert len(non_wan) == 32
    present = {row["knl_name"] for row in rows}
    assert {kernel for _, kernel in _IDEOGRAM_EXACT_CASES} <= present
    for row in rows:
        match = re.fullmatch(r"_ZN5aiter(\d+)(.+)E", row["knl_name"])
        assert match is not None, row["knl_name"]
        declared_length, entrypoint = match.groups()
        assert int(declared_length) == len(entrypoint)
        assert row["co_name"] == f"{entrypoint}.co"
        assert (manifest.parent / row["co_name"]).is_file()


@pytest.mark.parametrize("shape,specialized", _IDEOGRAM_EXACT_CASES)
def test_mxfp6_ideogram_exact_kernel_matches_default(shape, specialized):
    baseline_name = default_mxfp6_gemm_kernel(*shape)
    a, b = _operands(*shape)
    baseline = gemm_mxfp6(a, b, kernel_name=baseline_name)
    actual = gemm_mxfp6(a, b, kernel_name=specialized)
    assert jnp.array_equal(actual, baseline)


@pytest.mark.parametrize("shape,baseline_kernel,candidate", _DIRECT_SCALE_CASES)
def test_mxfp6_direct_scale_kernel_matches_fallback(shape, baseline_kernel, candidate):
    a, b = _operands(*shape)
    baseline = gemm_mxfp6(a, b, kernel_name=baseline_kernel)
    actual = gemm_mxfp6(a, b, kernel_name=candidate)
    assert jnp.array_equal(actual, baseline)


def test_mxfp6_manifest_excludes_incompatible_exact_kernels():
    small = _manifest_kernel_names(257, 513, 129)
    assert not any("f6gemm_bf16_per1x32Fp6_m9472_" in name for name in small)
    square = _manifest_kernel_names(9450, 5120, 5120)
    assert "_ZN5aiter44f6gemm_bf16_per1x32Fp6_m9472_n5120_k5120_m82E" in square
    assert "_ZN5aiter45f6gemm_bf16_per1x32Fp6_m9472_n13824_k5120_m82E" not in square
    assert "_ZN5aiter45f6gemm_bf16_per1x32Fp6_m9472_n5120_k13824_m82E" not in square


def test_mxfp6_specialized_kernel_rejects_incompatible_shape():
    a, b = _operands(257, 513, 129)
    with pytest.raises(jax.errors.JaxRuntimeError, match="exact padded shape"):
        gemm_mxfp6(
            a, b, kernel_name="_ZN5aiter44f6gemm_bf16_per1x32Fp6_m9472_n5120_k5120_m82E"
        ).block_until_ready()


@pytest.mark.parametrize("m,n,k", [(257, 513, 129), (513, 784, 1152)])
def test_mxfp6_gemm_matches_aiter(m, n, k):
    # The helper skips the test when torch is missing; import torch only after it.
    gemm_op = _aiter_mxfp6()
    import torch

    a, b = _operands(m, n, k)
    kernel = default_mxfp6_gemm_kernel(m, n, k)
    jax_out = gemm_mxfp6(a, b, kernel_name=kernel)
    x = torch.tensor(np.asarray(a, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    w = torch.tensor(np.asarray(b, dtype=np.float32), device="cuda", dtype=torch.bfloat16)
    x_packed, x_scale = gemm_op.quant_mxfp6_gemm(x)
    w_packed, w_scale = gemm_op.quant_mxfp6_gemm(w)
    aiter_out = gemm_op.gemm_a6w6(
        x_packed, w_packed, x_scale, w_scale, m, n, k, kernelName=kernel
    )
    assert np.array_equal(np.asarray(jax_out), aiter_out.cpu().float().numpy())
