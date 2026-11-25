# SPDX-License-Identifier: MIT
"""Integration tests for gfx950 A6W4 GEMM (MXFP6 activations by MXFP4 weights).

The GPU checks follow AITER's ``op_tests/test_gemm_a6w4.py`` and
``op_tests/test_mxfp_exact_shape_dispatch.py``: a quantize, dequantize, FP32
matmul reference with cosine and relative-L2 bounds; bitwise agreement between
the MFMA32, dmabig, long-K, and exact-shape kernels and the swizzle-0 kernel;
default and tuned kernel dispatch; manifest naming and eligibility; and
rejection of launches the kernel ABI cannot express. When PyTorch AITER is
importable, the GEMM result is also compared bitwise with AITER's
``quant_mxfp4_gemm`` and ``gemm_a6w4``. The reference checks use AITER's torch
quantize and dequantize functions, so they need PyTorch AITER and a GPU.
"""

import csv
import os
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import importlib

# ``jax_aiter.ops.gemm_a6w4`` is also the re-exported function, so name the module.
a6w4_ops = importlib.import_module("jax_aiter.ops.gemm_a6w4")
from jax_aiter.ops.gemm_a6w4 import (  # noqa: E402
    default_a6w4_gemm_kernel,
    gemm_a6w4,
    gemm_a6w4_packed,
    load_a6w4_gemm_config,
    load_a6w4_gemm_configs,
    mxfp4_gemm_pack_size,
    quantize_mxfp4_gemm,
)
from jax_aiter.ops.gemm_mxfp6 import quantize_mxfp6_gemm

MFMA32_SMALL_KERNEL = "_ZN5aiter40f6f4gemm_bf16_per1x32Fp6Fp4_m32_s0_a4_ntE"
MFMA32_SWZ0_KERNEL = "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s0_a4_tE"
MFMA32_GROUPED_KERNEL = "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s3_a4_tE"
MFMA32_LONG_K_KERNEL = "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s3_a5_tE"
DMABIG_SWZ0_KERNEL = "_ZN5aiter44f6f4gemm_bf16_per1x32Fp6Fp4_dmabig_m82_s0_ntE"
DMABIG_GROUPED_KERNEL = "_ZN5aiter44f6f4gemm_bf16_per1x32Fp6Fp4_dmabig_m82_s3_ntE"
SPECIALIZED_SQUARE_KERNEL = "_ZN5aiter49f6f4gemm_bf16_per1x32Fp6Fp4_m9472_n5120_k5120_m82E"
SPECIALIZED_UP_KERNEL = "_ZN5aiter50f6f4gemm_bf16_per1x32Fp6Fp4_m9472_n13824_k5120_m82E"
SPECIALIZED_DOWN_KERNEL = "_ZN5aiter50f6f4gemm_bf16_per1x32Fp6Fp4_m9472_n5120_k13824_m82E"
DEFAULT_ROUND_MODE = 1


# --- Tuning-table and sizing checks that need no device kernel ----------------


def test_a6w4_gemm_accepts_aiter_tuned_csv(tmp_path):
    config = tmp_path / "tuned.csv"
    config.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        "gfx950,256,9450,5120,5120,1,0,1.0,kernel_for_this_gpu,1,1,0\n"
        "gfx942,304,9450,5120,5120,2,0,1.0,kernel_for_other_gpu,1,1,0\n"
        "gfx950,256,256,256,128,3,0,1.0,padded_kernel,1,1,0\n",
        encoding="utf-8",
    )
    loaded = load_a6w4_gemm_config(config, device=("gfx950", 256))
    assert loaded == {
        (9450, 5120, 5120): "kernel_for_this_gpu",
        (256, 256, 128): "padded_kernel",
    }


def test_customer_a6w4_tuning_overrides_aiter_table(tmp_path):
    pretrained = tmp_path / "a6w4_asm_tuned_gemm.csv"
    customer = tmp_path / "a6w4_gemm_tuned.csv"
    pretrained.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        "gfx950,256,512,5120,5120,1,0,1.0,pretrained_kernel,1,1,0\n"
        "gfx950,256,256,256,128,2,0,1.0,untouched_kernel,1,1,0\n",
        encoding="utf-8",
    )
    customer.write_text("M,N,K,kernel_name\n512,5120,5120,customer_kernel\n", encoding="utf-8")
    loaded = load_a6w4_gemm_configs([pretrained, customer], device=("gfx950", 256))
    assert loaded[(512, 5120, 5120)] == "customer_kernel"
    assert loaded[(256, 256, 128)] == "untouched_kernel"


def test_a6w4_tuning_rejects_split_k(tmp_path):
    config = tmp_path / "split.csv"
    config.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        "gfx950,256,256,256,128,1,2,1.0,some_kernel,1,1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="splitK"):
        load_a6w4_gemm_config(config, device=("gfx950", 256))


def test_a6w4_pack_sizes_include_abi_guard_tiles():
    # ceil(257 / 256) * (ceil(129 / 128) + 2) = 8 tiles.
    assert mxfp4_gemm_pack_size(257, 129) == (8 * 16384, 8 * 1024)


def test_a6w4_pack_size_rejects_invalid_or_oversized_shapes():
    with pytest.raises(ValueError, match="positive dimensions"):
        mxfp4_gemm_pack_size(0, 128)
    with pytest.raises(ValueError, match="2 GiB"):
        mxfp4_gemm_pack_size(65536, 65536)


def test_a6w4_rejects_unsupported_launch_contracts_before_allocation():
    placeholder = jax.ShapeDtypeStruct((0,), jnp.uint8)
    args = (placeholder,) * 4
    with pytest.raises(ValueError, match="2 GiB"):
        gemm_a6w4_packed(*args, m=65792, n=16384, k=128)
    with pytest.raises(ValueError, match="int32"):
        gemm_a6w4_packed(*args, m=1, n=1, k=(1 << 32) + 128)
    with pytest.raises(ValueError, match="padded K"):
        gemm_a6w4_packed(*args, m=1, n=1, k=(1 << 31) - 1)
    with pytest.raises(ValueError, match="positive dimensions"):
        gemm_a6w4_packed(*args, m=0, n=1, k=128)


def test_a6w4_default_kernel_selection_and_bounds():
    assert default_a6w4_gemm_kernel(512, 5120, 5120) == MFMA32_SMALL_KERNEL
    assert default_a6w4_gemm_kernel(9450, 5120, 5120) == MFMA32_GROUPED_KERNEL
    assert default_a6w4_gemm_kernel(9450, 13824, 5120) == MFMA32_GROUPED_KERNEL
    assert default_a6w4_gemm_kernel(9450, 5120, 13824) == MFMA32_LONG_K_KERNEL
    assert default_a6w4_gemm_kernel(9450, 27648, 5120) == MFMA32_SWZ0_KERNEL
    assert default_a6w4_gemm_kernel(131073, 13824, 5120) == MFMA32_SWZ0_KERNEL


def test_a6w4_dispatch_uses_specialized_kernels_and_respects_default_bounds():
    """The shipped tuned table must send the Wan shapes to their exact kernels."""
    if a6w4_ops._pretrained_config() is None:
        pytest.skip("AITER's a6w4_asm_tuned_gemm.csv is not available")
    select = a6w4_ops._select_kernel
    assert select(512, 5120, 5120, None) == MFMA32_SMALL_KERNEL
    assert select(9450, 5120, 5120, None) == SPECIALIZED_SQUARE_KERNEL
    assert select(9450, 13824, 5120, None) == SPECIALIZED_UP_KERNEL
    assert select(9450, 5120, 13824, None) == SPECIALIZED_DOWN_KERNEL
    assert select(9450, 27648, 5120, None) == MFMA32_SWZ0_KERNEL
    assert select(131073, 13824, 5120, None) == MFMA32_SWZ0_KERNEL
    assert select(9450, 5120, 5120, MFMA32_SWZ0_KERNEL) == MFMA32_SWZ0_KERNEL


# --- Manifest checks ------------------------------------------------------------


def _manifest_csv():
    name = "f6f4gemm_bf16_per1x32Fp6Fp4.csv"
    configured = os.environ.get("AITER_ASM_DIR", "").strip()
    if configured:
        matches = list(Path(configured).glob(f"**/f6f4gemm/{name}"))
        if matches:
            return matches[0]
    checkout = Path(__file__).resolve().parents[1]
    return checkout / "third_party/aiter/hsa/gfx950/f6f4gemm" / name


def _manifest_rows():
    manifest = _manifest_csv()
    if not manifest.is_file():
        raise FileNotFoundError(f"A6W4 manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _exact_rows():
    return [
        row for row in _manifest_rows()
        if min(int(row[key]) for key in ("exact_M", "exact_N", "exact_K")) > 0
    ]


def _manifest_kernel_names(m, n, k):
    padded = ((m + 255) // 256 * 256, (n + 255) // 256 * 256, (k + 127) // 128 * 128)
    names = []
    for row in _manifest_rows():
        exact = tuple(int(row[key]) for key in ("exact_M", "exact_N", "exact_K"))
        limits = tuple(int(row[key]) for key in ("swizzle_max_M", "swizzle_max_N", "swizzle_max_K"))
        exact_ok = exact[2] <= 0 or exact == padded
        swizzle_ok = limits[2] <= 0 or padded[2] > limits[2] or (
            padded[0] <= limits[0] and padded[1] <= limits[1]
        )
        if exact_ok and swizzle_ok:
            names.append(row["knl_name"])
    return names


def test_a6w4_manifest_follows_aiter_kernel_names():
    rows = _manifest_rows()
    manifest = _manifest_csv()
    names = [row["knl_name"] for row in rows]
    assert len(names) == len(set(names))
    assert {
        MFMA32_SMALL_KERNEL, MFMA32_SWZ0_KERNEL, MFMA32_GROUPED_KERNEL, MFMA32_LONG_K_KERNEL,
        DMABIG_SWZ0_KERNEL, DMABIG_GROUPED_KERNEL, SPECIALIZED_SQUARE_KERNEL,
        SPECIALIZED_UP_KERNEL, SPECIALIZED_DOWN_KERNEL,
    } <= set(names)
    for row in rows:
        assert row["splitK"] == "0"
        assert row["pack_layout"] == "mxfp6_mxfp4_c0_256_padk2"
        match = re.fullmatch(r"_ZN5aiter(\d+)(.+)E", row["knl_name"])
        assert match is not None, row["knl_name"]
        declared_length, entrypoint = match.groups()
        assert int(declared_length) == len(entrypoint)
        assert row["co_name"] == f"{entrypoint}.co"
        assert (manifest.parent / row["co_name"]).is_file()


def test_a6w4_manifest_excludes_incompatible_exact_kernels():
    small = _manifest_kernel_names(257, 513, 129)
    assert not any("_m9472_" in name for name in small)
    square = _manifest_kernel_names(9450, 5120, 5120)
    assert SPECIALIZED_SQUARE_KERNEL in square
    assert SPECIALIZED_UP_KERNEL not in square
    assert SPECIALIZED_DOWN_KERNEL not in square


def test_a6w4_tuned_table_names_manifest_kernels():
    """Every kernel in AITER's shipped A6W4 table must exist in the manifest."""
    pretrained = a6w4_ops._pretrained_config()
    if pretrained is None:
        pytest.skip("AITER's a6w4_asm_tuned_gemm.csv is not available")
    known = {row["knl_name"] for row in _manifest_rows()}
    with pretrained.open(newline="", encoding="utf-8") as file:
        used = {row["kernelName"].strip() for row in csv.DictReader(file)}
    assert used <= known, sorted(used - known)


# --- Device tests ---------------------------------------------------------------


def _operands(m, n, k):
    key_a, key_b = jax.random.split(jax.random.key(m + n + k))
    return (
        jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16),
        jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16),
    )


def _packed(a, b, round_mode=DEFAULT_ROUND_MODE):
    a_packed, a_scale = quantize_mxfp6_gemm(a)
    b_packed, b_scale = quantize_mxfp4_gemm(b, round_mode=round_mode)
    return a_packed, b_packed, a_scale, b_scale


def _aiter_a6w4():
    """Import PyTorch AITER's A6W4 modules, falling back to the vendored tree."""
    pytest.importorskip("torch")
    try:
        import aiter.ops.gemm_op_a6w4 as a6w4
    except ImportError:
        root = Path(__file__).resolve().parents[1] / "third_party" / "aiter"
        sys.path.insert(0, str(root))
        import aiter.ops.gemm_op_a6w4 as a6w4
    from aiter.ops import gemm_op_a6w6
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility import dtypes, fp4_utils

    return a6w4, gemm_op_a6w6, per_1x32_f4_quant, dtypes, fp4_utils


def _to_torch(value):
    import torch

    return torch.tensor(np.asarray(value, dtype=np.float32), device="cuda", dtype=torch.bfloat16)


def _quantized_reference(x, w, round_mode):
    """AITER's reference: dequantized MXFP6 activation by rotated, dequantized MXFP4 weight."""
    _, a6w6, per_1x32_f4_quant, dtypes, fp4_utils = _aiter_a6w4()
    import torch
    import torch.nn.functional as F

    padded_k = (x.shape[1] + 127) // 128 * 128
    x = F.pad(x, (0, padded_k - x.shape[1]))
    w = F.pad(w, (0, padded_k - w.shape[1]))
    x_codes, x_scales = a6w6.quant_mxfp6_torch(x)
    x_dequant = a6w6.dequant_mxfp6_torch(x_codes, x_scales)
    w_rotated = a6w6._rotate_k32_torch(w).to(torch.bfloat16)
    w_codes, w_scales = per_1x32_f4_quant(w_rotated, quant_dtype=dtypes.fp4x2, round_mode=round_mode)
    w_dequant = fp4_utils.mxfp4_to_f32(w_codes)
    w_dequant *= fp4_utils.e8m0_to_f32(w_scales.view(torch.uint8)).repeat_interleave(32, dim=1)
    return x_dequant @ w_dequant.T


@pytest.mark.parametrize("m,n,k", [(256, 256, 128), (257, 259, 129)])
def test_a6w4_gemm_runs_and_jit_is_stable(m, n, k):
    a, b = _operands(m, n, k)
    packed, scales = quantize_mxfp4_gemm(b)
    expected_packed, expected_scales = mxfp4_gemm_pack_size(n, k)
    assert packed.shape == (expected_packed,)
    assert scales.shape == (expected_scales,)
    assert packed.dtype == scales.dtype == jnp.uint8

    eager = gemm_a6w4(a, b).block_until_ready()
    compiled = jax.jit(gemm_a6w4)(a, b).block_until_ready()
    assert eager.shape == (m, n)
    assert eager.dtype == jnp.bfloat16
    assert bool(jnp.isfinite(eager.astype(jnp.float32)).all())
    assert jnp.array_equal(eager, compiled)


@pytest.mark.parametrize("m,n,k", [(257, 513, 129), (2048, 512, 256), (300, 513, 129), (512, 5120, 5120)])
def test_a6w4_compiles_and_matches_eager(m, n, k):
    """JAX analogue of AITER's ``torch.compile`` full-graph check."""
    a, b = _operands(m, n, k)
    w_packed, w_scales = quantize_mxfp4_gemm(b)

    def quantize_and_gemm(activation, packed_weight, weight_scale):
        packed_activation, activation_scale = quantize_mxfp6_gemm(activation)
        return gemm_a6w4_packed(
            packed_activation, packed_weight, activation_scale, weight_scale, m=m, n=n, k=k
        )

    eager = quantize_and_gemm(a, w_packed, w_scales)
    compiled = jax.jit(quantize_and_gemm)(a, w_packed, w_scales)
    assert jnp.array_equal(eager, compiled)


def test_a6w4_gemm_rejects_invalid_operands():
    with pytest.raises(ValueError, match="bfloat16"):
        gemm_a6w4(jnp.ones((16, 16), jnp.float32), jnp.ones((16, 16), jnp.float32))
    with pytest.raises(ValueError, match="K mismatch"):
        gemm_a6w4(jnp.ones((16, 16), jnp.bfloat16), jnp.ones((16, 8), jnp.bfloat16))
    with pytest.raises(ValueError, match="bfloat16"):
        quantize_mxfp4_gemm(jnp.ones((16, 16), jnp.float32))
    with pytest.raises(ValueError, match="round_mode"):
        quantize_mxfp4_gemm(jnp.ones((16, 16), jnp.bfloat16), round_mode=4)


@pytest.mark.parametrize(
    "shape,kernel_name,round_mode",
    [
        ((256, 256, 128), None, DEFAULT_ROUND_MODE),
        ((256, 256, 256), MFMA32_SMALL_KERNEL, 2),
        ((512, 768, 256), MFMA32_SMALL_KERNEL, DEFAULT_ROUND_MODE),
        ((513, 769, 257), MFMA32_SMALL_KERNEL, DEFAULT_ROUND_MODE),
        ((257, 513, 129), MFMA32_SMALL_KERNEL, DEFAULT_ROUND_MODE),
        ((257, 513, 513), MFMA32_SMALL_KERNEL, DEFAULT_ROUND_MODE),
        ((257, 513, 129), DMABIG_SWZ0_KERNEL, DEFAULT_ROUND_MODE),
        ((513, 769, 257), DMABIG_GROUPED_KERNEL, DEFAULT_ROUND_MODE),
        ((2048, 512, 256), None, DEFAULT_ROUND_MODE),
        ((2048, 256, 512), None, DEFAULT_ROUND_MODE),
        ((9450, 5120, 5120), None, DEFAULT_ROUND_MODE),
        ((9450, 13824, 5120), None, DEFAULT_ROUND_MODE),
        ((9450, 5120, 13824), None, DEFAULT_ROUND_MODE),
    ],
)
def test_a6w4_gemm_matches_quantized_reference(shape, kernel_name, round_mode):
    # The helper skips the test when torch is missing; import torch only after it.
    _aiter_a6w4()
    import torch
    import torch.nn.functional as F

    m, n, k = shape
    a, b = _operands(m, n, k)
    out = gemm_a6w4(a, b, round_mode=round_mode, kernel_name=kernel_name)
    out_t = _to_torch(out).float()
    x, w = _to_torch(a), _to_torch(b)
    quantized_ref = _quantized_reference(x, w, round_mode)
    bf16_ref = x.float() @ w.float().T

    kernel_cosine = F.cosine_similarity(out_t.flatten(), quantized_ref.flatten(), dim=0).item()
    bf16_cosine = F.cosine_similarity(out_t.flatten(), bf16_ref.flatten(), dim=0).item()
    relative_l2 = ((out_t - quantized_ref).norm() / quantized_ref.norm()).item()
    norm_ratio = (out_t.norm() / quantized_ref.norm()).item()

    assert out.shape == (m, n)
    assert bool(torch.isfinite(out_t).all())
    assert kernel_cosine > 0.9999
    assert relative_l2 < 0.01
    assert abs(norm_ratio - 1.0) < 0.01
    assert bf16_cosine > 0.985


def test_a6w4_mfma32_variants_match_swizzle0_bitwise():
    m, n, k = 257, 513, 129
    a, b = _operands(m, n, k)
    operands = _packed(a, b)
    baseline = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=MFMA32_SWZ0_KERNEL)
    for kernel_name in (
        MFMA32_SMALL_KERNEL,
        MFMA32_GROUPED_KERNEL,
        MFMA32_LONG_K_KERNEL,
        DMABIG_SWZ0_KERNEL,
        DMABIG_GROUPED_KERNEL,
    ):
        actual = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=kernel_name)
        assert jnp.array_equal(actual, baseline), kernel_name


def test_a6w4_long_k_path_matches_swizzle0_bitwise():
    m, n, k = 256, 256, 6272
    a, b = _operands(m, n, k)
    operands = _packed(a, b)
    baseline = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=MFMA32_SWZ0_KERNEL)
    actual = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=MFMA32_LONG_K_KERNEL)
    assert jnp.array_equal(actual, baseline)


@pytest.mark.parametrize(
    "shape,specialized",
    [((9450, 5120, 5120), SPECIALIZED_SQUARE_KERNEL),
     ((9450, 13824, 5120), SPECIALIZED_UP_KERNEL),
     ((9450, 5120, 13824), SPECIALIZED_DOWN_KERNEL)],
    ids=("square", "ffn_up", "ffn_down"),
)
def test_a6w4_specialized_kernel_matches_default(shape, specialized):
    m, n, k = shape
    a, b = _operands(m, n, k)
    operands = _packed(a, b)
    baseline = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=default_a6w4_gemm_kernel(m, n, k))
    actual = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=specialized)
    assert jnp.array_equal(actual, baseline)


def _other_exact_rows():
    """Exact rows other than the three 9472-row Wan shapes, which have their own test."""
    try:
        return [row for row in _exact_rows() if int(row["exact_M"]) != 9472]
    except FileNotFoundError:
        return []


@pytest.mark.parametrize(
    "row",
    _other_exact_rows(),
    ids=lambda row: row["knl_name"].removeprefix("_ZN5aiter").split("f6f4gemm_bf16_per1x32Fp6Fp4_")[-1],
)
def test_a6w4_exact_shape_kernel_matches_default(row):
    """Every exact-shape manifest row must agree bitwise with the generic kernel."""
    m, n, k = (int(row[key]) for key in ("exact_M", "exact_N", "exact_K"))
    a, b = _operands(m, n, k)
    operands = _packed(a, b)
    baseline = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=default_a6w4_gemm_kernel(m, n, k))
    actual = gemm_a6w4_packed(*operands, m=m, n=n, k=k, kernel_name=row["knl_name"])
    assert jnp.array_equal(actual, baseline)


def test_a6w4_specialized_kernel_rejects_incompatible_shape():
    a, b = _operands(256, 256, 128)
    operands = _packed(a, b)
    with pytest.raises(jax.errors.JaxRuntimeError, match="exact padded shape"):
        gemm_a6w4_packed(
            *operands, m=256, n=256, k=128, kernel_name=SPECIALIZED_SQUARE_KERNEL
        ).block_until_ready()


def test_a6w4_swizzled_kernel_rejects_out_of_bound_shape():
    # padded M = 131328 exceeds the grouped kernel's swizzle bound of 131072.
    m, n, k = 131073, 256, 128
    a, b = _operands(m, n, k)
    operands = _packed(a, b)
    with pytest.raises(jax.errors.JaxRuntimeError, match="swizzle"):
        gemm_a6w4_packed(
            *operands, m=m, n=n, k=k, kernel_name=MFMA32_GROUPED_KERNEL
        ).block_until_ready()


def test_a6w4_unknown_kernel_is_rejected():
    a, b = _operands(256, 256, 128)
    operands = _packed(a, b)
    with pytest.raises(jax.errors.JaxRuntimeError, match="Unknown or unsupported"):
        gemm_a6w4_packed(
            *operands, m=256, n=256, k=128, kernel_name="_ZN5aiter3fooE"
        ).block_until_ready()


def test_a6w4_rejects_malformed_buffers():
    a, b = _operands(256, 256, 128)
    a_packed, b_packed, a_scale, b_scale = _packed(a, b)
    with pytest.raises(jax.errors.JaxRuntimeError, match="invalid shapes"):
        gemm_a6w4_packed(
            a_packed[:-16], b_packed, a_scale, b_scale, m=256, n=256, k=128
        ).block_until_ready()
    with pytest.raises(jax.errors.JaxRuntimeError, match="invalid shapes"):
        gemm_a6w4_packed(
            a_packed, b_packed, a_scale[:-16], b_scale, m=256, n=256, k=128
        ).block_until_ready()


@pytest.mark.parametrize("round_mode", range(4))
def test_a6w4_all_round_modes_run(round_mode):
    a, b = _operands(257, 259, 129)
    out = gemm_a6w4(a, b, round_mode=round_mode)
    assert bool(jnp.isfinite(out.astype(jnp.float32)).all())


@pytest.mark.parametrize("m,n,k,round_mode", [(257, 513, 129, 1), (513, 784, 1152, 2)])
def test_a6w4_matches_aiter(m, n, k, round_mode):
    """JAX quantize plus GEMM must agree bitwise with PyTorch AITER.

    Only the GEMM result is compared: AITER leaves guard-tile and row-padding
    bytes of the packed buffers unspecified, so those bytes are not a contract.
    """
    a6w4, a6w6, _, _, _ = _aiter_a6w4()
    a, b = _operands(m, n, k)
    x, w = _to_torch(a), _to_torch(b)

    x_packed, x_scale = a6w6.quant_mxfp6_gemm(x)
    w_packed, w_scale = a6w4.quant_mxfp4_gemm(w, round_mode=round_mode)
    kernel = a6w4_ops._select_kernel(m, n, k, None)
    jax_out = gemm_a6w4(a, b, round_mode=round_mode, kernel_name=kernel)
    aiter_out = a6w4.gemm_a6w4(x_packed, w_packed, x_scale, w_scale, m, n, k, kernelName=kernel)
    assert np.array_equal(np.asarray(jax_out), aiter_out.cpu().float().numpy())
