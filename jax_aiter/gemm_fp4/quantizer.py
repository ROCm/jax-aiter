# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 quantizer object — TE-parity abstraction over ``CastMxfp4JA`` / ``CastMxfp4DualJA``.

The quantizer owns the flags that control how BF16 -> MXFP4 conversion is
performed. It produces an ``Mxfp4Tensor`` (rowwise and/or columnwise packed
FP4 + E8M0 scales) in a single fused FFI kernel launch.

Role mapping (matches ``MXFP4BlockScalingRecipeState.make_quantizers`` in TE):

- **Input / activation quantizer**: ``rowwise=True``, ``columnwise=True``,
  ``shuffle_B_matrix_for_aiter=False`` (rowwise unshuffled, columnwise shuffled).
- **Weight quantizer**:                ``rowwise=True``, ``columnwise=True``,
  ``shuffle_B_matrix_for_aiter=True``  (both rowwise and columnwise shuffled).
- **Grad output quantizer**:          ``rowwise=True``, ``columnwise=True``,
  ``shuffle_B_matrix_for_aiter=False`` (both unshuffled; colwise is linear
  layout suitable as A operand for wgrad).

See :class:`~jax_aiter.gemm_fp4.mxfp4_tensor.Mxfp4Tensor` for the tensor layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from ..ops.gemm_fp4 import cast_mxfp4, cast_mxfp4_dual
from ..ffi.registry import register_ffi_target
from .fp4_utils import bf16_to_mxfp4, e8m0_shuffle, shuffle_weight
from .mxfp4_tensor import Mxfp4Tensor


@dataclass(frozen=True)
class MXFP4Quantizer:
    """Builder for MXFP4 tensors with block scaling.

    The flags are interpreted as follows:

    - ``rowwise``: produce ``Mxfp4Tensor.rowwise_data`` / ``rowwise_scale``.
    - ``columnwise``: produce ``Mxfp4Tensor.columnwise_data`` /
      ``columnwise_scale`` from the same kernel launch (single FFI call).
    - ``shuffle_B_matrix_for_aiter``: when ``True`` the **rowwise** FP4 output
      is B-preshuffle shuffled, suitable for use as the B operand of AITER's
      a4w4 kernel (forward and dA paths where this tensor is the weight).
      When ``False`` the rowwise output is linear (suitable as the A operand).
    - ``shuffle_colwise_fp4``: when ``True`` the **columnwise** FP4 output is
      B-preshuffle shuffled (suitable as B operand for dA, where the weight's
      columnwise data supplies ``B``). When ``False`` it is linear (suitable
      as A operand for wgrad / dB, where the input's or grad's columnwise
      data supplies ``A``).
    - ``shuffle_scales``: E8M0 scale shuffle (always ``True`` for AITER ASM).
    - ``use_hadamard``: apply Hadamard transform before quantization
      (experimental; off by default).

    See ``MXFP4Quantizer.for_weight()`` / ``for_activation()`` / ``for_grad()``
    for the canonical TE-parity presets.
    """

    rowwise: bool = True
    columnwise: bool = True
    shuffle_B_matrix_for_aiter: bool = False
    shuffle_colwise_fp4: bool = True
    shuffle_scales: bool = True
    use_hadamard: bool = False
    use_fused_kernel: bool = True

    # ------------------------------------------------------------------
    # Preset factories (match TE roles)
    # ------------------------------------------------------------------

    @classmethod
    def for_weight(cls, *, columnwise: bool = True, use_hadamard: bool = False):
        """Weight quantizer: rowwise + columnwise, both B-preshuffle shuffled.

        Rowwise output is used as the B operand of fprop GEMM
        (``out = A_input @ B_weight^T``).
        Columnwise output is used as the B operand of dgrad GEMM
        (``dA = grad @ B_weight_col^T``).
        """
        return cls(
            rowwise=True,
            columnwise=columnwise,
            shuffle_B_matrix_for_aiter=True,
            shuffle_colwise_fp4=True,
            shuffle_scales=True,
            use_hadamard=use_hadamard,
        )

    @classmethod
    def for_activation(cls, *, columnwise: bool = True, use_hadamard: bool = False):
        """Activation quantizer: rowwise linear + columnwise shuffled.

        Rowwise output is the A operand of fprop GEMM (``out = A @ B^T``).
        Columnwise output is the B operand of wgrad GEMM
        (``dB = grad_col @ A_input_col^T`` in AITER layout; see
        ``_fp4_ffi_partitioned_wgrad``). With ``shuffle_colwise_fp4=True``
        the columnwise layout is B-preshuffle shuffled.
        """
        return cls(
            rowwise=True,
            columnwise=columnwise,
            shuffle_B_matrix_for_aiter=False,
            shuffle_colwise_fp4=True,
            shuffle_scales=True,
            use_hadamard=use_hadamard,
        )

    @classmethod
    def for_grad(cls, *, columnwise: bool = True, use_hadamard: bool = False):
        """Grad-output quantizer: both rowwise and columnwise linear (unshuffled).

        Rowwise output is the A operand of dgrad GEMM.
        Columnwise output is the A operand of wgrad GEMM; unshuffled because
        it plays the A role (not B).
        """
        return cls(
            rowwise=True,
            columnwise=columnwise,
            shuffle_B_matrix_for_aiter=False,
            shuffle_colwise_fp4=False,
            shuffle_scales=True,
            use_hadamard=use_hadamard,
        )

    # ------------------------------------------------------------------
    # Main quantize entry point
    # ------------------------------------------------------------------

    def quantize(self, x_bf16) -> Mxfp4Tensor:
        """Quantize a BF16 tensor of shape ``[M, K]`` into an ``Mxfp4Tensor``.

        Uses the fused HIP cast kernel via ``CastMxfp4JA`` (rowwise only) or
        ``CastMxfp4DualJA`` (rowwise + columnwise) when ``use_fused_kernel``
        is True. Falls back to JAX quantization (``bf16_to_mxfp4`` +
        ``shuffle_weight`` / ``e8m0_shuffle``) otherwise; the JAX fallback
        only supports the rowwise output.
        """
        if not self.rowwise and not self.columnwise:
            raise ValueError(
                "MXFP4Quantizer requires at least one of rowwise or columnwise."
            )

        if self.use_fused_kernel and _fused_quant_available():
            return self._quantize_fused(x_bf16)
        return self._quantize_jax_fallback(x_bf16)

    # Convenience alias — matches TE spelling.
    def __call__(self, x_bf16) -> Mxfp4Tensor:
        return self.quantize(x_bf16)

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------

    def _quantize_fused(self, x_bf16) -> Mxfp4Tensor:
        if self.columnwise:
            row_fp4, row_scale, col_fp4, col_scale = cast_mxfp4_dual(
                x_bf16,
                shuffle_fp4=self.shuffle_B_matrix_for_aiter,
                shuffle_colwise_fp4=self.shuffle_colwise_fp4,
                shuffle_scales=self.shuffle_scales,
                use_hadamard=self.use_hadamard,
            )
            if not self.rowwise:
                row_fp4 = None
                row_scale = None
            return Mxfp4Tensor(
                rowwise_data=row_fp4,
                rowwise_scale=row_scale,
                columnwise_data=col_fp4,
                columnwise_scale=col_scale,
            )
        # rowwise only
        row_fp4, row_scale = cast_mxfp4(
            x_bf16,
            shuffle_fp4=self.shuffle_B_matrix_for_aiter,
            shuffle_scales=self.shuffle_scales,
            use_hadamard=self.use_hadamard,
        )
        return Mxfp4Tensor(rowwise_data=row_fp4, rowwise_scale=row_scale)

    def _quantize_jax_fallback(self, x_bf16) -> Mxfp4Tensor:
        if self.columnwise:
            raise NotImplementedError(
                "JAX-side MXFP4 quantization does not yet support columnwise output; "
                "enable the fused HIP kernel by building with 'make ja_mods' or set "
                "MXFP4Quantizer(use_fused_kernel=True)."
            )
        if self.use_hadamard:
            raise NotImplementedError(
                "JAX-side MXFP4 quantization does not support Hadamard transform."
            )
        packed, scales = bf16_to_mxfp4(x_bf16)
        if self.shuffle_B_matrix_for_aiter:
            packed = shuffle_weight(packed)
        if self.shuffle_scales:
            scales = e8m0_shuffle(scales)
        return Mxfp4Tensor(rowwise_data=packed, rowwise_scale=scales)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
_FUSED_QUANT_AVAILABLE = None


def _fused_quant_available() -> bool:
    """Detect once whether the fused HIP cast kernel FFI is loadable."""
    global _FUSED_QUANT_AVAILABLE
    if _FUSED_QUANT_AVAILABLE is None:
        try:
            register_ffi_target("CastMxfp4JA", "ROCM")
            register_ffi_target("CastMxfp4DualJA", "ROCM")
            _FUSED_QUANT_AVAILABLE = True
        except Exception:
            _FUSED_QUANT_AVAILABLE = False
    return _FUSED_QUANT_AVAILABLE


def reset_fused_quant_cache():
    """Test hook: force re-detection of the fused quant FFI availability."""
    global _FUSED_QUANT_AVAILABLE
    _FUSED_QUANT_AVAILABLE = None
