# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 tensor container: rowwise + columnwise FP4 data with E8M0 scales.

Mirrors TE's ``MXFP4Tensor`` storage concept. Either layout can be missing
(``None``) when the quantizer was configured with ``rowwise=False`` or
``columnwise=False``.

Use as the residual type in ``custom_vjp`` forward rules so the backward
pass can pick up pre-computed columnwise data without re-quantizing.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax


class Mxfp4Tensor(NamedTuple):
    """Packed MXFP4 tensor with optional rowwise + columnwise layouts.

    Attributes:
        rowwise_data:   ``[M, K/2]`` uint8, row-major FP4. ``None`` if rowwise
                        output was disabled.
        rowwise_scale:  ``[M_pad, Sp]`` uint8 E8M0 block scales for ``rowwise_data``.
        columnwise_data:
                        ``[K, M/2]`` uint8, column-major FP4 (physically the
                        quantized transpose). ``None`` if columnwise output
                        was disabled.
        columnwise_scale:
                        ``[K_pad, Sp]`` uint8 E8M0 block scales for
                        ``columnwise_data``.
    """

    rowwise_data: Optional[jax.Array] = None
    rowwise_scale: Optional[jax.Array] = None
    columnwise_data: Optional[jax.Array] = None
    columnwise_scale: Optional[jax.Array] = None

    @property
    def has_rowwise(self) -> bool:
        return self.rowwise_data is not None

    @property
    def has_columnwise(self) -> bool:
        return self.columnwise_data is not None

    def rowwise_tuple(self):
        """Return ``(rowwise_data, rowwise_scale)`` or raise if missing."""
        if not self.has_rowwise:
            raise ValueError("Mxfp4Tensor has no rowwise data; "
                             "construct the Quantizer with rowwise=True.")
        return self.rowwise_data, self.rowwise_scale

    def columnwise_tuple(self):
        """Return ``(columnwise_data, columnwise_scale)`` or raise if missing."""
        if not self.has_columnwise:
            raise ValueError("Mxfp4Tensor has no columnwise data; "
                             "construct the Quantizer with columnwise=True.")
        return self.columnwise_data, self.columnwise_scale


def mxfp4_tensor_from_rowwise(data, scale) -> Mxfp4Tensor:
    """Construct a rowwise-only Mxfp4Tensor."""
    return Mxfp4Tensor(rowwise_data=data, rowwise_scale=scale)


def mxfp4_tensor_from_dual(row_data, row_scale, col_data, col_scale) -> Mxfp4Tensor:
    """Construct a dual (row + col) Mxfp4Tensor."""
    return Mxfp4Tensor(
        rowwise_data=row_data,
        rowwise_scale=row_scale,
        columnwise_data=col_data,
        columnwise_scale=col_scale,
    )
