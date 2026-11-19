# SPDX-License-Identifier: Apache-2.0
"""JAX-AITER launchers built from the vendored Primus-Turbo MXFP4 grouped GEMM kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx

from .primus_turbo.grouped_gemm import grouped_gemm_mxfp4_kernel as _pt


def compile_grouped_mxfp4_nt(
    K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=None, span=16, cst_nt=False
):
    """Primus-Turbo's fused forward/dA launcher without its output-tail zeroing.

    Rows of ``C`` at or after the last group bound stay unwritten, so callers
    mask them wherever they are read.
    """
    K128 = K // 128
    k128_rd = (K if k_real is None else k_real) // 128
    gemm_k, attrs, n_blocks, b_ilv = _pt._build_grouped_mxfp4_nt_kernel(
        K,
        G,
        N,
        group_m=gm,
        num_xcds=xcd,
        group_n=gn,
        wlv=wlv,
        elgk=elgk,
        out_fp16=out_fp16,
        k_real=k_real,
        xcd_span=span,
        cst_nt=cst_nt,
    )
    preshuffle = _pt._build_grouped_mxfp4_ab_preshuffle(K128, G, N, k128_rd, b_ilv=b_ilv)
    b_pre_grid = _pt.ceildiv(
        G * _pt.ceildiv(N, 256) * 256 * K128, _pt._PRESHUF_FO * _pt._PRESHUF_BLK
    )

    @flyc.jit
    def launch(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        GO: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        slab_rows: fx.Int32,
        a_pre_grid: fx.Int32,
        grid_upper: fx.Int32,
        stream: fx.Stream,
    ):
        preshuffle(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
            grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_pt._PRESHUF_BLK, 1, 1), stream=stream
        )
        gemm_k(a8, b8, C, a_sp, b_sp, GO, c_m, c_n, slab_rows, value_attrs=attrs).launch(
            grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch, n_blocks
