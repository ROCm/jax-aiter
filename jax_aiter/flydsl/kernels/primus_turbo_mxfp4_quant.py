# SPDX-License-Identifier: Apache-2.0 AND MIT AND BSD-3-Clause
#
# The column-wise cast's decomposition into 32-row MX blocks, read with
# coalesced row-major loads and written column-major, derives from
# pytorch/ao#4357 ("Flydsl mxfp8 quantize", head fbefeb0):
#
#   Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#   Distributed under the BSD 3-Clause license.
#
# The row-wise cast's flat mapping over 32-element blocks follows AITER's
# ``per_1x32_mx_quant`` (aiter/ops/flydsl/kernels/mega_moe/quant.py):
#
#   Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#   Distributed under the MIT license.
#
# Primus-Turbo's dual cast, scale rule and fp4 convert are Apache-2.0, vendored
# under ``primus_turbo/``, whose ORIGIN.md lists every change to them.
"""JAX-AITER launchers for the vendored Primus-Turbo MXFP4 casts.

The dual runs Primus-Turbo's fused-dual tile. A one-direction cast has no
second direction to share that tile's LDS staging with, so the row-wise and the
column-wise cast read their source straight into registers with coalesced loads
and finish each 32-element block with Primus-Turbo's scale rule and fp4 convert.
Their gather forms read row ``r`` from source row ``IDX[r]`` instead.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .primus_turbo.quantization import mxfp4_quant_kernel as _q
from .primus_turbo.utils import buffer_ops

# Primus-Turbo reads the dual cast's tile constants from module globals while
# building and tracing, which JAX-AITER does during lowering. Pin the 128-row
# tile once; shapes that would pick another tile are rejected below. The
# source-load cache hint stays at its default (none); Primus-Turbo only turns
# it on for working sets between 256 MB and 896 MB.
_TILE = (128, 256)
_q._set_tile_geom(*_TILE)
_NT = 2  # buffer cache-policy bit 1: non-temporal

_WAVE = 64
_QUAD = 4  # lanes per 32-element block; each converts 8 elements into one fp4 dword
_ROW_STEPS = _QUAD  # blocks each lane has in flight; lane q stores step q's scale
_ROW_WAVE_BLOCKS = _ROW_STEPS * _WAVE // _QUAD
_ROW_WG_BLOCKS = _q.BLK // _WAVE * _ROW_WAVE_BLOCKS


def _row_block(raw, scale_rounding_bias):
    """A lane's fp4 dword of its 32-element block from the lane's 4 source dwords, and the block's biased E8M0 scale.

    The block's four lanes are xor-1 and xor-2 neighbours, so two shuffles give
    every one of them the block's amax.
    """
    bits = []
    for j in range(4):
        bits.append(raw[j] << 16)
        bits.append(raw[j] & 0xFFFF0000)
    vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in bits]
    amax = math.absf(vf[0])
    for i in range(1, 8):
        amax = amax.maximumf(math.absf(vf[i]))
    amax = amax.maximumf(amax.shuffle_xor(1, _WAVE))
    amax = amax.maximumf(amax.shuffle_xor(2, _WAVE))
    native_bits, biased = _q._compute_scale_native(
        Vec.from_elements([amax], fx.Float32).bitcast(fx.Int32)[0], scale_rounding_bias
    )
    scale = arith.bitcast(T.f32, native_bits)
    word = fx.Int32(0)
    for pair in range(4):
        word = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, word, vf[2 * pair], vf[2 * pair + 1], scale, pair)
    return word, biased


def _row_own_scale(biased, q):
    """Every lane of a block holds its scale: lane ``q`` stores step ``q``'s."""
    b = biased[_QUAD - 1]
    for c in range(_QUAD - 1):
        b = arith.select(q == c, biased[c], b)
    return b


def _build_row_kernel(load_aux: int, store_aux: int):
    """Row-wise MXFP4 cast of a row-major bf16 ``[R, C]``, ``C % 32 == 0``, for rows below a device ``int32[1]`` limit.

    Flat over the ``R * C / 32`` blocks of 32 elements, 64 per wave. Four lanes
    share a block, so each load instruction of a wave reads 1 KB of contiguous
    source and each fp4 store writes 256 B, and two xor shuffles reduce the
    block's amax over the four. Max is order-free, so the bytes are those of
    Primus-Turbo's dual. ``load_aux`` and ``store_aux`` are the cache policies
    of the source loads and of every store.
    """

    @flyc.kernel(name="pt_mxfp4_cast_rows", known_block_size=[_q.BLK, 1, 1])
    def _row_kernel(
        X: fx.Tensor,
        OUT: fx.Tensor,
        SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,  # int32[1] on device, at most R
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        limit_rsrc = buffer_ops.create_buffer_resource(ROW_LIMIT, max_size=True)
        limit = fx.Int32(buffer_ops.buffer_load(limit_rsrc, 0, vec_width=1, is_scalar=True))
        nblk = limit * (C >> 5)
        tid = fx.thread_idx.x
        lane = tid & (_WAVE - 1)
        quad = lane >> 2
        q = lane & (_QUAD - 1)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid >> 6))
        first = fx.block_idx.x * _ROW_WG_BLOCKS + wave * _ROW_WAVE_BLOCKS
        # Based at the wave's first block, so 32-bit offsets reach past 4 GiB of source.
        in_rsrc = buffer_ops.create_buffer_resource(
            X, max_size=True, base_byte_offset=arith.index_cast(T.index, first) * arith.index(64)
        )
        out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
        sc_rsrc = buffer_ops.create_buffer_resource(SC, max_size=True)
        if first < nblk:
            blocks, raws = [], []
            for c in range_constexpr(_ROW_STEPS):
                g = first + c * (_WAVE // _QUAD) + quad
                blocks.append(g)
                g = arith.select(g < nblk, g, nblk - 1)
                raws.append(
                    Vec(
                        buffer_ops.buffer_load(
                            in_rsrc, (g - first) * 16 + q * 4, vec_width=4, dtype=T.i32, cache_modifier=load_aux
                        )
                    )
                )
            biased = []
            for c in range_constexpr(_ROW_STEPS):
                word, b = _row_block(raws[c], SCALE_ROUNDING_BIAS)
                if blocks[c] < nblk:
                    buffer_ops.buffer_store(word, out_rsrc, blocks[c] * _QUAD + q, cache_modifier=store_aux)
                biased.append(b)
            b = _row_own_scale(biased, q)
            g = first + q * (_WAVE // _QUAD) + quad
            if g < nblk:
                buffer_ops.buffer_store(arith.trunci(T.i8, b & 0xFF), sc_rsrc, g, cache_modifier=store_aux)

    return _row_kernel


def _build_row_gather_kernel(cols: int, src_bytes: int, load_aux: int, store_aux: int):
    """:func:`_build_row_kernel` for the ``[R, cols]`` matrix whose row ``r`` is source row ``IDX[r]``, or zero for ``IDX[r] < 0``.

    The source is read through a buffer bounded at its ``src_bytes``; a
    negative index aims its block's load at that bound, where the hardware
    returns zeros, so the block quantizes to zero bytes.
    """
    blocks_per_row, row_dw, end_dw = cols // 32, cols // 2, src_bytes // 4

    @flyc.kernel(name="pt_mxfp4_cast_rows_gather", known_block_size=[_q.BLK, 1, 1])
    def _row_gather_kernel(
        X: fx.Tensor,
        IDX: fx.Tensor,  # int32[R]
        OUT: fx.Tensor,
        SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,  # int32[1] on device, at most R
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        limit_rsrc = buffer_ops.create_buffer_resource(ROW_LIMIT, max_size=True)
        limit = fx.Int32(buffer_ops.buffer_load(limit_rsrc, 0, vec_width=1, is_scalar=True))
        nblk = limit * blocks_per_row
        tid = fx.thread_idx.x
        lane = tid & (_WAVE - 1)
        quad = lane >> 2
        q = lane & (_QUAD - 1)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid >> 6))
        first = fx.block_idx.x * _ROW_WG_BLOCKS + wave * _ROW_WAVE_BLOCKS
        in_rsrc = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=src_bytes)
        idx_rsrc = buffer_ops.create_buffer_resource(IDX, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
        sc_rsrc = buffer_ops.create_buffer_resource(SC, max_size=True)
        if first < nblk:
            blocks, block_cols, sources = [], [], []
            for c in range_constexpr(_ROW_STEPS):
                g = first + c * (_WAVE // _QUAD) + quad
                blocks.append(g)
                g = arith.select(g < nblk, g, nblk - 1)
                row = g // blocks_per_row
                block_cols.append(g - row * blocks_per_row)
                sources.append(fx.Int32(buffer_ops.buffer_load(idx_rsrc, row, vec_width=1, dtype=T.i32)))
            # All four index loads issue before the first source load waits on one.
            raws = []
            for c in range_constexpr(_ROW_STEPS):
                offset = arith.select(
                    sources[c] < 0, fx.Int32(end_dw), sources[c] * row_dw + block_cols[c] * 16 + q * 4
                )
                raws.append(
                    Vec(buffer_ops.buffer_load(in_rsrc, offset, vec_width=4, dtype=T.i32, cache_modifier=load_aux))
                )
            biased = []
            for c in range_constexpr(_ROW_STEPS):
                word, b = _row_block(raws[c], SCALE_ROUNDING_BIAS)
                if blocks[c] < nblk:
                    buffer_ops.buffer_store(word, out_rsrc, blocks[c] * _QUAD + q, cache_modifier=store_aux)
                biased.append(b)
            b = _row_own_scale(biased, q)
            g = first + q * (_WAVE // _QUAD) + quad
            if g < nblk:
                buffer_ops.buffer_store(arith.trunci(T.i8, b & 0xFF), sc_rsrc, g, cache_modifier=store_aux)

    return _row_gather_kernel


_COL_TILE = 2 * _WAVE  # tile columns; each lane owns two adjacent ones, one dword per row


def _col_stack(rows: int) -> int:
    """32-row blocks per column-wise workgroup, one wave each: 8 fill a 128 B output line."""
    return 8 if rows % 256 == 0 else 4


def _build_col_kernel(stack: int, gather_src_bytes: int | None = None):
    """Column-wise MXFP4 cast of a row-major bf16 ``[R, C]`` into ``[C, R/2]`` packed and ``[C, R/32]`` scales.

    ``R % (32 * stack) == 0`` and ``C % 128 == 0``. A workgroup of ``stack``
    waves covers ``32 * stack`` rows (grid y) by 128 columns (grid x), and is
    skipped when its first row is at or past a device ``int32[1]`` limit. Wave
    ``w`` reads the tile's rows ``[32 w, 32 w + 32)`` one dword per lane per
    row, so each load instruction reads 256 contiguous bytes and a lane holds
    its two columns' 32-row blocks in registers; Primus-Turbo's micro-block
    finish quantizes each. The packed blocks and scales then go through LDS so
    that each output row's ``16 * stack`` bytes leave in whole-line stores.

    With ``gather_src_bytes``, row ``r`` is source row ``IDX[r]``, or zero for
    ``IDX[r] < 0``: the source is read through a buffer bounded at
    ``gather_src_bytes``, and a negative index aims its row's loads at that
    bound, where the hardware returns zeros.
    """
    threads = _WAVE * stack
    seg_words = 4 * stack  # one output row's packed tile, in i32
    chunks = _COL_TILE * stack // threads  # 16 B readback chunks per thread

    @fx.struct
    class _ColSS:
        q: fx.Array[fx.Int32, _COL_TILE * seg_words, 16]
        s: fx.Array[fx.Int32, _COL_TILE * stack, 16]

    def quantize_to_lds(lds, raws, lane, wave, scale_rounding_bias):
        for half in range(2):
            bits = [fx.Int32(r) << 16 if half == 0 else fx.Int32(r) & 0xFFFF0000 for r in raws]
            words, biased = _q._finish_microblock(bits, False, scale_rounding_bias)
            k_row = lane * 2 + half
            # Slots XOR-swizzled by the writing lane spread a write phase over all banks.
            slot = wave ^ (lane & (stack - 1))
            _q._lds_store_vec4(lds.q.ptr, k_row * seg_words + slot * 4, Vec.from_elements(words, fx.Int32))
            _q._lds_store1(lds.s.ptr, wave * _COL_TILE + k_row, biased)

    def store_packed(lds, out_rsrc, tid, k_tile, row_base, R):
        for c in range(chunks):
            chunk = c * threads + tid
            k_row = chunk // stack
            seg = chunk % stack
            slot = seg ^ ((k_row >> 1) & (stack - 1))
            buffer_ops.buffer_store(
                _q._lds_load_vec4(lds.q.ptr, k_row * seg_words + slot * 4),
                out_rsrc,
                (k_tile + k_row) * (R >> 3) + (row_base >> 3) + seg * 4,
            )

    def scale_words(lds, tid):
        scales = [_q._lds_load1(lds.s.ptr, w * _COL_TILE + tid) for w in range(stack)]
        packed = [
            scales[4 * w] | (scales[4 * w + 1] << 8) | (scales[4 * w + 2] << 16) | (scales[4 * w + 3] << 24)
            for w in range(stack // 4)
        ]
        return Vec.from_elements(packed, fx.Int32) if len(packed) > 1 else packed[0]

    if gather_src_bytes is None:

        @flyc.kernel(name="pt_mxfp4_cast_cols", known_block_size=[threads, 1, 1])
        def _col_kernel(
            X: fx.Tensor,
            OUT: fx.Tensor,
            SC: fx.Tensor,
            ROW_LIMIT: fx.Tensor,  # int32[1] on device
            R: fx.Int32,
            C: fx.Int32,
            SCALE_ROUNDING_BIAS: fx.Int32,
        ):
            limit_rsrc = buffer_ops.create_buffer_resource(ROW_LIMIT, max_size=True)
            limit = fx.Int32(buffer_ops.buffer_load(limit_rsrc, 0, vec_width=1, is_scalar=True))
            lds = fx.SharedAllocator().allocate(_ColSS).peek()
            tid = fx.thread_idx.x
            lane = tid & (_WAVE - 1)
            wave = fx.Int32(rocdl.readfirstlane(T.i32, tid >> 6))
            k_block = fx.block_idx.x
            row_base = fx.block_idx.y * (32 * stack)
            if row_base < limit:
                # Based at the tile's first row, so 32-bit offsets reach past 4 GiB of source.
                in_rsrc = buffer_ops.create_buffer_resource(
                    X,
                    max_size=True,
                    base_byte_offset=arith.index_cast(T.index, row_base)
                    * arith.index_cast(T.index, C)
                    * arith.index(2),
                )
                out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
                sc_rsrc = buffer_ops.create_buffer_resource(SC, max_size=True)
                row_dw = C >> 1
                first_dw = (wave * 32) * row_dw + k_block * _WAVE + lane
                raws = [
                    buffer_ops.buffer_load(in_rsrc, first_dw + i * row_dw, vec_width=1, dtype=T.i32)
                    for i in range_constexpr(32)
                ]
                quantize_to_lds(lds, raws, lane, wave, SCALE_ROUNDING_BIAS)
                fx.barrier()
                k_tile = k_block * _COL_TILE
                store_packed(lds, out_rsrc, tid, k_tile, row_base, R)
                if tid < _COL_TILE:
                    buffer_ops.buffer_store(
                        scale_words(lds, tid),
                        sc_rsrc,
                        (k_tile + tid) * (R >> 5) + (row_base >> 5),
                        offset_is_bytes=True,
                    )

        return _col_kernel

    end_dw = gather_src_bytes // 4

    @flyc.kernel(name="pt_mxfp4_cast_cols_gather", known_block_size=[threads, 1, 1])
    def _col_gather_kernel(
        X: fx.Tensor,
        IDX: fx.Tensor,  # int32[R]
        OUT: fx.Tensor,
        SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,  # int32[1] on device
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        limit_rsrc = buffer_ops.create_buffer_resource(ROW_LIMIT, max_size=True)
        limit = fx.Int32(buffer_ops.buffer_load(limit_rsrc, 0, vec_width=1, is_scalar=True))
        lds = fx.SharedAllocator().allocate(_ColSS).peek()
        tid = fx.thread_idx.x
        lane = tid & (_WAVE - 1)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid >> 6))
        k_block = fx.block_idx.x
        row_base = fx.block_idx.y * (32 * stack)
        if row_base < limit:
            in_rsrc = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=gather_src_bytes)
            idx_rsrc = buffer_ops.create_buffer_resource(IDX, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
            sc_rsrc = buffer_ops.create_buffer_resource(SC, max_size=True)
            row_dw = C >> 1
            lane_dw = k_block * _WAVE + lane
            first_row = row_base + wave * 32
            sources = []
            for j in range_constexpr(8):
                four = Vec(buffer_ops.buffer_load(idx_rsrc, first_row + 4 * j, vec_width=4, is_scalar=True))
                sources += [fx.Int32(four[e]) for e in range_constexpr(4)]
            raws = [
                buffer_ops.buffer_load(
                    in_rsrc,
                    arith.select(sources[i] < 0, fx.Int32(end_dw), sources[i] * row_dw) + lane_dw,
                    vec_width=1,
                    dtype=T.i32,
                )
                for i in range_constexpr(32)
            ]
            quantize_to_lds(lds, raws, lane, wave, SCALE_ROUNDING_BIAS)
            fx.barrier()
            k_tile = k_block * _COL_TILE
            store_packed(lds, out_rsrc, tid, k_tile, row_base, R)
            if tid < _COL_TILE:
                buffer_ops.buffer_store(
                    scale_words(lds, tid),
                    sc_rsrc,
                    (k_tile + tid) * (R >> 5) + (row_base >> 5),
                    offset_is_bytes=True,
                )

    return _col_gather_kernel


def _build_row_limited_kernel():
    """Primus-Turbo's dual tile, skipped for row tiles at or past a device ``int32[1]`` limit.

    Needs the row-tile-major block order (``col_locality`` off), where block
    ``b`` covers rows ``[(b // col_tiles) * 128, +128)``. Skipped tiles write
    neither output.
    """
    dual_lds = _q._make_dual_struct(False)

    @flyc.kernel(known_block_size=[_q.BLK, 1, 1])
    def _dual_kernel_row_limited(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,  # int32[1] on device
        R: fx.Int32,
        C: fx.Int32,
        CP: fx.Int32,
        RP: fx.Int32,
        SR_SEED: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
    ):
        limit_rsrc = buffer_ops.create_buffer_resource(ROW_LIMIT, max_size=True)
        limit = fx.Int32(buffer_ops.buffer_load(limit_rsrc, 0, vec_width=1, is_scalar=True))
        lds = fx.SharedAllocator().allocate(dual_lds).peek()
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        col_tiles = C // _TILE[1]
        if (bid // col_tiles) * _TILE[0] < limit:
            _q._emit_dual_body(
                False,
                False,
                False,
                False,
                lds,
                tid,
                X,
                ROW_OUT,
                ROW_SC,
                COL_OUT,
                COL_SC,
                R,
                C,
                bid,
                SCALE_ROUNDING_BIAS,
                ncblk=col_tiles,
                CP=CP,
                RP=RP,
                sr_seed=SR_SEED,
                sr_gbid=bid,
            )

    return _dual_kernel_row_limited


def _check_tile(rows: int, cols: int):
    if rows % _TILE[0] or cols % _TILE[1]:
        raise ValueError(f"dual cast needs rows % 128 == 0 and cols % 256 == 0, got {rows}x{cols}")
    if _q._pick_tile_geom(rows, cols) != _TILE:
        raise NotImplementedError(f"dual cast tile for {rows}x{cols} is not {_TILE}")


@functools.lru_cache(maxsize=None)
def compile_mxfp4_one_direction(rows: int, cols: int, rowwise: bool):
    """The row-wise or the transposed column-wise half of :func:`compile_mxfp4_dual`.

    Always row-limited: a device ``int32[1]`` operand follows the two outputs.
    Returns ``(launch, grid_x)``. The row-wise cast takes any ``cols % 32 == 0``
    and streams its source and outputs once each, so every load and store is
    non-temporal. The column-wise cast takes ``rows % 128 == 0`` and
    ``cols % 128 == 0``.
    """
    if rowwise:
        if cols % 32:
            raise ValueError(f"row-wise cast needs cols % 32 == 0, got {rows}x{cols}")
        kernel, threads = _build_row_kernel(_NT, _NT), _q.BLK
        grid_x = (rows * cols // 32 + _ROW_WG_BLOCKS - 1) // _ROW_WG_BLOCKS
        grid_y = 1
    else:
        if rows % 128 or cols % _COL_TILE:
            raise ValueError(f"column-wise cast needs rows % 128 == 0 and cols % 128 == 0, got {rows}x{cols}")
        stack = _col_stack(rows)
        kernel, threads = _build_col_kernel(stack), _WAVE * stack
        grid_x = cols // _COL_TILE
        grid_y = rows // (32 * stack)

    @flyc.jit
    def launch_one_direction(
        X: fx.Tensor,
        OUT: fx.Tensor,
        SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(X, OUT, SC, ROW_LIMIT, R, C, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, grid_y, 1), block=(threads, 1, 1), stream=stream
        )

    return launch_one_direction, grid_x


@functools.lru_cache(maxsize=None)
def compile_mxfp4_gather(src_rows: int, rows: int, cols: int, rowwise: bool):
    """:func:`compile_mxfp4_one_direction` of the ``[rows, cols]`` matrix whose row ``r`` is source row ``IDX[r]``, or zero for ``IDX[r] < 0``.

    The bf16 source is ``[src_rows, cols]``; ``IDX`` (``int32[rows]``) follows
    it. A negative index's loads go to the end of the source, so it and a row of
    32-bit offsets past it must stay below 4 GiB.
    """
    src_bytes = src_rows * cols * 2
    if src_bytes + 4 * cols > 1 << 32:
        raise ValueError(f"gather source {src_rows}x{cols} bf16 leaves no 32-bit offset past its end")
    if rowwise:
        if cols % 32:
            raise ValueError(f"row-wise cast needs cols % 32 == 0, got {rows}x{cols}")
        kernel, threads = _build_row_gather_kernel(cols, src_bytes, _NT, _NT), _q.BLK
        grid_x = (rows * cols // 32 + _ROW_WG_BLOCKS - 1) // _ROW_WG_BLOCKS
        grid_y = 1
    else:
        if rows % 128 or cols % _COL_TILE:
            raise ValueError(f"column-wise cast needs rows % 128 == 0 and cols % 128 == 0, got {rows}x{cols}")
        stack = _col_stack(rows)
        kernel, threads = _build_col_kernel(stack, src_bytes), _WAVE * stack
        grid_x = cols // _COL_TILE
        grid_y = rows // (32 * stack)

    @flyc.jit
    def launch_gather(
        X: fx.Tensor,
        IDX: fx.Tensor,
        OUT: fx.Tensor,
        SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(X, IDX, OUT, SC, ROW_LIMIT, R, C, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, grid_y, 1), block=(threads, 1, 1), stream=stream
        )

    return launch_gather, grid_x


@functools.lru_cache(maxsize=None)
def compile_mxfp4_dual(rows: int, cols: int, row_limited: bool = False):
    """Row-wise and transposed column-wise MXFP4 cast of ``[rows, cols]`` from one read.

    No Hadamard rotation, stochastic rounding or 2D blocks, and no padding:
    ``rows % 128 == 0`` and ``cols % 256 == 0``. ``row_limited`` adds a device
    ``int32[1]`` operand after the outputs; row tiles starting at or past it
    are not written. Returns ``(launch, grid_x)``.
    """
    _check_tile(rows, cols)
    col_locality = cols > rows and not _q._co_wide(*_TILE)
    grid_x = (rows // _TILE[0]) * (cols // _TILE[1])

    if not row_limited:
        kernel = _q._build_dual_kernel(False, False, col_locality=col_locality)

        @flyc.jit
        def launch(
            X: fx.Tensor,
            ROW_OUT: fx.Tensor,
            ROW_SC: fx.Tensor,
            COL_OUT: fx.Tensor,
            COL_SC: fx.Tensor,
            R: fx.Int32,
            C: fx.Int32,
            CP: fx.Int32,
            RP: fx.Int32,
            SR_SEED: fx.Int32,
            SCALE_ROUNDING_BIAS: fx.Int32,
            grid_x: fx.Int32,
            stream: fx.Stream,
        ):
            kernel(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, C, CP, RP, SR_SEED, SCALE_ROUNDING_BIAS).launch(
                grid=(grid_x, 1, 1), block=(_q.BLK, 1, 1), stream=stream
            )

        return launch, grid_x

    if col_locality:
        raise NotImplementedError(f"row-limited dual cast needs cols <= rows, got {rows}x{cols}")
    kernel = _build_row_limited_kernel()

    @flyc.jit
    def launch_row_limited(
        X: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        ROW_LIMIT: fx.Tensor,
        R: fx.Int32,
        C: fx.Int32,
        CP: fx.Int32,
        RP: fx.Int32,
        SR_SEED: fx.Int32,
        SCALE_ROUNDING_BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(X, ROW_OUT, ROW_SC, COL_OUT, COL_SC, ROW_LIMIT, R, C, CP, RP, SR_SEED, SCALE_ROUNDING_BIAS).launch(
            grid=(grid_x, 1, 1), block=(_q.BLK, 1, 1), stream=stream
        )

    return launch_row_limited, grid_x
