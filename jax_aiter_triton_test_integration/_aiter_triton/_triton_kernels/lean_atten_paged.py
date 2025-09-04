"""
Lean Attention + Paged Attention
================================
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480, enhanced
with Paged Attention from https://arxiv.org/abs/2309.06180, for the decode phase.
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.

It currently supports ragged batching decode

TO be added features:
- Add GQA support
- Misc
    - N_CTX with non-integer number of BLOCK_N (pad zeros or add mask)
    -
"""

import triton
import triton.language as tl


@triton.jit
def find_group(x):
    group_id = 0
    total_blocks = 0
    while total_blocks + (group_id + 1) <= x:
        total_blocks += group_id + 1
        group_id += 1
    group_size = group_id + 1
    return group_id, group_size, total_blocks


@triton.jit
def la_persistent_paged(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    kv_block_tables,
    kv_shape,
    batch_num_block_n,
    locks,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oh,
    stride_om,
    stride_on,
    stride_oph,
    stride_opm,
    stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    batch_size: tl.constexpr,
    num_m_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            current_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    # Loop context length
    while iter < cta_end_tile_gid:
        # Calculate index of current head output tile
        # The tiles_per_head is the numner of BLOCK_N in the K/V sequence
        tile_head_idx = iter // tiles_per_head

        # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        # [tile_iter, tile_iter_end) are in the form of global tile id
        tile_idx = tile_head_idx * batch_size
        tile_iter = tile_head_idx * tiles_per_head
        if batch_size == 1:
            req_size = tiles_per_head
        else:
            req_size = tl.load(batch_num_block_n)
        tile_iter_end = tile_iter + req_size
        for b in range(1, batch_size):
            next_req_size = tl.load(batch_num_block_n + b)
            local_head_iter = iter % tiles_per_head
            if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                tile_iter = tile_iter + req_size
                tile_idx = tile_idx + b
                tile_iter_end = tile_iter + (next_req_size - req_size)
            req_size = next_req_size
        # Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False

        KV_block_tables_ptr = kv_block_tables + iter
        kv_offset = tile_head_idx * stride_kh

        K_base = K + kv_offset
        V_base = V + kv_offset

        Q_base = Q + tile_idx * (stride_qh // batch_size)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        acc, l_i, m_i = _attn_lean_tile(
            acc,
            l_i,
            m_i,
            Q_base,
            stride_qm,
            stride_qk,
            kv_shape,
            K_base,
            V_base,
            KV_block_tables_ptr,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            tile_idx,
            local_iter,
            local_iter_end,
        )
        # initialize pointer to m and l
        m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # lean output tile epilogue
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)

        if not host_block:
            # Update pointers of partial results M[cta], L[cta], O[cta]
            mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
            lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
            op_ptrs = (
                Op
                + current_pid * stride_oph
                + offs_m[:, None] * stride_opm
                + offs_k[None, :] * stride_opn
            )

            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            tl.debug_barrier()
            # According to streamK gemm, store + cache_modifier won't work universally
            # atomic_xchg is better solution but a less performant variant
            tl.atomic_xchg(locks + current_pid, 1)

        if host_block and finishing_block:
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
            o_h_offs = Out + tile_idx * (stride_oh // batch_size)
            o_ptrs = (
                o_h_offs + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
            )
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        if host_block and not finishing_block:
            # if not finishing_block: # another CTA is processing the end of the output tile and store partial results
            o_h_offs = Out + tile_idx * (stride_oh // batch_size)
            o_ptrs = (
                o_h_offs + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
            )

            last_cta = current_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end):
                if last_cta < high_load_wgs:
                    if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg - 1

                last_cta += 1
                split += 1
            # Next, load nonHost partial restult
            for cta in range((current_pid + 1), last_cta):
                # According to treamK gemm, atomic_cas is universal solution but less performant
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                    pass

                # Partial results are stored in [nonHost, Host-nonFinishing] layout
                offs_mplp = cta * BLOCK_M + tl.arange(0, BLOCK_M)
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                op_h_offs = Op + cta * stride_oph
                op_ptrs = (
                    op_h_offs
                    + offs_m[:, None] * stride_opm
                    + offs_k[None, :] * stride_opn
                )
                m_cta = tl.load(mp_ptrs)
                l_cta = tl.load(lp_ptrs)
                acc_cta = tl.load(op_ptrs)

                # m_i is the host CTA's m, m_cta is other nonHost CTA's m
                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                # update m, l
                m_i = m_new
                l_i = l_new
            # host non-finishing CTA write final result to memory
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)


@triton.jit
def _attn_lean_tile(
    acc,
    l_i,
    m_i,
    Q_base,
    stride_qm,
    stride_qk,
    kv_shape,
    K_base,
    V_base,
    KV_block_tables_ptr,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    qk_scale: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    tile_idx,
    local_iter,
    local_iter_end,
):  #
    Q_block_ptr = tl.make_block_ptr(
        base=Q_base,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)

    K_block_ptr = tl.make_block_ptr(
        base=K_base,
        shape=(HEAD_DIM, kv_shape),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),  # K parent tensor shape [Z, H, CTX, HEAD_DIM]
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_base,
        shape=(kv_shape, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    for iter in range(local_iter, local_iter_end):
        # update k/v pointer
        kv_block_id = tl.load(KV_block_tables_ptr, cache_modifier=".cg")
        V_bptr = tl.advance(V_block_ptr, (kv_block_id * BLOCK_N, 0))
        K_bptr = tl.advance(K_block_ptr, (0, kv_block_id * BLOCK_N))

        # -- compute qk ----
        k = tl.load(K_bptr, cache_modifier=".cg")
        qk = tl.dot(q, k)
        qk = qk * qk_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)  # p.shape = [BLOCK_M, BLOCK_N]
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = (
            acc * alpha[:, None]
        )  # Scale each row of acc by the corresponding elements in alpha
        v = tl.load(V_bptr, cache_modifier=".cg")  # v.shape = [BLOCK_N, HEAD_DIM]
        acc += tl.dot(p.to(v.dtype), v)  # acc.shape = [BLOCK_M, HEAD_DIM]
        # -- update l_i
        l_ij = tl.sum(p, 1)  # rowsum(p)
        l_i = l_i * alpha + l_ij
        # update m_i
        m_i = m_ij.to(m_i.dtype)

        # update KV block tables pointer
        KV_block_tables_ptr += 1

    return acc, l_i, m_i
