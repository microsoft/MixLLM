import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_loop_inner_main(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,  #
    paged_block_list,
    cur_group_batch,
    stride_kvp,
    stride_bt_0,
    low_idx,
    high_idx,
    off_kv,


):

    # q = (q * qk_scale).to(k_ptrs.dtype.element_ty)


    for idx in range(low_idx, high_idx, 1):
        # -- update paged_block ----

        paged_block_id = tl.load(paged_block_list + cur_group_batch * stride_bt_0 + idx)
        # -- load_kv ----
        # TODO(xinji1): NOT the best loading conditions
        kv_block_offset = paged_block_id * stride_kvp + off_kv
        k = tl.load(
            k_ptrs + kv_block_offset
        )  # in common stage, there're always `common_len % page_block_size == 0 and page_block_size == BLOCK_N`


        v = tl.load(v_ptrs + kv_block_offset)
        # -- compute qk ----
        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T)
        # -- compute m_ij, p, l_ij ----
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])

        l_ij = tl.sum(p, 1)

        # -- update m_i, l_i ----
        alpha = (tl.math.exp2(m_i - m_ij)).to(q.dtype)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # ---- update acc ----
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

    return acc, l_i, m_i


@triton.jit
def _paged_attn_loop_inner_tail(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,  #
    paged_block_list,
    cur_group_batch,
    stride_kvp,
    stride_bt_0,
    low_idx,
    high_idx,
    high,
    off_kv,
    offs_m,
    offs_n,
    history_len,
    STAGE: tl.constexpr,
    padded_kv_group_num:tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):


    for idx in range(low_idx, high_idx, 1):
        # -- update paged_block ----

        # paged_block_offset = tl.load(paged_block_list  + cur_group_batch * stride_bt_0 + idx) * paged_block_size + offs_n

        paged_block_id = tl.load(paged_block_list + cur_group_batch * stride_bt_0 + idx)
        # -- load_kv ----
        # TODO(xinji1): NOT the best loading conditions
        kv_block_offset = paged_block_id *  stride_kvp + off_kv

        # -- load_kv ----
        # TODO(xinji1): NOT the best loading conditions

        mask_offset = idx * BLOCK_N + offs_n
        kv_mask = mask_offset[:, None] < high
        k = tl.load(k_ptrs + kv_block_offset, mask=kv_mask, other=0.0)  # TODO: EVEN_M/ EVEN_N
        v = tl.load(v_ptrs + kv_block_offset, mask=kv_mask, other=0.0)

        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if STAGE == 2:# causal part
            mask = tl.ravel( (history_len + offs_m[:, None]) + tl.zeros([padded_kv_group_num], dtype = tl.int32)[None, :] )[:, None]  >= mask_offset[None, :]

            qk = tl.where(
                mask, qk, float("-inf")
            )  # NOTE: inf - inf = nan, and nan will leads to error. Old expr `qk = qk + tl.where()` is unsafe.
        # TODO(xinji1):  maybe we can delete the following mask?
        qk = tl.where( mask_offset[None, :] < high, qk, float("-inf"))
        # -- compute m_ij, p, l_ij ----
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # -- update m_i, l_i ----
        alpha = (tl.math.exp2(m_i - m_ij)).to(q.dtype)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # ---- update acc ----
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

    return acc, l_i, m_i

@triton.jit
def invoke_paged_pacs_flash_attn_main_tail_gqa_decoding_1(
    Q,
    K,
    V,
    Out_f,
    LSE,
    cur_paged_block_list,
    stride_qbs,
    stride_qh,
    stride_kvp,
    stride_kvbs,
    stride_kvh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_bt_0,
    cur_group_batch,
    cur_kv_head,
    cur_kv_seq_len,
    cur_q_start_index,
    LOGSUMEXP,
    qk_scale: tl.constexpr,
    kv_group_num: tl.constexpr,
    PADDED_KV_GROUP_NUM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # in decoding scenarios, since cur_q_seq_len is always equal to 1, here we try to pad `kv_group_num` to 16 so that,
    # each time q_tile reads a [PADDED_GROUP_SIZE, BLOCK_D_MODEL], and it's friendly to gpu's calculation

    # grid = ( 1 or num_partions , request_num, kv_head )
    # BLOCK_M -> PADDED_KV_GROUP_NUM


    # for each tile

    offs_head = tl.arange(0, PADDED_KV_GROUP_NUM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # for qkv,

    off_q = cur_q_start_index * stride_qbs + (cur_kv_head * kv_group_num + offs_head[:, None]) * stride_qh + offs_d[None, :]
    off_kv =  cur_kv_head * stride_kvh  + offs_n[:,None] * stride_kvbs + offs_d[None, :]


    if kv_group_num < PADDED_KV_GROUP_NUM:
        q = tl.load(Q + off_q, mask=offs_head[:, None] < kv_group_num , other=0.0)
    else:
        q = tl.load(Q+  off_q)
    # q = q.reshape(q, (BLOCK_M * kv_group_num, BLOCK_DMODEL), can_reorder=True)
    q = (q * qk_scale).to(K.dtype.element_ty)
    # initial m_i, l_i, acc
    m_i = tl.zeros([PADDED_KV_GROUP_NUM], dtype=tl.float32) - float("inf")  # initial m_i-> maximum of each row in q*k ;
    l_i = tl.zeros([PADDED_KV_GROUP_NUM], dtype=tl.float32)  # l_i -> sum() of each row in (q*k - m_i) ;
    acc = tl.zeros([PADDED_KV_GROUP_NUM, BLOCK_DMODEL], dtype=tl.float32)  # the final result


    low = 0


    k_padded = True
    if cur_kv_seq_len < BLOCK_N:
        seqlen_k_faligned = 0 # floor aligned
        seqlen_k_faligned = seqlen_k_faligned.to(tl.int64)

    elif cur_kv_seq_len % BLOCK_N:
        extra_tokens_n = cur_kv_seq_len % BLOCK_N
        seqlen_k_faligned = cur_kv_seq_len - extra_tokens_n
        seqlen_k_faligned = seqlen_k_faligned.to(tl.int64)
    else:
        k_padded = False
        seqlen_k_faligned = cur_kv_seq_len
        seqlen_k_faligned = seqlen_k_faligned.to(tl.int64)
    # Step 1: all inner loop without any masks
    # attention inner loop

    high = seqlen_k_faligned


    high_unroll_block_idx =  high // BLOCK_N

    # tl.device_print("tesd", new_BLOCK_M)
    acc, l_i, m_i = _paged_attn_loop_inner_main(
        acc,
        l_i,
        m_i,
        q,
        K,
        V,
        cur_paged_block_list,
        cur_group_batch,
        stride_kvp,
        stride_bt_0,
        low,
        high_unroll_block_idx,
        off_kv,
    )

    # tl.debug_barrier()
    if k_padded :
        tl.debug_barrier()
        real_high = cur_kv_seq_len

        real_high_unroll_block_idx = tl.cdiv(real_high, BLOCK_N)
        acc, l_i, m_i = _paged_attn_loop_inner_tail(
            acc,
            l_i,
            m_i,
            q,
            K,
            V,
            cur_paged_block_list,
            cur_group_batch,
            stride_kvp,
            stride_bt_0,
            high_unroll_block_idx,
            real_high_unroll_block_idx,
            real_high,
            off_kv,
            offs_head,
            offs_n,  #
            0,
            STAGE=1,
            padded_kv_group_num=kv_group_num,
            BLOCK_M=PADDED_KV_GROUP_NUM,
            BLOCK_N=BLOCK_N,

        )


    acc = acc / l_i[:, None]  # final_result correction

    # # saving

    off_o = cur_q_start_index * stride_obs + (cur_kv_head * kv_group_num + offs_head[:, None]) * stride_oh + offs_d[None, :]

    # off_o = cur_q_start_index * stride_obs + flatten_o_ret[:, None] * stride_oh + offs_d[None, :]
    off_lse = cur_q_start_index * stride_lse_bs + cur_kv_head * kv_group_num + offs_head

    if kv_group_num < PADDED_KV_GROUP_NUM:
        if LOGSUMEXP:
            m_i += tl.math.log2(l_i)  # for logexpsum
            tl.store(LSE + off_lse, m_i, mask= offs_head < kv_group_num )
        tl.store(Out_f + off_o, acc, mask= offs_head[:, None] < kv_group_num)
    else:
        if LOGSUMEXP:
            m_i += tl.math.log2(l_i)  # for logexpsum
            tl.store(LSE + off_lse, m_i)
        tl.store(Out_f + off_o, acc)


    return


@triton.jit
def invoke_paged_pacs_flash_attn_main_tail_gqa(
    Q,
    K,
    V,
    Out_f,
    LSE,
    cur_paged_block_list,
    stride_qbs,
    stride_qh,
    stride_kvp,
    stride_kvbs,
    stride_kvh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_bt_0,
    start_m,
    cur_group_batch,
    cur_kv_head,
    cur_q_seq_len,
    cur_kv_seq_len,
    cur_q_start_index,
    LOGSUMEXP,
    qk_scale: tl.constexpr,
    kv_group_num: tl.constexpr,
    padded_kv_group_num: tl.constexpr,
    padded_kv_group_TF: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):

    # for each tile
    block_start = BLOCK_M * start_m
    if block_start >= cur_q_seq_len:
        return
    offs_m = block_start + tl.arange(0, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qbs_qh = stride_qbs // stride_qh

    if block_start + BLOCK_M < cur_q_seq_len:
        q_padded = False
    else:
        q_padded = True
    # for qkv,
    offs_head = cur_kv_head * kv_group_num +  tl.arange(0, padded_kv_group_num)
    ret = (qbs_qh * offs_m)[:, None]  + offs_head[None, :]
    flatten_ret = tl.ravel(ret)


    # TODO(xinji1): check
    # TODO(xinji1): kv_group = 7 or padded to 8?

    off_q = cur_q_start_index * stride_qbs + flatten_ret[:, None] * stride_qh + offs_d[None, :]
    off_kv =  cur_kv_head * stride_kvh  + offs_n[:,None] * stride_kvbs + offs_d[None, :]
    qo_mask = tl.ravel(offs_m[:, None] +tl.zeros([padded_kv_group_num], dtype = tl.int32)[None, :])
    head_mask = tl.ravel( tl.zeros([BLOCK_M],  dtype = tl.int32 )[:,None] + tl.arange(0, padded_kv_group_num)[None,:] )
    if q_padded:
        if padded_kv_group_TF:
            q = tl.load(Q + off_q, mask= (qo_mask[:, None] < cur_q_seq_len) & (head_mask[:, None] < kv_group_num), other=0.0)
        else:
            q = tl.load(Q + off_q, mask= (qo_mask[:, None] < cur_q_seq_len), other=0.0)
    else:
        if padded_kv_group_TF:
            q = tl.load(Q + off_q, mask= (head_mask[:, None] < kv_group_num), other=0.0)
        else:
            q = tl.load(Q + off_q)
    # q = q.reshape(q, (BLOCK_M * kv_group_num, BLOCK_DMODEL), can_reorder=True)
    q = (q * qk_scale).to(K.dtype.element_ty)
    # initial m_i, l_i, acc
    m_i = tl.zeros([BLOCK_M * padded_kv_group_num], dtype=tl.float32) - float("inf")  # initial m_i-> maximum of each row in q*k ;
    l_i = tl.zeros([BLOCK_M * padded_kv_group_num], dtype=tl.float32)  # l_i -> sum() of each row in (q*k - m_i) ;
    acc = tl.zeros([BLOCK_M * padded_kv_group_num, BLOCK_DMODEL], dtype=tl.float32)  # the final result

    history_len = cur_kv_seq_len - cur_q_seq_len
    low = 0

    # num_blocks =tl.cdiv(high, BLOCK_N)

    k_padded = True
    if cur_kv_seq_len < BLOCK_N:
        seqlen_k_faligned = 0 # floor aligned
        seqlen_k_faligned = seqlen_k_faligned.to(tl.int64)

    elif cur_kv_seq_len % BLOCK_N:
        extra_tokens_n = cur_kv_seq_len % BLOCK_N
        seqlen_k_faligned = cur_kv_seq_len - extra_tokens_n
        seqlen_k_faligned = seqlen_k_faligned.to(tl.int64)
    else:
        k_padded = False
        seqlen_k_faligned = cur_kv_seq_len
        seqlen_k_faligned = seqlen_k_faligned.to(tl.int64)
    # Step 1: all inner loop without any masks
    # attention inner loop
    if STAGE == 1:
        high = seqlen_k_faligned
    else:
        high = min(block_start,seqlen_k_faligned)

    high_unroll_block_idx =   high // BLOCK_N
    # high_unroll_block_idx =   high // BLOCK_N

    new_BLOCK_M: tl.constexpr = BLOCK_M *  padded_kv_group_num
    # tl.device_print("tesd", new_BLOCK_M)

    acc, l_i, m_i = _paged_attn_loop_inner_main(
        acc,
        l_i,
        m_i,
        q,
        K,
        V,
        cur_paged_block_list,
        cur_group_batch,
        stride_kvp,
        stride_bt_0,
        low,
        high_unroll_block_idx,
        off_kv,

    )

    # tl.debug_barrier()
    if k_padded or STAGE == 2:
        tl.debug_barrier()
        if STAGE == 1:
            real_high = cur_kv_seq_len
        else:
            real_high = tl.minimum((start_m + 1) * BLOCK_M, cur_q_seq_len) + history_len

        real_high_unroll_block_idx = tl.cdiv(real_high, BLOCK_N)
        acc, l_i, m_i = _paged_attn_loop_inner_tail(
            acc,
            l_i,
            m_i,
            q,
            K,
            V,
            cur_paged_block_list,
            cur_group_batch,
            stride_kvp,
            stride_bt_0,
            high_unroll_block_idx,
            real_high_unroll_block_idx,
            real_high,
            off_kv,
            offs_m,
            offs_n,  #
            history_len,
            STAGE=STAGE,
            padded_kv_group_num=padded_kv_group_num,
            BLOCK_M=new_BLOCK_M,
            BLOCK_N=BLOCK_N,

        )

    acc = acc / l_i[:, None]  # final_result correction

    # saving
    obs_oh = stride_obs // stride_oh
    o_ret = (obs_oh * offs_m)[:, None] + offs_head[None, :] # obs_oh should be equal to lse?
    flatten_o_ret = tl.ravel(o_ret)

    off_o = cur_q_start_index * stride_obs + flatten_o_ret[:, None] * stride_oh + offs_d[None, :]
    off_lse = cur_q_start_index * stride_lse_bs + flatten_o_ret

    # if q_padded:
    #     if LOGSUMEXP:
    #         m_i += tl.math.log2(l_i)  # for logexpsum
    #         tl.store(LSE + off_lse, m_i, mask= qo_mask < cur_q_seq_len )
    #     tl.store(Out_f + off_o, acc, mask= qo_mask[:, None] < cur_q_seq_len )
    # else:
    #     if LOGSUMEXP:
    #         m_i += tl.math.log2(l_i)  # for logexpsum
    #         tl.store(LSE + off_lse, m_i)
    #     tl.store(Out_f + off_o, acc)
    if q_padded:
        if LOGSUMEXP:
            m_i += tl.math.log2(l_i)  # for logexpsum
            if padded_kv_group_TF:
                tl.store(LSE + off_lse, m_i, mask= (qo_mask < cur_q_seq_len) & (head_mask < kv_group_num) )
            else:
                tl.store(LSE + off_lse, m_i, mask= qo_mask < cur_q_seq_len )
        if padded_kv_group_TF:
            tl.store(Out_f + off_o, acc, mask= (qo_mask[:, None] < cur_q_seq_len) & (head_mask[:, None] < kv_group_num)  )
        else:

            tl.store(Out_f + off_o, acc, mask= qo_mask[:, None] < cur_q_seq_len )
    else:
        if LOGSUMEXP:
            m_i += tl.math.log2(l_i)  # for logexpsum
            if padded_kv_group_TF:
                tl.store(LSE + off_lse, m_i, mask= (head_mask < kv_group_num) )
            else:
                tl.store(LSE + off_lse, m_i)
        if padded_kv_group_TF:
            tl.store(Out_f + off_o, acc, mask= head_mask[:, None] < kv_group_num)
        else:
            tl.store(Out_f + off_o, acc)
    return

@triton.jit
def _paged_pacs_stage_1_kernel_final_main_tail_gqa_fused_decoding_meta(
    Q,
    K_paged_block_pool,
    V_paged_block_pool,

    # instead of Group/Request metainfo in one tensor, we use different tensors to save each one.
    meta,
    paged_block_list,

    # Temp Out tensor
    Out_distinct,
    logsumexp_distinct,
    Out_common,
    logsumexp_common,

    # All necessary stride info for tensor loading
    stride_qbs,
    stride_qh,
    stride_kvp,
    stride_kvbs,
    stride_kvh,

    stride_obs,
    stride_oh,


    stride_lse_bs,

    stride_bt_bs,

    # Scalar
    max_group_tiling,
    batch,
    group,
    num_prefill_tokens,

    sm_scale: tl.constexpr,
    kv_group_num: tl.constexpr,
    P_PADDED_KV_GROUP_NUM: tl.constexpr,
    P_PADDED_KV_GROUP_TF: tl.constexpr,
    D_PADDED_KV_GROUP_NUM:  tl.constexpr,
    paged_block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):


    cur_kv_head = tl.program_id(2)
    p_d = tl.program_id(0) # 0 or 1, 0 means decoding, 1 means prefill

    # cur_kv_head = cur_head // kv_group_num

    # request-level


    if p_d == 0: # request
        cur_group_batch = tl.program_id(1) #
        if cur_group_batch >= batch:
            return
        Q_seq_len = meta
        KV_seq_len = meta + batch + group
        Q_start_index = meta + (batch + group) * 2
        cur_q_seq_len = tl.load(Q_seq_len + cur_group_batch)
        if cur_q_seq_len == 0:
            return
        cur_q_start_index = tl.load(Q_start_index + cur_group_batch) - num_prefill_tokens
        cur_kv_seq_len = tl.load(KV_seq_len + cur_group_batch)
        LOGSUMEXP = True
        if group == 0:
            LOGSUMEXP = False

        invoke_paged_pacs_flash_attn_main_tail_gqa_decoding_1(
            Q,
            K_paged_block_pool,
            V_paged_block_pool,
            Out_distinct,
            logsumexp_distinct,
            paged_block_list,
            stride_qbs,
            stride_qh,
            stride_kvp,
            stride_kvbs,
            stride_kvh,
            stride_obs,
            stride_oh,
            stride_lse_bs,
            stride_bt_bs,
            cur_group_batch,
            cur_kv_head=cur_kv_head,
            cur_kv_seq_len=cur_kv_seq_len,
            cur_q_start_index=cur_q_start_index,
            LOGSUMEXP=LOGSUMEXP,
            qk_scale=sm_scale,
            kv_group_num =kv_group_num,
            PADDED_KV_GROUP_NUM = D_PADDED_KV_GROUP_NUM,
            BLOCK_N=paged_block_size,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )
    else:

        total_index = tl.program_id(1)
        if group == 0 or total_index >= (max_group_tiling*group):
            return

        cur_group_batch = batch + total_index // max_group_tiling
        if cur_group_batch >= group + batch:
            return

        Q_seq_len = meta
        KV_seq_len = meta + batch + group
        Q_start_index = meta + (batch + group) * 2
        start_m = total_index % max_group_tiling
        cur_q_seq_len = tl.load(Q_seq_len + cur_group_batch)
        if cur_q_seq_len == 0:
            return
        cur_q_start_index = tl.load(Q_start_index  +  cur_group_batch) - num_prefill_tokens

        cur_kv_seq_len = tl.load(KV_seq_len + cur_group_batch)

        invoke_paged_pacs_flash_attn_main_tail_gqa(
            Q,
            K_paged_block_pool,
            V_paged_block_pool,
            Out_common,
            logsumexp_common,
            paged_block_list,
            stride_qbs,
            stride_qh,
            stride_kvp,
            stride_kvbs,
            stride_kvh,
            stride_obs,
            stride_oh,
            stride_lse_bs,
            stride_bt_bs,
            start_m=start_m,
            cur_group_batch = cur_group_batch,
            cur_kv_head=cur_kv_head,
            cur_q_seq_len=cur_q_seq_len,
            cur_kv_seq_len=cur_kv_seq_len,
            cur_q_start_index=cur_q_start_index,
            LOGSUMEXP=True,
            qk_scale=sm_scale,
            kv_group_num =kv_group_num,
            padded_kv_group_num=P_PADDED_KV_GROUP_NUM,
            padded_kv_group_TF=P_PADDED_KV_GROUP_TF,
            STAGE=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=paged_block_size,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )
    return


@triton.jit
def _paged_pacs_stage_1_kernel_final_main_tail_gqa_fused_2_meta(
    Q,
    K_paged_block_pool,
    V_paged_block_pool,

    # Group/Request level variables

    # instead of Group/Request metainfo in one tensor, we use different tensors to save each one.
    # Q_seq_len,
    # KV_seq_len,
    # Q_start_index,
    meta,
    paged_block_list,

    # Temp Out tensor
    Out_distinct, # 2, q0,q1,q2
    logsumexp_distinct, # 2, q0,q1
    Out_common, # 2, q0,q1,q2
    logsumexp_common, # 2, q0,q1


    # All necessary stride info for tensor loading
    stride_qbs,
    stride_qh,
    stride_kvp,
    stride_kvbs,
    stride_kvh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    # stride_group_request_list_tensor,
    stride_bt_bs,

    # Scalar

    # max_request_tiling,
    max_group_tiling,
    batch,
    group,

    sm_scale: tl.constexpr,
    kv_group_num: tl.constexpr,
    P_PADDED_KV_GROUP_NUM: tl.constexpr,
    P_PADDED_KV_GROUP_TF: tl.constexpr,
    paged_block_size: tl.constexpr,


    BLOCK_M: tl.constexpr,
    # BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):


    # max(group_num, request_num), kv_head, max_request_tiling_num + group * max_group_tiling


    cur_kv_head = tl.program_id(1)
    total_index = tl.program_id(2)
    block_z_count = tl.num_programs(2)
    max_request_tiling = block_z_count - group * max_group_tiling
    # cur_group_batch = tl.program_id(0)

    # start_m = tl.program_id(2) # (max_num_q_tokens) // BLOCK_M

    if total_index < max_request_tiling: # request
        cur_group_batch = tl.program_id(0)
        if cur_group_batch >= batch:
            return

        LOGSUMEXP=True
        if group ==0:
            LOGSUMEXP = False

        Q_seq_len = meta
        KV_seq_len = meta + batch + group
        Q_start_index = meta + (batch + group) * 2
        cur_q_seq_len = tl.load(Q_seq_len + cur_group_batch)
        cur_q_start_index = tl.load(Q_start_index  +  cur_group_batch)
        # cur_paged_block_list = paged_block_list +  cur_group_batch * stride_group_request_list_tensor
        cur_kv_seq_len = tl.load(KV_seq_len + cur_group_batch)
        invoke_paged_pacs_flash_attn_main_tail_gqa(
            Q,
            K_paged_block_pool,
            V_paged_block_pool,
            Out_distinct,
            logsumexp_distinct,
            # cur_paged_block_list,
            paged_block_list,
            stride_qbs,
            stride_qh,
            stride_kvp,
            stride_kvbs,
            stride_kvh,
            stride_obs,
            stride_oh,
            stride_lse_bs,
            stride_bt_bs,
            start_m=total_index,
            cur_group_batch = cur_group_batch,
            cur_kv_head=cur_kv_head,
            cur_q_seq_len=cur_q_seq_len,
            cur_kv_seq_len=cur_kv_seq_len,
            cur_q_start_index=cur_q_start_index,
            LOGSUMEXP=LOGSUMEXP,
            qk_scale=sm_scale,
            kv_group_num =kv_group_num,
            padded_kv_group_num=P_PADDED_KV_GROUP_NUM,
            padded_kv_group_TF=P_PADDED_KV_GROUP_TF,

            STAGE=2,
            BLOCK_M=BLOCK_M,
            BLOCK_N=paged_block_size,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )
    else:
        this_group = tl.program_id(0)
        if group == 0 or this_group > 0:
            return
        total_index_real = total_index - max_request_tiling
        if total_index_real >= group * max_group_tiling:
            return

        cur_group_batch = batch + total_index_real // max_group_tiling
        if cur_group_batch >= batch + group:
            return
        Q_seq_len = meta
        KV_seq_len = meta + batch + group
        Q_start_index = meta + (batch + group) * 2

        start_m = total_index_real % max_group_tiling
        cur_q_seq_len = tl.load(Q_seq_len + cur_group_batch)
        cur_q_start_index = tl.load(Q_start_index  +  cur_group_batch)
        # cur_paged_block_list = paged_block_list +  cur_group * stride_group_request_list_tensor
        cur_kv_seq_len = tl.load(KV_seq_len + cur_group_batch)
        invoke_paged_pacs_flash_attn_main_tail_gqa(
            Q,
            K_paged_block_pool,
            V_paged_block_pool,
            Out_common,
            logsumexp_common,
            paged_block_list,
            stride_qbs,
            stride_qh,
            stride_kvp,
            stride_kvbs,
            stride_kvh,
            stride_obs,
            stride_oh,
            stride_lse_bs,
            stride_bt_bs,
            start_m=start_m,
            cur_group_batch=cur_group_batch,
            cur_kv_head=cur_kv_head,
            cur_q_seq_len=cur_q_seq_len,
            cur_kv_seq_len=cur_kv_seq_len,
            cur_q_start_index=cur_q_start_index,
            LOGSUMEXP=True,
            qk_scale=sm_scale,
            kv_group_num =kv_group_num,
            padded_kv_group_num=P_PADDED_KV_GROUP_NUM,
            padded_kv_group_TF=P_PADDED_KV_GROUP_TF,
            STAGE=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=paged_block_size,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )
    return


@triton.jit
def _paged_pacs_stage_2_kernel_reduction2_meta(   meta,
                                            O_Common, O_Distinct, O_Final,
                                          logsumexp_common, logsumexp_distinct, stride_obs, stride_oh, stride_lse_bs,
                                          group,
                                          num_prefill_tokens,
                                          BLOCK_M: tl.constexpr,
                                          BLOCK_DMODEL: tl.constexpr):
    """
    Reduction kernel, split-k style.

    A = O_common * exp(logsumexp_common) / (exp(logsumexp_common) + exp(logsumexp_distinct))
    B = O_distinct * exp(logsumexp_distinct) / (exp(logsumexp_common) + exp(logsumexp_distinct))
    Final result = A + B
    """
    start_m = tl.program_id(0)
    cur_group = tl.program_id(1)
    cur_head = tl.program_id(2)


    stride_meta =  group
    # total_q_seq_len = tl.load(meta + batch + cur_group)
    total_q_seq_len = tl.load(meta + cur_group)

    if start_m * BLOCK_M >= total_q_seq_len:
        return
    cur_q_start_loc = tl.load(meta + stride_meta + cur_group) - num_prefill_tokens


    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    offs_o = (cur_q_start_loc + off_m)[:, None] * stride_obs + cur_head * stride_oh + off_d[None, :]

    o_common_tile = tl.load(O_Common + offs_o, mask=off_m[:, None] < total_q_seq_len, other=0.0)
    o_distinct_tile = tl.load(O_Distinct + offs_o, mask=off_m[:, None] < total_q_seq_len, other=0.0)

    offs_lse = (cur_q_start_loc + off_m) * stride_lse_bs + cur_head


    lse_common_tile = tl.load(logsumexp_common + offs_lse, mask=off_m < total_q_seq_len, other=-1e-6)
    lse_distinct_tile = tl.load(logsumexp_distinct + offs_lse, mask=off_m < total_q_seq_len, other=-1e-6)
    max_tile = tl.maximum(lse_common_tile, lse_distinct_tile)

    # correction weight
    common_sumexp = tl.math.exp2((lse_common_tile - max_tile).to(tl.float32))
    distinct_sumexp = tl.math.exp2((lse_distinct_tile - max_tile).to(tl.float32))
    total_sumexp = common_sumexp + distinct_sumexp

    fixed_common_tile = o_common_tile * common_sumexp[:, None] / total_sumexp[:, None]
    fixed_distinct_tile = o_distinct_tile * distinct_sumexp[:, None] / total_sumexp[:, None]
    final_o = fixed_common_tile + fixed_distinct_tile

    offs_o_final = (cur_q_start_loc + off_m)[:, None] * stride_obs + cur_head * stride_oh + off_d[None, :]
    tl.store(O_Final + offs_o_final, final_o, mask=off_m[:, None] < total_q_seq_len)

    return


prefill_cached_bin = None
# TODO(xinji1): if 2 cached kernels better
decoding_cached_bin = {1: None, 2: None}
from triton.compiler.compiler import CompiledKernel
reduction_cached_bin = None

@torch.inference_mode()
def cs_single_prefill_decoding(q, k_paged_block_pool, v_paged_block_pool,
                                                    o_common, o_distinct, logsumexp_common,
                                                    logsumexp_distinct, o_final,
                                                    context_share_request_kv_list_tensor,
                                                    context_share_request_meta_tensor,
                                                    max_group_q_len,
                                                    max_request_q_len,
                                                    real_group_num,
                                                    real_request_num,
                                                    num_prefill_tokens=0,
                                                    decoding = False,
                                                    red_tensor=None
                                                    ):
    """
    A split version here, using 2 separated kernels to handle (q, k_common, v_common) and (q, k_distinct, v_distinct)
    """
    # TODO(xinji1): not test gqa yet.

    Lk = q.size(-1)
    sm_scale = (Lk ** (-0.5)) * 1.44269504
    head = q.size(1)
    query_group_size = head // k_paged_block_pool.size(2)
    kv_group_num = query_group_size
    paged_block_size = k_paged_block_pool.size(1)

    # Get the target device from a known GPU tensor (e.g., q)
    device = q.device

    # Ensure meta tensors are on the correct device
    context_share_request_meta_tensor = context_share_request_meta_tensor.to(device, non_blocking=True)
    context_share_request_kv_list_tensor = context_share_request_kv_list_tensor.to(device, non_blocking=True)
    if red_tensor is not None:  # Also ensure red_tensor is on the correct device for the reduction kernel
        red_tensor = red_tensor.to(device, non_blocking=True)

    batch = real_request_num

    # batch = context_share_request_meta_tensor.shape[1] - real_group_num
    if kv_group_num == 1:
        padded_group_size = 1
    elif kv_group_num <= 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(kv_group_num)


    p_padded_kv_group_num = triton.next_power_of_2(kv_group_num)
    p_padded_kv_group_TF = (p_padded_kv_group_num != kv_group_num)
    META = Triton_meta(p_padded_kv_group_num, paged_block_size)


    # META["share_distinct"]["BLOCK_M"] = 32
    max_distinct_tiling_num = int(
        triton.cdiv(max_request_q_len,
                META["share_distinct"]["BLOCK_M"]))
    max_common_tiling_num = 0
    group_num = 0
    if real_group_num > 0:
        group_num = real_group_num
        max_common_tiling_num = int(
            triton.cdiv(max_group_q_len,
                    META["share_common"]["BLOCK_M"]))
        reduction_tiling_num = int(
            triton.cdiv(max_group_q_len,
                        META["share_reduction"]["BLOCK"]))

    args = [
        q,
        k_paged_block_pool,
        v_paged_block_pool,
        context_share_request_meta_tensor,  # Now on the correct device
        context_share_request_kv_list_tensor,  # Now on the correct device
        o_distinct,
        logsumexp_distinct,
        o_common,
        logsumexp_common,
        q.stride(0),
        q.stride(1),
        k_paged_block_pool.stride(0),
        k_paged_block_pool.stride(1),
        k_paged_block_pool.stride(2),
        o_distinct.stride(0),
        o_distinct.stride(1),
        logsumexp_distinct.stride(0),
        context_share_request_kv_list_tensor.stride(0),  # Stride of the (now device-correct) tensor
        max_common_tiling_num,
        batch,
        group_num,
    ]
    # print(args[-4:])
    const_args = {
            "sm_scale":sm_scale,
            "kv_group_num":kv_group_num,

            "paged_block_size":paged_block_size,
            "P_PADDED_KV_GROUP_NUM":  p_padded_kv_group_num,
            "P_PADDED_KV_GROUP_TF":  p_padded_kv_group_TF,
            "BLOCK_M":META["share_common"]["BLOCK_M"],
            "BLOCK_DMODEL":Lk,
    }


    if max_request_q_len == 1 and decoding:# decoding:
        args.append(num_prefill_tokens)
        const_args["D_PADDED_KV_GROUP_NUM"] = padded_group_size
        tmp = 2 if group_num > 0 else 1
        grid_1 = max(batch, int(group_num * max_common_tiling_num))


        grid = (tmp, grid_1 , k_paged_block_pool.size(2))

        global decoding_cached_bin
        # # NOTE(xinji1): Adapted from `/triton/compiler/compiler.py`
        if decoding_cached_bin[tmp] is not None:
            bin = decoding_cached_bin[tmp]
            stream = torch.cuda.current_stream().cuda_stream

            # bin.run(grid[0], grid[1], grid[2], bin.num_warps, bin.num_ctas, bin.cluster_dims[0],
            #     bin.cluster_dims[1], bin.cluster_dims[2], bin.shared, stream, bin.function,
            #     CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, bin, *args,)
            bin.run(grid[0], grid[1], grid[2], stream, bin.function, bin.packed_metadata, bin.launch_metadata(grid, stream, *args),
                CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, *args,)
        else:
            decoding_cached_bin[tmp] = _paged_pacs_stage_1_kernel_final_main_tail_gqa_fused_decoding_meta[grid](*args, **const_args ,  num_warps = META["share_distinct"]["num_warps"], num_stages = META["share_distinct"]["num_stages"],)

    else:


        grid_0 = batch
        grid_2 = max_distinct_tiling_num  + group_num * max_common_tiling_num

        # grid_0 = 1
        # grid_2 = batch * max_distinct_tiling_num  + group_num * max_common_tiling_num
        grid = (grid_0,  k_paged_block_pool.size(2), grid_2)


        global prefill_cached_bin
        # /scratch/singularity_webdata_ws01_eastus2_nfs/chec/anaconda3/envs/mii1/lib/python3.10/site-packages/triton/compiler/compiler.py
        if prefill_cached_bin is not None:
            bin = prefill_cached_bin
            stream = torch.cuda.current_stream().cuda_stream


            bin.run(grid[0], grid[1], grid[2], stream, bin.function, bin.packed_metadata, bin.launch_metadata(grid, stream, *args),
                CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, *args, )
        else:
            prefill_cached_bin = _paged_pacs_stage_1_kernel_final_main_tail_gqa_fused_2_meta[grid](*args, **const_args ,  num_warps = META["share_distinct"]["num_warps"], num_stages = META["share_distinct"]["num_stages"],)

    if real_group_num > 0:
        reduction_tiling_num = int(
            triton.cdiv(max_group_q_len,
                        META["share_reduction"]["BLOCK"]))

        grid_red = (reduction_tiling_num, group_num, head)

        # red_tensor is the first argument to _paged_pacs_stage_2_kernel_reduction2_meta
        # Ensure it's on the correct device (handled above if red_tensor was not None)
        red_args = [
            red_tensor,  # This was moved to device above
            o_common,
            o_distinct,
            o_final,
            logsumexp_common,
            logsumexp_distinct,
            o_final.stride(0),
            o_final.stride(1),
            logsumexp_distinct.stride(0),
            group_num,
            num_prefill_tokens,
        ]

        red_const_args = {
                "BLOCK_M" : META["share_reduction"]["BLOCK"],
                "BLOCK_DMODEL" : Lk,
        }

        global reduction_cached_bin
        # # /scratch/singularity_webdata_ws01_eastus2_nfs/chec/anaconda3/envs/mii1/lib/python3.10/site-packages/triton/compiler/compiler.py
        if reduction_cached_bin is not None:
            bin = reduction_cached_bin
            stream = torch.cuda.current_stream().cuda_stream

            bin.run(grid_red[0], grid_red[1], grid_red[2], stream, bin.function, bin.packed_metadata, bin.launch_metadata(grid_red, stream, *red_args),
                CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, *red_args,)
        else:
            reduction_cached_bin =   _paged_pacs_stage_2_kernel_reduction2_meta[grid_red](*red_args, **red_const_args ,
                                                                        num_warps=META["share_reduction"]["num_warps"],
                                                                        num_stages=META["share_reduction"]["num_stages"],)


def Triton_meta(padded_kv_group_num, paged_block_size):
    # FIXME: need to support more models. Performance tuning.
    if padded_kv_group_num > 8 or padded_kv_group_num == 1:
        raise NotImplementedError(
            "Currently batchllm only supports the model with grouped-query attention",
            "and the `kv_group_num`(`num_heads / num_kv_heads)` <=8 ",
            "like llama-3b/qwen-7b/qwen-0.5b/mistral-7b. ")

    META = {

            "share_common": {
                "BLOCK_M": 32 // padded_kv_group_num ,
                "BLOCK_N": paged_block_size,
                "num_stages": 2,
                "num_warps": 4
            },
            "share_distinct": {
                "BLOCK_M": 32 // padded_kv_group_num,
                "BLOCK_N": paged_block_size,
                "num_stages": 2,
                "num_warps": 4
            },
            "share_reduction": {
                "BLOCK": 128,
                "waves_per_eu": 1,
                "num_stages": 1,
                "num_warps": 4
            }
        }
    return META


@torch.inference_mode()
def single_prefill_decoding(
    q, k_paged_block_pool, v_paged_block_pool,
    context_share_request_kv_list_tensor,
    context_share_request_meta_tensor,
    max_group_q_len,
    max_request_q_len,
    real_group_num,
    real_request_num = 0,
    num_prefill_tokens = 0,
    o_distinct_pre = None,
    return_ocod = False,
    decoding= False,
    red_tensor=None
):

    o_common = torch.empty_like(q)
    # o_final = torch.empty_like(q)
    # o_final = torch.empty_like(q)
    if o_distinct_pre is not None:
        o_distinct = o_distinct_pre
    else:
        o_distinct = torch.empty_like(q)
    #TODO(xinji1): float32/ float16/ fp32/ fp16 issue?
    logsumexp_common = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=q.device)
    logsumexp_distinct = torch.empty_like(logsumexp_common)


    cs_single_prefill_decoding(q, k_paged_block_pool, v_paged_block_pool,
                                                o_common, o_distinct, logsumexp_common,
                                                logsumexp_distinct, o_distinct,
                                                context_share_request_kv_list_tensor,
                                                context_share_request_meta_tensor,
                                                max_group_q_len,
                                                max_request_q_len,
                                                real_group_num,
                                                real_request_num,
                                                num_prefill_tokens=num_prefill_tokens,
                                                decoding = decoding,
                                                red_tensor=red_tensor
                                                )

    if return_ocod :
        return o_common, o_distinct, o_distinct
    else:
        return o_distinct
