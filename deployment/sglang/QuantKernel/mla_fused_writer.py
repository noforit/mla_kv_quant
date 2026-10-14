"""Decode writer: fuse quantization, packing, RoPE/window stores and ring metadata."""

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.mla_latent_int2 import (
    _scatter_pack_kernel,
    _LM_THRESHOLDS,
    _LM_RATIO,
    _LM_SPAN,
    _LM_CENTROIDS,
)


@triton.jit
def _aux(
    C,
    PE,
    Loc,
    Seq,
    Req,
    Rope,
    HP,
    Rows,
    Owners,
    ps0: tl.constexpr,
    ps1: tl.constexpr,
    rs0: tl.constexpr,
    P: tl.constexpr,
    W: tl.constexpr,
    MAX_REQ: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Loc + row).to(tl.int64)
    seq = tl.load(Seq + row).to(tl.int64)
    req = tl.load(Req + row).to(tl.int64)
    pos = seq - 1
    keep = ((pos < P) | (pos >= seq - W)) & (req < MAX_REQ) & (req >= 0) & (slot > 0)
    # Slot zero belongs to CUDA Graph padding. It must never claim a live ring row.
    ring = 1 + req * (P + W) + tl.where(pos < P, pos, P + tl.maximum(pos - P, 0) % max(W, 1))
    d = tl.arange(0, D)
    r = tl.arange(0, R)
    pe = tl.load(PE + row * ps0 + r * ps1)
    tl.store(Rope + slot * R + r, pe, mask=slot > 0)
    c = tl.load(C + row * rs0 + d)
    tl.store(HP + ring * D + d, c, mask=keep)
    tl.store(Owners + ring, slot.to(tl.int32), mask=keep)
    tl.store(Rows + slot, tl.where(keep, ring, -1).to(tl.int32), mask=slot > 0)


@triton.jit
def _pack_and_aux(
    C,
    PE,
    Loc,
    Seq,
    Req,
    Codes,
    Params,
    Rope,
    HP,
    Rows,
    Owners,
    n_rows,
    ps0: tl.constexpr,
    ps1: tl.constexpr,
    rs0: tl.constexpr,
    P: tl.constexpr,
    W: tl.constexpr,
    MAX_REQ: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    GS: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    LM_SPAN3: tl.constexpr,
    LM_RATIO: tl.constexpr,
    LM_C0: tl.constexpr,
):
    # Inline the existing codec so its reduction order and threshold arithmetic
    # stay shared with the reference path. The rotation remains an FP32 GEMM.
    _scatter_pack_kernel(
        C,
        Loc,
        Codes,
        Params,
        n_rows,
        D,
        GS,
        D // GS,
        True,
        2,
        4,
        3,
        3.0,
        T0,
        T1,
        T2,
        LM_SPAN3,
        LM_RATIO,
        LM_C0,
    )
    _aux(C, PE, Loc, Seq, Req, Rope, HP, Rows, Owners, ps0, ps1, rs0, P, W, MAX_REQ, D, R)


def try_fast_decode_store(pool, layer_id, loc, c_kv, k_pe):
    """Return whether the decode write was handled; preserve the caller's fallback.

    The FP32 GEMM and quantizer are unchanged. The INT2 Lloyd codec and auxiliary
    stores share one launch; other codecs retain the separate-store path.
    Prefill uses the cache pool writer.
    """
    meta = pool._fb_window_meta
    n = c_kv.numel() // pool.kv_lora_rank
    if (
        n <= 0
        or pool.kv_lora_rank != 512
        or pool.qk_rope_head_dim != 64
        or not pool._latent_windows
        or not meta
        or meta.get("fallback")
        or not meta["is_decode"]
        or meta["seq_lens"] is None
        or meta["req_pool_indices"] is None
        or meta["seq_lens"].numel() != n
        or meta["req_pool_indices"].numel() != n
        or pool._selfcheck_budget > 0
        or pool._audit_budget > 0
    ):
        return False
    from sglang.QuantKernel.mla_latent_int2 import scatter_pack_rows

    li = layer_id - pool.start_layer
    r = pool.latent_rotation(layer_id, local=True)
    cached = getattr(pool, "_fast_write_rotation", None)
    if cached is None:
        cached = pool._fast_write_rotation = {}
    x = c_kv.reshape(-1, pool.kv_lora_rank).float()
    if r is not None:
        if layer_id not in cached:
            cached[layer_id] = (id(r), r.float())
        assert cached[layer_id][0] == id(r)
        c = x @ cached[layer_id][1]
    else:
        c = x
    frames = getattr(pool, "_rot_write_frame", None)
    if frames is None:
        frames = pool._rot_write_frame = {}
    if r is not None:
        frames.setdefault(layer_id, id(r))
    loc64 = loc.reshape(-1).to(torch.int64)
    pool._write_loc = loc64
    pe = k_pe.reshape(-1, pool.qk_rope_head_dim)
    if pool._lloyd_max and pool._bits == 2:
        params = pool._rolaq_quantizers[li]
        thresholds, gain = (
            ((-params[0], 0.0, params[0]), params[1])
            if params is not None
            else (_LM_THRESHOLDS, _LM_RATIO)
        )
        _pack_and_aux[(n,)](
            c,
            pe,
            loc64,
            meta["seq_lens"],
            meta["req_pool_indices"],
            pool.c_codes[li],
            pool.c_params[li],
            pool.rope_buf[li],
            pool.hp_c[li],
            pool.hp_row_of_slot,
            pool.hp_owner_of_row,
            n,
            pe.stride(0),
            pe.stride(1),
            c.stride(0),
            pool._win_p,
            pool._win_r,
            pool._max_reqs,
            pool.kv_lora_rank,
            pool.qk_rope_head_dim,
            pool._group_size,
            *thresholds,
            _LM_SPAN / 3.0,
            gain,
            _LM_CENTROIDS[0],
            num_warps=4,
        )
        return True
    scatter_pack_rows(
        c,
        loc64.to(torch.int32),
        pool.c_codes[li],
        pool.c_params[li],
        pool._group_size,
        pool._lloyd_max,
        pool._bits,
        quantizer_params=pool._rolaq_quantizers[li],
    )
    _aux[(n,)](
        c,
        pe,
        loc64,
        meta["seq_lens"],
        meta["req_pool_indices"],
        pool.rope_buf[li],
        pool.hp_c[li],
        pool.hp_row_of_slot,
        pool.hp_owner_of_row,
        pe.stride(0),
        pe.stride(1),
        c.stride(0),
        pool._win_p,
        pool._win_r,
        pool._max_reqs,
        pool.kv_lora_rank,
        pool.qk_rope_head_dim,
        num_warps=4,
    )
    return True
