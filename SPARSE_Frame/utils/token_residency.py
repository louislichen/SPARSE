# utils/token_residency.py
from __future__ import annotations
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

Tensor = torch.Tensor
PastKV = Tuple[Tuple[Tensor, Tensor], ...]  # len = num_layers, each: (K,V) [B, n_kv, L, hd]

@dataclass
class ResidencyCfg:
    window_layers: int = 4          # 每 i 层迁移一次
    enabled: bool = False
    # KV 压缩（SKVS）参数（跟你 skvs_attn.py 一致）
    sink: int = 4
    window: int = 32
    ratio: float = 0.6              # topk ratio on unprotected
    # 是否仅对 decode(q_len==1) 启用迁移（推荐先这样）
    decode_only: bool = True

@dataclass
class KVSlicePlan:
    # keep_idx: [B, kept_len] long, global positions in [0, L_kv-1]
    keep_idx: Tensor
    kept_len: int
    full_len: int

def compute_keep_idx_from_l0_score(
    l0_score: Optional[Tensor],     # [B, L_kv] or None
    full_kv_len: int,
    sink_len: int,
    window_len: int,
    ratio: float,
    device: torch.device,
) -> KVSlicePlan:
    sink_len = min(sink_len, full_kv_len)
    window_len = min(window_len, full_kv_len)
    if l0_score is None:
        # 没有 guidance：退化为 sink+window
        keep = torch.cat([
            torch.arange(sink_len, device=device),
            torch.arange(full_kv_len - window_len, full_kv_len, device=device),
        ], dim=0).unique(sorted=True).view(1, -1)
        return KVSlicePlan(keep_idx=keep, kept_len=keep.shape[1], full_len=full_kv_len)

    B = l0_score.shape[0]
    if l0_score.shape[1] < full_kv_len:
        l0_score = torch.nn.functional.pad(l0_score, (0, full_kv_len - l0_score.shape[1]), value=1.0)

    # protected positions: sink + window
    idx = torch.arange(full_kv_len, device=device).view(1, -1).expand(B, -1)
    is_sink = idx < sink_len
    is_window = idx >= (full_kv_len - window_len)
    is_prot = is_sink | is_window

    num_unprot = full_kv_len - sink_len - window_len
    topk = int(num_unprot * float(ratio))
    if topk <= 0:
        keep = torch.cat([
            torch.arange(sink_len, device=device),
            torch.arange(full_kv_len - window_len, full_kv_len, device=device),
        ], dim=0).unique(sorted=True).view(1, -1).expand(B, -1)
        return KVSlicePlan(keep_idx=keep, kept_len=keep.shape[1], full_len=full_kv_len)

    scores = l0_score.clone()
    scores[is_prot] = -float("inf")
    _, topk_idx = torch.topk(scores, k=topk, dim=-1)

    sink_idx = torch.arange(sink_len, device=device).view(1, -1).expand(B, -1)
    win_idx = torch.arange(full_kv_len - window_len, full_kv_len, device=device).view(1, -1).expand(B, -1)

    keep = torch.cat([sink_idx, topk_idx, win_idx], dim=-1)
    keep, _ = torch.sort(keep, dim=-1)
    return KVSlicePlan(keep_idx=keep, kept_len=keep.shape[1], full_len=full_kv_len)

def slice_past_kv_tuple(past: PastKV, plan: KVSlicePlan) -> PastKV:
    # past[layer] = (K,V): [B, n_kv, L, hd]
    keep = plan.keep_idx
    B, kept_len = keep.shape
    out: List[Tuple[Tensor, Tensor]] = []
    for (k, v) in past:
        # gather along seq_len dim=2
        gather_idx = keep.view(B, 1, kept_len, 1).expand(B, k.shape[1], kept_len, k.shape[3])
        k_s = torch.gather(k, dim=2, index=gather_idx).contiguous()
        v_s = torch.gather(v, dim=2, index=gather_idx).contiguous()
        out.append((k_s, v_s))
    return tuple(out)

def pack_kv_for_send(past_sliced: PastKV) -> Tuple[Tensor, Tensor]:
    # pack all layers into a single contiguous tensor for K and V
    # K_pack: [L, B, n_kv, kept_len, hd]
    ks = torch.stack([kv[0] for kv in past_sliced], dim=0)
    vs = torch.stack([kv[1] for kv in past_sliced], dim=0)
    return ks.contiguous(), vs.contiguous()

def unpack_kv_from_recv(k_pack: Tensor, v_pack: Tensor) -> PastKV:
    # inverse of pack
    L = k_pack.shape[0]
    out: List[Tuple[Tensor, Tensor]] = []
    for l in range(L):
        out.append((k_pack[l].contiguous(), v_pack[l].contiguous()))
    return tuple(out)

def _all_to_all_varlen_1d(send: Tensor, send_counts: Tensor, device: torch.device) -> Tensor:
    """
    Helper for all_to_all_single on 1D flattened buffers.
    send: [N] 1D contiguous
    send_counts: [world] long on device
    return: recv 1D
    """
    world = dist.get_world_size()
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    sc = send_counts.cpu().tolist()
    rc = recv_counts.cpu().tolist()
    recv = torch.empty((sum(rc),), device=device, dtype=send.dtype)
    dist.all_to_all_single(recv, send, rc, sc)
    return recv

def migrate_tokens_and_kv(
    env_rank: int,
    env_world: int,
    device: torch.device,
    # token-level: we assume decode q_len==1, tokens are positions in batch
    hidden_states: Tensor,              # [B, q_len, H] on current rank
    past: Optional[PastKV],             # tuple cache on current rank, or None (prefill first step)
    target_rank: Tensor,                # [B] long on device: each batch token's destination rank
    plan: Optional[KVSlicePlan],        # if past is not None, compress and send
) -> Tuple[Tensor, Optional[PastKV], Tensor]:
    """
    Returns:
      new_hidden_states: [B_local, q_len, H] tokens now residing on this rank
      new_past: tuple cache for local tokens (compressed), or None
      local_batch_ids: [B_local] original batch indices of tokens now on this rank
    """
    B, q_len, H = hidden_states.shape
    assert q_len == 1, "This migration implementation currently supports decode (q_len==1) only."

    # Build per-dest lists of batch indices
    dest = target_rank.to(torch.int64)
    order = torch.argsort(dest)  # group by dest
    dest_sorted = dest[order]
    # counts per dest: how many batch items to send to each rank
    send_counts = torch.bincount(dest_sorted, minlength=env_world).to(device=device, dtype=torch.int64)

    # Pack hidden send buffer: [B, H] -> [B, H]
    send_hidden = hidden_states[:, 0, :][order].contiguous()  # [B, H]
    # Also send original batch ids to reconstruct ordering on receiver
    send_bids = order.to(device=device, dtype=torch.int64).contiguous()  # [B]

    # all_to_all for hidden and batch ids (fixed per token, but variable per rank)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    sc = send_counts.cpu().tolist()
    rc = recv_counts.cpu().tolist()
    recv_total = int(sum(rc))

    recv_hidden = torch.empty((recv_total, H), device=device, dtype=send_hidden.dtype)
    recv_bids = torch.empty((recv_total,), device=device, dtype=torch.int64)

    dist.all_to_all_single(recv_hidden, send_hidden, rc, sc)
    dist.all_to_all_single(recv_bids, send_bids, rc, sc)

    # KV migration (optional)
    recv_past: Optional[PastKV] = None
    if past is not None and plan is not None:
        past_s = slice_past_kv_tuple(past, plan)
        k_pack, v_pack = pack_kv_for_send(past_s)   # [L, B, n_kv, kept, hd]
        # reorder along batch dim the same as hidden
        k_pack = k_pack[:, order, :, :, :].contiguous()
        v_pack = v_pack[:, order, :, :, :].contiguous()

        # flatten per rank send buffer (variable len per dest): pack to 1D
        # We keep shapes same across tokens, so bytes per token are constant -> we can still do varlen by counts
        L, _, n_kv, kept, hd = k_pack.shape
        per_tok = L * n_kv * kept * hd
        send_k_flat = k_pack.reshape(-1).contiguous()
        send_v_flat = v_pack.reshape(-1).contiguous()
        send_counts_k = send_counts * per_tok
        # recv
        recv_k_flat = _all_to_all_varlen_1d(send_k_flat, send_counts_k, device)
        recv_v_flat = _all_to_all_varlen_1d(send_v_flat, send_counts_k, device)

        # reshape back: total_tokens_local = recv_total
        if recv_total > 0:
            recv_k_pack = recv_k_flat.view(L, recv_total, n_kv, kept, hd).contiguous()
            recv_v_pack = recv_v_flat.view(L, recv_total, n_kv, kept, hd).contiguous()
            recv_past = unpack_kv_from_recv(recv_k_pack, recv_v_pack)
        else:
            recv_past = tuple()

    # restore hidden to [B_local, 1, H]
    new_hidden = recv_hidden.view(recv_total, 1, H).contiguous()
    local_batch_ids = recv_bids.contiguous()
    return new_hidden, recv_past, local_batch_ids