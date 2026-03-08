from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import Cache


@dataclass
class SKVSConfig:
    h2o_sink: int = 4
    h2o_window: int = 32
    l0_guidance_ratio: float = 0.6


GLOBAL_ATTN_STATE = {
    "layer0_score": None,
    "forward_dropped_bytes": 0,
    "forward_kept_remote_bytes": 0,
    "step_dropped_bytes": 0,
}

_SKVS_CFG = SKVSConfig()


def set_skvs_config(cfg: SKVSConfig) -> None:
    global _SKVS_CFG
    _SKVS_CFG = cfg


def _reset_forward_accounting() -> None:
    GLOBAL_ATTN_STATE["layer0_score"] = None
    GLOBAL_ATTN_STATE["forward_dropped_bytes"] = 0
    GLOBAL_ATTN_STATE["forward_kept_remote_bytes"] = 0
    GLOBAL_ATTN_STATE["step_dropped_bytes"] = 0


def _accumulate_accounting(dropped_bytes: int, kept_remote_bytes: int = 0) -> None:
    GLOBAL_ATTN_STATE["step_dropped_bytes"] = int(dropped_bytes)
    GLOBAL_ATTN_STATE["forward_dropped_bytes"] = int(GLOBAL_ATTN_STATE.get("forward_dropped_bytes", 0) or 0) + int(
        dropped_bytes
    )
    GLOBAL_ATTN_STATE["forward_kept_remote_bytes"] = int(
        GLOBAL_ATTN_STATE.get("forward_kept_remote_bytes", 0) or 0
    ) + int(kept_remote_bytes)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _kv_bytes_per_token(num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return 2 * num_kv_heads * head_dim * torch.tensor([], dtype=dtype).element_size()


def _gather_kv_by_keep_idx(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    keep_idx: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = key_states.shape[0]
    kept_len = keep_idx.shape[1]
    if kept_len == 0:
        empty_k = torch.empty(
            (bsz, num_kv_heads, 0, head_dim),
            device=key_states.device,
            dtype=key_states.dtype,
        )
        empty_v = torch.empty_like(empty_k)
        return empty_k, empty_v

    gather_idx = keep_idx.view(bsz, 1, kept_len, 1).expand(-1, num_kv_heads, -1, head_dim)
    sliced_k = torch.gather(key_states, dim=2, index=gather_idx)
    sliced_v = torch.gather(value_states, dim=2, index=gather_idx)
    return sliced_k, sliced_v


def _gather_mask_by_keep_idx(
    attention_mask: Optional[torch.Tensor],
    keep_idx: torch.Tensor,
    q_len: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    bsz = attention_mask.shape[0]
    kept_len = keep_idx.shape[1]
    if kept_len == 0:
        return attention_mask[:, :, :, :0]
    mask_idx = keep_idx.view(bsz, 1, 1, kept_len).expand(-1, 1, q_len, -1)
    return torch.gather(attention_mask, dim=-1, index=mask_idx)


def _resolve_kv_owner(
    self,
    full_kv_len: int,
    device: torch.device,
    bsz: int,
) -> Optional[torch.Tensor]:
    owner_map = getattr(self, "_skvs_token_owner", None)
    if owner_map is None:
        getter = getattr(self, "_skvs_get_token_owner", None)
        if getter is not None and callable(getter):
            owner_map = getter(full_kv_len, device, bsz)

    if owner_map is None:
        kv_owner_fn = getattr(self, "_skvs_kv_owner_fn", None)
        env_world = int(getattr(self, "_sparse_env_world", 1))
        if kv_owner_fn is not None and callable(kv_owner_fn) and env_world > 0:
            pos_idx = torch.arange(full_kv_len, device=device, dtype=torch.long)
            owner_1d = kv_owner_fn(pos_idx, env_world)
            owner_map = owner_1d.unsqueeze(0).expand(bsz, -1)

    if owner_map is None:
        return None

    if not torch.is_tensor(owner_map):
        return None

    owner_map = owner_map.to(device=device, dtype=torch.long)

    if owner_map.dim() == 1:
        owner_map = owner_map.unsqueeze(0).expand(bsz, -1)
    elif owner_map.dim() != 2:
        return None

    if owner_map.shape[0] != bsz:
        if owner_map.shape[0] == 1:
            owner_map = owner_map.expand(bsz, -1)
        else:
            return None

    if owner_map.shape[1] < full_kv_len:
        owner_map = F.pad(owner_map, (0, full_kv_len - owner_map.shape[1]), value=0)
    elif owner_map.shape[1] > full_kv_len:
        owner_map = owner_map[:, :full_kv_len]

    return owner_map.contiguous()


def _build_remote_keep_idx_for_owner(
    l0_score: Optional[torch.Tensor],
    kv_owner_map: torch.Tensor,
    owner_rank: int,
    full_kv_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    assert kv_owner_map.shape[0] == 1

    idx = torch.arange(full_kv_len, device=device, dtype=torch.long).unsqueeze(0)

    sink_len = min(_SKVS_CFG.h2o_sink, full_kv_len)
    window_len = min(_SKVS_CFG.h2o_window, full_kv_len)

    is_sink = idx < sink_len
    is_window = idx >= (full_kv_len - window_len)
    is_protected = is_sink | is_window

    is_owner_local = kv_owner_map == int(owner_rank)
    is_remote = ~is_owner_local
    remote_protected = is_remote & is_protected
    remote_unprotected = is_remote & (~is_protected)

    remote_unprotected_idx = idx[remote_unprotected].contiguous()
    num_remote_unprotected = int(remote_unprotected_idx.numel())

    if l0_score is None or num_remote_unprotected <= 0:
        remote_keep_mask = is_remote
        dropped_remote = 0
        remote_keep_idx = idx[remote_keep_mask].view(1, -1)
        return remote_keep_idx, dropped_remote

    if l0_score.shape[-1] < full_kv_len:
        l0_score = F.pad(l0_score, (0, full_kv_len - l0_score.shape[-1]), value=0.0)
    elif l0_score.shape[-1] > full_kv_len:
        l0_score = l0_score[:, :full_kv_len]

    topk = int(num_remote_unprotected * float(_SKVS_CFG.l0_guidance_ratio))
    topk = max(0, min(topk, num_remote_unprotected))

    if topk > 0:
        remote_scores = l0_score[0, remote_unprotected_idx]
        _, topk_sel = torch.topk(remote_scores, k=topk, dim=-1)
        remote_topk_idx = remote_unprotected_idx[topk_sel].contiguous()
    else:
        remote_topk_idx = torch.empty((0,), device=device, dtype=torch.long)

    protected_idx = idx[remote_protected].contiguous()
    remote_keep_idx_1d = torch.cat([protected_idx, remote_topk_idx], dim=0)
    if remote_keep_idx_1d.numel() > 0:
        remote_keep_idx_1d, _ = torch.sort(remote_keep_idx_1d)
    remote_keep_idx = remote_keep_idx_1d.view(1, -1)

    dropped_remote = num_remote_unprotected - topk
    return remote_keep_idx, int(dropped_remote)


def _run_attention_like_qwen(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_attentions: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    kv_seq_len = key_states.shape[-2]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_seq_len]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


def _call_orig_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_value: Optional[Cache],
    output_attentions: bool,
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
):
    orig_forward = getattr(self, "_skvs_orig_forward", None)
    if orig_forward is None:
        raise RuntimeError("Missing self._skvs_orig_forward; save original forward before patching")
    return orig_forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )


def _real_transport_owner_only_attention(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    kv_owner_map: torch.Tensor,
    remote_keep_idx: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_attentions: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    owner_rank = int(getattr(self, "_skvs_owner_rank", -1))
    env_rank = int(getattr(self, "_sparse_env_rank", -1))
    env_world = int(getattr(self, "_sparse_env_world", 1))

    if owner_rank < 0 or env_rank < 0 or env_world <= 1:
        raise RuntimeError("real transport path requires _skvs_owner_rank and distributed env hooks")

    bsz = query_states.shape[0]
    if bsz != 1:
        raise RuntimeError("real transport path currently supports batch_size == 1 only")

    full_kv_len = key_states.shape[-2]
    bytes_per_token = _kv_bytes_per_token(
        num_kv_heads=self.num_key_value_heads,
        head_dim=self.head_dim,
        dtype=key_states.dtype,
    )

    all_pos = torch.arange(full_kv_len, device=key_states.device, dtype=torch.long)
    owner_local_pos = all_pos[kv_owner_map[0] == owner_rank].contiguous()

    remote_keep_idx_1d = remote_keep_idx[0] if remote_keep_idx.numel() > 0 else torch.empty(
        (0,), device=key_states.device, dtype=torch.long
    )

    if env_rank == owner_rank or remote_keep_idx_1d.numel() == 0:
        send_pos = torch.empty((0,), device=key_states.device, dtype=torch.long)
    else:
        send_mask = (kv_owner_map[0, remote_keep_idx_1d] == env_rank)
        send_pos = remote_keep_idx_1d[send_mask].contiguous()

    send_len = int(send_pos.numel())
    if send_len > 0:
        send_pos_b = send_pos.unsqueeze(0)
        send_k, send_v = _gather_kv_by_keep_idx(
            key_states=key_states,
            value_states=value_states,
            keep_idx=send_pos_b,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )
    else:
        send_k = torch.empty(
            (1, self.num_key_value_heads, 0, self.head_dim),
            device=key_states.device,
            dtype=key_states.dtype,
        )
        send_v = torch.empty_like(send_k)

    len_t = torch.tensor([send_len], device=key_states.device, dtype=torch.long)
    all_lens = [torch.zeros_like(len_t) for _ in range(env_world)]
    dist.all_gather(all_lens, len_t)
    lens = [int(x.item()) for x in all_lens]
    max_len = max(lens) if lens else 0
    gather_slots = max(max_len, 1)

    pad_pos = torch.full((gather_slots,), -1, device=key_states.device, dtype=torch.long)
    pad_k = torch.zeros(
        (1, self.num_key_value_heads, gather_slots, self.head_dim),
        device=key_states.device,
        dtype=key_states.dtype,
    )
    pad_v = torch.zeros_like(pad_k)

    if send_len > 0:
        pad_pos[:send_len] = send_pos
        pad_k[:, :, :send_len, :] = send_k
        pad_v[:, :, :send_len, :] = send_v

    gathered_pos = [torch.empty_like(pad_pos) for _ in range(env_world)]
    gathered_k = [torch.empty_like(pad_k) for _ in range(env_world)]
    gathered_v = [torch.empty_like(pad_v) for _ in range(env_world)]

    dist.all_gather(gathered_pos, pad_pos)
    dist.all_gather(gathered_k, pad_k)
    dist.all_gather(gathered_v, pad_v)

    kept_remote_bytes = 0

    if env_rank == owner_rank:
        pos_list = []
        k_list = []
        v_list = []

        if owner_local_pos.numel() > 0:
            owner_local_pos_b = owner_local_pos.unsqueeze(0)
            owner_local_k, owner_local_v = _gather_kv_by_keep_idx(
                key_states=key_states,
                value_states=value_states,
                keep_idx=owner_local_pos_b,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
            )
            pos_list.append(owner_local_pos)
            k_list.append(owner_local_k)
            v_list.append(owner_local_v)

        for r in range(env_world):
            if r == owner_rank:
                continue
            n = lens[r]
            if n <= 0:
                continue
            pos_r = gathered_pos[r][:n]
            k_r = gathered_k[r][:, :, :n, :]
            v_r = gathered_v[r][:, :, :n, :]

            pos_list.append(pos_r)
            k_list.append(k_r)
            v_list.append(v_r)
            kept_remote_bytes += n * bytes_per_token

        if len(pos_list) == 0:
            raise RuntimeError("assembled KV is empty on owner; invalid transport state")

        merged_pos = torch.cat(pos_list, dim=0)
        merged_k = torch.cat(k_list, dim=2)
        merged_v = torch.cat(v_list, dim=2)

        order = torch.argsort(merged_pos)
        merged_pos = merged_pos[order]
        merged_k = merged_k[:, :, order, :]
        merged_v = merged_v[:, :, order, :]

        keep_idx_sorted = merged_pos.unsqueeze(0)
        sliced_mask = _gather_mask_by_keep_idx(
            attention_mask=attention_mask,
            keep_idx=keep_idx_sorted,
            q_len=query_states.shape[-2],
        )

        merged_k_rep = repeat_kv(merged_k, self.num_key_value_groups)
        merged_v_rep = repeat_kv(merged_v, self.num_key_value_groups)

        attn_output, attn_weights = _run_attention_like_qwen(
            self=self,
            query_states=query_states,
            key_states=merged_k_rep,
            value_states=merged_v_rep,
            attention_mask=sliced_mask,
            output_attentions=output_attentions,
        )
    else:
        attn_output = torch.zeros_like(query_states)
        attn_weights = None

    dist.broadcast(attn_output, src=owner_rank)
    return attn_output, attn_weights, kept_remote_bytes


def custom_skvs_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    正确语义：
    - 非 decode-only 路径：严格委托回原生 attention.forward
    - 只有 decode-only step 才允许进入 SKVS 逻辑
    """
    bsz, q_len, _ = hidden_states.size()

    decode_only_flag = bool(getattr(self, "_skvs_decode_only", False))
    is_decode_step = bool(decode_only_flag and use_cache and q_len == 1 and bsz == 1)

    if self.layer_idx == 0:
        _reset_forward_accounting()

    # 非 decode-only：严格走原生 forward，绝不改变 PPL / prefill 语义
    if not is_decode_step:
        if self.layer_idx == 0:
            GLOBAL_ATTN_STATE["layer0_score"] = None
        return _call_orig_forward(
            self=self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    # ---------- 以下才是 decode-only 路径 ----------
    if position_ids is None:
        raise RuntimeError("decode-only SKVS path requires non-None position_ids")

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                "The cache structure has changed since version v4.36. "
                "If using KV cache, self.layer_idx must not be None."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    full_kv_len = key_states.shape[-2]

    # layer0：生成 guidance，但仍 full attention
    if self.layer_idx == 0:
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        with torch.no_grad():
            attn_scores = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states_rep.shape[-2]]
                attn_scores = attn_scores + causal_mask

            probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
            guidance = probs[:, :, -1, :].max(dim=1).values
            GLOBAL_ATTN_STATE["layer0_score"] = guidance.detach()

        attn_output, attn_weights = _run_attention_like_qwen(
            self=self,
            query_states=query_states,
            key_states=key_states_rep,
            value_states=value_states_rep,
            attention_mask=attention_mask[:, :, :, : key_states_rep.shape[-2]] if attention_mask is not None else None,
            output_attentions=output_attentions,
        )
        _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)

    else:
        l0_score = GLOBAL_ATTN_STATE.get("layer0_score", None)

        owner_rank = int(getattr(self, "_skvs_owner_rank", -1))
        env_rank = int(getattr(self, "_sparse_env_rank", -1))
        env_world = int(getattr(self, "_sparse_env_world", 1))

        can_real_transport = (
            owner_rank >= 0
            and env_rank >= 0
            and env_world > 1
            and dist.is_available()
            and dist.is_initialized()
        )

        if can_real_transport:
            kv_owner_map = _resolve_kv_owner(
                self=self,
                full_kv_len=full_kv_len,
                device=hidden_states.device,
                bsz=bsz,
            )
            if kv_owner_map is None:
                raise RuntimeError("decode-only real SKVS requires kv owner resolution hook")

            remote_keep_idx, dropped_remote_tokens = _build_remote_keep_idx_for_owner(
                l0_score=l0_score,
                kv_owner_map=kv_owner_map,
                owner_rank=owner_rank,
                full_kv_len=full_kv_len,
                device=hidden_states.device,
            )

            dropped_bytes = dropped_remote_tokens * _kv_bytes_per_token(
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                dtype=key_states.dtype,
            )

            attn_output, attn_weights, kept_remote_bytes = _real_transport_owner_only_attention(
                self=self,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                kv_owner_map=kv_owner_map,
                remote_keep_idx=remote_keep_idx,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            _accumulate_accounting(
                dropped_bytes=dropped_bytes,
                kept_remote_bytes=kept_remote_bytes,
            )
        else:
            # decode-only 但条件不足时，退回原生 forward，保证语义严格一致
            return _call_orig_forward(
                self=self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    present_key_value = past_key_value if use_cache else None
    return attn_output, attn_weights, present_key_value