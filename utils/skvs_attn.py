from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import Cache


@dataclass
class SKVSConfig:
    h2o_sink: int = 4
    h2o_window: int = 32
    l0_guidance_ratio: float = 0.6
    min_kv_len: int = 0
    min_sparse_drop_ratio: float = 0.0
    guidance_every_k: int = 1
    guidance_start_kv_len: int = 0


GLOBAL_ATTN_STATE = {
    "layer0_score": None,
    "forward_dropped_bytes": 0,
    "forward_kept_remote_bytes": 0,
    "step_dropped_bytes": 0,
    "num_prefill_fallback": 0,
    "num_decode_fallback": 0,
    "fallback_output_attentions": 0,
    "fallback_skvs_disabled": 0,
    "fallback_prefill_decode_only": 0,
    "fallback_decode_bootstrap": 0,
    "fallback_missing_keep_plan": 0,
    "fallback_prefill_dense_storage": 0,
    "fallback_prefill_oom": 0,
    "fallback_force_dense_append_only": 0,
    "fallback_sparse_oom": 0,
    "num_ratio1_full": 0,
    "num_layer0_full": 0,
    "num_local_sparse": 0,
    "num_transport_sparse": 0,
    "num_local_full": 0,
    "num_transport_full": 0,
    "effective_mode_last": "unknown",
    "forward_candidate_total_tokens": 0,
    "forward_candidate_kept_tokens": 0,
    "forward_candidate_dropped_tokens": 0,
    "forward_total_tokens": 0,
    "forward_kept_tokens": 0,
    "forward_dropped_tokens": 0,
    "forward_dense_ref_tokens": 0,
    "forward_dense_kept_tokens": 0,
    "forward_dense_dropped_tokens": 0,
    "last_step_candidate_total_tokens": 0,
    "last_step_candidate_kept_tokens": 0,
    "last_step_candidate_dropped_tokens": 0,
    "last_step_total_tokens": 0,
    "last_step_kept_tokens": 0,
    "last_step_dropped_tokens": 0,
    "forward_local_selected_cached_tokens": 0,
    "forward_local_selected_protected_tokens": 0,
    "forward_local_selected_unprotected_tokens": 0,
    "forward_local_selected_kept_unprotected_tokens": 0,
    # retained for output-schema compatibility
    "forward_transport_selected_remote_tokens": 0,
    "forward_transport_selected_remote_protected_tokens": 0,
    "forward_transport_selected_remote_unprotected_tokens": 0,
    "forward_transport_selected_remote_kept_unprotected_tokens": 0,
    "migration_selected_kv_bytes": 0,
    "migration_full_kv_bytes": 0,
    "migration_saved_ratio": 0.0,
    "migration_tensor_payload_bytes": 0,
    "migration_tensor_wire_bytes": 0,
    "migration_impl_tensor_count": 0,
    "migration_impl_object_count": 0,
    "decode_step_idx": 0,
    "last_guidance_kv_len": 0,
    "num_guidance_refresh": 0,
    "num_guidance_reuse": 0,
    "num_guidance_start_suppressed": 0,
    "num_guidance_interval_suppressed": 0,
    "last_guidance_refreshed": False,
    # per-layer sparse states: layer_idx -> {pos, k, v, lens}
    "sparse_kv_cache": {},
    "sparse_cache_tokens": 0,
    "sparse_cache_bytes": 0,
    "sparse_cache_layer_tokens": {},
    "sparse_cache_layer_bytes": {},
    # global per-step keep plan, built once at layer0 and shared by layer1+
    "global_keep_step": -1,
    "global_keep_pos": None,   # [B, Kmax]
    "global_keep_lens": None,  # [B]
    "global_keep_kv_len": 0,
    "global_keep_force_dense": False,
    "num_force_dense_plan": 0,
    "num_force_dense_bootstrap_fallback": 0,
    "num_force_dense_append_only": 0,
    "last_plan_candidate_total_tokens": 0,
    "last_plan_candidate_kept_tokens": 0,
    "last_plan_candidate_dropped_tokens": 0,
    "last_plan_dense_ref_tokens": 0,
    "last_plan_dense_kept_tokens": 0,
    "last_plan_dense_dropped_tokens": 0,
}

_SKVS_CFG = SKVSConfig()


def set_skvs_config(cfg: SKVSConfig) -> None:
    global _SKVS_CFG
    _SKVS_CFG = cfg


def reset_skvs_run_accounting() -> None:
    GLOBAL_ATTN_STATE.update(
        {
            "layer0_score": None,
            "forward_dropped_bytes": 0,
            "forward_kept_remote_bytes": 0,
            "step_dropped_bytes": 0,
            "num_prefill_fallback": 0,
            "num_decode_fallback": 0,
            "fallback_output_attentions": 0,
            "fallback_skvs_disabled": 0,
            "fallback_prefill_decode_only": 0,
            "fallback_decode_bootstrap": 0,
            "fallback_missing_keep_plan": 0,
            "fallback_prefill_dense_storage": 0,
            "fallback_prefill_oom": 0,
            "fallback_force_dense_append_only": 0,
            "fallback_sparse_oom": 0,
            "num_ratio1_full": 0,
            "num_layer0_full": 0,
            "num_local_sparse": 0,
            "num_transport_sparse": 0,
            "num_local_full": 0,
            "num_transport_full": 0,
            "effective_mode_last": "unknown",
            "forward_candidate_total_tokens": 0,
            "forward_candidate_kept_tokens": 0,
            "forward_candidate_dropped_tokens": 0,
            "forward_total_tokens": 0,
            "forward_kept_tokens": 0,
            "forward_dropped_tokens": 0,
            "forward_dense_ref_tokens": 0,
            "forward_dense_kept_tokens": 0,
            "forward_dense_dropped_tokens": 0,
            "last_step_candidate_total_tokens": 0,
            "last_step_candidate_kept_tokens": 0,
            "last_step_candidate_dropped_tokens": 0,
            "last_step_total_tokens": 0,
            "last_step_kept_tokens": 0,
            "last_step_dropped_tokens": 0,
            "forward_local_selected_cached_tokens": 0,
            "forward_local_selected_protected_tokens": 0,
            "forward_local_selected_unprotected_tokens": 0,
            "forward_local_selected_kept_unprotected_tokens": 0,
            "forward_transport_selected_remote_tokens": 0,
            "forward_transport_selected_remote_protected_tokens": 0,
            "forward_transport_selected_remote_unprotected_tokens": 0,
            "forward_transport_selected_remote_kept_unprotected_tokens": 0,
            "migration_selected_kv_bytes": 0,
            "migration_full_kv_bytes": 0,
            "migration_saved_ratio": 0.0,
            "migration_tensor_payload_bytes": 0,
            "migration_tensor_wire_bytes": 0,
            "migration_impl_tensor_count": 0,
            "migration_impl_object_count": 0,
            "decode_step_idx": 0,
            "last_guidance_kv_len": 0,
            "num_guidance_refresh": 0,
            "num_guidance_reuse": 0,
            "num_guidance_start_suppressed": 0,
            "num_guidance_interval_suppressed": 0,
            "last_guidance_refreshed": False,
            "sparse_kv_cache": {},
            "sparse_cache_tokens": 0,
            "sparse_cache_bytes": 0,
            "sparse_cache_layer_tokens": {},
            "sparse_cache_layer_bytes": {},
            "global_keep_step": -1,
            "global_keep_pos": None,
            "global_keep_lens": None,
            "global_keep_kv_len": 0,
            "global_keep_force_dense": False,
            "num_force_dense_plan": 0,
            "num_force_dense_bootstrap_fallback": 0,
            "num_force_dense_append_only": 0,
            "last_plan_candidate_total_tokens": 0,
            "last_plan_candidate_kept_tokens": 0,
            "last_plan_candidate_dropped_tokens": 0,
            "last_plan_dense_ref_tokens": 0,
            "last_plan_dense_kept_tokens": 0,
            "last_plan_dense_dropped_tokens": 0,
        }
    )


def reset_skvs_forward_ephemeral() -> None:
    GLOBAL_ATTN_STATE["step_dropped_bytes"] = 0
    GLOBAL_ATTN_STATE["last_step_candidate_total_tokens"] = 0
    GLOBAL_ATTN_STATE["last_step_candidate_kept_tokens"] = 0
    GLOBAL_ATTN_STATE["last_step_candidate_dropped_tokens"] = 0
    GLOBAL_ATTN_STATE["last_step_total_tokens"] = 0
    GLOBAL_ATTN_STATE["last_step_kept_tokens"] = 0
    GLOBAL_ATTN_STATE["last_step_dropped_tokens"] = 0
    GLOBAL_ATTN_STATE["effective_mode_last"] = "unknown"
    GLOBAL_ATTN_STATE["last_guidance_refreshed"] = False


def _inc_state_counter(name: str, amount: int = 1) -> None:
    GLOBAL_ATTN_STATE[name] = int(GLOBAL_ATTN_STATE.get(name, 0) or 0) + int(amount)


def _record_decode_fallback(reason: str) -> None:
    _inc_state_counter("num_decode_fallback")
    _inc_state_counter(f"fallback_{reason}")


def _record_prefill_fallback(reason: str) -> None:
    _inc_state_counter("num_prefill_fallback")
    _inc_state_counter(f"fallback_{reason}")


def _accumulate_accounting(dropped_bytes: int, kept_remote_bytes: int = 0) -> None:
    GLOBAL_ATTN_STATE["step_dropped_bytes"] = int(dropped_bytes)
    GLOBAL_ATTN_STATE["forward_dropped_bytes"] = int(GLOBAL_ATTN_STATE.get("forward_dropped_bytes", 0) or 0) + int(
        dropped_bytes
    )
    GLOBAL_ATTN_STATE["forward_kept_remote_bytes"] = int(
        GLOBAL_ATTN_STATE.get("forward_kept_remote_bytes", 0) or 0
    ) + int(kept_remote_bytes)


def _accumulate_token_accounting(total_tokens: int, kept_tokens: int, dropped_tokens: int) -> None:
    total_tokens = int(total_tokens)
    kept_tokens = int(kept_tokens)
    dropped_tokens = int(dropped_tokens)
    GLOBAL_ATTN_STATE["last_step_candidate_total_tokens"] = total_tokens
    GLOBAL_ATTN_STATE["last_step_candidate_kept_tokens"] = kept_tokens
    GLOBAL_ATTN_STATE["last_step_candidate_dropped_tokens"] = dropped_tokens
    # Backward-compatible aliases retained for older readers; semantically these are
    # candidate-state counters rather than dense-history counters.
    GLOBAL_ATTN_STATE["last_step_total_tokens"] = total_tokens
    GLOBAL_ATTN_STATE["last_step_kept_tokens"] = kept_tokens
    GLOBAL_ATTN_STATE["last_step_dropped_tokens"] = dropped_tokens

    GLOBAL_ATTN_STATE["forward_candidate_total_tokens"] = int(GLOBAL_ATTN_STATE.get("forward_candidate_total_tokens", 0) or 0) + total_tokens
    GLOBAL_ATTN_STATE["forward_candidate_kept_tokens"] = int(GLOBAL_ATTN_STATE.get("forward_candidate_kept_tokens", 0) or 0) + kept_tokens
    GLOBAL_ATTN_STATE["forward_candidate_dropped_tokens"] = int(GLOBAL_ATTN_STATE.get("forward_candidate_dropped_tokens", 0) or 0) + dropped_tokens
    GLOBAL_ATTN_STATE["forward_total_tokens"] = int(GLOBAL_ATTN_STATE.get("forward_total_tokens", 0) or 0) + total_tokens
    GLOBAL_ATTN_STATE["forward_kept_tokens"] = int(GLOBAL_ATTN_STATE.get("forward_kept_tokens", 0) or 0) + kept_tokens
    GLOBAL_ATTN_STATE["forward_dropped_tokens"] = int(GLOBAL_ATTN_STATE.get("forward_dropped_tokens", 0) or 0) + dropped_tokens


def _accumulate_dense_equiv_accounting(full_dense_tokens: int, kept_tokens: int) -> None:
    full_dense_tokens = int(full_dense_tokens)
    kept_tokens = int(kept_tokens)
    dropped_tokens = max(full_dense_tokens - kept_tokens, 0)
    GLOBAL_ATTN_STATE["forward_dense_ref_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_dense_ref_tokens", 0) or 0
    ) + full_dense_tokens
    GLOBAL_ATTN_STATE["forward_dense_kept_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_dense_kept_tokens", 0) or 0
    ) + kept_tokens
    GLOBAL_ATTN_STATE["forward_dense_dropped_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_dense_dropped_tokens", 0) or 0
    ) + dropped_tokens


def _accumulate_local_selection_stats(
    cached_tokens: int,
    protected_tokens: int,
    unprotected_tokens: int,
    kept_unprotected_tokens: int,
) -> None:
    GLOBAL_ATTN_STATE["forward_local_selected_cached_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_local_selected_cached_tokens", 0) or 0
    ) + int(cached_tokens)
    GLOBAL_ATTN_STATE["forward_local_selected_protected_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_local_selected_protected_tokens", 0) or 0
    ) + int(protected_tokens)
    GLOBAL_ATTN_STATE["forward_local_selected_unprotected_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_local_selected_unprotected_tokens", 0) or 0
    ) + int(unprotected_tokens)
    GLOBAL_ATTN_STATE["forward_local_selected_kept_unprotected_tokens"] = int(
        GLOBAL_ATTN_STATE.get("forward_local_selected_kept_unprotected_tokens", 0) or 0
    ) + int(kept_unprotected_tokens)


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


def _call_orig_forward(self, **kwargs):
    orig_forward = getattr(self, "_skvs_orig_forward", None)
    if orig_forward is None:
        raise RuntimeError("Missing self._skvs_orig_forward; save original forward before patching")
    return orig_forward(**kwargs)


def _ensure_guidance_2d(guidance: Optional[torch.Tensor], bsz: int, need_len: int, device: torch.device) -> torch.Tensor:
    if guidance is None:
        return torch.zeros((bsz, need_len), device=device, dtype=torch.float32)
    g = guidance.to(device=device, dtype=torch.float32)
    if g.dim() == 1:
        g = g.view(1, -1).expand(bsz, -1)
    elif g.dim() > 2:
        g = g.view(g.shape[0], -1)
    if g.shape[0] == 1 and bsz > 1:
        g = g.expand(bsz, g.shape[1])
    elif g.shape[0] != bsz:
        g = g[:1, :].expand(bsz, g.shape[1])
    if g.shape[1] < need_len:
        pad = torch.zeros((bsz, need_len - g.shape[1]), device=device, dtype=torch.float32)
        g = torch.cat([g, pad], dim=1)
    elif g.shape[1] > need_len:
        g = g[:, :need_len]
    return g.contiguous()


def _infer_valid_positions_from_attention_mask(
    attention_mask: Optional[torch.Tensor],
    bsz: int,
    kv_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    abs_pos = torch.arange(kv_len, device=device, dtype=torch.long).view(1, kv_len).expand(bsz, kv_len)
    if attention_mask is None or attention_mask.dim() != 4:
        return torch.ones((bsz, kv_len), device=device, dtype=torch.bool), abs_pos
    neg_large = torch.finfo(attention_mask.dtype).min / 2 if attention_mask.dtype.is_floating_point else -1e30
    row = attention_mask[:, 0, -1, :kv_len]
    valid_mask = row > neg_large
    if not torch.any(valid_mask):
        valid_mask = torch.ones((bsz, kv_len), device=device, dtype=torch.bool)
    return valid_mask, abs_pos


def _run_attention_like_qwen(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, None


def _run_full_attention_after_cache_update(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    full_kv_len: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
    value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
    mask_full = attention_mask[:, :, :, :full_kv_len] if attention_mask is not None else None
    return _run_attention_like_qwen(
        self=self,
        query_states=query_states,
        key_states=key_states_rep,
        value_states=value_states_rep,
        attention_mask=mask_full,
    )


def _safe_current_abs_positions(
    position_ids: Optional[torch.LongTensor],
    cache_position: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    bsz: int,
    q_len: int,
    device: torch.device,
) -> torch.LongTensor:
    def _fallback() -> torch.LongTensor:
        return torch.arange(q_len, device=device, dtype=torch.long).view(1, q_len).expand(bsz, q_len)

    def _from_attention_mask() -> Optional[torch.LongTensor]:
        if attention_mask is None or int(attention_mask.shape[0]) != int(bsz):
            return None
        try:
            if attention_mask.dim() == 2:
                if int(attention_mask.shape[1]) < int(q_len):
                    return None
                lengths = attention_mask.to(device=device).to(torch.long).sum(dim=1).clamp_min(1)
            elif attention_mask.dim() == 4:
                # Inside HF attention modules the 2D mask is often already
                # expanded to [B,1,Q,K].  Its K dimension is the best available
                # dense-equivalent sequence length after we clear DynamicCache.
                kv_len = int(attention_mask.shape[-1])
                if kv_len < int(q_len):
                    return None
                lengths = torch.full((bsz,), kv_len, device=device, dtype=torch.long)
            else:
                return None
            if int(q_len) == 1:
                return (lengths - 1).view(bsz, 1)
            start = (lengths - int(q_len)).clamp_min(0).view(bsz, 1)
            offsets = torch.arange(q_len, device=device, dtype=torch.long).view(1, q_len)
            return start + offsets
        except Exception:
            return None

    def _normalize(pos: torch.Tensor) -> torch.Tensor:
        pos = pos.to(device=device, dtype=torch.long)
        if pos.numel() == 0:
            return _fallback()
        if pos.dim() == 0:
            pos = pos.view(1, 1)
        elif pos.dim() == 1:
            pos = pos.view(1, -1)
        elif pos.dim() > 2:
            pos = pos.view(pos.shape[0], -1)
        if pos.shape[0] == 0:
            return _fallback()
        if pos.shape[0] == 1 and bsz > 1:
            pos = pos.expand(bsz, pos.shape[1])
        elif pos.shape[0] != bsz:
            pos = pos[:1, :].expand(bsz, pos.shape[1])
        return pos

    # When SKVS drops the dense HF cache, some HF model wrappers can derive a
    # stale cache_position from the now-empty DynamicCache.  The explicit
    # attention_mask still carries the true decode length in our greedy loop, so
    # prefer it for q_len==1.
    if int(q_len) == 1:
        from_mask = _from_attention_mask()
        if from_mask is not None:
            return from_mask

    if position_ids is not None:
        return _normalize(position_ids)
    if cache_position is not None:
        return _normalize(cache_position)
    from_mask = _from_attention_mask()
    if from_mask is not None:
        return from_mask
    return _fallback()


def _dense_cache_clear_enabled() -> bool:
    val = str(os.environ.get("SPARSE_SKVS_CLEAR_DENSE_CACHE", "1")).strip().lower()
    return val not in ("0", "false", "no", "off")


def _skvs_start_kv_len() -> int:
    return max(int(os.environ.get("SPARSE_SKVS_START_KV_LEN", "0") or 0), 0)


def _estimate_current_kv_len(
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    cache_position: Optional[torch.LongTensor],
    q_len: int,
) -> int:
    try:
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                return int(attention_mask.shape[-1])
            if attention_mask.dim() == 4:
                return int(attention_mask.shape[-1])
        pos = cache_position if cache_position is not None else position_ids
        if torch.is_tensor(pos) and pos.numel() > 0:
            return int(pos.detach().max().item()) + 1
    except Exception:
        pass
    return int(q_len)


def _compact_this_step(step_idx: int) -> bool:
    every = max(int(os.environ.get("SPARSE_SKVS_COMPACT_EVERY_K", "1") or 1), 1)
    if every <= 1:
        return True
    return (int(step_idx) % every) == 1


def _skvs_block_size() -> int:
    return max(int(os.environ.get("SPARSE_SKVS_BLOCK_SIZE", "1") or 1), 1)


def _maybe_clear_dense_layer_cache(
    self,
    past_key_value: Optional[Cache],
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not _dense_cache_clear_enabled():
        return
    _try_clear_dense_layer_cache(
        past_key_value=past_key_value,
        layer_idx=int(getattr(self, "layer_idx", -1)),
        bsz=int(bsz),
        num_kv_heads=int(getattr(self, "num_key_value_heads", 0)),
        head_dim=int(getattr(self, "head_dim", 0)),
        device=device,
        dtype=dtype,
    )


def _reset_sparse_cache() -> None:
    GLOBAL_ATTN_STATE["sparse_kv_cache"] = {}
    GLOBAL_ATTN_STATE["sparse_cache_tokens"] = 0
    GLOBAL_ATTN_STATE["sparse_cache_bytes"] = 0
    GLOBAL_ATTN_STATE["sparse_cache_layer_tokens"] = {}
    GLOBAL_ATTN_STATE["sparse_cache_layer_bytes"] = {}


def _update_sparse_cache_accounting_for_layer(layer_idx: int, state: Dict[str, torch.Tensor]) -> None:
    token_map = GLOBAL_ATTN_STATE.setdefault("sparse_cache_layer_tokens", {})
    byte_map = GLOBAL_ATTN_STATE.setdefault("sparse_cache_layer_bytes", {})

    prev_tokens = int(token_map.get(layer_idx, 0) or 0)
    prev_bytes = int(byte_map.get(layer_idx, 0) or 0)

    lens = state.get("lens", None)
    k = state.get("k", None)
    v = state.get("v", None)
    if lens is None or k is None or v is None:
        cur_tokens = 0
        cur_bytes = 0
    else:
        cur_tokens = int(lens.sum().item())
        cur_bytes = int(k.numel() * k.element_size()) + int(v.numel() * v.element_size())

    GLOBAL_ATTN_STATE["sparse_cache_tokens"] = max(
        int(GLOBAL_ATTN_STATE.get("sparse_cache_tokens", 0) or 0) - prev_tokens + cur_tokens,
        0,
    )
    GLOBAL_ATTN_STATE["sparse_cache_bytes"] = max(
        int(GLOBAL_ATTN_STATE.get("sparse_cache_bytes", 0) or 0) - prev_bytes + cur_bytes,
        0,
    )
    token_map[layer_idx] = cur_tokens
    byte_map[layer_idx] = cur_bytes


def _try_clear_dense_layer_cache(
    past_key_value: Optional[Cache],
    layer_idx: int,
    bsz: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if past_key_value is None:
        return
    empty_k = torch.empty((bsz, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
    empty_v = torch.empty((bsz, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
    for kname, vname in (("key_cache", "value_cache"), ("keys", "values")):
        if hasattr(past_key_value, kname) and hasattr(past_key_value, vname):
            try:
                kcache = getattr(past_key_value, kname)
                vcache = getattr(past_key_value, vname)
                if isinstance(kcache, list) and isinstance(vcache, list) and len(kcache) > layer_idx and len(vcache) > layer_idx:
                    kcache[layer_idx] = empty_k
                    vcache[layer_idx] = empty_v
                    return
            except Exception:
                pass


def _vectorized_select_keep_mask(
    scores: torch.Tensor,          # [B,S]
    abs_pos: torch.Tensor,         # [B,S]
    valid_mask: torch.Tensor,      # [B,S]
    current_abs_pos: torch.Tensor, # [B]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    sink = max(int(_SKVS_CFG.h2o_sink), 0)
    window = max(int(_SKVS_CFG.h2o_window), 0)
    ratio = min(max(float(_SKVS_CFG.l0_guidance_ratio), 0.0), 1.0)

    protected = torch.zeros_like(valid_mask)
    if sink > 0:
        protected |= abs_pos < sink
    if window > 0:
        win_start = (current_abs_pos - window + 1).clamp_min(0).unsqueeze(1)
        protected |= abs_pos >= win_start
    protected &= valid_mask

    unprotected = valid_mask & (~protected)
    unprot_counts = unprotected.sum(dim=1)
    prot_counts = protected.sum(dim=1)
    keep_unprot = torch.ceil(unprot_counts.float() * ratio).to(torch.long)
    keep_unprot = torch.where(unprot_counts > 0, keep_unprot.clamp_min(1), keep_unprot)
    keep_unprot = torch.minimum(keep_unprot, unprot_counts)

    selected_unprotected = torch.zeros_like(valid_mask)
    block_size = _skvs_block_size()
    if block_size > 1:
        for b in range(int(valid_mask.shape[0])):
            target = int(keep_unprot[b].item())
            if target <= 0:
                continue
            idx = torch.nonzero(unprotected[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            block_ids = torch.div(abs_pos[b].index_select(0, idx), int(block_size), rounding_mode="floor")
            uniq = torch.unique(block_ids, sorted=True)
            ranked = []
            for ub in uniq.tolist():
                bid = int(ub)
                m = block_ids == bid
                idx_b = idx[m]
                if idx_b.numel() == 0:
                    continue
                score_b = float(scores[b].index_select(0, idx_b).max().item())
                ranked.append((score_b, bid, idx_b))
            ranked.sort(key=lambda x: x[0], reverse=True)
            kept = 0
            for _, _, idx_b in ranked:
                selected_unprotected[b, idx_b] = True
                kept += int(idx_b.numel())
                if kept >= target:
                    break
    else:
        max_keep = int(keep_unprot.max().item()) if keep_unprot.numel() > 0 else 0
        if max_keep <= 0:
            keep_mask = protected | selected_unprotected
            stats = {
                "cached": valid_mask.sum(dim=1),
                "protected": prot_counts,
                "unprotected": unprot_counts,
                "kept_unprotected": selected_unprotected.sum(dim=1),
            }
            return keep_mask, stats
        neg_inf = torch.full_like(scores, -1e30)
        masked_scores = torch.where(unprotected, scores, neg_inf)
        topv, topi = torch.topk(masked_scores, k=max_keep, dim=1, largest=True, sorted=True)
        rank_mask = torch.arange(max_keep, device=scores.device).view(1, max_keep) < keep_unprot.view(-1, 1)
        valid_top = rank_mask & torch.isfinite(topv)
        selected_unprotected.scatter_(1, topi, valid_top)

    keep_mask = protected | selected_unprotected
    stats = {
        "cached": valid_mask.sum(dim=1),
        "protected": prot_counts,
        "unprotected": unprot_counts,
        "kept_unprotected": selected_unprotected.sum(dim=1),
    }
    return keep_mask, stats

def _compress_selected_pos(abs_pos: torch.Tensor, keep_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, slen = abs_pos.shape
    counts = keep_mask.sum(dim=1).to(torch.long)
    kmax = int(counts.max().item()) if counts.numel() > 0 else 0
    if kmax <= 0:
        raise RuntimeError("[SKVS][global-plan] empty keep set")

    large = torch.full_like(abs_pos, fill_value=int(1e9))
    sort_key = torch.where(keep_mask, abs_pos, large)
    gather_order = torch.argsort(sort_key, dim=1)[:, :kmax]
    pos_sel = abs_pos.gather(1, gather_order)
    valid_sel = keep_mask.gather(1, gather_order)
    pos_sel = torch.where(valid_sel, pos_sel, torch.zeros_like(pos_sel))
    return pos_sel.contiguous(), counts.contiguous()


def _membership_keep_mask(
    candidate_pos: torch.Tensor,   # [B,S]
    candidate_valid: torch.Tensor, # [B,S]
    keep_pos: torch.Tensor,        # [B,K]
    keep_lens: torch.Tensor,       # [B]
) -> torch.Tensor:
    kmax = keep_pos.shape[1]
    if kmax <= 0:
        return torch.zeros_like(candidate_valid)

    idx = torch.searchsorted(keep_pos, candidate_pos, right=False)
    idx_clamped = idx.clamp(min=0, max=max(kmax - 1, 0))
    gathered = torch.gather(keep_pos, 1, idx_clamped)
    valid_idx = idx < keep_lens.view(-1, 1)
    return candidate_valid & valid_idx & (gathered == candidate_pos)


def _selection_stats_from_keep_mask(
    abs_pos: torch.Tensor,
    valid_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    current_abs_pos: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    sink = max(int(_SKVS_CFG.h2o_sink), 0)
    window = max(int(_SKVS_CFG.h2o_window), 0)

    protected = torch.zeros_like(valid_mask)
    if sink > 0:
        protected |= abs_pos < sink
    if window > 0:
        win_start = (current_abs_pos - window + 1).clamp_min(0).unsqueeze(1)
        protected |= abs_pos >= win_start
    protected &= valid_mask
    unprotected = valid_mask & (~protected)
    kept_unprotected = keep_mask & unprotected

    return {
        "cached": valid_mask.sum(dim=1),
        "protected": protected.sum(dim=1),
        "unprotected": unprotected.sum(dim=1),
        "kept_unprotected": kept_unprotected.sum(dim=1),
    }


def _compress_selected_kv(
    abs_pos: torch.Tensor,      # [B,S]
    keep_mask: torch.Tensor,    # [B,S]
    key_states: torch.Tensor,   # [B,H,S,D]
    value_states: torch.Tensor, # [B,H,S,D]
) -> Dict[str, torch.Tensor]:
    bsz, num_kv_heads, _, head_dim = key_states.shape
    counts = keep_mask.sum(dim=1).to(torch.long)
    kmax = int(counts.max().item()) if counts.numel() > 0 else 0
    if kmax <= 0:
        raise RuntimeError("[SKVS][sparse-storage] empty selected set")

    large = torch.full_like(abs_pos, fill_value=int(1e9))
    sort_key = torch.where(keep_mask, abs_pos, large)
    gather_order = torch.argsort(sort_key, dim=1)[:, :kmax]
    pos_sel = abs_pos.gather(1, gather_order)
    gather_valid = keep_mask.gather(1, gather_order)
    pos_sel = torch.where(gather_valid, pos_sel, torch.zeros_like(pos_sel))

    gather_idx = gather_order[:, None, :, None].expand(bsz, num_kv_heads, kmax, head_dim)
    k_sel = torch.gather(key_states, 2, gather_idx)
    v_sel = torch.gather(value_states, 2, gather_idx)
    return {"pos": pos_sel.contiguous(), "k": k_sel.contiguous(), "v": v_sel.contiguous(), "lens": counts.contiguous()}


def _append_to_sparse_state(
    state: Dict[str, torch.Tensor],
    cur_pos: torch.Tensor,  # [B]
    cur_k: torch.Tensor,    # [B,H,1,D]
    cur_v: torch.Tensor,    # [B,H,1,D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.cat([state["pos"], cur_pos.view(-1, 1)], dim=1)
    k = torch.cat([state["k"], cur_k], dim=2)
    v = torch.cat([state["v"], cur_v], dim=2)
    klen_prev = state["k"].shape[2]
    valid_prev = torch.arange(klen_prev, device=cur_k.device).view(1, klen_prev) < state["lens"].view(-1, 1)
    valid_mask = torch.cat([valid_prev, torch.ones((cur_pos.shape[0], 1), device=cur_k.device, dtype=torch.bool)], dim=1)
    return pos, k, v, valid_mask


def _run_sparse_state_attention_gqa(
    self,
    query_states: torch.Tensor,  # [B,H,1,D]
    state: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Global planning version still avoids repeat_kv materialization.
    k = state["k"]  # [B,Hkv,K,D]
    v = state["v"]  # [B,Hkv,K,D]
    lens = state["lens"]
    bsz, kv_heads, kmax, head_dim = k.shape
    num_heads = query_states.shape[1]
    groups = max(int(num_heads // max(kv_heads, 1)), 1)
    q = query_states.view(bsz, kv_heads, groups, 1, head_dim)  # [B,Hkv,G,1,D]

    scores = torch.einsum("bhgqd,bhkd->bhgqk", q, k) / math.sqrt(head_dim)
    valid = torch.arange(kmax, device=k.device).view(1, 1, 1, 1, kmax) < lens.view(bsz, 1, 1, 1, 1)
    neg_large = torch.finfo(scores.dtype).min / 2 if scores.dtype.is_floating_point else -1e30
    scores = scores.masked_fill(~valid, neg_large)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("bhgqk,bhkd->bhgqd", probs, v)
    out = out.reshape(bsz, num_heads, 1, head_dim).contiguous()
    return out, None

def _run_layer0_full_attention_with_guidance_gqa(
    self,
    query_states: torch.Tensor,   # [B,H,1,D]
    key_states: torch.Tensor,     # [B,Hkv,S,D]
    value_states: torch.Tensor,   # [B,Hkv,S,D]
    attention_mask: Optional[torch.Tensor],  # [B,1,1,S] or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Equivalent to repeat_kv + dense attention path, but avoids materializing repeated K/V.
    Returns:
      attn_output: [B,H,1,D]
      guidance: [B,S] (mean attention over heads at current decode step)
    """
    bsz, kv_heads, kv_len, head_dim = key_states.shape
    num_heads = int(query_states.shape[1])
    groups = max(int(num_heads // max(kv_heads, 1)), 1)

    q = query_states.view(bsz, kv_heads, groups, 1, head_dim)  # [B,Hkv,G,1,D]
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q, key_states) / math.sqrt(head_dim)  # [B,Hkv,G,1,S]
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :kv_len].unsqueeze(2)  # [B,1,1,1,S]
        scores = scores + mask

    probs_f32 = F.softmax(scores, dim=-1, dtype=torch.float32)  # [B,Hkv,G,1,S]
    out = torch.einsum("bhgqk,bhkd->bhgqd", probs_f32.to(q.dtype), value_states)  # [B,Hkv,G,1,D]
    out = out.reshape(bsz, num_heads, 1, head_dim).contiguous()
    guidance = probs_f32[..., 0, :].mean(dim=(1, 2)).contiguous()  # [B,S]
    return out, guidance


def _estimate_full_kv_len(attention_mask: Optional[torch.Tensor], current_abs: torch.Tensor, q_len: int) -> int:
    if attention_mask is not None and attention_mask.dim() == 4:
        return int(attention_mask.shape[-1])
    return int(current_abs.max().item()) + q_len


def _accumulate_full_path_stats(attention_mask: Optional[torch.Tensor], bsz: int, kv_len: int, device: torch.device) -> None:
    valid_mask, _ = _infer_valid_positions_from_attention_mask(attention_mask, bsz, kv_len, device)
    total_tokens = int(valid_mask.sum().item())
    _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=total_tokens, dropped_tokens=0)
    _accumulate_dense_equiv_accounting(full_dense_tokens=total_tokens, kept_tokens=total_tokens)
    _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)


def _build_and_store_global_keep_plan(
    guidance: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    bsz: int,
    kv_len: int,
    current_abs: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, bool, float]:
    valid_mask, abs_pos = _infer_valid_positions_from_attention_mask(attention_mask, bsz, kv_len, device)
    score = _ensure_guidance_2d(guidance, bsz, kv_len, device)
    keep_mask, _ = _vectorized_select_keep_mask(score, abs_pos, valid_mask, current_abs)
    keep_pos, keep_lens = _compress_selected_pos(abs_pos, keep_mask)
    total_tokens = int(valid_mask.sum().item())
    kept_tokens = int(keep_mask.sum().item())
    dropped_tokens = max(total_tokens - kept_tokens, 0)
    drop_ratio = float(dropped_tokens) / max(total_tokens, 1)
    force_dense = False
    min_kv_len = int(getattr(_SKVS_CFG, "min_kv_len", 0) or 0)
    min_sparse_drop_ratio = float(getattr(_SKVS_CFG, "min_sparse_drop_ratio", 0.0) or 0.0)
    if min_kv_len > 0 and kv_len < min_kv_len:
        force_dense = True
    if min_sparse_drop_ratio > 0.0 and drop_ratio < min_sparse_drop_ratio:
        force_dense = True

    GLOBAL_ATTN_STATE["global_keep_step"] = int(GLOBAL_ATTN_STATE.get("decode_step_idx", 0) or 0)
    GLOBAL_ATTN_STATE["global_keep_pos"] = keep_pos
    GLOBAL_ATTN_STATE["global_keep_lens"] = keep_lens
    GLOBAL_ATTN_STATE["global_keep_kv_len"] = int(kv_len)
    GLOBAL_ATTN_STATE["global_keep_force_dense"] = bool(force_dense)
    GLOBAL_ATTN_STATE["last_plan_candidate_total_tokens"] = total_tokens
    GLOBAL_ATTN_STATE["last_plan_candidate_kept_tokens"] = kept_tokens
    GLOBAL_ATTN_STATE["last_plan_candidate_dropped_tokens"] = dropped_tokens
    GLOBAL_ATTN_STATE["last_plan_dense_ref_tokens"] = total_tokens
    GLOBAL_ATTN_STATE["last_plan_dense_kept_tokens"] = kept_tokens
    GLOBAL_ATTN_STATE["last_plan_dense_dropped_tokens"] = dropped_tokens
    if force_dense:
        GLOBAL_ATTN_STATE["num_force_dense_plan"] = int(GLOBAL_ATTN_STATE.get("num_force_dense_plan", 0) or 0) + 1
    return keep_pos, keep_lens, force_dense, drop_ratio


def _owner_local_prefill_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    """
    Unified owner-local entry for prefill.
    We intentionally route prefill into the same explicit SKVS state machine as decode,
    so this helper only resets per-run state and then lets custom_skvs_attention_forward
    continue into the prefill sparse-storage path below.
    """
    if self.layer_idx == 0:
        GLOBAL_ATTN_STATE["layer0_score"] = None
        GLOBAL_ATTN_STATE["last_guidance_kv_len"] = 0
        _reset_sparse_cache()
        GLOBAL_ATTN_STATE["global_keep_step"] = -1
        GLOBAL_ATTN_STATE["global_keep_pos"] = None
        GLOBAL_ATTN_STATE["global_keep_lens"] = None
        GLOBAL_ATTN_STATE["global_keep_kv_len"] = 0
        GLOBAL_ATTN_STATE["global_keep_force_dense"] = False
        GLOBAL_ATTN_STATE["global_keep_sample_ids"] = None
    # Fall through to the explicit prefill sparse-state logic implemented in
    # custom_skvs_attention_forward instead of calling the original dense forward.
    return None




def _sample_ids_for_current_batch(self, bsz: int, device: torch.device) -> torch.Tensor:
    sample_ids = getattr(self, "_skvs_local_sample_ids", None)
    if torch.is_tensor(sample_ids):
        sample_ids = sample_ids.detach().to(device=device, dtype=torch.long).view(-1)
        if int(sample_ids.numel()) == int(bsz):
            return sample_ids.contiguous()
    return torch.arange(bsz, device=device, dtype=torch.long)


def _past_layer_kv_len(past_key_value: Optional[Cache], layer_idx: int) -> int:
    if past_key_value is None or int(layer_idx) < 0:
        return 0
    for kname, _ in (("key_cache", "value_cache"), ("keys", "values")):
        if not hasattr(past_key_value, kname):
            continue
        try:
            kcache = getattr(past_key_value, kname)
            if isinstance(kcache, list) and len(kcache) > int(layer_idx):
                k_hist = kcache[int(layer_idx)]
                if torch.is_tensor(k_hist) and k_hist.dim() >= 3:
                    return int(k_hist.shape[-2])
        except Exception:
            continue
    return 0


def _correct_decode_abs_positions(
    self,
    past_key_value: Optional[Cache],
    sample_ids: torch.Tensor,
    current_abs_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if current_abs_pos.numel() == 0:
        return current_abs_pos
    layer_idx = int(getattr(self, "layer_idx", -1))
    cur = current_abs_pos[:, -1].to(device=device, dtype=torch.long)
    corrected = cur
    try:
        state = _get_layer_state_for_sample_ids(layer_idx, sample_ids)
        if state is not None:
            prev_valid = _valid_mask_from_state(state["pos"], state["lens"])
            if prev_valid.numel() > 0 and torch.any(prev_valid):
                invalid_fill = torch.full_like(state["pos"], -1)
                prev_max_pos = torch.where(prev_valid, state["pos"], invalid_fill).max(dim=1).values
                stale_pos = corrected <= prev_max_pos
                if torch.any(stale_pos):
                    corrected = torch.where(stale_pos, prev_max_pos + 1, corrected)
                    _inc_state_counter("fallback_stale_cache_position")
        else:
            hist_len = _past_layer_kv_len(past_key_value, layer_idx)
            if hist_len > 0:
                hist_next = torch.full_like(corrected, int(hist_len))
                stale_pos = corrected < hist_next
                if torch.any(stale_pos):
                    corrected = torch.where(stale_pos, hist_next, corrected)
                    _inc_state_counter("fallback_stale_cache_position")
    except Exception:
        return current_abs_pos
    if current_abs_pos.shape[-1] == 1:
        return corrected.view(-1, 1)
    out = current_abs_pos.clone()
    out[:, -1] = corrected
    return out


def _dense_state_from_full_sequence(
    sample_ids: torch.Tensor,
    abs_pos: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    lens = torch.full((abs_pos.shape[0],), int(abs_pos.shape[1]), device=abs_pos.device, dtype=torch.long)
    return {
        "sample_ids": sample_ids.contiguous(),
        "pos": abs_pos.contiguous(),
        "k": key_states.contiguous(),
        "v": value_states.contiguous(),
        "lens": lens.contiguous(),
    }


def _empty_state_for_samples(
    sample_ids: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    bsz = int(sample_ids.numel())
    return {
        "sample_ids": sample_ids.to(device=device, dtype=torch.long).contiguous(),
        "pos": torch.empty((bsz, 0), device=device, dtype=torch.long),
        "k": torch.empty((bsz, num_kv_heads, 0, head_dim), device=device, dtype=dtype),
        "v": torch.empty((bsz, num_kv_heads, 0, head_dim), device=device, dtype=dtype),
        "lens": torch.zeros((bsz,), device=device, dtype=torch.long),
    }




def _bootstrap_state_from_past_key_value(
    self,
    past_key_value: Optional[Cache],
    sample_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Dict[str, torch.Tensor]]:
    if past_key_value is None:
        return None
    layer_idx = int(getattr(self, "layer_idx", -1))
    if layer_idx < 0:
        return None

    k_hist = None
    v_hist = None
    for kname, vname in (("key_cache", "value_cache"), ("keys", "values")):
        if hasattr(past_key_value, kname) and hasattr(past_key_value, vname):
            try:
                kcache = getattr(past_key_value, kname)
                vcache = getattr(past_key_value, vname)
                if isinstance(kcache, list) and isinstance(vcache, list) and len(kcache) > layer_idx and len(vcache) > layer_idx:
                    k_hist = kcache[layer_idx]
                    v_hist = vcache[layer_idx]
                    break
            except Exception:
                pass
    if k_hist is None or v_hist is None or (not torch.is_tensor(k_hist)) or (not torch.is_tensor(v_hist)):
        return None

    sample_ids = sample_ids.to(device=device, dtype=torch.long).view(-1)
    bsz = int(sample_ids.numel())
    k_hist = k_hist.to(device=device)
    v_hist = v_hist.to(device=device)
    if k_hist.dim() != 4 or v_hist.dim() != 4:
        return None

    cache_bsz = int(k_hist.shape[0])
    if cache_bsz == bsz:
        k_local = k_hist.contiguous()
        v_local = v_hist.contiguous()
    else:
        if int(sample_ids.max().item()) >= cache_bsz if bsz > 0 else False:
            return None
        rows = sample_ids.to(device=device, dtype=torch.long)
        k_local = k_hist.index_select(0, rows).contiguous()
        v_local = v_hist.index_select(0, rows).contiguous()

    hist_len = int(k_local.shape[-2])
    if hist_len <= 0:
        return _empty_state_for_samples(sample_ids, self.num_key_value_heads, self.head_dim, device, dtype)

    valid_mask, abs_pos = _infer_valid_positions_from_attention_mask(attention_mask, bsz, hist_len, device)
    selected = _compress_selected_kv(abs_pos, valid_mask, k_local.to(dtype=dtype), v_local.to(dtype=dtype))
    selected["sample_ids"] = sample_ids.contiguous()
    return selected




def _valid_mask_from_state(pos: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    kmax = int(pos.shape[1])
    if kmax <= 0:
        return torch.zeros_like(pos, dtype=torch.bool)
    return torch.arange(kmax, device=pos.device).view(1, kmax) < lens.view(-1, 1)




def _state_without_sample_ids(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k != "sample_ids"}




def _update_sparse_cache_accounting_full() -> None:
    GLOBAL_ATTN_STATE["sparse_cache_tokens"] = 0
    GLOBAL_ATTN_STATE["sparse_cache_bytes"] = 0
    GLOBAL_ATTN_STATE["sparse_cache_layer_tokens"] = {}
    GLOBAL_ATTN_STATE["sparse_cache_layer_bytes"] = {}
    sparse_cache: Dict[int, Dict[str, torch.Tensor]] = GLOBAL_ATTN_STATE.setdefault("sparse_kv_cache", {})
    for layer_idx, state in sparse_cache.items():
        _update_sparse_cache_accounting_for_layer(int(layer_idx), _state_without_sample_ids(state))




def _store_layer_state(layer_idx: int, state: Dict[str, torch.Tensor]) -> None:
    sparse_cache: Dict[int, Dict[str, torch.Tensor]] = GLOBAL_ATTN_STATE.setdefault("sparse_kv_cache", {})
    sparse_cache[int(layer_idx)] = dict(state)
    _update_sparse_cache_accounting_for_layer(int(layer_idx), _state_without_sample_ids(state))




def _select_rows_from_state(state: Dict[str, torch.Tensor], rows: torch.Tensor, sample_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    rows = rows.to(device=state["pos"].device, dtype=torch.long)
    return {
        "sample_ids": sample_ids.to(device=state["pos"].device, dtype=torch.long).contiguous(),
        "pos": state["pos"].index_select(0, rows).contiguous(),
        "k": state["k"].index_select(0, rows).contiguous(),
        "v": state["v"].index_select(0, rows).contiguous(),
        "lens": state["lens"].index_select(0, rows).contiguous(),
    }




def _get_layer_state_for_sample_ids(layer_idx: int, sample_ids: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
    sparse_cache: Dict[int, Dict[str, torch.Tensor]] = GLOBAL_ATTN_STATE.setdefault("sparse_kv_cache", {})
    state = sparse_cache.get(int(layer_idx), None)
    if state is None:
        return None
    state_sids = state.get("sample_ids", None)
    if state_sids is None:
        return None
    state_sids = state_sids.to(device=sample_ids.device, dtype=torch.long).view(-1)
    sample_ids = sample_ids.to(device=sample_ids.device, dtype=torch.long).view(-1)
    if int(state_sids.numel()) == int(sample_ids.numel()) and torch.equal(state_sids, sample_ids):
        return state
    row_map = {int(s): i for i, s in enumerate(state_sids.tolist())}
    rows = []
    for sid in sample_ids.tolist():
        sid = int(sid)
        if sid not in row_map:
            return None
        rows.append(int(row_map[sid]))
    rows_t = torch.tensor(rows, device=state["pos"].device, dtype=torch.long)
    return _select_rows_from_state(state, rows_t, sample_ids.to(device=state["pos"].device, dtype=torch.long))




def _store_keep_plan_for_sample_ids(
    sample_ids: torch.Tensor,
    keep_pos: torch.Tensor,
    keep_lens: torch.Tensor,
    kv_len: int,
    force_dense: bool,
    total_tokens: int,
    kept_tokens: int,
    dropped_tokens: int,
) -> None:
    GLOBAL_ATTN_STATE["global_keep_step"] = int(GLOBAL_ATTN_STATE.get("decode_step_idx", 0) or 0)
    GLOBAL_ATTN_STATE["global_keep_sample_ids"] = sample_ids.detach().to(dtype=torch.long).contiguous()
    GLOBAL_ATTN_STATE["global_keep_pos"] = keep_pos.contiguous()
    GLOBAL_ATTN_STATE["global_keep_lens"] = keep_lens.contiguous()
    GLOBAL_ATTN_STATE["global_keep_kv_len"] = int(kv_len)
    GLOBAL_ATTN_STATE["global_keep_force_dense"] = bool(force_dense)
    GLOBAL_ATTN_STATE["last_plan_candidate_total_tokens"] = int(total_tokens)
    GLOBAL_ATTN_STATE["last_plan_candidate_kept_tokens"] = int(kept_tokens)
    GLOBAL_ATTN_STATE["last_plan_candidate_dropped_tokens"] = int(dropped_tokens)
    GLOBAL_ATTN_STATE["last_plan_dense_ref_tokens"] = int(total_tokens)
    GLOBAL_ATTN_STATE["last_plan_dense_kept_tokens"] = int(kept_tokens)
    GLOBAL_ATTN_STATE["last_plan_dense_dropped_tokens"] = int(dropped_tokens)
    if force_dense:
        GLOBAL_ATTN_STATE["num_force_dense_plan"] = int(GLOBAL_ATTN_STATE.get("num_force_dense_plan", 0) or 0) + 1




def _get_keep_plan_for_sample_ids(sample_ids: torch.Tensor):
    keep_pos = GLOBAL_ATTN_STATE.get("global_keep_pos", None)
    keep_lens = GLOBAL_ATTN_STATE.get("global_keep_lens", None)
    keep_sids = GLOBAL_ATTN_STATE.get("global_keep_sample_ids", None)
    keep_force_dense = bool(GLOBAL_ATTN_STATE.get("global_keep_force_dense", False))
    if keep_pos is None or keep_lens is None or keep_sids is None:
        return None, None, keep_force_dense
    keep_sids = keep_sids.to(device=sample_ids.device, dtype=torch.long).view(-1)
    sample_ids = sample_ids.to(device=sample_ids.device, dtype=torch.long).view(-1)
    if int(keep_sids.numel()) == int(sample_ids.numel()) and torch.equal(keep_sids, sample_ids):
        return keep_pos, keep_lens, keep_force_dense
    row_map = {int(s): i for i, s in enumerate(keep_sids.tolist())}
    rows = []
    for sid in sample_ids.tolist():
        sid = int(sid)
        if sid not in row_map:
            return None, None, keep_force_dense
        rows.append(int(row_map[sid]))
    rows_t = torch.tensor(rows, device=keep_pos.device, dtype=torch.long)
    return keep_pos.index_select(0, rows_t), keep_lens.index_select(0, rows_t), keep_force_dense




def _build_keep_plan_from_state_guidance(
    sample_ids: torch.Tensor,
    guidance: torch.Tensor,
    pos: torch.Tensor,
    lens: torch.Tensor,
    current_abs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool, float, int, int, int]:
    valid_mask = _valid_mask_from_state(pos, lens)
    score = _ensure_guidance_2d(guidance, pos.shape[0], pos.shape[1], pos.device)
    keep_mask, _ = _vectorized_select_keep_mask(score, pos, valid_mask, current_abs)
    keep_pos, keep_lens = _compress_selected_pos(pos, keep_mask)
    total_tokens = int(valid_mask.sum().item())
    kept_tokens = int(keep_mask.sum().item())
    dropped_tokens = max(total_tokens - kept_tokens, 0)
    drop_ratio = float(dropped_tokens) / max(total_tokens, 1)
    force_dense = False
    min_kv_len = int(getattr(_SKVS_CFG, "min_kv_len", 0) or 0)
    min_sparse_drop_ratio = float(getattr(_SKVS_CFG, "min_sparse_drop_ratio", 0.0) or 0.0)
    max_kv_len = int(lens.max().item()) if lens.numel() > 0 else 0
    if min_kv_len > 0 and max_kv_len < min_kv_len:
        force_dense = True
    if min_sparse_drop_ratio > 0.0 and drop_ratio < min_sparse_drop_ratio:
        force_dense = True
    _store_keep_plan_for_sample_ids(
        sample_ids=sample_ids,
        keep_pos=keep_pos,
        keep_lens=keep_lens,
        kv_len=max_kv_len,
        force_dense=force_dense,
        total_tokens=total_tokens,
        kept_tokens=kept_tokens,
        dropped_tokens=dropped_tokens,
    )
    return keep_pos, keep_lens, force_dense, drop_ratio, total_tokens, kept_tokens, dropped_tokens




def _run_state_attention_with_guidance_gqa(
    self,
    query_states: torch.Tensor,   # [B,H,Q,D]
    state: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = state["k"]
    v = state["v"]
    lens = state["lens"]
    bsz, kv_heads, kmax, head_dim = k.shape
    num_heads = int(query_states.shape[1])
    q_len = int(query_states.shape[2])
    groups = max(int(num_heads // max(kv_heads, 1)), 1)
    q = query_states.view(bsz, kv_heads, groups, q_len, head_dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q, k) / math.sqrt(head_dim)
    valid = torch.arange(kmax, device=k.device).view(1, 1, 1, 1, kmax) < lens.view(bsz, 1, 1, 1, 1)
    neg_large = torch.finfo(scores.dtype).min / 2 if scores.dtype.is_floating_point else -1e30
    scores = scores.masked_fill(~valid, neg_large)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.einsum("bhgqk,bhkd->bhgqd", probs.to(q.dtype), v)
    out = out.reshape(bsz, num_heads, q_len, head_dim).contiguous()
    guidance = probs.mean(dim=(1, 2, 3)).contiguous()
    return out, guidance




def _run_prefill_attention_with_guidance_gqa(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, kv_heads, kv_len, head_dim = key_states.shape
    num_heads = int(query_states.shape[1])
    q_len = int(query_states.shape[2])
    groups = max(int(num_heads // max(kv_heads, 1)), 1)
    q = query_states.view(bsz, kv_heads, groups, q_len, head_dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q, key_states) / math.sqrt(head_dim)
    if attention_mask is not None:
        mask = attention_mask[:, :, :q_len, :kv_len].unsqueeze(2)
        scores = scores + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.einsum("bhgqk,bhkd->bhgqd", probs.to(q.dtype), value_states)
    out = out.reshape(bsz, num_heads, q_len, head_dim).contiguous()
    guidance = probs.mean(dim=(1, 2, 3)).contiguous()
    return out, guidance




def _cpu_serialize_state_obj(state: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if state is None:
        return None
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to("cpu")
        else:
            out[k] = v
    return out




def _device_deserialize_state_obj(state: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if state is None:
        return None
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out





def _pad_state_cpu_for_concat(state: Dict[str, torch.Tensor], target_k: int) -> Dict[str, torch.Tensor]:
    if state is None:
        return state
    cur_k = int(state["pos"].shape[1]) if torch.is_tensor(state.get("pos", None)) and state["pos"].dim() >= 2 else 0
    if cur_k >= int(target_k):
        return state
    pad_k = int(target_k) - cur_k
    out = dict(state)
    if torch.is_tensor(out.get("pos", None)) and out["pos"].dim() == 2:
        pos_pad = torch.full((out["pos"].shape[0], pad_k), -1, dtype=out["pos"].dtype)
        out["pos"] = torch.cat([out["pos"], pos_pad], dim=1)
    if torch.is_tensor(out.get("k", None)) and out["k"].dim() == 4:
        k_pad = torch.zeros((out["k"].shape[0], out["k"].shape[1], pad_k, out["k"].shape[3]), dtype=out["k"].dtype)
        out["k"] = torch.cat([out["k"], k_pad], dim=2)
    if torch.is_tensor(out.get("v", None)) and out["v"].dim() == 4:
        v_pad = torch.zeros((out["v"].shape[0], out["v"].shape[1], pad_k, out["v"].shape[3]), dtype=out["v"].dtype)
        out["v"] = torch.cat([out["v"], v_pad], dim=2)
    return out


def _pad_keep_cpu_for_concat(state: Dict[str, torch.Tensor], target_k: int) -> Dict[str, torch.Tensor]:
    if state is None:
        return state
    cur_k = int(state["pos"].shape[1]) if torch.is_tensor(state.get("pos", None)) and state["pos"].dim() >= 2 else 0
    if cur_k >= int(target_k):
        return state
    pad_k = int(target_k) - cur_k
    out = dict(state)
    if torch.is_tensor(out.get("pos", None)) and out["pos"].dim() == 2:
        pos_pad = torch.full((out["pos"].shape[0], pad_k), -1, dtype=out["pos"].dtype)
        out["pos"] = torch.cat([out["pos"], pos_pad], dim=1)
    return out


def _migrate_cpu_state_by_owner(
    gathered_states,
    new_owner_ranks: torch.Tensor,
    my_rank: int,
):
    merged: Dict[int, Dict[str, torch.Tensor]] = {}
    for payload in gathered_states:
        if not payload:
            continue
        for layer_idx_raw, state in payload.items():
            if state is None:
                continue
            layer_idx = int(layer_idx_raw)
            if layer_idx not in merged:
                merged[layer_idx] = state
            else:
                cur = merged[layer_idx]
                cur_sids = cur["sample_ids"].tolist()
                state_sids = state["sample_ids"].tolist()
                row_map = {int(s): i for i, s in enumerate(cur_sids)}
                append_rows = [i for i, s in enumerate(state_sids) if int(s) not in row_map]
                if append_rows:
                    rows = torch.tensor(append_rows, dtype=torch.long)
                    add_state = {
                        "sample_ids": state["sample_ids"].index_select(0, rows),
                        "pos": state["pos"].index_select(0, rows),
                        "k": state["k"].index_select(0, rows),
                        "v": state["v"].index_select(0, rows),
                        "lens": state["lens"].index_select(0, rows),
                    }
                    target_k = max(int(cur["pos"].shape[1]), int(add_state["pos"].shape[1]))
                    cur = _pad_state_cpu_for_concat(cur, target_k)
                    add_state = _pad_state_cpu_for_concat(add_state, target_k)
                    cur["sample_ids"] = torch.cat([cur["sample_ids"], add_state["sample_ids"]], dim=0)
                    cur["pos"] = torch.cat([cur["pos"], add_state["pos"]], dim=0)
                    cur["k"] = torch.cat([cur["k"], add_state["k"]], dim=0)
                    cur["v"] = torch.cat([cur["v"], add_state["v"]], dim=0)
                    cur["lens"] = torch.cat([cur["lens"], add_state["lens"]], dim=0)
                    merged[layer_idx] = cur

    target_sids = torch.nonzero(new_owner_ranks.to(dtype=torch.long, device=new_owner_ranks.device) == int(my_rank), as_tuple=False).flatten().to("cpu")
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    moved_selected_bytes = 0
    moved_full_bytes = 0
    for layer_idx, state in merged.items():
        sid_map = {int(s): i for i, s in enumerate(state["sample_ids"].tolist())}
        rows = [sid_map[int(s)] for s in target_sids.tolist() if int(s) in sid_map]
        if not rows:
            continue
        rows_t = torch.tensor(rows, dtype=torch.long)
        state_local = {
            "sample_ids": state["sample_ids"].index_select(0, rows_t).contiguous(),
            "pos": state["pos"].index_select(0, rows_t).contiguous(),
            "k": state["k"].index_select(0, rows_t).contiguous(),
            "v": state["v"].index_select(0, rows_t).contiguous(),
            "lens": state["lens"].index_select(0, rows_t).contiguous(),
        }
        out[int(layer_idx)] = state_local
        selected_bytes = int(state_local["k"].numel() * state_local["k"].element_size()) + int(state_local["v"].numel() * state_local["v"].element_size())
        full_bytes = int(state["k"].numel() * state["k"].element_size()) + int(state["v"].numel() * state["v"].element_size())
        moved_selected_bytes += selected_bytes
        moved_full_bytes += full_bytes
    return out, moved_selected_bytes, moved_full_bytes


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    if dtype is torch.float32:
        return 3
    return 1


def _code_to_dtype(code: int) -> torch.dtype:
    if int(code) == 2:
        return torch.bfloat16
    if int(code) == 3:
        return torch.float32
    return torch.float16


def _all_gather_1d_tensor(x: torch.Tensor, group=None) -> torch.Tensor:
    world = dist.get_world_size(group=group)
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.stack(gathered, dim=0)


def _migrate_sparse_cache_tensor_path(
    sparse_cache: Dict[int, Dict[str, torch.Tensor]],
    new_owner_ranks: torch.Tensor,
    my_rank: int,
    device: torch.device,
    group=None,
    selected_layers: Optional[set[int]] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], int, int, int, int]:
    world = dist.get_world_size(group=group)
    layers = sorted(selected_layers if selected_layers is not None else set(int(k) for k in sparse_cache.keys()))
    if selected_layers is None:
        local_layers_t = torch.tensor(sorted(int(k) for k in sparse_cache.keys()), device=device, dtype=torch.long)
        count_t = torch.tensor([int(local_layers_t.numel())], device=device, dtype=torch.long)
        counts = _all_gather_1d_tensor(count_t, group=group).view(-1).tolist()
        max_count = max([int(c) for c in counts] + [0])
        padded = torch.full((max_count,), -1, device=device, dtype=torch.long)
        if int(local_layers_t.numel()) > 0:
            padded[: int(local_layers_t.numel())] = local_layers_t
        gathered_layers = _all_gather_1d_tensor(padded, group=group)
        layer_set = set()
        for r in range(world):
            n = int(counts[r])
            if n > 0:
                layer_set.update(int(x) for x in gathered_layers[r, :n].tolist() if int(x) >= 0)
        layers = sorted(layer_set)

    target_sids = torch.nonzero(
        new_owner_ranks.to(device=device, dtype=torch.long) == int(my_rank),
        as_tuple=False,
    ).flatten()
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    moved_selected_bytes = 0
    moved_full_bytes = 0
    tensor_payload_bytes = 0
    tensor_wire_bytes = 0

    for layer_idx in layers:
        state = sparse_cache.get(int(layer_idx), None)
        has = 1 if state is not None else 0
        if has:
            sample_ids = state["sample_ids"].to(device=device, dtype=torch.long).view(-1)
            pos = state["pos"].to(device=device, dtype=torch.long).contiguous()
            lens = state["lens"].to(device=device, dtype=torch.long).view(-1).contiguous()
            k = state["k"].to(device=device).contiguous()
            v = state["v"].to(device=device).contiguous()
            bsz = int(sample_ids.numel())
            kmax = int(pos.shape[1]) if pos.dim() == 2 else 0
            kv_heads = int(k.shape[1]) if k.dim() == 4 else 0
            head_dim = int(k.shape[3]) if k.dim() == 4 else 0
            dtype_code = _dtype_to_code(k.dtype)
        else:
            sample_ids = torch.empty((0,), device=device, dtype=torch.long)
            pos = torch.empty((0, 0), device=device, dtype=torch.long)
            lens = torch.empty((0,), device=device, dtype=torch.long)
            k = torch.empty((0, 0, 0, 0), device=device, dtype=torch.float16)
            v = torch.empty((0, 0, 0, 0), device=device, dtype=torch.float16)
            bsz = kmax = kv_heads = head_dim = dtype_code = 0

        meta = torch.tensor([has, bsz, kmax, kv_heads, head_dim, dtype_code], device=device, dtype=torch.long)
        metas = _all_gather_1d_tensor(meta, group=group)
        has_any = bool((metas[:, 0] > 0).any().item())
        if not has_any:
            continue

        max_b = int(metas[:, 1].max().item())
        max_k = int(metas[:, 2].max().item())
        max_h = int(metas[:, 3].max().item())
        max_d = int(metas[:, 4].max().item())
        dtype_codes = [int(x.item()) for x in metas[:, 5] if int(x.item()) > 0]
        dtype = _code_to_dtype(dtype_codes[0] if dtype_codes else dtype_code)

        ids_pad = torch.full((max_b,), -1, device=device, dtype=torch.long)
        lens_pad = torch.zeros((max_b,), device=device, dtype=torch.long)
        pos_pad = torch.full((max_b, max_k), -1, device=device, dtype=torch.long)
        k_pad = torch.zeros((max_b, max_h, max_k, max_d), device=device, dtype=dtype)
        v_pad = torch.zeros((max_b, max_h, max_k, max_d), device=device, dtype=dtype)
        if has and bsz > 0:
            ids_pad[:bsz] = sample_ids
            lens_pad[:bsz] = lens
            pos_pad[:bsz, :kmax] = pos[:, :kmax]
            k_pad[:bsz, :kv_heads, :kmax, :head_dim] = k.to(dtype=dtype)
            v_pad[:bsz, :kv_heads, :kmax, :head_dim] = v.to(dtype=dtype)

        gathered_ids = [torch.empty_like(ids_pad) for _ in range(world)]
        gathered_lens = [torch.empty_like(lens_pad) for _ in range(world)]
        gathered_pos = [torch.empty_like(pos_pad) for _ in range(world)]
        gathered_k = [torch.empty_like(k_pad) for _ in range(world)]
        gathered_v = [torch.empty_like(v_pad) for _ in range(world)]
        dist.all_gather(gathered_ids, ids_pad, group=group)
        dist.all_gather(gathered_lens, lens_pad, group=group)
        dist.all_gather(gathered_pos, pos_pad, group=group)
        dist.all_gather(gathered_k, k_pad, group=group)
        dist.all_gather(gathered_v, v_pad, group=group)

        local_payload = int(ids_pad.numel() * ids_pad.element_size())
        local_payload += int(lens_pad.numel() * lens_pad.element_size())
        local_payload += int(pos_pad.numel() * pos_pad.element_size())
        local_payload += int(k_pad.numel() * k_pad.element_size()) + int(v_pad.numel() * v_pad.element_size())
        tensor_payload_bytes += local_payload
        tensor_wire_bytes += local_payload * max(world - 1, 1)

        selected_rows = []
        for sid in target_sids.tolist():
            sid_i = int(sid)
            for r in range(world):
                n_r = int(metas[r, 1].item())
                if n_r <= 0:
                    continue
                matches = torch.nonzero(gathered_ids[r][:n_r] == sid_i, as_tuple=False).flatten()
                if int(matches.numel()) > 0:
                    row = int(matches[0].item())
                    selected_rows.append((r, row))
                    break

        if not selected_rows:
            continue

        out_ids = torch.empty((len(selected_rows),), device=device, dtype=torch.long)
        out_lens = torch.empty((len(selected_rows),), device=device, dtype=torch.long)
        out_pos = torch.empty((len(selected_rows), max_k), device=device, dtype=torch.long)
        out_k = torch.empty((len(selected_rows), max_h, max_k, max_d), device=device, dtype=dtype)
        out_v = torch.empty((len(selected_rows), max_h, max_k, max_d), device=device, dtype=dtype)
        for i, (r, row) in enumerate(selected_rows):
            out_ids[i] = gathered_ids[r][row]
            out_lens[i] = gathered_lens[r][row]
            out_pos[i] = gathered_pos[r][row]
            out_k[i] = gathered_k[r][row]
            out_v[i] = gathered_v[r][row]
        out[int(layer_idx)] = {
            "sample_ids": out_ids.contiguous(),
            "pos": out_pos.contiguous(),
            "k": out_k.contiguous(),
            "v": out_v.contiguous(),
            "lens": out_lens.contiguous(),
        }
        moved_selected_bytes += int(out_k.numel() * out_k.element_size()) + int(out_v.numel() * out_v.element_size())
        moved_full_bytes += sum(
            int(metas[r, 1].item()) * max_h * max_k * max_d * torch.tensor([], dtype=dtype).element_size() * 2
            for r in range(world)
        )

    return out, moved_selected_bytes, moved_full_bytes, tensor_payload_bytes, tensor_wire_bytes


def _migrate_keep_plan_tensor_path(
    new_owner_ranks: torch.Tensor,
    my_rank: int,
    device: torch.device,
    group=None,
) -> Tuple[int, int]:
    world = dist.get_world_size(group=group)
    keep_sids = GLOBAL_ATTN_STATE.get("global_keep_sample_ids", None)
    keep_pos = GLOBAL_ATTN_STATE.get("global_keep_pos", None)
    keep_lens = GLOBAL_ATTN_STATE.get("global_keep_lens", None)
    has = 1 if keep_sids is not None and keep_pos is not None and keep_lens is not None else 0
    if has:
        sids = keep_sids.to(device=device, dtype=torch.long).view(-1).contiguous()
        pos = keep_pos.to(device=device, dtype=torch.long).contiguous()
        lens = keep_lens.to(device=device, dtype=torch.long).view(-1).contiguous()
        bsz = int(sids.numel())
        kmax = int(pos.shape[1]) if pos.dim() == 2 else 0
    else:
        sids = torch.empty((0,), device=device, dtype=torch.long)
        pos = torch.empty((0, 0), device=device, dtype=torch.long)
        lens = torch.empty((0,), device=device, dtype=torch.long)
        bsz = kmax = 0

    meta = torch.tensor([has, bsz, kmax], device=device, dtype=torch.long)
    metas = _all_gather_1d_tensor(meta, group=group)
    if not bool((metas[:, 0] > 0).any().item()):
        GLOBAL_ATTN_STATE["global_keep_sample_ids"] = None
        GLOBAL_ATTN_STATE["global_keep_pos"] = None
        GLOBAL_ATTN_STATE["global_keep_lens"] = None
        return 0, 0

    max_b = int(metas[:, 1].max().item())
    max_k = int(metas[:, 2].max().item())
    ids_pad = torch.full((max_b,), -1, device=device, dtype=torch.long)
    lens_pad = torch.zeros((max_b,), device=device, dtype=torch.long)
    pos_pad = torch.full((max_b, max_k), -1, device=device, dtype=torch.long)
    if has and bsz > 0:
        ids_pad[:bsz] = sids
        lens_pad[:bsz] = lens
        pos_pad[:bsz, :kmax] = pos[:, :kmax]

    gathered_ids = [torch.empty_like(ids_pad) for _ in range(world)]
    gathered_lens = [torch.empty_like(lens_pad) for _ in range(world)]
    gathered_pos = [torch.empty_like(pos_pad) for _ in range(world)]
    dist.all_gather(gathered_ids, ids_pad, group=group)
    dist.all_gather(gathered_lens, lens_pad, group=group)
    dist.all_gather(gathered_pos, pos_pad, group=group)
    local_payload = int(ids_pad.numel() * ids_pad.element_size())
    local_payload += int(lens_pad.numel() * lens_pad.element_size())
    local_payload += int(pos_pad.numel() * pos_pad.element_size())

    target_sids = torch.nonzero(
        new_owner_ranks.to(device=device, dtype=torch.long) == int(my_rank),
        as_tuple=False,
    ).flatten()
    selected = []
    for sid in target_sids.tolist():
        sid_i = int(sid)
        for r in range(world):
            n_r = int(metas[r, 1].item())
            if n_r <= 0:
                continue
            matches = torch.nonzero(gathered_ids[r][:n_r] == sid_i, as_tuple=False).flatten()
            if int(matches.numel()) > 0:
                selected.append((r, int(matches[0].item())))
                break

    if selected:
        out_ids = torch.empty((len(selected),), device=device, dtype=torch.long)
        out_lens = torch.empty((len(selected),), device=device, dtype=torch.long)
        out_pos = torch.empty((len(selected), max_k), device=device, dtype=torch.long)
        for i, (r, row) in enumerate(selected):
            out_ids[i] = gathered_ids[r][row]
            out_lens[i] = gathered_lens[r][row]
            out_pos[i] = gathered_pos[r][row]
        GLOBAL_ATTN_STATE["global_keep_sample_ids"] = out_ids.contiguous()
        GLOBAL_ATTN_STATE["global_keep_pos"] = out_pos.contiguous()
        GLOBAL_ATTN_STATE["global_keep_lens"] = out_lens.contiguous()
    else:
        GLOBAL_ATTN_STATE["global_keep_sample_ids"] = None
        GLOBAL_ATTN_STATE["global_keep_pos"] = None
        GLOBAL_ATTN_STATE["global_keep_lens"] = None
    return local_payload, local_payload * max(world - 1, 1)


def migrate_sparse_kv_cache_for_group_change(
    prev_owner_ranks: torch.Tensor,
    new_owner_ranks: torch.Tensor,
    my_rank: int,
    group=None,
    layer_indices: Optional[Tuple[int, ...]] = None,
    migrate_sparse_cache: bool = True,
    migrate_keep_plan: bool = True,
) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if prev_owner_ranks is None or new_owner_ranks is None:
        return
    if int(prev_owner_ranks.numel()) != int(new_owner_ranks.numel()):
        raise RuntimeError("owner-rank migration shape mismatch")

    changed_local = 1 if bool(torch.any(prev_owner_ranks.to(dtype=torch.long) != new_owner_ranks.to(dtype=torch.long)).item()) else 0
    changed_tensor = new_owner_ranks.new_tensor([changed_local], dtype=torch.int32)
    dist.all_reduce(changed_tensor, op=dist.ReduceOp.MAX, group=group)
    if int(changed_tensor.item()) == 0:
        return

    device = None
    sparse_cache: Dict[int, Dict[str, torch.Tensor]] = GLOBAL_ATTN_STATE.setdefault("sparse_kv_cache", {})
    for state in sparse_cache.values():
        if torch.is_tensor(state.get("pos", None)):
            device = state["pos"].device
            break
    if device is None:
        device = new_owner_ranks.device

    selected_layers = None if layer_indices is None else {int(x) for x in layer_indices}

    moved_selected_bytes = 0
    moved_full_bytes = 0
    tensor_payload_bytes = 0
    tensor_wire_bytes = 0
    migration_impl = str(os.environ.get("SPARSE_SKVS_MIGRATION_IMPL", "tensor")).strip().lower()
    use_tensor_migration = migration_impl not in ("object", "objects", "pickle", "python")

    if migrate_sparse_cache:
        if use_tensor_migration:
            new_local_tensor, moved_selected_bytes, moved_full_bytes, payload_b, wire_b = _migrate_sparse_cache_tensor_path(
                sparse_cache=sparse_cache,
                new_owner_ranks=new_owner_ranks,
                my_rank=int(my_rank),
                device=device,
                group=group,
                selected_layers=selected_layers,
            )
            tensor_payload_bytes += int(payload_b)
            tensor_wire_bytes += int(wire_b)
            if selected_layers is None:
                GLOBAL_ATTN_STATE["sparse_kv_cache"] = new_local_tensor
            else:
                remaining = {int(k): v for k, v in sparse_cache.items() if int(k) not in selected_layers}
                remaining.update(new_local_tensor)
                GLOBAL_ATTN_STATE["sparse_kv_cache"] = remaining
            _update_sparse_cache_accounting_full()
        else:
            if selected_layers is None:
                payload = {int(k): _cpu_serialize_state_obj(v) for k, v in sparse_cache.items()}
            else:
                payload = {int(k): _cpu_serialize_state_obj(v) for k, v in sparse_cache.items() if int(k) in selected_layers}
            world = dist.get_world_size(group=group)
            gathered = [None for _ in range(world)]
            dist.barrier(group=group)
            dist.all_gather_object(gathered, payload, group=group)
            new_local_cpu, moved_selected_bytes, moved_full_bytes = _migrate_cpu_state_by_owner(
                gathered_states=gathered,
                new_owner_ranks=new_owner_ranks.detach().to("cpu"),
                my_rank=int(my_rank),
            )
            if selected_layers is None:
                GLOBAL_ATTN_STATE["sparse_kv_cache"] = {int(k): _device_deserialize_state_obj(v, device=device) for k, v in new_local_cpu.items()}
            else:
                remaining = {int(k): v for k, v in sparse_cache.items() if int(k) not in selected_layers}
                migrated = {int(k): _device_deserialize_state_obj(v, device=device) for k, v in new_local_cpu.items()}
                remaining.update(migrated)
                GLOBAL_ATTN_STATE["sparse_kv_cache"] = remaining
            _update_sparse_cache_accounting_full()

    if migrate_keep_plan:
        if use_tensor_migration:
            payload_b, wire_b = _migrate_keep_plan_tensor_path(
                new_owner_ranks=new_owner_ranks,
                my_rank=int(my_rank),
                device=device,
                group=group,
            )
            tensor_payload_bytes += int(payload_b)
            tensor_wire_bytes += int(wire_b)
        else:
            keep_sids = GLOBAL_ATTN_STATE.get("global_keep_sample_ids", None)
            keep_pos = GLOBAL_ATTN_STATE.get("global_keep_pos", None)
            keep_lens = GLOBAL_ATTN_STATE.get("global_keep_lens", None)
            keep_obj = None
            if keep_sids is not None and keep_pos is not None and keep_lens is not None:
                keep_obj = _cpu_serialize_state_obj({
                    "sample_ids": keep_sids,
                    "pos": keep_pos,
                    "lens": keep_lens,
                })
            world = dist.get_world_size(group=group)
            keep_gathered = [None for _ in range(world)]
            dist.barrier(group=group)
            dist.all_gather_object(keep_gathered, keep_obj, group=group)
            merged_keep = None
            for obj in keep_gathered:
                if obj is None:
                    continue
                if merged_keep is None:
                    merged_keep = obj
                else:
                    sid_map = {int(s): i for i, s in enumerate(merged_keep["sample_ids"].tolist())}
                    add_rows = [i for i, s in enumerate(obj["sample_ids"].tolist()) if int(s) not in sid_map]
                    if add_rows:
                        rows = torch.tensor(add_rows, dtype=torch.long)
                        add_keep = {
                            "sample_ids": obj["sample_ids"].index_select(0, rows),
                            "pos": obj["pos"].index_select(0, rows),
                            "lens": obj["lens"].index_select(0, rows),
                        }
                        target_k = max(int(merged_keep["pos"].shape[1]), int(add_keep["pos"].shape[1]))
                        merged_keep = _pad_keep_cpu_for_concat(merged_keep, target_k)
                        add_keep = _pad_keep_cpu_for_concat(add_keep, target_k)
                        merged_keep["sample_ids"] = torch.cat([merged_keep["sample_ids"], add_keep["sample_ids"]], dim=0)
                        merged_keep["pos"] = torch.cat([merged_keep["pos"], add_keep["pos"]], dim=0)
                        merged_keep["lens"] = torch.cat([merged_keep["lens"], add_keep["lens"]], dim=0)
            target_sids = torch.nonzero(new_owner_ranks.detach().to("cpu", dtype=torch.long) == int(my_rank), as_tuple=False).flatten()
            if merged_keep is not None:
                sid_map = {int(s): i for i, s in enumerate(merged_keep["sample_ids"].tolist())}
                rows = [sid_map[int(s)] for s in target_sids.tolist() if int(s) in sid_map]
                if rows:
                    rows_t = torch.tensor(rows, dtype=torch.long)
                    GLOBAL_ATTN_STATE["global_keep_sample_ids"] = merged_keep["sample_ids"].index_select(0, rows_t).to(device=device)
                    GLOBAL_ATTN_STATE["global_keep_pos"] = merged_keep["pos"].index_select(0, rows_t).to(device=device)
                    GLOBAL_ATTN_STATE["global_keep_lens"] = merged_keep["lens"].index_select(0, rows_t).to(device=device)
                else:
                    GLOBAL_ATTN_STATE["global_keep_sample_ids"] = None
                    GLOBAL_ATTN_STATE["global_keep_pos"] = None
                    GLOBAL_ATTN_STATE["global_keep_lens"] = None

    GLOBAL_ATTN_STATE["migration_selected_kv_bytes"] = int(GLOBAL_ATTN_STATE.get("migration_selected_kv_bytes", 0) or 0) + int(moved_selected_bytes)
    GLOBAL_ATTN_STATE["migration_full_kv_bytes"] = int(GLOBAL_ATTN_STATE.get("migration_full_kv_bytes", 0) or 0) + int(moved_full_bytes)
    GLOBAL_ATTN_STATE["migration_tensor_payload_bytes"] = int(GLOBAL_ATTN_STATE.get("migration_tensor_payload_bytes", 0) or 0) + int(tensor_payload_bytes)
    GLOBAL_ATTN_STATE["migration_tensor_wire_bytes"] = int(GLOBAL_ATTN_STATE.get("migration_tensor_wire_bytes", 0) or 0) + int(tensor_wire_bytes)
    if use_tensor_migration:
        _inc_state_counter("migration_impl_tensor_count")
    else:
        _inc_state_counter("migration_impl_object_count")
    if moved_full_bytes > 0:
        GLOBAL_ATTN_STATE["migration_saved_ratio"] = 1.0 - float(moved_selected_bytes) / float(moved_full_bytes)

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
    bsz, q_len, _ = hidden_states.size()
    decode_only_flag = bool(getattr(self, "_skvs_decode_only", False))
    allow_prefill_sparse = bool(getattr(self, "_skvs_allow_prefill_sparse", False))
    batch_limit = int(getattr(self, "_skvs_batch_limit", 0))
    batch_ok = (batch_limit <= 0) or (bsz <= batch_limit)
    is_decode_step = bool(use_cache and q_len == 1 and batch_ok)

    if self.layer_idx == 0:
        reset_skvs_forward_ephemeral()
        if is_decode_step:
            GLOBAL_ATTN_STATE["decode_step_idx"] = int(GLOBAL_ATTN_STATE.get("decode_step_idx", 0) or 0) + 1
        else:
            GLOBAL_ATTN_STATE["decode_step_idx"] = 0

    if output_attentions:
        _record_decode_fallback("output_attentions")
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

    if not bool(getattr(self, "_skvs_enabled", True)):
        _record_decode_fallback("skvs_disabled")
        GLOBAL_ATTN_STATE["effective_mode_last"] = "orig_full_skvs_disabled"
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

    if float(_SKVS_CFG.l0_guidance_ratio) >= 0.999999:
        GLOBAL_ATTN_STATE["num_ratio1_full"] = int(GLOBAL_ATTN_STATE.get("num_ratio1_full", 0) or 0) + 1
        GLOBAL_ATTN_STATE["effective_mode_last"] = "orig_full_ratio1"
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

    if (not is_decode_step) and decode_only_flag and (not allow_prefill_sparse):
        _record_prefill_fallback("prefill_decode_only")
        if self.layer_idx == 0:
            GLOBAL_ATTN_STATE["layer0_score"] = None
            GLOBAL_ATTN_STATE["last_guidance_kv_len"] = 0
            _reset_sparse_cache()
            GLOBAL_ATTN_STATE["global_keep_step"] = -1
            GLOBAL_ATTN_STATE["global_keep_pos"] = None
            GLOBAL_ATTN_STATE["global_keep_lens"] = None
            GLOBAL_ATTN_STATE["global_keep_kv_len"] = 0
            GLOBAL_ATTN_STATE["global_keep_force_dense"] = False
            GLOBAL_ATTN_STATE["global_keep_sample_ids"] = None
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

    start_kv_len = _skvs_start_kv_len()
    if is_decode_step and start_kv_len > 0:
        cur_kv_len_est = _estimate_current_kv_len(attention_mask, position_ids, cache_position, q_len)
        if int(cur_kv_len_est) < int(start_kv_len):
            _record_decode_fallback("decode_start_kv_len")
            GLOBAL_ATTN_STATE["effective_mode_last"] = "orig_full_before_skvs_start_kv_len"
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

    # Unified SKVS semantics: once prefill sparse is enabled we keep using explicit
    # sample-aware sparse state for both prefill and decode.
    sample_ids = _sample_ids_for_current_batch(self, bsz=bsz, device=hidden_states.device)
    current_abs_pos = _safe_current_abs_positions(
        position_ids,
        cache_position,
        attention_mask,
        bsz,
        q_len,
        hidden_states.device,
    )
    if is_decode_step:
        current_abs_pos = _correct_decode_abs_positions(
            self=self,
            past_key_value=past_key_value,
            sample_ids=sample_ids,
            current_abs_pos=current_abs_pos,
            device=hidden_states.device,
        )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    rope_seq_len = max(int(current_abs_pos.max().item()) + 1, int(key_states.shape[-2]))
    cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, current_abs_pos)

    # ---------------- prefill: build sparse storage for decode ----------------
    if not is_decode_step:
        if self.layer_idx == 0:
            GLOBAL_ATTN_STATE["layer0_score"] = None
            GLOBAL_ATTN_STATE["last_guidance_kv_len"] = 0
            _reset_sparse_cache()
            GLOBAL_ATTN_STATE["global_keep_step"] = -1
            GLOBAL_ATTN_STATE["global_keep_pos"] = None
            GLOBAL_ATTN_STATE["global_keep_lens"] = None
            GLOBAL_ATTN_STATE["global_keep_kv_len"] = 0
            GLOBAL_ATTN_STATE["global_keep_force_dense"] = False
            GLOBAL_ATTN_STATE["global_keep_sample_ids"] = None

        full_kv_len = int(key_states.shape[-2])
        attn_output, guidance = _run_prefill_attention_with_guidance_gqa(
            self=self,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )

        abs_pos = current_abs_pos.to(device=hidden_states.device, dtype=torch.long)
        dense_state = _dense_state_from_full_sequence(sample_ids, abs_pos, key_states, value_states)

        if self.layer_idx == 0:
            GLOBAL_ATTN_STATE["layer0_score"] = guidance.detach()
            GLOBAL_ATTN_STATE["last_guidance_kv_len"] = int(full_kv_len)
            _store_layer_state(int(self.layer_idx), dense_state)
            _build_keep_plan_from_state_guidance(
                sample_ids=sample_ids,
                guidance=guidance,
                pos=dense_state["pos"],
                lens=dense_state["lens"],
                current_abs=current_abs_pos[:, -1],
            )
            GLOBAL_ATTN_STATE["num_local_full"] = int(GLOBAL_ATTN_STATE.get("num_local_full", 0) or 0) + 1
            GLOBAL_ATTN_STATE["num_layer0_full"] = int(GLOBAL_ATTN_STATE.get("num_layer0_full", 0) or 0) + 1
            GLOBAL_ATTN_STATE["effective_mode_last"] = "prefill_layer0_full_guidance"
            total_tokens = int(dense_state["lens"].sum().item())
            _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=total_tokens, dropped_tokens=0)
            _accumulate_dense_equiv_accounting(full_dense_tokens=total_tokens, kept_tokens=total_tokens)
            _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)
        else:
            keep_pos, keep_lens, keep_force_dense = _get_keep_plan_for_sample_ids(sample_ids)
            if keep_pos is None or keep_lens is None or keep_force_dense:
                _inc_state_counter("fallback_prefill_dense_storage")
                _store_layer_state(int(self.layer_idx), dense_state)
                total_tokens = int(dense_state["lens"].sum().item())
                _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=total_tokens, dropped_tokens=0)
                _accumulate_dense_equiv_accounting(full_dense_tokens=total_tokens, kept_tokens=total_tokens)
                _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)
                GLOBAL_ATTN_STATE["effective_mode_last"] = "prefill_dense_storage"
            else:
                valid_mask = _valid_mask_from_state(dense_state["pos"], dense_state["lens"])
                keep_mask = _membership_keep_mask(dense_state["pos"], valid_mask, keep_pos, keep_lens)
                try:
                    selected = _compress_selected_kv(dense_state["pos"], keep_mask, dense_state["k"], dense_state["v"])
                    selected["sample_ids"] = sample_ids.contiguous()
                    _store_layer_state(int(self.layer_idx), selected)
                    total_tokens = int(valid_mask.sum().item())
                    kept_tokens = int(keep_mask.sum().item())
                    dropped_tokens = max(total_tokens - kept_tokens, 0)
                    dropped_bytes = dropped_tokens * _kv_bytes_per_token(self.num_key_value_heads, self.head_dim, key_states.dtype)
                    _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=kept_tokens, dropped_tokens=dropped_tokens)
                    _accumulate_dense_equiv_accounting(full_dense_tokens=total_tokens, kept_tokens=kept_tokens)
                    _accumulate_accounting(dropped_bytes=dropped_bytes, kept_remote_bytes=0)
                    GLOBAL_ATTN_STATE["effective_mode_last"] = "prefill_sparse_storage"
                except torch.cuda.OutOfMemoryError:
                    _inc_state_counter("fallback_prefill_oom")
                    _store_layer_state(int(self.layer_idx), dense_state)
                    total_tokens = int(dense_state["lens"].sum().item())
                    _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=total_tokens, dropped_tokens=0)
                    _accumulate_dense_equiv_accounting(full_dense_tokens=total_tokens, kept_tokens=total_tokens)
                    _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)
                    GLOBAL_ATTN_STATE["effective_mode_last"] = "prefill_dense_storage_oom_fallback"

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        present_key_value = past_key_value if use_cache else None
        return attn_output, None, present_key_value

    # ---------------- decode ----------------
    state = _get_layer_state_for_sample_ids(int(self.layer_idx), sample_ids)
    if state is None:
        state = _bootstrap_state_from_past_key_value(
            self=self,
            past_key_value=past_key_value,
            sample_ids=sample_ids,
            attention_mask=attention_mask,
            device=hidden_states.device,
            dtype=key_states.dtype,
        )
        if state is None:
            state = _empty_state_for_samples(
                sample_ids=sample_ids,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                device=hidden_states.device,
                dtype=key_states.dtype,
            )
        _record_decode_fallback("decode_bootstrap")
        GLOBAL_ATTN_STATE["effective_mode_last"] = "decode_bootstrap_state_from_cache"

    cur_pos = current_abs_pos[:, -1].to(hidden_states.device, dtype=torch.long)
    try:
        prev_valid = _valid_mask_from_state(state["pos"], state["lens"])
        if prev_valid.numel() > 0 and torch.any(prev_valid):
            invalid_fill = torch.full_like(state["pos"], -1)
            prev_max_pos = torch.where(prev_valid, state["pos"], invalid_fill).max(dim=1).values
            stale_pos = cur_pos <= prev_max_pos
            if torch.any(stale_pos):
                cur_pos = torch.where(stale_pos, prev_max_pos + 1, cur_pos)
                _inc_state_counter("fallback_stale_cache_position")
    except Exception:
        pass
    pos_cand, k_cand, v_cand, valid_mask = _append_to_sparse_state(_state_without_sample_ids(state), cur_pos, key_states, value_states)
    cand_state = {
        "sample_ids": sample_ids.contiguous(),
        "pos": pos_cand.contiguous(),
        "k": k_cand.contiguous(),
        "v": v_cand.contiguous(),
        "lens": (state["lens"] + 1).contiguous(),
    }

    if self.layer_idx == 0:
        refresh_guidance = True
        suppressed_by_start = False
        suppressed_by_interval = False
        every_k = max(int(getattr(_SKVS_CFG, "guidance_every_k", 1) or 1), 1)
        start_kv = max(int(getattr(_SKVS_CFG, "guidance_start_kv_len", 0) or 0), 0)
        max_kv_len = int(cand_state["lens"].max().item()) if cand_state["lens"].numel() > 0 else 0
        if max_kv_len < start_kv:
            refresh_guidance = False
            suppressed_by_start = True
        step_idx = int(GLOBAL_ATTN_STATE.get("decode_step_idx", 0) or 0)
        if every_k > 1 and (step_idx % every_k) != 1:
            refresh_guidance = False
            suppressed_by_interval = True

        attn_output, guidance_new = _run_state_attention_with_guidance_gqa(self=self, query_states=query_states, state=_state_without_sample_ids(cand_state))
        has_cached_guidance = GLOBAL_ATTN_STATE.get("layer0_score", None) is not None
        use_new_guidance = bool(refresh_guidance or (not has_cached_guidance))
        guidance = guidance_new if use_new_guidance else GLOBAL_ATTN_STATE.get("layer0_score", None)
        if use_new_guidance:
            GLOBAL_ATTN_STATE["num_guidance_refresh"] = int(GLOBAL_ATTN_STATE.get("num_guidance_refresh", 0) or 0) + 1
        else:
            GLOBAL_ATTN_STATE["num_guidance_reuse"] = int(GLOBAL_ATTN_STATE.get("num_guidance_reuse", 0) or 0) + 1
        if suppressed_by_start:
            GLOBAL_ATTN_STATE["num_guidance_start_suppressed"] = int(GLOBAL_ATTN_STATE.get("num_guidance_start_suppressed", 0) or 0) + 1
        if suppressed_by_interval:
            GLOBAL_ATTN_STATE["num_guidance_interval_suppressed"] = int(GLOBAL_ATTN_STATE.get("num_guidance_interval_suppressed", 0) or 0) + 1
        GLOBAL_ATTN_STATE["last_guidance_refreshed"] = bool(use_new_guidance)
        GLOBAL_ATTN_STATE["layer0_score"] = guidance.detach() if torch.is_tensor(guidance) else guidance_new.detach()
        GLOBAL_ATTN_STATE["last_guidance_kv_len"] = int(max_kv_len)
        _store_layer_state(int(self.layer_idx), cand_state)
        _maybe_clear_dense_layer_cache(self, past_key_value, bsz, hidden_states.device, key_states.dtype)
        _build_keep_plan_from_state_guidance(
            sample_ids=sample_ids,
            guidance=guidance if torch.is_tensor(guidance) else guidance_new,
            pos=cand_state["pos"],
            lens=cand_state["lens"],
            current_abs=cur_pos,
        )
        GLOBAL_ATTN_STATE["num_local_full"] = int(GLOBAL_ATTN_STATE.get("num_local_full", 0) or 0) + 1
        GLOBAL_ATTN_STATE["num_layer0_full"] = int(GLOBAL_ATTN_STATE.get("num_layer0_full", 0) or 0) + 1
        GLOBAL_ATTN_STATE["effective_mode_last"] = "decode_layer0_full_guidance"
        total_tokens = int(cand_state["lens"].sum().item())
        _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=total_tokens, dropped_tokens=0)
        _accumulate_dense_equiv_accounting(full_dense_tokens=total_tokens, kept_tokens=total_tokens)
        _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        present_key_value = past_key_value if use_cache else None
        return attn_output, None, present_key_value

    keep_step = int(GLOBAL_ATTN_STATE.get("global_keep_step", -1) or -1)
    cur_step = int(GLOBAL_ATTN_STATE.get("decode_step_idx", 0) or 0)
    keep_pos, keep_lens, keep_force_dense = _get_keep_plan_for_sample_ids(sample_ids)
    if keep_step != cur_step or keep_pos is None or keep_lens is None:
        _record_decode_fallback("missing_keep_plan")
        GLOBAL_ATTN_STATE["effective_mode_last"] = "manual_full_missing_global_keep_plan"
        _store_layer_state(int(self.layer_idx), cand_state)
        _maybe_clear_dense_layer_cache(self, past_key_value, bsz, hidden_states.device, key_states.dtype)
        attn_output, _ = _run_sparse_state_attention_gqa(self=self, query_states=query_states, state=_state_without_sample_ids(cand_state))
        total_tokens = int(cand_state["lens"].sum().item())
        _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=total_tokens, dropped_tokens=0)
        dense_total_tokens = int((cur_pos + 1).sum().item())
        _accumulate_dense_equiv_accounting(full_dense_tokens=dense_total_tokens, kept_tokens=total_tokens)
        _accumulate_accounting(dropped_bytes=0, kept_remote_bytes=0)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        present_key_value = past_key_value if use_cache else None
        return attn_output, None, present_key_value

    keep_mask = _membership_keep_mask(cand_state["pos"], valid_mask, keep_pos, keep_lens)
    total_tokens = int(valid_mask.sum().item())
    kept_tokens = int(keep_mask.sum().item())
    dropped_tokens = max(total_tokens - kept_tokens, 0)
    stats = _selection_stats_from_keep_mask(cand_state["pos"], valid_mask, keep_mask, cur_pos)
    for b in range(bsz):
        _accumulate_local_selection_stats(
            cached_tokens=int(stats["cached"][b].item()),
            protected_tokens=int(stats["protected"][b].item()),
            unprotected_tokens=int(stats["unprotected"][b].item()),
            kept_unprotected_tokens=int(stats["kept_unprotected"][b].item()),
        )

    do_compact = (not keep_force_dense) and _compact_this_step(cur_step)

    if keep_force_dense:
        _store_layer_state(int(self.layer_idx), cand_state)
        _maybe_clear_dense_layer_cache(self, past_key_value, bsz, hidden_states.device, key_states.dtype)
        exec_state = cand_state
        exec_kept_tokens = total_tokens
        exec_dropped_tokens = 0
        dropped_bytes = 0
        _inc_state_counter("num_force_dense_append_only")
        _inc_state_counter("fallback_force_dense_append_only")
        GLOBAL_ATTN_STATE["effective_mode_last"] = "sparse_storage_append_only_shared"
    elif not do_compact:
        _store_layer_state(int(self.layer_idx), cand_state)
        _maybe_clear_dense_layer_cache(self, past_key_value, bsz, hidden_states.device, key_states.dtype)
        exec_state = cand_state
        exec_kept_tokens = total_tokens
        exec_dropped_tokens = 0
        dropped_bytes = 0
        _inc_state_counter("fallback_periodic_append_only")
        GLOBAL_ATTN_STATE["effective_mode_last"] = "sparse_storage_periodic_append_only"
    else:
        try:
            selected = _compress_selected_kv(cand_state["pos"], keep_mask, cand_state["k"], cand_state["v"])
            selected["sample_ids"] = sample_ids.contiguous()
            _store_layer_state(int(self.layer_idx), selected)
            _maybe_clear_dense_layer_cache(self, past_key_value, bsz, hidden_states.device, key_states.dtype)
            exec_state = selected
            exec_kept_tokens = kept_tokens
            exec_dropped_tokens = dropped_tokens
            dropped_bytes = dropped_tokens * _kv_bytes_per_token(self.num_key_value_heads, self.head_dim, key_states.dtype)
            GLOBAL_ATTN_STATE["effective_mode_last"] = "sparse_storage_selected_shared"
        except torch.cuda.OutOfMemoryError:
            # Correctness-first fallback: keep the append-only state for this step.
            exec_state = cand_state
            _store_layer_state(int(self.layer_idx), cand_state)
            _maybe_clear_dense_layer_cache(self, past_key_value, bsz, hidden_states.device, key_states.dtype)
            exec_kept_tokens = total_tokens
            exec_dropped_tokens = 0
            dropped_bytes = 0
            _inc_state_counter("num_force_dense_append_only")
            _inc_state_counter("fallback_sparse_oom")
            GLOBAL_ATTN_STATE["effective_mode_last"] = "sparse_storage_selected_oom_append_only"

    attn_output, _ = _run_sparse_state_attention_gqa(self=self, query_states=query_states, state=_state_without_sample_ids(exec_state))
    GLOBAL_ATTN_STATE["num_local_sparse"] = int(GLOBAL_ATTN_STATE.get("num_local_sparse", 0) or 0) + 1
    _accumulate_token_accounting(total_tokens=total_tokens, kept_tokens=exec_kept_tokens, dropped_tokens=exec_dropped_tokens)
    dense_total_tokens = int((cur_pos + 1).sum().item())
    _accumulate_dense_equiv_accounting(full_dense_tokens=dense_total_tokens, kept_tokens=exec_kept_tokens)
    _accumulate_accounting(dropped_bytes=dropped_bytes, kept_remote_bytes=0)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    present_key_value = past_key_value if use_cache else None
    return attn_output, None, present_key_value
