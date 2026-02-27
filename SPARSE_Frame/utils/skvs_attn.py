from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache


@dataclass
class SKVSConfig:
    h2o_sink: int = 4
    h2o_window: int = 32
    l0_guidance_ratio: float = 0.6


GLOBAL_ATTN_STATE = {
    # [B, L_kv] guidance score produced by layer 0
    "layer0_score": None,
    # per forward call (per layer) accounting only; must be aggregated outside if needed
    "step_dropped_bytes": 0,
}

_SKVS_CFG = SKVSConfig()


def set_skvs_config(cfg: SKVSConfig) -> None:
    global _SKVS_CFG
    _SKVS_CFG = cfg


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, n_kv * n_rep, slen, head_dim)


def _kv_bytes_per_token(num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    # K and V
    return 2 * num_kv_heads * head_dim * torch.tensor([], dtype=dtype).element_size()


def custom_skvs_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """
    A drop-in replacement for HF attention forward.

    Notes:
    - This function reduces *attention compute* by slicing the K/V used in SDPA.
    - It does NOT by itself implement cross-node KV transmission. To evaluate
      communication savings, you must connect the slice decision to your data plane.
    """
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    # QKV
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # RoPE
    kv_seq_len = key_states.shape[-2]
    if past_key_values is not None:
        kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    else:
        cos, sin = position_embeddings

    if cos.device != query_states.device:
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Update cache (full cache is kept; slicing affects only compute)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    full_kv_len = key_states.shape[-2]

    # Reset accounting for this forward
    GLOBAL_ATTN_STATE["step_dropped_bytes"] = 0

    if self.layer_idx == 0:
        # Layer 0 must compute a score to guide slicing
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        with torch.no_grad():
            attn_scores = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask[:, :, :, :full_kv_len]
            probs = torch.softmax(attn_scores, dim=-1)
            # guidance based on the last query position
            # shape: [B, L_kv]
            guidance = probs[:, :, -1, :].max(dim=1).values
            GLOBAL_ATTN_STATE["layer0_score"] = guidance.detach()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states_rep,
            value_states_rep,
            attn_mask=attention_mask[:, :, :, :full_kv_len] if attention_mask is not None else None,
            dropout_p=0.0,
            is_causal=False,
        )
    else:
        l0_score = GLOBAL_ATTN_STATE.get("layer0_score")

        sink_len = min(_SKVS_CFG.h2o_sink, full_kv_len)
        window_len = min(_SKVS_CFG.h2o_window, full_kv_len)

        do_slice = (l0_score is not None) and (full_kv_len > (sink_len + window_len))

        if do_slice:
            num_unprotected = full_kv_len - sink_len - window_len
            topk = int(num_unprotected * float(_SKVS_CFG.l0_guidance_ratio))

            if l0_score.shape[-1] < full_kv_len:
                # decode grows
                l0_score = F.pad(l0_score, (0, full_kv_len - l0_score.shape[-1]), value=1.0)

            if topk > 0:
                idx = torch.arange(full_kv_len, device=device).unsqueeze(0).expand(bsz, -1)
                is_sink = idx < sink_len
                is_window = idx >= (full_kv_len - window_len)
                is_protected = is_sink | is_window

                scores = l0_score.clone()
                scores[is_protected] = -float("inf")
                _, topk_idx = torch.topk(scores, k=topk, dim=-1)

                sink_idx = torch.arange(sink_len, device=device).unsqueeze(0).expand(bsz, -1)
                window_idx = torch.arange(full_kv_len - window_len, full_kv_len, device=device).unsqueeze(0).expand(bsz, -1)

                keep_idx = torch.cat([sink_idx, topk_idx, window_idx], dim=-1)
                keep_idx, _ = torch.sort(keep_idx, dim=-1)

                kept_len = keep_idx.shape[1]
                gather_idx = keep_idx.view(bsz, 1, kept_len, 1).expand(-1, self.num_key_value_heads, -1, self.head_dim)
                sliced_k = torch.gather(key_states, dim=2, index=gather_idx)
                sliced_v = torch.gather(value_states, dim=2, index=gather_idx)

                sliced_mask = None
                if attention_mask is not None:
                    mask_idx = keep_idx.view(bsz, 1, 1, kept_len).expand(-1, 1, q_len, -1)
                    sliced_mask = torch.gather(attention_mask, dim=-1, index=mask_idx)

                dropped = full_kv_len - kept_len
                GLOBAL_ATTN_STATE["step_dropped_bytes"] = dropped * _kv_bytes_per_token(
                    num_kv_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    dtype=key_states.dtype,
                )

                sliced_k_rep = repeat_kv(sliced_k, self.num_key_value_groups)
                sliced_v_rep = repeat_kv(sliced_v, self.num_key_value_groups)

                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    sliced_k_rep,
                    sliced_v_rep,
                    attn_mask=sliced_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                # keep only sink + window
                keep_len = sink_len + window_len
                keep_idx = torch.cat(
                    [
                        torch.arange(sink_len, device=device),
                        torch.arange(full_kv_len - window_len, full_kv_len, device=device),
                    ],
                    dim=0,
                ).unsqueeze(0).expand(bsz, -1)
                keep_idx, _ = torch.sort(keep_idx, dim=-1)

                gather_idx = keep_idx.view(bsz, 1, keep_len, 1).expand(-1, self.num_key_value_heads, -1, self.head_dim)
                sliced_k = torch.gather(key_states, dim=2, index=gather_idx)
                sliced_v = torch.gather(value_states, dim=2, index=gather_idx)

                sliced_mask = None
                if attention_mask is not None:
                    mask_idx = keep_idx.view(bsz, 1, 1, keep_len).expand(-1, 1, q_len, -1)
                    sliced_mask = torch.gather(attention_mask, dim=-1, index=mask_idx)

                dropped = full_kv_len - keep_len
                GLOBAL_ATTN_STATE["step_dropped_bytes"] = dropped * _kv_bytes_per_token(
                    num_kv_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    dtype=key_states.dtype,
                )

                sliced_k_rep = repeat_kv(sliced_k, self.num_key_value_groups)
                sliced_v_rep = repeat_kv(sliced_v, self.num_key_value_groups)

                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    sliced_k_rep,
                    sliced_v_rep,
                    attn_mask=sliced_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
        else:
            key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states_rep,
                value_states_rep,
                attn_mask=attention_mask[:, :, :, :full_kv_len] if attention_mask is not None else None,
                dropout_p=0.0,
                is_causal=False,
            )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None
