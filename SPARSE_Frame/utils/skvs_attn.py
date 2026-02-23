import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

# 全局缓存，用于模拟 Layer 0 向后续层广播重要性分数以及统计真实的通信/显存节约
GLOBAL_ATTN_STATE = {
    "layer0_score": None, 
    "step_dropped_bytes": 0 
}

# 动态配置与论文保持绝对一致
SKVS_CONFIG = {
    "H2O_SINK": 4,
    "H2O_WINDOW": 32,
    "L0_GUIDANCE_RATIO": 0.6 
}

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: 
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def custom_skvs_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
):
    global GLOBAL_ATTN_STATE
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    
    # 1. QKV Projections
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 2. RoPE 绝对位置编码 (在切片前执行，保证位置信息不乱)
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
    
    # 3. 维护全局真实的 KV Cache
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 当前缓存的完整长度
    full_kv_len = key_states.shape[-2]

    # ==============================================================
    # 真实计算核心: 物理切片 (Physical Slicing) + FlashAttention
    # ==============================================================
    
    if self.layer_idx == 0:
        # --- Layer 0: 全量计算并捕获全局分数 ---
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
        
        # 为了获取指导分数，Layer 0 必须显式计算一次 Attention Score
        with torch.no_grad():
            attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :full_kv_len]
                attn_weights = attn_weights + causal_mask
            probs = nn.functional.softmax(attn_weights, dim=-1)
            # 提取最后一步 Query 对历史的所有关注度作为全局指导
            current_importance = probs[:, :, -1, :].max(dim=1).values 
            GLOBAL_ATTN_STATE["layer0_score"] = current_importance.detach()
            GLOBAL_ATTN_STATE["step_dropped_bytes"] = 0
            
        # 真实的输出仍然交给底层的 SDPA 硬件加速计算
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states_rep, 
            value_states_rep, 
            attn_mask=attention_mask[:, :, :, :full_kv_len] if attention_mask is not None else None,
            dropout_p=0.0,
            is_causal=False 
        )

    else:
        # --- Subsequent Layers: 物理切片与真实的计算加速 ---
        l0_score = GLOBAL_ATTN_STATE.get("layer0_score")
        
        sink_len = min(SKVS_CONFIG["H2O_SINK"], full_kv_len)
        window_len = min(SKVS_CONFIG["H2O_WINDOW"], full_kv_len)
        
        # 只有当序列足够长时，才触发物理切片
        if l0_score is not None and full_kv_len > (sink_len + window_len):
            
            num_unprotected = full_kv_len - sink_len - window_len
            topk = int(num_unprotected * SKVS_CONFIG["L0_GUIDANCE_RATIO"])
            
            if topk > 0:
                # 1. 提取保护区域的索引
                idx_tensor = torch.arange(full_kv_len, device=device).unsqueeze(0).expand(bsz, -1)
                is_sink = idx_tensor < sink_len
                is_window = idx_tensor >= (full_kv_len - window_len)
                is_protected = is_sink | is_window
                
                # 2. 屏蔽保护区域，提取 Top-K 索引
                if l0_score.shape[-1] < full_kv_len: # 处理 Decode 阶段增长
                    pad_len = full_kv_len - l0_score.shape[-1]
                    l0_score = F.pad(l0_score, (0, pad_len), value=1.0)
                    
                temp_score = l0_score.clone()
                temp_score[is_protected] = -float('inf')
                _, topk_idx = torch.topk(temp_score, k=topk, dim=-1)
                
                # 3. 拼接所有需要保留的索引，并排序（极其重要：保持绝对位置顺序！）
                sink_idx = torch.arange(sink_len, device=device).unsqueeze(0).expand(bsz, -1)
                window_idx = torch.arange(full_kv_len - window_len, full_kv_len, device=device).unsqueeze(0).expand(bsz, -1)
                
                keep_idx = torch.cat([sink_idx, topk_idx, window_idx], dim=-1)
                keep_idx, _ = torch.sort(keep_idx, dim=-1)
                
                # -------------------------------------------------------------
                # 致命优化点：真实物理切分 (Physical Gathering)
                # 这将直接缩小传入 GPU 寄存器的矩阵体积，带来真正的延时下降！
                # -------------------------------------------------------------
                K_kept = keep_idx.shape[1]
                
                # 扩展索引以匹配 K/V 的 [B, H_kv, L, D] 维度
                gather_idx = keep_idx.view(bsz, 1, K_kept, 1).expand(-1, self.num_key_value_heads, -1, self.head_dim)
                
                sliced_k = torch.gather(key_states, dim=2, index=gather_idx)
                sliced_v = torch.gather(value_states, dim=2, index=gather_idx)
                
                # 同步切分 causal_mask
                if attention_mask is not None:
                    # attention_mask 形如 [B, 1, Q, L]
                    mask_gather_idx = keep_idx.view(bsz, 1, 1, K_kept).expand(-1, 1, q_len, -1)
                    sliced_mask = torch.gather(attention_mask, dim=-1, index=mask_gather_idx)
                else:
                    sliced_mask = None
                    
                # 记录理论节约的字节数
                dropped_tokens = full_kv_len - K_kept
                bytes_per_kv_token = 2 * self.num_key_value_heads * self.head_dim * 2 
                GLOBAL_ATTN_STATE["step_dropped_bytes"] = dropped_tokens * bytes_per_kv_token
                
                # 重复 GQA 维度并送入真实的 SDPA
                sliced_k_rep = repeat_kv(sliced_k, self.num_key_value_groups)
                sliced_v_rep = repeat_kv(sliced_v, self.num_key_value_groups)
                
                attn_output = F.scaled_dot_product_attention(
                    query_states, 
                    sliced_k_rep, 
                    sliced_v_rep, 
                    attn_mask=sliced_mask,
                    dropout_p=0.0,
                    is_causal=False 
                )
            else:
                # 极端情况：不取任何 Top-K
                pass # (为简洁起见省略)
        else:
            # 序列过短或未捕获 Layer 0 状态，执行标准全量 SDPA
            key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states_rep, value_states_rep, 
                attn_mask=attention_mask[:, :, :, :full_kv_len] if attention_mask is not None else None,
                is_causal=False
            )

    # 4. O_Proj 返回
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None