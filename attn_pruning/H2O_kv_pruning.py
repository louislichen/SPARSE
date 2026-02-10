import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import types
import gc
from collections import defaultdict

#H2O: Hybrid Heuristic and Heavy Hitter Pruning for Key-Value Pairs in Attention

H2O_CONFIG = {
    "SINK_SIZE": 4,          # 锚点：强制保留开头的 4 个 Token (防止 Softmax 崩溃)
    "WINDOW_SIZE": 32,       # 窗口：强制保留最近的 32 个 Token (保证局部语法)
    "FIXED_THRESHOLD": 0.6, # 筛选：对其余 Token 进行 Top-P 筛选 (捕捉长距离语义)
}

INTER_LAYER_CACHE = {"layer0_mask": None}
LAYER_STATS = defaultdict(lambda: {'total': 0, 'kept': 0})
TARGET_FOLLOW_DEPTH = 0


def custom_qwen2moe_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, 
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    global INTER_LAYER_CACHE
    
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
    
    # Attention Logic
    bsz, q_len, _ = hidden_states.size()
    num_heads = self.config.num_attention_heads
    head_dim = self.head_dim 
    
    input_shape = hidden_states.shape[:-1]
    # [B,Q,D] -> [B,Q,H,D/H]
    hidden_shape = (*input_shape, -1, head_dim)

    # projections and transform to [B,H,Q,D/H]
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # KV cache update
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # original attn score
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    # attn causual mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    kv_seq_len = key_states.shape[-2]
    
    # the quantity of valid attention scores (for later compute savings)
    if q_len == kv_seq_len:
        valid_total = bsz * (q_len * (q_len + 1)) // 2
    else:
        valid_total = attn_weights.numel() // num_heads

    #  H2O-Enhanced Pruning Logic (Layer 0)
    if self.layer_idx == 0:
        probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        # 1. 计算基于内容的 Top-P 掩码 (Heavy Hitters)
        sorted_vals, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_vals, dim=-1)
        
        # 初始 Mask: True 代表要被移除
        mask_to_remove = cumulative_probs > H2O_CONFIG["FIXED_THRESHOLD"]
        
        # 还原回原始顺序
        mask_L0 = torch.zeros_like(probs, dtype=torch.bool)
        mask_L0.scatter_(dim=-1, index=sorted_indices, src=mask_to_remove)
        
        # 2. 强制保留 Attention Sink (开头的几个 Token)
        # 无论概率多低，前 SINK_SIZE 个 Token 必须保留 (False = Keep)
        if H2O_CONFIG["SINK_SIZE"] > 0:
            mask_L0[..., :H2O_CONFIG["SINK_SIZE"]] = False 
        
        # 3. Local Window's tokens
        if H2O_CONFIG["WINDOW_SIZE"] > 0 and q_len == kv_seq_len:
            # 构建距离矩阵,计算每个 Query 与 Key 之间的相对距离,并根据窗口大小生成掩码，强制保留窗口内的 Token
            q_idx = torch.arange(q_len, device=mask_L0.device).unsqueeze(1)
            k_idx = torch.arange(kv_seq_len, device=mask_L0.device).unsqueeze(0)
            # 距离: distance = query_pos - key_pos
            distance = q_idx - k_idx
            in_window_mask = (distance >= 0) & (distance < H2O_CONFIG["WINDOW_SIZE"])
            # 广播到 [B, H, Q, K]
            in_window_mask = in_window_mask.unsqueeze(0).unsqueeze(0)
            # 强制保留窗口内的 Token
            mask_L0 = mask_L0.masked_fill(in_window_mask, False)
        
        # 4. mutli head Union Strategy
        mask_L0_union = torch.amin(mask_L0, dim=1, keepdim=True).to(torch.bool)
        INTER_LAYER_CACHE["layer0_mask"] = mask_L0_union.detach().clone()
        
        # 5. apply mask and Normalization
        final_weights = probs.masked_fill(mask_L0, 0.0)
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        kept_count = (~mask_L0_union).sum().item()
        LAYER_STATS[self.layer_idx]['total'] += valid_total
        LAYER_STATS[self.layer_idx]['kept'] += kept_count

    # ------------------------------------------------------------------
    #  Follower Layers (Layer 1 ~ Target)
    # ------------------------------------------------------------------
    elif 0 < self.layer_idx <= TARGET_FOLLOW_DEPTH:
        mask_L0 = INTER_LAYER_CACHE.get("layer0_mask")
        
        if mask_L0 is not None:
            if mask_L0.device != attn_weights.device:
                mask_L0 = mask_L0.to(attn_weights.device)
            
            # 直接复用 Layer 0 的决策
            attn_weights = attn_weights.masked_fill(mask_L0, torch.finfo(attn_weights.dtype).min)
            final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

            kept_count = (~mask_L0).sum().item()
            LAYER_STATS[self.layer_idx]['total'] += valid_total
            LAYER_STATS[self.layer_idx]['kept'] += kept_count
            
        else:
            final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            LAYER_STATS[self.layer_idx]['total'] += valid_total
            LAYER_STATS[self.layer_idx]['kept'] += valid_total

    # ------------------------------------------------------------------
    #  Other Layers (Full Attention)
    # ------------------------------------------------------------------
    else:
        final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        LAYER_STATS[self.layer_idx]['total'] += valid_total
        LAYER_STATS[self.layer_idx]['kept'] += valid_total

    final_weights = final_weights.to(query_states.dtype)
    attn_output = torch.matmul(final_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


def main():
    global TARGET_FOLLOW_DEPTH, INTER_LAYER_CACHE
    
    # 基础配置保持不变
    MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
    DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
    SAVE_DIR = "/home/ztl/wyf/SPARSE/results/h2o_hybrid_pruning_0.6"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SEQ_LEN = 1024  
    NUM_SAMPLES = 100
    DEPTH_EXPERIMENTS = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    MEMORY_MAP = {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "80GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        max_memory=MEMORY_MAP, 
        torch_dtype=torch.float16, 
        attn_implementation="eager", 
        trust_remote_code=True
    )

    # 注入修改后的 Forward 函数
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(custom_qwen2moe_forward, layer.self_attn)

    from datasets import load_dataset
    dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train", streaming=True)
    eval_texts = []
    iter_ds = iter(dataset)
    for _ in range(100): next(iter_ds) 
    for _ in range(NUM_SAMPLES):
        try:
            t = next(iter_ds)['text']
            if len(t) > SEQ_LEN: eval_texts.append(t)
        except: 
            print("Reached end of dataset or encountered an error while loading data.")
            break

    print(f"Running Experiments with H2O Logic: Sink={H2O_CONFIG['SINK_SIZE']}, Window={H2O_CONFIG['WINDOW_SIZE']}, Thresh={H2O_CONFIG['FIXED_THRESHOLD']}")
    results = []

    for depth in DEPTH_EXPERIMENTS:
        torch.cuda.empty_cache()
        gc.collect()
        
        TARGET_FOLLOW_DEPTH = depth
        LAYER_STATS.clear()
        losses = []
        
        print(f"--- Experiment: Layers 0 to {depth} use Layer 0's Hybrid Mask ---")
        
        try:
            for text in tqdm(eval_texts, leave=False):
                INTER_LAYER_CACHE.clear() 
                
                inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True).to("cuda")
                with torch.no_grad():
                    out = model(**inputs, labels=inputs["input_ids"])
                    losses.append(out.loss.item())
            
            avg_nll = np.mean(losses)
            ppl = np.exp(avg_nll)
            
            t_total = sum(s['total'] for s in LAYER_STATS.values())
            t_kept = sum(s['kept'] for s in LAYER_STATS.values())
            global_savings = 1.0 - (t_kept / t_total) if t_total > 0 else 0

            print(f"Follow Depth: {depth} | PPL: {ppl:.4f} | Avg Savings: {global_savings:.2%}")
            
            results.append({
                "Follow_Depth": depth, 
                "PPL": ppl, 
                "Global_Savings": global_savings
            })
            
        except Exception as e:
            print(f"Error at depth {depth}: {e}")
            import traceback
            traceback.print_exc()
            break

    if len(results) > 0:
        df = pd.DataFrame(results)
        baseline_ppl = df[df["Follow_Depth"] == 0]["PPL"].iloc[0]
        df["PPL_Rise_Pct"] = (df["PPL"] - baseline_ppl) / baseline_ppl * 100  # 转为百分比
        
        df.to_csv(os.path.join(SAVE_DIR, "h2o_consistency_results.csv"), index=False)
        
        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 7))  # 稍微放大图表，提升可读性
        
        ax1.set_xlabel('Number of Following Layers (Depth)', fontsize=12)
        ax1.set_ylabel('Perplexity (Lower is Better)', color='tab:red', fontweight='bold', fontsize=11)
        line1 = ax1.plot(df["Follow_Depth"], df["PPL"], marker='o', color='tab:red', label="PPL", linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(alpha=0.3)

        ax1_twin = ax1.twinx()  # 左轴的第二个刻度
        ax1_twin.set_ylabel('PPL Rise (%) (Higher = Worse)', color='darkred', fontweight='bold', fontsize=11)
        line2 = ax1_twin.plot(df["Follow_Depth"], df["PPL_Rise_Pct"], marker='^', color='darkred', 
                              linestyle='-.', label="PPL Rise %", linewidth=2)
        ax1_twin.tick_params(axis='y', labelcolor='darkred')
        for x, y in zip(df["Follow_Depth"], df["PPL_Rise_Pct"]):
            if y > 0:  # 只标注上升的部分
                ax1_twin.text(x, y+0.1, f"{y:.1f}%", fontsize=8, color='darkred')

        ax2 = ax1_twin.twinx()  # 右轴
        ax2.spines['right'].set_position(('outward', 60))  
        ax2.set_ylabel('Total Compute Savings (%)', color='tab:blue', fontweight='bold', fontsize=11)
        line3 = ax2.plot(df["Follow_Depth"], df["Global_Savings"]*100, marker='s', linestyle='--', 
                         color='tab:blue', label="Compute Savings %", linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10)

        plt.title(f"H2O Hybrid Pruning (Sink={H2O_CONFIG['SINK_SIZE']}, Win={H2O_CONFIG['WINDOW_SIZE']}, P={H2O_CONFIG['FIXED_THRESHOLD']})", 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "h2o_tradeoff.png"), bbox_inches='tight', dpi=300)
        print(f"Done. Saved to {SAVE_DIR}")
        print("\n=== PPL 上升百分比汇总 ===")
        print(df[["Follow_Depth", "PPL", "PPL_Rise_Pct", "Global_Savings"]].round(3))
        
        # df.to_csv(os.path.join(SAVE_DIR, "h2o_consistency_results.csv"), index=False)
        
        # sns.set_theme(style="whitegrid")
        # fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # ax1.set_xlabel('Number of Following Layers (Depth)')
        # ax1.set_ylabel('Perplexity (Lower is Better)', color='tab:red', fontweight='bold')
        # ax1.plot(df["Follow_Depth"], df["PPL"], marker='o', color='tab:red', label="PPL")
        # ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Total Compute Savings (%)', color='tab:blue', fontweight='bold')
        # ax2.plot(df["Follow_Depth"], df["Global_Savings"]*100, marker='s', linestyle='--', color='tab:blue', label="Savings")
        # ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        # plt.title(f"H2O Hybrid Pruning (Sink={H2O_CONFIG['SINK_SIZE']}, Win={H2O_CONFIG['WINDOW_SIZE']}, P={H2O_CONFIG['FIXED_THRESHOLD']})")
        # plt.tight_layout()
        # plt.savefig(os.path.join(SAVE_DIR, "h2o_tradeoff.png"))
        # print(f"Done. Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()