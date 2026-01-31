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
import math
import types
import gc
from collections import defaultdict

# ====== 1. 环境清理与配置 ======
# 建议在运行前手动执行: pkill -9 python 确保显存干净
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
SAVE_DIR = "experiment_logs/exp_ppl_increase_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 1024  
NUM_SAMPLES = 100
# 阈值设置：从 0.99 到 1.0
THRESHOLDS = np.linspace(0.99, 1.0, 21)[::-1].tolist() 
# 确保 1.0 在第一个，作为 Baseline
if THRESHOLDS[0] != 1.0:
    THRESHOLDS.insert(0, 1.0)

CURRENT_THRESHOLD = 1.0 
LAYER_STATS = defaultdict(lambda: {'total': 0, 'kept': 0})

# ====== 2. 官方辅助函数 (1:1 复刻 modeling_qwen2_moe.py) ======
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
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ====== 3. 核心：1:1 源码对齐的 Monkey Patch ======
def custom_qwen2moe_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, 
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # [Step 1] 参数获取
    bsz, q_len, _ = hidden_states.size()
    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = self.head_dim 
    
    # [Step 2] Projections
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # [Step 3] RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # [Step 4] Cache Update
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # [Step 5] Attention Core
    # GQA Repeat
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Score Calculation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # ============================================================
    # ✂️ [插入点] 稀疏化逻辑 (KV Cache Pruning Simulation)
    # ============================================================
    kv_seq_len = key_states.shape[-2]
    # 仅统计下三角有效区域，保证 Savings 准确
    if q_len == kv_seq_len:
        valid_total = bsz * num_heads * (q_len * (q_len + 1)) // 2
    else:
        valid_total = attn_weights.numel()

    if CURRENT_THRESHOLD < 1.0:
        # 转为 float 进行高精度排序
        probs = attn_weights.float()
        sorted_vals, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_vals, dim=-1)
        
        # 1. 确定哪些要丢弃
        mask_to_remove = cumulative_probs > CURRENT_THRESHOLD
        mask_to_remove[..., 0] = False # 永远保留最大的那个，防止全0
        
        # 2. 还原 Mask 位置
        final_mask = torch.zeros_like(probs, dtype=torch.bool)
        final_mask.scatter_(dim=-1, index=sorted_indices, src=mask_to_remove)
        
        # 3. 统计 Savings
        kept_count = (~final_mask).sum().item()
        LAYER_STATS[self.layer_idx]['total'] += valid_total
        LAYER_STATS[self.layer_idx]['kept'] += kept_count

        # 4. 应用 Mask (模拟 KV Cache 丢弃)
        attn_weights = attn_weights.masked_fill(final_mask, 0.0)
        
        # 5. ⚠️ 关键步骤：重新归一化 (Renormalization)
        # 这在数学上等价于 "Attention 仅在剩余的 KV 上进行 Softmax"
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
    else:
        LAYER_STATS[self.layer_idx]['total'] += valid_total
        LAYER_STATS[self.layer_idx]['kept'] += valid_total
    # ============================================================

    # Output Proj
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

# ====== 4. 加载模型 (9GB 安全模式) ======
print(">>> [Step 1] Loading Model (FP16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 调整为 9GB 安全显存
MEMORY_MAP = {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "80GiB"}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    max_memory=MEMORY_MAP, 
    torch_dtype=torch.float16, 
    attn_implementation="eager", 
    trust_remote_code=True
)

# 应用 Monkey Patch
for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(custom_qwen2moe_forward, layer.self_attn)

print(">>> Monkey Patch Applied.")

# ====== 5. 实验执行 ======
from datasets import load_dataset
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train", streaming=True)
eval_texts = []
iter_ds = iter(dataset)
for _ in range(100): next(iter_ds) 
for _ in range(NUM_SAMPLES):
    try:
        t = next(iter_ds)['text']
        if len(t) > SEQ_LEN: eval_texts.append(t)
    except: break

print(f">>> [Step 2] Running Experiments...")
results = []
baseline_ppl = None

for thresh in THRESHOLDS:
    torch.cuda.empty_cache()
    gc.collect()
    CURRENT_THRESHOLD = thresh
    LAYER_STATS.clear()
    losses = []
    
    try:
        desc = "Baseline" if thresh == 1.0 else f"Sparse {thresh:.3f}"
        for text in tqdm(eval_texts, desc=desc, leave=False):
            inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True).to("cuda")
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
                losses.append(out.loss.item())
        
        avg_nll = np.mean(losses)
        ppl = np.exp(avg_nll)
        
        # 记录 Baseline
        if thresh == 1.0:
            baseline_ppl = ppl
            print(f"\n✅ Baseline PPL: {ppl:.4f}")
        
        # 计算 Savings
        t_total = sum(s['total'] for s in LAYER_STATS.values())
        savings = 0.0
        if t_total > 0:
            t_kept = sum(s['kept'] for s in LAYER_STATS.values())
            savings = 1.0 - (t_kept / t_total)

        # 计算 PPL 增长百分比
        ppl_increase_pct = 0.0
        if baseline_ppl is not None:
            ppl_increase_pct = (ppl - baseline_ppl) / baseline_ppl * 100
            
        print(f"P={thresh:.3f} | PPL: {ppl:.4f} (+{ppl_increase_pct:.2f}%) | Savings: {savings:.2%}")
        
        results.append({
            "Threshold": thresh, 
            "PPL": ppl, 
            "PPL_Increase_%": ppl_increase_pct,
            "Savings": savings
        })
        
    except Exception as e:
        print(f"Error at {thresh}: {e}")
        import traceback
        traceback.print_exc()
        break

# ====== 6. 绘图与保存 ======
if len(results) > 0:
    df = pd.DataFrame(results)
    # 保存包含增长率的 CSV
    df.to_csv(os.path.join(SAVE_DIR, "final_results_with_increase.csv"), index=False)
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1, color2 = 'tab:red', 'tab:blue'
    
    # 左轴：PPL Increase %
    ax1.set_xlabel('Cumulative Probability Threshold')
    ax1.set_ylabel('PPL Increase (%)', color=color1, fontweight='bold')
    ax1.plot(df["Threshold"], df["PPL_Increase_%"], marker='o', color=color1, label="PPL Degradation")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 增加一条 1% 的容忍线
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='1% Threshold')
    ax1.set_xlim(1.001, 0.989) # 聚焦在 0.99 - 1.0 区域
    
    # 右轴：Savings
    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory/Compute Savings (%)', color=color2, fontweight='bold')
    ax2.plot(df["Threshold"], df["Savings"]*100, marker='s', linestyle='--', color=color2, label="Savings")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f"Trade-off: Precision Loss vs Memory Savings\n(Simulating KV Cache Pruning)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "final_tradeoff_chart.png"))
    print(f"Done. Saved to {SAVE_DIR}")