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

# ====== 1. 环境配置 ======
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
SAVE_DIR = "experiment_logs/exp_dynamic_cascade"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 1024
NUM_SAMPLES = 100 
THRESHOLDS = np.linspace(0.99, 1.0, 11)[::-1].tolist()
if THRESHOLDS[0] != 1.0: THRESHOLDS.insert(0, 1.0)

CURRENT_THRESHOLD = 1.0
# 统计：{layer_idx: {'total': int, 'kept': int, 'layer0_limit': int}}
LAYER_STATS = defaultdict(lambda: {'total': 0, 'kept': 0, 'layer0_limit': 0})
INTER_LAYER_CACHE = {"layer0_mask": None}

# ====== 2. 官方辅助函数 ======
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

# ====== 3. 核心：动态级联稀疏 Attention (Dynamic Cascade) ======
def dynamic_cascade_forward(
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
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 计算原始分数 (Raw Scores)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ============================================================
    # 动态级联剪枝 (Layer 0 Filtering + Layer i Re-Selection)
    # ============================================================
    
    # 统计基数
    kv_seq_len = key_states.shape[-2]
    if q_len == kv_seq_len:
        valid_total = bsz * num_heads * (q_len * (q_len + 1)) // 2
    else:
        valid_total = attn_weights.numel()

    # --- 逻辑分支 ---
    if CURRENT_THRESHOLD < 1.0:
        
        # [A] 处理 Layer 0：作为 "Filter"，决定搜索空间
        if self.layer_idx == 0:
            # 1. Softmax 得到概率
            probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            # 2. 排序 & 阈值截断
            # 这里的 mask_to_remove 是：True=丢弃, False=保留
            sorted_vals, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_vals, dim=-1)
            mask_to_remove = cumulative_probs > CURRENT_THRESHOLD
            mask_to_remove[..., 0] = False # 保护 Top-1
            
            # 3. 还原回原始位置得到 Mask_L0
            mask_L0 = torch.zeros_like(probs, dtype=torch.bool)
            mask_L0.scatter_(dim=-1, index=sorted_indices, src=mask_to_remove)
            
            # 4. 保存到全局缓存 (这是后续层的计算上限)
            INTER_LAYER_CACHE["layer0_mask"] = mask_L0.detach().clone()
            
            # 5. 生成本层权重
            final_weights = probs.masked_fill(mask_L0, 0.0)
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
            # 统计
            kept_count = (~mask_L0).sum().item()
            # Layer 0 的 Limit 就是它自己
            limit_count = kept_count 

        # [B] 处理 Layer > 0：在 Layer 0 的空间内重新计算
        else:
            # 1. 获取 Layer 0 的决定 (这是我们的搜索空间)
            mask_L0 = INTER_LAYER_CACHE["layer0_mask"]
            
            if mask_L0 is not None:
                if mask_L0.device != attn_weights.device:
                    mask_L0 = mask_L0.to(attn_weights.device)
                
                # 2. 【关键】物理屏蔽
                # 将 Layer 0 丢弃的位置设为 -inf。这模拟了我们只对保留的 Token 进行了计算。
                # 此时 Softmax 的分母只会包含 mask_L0=False 的部分。
                attn_weights = attn_weights.masked_fill(mask_L0, torch.finfo(attn_weights.dtype).min)
                
                # 统计 Layer 0 给我们留了多少 (计算上限)
                limit_count = (~mask_L0).sum().item()
            else:
                limit_count = valid_total

            # 3. Softmax (在 Layer 0 的子集上归一化)
            probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            # 4. 【关键】二次筛选 (Re-Selection)
            # 虽然我们被限制在 mask_L0 的范围内，但在该范围内，我们根据本层的分布再次截断
            # 比如 Layer 0 留了 100 个，Layer 5 可能发现其中只有 10 个加起来就够 0.99 了
            sorted_vals, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_vals, dim=-1)
            
            mask_to_remove_curr = cumulative_probs > CURRENT_THRESHOLD
            mask_to_remove_curr[..., 0] = False 
            
            # 还原当前层的掩码
            mask_L_curr = torch.zeros_like(probs, dtype=torch.bool)
            mask_L_curr.scatter_(dim=-1, index=sorted_indices, src=mask_to_remove_curr)
            
            # 5. 生成本层权重
            # 注意：mask_L_curr 包含了 mask_L0 的约束 (因为 mask_L0 的位置 probs 为 0，排序在最后)
            final_weights = probs.masked_fill(mask_L_curr, 0.0)
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
            kept_count = (~mask_L_curr).sum().item()

    # [C] Baseline 模式
    else:
        final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        kept_count = valid_total
        limit_count = valid_total
        if self.layer_idx == 0:
            INTER_LAYER_CACHE["layer0_mask"] = None

    # 统计写入
    LAYER_STATS[self.layer_idx]['total'] += valid_total
    LAYER_STATS[self.layer_idx]['kept'] += kept_count
    # 记录 Layer 0 传下来的上限 (用于计算真实的计算量节省)
    LAYER_STATS[self.layer_idx]['layer0_limit'] += limit_count
    
    # Output Proj
    final_weights = final_weights.to(query_states.dtype)
    attn_output = torch.matmul(final_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, final_weights

# ====== 4. 加载模型 ======
print(">>> [Step 1] Loading Model (Dynamic Cascade Mode)...")
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

for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(dynamic_cascade_forward, layer.self_attn)
print(">>> Monkey Patch Applied: Layer 0 filters, subsequent layers re-select.")

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

print(f">>> [Step 2] Running Experiments (Threshold 0.9 - 1.0)...")
results = []
baseline_ppl = None

for thresh in THRESHOLDS:
    torch.cuda.empty_cache()
    gc.collect()
    CURRENT_THRESHOLD = thresh
    LAYER_STATS.clear()
    losses = []
    
    try:
        desc = f"Thresh {thresh:.2f}"
        for text in tqdm(eval_texts, desc=desc, leave=False):
            inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True).to("cuda")
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
                losses.append(out.loss.item())
        
        avg_nll = np.mean(losses)
        ppl = np.exp(avg_nll)
        
        if thresh == 1.0:
            baseline_ppl = ppl
            print(f"\n✅ Baseline PPL: {ppl:.4f}")
        
        # --- 统计 Metrics ---
        # 1. 全局总有效 Token (分母)
        t_total = sum(s['total'] for s in LAYER_STATS.values())
        
        # 2. 实际参与计算的 Token (Layer 0 留下的 + Layer 0 本身)
        # 对于 Layer 0，limit 就是 kept。对于 Layer > 0，limit 是 Layer 0 留下的。
        # 这个指标衡量的是 "物理上我们需要加载多少 KV Cache 到计算单元"
        t_computed = sum(s['layer0_limit'] for s in LAYER_STATS.values())
        
        # 3. 最终保留用于 Softmax 结果的 Token (最稀疏)
        t_final_kept = sum(s['kept'] for s in LAYER_STATS.values())
        
        # 计算 Savings
        # 计算量节省 (相对于全量计算)
        compute_savings_pct = 0.0
        avg_tokens_computed = 0.0
        if t_total > 0:
            compute_savings_pct = 1.0 - (t_computed / t_total)
            # 平均每个 Head 每步计算多少个 Token
            avg_tokens_computed = t_computed / (len(eval_texts) * len(model.model.layers))

        ppl_increase_pct = 0.0
        if baseline_ppl is not None:
            ppl_increase_pct = (ppl - baseline_ppl) / baseline_ppl * 100
            
        print(f"P={thresh:.2f} | PPL: {ppl:.4f} (+{ppl_increase_pct:.2f}%) | Compute Savings: {compute_savings_pct:.2%}")
        
        results.append({
            "Threshold": thresh, 
            "PPL": ppl, 
            "PPL_Increase_%": ppl_increase_pct,
            "Compute_Savings_%": compute_savings_pct,
            "Avg_Tokens_Calculated": avg_tokens_computed # 新增：节省的Token数量指标
        })
        
    except Exception as e:
        print(f"Error at {thresh}: {e}")
        import traceback
        traceback.print_exc()
        break

# ====== 6. 绘图 ======
if len(results) > 0:
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_DIR, "dynamic_cascade_results.csv"), index=False)
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左轴: PPL Increase
    color = 'tab:red'
    ax1.set_xlabel('Cumulative Probability Threshold')
    ax1.set_ylabel('PPL Increase (%)', color=color, fontweight='bold')
    ax1.plot(df["Threshold"], df["PPL_Increase_%"], marker='o', color=color, linewidth=2, label="PPL Degradation")
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 右轴: Compute Savings
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Compute / KV Savings (%)', color=color, fontweight='bold')
    ax2.plot(df["Threshold"], df["Compute_Savings_%"]*100, marker='s', linestyle='--', color=color, linewidth=2, label="Compute Savings")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标注 Token 数量
    # 在 Savings 线上标注具体的 Token 数量，给用户直观感受
    for i, txt in enumerate(df["Avg_Tokens_Calculated"]):
        if i % 2 == 0: # 隔点标注防拥挤
             ax2.annotate(f"{int(txt)} toks", (df["Threshold"][i], df["Compute_Savings_%"][i]*100), 
                          xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, color='black')

    plt.title(f"Dynamic Cascade Pruning: Layer 0 Filtering -> Deep Layer Re-Selection\n(Metrics based on Physical Computation Reduction)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "dynamic_cascade_tradeoff.png"))
    print(f"Done. Results saved to {SAVE_DIR}")