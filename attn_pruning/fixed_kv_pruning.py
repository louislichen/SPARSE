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
from datasets import load_dataset
import types
import gc
from collections import defaultdict


def dynamic_cascade_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, 
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    
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
    
    # retrieve params
    bsz, q_len, _ = hidden_states.size()
    num_heads = self.config.num_attention_heads
    head_dim = self.head_dim 
    
    # projections
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # add pos
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # cache update
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # attn compute
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    kv_seq_len = key_states.shape[-2]
    if q_len == kv_seq_len:
        valid_total = bsz * num_heads * (q_len * (q_len + 1)) // 2
    else:
        valid_total = attn_weights.numel()

    if CURRENT_THRESHOLD < 1.0:
        
        if self.layer_idx == 0:
            probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            sorted_vals, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_vals, dim=-1)
            mask_to_remove = cumulative_probs > CURRENT_THRESHOLD
            mask_to_remove[..., 0] = False 
            
            mask_L0 = torch.zeros_like(probs, dtype=torch.bool)
            mask_L0.scatter_(dim=-1, index=sorted_indices, src=mask_to_remove)
            
            mask_L0_union = torch.amin(mask_L0, dim=1, keepdim=True).to(torch.bool)
            INTER_LAYER_CACHE["layer0_mask"] = mask_L0_union.detach().clone()
            
            final_weights = probs.masked_fill(mask_L0, 0.0)
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
            kept_count = (~mask_L0).sum().item()

        else:
            mask_L0 = INTER_LAYER_CACHE.get("layer0_mask")
            
            if mask_L0 is not None:
                if mask_L0.device != attn_weights.device:
                    mask_L0 = mask_L0.to(attn_weights.device)
                
                attn_weights = attn_weights.masked_fill(mask_L0, torch.finfo(attn_weights.dtype).min)
                final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                
                kept_count = (~mask_L0).sum().item() * num_heads
            else:
                final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                kept_count = valid_total
            
    else:
        final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        kept_count = valid_total
        if self.layer_idx == 0:
            INTER_LAYER_CACHE["layer0_mask"] = None

    LAYER_STATS[self.layer_idx]['total'] += valid_total
    LAYER_STATS[self.layer_idx]['kept'] += kept_count
    
    final_weights = final_weights.to(query_states.dtype)
    attn_output = torch.matmul(final_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, final_weights

CURRENT_THRESHOLD = 1.0 
LAYER_STATS = defaultdict(lambda: {'total': 0, 'kept': 0, 'layer0_limit': 0})
INTER_LAYER_CACHE = {"layer0_mask": None}

def main():
    global CURRENT_THRESHOLD

    MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
    DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
    SAVE_DIR = "/home/ztl/cl/LLM_Partition/results/fixed_kv_pruning"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SEQ_LEN = 1024
    NUM_SAMPLES = 100 
    THRESHOLDS = np.linspace(0.50, 1.0, 11)[::-1].tolist()
    if THRESHOLDS[0] != 1.0: THRESHOLDS.insert(0, 1.0)

    CURRENT_THRESHOLD = 1.0

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

    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(dynamic_cascade_forward, layer.self_attn)

    dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train", streaming=True)
    eval_texts = []
    iter_ds = iter(dataset)
    for _ in range(100): next(iter_ds) 
    for _ in range(NUM_SAMPLES):
        try:
            t = next(iter_ds)['text']
            if len(t) > SEQ_LEN: eval_texts.append(t)
        except: break

    print(f"Running Experiments...")
    results = []
    baseline_ppl = None

    for thresh in THRESHOLDS:
        torch.cuda.empty_cache()
        gc.collect()
        CURRENT_THRESHOLD = thresh

        INTER_LAYER_CACHE["layer0_mask"] = None
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
                print(f"Baseline PPL: {ppl:.4f}")
            
            t_total = sum(s['total'] for s in LAYER_STATS.values())
            t_kept = sum(s['kept'] for s in LAYER_STATS.values())
            
            savings = 0.0
            if t_total > 0:
                savings = 1.0 - (t_kept / t_total)

            ppl_increase_pct = 0.0
            if baseline_ppl is not None:
                ppl_increase_pct = (ppl - baseline_ppl) / baseline_ppl * 100
                
            print(f"P={thresh:.3f} | PPL: {ppl:.4f} (+ {ppl_increase_pct:.2f}%) | Savings: {savings:.2%}")
            
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

    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(SAVE_DIR, "results.csv"), index=False)
        
        sns.set_theme(style="whitegrid")
        _, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Cumulative Probability Threshold')
        ax1.set_ylabel('PPL Increase (%)', color=color, fontweight='bold')
        ax1.plot(df["Threshold"], df["PPL_Increase_%"], marker='o', color=color, linewidth=2, label="PPL Degradation")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('KV / Attention Sparsity (%)', color=color, fontweight='bold')
        ax2.plot(df["Threshold"], df["Savings"]*100, marker='s', linestyle='--', color=color, linewidth=2, label="Savings")
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"Dynamic Cascade Pruning: All Layers Mirror Layer 0 Selection")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "tradeoff.png"))
        print(f"Done. Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()