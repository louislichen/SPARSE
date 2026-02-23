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

# kv pruning based on specific_layer(layer0) Attention distribution

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

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (head_dim ** -0.5) # self.scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    kv_seq_len = key_states.shape[-2]
    if q_len == kv_seq_len:
        valid_total = bsz * (q_len * (q_len + 1)) // 2
    else:
        valid_total = attn_weights.numel() // num_heads

    if self.layer_idx == 0:
        probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        global_scores = probs.sum(dim=1) 
        
        mean_score = global_scores.mean(dim=-1, keepdim=True)
        std_score = global_scores.std(dim=-1, keepdim=True)
        
        z_scores = (global_scores - mean_score) / (std_score + 1e-6)
        
        keep_mask = z_scores > Z_SCORE_THRESHOLD
        
        if kv_seq_len > SINK_SIZE:
            keep_mask[..., :SINK_SIZE] = True
        else:
            keep_mask[..., :] = True
            
        if kv_seq_len > LOCAL_WINDOW:
            keep_mask[..., -LOCAL_WINDOW:] = True
        else:
            keep_mask[..., :] = True

        mask_to_remove = ~keep_mask
        mask_to_remove = mask_to_remove.unsqueeze(1) 
        
        INTER_LAYER_CACHE["layer0_mask"] = mask_to_remove.detach().clone()
        
        final_weights = probs
        kept_count = keep_mask.sum().item()

    elif 0 < self.layer_idx <= TARGET_FOLLOW_DEPTH:
        mask_to_remove = INTER_LAYER_CACHE.get("layer0_mask")
        
        if mask_to_remove is not None:
            if mask_to_remove.device != attn_weights.device:
                mask_to_remove = mask_to_remove.to(attn_weights.device)
            
            attn_weights = attn_weights.masked_fill(mask_to_remove, torch.finfo(attn_weights.dtype).min)
            final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

            kept_count = (~mask_to_remove).sum().item()
            LAYER_STATS[self.layer_idx]['total'] += valid_total
            LAYER_STATS[self.layer_idx]['kept'] += kept_count
            
        else:
            final_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            LAYER_STATS[self.layer_idx]['total'] += valid_total
            LAYER_STATS[self.layer_idx]['kept'] += valid_total

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


TARGET_FOLLOW_DEPTH = 0  
INTER_LAYER_CACHE = {"layer0_mask": None}      
LAYER_STATS = defaultdict(lambda: {'total': 0, 'kept': 0})
Z_SCORE_THRESHOLD = 0.5 
SINK_SIZE = 4           
LOCAL_WINDOW = 32


def main():
    global TARGET_FOLLOW_DEPTH, INTER_LAYER_CACHE, Z_SCORE_THRESHOLD, SINK_SIZE, LOCAL_WINDOW
    
    MODEL_PATH = "/data1/cl/models/Qwen1.5-MoE-A2.7B-Chat"
    DATA_PATH = "/data1/cl/Dataset/c4/c4-train.00000-of-01024.json.gz"
    SAVE_DIR = "/data1/cl/SPARSE/results/specific_layer_kv_pruning"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SEQ_LEN = 1024  
    NUM_SAMPLES = 100

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        attn_implementation="eager", 
        trust_remote_code=True
    )

    DEPTH_EXPERIMENTS = list(range(model.config.num_hidden_layers))

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
        except: break

    print(f"Running Experiments...")
    results = []

    for depth in DEPTH_EXPERIMENTS:
        torch.cuda.empty_cache()
        gc.collect()
        
        TARGET_FOLLOW_DEPTH = depth
        LAYER_STATS.clear()
        losses = []
        
        print(f"--- Experiment: Layers 0 to {depth} use Layer 0's Mask ---")
        
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
        df.to_csv(os.path.join(SAVE_DIR, "consistency_results.csv"), index=False)
        
        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Number of Following Layers (Depth)')
        ax1.set_ylabel('Perplexity (Lower is Better)', color='tab:red', fontweight='bold')
        ax1.plot(df["Follow_Depth"], df["PPL"], marker='o', color='tab:red', label="PPL")
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Total Compute Savings (%)', color='tab:blue', fontweight='bold')
        ax2.plot(df["Follow_Depth"], df["Global_Savings"]*100, marker='s', linestyle='--', color='tab:blue', label="Savings")
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        plt.title(f"Consistency Check: Propagating Layer 0 Mask\nEffect on PPL vs # of Layers Forced")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "consistency_tradeoff.png"))
        print(f"Done. Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()