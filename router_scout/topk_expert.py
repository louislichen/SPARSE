import os

import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def set_model_top_k(model, k):
    count = 0
    for layer in model.model.layers:
        if hasattr(layer, "mlp"):
            if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "top_k"):
                layer.mlp.gate.top_k = k
                count += 1
    return count


def main():
    MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
    DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
    SAVE_DIR = "/home/ztl/cl/LLM_Partition/results/router_scout"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SEQ_LEN = 1024      
    NUM_SAMPLES = 100   

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

    DEFAULT_TOP_K = model.config.num_experts_per_tok
    print(f"Default Top-K: {DEFAULT_TOP_K}")
    print(f"Total Experts: {model.config.num_experts}")

    dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train", streaming=True)
    eval_texts = []
    iter_ds = iter(dataset)

    for _ in range(100): next(iter_ds)
    with tqdm(total=NUM_SAMPLES, desc="Loading Samples") as pbar:
        while len(eval_texts) < NUM_SAMPLES:
            try:
                text = next(iter_ds)['text']
                if len(text) > 100:
                    eval_texts.append(text)
                    pbar.update(1)
            except StopIteration:
                break

    topks = [k for k in range(1, DEFAULT_TOP_K+1)]
    results = []

    for k in topks:
        modified_layers = set_model_top_k(model, k)
        print(f"Top-K = {k}, modify {modified_layers} layers...")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        losses = []
        
        for text in tqdm(eval_texts, desc=f"Eval K={k}"):
            inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True).to("cuda")
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
                losses.append(out.loss.item())
        
        avg_nll = np.mean(losses)
        ppl = np.exp(avg_nll)
        
        print(f"PPL: {ppl:.4f},  Avg NLL: {avg_nll:.4f}")
        
        results.append({
            "Top_K": k,
            "Is_Baseline": (k == DEFAULT_TOP_K),
            "PPL": ppl,
            "Avg_NLL": avg_nll,
        })

    df = pd.DataFrame(results)
    baseline_ppl = df[df["Is_Baseline"] == True]["PPL"].values[0]
    df["PPL_Degradation_%"] = (df["PPL"] - baseline_ppl) / baseline_ppl * 100
    print(df.to_string(index=False))

    csv_path = os.path.join(SAVE_DIR, "topk_ablation.csv")
    df.to_csv(csv_path, index=False)

    sns.set_theme(style="whitegrid")
    _, ax1 = plt.subplots(figsize=(10, 6))

    colors = ['tab:blue' if not is_base else 'tab:green' for is_base in df["Is_Baseline"]]
    bars = ax1.bar(df["Top_K"].astype(str), df["PPL"], color=colors, alpha=0.8, label='PPL')
    ax1.set_xlabel('Active Experts (Top-K)', fontsize=12)
    ax1.set_ylabel('Perplexity (Lower is Better)', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    ax2 = ax1.twinx()
    ax2.plot(df["Top_K"].astype(str), df["PPL_Degradation_%"], color='tab:red', marker='o', linewidth=2, linestyle='--')
    ax2.set_ylabel('PPL Degradation vs Baseline (%)', color='tab:red', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Effect of Top-K Experts on Model Performance\n(Model: Qwen1.5-MoE-A2.7B, Baseline K={DEFAULT_TOP_K})', fontsize=14)
    plt.tight_layout()

    img_path = os.path.join(SAVE_DIR, "topk_performance_chart.png")
    plt.savefig(img_path, dpi=300)


if __name__ == "__main__":
    main()