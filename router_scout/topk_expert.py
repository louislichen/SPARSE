import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import time

# ====== 2. 路径设置 ======
MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
SAVE_DIR = "dyexp"
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验参数
SEQ_LEN = 1024       # 增加长度以获得更稳定的 PPL
NUM_SAMPLES = 100    # 样本数
BATCH_SIZE = 1       # 1080Ti 建议 BS=1

# ====== 3. 辅助函数：动态修改 Top-K ======
def set_model_top_k(model, k):
    """
    遍历模型所有层，将 MoE Router 的 top_k 修改为指定值。
    """
    count = 0
    # Qwen2MoE 的层列表在 model.model.layers
    for layer in model.model.layers:
        # 检查该层是否有 mlp 模块
        if hasattr(layer, "mlp"):
            # 检查 mlp 是否是稀疏块 (SparseMoeBlock)，通常有 gate 属性
            # 如果是 Dense MLP (Qwen2MoeMLP)，则没有 gate，不需要改
            if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "top_k"):
                layer.mlp.gate.top_k = k
                count += 1
    return count

# ====== 4. 模型加载 (FP16 9GB 限制版) ======
print(">>> [Step 1] Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 1080Ti 显存策略：每张卡 9GB，留 2GB 给系统
MEMORY_MAP = {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "80GiB"}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    max_memory=MEMORY_MAP, 
    torch_dtype=torch.float16, 
    attn_implementation="eager", 
    trust_remote_code=True
)

# 获取默认配置作为基线
DEFAULT_TOP_K = model.config.num_experts_per_tok
print(f">>> Model Loaded. Default Top-K (Baseline): {DEFAULT_TOP_K}")
print(f">>> Total Experts: {model.config.num_experts}")

# ====== 5. 数据准备 ======
print(">>> [Step 2] Preparing Data...")
from datasets import load_dataset
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train", streaming=True)
eval_texts = []
iter_ds = iter(dataset)

# 跳过前 100 条以防数据头异常
for _ in range(100): next(iter_ds)

# 获取样本
with tqdm(total=NUM_SAMPLES, desc="Loading Samples") as pbar:
    while len(eval_texts) < NUM_SAMPLES:
        try:
            text = next(iter_ds)['text']
            if len(text) > 100: # 过滤太短的
                eval_texts.append(text)
                pbar.update(1)
        except StopIteration:
            break

# ====== 6. 实验主循环 ======
# 实验设置：[1, 2, 3, Baseline]
# 注意：确保 Baseline 在列表中
experiment_ks = [1, 2, 3]
if DEFAULT_TOP_K not in experiment_ks:
    experiment_ks.append(DEFAULT_TOP_K)
experiment_ks.sort()

results = []

print(f"\n>>> [Step 3] Running Top-K Ablation Experiment: K={experiment_ks}")

for k in experiment_ks:
    # 1. 修改 Top-K
    modified_layers = set_model_top_k(model, k)
    print(f"\n--- Testing Top-K = {k} (Modified {modified_layers} sparse layers) ---")
    
    # 2. 垃圾回收，清空缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 3. 评测 PPL
    losses = []
    start_time = time.time()
    
    # 使用 tqdm 显示进度
    for text in tqdm(eval_texts, desc=f"Eval K={k}"):
        inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True).to("cuda")
        with torch.no_grad():
            # 计算 Loss (NLL)
            out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
    
    end_time = time.time()
    
    # 4. 统计结果
    avg_nll = np.mean(losses)
    ppl = np.exp(avg_nll)
    avg_infer_time = (end_time - start_time) / len(eval_texts)
    
    print(f"  -> Result: PPL = {ppl:.4f} | Avg NLL = {avg_nll:.4f}")
    
    results.append({
        "Top_K": k,
        "Is_Baseline": (k == DEFAULT_TOP_K),
        "PPL": ppl,
        "Avg_NLL": avg_nll,
        "Inference_Time_Sec": avg_infer_time
    })

# ====== 7. 结果保存与可视化 ======
print("\n>>> [Step 4] Saving Results & Visualizing...")
df = pd.DataFrame(results)

# 计算相对于基线的 PPL 恶化比例
baseline_ppl = df[df["Is_Baseline"] == True]["PPL"].values[0]
df["PPL_Degradation_%"] = (df["PPL"] - baseline_ppl) / baseline_ppl * 100

# 打印表格
print(df.to_string(index=False))

# 保存 CSV
csv_path = os.path.join(SAVE_DIR, "topk_ablation_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")

# 绘图
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图：PPL
colors = ['tab:blue' if not is_base else 'tab:green' for is_base in df["Is_Baseline"]]
bars = ax1.bar(df["Top_K"].astype(str), df["PPL"], color=colors, alpha=0.8, label='PPL')
ax1.set_xlabel('Active Experts (Top-K)', fontsize=12)
ax1.set_ylabel('Perplexity (Lower is Better)', color='tab:blue', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 在柱子上标注数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

# 折线图：Relative Degradation (右轴)
ax2 = ax1.twinx()
ax2.plot(df["Top_K"].astype(str), df["PPL_Degradation_%"], color='tab:red', marker='o', linewidth=2, linestyle='--')
ax2.set_ylabel('PPL Degradation vs Baseline (%)', color='tab:red', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 图例与标题
plt.title(f'Effect of Top-K Experts on Model Performance\n(Model: Qwen1.5-MoE-A2.7B, Baseline K={DEFAULT_TOP_K})', fontsize=14)
plt.tight_layout()

# 保存图片
img_path = os.path.join(SAVE_DIR, "topk_performance_chart.png")
plt.savefig(img_path, dpi=300)
print(f"Saved Plot to: {img_path}")

print("\nExperiment Complete!")