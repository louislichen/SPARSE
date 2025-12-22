import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeAttention, Qwen2MoeSparseMoeBlock

# ================= 配置 =================
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda"
DTYPE = torch.bfloat16

SEQ_LEN_PREFILL = 512
MAX_BATCH_SIZE = 512
STEP_SIZE = 16
REPEAT_STEPS = 10  # 稍微减少次数以加快测试
# =======================================

def get_components(model_id):
    print(f"Loading Config from: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # 强制开启 Flash Attention 2 以达到 Attn 的理想状态
    config._attn_implementation = "flash_attention_2"
    
    try:
        attn = Qwen2MoeAttention(config, layer_idx=0).to(DEVICE).to(DTYPE)
        ffn = Qwen2MoeSparseMoeBlock(config).to(DEVICE).to(DTYPE)
        # 提取Router和单个专家（用于理想态计算，保留Router）
        router = ffn.gate  # Qwen2-MoE的Router层（gate）
        single_expert = ffn.experts[0]  # 单个专家（剥离多专家搬运）
    except Exception as e:
        print(f"Error loading components: {e}")
        print("请确保安装了 flash-attn: pip install flash-attn --no-build-isolation")
        return None, None, None, None

    return config, attn, ffn, (router, single_expert)

def create_dummy_cache(batch_size, seq_len, config):
    cache = DynamicCache()
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    k = torch.zeros(batch_size, num_kv_heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    v = torch.zeros(batch_size, num_kv_heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    cache.update(k, v, layer_idx=0)
    return cache

def measure_kernel_dynamic(module, inputs, kwargs_factory, repeats):
    """通用测量函数：保证所有计算（含Router）被执行"""
    # Warmup：确保权重/缓存预热，避免首次执行偏差
    try:
        for _ in range(3):
            with torch.no_grad():
                module(inputs, **kwargs_factory())
        torch.cuda.synchronize()
    except:
        pass

    latencies = []
    mem_costs = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(repeats):
        kwargs = kwargs_factory()  # 每次刷新参数（如 Cache）
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        try:
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                # 完整执行前向：包含所有内置计算（如FFN的Router）
                module(inputs, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            
            latencies.append(start_event.elapsed_time(end_event))
            peak = torch.cuda.max_memory_allocated()
            mem_costs.append((peak - start_mem) / 1024**2)
            
        except torch.cuda.OutOfMemoryError:
            return None, None

    return np.median(latencies), np.median(mem_costs)

def measure_ideal_moe(module_parts, inputs, repeats, top_k):
    """理想MoE测量：包含Router计算 + 单专家计算（仅剥离多专家权重搬运）"""
    router, single_expert = module_parts
    latencies = []
    mem_costs = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup：预热Router和单专家
    for _ in range(3):
        with torch.no_grad():
            router(inputs)  # 执行Router（真实计算）
            single_expert(inputs)  # 执行单专家
    torch.cuda.synchronize()
    
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        try:
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                # Step1：执行Router（完整计算，不跳过）
                router_scores = router(inputs)
                # Step2：强制选Top-K=1（避免多专家权重搬运，仅保留Router开销）
                _, _ = torch.topk(router_scores, k=1, dim=-1)
                # Step3：单专家计算（对齐Top-K计算量）
                expert_out = single_expert(inputs)
                # 模拟Top-K聚合（无额外开销，仅对齐计算量）
                ideal_out = expert_out * top_k
            end_event.record()
            torch.cuda.synchronize()
            
            latencies.append(start_event.elapsed_time(end_event))
            peak = torch.cuda.max_memory_allocated()
            mem_costs.append((peak - start_mem) / 1024**2)
            
        except torch.cuda.OutOfMemoryError:
            return None, None

    return np.median(latencies), np.median(mem_costs)

def run_ideal_benchmark():
    config, attn, ffn, moe_parts = get_components(MODEL_ID)
    if not attn:
        return

    # 获取 MoE Top-K（保证理想态计算量对齐）
    top_k = getattr(config, "num_experts_per_tok", 2)
    print(f"\n🚀 开始测试（包含Router计算，Top-K={top_k}, FlashAttn=On）...")

    results = []

    for bs in range(1, MAX_BATCH_SIZE + 1, STEP_SIZE):
        
        # ================= PREFILL 阶段 =================
        hidden = torch.randn(bs, SEQ_LEN_PREFILL, config.hidden_size, device=DEVICE, dtype=DTYPE)
        pos = torch.arange(0, SEQ_LEN_PREFILL, device=DEVICE).unsqueeze(0).expand(bs, -1)
        cos, sin = attn.rotary_emb(hidden, pos)
        
        # 1. Attention (Flash)：完整执行
        lat_attn, mem_attn = measure_kernel_dynamic(
            attn, hidden,
            lambda: {"position_ids": pos, "position_embeddings": (cos, sin)},
            REPEAT_STEPS
        )
        
        # 2. 真实MoE-FFN：完整执行（包含Router + Top-K专家 + 权重搬运）
        lat_moe_real, mem_moe_real = measure_kernel_dynamic(
            ffn, hidden, lambda: {}, REPEAT_STEPS
        )
        
        # 3. 理想MoE-FFN：包含Router + 单专家（仅剥离多专家权重搬运）
        lat_moe_ideal, mem_moe_ideal = measure_ideal_moe(
            moe_parts, hidden, REPEAT_STEPS, top_k
        )

        if lat_attn is None or lat_moe_real is None or lat_moe_ideal is None:
            print(f"Batch {bs} OOM，停止测试")
            break

        # 吞吐量计算（总Token数 / 延迟（秒））
        total_tokens = bs * SEQ_LEN_PREFILL
        tput_attn = total_tokens / (lat_attn / 1000 + 1e-9)
        tput_moe_real = total_tokens / (lat_moe_real / 1000 + 1e-9)
        tput_moe_ideal = total_tokens / (lat_moe_ideal / 1000 + 1e-9)

        results.append({
            "Batch": bs, "Stage": "Prefill",
            "Attn_Lat": lat_attn, "MoE_Real_Lat": lat_moe_real, "MoE_Ideal_Lat": lat_moe_ideal,
            "Attn_Tput": tput_attn, "MoE_Real_Tput": tput_moe_real, "MoE_Ideal_Tput": tput_moe_ideal,
            "Attn_Mem": mem_attn, "MoE_Real_Mem": mem_moe_real, "MoE_Ideal_Mem": mem_moe_ideal
        })

        # ================= DECODE 阶段 =================
        hidden_dec = torch.randn(bs, 1, config.hidden_size, device=DEVICE, dtype=DTYPE)
        pos_dec = torch.tensor([[SEQ_LEN_PREFILL]], device=DEVICE).expand(bs, -1)
        cos, sin = attn.rotary_emb(hidden_dec, pos_dec)

        # 1. Attention (Decode w/ Cache)：完整执行
        def attn_decode_args():
            cache = create_dummy_cache(bs, 1024, config)  # 模拟历史KV Cache
            return {"position_ids": pos_dec, "position_embeddings": (cos, sin), 
                    "past_key_value": cache, "use_cache": True}
        
        lat_attn, mem_attn = measure_kernel_dynamic(attn, hidden_dec, attn_decode_args, REPEAT_STEPS)

        # 2. 真实MoE-FFN：完整执行（包含Router）
        lat_moe_real, mem_moe_real = measure_kernel_dynamic(ffn, hidden_dec, lambda: {}, REPEAT_STEPS)

        # 3. 理想MoE-FFN：包含Router + 单专家
        lat_moe_ideal, mem_moe_ideal = measure_ideal_moe(moe_parts, hidden_dec, REPEAT_STEPS, top_k)
        
        if lat_attn is None:
            break

        # 吞吐量计算
        total_tokens = bs * 1
        tput_attn = total_tokens / (lat_attn / 1000 + 1e-9)
        tput_moe_real = total_tokens / (lat_moe_real / 1000 + 1e-9)
        tput_moe_ideal = total_tokens / (lat_moe_ideal / 1000 + 1e-9)

        results.append({
            "Batch": bs, "Stage": "Decode",
            "Attn_Lat": lat_attn, "MoE_Real_Lat": lat_moe_real, "MoE_Ideal_Lat": lat_moe_ideal,
            "Attn_Tput": tput_attn, "MoE_Real_Tput": tput_moe_real, "MoE_Ideal_Tput": tput_moe_ideal,
            "Attn_Mem": mem_attn, "MoE_Real_Mem": mem_moe_real, "MoE_Ideal_Mem": mem_moe_ideal
        })

        if bs % 32 == 1:
            print(f"已处理 Batch Size: {bs}")

    return pd.DataFrame(results)

def plot_6_charts(df):
    if df.empty:
        print("无测试数据，跳过绘图")
        return
    
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))  # 保持原6张子图布局
    
    # 定义通用绘图函数（不新增Router可视化，仅保证开销计入）
    def plot_metric(row, col, stage, metric_suffix, title, ylabel):
        ax = axes[row, col]
        data = df[df["Stage"] == stage]
        
        # 1. Attention (Flash)
        sns.lineplot(data=data, x="Batch", y=f"Attn_{metric_suffix}", ax=ax,
                     label="Attn (Flash)", color="tab:blue", marker="o", linewidth=2)
        
        # 2. 真实MoE（包含Router + 权重搬运）
        sns.lineplot(data=data, x="Batch", y=f"MoE_Real_{metric_suffix}", ax=ax,
                     label="MoE FFN (Real)", color="tab:orange", marker="s", linewidth=2)
        
        # 3. 理想MoE（包含Router + 单专家，仅剥离权重搬运）
        sns.lineplot(data=data, x="Batch", y=f"MoE_Ideal_{metric_suffix}", ax=ax,
                     label="MoE FFN (Ideal)", color="tab:green", marker="x", linestyle="--", linewidth=2)
        
        # 填充真实与理想的Gap（代表权重搬运的带宽开销）
        if "Lat" in metric_suffix:
            ax.fill_between(data["Batch"], data[f"MoE_Ideal_{metric_suffix}"], data[f"MoE_Real_{metric_suffix}"],
                            color="red", alpha=0.1, label="Bandwidth Cost (Weight Move)")
        
        # 针对Decode Throughput放大FFN（核心诉求：让FFN曲线明显）
        if stage == "Decode" and metric_suffix == "Tput":
            # 限制纵轴范围，聚焦FFN区间（可根据实际数据调整）
            ax.set_ylim(0, 20000)
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)

    # 第1行：延迟（Latency）
    plot_metric(0, 0, "Prefill", "Lat", "Prefill Latency (ms)", "Time (ms)")
    plot_metric(0, 1, "Decode", "Lat", "Decode Latency (ms)", "Time (ms)")
    
    # 第2行：吞吐量（Throughput）
    plot_metric(1, 0, "Prefill", "Tput", "Prefill Throughput (Tokens/s)", "Tokens/s")
    plot_metric(1, 1, "Decode", "Tput", "Decode Throughput (FFN Focused)", "Tokens/s")
    
    # 第3行：动态显存（Memory）
    plot_metric(2, 0, "Prefill", "Mem", "Prefill Dynamic Memory (MB)", "MB")
    plot_metric(2, 1, "Decode", "Mem", "Decode Dynamic Memory (MB)", "MB")

    plt.tight_layout()
    plt.savefig("benchmark_moe_with_router.png", dpi=150)
    print("\n✅ 测试完成，结果已保存: benchmark_moe_with_router.png")

if __name__ == "__main__":
    df = run_ideal_benchmark()
    plot_6_charts(df)