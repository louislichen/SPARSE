import os
import torch
import time
import torch.distributed as dist
from transformers import AutoConfig
import torch.nn.functional as F

# ====================== 工具函数：计算张量内存占用/形状 ======================
def get_tensor_info(tensor, name="tensor"):
    """返回张量的形状、元素数、内存占用(MB)"""
    if tensor is None:
        return {"name": name, "shape": None, "num_elements": 0, "memory_mb": 0.0}
    num_elements = tensor.numel()
    # 计算内存：元素数 × 每个元素的字节数（float16=2字节）
    memory_bytes = num_elements * tensor.element_size()
    memory_mb = memory_bytes / 1024 / 1024
    return {
        "name": name,
        "shape": list(tensor.shape),
        "num_elements": num_elements,
        "memory_mb": round(memory_mb, 2)
    }

def get_comm_volume(tensor, op_type="all_gather"):
    """计算通信量（字节）：all_gather/all_reduce的传输字节数"""
    if tensor is None:
        return 0
    world_size = dist.get_world_size()
    elem_bytes = tensor.element_size()
    if op_type == "all_gather":
        # all_gather：每个GPU发送tensor.size()字节，总通信量=单卡大小×world_size
        return tensor.numel() * elem_bytes * world_size
    elif op_type == "all_reduce":
        # all_reduce：总通信量=单卡大小×(world_size-1)
        return tensor.numel() * elem_bytes * (world_size - 1)
    else:
        return 0

# ====================== 手动实现Qwen RMSNorm ======================
class QwenRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

# ====================== 分布式初始化 ======================
def setup_distributed():
    if not dist.is_initialized():
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '4')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}")
world_size = dist.get_world_size()

# ====================== 加载配置 ======================
CONFIG_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
config = AutoConfig.from_pretrained(
    CONFIG_PATH,
    local_files_only=True,
    trust_remote_code=True
)

single_layer_config = {
    "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else 2048,          
    "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else 16,  
    "head_dim": (config.hidden_size // config.num_attention_heads) if hasattr(config, 'hidden_size') else 128,  
    "ffn_dim": config.intermediate_size if hasattr(config, 'intermediate_size') else 5120,        
    "num_experts": config.num_experts if hasattr(config, 'num_experts') else 60,          
    "num_local_experts": (config.num_experts // world_size) if hasattr(config, 'num_experts') else 15,  
    "num_experts_per_tok": 2,                   
    "dtype": torch.float16,
    "rms_norm_eps": config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6,
}

# ====================== Qwen Self-Attention（新增形状/大小测量） ======================
class QwenSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.dtype = config["dtype"]

        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=self.dtype)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=self.dtype)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=self.dtype)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=self.dtype)
        
        # 记录Attention模块的输入输出信息
        self.tensor_info = {}

    def forward(self, x):
        # 记录输入信息
        self.tensor_info["input"] = get_tensor_info(x, "attention_input")
        
        batch_size, seq_len, hidden_size = x.shape
        x = x.to(self.dtype)
        
        # Q/K/V投影
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 记录Q/K/V形状
        self.tensor_info["q"] = get_tensor_info(q, "q_tensor")
        self.tensor_info["k"] = get_tensor_info(k, "k_tensor")
        self.tensor_info["v"] = get_tensor_info(v, "v_tensor")
        
        # 注意力计算
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(self.dtype)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 输出投影
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        output = self.o_proj(attn_output)
        
        # 记录输出信息
        self.tensor_info["output"] = get_tensor_info(output, "attention_output")
        return output

# ====================== Qwen MoE FFN（新增通信量/通信起止点测量） ======================
class QwenMoEFFN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.dtype = config["dtype"]

        self.gate = torch.nn.Linear(config["hidden_size"], config["num_experts"], device=device, dtype=self.dtype)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(config["hidden_size"], config["ffn_dim"], device=device, dtype=self.dtype),
                torch.nn.GELU(),
                torch.nn.Linear(config["ffn_dim"], config["hidden_size"], device=device, dtype=self.dtype)
            ) for _ in range(config["num_local_experts"])
        ])
        
        # 记录MoE模块的张量信息+通信信息
        self.tensor_info = {}
        self.comm_info = {
            "all_gather": {"start": 0.0, "end": 0.0, "volume_bytes": 0},
            "all_reduce": {"start": 0.0, "end": 0.0, "volume_bytes": 0}
        }

    def forward(self, x):
        timers = {"gate": 0.0, "comm": 0.0, "expert": 0.0}
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size).to(self.dtype)
        
        # 记录MoE输入信息
        self.tensor_info["input"] = get_tensor_info(x, "moe_input")
        self.tensor_info["input_flat"] = get_tensor_info(x_flat, "moe_input_flat")

        # 门控选专家
        gate_start = time.time()
        gate_logits = self.gate(x_flat)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.config["num_experts_per_tok"], dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        timers["gate"] = time.time() - gate_start
        
        # 记录门控输出信息
        self.tensor_info["gate_logits"] = get_tensor_info(gate_logits, "gate_logits")
        self.tensor_info["top_k_indices"] = get_tensor_info(top_k_indices, "top_k_indices")

        # 跨卡通信：all_gather（记录起止点+通信量）
        comm_start = time.time()
        self.comm_info["all_gather"]["start"] = time.time()  # 精确到微秒
        all_indices = [torch.zeros_like(top_k_indices) for _ in range(self.world_size)]
        dist.all_gather(all_indices, top_k_indices)
        self.comm_info["all_gather"]["end"] = time.time()
        # 计算all_gather通信量
        self.comm_info["all_gather"]["volume_bytes"] = get_comm_volume(top_k_indices, "all_gather")
        
        local_mask = (top_k_indices >= self.local_rank * self.config["num_local_experts"]) & \
                     (top_k_indices < (self.local_rank + 1) * self.config["num_local_experts"])
        timers["comm"] += time.time() - comm_start

        # 本地专家计算
        expert_start = time.time()
        outputs = torch.zeros_like(x_flat, dtype=self.dtype, device=device)
        for i in range(self.config["num_experts_per_tok"]):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(-1)
            for local_expert_id in range(self.config["num_local_experts"]):
                global_expert_id = self.local_rank * self.config["num_local_experts"] + local_expert_id
                mask = (expert_idx == global_expert_id) & local_mask[:, i]
                if mask.any():
                    outputs[mask] += self.experts[local_expert_id](x_flat[mask]) * weight[mask]
        timers["expert"] = time.time() - expert_start
        
        # 跨卡通信：all_reduce（记录起止点+通信量）
        self.comm_info["all_reduce"]["start"] = time.time()
        dist.all_reduce(outputs, op=dist.ReduceOp.SUM)
        self.comm_info["all_reduce"]["end"] = time.time()
        # 计算all_reduce通信量
        self.comm_info["all_reduce"]["volume_bytes"] = get_comm_volume(outputs, "all_reduce")
        timers["comm"] += time.time() - expert_start

        # 恢复形状并记录输出信息
        output_reshaped = outputs.reshape(batch_size, seq_len, hidden_size)
        self.tensor_info["output_flat"] = get_tensor_info(outputs, "moe_output_flat")
        self.tensor_info["output"] = get_tensor_info(output_reshaped, "moe_output")
        
        return output_reshaped, timers

# ====================== 完整Transformer Block（汇总所有测量信息） ======================
class QwenMoEFullTransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = config["dtype"]

        self.self_attn = QwenSelfAttention(config)
        self.norm1 = QwenRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"], 
            device=device, dtype=self.dtype
        )
        
        self.moe_ffn = QwenMoEFFN(config)
        self.norm2 = QwenRMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"], 
            device=device, dtype=self.dtype
        )
        
        self.timers = {
            "attention": 0.0,
            "norm1": 0.0,
            "moe_gate": 0.0,
            "moe_comm": 0.0,
            "moe_expert": 0.0,
            "norm2": 0.0,
            "residual": 0.0,
            "total_layer": 0.0,
        }
        
        # 记录整个Block的输入输出信息
        self.block_tensor_info = {}

    def forward(self, x):
        total_start = time.time()
        x = x.to(self.dtype)
        
        # 记录Block输入信息
        self.block_tensor_info["input"] = get_tensor_info(x, "block_input")

        # Attention模块
        attn_start = time.time()
        x_norm1 = self.norm1(x)
        self.timers["norm1"] += time.time() - attn_start
        self.block_tensor_info["norm1_output"] = get_tensor_info(x_norm1, "norm1_output")
        
        attn_out = self.self_attn(x_norm1)
        self.timers["attention"] += time.time() - attn_start
        
        # 残差连接
        res1_start = time.time()
        x = x + attn_out
        self.timers["residual"] += time.time() - res1_start
        self.block_tensor_info["attention_residual_output"] = get_tensor_info(x, "attention_residual_output")

        # MoE FFN模块
        ffn_start = time.time()
        x_norm2 = self.norm2(x)
        self.timers["norm2"] += time.time() - ffn_start
        self.block_tensor_info["norm2_output"] = get_tensor_info(x_norm2, "norm2_output")
        
        moe_out, moe_timers = self.moe_ffn(x_norm2)
        self.timers["moe_gate"] += moe_timers["gate"]
        self.timers["moe_comm"] += moe_timers["comm"]
        self.timers["moe_expert"] += moe_timers["expert"]
        
        # 残差连接
        res2_start = time.time()
        x = x + moe_out
        self.timers["residual"] += time.time() - res2_start
        
        # 记录Block输出信息
        self.block_tensor_info["output"] = get_tensor_info(x, "block_output")

        self.timers["total_layer"] += time.time() - total_start
        return x

# ====================== 性能测试（新增所有测量项输出） ======================
@torch.no_grad()
def test_full_layer_performance():
    full_layer = QwenMoEFullTransformerBlock(single_layer_config).eval()
    
    # 推理输入配置（可调整）
    batch_size = 2
    seq_len = 32
    test_input = torch.randn(
        batch_size, seq_len, single_layer_config["hidden_size"],
        device=device, dtype=single_layer_config["dtype"]
    )

    # 预热
    for _ in range(5):
        _ = full_layer(test_input)
    
    # 重置计时器
    full_layer.timers = {k: 0.0 for k in full_layer.timers.keys()}
    
    # 正式测试
    test_times = 20
    for _ in range(test_times):
        _ = full_layer(test_input)
    
    # 计算平均耗时
    avg_timers = {k: v / test_times * 1000 for k, v in full_layer.timers.items()}

    # 仅主进程打印结果
    if local_rank == 0:
        print(f"\n=== Qwen-MoE 完整Transformer Block 性能分析（含张量/通信测量） ===")
        print(f"==================== 1. 输入配置 ====================")
        print(f"batch_size: {batch_size}, seq_len: {seq_len}, hidden_size: {single_layer_config['hidden_size']}")
        print(f"专家并行配置：{world_size}卡，每卡{single_layer_config['num_local_experts']}个专家")
        
        print(f"\n==================== 2. 各模块输入输出形状/大小 ====================")
        # Attention模块
        print(f"\n【Attention模块】")
        for k, v in full_layer.self_attn.tensor_info.items():
            print(f"  {v['name']}: 形状={v['shape']}, 元素数={v['num_elements']}, 内存={v['memory_mb']} MB")
        
        # MoE模块
        print(f"\n【MoE模块】")
        for k, v in full_layer.moe_ffn.tensor_info.items():
            print(f"  {v['name']}: 形状={v['shape']}, 元素数={v['num_elements']}, 内存={v['memory_mb']} MB")
        
        # 整个Block
        print(f"\n【完整Block】")
        for k, v in full_layer.block_tensor_info.items():
            print(f"  {v['name']}: 形状={v['shape']}, 元素数={v['num_elements']}, 内存={v['memory_mb']} MB")
        
        print(f"\n==================== 3. 通信量/通信起止点 ====================")
        # all_gather
        ag = full_layer.moe_ffn.comm_info["all_gather"]
        print(f"\n【all_gather通信】")
        print(f"  起始时间: {ag['start']:.6f} s")
        print(f"  结束时间: {ag['end']:.6f} s")
        print(f"  耗时: {(ag['end'] - ag['start'])*1000:.4f} ms")
        print(f"  通信量: {ag['volume_bytes']} 字节 ({ag['volume_bytes']/1024/1024:.2f} MB)")
        
        # all_reduce
        ar = full_layer.moe_ffn.comm_info["all_reduce"]
        print(f"\n【all_reduce通信】")
        print(f"  起始时间: {ar['start']:.6f} s")
        print(f"  结束时间: {ar['end']:.6f} s")
        print(f"  耗时: {(ar['end'] - ar['start'])*1000:.4f} ms")
        print(f"  通信量: {ar['volume_bytes']} 字节 ({ar['volume_bytes']/1024/1024:.2f} MB)")
        
        print(f"\n==================== 4. 模块耗时 ====================")
        print(f"1. Attention计算：{avg_timers['attention']:.2f} ms（占比：{avg_timers['attention']/avg_timers['total_layer']*100:.1f}%）")
        print(f"2. MoE门控选专家：{avg_timers['moe_gate']:.2f} ms（占比：{avg_timers['moe_gate']/avg_timers['total_layer']*100:.1f}%）")
        print(f"3. MoE跨卡通信：{avg_timers['moe_comm']:.2f} ms（占比：{avg_timers['moe_comm']/avg_timers['total_layer']*100:.1f}%）")
        print(f"4. MoE专家计算：{avg_timers['moe_expert']:.2f} ms（占比：{avg_timers['moe_expert']/avg_timers['total_layer']*100:.1f}%）")
        print(f"5. 归一化（RMSNorm）：{avg_timers['norm1']+avg_timers['norm2']:.2f} ms（占比：{(avg_timers['norm1']+avg_timers['norm2'])/avg_timers['total_layer']*100:.1f}%）")
        print(f"6. 残差连接：{avg_timers['residual']:.2f} ms（占比：{avg_timers['residual']/avg_timers['total_layer']*100:.1f}%）")
        print(f"\n【单层总耗时】：{avg_timers['total_layer']:.2f} ms")
        print(f"【单层显存占用】：{torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

# 执行测试
test_full_layer_performance()
dist.destroy_process_group()