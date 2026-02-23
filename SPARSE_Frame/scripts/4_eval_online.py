import os
import sys
import time
import gc
import json
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import Config
from utils.skvs_attn import custom_skvs_attention_forward, GLOBAL_ATTN_STATE

SCP_PATH = Config.SCP_PATH
RS_WEIGHTS_DIR = Config.RS_WEIGHTS_DIR
MODEL_PATH = Config.MODEL_PATH

class SparseContext:
    num_layers = 0
    routing_cache = {}

    @classmethod
    def initialize(cls, num_layers):
        cls.num_layers = num_layers
        cls.routing_cache.clear()

    @classmethod
    def reset(cls):
        cls.routing_cache.clear()

class MetricsCollector:
    def __init__(self):
        self.total_latency = 0.0
        self.total_tokens = 0
        self.total_comm_bytes = 0
        self.skvs_saved_bytes = 0.0
        self.start_time = 0

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self, tokens):
        torch.cuda.synchronize()
        self.total_latency += time.time() - self.start_time
        self.total_tokens += tokens

    def add_comm(self, tensor):
        self.total_comm_bytes += tensor.numel() * tensor.element_size()

    def add_skvs_savings(self, saved_bytes):
        self.skvs_saved_bytes += saved_bytes

GLOBAL_COLLECTOR = MetricsCollector()

class RoutingScoutMLP(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
    def forward(self, x):
        if len(x.shape) == 3: 
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)

class SlicedExpert(nn.Module):
    def __init__(self, expert_module, device):
        super().__init__()
        self.gate_proj = nn.Parameter(expert_module.gate_proj.weight.to(device, torch.float16), requires_grad=False)
        self.up_proj = nn.Parameter(expert_module.up_proj.weight.to(device, torch.float16), requires_grad=False)
        self.down_proj = nn.Parameter(expert_module.down_proj.weight.to(device, torch.float16), requires_grad=False)
        self.act = expert_module.act_fn

    def forward(self, x):
        gate = F.linear(x, self.gate_proj)
        up = F.linear(x, self.up_proj)
        return F.linear(self.act(gate) * up, self.down_proj)

class SparseDistributedMoELayer(nn.Module):
    def __init__(self, layer_idx, scp_map, rs_dir, rank, world_size, 
                 hidden_size, num_experts, top_k, hf_experts_list, shared_expert, shared_expert_gate):
        super().__init__()
        self.layer_idx = layer_idx
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.shared_expert = shared_expert.to(self.device, torch.float16) if shared_expert is not None else None
        self.shared_expert_gate = shared_expert_gate.to(self.device, torch.float16) if shared_expert_gate is not None else None

        if self.layer_idx == 0:
            self.all_rs_models = nn.ModuleList([
                RoutingScoutMLP(self.hidden_size, self.num_experts).to(self.device, torch.float16)
                for _ in range(SparseContext.num_layers)
            ])
            for l in range(SparseContext.num_layers):
                rs_path = os.path.join(rs_dir, f"rs_l{l}.pth")
                if os.path.exists(rs_path):
                    self.all_rs_models[l].load_state_dict(torch.load(rs_path, map_location=self.device))
            self.all_rs_models.eval()

        self.expert_to_node = torch.zeros(self.num_experts, dtype=torch.long, device=self.device)
        self.local_expert_ids = []
        
        global_offset = layer_idx * self.num_experts
        for eid in range(self.num_experts):
            gid = global_offset + eid
            node_id = int(scp_map.get(str(gid), eid % world_size)) % world_size
            self.expert_to_node[eid] = node_id
            if node_id == rank:
                self.local_expert_ids.append(eid)

        self.local_experts = nn.ModuleDict()
        for eid in range(self.num_experts):
            if eid in self.local_expert_ids:
                self.local_experts[str(eid)] = SlicedExpert(hf_experts_list[eid], self.device)
            hf_experts_list[eid] = None 

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        batch_size, seq_len, dim = original_shape
        flat_hidden = hidden_states.view(-1, dim)
        num_tokens = flat_hidden.shape[0]

        if self.layer_idx == 0:
            SparseContext.routing_cache.clear()
            with torch.no_grad():
                for l in range(SparseContext.num_layers):
                    logits = self.all_rs_models[l](flat_hidden)
                    topk_w, topk_i = torch.topk(logits, k=self.top_k, dim=-1)
                    SparseContext.routing_cache[l] = (torch.softmax(topk_w, dim=-1), topk_i)
        
        topk_weights, topk_indices = SparseContext.routing_cache[self.layer_idx]
        
        shared_out = 0.0
        if self.shared_expert is not None:
            shared_out = self.shared_expert(flat_hidden) * torch.sigmoid(self.shared_expert_gate(flat_hidden))

        expanded_hidden = flat_hidden.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, dim)
        flat_i = topk_indices.reshape(-1)
        flat_w = topk_weights.reshape(-1)
        dest_nodes = self.expert_to_node[flat_i]

        sort_idx = torch.argsort(dest_nodes)
        send_hidden = expanded_hidden[sort_idx].contiguous()
        send_i = flat_i[sort_idx].contiguous()
        send_w = flat_w[sort_idx].contiguous()
        send_dest = dest_nodes[sort_idx].contiguous()

        send_counts = torch.bincount(send_dest, minlength=self.world_size).cpu().tolist()
        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=self.device)
        recv_counts_t = torch.empty_like(send_counts_t)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.cpu().tolist()

        recv_total = sum(recv_counts)
        recv_hidden = torch.empty(recv_total, dim, dtype=hidden_states.dtype, device=self.device)
        recv_i = torch.empty(recv_total, dtype=send_i.dtype, device=self.device)
        recv_w = torch.empty(recv_total, dtype=send_w.dtype, device=self.device)

        dist.all_to_all_single(recv_hidden, send_hidden, recv_counts, send_counts)
        dist.all_to_all_single(recv_i, send_i, recv_counts, send_counts)
        dist.all_to_all_single(recv_w, send_w, recv_counts, send_counts)

        GLOBAL_COLLECTOR.add_comm(send_hidden)

        local_out = torch.zeros_like(recv_hidden)
        for eid in self.local_expert_ids:
            mask = (recv_i == eid)
            if mask.any():
                computed = self.local_experts[str(eid)](recv_hidden[mask])
                local_out[mask] = computed * recv_w[mask].unsqueeze(1)

        gathered_out = torch.empty_like(send_hidden)
        dist.all_to_all_single(gathered_out, local_out, send_counts, recv_counts)

        unsorted_out = torch.zeros_like(gathered_out)
        unsorted_out[sort_idx] = gathered_out
        
        sparse_out = unsorted_out.view(num_tokens, self.top_k, dim).sum(dim=1)
        final_output = (shared_out + sparse_out).view(original_shape)
        
        if GLOBAL_ATTN_STATE["step_dropped_bytes"] > 0:
            GLOBAL_COLLECTOR.add_skvs_savings(GLOBAL_ATTN_STATE["step_dropped_bytes"])
            
        return final_output

def get_dynamic_config(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers", 24)
    hidden_size = getattr(config, "hidden_size", 2048)
    num_experts = getattr(config, "num_experts", getattr(config, "n_routed_experts", 60))
    top_k = getattr(config, "num_experts_per_tok", 4)
    
    expert_attr_name = "mlp.experts"
    if hasattr(config, "model_type") and "mixtral" in config.model_type.lower():
        expert_attr_name = "block.sparse_moe.experts"
        
    return config, num_layers, hidden_size, num_experts, top_k, expert_attr_name

def get_experts_from_layer(layer, attr_name):
    obj = layer
    for attr in attr_name.split('.'):
        if hasattr(obj, attr): 
            obj = getattr(obj, attr)
        else: 
            return None
    return obj

def set_experts_to_none(layer, attr_name):
    parts = attr_name.split('.')
    obj = layer
    for attr in parts[:-1]: 
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], None)

def main():
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    scp_map = {}
    if os.path.exists(SCP_PATH):
        with open(SCP_PATH, "r") as f: 
            scp_map = json.load(f)

    hf_config, num_layers, hidden_size, num_experts, top_k, expert_attr_name = get_dynamic_config(MODEL_PATH)
    SparseContext.initialize(num_layers)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=hf_config,
        dtype=torch.float16,
        device_map="cpu",  
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    for i in range(num_layers):
        layer = model.model.layers[i]
        
        layer.self_attn.forward = types.MethodType(custom_skvs_attention_forward, layer.self_attn)
        
        hf_experts_module = get_experts_from_layer(layer, expert_attr_name)
        if hf_experts_module is not None:
            hf_experts_list = [expert for expert in hf_experts_module]
            shared_expert = getattr(layer.mlp, 'shared_expert', None)
            shared_expert_gate = getattr(layer.mlp, 'shared_expert_gate', None)

            layer.mlp = SparseDistributedMoELayer(
                i, scp_map, RS_WEIGHTS_DIR, rank, world_size,
                hidden_size, num_experts, top_k, hf_experts_list, shared_expert, shared_expert_gate
            )

        set_experts_to_none(layer, expert_attr_name)
        
        layer.self_attn = layer.self_attn.to(rank)
        layer.input_layernorm = layer.input_layernorm.to(rank)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(rank)
        
        gc.collect()
        torch.cuda.empty_cache()

    model.model.embed_tokens = model.model.embed_tokens.to(rank)
    model.model.norm = model.model.norm.to(rank)
    model.lm_head = model.lm_head.to(rank)

    model.eval()
    dist.barrier()
    
    text = "The quick brown fox jumps over the lazy dog. Distributed computing enables scalable systems."
    input_ids = tokenizer([text] * 4, return_tensors="pt", padding=True).input_ids.to(rank)

    SparseContext.reset()
    GLOBAL_COLLECTOR.start()
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=32,
        use_cache=True, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    
    GLOBAL_COLLECTOR.stop(outputs.shape[1] * outputs.shape[0])

    if rank == 0:
        print("Latency:", GLOBAL_COLLECTOR.total_latency)
        print("Throughput:", GLOBAL_COLLECTOR.total_tokens / GLOBAL_COLLECTOR.total_latency)
        print("Base Comm MB:", GLOBAL_COLLECTOR.total_comm_bytes / 1024**2)
        print("SKVS Reduced Comm MB:", GLOBAL_COLLECTOR.skvs_saved_bytes / 1024**2)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()