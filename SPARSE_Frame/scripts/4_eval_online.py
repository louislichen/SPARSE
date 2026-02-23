import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# Config
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import Config

SCP_PATH = Config.SCP_PATH
MODEL_PATH = Config.MODEL_PATH


# =========================
# Metrics
# =========================

class MetricsCollector:
    def __init__(self):
        self.total_latency = 0.0
        self.total_tokens = 0
        self.total_comm_bytes = 0
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


GLOBAL_COLLECTOR = MetricsCollector()


# =========================
# Expert Module (Worker only)
# =========================

class SlicedExpert(nn.Module):
    def __init__(self, expert_module, device):
        super().__init__()
        self.gate_proj = expert_module.gate_proj.to(device)
        self.up_proj = expert_module.up_proj.to(device)
        self.down_proj = expert_module.down_proj.to(device)
        self.act = expert_module.act_fn

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act(gate) * up)


# =========================
# Centralized MoE Layer
# =========================

class CentralizedMoELayer(nn.Module):
    def __init__(self, original_mlp, layer_idx,
                 scp_map, rank, world_size, config):

        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # Router only on rank0
        if rank == 0:
            self.router = original_mlp.gate.to(self.device)
        else:
            self.router = None

        # expert → gpu map
        self.expert_to_gpu = {}
        global_offset = layer_idx * self.num_experts

        for eid in range(self.num_experts):
            gid = global_offset + eid
            gpu = scp_map[str(gid)]
            self.expert_to_gpu[eid] = gpu

        # Load only local experts
        self.local_experts = nn.ModuleDict()

        if rank != 0:
            raw_experts = original_mlp.experts
            for eid in range(self.num_experts):
                if self.expert_to_gpu[eid] == rank:
                    expert_module = raw_experts[eid]
                    self.local_experts[str(eid)] = SlicedExpert(
                        expert_module, self.device
                    )

        # free original experts
        original_mlp.experts = None

    # =========================
    # Rank 0 forward
    # =========================

    def forward_rank0(self, hidden):

        B, S, D = hidden.shape
        flat = hidden.view(-1, D)

        # Router
        logits = self.router(flat)
        topk_w, topk_i = torch.topk(logits, k=self.top_k, dim=-1)
        topk_w = torch.softmax(topk_w, dim=-1)

        flat_i = topk_i.reshape(-1)
        flat_w = topk_w.reshape(-1)
        expanded = flat.unsqueeze(1).expand(-1, self.top_k, -1)
        expanded = expanded.reshape(-1, D)

        # Dispatch buffers
        send_dict = {r: [] for r in range(1, self.world_size)}
        send_w_dict = {r: [] for r in range(1, self.world_size)}

        for idx, eid in enumerate(flat_i):
            gpu = self.expert_to_gpu[int(eid)]
            if gpu != 0:
                send_dict[gpu].append(expanded[idx])
                send_w_dict[gpu].append(flat_w[idx])

        # Send to workers
        for r in range(1, self.world_size):
            if len(send_dict[r]) == 0:
                size_tensor = torch.tensor([0], device=self.device)
                dist.send(size_tensor, dst=r)
                continue

            tokens = torch.stack(send_dict[r])
            weights = torch.stack(send_w_dict[r])

            size_tensor = torch.tensor([tokens.shape[0]], device=self.device)
            dist.send(size_tensor, dst=r)
            dist.send(tokens, dst=r)
            dist.send(weights, dst=r)

            GLOBAL_COLLECTOR.add_comm(tokens)
            GLOBAL_COLLECTOR.add_comm(weights)

        # Receive outputs
        outputs = []

        for r in range(1, self.world_size):
            size_tensor = torch.zeros(1, device=self.device)
            dist.recv(size_tensor, src=r)
            recv_size = int(size_tensor.item())

            if recv_size == 0:
                continue

            recv_tensor = torch.empty(
                recv_size, D,
                dtype=hidden.dtype,
                device=self.device
            )
            dist.recv(recv_tensor, src=r)

            outputs.append(recv_tensor)

        if len(outputs) == 0:
            return hidden

        merged = torch.cat(outputs, dim=0)
        merged = merged.view(B * S, self.top_k, D).sum(dim=1)
        return merged.view(B, S, D)

    # =========================
    # Worker forward
    # =========================

    def forward_worker(self):

        device = self.device

        size_tensor = torch.zeros(1, device=device)
        dist.recv(size_tensor, src=0)
        recv_size = int(size_tensor.item())

        if recv_size == 0:
            size_back = torch.tensor([0], device=device)
            dist.send(size_back, dst=0)
            return

        hidden = torch.empty(
            recv_size,
            self.hidden_size,
            dtype=torch.float16,
            device=device
        )
        weights = torch.empty(
            recv_size,
            dtype=torch.float16,
            device=device
        )

        dist.recv(hidden, src=0)
        dist.recv(weights, src=0)

        outputs = torch.zeros_like(hidden)

        for name, expert in self.local_experts.items():
            outputs += expert(hidden)

        outputs *= weights.unsqueeze(1)

        size_back = torch.tensor([outputs.shape[0]], device=device)
        dist.send(size_back, dst=0)
        dist.send(outputs, dst=0)


# =========================
# Main
# =========================

def main():

    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)

    with open(SCP_PATH) as f:
        scp_map = json.load(f)

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map={"": rank},
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # Replace MoE layers
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            layer.mlp = CentralizedMoELayer(
                layer.mlp,
                i,
                scp_map,
                rank,
                world_size,
                config
            )

    model.eval()

    if rank == 0:
        text = "The quick brown fox jumps over the lazy dog."
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(rank)

        GLOBAL_COLLECTOR.start()
        outputs = model(input_ids)
        GLOBAL_COLLECTOR.stop(input_ids.numel())

        print("Latency:", GLOBAL_COLLECTOR.total_latency)
        print("Throughput:",
              GLOBAL_COLLECTOR.total_tokens /
              GLOBAL_COLLECTOR.total_latency)
        print("Comm MB:",
              GLOBAL_COLLECTOR.total_comm_bytes / 1024**2)

    else:
        while True:
            for layer in model.model.layers:
                if isinstance(layer.mlp, CentralizedMoELayer):
                    layer.mlp.forward_worker()


    dist.destroy_process_group()


if __name__ == "__main__":
    main()