from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Project root / imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.skvs_attn import (
    custom_skvs_attention_forward,
    GLOBAL_ATTN_STATE,
    set_skvs_config,
    SKVSConfig,
)

# ---------------------------------------------------------------------
# Metrics / timing utilities
# ---------------------------------------------------------------------
def _now_ms() -> float:
    return time.time() * 1000.0


@contextmanager
def wall_timer_ms():
    t0 = _now_ms()
    yield lambda: (_now_ms() - t0)


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


class RunningStats:
    """Collect per-step samples and report avg/p50/p95/p99."""
    def __init__(self):
        self.samples: List[float] = []

    def add(self, x: float):
        self.samples.append(float(x))

    def summary(self, prefix: str = "") -> Dict[str, float]:
        if not self.samples:
            return {f"{prefix}count": 0.0}
        arr = np.asarray(self.samples, dtype=np.float64)
        return {
            f"{prefix}count": float(arr.size),
            f"{prefix}avg_ms": float(arr.mean()),
            f"{prefix}p50_ms": float(np.percentile(arr, 50)),
            f"{prefix}p95_ms": float(np.percentile(arr, 95)),
            f"{prefix}p99_ms": float(np.percentile(arr, 99)),
            f"{prefix}min_ms": float(arr.min()),
            f"{prefix}max_ms": float(arr.max()),
        }


class CUDATimer:
    """GPU time using CUDA events (synchronized)."""
    def __init__(self, device: torch.device):
        self.device = device
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_evt.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_evt.record()

    def elapsed_ms(self) -> float:
        torch.cuda.synchronize(self.device)
        return float(self.start_evt.elapsed_time(self.end_evt))


@dataclass
class Metrics:
    # overall
    wall_total_ms: float = 0.0
    wall_prefill_ms: float = 0.0
    wall_decode_total_ms: float = 0.0

    # token counts
    prompt_tokens: int = 0
    gen_tokens: int = 0

    # GPU peak memory
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0

    # comm bytes (activation dispatch upper bound)
    comm_bytes_activation: int = 0

    # future KV comm bytes (real data-plane later)
    comm_bytes_kv: int = 0

    # expert offload traffic (H2D)
    expert_weight_h2d_bytes: int = 0

    # SKVS accounting only (not network bytes yet)
    skvs_accounted_bytes: int = 0

    # detailed time stats (ms)
    prefill_gpu_ms: float = 0.0
    decode_step_stats: Any = None  # RunningStats

    # MoE internals time stats (accumulate)
    moe_comm_gpu_ms: float = 0.0
    moe_expert_gpu_ms: float = 0.0
    moe_h2d_gpu_ms: float = 0.0
    rs_gpu_ms: float = 0.0

    # optional correctness
    next_token_agree: float = -1.0  # ratio, if computed

    def __post_init__(self):
        if self.decode_step_stats is None:
            self.decode_step_stats = RunningStats()

    def throughput_tok_s(self) -> float:
        total = self.prompt_tokens + self.gen_tokens
        return total / max(self.wall_total_ms / 1000.0, 1e-9)


# ---------------------------------------------------------------------
# Global routing context: cache predicted (weights, indices) for each layer.
# routing_cache[layer] = (weights [T,K], indices [T,K])
# ---------------------------------------------------------------------
class SparseContext:
    num_layers: int = 0
    routing_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def initialize(cls, num_layers: int) -> None:
        cls.num_layers = int(num_layers)
        cls.routing_cache.clear()

    @classmethod
    def reset(cls) -> None:
        cls.routing_cache.clear()


# ---------------------------------------------------------------------
# RS model
# ---------------------------------------------------------------------
class RoutingScoutMLP(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        return self.net(x)


# ---------------------------------------------------------------------
# Expert materialization / caching
# ---------------------------------------------------------------------
class ExpertWeightCache:
    """
    Per-layer GPU cache for expert weights (LRU).
    Stores tuples (gate_w, up_w, down_w) on GPU.
    Capacity is the number of experts cached (not bytes).
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._lru: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()

    def get(self, expert_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.capacity <= 0:
            return None
        item = self._lru.get(expert_id, None)
        if item is None:
            return None
        self._lru.move_to_end(expert_id, last=True)
        return item

    def put(self, expert_id: int, weights: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        if self.capacity <= 0:
            return
        if expert_id in self._lru:
            self._lru[expert_id] = weights
            self._lru.move_to_end(expert_id, last=True)
            return
        self._lru[expert_id] = weights
        self._lru.move_to_end(expert_id, last=True)
        while len(self._lru) > self.capacity:
            _, old = self._lru.popitem(last=False)
            del old


class OffloadedExpert:
    """
    Expert whose weights reside on CPU by default, optionally cached on GPU.
    """
    __slots__ = ("gate_cpu", "up_cpu", "down_cpu", "act", "dtype", "cache", "metrics")

    def __init__(
        self,
        expert_module: nn.Module,
        dtype: torch.dtype,
        cache: ExpertWeightCache,
        metrics: Metrics,
    ):
        self.dtype = dtype
        self.cache = cache
        self.metrics = metrics

        # keep weights on pinned CPU memory for async H2D
        self.gate_cpu = expert_module.gate_proj.weight.detach().to("cpu", dtype).contiguous().pin_memory()
        self.up_cpu = expert_module.up_proj.weight.detach().to("cpu", dtype).contiguous().pin_memory()
        self.down_cpu = expert_module.down_proj.weight.detach().to("cpu", dtype).contiguous().pin_memory()
        self.act = expert_module.act_fn

    def __call__(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        cached = self.cache.get(expert_id)
        if cached is not None:
            gate_w, up_w, down_w = cached
        else:
            with CUDATimer(x.device) as t_h2d:
                gate_w = self.gate_cpu.to(device=x.device, non_blocking=True)
                up_w = self.up_cpu.to(device=x.device, non_blocking=True)
                down_w = self.down_cpu.to(device=x.device, non_blocking=True)

            self.metrics.moe_h2d_gpu_ms += t_h2d.elapsed_ms()
            self.metrics.expert_weight_h2d_bytes += (
                tensor_nbytes(gate_w) + tensor_nbytes(up_w) + tensor_nbytes(down_w)
            )
            self.cache.put(expert_id, (gate_w, up_w, down_w))

        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        return F.linear(self.act(gate) * up, down_w)


class ResidentExpert(nn.Module):
    """Expert whose weights are resident on GPU (fast path)."""
    def __init__(self, expert_module: nn.Module, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("gate_w", expert_module.gate_proj.weight.detach().to(device, dtype).contiguous(), persistent=False)
        self.register_buffer("up_w", expert_module.up_proj.weight.detach().to(device, dtype).contiguous(), persistent=False)
        self.register_buffer("down_w", expert_module.down_proj.weight.detach().to(device, dtype).contiguous(), persistent=False)
        self.act = expert_module.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_w)
        up = F.linear(x, self.up_w)
        return F.linear(self.act(gate) * up, self.down_w)


# ---------------------------------------------------------------------
# SparseDistributedMoELayer: EP + RS routing + optional shared expert
# ---------------------------------------------------------------------
class SparseDistributedMoELayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        scp_map: Dict[str, int],
        rs_dir: Path,
        rank: int,
        local_rank: int,
        world_size: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        hf_experts_list: List[nn.Module],
        shared_expert: Optional[nn.Module],
        shared_expert_gate: Optional[nn.Module],
        metrics: Metrics,
        expert_offload: str,
        expert_cache: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.metrics = metrics
        self.dtype = dtype

        self.expert_offload = str(expert_offload)
        if self.expert_offload not in ("cpu", "gpu"):
            raise ValueError(f"expert_offload must be cpu|gpu, got {self.expert_offload}")

        self.shared_expert = shared_expert.to(self.device, self.dtype) if shared_expert is not None else None
        self.shared_expert_gate = shared_expert_gate.to(self.device, self.dtype) if shared_expert_gate is not None else None

        # RS models exist only once (layer 0 instantiates all layers' RS for routing cache)
        if self.layer_idx == 0:
            self.all_rs_models = nn.ModuleList(
                [RoutingScoutMLP(self.hidden_size, self.num_experts).to(self.device, self.dtype) for _ in range(SparseContext.num_layers)]
            )
            for l in range(SparseContext.num_layers):
                rs_path = rs_dir / f"rs_l{l}.pt"
                if rs_path.exists():
                    state = torch.load(rs_path, map_location="cpu")
                    self.all_rs_models[l].load_state_dict(state, strict=True)
            self.all_rs_models.eval()

        # expert placement map (expert_id -> rank)
        self.expert_to_node = torch.empty(self.num_experts, dtype=torch.long, device=self.device)
        self.local_expert_ids: List[int] = []

        global_offset = self.layer_idx * self.num_experts
        for eid in range(self.num_experts):
            gid = global_offset + eid
            node_id = int(scp_map.get(str(gid), eid % self.world_size)) % self.world_size
            self.expert_to_node[eid] = node_id
            if node_id == self.rank:
                self.local_expert_ids.append(eid)

        self._cache = ExpertWeightCache(expert_cache)

        if self.expert_offload == "cpu":
            self.local_experts_cpu: Dict[int, OffloadedExpert] = {}
            for eid in self.local_expert_ids:
                self.local_experts_cpu[eid] = OffloadedExpert(
                    hf_experts_list[eid],
                    dtype=self.dtype,
                    cache=self._cache,
                    metrics=self.metrics,
                )
            self.local_experts_gpu = None
        else:
            self.local_experts_gpu = nn.ModuleDict()
            for eid in self.local_expert_ids:
                self.local_experts_gpu[str(eid)] = ResidentExpert(hf_experts_list[eid], self.device, self.dtype)
            self.local_experts_cpu = {}

        # free original list references
        hf_experts_list.clear()

    @staticmethod
    def _add_comm_bytes(metrics: Metrics, t: torch.Tensor) -> None:
        metrics.comm_bytes_activation += tensor_nbytes(t)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = hidden_states.shape
        flat = hidden_states.view(-1, dim)
        n_tokens = flat.shape[0]

        # RS routing computed once at layer 0 for all layers; cached in SparseContext
        if self.layer_idx == 0:
            with CUDATimer(self.device) as t_rs:
                SparseContext.routing_cache.clear()
                with torch.no_grad():
                    for l in range(SparseContext.num_layers):
                        logits = self.all_rs_models[l](flat)  # [T, E]
                        topk_w, topk_i = torch.topk(logits, k=self.top_k, dim=-1)
                        weights = torch.softmax(topk_w, dim=-1)
                        SparseContext.routing_cache[l] = (weights, topk_i)
            self.metrics.rs_gpu_ms += t_rs.elapsed_ms()

        topk_weights, topk_indices = SparseContext.routing_cache[self.layer_idx]  # [T,K]

        shared_out = 0.0
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(flat) * torch.sigmoid(self.shared_expert_gate(flat))

        # Expand tokens by top-k
        expanded = flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, dim)  # [T*K, D]
        flat_i = topk_indices.reshape(-1)                                         # [T*K]
        flat_w = topk_weights.reshape(-1)                                         # [T*K]
        dest = self.expert_to_node[flat_i]                                        # [T*K]

        # Sort by destination for all_to_all
        sort_idx = torch.argsort(dest)
        send_hidden = expanded[sort_idx].contiguous()
        send_i = flat_i[sort_idx].contiguous()
        send_w = flat_w[sort_idx].contiguous()
        send_dest = dest[sort_idx].contiguous()

        # counts
        send_counts = torch.bincount(send_dest, minlength=self.world_size).to(dtype=torch.long)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        send_counts_list = send_counts.cpu().tolist()
        recv_counts_list = recv_counts.cpu().tolist()
        recv_total = int(sum(recv_counts_list))

        # ---- Correctness checks (debug / deadlock prevention) ----
        my_send_total = int(send_counts.sum().item())
        my_recv_total = int(recv_counts.sum().item())
        t_pair = torch.tensor([my_send_total, my_recv_total], device=self.device, dtype=torch.int64)
        g_pairs = [torch.empty_like(t_pair) for _ in range(self.world_size)]
        dist.all_gather(g_pairs, t_pair)
        vals = [(int(x[0].item()), int(x[1].item())) for x in g_pairs]
        sum_send = sum(v[0] for v in vals)
        sum_recv = sum(v[1] for v in vals)
        if sum_send != sum_recv:
            print(f"[R{self.rank}][L{self.layer_idx}] global mismatch sum_send={sum_send} sum_recv={sum_recv} vals={vals}", flush=True)
            raise RuntimeError("MoE all_to_all global totals mismatch")

        all_send = [torch.empty_like(send_counts) for _ in range(self.world_size)]
        dist.all_gather(all_send, send_counts)
        all_send_lists = [x.cpu().tolist() for x in all_send]
        for src in range(self.world_size):
            expected = all_send_lists[src][self.rank]
            got = recv_counts_list[src]
            if expected != got:
                print(f"[R{self.rank}][L{self.layer_idx}] split mismatch from src {src}: expected {expected} got {got}", flush=True)
                raise RuntimeError("MoE all_to_all split mismatch -> will deadlock")
        # ---------------------------------------------------------

        # recv buffers
        recv_hidden = torch.empty((recv_total, dim), dtype=hidden_states.dtype, device=self.device)
        recv_i = torch.empty((recv_total,), dtype=send_i.dtype, device=self.device)
        recv_w = torch.empty((recv_total,), dtype=send_w.dtype, device=self.device)

        # all_to_all payloads
        with CUDATimer(self.device) as t_comm:
            dist.all_to_all_single(recv_hidden, send_hidden, recv_counts_list, send_counts_list)
            dist.all_to_all_single(recv_i, send_i, recv_counts_list, send_counts_list)
            dist.all_to_all_single(recv_w, send_w, recv_counts_list, send_counts_list)
        self.metrics.moe_comm_gpu_ms += t_comm.elapsed_ms()

        # comm accounting (upper bound): sent payload sizes
        self._add_comm_bytes(self.metrics, send_hidden)
        self._add_comm_bytes(self.metrics, send_i)
        self._add_comm_bytes(self.metrics, send_w)

        # local compute
        with CUDATimer(self.device) as t_expert:
            local_out = torch.zeros_like(recv_hidden)
            if self.expert_offload == "cpu":
                for eid in self.local_expert_ids:
                    mask = (recv_i == eid)
                    if mask.any():
                        out = self.local_experts_cpu[eid](recv_hidden[mask], expert_id=eid)
                        local_out[mask] = out * recv_w[mask].unsqueeze(1)
            else:
                for eid in self.local_expert_ids:
                    mask = (recv_i == eid)
                    if mask.any():
                        out = self.local_experts_gpu[str(eid)](recv_hidden[mask])
                        local_out[mask] = out * recv_w[mask].unsqueeze(1)
        self.metrics.moe_expert_gpu_ms += t_expert.elapsed_ms()

        # send results back
        gathered = torch.empty_like(send_hidden)
        with CUDATimer(self.device) as t_comm2:
            dist.all_to_all_single(gathered, local_out, send_counts_list, recv_counts_list)
        self.metrics.moe_comm_gpu_ms += t_comm2.elapsed_ms()

        # unsort back to original token*K order
        unsorted = torch.empty_like(gathered)
        unsorted[sort_idx] = gathered

        sparse_out = unsorted.view(n_tokens, self.top_k, dim).sum(dim=1)
        out = (shared_out + sparse_out).view(bsz, seq_len, dim)

        # SKVS accounting only (still not data-plane)
        dropped = int(GLOBAL_ATTN_STATE.get("step_dropped_bytes", 0))
        if dropped > 0:
            self.metrics.skvs_accounted_bytes += dropped

        return out


# ---------------------------------------------------------------------
# HF model introspection helpers
# ---------------------------------------------------------------------
def _get_experts_attr_name(hf_config: Any) -> str:
    return "mlp.experts"


def _get_experts_from_layer(layer: Any, attr_name: str):
    obj = layer
    for attr in attr_name.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def _set_attr_to_none(layer: Any, attr_name: str) -> None:
    parts = attr_name.split(".")
    obj = layer
    for a in parts[:-1]:
        obj = getattr(obj, a)
    setattr(obj, parts[-1], None)


# TODO hook (future)
def ensure_kv_on_node_data_plane(*args, **kwargs) -> None:
    return


# ---------------------------------------------------------------------
# Profiled greedy decode (no logic change vs your current version)
# ---------------------------------------------------------------------
@torch.no_grad()
def greedy_decode_profiled(model, input_ids, attention_mask, max_new_tokens, tokenizer, device, metrics: Metrics):
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    # --- prefill ---
    with wall_timer_ms() as wt:
        with CUDATimer(device) as gt:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    metrics.wall_prefill_ms = wt()
    metrics.prefill_gpu_ms = gt.elapsed_ms()

    past = outputs.past_key_values
    logits = outputs.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    generated = [next_token]
    decode_wall_total = 0.0

    cur_ids = next_token
    for _ in range(max_new_tokens - 1):
        with wall_timer_ms() as wt_step:
            with CUDATimer(device) as _gt_step:
                out = model(
                    input_ids=cur_ids,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past,
                    return_dict=True,
                )
        step_wall = wt_step()
        metrics.decode_step_stats.add(step_wall)
        decode_wall_total += step_wall

        past = out.past_key_values
        cur_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur_ids)

    metrics.wall_decode_total_ms = decode_wall_total

    gen_tokens = torch.cat(generated, dim=1)
    out_ids = torch.cat([input_ids, gen_tokens], dim=1)

    metrics.peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    metrics.peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    return out_ids


# ---------------------------------------------------------------------
# Result aggregation / saving
# ---------------------------------------------------------------------
def gather_and_save(metrics: Metrics, args, rank, world_size, device):
    t = lambda x, dtype: torch.tensor([x], device=device, dtype=dtype)

    comm = t(metrics.comm_bytes_activation, torch.long)
    h2d  = t(metrics.expert_weight_h2d_bytes, torch.long)
    kvb  = t(metrics.comm_bytes_kv, torch.long)

    moe_comm_ms = t(metrics.moe_comm_gpu_ms, torch.float64)
    moe_expert_ms = t(metrics.moe_expert_gpu_ms, torch.float64)
    moe_h2d_ms = t(metrics.moe_h2d_gpu_ms, torch.float64)
    rs_ms = t(metrics.rs_gpu_ms, torch.float64)

    prefill_gpu = t(metrics.prefill_gpu_ms, torch.float64)
    prefill_wall = t(metrics.wall_prefill_ms, torch.float64)
    total_wall = t(metrics.wall_total_ms, torch.float64)
    decode_wall = t(metrics.wall_decode_total_ms, torch.float64)

    peak_alloc = t(metrics.peak_allocated_mb, torch.float64)
    peak_rsvd  = t(metrics.peak_reserved_mb, torch.float64)

    # bytes are SUM; lat/mem use MAX
    dist.all_reduce(comm, op=dist.ReduceOp.SUM)
    dist.all_reduce(h2d, op=dist.ReduceOp.SUM)
    dist.all_reduce(kvb, op=dist.ReduceOp.SUM)

    dist.all_reduce(moe_comm_ms, op=dist.ReduceOp.SUM)
    dist.all_reduce(moe_expert_ms, op=dist.ReduceOp.SUM)
    dist.all_reduce(moe_h2d_ms, op=dist.ReduceOp.SUM)
    dist.all_reduce(rs_ms, op=dist.ReduceOp.SUM)

    dist.all_reduce(prefill_gpu, op=dist.ReduceOp.MAX)
    dist.all_reduce(prefill_wall, op=dist.ReduceOp.MAX)
    dist.all_reduce(total_wall, op=dist.ReduceOp.MAX)
    dist.all_reduce(decode_wall, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_alloc, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_rsvd, op=dist.ReduceOp.MAX)

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        step_summary = metrics.decode_step_stats.summary(prefix="decode_step_")

        record = {
            "run_tag": args.run_tag,
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "world_size": world_size,
            "model_path": args.model_path,
            "batch": args.batch,
            "max_new": args.max_new,
            "dtype": args.dtype,
            "expert_offload": args.expert_offload,
            "expert_cache": args.expert_cache,
            "skvs_enabled": bool(args.enable_skvs),

            "prompt_tokens": metrics.prompt_tokens,
            "gen_tokens": metrics.gen_tokens,

            "wall_total_ms": float(total_wall.item()),
            "wall_prefill_ms": float(prefill_wall.item()),
            "wall_decode_total_ms": float(decode_wall.item()),
            "prefill_gpu_ms": float(prefill_gpu.item()),
            "throughput_tok_s": metrics.throughput_tok_s(),

            "comm_mb_activation": float(comm.item()) / 1024**2,
            "comm_mb_kv": float(kvb.item()) / 1024**2,
            "expert_h2d_mb": float(h2d.item()) / 1024**2,

            "moe_comm_gpu_ms_sum": float(moe_comm_ms.item()),
            "moe_expert_gpu_ms_sum": float(moe_expert_ms.item()),
            "moe_h2d_gpu_ms_sum": float(moe_h2d_ms.item()),
            "rs_gpu_ms_sum": float(rs_ms.item()),

            "peak_allocated_mb": float(peak_alloc.item()),
            "peak_reserved_mb": float(peak_rsvd.item()),

            **step_summary,
        }

        # pretty print
        print("\n===== RUN SUMMARY (rank0) =====")
        for k in [
            "wall_total_ms", "wall_prefill_ms", "wall_decode_total_ms",
            "decode_step_avg_ms", "decode_step_p95_ms",
            "throughput_tok_s", "comm_mb_activation", "expert_h2d_mb",
            "moe_comm_gpu_ms_sum", "moe_expert_gpu_ms_sum", "moe_h2d_gpu_ms_sum",
            "peak_allocated_mb", "peak_reserved_mb"
        ]:
            if k in record:
                print(f"{k}: {record[k]}")
        print("================================\n")

        out_json = os.path.join(args.out_dir, f"run_{record['timestamp']}_{args.run_tag or 'default'}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        out_jsonl = os.path.join(args.out_dir, "runs.jsonl")
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    faulthandler.enable()

    def _mark(msg: str):
        print(msg, flush=True)

    _mark("[S0] start")

    ap = argparse.ArgumentParser()

    ap.add_argument("--model-path", type=str, default="/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
    ap.add_argument("--scp-path", type=str, default="/home/ztl/wyf/SPARSE/SPARSE_gpt/results/scp_placement.json")
    ap.add_argument("--rs-dir", type=str, default="/home/ztl/wyf/SPARSE/SPARSE_gpt/results/rs_models")

    ap.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=32)

    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    # SKVS config (accounting for now)
    ap.add_argument("--skvs-sink", type=int, default=4)
    ap.add_argument("--skvs-window", type=int, default=32)
    ap.add_argument("--skvs-ratio", type=float, default=0.6)
    ap.add_argument("--enable-skvs", action="store_true", help="Enable custom SKVS attention patch")

    ap.add_argument("--warmup", type=int, default=1)

    # Experts residency control
    ap.add_argument("--expert-offload", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--expert-cache", type=int, default=0, help="Number of experts cached on GPU per layer (LRU). 0 disables caching.")

    # output
    ap.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "eval_runs"))
    ap.add_argument("--run-tag", type=str, default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    # allocator setting (still works even if set late; better than nothing)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Disable fused SDP/FlashAttention to avoid CUDA illegal access while debugging SKVS
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _mark("[S1] cuda set_device ok")

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _mark(f"[S2] init_process_group ok rank={rank} local_rank={local_rank} world={world_size}")

    if args.enable_skvs:
        set_skvs_config(SKVSConfig(h2o_sink=args.skvs_sink, h2o_window=args.skvs_window, l0_guidance_ratio=args.skvs_ratio))
        _mark("[S3] set_skvs_config ok")
    else:
        _mark("[S3] SKVS disabled;skipping set_skvs_config")

    with open(args.scp_path, "r", encoding="utf-8") as f:
        scp_map: Dict[str, int] = json.load(f)

    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    hidden_size = int(getattr(hf_config, "hidden_size"))
    top_k = int(getattr(hf_config, "num_experts_per_tok", 4))

    SparseContext.initialize(num_layers)

    # metrics must exist BEFORE patching layers
    metrics = Metrics()

    # Load model on CPU first
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=hf_config,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    try:
        model.tie_weights()
    except Exception:
        pass

    experts_attr = _get_experts_attr_name(hf_config)

    # Patch each layer
    for i in range(num_layers):
        layer = model.model.layers[i]

        # Patch attention with SKVS attention forward (accounting-only here)
        if args.enable_skvs:
            layer.self_attn.forward = types.MethodType(custom_skvs_attention_forward, layer.self_attn)

        hf_experts_module = _get_experts_from_layer(layer, experts_attr)
        if hf_experts_module is not None:
            hf_experts_list = list(hf_experts_module)
            num_experts = len(hf_experts_list)

            shared_expert = getattr(layer.mlp, "shared_expert", None)
            shared_expert_gate = getattr(layer.mlp, "shared_expert_gate", None)

            layer.mlp = SparseDistributedMoELayer(
                layer_idx=i,
                scp_map=scp_map,
                rs_dir=Path(args.rs_dir),
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                hf_experts_list=hf_experts_list,
                shared_expert=shared_expert,
                shared_expert_gate=shared_expert_gate,
                metrics=metrics,
                expert_offload=args.expert_offload,
                expert_cache=args.expert_cache,
                dtype=dtype,
            )

            # release the original expert container to reduce CPU refs
            _set_attr_to_none(layer, experts_attr)

        # Move backbone parts needed on every rank (prototype behavior unchanged)
        layer.self_attn = layer.self_attn.to(device, dtype=dtype)
        layer.input_layernorm = layer.input_layernorm.to(device, dtype=dtype)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(device, dtype=dtype)

        gc.collect()
        torch.cuda.empty_cache()

    # Move embeddings / norm / head
    model.model.embed_tokens = model.model.embed_tokens.to(device, dtype=dtype)
    model.model.norm = model.model.norm.to(device, dtype=dtype)
    model.lm_head = model.lm_head.to(device, dtype=dtype)

    model.eval()

    _mark("[S4] before barrier")
    dist.barrier(device_ids=[local_rank])
    _mark("[S5] after barrier, before prompt encoding")

    prompts = [args.prompt] * args.batch
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    metrics.prompt_tokens = int(input_ids.numel())

    # Warmup (kept exactly as your original logic)
    for _ in range(args.warmup):
        SparseContext.reset()
        _ = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=min(8, args.max_new),
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        dist.barrier(device_ids=[local_rank])

    # Timed run
    SparseContext.reset()
    torch.cuda.synchronize(device)
    t0 = time.time()

    out = greedy_decode_profiled(
        model=model,
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=args.max_new,
        tokenizer=tokenizer,
        device=device,
        metrics=metrics,
    )

    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[local_rank])
    metrics.wall_total_ms = (time.time() - t0) * 1000.0

    metrics.gen_tokens = int(out.shape[1] - input_ids.shape[1]) * int(out.shape[0])

    gather_and_save(metrics, args, rank, world_size, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()