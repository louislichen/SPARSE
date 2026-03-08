from __future__ import annotations

import gc
import json
import os
import time
import types
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.skvs_attn import (
    GLOBAL_ATTN_STATE,
    SKVSConfig as _SKVSConfig,
    custom_skvs_attention_forward,
    set_skvs_config,
)


# ---------------- dist helpers ----------------
@dataclass(frozen=True)
class DistEnv:
    rank: int
    world: int
    local_rank: int
    device: torch.device


def init_dist(backend: str = "nccl") -> DistEnv:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend=backend, init_method="env://")
    return DistEnv(
        rank=dist.get_rank(),
        world=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
    )


def barrier(env: DistEnv) -> None:
    dist.barrier(device_ids=[env.local_rank])


def destroy_dist() -> None:
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def all_reduce_sum_(x: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_reduce_max_(x: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def bcast_bool_from_rank0(flag_rank0: bool, env: DistEnv) -> bool:
    t = torch.tensor([1 if flag_rank0 else 0], device=env.device, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return bool(t.item())


def kv_owner_by_position(pos_idx: torch.Tensor, world: int) -> torch.Tensor:
    return torch.remainder(pos_idx, int(world)).to(torch.long)


# ---------------- sdpa mode ----------------
def set_sdpa_mode(flash: bool, mem_efficient: bool, math_sdp: bool) -> None:
    try:
        torch.backends.cuda.enable_flash_sdp(bool(flash))
        torch.backends.cuda.enable_mem_efficient_sdp(bool(mem_efficient))
        torch.backends.cuda.enable_math_sdp(bool(math_sdp))
    except Exception:
        pass


# ---------------- timing ----------------
def _now_s() -> float:
    return time.perf_counter()


@contextmanager
def wall_timer_s():
    t0 = _now_s()
    yield lambda: (_now_s() - t0)


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())

def kv_owner_by_position(pos_idx: torch.Tensor, world: int) -> torch.Tensor:
    return torch.remainder(pos_idx, world).to(torch.long)


# ---------------- perf stats ----------------
class RunningStats:
    def __init__(self):
        self.values_ms: List[float] = []

    def add_ms(self, v: float) -> None:
        self.values_ms.append(float(v))

    def reset(self) -> None:
        self.values_ms.clear()

    def summary(self, prefix: str = "") -> Dict[str, float]:
        if not self.values_ms:
            return {
                f"{prefix}avg_ms": 0.0,
                f"{prefix}p50_ms": 0.0,
                f"{prefix}p95_ms": 0.0,
                f"{prefix}p99_ms": 0.0,
                f"{prefix}max_ms": 0.0,
            }
        arr = np.array(self.values_ms, dtype=np.float64)
        return {
            f"{prefix}avg_ms": float(arr.mean()),
            f"{prefix}p50_ms": float(np.percentile(arr, 50)),
            f"{prefix}p95_ms": float(np.percentile(arr, 95)),
            f"{prefix}p99_ms": float(np.percentile(arr, 99)),
            f"{prefix}max_ms": float(arr.max()),
        }


@dataclass
class LayerPerf:
    expert_sum: float = 0.0
    expert_sumsq: float = 0.0
    expert_max: float = 0.0
    a2a_send_bytes: int = 0
    h2d_bytes: int = 0


@dataclass
class Metrics:
    comm_bytes_activation: int = 0
    comm_bytes_kv: int = 0

    expert_weight_h2d_bytes: int = 0
    expert_weight_h2d_prefetch_bytes: int = 0
    expert_weight_h2d_demand_bytes: int = 0

    skvs_accounted_bytes: int = 0

    prefetch_hit_count: int = 0
    prefetch_wasted_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    wall_prefill_ms: float = 0.0
    wall_decode_ms: float = 0.0
    wall_ttft_ms: float = 0.0
    wall_total_ms: float = 0.0

    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0

    prompt_tokens: int = 0
    gen_tokens: int = 0

    decode_step_stats: Optional[RunningStats] = None
    per_layer: Optional[Dict[int, LayerPerf]] = None

    def reset_for_repeat(self) -> None:
        self.comm_bytes_activation = 0
        self.comm_bytes_kv = 0

        self.expert_weight_h2d_bytes = 0
        self.expert_weight_h2d_prefetch_bytes = 0
        self.expert_weight_h2d_demand_bytes = 0

        self.skvs_accounted_bytes = 0

        self.prefetch_hit_count = 0
        self.prefetch_wasted_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        self.wall_prefill_ms = 0.0
        self.wall_decode_ms = 0.0
        self.wall_ttft_ms = 0.0
        self.wall_total_ms = 0.0
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0

        if self.decode_step_stats is not None:
            self.decode_step_stats.reset()

        if self.per_layer is not None:
            self.per_layer.clear()


# ---------------- routing cache + stats ----------------
class SparseContext:
    num_layers: int = 0
    routing_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    rs_bank: Optional[nn.ModuleList] = None
    rs_group: int = 3

    layer_blocks: Dict[int, "DistributedSparseMoeBlock"] = {}
    layer_attn_modules: Dict[int, nn.Module] = {}

    request_owner_rank: Optional[int] = None

    stats_enabled: bool = False
    stats_num_experts: int = 0
    _seen_expert_masks: Optional[List[torch.Tensor]] = None
    _entropy_sum: Optional[List[torch.Tensor]] = None
    _token_count: Optional[List[torch.Tensor]] = None

    @classmethod
    def initialize(cls, num_layers: int, rs_group: int = 3) -> None:
        cls.num_layers = int(num_layers)
        cls.rs_group = int(rs_group)
        cls.routing_cache.clear()
        cls.rs_bank = None
        cls.layer_blocks = {}
        cls.layer_attn_modules = {}
        cls.request_owner_rank = None
        cls.disable_stats()

    @classmethod
    def register_layer_block(cls, layer_idx: int, block: "DistributedSparseMoeBlock") -> None:
        cls.layer_blocks[int(layer_idx)] = block

    @classmethod
    def register_layer_attn(cls, layer_idx: int, attn_module: nn.Module) -> None:
        cls.layer_attn_modules[int(layer_idx)] = attn_module

    @classmethod
    def set_request_owner(cls, owner_rank: int) -> None:
        cls.request_owner_rank = int(owner_rank)
        for _, attn in cls.layer_attn_modules.items():
            setattr(attn, "_skvs_owner_rank", int(owner_rank))

    @classmethod
    def get_request_owner(cls) -> int:
        if cls.request_owner_rank is None:
            return -1
        return int(cls.request_owner_rank)

    @classmethod
    def clear_request_owner(cls) -> None:
        cls.request_owner_rank = None
        for _, attn in cls.layer_attn_modules.items():
            setattr(attn, "_skvs_owner_rank", -1)

    @classmethod
    def finalize_prefetch_all(cls) -> None:
        for _, blk in cls.layer_blocks.items():
            try:
                blk._finalize_prev_group_prefetch()
            except Exception:
                pass

    @classmethod
    def reset(cls) -> None:
        cls.routing_cache.clear()
        cls.clear_request_owner()

        if cls.stats_enabled:
            assert cls._seen_expert_masks is not None
            assert cls._entropy_sum is not None
            assert cls._token_count is not None
            for l in range(cls.num_layers):
                cls._seen_expert_masks[l].zero_()
                cls._entropy_sum[l].zero_()
                cls._token_count[l].zero_()

    @classmethod
    def enable_stats(cls, num_layers: int, num_experts: int, device: torch.device) -> None:
        cls.stats_enabled = True
        cls.stats_num_experts = int(num_experts)
        cls._seen_expert_masks = [
            torch.zeros((num_experts,), device=device, dtype=torch.bool) for _ in range(num_layers)
        ]
        cls._entropy_sum = [
            torch.zeros((), device=device, dtype=torch.float64) for _ in range(num_layers)
        ]
        cls._token_count = [
            torch.zeros((), device=device, dtype=torch.int64) for _ in range(num_layers)
        ]

    @classmethod
    def disable_stats(cls) -> None:
        cls.stats_enabled = False
        cls.stats_num_experts = 0
        cls._seen_expert_masks = None
        cls._entropy_sum = None
        cls._token_count = None

    @classmethod
    def update_stats(cls, layer_idx: int, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> None:
        if not cls.stats_enabled:
            return
        assert cls._seen_expert_masks is not None
        assert cls._entropy_sum is not None
        assert cls._token_count is not None

        idx = topk_indices.reshape(-1)
        cls._seen_expert_masks[layer_idx].scatter_(0, idx, True)

        p = topk_weights.float().clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)
        cls._entropy_sum[layer_idx] += ent.sum(dtype=torch.float64)
        cls._token_count[layer_idx] += torch.tensor(
            [topk_indices.shape[0]], device=idx.device, dtype=torch.int64
        ).squeeze(0)

    @classmethod
    def reduce_and_format_stats(cls, env: DistEnv) -> Optional[Dict[int, Dict[str, float]]]:
        if not cls.stats_enabled:
            return None
        assert cls._seen_expert_masks is not None
        assert cls._entropy_sum is not None
        assert cls._token_count is not None

        E = cls.stats_num_experts
        per_layer: Dict[int, Dict[str, float]] = {}
        for l in range(cls.num_layers):
            mask_int = cls._seen_expert_masks[l].to(dtype=torch.int32)
            all_reduce_sum_(mask_int)
            used = int((mask_int > 0).sum().item())

            ent = cls._entropy_sum[l].clone()
            cnt = cls._token_count[l].clone()
            all_reduce_sum_(ent)
            all_reduce_sum_(cnt)

            cnt_i = int(cnt.item())
            ent_avg = float(ent.item()) / max(cnt_i, 1)
            per_layer[l] = {
                "coverage": used / max(E, 1),
                "entropy_avg": ent_avg,
                "unique_experts": float(used),
            }
        return per_layer if env.rank == 0 else None
    
    @classmethod
    def get_request_owner(cls) -> int:
        if cls.request_owner_rank is None:
            return -1
        return int(cls.request_owner_rank)


def print_routing_stats_if_enabled(env: DistEnv, bundle, tag: str = "") -> None:
    per_layer = SparseContext.reduce_and_format_stats(env)
    if env.rank != 0 or per_layer is None:
        return

    hdr = (
        f"[RoutingStats]{'['+tag+']' if tag else ''} "
        f"mode={bundle.routing_mode} E={bundle.num_experts} K={bundle.top_k}"
    )
    print(hdr, flush=True)
    items = list(per_layer.items())

    def _fmt(l: int, d: Dict[str, float]) -> str:
        return f"L{l}: cov={d['coverage']:.3f} ent={d['entropy_avg']:.3f} uniq={int(d['unique_experts'])}"

    for l, d in items[:3]:
        print("  " + _fmt(l, d), flush=True)
    if len(items) > 6:
        print("  ...", flush=True)
    for l, d in items[-3:]:
        if l >= 3:
            print("  " + _fmt(l, d), flush=True)


# ---------------- RS model ----------------
class RoutingScoutMLP(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)


# ---------------- cache ----------------
@dataclass
class CacheEntry:
    payload: Any
    pinned: bool = False


class ExpertWeightCache:
    """
    group-aware cache:
    - pinned=True: 当前 group 预取得到，优先保留
    - eviction: 先淘汰最老的 non-pinned；如果全是 pinned，再淘汰最老的
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._lru: "OrderedDict[int, CacheEntry]" = OrderedDict()

    def get(self, expert_id: int):
        if self.capacity <= 0:
            return None
        entry = self._lru.get(expert_id)
        if entry is None:
            return None
        self._lru.move_to_end(expert_id, last=True)
        return entry.payload

    def has(self, expert_id: int) -> bool:
        return expert_id in self._lru

    def unpin_all(self) -> None:
        for k in list(self._lru.keys()):
            self._lru[k].pinned = False

    def put(self, expert_id: int, weights, pinned: bool = False) -> None:
        if self.capacity <= 0:
            return

        if expert_id in self._lru:
            old = self._lru[expert_id]
            old.payload = weights
            old.pinned = old.pinned or bool(pinned)
            self._lru.move_to_end(expert_id, last=True)
            return

        self._lru[expert_id] = CacheEntry(payload=weights, pinned=bool(pinned))
        self._lru.move_to_end(expert_id, last=True)

        while len(self._lru) > self.capacity:
            victim = None
            for k, v in self._lru.items():
                if not v.pinned:
                    victim = k
                    break
            if victim is None:
                victim = next(iter(self._lru.keys()))
            del self._lru[victim]


# ---------------- detect sparse moe block ----------------
def looks_like_sparse_moe_block(mlp: nn.Module) -> bool:
    return all(hasattr(mlp, a) for a in ["gate", "experts", "shared_expert", "shared_expert_gate"])


def experts_are_fused(experts: nn.Module) -> bool:
    return all(hasattr(experts, a) for a in ["gate_up_proj", "down_proj", "act_fn"])


def experts_are_modulelist(experts: Any) -> bool:
    try:
        xs = list(experts)
    except Exception:
        return False
    if len(xs) == 0:
        return False
    m0 = xs[0]
    return all(hasattr(m0, a) for a in ["gate_proj", "up_proj", "down_proj", "act_fn"])


# ---------------- distributed sparse moe block ----------------
class DistributedSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        mlp_block: nn.Module,
        scp_map: Dict[str, int],
        env: DistEnv,
        num_layers: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        expert_offload: str,
        expert_cache: int,
        routing_mode: str,
        rs_dir: str,
        rs_require_complete: bool,
        rs_prob_to_logit_eps: float,
        log_routing_stats: bool,
        rs_group: int = 3,
        rs_debug: bool = False,
        metrics: Optional[Metrics] = None,
        debug_a2a_checks: bool = False,
        collect_per_layer: bool = False,
        log_per_layer: bool = False,
        planner_mode: str = "none",
        prefetch_top_m: int = 1,
        rs_fallback_top1_min: float = 0.0,
        rs_fallback_entropy_max: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.env = env
        self.device = env.device
        self.rank = env.rank
        self.world = env.world

        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.dtype = dtype
        self.norm_topk_prob = bool(getattr(mlp_block, "norm_topk_prob", False))

        self.exec_routing_mode = str(routing_mode)
        if self.exec_routing_mode not in ("router", "rs"):
            raise ValueError(f"routing_mode must be router|rs, got {self.exec_routing_mode}")

        self.planner_mode = str(planner_mode)
        if self.planner_mode not in ("none", "rs_prefetch"):
            raise ValueError(f"planner_mode must be none|rs_prefetch, got {self.planner_mode}")

        self.rs_group = int(rs_group)
        self.rs_debug = bool(rs_debug)
        self.prefetch_top_m = int(prefetch_top_m)
        self.rs_fallback_top1_min = float(rs_fallback_top1_min)
        self.rs_fallback_entropy_max = float(rs_fallback_entropy_max)

        self.metrics = metrics
        self.debug_a2a_checks = bool(debug_a2a_checks)
        self.collect_per_layer = bool(collect_per_layer)
        self.log_per_layer = bool(log_per_layer)

        self.gate = mlp_block.gate.to(self.device, dtype=self.dtype).eval()
        self.shared_expert = mlp_block.shared_expert.to(self.device, dtype=self.dtype).eval()
        self.shared_expert_gate = mlp_block.shared_expert_gate.to(self.device, dtype=self.dtype).eval()

        self.experts = mlp_block.experts
        self.experts_kind = (
            "fused"
            if experts_are_fused(self.experts)
            else ("modulelist" if experts_are_modulelist(self.experts) else "unknown")
        )
        if self.experts_kind == "unknown":
            raise RuntimeError("Unsupported experts container. Need fused or modulelist experts.")

        # placement expert->rank
        self.expert_to_rank = torch.empty(self.num_experts, dtype=torch.long, device=self.device)
        self.local_expert_ids: List[int] = []
        self._local_eid_set = set()
        global_offset = self.layer_idx * self.num_experts
        for eid in range(self.num_experts):
            gid = global_offset + eid
            r = int(scp_map.get(str(gid), eid % self.world)) % self.world
            self.expert_to_rank[eid] = r
            if r == self.rank:
                self.local_expert_ids.append(eid)
                self._local_eid_set.add(eid)

        self.expert_offload = str(expert_offload)
        if self.expert_offload not in ("cpu", "gpu"):
            raise ValueError(f"expert_offload must be cpu|gpu, got {self.expert_offload}")
        self._cache = ExpertWeightCache(expert_cache)

        self.rs_prob_to_logit_eps = float(rs_prob_to_logit_eps)
        self.rs_require_complete = bool(rs_require_complete)

        self.prefetch_stream = (
            torch.cuda.Stream(device=self.device)
            if (self.expert_offload == "cpu" and torch.cuda.is_available())
            else None
        )
        self._prefetched_ids_this_group: set[int] = set()
        self._prefetch_used_ids_this_group: set[int] = set()

        # prepare local weights
        if self.experts_kind == "fused":
            self.act_fn = self.experts.act_fn
            self.local_gate_up_cpu: Dict[int, torch.Tensor] = {}
            self.local_down_cpu: Dict[int, torch.Tensor] = {}

            if self.expert_offload == "cpu":
                gate_up = self.experts.gate_up_proj.detach().to("cpu", dtype).contiguous()
                down = self.experts.down_proj.detach().to("cpu", dtype).contiguous()
                for eid in self.local_expert_ids:
                    self.local_gate_up_cpu[eid] = gate_up[eid].contiguous().pin_memory()
                    self.local_down_cpu[eid] = down[eid].contiguous().pin_memory()
            else:
                gate_up = self.experts.gate_up_proj.detach().to(self.device, dtype).contiguous()
                down = self.experts.down_proj.detach().to(self.device, dtype).contiguous()
                self.local_gate_up_gpu = {eid: gate_up[eid] for eid in self.local_expert_ids}
                self.local_down_gpu = {eid: down[eid] for eid in self.local_expert_ids}
        else:
            self.expert_modules: List[nn.Module] = list(self.experts)
            self.act_fn = self.expert_modules[0].act_fn
            if self.expert_offload == "gpu":
                for eid in self.local_expert_ids:
                    self.expert_modules[eid] = self.expert_modules[eid].to(self.device, dtype=self.dtype).eval()

            self.local_gate_cpu: Dict[int, torch.Tensor] = {}
            self.local_up_cpu: Dict[int, torch.Tensor] = {}
            self.local_down_cpu2: Dict[int, torch.Tensor] = {}
            if self.expert_offload == "cpu":
                for eid in self.local_expert_ids:
                    m = self.expert_modules[eid]
                    self.local_gate_cpu[eid] = m.gate_proj.weight.detach().to("cpu", dtype).contiguous().pin_memory()
                    self.local_up_cpu[eid] = m.up_proj.weight.detach().to("cpu", dtype).contiguous().pin_memory()
                    self.local_down_cpu2[eid] = m.down_proj.weight.detach().to("cpu", dtype).contiguous().pin_memory()

        # ---------------- RS BANK (global) ----------------
        need_rs_bank = (self.exec_routing_mode == "rs") or (self.planner_mode == "rs_prefetch")
        if need_rs_bank and self.layer_idx == 0:
            hidden_dim = getattr(self.gate, "hidden_dim", None)
            if hidden_dim is None and hasattr(self.gate, "weight"):
                hidden_dim = int(self.gate.weight.shape[-1])
            if hidden_dim is None:
                raise RuntimeError("Cannot infer hidden_dim for RS.")

            rs_path = Path(rs_dir)
            bank = nn.ModuleList(
                [
                    RoutingScoutMLP(input_dim=hidden_dim, num_experts=self.num_experts).to(
                        self.device, dtype=torch.float32
                    )
                    for _ in range(self.num_layers)
                ]
            )

            missing: List[str] = []
            for l in range(self.num_layers):
                in_layer = (l // self.rs_group) * self.rs_group
                p = rs_path / f"rs_l{l}_inL{in_layer}.pth"
                if not p.exists():
                    missing.append(str(p))
                    continue
                state = torch.load(str(p), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]
                if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                    state = state["model_state_dict"]
                bank[l].load_state_dict(state, strict=True)

            bank.eval()
            SparseContext.rs_bank = bank
            SparseContext.rs_group = self.rs_group

            if self.rank == 0:
                loaded = self.num_layers - len(missing)
                print(f"[RS] loaded {loaded}/{self.num_layers} from {rs_dir} (group={self.rs_group})", flush=True)
                if missing:
                    print(f"[RS] missing examples: {missing[:3]} ... (total {len(missing)})", flush=True)

            if self.rs_require_complete and missing:
                raise RuntimeError(f"RS weights missing ({len(missing)} files). Fix rs_dir or disable require_complete.")

        if log_routing_stats:
            SparseContext.enable_stats(num_layers=self.num_layers, num_experts=self.num_experts, device=self.device)

    # ----- perf helpers -----
    def _ensure_layer_perf(self) -> Optional[LayerPerf]:
        if self.metrics is None or self.metrics.per_layer is None:
            return None
        if self.layer_idx not in self.metrics.per_layer:
            self.metrics.per_layer[self.layer_idx] = LayerPerf()
        return self.metrics.per_layer[self.layer_idx]

    def _record_h2d_bytes(self, nbytes: int, *, is_prefetch: bool) -> None:
        if self.metrics is not None:
            self.metrics.expert_weight_h2d_bytes += int(nbytes)
            if is_prefetch:
                self.metrics.expert_weight_h2d_prefetch_bytes += int(nbytes)
            else:
                self.metrics.expert_weight_h2d_demand_bytes += int(nbytes)

        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.h2d_bytes += int(nbytes)

    def _record_activation_comm_bytes(self, nbytes: int) -> None:
        if self.metrics is not None:
            self.metrics.comm_bytes_activation += int(nbytes)
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.a2a_send_bytes += int(nbytes)

    def _record_local_expert_load(self, recv_i: torch.Tensor) -> None:
        lp = self._ensure_layer_perf()
        if lp is None:
            return

        for eid in self.local_expert_ids:
            cnt = int((recv_i == eid).sum().item())
            lp.expert_sum += float(cnt)
            lp.expert_sumsq += float(cnt * cnt)
            if cnt > lp.expert_max:
                lp.expert_max = float(cnt)

    # ---------------- router routing ----------------
    def _routing_router(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(x)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topv = topv.to(dtype=x.dtype)
        return topv, topi.to(torch.long), router_logits

    # ---------------- RS planner + routing ----------------
    def _finalize_prev_group_prefetch(self) -> None:
        if self.metrics is None:
            return
        wasted = self._prefetched_ids_this_group - self._prefetch_used_ids_this_group
        self.metrics.prefetch_wasted_count += len(wasted)

    def _begin_group_prefetch_cycle(self) -> None:
        self._finalize_prev_group_prefetch()
        self._cache.unpin_all()
        self._prefetched_ids_this_group.clear()
        self._prefetch_used_ids_this_group.clear()

    def _aggregate_local_prefetch_scores_for_this_layer(
        self,
        topi: torch.Tensor,
        topv: torch.Tensor,
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        flat_i = topi.reshape(-1)
        flat_w = topv.float().reshape(-1)
        for eid, wv in zip(flat_i.tolist(), flat_w.tolist()):
            if eid in self._local_eid_set:
                scores[eid] = scores.get(eid, 0.0) + float(wv)
        return scores

    def _prefetch_local_experts_for_this_layer(
        self,
        topi: torch.Tensor,
        topv: torch.Tensor,
    ) -> None:
        if self.planner_mode != "rs_prefetch":
            return
        if self.expert_offload != "cpu":
            return
        if self.prefetch_top_m <= 0:
            return
        if self._cache.capacity <= 0:
            return

        score_map = self._aggregate_local_prefetch_scores_for_this_layer(topi, topv)
        self._begin_group_prefetch_cycle()
        if len(score_map) == 0:
            return

        top_m = min(self.prefetch_top_m, self._cache.capacity, len(score_map))
        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_m]

        ctx = torch.cuda.stream(self.prefetch_stream) if self.prefetch_stream is not None else nullcontext()

        with ctx:
            for eid, _ in ranked:
                if self._cache.has(eid):
                    payload = self._cache.get(eid)
                    self._cache.put(eid, payload, pinned=True)
                    self._prefetched_ids_this_group.add(eid)
                    continue

                if self.experts_kind == "fused":
                    gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
                    down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
                    self._record_h2d_bytes(tensor_nbytes(gate_up) + tensor_nbytes(down), is_prefetch=True)
                    self._cache.put(eid, (gate_up, down), pinned=True)
                else:
                    gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
                    uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
                    dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
                    self._record_h2d_bytes(
                        tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw),
                        is_prefetch=True,
                    )
                    self._cache.put(eid, (gw, uw, dw), pinned=True)

                self._prefetched_ids_this_group.add(eid)

    def _compute_rs_cache_group(self, x: torch.Tensor, g: int, enable_prefetch: bool) -> None:
        """
        用 group 起始层输入 x，同时预测 [g, g+rs_group) 的 routing
        - 若 exec_routing_mode=rs，则结果写入 routing_cache
        - 若 planner_mode=rs_prefetch，则按目标层分别下发 prefetch plan
        """
        assert SparseContext.rs_bank is not None, "RS bank not initialized"
        bank = SparseContext.rs_bank
        end = min(g + self.rs_group, self.num_layers)

        if self.rs_debug and self.rank == 0:
            xx = x.detach()
            print(
                f"[RS dbg] trigger_group g={g} for layers [{g},{end}) | "
                f"x mean/std/min/max {float(xx.mean()):.4g} {float(xx.std()):.4g} "
                f"{float(xx.min()):.4g} {float(xx.max()):.4g}",
                flush=True,
            )

        group_topi: Dict[int, torch.Tensor] = {}
        group_topv: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x_rs = x.float()
            for l in range(g, end):
                probs = bank[l](x_rs).to(torch.float32)
                topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
                if self.norm_topk_prob:
                    topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                w = topv.to(dtype=x.dtype)
                i = topi.to(torch.long)

                group_topi[l] = i
                group_topv[l] = w

                if self.exec_routing_mode == "rs":
                    SparseContext.routing_cache[l] = (w, i)

        if enable_prefetch and self.planner_mode == "rs_prefetch":
            for l in range(g, end):
                blk = SparseContext.layer_blocks.get(l, None)
                if blk is None:
                    continue
                blk._prefetch_local_experts_for_this_layer(group_topi[l], group_topv[l])

    def _routing_rs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx not in SparseContext.routing_cache:
            raise RuntimeError(
                f"RS routing_cache miss at layer {self.layer_idx}. "
                f"This usually means the group-start layer did not recompute RS cache "
                f"for the current forward."
            )
        return SparseContext.routing_cache[self.layer_idx]

    def _rs_should_fallback(self, rs_w: torch.Tensor) -> bool:
        top1_mean = float(rs_w[:, 0].float().mean().item())
        p = rs_w.float().clamp_min(1e-12)
        ent = float((-(p * p.log()).sum(dim=-1)).mean().item())

        if self.rs_fallback_top1_min > 0.0 and top1_mean < self.rs_fallback_top1_min:
            return True
        if self.rs_fallback_entropy_max > 0.0 and ent > self.rs_fallback_entropy_max:
            return True
        return False

    def _estimate_request_owner_rank(self, idx: torch.Tensor, w: torch.Tensor) -> int:
        """
        第一版 owner policy:
        - per-request fixed
        - 基于 layer0 最终执行 routing 的 expert placement 做 weighted vote
        """
        flat_i = idx.reshape(-1).to(torch.long)
        flat_w = w.reshape(-1).float()
        flat_r = self.expert_to_rank[flat_i]

        rank_scores = torch.zeros((self.world,), device=flat_w.device, dtype=torch.float32)
        rank_scores.scatter_add_(0, flat_r, flat_w)
        owner_rank = int(torch.argmax(rank_scores).item())
        return owner_rank

    # ----- fused weight helpers -----
    def _maybe_wait_prefetch(self, eid: int) -> None:
        if self.prefetch_stream is not None and eid in self._prefetched_ids_this_group:
            torch.cuda.current_stream(self.device).wait_stream(self.prefetch_stream)

    def _get_local_fused(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expert_offload == "gpu":
            return self.local_gate_up_gpu[eid], self.local_down_gpu[eid]

        cached = self._cache.get(eid)
        if cached is not None:
            if self.metrics is not None:
                if eid in self._prefetched_ids_this_group:
                    self.metrics.prefetch_hit_count += 1
                    self._prefetch_used_ids_this_group.add(eid)
                else:
                    self.metrics.cache_hit_count += 1
            self._maybe_wait_prefetch(eid)
            return cached  # type: ignore[return-value]

        if self.metrics is not None:
            self.metrics.cache_miss_count += 1

        gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
        down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
        self._record_h2d_bytes(tensor_nbytes(gate_up) + tensor_nbytes(down), is_prefetch=False)
        self._cache.put(eid, (gate_up, down), pinned=False)
        return gate_up, down

    # ----- modulelist weight helpers -----
    def _get_local_module_weights(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cache.get(eid)
        if cached is not None:
            if self.metrics is not None:
                if eid in self._prefetched_ids_this_group:
                    self.metrics.prefetch_hit_count += 1
                    self._prefetch_used_ids_this_group.add(eid)
                else:
                    self.metrics.cache_hit_count += 1
            self._maybe_wait_prefetch(eid)
            return cached  # type: ignore[return-value]

        if self.metrics is not None:
            self.metrics.cache_miss_count += 1

        gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
        uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
        dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
        self._record_h2d_bytes(tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw), is_prefetch=False)
        self._cache.put(eid, (gw, uw, dw), pinned=False)
        return gw, uw, dw

    def _get_fused_from_cpu_once(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_up = self.experts.gate_up_proj.detach().to("cpu", self.dtype).contiguous()[eid].pin_memory()
        down = self.experts.down_proj.detach().to("cpu", self.dtype).contiguous()[eid].pin_memory()
        gate_up = gate_up.to(self.device, non_blocking=True)
        down = down.to(self.device, non_blocking=True)
        self._record_h2d_bytes(tensor_nbytes(gate_up) + tensor_nbytes(down), is_prefetch=False)
        return gate_up, down

    def _get_module_from_cpu_once(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.expert_modules[eid]
        gw = m.gate_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
        uw = m.up_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
        dw = m.down_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
        gw = gw.to(self.device, non_blocking=True)
        uw = uw.to(self.device, non_blocking=True)
        dw = dw.to(self.device, non_blocking=True)
        self._record_h2d_bytes(tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw), is_prefetch=False)
        return gw, uw, dw

    def _experts_world1_exact(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        final = torch.zeros((T, H), device=x.device, dtype=x.dtype)
        expert_mask = F.one_hot(idx.to(torch.long), num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx_k, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = x[None, top_x].reshape(-1, H)

            if self.experts_kind == "modulelist":
                if self.expert_offload == "gpu":
                    current_hidden_states = self.expert_modules[expert_idx](current_state)
                else:
                    if expert_idx in self.local_expert_ids:
                        gw, uw, dw = self._get_local_module_weights(expert_idx)
                    else:
                        gw, uw, dw = self._get_module_from_cpu_once(expert_idx)
                    gate = F.linear(current_state, gw)
                    up = F.linear(current_state, uw)
                    current_hidden_states = F.linear(self.act_fn(gate) * up, dw)
            else:
                if expert_idx in self.local_expert_ids:
                    gate_up, down = self._get_local_fused(expert_idx)
                else:
                    gate_up, down = self._get_fused_from_cpu_once(expert_idx)
                gate_up_out = F.linear(current_state, gate_up)
                gate, up = gate_up_out.chunk(2, dim=-1)
                current_hidden_states = F.linear(self.act_fn(gate) * up, down)

            current_hidden_states = current_hidden_states * w[top_x, idx_k, None]
            final.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        return final

    def _experts_ep(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        K = idx.shape[1]

        expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, H)
        flat_i = idx.reshape(-1).to(torch.long)
        flat_w = w.reshape(-1).to(expanded.dtype)

        dest = self.expert_to_rank[flat_i]
        sort_idx = torch.argsort(dest)

        send_hidden = expanded[sort_idx].contiguous()
        send_i = flat_i[sort_idx].contiguous()
        send_w = flat_w[sort_idx].contiguous()
        send_dest = dest[sort_idx].contiguous()

        send_counts = torch.bincount(send_dest, minlength=self.world).to(dtype=torch.long)
        recv_counts = torch.empty_like(send_counts)

        dist.all_to_all_single(recv_counts, send_counts)

        send_counts_list = send_counts.cpu().tolist()
        recv_counts_list = recv_counts.cpu().tolist()
        recv_total = int(sum(recv_counts_list))

        recv_hidden = torch.empty((recv_total, H), dtype=send_hidden.dtype, device=self.device)
        recv_i = torch.empty((recv_total,), dtype=send_i.dtype, device=self.device)
        recv_w = torch.empty((recv_total,), dtype=send_w.dtype, device=self.device)

        counts_bytes = tensor_nbytes(send_counts) + tensor_nbytes(recv_counts)
        dispatch_bytes = tensor_nbytes(send_hidden) + tensor_nbytes(send_i) + tensor_nbytes(send_w)
        gather_bytes = tensor_nbytes(send_hidden)
        self._record_activation_comm_bytes(counts_bytes + dispatch_bytes + gather_bytes)

        if self.debug_a2a_checks:
            if send_hidden.shape[0] != send_i.shape[0] or send_hidden.shape[0] != send_w.shape[0]:
                raise RuntimeError(
                    f"[rank{self.rank}] A2A shape mismatch: "
                    f"hidden={send_hidden.shape}, ids={send_i.shape}, w={send_w.shape}"
                )

        dist.all_to_all_single(recv_hidden, send_hidden, recv_counts_list, send_counts_list)
        dist.all_to_all_single(recv_i, send_i, recv_counts_list, send_counts_list)
        dist.all_to_all_single(recv_w, send_w, recv_counts_list, send_counts_list)

        self._record_local_expert_load(recv_i)

        local_out = torch.zeros_like(recv_hidden)
        for eid in self.local_expert_ids:
            m = (recv_i == eid)
            if not m.any():
                continue
            cur = recv_hidden[m]

            if self.experts_kind == "fused":
                gate_up, down = self._get_local_fused(eid)
                gate_up_out = F.linear(cur, gate_up)
                gate, up = gate_up_out.chunk(2, dim=-1)
                y = F.linear(self.act_fn(gate) * up, down)
            else:
                if self.expert_offload == "gpu":
                    y = self.expert_modules[eid](cur)
                else:
                    gw, uw, dw = self._get_local_module_weights(eid)
                    gate = F.linear(cur, gw)
                    up = F.linear(cur, uw)
                    y = F.linear(self.act_fn(gate) * up, dw)

            local_out[m] = y * recv_w[m].unsqueeze(1)

        gathered = torch.empty_like(send_hidden)
        dist.all_to_all_single(gathered, local_out, send_counts_list, recv_counts_list)

        unsorted = torch.empty_like(gathered)
        unsorted[sort_idx] = gathered
        return unsorted.view(T, K, H).sum(dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        x = hidden_states.view(-1, H)

        g = (self.layer_idx // self.rs_group) * self.rs_group

        # group start: recompute RS cache for THIS forward if needed
        if self.layer_idx == g and (self.exec_routing_mode == "rs" or self.planner_mode == "rs_prefetch"):
            self._compute_rs_cache_group(
                x,
                g,
                enable_prefetch=(self.planner_mode == "rs_prefetch"),
            )

        shared = self.shared_expert(x)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * shared

        if self.exec_routing_mode == "router":
            w, idx, router_logits = self._routing_router(x)
            SparseContext.update_stats(self.layer_idx, idx, w)
        else:
            w_rs, idx_rs = self._routing_rs(x)
            if self._rs_should_fallback(w_rs):
                w, idx, router_logits = self._routing_router(x)
                SparseContext.update_stats(self.layer_idx, idx, w)
            else:
                w, idx = w_rs, idx_rs
                router_logits = None
                SparseContext.update_stats(self.layer_idx, idx, w)

        # request-fixed owner: set once at layer 0 after final execution routing is known
        # this is early enough for layer1+ attention in the same forward.
        if self.layer_idx == 0 and SparseContext.request_owner_rank is None:
            owner_rank = self._estimate_request_owner_rank(idx, w)
            SparseContext.set_request_owner(owner_rank)

        if self.world == 1:
            expert_out = self._experts_world1_exact(x, idx, w)
        else:
            expert_out = self._experts_ep(x, idx, w)

        out = (expert_out + shared).view(B, S, H)
        return (out, router_logits)


# ---------------- build / patch ----------------
@dataclass(frozen=True)
class CoreBuildConfig:
    model_path: str
    scp_path: str
    rs_dir: str
    dtype: str = "float16"
    expert_offload: str = "cpu"
    expert_cache: int = 0
    enable_skvs: bool = False
    skvs_sink: int = 4
    skvs_window: int = 32
    skvs_ratio: float = 0.6
    routing_mode: str = "router"
    rs_require_complete: bool = True
    rs_prob_to_logit_eps: float = 1e-9
    log_routing_stats: bool = False
    lm_head_device: str = "cuda"
    rs_group: int = 3
    rs_debug: bool = False
    debug_a2a_checks: bool = False
    collect_per_layer: bool = False
    log_per_layer: bool = False
    planner_mode: str = "none"
    prefetch_top_m: int = 1
    rs_fallback_top1_min: float = 0.0
    rs_fallback_entropy_max: float = 0.0


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    hf_config: Any
    dtype: torch.dtype
    num_layers: int
    hidden_size: int
    top_k: int
    num_experts: int
    routing_mode: str


def build_model_and_tokenizer(cfg: CoreBuildConfig, env: DistEnv, metrics=None) -> ModelBundle:
    dtype = torch.float16 if cfg.dtype == "float16" else torch.bfloat16

    with open(cfg.scp_path, "r", encoding="utf-8") as f:
        scp_map: Dict[str, int] = json.load(f)

    hf_config = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    hidden_size = int(getattr(hf_config, "hidden_size"))

    top_k_cfg = int(getattr(hf_config, "num_experts_per_tok", 4))
    num_experts_cfg = int(getattr(hf_config, "num_experts", 0))

    SparseContext.initialize(num_layers, rs_group=int(cfg.rs_group))

    if cfg.enable_skvs:
        set_skvs_config(
            _SKVSConfig(
                h2o_sink=cfg.skvs_sink,
                h2o_window=cfg.skvs_window,
                l0_guidance_ratio=cfg.skvs_ratio,
            )
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        config=hf_config,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    sparse_layers = 0
    num_experts = num_experts_cfg
    top_k = top_k_cfg

    def _make_kv_owner_getter(world: int):
        def _getter(full_kv_len: int, device: torch.device, bsz: int) -> torch.Tensor:
            pos = torch.arange(full_kv_len, device=device, dtype=torch.long)
            owner = kv_owner_by_position(pos, world)
            return owner.unsqueeze(0).expand(bsz, -1).contiguous()
        return _getter

    for i in range(num_layers):
        layer = model.model.layers[i]

        if cfg.enable_skvs:
            orig_forward = layer.self_attn.forward
            layer.self_attn._skvs_orig_forward = orig_forward

            layer.self_attn.forward = types.MethodType(custom_skvs_attention_forward, layer.self_attn)

            # dist / layer hooks
            layer.self_attn._sparse_env_rank = env.rank
            layer.self_attn._sparse_env_world = env.world
            layer.self_attn._skvs_owner_rank = -1
            layer.self_attn._skvs_layer_idx = i

            # 第一版 SKVS 固定配置
            layer.self_attn._skvs_decode_only = True
            layer.self_attn._skvs_batch_limit = 1
            layer.self_attn._skvs_transport_impl = "all_gather_owner_bcast"
            layer.self_attn._skvs_token_owner_mode = "pos_mod_world"

            # 关键：KV ownership hook
            layer.self_attn._skvs_kv_owner_fn = kv_owner_by_position

            SparseContext.register_layer_attn(i, layer.self_attn)

        if looks_like_sparse_moe_block(layer.mlp):
            sparse_layers += 1

            E = int(getattr(layer.mlp, "num_experts", num_experts_cfg if num_experts_cfg > 0 else 0))
            K = int(getattr(layer.mlp, "top_k", top_k_cfg))

            if E <= 0:
                if experts_are_fused(layer.mlp.experts):
                    E = int(layer.mlp.experts.gate_up_proj.shape[0])
                elif experts_are_modulelist(layer.mlp.experts):
                    E = int(len(list(layer.mlp.experts)))
                else:
                    raise RuntimeError(f"Cannot infer num_experts at layer {i}")

            num_experts = E
            top_k = K

            layer.mlp = DistributedSparseMoeBlock(
                layer_idx=i,
                mlp_block=layer.mlp,
                scp_map=scp_map,
                env=env,
                num_layers=num_layers,
                num_experts=E,
                top_k=K,
                dtype=dtype,
                expert_offload=cfg.expert_offload,
                expert_cache=cfg.expert_cache,
                routing_mode=cfg.routing_mode,
                rs_dir=cfg.rs_dir,
                rs_require_complete=cfg.rs_require_complete,
                rs_prob_to_logit_eps=cfg.rs_prob_to_logit_eps,
                log_routing_stats=cfg.log_routing_stats,
                rs_group=int(cfg.rs_group),
                rs_debug=bool(cfg.rs_debug),
                metrics=metrics,
                debug_a2a_checks=bool(cfg.debug_a2a_checks),
                collect_per_layer=bool(cfg.collect_per_layer),
                log_per_layer=bool(cfg.log_per_layer),
                planner_mode=str(cfg.planner_mode),
                prefetch_top_m=int(cfg.prefetch_top_m),
                rs_fallback_top1_min=float(cfg.rs_fallback_top1_min),
                rs_fallback_entropy_max=float(cfg.rs_fallback_entropy_max),
            )
            SparseContext.register_layer_block(i, layer.mlp)
        else:
            layer.mlp = layer.mlp.to(env.device, dtype=dtype)

        layer.self_attn = layer.self_attn.to(env.device, dtype=dtype)
        layer.input_layernorm = layer.input_layernorm.to(env.device, dtype=dtype)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(env.device, dtype=dtype)

        gc.collect()
        torch.cuda.empty_cache()

    model.model.embed_tokens = model.model.embed_tokens.to(env.device, dtype=dtype)
    model.model.norm = model.model.norm.to(env.device, dtype=dtype)

    lm_head_on_cuda = cfg.lm_head_device == "cuda"
    if lm_head_on_cuda:
        model.lm_head = model.lm_head.to(env.device, dtype=dtype)
    else:
        model.lm_head = model.lm_head.to("cpu", dtype=dtype).eval()
        orig_forward = model.forward

        def forward_cpu_lm_head(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["output_hidden_states"] = False
            out = orig_forward(*args, **kwargs)
            last_hidden = out.hidden_states[-1]
            last_hidden_cpu = last_hidden.to("cpu")
            logits_cpu = F.linear(last_hidden_cpu, self.lm_head.weight).float()
            out.logits = logits_cpu
            return out

        model.forward = types.MethodType(forward_cpu_lm_head, model)

    model.eval()

    if env.rank == 0:
        print(
            f"[Routing] exec={cfg.routing_mode} planner={cfg.planner_mode} "
            f"(rs_dir={cfg.rs_dir})",
            flush=True,
        )
        print(f"[Patch] sparse_moe_layers={sparse_layers}", flush=True)
        if cfg.enable_skvs:
            print(
                f"[SKVS] decode_only=True batch_limit=1 transport=all_gather_owner_bcast "
                f"kv_owner=pos_mod_world",
                flush=True,
            )
        if cfg.routing_mode == "rs" or cfg.planner_mode == "rs_prefetch":
            print(
                f"[RS] group={cfg.rs_group} debug={cfg.rs_debug} "
                f"prefetch_top_m={cfg.prefetch_top_m} "
                f"fallback_top1_min={cfg.rs_fallback_top1_min} "
                f"fallback_entropy_max={cfg.rs_fallback_entropy_max}",
                flush=True,
            )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        hf_config=hf_config,
        dtype=dtype,
        num_layers=num_layers,
        hidden_size=hidden_size,
        top_k=int(top_k),
        num_experts=int(num_experts),
        routing_mode=cfg.routing_mode,
    )


# ---------------- perf decode ----------------
def _consume_skvs_accounting(metrics: Optional[Metrics]) -> None:
    if metrics is None:
        return

    dropped = int(GLOBAL_ATTN_STATE.get("forward_dropped_bytes", 0) or 0)
    kept_remote = int(GLOBAL_ATTN_STATE.get("forward_kept_remote_bytes", 0) or 0)

    if dropped > 0:
        metrics.skvs_accounted_bytes += dropped
    if kept_remote > 0:
        metrics.comm_bytes_kv += kept_remote

    GLOBAL_ATTN_STATE["forward_dropped_bytes"] = 0
    GLOBAL_ATTN_STATE["forward_kept_remote_bytes"] = 0


@torch.no_grad()
def greedy_decode_profiled(model, input_ids, attention_mask, max_new_tokens: int, env: DistEnv, metrics, measure_ttft: bool):
    model.eval()
    torch.cuda.reset_peak_memory_stats(env.device)

    # 每个 request 显式清理 owner，避免跨 request 污染
    SparseContext.clear_request_owner()

    with wall_timer_s() as wt:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        if measure_ttft:
            torch.cuda.synchronize(env.device)
    metrics.wall_prefill_ms = wt() * 1000.0
    _consume_skvs_accounting(metrics)

    past = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    generated = [next_token]
    decode_wall_total_s = 0.0
    cur_ids = next_token
    first_step_done = False

    for _ in range(max_new_tokens - 1):
        with wall_timer_s() as wt_step:
            out = model(
                input_ids=cur_ids,
                attention_mask=None,
                use_cache=True,
                past_key_values=past,
                return_dict=True,
            )
            if measure_ttft and (not first_step_done):
                torch.cuda.synchronize(env.device)

        step_s = wt_step()
        decode_wall_total_s += step_s
        if metrics.decode_step_stats is not None:
            metrics.decode_step_stats.add_ms(step_s * 1000.0)
        _consume_skvs_accounting(metrics)

        past = out.past_key_values
        cur_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur_ids)

        if measure_ttft and (not first_step_done):
            metrics.wall_ttft_ms = metrics.wall_prefill_ms + step_s * 1000.0
            first_step_done = True

    # finalize last group prefetch waste
    SparseContext.finalize_prefetch_all()

    metrics.wall_decode_ms = decode_wall_total_s * 1000.0
    out_ids = torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    metrics.peak_allocated_mb = torch.cuda.max_memory_allocated(env.device) / (1024 ** 2)
    metrics.peak_reserved_mb = torch.cuda.max_memory_reserved(env.device) / (1024 ** 2)

    # request 结束后再清一次，进一步避免污染
    SparseContext.clear_request_owner()
    return out_ids


# ---------------- quality kernel ----------------
@torch.no_grad()
def forward_nll_teacher_forcing(model, input_ids, attention_mask, valid_sample_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    logp = F.log_softmax(shift_logits, dim=-1)
    tgt_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    tok_mask = attention_mask[:, 1:].to(dtype=torch.float32)
    tok_mask = tok_mask * valid_sample_mask.to(dtype=torch.float32).unsqueeze(1)
    nll_sum = -(tgt_logp * tok_mask).sum(dtype=torch.float64)
    tok_cnt = tok_mask.sum().to(dtype=torch.int64)
    return nll_sum, tok_cnt