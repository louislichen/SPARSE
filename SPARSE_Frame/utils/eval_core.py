# utils/eval_core.py
from __future__ import annotations

import gc
import json
import os
import time
import types
from collections import OrderedDict
from contextlib import contextmanager
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
    return DistEnv(rank=dist.get_rank(), world=dist.get_world_size(), local_rank=local_rank, device=device)


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


def bcast_bool_from_rank0(flag_rank0: bool, env: DistEnv) -> bool:
    t = torch.tensor([1 if flag_rank0 else 0], device=env.device, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return bool(t.item())


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


# ---------------- routing cache + stats ----------------
class SparseContext:
    num_layers: int = 0
    routing_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # RS bank is stored globally so all layers can access it
    rs_bank: Optional[nn.ModuleList] = None
    rs_group: int = 3  # default: every 3 layers share the same input inL{0,3,6,...}

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
        cls.disable_stats()

    @classmethod
    def reset(cls) -> None:
        cls.routing_cache.clear()
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
        cls._seen_expert_masks = [torch.zeros((num_experts,), device=device, dtype=torch.bool) for _ in range(num_layers)]
        cls._entropy_sum = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(num_layers)]
        cls._token_count = [torch.zeros((), device=device, dtype=torch.int64) for _ in range(num_layers)]

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

        p = topk_weights.clamp_min(1e-12)
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
            per_layer[l] = {"coverage": used / max(E, 1), "entropy_avg": ent_avg, "unique_experts": float(used)}
        return per_layer if env.rank == 0 else None


def print_routing_stats_if_enabled(env: DistEnv, bundle, tag: str = "") -> None:
    per_layer = SparseContext.reduce_and_format_stats(env)
    if env.rank != 0 or per_layer is None:
        return
    hdr = f"[RoutingStats]{'['+tag+']' if tag else ''} mode={bundle.routing_mode} E={bundle.num_experts} K={bundle.top_k}"
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
            nn.Softmax(dim=-1),  # outputs probs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)


# LRU
class ExpertWeightCache:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._lru: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def get(self, expert_id: int):
        if self.capacity <= 0:
            return None
        v = self._lru.get(expert_id)
        if v is None:
            return None
        self._lru.move_to_end(expert_id, last=True)
        return v

    def put(self, expert_id: int, weights):
        if self.capacity <= 0:
            return
        self._lru[expert_id] = weights
        self._lru.move_to_end(expert_id, last=True)
        while len(self._lru) > self.capacity:
            _, old = self._lru.popitem(last=False)
            del old


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
        rs_group: int = 3,   # <= NEW: group size
        rs_debug: bool = False,
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

        self.routing_mode = str(routing_mode)
        if self.routing_mode not in ("router", "rs"):
            raise ValueError(f"routing_mode must be router|rs, got {self.routing_mode}")

        self.rs_group = int(rs_group)
        self.rs_debug = bool(rs_debug)

        self.gate = mlp_block.gate.to(self.device, dtype=self.dtype).eval()
        self.shared_expert = mlp_block.shared_expert.to(self.device, dtype=self.dtype).eval()
        self.shared_expert_gate = mlp_block.shared_expert_gate.to(self.device, dtype=self.dtype).eval()

        self.experts = mlp_block.experts
        self.experts_kind = "fused" if experts_are_fused(self.experts) else ("modulelist" if experts_are_modulelist(self.experts) else "unknown")
        if self.experts_kind == "unknown":
            raise RuntimeError("Unsupported experts container. Need fused (gate_up_proj/down_proj) or modulelist experts.")

        # placement expert->rank
        self.expert_to_rank = torch.empty(self.num_experts, dtype=torch.long, device=self.device)
        self.local_expert_ids: List[int] = []
        global_offset = self.layer_idx * self.num_experts
        for eid in range(self.num_experts):
            gid = global_offset + eid
            r = int(scp_map.get(str(gid), eid % self.world)) % self.world
            self.expert_to_rank[eid] = r
            if r == self.rank:
                self.local_expert_ids.append(eid)

        self.expert_offload = str(expert_offload)
        if self.expert_offload not in ("cpu", "gpu"):
            raise ValueError(f"expert_offload must be cpu|gpu, got {self.expert_offload}")
        self._cache = ExpertWeightCache(expert_cache)

        self.rs_prob_to_logit_eps = float(rs_prob_to_logit_eps)
        self.rs_require_complete = bool(rs_require_complete)

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
        # We build and load RS weights ONCE (layer0), then store to SparseContext.rs_bank.
        if self.routing_mode == "rs" and self.layer_idx == 0:
            hidden_dim = getattr(self.gate, "hidden_dim", None)
            if hidden_dim is None and hasattr(self.gate, "weight"):
                hidden_dim = int(self.gate.weight.shape[-1])
            if hidden_dim is None:
                raise RuntimeError("Cannot infer hidden_dim for RS.")

            rs_path = Path(rs_dir)
            bank = nn.ModuleList(
                [
                    RoutingScoutMLP(input_dim=hidden_dim, num_experts=self.num_experts).to(self.device, dtype=self.dtype)
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
                bank[l].load_state_dict(state, strict=True)

            bank.eval()
            SparseContext.rs_bank = bank  # share
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

    # ---------------- routing (router) ----------------
    def _routing_router(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(x)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topv = topv.to(dtype=x.dtype)
        return topv, topi.to(torch.long), router_logits

    # ---------------- routing (RS) ----------------
    def _compute_rs_cache_group(self, x: torch.Tensor, g: int) -> None:
        """
        Compute routing for layers [g, g+rs_group) using the SAME input x (post-attn of layer g).
        This matches your training scheme: rs_l{l}_inL{g}.pth
        """
        assert SparseContext.rs_bank is not None, "RS bank not initialized (did layer0 build it?)"
        bank = SparseContext.rs_bank
        end = min(g + self.rs_group, self.num_layers)

        if self.rs_debug and self.rank == 0:
            xx = x.detach()
            print(
                f"[RS dbg] trigger_group g={g} for layers [{g},{end}) | x mean/std/min/max "
                f"{float(xx.mean()):.4g} {float(xx.std()):.4g} {float(xx.min()):.4g} {float(xx.max()):.4g}",
                flush=True,
            )

        with torch.no_grad():
            for l in range(g, end):
                probs = bank[l](x)  # [T,E], already probs
                topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
                if self.norm_topk_prob:
                    topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                w = topv.to(dtype=x.dtype)
                SparseContext.routing_cache[l] = (w, topi.to(torch.long))
                SparseContext.update_stats(l, topi.to(torch.long), w)

    def _routing_rs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Grouped RS inference: at the first layer of each group (0,3,6,...) we populate cache for that group.
        """
        g = (self.layer_idx // self.rs_group) * self.rs_group
        if self.layer_idx == g:
            self._compute_rs_cache_group(x, g)

        if self.layer_idx not in SparseContext.routing_cache:
            raise RuntimeError(
                f"RS routing_cache miss at layer {self.layer_idx}. "
                f"Expected group trigger at g={g}. Check rs_group and forward order."
            )
        return SparseContext.routing_cache[self.layer_idx]

    # ----- fused weight helpers -----
    def _get_local_fused(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expert_offload == "gpu":
            return self.local_gate_up_gpu[eid], self.local_down_gpu[eid]
        cached = self._cache.get(eid)
        if cached is not None:
            return cached  # type: ignore[return-value]
        gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
        down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
        self._cache.put(eid, (gate_up, down))
        return gate_up, down

    # ----- modulelist weight helpers -----
    def _get_local_module_weights(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cache.get(eid)
        if cached is not None:
            return cached  # type: ignore[return-value]
        gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
        uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
        dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
        self._cache.put(eid, (gw, uw, dw))
        return gw, uw, dw

    def _get_fused_from_cpu_once(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_up = self.experts.gate_up_proj.detach().to("cpu", self.dtype).contiguous()[eid].pin_memory()
        down = self.experts.down_proj.detach().to("cpu", self.dtype).contiguous()[eid].pin_memory()
        return gate_up.to(self.device, non_blocking=True), down.to(self.device, non_blocking=True)

    def _get_module_from_cpu_once(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.expert_modules[eid]
        gw = m.gate_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
        uw = m.up_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
        dw = m.down_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
        return gw.to(self.device, non_blocking=True), uw.to(self.device, non_blocking=True), dw.to(self.device, non_blocking=True)

    def _experts_world1_exact(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        final = torch.zeros((T, H), device=x.device, dtype=x.dtype)
        expert_mask = torch.nn.functional.one_hot(idx.to(torch.long), num_classes=self.num_experts).permute(2, 1, 0)

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

        dist.all_to_all_single(recv_hidden, send_hidden, recv_counts_list, send_counts_list)
        dist.all_to_all_single(recv_i, send_i, recv_counts_list, send_counts_list)
        dist.all_to_all_single(recv_w, send_w, recv_counts_list, send_counts_list)

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

        shared = self.shared_expert(x)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * shared

        if self.routing_mode == "router":
            w, idx, router_logits = self._routing_router(x)
            SparseContext.update_stats(self.layer_idx, idx, w)
        else:
            w, idx = self._routing_rs(x)
            router_logits = None

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

    # For Qwen1.5MoE in your local implementation:
    # - num_experts/top_k are stored on the moe block (layer.mlp), not on gate Linear.
    top_k_cfg = int(getattr(hf_config, "num_experts_per_tok", 4))
    num_experts_cfg = int(getattr(hf_config, "num_experts", 0))

    SparseContext.initialize(num_layers, rs_group=int(cfg.rs_group))

    if cfg.enable_skvs:
        set_skvs_config(_SKVSConfig(h2o_sink=cfg.skvs_sink, h2o_window=cfg.skvs_window, l0_guidance_ratio=cfg.skvs_ratio))

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

    for i in range(num_layers):
        layer = model.model.layers[i]

        if cfg.enable_skvs:
            layer.self_attn.forward = types.MethodType(custom_skvs_attention_forward, layer.self_attn)

        if looks_like_sparse_moe_block(layer.mlp):
            sparse_layers += 1

            # Prefer reading from the MoE block itself (Qwen2MoeSparseMoeBlock)
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
            )
        else:
            layer.mlp = layer.mlp.to(env.device, dtype=dtype)

        layer.self_attn = layer.self_attn.to(env.device, dtype=dtype)
        layer.input_layernorm = layer.input_layernorm.to(env.device, dtype=dtype)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(env.device, dtype=dtype)

        gc.collect()
        torch.cuda.empty_cache()

    model.model.embed_tokens = model.model.embed_tokens.to(env.device, dtype=dtype)
    model.model.norm = model.model.norm.to(env.device, dtype=dtype)

    lm_head_on_cuda = (cfg.lm_head_device == "cuda")
    if lm_head_on_cuda:
        model.lm_head = model.lm_head.to(env.device, dtype=dtype)
    else:
        model.lm_head = model.lm_head.to("cpu", dtype=dtype).eval()
        orig_forward = model.forward

        def forward_cpu_lm_head(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["output_hidden_states"] = False
            out = orig_forward(*args, **kwargs)
            # NOTE: if output_hidden_states=False, out.hidden_states may be None in many HF versions.
            # Keep cpu-lm_head only if you already verified your specific version returns hidden_states.
            last_hidden = out.hidden_states[-1]
            last_hidden_cpu = last_hidden.to("cpu")
            logits_cpu = F.linear(last_hidden_cpu, self.lm_head.weight).float()
            out.logits = logits_cpu
            return out

        model.forward = types.MethodType(forward_cpu_lm_head, model)

    model.eval()

    if env.rank == 0:
        print(f"[Routing] mode={cfg.routing_mode}  (rs_dir={cfg.rs_dir})", flush=True)
        print(f"[Patch] sparse_moe_layers={sparse_layers}", flush=True)
        if cfg.routing_mode == "rs":
            print(f"[RS] group={cfg.rs_group} debug={cfg.rs_debug}", flush=True)

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
@torch.no_grad()
def greedy_decode_profiled(model, input_ids, attention_mask, max_new_tokens: int, env: DistEnv, metrics, measure_ttft: bool):
    model.eval()
    torch.cuda.reset_peak_memory_stats(env.device)
    with wall_timer_s() as wt:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        if measure_ttft:
            torch.cuda.synchronize(env.device)
    metrics.wall_prefill_ms = wt() * 1000.0

    past = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    generated = [next_token]
    decode_wall_total_s = 0.0
    cur_ids = next_token
    first_step_done = False

    for _ in range(max_new_tokens - 1):
        with wall_timer_s() as wt_step:
            out = model(input_ids=cur_ids, attention_mask=None, use_cache=True, past_key_values=past, return_dict=True)
            if measure_ttft and (not first_step_done):
                torch.cuda.synchronize(env.device)
        step_s = wt_step()
        decode_wall_total_s += step_s
        metrics.decode_step_stats.add_ms(step_s * 1000.0)

        past = out.past_key_values
        cur_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur_ids)

        if measure_ttft and (not first_step_done):
            metrics.wall_ttft_ms = metrics.wall_prefill_ms + step_s * 1000.0
            first_step_done = True

    metrics.wall_decode_ms = decode_wall_total_s * 1000.0
    out_ids = torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    metrics.peak_allocated_mb = torch.cuda.max_memory_allocated(env.device) / (1024**2)
    metrics.peak_reserved_mb = torch.cuda.max_memory_reserved(env.device) / (1024**2)
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