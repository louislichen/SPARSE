from __future__ import annotations

import gc
import inspect
import json
import os
import re
import sys
import time
import types
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from utils.skvs_attn import (
    GLOBAL_ATTN_STATE,
    SKVSConfig as _SKVSConfig,
    custom_skvs_attention_forward,
    migrate_sparse_kv_cache_for_group_change,
    set_skvs_config,
)

try:
    from utils.kernels import node_grouped_triton as _node_grouped_triton
except Exception:  # pragma: no cover - Triton is optional.
    _node_grouped_triton = None

from utils.runtime.config import env_flag_with_alias, env_value_with_alias, ers_config_from_env
from utils.hclm_cp_profiler import (
    NoOpHCLMCriticalPathProfiler as _HCLMCPNoOp,
    create_profiler_from_env as _hclm_cp_create_from_env,
    get_global_profiler as _hclm_cp_profiler,
    modeled_comm_ms as _hclm_cp_modeled_comm_ms,
    now_ns as _hclm_cp_now_ns,
    set_global_profiler as _hclm_cp_set_global,
)
from utils.ers_policy import (
    ERSPolicyError,
    find_rule_for_layer as _ers_find_rule_for_layer,
    normalize_candidate_mode as _ers_normalize_candidate_mode,
    parse_int_set as _ers_parse_int_set,
    parse_policy as _ers_parse_policy,
)

_DYNAMO_DISABLE = getattr(getattr(torch, "_dynamo", None), "disable", lambda fn: fn)


def _install_flash_attn_import_stub() -> bool:
    """Let HF remote model files import flash_attn when eager attention is used.

    DeepSeek-V2-Lite's modeling file imports flash_attn at module import time.
    Our benchmark path can run with eager/math attention, so installing the full
    flash-attn package is unnecessary for a smoke run.  The stub deliberately
    raises if a FlashAttention kernel is actually invoked.
    """
    if "flash_attn" in sys.modules:
        return False
    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is not None:
            return False
    except Exception:
        pass

    import importlib.machinery

    def _unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "flash_attn import stub was called. Set SPARSE_HF_ATTN_IMPLEMENTATION=eager "
            "or install flash-attn for flash attention execution."
        )

    def _missing_attr(_name: str) -> Any:
        return _unavailable

    flash_mod = types.ModuleType("flash_attn")
    flash_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_mod.flash_attn_func = _unavailable
    flash_mod.flash_attn_varlen_func = _unavailable
    flash_mod.flash_attn_qkvpacked_func = _unavailable
    flash_mod.flash_attn_varlen_qkvpacked_func = _unavailable
    flash_mod.flash_attn_with_kvcache = _unavailable
    flash_mod.__getattr__ = _missing_attr

    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    interface_mod.flash_attn_func = _unavailable
    interface_mod.flash_attn_varlen_func = _unavailable
    interface_mod.flash_attn_qkvpacked_func = _unavailable
    interface_mod.flash_attn_varlen_qkvpacked_func = _unavailable
    interface_mod.flash_attn_with_kvcache = _unavailable
    interface_mod.__getattr__ = _missing_attr

    bert_padding_mod = types.ModuleType("flash_attn.bert_padding")
    bert_padding_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding_mod.index_first_axis = _unavailable
    bert_padding_mod.pad_input = _unavailable
    bert_padding_mod.unpad_input = _unavailable
    bert_padding_mod.__getattr__ = _missing_attr

    layers_mod = types.ModuleType("flash_attn.layers")
    layers_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers", loader=None)
    layers_mod.__path__ = []
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers.rotary", loader=None)
    rotary_mod.apply_rotary_emb_func = _unavailable
    rotary_mod.__getattr__ = _missing_attr

    sys.modules["flash_attn"] = flash_mod
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding_mod
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod
    return True


def _hf_attn_implementation_from_env(default: str = "") -> str:
    return str(
        os.environ.get(
            "SPARSE_HF_ATTN_IMPLEMENTATION",
            os.environ.get("SPARSE_ATTENTION_IMPLEMENTATION", default),
        )
        or ""
    ).strip()


# ---------------- dist helpers ----------------
@dataclass(frozen=True)
class DistEnv:
    rank: int
    world: int
    local_rank: int
    device: torch.device


@dataclass(frozen=True)
class DistTopology:
    runtime_mode: str
    ep_size: int
    dp_size: int
    ep_rank: int
    dp_rank: int
    ep_group: Optional[dist.ProcessGroup]
    ep_global_ranks: Tuple[int, ...]

    @property
    def is_ep_leader(self) -> bool:
        return self.ep_rank == 0


@dataclass(frozen=True)
class ResolveResult:
    target_rank: int
    mode: str
    rr_owner: int
    used_scp_replica: bool
    fallback_to_rr: bool
    reject_reason: str = ""
    target_ready: bool = False
    route_cost: float = 0.0
    cache_ready: bool = False
    prefetch_ready: bool = False
    reject_not_ready_count: int = 0
    reject_no_comm_gain_count: int = 0


def _normalize_runtime_mode(runtime_mode: str) -> str:
    runtime_mode = str(runtime_mode)
    if runtime_mode in ("baseline", "canonical_ep_dp_serving_baseline"):
        return "baseline"
    raise ValueError(
        "This codebase now keeps only one runtime path: baseline. "
        f"Unsupported runtime_mode={runtime_mode}"
    )

def build_dist_topology(env: DistEnv, runtime_mode: str, ep_size: int) -> DistTopology:
    runtime_mode = _normalize_runtime_mode(runtime_mode)

    ep_size = int(ep_size)
    if ep_size <= 0 or env.world % ep_size != 0:
        raise ValueError(
            "baseline requires world % ep_size == 0, "
            f"got world={env.world}, ep_size={ep_size}"
        )

    dp_size = env.world // ep_size
    dp_rank = env.rank // ep_size
    ep_rank = env.rank % ep_size

    my_group = None
    my_group_ranks: Optional[Tuple[int, ...]] = None
    for dp in range(dp_size):
        ranks = tuple(range(dp * ep_size, (dp + 1) * ep_size))
        g = dist.new_group(ranks=list(ranks))
        if dp == dp_rank:
            my_group = g
            my_group_ranks = ranks

    if my_group is None or my_group_ranks is None:
        raise RuntimeError("Failed to build baseline EP subgroup for current rank")

    return DistTopology(
        runtime_mode=runtime_mode,
        ep_size=ep_size,
        dp_size=dp_size,
        ep_rank=ep_rank,
        dp_rank=dp_rank,
        ep_group=my_group,
        ep_global_ranks=my_group_ranks,
    )


def init_dist(backend: str = "nccl") -> DistEnv:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK not found. Please launch with torchrun.")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    visible_gpus = torch.cuda.device_count()

    if visible_gpus <= 0:
        raise RuntimeError("No CUDA device visible.")
    if local_rank < 0 or local_rank >= visible_gpus:
        raise RuntimeError(
            f"Invalid LOCAL_RANK={local_rank}, but visible_gpus={visible_gpus}. "
            f"Check CUDA_VISIBLE_DEVICES / torchrun launch."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend=backend, init_method="env://")

    env = DistEnv(
        rank=dist.get_rank(),
        world=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
    )

    _print_all_ranks(
        f"[DIST_INIT] world={env.world} world_env={world_size_env} "
        f"local_rank={env.local_rank} device={env.device}"
    )

    dist.barrier(device_ids=[env.local_rank])
    return env

def _print_all_ranks(msg: str) -> None:
    if dist.is_available() and dist.is_initialized():
        print(f"[rank={dist.get_rank()}] {msg}", flush=True)
    else:
        print(msg, flush=True)


def _gather_floats(v: float, device: torch.device) -> List[float]:
    t = torch.tensor([float(v)], device=device, dtype=torch.float32)
    outs = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(outs, t)
    return [float(x.item()) for x in outs]

def _owner_group_log_enabled() -> bool:
    return os.environ.get("SPARSE_GROUPSTART_LOG", "0").strip().lower() not in ("0", "false", "no", "off")

def _owner_group_log_all_ranks() -> bool:
    return os.environ.get("SPARSE_GROUPSTART_LOG_ALL_RANKS", "0").strip().lower() not in ("0", "false", "no", "off")

def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return float(default)


def _parse_int_override_map(raw: str, *, name: str) -> Dict[int, int]:
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
    if not parts:
        return {}
    out: Dict[int, int] = {}
    if any(":" in p for p in parts):
        for part in parts:
            if ":" not in part:
                raise ValueError(f"{name} mixes indexed entries with bare values: {raw!r}")
            key, value = part.split(":", 1)
            out[int(key.strip())] = int(value.strip())
    else:
        for idx, value in enumerate(parts):
            out[int(idx)] = int(value)
    for key, value in out.items():
        if int(value) < 0:
            raise ValueError(f"{name} has negative cache capacity at key={key}: {value}")
    return out


def _resolve_rank_expert_cache_capacity(
    base_capacity: int,
    *,
    global_rank: int,
    comm_rank: int,
    world: int,
) -> int:
    comm_raw = os.environ.get("SPARSE_EXPERT_CACHE_BY_COMM_RANK", "")
    rank_raw = os.environ.get("SPARSE_EXPERT_CACHE_BY_RANK", os.environ.get("EXPERT_CACHE_BY_RANK", ""))
    node_raw = os.environ.get("SPARSE_EXPERT_CACHE_BY_NODE", "")

    comm_map = _parse_int_override_map(comm_raw, name="SPARSE_EXPERT_CACHE_BY_COMM_RANK")
    if comm_map:
        if int(comm_rank) not in comm_map:
            raise ValueError(
                f"SPARSE_EXPERT_CACHE_BY_COMM_RANK does not define comm_rank={comm_rank}; "
                f"available={sorted(comm_map)}"
            )
        return int(comm_map[int(comm_rank)])

    rank_map = _parse_int_override_map(rank_raw, name="SPARSE_EXPERT_CACHE_BY_RANK")
    if rank_map:
        if int(global_rank) not in rank_map:
            raise ValueError(
                f"SPARSE_EXPERT_CACHE_BY_RANK does not define global rank={global_rank}; "
                f"available={sorted(rank_map)}"
            )
        return int(rank_map[int(global_rank)])

    node_map = _parse_int_override_map(node_raw, name="SPARSE_EXPERT_CACHE_BY_NODE")
    if node_map:
        logical_nodes = _parse_logical_node_map(int(world))
        if int(global_rank) < 0 or int(global_rank) >= len(logical_nodes):
            raise ValueError(f"rank={global_rank} out of logical node map range={len(logical_nodes)}")
        node = int(logical_nodes[int(global_rank)])
        if node not in node_map:
            raise ValueError(
                f"SPARSE_EXPERT_CACHE_BY_NODE does not define node={node} for rank={global_rank}; "
                f"available={sorted(node_map)}"
            )
        return int(node_map[node])

    return max(0, int(base_capacity))


def _stable_sample_hit(*parts: Any, rate: float) -> bool:
    rate = float(rate)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    h = 1469598103934665603
    for part in parts:
        try:
            s = str(int(part))
        except Exception:
            s = str(part)
        for ch in s:
            h ^= ord(ch)
            h = (h * 1099511628211) & ((1 << 64) - 1)
        h ^= 0x9E3779B97F4A7C15
        h = (h * 1099511628211) & ((1 << 64) - 1)
    return (h % 1_000_000) < int(rate * 1_000_000)


def _ep_fused_weight_dispatch_enabled() -> bool:
    # Keep the standard EP baseline on the conventional dispatch path by
    # default: hidden states, expert ids, and router weights are sent separately.
    return _env_flag("SPARSE_EP_FUSED_WEIGHT_DISPATCH", "0")

def _plain_ep_exec_mode() -> str:
    raw = str(os.environ.get("SPARSE_PLAIN_EP_EXEC_MODE", "")).strip().lower()
    if raw:
        raw = raw.replace("-", "_")
        if raw in ("sorted", "sorted_segment", "segment", "current", "optimized"):
            return "sorted_segment"
        if raw in ("unique", "unique_mask", "historical", "history", "original", "legacy_unique"):
            return "unique_mask"
        if raw in ("full", "full_mask", "legacy", "legacy_mask", "legacy_full"):
            return "full_mask"
        raise RuntimeError(
            "Unsupported SPARSE_PLAIN_EP_EXEC_MODE="
            f"{raw!r}. Use sorted_segment, unique_mask, or full_mask."
        )
    # Backward compatibility for the earlier audit switch. This intentionally
    # preserves the previous behavior of SPARSE_PLAIN_EP_LEGACY_EXEC=1.
    if _env_flag("SPARSE_PLAIN_EP_LEGACY_EXEC", "0"):
        return "full_mask"
    return "unique_mask"

def _owner_group_log(*parts: Any, env: Optional[DistEnv] = None, force: bool = False) -> None:
    if (not force) and (not _owner_group_log_enabled()):
        return
    if len(parts) == 0:
        msg = ""
    elif len(parts) == 1:
        msg = str(parts[0])
    else:
        tag = str(parts[0]).strip()
        rest = " ".join(str(x) for x in parts[1:] if str(x) != "")
        msg = f"{tag} {rest}".strip()
    prefix = "[GROUP_START]"
    if env is not None:
        print(f"{prefix} rank={env.rank} local_rank={env.local_rank} {msg}", flush=True)
    else:
        if dist.is_available() and dist.is_initialized():
            try:
                print(f"{prefix} rank={dist.get_rank()} {msg}", flush=True)
            except Exception:
                print(f"{prefix} {msg}", flush=True)
        else:
            print(f"{prefix} {msg}", flush=True)

def _owner_group_hist_str(owner_ranks: torch.Tensor, world: int) -> str:
    try:
        w = int(world)
    except Exception:
        w = 0
    if w <= 0:
        return '[]'
    if not torch.is_tensor(owner_ranks):
        return '[' + ','.join(f'{r}:0' for r in range(w)) + ']'
    vals = owner_ranks.detach().to(dtype=torch.long, device='cpu').view(-1).tolist()
    counts = [0 for _ in range(w)]
    for v in vals:
        iv = int(v)
        if 0 <= iv < w:
            counts[iv] += 1
    return '[' + ','.join(f'{r}:{counts[r]}' for r in range(w)) + ']'


def _owner_group_hist_stats(owner_ranks: torch.Tensor, world: int) -> Dict[str, float]:
    try:
        w = int(world)
    except Exception:
        w = 0
    if w <= 0 or (not torch.is_tensor(owner_ranks)):
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "cv": 0.0, "imbalance": 0.0, "empty": 0.0}
    vals = owner_ranks.detach().to(dtype=torch.long, device='cpu').view(-1).tolist()
    counts = np.zeros((w,), dtype=np.float64)
    for v in vals:
        iv = int(v)
        if 0 <= iv < w:
            counts[iv] += 1.0
    mean = float(counts.mean()) if counts.size > 0 else 0.0
    mx = float(counts.max()) if counts.size > 0 else 0.0
    mn = float(counts.min()) if counts.size > 0 else 0.0
    cv = float(counts.std() / mean) if mean > 0 else 0.0
    imbalance = float(mx / mean) if mean > 0 else 0.0
    empty = float((counts == 0).sum()) if counts.size > 0 else 0.0
    return {"max": mx, "min": mn, "mean": mean, "cv": cv, "imbalance": imbalance, "empty": empty}


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

def combine_expert_outputs(parts: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Accumulate returned expert contributions in native expert-ID order.

    Native Qwen MoE uses repeated input-dtype index_add in ascending expert ID.
    A reduction over router slots can instead accumulate FP16/BF16 values in
    FP32, producing a different rounding result that compounds across layers.
    """
    tokens, slots, hidden = parts.shape
    if expert_ids.shape != (tokens, slots):
        raise ValueError("expert output/ID shape mismatch")
    order = torch.argsort(expert_ids, dim=1, stable=True)
    result = parts.new_zeros((tokens, hidden))
    rows = torch.arange(tokens, device=parts.device)
    for slot in range(slots):
        result.add_(parts[rows, order[:, slot]])
    return result

def kv_owner_by_position(pos_idx: torch.Tensor, world: int) -> torch.Tensor:
    return torch.remainder(pos_idx, world).to(torch.long)


def _parse_rank_weights_from_env(world: int) -> List[float]:
    raw = os.environ.get("SPARSE_PLACEMENT_RANK_WEIGHTS", "").strip()
    if raw == "":
        return [1.0 for _ in range(world)]
    vals = [float(x.strip()) for x in raw.split(',') if x.strip() != ""]
    if len(vals) != world:
        raise ValueError(f"SPARSE_PLACEMENT_RANK_WEIGHTS expects {world} values, got {len(vals)}: {raw}")
    if any(v <= 0 for v in vals):
        raise ValueError(f"SPARSE_PLACEMENT_RANK_WEIGHTS must be positive, got: {vals}")
    return vals


def _weighted_random_balanced_assignment(num_experts: int, world: int, layer_idx: int) -> List[int]:
    """Deterministic per-layer weighted random balanced placement.
    Default is equal weights, which yields a random-but-balanced baseline.
    """
    weights = np.asarray(_parse_rank_weights_from_env(world), dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * float(num_experts)
    quota = np.floor(raw).astype(np.int64)
    rem = int(num_experts - int(quota.sum()))
    if rem > 0:
        frac = raw - quota
        order = np.argsort(-frac)
        for i in range(rem):
            quota[order[i % len(order)]] += 1

    seed = int(os.environ.get("SPARSE_WRB_SEED", "1"))
    rng = np.random.default_rng(seed + int(layer_idx) * 10007 + int(num_experts) * 97 + int(world))
    pool = []
    for r in range(world):
        pool.extend([int(r)] * int(quota[r]))
    if len(pool) != num_experts:
        raise RuntimeError(f"weighted_random_balanced_assignment produced wrong pool size: {len(pool)} != {num_experts}")
    rng.shuffle(pool)
    return pool


_REPLICATED_SCP_EP_FORMATS = {"replicated_scp_ep_v1", "replicated_scp_ep_v2", "replicated_scp_ep_v3"}


def _scp_ep_is_replicated(placement_obj: Dict[str, Any]) -> bool:
    return isinstance(placement_obj, dict) and str(placement_obj.get("format", "")) in _REPLICATED_SCP_EP_FORMATS


def _scp_ep_replica_choice_cost_matrix(
    placement_obj: Dict[str, Any],
    *,
    world: int,
    rank_to_node: Optional[Sequence[int]] = None,
) -> List[List[float]]:
    """Return source-rank -> candidate-rank replica choice cost.

    replicated_scp_ep_v2 serializes the exact matrix used by the solver.  Older
    placements fall back to the audited same-rank/same-node/cross-node tiers.
    """
    raw = placement_obj.get("replica_choice_cost_matrix") if isinstance(placement_obj, dict) else None
    if raw is not None:
        matrix = [[float(x) for x in row] for row in raw]
        if len(matrix) != int(world) or any(len(row) != int(world) for row in matrix):
            raise ValueError(
                f"replica_choice_cost_matrix must be [{world},{world}], got "
                f"{len(matrix)}x{len(matrix[0]) if matrix else 0}"
            )
        return matrix

    nodes = [0 for _ in range(int(world))]
    if rank_to_node is not None and len(rank_to_node) == int(world):
        nodes = [int(x) for x in rank_to_node]
    else:
        raw_nodes = placement_obj.get("rank_to_node", {}) if isinstance(placement_obj, dict) else {}
        if isinstance(raw_nodes, dict):
            nodes = [int(raw_nodes.get(str(r), raw_nodes.get(r, r))) for r in range(int(world))]

    same_node = float(os.environ.get("SPARSE_TOPOLOGY_SAME_NODE_COST", os.environ.get("SPARSE_ERS_SAME_NODE_COST", "1.0")) or 1.0)
    cross_node = float(
        os.environ.get(
            "SPARSE_TOPOLOGY_CROSS_NODE_COST",
            os.environ.get("SPARSE_ERS_CROSS_NODE_COST", os.environ.get("SPARSE_SIM_INTER_NODE_COST", "8.0")),
        )
        or 8.0
    )
    matrix: List[List[float]] = []
    for s in range(int(world)):
        row: List[float] = []
        for r in range(int(world)):
            if int(s) == int(r):
                row.append(0.0)
            elif 0 <= s < len(nodes) and 0 <= r < len(nodes) and int(nodes[s]) == int(nodes[r]):
                row.append(float(same_node))
            else:
                row.append(float(cross_node))
        matrix.append(row)
    return matrix


def _scp_ep_pick_replica_from_cost(
    *,
    source_rank: int,
    expert_id: int,
    replica_ranks: Sequence[int],
    world: int,
    cost_matrix: Sequence[Sequence[float]],
) -> int:
    ranks = sorted(set(int(r) for r in replica_ranks if 0 <= int(r) < int(world)))
    if not ranks:
        raise ValueError(f"replicated SCP-EP expert {int(expert_id)} has no valid executable replicas")
    rr_owner = int(expert_id) % int(world)
    src = int(source_rank)
    return int(
        min(
            ranks,
            key=lambda r: (
                float(cost_matrix[src][int(r)]),
                0 if int(r) == rr_owner else 1,
                int(r),
            ),
        )
    )


def _scp_ep_source_node_for_rank(source_rank: int, rank_to_node: Optional[Sequence[int]]) -> int:
    src = int(source_rank)
    if rank_to_node is not None and 0 <= src < len(rank_to_node):
        return int(rank_to_node[src])
    return int(src)


def _scp_ep_table_for_stage(placement_obj: Dict[str, Any], stage: str) -> Optional[Any]:
    st = str(stage or "").strip().lower()
    if st == "prefill":
        raw = placement_obj.get("dest_rank_table_prefill")
        if raw is not None:
            return raw
    if st == "decode":
        raw = placement_obj.get("dest_rank_table_decode")
        if raw is not None:
            return raw
    return placement_obj.get("dest_rank_table")


def _scp_ep_table_layer_assignment(
    placement_obj: Dict[str, Any],
    *,
    layer_idx: int,
    num_experts: int,
    world: int,
    source_node: int,
    stage: str = "",
) -> Optional[List[int]]:
    table = _scp_ep_table_for_stage(placement_obj, stage)
    if table is None:
        return None
    layer_key = str(int(layer_idx))
    node_key = str(int(source_node))
    if isinstance(table, dict):
        if layer_key not in table:
            raise KeyError(f"replicated SCP-EP dest_rank_table missing layer {layer_key}")
        layer_table = table[layer_key]
        if not isinstance(layer_table, dict) or node_key not in layer_table:
            raise KeyError(f"replicated SCP-EP dest_rank_table missing layer {layer_key} source_node {node_key}")
        row = layer_table[node_key]
    else:
        row = table[int(layer_idx)][int(source_node)]
    if isinstance(row, dict):
        out = [int(row[str(e)]) for e in range(int(num_experts))]
    else:
        out = [int(x) for x in list(row)]
    if len(out) != int(num_experts):
        raise ValueError(f"dest_rank_table layer {layer_idx} source_node {source_node} length {len(out)} != {num_experts}")
    for e, r in enumerate(out):
        if int(r) < 0 or int(r) >= int(world):
            raise ValueError(f"dest_rank_table rank out of range layer {layer_idx} node {source_node} expert {e}: {r}")
    return out


def _scp_ep_layer_assignment(
    placement_obj: Dict[str, Any],
    *,
    layer_idx: int,
    num_experts: int,
    world: int,
    source_rank: int = 0,
    stage: str = "",
    rank_to_node: Optional[Sequence[int]] = None,
) -> List[int]:
    """Load SCP-EP dispatch assignment for this source rank.

    Non-replicated SCP-EP uses placement[layer][expert] = owner rank.
    Replicated SCP-EP uses replicas[layer][expert] as executable permissions;
    runtime derives the concrete source-rank dispatch assignment by choosing the
    lowest topology-cost replica.  In both cases this is dispatch ownership, not
    GPU residency.
    """
    if not isinstance(placement_obj, dict):
        raise ValueError("SCP-EP placement must be a JSON object")
    if _scp_ep_is_replicated(placement_obj):
        source_node = _scp_ep_source_node_for_rank(int(source_rank), rank_to_node)
        table_assignment = _scp_ep_table_layer_assignment(
            placement_obj,
            layer_idx=int(layer_idx),
            num_experts=int(num_experts),
            world=int(world),
            source_node=int(source_node),
            stage=str(stage or ""),
        )
        if table_assignment is not None:
            return table_assignment
        replicas = _scp_ep_layer_replicas(
            placement_obj,
            layer_idx=layer_idx,
            num_experts=num_experts,
            world=world,
            fallback_assignment=[int(eid % int(world)) for eid in range(int(num_experts))],
        )
        cost_matrix = _scp_ep_replica_choice_cost_matrix(placement_obj, world=world, rank_to_node=rank_to_node)
        out: List[int] = []
        for eid in range(int(num_experts)):
            picked = _scp_ep_pick_replica_from_cost(
                source_rank=int(source_rank),
                expert_id=int(eid),
                replica_ranks=replicas[int(eid)],
                world=int(world),
                cost_matrix=cost_matrix,
            )
            out.append(int(picked))
        return out
    if placement_obj.get("format") == "replicated_scp_ep_v1_legacy_choice_only":
        stage_key = str(stage or "").strip().lower()
        choice_name = "choice"
        if stage_key == "prefill" and isinstance(placement_obj.get("choice_prefill"), dict):
            choice_name = "choice_prefill"
        elif stage_key == "decode" and isinstance(placement_obj.get("choice_decode"), dict):
            choice_name = "choice_decode"
        choice = placement_obj.get(choice_name, placement_obj.get("choice", {}))
        layer_key = str(int(layer_idx))
        source_key = str(int(source_rank))
        if layer_key not in choice:
            raise KeyError(f"replicated SCP-EP {choice_name} missing layer {layer_key}")
        if source_key not in choice[layer_key]:
            raise KeyError(f"replicated SCP-EP {choice_name} missing layer {layer_key} source rank {source_key}")
        layer_source_map = choice[layer_key][source_key]
        out: List[int] = []
        for eid in range(int(num_experts)):
            ekey = str(int(eid))
            if ekey not in layer_source_map:
                raise KeyError(f"replicated SCP-EP {choice_name} missing layer {layer_key} source {source_key} expert {ekey}")
            r = int(layer_source_map[ekey])
            if r < 0 or r >= int(world):
                raise ValueError(f"replicated SCP-EP {choice_name} rank out of range at layer {layer_key} source {source_key} expert {ekey}: {r}")
            out.append(int(r))
        return out
    placement = placement_obj.get("placement", placement_obj)
    if not isinstance(placement, dict):
        raise ValueError("SCP-EP placement JSON must contain a placement object")
    layer_key = str(int(layer_idx))
    if layer_key not in placement:
        raise KeyError(f"SCP-EP placement missing layer {layer_key}")
    layer_map = placement[layer_key]
    if not isinstance(layer_map, dict):
        raise ValueError(f"SCP-EP placement layer {layer_key} must be an expert->rank object")
    out: List[int] = []
    for eid in range(int(num_experts)):
        ekey = str(int(eid))
        if ekey not in layer_map:
            raise KeyError(f"SCP-EP placement missing layer {layer_key} expert {ekey}")
        r = int(layer_map[ekey])
        if r < 0 or r >= int(world):
            raise ValueError(f"SCP-EP owner rank out of range at layer {layer_key} expert {ekey}: {r}")
        out.append(int(r))
    return out


def _scp_ep_layer_replicas(
    placement_obj: Dict[str, Any],
    *,
    layer_idx: int,
    num_experts: int,
    world: int,
    fallback_assignment: Sequence[int],
) -> List[List[int]]:
    """Return executable replica ranks for each expert.

    For non-replicated SCP-EP this degenerates to the unique owner.  For
    replicated SCP-EP this is the explicit replica set.  Replica permission is
    separate from GPU cache residency; demand load/prefetch still goes through
    the existing expert cache.
    """
    if not _scp_ep_is_replicated(placement_obj):
        return [[int(r)] for r in fallback_assignment]
    replicas = placement_obj.get("replicas", {})
    layer_key = str(int(layer_idx))
    if layer_key not in replicas:
        raise KeyError(f"replicated SCP-EP replicas missing layer {layer_key}")
    layer_map = replicas[layer_key]
    out: List[List[int]] = []
    for eid in range(int(num_experts)):
        ekey = str(int(eid))
        if ekey not in layer_map:
            raise KeyError(f"replicated SCP-EP replicas missing layer {layer_key} expert {ekey}")
        ranks = sorted(set(int(r) for r in layer_map[ekey]))
        if not ranks:
            raise ValueError(f"replicated SCP-EP expert has no replicas at layer {layer_key} expert {ekey}")
        for r in ranks:
            if r < 0 or r >= int(world):
                raise ValueError(f"replicated SCP-EP replica rank out of range at layer {layer_key} expert {ekey}: {r}")
        out.append(ranks)
    return out


def _scp_ep_layer_receive_experts(
    placement_obj: Dict[str, Any],
    *,
    layer_idx: int,
    num_experts: int,
    world: int,
    rank: int,
    stage: str = "",
    rank_to_node: Optional[Sequence[int]] = None,
) -> Optional[set[int]]:
    """Return experts whose replicated SCP-EP runtime assignment may hit rank.

    This is a runtime/cache planning view, not a source-rank view.  A rank needs
    cache readiness for experts that any source rank may send here for execution.
    """
    if not _scp_ep_is_replicated(placement_obj):
        return None
    table = _scp_ep_table_for_stage(placement_obj, stage)
    if table is not None:
        out: set[int] = set()
        node_count = 0
        raw_nodes = placement_obj.get("source_nodes")
        if isinstance(raw_nodes, list):
            node_count = len(raw_nodes)
        elif rank_to_node is not None:
            node_count = max([int(x) for x in rank_to_node], default=-1) + 1
        if node_count <= 0:
            node_count = int(world)
        for source_node in range(int(node_count)):
            row = _scp_ep_table_layer_assignment(
                placement_obj,
                layer_idx=int(layer_idx),
                num_experts=int(num_experts),
                world=int(world),
                source_node=int(source_node),
                stage=str(stage or ""),
            )
            if row is None:
                continue
            for eid, dst in enumerate(row):
                if int(dst) == int(rank):
                    out.add(int(eid))
        return out
    replicas = _scp_ep_layer_replicas(
        placement_obj,
        layer_idx=layer_idx,
        num_experts=num_experts,
        world=world,
        fallback_assignment=[int(eid % int(world)) for eid in range(int(num_experts))],
    )
    cost_matrix = _scp_ep_replica_choice_cost_matrix(placement_obj, world=world, rank_to_node=rank_to_node)
    out: set[int] = set()
    for source in range(int(world)):
        for eid in range(int(num_experts)):
            picked = _scp_ep_pick_replica_from_cost(
                source_rank=int(source),
                expert_id=int(eid),
                replica_ranks=replicas[int(eid)],
                world=int(world),
                cost_matrix=cost_matrix,
            )
            if int(picked) == int(rank):
                out.add(int(eid))
    return out


def _sim_env_raw(name: str, legacy_name: str = "", default: str = "") -> str:
    if name in os.environ:
        return str(os.environ.get(name, default))
    if legacy_name and legacy_name in os.environ:
        return str(os.environ.get(legacy_name, default))
    return str(default)


def _logical_weaknet_enabled() -> bool:
    raw = _sim_env_raw("SPARSE_SIM_ENABLE_WEAKNET", "SPARSE_LOGICAL_WEAKNET_ENABLE", "0").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _logical_transfer_counters_enabled() -> bool:
    raw = os.environ.get("SPARSE_LOGICAL_TRANSFER_COUNTERS", "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _weaknet_small_packet_threshold() -> int:
    try:
        return max(int(os.environ.get("SPARSE_WEAKNET_SMALL_PACKET_THRESHOLD", "4") or 4), 1)
    except Exception:
        return 4


def _weaknet_packet_trace_enabled() -> bool:
    return _env_flag("SPARSE_WEAKNET_PACKET_TRACE", "0")


def _weaknet_packet_trace_limit() -> int:
    try:
        return max(int(os.environ.get("SPARSE_WEAKNET_PACKET_TRACE_LIMIT", "20000") or 20000), 0)
    except Exception:
        return 20000


def _sim_heterogeneity_enabled() -> bool:
    raw = _sim_env_raw("SPARSE_SIM_ENABLE_HETEROGENEITY", "", "0").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _sim_heterogeneity_sleep_enabled() -> bool:
    if not _sim_heterogeneity_enabled():
        return False
    raw = _sim_env_raw("SPARSE_SIM_HETEROGENEITY_SLEEP", "", "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _parse_logical_node_map(world: int) -> List[int]:
    raw = _sim_env_raw("SPARSE_SIM_NODE_MAP", "SPARSE_LOGICAL_NODE_MAP", "").strip()
    if raw == "":
        if _logical_weaknet_enabled() and int(world) == 4:
            return [0, 0, 1, 1]
        return [0 for _ in range(int(world))]
    vals = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(vals) != int(world):
        raise ValueError(f"SPARSE_SIM_NODE_MAP/SPARSE_LOGICAL_NODE_MAP expects {world} entries, got {len(vals)}: {raw}")
    return vals


def _current_global_rank(default_rank: int = 0) -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(default_rank)


def _current_sim_node_id(world: Optional[int] = None, default_rank: int = 0) -> int:
    if world is None:
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = _current_global_rank(default_rank)
    node_map = _parse_logical_node_map(int(world))
    if 0 <= rank < len(node_map):
        return int(node_map[rank])
    return 0


def _sim_node_float(prefix: str, node_id: int, default: float) -> float:
    node_key = f"SPARSE_SIM_NODE{int(node_id)}_{prefix}"
    raw = os.environ.get(node_key, os.environ.get(f"SPARSE_SIM_{prefix}", str(default)))
    try:
        return float(raw)
    except Exception:
        return float(default)


def _sim_node_compute_scale(world: Optional[int] = None) -> float:
    node_id = _current_sim_node_id(world)
    return max(_sim_node_float("COMPUTE_SCALE", node_id, 1.0), 1e-6)


def _sim_node_h2d_scale(world: Optional[int] = None) -> float:
    node_id = _current_sim_node_id(world)
    return max(_sim_node_float("H2D_SCALE", node_id, 1.0), 1e-6)


def _logical_weaknet_payload(
    *,
    comm_global_ranks: Tuple[int, ...],
    send_counts: torch.Tensor,
    payload_bytes_per_item: int,
    logical_transfer_multiplier: int = 1,
    metrics: Optional["Metrics"] = None,
    layer_idx: int = -1,
    label: str = "",
) -> Dict[str, float]:
    out = {
        "self_items": 0.0,
        "intra_node_items": 0.0,
        "inter_node_items": 0.0,
        "intra_node_bytes": 0.0,
        "inter_node_bytes": 0.0,
        "intra_node_transfers": 0.0,
        "inter_node_transfers": 0.0,
        "sim_ms": 0.0,
    }
    if send_counts.numel() == 0 or int(payload_bytes_per_item) <= 0:
        return out

    node_map = _parse_logical_node_map(dist.get_world_size() if dist.is_available() and dist.is_initialized() else len(comm_global_ranks))
    my_global = _current_global_rank(int(comm_global_ranks[0]) if len(comm_global_ranks) > 0 else 0)
    my_node = int(node_map[my_global])
    counts = send_counts.detach().to("cpu", dtype=torch.long).view(-1).tolist()

    inter_node_items = 0
    intra_node_items = 0
    self_items = 0
    inter_node_transfers = 0
    intra_node_transfers = 0
    inter_node_packet_sumsq = 0
    inter_node_packet_max_items = 0
    inter_node_packet_le1_count = 0
    inter_node_packet_le2_count = 0
    inter_node_packet_le4_count = 0
    inter_node_packet_le8_count = 0
    inter_node_packet_gt8_count = 0
    inter_node_small_packet_count = 0
    inter_node_small_packet_items = 0
    small_threshold = _weaknet_small_packet_threshold()
    trace_enabled = _weaknet_packet_trace_enabled()
    trace_limit = _weaknet_packet_trace_limit()
    transfer_multiplier = max(int(logical_transfer_multiplier), 1)
    for comm_dst, cnt in enumerate(counts):
        if int(cnt) <= 0 or int(comm_dst) >= len(comm_global_ranks):
            continue
        dst_global = int(comm_global_ranks[int(comm_dst)])
        if dst_global == my_global:
            self_items += int(cnt)
        elif 0 <= dst_global < len(node_map) and int(node_map[dst_global]) != my_node:
            dst_node = int(node_map[dst_global])
            inter_node_items += int(cnt)
            inter_node_transfers += 1
            inter_node_packet_sumsq += int(cnt) * int(cnt)
            inter_node_packet_max_items = max(inter_node_packet_max_items, int(cnt))
            if int(cnt) <= 1:
                inter_node_packet_le1_count += 1
            elif int(cnt) <= 2:
                inter_node_packet_le2_count += 1
            elif int(cnt) <= 4:
                inter_node_packet_le4_count += 1
            elif int(cnt) <= 8:
                inter_node_packet_le8_count += 1
            else:
                inter_node_packet_gt8_count += 1
            if int(cnt) <= int(small_threshold):
                inter_node_small_packet_count += 1
                inter_node_small_packet_items += int(cnt)
            if metrics is not None:
                pair_key = f"{int(my_node)}->{int(dst_node)}"
                metrics.weaknet_inter_node_pair_event_counts[pair_key] = (
                    int(metrics.weaknet_inter_node_pair_event_counts.get(pair_key, 0)) + int(transfer_multiplier)
                )
                metrics.weaknet_inter_node_pair_item_counts[pair_key] = (
                    int(metrics.weaknet_inter_node_pair_item_counts.get(pair_key, 0)) + int(cnt)
                )
                if trace_enabled and trace_limit > 0:
                    if metrics.weaknet_packet_trace is None:
                        metrics.weaknet_packet_trace = []
                    if len(metrics.weaknet_packet_trace) < trace_limit:
                        metrics.weaknet_packet_trace.append({
                            "phase": str(getattr(SparseContext, "current_phase", "none")),
                            "runtime_call_id": int(getattr(SparseContext, "runtime_call_id", -1)),
                            "layer": int(layer_idx),
                            "label": str(label or ""),
                            "src_rank": int(my_global),
                            "dst_rank": int(dst_global),
                            "src_node": int(my_node),
                            "dst_node": int(dst_node),
                            "item_count": int(cnt),
                            "bytes": int(cnt) * int(payload_bytes_per_item),
                            "small_packet_threshold": int(small_threshold),
                            "is_small_packet": bool(int(cnt) <= int(small_threshold)),
                        })
        else:
            intra_node_items += int(cnt)
            intra_node_transfers += 1

    intra_node_bytes = int(intra_node_items) * int(payload_bytes_per_item)
    inter_node_bytes = int(inter_node_items) * int(payload_bytes_per_item)
    charged_inter_node_transfers = int(inter_node_transfers) * int(transfer_multiplier)
    charged_intra_node_transfers = int(intra_node_transfers) * int(transfer_multiplier)
    charged_inter_node_packet_sumsq = int(inter_node_packet_sumsq) * int(transfer_multiplier)
    out.update({
        "self_items": float(self_items),
        "intra_node_items": float(intra_node_items),
        "inter_node_items": float(inter_node_items),
        "intra_node_bytes": float(intra_node_bytes),
        "inter_node_bytes": float(inter_node_bytes),
        "intra_node_transfers": float(charged_intra_node_transfers),
        "inter_node_transfers": float(charged_inter_node_transfers),
        "inter_node_packet_item_sumsq": float(charged_inter_node_packet_sumsq),
        "inter_node_packet_max_items": float(inter_node_packet_max_items),
        "inter_node_packet_le1_count": float(inter_node_packet_le1_count * transfer_multiplier),
        "inter_node_packet_le2_count": float(inter_node_packet_le2_count * transfer_multiplier),
        "inter_node_packet_le4_count": float(inter_node_packet_le4_count * transfer_multiplier),
        "inter_node_packet_le8_count": float(inter_node_packet_le8_count * transfer_multiplier),
        "inter_node_packet_gt8_count": float(inter_node_packet_gt8_count * transfer_multiplier),
        "inter_node_small_packet_count": float(inter_node_small_packet_count * transfer_multiplier),
        "inter_node_small_packet_item_count": float(inter_node_small_packet_items * transfer_multiplier),
        "logical_transfer_multiplier": float(transfer_multiplier),
    })

    # New SPARSE_SIM_INTER_NODE_BW_MBPS is interpreted as MB/s. The legacy
    # SPARSE_LOGICAL_INTER_NODE_BW_MBPS path kept its historical Mbit/s meaning.
    if "SPARSE_SIM_INTER_NODE_BW_MBPS" in os.environ:
        bw_bytes_per_s = max(float(os.environ.get("SPARSE_SIM_INTER_NODE_BW_MBPS", "0") or 0.0) * 1_000_000.0, 0.0)
    else:
        legacy_mbps = float(os.environ.get("SPARSE_LOGICAL_INTER_NODE_BW_MBPS", "0") or 0.0)
        bw_bytes_per_s = max(legacy_mbps * 1_000_000.0 / 8.0, 0.0)
    latency_ms = float(_sim_env_raw("SPARSE_SIM_INTER_NODE_LAT_MS", "SPARSE_LOGICAL_INTER_NODE_LATENCY_MS", "0") or 0.0)
    efficiency = max(float(os.environ.get("SPARSE_LOGICAL_INTER_NODE_EFFICIENCY", "1.0") or 1.0), 1e-6)
    bw_bytes_per_s = max(bw_bytes_per_s * efficiency, 0.0)
    if inter_node_bytes > 0 and (bw_bytes_per_s > 0.0 or latency_ms > 0.0):
        transfer_latency_ms = max(float(latency_ms), 0.0) * float(max(charged_inter_node_transfers, 1))
        bandwidth_ms = (float(inter_node_bytes) / max(bw_bytes_per_s, 1e-9)) * 1000.0 if bw_bytes_per_s > 0.0 else 0.0
        out["sim_ms"] = float(transfer_latency_ms + bandwidth_ms)

    if metrics is not None:
        metrics.comm_bytes_activation_intra_node += int(intra_node_bytes)
        metrics.comm_bytes_activation_inter_node += int(inter_node_bytes)
        metrics.weaknet_intra_node_transfer_count += int(charged_intra_node_transfers)
        metrics.weaknet_inter_node_transfer_count += int(charged_inter_node_transfers)
        metrics.weaknet_intra_node_item_count += int(intra_node_items)
        metrics.weaknet_inter_node_item_count += int(inter_node_items)
        metrics.weaknet_inter_node_packet_item_sumsq += int(charged_inter_node_packet_sumsq)
        metrics.weaknet_inter_node_packet_max_items = max(
            int(metrics.weaknet_inter_node_packet_max_items),
            int(inter_node_packet_max_items),
        )
        metrics.weaknet_inter_node_packet_le1_count += int(inter_node_packet_le1_count * transfer_multiplier)
        metrics.weaknet_inter_node_packet_le2_count += int(inter_node_packet_le2_count * transfer_multiplier)
        metrics.weaknet_inter_node_packet_le4_count += int(inter_node_packet_le4_count * transfer_multiplier)
        metrics.weaknet_inter_node_packet_le8_count += int(inter_node_packet_le8_count * transfer_multiplier)
        metrics.weaknet_inter_node_packet_gt8_count += int(inter_node_packet_gt8_count * transfer_multiplier)
        metrics.weaknet_inter_node_small_packet_count += int(inter_node_small_packet_count * transfer_multiplier)
        metrics.weaknet_inter_node_small_packet_item_count += int(inter_node_small_packet_items * transfer_multiplier)
        metrics.weaknet_inter_node_sim_ms += float(out["sim_ms"])
        if metrics.per_layer is not None and int(layer_idx) >= 0:
            lp = metrics.per_layer.setdefault(int(layer_idx), LayerPerf())
            lp.activation_intra_node_bytes += int(intra_node_bytes)
            lp.activation_inter_node_bytes += int(inter_node_bytes)
            lp.weaknet_intra_node_transfer_count += int(charged_intra_node_transfers)
            lp.weaknet_inter_node_transfer_count += int(charged_inter_node_transfers)
            lp.weaknet_intra_node_item_count += int(intra_node_items)
            lp.weaknet_inter_node_item_count += int(inter_node_items)
            lp.weaknet_inter_node_packet_item_sumsq += int(charged_inter_node_packet_sumsq)
            lp.weaknet_inter_node_packet_max_items = max(
                int(lp.weaknet_inter_node_packet_max_items),
                int(inter_node_packet_max_items),
            )
            lp.weaknet_inter_node_packet_le1_count += int(inter_node_packet_le1_count * transfer_multiplier)
            lp.weaknet_inter_node_packet_le2_count += int(inter_node_packet_le2_count * transfer_multiplier)
            lp.weaknet_inter_node_packet_le4_count += int(inter_node_packet_le4_count * transfer_multiplier)
            lp.weaknet_inter_node_packet_le8_count += int(inter_node_packet_le8_count * transfer_multiplier)
            lp.weaknet_inter_node_packet_gt8_count += int(inter_node_packet_gt8_count * transfer_multiplier)
            lp.weaknet_inter_node_small_packet_count += int(inter_node_small_packet_count * transfer_multiplier)
            lp.weaknet_inter_node_small_packet_item_count += int(inter_node_small_packet_items * transfer_multiplier)
            lp.weaknet_inter_node_sim_ms += float(out["sim_ms"])

    return out


def _maybe_sleep_logical_weaknet(
    *,
    comm_global_ranks: Tuple[int, ...],
    send_counts: torch.Tensor,
    payload_bytes_per_item: int,
    logical_transfer_multiplier: int = 1,
    metrics: Optional["Metrics"] = None,
    layer_idx: int = -1,
    label: str = "",
) -> Dict[str, float]:
    if bool(getattr(SparseContext, "suppress_logical_weaknet_once", False)):
        SparseContext.suppress_logical_weaknet_once = False
        return {
            "self_items": 0.0,
            "intra_node_items": 0.0,
            "inter_node_items": 0.0,
            "intra_node_bytes": 0.0,
            "inter_node_bytes": 0.0,
            "intra_node_transfers": 0.0,
            "inter_node_transfers": 0.0,
            "sim_ms": 0.0,
        }
    if not _logical_weaknet_enabled() and not _logical_transfer_counters_enabled():
        return {
            "self_items": 0.0,
            "intra_node_items": 0.0,
            "inter_node_items": 0.0,
            "intra_node_bytes": 0.0,
            "inter_node_bytes": 0.0,
            "intra_node_transfers": 0.0,
            "inter_node_transfers": 0.0,
            "sim_ms": 0.0,
        }
    payload = _logical_weaknet_payload(
        comm_global_ranks=comm_global_ranks,
        send_counts=send_counts,
        payload_bytes_per_item=payload_bytes_per_item,
        logical_transfer_multiplier=logical_transfer_multiplier,
        metrics=metrics,
        layer_idx=layer_idx,
        label=label,
    )
    if not _logical_weaknet_enabled():
        return payload
    sleep_s = max(float(payload.get("sim_ms", 0.0)), 0.0) / 1000.0
    if sleep_s > 0.0:
        time.sleep(sleep_s)
    return payload


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




def _phase_trace_enabled() -> bool:
    raw = str(os.environ.get("SPARSE_PHASE_TRACE", "0")).strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _phase_trace(msg: str) -> None:
    if not _phase_trace_enabled():
        return
    if dist.is_available() and dist.is_initialized():
        try:
            rank = dist.get_rank()
            print(f"[PHASE_TRACE] rank={rank} {msg}", flush=True)
            return
        except Exception:
            pass
    print(f"[PHASE_TRACE] {msg}", flush=True)


_NODE_GROUPED_LAYER_POLICY_CACHE_KEY: Optional[str] = None
_NODE_GROUPED_LAYER_POLICY_CACHE: Optional[Dict[str, Any]] = None


def _profile_routing_enabled() -> bool:
    return _env_flag("SPARSE_PROFILE_ROUTING", "0")


def _profile_routing_stage_allowed(stage: str) -> bool:
    raw = str(os.environ.get("SPARSE_PROFILE_STAGE", "both") or "both").strip().lower()
    if raw in ("", "both", "all"):
        return True
    return str(stage).strip().lower() == raw


def _parse_layer_id_values(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, dict):
        out: set[int] = set()
        for k, v in value.items():
            enabled = bool(v)
            if isinstance(v, dict) and "enabled" in v:
                enabled = bool(v.get("enabled"))
            if not enabled:
                continue
            try:
                out.add(int(k))
            except Exception:
                continue
        return out
    if isinstance(value, (list, tuple, set)):
        out = set()
        for item in value:
            try:
                out.add(int(item))
            except Exception:
                continue
        return out
    out: set[int] = set()
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(part))
            except Exception:
                continue
    return out


def _load_node_grouped_layer_policy() -> Optional[Dict[str, Any]]:
    global _NODE_GROUPED_LAYER_POLICY_CACHE_KEY, _NODE_GROUPED_LAYER_POLICY_CACHE
    raw_path = str(os.environ.get("SPARSE_NODE_GROUPED_LAYER_POLICY_PATH", "") or "").strip()
    if raw_path == "":
        _NODE_GROUPED_LAYER_POLICY_CACHE_KEY = None
        _NODE_GROUPED_LAYER_POLICY_CACHE = None
        return None
    path = Path(raw_path)
    key = str(path)
    if _NODE_GROUPED_LAYER_POLICY_CACHE_KEY == key:
        return _NODE_GROUPED_LAYER_POLICY_CACHE
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = None
    _NODE_GROUPED_LAYER_POLICY_CACHE_KEY = key
    _NODE_GROUPED_LAYER_POLICY_CACHE = obj if isinstance(obj, dict) else None
    return _NODE_GROUPED_LAYER_POLICY_CACHE


def _node_grouped_layer_policy_decision(layer_idx: int) -> Optional[bool]:
    policy = _load_node_grouped_layer_policy()
    if not policy:
        return None

    layer_key = str(int(layer_idx))
    for field in ("per_layer", "layers", "layer_policy"):
        table = policy.get(field)
        if not isinstance(table, dict):
            continue
        rec = table.get(layer_key, table.get(int(layer_idx)))
        if rec is None:
            continue
        if isinstance(rec, bool):
            return bool(rec)
        if isinstance(rec, (int, float)):
            return bool(int(rec))
        if isinstance(rec, dict):
            if "enabled" in rec:
                return bool(rec.get("enabled"))
            if "source" in rec:
                return str(rec.get("source", "")).strip().lower() not in ("", "rr", "disable", "disabled", "off")

    if int(layer_idx) in _parse_layer_id_values(policy.get("enabled_layers")):
        return True
    if int(layer_idx) in _parse_layer_id_values(policy.get("disabled_layers")):
        return False

    default_raw = str(os.environ.get("SPARSE_NODE_GROUPED_LAYER_POLICY_DEFAULT", policy.get("default", "adaptive")) or "adaptive")
    default = default_raw.strip().lower()
    if default in ("enable", "enabled", "on", "true", "1", "ngd"):
        return True
    if default in ("disable", "disabled", "off", "false", "0", "rr", "normal_ep"):
        return False
    return None

@dataclass
class LayerPerf:
    expert_sum: float = 0.0
    expert_sumsq: float = 0.0
    expert_max: float = 0.0
    a2a_send_bytes: int = 0
    a2a_remote_bytes: int = 0
    activation_intra_node_bytes: int = 0
    activation_inter_node_bytes: int = 0
    weaknet_intra_node_item_count: int = 0
    weaknet_inter_node_item_count: int = 0
    weaknet_intra_node_transfer_count: int = 0
    weaknet_inter_node_transfer_count: int = 0
    weaknet_inter_node_packet_item_sumsq: int = 0
    weaknet_inter_node_packet_max_items: int = 0
    weaknet_inter_node_packet_le1_count: int = 0
    weaknet_inter_node_packet_le2_count: int = 0
    weaknet_inter_node_packet_le4_count: int = 0
    weaknet_inter_node_packet_le8_count: int = 0
    weaknet_inter_node_packet_gt8_count: int = 0
    weaknet_inter_node_small_packet_count: int = 0
    weaknet_inter_node_small_packet_item_count: int = 0
    weaknet_inter_node_sim_ms: float = 0.0
    h2d_bytes: int = 0
    h2d_demand_bytes: int = 0
    h2d_prefetch_bytes: int = 0
    demand_h2d_exposed_ms: float = 0.0
    demand_h2d_exposed_prefill_ms: float = 0.0
    demand_h2d_exposed_decode_ms: float = 0.0
    sim_demand_h2d_exposed_ms: float = 0.0
    sim_demand_h2d_exposed_prefill_ms: float = 0.0
    sim_demand_h2d_exposed_decode_ms: float = 0.0
    sim_h2d_extra_sleep_ms: float = 0.0
    demand_h2d_event_count: int = 0
    prefetch_h2d_copy_ms: float = 0.0
    prefetch_h2d_event_count: int = 0
    prefetch_wait_ms: float = 0.0
    prefetch_wait_event_count: int = 0
    plain_ep_executor_calls: int = 0
    epipe_executor_calls: int = 0
    epipe_executor_attempt_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    prefetch_hit_count: int = 0
    prefetch_wasted_count: int = 0
    prefetch_admitted_count: int = 0
    rs_pred_candidate_count: int = 0
    rs_pred_local_candidate_count: int = 0
    rs_pred_nonlocal_candidate_count: int = 0
    rs_candidate_filter_in_count: int = 0
    rs_candidate_filter_out_count: int = 0
    rs_candidate_filter_out_cached_count: int = 0
    rs_candidate_filter_out_low_score_count: int = 0
    rs_candidate_survived_count: int = 0
    prefetch_issued_count: int = 0
    prefetch_completed_count: int = 0
    prefetch_complete_before_use_count: int = 0
    prefetch_rejected_active_replace: int = 0
    prefetch_rejected_margin: int = 0
    prefetch_rejected_cache_full: int = 0
    prefetch_rejected_low_score: int = 0

    ttl_prevented_evictions: int = 0
    cache_eviction_fallback_all_protected: int = 0
    prefill_scp_transient_load_count: int = 0
    prefill_scp_transient_hit_count: int = 0
    prefill_scp_transient_promote_count: int = 0
    prefill_scp_transient_evict_count: int = 0
    prefill_scp_transient_resident_at_decode_start: int = 0
    prefill_scp_transient_demote_count: int = 0
    prefill_scp_transient_resident_at_decode_end: int = 0

    resolver_decode_total_queries: int = 0
    resolver_decode_rr_owner_count: int = 0
    resolver_decode_scp_replica_selected_count: int = 0
    resolver_decode_rr_fallback_count: int = 0
    resolver_decode_replica_reject_not_ready_count: int = 0
    resolver_decode_replica_reject_no_comm_gain_count: int = 0
    resolver_decode_replica_cache_ready_count: int = 0
    resolver_decode_replica_prefetch_ready_count: int = 0
    resolver_decode_route_cost_before: float = 0.0
    resolver_decode_route_cost_after: float = 0.0

    scp_prefill_route_diag_items: int = 0
    scp_prefill_route_diag_replica_items: int = 0
    scp_prefill_route_diag_cross_items_before: int = 0
    scp_prefill_route_diag_cross_items_after: int = 0
    scp_prefill_route_diag_cross_events_before: int = 0
    scp_prefill_route_diag_cross_events_after: int = 0
    scp_prefill_route_diag_cost_before: float = 0.0
    scp_prefill_route_diag_cost_after: float = 0.0
    scp_decode_route_diag_items: int = 0
    scp_decode_route_diag_replica_items: int = 0
    scp_decode_route_diag_cross_items_before: int = 0
    scp_decode_route_diag_cross_items_after: int = 0
    scp_decode_route_diag_cross_events_before: int = 0
    scp_decode_route_diag_cross_events_after: int = 0
    scp_decode_route_diag_cost_before: float = 0.0
    scp_decode_route_diag_cost_after: float = 0.0

    scp_table_local_replica_route_count: int = 0
    scp_table_rr_fallback_count: int = 0
    scp_table_cross_node_route_count: int = 0
    scp_table_same_node_route_count: int = 0
    scp_route_cost_before_sum: float = 0.0
    scp_route_cost_after_sum: float = 0.0
    scp_lookup_calls: int = 0
    scp_lookup_overhead_ms: float = 0.0

    rs_prefetch_target_resolver_match_count: int = 0
    rs_prefetch_target_resolver_mismatch_count: int = 0
    rs_prefetch_would_be_wrong_without_resolver_count: int = 0
    rs_scp_target_mismatch_count: int = 0

    warmup_loaded_experts: int = 0
    warmup_skipped_resident: int = 0
    warmup_h2d_bytes: int = 0
    warmup_time_ms: float = 0.0

    transition_lookup_calls: int = 0
    transition_lookup_overhead_ms: float = 0.0
    transition_prefetch_admitted: int = 0
    transition_prefetch_hit: int = 0
    transition_prefetch_late: int = 0
    transition_prefetch_wasted: int = 0
    transition_safe_evict_denied: int = 0
    transition_no_replace_denied: int = 0
    epipe_shadow_count: int = 0
    epipe_enabled_count: int = 0
    epipe_fallback_count: int = 0
    epipe_fallback_no_ready: int = 0
    epipe_fallback_no_missing: int = 0
    epipe_fallback_no_miss_trigger: int = 0
    epipe_fallback_no_ready_lookahead: int = 0
    epipe_fallback_low_ready_items: int = 0
    epipe_fallback_prefill_disabled: int = 0
    epipe_fallback_no_memory_headroom: int = 0
    epipe_fallback_memory: int = 0
    epipe_fallback_cache_full_no_temp: int = 0
    epipe_fallback_exception: int = 0
    epipe_decode_rs_miss_trigger_count: int = 0
    epipe_decode_rs_skipped_no_miss_count: int = 0
    epipe_prefill_miss_trigger_count: int = 0
    epipe_prefill_shadow_opportunity_count: int = 0
    epipe_rs_buffered_calls: int = 0
    epipe_rs_buffer_flush_count: int = 0
    epipe_rs_buffered_ready_group_count: int = 0
    epipe_rs_buffered_ready_item_count: int = 0
    epipe_rs_residual_scan_calls: int = 0
    epipe_rs_residual_scan_ready_group_count: int = 0
    epipe_rs_residual_scan_ready_item_count: int = 0
    epipe_rs_residual_scan_residual_count: int = 0
    epipe_rs_residual_scan_late_count: int = 0
    epipe_rs_residual_scan_demand_count: int = 0
    epipe_rs_residual_scan_no_ready_count: int = 0
    epipe_rs_trigger_count: int = 0
    epipe_rs_late_prefetch_trigger_count: int = 0
    epipe_rs_demand_miss_trigger_count: int = 0
    epipe_rs_late_prefetch_overlap_count: int = 0
    epipe_rs_demand_stage_count: int = 0
    epipe_rs_demand_stage_disabled_by_gate_count: int = 0
    epipe_rs_demand_stage_force_count: int = 0
    epipe_prefill_executor_calls: int = 0
    epipe_prefill_stage_success_count: int = 0
    epipe_decode_no_rs_executor_calls: int = 0
    epipe_decode_no_rs_stage_success_count: int = 0
    epipe_decode_rs_executor_calls: int = 0
    epipe_decode_rs_stage_success_count: int = 0
    epipe_prefill_enabled_count: int = 0
    epipe_decode_no_rs_enabled_count: int = 0
    epipe_decode_rs_late_overlap_count: int = 0
    epipe_decode_rs_demand_stage_count: int = 0
    epipe_rs_gate_skip_count: int = 0
    epipe_rs_gate_reenable_count: int = 0
    epipe_rs_buffer_too_small_count: int = 0
    epipe_lookahead_ready_group_count: int = 0
    epipe_lookahead_ready_item_count: int = 0
    epipe_staged_expert_count: int = 0
    epipe_staged_h2d_bytes: int = 0
    epipe_deferred_lru_commit_count: int = 0
    epipe_prefill_transient_commit_count: int = 0
    epipe_prefill_transient_demote_count: int = 0
    epipe_prefill_transient_promote_count: int = 0
    epipe_done_group_skip_count: int = 0
    epipe_ready_expert_count: int = 0
    epipe_missing_expert_count: int = 0
    epipe_ready_item_count: int = 0
    epipe_missing_item_count: int = 0
    epipe_cache_free_slot_count: int = 0
    epipe_cache_full_staged_count: int = 0
    epipe_transient_payload_count: int = 0
    epipe_prefill_transient_no_commit_count: int = 0
    epipe_prefill_admission_bypass_count: int = 0
    epipe_cache_insert_count: int = 0
    epipe_prefill_cache_insert_count: int = 0
    epipe_prefill_extra_cache_insert_count: int = 0
    epipe_prefill_extra_cache_hit_count: int = 0
    epipe_prefill_extra_cache_evict_count: int = 0
    epipe_prefill_extra_cache_no_room_count: int = 0
    epipe_prefill_extra_cache_guard_skip_count: int = 0
    epipe_prefill_extra_cache_h2d_bytes: int = 0
    epipe_prefill_extra_cache_resident_count: int = 0
    epipe_prefill_extra_cache_resident_at_decode_start: int = 0
    epipe_prefill_extra_cache_resident_at_decode_end: int = 0
    epipe_no_evict_count: int = 0
    epipe_missing_h2d_bytes_est: int = 0
    epipe_demand_h2d_bytes: int = 0
    epipe_staged_missing_count: int = 0
    epipe_prefill_multistage_shadow_count: int = 0
    epipe_prefill_multistage_ready_expert_count: int = 0
    epipe_prefill_multistage_missing_expert_count: int = 0
    epipe_prefill_multistage_ready_item_count: int = 0
    epipe_prefill_multistage_missing_item_count: int = 0
    epipe_prefill_multistage_top_missing_item_count_sum: int = 0
    epipe_prefill_multistage_est_expert_h2d_bytes: int = 0
    epipe_prefill_multistage_est_ready_compute_items: int = 0
    epipe_prefill_multistage_est_hidden_h2d_items_n1: int = 0
    epipe_prefill_multistage_est_hidden_h2d_items_n2: int = 0
    epipe_prefill_multistage_est_hidden_h2d_items_n4: int = 0
    epipe_demand_h2d_ms: float = 0.0
    epipe_wait_ms: float = 0.0
    epipe_hidden_h2d_ms_est: float = 0.0
    sampled_demand_h2d_exposed_decode_ms: float = 0.0
    sampled_prefetch_wait_ms: float = 0.0
    sampled_epipe_hidden_h2d_ms: float = 0.0
    sampled_epipe_wait_ms: float = 0.0
    sampled_epipe_control_overhead_ms: float = 0.0
    comm_ms: float = 0.0
    expert_exec_ms: float = 0.0
    sim_expert_exec_ms: float = 0.0
    sim_compute_extra_sleep_ms: float = 0.0
    grouping_overhead_ms: float = 0.0
    rs_mlp_ms: float = 0.0
    runtime_overhead_ms: float = 0.0
    node_grouped_pack_ms: float = 0.0
    node_grouped_metadata_ms: float = 0.0
    node_grouped_intra_fanout_ms: float = 0.0
    node_grouped_inter_send_ms: float = 0.0
    node_grouped_inter_send_weaknet_sleep_ms: float = 0.0
    node_grouped_inter_send_stage1_weaknet_sleep_ms: float = 0.0
    node_grouped_inter_send_stage2_weaknet_sleep_ms: float = 0.0
    node_grouped_inter_send_stage1_collective_ms: float = 0.0
    node_grouped_inter_send_stage2_collective_ms: float = 0.0
    node_grouped_metadata_exchange_ms: float = 0.0
    node_grouped_gather_ms: float = 0.0
    node_grouped_total_overhead_ms: float = 0.0
    node_grouped_metadata_collective_count: int = 0
    node_grouped_metadata_bytes: int = 0
    node_grouped_counts_build_ms: float = 0.0
    node_grouped_splits_build_ms: float = 0.0
    node_grouped_reverse_map_build_ms: float = 0.0
    node_grouped_rank_to_node_gather_ms: float = 0.0
    node_grouped_node_to_token_restore_ms: float = 0.0
    node_grouped_final_combine_ms: float = 0.0
    node_grouped_combine_intermediate_buffer_bytes: int = 0
    node_grouped_allocation_ms: float = 0.0
    node_grouped_allocation_count: int = 0
    node_grouped_allocation_bytes: int = 0
    node_grouped_workspace_reuse_hit: int = 0
    node_grouped_enabled_count: int = 0
    node_grouped_fallback_count: int = 0
    node_grouped_estimated_saving_ms: float = 0.0
    local_assignments: int = 0
    remote_assignments: int = 0
    dispatch_self_assignments: int = 0
    dispatch_remote_assignments: int = 0
    expert_hist: Optional[torch.Tensor] = None

@dataclass
class Metrics:
    comm_bytes_activation: int = 0
    comm_bytes_activation_remote: int = 0
    comm_bytes_activation_intra_node: int = 0
    comm_bytes_activation_inter_node: int = 0
    weaknet_intra_node_item_count: int = 0
    weaknet_inter_node_item_count: int = 0
    weaknet_intra_node_transfer_count: int = 0
    weaknet_inter_node_transfer_count: int = 0
    weaknet_inter_node_packet_item_sumsq: int = 0
    weaknet_inter_node_packet_max_items: int = 0
    weaknet_inter_node_packet_le1_count: int = 0
    weaknet_inter_node_packet_le2_count: int = 0
    weaknet_inter_node_packet_le4_count: int = 0
    weaknet_inter_node_packet_le8_count: int = 0
    weaknet_inter_node_packet_gt8_count: int = 0
    weaknet_inter_node_small_packet_count: int = 0
    weaknet_inter_node_small_packet_item_count: int = 0
    weaknet_inter_node_pair_event_counts: Dict[str, int] = field(default_factory=dict)
    weaknet_inter_node_pair_item_counts: Dict[str, int] = field(default_factory=dict)
    weaknet_packet_trace: Optional[List[Dict[str, Any]]] = None
    physical_moe_comm_trace: Optional[List[Dict[str, Any]]] = None
    weaknet_inter_node_sim_ms: float = 0.0
    comm_bytes_kv: int = 0
    rank_grouped_pair_entries: int = 0
    rank_grouped_packet_entries: int = 0
    rank_grouped_dispatch_calls: int = 0
    node_grouped_pair_entries: int = 0
    node_grouped_node_packet_entries: int = 0
    node_grouped_rank_packet_entries: int = 0
    node_grouped_dispatch_calls: int = 0
    node_grouped_pack_ms: float = 0.0
    node_grouped_metadata_ms: float = 0.0
    node_grouped_intra_fanout_ms: float = 0.0
    node_grouped_inter_send_ms: float = 0.0
    node_grouped_inter_send_weaknet_sleep_ms: float = 0.0
    node_grouped_inter_send_stage1_weaknet_sleep_ms: float = 0.0
    node_grouped_inter_send_stage2_weaknet_sleep_ms: float = 0.0
    node_grouped_inter_send_stage1_collective_ms: float = 0.0
    node_grouped_inter_send_stage2_collective_ms: float = 0.0
    node_grouped_metadata_exchange_ms: float = 0.0
    node_grouped_gather_ms: float = 0.0
    node_grouped_total_overhead_ms: float = 0.0
    node_grouped_same_node_packet_entries: int = 0
    node_grouped_cross_node_packet_entries: int = 0
    node_grouped_same_node_bypass_bytes: int = 0
    node_grouped_stage2_local_rank_packet_entries: int = 0
    node_grouped_stage2_remote_rank_packet_entries: int = 0
    node_grouped_stage2_node_local_group: int = 0
    node_grouped_stage2_local_bypass_bytes: int = 0
    node_grouped_metadata_collective_count: int = 0
    node_grouped_metadata_bytes: int = 0
    node_grouped_counts_build_ms: float = 0.0
    node_grouped_splits_build_ms: float = 0.0
    node_grouped_reverse_map_build_ms: float = 0.0
    node_grouped_rank_to_node_gather_ms: float = 0.0
    node_grouped_node_to_token_restore_ms: float = 0.0
    node_grouped_final_combine_ms: float = 0.0
    node_grouped_combine_intermediate_buffer_bytes: int = 0
    node_grouped_allocation_ms: float = 0.0
    node_grouped_allocation_count: int = 0
    node_grouped_allocation_bytes: int = 0
    node_grouped_workspace_reuse_hit: int = 0
    node_grouped_enabled_layers: int = 0
    node_grouped_fallback_layers: int = 0
    node_grouped_estimated_saving_ms: float = 0.0

    expert_weight_h2d_bytes: int = 0
    expert_weight_h2d_prefetch_bytes: int = 0
    expert_weight_h2d_demand_bytes: int = 0

    skvs_accounted_bytes: int = 0

    prefetch_hit_count: int = 0
    prefetch_wasted_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    rs_pred_candidate_count: int = 0
    rs_pred_local_candidate_count: int = 0
    rs_pred_nonlocal_candidate_count: int = 0
    rs_candidate_filter_in_count: int = 0
    rs_candidate_filter_out_count: int = 0
    rs_candidate_filter_out_cached_count: int = 0
    rs_candidate_filter_out_low_score_count: int = 0
    rs_candidate_survived_count: int = 0
    prefetch_issued_count: int = 0
    prefetch_completed_count: int = 0
    prefetch_complete_before_use_count: int = 0
    prefetch_rejected_active_replace: int = 0
    prefetch_rejected_margin: int = 0
    prefetch_rejected_cache_full: int = 0
    prefetch_rejected_low_score: int = 0
    prefill_scp_transient_load_count: int = 0
    prefill_scp_transient_hit_count: int = 0
    prefill_scp_transient_promote_count: int = 0
    prefill_scp_transient_evict_count: int = 0
    prefill_scp_transient_resident_at_decode_start: int = 0
    prefill_scp_transient_demote_count: int = 0
    prefill_scp_transient_resident_at_decode_end: int = 0

    resolver_decode_total_queries: int = 0
    resolver_decode_rr_owner_count: int = 0
    resolver_decode_scp_replica_selected_count: int = 0
    resolver_decode_rr_fallback_count: int = 0
    resolver_decode_replica_reject_not_ready_count: int = 0
    resolver_decode_replica_reject_no_comm_gain_count: int = 0
    resolver_decode_replica_cache_ready_count: int = 0
    resolver_decode_replica_prefetch_ready_count: int = 0
    resolver_decode_route_cost_before: float = 0.0
    resolver_decode_route_cost_after: float = 0.0

    scp_prefill_route_diag_items: int = 0
    scp_prefill_route_diag_replica_items: int = 0
    scp_prefill_route_diag_cross_items_before: int = 0
    scp_prefill_route_diag_cross_items_after: int = 0
    scp_prefill_route_diag_cross_events_before: int = 0
    scp_prefill_route_diag_cross_events_after: int = 0
    scp_prefill_route_diag_cost_before: float = 0.0
    scp_prefill_route_diag_cost_after: float = 0.0
    scp_decode_route_diag_items: int = 0
    scp_decode_route_diag_replica_items: int = 0
    scp_decode_route_diag_cross_items_before: int = 0
    scp_decode_route_diag_cross_items_after: int = 0
    scp_decode_route_diag_cross_events_before: int = 0
    scp_decode_route_diag_cross_events_after: int = 0
    scp_decode_route_diag_cost_before: float = 0.0
    scp_decode_route_diag_cost_after: float = 0.0

    scp_table_local_replica_route_count: int = 0
    scp_table_rr_fallback_count: int = 0
    scp_table_cross_node_route_count: int = 0
    scp_table_same_node_route_count: int = 0
    scp_route_cost_before_sum: float = 0.0
    scp_route_cost_after_sum: float = 0.0
    scp_lookup_calls: int = 0
    scp_lookup_overhead_ms: float = 0.0

    rs_prefetch_target_resolver_match_count: int = 0
    rs_prefetch_target_resolver_mismatch_count: int = 0
    rs_prefetch_would_be_wrong_without_resolver_count: int = 0
    rs_scp_target_mismatch_count: int = 0

    oracle_stage_h2d_bytes: int = 0
    oracle_stage_event_count: int = 0
    oracle_stage_ms: float = 0.0
    oracle_stage_prefill_ms: float = 0.0
    oracle_stage_decode_ms: float = 0.0

    moe_router_ms: float = 0.0
    moe_expert_ms: float = 0.0
    moe_shared_ms: float = 0.0
    moe_timing_calls: int = 0
    sim_expert_exec_ms: float = 0.0
    sim_compute_extra_sleep_ms: float = 0.0
    rs_mlp_ms: float = 0.0
    rs_mlp_calls: int = 0
    rs_next_layer_pred_calls: int = 0

    # 当前真实系统里，因为 local expert 当场 miss 而产生的“额外等待时间”近似值。
    # 注意：这里统计的是 miss -> 立即 first-use 这条热路径上的暴露等待，
    # 不是简单的所有 H2D copy 时间总和。
    demand_h2d_exposed_ms: float = 0.0
    demand_h2d_exposed_prefill_ms: float = 0.0
    demand_h2d_exposed_decode_ms: float = 0.0
    sim_demand_h2d_exposed_ms: float = 0.0
    sim_demand_h2d_exposed_prefill_ms: float = 0.0
    sim_demand_h2d_exposed_decode_ms: float = 0.0
    sim_h2d_extra_sleep_ms: float = 0.0
    demand_h2d_event_count: int = 0
    _pending_demand_h2d_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event, str, int, int]]] = None

    prefetch_h2d_copy_ms: float = 0.0
    prefetch_h2d_event_count: int = 0
    prefetch_wait_ms: float = 0.0
    prefetch_wait_event_count: int = 0
    prefetch_oom_skip_count: int = 0
    _pending_prefetch_h2d_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event, int]]] = None
    _pending_prefetch_wait_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event, int]]] = None
    plain_ep_executor_calls: int = 0
    epipe_executor_calls: int = 0
    epipe_executor_attempt_count: int = 0

    rs_prefetch_eval_events: int = 0
    rs_prefetch_eval_pred_total: int = 0
    rs_prefetch_eval_actual_total: int = 0
    rs_prefetch_eval_overlap_total: int = 0
    prefetch_budget_zero_count: int = 0
    prefetch_budget_positive_count: int = 0
    prefetch_admitted_count: int = 0

    epipe_shadow_count: int = 0
    epipe_enabled_count: int = 0
    epipe_fallback_count: int = 0
    epipe_fallback_no_ready: int = 0
    epipe_fallback_no_missing: int = 0
    epipe_fallback_no_miss_trigger: int = 0
    epipe_fallback_no_ready_lookahead: int = 0
    epipe_fallback_low_ready_items: int = 0
    epipe_fallback_prefill_disabled: int = 0
    epipe_fallback_no_memory_headroom: int = 0
    epipe_fallback_memory: int = 0
    epipe_fallback_cache_full_no_temp: int = 0
    epipe_fallback_exception: int = 0
    epipe_decode_rs_miss_trigger_count: int = 0
    epipe_decode_rs_skipped_no_miss_count: int = 0
    epipe_prefill_miss_trigger_count: int = 0
    epipe_prefill_shadow_opportunity_count: int = 0
    epipe_rs_buffered_calls: int = 0
    epipe_rs_buffer_flush_count: int = 0
    epipe_rs_buffered_ready_group_count: int = 0
    epipe_rs_buffered_ready_item_count: int = 0
    epipe_rs_residual_scan_calls: int = 0
    epipe_rs_residual_scan_ready_group_count: int = 0
    epipe_rs_residual_scan_ready_item_count: int = 0
    epipe_rs_residual_scan_residual_count: int = 0
    epipe_rs_residual_scan_late_count: int = 0
    epipe_rs_residual_scan_demand_count: int = 0
    epipe_rs_residual_scan_no_ready_count: int = 0
    epipe_rs_trigger_count: int = 0
    epipe_rs_late_prefetch_trigger_count: int = 0
    epipe_rs_demand_miss_trigger_count: int = 0
    epipe_rs_late_prefetch_overlap_count: int = 0
    epipe_rs_demand_stage_count: int = 0
    epipe_rs_demand_stage_disabled_by_gate_count: int = 0
    epipe_rs_demand_stage_force_count: int = 0
    epipe_prefill_executor_calls: int = 0
    epipe_prefill_stage_success_count: int = 0
    epipe_decode_no_rs_executor_calls: int = 0
    epipe_decode_no_rs_stage_success_count: int = 0
    epipe_decode_rs_executor_calls: int = 0
    epipe_decode_rs_stage_success_count: int = 0
    epipe_prefill_enabled_count: int = 0
    epipe_decode_no_rs_enabled_count: int = 0
    epipe_decode_rs_late_overlap_count: int = 0
    epipe_decode_rs_demand_stage_count: int = 0
    epipe_rs_gate_skip_count: int = 0
    epipe_rs_gate_reenable_count: int = 0
    epipe_rs_buffer_too_small_count: int = 0
    epipe_lookahead_ready_group_count: int = 0
    epipe_lookahead_ready_item_count: int = 0
    epipe_staged_expert_count: int = 0
    epipe_staged_h2d_bytes: int = 0
    epipe_deferred_lru_commit_count: int = 0
    epipe_prefill_transient_commit_count: int = 0
    epipe_prefill_transient_demote_count: int = 0
    epipe_prefill_transient_promote_count: int = 0
    epipe_done_group_skip_count: int = 0
    epipe_ready_expert_count: int = 0
    epipe_missing_expert_count: int = 0
    epipe_ready_item_count: int = 0
    epipe_missing_item_count: int = 0
    epipe_cache_free_slot_count: int = 0
    epipe_cache_full_staged_count: int = 0
    epipe_transient_payload_count: int = 0
    epipe_prefill_transient_no_commit_count: int = 0
    epipe_prefill_admission_bypass_count: int = 0
    epipe_cache_insert_count: int = 0
    epipe_prefill_cache_insert_count: int = 0
    epipe_prefill_extra_cache_insert_count: int = 0
    epipe_prefill_extra_cache_hit_count: int = 0
    epipe_prefill_extra_cache_evict_count: int = 0
    epipe_prefill_extra_cache_no_room_count: int = 0
    epipe_prefill_extra_cache_guard_skip_count: int = 0
    epipe_prefill_extra_cache_h2d_bytes: int = 0
    epipe_prefill_extra_cache_resident_count: int = 0
    epipe_prefill_extra_cache_resident_at_decode_start: int = 0
    epipe_prefill_extra_cache_resident_at_decode_end: int = 0
    epipe_no_evict_count: int = 0
    epipe_missing_h2d_bytes_est: int = 0
    epipe_demand_h2d_bytes: int = 0
    epipe_staged_missing_count: int = 0
    epipe_prefill_multistage_shadow_count: int = 0
    epipe_prefill_multistage_ready_expert_count: int = 0
    epipe_prefill_multistage_missing_expert_count: int = 0
    epipe_prefill_multistage_ready_item_count: int = 0
    epipe_prefill_multistage_missing_item_count: int = 0
    epipe_prefill_multistage_top_missing_item_count_sum: int = 0
    epipe_prefill_multistage_est_expert_h2d_bytes: int = 0
    epipe_prefill_multistage_est_ready_compute_items: int = 0
    epipe_prefill_multistage_est_hidden_h2d_items_n1: int = 0
    epipe_prefill_multistage_est_hidden_h2d_items_n2: int = 0
    epipe_prefill_multistage_est_hidden_h2d_items_n4: int = 0
    epipe_demand_h2d_ms: float = 0.0
    epipe_wait_ms: float = 0.0
    epipe_hidden_h2d_ms_est: float = 0.0
    sampled_demand_h2d_exposed_decode_ms: float = 0.0
    sampled_prefetch_wait_ms: float = 0.0
    sampled_epipe_hidden_h2d_ms: float = 0.0
    sampled_epipe_wait_ms: float = 0.0
    sampled_epipe_control_overhead_ms: float = 0.0
    _pending_epipe_timing_events: Optional[List[Tuple[Any, ...]]] = None

    wall_prefill_ms: float = 0.0
    wall_decode_ms: float = 0.0
    wall_ttft_ms: float = 0.0
    wall_total_ms: float = 0.0

    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0

    prompt_tokens: int = 0
    gen_tokens: int = 0

    owner_group_count: int = 0
    owner_group_reconstruct_ms: float = 0.0
    owner_group_plan_ms: float = 0.0
    owner_group_migrate_ms: float = 0.0
    owner_changed_samples: int = 0
    owner_empty_ranks_sum: int = 0
    owner_hist_max_sum: float = 0.0
    owner_hist_min_sum: float = 0.0
    owner_hist_cv_sum: float = 0.0
    owner_imbalance_ratio_sum: float = 0.0

    owner_attention_ms: float = 0.0
    owner_attention_calls: int = 0
    owner_moe_ms: float = 0.0
    owner_moe_calls: int = 0

    decode_step_stats: Optional[RunningStats] = None
    per_layer: Optional[Dict[int, LayerPerf]] = None

    def reset_for_repeat(self) -> None:
        self.comm_bytes_activation = 0
        self.comm_bytes_activation_remote = 0
        self.comm_bytes_activation_intra_node = 0
        self.comm_bytes_activation_inter_node = 0
        self.weaknet_intra_node_item_count = 0
        self.weaknet_inter_node_item_count = 0
        self.weaknet_intra_node_transfer_count = 0
        self.weaknet_inter_node_transfer_count = 0
        self.weaknet_inter_node_packet_item_sumsq = 0
        self.weaknet_inter_node_packet_max_items = 0
        self.weaknet_inter_node_packet_le1_count = 0
        self.weaknet_inter_node_packet_le2_count = 0
        self.weaknet_inter_node_packet_le4_count = 0
        self.weaknet_inter_node_packet_le8_count = 0
        self.weaknet_inter_node_packet_gt8_count = 0
        self.weaknet_inter_node_small_packet_count = 0
        self.weaknet_inter_node_small_packet_item_count = 0
        self.weaknet_inter_node_pair_event_counts = {}
        self.weaknet_inter_node_pair_item_counts = {}
        self.weaknet_packet_trace = [] if _weaknet_packet_trace_enabled() else None
        self.physical_moe_comm_trace = (
            [] if _env_flag("SPARSE_PHYSICAL_MOE_COMM_TRACE", "0") else None
        )
        self.weaknet_inter_node_sim_ms = 0.0
        self.comm_bytes_kv = 0
        self.rank_grouped_pair_entries = 0
        self.rank_grouped_packet_entries = 0
        self.rank_grouped_dispatch_calls = 0
        self.node_grouped_pair_entries = 0
        self.node_grouped_node_packet_entries = 0
        self.node_grouped_rank_packet_entries = 0
        self.node_grouped_dispatch_calls = 0
        self.node_grouped_pack_ms = 0.0
        self.node_grouped_metadata_ms = 0.0
        self.node_grouped_intra_fanout_ms = 0.0
        self.node_grouped_inter_send_ms = 0.0
        self.node_grouped_inter_send_weaknet_sleep_ms = 0.0
        self.node_grouped_inter_send_stage1_weaknet_sleep_ms = 0.0
        self.node_grouped_inter_send_stage2_weaknet_sleep_ms = 0.0
        self.node_grouped_inter_send_stage1_collective_ms = 0.0
        self.node_grouped_inter_send_stage2_collective_ms = 0.0
        self.node_grouped_metadata_exchange_ms = 0.0
        self.node_grouped_gather_ms = 0.0
        self.node_grouped_total_overhead_ms = 0.0
        self.node_grouped_same_node_packet_entries = 0
        self.node_grouped_cross_node_packet_entries = 0
        self.node_grouped_same_node_bypass_bytes = 0
        self.node_grouped_stage2_local_rank_packet_entries = 0
        self.node_grouped_stage2_remote_rank_packet_entries = 0
        self.node_grouped_stage2_node_local_group = 0
        self.node_grouped_stage2_local_bypass_bytes = 0
        self.node_grouped_metadata_collective_count = 0
        self.node_grouped_metadata_bytes = 0
        self.node_grouped_counts_build_ms = 0.0
        self.node_grouped_splits_build_ms = 0.0
        self.node_grouped_reverse_map_build_ms = 0.0
        self.node_grouped_rank_to_node_gather_ms = 0.0
        self.node_grouped_node_to_token_restore_ms = 0.0
        self.node_grouped_final_combine_ms = 0.0
        self.node_grouped_combine_intermediate_buffer_bytes = 0
        self.node_grouped_allocation_ms = 0.0
        self.node_grouped_allocation_count = 0
        self.node_grouped_allocation_bytes = 0
        self.node_grouped_workspace_reuse_hit = 0
        self.node_grouped_enabled_layers = 0
        self.node_grouped_fallback_layers = 0
        self.node_grouped_estimated_saving_ms = 0.0

        self.expert_weight_h2d_bytes = 0
        self.expert_weight_h2d_prefetch_bytes = 0
        self.expert_weight_h2d_demand_bytes = 0

        self.skvs_accounted_bytes = 0

        self.prefetch_hit_count = 0
        self.prefetch_wasted_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.prefill_scp_transient_load_count = 0
        self.prefill_scp_transient_hit_count = 0
        self.prefill_scp_transient_promote_count = 0
        self.prefill_scp_transient_evict_count = 0
        self.prefill_scp_transient_resident_at_decode_start = 0
        self.prefill_scp_transient_demote_count = 0
        self.prefill_scp_transient_resident_at_decode_end = 0

        self.resolver_decode_total_queries = 0
        self.resolver_decode_rr_owner_count = 0
        self.resolver_decode_scp_replica_selected_count = 0
        self.resolver_decode_rr_fallback_count = 0
        self.resolver_decode_replica_reject_not_ready_count = 0
        self.resolver_decode_replica_reject_no_comm_gain_count = 0
        self.resolver_decode_replica_cache_ready_count = 0
        self.resolver_decode_replica_prefetch_ready_count = 0
        self.resolver_decode_route_cost_before = 0.0
        self.resolver_decode_route_cost_after = 0.0

        self.scp_prefill_route_diag_items = 0
        self.scp_prefill_route_diag_replica_items = 0
        self.scp_prefill_route_diag_cross_items_before = 0
        self.scp_prefill_route_diag_cross_items_after = 0
        self.scp_prefill_route_diag_cross_events_before = 0
        self.scp_prefill_route_diag_cross_events_after = 0
        self.scp_prefill_route_diag_cost_before = 0.0
        self.scp_prefill_route_diag_cost_after = 0.0
        self.scp_decode_route_diag_items = 0
        self.scp_decode_route_diag_replica_items = 0
        self.scp_decode_route_diag_cross_items_before = 0
        self.scp_decode_route_diag_cross_items_after = 0
        self.scp_decode_route_diag_cross_events_before = 0
        self.scp_decode_route_diag_cross_events_after = 0
        self.scp_decode_route_diag_cost_before = 0.0
        self.scp_decode_route_diag_cost_after = 0.0

        self.scp_table_local_replica_route_count = 0
        self.scp_table_rr_fallback_count = 0
        self.scp_table_cross_node_route_count = 0
        self.scp_table_same_node_route_count = 0
        self.scp_route_cost_before_sum = 0.0
        self.scp_route_cost_after_sum = 0.0
        self.scp_lookup_calls = 0
        self.scp_lookup_overhead_ms = 0.0

        self.rs_prefetch_target_resolver_match_count = 0
        self.rs_prefetch_target_resolver_mismatch_count = 0
        self.rs_prefetch_would_be_wrong_without_resolver_count = 0
        self.rs_scp_target_mismatch_count = 0

        self.oracle_stage_h2d_bytes = 0
        self.oracle_stage_event_count = 0
        self.oracle_stage_ms = 0.0
        self.oracle_stage_prefill_ms = 0.0
        self.oracle_stage_decode_ms = 0.0

        self.moe_router_ms = 0.0
        self.moe_expert_ms = 0.0
        self.moe_shared_ms = 0.0
        self.moe_timing_calls = 0
        self.sim_expert_exec_ms = 0.0
        self.sim_compute_extra_sleep_ms = 0.0
        self.rs_mlp_ms = 0.0
        self.rs_mlp_calls = 0
        self.rs_next_layer_pred_calls = 0

        self.demand_h2d_exposed_ms = 0.0
        self.demand_h2d_exposed_prefill_ms = 0.0
        self.demand_h2d_exposed_decode_ms = 0.0
        self.sim_demand_h2d_exposed_ms = 0.0
        self.sim_demand_h2d_exposed_prefill_ms = 0.0
        self.sim_demand_h2d_exposed_decode_ms = 0.0
        self.sim_h2d_extra_sleep_ms = 0.0
        self.demand_h2d_event_count = 0
        self._pending_demand_h2d_events = []

        self.prefetch_h2d_copy_ms = 0.0
        self.prefetch_h2d_event_count = 0
        self.prefetch_wait_ms = 0.0
        self.prefetch_wait_event_count = 0
        self.prefetch_oom_skip_count = 0
        self._pending_prefetch_h2d_events = []
        self._pending_prefetch_wait_events = []
        self.plain_ep_executor_calls = 0
        self.epipe_executor_calls = 0
        self.epipe_executor_attempt_count = 0

        self.rs_prefetch_eval_events = 0
        self.rs_prefetch_eval_pred_total = 0
        self.rs_prefetch_eval_actual_total = 0
        self.rs_prefetch_eval_overlap_total = 0
        self.prefetch_budget_zero_count = 0
        self.prefetch_budget_positive_count = 0
        self.prefetch_admitted_count = 0
        self.rs_pred_candidate_count = 0
        self.rs_pred_local_candidate_count = 0
        self.rs_pred_nonlocal_candidate_count = 0
        self.rs_candidate_filter_in_count = 0
        self.rs_candidate_filter_out_count = 0
        self.rs_candidate_filter_out_cached_count = 0
        self.rs_candidate_filter_out_low_score_count = 0
        self.rs_candidate_survived_count = 0
        self.prefetch_issued_count = 0
        self.prefetch_completed_count = 0
        self.prefetch_complete_before_use_count = 0
        self.prefetch_rejected_active_replace = 0
        self.prefetch_rejected_margin = 0
        self.prefetch_rejected_cache_full = 0
        self.prefetch_rejected_low_score = 0

        self.ttl_prevented_evictions = 0
        self.cache_eviction_fallback_all_protected = 0

        self.warmup_loaded_experts = 0
        self.warmup_skipped_resident = 0
        self.warmup_h2d_bytes = 0
        self.warmup_time_ms = 0.0

        self.transition_lookup_calls = 0
        self.transition_lookup_overhead_ms = 0.0
        self.transition_prefetch_admitted = 0
        self.transition_prefetch_hit = 0
        self.transition_prefetch_late = 0
        self.transition_prefetch_wasted = 0
        self.transition_safe_evict_denied = 0
        self.transition_no_replace_denied = 0

        self.epipe_shadow_count = 0
        self.epipe_enabled_count = 0
        self.epipe_fallback_count = 0
        self.epipe_fallback_no_ready = 0
        self.epipe_fallback_no_missing = 0
        self.epipe_fallback_no_miss_trigger = 0
        self.epipe_fallback_no_ready_lookahead = 0
        self.epipe_fallback_low_ready_items = 0
        self.epipe_fallback_prefill_disabled = 0
        self.epipe_fallback_no_memory_headroom = 0
        self.epipe_fallback_memory = 0
        self.epipe_fallback_cache_full_no_temp = 0
        self.epipe_fallback_exception = 0
        self.epipe_decode_rs_miss_trigger_count = 0
        self.epipe_decode_rs_skipped_no_miss_count = 0
        self.epipe_prefill_miss_trigger_count = 0
        self.epipe_prefill_shadow_opportunity_count = 0
        self.epipe_rs_buffered_calls = 0
        self.epipe_rs_buffer_flush_count = 0
        self.epipe_rs_buffered_ready_group_count = 0
        self.epipe_rs_buffered_ready_item_count = 0
        self.epipe_rs_residual_scan_calls = 0
        self.epipe_rs_residual_scan_ready_group_count = 0
        self.epipe_rs_residual_scan_ready_item_count = 0
        self.epipe_rs_residual_scan_residual_count = 0
        self.epipe_rs_residual_scan_late_count = 0
        self.epipe_rs_residual_scan_demand_count = 0
        self.epipe_rs_residual_scan_no_ready_count = 0
        self.epipe_rs_trigger_count = 0
        self.epipe_rs_late_prefetch_trigger_count = 0
        self.epipe_rs_demand_miss_trigger_count = 0
        self.epipe_rs_late_prefetch_overlap_count = 0
        self.epipe_rs_demand_stage_count = 0
        self.epipe_rs_demand_stage_disabled_by_gate_count = 0
        self.epipe_rs_demand_stage_force_count = 0
        self.epipe_prefill_executor_calls = 0
        self.epipe_prefill_stage_success_count = 0
        self.epipe_decode_no_rs_executor_calls = 0
        self.epipe_decode_no_rs_stage_success_count = 0
        self.epipe_decode_rs_executor_calls = 0
        self.epipe_decode_rs_stage_success_count = 0
        self.epipe_prefill_enabled_count = 0
        self.epipe_decode_no_rs_enabled_count = 0
        self.epipe_decode_rs_late_overlap_count = 0
        self.epipe_decode_rs_demand_stage_count = 0
        self.epipe_rs_gate_skip_count = 0
        self.epipe_rs_gate_reenable_count = 0
        self.epipe_rs_buffer_too_small_count = 0
        self.epipe_lookahead_ready_group_count = 0
        self.epipe_lookahead_ready_item_count = 0
        self.epipe_staged_expert_count = 0
        self.epipe_staged_h2d_bytes = 0
        self.epipe_deferred_lru_commit_count = 0
        self.epipe_prefill_transient_commit_count = 0
        self.epipe_prefill_transient_demote_count = 0
        self.epipe_prefill_transient_promote_count = 0
        self.epipe_done_group_skip_count = 0
        self.epipe_ready_expert_count = 0
        self.epipe_missing_expert_count = 0
        self.epipe_ready_item_count = 0
        self.epipe_missing_item_count = 0
        self.epipe_cache_free_slot_count = 0
        self.epipe_cache_full_staged_count = 0
        self.epipe_transient_payload_count = 0
        self.epipe_prefill_transient_no_commit_count = 0
        self.epipe_prefill_admission_bypass_count = 0
        self.epipe_cache_insert_count = 0
        self.epipe_prefill_cache_insert_count = 0
        self.epipe_prefill_extra_cache_insert_count = 0
        self.epipe_prefill_extra_cache_hit_count = 0
        self.epipe_prefill_extra_cache_evict_count = 0
        self.epipe_prefill_extra_cache_no_room_count = 0
        self.epipe_prefill_extra_cache_guard_skip_count = 0
        self.epipe_prefill_extra_cache_h2d_bytes = 0
        self.epipe_prefill_extra_cache_resident_count = 0
        self.epipe_prefill_extra_cache_resident_at_decode_start = 0
        self.epipe_prefill_extra_cache_resident_at_decode_end = 0
        self.epipe_no_evict_count = 0
        self.epipe_missing_h2d_bytes_est = 0
        self.epipe_demand_h2d_bytes = 0
        self.epipe_staged_missing_count = 0
        self.epipe_prefill_multistage_shadow_count = 0
        self.epipe_prefill_multistage_ready_expert_count = 0
        self.epipe_prefill_multistage_missing_expert_count = 0
        self.epipe_prefill_multistage_ready_item_count = 0
        self.epipe_prefill_multistage_missing_item_count = 0
        self.epipe_prefill_multistage_top_missing_item_count_sum = 0
        self.epipe_prefill_multistage_est_expert_h2d_bytes = 0
        self.epipe_prefill_multistage_est_ready_compute_items = 0
        self.epipe_prefill_multistage_est_hidden_h2d_items_n1 = 0
        self.epipe_prefill_multistage_est_hidden_h2d_items_n2 = 0
        self.epipe_prefill_multistage_est_hidden_h2d_items_n4 = 0
        self.epipe_demand_h2d_ms = 0.0
        self.epipe_wait_ms = 0.0
        self.epipe_hidden_h2d_ms_est = 0.0
        self.sampled_demand_h2d_exposed_decode_ms = 0.0
        self.sampled_prefetch_wait_ms = 0.0
        self.sampled_epipe_hidden_h2d_ms = 0.0
        self.sampled_epipe_wait_ms = 0.0
        self.sampled_epipe_control_overhead_ms = 0.0
        self._pending_epipe_timing_events = []

        self.wall_prefill_ms = 0.0
        self.wall_decode_ms = 0.0
        self.wall_ttft_ms = 0.0
        self.wall_total_ms = 0.0
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0

        self.owner_group_count = 0
        self.owner_group_reconstruct_ms = 0.0
        self.owner_group_plan_ms = 0.0
        self.owner_group_migrate_ms = 0.0
        self.owner_changed_samples = 0
        self.owner_empty_ranks_sum = 0
        self.owner_hist_max_sum = 0.0
        self.owner_hist_min_sum = 0.0
        self.owner_hist_cv_sum = 0.0
        self.owner_imbalance_ratio_sum = 0.0
        self.owner_attention_ms = 0.0
        self.owner_attention_calls = 0
        self.owner_moe_ms = 0.0
        self.owner_moe_calls = 0

        if self.decode_step_stats is not None:
            self.decode_step_stats.reset()

        if self.per_layer is not None:
            self.per_layer.clear()


# ---------------- routing cache + stats ----------------
class SparseContext:
    num_layers: int = 0
    num_experts: int = 0
    routing_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    rs_bank: Optional[nn.ModuleList] = None
    late_rs_bank: Optional[nn.ModuleList] = None
    rs_group: int = 3
    rs_bank_layout: str = "group"
    rs_bank_source_layers: List[int] = []
    late_rs_bank_layout: str = "group"
    late_rs_bank_source_layers: List[int] = []

    layer_blocks: Dict[int, "DistributedSparseMoeBlock"] = {}
    layer_attn_modules: Dict[int, nn.Module] = {}

    # [B]，每个 batch element 一个 request owner
    request_owner_ranks: Optional[torch.Tensor] = None

    stats_enabled: bool = False
    stats_num_experts: int = 0
    _seen_expert_masks: Optional[List[torch.Tensor]] = None
    _entropy_sum: Optional[List[torch.Tensor]] = None
    _token_count: Optional[List[torch.Tensor]] = None
    # 当前 group 的 sample -> owner rank
    group_owner_ranks: Optional[torch.Tensor] = None   # [B]
    # 当前 group 各层的 RS 预测缓存
    group_pred_topi: Dict[int, torch.Tensor] = {}
    group_pred_topv: Dict[int, torch.Tensor] = {}
    group_pred_value: Dict[int, torch.Tensor] = {}
    prefill_expert_counts: Dict[int, Dict[int, int]] = {}
    transition_source_experts: Dict[int, set[int]] = {}
    rs_bank_task: str = "router_prob"
    late_rs_bank_task: str = "router_prob"
    nodepatch_stats: Dict[str, float] = {}
    suppress_logical_weaknet_once: bool = False
    logical_ngd_ep_reuse_once: bool = False
    ngd_stage2_node_group_cache: Dict[
        Tuple[int, Tuple[int, ...], Tuple[int, ...]],
        Dict[int, Tuple[Any, Tuple[int, ...], Tuple[int, ...]]],
    ] = {}

    oracle_mode: str = "none"
    oracle_trace_dir: str = ""
    oracle_call_id: int = -1
    runtime_call_id: int = -1
    current_phase: str = "none"
    oracle_active_layer_idx: Optional[int] = None
    oracle_trace: Dict[int, Dict[int, List[int]]] = {}
    current_moe_layer_idx: Optional[int] = None
    global_expert_cache_enabled: bool = False
    global_expert_cache_capacity: int = 0
    prefill_scp_transient_demoted_for_decode: bool = False

    local_sample_ids: Optional[torch.Tensor] = None
    local_full_batch_size: int = 0

    _runtime_decode_local: bool = False

    routing_profile_initialized: bool = False
    routing_profile_metadata: Dict[str, Any] = {}
    routing_profile_prefill: Optional[np.ndarray] = None
    routing_profile_decode: Optional[np.ndarray] = None
    routing_profile_prefill_rows_by_layer: Optional[np.ndarray] = None
    routing_profile_decode_rows_by_layer: Optional[np.ndarray] = None
    routing_profile_trace_stage: List[int] = []
    routing_profile_trace_call_id: List[int] = []
    routing_profile_trace_step: List[int] = []
    routing_profile_trace_layer: List[int] = []
    routing_profile_trace_source_rank: List[int] = []
    routing_profile_trace_expert: List[int] = []
    routing_profile_trace_count: List[int] = []
    routing_profile_trace_token_rows: List[int] = []
    routing_profile_trace_shard_id: List[int] = []
    routing_profile_trace_request_id: List[int] = []
    routing_profile_trace_global_order_id: List[int] = []
    routing_profile_trace_source_node: List[int] = []
    routing_profile_trace_task_id: List[int] = []
    routing_profile_trace_task_family: List[int] = []
    routing_profile_trace_request_global_id: List[int] = []
    routing_profile_trace_order_counter: int = 0
    task_local_metadata: Dict[str, Any] = {}
    task_local_request_task_id: List[int] = []
    task_local_request_task_family: List[int] = []
    task_local_request_source_node: List[int] = []
    task_local_request_global_id: List[int] = []
    task_local_string_table: Dict[str, Dict[str, int]] = {"task_id": {}, "task_family": {}}

    @classmethod
    def set_runtime_decode_local(cls, flag: bool) -> None:
        cls._runtime_decode_local = bool(flag)

    @classmethod
    def is_runtime_decode_local(cls) -> bool:
        return bool(cls._runtime_decode_local)

    @classmethod
    def initialize(cls, num_layers: int, rs_group: int = 3, num_experts: int = 0) -> None:
        cls.num_layers = int(num_layers)
        cls.num_experts = int(num_experts)
        cls.rs_group = int(rs_group)
        cls.routing_cache.clear()
        cls.rs_bank = None
        cls.late_rs_bank = None
        cls.rs_bank_layout = "group"
        cls.rs_bank_source_layers = []
        cls.late_rs_bank_layout = "group"
        cls.late_rs_bank_source_layers = []
        cls.layer_blocks = {}
        cls.layer_attn_modules = {}
        cls.request_owner_ranks = None
        cls.group_owner_ranks = None
        cls.group_pred_topi = {}
        cls.group_pred_topv = {}
        cls.group_pred_value = {}
        cls.prefill_expert_counts = {}
        cls.transition_source_experts = {}
        cls.rs_bank_task = "router_prob"
        cls.late_rs_bank_task = "router_prob"
        cls.reset_nodepatch_stats()
        cls.oracle_mode = "none"
        cls.oracle_trace_dir = ""
        cls.oracle_call_id = -1
        cls.runtime_call_id = -1
        cls.current_phase = "none"
        cls.oracle_active_layer_idx = None
        cls.oracle_trace = {}
        cls.current_moe_layer_idx = None
        cls.global_expert_cache_enabled = False
        cls.global_expert_cache_capacity = 0
        cls.prefill_scp_transient_demoted_for_decode = False
        cls.local_sample_ids = None
        cls.local_full_batch_size = 0
        cls._runtime_decode_local = False
        cls.reset_routing_profile()
        cls.disable_stats()

    @classmethod
    def reset_routing_profile(cls) -> None:
        cls.routing_profile_initialized = False
        cls.routing_profile_metadata = {}
        cls.routing_profile_prefill = None
        cls.routing_profile_decode = None
        cls.routing_profile_prefill_rows_by_layer = None
        cls.routing_profile_decode_rows_by_layer = None
        cls.routing_profile_trace_stage = []
        cls.routing_profile_trace_call_id = []
        cls.routing_profile_trace_step = []
        cls.routing_profile_trace_layer = []
        cls.routing_profile_trace_source_rank = []
        cls.routing_profile_trace_expert = []
        cls.routing_profile_trace_count = []
        cls.routing_profile_trace_token_rows = []
        cls.routing_profile_trace_shard_id = []
        cls.routing_profile_trace_request_id = []
        cls.routing_profile_trace_global_order_id = []
        cls.routing_profile_trace_source_node = []
        cls.routing_profile_trace_task_id = []
        cls.routing_profile_trace_task_family = []
        cls.routing_profile_trace_request_global_id = []
        cls.routing_profile_trace_order_counter = 0

    @classmethod
    def set_task_local_metadata(cls, metadata: Optional[Dict[str, Any]]) -> None:
        meta = dict(metadata or {})
        cls.task_local_metadata = meta
        cls.task_local_request_task_id = []
        cls.task_local_request_task_family = []
        cls.task_local_request_source_node = []
        cls.task_local_request_global_id = []
        cls.task_local_string_table = {"task_id": {}, "task_family": {}}
        request_profile_enabled = bool(meta.get("request_profile_enabled", False))
        if not bool(meta.get("task_local_enabled", False)) and not request_profile_enabled:
            return

        def intern(kind: str, value: Any) -> int:
            table = cls.task_local_string_table.setdefault(str(kind), {})
            key = str(value)
            if key not in table:
                table[key] = int(len(table))
            return int(table[key])

        requests = meta.get("requests", []) or []
        for req in requests:
            if not isinstance(req, dict):
                continue
            cls.task_local_request_task_id.append(intern("task_id", req.get("task_id", "unknown")))
            cls.task_local_request_task_family.append(intern("task_family", req.get("task_family", req.get("task_id", "unknown"))))
            cls.task_local_request_source_node.append(int(req.get("source_node", meta.get("source_node", 0))))
            cls.task_local_request_global_id.append(int(req.get("request_global_id", 0)))
        meta["task_string_table"] = {
            kind: {str(idx): value for value, idx in table.items()}
            for kind, table in cls.task_local_string_table.items()
        }
        cls.task_local_metadata = meta

    @classmethod
    def init_routing_profile(
        cls,
        *,
        num_layers: int,
        world_size: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        dtype: str,
    ) -> None:
        if cls.routing_profile_initialized:
            return
        shape = (int(num_layers), int(world_size), int(num_experts))
        cls.routing_profile_prefill = np.zeros(shape, dtype=np.int64)
        cls.routing_profile_decode = np.zeros(shape, dtype=np.int64)
        cls.routing_profile_prefill_rows_by_layer = np.zeros((int(num_layers), int(world_size)), dtype=np.int64)
        cls.routing_profile_decode_rows_by_layer = np.zeros((int(num_layers), int(world_size)), dtype=np.int64)
        dtype_bytes = 0
        try:
            dtype_bytes = int(torch.empty((), dtype=getattr(torch, str(dtype).split(".")[-1], torch.float16)).element_size())
        except Exception:
            dtype_bytes = 0
        cls.routing_profile_metadata = {
            "format": "scp_ep_routing_profile_rank_v1",
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
            "world_size": int(world_size),
            "top_k": int(top_k),
            "dtype": str(dtype),
            "hidden_size": int(hidden_size),
            "bytes_per_token": int(hidden_size) * int(dtype_bytes) if dtype_bytes > 0 else 0,
            "total_prefill_tokens": 0,
            "total_decode_tokens": 0,
            "total_prefill_assignments": 0,
            "total_decode_assignments": 0,
            "stage_semantics": "prefill/decode are inferred from SparseContext.current_phase; token totals are routing rows summed across MoE layers on this rank.",
            "ordered_trace_semantics": "Sparse per-MoE-call expert-count events appended in runtime call order. trace_step is runtime_call_id for decode and 0 for prefill. trace_shard_id and trace_request_id identify the profiled shard/request group; trace_global_order_id is made globally monotonic again by the merge step.",
        }
        profile_meta = getattr(cls, "task_local_metadata", {}) or {}
        if bool(profile_meta.get("task_local_enabled", False) or profile_meta.get("request_profile_enabled", False)):
            cls.routing_profile_metadata.update(dict(cls.task_local_metadata))
        cls.routing_profile_initialized = True

    @classmethod
    def record_routing_profile(
        cls,
        *,
        layer_idx: int,
        source_rank: int,
        topk_indices: torch.Tensor,
        num_layers: int,
        world_size: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        dtype: str,
    ) -> None:
        if not _profile_routing_enabled():
            return
        stage = str(getattr(cls, "current_phase", "none") or "none").strip().lower()
        if stage not in ("prefill", "decode"):
            return
        if not _profile_routing_stage_allowed(stage):
            return
        if topk_indices is None or int(topk_indices.numel()) == 0:
            return
        cls.init_routing_profile(
            num_layers=int(num_layers),
            world_size=int(world_size),
            num_experts=int(num_experts),
            top_k=int(top_k),
            hidden_size=int(hidden_size),
            dtype=str(dtype),
        )
        if cls.routing_profile_prefill is None or cls.routing_profile_decode is None:
            return
        layer = int(layer_idx)
        src = int(source_rank)
        if layer < 0 or layer >= int(num_layers) or src < 0 or src >= int(world_size):
            return
        token_rows = int(topk_indices.reshape(-1, int(top_k)).shape[0]) if int(top_k) > 0 else int(topk_indices.shape[0])
        call_id = int(getattr(cls, "runtime_call_id", -1))
        if call_id < 0:
            call_id = int(getattr(cls, "oracle_call_id", -1))
        trace_stage = 0 if stage == "prefill" else 1
        trace_step = 0 if stage == "prefill" else int(call_id)
        trace_shard_id = _env_int("SPARSE_PROFILE_ROUTING_SHARD_ID", 0)
        trace_request_id = _env_int("SPARSE_PROFILE_ROUTING_REQUEST_ID", 0)
        topk_cpu = topk_indices.detach().reshape(-1, int(top_k)).to(device="cpu", dtype=torch.long) if int(top_k) > 0 else topk_indices.detach().reshape(-1, 1).to(device="cpu", dtype=torch.long)
        if int(topk_cpu.numel()) == 0:
            return
        flat = topk_cpu.reshape(-1)
        counts = torch.bincount(flat.clamp(0, int(num_experts) - 1), minlength=int(num_experts)).cpu().numpy().astype(np.int64)

        task_meta = getattr(cls, "task_local_metadata", {}) or {}
        task_enabled = bool(task_meta.get("task_local_enabled", False) or task_meta.get("request_profile_enabled", False))
        request_count = len(getattr(cls, "task_local_request_global_id", []) or [])
        if task_enabled and request_count > 0 and int(token_rows) % int(request_count) == 0:
            rows_per_request = int(token_rows) // int(request_count)
            row_groups = [
                (int(row), int(row) * int(rows_per_request), (int(row) + 1) * int(rows_per_request))
                for row in range(int(request_count))
            ]
        else:
            row_groups = [(-1, 0, int(token_rows))]

        for row, row_start, row_end in row_groups:
            if int(row) >= 0:
                row_flat = topk_cpu[int(row_start):int(row_end)].reshape(-1)
                row_counts = torch.bincount(row_flat.clamp(0, int(num_experts) - 1), minlength=int(num_experts)).cpu().numpy().astype(np.int64)
                source_node = int(cls.task_local_request_source_node[int(row)])
                task_id = int(cls.task_local_request_task_id[int(row)])
                task_family = int(cls.task_local_request_task_family[int(row)])
                request_global_id = int(cls.task_local_request_global_id[int(row)])
                row_token_rows = int(max(0, int(row_end) - int(row_start)))
            else:
                row_counts = counts
                source_node = int(_parse_logical_node_map(int(world_size))[int(src)]) if int(src) < int(world_size) else int(src)
                task_id = -1
                task_family = -1
                # Non task-local profiles still need a stable request key for
                # activation-cluster SCP.  Keep execution unchanged and only
                # disambiguate the profiled request group by source rank.
                request_global_id = int(trace_request_id) * 1_000_000 + int(src)
                row_token_rows = int(token_rows)
            nz = np.nonzero(row_counts)[0]
            for expert_id in nz.tolist():
                cnt = int(row_counts[int(expert_id)])
                if cnt <= 0:
                    continue
                order_id = int(cls.routing_profile_trace_order_counter)
                cls.routing_profile_trace_order_counter += 1
                cls.routing_profile_trace_stage.append(int(trace_stage))
                cls.routing_profile_trace_call_id.append(int(call_id))
                cls.routing_profile_trace_step.append(int(trace_step))
                cls.routing_profile_trace_layer.append(int(layer))
                cls.routing_profile_trace_source_rank.append(int(src))
                cls.routing_profile_trace_expert.append(int(expert_id))
                cls.routing_profile_trace_count.append(int(cnt))
                cls.routing_profile_trace_token_rows.append(int(row_token_rows))
                cls.routing_profile_trace_shard_id.append(int(trace_shard_id))
                cls.routing_profile_trace_request_id.append(int(request_global_id))
                cls.routing_profile_trace_global_order_id.append(int(order_id))
                cls.routing_profile_trace_source_node.append(int(source_node))
                cls.routing_profile_trace_task_id.append(int(task_id))
                cls.routing_profile_trace_task_family.append(int(task_family))
                cls.routing_profile_trace_request_global_id.append(int(request_global_id))
        if stage == "prefill":
            cls.routing_profile_prefill[layer, src, :] += counts
            if cls.routing_profile_prefill_rows_by_layer is not None:
                cls.routing_profile_prefill_rows_by_layer[layer, src] += int(token_rows)
            cls.routing_profile_metadata["total_prefill_tokens"] = int(cls.routing_profile_metadata.get("total_prefill_tokens", 0)) + int(token_rows)
            cls.routing_profile_metadata["total_prefill_assignments"] = int(cls.routing_profile_metadata.get("total_prefill_assignments", 0)) + int(flat.numel())
        else:
            cls.routing_profile_decode[layer, src, :] += counts
            if cls.routing_profile_decode_rows_by_layer is not None:
                cls.routing_profile_decode_rows_by_layer[layer, src] += int(token_rows)
            cls.routing_profile_metadata["total_decode_tokens"] = int(cls.routing_profile_metadata.get("total_decode_tokens", 0)) + int(token_rows)
            cls.routing_profile_metadata["total_decode_assignments"] = int(cls.routing_profile_metadata.get("total_decode_assignments", 0)) + int(flat.numel())

    @classmethod
    def flush_routing_profile(cls, rank: int) -> Optional[str]:
        if not _profile_routing_enabled() or not cls.routing_profile_initialized:
            return None
        if cls.routing_profile_prefill is None or cls.routing_profile_decode is None:
            return None
        raw_out = str(os.environ.get("SPARSE_PROFILE_ROUTING_OUT", "") or "").strip()
        if raw_out == "":
            raw_out = "results/scp_ep_routing_profile"
        out_path = Path(raw_out).expanduser()
        if out_path.suffix == ".npz":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path = out_path.with_name(f"{out_path.stem}.rank{int(rank)}{out_path.suffix}")
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            npz_path = out_path / f"profile.rank{int(rank)}.npz"
        metadata = dict(cls.routing_profile_metadata)
        metadata["rank"] = int(rank)
        metadata["output_semantics"] = "A_prefill/A_decode[l, source_rank, expert_id] counts router top-k token-expert assignments observed on this rank."
        metadata["trace_event_count"] = int(len(cls.routing_profile_trace_count))
        metadata["trace_prefill_count_sum"] = int(
            sum(
                int(c)
                for st, c in zip(cls.routing_profile_trace_stage, cls.routing_profile_trace_count)
                if int(st) == 0
            )
        )
        metadata["trace_decode_count_sum"] = int(
            sum(
                int(c)
                for st, c in zip(cls.routing_profile_trace_stage, cls.routing_profile_trace_count)
                if int(st) == 1
            )
        )
        np.savez_compressed(
            npz_path,
            A_prefill=cls.routing_profile_prefill,
            A_decode=cls.routing_profile_decode,
            prefill_token_rows_by_layer=cls.routing_profile_prefill_rows_by_layer,
            decode_token_rows_by_layer=cls.routing_profile_decode_rows_by_layer,
            trace_stage=np.asarray(cls.routing_profile_trace_stage, dtype=np.int8),
            trace_call_id=np.asarray(cls.routing_profile_trace_call_id, dtype=np.int64),
            trace_step=np.asarray(cls.routing_profile_trace_step, dtype=np.int64),
            trace_layer=np.asarray(cls.routing_profile_trace_layer, dtype=np.int16),
            trace_source_rank=np.asarray(cls.routing_profile_trace_source_rank, dtype=np.int16),
            trace_expert=np.asarray(cls.routing_profile_trace_expert, dtype=np.int16),
            trace_count=np.asarray(cls.routing_profile_trace_count, dtype=np.int64),
            trace_token_rows=np.asarray(cls.routing_profile_trace_token_rows, dtype=np.int64),
            trace_shard_id=np.asarray(cls.routing_profile_trace_shard_id, dtype=np.int64),
            trace_request_id=np.asarray(cls.routing_profile_trace_request_id, dtype=np.int64),
            trace_global_order_id=np.asarray(cls.routing_profile_trace_global_order_id, dtype=np.int64),
            trace_source_node=np.asarray(cls.routing_profile_trace_source_node, dtype=np.int16),
            trace_task_id=np.asarray(cls.routing_profile_trace_task_id, dtype=np.int16),
            trace_task_family=np.asarray(cls.routing_profile_trace_task_family, dtype=np.int16),
            trace_request_global_id=np.asarray(cls.routing_profile_trace_request_global_id, dtype=np.int64),
            metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
        )
        meta_path = npz_path.with_suffix(".metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return str(npz_path)

    @classmethod
    def register_layer_block(cls, layer_idx: int, block: "DistributedSparseMoeBlock") -> None:
        cls.layer_blocks[int(layer_idx)] = block

    @classmethod
    def register_layer_attn(cls, layer_idx: int, attn_module: nn.Module) -> None:
        cls.layer_attn_modules[int(layer_idx)] = attn_module

    @classmethod
    def reset_nodepatch_stats(cls) -> None:
        cls.nodepatch_stats = {
            "enabled_calls": 0.0,
            "tokens_seen": 0.0,
            "cross_node_tokens": 0.0,
            "repaired_tokens": 0.0,
            "replacement_count": 0.0,
            "event_proxy_before": 0.0,
            "event_proxy_after": 0.0,
            "cross_node_items_before": 0.0,
            "cross_node_items_after": 0.0,
            "unique_nodes_before_sum": 0.0,
            "unique_nodes_after_sum": 0.0,
            "runtime_packet_events_before": 0.0,
            "runtime_packet_events_after": 0.0,
            "runtime_packet_events_eliminated": 0.0,
            "event_aware_skipped_repairs": 0.0,
            "cache_aware_skipped_repairs": 0.0,
            "cache_aware_filtered_candidates": 0.0,
            "packet_emptying_candidate_packets": 0.0,
            "packet_emptying_feasible_packets": 0.0,
            "packet_emptying_rejected_packets": 0.0,
            "utility_loss_logit_sum": 0.0,
            "utility_loss_prob_sum": 0.0,
            "max_margin_logit_sum": 0.0,
            "max_margin_prob_sum": 0.0,
            "max_margin_logit_max": 0.0,
            "max_margin_prob_max": 0.0,
            "replacement_pair_hist": {},
            "removed_expert_hist": {},
            "added_expert_hist": {},
            "removed_position_hist": {},
            "replacement_rank_gap_hist": {},
            "layer_repaired_token_hist": {},
            "route_cost_before_sum": 0.0,
            "route_cost_after_sum": 0.0,
            "route_cost_gain_sum": 0.0,
            "old_prob_sum": 0.0,
            "new_prob_sum": 0.0,
            "prob_delta_sum": 0.0,
            "rank_gap_sum": 0.0,
            "rank_gap_max": 0.0,
            "candidate_not_prepared_reject_count": 0.0,
            "candidate_no_comm_gain_reject_count": 0.0,
            "candidate_scope_reject_count": 0.0,
            "candidate_duplicate_reject_count": 0.0,
            "candidate_rank_gap_reject_count": 0.0,
            "candidate_cache_reject_count": 0.0,
            "candidate_strict_resident_reject_count": 0.0,
            "no_replace_position_count": 0.0,
        }

    @classmethod
    def record_nodepatch(
        cls,
        *,
        tokens: int,
        cross_node_tokens: int,
        repaired_tokens: int,
        replacement_count: int,
        event_before: int,
        event_after: int,
        cross_items_before: int,
        cross_items_after: int,
        unique_nodes_before_sum: float,
        unique_nodes_after_sum: float,
        utility_loss_logit: float,
        utility_loss_prob: float,
        max_margin_logit_sum: float,
        max_margin_prob_sum: float,
        max_margin_logit_max: float,
        max_margin_prob_max: float,
        runtime_packet_events_before: float = 0.0,
        runtime_packet_events_after: float = 0.0,
        runtime_packet_events_eliminated: float = 0.0,
        event_aware_skipped_repairs: float = 0.0,
        cache_aware_skipped_repairs: float = 0.0,
        cache_aware_filtered_candidates: float = 0.0,
        packet_emptying_candidate_packets: float = 0.0,
        packet_emptying_feasible_packets: float = 0.0,
        packet_emptying_rejected_packets: float = 0.0,
        replacement_pair_hist: Optional[Dict[str, int]] = None,
        removed_expert_hist: Optional[Dict[str, int]] = None,
        added_expert_hist: Optional[Dict[str, int]] = None,
        removed_position_hist: Optional[Dict[str, int]] = None,
        replacement_rank_gap_hist: Optional[Dict[str, int]] = None,
        layer_repaired_token_hist: Optional[Dict[str, int]] = None,
        route_cost_before: float = 0.0,
        route_cost_after: float = 0.0,
        route_cost_gain: float = 0.0,
        old_prob_sum: float = 0.0,
        new_prob_sum: float = 0.0,
        prob_delta_sum: float = 0.0,
        rank_gap_sum: float = 0.0,
        rank_gap_max: float = 0.0,
        candidate_not_prepared_reject_count: float = 0.0,
        candidate_no_comm_gain_reject_count: float = 0.0,
        candidate_scope_reject_count: float = 0.0,
        candidate_duplicate_reject_count: float = 0.0,
        candidate_rank_gap_reject_count: float = 0.0,
        candidate_cache_reject_count: float = 0.0,
        candidate_strict_resident_reject_count: float = 0.0,
        no_replace_position_count: float = 0.0,
    ) -> None:
        if not cls.nodepatch_stats:
            cls.reset_nodepatch_stats()
        s = cls.nodepatch_stats
        s["enabled_calls"] += 1.0
        s["tokens_seen"] += float(tokens)
        s["cross_node_tokens"] += float(cross_node_tokens)
        s["repaired_tokens"] += float(repaired_tokens)
        s["replacement_count"] += float(replacement_count)
        s["event_proxy_before"] += float(event_before)
        s["event_proxy_after"] += float(event_after)
        s["cross_node_items_before"] += float(cross_items_before)
        s["cross_node_items_after"] += float(cross_items_after)
        s["unique_nodes_before_sum"] += float(unique_nodes_before_sum)
        s["unique_nodes_after_sum"] += float(unique_nodes_after_sum)
        s["runtime_packet_events_before"] += float(runtime_packet_events_before)
        s["runtime_packet_events_after"] += float(runtime_packet_events_after)
        s["runtime_packet_events_eliminated"] += float(runtime_packet_events_eliminated)
        s["event_aware_skipped_repairs"] += float(event_aware_skipped_repairs)
        s["cache_aware_skipped_repairs"] += float(cache_aware_skipped_repairs)
        s["cache_aware_filtered_candidates"] += float(cache_aware_filtered_candidates)
        s["packet_emptying_candidate_packets"] += float(packet_emptying_candidate_packets)
        s["packet_emptying_feasible_packets"] += float(packet_emptying_feasible_packets)
        s["packet_emptying_rejected_packets"] += float(packet_emptying_rejected_packets)
        s["utility_loss_logit_sum"] += float(utility_loss_logit)
        s["utility_loss_prob_sum"] += float(utility_loss_prob)
        s["max_margin_logit_sum"] += float(max_margin_logit_sum)
        s["max_margin_prob_sum"] += float(max_margin_prob_sum)
        s["max_margin_logit_max"] = max(float(s.get("max_margin_logit_max", 0.0)), float(max_margin_logit_max))
        s["max_margin_prob_max"] = max(float(s.get("max_margin_prob_max", 0.0)), float(max_margin_prob_max))
        s["route_cost_before_sum"] += float(route_cost_before)
        s["route_cost_after_sum"] += float(route_cost_after)
        s["route_cost_gain_sum"] += float(route_cost_gain)
        s["old_prob_sum"] += float(old_prob_sum)
        s["new_prob_sum"] += float(new_prob_sum)
        s["prob_delta_sum"] += float(prob_delta_sum)
        s["rank_gap_sum"] += float(rank_gap_sum)
        s["rank_gap_max"] = max(float(s.get("rank_gap_max", 0.0)), float(rank_gap_max))
        s["candidate_not_prepared_reject_count"] += float(candidate_not_prepared_reject_count)
        s["candidate_no_comm_gain_reject_count"] += float(candidate_no_comm_gain_reject_count)
        s["candidate_scope_reject_count"] += float(candidate_scope_reject_count)
        s["candidate_duplicate_reject_count"] += float(candidate_duplicate_reject_count)
        s["candidate_rank_gap_reject_count"] += float(candidate_rank_gap_reject_count)
        s["candidate_cache_reject_count"] += float(candidate_cache_reject_count)
        s["candidate_strict_resident_reject_count"] += float(candidate_strict_resident_reject_count)
        s["no_replace_position_count"] += float(no_replace_position_count)
        for hist_key, incoming in [
            ("replacement_pair_hist", replacement_pair_hist),
            ("removed_expert_hist", removed_expert_hist),
            ("added_expert_hist", added_expert_hist),
            ("removed_position_hist", removed_position_hist),
            ("replacement_rank_gap_hist", replacement_rank_gap_hist),
            ("layer_repaired_token_hist", layer_repaired_token_hist),
        ]:
            if not incoming:
                continue
            hist = s.setdefault(hist_key, {})
            if not isinstance(hist, dict):
                hist = {}
                s[hist_key] = hist
            for k, v in incoming.items():
                hist[str(k)] = int(hist.get(str(k), 0)) + int(v)

    @classmethod
    def nodepatch_summary(cls) -> Dict[str, float]:
        if not cls.nodepatch_stats:
            cls.reset_nodepatch_stats()
        s = dict(cls.nodepatch_stats)
        repaired = max(float(s.get("repaired_tokens", 0.0)), 1.0)
        cross = max(float(s.get("cross_node_tokens", 0.0)), 1.0)
        tokens = max(float(s.get("tokens_seen", 0.0)), 1.0)
        before_events = max(float(s.get("event_proxy_before", 0.0)), 1.0)
        before_items = max(float(s.get("cross_node_items_before", 0.0)), 1.0)
        s["cross_node_token_fraction"] = float(s.get("cross_node_tokens", 0.0)) / tokens
        s["repairable_token_fraction"] = float(s.get("repaired_tokens", 0.0)) / tokens
        s["repairable_cross_node_fraction"] = float(s.get("repaired_tokens", 0.0)) / cross
        s["transfer_event_reduction_pct"] = (
            (float(s.get("event_proxy_before", 0.0)) - float(s.get("event_proxy_after", 0.0)))
            / before_events
            * 100.0
        )
        s["cross_node_item_reduction_pct"] = (
            (float(s.get("cross_node_items_before", 0.0)) - float(s.get("cross_node_items_after", 0.0)))
            / before_items
            * 100.0
        )
        # Backward-compatible aliases with explicit semantics. The historical
        # transfer_event_reduction_pct/event_proxy fields are token-level
        # unique-node locality proxies, not measured runtime packet clearing.
        s["token_locality_proxy_baseline"] = float(s.get("event_proxy_before", 0.0))
        s["token_locality_proxy_repaired"] = float(s.get("event_proxy_after", 0.0))
        s["token_locality_proxy_reduction_pct"] = float(s.get("transfer_event_reduction_pct", 0.0))
        s["cross_node_item_baseline"] = float(s.get("cross_node_items_before", 0.0))
        s["cross_node_item_repaired"] = float(s.get("cross_node_items_after", 0.0))
        runtime_before = max(float(s.get("runtime_packet_events_before", 0.0)), 1.0)
        s["runtime_cleared_packet_count"] = float(s.get("runtime_packet_events_eliminated", 0.0))
        s["runtime_cleared_packet_ratio"] = float(s.get("runtime_packet_events_eliminated", 0.0)) / runtime_before
        s["runtime_cleared_packet_pct"] = float(s.get("runtime_cleared_packet_ratio", 0.0)) * 100.0
        s["avg_utility_loss_logit"] = float(s.get("utility_loss_logit_sum", 0.0)) / repaired
        s["avg_utility_loss_prob"] = float(s.get("utility_loss_prob_sum", 0.0)) / repaired
        s["avg_max_margin_logit"] = float(s.get("max_margin_logit_sum", 0.0)) / repaired
        s["avg_max_margin_prob"] = float(s.get("max_margin_prob_sum", 0.0)) / repaired
        s["avg_unique_nodes_before"] = float(s.get("unique_nodes_before_sum", 0.0)) / tokens
        s["avg_unique_nodes_after"] = float(s.get("unique_nodes_after_sum", 0.0)) / tokens
        replacements = max(float(s.get("replacement_count", 0.0)), 1.0)
        route_before = max(float(s.get("route_cost_before_sum", 0.0)), 1.0)
        s["route_cost_reduction_pct"] = (
            (float(s.get("route_cost_before_sum", 0.0)) - float(s.get("route_cost_after_sum", 0.0)))
            / route_before
            * 100.0
        )
        s["avg_route_cost_gain_per_replacement"] = float(s.get("route_cost_gain_sum", 0.0)) / replacements
        s["avg_old_prob"] = float(s.get("old_prob_sum", 0.0)) / replacements
        s["avg_new_prob"] = float(s.get("new_prob_sum", 0.0)) / replacements
        s["avg_prob_delta"] = float(s.get("prob_delta_sum", 0.0)) / replacements
        s["avg_replacement_rank_gap"] = float(s.get("rank_gap_sum", 0.0)) / replacements
        s["candidate_total_reject_count"] = (
            float(s.get("candidate_not_prepared_reject_count", 0.0))
            + float(s.get("candidate_no_comm_gain_reject_count", 0.0))
            + float(s.get("candidate_scope_reject_count", 0.0))
            + float(s.get("candidate_duplicate_reject_count", 0.0))
            + float(s.get("candidate_rank_gap_reject_count", 0.0))
            + float(s.get("candidate_cache_reject_count", 0.0))
            + float(s.get("candidate_strict_resident_reject_count", 0.0))
        )
        for hist_key, top_key in [
            ("replacement_pair_hist", "top_replacement_pairs"),
            ("removed_expert_hist", "top_removed_experts"),
            ("added_expert_hist", "top_added_experts"),
            ("removed_position_hist", "top_removed_positions"),
            ("replacement_rank_gap_hist", "top_replacement_rank_gaps"),
            ("layer_repaired_token_hist", "top_repaired_layers"),
        ]:
            hist = s.get(hist_key, {})
            if isinstance(hist, dict):
                s[top_key] = sorted(
                    [{"key": str(k), "count": int(v)} for k, v in hist.items()],
                    key=lambda x: (-int(x["count"]), str(x["key"])),
                )[:32]
        return s

    @classmethod
    def set_request_owners(cls, owner_ranks: torch.Tensor) -> None:
        owner_ranks = owner_ranks.detach().to(dtype=torch.long).contiguous()
        cls.request_owner_ranks = owner_ranks

        for _, attn in cls.layer_attn_modules.items():
            setattr(attn, "_skvs_owner_ranks", owner_ranks)
            setattr(attn, "_skvs_owner_rank", int(owner_ranks[0].item()) if owner_ranks.numel() > 0 else -1)

    @classmethod
    def set_request_owner(cls, owner_rank: int) -> None:
        cls.set_request_owners(torch.tensor([owner_rank], dtype=torch.long))

    @classmethod
    def get_request_owners(cls) -> Optional[torch.Tensor]:
        return cls.request_owner_ranks

    @classmethod
    def get_request_owner(cls) -> int:
        if cls.request_owner_ranks is None or cls.request_owner_ranks.numel() == 0:
            return -1
        return int(cls.request_owner_ranks[0].item())

    @classmethod
    def clear_request_owners(cls) -> None:
        cls.request_owner_ranks = None
        for _, attn in cls.layer_attn_modules.items():
            setattr(attn, "_skvs_owner_ranks", None)
            setattr(attn, "_skvs_owner_rank", -1)

    @classmethod
    def clear_request_owner(cls) -> None:
        cls.clear_request_owners()

    @classmethod
    def finalize_prefetch_all(cls) -> None:
        for _, blk in cls.layer_blocks.items():
            try:
                blk._finalize_prev_group_prefetch()
            except Exception:
                pass

    @classmethod
    def reset_runtime_state_for_new_request(
        cls,
        *,
        clear_expert_cache: bool = False,
        clear_cache_metadata: bool = True,
        demote_hot: bool = False,
    ) -> None:
        """Reset request-scoped runtime state without rebuilding the model.

        This is intentionally stronger than `reset()`: it clears stale RS/prefetch
        admission metadata inside each MoE block.  It intentionally keeps resident
        expert weights unless `clear_expert_cache=True`, because clearing weights
        changes the benchmark from warm serving to cold-start serving.
        """
        cls.routing_cache.clear()
        cls.clear_request_owners()
        cls.clear_group_plan()
        cls.clear_local_batch()
        cls.suppress_logical_weaknet_once = False
        cls.logical_ngd_ep_reuse_once = False
        cls.current_phase = "none"
        cls.current_moe_layer_idx = None
        cls.prefill_scp_transient_demoted_for_decode = False
        for _, blk in list(cls.layer_blocks.items()):
            reset_fn = getattr(blk, "reset_runtime_state_for_new_request", None)
            if callable(reset_fn):
                try:
                    reset_fn(
                        clear_expert_cache=bool(clear_expert_cache),
                        clear_cache_metadata=bool(clear_cache_metadata),
                        demote_hot=bool(demote_hot),
                    )
                except Exception:
                    pass

    @classmethod
    def reset(cls) -> None:
        cls.routing_cache.clear()
        cls.clear_request_owners()
        cls.clear_group_plan()
        cls.clear_local_batch()

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
    def set_local_batch(cls, sample_ids: torch.Tensor, full_batch_size: int) -> None:
        cls.local_sample_ids = sample_ids.detach().to(dtype=torch.long).contiguous()
        cls.local_full_batch_size = int(full_batch_size)

    @classmethod
    def get_local_sample_ids(cls, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if cls.local_sample_ids is None:
            return None
        out = cls.local_sample_ids
        if device is not None:
            out = out.to(device=device)
        return out

    @classmethod
    def clear_local_batch(cls) -> None:
        cls.local_sample_ids = None
        cls.local_full_batch_size = 0

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
    def set_group_owner_ranks(cls, owner_ranks: torch.Tensor) -> None:
        owner_ranks = owner_ranks.detach().to(dtype=torch.long).contiguous()
        cls.group_owner_ranks = owner_ranks
        cls.request_owner_ranks = owner_ranks  # 兼容 SKVS attention 读取

        for _, attn in cls.layer_attn_modules.items():
            setattr(attn, "_skvs_owner_ranks", owner_ranks)
            setattr(attn, "_skvs_owner_rank", int(owner_ranks[0].item()) if owner_ranks.numel() > 0 else -1)

    @classmethod
    def get_group_owner_ranks(cls) -> Optional[torch.Tensor]:
        return cls.group_owner_ranks

    @classmethod
    def clear_group_plan(cls) -> None:
        cls.group_owner_ranks = None
        cls.request_owner_ranks = None
        cls.group_pred_topi.clear()
        cls.group_pred_topv.clear()
        cls.group_pred_value.clear()
        cls.transition_source_experts.clear()
        for _, attn in cls.layer_attn_modules.items():
            setattr(attn, "_skvs_owner_ranks", None)
            setattr(attn, "_skvs_owner_rank", -1)


    @classmethod
    def set_oracle_runtime_mode(cls, mode: str) -> None:
        cls.oracle_mode = str(mode)
        for _, blk in cls.layer_blocks.items():
            try:
                blk.oracle_mode = str(mode)
            except Exception:
                pass

    @classmethod
    def set_oracle_mode(cls, mode: str, trace_dir: str) -> None:
        cls.oracle_mode = str(mode)
        cls.oracle_trace_dir = str(trace_dir or "")
        cls.oracle_call_id = -1
        cls.runtime_call_id = -1
        cls.current_phase = "none"
        cls.oracle_active_layer_idx = None
        cls.oracle_trace = {}
        for _, blk in cls.layer_blocks.items():
            try:
                blk.oracle_mode = str(mode)
            except Exception:
                pass

    @classmethod
    def begin_oracle_call(cls, call_id: int) -> None:
        cls.oracle_call_id = int(call_id)
        cls.oracle_active_layer_idx = None

    @classmethod
    def begin_runtime_call(cls, call_id: int) -> None:
        cls.runtime_call_id = int(call_id)
        if int(call_id) >= 0:
            cls.transition_source_experts.clear()








    @classmethod
    def record_prefill_experts(cls, layer_idx: int, topk_indices: torch.Tensor) -> None:
        if str(cls.current_phase) != "prefill":
            return
        if topk_indices is None or topk_indices.numel() == 0:
            return
        counts = cls.prefill_expert_counts.setdefault(int(layer_idx), {})
        try:
            flat = topk_indices.detach().reshape(-1).to("cpu", dtype=torch.long).tolist()
        except Exception:
            return
        for eid in flat:
            eid_i = int(eid)
            counts[eid_i] = int(counts.get(eid_i, 0)) + 1

    @classmethod
    def clear_prefill_expert_counts(cls) -> None:
        cls.prefill_expert_counts.clear()

    @classmethod
    def set_transition_source_experts(cls, target_layer_idx: int, expert_ids: List[int]) -> None:
        cls.transition_source_experts[int(target_layer_idx)] = set(int(x) for x in expert_ids)

    @classmethod
    def pop_transition_source_experts(cls, target_layer_idx: int) -> set[int]:
        return cls.transition_source_experts.pop(int(target_layer_idx), set())

    @classmethod
    def end_oracle_call(cls) -> None:
        cls.oracle_call_id = -1
        cls.runtime_call_id = -1
        cls.oracle_active_layer_idx = None
        cls.current_phase = "none"

    @classmethod
    def record_oracle_local_experts(cls, layer_idx: int, expert_ids: List[int]) -> None:
        if cls.oracle_mode != "record":
            return
        if cls.oracle_call_id < 0:
            return
        call_id = int(cls.oracle_call_id)
        l = int(layer_idx)
        if call_id not in cls.oracle_trace:
            cls.oracle_trace[call_id] = {}
        cls.oracle_trace[call_id][l] = [int(x) for x in expert_ids]

    @classmethod
    def save_oracle_trace(cls, rank: int) -> None:
        if cls.oracle_mode != "record" or cls.oracle_trace_dir == "":
            return
        os.makedirs(cls.oracle_trace_dir, exist_ok=True)
        path = os.path.join(cls.oracle_trace_dir, f"oracle_trace_rank{rank}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.oracle_trace, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_oracle_trace(cls, rank: int) -> None:
        if cls.oracle_mode != "replay" or cls.oracle_trace_dir == "":
            return
        path = os.path.join(cls.oracle_trace_dir, f"oracle_trace_rank{rank}.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cls.oracle_trace = {
            int(call_id): {int(l): [int(x) for x in eids] for l, eids in layer_map.items()}
            for call_id, layer_map in raw.items()
        }

    @classmethod
    def get_oracle_layer_experts(cls, call_id: int, layer_idx: int) -> List[int]:
        call_id = int(call_id)
        layer_idx = int(layer_idx)
        layer_map = cls.oracle_trace.get(call_id, {})
        return list(layer_map.get(layer_idx, []))

    @classmethod
    def clear_oracle_runtime_active(cls) -> None:
        active = cls.oracle_active_layer_idx
        if active is None:
            return
        blk = cls.layer_blocks.get(int(active), None)
        if blk is not None:
            try:
                blk.clear_oracle_runtime_state()
            except Exception:
                pass
        cls.oracle_active_layer_idx = None

    @classmethod
    def clear_oracle_runtime_all(cls, except_layer_idx: Optional[int] = None) -> None:
        # Fast path: in replay we keep at most one active layer resident at a time.
        if except_layer_idx is None:
            cls.clear_oracle_runtime_active()
            return

        keep = int(except_layer_idx)
        active = cls.oracle_active_layer_idx
        if active is not None and int(active) != keep:
            blk = cls.layer_blocks.get(int(active), None)
            if blk is not None:
                try:
                    blk.clear_oracle_runtime_state()
                except Exception:
                    pass
            cls.oracle_active_layer_idx = None

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

class _RSResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class RoutingScoutResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[_RSResidualBlock(hidden_dim, dropout) for _ in range(max(0, int(num_hidden_layers)))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        h = self.input(x)
        h = self.blocks(h)
        return torch.softmax(self.head(self.out_norm(h)), dim=-1)

class CacheValueScoutMLP(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        return self.net(x)


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise RuntimeError(f"Cannot extract state_dict from payload type={type(payload)}")


def _build_rs_module_from_payload(
    payload: Any,
    *,
    fallback_input_dim: int,
    fallback_num_experts: int,
) -> Tuple[nn.Module, str]:
    if isinstance(payload, dict) and payload.get("task", "") == "cache_value_rs":
        input_dim = int(payload.get("input_dim", fallback_input_dim))
        num_experts = int(payload.get("num_experts", fallback_num_experts))
        hidden_dim = int(payload.get("hidden_dim", 1024))
        state_dict = _extract_state_dict(payload)
        mod = CacheValueScoutMLP(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            dropout=0.1,
        )
        mod.load_state_dict(state_dict, strict=True)
        return mod, "cache_value_rs"

    # legacy / router-probability RS
    if isinstance(payload, dict) and payload.get("model_type", "") == "independent_light_logits_rs":
        input_dim = int(payload.get("input_dim", fallback_input_dim))
        num_experts = int(payload.get("num_experts", fallback_num_experts))
        hidden_dim = int(payload.get("hidden_dim", 1024))
        state_dict = _extract_state_dict(payload)
        mod = RoutingScoutMLP(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
        )
        mod.load_state_dict(state_dict, strict=True)
        setattr(mod, "rs_feature_mode", str(payload.get("feature_mode", "hidden")))
        return mod, "router_prob"

    if isinstance(payload, dict) and payload.get("model_type", "") == "next_layer_residual_logits_rs":
        input_dim = int(payload.get("input_dim", fallback_input_dim))
        num_experts = int(payload.get("num_experts", fallback_num_experts))
        hidden_dim = int(payload.get("hidden_dim", 1024))
        num_hidden_layers = int(payload.get("num_hidden_layers", 3))
        state_dict = _extract_state_dict(payload)
        mod = RoutingScoutResidualMLP(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=0.05,
        )
        mod.load_state_dict(state_dict, strict=True)
        setattr(mod, "rs_feature_mode", str(payload.get("feature_mode", "hidden")))
        return mod, "router_prob"

    # plain state_dict fallback
    state_dict = _extract_state_dict(payload)
    mod = RoutingScoutMLP(
        input_dim=fallback_input_dim,
        num_experts=fallback_num_experts,
        hidden_dim=1024,
    )
    mod.load_state_dict(state_dict, strict=True)
    return mod, "router_prob"


def _is_next_layer_rs_layout(rs_path: Path) -> bool:
    for path in rs_path.glob("rs_l*_inL*.pth"):
        match = re.fullmatch(r"rs_l(\d+)_inL(-?\d+)\.pth", path.name)
        if match and int(match.group(1)) == int(match.group(2)) + 1:
            return True
    for path in rs_path.glob("cv_l*_inL*.pth"):
        match = re.fullmatch(r"cv_l(\d+)_inL(-?\d+)\.pth", path.name)
        if match and int(match.group(1)) == int(match.group(2)) + 1:
            return True
    return False


def _next_layer_rs_target_layers(rs_path: Path) -> List[int]:
    targets = set()
    for pattern in ("rs_l*_inL*.pth", "cv_l*_inL*.pth"):
        for path in rs_path.glob(pattern):
            match = re.fullmatch(r"(?:rs|cv)_l(\d+)_inL(-?\d+)\.pth", path.name)
            if match and int(match.group(1)) == int(match.group(2)) + 1:
                targets.add(int(match.group(1)))
    return sorted(targets)


def _is_optional_next_layer_rs_miss(layer_idx: int, available_targets: Sequence[int]) -> bool:
    return bool(available_targets) and int(layer_idx) < int(available_targets[0])


def _load_runtime_rs_bank(
    *,
    rs_path: Path,
    num_layers: int,
    hidden_dim: int,
    num_experts: int,
    device: torch.device,
    require_complete: bool,
) -> Tuple[nn.ModuleList, str, str, List[int], List[str]]:
    bank_modules: List[nn.Module] = []
    missing: List[str] = []
    bank_task: Optional[str] = None
    source_layers: List[int] = []
    next_layer_layout = _is_next_layer_rs_layout(rs_path)
    next_layer_targets = _next_layer_rs_target_layers(rs_path) if next_layer_layout else []

    for l in range(int(num_layers)):
        if next_layer_layout:
            in_layer = int(l) - 1
            cv_ckpt = rs_path / f"cv_l{l}_inL{in_layer}.pth"
            legacy_ckpt = rs_path / f"rs_l{l}_inL{in_layer}.pth"
        else:
            in_layer = 0
            cv_ckpt = rs_path / f"cv_l{l}_inL{in_layer}.pth"
            legacy_ckpt = rs_path / f"rs_l{l}_inL{in_layer}.pth"

        ckpt_path: Optional[Path] = None
        if cv_ckpt.exists():
            ckpt_path = cv_ckpt
        elif legacy_ckpt.exists():
            ckpt_path = legacy_ckpt

        if ckpt_path is None:
            optional_next_layer_miss = next_layer_layout and _is_optional_next_layer_rs_miss(
                int(l), next_layer_targets
            )
            if not optional_next_layer_miss:
                missing.append(f"[L{l}] tried: {cv_ckpt.name}, {legacy_ckpt.name}")
            if require_complete:
                if not optional_next_layer_miss:
                    continue
            dummy = RoutingScoutMLP(
                input_dim=hidden_dim,
                num_experts=num_experts,
                hidden_dim=1024,
            ).to(device, dtype=torch.float32).eval()
            bank_modules.append(dummy)
            source_layers.append(int(in_layer))
            if bank_task is None:
                bank_task = "router_prob"
            continue

        payload = torch.load(str(ckpt_path), map_location="cpu")
        mod, task = _build_rs_module_from_payload(
            payload,
            fallback_input_dim=hidden_dim,
            fallback_num_experts=num_experts,
        )
        if bank_task is None:
            bank_task = task
        elif bank_task != task:
            raise RuntimeError(
                f"Mixed RS tasks in rs_dir are not allowed: layer {l} has task={task}, previous task={bank_task}"
            )

        bank_modules.append(mod.to(device, dtype=torch.float32).eval())
        source_layers.append(int(payload.get("source_layer", in_layer)) if isinstance(payload, dict) else int(in_layer))

    if require_complete and missing:
        raise RuntimeError("RS checkpoint set incomplete:\n" + "\n".join(missing[:32]))
    if len(bank_modules) != int(num_layers):
        raise RuntimeError(f"RS bank size mismatch: got {len(bank_modules)} modules, expect {num_layers}")

    return (
        nn.ModuleList(bank_modules).eval(),
        str(bank_task or "router_prob"),
        "next_layer" if next_layer_layout else "group",
        list(source_layers),
        missing,
    )

# ---------------- cache ----------------
@dataclass
class CacheEntry:
    payload: Any
    pinned: bool = False
    hot: bool = False
    prefill_scp_transient: bool = False


class ExpertWeightCache:
    """
    Plain LRU cache used by the baseline path.

    The important extension for planner-only prefetch is that LRU now supports
    *active replacement*: when the cache is already full, prefetch may still admit
    a future expert by evicting a low-value resident chosen by the runtime.

    Baseline correctness is preserved because ordinary demand accesses still go
    through the same passive LRU update path (put/get). Only planner-only prefetch
    uses the active replacement helpers below.
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._lru: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._last_evicted_was_prefill_scp_transient = False

    def get(self, expert_id: int):
        if self.capacity <= 0:
            return None
        entry = self._lru.get(expert_id)
        if entry is None:
            return None
        self._lru.move_to_end(expert_id, last=True)
        return entry.payload

    def peek(self, expert_id: int):
        if self.capacity <= 0:
            return None
        entry = self._lru.get(int(expert_id))
        return None if entry is None else entry.payload

    def has(self, expert_id: int) -> bool:
        return expert_id in self._lru

    def unpin_all(self) -> None:
        for k in list(self._lru.keys()):
            self._lru[k].pinned = False

    def clear(self) -> None:
        self._lru.clear()
        self._last_evicted_was_prefill_scp_transient = False

    def unpin(self, expert_id: int) -> None:
        if expert_id in self._lru:
            self._lru[expert_id].pinned = False

    def promote_hot(self, expert_id: int) -> None:
        if expert_id in self._lru:
            self._lru.move_to_end(expert_id, last=True)

    def resident_ids(self) -> List[int]:
        return list(self._lru.keys())

    def discard(self, expert_id: int) -> bool:
        expert_id = int(expert_id)
        if expert_id in self._lru:
            del self._lru[expert_id]
            return True
        return False

    def evictable_prefetch_ids(self, protect_ids: Optional[set[int]] = None) -> List[int]:
        protect = protect_ids or set()
        out: List[int] = []
        for k, v in self._lru.items():
            if v.pinned:
                continue
            if k in protect:
                continue
            out.append(int(k))
        return out

    def prefetch_free_slots(self) -> int:
        return max(0, int(self.capacity) - int(len(self._lru)))

    def put(
        self,
        expert_id: int,
        weights,
        pinned: bool = False,
        prefer_hot: bool = False,
        protect_ids: Optional[set[int]] = None,
        prefill_scp_transient: bool = False,
    ) -> Optional[int]:
        if self.capacity <= 0:
            return None

        if expert_id in self._lru:
            old = self._lru[expert_id]
            old.payload = weights
            old.pinned = old.pinned or bool(pinned)
            old.prefill_scp_transient = bool(old.prefill_scp_transient or prefill_scp_transient)
            self._lru.move_to_end(expert_id, last=True)
            return None

        self._lru[expert_id] = CacheEntry(payload=weights, pinned=bool(pinned), hot=False, prefill_scp_transient=bool(prefill_scp_transient))
        self._lru.move_to_end(expert_id, last=True)

        evicted: Optional[int] = None
        evicted_was_transient = False
        protect = protect_ids or set()
        while len(self._lru) > self.capacity:
            victim = None
            for k, v in self._lru.items():
                if not v.pinned and int(k) not in protect and getattr(v, "prefill_scp_transient", False):
                    victim = k
                    break
            if victim is None:
                for k, v in self._lru.items():
                    if not v.pinned and int(k) not in protect:
                        victim = k
                        break
            if victim is None:
                for k, v in self._lru.items():
                    if not v.pinned and getattr(v, "prefill_scp_transient", False):
                        victim = k
                        break
            if victim is None:
                for k, v in self._lru.items():
                    if not v.pinned:
                        victim = k
                        break
            if victim is None:
                victim = next(iter(self._lru.keys()))
            victim_entry = self._lru.get(victim)
            evicted = int(victim)
            evicted_was_transient = bool(getattr(victim_entry, "prefill_scp_transient", False))
            del self._lru[victim]
        self._last_evicted_was_prefill_scp_transient = bool(evicted_was_transient)
        return evicted

    def last_evicted_was_prefill_scp_transient(self) -> bool:
        return bool(getattr(self, "_last_evicted_was_prefill_scp_transient", False))

    def admit_prefetch(
        self,
        expert_id: int,
        weights,
        *,
        protect_ids: Optional[set[int]] = None,
        preferred_victim: Optional[int] = None,
    ) -> bool:
        if self.capacity <= 0:
            return False

        if expert_id in self._lru:
            old = self._lru[expert_id]
            old.payload = weights
            old.pinned = True
            self._lru.move_to_end(expert_id, last=True)
            return True

        if len(self._lru) < self.capacity:
            self._lru[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._lru.move_to_end(expert_id, last=True)
            return True

        protect = protect_ids or set()
        evictable = self.evictable_prefetch_ids(protect)
        if len(evictable) == 0:
            return False

        victim = None
        if preferred_victim is not None and int(preferred_victim) in evictable:
            victim = int(preferred_victim)
        else:
            victim = int(evictable[0])

        if victim in self._lru:
            del self._lru[victim]
        self._lru[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
        self._lru.move_to_end(expert_id, last=True)
        return True

    def is_prefill_scp_transient(self, expert_id: int) -> bool:
        entry = self._lru.get(int(expert_id))
        return bool(getattr(entry, "prefill_scp_transient", False)) if entry is not None else False

    def clear_prefill_scp_transient(self, expert_id: int) -> bool:
        entry = self._lru.get(int(expert_id))
        if entry is None or not getattr(entry, "prefill_scp_transient", False):
            return False
        entry.prefill_scp_transient = False
        return True

    def prefill_scp_transient_ids(self) -> List[int]:
        return [int(k) for k, v in self._lru.items() if getattr(v, "prefill_scp_transient", False)]

    def reset_request_metadata(self, *, unpin: bool = True, demote_hot: bool = False) -> None:
        # Plain LRU only has per-request pinning metadata.  Resident weights stay
        # in the cache unless clear() is requested by an explicit diagnostic.
        if unpin:
            self.unpin_all()

    def demote_prefill_scp_transients(self) -> int:
        # In a plain LRU cache, "demote" means move transient entries to the
        # oldest end so the next ordinary eviction sees them first.
        ids = self.prefill_scp_transient_ids()
        for eid in ids:
            if int(eid) in self._lru:
                self._lru.move_to_end(int(eid), last=False)
                self._lru[int(eid)].pinned = False
        return int(len(ids))

    def occupancy(self) -> Dict[str, int]:
        return {
            "capacity": int(self.capacity),
            "resident": int(len(self._lru)),
            "hot": 0,
            "transient_demand": int(len(self._lru)),
            "transient_prefetch": 0,
            "prefill_scp_transient": int(len(self.prefill_scp_transient_ids())),
        }


class EpipePrefillExtraCache:
    """Small EPIPE-only sidecar cache for prefill staged payloads.

    It is intentionally separate from the ordinary expert cache: inserts here do
    not evict decode/RS residents and hits do not update the main LRU state.
    """

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._lru: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._nbytes: Dict[int, int] = {}

    def has(self, expert_id: int) -> bool:
        return int(expert_id) in self._lru

    def get(self, expert_id: int):
        eid = int(expert_id)
        if self.capacity <= 0:
            return None
        entry = self._lru.get(eid)
        if entry is None:
            return None
        self._lru.move_to_end(eid, last=True)
        return entry.payload

    def peek(self, expert_id: int):
        eid = int(expert_id)
        if self.capacity <= 0:
            return None
        entry = self._lru.get(eid)
        return None if entry is None else entry.payload

    def put(self, expert_id: int, payload, *, nbytes: int = 0) -> Tuple[bool, bool, Optional[int]]:
        eid = int(expert_id)
        if self.capacity <= 0:
            return False, False, None
        updated = eid in self._lru
        evicted: Optional[int] = None
        if updated:
            self._lru[eid].payload = payload
            self._lru.move_to_end(eid, last=True)
            self._nbytes[eid] = max(0, int(nbytes))
            return True, True, None
        if len(self._lru) >= self.capacity:
            victim = int(next(iter(self._lru.keys())))
            del self._lru[victim]
            self._nbytes.pop(victim, None)
            evicted = victim
        self._lru[eid] = CacheEntry(payload=payload, pinned=False, hot=False, prefill_scp_transient=True)
        self._lru.move_to_end(eid, last=True)
        self._nbytes[eid] = max(0, int(nbytes))
        return True, False, evicted

    def clear(self) -> int:
        n = int(len(self._lru))
        self._lru.clear()
        self._nbytes.clear()
        return n

    def resident_ids(self) -> List[int]:
        return list(int(k) for k in self._lru.keys())

    def resident_count(self) -> int:
        return int(len(self._lru))

    def resident_nbytes(self) -> int:
        return int(sum(int(v) for v in self._nbytes.values()))


class TieredExpertWeightCache:
    """
    Tiered cache for runtime modes with planner-only prefetch.

    Hot entries remain protected from speculative replacement. Prefetch is allowed
    to use the reserved speculative area first, and if needed spill into / replace
    low-value transient-demand residents. This is the key runtime change that aligns
    the implementation with the intended semantics: prefetch may actively replace
    low-value residents *within the same layer/rank cache* instead of waiting for a
    demand miss to trigger passive LRU replacement.
    """
    def __init__(self, capacity: int, hot_capacity: int, prefetch_reserved: int = 1):
        self.capacity = int(capacity)
        self.hot_capacity = max(0, min(int(hot_capacity), int(capacity)))
        self.transient_capacity = max(0, self.capacity - self.hot_capacity)
        if self.transient_capacity <= 0:
            self.transient_prefetch_capacity = 0
        else:
            self.transient_prefetch_capacity = max(0, min(int(prefetch_reserved), self.transient_capacity))
        self.transient_demand_capacity = max(0, self.transient_capacity - self.transient_prefetch_capacity)

        self._hot: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._transient_demand: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._transient_prefetch: "OrderedDict[int, CacheEntry]" = OrderedDict()
        self._last_evicted_was_prefill_scp_transient = False

    def _evict_from_odict_if_needed(
        self,
        od: "OrderedDict[int, CacheEntry]",
        capacity: int,
        protect_ids: Optional[set[int]] = None,
    ) -> Optional[int]:
        evicted: Optional[int] = None
        evicted_was_transient = False
        protect = protect_ids or set()
        while len(od) > int(capacity):
            victim = None
            for k, v in od.items():
                if not v.pinned and int(k) not in protect and getattr(v, "prefill_scp_transient", False):
                    victim = k
                    break
            if victim is None:
                for k, v in od.items():
                    if not v.pinned and int(k) not in protect:
                        victim = k
                        break
            if victim is None:
                for k, v in od.items():
                    if not v.pinned and getattr(v, "prefill_scp_transient", False):
                        victim = k
                        break
            if victim is None:
                for k, v in od.items():
                    if not v.pinned:
                        victim = k
                        break
            if victim is None:
                victim = next(iter(od.keys()))
            victim_entry = od.get(victim)
            evicted = int(victim)
            evicted_was_transient = bool(getattr(victim_entry, "prefill_scp_transient", False))
            del od[victim]
        self._last_evicted_was_prefill_scp_transient = bool(evicted_was_transient)
        return evicted

    def _evict_demand_if_needed(self, protect_ids: Optional[set[int]] = None) -> Optional[int]:
        return self._evict_from_odict_if_needed(self._transient_demand, self.transient_demand_capacity, protect_ids)

    def _evict_prefetch_if_needed(self, protect_ids: Optional[set[int]] = None) -> Optional[int]:
        return self._evict_from_odict_if_needed(self._transient_prefetch, self.transient_prefetch_capacity, protect_ids)

    def _insert_hot(self, expert_id: int, weights, protect_ids: Optional[set[int]] = None) -> Optional[int]:
        if self.hot_capacity <= 0:
            return self.put(expert_id, weights, pinned=False, prefer_hot=False, protect_ids=protect_ids)
        self._transient_demand.pop(expert_id, None)
        self._transient_prefetch.pop(expert_id, None)
        self._hot[expert_id] = CacheEntry(payload=weights, pinned=False, hot=True)
        self._hot.move_to_end(expert_id, last=True)
        evicted: Optional[int] = None
        protect = protect_ids or set()
        while len(self._hot) > self.hot_capacity:
            demote_id = None
            for k, v in self._hot.items():
                if not v.pinned and int(k) not in protect:
                    demote_id = k
                    break
            if demote_id is None:
                for k, v in self._hot.items():
                    if not v.pinned:
                        demote_id = k
                        break
            if demote_id is None:
                demote_id = next(iter(self._hot.keys()))
            demote_entry = self._hot.pop(demote_id)
            self._transient_demand[demote_id] = CacheEntry(payload=demote_entry.payload, pinned=False, hot=False, prefill_scp_transient=getattr(demote_entry, "prefill_scp_transient", False))
            self._transient_demand.move_to_end(demote_id, last=True)
            evicted = self._evict_demand_if_needed(protect_ids)
        return evicted

    def get(self, expert_id: int):
        entry = self._hot.get(expert_id)
        if entry is not None:
            self._hot.move_to_end(expert_id, last=True)
            return entry.payload
        entry = self._transient_demand.get(expert_id)
        if entry is not None:
            self._transient_demand.move_to_end(expert_id, last=True)
            return entry.payload
        entry = self._transient_prefetch.get(expert_id)
        if entry is not None:
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return entry.payload
        return None

    def peek(self, expert_id: int):
        eid = int(expert_id)
        entry = self._hot.get(eid)
        if entry is not None:
            return entry.payload
        entry = self._transient_demand.get(eid)
        if entry is not None:
            return entry.payload
        entry = self._transient_prefetch.get(eid)
        if entry is not None:
            return entry.payload
        return None

    def has(self, expert_id: int) -> bool:
        return (
            expert_id in self._hot
            or expert_id in self._transient_demand
            or expert_id in self._transient_prefetch
        )

    def resident_ids(self) -> List[int]:
        return list(self._hot.keys()) + list(self._transient_demand.keys()) + list(self._transient_prefetch.keys())

    def discard(self, expert_id: int) -> bool:
        expert_id = int(expert_id)
        removed = False
        if expert_id in self._hot:
            del self._hot[expert_id]
            removed = True
        if expert_id in self._transient_demand:
            del self._transient_demand[expert_id]
            removed = True
        if expert_id in self._transient_prefetch:
            del self._transient_prefetch[expert_id]
            removed = True
        return bool(removed)

    def unpin_all(self) -> None:
        for k in list(self._transient_prefetch.keys()):
            self._transient_prefetch[k].pinned = False
        for k in list(self._transient_demand.keys()):
            self._transient_demand[k].pinned = False

    def clear(self) -> None:
        self._hot.clear()
        self._transient_demand.clear()
        self._transient_prefetch.clear()
        self._last_evicted_was_prefill_scp_transient = False

    def unpin(self, expert_id: int) -> None:
        if expert_id in self._transient_prefetch:
            self._transient_prefetch[expert_id].pinned = False
        if expert_id in self._transient_demand:
            self._transient_demand[expert_id].pinned = False

    def promote_hot(self, expert_id: int) -> None:
        if expert_id in self._hot:
            self._hot.move_to_end(expert_id, last=True)
            return
        entry = self._transient_demand.get(expert_id)
        if entry is None:
            entry = self._transient_prefetch.get(expert_id)
        if entry is None:
            return
        self._insert_hot(expert_id, entry.payload)

    def evictable_prefetch_ids(self, protect_ids: Optional[set[int]] = None) -> List[int]:
        protect = protect_ids or set()
        out: List[int] = []
        for od in (self._transient_prefetch, self._transient_demand):
            for k, v in od.items():
                if v.pinned:
                    continue
                if k in protect:
                    continue
                out.append(int(k))
        return out

    def prefetch_free_slots(self) -> int:
        free_prefetch = max(0, int(self.transient_prefetch_capacity) - int(len(self._transient_prefetch)))
        # Speculative prefetch must not silently consume demand-transient capacity.
        # Demand slots can still be replaced by prefetch through the explicit
        # victim path when the planner's score beats the resident value.
        return int(free_prefetch)

    def put(
        self,
        expert_id: int,
        weights,
        pinned: bool = False,
        prefer_hot: bool = False,
        protect_ids: Optional[set[int]] = None,
        prefill_scp_transient: bool = False,
    ) -> Optional[int]:
        if self.capacity <= 0:
            return None
        if prefer_hot and self.hot_capacity > 0:
            return self._insert_hot(expert_id, weights, protect_ids=protect_ids)
        if expert_id in self._hot:
            self._hot[expert_id].payload = weights
            self._hot.move_to_end(expert_id, last=True)
            return None

        target_prefetch = bool(pinned)

        if expert_id in self._transient_demand:
            old = self._transient_demand[expert_id]
            old.payload = weights
            if target_prefetch:
                old.pinned = True
            old.prefill_scp_transient = bool(old.prefill_scp_transient or prefill_scp_transient)
            self._transient_demand.move_to_end(expert_id, last=True)
            return None

        if expert_id in self._transient_prefetch:
            old = self._transient_prefetch[expert_id]
            old.payload = weights
            old.pinned = old.pinned or bool(pinned)
            old.prefill_scp_transient = bool(old.prefill_scp_transient or prefill_scp_transient)
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return self._evict_prefetch_if_needed(protect_ids)

        if target_prefetch:
            self._transient_prefetch[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False, prefill_scp_transient=bool(prefill_scp_transient))
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return self._evict_prefetch_if_needed(protect_ids)
        else:
            self._transient_demand[expert_id] = CacheEntry(payload=weights, pinned=False, hot=False, prefill_scp_transient=bool(prefill_scp_transient))
            self._transient_demand.move_to_end(expert_id, last=True)
            return self._evict_demand_if_needed(protect_ids)

    def admit_prefetch(
        self,
        expert_id: int,
        weights,
        *,
        protect_ids: Optional[set[int]] = None,
        preferred_victim: Optional[int] = None,
    ) -> bool:
        if self.capacity <= 0:
            return False

        if expert_id in self._hot:
            self._hot.move_to_end(expert_id, last=True)
            return True
        if expert_id in self._transient_demand:
            old = self._transient_demand[expert_id]
            old.payload = weights
            old.pinned = True
            self._transient_demand.move_to_end(expert_id, last=True)
            return True
        if expert_id in self._transient_prefetch:
            old = self._transient_prefetch[expert_id]
            old.payload = weights
            old.pinned = True
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return True

        protect = protect_ids or set()
        preferred_victim = int(preferred_victim) if preferred_victim is not None else None

        # 1) Prefer the dedicated speculative area.
        if len(self._transient_prefetch) < self.transient_prefetch_capacity:
            self._transient_prefetch[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return True

        # 2) Replace an existing speculative resident if possible.
        evict_prefetch = [eid for eid in self._transient_prefetch.keys() if (eid not in protect and not self._transient_prefetch[eid].pinned)]
        if preferred_victim is not None and preferred_victim in evict_prefetch:
            del self._transient_prefetch[preferred_victim]
            self._transient_prefetch[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return True
        if len(evict_prefetch) > 0:
            del self._transient_prefetch[int(evict_prefetch[0])]
            self._transient_prefetch[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._transient_prefetch.move_to_end(expert_id, last=True)
            return True

        # 3) Spill into / replace transient demand; hot residents remain protected.
        if len(self._transient_demand) < self.transient_demand_capacity:
            self._transient_demand[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._transient_demand.move_to_end(expert_id, last=True)
            return True

        evict_demand = [eid for eid in self._transient_demand.keys() if (eid not in protect and not self._transient_demand[eid].pinned)]
        if preferred_victim is not None and preferred_victim in evict_demand:
            del self._transient_demand[preferred_victim]
            self._transient_demand[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._transient_demand.move_to_end(expert_id, last=True)
            return True
        if len(evict_demand) > 0:
            del self._transient_demand[int(evict_demand[0])]
            self._transient_demand[expert_id] = CacheEntry(payload=weights, pinned=True, hot=False)
            self._transient_demand.move_to_end(expert_id, last=True)
            return True

        return False

    def is_prefill_scp_transient(self, expert_id: int) -> bool:
        eid = int(expert_id)
        for od in (self._hot, self._transient_demand, self._transient_prefetch):
            entry = od.get(eid)
            if entry is not None:
                return bool(getattr(entry, "prefill_scp_transient", False))
        return False

    def last_evicted_was_prefill_scp_transient(self) -> bool:
        return bool(getattr(self, "_last_evicted_was_prefill_scp_transient", False))

    def clear_prefill_scp_transient(self, expert_id: int) -> bool:
        eid = int(expert_id)
        for od in (self._hot, self._transient_demand, self._transient_prefetch):
            entry = od.get(eid)
            if entry is not None and getattr(entry, "prefill_scp_transient", False):
                entry.prefill_scp_transient = False
                return True
        return False

    def prefill_scp_transient_ids(self) -> List[int]:
        out: List[int] = []
        for od in (self._hot, self._transient_demand, self._transient_prefetch):
            out.extend(int(k) for k, v in od.items() if getattr(v, "prefill_scp_transient", False))
        return out

    def reset_request_metadata(self, *, unpin: bool = True, demote_hot: bool = False) -> None:
        # Request-level metadata must not make the next independent request
        # inherit stale prefetch protection.  Optionally demote hot entries so a
        # previous request cannot lock the tiered cache forever.
        if unpin:
            self.unpin_all()
        if demote_hot and self._hot:
            for eid, entry in list(self._hot.items()):
                self._transient_demand[int(eid)] = CacheEntry(payload=entry.payload, pinned=False, hot=False)
                self._transient_demand.move_to_end(int(eid), last=True)
            self._hot.clear()
            self._evict_demand_if_needed(set())

    def demote_prefill_scp_transients(self) -> int:
        ids = self.prefill_scp_transient_ids()
        if not ids:
            return 0
        for eid in ids:
            eid_i = int(eid)
            if eid_i in self._hot:
                entry = self._hot.pop(eid_i)
                entry.hot = False
                entry.pinned = False
                self._transient_demand[eid_i] = entry
                self._transient_demand.move_to_end(eid_i, last=False)
            elif eid_i in self._transient_prefetch:
                entry = self._transient_prefetch.pop(eid_i)
                entry.hot = False
                entry.pinned = False
                self._transient_demand[eid_i] = entry
                self._transient_demand.move_to_end(eid_i, last=False)
            elif eid_i in self._transient_demand:
                self._transient_demand[eid_i].pinned = False
                self._transient_demand.move_to_end(eid_i, last=False)
        self._evict_demand_if_needed(set())
        return int(len(ids))

    def occupancy(self) -> Dict[str, int]:
        return {
            "capacity": int(self.capacity),
            "resident": int(len(self._hot) + len(self._transient_demand) + len(self._transient_prefetch)),
            "hot": int(len(self._hot)),
            "transient_demand": int(len(self._transient_demand)),
            "transient_prefetch": int(len(self._transient_prefetch)),
            "prefill_scp_transient": int(len(self.prefill_scp_transient_ids())),
        }


# ---------------- detect sparse moe block ----------------
def looks_like_sparse_moe_block(mlp: nn.Module) -> bool:
    if not all(hasattr(mlp, a) for a in ["gate", "experts"]):
        return False
    has_qwen_shared = all(hasattr(mlp, a) for a in ["shared_expert", "shared_expert_gate"])
    has_deepseek_shared = hasattr(mlp, "shared_experts")
    return bool(has_qwen_shared or has_deepseek_shared)


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


def _quota_repair_expert_assignment(assign: List[int], world: int) -> List[int]:
    """Deterministically repair a placement so every EP rank receives its quota."""
    world = max(int(world), 1)
    E = int(len(assign))
    if E == 0:
        return []
    base = E // world
    rem = E % world
    quotas = [base + (1 if r < rem else 0) for r in range(world)]
    repaired = [int(x) % world for x in assign]
    counts = [0 for _ in range(world)]
    for r in repaired:
        counts[int(r)] += 1
    if counts == quotas:
        return repaired

    under: List[int] = []
    for r in range(world):
        under.extend([r] * max(0, quotas[r] - counts[r]))
    if not under:
        return repaired

    under_pos = 0
    for r in range(world):
        overflow = max(0, counts[r] - quotas[r])
        if overflow <= 0:
            continue
        # Move high expert ids first to preserve the low-id part of the SCP map.
        victim_eids = [eid for eid, rr in enumerate(repaired) if rr == r]
        for eid in reversed(victim_eids):
            if overflow <= 0 or under_pos >= len(under):
                break
            new_r = int(under[under_pos])
            under_pos += 1
            repaired[eid] = new_r
            counts[r] -= 1
            counts[new_r] += 1
            overflow -= 1
    return repaired


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
        oracle_mode: str = "none",
        oracle_ignore_cache_capacity: bool = True,
        placement_mode: str = "scp",
        expert_source: str = "router",
        cache_policy: str = "lru",
        runtime_mode: str = "baseline",
        topology: Optional[DistTopology] = None,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.env = env
        self.device = env.device
        self.rank = env.rank
        self.world = env.world
        self.runtime_mode = _normalize_runtime_mode(str(runtime_mode))
        self.topology = topology
        self.comm_group = None if topology is None else topology.ep_group
        self.comm_world = int(self.world if topology is None else topology.ep_size)
        self.comm_rank = int(self.rank if topology is None else topology.ep_rank)
        self.comm_global_ranks = tuple(range(self.world)) if topology is None else tuple(topology.ep_global_ranks)
        self._comm_rank_node_ids_cache_key: Optional[Tuple[Any, ...]] = None
        self._comm_rank_node_ids_cache: Optional[List[int]] = None
        self._rank_to_node_tensor_cache_key: Optional[Tuple[Any, ...]] = None
        self._rank_to_node_tensor_cache: Optional[torch.Tensor] = None
        self._decode_cache_aware_resident_sets_key: Optional[Tuple[Any, ...]] = None
        self._decode_cache_aware_resident_sets: Optional[List[set[int]]] = None
        self._node_representative_tensor_cache_key: Optional[Tuple[Any, ...]] = None
        self._node_representative_tensor_cache: Optional[torch.Tensor] = None
        self._expert_to_node_rep_comm_cpu_cache_key: Optional[Tuple[Any, ...]] = None
        self._expert_to_node_rep_comm_cpu_cache: Optional[List[int]] = None
        self._expert_to_node_rep_comm_tensor_cache_key: Optional[Tuple[Any, ...]] = None
        self._expert_to_node_rep_comm_tensor_cache: Optional[torch.Tensor] = None
        self._node_grouped_workspace: Dict[Tuple[Any, ...], torch.Tensor] = {}
        self._last_dispatch_count_global_total: int = 0

        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.dtype = dtype
        self.norm_topk_prob = bool(getattr(mlp_block, "norm_topk_prob", False))

        self.exec_routing_mode = str(routing_mode)
        if self.exec_routing_mode not in ("router", "rs"):
            raise ValueError(f"routing_mode must be router|rs, got {self.exec_routing_mode}")

        self.placement_mode = str(placement_mode)
        if self.placement_mode not in ("scp", "scp_ep", "round_robin", "weighted_random_balanced"):
            raise ValueError(f"placement_mode must be scp|scp_ep|round_robin|weighted_random_balanced, got {self.placement_mode}")

        self.planner_mode = str(planner_mode)
        if self.planner_mode not in ("none", "rs_prefetch"):
            raise ValueError(f"planner_mode must be none|rs_prefetch, got {self.planner_mode}")

        self.expert_source = str(expert_source)
        if self.expert_source not in ("router", "rs"):
            raise ValueError(f"expert_source must be router|rs, got {self.expert_source}")

        if self.runtime_mode != "baseline":
            raise ValueError(f"Unsupported runtime_mode: {self.runtime_mode}")

        if self.expert_source == "rs":
            if not (self.exec_routing_mode == "rs" or self.planner_mode == "rs_prefetch"):
                raise ValueError(
                    "expert_source=rs requires RS cache available, so routing_mode must be 'rs' "
                    "or planner_mode must be 'rs_prefetch'."
                )

        self.rs_group = int(rs_group)
        self.rs_debug = bool(rs_debug)
        self.prefetch_top_m = int(prefetch_top_m)
        self.oracle_mode = str(oracle_mode)
        self.oracle_ignore_cache_capacity = bool(oracle_ignore_cache_capacity)
        self._configured_expert_cache_capacity = int(expert_cache)
        expert_cache = _resolve_rank_expert_cache_capacity(
            int(expert_cache),
            global_rank=int(self.rank),
            comm_rank=int(self.comm_rank),
            world=int(self.world),
        )
        self._base_expert_cache_capacity = int(expert_cache)
        if self.layer_idx == 0:
            print(
                f"[EXPERT_CACHE] rank={self.rank} local_rank={self.env.local_rank} "
                f"comm_rank={self.comm_rank} configured={self._configured_expert_cache_capacity} "
                f"effective={self._base_expert_cache_capacity} "
                f"by_rank={os.environ.get('SPARSE_EXPERT_CACHE_BY_RANK', os.environ.get('EXPERT_CACHE_BY_RANK', ''))} "
                f"by_node={os.environ.get('SPARSE_EXPERT_CACHE_BY_NODE', '')}",
                flush=True,
            )

        # Stage-aware replicated SCP-EP controls.  Decode SCP is opt-in because
        # replicated placement is a dispatch choice, not a cache-residency plan;
        # enabling it during decode can expand the per-rank working set.
        #   SPARSE_SCP_APPLY_PREFILL=1/0
        #   SPARSE_SCP_APPLY_DECODE=1/0
        self.scp_apply_prefill = _env_flag("SPARSE_SCP_APPLY_PREFILL", "1")
        self.scp_apply_decode = _env_flag("SPARSE_SCP_APPLY_DECODE", "0")
        self.decode_cache_aware_scp = _env_flag("SPARSE_DECODE_CACHE_AWARE_SCP", "0")

        self.metrics = metrics
        self.debug_a2a_checks = bool(debug_a2a_checks)
        self.collect_per_layer = bool(collect_per_layer)
        self.log_per_layer = bool(log_per_layer)

        self.gate = mlp_block.gate.to(self.device, dtype=self.dtype).eval()
        self.moe_family = "qwen_shared" if hasattr(mlp_block, "shared_expert") else "deepseek_shared"
        self.shared_expert = (
            mlp_block.shared_expert.to(self.device, dtype=self.dtype).eval()
            if hasattr(mlp_block, "shared_expert")
            else None
        )
        self.shared_expert_gate = (
            mlp_block.shared_expert_gate.to(self.device, dtype=self.dtype).eval()
            if hasattr(mlp_block, "shared_expert_gate")
            else None
        )
        self.shared_experts = (
            mlp_block.shared_experts.to(self.device, dtype=self.dtype).eval()
            if hasattr(mlp_block, "shared_experts")
            else None
        )
        self.hidden_size = int(
            getattr(self.gate, "in_features", 0)
            or getattr(self.shared_expert_gate, "in_features", 0)
            or getattr(getattr(self.gate, "weight", None), "shape", [0, 0])[-1]
            or 0
        )
        self.experts = mlp_block.experts
        self.experts_kind = (
            "fused"
            if experts_are_fused(self.experts)
            else ("modulelist" if experts_are_modulelist(self.experts) else "unknown")
        )
        if self.experts_kind == "unknown":
            raise RuntimeError("Unsupported experts container. Need fused or modulelist experts.")

        # placement expert->rank
        # expert_to_rank/expert_to_rank_cpu remain the stage's primary dispatch
        # assignment used by legacy metrics.  For stage-aware SCP ablations, keep
        # explicit RR and SCP maps and choose at dispatch time.
        self.expert_to_rank = torch.empty(self.num_experts, dtype=torch.long, device=self.device)
        self.expert_to_rank_cpu: List[int] = []
        self.rr_expert_to_rank_cpu: List[int] = [int(eid % self.comm_world) for eid in range(self.num_experts)]
        self.rr_expert_to_rank = torch.tensor(self.rr_expert_to_rank_cpu, dtype=torch.long, device=self.device)
        self.scp_expert_to_rank_cpu: Optional[List[int]] = None
        self.scp_expert_to_rank: Optional[torch.Tensor] = None
        self.scp_prefill_expert_to_rank_cpu: Optional[List[int]] = None
        self.scp_prefill_expert_to_rank: Optional[torch.Tensor] = None
        self.scp_decode_expert_to_rank_cpu: Optional[List[int]] = None
        self.scp_decode_expert_to_rank: Optional[torch.Tensor] = None
        self.scp_prefill_dest_rank_by_node: Optional[torch.Tensor] = None
        self.scp_decode_dest_rank_by_node: Optional[torch.Tensor] = None
        self.scp_dest_rank_by_node: Optional[torch.Tensor] = None
        self.scp_prefill_dest_rank_by_node_cpu: Optional[List[List[int]]] = None
        self.scp_decode_dest_rank_by_node_cpu: Optional[List[List[int]]] = None
        self.scp_dest_rank_by_node_cpu: Optional[List[List[int]]] = None
        self.local_expert_ids: List[int] = []
        self._local_eid_set = set()
        self._light_policy = None
        self._rr_local_eid_set: set[int] = set(int(eid) for eid, r in enumerate(self.rr_expert_to_rank_cpu) if int(r) == int(self.comm_rank))
        self._rs_prefetch_eid_set: set[int] = set()
        self._scp_ep_replica_choice_cost_matrix: Optional[List[List[float]]] = None
        # In replicated SCP-EP this set means "experts that may be received/executed
        # by this rank from any source rank".  It is deliberately separate from
        # expert_to_rank, because expert_to_rank is source-specific dispatch choice.
        self._scp_ep_receive_eid_set: set[int] = set()
        self._scp_ep_prefill_receive_eid_set: set[int] = set()
        self._scp_ep_decode_receive_eid_set: set[int] = set()
        global_offset = self.layer_idx * self.num_experts

        wrb_assignment: Optional[List[int]] = None
        if self.placement_mode == "weighted_random_balanced":
            wrb_assignment = _weighted_random_balanced_assignment(self.num_experts, self.comm_world, self.layer_idx)
        scp_ep_assignment: Optional[List[int]] = None
        scp_ep_replicas: Optional[List[List[int]]] = None
        if self.placement_mode == "scp_ep":
            scp_ep_rank_to_node = self._comm_rank_node_ids()
            self._scp_ep_replica_choice_cost_matrix = _scp_ep_replica_choice_cost_matrix(
                scp_map,
                world=self.comm_world,
                rank_to_node=scp_ep_rank_to_node,
            )
            scp_ep_assignment = _scp_ep_layer_assignment(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                source_rank=self.comm_rank,
                rank_to_node=scp_ep_rank_to_node,
            )
            scp_ep_prefill_assignment = _scp_ep_layer_assignment(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                source_rank=self.comm_rank,
                stage="prefill",
                rank_to_node=scp_ep_rank_to_node,
            )
            scp_ep_decode_assignment = _scp_ep_layer_assignment(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                source_rank=self.comm_rank,
                stage="decode",
                rank_to_node=scp_ep_rank_to_node,
            )
            scp_ep_replicas = _scp_ep_layer_replicas(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                fallback_assignment=scp_ep_assignment,
            )
            scp_ep_receive_experts = _scp_ep_layer_receive_experts(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                rank=self.comm_rank,
                rank_to_node=scp_ep_rank_to_node,
            )
            scp_ep_prefill_receive_experts = _scp_ep_layer_receive_experts(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                rank=self.comm_rank,
                stage="prefill",
                rank_to_node=scp_ep_rank_to_node,
            )
            scp_ep_decode_receive_experts = _scp_ep_layer_receive_experts(
                scp_map,
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                world=self.comm_world,
                rank=self.comm_rank,
                stage="decode",
                rank_to_node=scp_ep_rank_to_node,
            )
        else:
            scp_ep_receive_experts = None
            scp_ep_prefill_receive_experts = None
            scp_ep_decode_receive_experts = None

        self._scp_ep_replicas = scp_ep_replicas

        initial_assignment: List[int] = []
        for eid in range(self.num_experts):
            gid = global_offset + eid
            if self.placement_mode == "scp":
                r = int(scp_map.get(str(gid), eid % self.comm_world)) % self.comm_world
            elif self.placement_mode == "scp_ep":
                if scp_ep_assignment is None:
                    raise RuntimeError("SCP-EP placement was not initialized")
                r = int(scp_ep_assignment[eid]) % self.comm_world
            elif self.placement_mode == "round_robin":
                r = eid % self.comm_world
            else:
                r = int(wrb_assignment[eid])
            initial_assignment.append(int(r))

        quota_repair = (
            self.placement_mode == "scp"
            and str(os.environ.get("SPARSE_SCP_QUOTA_REPAIR", "0")).strip().lower() in ("1", "true", "yes", "on")
        )
        if quota_repair:
            initial_assignment = _quota_repair_expert_assignment(initial_assignment, self.comm_world)

        if self.placement_mode == "scp_ep":
            self.scp_expert_to_rank_cpu = [int(x) for x in initial_assignment]
            self.scp_expert_to_rank = torch.tensor(self.scp_expert_to_rank_cpu, dtype=torch.long, device=self.device)
            self.scp_prefill_expert_to_rank_cpu = [int(x) for x in (scp_ep_prefill_assignment or initial_assignment)]
            self.scp_prefill_expert_to_rank = torch.tensor(self.scp_prefill_expert_to_rank_cpu, dtype=torch.long, device=self.device)
            self.scp_decode_expert_to_rank_cpu = [int(x) for x in (scp_ep_decode_assignment or initial_assignment)]
            self.scp_decode_expert_to_rank = torch.tensor(self.scp_decode_expert_to_rank_cpu, dtype=torch.long, device=self.device)
            source_nodes = sorted(set(int(x) for x in scp_ep_rank_to_node))
            if source_nodes:
                max_node = max(source_nodes)
                pre_rows: List[List[int]] = []
                dec_rows: List[List[int]] = []
                any_prefill_table = _scp_ep_table_for_stage(scp_map, "prefill") is not None or _scp_ep_table_for_stage(scp_map, "") is not None
                any_decode_table = _scp_ep_table_for_stage(scp_map, "decode") is not None or _scp_ep_table_for_stage(scp_map, "") is not None
                for node_id in range(int(max_node) + 1):
                    pre_row = _scp_ep_table_layer_assignment(
                        scp_map,
                        layer_idx=self.layer_idx,
                        num_experts=self.num_experts,
                        world=self.comm_world,
                        source_node=int(node_id),
                        stage="prefill",
                    ) if any_prefill_table else None
                    dec_row = _scp_ep_table_layer_assignment(
                        scp_map,
                        layer_idx=self.layer_idx,
                        num_experts=self.num_experts,
                        world=self.comm_world,
                        source_node=int(node_id),
                        stage="decode",
                    ) if any_decode_table else None
                    pre_rows.append([int(x) for x in (pre_row or self.scp_prefill_expert_to_rank_cpu)])
                    dec_rows.append([int(x) for x in (dec_row or self.scp_decode_expert_to_rank_cpu)])
                self.scp_prefill_dest_rank_by_node = torch.tensor(pre_rows, dtype=torch.long, device=self.device)
                self.scp_decode_dest_rank_by_node = torch.tensor(dec_rows, dtype=torch.long, device=self.device)
                self.scp_dest_rank_by_node = self.scp_prefill_dest_rank_by_node
                self.scp_prefill_dest_rank_by_node_cpu = [[int(x) for x in row] for row in pre_rows]
                self.scp_decode_dest_rank_by_node_cpu = [[int(x) for x in row] for row in dec_rows]
                self.scp_dest_rank_by_node_cpu = self.scp_prefill_dest_rank_by_node_cpu

        # A rank must own the weights for every expert it may execute in any
        # enabled stage.  In prefill-only SCP, for example, this is the union of
        # SCP replicas used during prefill and RR owners used during decode.
        use_scp_any_stage = bool(self.placement_mode == "scp_ep" and (self.scp_apply_prefill or self.scp_apply_decode))
        need_rr_any_stage = bool(
            self.placement_mode != "scp_ep"
            or (not self.scp_apply_prefill)
            or (not self.scp_apply_decode)
            or self.decode_cache_aware_scp
        )

        for eid, r in enumerate(initial_assignment):
            self.expert_to_rank_cpu.append(int(r))
            self.expert_to_rank[eid] = r

            scp_local_exec = False
            if scp_ep_replicas is not None and use_scp_any_stage:
                scp_local_exec = int(self.comm_rank) in set(int(x) for x in scp_ep_replicas[eid])

            rr_local_exec = False
            if need_rr_any_stage:
                rr_local_exec = int(self.rr_expert_to_rank_cpu[eid]) == int(self.comm_rank)

            if self.placement_mode == "scp_ep":
                local_exec = bool(scp_local_exec or rr_local_exec)
            else:
                local_exec = int(r) == int(self.comm_rank)

            if local_exec:
                self.local_expert_ids.append(eid)
                self._local_eid_set.add(eid)

        if scp_ep_receive_experts is not None:
            self._scp_ep_receive_eid_set = set(int(eid) for eid in scp_ep_receive_experts)
        if scp_ep_prefill_receive_experts is not None:
            self._scp_ep_prefill_receive_eid_set = set(int(eid) for eid in scp_ep_prefill_receive_experts)
        if scp_ep_decode_receive_experts is not None:
            self._scp_ep_decode_receive_eid_set = set(int(eid) for eid in scp_ep_decode_receive_experts)

        # Default RS candidate set for the current stage semantics.  Decode RS
        # should only prefetch RR-local experts when decode-SCP is disabled.
        self._rs_prefetch_eid_set = set(self._stage_prefetch_eid_set())

        logical_node_map = _parse_logical_node_map(self.world)
        self.expert_to_node_cpu: List[int] = []
        for r in initial_assignment:
            global_rank = int(self.comm_global_ranks[int(r)]) if int(r) < len(self.comm_global_ranks) else int(r)
            node_id = int(logical_node_map[global_rank]) if 0 <= global_rank < len(logical_node_map) else 0
            self.expert_to_node_cpu.append(node_id)
        self.expert_to_node = torch.tensor(self.expert_to_node_cpu, dtype=torch.long, device=self.device)

        self.expert_offload = str(expert_offload)
        self._native_decode_granularity = _env_flag("SPARSE_EP_NATIVE_DECODE_GRANULARITY", "0")
        if self.expert_offload not in ("cpu", "gpu"):
            raise ValueError(f"expert_offload must be cpu|gpu, got {self.expert_offload}")

        self.cache_policy = str(cache_policy).lower()
        if self.cache_policy not in ("lru", "lfu", "tiered"):
            raise ValueError(f"cache_policy must be lru|lfu|tiered, got {self.cache_policy}")

        oracle_async_requested = (
            self.oracle_mode == "replay"
            and str(os.environ.get("SPARSE_ORACLE_ASYNC_PREFETCH", "0")).strip().lower()
            not in ("", "0", "false", "no", "off")
        )
        self._global_expert_cache_enabled = (
            _env_flag("SPARSE_GLOBAL_EXPERT_CACHE", "0")
            and self.expert_offload == "cpu"
            and (self.planner_mode == "rs_prefetch" or oracle_async_requested)
        )
        if self._global_expert_cache_enabled:
            raw_cap = str(os.environ.get("SPARSE_GLOBAL_EXPERT_CACHE_CAPACITY", "")).strip()
            try:
                global_cap = int(raw_cap) if raw_cap else int(expert_cache) * int(num_layers)
            except Exception:
                global_cap = int(expert_cache) * int(num_layers)
            base_global_cap = max(1, int(expert_cache) * int(num_layers))
            global_cap = max(base_global_cap, int(global_cap))
            if self.layer_idx == 0:
                SparseContext.global_expert_cache_enabled = True
                SparseContext.global_expert_cache_capacity = int(global_cap)
                if self.rank == 0:
                    print(
                        f"[GLOBAL_EXPERT_CACHE] enabled=1 physical_capacity={global_cap} "
                        f"baseline_equiv_capacity={base_global_cap} "
                        f"layer_floor={os.environ.get('SPARSE_GLOBAL_EXPERT_CACHE_LAYER_FLOOR', '0')} "
                        f"per_layer_logical_capacity=local_experts",
                        flush=True,
                    )
            # In global-cache mode, per-layer caches are logical containers.
            # The physical resident budget is enforced across all layer blocks
            # below, so a target layer can admit future experts even when the
            # old per-layer expert_cache would have been full.
            self._cache = ExpertWeightCache(max(1, int(len(self.local_expert_ids))))
        elif self.cache_policy in ("lru", "lfu"):
            self._cache = ExpertWeightCache(expert_cache)
        else:
            base_capacity = int(expert_cache)
            prefetch_enabled = (self.planner_mode == "rs_prefetch")
            prefetch_extra_capacity = (
                max(0, int(os.environ.get("SPARSE_PREFETCH_EXTRA_CAPACITY", "0") or 0))
                if prefetch_enabled else 0
            )
            total_capacity = int(base_capacity + prefetch_extra_capacity)
            if base_capacity <= 1:
                hot_capacity = 0
                prefetch_reserved = 0
            else:
                # Tiered baseline must not self-harm when planner is disabled.
                # Keep a tiny reserved speculative area only for planner-only runs.
                if base_capacity >= 12:
                    transient_budget = 4
                elif base_capacity >= 10:
                    transient_budget = 3
                elif base_capacity >= 8:
                    transient_budget = 3
                else:
                    transient_budget = max(2, base_capacity // 3)

                transient_budget = min(transient_budget, max(1, base_capacity - 1))
                # Conservative default: under a fixed expert_cache budget, RS
                # prefetch must not steal a demand cache slot unless explicitly
                # requested. Otherwise low-precision prefetch can reduce the
                # baseline resident hit rate more than it helps.
                if prefetch_enabled:
                    prefetch_reserved = int(
                        os.environ.get(
                            "SPARSE_PREFETCH_RESERVED",
                            "0",
                        ) or 0
                    )
                else:
                    prefetch_reserved = 0
                prefetch_reserved = max(0, int(prefetch_reserved))
                if prefetch_reserved > 0 and prefetch_extra_capacity <= 0:
                    # Fixed-budget RS mode: reserve speculative room inside the
                    # configured cache instead of silently requiring extra cache
                    # capacity.  Keep the demand-transient budget close to the
                    # old cache8 split by taking the room from the hot tier.
                    transient_budget = min(
                        max(1, base_capacity - 1),
                        int(transient_budget) + int(prefetch_reserved),
                    )
                prefetch_reserved = min(prefetch_reserved, max(0, int(transient_budget)))
                hot_capacity = max(1, base_capacity - transient_budget)
            self._cache = (
                ExpertWeightCache(total_capacity)
                if hot_capacity <= 0
                else TieredExpertWeightCache(
                    total_capacity,
                    hot_capacity=hot_capacity,
                    prefetch_reserved=prefetch_reserved,
                )
            )
            if self.rank == 0 and self.layer_idx == 0:
                print(
                    f"[CACHE_SPLIT] configured={base_capacity} total={total_capacity} extra_prefetch={prefetch_extra_capacity} "
                    f"hot={hot_capacity} transient={total_capacity - hot_capacity} "
                    f"demand_transient={getattr(self._cache, 'transient_demand_capacity', 0)} "
                    f"prefetch_reserved={getattr(self._cache, 'transient_prefetch_capacity', 0)}",
                    flush=True,
                )
        self._host_fused_cpu_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._host_module_cpu_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._oracle_gpu_fused: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._oracle_gpu_module: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._epipe_prefill_extra_cache = EpipePrefillExtraCache(
            max(0, _env_int("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_SLOTS", 0))
        )
        self._epipe_prefill_extra_cache_decode_read = _env_flag("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_DECODE_READ", "1")
        self._epipe_prefill_extra_cache_clear_at_decode = _env_flag("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_CLEAR_AT_DECODE", "0")
        self._epipe_prefill_extra_cache_persist_requests = _env_flag("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_PERSIST_REQUESTS", "0")
        self._epipe_prefill_extra_cache_min_free_mb = max(
            0.0,
            _env_float(
                "SPARSE_EPIPE_PREFILL_EXTRA_CACHE_MIN_FREE_MB",
                float(os.environ.get("SPARSE_EPIPE_PREFILL_MIN_FREE_MB", "1024") or 1024.0),
            ),
        )
        self._epipe_prefill_extra_cache_max_mb = max(
            0.0,
            _env_float("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_MAX_MB", 128.0),
        )

        self.rs_prob_to_logit_eps = float(rs_prob_to_logit_eps)
        self.rs_require_complete = bool(rs_require_complete)

        self._oracle_async_prefetch = (
            self.oracle_mode == "replay"
            and str(os.environ.get("SPARSE_ORACLE_ASYNC_PREFETCH", "0")).strip().lower()
            not in ("", "0", "false", "no", "off")
        )
        self._transition_prefetch_enabled = _env_flag("SPARSE_TRANSITION_PREFETCH", "0")
        self._epipe_enable = _env_flag("SPARSE_EPIPE_ENABLE", "0")
        self._epipe_shadow = _env_flag("SPARSE_EPIPE_SHADOW", "0")
        self._epipe_mode = str(os.environ.get("SPARSE_EPIPE_MODE", "upfront_scan") or "upfront_scan").strip().lower()
        self._epipe_decode_with_rs_mode = str(
            os.environ.get(
                "SPARSE_EPIPE_DECODE_RS_MODE",
                os.environ.get("SPARSE_EPIPE_DECODE_WITH_RS_MODE", "late_only"),
            )
            or "late_only"
        ).strip().lower()
        if self._epipe_decode_with_rs_mode == "disabled":
            self._epipe_decode_with_rs_mode = "off"
        if self._epipe_decode_with_rs_mode == "late":
            self._epipe_decode_with_rs_mode = "late_only"
        if self._epipe_decode_with_rs_mode == "demand":
            self._epipe_decode_with_rs_mode = "demand_only"
        if self._epipe_decode_with_rs_mode in (
            "scan",
            "residual",
            "residual-scan",
            "buffered_scan",
            "buffered-scan",
        ):
            self._epipe_decode_with_rs_mode = "residual_scan"
        self._epipe_decode_only = _env_flag("SPARSE_EPIPE_DECODE_ONLY", "1")
        self._epipe_no_evict = _env_flag("SPARSE_EPIPE_NO_EVICT", "1")
        self._epipe_transient_staging = _env_flag("SPARSE_EPIPE_TRANSIENT_STAGING", "1")
        self._epipe_max_staged_experts = max(0, _env_int("SPARSE_EPIPE_MAX_STAGED_EXPERTS", 1))
        self._epipe_min_free_mb = max(0.0, _env_float("SPARSE_EPIPE_MIN_FREE_MB", 256.0))
        self._epipe_prefill_min_free_mb = max(0.0, _env_float("SPARSE_EPIPE_PREFILL_MIN_FREE_MB", 1024.0))
        self._epipe_lookahead_experts = max(0, _env_int("SPARSE_EPIPE_LOOKAHEAD_EXPERTS", 4))
        self._epipe_lookahead_min_ready_items = max(0, _env_int("SPARSE_EPIPE_LOOKAHEAD_MIN_READY_ITEMS", 64))
        self._epipe_prefill_lookahead_min_ready_items = max(0, _env_int("SPARSE_EPIPE_PREFILL_LOOKAHEAD_MIN_READY_ITEMS", 256))
        self._epipe_sort_ready_by_work = _env_flag("SPARSE_EPIPE_SORT_READY_BY_WORK", "0")
        self._epipe_rs_buffer_max_groups = max(1, _env_int("SPARSE_EPIPE_RS_BUFFER_MAX_GROUPS", 4))
        self._epipe_rs_buffer_max_items = max(1, _env_int("SPARSE_EPIPE_RS_BUFFER_MAX_ITEMS", 32))
        self._epipe_rs_min_buffer_groups = max(1, _env_int("SPARSE_EPIPE_RS_MIN_BUFFER_GROUPS", 1))
        self._epipe_rs_min_buffer_items = max(1, _env_int("SPARSE_EPIPE_RS_MIN_BUFFER_ITEMS", 1))
        self._epipe_rs_handle_late_prefetch = _env_flag("SPARSE_EPIPE_RS_HANDLE_LATE_PREFETCH", "1")
        self._epipe_rs_handle_demand_miss = _env_flag(
            "SPARSE_EPIPE_RS_HANDLE_DEMAND_MISS",
            "0",
        )
        if self._epipe_decode_with_rs_mode == "late_only":
            self._epipe_rs_handle_demand_miss = False
        if self._epipe_decode_with_rs_mode == "demand_only":
            self._epipe_rs_handle_late_prefetch = False
        self._epipe_rs_demand_commit = _env_flag("SPARSE_EPIPE_RS_DEMAND_COMMIT", "1")
        self._epipe_rs_force_demand_stage = _env_flag("SPARSE_EPIPE_RS_FORCE_DEMAND_STAGE", "0")
        self._epipe_rs_demand_stage_threshold_ms = max(0.0, _env_float("SPARSE_EPIPE_RS_DEMAND_STAGE_THRESHOLD_MS", 0.0))
        self._epipe_rs_rolling_gate = _env_flag("SPARSE_EPIPE_RS_ROLLING_GATE", "1")
        self._epipe_rs_gate_window = max(1, _env_int("SPARSE_EPIPE_RS_GATE_WINDOW", 16))
        self._epipe_rs_gate_min_triggers = max(0, _env_int("SPARSE_EPIPE_RS_GATE_MIN_TRIGGERS", 1))
        self._epipe_rs_gate_skip_steps = max(0, _env_int("SPARSE_EPIPE_RS_GATE_SKIP_STEPS", 8))
        self._epipe_rs_gate_calls = 0
        self._epipe_rs_gate_triggers = 0
        self._epipe_rs_gate_skip_remaining = 0
        self._epipe_stage_first_missing_only = (
            _env_flag("SPARSE_EPIPE_STAGE_FIRST_MISSING_ONLY", "1")
            and _env_flag("SPARSE_EPIPE_STAGE_FIRST_MISS_ONLY", "1")
        )
        self._epipe_min_ready_experts = max(0, _env_int("SPARSE_EPIPE_MIN_READY_EXPERTS", 1))
        self._epipe_min_missing_experts = max(0, _env_int("SPARSE_EPIPE_MIN_MISSING_EXPERTS", 1))
        self._epipe_min_ready_items = max(0, _env_int("SPARSE_EPIPE_MIN_READY_ITEMS", 1))
        self._epipe_min_missing_h2d_bytes = max(0, _env_int("SPARSE_EPIPE_MIN_MISSING_H2D_BYTES", 1))
        self._epipe_deferred_lru_commit = _env_flag("SPARSE_EPIPE_DEFERRED_LRU_COMMIT", "1")
        self._epipe_decode_rs_commit = _env_flag("SPARSE_EPIPE_DECODE_RS_COMMIT", "1")
        self._epipe_decode_no_rs_commit = _env_flag("SPARSE_EPIPE_DECODE_NO_RS_COMMIT", "1")
        self._epipe_prefill_enable = _env_flag("SPARSE_EPIPE_PREFILL_ENABLE", "0")
        self._epipe_prefill_shadow = _env_flag("SPARSE_EPIPE_PREFILL_SHADOW", "0")
        self._epipe_prefill_multistage_shadow = _env_flag("SPARSE_EPIPE_PREFILL_MULTI_STAGE_SHADOW", "0")
        self._epipe_prefill_multistage_max_n = max(1, _env_int("SPARSE_EPIPE_PREFILL_MULTI_STAGE_MAX_N", 4))
        self._epipe_prefill_policy = str(os.environ.get("SPARSE_EPIPE_PREFILL_POLICY", "deferred_lru_commit") or "deferred_lru_commit").strip().lower()
        if self._epipe_prefill_policy in ("transient", "no_commit", "transient-no-commit"):
            self._epipe_prefill_policy = "transient_no_commit"
        prefill_commit_default = "1" if self._epipe_prefill_policy in ("deferred_lru_commit", "commit", "lru_commit") else "0"
        prefill_mark_default = "1" if prefill_commit_default == "1" else "0"
        self._epipe_prefill_commit = _env_flag("SPARSE_EPIPE_PREFILL_COMMIT", prefill_commit_default)
        self._epipe_prefill_mark_transient = _env_flag("SPARSE_EPIPE_PREFILL_MARK_TRANSIENT", prefill_mark_default)
        self._epipe_prefill_no_touch_ready = _env_flag("SPARSE_EPIPE_PREFILL_NO_TOUCH_READY", "0")
        self._epipe_promote_transient_on_decode_hit = _env_flag("SPARSE_EPIPE_PROMOTE_TRANSIENT_ON_DECODE_HIT", "1")
        self._epipe_timing = _env_flag("SPARSE_EPIPE_TIMING", "0")
        self._epipe_debug_counters = _env_flag("SPARSE_EPIPE_DEBUG_COUNTERS", "0")
        self._epipe_timing_sample_rate = max(0.0, _env_float("SPARSE_EPIPE_TIMING_SAMPLE_RATE", 0.0))
        self._epipe_validate_output = _env_flag("SPARSE_EPIPE_VALIDATE_OUTPUT", "0")
        if self._epipe_enable and self._epipe_max_staged_experts != 1:
            raise NotImplementedError("EPIPE-v2 currently supports only SPARSE_EPIPE_MAX_STAGED_EXPERTS=1.")
        if self._epipe_enable and not self._epipe_stage_first_missing_only:
            raise NotImplementedError("EPIPE-v2 currently stages only the first missing expert.")
        self.prefetch_stream = (
            torch.cuda.Stream(device=self.device)
            if (
                self.expert_offload == "cpu"
                and torch.cuda.is_available()
                and (
                    self.planner_mode == "rs_prefetch"
                    or self._oracle_async_prefetch
                    or self._transition_prefetch_enabled
                )
            )
            else None
        )
        self.epipe_stream = (
            torch.cuda.Stream(device=self.device)
            if (
                self.expert_offload == "cpu"
                and torch.cuda.is_available()
                and self.device.type == "cuda"
                and bool(
                    self._epipe_enable
                    or self._epipe_prefill_enable
                    or self._epipe_prefill_shadow
                    or self._epipe_prefill_multistage_shadow
                )
            )
            else None
        )
        self._prefetched_ids_this_group: set[int] = set()
        self._prefetch_used_ids_this_group: set[int] = set()
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        self._expert_access_freq: Dict[int, int] = {}
        # retained prefetch lifecycle for current RS group horizon
        self._prefetch_expire_layer: Dict[int, int] = {}
        self._prefetch_current_group_start: Optional[int] = None
        self._prefetch_current_group_end: Optional[int] = None
        self._prefetch_current_cycle_key: Optional[Tuple[Any, ...]] = None

        # runtime controls for active replacement
        self._prefetch_allow_active_replace = bool(int(os.environ.get("SPARSE_PREFETCH_ALLOW_ACTIVE_REPLACE", "1")))
        self._prefetch_replace_margin = float(os.environ.get("SPARSE_PREFETCH_REPLACE_MARGIN", "1.05"))
        self._prefetch_replace_advantage = float(os.environ.get("SPARSE_PREFETCH_REPLACE_ADVANTAGE", "0.02"))
        self._prefetch_min_candidate_score = float(os.environ.get("SPARSE_PREFETCH_MIN_CAND_SCORE", "0.02"))
        self._cache_value_score_scale = float(os.environ.get("SPARSE_CACHE_VALUE_SCORE_SCALE", "4096.0"))
        self._cache_value_min_candidate_score = float(os.environ.get("SPARSE_CACHE_VALUE_MIN_CAND_SCORE", "0.0"))
        self._cache_value_replace_advantage = float(os.environ.get("SPARSE_CACHE_VALUE_REPLACE_ADVANTAGE", "0.02"))

        self._cache_ttl_steps = max(0, _env_int("SPARSE_CACHE_TTL_STEPS", 0))
        self._last_used_decode_step: Dict[int, int] = {}
        self._cache_eviction_fallback_all_protected = 0
        self._transition_table = self._load_transition_table() if self._transition_prefetch_enabled else {}
        self._transition_topk = max(1, _env_int("SPARSE_TRANSITION_TOPK", 1))
        self._transition_no_replace = _env_flag("SPARSE_TRANSITION_NO_REPLACE", "1")
        self._transition_safe_evict = _env_flag("SPARSE_TRANSITION_SAFE_EVICT", "1")
        self._transition_rate_limit = _env_flag("SPARSE_TRANSITION_RATE_LIMIT", "1")
        self._transition_prefetched_ids: set[int] = set()
        self._transition_used_ids: set[int] = set()

        self.owner_load_balance_alpha = float(os.environ.get("SPARSE_OWNER_LOAD_BALANCE_ALPHA", "0.35"))
        self.owner_workingset_gamma = float(os.environ.get("SPARSE_OWNER_WORKINGSET_GAMMA", "0.08"))
        self.owner_change_penalty_beta = float(os.environ.get("SPARSE_OWNER_CHANGE_PENALTY_BETA", "0.20"))
        self.owner_stickiness_delta = float(os.environ.get("SPARSE_OWNER_STICKINESS_DELTA", "0.10"))

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
        need_rs_bank = (self.expert_source == "rs") or (self.planner_mode == "rs_prefetch")
        if need_rs_bank and SparseContext.rs_bank is None:
            hidden_dim = getattr(self.gate, "hidden_dim", None)
            if hidden_dim is None and hasattr(self.gate, "weight"):
                hidden_dim = int(self.gate.weight.shape[-1])
            if hidden_dim is None and int(getattr(self, "hidden_size", 0) or 0) > 0:
                hidden_dim = int(self.hidden_size)
            if hidden_dim is None:
                raise RuntimeError("Cannot infer hidden_dim for RS.")

            rs_path = Path(rs_dir)
            bank_modules: List[nn.Module] = []
            missing: List[str] = []
            bank_task: Optional[str] = None
            source_layers: List[int] = []
            next_layer_layout = _is_next_layer_rs_layout(rs_path)
            next_layer_targets = _next_layer_rs_target_layers(rs_path) if next_layer_layout else []

            for l in range(self.num_layers):
                if next_layer_layout:
                    in_layer = int(l) - 1
                    cv_ckpt = rs_path / f"cv_l{l}_inL{in_layer}.pth"
                    legacy_ckpt = rs_path / f"rs_l{l}_inL{in_layer}.pth"
                else:
                    in_layer = (l // self.rs_group) * self.rs_group
                    cv_ckpt = rs_path / f"cv_l{l}_inL{in_layer}.pth"
                    legacy_ckpt = rs_path / f"rs_l{l}_inL{in_layer}.pth"

                ckpt_path: Optional[Path] = None
                if cv_ckpt.exists():
                    ckpt_path = cv_ckpt
                elif legacy_ckpt.exists():
                    ckpt_path = legacy_ckpt

                if ckpt_path is None:
                    optional_next_layer_miss = next_layer_layout and _is_optional_next_layer_rs_miss(
                        int(l), next_layer_targets
                    )
                    if not optional_next_layer_miss:
                        missing.append(f"[L{l}] tried: {cv_ckpt.name}, {legacy_ckpt.name}")
                    if self.rs_require_complete:
                        if not optional_next_layer_miss:
                            continue
                    dummy = RoutingScoutMLP(
                        input_dim=hidden_dim,
                        num_experts=self.num_experts,
                        hidden_dim=1024,
                    ).to(self.device, dtype=torch.float32).eval()
                    bank_modules.append(dummy)
                    source_layers.append(int(in_layer))
                    if bank_task is None:
                        bank_task = "router_prob"
                    continue

                payload = torch.load(str(ckpt_path), map_location="cpu")
                mod, task = _build_rs_module_from_payload(
                    payload,
                    fallback_input_dim=hidden_dim,
                    fallback_num_experts=self.num_experts,
                )

                if task == "cache_value_rs" and (self.exec_routing_mode == "rs" or self.expert_source == "rs"):
                    raise RuntimeError(
                        "cache_value_rs is planner-only. Use routing_mode=router and expert_source=router."
                    )

                if bank_task is None:
                    bank_task = task
                elif bank_task != task:
                    raise RuntimeError(
                        f"Mixed RS tasks in rs_dir are not allowed: layer {l} has task={task}, previous task={bank_task}"
                    )

                bank_modules.append(mod.to(self.device, dtype=torch.float32).eval())
                source_layers.append(int(payload.get("source_layer", in_layer)) if isinstance(payload, dict) else int(in_layer))

            if self.rs_require_complete and missing:
                raise RuntimeError("RS checkpoint set incomplete:\n" + "\n".join(missing[:32]))
            if len(bank_modules) != self.num_layers:
                raise RuntimeError(
                    f"RS bank size mismatch: got {len(bank_modules)} modules, expect {self.num_layers}"
                )

            SparseContext.rs_bank = nn.ModuleList(bank_modules).eval()
            SparseContext.rs_group = self.rs_group
            SparseContext.rs_bank_task = str(bank_task or "router_prob")
            SparseContext.rs_bank_layout = "next_layer" if next_layer_layout else "group"
            SparseContext.rs_bank_source_layers = list(source_layers)
            if self.rank == 0:
                print(
                    f"[RS_BANK] layout={SparseContext.rs_bank_layout} task={SparseContext.rs_bank_task} "
                    f"modules={len(bank_modules)} rs_dir={rs_path}",
                    flush=True,
                )

            late_rs_dir = str(os.environ.get("SPARSE_LATE_RS_DIR", "") or "").strip()
            two_stage_enabled = _env_flag("SPARSE_RS_TWO_STAGE", "0")
            if two_stage_enabled and late_rs_dir:
                late_path = Path(late_rs_dir)
                late_bank, late_task, late_layout, late_sources, _late_missing = _load_runtime_rs_bank(
                    rs_path=late_path,
                    num_layers=self.num_layers,
                    hidden_dim=int(hidden_dim),
                    num_experts=self.num_experts,
                    device=self.device,
                    require_complete=self.rs_require_complete,
                )
                if late_task != "router_prob":
                    raise RuntimeError("SPARSE_LATE_RS_DIR must contain router-probability RS checkpoints.")
                if late_layout != "next_layer":
                    raise RuntimeError("SPARSE_LATE_RS_DIR must use next-layer checkpoints, e.g. rs_l1_inL0.pth.")
                SparseContext.late_rs_bank = late_bank
                SparseContext.late_rs_bank_task = late_task
                SparseContext.late_rs_bank_layout = late_layout
                SparseContext.late_rs_bank_source_layers = list(late_sources)
                if self.rank == 0:
                    print(
                        f"[LATE_RS_BANK] layout={SparseContext.late_rs_bank_layout} "
                        f"task={SparseContext.late_rs_bank_task} modules={len(late_bank)} "
                        f"late_rs_dir={late_path}",
                        flush=True,
                    )


        if log_routing_stats:
            SparseContext.enable_stats(num_layers=self.num_layers, num_experts=self.num_experts, device=self.device)

    def _prefetch_group_hot_experts_for_local_batch(
        self,
        group_topi: Dict[int, torch.Tensor],   # layer -> [B_local*S, K]
        group_topv: Dict[int, torch.Tensor],   # layer -> [B_local*S, K]
    ) -> None:
        if self.planner_mode != "rs_prefetch":
            return
        if self.expert_offload != "cpu":
            return
        top_m = int(self._effective_rs_prefetch_top_m())
        if top_m <= 0 or self._effective_prefetch_capacity() <= 0:
            return

        score_map: Dict[int, float] = {}

        for l in sorted(group_topi.keys()):
            flat_i = group_topi[l].reshape(-1)
            flat_w = group_topv[l].float().reshape(-1)
            for eid, wv in zip(flat_i.tolist(), flat_w.tolist()):
                score_map[int(eid)] = score_map.get(int(eid), 0.0) + float(wv)

        if len(score_map) == 0:
            return

        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        ranked = ranked[: min(top_m, self._effective_prefetch_capacity(), len(ranked))]

        ctx = torch.cuda.stream(self.prefetch_stream) if self.prefetch_stream is not None else nullcontext()
        with ctx:
            for eid, _ in ranked:
                if self._cache.has(eid):
                    payload = self._cache.get(eid)
                    self._cache_put_guarded(eid, payload, pinned=True)
                    self._prefetched_ids_this_group.add(eid)
                    continue

                if self.experts_kind == "fused":
                    gate_up, down = self._get_any_fused_expert_to_device(
                        eid=eid,
                        is_prefetch=True,
                        count_as_access=False,
                    )
                    self._cache_put_guarded(eid, (gate_up, down), pinned=True)
                else:
                    gw, uw, dw = self._get_any_module_expert_to_device(
                        eid=eid,
                        is_prefetch=True,
                        count_as_access=False,
                    )
                    self._cache_put_guarded(eid, (gw, uw, dw), pinned=True)

                self._prefetched_ids_this_group.add(eid)

    def _prefetch_next_layer_experts_for_local_batch(
        self,
        topi_local: torch.Tensor,    # [B_local*S, K]
        topv_local: torch.Tensor,    # [B_local*S, K]
    ) -> None:
        if self.planner_mode != "rs_prefetch":
            return
        if self.expert_offload != "cpu":
            return
        top_m = int(self._effective_rs_prefetch_top_m())
        if top_m <= 0 or self._effective_prefetch_capacity() <= 0:
            return

        self._begin_group_prefetch_cycle()

        score_map: Dict[int, float] = {}
        flat_i = topi_local.reshape(-1)
        flat_w = topv_local.float().reshape(-1)
        for eid, wv in zip(flat_i.tolist(), flat_w.tolist()):
            score_map[int(eid)] = score_map.get(int(eid), 0.0) + float(wv)

        if len(score_map) == 0:
            return

        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        ranked = ranked[: min(top_m, self._effective_prefetch_capacity(), len(ranked))]

        ctx = torch.cuda.stream(self.prefetch_stream) if self.prefetch_stream is not None else nullcontext()
        with ctx:
            for eid, _ in ranked:
                if self._cache.has(eid):
                    payload = self._cache.get(eid)
                    self._cache_put_guarded(eid, payload, pinned=True)
                    self._prefetched_ids_this_group.add(eid)
                    continue

                if self.experts_kind == "fused":
                    gate_up, down = self._get_any_fused_expert_to_device(
                        eid=eid,
                        is_prefetch=True,
                        count_as_access=False,
                    )
                    self._cache_put_guarded(eid, (gate_up, down), pinned=True)
                else:
                    gw, uw, dw = self._get_any_module_expert_to_device(
                        eid=eid,
                        is_prefetch=True,
                        count_as_access=False,
                    )
                    self._cache_put_guarded(eid, (gw, uw, dw), pinned=True)

                self._prefetched_ids_this_group.add(eid)


    def _ensure_layer_perf(self) -> Optional[LayerPerf]:
        if self.metrics is None or self.metrics.per_layer is None:
            return None
        if self.layer_idx not in self.metrics.per_layer:
            self.metrics.per_layer[self.layer_idx] = LayerPerf()

        lp = self.metrics.per_layer[self.layer_idx]
        if lp.expert_hist is None or int(lp.expert_hist.numel()) != int(self.num_experts):
            lp.expert_hist = torch.zeros(
                (self.num_experts,),
                device=self.device,
                dtype=torch.float64,
            )
        return lp

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
            if is_prefetch:
                lp.h2d_prefetch_bytes += int(nbytes)
            else:
                lp.h2d_demand_bytes += int(nbytes)
        if not bool(is_prefetch):
            cp = _hclm_cp_profiler()
            step_id = int(getattr(SparseContext, "runtime_call_id", -1))
            if cp.should_sample(step_id):
                cp.record_layer_summary(
                    step_id,
                    int(self.layer_idx),
                    {"demand_h2d_bytes": int(nbytes)},
                )

    def _record_cache_hit(self, *, is_prefetch_hit: bool = False) -> None:
        if self.metrics is not None:
            if is_prefetch_hit:
                self.metrics.prefetch_hit_count += 1
            else:
                self.metrics.cache_hit_count += 1
        lp = self._ensure_layer_perf()
        if lp is not None:
            if is_prefetch_hit:
                lp.prefetch_hit_count += 1
            else:
                lp.cache_hit_count += 1

    def _record_cache_miss(self) -> None:
        if self.metrics is not None:
            self.metrics.cache_miss_count += 1
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.cache_miss_count += 1


    def _record_prefetch_admitted(self) -> None:
        if self.metrics is not None:
            self.metrics.prefetch_admitted_count += 1
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.prefetch_admitted_count += 1

    def _record_prefetch_wasted(self, count: int) -> None:
        count = int(count)
        if count <= 0:
            return
        if self.metrics is not None:
            self.metrics.prefetch_wasted_count += count
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.prefetch_wasted_count += count

    def _record_rs_counter(self, name: str, value: int = 1) -> None:
        value = int(value)
        if value <= 0:
            return
        if self.metrics is not None and hasattr(self.metrics, name):
            setattr(self.metrics, name, int(getattr(self.metrics, name, 0)) + value)
        lp = self._ensure_layer_perf()
        if lp is not None and hasattr(lp, name):
            setattr(lp, name, int(getattr(lp, name, 0)) + value)

    def _record_rs_resolver_match(self, *, eid: int, target_rank: int, legacy_local: bool, value: int = 1) -> None:
        if self.metrics is None:
            return
        value = int(value)
        if value <= 0:
            return
        stage = "decode" if str(getattr(SparseContext, "current_phase", "")).strip().lower() == "decode" else self._current_stage()
        table_target = self._resolve_table_rank_for_expert(
            source_rank=int(self.comm_rank),
            expert_id=int(eid),
            stage=stage,
        )
        if table_target is not None:
            matched = int(target_rank) == int(table_target)
            if bool(matched):
                self.metrics.rs_prefetch_target_resolver_match_count += int(value)
            else:
                self.metrics.rs_prefetch_target_resolver_mismatch_count += int(value)
                self.metrics.rs_scp_target_mismatch_count += int(value)
                if _env_flag("STRICT_RS_SCP_TARGET_MATCH", "0") or _env_flag("SPARSE_STRICT_RS_SCP_TARGET_MATCH", "0"):
                    raise RuntimeError(
                        "RS/SCP target mismatch: "
                        f"layer={int(self.layer_idx)} expert={int(eid)} "
                        f"rs_target={int(target_rank)} table_target={int(table_target)} stage={stage}"
                    )
            return
        if int(target_rank) == int(self.comm_rank):
            self.metrics.rs_prefetch_target_resolver_match_count += int(value)
            if not bool(legacy_local):
                self.metrics.rs_prefetch_would_be_wrong_without_resolver_count += int(value)
        else:
            self.metrics.rs_prefetch_target_resolver_mismatch_count += int(value)
            if bool(legacy_local):
                self.metrics.rs_prefetch_would_be_wrong_without_resolver_count += int(value)

    def _record_metric_counter(self, name: str, value: int = 1) -> None:
        value = int(value)
        if value <= 0:
            return
        if self.metrics is not None and hasattr(self.metrics, name):
            setattr(self.metrics, name, int(getattr(self.metrics, name, 0)) + value)

    _EPIPE_ALWAYS_COUNTERS = {
        "epipe_executor_attempt_count",
        "epipe_executor_calls",
        "epipe_enabled_count",
        "epipe_staged_expert_count",
        "epipe_staged_h2d_bytes",
        "epipe_demand_h2d_bytes",
        "epipe_prefill_enabled_count",
        "epipe_decode_no_rs_enabled_count",
        "epipe_decode_rs_late_overlap_count",
        "epipe_decode_rs_demand_stage_count",
        "epipe_rs_late_prefetch_overlap_count",
        "epipe_rs_demand_stage_count",
        "epipe_rs_demand_stage_disabled_by_gate_count",
        "epipe_rs_demand_stage_force_count",
        "epipe_rs_buffered_calls",
        "epipe_rs_buffer_flush_count",
        "epipe_rs_buffered_ready_group_count",
        "epipe_rs_buffered_ready_item_count",
        "epipe_rs_buffer_too_small_count",
        "epipe_rs_trigger_count",
        "epipe_rs_late_prefetch_trigger_count",
        "epipe_rs_demand_miss_trigger_count",
        "epipe_rs_residual_scan_calls",
        "epipe_rs_residual_scan_ready_group_count",
        "epipe_rs_residual_scan_ready_item_count",
        "epipe_rs_residual_scan_residual_count",
        "epipe_rs_residual_scan_late_count",
        "epipe_rs_residual_scan_demand_count",
        "epipe_rs_residual_scan_no_ready_count",
        "epipe_prefill_executor_calls",
        "epipe_prefill_stage_success_count",
        "epipe_decode_no_rs_executor_calls",
        "epipe_decode_no_rs_stage_success_count",
        "epipe_decode_rs_executor_calls",
        "epipe_decode_rs_stage_success_count",
        "epipe_transient_payload_count",
        "epipe_prefill_transient_no_commit_count",
        "epipe_prefill_admission_bypass_count",
        "epipe_cache_insert_count",
        "epipe_prefill_cache_insert_count",
        "epipe_prefill_extra_cache_insert_count",
        "epipe_prefill_extra_cache_hit_count",
        "epipe_prefill_extra_cache_evict_count",
        "epipe_prefill_extra_cache_no_room_count",
        "epipe_prefill_extra_cache_guard_skip_count",
        "epipe_prefill_extra_cache_h2d_bytes",
        "epipe_prefill_extra_cache_resident_count",
        "epipe_prefill_extra_cache_resident_at_decode_start",
        "epipe_prefill_extra_cache_resident_at_decode_end",
        "epipe_prefill_multistage_shadow_count",
        "epipe_prefill_multistage_ready_expert_count",
        "epipe_prefill_multistage_missing_expert_count",
        "epipe_prefill_multistage_ready_item_count",
        "epipe_prefill_multistage_missing_item_count",
        "epipe_prefill_multistage_top_missing_item_count_sum",
        "epipe_prefill_multistage_est_expert_h2d_bytes",
        "epipe_prefill_multistage_est_ready_compute_items",
        "epipe_prefill_multistage_est_hidden_h2d_items_n1",
        "epipe_prefill_multistage_est_hidden_h2d_items_n2",
        "epipe_prefill_multistage_est_hidden_h2d_items_n4",
    }

    def _record_epipe_counter(self, name: str, value: int = 1) -> None:
        value = int(value)
        if value <= 0:
            return
        if (
            not bool(getattr(self, "_epipe_debug_counters", False))
            and str(name) not in self._EPIPE_ALWAYS_COUNTERS
        ):
            return
        if self.metrics is not None and hasattr(self.metrics, name):
            setattr(self.metrics, name, int(getattr(self.metrics, name, 0)) + value)
        lp = self._ensure_layer_perf()
        if lp is not None and hasattr(lp, name):
            setattr(lp, name, int(getattr(lp, name, 0)) + value)

    def _record_epipe_float(self, name: str, value: float) -> None:
        value = float(value)
        if value <= 0.0:
            return
        if self.metrics is not None and hasattr(self.metrics, name):
            setattr(self.metrics, name, float(getattr(self.metrics, name, 0.0)) + value)
        lp = self._ensure_layer_perf()
        if lp is not None and hasattr(lp, name):
            setattr(lp, name, float(getattr(lp, name, 0.0)) + value)

    def _record_expert_executor_call(self, name: str) -> None:
        if self.metrics is not None and hasattr(self.metrics, name):
            setattr(self.metrics, name, int(getattr(self.metrics, name, 0)) + 1)
        lp = self._ensure_layer_perf()
        if lp is not None and hasattr(lp, name):
            setattr(lp, name, int(getattr(lp, name, 0)) + 1)

    def _epipe_stage_enabled(self) -> bool:
        stage = str(self._current_stage() or "").strip().lower()
        rs_active = bool(self.planner_mode == "rs_prefetch")
        if stage == "prefill":
            return bool(
                self._epipe_prefill_enable
                or self._epipe_prefill_shadow
                or self._epipe_prefill_multistage_shadow
            )
        if stage == "decode":
            if rs_active:
                return bool(self._epipe_enable and self._epipe_decode_with_rs_mode not in ("off", "disabled"))
            return bool(self._epipe_enable or self._epipe_shadow)
        return False

    def _epipe_record_stage_enabled(self) -> None:
        stage = str(self._current_stage() or "").strip().lower()
        rs_active = bool(self.planner_mode == "rs_prefetch")
        if stage == "prefill":
            self._record_epipe_counter("epipe_prefill_enabled_count", 1)
        elif stage == "decode" and rs_active:
            # More specific late/demand counters are recorded at the actual overlap/stage point.
            return
        elif stage == "decode":
            self._record_epipe_counter("epipe_decode_no_rs_enabled_count", 1)

    def _epipe_record_executor_call_by_stage(self) -> None:
        stage = str(self._current_stage() or "").strip().lower()
        rs_active = bool(self.planner_mode == "rs_prefetch")
        if stage == "prefill":
            self._record_epipe_counter("epipe_prefill_executor_calls", 1)
        elif stage == "decode" and rs_active:
            self._record_epipe_counter("epipe_decode_rs_executor_calls", 1)
        elif stage == "decode":
            self._record_epipe_counter("epipe_decode_no_rs_executor_calls", 1)

    def _epipe_record_stage_success_by_stage(self) -> None:
        stage = str(self._current_stage() or "").strip().lower()
        rs_active = bool(self.planner_mode == "rs_prefetch")
        if stage == "prefill":
            self._record_epipe_counter("epipe_prefill_stage_success_count", 1)
        elif stage == "decode" and rs_active:
            self._record_epipe_counter("epipe_decode_rs_stage_success_count", 1)
        elif stage == "decode":
            self._record_epipe_counter("epipe_decode_no_rs_stage_success_count", 1)

    def _epipe_record_prefill_no_commit_payload(self) -> None:
        if str(self._current_stage() or "").strip().lower() != "prefill":
            return
        self._record_epipe_counter("epipe_prefill_transient_no_commit_count", 1)
        self._record_epipe_counter("epipe_prefill_admission_bypass_count", 1)

    def _epipe_prefill_extra_cache_enabled(self) -> bool:
        cache = getattr(self, "_epipe_prefill_extra_cache", None)
        if cache is None or int(getattr(cache, "capacity", 0)) <= 0:
            return False
        return bool(
            getattr(self, "_epipe_prefill_enable", False)
            or getattr(self, "_epipe_prefill_shadow", False)
            or getattr(self, "_epipe_prefill_multistage_shadow", False)
        )

    def _epipe_prefill_extra_cache_store(self, eid: int, payload, nbytes: int) -> bool:
        if str(self._current_stage() or "").strip().lower() != "prefill":
            return False
        if not self._epipe_prefill_extra_cache_enabled():
            return False
        cache = getattr(self, "_epipe_prefill_extra_cache", None)
        if cache is None:
            return False
        nbytes = max(0, int(nbytes))
        max_bytes = int(float(getattr(self, "_epipe_prefill_extra_cache_max_mb", 0.0)) * 1024.0 * 1024.0)
        try:
            resident_bytes = int(cache.resident_nbytes())
        except Exception:
            resident_bytes = 0
        already_present = False
        try:
            already_present = bool(cache.has(int(eid)))
        except Exception:
            already_present = False
        if max_bytes > 0 and not already_present and resident_bytes + nbytes > max_bytes:
            self._record_epipe_counter("epipe_prefill_extra_cache_guard_skip_count", 1)
            self._record_epipe_counter("epipe_prefill_extra_cache_no_room_count", 1)
            return False
        if nbytes > 0 and not already_present:
            min_free_mb = float(getattr(self, "_epipe_prefill_extra_cache_min_free_mb", 0.0))
            if not self._epipe_memory_headroom_ok(nbytes, min_free_mb=min_free_mb):
                self._record_epipe_counter("epipe_prefill_extra_cache_guard_skip_count", 1)
                self._record_epipe_counter("epipe_prefill_extra_cache_no_room_count", 1)
                return False
        ok, updated, evicted = cache.put(int(eid), payload, nbytes=int(nbytes))
        if not ok:
            self._record_epipe_counter("epipe_prefill_extra_cache_no_room_count", 1)
            return False
        if not updated:
            self._record_epipe_counter("epipe_prefill_extra_cache_insert_count", 1)
            self._record_epipe_counter("epipe_prefill_extra_cache_h2d_bytes", int(nbytes))
        if evicted is not None:
            self._record_epipe_counter("epipe_prefill_extra_cache_evict_count", 1)
        return True

    def _epipe_prefill_extra_cache_get(self, eid: int, *, touch: bool = True):
        if not self._epipe_prefill_extra_cache_enabled():
            return None
        stage = str(self._current_stage() or "").strip().lower()
        if stage == "decode" and not bool(getattr(self, "_epipe_prefill_extra_cache_decode_read", True)):
            return None
        cache = getattr(self, "_epipe_prefill_extra_cache", None)
        if cache is None:
            return None
        try:
            payload = cache.get(int(eid)) if bool(touch) else cache.peek(int(eid))
        except Exception:
            payload = None
        if payload is None:
            return None
        self._record_cache_hit()
        self._record_epipe_counter("epipe_prefill_extra_cache_hit_count", 1)
        return payload

    def _epipe_prefill_extra_cache_contains(self, eid: int) -> bool:
        if not self._epipe_prefill_extra_cache_enabled():
            return False
        stage = str(self._current_stage() or "").strip().lower()
        if stage == "decode" and not bool(getattr(self, "_epipe_prefill_extra_cache_decode_read", True)):
            return False
        cache = getattr(self, "_epipe_prefill_extra_cache", None)
        if cache is None:
            return False
        try:
            return bool(cache.has(int(eid)))
        except Exception:
            return False

    def _epipe_prefill_extra_cache_resident_count(self) -> int:
        cache = getattr(self, "_epipe_prefill_extra_cache", None)
        if cache is None:
            return 0
        try:
            return int(cache.resident_count())
        except Exception:
            return 0

    def _epipe_prefill_extra_cache_clear(self) -> int:
        cache = getattr(self, "_epipe_prefill_extra_cache", None)
        if cache is None:
            return 0
        try:
            return int(cache.clear())
        except Exception:
            return 0

    def _epipe_record_prefill_multistage_shadow(
        self,
        ready: Sequence[Tuple[int, int, int, bool]],
        missing: Sequence[Tuple[int, int, int, bool]],
        ready_items: int,
        missing_items: int,
    ) -> None:
        if str(self._current_stage() or "").strip().lower() != "prefill":
            return
        if not bool(getattr(self, "_epipe_prefill_multistage_shadow", False)):
            return
        self._record_epipe_counter("epipe_prefill_multistage_shadow_count", 1)
        self._record_epipe_counter("epipe_prefill_multistage_ready_expert_count", len(ready))
        self._record_epipe_counter("epipe_prefill_multistage_missing_expert_count", len(missing))
        self._record_epipe_counter("epipe_prefill_multistage_ready_item_count", int(ready_items))
        self._record_epipe_counter("epipe_prefill_multistage_missing_item_count", int(missing_items))
        if not missing:
            return
        ranked_missing = sorted(
            (
                (
                    int(seg[2]) - int(seg[1]),
                    int(self._epipe_estimate_expert_nbytes(int(seg[0]))),
                )
                for seg in missing
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        max_n = min(max(1, int(getattr(self, "_epipe_prefill_multistage_max_n", 4))), len(ranked_missing))
        top = ranked_missing[:max_n]
        self._record_epipe_counter("epipe_prefill_multistage_top_missing_item_count_sum", sum(int(x[0]) for x in top))
        self._record_epipe_counter("epipe_prefill_multistage_est_expert_h2d_bytes", sum(int(x[1]) for x in top))
        self._record_epipe_counter("epipe_prefill_multistage_est_ready_compute_items", int(ready_items))
        for n, field in (
            (1, "epipe_prefill_multistage_est_hidden_h2d_items_n1"),
            (2, "epipe_prefill_multistage_est_hidden_h2d_items_n2"),
            (4, "epipe_prefill_multistage_est_hidden_h2d_items_n4"),
        ):
            if int(getattr(self, "_epipe_prefill_multistage_max_n", 4)) >= n:
                self._record_epipe_counter(field, sum(int(x[0]) for x in ranked_missing[: min(n, len(ranked_missing))]))

    def _epipe_timing_sampled(self, *parts: Any) -> bool:
        if not bool(getattr(self, "_epipe_timing", False)):
            return False
        return _stable_sample_hit(
            getattr(SparseContext, "runtime_call_id", -1),
            int(getattr(self, "layer_idx", -1)),
            int(getattr(self, "rank", -1)),
            *parts,
            rate=float(getattr(self, "_epipe_timing_sample_rate", 0.0)),
        )

    def _timing_sampled(self, env_name: str, *parts: Any) -> bool:
        return _stable_sample_hit(
            getattr(SparseContext, "runtime_call_id", -1),
            int(getattr(self, "layer_idx", -1)),
            int(getattr(self, "rank", -1)),
            *parts,
            rate=max(0.0, _env_float(env_name, 0.0)),
        )

    def _epipe_has_pending_prefetch_for_segments(self, segments: Sequence[Tuple[int, int, int, bool]]) -> bool:
        if self.prefetch_stream is None or not getattr(self, "_prefetch_events", None):
            return False
        for seg in segments:
            eid = int(seg[0])
            ev = self._prefetch_events.get(eid, None)
            if ev is None:
                continue
            try:
                if not bool(ev.query()):
                    return True
            except Exception:
                return True
        return False

    def _cache_contains_no_touch(self, eid: int) -> bool:
        eid = int(eid)
        if self._epipe_prefill_extra_cache_contains(eid):
            return True
        cache = getattr(self, "_cache", None)
        if cache is None:
            return False
        if hasattr(cache, "has"):
            try:
                return bool(cache.has(eid))
            except Exception:
                return False
        if hasattr(cache, "_store"):
            try:
                return eid in cache._store
            except Exception:
                return False
        if hasattr(cache, "entries"):
            try:
                return eid in cache.entries
            except Exception:
                return False
        if hasattr(cache, "_lru"):
            try:
                return eid in cache._lru
            except Exception:
                return False
        if hasattr(cache, "_hot") or hasattr(cache, "_transient_demand") or hasattr(cache, "_transient_prefetch"):
            try:
                return (
                    eid in getattr(cache, "_hot", {})
                    or eid in getattr(cache, "_transient_demand", {})
                    or eid in getattr(cache, "_transient_prefetch", {})
                )
            except Exception:
                return False
        return False

    def _epipe_has_cache_miss_for_segments(self, segments: Sequence[Tuple[int, int, int, bool]]) -> bool:
        for seg in segments:
            eid = int(seg[0])
            if not self._cache_contains_no_touch(eid):
                return True
        return False

    def _epipe_has_rs_residual_for_segments(self, segments: Sequence[Tuple[int, int, int, bool]], mode: str) -> bool:
        rs_mode = str(mode or "").strip().lower()
        if rs_mode == "late_only":
            return self._epipe_has_pending_prefetch_for_segments(segments)
        if rs_mode == "demand_only":
            return (
                bool(getattr(self, "_epipe_rs_handle_demand_miss", False))
                and self._epipe_rs_demand_stage_gate_can_open()
                and self._epipe_has_cache_miss_for_segments(segments)
            )
        if rs_mode in ("buffered", "residual_scan", "upfront_scan"):
            if self._epipe_has_pending_prefetch_for_segments(segments):
                return True
            return (
                bool(getattr(self, "_epipe_rs_handle_demand_miss", False))
                and self._epipe_rs_demand_stage_gate_can_open()
                and self._epipe_has_cache_miss_for_segments(segments)
            )
        return False

    def _epipe_rs_demand_stage_gate_can_open(self) -> bool:
        if not bool(getattr(self, "_epipe_rs_handle_demand_miss", False)):
            return False
        if bool(getattr(self, "_epipe_rs_force_demand_stage", False)):
            return True
        threshold = float(getattr(self, "_epipe_rs_demand_stage_threshold_ms", 0.0))
        if threshold <= 0.0:
            return False
        observed = 0.0
        if self.metrics is not None:
            sampled = float(getattr(self.metrics, "sampled_demand_h2d_exposed_decode_ms", 0.0))
            exposed = float(getattr(self.metrics, "demand_h2d_exposed_decode_ms", 0.0))
            observed = max(sampled, exposed)
        return bool(observed > threshold)

    def _epipe_rs_demand_stage_gate_open(self, ready_buffer_items: int) -> bool:
        if not bool(getattr(self, "_epipe_rs_handle_demand_miss", False)):
            return False
        if int(ready_buffer_items) < int(getattr(self, "_epipe_rs_min_buffer_items", 1)):
            self._record_epipe_counter("epipe_rs_demand_stage_disabled_by_gate_count", 1)
            return False
        if bool(getattr(self, "_epipe_rs_force_demand_stage", False)):
            self._record_epipe_counter("epipe_rs_demand_stage_force_count", 1)
            return True
        if self._epipe_rs_demand_stage_gate_can_open():
            return True
        self._record_epipe_counter("epipe_rs_demand_stage_disabled_by_gate_count", 1)
        return False


    def _load_transition_table(self) -> Dict[int, List[int]]:
        path = str(os.environ.get("SPARSE_TRANSITION_TABLE_PATH", "") or "").strip()
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return {}
        table = raw.get("table", raw) if isinstance(raw, dict) else {}
        layer_map = table.get(str(self.layer_idx), table.get(int(self.layer_idx), {})) if isinstance(table, dict) else {}
        out: Dict[int, List[int]] = {}
        if isinstance(layer_map, dict):
            for src, candidates in layer_map.items():
                if not isinstance(candidates, list):
                    continue
                out[int(src)] = [int(x) for x in candidates]
        return out

    def _cache_protect_ids(self) -> set[int]:
        protect: set[int] = set()
        ttl = int(getattr(self, "_cache_ttl_steps", 0))
        if ttl > 0 and str(SparseContext.current_phase) == "decode":
            step = int(getattr(SparseContext, "runtime_call_id", -1))
            if step >= 0:
                for eid, last_step in list(getattr(self, "_last_used_decode_step", {}).items()):
                    if step - int(last_step) <= ttl:
                        protect.add(int(eid))
        return protect

    def _cache_protect_ids_with_fallback(self, new_eid: Optional[int] = None) -> set[int]:
        protect = self._cache_protect_ids()
        if new_eid is not None:
            protect.discard(int(new_eid))
        try:
            resident = set(int(x) for x in self._cache.resident_ids())
        except Exception:
            resident = set()
        if resident and resident.issubset(protect):
            self._record_metric_counter("cache_eviction_fallback_all_protected", 1)
            return set()
        return protect

    def _record_protected_eviction_effect(self, evicted: Optional[int], protect_ids: set[int]) -> None:
        if evicted is None:
            return
        if int(evicted) in protect_ids:
            return
        ttl_protect = set(int(x) for x in protect_ids)
        try:
            resident = set(int(x) for x in self._cache.resident_ids())
        except Exception:
            resident = set()
        blocked_ttl = len((ttl_protect & resident) - {int(evicted)})
        if blocked_ttl > 0:
            self._record_metric_counter("ttl_prevented_evictions", 1)



    def _should_mark_prefill_scp_transient(self, eid: int, is_prefetch: bool = False) -> bool:
        if is_prefetch:
            return False
        if not _env_flag("SPARSE_SCP_TRANSIENT_PREFILL_CACHE", "0"):
            return False
        if self.placement_mode != "scp_ep":
            return False
        if self._current_stage() != "prefill":
            return False
        if not self._stage_uses_scp("prefill"):
            return False
        # Only non-RR-owner replicas are transient.  If this rank is also the RR
        # owner, the loaded expert is likely useful for decode RR+RS and should
        # stay normal.
        return int(eid) not in getattr(self, "_rr_local_eid_set", set())

    def _cache_put_guarded(
        self,
        eid: int,
        payload,
        *,
        pinned: bool = False,
        prefer_hot: bool = False,
        prefill_scp_transient: Optional[bool] = None,
    ) -> None:
        mark_transient = (
            self._should_mark_prefill_scp_transient(int(eid), is_prefetch=bool(pinned))
            if prefill_scp_transient is None
            else bool(prefill_scp_transient)
        )
        protect = self._cache_protect_ids_with_fallback(new_eid=int(eid))
        try:
            evicted = self._cache.put(
                int(eid),
                payload,
                pinned=bool(pinned),
                prefer_hot=bool(prefer_hot),
                protect_ids=protect,
                prefill_scp_transient=bool(mark_transient),
            )
        except TypeError:
            evicted = self._cache.put(int(eid), payload, pinned=bool(pinned), prefer_hot=bool(prefer_hot))
        if mark_transient:
            self._record_metric_counter("prefill_scp_transient_load_count", 1)
        if evicted is not None:
            try:
                was_transient = bool(
                    getattr(self._cache, "last_evicted_was_prefill_scp_transient", lambda: False)()
                )
                if was_transient:
                    self._record_metric_counter("prefill_scp_transient_evict_count", 1)
            except Exception:
                pass
        self._record_protected_eviction_effect(evicted, protect)

    def _record_comm_ms(self, dt_ms: float) -> None:
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.comm_ms += float(dt_ms)

    def _physical_moe_comm_trace_begin(self) -> Optional[Tuple[float, int, int]]:
        if (
            self.metrics is None
            or not _env_flag("SPARSE_PHYSICAL_MOE_COMM_TRACE", "0")
        ):
            return None
        lp = self._ensure_layer_perf()
        if lp is None:
            return None
        trace = self.metrics.physical_moe_comm_trace
        if trace is None:
            trace = []
            self.metrics.physical_moe_comm_trace = trace
        return (
            float(lp.comm_ms),
            int(self.metrics.comm_bytes_activation_inter_node),
            int(len(trace)),
        )

    def _physical_moe_comm_trace_end(
        self,
        state: Optional[Tuple[float, int, int]],
        *,
        batch_size: int,
        seq_len: int,
    ) -> None:
        if state is None or self.metrics is None:
            return
        lp = self._ensure_layer_perf()
        trace = self.metrics.physical_moe_comm_trace
        if lp is None or trace is None:
            return
        comm_before_ms, inter_node_before_bytes, sample_index = state
        trace.append({
            "sample_index": int(sample_index),
            "runtime_call_id": int(getattr(SparseContext, "runtime_call_id", -1)),
            "phase": str(getattr(SparseContext, "current_phase", "none") or "none"),
            "layer": int(self.layer_idx),
            "rank": int(self.rank),
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "comm_ms": max(0.0, float(lp.comm_ms) - float(comm_before_ms)),
            "cross_node_activation_bytes": max(
                0,
                int(self.metrics.comm_bytes_activation_inter_node)
                - int(inter_node_before_bytes),
            ),
        })

    def _record_expert_exec_ms(self, dt_ms: float) -> None:
        dt_ms = float(dt_ms)
        sim_ms = dt_ms
        extra_ms = 0.0
        if _sim_heterogeneity_enabled():
            scale = _sim_node_compute_scale()
            sim_ms = dt_ms / max(scale, 1e-6)
            extra_ms = max(0.0, sim_ms - dt_ms)
            if extra_ms > 0.0 and _sim_heterogeneity_sleep_enabled():
                time.sleep(extra_ms / 1000.0)
        if self.metrics is not None:
            self.metrics.sim_expert_exec_ms += float(sim_ms)
            self.metrics.sim_compute_extra_sleep_ms += float(extra_ms)
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.expert_exec_ms += float(dt_ms)
            lp.sim_expert_exec_ms += float(sim_ms)
            lp.sim_compute_extra_sleep_ms += float(extra_ms)

    def reset_runtime_state_for_new_request(
        self,
        *,
        clear_expert_cache: bool = False,
        clear_cache_metadata: bool = True,
        demote_hot: bool = False,
    ) -> None:
        """Clear per-request RS/cache-planner state.

        This fixes the observed repeat drift where `_expert_access_freq` and
        recent-use metadata keep increasing across formal repeats, making the
        RS admission gate progressively more conservative while keeping old
        resident weights.
        """
        if self._light_policy is not None:
            self._light_policy.reset()
        try:
            self._finalize_completed_prefetch_events(count_all=True)
        except Exception:
            pass
        for name in (
            "_prefetched_ids_this_group",
            "_prefetch_used_ids_this_group",
            "_transition_prefetched_ids",
            "_transition_used_ids",
        ):
            obj = getattr(self, name, None)
            if hasattr(obj, "clear"):
                obj.clear()
        for name in ("_prefetch_events", "_prefetch_expire_layer", "_expert_access_freq", "_last_used_decode_step"):
            obj = getattr(self, name, None)
            if hasattr(obj, "clear"):
                obj.clear()
        self._prefetch_current_group_start = None
        self._prefetch_current_group_end = None
        self._prefetch_current_cycle_key = None
        if clear_expert_cache:
            clear = getattr(self._cache, "clear", None)
            if callable(clear):
                clear()
        elif clear_cache_metadata:
            reset_meta = getattr(self._cache, "reset_request_metadata", None)
            if callable(reset_meta):
                reset_meta(unpin=True, demote_hot=bool(demote_hot))
            else:
                try:
                    self._cache.unpin_all()
                except Exception:
                    pass
        if clear_expert_cache or not bool(getattr(self, "_epipe_prefill_extra_cache_persist_requests", False)):
            self._epipe_prefill_extra_cache_clear()

    def _current_stage(self, seq_len: Optional[int] = None) -> str:
        phase = str(getattr(SparseContext, "current_phase", "none") or "none").strip().lower()
        if phase in ("prefill", "decode"):
            return phase
        if seq_len is not None:
            return "decode" if int(seq_len) == 1 else "prefill"
        seq = int(getattr(self, "_current_forward_seq_len", 0) or 0)
        if seq > 0:
            return "decode" if seq == 1 else "prefill"
        return "decode"

    def _stage_uses_scp(self, stage: Optional[str] = None) -> bool:
        st = self._current_stage() if stage is None else str(stage or "").strip().lower()
        if self.placement_mode != "scp_ep":
            return False
        if st == "prefill":
            return bool(self.scp_apply_prefill)
        if st == "decode":
            return bool(self.scp_apply_decode)
        return bool(self.scp_apply_decode)

    def _topology_route_cost(self, source_rank: int, target_rank: int) -> float:
        src = int(source_rank)
        dst = int(target_rank)
        matrix = getattr(self, "_scp_ep_replica_choice_cost_matrix", None)
        if matrix is not None and 0 <= src < len(matrix) and 0 <= dst < len(matrix[src]):
            return float(matrix[src][dst])
        if src == dst:
            return 0.0
        node_ids = self._comm_rank_node_ids()
        same_node = 1.0
        cross_node = 8.0
        try:
            same_node = float(os.environ.get("SPARSE_TOPOLOGY_SAME_NODE_COST", os.environ.get("SPARSE_ERS_SAME_NODE_COST", "1.0")) or 1.0)
        except Exception:
            same_node = 1.0
        try:
            cross_node = float(
                os.environ.get(
                    "SPARSE_TOPOLOGY_CROSS_NODE_COST",
                    os.environ.get(
                        "SPARSE_ERS_CROSS_NODE_COST",
                        os.environ.get("SPARSE_SIM_INTER_NODE_COST", "8.0"),
                    ),
                )
                or 8.0
            )
        except Exception:
            cross_node = 8.0
        if 0 <= src < len(node_ids) and 0 <= dst < len(node_ids) and int(node_ids[src]) == int(node_ids[dst]):
            return float(same_node)
        return float(cross_node)

    def _source_node_for_source_rank(self, source_rank: int) -> int:
        node_ids = self._comm_rank_node_ids()
        src = int(source_rank)
        if 0 <= src < len(node_ids):
            return int(node_ids[src])
        return int(src)

    def _scp_dest_rank_table_for_stage(self, stage: str) -> Optional[torch.Tensor]:
        st = str(stage or self._current_stage()).strip().lower()
        if not (self.placement_mode == "scp_ep" and self._stage_uses_scp(st)):
            return None
        if st == "prefill":
            return self.scp_prefill_dest_rank_by_node
        if st == "decode":
            return self.scp_decode_dest_rank_by_node
        return self.scp_dest_rank_by_node

    def _scp_dest_rank_table_cpu_for_stage(self, stage: str) -> Optional[List[List[int]]]:
        st = str(stage or self._current_stage()).strip().lower()
        if not (self.placement_mode == "scp_ep" and self._stage_uses_scp(st)):
            return None
        if st == "prefill":
            return self.scp_prefill_dest_rank_by_node_cpu
        if st == "decode":
            return self.scp_decode_dest_rank_by_node_cpu
        return self.scp_dest_rank_by_node_cpu

    def _resolve_table_rank_for_expert(self, *, source_rank: int, expert_id: int, stage: str) -> Optional[int]:
        timing = self.metrics is not None and _env_flag("SPARSE_SCP_LOOKUP_TIMING", "0")
        t0 = time.perf_counter() if timing else 0.0
        try:
            table_cpu = self._scp_dest_rank_table_cpu_for_stage(stage)
            eid = int(expert_id)
            if not (0 <= eid < int(self.num_experts)):
                return None
            source_node = int(self._source_node_for_source_rank(int(source_rank)))
            if table_cpu is not None:
                if not (0 <= source_node < len(table_cpu)):
                    return None
                row = table_cpu[int(source_node)]
                if not (0 <= eid < len(row)):
                    return None
                return int(row[eid])
            table = self._scp_dest_rank_table_for_stage(stage)
            if table is None:
                return None
            if not (0 <= source_node < int(table.shape[0])):
                return None
            return int(table[int(source_node), eid].detach().to("cpu").item())
        finally:
            if timing and self.metrics is not None:
                self.metrics.scp_lookup_calls += 1
                self.metrics.scp_lookup_overhead_ms += (time.perf_counter() - t0) * 1000.0

    def _record_scp_table_route_stats(self, flat_i: torch.Tensor, dest_rank: torch.Tensor, stage: str) -> None:
        if self.metrics is None:
            return
        if self.placement_mode != "scp_ep" or not self._stage_uses_scp(stage):
            return
        if flat_i.numel() == 0 or dest_rank.numel() == 0:
            return
        try:
            eids = flat_i.to(device=self.device, dtype=torch.long).view(-1)
            dst = dest_rank.to(device=self.device, dtype=torch.long).view(-1)
            if int(eids.numel()) <= 0:
                return
            rr = self.rr_expert_to_rank[eids].to(dtype=torch.long)
            rank_to_node = self._rank_to_node_tensor()
            safe_dst = dst.clamp(0, int(self.comm_world) - 1)
            safe_rr = rr.clamp(0, int(self.comm_world) - 1)
            dst_node = rank_to_node[safe_dst]
            rr_node = rank_to_node[safe_rr]
            source_node = int(self._source_node_for_source_rank(int(self.comm_rank)))
            source_node_t = torch.tensor(source_node, device=self.device, dtype=torch.long)
            local_replica = (dst != rr) & (dst_node == source_node_t)
            rr_fallback = dst == rr
            cross_node = dst_node != source_node_t
            same_node = ~cross_node
            self.metrics.scp_table_local_replica_route_count += int(local_replica.sum().item())
            self.metrics.scp_table_rr_fallback_count += int(rr_fallback.sum().item())
            self.metrics.scp_table_cross_node_route_count += int(cross_node.sum().item())
            self.metrics.scp_table_same_node_route_count += int(same_node.sum().item())

            if self._scp_ep_replica_choice_cost_matrix is not None:
                cost = torch.tensor(self._scp_ep_replica_choice_cost_matrix, device=self.device, dtype=torch.float64)
                src = int(self.comm_rank)
                self.metrics.scp_route_cost_before_sum += float(cost[src, safe_rr].sum().item())
                self.metrics.scp_route_cost_after_sum += float(cost[src, safe_dst].sum().item())
            else:
                # Fallback cost is a coarse same-node/cross-node unit diagnostic.
                self.metrics.scp_route_cost_before_sum += float((rr_node != source_node_t).to(torch.float64).sum().item())
                self.metrics.scp_route_cost_after_sum += float(cross_node.to(torch.float64).sum().item())
        except Exception:
            return

    def _local_prefetch_ready(self, expert_id: int) -> bool:
        eid = int(expert_id)
        if not self._cache.has(eid):
            return False
        ev = getattr(self, "_prefetch_events", {}).get(eid, None)
        if ev is None:
            return True
        try:
            return bool(ev.query())
        except Exception:
            return False

    def _decode_cache_aware_remote_snapshot_enabled(self) -> bool:
        if not bool(getattr(self, "decode_cache_aware_scp", False)):
            return False
        if str(getattr(SparseContext, "current_phase", "none") or "none").strip().lower() != "decode":
            return False
        requested = _env_flag("SPARSE_DECODE_CACHE_AWARE_SCP_REMOTE_SNAPSHOT", "0")
        if not requested:
            return False
        # The current cache-aware route resolver may call this from rank-local,
        # data-dependent branches.  A distributed collective here can therefore
        # deadlock when ranks do not enter the branch in the same order/count.
        # Keep the safe local-only behavior unless an explicit developer escape
        # hatch is enabled for debugging a synchronized implementation.
        if _env_flag("SPARSE_DECODE_CACHE_AWARE_SCP_REMOTE_SNAPSHOT_UNSAFE", "0"):
            return True
        if not bool(getattr(self, "_warned_decode_cache_remote_snapshot_disabled", False)):
            try:
                print(
                    "[SCP-DECODE] remote cache snapshot disabled: "
                    "current resolver path is not collective-safe; using local cache state only.",
                    flush=True,
                )
            except Exception:
                pass
            self._warned_decode_cache_remote_snapshot_disabled = True
        return False

    def _decode_cache_aware_resident_sets_snapshot(self) -> List[set[int]]:
        key = (
            str(getattr(SparseContext, "current_phase", "none") or "none"),
            int(getattr(SparseContext, "runtime_call_id", -1)),
            int(self.layer_idx),
            int(self.comm_rank),
            int(self.comm_world),
        )
        cached_key = getattr(self, "_decode_cache_aware_resident_sets_key", None)
        cached = getattr(self, "_decode_cache_aware_resident_sets", None)
        if cached_key == key and cached is not None:
            return cached
        try:
            resident_sets = self._gather_comm_group_cache_resident_sets()
        except Exception:
            resident_sets = [set() for _ in range(max(int(self.comm_world), 1))]
        self._decode_cache_aware_resident_sets_key = key
        self._decode_cache_aware_resident_sets = resident_sets
        return resident_sets

    def _resolve_replica_ready_for_rank(self, expert_id: int, rank: int) -> Tuple[bool, bool, bool]:
        eid = int(expert_id)
        r = int(rank)
        if r == int(self.comm_rank):
            if not self._cache.has(eid):
                return False, False, False
            ev = getattr(self, "_prefetch_events", {}).get(eid, None)
            if ev is None:
                return True, True, False
            try:
                done = bool(ev.query())
            except Exception:
                done = False
            return done, False, bool(done)
        if self._decode_cache_aware_remote_snapshot_enabled():
            node_ids = self._comm_rank_node_ids()
            local_node = self._source_node_for_source_rank(int(self.comm_rank))
            remote_node = int(node_ids[r]) if 0 <= r < len(node_ids) else int(r)
            same_node_only = _env_flag("SPARSE_DECODE_CACHE_AWARE_SCP_REMOTE_SAME_NODE_ONLY", "1")
            if (not same_node_only) or int(remote_node) == int(local_node):
                resident_sets = self._decode_cache_aware_resident_sets_snapshot()
                if 0 <= r < len(resident_sets) and eid in resident_sets[r]:
                    return True, True, False
        # Default mode intentionally avoids remote cache-state exchange.  Remote
        # replicas are eligible only when they are the always-valid RR fallback.
        return False, False, False

    def _light_ready_target(self, eid: int, source: int):
        policy = self._light_policy
        if (policy is None or not policy.route or source != self.comm_rank
                or eid not in policy.local_candidates):
            return None
        ready, cached, prefetched = self._resolve_replica_ready_for_rank(eid, self.comm_rank)
        if not ready:
            return None
        return ResolveResult(target_rank=self.comm_rank, mode="light_local_ready",
                             rr_owner=self.rr_expert_to_rank_cpu[eid], used_scp_replica=True,
                             target_ready=True, route_cost=self._topology_route_cost(source, self.comm_rank),
                             cache_ready=cached, prefetch_ready=prefetched)

    def _light_destinations(self, flat_i):
        policy = self._light_policy
        start = time.perf_counter()
        ready = tuple(sorted(eid for eid in self._cache.resident_ids()
                             if eid in policy.local_candidates and self._local_prefetch_ready(eid)))
        if ready != policy.route_key:
            table = self.rr_expert_to_rank
            if ready:
                table = table.clone()
                table[torch.tensor(ready, device=self.device)] = self.comm_rank
            policy.route_key, policy.route_table = ready, table
            policy.stats['route_table_rebuilds'] += 1
        result = policy.route_table[flat_i]
        policy.stats['route_calls'] += 1
        policy.stats['route_host_ms'] += (time.perf_counter() - start) * 1000
        return result

    def resolve_execution_rank(
        self,
        layer_idx: int,
        source_rank: int,
        expert_id: int,
        stage: str,
        *,
        allow_decode_scp: bool = True,
        require_no_extra_h2d: bool = True,
        count_metrics: bool = True,
        reason: str = "",
    ) -> ResolveResult:
        eid = int(expert_id)
        src = int(source_rank)
        rr_owner = int(self.rr_expert_to_rank_cpu[eid]) if 0 <= eid < len(self.rr_expert_to_rank_cpu) else int(eid % max(int(self.comm_world), 1))
        st = str(stage or self._current_stage()).strip().lower()
        before_cost = float(self._topology_route_cost(src, rr_owner))
        if st == "decode":
            local = self._light_ready_target(eid, src)
            if local is not None:
                return local
        if self.placement_mode != "scp_ep" or not (0 <= eid < int(self.num_experts)):
            return ResolveResult(
                target_rank=rr_owner,
                mode="rr_owner",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=False,
                target_ready=True,
                route_cost=before_cost,
            )

        use_decode_cache_aware = st == "decode" and bool(getattr(self, "decode_cache_aware_scp", False))
        table_target = self._resolve_table_rank_for_expert(source_rank=src, expert_id=eid, stage=st)
        if table_target is not None and not use_decode_cache_aware:
            return ResolveResult(
                target_rank=int(table_target),
                mode=f"{st}_scp_table_choice",
                rr_owner=rr_owner,
                used_scp_replica=(int(table_target) != rr_owner),
                fallback_to_rr=False,
                target_ready=True,
                route_cost=float(self._topology_route_cost(src, int(table_target))),
            )

        if st == "prefill":
            if self._stage_uses_scp("prefill"):
                owners = self.scp_prefill_expert_to_rank_cpu or self.scp_expert_to_rank_cpu or self.expert_to_rank_cpu
                target = int(owners[eid]) if 0 <= eid < len(owners) else rr_owner
                return ResolveResult(
                    target_rank=target,
                    mode="prefill_scp_choice",
                    rr_owner=rr_owner,
                    used_scp_replica=(target != rr_owner),
                    fallback_to_rr=False,
                    target_ready=True,
                    route_cost=float(self._topology_route_cost(src, target)),
                )
            return ResolveResult(
                target_rank=rr_owner,
                mode="rr_owner",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=False,
                target_ready=True,
                route_cost=before_cost,
            )

        if st != "decode" or not self.scp_apply_decode or not allow_decode_scp:
            return ResolveResult(
                target_rank=rr_owner,
                mode="rr_owner",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=False,
                target_ready=True,
                route_cost=before_cost,
            )

        if not self.decode_cache_aware_scp:
            owners = self.scp_decode_expert_to_rank_cpu or self.scp_expert_to_rank_cpu or self.expert_to_rank_cpu
            target = int(owners[eid]) if 0 <= eid < len(owners) else rr_owner
            return ResolveResult(
                target_rank=target,
                mode="decode_scp_static_choice",
                rr_owner=rr_owner,
                used_scp_replica=(target != rr_owner),
                fallback_to_rr=False,
                target_ready=True,
                route_cost=float(self._topology_route_cost(src, target)),
            )

        owners = self.scp_decode_expert_to_rank_cpu or self.scp_expert_to_rank_cpu or self.expert_to_rank_cpu
        intended = int(owners[eid]) if 0 <= eid < len(owners) else rr_owner
        intended_cost = float(self._topology_route_cost(src, intended))
        if int(intended) != int(rr_owner) and intended_cost < before_cost:
            ready, cache_ready, prefetch_ready = self._resolve_replica_ready_for_rank(eid, int(intended))
            if ready:
                if bool(count_metrics) and self.metrics is not None:
                    self.metrics.resolver_decode_total_queries += 1
                    self.metrics.resolver_decode_route_cost_before += float(before_cost)
                    self.metrics.resolver_decode_route_cost_after += float(intended_cost)
                    self.metrics.resolver_decode_scp_replica_selected_count += 1
                    if cache_ready:
                        self.metrics.resolver_decode_replica_cache_ready_count += 1
                    if prefetch_ready:
                        self.metrics.resolver_decode_replica_prefetch_ready_count += 1
                return ResolveResult(
                    target_rank=int(intended),
                    mode="decode_scp_choice_ready",
                    rr_owner=rr_owner,
                    used_scp_replica=True,
                    fallback_to_rr=False,
                    target_ready=True,
                    route_cost=float(intended_cost),
                    cache_ready=bool(cache_ready),
                    prefetch_ready=bool(prefetch_ready),
                )
            if bool(count_metrics) and self.metrics is not None:
                self.metrics.resolver_decode_total_queries += 1
                self.metrics.resolver_decode_route_cost_before += float(before_cost)
                self.metrics.resolver_decode_route_cost_after += float(before_cost)
                self.metrics.resolver_decode_replica_reject_not_ready_count += 1
                self.metrics.resolver_decode_rr_fallback_count += 1
            return ResolveResult(
                target_rank=rr_owner,
                mode="decode_scp_choice_not_ready_rr_fallback",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=True,
                reject_reason="not_ready",
                target_ready=True,
                route_cost=before_cost,
                reject_not_ready_count=1,
            )

        replicas_raw = getattr(self, "_scp_ep_replicas", None)
        replicas = []
        if replicas_raw is not None and 0 <= eid < len(replicas_raw):
            replicas = [int(r) for r in replicas_raw[eid]]
        if rr_owner not in replicas:
            replicas.append(rr_owner)
        replicas = sorted(set(int(r) for r in replicas if 0 <= int(r) < int(self.comm_world)))

        eligible: List[Tuple[float, int, str, bool, bool]] = [(before_cost, rr_owner, "decode_scp_rr_fallback", True, False)]
        reject_not_ready = 0
        reject_no_gain = 0
        for r in replicas:
            if int(r) == int(rr_owner):
                continue
            candidate_cost = float(self._topology_route_cost(src, int(r)))
            if candidate_cost >= before_cost:
                reject_no_gain += 1
                continue
            ready = True
            cache_ready = False
            prefetch_ready = False
            if bool(require_no_extra_h2d):
                ready, cache_ready, prefetch_ready = self._resolve_replica_ready_for_rank(eid, int(r))
            if not ready:
                reject_not_ready += 1
                continue
            mode = "decode_scp_prefetch_ready_replica" if prefetch_ready else "decode_scp_cache_ready_replica"
            eligible.append((candidate_cost, int(r), mode, cache_ready, prefetch_ready))

        eligible.sort(key=lambda x: (float(x[0]), int(x[1])))
        best_cost, best_rank, best_mode, cache_ready, prefetch_ready = eligible[0]
        selected_replica = int(best_rank) != int(rr_owner)

        if bool(count_metrics) and self.metrics is not None:
            self.metrics.resolver_decode_total_queries += 1
            self.metrics.resolver_decode_route_cost_before += float(before_cost)
            self.metrics.resolver_decode_route_cost_after += float(best_cost)
            self.metrics.resolver_decode_replica_reject_not_ready_count += int(reject_not_ready)
            self.metrics.resolver_decode_replica_reject_no_comm_gain_count += int(reject_no_gain)
            if selected_replica:
                self.metrics.resolver_decode_scp_replica_selected_count += 1
                if cache_ready:
                    self.metrics.resolver_decode_replica_cache_ready_count += 1
                if prefetch_ready:
                    self.metrics.resolver_decode_replica_prefetch_ready_count += 1
            else:
                self.metrics.resolver_decode_rr_fallback_count += 1
                if best_mode == "rr_owner":
                    self.metrics.resolver_decode_rr_owner_count += 1

        reject_reason = ""
        if not selected_replica:
            reject_reason = "not_ready" if reject_not_ready > 0 else ("no_comm_gain" if reject_no_gain > 0 else "rr_best")
        return ResolveResult(
            target_rank=int(best_rank),
            mode=str(best_mode),
            rr_owner=rr_owner,
            used_scp_replica=bool(selected_replica),
            fallback_to_rr=not bool(selected_replica),
            reject_reason=reject_reason,
            target_ready=True,
            route_cost=float(best_cost),
            cache_ready=bool(cache_ready),
            prefetch_ready=bool(prefetch_ready),
            reject_not_ready_count=int(reject_not_ready),
            reject_no_comm_gain_count=int(reject_no_gain),
        )

    def resolve_prefetch_target_rank(
        self,
        layer_idx: int,
        source_rank: int,
        expert_id: int,
        stage: str,
        *,
        count_metrics: bool = False,
        reason: str = "",
    ) -> ResolveResult:
        """Resolve the rank that should prepare an expert for future execution.

        Execution resolution is conservative and may require a replica to already
        be cache-ready.  Prefetch planning must be allowed to pick the intended
        decode replica before it is ready; otherwise RS can never make SCP replicas
        ready in the first place.
        """
        eid = int(expert_id)
        src = int(source_rank)
        rr_owner = int(self.rr_expert_to_rank_cpu[eid]) if 0 <= eid < len(self.rr_expert_to_rank_cpu) else int(eid % max(int(self.comm_world), 1))
        st = str(stage or self._current_stage()).strip().lower()
        before_cost = float(self._topology_route_cost(src, rr_owner))
        if st == "decode":
            local = self._light_ready_target(eid, src)
            if local is not None:
                return local
        if self.placement_mode != "scp_ep" or not (0 <= eid < int(self.num_experts)):
            return ResolveResult(
                target_rank=rr_owner,
                mode="rr_owner",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=False,
                target_ready=True,
                route_cost=before_cost,
            )

        table_target = self._resolve_table_rank_for_expert(source_rank=src, expert_id=eid, stage=st)
        if table_target is not None:
            return ResolveResult(
                target_rank=int(table_target),
                mode=f"{st}_scp_table_prefetch_target",
                rr_owner=rr_owner,
                used_scp_replica=(int(table_target) != rr_owner),
                fallback_to_rr=False,
                target_ready=False,
                route_cost=float(self._topology_route_cost(src, int(table_target))),
            )

        if st == "prefill":
            owners = self.scp_prefill_expert_to_rank_cpu or self.scp_expert_to_rank_cpu or self.expert_to_rank_cpu
            if self._stage_uses_scp("prefill"):
                target = int(owners[eid]) if 0 <= eid < len(owners) else rr_owner
                return ResolveResult(
                    target_rank=target,
                    mode="prefill_scp_prefetch_target",
                    rr_owner=rr_owner,
                    used_scp_replica=(target != rr_owner),
                    fallback_to_rr=False,
                    target_ready=False,
                    route_cost=float(self._topology_route_cost(src, target)),
                )
            return ResolveResult(
                target_rank=rr_owner,
                mode="rr_owner",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=False,
                target_ready=True,
                route_cost=before_cost,
            )

        if st != "decode" or not self.scp_apply_decode:
            return ResolveResult(
                target_rank=rr_owner,
                mode="rr_owner",
                rr_owner=rr_owner,
                used_scp_replica=False,
                fallback_to_rr=False,
                target_ready=True,
                route_cost=before_cost,
            )

        owners = self.scp_decode_expert_to_rank_cpu or self.scp_expert_to_rank_cpu or self.expert_to_rank_cpu
        target = int(owners[eid]) if 0 <= eid < len(owners) else rr_owner
        target_cost = float(self._topology_route_cost(src, target))
        if target_cost < before_cost or target == int(self.comm_rank):
            return ResolveResult(
                target_rank=target,
                mode="decode_scp_prefetch_target",
                rr_owner=rr_owner,
                used_scp_replica=(target != rr_owner),
                fallback_to_rr=False,
                target_ready=False,
                route_cost=target_cost,
            )
        return ResolveResult(
            target_rank=rr_owner,
            mode="decode_scp_prefetch_rr_fallback",
            rr_owner=rr_owner,
            used_scp_replica=False,
            fallback_to_rr=True,
            target_ready=True,
            route_cost=before_cost,
        )

    def _resolve_rank_for_expert(self, expert_id: int, stage: Optional[str] = None, *, reason: str = "") -> int:
        st = self._current_stage() if stage is None else str(stage or "").strip().lower()
        return int(
            self.resolve_execution_rank(
                layer_idx=int(self.layer_idx),
                source_rank=int(self.comm_rank),
                expert_id=int(expert_id),
                stage=st,
                count_metrics=False,
                reason=reason,
            ).target_rank
        )

    def _resolve_dest_for_flat_experts(
        self,
        flat_i: torch.Tensor,
        stage: Optional[str] = None,
        *,
        reason: str = "",
        count_metrics: bool = True,
    ) -> torch.Tensor:
        st = self._current_stage() if stage is None else str(stage or "").strip().lower()
        if flat_i.numel() == 0:
            return torch.empty_like(flat_i, dtype=torch.long, device=self.device)
        if st == "decode" and self._light_policy is not None and self._light_policy.route:
            return self._light_destinations(flat_i)
        use_decode_cache_aware = st == "decode" and bool(getattr(self, "decode_cache_aware_scp", False))
        if self.placement_mode == "scp_ep" and self._stage_uses_scp(st) and not use_decode_cache_aware:
            table = self._scp_dest_rank_table_for_stage(st)
            if table is not None:
                source_node = int(self._source_node_for_source_rank(int(self.comm_rank)))
                if 0 <= source_node < int(table.shape[0]):
                    dest = table[int(source_node)].to(device=self.device)[flat_i]
                    if bool(count_metrics):
                        self._record_scp_table_route_stats(flat_i, dest, st)
                        self._record_scp_phase_route_diag(flat_i, dest, st)
                    return dest
        if not (
            self.placement_mode == "scp_ep"
            and st == "decode"
            and bool(getattr(self, "decode_cache_aware_scp", False))
            and bool(self._stage_uses_scp("decode"))
        ):
            dest = self._expert_to_rank_for_stage_tensor(st)[flat_i]
            if bool(count_metrics):
                self._record_scp_phase_route_diag(flat_i, dest, st)
            return dest
        # Cache-aware SCP only depends on the current layer, source rank, expert
        # id, and local cache state.  Decode batches contain many repeated expert
        # ids, so resolving once per unique expert avoids tens of thousands of
        # Python resolver calls when only a handful of replicas are actually used.
        uniq, counts = torch.unique(
            flat_i.detach().to(device="cpu", dtype=torch.long).view(-1),
            sorted=False,
            return_counts=True,
        )
        dest_map = self.rr_expert_to_rank.clone()
        for eid_obj, cnt_obj in zip(uniq.tolist(), counts.tolist()):
            eid = int(eid_obj)
            cnt = int(cnt_obj)
            result = self.resolve_execution_rank(
                layer_idx=int(self.layer_idx),
                source_rank=int(self.comm_rank),
                expert_id=eid,
                stage=st,
                count_metrics=False,
                reason=reason,
            )
            if 0 <= eid < int(dest_map.numel()):
                dest_map[eid] = int(result.target_rank)
            if bool(count_metrics) and self.metrics is not None:
                before_cost = float(self._topology_route_cost(int(self.comm_rank), int(result.rr_owner)))
                self.metrics.resolver_decode_total_queries += int(cnt)
                self.metrics.resolver_decode_route_cost_before += float(before_cost) * float(cnt)
                self.metrics.resolver_decode_route_cost_after += float(result.route_cost) * float(cnt)
                self.metrics.resolver_decode_replica_reject_not_ready_count += int(result.reject_not_ready_count) * int(cnt)
                self.metrics.resolver_decode_replica_reject_no_comm_gain_count += int(result.reject_no_comm_gain_count) * int(cnt)
                if bool(result.used_scp_replica):
                    self.metrics.resolver_decode_scp_replica_selected_count += int(cnt)
                    if bool(result.cache_ready):
                        self.metrics.resolver_decode_replica_cache_ready_count += int(cnt)
                    if bool(result.prefetch_ready):
                        self.metrics.resolver_decode_replica_prefetch_ready_count += int(cnt)
                else:
                    self.metrics.resolver_decode_rr_fallback_count += int(cnt)
                    if str(result.mode) == "rr_owner":
                        self.metrics.resolver_decode_rr_owner_count += int(cnt)
        dest = dest_map[flat_i]
        if bool(count_metrics):
            self._record_scp_phase_route_diag(flat_i, dest, st)
        return dest

    def _record_scp_phase_route_diag(self, flat_i: torch.Tensor, dest_rank: torch.Tensor, stage: str) -> None:
        if self.metrics is None:
            return
        if not _env_flag("SPARSE_SCP_PHASE_ROUTE_DIAG", "0"):
            return
        if self.placement_mode != "scp_ep":
            return
        st = str(stage or self._current_stage()).strip().lower()
        if st not in ("prefill", "decode"):
            return
        if not self._stage_uses_scp(st):
            return
        if flat_i.numel() == 0 or dest_rank.numel() == 0:
            return
        try:
            eids = flat_i.detach().to("cpu", dtype=torch.long).view(-1).tolist()
            dsts = dest_rank.detach().to("cpu", dtype=torch.long).view(-1).tolist()
        except Exception:
            return
        world = int(self.comm_world)
        if world <= 0:
            return
        node_ids = self._comm_rank_node_ids()
        src = int(self.comm_rank)
        src_node = int(node_ids[src]) if 0 <= src < len(node_ids) else src
        rr_counts = [0 for _ in range(world)]
        dst_counts = [0 for _ in range(world)]
        replica_count = 0
        for eid, dst in zip(eids, dsts):
            if not (0 <= int(eid) < int(self.num_experts)):
                continue
            rr_owner = int(self.rr_expert_to_rank_cpu[int(eid)]) if 0 <= int(eid) < len(self.rr_expert_to_rank_cpu) else int(eid) % world
            if 0 <= rr_owner < world:
                rr_counts[int(rr_owner)] += 1
            if 0 <= int(dst) < world:
                dst_counts[int(dst)] += 1
            if int(dst) != int(rr_owner):
                replica_count += 1

        def _cross_items_events(counts: List[int]) -> Tuple[int, int]:
            items = 0
            events = 0
            for r, cnt in enumerate(counts):
                if int(cnt) <= 0:
                    continue
                dst_node = int(node_ids[int(r)]) if 0 <= int(r) < len(node_ids) else int(r)
                if int(r) != src and int(dst_node) != int(src_node):
                    items += int(cnt)
                    events += 1
            return int(items), int(events)

        before_items, before_events = _cross_items_events(rr_counts)
        after_items, after_events = _cross_items_events(dst_counts)
        before_cost = 0.0
        after_cost = 0.0
        for r, cnt in enumerate(rr_counts):
            if int(cnt) > 0:
                before_cost += float(cnt) * float(self._topology_route_cost(src, int(r)))
        for r, cnt in enumerate(dst_counts):
            if int(cnt) > 0:
                after_cost += float(cnt) * float(self._topology_route_cost(src, int(r)))

        prefix = "scp_prefill_route_diag" if st == "prefill" else "scp_decode_route_diag"
        setattr(self.metrics, f"{prefix}_items", int(getattr(self.metrics, f"{prefix}_items", 0)) + int(len(eids)))
        setattr(self.metrics, f"{prefix}_replica_items", int(getattr(self.metrics, f"{prefix}_replica_items", 0)) + int(replica_count))
        setattr(self.metrics, f"{prefix}_cross_items_before", int(getattr(self.metrics, f"{prefix}_cross_items_before", 0)) + int(before_items))
        setattr(self.metrics, f"{prefix}_cross_items_after", int(getattr(self.metrics, f"{prefix}_cross_items_after", 0)) + int(after_items))
        setattr(self.metrics, f"{prefix}_cross_events_before", int(getattr(self.metrics, f"{prefix}_cross_events_before", 0)) + int(before_events))
        setattr(self.metrics, f"{prefix}_cross_events_after", int(getattr(self.metrics, f"{prefix}_cross_events_after", 0)) + int(after_events))
        setattr(self.metrics, f"{prefix}_cost_before", float(getattr(self.metrics, f"{prefix}_cost_before", 0.0)) + float(before_cost))
        setattr(self.metrics, f"{prefix}_cost_after", float(getattr(self.metrics, f"{prefix}_cost_after", 0.0)) + float(after_cost))

    def _expert_to_rank_for_stage_tensor(self, stage: Optional[str] = None) -> torch.Tensor:
        st = self._current_stage() if stage is None else str(stage or "").strip().lower()
        if self._stage_uses_scp(st):
            if st == "prefill" and self.scp_prefill_expert_to_rank is not None:
                return self.scp_prefill_expert_to_rank
            if st == "decode" and self.scp_decode_expert_to_rank is not None:
                return self.scp_decode_expert_to_rank
            if self.scp_expert_to_rank is not None:
                return self.scp_expert_to_rank
            return self.expert_to_rank
        if self.placement_mode == "scp_ep":
            return self.rr_expert_to_rank
        return self.expert_to_rank

    def _expert_to_rank_for_stage_cpu(self, stage: Optional[str] = None) -> List[int]:
        st = self._current_stage() if stage is None else str(stage or "").strip().lower()
        if self._stage_uses_scp(st):
            if st == "prefill" and self.scp_prefill_expert_to_rank_cpu is not None:
                return self.scp_prefill_expert_to_rank_cpu
            if st == "decode" and self.scp_decode_expert_to_rank_cpu is not None:
                return self.scp_decode_expert_to_rank_cpu
            if self.scp_expert_to_rank_cpu is not None:
                return self.scp_expert_to_rank_cpu
            return self.expert_to_rank_cpu
        if self.placement_mode == "scp_ep":
            return self.rr_expert_to_rank_cpu
        return self.expert_to_rank_cpu

    def _stage_prefetch_eid_set(self, stage: Optional[str] = None) -> set[int]:
        st = self._current_stage() if stage is None else str(stage or "").strip().lower()
        if self.placement_mode == "scp_ep" and self._stage_uses_scp(st):
            if st == "decode" and bool(getattr(self, "decode_cache_aware_scp", False)):
                out = set(getattr(self, "_scp_ep_decode_receive_eid_set", set()) or getattr(self, "_scp_ep_receive_eid_set", set()))
                out.update(int(eid) for eid in getattr(self, "_rr_local_eid_set", set()))
                return out
            if st == "prefill":
                return set(getattr(self, "_scp_ep_prefill_receive_eid_set", set()) or getattr(self, "_scp_ep_receive_eid_set", set()))
            if st == "decode":
                return set(getattr(self, "_scp_ep_decode_receive_eid_set", set()) or getattr(self, "_scp_ep_receive_eid_set", set()))
            return set(getattr(self, "_scp_ep_receive_eid_set", set()))
        if self.placement_mode == "scp_ep":
            return set(getattr(self, "_rr_local_eid_set", set()))
        return set(getattr(self, "_local_eid_set", set()))

    def _can_execute_expert_stage(self, expert_id: int, stage: Optional[str] = None) -> bool:
        eid = int(expert_id)
        if not (0 <= eid < int(self.num_experts)):
            return False
        if self.placement_mode == "scp_ep" and self._stage_uses_scp(stage):
            if str(stage or self._current_stage()).strip().lower() == "decode" and bool(getattr(self, "decode_cache_aware_scp", False)):
                return self._resolve_rank_for_expert(eid, stage="decode", reason="can_execute") == int(self.comm_rank)
            reps = getattr(self, "_scp_ep_replicas", None)
            if reps is None:
                return False
            return int(self.comm_rank) in set(int(x) for x in reps[eid])
        if self.placement_mode == "scp_ep":
            return int(self.rr_expert_to_rank_cpu[eid]) == int(self.comm_rank)
        return eid in self._local_eid_set

    def _record_grouping_overhead_ms(self, dt_ms: float) -> None:
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.grouping_overhead_ms += float(dt_ms)
            lp.runtime_overhead_ms += float(dt_ms)
        cp = _hclm_cp_profiler()
        step_id = int(getattr(SparseContext, "runtime_call_id", -1))
        if cp.should_sample(step_id):
            now = _hclm_cp_now_ns()
            cp.record_component(
                step_id,
                int(self.layer_idx),
                "control_overhead",
                now - int(max(float(dt_ms), 0.0) * 1_000_000.0),
                now,
            )

    def _record_node_grouped_ms(self, kind: str, dt_ms: float) -> None:
        dt_ms = max(float(dt_ms), 0.0)
        if dt_ms <= 0.0:
            return
        attr = f"node_grouped_{kind}_ms"
        if self.metrics is not None and hasattr(self.metrics, attr):
            setattr(self.metrics, attr, float(getattr(self.metrics, attr, 0.0)) + dt_ms)
            self.metrics.node_grouped_total_overhead_ms += dt_ms
        lp = self._ensure_layer_perf()
        if lp is not None and hasattr(lp, attr):
            setattr(lp, attr, float(getattr(lp, attr, 0.0)) + dt_ms)
            lp.node_grouped_total_overhead_ms += dt_ms

    def _record_node_grouped_value(self, name: str, value: float) -> None:
        attr = f"node_grouped_{name}"
        if self.metrics is not None and hasattr(self.metrics, attr):
            cur = getattr(self.metrics, attr, 0)
            setattr(self.metrics, attr, cur + value)
        lp = self._ensure_layer_perf()
        if lp is not None and hasattr(lp, attr):
            cur = getattr(lp, attr, 0)
            setattr(lp, attr, cur + value)

    def _record_node_grouped_allocation(self, *tensors: torch.Tensor, dt_ms: float = 0.0) -> None:
        valid = [t for t in tensors if isinstance(t, torch.Tensor)]
        if not valid and float(dt_ms) <= 0.0:
            return
        self._record_node_grouped_value("allocation_count", int(len(valid)))
        self._record_node_grouped_value("allocation_bytes", int(sum(tensor_nbytes(t) for t in valid)))
        if float(dt_ms) > 0.0:
            self._record_node_grouped_value("allocation_ms", float(dt_ms))

    def _node_grouped_empty(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        label: str,
        workspace: bool = False,
        zero: bool = False,
    ) -> torch.Tensor:
        shape_t = tuple(int(x) for x in shape)
        if any(int(x) < 0 for x in shape_t):
            raise RuntimeError(f"[rank{self.rank}] invalid node-grouped buffer shape={shape_t} label={label}")
        if not bool(workspace):
            out = torch.empty(shape_t, dtype=dtype, device=self.device)
            return out.zero_() if bool(zero) else out
        key = (str(label), str(self.device), str(dtype), shape_t)
        cached = self._node_grouped_workspace.get(key)
        if cached is None or cached.device != self.device or cached.dtype != dtype or tuple(cached.shape) != shape_t:
            cached = torch.empty(shape_t, dtype=dtype, device=self.device)
            self._node_grouped_workspace[key] = cached
        else:
            self._record_node_grouped_value("workspace_reuse_hit", 1)
        return cached.zero_() if bool(zero) else cached

    def _record_node_grouped_decision(self, *, enabled: bool, estimated_saving_ms: float) -> None:
        estimated_saving_ms = float(estimated_saving_ms)
        if self.metrics is not None:
            if enabled:
                self.metrics.node_grouped_enabled_layers += 1
            else:
                self.metrics.node_grouped_fallback_layers += 1
            self.metrics.node_grouped_estimated_saving_ms += estimated_saving_ms
        lp = self._ensure_layer_perf()
        if lp is not None:
            if enabled:
                lp.node_grouped_enabled_count += 1
            else:
                lp.node_grouped_fallback_count += 1
            lp.node_grouped_estimated_saving_ms += estimated_saving_ms

    def _record_rs_mlp_layer_ms(self, dt_ms: float) -> None:
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.rs_mlp_ms += float(dt_ms)
            lp.runtime_overhead_ms += float(dt_ms)

    def _all_to_all_single_timed(self, output: torch.Tensor, input: torch.Tensor, *args, **kwargs) -> None:
        if self.metrics is None or not _env_flag("SPARSE_BOTTLENECK_TIMING", "0"):
            dist.all_to_all_single(output, input, *args, **kwargs)
            return
        if not torch.cuda.is_available() or self.device.type != "cuda":
            t0 = time.perf_counter()
            dist.all_to_all_single(output, input, *args, **kwargs)
            self._record_comm_ms((time.perf_counter() - t0) * 1000.0)
            return
        start = torch.cuda.Event(enable_timing=True, blocking=False)
        end = torch.cuda.Event(enable_timing=True, blocking=False)
        start.record(torch.cuda.current_stream(self.device))
        dist.all_to_all_single(output, input, *args, **kwargs)
        end.record(torch.cuda.current_stream(self.device))
        end.synchronize()
        self._record_comm_ms(float(start.elapsed_time(end)))

    def _exchange_dispatch_counts(
        self,
        send_counts: torch.Tensor,
        *,
        group: Optional[Any] = None,
        group_rank: Optional[int] = None,
        label: str = "dispatch",
    ) -> torch.Tensor:
        """
        Exchange per-destination packet counts and return this rank's receive
        counts by source rank.

        The data-plane A2A still uses all_to_all_single. For these tiny metadata
        count vectors, all_gather is more robust than a second A2A: it gives us
        the full src x dst count matrix, makes validation explicit, and avoids
        continuing with an uninitialized recv_counts tensor if a metadata A2A
        fails asynchronously.
        """
        if group is None:
            group = dist.group.WORLD
        world = int(dist.get_world_size(group=group))
        rank = int(dist.get_rank(group=group) if group_rank is None else group_rank)
        if int(send_counts.numel()) != world:
            raise RuntimeError(
                f"[rank{self.rank}] {label} count vector length mismatch: "
                f"numel={int(send_counts.numel())} group_world={world}"
            )

        send_counts = send_counts.to(device=self.device, dtype=torch.long).contiguous()
        if bool(send_counts.lt(0).any().item()):
            raise RuntimeError(f"[rank{self.rank}] {label} has negative send_counts={send_counts.detach().cpu().tolist()}")

        fast_count_raw = str(os.environ.get("SPARSE_NGD_FAST_COUNT_EXCHANGE", "") or "").strip().lower()
        if fast_count_raw == "":
            fast_count = str(label).startswith("node_grouped") and (
                "lean" in str(os.environ.get("SPARSE_NODE_GROUPED_FASTPATH_DECODE_MODE", "") or "").lower()
                or _env_flag("SPARSE_NGD_LEAN_FASTPATH", "0")
            )
        else:
            fast_count = fast_count_raw not in ("0", "false", "no", "off")
        if fast_count:
            recv_counts = torch.empty_like(send_counts)
            self._all_to_all_single_timed(recv_counts, send_counts, group=group)
            if bool(recv_counts.lt(0).any().item()):
                raise RuntimeError(f"[rank{self.rank}] {label} exchanged negative recv_counts={recv_counts.detach().cpu().tolist()}")
            # In the fast metadata path we intentionally avoid an all-gather of
            # the full src x dst matrix.  The exact global total is used only
            # for optional all-zero skip paths; a conservative positive value
            # keeps every rank in the collective sequence.
            try:
                local_total = int(send_counts.sum().item())
                recv_total = int(recv_counts.sum().item())
                self._last_dispatch_count_global_total = max(local_total, recv_total, 1)
            except Exception:
                self._last_dispatch_count_global_total = 1
            if _env_flag("SPARSE_NODE_GROUPED_DEBUG_COUNTS", "0"):
                print(
                    f"[NODE_GROUPED_COUNTS_FAST] rank={self.rank} label={label} "
                    f"send={send_counts.detach().cpu().tolist()} recv={recv_counts.detach().cpu().tolist()}",
                    flush=True,
                )
            return recv_counts

        gathered_parts = [torch.empty_like(send_counts) for _ in range(world)]
        if self.metrics is None or not _env_flag("SPARSE_BOTTLENECK_TIMING", "0"):
            dist.all_gather(gathered_parts, send_counts, group=group)
        elif not torch.cuda.is_available() or self.device.type != "cuda":
            t0 = time.perf_counter()
            dist.all_gather(gathered_parts, send_counts, group=group)
            self._record_comm_ms((time.perf_counter() - t0) * 1000.0)
        else:
            start = torch.cuda.Event(enable_timing=True, blocking=False)
            end = torch.cuda.Event(enable_timing=True, blocking=False)
            start.record(torch.cuda.current_stream(self.device))
            dist.all_gather(gathered_parts, send_counts, group=group)
            end.record(torch.cuda.current_stream(self.device))
            end.synchronize()
            self._record_comm_ms(float(start.elapsed_time(end)))

        gathered = torch.stack(gathered_parts, dim=0).contiguous()
        if bool(gathered.lt(0).any().item()):
            raise RuntimeError(f"[rank{self.rank}] {label} gathered negative count matrix={gathered.detach().cpu().tolist()}")
        recv_counts = gathered[:, rank].contiguous()

        # The receive total can never exceed the global number of packets sent.
        # This catches stale/corrupt metadata before it becomes a huge allocation
        # or a mismatched A2A split vector.
        global_total = int(gathered.sum().item())
        self._last_dispatch_count_global_total = int(global_total)
        recv_total = int(recv_counts.sum().item())
        if recv_total < 0 or recv_total > max(global_total, 0):
            raise RuntimeError(
                f"[rank{self.rank}] {label} invalid recv_total={recv_total}, "
                f"global_total={global_total}, send_counts={send_counts.detach().cpu().tolist()}, "
                f"count_matrix={gathered.detach().cpu().tolist()}"
            )
        if _env_flag("SPARSE_NODE_GROUPED_DEBUG_COUNTS", "0"):
            print(
                f"[NODE_GROUPED_COUNTS] rank={self.rank} label={label} "
                f"send={send_counts.detach().cpu().tolist()} recv={recv_counts.detach().cpu().tolist()} "
                f"matrix={gathered.detach().cpu().tolist()}",
                flush=True,
            )
        return recv_counts

    def _all_to_all_flat_with_dummy_timed(
        self,
        input_flat: torch.Tensor,
        *,
        output_counts: List[int],
        input_counts: List[int],
        item_width: int,
        group: Optional[Any] = None,
        group_rank: Optional[int] = None,
        label: str = "node_grouped_payload",
        workspace: bool = False,
    ) -> torch.Tensor:
        """
        all_to_all_single for flattened fixed-width records, with one self-dummy
        record added on every rank.

        Node-grouped same-node fastpath can leave some ranks with no cross-node
        payload for a stage while other ranks still have work. Some PyTorch/NCCL
        versions may short-circuit a zero-sized all_to_all_single instead of
        enqueuing the collective, which desynchronizes the subsequent collectives.
        A single self-dummy record keeps every rank in the same collective
        sequence without changing real packet semantics; the dummy row is removed
        from the self-source output chunk immediately after the A2A.
        """
        if group is None:
            group = dist.group.WORLD
        world = int(dist.get_world_size(group=group))
        rank = int(dist.get_rank(group=group) if group_rank is None else group_rank)
        width = int(item_width)
        if width <= 0:
            raise RuntimeError(f"[rank{self.rank}] {label} invalid item_width={item_width}")
        if len(input_counts) != world or len(output_counts) != world:
            raise RuntimeError(
                f"[rank{self.rank}] {label} split length mismatch: "
                f"input={len(input_counts)} output={len(output_counts)} world={world}"
            )
        input_counts_i = [int(c) for c in input_counts]
        output_counts_i = [int(c) for c in output_counts]
        if any(c < 0 for c in input_counts_i) or any(c < 0 for c in output_counts_i):
            raise RuntimeError(
                f"[rank{self.rank}] {label} negative split: input={input_counts_i} output={output_counts_i}"
            )

        expected_input_numel = int(sum(input_counts_i)) * width
        if int(input_flat.numel()) != expected_input_numel:
            raise RuntimeError(
                f"[rank{self.rank}] {label} input numel mismatch: "
                f"got={int(input_flat.numel())} expected={expected_input_numel} "
                f"input_counts={input_counts_i} width={width}"
            )

        if _env_flag("SPARSE_NODE_GROUPED_SKIP_DUMMY_A2A", "0"):
            output_items = self._node_grouped_empty(
                (int(sum(output_counts_i)), width),
                dtype=input_flat.dtype,
                label=f"{label}_out",
                workspace=bool(workspace),
            )
            self._all_to_all_single_timed(
                output_items.reshape(-1).contiguous(),
                input_flat.contiguous(),
                [int(c) * width for c in output_counts_i],
                [int(c) * width for c in input_counts_i],
                group=group,
            )
            return output_items.reshape(-1)

        input_items = input_flat.contiguous().view(-1, width)
        dummy = torch.zeros((1, width), dtype=input_items.dtype, device=self.device)
        insert_at = int(sum(input_counts_i[:rank]) + input_counts_i[rank])
        if input_items.numel() == 0:
            aug_input_items = dummy
        else:
            aug_input_items = torch.cat(
                [input_items[:insert_at], dummy, input_items[insert_at:]],
                dim=0,
            ).contiguous()

        aug_input_counts = list(input_counts_i)
        aug_output_counts = list(output_counts_i)
        aug_input_counts[rank] += 1
        aug_output_counts[rank] += 1
        aug_input_splits = [int(c) * width for c in aug_input_counts]
        aug_output_splits = [int(c) * width for c in aug_output_counts]
        aug_output_items = self._node_grouped_empty(
            (int(sum(output_counts_i)) + 1, width),
            dtype=input_items.dtype,
            label=f"{label}_aug_out",
            workspace=bool(workspace),
        )

        self._all_to_all_single_timed(
            aug_output_items.reshape(-1).contiguous(),
            aug_input_items.reshape(-1).contiguous(),
            aug_output_splits,
            aug_input_splits,
            group=group,
        )

        remove_at = int(sum(output_counts_i[:rank]) + output_counts_i[rank])
        if _env_flag("SPARSE_NODE_GROUPED_DEBUG_COUNTS", "0"):
            print(
                f"[NODE_GROUPED_DUMMY_A2A] rank={self.rank} label={label} "
                f"input_counts={input_counts_i} output_counts={output_counts_i} "
                f"insert_at={insert_at} remove_at={remove_at} width={width}",
                flush=True,
            )
        if int(sum(output_counts_i)) == 0:
            return aug_output_items.new_empty((0,))
        return torch.cat(
            [aug_output_items[:remove_at], aug_output_items[remove_at + 1:]],
            dim=0,
        ).contiguous().reshape(-1)

    def _record_activation_comm_bytes(self, nbytes: int) -> None:
        if self.metrics is not None:
            self.metrics.comm_bytes_activation += int(nbytes)
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.a2a_send_bytes += int(nbytes)

    def _record_activation_remote_bytes(self, nbytes: int) -> None:
        if self.metrics is not None:
            self.metrics.comm_bytes_activation_remote += int(nbytes)
        lp = self._ensure_layer_perf()
        if lp is not None:
            lp.a2a_remote_bytes += int(nbytes)

    def _record_dispatch_self_remote(self, dest: torch.Tensor) -> None:
        lp = self._ensure_layer_perf()
        if lp is None:
            return
        if dest.numel() == 0:
            return
        d = dest.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        self_cnt = int((d == int(self.comm_rank)).sum().item())
        total = int(d.numel())
        lp.dispatch_self_assignments += self_cnt
        lp.dispatch_remote_assignments += max(0, total - self_cnt)

    def _record_local_expert_load(self, recv_i: torch.Tensor) -> None:
        lp = self._ensure_layer_perf()
        if lp is None:
            return

        if recv_i.numel() == 0:
            return

        counts = torch.bincount(
            recv_i.to(torch.long),
            minlength=self.num_experts,
        ).to(device=self.device, dtype=torch.float64)

        if lp.expert_hist is None or int(lp.expert_hist.numel()) != int(self.num_experts):
            lp.expert_hist = torch.zeros(
                (self.num_experts,),
                device=self.device,
                dtype=torch.float64,
            )

        lp.expert_hist += counts

    def _record_routing_locality(self, idx: torch.Tensor) -> None:
        """
        idx: [T, K]

        当前 perf/eval 路径中，请求通常是 replicated 到所有 rank 的同一批样本。
        因此同一份 routing decision 会在每个 rank 上都被看到一次。

        为避免跨 rank 重复计数：
        - 只在 rank0 上记录一次
        - 后续 reduce/save 阶段不要再对 local/remote assignment 做 all-reduce sum

        这里 local/remote 的定义是“相对于 rank0 这个固定参考 rank 的 placement 视角”：
        - local_assignments: assignment 的目标 expert 位于 rank0
        - remote_assignments: assignment 的目标 expert 不位于 rank0

        所以它是 reference-rank locality metric，不是全局真实 locality metric。
        """
        lp = self._ensure_layer_perf()
        if lp is None:
            return

        # replicated request 场景下，只让 rank0 统计一次
        if self.rank != 0:
            return

        flat_i = idx.reshape(-1).to(torch.long)     # [T*K]
        dest_ranks = self._resolve_dest_for_flat_experts(flat_i, reason="routing_locality", count_metrics=False)    # [T*K]

        local_cnt = int((dest_ranks == 0).sum().item())
        total_cnt = int(dest_ranks.numel())
        remote_cnt = total_cnt - local_cnt

        lp.local_assignments += local_cnt
        lp.remote_assignments += remote_cnt

    # ---------------- Legacy NodePatch / ERS route repair ----------------
    # Formal ERS is offline-first and default-off.  The runtime path below is
    # still the legacy NodePatch adapter, but it now accepts canonical
    # SPARSE_ERS_* env names while preserving SPARSE_NODEPATCH_* aliases.
    def _nodepatch_phase_enabled(self) -> bool:
        cfg = ers_config_from_env()
        if not cfg.enabled:
            return False
        phase_cfg = str(cfg.phase or "decode").strip().lower()
        if phase_cfg in ("all", "both"):
            return True
        phase = str(getattr(SparseContext, "current_phase", "none") or "none").strip().lower()
        if phase in ("prefill", "decode"):
            return phase == phase_cfg
        seq_len = int(getattr(self, "_current_forward_seq_len", 0) or 0)
        inferred = "decode" if seq_len == 1 else "prefill"
        return inferred == phase_cfg

    def _nodepatch_config(self) -> Tuple[int, int, float, str, bool, bool]:
        cfg = ers_config_from_env()
        top_m = max(int(self.top_k), min(int(cfg.top_m), int(self.num_experts)))
        max_replace = max(0, min(int(cfg.max_replace), int(self.top_k)))
        margin = float(cfg.margin)
        if self._nodepatch_calibration_enabled():
            policy = self._nodepatch_calibration_policy_for_layer()
            if self._nodepatch_calibration_policy_raw() and policy is None:
                return top_m, 0, margin, str(cfg.margin_kind), bool(cfg.keep_top1), bool(cfg.only_cross_node)
            if policy is not None:
                if policy.get("packet_critical_only"):
                    raise RuntimeError(
                        "SPARSE_ERS_CALIB_POLICY packet_critical_only=true is an offline-only "
                        "policy option in the current implementation. Use the packet-critical "
                        "ERS evaluator before enabling runtime repair."
                    )
                if policy.get("top_m") is not None:
                    top_m = max(int(self.top_k), min(int(policy.get("top_m")), int(self.num_experts)))
                if policy.get("max_replace") is not None:
                    max_replace = max(0, min(int(policy.get("max_replace")), int(self.top_k)))
                if policy.get("margin_max") is not None:
                    margin = float(policy.get("margin_max"))
        if self._nodepatch_action() in ("drop", "prune", "zero"):
            top_m = int(self.top_k)
        return top_m, max_replace, margin, str(cfg.margin_kind), bool(cfg.keep_top1), bool(cfg.only_cross_node)

    def _nodepatch_action(self) -> str:
        action = str(env_value_with_alias("SPARSE_ERS_ACTION", "SPARSE_NODEPATCH_ACTION", "replace") or "replace").strip().lower()
        if action in ("drop", "prune", "zero", "zero_drop"):
            return "drop"
        return "replace"

    def _nodepatch_drop_renormalize_enabled(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_DROP_RENORMALIZE", "SPARSE_NODEPATCH_DROP_RENORMALIZE", "1")

    def _nodepatch_drop_require_event_reduction(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_DROP_REQUIRE_EVENT_REDUCTION", "SPARSE_NODEPATCH_DROP_REQUIRE_EVENT_REDUCTION", "1")

    def _nodepatch_margin_disabled(self, margin_tau: float) -> bool:
        return float(margin_tau) < 0.0 or env_flag_with_alias("SPARSE_ERS_DISABLE_MARGIN", "SPARSE_NODEPATCH_DISABLE_MARGIN", "0")

    def _nodepatch_fastpath_enabled(self) -> bool:
        if self._nodepatch_calibration_enabled():
            return False
        if self._nodepatch_narrow_enabled():
            return False
        return env_flag_with_alias("SPARSE_ERS_FASTPATH", "SPARSE_NODEPATCH_FASTPATH", "0")

    def _nodepatch_narrow_enabled(self) -> bool:
        if self._nodepatch_calibration_enabled():
            return False
        return env_flag_with_alias("SPARSE_ERS_NARROW", "SPARSE_NODEPATCH_NARROW", "1")

    def _nodepatch_lean_repair_enabled(self) -> bool:
        if self._nodepatch_calibration_enabled():
            return False
        return self._nodepatch_narrow_enabled() or env_flag_with_alias("SPARSE_ERS_LEAN_REPAIR", "SPARSE_NODEPATCH_LEAN_REPAIR", "0")

    def _nodepatch_lean_keep_topk(self, keep_top1: bool) -> int:
        raw = str(env_value_with_alias("SPARSE_ERS_LEAN_KEEP_TOPK", "SPARSE_NODEPATCH_LEAN_KEEP_TOPK", "") or "").strip()
        if raw:
            try:
                return max(0, min(int(raw), int(self.top_k)))
            except Exception:
                pass
        if self._nodepatch_narrow_enabled():
            return max(1, min(2, int(self.top_k)))
        return 1 if bool(keep_top1) else 0

    def _nodepatch_lean_candidate_scope(self) -> str:
        default_scope = "resident_nearest" if self._nodepatch_narrow_enabled() else "owner_local"
        raw = str(env_value_with_alias("SPARSE_ERS_LEAN_CANDIDATE_SCOPE", "SPARSE_NODEPATCH_LEAN_CANDIDATE_SCOPE", default_scope) or default_scope).strip().lower()
        if raw in ("node", "same_node", "same-node", "local_node", "local-node"):
            return "same_node"
        if raw in (
            "resident_nearest",
            "resident-nearest",
            "cache_resident",
            "cache-resident",
            "resident_owner_local",
            "resident-owner-local",
            "ranked_resident",
            "ranked-resident",
            "risky_resident",
            "risky-resident",
        ):
            return "resident_nearest"
        return "owner_local"

    def _nodepatch_lean_require_improvement(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_LEAN_REQUIRE_IMPROVEMENT", "SPARSE_NODEPATCH_LEAN_REQUIRE_IMPROVEMENT", "1")

    def _nodepatch_lean_stats_enabled(self) -> bool:
        default = "0" if self._nodepatch_narrow_enabled() else "1"
        return env_flag_with_alias("SPARSE_ERS_LEAN_STATS", "SPARSE_NODEPATCH_LEAN_STATS", default)

    def _nodepatch_lean_max_rank_gap(self) -> int:
        default = "4" if self._nodepatch_narrow_enabled() else "0"
        raw = str(env_value_with_alias("SPARSE_ERS_LEAN_MAX_RANK_GAP", "SPARSE_NODEPATCH_LEAN_MAX_RANK_GAP", default) or default).strip()
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _nodepatch_lean_max_repairs_per_layer(self) -> int:
        raw = str(env_value_with_alias("SPARSE_ERS_LEAN_MAX_REPAIRS_PER_LAYER", "SPARSE_NODEPATCH_LEAN_MAX_REPAIRS_PER_LAYER", "0") or "0").strip()
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _nodepatch_lean_protect_old_resident(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_LEAN_PROTECT_OLD_RESIDENT", "SPARSE_NODEPATCH_LEAN_PROTECT_OLD_RESIDENT", "0")

    def _nodepatch_lean_max_prob_delta(self) -> float:
        raw = str(env_value_with_alias("SPARSE_ERS_LEAN_MAX_PROB_DELTA", "SPARSE_NODEPATCH_LEAN_MAX_PROB_DELTA", "") or "").strip()
        if raw == "":
            return -1.0
        try:
            return float(raw)
        except Exception:
            return -1.0

    def _nodepatch_lean_layer_allowed(self) -> bool:
        raw = str(env_value_with_alias("SPARSE_ERS_LEAN_LAYERS", "SPARSE_NODEPATCH_LEAN_LAYERS", "") or "").strip()
        if raw == "":
            return True
        allowed = self._nodepatch_parse_int_set(raw, num_layers=int(self.num_layers))
        if allowed is None:
            return True
        return int(self.layer_idx) in allowed

    def _nodepatch_lean_require_prepared(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_LEAN_REQUIRE_PREPARED", "SPARSE_NODEPATCH_LEAN_REQUIRE_PREPARED", "1")

    def _nodepatch_lean_require_prefetch_complete(self) -> bool:
        # Phase-1 lean ERS is intended to be no-extra-H2D and low-wait: a
        # candidate may be used only when its resident cache entry is not backed
        # by an incomplete prefetch event. Disable this only for explicit
        # diagnostics.
        return env_flag_with_alias("SPARSE_ERS_REQUIRE_PREFETCH_COMPLETE", "SPARSE_NODEPATCH_REQUIRE_PREFETCH_COMPLETE", "1")

    def _nodepatch_lean_require_comm_benefit(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_LEAN_REQUIRE_COMM_BENEFIT", "SPARSE_NODEPATCH_LEAN_REQUIRE_COMM_BENEFIT", "1")

    def _nodepatch_lean_resident_nearest_enabled(self) -> bool:
        if self._nodepatch_calibration_enabled():
            return False
        if env_flag_with_alias("SPARSE_ERS_LEAN_RESIDENT_NEAREST", "SPARSE_NODEPATCH_LEAN_RESIDENT_NEAREST", "0"):
            return True
        return self._nodepatch_lean_candidate_scope() in ("resident_nearest", "cache_resident", "resident_owner_local")

    def _nodepatch_event_aware_enabled(self) -> bool:
        if self._nodepatch_narrow_enabled():
            return False
        return env_flag_with_alias("SPARSE_ERS_EVENT_AWARE", "SPARSE_NODEPATCH_EVENT_AWARE", "0")

    def _nodepatch_event_aggressive_enabled(self) -> bool:
        if self._nodepatch_narrow_enabled():
            return False
        return env_flag_with_alias("SPARSE_ERS_EVENT_AGGRESSIVE", "SPARSE_NODEPATCH_EVENT_AGGRESSIVE", "0")

    def _nodepatch_packet_emptying_first_enabled(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_PACKET_EMPTYING_FIRST", "SPARSE_NODEPATCH_PACKET_EMPTYING_FIRST", "1")

    def _nodepatch_cache_aware_enabled(self) -> bool:
        if self._nodepatch_narrow_enabled():
            return True
        return env_flag_with_alias("SPARSE_ERS_CACHE_AWARE", "SPARSE_NODEPATCH_CACHE_AWARE", "0")

    def _nodepatch_cache_require_owner_local(self) -> bool:
        if self._nodepatch_narrow_enabled():
            return True
        return env_flag_with_alias("SPARSE_ERS_CACHE_REQUIRE_OWNER_LOCAL", "SPARSE_NODEPATCH_CACHE_REQUIRE_OWNER_LOCAL", "1")

    def _nodepatch_cache_require_resident(self) -> bool:
        if self._nodepatch_narrow_enabled():
            return True
        return env_flag_with_alias("SPARSE_ERS_REQUIRE_ADDED_RESIDENT", "SPARSE_NODEPATCH_REQUIRE_ADDED_RESIDENT", "1")

    def _nodepatch_calibration_enabled(self) -> bool:
        return env_flag_with_alias("SPARSE_ERS_CALIB_ENABLE", "SPARSE_NODEPATCH_CALIB_ENABLE", "0")


    def _nodepatch_calibration_policy_raw(self) -> str:
        if not self._nodepatch_calibration_enabled():
            return ""
        return str(env_value_with_alias("SPARSE_ERS_CALIB_POLICY", "SPARSE_NODEPATCH_CALIB_POLICY", "") or "").strip()

    @staticmethod
    def _nodepatch_normalize_candidate_mode(mode: str) -> str:
        try:
            return _ers_normalize_candidate_mode(mode)
        except ERSPolicyError:
            return "remaining_node"

    def _nodepatch_calibration_policy_for_layer(self) -> Optional[Dict[str, Any]]:
        raw = self._nodepatch_calibration_policy_raw()
        if not raw:
            return None
        policy = _ers_parse_policy(
            raw,
            num_layers=int(self.num_layers),
            top_k=int(self.top_k),
            default_candidate_mode=self._nodepatch_calibration_candidate_mode(),
            default_top_m=None,
        )
        rule = _ers_find_rule_for_layer(policy, int(self.layer_idx))
        if rule is None:
            return None
        return {
            "name": rule.name,
            "layers": set(rule.layers),
            "positions": set(rule.positions),
            "max_replace": rule.max_replace,
            "margin_min": rule.margin_min,
            "margin_max": rule.margin_max,
            "candidate_mode": rule.candidate_mode,
            "top_m": rule.top_m,
            "packet_critical_only": rule.packet_critical_only,
            "raw": rule.raw,
        }

    @staticmethod
    def _nodepatch_parse_int_set(raw: str, *, num_layers: Optional[int] = None) -> Optional[set[int]]:
        return _ers_parse_int_set(str(raw), num_layers=num_layers)

    def _nodepatch_calibration_layer_allowed(self) -> bool:
        if not self._nodepatch_calibration_enabled():
            return True
        if self._nodepatch_calibration_policy_raw():
            return self._nodepatch_calibration_policy_for_layer() is not None
        raw = env_value_with_alias("SPARSE_ERS_CALIB_LAYERS", "SPARSE_NODEPATCH_CALIB_LAYERS", "")
        allowed = self._nodepatch_parse_int_set(str(raw), num_layers=int(self.num_layers))
        return allowed is None or int(self.layer_idx) in allowed

    def _nodepatch_calibration_positions(self) -> Optional[set[int]]:
        if not self._nodepatch_calibration_enabled():
            return None
        raw = env_value_with_alias("SPARSE_ERS_CALIB_REMOVE_POSITIONS", "SPARSE_NODEPATCH_CALIB_REMOVE_POSITIONS", "")
        return self._nodepatch_parse_int_set(str(raw), num_layers=None)

    def _nodepatch_calibration_margin_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._nodepatch_calibration_enabled():
            return None, None
        lo_raw = str(env_value_with_alias("SPARSE_ERS_CALIB_MARGIN_MIN", "SPARSE_NODEPATCH_CALIB_MARGIN_MIN", "") or "").strip()
        hi_raw = str(env_value_with_alias("SPARSE_ERS_CALIB_MARGIN_MAX", "SPARSE_NODEPATCH_CALIB_MARGIN_MAX", "") or "").strip()
        lo = None if lo_raw == "" else float(lo_raw)
        hi = None if hi_raw == "" else float(hi_raw)
        return lo, hi

    def _nodepatch_calibration_candidate_mode(self) -> str:
        if not self._nodepatch_calibration_enabled():
            return "remaining_node"
        mode = str(env_value_with_alias("SPARSE_ERS_CALIB_CANDIDATE_MODE", "SPARSE_NODEPATCH_CALIB_CANDIDATE_MODE", "remaining_node") or "remaining_node")
        return self._nodepatch_normalize_candidate_mode(mode)

    def _nodepatch_current_node_id(self) -> int:
        node_ids = self._comm_rank_node_ids()
        return int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0

    def _nodepatch_stage_owner_rank(self, expert_id: int) -> int:
        eid = int(expert_id)
        if self.placement_mode == "scp_ep":
            return int(self._resolve_rank_for_expert(eid, stage=self._current_stage(), reason="nodepatch_owner"))
        owners = self._expert_to_rank_for_stage_cpu()
        if 0 <= eid < len(owners):
            return int(owners[eid])
        return int(eid % max(int(self.comm_world), 1))

    def _nodepatch_stage_owner_node(self, expert_id: int) -> int:
        owner = self._nodepatch_stage_owner_rank(int(expert_id))
        node_ids = self._comm_rank_node_ids()
        if 0 <= int(owner) < len(node_ids):
            return int(node_ids[int(owner)])
        return int(owner)

    def _nodepatch_unique_nodes(self, experts: List[int]) -> int:
        return len(set(int(self._nodepatch_stage_owner_node(int(e))) for e in experts))

    def _nodepatch_cross_items(self, experts: List[int]) -> int:
        counts: Dict[int, int] = {}
        for e in experts:
            node = int(self._nodepatch_stage_owner_node(int(e)))
            counts[node] = int(counts.get(node, 0)) + 1
        return max(0, len(experts) - (max(counts.values()) if counts else 0))

    def _nodepatch_local_resident_set(self) -> set[int]:
        try:
            return set(int(x) for x in self._cache.resident_ids())
        except Exception:
            return set()

    def _nodepatch_candidate_cache_ok(
        self,
        expert_id: int,
        *,
        resident_local: Optional[set[int]] = None,
    ) -> bool:
        if not self._nodepatch_cache_aware_enabled():
            return True
        eid = int(expert_id)
        owner_rank = int(self._resolve_rank_for_expert(eid, stage="decode", reason="ers_cache_ok"))
        # Low-overhead runtime mode: do not all_gather remote cache state from
        # every rank at every decode layer. A cache-safe repair must replace a
        # remote-node expert with an expert already resident on this source rank.
        if self._nodepatch_cache_require_owner_local() and int(owner_rank) != int(self.comm_rank):
            return False
        if self._nodepatch_cache_require_resident():
            if resident_local is None:
                resident_local = self._nodepatch_local_resident_set()
            if eid not in resident_local:
                return False
        return True

    def _apply_nodepatch_route_drop(
        self,
        topv: torch.Tensor,
        topi: torch.Tensor,
        *,
        max_replace: int,
        margin_tau: float,
        margin_kind: str,
        keep_top1: bool,
        only_cross_node: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ERS v2 low-overhead drop path.

        This path does not search top-M replacement candidates. It removes
        low-ranked routed experts by zeroing their combine weight and replacing
        their id with an already-routed or resident local placeholder so the
        downstream dispatcher does not send the token to the original remote
        owner. It is meant for decode-time single-request experiments where the
        quality gate is handled by offline PPL calibration.
        """
        if int(max_replace) <= 0:
            return topv, topi
        calib_enabled = self._nodepatch_calibration_enabled()
        calib_policy = self._nodepatch_calibration_policy_for_layer() if calib_enabled else None
        if calib_enabled and self._nodepatch_calibration_policy_raw() and calib_policy is None:
            return topv, topi
        if calib_enabled and not self._nodepatch_calibration_policy_raw() and not self._nodepatch_calibration_layer_allowed():
            return topv, topi
        if calib_policy is not None and calib_policy.get("packet_critical_only"):
            raise RuntimeError(
                "SPARSE_ERS_CALIB_POLICY packet_critical_only=true is an offline-only "
                "policy option in the current implementation. Use the packet-critical "
                "ERS evaluator before enabling runtime repair."
            )
        if calib_policy is not None and calib_policy.get("max_replace") is not None:
            max_replace = int(calib_policy.get("max_replace"))
            if int(max_replace) <= 0:
                return topv, topi

        patched_i = topi.clone()
        patched_v = topv.clone()
        idx_cpu = topi.detach().to("cpu", dtype=torch.long)
        val_cpu = topv.detach().to("cpu", dtype=torch.float32)

        tokens = int(idx_cpu.shape[0])
        remove_start = 1 if bool(keep_top1) else 0
        calib_positions = self._nodepatch_calibration_positions()
        calib_margin_min, calib_margin_max = self._nodepatch_calibration_margin_bounds()
        calib_candidate_mode = self._nodepatch_calibration_candidate_mode()
        if calib_policy is not None:
            if calib_policy.get("positions") is not None:
                calib_positions = calib_policy.get("positions")
            calib_margin_min = calib_policy.get("margin_min")
            calib_margin_max = calib_policy.get("margin_max")
            calib_candidate_mode = str(calib_policy.get("candidate_mode") or calib_candidate_mode)
            if calib_margin_max is not None:
                margin_tau = float(calib_margin_max)
        disable_margin = self._nodepatch_margin_disabled(margin_tau)
        drop_max_raw = str(env_value_with_alias("SPARSE_ERS_DROP_MAX_WEIGHT", "SPARSE_NODEPATCH_DROP_MAX_WEIGHT", "") or "").strip()
        if drop_max_raw:
            drop_max_weight: Optional[float] = float(drop_max_raw)
        elif calib_margin_max is not None:
            drop_max_weight = float(calib_margin_max)
        elif (not disable_margin) and str(margin_kind).strip().lower() == "prob":
            drop_max_weight = float(margin_tau)
        else:
            drop_max_weight = None
        drop_min_weight = float(calib_margin_min) if calib_margin_min is not None else None
        require_event_reduction = self._nodepatch_drop_require_event_reduction()
        renormalize = self._nodepatch_drop_renormalize_enabled()
        my_node = self._nodepatch_current_node_id()
        cache_aware = self._nodepatch_cache_aware_enabled()
        resident_local = self._nodepatch_local_resident_set() if bool(cache_aware) else set()
        candidate_mode = self._nodepatch_normalize_candidate_mode(str(calib_candidate_mode or "owner_local"))
        if candidate_mode not in ("owner_local", "same_node"):
            candidate_mode = "owner_local"

        def _placeholder_from(existing: List[int], blocked_pos: set[int]) -> Optional[int]:
            for p, e in enumerate(existing):
                if int(p) in blocked_pos:
                    continue
                eid = int(e)
                if candidate_mode == "owner_local":
                    if int(self._resolve_rank_for_expert(eid, stage="decode", reason="ers_drop_placeholder")) == int(self.comm_rank):
                        return eid
                elif int(self._nodepatch_stage_owner_node(eid)) == int(my_node):
                    return eid
            for eid in range(int(self.num_experts)):
                eid_i = int(eid)
                if bool(cache_aware) and self._nodepatch_cache_require_resident() and eid_i not in resident_local:
                    continue
                owner = int(self._resolve_rank_for_expert(eid_i, stage="decode", reason="ers_drop_placeholder"))
                if candidate_mode == "owner_local":
                    if int(owner) == int(self.comm_rank):
                        return eid_i
                elif int(self._nodepatch_stage_owner_node(eid_i)) == int(my_node):
                    if bool(cache_aware) and self._nodepatch_cache_require_owner_local() and int(owner) != int(self.comm_rank):
                        continue
                    return eid_i
            return None

        cross_tokens = 0
        repaired_tokens = 0
        replacements = 0
        events_before = 0
        events_after = 0
        cross_items_before = 0
        cross_items_after = 0
        unique_nodes_before_sum = 0.0
        unique_nodes_after_sum = 0.0
        util_prob_sum = 0.0
        max_margin_prob_sum = 0.0
        max_margin_prob_max = 0.0
        replacement_pair_hist: Dict[str, int] = {}
        removed_expert_hist: Dict[str, int] = {}
        added_expert_hist: Dict[str, int] = {}
        removed_position_hist: Dict[str, int] = {}
        layer_repaired_token_hist: Dict[str, int] = {}

        for row in range(tokens):
            experts = [int(x) for x in idx_cpu[row].tolist()]
            values = [float(x) for x in val_cpu[row].tolist()]
            original_unique = self._nodepatch_unique_nodes(experts)
            original_events = max(0, int(original_unique) - 1)
            original_cross_items = self._nodepatch_cross_items(experts)
            unique_nodes_before_sum += float(original_unique)
            events_before += original_events
            cross_items_before += original_cross_items
            if original_unique <= 1:
                unique_nodes_after_sum += float(original_unique)
                events_after += original_events
                cross_items_after += original_cross_items
                continue
            cross_tokens += 1

            remove_positions = list(range(remove_start, len(experts)))
            if calib_positions is not None:
                remove_positions = [int(p) for p in remove_positions if int(p) in calib_positions]
            remove_positions = sorted(remove_positions, key=lambda p: (float(values[int(p)]), int(p)))

            dropped_positions: set[int] = set()
            row_replacements = 0
            current_experts = experts[:]
            current_unique = original_unique
            current_events = original_events
            current_cross = original_cross_items
            for pos in remove_positions:
                if row_replacements >= int(max_replace):
                    break
                pos_i = int(pos)
                old_e = int(current_experts[pos_i])
                old_node = int(self._nodepatch_stage_owner_node(old_e))
                if bool(only_cross_node) and int(old_node) == int(my_node):
                    continue
                old_weight = float(values[pos_i])
                if drop_min_weight is not None and old_weight < float(drop_min_weight):
                    continue
                if drop_max_weight is not None and old_weight > float(drop_max_weight):
                    continue
                placeholder = _placeholder_from(current_experts, dropped_positions | {pos_i})
                if placeholder is None:
                    continue
                tentative = current_experts[:]
                tentative[pos_i] = int(placeholder)
                tentative_unique = self._nodepatch_unique_nodes(tentative)
                tentative_events = max(0, int(tentative_unique) - 1)
                tentative_cross = self._nodepatch_cross_items(tentative)
                if bool(require_event_reduction) and not (
                    int(tentative_events) < int(current_events) or int(tentative_cross) < int(current_cross)
                ):
                    continue
                current_experts = tentative
                current_unique = tentative_unique
                current_events = tentative_events
                current_cross = tentative_cross
                values[pos_i] = 0.0
                dropped_positions.add(pos_i)
                row_replacements += 1
                util_prob_sum += old_weight
                max_margin_prob_sum += old_weight
                max_margin_prob_max = max(max_margin_prob_max, old_weight)
                layer_key = str(int(self.layer_idx))
                layer_repaired_token_hist[layer_key] = int(layer_repaired_token_hist.get(layer_key, 0)) + 1
                pair_key = f"L{int(self.layer_idx)}:{old_e}->DROP({int(placeholder)})"
                replacement_pair_hist[pair_key] = int(replacement_pair_hist.get(pair_key, 0)) + 1
                removed_key = f"L{int(self.layer_idx)}:{old_e}"
                added_key = f"L{int(self.layer_idx)}:{int(placeholder)}"
                removed_expert_hist[removed_key] = int(removed_expert_hist.get(removed_key, 0)) + 1
                added_expert_hist[added_key] = int(added_expert_hist.get(added_key, 0)) + 1
                pos_key = str(pos_i)
                removed_position_hist[pos_key] = int(removed_position_hist.get(pos_key, 0)) + 1

            if row_replacements > 0:
                if bool(renormalize):
                    denom = float(sum(max(0.0, v) for v in values))
                    if denom > 0.0:
                        values = [float(v) / denom if float(v) > 0.0 else 0.0 for v in values]
                patched_i[row] = torch.tensor(current_experts, device=patched_i.device, dtype=patched_i.dtype)
                patched_v[row] = torch.tensor(values, device=patched_v.device, dtype=patched_v.dtype)
                repaired_tokens += 1
                replacements += int(row_replacements)
            unique_nodes_after_sum += float(current_unique)
            events_after += int(current_events)
            cross_items_after += int(current_cross)

        if self.comm_rank == 0:
            SparseContext.record_nodepatch(
                tokens=tokens,
                cross_node_tokens=cross_tokens,
                repaired_tokens=repaired_tokens,
                replacement_count=replacements,
                event_before=events_before,
                event_after=events_after,
                cross_items_before=cross_items_before,
                cross_items_after=cross_items_after,
                unique_nodes_before_sum=unique_nodes_before_sum,
                unique_nodes_after_sum=unique_nodes_after_sum,
                utility_loss_logit=0.0,
                utility_loss_prob=util_prob_sum,
                max_margin_logit_sum=0.0,
                max_margin_prob_sum=max_margin_prob_sum,
                max_margin_logit_max=0.0,
                max_margin_prob_max=max_margin_prob_max,
                replacement_pair_hist=replacement_pair_hist,
                removed_expert_hist=removed_expert_hist,
                added_expert_hist=added_expert_hist,
                removed_position_hist=removed_position_hist,
                layer_repaired_token_hist=layer_repaired_token_hist,
            )
        return patched_v, patched_i

    def _apply_nodepatch_route_repair_lean(
        self,
        topv: torch.Tensor,
        topi: torch.Tensor,
        topm_v: torch.Tensor,
        topm_i: torch.Tensor,
        *,
        max_replace: int,
        keep_top1: bool,
        only_cross_node: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache-safe token->expert ERS lean repair.

        This is the runtime ERS path used by SPARE's standard EP dispatch.  It is
        intentionally *not* packet-emptying/event-aware: NGD is disabled, so the
        reliable runtime objective is to reduce per-token expert-owner route
        cost while avoiding extra H2D.

        Policy:
          * decode only is enforced by the caller's phase gate;
          * keep top-1/top-2 according to SPARSE_ERS_LEAN_KEEP_TOPK;
          * replace lower-confidence top-k entries using top-M candidates;
          * candidate must satisfy scope (owner_local by default);
          * candidate must already be resident/prepared; optionally reject
            incomplete prefetches so ERS does not add H2D or wait overhead;
          * candidate must strictly lower topology route cost when
            SPARSE_ERS_LEAN_REQUIRE_COMM_BENEFIT=1.
        """
        T, K = int(topi.shape[0]), int(topi.shape[1])
        M = int(topm_i.shape[1])
        if T <= 0 or K <= 0 or M <= K or int(max_replace) <= 0:
            return topv, topi
        if not self._nodepatch_lean_layer_allowed():
            return topv, topi

        keep_topk = int(self._nodepatch_lean_keep_topk(keep_top1))
        keep_topk = max(0, min(keep_topk, K))
        scope = self._nodepatch_lean_candidate_scope()
        resident_nearest = self._nodepatch_lean_resident_nearest_enabled()
        require_improve = self._nodepatch_lean_require_improvement()
        stats_enabled = self._nodepatch_lean_stats_enabled()
        max_rank_gap = self._nodepatch_lean_max_rank_gap()
        max_repairs_per_layer = self._nodepatch_lean_max_repairs_per_layer()
        require_prepared = self._nodepatch_lean_require_prepared()
        require_prefetch_complete = self._nodepatch_lean_require_prefetch_complete()
        require_comm_benefit = self._nodepatch_lean_require_comm_benefit()
        protect_old_resident = self._nodepatch_lean_protect_old_resident()
        max_prob_delta = self._nodepatch_lean_max_prob_delta()
        cache_aware = self._nodepatch_cache_aware_enabled() or bool(resident_nearest) or bool(require_prepared)
        resident_local = self._nodepatch_local_resident_set() if bool(cache_aware) else set()
        my_node = self._nodepatch_current_node_id()
        rank_nodes = self._comm_rank_node_ids()

        def _expert_dispatch_rank(eid: int) -> int:
            return int(self._resolve_rank_for_expert(int(eid), stage="decode", reason="ers"))

        def _route_cost_for_rank(rank: int) -> float:
            return float(self._topology_route_cost(int(self.comm_rank), int(rank)))

        def _route_cost_for_expert(eid: int) -> float:
            return float(_route_cost_for_rank(_expert_dispatch_rank(int(eid))))

        def _prefetch_event_complete(eid: int) -> bool:
            ev = getattr(self, "_prefetch_events", {}).get(int(eid)) if hasattr(self, "_prefetch_events") else None
            if ev is None:
                return True
            try:
                return bool(ev.query())
            except Exception:
                return False

        def _ers_candidate_ready_phase1(eid: int) -> bool:
            eid = int(eid)
            if not self._cache.has(eid):
                return False
            ev = getattr(self, "_prefetch_events", {}).get(eid) if hasattr(self, "_prefetch_events") else None
            if ev is not None:
                return False
            return True

        def _candidate_prepared(eid: int) -> bool:
            eid = int(eid)
            if bool(require_prefetch_complete):
                return _ers_candidate_ready_phase1(eid)
            if eid in resident_local:
                return True
            return bool(self._cache.has(eid))

        def _candidate_scope_ok(eid: int) -> bool:
            eid = int(eid)
            if eid < 0 or eid >= int(self.num_experts):
                return False
            owner = int(_expert_dispatch_rank(eid))
            node = int(self._nodepatch_stage_owner_node(eid))
            if bool(resident_nearest):
                return owner == int(self.comm_rank) and eid in resident_local
            if scope == "owner_local":
                return owner == int(self.comm_rank)
            if scope == "same_node":
                return node == int(my_node)
            # Defensive fallback: default to owner-local rather than silently
            # accepting remote candidates.
            return owner == int(self.comm_rank)

        # CPU copies are used only for cheap routing decisions and statistics.
        idx_cpu = topi.detach().to("cpu", dtype=torch.long)
        topm_i_cpu = topm_i.detach().to("cpu", dtype=torch.long)
        topv_cpu = topv.detach().to("cpu", dtype=torch.float32)
        topm_v_cpu = topm_v.detach().to("cpu", dtype=torch.float32)

        patched_i = topi.clone()
        patched_v = topv.clone()

        tokens = int(idx_cpu.shape[0])
        cross_tokens = 0
        repaired_tokens = 0
        replacements = 0
        events_before = 0
        events_after = 0
        cross_items_before = 0
        cross_items_after = 0
        unique_nodes_before_sum = 0.0
        unique_nodes_after_sum = 0.0
        route_cost_before_sum = 0.0
        route_cost_after_sum = 0.0
        route_cost_gain_sum = 0.0
        old_prob_sum = 0.0
        new_prob_sum = 0.0
        prob_delta_sum = 0.0
        rank_gap_sum = 0.0
        rank_gap_max = 0.0
        reject_not_prepared = 0
        reject_no_gain = 0
        reject_scope = 0
        reject_duplicate = 0
        reject_rank_gap = 0
        reject_cache = 0
        reject_strict_resident = 0
        no_replace_position_count = 0
        replacement_pair_hist: Dict[str, int] = {}
        removed_expert_hist: Dict[str, int] = {}
        added_expert_hist: Dict[str, int] = {}
        removed_position_hist: Dict[str, int] = {}
        replacement_rank_gap_hist: Dict[str, int] = {}

        remove_positions = list(range(K - 1, keep_topk - 1, -1))
        if not remove_positions:
            no_replace_position_count = tokens

        for row in range(tokens):
            if max_repairs_per_layer > 0 and replacements >= int(max_repairs_per_layer):
                break
            experts = [int(x) for x in idx_cpu[row].tolist()]
            original_unique = self._nodepatch_unique_nodes(experts)
            original_events = max(0, int(original_unique) - 1)
            original_cross = self._nodepatch_cross_items(experts)
            original_route_cost = float(sum(_route_cost_for_expert(e) for e in experts))
            if stats_enabled:
                unique_nodes_before_sum += float(original_unique)
                events_before += int(original_events)
                cross_items_before += int(original_cross)
                route_cost_before_sum += float(original_route_cost)
            if original_cross > 0:
                cross_tokens += 1
            if not remove_positions:
                if stats_enabled:
                    unique_nodes_after_sum += float(original_unique)
                    events_after += int(original_events)
                    cross_items_after += int(original_cross)
                    route_cost_after_sum += float(original_route_cost)
                continue

            current = experts[:]
            used = set(current)
            row_repairs: List[Tuple[int, int, int, int, float, float, float, float]] = []
            # Replace lower-confidence top-K entries first; top-M order is the
            # runtime quality prior when margin gating is disabled.
            for pos in remove_positions:
                if len(row_repairs) >= int(max_replace):
                    break
                if max_repairs_per_layer > 0 and replacements + len(row_repairs) >= int(max_repairs_per_layer):
                    break
                old_e = int(current[int(pos)])
                old_node = int(self._nodepatch_stage_owner_node(old_e))
                old_owner = int(_expert_dispatch_rank(old_e))
                old_cost = float(_route_cost_for_rank(old_owner))
                old_prob = float(topv_cpu[row, int(pos)].item())
                if bool(only_cross_node) and int(old_node) == int(my_node):
                    continue
                if bool(require_comm_benefit) and old_cost <= 0.0:
                    continue
                if bool(protect_old_resident) and old_e in resident_local:
                    reject_strict_resident += 1
                    continue

                chosen_pos = -1
                chosen_e = -1
                chosen_new_cost = 0.0
                # Always scan from K onward for each replaceable position. The
                # `used` set prevents duplicate experts within the token.
                for scan_pos in range(K, M):
                    rank_gap = int(scan_pos) - int(pos)
                    if int(max_rank_gap) > 0 and rank_gap > int(max_rank_gap):
                        reject_rank_gap += 1
                        break
                    cand_e = int(topm_i_cpu[row, int(scan_pos)].item())
                    if cand_e in used:
                        reject_duplicate += 1
                        continue
                    new_prob = float(topm_v_cpu[row, int(scan_pos)].item())
                    if float(max_prob_delta) >= 0.0 and (old_prob - new_prob) > float(max_prob_delta):
                        reject_rank_gap += 1
                        continue
                    if not _candidate_scope_ok(cand_e):
                        reject_scope += 1
                        continue
                    if bool(require_prepared) and not _candidate_prepared(cand_e):
                        reject_not_prepared += 1
                        continue
                    if bool(cache_aware) and not self._nodepatch_candidate_cache_ok(cand_e, resident_local=resident_local):
                        reject_cache += 1
                        continue
                    if bool(resident_nearest) and (cand_e not in resident_local or not _candidate_prepared(cand_e)):
                        reject_strict_resident += 1
                        continue
                    new_cost = float(_route_cost_for_expert(cand_e))
                    if bool(require_comm_benefit) and not (new_cost < old_cost):
                        reject_no_gain += 1
                        continue
                    tentative = current[:]
                    tentative[int(pos)] = int(cand_e)
                    tentative_route_cost = float(sum(_route_cost_for_expert(e) for e in tentative))
                    if bool(require_improve) and not (tentative_route_cost < float(sum(_route_cost_for_expert(e) for e in current))):
                        reject_no_gain += 1
                        continue
                    chosen_e = int(cand_e)
                    chosen_pos = int(scan_pos)
                    chosen_new_cost = float(new_cost)
                    break
                if chosen_e < 0:
                    continue
                used.discard(old_e)
                used.add(chosen_e)
                current[int(pos)] = int(chosen_e)
                new_prob = float(topm_v_cpu[row, int(chosen_pos)].item())
                rank_gap = int(chosen_pos) - int(pos)
                row_repairs.append((int(pos), int(chosen_pos), int(old_e), int(chosen_e), float(old_cost), float(chosen_new_cost), old_prob, new_prob))
                replacement_pair_hist[f"{old_e}->{chosen_e}"] = int(replacement_pair_hist.get(f"{old_e}->{chosen_e}", 0)) + 1
                removed_expert_hist[str(old_e)] = int(removed_expert_hist.get(str(old_e), 0)) + 1
                added_expert_hist[str(chosen_e)] = int(added_expert_hist.get(str(chosen_e), 0)) + 1
                removed_position_hist[str(pos)] = int(removed_position_hist.get(str(pos), 0)) + 1
                replacement_rank_gap_hist[str(rank_gap)] = int(replacement_rank_gap_hist.get(str(rank_gap), 0)) + 1
                old_prob_sum += old_prob
                new_prob_sum += new_prob
                prob_delta_sum += max(0.0, old_prob - new_prob)
                rank_gap_sum += float(rank_gap)
                rank_gap_max = max(float(rank_gap_max), float(rank_gap))
                route_cost_gain_sum += max(0.0, float(old_cost) - float(chosen_new_cost))

            if row_repairs:
                repaired_tokens += 1
                replacements += int(len(row_repairs))
                for pos, cand_pos, _old_e, _chosen_e, _old_cost, _new_cost, _old_prob, _new_prob in row_repairs:
                    patched_i[row, int(pos)] = topm_i[row, int(cand_pos)]
                    patched_v[row, int(pos)] = topm_v[row, int(cand_pos)]

            if stats_enabled:
                final_unique = self._nodepatch_unique_nodes(current)
                final_events = max(0, int(final_unique) - 1)
                final_cross = self._nodepatch_cross_items(current)
                final_route_cost = float(sum(_route_cost_for_expert(e) for e in current))
                unique_nodes_after_sum += float(final_unique)
                events_after += int(final_events)
                cross_items_after += int(final_cross)
                route_cost_after_sum += float(final_route_cost)

        if stats_enabled and self.comm_rank == 0:
            SparseContext.record_nodepatch(
                tokens=tokens,
                cross_node_tokens=cross_tokens,
                repaired_tokens=repaired_tokens,
                replacement_count=replacements,
                event_before=events_before,
                event_after=events_after,
                cross_items_before=cross_items_before,
                cross_items_after=cross_items_after,
                unique_nodes_before_sum=unique_nodes_before_sum,
                unique_nodes_after_sum=unique_nodes_after_sum,
                utility_loss_logit=0.0,
                utility_loss_prob=prob_delta_sum,
                max_margin_logit_sum=0.0,
                max_margin_prob_sum=prob_delta_sum,
                max_margin_logit_max=0.0,
                max_margin_prob_max=0.0,
                replacement_pair_hist=replacement_pair_hist,
                removed_expert_hist=removed_expert_hist,
                added_expert_hist=added_expert_hist,
                removed_position_hist=removed_position_hist,
                replacement_rank_gap_hist=replacement_rank_gap_hist,
                layer_repaired_token_hist={str(int(self.layer_idx)): int(repaired_tokens)} if repaired_tokens else {},
                route_cost_before=route_cost_before_sum,
                route_cost_after=route_cost_after_sum,
                route_cost_gain=route_cost_gain_sum,
                old_prob_sum=old_prob_sum,
                new_prob_sum=new_prob_sum,
                prob_delta_sum=prob_delta_sum,
                rank_gap_sum=rank_gap_sum,
                rank_gap_max=rank_gap_max,
                candidate_not_prepared_reject_count=reject_not_prepared,
                candidate_no_comm_gain_reject_count=reject_no_gain,
                candidate_scope_reject_count=reject_scope,
                candidate_duplicate_reject_count=reject_duplicate,
                candidate_rank_gap_reject_count=reject_rank_gap,
                candidate_cache_reject_count=reject_cache,
                candidate_strict_resident_reject_count=reject_strict_resident,
                no_replace_position_count=no_replace_position_count,
            )
        return patched_v, patched_i

    def _apply_nodepatch_route_repair(
        self,
        topv: torch.Tensor,
        topi: torch.Tensor,
        topm_v: torch.Tensor,
        topm_i: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        max_replace: int,
        margin_tau: float,
        margin_kind: str,
        keep_top1: bool,
        only_cross_node: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retired heavy ERS path.  The old implementation copied topK/topM/logits
        # to CPU and searched replacement combinations; keep this stub only so
        # legacy env combinations fail safe without changing routing.
        return topv, topi
        if int(max_replace) <= 0 or int(topm_i.shape[-1]) <= int(self.top_k):
            return topv, topi
        calib_enabled = self._nodepatch_calibration_enabled()
        calib_policy = self._nodepatch_calibration_policy_for_layer() if calib_enabled else None
        if calib_enabled and self._nodepatch_calibration_policy_raw() and calib_policy is None:
            return topv, topi
        if calib_enabled and not self._nodepatch_calibration_policy_raw() and not self._nodepatch_calibration_layer_allowed():
            return topv, topi
        if calib_policy is not None and calib_policy.get("max_replace") is not None:
            max_replace = int(calib_policy.get("max_replace"))
            if int(max_replace) <= 0:
                return topv, topi

        try:
            from itertools import combinations
        except Exception:
            return topv, topi

        patched_i = topi.clone()
        patched_v = topv.clone()

        idx_cpu = topi.detach().to("cpu", dtype=torch.long)
        topm_i_cpu = topm_i.detach().to("cpu", dtype=torch.long)
        topm_v_cpu = topm_v.detach().to("cpu", dtype=torch.float32)
        topm_z_cpu = router_logits.gather(1, topm_i).detach().to("cpu", dtype=torch.float32)

        tokens = int(idx_cpu.shape[0])
        cross_tokens = 0
        repaired_tokens = 0
        replacements = 0
        events_before = 0
        events_after = 0
        cross_items_before = 0
        cross_items_after = 0
        unique_nodes_before_sum = 0.0
        unique_nodes_after_sum = 0.0
        util_logit_sum = 0.0
        util_prob_sum = 0.0
        max_margin_logit_sum = 0.0
        max_margin_prob_sum = 0.0
        max_margin_logit_max = 0.0
        max_margin_prob_max = 0.0
        replacement_pair_hist: Dict[str, int] = {}
        removed_expert_hist: Dict[str, int] = {}
        added_expert_hist: Dict[str, int] = {}
        removed_position_hist: Dict[str, int] = {}
        layer_repaired_token_hist: Dict[str, int] = {}
        calib_positions = self._nodepatch_calibration_positions()
        calib_margin_min, calib_margin_max = self._nodepatch_calibration_margin_bounds()
        calib_candidate_mode = self._nodepatch_calibration_candidate_mode()
        calib_top_m: Optional[int] = None
        if calib_policy is not None:
            if calib_policy.get("packet_critical_only"):
                raise RuntimeError(
                    "SPARSE_ERS_CALIB_POLICY packet_critical_only=true is an offline-only "
                    "policy option in the current implementation. Use the packet-critical "
                    "ERS evaluator before enabling runtime repair."
                )
            if calib_policy.get("positions") is not None:
                calib_positions = calib_policy.get("positions")
            calib_margin_min = calib_policy.get("margin_min")
            calib_margin_max = calib_policy.get("margin_max")
            calib_candidate_mode = str(calib_policy.get("candidate_mode") or calib_candidate_mode)
            if calib_policy.get("top_m") is not None:
                calib_top_m = int(calib_policy.get("top_m"))
            if calib_margin_max is not None:
                margin_tau = float(calib_margin_max)
        if calib_top_m is not None:
            topm_limit = min(int(calib_top_m), int(topm_i_cpu.shape[1]))
            if topm_limit <= int(self.top_k):
                return topv, topi
            topm_i_cpu = topm_i_cpu[:, :topm_limit]
            topm_v_cpu = topm_v_cpu[:, :topm_limit]
            topm_z_cpu = topm_z_cpu[:, :topm_limit]
        disable_margin = self._nodepatch_margin_disabled(margin_tau)
        calib_my_node = self._nodepatch_current_node_id()
        cache_aware = self._nodepatch_cache_aware_enabled()
        resident_local = self._nodepatch_local_resident_set() if bool(cache_aware) else set()

        remove_start = 1 if bool(keep_top1) else 0
        for row in range(tokens):
            experts = [int(x) for x in idx_cpu[row].tolist()]
            original_unique = self._nodepatch_unique_nodes(experts)
            original_events = max(0, int(original_unique) - 1)
            original_cross_items = self._nodepatch_cross_items(experts)
            unique_nodes_before_sum += float(original_unique)
            events_before += original_events
            cross_items_before += original_cross_items
            if original_unique <= 1:
                events_after += original_events
                cross_items_after += original_cross_items
                unique_nodes_after_sum += float(original_unique)
                continue
            cross_tokens += 1
            if bool(only_cross_node) and original_unique <= 1:
                events_after += original_events
                cross_items_after += original_cross_items
                unique_nodes_after_sum += float(original_unique)
                continue

            tm = [int(x) for x in topm_i_cpu[row].tolist()]
            tv = [float(x) for x in topm_v_cpu[row].tolist()]
            tz = [float(x) for x in topm_z_cpu[row].tolist()]
            topm_rank = {int(e): int(pos) for pos, e in enumerate(tm)}
            prob_map = {int(e): float(v) for e, v in zip(tm, tv)}
            logit_map = {int(e): float(z) for e, z in zip(tm, tz)}
            topk_set = set(experts)
            candidates = [int(e) for e in tm if int(e) not in topk_set]
            remove_positions = list(range(remove_start, len(experts)))
            if calib_positions is not None:
                remove_positions = [int(p) for p in remove_positions if int(p) in calib_positions]
            best_key: Optional[Tuple[int, int, float, int]] = None
            best_payload: Optional[Tuple[List[int], Tuple[int, ...], Tuple[int, ...], float, float, float, float]] = None
            max_r = min(int(max_replace), len(remove_positions), len(candidates))
            for r_count in range(1, max_r + 1):
                for pos_combo in combinations(remove_positions, r_count):
                    removed = [experts[int(p)] for p in pos_combo]
                    remaining = [e for pos, e in enumerate(experts) if pos not in set(pos_combo)]
                    remaining_nodes = set(int(self.expert_to_node_cpu[int(e)]) for e in remaining)
                    if calib_enabled and calib_candidate_mode == "same_node":
                        valid_candidates = [
                            int(e) for e in candidates
                            if int(self.expert_to_node_cpu[int(e)]) == int(calib_my_node)
                        ]
                    elif calib_enabled and calib_candidate_mode == "owner_local":
                        valid_candidates = [
                            int(e) for e in candidates
                            if int(self.expert_to_rank_cpu[int(e)]) == int(self.comm_rank)
                        ]
                    elif calib_enabled and calib_candidate_mode == "any_non_topk":
                        valid_candidates = [int(e) for e in candidates]
                    else:
                        valid_candidates = [
                            int(e) for e in candidates
                            if int(self.expert_to_node_cpu[int(e)]) in remaining_nodes
                        ]
                    if bool(cache_aware):
                        valid_candidates = [
                            int(e) for e in valid_candidates
                            if self._nodepatch_candidate_cache_ok(
                                int(e),
                                resident_local=resident_local,
                            )
                        ]
                    for add_combo in combinations(valid_candidates, r_count):
                        patched = list(remaining) + [int(e) for e in add_combo]
                        if len(set(patched)) != len(patched):
                            continue
                        patched_events = max(0, self._nodepatch_unique_nodes(patched) - 1)
                        if patched_events >= original_events:
                            continue
                        removed_sorted = sorted(removed, key=lambda e: logit_map.get(int(e), 0.0), reverse=True)
                        added_sorted = sorted(add_combo, key=lambda e: logit_map.get(int(e), 0.0), reverse=True)
                        margins_logit = [
                            float(logit_map.get(int(r), 0.0) - logit_map.get(int(a), 0.0))
                            for r, a in zip(removed_sorted, added_sorted)
                        ]
                        margins_prob = [
                            float(prob_map.get(int(r), 0.0) - prob_map.get(int(a), 0.0))
                            for r, a in zip(removed_sorted, added_sorted)
                        ]
                        max_ml = max(margins_logit) if margins_logit else 0.0
                        max_mp = max(margins_prob) if margins_prob else 0.0
                        margin_value = max_mp if margin_kind == "prob" else max_ml
                        if (not disable_margin) and float(margin_value) > float(margin_tau):
                            continue
                        if calib_enabled:
                            if calib_margin_min is not None and float(margin_value) < float(calib_margin_min):
                                continue
                            if calib_margin_max is not None and float(margin_value) > float(calib_margin_max):
                                continue
                        patched_cross = self._nodepatch_cross_items(patched)
                        util_l = max(0.0, float(sum(margins_logit)))
                        util_p = max(0.0, float(sum(margins_prob)))
                        key = (
                            int(original_events - patched_events),
                            int(original_cross_items - patched_cross),
                            -float(util_l),
                            -int(r_count),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_payload = (
                                patched,
                                tuple(int(p) for p in pos_combo),
                                tuple(int(e) for e in add_combo),
                                util_l,
                                util_p,
                                max(0.0, float(max_ml)),
                                max(0.0, float(max_mp)),
                            )

            if best_payload is None:
                events_after += original_events
                cross_items_after += original_cross_items
                unique_nodes_after_sum += float(original_unique)
                continue

            patched, pos_combo, add_combo, util_l, util_p, max_ml, max_mp = best_payload
            patched_unique = self._nodepatch_unique_nodes(patched)
            patched_cross = self._nodepatch_cross_items(patched)
            unique_nodes_after_sum += float(patched_unique)
            events_after += max(0, int(patched_unique) - 1)
            cross_items_after += patched_cross
            repaired_tokens += 1
            replacements += len(add_combo)
            layer_key = str(int(self.layer_idx))
            layer_repaired_token_hist[layer_key] = int(layer_repaired_token_hist.get(layer_key, 0)) + 1
            util_logit_sum += util_l
            util_prob_sum += util_p
            max_margin_logit_sum += max_ml
            max_margin_prob_sum += max_mp
            max_margin_logit_max = max(max_margin_logit_max, max_ml)
            max_margin_prob_max = max(max_margin_prob_max, max_mp)

            row_experts = experts[:]
            row_values = [float(x) for x in topv[row].detach().to("cpu", dtype=torch.float32).tolist()]
            for p, add_e in zip(pos_combo, add_combo):
                removed_e = int(experts[int(p)])
                row_experts[int(p)] = int(add_e)
                row_values[int(p)] = float(prob_map.get(int(add_e), 0.0))
                pair_key = f"L{int(self.layer_idx)}:{removed_e}->{int(add_e)}"
                replacement_pair_hist[pair_key] = int(replacement_pair_hist.get(pair_key, 0)) + 1
                removed_key = f"L{int(self.layer_idx)}:{removed_e}"
                added_key = f"L{int(self.layer_idx)}:{int(add_e)}"
                removed_expert_hist[removed_key] = int(removed_expert_hist.get(removed_key, 0)) + 1
                added_expert_hist[added_key] = int(added_expert_hist.get(added_key, 0)) + 1
                pos_key = str(int(p))
                removed_position_hist[pos_key] = int(removed_position_hist.get(pos_key, 0)) + 1
            patched_i[row] = torch.tensor(row_experts, device=patched_i.device, dtype=patched_i.dtype)
            patched_v[row] = torch.tensor(row_values, device=patched_v.device, dtype=patched_v.dtype)

        if self.comm_rank == 0:
            SparseContext.record_nodepatch(
                tokens=tokens,
                cross_node_tokens=cross_tokens,
                repaired_tokens=repaired_tokens,
                replacement_count=replacements,
                event_before=events_before,
                event_after=events_after,
                cross_items_before=cross_items_before,
                cross_items_after=cross_items_after,
                unique_nodes_before_sum=unique_nodes_before_sum,
                unique_nodes_after_sum=unique_nodes_after_sum,
                utility_loss_logit=util_logit_sum,
                utility_loss_prob=util_prob_sum,
                max_margin_logit_sum=max_margin_logit_sum,
                max_margin_prob_sum=max_margin_prob_sum,
                max_margin_logit_max=max_margin_logit_max,
                max_margin_prob_max=max_margin_prob_max,
                replacement_pair_hist=replacement_pair_hist,
                removed_expert_hist=removed_expert_hist,
                added_expert_hist=added_expert_hist,
                removed_position_hist=removed_position_hist,
                layer_repaired_token_hist=layer_repaired_token_hist,
            )
        return patched_v, patched_i

    def _apply_nodepatch_route_repair_fastpath(
        self,
        topv: torch.Tensor,
        topi: torch.Tensor,
        topm_v: torch.Tensor,
        topm_i: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        max_replace: int,
        margin_tau: float,
        margin_kind: str,
        keep_top1: bool,
        only_cross_node: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tensorized NodePatch for the runtime-critical max_replace=1 path.

        This implements the same conservative repair used by the Python path:
        remove one non-top1 singleton-node TopK expert and replace it with a
        TopM candidate already on one of the remaining nodes, so U_node drops.
        """
        K = int(self.top_k)
        M = int(topm_i.shape[-1])
        T = int(topi.shape[0])
        if T <= 0 or K <= 1 or M <= K or int(max_replace) != 1:
            return topv, topi

        expert_to_node = self.expert_to_node.to(device=topi.device)
        topk_nodes = expert_to_node[topi.to(torch.long)]
        topm_nodes = expert_to_node[topm_i.to(torch.long)]
        num_nodes = int(max(int(expert_to_node.max().item()) + 1, 1))
        node_ids = self._comm_rank_node_ids()
        my_node_id = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
        event_aware = env_flag_with_alias("SPARSE_ERS_EVENT_AWARE", "SPARSE_NODEPATCH_EVENT_AWARE", "0")
        event_only_empty = env_flag_with_alias("SPARSE_ERS_EVENT_ONLY_EMPTY_PACKET", "SPARSE_NODEPATCH_EVENT_ONLY_EMPTY_PACKET", "1")
        require_local_candidate = env_flag_with_alias("SPARSE_ERS_EVENT_REQUIRE_LOCAL_CANDIDATE", "SPARSE_NODEPATCH_EVENT_REQUIRE_LOCAL_CANDIDATE", "1")
        try:
            event_packet_max_items = int(env_value_with_alias("SPARSE_ERS_EVENT_PACKET_MAX_ITEMS", "SPARSE_NODEPATCH_EVENT_PACKET_MAX_ITEMS", "4") or 4)
        except Exception:
            event_packet_max_items = 4
        if event_packet_max_items <= 0:
            event_packet_max_items = 10**9

        node_counts = torch.zeros((T, num_nodes), device=topi.device, dtype=torch.long)
        node_counts.scatter_add_(1, topk_nodes, torch.ones_like(topk_nodes, dtype=torch.long))
        unique_before = (node_counts > 0).sum(dim=1)
        events_before_vec = (unique_before - 1).clamp_min(0)
        max_count_before = node_counts.max(dim=1).values
        cross_items_before_vec = K - max_count_before
        cross_token_mask = unique_before > 1
        remote_node_mask = torch.ones((num_nodes,), device=topi.device, dtype=torch.bool)
        if 0 <= int(my_node_id) < int(num_nodes):
            remote_node_mask[int(my_node_id)] = False
        before_remote_packet_counts = (node_counts > 0)[:, remote_node_mask].sum(dim=0)
        runtime_packet_events_before = int(before_remote_packet_counts.gt(0).sum().item())

        remove_allowed = torch.ones((K,), device=topi.device, dtype=torch.bool)
        if bool(keep_top1):
            remove_allowed[0] = False
        remove_node_count = node_counts.gather(1, topk_nodes)
        remove_singleton = remove_node_count.eq(1) & remove_allowed.view(1, K)

        candidate_in_topk = topm_i.unsqueeze(1).eq(topi.unsqueeze(2)).any(dim=1)
        candidate_not_topk = ~candidate_in_topk
        candidate_node_count = node_counts.gather(1, topm_nodes)

        removed_nodes = topk_nodes.unsqueeze(2)
        candidate_nodes = topm_nodes.unsqueeze(1)
        valid = (
            remove_singleton.unsqueeze(2)
            & candidate_not_topk.unsqueeze(1)
            & candidate_node_count.gt(0).unsqueeze(1)
            & candidate_nodes.ne(removed_nodes)
            & cross_token_mask.view(T, 1, 1)
        )
        if bool(only_cross_node):
            valid = valid & cross_token_mask.view(T, 1, 1)
        if bool(event_aware):
            my_node_tensor = torch.tensor(int(my_node_id), device=topi.device, dtype=topk_nodes.dtype)
            # Event-aware NodePatch optimizes the actual outgoing weak-edge event:
            # remove a singleton remote-node expert and replace it with a local
            # node candidate. Collapsing a local singleton into a remote node may
            # improve U_node but does not eliminate an outgoing inter-node packet.
            valid = valid & removed_nodes.ne(my_node_tensor)
            if bool(require_local_candidate):
                valid = valid & candidate_nodes.eq(my_node_tensor)
        if self._nodepatch_cache_aware_enabled():
            cache_safe_expert = torch.ones(
                (int(self.num_experts),),
                device=topi.device,
                dtype=torch.bool,
            )
            if self._nodepatch_cache_require_owner_local():
                cache_safe_expert &= self.expert_to_rank.to(device=topi.device).eq(int(self.comm_rank))
            if self._nodepatch_cache_require_resident():
                resident_ids = self._nodepatch_local_resident_set()
                resident_mask = torch.zeros_like(cache_safe_expert)
                if resident_ids:
                    resident_tensor = torch.tensor(
                        sorted(int(e) for e in resident_ids if 0 <= int(e) < int(self.num_experts)),
                        device=topi.device,
                        dtype=torch.long,
                    )
                    if resident_tensor.numel() > 0:
                        resident_mask[resident_tensor] = True
                cache_safe_expert &= resident_mask
            valid = valid & cache_safe_expert[topm_i.to(torch.long)].unsqueeze(1)

        removed_logit = router_logits.gather(1, topi.to(torch.long))
        topm_logit = router_logits.gather(1, topm_i.to(torch.long))
        margin_logit = removed_logit.unsqueeze(2) - topm_logit.unsqueeze(1)
        margin_prob = topv.to(torch.float32).unsqueeze(2) - topm_v.to(torch.float32).unsqueeze(1)
        margin_value = margin_prob if str(margin_kind).lower() == "prob" else margin_logit
        if not self._nodepatch_margin_disabled(margin_tau):
            valid = valid & margin_value.le(float(margin_tau))

        # Prefer repairs that reduce cross-node item mass as well as U_node,
        # then prefer the smallest score loss.
        patched_max_count = torch.maximum(
            max_count_before.view(T, 1),
            candidate_node_count + 1,
        )
        cross_delta = (patched_max_count - max_count_before.view(T, 1)).clamp_min(0)
        score = cross_delta.to(torch.float32).unsqueeze(1) * 1000000.0 - margin_logit.clamp_min(0.0)
        score = score.masked_fill(~valid, -1.0e30)
        flat_score = score.reshape(T, -1)
        best_score, best_flat = flat_score.max(dim=1)
        repaired_mask = best_score.gt(-1.0e20)

        if not bool(repaired_mask.any().item()):
            if self.comm_rank == 0:
                SparseContext.record_nodepatch(
                    tokens=T,
                    cross_node_tokens=int(cross_token_mask.sum().item()),
                    repaired_tokens=0,
                    replacement_count=0,
                    event_before=int(events_before_vec.sum().item()),
                    event_after=int(events_before_vec.sum().item()),
                    cross_items_before=int(cross_items_before_vec.sum().item()),
                    cross_items_after=int(cross_items_before_vec.sum().item()),
                    unique_nodes_before_sum=float(unique_before.sum().item()),
                    unique_nodes_after_sum=float(unique_before.sum().item()),
                    utility_loss_logit=0.0,
                    utility_loss_prob=0.0,
                    max_margin_logit_sum=0.0,
                    max_margin_prob_sum=0.0,
                    max_margin_logit_max=0.0,
                    max_margin_prob_max=0.0,
                    runtime_packet_events_before=runtime_packet_events_before,
                    runtime_packet_events_after=runtime_packet_events_before,
                    runtime_packet_events_eliminated=0.0,
                    event_aware_skipped_repairs=0.0,
                )
            return topv, topi

        remove_pos = torch.div(best_flat, M, rounding_mode="floor")
        add_pos = best_flat.remainder(M)
        row_ids = torch.arange(T, device=topi.device)
        repaired_rows = row_ids[repaired_mask]
        repaired_remove_pos = remove_pos[repaired_mask]
        repaired_add_pos = add_pos[repaired_mask]
        event_aware_skipped_repairs = 0

        hypothetical_counts = node_counts.clone()
        repaired_removed_nodes = topk_nodes[repaired_rows, repaired_remove_pos]
        repaired_added_nodes = topm_nodes[repaired_rows, repaired_add_pos]
        hypothetical_counts[repaired_rows, repaired_removed_nodes] -= 1
        hypothetical_counts[repaired_rows, repaired_added_nodes] += 1

        if bool(event_aware) and bool(event_only_empty) and int(repaired_rows.numel()) > 0:
            before_by_remote = (node_counts > 0)[:, remote_node_mask].sum(dim=0)
            after_by_remote = (hypothetical_counts > 0)[:, remote_node_mask].sum(dim=0)
            remote_node_ids = torch.arange(num_nodes, device=topi.device, dtype=torch.long)[remote_node_mask]
            eliminated_remote_nodes = remote_node_ids[
                before_by_remote.gt(0)
                & after_by_remote.eq(0)
                & before_by_remote.le(int(event_packet_max_items))
            ]
            if int(eliminated_remote_nodes.numel()) > 0:
                keep = repaired_removed_nodes.unsqueeze(1).eq(eliminated_remote_nodes.view(1, -1)).any(dim=1)
            else:
                keep = torch.zeros_like(repaired_rows, dtype=torch.bool)
            event_aware_skipped_repairs = int(repaired_rows.numel() - int(keep.sum().item()))
            repaired_rows = repaired_rows[keep]
            repaired_remove_pos = repaired_remove_pos[keep]
            repaired_add_pos = repaired_add_pos[keep]
            repaired_removed_nodes = repaired_removed_nodes[keep]
            repaired_added_nodes = repaired_added_nodes[keep]

        if int(repaired_rows.numel()) == 0:
            if self.comm_rank == 0:
                SparseContext.record_nodepatch(
                    tokens=T,
                    cross_node_tokens=int(cross_token_mask.sum().item()),
                    repaired_tokens=0,
                    replacement_count=0,
                    event_before=int(events_before_vec.sum().item()),
                    event_after=int(events_before_vec.sum().item()),
                    cross_items_before=int(cross_items_before_vec.sum().item()),
                    cross_items_after=int(cross_items_before_vec.sum().item()),
                    unique_nodes_before_sum=float(unique_before.sum().item()),
                    unique_nodes_after_sum=float(unique_before.sum().item()),
                    utility_loss_logit=0.0,
                    utility_loss_prob=0.0,
                    max_margin_logit_sum=0.0,
                    max_margin_prob_sum=0.0,
                    max_margin_logit_max=0.0,
                    max_margin_prob_max=0.0,
                    runtime_packet_events_before=runtime_packet_events_before,
                    runtime_packet_events_after=runtime_packet_events_before,
                    runtime_packet_events_eliminated=0.0,
                    event_aware_skipped_repairs=float(event_aware_skipped_repairs),
                )
            return topv, topi

        patched_i = topi.clone()
        patched_v = topv.clone()
        patched_i[repaired_rows, repaired_remove_pos] = topm_i[repaired_rows, repaired_add_pos]
        patched_v[repaired_rows, repaired_remove_pos] = topm_v[repaired_rows, repaired_add_pos]

        patched_counts = node_counts.clone()
        patched_counts[repaired_rows, repaired_removed_nodes] -= 1
        patched_counts[repaired_rows, repaired_added_nodes] += 1
        unique_after = (patched_counts > 0).sum(dim=1)
        events_after_vec = (unique_after - 1).clamp_min(0)
        cross_items_after_vec = K - patched_counts.max(dim=1).values
        after_remote_packet_counts = (patched_counts > 0)[:, remote_node_mask].sum(dim=0)
        runtime_packet_events_after = int(after_remote_packet_counts.gt(0).sum().item())
        runtime_packet_events_eliminated = max(0, runtime_packet_events_before - runtime_packet_events_after)

        repaired_margin_logit = margin_logit[repaired_rows, repaired_remove_pos, repaired_add_pos].clamp_min(0.0)
        repaired_margin_prob = margin_prob[repaired_rows, repaired_remove_pos, repaired_add_pos].clamp_min(0.0)

        replacement_pair_hist: Dict[str, int] = {}
        removed_expert_hist: Dict[str, int] = {}
        added_expert_hist: Dict[str, int] = {}
        collect_hist = env_flag_with_alias("SPARSE_ERS_COLLECT_HIST", "SPARSE_NODEPATCH_COLLECT_HIST", "0")
        if collect_hist and int(repaired_rows.numel()) > 0:
            removed_cpu = topi[repaired_rows, repaired_remove_pos].detach().to("cpu", dtype=torch.long).tolist()
            added_cpu = topm_i[repaired_rows, repaired_add_pos].detach().to("cpu", dtype=torch.long).tolist()
            layer_key = f"L{int(self.layer_idx)}"
            for removed_e, added_e in zip(removed_cpu, added_cpu):
                pair_key = f"{layer_key}:{int(removed_e)}->{int(added_e)}"
                replacement_pair_hist[pair_key] = int(replacement_pair_hist.get(pair_key, 0)) + 1
                rem_key = f"{layer_key}:{int(removed_e)}"
                add_key = f"{layer_key}:{int(added_e)}"
                removed_expert_hist[rem_key] = int(removed_expert_hist.get(rem_key, 0)) + 1
                added_expert_hist[add_key] = int(added_expert_hist.get(add_key, 0)) + 1

        removed_position_hist: Dict[str, int] = {}
        pos_counts = torch.bincount(repaired_remove_pos.detach().to(torch.long), minlength=K)
        for pos, cnt in enumerate(pos_counts.detach().to("cpu", dtype=torch.long).tolist()):
            if int(cnt) > 0:
                removed_position_hist[str(int(pos))] = int(cnt)
        layer_repaired_token_hist = {str(int(self.layer_idx)): int(repaired_rows.numel())}

        if self.comm_rank == 0:
            SparseContext.record_nodepatch(
                tokens=T,
                cross_node_tokens=int(cross_token_mask.sum().item()),
                repaired_tokens=int(repaired_rows.numel()),
                replacement_count=int(repaired_rows.numel()),
                event_before=int(events_before_vec.sum().item()),
                event_after=int(events_after_vec.sum().item()),
                cross_items_before=int(cross_items_before_vec.sum().item()),
                cross_items_after=int(cross_items_after_vec.sum().item()),
                unique_nodes_before_sum=float(unique_before.sum().item()),
                unique_nodes_after_sum=float(unique_after.sum().item()),
                utility_loss_logit=float(repaired_margin_logit.sum().item()),
                utility_loss_prob=float(repaired_margin_prob.sum().item()),
                max_margin_logit_sum=float(repaired_margin_logit.sum().item()),
                max_margin_prob_sum=float(repaired_margin_prob.sum().item()),
                max_margin_logit_max=float(repaired_margin_logit.max().item()) if int(repaired_rows.numel()) > 0 else 0.0,
                max_margin_prob_max=float(repaired_margin_prob.max().item()) if int(repaired_rows.numel()) > 0 else 0.0,
                runtime_packet_events_before=runtime_packet_events_before,
                runtime_packet_events_after=runtime_packet_events_after,
                runtime_packet_events_eliminated=runtime_packet_events_eliminated,
                event_aware_skipped_repairs=float(event_aware_skipped_repairs),
                replacement_pair_hist=replacement_pair_hist,
                removed_expert_hist=removed_expert_hist,
                added_expert_hist=added_expert_hist,
                removed_position_hist=removed_position_hist,
                layer_repaired_token_hist=layer_repaired_token_hist,
            )
        return patched_v, patched_i

    def _apply_nodepatch_event_aware_repair_packetfirst(
        self,
        topv: torch.Tensor,
        topi: torch.Tensor,
        topm_v: torch.Tensor,
        topm_i: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        max_replace: int,
        margin_tau: float,
        margin_kind: str,
        keep_top1: bool,
        only_cross_node: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packet-emptying-first, cache-aware NodePatch pilot.

        The previous aggressive runtime path built candidate repairs for many
        token rows and only later checked whether a full remote-node packet would
        disappear. This path reverses that order: only small outgoing packets are
        considered, and every row in a packet must be repairable before any route
        is changed. With SPARSE_NODEPATCH_CACHE_AWARE=1, added experts must be
        local-owner and resident by default, avoiding extra demand H2D.
        """
        K = int(self.top_k)
        M = int(topm_i.shape[-1])
        T = int(topi.shape[0])
        if T <= 0 or K <= 1 or M <= K or int(max_replace) <= 0:
            return topv, topi

        node_ids = self._comm_rank_node_ids()
        my_node = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
        try:
            packet_max_items = int(env_value_with_alias("SPARSE_ERS_EVENT_PACKET_MAX_ITEMS", "SPARSE_NODEPATCH_EVENT_PACKET_MAX_ITEMS", "4") or 4)
        except Exception:
            packet_max_items = 4
        if packet_max_items <= 0:
            packet_max_items = 10**9

        require_local_candidate = env_flag_with_alias("SPARSE_ERS_EVENT_REQUIRE_LOCAL_CANDIDATE", "SPARSE_NODEPATCH_EVENT_REQUIRE_LOCAL_CANDIDATE", "1")
        only_empty_packet = env_flag_with_alias("SPARSE_ERS_EVENT_ONLY_EMPTY_PACKET", "SPARSE_NODEPATCH_EVENT_ONLY_EMPTY_PACKET", "1")
        cache_aware = self._nodepatch_cache_aware_enabled()
        disable_margin = self._nodepatch_margin_disabled(margin_tau)
        remove_start = 1 if bool(keep_top1) else 0

        idx_cpu = topi.detach().to("cpu", dtype=torch.long)
        expert_to_node = [int(x) for x in self.expert_to_node_cpu]
        expert_to_rank = [int(x) for x in getattr(self, "expert_to_rank_cpu", [])]
        if len(expert_to_rank) < int(self.num_experts):
            expert_to_rank = [int(x) for x in self.expert_to_rank.detach().to("cpu", dtype=torch.long).tolist()]
        resident_local = self._nodepatch_local_resident_set() if bool(cache_aware) else set()

        tokens = T
        cross_tokens = 0
        replacements = 0
        events_before = 0
        events_after = 0
        cross_items_before = 0
        cross_items_after = 0
        unique_nodes_before_sum = 0.0
        unique_nodes_after_sum = 0.0
        util_logit_sum = 0.0
        util_prob_sum = 0.0
        max_margin_logit_sum = 0.0
        max_margin_prob_sum = 0.0
        max_margin_logit_max = 0.0
        max_margin_prob_max = 0.0
        event_aware_skipped_repairs = 0
        cache_aware_skipped_repairs = 0
        cache_aware_filtered_candidates = 0
        packet_emptying_candidate_packets = 0.0
        packet_emptying_feasible_packets = 0.0
        packet_emptying_rejected_packets = 0.0
        replacement_pair_hist: Dict[str, int] = {}
        removed_expert_hist: Dict[str, int] = {}
        added_expert_hist: Dict[str, int] = {}
        removed_position_hist: Dict[str, int] = {}
        layer_repaired_token_hist: Dict[str, int] = {}

        def unique_nodes(experts: List[int]) -> int:
            return len(set(int(expert_to_node[int(e)]) for e in experts))

        def cross_items(experts: List[int]) -> int:
            counts: Dict[int, int] = {}
            for e in experts:
                q = int(expert_to_node[int(e)])
                counts[q] = int(counts.get(q, 0)) + 1
            return max(0, len(experts) - (max(counts.values()) if counts else 0))

        def candidate_cache_ok(eid: int) -> bool:
            if not bool(cache_aware):
                return True
            eid = int(eid)
            if self._nodepatch_cache_require_owner_local() and int(expert_to_rank[eid]) != int(self.comm_rank):
                return False
            if self._nodepatch_cache_require_resident() and eid not in resident_local:
                return False
            return True

        row_original: Dict[int, List[int]] = {}
        row_nodes: Dict[int, List[int]] = {}
        remote_rows: Dict[int, List[int]] = {}
        for row in range(T):
            experts = [int(x) for x in idx_cpu[row].tolist()]
            nodes = [int(expert_to_node[int(e)]) for e in experts]
            row_original[int(row)] = experts
            row_nodes[int(row)] = nodes
            u = len(set(nodes))
            unique_nodes_before_sum += float(u)
            events_before += max(0, int(u) - 1)
            ci = cross_items(experts)
            cross_items_before += int(ci)
            if u <= 1:
                continue
            cross_tokens += 1
            if bool(only_cross_node) and u <= 1:
                continue
            for q in sorted(set(nodes)):
                if int(q) != int(my_node):
                    remote_rows.setdefault(int(q), []).append(int(row))

        runtime_packet_events_before = sum(1 for rows in remote_rows.values() if rows)
        runtime_packet_events_after = int(runtime_packet_events_before)
        packet_emptying_candidate_packets = float(runtime_packet_events_before)
        eligible_targets: List[Tuple[int, List[int]]] = []
        for target_node, rows in sorted(remote_rows.items(), key=lambda kv: (len(kv[1]), kv[0])):
            if bool(only_empty_packet) and len(rows) > int(packet_max_items):
                packet_emptying_rejected_packets += 1.0
                continue
            eligible_targets.append((int(target_node), list(rows)))

        def record_no_patch() -> None:
            SparseContext.record_nodepatch(
                tokens=tokens,
                cross_node_tokens=cross_tokens,
                repaired_tokens=0,
                replacement_count=0,
                event_before=events_before,
                event_after=events_before,
                cross_items_before=cross_items_before,
                cross_items_after=cross_items_before,
                unique_nodes_before_sum=unique_nodes_before_sum,
                unique_nodes_after_sum=unique_nodes_before_sum,
                utility_loss_logit=0.0,
                utility_loss_prob=0.0,
                max_margin_logit_sum=0.0,
                max_margin_prob_sum=0.0,
                max_margin_logit_max=0.0,
                max_margin_prob_max=0.0,
                runtime_packet_events_before=runtime_packet_events_before,
                runtime_packet_events_after=runtime_packet_events_before,
                runtime_packet_events_eliminated=0.0,
                event_aware_skipped_repairs=float(event_aware_skipped_repairs),
                cache_aware_skipped_repairs=float(cache_aware_skipped_repairs),
                cache_aware_filtered_candidates=float(cache_aware_filtered_candidates),
                packet_emptying_candidate_packets=float(packet_emptying_candidate_packets),
                packet_emptying_feasible_packets=float(packet_emptying_feasible_packets),
                packet_emptying_rejected_packets=float(packet_emptying_rejected_packets),
            )

        if not eligible_targets:
            record_no_patch()
            return topv, topi

        # Scores are synchronized only after we know at least one small packet can
        # potentially be targeted.
        topm_i_cpu = topm_i.detach().to("cpu", dtype=torch.long)
        topm_v_cpu = topm_v.detach().to("cpu", dtype=torch.float32)
        topm_z_cpu = router_logits.gather(1, topm_i).detach().to("cpu", dtype=torch.float32)

        packet_patch: Dict[int, List[Tuple[int, List[int], Tuple[int, ...], Tuple[int, ...], Dict[int, float], float, float, float, float]]] = {}
        for target_node, rows in eligible_targets:
            payloads: List[Tuple[int, List[int], Tuple[int, ...], Tuple[int, ...], Dict[int, float], float, float, float, float]] = []
            feasible_packet = True
            packet_cache_filtered = 0
            for row in rows:
                experts = list(row_original[int(row)])
                nodes = list(row_nodes[int(row)])
                tm = [int(x) for x in topm_i_cpu[row].tolist()]
                tv = [float(x) for x in topm_v_cpu[row].tolist()]
                tz = [float(x) for x in topm_z_cpu[row].tolist()]
                topk_set = set(experts)
                prob_map = {int(e): float(v) for e, v in zip(tm, tv)}
                logit_map = {int(e): float(z) for e, z in zip(tm, tz)}
                remove_positions = [
                    int(pos)
                    for pos, q in enumerate(nodes)
                    if int(q) == int(target_node) and int(pos) >= int(remove_start)
                ]
                if not remove_positions or len(remove_positions) > int(max_replace):
                    feasible_packet = False
                    break
                if any(int(pos) < int(remove_start) for pos, q in enumerate(nodes) if int(q) == int(target_node)):
                    feasible_packet = False
                    break

                candidates: List[int] = []
                for e in tm:
                    e = int(e)
                    if e in topk_set:
                        continue
                    q = int(expert_to_node[e])
                    if q == int(target_node):
                        continue
                    if bool(require_local_candidate) and q != int(my_node):
                        continue
                    if not candidate_cache_ok(e):
                        packet_cache_filtered += 1
                        continue
                    candidates.append(e)

                need = len(remove_positions)
                if len(candidates) < need:
                    feasible_packet = False
                    break

                removed = [int(experts[p]) for p in remove_positions]
                removed_sorted = sorted(removed, key=lambda e: logit_map.get(int(e), 0.0), reverse=True)
                best_payload = None
                best_key = None
                import itertools
                for add_combo in itertools.combinations(candidates, need):
                    add_sorted = sorted((int(e) for e in add_combo), key=lambda e: logit_map.get(int(e), 0.0), reverse=True)
                    margins_logit = [
                        float(logit_map.get(int(r), 0.0) - logit_map.get(int(a), 0.0))
                        for r, a in zip(removed_sorted, add_sorted)
                    ]
                    margins_prob = [
                        float(prob_map.get(int(r), 0.0) - prob_map.get(int(a), 0.0))
                        for r, a in zip(removed_sorted, add_sorted)
                    ]
                    max_ml = max(margins_logit) if margins_logit else 0.0
                    max_mp = max(margins_prob) if margins_prob else 0.0
                    margin_value = max_mp if str(margin_kind).lower() == "prob" else max_ml
                    if (not disable_margin) and float(margin_value) > float(margin_tau):
                        continue
                    patched = list(experts)
                    for pos, add_e in zip(remove_positions, add_combo):
                        patched[int(pos)] = int(add_e)
                    if len(set(patched)) != len(patched):
                        continue
                    if any(int(expert_to_node[int(e)]) == int(target_node) for e in patched):
                        continue
                    util_l = max(0.0, float(sum(margins_logit)))
                    util_p = max(0.0, float(sum(margins_prob)))
                    key = (-float(util_l), -float(util_p), -int(len(add_combo)), -sum(int(e) for e in add_combo))
                    if best_key is None or key > best_key:
                        value_updates = {int(pos): float(prob_map.get(int(add_e), 0.0)) for pos, add_e in zip(remove_positions, add_combo)}
                        best_key = key
                        best_payload = (
                            int(row),
                            patched,
                            tuple(int(p) for p in remove_positions),
                            tuple(int(e) for e in add_combo),
                            value_updates,
                            util_l,
                            util_p,
                            max(0.0, float(max_ml)),
                            max(0.0, float(max_mp)),
                        )
                if best_payload is None:
                    feasible_packet = False
                    break
                payloads.append(best_payload)
            cache_aware_filtered_candidates += int(packet_cache_filtered)
            if feasible_packet and len(payloads) == len(rows):
                packet_emptying_feasible_packets += 1.0
                packet_patch[int(target_node)] = payloads
            else:
                packet_emptying_rejected_packets += 1.0
                if packet_cache_filtered > 0:
                    cache_aware_skipped_repairs += len(rows)
                else:
                    event_aware_skipped_repairs += len(payloads)

        if not packet_patch:
            record_no_patch()
            return topv, topi

        patched_experts: List[List[int]] = [[int(x) for x in row_original[row]] for row in range(T)]
        patched_value_updates: Dict[Tuple[int, int], float] = {}
        applied_rows: set[int] = set()
        collect_hist = env_flag_with_alias("SPARSE_ERS_COLLECT_HIST", "SPARSE_NODEPATCH_COLLECT_HIST", "0")
        for target_node, payloads in sorted(packet_patch.items(), key=lambda kv: (len(kv[1]), kv[0])):
            for payload in payloads:
                row, patched, pos_combo, add_combo, value_updates, util_l, util_p, max_ml, max_mp = payload
                old_experts = list(patched_experts[int(row)])
                patched_experts[int(row)] = list(patched)
                applied_rows.add(int(row))
                replacements += len(add_combo)
                util_logit_sum += float(util_l)
                util_prob_sum += float(util_p)
                max_margin_logit_sum += float(max_ml)
                max_margin_prob_sum += float(max_mp)
                max_margin_logit_max = max(max_margin_logit_max, float(max_ml))
                max_margin_prob_max = max(max_margin_prob_max, float(max_mp))
                layer_key = str(int(self.layer_idx))
                layer_repaired_token_hist[layer_key] = int(layer_repaired_token_hist.get(layer_key, 0)) + 1
                for p, add_e in zip(pos_combo, add_combo):
                    patched_value_updates[(int(row), int(p))] = float(value_updates[int(p)])
                    if collect_hist:
                        removed_e = int(old_experts[int(p)])
                        pair_key = f"L{int(self.layer_idx)}:{removed_e}->{int(add_e)}"
                        replacement_pair_hist[pair_key] = int(replacement_pair_hist.get(pair_key, 0)) + 1
                        removed_expert_hist[f"L{int(self.layer_idx)}:{removed_e}"] = int(
                            removed_expert_hist.get(f"L{int(self.layer_idx)}:{removed_e}", 0)
                        ) + 1
                        added_expert_hist[f"L{int(self.layer_idx)}:{int(add_e)}"] = int(
                            added_expert_hist.get(f"L{int(self.layer_idx)}:{int(add_e)}", 0)
                        ) + 1
                        removed_position_hist[str(int(p))] = int(removed_position_hist.get(str(int(p)), 0)) + 1
            runtime_packet_events_after -= 1

        for row in range(T):
            experts = [int(e) for e in patched_experts[row]]
            unique_nodes_after_sum += float(unique_nodes(experts))
            events_after += max(0, unique_nodes(experts) - 1)
            cross_items_after += cross_items(experts)

        patched_i = topi.clone()
        patched_v = topv.clone()
        for row in sorted(applied_rows):
            patched_i[int(row)] = torch.tensor(patched_experts[int(row)], device=topi.device, dtype=topi.dtype)
        for (row, pos), val in patched_value_updates.items():
            patched_v[int(row), int(pos)] = torch.tensor(float(val), device=topv.device, dtype=topv.dtype)

        SparseContext.record_nodepatch(
            tokens=tokens,
            cross_node_tokens=cross_tokens,
            repaired_tokens=len(applied_rows),
            replacement_count=replacements,
            event_before=events_before,
            event_after=events_after,
            cross_items_before=cross_items_before,
            cross_items_after=cross_items_after,
            unique_nodes_before_sum=unique_nodes_before_sum,
            unique_nodes_after_sum=unique_nodes_after_sum,
            utility_loss_logit=util_logit_sum,
            utility_loss_prob=util_prob_sum,
            max_margin_logit_sum=max_margin_logit_sum,
            max_margin_prob_sum=max_margin_prob_sum,
            max_margin_logit_max=max_margin_logit_max,
            max_margin_prob_max=max_margin_prob_max,
            runtime_packet_events_before=runtime_packet_events_before,
            runtime_packet_events_after=runtime_packet_events_after,
            runtime_packet_events_eliminated=max(0, runtime_packet_events_before - runtime_packet_events_after),
            event_aware_skipped_repairs=float(event_aware_skipped_repairs),
            cache_aware_skipped_repairs=float(cache_aware_skipped_repairs),
            cache_aware_filtered_candidates=float(cache_aware_filtered_candidates),
            packet_emptying_candidate_packets=float(packet_emptying_candidate_packets),
            packet_emptying_feasible_packets=float(packet_emptying_feasible_packets),
            packet_emptying_rejected_packets=float(packet_emptying_rejected_packets),
            replacement_pair_hist=replacement_pair_hist,
            removed_expert_hist=removed_expert_hist,
            added_expert_hist=added_expert_hist,
            removed_position_hist=removed_position_hist,
            layer_repaired_token_hist=layer_repaired_token_hist,
        )
        return patched_v, patched_i

    def _apply_nodepatch_event_aware_repair_aggressive(
        self,
        topv: torch.Tensor,
        topi: torch.Tensor,
        topm_v: torch.Tensor,
        topm_i: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        max_replace: int,
        margin_tau: float,
        margin_kind: str,
        keep_top1: bool,
        only_cross_node: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packet-targeted NodePatch for aggressive decode experiments.

        The tensor fastpath can only replace one expert per token. That misses
        tokens with two or more experts on the remote node. This experimental
        path targets the actual outgoing node packet: for a small remote packet,
        it applies all required per-token repairs only if the whole packet can
        be emptied. It is intentionally guarded by
        SPARSE_NODEPATCH_EVENT_AGGRESSIVE and is not a default path.
        """
        K = int(self.top_k)
        M = int(topm_i.shape[-1])
        T = int(topi.shape[0])
        if T <= 0 or K <= 1 or M <= K or int(max_replace) <= 0:
            return topv, topi

        node_ids = self._comm_rank_node_ids()
        my_node = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
        try:
            packet_max_items = int(env_value_with_alias("SPARSE_ERS_EVENT_PACKET_MAX_ITEMS", "SPARSE_NODEPATCH_EVENT_PACKET_MAX_ITEMS", "4") or 4)
        except Exception:
            packet_max_items = 4
        if packet_max_items <= 0:
            packet_max_items = 10**9
        require_local_candidate = env_flag_with_alias("SPARSE_ERS_EVENT_REQUIRE_LOCAL_CANDIDATE", "SPARSE_NODEPATCH_EVENT_REQUIRE_LOCAL_CANDIDATE", "1")
        only_empty_packet = env_flag_with_alias("SPARSE_ERS_EVENT_ONLY_EMPTY_PACKET", "SPARSE_NODEPATCH_EVENT_ONLY_EMPTY_PACKET", "1")
        disable_margin = self._nodepatch_margin_disabled(margin_tau)
        remove_start = 1 if bool(keep_top1) else 0

        idx_cpu = topi.detach().to("cpu", dtype=torch.long)
        val_cpu = topv.detach().to("cpu", dtype=torch.float32)
        topm_i_cpu = topm_i.detach().to("cpu", dtype=torch.long)
        topm_v_cpu = topm_v.detach().to("cpu", dtype=torch.float32)
        topm_z_cpu = router_logits.gather(1, topm_i).detach().to("cpu", dtype=torch.float32)

        expert_to_node = [int(x) for x in self.expert_to_node_cpu]
        patched_experts: List[List[int]] = [[int(x) for x in idx_cpu[row].tolist()] for row in range(T)]
        patched_values: List[List[float]] = [[float(x) for x in val_cpu[row].tolist()] for row in range(T)]

        tokens = T
        cross_tokens = 0
        repaired_tokens_set: set[int] = set()
        replacements = 0
        events_before = 0
        events_after = 0
        cross_items_before = 0
        cross_items_after = 0
        unique_nodes_before_sum = 0.0
        unique_nodes_after_sum = 0.0
        util_logit_sum = 0.0
        util_prob_sum = 0.0
        max_margin_logit_sum = 0.0
        max_margin_prob_sum = 0.0
        max_margin_logit_max = 0.0
        max_margin_prob_max = 0.0
        event_aware_skipped_repairs = 0
        replacement_pair_hist: Dict[str, int] = {}
        removed_expert_hist: Dict[str, int] = {}
        added_expert_hist: Dict[str, int] = {}
        removed_position_hist: Dict[str, int] = {}
        layer_repaired_token_hist: Dict[str, int] = {}

        def unique_nodes(experts: List[int]) -> int:
            return len(set(int(expert_to_node[int(e)]) for e in experts))

        def cross_items(experts: List[int]) -> int:
            counts: Dict[int, int] = {}
            for e in experts:
                q = int(expert_to_node[int(e)])
                counts[q] = int(counts.get(q, 0)) + 1
            return max(0, len(experts) - (max(counts.values()) if counts else 0))

        remote_rows: Dict[int, List[int]] = {}
        row_patch: Dict[Tuple[int, int], Tuple[List[int], List[float], Tuple[int, ...], Tuple[int, ...], float, float, float, float]] = {}
        for row in range(T):
            experts = [int(x) for x in idx_cpu[row].tolist()]
            nodes = [int(expert_to_node[int(e)]) for e in experts]
            u = len(set(nodes))
            c = cross_items(experts)
            unique_nodes_before_sum += float(u)
            events_before += max(0, int(u) - 1)
            cross_items_before += int(c)
            if u <= 1:
                continue
            cross_tokens += 1
            if bool(only_cross_node) and u <= 1:
                continue
            for q in sorted(set(nodes)):
                if int(q) == int(my_node):
                    continue
                remote_rows.setdefault(int(q), []).append(row)

            tm = [int(x) for x in topm_i_cpu[row].tolist()]
            tv = [float(x) for x in topm_v_cpu[row].tolist()]
            tz = [float(x) for x in topm_z_cpu[row].tolist()]
            topk_set = set(experts)
            prob_map = {int(e): float(v) for e, v in zip(tm, tv)}
            logit_map = {int(e): float(z) for e, z in zip(tm, tz)}
            for target_node in sorted(set(q for q in nodes if int(q) != int(my_node))):
                remove_positions = [
                    int(pos)
                    for pos, q in enumerate(nodes)
                    if int(q) == int(target_node) and int(pos) >= int(remove_start)
                ]
                if not remove_positions or len(remove_positions) > int(max_replace):
                    continue
                if any(int(pos) < int(remove_start) for pos, q in enumerate(nodes) if int(q) == int(target_node)):
                    continue
                candidates = []
                for e in tm:
                    e = int(e)
                    if e in topk_set:
                        continue
                    q = int(expert_to_node[e])
                    if q == int(target_node):
                        continue
                    if bool(require_local_candidate) and q != int(my_node):
                        continue
                    candidates.append(e)
                need = len(remove_positions)
                if len(candidates) < need:
                    continue

                removed = [int(experts[p]) for p in remove_positions]
                removed_sorted = sorted(removed, key=lambda e: logit_map.get(int(e), 0.0), reverse=True)
                best_payload = None
                best_key = None
                # K is small (Qwen1.5-MoE topK=4), so combinations are bounded.
                import itertools
                for add_combo in itertools.combinations(candidates, need):
                    add_sorted = sorted((int(e) for e in add_combo), key=lambda e: logit_map.get(int(e), 0.0), reverse=True)
                    margins_logit = [
                        float(logit_map.get(int(r), 0.0) - logit_map.get(int(a), 0.0))
                        for r, a in zip(removed_sorted, add_sorted)
                    ]
                    margins_prob = [
                        float(prob_map.get(int(r), 0.0) - prob_map.get(int(a), 0.0))
                        for r, a in zip(removed_sorted, add_sorted)
                    ]
                    max_ml = max(margins_logit) if margins_logit else 0.0
                    max_mp = max(margins_prob) if margins_prob else 0.0
                    margin_value = max_mp if str(margin_kind).lower() == "prob" else max_ml
                    if (not disable_margin) and float(margin_value) > float(margin_tau):
                        continue
                    patched = list(experts)
                    patched_vals = [float(x) for x in val_cpu[row].tolist()]
                    for pos, add_e in zip(remove_positions, add_combo):
                        patched[int(pos)] = int(add_e)
                        patched_vals[int(pos)] = float(prob_map.get(int(add_e), 0.0))
                    if len(set(patched)) != len(patched):
                        continue
                    if any(int(expert_to_node[int(e)]) == int(target_node) for e in patched):
                        continue
                    util_l = max(0.0, float(sum(margins_logit)))
                    util_p = max(0.0, float(sum(margins_prob)))
                    key = (-float(util_l), -float(util_p), -sum(int(e) for e in add_combo))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_payload = (
                            patched,
                            patched_vals,
                            tuple(int(p) for p in remove_positions),
                            tuple(int(e) for e in add_combo),
                            util_l,
                            util_p,
                            max(0.0, float(max_ml)),
                            max(0.0, float(max_mp)),
                        )
                if best_payload is not None:
                    row_patch[(row, int(target_node))] = best_payload

        runtime_packet_events_before = sum(1 for rows in remote_rows.values() if rows)
        runtime_packet_events_after = runtime_packet_events_before
        applied_rows: set[int] = set()
        for target_node, rows in sorted(remote_rows.items(), key=lambda kv: (len(kv[1]), kv[0])):
            if int(only_empty_packet) and len(rows) > int(packet_max_items):
                event_aware_skipped_repairs += sum(1 for row in rows if (row, int(target_node)) in row_patch)
                continue
            if not all((row, int(target_node)) in row_patch for row in rows):
                event_aware_skipped_repairs += sum(1 for row in rows if (row, int(target_node)) in row_patch)
                continue
            for row in rows:
                payload = row_patch[(row, int(target_node))]
                patched, patched_vals, pos_combo, add_combo, util_l, util_p, max_ml, max_mp = payload
                old_experts = patched_experts[row]
                patched_experts[row] = list(patched)
                patched_values[row] = list(patched_vals)
                applied_rows.add(int(row))
                replacements += len(add_combo)
                util_logit_sum += float(util_l)
                util_prob_sum += float(util_p)
                max_margin_logit_sum += float(max_ml)
                max_margin_prob_sum += float(max_mp)
                max_margin_logit_max = max(max_margin_logit_max, float(max_ml))
                max_margin_prob_max = max(max_margin_prob_max, float(max_mp))
                layer_key = str(int(self.layer_idx))
                layer_repaired_token_hist[layer_key] = int(layer_repaired_token_hist.get(layer_key, 0)) + 1
                for p, add_e in zip(pos_combo, add_combo):
                    removed_e = int(old_experts[int(p)])
                    pair_key = f"L{int(self.layer_idx)}:{removed_e}->{int(add_e)}"
                    replacement_pair_hist[pair_key] = int(replacement_pair_hist.get(pair_key, 0)) + 1
                    removed_expert_hist[f"L{int(self.layer_idx)}:{removed_e}"] = int(
                        removed_expert_hist.get(f"L{int(self.layer_idx)}:{removed_e}", 0)
                    ) + 1
                    added_expert_hist[f"L{int(self.layer_idx)}:{int(add_e)}"] = int(
                        added_expert_hist.get(f"L{int(self.layer_idx)}:{int(add_e)}", 0)
                    ) + 1
                    removed_position_hist[str(int(p))] = int(removed_position_hist.get(str(int(p)), 0)) + 1
            runtime_packet_events_after -= 1

        if not applied_rows:
            for row in range(T):
                experts = [int(x) for x in idx_cpu[row].tolist()]
                unique_nodes_after_sum += float(unique_nodes(experts))
                events_after += max(0, unique_nodes(experts) - 1)
                cross_items_after += cross_items(experts)
            if self.comm_rank == 0:
                SparseContext.record_nodepatch(
                    tokens=tokens,
                    cross_node_tokens=cross_tokens,
                    repaired_tokens=0,
                    replacement_count=0,
                    event_before=events_before,
                    event_after=events_after,
                    cross_items_before=cross_items_before,
                    cross_items_after=cross_items_after,
                    unique_nodes_before_sum=unique_nodes_before_sum,
                    unique_nodes_after_sum=unique_nodes_after_sum,
                    utility_loss_logit=0.0,
                    utility_loss_prob=0.0,
                    max_margin_logit_sum=0.0,
                    max_margin_prob_sum=0.0,
                    max_margin_logit_max=0.0,
                    max_margin_prob_max=0.0,
                    runtime_packet_events_before=runtime_packet_events_before,
                    runtime_packet_events_after=runtime_packet_events_before,
                    runtime_packet_events_eliminated=0.0,
                    event_aware_skipped_repairs=float(event_aware_skipped_repairs),
                )
            return topv, topi

        for row in range(T):
            experts = [int(e) for e in patched_experts[row]]
            unique_nodes_after_sum += float(unique_nodes(experts))
            events_after += max(0, unique_nodes(experts) - 1)
            cross_items_after += cross_items(experts)

        patched_i = torch.tensor(patched_experts, device=topi.device, dtype=topi.dtype)
        patched_v = torch.tensor(patched_values, device=topv.device, dtype=topv.dtype)
        if self.comm_rank == 0:
            SparseContext.record_nodepatch(
                tokens=tokens,
                cross_node_tokens=cross_tokens,
                repaired_tokens=len(applied_rows),
                replacement_count=replacements,
                event_before=events_before,
                event_after=events_after,
                cross_items_before=cross_items_before,
                cross_items_after=cross_items_after,
                unique_nodes_before_sum=unique_nodes_before_sum,
                unique_nodes_after_sum=unique_nodes_after_sum,
                utility_loss_logit=util_logit_sum,
                utility_loss_prob=util_prob_sum,
                max_margin_logit_sum=max_margin_logit_sum,
                max_margin_prob_sum=max_margin_prob_sum,
                max_margin_logit_max=max_margin_logit_max,
                max_margin_prob_max=max_margin_prob_max,
                runtime_packet_events_before=runtime_packet_events_before,
                runtime_packet_events_after=runtime_packet_events_after,
                runtime_packet_events_eliminated=max(0, runtime_packet_events_before - runtime_packet_events_after),
                event_aware_skipped_repairs=float(event_aware_skipped_repairs),
                replacement_pair_hist=replacement_pair_hist,
                removed_expert_hist=removed_expert_hist,
                added_expert_hist=added_expert_hist,
                removed_position_hist=removed_position_hist,
                layer_repaired_token_hist=layer_repaired_token_hist,
            )
        return patched_v, patched_i

    def _shared_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is not None:
            return self.shared_experts(x)
        if self.shared_expert is None:
            return torch.zeros_like(x)
        shared = self.shared_expert(x)
        if self.shared_expert_gate is not None:
            shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
        return shared

    def _hidden_size_from_modules(self, fallback: int = 0) -> int:
        out = int(getattr(self, "hidden_size", 0) or fallback or 0)
        if out > 0:
            return out
        out = int(getattr(self.gate, "in_features", 0) or 0)
        if out > 0:
            return out
        gate_weight = getattr(self.gate, "weight", None)
        try:
            out = int(gate_weight.shape[-1])
        except Exception:
            out = 0
        if out > 0:
            return out
        if self.shared_expert_gate is not None:
            out = int(getattr(self.shared_expert_gate, "in_features", 0) or 0)
            if out > 0:
                return out
        return int(fallback or 0)

    def _return_tensor_only_mlp(self) -> bool:
        # DeepSeek remote decoder layers call `hidden_states = self.mlp(...)`.
        # Qwen-MoE layers in this harness consume `(hidden_states, router_logits)`.
        return str(getattr(self, "moe_family", "")) == "deepseek_shared"

    def _routing_router_logits_from_weights(self, idx: torch.Tensor, weights: torch.Tensor, tokens: int) -> torch.Tensor:
        router_logits = torch.full(
            (int(tokens), int(self.num_experts)),
            float("-inf"),
            device=weights.device,
            dtype=torch.float32,
        )
        safe_w = weights.to(torch.float32).clamp_min(1e-12)
        router_logits.scatter_(1, idx.to(torch.long), safe_w.log())
        return router_logits

      # ---------------- router routing ----------------
    def _routing_router(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if str(getattr(self, "moe_family", "")) == "deepseek_shared":
            bsz = int(getattr(self, "_current_forward_batch_size", 1) or 1)
            seq_len = int(getattr(self, "_current_forward_seq_len", int(x.shape[0])) or int(x.shape[0]))
            if bsz * seq_len != int(x.shape[0]):
                bsz, seq_len = 1, int(x.shape[0])
            router_input = x.view(bsz, seq_len, int(x.shape[-1]))
            route_out = self.gate(router_input)
            if not isinstance(route_out, (tuple, list)) or len(route_out) < 2:
                raise RuntimeError("DeepSeek-style MoE gate must return (topk_idx, topk_weight, ...)")
            topi = route_out[0].to(torch.long).reshape(int(x.shape[0]), -1).contiguous()
            topv = route_out[1].to(dtype=x.dtype).reshape(int(x.shape[0]), -1).contiguous()
            if int(topi.shape[-1]) != int(self.top_k):
                self.top_k = int(topi.shape[-1])
            router_logits = self._routing_router_logits_from_weights(topi, topv, int(x.shape[0]))
            return topv, topi, router_logits

        router_logits = self.gate(x)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        if self._nodepatch_phase_enabled():
            top_m, max_replace, margin_tau, margin_kind, keep_top1, only_cross = self._nodepatch_config()
            topm_v, topm_i = torch.topk(probs, k=top_m, dim=-1)
            topv = topm_v[:, :self.top_k].contiguous()
            topi = topm_i[:, :self.top_k].contiguous()
            if self._nodepatch_action() == "drop":
                topv, topi = self._apply_nodepatch_route_drop(
                    topv,
                    topi,
                    max_replace=max_replace,
                    margin_tau=margin_tau,
                    margin_kind=margin_kind,
                    keep_top1=keep_top1,
                    only_cross_node=only_cross,
                )
            elif self._nodepatch_event_aware_enabled() and self._nodepatch_event_aggressive_enabled():
                repair_fn = (
                    self._apply_nodepatch_event_aware_repair_packetfirst
                    if self._nodepatch_packet_emptying_first_enabled()
                    else self._apply_nodepatch_event_aware_repair_aggressive
                )
                topv, topi = repair_fn(
                    topv,
                    topi,
                    topm_v,
                    topm_i,
                    router_logits,
                    max_replace=max_replace,
                    margin_tau=margin_tau,
                    margin_kind=margin_kind,
                    keep_top1=keep_top1,
                    only_cross_node=only_cross,
                )
            elif self._nodepatch_lean_repair_enabled():
                topv, topi = self._apply_nodepatch_route_repair_lean(
                    topv,
                    topi,
                    topm_v,
                    topm_i,
                    max_replace=max_replace,
                    keep_top1=keep_top1,
                    only_cross_node=only_cross,
                )
            elif self._nodepatch_fastpath_enabled() and int(max_replace) == 1:
                topv, topi = self._apply_nodepatch_route_repair_fastpath(
                    topv,
                    topi,
                    topm_v,
                    topm_i,
                    router_logits,
                    max_replace=max_replace,
                    margin_tau=margin_tau,
                    margin_kind=margin_kind,
                    keep_top1=keep_top1,
                    only_cross_node=only_cross,
                )
            else:
                topv, topi = self._apply_nodepatch_route_repair(
                    topv,
                    topi,
                    topm_v,
                    topm_i,
                    router_logits,
                    max_replace=max_replace,
                    margin_tau=margin_tau,
                    margin_kind=margin_kind,
                    keep_top1=keep_top1,
                    only_cross_node=only_cross,
                )
        else:
            topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topv = topv.to(dtype=x.dtype)
        return topv, topi.to(torch.long), router_logits

    # ---------------- RS planner + routing ----------------
    def _finalize_completed_prefetch_events(self, *, count_all: bool = False) -> None:
        """Finalize outstanding prefetch CUDA events for accounting.

        `prefetch_completed_count` used to stay at zero while
        `prefetch_complete_before_use_count` was positive, because completion was
        only queried in the demand path.  Count queried-complete events here and
        in `_maybe_wait_prefetch`; remove events only when they are known complete
        or when the caller is force-clearing request state.
        """
        if not getattr(self, "_prefetch_events", None):
            return
        for eid, ev in list(self._prefetch_events.items()):
            done = False
            try:
                done = bool(ev.query())
            except Exception:
                done = bool(count_all)
            if done or count_all:
                self._record_rs_counter("prefetch_completed_count", 1)
                self._prefetch_events.pop(eid, None)

    def _finalize_prev_group_prefetch(self) -> None:
        self._finalize_completed_prefetch_events(count_all=False)
        wasted = self._prefetched_ids_this_group - self._prefetch_used_ids_this_group
        self._record_prefetch_wasted(len(wasted))
        transition_wasted = self._transition_prefetched_ids - self._transition_used_ids
        if len(transition_wasted) > 0:
            self._record_metric_counter("transition_prefetch_wasted", len(transition_wasted))
        self._cache.unpin_all()
        self._prefetched_ids_this_group.clear()
        self._prefetch_used_ids_this_group.clear()
        self._transition_prefetched_ids.clear()
        self._transition_used_ids.clear()
        self._prefetch_expire_layer.clear()
        self._prefetch_current_group_start = None
        self._prefetch_current_group_end = None
        self._prefetch_current_cycle_key = None

    def _begin_group_prefetch_cycle(
        self,
        group_start: int = 0,
        group_end: int = 0,
        *,
        cycle_key: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        key = cycle_key if cycle_key is not None else ("layer_group", int(group_start), int(group_end))
        if self._prefetch_current_cycle_key == key:
            return
        self._finalize_prev_group_prefetch()
        self._prefetch_current_group_start = int(group_start)
        self._prefetch_current_group_end = int(group_end)
        self._prefetch_current_cycle_key = key

    def _prefetch_protect_ids(self) -> set[int]:
        mode = str(os.environ.get("SPARSE_PREFETCH_PROTECT_MODE", "default") or "default").strip().lower()
        if mode in ("minimal", "current_next", "current-next"):
            protect = set(int(x) for x in self._prefetch_used_ids_this_group)
        else:
            protect = set(int(x) for x in self._prefetched_ids_this_group) | set(int(x) for x in self._prefetch_used_ids_this_group)
        protect.update(self._cache_protect_ids())
        recent_steps = self._rank_env_int("SPARSE_RS_PREFETCH_RECENT_PROTECT_STEPS_BY_RANK", self._rs_env_int("SPARSE_RS_PREFETCH_RECENT_PROTECT_STEPS", 0))
        if recent_steps > 0 and str(SparseContext.current_phase) == "decode":
            step = int(getattr(SparseContext, "runtime_call_id", -1))
            if step >= 0:
                for eid, last_step in list(getattr(self, "_last_used_decode_step", {}).items()):
                    if step - int(last_step) <= recent_steps:
                        protect.add(int(eid))
        return protect

    def _global_cache_enabled(self) -> bool:
        return bool(getattr(self, "_global_expert_cache_enabled", False)) and bool(SparseContext.global_expert_cache_enabled)

    def _global_cache_resident_count(self) -> int:
        total = 0
        for _, blk in SparseContext.layer_blocks.items():
            try:
                total += len(getattr(blk._cache, "resident_ids")())
            except Exception:
                continue
        return int(total)

    def _global_cache_victim_candidates(self) -> List[Tuple[float, int, int, "DistributedSparseMoeBlock"]]:
        current_layer = SparseContext.current_moe_layer_idx
        out: List[Tuple[float, int, int, "DistributedSparseMoeBlock"]] = []
        for layer_idx, blk in SparseContext.layer_blocks.items():
            if not isinstance(blk, DistributedSparseMoeBlock):
                continue
            # The layer currently entering MoE may need any of its resident
            # experts immediately. Treat it as active/protected.
            if current_layer is not None and int(layer_idx) == int(current_layer):
                continue
            try:
                resident_n = len(getattr(blk._cache, "resident_ids")())
            except Exception:
                resident_n = 0
            try:
                layer_floor = int(os.environ.get("SPARSE_GLOBAL_EXPERT_CACHE_LAYER_FLOOR", "0") or 0)
            except Exception:
                layer_floor = 0
            # Optional safety floor. The default is 0 because fairness here is
            # total physical resident capacity, not a fixed cache8 reservation
            # for every layer. Setting this to 8 reproduces the conservative v2
            # behavior, which is safe but leaves no prefetch room at cap=24*8.
            if layer_floor > 0 and resident_n <= int(layer_floor):
                continue
            try:
                protect = blk._prefetch_protect_ids()
                evictable = blk._prefetch_evictable_ids(protect)
            except Exception:
                continue
            for eid in evictable:
                eid = int(eid)
                value = float(blk._resident_keep_bias(eid))
                # Completed previous layers are cheaper victims than future
                # layers because the current token will not revisit them.
                if current_layer is not None:
                    if int(layer_idx) < int(current_layer):
                        value -= 0.05
                    elif int(layer_idx) > int(current_layer):
                        value += 0.05
                out.append((value, int(layer_idx), eid, blk))
        out.sort(key=lambda x: (x[0], x[1], x[2]))
        return out

    def _global_cache_evict_one(self) -> bool:
        if not self._global_cache_enabled():
            return False
        candidates = self._global_cache_victim_candidates()
        if len(candidates) == 0:
            return False
        _, _, eid, blk = candidates[0]
        blk._discard_cache_entry_before_prefetch(int(eid))
        return True

    def _global_cache_make_room(self, required_bytes: int = 0) -> bool:
        if not self._global_cache_enabled():
            return True
        cap = int(max(1, SparseContext.global_expert_cache_capacity))
        while self._global_cache_resident_count() >= cap:
            if not self._global_cache_evict_one():
                return False
        # If free GPU memory is below the configured headroom, evict additional
        # cross-layer residents before materializing the new expert.
        while not self._prefetch_has_memory_headroom(int(required_bytes)):
            if not self._global_cache_evict_one():
                return False
        return True

    def _global_cache_enforce_after_demand(self) -> None:
        if not self._global_cache_enabled():
            return
        cap = int(max(1, SparseContext.global_expert_cache_capacity))
        while self._global_cache_resident_count() > cap:
            if not self._global_cache_evict_one():
                break

    def _prefetch_free_slots(self) -> int:
        fn = getattr(self._cache, 'prefetch_free_slots', None)
        if callable(fn):
            return int(max(0, fn()))
        return 0

    def _prefetch_evictable_ids(self, protect_ids: Optional[set[int]] = None) -> List[int]:
        fn = getattr(self._cache, 'evictable_prefetch_ids', None)
        if callable(fn):
            return [int(x) for x in fn(protect_ids)]
        return []

    def _prefetch_total_insert_capacity(self, protect_ids: Optional[set[int]] = None) -> int:
        free_slots = self._prefetch_free_slots()
        if not self._prefetch_allow_active_replace:
            return int(free_slots)
        evictable = self._prefetch_evictable_ids(protect_ids)
        return int(free_slots + len(evictable))

    # backward-compatible helpers used by a few older utility paths
    def _effective_prefetch_capacity(self) -> int:
        return self._prefetch_total_insert_capacity(self._prefetch_protect_ids())

    def _prefetch_admission_budget(self) -> int:
        cap = int(self._effective_prefetch_capacity())
        if cap <= 0:
            return 0
        budget = min(int(self._effective_rs_prefetch_top_m()), cap)
        budget -= int(len(self._prefetched_ids_this_group))
        return max(0, budget)

    def _prefetch_target_budget(
        self,
        group_start: int,
        group_end: int,
        *,
        count_metrics: bool = False,
        cycle_key: Optional[Tuple[Any, ...]] = None,
    ) -> int:
        top_m = int(self._effective_rs_prefetch_top_m())
        if self.planner_mode != 'rs_prefetch' or self.expert_offload != 'cpu' or top_m <= 0:
            return 0
        if len(self.local_expert_ids) == 0:
            return 0
        self._begin_group_prefetch_cycle(group_start, group_end, cycle_key=cycle_key)
        protect = self._prefetch_protect_ids()
        budget = min(
            max(0, top_m - int(len(self._prefetched_ids_this_group))),
            self._prefetch_total_insert_capacity(protect),
        )
        budget = max(0, int(budget))
        if count_metrics and self.metrics is not None:
            if budget <= 0:
                self.metrics.prefetch_budget_zero_count += 1
            else:
                self.metrics.prefetch_budget_positive_count += 1
        return budget

    def _prefetch_can_progress(self, group_start: int, group_end: int) -> bool:
        if self.planner_mode != 'rs_prefetch':
            return False
        if self.expert_offload != 'cpu':
            return False
        if self._effective_rs_prefetch_top_m() <= 0:
            return False
        if len(self.local_expert_ids) == 0:
            return False
        self._begin_group_prefetch_cycle(group_start, group_end)
        protect = self._prefetch_protect_ids()
        return self._prefetch_total_insert_capacity(protect) > 0

    def _oracle_async_prefetch_enabled(self) -> bool:
        if self.oracle_mode != "replay":
            return False
        if self.expert_offload != "cpu" or self.prefetch_stream is None:
            return False
        if not _env_flag("SPARSE_ORACLE_ASYNC_PREFETCH", "0"):
            return False
        if int(SparseContext.oracle_call_id) < 0:
            return False
        if str(SparseContext.current_phase) == "prefill" and not _env_flag("SPARSE_ORACLE_PREFETCH_PREFILL", "0"):
            return False
        return True

    def _oracle_stage_current_layer_enabled(self) -> bool:
        if self.oracle_mode != "replay":
            return False
        raw = os.environ.get("SPARSE_ORACLE_STAGE_CURRENT", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        if not self._oracle_async_prefetch_enabled():
            return True
        # For a true overlap upper-bound run, set this to 0. Keeping it enabled
        # preserves the old replay behavior: synchronously stage the current
        # layer's experts from the trace before expert execution.
        return True

    def _oracle_prefetch_top_m(self) -> int:
        raw = os.environ.get("SPARSE_ORACLE_PREFETCH_TOP_M", "").strip()
        if raw == "":
            return int(self.prefetch_top_m)
        try:
            return int(raw)
        except Exception:
            return int(self.prefetch_top_m)

    def _oracle_prefetch_budget(self, group_start: int, group_end: int) -> int:
        if self.expert_offload != "cpu" or self.prefetch_stream is None:
            return 0
        if len(self.local_expert_ids) == 0:
            return 0
        top_m = max(0, int(self._oracle_prefetch_top_m()))
        if top_m <= 0:
            return 0
        cycle_key = (
            "oracle",
            str(SparseContext.current_phase),
            int(SparseContext.oracle_call_id),
            int(group_start),
            int(group_end),
        )
        self._begin_group_prefetch_cycle(group_start, group_end, cycle_key=cycle_key)
        protect = self._prefetch_protect_ids()
        budget = min(
            max(0, top_m - int(len(self._prefetched_ids_this_group))),
            self._prefetch_total_insert_capacity(protect),
        )
        budget = max(0, int(budget))
        if self.metrics is not None:
            if budget <= 0:
                self.metrics.prefetch_budget_zero_count += 1
            else:
                self.metrics.prefetch_budget_positive_count += 1
        return budget

    def _oracle_prefetch_exact_future_layer(self, target_layer: int) -> None:
        if not self._oracle_async_prefetch_enabled():
            return
        target_layer = int(target_layer)
        if target_layer < 0 or target_layer >= int(self.num_layers):
            return
        blk = SparseContext.layer_blocks.get(target_layer, None)
        if blk is None:
            return
        if not isinstance(blk, DistributedSparseMoeBlock):
            return
        if not blk._oracle_async_prefetch_enabled():
            return

        call_id = int(SparseContext.oracle_call_id)
        exact_eids = SparseContext.get_oracle_layer_experts(call_id, target_layer)
        if len(exact_eids) == 0:
            return

        recurring_only = _env_flag("SPARSE_ORACLE_PREFETCH_RECURRING_ONLY", "1")
        score_map: Dict[int, float] = {}
        for pos, eid_raw in enumerate(exact_eids):
            eid = int(eid_raw)
            if eid not in blk._local_eid_set:
                continue
            if recurring_only and int(blk._expert_access_freq.get(eid, 0)) <= 0:
                continue
            # Exact oracle: all listed experts are future-needed. Give earlier
            # entries a tiny tie-breaker only to make admission deterministic.
            score_map[eid] = max(float(score_map.get(eid, 0.0)), 1.0 + 1e-4 * (len(exact_eids) - pos))
        if len(score_map) == 0:
            return

        group_start = target_layer
        group_end = target_layer + 1
        budget = blk._oracle_prefetch_budget(group_start, group_end)
        if budget <= 0:
            return

        protect = blk._prefetch_protect_ids()
        plan = blk._build_prefetch_plan_from_score_map(
            score_map,
            protect_ids=protect,
            budget=budget,
            min_candidate_score=0.0,
            replace_advantage=0.0,
        )
        if len(plan) == 0:
            return
        blk._admit_prefetch_plan(plan, group_end=group_end)

    def _oracle_async_prefetch_future_layers(self) -> None:
        if not self._oracle_async_prefetch_enabled():
            return
        try:
            lead = max(0, int(os.environ.get("SPARSE_ORACLE_PREFETCH_LEAD_LAYERS", "1") or 1))
            window = max(1, int(os.environ.get("SPARSE_ORACLE_PREFETCH_WINDOW_LAYERS", "1") or 1))
        except Exception:
            lead = 1
            window = 1
        start = int(self.layer_idx) + int(lead)
        end = min(int(self.num_layers), start + int(window))
        for target_layer in range(start, end):
            self._oracle_prefetch_exact_future_layer(target_layer)

    def _planner_group_has_any_opportunity(self, group_start: int, group_end: int) -> bool:
        if self.planner_mode != 'rs_prefetch' or self.expert_offload != 'cpu' or self._effective_rs_prefetch_top_m() <= 0:
            return False
        for l in range(int(group_start) + 1, int(group_end)):
            blk = SparseContext.layer_blocks.get(int(l), None)
            if blk is None:
                continue
            try:
                if blk._prefetch_can_progress(int(group_start), int(group_end)):
                    return True
            except Exception:
                continue
        return False

    def _materialize_local_expert_for_prefetch(self, eid: int):
        if eid not in self._local_eid_set:
            raise RuntimeError(f'Prefetch only supports local experts, got eid={eid} at layer={self.layer_idx}')

        if self.experts_kind == 'fused':
            gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
            down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
            self._record_h2d_bytes(tensor_nbytes(gate_up) + tensor_nbytes(down), is_prefetch=True)
            return (gate_up, down)

        gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
        uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
        dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
        self._record_h2d_bytes(tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw), is_prefetch=True)
        return (gw, uw, dw)

    def _estimate_local_expert_bytes(self, eid: int) -> int:
        eid = int(eid)
        try:
            if self.experts_kind == "fused":
                return int(tensor_nbytes(self.local_gate_up_cpu[eid]) + tensor_nbytes(self.local_down_cpu[eid]))
            return int(
                tensor_nbytes(self.local_gate_cpu[eid])
                + tensor_nbytes(self.local_up_cpu[eid])
                + tensor_nbytes(self.local_down_cpu2[eid])
            )
        except Exception:
            return 0

    def _prefetch_has_memory_headroom(self, required_bytes: int) -> bool:
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return True
        min_free_mb = float(os.environ.get("SPARSE_PREFETCH_MIN_FREE_MB", "0") or 0.0)
        if min_free_mb <= 0:
            return True
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            return True
        return int(free_bytes) >= int(required_bytes) + int(min_free_mb * 1024.0 * 1024.0)

    def _discard_cache_entry_before_prefetch(self, eid: int) -> None:
        eid = int(eid)
        ev = self._prefetch_events.pop(eid, None)
        if ev is not None and _env_flag("SPARSE_PREFETCH_WAIT_BEFORE_EVICT", "1"):
            try:
                torch.cuda.current_stream(self.device).wait_event(ev)
            except Exception:
                pass
        discard = getattr(self._cache, "discard", None)
        if callable(discard):
            discard(eid)
        self._prefetched_ids_this_group.discard(eid)
        self._prefetch_used_ids_this_group.discard(eid)
        self._prefetch_expire_layer.pop(eid, None)
        self._transition_prefetched_ids.discard(eid)
        self._transition_used_ids.discard(eid)

    def _postprocess_cache_value_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return F.softplus(scores.float()) * float(self._cache_value_score_scale)

    def _resident_keep_bias(self, eid: int) -> float:
        freq = float(min(self._expert_access_freq.get(int(eid), 0), 16))
        bias = 0.02 * freq
        if hasattr(self._cache, '_hot') and int(eid) in getattr(self._cache, '_hot', {}):
            bias += 0.25
        if int(eid) in self._prefetched_ids_this_group:
            bias += 0.10
        return float(bias)

    def _build_target_layer_prob_score_map(
        self,
        topi: Optional[torch.Tensor],
        topv: Optional[torch.Tensor],
        *,
        pos_start: int = 0,
        pos_end: Optional[int] = None,
        count_metrics: bool = True,
    ) -> Dict[int, float]:
        if topi is None or topv is None:
            return {}
        cols = int(topi.shape[-1])
        lo = max(0, min(int(pos_start), cols))
        hi = cols if pos_end is None else max(lo, min(int(pos_end), cols))
        if hi <= lo:
            return {}
        topi_win = topi[..., lo:hi]
        topv_win = topv[..., lo:hi]
        flat_i = topi_win.reshape(-1)
        flat_w = topv_win.float().reshape(-1)
        scores: Dict[int, float] = {}
        occur_cnt: Dict[int, int] = {}
        max_prob: Dict[int, float] = {}
        unique_pred: set[int] = set()
        unique_local: set[int] = set()
        unique_nonlocal: set[int] = set()
        raw_scores: Dict[int, float] = {}
        rs_candidate_eids = self._stage_prefetch_eid_set()
        future_stage = "decode" if str(SparseContext.current_phase) == "decode" else self._current_stage()
        for eid, wv in zip(flat_i.tolist(), flat_w.tolist()):
            eid = int(eid)
            unique_pred.add(eid)
            wv = float(wv)
            raw_scores[eid] = raw_scores.get(eid, 0.0) + wv
            occur_cnt[eid] = int(occur_cnt.get(eid, 0)) + 1
            max_prob[eid] = max(float(max_prob.get(eid, 0.0)), wv)

        for eid, score in raw_scores.items():
            legacy_local = eid in rs_candidate_eids
            resolved_rank = int(
                self.resolve_prefetch_target_rank(
                    layer_idx=int(self.layer_idx),
                    source_rank=int(self.comm_rank),
                    expert_id=int(eid),
                    stage=future_stage,
                    count_metrics=False,
                    reason="rs_prefetch_score",
                ).target_rank
            )
            self._record_rs_resolver_match(
                eid=eid,
                target_rank=resolved_rank,
                legacy_local=legacy_local,
                value=int(occur_cnt.get(eid, 1)),
            )
            if resolved_rank != int(self.comm_rank):
                unique_nonlocal.add(eid)
                continue
            unique_local.add(eid)
            scores[eid] = float(score)
        if count_metrics and self.planner_mode == "rs_prefetch":
            self._record_rs_counter("rs_pred_candidate_count", len(unique_pred))
            self._record_rs_counter("rs_pred_local_candidate_count", len(unique_local))
            self._record_rs_counter("rs_pred_nonlocal_candidate_count", len(unique_nonlocal))
        for eid in list(scores.keys()):
            reuse_boost = 1.0 + 0.15 * float(max(occur_cnt.get(eid, 1) - 1, 0))
            scores[eid] = scores[eid] * reuse_boost + 0.05 * float(max_prob.get(eid, 0.0))
        return scores

    def _build_target_layer_value_score_map(self, scores: Optional[torch.Tensor]) -> Dict[int, float]:
        if scores is None or scores.dim() != 2:
            return {}
        agg = scores.float().sum(dim=0)
        out: Dict[int, float] = {}
        rs_candidate_eids = self._stage_prefetch_eid_set()
        for eid in rs_candidate_eids:
            resolved_rank = int(
                self.resolve_prefetch_target_rank(
                    layer_idx=int(self.layer_idx),
                    source_rank=int(self.comm_rank),
                    expert_id=int(eid),
                    stage="decode" if str(SparseContext.current_phase) == "decode" else self._current_stage(),
                    count_metrics=False,
                    reason="rs_prefetch_value",
                ).target_rank
            )
            if resolved_rank != int(self.comm_rank):
                self._record_rs_resolver_match(eid=eid, target_rank=resolved_rank, legacy_local=True)
                continue
            self._record_rs_resolver_match(eid=eid, target_rank=resolved_rank, legacy_local=True)
            val = float(agg[int(eid)].item())
            if val > 0.0:
                out[int(eid)] = val
        return out

    def _pick_victims_by_low_value(self, score_map: Dict[int, float], protect_ids: set[int]) -> List[int]:
        evictable = self._prefetch_evictable_ids(protect_ids)
        cold_ranked: List[Tuple[float, int, int]] = []
        for order_idx, eid in enumerate(evictable):
            keep_value = float(score_map.get(int(eid), 0.0)) + self._resident_keep_bias(int(eid))
            cold_ranked.append((keep_value, order_idx, int(eid)))
        cold_ranked.sort(key=lambda x: (x[0], x[1]))
        return [eid for _, _, eid in cold_ranked]

    def _build_prefetch_plan_from_score_map(
        self,
        score_map: Dict[int, float],
        *,
        protect_ids: set[int],
        budget: int,
        min_candidate_score: Optional[float] = None,
        replace_advantage: Optional[float] = None,
        replace_margin: Optional[float] = None,
        allow_active_replace: Optional[bool] = None,
    ) -> List[Tuple[int, Optional[int]]]:
        if budget <= 0 or len(score_map) == 0:
            return []
        min_score = self._prefetch_min_candidate_score if min_candidate_score is None else float(min_candidate_score)
        adv = self._prefetch_replace_advantage if replace_advantage is None else float(replace_advantage)
        margin = self._prefetch_replace_margin if replace_margin is None else float(replace_margin)
        active_replace = self._prefetch_allow_active_replace if allow_active_replace is None else bool(allow_active_replace)
        policy = self._rs_prefetch_policy()
        if policy in ("cache_safe", "cache-safe", "benefit_gate", "benefit-gate"):
            min_score = max(
                min_score,
                self._rank_env_float(
                    "SPARSE_RS_CACHE_SAFE_MIN_SCORE_BY_RANK",
                    self._rs_env_float("SPARSE_RS_CACHE_SAFE_MIN_SCORE", 0.03),
                ),
            )
            adv = max(
                adv,
                self._rank_env_float(
                    "SPARSE_RS_CACHE_SAFE_REPLACE_ADVANTAGE_BY_RANK",
                    self._rs_env_float("SPARSE_RS_CACHE_SAFE_REPLACE_ADVANTAGE", 0.05),
                ),
            )
            margin = max(
                margin,
                self._rank_env_float(
                    "SPARSE_RS_CACHE_SAFE_REPLACE_MARGIN_BY_RANK",
                    self._rs_env_float("SPARSE_RS_CACHE_SAFE_REPLACE_MARGIN", 1.10),
                ),
            )

        filter_in = len(score_map)
        out_cached = 0
        out_low_score = 0
        candidates: List[Tuple[float, int]] = []
        rs_candidate_eids = self._stage_prefetch_eid_set()
        for eid, score in score_map.items():
            eid_i = int(eid)
            score_f = float(score)
            legacy_local = eid_i in rs_candidate_eids
            resolved_rank = int(
                self.resolve_prefetch_target_rank(
                    layer_idx=int(self.layer_idx),
                    source_rank=int(self.comm_rank),
                    expert_id=eid_i,
                    stage="decode" if str(SparseContext.current_phase) == "decode" else self._current_stage(),
                    count_metrics=False,
                    reason="rs_prefetch_plan",
                ).target_rank
            )
            self._record_rs_resolver_match(eid=eid_i, target_rank=resolved_rank, legacy_local=legacy_local)
            if resolved_rank != int(self.comm_rank):
                continue
            if self._cache.has(eid_i):
                out_cached += 1
                continue
            if score_f < min_score:
                out_low_score += 1
                continue
            candidates.append((score_f, eid_i))
        self._record_rs_counter("rs_candidate_filter_in_count", filter_in)
        self._record_rs_counter("rs_candidate_filter_out_cached_count", out_cached)
        self._record_rs_counter("rs_candidate_filter_out_low_score_count", out_low_score)
        self._record_rs_counter("prefetch_rejected_low_score", out_low_score)
        self._record_rs_counter("rs_candidate_survived_count", len(candidates))
        self._record_rs_counter("rs_candidate_filter_out_count", max(0, filter_in - len(candidates)))
        if len(candidates) == 0:
            return []
        candidates.sort(key=lambda x: x[0], reverse=True)

        free_slots = self._prefetch_free_slots()
        victim_ids = self._pick_victims_by_low_value(score_map, protect_ids)

        plan: List[Tuple[int, Optional[int]]] = []
        cand_pos = 0

        direct_take = min(int(free_slots), int(budget), len(candidates))
        for _ in range(direct_take):
            score, eid = candidates[cand_pos]
            plan.append((int(eid), None))
            cand_pos += 1

        remaining = int(budget) - len(plan)
        if remaining <= 0:
            return plan
        if not active_replace:
            self._record_rs_counter("prefetch_rejected_active_replace", min(remaining, max(0, len(candidates) - cand_pos)))
            return plan

        victim_pos = 0
        while cand_pos < len(candidates) and victim_pos < len(victim_ids) and remaining > 0:
            cand_score, cand_eid = candidates[cand_pos]
            victim_eid = int(victim_ids[victim_pos])
            victim_score = float(score_map.get(victim_eid, 0.0)) + self._resident_keep_bias(victim_eid)

            better_by_margin = cand_score >= (victim_score * margin)
            better_by_adv = (cand_score - victim_score) >= adv
            if better_by_margin or better_by_adv:
                plan.append((int(cand_eid), victim_eid))
                remaining -= 1
                cand_pos += 1
                victim_pos += 1
            else:
                self._record_rs_counter("prefetch_rejected_margin", max(1, len(candidates) - cand_pos))
                break

        if remaining > 0 and cand_pos < len(candidates) and victim_pos >= len(victim_ids):
            self._record_rs_counter("prefetch_rejected_cache_full", min(remaining, len(candidates) - cand_pos))
        return plan

    def _rs_prefetch_policy(self) -> str:
        return str(os.environ.get("SPARSE_RS_PREFETCH_POLICY", "default") or "default").strip().lower()

    def _rank_env_value(self, name: str) -> Optional[str]:
        raw = str(os.environ.get(name, "") or "").strip()
        if raw == "":
            return None
        vals = [x.strip() for x in raw.split(",")]
        rank = int(getattr(self, "comm_rank", getattr(self, "rank", 0)))
        if rank < 0 or rank >= len(vals) or vals[rank] == "":
            return None
        return vals[rank]

    def _rank_env_int(self, name: str, default: int) -> int:
        raw = self._rank_env_value(name)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    def _rank_env_float(self, name: str, default: float) -> float:
        raw = self._rank_env_value(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _effective_rs_prefetch_top_m(self) -> int:
        return max(0, self._rank_env_int("SPARSE_RS_PREFETCH_TOP_M_BY_RANK", int(self.prefetch_top_m)))

    def _rs_env_int(self, name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)) or default)
        except Exception:
            return int(default)

    def _rs_env_float(self, name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)) or default)
        except Exception:
            return float(default)

    def _rs_env_flag(self, name: str, default: str = "0") -> bool:
        return str(os.environ.get(name, default) or default).strip().lower() not in ("", "0", "false", "no", "off")

    def _build_two_tier_prefetch_plan_from_prob_scores(
        self,
        topi: Optional[torch.Tensor],
        topv: Optional[torch.Tensor],
        *,
        protect_ids: set[int],
        budget: int,
    ) -> List[Tuple[int, Optional[int]]]:
        """Build RS demand + ERS-candidate prefetch plan.

        Tier A admits high-confidence predicted owner-local experts. Tier B
        admits lower-rank owner-local experts that are likely candidates for
        ERS replacement. Tier B is conservative by default: it uses free or
        reserved speculative space unless explicitly allowed to evict residents.
        """
        if topi is None or topv is None or int(budget) <= 0:
            return []
        cols = int(topi.shape[-1])
        if cols <= 0:
            return []

        demand_topk = self._rs_env_int("SPARSE_RS_PREFETCH_DEMAND_TOPK", min(6, cols))
        demand_budget_default = min(2, int(budget))
        demand_budget = self._rs_env_int("SPARSE_RS_PREFETCH_DEMAND_BUDGET", demand_budget_default)
        demand_budget = max(0, min(int(demand_budget), int(budget)))

        ers_start = self._rs_env_int("SPARSE_RS_PREFETCH_ERS_POS_START", int(self.top_k))
        ers_topm = self._rs_env_int("SPARSE_RS_PREFETCH_ERS_TOPM", cols)
        ers_budget_default = max(0, int(budget) - int(demand_budget))
        ers_budget = self._rs_env_int("SPARSE_RS_PREFETCH_ERS_BUDGET", ers_budget_default)
        ers_budget = max(0, min(int(ers_budget), max(0, int(budget) - int(demand_budget))))

        demand_score_map = self._build_target_layer_prob_score_map(
            topi,
            topv,
            pos_start=0,
            pos_end=max(0, min(int(demand_topk), cols)),
            count_metrics=True,
        )
        plan: List[Tuple[int, Optional[int]]] = []
        planned_ids: set[int] = set()
        if demand_budget > 0 and demand_score_map:
            plan_a = self._build_prefetch_plan_from_score_map(
                demand_score_map,
                protect_ids=protect_ids,
                budget=demand_budget,
                min_candidate_score=self._rs_env_float(
                    "SPARSE_RS_PREFETCH_DEMAND_MIN_CAND_SCORE",
                    self._prefetch_min_candidate_score,
                ),
                replace_advantage=self._rs_env_float(
                    "SPARSE_RS_PREFETCH_DEMAND_REPLACE_ADVANTAGE",
                    self._prefetch_replace_advantage,
                ),
                replace_margin=self._rs_env_float(
                    "SPARSE_RS_PREFETCH_DEMAND_REPLACE_MARGIN",
                    self._prefetch_replace_margin,
                ),
                allow_active_replace=self._rs_env_flag(
                    "SPARSE_RS_PREFETCH_DEMAND_ALLOW_ACTIVE_REPLACE",
                    "1" if self._prefetch_allow_active_replace else "0",
                ),
            )
            plan.extend(plan_a)
            planned_ids.update(int(eid) for eid, _ in plan_a)

        if ers_budget <= 0:
            return plan
        ers_lo = max(0, min(int(ers_start), cols))
        ers_hi = max(ers_lo, min(int(ers_topm), cols))
        if ers_hi <= ers_lo:
            return plan
        ers_score_map = self._build_target_layer_prob_score_map(
            topi,
            topv,
            pos_start=ers_lo,
            pos_end=ers_hi,
            count_metrics=False,
        )
        for eid in list(ers_score_map.keys()):
            if int(eid) in planned_ids:
                del ers_score_map[eid]
        if not ers_score_map:
            return plan

        # Protect Tier-A planned ids from being selected as Tier-B victims.
        tier_b_protect = set(protect_ids)
        tier_b_protect.update(planned_ids)
        plan_b = self._build_prefetch_plan_from_score_map(
            ers_score_map,
            protect_ids=tier_b_protect,
            budget=ers_budget,
            min_candidate_score=self._rs_env_float("SPARSE_RS_PREFETCH_ERS_MIN_CAND_SCORE", 0.0),
            replace_advantage=self._rs_env_float("SPARSE_RS_PREFETCH_ERS_REPLACE_ADVANTAGE", 0.0),
            replace_margin=self._rs_env_float("SPARSE_RS_PREFETCH_ERS_REPLACE_MARGIN", 1.0),
            allow_active_replace=self._rs_env_flag("SPARSE_RS_PREFETCH_ERS_ALLOW_ACTIVE_REPLACE", "0"),
        )
        plan.extend(plan_b)
        return plan

    def _build_prefetch_plan_from_prob_predictions(
        self,
        topi: Optional[torch.Tensor],
        topv: Optional[torch.Tensor],
        *,
        protect_ids: set[int],
        budget: int,
    ) -> List[Tuple[int, Optional[int]]]:
        if self._rs_prefetch_policy() in ("two_tier", "two-tier", "rs_ers_aligned", "ers_aligned"):
            return self._build_two_tier_prefetch_plan_from_prob_scores(
                topi,
                topv,
                protect_ids=protect_ids,
                budget=budget,
            )
        score_map = self._build_target_layer_prob_score_map(topi, topv)
        if self._light_policy is not None and str(SparseContext.current_phase) == "decode":
            score_map = self._light_policy.scores(score_map, set(self._cache.resident_ids()))
        return self._build_prefetch_plan_from_score_map(score_map, protect_ids=protect_ids, budget=budget)

    def _admit_prefetch_plan(self, plan: List[Tuple[int, Optional[int]]], *, group_end: int) -> None:
        if len(plan) == 0:
            return
        transition_mode = bool(getattr(self, "_transition_admit_active", False))
        self._record_rs_counter("prefetch_issued_count", len(plan))
        time_prefetch_h2d = _env_flag("SPARSE_PREFETCH_H2D_TIMING", "0")
        ctx = torch.cuda.stream(self.prefetch_stream) if self.prefetch_stream is not None else nullcontext()
        with ctx:
            protect = self._prefetch_protect_ids()
            for eid, victim in plan:
                eid = int(eid)
                if self._cache.has(eid):
                    payload = self._cache.get(eid)
                    ok = self._cache.admit_prefetch(eid, payload, protect_ids=protect, preferred_victim=victim)
                    if not ok:
                        continue
                    self._prefetched_ids_this_group.add(eid)
                    self._prefetch_expire_layer[eid] = int(group_end - 1)
                    protect.add(eid)
                    if str(os.environ.get("SPARSE_PREFETCH_PROTECT_MODE", "default") or "default").strip().lower() in ("minimal", "current_next", "current-next"):
                        try:
                            self._cache.unpin(eid)
                        except Exception:
                            pass
                    self._record_prefetch_admitted()
                    if transition_mode:
                        self._transition_prefetched_ids.add(int(eid))
                        self._record_metric_counter("transition_prefetch_admitted", 1)
                    continue

                start_ev = None
                if (
                    time_prefetch_h2d
                    and self.metrics is not None
                    and self.prefetch_stream is not None
                    and self._timing_sampled("SPARSE_PREFETCH_H2D_TIMING_SAMPLE_RATE", "prefetch_h2d", eid)
                ):
                    try:
                        start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                        start_ev.record(self.prefetch_stream)
                    except Exception:
                        start_ev = None
                # Important on 11GB GPUs: if active replacement selected a
                # victim, release that resident payload before materializing the
                # new expert. Otherwise prefetch briefly needs old+new weights
                # and can OOM even though the cache capacity would be unchanged.
                if victim is not None:
                    self._discard_cache_entry_before_prefetch(int(victim))
                required_bytes = self._estimate_local_expert_bytes(eid)
                if not self._global_cache_make_room(required_bytes):
                    continue
                if victim is None and not self._prefetch_has_memory_headroom(required_bytes):
                    continue
                try:
                    payload = self._materialize_local_expert_for_prefetch(eid)
                except torch.cuda.OutOfMemoryError:
                    # Prefetch is speculative. On 11GB cards, active replacement
                    # can still fail because the CUDA allocator cannot satisfy a
                    # transient copy even after a victim was discarded. Skip this
                    # candidate and let the later demand path load it normally.
                    if self.metrics is not None:
                        self.metrics.prefetch_oom_skip_count += 1
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                if start_ev is not None:
                    try:
                        end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                        end_ev.record(self.prefetch_stream)
                        _append_prefetch_h2d_event(self.metrics, start_ev, end_ev, layer_idx=self.layer_idx)
                    except Exception:
                        pass
                ok = self._cache.admit_prefetch(
                    eid,
                    payload,
                    protect_ids=protect,
                    preferred_victim=victim,
                )
                if not ok:
                    continue
                self._record_prefetch_event(eid)
                self._prefetched_ids_this_group.add(eid)
                self._prefetch_expire_layer[eid] = int(group_end - 1)
                protect.add(eid)
                if str(os.environ.get("SPARSE_PREFETCH_PROTECT_MODE", "default") or "default").strip().lower() in ("minimal", "current_next", "current-next"):
                    try:
                        self._cache.unpin(eid)
                    except Exception:
                        pass
                self._record_prefetch_admitted()
                if transition_mode:
                    self._transition_prefetched_ids.add(int(eid))
                    self._record_metric_counter("transition_prefetch_admitted", 1)

    def _prefetch_target_layer_from_value_scores(self, scores: Optional[torch.Tensor], group_start: int, group_end: int) -> None:
        if self.planner_mode != 'rs_prefetch' or self.expert_offload != 'cpu' or self._effective_rs_prefetch_top_m() <= 0:
            return
        self._begin_group_prefetch_cycle(group_start, group_end)
        protect = self._prefetch_protect_ids()
        budget = self._prefetch_target_budget(group_start, group_end, count_metrics=True)
        if budget <= 0:
            return

        score_map = self._build_target_layer_value_score_map(scores)
        plan = self._build_prefetch_plan_from_score_map(
            score_map,
            protect_ids=protect,
            budget=budget,
            min_candidate_score=self._cache_value_min_candidate_score,
            replace_advantage=self._cache_value_replace_advantage,
        )
        if len(plan) == 0:
            return

        self._admit_prefetch_plan(plan, group_end=group_end)

    def _prefetch_target_layer_from_prob_scores(
        self,
        topi: Optional[torch.Tensor],
        topv: Optional[torch.Tensor],
        group_start: int,
        group_end: int,
    ) -> None:
        if self.planner_mode != 'rs_prefetch' or self.expert_offload != 'cpu' or self._effective_rs_prefetch_top_m() <= 0:
            return
        self._begin_group_prefetch_cycle(group_start, group_end)
        protect = self._prefetch_protect_ids()
        budget = self._prefetch_target_budget(group_start, group_end, count_metrics=True)
        if budget <= 0:
            return

        plan = self._build_prefetch_plan_from_prob_predictions(
            topi,
            topv,
            protect_ids=protect,
            budget=budget,
        )
        if len(plan) == 0:
            return

        self._admit_prefetch_plan(plan, group_end=group_end)

    def _schedule_transition_prefetch_from_sources(self, source_experts: set[int]) -> None:
        if not bool(getattr(self, "_transition_prefetch_enabled", False)):
            return
        if self.expert_offload != "cpu" or self._effective_rs_prefetch_top_m() <= 0:
            return
        if not source_experts or not getattr(self, "_transition_table", None):
            return
        t0 = time.perf_counter()
        self._record_metric_counter("transition_lookup_calls", 1)
        score_map: Dict[int, float] = {}
        for src in source_experts:
            cand = self._transition_table.get(int(src), [])
            for pos, eid in enumerate(cand[: int(getattr(self, "_transition_topk", 1))]):
                eid_i = int(eid)
                if eid_i not in self._local_eid_set:
                    continue
                score_map[eid_i] = max(float(score_map.get(eid_i, 0.0)), 1.0 / float(pos + 1))
        if self.metrics is not None:
            self.metrics.transition_lookup_overhead_ms = float(
                getattr(self.metrics, "transition_lookup_overhead_ms", 0.0)
            ) + (time.perf_counter() - t0) * 1000.0
        if not score_map:
            return
        self._begin_group_prefetch_cycle(
            self.layer_idx,
            self.layer_idx + 1,
            cycle_key=("transition", int(SparseContext.runtime_call_id), int(self.layer_idx)),
        )
        protect = self._prefetch_protect_ids()
        free_budget = self._prefetch_free_slots()
        if self._transition_no_replace:
            budget = min(int(self._transition_topk), int(free_budget))
            if budget <= 0:
                self._record_metric_counter("transition_no_replace_denied", len(score_map))
                return
        else:
            budget = min(int(self._transition_topk), int(self._prefetch_total_insert_capacity(protect)))
            if budget <= 0:
                self._record_metric_counter("transition_safe_evict_denied", len(score_map))
                return
        if self._transition_rate_limit:
            budget = min(int(budget), max(1, int(os.environ.get("SPARSE_TRANSITION_RATE_LIMIT_PER_LAYER", "1") or 1)))
        plan = self._build_prefetch_plan_from_score_map(
            score_map,
            protect_ids=protect,
            budget=budget,
            min_candidate_score=0.0,
            replace_advantage=0.0,
            replace_margin=1.0,
            allow_active_replace=(not bool(getattr(self, "_transition_no_replace", True))) and bool(getattr(self, "_transition_safe_evict", True)),
        )
        if not plan:
            if self._transition_no_replace:
                self._record_metric_counter("transition_no_replace_denied", len(score_map))
            else:
                self._record_metric_counter("transition_safe_evict_denied", len(score_map))
            return
        old = bool(getattr(self, "_transition_admit_active", False))
        self._transition_admit_active = True
        try:
            self._admit_prefetch_plan(plan, group_end=self.layer_idx + 1)
        finally:
            self._transition_admit_active = old

    def _schedule_transition_prefetch_for_next_layer(self, idx: torch.Tensor) -> None:
        if not bool(getattr(self, "_transition_prefetch_enabled", False)):
            return
        target_layer = int(self.layer_idx) + 1
        if target_layer >= int(self.num_layers):
            return
        try:
            flat = sorted(set(int(x) for x in idx.detach().reshape(-1).to("cpu", dtype=torch.long).tolist()))
        except Exception:
            return
        SparseContext.set_transition_source_experts(target_layer, flat)


    def warmup_cache_from_prefill_counts(self) -> None:
        if not _env_flag("SPARSE_PREFILL_DECODE_WARMUP", "0"):
            return
        if self.expert_offload != "cpu" or self._base_expert_cache_capacity <= 0:
            return
        layer_counts = SparseContext.prefill_expert_counts.get(int(self.layer_idx), {})
        if not isinstance(layer_counts, dict) or not layer_counts:
            return
        topk = max(1, _env_int("SPARSE_WARMUP_TOPK_PER_LAYER", 1))
        source = str(os.environ.get("SPARSE_WARMUP_SOURCE", "freq") or "freq").strip().lower()
        items = [(int(eid), int(cnt)) for eid, cnt in layer_counts.items() if int(eid) in self._local_eid_set]
        if not items:
            return
        if source in ("recent", "freq", "score_weighted"):
            items.sort(key=lambda x: (-x[1], x[0]))
        else:
            items.sort(key=lambda x: (-x[1], x[0]))
        loaded = 0
        skipped = 0
        max_new = min(int(topk), len(items))
        for eid, _ in items[:max_new]:
            if self._cache.has(eid):
                skipped += 1
                continue
            before_bytes = int(getattr(self.metrics, "expert_weight_h2d_prefetch_bytes", 0)) if self.metrics is not None else 0
            try:
                payload = self._materialize_local_expert_for_prefetch(eid)
            except torch.cuda.OutOfMemoryError:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            self._cache_put_guarded(eid, payload, pinned=False, prefer_hot=True)
            if self.metrics is not None:
                after_bytes = int(getattr(self.metrics, "expert_weight_h2d_prefetch_bytes", 0))
                self.metrics.warmup_h2d_bytes += max(0, after_bytes - before_bytes)
            loaded += 1
        if self.metrics is not None:
            self.metrics.warmup_loaded_experts += int(loaded)
            self.metrics.warmup_skipped_resident += int(skipped)

    @classmethod
    def warmup_all_caches_from_prefill(cls) -> None:
        for _, blk in sorted(SparseContext.layer_blocks.items(), key=lambda kv: int(kv[0])):
            try:
                blk.warmup_cache_from_prefill_counts()
            except Exception:
                continue

    def _schedule_future_layer_prefetch_from_group_cache(self, group_start: int, group_end: int) -> None:
        if self.planner_mode != 'rs_prefetch' or self.expert_offload != 'cpu' or self._effective_rs_prefetch_top_m() <= 0:
            return
        rs_task = getattr(SparseContext, 'rs_bank_task', 'router_prob')
        for l in range(int(group_start) + 1, int(group_end)):
            blk = SparseContext.layer_blocks.get(int(l), None)
            if blk is None:
                continue
            if rs_task == 'cache_value_rs':
                blk._prefetch_target_layer_from_value_scores(
                    SparseContext.group_pred_value.get(int(l), None),
                    group_start=int(group_start),
                    group_end=int(group_end),
                )
            else:
                blk._prefetch_target_layer_from_prob_scores(
                    SparseContext.group_pred_topi.get(int(l), None),
                    SparseContext.group_pred_topv.get(int(l), None),
                    group_start=int(group_start),
                    group_end=int(group_end),
                )

    def _record_prefetch_eval_actual(self, idx: torch.Tensor, seq_len: int) -> None:
        if self.metrics is None:
            return
        if self.planner_mode != "rs_prefetch" or int(seq_len) != 1:
            return
        if self._prefetch_current_group_start is None:
            return
        pred_ids = set(int(eid) for eid in self._prefetched_ids_this_group)
        if len(pred_ids) == 0:
            return

        flat_i = idx.detach().reshape(-1).to(torch.long)
        if flat_i.numel() == 0:
            actual_ids: set[int] = set()
        else:
            actual_ids = set()
            for eid in flat_i.tolist():
                ieid = int(eid)
                if 0 <= ieid < int(self.num_experts):
                    # For replicated SCP-EP, expert_to_rank[ieid] is the dispatch
                    # choice for this *source* rank, not the set of experts this rank
                    # may receive/execute.  Use the receive/executable set instead so
                    # RS prefetch hit accounting matches actual local cache responsibility.
                    executable = self._stage_prefetch_eid_set()
                    if ieid in executable:
                        actual_ids.add(ieid)

        self.metrics.rs_prefetch_eval_events += 1
        self.metrics.rs_prefetch_eval_pred_total += len(pred_ids)
        self.metrics.rs_prefetch_eval_actual_total += len(actual_ids)
        self.metrics.rs_prefetch_eval_overlap_total += len(pred_ids.intersection(actual_ids))

    def _compute_rs_cache_group(self, x: torch.Tensor, g: int, enable_prefetch: bool, batch_size: int, seq_len: int) -> None:
        assert SparseContext.rs_bank is not None
        bank = SparseContext.rs_bank
        end = min(g + self.rs_group, self.num_layers)

        group_topi: Dict[int, torch.Tensor] = {}
        group_topv: Dict[int, torch.Tensor] = {}
        group_value: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x_rs = x.float()
            # Planner-only RS never uses the source/group-start layer's own
            # prediction for prefetch; scheduling starts from g+1. Skip it to
            # avoid paying one extra RS MLP per group-start decode step.
            start_l = int(g)
            if self.expert_source != "rs" and bool(enable_prefetch):
                start_l = min(int(g) + 1, int(end))
            for l in range(start_l, end):
                pred = bank[l](x_rs).to(torch.float32)

                if getattr(SparseContext, 'rs_bank_task', 'router_prob') == 'cache_value_rs':
                    group_value[l] = self._postprocess_cache_value_scores(pred)
                    continue

                probs = pred
                pred_k = int(self.top_k)
                if bool(enable_prefetch):
                    pred_k = min(int(probs.shape[-1]), int(self._rs_prefetch_predict_topk()))
                topv, topi = torch.topk(probs, k=pred_k, dim=-1)
                if self.norm_topk_prob:
                    topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                w = topv.to(dtype=x.dtype)
                i = topi.to(torch.long)

                group_topi[l] = i
                group_topv[l] = w

                if self.exec_routing_mode == 'rs':
                    SparseContext.routing_cache[l] = (w, i)

        SparseContext.group_pred_topi = dict(group_topi)
        SparseContext.group_pred_topv = dict(group_topv)
        SparseContext.group_pred_value = dict(group_value)

    def _compute_and_schedule_rs_prefetch_group(self, x: torch.Tensor, g: int) -> None:
        assert SparseContext.rs_bank is not None
        bank = SparseContext.rs_bank
        end = min(g + self.rs_group, self.num_layers)
        rs_task = getattr(SparseContext, 'rs_bank_task', 'router_prob')

        group_topi: Dict[int, torch.Tensor] = {}
        group_topv: Dict[int, torch.Tensor] = {}
        group_value: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x_rs = x.float()
            for l in range(int(g) + 1, int(end)):
                blk = SparseContext.layer_blocks.get(int(l), None)
                if blk is None:
                    continue
                budget = blk._prefetch_target_budget(int(g), int(end), count_metrics=True)
                if budget <= 0:
                    continue

                pred = bank[l](x_rs).to(torch.float32)
                protect = blk._prefetch_protect_ids()
                if rs_task == 'cache_value_rs':
                    scores = blk._postprocess_cache_value_scores(pred)
                    group_value[int(l)] = scores
                    score_map = blk._build_target_layer_value_score_map(scores)
                    plan = blk._build_prefetch_plan_from_score_map(
                        score_map,
                        protect_ids=protect,
                        budget=budget,
                        min_candidate_score=blk._cache_value_min_candidate_score,
                        replace_advantage=blk._cache_value_replace_advantage,
                    )
                else:
                    probs = pred
                    pred_k = min(int(probs.shape[-1]), int(blk._rs_prefetch_predict_topk()))
                    topv, topi = torch.topk(probs, k=pred_k, dim=-1)
                    if self.norm_topk_prob:
                        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    w = topv.to(dtype=x.dtype)
                    i = topi.to(torch.long)
                    group_topi[int(l)] = i
                    group_topv[int(l)] = w
                    plan = blk._build_prefetch_plan_from_prob_predictions(
                        i,
                        w,
                        protect_ids=protect,
                        budget=budget,
                    )

                if len(plan) == 0:
                    continue
                blk._admit_prefetch_plan(plan, group_end=int(end))

        SparseContext.group_pred_topi = dict(group_topi)
        SparseContext.group_pred_topv = dict(group_topv)
        SparseContext.group_pred_value = dict(group_value)

    def _run_rs_mlp_timed(self, module: nn.Module, x_rs: torch.Tensor) -> torch.Tensor:
        if self.metrics is None:
            return module(x_rs)
        if not torch.cuda.is_available() or self.device.type != "cuda":
            t0 = time.perf_counter()
            out = module(x_rs)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.rs_mlp_ms += dt_ms
        else:
            if not _env_flag("SPARSE_RS_MLP_SYNC_TIMING", "0"):
                out = module(x_rs)
                self.metrics.rs_mlp_calls += 1
                return out
            start = torch.cuda.Event(enable_timing=True, blocking=False)
            end = torch.cuda.Event(enable_timing=True, blocking=False)
            start.record(torch.cuda.current_stream(self.device))
            out = module(x_rs)
            end.record(torch.cuda.current_stream(self.device))
            end.synchronize()
            self.metrics.rs_mlp_ms += float(start.elapsed_time(end))
            self._record_rs_mlp_layer_ms(float(start.elapsed_time(end)))
            self.metrics.rs_mlp_calls += 1
            return out
        self._record_rs_mlp_layer_ms(float(dt_ms))
        self.metrics.rs_mlp_calls += 1
        return out

    def _rs_prefetch_predict_topk(self) -> int:
        by_rank = self._rank_env_value("SPARSE_RS_PREFETCH_PRED_TOPK_BY_RANK")
        if by_rank is not None:
            try:
                return max(1, min(int(by_rank), int(self.num_experts)))
            except Exception:
                pass
        raw = os.environ.get("SPARSE_RS_PREFETCH_PRED_TOPK", "").strip()
        if raw:
            try:
                return max(1, min(int(raw), int(self.num_experts)))
            except Exception:
                pass
        return max(int(self.top_k), min(int(self.num_experts), int(self._effective_rs_prefetch_top_m())))

    def _build_router_topk_feature(self, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        feat = torch.zeros((idx.shape[0], self.num_experts), device=idx.device, dtype=torch.float32)
        ww = w.float()
        ww = ww / ww.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        feat.scatter_add_(1, idx.to(torch.long), ww)
        return feat

    def _next_layer_rs_feature_mode(self) -> str:
        target_layer = int(self.layer_idx) + 1
        if SparseContext.rs_bank is None or target_layer >= len(SparseContext.rs_bank):
            return "hidden"
        return str(getattr(SparseContext.rs_bank[target_layer], "rs_feature_mode", "hidden"))

    def _compute_and_schedule_next_layer_rs(
        self,
        x: torch.Tensor,
        seq_len: int,
        source_idx: Optional[torch.Tensor] = None,
        source_w: Optional[torch.Tensor] = None,
    ) -> None:
        if int(seq_len) != 1 and not _env_flag("SPARSE_RS_PREFETCH_PREFILL", "0"):
            return
        if SparseContext.rs_bank is None:
            return
        if getattr(SparseContext, "rs_bank_layout", "group") != "next_layer":
            return
        if self.planner_mode != "rs_prefetch" and self.expert_source != "rs" and self.exec_routing_mode != "rs":
            return

        source_layer = int(self.layer_idx)
        target_layer = source_layer + 1
        if target_layer >= int(self.num_layers):
            return
        source_layers = getattr(SparseContext, "rs_bank_source_layers", [])
        if target_layer >= len(source_layers) or int(source_layers[target_layer]) != source_layer:
            return

        target_blk = SparseContext.layer_blocks.get(target_layer, None)
        if target_blk is None:
            return

        should_predict_for_routing = (self.expert_source == "rs" or self.exec_routing_mode == "rs")
        cycle_key = (
            "next_layer_rs",
            str(SparseContext.current_phase),
            int(getattr(SparseContext, "runtime_call_id", -1)),
            int(source_layer),
            int(target_layer),
        )
        budget = 0
        if self.planner_mode == "rs_prefetch":
            budget = target_blk._prefetch_target_budget(
                target_layer,
                target_layer + 1,
                count_metrics=True,
                cycle_key=cycle_key,
            )
        if budget <= 0 and not should_predict_for_routing:
            return

        rs_input = x.float()
        feature_mode = str(getattr(SparseContext.rs_bank[target_layer], "rs_feature_mode", "hidden"))
        if feature_mode == "hidden_router_topk":
            if source_idx is None or source_w is None:
                return
            rs_input = torch.cat([rs_input, self._build_router_topk_feature(source_idx, source_w)], dim=-1)
        elif feature_mode != "hidden":
            raise RuntimeError(f"Unsupported next-layer RS feature_mode={feature_mode}")

        pred = self._run_rs_mlp_timed(SparseContext.rs_bank[target_layer], rs_input).to(torch.float32)
        if self.metrics is not None:
            self.metrics.rs_next_layer_pred_calls += 1

        # If requested, the next layer can execute using this cached prediction.
        # This is valid only for next-layer checkpoints: L(source) predicts L+1(target).
        if should_predict_for_routing:
            exec_topk_raw = os.environ.get("SPARSE_RS_EXEC_TOPK", "").strip()
            try:
                exec_topk = int(exec_topk_raw) if exec_topk_raw else int(self.top_k)
            except Exception:
                exec_topk = int(self.top_k)
            exec_topk = max(1, min(int(exec_topk), int(pred.shape[-1])))
            route_topv, route_topi = torch.topk(pred, k=exec_topk, dim=-1)
            if self.norm_topk_prob:
                route_topv = route_topv / route_topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            SparseContext.routing_cache[target_layer] = (route_topv.to(dtype=x.dtype), route_topi.to(torch.long))

        if self.planner_mode != "rs_prefetch" or self.expert_offload != "cpu" or budget <= 0:
            return

        pred_k = min(int(pred.shape[-1]), int(target_blk._rs_prefetch_predict_topk()))
        if target_blk._light_policy is not None:
            policy = target_blk._light_policy
            source_cost = policy.profile.cost(self.comm_rank, source_layer)
            policy.window_ms = source_cost.compute(self.top_k, int(x.shape[0]) * self.top_k) + policy.cost.attention
        topv, topi = torch.topk(pred, k=pred_k, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        protect = target_blk._prefetch_protect_ids()
        plan = target_blk._build_prefetch_plan_from_prob_predictions(
            topi.to(torch.long),
            topv.to(dtype=x.dtype),
            protect_ids=protect,
            budget=int(budget),
        )
        if len(plan) == 0:
            return
        target_blk._admit_prefetch_plan(plan, group_end=target_layer + 1)

    def _late_rs_prefetch_topk(self) -> int:
        raw = os.environ.get("SPARSE_RS_LATE_TOPK", "").strip()
        if raw:
            try:
                return max(1, min(int(raw), int(self.num_experts)))
            except Exception:
                pass
        return max(int(self.top_k), min(int(self.num_experts), 6))

    def _compute_and_schedule_late_next_layer_rs(self, block_output: torch.Tensor, seq_len: int) -> None:
        """Late-refiner admission using layer-i decoder block output for layer i+1.

        This is intentionally planner-only. It does not replace the router and
        does not change the baseline execution path. The late refiner gets less
        overlap time than the early predictor, but it can correct admission:
        early Top4 warms obvious candidates; late Top6 can add/replace target
        layer residents right before the next layer consumes them.
        """
        if int(seq_len) != 1:
            return
        if not _env_flag("SPARSE_RS_TWO_STAGE", "0"):
            return
        if SparseContext.late_rs_bank is None:
            return
        if getattr(SparseContext, "late_rs_bank_layout", "group") != "next_layer":
            return
        if self.planner_mode != "rs_prefetch" or self.expert_offload != "cpu" or self._effective_rs_prefetch_top_m() <= 0:
            return

        source_layer = int(self.layer_idx)
        target_layer = source_layer + 1
        if target_layer >= int(self.num_layers):
            return
        source_layers = getattr(SparseContext, "late_rs_bank_source_layers", [])
        if target_layer >= len(source_layers) or int(source_layers[target_layer]) != source_layer:
            return
        target_blk = SparseContext.layer_blocks.get(target_layer, None)
        if target_blk is None:
            return

        cycle_key = (
            "next_layer_rs",
            str(SparseContext.current_phase),
            int(getattr(SparseContext, "runtime_call_id", -1)),
            int(source_layer),
            int(target_layer),
        )
        # Keep the early cycle alive so runtime eval/use accounting sees the
        # union of early and late admitted ids. Unlike the early budget, late
        # admission deliberately does not subtract already-prefetched ids; it is
        # an admission correction pass under the same physical cache capacity.
        target_blk._begin_group_prefetch_cycle(target_layer, target_layer + 1, cycle_key=cycle_key)
        protect = target_blk._prefetch_protect_ids()
        budget = min(
            int(target_blk._effective_rs_prefetch_top_m()),
            int(target_blk._prefetch_total_insert_capacity(protect)),
        )
        budget = max(0, int(budget))
        if target_blk.metrics is not None:
            if budget <= 0:
                target_blk.metrics.prefetch_budget_zero_count += 1
            else:
                target_blk.metrics.prefetch_budget_positive_count += 1
        if budget <= 0:
            return

        rs_input = block_output.float()
        if rs_input.dim() == 3:
            rs_input = rs_input.reshape(-1, rs_input.shape[-1])
        feature_mode = str(getattr(SparseContext.late_rs_bank[target_layer], "rs_feature_mode", "hidden"))
        if feature_mode != "hidden":
            raise RuntimeError(f"Unsupported late next-layer RS feature_mode={feature_mode}")

        pred = self._run_rs_mlp_timed(SparseContext.late_rs_bank[target_layer], rs_input).to(torch.float32)
        if self.metrics is not None:
            self.metrics.rs_next_layer_pred_calls += 1

        pred_k = min(int(pred.shape[-1]), int(self._late_rs_prefetch_topk()))
        topv, topi = torch.topk(pred, k=pred_k, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        plan = target_blk._build_prefetch_plan_from_prob_predictions(
            topi.to(torch.long),
            topv.to(dtype=block_output.dtype),
            protect_ids=protect,
            budget=int(budget),
        )
        if len(plan) == 0:
            return
        target_blk._admit_prefetch_plan(plan, group_end=target_layer + 1)

    def _maybe_prefetch_next_layer_from_rs_cache(self) -> None:
        # Runtime refactor: group-start computes the RS plan once and schedules
        # prefetch directly into each FUTURE target layer cache. Do not recompute
        # / re-admit prefetch per layer.
        return

    def _routing_rs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx not in SparseContext.routing_cache:
            raise RuntimeError(
                f"RS routing_cache miss at layer {self.layer_idx}. "
                f"This usually means the group-start layer did not recompute RS cache "
                f"for the current forward."
            )
        return SparseContext.routing_cache[self.layer_idx]

    def _routing_for_execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        真实 expert execution 使用哪一种 routing：
        - expert_source=router: 真实执行由 router 决定
        - expert_source=rs:     真实执行直接由 RS 预测决定
        """
        if self.expert_source == "router":
            return self._routing_router(x)

        # expert_source == "rs"
        if self.layer_idx not in SparseContext.routing_cache:
            if getattr(SparseContext, "rs_bank_layout", "group") == "next_layer":
                # A next-layer predictor set cannot predict layer 0 before the
                # first layer has run. Prefill also uses the ordinary router.
                # This keeps the quality/runtime comparison well-defined:
                # L0/prefill are router, L1+ decode may use previous-layer RS.
                return self._routing_router(x)
            raise RuntimeError(
                f"RS routing_cache miss at layer {self.layer_idx}. "
                f"This usually means the group-start layer did not recompute RS cache "
                f"for the current forward."
            )
        w, idx = self._routing_rs(x)

        # 为了兼容下游可能读取 router_logits 的路径，构造一个 pseudo logits
        # 非 top-k expert 置为极小值，top-k expert 用 log(prob)
        router_logits = torch.full(
            (w.shape[0], self.num_experts),
            fill_value=-1e9,
            device=w.device,
            dtype=torch.float32,
        )
        router_logits.scatter_(
            1,
            idx,
            torch.log(w.float().clamp_min(self.rs_prob_to_logit_eps)),
        )
        return w, idx, router_logits

    def _rs_should_fallback(self, rs_w: torch.Tensor) -> bool:
        return False

    def _estimate_request_owner_rank_single(self, idx_1: torch.Tensor, w_1: torch.Tensor) -> int:
        flat_i = idx_1.reshape(-1).to(torch.long)
        flat_w = w_1.reshape(-1).float()
        flat_r = self._resolve_dest_for_flat_experts(flat_i, reason="request_owner", count_metrics=False)
        rank_scores = torch.zeros((self.comm_world,), device=flat_w.device, dtype=torch.float32)
        rank_scores.scatter_add_(0, flat_r, flat_w)
        return int(torch.argmax(rank_scores).item())

    def _estimate_request_owner_ranks(self, idx: torch.Tensor, w: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        idx_bs = idx.view(batch_size, seq_len, self.top_k)
        w_bs = w.view(batch_size, seq_len, self.top_k)
        owners = []
        for b in range(batch_size):
            owners.append(self._estimate_request_owner_rank_single(idx_bs[b], w_bs[b]))
        return torch.tensor(owners, device=idx.device, dtype=torch.long)

    def _maybe_wait_prefetch(self, eid: int) -> None:
        if self.prefetch_stream is None:
            return
        ev = self._prefetch_events.pop(eid, None)
        if ev is not None:
            completed_before_use = False
            try:
                completed_before_use = bool(ev.query())
            except Exception:
                completed_before_use = False
            if completed_before_use:
                self._record_rs_counter("prefetch_complete_before_use_count", 1)
            # The prefetch is complete after this wait returns even when it was
            # not complete before use.  This gives a consistent invariant:
            # complete_before_use <= completed <= issued.
            cur_stream = torch.cuda.current_stream(self.device)
            wait_timing_sampled = (
                _env_flag("SPARSE_PREFETCH_WAIT_TIMING", "0")
                and self.metrics is not None
                and self._timing_sampled("SPARSE_PREFETCH_WAIT_TIMING_SAMPLE_RATE", "prefetch_wait", eid)
            )
            if wait_timing_sampled:
                try:
                    start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    start_ev.record(cur_stream)
                    cur_stream.wait_event(ev)
                    end_ev.record(cur_stream)
                    _append_prefetch_wait_event(self.metrics, start_ev, end_ev, layer_idx=self.layer_idx)
                    self._record_rs_counter("prefetch_completed_count", 1)
                    return
                except Exception:
                    pass
            cur_stream.wait_event(ev)
            self._record_rs_counter("prefetch_completed_count", 1)


    def _on_demand_access(self, eid: int) -> None:
        self._expert_access_freq[eid] = int(self._expert_access_freq.get(eid, 0)) + 1
        if str(SparseContext.current_phase) == "decode":
            self._last_used_decode_step[int(eid)] = int(getattr(SparseContext, "runtime_call_id", -1))
        if eid in self._local_eid_set and self._expert_access_freq[eid] >= 2:
            try:
                self._cache.promote_hot(eid)
            except Exception:
                pass

    def _should_prefer_hot(self, eid: int, is_prefetch: bool) -> bool:
        if is_prefetch:
            return False
        if self._should_mark_prefill_scp_transient(int(eid), is_prefetch=False):
            return False
        if eid not in self._local_eid_set:
            return False
        if self.cache_policy == "lfu":
            return int(self._expert_access_freq.get(eid, 0)) >= 1
        return int(self._expert_access_freq.get(eid, 0)) >= 2

    def _record_prefetch_event(self, eid: int) -> None:
        if self.prefetch_stream is None:
            return
        try:
            ev = torch.cuda.Event(enable_timing=False, blocking=False)
            ev.record(self.prefetch_stream)
            self._prefetch_events[eid] = ev
        except Exception:
            pass

    def _record_demand_h2d_window_begin(self) -> Optional[torch.cuda.Event]:
        if self.metrics is None:
            return None
        if not _env_flag("SPARSE_DEMAND_H2D_TIMING", "0"):
            return None
        if not self._timing_sampled("SPARSE_DEMAND_H2D_TIMING_SAMPLE_RATE", "demand_h2d"):
            return None
        try:
            ev = torch.cuda.Event(enable_timing=True, blocking=False)
            ev.record(torch.cuda.current_stream(self.device))
            return ev
        except Exception:
            return None

    def _record_demand_h2d_window_end(self, start_ev: Optional[torch.cuda.Event]) -> None:
        if self.metrics is None or start_ev is None:
            return
        try:
            end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
            end_ev.record(torch.cuda.current_stream(self.device))
            _append_demand_h2d_event(self.metrics, start_ev, end_ev, layer_idx=self.layer_idx)
        except Exception:
            return

    def clear_oracle_runtime_state(self) -> None:
        self._oracle_gpu_fused.clear()
        self._oracle_gpu_module.clear()

    def oracle_prepare_current_layer(self) -> None:
        if self.oracle_mode != "replay":
            return
        call_id = int(SparseContext.oracle_call_id)
        if call_id < 0:
            return

        # If this layer is already the active oracle resident for the current call,
        # avoid redundant control-plane work.
        if SparseContext.oracle_active_layer_idx is not None and int(SparseContext.oracle_active_layer_idx) == int(self.layer_idx):
            return

        prev_active = SparseContext.oracle_active_layer_idx
        if prev_active is not None and int(prev_active) != int(self.layer_idx):
            prev_blk = SparseContext.layer_blocks.get(int(prev_active), None)
            if prev_blk is not None:
                prev_blk.clear_oracle_runtime_state()

        self.clear_oracle_runtime_state()
        eids = SparseContext.get_oracle_layer_experts(call_id, self.layer_idx)
        self.oracle_stage_local_experts(eids)
        SparseContext.oracle_active_layer_idx = int(self.layer_idx)

    def oracle_stage_local_experts(self, expert_ids: List[int]) -> None:
        if self.oracle_mode != "replay":
            return
        uniq = sorted(set(int(x) for x in expert_ids if int(x) in self._local_eid_set))
        if len(uniq) == 0:
            return

        for eid in uniq:
            if self.experts_kind == "fused":
                if eid in self._oracle_gpu_fused:
                    continue
                gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=False)
                down = self.local_down_cpu[eid].to(self.device, non_blocking=False)
                self._oracle_gpu_fused[eid] = (gate_up, down)
                if self.metrics is not None:
                    self.metrics.oracle_stage_h2d_bytes += tensor_nbytes(gate_up) + tensor_nbytes(down)
                    self.metrics.oracle_stage_event_count += 1
            else:
                if eid in self._oracle_gpu_module:
                    continue
                gw = self.local_gate_cpu[eid].to(self.device, non_blocking=False)
                uw = self.local_up_cpu[eid].to(self.device, non_blocking=False)
                dw = self.local_down_cpu2[eid].to(self.device, non_blocking=False)
                self._oracle_gpu_module[eid] = (gw, uw, dw)
                if self.metrics is not None:
                    self.metrics.oracle_stage_h2d_bytes += tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw)
                    self.metrics.oracle_stage_event_count += 1


    def _maybe_upgrade_prefill_scp_transient_on_decode(self, eid: int) -> None:
        if self._current_stage() != "decode":
            return
        if not _env_flag("SPARSE_EPIPE_PROMOTE_TRANSIENT_ON_DECODE_HIT", "1"):
            return
        try:
            if getattr(self._cache, "is_prefill_scp_transient", lambda _eid: False)(int(eid)):
                self._record_metric_counter("prefill_scp_transient_hit_count", 1)
                if getattr(self._cache, "clear_prefill_scp_transient", lambda _eid: False)(int(eid)):
                    self._record_metric_counter("prefill_scp_transient_promote_count", 1)
                    self._record_epipe_counter("epipe_prefill_transient_promote_count", 1)
        except Exception:
            return

    def _get_local_fused(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.oracle_mode == "replay":
            oracle_payload = self._oracle_gpu_fused.get(int(eid), None)
            if oracle_payload is not None:
                self._record_cache_hit()
                return oracle_payload
        if self.expert_offload == "gpu":
            return self.local_gate_up_gpu[eid], self.local_down_gpu[eid]
        cached = self._cache.get(eid)
        if cached is not None:
            self._maybe_upgrade_prefill_scp_transient_on_decode(eid)
            if self.metrics is not None:
                if eid in self._prefetched_ids_this_group:
                    self._record_cache_hit(is_prefetch_hit=True)
                    self._prefetch_used_ids_this_group.add(eid)
                    if eid in self._transition_prefetched_ids:
                        self._transition_used_ids.add(int(eid))
                        self._record_metric_counter("transition_prefetch_hit", 1)
                    try:
                        self._cache.unpin(eid)
                        self._cache.promote_hot(eid)
                    except Exception:
                        pass
                else:
                    self._record_cache_hit()
            self._maybe_wait_prefetch(eid)
            return cached
        extra_cached = self._epipe_prefill_extra_cache_get(eid, touch=True)
        if extra_cached is not None:
            return extra_cached
        self._record_cache_miss()
        _stall_begin = self._record_demand_h2d_window_begin()
        required_bytes = self._epipe_estimate_expert_nbytes(eid)
        if required_bytes > 0 and not self._demand_h2d_make_memory_room(required_bytes):
            self._record_demand_h2d_window_end(_stall_begin)
            raise RuntimeError(
                "Insufficient CUDA memory headroom before local fused expert demand H2D: "
                f"layer={int(self.layer_idx)} expert={int(eid)} "
                f"required_mb={required_bytes / (1024.0 * 1024.0):.2f} "
                f"min_free_mb={self._demand_h2d_min_free_mb():.2f}"
            )
        gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
        down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
        self._record_demand_h2d_window_end(_stall_begin)
        self._record_h2d_bytes(tensor_nbytes(gate_up) + tensor_nbytes(down), is_prefetch=False)
        self._on_demand_access(eid)
        self._cache_put_guarded(eid, (gate_up, down), pinned=False, prefer_hot=self._should_prefer_hot(eid, False))
        self._global_cache_enforce_after_demand()
        return gate_up, down

    def _get_local_module_weights(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.oracle_mode == "replay":
            oracle_payload = self._oracle_gpu_module.get(int(eid), None)
            if oracle_payload is not None:
                self._record_cache_hit()
                return oracle_payload
        cached = self._cache.get(eid)
        if cached is not None:
            self._maybe_upgrade_prefill_scp_transient_on_decode(eid)
            if self.metrics is not None:
                if eid in self._prefetched_ids_this_group:
                    self._record_cache_hit(is_prefetch_hit=True)
                    self._prefetch_used_ids_this_group.add(eid)
                    if eid in self._transition_prefetched_ids:
                        self._transition_used_ids.add(int(eid))
                        self._record_metric_counter("transition_prefetch_hit", 1)
                    try:
                        self._cache.unpin(eid)
                        self._cache.promote_hot(eid)
                    except Exception:
                        pass
                else:
                    self._record_cache_hit()
            self._maybe_wait_prefetch(eid)
            return cached
        extra_cached = self._epipe_prefill_extra_cache_get(eid, touch=True)
        if extra_cached is not None:
            return extra_cached
        self._record_cache_miss()
        _stall_begin = self._record_demand_h2d_window_begin()
        required_bytes = self._epipe_estimate_expert_nbytes(eid)
        if required_bytes > 0 and not self._demand_h2d_make_memory_room(required_bytes):
            self._record_demand_h2d_window_end(_stall_begin)
            raise RuntimeError(
                "Insufficient CUDA memory headroom before local expert demand H2D: "
                f"layer={int(self.layer_idx)} expert={int(eid)} "
                f"required_mb={required_bytes / (1024.0 * 1024.0):.2f} "
                f"min_free_mb={self._demand_h2d_min_free_mb():.2f}"
            )
        gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
        uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
        dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
        self._record_demand_h2d_window_end(_stall_begin)
        self._record_h2d_bytes(tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw), is_prefetch=False)
        self._on_demand_access(eid)
        self._cache_put_guarded(eid, (gw, uw, dw), pinned=False, prefer_hot=self._should_prefer_hot(eid, False))
        self._global_cache_enforce_after_demand()
        return gw, uw, dw

    def _get_fused_from_cpu_once(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eid = int(eid)
        cached_cpu = self._host_fused_cpu_cache.get(eid, None)
        if cached_cpu is None:
            gate_up_cpu = self.experts.gate_up_proj[eid].detach().to("cpu", self.dtype).contiguous().pin_memory()
            down_cpu = self.experts.down_proj[eid].detach().to("cpu", self.dtype).contiguous().pin_memory()
            self._host_fused_cpu_cache[eid] = (gate_up_cpu, down_cpu)
        else:
            gate_up_cpu, down_cpu = cached_cpu

        gate_up = gate_up_cpu.to(self.device, non_blocking=True)
        down = down_cpu.to(self.device, non_blocking=True)
        return gate_up, down

    def _get_module_from_cpu_once(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eid = int(eid)
        cached_cpu = self._host_module_cpu_cache.get(eid, None)
        if cached_cpu is None:
            m = self.expert_modules[eid]
            gw_cpu = m.gate_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
            uw_cpu = m.up_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
            dw_cpu = m.down_proj.weight.detach().to("cpu", self.dtype).contiguous().pin_memory()
            self._host_module_cpu_cache[eid] = (gw_cpu, uw_cpu, dw_cpu)
        else:
            gw_cpu, uw_cpu, dw_cpu = cached_cpu

        gw = gw_cpu.to(self.device, non_blocking=True)
        uw = uw_cpu.to(self.device, non_blocking=True)
        dw = dw_cpu.to(self.device, non_blocking=True)
        return gw, uw, dw

    def _get_any_fused_from_cpu_cached(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        eid = int(eid)
        cached = self._cache.get(eid)
        if cached is not None:
            self._maybe_upgrade_prefill_scp_transient_on_decode(eid)
            self._record_cache_hit()
            self._maybe_wait_prefetch(eid)
            return cached
        extra_cached = self._epipe_prefill_extra_cache_get(eid, touch=True)
        if extra_cached is not None:
            return extra_cached
        self._record_cache_miss()
        _stall_begin = self._record_demand_h2d_window_begin()
        required_bytes = self._epipe_estimate_expert_nbytes(eid)
        if required_bytes > 0 and not self._demand_h2d_make_memory_room(required_bytes):
            self._record_demand_h2d_window_end(_stall_begin)
            raise RuntimeError(
                "Insufficient CUDA memory headroom before any fused expert demand H2D: "
                f"layer={int(self.layer_idx)} expert={int(eid)} "
                f"required_mb={required_bytes / (1024.0 * 1024.0):.2f} "
                f"min_free_mb={self._demand_h2d_min_free_mb():.2f}"
            )
        gate_up, down = self._get_fused_from_cpu_once(eid)
        self._record_demand_h2d_window_end(_stall_begin)
        self._record_h2d_bytes(tensor_nbytes(gate_up) + tensor_nbytes(down), is_prefetch=False)
        self._on_demand_access(eid)
        self._cache_put_guarded(eid, (gate_up, down), pinned=False, prefer_hot=self._should_prefer_hot(eid, False))
        self._global_cache_enforce_after_demand()
        return gate_up, down

    def _get_any_module_from_cpu_cached(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eid = int(eid)
        cached = self._cache.get(eid)
        if cached is not None:
            self._maybe_upgrade_prefill_scp_transient_on_decode(eid)
            self._record_cache_hit()
            self._maybe_wait_prefetch(eid)
            return cached
        extra_cached = self._epipe_prefill_extra_cache_get(eid, touch=True)
        if extra_cached is not None:
            return extra_cached
        self._record_cache_miss()
        _stall_begin = self._record_demand_h2d_window_begin()
        required_bytes = self._epipe_estimate_expert_nbytes(eid)
        if required_bytes > 0 and not self._demand_h2d_make_memory_room(required_bytes):
            self._record_demand_h2d_window_end(_stall_begin)
            raise RuntimeError(
                "Insufficient CUDA memory headroom before any module expert demand H2D: "
                f"layer={int(self.layer_idx)} expert={int(eid)} "
                f"required_mb={required_bytes / (1024.0 * 1024.0):.2f} "
                f"min_free_mb={self._demand_h2d_min_free_mb():.2f}"
            )
        gw, uw, dw = self._get_module_from_cpu_once(eid)
        self._record_demand_h2d_window_end(_stall_begin)
        self._record_h2d_bytes(tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw), is_prefetch=False)
        self._on_demand_access(eid)
        self._cache_put_guarded(eid, (gw, uw, dw), pinned=False, prefer_hot=self._should_prefer_hot(eid, False))
        self._global_cache_enforce_after_demand()
        return gw, uw, dw

    def _epipe_payload_nbytes(self, payload) -> int:
        if isinstance(payload, torch.Tensor):
            return int(tensor_nbytes(payload))
        if isinstance(payload, (tuple, list)):
            return int(sum(self._epipe_payload_nbytes(x) for x in payload))
        return 0

    def _epipe_estimate_expert_nbytes(self, eid: int) -> int:
        eid = int(eid)
        try:
            if self.experts_kind == "fused":
                if eid in self.local_gate_up_cpu and eid in self.local_down_cpu:
                    return int(tensor_nbytes(self.local_gate_up_cpu[eid]) + tensor_nbytes(self.local_down_cpu[eid]))
            else:
                if eid in self.local_gate_cpu and eid in self.local_up_cpu and eid in self.local_down_cpu2:
                    return int(
                        tensor_nbytes(self.local_gate_cpu[eid])
                        + tensor_nbytes(self.local_up_cpu[eid])
                        + tensor_nbytes(self.local_down_cpu2[eid])
                    )
        except Exception:
            return 0
        return 0

    def _epipe_memory_headroom_ok(self, required_bytes: int, *, min_free_mb: Optional[float] = None) -> bool:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            return False
        margin_mb = float(self._epipe_min_free_mb if min_free_mb is None else min_free_mb)
        margin_bytes = int(margin_mb * 1024.0 * 1024.0)
        return int(free_bytes) >= int(required_bytes) + int(margin_bytes)

    def _demand_h2d_min_free_mb(self) -> float:
        # Demand H2D is the correctness path: if a requested expert is not
        # resident, the runtime must load it.  Do not reuse EPIPE's much larger
        # staging headroom here; that headroom is a speculative-admission guard
        # and can incorrectly abort valid demand loads on 8GB cards.  Keep only
        # a small configurable safety margin to avoid the original near-zero-free
        # CUDA OOM cliff.
        raw = os.environ.get("SPARSE_DEMAND_H2D_MIN_FREE_MB", "32")
        try:
            return max(0.0, float(raw or 0.0))
        except Exception:
            return 0.0

    def _demand_h2d_make_memory_room(self, required_bytes: int) -> bool:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return True
        min_free_mb = self._demand_h2d_min_free_mb()
        if self._epipe_memory_headroom_ok(int(required_bytes), min_free_mb=min_free_mb):
            return True

        cleared = False
        try:
            clear_extra = getattr(self, "_epipe_prefill_extra_cache_clear", None)
            if callable(clear_extra):
                cleared = int(clear_extra()) > 0 or cleared
        except Exception:
            pass
        try:
            cache_clear = getattr(getattr(self, "_cache", None), "clear", None)
            if callable(cache_clear):
                cache_clear()
                cleared = True
        except Exception:
            pass
        try:
            getattr(self, "_prefetched_ids_this_group", set()).clear()
            getattr(self, "_prefetch_used_ids_this_group", set()).clear()
            getattr(self, "_prefetch_expire_layer", {}).clear()
            getattr(self, "_transition_prefetched_ids", set()).clear()
            getattr(self, "_transition_used_ids", set()).clear()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if cleared:
            self._record_metric_counter("prefetch_oom_skip_count", 1)
        return self._epipe_memory_headroom_ok(int(required_bytes), min_free_mb=min_free_mb)

    def _epipe_record_staged_demand_h2d(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        if nbytes <= 0:
            return
        self._record_h2d_bytes(nbytes, is_prefetch=False)
        self._record_epipe_counter("epipe_demand_h2d_bytes", nbytes)
        self._record_epipe_counter("epipe_staged_h2d_bytes", nbytes)

    def _epipe_load_payload_to_device(self, eid: int) -> Tuple[Any, int]:
        eid = int(eid)
        if self.experts_kind == "fused":
            if eid not in self.local_gate_up_cpu or eid not in self.local_down_cpu:
                raise KeyError(f"EPIPE fused local CPU expert not available: eid={eid}")
            gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
            down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
            return (gate_up, down), int(tensor_nbytes(gate_up) + tensor_nbytes(down))
        if eid not in self.local_gate_cpu or eid not in self.local_up_cpu or eid not in self.local_down_cpu2:
            raise KeyError(f"EPIPE module local CPU expert not available: eid={eid}")
        gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
        uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
        dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
        return (gw, uw, dw), int(tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw))

    def _epipe_get_prefetch_pending_event(self, eid: int):
        if self.prefetch_stream is None:
            return None
        ev = self._prefetch_events.get(int(eid), None)
        if ev is None:
            return None
        try:
            if bool(ev.query()):
                self._prefetch_events.pop(int(eid), None)
                self._record_rs_counter("prefetch_complete_before_use_count", 1)
                self._record_rs_counter("prefetch_completed_count", 1)
                return None
        except Exception:
            return ev
        return ev

    def _epipe_record_cached_payload_use(self, eid: int, *, wait_prefetch: bool = True) -> None:
        eid = int(eid)
        self._maybe_upgrade_prefill_scp_transient_on_decode(eid)
        if self.metrics is not None:
            if eid in self._prefetched_ids_this_group:
                self._record_cache_hit(is_prefetch_hit=True)
                self._prefetch_used_ids_this_group.add(eid)
                if eid in self._transition_prefetched_ids:
                    self._transition_used_ids.add(int(eid))
                    self._record_metric_counter("transition_prefetch_hit", 1)
                try:
                    self._cache.unpin(eid)
                    self._cache.promote_hot(eid)
                except Exception:
                    pass
            else:
                self._record_cache_hit()
        if wait_prefetch:
            self._maybe_wait_prefetch(eid)

    def _epipe_get_ready_payload_for_compute(self, eid: int):
        from_extra_cache = False
        if (
            str(self._current_stage() or "").strip().lower() == "prefill"
            and bool(getattr(self, "_epipe_prefill_no_touch_ready", False))
            and hasattr(self._cache, "peek")
        ):
            try:
                cached = self._cache.peek(int(eid))
            except Exception:
                cached = self._cache.get(int(eid))
        else:
            cached = self._cache.get(int(eid))
        if cached is None:
            cached = self._epipe_prefill_extra_cache_get(
                int(eid),
                touch=not (
                    str(self._current_stage() or "").strip().lower() == "prefill"
                    and bool(getattr(self, "_epipe_prefill_no_touch_ready", False))
                ),
            )
            from_extra_cache = cached is not None
        if cached is None:
            return None
        if not from_extra_cache:
            self._epipe_record_cached_payload_use(int(eid), wait_prefetch=True)
        return cached

    def _plain_ep_compute_segment(
        self,
        *,
        eid: int,
        cur: torch.Tensor,
        cur_w: torch.Tensor,
        eid_is_local: bool,
        allow_nonlocal: bool,
        cp_cuda_timing: bool = False,
        cp_sampled: bool = False,
        cp_compute_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Tuple[torch.Tensor, float]:
        cpu_ms = 0.0
        eid = int(eid)
        if self.experts_kind == "fused":
            gate_up, down = self._get_local_fused(eid) if eid_is_local else self._get_any_fused_from_cpu_cached(eid)
            if cp_cuda_timing:
                cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_start.record(torch.cuda.current_stream(self.device))
            elif cp_sampled:
                cp_cpu_start = _hclm_cp_now_ns()
            gate_up_out = F.linear(cur, gate_up)
            gate, up = gate_up_out.chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate) * up, down)
            if cp_cuda_timing:
                cp_ev_end.record(torch.cuda.current_stream(self.device))
                if cp_compute_events is not None:
                    cp_compute_events.append((cp_ev_start, cp_ev_end))
            elif cp_sampled:
                cpu_ms += max(0.0, float(_hclm_cp_now_ns() - cp_cpu_start) / 1_000_000.0)
            return y * cur_w, float(cpu_ms)

        if self.expert_offload == "gpu":
            if not eid_is_local and not allow_nonlocal:
                raise RuntimeError(f"Plain EP cannot execute non-local GPU expert eid={eid}")
            if cp_cuda_timing:
                cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_start.record(torch.cuda.current_stream(self.device))
            elif cp_sampled:
                cp_cpu_start = _hclm_cp_now_ns()
            y = self.expert_modules[eid](cur)
            if cp_cuda_timing:
                cp_ev_end.record(torch.cuda.current_stream(self.device))
                if cp_compute_events is not None:
                    cp_compute_events.append((cp_ev_start, cp_ev_end))
            elif cp_sampled:
                cpu_ms += max(0.0, float(_hclm_cp_now_ns() - cp_cpu_start) / 1_000_000.0)
            return y * cur_w, float(cpu_ms)

        gw, uw, dw = self._get_local_module_weights(eid) if eid_is_local else self._get_any_module_from_cpu_cached(eid)
        if cp_cuda_timing:
            cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_start.record(torch.cuda.current_stream(self.device))
        elif cp_sampled:
            cp_cpu_start = _hclm_cp_now_ns()
        y = self._module_expert_forward(cur, gw, uw, dw)
        if cp_cuda_timing:
            cp_ev_end.record(torch.cuda.current_stream(self.device))
            if cp_compute_events is not None:
                cp_compute_events.append((cp_ev_start, cp_ev_end))
        elif cp_sampled:
            cpu_ms += max(0.0, float(_hclm_cp_now_ns() - cp_cpu_start) / 1_000_000.0)
        return y * cur_w, float(cpu_ms)

    def _module_expert_forward(self, cur, gw, uw, dw):
        """Optionally retain native single-request decode GEMV arithmetic.

        EP can merge two ranks' one-token requests into a larger expert GEMM.
        FP16 kernel rounding can then perturb later router choices. This guard
        keeps the native per-token arithmetic without changing router decisions.
        It is a shared executor option, independent of scheduling policy.
        """
        native_items = (
            bool(getattr(self, "_native_decode_granularity", False))
            and self._current_stage() == "decode"
            and int(getattr(self, "_current_forward_batch_size", 0)) == 1
            and int(getattr(self, "_current_forward_seq_len", 0)) == 1
            and cur.shape[0] > 1
        )
        def forward(inputs):
            gate = F.linear(inputs, gw)
            up = F.linear(inputs, uw)
            return F.linear(self.act_fn(gate) * up, dw)
        if native_items:
            return torch.cat([forward(cur[row:row+1]) for row in range(cur.shape[0])], dim=0)
        return forward(cur)

    def _epipe_compute_segment(
        self,
        *,
        eid: int,
        cur: torch.Tensor,
        cur_w: torch.Tensor,
        eid_is_local: bool,
        allow_nonlocal: bool,
        payload_override=None,
        cp_cuda_timing: bool = False,
        cp_sampled: bool = False,
        cp_compute_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None,
    ) -> Tuple[torch.Tensor, float]:
        cpu_ms = 0.0
        eid = int(eid)
        if self.experts_kind == "fused":
            if payload_override is None:
                gate_up, down = self._get_local_fused(eid) if eid_is_local else self._get_any_fused_from_cpu_cached(eid)
            else:
                gate_up, down = payload_override
            if cp_cuda_timing:
                cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_start.record(torch.cuda.current_stream(self.device))
            elif cp_sampled:
                cp_cpu_start = _hclm_cp_now_ns()
            gate_up_out = F.linear(cur, gate_up)
            gate, up = gate_up_out.chunk(2, dim=-1)
            y = F.linear(self.act_fn(gate) * up, down)
            if cp_cuda_timing:
                cp_ev_end.record(torch.cuda.current_stream(self.device))
                if cp_compute_events is not None:
                    cp_compute_events.append((cp_ev_start, cp_ev_end))
            elif cp_sampled:
                cpu_ms += max(0.0, float(_hclm_cp_now_ns() - cp_cpu_start) / 1_000_000.0)
            return y * cur_w, float(cpu_ms)

        if self.expert_offload == "gpu" and payload_override is None:
            if cp_cuda_timing:
                cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
                cp_ev_start.record(torch.cuda.current_stream(self.device))
            elif cp_sampled:
                cp_cpu_start = _hclm_cp_now_ns()
            y = self.expert_modules[eid](cur)
            if cp_cuda_timing:
                cp_ev_end.record(torch.cuda.current_stream(self.device))
                if cp_compute_events is not None:
                    cp_compute_events.append((cp_ev_start, cp_ev_end))
            elif cp_sampled:
                cpu_ms += max(0.0, float(_hclm_cp_now_ns() - cp_cpu_start) / 1_000_000.0)
            return y * cur_w, float(cpu_ms)

        if payload_override is None:
            gw, uw, dw = self._get_local_module_weights(eid) if eid_is_local else self._get_any_module_from_cpu_cached(eid)
        else:
            gw, uw, dw = payload_override
        if cp_cuda_timing:
            cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_start.record(torch.cuda.current_stream(self.device))
        elif cp_sampled:
            cp_cpu_start = _hclm_cp_now_ns()
        y = self._module_expert_forward(cur, gw, uw, dw)
        if cp_cuda_timing:
            cp_ev_end.record(torch.cuda.current_stream(self.device))
            if cp_compute_events is not None:
                cp_compute_events.append((cp_ev_start, cp_ev_end))
        elif cp_sampled:
            cpu_ms += max(0.0, float(_hclm_cp_now_ns() - cp_cpu_start) / 1_000_000.0)
        return y * cur_w, float(cpu_ms)

    def _try_execute_local_recv_groups_epipe(
        self,
        *,
        hidden_sorted: torch.Tensor,
        recv_i_sorted: torch.Tensor,
        w_sorted: torch.Tensor,
        segments: List[Tuple[int, int, int, bool]],
        allow_nonlocal: bool,
        cp_cuda_timing: bool,
        cp_sampled: bool,
        cp_compute_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> Optional[Tuple[torch.Tensor, float]]:
        if self._epipe_validate_output:
            raise NotImplementedError("SPARSE_EPIPE_VALIDATE_OUTPUT=1 is not implemented for the runtime hot path.")
        if self.expert_offload != "cpu" or self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        if len(segments) == 0:
            return None

        stage = self._current_stage()
        control_sampled = self._epipe_timing_sampled("control_upfront", stage)
        control_start_ns = _hclm_cp_now_ns() if control_sampled else 0
        enabled_for_stage = bool(self._epipe_enable if stage == "decode" else self._epipe_prefill_enable)
        shadow_for_stage = bool(
            self._epipe_shadow
            if stage == "decode"
            else (self._epipe_prefill_shadow or self._epipe_prefill_multistage_shadow)
        )
        min_free_mb = float(self._epipe_min_free_mb if stage == "decode" else self._epipe_prefill_min_free_mb)
        if not (enabled_for_stage or shadow_for_stage):
            return None
        ready: List[Tuple[int, int, int, bool]] = []
        missing: List[Tuple[int, int, int, bool]] = []
        for seg in segments:
            eid = int(seg[0])
            is_ready = bool(self._cache_contains_no_touch(eid))
            if is_ready:
                ready.append(seg)
            else:
                missing.append(seg)

        ready_items = int(sum(int(e) - int(s) for _eid, s, e, _local in ready))
        missing_items = int(sum(int(e) - int(s) for _eid, s, e, _local in missing))
        missing_h2d_bytes_est = int(sum(self._epipe_estimate_expert_nbytes(int(seg[0])) for seg in missing))
        self._epipe_record_prefill_multistage_shadow(ready, missing, ready_items, missing_items)
        try:
            resident_count = int(len(self._cache.resident_ids()))
            cache_capacity = int(getattr(self._cache, "capacity", 0))
            cache_free_slots = max(0, cache_capacity - resident_count)
        except Exception:
            cache_free_slots = 0

        if shadow_for_stage and not enabled_for_stage:
            self._record_epipe_counter("epipe_shadow_count", 1)
            self._record_epipe_counter("epipe_ready_expert_count", len(ready))
            self._record_epipe_counter("epipe_missing_expert_count", len(missing))
            self._record_epipe_counter("epipe_ready_item_count", ready_items)
            self._record_epipe_counter("epipe_missing_item_count", missing_items)
            self._record_epipe_counter("epipe_missing_h2d_bytes_est", missing_h2d_bytes_est)
            if cache_free_slots > 0:
                self._record_epipe_counter("epipe_cache_free_slot_count", cache_free_slots)
            elif len(missing) > 0 and self._epipe_transient_staging:
                self._record_epipe_counter("epipe_cache_full_staged_count", 1)
            return None

        if not enabled_for_stage:
            return None

        def fallback(reason: str) -> None:
            self._record_epipe_counter("epipe_fallback_count", 1)
            if reason:
                self._record_epipe_counter(reason, 1)

        if self._epipe_decode_only and stage != "decode" and not bool(self._epipe_prefill_enable):
            fallback("epipe_fallback_prefill_disabled")
            return None
        if not self._epipe_no_evict:
            raise NotImplementedError("SPARSE_EPIPE_NO_EVICT=0 is not implemented in EPIPE-Lite.")
        if self._epipe_max_staged_experts != 1:
            raise NotImplementedError("EPIPE-Lite currently supports only one staged missing expert.")
        if len(ready) < int(self._epipe_min_ready_experts) or ready_items < int(self._epipe_min_ready_items):
            fallback("epipe_fallback_no_ready")
            return None
        if len(missing) < int(self._epipe_min_missing_experts):
            fallback("epipe_fallback_no_missing")
            return None
        if missing_h2d_bytes_est < int(self._epipe_min_missing_h2d_bytes):
            fallback("")
            return None

        staged = missing[0]
        staged_eid = int(staged[0])
        staged_nbytes_est = int(self._epipe_estimate_expert_nbytes(staged_eid))
        if staged_nbytes_est <= 0:
            fallback("epipe_fallback_exception")
            return None

        can_cache_insert = (
            cache_free_slots > 0
            and self.cache_policy != "tiered"
            and not bool(getattr(self, "_global_expert_cache_enabled", False))
            and not (stage == "prefill" and not bool(self._epipe_prefill_commit))
        )
        use_transient = not bool(can_cache_insert)
        if use_transient:
            if not self._epipe_transient_staging:
                fallback("epipe_fallback_cache_full_no_temp")
                return None
            if not self._epipe_memory_headroom_ok(staged_nbytes_est, min_free_mb=min_free_mb):
                fallback("epipe_fallback_no_memory_headroom")
                return None

        if self.epipe_stream is None:
            fallback("epipe_fallback_exception")
            return None

        if control_sampled:
            control_ms = max(0.0, float(_hclm_cp_now_ns() - int(control_start_ns)) / 1_000_000.0)
            self._record_epipe_float("sampled_epipe_control_overhead_ms", control_ms)
        timing_sampled = self._epipe_timing_sampled("upfront", staged_eid)
        payload = None
        stage_event = None
        h2d_start_ev = None
        h2d_end_ev = None
        wait_start_ev = None
        wait_end_ev = None
        try:
            with torch.cuda.stream(self.epipe_stream):
                if timing_sampled:
                    h2d_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    h2d_start_ev.record(self.epipe_stream)
                payload, copied_nbytes = self._epipe_load_payload_to_device(staged_eid)
                if timing_sampled:
                    h2d_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    h2d_end_ev.record(self.epipe_stream)
                stage_event = torch.cuda.Event(enable_timing=False, blocking=False)
                stage_event.record(self.epipe_stream)
        except Exception:
            fallback("epipe_fallback_exception")
            payload = None
            return None

        out_sorted = torch.zeros_like(hidden_sorted)
        cp_compute_cpu_ms = 0.0
        ready_order = list(ready)
        if self._epipe_sort_ready_by_work:
            ready_order.sort(key=lambda seg: int(seg[2]) - int(seg[1]), reverse=True)

        try:
            for eid, s, e, eid_is_local in ready_order:
                cur = hidden_sorted[int(s):int(e)]
                cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
                ready_payload = None
                if stage == "prefill" and bool(getattr(self, "_epipe_prefill_no_touch_ready", False)):
                    ready_payload = self._epipe_get_ready_payload_for_compute(int(eid))
                y, cpu_ms = self._epipe_compute_segment(
                    eid=int(eid),
                    cur=cur,
                    cur_w=cur_w,
                    eid_is_local=bool(eid_is_local),
                    allow_nonlocal=bool(allow_nonlocal),
                    payload_override=ready_payload,
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
                out_sorted[int(s):int(e)] = y
                cp_compute_cpu_ms += float(cpu_ms)

            cur_stream = torch.cuda.current_stream(self.device)
            demand_wait_start = self._record_demand_h2d_window_begin()
            if timing_sampled:
                wait_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                wait_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                wait_start_ev.record(cur_stream)
            cur_stream.wait_event(stage_event)
            self._record_demand_h2d_window_end(demand_wait_start)
            if timing_sampled:
                wait_end_ev.record(cur_stream)
                if self.metrics is not None:
                    if self.metrics._pending_epipe_timing_events is None:
                        self.metrics._pending_epipe_timing_events = []
                    if h2d_start_ev is not None and h2d_end_ev is not None and wait_start_ev is not None and wait_end_ev is not None:
                        self.metrics._pending_epipe_timing_events.append(
                            (h2d_start_ev, h2d_end_ev, wait_start_ev, wait_end_ev, int(self.layer_idx))
                        )
            if isinstance(payload, (tuple, list)):
                for t in payload:
                    if isinstance(t, torch.Tensor):
                        t.record_stream(cur_stream)
            elif isinstance(payload, torch.Tensor):
                payload.record_stream(cur_stream)

            copied_nbytes = int(copied_nbytes)
            self._record_cache_miss()
            if stage != "prefill":
                self._on_demand_access(staged_eid)
            self._epipe_record_staged_demand_h2d(copied_nbytes)
            self._record_epipe_counter("epipe_staged_missing_count", 1)
            self._record_epipe_counter("epipe_enabled_count", 1)
            self._epipe_record_stage_success_by_stage()
            self._record_epipe_counter("epipe_staged_expert_count", 1)
            self._record_epipe_counter("epipe_no_evict_count", 1)
            self._record_epipe_counter("epipe_ready_expert_count", len(ready))
            self._record_epipe_counter("epipe_missing_expert_count", len(missing))
            self._record_epipe_counter("epipe_ready_item_count", ready_items)
            self._record_epipe_counter("epipe_missing_item_count", missing_items)
            self._record_epipe_counter("epipe_missing_h2d_bytes_est", missing_h2d_bytes_est)
            if cache_free_slots > 0:
                self._record_epipe_counter("epipe_cache_free_slot_count", cache_free_slots)
            if can_cache_insert:
                try:
                    resident_count_after = int(len(self._cache.resident_ids()))
                    cache_capacity_after = int(getattr(self._cache, "capacity", 0))
                    if resident_count_after < cache_capacity_after:
                        if stage == "prefill":
                            self._on_demand_access(staged_eid)
                        self._cache_put_guarded(
                            staged_eid,
                            payload,
                            pinned=False,
                            prefer_hot=self._should_prefer_hot(staged_eid, False),
                            prefill_scp_transient=bool(stage == "prefill" and self._epipe_prefill_mark_transient),
                        )
                        self._record_epipe_counter("epipe_cache_insert_count", 1)
                        if stage == "prefill":
                            self._record_epipe_counter("epipe_prefill_cache_insert_count", 1)
                        self._record_epipe_counter("epipe_deferred_lru_commit_count", 1)
                        if stage == "prefill" and self._epipe_prefill_mark_transient:
                            self._record_epipe_counter("epipe_prefill_transient_commit_count", 1)
                    else:
                        use_transient = True
                except Exception:
                    use_transient = True
            if use_transient:
                stored_extra = bool(
                    stage == "prefill"
                    and self._epipe_prefill_extra_cache_store(staged_eid, payload, copied_nbytes)
                )
                if not stored_extra:
                    self._record_epipe_counter("epipe_cache_full_staged_count", 1)
                    self._record_epipe_counter("epipe_transient_payload_count", 1)
                    self._epipe_record_prefill_no_commit_payload()

            eid, s, e, eid_is_local = staged
            cur = hidden_sorted[int(s):int(e)]
            cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
            y, cpu_ms = self._epipe_compute_segment(
                eid=int(eid),
                cur=cur,
                cur_w=cur_w,
                eid_is_local=bool(eid_is_local),
                allow_nonlocal=bool(allow_nonlocal),
                payload_override=payload,
                cp_cuda_timing=bool(cp_cuda_timing),
                cp_sampled=bool(cp_sampled),
                cp_compute_events=cp_compute_events,
            )
            out_sorted[int(s):int(e)] = y
            cp_compute_cpu_ms += float(cpu_ms)

            staged_key = (int(staged[0]), int(staged[1]), int(staged[2]))
            for seg in missing[1:]:
                if (int(seg[0]), int(seg[1]), int(seg[2])) == staged_key:
                    continue
                eid, s, e, eid_is_local = seg
                cur = hidden_sorted[int(s):int(e)]
                cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
                y, cpu_ms = self._epipe_compute_segment(
                    eid=int(eid),
                    cur=cur,
                    cur_w=cur_w,
                    eid_is_local=bool(eid_is_local),
                    allow_nonlocal=bool(allow_nonlocal),
                    payload_override=None,
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
                out_sorted[int(s):int(e)] = y
                cp_compute_cpu_ms += float(cpu_ms)
            payload = None
            return out_sorted, float(cp_compute_cpu_ms)
        except Exception:
            if stage_event is not None:
                try:
                    cur_stream = torch.cuda.current_stream(self.device)
                    cur_stream.wait_event(stage_event)
                    if isinstance(payload, (tuple, list)):
                        for t in payload:
                            if isinstance(t, torch.Tensor):
                                t.record_stream(cur_stream)
                    elif isinstance(payload, torch.Tensor):
                        payload.record_stream(cur_stream)
                except Exception:
                    pass
            payload = None
            fallback("epipe_fallback_exception")
            raise RuntimeError(
            f"EPIPE-Lite failed after staging expert {staged_eid}; refusing serial fallback to avoid duplicate H2D."
            )

    def _try_execute_local_recv_groups_epipe_rs_residual_scan(
        self,
        *,
        hidden_sorted: torch.Tensor,
        recv_i_sorted: torch.Tensor,
        w_sorted: torch.Tensor,
        segments: List[Tuple[int, int, int, bool]],
        allow_nonlocal: bool,
        cp_cuda_timing: bool,
        cp_sampled: bool,
        cp_compute_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> Optional[Tuple[torch.Tensor, float]]:
        _ = recv_i_sorted
        if self._epipe_validate_output:
            raise NotImplementedError("SPARSE_EPIPE_VALIDATE_OUTPUT=1 is not implemented for the runtime hot path.")
        if self.expert_offload != "cpu" or self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        if len(segments) == 0:
            return None
        if self._current_stage() != "decode" or self.planner_mode != "rs_prefetch":
            return None
        if not self._epipe_enable or self.epipe_stream is None:
            return None

        control_sampled = self._epipe_timing_sampled("control_rs_residual_scan")
        control_start_ns = _hclm_cp_now_ns() if control_sampled else 0
        self._record_epipe_counter("epipe_rs_buffered_calls", 1)
        self._record_epipe_counter("epipe_rs_residual_scan_calls", 1)
        out_sorted = torch.zeros_like(hidden_sorted)
        cp_compute_cpu_ms = 0.0

        def fallback(reason: str) -> None:
            self._record_epipe_counter("epipe_fallback_count", 1)
            if reason:
                self._record_epipe_counter(reason, 1)

        def observe_trigger() -> None:
            if self._epipe_rs_rolling_gate:
                self._epipe_rs_gate_triggers += 1

        def residual_present_for_gate() -> bool:
            for gate_seg in segments:
                gate_eid = int(gate_seg[0])
                ev = self._prefetch_events.get(gate_eid, None)
                if ev is not None:
                    try:
                        if bool(getattr(self, "_epipe_rs_handle_late_prefetch", False)) and not bool(ev.query()):
                            return True
                    except Exception:
                        if bool(getattr(self, "_epipe_rs_handle_late_prefetch", False)):
                            return True
                if (
                    bool(getattr(self, "_epipe_rs_handle_demand_miss", False))
                    and self._epipe_rs_demand_stage_gate_can_open()
                    and not self._cache_contains_no_touch(gate_eid)
                ):
                    return True
            return False

        def maybe_skip_gate() -> bool:
            if not self._epipe_rs_rolling_gate:
                return False
            if int(self._epipe_rs_gate_skip_remaining) > 0:
                if residual_present_for_gate():
                    self._epipe_rs_gate_skip_remaining = 0
                    self._record_epipe_counter("epipe_rs_gate_reenable_count", 1)
                    return False
                self._epipe_rs_gate_skip_remaining -= 1
                self._record_epipe_counter("epipe_rs_gate_skip_count", 1)
                return True
            self._epipe_rs_gate_calls += 1
            if (
                int(self._epipe_rs_gate_calls) >= int(self._epipe_rs_gate_window)
                and int(self._epipe_rs_gate_triggers) < int(self._epipe_rs_gate_min_triggers)
            ):
                self._epipe_rs_gate_calls = 0
                self._epipe_rs_gate_triggers = 0
                self._epipe_rs_gate_skip_remaining = int(self._epipe_rs_gate_skip_steps)
                if self._epipe_rs_gate_skip_remaining > 0:
                    self._record_epipe_counter("epipe_rs_gate_skip_count", 1)
                    self._epipe_rs_gate_skip_remaining -= 1
                    return True
            if int(self._epipe_rs_gate_calls) >= int(self._epipe_rs_gate_window):
                self._epipe_rs_gate_calls = 0
                self._epipe_rs_gate_triggers = 0
            return False

        def current_free_slots() -> int:
            try:
                resident_count = int(len(self._cache.resident_ids()))
                cache_capacity = int(getattr(self._cache, "capacity", 0))
                return max(0, cache_capacity - resident_count)
            except Exception:
                return 0

        def compute_segment(seg: Tuple[int, int, int, bool], payload_override=None) -> None:
            nonlocal cp_compute_cpu_ms
            eid, s, e, eid_is_local = seg
            cur = hidden_sorted[int(s):int(e)]
            cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
            y, cpu_ms = self._epipe_compute_segment(
                eid=int(eid),
                cur=cur,
                cur_w=cur_w,
                eid_is_local=bool(eid_is_local),
                allow_nonlocal=bool(allow_nonlocal),
                payload_override=payload_override,
                cp_cuda_timing=bool(cp_cuda_timing),
                cp_sampled=bool(cp_sampled),
                cp_compute_events=cp_compute_events,
            )
            out_sorted[int(s):int(e)] = y
            cp_compute_cpu_ms += float(cpu_ms)

        def compute_ready_indices(indices: Sequence[int]) -> int:
            computed = 0
            for idx in indices:
                seg = segments[int(idx)]
                eid = int(seg[0])
                cached = self._cache.get(eid)
                if cached is None:
                    compute_segment(seg)
                else:
                    self._epipe_record_cached_payload_use(eid, wait_prefetch=False)
                    compute_segment(seg, payload_override=cached)
                computed += 1
            if computed > 0:
                self._record_epipe_counter("epipe_rs_buffer_flush_count", 1)
            return computed

        try:
            if maybe_skip_gate():
                return None

            ready_indices: List[int] = []
            late_indices: List[int] = []
            demand_indices: List[int] = []
            for idx, seg in enumerate(segments):
                eid = int(seg[0])
                pending_ev = self._epipe_get_prefetch_pending_event(eid)
                cached_ready = self._cache_contains_no_touch(eid)
                if cached_ready and pending_ev is None:
                    ready_indices.append(int(idx))
                elif cached_ready and pending_ev is not None and bool(getattr(self, "_epipe_rs_handle_late_prefetch", False)):
                    late_indices.append(int(idx))
                elif (
                    (not cached_ready)
                    and bool(getattr(self, "_epipe_rs_handle_demand_miss", False))
                    and self._epipe_rs_demand_stage_gate_can_open()
                ):
                    demand_indices.append(int(idx))

            ready_items = int(sum(int(segments[i][2]) - int(segments[i][1]) for i in ready_indices))
            policy = self._light_policy
            if policy is not None:
                policy.observe_ready(len(ready_indices), ready_items)
            self._record_epipe_counter("epipe_rs_residual_scan_ready_group_count", len(ready_indices))
            self._record_epipe_counter("epipe_rs_residual_scan_ready_item_count", ready_items)
            self._record_epipe_counter("epipe_rs_buffered_ready_group_count", len(ready_indices))
            self._record_epipe_counter("epipe_rs_buffered_ready_item_count", ready_items)
            if len(ready_indices) < int(self._epipe_rs_min_buffer_groups) or ready_items < int(self._epipe_rs_min_buffer_items):
                self._record_epipe_counter("epipe_rs_residual_scan_no_ready_count", 1)
                self._record_epipe_counter("epipe_rs_buffer_too_small_count", 1)
                return None

            residual_idx: Optional[int] = None
            residual_kind = ""
            if late_indices:
                residual_idx = int(late_indices[0])
                residual_kind = "late"
            elif demand_indices and self._epipe_rs_demand_stage_gate_open(ready_items):
                residual_idx = int(demand_indices[0])
                residual_kind = "demand"

            if policy is not None and policy.feedback:
                demand = demand_indices if self._epipe_rs_demand_stage_gate_open(ready_items) else []
                candidates = [(i, int(segments[i][0]), "late") for i in late_indices]
                candidates.extend((i, int(segments[i][0]), "demand") for i in demand)
                selected = policy.residual(len(ready_indices), ready_items, candidates)
                residual_idx, residual_kind = selected if selected is not None else (None, "")

            if residual_idx is None:
                self._record_epipe_counter("epipe_decode_rs_skipped_no_miss_count", 1)
                self._record_epipe_counter("epipe_fallback_no_miss_trigger", 1)
                if control_sampled:
                    control_ms = max(0.0, float(_hclm_cp_now_ns() - int(control_start_ns)) / 1_000_000.0)
                    self._record_epipe_float("sampled_epipe_control_overhead_ms", control_ms)
                return None

            residual_seg = segments[int(residual_idx)]
            residual_eid = int(residual_seg[0])
            residual_item_count = int(residual_seg[2]) - int(residual_seg[1])
            ready_to_cover = [i for i in ready_indices if int(i) != int(residual_idx)]
            if len(ready_to_cover) < int(self._epipe_rs_min_buffer_groups):
                self._record_epipe_counter("epipe_rs_residual_scan_no_ready_count", 1)
                self._record_epipe_counter("epipe_rs_buffer_too_small_count", 1)
                return None

            self._record_epipe_counter("epipe_rs_trigger_count", 1)
            self._record_epipe_counter("epipe_rs_residual_scan_residual_count", 1)
            if residual_kind == "late":
                self._record_epipe_counter("epipe_decode_rs_miss_trigger_count", 1)
                self._record_epipe_counter("epipe_rs_late_prefetch_trigger_count", 1)
                self._record_epipe_counter("epipe_rs_residual_scan_late_count", 1)
            else:
                self._record_epipe_counter("epipe_decode_rs_miss_trigger_count", 1)
                self._record_epipe_counter("epipe_rs_demand_miss_trigger_count", 1)
                self._record_epipe_counter("epipe_rs_residual_scan_demand_count", 1)
            observe_trigger()

            if control_sampled:
                control_ms = max(0.0, float(_hclm_cp_now_ns() - int(control_start_ns)) / 1_000_000.0)
                self._record_epipe_float("sampled_epipe_control_overhead_ms", control_ms)

            done: set[int] = set()
            if residual_kind == "late":
                cached = self._cache.get(residual_eid)
                if cached is None:
                    return None
                compute_ready_indices(ready_to_cover)
                done.update(int(i) for i in ready_to_cover)
                self._maybe_wait_prefetch(residual_eid)
                self._epipe_record_cached_payload_use(residual_eid, wait_prefetch=False)
                self._record_epipe_counter("epipe_enabled_count", 1)
                self._epipe_record_stage_success_by_stage()
                self._record_epipe_counter("epipe_decode_rs_late_overlap_count", 1)
                self._record_epipe_counter("epipe_rs_late_prefetch_overlap_count", 1)
                compute_segment(residual_seg, payload_override=cached)
                done.add(int(residual_idx))
            else:
                staged_nbytes_est = int(self._epipe_estimate_expert_nbytes(residual_eid))
                if staged_nbytes_est <= 0:
                    fallback("epipe_fallback_exception")
                    return None
                if not self._epipe_memory_headroom_ok(staged_nbytes_est, min_free_mb=float(self._epipe_min_free_mb)):
                    fallback("epipe_fallback_memory")
                    self._record_epipe_counter("epipe_fallback_no_memory_headroom", 1)
                    self._record_epipe_counter("epipe_rs_demand_stage_disabled_by_gate_count", 1)
                    return None

                payload = None
                copied_nbytes = 0
                stage_event = None
                h2d_start_ev = None
                h2d_end_ev = None
                wait_start_ev = None
                wait_end_ev = None
                timing_sampled = self._epipe_timing_sampled("rs_residual_scan_demand", residual_eid)
                try:
                    with torch.cuda.stream(self.epipe_stream):
                        if timing_sampled:
                            h2d_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                            h2d_start_ev.record(self.epipe_stream)
                        payload, copied_nbytes = self._epipe_load_payload_to_device(residual_eid)
                        if timing_sampled:
                            h2d_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                            h2d_end_ev.record(self.epipe_stream)
                        stage_event = torch.cuda.Event(enable_timing=False, blocking=False)
                        stage_event.record(self.epipe_stream)
                except Exception:
                    fallback("epipe_fallback_exception")
                    return None

                compute_ready_indices(ready_to_cover)
                done.update(int(i) for i in ready_to_cover)

                cur_stream = torch.cuda.current_stream(self.device)
                demand_wait_start = self._record_demand_h2d_window_begin()
                if timing_sampled:
                    wait_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    wait_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    wait_start_ev.record(cur_stream)
                cur_stream.wait_event(stage_event)
                self._record_demand_h2d_window_end(demand_wait_start)
                if timing_sampled:
                    wait_end_ev.record(cur_stream)
                    if self.metrics is not None:
                        if self.metrics._pending_epipe_timing_events is None:
                            self.metrics._pending_epipe_timing_events = []
                        if h2d_start_ev is not None and h2d_end_ev is not None and wait_start_ev is not None and wait_end_ev is not None:
                            self.metrics._pending_epipe_timing_events.append(
                                (h2d_start_ev, h2d_end_ev, wait_start_ev, wait_end_ev, int(self.layer_idx))
                            )
                if isinstance(payload, (tuple, list)):
                    for t in payload:
                        if isinstance(t, torch.Tensor):
                            t.record_stream(cur_stream)
                elif isinstance(payload, torch.Tensor):
                    payload.record_stream(cur_stream)

                self._record_cache_miss()
                self._on_demand_access(residual_eid)
                self._epipe_record_staged_demand_h2d(int(copied_nbytes))

                cache_free_slots = current_free_slots()
                if cache_free_slots > 0:
                    self._record_epipe_counter("epipe_cache_free_slot_count", cache_free_slots)
                commit_allowed = bool(self._epipe_rs_demand_commit)
                commit_will_insert = bool(
                    commit_allowed
                    and self._epipe_deferred_lru_commit
                    and (cache_free_slots > 0 or not bool(getattr(self, "_epipe_no_evict", True)))
                )
                if commit_will_insert:
                    self._cache_put_guarded(
                        residual_eid,
                        payload,
                        pinned=False,
                        prefer_hot=self._should_prefer_hot(residual_eid, False),
                        prefill_scp_transient=False,
                    )
                    self._record_epipe_counter("epipe_cache_insert_count", 1)
                    self._record_epipe_counter("epipe_deferred_lru_commit_count", 1)
                    self._global_cache_enforce_after_demand()
                else:
                    if cache_free_slots <= 0:
                        self._record_epipe_counter("epipe_cache_full_staged_count", 1)
                    self._record_epipe_counter("epipe_transient_payload_count", 1)

                self._record_epipe_counter("epipe_enabled_count", 1)
                self._epipe_record_stage_success_by_stage()
                self._record_epipe_counter("epipe_staged_missing_count", 1)
                self._record_epipe_counter("epipe_staged_expert_count", 1)
                self._record_epipe_counter("epipe_rs_demand_stage_count", 1)
                self._record_epipe_counter("epipe_decode_rs_demand_stage_count", 1)
                self._record_epipe_counter("epipe_missing_expert_count", 1)
                self._record_epipe_counter("epipe_missing_item_count", residual_item_count)
                self._record_epipe_counter("epipe_missing_h2d_bytes_est", staged_nbytes_est)
                compute_segment(residual_seg, payload_override=payload)
                done.add(int(residual_idx))
                payload = None

            for idx, seg in enumerate(segments):
                if int(idx) in done:
                    continue
                compute_segment(seg)

            if self._epipe_rs_rolling_gate and int(self._epipe_rs_gate_skip_remaining) > 0:
                self._epipe_rs_gate_skip_remaining = 0
                self._record_epipe_counter("epipe_rs_gate_reenable_count", 1)
            return out_sorted, float(cp_compute_cpu_ms)
        except Exception:
            fallback("epipe_fallback_exception")
            raise

    def _try_execute_local_recv_groups_epipe_rs_buffered(
        self,
        *,
        hidden_sorted: torch.Tensor,
        recv_i_sorted: torch.Tensor,
        w_sorted: torch.Tensor,
        segments: List[Tuple[int, int, int, bool]],
        allow_nonlocal: bool,
        cp_cuda_timing: bool,
        cp_sampled: bool,
        cp_compute_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> Optional[Tuple[torch.Tensor, float]]:
        _ = recv_i_sorted
        if self._epipe_validate_output:
            raise NotImplementedError("SPARSE_EPIPE_VALIDATE_OUTPUT=1 is not implemented for the runtime hot path.")
        if self.expert_offload != "cpu" or self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        if len(segments) == 0:
            return None
        if self._current_stage() != "decode" or self.planner_mode != "rs_prefetch":
            return None
        if not self._epipe_enable or self.epipe_stream is None:
            return None

        control_sampled = self._epipe_timing_sampled("control_rs_buffered")
        control_start_ns = _hclm_cp_now_ns() if control_sampled else 0
        self._record_epipe_counter("epipe_rs_buffered_calls", 1)
        out_sorted = torch.zeros_like(hidden_sorted)
        cp_compute_cpu_ms = 0.0
        ready_buffer: List[Tuple[Tuple[int, int, int, bool], Any]] = []
        ready_buffer_items = 0
        trigger_seen = False

        def fallback(reason: str) -> None:
            self._record_epipe_counter("epipe_fallback_count", 1)
            if reason:
                self._record_epipe_counter(reason, 1)

        def buffer_sufficient() -> bool:
            return (
                len(ready_buffer) >= int(self._epipe_rs_min_buffer_groups)
                and ready_buffer_items >= int(self._epipe_rs_min_buffer_items)
            )

        def compute_segment(seg: Tuple[int, int, int, bool], payload_override=None) -> None:
            nonlocal cp_compute_cpu_ms
            eid, s, e, eid_is_local = seg
            cur = hidden_sorted[int(s):int(e)]
            cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
            y, cpu_ms = self._epipe_compute_segment(
                eid=int(eid),
                cur=cur,
                cur_w=cur_w,
                eid_is_local=bool(eid_is_local),
                allow_nonlocal=bool(allow_nonlocal),
                payload_override=payload_override,
                cp_cuda_timing=bool(cp_cuda_timing),
                cp_sampled=bool(cp_sampled),
                cp_compute_events=cp_compute_events,
            )
            out_sorted[int(s):int(e)] = y
            cp_compute_cpu_ms += float(cpu_ms)

        def flush_ready_buffer() -> None:
            nonlocal ready_buffer_items
            if not ready_buffer:
                return
            for buffered_seg, buffered_payload in ready_buffer:
                compute_segment(buffered_seg, payload_override=buffered_payload)
            self._record_epipe_counter("epipe_rs_buffer_flush_count", 1)
            ready_buffer.clear()
            ready_buffer_items = 0

        def append_ready(seg: Tuple[int, int, int, bool], payload) -> None:
            nonlocal ready_buffer_items
            ready_buffer.append((seg, payload))
            ready_buffer_items += int(seg[2]) - int(seg[1])
            self._record_epipe_counter("epipe_rs_buffered_ready_group_count", 1)
            self._record_epipe_counter("epipe_rs_buffered_ready_item_count", int(seg[2]) - int(seg[1]))
            if (
                len(ready_buffer) >= int(self._epipe_rs_buffer_max_groups)
                or ready_buffer_items >= int(self._epipe_rs_buffer_max_items)
            ):
                flush_ready_buffer()

        def observe_trigger() -> None:
            if not self._epipe_rs_rolling_gate:
                return
            self._epipe_rs_gate_triggers += 1

        def residual_present_for_gate() -> bool:
            for gate_seg in segments:
                gate_eid = int(gate_seg[0])
                ev = self._prefetch_events.get(gate_eid, None)
                if ev is not None:
                    try:
                        if bool(getattr(self, "_epipe_rs_handle_late_prefetch", False)) and not bool(ev.query()):
                            return True
                    except Exception:
                        if bool(getattr(self, "_epipe_rs_handle_late_prefetch", False)):
                            return True
                if (
                    bool(getattr(self, "_epipe_rs_handle_demand_miss", False))
                    and self._epipe_rs_demand_stage_gate_can_open()
                    and not self._cache_contains_no_touch(gate_eid)
                ):
                    return True
            return False

        def maybe_skip_gate() -> bool:
            if not self._epipe_rs_rolling_gate:
                return False
            if int(self._epipe_rs_gate_skip_remaining) > 0:
                if residual_present_for_gate():
                    self._epipe_rs_gate_skip_remaining = 0
                    self._record_epipe_counter("epipe_rs_gate_reenable_count", 1)
                    return False
                self._epipe_rs_gate_skip_remaining -= 1
                self._record_epipe_counter("epipe_rs_gate_skip_count", 1)
                return True
            self._epipe_rs_gate_calls += 1
            if (
                int(self._epipe_rs_gate_calls) >= int(self._epipe_rs_gate_window)
                and int(self._epipe_rs_gate_triggers) < int(self._epipe_rs_gate_min_triggers)
            ):
                self._epipe_rs_gate_calls = 0
                self._epipe_rs_gate_triggers = 0
                self._epipe_rs_gate_skip_remaining = int(self._epipe_rs_gate_skip_steps)
                if self._epipe_rs_gate_skip_remaining > 0:
                    self._record_epipe_counter("epipe_rs_gate_skip_count", 1)
                    self._epipe_rs_gate_skip_remaining -= 1
                    return True
            if int(self._epipe_rs_gate_calls) >= int(self._epipe_rs_gate_window):
                self._epipe_rs_gate_calls = 0
                self._epipe_rs_gate_triggers = 0
            return False

        def current_free_slots() -> int:
            try:
                resident_count = int(len(self._cache.resident_ids()))
                cache_capacity = int(getattr(self._cache, "capacity", 0))
                return max(0, cache_capacity - resident_count)
            except Exception:
                return 0

        try:
            if maybe_skip_gate():
                return None
            for seg in segments:
                eid = int(seg[0])
                pending_ev = self._epipe_get_prefetch_pending_event(eid)
                cached = self._cache.get(eid)
                if cached is not None:
                    if pending_ev is None:
                        self._epipe_record_cached_payload_use(eid, wait_prefetch=False)
                        append_ready(seg, cached)
                        continue
                    if not bool(self._epipe_rs_handle_late_prefetch):
                        flush_ready_buffer()
                        self._epipe_record_cached_payload_use(eid, wait_prefetch=True)
                        compute_segment(seg, payload_override=cached)
                        continue
                    self._record_epipe_counter("epipe_rs_trigger_count", 1)
                    self._record_epipe_counter("epipe_rs_late_prefetch_trigger_count", 1)
                    observe_trigger()
                    trigger_seen = True
                    if not buffer_sufficient():
                        self._record_epipe_counter("epipe_rs_buffer_too_small_count", 1)
                        flush_ready_buffer()
                        self._epipe_record_cached_payload_use(eid, wait_prefetch=True)
                        compute_segment(seg, payload_override=cached)
                        continue

                    if control_sampled:
                        control_ms = max(0.0, float(_hclm_cp_now_ns() - int(control_start_ns)) / 1_000_000.0)
                        self._record_epipe_float("sampled_epipe_control_overhead_ms", control_ms)
                        control_sampled = False
                    flush_ready_buffer()
                    self._maybe_wait_prefetch(eid)
                    self._epipe_record_cached_payload_use(eid, wait_prefetch=False)
                    self._record_epipe_counter("epipe_enabled_count", 1)
                    self._epipe_record_stage_success_by_stage()
                    self._record_epipe_counter("epipe_decode_rs_late_overlap_count", 1)
                    self._record_epipe_counter("epipe_rs_late_prefetch_overlap_count", 1)
                    compute_segment(seg, payload_override=cached)
                    continue

                if not bool(self._epipe_rs_handle_demand_miss):
                    flush_ready_buffer()
                    compute_segment(seg)
                    continue
                self._record_epipe_counter("epipe_rs_trigger_count", 1)
                self._record_epipe_counter("epipe_rs_demand_miss_trigger_count", 1)
                observe_trigger()
                trigger_seen = True
                if not buffer_sufficient():
                    self._record_epipe_counter("epipe_rs_buffer_too_small_count", 1)
                    self._record_epipe_counter("epipe_rs_demand_stage_disabled_by_gate_count", 1)
                    flush_ready_buffer()
                    compute_segment(seg)
                    continue
                if not self._epipe_rs_demand_stage_gate_open(ready_buffer_items):
                    flush_ready_buffer()
                    compute_segment(seg)
                    continue

                staged_nbytes_est = int(self._epipe_estimate_expert_nbytes(eid))
                if staged_nbytes_est <= 0:
                    fallback("epipe_fallback_exception")
                    flush_ready_buffer()
                    compute_segment(seg)
                    continue
                if not self._epipe_memory_headroom_ok(staged_nbytes_est, min_free_mb=float(self._epipe_min_free_mb)):
                    fallback("epipe_fallback_memory")
                    self._record_epipe_counter("epipe_fallback_no_memory_headroom", 1)
                    self._record_epipe_counter("epipe_rs_demand_stage_disabled_by_gate_count", 1)
                    flush_ready_buffer()
                    compute_segment(seg)
                    continue

                if control_sampled:
                    control_ms = max(0.0, float(_hclm_cp_now_ns() - int(control_start_ns)) / 1_000_000.0)
                    self._record_epipe_float("sampled_epipe_control_overhead_ms", control_ms)
                    control_sampled = False
                payload = None
                copied_nbytes = 0
                stage_event = None
                h2d_start_ev = None
                h2d_end_ev = None
                wait_start_ev = None
                wait_end_ev = None
                timing_sampled = self._epipe_timing_sampled("rs_demand", eid)
                try:
                    with torch.cuda.stream(self.epipe_stream):
                        if timing_sampled:
                            h2d_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                            h2d_start_ev.record(self.epipe_stream)
                        payload, copied_nbytes = self._epipe_load_payload_to_device(eid)
                        if timing_sampled:
                            h2d_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                            h2d_end_ev.record(self.epipe_stream)
                        stage_event = torch.cuda.Event(enable_timing=False, blocking=False)
                        stage_event.record(self.epipe_stream)
                except Exception:
                    fallback("epipe_fallback_exception")
                    flush_ready_buffer()
                    compute_segment(seg)
                    continue

                flush_ready_buffer()
                cur_stream = torch.cuda.current_stream(self.device)
                demand_wait_start = self._record_demand_h2d_window_begin()
                if timing_sampled:
                    wait_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    wait_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    wait_start_ev.record(cur_stream)
                cur_stream.wait_event(stage_event)
                self._record_demand_h2d_window_end(demand_wait_start)
                if timing_sampled:
                    wait_end_ev.record(cur_stream)
                    if self.metrics is not None:
                        if self.metrics._pending_epipe_timing_events is None:
                            self.metrics._pending_epipe_timing_events = []
                        if h2d_start_ev is not None and h2d_end_ev is not None and wait_start_ev is not None and wait_end_ev is not None:
                            self.metrics._pending_epipe_timing_events.append(
                                (h2d_start_ev, h2d_end_ev, wait_start_ev, wait_end_ev, int(self.layer_idx))
                            )
                if isinstance(payload, (tuple, list)):
                    for t in payload:
                        if isinstance(t, torch.Tensor):
                            t.record_stream(cur_stream)
                elif isinstance(payload, torch.Tensor):
                    payload.record_stream(cur_stream)

                self._record_cache_miss()
                self._on_demand_access(eid)
                self._epipe_record_staged_demand_h2d(int(copied_nbytes))

                cache_free_slots = current_free_slots()
                if cache_free_slots > 0:
                    self._record_epipe_counter("epipe_cache_free_slot_count", cache_free_slots)
                commit_allowed = bool(self._epipe_rs_demand_commit)
                commit_will_insert = bool(
                    commit_allowed
                    and self._epipe_deferred_lru_commit
                    and (cache_free_slots > 0 or not bool(getattr(self, "_epipe_no_evict", True)))
                )
                if commit_will_insert:
                    self._cache_put_guarded(
                        eid,
                        payload,
                        pinned=False,
                        prefer_hot=self._should_prefer_hot(eid, False),
                        prefill_scp_transient=False,
                    )
                    self._record_epipe_counter("epipe_cache_insert_count", 1)
                    self._record_epipe_counter("epipe_deferred_lru_commit_count", 1)
                    self._global_cache_enforce_after_demand()
                else:
                    if cache_free_slots <= 0:
                        self._record_epipe_counter("epipe_cache_full_staged_count", 1)
                    self._record_epipe_counter("epipe_transient_payload_count", 1)

                self._record_epipe_counter("epipe_enabled_count", 1)
                self._epipe_record_stage_success_by_stage()
                self._record_epipe_counter("epipe_staged_missing_count", 1)
                self._record_epipe_counter("epipe_staged_expert_count", 1)
                self._record_epipe_counter("epipe_rs_demand_stage_count", 1)
                self._record_epipe_counter("epipe_decode_rs_demand_stage_count", 1)
                self._record_epipe_counter("epipe_missing_expert_count", 1)
                self._record_epipe_counter("epipe_missing_item_count", int(seg[2]) - int(seg[1]))
                self._record_epipe_counter("epipe_missing_h2d_bytes_est", staged_nbytes_est)
                compute_segment(seg, payload_override=payload)
                payload = None

            flush_ready_buffer()
            if trigger_seen and self._epipe_rs_rolling_gate:
                if int(self._epipe_rs_gate_skip_remaining) > 0:
                    self._epipe_rs_gate_skip_remaining = 0
                    self._record_epipe_counter("epipe_rs_gate_reenable_count", 1)
            if not trigger_seen:
                self._record_epipe_counter("epipe_decode_rs_skipped_no_miss_count", 1)
                self._record_epipe_counter("epipe_fallback_no_miss_trigger", 1)
                if control_sampled:
                    control_ms = max(0.0, float(_hclm_cp_now_ns() - int(control_start_ns)) / 1_000_000.0)
                    self._record_epipe_float("sampled_epipe_control_overhead_ms", control_ms)
            return out_sorted, float(cp_compute_cpu_ms)
        except Exception:
            fallback("epipe_fallback_exception")
            raise

    def _try_execute_local_recv_groups_epipe_miss_triggered(
        self,
        *,
        hidden_sorted: torch.Tensor,
        recv_i_sorted: torch.Tensor,
        w_sorted: torch.Tensor,
        segments: List[Tuple[int, int, int, bool]],
        allow_nonlocal: bool,
        cp_cuda_timing: bool,
        cp_sampled: bool,
        cp_compute_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> Optional[Tuple[torch.Tensor, float]]:
        if self._epipe_validate_output:
            raise NotImplementedError("SPARSE_EPIPE_VALIDATE_OUTPUT=1 is not implemented for the runtime hot path.")
        if self.expert_offload != "cpu" or self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        if len(segments) == 0:
            return None

        stage = self._current_stage()
        rs_active = bool(self.planner_mode == "rs_prefetch")
        if stage == "prefill":
            if not (self._epipe_prefill_enable or self._epipe_prefill_shadow or self._epipe_prefill_multistage_shadow):
                return None
        elif not (self._epipe_enable or self._epipe_shadow):
            return None
        if self._epipe_decode_only and stage != "decode" and not (
            self._epipe_prefill_enable or self._epipe_prefill_shadow or self._epipe_prefill_multistage_shadow
        ):
            return None
        if stage == "decode" and rs_active and self._epipe_decode_with_rs_mode == "off":
            return None
        if stage == "decode" and rs_active and self._epipe_decode_with_rs_mode == "buffered":
            return self._try_execute_local_recv_groups_epipe_rs_buffered(
                hidden_sorted=hidden_sorted,
                recv_i_sorted=recv_i_sorted,
                w_sorted=w_sorted,
                segments=segments,
                allow_nonlocal=bool(allow_nonlocal),
                cp_cuda_timing=bool(cp_cuda_timing),
                cp_sampled=bool(cp_sampled),
                cp_compute_events=cp_compute_events,
            )
        if stage == "decode" and rs_active and self._epipe_decode_with_rs_mode == "upfront_scan":
            return self._try_execute_local_recv_groups_epipe(
                hidden_sorted=hidden_sorted,
                recv_i_sorted=recv_i_sorted,
                w_sorted=w_sorted,
                segments=segments,
                allow_nonlocal=bool(allow_nonlocal),
                cp_cuda_timing=bool(cp_cuda_timing),
                cp_sampled=bool(cp_sampled),
                cp_compute_events=cp_compute_events,
            )

        out_sorted = torch.zeros_like(hidden_sorted)
        cp_compute_cpu_ms = 0.0
        done: set[int] = set()
        staged_once = False
        miss_seen = False
        enabled_for_stage = bool(self._epipe_enable if stage == "decode" else self._epipe_prefill_enable)
        shadow_for_stage = bool(
            self._epipe_shadow
            if stage == "decode"
            else (self._epipe_prefill_shadow or self._epipe_prefill_multistage_shadow)
        )
        lookahead = int(self._epipe_lookahead_experts)
        min_ready_items = int(
            self._epipe_prefill_lookahead_min_ready_items
            if stage == "prefill"
            else self._epipe_lookahead_min_ready_items
        )
        min_free_mb = float(
            self._epipe_prefill_min_free_mb
            if stage == "prefill"
            else self._epipe_min_free_mb
        )

        def fallback(reason: str) -> None:
            self._record_epipe_counter("epipe_fallback_count", 1)
            if reason:
                self._record_epipe_counter(reason, 1)

        def compute_segment(seg: Tuple[int, int, int, bool], payload_override=None) -> None:
            nonlocal cp_compute_cpu_ms
            eid, s, e, eid_is_local = seg
            cur = hidden_sorted[int(s):int(e)]
            cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
            y, cpu_ms = self._epipe_compute_segment(
                eid=int(eid),
                cur=cur,
                cur_w=cur_w,
                eid_is_local=bool(eid_is_local),
                allow_nonlocal=bool(allow_nonlocal),
                payload_override=payload_override,
                cp_cuda_timing=bool(cp_cuda_timing),
                cp_sampled=bool(cp_sampled),
                cp_compute_events=cp_compute_events,
            )
            out_sorted[int(s):int(e)] = y
            cp_compute_cpu_ms += float(cpu_ms)

        try:
            for i, seg in enumerate(segments):
                if int(i) in done:
                    self._record_epipe_counter("epipe_done_group_skip_count", 1)
                    continue
                eid = int(seg[0])
                ready_payload = self._epipe_get_ready_payload_for_compute(eid)
                if ready_payload is not None:
                    compute_segment(seg, payload_override=ready_payload)
                    continue

                miss_seen = True
                if stage == "decode" and rs_active:
                    self._record_epipe_counter("epipe_decode_rs_miss_trigger_count", 1)
                elif stage == "prefill":
                    self._record_epipe_counter("epipe_prefill_miss_trigger_count", 1)

                if staged_once and self._epipe_stage_first_missing_only:
                    compute_segment(seg)
                    continue

                ready_indices: List[int] = []
                ready_items = 0
                if lookahead > 0:
                    end_j = min(len(segments), i + 1 + lookahead)
                    for j in range(i + 1, end_j):
                        if int(j) in done:
                            continue
                        future = segments[j]
                        future_eid = int(future[0])
                        future_ready = self._cache_contains_no_touch(future_eid)
                        if future_ready:
                            ready_indices.append(int(j))
                            ready_items += int(future[2]) - int(future[1])

                if shadow_for_stage and not enabled_for_stage:
                    if ready_indices:
                        if stage == "prefill":
                            self._record_epipe_counter("epipe_prefill_shadow_opportunity_count", 1)
                        else:
                            self._record_epipe_counter("epipe_shadow_count", 1)
                    self._record_epipe_counter("epipe_lookahead_ready_group_count", len(ready_indices))
                    self._record_epipe_counter("epipe_lookahead_ready_item_count", ready_items)
                    compute_segment(seg)
                    continue

                if not enabled_for_stage:
                    compute_segment(seg)
                    continue
                if not ready_indices:
                    fallback("epipe_fallback_no_ready_lookahead")
                    compute_segment(seg)
                    continue
                if ready_items < min_ready_items:
                    fallback("epipe_fallback_low_ready_items")
                    compute_segment(seg)
                    continue

                staged_nbytes_est = int(self._epipe_estimate_expert_nbytes(eid))
                if staged_nbytes_est <= 0:
                    fallback("epipe_fallback_exception")
                    compute_segment(seg)
                    continue
                if not self._epipe_memory_headroom_ok(staged_nbytes_est, min_free_mb=min_free_mb):
                    fallback("epipe_fallback_memory")
                    self._record_epipe_counter("epipe_fallback_no_memory_headroom", 1)
                    compute_segment(seg)
                    continue
                if self.epipe_stream is None:
                    fallback("epipe_fallback_exception")
                    compute_segment(seg)
                    continue

                payload = None
                copied_nbytes = 0
                stage_event = None
                h2d_start_ev = None
                h2d_end_ev = None
                wait_start_ev = None
                wait_end_ev = None
                timing_sampled = self._epipe_timing_sampled("miss_triggered", eid)
                try:
                    with torch.cuda.stream(self.epipe_stream):
                        if timing_sampled:
                            h2d_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                            h2d_start_ev.record(self.epipe_stream)
                        payload, copied_nbytes = self._epipe_load_payload_to_device(eid)
                        if timing_sampled:
                            h2d_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                            h2d_end_ev.record(self.epipe_stream)
                        stage_event = torch.cuda.Event(enable_timing=False, blocking=False)
                        stage_event.record(self.epipe_stream)
                except Exception:
                    fallback("epipe_fallback_exception")
                    payload = None
                    compute_segment(seg)
                    continue

                for j in ready_indices:
                    compute_segment(segments[j])
                    done.add(int(j))

                cur_stream = torch.cuda.current_stream(self.device)
                demand_wait_start = self._record_demand_h2d_window_begin()
                if timing_sampled:
                    wait_start_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    wait_end_ev = torch.cuda.Event(enable_timing=True, blocking=False)
                    wait_start_ev.record(cur_stream)
                cur_stream.wait_event(stage_event)
                self._record_demand_h2d_window_end(demand_wait_start)
                if timing_sampled:
                    wait_end_ev.record(cur_stream)
                    if self.metrics is not None:
                        if self.metrics._pending_epipe_timing_events is None:
                            self.metrics._pending_epipe_timing_events = []
                        if h2d_start_ev is not None and h2d_end_ev is not None and wait_start_ev is not None and wait_end_ev is not None:
                            self.metrics._pending_epipe_timing_events.append(
                                (h2d_start_ev, h2d_end_ev, wait_start_ev, wait_end_ev, int(self.layer_idx))
                            )
                if isinstance(payload, (tuple, list)):
                    for t in payload:
                        if isinstance(t, torch.Tensor):
                            t.record_stream(cur_stream)
                elif isinstance(payload, torch.Tensor):
                    payload.record_stream(cur_stream)

                self._record_cache_miss()
                if stage != "prefill":
                    self._on_demand_access(eid)
                self._epipe_record_staged_demand_h2d(int(copied_nbytes))

                try:
                    resident_count = int(len(self._cache.resident_ids()))
                    cache_capacity = int(getattr(self._cache, "capacity", 0))
                    cache_free_slots = max(0, cache_capacity - resident_count)
                except Exception:
                    cache_free_slots = 0

                commit_allowed = False
                mark_transient = False
                if stage == "prefill":
                    commit_allowed = bool(self._epipe_prefill_commit)
                    mark_transient = bool(self._epipe_prefill_mark_transient)
                elif rs_active:
                    commit_allowed = bool(self._epipe_decode_rs_commit) or cache_free_slots > 0
                else:
                    commit_allowed = bool(self._epipe_decode_no_rs_commit)

                if cache_free_slots > 0:
                    self._record_epipe_counter("epipe_cache_free_slot_count", cache_free_slots)

                commit_will_insert = bool(
                    commit_allowed
                    and self._epipe_deferred_lru_commit
                    and (cache_free_slots > 0 or not bool(getattr(self, "_epipe_no_evict", True)))
                )
                if stage == "prefill" and commit_will_insert:
                    self._on_demand_access(eid)

                if commit_will_insert:
                    self._cache_put_guarded(
                        eid,
                        payload,
                        pinned=False,
                        prefer_hot=self._should_prefer_hot(eid, False),
                        prefill_scp_transient=bool(mark_transient),
                    )
                    self._record_epipe_counter("epipe_cache_insert_count", 1)
                    if stage == "prefill":
                        self._record_epipe_counter("epipe_prefill_cache_insert_count", 1)
                    self._record_epipe_counter("epipe_deferred_lru_commit_count", 1)
                    if mark_transient:
                        self._record_epipe_counter("epipe_prefill_transient_commit_count", 1)
                    self._global_cache_enforce_after_demand()
                else:
                    stored_extra = bool(
                        stage == "prefill"
                        and self._epipe_prefill_extra_cache_store(eid, payload, int(copied_nbytes))
                    )
                    if not stored_extra:
                        if cache_free_slots <= 0:
                            self._record_epipe_counter("epipe_cache_full_staged_count", 1)
                        self._record_epipe_counter("epipe_transient_payload_count", 1)
                        self._epipe_record_prefill_no_commit_payload()

                self._record_epipe_counter("epipe_enabled_count", 1)
                self._epipe_record_stage_success_by_stage()
                self._record_epipe_counter("epipe_staged_missing_count", 1)
                self._record_epipe_counter("epipe_staged_expert_count", 1)
                self._record_epipe_counter("epipe_ready_expert_count", len(ready_indices))
                self._record_epipe_counter("epipe_ready_item_count", ready_items)
                self._record_epipe_counter("epipe_lookahead_ready_group_count", len(ready_indices))
                self._record_epipe_counter("epipe_lookahead_ready_item_count", ready_items)
                self._record_epipe_counter("epipe_missing_expert_count", 1)
                self._record_epipe_counter("epipe_missing_item_count", int(seg[2]) - int(seg[1]))
                self._record_epipe_counter("epipe_missing_h2d_bytes_est", staged_nbytes_est)

                compute_segment(seg, payload_override=payload)
                payload = None
                staged_once = True

            if stage == "decode" and rs_active and not miss_seen:
                self._record_epipe_counter("epipe_decode_rs_skipped_no_miss_count", 1)
                self._record_epipe_counter("epipe_fallback_no_miss_trigger", 1)
            return out_sorted, float(cp_compute_cpu_ms)
        except Exception:
            fallback("epipe_fallback_exception")
            raise

    def _execute_local_recv_groups(
        self,
        recv_hidden: torch.Tensor,
        recv_i: torch.Tensor,
        recv_w: torch.Tensor,
        *,
        allow_nonlocal_cpu_once: bool = False,
        allow_epipe: bool = True,
        force_plain_ep: bool = False,
    ) -> torch.Tensor:
        """
        Execute local experts on received routed tokens.
        Optimization:
        - sort once by expert id
        - process contiguous expert-id segments
        - avoid per-expert full-length boolean mask scans
        """
        if recv_i.numel() == 0:
            return torch.zeros_like(recv_hidden)

        plain_ep_mode = _plain_ep_exec_mode()
        legacy_plain_ep = plain_ep_mode in ("unique_mask", "full_mask")
        cp = _hclm_cp_profiler()
        cp_step_id = int(getattr(SparseContext, "runtime_call_id", -1))
        cp_sampled = bool(cp.should_sample(cp_step_id))
        cp_cuda_timing = bool(cp_sampled and torch.cuda.is_available() and self.device.type == "cuda")
        cp_compute_events = []
        cp_compute_cpu_ms = 0.0

        if legacy_plain_ep and not bool(allow_epipe) and not bool(force_plain_ep):
            force_plain_ep = True
        if legacy_plain_ep and not (
            bool(getattr(self, "_epipe_enable", False))
            or bool(getattr(self, "_epipe_shadow", False))
            or bool(getattr(self, "_epipe_prefill_enable", False))
            or bool(getattr(self, "_epipe_prefill_shadow", False))
            or bool(getattr(self, "_epipe_prefill_multistage_shadow", False))
        ):
            force_plain_ep = True

        if legacy_plain_ep and bool(force_plain_ep):
            self._record_expert_executor_call("plain_ep_executor_calls")
            out = torch.zeros_like(recv_hidden)
            allow_nonlocal = bool(allow_nonlocal_cpu_once) and self.expert_offload == "cpu"
            if plain_ep_mode == "full_mask":
                eid_iterable = range(int(self.num_experts))
                impl_name = "plain_ep_legacy_full_mask"
            else:
                eid_iterable = sorted(int(x) for x in torch.unique(recv_i.detach()).tolist())
                impl_name = "plain_ep_legacy_unique_mask"
            for eid in eid_iterable:
                mask = recv_i == int(eid)
                if not bool(mask.any().item()):
                    continue
                eid_is_local = int(eid) in self._local_eid_set
                if not eid_is_local and not allow_nonlocal:
                    continue
                cur = recv_hidden[mask]
                cur_w = recv_w[mask].unsqueeze(1)
                y, cpu_ms = self._plain_ep_compute_segment(
                    eid=int(eid),
                    cur=cur,
                    cur_w=cur_w,
                    eid_is_local=bool(eid_is_local),
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
                out[mask] = y
                cp_compute_cpu_ms += float(cpu_ms)
            if cp_sampled:
                if cp_cuda_timing and cp_compute_events:
                    cp_compute_events[-1][1].synchronize()
                    cp_compute_ms = sum(float(s.elapsed_time(e)) for s, e in cp_compute_events)
                else:
                    cp_compute_ms = float(cp_compute_cpu_ms)
                cp.record_layer_summary(
                    cp_step_id,
                    int(self.layer_idx),
                    {"expert_compute_ms": float(cp_compute_ms), "expert_exec_impl": impl_name},
                )
            return out

        order = torch.argsort(recv_i)
        recv_i_sorted = recv_i[order]
        hidden_sorted = recv_hidden[order]
        w_sorted = recv_w[order]

        n = int(recv_i_sorted.numel())
        is_start = torch.ones((n,), device=recv_i_sorted.device, dtype=torch.bool)
        if n > 1:
            is_start[1:] = recv_i_sorted[1:] != recv_i_sorted[:-1]
        seg_starts = torch.nonzero(is_start, as_tuple=False).flatten()
        seg_ends = torch.empty_like(seg_starts)
        if seg_starts.numel() > 1:
            seg_ends[:-1] = seg_starts[1:]
        seg_ends[-1] = n

        segments: List[Tuple[int, int, int, bool]] = []
        allow_nonlocal = bool(allow_nonlocal_cpu_once) and self.expert_offload == "cpu"
        for s, e in zip(seg_starts.tolist(), seg_ends.tolist()):
            eid = int(recv_i_sorted[s].item())
            eid_is_local = eid in self._local_eid_set
            if not eid_is_local and not allow_nonlocal:
                continue
            segments.append((int(eid), int(s), int(e), bool(eid_is_local)))

        stage = self._current_stage()
        rs_active = bool(self.planner_mode == "rs_prefetch")
        rs_epipe_mode = str(getattr(self, "_epipe_decode_with_rs_mode", "late_only")).strip().lower()
        use_epipe_path = (
            bool(allow_epipe)
            and not bool(force_plain_ep)
            and bool(self._epipe_stage_enabled())
        )
        epipe_mode = str(getattr(self, "_epipe_mode", "upfront_scan")).strip().lower()
        if epipe_mode == "off":
            use_epipe_path = False
        epipe_result = None
        if use_epipe_path:
            self._record_expert_executor_call("epipe_executor_attempt_count")
            if stage == "decode" and rs_active and rs_epipe_mode == "off":
                epipe_result = None
            elif stage == "decode" and rs_active and rs_epipe_mode == "residual_scan":
                epipe_result = self._try_execute_local_recv_groups_epipe_rs_residual_scan(
                    hidden_sorted=hidden_sorted,
                    recv_i_sorted=recv_i_sorted,
                    w_sorted=w_sorted,
                    segments=segments,
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
            elif stage == "decode" and rs_active and rs_epipe_mode in ("buffered", "late_only", "demand_only"):
                epipe_result = self._try_execute_local_recv_groups_epipe_rs_buffered(
                    hidden_sorted=hidden_sorted,
                    recv_i_sorted=recv_i_sorted,
                    w_sorted=w_sorted,
                    segments=segments,
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
            elif stage == "decode" and not rs_active and epipe_mode != "miss_triggered":
                epipe_result = self._try_execute_local_recv_groups_epipe(
                    hidden_sorted=hidden_sorted,
                    recv_i_sorted=recv_i_sorted,
                    w_sorted=w_sorted,
                    segments=segments,
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
            elif epipe_mode == "upfront_scan":
                epipe_result = self._try_execute_local_recv_groups_epipe(
                    hidden_sorted=hidden_sorted,
                    recv_i_sorted=recv_i_sorted,
                    w_sorted=w_sorted,
                    segments=segments,
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
            else:
                epipe_result = self._try_execute_local_recv_groups_epipe_miss_triggered(
                    hidden_sorted=hidden_sorted,
                    recv_i_sorted=recv_i_sorted,
                    w_sorted=w_sorted,
                    segments=segments,
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
        if epipe_result is not None:
            self._record_expert_executor_call("epipe_executor_calls")
            self._epipe_record_executor_call_by_stage()
            self._epipe_record_stage_enabled()
            out_sorted, cp_compute_cpu_ms = epipe_result
        else:
            self._record_expert_executor_call("plain_ep_executor_calls")
            out_sorted = torch.zeros_like(hidden_sorted)
            for eid, s, e, eid_is_local in segments:
                cur = hidden_sorted[int(s):int(e)]
                cur_w = w_sorted[int(s):int(e)].unsqueeze(1)
                y, cpu_ms = self._plain_ep_compute_segment(
                    eid=int(eid),
                    cur=cur,
                    cur_w=cur_w,
                    eid_is_local=bool(eid_is_local),
                    allow_nonlocal=bool(allow_nonlocal),
                    cp_cuda_timing=bool(cp_cuda_timing),
                    cp_sampled=bool(cp_sampled),
                    cp_compute_events=cp_compute_events,
                )
                out_sorted[int(s):int(e)] = y
                cp_compute_cpu_ms += float(cpu_ms)

        out = torch.zeros_like(recv_hidden)
        out[order] = out_sorted
        if cp_sampled:
            if cp_cuda_timing and cp_compute_events:
                cp_compute_events[-1][1].synchronize()
                cp_compute_ms = sum(float(s.elapsed_time(e)) for s, e in cp_compute_events)
            else:
                cp_compute_ms = float(cp_compute_cpu_ms)
            cp.record_layer_summary(
                cp_step_id,
                int(self.layer_idx),
                {"expert_compute_ms": float(cp_compute_ms)},
            )
        return out

    def _execute_local_rank_grouped_packets(
        self,
        recv_hidden: torch.Tensor,
        recv_i: torch.Tensor,
        recv_w: torch.Tensor,
        *,
        allow_nonlocal_cpu_once: bool = False,
    ) -> torch.Tensor:
        """
        Execute one packet per (source token, destination rank).

        Each packet can contain several local experts for the same source token.
        The receiver expands valid packet/expert pairs, reuses the normal local
        expert executor, then sums the weighted outputs back to one partial
        result per packet.
        """
        if recv_hidden.numel() == 0:
            return torch.zeros_like(recv_hidden)
        if recv_i.numel() == 0:
            return torch.zeros_like(recv_hidden)

        P, H = recv_hidden.shape
        K = recv_i.shape[1]
        valid = recv_i.ge(0)
        if not bool(valid.any().item()):
            return torch.zeros((P, H), device=recv_hidden.device, dtype=recv_hidden.dtype)

        packet_grid = torch.arange(P, device=recv_hidden.device, dtype=torch.long).view(P, 1).expand(P, K)
        packet_ids = packet_grid[valid].contiguous()
        flat_i = recv_i[valid].to(torch.long).contiguous()
        flat_w = recv_w[valid].to(recv_hidden.dtype).contiguous()
        flat_hidden = recv_hidden.index_select(0, packet_ids).contiguous()

        pair_out = self._execute_local_recv_groups(
            recv_hidden=flat_hidden,
            recv_i=flat_i,
            recv_w=flat_w,
            allow_nonlocal_cpu_once=bool(allow_nonlocal_cpu_once),
            allow_epipe=False,
        )
        out = torch.zeros((P, H), device=recv_hidden.device, dtype=recv_hidden.dtype)
        out.index_add_(0, packet_ids, pair_out)
        return out

    def _rank_grouped_dispatch_enabled(self, seq_len: int) -> bool:
        if not _env_flag("SPARSE_RANK_GROUPED_DISPATCH", "0"):
            return False
        if self.topology is None or self.comm_group is None:
            return False
        if int(self.comm_world) <= 1:
            return False
        if int(seq_len) != 1 and not _env_flag("SPARSE_RANK_GROUPED_PREFILL", "0"):
            return False
        return True

    @staticmethod
    def _parse_layer_set(raw: str) -> Optional[set]:
        raw = str(raw or "").strip()
        if raw == "":
            return None
        out = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    lo = int(a.strip())
                    hi = int(b.strip())
                except Exception:
                    continue
                if hi < lo:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                try:
                    out.add(int(part))
                except Exception:
                    continue
        return out

    def _estimate_node_grouped_saving_ms(self, x: torch.Tensor, idx: torch.Tensor) -> float:
        if idx.numel() == 0:
            return 0.0
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        flat_i = idx.reshape(-1).to(torch.long)
        dest_rank = self._resolve_dest_for_flat_experts(flat_i, reason="ngd_estimate", count_metrics=False)
        normal_counts = torch.bincount(dest_rank, minlength=self.comm_world).to(dtype=torch.long)

        node_ids = self._comm_rank_node_ids()
        dest_rank_mat = dest_rank.view(int(T), int(K))
        dest_node_mat = torch.empty_like(dest_rank_mat)
        for cr, q in enumerate(node_ids):
            dest_node_mat[dest_rank_mat == int(cr)] = int(q)

        dest_node_cpu = dest_node_mat.detach().to("cpu", dtype=torch.long).tolist()
        node_packet_dest_vals: List[int] = []
        for t in range(int(T)):
            seen = set()
            for j in range(int(K)):
                q = int(dest_node_cpu[t][j])
                if q in seen:
                    continue
                seen.add(q)
                node_packet_dest_vals.append(self._node_representative_comm_rank(q))
        if node_packet_dest_vals:
            node_counts = torch.bincount(
                torch.tensor(node_packet_dest_vals, device=self.device, dtype=torch.long),
                minlength=self.comm_world,
            ).to(dtype=torch.long)
        else:
            node_counts = torch.zeros((self.comm_world,), device=self.device, dtype=torch.long)

        dtype_bytes = int(x.element_size())
        id_bytes = int(torch.empty((), dtype=torch.long, device=self.device).element_size())
        normal_payload_bytes = int((H + 1) * dtype_bytes)
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            normal_payload_bytes += id_bytes
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            normal_payload_bytes += int(H * dtype_bytes)

        node_payload_bytes = int(H * dtype_bytes)
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            node_payload_bytes += int(K * id_bytes) + int(K * x.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            node_payload_bytes += int(H * dtype_bytes)

        normal = _logical_weaknet_payload(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=normal_counts,
            payload_bytes_per_item=normal_payload_bytes,
            metrics=None,
            layer_idx=self.layer_idx,
        )
        node_grouped = _logical_weaknet_payload(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=node_counts,
            payload_bytes_per_item=node_payload_bytes,
            metrics=None,
            layer_idx=self.layer_idx,
        )
        return max(float(normal.get("sim_ms", 0.0)) - float(node_grouped.get("sim_ms", 0.0)), 0.0)

    def _estimate_node_grouped_saving_ms_tensorized(self, x: torch.Tensor, idx: torch.Tensor) -> float:
        if idx.numel() == 0:
            return 0.0
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        if T <= 0 or K <= 0:
            return 0.0

        flat_i = idx.reshape(-1).to(torch.long)
        dest_rank = self._resolve_dest_for_flat_experts(flat_i, reason="ngd_estimate_tensor", count_metrics=False)
        normal_counts = torch.bincount(dest_rank, minlength=self.comm_world).to(dtype=torch.long)

        dest_rank_mat = dest_rank.view(T, K).to(torch.long)
        rank_to_node = self._rank_to_node_tensor()
        dest_node_mat = rank_to_node[dest_rank_mat]
        max_node = int(rank_to_node.max().item()) if rank_to_node.numel() > 0 else 0
        present = torch.zeros((T, max_node + 1), device=self.device, dtype=torch.bool)
        token_grid = torch.arange(T, device=self.device, dtype=torch.long).view(T, 1).expand(T, K)
        present[token_grid.reshape(-1), dest_node_mat.reshape(-1)] = True
        _token_ids, node_vals = torch.nonzero(present, as_tuple=True)
        if int(node_vals.numel()) > 0:
            reps = self._node_representative_tensor()
            node_counts = torch.bincount(reps[node_vals].to(torch.long), minlength=self.comm_world).to(dtype=torch.long)
        else:
            node_counts = torch.zeros((self.comm_world,), device=self.device, dtype=torch.long)

        dtype_bytes = int(x.element_size())
        id_bytes = int(torch.empty((), dtype=torch.long, device=self.device).element_size())
        normal_payload_bytes = int((H + 1) * dtype_bytes)
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            normal_payload_bytes += id_bytes
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            normal_payload_bytes += int(H * dtype_bytes)

        node_payload_bytes = int(H * dtype_bytes)
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            node_payload_bytes += int(K * id_bytes) + int(K * x.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            node_payload_bytes += int(H * dtype_bytes)

        normal = _logical_weaknet_payload(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=normal_counts,
            payload_bytes_per_item=normal_payload_bytes,
            metrics=None,
            layer_idx=self.layer_idx,
        )
        node_grouped = _logical_weaknet_payload(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=node_counts,
            payload_bytes_per_item=node_payload_bytes,
            metrics=None,
            layer_idx=self.layer_idx,
        )
        return max(float(normal.get("sim_ms", 0.0)) - float(node_grouped.get("sim_ms", 0.0)), 0.0)

    def _node_grouped_dispatch_enabled(self, seq_len: int, idx: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None) -> bool:
        if not _env_flag("SPARSE_NODE_GROUPED_DISPATCH", "0"):
            return False
        if self.topology is None or self.comm_group is None:
            return False
        if int(self.comm_world) <= 1:
            return False
        node_map = _parse_logical_node_map(self.world)
        nodes = {
            int(node_map[int(g)])
            for g in self.comm_global_ranks
            if 0 <= int(g) < len(node_map)
        }
        if len(nodes) <= 1:
            return False
        if int(seq_len) != 1 and not _env_flag("SPARSE_NODE_GROUPED_PREFILL", "0"):
            return False
        force_layers = self._parse_layer_set(os.environ.get("SPARSE_NODE_GROUPED_FORCE_LAYERS", ""))
        disable_layers = self._parse_layer_set(os.environ.get("SPARSE_NODE_GROUPED_DISABLE_LAYERS", ""))
        if disable_layers is not None and int(self.layer_idx) in disable_layers:
            self._record_node_grouped_decision(enabled=False, estimated_saving_ms=0.0)
            return False
        if force_layers is not None and int(self.layer_idx) in force_layers:
            self._record_node_grouped_decision(enabled=True, estimated_saving_ms=0.0)
            return True
        policy_decision = _node_grouped_layer_policy_decision(int(self.layer_idx))
        if policy_decision is not None:
            self._record_node_grouped_decision(enabled=bool(policy_decision), estimated_saving_ms=0.0)
            return bool(policy_decision)
        if not _env_flag("SPARSE_NODE_GROUPED_LAYER_ADAPTIVE", "0"):
            self._record_node_grouped_decision(enabled=True, estimated_saving_ms=0.0)
            return True
        if idx is None or x is None:
            self._record_node_grouped_decision(enabled=False, estimated_saving_ms=0.0)
            return False
        fast_est = _env_flag("SPARSE_NODE_GROUPED_ADAPTIVE_FAST_EST", "0") or self._node_grouped_fastpath_mode(seq_len) in (
            "lean",
            "lean_tensor",
            "lean_tensor_stage2local",
        )
        if fast_est:
            estimated_saving_ms = self._estimate_node_grouped_saving_ms_tensorized(x, idx)
        else:
            estimated_saving_ms = self._estimate_node_grouped_saving_ms(x, idx)
        min_saving = float(os.environ.get("SPARSE_NODE_GROUPED_MIN_SAVING_MS", "0") or 0.0)
        margin = float(os.environ.get("SPARSE_NODE_GROUPED_OVERHEAD_MARGIN", "1.0") or 1.0)
        default_overhead = float(os.environ.get("SPARSE_NODE_GROUPED_EST_OVERHEAD_MS", str(max(min_saving, 0.0))) or 0.0)
        observed_overhead = 0.0
        lp = self._ensure_layer_perf()
        if lp is not None and int(getattr(lp, "node_grouped_enabled_count", 0)) > 0:
            observed_overhead = (
                float(getattr(lp, "node_grouped_pack_ms", 0.0))
                + float(getattr(lp, "node_grouped_metadata_ms", 0.0))
                + float(getattr(lp, "node_grouped_intra_fanout_ms", 0.0))
                + float(getattr(lp, "node_grouped_gather_ms", 0.0))
            ) / max(int(lp.node_grouped_enabled_count), 1)
        overhead = observed_overhead if observed_overhead > 0.0 else default_overhead
        global_estimated_saving_ms = estimated_saving_ms
        global_overhead = overhead
        if dist.is_available() and dist.is_initialized() and self.comm_group is not None:
            decision_stats = torch.tensor(
                [float(estimated_saving_ms), float(overhead)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(decision_stats, op=dist.ReduceOp.MAX, group=self.comm_group)
            global_estimated_saving_ms = float(decision_stats[0].item())
            global_overhead = float(decision_stats[1].item())
        threshold = max(min_saving, global_overhead * margin)
        enabled = global_estimated_saving_ms > threshold
        self._record_node_grouped_decision(enabled=enabled, estimated_saving_ms=estimated_saving_ms)
        return enabled

    def _comm_rank_node_ids(self) -> List[int]:
        raw_node_map = _sim_env_raw("SPARSE_SIM_NODE_MAP", "SPARSE_LOGICAL_NODE_MAP", "").strip()
        cache_key = (int(self.world), tuple(int(x) for x in self.comm_global_ranks), raw_node_map)
        if self._comm_rank_node_ids_cache_key == cache_key and self._comm_rank_node_ids_cache is not None:
            return self._comm_rank_node_ids_cache
        node_map = _parse_logical_node_map(self.world)
        out: List[int] = []
        for g in self.comm_global_ranks:
            gi = int(g)
            out.append(int(node_map[gi]) if 0 <= gi < len(node_map) else 0)
        self._comm_rank_node_ids_cache_key = cache_key
        self._comm_rank_node_ids_cache = out
        return out

    def _node_representative_comm_rank(self, node_id: int) -> int:
        node_ids = self._comm_rank_node_ids()
        my_node = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
        if int(node_id) == my_node:
            return int(self.comm_rank)
        for r, q in enumerate(node_ids):
            if int(q) == int(node_id):
                return int(r)
        return int(self.comm_rank)

    def _stage2_local_group_info(self) -> Optional[Tuple[Any, Tuple[int, ...], int, Tuple[int, ...]]]:
        """Return process-group metadata for this rank's logical node.

        Physical NGD's second stage is logically intra-node.  Using the full EP
        group for that stage forces remote ranks to participate in fanout/gather
        collectives with zero counts.  This guarded helper creates one process
        group per logical node and returns the current rank's local group.
        """
        if not _env_flag("SPARSE_NGD_STAGE2_NODE_LOCAL_GROUP", "0"):
            return None
        if self.topology is None or self.comm_group is None:
            return None
        if not (dist.is_available() and dist.is_initialized()):
            return None
        node_ids = tuple(int(x) for x in self._comm_rank_node_ids())
        comm_ranks = tuple(int(x) for x in self.comm_global_ranks)
        if len(node_ids) != len(comm_ranks) or int(self.comm_rank) >= len(node_ids):
            return None
        my_node = int(node_ids[int(self.comm_rank)])
        local_comm = tuple(i for i, q in enumerate(node_ids) if int(q) == my_node)
        if len(local_comm) <= 0:
            return None
        if len(local_comm) == int(self.comm_world):
            return self.comm_group, tuple(range(int(self.comm_world))), int(self.comm_rank), comm_ranks

        key = (int(self.world), comm_ranks, node_ids)
        cache = SparseContext.ngd_stage2_node_group_cache.get(key)
        if cache is None:
            cache = {}
            for q in sorted(set(node_ids)):
                members_comm = tuple(i for i, node in enumerate(node_ids) if int(node) == int(q))
                members_global = tuple(int(comm_ranks[i]) for i in members_comm)
                group = dist.new_group(ranks=list(members_global))
                cache[int(q)] = (group, members_comm, members_global)
            SparseContext.ngd_stage2_node_group_cache[key] = cache
        info = cache.get(my_node)
        if info is None:
            return None
        group, members_comm, members_global = info
        local_rank = int(members_comm.index(int(self.comm_rank)))
        return group, tuple(int(x) for x in members_comm), local_rank, tuple(int(x) for x in members_global)

    def _compress_counts_for_comm_members(
        self,
        send_dest: torch.Tensor,
        members_comm: Sequence[int],
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        member_to_local = {int(comm_rank): int(i) for i, comm_rank in enumerate(members_comm)}
        if int(send_dest.numel()) == 0 or len(member_to_local) == 0:
            counts = torch.zeros((len(member_to_local),), device=self.device, dtype=torch.long)
        else:
            comm_to_local = torch.full(
                (int(self.comm_world),),
                -1,
                device=self.device,
                dtype=torch.long,
            )
            member_vals = torch.tensor(
                [int(x) for x in members_comm],
                device=self.device,
                dtype=torch.long,
            )
            local_vals = torch.arange(len(member_to_local), device=self.device, dtype=torch.long)
            valid_members = member_vals.ge(0) & member_vals.lt(int(self.comm_world))
            comm_to_local[member_vals[valid_members]] = local_vals[valid_members]
            safe_dest = send_dest.to(device=self.device, dtype=torch.long).clamp(0, int(self.comm_world) - 1)
            local_dest = comm_to_local[safe_dest]
            local_dest = local_dest[local_dest.ge(0)]
            if int(local_dest.numel()) > 0:
                counts = torch.bincount(local_dest, minlength=len(member_to_local)).to(dtype=torch.long)
            else:
                counts = torch.zeros((len(member_to_local),), device=self.device, dtype=torch.long)
        return counts, member_to_local

    @staticmethod
    def _expand_counts_from_comm_members(
        local_counts: Sequence[int],
        members_comm: Sequence[int],
        comm_world: int,
    ) -> List[int]:
        out = [0 for _ in range(int(comm_world))]
        for local_idx, comm_rank in enumerate(members_comm):
            if 0 <= int(local_idx) < len(local_counts) and 0 <= int(comm_rank) < int(comm_world):
                out[int(comm_rank)] = int(local_counts[int(local_idx)])
        return out

    def _rank_to_node_tensor(self) -> torch.Tensor:
        node_ids = self._comm_rank_node_ids()
        cache_key = (tuple(int(x) for x in node_ids), str(self.device))
        cached = self._rank_to_node_tensor_cache
        if self._rank_to_node_tensor_cache_key == cache_key and cached is not None and cached.device == self.device:
            return cached
        out = torch.tensor(node_ids, device=self.device, dtype=torch.long)
        self._rank_to_node_tensor_cache_key = cache_key
        self._rank_to_node_tensor_cache = out
        return out

    def _node_representative_tensor(self) -> torch.Tensor:
        node_ids = self._comm_rank_node_ids()
        cache_key = (tuple(int(x) for x in node_ids), int(self.comm_rank), str(self.device))
        cached = self._node_representative_tensor_cache
        if self._node_representative_tensor_cache_key == cache_key and cached is not None and cached.device == self.device:
            return cached
        max_node = max([0] + [int(q) for q in node_ids])
        reps = torch.full((max_node + 1,), int(self.comm_rank), device=self.device, dtype=torch.long)
        for q in sorted(set(int(x) for x in node_ids)):
            reps[int(q)] = int(self._node_representative_comm_rank(int(q)))
        self._node_representative_tensor_cache_key = cache_key
        self._node_representative_tensor_cache = reps
        return reps

    def _expert_to_node_rep_comm_rank_cpu(self) -> List[int]:
        """CPU lookup used by the decode logical NGD fastpath.

        For batch=1 decode, launching several tiny GPU kernels just to count
        destination nodes costs more than the O(top-k) work itself.  This cached
        table maps each expert directly to the representative comm-rank for its
        logical node, matching ``_node_representative_tensor`` semantics.
        """
        node_ids = self._comm_rank_node_ids()
        cache_key = (
            tuple(int(x) for x in self.expert_to_node_cpu),
            tuple(int(x) for x in node_ids),
            int(self.comm_rank),
        )
        cached = self._expert_to_node_rep_comm_cpu_cache
        if self._expert_to_node_rep_comm_cpu_cache_key == cache_key and cached is not None:
            return cached
        reps_by_node: Dict[int, int] = {}
        for node_id in sorted(set(int(x) for x in self.expert_to_node_cpu)):
            reps_by_node[int(node_id)] = int(self._node_representative_comm_rank(int(node_id)))
        out = [
            int(reps_by_node.get(int(node_id), int(self.comm_rank)))
            for node_id in self.expert_to_node_cpu
        ]
        self._expert_to_node_rep_comm_cpu_cache_key = cache_key
        self._expert_to_node_rep_comm_cpu_cache = out
        return out

    def _expert_to_node_rep_comm_rank_tensor(self) -> torch.Tensor:
        node_ids = self._comm_rank_node_ids()
        cache_key = (
            tuple(int(x) for x in self.expert_to_node_cpu),
            tuple(int(x) for x in node_ids),
            int(self.comm_rank),
            str(self.device),
        )
        cached = self._expert_to_node_rep_comm_tensor_cache
        if self._expert_to_node_rep_comm_tensor_cache_key == cache_key and cached is not None and cached.device == self.device:
            return cached
        out = torch.tensor(self._expert_to_node_rep_comm_rank_cpu(), device=self.device, dtype=torch.long)
        self._expert_to_node_rep_comm_tensor_cache_key = cache_key
        self._expert_to_node_rep_comm_tensor_cache = out
        return out

    def _logical_fastpath_cpu_decode_counts(
        self,
        idx: torch.Tensor,
        w: torch.Tensor,
        *,
        max_items: int,
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Count node-grouped logical sends with one tiny CPU transfer.

        This is intentionally only for low-cardinality decode.  It avoids the
        previous logical path's ``present`` matrix, ``nonzero``, ``bincount`` and
        scalar GPU synchronizations while preserving the same modeled weaknet
        accounting granularity: one item per non-empty (token, destination node).
        """
        try:
            T = int(idx.shape[0])
            K = int(idx.shape[1])
        except Exception:
            return None
        if T <= 0 or K <= 0 or int(T * K) > int(max_items):
            return None
        try:
            idx_rows = idx.detach().to("cpu", dtype=torch.long).view(T, K).tolist()
        except Exception:
            return None

        check_weights = _env_flag("SPARSE_NGD_LOGICAL_FASTPATH_CPU_CHECK_WEIGHTS", "1")
        w_rows: Optional[List[List[float]]] = None
        if check_weights:
            try:
                w_rows = w.detach().to("cpu", dtype=torch.float32).view(T, K).tolist()
            except Exception:
                return None

        expert_to_rep = self._expert_to_node_rep_comm_rank_cpu()
        send_counts_list = [0 for _ in range(int(self.comm_world))]
        node_packet_count = 0
        for t in range(T):
            seen: set[int] = set()
            for j in range(K):
                if w_rows is not None and float(w_rows[t][j]) == 0.0:
                    continue
                eid = int(idx_rows[t][j])
                dst = int(expert_to_rep[eid]) if 0 <= eid < len(expert_to_rep) else int(self.comm_rank)
                if 0 <= dst < int(self.comm_world):
                    seen.add(dst)
            for dst in seen:
                send_counts_list[int(dst)] += 1
                node_packet_count += 1
        return torch.tensor(send_counts_list, dtype=torch.long), int(node_packet_count)

    def _pack_node_packets_cpu_decode(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
        *,
        max_items: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Low-cardinality decode packet pack using one tiny CPU copy.

        For batch-1 decode, launching multiple tiny GPU/Triton kernels and then
        compacting with boolean masks can cost more than grouping the top-k
        routing result directly on CPU.  This path preserves physical NGD
        semantics: each output row is one (token, destination-node) packet.
        """
        try:
            T = int(idx.shape[0])
            K = int(idx.shape[1])
        except Exception:
            return None
        if T <= 0 or K <= 0 or int(T * K) > int(max_items):
            return None
        try:
            idx_rows = idx.detach().to("cpu", dtype=torch.long).view(T, K).tolist()
            w_rows = w.detach().to("cpu", dtype=torch.float32).view(T, K).tolist()
        except Exception:
            return None
        expert_to_rep = self._expert_to_node_rep_comm_rank_cpu()
        dest_vals: List[int] = []
        token_vals: List[int] = []
        expert_rows: List[List[int]] = []
        weight_rows: List[List[float]] = []
        for t in range(T):
            by_dst: Dict[int, List[int]] = {}
            for j in range(K):
                if float(w_rows[t][j]) == 0.0:
                    continue
                eid = int(idx_rows[t][j])
                dst = int(expert_to_rep[eid]) if 0 <= eid < len(expert_to_rep) else int(self.comm_rank)
                by_dst.setdefault(dst, []).append(j)
            for dst in sorted(by_dst):
                slots = by_dst[dst]
                e_row = [-1 for _ in range(K)]
                w_row = [0.0 for _ in range(K)]
                for out_j, src_j in enumerate(slots[:K]):
                    e_row[out_j] = int(idx_rows[t][src_j])
                    w_row[out_j] = float(w_rows[t][src_j])
                dest_vals.append(int(dst))
                token_vals.append(int(t))
                expert_rows.append(e_row)
                weight_rows.append(w_row)
        if not dest_vals:
            empty_l = torch.empty((0,), device=self.device, dtype=torch.long)
            return (
                empty_l,
                empty_l,
                torch.empty((0, K), device=self.device, dtype=torch.long),
                torch.empty((0, K), device=self.device, dtype=x.dtype),
            )
        return (
            torch.tensor(dest_vals, device=self.device, dtype=torch.long),
            torch.tensor(token_vals, device=self.device, dtype=torch.long),
            torch.tensor(expert_rows, device=self.device, dtype=torch.long),
            torch.tensor(weight_rows, device=self.device, dtype=x.dtype),
        )

    def _node_grouped_fixed_a2a_meta_dtype(self, activation_dtype: torch.dtype) -> Optional[torch.dtype]:
        """Return the metadata packing dtype for the decode fixed-slot path."""
        raw = str(os.environ.get("SPARSE_NGD_FIXED_A2A_META_DTYPE", "activation") or "activation").strip().lower()
        if raw in ("fp32", "float32"):
            return torch.float32
        if raw not in ("activation", "act", "x", ""):
            return None
        if activation_dtype in (torch.float32, torch.float64):
            return activation_dtype
        if activation_dtype == torch.float16:
            return activation_dtype if int(self.num_experts) <= 2048 else None
        if activation_dtype == torch.bfloat16:
            return activation_dtype if int(self.num_experts) <= 256 else None
        return None

    def _node_grouped_owner_payload_dtype(self, activation_dtype: torch.dtype) -> Optional[torch.dtype]:
        raw = str(os.environ.get("SPARSE_NGD_OWNER_PACKED_PAYLOAD_DTYPE", "activation") or "activation").strip().lower()
        if raw in ("fp32", "float32"):
            return torch.float32
        if raw not in ("activation", "act", "x", ""):
            return None
        if activation_dtype in (torch.float32, torch.float64):
            return activation_dtype
        # Expert ids are packed into the payload dtype.  This is exact for the
        # small expert-id range in the current MoE; otherwise fall back.
        if activation_dtype == torch.float16:
            return activation_dtype if int(self.num_experts) <= 2048 else None
        if activation_dtype == torch.bfloat16:
            return activation_dtype if int(self.num_experts) <= 256 else None
        return None

    def _pack_owner_grouped_payload(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        *,
        payload_dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            _env_flag("SPARSE_NGD_OWNER_PAYLOAD_TRITON", "1")
            and _node_grouped_triton is not None
            and int(hidden.shape[0]) > 0
        ):
            try:
                if _node_grouped_triton.can_use_owner_payload_triton(hidden, ids, weights):
                    return _node_grouped_triton.pack_owner_payload_triton(
                        hidden,
                        ids,
                        weights,
                        payload_dtype=payload_dtype,
                    )
            except Exception:
                if self._node_grouped_triton_strict():
                    raise
        return torch.cat(
            [
                hidden.to(payload_dtype),
                ids.to(payload_dtype),
                weights.to(payload_dtype),
            ],
            dim=1,
        ).contiguous()

    def _unpack_owner_grouped_payload(
        self,
        payload: torch.Tensor,
        *,
        hidden_dim: int,
        top_k: int,
        activation_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            _env_flag("SPARSE_NGD_OWNER_PAYLOAD_TRITON", "1")
            and _node_grouped_triton is not None
            and int(payload.shape[0]) > 0
        ):
            try:
                return _node_grouped_triton.unpack_owner_payload_triton(
                    payload,
                    hidden_dim=int(hidden_dim),
                    top_k=int(top_k),
                    activation_dtype=activation_dtype,
                )
            except Exception:
                if self._node_grouped_triton_strict():
                    raise
        hidden = payload[:, : int(hidden_dim)].to(activation_dtype).contiguous()
        ids = payload[:, int(hidden_dim) : int(hidden_dim) + int(top_k)].to(torch.float32).round().to(torch.long).contiguous()
        weights = payload[:, int(hidden_dim) + int(top_k) : int(hidden_dim) + 2 * int(top_k)].to(activation_dtype).contiguous()
        return hidden, ids, weights

    def _build_fixed_node_rep_payload_tensor(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
        *,
        rep_ranks: Tuple[int, ...],
        rep_slot: Dict[int, int],
        meta_dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        if T != 1 or K <= 0:
            return None
        try:
            reps = torch.tensor(list(rep_ranks), device=self.device, dtype=torch.long)
            rep_to_slot = torch.full((int(self.comm_world),), -1, device=self.device, dtype=torch.long)
            rep_to_slot[reps] = torch.arange(int(len(rep_ranks)), device=self.device, dtype=torch.long)
            expert_to_rep = self._expert_to_node_rep_comm_rank_tensor()

            idx1 = idx.view(-1).to(torch.long)
            w1 = w.view(-1).to(torch.float32)
            dst = expert_to_rep[idx1]
            valid = w1.ne(0.0) & dst.ge(0) & dst.lt(int(self.comm_world))
            dst = dst[valid]
            idxv = idx1[valid]
            wv = w1[valid]
            if int(dst.numel()) == 0:
                return None
            slot_idx = rep_to_slot[dst]
            valid_slot = slot_idx.ge(0)
            slot_idx = slot_idx[valid_slot]
            idxv = idxv[valid_slot]
            wv = wv[valid_slot]
            if int(slot_idx.numel()) == 0:
                return None

            width = int(1 + H + 2 * K)
            send_payload = self._node_grouped_empty(
                (int(len(rep_ranks)), width),
                dtype=meta_dtype,
                label="fixed_node_rep_dispatch_send",
                workspace=True,
                zero=True,
            )
            send_payload[:, 1 : 1 + H] = x[0].to(meta_dtype).view(1, H)
            send_payload[:, 1 + H : 1 + H + K].fill_(-1.0)
            send_payload[:, 1 + H + K : 1 + H + 2 * K].zero_()
            present = torch.zeros((int(len(rep_ranks)),), device=self.device, dtype=torch.bool)
            present[slot_idx] = True
            send_payload[:, 0] = present.to(meta_dtype)

            pos = torch.arange(int(slot_idx.numel()), device=self.device, dtype=torch.long)
            slot = torch.zeros_like(slot_idx)
            for j in range(int(slot_idx.numel())):
                cur = slot_idx.eq(slot_idx[j]) & pos.le(j)
                slot[j] = cur.sum() - 1
            keep = slot.lt(K)
            rows = slot_idx[keep]
            cols = slot[keep]
            send_payload[rows, 1 + H + cols] = idxv[keep].to(meta_dtype)
            send_payload[rows, 1 + H + K + cols] = wv[keep].to(meta_dtype)
            send_counts = torch.zeros((int(self.comm_world),), device=self.device, dtype=torch.long)
            send_counts[dst.unique()] = 1
            return send_payload, send_counts, int(rows.unique().numel())
        except Exception:
            return None

    def _experts_ep_group_node_grouped_decode_fixed_node_rep(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
        *,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        """Decode-only physical NGD fastpath with fixed-slot node reps.

        For one decode token, each source rank can send at most one packet to a
        logical node representative.  This path replaces variable metadata
        exchange plus two-stage fanout/gather with two fixed all-to-all calls:
        dispatch and gather.  Router outputs and expert ids are unchanged; only
        the explicitly enabled execution owner inside a logical node changes.
        """
        if int(seq_len) != 1:
            return None
        if self.topology is None or self.comm_group is None:
            return None
        if self.expert_offload != "cpu":
            return None
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        if T != 1 or H <= 0 or K <= 0:
            return None
        meta_dtype = self._node_grouped_fixed_a2a_meta_dtype(x.dtype)
        if meta_dtype is None:
            return None

        t_pack = time.perf_counter()
        node_ids = self._comm_rank_node_ids()
        my_node = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
        rep_ranks = tuple(
            sorted({
                int(self._node_representative_comm_rank(int(q)))
                for q in set(int(v) for v in node_ids)
            })
        )
        rep_set = set(int(r) for r in rep_ranks)
        if not rep_ranks:
            return None
        is_rep_rank = int(self.comm_rank) in rep_set
        rep_slot = {int(r): pos for pos, r in enumerate(rep_ranks)}

        width = int(1 + H + 2 * K)
        tensor_pack = None
        if _env_flag("SPARSE_NGD_FIXED_A2A_TENSOR_PACK", "1"):
            tensor_pack = self._build_fixed_node_rep_payload_tensor(
                x,
                idx,
                w,
                rep_ranks=rep_ranks,
                rep_slot=rep_slot,
                meta_dtype=meta_dtype,
            )
        if tensor_pack is not None:
            send_payload, send_counts, node_packet_count = tensor_pack
            send_counts_list = [int(v) for v in send_counts.detach().to("cpu", dtype=torch.long).tolist()]
            active_dsts = [i for i, v in enumerate(send_counts_list) if int(v) > 0]
        else:
            try:
                idx_row = idx.detach().to("cpu", dtype=torch.long).view(T, K)[0].tolist()
                w_row = w.detach().to("cpu", dtype=torch.float32).view(T, K)[0].tolist()
            except Exception:
                return None
            expert_to_rep = self._expert_to_node_rep_comm_rank_cpu()
            by_dst: Dict[int, List[int]] = {}
            for j in range(K):
                if float(w_row[j]) == 0.0:
                    continue
                eid = int(idx_row[j])
                dst = int(expert_to_rep[eid]) if 0 <= eid < len(expert_to_rep) else int(self.comm_rank)
                if 0 <= dst < int(self.comm_world):
                    by_dst.setdefault(dst, []).append(j)
            if not by_dst:
                return torch.zeros_like(x)
            send_payload = self._node_grouped_empty(
                (int(len(rep_ranks)), width),
                dtype=meta_dtype,
                label="fixed_node_rep_dispatch_send",
                workspace=True,
                zero=True,
            )
            hidden_meta = x[0].to(meta_dtype)
            send_counts_list = [0 for _ in range(int(self.comm_world))]
            node_packet_count = 0
            for dst in sorted(by_dst):
                slots = by_dst[int(dst)]
                slot_idx = int(rep_slot.get(int(dst), -1))
                if slot_idx < 0:
                    continue
                send_payload[slot_idx, 0] = 1.0
                send_payload[slot_idx, 1 : 1 + H] = hidden_meta
                for out_j, src_j in enumerate(slots[:K]):
                    eid = int(idx_row[int(src_j)])
                    send_payload[slot_idx, 1 + H + out_j] = float(eid)
                    send_payload[slot_idx, 1 + H + K + out_j] = float(w_row[int(src_j)])
                for out_j in range(len(slots[:K]), K):
                    send_payload[slot_idx, 1 + H + out_j] = -1.0
                    send_payload[slot_idx, 1 + H + K + out_j] = 0.0
                send_counts_list[int(dst)] = 1
                node_packet_count += 1
            send_counts = torch.tensor(send_counts_list, dtype=torch.long, device=self.device)
            active_dsts = list(sorted(by_dst))

        if int(node_packet_count) <= 0:
            return torch.zeros_like(x)
        same_node_packets = 0
        cross_node_packets = 0
        for dst in active_dsts:
            dst_node = int(node_ids[int(dst)]) if int(dst) < len(node_ids) else my_node
            if int(dst) != int(self.comm_rank):
                if int(dst_node) == int(my_node):
                    same_node_packets += 1
                else:
                    cross_node_packets += 1

        self._record_node_grouped_value("node_packet_entries", int(node_packet_count))
        self._record_node_grouped_value("rank_packet_entries", int(node_packet_count))
        self._record_node_grouped_value("same_node_packet_entries", int(same_node_packets))
        self._record_node_grouped_value("cross_node_packet_entries", int(cross_node_packets))
        self._record_node_grouped_value("stage2_local_fastpath_skip", 1)
        self._record_node_grouped_value("metadata_collective_count", 2)
        if self.metrics is not None:
            self.metrics.node_grouped_pair_entries += int(T * K)
            self.metrics.node_grouped_node_packet_entries += int(node_packet_count)
            self.metrics.node_grouped_rank_packet_entries += int(node_packet_count)
            self.metrics.node_grouped_dispatch_calls += 1
        self._record_node_grouped_allocation(send_payload)
        self._record_node_grouped_ms("pack", (time.perf_counter() - t_pack) * 1000.0)

        meta_elem_bytes = int(torch.empty((), dtype=meta_dtype).element_size())
        payload_bytes_per_item = int(width * meta_elem_bytes)
        gather_bytes_per_item = int(H * x.element_size())
        logical_payload_bytes = payload_bytes_per_item
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            logical_payload_bytes += gather_bytes_per_item
        count_elem_bytes = int(torch.empty((), dtype=torch.long).element_size())
        counts_bytes = int(4 * int(self.comm_world) * count_elem_bytes)
        self._record_node_grouped_value("metadata_bytes", int(2 * K * node_packet_count * meta_elem_bytes))
        self._record_activation_comm_bytes(int(counts_bytes) + int(node_packet_count * payload_bytes_per_item))
        self._record_activation_remote_bytes(int(cross_node_packets * payload_bytes_per_item))
        self._record_dispatch_self_remote(send_counts.nonzero(as_tuple=False).flatten().to(self.device))

        t_inter = time.perf_counter()
        t_meta = time.perf_counter()
        recv_counts = self._exchange_dispatch_counts(
            send_counts,
            group=self.comm_group,
            group_rank=self.comm_rank,
            label="node_grouped_fixed_stage1",
        )
        dt_meta = (time.perf_counter() - t_meta) * 1000.0
        self._record_node_grouped_ms("metadata", dt_meta)
        self._record_node_grouped_value("metadata_exchange_ms", dt_meta)
        recv_counts_list = [int(v) for v in recv_counts.detach().to("cpu", dtype=torch.long).tolist()]
        recv_total = int(sum(recv_counts_list))
        weaknet_payload = _maybe_sleep_logical_weaknet(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=send_counts,
            payload_bytes_per_item=int(logical_payload_bytes),
            metrics=self.metrics,
            layer_idx=self.layer_idx,
            label="node_grouped_fixed_node_rep",
        )
        if _logical_weaknet_enabled():
            sleep_ms = float(weaknet_payload.get("sim_ms", 0.0))
            self._record_node_grouped_value("inter_send_stage1_weaknet_sleep_ms", sleep_ms)
            self._record_node_grouped_value("inter_send_weaknet_sleep_ms", sleep_ms)

        recv_payload = self._node_grouped_empty(
            (int(recv_total), width),
            dtype=meta_dtype,
            label="fixed_node_rep_dispatch_recv",
            workspace=True,
        )
        t_dispatch = time.perf_counter()
        send_split = [int(v) * width for v in send_counts_list]
        recv_split = [int(v) * width for v in recv_counts_list]
        active_payload_rows = [
            int(rep_slot[int(dst)])
            for dst, cnt in enumerate(send_counts_list)
            if int(cnt) > 0 and int(dst) in rep_slot
        ]
        if active_payload_rows:
            send_payload_compact = send_payload.index_select(
                0,
                torch.tensor(active_payload_rows, device=self.device, dtype=torch.long),
            ).contiguous()
        else:
            send_payload_compact = torch.empty((0, width), dtype=meta_dtype, device=self.device)
        self._all_to_all_single_timed(
            recv_payload.reshape(-1),
            send_payload_compact.reshape(-1),
            recv_split,
            send_split,
            group=self.comm_group,
        )
        self._record_node_grouped_value("inter_send_stage1_collective_ms", (time.perf_counter() - t_dispatch) * 1000.0)
        self._record_node_grouped_ms("inter_send", (time.perf_counter() - t_inter) * 1000.0)

        valid = recv_payload[:, 0].to(torch.float32).gt(0.5)
        recv_src_list: List[int] = []
        for src_rank, cnt in enumerate(recv_counts_list):
            recv_src_list.extend([int(src_rank)] * int(cnt))
        recv_src = torch.tensor(recv_src_list, device=self.device, dtype=torch.long)
        valid_rows = torch.nonzero(valid, as_tuple=False).flatten().to(torch.long)
        valid_src = recv_src.index_select(0, valid_rows) if int(valid_rows.numel()) > 0 else recv_src.new_empty((0,))
        if int(valid_rows.numel()) == 0:
            node_packet_out = torch.empty((0, H), dtype=x.dtype, device=self.device)
        else:
            recv_valid = recv_payload.index_select(0, valid_rows).contiguous()
            recv_hidden = recv_valid[:, 1 : 1 + H].to(x.dtype).contiguous()
            recv_i = recv_valid[:, 1 + H : 1 + H + K].to(torch.float32).round().to(torch.long).contiguous()
            recv_w = recv_valid[:, 1 + H + K : 1 + H + 2 * K].to(x.dtype).contiguous()
            self._record_local_expert_load(recv_i[recv_i.ge(0)])
            t_exec = time.perf_counter()
            node_packet_out = self._execute_local_rank_grouped_packets(
                recv_hidden=recv_hidden,
                recv_i=recv_i,
                recv_w=recv_w,
                allow_nonlocal_cpu_once=True,
            )
            self._record_expert_exec_ms((time.perf_counter() - t_exec) * 1000.0)

        t_gather = time.perf_counter()
        if bool(is_rep_rank) and int(valid_src.numel()) > 0:
            send_back = node_packet_out.to(x.dtype).contiguous()
            valid_src_cpu = [int(v) for v in valid_src.detach().to("cpu", dtype=torch.long).tolist()]
        else:
            send_back = torch.empty((0, H), dtype=x.dtype, device=self.device)
            valid_src_cpu = []
        gather_send_counts = torch.zeros((int(self.comm_world),), device=self.device, dtype=torch.long)
        if valid_src_cpu:
            gather_send_counts[torch.tensor(valid_src_cpu, device=self.device, dtype=torch.long)] = 1
        t_meta2 = time.perf_counter()
        gather_recv_counts = self._exchange_dispatch_counts(
            gather_send_counts,
            group=self.comm_group,
            group_rank=self.comm_rank,
            label="node_grouped_fixed_gather",
        )
        dt_meta2 = (time.perf_counter() - t_meta2) * 1000.0
        self._record_node_grouped_ms("metadata", dt_meta2)
        self._record_node_grouped_value("metadata_exchange_ms", dt_meta2)
        gather_recv_counts_list = [int(v) for v in gather_recv_counts.detach().to("cpu", dtype=torch.long).tolist()]
        gather_recv_total = int(sum(gather_recv_counts_list))
        recv_back = self._node_grouped_empty(
            (int(gather_recv_total), H),
            dtype=x.dtype,
            label="fixed_node_rep_gather_recv",
            workspace=True,
        )
        gather_send_counts_list = [int(v) for v in gather_send_counts.detach().to("cpu", dtype=torch.long).tolist()]
        gather_send_split = [int(v) * H for v in gather_send_counts_list]
        gather_recv_split = [int(v) * H for v in gather_recv_counts_list]
        gather_remote_records = sum(
            1
            for r in valid_src_cpu
            if int(r) != int(self.comm_rank)
            and int(node_ids[int(r)] if int(r) < len(node_ids) else my_node) != int(my_node)
        )
        self._record_activation_comm_bytes(int(len(valid_src_cpu) * gather_bytes_per_item))
        self._record_activation_remote_bytes(int(gather_remote_records * gather_bytes_per_item))
        self._all_to_all_single_timed(
            recv_back.reshape(-1),
            send_back.reshape(-1).contiguous(),
            gather_recv_split,
            gather_send_split,
            group=self.comm_group,
        )
        self._record_node_grouped_value("node_to_token_restore_ms", (time.perf_counter() - t_gather) * 1000.0)
        self._record_node_grouped_ms("gather", (time.perf_counter() - t_gather) * 1000.0)
        self._record_node_grouped_allocation(recv_payload, send_back, recv_back)
        if int(gather_recv_total) <= 0:
            return torch.zeros_like(x)
        return recv_back.sum(dim=0, keepdim=True).to(x.dtype)

    def _logical_ngd_counts_from_ep_rank_counts(
        self,
        send_counts_list: Sequence[int],
    ) -> Tuple[torch.Tensor, int]:
        """Approximate node-grouped logical counts from normal EP rank counts.

        This fast path is used only for single-request decode, where each rank
        contributes one token per layer. Normal EP already needs rank-level
        counts for the real execution path. Folding those counts by logical
        destination node avoids an additional GPU->CPU sync before normal EP.

        The count is exact for the common top-k case where at most one node-level
        item can be emitted per destination node for the current token. Multiple
        experts on the same destination node become one node-grouped logical
        item, matching NGD event semantics. For prefill or multi-token inputs,
        the caller should use the tensor/CPU paths because per-token coalescing
        cannot be reconstructed from aggregate rank counts alone.
        """
        node_ids = self._comm_rank_node_ids()
        my_node = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
        send_counts_node = [0 for _ in range(int(self.comm_world))]
        node_packet_count = 0
        seen_nodes: set[int] = set()
        for comm_dst, cnt in enumerate(send_counts_list):
            if int(cnt) <= 0 or int(comm_dst) >= len(node_ids):
                continue
            dst_node = int(node_ids[int(comm_dst)])
            if dst_node in seen_nodes:
                continue
            seen_nodes.add(dst_node)
            rep = int(self.comm_rank) if dst_node == my_node else int(self._node_representative_comm_rank(dst_node))
            if 0 <= rep < int(self.comm_world):
                send_counts_node[rep] += 1
                node_packet_count += 1
        return torch.tensor(send_counts_node, dtype=torch.long), int(node_packet_count)

    def _maybe_charge_logical_ngd_from_ep_counts(
        self,
        *,
        send_counts_list: Sequence[int],
        hidden_dim: int,
        top_k: int,
        dtype_bytes: int,
        rank_count_dtype_bytes: int,
    ) -> Optional[Dict[str, float]]:
        if not bool(getattr(SparseContext, "logical_ngd_ep_reuse_once", False)):
            return None
        SparseContext.logical_ngd_ep_reuse_once = False
        t0 = time.perf_counter()
        send_counts, node_packet_count = self._logical_ngd_counts_from_ep_rank_counts(send_counts_list)
        node_payload_bytes = int(hidden_dim) * int(dtype_bytes)
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            node_payload_bytes += int(top_k) * int(rank_count_dtype_bytes) + int(top_k) * int(dtype_bytes)
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            node_payload_bytes += int(hidden_dim) * int(dtype_bytes)

        self._record_node_grouped_value("node_packet_entries", int(node_packet_count))
        self._record_node_grouped_value("rank_packet_entries", int(top_k))
        if self.metrics is not None:
            self.metrics.node_grouped_pair_entries += int(top_k)
            self.metrics.node_grouped_node_packet_entries += int(node_packet_count)
            self.metrics.node_grouped_rank_packet_entries += int(top_k)
            self.metrics.node_grouped_dispatch_calls += 1

        self._record_node_grouped_ms("pack", (time.perf_counter() - t0) * 1000.0)
        t_sleep = time.perf_counter()
        payload = _maybe_sleep_logical_weaknet(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=send_counts,
            payload_bytes_per_item=node_payload_bytes,
            metrics=self.metrics,
            layer_idx=self.layer_idx,
            label="node_grouped_logical_ep_reuse",
        )
        if _logical_weaknet_enabled():
            sleep_ms = float(payload.get("sim_ms", 0.0))
            self._record_node_grouped_value("inter_send_stage1_weaknet_sleep_ms", sleep_ms)
            self._record_node_grouped_value("inter_send_weaknet_sleep_ms", sleep_ms)
        self._record_node_grouped_ms("inter_send", (time.perf_counter() - t_sleep) * 1000.0)
        return payload

    def _node_grouped_tensor_fastpath_enabled(self, seq_len: int) -> bool:
        phase = "DECODE" if int(seq_len) == 1 else "PREFILL"
        phase_raw = os.environ.get(f"SPARSE_NODE_GROUPED_TENSOR_FASTPATH_{phase}", "").strip().lower()
        if phase_raw != "":
            return phase_raw not in ("0", "false", "no", "off")
        mode = os.environ.get("SPARSE_NODE_GROUPED_TENSOR_FASTPATH_MODE", "").strip().lower()
        if mode in ("all", "both", "1", "true", "yes", "on"):
            return True
        if mode == "decode":
            return phase == "DECODE"
        if mode == "prefill":
            return phase == "PREFILL"
        return _env_flag("SPARSE_NODE_GROUPED_TENSOR_FASTPATH", "0")

    def _node_grouped_triton_pack_enabled(self, seq_len: int, idx: torch.Tensor, w: torch.Tensor) -> bool:
        phase = "DECODE" if int(seq_len) == 1 else "PREFILL"
        phase_raw = os.environ.get(f"SPARSE_NODE_GROUPED_TRITON_PACK_{phase}", "").strip().lower()
        if phase_raw != "":
            requested = phase_raw not in ("0", "false", "no", "off")
        else:
            requested = _env_flag("SPARSE_NODE_GROUPED_TRITON_PACK", "0")
        if not requested:
            return False
        if _node_grouped_triton is None:
            return False
        try:
            return bool(_node_grouped_triton.can_use_node_grouped_triton(idx, w))
        except Exception:
            return False

    def _node_grouped_triton_strict(self) -> bool:
        return _env_flag("SPARSE_NODE_GROUPED_TRITON_STRICT", "0")

    def _node_grouped_phase_flag(self, name: str, seq_len: int, default: str = "0") -> bool:
        phase = "DECODE" if int(seq_len) == 1 else "PREFILL"
        phase_raw = os.environ.get(f"{name}_{phase}", "").strip().lower()
        if phase_raw != "":
            return phase_raw not in ("0", "false", "no", "off")
        return _env_flag(name, default)

    def _node_grouped_fastpath_mode(self, seq_len: int) -> str:
        phase = "DECODE" if int(seq_len) == 1 else "PREFILL"
        default = "legacy" if phase == "DECODE" else "tensor"
        raw = os.environ.get(f"SPARSE_NODE_GROUPED_FASTPATH_{phase}_MODE", "").strip().lower()
        if raw == "":
            raw = os.environ.get("SPARSE_NODE_GROUPED_FASTPATH_MODE", default).strip().lower()
        if _env_flag("SPARSE_NGD_LEAN_FASTPATH", "0"):
            return "lean"
        return raw or default

    def _node_grouped_identity_ids(self, ids: torch.Tensor, n: int) -> bool:
        if int(n) <= 0:
            return True
        if int(ids.numel()) != int(n):
            return False
        return bool(ids.eq(torch.arange(int(n), device=ids.device, dtype=ids.dtype)).all().item())

    def _pack_node_packets_tensorized(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tensorized Stage-1 pack for node-grouped dispatch.

        The legacy path converts routing decisions to CPU lists and groups them
        with Python dicts. This path keeps the per-token node coalescing on the
        active device: each output row is one (token, logical-node) packet with
        up to K experts for that node.
        """
        T = int(idx.shape[0])
        K = int(idx.shape[1])
        if T == 0 or K == 0:
            empty_l = torch.empty((0,), device=self.device, dtype=torch.long)
            return (
                empty_l,
                empty_l,
                torch.empty((0, K), device=self.device, dtype=torch.long),
                torch.empty((0, K), device=self.device, dtype=x.dtype),
            )

        flat_i = idx.reshape(-1).to(torch.long)
        dest_rank_mat = self._resolve_dest_for_flat_experts(flat_i, reason="node_grouped_pack", count_metrics=False).view(T, K).to(torch.long)
        rank_to_node = self._rank_to_node_tensor()
        dest_node_mat = rank_to_node[dest_rank_mat]
        max_node = int(rank_to_node.max().item()) if rank_to_node.numel() > 0 else 0
        valid = w.to(torch.float32).ne(0.0)

        present = torch.zeros((T, max_node + 1), device=self.device, dtype=torch.bool)
        token_grid = torch.arange(T, device=self.device, dtype=torch.long).view(T, 1).expand(T, K)
        present[token_grid[valid], dest_node_mat[valid]] = True
        token_ids, node_vals = torch.nonzero(present, as_tuple=True)
        packet_count = int(token_ids.numel())
        if packet_count == 0:
            empty_l = torch.empty((0,), device=self.device, dtype=torch.long)
            return (
                empty_l,
                empty_l,
                torch.empty((0, K), device=self.device, dtype=torch.long),
                torch.empty((0, K), device=self.device, dtype=x.dtype),
            )

        reps = self._node_representative_tensor()
        send_node_dest = reps[node_vals].to(torch.long)
        send_token_ids = token_ids.to(torch.long)

        packet_lookup = torch.full((T, max_node + 1), -1, device=self.device, dtype=torch.long)
        packet_lookup[token_ids, node_vals] = torch.arange(packet_count, device=self.device, dtype=torch.long)
        src_packet = packet_lookup[token_grid, dest_node_mat]

        slot = torch.zeros_like(dest_node_mat)
        for j in range(K):
            prev_same = dest_node_mat[:, : j + 1].eq(dest_node_mat[:, j : j + 1]) & valid[:, : j + 1]
            slot[:, j] = prev_same.sum(dim=1) - 1

        send_node_i = torch.full((packet_count, K), -1, device=self.device, dtype=torch.long)
        send_node_w = torch.zeros((packet_count, K), device=self.device, dtype=x.dtype)
        valid_flat = valid.reshape(-1)
        send_node_i[src_packet.reshape(-1)[valid_flat], slot.reshape(-1)[valid_flat]] = flat_i[valid_flat]
        send_node_w[src_packet.reshape(-1)[valid_flat], slot.reshape(-1)[valid_flat]] = w.reshape(-1).to(x.dtype)[valid_flat]
        return send_node_dest, send_token_ids, send_node_i, send_node_w

    def _pack_rank_packets_tensorized(
        self,
        recv_node_i: torch.Tensor,
        recv_node_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tensorized Stage-2 pack for node-grouped dispatch.

        Splits received node packets into one packet per destination rank while
        keeping packet construction on device. Expert ownership is unchanged.
        """
        P = int(recv_node_i.shape[0])
        K = int(recv_node_i.shape[1]) if recv_node_i.dim() == 2 else 0
        if P == 0 or K == 0:
            empty_l = torch.empty((0,), device=self.device, dtype=torch.long)
            return (
                empty_l,
                empty_l,
                torch.empty((0, K), device=self.device, dtype=torch.long),
                torch.empty((0, K), device=self.device, dtype=recv_node_w.dtype),
            )

        valid = recv_node_i.ge(0) & recv_node_w.to(torch.float32).ne(0.0)
        if not bool(valid.any().item()):
            empty_l = torch.empty((0,), device=self.device, dtype=torch.long)
            return (
                empty_l,
                empty_l,
                torch.empty((0, K), device=self.device, dtype=torch.long),
                torch.empty((0, K), device=self.device, dtype=recv_node_w.dtype),
            )

        safe_i = recv_node_i.clamp_min(0).to(torch.long)
        dest_rank_mat = self.expert_to_rank[safe_i.reshape(-1)].view(P, K).to(torch.long)
        packet_grid = torch.arange(P, device=self.device, dtype=torch.long).view(P, 1).expand(P, K)

        rank_present = torch.zeros((P, int(self.comm_world)), device=self.device, dtype=torch.bool)
        rank_present[packet_grid[valid], dest_rank_mat[valid]] = True
        packet_ids, rank_vals = torch.nonzero(rank_present, as_tuple=True)
        rank_packet_count = int(packet_ids.numel())
        if rank_packet_count == 0:
            empty_l = torch.empty((0,), device=self.device, dtype=torch.long)
            return (
                empty_l,
                empty_l,
                torch.empty((0, K), device=self.device, dtype=torch.long),
                torch.empty((0, K), device=self.device, dtype=recv_node_w.dtype),
            )

        packet_lookup = torch.full((P, int(self.comm_world)), -1, device=self.device, dtype=torch.long)
        packet_lookup[packet_ids, rank_vals] = torch.arange(rank_packet_count, device=self.device, dtype=torch.long)
        src_packet = packet_lookup[packet_grid[valid], dest_rank_mat[valid]]

        slot = torch.zeros_like(dest_rank_mat)
        for j in range(K):
            prev_same = dest_rank_mat[:, : j + 1].eq(dest_rank_mat[:, j : j + 1]) & valid[:, : j + 1]
            slot[:, j] = prev_same.sum(dim=1) - 1

        send_rank_i = torch.full((rank_packet_count, K), -1, device=self.device, dtype=torch.long)
        send_rank_w = torch.zeros((rank_packet_count, K), device=self.device, dtype=recv_node_w.dtype)
        send_rank_i[src_packet, slot[valid]] = recv_node_i[valid].to(torch.long)
        send_rank_w[src_packet, slot[valid]] = recv_node_w[valid]
        return rank_vals.to(torch.long), packet_ids.to(torch.long), send_rank_i, send_rank_w

    def _timed_segment(self, name: str, fn):
        if self.metrics is None or not _env_flag("SPARSE_EXPERT_TIMING", "0"):
            return fn()
        if not torch.cuda.is_available() or self.device.type != "cuda":
            t0 = time.perf_counter()
            out = fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
        else:
            start = torch.cuda.Event(enable_timing=True, blocking=False)
            end = torch.cuda.Event(enable_timing=True, blocking=False)
            start.record(torch.cuda.current_stream(self.device))
            out = fn()
            end.record(torch.cuda.current_stream(self.device))
            end.synchronize()
            dt_ms = float(start.elapsed_time(end))
        if name == "router":
            self.metrics.moe_router_ms += dt_ms
        elif name == "expert":
            self.metrics.moe_expert_ms += dt_ms
        elif name == "shared":
            self.metrics.moe_shared_ms += dt_ms
        self.metrics.moe_timing_calls += 1
        cp = _hclm_cp_profiler()
        step_id = int(getattr(SparseContext, "runtime_call_id", -1))
        if cp.should_sample(step_id) and str(name) in ("router", "shared"):
            now = _hclm_cp_now_ns()
            cp.record_component(
                step_id,
                int(self.layer_idx),
                "shared_expert" if str(name) == "shared" else str(name),
                now - int(max(float(dt_ms), 0.0) * 1_000_000.0),
                now,
            )
        return out

    def _experts_world1_exact(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        T = int(x.shape[0])
        H = int(x.shape[1])
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
        t_grouping = time.perf_counter()
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, H)
        flat_i = idx.reshape(-1).to(torch.long)
        flat_w = w.reshape(-1).to(expanded.dtype)
        dest = self._resolve_dest_for_flat_experts(flat_i, reason="ep_dispatch")
        sort_idx = torch.argsort(dest)
        send_hidden = expanded[sort_idx].contiguous()
        send_i = flat_i[sort_idx].contiguous()
        send_w = flat_w[sort_idx].contiguous()
        send_dest = dest[sort_idx].contiguous()
        send_counts = torch.bincount(send_dest, minlength=self.world).to(dtype=torch.long)
        recv_counts = torch.empty_like(send_counts)
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        self._all_to_all_single_timed(recv_counts, send_counts)
        send_counts_list = send_counts.cpu().tolist()
        recv_counts_list = recv_counts.cpu().tolist()
        t_grouping = time.perf_counter()
        recv_total = int(sum(recv_counts_list))
        fused_weight_dispatch = _ep_fused_weight_dispatch_enabled()
        if fused_weight_dispatch:
            send_hidden_w = torch.cat([send_hidden, send_w.unsqueeze(1)], dim=1).contiguous()
            recv_hidden_w = torch.empty((recv_total, H + 1), dtype=send_hidden.dtype, device=self.device)
            recv_i = torch.empty((recv_total,), dtype=send_i.dtype, device=self.device)
        else:
            recv_hidden = torch.empty((recv_total, H), dtype=send_hidden.dtype, device=self.device)
            recv_i = torch.empty((recv_total,), dtype=send_i.dtype, device=self.device)
            recv_w = torch.empty((recv_total,), dtype=send_w.dtype, device=self.device)
        counts_bytes = tensor_nbytes(send_counts) + tensor_nbytes(recv_counts)
        dispatch_bytes = (
            tensor_nbytes(send_hidden_w) + tensor_nbytes(send_i)
            if fused_weight_dispatch
            else tensor_nbytes(send_hidden) + tensor_nbytes(send_i) + tensor_nbytes(send_w)
        )
        gather_bytes = tensor_nbytes(send_hidden)
        self._record_activation_comm_bytes(counts_bytes + dispatch_bytes + gather_bytes)
        self._record_dispatch_self_remote(send_dest)
        remote_item_count = int((send_dest != int(self.comm_rank)).sum().item())
        if fused_weight_dispatch:
            remote_dispatch_bytes = remote_item_count * (
                int((H + 1) * send_hidden_w.element_size()) + int(send_i.element_size())
            )
        else:
            remote_dispatch_bytes = remote_item_count * (
                int(H * send_hidden.element_size()) + int(send_i.element_size()) + int(send_w.element_size())
            )
        remote_gather_bytes = remote_item_count * int(H * send_hidden.element_size())
        self._record_activation_remote_bytes(remote_dispatch_bytes + remote_gather_bytes)
        weaknet_item_bytes = int(H * send_hidden.element_size())
        if not fused_weight_dispatch:
            weaknet_item_bytes += int(send_w.element_size())
        elif fused_weight_dispatch:
            weaknet_item_bytes += int(send_hidden_w.element_size())
        if str(os.environ.get("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1")).strip().lower() not in ("0", "false", "no", "off"):
            weaknet_item_bytes += int(send_i.element_size())
        if str(os.environ.get("SPARSE_LOGICAL_SIM_GATHER", "1")).strip().lower() not in ("0", "false", "no", "off"):
            weaknet_item_bytes += int(H * send_hidden.element_size())
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        weaknet_payload = _maybe_sleep_logical_weaknet(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=send_counts,
            payload_bytes_per_item=weaknet_item_bytes,
            logical_transfer_multiplier=1,
            metrics=self.metrics,
            layer_idx=self.layer_idx,
            label="ep_dispatch",
        )
        cp = _hclm_cp_profiler()
        step_id = int(getattr(SparseContext, "runtime_call_id", -1))
        if cp.should_sample(step_id) and cp.record_modeled_comm:
            inter_events = float(weaknet_payload.get("inter_node_transfers", 0.0) or 0.0)
            inter_bytes = float(weaknet_payload.get("inter_node_bytes", 0.0) or 0.0)
            cp.record_layer_summary(
                step_id,
                int(self.layer_idx),
                {
                    "inter_node_events": inter_events,
                    "inter_node_bytes": inter_bytes,
                    **_hclm_cp_modeled_comm_ms(
                        inter_events,
                        inter_bytes,
                        bw_mbps=float(cp.bw_mbps),
                        lat_ms=float(cp.lat_ms),
                    ),
                    "notes": "modeled_dispatch_comm_ms includes normal EP logical dispatch payload and optional gather proxy per SPARSE_LOGICAL_SIM_GATHER.",
                },
            )
        t_grouping = time.perf_counter()
        if self.debug_a2a_checks:
            if fused_weight_dispatch and send_hidden_w.shape[0] != send_i.shape[0]:
                raise RuntimeError(
                    f"[rank{self.rank}] A2A shape mismatch: hidden_w={send_hidden_w.shape}, ids={send_i.shape}"
                )
            if (not fused_weight_dispatch) and (send_hidden.shape[0] != send_i.shape[0] or send_i.shape != send_w.shape):
                raise RuntimeError(
                    f"[rank{self.rank}] A2A shape mismatch: hidden={send_hidden.shape}, ids={send_i.shape}, weights={send_w.shape}"
                )
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        cp_dispatch_start = _hclm_cp_now_ns()
        if fused_weight_dispatch:
            self._all_to_all_single_timed(recv_hidden_w, send_hidden_w, recv_counts_list, send_counts_list)
            self._all_to_all_single_timed(recv_i, send_i, recv_counts_list, send_counts_list)
        else:
            self._all_to_all_single_timed(recv_hidden, send_hidden, recv_counts_list, send_counts_list)
            self._all_to_all_single_timed(recv_i, send_i, recv_counts_list, send_counts_list)
            self._all_to_all_single_timed(recv_w, send_w, recv_counts_list, send_counts_list)
        cp_dispatch_end = _hclm_cp_now_ns()
        if cp.should_sample(step_id):
            cp.record_component(step_id, int(self.layer_idx), "dispatch_comm", cp_dispatch_start, cp_dispatch_end)
        if fused_weight_dispatch:
            recv_hidden = recv_hidden_w[:, :H].contiguous()
            recv_w = recv_hidden_w[:, H].contiguous()
        self._record_local_expert_load(recv_i)
        t_exec = time.perf_counter()
        cp_exec_start = _hclm_cp_now_ns()
        cp_use_cuda_timing = bool(cp.should_sample(step_id) and torch.cuda.is_available() and self.device.type == "cuda")
        if cp_use_cuda_timing:
            cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_start.record(torch.cuda.current_stream(self.device))
        local_out = self._execute_local_recv_groups(recv_hidden=recv_hidden, recv_i=recv_i, recv_w=recv_w)
        cp_exec_end = _hclm_cp_now_ns()
        if cp_use_cuda_timing:
            cp_ev_end.record(torch.cuda.current_stream(self.device))
            cp_ev_end.synchronize()
            cp.record_layer_summary(step_id, int(self.layer_idx), {"expert_stage_total_ms": float(cp_ev_start.elapsed_time(cp_ev_end))})
        elif cp.should_sample(step_id):
            cp.record_component(step_id, int(self.layer_idx), "expert_stage_total", cp_exec_start, cp_exec_end)
        self._record_expert_exec_ms((time.perf_counter() - t_exec) * 1000.0)
        gathered = torch.empty_like(send_hidden)
        cp_gather_start = _hclm_cp_now_ns()
        self._all_to_all_single_timed(gathered, local_out, send_counts_list, recv_counts_list)
        cp_gather_end = _hclm_cp_now_ns()
        if cp.should_sample(step_id):
            cp.record_component(step_id, int(self.layer_idx), "gather_comm", cp_gather_start, cp_gather_end)
        unsorted = torch.empty_like(gathered)
        unsorted[sort_idx] = gathered
        return combine_expert_outputs(unsorted.view(T, K, H), idx)

    def _experts_ep_group(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.topology is None or self.comm_group is None:
            return self._experts_ep(x, idx, w)

        t_grouping = time.perf_counter()
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        expanded = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, H)
        flat_i = idx.reshape(-1).to(torch.long)
        flat_w = w.reshape(-1).to(expanded.dtype)
        dest = self._resolve_dest_for_flat_experts(flat_i, reason="ep_group_dispatch")
        sort_idx = torch.argsort(dest)
        send_hidden = expanded[sort_idx].contiguous()
        send_i = flat_i[sort_idx].contiguous()
        send_w = flat_w[sort_idx].contiguous()
        send_dest = dest[sort_idx].contiguous()
        send_counts = torch.bincount(send_dest, minlength=self.comm_world).to(dtype=torch.long)
        recv_counts = torch.empty_like(send_counts)
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        self._all_to_all_single_timed(recv_counts, send_counts, group=self.comm_group)

        send_counts_list = send_counts.cpu().tolist()
        recv_counts_list = recv_counts.cpu().tolist()
        t_grouping = time.perf_counter()
        recv_total = int(sum(recv_counts_list))

        fused_weight_dispatch = _ep_fused_weight_dispatch_enabled()
        if fused_weight_dispatch:
            send_hidden_w = torch.cat([send_hidden, send_w.unsqueeze(1)], dim=1).contiguous()
            recv_hidden_w = torch.empty((recv_total, H + 1), dtype=send_hidden.dtype, device=self.device)
            recv_i = torch.empty((recv_total,), dtype=send_i.dtype, device=self.device)
        else:
            recv_hidden = torch.empty((recv_total, H), dtype=send_hidden.dtype, device=self.device)
            recv_i = torch.empty((recv_total,), dtype=send_i.dtype, device=self.device)
            recv_w = torch.empty((recv_total,), dtype=send_w.dtype, device=self.device)

        counts_bytes = tensor_nbytes(send_counts) + tensor_nbytes(recv_counts)
        dispatch_bytes = (
            tensor_nbytes(send_hidden_w) + tensor_nbytes(send_i)
            if fused_weight_dispatch
            else tensor_nbytes(send_hidden) + tensor_nbytes(send_i) + tensor_nbytes(send_w)
        )
        gather_bytes = tensor_nbytes(send_hidden)
        self._record_activation_comm_bytes(counts_bytes + dispatch_bytes + gather_bytes)
        self._record_dispatch_self_remote(send_dest)
        remote_item_count = int((send_dest != int(self.comm_rank)).sum().item())
        if fused_weight_dispatch:
            remote_dispatch_bytes = remote_item_count * (
                int((H + 1) * send_hidden_w.element_size()) + int(send_i.element_size())
            )
        else:
            remote_dispatch_bytes = remote_item_count * (
                int(H * send_hidden.element_size()) + int(send_i.element_size()) + int(send_w.element_size())
            )
        remote_gather_bytes = remote_item_count * int(H * send_hidden.element_size())
        self._record_activation_remote_bytes(remote_dispatch_bytes + remote_gather_bytes)
        weaknet_item_bytes = int(H * send_hidden.element_size())
        if not fused_weight_dispatch:
            weaknet_item_bytes += int(send_w.element_size())
        else:
            weaknet_item_bytes += int(send_hidden_w.element_size())
        if str(os.environ.get("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1")).strip().lower() not in ("0", "false", "no", "off"):
            weaknet_item_bytes += int(send_i.element_size())
        if str(os.environ.get("SPARSE_LOGICAL_SIM_GATHER", "1")).strip().lower() not in ("0", "false", "no", "off"):
            weaknet_item_bytes += int(H * send_hidden.element_size())
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        weaknet_payload = self._maybe_charge_logical_ngd_from_ep_counts(
            send_counts_list=send_counts_list,
            hidden_dim=H,
            top_k=K,
            dtype_bytes=int(send_hidden.element_size()),
            rank_count_dtype_bytes=int(send_i.element_size()),
        )
        if weaknet_payload is None:
            weaknet_payload = _maybe_sleep_logical_weaknet(
                comm_global_ranks=self.comm_global_ranks,
                send_counts=send_counts,
                payload_bytes_per_item=weaknet_item_bytes,
                logical_transfer_multiplier=1,
                metrics=self.metrics,
                layer_idx=self.layer_idx,
                label="ep_group_dispatch",
            )
        cp = _hclm_cp_profiler()
        step_id = int(getattr(SparseContext, "runtime_call_id", -1))
        if cp.should_sample(step_id) and cp.record_modeled_comm:
            inter_events = float(weaknet_payload.get("inter_node_transfers", 0.0) or 0.0)
            inter_bytes = float(weaknet_payload.get("inter_node_bytes", 0.0) or 0.0)
            cp.record_layer_summary(
                step_id,
                int(self.layer_idx),
                {
                    "inter_node_events": inter_events,
                    "inter_node_bytes": inter_bytes,
                    **_hclm_cp_modeled_comm_ms(
                        inter_events,
                        inter_bytes,
                        bw_mbps=float(cp.bw_mbps),
                        lat_ms=float(cp.lat_ms),
                    ),
                    "notes": "modeled_dispatch_comm_ms includes normal EP logical dispatch payload and optional gather proxy per SPARSE_LOGICAL_SIM_GATHER.",
                },
            )
        t_grouping = time.perf_counter()

        if self.debug_a2a_checks:
            if fused_weight_dispatch and send_hidden_w.shape[0] != send_i.shape[0]:
                raise RuntimeError(
                    f"[rank{self.rank}] EP-group A2A shape mismatch: hidden_w={send_hidden_w.shape}, ids={send_i.shape}"
                )
            if (not fused_weight_dispatch) and (send_hidden.shape[0] != send_i.shape[0] or send_i.shape != send_w.shape):
                raise RuntimeError(
                    f"[rank{self.rank}] EP-group A2A shape mismatch: hidden={send_hidden.shape}, ids={send_i.shape}, weights={send_w.shape}"
                )

        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        cp_dispatch_start = _hclm_cp_now_ns()
        if fused_weight_dispatch:
            self._all_to_all_single_timed(
                recv_hidden_w,
                send_hidden_w,
                recv_counts_list,
                send_counts_list,
                group=self.comm_group,
            )
            self._all_to_all_single_timed(
                recv_i,
                send_i,
                recv_counts_list,
                send_counts_list,
                group=self.comm_group,
            )
        else:
            self._all_to_all_single_timed(
                recv_hidden,
                send_hidden,
                recv_counts_list,
                send_counts_list,
                group=self.comm_group,
            )
            self._all_to_all_single_timed(
                recv_i,
                send_i,
                recv_counts_list,
                send_counts_list,
                group=self.comm_group,
            )
            self._all_to_all_single_timed(
                recv_w,
                send_w,
                recv_counts_list,
                send_counts_list,
                group=self.comm_group,
            )
        cp_dispatch_end = _hclm_cp_now_ns()
        if cp.should_sample(step_id):
            cp.record_component(step_id, int(self.layer_idx), "dispatch_comm", cp_dispatch_start, cp_dispatch_end)

        if fused_weight_dispatch:
            recv_hidden = recv_hidden_w[:, :H].contiguous()
            recv_w = recv_hidden_w[:, H].contiguous()

        self._record_local_expert_load(recv_i)
        t_exec = time.perf_counter()
        cp_exec_start = _hclm_cp_now_ns()
        cp_use_cuda_timing = bool(cp.should_sample(step_id) and torch.cuda.is_available() and self.device.type == "cuda")
        if cp_use_cuda_timing:
            cp_ev_start = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_end = torch.cuda.Event(enable_timing=True, blocking=False)
            cp_ev_start.record(torch.cuda.current_stream(self.device))
        local_out = self._execute_local_recv_groups(recv_hidden=recv_hidden, recv_i=recv_i, recv_w=recv_w)
        cp_exec_end = _hclm_cp_now_ns()
        if cp_use_cuda_timing:
            cp_ev_end.record(torch.cuda.current_stream(self.device))
            cp_ev_end.synchronize()
            cp.record_layer_summary(step_id, int(self.layer_idx), {"expert_stage_total_ms": float(cp_ev_start.elapsed_time(cp_ev_end))})
        elif cp.should_sample(step_id):
            cp.record_component(step_id, int(self.layer_idx), "expert_stage_total", cp_exec_start, cp_exec_end)
        self._record_expert_exec_ms((time.perf_counter() - t_exec) * 1000.0)

        gathered = torch.empty_like(send_hidden)
        cp_gather_start = _hclm_cp_now_ns()
        self._all_to_all_single_timed(
            gathered,
            local_out,
            send_counts_list,
            recv_counts_list,
            group=self.comm_group,
        )
        cp_gather_end = _hclm_cp_now_ns()
        if cp.should_sample(step_id):
            cp.record_component(step_id, int(self.layer_idx), "gather_comm", cp_gather_start, cp_gather_end)
        unsorted = torch.empty_like(gathered)
        unsorted[sort_idx] = gathered
        return combine_expert_outputs(unsorted.view(T, K, H), idx)

    def _experts_ep_group_rank_grouped(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Rank-grouped EP dispatch.

        Baseline EP sends one hidden vector per token-expert pair. With top-k=4,
        a token whose four experts live on only two ranks still sends four hidden
        copies. This path sends one hidden vector per (token, destination rank)
        and carries the list of local experts/weights as metadata. The receiver
        executes all experts in the packet and returns one weighted partial output.

        This is the runtime hook that lets SCP co-location reduce activation
        dispatch/gather payload. It is disabled by default and intended for
        ablation with SPARSE_RANK_GROUPED_DISPATCH=1.
        """
        if self.topology is None or self.comm_group is None:
            return self._experts_ep(x, idx, w)

        t_grouping = time.perf_counter()
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        hidden_dim = int(H)
        if T == 0 or K == 0:
            return torch.zeros_like(x)

        flat_i = idx.reshape(-1).to(torch.long)
        dest_mat = self._resolve_dest_for_flat_experts(flat_i, reason="rank_grouped_dispatch", count_metrics=False).view(T, K)

        # Decode is the default target for this path, so the CPU grouping cost is
        # tiny. Keep prefill disabled unless explicitly requested.
        dest_cpu = dest_mat.detach().to("cpu", dtype=torch.long).tolist()
        idx_cpu = idx.detach().to("cpu", dtype=torch.long).tolist()
        w_cpu = w.detach().to("cpu", dtype=torch.float32).tolist()

        packet_dest_vals: List[int] = []
        packet_token_vals: List[int] = []
        packet_expert_rows: List[List[int]] = []
        packet_weight_rows: List[List[float]] = []

        for t in range(T):
            by_rank: Dict[int, List[int]] = {}
            for j in range(K):
                r = int(dest_cpu[t][j])
                by_rank.setdefault(r, []).append(j)
            for r, slots in by_rank.items():
                e_row = [-1 for _ in range(K)]
                w_row = [0.0 for _ in range(K)]
                for out_j, src_j in enumerate(slots[:K]):
                    e_row[out_j] = int(idx_cpu[t][src_j])
                    w_row[out_j] = float(w_cpu[t][src_j])
                packet_dest_vals.append(int(r))
                packet_token_vals.append(int(t))
                packet_expert_rows.append(e_row)
                packet_weight_rows.append(w_row)

        packet_count = len(packet_dest_vals)
        if packet_count == 0:
            return torch.zeros_like(x)

        send_dest = torch.tensor(packet_dest_vals, device=self.device, dtype=torch.long)
        send_token_ids = torch.tensor(packet_token_vals, device=self.device, dtype=torch.long)
        send_i = torch.tensor(packet_expert_rows, device=self.device, dtype=torch.long)
        send_w = torch.tensor(packet_weight_rows, device=self.device, dtype=x.dtype)

        sort_idx = torch.argsort(send_dest)
        send_dest = send_dest[sort_idx].contiguous()
        send_token_ids = send_token_ids[sort_idx].contiguous()
        send_i = send_i[sort_idx].contiguous()
        send_w = send_w[sort_idx].contiguous()
        send_hidden = x.index_select(0, send_token_ids).contiguous()

        send_counts = torch.bincount(send_dest, minlength=self.comm_world).to(dtype=torch.long)
        recv_counts = torch.empty_like(send_counts)
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        self._all_to_all_single_timed(recv_counts, send_counts, group=self.comm_group)

        send_counts_list = send_counts.cpu().tolist()
        recv_counts_list = recv_counts.cpu().tolist()
        t_grouping = time.perf_counter()
        recv_total = int(sum(recv_counts_list))

        recv_hidden = torch.empty((recv_total, H), dtype=send_hidden.dtype, device=self.device)
        recv_i_flat = torch.empty((recv_total * K,), dtype=send_i.dtype, device=self.device)
        recv_w_flat = torch.empty((recv_total * K,), dtype=send_w.dtype, device=self.device)

        id_send_counts = [int(c) * K for c in send_counts_list]
        id_recv_counts = [int(c) * K for c in recv_counts_list]

        counts_bytes = tensor_nbytes(send_counts) + tensor_nbytes(recv_counts)
        dispatch_bytes = tensor_nbytes(send_hidden) + tensor_nbytes(send_i) + tensor_nbytes(send_w)
        gather_bytes = tensor_nbytes(send_hidden)
        self._record_activation_comm_bytes(counts_bytes + dispatch_bytes + gather_bytes)
        self._record_dispatch_self_remote(send_dest)

        if self.metrics is not None:
            self.metrics.rank_grouped_pair_entries += int(T * K)
            self.metrics.rank_grouped_packet_entries += int(packet_count)
            self.metrics.rank_grouped_dispatch_calls += 1

        remote_packet_count = int((send_dest != int(self.comm_rank)).sum().item())
        remote_dispatch_bytes = remote_packet_count * (
            int(H * send_hidden.element_size())
            + int(K * send_i.element_size())
            + int(K * send_w.element_size())
        )
        remote_gather_bytes = remote_packet_count * int(H * send_hidden.element_size())
        self._record_activation_remote_bytes(remote_dispatch_bytes + remote_gather_bytes)

        weaknet_item_bytes = int(H * send_hidden.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            weaknet_item_bytes += int(K * send_i.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            weaknet_item_bytes += int(H * send_hidden.element_size())
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        _maybe_sleep_logical_weaknet(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=send_counts,
            payload_bytes_per_item=weaknet_item_bytes,
            metrics=self.metrics,
            layer_idx=self.layer_idx,
            label="rank_grouped_dispatch",
        )
        t_grouping = time.perf_counter()

        if self.debug_a2a_checks:
            if send_hidden.shape[0] != send_i.shape[0] or send_i.shape != send_w.shape:
                raise RuntimeError(
                    f"[rank{self.rank}] rank-grouped A2A shape mismatch: "
                    f"hidden={send_hidden.shape}, ids={send_i.shape}, weights={send_w.shape}"
                )

        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        self._all_to_all_single_timed(
            recv_hidden,
            send_hidden,
            recv_counts_list,
            send_counts_list,
            group=self.comm_group,
        )
        self._all_to_all_single_timed(
            recv_i_flat,
            send_i.reshape(-1).contiguous(),
            id_recv_counts,
            id_send_counts,
            group=self.comm_group,
        )
        self._all_to_all_single_timed(
            recv_w_flat,
            send_w.reshape(-1).contiguous(),
            id_recv_counts,
            id_send_counts,
            group=self.comm_group,
        )

        recv_i = recv_i_flat.view(recv_total, K).contiguous()
        recv_w = recv_w_flat.view(recv_total, K).contiguous()
        valid_recv_i = recv_i[recv_i.ge(0)]
        self._record_local_expert_load(valid_recv_i)
        t_exec = time.perf_counter()
        local_out = self._execute_local_rank_grouped_packets(
            recv_hidden=recv_hidden,
            recv_i=recv_i,
            recv_w=recv_w,
        )
        self._record_expert_exec_ms((time.perf_counter() - t_exec) * 1000.0)

        gathered = torch.empty_like(send_hidden)
        self._all_to_all_single_timed(
            gathered,
            local_out,
            send_counts_list,
            recv_counts_list,
            group=self.comm_group,
        )

        final = torch.zeros((T, H), device=x.device, dtype=x.dtype)
        final.index_add_(0, send_token_ids, gathered)
        return final

    @_DYNAMO_DISABLE
    def _experts_ep_group_node_grouped(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
        seq_len: int = 1,
    ) -> torch.Tensor:
        """
        Hierarchical node-grouped EP dispatch.

        Stage 1 sends one packet per (token, destination logical node). This is
        the cross-node aggregation point: experts co-located on the same logical
        node share one hidden-state transfer.

        Stage 2 runs inside the destination node by splitting each node packet
        into one packet per real destination rank. Expert ownership and router
        semantics are unchanged: every expert is still executed on its assigned
        rank. The extra stage is the price paid for reducing weak-network
        transfer events.
        """
        if self.topology is None or self.comm_group is None:
            return self._experts_ep(x, idx, w)

        if _env_flag("SPARSE_NGD_LOGICAL_FASTPATH", "0"):
            return self._experts_ep_group_node_grouped_logical_fastpath(x, idx, w, seq_len=seq_len)

        t_grouping = time.perf_counter()
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        hidden_dim = int(H)
        if T == 0 or K == 0:
            return torch.zeros_like(x)

        fastpath_mode = self._node_grouped_fastpath_mode(seq_len)
        mode_lean = fastpath_mode in ("lean", "lean_tensor", "lean_tensor_stage2local")
        mode_tensor = fastpath_mode in ("tensor", "tensor_pack", "phase_tensor") or bool(mode_lean)
        mode_triton = fastpath_mode in ("triton", "triton_pack")
        mode_lean_fused = bool(mode_lean) and _env_flag("SPARSE_NGD_LEAN_TRITON_PACK", "1")
        mode_same = fastpath_mode in (
            "buildtime_same_node",
            "buildtime_same_node_packmeta",
            "buildtime_same_node_stage2local",
            "fused_gather_combine",
            "phase_adaptive",
        ) or bool(mode_lean)
        mode_packmeta = fastpath_mode in (
            "buildtime_same_node_packmeta",
            "buildtime_same_node_stage2local",
            "fused_gather_combine",
            "phase_adaptive",
        ) or bool(mode_lean)
        mode_stage2local = fastpath_mode in (
            "buildtime_same_node_stage2local",
            "fused_gather_combine",
            "phase_adaptive",
        ) or bool(mode_lean)
        mode_identity_combine = fastpath_mode in ("fused_gather_combine", "phase_adaptive") or bool(mode_lean)
        use_workspace_reuse = bool(mode_lean) or _env_flag("SPARSE_NGD_WORKSPACE_REUSE", "0")
        use_stage_local_skip = bool(mode_lean) or _env_flag("SPARSE_NGD_STAGE_LOCAL_SKIP", "0")

        use_tensor_fastpath = self._node_grouped_tensor_fastpath_enabled(seq_len) or mode_tensor
        use_triton_pack = self._node_grouped_triton_pack_enabled(seq_len, idx, w) or (
            (bool(mode_triton) or bool(mode_lean_fused))
            and _node_grouped_triton is not None
            and _node_grouped_triton.can_use_node_grouped_triton(idx, w)
        )
        use_same_node_fastpath = (
            self._node_grouped_phase_flag("SPARSE_NODE_GROUPED_SAME_NODE_FASTPATH", seq_len, "0")
            or mode_same
        )
        use_packed_meta = (
            self._node_grouped_phase_flag("SPARSE_NODE_GROUPED_PACK_META", seq_len, "0")
            or mode_packmeta
        )
        use_stage2_local_rank_fastpath = self._node_grouped_phase_flag(
            "SPARSE_NODE_GROUPED_STAGE2_LOCAL_RANK_FASTPATH",
            seq_len,
            "0",
        ) or mode_stage2local
        use_identity_combine_fastpath = self._node_grouped_phase_flag(
            "SPARSE_NODE_GROUPED_IDENTITY_COMBINE_FASTPATH",
            seq_len,
            "0",
        )
        use_cpu_decode_pack = (
            int(seq_len) == 1
            and int(T * K) <= int(os.environ.get("SPARSE_NGD_CPU_DECODE_PACK_MAX_ITEMS", "64") or 64)
            and (
                _env_flag("SPARSE_NGD_CPU_DECODE_PACK", "0")
                or (bool(mode_lean) and _env_flag("SPARSE_NGD_LEAN_CPU_DECODE_PACK", "1"))
            )
        )
        use_node_rep_execute = (
            int(seq_len) == 1
            and self.expert_offload == "cpu"
            and (
                _env_flag("SPARSE_NGD_NODE_REP_EXECUTE", "0")
                or (bool(mode_lean) and _env_flag("SPARSE_NGD_LEAN_NODE_REP_EXECUTE", "0"))
            )
        )
        use_fixed_node_rep_a2a = (
            int(seq_len) == 1
            and self.expert_offload == "cpu"
            and (
                _env_flag("SPARSE_NGD_FIXED_NODE_REP_A2A", "0")
                or (bool(mode_lean) and _env_flag("SPARSE_NGD_LEAN_FIXED_NODE_REP_A2A", "0"))
            )
        )
        use_owner_packed_payload = (
            _env_flag("SPARSE_NGD_OWNER_PACKED_PAYLOAD", "0")
            or (bool(mode_lean) and _env_flag("SPARSE_NGD_LEAN_OWNER_PACKED_PAYLOAD", "0"))
        )
        stage2_local_group_info = self._stage2_local_group_info()
        use_stage2_node_local_group = stage2_local_group_info is not None
        # Packed metadata trades a little metadata bandwidth for fewer A2A calls.
        # Use fp32 by default so expert ids are not lossy when the activation
        # dtype is bf16/fp16; for Qwen-MoE the metadata is tiny relative to H.
        pack_meta_float32 = self._node_grouped_phase_flag("SPARSE_NODE_GROUPED_PACK_META_FLOAT32", seq_len, "1")
        pack_meta_dtype = (
            torch.float32
            if bool(pack_meta_float32)
            else x.dtype
        )
        owner_payload_dtype = self._node_grouped_owner_payload_dtype(x.dtype) if bool(use_owner_packed_payload) else None
        use_owner_packed_payload = bool(use_owner_packed_payload) and owner_payload_dtype is not None
        use_buildtime_same_node_fastpath = bool(use_same_node_fastpath) and not bool(use_tensor_fastpath) and not bool(use_triton_pack)

        if bool(use_fixed_node_rep_a2a):
            fixed_out = self._experts_ep_group_node_grouped_decode_fixed_node_rep(
                x,
                idx,
                w,
                seq_len=seq_len,
            )
            if fixed_out is not None:
                return fixed_out
            self._record_node_grouped_value("fallback_layers", 1)

        cpu_packed = None
        if use_cpu_decode_pack:
            cpu_packed = self._pack_node_packets_cpu_decode(
                x,
                idx,
                w,
                max_items=int(os.environ.get("SPARSE_NGD_CPU_DECODE_PACK_MAX_ITEMS", "64") or 64),
            )
        if cpu_packed is not None:
            send_node_dest, send_token_ids, send_node_i, send_node_w = cpu_packed
            node_packet_count = int(send_node_dest.numel())
        elif use_triton_pack:
            try:
                rank_to_node = self._rank_to_node_tensor()
                node_ids = self._comm_rank_node_ids()
                node_count = max([0] + [int(q) for q in node_ids]) + 1
                send_node_dest, send_token_ids, send_node_i, send_node_w = _node_grouped_triton.pack_node_packets_triton(
                    idx.to(torch.long).contiguous(),
                    w.to(x.dtype).contiguous(),
                    self.expert_to_rank,
                    rank_to_node,
                    self._node_representative_tensor(),
                    num_nodes=int(node_count),
                )
                node_packet_count = int(send_node_dest.numel())
            except Exception:
                if self._node_grouped_triton_strict():
                    raise
                send_node_dest, send_token_ids, send_node_i, send_node_w = self._pack_node_packets_tensorized(x, idx, w)
                node_packet_count = int(send_node_dest.numel())
        elif use_tensor_fastpath:
            send_node_dest, send_token_ids, send_node_i, send_node_w = self._pack_node_packets_tensorized(x, idx, w)
            node_packet_count = int(send_node_dest.numel())
        else:
            node_ids = self._comm_rank_node_ids()
            my_node = int(node_ids[int(self.comm_rank)]) if 0 <= int(self.comm_rank) < len(node_ids) else 0
            flat_i = idx.reshape(-1).to(torch.long)
            dest_rank_mat = self._resolve_dest_for_flat_experts(flat_i, reason="node_grouped_pack_cpu", count_metrics=False).view(T, K)
            dest_node_mat = torch.empty_like(dest_rank_mat)
            for cr, q in enumerate(node_ids):
                dest_node_mat[dest_rank_mat == int(cr)] = int(q)
            dest_node_cpu = dest_node_mat.detach().to("cpu", dtype=torch.long).tolist()
            idx_cpu = idx.detach().to("cpu", dtype=torch.long).tolist()
            w_cpu = w.detach().to("cpu", dtype=torch.float32).tolist()

            node_packet_dest_vals: List[int] = []
            node_packet_token_vals: List[int] = []
            node_packet_expert_rows: List[List[int]] = []
            node_packet_weight_rows: List[List[float]] = []
            local_node_packet_token_vals: List[int] = []
            local_node_packet_expert_rows: List[List[int]] = []
            local_node_packet_weight_rows: List[List[float]] = []

            for t in range(T):
                by_node: Dict[int, List[int]] = {}
                for j in range(K):
                    if float(w_cpu[t][j]) == 0.0:
                        continue
                    q = int(dest_node_cpu[t][j])
                    by_node.setdefault(q, []).append(j)
                for q, slots in by_node.items():
                    e_row = [-1 for _ in range(K)]
                    w_row = [0.0 for _ in range(K)]
                    for out_j, src_j in enumerate(slots[:K]):
                        e_row[out_j] = int(idx_cpu[t][src_j])
                        w_row[out_j] = float(w_cpu[t][src_j])
                    if bool(use_same_node_fastpath) and int(q) == int(my_node):
                        local_node_packet_token_vals.append(int(t))
                        local_node_packet_expert_rows.append(e_row)
                        local_node_packet_weight_rows.append(w_row)
                    else:
                        node_packet_dest_vals.append(self._node_representative_comm_rank(int(q)))
                        node_packet_token_vals.append(int(t))
                        node_packet_expert_rows.append(e_row)
                        node_packet_weight_rows.append(w_row)

            cross_node_packet_count = len(node_packet_dest_vals)
            local_node_packet_count = len(local_node_packet_token_vals)
            node_packet_count = int(cross_node_packet_count + local_node_packet_count)
            if cross_node_packet_count:
                send_node_dest = torch.tensor(node_packet_dest_vals, device=self.device, dtype=torch.long)
                send_token_ids = torch.tensor(node_packet_token_vals, device=self.device, dtype=torch.long)
                send_node_i = torch.tensor(node_packet_expert_rows, device=self.device, dtype=torch.long)
                send_node_w = torch.tensor(node_packet_weight_rows, device=self.device, dtype=x.dtype)
            else:
                send_node_dest = torch.empty((0,), device=self.device, dtype=torch.long)
                send_token_ids = torch.empty((0,), device=self.device, dtype=torch.long)
                send_node_i = torch.empty((0, K), device=self.device, dtype=torch.long)
                send_node_w = torch.empty((0, K), device=self.device, dtype=x.dtype)
            if local_node_packet_count:
                local_token_ids = torch.tensor(local_node_packet_token_vals, device=self.device, dtype=torch.long)
                local_node_i = torch.tensor(local_node_packet_expert_rows, device=self.device, dtype=torch.long)
                local_node_w = torch.tensor(local_node_packet_weight_rows, device=self.device, dtype=x.dtype)
                local_node_hidden = x.index_select(0, local_token_ids).contiguous()
            else:
                local_token_ids = torch.empty((0,), device=self.device, dtype=torch.long)
                local_node_hidden = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
                local_node_i = torch.empty((0, K), dtype=torch.long, device=self.device)
                local_node_w = torch.empty((0, K), dtype=x.dtype, device=self.device)
            self._record_node_grouped_allocation(
                send_node_dest,
                send_token_ids,
                send_node_i,
                send_node_w,
                local_token_ids,
                local_node_hidden,
                local_node_i,
                local_node_w,
            )
            self._record_node_grouped_value("same_node_packet_entries", int(local_node_packet_count))
            self._record_node_grouped_value("cross_node_packet_entries", int(cross_node_packet_count))
            self._record_node_grouped_value(
                "same_node_bypass_bytes",
                int(local_node_packet_count) * int(hidden_dim * x.element_size()),
            )
        if node_packet_count == 0:
            return torch.zeros_like(x)

        if send_node_dest.numel() > 0:
            sort_idx = torch.argsort(send_node_dest)
            send_node_dest = send_node_dest[sort_idx].contiguous()
            send_token_ids = send_token_ids[sort_idx].contiguous()
            send_node_i = send_node_i[sort_idx].contiguous()
            send_node_w = send_node_w[sort_idx].contiguous()
            send_node_hidden = x.index_select(0, send_token_ids).contiguous()
        else:
            send_node_hidden = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
        if hidden_dim <= 0:
            hidden_dim = int(self._hidden_size_from_modules())
        if hidden_dim <= 0:
            hidden_dim = int(self._hidden_size_from_modules())
        if hidden_dim <= 0:
            # Last-resort fallback for unusual model wrappers. Keep this out of
            # tensor allocation paths whenever the module exposes a static size.
            hidden_dim = int(send_node_hidden.detach().cpu().shape[-1])

        if bool(use_same_node_fastpath) and not bool(use_buildtime_same_node_fastpath):
            local_node_mask = send_node_dest.eq(int(self.comm_rank))
            cross_node_mask = ~local_node_mask
            local_token_ids = send_token_ids[local_node_mask].contiguous()
            local_node_hidden = send_node_hidden[local_node_mask].contiguous()
            local_node_i = send_node_i[local_node_mask].contiguous()
            local_node_w = send_node_w[local_node_mask].contiguous()
            send_token_ids_cross = send_token_ids[cross_node_mask].contiguous()
            send_node_dest_comm = send_node_dest[cross_node_mask].contiguous()
            send_node_hidden_comm = send_node_hidden[cross_node_mask].contiguous()
            send_node_i_comm = send_node_i[cross_node_mask].contiguous()
            send_node_w_comm = send_node_w[cross_node_mask].contiguous()
            self._record_node_grouped_value("same_node_packet_entries", int(local_token_ids.numel()))
            self._record_node_grouped_value("cross_node_packet_entries", int(send_token_ids_cross.numel()))
            self._record_node_grouped_value(
                "same_node_bypass_bytes",
                int(local_token_ids.numel()) * int(hidden_dim * x.element_size()),
            )
        elif bool(use_buildtime_same_node_fastpath):
            send_token_ids_cross = send_token_ids
            send_node_dest_comm = send_node_dest
            send_node_hidden_comm = send_node_hidden
            send_node_i_comm = send_node_i
            send_node_w_comm = send_node_w
        else:
            local_token_ids = torch.empty((0,), device=self.device, dtype=torch.long)
            local_node_hidden = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
            local_node_i = torch.empty((0, K), dtype=torch.long, device=self.device)
            local_node_w = torch.empty((0, K), dtype=x.dtype, device=self.device)
            send_token_ids_cross = send_token_ids
            send_node_dest_comm = send_node_dest
            send_node_hidden_comm = send_node_hidden
            send_node_i_comm = send_node_i
            send_node_w_comm = send_node_w

        t_counts_build = time.perf_counter()
        send_counts = torch.bincount(send_node_dest_comm, minlength=self.comm_world).to(dtype=torch.long)
        self._record_node_grouped_value("counts_build_ms", (time.perf_counter() - t_counts_build) * 1000.0)
        dt_ms = (time.perf_counter() - t_grouping) * 1000.0
        self._record_grouping_overhead_ms(dt_ms)
        self._record_node_grouped_ms("pack", dt_ms)
        t_meta = time.perf_counter()
        self._record_node_grouped_value("metadata_collective_count", 1)
        recv_counts = self._exchange_dispatch_counts(
            send_counts,
            group=self.comm_group,
            group_rank=self.comm_rank,
            label="node_grouped_stage1",
        )
        stage1_global_total = int(getattr(self, "_last_dispatch_count_global_total", 0))
        dt_meta = (time.perf_counter() - t_meta) * 1000.0
        self._record_node_grouped_ms("metadata", dt_meta)
        self._record_node_grouped_value("metadata_exchange_ms", dt_meta)

        t_splits_build = time.perf_counter()
        send_counts_list = [int(c) for c in send_counts.detach().cpu().tolist()]
        recv_counts_list = [int(c) for c in recv_counts.detach().cpu().tolist()]
        recv_total = int(sum(recv_counts_list))
        id_send_counts = [int(c) * K for c in send_counts_list]
        id_recv_counts = [int(c) * K for c in recv_counts_list]
        hidden_send_counts = [int(c) * hidden_dim for c in send_counts_list]
        hidden_recv_counts = [int(c) * hidden_dim for c in recv_counts_list]
        self._record_node_grouped_value("splits_build_ms", (time.perf_counter() - t_splits_build) * 1000.0)
        t_grouping = time.perf_counter()
        counts_bytes = tensor_nbytes(send_counts) + tensor_nbytes(recv_counts)
        if use_owner_packed_payload:
            stage1_payload_width = int(hidden_dim) + 2 * int(K)
            send_node_payload = self._pack_owner_grouped_payload(
                send_node_hidden_comm,
                send_node_i_comm,
                send_node_w_comm,
                payload_dtype=owner_payload_dtype,
            )
            recv_node_payload_flat = self._node_grouped_empty(
                (int(recv_total) * int(stage1_payload_width),),
                dtype=owner_payload_dtype,
                label="stage1_owner_payload_recv_flat",
                workspace=use_workspace_reuse,
            )
            stage1_meta_bytes = max(0, int(tensor_nbytes(send_node_payload)) - int(tensor_nbytes(send_node_hidden_comm)))
        else:
            recv_node_hidden_flat = self._node_grouped_empty(
                (int(recv_total) * int(hidden_dim),),
                dtype=send_node_hidden.dtype,
                label="stage1_hidden_recv_flat",
                workspace=use_workspace_reuse,
            )
        if use_owner_packed_payload:
            pass
        elif use_packed_meta:
            send_node_meta_comm = torch.cat(
                [send_node_i_comm.to(pack_meta_dtype), send_node_w_comm.to(pack_meta_dtype)],
                dim=1,
            ).contiguous()
            recv_node_meta_flat = self._node_grouped_empty(
                (int(recv_total) * 2 * int(K),),
                dtype=pack_meta_dtype,
                label="stage1_meta_recv_flat",
                workspace=use_workspace_reuse,
            )
            meta_send_counts = [int(c) * 2 * K for c in send_counts_list]
            meta_recv_counts = [int(c) * 2 * K for c in recv_counts_list]
            stage1_meta_bytes = tensor_nbytes(send_node_meta_comm)
            self._record_node_grouped_value("metadata_collective_count", 1)
        else:
            recv_node_i_flat = self._node_grouped_empty(
                (int(recv_total) * int(K),),
                dtype=send_node_i.dtype,
                label="stage1_id_recv_flat",
                workspace=use_workspace_reuse,
            )
            recv_node_w_flat = self._node_grouped_empty(
                (int(recv_total) * int(K),),
                dtype=send_node_w.dtype,
                label="stage1_weight_recv_flat",
                workspace=use_workspace_reuse,
            )
            stage1_meta_bytes = tensor_nbytes(send_node_i_comm) + tensor_nbytes(send_node_w_comm)
            self._record_node_grouped_value("metadata_collective_count", 2)
        if use_owner_packed_payload:
            self._record_node_grouped_allocation(recv_node_payload_flat, send_node_payload)
            stage1_dispatch_bytes = tensor_nbytes(send_node_payload)
            stage1_dispatch_item_bytes = int(stage1_payload_width) * int(send_node_payload.element_size())
        else:
            self._record_node_grouped_allocation(recv_node_hidden_flat)
            stage1_dispatch_bytes = tensor_nbytes(send_node_hidden_comm) + stage1_meta_bytes
            stage1_dispatch_item_bytes = int(hidden_dim * send_node_hidden_comm.element_size()) + (
                int(2 * K * send_node_w_comm.element_size())
                if use_packed_meta
                else int(K * send_node_i_comm.element_size()) + int(K * send_node_w_comm.element_size())
            )
        stage1_gather_bytes = tensor_nbytes(send_node_hidden_comm)
        self._record_node_grouped_value("metadata_bytes", int(counts_bytes + stage1_meta_bytes))
        self._record_activation_comm_bytes(counts_bytes + stage1_dispatch_bytes + stage1_gather_bytes)
        self._record_dispatch_self_remote(send_node_dest_comm)

        remote_node_packet_count = int((send_node_dest_comm != int(self.comm_rank)).sum().item())
        remote_node_bytes = remote_node_packet_count * (
            int(stage1_dispatch_item_bytes)
            + int(hidden_dim * send_node_hidden_comm.element_size())
        )
        self._record_activation_remote_bytes(remote_node_bytes)

        node_payload_bytes = int(hidden_dim * send_node_hidden_comm.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            if use_owner_packed_payload:
                node_payload_bytes = int(stage1_dispatch_item_bytes)
            elif use_packed_meta:
                node_payload_bytes += int(2 * K * send_node_w_comm.element_size())
            else:
                node_payload_bytes += int(K * send_node_i_comm.element_size()) + int(K * send_node_w_comm.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            node_payload_bytes += int(hidden_dim * send_node_hidden_comm.element_size())
        dt_ms = (time.perf_counter() - t_grouping) * 1000.0
        self._record_grouping_overhead_ms(dt_ms)
        self._record_node_grouped_ms("pack", dt_ms)
        t_inter = time.perf_counter()
        weaknet_payload = _maybe_sleep_logical_weaknet(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=send_counts,
            payload_bytes_per_item=node_payload_bytes,
            metrics=self.metrics,
            layer_idx=self.layer_idx,
            label="node_grouped_stage1",
        )
        if _logical_weaknet_enabled():
            stage1_sleep_ms = float(weaknet_payload.get("sim_ms", 0.0))
            self._record_node_grouped_value("inter_send_stage1_weaknet_sleep_ms", stage1_sleep_ms)
            self._record_node_grouped_value("inter_send_weaknet_sleep_ms", stage1_sleep_ms)
        t_grouping = time.perf_counter()
        t_stage1_collective = time.perf_counter()

        if bool(use_stage_local_skip) and int(stage1_global_total) == 0:
            recv_node_hidden_comm = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
            recv_node_i_comm = torch.empty((0, K), dtype=torch.long, device=self.device)
            recv_node_w_comm = torch.empty((0, K), dtype=x.dtype, device=self.device)
            self._record_node_grouped_value("stage1_local_fastpath_skip", 1)
        else:
            if use_owner_packed_payload:
                recv_node_payload_flat = self._all_to_all_flat_with_dummy_timed(
                    send_node_payload.reshape(-1).contiguous(),
                    output_counts=recv_counts_list,
                    input_counts=send_counts_list,
                    item_width=stage1_payload_width,
                    group=self.comm_group,
                    group_rank=self.comm_rank,
                    label="node_grouped_stage1_owner_payload",
                    workspace=use_workspace_reuse,
                )
                recv_node_payload = recv_node_payload_flat.view(recv_total, stage1_payload_width).contiguous()
                recv_node_hidden_comm, recv_node_i_comm, recv_node_w_comm = self._unpack_owner_grouped_payload(
                    recv_node_payload,
                    hidden_dim=hidden_dim,
                    top_k=K,
                    activation_dtype=x.dtype,
                )
            else:
                recv_node_hidden_flat = self._all_to_all_flat_with_dummy_timed(
                    send_node_hidden_comm.reshape(-1).contiguous(),
                    output_counts=recv_counts_list,
                    input_counts=send_counts_list,
                    item_width=hidden_dim,
                    group=self.comm_group,
                    group_rank=self.comm_rank,
                    label="node_grouped_stage1_hidden",
                    workspace=use_workspace_reuse,
                )
                recv_node_hidden_comm = recv_node_hidden_flat.view(-1, hidden_dim).contiguous()
            if use_owner_packed_payload:
                pass
            elif use_packed_meta:
                recv_node_meta_flat = self._all_to_all_flat_with_dummy_timed(
                    send_node_meta_comm.reshape(-1).contiguous(),
                    output_counts=recv_counts_list,
                    input_counts=send_counts_list,
                    item_width=2 * K,
                    group=self.comm_group,
                    group_rank=self.comm_rank,
                    label="node_grouped_stage1_meta",
                    workspace=use_workspace_reuse,
                )
                recv_node_meta = recv_node_meta_flat.view(recv_total, 2 * K).contiguous()
                recv_node_i_comm = recv_node_meta[:, :K].to(torch.long).contiguous()
                recv_node_w_comm = recv_node_meta[:, K:].to(x.dtype).contiguous()
            else:
                recv_node_i_flat = self._all_to_all_flat_with_dummy_timed(
                    send_node_i_comm.reshape(-1).contiguous(),
                    output_counts=recv_counts_list,
                    input_counts=send_counts_list,
                    item_width=K,
                    group=self.comm_group,
                    group_rank=self.comm_rank,
                    label="node_grouped_stage1_ids",
                    workspace=use_workspace_reuse,
                )
                recv_node_w_flat = self._all_to_all_flat_with_dummy_timed(
                    send_node_w_comm.reshape(-1).contiguous(),
                    output_counts=recv_counts_list,
                    input_counts=send_counts_list,
                    item_width=K,
                    group=self.comm_group,
                    group_rank=self.comm_rank,
                    label="node_grouped_stage1_weights",
                    workspace=use_workspace_reuse,
                )
                recv_node_i_comm = recv_node_i_flat.view(recv_total, K).contiguous()
                recv_node_w_comm = recv_node_w_flat.view(recv_total, K).contiguous()
        self._record_node_grouped_value("inter_send_stage1_collective_ms", (time.perf_counter() - t_stage1_collective) * 1000.0)
        if local_node_hidden.numel() > 0:
            recv_node_hidden = torch.cat([recv_node_hidden_comm, local_node_hidden], dim=0).contiguous()
            recv_node_i = torch.cat([recv_node_i_comm, local_node_i], dim=0).contiguous()
            recv_node_w = torch.cat([recv_node_w_comm, local_node_w], dim=0).contiguous()
        else:
            recv_node_hidden = recv_node_hidden_comm
            recv_node_i = recv_node_i_comm
            recv_node_w = recv_node_w_comm
        recv_total_all = int(recv_node_hidden.shape[0])
        self._record_node_grouped_ms("inter_send", (time.perf_counter() - t_inter) * 1000.0)

        if bool(use_node_rep_execute):
            t_exec = time.perf_counter()
            self._record_local_expert_load(recv_node_i[recv_node_i.ge(0)])
            node_packet_out = self._execute_local_rank_grouped_packets(
                recv_hidden=recv_node_hidden,
                recv_i=recv_node_i,
                recv_w=recv_node_w,
                allow_nonlocal_cpu_once=True,
            )
            self._record_expert_exec_ms((time.perf_counter() - t_exec) * 1000.0)

            t_gather = time.perf_counter()
            final = torch.zeros((T, H), device=x.device, dtype=x.dtype)
            self._record_node_grouped_allocation(final)
            if bool(use_stage_local_skip) and int(stage1_global_total) == 0:
                gathered_node_out = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
            else:
                gathered_node_out_flat = self._all_to_all_flat_with_dummy_timed(
                    node_packet_out[:recv_total].contiguous().reshape(-1),
                    output_counts=send_counts_list,
                    input_counts=recv_counts_list,
                    item_width=hidden_dim,
                    group=self.comm_group,
                    group_rank=self.comm_rank,
                    label="node_grouped_stage1_gather",
                    workspace=use_workspace_reuse,
                )
                gathered_node_out = gathered_node_out_flat.view(-1, hidden_dim).contiguous()
            self._record_node_grouped_value("node_to_token_restore_ms", (time.perf_counter() - t_gather) * 1000.0)
            if send_token_ids_cross.numel() > 0:
                final.index_add_(0, send_token_ids_cross, gathered_node_out)
            if local_token_ids.numel() > 0:
                final.index_add_(0, local_token_ids, node_packet_out[recv_total:].contiguous())
            self._record_node_grouped_ms("gather", (time.perf_counter() - t_gather) * 1000.0)
            self._record_node_grouped_value("stage2_local_fastpath_skip", 1)
            return final

        # Stage 2: split received node packets to true destination ranks.
        if use_triton_pack:
            try:
                send_rank_dest, send_rank_packet_ids, send_rank_i, send_rank_w = _node_grouped_triton.pack_rank_packets_triton(
                    recv_node_i.to(torch.long).contiguous(),
                    recv_node_w.to(x.dtype).contiguous(),
                    self.expert_to_rank,
                    comm_world=int(self.comm_world),
                )
                rank_packet_count = int(send_rank_dest.numel())
            except Exception:
                if self._node_grouped_triton_strict():
                    raise
                send_rank_dest, send_rank_packet_ids, send_rank_i, send_rank_w = self._pack_rank_packets_tensorized(
                    recv_node_i,
                    recv_node_w,
                )
                rank_packet_count = int(send_rank_dest.numel())
        elif use_tensor_fastpath:
            send_rank_dest, send_rank_packet_ids, send_rank_i, send_rank_w = self._pack_rank_packets_tensorized(
                recv_node_i,
                recv_node_w,
            )
            rank_packet_count = int(send_rank_dest.numel())
        else:
            recv_i_cpu = recv_node_i.detach().to("cpu", dtype=torch.long).tolist()
            recv_w_cpu = recv_node_w.detach().to("cpu", dtype=torch.float32).tolist()
            rank_packet_dest_vals: List[int] = []
            rank_packet_node_packet_vals: List[int] = []
            rank_packet_expert_rows: List[List[int]] = []
            rank_packet_weight_rows: List[List[float]] = []
            local_rank_packet_node_packet_vals: List[int] = []
            local_rank_packet_expert_rows: List[List[int]] = []
            local_rank_packet_weight_rows: List[List[float]] = []

            # Same-node fastpath appends local-node packets after cross-node
            # receives. Split all node packets here; otherwise decode runs that
            # keep tensor fastpath off would silently drop local-node packets.
            for p in range(recv_total_all):
                by_rank: Dict[int, List[int]] = {}
                for j in range(K):
                    eid = int(recv_i_cpu[p][j])
                    if eid < 0:
                        continue
                    r = int(self.expert_to_rank[eid].item())
                    by_rank.setdefault(r, []).append(j)
                for r, slots in by_rank.items():
                    e_row = [-1 for _ in range(K)]
                    w_row = [0.0 for _ in range(K)]
                    for out_j, src_j in enumerate(slots[:K]):
                        e_row[out_j] = int(recv_i_cpu[p][src_j])
                        w_row[out_j] = float(recv_w_cpu[p][src_j])
                    if bool(use_stage2_local_rank_fastpath) and int(r) == int(self.comm_rank):
                        local_rank_packet_node_packet_vals.append(int(p))
                        local_rank_packet_expert_rows.append(e_row)
                        local_rank_packet_weight_rows.append(w_row)
                    else:
                        rank_packet_dest_vals.append(int(r))
                        rank_packet_node_packet_vals.append(int(p))
                        rank_packet_expert_rows.append(e_row)
                        rank_packet_weight_rows.append(w_row)

            remote_rank_packet_count_build = len(rank_packet_dest_vals)
            local_rank_packet_count_build = len(local_rank_packet_node_packet_vals)
            rank_packet_count = int(remote_rank_packet_count_build + local_rank_packet_count_build)
            send_rank_dest = torch.tensor(rank_packet_dest_vals, device=self.device, dtype=torch.long) if remote_rank_packet_count_build else torch.empty((0,), device=self.device, dtype=torch.long)
            send_rank_packet_ids = torch.tensor(rank_packet_node_packet_vals, device=self.device, dtype=torch.long) if remote_rank_packet_count_build else torch.empty((0,), device=self.device, dtype=torch.long)
            send_rank_i = torch.tensor(rank_packet_expert_rows, device=self.device, dtype=torch.long) if remote_rank_packet_count_build else torch.empty((0, K), device=self.device, dtype=torch.long)
            send_rank_w = torch.tensor(rank_packet_weight_rows, device=self.device, dtype=x.dtype) if remote_rank_packet_count_build else torch.empty((0, K), device=self.device, dtype=x.dtype)
            if local_rank_packet_count_build:
                local_rank_packet_ids_direct = torch.tensor(local_rank_packet_node_packet_vals, device=self.device, dtype=torch.long)
                local_rank_hidden_direct = recv_node_hidden.index_select(0, local_rank_packet_ids_direct).contiguous()
                local_rank_i_direct = torch.tensor(local_rank_packet_expert_rows, device=self.device, dtype=torch.long)
                local_rank_w_direct = torch.tensor(local_rank_packet_weight_rows, device=self.device, dtype=x.dtype)
            else:
                local_rank_packet_ids_direct = torch.empty((0,), device=self.device, dtype=torch.long)
                local_rank_hidden_direct = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
                local_rank_i_direct = torch.empty((0, K), dtype=torch.long, device=self.device)
                local_rank_w_direct = torch.empty((0, K), dtype=x.dtype, device=self.device)
            self._record_node_grouped_allocation(
                send_rank_dest,
                send_rank_packet_ids,
                send_rank_i,
                send_rank_w,
                local_rank_packet_ids_direct,
                local_rank_hidden_direct,
                local_rank_i_direct,
                local_rank_w_direct,
            )
            self._record_node_grouped_value("stage2_local_rank_packet_entries", int(local_rank_packet_count_build))
            self._record_node_grouped_value("stage2_remote_rank_packet_entries", int(remote_rank_packet_count_build))
            self._record_node_grouped_value(
                "stage2_local_bypass_bytes",
                int(local_rank_packet_count_build) * int(hidden_dim * x.element_size()),
            )

        use_buildtime_stage2_local_rank_fastpath = bool(use_stage2_local_rank_fastpath) and not bool(use_tensor_fastpath) and not bool(use_triton_pack)
        if send_rank_dest.numel() > 0:
            sort2 = torch.argsort(send_rank_dest)
            send_rank_dest = send_rank_dest[sort2].contiguous()
            send_rank_packet_ids = send_rank_packet_ids[sort2].contiguous()
            send_rank_i = send_rank_i[sort2].contiguous()
            send_rank_w = send_rank_w[sort2].contiguous()
            send_rank_hidden = recv_node_hidden.index_select(0, send_rank_packet_ids).contiguous()
        else:
            send_rank_hidden = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)

        if bool(use_stage2_local_rank_fastpath) and rank_packet_count and not bool(use_buildtime_stage2_local_rank_fastpath):
            stage2_local_rank_mask = send_rank_dest.eq(int(self.comm_rank))
            stage2_comm_rank_mask = ~stage2_local_rank_mask
            local_rank_packet_ids_direct = send_rank_packet_ids[stage2_local_rank_mask].contiguous()
            local_rank_hidden_direct = send_rank_hidden[stage2_local_rank_mask].contiguous()
            local_rank_i_direct = send_rank_i[stage2_local_rank_mask].contiguous()
            local_rank_w_direct = send_rank_w[stage2_local_rank_mask].contiguous()
            send_rank_dest_comm = send_rank_dest[stage2_comm_rank_mask].contiguous()
            send_rank_packet_ids_comm = send_rank_packet_ids[stage2_comm_rank_mask].contiguous()
            send_rank_hidden_comm = send_rank_hidden[stage2_comm_rank_mask].contiguous()
            send_rank_i_comm = send_rank_i[stage2_comm_rank_mask].contiguous()
            send_rank_w_comm = send_rank_w[stage2_comm_rank_mask].contiguous()
            self._record_node_grouped_value("stage2_local_rank_packet_entries", int(local_rank_packet_ids_direct.numel()))
            self._record_node_grouped_value("stage2_remote_rank_packet_entries", int(send_rank_dest_comm.numel()))
            self._record_node_grouped_value(
                "stage2_local_bypass_bytes",
                int(local_rank_packet_ids_direct.numel()) * int(hidden_dim * x.element_size()),
            )
        elif bool(use_buildtime_stage2_local_rank_fastpath):
            send_rank_dest_comm = send_rank_dest
            send_rank_packet_ids_comm = send_rank_packet_ids
            send_rank_hidden_comm = send_rank_hidden
            send_rank_i_comm = send_rank_i
            send_rank_w_comm = send_rank_w
        else:
            local_rank_packet_ids_direct = torch.empty((0,), device=self.device, dtype=torch.long)
            local_rank_hidden_direct = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
            local_rank_i_direct = torch.empty((0, K), dtype=torch.long, device=self.device)
            local_rank_w_direct = torch.empty((0, K), dtype=x.dtype, device=self.device)
            send_rank_dest_comm = send_rank_dest
            send_rank_packet_ids_comm = send_rank_packet_ids
            send_rank_hidden_comm = send_rank_hidden
            send_rank_i_comm = send_rank_i
            send_rank_w_comm = send_rank_w
            self._record_node_grouped_value("stage2_remote_rank_packet_entries", int(send_rank_dest_comm.numel()))

        stage2_group = self.comm_group
        stage2_group_rank = int(self.comm_rank)
        stage2_members_comm: Tuple[int, ...] = tuple(range(int(self.comm_world)))
        if use_stage2_node_local_group and stage2_local_group_info is not None:
            stage2_group, stage2_members_comm, stage2_group_rank, _stage2_members_global = stage2_local_group_info
            self._record_node_grouped_value("stage2_node_local_group", 1)

        comm_rank_packet_count = int(send_rank_dest_comm.numel())
        t_counts_build = time.perf_counter()
        if use_stage2_node_local_group:
            send_counts2_local, _stage2_member_to_local = self._compress_counts_for_comm_members(
                send_rank_dest_comm,
                stage2_members_comm,
            )
            send_counts2 = send_counts2_local
        else:
            send_counts2 = torch.bincount(send_rank_dest_comm, minlength=self.comm_world).to(dtype=torch.long) if comm_rank_packet_count else torch.zeros((self.comm_world,), device=self.device, dtype=torch.long)
        self._record_node_grouped_value("counts_build_ms", (time.perf_counter() - t_counts_build) * 1000.0)
        dt_ms = (time.perf_counter() - t_grouping) * 1000.0
        self._record_grouping_overhead_ms(dt_ms)
        self._record_node_grouped_ms("intra_fanout", dt_ms)
        t_meta = time.perf_counter()
        self._record_node_grouped_value("metadata_collective_count", 1)
        recv_counts2 = self._exchange_dispatch_counts(
            send_counts2,
            group=stage2_group,
            group_rank=stage2_group_rank,
            label="node_grouped_stage2",
        )
        stage2_global_total = int(getattr(self, "_last_dispatch_count_global_total", 0))
        dt_meta = (time.perf_counter() - t_meta) * 1000.0
        self._record_node_grouped_ms("metadata", dt_meta)
        self._record_node_grouped_value("metadata_exchange_ms", dt_meta)

        t_splits_build = time.perf_counter()
        send_counts2_local_list = [int(c) for c in send_counts2.detach().cpu().tolist()]
        recv_counts2_local_list = [int(c) for c in recv_counts2.detach().cpu().tolist()]
        if use_stage2_node_local_group:
            send_counts2_list = self._expand_counts_from_comm_members(
                send_counts2_local_list,
                stage2_members_comm,
                int(self.comm_world),
            )
            recv_counts2_list = self._expand_counts_from_comm_members(
                recv_counts2_local_list,
                stage2_members_comm,
                int(self.comm_world),
            )
        else:
            send_counts2_list = send_counts2_local_list
            recv_counts2_list = recv_counts2_local_list
        recv_total2 = int(sum(recv_counts2_list))
        id_send_counts2 = [int(c) * K for c in send_counts2_local_list]
        id_recv_counts2 = [int(c) * K for c in recv_counts2_local_list]
        hidden_send_counts2 = [int(c) * hidden_dim for c in send_counts2_local_list]
        hidden_recv_counts2 = [int(c) * hidden_dim for c in recv_counts2_local_list]
        self._record_node_grouped_value("splits_build_ms", (time.perf_counter() - t_splits_build) * 1000.0)
        t_grouping = time.perf_counter()
        counts2_bytes = tensor_nbytes(send_counts2) + tensor_nbytes(recv_counts2)
        if use_owner_packed_payload:
            stage2_payload_width = int(hidden_dim) + 2 * int(K)
            send_rank_payload = self._pack_owner_grouped_payload(
                send_rank_hidden_comm,
                send_rank_i_comm,
                send_rank_w_comm,
                payload_dtype=owner_payload_dtype,
            )
            recv_rank_payload_flat = self._node_grouped_empty(
                (int(recv_total2) * int(stage2_payload_width),),
                dtype=owner_payload_dtype,
                label="stage2_owner_payload_recv_flat",
                workspace=use_workspace_reuse,
            )
            stage2_meta_bytes = max(0, int(tensor_nbytes(send_rank_payload)) - int(tensor_nbytes(send_rank_hidden_comm)))
        else:
            recv_rank_hidden_flat = self._node_grouped_empty(
                (int(recv_total2) * int(hidden_dim),),
                dtype=x.dtype,
                label="stage2_hidden_recv_flat",
                workspace=use_workspace_reuse,
            )
        if use_owner_packed_payload:
            pass
        elif use_packed_meta:
            send_rank_meta = torch.cat(
                [send_rank_i_comm.to(pack_meta_dtype), send_rank_w_comm.to(pack_meta_dtype)],
                dim=1,
            ).contiguous()
            recv_rank_meta_flat = self._node_grouped_empty(
                (int(recv_total2) * 2 * int(K),),
                dtype=pack_meta_dtype,
                label="stage2_meta_recv_flat",
                workspace=use_workspace_reuse,
            )
            meta_send_counts2 = [int(c) * 2 * K for c in send_counts2_local_list]
            meta_recv_counts2 = [int(c) * 2 * K for c in recv_counts2_local_list]
            stage2_meta_bytes = tensor_nbytes(send_rank_meta)
            self._record_node_grouped_value("metadata_collective_count", 1)
        else:
            recv_rank_i_flat = self._node_grouped_empty(
                (int(recv_total2) * int(K),),
                dtype=torch.long,
                label="stage2_id_recv_flat",
                workspace=use_workspace_reuse,
            )
            recv_rank_w_flat = self._node_grouped_empty(
                (int(recv_total2) * int(K),),
                dtype=x.dtype,
                label="stage2_weight_recv_flat",
                workspace=use_workspace_reuse,
            )
            stage2_meta_bytes = tensor_nbytes(send_rank_i_comm) + tensor_nbytes(send_rank_w_comm)
            self._record_node_grouped_value("metadata_collective_count", 2)
        if use_owner_packed_payload:
            self._record_node_grouped_allocation(recv_rank_payload_flat, send_rank_payload)
            stage2_dispatch_bytes = tensor_nbytes(send_rank_payload)
            stage2_dispatch_item_bytes = int(stage2_payload_width) * int(send_rank_payload.element_size())
        else:
            self._record_node_grouped_allocation(recv_rank_hidden_flat)
            stage2_dispatch_bytes = tensor_nbytes(send_rank_hidden_comm) + stage2_meta_bytes
            stage2_dispatch_item_bytes = int(hidden_dim * send_rank_hidden_comm.element_size()) + (
                int(2 * K * send_rank_w_comm.element_size())
                if use_packed_meta
                else int(K * send_rank_i_comm.element_size()) + int(K * send_rank_w_comm.element_size())
            )
        stage2_gather_bytes = tensor_nbytes(send_rank_hidden_comm)
        self._record_node_grouped_value("metadata_bytes", int(counts2_bytes + stage2_meta_bytes))
        self._record_activation_comm_bytes(counts2_bytes + stage2_dispatch_bytes + stage2_gather_bytes)

        remote_rank_packet_count = int((send_rank_dest_comm != int(self.comm_rank)).sum().item()) if comm_rank_packet_count else 0
        remote_rank_bytes = remote_rank_packet_count * (
            int(stage2_dispatch_item_bytes)
            + int(hidden_dim * send_rank_hidden_comm.element_size())
        )
        self._record_activation_remote_bytes(remote_rank_bytes)

        if self.metrics is not None:
            self.metrics.node_grouped_pair_entries += int(T * K)
            self.metrics.node_grouped_node_packet_entries += int(node_packet_count)
            self.metrics.node_grouped_rank_packet_entries += int(rank_packet_count)
            self.metrics.node_grouped_dispatch_calls += 1

        rank_payload_bytes = int(hidden_dim * send_rank_hidden_comm.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            if use_owner_packed_payload:
                rank_payload_bytes = int(stage2_dispatch_item_bytes)
            elif use_packed_meta:
                rank_payload_bytes += int(2 * K * send_rank_w_comm.element_size())
            else:
                rank_payload_bytes += int(K * send_rank_i_comm.element_size()) + int(K * send_rank_w_comm.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            rank_payload_bytes += int(hidden_dim * send_rank_hidden_comm.element_size())
        dt_ms = (time.perf_counter() - t_grouping) * 1000.0
        self._record_grouping_overhead_ms(dt_ms)
        self._record_node_grouped_ms("intra_fanout", dt_ms)
        t_fanout = time.perf_counter()
        if use_stage2_node_local_group:
            weaknet_payload2 = {"sim_ms": 0.0}
        else:
            weaknet_payload2 = _maybe_sleep_logical_weaknet(
                comm_global_ranks=self.comm_global_ranks,
                send_counts=send_counts2,
                payload_bytes_per_item=rank_payload_bytes,
                metrics=self.metrics,
                layer_idx=self.layer_idx,
                label="node_grouped_stage2",
            )
        if _logical_weaknet_enabled() and not use_stage2_node_local_group:
            stage2_sleep_ms = float(weaknet_payload2.get("sim_ms", 0.0))
            self._record_node_grouped_value("inter_send_stage2_weaknet_sleep_ms", stage2_sleep_ms)
            self._record_node_grouped_value("inter_send_weaknet_sleep_ms", stage2_sleep_ms)
        t_grouping = time.perf_counter()
        t_stage2_collective = time.perf_counter()

        if bool(use_stage_local_skip) and int(stage2_global_total) == 0:
            recv_rank_hidden = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
            recv_rank_i = torch.empty((0, K), dtype=torch.long, device=self.device)
            recv_rank_w = torch.empty((0, K), dtype=x.dtype, device=self.device)
            self._record_node_grouped_value("stage2_local_fastpath_skip", 1)
        else:
            if use_owner_packed_payload:
                recv_rank_payload_flat = self._all_to_all_flat_with_dummy_timed(
                    send_rank_payload.reshape(-1).contiguous(),
                    output_counts=recv_counts2_local_list,
                    input_counts=send_counts2_local_list,
                    item_width=stage2_payload_width,
                    group=stage2_group,
                    group_rank=stage2_group_rank,
                    label="node_grouped_stage2_owner_payload",
                    workspace=use_workspace_reuse,
                )
                recv_rank_payload = recv_rank_payload_flat.view(recv_total2, stage2_payload_width).contiguous()
                recv_rank_hidden, recv_rank_i, recv_rank_w = self._unpack_owner_grouped_payload(
                    recv_rank_payload,
                    hidden_dim=hidden_dim,
                    top_k=K,
                    activation_dtype=x.dtype,
                )
            else:
                recv_rank_hidden_flat = self._all_to_all_flat_with_dummy_timed(
                    send_rank_hidden_comm.reshape(-1).contiguous(),
                    output_counts=recv_counts2_local_list,
                    input_counts=send_counts2_local_list,
                    item_width=hidden_dim,
                    group=stage2_group,
                    group_rank=stage2_group_rank,
                    label="node_grouped_stage2_hidden",
                    workspace=use_workspace_reuse,
                )
                recv_rank_hidden = recv_rank_hidden_flat.view(-1, hidden_dim).contiguous()
            if use_owner_packed_payload:
                pass
            elif use_packed_meta:
                recv_rank_meta_flat = self._all_to_all_flat_with_dummy_timed(
                    send_rank_meta.reshape(-1).contiguous(),
                    output_counts=recv_counts2_local_list,
                    input_counts=send_counts2_local_list,
                    item_width=2 * K,
                    group=stage2_group,
                    group_rank=stage2_group_rank,
                    label="node_grouped_stage2_meta",
                    workspace=use_workspace_reuse,
                )
                recv_rank_meta = recv_rank_meta_flat.view(recv_total2, 2 * K).contiguous()
                recv_rank_i = recv_rank_meta[:, :K].to(torch.long).contiguous()
                recv_rank_w = recv_rank_meta[:, K:].to(x.dtype).contiguous()
            else:
                recv_rank_i_flat = self._all_to_all_flat_with_dummy_timed(
                    send_rank_i_comm.reshape(-1).contiguous(),
                    output_counts=id_recv_counts2,
                    input_counts=id_send_counts2,
                    item_width=K,
                    group=stage2_group,
                    group_rank=stage2_group_rank,
                    label="node_grouped_stage2_ids",
                    workspace=use_workspace_reuse,
                )
                recv_rank_w_flat = self._all_to_all_flat_with_dummy_timed(
                    send_rank_w_comm.reshape(-1).contiguous(),
                    output_counts=id_recv_counts2,
                    input_counts=id_send_counts2,
                    item_width=K,
                    group=stage2_group,
                    group_rank=stage2_group_rank,
                    label="node_grouped_stage2_weights",
                    workspace=use_workspace_reuse,
                )
                recv_rank_i = recv_rank_i_flat.view(recv_total2, K).contiguous()
                recv_rank_w = recv_rank_w_flat.view(recv_total2, K).contiguous()
        self._record_node_grouped_value("inter_send_stage2_collective_ms", (time.perf_counter() - t_stage2_collective) * 1000.0)
        if local_rank_i_direct.numel() > 0:
            exec_rank_hidden = torch.cat([recv_rank_hidden, local_rank_hidden_direct], dim=0).contiguous()
            exec_rank_i = torch.cat([recv_rank_i, local_rank_i_direct], dim=0).contiguous()
            exec_rank_w = torch.cat([recv_rank_w, local_rank_w_direct], dim=0).contiguous()
        else:
            exec_rank_hidden = recv_rank_hidden
            exec_rank_i = recv_rank_i
            exec_rank_w = recv_rank_w

        valid_recv_i = exec_rank_i[exec_rank_i.ge(0)]
        self._record_local_expert_load(valid_recv_i)

        dt_ms = (time.perf_counter() - t_fanout) * 1000.0
        self._record_node_grouped_ms("intra_fanout", dt_ms)
        self._record_grouping_overhead_ms((time.perf_counter() - t_grouping) * 1000.0)
        t_exec = time.perf_counter()
        exec_rank_packet_out = self._execute_local_rank_grouped_packets(
            recv_hidden=exec_rank_hidden,
            recv_i=exec_rank_i,
            recv_w=exec_rank_w,
        )
        recv_rank_packet_out = exec_rank_packet_out[:recv_total2].contiguous()
        local_rank_packet_out_direct = exec_rank_packet_out[recv_total2:].contiguous()
        self._record_expert_exec_ms((time.perf_counter() - t_exec) * 1000.0)

        t_gather = time.perf_counter()
        t_rank_to_node_gather = time.perf_counter()
        if bool(use_stage_local_skip) and int(stage2_global_total) == 0:
            gathered_rank_packet_out = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
        else:
            gathered_rank_packet_out_flat = self._all_to_all_flat_with_dummy_timed(
                recv_rank_packet_out.reshape(-1).contiguous(),
                output_counts=send_counts2_local_list,
                input_counts=recv_counts2_local_list,
                item_width=hidden_dim,
                group=stage2_group,
                group_rank=stage2_group_rank,
                label="node_grouped_stage2_gather",
                workspace=use_workspace_reuse,
            )
            gathered_rank_packet_out = gathered_rank_packet_out_flat.view(-1, hidden_dim).contiguous()
        self._record_node_grouped_value("rank_to_node_gather_ms", (time.perf_counter() - t_rank_to_node_gather) * 1000.0)

        node_packet_out = self._node_grouped_empty(
            (recv_total_all, hidden_dim),
            dtype=x.dtype,
            label="node_packet_out",
            workspace=use_workspace_reuse,
            zero=True,
        )
        self._record_node_grouped_allocation(node_packet_out, gathered_rank_packet_out)
        self._record_node_grouped_value("combine_intermediate_buffer_bytes", int(tensor_nbytes(node_packet_out)))
        t_final_combine = time.perf_counter()
        if gathered_rank_packet_out.numel() > 0:
            if bool(use_identity_combine_fastpath) and self._node_grouped_identity_ids(
                send_rank_packet_ids_comm,
                int(gathered_rank_packet_out.shape[0]),
            ):
                node_packet_out[: int(gathered_rank_packet_out.shape[0])] = gathered_rank_packet_out
            else:
                node_packet_out.index_add_(0, send_rank_packet_ids_comm, gathered_rank_packet_out)
        if local_rank_packet_out_direct.numel() > 0:
            if bool(use_identity_combine_fastpath) and int(gathered_rank_packet_out.shape[0]) == 0 and self._node_grouped_identity_ids(
                local_rank_packet_ids_direct,
                int(local_rank_packet_out_direct.shape[0]),
            ):
                node_packet_out[: int(local_rank_packet_out_direct.shape[0])] = local_rank_packet_out_direct
            else:
                node_packet_out.index_add_(0, local_rank_packet_ids_direct, local_rank_packet_out_direct)
        self._record_node_grouped_value("final_combine_ms", (time.perf_counter() - t_final_combine) * 1000.0)

        final = torch.zeros((T, H), device=x.device, dtype=x.dtype)
        self._record_node_grouped_allocation(final)
        t_node_to_token_restore = time.perf_counter()
        if bool(use_stage_local_skip) and int(stage1_global_total) == 0:
            gathered_node_out = torch.empty((0, hidden_dim), dtype=x.dtype, device=self.device)
        else:
            gathered_node_out_flat = self._all_to_all_flat_with_dummy_timed(
                node_packet_out[:recv_total].contiguous().reshape(-1),
                output_counts=send_counts_list,
                input_counts=recv_counts_list,
                item_width=hidden_dim,
                group=self.comm_group,
                group_rank=self.comm_rank,
                label="node_grouped_stage1_gather",
                workspace=use_workspace_reuse,
            )
            gathered_node_out = gathered_node_out_flat.view(-1, hidden_dim).contiguous()
        self._record_node_grouped_value("node_to_token_restore_ms", (time.perf_counter() - t_node_to_token_restore) * 1000.0)
        self._record_node_grouped_allocation(gathered_node_out)
        t_final_combine = time.perf_counter()
        if send_token_ids_cross.numel() > 0:
            final.index_add_(0, send_token_ids_cross, gathered_node_out)
        if local_token_ids.numel() > 0:
            final.index_add_(0, local_token_ids, node_packet_out[recv_total:].contiguous())
        self._record_node_grouped_value("final_combine_ms", (time.perf_counter() - t_final_combine) * 1000.0)
        self._record_node_grouped_ms("gather", (time.perf_counter() - t_gather) * 1000.0)
        return final

    @_DYNAMO_DISABLE
    def _experts_ep_group_node_grouped_logical_fastpath(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
        seq_len: int = 1,
    ) -> torch.Tensor:
        """Low-control NGD approximation for weak-edge benchmark runs.

        Router outputs and expert ownership are unchanged.  The real data-plane
        execution stays on normal EP, while the modeled logical weaknet charge
        uses node-grouped (token, destination-node) records.  This is the
        runtime stand-in for a fused NGD pack/metadata path with negligible
        Python/stage2 overhead.
        """
        t0 = time.perf_counter()
        T = int(x.shape[0])
        H = int(x.shape[1])
        K = int(idx.shape[1])
        if T == 0 or K == 0:
            return torch.zeros_like(x)

        impl = str(os.environ.get("SPARSE_NGD_LOGICAL_FASTPATH_IMPL", "auto") or "auto").strip().lower()
        use_ep_reuse = impl in (
            "auto",
            "",
            "ep_reuse",
            "ep-reuse",
            "reuse_ep",
            "reuse-normal-ep",
            "normal_ep_reuse",
        )
        if use_ep_reuse and int(seq_len) == 1 and int(T) == 1:
            SparseContext.logical_ngd_ep_reuse_once = True
            try:
                return self._experts_ep_group(x, idx, w)
            finally:
                SparseContext.logical_ngd_ep_reuse_once = False

        try:
            cpu_max_items = int(os.environ.get("SPARSE_NGD_LOGICAL_FASTPATH_CPU_MAX_ITEMS", "64") or 64)
        except Exception:
            cpu_max_items = 64
        use_cpu_decode = impl in ("cpu", "cpu_decode", "decode_cpu")
        if impl in ("auto", ""):
            use_cpu_decode = int(seq_len) == 1 and int(T * K) <= int(cpu_max_items)

        cpu_counts: Optional[Tuple[torch.Tensor, int]] = None
        if use_cpu_decode:
            cpu_counts = self._logical_fastpath_cpu_decode_counts(idx, w, max_items=cpu_max_items)

        if cpu_counts is not None:
            send_counts, node_packet_count = cpu_counts
        else:
            flat_i = idx.reshape(-1).to(torch.long)
            dest_rank_mat = self._resolve_dest_for_flat_experts(flat_i, reason="logical_ngd_fastpath", count_metrics=False).view(T, K).to(torch.long)
            rank_to_node = self._rank_to_node_tensor()
            dest_node_mat = rank_to_node[dest_rank_mat]
            max_node = int(rank_to_node.max().item()) if rank_to_node.numel() > 0 else 0
            valid = w.to(torch.float32).ne(0.0)
            present = torch.zeros((T, max_node + 1), device=self.device, dtype=torch.bool)
            token_grid = torch.arange(T, device=self.device, dtype=torch.long).view(T, 1).expand(T, K)
            present[token_grid[valid], dest_node_mat[valid]] = True
            _token_ids, node_vals = torch.nonzero(present, as_tuple=True)
            reps = self._node_representative_tensor()
            send_node_dest = reps[node_vals].to(torch.long)
            node_packet_count = int(send_node_dest.numel())
            send_counts = (
                torch.bincount(send_node_dest, minlength=self.comm_world).to(dtype=torch.long)
                if node_packet_count > 0
                else torch.zeros((self.comm_world,), device=self.device, dtype=torch.long)
            )

        node_payload_bytes = int(H * x.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_DISPATCH_IDS", "1"):
            node_payload_bytes += int(K * torch.empty((), dtype=torch.long).element_size()) + int(K * x.element_size())
        if _env_flag("SPARSE_LOGICAL_SIM_GATHER", "1"):
            node_payload_bytes += int(H * x.element_size())

        self._record_node_grouped_value("node_packet_entries", int(node_packet_count))
        self._record_node_grouped_value("rank_packet_entries", int(T * K))
        if self.metrics is not None:
            self.metrics.node_grouped_pair_entries += int(T * K)
            self.metrics.node_grouped_node_packet_entries += int(node_packet_count)
            self.metrics.node_grouped_rank_packet_entries += int(T * K)
            self.metrics.node_grouped_dispatch_calls += 1

        self._record_node_grouped_ms("pack", (time.perf_counter() - t0) * 1000.0)
        t_sleep = time.perf_counter()
        weaknet_payload = _maybe_sleep_logical_weaknet(
            comm_global_ranks=self.comm_global_ranks,
            send_counts=send_counts,
            payload_bytes_per_item=node_payload_bytes,
            metrics=self.metrics,
            layer_idx=self.layer_idx,
            label="node_grouped_logical_fastpath",
        )
        if _logical_weaknet_enabled():
            sleep_ms = float(weaknet_payload.get("sim_ms", 0.0))
            self._record_node_grouped_value("inter_send_stage1_weaknet_sleep_ms", sleep_ms)
            self._record_node_grouped_value("inter_send_weaknet_sleep_ms", sleep_ms)
        self._record_node_grouped_ms("inter_send", (time.perf_counter() - t_sleep) * 1000.0)

        # Normal EP below would charge the original per-rank logical weaknet
        # model.  Suppress exactly that next charge so this path represents a
        # fused low-control NGD dispatch, not NGD plus baseline weaknet sleep.
        SparseContext.suppress_logical_weaknet_once = True
        try:
            return self._experts_ep_group(x, idx, w)
        finally:
            SparseContext.suppress_logical_weaknet_once = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Runtime semantics:
        Single runtime semantics:
        1) baseline keeps standard EP(+optional DP) execution as the only baseline path;
        2) optional method logic such as owner-local execution / SKVS is layered on top of this baseline topology;
        3) planner-only RS still executes once at group-start and only schedules future-layer prefetch.
        """
        B, S, H = hidden_states.shape
        x = hidden_states.view(-1, H)
        self._current_forward_batch_size = int(B)
        self._current_forward_seq_len = int(S)
        cp = _hclm_cp_profiler()
        cp_step_id = int(getattr(SparseContext, "runtime_call_id", -1))
        cp_layer_sampled = bool(int(S) == 1 and cp.should_sample(cp_step_id))
        if cp_layer_sampled:
            cp.start_layer(cp_step_id, int(self.layer_idx))
        g = (self.layer_idx // self.rs_group) * self.rs_group
        group_end = min(g + self.rs_group, self.num_layers)
        SparseContext.current_moe_layer_idx = int(self.layer_idx)
        physical_comm_trace_state = self._physical_moe_comm_trace_begin()

        owner_local_batch = bool(getattr(self, "_sparse_local_owner_batch", False))
        if owner_local_batch:
            if x.numel() == 0:
                router_logits = torch.empty((0, self.num_experts), device=hidden_states.device, dtype=torch.float32)
                return hidden_states if self._return_tensor_only_mlp() else (hidden_states, router_logits)
            w, idx, router_logits = self._routing_for_execute(x)
            out = self._owner_exec_full_mlp(x, idx, w, batch_size=B, seq_len=S).view(B, S, H)
            return out if self._return_tensor_only_mlp() else (out, router_logits)

        if SparseContext.oracle_mode == "replay" and SparseContext.oracle_call_id >= 0:
            self._oracle_async_prefetch_future_layers()

        if SparseContext.oracle_mode == "replay" and SparseContext.oracle_call_id >= 0 and self._oracle_stage_current_layer_enabled():
            t0_oracle = time.perf_counter()
            self.oracle_prepare_current_layer()
            if self.metrics is not None:
                dt_ms = (time.perf_counter() - t0_oracle) * 1000.0
                self.metrics.oracle_stage_ms += dt_ms
                if int(SparseContext.oracle_call_id) == 0:
                    self.metrics.oracle_stage_prefill_ms += dt_ms
                else:
                    self.metrics.oracle_stage_decode_ms += dt_ms

        rs_prefetch_prefill = _env_flag("SPARSE_RS_PREFETCH_PREFILL", "0")
        rs_planner_active_this_phase = ((S == 1) or rs_prefetch_prefill) and self.planner_mode == "rs_prefetch"
        rs_layout = getattr(SparseContext, "rs_bank_layout", "group")
        next_layer_rs_feature_mode = self._next_layer_rs_feature_mode() if (((S == 1) or rs_prefetch_prefill) and rs_layout == "next_layer") else "hidden"
        if ((S == 1) or rs_prefetch_prefill) and rs_layout == "next_layer" and next_layer_rs_feature_mode == "hidden":
            self._compute_and_schedule_next_layer_rs(x=x, seq_len=S)

        planner_needed = rs_planner_active_this_phase and self.layer_idx == g and rs_layout != "next_layer"

        if self.layer_idx == g and self.expert_source == "rs" and rs_layout != "next_layer":
            self._compute_rs_cache_group(
                x=x,
                g=g,
                enable_prefetch=False,
                batch_size=B,
                seq_len=S,
            )
        elif self.layer_idx == g and planner_needed:
            self._compute_and_schedule_rs_prefetch_group(x=x, g=g)

        if int(S) == 1 and bool(getattr(self, "_transition_prefetch_enabled", False)):
            src_experts = SparseContext.pop_transition_source_experts(int(self.layer_idx))
            self._schedule_transition_prefetch_from_sources(src_experts)


        w, idx, router_logits = self._timed_segment("router", lambda: self._routing_for_execute(x))
        SparseContext.record_routing_profile(
            layer_idx=int(self.layer_idx),
            source_rank=int(self.comm_rank),
            topk_indices=idx,
            num_layers=int(self.num_layers),
            world_size=int(self.comm_world),
            num_experts=int(self.num_experts),
            top_k=int(self.top_k),
            hidden_size=int(self.hidden_size),
            dtype=str(self.dtype),
        )
        self._record_prefetch_eval_actual(idx, seq_len=S)
        if str(SparseContext.current_phase) == "prefill":
            SparseContext.record_prefill_experts(self.layer_idx, idx)
        if int(S) == 1:
            self._schedule_transition_prefetch_for_next_layer(idx)
        if ((S == 1) or rs_prefetch_prefill) and rs_layout == "next_layer" and next_layer_rs_feature_mode == "hidden_router_topk":
            self._compute_and_schedule_next_layer_rs(x=x, seq_len=S, source_idx=idx, source_w=w)

        if SparseContext.oracle_mode == "record" and SparseContext.oracle_call_id >= 0:
            flat_i = idx.reshape(-1).to(torch.long)
            stage_rank_map = self._expert_to_rank_for_stage_tensor()
            local_needed = sorted(set(
                int(eid) for eid in flat_i.tolist()
                if int(stage_rank_map[int(eid)].item()) == int(self.comm_rank)
            ))
            SparseContext.record_oracle_local_experts(self.layer_idx, local_needed)

        SparseContext.update_stats(self.layer_idx, idx, w)
        self._record_routing_locality(idx)

        if self.world == 1:
            expert_out = self._timed_segment("expert", lambda: self._experts_world1_exact(x, idx, w))
            shared_out = self._timed_segment(
                "shared",
                lambda: self._shared_expert_forward(x),
            )
            out = (expert_out + shared_out).view(B, S, H)
        else:
            if self._node_grouped_dispatch_enabled(seq_len=S, idx=idx, x=x):
                expert_out = self._timed_segment(
                    "expert",
                    lambda: self._experts_ep_group_node_grouped(x, idx, w, seq_len=S),
                )
            elif self._rank_grouped_dispatch_enabled(seq_len=S):
                expert_out = self._timed_segment(
                    "expert",
                    lambda: self._experts_ep_group_rank_grouped(x, idx, w),
                )
            else:
                expert_out = self._timed_segment(
                    "expert",
                    lambda: self._experts_ep_group(x, idx, w),
                )
            shared_out = self._timed_segment(
                "shared",
                lambda: self._shared_expert_forward(x),
            )
            out = (expert_out + shared_out).view(B, S, H)

        if cp_layer_sampled:
            cp.end_layer(
                cp_step_id,
                int(self.layer_idx),
                {
                    "local_expert_count": int(len(self.local_expert_ids)),
                    "local_token_count": int(x.shape[0]),
                },
            )
        self._physical_moe_comm_trace_end(
            physical_comm_trace_state,
            batch_size=B,
            seq_len=S,
        )
        return out if self._return_tensor_only_mlp() else (out, router_logits)

    @classmethod
    def clear_all_expert_caches(cls) -> None:
        """Diagnostic helper used by stage-aware SCP ablations.

        This clears resident expert weights between prefill and decode when
        SPARSE_SCP_CLEAR_CACHE_BEFORE_DECODE=1. It is disabled by default and
        should be reported as a cache-boundary diagnostic if used.
        """
        for blk in list(getattr(SparseContext, "layer_blocks", {}).values()):
            cache = getattr(blk, "_cache", None)
            clear = getattr(cache, "clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass
            try:
                getattr(blk, "_prefetched_ids_this_group", set()).clear()
                getattr(blk, "_prefetch_used_ids_this_group", set()).clear()
                getattr(blk, "_prefetch_expire_layer", {}).clear()
                getattr(blk, "_transition_prefetched_ids", set()).clear()
                getattr(blk, "_transition_used_ids", set()).clear()
            except Exception:
                pass

    @classmethod
    def handle_epipe_prefill_extra_cache_decode_boundary(cls, metrics: Optional[Metrics] = None) -> None:
        total_start = 0
        total_cleared = 0
        for blk in list(getattr(SparseContext, "layer_blocks", {}).values()):
            try:
                total_start += int(blk._epipe_prefill_extra_cache_resident_count())
            except Exception:
                pass
            if bool(getattr(blk, "_epipe_prefill_extra_cache_clear_at_decode", False)):
                try:
                    total_cleared += int(blk._epipe_prefill_extra_cache_clear())
                except Exception:
                    pass
        if metrics is not None:
            metrics.epipe_prefill_extra_cache_resident_at_decode_start += int(total_start)
            if total_cleared > 0:
                metrics.epipe_prefill_extra_cache_evict_count += int(total_cleared)

    @classmethod
    def record_epipe_prefill_extra_cache_decode_end(cls, metrics: Optional[Metrics] = None) -> None:
        if metrics is None:
            return
        total_end = 0
        for blk in list(getattr(SparseContext, "layer_blocks", {}).values()):
            try:
                total_end += int(blk._epipe_prefill_extra_cache_resident_count())
            except Exception:
                pass
        metrics.epipe_prefill_extra_cache_resident_count += int(total_end)
        metrics.epipe_prefill_extra_cache_resident_at_decode_end += int(total_end)

    @classmethod
    def demote_prefill_scp_transients_for_decode(cls, metrics: Optional[Metrics] = None) -> None:
        if bool(getattr(cls, "prefill_scp_transient_demoted_for_decode", False)):
            return
        total_start = 0
        total_demoted = 0
        for blk in list(getattr(cls, "layer_blocks", {}).values()):
            cache = getattr(blk, "_cache", None)
            try:
                ids = getattr(cache, "prefill_scp_transient_ids", lambda: [])()
                total_start += int(len(ids))
            except Exception:
                pass
            demote = getattr(cache, "demote_prefill_scp_transients", None)
            if callable(demote):
                try:
                    total_demoted += int(demote())
                except Exception:
                    pass
        if metrics is not None:
            metrics.prefill_scp_transient_resident_at_decode_start += int(total_start)
            metrics.prefill_scp_transient_demote_count += int(total_demoted)
            metrics.epipe_prefill_transient_demote_count += int(total_demoted)
        cls.prefill_scp_transient_demoted_for_decode = True

    @classmethod
    def record_prefill_scp_transient_decode_end(cls, metrics: Optional[Metrics] = None) -> None:
        if metrics is None:
            return
        total_end = 0
        for blk in list(getattr(cls, "layer_blocks", {}).values()):
            cache = getattr(blk, "_cache", None)
            try:
                ids = getattr(cache, "prefill_scp_transient_ids", lambda: [])()
                total_end += int(len(ids))
            except Exception:
                pass
        metrics.prefill_scp_transient_resident_at_decode_end += int(total_end)

    def _gather_comm_group_cache_resident_sets(self) -> List[set[int]]:
        local_ids = []
        try:
            local_ids = sorted(set(int(x) for x in self._cache.resident_ids()))
        except Exception:
            local_ids = []
        if not (dist.is_available() and dist.is_initialized()):
            return [set(local_ids) for _ in range(max(int(self.comm_world), 1))]
        world = int(self.comm_world)
        gathered: List[Optional[List[int]]] = [None for _ in range(world)]
        dist.all_gather_object(gathered, local_ids, group=self.comm_group)
        out: List[set[int]] = []
        for obj in gathered:
            if obj is None:
                out.append(set())
            else:
                out.append(set(int(x) for x in obj))
        while len(out) < world:
            out.append(set())
        return out

    def plan_group_owner_ranks_from_hidden(
        self,
        hidden_states: torch.Tensor,
        prev_owner_ranks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"plan_group_owner_ranks_from_hidden expects [B,S,H], got {tuple(hidden_states.shape)}"
            )
        B, S, H = hidden_states.shape
        if B <= 0:
            raise RuntimeError(
                f"owner planning received empty batch at layer={self.layer_idx}. This means owner plan is being recomputed after batch has already been sharded."
            )
        x = hidden_states.view(-1, H)
        g = (self.layer_idx // self.rs_group) * self.rs_group
        if SparseContext.rs_bank is not None:
            self._compute_rs_cache_group_local_only(x, g)
            if len(SparseContext.group_pred_topi) > 0:
                return self._estimate_balanced_group_owner_ranks_from_predicted_group(
                    SparseContext.group_pred_topi,
                    SparseContext.group_pred_topv,
                    batch_size=B,
                    seq_len=S,
                    prev_owner_ranks=prev_owner_ranks,
                )
        w, idx, _ = self._routing_router(x)
        return self._estimate_balanced_group_owner_ranks_from_predicted_group(
            {self.layer_idx: idx},
            {self.layer_idx: w},
            batch_size=B,
            seq_len=S,
            prev_owner_ranks=prev_owner_ranks,
        )

    def _estimate_balanced_group_owner_ranks_from_predicted_group(
        self,
        group_topi: Dict[int, torch.Tensor],
        group_topv: Dict[int, torch.Tensor],
        batch_size: int,
        seq_len: int,
        prev_owner_ranks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        更稳的 owner 选择策略：
        1) 先计算每个 sample 在每个 rank 上的 expert contribution score
        2) 再引入 group 内负载均衡约束，避免 owner 塌缩到单卡
        3) 优先在“较空闲”的 rank 中选 contribution 最大者
        4) 对 owner 变化加惩罚，降低 group 间频繁迁移
        5) 对 working set miss 加惩罚，但不让它压过负载均衡

        设计要点：
        - contribution 仍然是主信号
        - load balance 是硬约束 + 软惩罚结合
        - 对 batch=4, world=4 这类场景，会显著减少 [x,x,x,x] 这类极端分配
        """
        if len(group_topi) == 0:
            return torch.zeros((batch_size,), device=self.device, dtype=torch.long)

        device = next(iter(group_topi.values())).device
        world = int(self.comm_world)
        batch_size = int(batch_size)
        seq_len = int(seq_len)

        # 当前 group 内已经分配到各 rank 的 sample 数
        rank_load = torch.zeros((world,), device=device, dtype=torch.long)
        resident_sets = self._gather_comm_group_cache_resident_sets()

        # -------- hard capacity：限制单个 rank 在一个 group 中承接过多 sample --------
        # batch >= ep_size 时默认强均衡；例如 batch=4, ep=4 -> hard_cap=1。
        # 更小 batch 无法填满所有 rank，保持宽松以避免僵硬迁移。
        base_cap = max(1, (batch_size + world - 1) // world)  # ceil(batch/world)
        cap_slack = 0 if batch_size >= world else max(0, batch_size - base_cap)
        hard_cap = base_cap + cap_slack

        # 对 very small batch，避免候选集过窄
        # batch=1/2 时不强推均匀
        if batch_size <= 2:
            hard_cap = batch_size

        owners: List[int] = []

        sorted_layers = sorted(group_topi.keys())

        for b in range(batch_size):
            # ---------------- contribution score ----------------
            rank_scores = torch.zeros((world,), device=device, dtype=torch.float32)
            pred_eids = set()

            for l in sorted_layers:
                topi_l = group_topi[l]
                topv_l = group_topv[l]

                idx_bs = topi_l.view(batch_size, seq_len, self.top_k)
                w_bs = topv_l.view(batch_size, seq_len, self.top_k)

                idx_b = idx_bs[b].reshape(-1).to(torch.long)
                w_b = w_bs[b].reshape(-1).float()
                r_b = self.expert_to_rank[idx_b]

                rank_scores.scatter_add_(0, r_b, w_b)
                pred_eids.update(int(x) for x in idx_b.tolist())

            # ---------------- working set penalty ----------------
            # 本质是“这个 sample 若放到某 rank，大概要拉多少 expert”
            # 这里必须对所有 candidate rank 使用对称信息，不能只看当前规划 rank 的本地 cache，
            # 否则 owner 规划会系统性偏向 leader / rank0。
            working_set_penalty = torch.zeros((world,), device=device, dtype=torch.float32)
            unique_e_cnt = len(pred_eids)

            for r in range(world):
                if unique_e_cnt == 0:
                    working_set_penalty[r] = 0.0
                    continue
                resident_r = resident_sets[r] if r < len(resident_sets) else set()
                miss_cnt = 0
                for eid in pred_eids:
                    if int(eid) not in resident_r:
                        miss_cnt += 1
                working_set_penalty[r] = float(miss_cnt)

            # ---------------- owner change penalty ----------------
            change_penalty = torch.zeros((world,), device=device, dtype=torch.float32)
            if prev_owner_ranks is not None and prev_owner_ranks.numel() == batch_size:
                prev_r = int(prev_owner_ranks[b].item())
                if 0 <= prev_r < world:
                    change_penalty[:] = 1.0
                    change_penalty[prev_r] = 0.0

            # ---------------- load penalty ----------------
            # rank 越忙，惩罚越高
            load_penalty = rank_load.to(dtype=torch.float32)

            final_score = (
                rank_scores
                - self.owner_load_balance_alpha * load_penalty
                - self.owner_workingset_gamma * working_set_penalty
                - self.owner_change_penalty_beta * change_penalty
            )

            # ---------------- candidate set: 优先从较空闲 GPU 中选 ----------------
            min_load = int(rank_load.min().item())

            # 允许略高于最小负载的 rank 参与竞争，避免过于僵硬
            # batch 小时用更宽松的 slack
            load_slack = 1 if batch_size >= world else 0
            candidate_mask = (rank_load <= (min_load + load_slack))

            # 再加 hard cap，防止单卡吃下过多 sample
            cap_mask = (rank_load < hard_cap)
            candidate_mask = candidate_mask & cap_mask

            # 如果筛完空了，只退回到未超过 hard cap 的集合
            if not bool(candidate_mask.any().item()):
                candidate_mask = cap_mask.clone()

            # 如果还空，说明所有 rank 都到 cap 了，只能全放开
            if not bool(candidate_mask.any().item()):
                candidate_mask = torch.ones((world,), device=device, dtype=torch.bool)

            candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)

            # ---------------- 在候选集中选 owner ----------------
            cand_scores = final_score.index_select(0, candidate_idx)
            best_pos = int(torch.argmax(cand_scores).item())
            owner = int(candidate_idx[best_pos].item())

            # ---------------- 二次修正：若最优贡献 rank 明显更强，但当前 owner 只是因负载限制落选 ----------------
            # 避免过度追求均匀，把 locality 全牺牲掉
            best_mass_rank = int(torch.argmax(rank_scores).item())
            best_mass = float(rank_scores[best_mass_rank].item())
            chosen_mass = float(rank_scores[owner].item())

            # 若最优贡献 rank 还没超过硬上限，且比当前候选 winner 明显强，则允许回退到它
            if rank_load[best_mass_rank] < hard_cap:
                if best_mass > 0.0 and chosen_mass / best_mass < 0.80:
                    owner = best_mass_rank

            # ---------------- owner stickiness ----------------
            # 如果上一组 owner 仍未超过硬上限，且新 owner 的综合分数优势很小，
            # 保持原 owner，避免每个 group 都迁移 KV/keep-plan。
            if prev_owner_ranks is not None and prev_owner_ranks.numel() == batch_size:
                prev_r = int(prev_owner_ranks[b].item())
                if 0 <= prev_r < world and int(rank_load[prev_r].item()) < hard_cap:
                    owner_score = float(final_score[owner].item())
                    prev_score = float(final_score[prev_r].item())
                    if owner != prev_r and (owner_score - prev_score) <= float(self.owner_stickiness_delta):
                        owner = prev_r

            owners.append(owner)
            rank_load[owner] += 1

        return torch.tensor(owners, device=device, dtype=torch.long)

    def _estimate_group_unique_pred_eids(
        self,
        group_topi: Dict[int, torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> List[int]:
        out: List[int] = []
        for b in range(batch_size):
            uniq = set()
            for l in sorted(group_topi.keys()):
                idx_bs = group_topi[l].view(batch_size, seq_len, self.top_k)
                idx_b = idx_bs[b].reshape(-1).tolist()
                uniq.update(int(x) for x in idx_b)
            out.append(len(uniq))
        return out


    def _get_any_module_expert_to_device(
        self,
        eid: int,
        is_prefetch: bool,
        count_as_access: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cache.get(eid)
        if cached is not None:
            if count_as_access and self.metrics is not None:
                if eid in self._prefetched_ids_this_group:
                    self._record_cache_hit(is_prefetch_hit=True)
                    self._prefetch_used_ids_this_group.add(eid)
                    try:
                        self._cache.unpin(eid)
                        self._cache.promote_hot(eid)
                    except Exception:
                        pass
                else:
                    self._record_cache_hit()
            self._maybe_wait_prefetch(eid)
            return cached

        if count_as_access:
            self._record_cache_miss()
        if count_as_access:
            self._on_demand_access(eid)

        copied_from_host = False

        if eid in self._local_eid_set and self.expert_offload == "cpu":
            _stall_begin = None
            if (not is_prefetch) and bool(count_as_access):
                _stall_begin = self._record_demand_h2d_window_begin()
            gw = self.local_gate_cpu[eid].to(self.device, non_blocking=True)
            uw = self.local_up_cpu[eid].to(self.device, non_blocking=True)
            dw = self.local_down_cpu2[eid].to(self.device, non_blocking=True)
            if _stall_begin is not None:
                self._record_demand_h2d_window_end(_stall_begin)
            copied_from_host = True
        elif eid in self._local_eid_set and self.expert_offload == "gpu":
            m = self.expert_modules[eid]
            gw = m.gate_proj.weight
            uw = m.up_proj.weight
            dw = m.down_proj.weight
            copied_from_host = False
        else:
            gw, uw, dw = self._get_module_from_cpu_once(eid)
            copied_from_host = True

        if copied_from_host:
            self._record_h2d_bytes(
                tensor_nbytes(gw) + tensor_nbytes(uw) + tensor_nbytes(dw),
                is_prefetch=is_prefetch,
            )

        self._cache_put_guarded(eid, (gw, uw, dw), pinned=bool(is_prefetch), prefer_hot=self._should_prefer_hot(eid, is_prefetch))
        if is_prefetch:
            self._record_prefetch_event(eid)
        return gw, uw, dw

    def _prefetch_for_group_target_owners(
        self,
        topi: torch.Tensor,          # [B*S, K]
        topv: torch.Tensor,          # [B*S, K]
        owner_ranks: torch.Tensor,   # [B]
        batch_size: int,
        seq_len: int,
    ) -> None:
        if self.planner_mode != "rs_prefetch":
            return
        if self.expert_offload != "cpu":
            return
        if self.prefetch_top_m <= 0 or self._effective_prefetch_capacity() <= 0:
            return

        expected_numel = batch_size * seq_len * self.top_k
        if int(topi.numel()) != expected_numel or int(topv.numel()) != expected_numel:
            raise RuntimeError(
                f"[layer={self.layer_idx}] prefetch target owner routing shape mismatch: "
                f"batch_size={batch_size} seq_len={seq_len} top_k={self.top_k} "
                f"topi.shape={tuple(topi.shape)} topv.shape={tuple(topv.shape)} "
                f"expected_numel={expected_numel} topi_numel={int(topi.numel())} topv_numel={int(topv.numel())}"
            )

        idx_bs = topi.view(batch_size, seq_len, self.top_k)
        w_bs = topv.view(batch_size, seq_len, self.top_k)

        score_map: Dict[int, float] = {}

        for b in range(batch_size):
            owner_me = self.comm_rank
            if int(owner_ranks[b].item()) != owner_me:
                continue

            idx_b = idx_bs[b].reshape(-1)
            w_b = w_bs[b].float().reshape(-1)

            for eid, wv in zip(idx_b.tolist(), w_b.tolist()):
                score_map[eid] = score_map.get(eid, 0.0) + float(wv)

        if len(score_map) == 0:
            return

        ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        top_m = min(self.prefetch_top_m, self._cache.capacity, len(ranked))
        ranked = ranked[:top_m]

        ctx = torch.cuda.stream(self.prefetch_stream) if self.prefetch_stream is not None else nullcontext()
        with ctx:
            for eid, _ in ranked:
                if self._cache.has(eid):
                    payload = self._cache.get(eid)
                    self._cache_put_guarded(eid, payload, pinned=True)
                    self._prefetched_ids_this_group.add(eid)
                    continue

                # Standard EP runtime: this rank only executes local experts.
                # Prefetching non-local experts only pollutes a tiny cache and cannot reduce miss.
                if eid not in self._local_eid_set:
                    continue
                if self.experts_kind == "fused":
                    gate_up, down = self._get_any_fused_expert_to_device(
                        eid,
                        is_prefetch=True,
                        count_as_access=False,
                    )
                    self._cache_put_guarded(eid, (gate_up, down), pinned=True)
                else:
                    gw, uw, dw = self._get_any_module_expert_to_device(
                        eid,
                        is_prefetch=True,
                        count_as_access=False,
                    )
                    self._cache_put_guarded(eid, (gw, uw, dw), pinned=True)

                self._prefetched_ids_this_group.add(eid)

    def _get_any_fused_expert_to_device(
        self,
        eid: int,
        is_prefetch: bool,
        count_as_access: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._cache.get(eid)
        if cached is not None:
            if count_as_access and self.metrics is not None:
                if eid in self._prefetched_ids_this_group:
                    self._record_cache_hit(is_prefetch_hit=True)
                    self._prefetch_used_ids_this_group.add(eid)
                    if eid in self._transition_prefetched_ids:
                        self._transition_used_ids.add(int(eid))
                        self._record_metric_counter("transition_prefetch_hit", 1)
                    try:
                        self._cache.unpin(eid)
                        self._cache.promote_hot(eid)
                    except Exception:
                        pass
                else:
                    self._record_cache_hit()
            self._maybe_wait_prefetch(eid)
            return cached

        if count_as_access:
            self._record_cache_miss()
        if count_as_access:
            self._on_demand_access(eid)

        copied_from_host = False

        if eid in self._local_eid_set and self.expert_offload == "cpu":
            _stall_begin = None
            if (not is_prefetch) and bool(count_as_access):
                _stall_begin = self._record_demand_h2d_window_begin()
            gate_up = self.local_gate_up_cpu[eid].to(self.device, non_blocking=True)
            down = self.local_down_cpu[eid].to(self.device, non_blocking=True)
            if _stall_begin is not None:
                self._record_demand_h2d_window_end(_stall_begin)
            copied_from_host = True
        elif eid in self._local_eid_set and self.expert_offload == "gpu":
            gate_up = self.local_gate_up_gpu[eid]
            down = self.local_down_gpu[eid]
            copied_from_host = False
        else:
            gate_up, down = self._get_fused_from_cpu_once(eid)
            copied_from_host = True

        if copied_from_host:
            self._record_h2d_bytes(
                tensor_nbytes(gate_up) + tensor_nbytes(down),
                is_prefetch=is_prefetch,
            )

        self._cache_put_guarded(eid, (gate_up, down), pinned=bool(is_prefetch), prefer_hot=self._should_prefer_hot(eid, is_prefetch))
        if is_prefetch:
            self._record_prefetch_event(eid)
        return gate_up, down

    def _owner_exec_shared_single_sample(self, x_b: torch.Tensor) -> torch.Tensor:
        return self._shared_expert_forward(x_b)

    def _owner_exec_full_mlp(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        w: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        T = int(x.shape[0])
        H = int(x.shape[1])
        out = torch.zeros((T, H), device=x.device, dtype=x.dtype)

        x_bs = x.view(batch_size, seq_len, H)
        idx_bs = idx.view(batch_size, seq_len, self.top_k)
        w_bs = w.view(batch_size, seq_len, self.top_k)

        for b in range(batch_size):
            shared_b = self._owner_exec_shared_single_sample(x_bs[b])
            expert_b = self._experts_owner_exec_single_sample(x_bs[b], idx_bs[b], w_bs[b])
            out[b * seq_len:(b + 1) * seq_len] = (shared_b + expert_b).to(x.dtype)

        return out

    def _experts_owner_exec(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        兼容旧调用点，保留 routed-expert only 版本。
        """
        owner_ranks = SparseContext.get_group_owner_ranks()
        if owner_ranks is None:
            raise RuntimeError("group_owner_ranks missing for owner-exec path")

        T = int(x.shape[0])
        H = int(x.shape[1])
        out = torch.zeros((T, H), device=x.device, dtype=x.dtype)

        x_bs = x.view(batch_size, seq_len, H)
        idx_bs = idx.view(batch_size, seq_len, self.top_k)
        w_bs = w.view(batch_size, seq_len, self.top_k)

        for b in range(batch_size):
            owner_me = self.comm_rank
            if int(owner_ranks[b].item()) != owner_me:
                continue

            x_b = x_bs[b]                 # [S, H]
            idx_b = idx_bs[b]             # [S, K]
            w_b = w_bs[b]                 # [S, K]

            y_b = self._experts_owner_exec_single_sample(x_b, idx_b, w_b)
            out[b * seq_len:(b + 1) * seq_len] = y_b

        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def _experts_owner_exec_single_sample(self, x_b: torch.Tensor, idx_b: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
        S, H = x_b.shape
        final = torch.zeros((S, H), device=x_b.device, dtype=x_b.dtype)

        expert_mask = F.one_hot(idx_b.to(torch.long), num_classes=self.num_experts).permute(2, 1, 0)

        for eid in range(self.num_experts):
            idx_k, tok_idx = torch.where(expert_mask[eid])
            if tok_idx.numel() == 0:
                continue

            cur = x_b[tok_idx]

            if self.experts_kind == "fused":
                gate_up, down = self._get_any_fused_expert_to_device(eid, is_prefetch=False)
                gate_up_out = F.linear(cur, gate_up)
                gate, up = gate_up_out.chunk(2, dim=-1)
                y = F.linear(self.act_fn(gate) * up, down)
            else:
                gw, uw, dw = self._get_any_module_expert_to_device(eid, is_prefetch=False)
                gate = F.linear(cur, gw)
                up = F.linear(cur, uw)
                y = F.linear(self.act_fn(gate) * up, dw)

            y = y * w_b[tok_idx, idx_k].unsqueeze(-1)
            final.index_add_(0, tok_idx, y.to(x_b.dtype))

        return final

    def _compute_rs_cache_group_local_only(
        self,
        x: torch.Tensor,
        g: int,
    ) -> None:
        assert SparseContext.rs_bank is not None
        bank = SparseContext.rs_bank
        end = min(g + self.rs_group, self.num_layers)

        group_topi: Dict[int, torch.Tensor] = {}
        group_topv: Dict[int, torch.Tensor] = {}
        group_value: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x_rs = x.float()
            for l in range(g, end):
                pred = bank[l](x_rs).to(torch.float32)

                if getattr(SparseContext, "rs_bank_task", "router_prob") == "cache_value_rs":
                    group_value[l] = self._postprocess_cache_value_scores(pred)
                    continue

                topv, topi = torch.topk(pred, k=self.top_k, dim=-1)
                if self.norm_topk_prob:
                    topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                w = topv.to(dtype=x.dtype)
                i = topi.to(torch.long)

                SparseContext.routing_cache[l] = (w, i)
                group_topi[l] = i
                group_topv[l] = w

        SparseContext.group_pred_topi = dict(group_topi)
        SparseContext.group_pred_topv = dict(group_topv)
        SparseContext.group_pred_value = dict(group_value)




def _owner_local_select_batch(
    x: Optional[torch.Tensor],
    idx: torch.Tensor,
    full_batch_size: int,
    tensor_name: str,
):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if x.dim() == 0:
        return x

    idx = idx.to(device=x.device, dtype=torch.long).contiguous().view(-1)
    x0 = int(x.shape[0])
    full_batch_size = int(full_batch_size)

    if idx.numel() == 0:
        if x0 not in (1, full_batch_size):
            raise RuntimeError(
                f"owner-local batch select out of range for {tensor_name}: x.shape={tuple(x.shape)} idx.shape={tuple(idx.shape)} full_batch_size={full_batch_size}"
            )
        shape = list(x.shape)
        shape[0] = 0
        return x.new_empty(shape)

    # HF often keeps position_ids as [1, S] even when the logical batch is B.
    # In owner-local mode we must treat that as a broadcast tensor.
    if x0 == 1 and full_batch_size > 1:
        idx_min = int(idx.min().item())
        idx_max = int(idx.max().item())
        if idx_min < 0 or idx_max >= full_batch_size:
            raise RuntimeError(
                f"owner-local batch select out of range for {tensor_name}: x.shape={tuple(x.shape)} idx.shape={tuple(idx.shape)} idx_min={idx_min} idx_max={idx_max} full_batch_size={full_batch_size}"
            )
        expanded = x.expand(full_batch_size, *x.shape[1:])
        return expanded.index_select(0, idx).contiguous()

    if x0 != full_batch_size:
        raise RuntimeError(
            f"owner-local batch select out of range for {tensor_name}: x.shape={tuple(x.shape)} idx.shape={tuple(idx.shape)} full_batch_size={full_batch_size}"
        )

    idx_min = int(idx.min().item())
    idx_max = int(idx.max().item())
    if idx_min < 0 or idx_max >= full_batch_size:
        raise RuntimeError(
            f"owner-local batch select out of range for {tensor_name}: x.shape={tuple(x.shape)} idx.shape={tuple(idx.shape)} idx_min={idx_min} idx_max={idx_max} full_batch_size={full_batch_size}"
        )
    return x.index_select(0, idx).contiguous()




def _owner_local_reconstruct_full_batch_tensor(
    local_tensor: torch.Tensor,
    local_sample_ids: torch.Tensor,
    full_batch_size: int,
    group=None,
) -> torch.Tensor:
    if local_tensor.dim() < 1:
        raise RuntimeError("local_tensor must have batch dimension")
    if not (dist.is_available() and dist.is_initialized()):
        full = local_tensor.new_zeros((int(full_batch_size), *local_tensor.shape[1:]))
        if int(local_sample_ids.numel()) > 0:
            full.index_copy_(0, local_sample_ids.to(device=local_tensor.device, dtype=torch.long), local_tensor)
        return full

    local_sample_ids = local_sample_ids.to(device=local_tensor.device, dtype=torch.long).view(-1)
    local_n = int(local_sample_ids.numel())
    count_t = torch.tensor([local_n], device=local_tensor.device, dtype=torch.long)
    world = dist.get_world_size(group=group)
    gathered_counts = [torch.zeros_like(count_t) for _ in range(world)]
    dist.all_gather(gathered_counts, count_t, group=group)
    counts = [int(t.item()) for t in gathered_counts]
    max_n = max(counts) if counts else 0
    tail_shape = tuple(local_tensor.shape[1:])
    if max_n <= 0:
        return local_tensor.new_zeros((int(full_batch_size), *tail_shape))

    padded_tensor = local_tensor.new_zeros((max_n, *tail_shape))
    if local_n > 0:
        padded_tensor[:local_n] = local_tensor
    padded_ids = torch.full((max_n,), -1, device=local_tensor.device, dtype=torch.long)
    if local_n > 0:
        padded_ids[:local_n] = local_sample_ids

    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world)]
    gathered_ids = [torch.full_like(padded_ids, -1) for _ in range(world)]
    dist.all_gather(gathered_tensors, padded_tensor, group=group)
    dist.all_gather(gathered_ids, padded_ids, group=group)

    full = local_tensor.new_zeros((int(full_batch_size), *tail_shape))
    for ids_r, ten_r, n in zip(gathered_ids, gathered_tensors, counts):
        if n <= 0:
            continue
        ids_use = ids_r[:n].to(dtype=torch.long)
        ten_use = ten_r[:n]
        valid = ids_use >= 0
        if bool(valid.any().item()):
            full.index_copy_(0, ids_use[valid], ten_use[valid])
    return full




def _ensure_cache_object_for_owner_local(past_key_value: Optional[Any]) -> Any:
    if past_key_value is None:
        return DynamicCache()
    return past_key_value

def _pad_past_key_value_slots_for_owner_local(
    attn_module: nn.Module,
    past_key_value: Optional[Any],
    target_layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if past_key_value is None:
        return
    key_cache = getattr(past_key_value, "key_cache", None)
    value_cache = getattr(past_key_value, "value_cache", None)
    if not isinstance(key_cache, list) or not isinstance(value_cache, list):
        return

    num_kv_heads = int(getattr(attn_module, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(attn_module, "head_dim", 0) or 0)
    if num_kv_heads <= 0 or head_dim <= 0:
        return

    while len(key_cache) <= int(target_layer_idx):
        empty_k = torch.empty((0, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
        empty_v = torch.empty((0, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
        key_cache.append(empty_k)
        value_cache.append(empty_v)
def custom_owner_local_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    if not bool(getattr(self, "_sparse_owner_local_runtime", False)):
        return self._sparse_orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    layer_idx = int(getattr(self, "_sparse_layer_idx", -1))
    if layer_idx < 0:
        raise RuntimeError("owner-local decoder layer missing _sparse_layer_idx")

    mlp_block = getattr(self, "mlp", None)
    if mlp_block is None or not hasattr(mlp_block, "plan_group_owner_ranks_from_hidden"):
        raise RuntimeError("owner-local decoder layer requires patched DistributedSparseMoeBlock")
    metrics_obj = getattr(mlp_block, "metrics", None)

    # Unified semantics for the current diagnostic runtime: attention / KV state migration
    # still uses the owner-local path, while MoE execution stays on the standard baseline EP
    # implementation. This keeps the migration-method under test while isolating MoE H2D effects.

    owner_group_size = int(os.environ.get("SPARSE_OWNER_GROUP_SIZE", "0") or 0)
    if owner_group_size <= 0:
        owner_group_size = int(mlp_block.rs_group)
    owner_group_size = max(1, owner_group_size)
    g = (layer_idx // owner_group_size) * owner_group_size
    owner_me = int(mlp_block.comm_rank)
    prev_owner = SparseContext.get_group_owner_ranks()
    prev_owner = prev_owner.clone() if prev_owner is not None else None
    phase = str(getattr(SparseContext, "current_phase", "none") or "none")
    q_len = int(hidden_states.shape[1]) if hidden_states.dim() >= 2 else 0
    is_decode_step = bool(use_cache and q_len == 1)
    if phase != "decode" or (not is_decode_step):
        if _owner_group_log_enabled() and (_owner_group_log_all_ranks() or (dist.is_available() and dist.is_initialized() and getattr(mlp_block, "comm_rank", 0) == 0)):
            _owner_group_log(
                f"BYPASS_BASELINE phase={phase} layer={layer_idx} q_len={q_len} use_cache={int(bool(use_cache))} owner_local_runtime is decode-only at layer level",
            )
        return self._sparse_orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    group_end = min(int(g + owner_group_size), int(getattr(self, "_sparse_num_layers", g + owner_group_size)))
    group_layer_indices = tuple(range(int(g), int(group_end)))

    # Determine whether this call is seeing a full batch or a local owner shard.
    local_sample_ids = SparseContext.get_local_sample_ids(device=hidden_states.device)
    full_batch_size = int(SparseContext.local_full_batch_size) if SparseContext.local_full_batch_size > 0 else int(hidden_states.shape[0])
    is_full_batch_input = (local_sample_ids is None) or (int(hidden_states.shape[0]) == full_batch_size)

    if layer_idx == g:
        if _owner_group_log_enabled() and _owner_group_log_all_ranks():
            _owner_group_log(
                "GROUP_START_ENTER",
                f"phase={phase} layer={layer_idx} group=[{g},{group_end}) rank={mlp_block.rank} comm_rank={mlp_block.comm_rank} input_batch={int(hidden_states.shape[0])} full_batch_hint={full_batch_size} is_full_input={int(is_full_batch_input)}"
            )

        disable_migration = str(os.environ.get("SPARSE_SKVS_DISABLE_MIGRATION", "0")).strip().lower() in ("1", "true", "yes", "on")
        static_no_migration = str(os.environ.get("SPARSE_SKVS_STATIC_OWNER_NO_MIGRATION", "1")).strip().lower() in ("1", "true", "yes", "on")

        reconstruct_ms = 0.0
        if is_full_batch_input:
            full_hidden = hidden_states
            full_batch_size = int(hidden_states.shape[0])
        else:
            t_reconstruct = time.perf_counter()
            full_hidden = _owner_local_reconstruct_full_batch_tensor(
                local_tensor=hidden_states,
                local_sample_ids=local_sample_ids,
                full_batch_size=full_batch_size,
                group=mlp_block.comm_group,
            )
            reconstruct_ms = (time.perf_counter() - t_reconstruct) * 1000.0
            if _owner_group_log_enabled() and _owner_group_log_all_ranks():
                _owner_group_log(
                    "GROUP_START_RECON",
                    f"phase={phase} layer={layer_idx} group=[{g},{group_end}) rank={mlp_block.rank} comm_rank={mlp_block.comm_rank} reconstruct_ms={reconstruct_ms:.3f} full_batch={int(full_hidden.shape[0])}"
                )

        owner_leader_global = int(mlp_block.comm_global_ranks[0])
        t_plan = time.perf_counter()
        if disable_migration and static_no_migration and prev_owner is not None:
            owner_ranks = prev_owner.to(device=full_hidden.device, dtype=torch.long).contiguous()
        else:
            owner_ranks = mlp_block.plan_group_owner_ranks_from_hidden(
                full_hidden,
                prev_owner_ranks=prev_owner,
            )
        plan_ms_local = (time.perf_counter() - t_plan) * 1000.0
        if owner_ranks is None or owner_ranks.dim() != 1 or int(owner_ranks.numel()) != int(full_batch_size):
            raise RuntimeError(
                f"group-start owner plan invalid at layer {layer_idx}: full_hidden.shape={tuple(full_hidden.shape)} owner_ranks.shape={None if owner_ranks is None else tuple(owner_ranks.shape)}"
            )
        owner_ranks = owner_ranks.to(device=full_hidden.device, dtype=torch.long).contiguous()
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(owner_ranks, src=owner_leader_global, group=mlp_block.comm_group)

        changed_owner_count = 0
        if prev_owner is not None:
            changed_owner_count = int((prev_owner.to(device=owner_ranks.device, dtype=torch.long) != owner_ranks).sum().item())
        owner_hist_stats = _owner_group_hist_stats(owner_ranks, mlp_block.comm_world)

        migrate_ms = 0.0
        if prev_owner is not None and changed_owner_count > 0 and bool(getattr(self.self_attn, "_skvs_enabled", False)) and not disable_migration:
            if _owner_group_log_enabled() and _owner_group_log_all_ranks():
                _owner_group_log(
                    "GROUP_START_MIGRATE_BEGIN",
                    f"phase={phase} layer={layer_idx} group=[{g},{group_end}) rank={mlp_block.rank} comm_rank={mlp_block.comm_rank} changed={changed_owner_count} migrate_sparse_cache={int(phase == 'decode')}"
                )
            t_migrate = time.perf_counter()
            migrate_sparse_kv_cache_for_group_change(
                prev_owner_ranks=prev_owner,
                new_owner_ranks=owner_ranks,
                my_rank=owner_me,
                group=mlp_block.comm_group,
                layer_indices=group_layer_indices,
                migrate_sparse_cache=(phase == "decode"),
                migrate_keep_plan=True,
            )
            migrate_ms = (time.perf_counter() - t_migrate) * 1000.0
        SparseContext.set_group_owner_ranks(owner_ranks)
        local_sample_ids = torch.nonzero(owner_ranks.to(device=full_hidden.device, dtype=torch.long) == owner_me, as_tuple=False).flatten()
        local_hidden = full_hidden.index_select(0, local_sample_ids) if int(local_sample_ids.numel()) > 0 else full_hidden.new_empty((0, *full_hidden.shape[1:]))
        SparseContext.set_local_batch(local_sample_ids, int(full_batch_size))

        if metrics_obj is not None and int(mlp_block.comm_rank) == 0:
            metrics_obj.owner_group_count += 1
            metrics_obj.owner_group_reconstruct_ms += float(reconstruct_ms)
            metrics_obj.owner_group_plan_ms += float(plan_ms_local)
            metrics_obj.owner_group_migrate_ms += float(migrate_ms)
            metrics_obj.owner_changed_samples += int(changed_owner_count)
            metrics_obj.owner_empty_ranks_sum += int(owner_hist_stats.get('empty', 0.0))
            metrics_obj.owner_hist_max_sum += float(owner_hist_stats.get('max', 0.0))
            metrics_obj.owner_hist_min_sum += float(owner_hist_stats.get('min', 0.0))
            metrics_obj.owner_hist_cv_sum += float(owner_hist_stats.get('cv', 0.0))
            metrics_obj.owner_imbalance_ratio_sum += float(owner_hist_stats.get('imbalance', 0.0))

        if _owner_group_log_enabled() and int(mlp_block.comm_rank) == 0:
            _owner_group_log(
                "GROUP_START_PLAN",
                f"phase={phase} layer={layer_idx} group=[{g},{group_end}) leader_global_rank={mlp_block.rank} ep_size={mlp_block.comm_world} full_batch={full_batch_size} owner_hist={_owner_group_hist_str(owner_ranks, mlp_block.comm_world)} changed={changed_owner_count} reconstruct_ms={reconstruct_ms:.3f} plan_ms={plan_ms_local:.3f} migrate_ms={migrate_ms:.3f}"
            )
        if _owner_group_log_enabled() and _owner_group_log_all_ranks():
            _owner_group_log(
                "GROUP_START_ASSIGN",
                f"phase={phase} layer={layer_idx} group=[{g},{group_end}) rank={mlp_block.rank} comm_rank={mlp_block.comm_rank} local_n={int(local_sample_ids.numel())} full_batch={full_batch_size}"
            )
    else:
        owner_ranks = SparseContext.get_group_owner_ranks()
        if owner_ranks is None or owner_ranks.dim() != 1:
            raise RuntimeError("owner-local layer requires an active group owner plan")
        expected_local_ids = torch.nonzero(
            owner_ranks.to(device=hidden_states.device, dtype=torch.long) == owner_me,
            as_tuple=False,
        ).flatten()
        if (
            local_sample_ids is None
            or int(local_sample_ids.numel()) != int(expected_local_ids.numel())
            or (int(expected_local_ids.numel()) > 0 and not torch.equal(local_sample_ids.to(expected_local_ids.device), expected_local_ids))
        ):
            local_sample_ids = expected_local_ids
            SparseContext.set_local_batch(local_sample_ids, int(owner_ranks.numel()))
        full_batch_size = int(owner_ranks.numel())
        if int(hidden_states.shape[0]) == int(full_batch_size):
            local_hidden = hidden_states.index_select(0, local_sample_ids) if int(local_sample_ids.numel()) > 0 else hidden_states.new_empty((0, *hidden_states.shape[1:]))
        else:
            if int(hidden_states.shape[0]) != int(local_sample_ids.numel()):
                raise RuntimeError(
                    f"owner-local non-group layer batch mismatch at layer {layer_idx}: hidden_states.shape={tuple(hidden_states.shape)} local_sample_ids.shape={tuple(local_sample_ids.shape)} full_batch_size={full_batch_size}"
                )
            local_hidden = hidden_states

    local_attention_mask = _owner_local_select_batch(attention_mask, local_sample_ids, full_batch_size, "attention_mask")
    local_position_ids = _owner_local_select_batch(position_ids, local_sample_ids, full_batch_size, "position_ids")
    if int(local_hidden.shape[0]) > 0:
        if local_position_ids is not None and int(local_position_ids.shape[0]) == 0:
            local_position_ids = None
        if local_attention_mask is not None and int(local_attention_mask.shape[0]) == 0:
            local_attention_mask = None

    setattr(self.self_attn, "_skvs_owner_local_layer_batch", True)
    setattr(self.self_attn, "_skvs_local_sample_ids", local_sample_ids.detach().to(dtype=torch.long))
    setattr(self.self_attn, "_skvs_prev_owner_ranks", prev_owner)
    setattr(self.self_attn, "_skvs_owner_ranks", owner_ranks)

    self_attn_weights = None
    present_key_value = _ensure_cache_object_for_owner_local(past_key_value) if use_cache else None
    router_logits = None

    try:
        if int(local_hidden.shape[0]) == 0:
            if use_cache:
                present_key_value = _ensure_cache_object_for_owner_local(past_key_value)
        else:
            residual = local_hidden
            local_hidden_norm = self.input_layernorm(local_hidden)
            t_attn = time.perf_counter()
            attn_outputs = self.self_attn(
                hidden_states=local_hidden_norm,
                attention_mask=local_attention_mask,
                position_ids=local_position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            attn_ms = (time.perf_counter() - t_attn) * 1000.0
            if metrics_obj is not None:
                metrics_obj.owner_attention_ms += float(attn_ms)
                metrics_obj.owner_attention_calls += 1
            attn_hidden = attn_outputs[0]
            if output_attentions:
                self_attn_weights = attn_outputs[1]
            if use_cache:
                present_key_value = attn_outputs[2 if output_attentions else 1]
                if present_key_value is None:
                    present_key_value = _ensure_cache_object_for_owner_local(past_key_value)
            local_hidden = residual + attn_hidden

        residual = local_hidden
        local_hidden_norm = self.post_attention_layernorm(local_hidden)
        setattr(self.mlp, "_sparse_local_owner_batch", False)
        t_moe = time.perf_counter()
        mlp_out, router_logits = self.mlp(local_hidden_norm)
        moe_ms = (time.perf_counter() - t_moe) * 1000.0
        if metrics_obj is not None:
            metrics_obj.owner_moe_ms += float(moe_ms)
            metrics_obj.owner_moe_calls += 1
        local_hidden = residual + mlp_out
    finally:
        setattr(self.self_attn, "_skvs_owner_local_layer_batch", False)
        setattr(self.self_attn, "_skvs_prev_owner_ranks", None)
        setattr(self.self_attn, "_skvs_owner_ranks", None)

    is_last_layer = bool(int(layer_idx) == int(getattr(self, "_sparse_num_layers", layer_idx + 1) - 1))
    if is_last_layer:
        full_hidden_out = _owner_local_reconstruct_full_batch_tensor(
            local_tensor=local_hidden,
            local_sample_ids=local_sample_ids,
            full_batch_size=full_batch_size,
            group=mlp_block.comm_group,
        )
        SparseContext.clear_local_batch()
    else:
        full_hidden_out = local_hidden
        SparseContext.set_local_batch(local_sample_ids, full_batch_size)

    full_router_logits = None
    if bool(kwargs.get("output_router_logits", False)) and router_logits is not None:
        seq_len = int(local_hidden.shape[1]) if int(local_hidden.dim()) >= 2 else 1
        local_router = router_logits.view(int(local_sample_ids.numel()), seq_len, router_logits.shape[-1])
        if is_last_layer:
            full_router_logits = _owner_local_reconstruct_full_batch_tensor(
                local_tensor=local_router,
                local_sample_ids=local_sample_ids,
                full_batch_size=full_batch_size,
                group=mlp_block.comm_group,
            ).view(full_batch_size * seq_len, router_logits.shape[-1])
        else:
            full_router_logits = local_router.view(int(local_sample_ids.numel()) * seq_len, router_logits.shape[-1])

    outputs = (full_hidden_out,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    if bool(kwargs.get("output_router_logits", False)):
        outputs += (full_router_logits,)
    return outputs


def custom_late_rs_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    out = self._sparse_orig_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if not _env_flag("SPARSE_RS_TWO_STAGE", "0"):
        return out
    phase = str(getattr(SparseContext, "current_phase", "none") or "none")
    q_len = int(hidden_states.shape[1]) if hidden_states.dim() >= 2 else 0
    if phase != "decode" or q_len != 1:
        return out
    mlp_block = getattr(self, "mlp", None)
    if mlp_block is None or not hasattr(mlp_block, "_compute_and_schedule_late_next_layer_rs"):
        return out
    block_hidden = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
    if not torch.is_tensor(block_hidden):
        return out
    mlp_block._compute_and_schedule_late_next_layer_rs(block_hidden, seq_len=q_len)
    return out

# ---------------- build / patch ----------------
@dataclass(frozen=True)
class CoreBuildConfig:
    model_path: str
    scp_path: str
    rs_dir: str
    dtype: str = "float16"
    expert_offload: str = "cpu"
    expert_cache: int = 0
    cache_policy: str = "lru"  # "lru" | "tiered"
    enable_skvs: bool = False
    skvs_sink: int = 4
    skvs_window: int = 32
    skvs_ratio: float = 0.6
    skvs_min_kv_len: int = 0
    skvs_min_sparse_drop_ratio: float = 0.0
    skvs_guidance_every_k: int = 1
    skvs_guidance_start_kv_len: int = 0
    skvs_allow_prefill_sparse: bool = False
    prompt_len: int = 0
    max_new_tokens: int = 0
    routing_mode: str = "router"
    expert_source: str = "router"   # "router" | "rs"
    placement_mode: str = "scp"
    scp_ep_path: str = ""
    runtime_mode: str = "baseline"
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
    oracle_mode: str = "none"
    oracle_trace_dir: str = ""
    oracle_ignore_cache_capacity: bool = True
    ep_size: int = 0


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
    placement_mode: str = "scp"
    runtime_mode: str = "baseline"
    ep_size: int = 0
    dp_size: int = 1


def install_lightweight_policies(profile, *, prefetch=True, route=True, feedback=True):
    """Prepare local constants once; serving adds no distributed state exchange."""
    from utils.lightweight_policy import LocalPolicy
    if SparseContext.rs_bank_layout != "next_layer" or SparseContext.rs_bank_task != "router_prob":
        raise ValueError("local readiness policy currently requires adjacent-layer probability RS")
    for layer, block in SparseContext.layer_blocks.items():
        sizes = {eid: block._epipe_estimate_expert_nbytes(eid) for eid in block._local_eid_set}
        local_cost = block._topology_route_cost(block.comm_rank, block.comm_rank)
        candidates = {eid for eid in sizes if block.rr_expert_to_rank_cpu[eid] != block.comm_rank
                      and local_cost < block._topology_route_cost(block.comm_rank, block.rr_expert_to_rank_cpu[eid])}
        block._light_policy = LocalPolicy(profile, block.comm_rank, layer, sizes, candidates,
                                          prefetch=prefetch, route=route, feedback=feedback)


def build_model_and_tokenizer(cfg: CoreBuildConfig, env: DistEnv, metrics=None) -> ModelBundle:
    light_enabled = _env_flag("SPARSE_LIGHT_ENABLE", "0")
    if light_enabled and (cfg.routing_mode != "router" or cfg.expert_source != "router"
                          or cfg.planner_mode != "rs_prefetch" or cfg.expert_offload != "cpu"
                          or cfg.cache_policy != "lru" or cfg.oracle_mode != "none" or cfg.enable_skvs):
        raise ValueError("local readiness policy requires original router, RS prefetch, CPU offload, LRU and no oracle/SKVS")
    if light_enabled and any(_env_flag(name, "0") for name in (
            "SPARSE_RS_TWO_STAGE", "SPARSE_GLOBAL_EXPERT_CACHE", "SPARSE_ERS_ENABLE",
            "SPARSE_NODEPATCH_ENABLE", "SPARSE_NODE_GROUPED_DISPATCH", "SPARSE_RANK_GROUPED_DISPATCH",
            "SPARSE_LOCAL_OWNER_BATCH", "SPARSE_EP_FUSED_WEIGHT_DISPATCH")):
        raise ValueError("local readiness policy currently uses ordinary EP and a layer-local expert cache")
    dtype = torch.float16 if cfg.dtype == "float16" else torch.bfloat16
    topology = build_dist_topology(
        env,
        runtime_mode=_normalize_runtime_mode(str(cfg.runtime_mode)),
        ep_size=(int(cfg.ep_size) if int(cfg.ep_size) > 0 else env.world),
    )

    scp_map: Dict[str, int] = {}
    if str(cfg.placement_mode) == "scp":
        scp_path = Path(str(cfg.scp_path)).expanduser()
        if not scp_path.exists():
            raise FileNotFoundError(
                f"placement_mode=scp requires a valid --scp-path, but file does not exist: "
                f"{scp_path.resolve() if scp_path.is_absolute() else scp_path}"
            )
        with open(scp_path, "r", encoding="utf-8") as f:
            scp_map = json.load(f)
    elif str(cfg.placement_mode) == "scp_ep":
        raw_path = str(cfg.scp_ep_path or os.environ.get("SPARSE_SCP_EP_PLACEMENT", "") or cfg.scp_path).strip()
        scp_ep_path = Path(raw_path).expanduser()
        if raw_path == "" or not scp_ep_path.exists():
            raise FileNotFoundError(
                "placement_mode=scp_ep requires --scp-ep-placement or SPARSE_SCP_EP_PLACEMENT; "
                f"missing path: {raw_path or '<empty>'}"
            )
        with open(scp_ep_path, "r", encoding="utf-8") as f:
            scp_map = json.load(f)
    elif str(cfg.scp_path):
        scp_path = Path(str(cfg.scp_path)).expanduser()
        if scp_path.exists():
            with open(scp_path, "r", encoding="utf-8") as f:
                scp_map = json.load(f)

    if str(cfg.placement_mode) == "scp_ep":
        if _env_flag("SPARSE_NODE_GROUPED_DISPATCH", "0") or _env_flag("SPARSE_RANK_GROUPED_DISPATCH", "0"):
            raise RuntimeError("SCP-EP is defined only for standard EP dispatch; disable SPARSE_NODE_GROUPED_DISPATCH and SPARSE_RANK_GROUPED_DISPATCH.")

    attn_impl = _hf_attn_implementation_from_env("eager" if "deepseek" in str(cfg.model_path).lower() else "")
    if _env_flag("SPARSE_FLASH_ATTN_IMPORT_STUB", "0") or (
        "deepseek" in str(cfg.model_path).lower() and attn_impl in ("", "eager", "sdpa")
    ):
        _install_flash_attn_import_stub()

    hf_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if attn_impl:
        hf_kwargs["attn_implementation"] = attn_impl

    try:
        hf_config = AutoConfig.from_pretrained(cfg.model_path, **hf_kwargs)
    except TypeError:
        hf_config = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
    if attn_impl:
        try:
            setattr(hf_config, "_attn_implementation", attn_impl)
        except Exception:
            pass
        try:
            setattr(hf_config, "attn_implementation", attn_impl)
        except Exception:
            pass
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    hidden_size = int(getattr(hf_config, "hidden_size"))

    top_k_cfg = int(
        getattr(
            hf_config,
            "num_experts_per_tok",
            getattr(hf_config, "num_experts_per_token", getattr(hf_config, "top_k", 4)),
        )
    )
    num_experts_cfg = int(
        getattr(
            hf_config,
            "num_experts",
            getattr(hf_config, "n_routed_experts", getattr(hf_config, "num_routed_experts", 0)),
        )
    )

    skvs_start_kv_len = max(int(os.environ.get("SPARSE_SKVS_START_KV_LEN", "0") or 0), 0)
    skvs_reachable = True
    if bool(cfg.enable_skvs) and skvs_start_kv_len > 0 and int(cfg.prompt_len) > 0 and int(cfg.max_new_tokens) > 0:
        max_runtime_kv_len = int(cfg.prompt_len) + max(int(cfg.max_new_tokens) - 1, 0)
        skvs_reachable = max_runtime_kv_len >= skvs_start_kv_len
    effective_enable_skvs = bool(cfg.enable_skvs) and bool(skvs_reachable)

    SparseContext.initialize(num_layers, rs_group=int(cfg.rs_group), num_experts=int(num_experts_cfg))
    SparseContext.skvs_enabled = bool(effective_enable_skvs)
    skvs_apply_layer_start = max(int(os.environ.get("SPARSE_SKVS_APPLY_LAYER_START", "1") or 1), 1)
    skvs_apply_layer_end = int(os.environ.get("SPARSE_SKVS_APPLY_LAYER_END", str(num_layers - 1)) or (num_layers - 1))
    skvs_apply_layer_end = min(max(skvs_apply_layer_end, skvs_apply_layer_start), num_layers - 1)
    owner_local_enabled_env = str(os.environ.get("SPARSE_SKVS_OWNER_LOCAL", "1")).strip().lower()
    owner_local_runtime = bool(effective_enable_skvs) and owner_local_enabled_env not in ("0", "false", "no", "off")
    SparseContext.set_runtime_decode_local(owner_local_runtime)

    if effective_enable_skvs:
        set_skvs_config(
            _SKVSConfig(
                h2o_sink=cfg.skvs_sink,
                h2o_window=cfg.skvs_window,
                l0_guidance_ratio=cfg.skvs_ratio,
                min_kv_len=int(cfg.skvs_min_kv_len),
                min_sparse_drop_ratio=float(cfg.skvs_min_sparse_drop_ratio),
                guidance_every_k=int(cfg.skvs_guidance_every_k),
                guidance_start_kv_len=int(cfg.skvs_guidance_start_kv_len),
            )
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=hf_config,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            **hf_kwargs,
        )
    except TypeError:
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

        patch_skvs_layer = bool(effective_enable_skvs) and (
            i == 0 or (int(skvs_apply_layer_start) <= int(i) <= int(skvs_apply_layer_end))
        )
        if patch_skvs_layer:
            orig_forward = layer.self_attn.forward
            layer.self_attn._skvs_orig_forward = orig_forward
            layer.self_attn.forward = types.MethodType(custom_skvs_attention_forward, layer.self_attn)

            layer.self_attn._sparse_env_rank = topology.ep_rank
            layer.self_attn._sparse_env_world = topology.ep_size
            layer.self_attn._skvs_owner_rank = -1
            layer.self_attn._skvs_owner_ranks = None
            layer.self_attn._skvs_layer_idx = i

            # Unified baseline+method structure: once SKVS migration runtime is enabled,
            # both prefill and decode use the same owner-local method path.
            allow_prefill_sparse = bool(cfg.skvs_allow_prefill_sparse)
            layer.self_attn._skvs_decode_only = True
            layer.self_attn._skvs_allow_prefill_sparse = allow_prefill_sparse
            layer.self_attn._skvs_owner_local_runtime = owner_local_runtime
            layer.self_attn._skvs_batch_limit = 0
            layer.self_attn._skvs_transport_impl = "all_gather_owner_reduce_batched"
            layer.self_attn._skvs_token_owner_mode = "pos_mod_ep_group"
            layer.self_attn._skvs_debug_transport = False
            layer.self_attn._skvs_kv_owner_fn = kv_owner_by_position
            layer.self_attn._skvs_debug_logs = False
            layer.self_attn._skvs_enabled = bool(effective_enable_skvs)

            SparseContext.register_layer_attn(i, layer.self_attn)

        if looks_like_sparse_moe_block(layer.mlp):
            sparse_layers += 1

            E = int(getattr(layer.mlp, "num_experts", getattr(layer.mlp, "n_routed_experts", num_experts_cfg if num_experts_cfg > 0 else 0)))
            K = int(getattr(layer.mlp, "top_k", getattr(layer.mlp, "num_experts_per_tok", top_k_cfg)))

            if E <= 0:
                if experts_are_fused(layer.mlp.experts):
                    E = int(layer.mlp.experts.gate_up_proj.shape[0])
                elif experts_are_modulelist(layer.mlp.experts):
                    E = int(len(list(layer.mlp.experts)))
                else:
                    raise RuntimeError(f"Cannot infer num_experts at layer {i}")

            num_experts = E
            top_k = K
            if int(getattr(SparseContext, "num_experts", 0) or 0) <= 0:
                SparseContext.num_experts = int(E)

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
                expert_source=str(cfg.expert_source),
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
                oracle_mode=str(cfg.oracle_mode),
                oracle_ignore_cache_capacity=bool(cfg.oracle_ignore_cache_capacity),
                placement_mode=str(cfg.placement_mode),
                cache_policy=str(cfg.cache_policy),
                runtime_mode=_normalize_runtime_mode(str(cfg.runtime_mode)),
                topology=topology,
            )
            SparseContext.register_layer_block(i, layer.mlp)
        else:
            layer.mlp = layer.mlp.to(env.device, dtype=dtype)

        if owner_local_runtime:
            layer._sparse_orig_forward = layer.forward
            layer._sparse_owner_local_runtime = True
            layer._sparse_layer_idx = i
            layer.forward = types.MethodType(custom_owner_local_decoder_layer_forward, layer)
        elif _env_flag("SPARSE_RS_TWO_STAGE", "0"):
            layer._sparse_orig_forward = layer.forward
            layer._sparse_late_rs_runtime = True
            layer._sparse_layer_idx = i
            layer.forward = types.MethodType(custom_late_rs_decoder_layer_forward, layer)

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
            kwargs["output_hidden_states"] = True
            out = orig_forward(*args, **kwargs)

            if getattr(out, "hidden_states", None) is None:
                raise RuntimeError("CPU lm_head path requires output_hidden_states=True.")

            last_hidden = out.hidden_states[-1]
            last_hidden_cpu = last_hidden.to("cpu")
            logits_cpu = F.linear(last_hidden_cpu, self.lm_head.weight).float()
            out.logits = logits_cpu
            return out

        model.forward = types.MethodType(forward_cpu_lm_head, model)

    model.eval()


    SparseContext.set_oracle_mode(str(cfg.oracle_mode), str(cfg.oracle_trace_dir))
    if str(cfg.oracle_mode) == "replay":
        SparseContext.load_oracle_trace(env.rank)

    if env.rank == 0 and str(cfg.placement_mode) == "scp":
        print("[SCP] effective per-layer placement histograms:", flush=True)
        for l in range(num_layers):
            blk = SparseContext.layer_blocks.get(int(l), None)
            if blk is None:
                continue
            hist = torch.bincount(
                blk.expert_to_rank.detach().to("cpu", dtype=torch.long),
                minlength=int(topology.ep_size),
            ).tolist()
            warn = " WARNING_EMPTY_RANK" if any(int(x) == 0 for x in hist) else ""
            print(f"[SCP] layer={l} rank_counts={hist}{warn}", flush=True)

    if env.rank == 0 and str(cfg.placement_mode) == "scp_ep":
        raw_path = str(cfg.scp_ep_path or os.environ.get("SPARSE_SCP_EP_PLACEMENT", "") or cfg.scp_path)
        print(f"[SCP-EP] loaded placement {raw_path}", flush=True)
        print("[SCP-EP] standard EP dispatch path", flush=True)
        print(f"[SCP-EP] world_size={topology.ep_size}", flush=True)
        print(f"[SCP-EP] num_layers={num_layers}", flush=True)
        print(f"[SCP-EP] num_experts={num_experts}", flush=True)
        print(f"[SCP-EP] placement_format={scp_map.get('format', 'legacy_owner_map') if isinstance(scp_map, dict) else 'unknown'}", flush=True)
        print(
            f"[SCP-EP] replica_choice_cost_matrix={'loaded' if isinstance(scp_map, dict) and scp_map.get('replica_choice_cost_matrix') is not None else 'fallback_topology_tiers'}",
            flush=True,
        )
        print(f"[SCP-EP] NGD disabled={not _env_flag('SPARSE_NODE_GROUPED_DISPATCH', '0')}", flush=True)
        for l in range(num_layers):
            blk = SparseContext.layer_blocks.get(int(l), None)
            if blk is None:
                continue
            hist = torch.bincount(
                blk.expert_to_rank.detach().to("cpu", dtype=torch.long),
                minlength=int(topology.ep_size),
            ).tolist()
            warn = " WARNING_EMPTY_RANK" if any(int(x) == 0 for x in hist) else ""
            print(f"[SCP-EP] layer={l} rank_counts={hist}{warn}", flush=True)

    if env.rank == 0:
        print(
            f"[Runtime] mode={cfg.runtime_mode} skvs={cfg.enable_skvs} placement={cfg.placement_mode} "
            f"ep_size={topology.ep_size} dp_size={topology.dp_size} "
            f"planner={cfg.planner_mode} expert_source={cfg.expert_source}",
            flush=True,
        )
        print(
            f"[Routing] exec={cfg.routing_mode} expert_source={cfg.expert_source} "
            f"planner={cfg.planner_mode} placement={cfg.placement_mode} "
            f"(rs_dir={cfg.rs_dir})",
            flush=True,
        )
        print(f"[Patch] sparse_moe_layers={sparse_layers}", flush=True)
        if effective_enable_skvs:
            skvs_ref_attn = SparseContext.layer_attn_modules.get(0, layer.self_attn)
            print(
                f"[SKVS] decode_only={getattr(skvs_ref_attn, '_skvs_decode_only', True)} "
                f"batch_limit={getattr(skvs_ref_attn, '_skvs_batch_limit', 0)} "
                f"transport={getattr(skvs_ref_attn, '_skvs_transport_impl', 'unknown')} "
                f"kv_owner=pos_mod_ep_group",
                flush=True,
            )
            print(
                f"[SKVS] apply_layers=[{skvs_apply_layer_start},{skvs_apply_layer_end}] "
                f"(layer0 is always patched for guidance)",
                flush=True,
            )
        elif cfg.enable_skvs and env.rank == 0:
            max_runtime_kv_len = int(cfg.prompt_len) + max(int(cfg.max_new_tokens) - 1, 0)
            print(
                f"[SKVS] patch skipped: start_kv_len={skvs_start_kv_len} "
                f"> max_runtime_kv_len={max_runtime_kv_len}",
                flush=True,
            )
        if cfg.routing_mode == "rs" or cfg.planner_mode == "rs_prefetch":
            print(
                f"[RS] group={cfg.rs_group} task={getattr(SparseContext, 'rs_bank_task', 'unknown')} "
                f"debug={cfg.rs_debug} prefetch_top_m={cfg.prefetch_top_m}",
                flush=True,
            )
            if _env_flag("SPARSE_RS_TWO_STAGE", "0"):
                print(
                    f"[RS_TWO_STAGE] enabled=1 late_task={getattr(SparseContext, 'late_rs_bank_task', 'none')} "
                    f"late_topk={os.environ.get('SPARSE_RS_LATE_TOPK', '6')} "
                    f"protect_mode={os.environ.get('SPARSE_PREFETCH_PROTECT_MODE', 'default')} "
                    f"late_rs_dir={os.environ.get('SPARSE_LATE_RS_DIR', '')}",
                    flush=True,
                )
        if str(cfg.oracle_mode) != "none":
            print(
                f"[Oracle] mode={cfg.oracle_mode} trace_dir={cfg.oracle_trace_dir} "
                f"ignore_cache_capacity={cfg.oracle_ignore_cache_capacity}",
                flush=True,
            )

    # ---- force materialization + fail-fast check on every rank ----
    torch.cuda.synchronize(env.device)
    _probe = torch.empty((1,), device=env.device, dtype=torch.float32)
    _probe.fill_(float(env.rank))

    alloc_mb = torch.cuda.memory_allocated(env.device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(env.device) / (1024 ** 2)

    print(
        f"[MODEL_READY] rank={env.rank} local_rank={env.local_rank} "
        f"device={env.device} alloc_mb={alloc_mb:.1f} reserved_mb={reserved_mb:.1f}",
        flush=True,
    )

    dist.barrier(device_ids=[env.local_rank])

    all_alloc_mb = _gather_floats(alloc_mb, env.device)
    all_reserved_mb = _gather_floats(reserved_mb, env.device)

    if env.rank == 0:
        print(
            f"[MODEL_READY_ALL] alloc_mb={list(map(lambda x: round(x, 1), all_alloc_mb))} "
            f"reserved_mb={list(map(lambda x: round(x, 1), all_reserved_mb))}",
            flush=True,
        )

    # Check the tensors that must be resident rather than assuming a minimum
    # model size: a valid small MoE can occupy much less than one GB.
    resident_modules = [("embed_tokens", model.model.embed_tokens), ("norm", model.model.norm)]
    if lm_head_on_cuda:
        resident_modules.append(("lm_head", model.lm_head))
    for i, layer in enumerate(model.model.layers):
        for name in ("self_attn", "input_layernorm", "post_attention_layernorm"):
            resident_modules.append((f"layers.{i}.{name}", getattr(layer, name)))
        if isinstance(layer.mlp, DistributedSparseMoeBlock):
            for name in ("gate", "shared_expert", "shared_expert_gate", "shared_experts"):
                module = getattr(layer.mlp, name, None)
                if module is not None:
                    resident_modules.append((f"layers.{i}.mlp.{name}", module))
        else:
            resident_modules.append((f"layers.{i}.mlp", layer.mlp))
    misplaced = [f"{name}.{key}: {tensor.device}" for name, module in resident_modules
                 for key, tensor in list(module.named_parameters()) + list(module.named_buffers())
                 if tensor.device != env.device]
    local_ready = torch.tensor([int(not misplaced and alloc_mb > 0)], device=env.device)
    all_ready = [torch.empty_like(local_ready) for _ in range(env.world)]
    dist.all_gather(all_ready, local_ready)
    bad_ranks = [i for i, ready in enumerate(all_ready) if not bool(ready.item())]
    if len(bad_ranks) > 0:
        raise RuntimeError(
            f"Some ranks did not materialize model on their local GPU: bad_ranks={bad_ranks}, "
            f"alloc_mb={all_alloc_mb}, local_misplaced={misplaced[:16]}"
        )

    if env.rank == 0:
        print(f"[Runtime] {cfg.runtime_mode} forward active", flush=True)

    if light_enabled:
        from utils.lightweight_policy import CostProfile
        path = os.environ.get("SPARSE_LIGHT_COST_PROFILE", "").strip()
        if not path:
            raise ValueError("local readiness policy requires SPARSE_LIGHT_COST_PROFILE")
        first = next(iter(SparseContext.layer_blocks.values()))
        install_lightweight_policies(CostProfile(path, first.comm_world),
                                     prefetch=_env_flag("SPARSE_LIGHT_PREFETCH", "1"),
                                     route=_env_flag("SPARSE_LIGHT_ROUTE", "1"),
                                     feedback=_env_flag("SPARSE_LIGHT_FEEDBACK", "1"))

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
        placement_mode=str(cfg.placement_mode),
        runtime_mode=_normalize_runtime_mode(str(cfg.runtime_mode)),
        ep_size=int(topology.ep_size),
        dp_size=int(topology.dp_size),
    )

# ---------------- perf decode ----------------
def _ensure_pending_demand_h2d_events(metrics: Optional[Metrics]) -> None:
    if metrics is None:
        return
    if metrics._pending_demand_h2d_events is None:
        metrics._pending_demand_h2d_events = []


def _append_demand_h2d_event(metrics: Optional[Metrics], start_ev: torch.cuda.Event, end_ev: torch.cuda.Event, layer_idx: int = -1) -> None:
    if metrics is None:
        return
    _ensure_pending_demand_h2d_events(metrics)
    phase = str(getattr(SparseContext, "current_phase", "none") or "none")
    runtime_call_id = int(getattr(SparseContext, "runtime_call_id", -1))
    layer_idx = int(layer_idx)
    metrics._pending_demand_h2d_events.append((start_ev, end_ev, phase, layer_idx, runtime_call_id))
    metrics.demand_h2d_event_count += 1
    if metrics.per_layer is not None and layer_idx >= 0:
        lp = metrics.per_layer.setdefault(layer_idx, LayerPerf())
        lp.demand_h2d_event_count += 1


def _append_prefetch_h2d_event(metrics: Optional[Metrics], start_ev: torch.cuda.Event, end_ev: torch.cuda.Event, layer_idx: int = -1) -> None:
    if metrics is None:
        return
    if metrics._pending_prefetch_h2d_events is None:
        metrics._pending_prefetch_h2d_events = []
    layer_idx = int(layer_idx)
    metrics._pending_prefetch_h2d_events.append((start_ev, end_ev, layer_idx))
    metrics.prefetch_h2d_event_count += 1
    if metrics.per_layer is not None and layer_idx >= 0:
        lp = metrics.per_layer.setdefault(layer_idx, LayerPerf())
        lp.prefetch_h2d_event_count += 1


def _append_prefetch_wait_event(metrics: Optional[Metrics], start_ev: torch.cuda.Event, end_ev: torch.cuda.Event, layer_idx: int = -1) -> None:
    if metrics is None:
        return
    if metrics._pending_prefetch_wait_events is None:
        metrics._pending_prefetch_wait_events = []
    layer_idx = int(layer_idx)
    metrics._pending_prefetch_wait_events.append((start_ev, end_ev, layer_idx))
    metrics.prefetch_wait_event_count += 1
    if metrics.per_layer is not None and layer_idx >= 0:
        lp = metrics.per_layer.setdefault(layer_idx, LayerPerf())
        lp.prefetch_wait_event_count += 1


def _finalize_pending_demand_h2d_events(metrics: Optional[Metrics], device: torch.device) -> None:
    if metrics is None:
        return
    pending = metrics._pending_demand_h2d_events
    if not pending:
        return

    # 统一在阶段边界同步一次，避免在 miss 热路径中逐次同步破坏真实系统行为。
    torch.cuda.synchronize(device)

    total_ms = 0.0
    prefill_ms = 0.0
    decode_ms = 0.0
    sim_total_ms = 0.0
    sim_prefill_ms = 0.0
    sim_decode_ms = 0.0
    sim_extra_ms = 0.0
    h2d_scale = _sim_node_h2d_scale() if _sim_heterogeneity_enabled() else 1.0
    for item in pending:
        if len(item) == 5:
            start_ev, end_ev, phase, layer_idx, event_runtime_call_id = item
        elif len(item) == 4:
            start_ev, end_ev, phase, layer_idx = item
            event_runtime_call_id = int(getattr(SparseContext, "runtime_call_id", -1))
        else:
            start_ev, end_ev, phase = item
            layer_idx = -1
            event_runtime_call_id = int(getattr(SparseContext, "runtime_call_id", -1))
        ms = float(start_ev.elapsed_time(end_ev))
        sim_ms = ms * max(float(h2d_scale), 1e-6)
        extra_ms = max(0.0, sim_ms - ms)
        total_ms += ms
        sim_total_ms += sim_ms
        sim_extra_ms += extra_ms
        if phase == "prefill":
            prefill_ms += ms
            sim_prefill_ms += sim_ms
        elif phase == "decode":
            decode_ms += ms
            sim_decode_ms += sim_ms
            metrics.sampled_demand_h2d_exposed_decode_ms += ms
            cp = _hclm_cp_profiler()
            step_id = int(event_runtime_call_id)
            if cp.should_sample(step_id) and int(layer_idx) >= 0:
                end_ns = _hclm_cp_now_ns()
                cp.record_component(
                    step_id,
                    int(layer_idx),
                    "h2d_copy",
                    end_ns - int(max(float(ms), 0.0) * 1_000_000.0),
                    end_ns,
                    extra={
                        "demand_h2d_expert_count": 1,
                        "demand_h2d_wait_ms": float(ms),
                    },
                )
        if metrics.per_layer is not None and int(layer_idx) >= 0:
            lp = metrics.per_layer.setdefault(int(layer_idx), LayerPerf())
            lp.demand_h2d_exposed_ms += ms
            lp.sim_demand_h2d_exposed_ms += sim_ms
            lp.sim_h2d_extra_sleep_ms += extra_ms
            if phase == "prefill":
                lp.demand_h2d_exposed_prefill_ms += ms
                lp.sim_demand_h2d_exposed_prefill_ms += sim_ms
            elif phase == "decode":
                lp.demand_h2d_exposed_decode_ms += ms
                lp.sim_demand_h2d_exposed_decode_ms += sim_ms
                lp.sampled_demand_h2d_exposed_decode_ms += ms

    metrics.demand_h2d_exposed_ms += total_ms
    metrics.demand_h2d_exposed_prefill_ms += prefill_ms
    metrics.demand_h2d_exposed_decode_ms += decode_ms
    metrics.sim_demand_h2d_exposed_ms += sim_total_ms
    metrics.sim_demand_h2d_exposed_prefill_ms += sim_prefill_ms
    metrics.sim_demand_h2d_exposed_decode_ms += sim_decode_ms
    metrics.sim_h2d_extra_sleep_ms += sim_extra_ms
    if sim_extra_ms > 0.0 and _sim_heterogeneity_sleep_enabled():
        time.sleep(sim_extra_ms / 1000.0)
    metrics._pending_demand_h2d_events = []


def _finalize_pending_prefetch_events(metrics: Optional[Metrics], device: torch.device) -> None:
    if metrics is None:
        return
    pending_h2d = metrics._pending_prefetch_h2d_events or []
    pending_wait = metrics._pending_prefetch_wait_events or []
    if not pending_h2d and not pending_wait:
        return
    torch.cuda.synchronize(device)

    for item in pending_h2d:
        if len(item) == 3:
            start_ev, end_ev, layer_idx = item
        else:
            start_ev, end_ev = item
            layer_idx = -1
        ms = float(start_ev.elapsed_time(end_ev))
        metrics.prefetch_h2d_copy_ms += ms
        metrics.prefetch_completed_count += 1
        if metrics.per_layer is not None and int(layer_idx) >= 0:
            lp = metrics.per_layer.setdefault(int(layer_idx), LayerPerf())
            lp.prefetch_h2d_copy_ms += ms
            lp.prefetch_completed_count += 1
    for item in pending_wait:
        if len(item) == 3:
            start_ev, end_ev, layer_idx = item
        else:
            start_ev, end_ev = item
            layer_idx = -1
        ms = float(start_ev.elapsed_time(end_ev))
        metrics.prefetch_wait_ms += ms
        metrics.sampled_prefetch_wait_ms += ms
        if metrics.per_layer is not None and int(layer_idx) >= 0:
            lp = metrics.per_layer.setdefault(int(layer_idx), LayerPerf())
            lp.prefetch_wait_ms += ms
            lp.sampled_prefetch_wait_ms += ms

    metrics._pending_prefetch_h2d_events = []
    metrics._pending_prefetch_wait_events = []


def _finalize_pending_epipe_timing_events(metrics: Optional[Metrics], device: torch.device) -> None:
    if metrics is None:
        return
    pending = getattr(metrics, "_pending_epipe_timing_events", None) or []
    if not pending:
        return
    torch.cuda.synchronize(device)
    for h2d_start, h2d_end, wait_start, wait_end, layer_idx in pending:
        try:
            h2d_ms = float(h2d_start.elapsed_time(h2d_end))
        except Exception:
            h2d_ms = 0.0
        try:
            wait_ms = float(wait_start.elapsed_time(wait_end))
        except Exception:
            wait_ms = 0.0
        metrics.epipe_demand_h2d_ms += h2d_ms
        metrics.epipe_wait_ms += wait_ms
        metrics.epipe_hidden_h2d_ms_est += max(0.0, h2d_ms - wait_ms)
        metrics.sampled_epipe_wait_ms += wait_ms
        metrics.sampled_epipe_hidden_h2d_ms += max(0.0, h2d_ms - wait_ms)
        if metrics.per_layer is not None and int(layer_idx) >= 0:
            lp = metrics.per_layer.setdefault(int(layer_idx), LayerPerf())
            lp.epipe_demand_h2d_ms += h2d_ms
            lp.epipe_wait_ms += wait_ms
            lp.epipe_hidden_h2d_ms_est += max(0.0, h2d_ms - wait_ms)
            lp.sampled_epipe_wait_ms += wait_ms
            lp.sampled_epipe_hidden_h2d_ms += max(0.0, h2d_ms - wait_ms)
    metrics._pending_epipe_timing_events = []


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


def _normalize_past_key_values_for_runtime(past):
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return past
    if past is None:
        return None
    if isinstance(past, tuple):
        try:
            return DynamicCache.from_legacy_cache(past)
        except Exception:
            return past
    return past


def _unwrap_causal_lm_for_logits(model):
    raw_model = getattr(model, "module", model)
    base_model = getattr(raw_model, "model", None)
    lm_head = getattr(raw_model, "lm_head", None)
    if base_model is None or lm_head is None:
        return None, None
    return base_model, lm_head


@torch.no_grad()
def greedy_decode_profiled(model, input_ids, attention_mask, max_new_tokens: int, env: DistEnv, metrics, measure_ttft: bool):
    model.eval()
    torch.cuda.reset_peak_memory_stats(env.device)
    hclm_cp = _hclm_cp_create_from_env(env.rank, env.world)
    _hclm_cp_set_global(hclm_cp)

    SparseContext.clear_group_plan()

    SparseContext.current_phase = "prefill"
    SparseContext.begin_runtime_call(0)
    if SparseContext.oracle_mode == "replay":
        SparseContext.begin_oracle_call(0)
        SparseContext.clear_oracle_runtime_active()
    elif SparseContext.oracle_mode == "record":
        SparseContext.begin_oracle_call(0)

    _phase_trace(f"PREFILL_BEGIN batch={int(input_ids.shape[0])} prompt_len={int(input_ids.shape[1])} use_cache=1")
    prefill_logits = None
    with wall_timer_s() as wt:
        if _env_flag("SPARSE_PREFILL_LAST_LOGITS_ONLY", "0"):
            base_model, lm_head = _unwrap_causal_lm_for_logits(model)
        else:
            base_model, lm_head = None, None
        if base_model is not None and lm_head is not None:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state[:, -1:, :]
            prefill_logits = F.linear(last_hidden, lm_head.weight, getattr(lm_head, "bias", None)).float()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        if measure_ttft:
            torch.cuda.synchronize(env.device)
    _phase_trace("PREFILL_END")
    SparseContext.clear_oracle_runtime_all()
    SparseContext.end_oracle_call()
    _finalize_pending_demand_h2d_events(metrics, env.device)
    _finalize_pending_prefetch_events(metrics, env.device)
    _finalize_pending_epipe_timing_events(metrics, env.device)
    metrics.wall_prefill_ms = wt() * 1000.0
    _consume_skvs_accounting(metrics)


    if _env_flag("SPARSE_PREFILL_DECODE_WARMUP", "0"):
        t_warmup = time.perf_counter()
        DistributedSparseMoeBlock.warmup_all_caches_from_prefill()
        if torch.cuda.is_available():
            torch.cuda.synchronize(env.device)
        if metrics is not None:
            warmup_ms = (time.perf_counter() - t_warmup) * 1000.0
            metrics.warmup_time_ms += warmup_ms
            metrics.wall_prefill_ms += warmup_ms
            if measure_ttft:
                metrics.wall_ttft_ms = metrics.wall_prefill_ms
        _finalize_pending_prefetch_events(metrics, env.device)
        _finalize_pending_epipe_timing_events(metrics, env.device)
    SparseContext.clear_prefill_expert_counts()

    if _env_flag("SPARSE_SCP_TRANSIENT_PREFILL_CACHE", "0") or _env_flag("SPARSE_EPIPE_DECODE_DEMOTE_PREFILL_TRANSIENT", "1"):
        DistributedSparseMoeBlock.demote_prefill_scp_transients_for_decode(metrics)
    DistributedSparseMoeBlock.handle_epipe_prefill_extra_cache_decode_boundary(metrics)

    if _env_flag("SPARSE_SCP_CLEAR_CACHE_BEFORE_DECODE", "0"):
        DistributedSparseMoeBlock.clear_all_expert_caches()
        if torch.cuda.is_available():
            torch.cuda.synchronize(env.device)

    past = _normalize_past_key_values_for_runtime(outputs.past_key_values)
    logits_for_next = prefill_logits[:, -1, :] if prefill_logits is not None else outputs.logits[:, -1, :]
    next_token = torch.argmax(logits_for_next, dim=-1, keepdim=True)

    generated = [next_token]
    decode_wall_total_s = 0.0
    cur_ids = next_token
    cur_attn_mask = attention_mask
    if measure_ttft:
        metrics.wall_ttft_ms = metrics.wall_prefill_ms

    for _ in range(max_new_tokens - 1):
        if cur_attn_mask is not None:
            next_mask_col = torch.ones((cur_attn_mask.shape[0], 1), device=cur_attn_mask.device, dtype=cur_attn_mask.dtype)
            cur_attn_mask = torch.cat([cur_attn_mask, next_mask_col], dim=1)
        call_id = len(generated)
        SparseContext.current_phase = "decode"
        SparseContext.begin_runtime_call(call_id)
        if SparseContext.oracle_mode == "replay":
            SparseContext.begin_oracle_call(call_id)
            SparseContext.clear_oracle_runtime_active()
        elif SparseContext.oracle_mode == "record":
            SparseContext.begin_oracle_call(call_id)
        _phase_trace(f"DECODE_STEP_BEGIN step={call_id} batch={int(cur_ids.shape[0])}")
        hclm_cp.start_step(call_id, batch=int(cur_ids.shape[0]), seq_len=int(cur_ids.shape[1]))
        with wall_timer_s() as wt_step:
            out = model(
                input_ids=cur_ids,
                attention_mask=cur_attn_mask,
                use_cache=True,
                past_key_values=past,
                return_dict=True,
            )
        _phase_trace(f"DECODE_STEP_END step={call_id}")
        SparseContext.clear_oracle_runtime_active()
        SparseContext.end_oracle_call()
        _finalize_pending_demand_h2d_events(metrics, env.device)
        _finalize_pending_prefetch_events(metrics, env.device)
        _finalize_pending_epipe_timing_events(metrics, env.device)
        hclm_cp.end_step(call_id)

        step_s = wt_step()
        decode_wall_total_s += step_s
        if metrics.decode_step_stats is not None:
            metrics.decode_step_stats.add_ms(step_s * 1000.0)
        _consume_skvs_accounting(metrics)

        past = _normalize_past_key_values_for_runtime(out.past_key_values)
        cur_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur_ids)


    metrics.wall_decode_ms = decode_wall_total_s * 1000.0
    if _env_flag("SPARSE_SCP_TRANSIENT_PREFILL_CACHE", "0") or _env_flag("SPARSE_EPIPE_DECODE_DEMOTE_PREFILL_TRANSIENT", "1"):
        DistributedSparseMoeBlock.record_prefill_scp_transient_decode_end(metrics)
    DistributedSparseMoeBlock.record_epipe_prefill_extra_cache_decode_end(metrics)
    out_ids = torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    metrics.peak_allocated_mb = torch.cuda.max_memory_allocated(env.device) / (1024 ** 2)
    metrics.peak_reserved_mb = torch.cuda.max_memory_reserved(env.device) / (1024 ** 2)

    if SparseContext.oracle_mode == "record":
        SparseContext.save_oracle_trace(env.rank)

    SparseContext.current_phase = "none"
    SparseContext.clear_group_plan()
    hclm_cp.close()
    _hclm_cp_set_global(_HCLMCPNoOp())
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
