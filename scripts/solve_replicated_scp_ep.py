#!/usr/bin/env python3
"""Profile-calibrated replicated SCP-EP solver.

This solver intentionally models replicated expert placement for the standard EP
runtime.  It uses real routing demand A[l, source_rank, expert] plus a calibrated
rank-pair communication model.  Expert replicas are executable logical copies;
they do NOT imply GPU residency.  Cache/H2D risk is reported separately and must
be validated with runtime/cache replay before claiming end-to-end benefit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scp_cache_replay import (
        evaluate_grouped_comm_tail,
        evaluate_profile_placement_with_cache_replay,
        grouped_trace_indices,
        load_profile_with_trace,
    )
except Exception:  # pragma: no cover - allows legacy objectives without trace/replay deps.
    evaluate_grouped_comm_tail = None
    evaluate_profile_placement_with_cache_replay = None
    grouped_trace_indices = None
    load_profile_with_trace = None


@dataclass(frozen=True)
class CommModel:
    world: int
    rank_to_node: Dict[int, int]
    dispatch_item_bytes: float
    combine_item_bytes: float
    hidden_bytes: float
    router_weight_bytes: float
    expert_id_bytes: float
    dispatch_item_cost: np.ndarray  # ms or relative, source -> replica
    combine_item_cost: np.ndarray   # ms or relative, replica -> source
    total_item_cost: np.ndarray     # dispatch[source, replica] + combine[replica, source]
    dispatch_alpha_ms: np.ndarray   # source -> replica, event-level only
    combine_alpha_ms: np.ndarray    # replica -> source, event-level only
    dispatch_bw_MBps: np.ndarray
    combine_bw_MBps: np.ndarray
    cost_unit: str
    bytes_source: str
    latency_semantics: str


@dataclass(frozen=True)
class ReplicatedSCPConfig:
    lambda_prefill: float
    lambda_decode: float
    extra_replicas_per_layer: int
    extra_replicas_per_rank_per_layer: int
    global_extra_replicas: int
    max_replicas_per_expert: int
    stage: str
    objective: str
    lambda_event: float
    lambda_small_packet: float
    small_packet_threshold: int
    load_slack: float
    expert_cache_capacity: int
    expert_cache_capacity_by_rank: Tuple[int, ...]
    h2d_slack: float
    compute_slack: float
    min_benefit: float
    min_net_gain_ms: float
    min_rankmax_benefit: float
    h2d_ms_per_proxy: float
    compute_ms_per_proxy: float
    latency_ms_per_event: Optional[float]
    decode_rs_gain_weight: float
    require_rankmax_gain: bool
    require_h2d_nondegrade: bool
    require_compute_nondegrade: bool
    require_hotset_nondegrade: bool
    require_inter_node_rankmax_bytes_gain: bool
    allow_same_node_replicas: bool
    min_inter_node_rankmax_bytes_reduction_pct: float
    min_inter_node_bytes_reduction_pct: float
    hotset_min_load_fraction: float
    hotset_slack: float
    rank_compute_time_scale: Tuple[float, ...]
    rank_h2d_time_scale: Tuple[float, ...]
    disable_h2d_gate: bool = False
    cache_hit_drop_limit: float = 0.0
    candidate_prefilter_topn: int = 256
    require_rs_safe: bool = False
    rs_total_slack: float = 0.0
    rs_h2d_slack: float = 0.03
    rs_service_hit_drop_limit: float = 0.0
    rs_prefetch_use_drop_limit: float = 0.0
    require_valid_positive: bool = False
    min_tail_headroom_pct: float = 0.5
    min_comm_gain_ms: float = 0.0
    cache_validation_interval: int = 4
    event_aware: bool = True
    event_first: bool = True
    require_event_gain_when_latency: bool = False


@dataclass
class LayerMetrics:
    comm_cost: float
    dispatch_bw_cost: float
    combine_bw_cost: float
    latency_event_cost_ms: float
    latency_event_cost_by_source_rank: np.ndarray
    comm_cost_by_source_rank: np.ndarray
    same_rank_items: float
    same_node_items: float
    cross_node_items: float
    cross_node_items_by_source_rank: np.ndarray
    active_pairs: int
    active_cross_node_pairs: int
    active_cross_node_pairs_by_source_rank: np.ndarray
    small_cross_node_pairs: int
    small_cross_node_pairs_by_source_rank: np.ndarray
    small_cross_node_items: float
    dispatch_bytes_total: float
    combine_bytes_total: float
    inter_node_dispatch_bytes: float
    inter_node_combine_bytes: float
    inter_node_dispatch_bytes_by_rank: np.ndarray
    inter_node_combine_bytes_by_rank: np.ndarray
    rank_load: np.ndarray
    rank_expert_load: np.ndarray
    h2d_miss_proxy_by_rank: np.ndarray
    compute_proxy_by_rank: np.ndarray
    hotset_count_by_rank: np.ndarray

    @property
    def h2d_miss_proxy_total(self) -> float:
        return float(self.h2d_miss_proxy_by_rank.sum())

    @property
    def h2d_miss_proxy_rankmax(self) -> float:
        return float(self.h2d_miss_proxy_by_rank.max(initial=0.0))

    @property
    def compute_proxy_rankmax(self) -> float:
        return float(self.compute_proxy_by_rank.max(initial=0.0))

    @property
    def hotset_count_total(self) -> float:
        return float(self.hotset_count_by_rank.sum())

    @property
    def hotset_count_rankmax(self) -> float:
        return float(self.hotset_count_by_rank.max(initial=0.0))

    @property
    def comm_cost_rankmax(self) -> float:
        return float(self.comm_cost_by_source_rank.max(initial=0.0))

    @property
    def cross_node_items_rankmax(self) -> float:
        return float(self.cross_node_items_by_source_rank.max(initial=0.0))

    @property
    def inter_node_dispatch_bytes_rankmax(self) -> float:
        return float(self.inter_node_dispatch_bytes_by_rank.max(initial=0.0))

    @property
    def inter_node_combine_bytes_rankmax(self) -> float:
        return float(self.inter_node_combine_bytes_by_rank.max(initial=0.0))

    @property
    def inter_node_bytes_by_rank(self) -> np.ndarray:
        return np.asarray(self.inter_node_dispatch_bytes_by_rank, dtype=np.float64) + np.asarray(
            self.inter_node_combine_bytes_by_rank, dtype=np.float64
        )

    @property
    def inter_node_bytes_rankmax(self) -> float:
        return float(self.inter_node_bytes_by_rank.max(initial=0.0))

    @property
    def load_imbalance_ratio(self) -> float:
        avg = float(self.rank_load.mean())
        return float(self.rank_load.max(initial=0.0) / max(avg, 1e-9))


def _runtime_packet_gate_reason(
    *,
    require_event_gain_when_latency: bool,
    latency_ms: float,
    runtime_packet_cross_event_reduction_pct: float,
    runtime_packet_cost_reduction_pct: float,
) -> str:
    """Gate runtime on replayed event+payload cost, not event count alone."""
    if not bool(require_event_gain_when_latency):
        return ""
    if float(latency_ms or 0.0) <= 0.0:
        return ""
    if float(runtime_packet_cross_event_reduction_pct) > 0.0:
        return ""
    if float(runtime_packet_cost_reduction_pct) > 0.0:
        return ""
    return "runtime_packet_cost_gain_nonpositive"


def load_profile(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        prefill = data["A_prefill"].astype(np.float64)
        decode = data["A_decode"].astype(np.float64)
        raw_meta = data.get("metadata_json")
    if prefill.shape != decode.shape or prefill.ndim != 3:
        raise ValueError(
            f"profile must contain A_prefill/A_decode [layers, ranks, experts], got {prefill.shape}/{decode.shape}"
        )
    meta: Dict[str, Any] = {}
    if raw_meta is not None:
        try:
            meta = json.loads(str(raw_meta.item()))
        except Exception:
            meta = {}
    return prefill, decode, meta


def select_num_experts(
    prefill: np.ndarray,
    decode: np.ndarray,
    meta: Dict[str, Any],
    requested: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    requested = int(requested or 0)
    if requested <= 0:
        return prefill, decode, meta
    profile_experts = int(prefill.shape[-1])
    if requested > profile_experts:
        raise ValueError(f"--num-experts={requested} exceeds profile expert dimension {profile_experts}")
    out_meta = dict(meta)
    out_meta["profile_num_experts"] = int(profile_experts)
    out_meta["solver_num_experts"] = int(requested)
    out_meta["solver_expert_dim_source"] = "explicit_num_experts"
    if requested == profile_experts:
        return prefill, decode, out_meta
    return prefill[:, :, :requested], decode[:, :, :requested], out_meta


def select_trace_profile_num_experts(profile: Dict[str, Any], requested: int) -> Dict[str, Any]:
    requested = int(requested or 0)
    if requested <= 0:
        return profile
    out = dict(profile)
    if "A_prefill" not in out or "A_decode" not in out:
        return out
    profile_experts = int(np.asarray(out["A_prefill"]).shape[-1])
    if requested > profile_experts:
        raise ValueError(f"--num-experts={requested} exceeds trace profile expert dimension {profile_experts}")
    out["A_prefill"] = np.asarray(out["A_prefill"])[:, :, :requested]
    out["A_decode"] = np.asarray(out["A_decode"])[:, :, :requested]
    meta = dict(out.get("metadata", {}) or {})
    meta["profile_num_experts"] = int(profile_experts)
    meta["solver_num_experts"] = int(requested)
    meta["solver_expert_dim_source"] = "explicit_num_experts"
    out["metadata"] = meta
    if "trace_expert" not in out or "trace_count" not in out:
        return out
    expert = np.asarray(out["trace_expert"], dtype=np.int64)
    mask = expert < int(requested)
    n = int(expert.shape[0])
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            out[key] = value[mask]
    pre = int(out["trace_count"][out["trace_stage"].astype(np.int64) == 0].sum())
    dec = int(out["trace_count"][out["trace_stage"].astype(np.int64) == 1].sum())
    if pre != int(out["A_prefill"].sum()) or dec != int(out["A_decode"].sum()):
        raise ValueError(
            "trimmed ordered trace/A mismatch: "
            f"trace_prefill={pre} A_prefill={int(out['A_prefill'].sum())} "
            f"trace_decode={dec} A_decode={int(out['A_decode'].sum())}"
        )
    return out


def load_topology(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_scale_map(raw: str) -> Dict[int, float]:
    raw = str(raw or "").strip()
    if not raw:
        return {}
    out: Dict[int, float] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if any(":" in p for p in parts):
        for part in parts:
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            out[int(k.strip())] = max(float(v.strip()), 1e-9)
        return out
    for idx, part in enumerate(parts):
        out[int(idx)] = max(float(part), 1e-9)
    return out


def _parse_int_capacity_vector(raw: str, *, world: int, default: int, name: str) -> Tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(max(0, int(default)) for _ in range(int(world)))
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    if not parts:
        return tuple(max(0, int(default)) for _ in range(int(world)))
    if any(":" in p for p in parts):
        values = [max(0, int(default)) for _ in range(int(world))]
        for part in parts:
            if ":" not in part:
                raise ValueError(f"{name} mixes rank:value entries with bare values: {raw!r}")
            key, value = part.split(":", 1)
            rank = int(key.strip())
            if rank < 0 or rank >= int(world):
                raise ValueError(f"{name} rank out of range: {rank} for world {world}")
            values[rank] = max(0, int(value.strip()))
        return tuple(int(x) for x in values)
    if len(parts) != int(world):
        raise ValueError(f"{name} length {len(parts)} != world {world}: {raw!r}")
    return tuple(max(0, int(x)) for x in parts)


def resolve_rank_scales(
    *,
    world: int,
    rank_to_node: Dict[int, int],
    rank_scale: str = "",
    node_scale: str = "",
    topology_values: Optional[Sequence[float]] = None,
    default: float = 1.0,
) -> Tuple[float, ...]:
    rank_map = _parse_scale_map(rank_scale)
    node_map = _parse_scale_map(node_scale)
    values: List[float] = []
    for rank in range(int(world)):
        if rank in rank_map:
            values.append(max(float(rank_map[rank]), 1e-9))
            continue
        node = int(rank_to_node.get(rank, rank))
        if node in node_map:
            values.append(max(float(node_map[node]), 1e-9))
            continue
        if topology_values is not None and len(topology_values) == world:
            values.append(max(float(topology_values[rank]), 1e-9))
            continue
        values.append(max(float(default), 1e-9))
    return tuple(values)


def _invert_time_from_speed(speed_scale: Sequence[float]) -> Tuple[float, ...]:
    return tuple(1.0 / max(float(v), 1e-9) for v in speed_scale)


def _matrix_from_topology(
    topology: Dict[str, Any],
    rank_to_node: Dict[int, int],
    *,
    same_rank: float,
    same_node: float,
    cross_node: float,
) -> np.ndarray:
    world = int(topology["world_size"])
    M = np.zeros((world, world), dtype=np.float64)
    for i in range(world):
        for j in range(world):
            if i == j:
                M[i, j] = float(same_rank)
            elif rank_to_node.get(i, i) == rank_to_node.get(j, j):
                M[i, j] = float(same_node)
            else:
                M[i, j] = float(cross_node)
    return M


def _first_number(*values: Any, default: float = 0.0) -> float:
    for v in values:
        try:
            if v is not None and float(v) > 0.0:
                return float(v)
        except Exception:
            pass
    return float(default)


def build_comm_model(
    topology: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    dispatch_item_bytes_override: float = 0.0,
    combine_item_bytes_override: float = 0.0,
    hidden_bytes_override: float = 0.0,
    dispatch_weight_bytes: float = 2.0,
    dispatch_expert_id_bytes: float = 8.0,
    include_dispatch_metadata: bool = True,
    runtime_latency_ms_override: Optional[float] = None,
) -> CommModel:
    world = int(topology["world_size"])
    rank_to_node = {int(k): int(v) for k, v in topology.get("rank_to_node", {}).items()}
    if len(rank_to_node) != world:
        rank_to_node = {r: int(rank_to_node.get(r, r)) for r in range(world)}

    bw = topology.get("bandwidth_model", {}) or {}
    hidden_bytes = _first_number(
        hidden_bytes_override,
        meta.get("effective_hidden_bytes_per_item"),
        meta.get("hidden_bytes_per_item"),
        meta.get("bytes_per_token_expert_item"),
        meta.get("bytes_per_token"),
        bw.get("bytes_per_token"),
        default=4096.0,
    )
    dispatch_item_bytes = _first_number(
        dispatch_item_bytes_override,
        meta.get("effective_dispatch_bytes_per_item"),
        bw.get("dispatch_item_bytes"),
        default=0.0,
    )
    combine_item_bytes = _first_number(
        combine_item_bytes_override,
        meta.get("effective_combine_bytes_per_item"),
        bw.get("combine_item_bytes"),
        default=0.0,
    )
    bytes_source = "profile_or_topology_effective_bytes"
    if dispatch_item_bytes <= 0.0:
        # This repository's audited standard EP path sends hidden+router_weight and expert id in dispatch.
        dispatch_item_bytes = hidden_bytes + (float(dispatch_weight_bytes) + float(dispatch_expert_id_bytes) if include_dispatch_metadata else 0.0)
        bytes_source = "audited_eval_core_hidden_plus_weight_plus_expert_id_default"
    if combine_item_bytes <= 0.0:
        # Gather/combine returns expert output hidden vector.
        combine_item_bytes = hidden_bytes
        if bytes_source == "profile_or_topology_effective_bytes":
            bytes_source = "audited_eval_core_hidden_output_default"

    if bool(bw.get("enabled", False)):
        same_node_bw = max(float(bw.get("same_node_bw_MBps", 12000.0) or 12000.0), 1e-9)
        cross_node_bw = max(float(bw.get("cross_node_bw_MBps", 100.0) or 100.0), 1e-9)
        if runtime_latency_ms_override is None:
            cross_node_alpha = max(float(bw.get("cross_node_latency_ms", 0.0) or 0.0), 0.0)
        else:
            cross_node_alpha = max(float(runtime_latency_ms_override), 0.0)
        bw_matrix = _matrix_from_topology(
            topology,
            rank_to_node,
            same_rank=1e30,  # same-rank is charged zero below; this avoids division by zero.
            same_node=same_node_bw,
            cross_node=cross_node_bw,
        )
        dispatch_cost = np.zeros((world, world), dtype=np.float64)
        combine_cost = np.zeros((world, world), dtype=np.float64)
        for s in range(world):
            for r in range(world):
                if s == r:
                    dispatch_cost[s, r] = 0.0
                    combine_cost[s, r] = 0.0
                else:
                    dispatch_cost[s, r] = dispatch_item_bytes / (bw_matrix[s, r] * 1_000_000.0) * 1000.0
                    combine_cost[s, r] = combine_item_bytes / (bw_matrix[s, r] * 1_000_000.0) * 1000.0
        alpha = _matrix_from_topology(
            topology,
            rank_to_node,
            same_rank=0.0,
            same_node=0.0,
            cross_node=cross_node_alpha,
        )
        total = np.zeros((world, world), dtype=np.float64)
        for s in range(world):
            for r in range(world):
                total[s, r] = dispatch_cost[s, r] + combine_cost[r, s]
        return CommModel(
            world=world,
            rank_to_node=rank_to_node,
            dispatch_item_bytes=float(dispatch_item_bytes),
            combine_item_bytes=float(combine_item_bytes),
            hidden_bytes=float(hidden_bytes),
            router_weight_bytes=float(dispatch_weight_bytes if include_dispatch_metadata else 0.0),
            expert_id_bytes=float(dispatch_expert_id_bytes if include_dispatch_metadata else 0.0),
            dispatch_item_cost=dispatch_cost,
            combine_item_cost=combine_cost,
            total_item_cost=total,
            dispatch_alpha_ms=alpha,
            combine_alpha_ms=alpha.T.copy(),
            dispatch_bw_MBps=bw_matrix,
            combine_bw_MBps=bw_matrix.T.copy(),
            cost_unit="ms_per_token_expert_assignment_bandwidth_roundtrip",
            bytes_source=bytes_source,
            latency_semantics="runtime_weaknet_one_cross_node_event_per_nonempty_source_destination",
        )

    # Relative fallback.  Treat relative cost as one direction; total is dispatch + combine.
    cost = topology.get("cost", {}) or {}
    one_way = _matrix_from_topology(
        topology,
        rank_to_node,
        same_rank=float(cost.get("same_rank", 0.0)),
        same_node=float(cost.get("same_node", 1.0)),
        cross_node=float(cost.get("cross_node", 8.0)),
    )
    total = one_way + one_way.T
    alpha0 = np.zeros_like(one_way)
    return CommModel(
        world=world,
        rank_to_node=rank_to_node,
        dispatch_item_bytes=float(dispatch_item_bytes),
        combine_item_bytes=float(combine_item_bytes),
        hidden_bytes=float(hidden_bytes),
        router_weight_bytes=float(dispatch_weight_bytes if include_dispatch_metadata else 0.0),
        expert_id_bytes=float(dispatch_expert_id_bytes if include_dispatch_metadata else 0.0),
        dispatch_item_cost=one_way,
        combine_item_cost=one_way.T.copy(),
        total_item_cost=total,
        dispatch_alpha_ms=alpha0,
        combine_alpha_ms=alpha0,
        dispatch_bw_MBps=np.zeros_like(one_way),
        combine_bw_MBps=np.zeros_like(one_way),
        cost_unit="relative_roundtrip_cost",
        bytes_source=bytes_source,
        latency_semantics="not_available_in_relative_cost_mode",
    )


def round_robin_replicas(layers: int, experts: int, world: int) -> Dict[str, Dict[str, List[int]]]:
    return {str(l): {str(e): [int(e % world)] for e in range(experts)} for l in range(layers)}


def _topology_fingerprint(topology: Dict[str, Any]) -> str:
    blob = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _replica_choice_cost_matrix(model: CommModel) -> List[List[float]]:
    return np.asarray(model.total_item_cost, dtype=np.float64).astype(float).tolist()


def _base_placement_obj(
    *,
    replicas: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    topology: Dict[str, Any],
    layers: int,
    experts: int,
    objective: str,
    cfg: Optional[ReplicatedSCPConfig] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "format": "replicated_scp_ep_v2",
        "world_size": int(model.world),
        "num_layers": int(layers),
        "num_experts": int(experts),
        "objective": str(objective),
        "rank_to_node": {str(k): int(v) for k, v in model.rank_to_node.items()},
        "replicas": replicas,
        "replica_choice_cost_matrix": _replica_choice_cost_matrix(model),
        "cost_unit": str(model.cost_unit),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "hidden_bytes": float(model.hidden_bytes),
        "router_weight_bytes": float(model.router_weight_bytes),
        "expert_id_bytes": float(model.expert_id_bytes),
        "topology_fingerprint": _topology_fingerprint(topology),
        "runtime_assignment_semantics": (
            "runtime chooses the executable replica with the lowest replica_choice_cost_matrix[source][rank]; "
            "ties keep the round-robin owner, then lower rank id"
        ),
    }
    if cfg is not None:
        out.update(
            {
                "lambda_prefill": float(cfg.lambda_prefill),
                "lambda_decode": float(cfg.lambda_decode),
                "stage": str(cfg.stage),
                "global_extra_replicas": int(cfg.global_extra_replicas),
                "load_slack": float(cfg.load_slack),
                "h2d_slack": float(cfg.h2d_slack),
                "compute_slack": float(cfg.compute_slack),
                "cache_hit_drop_limit": float(cfg.cache_hit_drop_limit),
                "disable_h2d_gate": bool(cfg.disable_h2d_gate),
                "candidate_prefilter_topn": int(cfg.candidate_prefilter_topn),
                "event_aware": bool(cfg.event_aware),
                "event_first": bool(cfg.event_first),
                "require_event_gain_when_latency": bool(cfg.require_event_gain_when_latency),
                "lambda_event": float(cfg.lambda_event),
                "lambda_small_packet": float(cfg.lambda_small_packet),
                "small_packet_threshold": int(cfg.small_packet_threshold),
                "expert_cache_capacity": int(cfg.expert_cache_capacity),
                "expert_cache_capacity_by_rank": [int(x) for x in cfg.expert_cache_capacity_by_rank],
                "rank_compute_time_scale": [float(x) for x in cfg.rank_compute_time_scale],
                "rank_h2d_time_scale": [float(x) for x in cfg.rank_h2d_time_scale],
                "replica_budget": {
                    "extra_replicas_per_layer": int(cfg.extra_replicas_per_layer),
                    "extra_replicas_per_rank_per_layer": int(cfg.extra_replicas_per_rank_per_layer),
                    "global_extra_replicas": int(cfg.global_extra_replicas),
                    "max_replicas_per_expert": int(cfg.max_replicas_per_expert),
                },
            }
        )
    return out


def _as_int_replicas(replicas: Dict[str, Dict[str, List[int]]]) -> Dict[int, Dict[int, List[int]]]:
    return {int(l): {int(e): [int(r) for r in ranks] for e, ranks in layer.items()} for l, layer in replicas.items()}


def _speed_from_time_scale(time_scale: Sequence[float]) -> np.ndarray:
    arr = np.maximum(np.asarray(time_scale, dtype=np.float64), 1e-9)
    return 1.0 / arr


def _hetero_raw_load_cap(total_load: float, rank_compute_time_scale: Tuple[float, ...], load_slack: float) -> np.ndarray:
    # rank_compute_time_scale is a time multiplier; lower is faster and may accept more raw load.
    speed = _speed_from_time_scale(rank_compute_time_scale)
    share = speed / max(float(speed.sum()), 1e-9)
    return float(total_load) * share * (1.0 + float(load_slack))


def replica_benefit(A: np.ndarray, model: CommModel, reps: Dict[int, Dict[int, List[int]]], layer: int, expert: int, rank: int) -> float:
    gain = 0.0
    C = model.total_item_cost
    for source in range(A.shape[1]):
        count = float(A[layer, source, expert])
        if count <= 0.0:
            continue
        current = min(float(C[source, r]) for r in reps[layer][expert])
        proposed = min(current, float(C[source, rank]))
        gain += count * (current - proposed)
    return float(gain)


def _runtime_nearest_rank_for_expert(
    *,
    source: int,
    expert: int,
    replica_ranks: Sequence[int],
    model: CommModel,
) -> int:
    """Runtime dispatch rule derived from a replica set.

    The solver decides only which ranks may execute an expert.  The concrete
    source-rank -> executor-rank assignment is not a solver variable: runtime
    derives it by choosing the replica with the lowest topology cost for the
    current source rank.  Ties keep the round-robin owner to avoid needless cache
    churn when communication cost is identical.
    """
    candidates = sorted(set(int(r) for r in replica_ranks if 0 <= int(r) < int(model.world)))
    if not candidates:
        raise ValueError(f"expert {expert} has no valid replicas")
    rr_owner = int(expert) % int(model.world)
    C = model.total_item_cost
    return int(
        min(
            candidates,
            key=lambda r: (
                float(C[int(source), int(r)]),
                0 if int(r) == rr_owner else 1,
                int(r),
            ),
        )
    )


def build_choice_layer(
    A_layer: np.ndarray,
    model: CommModel,
    reps_layer: Dict[int, List[int]],
    *,
    load_slack: float,
    rank_compute_time_scale: Tuple[float, ...],
    rank_h2d_time_scale: Tuple[float, ...],
) -> Tuple[Dict[str, Dict[str, int]], np.ndarray]:
    """Build diagnostic runtime assignment from replicas.

    This function used to greedily solve a source/expert -> rank `choice` table
    under a load cap.  That was the wrong abstraction for replicated SCP-EP: the
    placement file should contain only executable replica sets.  Runtime then
    derives the destination rank from the current source rank and topology.

    The returned table is therefore diagnostic/offline accounting only.  It is
    deliberately not emitted as a runtime contract.
    """
    world, experts = A_layer.shape
    _ = (load_slack, rank_compute_time_scale, rank_h2d_time_scale)
    choice: Dict[str, Dict[str, int]] = {str(s): {} for s in range(world)}
    rank_expert_load = np.zeros((world, experts), dtype=np.float64)
    for source in range(world):
        for expert in range(experts):
            count = float(A_layer[source, expert])
            picked = _runtime_nearest_rank_for_expert(
                source=int(source),
                expert=int(expert),
                replica_ranks=reps_layer[int(expert)],
                model=model,
            )
            choice[str(source)][str(expert)] = int(picked)
            rank_expert_load[picked, expert] += float(count)
    return choice, rank_expert_load


def build_choice(
    A: np.ndarray,
    model: CommModel,
    replicas: Dict[str, Dict[str, List[int]]],
    cfg: ReplicatedSCPConfig,
) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], np.ndarray]:
    layers, world, experts = A.shape
    reps = _as_int_replicas(replicas)
    choice: Dict[str, Dict[str, Dict[str, int]]] = {str(l): {str(s): {} for s in range(world)} for l in range(layers)}
    rank_expert_load = np.zeros((layers, world, experts), dtype=np.float64)
    for layer in range(layers):
        layer_choice, layer_load = build_choice_layer(
            A[layer],
            model,
            reps[layer],
            load_slack=cfg.load_slack,
            rank_compute_time_scale=cfg.rank_compute_time_scale,
            rank_h2d_time_scale=cfg.rank_h2d_time_scale,
        )
        choice[str(layer)] = layer_choice
        rank_expert_load[layer] = layer_load
    return choice, rank_expert_load


def _cache_capacity_vector(cache_capacity: int | Sequence[int], world: int) -> Tuple[int, ...]:
    if isinstance(cache_capacity, (list, tuple, np.ndarray)):
        values = tuple(max(0, int(x)) for x in list(cache_capacity))
        if len(values) != int(world):
            raise ValueError(f"cache capacity vector length {len(values)} != world {world}")
        return values
    return tuple(max(0, int(cache_capacity)) for _ in range(int(world)))


def _layer_h2d_proxy(rank_expert_load: np.ndarray, rank_h2d_time_scale: Tuple[float, ...], cache_capacity: int | Sequence[int]) -> np.ndarray:
    """Frequency-only H2D risk proxy.

    This is not a cache replay.  It approximates miss pressure by assuming the top-K
    most frequent experts per rank are resident and all remaining accesses miss.
    Solver output marks this as unverified so runtime/cache replay remains required.
    """
    world = rank_expert_load.shape[0]
    caps = _cache_capacity_vector(cache_capacity, int(world))
    out = np.zeros((world,), dtype=np.float64)
    h2d_scale = np.maximum(np.asarray(rank_h2d_time_scale, dtype=np.float64), 1e-9)
    for rank in range(world):
        cap = max(int(caps[int(rank)]), 0)
        loads = np.asarray(rank_expert_load[rank], dtype=np.float64)
        total = float(loads.sum())
        if cap <= 0:
            hot = 0.0
        elif cap >= loads.size:
            hot = total
        else:
            hot = float(np.partition(loads, -cap)[-cap:].sum())
        out[rank] = max(total - hot, 0.0) * float(h2d_scale[rank])
    return out


def _layer_hotset_count(rank_expert_load: np.ndarray, min_load_fraction: float) -> np.ndarray:
    world = rank_expert_load.shape[0]
    out = np.zeros((world,), dtype=np.float64)
    frac = max(float(min_load_fraction), 0.0)
    for rank in range(world):
        loads = np.asarray(rank_expert_load[rank], dtype=np.float64)
        total = float(loads.sum())
        if total <= 0.0:
            out[rank] = 0.0
            continue
        threshold = max(float(total) * frac, 1e-9)
        out[rank] = float(np.count_nonzero(loads >= threshold))
    return out


def _choice_pair_counts(A_layer: np.ndarray, choice_layer: Dict[str, Dict[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
    world, experts = A_layer.shape
    pair_items = np.zeros((world, world), dtype=np.float64)  # source -> replica
    rank_expert_load = np.zeros((world, experts), dtype=np.float64)
    for source in range(world):
        cmap = choice_layer[str(source)]
        for expert in range(experts):
            count = float(A_layer[source, expert])
            rank = int(cmap[str(expert)])
            pair_items[source, rank] += count
            rank_expert_load[rank, expert] += count
    return pair_items, rank_expert_load


def evaluate_layer_choice(
    A_layer: np.ndarray,
    model: CommModel,
    choice_layer: Dict[str, Dict[str, int]],
    *,
    rank_compute_time_scale: Tuple[float, ...],
    rank_h2d_time_scale: Tuple[float, ...],
    expert_cache_capacity: int | Sequence[int],
    hotset_min_load_fraction: float,
    small_packet_threshold: int = 4,
) -> LayerMetrics:
    world, _experts = A_layer.shape
    pair_items, rank_expert_load = _choice_pair_counts(A_layer, choice_layer)
    rank_load = rank_expert_load.sum(axis=1)

    dispatch_bw = pair_items * model.dispatch_item_cost
    combine_bw = pair_items * model.combine_item_cost.T  # pair_items[source, replica], combine replica->source
    total_pair_cost = dispatch_bw + combine_bw
    comm_by_source = total_pair_cost.sum(axis=1)
    comm_cost = float(total_pair_cost.sum())

    active = pair_items > 0.0
    same_rank_items = 0.0
    same_node_items = 0.0
    cross_node_items = 0.0
    cross_by_source = np.zeros((world,), dtype=np.float64)
    latency_by_source = np.zeros((world,), dtype=np.float64)
    active_cross_by_source = np.zeros((world,), dtype=np.float64)
    small_cross_by_source = np.zeros((world,), dtype=np.float64)
    inter_dispatch_by_source = np.zeros((world,), dtype=np.float64)
    inter_combine_by_source = np.zeros((world,), dtype=np.float64)
    inter_dispatch_bytes = 0.0
    inter_combine_bytes = 0.0
    active_cross_pairs = 0
    small_cross_pairs = 0
    small_cross_items = 0.0
    small_threshold = max(int(small_packet_threshold), 1)
    for s in range(world):
        for r in range(world):
            items = float(pair_items[s, r])
            if items <= 0.0:
                continue
            if s == r:
                same_rank_items += items
            elif model.rank_to_node.get(s, s) == model.rank_to_node.get(r, r):
                same_node_items += items
            else:
                cross_node_items += items
                cross_by_source[s] += items
                active_cross_pairs += 1
                active_cross_by_source[s] += 1.0
                if items <= float(small_threshold):
                    small_cross_pairs += 1
                    small_cross_by_source[s] += 1.0
                    small_cross_items += float(items)
                dispatch_bytes = items * float(model.dispatch_item_bytes)
                combine_bytes = items * float(model.combine_item_bytes)
                inter_dispatch_bytes += dispatch_bytes
                inter_combine_bytes += combine_bytes
                inter_dispatch_by_source[s] += dispatch_bytes
                inter_combine_by_source[s] += combine_bytes
    latency_event = 0.0
    for s in range(world):
        for r in range(world):
            if active[s, r]:
                pair_latency = float(model.dispatch_alpha_ms[s, r])
                latency_event += pair_latency
                latency_by_source[s] += pair_latency

    compute_time_scale = np.maximum(np.asarray(rank_compute_time_scale, dtype=np.float64), 1e-9)
    return LayerMetrics(
        comm_cost=comm_cost,
        dispatch_bw_cost=float(dispatch_bw.sum()),
        combine_bw_cost=float(combine_bw.sum()),
        latency_event_cost_ms=float(latency_event),
        latency_event_cost_by_source_rank=latency_by_source,
        comm_cost_by_source_rank=comm_by_source,
        same_rank_items=float(same_rank_items),
        same_node_items=float(same_node_items),
        cross_node_items=float(cross_node_items),
        cross_node_items_by_source_rank=cross_by_source,
        active_pairs=int(np.count_nonzero(active)),
        active_cross_node_pairs=int(active_cross_pairs),
        active_cross_node_pairs_by_source_rank=active_cross_by_source,
        small_cross_node_pairs=int(small_cross_pairs),
        small_cross_node_pairs_by_source_rank=small_cross_by_source,
        small_cross_node_items=float(small_cross_items),
        dispatch_bytes_total=float(pair_items.sum() * float(model.dispatch_item_bytes)),
        combine_bytes_total=float(pair_items.sum() * float(model.combine_item_bytes)),
        inter_node_dispatch_bytes=float(inter_dispatch_bytes),
        inter_node_combine_bytes=float(inter_combine_bytes),
        inter_node_dispatch_bytes_by_rank=inter_dispatch_by_source,
        inter_node_combine_bytes_by_rank=inter_combine_by_source,
        rank_load=rank_load,
        rank_expert_load=rank_expert_load,
        h2d_miss_proxy_by_rank=_layer_h2d_proxy(rank_expert_load, rank_h2d_time_scale, expert_cache_capacity),
        compute_proxy_by_rank=rank_load * compute_time_scale,
        hotset_count_by_rank=_layer_hotset_count(rank_expert_load, hotset_min_load_fraction),
    )


def evaluate_choice(
    A: np.ndarray,
    model: CommModel,
    choice: Dict[str, Dict[str, Dict[str, int]]],
    cfg: ReplicatedSCPConfig,
) -> Dict[str, Any]:
    layers, world, experts = A.shape
    layer_metrics: List[LayerMetrics] = []
    for layer in range(layers):
        layer_metrics.append(
            evaluate_layer_choice(
                A[layer],
                model,
                choice[str(layer)],
                rank_compute_time_scale=cfg.rank_compute_time_scale,
                rank_h2d_time_scale=cfg.rank_h2d_time_scale,
                expert_cache_capacity=cfg.expert_cache_capacity_by_rank,
                hotset_min_load_fraction=cfg.hotset_min_load_fraction,
            )
        )
    rank_load = np.stack([m.rank_load for m in layer_metrics], axis=0)
    avg = np.maximum(rank_load.mean(axis=1), 1e-9)
    h2d_layer_rank = np.stack([m.h2d_miss_proxy_by_rank for m in layer_metrics], axis=0)
    compute_layer_rank = np.stack([m.compute_proxy_by_rank for m in layer_metrics], axis=0)
    hotset_layer_rank = np.stack([m.hotset_count_by_rank for m in layer_metrics], axis=0)
    comm_by_source = np.sum([m.comm_cost_by_source_rank for m in layer_metrics], axis=0)
    latency_by_source = np.sum([m.latency_event_cost_by_source_rank for m in layer_metrics], axis=0)
    cross_by_source = np.sum([m.cross_node_items_by_source_rank for m in layer_metrics], axis=0)
    active_cross_by_source = np.sum([m.active_cross_node_pairs_by_source_rank for m in layer_metrics], axis=0)
    small_cross_by_source = np.sum([m.small_cross_node_pairs_by_source_rank for m in layer_metrics], axis=0)
    inter_dispatch_by_source = np.sum([m.inter_node_dispatch_bytes_by_rank for m in layer_metrics], axis=0)
    inter_combine_by_source = np.sum([m.inter_node_combine_bytes_by_rank for m in layer_metrics], axis=0)
    inter_total_by_source = inter_dispatch_by_source + inter_combine_by_source
    rank_runtime_proxy = (
        comm_by_source
        + latency_by_source
        + float(cfg.compute_ms_per_proxy) * compute_layer_rank.sum(axis=0)
        + float(cfg.h2d_ms_per_proxy) * h2d_layer_rank.sum(axis=0)
    )
    return {
        "comm_cost": float(sum(m.comm_cost for m in layer_metrics)),
        "dispatch_bw_cost": float(sum(m.dispatch_bw_cost for m in layer_metrics)),
        "combine_bw_cost": float(sum(m.combine_bw_cost for m in layer_metrics)),
        "latency_event_cost_ms": float(sum(m.latency_event_cost_ms for m in layer_metrics)),
        "latency_event_cost_by_source_rank": latency_by_source.astype(float).tolist(),
        "estimated_comm_with_latency_event_ms": float(sum(m.comm_cost for m in layer_metrics) + sum(m.latency_event_cost_ms for m in layer_metrics)),
        "comm_cost_by_source_rank": comm_by_source.astype(float).tolist(),
        "comm_cost_rankmax": float(comm_by_source.max(initial=0.0)),
        "rank_runtime_proxy_by_rank": rank_runtime_proxy.astype(float).tolist(),
        "rank_runtime_proxy_rankmax": float(rank_runtime_proxy.max(initial=0.0)),
        "same_rank_tokens": float(sum(m.same_rank_items for m in layer_metrics)),
        "same_node_tokens": float(sum(m.same_node_items for m in layer_metrics)),
        "cross_node_tokens": float(sum(m.cross_node_items for m in layer_metrics)),
        "cross_node_tokens_by_source_rank": cross_by_source.astype(float).tolist(),
        "cross_node_tokens_rankmax": float(cross_by_source.max(initial=0.0)),
        "active_pairs_total": int(sum(m.active_pairs for m in layer_metrics)),
        "active_cross_node_pairs_total": int(sum(m.active_cross_node_pairs for m in layer_metrics)),
        "active_cross_node_pairs_by_source_rank": active_cross_by_source.astype(float).tolist(),
        "small_cross_node_pairs_total": int(sum(m.small_cross_node_pairs for m in layer_metrics)),
        "small_cross_node_pairs_by_source_rank": small_cross_by_source.astype(float).tolist(),
        "small_cross_node_items_total": float(sum(m.small_cross_node_items for m in layer_metrics)),
        "dispatch_bytes_total": float(sum(m.dispatch_bytes_total for m in layer_metrics)),
        "combine_bytes_total": float(sum(m.combine_bytes_total for m in layer_metrics)),
        "inter_node_dispatch_bytes": float(sum(m.inter_node_dispatch_bytes for m in layer_metrics)),
        "inter_node_combine_bytes": float(sum(m.inter_node_combine_bytes for m in layer_metrics)),
        "inter_node_dispatch_bytes_by_source_rank": inter_dispatch_by_source.astype(float).tolist(),
        "inter_node_combine_bytes_by_source_rank": inter_combine_by_source.astype(float).tolist(),
        "inter_node_total_bytes_by_source_rank": inter_total_by_source.astype(float).tolist(),
        "inter_node_dispatch_bytes_rankmax": float(inter_dispatch_by_source.max(initial=0.0)),
        "inter_node_combine_bytes_rankmax": float(inter_combine_by_source.max(initial=0.0)),
        "inter_node_total_bytes_rankmax": float(inter_total_by_source.max(initial=0.0)),
        "load_imbalance_ratio": float(np.max(rank_load.max(axis=1) / avg)),
        "max_rank_load": float(rank_load.max(initial=0.0)),
        "avg_rank_load": float(rank_load.mean()),
        "rank_load_by_layer": rank_load.astype(float).tolist(),
        "h2d_miss_proxy_by_rank": h2d_layer_rank.sum(axis=0).astype(float).tolist(),
        "h2d_miss_proxy_total": float(h2d_layer_rank.sum()),
        "h2d_miss_proxy_rankmax": float(h2d_layer_rank.sum(axis=0).max(initial=0.0)),
        "compute_proxy_by_rank": compute_layer_rank.sum(axis=0).astype(float).tolist(),
        "compute_proxy_rankmax": float(compute_layer_rank.sum(axis=0).max(initial=0.0)),
        "hotset_count_by_rank": hotset_layer_rank.sum(axis=0).astype(float).tolist(),
        "hotset_count_total": float(hotset_layer_rank.sum()),
        "hotset_count_rankmax": float(hotset_layer_rank.sum(axis=0).max(initial=0.0)),
    }


def round_robin_choice(layers: int, world: int, experts: int) -> Dict[str, Dict[str, Dict[str, int]]]:
    return {str(l): {str(s): {str(e): int(e % world) for e in range(experts)} for s in range(world)} for l in range(layers)}


def _layer_metric_from_reps(
    A: np.ndarray,
    model: CommModel,
    reps_layer: Dict[int, List[int]],
    layer: int,
    cfg: ReplicatedSCPConfig,
) -> Tuple[Dict[str, Dict[str, int]], LayerMetrics]:
    choice_layer, _ = build_choice_layer(
        A[layer],
        model,
        reps_layer,
        load_slack=cfg.load_slack,
        rank_compute_time_scale=cfg.rank_compute_time_scale,
        rank_h2d_time_scale=cfg.rank_h2d_time_scale,
    )
    return choice_layer, evaluate_layer_choice(
        A[layer],
        model,
        choice_layer,
        rank_compute_time_scale=cfg.rank_compute_time_scale,
        rank_h2d_time_scale=cfg.rank_h2d_time_scale,
        expert_cache_capacity=cfg.expert_cache_capacity_by_rank,
        hotset_min_load_fraction=cfg.hotset_min_load_fraction,
        small_packet_threshold=cfg.small_packet_threshold,
    )


def _within_slack(value: float, baseline: float, slack: float) -> bool:
    return float(value) <= float(baseline) * (1.0 + float(slack)) + 1e-9


def _net_gain(*, comm_gain: float, h2d_proxy_delta: float, compute_proxy_delta: float, cfg: ReplicatedSCPConfig) -> float:
    return (
        float(comm_gain)
        - max(float(h2d_proxy_delta), 0.0) * float(cfg.h2d_ms_per_proxy)
        - max(float(compute_proxy_delta), 0.0) * float(cfg.compute_ms_per_proxy)
    )


def _is_throughput_objective(cfg: ReplicatedSCPConfig) -> bool:
    return str(cfg.objective).lower() in (
        "throughput_rankmax_v1",
        "runtime_rankmax_v1",
        "rankmax_runtime",
        "throughput",
    )


def _is_cacheaware_objective_name(name: str) -> bool:
    return str(name).lower() in ("cacheaware_ranktail_v1", "cache_aware_ranktail_v1", "cacheaware")


def _is_online_aware_objective_name(name: str) -> bool:
    return str(name).lower() in (
        "online_aware_scp_v1",
        "online_aware_scp",
        "online_aware",
        "onlineaware_scp_v1",
        "onlineaware",
    )


def _is_tail_headroom_objective_name(name: str) -> bool:
    return str(name).lower() in (
        "tail_headroom_scp_v2",
        "tail_headroom",
        "grouped_tail",
        "phase_event_cache_scp_v2",
        "phase_event_cache",
    )


def _is_phase_event_cache_objective_name(name: str) -> bool:
    return str(name).lower() in ("phase_event_cache_scp_v2", "phase_event_cache")


def _is_task_local_objective_name(name: str) -> bool:
    return str(name).lower() in ("task_local_tail_scp_v1", "tasklocal_tail_scp_v1", "task_local")


def _is_task_local_diagnostic_objective_name(name: str) -> bool:
    return str(name).lower() in (
        "load_only_replicated_v1",
        "loadonly_replicated_v1",
        "load_only",
        "loadonly",
        "comm_only_tasklocal_v1",
        "commonly_tasklocal_v1",
        "comm_only",
        "commonly",
    )


def _is_load_only_objective_name(name: str) -> bool:
    return str(name).lower() in (
        "load_only_replicated_v1",
        "loadonly_replicated_v1",
        "load_only",
        "loadonly",
    )


def _is_comm_only_tasklocal_objective_name(name: str) -> bool:
    return str(name).lower() in (
        "comm_only_tasklocal_v1",
        "commonly_tasklocal_v1",
        "comm_only",
        "commonly",
    )


def _require_task_local_profile(profile: Dict[str, Any]) -> None:
    required = (
        "trace_stage",
        "trace_layer",
        "trace_source_rank",
        "trace_source_node",
        "trace_expert",
        "trace_count",
        "trace_task_id",
        "trace_task_family",
        "trace_request_global_id",
    )
    missing = [k for k in required if k not in profile]
    if missing:
        raise RuntimeError(f"task_local_tail_scp_v1 requires task-local ordered trace fields, missing: {missing}")
    meta = profile.get("metadata", {}) or {}
    if not bool(meta.get("task_local_enabled", False)):
        raise RuntimeError(
            "task_local_tail_scp_v1 requires profile metadata task_local_enabled=true. "
            "Use SPARSE_TASK_MANIFEST/TASK_MANIFEST during profiling; do not solve task-local SCP from generic C4 traces."
        )
    task_id = np.asarray(profile["trace_task_id"], dtype=np.int64)
    if int(task_id.size) <= 0 or not np.any(task_id >= 0):
        raise RuntimeError("task_local_tail_scp_v1 requires non-default trace_task_id values")
    if not str(meta.get("task_manifest_path", "") or "").strip():
        raise RuntimeError(
            "task_local_tail_scp_v1 requires task_manifest_path in profile metadata. "
            "Use SPARSE_TASK_MANIFEST/TASK_MANIFEST during profiling."
        )


def _task_local_node_counts(profile: Dict[str, Any], stage: str, layers: int, experts: int) -> np.ndarray:
    _require_task_local_profile(profile)
    nodes_arr = profile["trace_source_node"].astype(np.int64)
    node_count = int(max(nodes_arr.max(initial=0), 0) + 1)
    out = np.zeros((node_count, int(layers), int(experts)), dtype=np.float64)
    stages = profile["trace_stage"].astype(np.int64)
    if str(stage).lower() == "prefill":
        mask = stages == 0
    elif str(stage).lower() == "decode":
        mask = stages == 1
    else:
        mask = np.ones_like(stages, dtype=bool)
    layer_arr = profile["trace_layer"].astype(np.int64)
    expert_arr = profile["trace_expert"].astype(np.int64)
    count_arr = profile["trace_count"].astype(np.float64)
    for idx in np.nonzero(mask)[0].tolist():
        n = int(nodes_arr[idx])
        l = int(layer_arr[idx])
        e = int(expert_arr[idx])
        if 0 <= n < node_count and 0 <= l < int(layers) and 0 <= e < int(experts):
            out[n, l, e] += float(count_arr[idx])
    return out


def _stage_filter_for_profile(stage: str) -> Optional[int]:
    name = str(stage or "all").lower()
    if name == "prefill":
        return 0
    if name == "decode":
        return 1
    return None


def _runtime_packet_metrics_from_table(
    *,
    trace_profile: Dict[str, Any],
    dest_table: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    stage: str,
    layers: int,
    experts: int,
) -> Dict[str, Any]:
    """Replay routing trace with eval_core's SCP phase-route packet semantics.

    Runtime counts one cross-node transfer event per non-empty source-rank ->
    destination-rank packet inside each MoE dispatch call.  The older grouped
    objective can report event reductions even when this packet count is
    unchanged, so SCP admission must expose this runtime-equivalent estimate.
    """
    if grouped_trace_indices is None:
        raise RuntimeError("runtime packet replay requires scripts/scp_cache_replay.py")
    _require_task_local_profile(trace_profile)
    groups = grouped_trace_indices(trace_profile, stage=stage if str(stage).lower() in ("prefill", "decode") else "all")
    if not groups:
        return {
            "format": "scp_runtime_packet_replay_v1",
            "stage": str(stage),
            "groups": 0,
            "cross_items": 0,
            "cross_events": 0,
            "cost": 0.0,
            "replica_items": 0,
            "items": 0,
            "per_direction_events": {},
            "per_direction_items": {},
        }
    node_ids = [int(model.rank_to_node.get(int(r), int(r))) for r in range(int(model.world))]
    trace_layers = trace_profile["trace_layer"].astype(np.int64)
    trace_sources = trace_profile["trace_source_rank"].astype(np.int64)
    trace_nodes = trace_profile["trace_source_node"].astype(np.int64)
    trace_experts = trace_profile["trace_expert"].astype(np.int64)
    trace_counts = trace_profile["trace_count"].astype(np.int64)
    world = int(model.world)
    total_items = 0
    replica_items = 0
    cross_items = 0
    cross_events = 0
    total_cost = 0.0
    per_direction_events: Dict[str, int] = {}
    per_direction_items: Dict[str, int] = {}

    for idxs in groups:
        # Match eval_core._record_scp_phase_route_diag: one source rank observes
        # rank-level packet counts for this dispatch call.
        by_source: Dict[int, List[int]] = {}
        by_source_node: Dict[int, int] = {}
        for idx in idxs.tolist():
            i = int(idx)
            cnt = int(trace_counts[i])
            if cnt <= 0:
                continue
            l = int(trace_layers[i])
            s = int(trace_sources[i])
            n = int(trace_nodes[i])
            e = int(trace_experts[i])
            if not (0 <= l < int(layers) and 0 <= e < int(experts) and 0 <= s < world):
                continue
            layer_table = dest_table.get(str(l), {})
            row = layer_table.get(str(n))
            if row is None:
                # This mirrors runtime failure semantics: missing table rows must
                # be visible during solve instead of silently using a different
                # assignment policy.
                raise KeyError(f"dest_rank_table missing layer {l} source_node {n}")
            dst = int(row[e])
            if not (0 <= dst < world):
                raise ValueError(f"dest_rank_table rank out of range layer {l} source_node {n} expert {e}: {dst}")
            counts = by_source.setdefault(int(s), [0 for _ in range(world)])
            counts[int(dst)] += int(cnt)
            by_source_node[int(s)] = int(n)
            total_items += int(cnt)
            rr_owner = int(e) % world
            if int(dst) != int(rr_owner):
                replica_items += int(cnt)
        for s, counts in by_source.items():
            src_node = int(node_ids[int(s)]) if 0 <= int(s) < len(node_ids) else int(by_source_node.get(int(s), int(s)))
            for dst, cnt in enumerate(counts):
                if int(cnt) <= 0:
                    continue
                dst_node = int(node_ids[int(dst)]) if 0 <= int(dst) < len(node_ids) else int(dst)
                total_cost += float(cnt) * float(model.total_item_cost[int(s), int(dst)])
                if int(dst) != int(s) and int(dst_node) != int(src_node):
                    cross_items += int(cnt)
                    cross_events += 1
                    direction = f"node{src_node}->node{dst_node}"
                    per_direction_events[direction] = int(per_direction_events.get(direction, 0) + 1)
                    per_direction_items[direction] = int(per_direction_items.get(direction, 0) + int(cnt))
                    total_cost += float(model.dispatch_alpha_ms[int(s), int(dst)])

    return {
        "format": "scp_runtime_packet_replay_v1",
        "stage": str(stage),
        "groups": int(len(groups)),
        "cross_items": int(cross_items),
        "cross_events": int(cross_events),
        "cost": float(total_cost),
        "replica_items": int(replica_items),
        "items": int(total_items),
        "per_direction_events": {k: int(v) for k, v in sorted(per_direction_events.items())},
        "per_direction_items": {k: int(v) for k, v in sorted(per_direction_items.items())},
        "semantics": "matches eval_core _record_scp_phase_route_diag per dispatch source-rank destination-rank packet counting",
    }


def _runtime_packet_metrics_from_replicas(
    *,
    trace_profile: Dict[str, Any],
    replicas: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    layers: int,
    experts: int,
) -> Dict[str, Any]:
    """Replay replica-set placement with runtime packet counting semantics."""
    if grouped_trace_indices is None:
        raise RuntimeError("runtime packet replay requires scripts/scp_cache_replay.py")
    groups = grouped_trace_indices(trace_profile, stage=_stage_for_trace(cfg))
    world = int(model.world)
    reps = _as_int_replicas(replicas)
    trace_layers = trace_profile["trace_layer"].astype(np.int64)
    trace_sources = trace_profile["trace_source_rank"].astype(np.int64)
    trace_experts = trace_profile["trace_expert"].astype(np.int64)
    trace_counts = trace_profile["trace_count"].astype(np.int64)
    node_ids = [int(model.rank_to_node.get(int(r), int(r))) for r in range(world)]
    total_items = 0
    replica_items = 0
    cross_items = 0
    cross_events = 0
    total_cost = 0.0
    per_direction_events: Dict[str, int] = {}
    per_direction_items: Dict[str, int] = {}
    for idxs in groups:
        by_source: Dict[int, List[int]] = {}
        for idx in idxs.tolist():
            i = int(idx)
            cnt = int(trace_counts[i])
            if cnt <= 0:
                continue
            l = int(trace_layers[i])
            s = int(trace_sources[i])
            e = int(trace_experts[i])
            if not (0 <= l < int(layers) and 0 <= e < int(experts) and 0 <= s < world):
                continue
            dst = _runtime_nearest_rank_for_expert(
                source=int(s),
                expert=int(e),
                replica_ranks=reps[int(l)][int(e)],
                model=model,
            )
            counts = by_source.setdefault(int(s), [0 for _ in range(world)])
            counts[int(dst)] += int(cnt)
            total_items += int(cnt)
            if int(dst) != int(e) % world:
                replica_items += int(cnt)
        for s, counts in by_source.items():
            src_node = int(node_ids[int(s)]) if 0 <= int(s) < len(node_ids) else int(s)
            for dst, cnt in enumerate(counts):
                if int(cnt) <= 0:
                    continue
                dst_node = int(node_ids[int(dst)]) if 0 <= int(dst) < len(node_ids) else int(dst)
                total_cost += float(cnt) * float(model.total_item_cost[int(s), int(dst)])
                if int(dst) != int(s) and int(dst_node) != int(src_node):
                    cross_items += int(cnt)
                    cross_events += 1
                    direction = f"node{src_node}->node{dst_node}"
                    per_direction_events[direction] = int(per_direction_events.get(direction, 0) + 1)
                    per_direction_items[direction] = int(per_direction_items.get(direction, 0) + int(cnt))
                    total_cost += float(model.dispatch_alpha_ms[int(s), int(dst)])
    return {
        "format": "scp_runtime_packet_replay_v1",
        "stage": _stage_for_trace(cfg),
        "groups": int(len(groups)),
        "cross_items": int(cross_items),
        "cross_events": int(cross_events),
        "cost": float(total_cost),
        "replica_items": int(replica_items),
        "items": int(total_items),
        "per_direction_events": {k: int(v) for k, v in sorted(per_direction_events.items())},
        "per_direction_items": {k: int(v) for k, v in sorted(per_direction_items.items())},
        "semantics": "matches eval_core _record_scp_phase_route_diag per dispatch source-rank destination-rank packet counting",
    }


def _rank_for_node(model: CommModel, node_id: int, layer: int, expert: int, per_layer_rank_counts: np.ndarray) -> int:
    ranks = [int(r) for r in range(int(model.world)) if int(model.rank_to_node.get(int(r), int(r))) == int(node_id)]
    if not ranks:
        raise ValueError(f"no rank found for node {node_id}")
    rr_owner = int(expert) % int(model.world)
    if rr_owner in ranks:
        return int(rr_owner)
    return int(min(ranks, key=lambda r: (float(per_layer_rank_counts[int(layer), int(r)]), int(r))))


def _source_rank_for_node(model: CommModel, node_id: int) -> int:
    ranks = [int(r) for r in range(int(model.world)) if int(model.rank_to_node.get(int(r), int(r))) == int(node_id)]
    if not ranks:
        raise ValueError(f"no rank found for node {node_id}")
    return int(min(ranks))


def _build_tasklocal_dest_rank_table(
    *,
    replicas: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    layers: int,
    experts: int,
    node_count: int,
) -> Dict[str, Dict[str, List[int]]]:
    reps = _as_int_replicas(replicas)
    out: Dict[str, Dict[str, List[int]]] = {}
    for l in range(int(layers)):
        layer_map: Dict[str, List[int]] = {}
        for n in range(int(node_count)):
            row: List[int] = []
            for e in range(int(experts)):
                allowed = sorted(set(int(r) for r in reps[int(l)][int(e)]))
                rr_owner = int(e) % int(model.world)
                local = [r for r in allowed if int(model.rank_to_node.get(int(r), int(r))) == int(n)]
                if local:
                    picked = min(local, key=lambda r: (0 if int(r) == rr_owner else 1, int(r)))
                else:
                    picked = _runtime_nearest_rank_for_expert(
                        source=_source_rank_for_node(model, int(n)),
                        expert=int(e),
                        replica_ranks=allowed,
                        model=model,
                    )
                row.append(int(picked))
            layer_map[str(int(n))] = row
        out[str(int(l))] = layer_map
    return out


def _tasklocal_grouped_comm_proxy(node_counts: np.ndarray, dest_table: Dict[str, Dict[str, List[int]]], model: CommModel, cfg: ReplicatedSCPConfig) -> Dict[str, Any]:
    node_count, layers, experts = node_counts.shape
    node_to_sources: Dict[int, List[int]] = {}
    for r in range(int(model.world)):
        node_to_sources.setdefault(int(model.rank_to_node.get(r, r)), []).append(int(r))
    by_node = np.zeros((int(node_count),), dtype=np.float64)
    active_cross = 0
    cross_bytes = 0.0
    for n in range(int(node_count)):
        source_rank = min(node_to_sources.get(int(n), [int(n)]))
        for l in range(int(layers)):
            counts_by_rank = np.zeros((int(model.world),), dtype=np.float64)
            for e in range(int(experts)):
                cnt = float(node_counts[int(n), int(l), int(e)])
                if cnt <= 0.0:
                    continue
                dst = int(dest_table[str(int(l))][str(int(n))][int(e)])
                counts_by_rank[dst] += cnt
            for dst, cnt in enumerate(counts_by_rank.tolist()):
                if float(cnt) <= 0.0:
                    continue
                by_node[int(n)] += float(cnt) * float(model.total_item_cost[int(source_rank), int(dst)])
                if int(model.rank_to_node.get(int(dst), int(dst))) != int(n):
                    active_cross += 1
                    by_node[int(n)] += float(model.dispatch_alpha_ms[int(source_rank), int(dst)])
                    cross_bytes += float(cnt) * (float(model.dispatch_item_bytes) + float(model.combine_item_bytes))
    return {
        "grouped_J_comm": float(by_node.max(initial=0.0)),
        "grouped_J_comm_by_source_node": by_node.astype(float).tolist(),
        "active_cross_node_events": int(active_cross),
        "cross_node_bytes": float(cross_bytes),
    }


def _tasklocal_selected_counts(selected: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    by_node: Dict[str, int] = {}
    by_direction: Dict[str, int] = {}
    for row in selected:
        sn_raw = row.get("source_node", "global")
        sn = str(sn_raw if sn_raw is not None else "global")
        by_node[sn] = int(by_node.get(sn, 0) + 1)
        src_node = row.get("rr_owner_node", row.get("owner_node", "?"))
        dst_node = row.get("target_node", row.get("replica_node", "?"))
        direction = str(row.get("direction", f"{src_node}->{dst_node}"))
        by_direction[direction] = int(by_direction.get(direction, 0) + 1)
    return {k: int(v) for k, v in sorted(by_node.items())}, {k: int(v) for k, v in sorted(by_direction.items())}


def _tasklocal_finalize_placement(
    *,
    replicas: Dict[str, Dict[str, List[int]]],
    selected: List[Dict[str, Any]],
    rejected: Dict[str, int],
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    topology: Dict[str, Any],
    trace_profile: Dict[str, Any],
    valid_trace_profile: Optional[Dict[str, Any]],
    profile_meta: Optional[Dict[str, Any]],
    valid_profile_meta: Optional[Dict[str, Any]],
    diagnostic_mode: str,
    h2d_semantics: str,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layers, world, experts = A.shape
    node_counts = _task_local_node_counts(trace_profile, str(cfg.stage), int(layers), int(experts))
    node_count = int(node_counts.shape[0])
    rr_replicas = round_robin_replicas(int(layers), int(experts), int(world))
    baseline_table = _build_tasklocal_dest_rank_table(
        replicas=rr_replicas,
        model=model,
        layers=int(layers),
        experts=int(experts),
        node_count=int(node_count),
    )
    cur_table = _build_tasklocal_dest_rank_table(
        replicas=replicas,
        model=model,
        layers=int(layers),
        experts=int(experts),
        node_count=int(node_count),
    )
    baseline_proxy = _tasklocal_grouped_comm_proxy(node_counts, baseline_table, model, cfg)
    cur_proxy = _tasklocal_grouped_comm_proxy(node_counts, cur_table, model, cfg)
    runtime_packet_before = _runtime_packet_metrics_from_table(
        trace_profile=trace_profile,
        dest_table=baseline_table,
        model=model,
        stage=str(cfg.stage),
        layers=int(layers),
        experts=int(experts),
    )
    runtime_packet_after = _runtime_packet_metrics_from_table(
        trace_profile=trace_profile,
        dest_table=cur_table,
        model=model,
        stage=str(cfg.stage),
        layers=int(layers),
        experts=int(experts),
    )
    baseline_state = _make_grouped_state(
        trace_profile=trace_profile,
        topology=topology,
        replicas=rr_replicas,
        model=model,
        cfg=cfg,
        layers=int(layers),
        experts=int(experts),
    )
    cur_state = _make_grouped_state(
        trace_profile=trace_profile,
        topology=topology,
        replicas=replicas,
        model=model,
        cfg=cfg,
        layers=int(layers),
        experts=int(experts),
    )
    before = float(_state_J_comm(baseline_state))
    after = float(_state_J_comm(cur_state))
    reduction_pct = 100.0 * (before - after) / max(before, 1e-9)

    valid_proxy = None
    valid_baseline_proxy = None
    valid_reduction_pct = None
    if valid_trace_profile is not None:
        valid_counts = _task_local_node_counts(valid_trace_profile, str(cfg.stage), int(layers), int(experts))
        valid_baseline_proxy = _tasklocal_grouped_comm_proxy(valid_counts, baseline_table, model, cfg)
        valid_proxy = _tasklocal_grouped_comm_proxy(valid_counts, cur_table, model, cfg)
        vb = float(valid_baseline_proxy["grouped_J_comm"])
        va = float(valid_proxy["grouped_J_comm"])
        valid_reduction_pct = 100.0 * (vb - va) / max(vb, 1e-9)

    selected_by_node, selected_by_direction = _tasklocal_selected_counts(selected)
    selected_cross_node = int(sum(1 for row in selected if bool(row.get("cross_node_replica", True))))
    selected_same_node = int(len(selected) - selected_cross_node)
    event_eliminated = int(sum(int(row.get("cross_node_events_eliminated", 0)) for row in selected))
    event_created = int(sum(int(row.get("cross_node_events_created", 0)) for row in selected))
    event_reducing = int(sum(1 for row in selected if int(row.get("active_cross_node_event_reduction", 0)) > 0))
    cross_before = float(baseline_proxy["cross_node_bytes"])
    cross_after = float(cur_proxy["cross_node_bytes"])
    active_before = int(runtime_packet_before["cross_events"])
    active_after = int(runtime_packet_after["cross_events"])
    cross_reduction_pct = 100.0 * (cross_before - cross_after) / max(cross_before, 1.0)
    active_reduction_pct = 100.0 * (float(active_before) - float(active_after)) / max(float(active_before), 1.0)
    runtime_packet_cross_items_before = int(runtime_packet_before["cross_items"])
    runtime_packet_cross_items_after = int(runtime_packet_after["cross_items"])
    runtime_packet_cross_item_reduction_pct = 100.0 * (
        float(runtime_packet_cross_items_before) - float(runtime_packet_cross_items_after)
    ) / max(float(runtime_packet_cross_items_before), 1.0)
    runtime_packet_cost_before = float(runtime_packet_before["cost"])
    runtime_packet_cost_after = float(runtime_packet_after["cost"])
    runtime_packet_cost_reduction_pct = 100.0 * (
        float(runtime_packet_cost_before) - float(runtime_packet_cost_after)
    ) / max(float(runtime_packet_cost_before), 1e-9)

    noop_reasons: List[str] = []
    if not selected:
        noop_reasons.append("no_extra_replicas_selected")
    if reduction_pct <= 0.0:
        noop_reasons.append("grouped_J_comm_gain_nonpositive")
    if bool(cfg.require_valid_positive) and valid_reduction_pct is not None and float(valid_reduction_pct) <= 0.0:
        noop_reasons.append("valid_grouped_J_comm_not_positive")
    packet_gate_reason = _runtime_packet_gate_reason(
        require_event_gain_when_latency=bool(cfg.require_event_gain_when_latency),
        latency_ms=float(cfg.latency_ms_per_event or 0.0),
        runtime_packet_cross_event_reduction_pct=float(active_reduction_pct),
        runtime_packet_cost_reduction_pct=float(runtime_packet_cost_reduction_pct),
    )
    if packet_gate_reason:
        noop_reasons.append(packet_gate_reason)

    placement = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=topology,
        layers=int(layers),
        experts=int(experts),
        objective=str(cfg.objective),
        cfg=cfg,
    )
    placement["format"] = "replicated_scp_ep_v3"
    placement["task_locality_aware"] = bool(diagnostic_mode == "tasklocal")
    placement["diagnostic_mode"] = str(diagnostic_mode)
    placement["source_nodes"] = list(range(int(node_count)))
    placement["dest_rank_table"] = cur_table
    placement["dest_rank_table_prefill"] = cur_table
    # v1 placements are prefill-only.  Keep decode explicitly at RR so a
    # misconfigured runtime cannot apply a prefill placement to decode.
    placement["dest_rank_table_decode"] = baseline_table
    placement["runtime_assignment_semantics"] = (
        "Replicated SCP placement-only contract: router top-k/expert ids/scores/weights stay unchanged; "
        "runtime indexes dest_rank_table[layer][source_node][expert] for enabled prefill SCP. "
        "dest_rank_table_decode is RR in v1 unless a decode-safe solver is used."
    )
    placement["selected_replicas"] = selected
    placement["profile_metadata"] = profile_meta or {}
    if valid_profile_meta:
        placement["valid_profile_metadata"] = valid_profile_meta

    offline = {
        "objective": str(cfg.objective),
        "stage": str(cfg.stage),
        "diagnostic_mode": str(diagnostic_mode),
        "num_extra_replicas": int(len(selected)),
        "selected_replicas": selected,
        "selected_replicas_by_source_node": selected_by_node,
        "selected_replicas_by_direction": selected_by_direction,
        "safety_rejected_candidates": rejected,
        "grouped_J_comm_before": float(before),
        "grouped_J_comm_after": float(after),
        "grouped_J_comm_reduction_pct": float(reduction_pct),
        "baseline_grouped_J_comm_ms": float(before),
        "replicated_grouped_J_comm_ms": float(after),
        "valid_grouped_J_comm_before": None if valid_baseline_proxy is None else float(valid_baseline_proxy["grouped_J_comm"]),
        "valid_grouped_J_comm_after": None if valid_proxy is None else float(valid_proxy["grouped_J_comm"]),
        "valid_grouped_J_comm_reduction_pct": valid_reduction_pct,
        "cross_node_bytes_before": float(cross_before),
        "cross_node_bytes_after": float(cross_after),
        "baseline_cross_node_bytes": float(cross_before),
        "replicated_cross_node_bytes": float(cross_after),
        "active_cross_node_events_before": int(active_before),
        "active_cross_node_events_after": int(active_after),
        "baseline_active_cross_node_events": int(active_before),
        "replicated_active_cross_node_events": int(active_after),
        "runtime_packet_cross_events_before": int(active_before),
        "runtime_packet_cross_events_after": int(active_after),
        "runtime_packet_cross_event_reduction_pct": float(active_reduction_pct),
        "runtime_packet_cross_items_before": int(runtime_packet_cross_items_before),
        "runtime_packet_cross_items_after": int(runtime_packet_cross_items_after),
        "runtime_packet_cross_item_reduction_pct": float(runtime_packet_cross_item_reduction_pct),
        "runtime_packet_cost_before": float(runtime_packet_cost_before),
        "runtime_packet_cost_after": float(runtime_packet_cost_after),
        "runtime_packet_cost_reduction_pct": float(runtime_packet_cost_reduction_pct),
        "runtime_packet_metrics_before": runtime_packet_before,
        "runtime_packet_metrics_after": runtime_packet_after,
        "grouped_active_cross_node_events_before": int(baseline_proxy["active_cross_node_events"]),
        "grouped_active_cross_node_events_after": int(cur_proxy["active_cross_node_events"]),
        "grouped_active_cross_node_events_reduction_pct": 100.0
        * (float(baseline_proxy["active_cross_node_events"]) - float(cur_proxy["active_cross_node_events"]))
        / max(float(baseline_proxy["active_cross_node_events"]), 1.0),
        "cross_node_reduction_pct": float(cross_reduction_pct),
        "cross_node_bytes_reduction_pct": float(cross_reduction_pct),
        "rankmax_inter_node_bytes_reduction_pct": float(reduction_pct),
        "rank_runtime_proxy_reduction_pct": float(reduction_pct),
        "objective_reduction_pct": float(reduction_pct),
        "comm_reduction_pct": float(reduction_pct),
        "baseline_objective_ms": float(before),
        "replicated_objective_ms": float(after),
        "baseline_rankmax_comm_ms": float(before),
        "replicated_rankmax_comm_ms": float(after),
        "rankmax_comm_reduction_pct": float(reduction_pct),
        "baseline_rankmax_h2d_ms": 0.0,
        "replicated_rankmax_h2d_ms": 0.0,
        "rankmax_h2d_change_pct": 0.0,
        "baseline_resident_hit_rate_by_item": 0.0,
        "replicated_resident_hit_rate_by_item": 0.0,
        "baseline_cross_node_tokens": float(cross_before) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "replicated_cross_node_tokens": float(cross_after) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "baseline_cross_node_items": float(cross_before) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "replicated_cross_node_items": float(cross_after) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "load_imbalance_ratio": 0.0,
        "h2d_miss_proxy_change_pct": 0.0,
        "compute_proxy_rankmax_change_pct": 0.0,
        "throughput_objective_enabled": False,
        "h2d_ms_per_proxy": float(cfg.h2d_ms_per_proxy),
        "compute_ms_per_proxy": float(cfg.compute_ms_per_proxy),
        "decode_rs_gain_weight": 0.0,
        "prefill_runtime_assignment_comm_reduction_pct": float(reduction_pct) if str(cfg.stage).lower() in ("prefill", "mixed") else None,
        "decode_runtime_assignment_comm_reduction_pct": None,
        "scp_stackability_status": "prefill_placement_only",
        "scp_stackability_reasons": [],
        "selected_cross_node_replicas": int(selected_cross_node),
        "selected_same_node_replicas": int(selected_same_node),
        "selected_event_reducing_replicas": int(event_reducing),
        "selected_cross_node_events_eliminated": int(event_eliminated),
        "selected_cross_node_events_created": int(event_created),
        "active_cross_node_events_reduction_pct": float(active_reduction_pct),
        "should_run_runtime": bool(not noop_reasons),
        "scp_noop": bool(noop_reasons),
        "noop_reason": ";".join(noop_reasons),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "bytes_source": str(model.bytes_source),
        "latency_semantics": str(model.latency_semantics),
        "cost_unit": str(model.cost_unit),
        "h2d_model_semantics": str(h2d_semantics),
        "runtime_assignment_policy": "dest_rank_table[layer][source_node][expert]_prefill_only_decode_rr",
        "gate_status": {
            "h2d_cache_compute_gates": str(h2d_semantics),
            "valid_positive": bool(valid_reduction_pct is None or valid_reduction_pct > 0.0),
            "runtime_packet_cost_positive": bool(float(runtime_packet_cost_reduction_pct) > 0.0),
            "runtime_packet_event_positive": bool(float(active_reduction_pct) > 0.0),
            "runtime_packet_gate": "pass" if not packet_gate_reason else packet_gate_reason,
            "runtime_packet_gate_semantics": "latency guard uses replayed event+payload cost; event-only non-reduction is reported but not a no-op when payload cost drops",
        },
    }
    if extra_metrics:
        offline.update(extra_metrics)
    placement["offline_metrics"] = offline
    return placement


def solve_tasklocal_diagnostic_replicated_v1(
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    *,
    topology: Dict[str, Any],
    trace_profile: Dict[str, Any],
    valid_trace_profile: Optional[Dict[str, Any]] = None,
    profile_meta: Optional[Dict[str, Any]] = None,
    valid_profile_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Diagnostic placements for task-local workloads.

    load_only_replicated_v1 is intentionally source-node agnostic: it replicates
    globally hottest logical experts and packs them across ranks.  comm_only is
    source-node aware but optimizes only total cross-node bytes.  Both emit the
    same v3 runtime table as TaskLocal-SCP so runtime differences come only from
    placement, not from a different dispatch implementation.
    """
    layers, world, experts = A.shape
    _require_task_local_profile(trace_profile)
    if valid_trace_profile is not None:
        _require_task_local_profile(valid_trace_profile)
    if str(cfg.stage).lower() == "decode":
        raise RuntimeError(f"{cfg.objective} is a prefill diagnostic in v1; decode SCP requires cache/H2D-safe gates")

    node_counts = _task_local_node_counts(trace_profile, str(cfg.stage), int(layers), int(experts))
    node_count = int(node_counts.shape[0])
    replicas = round_robin_replicas(int(layers), int(experts), int(world))
    reps = _as_int_replicas(replicas)
    per_layer_rank_counts = np.zeros((int(layers), int(world)), dtype=np.int64)
    selected: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {
        "same_node_replica": 0,
        "source_node_already_has_replica": 0,
        "max_replicas_per_expert": 0,
        "rank_budget": 0,
        "no_positive_gain": 0,
    }
    budget = max(0, int(cfg.global_extra_replicas))
    if budget <= 0:
        budget = max(0, int(cfg.extra_replicas_per_layer) * int(layers))

    if _is_load_only_objective_name(cfg.objective):
        demand = node_counts.sum(axis=0)  # [layer, expert], no source-node/task signal.
        hot_entries = [
            (float(demand[int(l), int(e)]), int(l), int(e))
            for l in range(int(layers))
            for e in range(int(experts))
            if float(demand[int(l), int(e)]) > 0.0
        ]
        hot_entries.sort(key=lambda x: (-float(x[0]), int(x[1]), int(x[2])))
        for mass, l, e in hot_entries:
            if len(selected) >= int(budget):
                break
            if len(reps[int(l)][int(e)]) >= int(cfg.max_replicas_per_expert):
                rejected["max_replicas_per_expert"] += 1
                continue
            owner = int(e) % int(world)
            owner_node = int(model.rank_to_node.get(int(owner), int(owner)))
            candidate_ranks = [
                int(r)
                for r in range(int(world))
                if int(r) not in set(int(x) for x in reps[int(l)][int(e)])
                and (bool(cfg.allow_same_node_replicas) or int(model.rank_to_node.get(int(r), int(r))) != owner_node)
            ]
            if not candidate_ranks:
                rejected["same_node_replica"] += 1
                continue
            candidate_ranks.sort(key=lambda r: (int(per_layer_rank_counts[int(l), int(r)]), int(r)))
            target_rank = None
            for r in candidate_ranks:
                if int(cfg.extra_replicas_per_rank_per_layer) > 0 and int(per_layer_rank_counts[int(l), int(r)]) >= int(cfg.extra_replicas_per_rank_per_layer):
                    continue
                target_rank = int(r)
                break
            if target_rank is None:
                rejected["rank_budget"] += 1
                continue
            target_node = int(model.rank_to_node.get(int(target_rank), int(target_rank)))
            replicas[str(int(l))][str(int(e))].append(int(target_rank))
            reps[int(l)][int(e)].append(int(target_rank))
            per_layer_rank_counts[int(l), int(target_rank)] += 1
            selected.append(
                {
                    "source_node": None,
                    "layer": int(l),
                    "expert": int(e),
                    "rr_owner": int(owner),
                    "rank": int(target_rank),
                    "target_rank": int(target_rank),
                    "rr_owner_node": int(owner_node),
                    "target_node": int(target_node),
                    "replica_node": int(target_node),
                    "direction": f"{owner_node}->{target_node}",
                    "cross_node_replica": bool(int(owner_node) != int(target_node)),
                    "global_demand": float(mass),
                    "J_comm_gain_ms": 0.0,
                }
            )
    else:
        cur_table = _build_tasklocal_dest_rank_table(
            replicas=replicas,
            model=model,
            layers=int(layers),
            experts=int(experts),
            node_count=int(node_count),
        )
        cur_proxy = _tasklocal_grouped_comm_proxy(node_counts, cur_table, model, cfg)
        for _ in range(int(budget)):
            best_key = None
            best_payload = None
            for n in range(int(node_count)):
                for l in range(int(layers)):
                    for e in range(int(experts)):
                        demand = float(node_counts[int(n), int(l), int(e)])
                        if demand <= 0.0:
                            continue
                        owner = int(e) % int(world)
                        owner_node = int(model.rank_to_node.get(int(owner), int(owner)))
                        if int(owner_node) == int(n):
                            rejected["same_node_replica"] += 1
                            continue
                        if any(int(model.rank_to_node.get(int(r), int(r))) == int(n) for r in reps[int(l)][int(e)]):
                            rejected["source_node_already_has_replica"] += 1
                            continue
                        if len(reps[int(l)][int(e)]) >= int(cfg.max_replicas_per_expert):
                            rejected["max_replicas_per_expert"] += 1
                            continue
                        target_rank = _rank_for_node(model, int(n), int(l), int(e), per_layer_rank_counts)
                        if int(cfg.extra_replicas_per_rank_per_layer) > 0 and int(per_layer_rank_counts[int(l), int(target_rank)]) >= int(cfg.extra_replicas_per_rank_per_layer):
                            rejected["rank_budget"] += 1
                            continue
                        trial = json.loads(json.dumps(replicas))
                        trial[str(int(l))][str(int(e))].append(int(target_rank))
                        trial_table = _build_tasklocal_dest_rank_table(
                            replicas=trial,
                            model=model,
                            layers=int(layers),
                            experts=int(experts),
                            node_count=int(node_count),
                        )
                        trial_proxy = _tasklocal_grouped_comm_proxy(node_counts, trial_table, model, cfg)
                        byte_gain = float(cur_proxy["cross_node_bytes"]) - float(trial_proxy["cross_node_bytes"])
                        event_gain = int(cur_proxy["active_cross_node_events"]) - int(trial_proxy["active_cross_node_events"])
                        if byte_gain <= 0.0:
                            rejected["no_positive_gain"] += 1
                            continue
                        key = (
                            float(byte_gain),
                            int(event_gain),
                            float(demand),
                            -int(l),
                            -int(e),
                            -int(target_rank),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_payload = (int(n), int(l), int(e), int(target_rank), float(byte_gain), int(event_gain), trial, trial_table, trial_proxy)
            if best_payload is None:
                break
            n, l, e, target_rank, byte_gain, event_gain, trial, trial_table, trial_proxy = best_payload
            owner = int(e) % int(world)
            owner_node = int(model.rank_to_node.get(int(owner), int(owner)))
            target_node = int(model.rank_to_node.get(int(target_rank), int(target_rank)))
            replicas = trial
            reps = _as_int_replicas(replicas)
            cur_table = trial_table
            cur_proxy = trial_proxy
            per_layer_rank_counts[int(l), int(target_rank)] += 1
            selected.append(
                {
                    "source_node": int(n),
                    "layer": int(l),
                    "expert": int(e),
                    "rr_owner": int(owner),
                    "rank": int(target_rank),
                    "target_rank": int(target_rank),
                    "rr_owner_node": int(owner_node),
                    "target_node": int(target_node),
                    "replica_node": int(target_node),
                    "direction": f"{owner_node}->{target_node}",
                    "cross_node_replica": bool(int(owner_node) != int(target_node)),
                    "cross_node_bytes_reduction": float(byte_gain),
                    "active_cross_node_event_reduction": int(event_gain),
                    "cross_node_events_eliminated": max(0, int(event_gain)),
                    "cross_node_events_created": max(0, -int(event_gain)),
                    "J_comm_gain_ms": 0.0,
                }
            )

    mode = "load_only" if _is_load_only_objective_name(cfg.objective) else "comm_only"
    placement = _tasklocal_finalize_placement(
        replicas=replicas,
        selected=selected,
        rejected=rejected,
        A=A,
        model=model,
        cfg=cfg,
        topology=topology,
        trace_profile=trace_profile,
        valid_trace_profile=valid_trace_profile,
        profile_meta=profile_meta,
        valid_profile_meta=valid_profile_meta,
        diagnostic_mode=mode,
        h2d_semantics="diagnostic_only_no_h2d_cache_compute_gate",
        extra_metrics={
            "load_only_source_node_aware": bool(mode != "load_only"),
            "comm_only_total_cross_node_bytes_objective": bool(mode == "comm_only"),
        },
    )
    placement["task_locality_aware"] = bool(mode == "comm_only")
    return placement


def solve_task_local_tail_scp_v1(
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    *,
    topology: Dict[str, Any],
    trace_profile: Dict[str, Any],
    valid_trace_profile: Optional[Dict[str, Any]] = None,
    profile_meta: Optional[Dict[str, Any]] = None,
    valid_profile_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if grouped_trace_indices is None:
        raise RuntimeError("task_local_tail_scp_v1 requires scripts/scp_cache_replay.py")
    layers, world, experts = A.shape
    _require_task_local_profile(trace_profile)
    if valid_trace_profile is not None:
        _require_task_local_profile(valid_trace_profile)
    if str(cfg.stage).lower() == "decode":
        raise RuntimeError(
            "task_local_tail_scp_v1 first version is intended for prefill/mixed placement. "
            "Decode SCP needs cache/H2D-safe gates before becoming a primary runtime case."
        )
    node_counts = _task_local_node_counts(trace_profile, str(cfg.stage), int(layers), int(experts))
    node_count = int(node_counts.shape[0])
    replicas = round_robin_replicas(int(layers), int(experts), int(world))
    reps = _as_int_replicas(replicas)
    per_layer_rank_counts = np.zeros((int(layers), int(world)), dtype=np.int64)
    selected: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {
        "owner_already_source_local": 0,
        "source_node_already_has_replica": 0,
        "max_replicas_per_expert": 0,
        "rank_budget": 0,
        "candidate_no_delta": 0,
        "low_comm_gain": 0,
        "event_increase": 0,
        "valid_nonpositive": 0,
    }
    owner_node = [int(model.rank_to_node.get(int(e % world), int(e % world))) for e in range(int(experts))]

    baseline_table = _build_tasklocal_dest_rank_table(
        replicas=replicas,
        model=model,
        layers=int(layers),
        experts=int(experts),
        node_count=int(node_count),
    )
    baseline_proxy = _tasklocal_grouped_comm_proxy(node_counts, baseline_table, model, cfg)
    state = _make_grouped_state(
        trace_profile=trace_profile,
        topology=topology,
        replicas=replicas,
        model=model,
        cfg=cfg,
        layers=int(layers),
        experts=int(experts),
    )
    baseline_J = _state_J_comm(state)

    cur_table = baseline_table
    cur_proxy = baseline_proxy
    total_budget = max(0, int(cfg.global_extra_replicas))
    valid_counts = (
        _task_local_node_counts(valid_trace_profile, str(cfg.stage), int(layers), int(experts))
        if valid_trace_profile is not None and bool(cfg.require_valid_positive)
        else None
    )
    valid_state = (
        _make_grouped_state(
            trace_profile=valid_trace_profile,
            topology=topology,
            replicas=replicas,
            model=model,
            cfg=cfg,
            layers=int(layers),
            experts=int(experts),
        )
        if valid_trace_profile is not None and bool(cfg.require_valid_positive)
        else None
    )
    for _ in range(total_budget):
        best = None
        best_payload = None
        bottleneck_info = _state_bottleneck_info(state, top_fraction=0.20)
        valid_bottleneck_info = _state_bottleneck_info(valid_state, top_fraction=0.20) if valid_state is not None else None
        for n in range(int(node_count)):
            for l in range(int(layers)):
                for e in range(int(experts)):
                    demand = float(node_counts[int(n), int(l), int(e)])
                    if demand <= 0.0:
                        continue
                    if int(owner_node[int(e)]) == int(n):
                        rejected["owner_already_source_local"] += 1
                        continue
                    cur_reps = reps[int(l)][int(e)]
                    if any(int(model.rank_to_node.get(int(r), int(r))) == int(n) for r in cur_reps):
                        rejected["source_node_already_has_replica"] += 1
                        continue
                    if len(cur_reps) >= int(cfg.max_replicas_per_expert):
                        rejected["max_replicas_per_expert"] += 1
                        continue
                    target_rank = _rank_for_node(model, int(n), int(l), int(e), per_layer_rank_counts)
                    if int(target_rank) in set(int(r) for r in cur_reps):
                        continue
                    if int(cfg.extra_replicas_per_rank_per_layer) > 0 and int(per_layer_rank_counts[int(l), int(target_rank)]) >= int(cfg.extra_replicas_per_rank_per_layer):
                        rejected["rank_budget"] += 1
                        continue
                    cand = _candidate_delta_grouped(
                        state=state,
                        model=model,
                        reps=reps,
                        layer=int(l),
                        expert=int(e),
                        rank=int(target_rank),
                        bottleneck_info=bottleneck_info,
                    )
                    if cand is None:
                        rejected["candidate_no_delta"] += 1
                        continue
                    delta = float(cand.get("J_comm_gain_ms", 0.0))
                    event_inc = max(0, int(cand.get("active_cross_node_event_delta", 0)))
                    if delta <= float(cfg.min_comm_gain_ms):
                        rejected["low_comm_gain"] += 1
                        continue
                    if event_inc > 0:
                        rejected["event_increase"] += 1
                        continue
                    valid_cand = None
                    if valid_state is not None and valid_bottleneck_info is not None:
                        valid_cand = _candidate_delta_grouped(
                            state=valid_state,
                            model=model,
                            reps=reps,
                            layer=int(l),
                            expert=int(e),
                            rank=int(target_rank),
                            bottleneck_info=valid_bottleneck_info,
                        )
                        valid_delta = 0.0 if valid_cand is None else float(valid_cand.get("J_comm_gain_ms", 0.0))
                        if valid_delta <= 0.0:
                            rejected["valid_nonpositive"] += 1
                            continue
                    score = (
                        float(delta)
                        + float(cand.get("event_gain_ms", 0.0))
                        + 0.05 * float(cand.get("cross_node_bytes_reduction", 0.0)) / 1_000_000.0
                    )
                    key = (
                        score,
                        int(cand.get("active_cross_node_event_reduction", 0)),
                        float(cand.get("cross_node_bytes_reduction", 0.0)),
                        float(demand),
                        -int(l),
                        -int(e),
                        -int(target_rank),
                    )
                    if best is None or key > best:
                        best = key
                        best_payload = (int(n), int(l), int(e), int(target_rank), cand)
        if best_payload is None:
            break
        n, l, e, target_rank, cand = best_payload
        _apply_candidate_delta(state, cand)
        if valid_state is not None:
            valid_apply = _candidate_delta_grouped(
                state=valid_state,
                model=model,
                reps=reps,
                layer=int(l),
                expert=int(e),
                rank=int(target_rank),
                bottleneck_info=_state_bottleneck_info(valid_state, top_fraction=0.20),
            )
            if valid_apply is not None and float(valid_apply.get("J_comm_gain_ms", 0.0)) > 0.0:
                _apply_candidate_delta(valid_state, valid_apply)
        replicas[str(int(l))][str(int(e))].append(int(target_rank))
        reps[int(l)][int(e)].append(int(target_rank))
        per_layer_rank_counts[int(l), int(target_rank)] += 1
        cur_table = _build_tasklocal_dest_rank_table(
            replicas=replicas,
            model=model,
            layers=int(layers),
            experts=int(experts),
            node_count=int(node_count),
        )
        cur_proxy = _tasklocal_grouped_comm_proxy(node_counts, cur_table, model, cfg)
        selected.append(
            {
                "source_node": int(n),
                "layer": int(l),
                "expert": int(e),
                "rr_owner": int(e % world),
                "target_rank": int(target_rank),
                "rank": int(target_rank),
                "target_node": int(model.rank_to_node.get(int(target_rank), int(target_rank))),
                "replica_node": int(model.rank_to_node.get(int(target_rank), int(target_rank))),
                "cross_node_replica": True,
                "delta_grouped_J_comm": float(cand.get("J_comm_gain_ms", 0.0)),
                "J_comm_gain_ms": float(cand.get("J_comm_gain_ms", 0.0)),
                "cross_node_bytes_reduction": float(cand.get("cross_node_bytes_reduction", 0.0)),
                "active_cross_node_event_reduction": int(cand.get("active_cross_node_event_reduction", 0)),
                "cross_node_events_eliminated": int(cand.get("cross_node_events_eliminated", 0)),
                "cross_node_events_created": int(cand.get("cross_node_events_created", 0)),
            }
        )

    valid_proxy = None
    valid_baseline_proxy = None
    if valid_trace_profile is not None:
        valid_counts = _task_local_node_counts(valid_trace_profile, str(cfg.stage), int(layers), int(experts))
        valid_baseline_proxy = _tasklocal_grouped_comm_proxy(valid_counts, baseline_table, model, cfg)
        valid_proxy = _tasklocal_grouped_comm_proxy(valid_counts, cur_table, model, cfg)

    before = float(baseline_J)
    after = float(_state_J_comm(state))
    reduction_pct = 100.0 * (before - after) / max(before, 1e-9)
    valid_reduction_pct = None
    if valid_proxy is not None and valid_baseline_proxy is not None:
        vb = float(valid_baseline_proxy["grouped_J_comm"])
        va = float(valid_proxy["grouped_J_comm"])
        valid_reduction_pct = 100.0 * (vb - va) / max(vb, 1e-9)
    runtime_packet_before = _runtime_packet_metrics_from_table(
        trace_profile=trace_profile,
        dest_table=baseline_table,
        model=model,
        stage=str(cfg.stage),
        layers=int(layers),
        experts=int(experts),
    )
    runtime_packet_after = _runtime_packet_metrics_from_table(
        trace_profile=trace_profile,
        dest_table=cur_table,
        model=model,
        stage=str(cfg.stage),
        layers=int(layers),
        experts=int(experts),
    )
    runtime_packet_cross_events_before = int(runtime_packet_before["cross_events"])
    runtime_packet_cross_events_after = int(runtime_packet_after["cross_events"])
    runtime_packet_cross_event_reduction_pct = _metric_drop_pct(
        float(runtime_packet_cross_events_before),
        float(runtime_packet_cross_events_after),
    )
    runtime_packet_cross_items_before = int(runtime_packet_before["cross_items"])
    runtime_packet_cross_items_after = int(runtime_packet_after["cross_items"])
    runtime_packet_cross_item_reduction_pct = _metric_drop_pct(
        float(runtime_packet_cross_items_before),
        float(runtime_packet_cross_items_after),
    )
    runtime_packet_cost_before = float(runtime_packet_before["cost"])
    runtime_packet_cost_after = float(runtime_packet_after["cost"])
    runtime_packet_cost_reduction_pct = _metric_drop_pct(runtime_packet_cost_before, runtime_packet_cost_after)

    noop_reason = ""
    if not selected:
        noop_reason = "no_candidate_lowers_grouped_J_comm"
    elif valid_reduction_pct is not None and float(valid_reduction_pct) <= 0.0:
        noop_reason = "valid_grouped_J_comm_not_positive"
    else:
        noop_reason = _runtime_packet_gate_reason(
            require_event_gain_when_latency=bool(cfg.require_event_gain_when_latency),
            latency_ms=float(cfg.latency_ms_per_event or 0.0),
            runtime_packet_cross_event_reduction_pct=float(runtime_packet_cross_event_reduction_pct),
            runtime_packet_cost_reduction_pct=float(runtime_packet_cost_reduction_pct),
        )
    placement = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=topology,
        layers=int(layers),
        experts=int(experts),
        objective=str(cfg.objective),
        cfg=cfg,
    )
    placement["format"] = "replicated_scp_ep_v3"
    placement["task_locality_aware"] = True
    placement["diagnostic_mode"] = "tasklocal"
    placement["source_nodes"] = list(range(int(node_count)))
    placement["dest_rank_table"] = cur_table
    placement["dest_rank_table_prefill"] = cur_table
    # v1 TaskLocal-SCP is prefill-only.  Keep decode explicitly at RR so an
    # accidental SPARSE_SCP_APPLY_DECODE=1 cannot reuse the prefill placement.
    placement["dest_rank_table_decode"] = baseline_table
    placement["runtime_assignment_semantics"] = (
        "TaskLocal-SCP placement-only contract: router top-k/expert ids/scores/weights stay unchanged; "
        "runtime indexes dest_rank_table[layer][source_node][expert] for enabled prefill SCP; "
        "decode table is RR in v1 until a decode-safe solver is used."
    )
    placement["profile_metadata"] = profile_meta or {}
    if valid_profile_meta:
        placement["valid_profile_metadata"] = valid_profile_meta
    placement["selected_replicas"] = selected

    selected_by_node: Dict[str, int] = {}
    selected_by_direction: Dict[str, int] = {}
    for item in selected:
        sn = str(int(item["source_node"]))
        tn = str(int(item["target_node"]))
        selected_by_node[sn] = int(selected_by_node.get(sn, 0)) + 1
        selected_by_direction[f"{sn}->{tn}"] = int(selected_by_direction.get(f"{sn}->{tn}", 0)) + 1

    offline = {
        "objective": str(cfg.objective),
        "stage": str(cfg.stage),
        "num_extra_replicas": int(len(selected)),
        "selected_replicas": selected,
        "selected_replicas_by_source_node": selected_by_node,
        "selected_replicas_by_direction": selected_by_direction,
        "safety_rejected_candidates": rejected,
        "grouped_J_comm_before": float(before),
        "grouped_J_comm_after": float(after),
        "grouped_J_comm_reduction_pct": float(reduction_pct),
        "baseline_grouped_J_comm_ms": float(before),
        "replicated_grouped_J_comm_ms": float(after),
        "valid_grouped_J_comm_before": None if valid_baseline_proxy is None else float(valid_baseline_proxy["grouped_J_comm"]),
        "valid_grouped_J_comm_after": None if valid_proxy is None else float(valid_proxy["grouped_J_comm"]),
        "valid_grouped_J_comm_reduction_pct": valid_reduction_pct,
        "cross_node_bytes_before": float(baseline_proxy["cross_node_bytes"]),
        "cross_node_bytes_after": float(cur_proxy["cross_node_bytes"]),
        "active_cross_node_events_before": int(runtime_packet_cross_events_before),
        "active_cross_node_events_after": int(runtime_packet_cross_events_after),
        "runtime_packet_cross_events_before": int(runtime_packet_cross_events_before),
        "runtime_packet_cross_events_after": int(runtime_packet_cross_events_after),
        "runtime_packet_cross_event_reduction_pct": float(runtime_packet_cross_event_reduction_pct),
        "runtime_packet_cross_items_before": int(runtime_packet_cross_items_before),
        "runtime_packet_cross_items_after": int(runtime_packet_cross_items_after),
        "runtime_packet_cross_item_reduction_pct": float(runtime_packet_cross_item_reduction_pct),
        "runtime_packet_cost_before": float(runtime_packet_cost_before),
        "runtime_packet_cost_after": float(runtime_packet_cost_after),
        "runtime_packet_cost_reduction_pct": float(runtime_packet_cost_reduction_pct),
        "runtime_packet_metrics_before": runtime_packet_before,
        "runtime_packet_metrics_after": runtime_packet_after,
        "grouped_active_cross_node_events_before": int(baseline_proxy["active_cross_node_events"]),
        "grouped_active_cross_node_events_after": int(cur_proxy["active_cross_node_events"]),
        "grouped_active_cross_node_events_reduction_pct": 100.0
        * (float(baseline_proxy["active_cross_node_events"]) - float(cur_proxy["active_cross_node_events"]))
        / max(float(baseline_proxy["active_cross_node_events"]), 1.0),
        "cross_node_reduction_pct": 100.0 * (float(baseline_proxy["cross_node_bytes"]) - float(cur_proxy["cross_node_bytes"])) / max(float(baseline_proxy["cross_node_bytes"]), 1.0),
        "cross_node_bytes_reduction_pct": 100.0 * (float(baseline_proxy["cross_node_bytes"]) - float(cur_proxy["cross_node_bytes"])) / max(float(baseline_proxy["cross_node_bytes"]), 1.0),
        "rankmax_inter_node_bytes_reduction_pct": float(reduction_pct),
        "rank_runtime_proxy_reduction_pct": float(reduction_pct),
        "objective_reduction_pct": float(reduction_pct),
        "comm_reduction_pct": float(reduction_pct),
        "baseline_objective_ms": float(before),
        "replicated_objective_ms": float(after),
        "baseline_rankmax_comm_ms": float(before),
        "replicated_rankmax_comm_ms": float(after),
        "rankmax_comm_reduction_pct": float(reduction_pct),
        "baseline_rankmax_h2d_ms": 0.0,
        "replicated_rankmax_h2d_ms": 0.0,
        "rankmax_h2d_change_pct": 0.0,
        "baseline_resident_hit_rate_by_item": 0.0,
        "replicated_resident_hit_rate_by_item": 0.0,
        "baseline_cross_node_tokens": float(baseline_proxy["cross_node_bytes"]) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "replicated_cross_node_tokens": float(cur_proxy["cross_node_bytes"]) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "baseline_cross_node_items": float(baseline_proxy["cross_node_bytes"]) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "replicated_cross_node_items": float(cur_proxy["cross_node_bytes"]) / max(float(model.dispatch_item_bytes + model.combine_item_bytes), 1.0),
        "load_imbalance_ratio": 0.0,
        "h2d_miss_proxy_change_pct": 0.0,
        "compute_proxy_rankmax_change_pct": 0.0,
        "throughput_objective_enabled": False,
        "h2d_ms_per_proxy": float(cfg.h2d_ms_per_proxy),
        "compute_ms_per_proxy": float(cfg.compute_ms_per_proxy),
        "decode_rs_gain_weight": 0.0,
        "prefill_runtime_assignment_comm_reduction_pct": float(reduction_pct) if str(cfg.stage).lower() in ("prefill", "mixed") else None,
        "decode_runtime_assignment_comm_reduction_pct": None,
        "scp_stackability_status": "prefill_placement_only_rs_can_prefetch_decode_rr_targets",
        "scp_stackability_reasons": [],
        "selected_cross_node_replicas": int(len(selected)),
        "selected_same_node_replicas": 0,
        "selected_event_reducing_replicas": int(sum(1 for row in selected if int(row.get("active_cross_node_event_reduction", 0)) > 0)),
        "selected_cross_node_events_eliminated": int(sum(int(row.get("cross_node_events_eliminated", 0)) for row in selected)),
        "selected_cross_node_events_created": int(sum(int(row.get("cross_node_events_created", 0)) for row in selected)),
        "active_cross_node_events_reduction_pct": float(runtime_packet_cross_event_reduction_pct),
        "should_run_runtime": bool(not noop_reason),
        "scp_noop": bool(noop_reason),
        "noop_reason": str(noop_reason),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "bytes_source": str(model.bytes_source),
        "latency_semantics": str(model.latency_semantics),
        "cost_unit": str(model.cost_unit),
        "h2d_model_semantics": "not_modeled_in_task_local_tail_scp_v1_first_version; decode SCP stays disabled by default",
        "runtime_assignment_policy": "dest_rank_table[layer][source_node][expert]_prefill_only_decode_rr",
        "gate_status": {
            "h2d_cache_compute_gates": "not_positive_rewards; v1 uses communication gate and valid positive check",
            "valid_positive": bool(valid_reduction_pct is None or valid_reduction_pct > 0.0),
            "runtime_packet_cost_positive": bool(float(runtime_packet_cost_reduction_pct) > 0.0),
            "runtime_packet_event_positive": bool(float(runtime_packet_cross_event_reduction_pct) > 0.0),
            "runtime_packet_gate": "pass" if not noop_reason else str(noop_reason),
            "runtime_packet_gate_semantics": "latency guard uses replayed event+payload cost; event-only non-reduction is reported but not a no-op when payload cost drops",
        },
    }
    placement["offline_metrics"] = offline
    return placement


def _rank_runtime_proxy_by_rank(m: LayerMetrics, cfg: ReplicatedSCPConfig) -> np.ndarray:
    return (
        np.asarray(m.comm_cost_by_source_rank, dtype=np.float64)
        + np.asarray(m.latency_event_cost_by_source_rank, dtype=np.float64)
        + float(cfg.compute_ms_per_proxy) * np.asarray(m.compute_proxy_by_rank, dtype=np.float64)
        + float(cfg.h2d_ms_per_proxy) * np.asarray(m.h2d_miss_proxy_by_rank, dtype=np.float64)
    )


def _rank_runtime_proxy_rankmax(m: LayerMetrics, cfg: ReplicatedSCPConfig) -> float:
    return float(_rank_runtime_proxy_by_rank(m, cfg).max(initial=0.0))


def _is_cross_node_replica(model: CommModel, expert: int, rank: int, world: int) -> bool:
    rr_owner = int(expert) % int(world)
    return int(model.rank_to_node.get(int(rr_owner), int(rr_owner))) != int(model.rank_to_node.get(int(rank), int(rank)))


def _layer_objective(m: LayerMetrics, cfg: ReplicatedSCPConfig) -> float:
    """Return the scalar objective minimized by greedy replica selection."""
    if _is_throughput_objective(cfg):
        base = _rank_runtime_proxy_rankmax(m, cfg)
        return (
            base
            + float(cfg.lambda_event) * float(m.active_cross_node_pairs)
            + float(cfg.lambda_small_packet) * float(m.small_cross_node_pairs)
        )
    elif str(cfg.objective).lower() in ("prefill_ranktail_v1", "ranktail", "rank_tail"):
        base = float(m.comm_cost_rankmax)
    else:
        base = float(m.comm_cost)
    return (
        base
        + float(cfg.lambda_event) * float(m.active_cross_node_pairs)
        + float(cfg.lambda_small_packet) * float(m.small_cross_node_pairs)
        + float(cfg.compute_ms_per_proxy) * float(max(m.compute_proxy_rankmax, 0.0))
        + float(cfg.h2d_ms_per_proxy) * float(max(m.h2d_miss_proxy_rankmax, 0.0))
    )


def _cache_replay_metrics(
    *,
    trace_profile: Dict[str, Any],
    topology: Dict[str, Any],
    replicas: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    layers: int,
    experts: int,
    mode: str = "no_rs",
) -> Dict[str, Any]:
    if evaluate_profile_placement_with_cache_replay is None:
        raise RuntimeError("cacheaware_ranktail_v1 requires scripts/scp_cache_replay.py to be importable")
    placement = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=topology,
        layers=layers,
        experts=experts,
        objective=str(cfg.objective),
        cfg=cfg,
    )
    warmup_placement = None
    warm_cache = False
    warmup_stage = None
    if str(cfg.stage or "").lower() == "decode":
        # Runtime decode is not a cold-cache phase: prefill has already touched
        # experts.  When validating decode-only SCP, warm the cache with RR
        # prefill first, then evaluate decode under the candidate placement.
        warmup_placement = _base_placement_obj(
            replicas=round_robin_replicas(layers, experts, model.world),
            model=model,
            topology=topology,
            layers=layers,
            experts=experts,
            objective=str(cfg.objective),
            cfg=cfg,
        )
        warm_cache = True
        warmup_stage = "prefill"
    replay_topology = dict(topology)
    _inject_runtime_scales(replay_topology, cfg)
    compute_ms_per_item = float(cfg.compute_ms_per_proxy) if float(cfg.compute_ms_per_proxy) > 0.0 else None
    return evaluate_profile_placement_with_cache_replay(
        trace_profile,
        placement,
        replay_topology,
        cache_capacity=int(cfg.expert_cache_capacity),
        cache_capacity_by_rank=cfg.expert_cache_capacity_by_rank,
        cache_policy="lru",
        mode=str(mode),
        stage=_stage_for_trace(cfg),
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
        compute_ms_per_item=compute_ms_per_item,
        warm_cache=warm_cache,
        warmup_placement=warmup_placement,
        warmup_stage=warmup_stage,
    )


def _inject_runtime_scales(topology: Dict[str, Any], cfg: ReplicatedSCPConfig) -> Dict[str, Any]:
    # Keep replay aligned with solver/runtime scale overrides. The standalone
    # replay CLI reads scales from topology; the solver may receive them via
    # CLI/env instead, so inject the resolved time multipliers here.
    topology["rank_compute_time_scale"] = [float(x) for x in cfg.rank_compute_time_scale]
    topology["rank_h2d_time_scale"] = [float(x) for x in cfg.rank_h2d_time_scale]
    if cfg.latency_ms_per_event is not None:
        bw = dict(topology.get("bandwidth_model", {}) or {})
        bw["cross_node_latency_ms"] = float(cfg.latency_ms_per_event)
        topology["bandwidth_model"] = bw
    return topology


def _online_proxy_metrics(
    *,
    trace_profile: Dict[str, Any],
    topology: Dict[str, Any],
    replicas: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    layers: int,
    experts: int,
) -> Dict[str, Any]:
    """Evaluate the same offline proxy used by the SCP online-aware summary.

    The proxy is prefill communication tail plus decode cache/H2D/compute replay
    after prefill has warmed the cache under the same placement.
    """
    if evaluate_grouped_comm_tail is None or evaluate_profile_placement_with_cache_replay is None:
        raise RuntimeError("online_aware_scp_v1 requires scripts/scp_cache_replay.py")
    placement = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=topology,
        layers=layers,
        experts=experts,
        objective=str(cfg.objective),
        cfg=cfg,
    )
    replay_topology = _inject_runtime_scales(dict(topology), cfg)
    prefill_comm = evaluate_grouped_comm_tail(
        trace_profile,
        placement,
        replay_topology,
        stage="prefill",
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
    )
    compute_ms_per_item = float(cfg.compute_ms_per_proxy) if float(cfg.compute_ms_per_proxy) > 0.0 else None
    decode_cache = evaluate_profile_placement_with_cache_replay(
        trace_profile,
        placement,
        replay_topology,
        cache_capacity=int(cfg.expert_cache_capacity),
        cache_capacity_by_rank=cfg.expert_cache_capacity_by_rank,
        cache_policy="lru",
        mode="no_rs",
        stage="decode",
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
        compute_ms_per_item=compute_ms_per_item,
        warm_cache=True,
        warmup_stage="prefill",
    )
    return {
        "format": "online_aware_scp_proxy_v1",
        "online_proxy_ms": float(prefill_comm["J_comm_ms"]) + float(decode_cache["objective_ms"]),
        "prefill_comm": prefill_comm,
        "decode_cache": decode_cache,
        "prefill_tail_ms": float(prefill_comm["J_comm_ms"]),
        "decode_objective_ms": float(decode_cache["objective_ms"]),
        "decode_hit_item": float(decode_cache.get("resident_hit_rate_by_item", 0.0) or 0.0),
        "decode_h2d_mb": float(decode_cache.get("h2d_demand_mb", 0.0) or 0.0),
        "decode_rankmax_h2d_ms": float(decode_cache.get("rankmax_h2d_ms", 0.0) or 0.0),
        "decode_rankmax_comm_ms": float(decode_cache.get("rankmax_comm_ms", 0.0) or 0.0),
        "decode_rankmax_compute_items": float(decode_cache.get("rankmax_compute_items", 0.0) or 0.0),
        "decode_cross_node_items": int(decode_cache.get("cross_node_items", 0) or 0),
        "decode_cross_node_events": int(decode_cache.get("active_cross_node_pair_events", 0) or 0),
    }


def _cache_replay_candidate_prefilter(
    A: np.ndarray,
    model: CommModel,
    reps: Dict[int, Dict[int, List[int]]],
    cfg: ReplicatedSCPConfig,
    *,
    extra_by_layer_rank: np.ndarray,
) -> List[Tuple[float, int, int, int]]:
    layers, world, experts = A.shape
    rows: List[Tuple[float, int, int, int]] = []
    for layer in range(layers):
        for expert in range(experts):
            if len(reps[layer][expert]) >= int(cfg.max_replicas_per_expert):
                continue
            rr_owner = int(expert) % int(world)
            rr_owner_node = int(model.rank_to_node.get(rr_owner, rr_owner))
            for rank in range(world):
                if rank in reps[layer][expert]:
                    continue
                if cfg.extra_replicas_per_rank_per_layer > 0 and extra_by_layer_rank[layer, rank] >= cfg.extra_replicas_per_rank_per_layer:
                    continue
                replica_node = int(model.rank_to_node.get(int(rank), int(rank)))
                if not bool(cfg.allow_same_node_replicas) and int(replica_node) == int(rr_owner_node):
                    continue
                benefit = replica_benefit(A, model, reps, layer, expert, rank)
                if benefit <= max(float(cfg.min_benefit), 0.0):
                    continue
                rows.append((float(benefit), int(layer), int(expert), int(rank)))
    rows.sort(key=lambda x: (-float(x[0]), int(x[1]), int(x[2]), int(x[3])))
    topn = int(cfg.candidate_prefilter_topn)
    if topn > 0:
        return rows[:topn]
    return rows


def _metric_pct_drop(before: float, after: float) -> float:
    return float(100.0 * (float(before) - float(after)) / max(float(before), 1e-9))


def solve_cacheaware_ranktail(
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    *,
    topology: Dict[str, Any],
    trace_profile: Dict[str, Any],
    valid_trace_profile: Optional[Dict[str, Any]] = None,
    profile_meta: Optional[Dict[str, Any]] = None,
    valid_profile_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if trace_profile is None:
        raise RuntimeError("cacheaware_ranktail_v1 requires ordered routing trace")
    layers, world, experts = A.shape
    replicas = round_robin_replicas(layers, experts, world)
    reps = _as_int_replicas(replicas)
    extra_by_layer = np.zeros((layers,), dtype=np.int64)
    extra_by_layer_rank = np.zeros((layers, world), dtype=np.int64)
    selected: List[Dict[str, Any]] = []
    rejected = {
        "prefiltered_out": 0,
        "no_replay_gain": 0,
        "h2d_gate": 0,
        "cache_hit_gate": 0,
        "comm_gate": 0,
        "rs_trace_missing": 0,
    }

    rr_replicas = round_robin_replicas(layers, experts, world)
    rr_no_rs = _cache_replay_metrics(
        trace_profile=trace_profile,
        topology=topology,
        replicas=rr_replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
        mode="no_rs",
    )
    current_no_rs = dict(rr_no_rs)
    rs_replay_status: Dict[str, Any] = {"rs_replay_available": False, "reason": "RS trace not present"}
    if bool(cfg.require_rs_safe):
        rejected["rs_trace_missing"] += 1
        raise RuntimeError("--require-rs-safe was set, but RS replay trace is not available in current profile")

    budget = int(cfg.global_extra_replicas) if int(cfg.global_extra_replicas) > 0 else int(cfg.extra_replicas_per_layer) * int(layers)
    budget = max(int(budget), 0)
    for _step in range(budget):
        candidates = _cache_replay_candidate_prefilter(
            A,
            model,
            reps,
            cfg,
            extra_by_layer_rank=extra_by_layer_rank,
        )
        if not candidates:
            break
        best: Optional[Tuple[float, Dict[str, Any], Dict[str, Dict[str, List[int]]], Dict[str, Any]]] = None
        evaluated = 0
        for approx_gain, layer, expert, rank in candidates:
            if int(cfg.global_extra_replicas) <= 0 and extra_by_layer[layer] >= int(cfg.extra_replicas_per_layer):
                continue
            trial_replicas = json.loads(json.dumps(replicas))
            trial_replicas[str(layer)][str(expert)].append(int(rank))
            after = _cache_replay_metrics(
                trace_profile=trace_profile,
                topology=topology,
                replicas=trial_replicas,
                model=model,
                cfg=cfg,
                layers=layers,
                experts=experts,
                mode="no_rs",
            )
            evaluated += 1
            gain = float(current_no_rs["objective_ms"]) - float(after["objective_ms"])
            if gain <= float(cfg.min_net_gain_ms):
                rejected["no_replay_gain"] += 1
                continue
            if not bool(cfg.disable_h2d_gate):
                if float(after["rankmax_h2d_ms"]) > float(rr_no_rs["rankmax_h2d_ms"]) * (1.0 + max(float(cfg.h2d_slack), 0.0)) + 1e-9:
                    rejected["h2d_gate"] += 1
                    continue
            if float(after["resident_hit_rate_by_item"]) < float(rr_no_rs["resident_hit_rate_by_item"]) - max(float(cfg.cache_hit_drop_limit), 0.0) - 1e-12:
                rejected["cache_hit_gate"] += 1
                continue
            if float(after["rankmax_comm_ms"]) >= float(current_no_rs["rankmax_comm_ms"]) and float(approx_gain) <= 0.0:
                rejected["comm_gate"] += 1
                continue
            score = (gain, float(current_no_rs["rankmax_comm_ms"]) - float(after["rankmax_comm_ms"]), float(approx_gain), -int(layer), -int(expert), -int(rank))
            row = {
                "layer": int(layer),
                "expert": int(expert),
                "rank": int(rank),
                "rr_owner": int(expert % world),
                "rr_owner_node": int(model.rank_to_node.get(int(expert % world), int(expert % world))),
                "replica_node": int(model.rank_to_node.get(int(rank), int(rank))),
                "cross_node_replica": bool(model.rank_to_node.get(int(expert % world), int(expert % world)) != model.rank_to_node.get(int(rank), int(rank))),
                "approx_comm_gain": float(approx_gain),
                "gain_noRS_objective_ms": float(gain),
                "objective_before_ms": float(current_no_rs["objective_ms"]),
                "objective_after_ms": float(after["objective_ms"]),
                "rankmax_comm_before_ms": float(current_no_rs["rankmax_comm_ms"]),
                "rankmax_comm_after_ms": float(after["rankmax_comm_ms"]),
                "rankmax_h2d_before_ms": float(current_no_rs["rankmax_h2d_ms"]),
                "rankmax_h2d_after_ms": float(after["rankmax_h2d_ms"]),
                "hit_rate_item_before": float(current_no_rs["resident_hit_rate_by_item"]),
                "hit_rate_item_after": float(after["resident_hit_rate_by_item"]),
            }
            if best is None or score > best[0]:
                best = (score, row, trial_replicas, after)
        rejected["prefiltered_out"] += max(0, len(candidates) - evaluated)
        if best is None:
            break
        _score, row, replicas, current_no_rs = best
        reps = _as_int_replicas(replicas)
        layer = int(row["layer"])
        rank = int(row["rank"])
        extra_by_layer[layer] += 1
        extra_by_layer_rank[layer, rank] += 1
        selected.append(row)

    train_no_rs = current_no_rs
    valid_no_rs = None
    valid_gain = None
    overfit_warning = False
    if valid_trace_profile is not None:
        valid_rr = _cache_replay_metrics(
            trace_profile=valid_trace_profile,
            topology=topology,
            replicas=rr_replicas,
            model=model,
            cfg=cfg,
            layers=layers,
            experts=experts,
            mode="no_rs",
        )
        valid_no_rs = _cache_replay_metrics(
            trace_profile=valid_trace_profile,
            topology=topology,
            replicas=replicas,
            model=model,
            cfg=cfg,
            layers=layers,
            experts=experts,
            mode="no_rs",
        )
        valid_gain = float(valid_rr["objective_ms"]) - float(valid_no_rs["objective_ms"])
        overfit_warning = bool((float(rr_no_rs["objective_ms"]) - float(train_no_rs["objective_ms"])) > 0.0 and float(valid_gain) < 0.0)
        if bool(cfg.require_valid_positive) and float(valid_gain) <= 0.0:
            # Keep the file valid but mark it as a no-op so run scripts can skip runtime.
            replicas = rr_replicas
            selected = []
            train_no_rs = rr_no_rs

    selected_cross_node_replicas = int(sum(1 for row in selected if bool(row.get("cross_node_replica", False))))
    selected_same_node_replicas = int(len(selected) - selected_cross_node_replicas)
    cross_node_bytes_reduction_pct = _metric_pct_drop(float(rr_no_rs["cross_node_bytes"]), float(train_no_rs["cross_node_bytes"]))
    rankmax_inter_node_bytes_reduction_pct = _metric_pct_drop(
        float(rr_no_rs.get("inter_node_total_bytes_rankmax", 0.0)),
        float(train_no_rs.get("inter_node_total_bytes_rankmax", 0.0)),
    )
    noop_reasons: List[str] = []
    if selected_cross_node_replicas <= 0:
        noop_reasons.append("no_cross_node_replicas_selected")
    if cross_node_bytes_reduction_pct < float(cfg.min_inter_node_bytes_reduction_pct):
        noop_reasons.append("cross_node_bytes_reduction_below_threshold")
    if rankmax_inter_node_bytes_reduction_pct < float(cfg.min_inter_node_rankmax_bytes_reduction_pct):
        noop_reasons.append("rankmax_inter_node_bytes_reduction_below_threshold")

    metrics = {
        "objective": str(cfg.objective),
        "baseline_is": "RR + configured per-rank LRU expert cache budget",
        "baseline_objective_ms": float(rr_no_rs["objective_ms"]),
        "replicated_objective_ms": float(train_no_rs["objective_ms"]),
        "objective_reduction_pct": _metric_pct_drop(float(rr_no_rs["objective_ms"]), float(train_no_rs["objective_ms"])),
        "baseline_rankmax_comm_ms": float(rr_no_rs["rankmax_comm_ms"]),
        "replicated_rankmax_comm_ms": float(train_no_rs["rankmax_comm_ms"]),
        "rankmax_comm_reduction_pct": _metric_pct_drop(float(rr_no_rs["rankmax_comm_ms"]), float(train_no_rs["rankmax_comm_ms"])),
        "baseline_rankmax_h2d_ms": float(rr_no_rs["rankmax_h2d_ms"]),
        "replicated_rankmax_h2d_ms": float(train_no_rs["rankmax_h2d_ms"]),
        "rankmax_h2d_change_pct": float(100.0 * (float(train_no_rs["rankmax_h2d_ms"]) - float(rr_no_rs["rankmax_h2d_ms"])) / max(float(rr_no_rs["rankmax_h2d_ms"]), 1e-9)),
        "baseline_resident_hit_rate_by_item": float(rr_no_rs["resident_hit_rate_by_item"]),
        "replicated_resident_hit_rate_by_item": float(train_no_rs["resident_hit_rate_by_item"]),
        "resident_hit_rate_item_delta": float(train_no_rs["resident_hit_rate_by_item"]) - float(rr_no_rs["resident_hit_rate_by_item"]),
        "baseline_cross_node_items": int(rr_no_rs["cross_node_items"]),
        "replicated_cross_node_items": int(train_no_rs["cross_node_items"]),
        "cross_node_reduction_pct": _metric_pct_drop(float(rr_no_rs["cross_node_items"]), float(train_no_rs["cross_node_items"])),
        "baseline_cross_node_bytes": float(rr_no_rs["cross_node_bytes"]),
        "replicated_cross_node_bytes": float(train_no_rs["cross_node_bytes"]),
        "cross_node_bytes_reduction_pct": float(cross_node_bytes_reduction_pct),
        "baseline_inter_node_total_bytes_rankmax": float(rr_no_rs.get("inter_node_total_bytes_rankmax", 0.0)),
        "replicated_inter_node_total_bytes_rankmax": float(train_no_rs.get("inter_node_total_bytes_rankmax", 0.0)),
        "rankmax_inter_node_bytes_reduction_pct": float(rankmax_inter_node_bytes_reduction_pct),
        "baseline_active_cross_node_pair_events": int(rr_no_rs["active_cross_node_pair_events"]),
        "replicated_active_cross_node_pair_events": int(train_no_rs["active_cross_node_pair_events"]),
        "num_extra_replicas": int(len(selected)),
        "selected_cross_node_replicas": int(selected_cross_node_replicas),
        "selected_same_node_replicas": int(selected_same_node_replicas),
        "extra_replicas_per_layer": extra_by_layer.astype(int).tolist(),
        "extra_replicas_per_rank_per_layer": extra_by_layer_rank.astype(int).tolist(),
        "candidate_prefilter_topn": int(cfg.candidate_prefilter_topn),
        "h2d_slack": float(cfg.h2d_slack),
        "disable_h2d_gate": bool(cfg.disable_h2d_gate),
        "cache_hit_drop_limit": float(cfg.cache_hit_drop_limit),
        "rs_replay_status": rs_replay_status,
        "valid_noRS_gain_ms": valid_gain,
        "overfit_warning": bool(overfit_warning),
        "safety_rejected_candidates": rejected,
        "cache_replay_baseline": rr_no_rs,
        "cache_replay_replicated": train_no_rs,
        "cache_replay_valid": valid_no_rs,
        "h2d_model_semantics": "ordered_trace_lru_cache_replay",
        "runtime_assignment_policy": "nearest_replica_by_replica_choice_cost_matrix_tie_keep_round_robin_owner",
        "should_run_runtime": bool(len(noop_reasons) == 0 and (valid_gain is None or valid_gain > 0.0 or not bool(cfg.require_valid_positive))),
        "scp_noop": bool(len(noop_reasons) > 0),
        "noop_reason": ";".join(noop_reasons) if noop_reasons else "",
        "cost_unit": str(model.cost_unit),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "hidden_bytes": float(model.hidden_bytes),
        "router_weight_bytes": float(model.router_weight_bytes),
        "expert_id_bytes": float(model.expert_id_bytes),
        "bytes_source": str(model.bytes_source),
        "latency_semantics": str(model.latency_semantics),
    }
    result = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=topology,
        layers=layers,
        experts=experts,
        objective=str(cfg.objective),
        cfg=cfg,
    )
    result.update(
        {
            "selected_replicas": selected,
            "offline_metrics": metrics,
            "profile_metadata": profile_meta or {},
            "valid_profile_metadata": valid_profile_meta or {},
        }
    )
    return result


def _stage_for_trace(cfg: ReplicatedSCPConfig) -> str:
    stage = str(cfg.stage or "mixed").lower()
    if stage in ("prefill", "decode"):
        return stage
    return "all"


def _metric_drop_pct(before: float, after: float) -> float:
    return float(100.0 * (float(before) - float(after)) / max(float(before), 1e-9))


def _state_comm_ms(
    *,
    s: int,
    r: int,
    items: float,
    item_bytes: float,
    same_node_bw: float,
    cross_node_bw: float,
    latency_ms: float,
    rank_to_node: Dict[int, int],
) -> float:
    if float(items) <= 0.0 or int(s) == int(r):
        return 0.0
    b = float(items) * float(item_bytes)
    if int(rank_to_node.get(int(s), int(s))) == int(rank_to_node.get(int(r), int(r))):
        return b / max(float(same_node_bw) * 1_000_000.0, 1e-9) * 1000.0
    return b / max(float(cross_node_bw) * 1_000_000.0, 1e-9) * 1000.0 + max(float(latency_ms), 0.0)


def _make_grouped_state(
    *,
    trace_profile: Dict[str, Any],
    topology: Dict[str, Any],
    replicas: Dict[str, Dict[str, List[int]]],
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    layers: int,
    experts: int,
) -> Dict[str, Any]:
    if grouped_trace_indices is None:
        raise RuntimeError("tail_headroom_scp_v2 requires scripts/scp_cache_replay.py")
    groups = grouped_trace_indices(trace_profile, stage=_stage_for_trace(cfg))
    world = int(model.world)
    reps = _as_int_replicas(replicas)
    bw = topology.get("bandwidth_model", {}) or {}
    same_node_bw = float(bw.get("same_node_bw_MBps", 12000.0) or 12000.0)
    cross_node_bw = float(bw.get("cross_node_bw_MBps", 100.0) or 100.0)
    latency_ms = float(bw.get("cross_node_latency_ms", 0.0) or 0.0)
    item_bytes = float(model.dispatch_item_bytes) + float(model.combine_item_bytes)

    trace_layers = trace_profile["trace_layer"].astype(np.int64)
    trace_sources = trace_profile["trace_source_rank"].astype(np.int64)
    trace_experts = trace_profile["trace_expert"].astype(np.int64)
    trace_counts = trace_profile["trace_count"].astype(np.int64)

    row_group: List[int] = []
    row_layer: List[int] = []
    row_source: List[int] = []
    row_expert: List[int] = []
    row_count: List[int] = []
    row_dest: List[int] = []
    rows_by_le: Dict[Tuple[int, int], List[int]] = {}
    pair_items = np.zeros((len(groups), world, world), dtype=np.float64)

    for group_id, idxs in enumerate(groups):
        for trace_idx in idxs.tolist():
            cnt = int(trace_counts[int(trace_idx)])
            if cnt <= 0:
                continue
            l = int(trace_layers[int(trace_idx)])
            s = int(trace_sources[int(trace_idx)])
            e = int(trace_experts[int(trace_idx)])
            r = _runtime_nearest_rank_for_expert(source=s, expert=e, replica_ranks=reps[l][e], model=model)
            row_id = len(row_group)
            row_group.append(int(group_id))
            row_layer.append(int(l))
            row_source.append(int(s))
            row_expert.append(int(e))
            row_count.append(int(cnt))
            row_dest.append(int(r))
            rows_by_le.setdefault((int(l), int(e)), []).append(int(row_id))
            pair_items[int(group_id), int(s), int(r)] += float(cnt)

    source_comm = np.zeros((len(groups), world), dtype=np.float64)
    source_cross_bytes = np.zeros((world,), dtype=np.float64)
    for g in range(len(groups)):
        for s in range(world):
            for r in range(world):
                if (
                    int(s) != int(r)
                    and int(model.rank_to_node.get(int(s), int(s))) != int(model.rank_to_node.get(int(r), int(r)))
                ):
                    source_cross_bytes[int(s)] += float(pair_items[g, s, r]) * item_bytes
                source_comm[g, s] += _state_comm_ms(
                    s=s,
                    r=r,
                    items=float(pair_items[g, s, r]),
                    item_bytes=item_bytes,
                    same_node_bw=same_node_bw,
                    cross_node_bw=cross_node_bw,
                    latency_ms=latency_ms,
                    rank_to_node=model.rank_to_node,
                )
    return {
        "groups": groups,
        "world": int(world),
        "layers": int(layers),
        "experts": int(experts),
        "rank_to_node": dict(model.rank_to_node),
        "same_node_bw": float(same_node_bw),
        "cross_node_bw": float(cross_node_bw),
        "latency_ms": float(latency_ms),
        "event_weight_ms": float(max(float(latency_ms), float(cfg.lambda_event))),
        "item_bytes": float(item_bytes),
        "row_group": np.asarray(row_group, dtype=np.int64),
        "row_layer": np.asarray(row_layer, dtype=np.int64),
        "row_source": np.asarray(row_source, dtype=np.int64),
        "row_expert": np.asarray(row_expert, dtype=np.int64),
        "row_count": np.asarray(row_count, dtype=np.int64),
        "row_dest": np.asarray(row_dest, dtype=np.int64),
        "rows_by_le": rows_by_le,
        "pair_items": pair_items,
        "source_cross_bytes": source_cross_bytes,
        "source_comm": source_comm,
    }


def _copy_grouped_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    out["row_dest"] = np.asarray(state["row_dest"], dtype=np.int64).copy()
    out["pair_items"] = np.asarray(state["pair_items"], dtype=np.float64).copy()
    out["source_cross_bytes"] = np.asarray(state["source_cross_bytes"], dtype=np.float64).copy()
    out["source_comm"] = np.asarray(state["source_comm"], dtype=np.float64).copy()
    return out


def _state_J_comm(state: Dict[str, Any]) -> float:
    source_comm = np.asarray(state["source_comm"], dtype=np.float64)
    if int(source_comm.size) <= 0:
        return 0.0
    return float(source_comm.max(axis=1).sum())


def _state_bottleneck_info(state: Dict[str, Any], *, top_fraction: float = 0.20) -> Dict[str, Any]:
    source_comm = np.asarray(state["source_comm"], dtype=np.float64)
    world = int(state["world"])
    if int(source_comm.shape[0]) <= 0:
        return {
            "top_groups": set(),
            "bottleneck_source_by_group": np.zeros((0,), dtype=np.int64),
            "tail_by_group": np.zeros((0,), dtype=np.float64),
            "tail_by_source_rank": np.zeros((world,), dtype=np.float64),
            "tail_by_source_node": {},
            "bottleneck_node": 0,
        }
    bottleneck_source = np.argmax(source_comm, axis=1).astype(np.int64)
    tail_by_group = source_comm[np.arange(source_comm.shape[0]), bottleneck_source]
    positive = np.nonzero(tail_by_group > 0.0)[0]
    if int(positive.size) <= 0:
        top_groups = set()
    else:
        order = positive[np.argsort(-tail_by_group[positive])]
        keep = max(1, int(np.ceil(float(order.size) * float(top_fraction))))
        top_groups = set(int(x) for x in order[:keep].tolist())
    tail_by_source_rank = np.zeros((world,), dtype=np.float64)
    for g, s in enumerate(bottleneck_source.tolist()):
        tail_by_source_rank[int(s)] += float(tail_by_group[int(g)])
    tail_by_source_node: Dict[int, float] = {}
    for s in range(world):
        node = int(state["rank_to_node"].get(int(s), int(s)))
        tail_by_source_node[node] = float(tail_by_source_node.get(node, 0.0) + tail_by_source_rank[s])
    bottleneck_node = int(max(tail_by_source_node.items(), key=lambda kv: (float(kv[1]), -int(kv[0])))[0]) if tail_by_source_node else 0
    return {
        "top_groups": top_groups,
        "bottleneck_source_by_group": bottleneck_source,
        "tail_by_group": tail_by_group,
        "tail_by_source_rank": tail_by_source_rank,
        "tail_by_source_node": tail_by_source_node,
        "bottleneck_node": int(bottleneck_node),
    }


def _candidate_delta_grouped(
    *,
    state: Dict[str, Any],
    model: CommModel,
    reps: Dict[int, Dict[int, List[int]]],
    layer: int,
    expert: int,
    rank: int,
    bottleneck_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    rows = state["rows_by_le"].get((int(layer), int(expert)), [])
    if not rows:
        return None
    world = int(state["world"])
    if int(rank) in reps[int(layer)][int(expert)]:
        return None
    trial_ranks = list(reps[int(layer)][int(expert)]) + [int(rank)]
    pair_items = np.asarray(state["pair_items"], dtype=np.float64)
    row_group = np.asarray(state["row_group"], dtype=np.int64)
    row_source = np.asarray(state["row_source"], dtype=np.int64)
    row_count = np.asarray(state["row_count"], dtype=np.int64)
    row_dest = np.asarray(state["row_dest"], dtype=np.int64)
    source_delta: Dict[Tuple[int, int], float] = {}
    changes: Dict[Tuple[int, int, int, int], int] = {}
    row_moves: List[Tuple[int, int]] = []
    cross_bytes_reduction = 0.0
    active_event_delta = 0
    cross_event_eliminated = 0
    cross_event_created = 0
    cross_event_progress_units = 0.0
    cross_event_progress_items = 0
    rank_to_node = state["rank_to_node"]
    item_bytes = float(state["item_bytes"])
    cross_bytes_by_source_before = np.asarray(state["source_cross_bytes"], dtype=np.float64)
    cross_bytes_delta_by_source = np.zeros((world,), dtype=np.float64)
    for row_id in rows:
        rid = int(row_id)
        g = int(row_group[rid])
        s = int(row_source[rid])
        cnt = int(row_count[rid])
        old = int(row_dest[rid])
        new = _runtime_nearest_rank_for_expert(source=s, expert=int(expert), replica_ranks=trial_ranks, model=model)
        if int(new) == int(old):
            continue
        changes[(g, s, old, int(new))] = int(changes.get((g, s, old, int(new)), 0) + cnt)
        row_moves.append((rid, int(new)))
    if not changes:
        return None

    for (g, s, old, new), cnt in changes.items():
        old_before = float(pair_items[g, s, old])
        new_before = float(pair_items[g, s, new])
        old_after = max(old_before - float(cnt), 0.0)
        new_after = new_before + float(cnt)
        old_ms_before = _state_comm_ms(
            s=s,
            r=old,
            items=old_before,
            item_bytes=item_bytes,
            same_node_bw=float(state["same_node_bw"]),
            cross_node_bw=float(state["cross_node_bw"]),
            latency_ms=float(state["latency_ms"]),
            rank_to_node=rank_to_node,
        )
        old_ms_after = _state_comm_ms(
            s=s,
            r=old,
            items=old_after,
            item_bytes=item_bytes,
            same_node_bw=float(state["same_node_bw"]),
            cross_node_bw=float(state["cross_node_bw"]),
            latency_ms=float(state["latency_ms"]),
            rank_to_node=rank_to_node,
        )
        new_ms_before = _state_comm_ms(
            s=s,
            r=new,
            items=new_before,
            item_bytes=item_bytes,
            same_node_bw=float(state["same_node_bw"]),
            cross_node_bw=float(state["cross_node_bw"]),
            latency_ms=float(state["latency_ms"]),
            rank_to_node=rank_to_node,
        )
        new_ms_after = _state_comm_ms(
            s=s,
            r=new,
            items=new_after,
            item_bytes=item_bytes,
            same_node_bw=float(state["same_node_bw"]),
            cross_node_bw=float(state["cross_node_bw"]),
            latency_ms=float(state["latency_ms"]),
            rank_to_node=rank_to_node,
        )
        delta_ms = (old_ms_after + new_ms_after) - (old_ms_before + new_ms_before)
        source_delta[(int(g), int(s))] = float(source_delta.get((int(g), int(s)), 0.0) + delta_ms)
        old_cross = int(s) != int(old) and int(rank_to_node.get(int(s), int(s))) != int(rank_to_node.get(int(old), int(old)))
        new_cross = int(s) != int(new) and int(rank_to_node.get(int(s), int(s))) != int(rank_to_node.get(int(new), int(new)))
        if old_cross:
            cross_bytes_reduction += float(cnt) * item_bytes
            cross_bytes_delta_by_source[int(s)] -= float(cnt) * item_bytes
            cross_event_progress_items += int(cnt)
            if old_before > 0.0 and old_after <= 0.0:
                active_event_delta -= 1
                cross_event_eliminated += 1
                cross_event_progress_units += 1.0
            elif old_before > 0.0 and old_after < old_before:
                cross_event_progress_units += min(float(cnt) / max(float(old_before), 1e-9), 1.0)
        if new_cross:
            cross_bytes_reduction -= float(cnt) * item_bytes
            cross_bytes_delta_by_source[int(s)] += float(cnt) * item_bytes
            if new_before <= 0.0 and new_after > 0.0:
                active_event_delta += 1
                cross_event_created += 1

    source_delta_by_group: Dict[int, np.ndarray] = {}
    for (g, s), delta in source_delta.items():
        arr = source_delta_by_group.setdefault(int(g), np.zeros((world,), dtype=np.float64))
        arr[int(s)] += float(delta)
    J_gain = 0.0
    gain_by_source_rank = np.zeros((world,), dtype=np.float64)
    lowers_bottleneck = False
    top_groups = bottleneck_info.get("top_groups", set())
    bottleneck_source_by_group = bottleneck_info.get("bottleneck_source_by_group")
    source_comm = np.asarray(state["source_comm"], dtype=np.float64)
    for g, delta_vec in source_delta_by_group.items():
        before = np.asarray(source_comm[int(g)], dtype=np.float64)
        after = before + delta_vec
        before_tail = float(before.max(initial=0.0))
        after_tail = float(after.max(initial=0.0))
        J_gain += before_tail - after_tail
        for s in range(world):
            if float(delta_vec[s]) < 0.0:
                gain_by_source_rank[s] += -float(delta_vec[s])
        if int(g) in top_groups and bottleneck_source_by_group is not None:
            bs = int(bottleneck_source_by_group[int(g)])
            if float(delta_vec[bs]) < 0.0:
                lowers_bottleneck = True
    gain_by_source_node: Dict[int, float] = {}
    for s in range(world):
        node = int(rank_to_node.get(int(s), int(s)))
        gain_by_source_node[node] = float(gain_by_source_node.get(node, 0.0) + gain_by_source_rank[s])
    owner = int(expert) % world
    owner_node = int(rank_to_node.get(owner, owner))
    replica_node = int(rank_to_node.get(int(rank), int(rank)))
    direction = f"node{owner_node}->node{replica_node}"
    event_reduction = -int(active_event_delta)
    event_weight_ms = max(float(state.get("event_weight_ms", state["latency_ms"])), 0.0)
    event_gain_ms = event_weight_ms * float(event_reduction)
    packet_clear_progress_ms = event_weight_ms * float(cross_event_progress_units)
    cross_bytes_by_source_after = np.maximum(cross_bytes_by_source_before + cross_bytes_delta_by_source, 0.0)
    rankmax_cross_bytes_gain = float(
        cross_bytes_by_source_before.max(initial=0.0) - cross_bytes_by_source_after.max(initial=0.0)
    )
    prefilter_score = (
        float(J_gain)
        + float(event_gain_ms)
        + 0.25 * float(packet_clear_progress_ms)
        + 0.05 * float(cross_bytes_reduction / 1_000_000.0)
        - event_weight_ms * max(0, int(active_event_delta))
    )
    return {
        "layer": int(layer),
        "expert": int(expert),
        "rank": int(rank),
        "rr_owner": int(owner),
        "rr_owner_node": int(owner_node),
        "replica_node": int(replica_node),
        "direction": direction,
        "cross_node_replica": bool(owner_node != replica_node),
        "J_comm_gain_ms": float(J_gain),
        "event_gain_ms": float(event_gain_ms),
        "packet_clear_progress_ms": float(packet_clear_progress_ms),
        "cross_node_event_progress_units": float(cross_event_progress_units),
        "cross_node_event_progress_items": int(cross_event_progress_items),
        "cross_node_bytes_reduction": float(cross_bytes_reduction),
        "rankmax_inter_node_bytes_gain": float(rankmax_cross_bytes_gain),
        "cross_node_bytes_delta_by_source_rank": cross_bytes_delta_by_source.astype(float).tolist(),
        "cross_node_bytes_reduction_by_source_rank": (-cross_bytes_delta_by_source).astype(float).tolist(),
        "active_cross_node_event_delta": int(active_event_delta),
        "active_cross_node_event_reduction": int(event_reduction),
        "runtime_packet_event_reduction": int(event_reduction),
        "runtime_packet_events_eliminated": int(cross_event_eliminated),
        "runtime_packet_events_created": int(cross_event_created),
        "runtime_packet_event_progress_units": float(cross_event_progress_units),
        "runtime_packet_event_progress_items": int(cross_event_progress_items),
        "cross_node_events_eliminated": int(cross_event_eliminated),
        "cross_node_events_created": int(cross_event_created),
        "prefilter_score": float(prefilter_score),
        "lowers_bottleneck": bool(lowers_bottleneck),
        "gain_by_source_rank": gain_by_source_rank.astype(float).tolist(),
        "gain_by_source_node": {str(k): float(v) for k, v in sorted(gain_by_source_node.items())},
        "changes": changes,
        "source_delta_by_group": source_delta_by_group,
        "row_moves": row_moves,
    }


def _apply_candidate_delta(state: Dict[str, Any], cand: Dict[str, Any]) -> None:
    cross_delta_by_source = np.asarray(cand.get("cross_node_bytes_delta_by_source_rank", []), dtype=np.float64)
    if cross_delta_by_source.size:
        state["source_cross_bytes"] = np.maximum(
            np.asarray(state["source_cross_bytes"], dtype=np.float64) + cross_delta_by_source,
            0.0,
        )
    for (g, s, old, new), cnt in cand["changes"].items():
        state["pair_items"][int(g), int(s), int(old)] = max(float(state["pair_items"][int(g), int(s), int(old)]) - float(cnt), 0.0)
        state["pair_items"][int(g), int(s), int(new)] += float(cnt)
    for g, delta_vec in cand["source_delta_by_group"].items():
        state["source_comm"][int(g)] += np.asarray(delta_vec, dtype=np.float64)
    for row_id, new in cand["row_moves"]:
        state["row_dest"][int(row_id)] = int(new)


def _direction_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        direction = str(row.get("direction", f"node{row.get('rr_owner_node', '?')}->node{row.get('replica_node', '?')}"))
        out[direction] = int(out.get(direction, 0) + 1)
    return {k: int(v) for k, v in sorted(out.items())}


def solve_tail_headroom_scp_v2(
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    *,
    topology: Dict[str, Any],
    trace_profile: Dict[str, Any],
    valid_trace_profile: Optional[Dict[str, Any]] = None,
    profile_meta: Optional[Dict[str, Any]] = None,
    valid_profile_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if trace_profile is None:
        raise RuntimeError("tail_headroom_scp_v2 requires grouped ordered routing trace")
    if evaluate_grouped_comm_tail is None:
        raise RuntimeError("tail_headroom_scp_v2 requires scripts/scp_cache_replay.py")
    if bool(cfg.require_rs_safe):
        raise RuntimeError("--require-rs-safe was set, but RS prediction/admission trace is not available in current profile")
    layers, world, experts = A.shape
    eval_topology = dict(topology)
    if cfg.latency_ms_per_event is not None:
        bw = dict(eval_topology.get("bandwidth_model", {}) or {})
        bw["cross_node_latency_ms"] = float(cfg.latency_ms_per_event)
        eval_topology["bandwidth_model"] = bw
    replicas = round_robin_replicas(layers, experts, world)
    reps = _as_int_replicas(replicas)
    rr_replicas = round_robin_replicas(layers, experts, world)
    extra_by_layer = np.zeros((layers,), dtype=np.int64)
    extra_by_layer_rank = np.zeros((layers, world), dtype=np.int64)
    selected: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {
        "same_node_replica": 0,
        "rank_budget": 0,
        "max_replicas_per_expert": 0,
        "no_candidate_delta": 0,
        "no_bottleneck_gain": 0,
        "no_event_gain": 0,
        "low_comm_gain": 0,
            "direction_mismatch": 0,
            "inter_node_rankmax_bytes": 0,
            "prefiltered_out": 0,
        "event_prefiltered_out": 0,
        "cache_validation_failed": 0,
        "net_gain": 0,
        "h2d_rankmax": 0,
        "h2d_total": 0,
        "compute_rankmax": 0,
        "hotset_rankmax": 0,
        "hotset_total": 0,
        "load_imbalance": 0,
    }

    rr_eval = evaluate_grouped_comm_tail(
        trace_profile,
        _base_placement_obj(replicas=rr_replicas, model=model, topology=eval_topology, layers=layers, experts=experts, objective=str(cfg.objective), cfg=cfg),
        eval_topology,
        stage=_stage_for_trace(cfg),
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
    )
    state = _make_grouped_state(
        trace_profile=trace_profile,
        topology=eval_topology,
        replicas=replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
    )
    rr_reps_int = _as_int_replicas(rr_replicas)
    baseline_layers: List[LayerMetrics] = []
    current_layers: List[LayerMetrics] = []
    for layer in range(layers):
        _, base_metric = _layer_metric_from_reps(A, model, rr_reps_int[layer], layer, cfg)
        _, cur_metric = _layer_metric_from_reps(A, model, reps[layer], layer, cfg)
        baseline_layers.append(base_metric)
        current_layers.append(cur_metric)
    h2d_gate_enabled = (not bool(cfg.disable_h2d_gate)) and (bool(cfg.require_h2d_nondegrade) or float(cfg.h2d_slack) >= 0.0)
    compute_gate_enabled = bool(cfg.require_compute_nondegrade) or float(cfg.compute_slack) >= 0.0
    hotset_gate_enabled = bool(cfg.require_hotset_nondegrade) or float(cfg.hotset_slack) > 0.0
    h2d_slack = 0.0 if cfg.require_h2d_nondegrade else max(float(cfg.h2d_slack), 0.0)
    compute_slack = 0.0 if cfg.require_compute_nondegrade else max(float(cfg.compute_slack), 0.0)
    hotset_slack = 0.0 if cfg.require_hotset_nondegrade else max(float(cfg.hotset_slack), 0.0)

    budget = int(cfg.global_extra_replicas) if int(cfg.global_extra_replicas) > 0 else int(cfg.extra_replicas_per_layer) * int(layers)
    budget = max(int(budget), 0)
    upper_info = _state_bottleneck_info(state, top_fraction=1.0)
    optimistic: List[float] = []
    for layer in range(layers):
        for expert in range(experts):
            owner = int(expert) % int(world)
            owner_node = int(model.rank_to_node.get(owner, owner))
            for rank in range(world):
                if int(rank) in reps[layer][expert]:
                    continue
                replica_node = int(model.rank_to_node.get(int(rank), int(rank)))
                if not bool(cfg.allow_same_node_replicas) and owner_node == replica_node:
                    continue
                cand = _candidate_delta_grouped(
                    state=state,
                    model=model,
                    reps=reps,
                    layer=layer,
                    expert=expert,
                    rank=rank,
                    bottleneck_info=upper_info,
                )
                if cand is not None and float(cand["J_comm_gain_ms"]) > 0.0:
                    optimistic.append(float(cand["J_comm_gain_ms"]))
    optimistic.sort(reverse=True)
    optimistic_gain = float(sum(optimistic[:budget])) if budget > 0 else 0.0
    optimistic_headroom_pct = _metric_drop_pct(float(rr_eval["J_comm_ms"]), float(rr_eval["J_comm_ms"]) - optimistic_gain)

    rr_cache = None
    last_snapshot: Optional[
        Tuple[
            Dict[str, Dict[str, List[int]]],
            Dict[str, Any],
            List[Dict[str, Any]],
            np.ndarray,
            np.ndarray,
            List[LayerMetrics],
        ]
    ] = None
    accepted_since_validation = 0
    validation_failed = False
    if optimistic_headroom_pct >= float(cfg.min_tail_headroom_pct) and budget > 0:
        if evaluate_profile_placement_with_cache_replay is not None:
            try:
                rr_cache = _cache_replay_metrics(
                    trace_profile=trace_profile,
                    topology=topology,
                    replicas=rr_replicas,
                    model=model,
                    cfg=cfg,
                    layers=layers,
                    experts=experts,
                    mode="no_rs",
                )
            except Exception:
                rr_cache = None
        for _step in range(budget):
            bottleneck_info = _state_bottleneck_info(state, top_fraction=0.20)
            bottleneck_node = int(bottleneck_info["bottleneck_node"])
            event_aware_enabled = bool(cfg.event_aware) and float(state.get("latency_ms", 0.0)) > 0.0
            event_first_enabled = bool(cfg.event_first) and bool(event_aware_enabled)
            candidates: List[Dict[str, Any]] = []
            top_groups = bottleneck_info["top_groups"]
            bottleneck_source_by_group = bottleneck_info["bottleneck_source_by_group"]
            candidate_pairs: set[Tuple[int, int, int]] = set()
            row_group = np.asarray(state["row_group"], dtype=np.int64)
            row_layer = np.asarray(state["row_layer"], dtype=np.int64)
            row_source = np.asarray(state["row_source"], dtype=np.int64)
            row_expert = np.asarray(state["row_expert"], dtype=np.int64)
            for (layer, expert), rows in state["rows_by_le"].items():
                for row_id in rows:
                    g = int(row_group[int(row_id)])
                    if g not in top_groups:
                        continue
                    s = int(row_source[int(row_id)])
                    if not bool(event_aware_enabled) and int(bottleneck_source_by_group[g]) != s:
                        continue
                    if not bool(event_aware_enabled) and int(model.rank_to_node.get(s, s)) != bottleneck_node:
                        continue
                    owner = int(expert) % int(world)
                    owner_node = int(model.rank_to_node.get(owner, owner))
                    target_node = int(model.rank_to_node.get(s, s)) if bool(event_aware_enabled) else int(bottleneck_node)
                    if owner_node == target_node:
                        continue
                    for rank in range(world):
                        if int(model.rank_to_node.get(rank, rank)) != target_node:
                            continue
                        candidate_pairs.add((int(layer), int(expert), int(rank)))
            for layer, expert, rank in sorted(candidate_pairs):
                if len(reps[layer][expert]) >= int(cfg.max_replicas_per_expert):
                    rejected["max_replicas_per_expert"] += 1
                    continue
                if rank in reps[layer][expert]:
                    continue
                owner = int(expert) % int(world)
                owner_node = int(model.rank_to_node.get(owner, owner))
                replica_node = int(model.rank_to_node.get(rank, rank))
                if not bool(cfg.allow_same_node_replicas) and owner_node == replica_node:
                    rejected["same_node_replica"] += 1
                    continue
                if int(cfg.global_extra_replicas) <= 0 and extra_by_layer[layer] >= int(cfg.extra_replicas_per_layer):
                    rejected["rank_budget"] += 1
                    continue
                if cfg.extra_replicas_per_rank_per_layer > 0 and extra_by_layer_rank[layer, rank] >= cfg.extra_replicas_per_rank_per_layer:
                    rejected["rank_budget"] += 1
                    continue
                cand = _candidate_delta_grouped(
                    state=state,
                    model=model,
                    reps=reps,
                    layer=layer,
                    expert=expert,
                    rank=rank,
                    bottleneck_info=bottleneck_info,
                )
                if cand is None:
                    rejected["no_candidate_delta"] += 1
                    continue
                event_reduction = int(cand.get("runtime_packet_event_reduction", cand.get("active_cross_node_event_reduction", 0)))
                event_gain_ms = float(cand.get("event_gain_ms", 0.0))
                packet_progress_ms = float(cand.get("packet_clear_progress_ms", 0.0))
                rankmax_bytes_gain = float(cand.get("rankmax_inter_node_bytes_gain", 0.0))
                if bool(cfg.require_event_gain_when_latency) and bool(event_aware_enabled) and event_reduction <= 0:
                    rejected["no_event_gain"] += 1
                    continue
                if bool(cfg.require_inter_node_rankmax_bytes_gain) and rankmax_bytes_gain <= 1e-9 and event_reduction <= 0:
                    rejected["inter_node_rankmax_bytes"] += 1
                    continue
                if not bool(cand["lowers_bottleneck"]) and event_reduction <= 0:
                    rejected["no_bottleneck_gain"] += 1
                    continue
                if float(cand["J_comm_gain_ms"]) <= float(cfg.min_comm_gain_ms) and event_reduction <= 0:
                    rejected["low_comm_gain"] += 1
                    continue
                if float(cand["gain_by_source_node"].get(str(bottleneck_node), 0.0)) <= 0.0 and event_reduction <= 0:
                    rejected["direction_mismatch"] += 1
                    continue
                trial_reps = {int(e): list(rs) for e, rs in reps[layer].items()}
                trial_reps[int(expert)] = list(trial_reps[int(expert)]) + [int(rank)]
                _, proxy_metric = _layer_metric_from_reps(A, model, trial_reps, layer, cfg)
                base_proxy = baseline_layers[layer]
                cur_proxy = current_layers[layer]
                h2d_total_delta = float(proxy_metric.h2d_miss_proxy_total - cur_proxy.h2d_miss_proxy_total)
                h2d_rankmax_delta = float(proxy_metric.h2d_miss_proxy_rankmax - cur_proxy.h2d_miss_proxy_rankmax)
                compute_rankmax_delta = float(proxy_metric.compute_proxy_rankmax - cur_proxy.compute_proxy_rankmax)
                net_gain = _net_gain(
                    comm_gain=(
                        float(cand["J_comm_gain_ms"])
                        + ((float(event_gain_ms) + 0.25 * float(packet_progress_ms)) if bool(event_aware_enabled) else 0.0)
                    ),
                    h2d_proxy_delta=max(h2d_total_delta, h2d_rankmax_delta),
                    compute_proxy_delta=compute_rankmax_delta,
                    cfg=cfg,
                )
                if float(net_gain) <= float(cfg.min_net_gain_ms):
                    rejected["net_gain"] += 1
                    continue
                if h2d_gate_enabled:
                    if not _within_slack(proxy_metric.h2d_miss_proxy_rankmax, base_proxy.h2d_miss_proxy_rankmax, h2d_slack):
                        rejected["h2d_rankmax"] += 1
                        continue
                    if (
                        not bool(cfg.require_h2d_nondegrade)
                        and not _within_slack(proxy_metric.h2d_miss_proxy_total, base_proxy.h2d_miss_proxy_total, h2d_slack)
                    ):
                        rejected["h2d_total"] += 1
                        continue
                if compute_gate_enabled:
                    if not _within_slack(proxy_metric.compute_proxy_rankmax, base_proxy.compute_proxy_rankmax, compute_slack):
                        rejected["compute_rankmax"] += 1
                        continue
                if hotset_gate_enabled:
                    if not _within_slack(proxy_metric.hotset_count_rankmax, base_proxy.hotset_count_rankmax, hotset_slack):
                        rejected["hotset_rankmax"] += 1
                        continue
                    if (
                        not bool(cfg.require_hotset_nondegrade)
                        and not _within_slack(proxy_metric.hotset_count_total, base_proxy.hotset_count_total, hotset_slack)
                    ):
                        rejected["hotset_total"] += 1
                        continue
                load_slack_eff = max(float(cfg.load_slack), float(compute_slack), 0.0)
                if proxy_metric.load_imbalance_ratio > max(
                    base_proxy.load_imbalance_ratio * (1.0 + load_slack_eff),
                    1.0 + load_slack_eff + 0.05,
                ):
                    rejected["load_imbalance"] += 1
                    continue
                cand["net_gain"] = float(net_gain)
                cand["h2d_proxy_delta_total"] = float(h2d_total_delta)
                cand["h2d_proxy_delta_rankmax"] = float(h2d_rankmax_delta)
                cand["compute_proxy_delta_rankmax"] = float(compute_rankmax_delta)
                cand["h2d_miss_proxy_total"] = float(proxy_metric.h2d_miss_proxy_total)
                cand["h2d_miss_proxy_rankmax"] = float(proxy_metric.h2d_miss_proxy_rankmax)
                cand["compute_proxy_rankmax"] = float(proxy_metric.compute_proxy_rankmax)
                cand["hotset_count_total"] = float(proxy_metric.hotset_count_total)
                cand["hotset_count_rankmax"] = float(proxy_metric.hotset_count_rankmax)
                cand["load_imbalance_ratio"] = float(proxy_metric.load_imbalance_ratio)
                cand["_proxy_metric"] = proxy_metric
                candidates.append(cand)
            if bool(event_first_enabled) and any(int(row.get("active_cross_node_event_reduction", 0)) > 0 for row in candidates):
                before = len(candidates)
                candidates = [
                    row
                    for row in candidates
                    if int(row.get("runtime_packet_event_reduction", row.get("active_cross_node_event_reduction", 0))) > 0
                ]
                rejected["event_prefiltered_out"] += max(0, before - len(candidates))
            candidates.sort(
                key=lambda row: (
                    -int(row.get("runtime_packet_event_reduction", row.get("active_cross_node_event_reduction", 0))) if bool(event_first_enabled) else 0,
                    -int(row.get("cross_node_events_eliminated", 0)) if bool(event_first_enabled) else 0,
                    -float(row.get("cross_node_event_progress_units", 0.0)) if bool(event_first_enabled) else 0.0,
                    -float(row.get("net_gain", row["prefilter_score"])),
                    -float(row["prefilter_score"]),
                    -float(row.get("event_gain_ms", 0.0)),
                    -float(row.get("packet_clear_progress_ms", 0.0)),
                    -float(row["J_comm_gain_ms"]),
                    -float(row.get("rankmax_inter_node_bytes_gain", 0.0)),
                    -float(row.get("h2d_proxy_delta_total", 0.0) <= 0.0),
                    -float(row["cross_node_bytes_reduction"]),
                    float(max(row.get("h2d_proxy_delta_total", 0.0), 0.0)),
                    float(max(row.get("compute_proxy_delta_rankmax", 0.0), 0.0)),
                    int(row["layer"]),
                    int(row["expert"]),
                    int(row["rank"]),
                )
            )
            topn = int(cfg.candidate_prefilter_topn) if int(cfg.candidate_prefilter_topn) > 0 else 32
            rejected["prefiltered_out"] += max(0, len(candidates) - topn)
            candidates = candidates[:topn]
            if not candidates:
                break
            chosen = candidates[0]
            if accepted_since_validation == 0:
                last_snapshot = (
                    json.loads(json.dumps(replicas)),
                    _copy_grouped_state(state),
                    list(selected),
                    extra_by_layer.copy(),
                    extra_by_layer_rank.copy(),
                    list(current_layers),
                )
            before_J = _state_J_comm(state)
            _apply_candidate_delta(state, chosen)
            layer = int(chosen["layer"])
            expert = int(chosen["expert"])
            rank = int(chosen["rank"])
            reps[layer][expert].append(rank)
            replicas[str(layer)][str(expert)].append(rank)
            extra_by_layer[layer] += 1
            extra_by_layer_rank[layer, rank] += 1
            after_J = _state_J_comm(state)
            current_layers[layer] = chosen["_proxy_metric"]
            row = {k: v for k, v in chosen.items() if k not in ("changes", "source_delta_by_group", "row_moves", "_proxy_metric")}
            row["J_comm_before_ms"] = float(before_J)
            row["J_comm_after_ms"] = float(after_J)
            selected.append(row)
            accepted_since_validation += 1
            if (
                rr_cache is not None
                and int(cfg.cache_validation_interval) > 0
                and accepted_since_validation >= int(cfg.cache_validation_interval)
            ):
                cur_cache = _cache_replay_metrics(
                    trace_profile=trace_profile,
                    topology=topology,
                    replicas=replicas,
                    model=model,
                    cfg=cfg,
                    layers=layers,
                    experts=experts,
                    mode="no_rs",
                )
                hit_ok = float(cur_cache["resident_hit_rate_by_item"]) >= float(rr_cache["resident_hit_rate_by_item"]) - max(float(cfg.cache_hit_drop_limit), 0.0) - 1e-12
                h2d_ok = bool(cfg.disable_h2d_gate) or float(cur_cache["rankmax_h2d_ms"]) <= float(rr_cache["rankmax_h2d_ms"]) * (1.0 + max(float(cfg.h2d_slack), 0.0)) + 1e-9
                if not hit_ok or not h2d_ok:
                    rejected["cache_validation_failed"] += 1
                    validation_failed = True
                    if last_snapshot is not None:
                        replicas, state, selected, extra_by_layer, extra_by_layer_rank, current_layers = last_snapshot
                        reps = _as_int_replicas(replicas)
                    break
                accepted_since_validation = 0
                last_snapshot = None

    placement = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=eval_topology,
        layers=layers,
        experts=experts,
        objective=str(cfg.objective),
        cfg=cfg,
    )
    train_eval = evaluate_grouped_comm_tail(
        trace_profile,
        placement,
        eval_topology,
        stage=_stage_for_trace(cfg),
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
    )
    runtime_packet_before = _runtime_packet_metrics_from_replicas(
        trace_profile=trace_profile,
        replicas=rr_replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
    )
    runtime_packet_after = _runtime_packet_metrics_from_replicas(
        trace_profile=trace_profile,
        replicas=replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
    )
    runtime_packet_cross_events_before = int(runtime_packet_before["cross_events"])
    runtime_packet_cross_events_after = int(runtime_packet_after["cross_events"])
    runtime_packet_cross_event_reduction_pct = _metric_drop_pct(
        float(runtime_packet_cross_events_before),
        float(runtime_packet_cross_events_after),
    )
    runtime_packet_cross_items_before = int(runtime_packet_before["cross_items"])
    runtime_packet_cross_items_after = int(runtime_packet_after["cross_items"])
    runtime_packet_cross_item_reduction_pct = _metric_drop_pct(
        float(runtime_packet_cross_items_before),
        float(runtime_packet_cross_items_after),
    )
    runtime_packet_cost_before = float(runtime_packet_before["cost"])
    runtime_packet_cost_after = float(runtime_packet_after["cost"])
    runtime_packet_cost_reduction_pct = _metric_drop_pct(runtime_packet_cost_before, runtime_packet_cost_after)
    final_cache = None
    cache_validation_pass = None
    if rr_cache is not None:
        final_cache = _cache_replay_metrics(
            trace_profile=trace_profile,
            topology=topology,
            replicas=replicas,
            model=model,
            cfg=cfg,
            layers=layers,
            experts=experts,
            mode="no_rs",
        )
        hit_ok = float(final_cache["resident_hit_rate_by_item"]) >= float(rr_cache["resident_hit_rate_by_item"]) - max(float(cfg.cache_hit_drop_limit), 0.0) - 1e-12
        h2d_ok = bool(cfg.disable_h2d_gate) or float(final_cache["rankmax_h2d_ms"]) <= float(rr_cache["rankmax_h2d_ms"]) * (1.0 + max(float(cfg.h2d_slack), 0.0)) + 1e-9
        cache_validation_pass = bool(hit_ok and h2d_ok)

    valid_eval = None
    valid_tail_gain_pct = None
    if valid_trace_profile is not None:
        valid_rr_eval = evaluate_grouped_comm_tail(
            valid_trace_profile,
            _base_placement_obj(replicas=rr_replicas, model=model, topology=eval_topology, layers=layers, experts=experts, objective=str(cfg.objective), cfg=cfg),
            eval_topology,
            stage=_stage_for_trace(cfg),
            dispatch_item_bytes=float(model.dispatch_item_bytes),
            combine_item_bytes=float(model.combine_item_bytes),
        )
        valid_eval = evaluate_grouped_comm_tail(
            valid_trace_profile,
            placement,
            eval_topology,
            stage=_stage_for_trace(cfg),
            dispatch_item_bytes=float(model.dispatch_item_bytes),
            combine_item_bytes=float(model.combine_item_bytes),
        )
        valid_tail_gain_pct = _metric_drop_pct(float(valid_rr_eval["J_comm_ms"]), float(valid_eval["J_comm_ms"]))

    selected_cross_node_replicas = int(sum(1 for row in selected if bool(row.get("cross_node_replica", False))))
    selected_same_node_replicas = int(len(selected) - selected_cross_node_replicas)
    selected_event_reducing_replicas = int(sum(1 for row in selected if int(row.get("active_cross_node_event_reduction", 0)) > 0))
    selected_event_neutral_replicas = int(sum(1 for row in selected if int(row.get("active_cross_node_event_reduction", 0)) == 0))
    selected_event_increasing_replicas = int(sum(1 for row in selected if int(row.get("active_cross_node_event_reduction", 0)) < 0))
    selected_cross_node_events_eliminated = int(sum(int(row.get("cross_node_events_eliminated", 0)) for row in selected))
    selected_cross_node_events_created = int(sum(int(row.get("cross_node_events_created", 0)) for row in selected))
    selected_event_gain_ms = float(sum(float(row.get("event_gain_ms", 0.0)) for row in selected))
    selected_packet_clear_progress_units = float(sum(float(row.get("cross_node_event_progress_units", 0.0)) for row in selected))
    selected_packet_clear_progress_items = int(sum(int(row.get("cross_node_event_progress_items", 0)) for row in selected))
    cross_node_bytes_reduction_pct = _metric_drop_pct(float(rr_eval["total_cross_node_bytes"]), float(train_eval["total_cross_node_bytes"]))
    rankmax_inter_node_bytes_reduction_pct = _metric_drop_pct(float(rr_eval["rankmax_inter_node_bytes"]), float(train_eval["rankmax_inter_node_bytes"]))
    tail_reduction_pct = _metric_drop_pct(float(rr_eval["J_comm_ms"]), float(train_eval["J_comm_ms"]))
    active_cross_node_events_reduction_pct = _metric_drop_pct(
        float(rr_eval["active_cross_node_events"]),
        float(train_eval["active_cross_node_events"]),
    )
    noop_reasons: List[str] = []
    if optimistic_headroom_pct < float(cfg.min_tail_headroom_pct):
        noop_reasons.append("insufficient_tail_headroom")
    if selected_cross_node_replicas <= 0:
        noop_reasons.append("no_cross_node_replicas_selected")
    if tail_reduction_pct < float(cfg.min_tail_headroom_pct):
        noop_reasons.append("grouped_tail_reduction_below_threshold")
    if cross_node_bytes_reduction_pct < float(cfg.min_inter_node_bytes_reduction_pct):
        noop_reasons.append("cross_node_bytes_reduction_below_threshold")
    if rankmax_inter_node_bytes_reduction_pct < float(cfg.min_inter_node_rankmax_bytes_reduction_pct):
        noop_reasons.append("rankmax_inter_node_bytes_reduction_below_threshold")
    packet_gate_reason = _runtime_packet_gate_reason(
        require_event_gain_when_latency=bool(cfg.require_event_gain_when_latency),
        latency_ms=float(state.get("latency_ms", 0.0) or 0.0),
        runtime_packet_cross_event_reduction_pct=float(runtime_packet_cross_event_reduction_pct),
        runtime_packet_cost_reduction_pct=float(runtime_packet_cost_reduction_pct),
    )
    if packet_gate_reason:
        noop_reasons.append(packet_gate_reason)
    if cache_validation_pass is False:
        noop_reasons.append("cache_validation_failed")
    if valid_tail_gain_pct is not None and bool(cfg.require_valid_positive) and float(valid_tail_gain_pct) <= 0.0:
        noop_reasons.append("valid_tail_gain_nonpositive")

    stackability_reasons: List[str] = []
    latency_for_event_ms = float(state.get("latency_ms", 0.0) or 0.0)
    stage_name_for_stack = str(cfg.stage).lower()
    if stage_name_for_stack not in ("prefill", "mixed"):
        stackability_reasons.append("not_prefill_or_mixed_stage")
    if selected_cross_node_replicas <= 0:
        stackability_reasons.append("no_cross_node_replicas")
    if tail_reduction_pct <= 0.0:
        stackability_reasons.append("no_grouped_tail_gain")
    if rankmax_inter_node_bytes_reduction_pct <= 0.0:
        stackability_reasons.append("no_rankmax_inter_node_bytes_gain")
    if latency_for_event_ms > 0.0 and active_cross_node_events_reduction_pct <= 0.0:
        stackability_reasons.append("no_latency_event_gain")
    if latency_for_event_ms > 0.0 and runtime_packet_cross_event_reduction_pct <= 0.0:
        stackability_reasons.append("no_runtime_packet_event_gain")
    if latency_for_event_ms > 0.0 and selected_event_reducing_replicas <= 0:
        stackability_reasons.append("no_event_reducing_replicas")
    if cache_validation_pass is False:
        stackability_reasons.append("cache_validation_failed")
    if valid_tail_gain_pct is not None and float(valid_tail_gain_pct) <= 0.0:
        stackability_reasons.append("valid_tail_gain_nonpositive")
    if noop_reasons:
        stackability_reasons.extend(f"noop:{reason}" for reason in noop_reasons)
    if not stackability_reasons and tail_reduction_pct >= 3.0 and (
        latency_for_event_ms <= 0.0 or active_cross_node_events_reduction_pct > 0.0
    ):
        stackability_status = "strong_candidate"
    elif not stackability_reasons:
        stackability_status = "weak_candidate"
    else:
        stackability_status = "no_op_candidate" if noop_reasons else "risky_candidate"

    per_source_before = np.asarray(rr_eval["per_source_total_comm_ms"], dtype=np.float64)
    per_source_after = np.asarray(train_eval["per_source_total_comm_ms"], dtype=np.float64)
    per_source_gain = per_source_before - per_source_after
    comm_gain_by_source_node: Dict[str, float] = {}
    for s in range(world):
        node = int(model.rank_to_node.get(int(s), int(s)))
        comm_gain_by_source_node[str(node)] = float(comm_gain_by_source_node.get(str(node), 0.0) + float(per_source_gain[s]))

    metrics = {
        "objective": str(cfg.objective),
        "stage": str(cfg.stage),
        "baseline_is": "RR + configured per-rank LRU expert cache budget",
        "baseline_objective_ms": float(rr_eval["J_comm_ms"]),
        "replicated_objective_ms": float(train_eval["J_comm_ms"]),
        "objective_reduction_pct": float(tail_reduction_pct),
        "baseline_grouped_J_comm_ms": float(rr_eval["J_comm_ms"]),
        "replicated_grouped_J_comm_ms": float(train_eval["J_comm_ms"]),
        "grouped_J_comm_reduction_pct": float(tail_reduction_pct),
        "baseline_rankmax_comm_ms": float(rr_eval["rankmax_comm_ms_legacy"]),
        "replicated_rankmax_comm_ms": float(train_eval["rankmax_comm_ms_legacy"]),
        "rankmax_comm_reduction_pct": _metric_drop_pct(float(rr_eval["rankmax_comm_ms_legacy"]), float(train_eval["rankmax_comm_ms_legacy"])),
        "baseline_cross_node_items": int(rr_eval["total_cross_node_items"]),
        "replicated_cross_node_items": int(train_eval["total_cross_node_items"]),
        "baseline_cross_node_tokens": int(rr_eval["total_cross_node_items"]),
        "replicated_cross_node_tokens": int(train_eval["total_cross_node_items"]),
        "cross_node_reduction_pct": _metric_drop_pct(float(rr_eval["total_cross_node_items"]), float(train_eval["total_cross_node_items"])),
        "baseline_cross_node_bytes": float(rr_eval["total_cross_node_bytes"]),
        "replicated_cross_node_bytes": float(train_eval["total_cross_node_bytes"]),
        "cross_node_bytes_reduction_pct": float(cross_node_bytes_reduction_pct),
        "baseline_active_cross_node_events": int(rr_eval["active_cross_node_events"]),
        "replicated_active_cross_node_events": int(train_eval["active_cross_node_events"]),
        "active_cross_node_events_reduction_pct": float(active_cross_node_events_reduction_pct),
        "runtime_packet_cross_events_before": int(runtime_packet_cross_events_before),
        "runtime_packet_cross_events_after": int(runtime_packet_cross_events_after),
        "runtime_packet_cross_event_reduction_pct": float(runtime_packet_cross_event_reduction_pct),
        "runtime_packet_cross_items_before": int(runtime_packet_cross_items_before),
        "runtime_packet_cross_items_after": int(runtime_packet_cross_items_after),
        "runtime_packet_cross_item_reduction_pct": float(runtime_packet_cross_item_reduction_pct),
        "runtime_packet_cost_before": float(runtime_packet_cost_before),
        "runtime_packet_cost_after": float(runtime_packet_cost_after),
        "runtime_packet_cost_reduction_pct": float(runtime_packet_cost_reduction_pct),
        "runtime_packet_metrics_before": runtime_packet_before,
        "runtime_packet_metrics_after": runtime_packet_after,
        "grouped_active_cross_node_events_reduction_pct": float(active_cross_node_events_reduction_pct),
        "baseline_inter_node_total_bytes_rankmax": float(rr_eval["rankmax_inter_node_bytes"]),
        "replicated_inter_node_total_bytes_rankmax": float(train_eval["rankmax_inter_node_bytes"]),
        "rankmax_inter_node_bytes_reduction_pct": float(rankmax_inter_node_bytes_reduction_pct),
        "num_extra_replicas": int(len(selected)),
        "selected_cross_node_replicas": int(selected_cross_node_replicas),
        "selected_same_node_replicas": int(selected_same_node_replicas),
        "selected_event_reducing_replicas": int(selected_event_reducing_replicas),
        "selected_event_neutral_replicas": int(selected_event_neutral_replicas),
        "selected_event_increasing_replicas": int(selected_event_increasing_replicas),
        "selected_cross_node_events_eliminated": int(selected_cross_node_events_eliminated),
        "selected_cross_node_events_created": int(selected_cross_node_events_created),
        "selected_event_gain_ms": float(selected_event_gain_ms),
        "selected_packet_clear_progress_units": float(selected_packet_clear_progress_units),
        "selected_packet_clear_progress_items": int(selected_packet_clear_progress_items),
        "selected_replicas_by_direction": _direction_counts(selected),
        "comm_gain_by_source_rank": per_source_gain.astype(float).tolist(),
        "comm_gain_by_source_node": comm_gain_by_source_node,
        "baseline_per_source_comm_ms": rr_eval["per_source_total_comm_ms"],
        "replicated_per_source_comm_ms": train_eval["per_source_total_comm_ms"],
        "baseline_per_source_cross_node_bytes": rr_eval["per_source_total_cross_node_bytes"],
        "replicated_per_source_cross_node_bytes": train_eval["per_source_total_cross_node_bytes"],
        "baseline_per_direction_bytes": rr_eval["per_direction_bytes"],
        "replicated_per_direction_bytes": train_eval["per_direction_bytes"],
        "baseline_per_direction_events": rr_eval["per_direction_events"],
        "replicated_per_direction_events": train_eval["per_direction_events"],
        "optimistic_upper_bound_J_comm_gain_ms": float(optimistic_gain),
        "optimistic_upper_bound_tail_headroom_pct": float(optimistic_headroom_pct),
        "min_tail_headroom_pct": float(cfg.min_tail_headroom_pct),
        "min_comm_gain_ms": float(cfg.min_comm_gain_ms),
        "cache_validation_interval": int(cfg.cache_validation_interval),
        "cache_validation_pass": cache_validation_pass,
        "cache_replay_baseline": rr_cache,
        "cache_replay_replicated": final_cache,
        "h2d_gate_enabled": bool(h2d_gate_enabled),
        "compute_gate_enabled": bool(compute_gate_enabled),
        "hotset_gate_enabled": bool(hotset_gate_enabled),
        "effective_h2d_slack": float(h2d_slack),
        "effective_compute_slack": float(compute_slack),
        "effective_hotset_slack": float(hotset_slack),
        "runtime_packet_gate": "pass" if not packet_gate_reason else str(packet_gate_reason),
        "runtime_packet_cost_positive": bool(float(runtime_packet_cost_reduction_pct) > 0.0),
        "runtime_packet_event_positive": bool(float(runtime_packet_cross_event_reduction_pct) > 0.0),
        "runtime_packet_gate_semantics": "latency guard uses replayed event+payload cost; event-only non-reduction is reported but not a no-op when payload cost drops",
        "h2d_ms_per_proxy": float(cfg.h2d_ms_per_proxy),
        "compute_ms_per_proxy": float(cfg.compute_ms_per_proxy),
        "rs_replay_available": False,
        "rs_replay_unavailable_reason": "profile has ordered router demand trace only; RS prediction/admission trace is not present",
        "valid_grouped_comm_tail": valid_eval,
        "valid_grouped_J_comm_reduction_pct": valid_tail_gain_pct,
        "extra_replicas_per_layer": extra_by_layer.astype(int).tolist(),
        "extra_replicas_per_rank_per_layer": extra_by_layer_rank.astype(int).tolist(),
        "candidate_prefilter_topn": int(cfg.candidate_prefilter_topn),
        "event_aware": bool(cfg.event_aware),
        "event_first": bool(cfg.event_first),
        "require_event_gain_when_latency": bool(cfg.require_event_gain_when_latency),
        "event_weight_ms": float(state.get("event_weight_ms", state.get("latency_ms", 0.0))),
        "lambda_event": float(cfg.lambda_event),
        "lambda_small_packet": float(cfg.lambda_small_packet),
        "small_packet_threshold": int(cfg.small_packet_threshold),
        "safety_rejected_candidates": rejected,
        "scp_stackability_status": str(stackability_status),
        "scp_stackability_reasons": stackability_reasons,
        "latency_event_model_active": bool(latency_for_event_ms > 0.0),
        "latency_event_model_ms": float(latency_for_event_ms),
        "scp_noop": bool(len(noop_reasons) > 0),
        "should_run_runtime": bool(len(noop_reasons) == 0),
        "noop_reason": ";".join(noop_reasons),
        "h2d_model_semantics": (
            "candidate-level frequency topK H2D/hotset proxy gate plus ordered-trace LRU cache replay validation"
        ),
        "runtime_assignment_policy": "nearest_topology_cost_from_replicas_tie_keep_round_robin_owner",
        "grouped_comm_tail_baseline": rr_eval,
        "grouped_comm_tail_replicated": train_eval,
        "cost_unit": str(model.cost_unit),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "hidden_bytes": float(model.hidden_bytes),
        "router_weight_bytes": float(model.router_weight_bytes),
        "expert_id_bytes": float(model.expert_id_bytes),
        "bytes_source": str(model.bytes_source),
        "latency_semantics": "sum_group_max_source_aligned_with_standard_ep_call_tail",
    }
    result = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=eval_topology,
        layers=layers,
        experts=experts,
        objective=str(cfg.objective),
        cfg=cfg,
    )
    result.update(
        {
            "selected_replicas": selected,
            "offline_metrics": metrics,
            "profile_metadata": profile_meta or {},
            "valid_profile_metadata": valid_profile_meta or {},
        }
    )
    return result


def solve_online_aware_scp_v1(
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    *,
    topology: Dict[str, Any],
    trace_profile: Dict[str, Any],
    valid_trace_profile: Optional[Dict[str, Any]] = None,
    profile_meta: Optional[Dict[str, Any]] = None,
    valid_profile_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if trace_profile is None:
        raise RuntimeError("online_aware_scp_v1 requires grouped ordered routing trace")
    if evaluate_grouped_comm_tail is None or evaluate_profile_placement_with_cache_replay is None:
        raise RuntimeError("online_aware_scp_v1 requires scripts/scp_cache_replay.py")
    layers, world, experts = A.shape
    eval_topology = dict(topology)
    if cfg.latency_ms_per_event is not None:
        bw = dict(eval_topology.get("bandwidth_model", {}) or {})
        bw["cross_node_latency_ms"] = float(cfg.latency_ms_per_event)
        eval_topology["bandwidth_model"] = bw

    replicas = round_robin_replicas(layers, experts, world)
    reps = _as_int_replicas(replicas)
    rr_replicas = round_robin_replicas(layers, experts, world)
    extra_by_layer = np.zeros((layers,), dtype=np.int64)
    extra_by_layer_rank = np.zeros((layers, world), dtype=np.int64)
    selected: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {
        "same_node_replica": 0,
        "rank_budget": 0,
        "max_replicas_per_expert": 0,
        "no_candidate_delta": 0,
        "low_prefilter_gain": 0,
        "online_proxy_no_gain": 0,
        "decode_hit_gate": 0,
        "decode_h2d_gate": 0,
        "prefiltered_out": 0,
        "duplicate_candidate": 0,
    }

    budget = int(cfg.global_extra_replicas) if int(cfg.global_extra_replicas) > 0 else int(cfg.extra_replicas_per_layer) * int(layers)
    budget = max(int(budget), 0)
    baseline_prefill = evaluate_grouped_comm_tail(
        trace_profile,
        _base_placement_obj(replicas=rr_replicas, model=model, topology=eval_topology, layers=layers, experts=experts, objective=str(cfg.objective), cfg=cfg),
        eval_topology,
        stage="prefill",
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
    )
    current_fast_proxy_ms = float(baseline_prefill["J_comm_ms"])
    baseline_layers: List[LayerMetrics] = []
    current_layers: List[LayerMetrics] = []
    prefill_counts = np.asarray(trace_profile["A_prefill"], dtype=np.float64)
    decode_counts = np.asarray(trace_profile["A_decode"], dtype=np.float64)
    for layer in range(layers):
        _, base_metric = _layer_metric_from_reps(decode_counts, model, reps[layer], layer, cfg)
        baseline_layers.append(base_metric)
        current_layers.append(base_metric)
    state = _make_grouped_state(
        trace_profile=trace_profile,
        topology=eval_topology,
        replicas=replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
    )

    for _step in range(budget):
        bottleneck_info = _state_bottleneck_info(state, top_fraction=0.35)
        top_groups = bottleneck_info.get("top_groups", set())
        row_group = np.asarray(state["row_group"], dtype=np.int64)
        row_source = np.asarray(state["row_source"], dtype=np.int64)
        candidate_pairs: set[Tuple[int, int, int]] = set()
        for (layer, expert), rows in state["rows_by_le"].items():
            owner = int(expert) % int(world)
            owner_node = int(model.rank_to_node.get(owner, owner))
            for row_id in rows:
                if int(row_group[int(row_id)]) not in top_groups:
                    continue
                source = int(row_source[int(row_id)])
                source_node = int(model.rank_to_node.get(source, source))
                if source_node == owner_node:
                    continue
                for rank in range(world):
                    if int(model.rank_to_node.get(rank, rank)) != source_node:
                        continue
                    candidate_pairs.add((int(layer), int(expert), int(rank)))

        candidates: List[Dict[str, Any]] = []
        for layer, expert, rank in sorted(candidate_pairs):
            if len(reps[layer][expert]) >= int(cfg.max_replicas_per_expert):
                rejected["max_replicas_per_expert"] += 1
                continue
            if rank in reps[layer][expert]:
                rejected["duplicate_candidate"] += 1
                continue
            owner = int(expert) % int(world)
            owner_node = int(model.rank_to_node.get(owner, owner))
            replica_node = int(model.rank_to_node.get(rank, rank))
            if not bool(cfg.allow_same_node_replicas) and owner_node == replica_node:
                rejected["same_node_replica"] += 1
                continue
            if int(cfg.global_extra_replicas) <= 0 and extra_by_layer[layer] >= int(cfg.extra_replicas_per_layer):
                rejected["rank_budget"] += 1
                continue
            if int(cfg.extra_replicas_per_rank_per_layer) > 0 and extra_by_layer_rank[layer, rank] >= int(cfg.extra_replicas_per_rank_per_layer):
                rejected["rank_budget"] += 1
                continue
            cand = _candidate_delta_grouped(
                state=state,
                model=model,
                reps=reps,
                layer=layer,
                expert=expert,
                rank=rank,
                bottleneck_info=bottleneck_info,
            )
            if cand is None:
                rejected["no_candidate_delta"] += 1
                continue
            if float(cand.get("J_comm_gain_ms", 0.0)) <= float(cfg.min_comm_gain_ms):
                rejected["low_prefilter_gain"] += 1
                continue
            cand["prefilter_gain_ms"] = float(
                float(cand.get("J_comm_gain_ms", 0.0))
                + float(cand.get("event_gain_ms", 0.0))
                + 0.25 * float(cand.get("packet_clear_progress_ms", 0.0))
            )
            candidates.append(cand)

        candidates.sort(
            key=lambda row: (
                -float(row.get("prefilter_gain_ms", 0.0)),
                -float(row.get("rankmax_inter_node_bytes_gain", 0.0)),
                -float(row.get("cross_node_bytes_reduction", 0.0)),
                int(row["layer"]),
                int(row["expert"]),
                int(row["rank"]),
            )
        )
        topn = int(cfg.candidate_prefilter_topn) if int(cfg.candidate_prefilter_topn) > 0 else 64
        rejected["prefiltered_out"] += max(0, len(candidates) - topn)
        candidates = candidates[:topn]
        if not candidates:
            break

        best: Optional[
            Tuple[
                Tuple[float, ...],
                Dict[str, Any],
                Dict[str, Dict[str, List[int]]],
                float,
                LayerMetrics,
                Dict[str, Any],
            ]
        ] = None
        for cand in candidates:
            layer = int(cand["layer"])
            expert = int(cand["expert"])
            rank = int(cand["rank"])
            trial_replicas = json.loads(json.dumps(replicas))
            trial_replicas[str(layer)][str(expert)].append(int(rank))
            trial_reps_layer = {int(e): list(rs) for e, rs in reps[layer].items()}
            trial_reps_layer[int(expert)] = list(trial_reps_layer[int(expert)]) + [int(rank)]
            _, decode_metric = _layer_metric_from_reps(decode_counts, model, trial_reps_layer, layer, cfg)
            cur_decode_metric = current_layers[layer]
            base_decode_metric = baseline_layers[layer]
            decode_proxy_gain = float(
                (cur_decode_metric.comm_cost_rankmax - decode_metric.comm_cost_rankmax)
                + float(cfg.h2d_ms_per_proxy)
                * (cur_decode_metric.h2d_miss_proxy_rankmax - decode_metric.h2d_miss_proxy_rankmax)
                + float(cfg.compute_ms_per_proxy)
                * (cur_decode_metric.compute_proxy_rankmax - decode_metric.compute_proxy_rankmax)
            )
            online_gain = float(cand.get("J_comm_gain_ms", 0.0)) + float(decode_proxy_gain)
            decode_hotset_delta_vs_rr = float(
                decode_metric.hotset_count_total - base_decode_metric.hotset_count_total
            )
            decode_h2d_delta_vs_rr = float(
                decode_metric.h2d_miss_proxy_rankmax - base_decode_metric.h2d_miss_proxy_rankmax
            )
            if online_gain <= float(cfg.min_net_gain_ms):
                rejected["online_proxy_no_gain"] += 1
                continue
            if bool(cfg.require_hotset_nondegrade) and float(decode_metric.hotset_count_total) > float(base_decode_metric.hotset_count_total) * (1.0 + max(float(cfg.hotset_slack), 0.0)) + 1e-9:
                rejected["decode_hit_gate"] += 1
                continue
            if not bool(cfg.disable_h2d_gate):
                allowed_h2d = float(base_decode_metric.h2d_miss_proxy_rankmax) * (1.0 + max(float(cfg.h2d_slack), 0.0)) + 1e-9
                if float(decode_metric.h2d_miss_proxy_rankmax) > allowed_h2d:
                    rejected["decode_h2d_gate"] += 1
                    continue
            score = (
                float(online_gain),
                float(cand.get("J_comm_gain_ms", 0.0)),
                float(decode_proxy_gain),
                -max(float(decode_h2d_delta_vs_rr), 0.0),
                -max(float(decode_hotset_delta_vs_rr), 0.0),
                float(cand.get("rankmax_inter_node_bytes_gain", 0.0)),
                float(cand.get("cross_node_bytes_reduction", 0.0)),
                -int(layer),
                -int(expert),
                -int(rank),
            )
            row = {k: v for k, v in cand.items() if k not in ("changes", "source_delta_by_group", "row_moves")}
            row.update(
                {
                    "online_proxy_gain_ms": float(online_gain),
                    "online_proxy_before_ms": float(current_fast_proxy_ms),
                    "online_proxy_after_ms": float(current_fast_proxy_ms - online_gain),
                    "prefill_tail_before_ms": float(current_fast_proxy_ms),
                    "prefill_tail_after_ms": float(current_fast_proxy_ms - float(cand.get("J_comm_gain_ms", 0.0))),
                    "decode_proxy_gain_ms": float(decode_proxy_gain),
                    "decode_hit_item_after": float(decode_metric.hotset_count_total),
                    "decode_hit_item_delta_vs_rr": float(decode_hotset_delta_vs_rr),
                    "decode_h2d_mb_after": float(decode_metric.h2d_miss_proxy_rankmax),
                    "decode_h2d_mb_delta_vs_rr": float(decode_h2d_delta_vs_rr),
                    "decode_rankmax_h2d_ms_after": float(decode_metric.h2d_miss_proxy_rankmax),
                    "decode_rankmax_comm_ms_after": float(decode_metric.comm_cost_rankmax),
                    "decode_rankmax_compute_items_after": float(decode_metric.compute_proxy_rankmax),
                }
            )
            if best is None or score > best[0]:
                best = (score, row, trial_replicas, float(online_gain), decode_metric, cand)

        if best is None:
            break
        _score, row, replicas, fast_gain, decode_metric, apply_cand = best
        reps = _as_int_replicas(replicas)
        layer = int(row["layer"])
        expert = int(row["expert"])
        rank = int(row["rank"])
        extra_by_layer[layer] += 1
        extra_by_layer_rank[layer, rank] += 1
        _apply_candidate_delta(state, apply_cand)
        current_layers[layer] = decode_metric
        current_fast_proxy_ms -= float(fast_gain)
        selected.append(row)

    baseline_proxy = _online_proxy_metrics(
        trace_profile=trace_profile,
        topology=eval_topology,
        replicas=rr_replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
    )
    train_proxy = _online_proxy_metrics(
        trace_profile=trace_profile,
        topology=eval_topology,
        replicas=replicas,
        model=model,
        cfg=cfg,
        layers=layers,
        experts=experts,
    )
    valid_proxy = None
    valid_baseline_proxy = None
    valid_gain = None
    if valid_trace_profile is not None:
        valid_baseline_proxy = _online_proxy_metrics(
            trace_profile=valid_trace_profile,
            topology=eval_topology,
            replicas=rr_replicas,
            model=model,
            cfg=cfg,
            layers=layers,
            experts=experts,
        )
        valid_proxy = _online_proxy_metrics(
            trace_profile=valid_trace_profile,
            topology=eval_topology,
            replicas=replicas,
            model=model,
            cfg=cfg,
            layers=layers,
            experts=experts,
        )
        valid_gain = float(valid_baseline_proxy["online_proxy_ms"]) - float(valid_proxy["online_proxy_ms"])

    train_eval = evaluate_grouped_comm_tail(
        trace_profile,
        _base_placement_obj(replicas=replicas, model=model, topology=eval_topology, layers=layers, experts=experts, objective=str(cfg.objective), cfg=cfg),
        eval_topology,
        stage="prefill",
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
    )
    rr_eval = evaluate_grouped_comm_tail(
        trace_profile,
        _base_placement_obj(replicas=rr_replicas, model=model, topology=eval_topology, layers=layers, experts=experts, objective=str(cfg.objective), cfg=cfg),
        eval_topology,
        stage="prefill",
        dispatch_item_bytes=float(model.dispatch_item_bytes),
        combine_item_bytes=float(model.combine_item_bytes),
    )
    selected_cross_node_replicas = int(sum(1 for row in selected if bool(row.get("cross_node_replica", False))))
    selected_same_node_replicas = int(len(selected) - selected_cross_node_replicas)
    online_reduction_pct = _metric_drop_pct(float(baseline_proxy["online_proxy_ms"]), float(train_proxy["online_proxy_ms"]))
    prefill_reduction_pct = _metric_drop_pct(float(baseline_proxy["prefill_tail_ms"]), float(train_proxy["prefill_tail_ms"]))
    decode_reduction_pct = _metric_drop_pct(float(baseline_proxy["decode_objective_ms"]), float(train_proxy["decode_objective_ms"]))
    cross_node_bytes_reduction_pct = _metric_drop_pct(float(rr_eval["total_cross_node_bytes"]), float(train_eval["total_cross_node_bytes"]))
    rankmax_inter_node_bytes_reduction_pct = _metric_drop_pct(float(rr_eval["rankmax_inter_node_bytes"]), float(train_eval["rankmax_inter_node_bytes"]))

    noop_reasons: List[str] = []
    if selected_cross_node_replicas <= 0:
        noop_reasons.append("no_cross_node_replicas_selected")
    if online_reduction_pct < float(cfg.min_tail_headroom_pct):
        noop_reasons.append("online_proxy_reduction_below_threshold")
    if cross_node_bytes_reduction_pct < float(cfg.min_inter_node_bytes_reduction_pct):
        noop_reasons.append("cross_node_bytes_reduction_below_threshold")
    if rankmax_inter_node_bytes_reduction_pct < float(cfg.min_inter_node_rankmax_bytes_reduction_pct):
        noop_reasons.append("rankmax_inter_node_bytes_reduction_below_threshold")
    if valid_gain is not None and bool(cfg.require_valid_positive) and float(valid_gain) <= 0.0:
        noop_reasons.append("valid_online_proxy_gain_nonpositive")

    metrics = {
        "objective": str(cfg.objective),
        "stage": str(cfg.stage),
        "baseline_is": "RR + prefill comm tail + decode warm-cache LRU replay",
        "baseline_objective_ms": float(baseline_proxy["online_proxy_ms"]),
        "replicated_objective_ms": float(train_proxy["online_proxy_ms"]),
        "objective_reduction_pct": float(online_reduction_pct),
        "baseline_online_proxy_ms": float(baseline_proxy["online_proxy_ms"]),
        "replicated_online_proxy_ms": float(train_proxy["online_proxy_ms"]),
        "online_proxy_reduction_pct": float(online_reduction_pct),
        "baseline_prefill_tail_ms": float(baseline_proxy["prefill_tail_ms"]),
        "replicated_prefill_tail_ms": float(train_proxy["prefill_tail_ms"]),
        "prefill_tail_reduction_pct": float(prefill_reduction_pct),
        "baseline_decode_warm_objective_ms": float(baseline_proxy["decode_objective_ms"]),
        "replicated_decode_warm_objective_ms": float(train_proxy["decode_objective_ms"]),
        "decode_warm_objective_reduction_pct": float(decode_reduction_pct),
        "baseline_resident_hit_rate_by_item": float(baseline_proxy["decode_hit_item"]),
        "replicated_resident_hit_rate_by_item": float(train_proxy["decode_hit_item"]),
        "resident_hit_rate_item_delta": float(train_proxy["decode_hit_item"]) - float(baseline_proxy["decode_hit_item"]),
        "baseline_rankmax_h2d_ms": float(baseline_proxy["decode_rankmax_h2d_ms"]),
        "replicated_rankmax_h2d_ms": float(train_proxy["decode_rankmax_h2d_ms"]),
        "baseline_decode_h2d_mb": float(baseline_proxy["decode_h2d_mb"]),
        "replicated_decode_h2d_mb": float(train_proxy["decode_h2d_mb"]),
        "decode_h2d_mb_delta": float(train_proxy["decode_h2d_mb"]) - float(baseline_proxy["decode_h2d_mb"]),
        "baseline_decode_rankmax_comm_ms": float(baseline_proxy["decode_rankmax_comm_ms"]),
        "replicated_decode_rankmax_comm_ms": float(train_proxy["decode_rankmax_comm_ms"]),
        "baseline_decode_rankmax_compute_items": float(baseline_proxy["decode_rankmax_compute_items"]),
        "replicated_decode_rankmax_compute_items": float(train_proxy["decode_rankmax_compute_items"]),
        "baseline_cross_node_bytes": float(rr_eval["total_cross_node_bytes"]),
        "replicated_cross_node_bytes": float(train_eval["total_cross_node_bytes"]),
        "cross_node_bytes_reduction_pct": float(cross_node_bytes_reduction_pct),
        "baseline_cross_node_items": int(rr_eval["total_cross_node_items"]),
        "replicated_cross_node_items": int(train_eval["total_cross_node_items"]),
        "baseline_cross_node_tokens": int(rr_eval["total_cross_node_items"]),
        "replicated_cross_node_tokens": int(train_eval["total_cross_node_items"]),
        "cross_node_reduction_pct": _metric_drop_pct(float(rr_eval["total_cross_node_items"]), float(train_eval["total_cross_node_items"])),
        "baseline_inter_node_total_bytes_rankmax": float(rr_eval["rankmax_inter_node_bytes"]),
        "replicated_inter_node_total_bytes_rankmax": float(train_eval["rankmax_inter_node_bytes"]),
        "rankmax_inter_node_bytes_reduction_pct": float(rankmax_inter_node_bytes_reduction_pct),
        "num_extra_replicas": int(len(selected)),
        "selected_cross_node_replicas": int(selected_cross_node_replicas),
        "selected_same_node_replicas": int(selected_same_node_replicas),
        "selected_replicas_by_direction": _direction_counts(selected),
        "extra_replicas_per_layer": extra_by_layer.astype(int).tolist(),
        "extra_replicas_per_rank_per_layer": extra_by_layer_rank.astype(int).tolist(),
        "candidate_prefilter_topn": int(cfg.candidate_prefilter_topn),
        "h2d_gate_enabled": bool(not cfg.disable_h2d_gate),
        "effective_h2d_slack": float(max(float(cfg.h2d_slack), 0.0)),
        "cache_hit_drop_limit": float(cfg.cache_hit_drop_limit),
        "safety_rejected_candidates": rejected,
        "online_proxy_baseline": baseline_proxy,
        "online_proxy_replicated": train_proxy,
        "online_proxy_valid_baseline": valid_baseline_proxy,
        "online_proxy_valid": valid_proxy,
        "valid_online_proxy_gain_ms": valid_gain,
        "valid_online_proxy_reduction_pct": (
            None
            if valid_baseline_proxy is None or valid_proxy is None
            else _metric_drop_pct(float(valid_baseline_proxy["online_proxy_ms"]), float(valid_proxy["online_proxy_ms"]))
        ),
        "h2d_model_semantics": "ordered_trace_lru_cache_replay",
        "runtime_assignment_policy": "nearest_topology_cost_from_replicas_tie_keep_round_robin_owner",
        "scp_noop": bool(len(noop_reasons) > 0),
        "should_run_runtime": bool(len(noop_reasons) == 0),
        "noop_reason": ";".join(noop_reasons),
        "cost_unit": str(model.cost_unit),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "hidden_bytes": float(model.hidden_bytes),
        "router_weight_bytes": float(model.router_weight_bytes),
        "expert_id_bytes": float(model.expert_id_bytes),
        "bytes_source": str(model.bytes_source),
        "latency_semantics": "prefill_grouped_tail_plus_decode_warm_cache_replay",
    }
    result = _base_placement_obj(
        replicas=replicas,
        model=model,
        topology=eval_topology,
        layers=layers,
        experts=experts,
        objective=str(cfg.objective),
        cfg=cfg,
    )
    result.update(
        {
            "selected_replicas": selected,
            "offline_metrics": metrics,
            "profile_metadata": profile_meta or {},
            "valid_profile_metadata": valid_profile_meta or {},
        }
    )
    return result


def solve(
    A: np.ndarray,
    model: CommModel,
    cfg: ReplicatedSCPConfig,
    *,
    A_prefill_choice: Optional[np.ndarray] = None,
    A_decode_choice: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    layers, world, experts = A.shape
    A_prefill_choice = A if A_prefill_choice is None else A_prefill_choice
    A_decode_choice = A if A_decode_choice is None else A_decode_choice
    replicas = round_robin_replicas(layers, experts, world)
    reps = _as_int_replicas(replicas)
    extra_by_layer = np.zeros((layers,), dtype=np.int64)
    extra_by_layer_rank = np.zeros((layers, world), dtype=np.int64)
    selected: List[Dict[str, Any]] = []

    rr_reps = _as_int_replicas(round_robin_replicas(layers, experts, world))
    baseline_layers: List[LayerMetrics] = []
    current_layers: List[LayerMetrics] = []
    for layer in range(layers):
        _, base_metric = _layer_metric_from_reps(A, model, rr_reps[layer], layer, cfg)
        _, cur_metric = _layer_metric_from_reps(A, model, reps[layer], layer, cfg)
        baseline_layers.append(base_metric)
        current_layers.append(cur_metric)

    rejected = {
        "no_comm_gain": 0,
        "comm_rankmax": 0,
        "net_gain": 0,
        "h2d_rankmax": 0,
        "h2d_total": 0,
        "compute_rankmax": 0,
        "hotset_rankmax": 0,
        "hotset_total": 0,
        "load_imbalance": 0,
        "same_node_replica": 0,
        "inter_node_rankmax_bytes": 0,
    }
    # Safety gates use direct slack semantics: --h2d-slack 0.03 means "allow
    # 3% growth versus RR"; --disable-h2d-gate is the only way to disable the
    # H2D gate.  Deprecated require_* flags are kept in reports only and no
    # longer silently clamp a user-provided positive slack to zero.
    h2d_gate_enabled = (not bool(cfg.disable_h2d_gate)) and (bool(cfg.require_h2d_nondegrade) or float(cfg.h2d_slack) >= 0.0)
    compute_gate_enabled = bool(cfg.require_compute_nondegrade) or float(cfg.compute_slack) >= 0.0
    hotset_gate_enabled = bool(cfg.require_hotset_nondegrade) or float(cfg.hotset_slack) > 0.0
    h2d_slack = 0.0 if cfg.require_h2d_nondegrade else max(float(cfg.h2d_slack), 0.0)
    compute_slack = 0.0 if cfg.require_compute_nondegrade else max(float(cfg.compute_slack), 0.0)
    hotset_slack = 0.0 if cfg.require_hotset_nondegrade else max(float(cfg.hotset_slack), 0.0)

    def _candidate_best_for_layer(layer: int) -> Optional[Tuple[Tuple[float, ...], Dict[str, Any], LayerMetrics]]:
        best: Optional[Tuple[Tuple[float, ...], Dict[str, Any], LayerMetrics]] = None
        base = baseline_layers[layer]
        cur = current_layers[layer]
        cur_obj = _layer_objective(cur, cfg)
        for expert in range(experts):
            if len(reps[layer][expert]) >= cfg.max_replicas_per_expert:
                continue
            for rank in range(world):
                if rank in reps[layer][expert]:
                    continue
                rr_owner = int(expert) % int(world)
                rr_owner_node = int(model.rank_to_node.get(rr_owner, rr_owner))
                replica_node = int(model.rank_to_node.get(int(rank), int(rank)))
                cross_node_replica = bool(rr_owner_node != replica_node)
                if not cfg.allow_same_node_replicas and not cross_node_replica:
                    rejected["same_node_replica"] += 1
                    continue
                if cfg.extra_replicas_per_rank_per_layer > 0 and extra_by_layer_rank[layer, rank] >= cfg.extra_replicas_per_rank_per_layer:
                    continue
                approx_benefit = replica_benefit(A, model, reps, layer, expert, rank)
                if approx_benefit < cfg.min_benefit:
                    rejected["no_comm_gain"] += 1
                    continue
                trial_reps = {int(e): list(rs) for e, rs in reps[layer].items()}
                trial_reps[int(expert)] = list(trial_reps[int(expert)]) + [int(rank)]
                _, metric = _layer_metric_from_reps(A, model, trial_reps, layer, cfg)
                exact_gain = float(cur.comm_cost - metric.comm_cost)
                rankmax_gain = float(cur.comm_cost_rankmax - metric.comm_cost_rankmax)
                inter_node_dispatch_bytes_gain = float(cur.inter_node_dispatch_bytes - metric.inter_node_dispatch_bytes)
                inter_node_combine_bytes_gain = float(cur.inter_node_combine_bytes - metric.inter_node_combine_bytes)
                inter_node_total_bytes_gain = float(
                    (cur.inter_node_dispatch_bytes + cur.inter_node_combine_bytes)
                    - (metric.inter_node_dispatch_bytes + metric.inter_node_combine_bytes)
                )
                inter_node_rankmax_bytes_gain = float(cur.inter_node_bytes_rankmax - metric.inter_node_bytes_rankmax)
                obj_gain = float(cur_obj - _layer_objective(metric, cfg))
                runtime_rankmax_before = _rank_runtime_proxy_rankmax(cur, cfg)
                runtime_rankmax_after = _rank_runtime_proxy_rankmax(metric, cfg)
                runtime_rankmax_gain = float(runtime_rankmax_before - runtime_rankmax_after)
                event_delta = int(metric.active_cross_node_pairs - cur.active_cross_node_pairs)
                small_delta = int(metric.small_cross_node_pairs - cur.small_cross_node_pairs)
                if exact_gain < cfg.min_benefit and obj_gain <= 0.0:
                    rejected["no_comm_gain"] += 1
                    continue
                if _is_throughput_objective(cfg) and runtime_rankmax_gain <= float(cfg.min_rankmax_benefit):
                    rejected["comm_rankmax"] += 1
                    continue
                if cfg.require_rankmax_gain and rankmax_gain <= float(cfg.min_rankmax_benefit):
                    rejected["comm_rankmax"] += 1
                    continue
                if cfg.require_inter_node_rankmax_bytes_gain and inter_node_rankmax_bytes_gain <= 1e-9:
                    rejected["inter_node_rankmax_bytes"] += 1
                    continue
                h2d_total_delta = float(metric.h2d_miss_proxy_total - cur.h2d_miss_proxy_total)
                h2d_rankmax_delta = float(metric.h2d_miss_proxy_rankmax - cur.h2d_miss_proxy_rankmax)
                compute_rankmax_delta = float(metric.compute_proxy_rankmax - cur.compute_proxy_rankmax)
                decode_rs_gain = 0.0
                if _is_throughput_objective(cfg):
                    # The throughput objective already includes comm, latency,
                    # H2D proxy, and compute proxy in one rankmax scalar.
                    trial_net_gain = float(runtime_rankmax_gain)
                else:
                    trial_net_gain = _net_gain(
                        comm_gain=(
                            obj_gain
                            if str(cfg.objective).lower() in ("prefill_ranktail_v1", "ranktail", "rank_tail")
                            else (rankmax_gain if cfg.require_rankmax_gain else exact_gain)
                        ),
                        h2d_proxy_delta=max(h2d_total_delta, h2d_rankmax_delta),
                        compute_proxy_delta=compute_rankmax_delta,
                        cfg=cfg,
                    )
                composite_net_gain = float(trial_net_gain)
                if composite_net_gain <= float(cfg.min_net_gain_ms):
                    rejected["net_gain"] += 1
                    continue
                if h2d_gate_enabled:
                    if not _within_slack(metric.h2d_miss_proxy_rankmax, base.h2d_miss_proxy_rankmax, h2d_slack):
                        rejected["h2d_rankmax"] += 1
                        continue
                    if (
                        not bool(cfg.require_h2d_nondegrade)
                        and not _within_slack(metric.h2d_miss_proxy_total, base.h2d_miss_proxy_total, h2d_slack)
                    ):
                        rejected["h2d_total"] += 1
                        continue
                if compute_gate_enabled:
                    if not _within_slack(metric.compute_proxy_rankmax, base.compute_proxy_rankmax, compute_slack):
                        rejected["compute_rankmax"] += 1
                        continue
                if hotset_gate_enabled:
                    if not _within_slack(metric.hotset_count_rankmax, base.hotset_count_rankmax, hotset_slack):
                        rejected["hotset_rankmax"] += 1
                        continue
                    if (
                        not bool(cfg.require_hotset_nondegrade)
                        and not _within_slack(metric.hotset_count_total, base.hotset_count_total, hotset_slack)
                    ):
                        rejected["hotset_total"] += 1
                        continue
                # Keep a broad load-balance guard even when the compute proxy gate is relaxed.
                load_slack_eff = max(float(cfg.load_slack), float(compute_slack), 0.0)
                if metric.load_imbalance_ratio > max(base.load_imbalance_ratio * (1.0 + load_slack_eff), 1.0 + load_slack_eff + 0.05):
                    rejected["load_imbalance"] += 1
                    continue
                rank_pressure = float(extra_by_layer_rank[layer, rank])
                # Prefer objective gain, then actual rank-tail/event gains, then lower cache/compute pressure.
                score = (
                    float(obj_gain),
                    float(runtime_rankmax_gain),
                    float(inter_node_rankmax_bytes_gain),
                    float(inter_node_total_bytes_gain),
                    float(rankmax_gain),
                    -float(max(event_delta, 0)),
                    -float(max(small_delta, 0)),
                    float(trial_net_gain),
                    float(exact_gain),
                    -max(h2d_total_delta, 0.0),
                    -max(compute_rankmax_delta, 0.0),
                    -rank_pressure,
                    -int(rank),
                    -int(expert),
                )
                row = {
                    "layer": int(layer),
                    "expert": int(expert),
                    "rank": int(rank),
                    "rr_owner": int(rr_owner),
                    "rr_owner_node": int(rr_owner_node),
                    "replica_node": int(replica_node),
                    "cross_node_replica": bool(cross_node_replica),
                    "benefit": float(exact_gain),
                    "rankmax_benefit": float(rankmax_gain),
                    "inter_node_dispatch_bytes_gain": float(inter_node_dispatch_bytes_gain),
                    "inter_node_combine_bytes_gain": float(inter_node_combine_bytes_gain),
                    "inter_node_total_bytes_gain": float(inter_node_total_bytes_gain),
                    "inter_node_rankmax_bytes_gain": float(inter_node_rankmax_bytes_gain),
                    "inter_node_rankmax_bytes_before": float(cur.inter_node_bytes_rankmax),
                    "inter_node_rankmax_bytes_after": float(metric.inter_node_bytes_rankmax),
                    "objective_gain": float(obj_gain),
                    "runtime_rankmax_gain": float(runtime_rankmax_gain),
                    "runtime_rankmax_before": float(runtime_rankmax_before),
                    "runtime_rankmax_after": float(runtime_rankmax_after),
                    "event_delta": int(event_delta),
                    "small_packet_delta": int(small_delta),
                    "net_gain": float(trial_net_gain),
                    "composite_net_gain": float(composite_net_gain),
                    "decode_rs_gain": float(decode_rs_gain),
                    "raw_comm_benefit": float(approx_benefit),
                    "comm_cost_rankmax": float(metric.comm_cost_rankmax),
                    "h2d_proxy_delta_total": float(h2d_total_delta),
                    "h2d_proxy_delta_rankmax": float(h2d_rankmax_delta),
                    "compute_proxy_delta_rankmax": float(compute_rankmax_delta),
                    "hotset_count_total": float(metric.hotset_count_total),
                    "hotset_count_rankmax": float(metric.hotset_count_rankmax),
                    "h2d_miss_proxy_total": float(metric.h2d_miss_proxy_total),
                    "h2d_miss_proxy_rankmax": float(metric.h2d_miss_proxy_rankmax),
                    "compute_proxy_rankmax": float(metric.compute_proxy_rankmax),
                    "load_imbalance_ratio": float(metric.load_imbalance_ratio),
                }
                if best is None or score > best[0]:
                    best = (score, row, metric)
        return best

    global_budget = int(cfg.global_extra_replicas)
    if global_budget > 0:
        for _ in range(global_budget):
            best_global: Optional[Tuple[Tuple[float, ...], Dict[str, Any], LayerMetrics]] = None
            for layer in range(layers):
                cand = _candidate_best_for_layer(layer)
                if cand is not None and (best_global is None or cand[0] > best_global[0]):
                    best_global = cand
            if best_global is None:
                break
            _score, row, metric = best_global
            layer = int(row["layer"])
            expert = int(row["expert"])
            rank = int(row["rank"])
            reps[layer][expert].append(rank)
            replicas[str(layer)][str(expert)].append(rank)
            extra_by_layer[layer] += 1
            extra_by_layer_rank[layer, rank] += 1
            current_layers[layer] = metric
            selected.append(row)
    else:
        for layer in range(layers):
            while extra_by_layer[layer] < cfg.extra_replicas_per_layer:
                best = _candidate_best_for_layer(layer)
                if best is None:
                    break
                _score, row, metric = best
                expert = int(row["expert"])
                rank = int(row["rank"])
                reps[layer][expert].append(rank)
                replicas[str(layer)][str(expert)].append(rank)
                extra_by_layer[layer] += 1
                extra_by_layer_rank[layer, rank] += 1
                current_layers[layer] = metric
                selected.append(row)

    # Diagnostic assignments for offline accounting only.  These are derived
    # from the final replica sets using the same nearest-replica rule that
    # runtime precomputes at layer initialization.  They are not serialized as
    # placement decisions.
    runtime_assignment, _runtime_rank_expert_load = build_choice(A, model, replicas, cfg)
    prefill_runtime_assignment, _prefill_rank_expert_load = build_choice(A_prefill_choice, model, replicas, cfg)
    decode_runtime_assignment, _decode_rank_expert_load = build_choice(A_decode_choice, model, replicas, cfg)

    baseline = evaluate_choice(A, model, round_robin_choice(layers, world, experts), cfg)
    current = evaluate_choice(A, model, runtime_assignment, cfg)
    prefill_current = evaluate_choice(A_prefill_choice, model, prefill_runtime_assignment, cfg)
    prefill_baseline = evaluate_choice(A_prefill_choice, model, round_robin_choice(layers, world, experts), cfg)
    decode_current = evaluate_choice(A_decode_choice, model, decode_runtime_assignment, cfg)
    decode_baseline = evaluate_choice(A_decode_choice, model, round_robin_choice(layers, world, experts), cfg)

    def pct_drop(before: float, after: float) -> float:
        return float(100.0 * (float(before) - float(after)) / max(float(before), 1e-9))

    baseline_inter_node_total_bytes = float(baseline["inter_node_dispatch_bytes"] + baseline["inter_node_combine_bytes"])
    current_inter_node_total_bytes = float(current["inter_node_dispatch_bytes"] + current["inter_node_combine_bytes"])
    selected_cross_node_replicas = int(sum(1 for row in selected if bool(row.get("cross_node_replica", False))))
    selected_same_node_replicas = int(len(selected) - selected_cross_node_replicas)
    cross_node_bytes_reduction_pct = pct_drop(baseline_inter_node_total_bytes, current_inter_node_total_bytes)
    rankmax_inter_node_bytes_reduction_pct = pct_drop(
        baseline["inter_node_total_bytes_rankmax"], current["inter_node_total_bytes_rankmax"]
    )
    noop_reasons: List[str] = []
    if selected_cross_node_replicas <= 0:
        noop_reasons.append("no_cross_node_replicas_selected")
    if cross_node_bytes_reduction_pct < float(cfg.min_inter_node_bytes_reduction_pct):
        noop_reasons.append("cross_node_bytes_reduction_below_threshold")
    if rankmax_inter_node_bytes_reduction_pct < float(cfg.min_inter_node_rankmax_bytes_reduction_pct):
        noop_reasons.append("rankmax_inter_node_bytes_reduction_below_threshold")

    metrics = {
        "baseline_comm_cost": baseline["comm_cost"],
        "replicated_comm_cost": current["comm_cost"],
        "comm_reduction_pct": pct_drop(baseline["comm_cost"], current["comm_cost"]),
        "baseline_dispatch_bw_cost": baseline["dispatch_bw_cost"],
        "replicated_dispatch_bw_cost": current["dispatch_bw_cost"],
        "baseline_combine_bw_cost": baseline["combine_bw_cost"],
        "replicated_combine_bw_cost": current["combine_bw_cost"],
        "baseline_latency_event_cost_ms": baseline["latency_event_cost_ms"],
        "replicated_latency_event_cost_ms": current["latency_event_cost_ms"],
        "latency_event_reduction_pct": pct_drop(baseline["latency_event_cost_ms"], current["latency_event_cost_ms"]),
        "baseline_estimated_comm_with_latency_event_ms": baseline["estimated_comm_with_latency_event_ms"],
        "replicated_estimated_comm_with_latency_event_ms": current["estimated_comm_with_latency_event_ms"],
        "estimated_comm_with_latency_event_reduction_pct": pct_drop(
            baseline["estimated_comm_with_latency_event_ms"], current["estimated_comm_with_latency_event_ms"]
        ),
        "baseline_rank_runtime_proxy_rankmax": baseline["rank_runtime_proxy_rankmax"],
        "replicated_rank_runtime_proxy_rankmax": current["rank_runtime_proxy_rankmax"],
        "rank_runtime_proxy_reduction_pct": pct_drop(
            baseline["rank_runtime_proxy_rankmax"], current["rank_runtime_proxy_rankmax"]
        ),
        "baseline_rank_runtime_proxy_by_rank": baseline["rank_runtime_proxy_by_rank"],
        "replicated_rank_runtime_proxy_by_rank": current["rank_runtime_proxy_by_rank"],
        "baseline_comm_cost_rankmax": baseline["comm_cost_rankmax"],
        "replicated_comm_cost_rankmax": current["comm_cost_rankmax"],
        "comm_rankmax_reduction_pct": pct_drop(baseline["comm_cost_rankmax"], current["comm_cost_rankmax"]),
        "baseline_comm_cost_by_source_rank": baseline["comm_cost_by_source_rank"],
        "replicated_comm_cost_by_source_rank": current["comm_cost_by_source_rank"],
        "baseline_cross_node_tokens": baseline["cross_node_tokens"],
        "replicated_cross_node_tokens": current["cross_node_tokens"],
        "cross_node_reduction_pct": pct_drop(baseline["cross_node_tokens"], current["cross_node_tokens"]),
        "baseline_cross_node_tokens_rankmax": baseline["cross_node_tokens_rankmax"],
        "replicated_cross_node_tokens_rankmax": current["cross_node_tokens_rankmax"],
        "cross_node_rankmax_reduction_pct": pct_drop(baseline["cross_node_tokens_rankmax"], current["cross_node_tokens_rankmax"]),
        "baseline_cross_node_tokens_by_source_rank": baseline["cross_node_tokens_by_source_rank"],
        "replicated_cross_node_tokens_by_source_rank": current["cross_node_tokens_by_source_rank"],
        "baseline_inter_node_dispatch_bytes": baseline["inter_node_dispatch_bytes"],
        "replicated_inter_node_dispatch_bytes": current["inter_node_dispatch_bytes"],
        "inter_node_dispatch_bytes_reduction_pct": pct_drop(baseline["inter_node_dispatch_bytes"], current["inter_node_dispatch_bytes"]),
        "baseline_inter_node_combine_bytes": baseline["inter_node_combine_bytes"],
        "replicated_inter_node_combine_bytes": current["inter_node_combine_bytes"],
        "inter_node_combine_bytes_reduction_pct": pct_drop(baseline["inter_node_combine_bytes"], current["inter_node_combine_bytes"]),
        "baseline_inter_node_total_bytes": baseline_inter_node_total_bytes,
        "replicated_inter_node_total_bytes": current_inter_node_total_bytes,
        "cross_node_bytes_reduction_pct": float(cross_node_bytes_reduction_pct),
        "baseline_inter_node_dispatch_bytes_by_source_rank": baseline["inter_node_dispatch_bytes_by_source_rank"],
        "replicated_inter_node_dispatch_bytes_by_source_rank": current["inter_node_dispatch_bytes_by_source_rank"],
        "baseline_inter_node_combine_bytes_by_source_rank": baseline["inter_node_combine_bytes_by_source_rank"],
        "replicated_inter_node_combine_bytes_by_source_rank": current["inter_node_combine_bytes_by_source_rank"],
        "baseline_inter_node_total_bytes_by_source_rank": baseline["inter_node_total_bytes_by_source_rank"],
        "replicated_inter_node_total_bytes_by_source_rank": current["inter_node_total_bytes_by_source_rank"],
        "baseline_inter_node_dispatch_bytes_rankmax": baseline["inter_node_dispatch_bytes_rankmax"],
        "replicated_inter_node_dispatch_bytes_rankmax": current["inter_node_dispatch_bytes_rankmax"],
        "rankmax_inter_node_dispatch_bytes_reduction_pct": pct_drop(
            baseline["inter_node_dispatch_bytes_rankmax"], current["inter_node_dispatch_bytes_rankmax"]
        ),
        "baseline_inter_node_combine_bytes_rankmax": baseline["inter_node_combine_bytes_rankmax"],
        "replicated_inter_node_combine_bytes_rankmax": current["inter_node_combine_bytes_rankmax"],
        "rankmax_inter_node_combine_bytes_reduction_pct": pct_drop(
            baseline["inter_node_combine_bytes_rankmax"], current["inter_node_combine_bytes_rankmax"]
        ),
        "baseline_inter_node_total_bytes_rankmax": baseline["inter_node_total_bytes_rankmax"],
        "replicated_inter_node_total_bytes_rankmax": current["inter_node_total_bytes_rankmax"],
        "rankmax_inter_node_bytes_reduction_pct": float(rankmax_inter_node_bytes_reduction_pct),
        "baseline_active_cross_node_pairs_total": baseline["active_cross_node_pairs_total"],
        "replicated_active_cross_node_pairs_total": current["active_cross_node_pairs_total"],
        "active_cross_node_pairs_reduction_pct": pct_drop(baseline["active_cross_node_pairs_total"], current["active_cross_node_pairs_total"]),
        "baseline_small_cross_node_pairs_total": baseline.get("small_cross_node_pairs_total", 0),
        "replicated_small_cross_node_pairs_total": current.get("small_cross_node_pairs_total", 0),
        "small_cross_node_pairs_reduction_pct": pct_drop(baseline.get("small_cross_node_pairs_total", 0), current.get("small_cross_node_pairs_total", 0)),
        "num_extra_replicas": int(len(selected)),
        "selected_cross_node_replicas": int(selected_cross_node_replicas),
        "selected_same_node_replicas": int(selected_same_node_replicas),
        "extra_replicas_per_layer": extra_by_layer.astype(int).tolist(),
        "extra_replicas_per_rank_per_layer": extra_by_layer_rank.astype(int).tolist(),
        "assigned_load_by_rank": current["rank_load_by_layer"],
        "max_rank_load": current["max_rank_load"],
        "load_imbalance_ratio": current["load_imbalance_ratio"],
        "baseline_h2d_miss_proxy_total": baseline["h2d_miss_proxy_total"],
        "replicated_h2d_miss_proxy_total": current["h2d_miss_proxy_total"],
        "h2d_miss_proxy_change_pct": float(
            100.0 * (current["h2d_miss_proxy_total"] - baseline["h2d_miss_proxy_total"]) / max(baseline["h2d_miss_proxy_total"], 1e-9)
        ),
        "baseline_h2d_miss_proxy_rankmax": baseline["h2d_miss_proxy_rankmax"],
        "replicated_h2d_miss_proxy_rankmax": current["h2d_miss_proxy_rankmax"],
        "baseline_compute_proxy_rankmax": baseline["compute_proxy_rankmax"],
        "replicated_compute_proxy_rankmax": current["compute_proxy_rankmax"],
        "compute_proxy_rankmax_change_pct": float(
            100.0 * (current["compute_proxy_rankmax"] - baseline["compute_proxy_rankmax"]) / max(baseline["compute_proxy_rankmax"], 1e-9)
        ),
        "baseline_hotset_count_total": baseline["hotset_count_total"],
        "replicated_hotset_count_total": current["hotset_count_total"],
        "hotset_count_change_pct": float(
            100.0 * (current["hotset_count_total"] - baseline["hotset_count_total"]) / max(baseline["hotset_count_total"], 1e-9)
        ),
        "baseline_hotset_count_rankmax": baseline["hotset_count_rankmax"],
        "replicated_hotset_count_rankmax": current["hotset_count_rankmax"],
        "rank_compute_time_scale": [float(x) for x in cfg.rank_compute_time_scale],
        "rank_h2d_time_scale": [float(x) for x in cfg.rank_h2d_time_scale],
        "cost_unit": str(model.cost_unit),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "bytes_source": str(model.bytes_source),
        "latency_semantics": str(model.latency_semantics),
        "min_net_gain_ms": float(cfg.min_net_gain_ms),
        "min_rankmax_benefit": float(cfg.min_rankmax_benefit),
        "h2d_ms_per_proxy": float(cfg.h2d_ms_per_proxy),
        "compute_ms_per_proxy": float(cfg.compute_ms_per_proxy),
        "latency_ms_per_event": (
            None if cfg.latency_ms_per_event is None else float(cfg.latency_ms_per_event)
        ),
        "throughput_objective_enabled": bool(_is_throughput_objective(cfg)),
        "decode_rs_gain_weight": float(cfg.decode_rs_gain_weight),
        "require_rankmax_gain": bool(cfg.require_rankmax_gain),
        "require_h2d_nondegrade": bool(cfg.require_h2d_nondegrade),
        "require_compute_nondegrade": bool(cfg.require_compute_nondegrade),
        "require_hotset_nondegrade": bool(cfg.require_hotset_nondegrade),
        "require_inter_node_rankmax_bytes_gain": bool(cfg.require_inter_node_rankmax_bytes_gain),
        "allow_same_node_replicas": bool(cfg.allow_same_node_replicas),
        "min_inter_node_rankmax_bytes_reduction_pct": float(cfg.min_inter_node_rankmax_bytes_reduction_pct),
        "min_inter_node_bytes_reduction_pct": float(cfg.min_inter_node_bytes_reduction_pct),
        "hotset_min_load_fraction": float(cfg.hotset_min_load_fraction),
        "hotset_slack": float(cfg.hotset_slack),
        "h2d_gate_enabled": bool(h2d_gate_enabled),
        "compute_gate_enabled": bool(compute_gate_enabled),
        "hotset_gate_enabled": bool(hotset_gate_enabled),
        "effective_h2d_slack": float(h2d_slack),
        "effective_compute_slack": float(compute_slack),
        "effective_hotset_slack": float(hotset_slack),
        "expert_cache_capacity": int(cfg.expert_cache_capacity),
        "expert_cache_capacity_by_rank": [int(x) for x in cfg.expert_cache_capacity_by_rank],
        "safety_rejected_candidates": rejected,
        "scp_noop": bool(len(noop_reasons) > 0),
        "should_run_runtime": bool(len(noop_reasons) == 0),
        "noop_reason": ";".join(noop_reasons),
        "estimated_h2d_risk_score": float(max(0.0, current["h2d_miss_proxy_rankmax"] / max(baseline["h2d_miss_proxy_rankmax"], 1e-9) - 1.0)),
        "h2d_model_semantics": "frequency_only_topK_proxy_not_cache_replay_required_for_final_claims",
        "prefill_runtime_assignment_comm_reduction_pct": pct_drop(prefill_baseline["comm_cost"], prefill_current["comm_cost"]),
        "prefill_runtime_assignment_cross_node_reduction_pct": pct_drop(prefill_baseline["cross_node_tokens"], prefill_current["cross_node_tokens"]),
        "decode_runtime_assignment_comm_reduction_pct": pct_drop(decode_baseline["comm_cost"], decode_current["comm_cost"]),
        "decode_runtime_assignment_cross_node_reduction_pct": pct_drop(decode_baseline["cross_node_tokens"], decode_current["cross_node_tokens"]),
        # Backward-compatible metric aliases; the semantics are no longer
        # solver-owned choice tables.
        "prefill_choice_comm_reduction_pct": pct_drop(prefill_baseline["comm_cost"], prefill_current["comm_cost"]),
        "prefill_choice_cross_node_reduction_pct": pct_drop(prefill_baseline["cross_node_tokens"], prefill_current["cross_node_tokens"]),
        "decode_choice_comm_reduction_pct": pct_drop(decode_baseline["comm_cost"], decode_current["comm_cost"]),
        "decode_choice_cross_node_reduction_pct": pct_drop(decode_baseline["cross_node_tokens"], decode_current["cross_node_tokens"]),
        "runtime_assignment_policy": "nearest_topology_cost_from_replicas_tie_keep_round_robin_owner",
        "decode_choice_policy": "deprecated_solver_choice_not_used_runtime_derives_nearest_replica",
    }

    return {
        "format": "replicated_scp_ep_v2",
        "world_size": int(world),
        "num_layers": int(layers),
        "num_experts": int(experts),
        "lambda_prefill": float(cfg.lambda_prefill),
        "lambda_decode": float(cfg.lambda_decode),
        "stage": str(cfg.stage),
        "objective": str(cfg.objective),
        "global_extra_replicas": int(cfg.global_extra_replicas),
        "lambda_event": float(cfg.lambda_event),
        "lambda_small_packet": float(cfg.lambda_small_packet),
        "small_packet_threshold": int(cfg.small_packet_threshold),
        "load_slack": float(cfg.load_slack),
        "h2d_slack": float(cfg.h2d_slack),
        "compute_slack": float(cfg.compute_slack),
        "min_net_gain_ms": float(cfg.min_net_gain_ms),
        "min_rankmax_benefit": float(cfg.min_rankmax_benefit),
        "h2d_ms_per_proxy": float(cfg.h2d_ms_per_proxy),
        "compute_ms_per_proxy": float(cfg.compute_ms_per_proxy),
        "latency_ms_per_event": (
            None if cfg.latency_ms_per_event is None else float(cfg.latency_ms_per_event)
        ),
        "decode_rs_gain_weight": float(cfg.decode_rs_gain_weight),
        "require_rankmax_gain": bool(cfg.require_rankmax_gain),
        "require_h2d_nondegrade": bool(cfg.require_h2d_nondegrade),
        "require_compute_nondegrade": bool(cfg.require_compute_nondegrade),
        "require_hotset_nondegrade": bool(cfg.require_hotset_nondegrade),
        "require_inter_node_rankmax_bytes_gain": bool(cfg.require_inter_node_rankmax_bytes_gain),
        "allow_same_node_replicas": bool(cfg.allow_same_node_replicas),
        "min_inter_node_rankmax_bytes_reduction_pct": float(cfg.min_inter_node_rankmax_bytes_reduction_pct),
        "min_inter_node_bytes_reduction_pct": float(cfg.min_inter_node_bytes_reduction_pct),
        "hotset_min_load_fraction": float(cfg.hotset_min_load_fraction),
        "hotset_slack": float(cfg.hotset_slack),
        "cost_unit": str(model.cost_unit),
        "replica_choice_cost_matrix": _replica_choice_cost_matrix(model),
        "dispatch_item_bytes": float(model.dispatch_item_bytes),
        "combine_item_bytes": float(model.combine_item_bytes),
        "hidden_bytes": float(model.hidden_bytes),
        "router_weight_bytes": float(model.router_weight_bytes),
        "expert_id_bytes": float(model.expert_id_bytes),
        "communication_model": {
            "dispatch_item_bytes": float(model.dispatch_item_bytes),
            "combine_item_bytes": float(model.combine_item_bytes),
            "hidden_bytes": float(model.hidden_bytes),
            "router_weight_bytes": float(model.router_weight_bytes),
            "expert_id_bytes": float(model.expert_id_bytes),
            "bytes_source": str(model.bytes_source),
            "latency_semantics": str(model.latency_semantics),
        },
        "expert_cache_capacity": int(cfg.expert_cache_capacity),
        "expert_cache_capacity_by_rank": [int(x) for x in cfg.expert_cache_capacity_by_rank],
        "rank_compute_time_scale": [float(x) for x in cfg.rank_compute_time_scale],
        "rank_h2d_time_scale": [float(x) for x in cfg.rank_h2d_time_scale],
        "replica_budget": {
            "extra_replicas_per_layer": int(cfg.extra_replicas_per_layer),
            "extra_replicas_per_rank_per_layer": int(cfg.extra_replicas_per_rank_per_layer),
            "global_extra_replicas": int(cfg.global_extra_replicas),
            "max_replicas_per_expert": int(cfg.max_replicas_per_expert),
        },
        "rank_to_node": {str(k): int(v) for k, v in model.rank_to_node.items()},
        "replicas": replicas,
        "runtime_assignment_semantics": (
            "solver outputs only replicas; standard EP runtime derives source-rank "
            "dispatch targets from replicas and replica_choice_cost_matrix at layer initialization"
        ),
        "selected_replicas": selected,
        "offline_metrics": metrics,
    }


def report_markdown(result: Dict[str, Any]) -> str:
    m = result["offline_metrics"]
    h2d_semantics = str(m.get("h2d_model_semantics", ""))
    is_cacheaware = "ordered_trace" in h2d_semantics or "cache replay" in h2d_semantics.lower()

    def _fmt_optional_pct(key: str) -> str:
        value = m.get(key)
        if value is None:
            return "n/a"
        return f"{float(value):.2f}"

    lines = [
        "# Replicated SCP-EP Report",
        "",
        "## Strategy",
        "Router expert IDs and router scores are unchanged. Each logical expert keeps its round-robin primary owner and may receive limited executable replicas. The solver outputs only `replicas[layer][expert]`; standard EP runtime derives the concrete source-rank -> executor-rank lookup from the replica set and topology at layer initialization. Each token-expert item is executed by exactly one replica.",
        "",
        (
            "Replica placement is not GPU residency. H2D is estimated by ordered-trace per-layer/per-rank LRU replay."
            if is_cacheaware
            else "Replica placement is not GPU residency. The reported H2D metric is a frequency-only proxy, not a cache replay. Final claims require runtime/cache replay validation."
        ),
        "",
        "## Communication Model",
        f"- cost_unit: {m['cost_unit']}",
        f"- dispatch_item_bytes: {m['dispatch_item_bytes']:.1f}",
        f"- combine_item_bytes: {m['combine_item_bytes']:.1f}",
        f"- hidden_bytes: {m.get('hidden_bytes', result.get('hidden_bytes', 0.0)):.1f}",
        f"- router_weight_bytes: {m.get('router_weight_bytes', result.get('router_weight_bytes', 0.0)):.1f}",
        f"- expert_id_bytes: {m.get('expert_id_bytes', result.get('expert_id_bytes', 0.0)):.1f}",
        f"- bytes_source: {m.get('bytes_source', result.get('communication_model', {}).get('bytes_source', 'unknown'))}",
        f"- latency_semantics: {m.get('latency_semantics', result.get('communication_model', {}).get('latency_semantics', 'unknown'))}",
        "",
        "## Metrics",
        f"- baseline_objective_ms: {m.get('baseline_objective_ms', m.get('baseline_rank_runtime_proxy_rankmax', 0.0)):.6f}",
        f"- replicated_objective_ms: {m.get('replicated_objective_ms', m.get('replicated_rank_runtime_proxy_rankmax', 0.0)):.6f}",
        f"- objective_reduction_pct: {m.get('objective_reduction_pct', m.get('rank_runtime_proxy_reduction_pct', 0.0)):.2f}",
        f"- baseline_grouped_J_comm_ms: {m.get('baseline_grouped_J_comm_ms', 0.0):.6f}",
        f"- replicated_grouped_J_comm_ms: {m.get('replicated_grouped_J_comm_ms', 0.0):.6f}",
        f"- grouped_J_comm_reduction_pct: {m.get('grouped_J_comm_reduction_pct', 0.0):.2f}",
        f"- baseline_rankmax_comm_ms: {m.get('baseline_rankmax_comm_ms', m.get('baseline_comm_cost_rankmax', 0.0)):.6f}",
        f"- replicated_rankmax_comm_ms: {m.get('replicated_rankmax_comm_ms', m.get('replicated_comm_cost_rankmax', 0.0)):.6f}",
        f"- rankmax_comm_reduction_pct: {m.get('rankmax_comm_reduction_pct', m.get('comm_rankmax_reduction_pct', 0.0)):.2f}",
        f"- baseline_rankmax_h2d_ms: {m.get('baseline_rankmax_h2d_ms', m.get('baseline_h2d_miss_proxy_rankmax', 0.0)):.6f}",
        f"- replicated_rankmax_h2d_ms: {m.get('replicated_rankmax_h2d_ms', m.get('replicated_h2d_miss_proxy_rankmax', 0.0)):.6f}",
        f"- rankmax_h2d_change_pct: {m.get('rankmax_h2d_change_pct', m.get('h2d_miss_proxy_change_pct', 0.0)):.2f}",
        f"- baseline_resident_hit_rate_by_item: {m.get('baseline_resident_hit_rate_by_item', 0.0):.6f}",
        f"- replicated_resident_hit_rate_by_item: {m.get('replicated_resident_hit_rate_by_item', 0.0):.6f}",
        f"- cross_node_bytes_reduction_pct: {m['cross_node_bytes_reduction_pct']:.2f}",
        f"- rankmax_inter_node_bytes_reduction_pct: {m.get('rankmax_inter_node_bytes_reduction_pct', 0.0):.2f}",
        f"- baseline_cross_node_tokens: {m.get('baseline_cross_node_tokens', m.get('baseline_cross_node_items', 0.0)):.1f}",
        f"- replicated_cross_node_tokens: {m.get('replicated_cross_node_tokens', m.get('replicated_cross_node_items', 0.0)):.1f}",
        f"- cross_node_reduction_pct: {m['cross_node_reduction_pct']:.2f}",
        f"- num_extra_replicas: {m['num_extra_replicas']}",
        f"- selected_cross_node_replicas: {m.get('selected_cross_node_replicas', 0)}",
        f"- selected_same_node_replicas: {m.get('selected_same_node_replicas', 0)}",
        f"- selected_event_reducing_replicas: {m.get('selected_event_reducing_replicas', 0)}",
        f"- selected_cross_node_events_eliminated: {m.get('selected_cross_node_events_eliminated', 0)}",
        f"- selected_cross_node_events_created: {m.get('selected_cross_node_events_created', 0)}",
        f"- active_cross_node_events_reduction_pct: {m.get('active_cross_node_events_reduction_pct', 0.0):.2f}",
        f"- scp_stackability_status: {m.get('scp_stackability_status', 'n/a')}",
        f"- scp_stackability_reasons: {m.get('scp_stackability_reasons', [])}",
        f"- selected_replicas_by_direction: {m.get('selected_replicas_by_direction', {})}",
        f"- comm_gain_by_source_node: {m.get('comm_gain_by_source_node', {})}",
        f"- optimistic_upper_bound_tail_headroom_pct: {m.get('optimistic_upper_bound_tail_headroom_pct', 0.0):.2f}",
        f"- cache_validation_pass: {m.get('cache_validation_pass', 'n/a')}",
        f"- should_run_runtime: {m.get('should_run_runtime', True)}",
        f"- noop_reason: {m.get('noop_reason', '')}",
        f"- load_imbalance_ratio: {m.get('load_imbalance_ratio', 0.0):.3f}",
        f"- h2d_miss_proxy_change_pct: {m.get('h2d_miss_proxy_change_pct', 0.0):.2f}",
        f"- compute_proxy_rankmax_change_pct: {m.get('compute_proxy_rankmax_change_pct', 0.0):.2f}",
        f"- throughput_objective_enabled: {m.get('throughput_objective_enabled', False)}",
        f"- h2d_ms_per_proxy: {m.get('h2d_ms_per_proxy', 0.0):.8f}",
        f"- compute_ms_per_proxy: {m.get('compute_ms_per_proxy', 0.0):.8f}",
        f"- decode_rs_gain_weight: {m.get('decode_rs_gain_weight', 0.0):.3f} (deprecated, ignored by SCP solver)",
        f"- prefill_runtime_assignment_comm_reduction_pct: {_fmt_optional_pct('prefill_runtime_assignment_comm_reduction_pct')}",
        f"- decode_runtime_assignment_comm_reduction_pct: {_fmt_optional_pct('decode_runtime_assignment_comm_reduction_pct')}",
        f"- runtime_assignment_policy: {m.get('runtime_assignment_policy', '')}",
        f"- h2d_model_semantics: {m['h2d_model_semantics']}",
        "",
        "## Selected Replicas",
        "| layer | expert | rr owner | replica rank | cross-node | event_reduction | objective_gain_ms | comm_gain | h2d_delta | compute_delta |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selected_replicas"]:
        lines.append(
            f"| {row['layer']} | {row['expert']} | {row.get('rr_owner', '')} | {row['rank']} | "
            f"{bool(row.get('cross_node_replica', False))} | "
            f"{int(row.get('active_cross_node_event_reduction', row.get('cross_node_events_eliminated', 0)))} | "
            f"{row.get('gain_noRS_objective_ms', row.get('runtime_rankmax_gain', row.get('J_comm_gain_ms', row.get('net_gain', 0.0)))):.6f} | "
            f"{row.get('benefit', row.get('approx_comm_gain', row.get('J_comm_gain_ms', 0.0))):.6f} | "
            f"{row.get('h2d_proxy_delta_total', 0.0):.6f} | {row.get('compute_proxy_delta_rankmax', 0.0):.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve profile-calibrated replicated SCP-EP placement for standard EP dispatch.")
    ap.add_argument("--profile", default="", help="Routing profile. Used as train profile unless --train-profile is set.")
    ap.add_argument("--train-profile", default="", help="Profile used to select replicated SCP candidates.")
    ap.add_argument("--valid-profile", default="", help="Held-out profile used only for validation replay.")
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", default="data/placements/replicated_scp_ep.json")
    ap.add_argument("--lambda-prefill", type=float, default=0.2)
    ap.add_argument("--lambda-decode", type=float, default=1.0)
    ap.add_argument("--stage", choices=["mixed", "prefill", "decode"], default="mixed", help="Profile stage used for solving. Use prefill for stage-aware SCP.")
    ap.add_argument(
        "--objective",
        default="throughput_rankmax_v1",
        choices=[
            "legacy",
            "prefill_ranktail_v1",
            "ranktail",
            "rank_tail",
            "throughput_rankmax_v1",
            "runtime_rankmax_v1",
            "rankmax_runtime",
            "throughput",
            "cacheaware_ranktail_v1",
            "cache_aware_ranktail_v1",
            "cacheaware",
            "tail_headroom_scp_v2",
            "tail_headroom",
            "grouped_tail",
            "phase_event_cache_scp_v2",
            "phase_event_cache",
            "online_aware_scp_v1",
            "online_aware_scp",
            "online_aware",
            "onlineaware_scp_v1",
            "onlineaware",
            "task_local_tail_scp_v1",
            "tasklocal_tail_scp_v1",
            "task_local",
            "load_only_replicated_v1",
            "loadonly_replicated_v1",
            "load_only",
            "loadonly",
            "comm_only_tasklocal_v1",
            "commonly_tasklocal_v1",
            "comm_only",
            "commonly",
        ],
        help=(
            "SCP objective. phase_event_cache_scp_v2 is the safe grouped-tail objective: "
            "phase-aware, event-aware when latency is active, and cache/H2D guarded."
        ),
    )
    ap.add_argument("--extra-replicas-per-layer", type=int, default=4)
    ap.add_argument("--extra-replicas-per-rank-per-layer", type=int, default=2, help="0 disables the per-layer-rank cap.")
    ap.add_argument("--global-extra-replicas", type=int, default=0, help="If >0, use a global replica budget instead of fixed per-layer budget.")
    ap.add_argument("--max-replicas-per-expert", type=int, default=2)
    ap.add_argument("--lambda-event", type=float, default=0.0, help="Penalty per active cross-node source/destination event in the solver objective.")
    ap.add_argument("--lambda-small-packet", type=float, default=0.0, help="Penalty per small cross-node packet in the solver objective.")
    ap.add_argument("--small-packet-threshold", type=int, default=4, help="Cross-node pair item threshold for small-packet counting.")
    ap.add_argument("--event-aware", action="store_true", default=True, help="For grouped-tail SCP, include cross-node transfer/event reduction in candidate ranking.")
    ap.add_argument("--no-event-aware", action="store_false", dest="event_aware")
    ap.add_argument("--event-first", action="store_true", default=True, help="For latency>0 grouped-tail SCP, prefer candidates that eliminate a cross-node transfer/event.")
    ap.add_argument("--no-event-first", action="store_false", dest="event_first")
    ap.add_argument(
        "--require-event-gain-when-latency",
        action="store_true",
        help=(
            "For latency>0 grouped-tail SCP, require replayed runtime packet cost to improve. "
            "Event-count reduction is reported separately; bandwidth-only gains are still allowed."
        ),
    )
    ap.add_argument("--load-slack", type=float, default=0.15)
    ap.add_argument("--expert-cache-capacity", type=int, default=8)
    ap.add_argument("--expert-cache-capacity-by-rank", default="", help="Comma list or rank:value map, e.g. 8,8,4,4")
    ap.add_argument("--h2d-slack", type=float, default=0.03, help="Allow replay H2D rankmax ms to grow by this fraction vs RR. Use --disable-h2d-gate to turn off.")
    ap.add_argument("--disable-h2d-gate", action="store_true")
    ap.add_argument("--compute-slack", type=float, default=0.0)
    ap.add_argument("--cache-hit-drop-limit", type=float, default=0.0)
    ap.add_argument("--candidate-prefilter-topn", type=int, default=-1, help="Default is 32 for tail_headroom_scp_v2 and 256 for legacy objectives.")
    ap.add_argument("--require-rs-safe", action="store_true")
    ap.add_argument("--rs-total-slack", type=float, default=0.0)
    ap.add_argument("--rs-h2d-slack", type=float, default=0.03)
    ap.add_argument("--rs-service-hit-drop-limit", type=float, default=0.0)
    ap.add_argument("--rs-prefetch-use-drop-limit", type=float, default=0.0)
    ap.add_argument("--require-valid-positive", action="store_true")
    ap.add_argument("--min-tail-headroom-pct", type=float, default=0.5)
    ap.add_argument("--min-comm-gain-ms", type=float, default=0.0)
    ap.add_argument("--cache-validation-interval", type=int, default=4)
    ap.add_argument("--min-benefit", type=float, default=0.0)
    ap.add_argument("--min-net-gain-ms", type=float, default=0.0)
    ap.add_argument("--min-rankmax-benefit", type=float, default=0.0)
    ap.add_argument("--h2d-ms-per-proxy", type=float, default=0.001)
    ap.add_argument("--compute-ms-per-proxy", type=float, default=0.001)
    ap.add_argument(
        "--runtime-latency-ms",
        type=float,
        default=None,
        help=(
            "Runtime SPARSE_SIM_INTER_NODE_LAT_MS used for the solver. "
            "When set, it overrides topology cross_node_latency_ms so offline "
            "placement and runtime weaknet sleep optimize the same latency."
        ),
    )
    ap.add_argument(
        "--decode-rs-gain-weight",
        type=float,
        default=0.0,
        help="Deprecated compatibility option. Replicated SCP solver is independent of RS and does not use this value.",
    )
    ap.add_argument("--require-rankmax-gain", action="store_true")
    ap.add_argument("--require-h2d-nondegrade", action="store_true", default=False, help=argparse.SUPPRESS)
    ap.add_argument("--allow-h2d-degrade", action="store_false", dest="require_h2d_nondegrade")
    ap.add_argument("--require-compute-nondegrade", action="store_true", default=False, help=argparse.SUPPRESS)
    ap.add_argument("--allow-compute-degrade", action="store_false", dest="require_compute_nondegrade")
    ap.add_argument("--require-hotset-nondegrade", action="store_true", default=True)
    ap.add_argument("--allow-hotset-degrade", action="store_false", dest="require_hotset_nondegrade")
    ap.add_argument(
        "--require-inter-node-rankmax-bytes-gain",
        action="store_true",
        help="Require every selected replica to reduce the rankmax inter-node bytes. This is the lat0/bandwidth-only SCP gate.",
    )
    ap.add_argument(
        "--allow-same-node-replicas",
        action="store_true",
        help="Allow replicas on the same topology node as the round-robin owner. Default forbids them because they cannot reduce cross-node weaknet bytes.",
    )
    ap.add_argument("--min-inter-node-rankmax-bytes-reduction-pct", type=float, default=0.5)
    ap.add_argument("--min-inter-node-bytes-reduction-pct", type=float, default=0.1)
    ap.add_argument("--hotset-min-load-fraction", type=float, default=0.02)
    ap.add_argument("--hotset-slack", type=float, default=0.0)
    ap.add_argument("--hidden-bytes", type=float, default=0.0, help="Override hidden bytes per token-expert item. Used only if effective dispatch/combine bytes are absent.")
    ap.add_argument("--dispatch-item-bytes", type=float, default=0.0, help="Override effective dispatch bytes per token-expert item.")
    ap.add_argument("--combine-item-bytes", type=float, default=0.0, help="Override effective combine bytes per token-expert item.")
    ap.add_argument("--dispatch-weight-bytes", type=float, default=2.0, help="Audited default router weight bytes sent with hidden in dispatch.")
    ap.add_argument("--dispatch-expert-id-bytes", type=float, default=8.0, help="Audited default expert-id bytes sent in dispatch.")
    ap.add_argument("--disable-dispatch-metadata-bytes", action="store_true", help="Do not add router-weight/expert-id bytes when effective dispatch bytes are absent.")
    ap.add_argument("--num-experts", type=int, default=0, help="Override routed expert dimension for model-specific placement output.")
    ap.add_argument("--rank-compute-time-scale", default="", help="Comma list or rank:value map. Time multiplier; lower means faster, e.g. 1,1,0.7,0.7")
    ap.add_argument("--node-compute-time-scale", default="", help="Comma list or node:value map of compute time multipliers.")
    ap.add_argument("--rank-h2d-time-scale", default="", help="Comma list or rank:value map. Time multiplier; higher means slower H2D.")
    ap.add_argument("--node-h2d-time-scale", default="", help="Comma list or node:value map of H2D time multipliers.")
    # Backward-compatible aliases.  They are interpreted as time scales, not speed scales.
    ap.add_argument("--rank-compute-scale", default="", help=argparse.SUPPRESS)
    ap.add_argument("--node-compute-scale", default="", help=argparse.SUPPRESS)
    ap.add_argument("--rank-h2d-scale", default="", help=argparse.SUPPRESS)
    ap.add_argument("--node-h2d-scale", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    profile_path_raw = str(args.train_profile or args.profile or "").strip()
    if not profile_path_raw:
        raise SystemExit("--profile or --train-profile is required")
    profile_path = Path(profile_path_raw).expanduser()
    prefill, decode, meta = load_profile(profile_path)
    prefill, decode, meta = select_num_experts(prefill, decode, meta, int(args.num_experts))
    valid_prefill = None
    valid_decode = None
    valid_meta: Dict[str, Any] = {}
    valid_profile_path_raw = str(args.valid_profile or "").strip()
    if valid_profile_path_raw:
        valid_prefill, valid_decode, valid_meta = load_profile(Path(valid_profile_path_raw).expanduser())
        valid_prefill, valid_decode, valid_meta = select_num_experts(valid_prefill, valid_decode, valid_meta, int(args.num_experts))
    topology = load_topology(Path(args.topology).expanduser())
    model = build_comm_model(
        topology,
        meta,
        dispatch_item_bytes_override=float(args.dispatch_item_bytes),
        combine_item_bytes_override=float(args.combine_item_bytes),
        hidden_bytes_override=float(args.hidden_bytes),
        dispatch_weight_bytes=float(args.dispatch_weight_bytes),
        dispatch_expert_id_bytes=float(args.dispatch_expert_id_bytes),
        include_dispatch_metadata=not bool(args.disable_dispatch_metadata_bytes),
        runtime_latency_ms_override=args.runtime_latency_ms,
    )

    compute_rank_arg = str(args.rank_compute_time_scale or args.rank_compute_scale or "")
    compute_node_arg = str(args.node_compute_time_scale or args.node_compute_scale or "")
    h2d_rank_arg = str(args.rank_h2d_time_scale or args.rank_h2d_scale or "")
    h2d_node_arg = str(args.node_h2d_time_scale or args.node_h2d_scale or "")
    topo_compute_time = topology.get("rank_compute_time_scale")
    topo_compute_speed = topology.get("rank_compute_scale")
    topo_h2d_time = topology.get("rank_h2d_time_scale", topology.get("rank_h2d_scale"))

    compute_topology_values = topo_compute_time if isinstance(topo_compute_time, list) else None
    if compute_topology_values is None and isinstance(topo_compute_speed, list):
        compute_topology_values = _invert_time_from_speed(topo_compute_speed)

    compute_from_legacy_speed = False
    compute_rank_arg_raw = str(args.rank_compute_scale or "")
    compute_node_arg_raw = str(args.node_compute_scale or "")
    if not str(args.rank_compute_time_scale or "").strip() and not str(args.node_compute_time_scale or "").strip():
        compute_from_legacy_speed = bool(compute_rank_arg_raw.strip() or compute_node_arg_raw.strip())

    rank_compute_time_scale = resolve_rank_scales(
        world=model.world,
        rank_to_node=model.rank_to_node,
        rank_scale=compute_rank_arg,
        node_scale=compute_node_arg,
        topology_values=compute_topology_values,
        default=1.0,
    )
    if compute_from_legacy_speed:
        rank_compute_time_scale = _invert_time_from_speed(rank_compute_time_scale)
    rank_h2d_time_scale = resolve_rank_scales(
        world=model.world,
        rank_to_node=model.rank_to_node,
        rank_scale=h2d_rank_arg,
        node_scale=h2d_node_arg,
        topology_values=topo_h2d_time if isinstance(topo_h2d_time, list) else None,
        default=1.0,
    )
    expert_cache_capacity_by_rank = _parse_int_capacity_vector(
        str(args.expert_cache_capacity_by_rank or ""),
        world=int(model.world),
        default=int(args.expert_cache_capacity),
        name="--expert-cache-capacity-by-rank",
    )

    cfg = ReplicatedSCPConfig(
        lambda_prefill=float(args.lambda_prefill),
        lambda_decode=float(args.lambda_decode),
        extra_replicas_per_layer=int(args.extra_replicas_per_layer),
        extra_replicas_per_rank_per_layer=int(args.extra_replicas_per_rank_per_layer),
        global_extra_replicas=int(args.global_extra_replicas),
        max_replicas_per_expert=int(args.max_replicas_per_expert),
        stage=str(args.stage),
        objective=str(args.objective),
        lambda_event=float(args.lambda_event),
        lambda_small_packet=float(args.lambda_small_packet),
        small_packet_threshold=max(1, int(args.small_packet_threshold)),
        load_slack=float(args.load_slack),
        expert_cache_capacity=int(args.expert_cache_capacity),
        expert_cache_capacity_by_rank=expert_cache_capacity_by_rank,
        h2d_slack=float(args.h2d_slack),
        compute_slack=float(args.compute_slack),
        min_benefit=float(args.min_benefit),
        min_net_gain_ms=float(args.min_net_gain_ms),
        min_rankmax_benefit=float(args.min_rankmax_benefit),
        h2d_ms_per_proxy=float(args.h2d_ms_per_proxy),
        compute_ms_per_proxy=float(args.compute_ms_per_proxy),
        latency_ms_per_event=None if args.runtime_latency_ms is None else float(args.runtime_latency_ms),
        decode_rs_gain_weight=float(args.decode_rs_gain_weight),
        require_rankmax_gain=bool(args.require_rankmax_gain),
        require_h2d_nondegrade=bool(args.require_h2d_nondegrade),
        require_compute_nondegrade=bool(args.require_compute_nondegrade),
        require_hotset_nondegrade=bool(args.require_hotset_nondegrade),
        require_inter_node_rankmax_bytes_gain=bool(args.require_inter_node_rankmax_bytes_gain),
        allow_same_node_replicas=bool(args.allow_same_node_replicas),
        min_inter_node_rankmax_bytes_reduction_pct=float(args.min_inter_node_rankmax_bytes_reduction_pct),
        min_inter_node_bytes_reduction_pct=float(args.min_inter_node_bytes_reduction_pct),
        hotset_min_load_fraction=float(args.hotset_min_load_fraction),
        hotset_slack=float(args.hotset_slack),
        rank_compute_time_scale=rank_compute_time_scale,
        rank_h2d_time_scale=rank_h2d_time_scale,
        disable_h2d_gate=bool(args.disable_h2d_gate),
        cache_hit_drop_limit=float(args.cache_hit_drop_limit),
        candidate_prefilter_topn=(
            (64 if _is_online_aware_objective_name(str(args.objective)) else (32 if _is_tail_headroom_objective_name(str(args.objective)) else 256))
            if int(args.candidate_prefilter_topn) < 0
            else int(args.candidate_prefilter_topn)
        ),
        require_rs_safe=bool(args.require_rs_safe),
        rs_total_slack=float(args.rs_total_slack),
        rs_h2d_slack=float(args.rs_h2d_slack),
        rs_service_hit_drop_limit=float(args.rs_service_hit_drop_limit),
        rs_prefetch_use_drop_limit=float(args.rs_prefetch_use_drop_limit),
        require_valid_positive=bool(args.require_valid_positive),
        min_tail_headroom_pct=float(args.min_tail_headroom_pct),
        min_comm_gain_ms=float(args.min_comm_gain_ms),
        cache_validation_interval=max(0, int(args.cache_validation_interval)),
        event_aware=bool(args.event_aware),
        event_first=bool(args.event_first),
        require_event_gain_when_latency=bool(args.require_event_gain_when_latency),
    )
    if _is_phase_event_cache_objective_name(cfg.objective):
        alpha = np.asarray(getattr(model, "dispatch_alpha_ms", np.zeros((1, 1))), dtype=np.float64)
        latency_ms = float(args.runtime_latency_ms) if args.runtime_latency_ms is not None else (float(alpha.max()) if alpha.size else 0.0)
        latency_active = latency_ms > 0.0
        if float(cfg.lambda_event) <= 0.0 and latency_active:
            lambda_event = max(float(latency_ms), 1e-9)
        else:
            lambda_event = float(cfg.lambda_event)
        cfg = ReplicatedSCPConfig(
            lambda_prefill=float(cfg.lambda_prefill),
            lambda_decode=float(cfg.lambda_decode),
            extra_replicas_per_layer=int(cfg.extra_replicas_per_layer),
            extra_replicas_per_rank_per_layer=int(cfg.extra_replicas_per_rank_per_layer),
            global_extra_replicas=int(cfg.global_extra_replicas),
            max_replicas_per_expert=int(cfg.max_replicas_per_expert),
            stage=str(cfg.stage),
            objective=str(cfg.objective),
            lambda_event=float(lambda_event),
            lambda_small_packet=float(cfg.lambda_small_packet),
            small_packet_threshold=int(cfg.small_packet_threshold),
            load_slack=float(cfg.load_slack),
            expert_cache_capacity=int(cfg.expert_cache_capacity),
            expert_cache_capacity_by_rank=tuple(int(x) for x in cfg.expert_cache_capacity_by_rank),
            h2d_slack=0.0,
            compute_slack=float(cfg.compute_slack),
            min_benefit=float(cfg.min_benefit),
            min_net_gain_ms=float(cfg.min_net_gain_ms),
            min_rankmax_benefit=float(cfg.min_rankmax_benefit),
            h2d_ms_per_proxy=float(cfg.h2d_ms_per_proxy),
            compute_ms_per_proxy=float(cfg.compute_ms_per_proxy),
            latency_ms_per_event=cfg.latency_ms_per_event,
            decode_rs_gain_weight=float(cfg.decode_rs_gain_weight),
            require_rankmax_gain=True,
            require_h2d_nondegrade=True,
            require_compute_nondegrade=bool(cfg.require_compute_nondegrade),
            require_hotset_nondegrade=True,
            require_inter_node_rankmax_bytes_gain=True,
            allow_same_node_replicas=bool(cfg.allow_same_node_replicas),
            min_inter_node_rankmax_bytes_reduction_pct=float(cfg.min_inter_node_rankmax_bytes_reduction_pct),
            min_inter_node_bytes_reduction_pct=float(cfg.min_inter_node_bytes_reduction_pct),
            hotset_min_load_fraction=float(cfg.hotset_min_load_fraction),
            hotset_slack=0.0,
            rank_compute_time_scale=cfg.rank_compute_time_scale,
            rank_h2d_time_scale=cfg.rank_h2d_time_scale,
            disable_h2d_gate=False,
            cache_hit_drop_limit=0.0,
            candidate_prefilter_topn=int(cfg.candidate_prefilter_topn),
            require_rs_safe=bool(cfg.require_rs_safe),
            rs_total_slack=float(cfg.rs_total_slack),
            rs_h2d_slack=float(cfg.rs_h2d_slack),
            rs_service_hit_drop_limit=float(cfg.rs_service_hit_drop_limit),
            rs_prefetch_use_drop_limit=float(cfg.rs_prefetch_use_drop_limit),
            require_valid_positive=bool(cfg.require_valid_positive),
            min_tail_headroom_pct=float(cfg.min_tail_headroom_pct),
            min_comm_gain_ms=float(cfg.min_comm_gain_ms),
            cache_validation_interval=max(1, int(cfg.cache_validation_interval)),
            event_aware=True,
            event_first=bool(latency_active),
            require_event_gain_when_latency=bool(latency_active),
        )
    if str(cfg.stage).lower() == "prefill":
        A = prefill
    elif str(cfg.stage).lower() == "decode":
        A = decode
    else:
        A = cfg.lambda_prefill * prefill + cfg.lambda_decode * decode
    if _is_task_local_objective_name(cfg.objective) or _is_task_local_diagnostic_objective_name(cfg.objective):
        if load_profile_with_trace is None:
            raise RuntimeError(f"{cfg.objective} requires scripts/scp_cache_replay.py")
        trace_profile = load_profile_with_trace(profile_path)
        trace_profile = select_trace_profile_num_experts(trace_profile, int(args.num_experts))
        valid_trace_profile = None
        if valid_profile_path_raw:
            valid_trace_profile = load_profile_with_trace(Path(valid_profile_path_raw).expanduser())
            valid_trace_profile = select_trace_profile_num_experts(valid_trace_profile, int(args.num_experts))
        if _is_task_local_diagnostic_objective_name(cfg.objective):
            result = solve_tasklocal_diagnostic_replicated_v1(
                A,
                model,
                cfg,
                topology=topology,
                trace_profile=trace_profile,
                valid_trace_profile=valid_trace_profile,
                profile_meta=meta,
                valid_profile_meta=valid_meta,
            )
        else:
            result = solve_task_local_tail_scp_v1(
                A,
                model,
                cfg,
                topology=topology,
                trace_profile=trace_profile,
                valid_trace_profile=valid_trace_profile,
                profile_meta=meta,
                valid_profile_meta=valid_meta,
            )
    elif _is_online_aware_objective_name(cfg.objective):
        if load_profile_with_trace is None:
            raise RuntimeError("online_aware_scp_v1 requires scripts/scp_cache_replay.py")
        trace_profile = load_profile_with_trace(profile_path)
        trace_profile = select_trace_profile_num_experts(trace_profile, int(args.num_experts))
        valid_trace_profile = None
        if valid_profile_path_raw:
            valid_trace_profile = load_profile_with_trace(Path(valid_profile_path_raw).expanduser())
            valid_trace_profile = select_trace_profile_num_experts(valid_trace_profile, int(args.num_experts))
        result = solve_online_aware_scp_v1(
            A,
            model,
            cfg,
            topology=topology,
            trace_profile=trace_profile,
            valid_trace_profile=valid_trace_profile,
            profile_meta=meta,
            valid_profile_meta=valid_meta,
        )
    elif _is_tail_headroom_objective_name(cfg.objective):
        if load_profile_with_trace is None:
            raise RuntimeError("tail_headroom_scp_v2 requires scripts/scp_cache_replay.py")
        trace_profile = load_profile_with_trace(profile_path)
        trace_profile = select_trace_profile_num_experts(trace_profile, int(args.num_experts))
        valid_trace_profile = None
        if valid_profile_path_raw:
            valid_trace_profile = load_profile_with_trace(Path(valid_profile_path_raw).expanduser())
            valid_trace_profile = select_trace_profile_num_experts(valid_trace_profile, int(args.num_experts))
        result = solve_tail_headroom_scp_v2(
            A,
            model,
            cfg,
            topology=topology,
            trace_profile=trace_profile,
            valid_trace_profile=valid_trace_profile,
            profile_meta=meta,
            valid_profile_meta=valid_meta,
        )
    elif _is_cacheaware_objective_name(cfg.objective):
        if load_profile_with_trace is None:
            raise RuntimeError("cacheaware_ranktail_v1 requires scripts/scp_cache_replay.py")
        trace_profile = load_profile_with_trace(profile_path)
        trace_profile = select_trace_profile_num_experts(trace_profile, int(args.num_experts))
        valid_trace_profile = None
        if valid_profile_path_raw:
            valid_trace_profile = load_profile_with_trace(Path(valid_profile_path_raw).expanduser())
            valid_trace_profile = select_trace_profile_num_experts(valid_trace_profile, int(args.num_experts))
        result = solve_cacheaware_ranktail(
            A,
            model,
            cfg,
            topology=topology,
            trace_profile=trace_profile,
            valid_trace_profile=valid_trace_profile,
            profile_meta=meta,
            valid_profile_meta=valid_meta,
        )
    else:
        result = solve(A, model, cfg, A_prefill_choice=prefill, A_decode_choice=decode)
        result["topology_fingerprint"] = _topology_fingerprint(topology)
        result["profile_metadata"] = meta
        if valid_profile_path_raw:
            if valid_prefill is None or valid_decode is None:
                raise RuntimeError("internal error: valid profile was not loaded")
            if str(cfg.stage).lower() == "prefill":
                valid_A = valid_prefill
            elif str(cfg.stage).lower() == "decode":
                valid_A = valid_decode
            else:
                valid_A = cfg.lambda_prefill * valid_prefill + cfg.lambda_decode * valid_decode
            valid_result = solve(valid_A, model, cfg, A_prefill_choice=valid_prefill, A_decode_choice=valid_decode)
            valid_metrics = valid_result.get("offline_metrics", {})
            train_metrics = result.get("offline_metrics", {})
            result["valid_profile_metadata"] = valid_meta
            result["valid_offline_metrics"] = valid_metrics
            result["offline_metrics"]["valid_rank_runtime_proxy_reduction_pct"] = valid_metrics.get("rank_runtime_proxy_reduction_pct")
            result["offline_metrics"]["overfit_warning"] = bool(
                float(train_metrics.get("rank_runtime_proxy_reduction_pct", 0.0) or 0.0) > 0.0
                and float(valid_metrics.get("rank_runtime_proxy_reduction_pct", 0.0) or 0.0) < 0.0
            )

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    report = out.with_name("replicated_scp_ep_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write(report_markdown(result))
    metrics = out.with_name("replicated_scp_ep_offline_metrics.json")
    with open(metrics, "w", encoding="utf-8") as f:
        json.dump(result["offline_metrics"], f, indent=2, sort_keys=True)

    m = result["offline_metrics"]
    print(f"[replicated-scp-ep] wrote {out}")
    print(f"[replicated-scp-ep] stage={cfg.stage} objective={cfg.objective} global_extra_replicas={cfg.global_extra_replicas}")
    print(
        f"[replicated-scp-ep] objective_reduction_pct={m.get('objective_reduction_pct', m.get('rank_runtime_proxy_reduction_pct', 0.0)):.2f} "
        f"comm_reduction_pct={m.get('comm_reduction_pct', m.get('rankmax_comm_reduction_pct', 0.0)):.2f} "
        f"cross_node_reduction_pct={m['cross_node_reduction_pct']:.2f} "
        f"cross_node_bytes_reduction_pct={m['cross_node_bytes_reduction_pct']:.2f} "
        f"rankmax_inter_node_bytes_reduction_pct={m.get('rankmax_inter_node_bytes_reduction_pct', 0.0):.2f} "
        f"inter_node_dispatch_bytes_reduction_pct={m.get('inter_node_dispatch_bytes_reduction_pct', 0.0):.2f} "
        f"replicas={m['num_extra_replicas']}"
    )
    print(
        f"[replicated-scp-ep] selected_cross_node_replicas={m['selected_cross_node_replicas']} "
        f"selected_same_node_replicas={m['selected_same_node_replicas']} "
        f"should_run_runtime={m['should_run_runtime']} noop_reason={m['noop_reason']}"
    )
    print(
        f"[replicated-scp-ep] dispatch_item_bytes={m['dispatch_item_bytes']:.1f} "
        f"combine_item_bytes={m['combine_item_bytes']:.1f} "
        f"bytes_source={m.get('bytes_source', result.get('communication_model', {}).get('bytes_source', 'unknown'))}"
    )


if __name__ == "__main__":
    main()
