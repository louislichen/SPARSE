#!/usr/bin/env python3
"""Paper-faithful static Prism/DanceMoE expert placement reimplementation.

This baseline implements the static placement part described for Prism /
DanceMoE-style distributed edge MoE inference when no official implementation
is available in this repository:

1. Build per-node, per-layer expert activation frequencies from a routing trace.
2. Allocate per-node, per-layer expert slots according to activation entropy and
   node memory capacity.
3. Select local experts by node-local activation frequency.
4. Repair coverage under our runtime contract by keeping one round-robin owner
   copy for every logical expert, then fill remaining replica budget with the
   highest-frequency local misses.

The output format is the same replicated_scp_ep_v2 placement consumed by the
SPARSE runtime.  This is intentionally a placement-only baseline: router
semantics, RS, and EPIPE are not modified by this solver.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _load_profile(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        prefill = np.asarray(data["A_prefill"], dtype=np.float64)
        decode = np.asarray(data["A_decode"], dtype=np.float64)
        raw_meta = data.get("metadata_json")
    meta: Dict[str, Any] = {}
    if raw_meta is not None:
        try:
            meta = json.loads(str(raw_meta.item()))
        except Exception:
            meta = {}
    if prefill.shape != decode.shape or prefill.ndim != 3:
        raise ValueError(f"profile must contain A_prefill/A_decode [layers,ranks,experts], got {prefill.shape}/{decode.shape}")
    return prefill, decode, meta


def _select_num_experts(
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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stage_demand(prefill: np.ndarray, decode: np.ndarray, stage: str, lambda_prefill: float, lambda_decode: float) -> np.ndarray:
    st = str(stage or "mixed").lower()
    if st == "prefill":
        return np.asarray(prefill, dtype=np.float64)
    if st == "decode":
        return np.asarray(decode, dtype=np.float64)
    if st == "mixed":
        return float(lambda_prefill) * np.asarray(prefill, dtype=np.float64) + float(lambda_decode) * np.asarray(decode, dtype=np.float64)
    raise ValueError(f"stage must be prefill|decode|mixed, got {stage}")


def _rank_to_node(topology: Mapping[str, Any], world: int) -> Dict[int, int]:
    raw = topology.get("rank_to_node", {}) or {}
    return {r: int(raw.get(str(r), raw.get(r, r))) for r in range(world)}


def _node_count(rank_to_node: Mapping[int, int], world: int) -> int:
    return max(max((int(v) for v in rank_to_node.values()), default=world - 1) + 1, 1)


def _node_ranks(rank_to_node: Mapping[int, int], world: int) -> Dict[int, List[int]]:
    nodes = _node_count(rank_to_node, world)
    return {n: [r for r in range(world) if int(rank_to_node.get(r, r)) == n] for n in range(nodes)}


def _rank_scales(topology: Mapping[str, Any], key: str, world: int) -> np.ndarray:
    raw = topology.get(key, None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= world:
        vals = [float(raw[i]) if float(raw[i]) > 0 else 1.0 for i in range(world)]
    else:
        vals = [1.0 for _ in range(world)]
    return np.asarray(vals, dtype=np.float64)


def _parse_node_capacity(raw: str, node_ranks: Mapping[int, Sequence[int]], world: int) -> np.ndarray:
    nodes = len(node_ranks)
    if raw:
        vals = [float(x) for x in raw.split(",") if x.strip()]
        if len(vals) != nodes:
            raise ValueError(f"--node-capacity-weights expects {nodes} comma-separated values, got {raw}")
        arr = np.asarray(vals, dtype=np.float64)
    else:
        arr = np.asarray([max(len(node_ranks.get(n, [])), 1) for n in range(nodes)], dtype=np.float64)
    arr = np.maximum(arr, 1e-9)
    return arr / arr.sum() * float(nodes)


def _node_frequency(demand: np.ndarray, rank_to_node: Mapping[int, int]) -> np.ndarray:
    layers, world, experts = demand.shape
    nodes = _node_count(rank_to_node, world)
    out = np.zeros((layers, nodes, experts), dtype=np.float64)
    for r in range(world):
        out[:, int(rank_to_node.get(r, r)), :] += demand[:, r, :]
    return out


def _entropy(freq: np.ndarray) -> np.ndarray:
    layers, nodes, experts = freq.shape
    out = np.zeros((layers, nodes), dtype=np.float64)
    denom = math.log(max(experts, 2))
    for l in range(layers):
        for n in range(nodes):
            total = float(freq[l, n].sum())
            if total <= 0.0:
                out[l, n] = 0.0
                continue
            p = freq[l, n] / total
            nz = p[p > 0.0]
            out[l, n] = float(-(nz * np.log(nz)).sum() / denom)
    return out


def _comm_cost_matrix(topology: Mapping[str, Any], rank_to_node: Mapping[int, int], world: int, item_bytes: float) -> np.ndarray:
    cost_cfg = topology.get("cost", {}) or {}
    same_rank = float(cost_cfg.get("same_rank", 0.0))
    same_node = float(cost_cfg.get("same_node", 1.0))
    cross_node = float(cost_cfg.get("cross_node", max(same_node, 1.0)))
    cost = np.zeros((world, world), dtype=np.float64)
    for s in range(world):
        for r in range(world):
            if s == r:
                cost[s, r] = same_rank
            elif int(rank_to_node.get(s, s)) == int(rank_to_node.get(r, r)):
                cost[s, r] = same_node
            else:
                cost[s, r] = cross_node
    return cost


def _base_replicas(layers: int, experts: int, world: int) -> Dict[str, Dict[str, List[int]]]:
    return {str(l): {str(e): [int(e % world)] for e in range(experts)} for l in range(layers)}


def _has_local_replica(replicas: Mapping[str, Mapping[str, Sequence[int]]], l: int, e: int, node: int, rank_to_node: Mapping[int, int]) -> bool:
    return any(int(rank_to_node.get(int(r), int(r))) == int(node) for r in replicas[str(l)][str(e)])


def _pick_local_rank(
    ranks: Sequence[int],
    *,
    rank_load: np.ndarray,
    compute_scale: np.ndarray,
) -> int:
    if not ranks:
        raise ValueError("node has no executable ranks")
    scores = [(float(rank_load[int(r)] / max(float(compute_scale[int(r)]), 1e-9)), int(r)) for r in ranks]
    scores.sort()
    return scores[0][1]


def _allocate_extra_slots(
    *,
    freq: np.ndarray,
    entropy: np.ndarray,
    base_local_count: np.ndarray,
    node_capacity: np.ndarray,
    global_extra_replicas: int,
    entropy_floor: float,
    activity_weight: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    layers, nodes, experts = freq.shape
    max_extra = np.maximum(experts - base_local_count, 0)
    activity = freq.sum(axis=2)
    if activity.max(initial=0.0) > 0:
        activity_norm = activity / max(float(activity.mean()), 1e-9)
    else:
        activity_norm = np.ones_like(activity)
    scores = np.zeros((layers, nodes), dtype=np.float64)
    for l in range(layers):
        for n in range(nodes):
            scores[l, n] = float(node_capacity[n]) * (float(entropy[l, n]) + float(entropy_floor)) * (float(activity_norm[l, n]) ** float(activity_weight))
            if max_extra[l, n] <= 0:
                scores[l, n] = 0.0
    total_slots = int(min(max(0, int(global_extra_replicas)), int(max_extra.sum())))
    if total_slots <= 0 or scores.sum() <= 0.0:
        return np.zeros((layers, nodes), dtype=np.int64), {
            "total_extra_slots": 0,
            "score_sum": float(scores.sum()),
            "entropy_mean": float(entropy.mean()),
        }

    raw = scores / scores.sum() * float(total_slots)
    alloc = np.floor(raw).astype(np.int64)
    alloc = np.minimum(alloc, max_extra.astype(np.int64))
    remaining = total_slots - int(alloc.sum())
    remainders: List[Tuple[float, int, int]] = []
    for l in range(layers):
        for n in range(nodes):
            if alloc[l, n] < max_extra[l, n]:
                remainders.append((float(raw[l, n] - math.floor(raw[l, n])), int(l), int(n)))
    remainders.sort(key=lambda x: (-x[0], x[1], x[2]))
    i = 0
    while remaining > 0 and remainders:
        _, l, n = remainders[i % len(remainders)]
        if alloc[l, n] < max_extra[l, n]:
            alloc[l, n] += 1
            remaining -= 1
        i += 1
        if i > len(remainders) * (total_slots + 1):
            break
    return alloc, {
        "total_extra_slots": int(alloc.sum()),
        "score_sum": float(scores.sum()),
        "entropy_mean": float(entropy.mean()),
        "entropy_floor": float(entropy_floor),
        "activity_weight": float(activity_weight),
    }


def _route_tables(
    demand: np.ndarray,
    replicas: Mapping[str, Mapping[str, Sequence[int]]],
    rank_to_node: Mapping[int, int],
    item_cost: np.ndarray,
    compute_scale: np.ndarray,
    local_preferred: bool,
) -> Dict[str, Dict[str, List[int]]]:
    layers, world, experts = demand.shape
    node_ranks = _node_ranks(rank_to_node, world)
    nodes = len(node_ranks)
    rank_load = np.zeros((world,), dtype=np.float64)
    out: Dict[str, Dict[str, List[int]]] = {str(l): {str(n): [int(e % world) for e in range(experts)] for n in range(nodes)} for l in range(layers)}
    hot: List[Tuple[float, int, int]] = []
    for l in range(layers):
        totals = demand[l].sum(axis=0)
        for e in range(experts):
            hot.append((float(totals[e]), int(l), int(e)))
    hot.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, l, e in hot:
        reps = sorted(set(int(r) for r in replicas[str(l)][str(e)]))
        for n in range(nodes):
            srcs = node_ranks.get(n, [])
            local_reps = [r for r in reps if int(rank_to_node.get(r, r)) == n]
            candidates = local_reps if (local_preferred and local_reps) else reps
            best: Tuple[float, int] | None = None
            for r in candidates:
                comm = sum(float(demand[l, s, e]) * float(item_cost[s, r]) for s in srcs)
                load = float(rank_load[r] / max(float(compute_scale[r]), 1e-9))
                score = comm + 1e-3 * load
                if best is None or score < best[0]:
                    best = (score, int(r))
            if best is None:
                best = (0.0, int(e % world))
            dst = int(best[1])
            out[str(l)][str(n)][int(e)] = dst
            rank_load[dst] += sum(float(demand[l, s, e]) for s in srcs)
    return out


def _route_metrics(demand: np.ndarray, tables: Mapping[str, Mapping[str, Sequence[int]]], rank_to_node: Mapping[int, int], item_bytes: float) -> Dict[str, Any]:
    layers, world, experts = demand.shape
    rank_load = np.zeros((world,), dtype=np.float64)
    source_cross_items = np.zeros((world,), dtype=np.float64)
    source_cross_bytes = np.zeros((world,), dtype=np.float64)
    cross_events = 0
    for l in range(layers):
        for s in range(world):
            node = int(rank_to_node.get(s, s))
            row = tables[str(l)][str(node)]
            for e in range(experts):
                cnt = float(demand[l, s, e])
                if cnt <= 0.0:
                    continue
                r = int(row[e])
                rank_load[r] += cnt
                if int(rank_to_node.get(r, r)) != node:
                    source_cross_items[s] += cnt
                    source_cross_bytes[s] += cnt * float(item_bytes)
                    cross_events += 1
    mean = float(rank_load.mean()) if rank_load.size else 0.0
    return {
        "rank_load": rank_load.astype(float).tolist(),
        "rankmax_load": float(rank_load.max(initial=0.0)),
        "rankmean_load": mean,
        "load_imbalance_ratio": float(rank_load.max(initial=0.0) / max(mean, 1e-9)),
        "cross_node_items": float(source_cross_items.sum()),
        "cross_node_bytes": float(source_cross_bytes.sum()),
        "rankmax_cross_node_bytes": float(source_cross_bytes.max(initial=0.0)),
        "active_cross_node_events_proxy": int(cross_events),
        "source_cross_node_bytes": source_cross_bytes.astype(float).tolist(),
    }


def solve_prism_paper_static(
    prefill: np.ndarray,
    decode: np.ndarray,
    *,
    topology: Mapping[str, Any],
    stage: str,
    global_extra_replicas: int,
    max_replicas_per_expert: int,
    lambda_prefill: float,
    lambda_decode: float,
    dispatch_item_bytes: float,
    combine_item_bytes: float,
    node_capacity_weights: str,
    entropy_floor: float,
    activity_weight: float,
    fill_remaining_budget: bool,
) -> Dict[str, Any]:
    demand = _stage_demand(prefill, decode, stage, lambda_prefill, lambda_decode)
    layers, world, experts = demand.shape
    rank_to_node = _rank_to_node(topology, world)
    nodes = _node_count(rank_to_node, world)
    ranks_by_node = _node_ranks(rank_to_node, world)
    compute_scale = _rank_scales(topology, "rank_compute_scale", world)
    node_capacity = _parse_node_capacity(node_capacity_weights, ranks_by_node, world)
    item_bytes = float(dispatch_item_bytes) + float(combine_item_bytes)
    item_cost = _comm_cost_matrix(topology, rank_to_node, world, item_bytes)

    freq = _node_frequency(demand, rank_to_node)
    ent = _entropy(freq)
    replicas = _base_replicas(layers, experts, world)
    rank_load_for_assignment = np.zeros((world,), dtype=np.float64)

    base_local_count = np.zeros((layers, nodes), dtype=np.int64)
    for l in range(layers):
        for n in range(nodes):
            base_local_count[l, n] = sum(1 for e in range(experts) if _has_local_replica(replicas, l, e, n, rank_to_node))

    extra_alloc, alloc_meta = _allocate_extra_slots(
        freq=freq,
        entropy=ent,
        base_local_count=base_local_count,
        node_capacity=node_capacity,
        global_extra_replicas=global_extra_replicas,
        entropy_floor=entropy_floor,
        activity_weight=activity_weight,
    )
    target_local_count = np.minimum(base_local_count + extra_alloc, experts)

    selected: List[Dict[str, Any]] = []
    rejected = {"already_local": 0, "max_replicas_per_expert": 0, "zero_frequency": 0, "no_rank": 0}

    def add_local(l: int, n: int, e: int, reason: str) -> bool:
        if len(selected) >= int(global_extra_replicas):
            return False
        if float(freq[l, n, e]) <= 0.0:
            rejected["zero_frequency"] += 1
            return False
        if _has_local_replica(replicas, l, e, n, rank_to_node):
            rejected["already_local"] += 1
            return False
        cur = set(int(r) for r in replicas[str(l)][str(e)])
        if len(cur) >= int(max_replicas_per_expert):
            rejected["max_replicas_per_expert"] += 1
            return False
        ranks = ranks_by_node.get(n, [])
        if not ranks:
            rejected["no_rank"] += 1
            return False
        r = _pick_local_rank(ranks, rank_load=rank_load_for_assignment, compute_scale=compute_scale)
        cur.add(int(r))
        replicas[str(l)][str(e)] = sorted(cur)
        moved = float(freq[l, n, e])
        rank_load_for_assignment[r] += moved
        selected.append({
            "layer": int(l),
            "expert": int(e),
            "node": int(n),
            "replica_rank": int(r),
            "node_local_frequency": float(freq[l, n, e]),
            "node_layer_entropy": float(ent[l, n]),
            "target_local_count": int(target_local_count[l, n]),
            "reason": reason,
        })
        return True

    # Algorithm 2 style local-frequency assignment under Algorithm 1 slots.
    for l in range(layers):
        for n in range(nodes):
            target = int(target_local_count[l, n])
            order = np.argsort(-freq[l, n]).astype(int).tolist()
            local_now = int(base_local_count[l, n])
            for e in order:
                if local_now >= target or len(selected) >= int(global_extra_replicas):
                    break
                before = len(selected)
                if add_local(l, n, int(e), "entropy_slot_local_top_frequency"):
                    local_now += 1
                elif len(selected) == before and _has_local_replica(replicas, l, int(e), n, rank_to_node):
                    local_now += 1

    if fill_remaining_budget and len(selected) < int(global_extra_replicas):
        fill: List[Tuple[float, int, int, int]] = []
        for l in range(layers):
            for n in range(nodes):
                for e in range(experts):
                    if not _has_local_replica(replicas, l, e, n, rank_to_node):
                        fill.append((float(freq[l, n, e]), int(l), int(n), int(e)))
        fill.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        for _, l, n, e in fill:
            if len(selected) >= int(global_extra_replicas):
                break
            add_local(l, n, e, "coverage_fill_local_top_frequency")

    rr_tables = _route_tables(
        demand,
        _base_replicas(layers, experts, world),
        rank_to_node,
        item_cost,
        compute_scale,
        local_preferred=True,
    )
    final_tables = _route_tables(demand, replicas, rank_to_node, item_cost, compute_scale, local_preferred=True)
    baseline = _route_metrics(demand, rr_tables, rank_to_node, item_bytes)
    final = _route_metrics(demand, final_tables, rank_to_node, item_bytes)

    cross_delta = float(final["cross_node_bytes"] - baseline["cross_node_bytes"])
    rankmax_delta = float(baseline["rankmax_load"] - final["rankmax_load"])
    return {
        "format": "replicated_scp_ep_v2",
        "baseline_name": "prism_paper_static_reimplementation",
        "objective": "entropy_allocated_node_local_frequency_static_expert_placement",
        "implementation_note": (
            "Paper-faithful static placement reimplementation from algorithm descriptions. "
            "Base round-robin owner copies are kept to satisfy this runtime's coverage contract; "
            "extra slots implement Prism/DanceMoE-style entropy allocation and local-frequency assignment."
        ),
        "stage": str(stage),
        "world_size": int(world),
        "num_layers": int(layers),
        "num_experts": int(experts),
        "rank_to_node": {str(k): int(v) for k, v in rank_to_node.items()},
        "source_nodes": list(range(nodes)),
        "replicas": replicas,
        "dest_rank_table": final_tables,
        "dest_rank_table_prefill": final_tables if str(stage).lower() in ("prefill", "mixed") else rr_tables,
        "dest_rank_table_decode": final_tables if str(stage).lower() in ("decode", "mixed") else rr_tables,
        "dispatch_item_bytes": float(dispatch_item_bytes),
        "combine_item_bytes": float(combine_item_bytes),
        "global_extra_replicas": int(global_extra_replicas),
        "max_replicas_per_expert": int(max_replicas_per_expert),
        "lambda_prefill": float(lambda_prefill),
        "lambda_decode": float(lambda_decode),
        "node_capacity_weights": node_capacity.astype(float).tolist(),
        "entropy_allocation": {
            **alloc_meta,
            "base_local_count_mean": float(base_local_count.mean()),
            "extra_alloc_sum": int(extra_alloc.sum()),
            "target_local_count_mean": float(target_local_count.mean()),
        },
        "selected_replicas": selected,
        "chosen_replicas": selected,
        "rejected": rejected,
        "offline_metrics": {
            "baseline": baseline,
            "prism_paper_static": final,
            "cross_node_bytes_delta": cross_delta,
            "cross_node_bytes_delta_pct": float(cross_delta / max(float(baseline["cross_node_bytes"]), 1e-9) * 100.0),
            "rankmax_load_reduction": rankmax_delta,
            "rankmax_load_reduction_pct": float(rankmax_delta / max(float(baseline["rankmax_load"]), 1e-9) * 100.0),
            "remote_event_delta": int(final["active_cross_node_events_proxy"] - baseline["active_cross_node_events_proxy"]),
            "selected_replica_count": int(len(selected)),
            "interpretation": (
                "Negative cross_node_bytes_delta_pct means Prism-static reduced remote traffic. "
                "This baseline is placement-only and does not use SPARSE semantic co-activation, RS, or EPIPE."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve Prism/DanceMoE paper-static placement from a grouped routing profile.")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["prefill", "decode", "mixed"], default="mixed")
    ap.add_argument("--global-extra-replicas", type=int, default=96)
    ap.add_argument("--max-replicas-per-expert", type=int, default=2)
    ap.add_argument("--lambda-prefill", type=float, default=0.2)
    ap.add_argument("--lambda-decode", type=float, default=1.0)
    ap.add_argument("--dispatch-item-bytes", type=float, default=4106.0)
    ap.add_argument("--combine-item-bytes", type=float, default=4096.0)
    ap.add_argument("--node-capacity-weights", default="")
    ap.add_argument("--entropy-floor", type=float, default=0.05)
    ap.add_argument("--activity-weight", type=float, default=1.0)
    ap.add_argument("--fill-remaining-budget", action="store_true", default=True)
    ap.add_argument("--num-experts", type=int, default=0, help="Override routed expert dimension for model-specific placement output.")
    args = ap.parse_args()

    prefill, decode, meta = _load_profile(Path(args.profile).expanduser())
    prefill, decode, meta = _select_num_experts(prefill, decode, meta, int(args.num_experts))
    topology = _load_json(Path(args.topology).expanduser())
    result = solve_prism_paper_static(
        prefill,
        decode,
        topology=topology,
        stage=str(args.stage),
        global_extra_replicas=int(args.global_extra_replicas),
        max_replicas_per_expert=int(args.max_replicas_per_expert),
        lambda_prefill=float(args.lambda_prefill),
        lambda_decode=float(args.lambda_decode),
        dispatch_item_bytes=float(args.dispatch_item_bytes),
        combine_item_bytes=float(args.combine_item_bytes),
        node_capacity_weights=str(args.node_capacity_weights),
        entropy_floor=float(args.entropy_floor),
        activity_weight=float(args.activity_weight),
        fill_remaining_budget=bool(args.fill_remaining_budget),
    )
    result["profile"] = str(args.profile)
    result["profile_num_experts"] = int(meta.get("profile_num_experts", prefill.shape[-1]))
    result["solver_num_experts"] = int(meta.get("solver_num_experts", prefill.shape[-1]))
    result["solver_expert_dim_source"] = str(meta.get("solver_expert_dim_source", "profile_shape"))
    result["profile_metadata"] = meta
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[prism-paper-static] wrote {out}")


if __name__ == "__main__":
    main()
