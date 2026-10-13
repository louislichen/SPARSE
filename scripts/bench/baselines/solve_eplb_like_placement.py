#!/usr/bin/env python3
"""EPLB-like replicated placement baseline.

This baseline is intentionally outside the SPARE runtime.  It consumes the same
SCP routing profile, but optimizes a different objective: reduce rank load
imbalance by routing hot experts to the currently lightest executable rank.

It does not use source-node/task locality in the objective.  The output is a
replicated_scp_ep_v2 placement with explicit source-node tables so the runtime
does not silently choose the nearest replica and turn this into SCP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _rank_to_node(topology: Dict[str, Any], world: int) -> Dict[int, int]:
    raw = topology.get("rank_to_node", {}) or {}
    return {r: int(raw.get(str(r), raw.get(r, r))) for r in range(world)}


def _stage_demand(prefill: np.ndarray, decode: np.ndarray, stage: str, lambda_prefill: float, lambda_decode: float) -> np.ndarray:
    st = str(stage or "mixed").lower()
    if st == "prefill":
        return np.asarray(prefill, dtype=np.float64)
    if st == "decode":
        return np.asarray(decode, dtype=np.float64)
    if st == "mixed":
        return float(lambda_prefill) * np.asarray(prefill, dtype=np.float64) + float(lambda_decode) * np.asarray(decode, dtype=np.float64)
    raise ValueError(f"stage must be prefill|decode|mixed, got {stage}")


def _node_count(rank_to_node: Dict[int, int], world: int) -> int:
    return max(max(rank_to_node.values(), default=world - 1) + 1, 1)


def _route_metrics_for_tables(
    demand: np.ndarray,
    *,
    rank_to_node: Dict[int, int],
    dest_tables: Dict[str, Dict[str, List[int]]],
    dispatch_item_bytes: float,
    combine_item_bytes: float,
) -> Dict[str, Any]:
    layers, world, experts = demand.shape
    nodes = _node_count(rank_to_node, world)
    item_bytes = float(dispatch_item_bytes) + float(combine_item_bytes)
    rank_load = np.zeros((world,), dtype=np.float64)
    source_cross_bytes = np.zeros((world,), dtype=np.float64)
    source_cross_items = np.zeros((world,), dtype=np.float64)
    cross_events = 0
    for l in range(layers):
        layer_tables = dest_tables[str(l)]
        for s in range(world):
            src_node = int(rank_to_node.get(s, s))
            node_key = str(min(max(src_node, 0), nodes - 1))
            row = layer_tables[node_key]
            for e in range(experts):
                cnt = float(demand[l, s, e])
                if cnt <= 0.0:
                    continue
                r = int(row[e])
                rank_load[r] += cnt
                if int(rank_to_node.get(r, r)) != src_node:
                    source_cross_items[s] += cnt
                    source_cross_bytes[s] += cnt * item_bytes
                    cross_events += 1
    avg = float(rank_load.mean()) if rank_load.size else 0.0
    return {
        "rank_load": rank_load.astype(float).tolist(),
        "rankmax_load": float(rank_load.max(initial=0.0)),
        "rankmean_load": float(avg),
        "load_imbalance_ratio": float(rank_load.max(initial=0.0) / max(avg, 1e-9)),
        "cross_node_items": float(source_cross_items.sum()),
        "cross_node_bytes": float(source_cross_bytes.sum()),
        "rankmax_cross_node_bytes": float(source_cross_bytes.max(initial=0.0)),
        "active_cross_node_events_proxy": int(cross_events),
        "source_cross_node_bytes": source_cross_bytes.astype(float).tolist(),
    }


def _make_rr_tables(layers: int, experts: int, rank_to_node: Dict[int, int], world: int) -> Dict[str, Dict[str, List[int]]]:
    nodes = _node_count(rank_to_node, world)
    return {
        str(l): {str(n): [int(e % world) for e in range(experts)] for n in range(nodes)}
        for l in range(layers)
    }


def solve_eplb_like(
    prefill: np.ndarray,
    decode: np.ndarray,
    *,
    topology: Dict[str, Any],
    stage: str,
    global_extra_replicas: int,
    max_replicas_per_expert: int,
    lambda_prefill: float,
    lambda_decode: float,
    dispatch_item_bytes: float,
    combine_item_bytes: float,
) -> Dict[str, Any]:
    demand = _stage_demand(prefill, decode, stage, lambda_prefill, lambda_decode)
    layers, world, experts = demand.shape
    rank_to_node = _rank_to_node(topology, world)
    nodes = _node_count(rank_to_node, world)

    replicas: Dict[str, Dict[str, List[int]]] = {
        str(l): {str(e): [int(e % world)] for e in range(experts)}
        for l in range(layers)
    }
    dest_tables = _make_rr_tables(layers, experts, rank_to_node, world)
    rank_load = np.zeros((world,), dtype=np.float64)
    for l in range(layers):
        for s in range(world):
            for e in range(experts):
                rank_load[int(e % world)] += float(demand[l, s, e])

    candidates: List[Tuple[float, int, int]] = []
    for l in range(layers):
        expert_total = demand[l].sum(axis=0)
        for e in range(experts):
            candidates.append((float(expert_total[e]), int(l), int(e)))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    added = 0
    chosen: List[Dict[str, Any]] = []
    for hot, l, e in candidates:
        if added >= int(global_extra_replicas):
            break
        if hot <= 0.0:
            continue
        owner = int(e % world)
        current_replicas = set(int(x) for x in replicas[str(l)][str(e)])
        if len(current_replicas) >= int(max_replicas_per_expert):
            continue
        target = int(np.argmin(rank_load))
        if target in current_replicas:
            order = np.argsort(rank_load)
            target = -1
            for r in order.tolist():
                if int(r) not in current_replicas:
                    target = int(r)
                    break
            if target < 0:
                continue

        # EPLB-like policy: once a hot expert is replicated, every source node
        # routes that logical expert to the selected rank.  This intentionally
        # models load balance, not locality.  Source-node tables prevent the
        # runtime's nearest-replica resolver from converting this baseline into
        # task-local SCP.
        old_target = owner
        moved_load = 0.0
        for s in range(world):
            moved_load += float(demand[l, s, e])
        rank_load[old_target] -= moved_load
        rank_load[target] += moved_load
        replicas[str(l)][str(e)] = sorted(current_replicas | {int(target)})
        for n in range(nodes):
            dest_tables[str(l)][str(n)][int(e)] = int(target)
        chosen.append(
            {
                "layer": int(l),
                "expert": int(e),
                "owner_rank": int(owner),
                "replica_rank": int(target),
                "expert_total_items": float(hot),
                "rank_load_after": rank_load.astype(float).tolist(),
            }
        )
        added += 1

    rr_tables = _make_rr_tables(layers, experts, rank_to_node, world)
    baseline_metrics = _route_metrics_for_tables(
        demand,
        rank_to_node=rank_to_node,
        dest_tables=rr_tables,
        dispatch_item_bytes=dispatch_item_bytes,
        combine_item_bytes=combine_item_bytes,
    )
    eplb_metrics = _route_metrics_for_tables(
        demand,
        rank_to_node=rank_to_node,
        dest_tables=dest_tables,
        dispatch_item_bytes=dispatch_item_bytes,
        combine_item_bytes=combine_item_bytes,
    )
    rankmax_gain = baseline_metrics["rankmax_load"] - eplb_metrics["rankmax_load"]
    cross_byte_delta = eplb_metrics["cross_node_bytes"] - baseline_metrics["cross_node_bytes"]

    return {
        "format": "replicated_scp_ep_v2",
        "baseline_name": "eplb_like_load_balanced_replication",
        "objective": "rankmax_expert_load_balance_without_source_node_locality",
        "stage": str(stage),
        "world_size": int(world),
        "num_layers": int(layers),
        "num_experts": int(experts),
        "rank_to_node": {str(k): int(v) for k, v in rank_to_node.items()},
        "source_nodes": list(range(nodes)),
        "replicas": replicas,
        "dest_rank_table": dest_tables,
        "dest_rank_table_prefill": dest_tables if str(stage).lower() in ("prefill", "mixed") else rr_tables,
        "dest_rank_table_decode": dest_tables if str(stage).lower() in ("decode", "mixed") else rr_tables,
        "dispatch_item_bytes": float(dispatch_item_bytes),
        "combine_item_bytes": float(combine_item_bytes),
        "global_extra_replicas": int(global_extra_replicas),
        "max_replicas_per_expert": int(max_replicas_per_expert),
        "lambda_prefill": float(lambda_prefill),
        "lambda_decode": float(lambda_decode),
        "chosen_replicas": chosen,
        "offline_metrics": {
            "baseline": baseline_metrics,
            "eplb_like": eplb_metrics,
            "rankmax_load_reduction": float(rankmax_gain),
            "rankmax_load_reduction_pct": float(rankmax_gain / max(float(baseline_metrics["rankmax_load"]), 1e-9) * 100.0),
            "cross_node_bytes_delta": float(cross_byte_delta),
            "cross_node_bytes_delta_pct": float(cross_byte_delta / max(float(baseline_metrics["cross_node_bytes"]), 1e-9) * 100.0),
            "interpretation": "Positive rankmax_load_reduction means EPLB-like load balancing helped. Positive cross_node_bytes_delta means it paid extra communication because locality was not optimized.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve an EPLB-like load-balanced replicated placement from a SPARE routing profile.")
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
    ap.add_argument("--num-experts", type=int, default=0, help="Override routed expert dimension for model-specific placement output.")
    args = ap.parse_args()

    prefill, decode, meta = _load_profile(Path(args.profile).expanduser())
    prefill, decode, meta = _select_num_experts(prefill, decode, meta, int(args.num_experts))
    topology = _load_json(Path(args.topology).expanduser())
    result = solve_eplb_like(
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
    )
    result["profile"] = str(args.profile)
    result["profile_num_experts"] = int(meta.get("profile_num_experts", prefill.shape[-1]))
    result["solver_num_experts"] = int(meta.get("solver_num_experts", prefill.shape[-1]))
    result["solver_expert_dim_source"] = str(meta.get("solver_expert_dim_source", "profile_shape"))
    result["profile_metadata"] = meta
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[eplb-like] wrote {out}")


if __name__ == "__main__":
    main()
