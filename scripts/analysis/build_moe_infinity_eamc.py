#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _metadata(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "metadata_json" not in data.files:
        return {}
    try:
        return json.loads(str(np.asarray(data["metadata_json"]).item()))
    except Exception:
        return {}


def _infer_dims(data: np.lib.npyio.NpzFile, meta: Dict[str, Any]) -> Tuple[int, int]:
    num_layers = int(meta.get("num_layers", 0) or 0)
    num_experts = int(meta.get("num_experts", 0) or 0)
    if num_layers <= 0 and "A_decode" in data.files:
        num_layers = int(np.asarray(data["A_decode"]).shape[0])
    if num_experts <= 0 and "A_decode" in data.files:
        num_experts = int(np.asarray(data["A_decode"]).shape[-1])
    if num_layers <= 0 and "trace_layer" in data.files and np.asarray(data["trace_layer"]).size:
        num_layers = int(np.asarray(data["trace_layer"]).max()) + 1
    if num_experts <= 0 and "trace_expert" in data.files and np.asarray(data["trace_expert"]).size:
        num_experts = int(np.asarray(data["trace_expert"]).max()) + 1
    if num_layers <= 0 or num_experts <= 0:
        raise RuntimeError("cannot infer num_layers/num_experts from grouped profile")
    return num_layers, num_experts


def _request_keys(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "trace_request_global_id" in data.files:
        return np.asarray(data["trace_request_global_id"], dtype=np.int64)
    shard = np.asarray(data["trace_shard_id"], dtype=np.int64) if "trace_shard_id" in data.files else None
    req = np.asarray(data["trace_request_id"], dtype=np.int64) if "trace_request_id" in data.files else None
    call = np.asarray(data["trace_call_id"], dtype=np.int64) if "trace_call_id" in data.files else None
    n = int(np.asarray(data["trace_expert"]).shape[0])
    if req is None:
        req = np.zeros((n,), dtype=np.int64)
    if shard is None:
        shard = np.zeros((n,), dtype=np.int64)
    if call is None:
        call = np.zeros((n,), dtype=np.int64)
    return shard * 1_000_000_000_000 + req * 1_000_000 + call


def _build_request_eams(
    grouped_profile: Path,
    *,
    stage: str,
    max_requests: int,
    min_events: int,
    num_clusters: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = np.load(grouped_profile, allow_pickle=True)
    required = {"trace_stage", "trace_layer", "trace_expert"}
    missing = sorted(required - set(data.files))
    if missing:
        raise RuntimeError(f"grouped profile lacks ordered trace arrays: {missing}")
    meta = _metadata(data)
    num_layers, num_experts = _infer_dims(data, meta)

    stage_id = 1 if str(stage).lower() == "decode" else 0
    trace_stage = np.asarray(data["trace_stage"], dtype=np.int64)
    layer = np.asarray(data["trace_layer"], dtype=np.int64)
    expert = np.asarray(data["trace_expert"], dtype=np.int64)
    count = (
        np.asarray(data["trace_count"], dtype=np.float32)
        if "trace_count" in data.files
        else np.ones_like(expert, dtype=np.float32)
    )
    req_key = _request_keys(data)

    mask = (
        (trace_stage == int(stage_id))
        & (layer >= 0)
        & (layer < int(num_layers))
        & (expert >= 0)
        & (expert < int(num_experts))
        & (count > 0)
    )
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise RuntimeError(f"no trace events for stage={stage}")

    order = np.argsort(req_key[idx], kind="stable")
    idx = idx[order]
    keys = req_key[idx]
    unique_keys, starts = np.unique(keys, return_index=True)
    if max_requests > 0:
        unique_keys = unique_keys[: int(max_requests)]
        starts = starts[: int(max_requests)]

    matrices: List[np.ndarray] = []
    event_counts: List[int] = []
    for pos, key in enumerate(unique_keys.tolist()):
        start = int(starts[pos])
        end = int(starts[pos + 1]) if pos + 1 < len(starts) else int(idx.shape[0])
        rows = idx[start:end]
        if int(rows.shape[0]) < int(min_events):
            continue
        mat = np.zeros((int(num_layers), int(num_experts)), dtype=np.float32)
        np.add.at(mat, (layer[rows], expert[rows]), count[rows].astype(np.float32, copy=False))
        if float(mat.sum()) <= 0.0:
            continue
        matrices.append(mat)
        event_counts.append(int(rows.shape[0]))

    if not matrices:
        raise RuntimeError("no request EAMs survived min_events filter")
    eams = np.stack(matrices, axis=0).astype(np.float32, copy=False)
    info = {
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "stage": str(stage),
        "request_eams": int(eams.shape[0]),
        "num_clusters_requested": int(num_clusters),
        "events_per_request_mean": float(np.mean(event_counts)),
        "events_per_request_min": int(np.min(event_counts)),
        "events_per_request_max": int(np.max(event_counts)),
    }
    return eams, info


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def _kmeans_medoids(eams: np.ndarray, k: int, seed: int, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(eams.shape[0])
    k = max(1, min(int(k), n))
    flat = _l2_normalize_rows(eams.reshape(n, -1).astype(np.float32, copy=False))
    rng = np.random.default_rng(int(seed))
    init = rng.choice(n, size=k, replace=False)
    centers = flat[init].copy()
    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(max(1, int(iters))):
        sim = flat @ centers.T
        new_labels = np.argmax(sim, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cid in range(k):
            members = flat[labels == cid]
            if members.shape[0] == 0:
                centers[cid] = flat[int(rng.integers(0, n))]
                continue
            center = members.mean(axis=0)
            norm = float(np.linalg.norm(center))
            centers[cid] = center / max(norm, 1e-12)

    medoids: List[np.ndarray] = []
    for cid in range(k):
        member_ids = np.nonzero(labels == cid)[0]
        if member_ids.size == 0:
            continue
        sim = flat[member_ids] @ centers[cid]
        medoids.append(eams[int(member_ids[int(np.argmax(sim))])])
    if not medoids:
        medoids = [eams[0]]
    return np.stack(medoids, axis=0).astype(np.float32, copy=False), labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MoE-Infinity EAMC cluster representatives from grouped routing traces.")
    ap.add_argument("--grouped-profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--num-clusters", type=int, default=16)
    ap.add_argument("--max-requests", type=int, default=4096)
    ap.add_argument("--min-events", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    eams, info = _build_request_eams(
        Path(args.grouped_profile),
        stage=str(args.stage),
        max_requests=int(args.max_requests),
        min_events=int(args.min_events),
        num_clusters=int(args.num_clusters),
    )
    eamc, labels = _kmeans_medoids(eams, int(args.num_clusters), int(args.seed), int(args.iters))
    hist = np.bincount(labels, minlength=int(eamc.shape[0])).astype(np.int64)
    obj: Dict[str, Any] = {
        "status": "ok",
        "method": "moe_infinity_eamc_kmeans_medoids",
        "source_profile": str(args.grouped_profile),
        "source": "request_level_expert_activation_matrix",
        "distance": "cosine",
        "eam_semantics": "layers x experts activation-count matrix per request/sequence",
        "eamc_semantics": "K-Means clusters represented by nearest request EAM medoids",
        "num_clusters": int(eamc.shape[0]),
        "cluster_size_hist": [int(x) for x in hist[: int(eamc.shape[0])].tolist()],
        **info,
        "eamc": eamc.tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[moe-infinity-eamc] wrote {out} centers={int(eamc.shape[0])} request_eams={int(eams.shape[0])}")


if __name__ == "__main__":
    main()
