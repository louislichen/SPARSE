#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from scripts.node_locality.io import load_layer_records, load_profile_summary
except Exception:
    load_layer_records = None
    load_profile_summary = None


def _write_missing(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "status": "missing",
        "missing_reason": reason,
        "layers": {},
    }
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_from_grouped_npz(path: Path, topk: int) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "A_decode" in data.files:
        arr = np.asarray(data["A_decode"])
        source = "grouped_npz_A_decode"
    elif "A_prefill" in data.files:
        arr = np.asarray(data["A_prefill"])
        source = "grouped_npz_A_prefill"
    else:
        raise RuntimeError(f"grouped profile lacks A_decode/A_prefill: {path}")
    if arr.ndim != 3:
        raise RuntimeError(f"expected grouped profile shape [layers,ranks,experts], got {arr.shape}")
    counts_by_layer = arr.sum(axis=1)
    num_layers, num_experts = int(counts_by_layer.shape[0]), int(counts_by_layer.shape[1])
    layers: Dict[str, Any] = {}
    for layer in range(num_layers):
        counts = np.asarray(counts_by_layer[layer], dtype=np.float64)
        order = np.argsort(-counts)
        top = [int(e) for e in order[: max(1, int(topk))].tolist() if float(counts[int(e)]) > 0]
        nonzero = {str(int(e)): int(float(counts[int(e)])) for e in order.tolist() if float(counts[int(e)]) > 0}
        layers[str(layer)] = {
            "top_experts": top,
            "counts": nonzero,
        }
    return {
        "status": "ok",
        "source_root": str(path),
        "source": source,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "topk_per_layer": int(topk),
        "layers": layers,
        **layers,
    }


def _layer_counts_from_records(root: str, layer: int, num_experts: int, max_records: int) -> np.ndarray | None:
    if load_layer_records is None:
        return None
    try:
        idx, _wt = load_layer_records(root, layer, max_records=max_records)
    except Exception:
        return None
    if idx is None:
        return None
    out = np.zeros((int(num_experts),), dtype=np.int64)
    flat = np.asarray(idx, dtype=np.int64).reshape(-1)
    for eid in flat.tolist():
        if 0 <= int(eid) < int(num_experts):
            out[int(eid)] += 1
    return out


def _build(root: str, topk: int, max_records: int) -> Dict[str, Any]:
    if load_profile_summary is None:
        raise RuntimeError("scripts.node_locality.io is unavailable")
    summary = load_profile_summary(root)
    num_layers = int(summary.num_layers)
    num_experts = int(summary.num_experts)
    layers: Dict[str, Any] = {}
    used_raw = False
    for layer in range(num_layers):
        counts = _layer_counts_from_records(root, layer, num_experts, max_records)
        if counts is not None:
            used_raw = True
        else:
            counts = np.asarray(summary.layer_expert_freq[layer], dtype=np.float64)
        order = np.argsort(-counts)
        top = [int(e) for e in order[: max(1, int(topk))].tolist() if float(counts[int(e)]) > 0]
        nonzero = {str(int(e)): int(float(counts[int(e)])) for e in order.tolist() if float(counts[int(e)]) > 0}
        layers[str(layer)] = {
            "top_experts": top,
            "counts": nonzero,
        }
    return {
        "status": "ok",
        "source_root": str(root),
        "source": "raw_trace" if used_raw else "decode_profile_summary",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "topk_per_layer": int(topk),
        "layers": layers,
        **layers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a static per-layer hot expert profile from routing traces.")
    ap.add_argument("--root", required=True, help="Routing profile root containing summary/ or trace_chunks/.")
    ap.add_argument("--out", default="results/low_control_stable_strategies/profiles/hot_expert_profile.json")
    ap.add_argument("--topk-per-layer", type=int, default=2)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--grouped-profile", default="", help="Merged grouped profile npz with A_decode/A_prefill.")
    args = ap.parse_args()

    out = Path(args.out)
    try:
        if args.grouped_profile:
            obj = _build_from_grouped_npz(Path(args.grouped_profile), args.topk_per_layer)
        else:
            obj = _build(args.root, args.topk_per_layer, args.max_records)
    except Exception as exc:
        _write_missing(out, str(exc))
        print(f"[hot-profile] missing input: {exc}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[hot-profile] wrote {out}")


if __name__ == "__main__":
    main()
