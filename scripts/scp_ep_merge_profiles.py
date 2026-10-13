#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _task_counts(metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
    # The formal C4 profiles do not use the optional task-local extension.
    if not any(md.get("task_local_enabled") or md.get("task_local") for md in metadatas):
        return {}
    try:
        from scripts.task_locality import counts_by_rank_node
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Task-local profile merging requires the optional scripts/task_locality.py; "
            "ordinary C4 profiles do not require this extension."
        ) from exc
    return counts_by_rank_node(metadatas)


def _load_metadata(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    try:
        return json.loads(str(raw.item()))
    except Exception:
        return {}


TRACE_KEYS = (
    "trace_stage",
    "trace_call_id",
    "trace_step",
    "trace_layer",
    "trace_source_rank",
    "trace_expert",
    "trace_count",
    "trace_token_rows",
    "trace_shard_id",
    "trace_request_id",
    "trace_global_order_id",
    "trace_source_node",
    "trace_task_id",
    "trace_task_family",
    "trace_request_global_id",
)

BASE_TRACE_KEYS = (
    "trace_stage",
    "trace_call_id",
    "trace_step",
    "trace_layer",
    "trace_source_rank",
    "trace_expert",
    "trace_count",
    "trace_token_rows",
)

GROUP_TRACE_KEYS = (
    "trace_shard_id",
    "trace_request_id",
    "trace_global_order_id",
)

TASK_TRACE_KEYS = (
    "trace_source_node",
    "trace_task_id",
    "trace_task_family",
    "trace_request_global_id",
)


def _metadata_first(metadata: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in metadata and metadata.get(key) is not None:
            return metadata.get(key)
    return default


def _infer_int_from_path(path: Path, pattern: str, default: int = 0) -> int:
    rx = re.compile(pattern)
    for part in reversed(path.parts):
        match = rx.search(part)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return int(default)
    return int(default)


def _trace_group_defaults(path: Path, metadata: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    shard_id = _metadata_first(
        metadata,
        "trace_shard_id",
        "profile_shard_id",
        "shard_id",
        default=None,
    )
    if shard_id is None:
        shard_id = _infer_int_from_path(path, r"shard[_-]?(\d+)", 0)
    request_id = _metadata_first(
        metadata,
        "trace_request_id",
        "profile_request_id",
        "dataset_skip_valid",
        "DATASET_SKIP_VALID",
        default=None,
    )
    if request_id is None:
        request_id = int(shard_id)
    return {
        "trace_shard_id": np.full((int(n),), int(shard_id), dtype=np.int64),
        "trace_request_id": np.full((int(n),), int(request_id), dtype=np.int64),
        "trace_global_order_id": np.arange(int(n), dtype=np.int64),
    }


def _trace_task_defaults(metadata: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    rank = int(metadata.get("rank", metadata.get("source_rank", 0)) or 0)
    source_node = int(metadata.get("source_node", rank) or rank)
    request_id = int(_metadata_first(metadata, "trace_request_id", "profile_request_id", "dataset_skip_valid", default=0) or 0)
    return {
        "trace_source_node": np.full((int(n),), int(source_node), dtype=np.int16),
        "trace_task_id": np.full((int(n),), -1, dtype=np.int16),
        "trace_task_family": np.full((int(n),), -1, dtype=np.int16),
        "trace_request_global_id": np.full((int(n),), int(request_id), dtype=np.int64),
    }


def _task_string_table(metadata: Dict[str, Any], kind: str) -> Dict[int, str]:
    raw_tables = metadata.get("task_string_table", {}) or {}
    raw = raw_tables.get(str(kind), {}) if isinstance(raw_tables, dict) else {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    return out


def _remap_task_trace_ids(
    arr: np.ndarray,
    *,
    metadata: Dict[str, Any],
    kind: str,
    global_value_to_id: Dict[str, int],
) -> np.ndarray:
    """Map per-rank task ids to a merged global task-id table.

    Each rank interns task strings independently while profiling.  Directly
    concatenating trace_task_id/trace_task_family would therefore alias unrelated
    tasks, e.g. rank0 code=0 and rank2 translation=0.  The merged profile must
    use one global string table.
    """
    arr = np.asarray(arr)
    if int(arr.size) == 0:
        return arr.astype(np.int16, copy=False)
    table = _task_string_table(metadata, kind)
    unique = [int(x) for x in np.unique(arr).tolist()]
    non_default = [x for x in unique if int(x) >= 0]
    if not non_default:
        return arr.astype(np.int16, copy=False)
    if not table:
        # Older non-task-local profiles either have only -1 or already use a
        # global numeric id.  For task-local profiles, absence of the string
        # table is unsafe because it can silently corrupt skew analysis.
        if bool(metadata.get("task_local_enabled", False)):
            raise ValueError(f"task-local profile has non-default {kind} ids but no task_string_table[{kind!r}]")
        return arr.astype(np.int16, copy=False)

    remap: Dict[int, int] = {-1: -1}
    for local_id in non_default:
        if int(local_id) not in table:
            raise ValueError(f"task-local profile {kind} id {local_id} missing from task_string_table")
        value = str(table[int(local_id)])
        if value not in global_value_to_id:
            global_value_to_id[value] = int(len(global_value_to_id))
        remap[int(local_id)] = int(global_value_to_id[value])

    out = np.full(arr.shape, -1, dtype=np.int16)
    for local_id, global_id in remap.items():
        out[arr.astype(np.int64) == int(local_id)] = int(global_id)
    return out


def _summary(
    A_prefill: np.ndarray,
    A_decode: np.ndarray,
    rows_prefill: np.ndarray,
    rows_decode: np.ndarray,
    metadata: Dict[str, Any],
    trace_arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    total_prefill = int(A_prefill.sum())
    total_decode = int(A_decode.sum())
    layer_total = (A_prefill + A_decode).sum(axis=(1, 2)).astype(np.int64)
    expert_total = (A_prefill + A_decode).sum(axis=1).astype(np.int64)
    source_total = (A_prefill + A_decode).sum(axis=2).astype(np.int64)
    trace_count = trace_arrays.get("trace_count", np.zeros((0,), dtype=np.int64))
    trace_stage = trace_arrays.get("trace_stage", np.zeros((0,), dtype=np.int8))
    has_trace = int(trace_count.size) > 0
    trace_prefill = int(trace_count[trace_stage.astype(np.int64) == 0].sum()) if has_trace else 0
    trace_decode = int(trace_count[trace_stage.astype(np.int64) == 1].sum()) if has_trace else 0
    trace_ok = bool((not has_trace) or (trace_prefill == total_prefill and trace_decode == total_decode))
    return {
        "format": "scp_ep_profile_summary_v1",
        "metadata": metadata,
        "has_ordered_trace": bool(has_trace),
        "trace_event_count": int(trace_count.size),
        "trace_prefill_count_sum": int(trace_prefill),
        "trace_decode_count_sum": int(trace_decode),
        "trace_vs_A_consistency_pass": bool(trace_ok),
        "profile_batch": _metadata_first(metadata, "profile_batch", "batch", "BATCH"),
        "profile_prompt_len": _metadata_first(metadata, "profile_prompt_len", "prompt_len", "PROMPT_LEN"),
        "profile_max_new": _metadata_first(metadata, "profile_max_new", "max_new", "MAX_NEW"),
        "top_k": _metadata_first(metadata, "top_k"),
        "world_size": _metadata_first(metadata, "world_size"),
        "num_layers": _metadata_first(metadata, "num_layers"),
        "num_experts": _metadata_first(metadata, "num_experts"),
        "total_prefill_assignments": total_prefill,
        "total_decode_assignments": total_decode,
        "decode_assignment_fraction": float(total_decode / max(total_prefill + total_decode, 1)),
        "prefill_token_rows_by_layer": rows_prefill.astype(np.int64).tolist(),
        "decode_token_rows_by_layer": rows_decode.astype(np.int64).tolist(),
        "per_layer_total_assignments": layer_total.tolist(),
        "per_layer_per_expert_load": expert_total.tolist(),
        "per_layer_per_source_rank_load": source_total.tolist(),
    }


def merge_profiles(input_dir: Path, out_path: Path) -> Dict[str, Any]:
    files = sorted(input_dir.glob("profile.rank*.npz"))
    if not files:
        files = sorted(input_dir.glob("*.rank*.npz"))
    if not files:
        files = sorted(input_dir.glob("**/profile.rank*.npz"))
    if not files:
        files = sorted(input_dir.glob("**/*.rank*.npz"))
    if not files:
        raise FileNotFoundError(f"no profile.rank*.npz files found in {input_dir}")

    A_prefill = None
    A_decode = None
    rows_prefill = None
    rows_decode = None
    metadatas: List[Dict[str, Any]] = []
    trace_parts: Dict[str, List[np.ndarray]] = {k: [] for k in TRACE_KEYS}
    global_task_value_to_id: Dict[str, int] = {}
    global_family_value_to_id: Dict[str, int] = {}

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            p = data["A_prefill"].astype(np.int64)
            d = data["A_decode"].astype(np.int64)
            rp = data.get("prefill_token_rows_by_layer")
            rd = data.get("decode_token_rows_by_layer")
            if rp is None:
                rp = np.zeros(p.shape[:2], dtype=np.int64)
            if rd is None:
                rd = np.zeros(d.shape[:2], dtype=np.int64)
            rp = rp.astype(np.int64)
            rd = rd.astype(np.int64)
            if A_prefill is None:
                A_prefill = np.zeros_like(p, dtype=np.int64)
                A_decode = np.zeros_like(d, dtype=np.int64)
                rows_prefill = np.zeros_like(rp, dtype=np.int64)
                rows_decode = np.zeros_like(rd, dtype=np.int64)
            if p.shape != A_prefill.shape or d.shape != A_decode.shape:
                raise ValueError(f"profile shape mismatch in {path}: {p.shape}/{d.shape}")
            A_prefill += p
            A_decode += d
            rows_prefill += rp
            rows_decode += rd
            metadata = _load_metadata(data)
            if "trace_count" in data:
                missing_base = [key for key in BASE_TRACE_KEYS if key not in data]
                if missing_base:
                    raise ValueError(f"profile {path} has trace_count but misses base trace keys {missing_base}")
                n_trace = int(np.asarray(data["trace_count"]).shape[0])
                for key in BASE_TRACE_KEYS:
                    trace_parts[key].append(np.asarray(data[key]))
                group_defaults = _trace_group_defaults(path, metadata, n_trace)
                for key in GROUP_TRACE_KEYS:
                    if key in data:
                        trace_parts[key].append(np.asarray(data[key]))
                    else:
                        trace_parts[key].append(group_defaults[key])
                task_defaults = _trace_task_defaults(metadata, n_trace)
                for key in TASK_TRACE_KEYS:
                    if key in data:
                        arr = np.asarray(data[key])
                        if key == "trace_task_id":
                            arr = _remap_task_trace_ids(
                                arr,
                                metadata=metadata,
                                kind="task_id",
                                global_value_to_id=global_task_value_to_id,
                            )
                        elif key == "trace_task_family":
                            arr = _remap_task_trace_ids(
                                arr,
                                metadata=metadata,
                                kind="task_family",
                                global_value_to_id=global_family_value_to_id,
                            )
                        trace_parts[key].append(arr)
                    else:
                        trace_parts[key].append(task_defaults[key])
            metadatas.append(metadata)

    assert A_prefill is not None and A_decode is not None
    assert rows_prefill is not None and rows_decode is not None
    base_meta = dict(metadatas[0] if metadatas else {})
    base_meta["merged_rank_files"] = [str(p) for p in files]
    base_meta["rank_count"] = len(files)
    base_meta["total_prefill_assignments"] = int(A_prefill.sum())
    base_meta["total_decode_assignments"] = int(A_decode.sum())
    base_meta["total_prefill_tokens"] = int(rows_prefill.sum())
    base_meta["total_decode_tokens"] = int(rows_decode.sum())
    base_meta["format"] = "scp_ep_routing_profile_merged_v1"
    request_records: List[Dict[str, Any]] = []
    for metadata in metadatas:
        raw_requests = metadata.get("requests", []) if isinstance(metadata, dict) else []
        if not isinstance(raw_requests, list):
            continue
        source_rank = int(metadata.get("source_rank", metadata.get("rank", 0)) or 0)
        source_node = int(metadata.get("source_node", source_rank) or source_rank)
        profile_split = str(metadata.get("profile_split", ""))
        for idx, req in enumerate(raw_requests):
            if not isinstance(req, dict):
                continue
            rec = dict(req)
            rec.setdefault("source_rank", int(source_rank))
            rec.setdefault("source_node", int(source_node))
            rec.setdefault("split", profile_split)
            rec.setdefault("request_local_index", int(idx))
            if "request_global_id" in rec:
                try:
                    rec["request_global_id"] = int(rec["request_global_id"])
                except Exception:
                    pass
            request_records.append(rec)
    if request_records:
        base_meta["request_profile_records"] = request_records
        base_meta["request_profile_record_count"] = int(len(request_records))
    task_counts = _task_counts(metadatas)
    if task_counts.get("sampled_task_counts_by_rank"):
        base_meta.update(task_counts)
        base_meta["task_local_enabled"] = True
        for key in ("task_manifest_path", "node_task_mix", "rank_to_node", "profile_split"):
            for md in metadatas:
                if key in md:
                    base_meta[key] = md[key]
                    break
        base_meta["task_string_table"] = {
            "task_id": {str(idx): value for value, idx in sorted(global_task_value_to_id.items(), key=lambda kv: kv[1])},
            "task_family": {str(idx): value for value, idx in sorted(global_family_value_to_id.items(), key=lambda kv: kv[1])},
        }
    trace_arrays: Dict[str, np.ndarray] = {}
    for key, parts in trace_parts.items():
        if parts:
            trace_arrays[key] = np.concatenate(parts, axis=0)
        else:
            dtype = np.int8 if key == "trace_stage" else np.int64
            trace_arrays[key] = np.zeros((0,), dtype=dtype)
    if int(trace_arrays["trace_count"].size) > 0:
        order = np.lexsort(
            (
                trace_arrays["trace_global_order_id"].astype(np.int64),
                trace_arrays["trace_source_rank"].astype(np.int64),
                trace_arrays["trace_layer"].astype(np.int64),
                trace_arrays["trace_call_id"].astype(np.int64),
                trace_arrays["trace_stage"].astype(np.int64),
                trace_arrays["trace_request_id"].astype(np.int64),
                trace_arrays["trace_shard_id"].astype(np.int64),
            )
        )
        for key in TRACE_KEYS:
            trace_arrays[key] = trace_arrays[key][order]
        trace_arrays["trace_global_order_id"] = np.arange(int(trace_arrays["trace_count"].size), dtype=np.int64)
        trace_stage = trace_arrays["trace_stage"].astype(np.int64)
        trace_count = trace_arrays["trace_count"].astype(np.int64)
        trace_prefill = int(trace_count[trace_stage == 0].sum())
        trace_decode = int(trace_count[trace_stage == 1].sum())
        if trace_prefill != int(A_prefill.sum()) or trace_decode != int(A_decode.sum()):
            raise ValueError(
                "ordered trace/A mismatch: "
                f"trace_prefill={trace_prefill} A_prefill={int(A_prefill.sum())} "
                f"trace_decode={trace_decode} A_decode={int(A_decode.sum())}"
            )
        base_meta["trace_prefill_count_sum"] = int(trace_prefill)
        base_meta["trace_decode_count_sum"] = int(trace_decode)
        base_meta["trace_group_semantics"] = "(trace_shard_id, trace_request_id, trace_stage, trace_call_id, trace_layer)"
    base_meta["has_ordered_trace"] = bool(int(trace_arrays["trace_count"].size) > 0)
    base_meta["trace_event_count"] = int(trace_arrays["trace_count"].size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_kwargs = {
        "A_prefill": A_prefill,
        "A_decode": A_decode,
        "prefill_token_rows_by_layer": rows_prefill,
        "decode_token_rows_by_layer": rows_decode,
        "metadata_json": np.array(json.dumps(base_meta, indent=2, sort_keys=True)),
    }
    out_kwargs.update(trace_arrays)
    np.savez_compressed(out_path, **out_kwargs)
    summary = _summary(A_prefill, A_decode, rows_prefill, rows_decode, base_meta, trace_arrays)
    summary_name = "profile_summary.json" if out_path.stem == "profile" else f"{out_path.stem}_summary.json"
    summary_path = out_path.with_name(summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return {"out": str(out_path), "summary": str(summary_path), "rank_files": len(files)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-rank SCP-EP routing profiles.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    input_dir = Path(args.input_dir).expanduser()
    out = Path(args.out).expanduser() if args.out else input_dir / "profile.npz"
    result = merge_profiles(input_dir, out)
    print(f"[scp-ep-merge] wrote {result['out']} rank_files={result['rank_files']}")
    print(f"[scp-ep-merge] summary {result['summary']}")


if __name__ == "__main__":
    main()
