from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import gzip
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.eval_core import (
    CoreBuildConfig,
    DistEnv,
    Metrics,
    RunningStats,
    SparseContext,
    all_reduce_max_,
    all_reduce_sum_,
    barrier,
    build_model_and_tokenizer,
    destroy_dist,
    greedy_decode_profiled,
    init_dist,
    print_routing_stats_if_enabled,
    set_sdpa_mode,
    wall_timer_s,
)
from utils.skvs_attn import GLOBAL_ATTN_STATE, reset_skvs_run_accounting
from utils.runtime.config import ers_config_from_env

_TASK_LOCALITY_HELPERS = None


def _gather_lightweight_stats(env):
    local = [block._light_policy.summary() for block in SparseContext.layer_blocks.values()
             if block._light_policy is not None]
    if not dist.is_initialized():
        return [local]
    values = [None] * env.world
    dist.all_gather_object(values, local)
    return values


def _load_task_locality_helpers():
    """Load TaskLocal helpers lazily so unrelated RR/RS/EPIPE runs stay isolated."""
    global _TASK_LOCALITY_HELPERS
    if _TASK_LOCALITY_HELPERS is not None:
        return _TASK_LOCALITY_HELPERS
    task_locality_path = Path(__file__).resolve().parent / "task_locality.py"
    try:
        from scripts.task_locality import counts_by_rank_node, sample_task_local_prompts
    except ModuleNotFoundError:
        if not task_locality_path.exists():
            raise ModuleNotFoundError(
                "TaskLocal-SCP is enabled but scripts/task_locality.py is missing. "
                f"Expected file: {task_locality_path}. Sync scripts/task_locality.py, "
                "scripts/__init__.py, and the TaskLocal run scripts to the server."
            )
        spec = importlib.util.spec_from_file_location("_spare_task_locality", task_locality_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Cannot load TaskLocal helper from {task_locality_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_spare_task_locality", module)
        spec.loader.exec_module(module)
        counts_by_rank_node = getattr(module, "counts_by_rank_node")
        sample_task_local_prompts = getattr(module, "sample_task_local_prompts")
    _TASK_LOCALITY_HELPERS = (counts_by_rank_node, sample_task_local_prompts)
    return _TASK_LOCALITY_HELPERS


def _counts_task_local_by_rank_node(metas):
    if not metas:
        return {}
    has_task_local = any(
        isinstance(meta, dict) and bool(meta.get("task_local_enabled"))
        for meta in metas
    )
    if not has_task_local:
        return {}
    counts_by_rank_node, _ = _load_task_locality_helpers()
    return counts_by_rank_node(metas)


def _sample_task_local_prompts(**kwargs):
    _, sample_task_local_prompts = _load_task_locality_helpers()
    return sample_task_local_prompts(**kwargs)


def _assert_runtime_alignment() -> None:
    import inspect
    import utils.eval_core as ec
    sigs = inspect.signature(ec.CoreBuildConfig)
    missing = [name for name in ("runtime_mode", "ep_size") if name not in sigs.parameters]
    if missing:
        raise RuntimeError(
            f"Imported stale utils.eval_core from {ec.__file__}. Missing CoreBuildConfig fields: {missing}"
        )


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in ("", "0", "false", "no", "off")


def _node_grouped_mode_flags(mode: str) -> Dict[str, bool]:
    mode = str(mode or "").strip().lower()
    return {
        "tensor": mode in ("tensor", "tensor_pack", "phase_tensor"),
        "triton": mode in ("triton", "triton_pack"),
        "same_node": mode in (
            "buildtime_same_node",
            "buildtime_same_node_packmeta",
            "buildtime_same_node_stage2local",
            "fused_gather_combine",
            "phase_adaptive",
        ),
        "pack_meta": mode in (
            "buildtime_same_node_packmeta",
            "buildtime_same_node_stage2local",
            "fused_gather_combine",
            "phase_adaptive",
        ),
        "stage2_local": mode in (
            "buildtime_same_node_stage2local",
            "fused_gather_combine",
            "phase_adaptive",
        ),
        "identity_combine": mode in ("fused_gather_combine", "phase_adaptive"),
    }


def _sim_env_raw(name: str, legacy_name: str = "", default: str = "") -> str:
    if name in os.environ:
        return str(os.environ.get(name, default))
    if legacy_name and legacy_name in os.environ:
        return str(os.environ.get(legacy_name, default))
    return str(default)


def _parse_sim_node_map(world: int) -> List[int]:
    raw = _sim_env_raw("SPARSE_SIM_NODE_MAP", "SPARSE_LOGICAL_NODE_MAP", "").strip()
    if not raw:
        if _env_flag("SPARSE_SIM_ENABLE_WEAKNET", os.environ.get("SPARSE_LOGICAL_WEAKNET_ENABLE", "0")) and int(world) == 4:
            return [0, 0, 1, 1]
        return [0 for _ in range(int(world))]
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != int(world):
        raise ValueError(f"SPARSE_SIM_NODE_MAP/SPARSE_LOGICAL_NODE_MAP expects {world} entries, got {len(vals)}: {raw}")
    return vals


def _node_float(prefix: str, node_id: int, default: float) -> float:
    raw = os.environ.get(f"SPARSE_SIM_NODE{int(node_id)}_{prefix}", os.environ.get(f"SPARSE_SIM_{prefix}", str(default)))
    try:
        return float(raw)
    except Exception:
        return float(default)


def _sim_config_summary(world: int) -> Dict[str, Any]:
    node_map = _parse_sim_node_map(world)
    nodes = sorted(set(int(x) for x in node_map))
    if "SPARSE_SIM_INTER_NODE_BW_MBPS" in os.environ:
        bw_raw = float(os.environ.get("SPARSE_SIM_INTER_NODE_BW_MBPS", "0") or 0.0)
        bw_mb_s = bw_raw
        bw_source = "SPARSE_SIM_INTER_NODE_BW_MBPS"
        bw_unit = "MB/s"
    else:
        bw_raw = float(os.environ.get("SPARSE_LOGICAL_INTER_NODE_BW_MBPS", "0") or 0.0)
        bw_mb_s = bw_raw / 8.0
        bw_source = "SPARSE_LOGICAL_INTER_NODE_BW_MBPS"
        bw_unit = "Mbit/s legacy input, converted to MB/s"
    hetero_enabled = _env_flag("SPARSE_SIM_ENABLE_HETEROGENEITY", "0")
    return {
        "sim_weaknet_enabled": int(_env_flag("SPARSE_SIM_ENABLE_WEAKNET", os.environ.get("SPARSE_LOGICAL_WEAKNET_ENABLE", "0"))),
        "sim_heterogeneity_enabled": int(hetero_enabled),
        "sim_heterogeneity_sleep": int(hetero_enabled and _env_flag("SPARSE_SIM_HETEROGENEITY_SLEEP", "1")),
        "sim_node_map": node_map,
        "sim_logical_node_count": len(nodes),
        "sim_inter_node_bw_raw": bw_raw,
        "sim_inter_node_bw_mb_s": bw_mb_s,
        "sim_inter_node_bw_source": bw_source,
        "sim_inter_node_bw_unit": bw_unit,
        "sim_inter_node_lat_ms": float(_sim_env_raw("SPARSE_SIM_INTER_NODE_LAT_MS", "SPARSE_LOGICAL_INTER_NODE_LATENCY_MS", "0") or 0.0),
        "sim_node_compute_scale": {str(n): _node_float("COMPUTE_SCALE", n, 1.0) for n in nodes},
        "sim_node_h2d_scale": {str(n): _node_float("H2D_SCALE", n, 1.0) for n in nodes},
        "sim_semantics": "weaknet inter-node cost is injected as sleep when enabled; heterogeneity scales are reflected in simulated timing fields and optional sleep for slower nodes.",
    }


def _build_per_node_breakdown(per_rank: List[Dict[str, Any]], node_map: List[int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in per_rank or []:
        if not isinstance(r, dict):
            continue
        rank = int(r.get("rank", -1))
        node = int(node_map[rank]) if 0 <= rank < len(node_map) else -1
        rec = out.setdefault(str(node), {
            "node": node,
            "ranks": [],
            "activation_inter_node_mb": 0.0,
            "activation_intra_node_mb": 0.0,
            "weaknet_inter_node_transfer_count": 0.0,
            "weaknet_inter_node_item_count": 0.0,
            "weaknet_inter_node_small_packet_count": 0.0,
            "weaknet_inter_node_small_packet_item_count": 0.0,
            "weaknet_inter_node_sim_ms_sum": 0.0,
            "weaknet_inter_node_sim_ms_rankmax": 0.0,
            "demand_h2d_exposed_decode_ms_sum": 0.0,
            "demand_h2d_exposed_decode_ms_rankmax": 0.0,
            "sim_demand_h2d_exposed_decode_ms_sum": 0.0,
            "sim_demand_h2d_exposed_decode_ms_rankmax": 0.0,
            "expert_exec_ms_sum": 0.0,
            "expert_exec_ms_rankmax": 0.0,
            "sim_expert_exec_ms_sum": 0.0,
            "sim_expert_exec_ms_rankmax": 0.0,
            "comm_ms_sum": 0.0,
            "comm_ms_rankmax": 0.0,
        })
        rec["ranks"].append(rank)
        for key in (
            "activation_inter_node_mb",
            "activation_intra_node_mb",
            "weaknet_inter_node_transfer_count",
            "weaknet_inter_node_item_count",
            "weaknet_inter_node_small_packet_count",
            "weaknet_inter_node_small_packet_item_count",
            "weaknet_inter_node_sim_ms",
            "demand_h2d_exposed_decode_ms",
            "sim_demand_h2d_exposed_decode_ms",
            "expert_exec_ms",
            "sim_expert_exec_ms",
            "comm_ms",
        ):
            val = float(r.get(key, 0.0) or 0.0)
            if key == "weaknet_inter_node_sim_ms":
                rec["weaknet_inter_node_sim_ms_sum"] += val
                rec["weaknet_inter_node_sim_ms_rankmax"] = max(rec["weaknet_inter_node_sim_ms_rankmax"], val)
            elif key == "demand_h2d_exposed_decode_ms":
                rec["demand_h2d_exposed_decode_ms_sum"] += val
                rec["demand_h2d_exposed_decode_ms_rankmax"] = max(rec["demand_h2d_exposed_decode_ms_rankmax"], val)
            elif key == "sim_demand_h2d_exposed_decode_ms":
                rec["sim_demand_h2d_exposed_decode_ms_sum"] += val
                rec["sim_demand_h2d_exposed_decode_ms_rankmax"] = max(rec["sim_demand_h2d_exposed_decode_ms_rankmax"], val)
            elif key == "expert_exec_ms":
                rec["expert_exec_ms_sum"] += val
                rec["expert_exec_ms_rankmax"] = max(rec["expert_exec_ms_rankmax"], val)
            elif key == "sim_expert_exec_ms":
                rec["sim_expert_exec_ms_sum"] += val
                rec["sim_expert_exec_ms_rankmax"] = max(rec["sim_expert_exec_ms_rankmax"], val)
            elif key == "comm_ms":
                rec["comm_ms_sum"] += val
                rec["comm_ms_rankmax"] = max(rec["comm_ms_rankmax"], val)
            else:
                rec[key] += val
    for rec in out.values():
        rec["ranks"] = sorted(int(x) for x in rec.get("ranks", []))
    return out


def _merge_counter_dicts(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        src = item.get(key, {})
        if not isinstance(src, dict):
            continue
        for k, v in src.items():
            out[str(k)] = int(out.get(str(k), 0)) + int(v or 0)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _collect_packet_trace(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        trace = item.get("weaknet_packet_trace", [])
        if isinstance(trace, list):
            out.extend(x for x in trace if isinstance(x, dict))
    return out


def _collect_physical_moe_comm_trace(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        parent_rank = int(item.get("rank", -1))
        trace = item.get("physical_moe_comm_trace", [])
        if not isinstance(trace, list):
            continue
        for raw in trace:
            if not isinstance(raw, dict):
                continue
            key = (
                int(raw.get("sample_index", -1)),
                int(raw.get("runtime_call_id", -1)),
                str(raw.get("phase", "none")),
                int(raw.get("layer", -1)),
                int(raw.get("batch_size", 0)),
                int(raw.get("seq_len", 0)),
            )
            entry = grouped.setdefault(key, {
                "sample_index": key[0],
                "runtime_call_id": key[1],
                "phase": key[2],
                "layer": key[3],
                "batch_size": key[4],
                "seq_len": key[5],
                "rank_comm_ms": {},
                "rank_cross_node_activation_bytes": {},
            })
            rank = int(raw.get("rank", parent_rank))
            entry["rank_comm_ms"][str(rank)] = float(raw.get("comm_ms", 0.0) or 0.0)
            entry["rank_cross_node_activation_bytes"][str(rank)] = int(
                raw.get("cross_node_activation_bytes", 0) or 0
            )

    out: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        comm_by_rank = dict(sorted(
            entry.pop("rank_comm_ms").items(),
            key=lambda item: int(item[0]),
        ))
        bytes_by_rank = dict(sorted(
            entry.pop("rank_cross_node_activation_bytes").items(),
            key=lambda item: int(item[0]),
        ))
        values = [float(value) for value in comm_by_rank.values()]
        rankmean = float(np.mean(values)) if values else 0.0
        rankmax = max(values, default=0.0)
        rankmax_rank = (
            int(max(comm_by_rank, key=lambda rank: float(comm_by_rank[rank])))
            if comm_by_rank
            else -1
        )
        out.append({
            **entry,
            "rank_count": int(len(values)),
            "rank_comm_ms": comm_by_rank,
            "rank_cross_node_activation_bytes": bytes_by_rank,
            "rankmean_comm_ms": rankmean,
            "rankmax_comm_ms": rankmax,
            "rankmax_rank": rankmax_rank,
            "rankmax_over_rankmean": rankmax / max(rankmean, 1e-12),
            "cross_node_activation_bytes": int(sum(bytes_by_rank.values())),
        })
    return out


def _physical_moe_comm_summary(trace: List[Dict[str, Any]]) -> Dict[str, float]:
    rankmax = np.asarray(
        [float(row.get("rankmax_comm_ms", 0.0) or 0.0) for row in trace],
        dtype=np.float64,
    )
    rankmean = np.asarray(
        [float(row.get("rankmean_comm_ms", 0.0) or 0.0) for row in trace],
        dtype=np.float64,
    )
    imbalance = np.asarray(
        [float(row.get("rankmax_over_rankmean", 0.0) or 0.0) for row in trace],
        dtype=np.float64,
    )

    def pct(values: np.ndarray, percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values.size else 0.0

    return {
        "physical_moe_comm_sample_count": float(rankmax.size),
        "physical_moe_comm_tail_p50_ms": pct(rankmax, 50.0),
        "physical_moe_comm_tail_p95_ms": pct(rankmax, 95.0),
        "physical_moe_comm_tail_p99_ms": pct(rankmax, 99.0),
        "physical_moe_comm_rankmean_p50_ms": pct(rankmean, 50.0),
        "physical_moe_comm_rankmax_over_mean_p50": pct(imbalance, 50.0),
        "physical_moe_comm_rankmax_over_mean_p95": pct(imbalance, 95.0),
    }


def _reduce_nodepatch_summary(device: torch.device) -> Dict[str, float]:
    local = SparseContext.nodepatch_summary()
    sum_keys = [
        "enabled_calls",
        "tokens_seen",
        "cross_node_tokens",
        "repaired_tokens",
        "replacement_count",
        "event_proxy_before",
        "event_proxy_after",
        "cross_node_items_before",
        "cross_node_items_after",
        "unique_nodes_before_sum",
        "unique_nodes_after_sum",
        "runtime_packet_events_before",
        "runtime_packet_events_after",
        "runtime_packet_events_eliminated",
        "event_aware_skipped_repairs",
        "cache_aware_skipped_repairs",
        "cache_aware_filtered_candidates",
        "packet_emptying_candidate_packets",
        "packet_emptying_feasible_packets",
        "packet_emptying_rejected_packets",
        "utility_loss_logit_sum",
        "utility_loss_prob_sum",
        "max_margin_logit_sum",
        "max_margin_prob_sum",
        "route_cost_before_sum",
        "route_cost_after_sum",
        "route_cost_gain_sum",
        "old_prob_sum",
        "new_prob_sum",
        "prob_delta_sum",
        "rank_gap_sum",
        "candidate_not_prepared_reject_count",
        "candidate_no_comm_gain_reject_count",
        "candidate_scope_reject_count",
        "candidate_duplicate_reject_count",
        "candidate_rank_gap_reject_count",
        "candidate_cache_reject_count",
        "candidate_strict_resident_reject_count",
        "no_replace_position_count",
    ]
    max_keys = ["max_margin_logit_max", "max_margin_prob_max", "rank_gap_max"]
    out: Dict[str, float] = {}
    for key in sum_keys:
        t = torch.tensor([float(local.get(key, 0.0))], device=device, dtype=torch.float64)
        all_reduce_sum_(t)
        out[key] = float(t.item())
    for key in max_keys:
        t = torch.tensor([float(local.get(key, 0.0))], device=device, dtype=torch.float64)
        all_reduce_max_(t)
        out[key] = float(t.item())

    repaired = max(float(out.get("repaired_tokens", 0.0)), 1.0)
    tokens = max(float(out.get("tokens_seen", 0.0)), 1.0)
    cross = max(float(out.get("cross_node_tokens", 0.0)), 1.0)
    event_before = max(float(out.get("event_proxy_before", 0.0)), 1.0)
    item_before = max(float(out.get("cross_node_items_before", 0.0)), 1.0)
    out["cross_node_token_fraction"] = float(out.get("cross_node_tokens", 0.0)) / tokens
    out["repairable_token_fraction"] = float(out.get("repaired_tokens", 0.0)) / tokens
    out["repairable_cross_node_fraction"] = float(out.get("repaired_tokens", 0.0)) / cross
    out["transfer_event_reduction_pct"] = (
        (float(out.get("event_proxy_before", 0.0)) - float(out.get("event_proxy_after", 0.0)))
        / event_before
        * 100.0
    )
    out["cross_node_item_reduction_pct"] = (
        (float(out.get("cross_node_items_before", 0.0)) - float(out.get("cross_node_items_after", 0.0)))
        / item_before
        * 100.0
    )
    out["avg_utility_loss_logit"] = float(out.get("utility_loss_logit_sum", 0.0)) / repaired
    out["avg_utility_loss_prob"] = float(out.get("utility_loss_prob_sum", 0.0)) / repaired
    out["avg_max_margin_logit"] = float(out.get("max_margin_logit_sum", 0.0)) / repaired
    out["avg_max_margin_prob"] = float(out.get("max_margin_prob_sum", 0.0)) / repaired
    out["avg_unique_nodes_before"] = float(out.get("unique_nodes_before_sum", 0.0)) / tokens
    out["avg_unique_nodes_after"] = float(out.get("unique_nodes_after_sum", 0.0)) / tokens
    replacements = max(float(out.get("replacement_count", 0.0)), 1.0)
    route_before = max(float(out.get("route_cost_before_sum", 0.0)), 1.0)
    out["route_cost_reduction_pct"] = (
        (float(out.get("route_cost_before_sum", 0.0)) - float(out.get("route_cost_after_sum", 0.0)))
        / route_before
        * 100.0
    )
    out["avg_route_cost_gain_per_replacement"] = float(out.get("route_cost_gain_sum", 0.0)) / replacements
    out["avg_old_prob"] = float(out.get("old_prob_sum", 0.0)) / replacements
    out["avg_new_prob"] = float(out.get("new_prob_sum", 0.0)) / replacements
    out["avg_prob_delta"] = float(out.get("prob_delta_sum", 0.0)) / replacements
    out["avg_replacement_rank_gap"] = float(out.get("rank_gap_sum", 0.0)) / replacements
    out["candidate_total_reject_count"] = (
        float(out.get("candidate_not_prepared_reject_count", 0.0))
        + float(out.get("candidate_no_comm_gain_reject_count", 0.0))
        + float(out.get("candidate_scope_reject_count", 0.0))
        + float(out.get("candidate_duplicate_reject_count", 0.0))
        + float(out.get("candidate_rank_gap_reject_count", 0.0))
        + float(out.get("candidate_cache_reject_count", 0.0))
        + float(out.get("candidate_strict_resident_reject_count", 0.0))
    )
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        for key in [
            "top_replacement_pairs",
            "top_removed_experts",
            "top_added_experts",
            "top_removed_positions",
            "top_repaired_layers",
        ]:
            out[key] = local.get(key, [])
    return out



# -----------------------------------------------------------------------------
# runtime-state cleanup
# -----------------------------------------------------------------------------
def _clear_all_prefetch_runtime_state() -> None:
    """
    清理所有 layer block 的 pending prefetch 运行态，但不把它们记入 metrics。
    用途：
    1) warmup 之前/之后清理残留状态
    2) 每次正式 repeat 开始前确保状态干净
    """
    for blk in SparseContext.layer_blocks.values():
        if hasattr(blk, "_prefetched_ids_this_group"):
            blk._prefetched_ids_this_group.clear()
        if hasattr(blk, "_prefetch_used_ids_this_group"):
            blk._prefetch_used_ids_this_group.clear()

        if hasattr(blk, "_cache") and blk._cache is not None:
            try:
                blk._cache.unpin_all()
            except Exception:
                pass
            if str(os.environ.get("SPARSE_CLEAR_EXPERT_CACHE_FOR_PERF", "0")).strip().lower() in ("1", "true", "yes", "on"):
                try:
                    clear_fn = getattr(blk._cache, "clear", None)
                    if callable(clear_fn):
                        clear_fn()
                except Exception:
                    pass


def _clear_attention_runtime_state() -> None:
    for attn in SparseContext.layer_attn_modules.values():
        for name, default in [
            ("_skvs_last_keep_pos_global", None),
            ("_skvs_last_full_kv_len", 0),
        ]:
            try:
                setattr(attn, name, default)
            except Exception:
                pass


def _clear_skvs_global_state() -> None:
    """perf 路径下统一清理 skvs 全局统计。"""
    reset_skvs_run_accounting()


def _clear_all_runtime_state_for_perf() -> None:
    """
    perf 路径统一清理入口：
    - 清 request-scoped runtime state，不改 metrics
    - 默认保留 resident expert weights，但重置 RS/prefetch/cache metadata
    - 避免 warmup / 上一轮 repeat 残留状态污染正式测量
    """
    reset_each_repeat = str(os.environ.get("SPARSE_RESET_RUNTIME_STATE_EACH_REPEAT", "1")).strip().lower() not in ("", "0", "false", "no", "off")
    clear_expert_cache = str(os.environ.get("SPARSE_CLEAR_EXPERT_CACHE_EACH_REPEAT", "0")).strip().lower() in ("1", "true", "yes", "on")
    demote_hot = str(os.environ.get("SPARSE_DEMOTE_HOT_CACHE_EACH_REPEAT", "0")).strip().lower() in ("1", "true", "yes", "on")

    if reset_each_repeat and hasattr(SparseContext, "reset_runtime_state_for_new_request"):
        try:
            SparseContext.reset_runtime_state_for_new_request(
                clear_expert_cache=clear_expert_cache,
                clear_cache_metadata=True,
                demote_hot=demote_hot,
            )
        except Exception:
            SparseContext.reset()
    else:
        SparseContext.reset()

    if not reset_each_repeat:
        _clear_all_prefetch_runtime_state()
    _clear_attention_runtime_state()
    _clear_skvs_global_state()

    # perf 路径默认禁用 runtime debug，避免打印和 debug 逻辑干扰测量
    if hasattr(SparseContext, "runtime_debug_state"):
        try:
            SparseContext.runtime_debug_state.enabled = False
            SparseContext.runtime_debug_state.log_owner_plan = False
            SparseContext.runtime_debug_state.log_migration = False
        except Exception:
            pass


def _finalize_all_prefetch_wasted() -> None:
    """greedy_decode_profiled() 之外清理 prefetch 运行态，不重复记账。"""
    for blk in SparseContext.layer_blocks.values():
        if hasattr(blk, "_prefetched_ids_this_group"):
            blk._prefetched_ids_this_group.clear()
        if hasattr(blk, "_prefetch_used_ids_this_group"):
            blk._prefetch_used_ids_this_group.clear()
        if hasattr(blk, "_cache") and blk._cache is not None:
            try:
                blk._cache.unpin_all()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# prompt loader
# -----------------------------------------------------------------------------
def _open_jsonl_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _prompt_text_from_dataset_row(ex: Any) -> str:
    if not isinstance(ex, dict):
        return ""
    for key in ("text", "prompt", "input", "query", "instruction"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    messages = ex.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                content = msg["content"].strip()
                if content:
                    parts.append(content)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _load_prompts_from_jsonl_gz(
    path: str,
    tokenizer,
    batch: int,
    prompt_len: int,
    max_scan_samples: int = 2000,
    skip_valid_prompts: int = 0,
    return_metadata: bool = False,
) -> List[str] | tuple[List[str], List[Dict[str, Any]]]:
    """
    从 jsonl.gz 数据集中读取文本，并筛选出 token 长度足够的样本。
    每条样本只截取前 prompt_len 个 token 作为 prompt。
    """
    prompts: List[str] = []
    request_rows: List[Dict[str, Any]] = []
    skipped_valid = 0
    skip_valid_prompts = max(0, int(skip_valid_prompts))
    scanned_by_path: Dict[str, int] = {}

    dataset_paths = _dataset_prompt_candidate_paths(path)

    for dataset_path in dataset_paths:
        scanned = 0
        with _open_jsonl_maybe_gz(Path(dataset_path)) as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                scanned += 1
                if scanned > max_scan_samples:
                    break

                try:
                    ex = json.loads(line)
                except Exception:
                    continue

                text = _prompt_text_from_dataset_row(ex)
                if not text:
                    continue

                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < prompt_len:
                    continue

                if skipped_valid < skip_valid_prompts:
                    skipped_valid += 1
                    continue

                ids = ids[:prompt_len]
                prompt_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                if not prompt_text:
                    continue

                prompts.append(prompt_text)
                request_rows.append(
                    {
                        "dataset_path": str(dataset_path),
                        "dataset_index": int(line_no),
                        "skip_valid_prompts": int(skip_valid_prompts),
                    }
                )
                if len(prompts) >= batch:
                    break
        scanned_by_path[str(dataset_path)] = int(scanned)
        if len(prompts) >= batch:
            break

    if len(prompts) == 0:
        detail = ", ".join(f"{p}:{n}" for p, n in scanned_by_path.items())
        raise RuntimeError(
            f"No valid prompts with >= {prompt_len} tokens found in dataset candidates "
            f"starting from {path}; scanned={detail}"
        )

    while len(prompts) < batch:
        prompts.append(prompts[-1])
        request_rows.append(dict(request_rows[-1]) if request_rows else {})

    if return_metadata:
        return prompts, request_rows
    return prompts


def _dataset_prompt_candidate_paths(path: str) -> List[str]:
    """Return the primary dataset shard plus optional same-split C4 fallback shards."""
    primary = Path(path)
    raw = str(os.environ.get("DATASET_PROMPT_FALLBACK_SHARDS", "") or "").strip()
    if not raw:
        raw = str(os.environ.get("SPARSE_DATASET_PROMPT_FALLBACK_SHARDS", "32") or "32").strip()
    try:
        max_extra = max(0, int(raw))
    except Exception:
        max_extra = 0
    paths = [str(primary)]
    if max_extra <= 0:
        return paths
    m = re.match(r"^(?P<prefix>.+\.)(?P<idx>\d+)-of-(?P<total>\d+)(?P<suffix>\.json\.gz)$", primary.name)
    if not m:
        return paths
    idx = int(m.group("idx"))
    total = int(m.group("total"))
    width = len(m.group("idx"))
    prefix = m.group("prefix")
    suffix = m.group("suffix")
    for off in range(1, max_extra + 1):
        next_idx = idx + off
        if next_idx >= total:
            break
        candidate = primary.with_name(f"{prefix}{next_idx:0{width}d}-of-{total:0{len(m.group('total'))}d}{suffix}")
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def _rank_specific_dataset_skip(default_skip: int, rank: int, world: int) -> int:
    raw = str(os.environ.get("DATASET_SKIP_VALID_PER_RANK", "") or "").strip()
    if not raw:
        raw = str(os.environ.get("SPARSE_DATASET_SKIP_VALID_PER_RANK", "") or "").strip()
    if raw:
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        if len(parts) != int(world):
            raise ValueError(
                f"DATASET_SKIP_VALID_PER_RANK expects {world} comma-separated values, got {len(parts)}: {raw}"
            )
        return max(0, int(parts[int(rank)]))
    stride_raw = str(os.environ.get("DATASET_SKIP_VALID_RANK_STRIDE", "") or "").strip()
    if not stride_raw:
        stride_raw = str(os.environ.get("SPARSE_DATASET_SKIP_VALID_RANK_STRIDE", "") or "").strip()
    if stride_raw:
        return max(0, int(default_skip) + int(rank) * int(stride_raw))
    return max(0, int(default_skip))


def _task_manifest_path_from_env() -> str:
    return str(os.environ.get("SPARSE_TASK_MANIFEST", os.environ.get("TASK_MANIFEST", "")) or "").strip()


def _task_local_split_from_env(default: str = "test") -> str:
    return str(os.environ.get("SPARSE_TASK_SPLIT", os.environ.get("TASK_SPLIT", default)) or default).strip()


def _balanced_shard_bounds(total: int, parts: int, part_idx: int) -> tuple[int, int]:
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    if part_idx < 0 or part_idx >= parts:
        raise ValueError(f"part_idx out of range: {part_idx} for parts={parts}")
    base = total // parts
    rem = total % parts
    start = part_idx * base + min(part_idx, rem)
    size = base + (1 if part_idx < rem else 0)
    return start, start + size


def _slice_local_dp_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dp_rank: int,
    dp_size: int,
):
    start, end = _balanced_shard_bounds(int(input_ids.shape[0]), int(dp_size), int(dp_rank))
    if end <= start:
        raise RuntimeError(
            f"Empty local DP shard: dp_rank={dp_rank}, dp_size={dp_size}, global_batch={input_ids.shape[0]}"
        )
    local_ids = input_ids[start:end].contiguous()
    local_mask = None if attention_mask is None else attention_mask[start:end].contiguous()
    return local_ids, local_mask


# -----------------------------------------------------------------------------
# light-weight skvs summary
# -----------------------------------------------------------------------------
def _collect_skvs_debug_summary() -> Dict[str, Any]:
    total_tokens = int(GLOBAL_ATTN_STATE.get("forward_total_tokens", 0) or 0)
    kept_tokens = int(GLOBAL_ATTN_STATE.get("forward_kept_tokens", 0) or 0)
    dropped_tokens = int(GLOBAL_ATTN_STATE.get("forward_dropped_tokens", 0) or 0)
    dense_ref_tokens = int(GLOBAL_ATTN_STATE.get("forward_dense_ref_tokens", 0) or 0)
    dense_kept_tokens = int(GLOBAL_ATTN_STATE.get("forward_dense_kept_tokens", 0) or 0)
    dense_dropped_tokens = int(GLOBAL_ATTN_STATE.get("forward_dense_dropped_tokens", 0) or 0)
    keep_ratio = float(kept_tokens) / max(total_tokens, 1)
    drop_ratio = float(dropped_tokens) / max(total_tokens, 1)
    out = {
        "num_prefill_fallback": int(GLOBAL_ATTN_STATE.get("num_prefill_fallback", 0) or 0),
        "num_decode_fallback": int(GLOBAL_ATTN_STATE.get("num_decode_fallback", 0) or 0),
        "num_ratio1_full": int(GLOBAL_ATTN_STATE.get("num_ratio1_full", 0) or 0),
        "num_layer0_full": int(GLOBAL_ATTN_STATE.get("num_layer0_full", 0) or 0),
        "num_local_sparse": int(GLOBAL_ATTN_STATE.get("num_local_sparse", 0) or 0),
        "num_local_full": int(GLOBAL_ATTN_STATE.get("num_local_full", 0) or 0),
        "effective_mode_last": str(GLOBAL_ATTN_STATE.get("effective_mode_last", "unknown")),
        "forward_total_tokens": total_tokens,
        "forward_kept_tokens": kept_tokens,
        "forward_dropped_tokens": dropped_tokens,
        "forward_dense_ref_tokens": dense_ref_tokens,
        "forward_dense_kept_tokens": dense_kept_tokens,
        "forward_dense_dropped_tokens": dense_dropped_tokens,
        "avg_keep_ratio": keep_ratio,
        "avg_drop_ratio": drop_ratio,
        "dense_equiv_keep_ratio": float(dense_kept_tokens) / max(dense_ref_tokens, 1),
        "dense_equiv_drop_ratio": float(dense_dropped_tokens) / max(dense_ref_tokens, 1),
        "forward_dropped_bytes": int(GLOBAL_ATTN_STATE.get("forward_dropped_bytes", 0) or 0),
        "local_selected_cached_tokens": int(GLOBAL_ATTN_STATE.get("forward_local_selected_cached_tokens", 0) or 0),
        "local_selected_protected_tokens": int(GLOBAL_ATTN_STATE.get("forward_local_selected_protected_tokens", 0) or 0),
        "local_selected_unprotected_tokens": int(GLOBAL_ATTN_STATE.get("forward_local_selected_unprotected_tokens", 0) or 0),
        "local_selected_kept_unprotected_tokens": int(GLOBAL_ATTN_STATE.get("forward_local_selected_kept_unprotected_tokens", 0) or 0),
        "last_plan_candidate_total_tokens": int(GLOBAL_ATTN_STATE.get("last_plan_candidate_total_tokens", 0) or 0),
        "last_plan_candidate_kept_tokens": int(GLOBAL_ATTN_STATE.get("last_plan_candidate_kept_tokens", 0) or 0),
        "last_plan_candidate_dropped_tokens": int(GLOBAL_ATTN_STATE.get("last_plan_candidate_dropped_tokens", 0) or 0),
        "last_plan_keep_ratio": float(int(GLOBAL_ATTN_STATE.get("last_plan_candidate_kept_tokens", 0) or 0)) / max(int(GLOBAL_ATTN_STATE.get("last_plan_candidate_total_tokens", 0) or 0), 1),
        "last_plan_drop_ratio": float(int(GLOBAL_ATTN_STATE.get("last_plan_candidate_dropped_tokens", 0) or 0)) / max(int(GLOBAL_ATTN_STATE.get("last_plan_candidate_total_tokens", 0) or 0), 1),
        "last_plan_dense_ref_tokens": int(GLOBAL_ATTN_STATE.get("last_plan_dense_ref_tokens", 0) or 0),
        "last_plan_dense_kept_tokens": int(GLOBAL_ATTN_STATE.get("last_plan_dense_kept_tokens", 0) or 0),
        "last_plan_dense_dropped_tokens": int(GLOBAL_ATTN_STATE.get("last_plan_dense_dropped_tokens", 0) or 0),
        "last_plan_dense_keep_ratio": float(int(GLOBAL_ATTN_STATE.get("last_plan_dense_kept_tokens", 0) or 0)) / max(int(GLOBAL_ATTN_STATE.get("last_plan_dense_ref_tokens", 0) or 0), 1),
        "last_plan_dense_drop_ratio": float(int(GLOBAL_ATTN_STATE.get("last_plan_dense_dropped_tokens", 0) or 0)) / max(int(GLOBAL_ATTN_STATE.get("last_plan_dense_ref_tokens", 0) or 0), 1),
        "global_keep_kv_len": int(GLOBAL_ATTN_STATE.get("global_keep_kv_len", 0) or 0),
        "global_keep_force_dense": bool(GLOBAL_ATTN_STATE.get("global_keep_force_dense", False)),
        "last_guidance_kv_len": int(GLOBAL_ATTN_STATE.get("last_guidance_kv_len", 0) or 0),
        "num_guidance_refresh": int(GLOBAL_ATTN_STATE.get("num_guidance_refresh", 0) or 0),
        "num_guidance_reuse": int(GLOBAL_ATTN_STATE.get("num_guidance_reuse", 0) or 0),
        "num_guidance_start_suppressed": int(GLOBAL_ATTN_STATE.get("num_guidance_start_suppressed", 0) or 0),
        "num_guidance_interval_suppressed": int(GLOBAL_ATTN_STATE.get("num_guidance_interval_suppressed", 0) or 0),
        "last_guidance_refreshed": bool(GLOBAL_ATTN_STATE.get("last_guidance_refreshed", False)),
        "migration_selected_kv_bytes": int(GLOBAL_ATTN_STATE.get("migration_selected_kv_bytes", 0) or 0),
        "migration_full_kv_bytes": int(GLOBAL_ATTN_STATE.get("migration_full_kv_bytes", 0) or 0),
        "migration_saved_ratio": float(GLOBAL_ATTN_STATE.get("migration_saved_ratio", 0.0) or 0.0),
        "migration_tensor_payload_bytes": int(GLOBAL_ATTN_STATE.get("migration_tensor_payload_bytes", 0) or 0),
        "migration_tensor_wire_bytes": int(GLOBAL_ATTN_STATE.get("migration_tensor_wire_bytes", 0) or 0),
        "migration_impl_tensor_count": int(GLOBAL_ATTN_STATE.get("migration_impl_tensor_count", 0) or 0),
        "migration_impl_object_count": int(GLOBAL_ATTN_STATE.get("migration_impl_object_count", 0) or 0),
        "decode_step_idx": int(GLOBAL_ATTN_STATE.get("decode_step_idx", 0) or 0),
    }
    for k, v in GLOBAL_ATTN_STATE.items():
        if str(k).startswith("fallback_"):
            out[str(k)] = int(v or 0)
    return out

# -----------------------------------------------------------------------------
# reduce + save
# -----------------------------------------------------------------------------
def reduce_and_build_perf_record(metrics: Metrics, args, env: DistEnv, num_experts: int, repeat_idx: int) -> Dict[str, Any] | None:
    device = env.device
    runtime_mode = str(getattr(args, "runtime_mode", "baseline"))
    ep_size = int(getattr(args, "ep_size", 0) or env.world)
    dp_size = int(env.world // ep_size)

    def t(x, dtype):
        return torch.tensor([x], device=device, dtype=dtype)

    task_local_enabled = bool(_task_manifest_path_from_env())
    token_count_mode_requested = str(os.environ.get("SPARSE_TOKEN_COUNT_MODE", "auto") or "auto").strip().lower().replace("-", "_")
    if token_count_mode_requested in ("rank0", "ep_rank0", "ep_rank0_only", "rank0_only"):
        token_count_mode = "ep_rank0_only"
    elif token_count_mode_requested in ("all", "all_ranks", "service", "service_all_ranks", "task_local_all_ranks"):
        token_count_mode = "all_ranks"
    elif token_count_mode_requested in ("auto", ""):
        token_count_mode = "task_local_all_ranks" if task_local_enabled else "ep_rank0_only"
    else:
        raise RuntimeError(
            f"Unsupported SPARSE_TOKEN_COUNT_MODE={token_count_mode_requested!r}; "
            "use auto, ep_rank0_only, or all_ranks"
        )
    token_count_contributor = bool(True if token_count_mode in ("all_ranks", "task_local_all_ranks") else (env.rank % ep_size == 0))
    ep_rank0_token_count_contributor = bool(env.rank % ep_size == 0)

    prompt_tokens_t = t(metrics.prompt_tokens if token_count_contributor else 0, torch.long)
    gen_tokens_t = t(metrics.gen_tokens if token_count_contributor else 0, torch.long)
    token_count_contributor_ranks_t = t(1 if token_count_contributor else 0, torch.long)
    prompt_tokens_ep_rank0_t = t(metrics.prompt_tokens if ep_rank0_token_count_contributor else 0, torch.long)
    gen_tokens_ep_rank0_t = t(metrics.gen_tokens if ep_rank0_token_count_contributor else 0, torch.long)
    ep_rank0_token_count_contributor_ranks_t = t(1 if ep_rank0_token_count_contributor else 0, torch.long)
    prompt_tokens_all_ranks_t = t(metrics.prompt_tokens, torch.long)
    gen_tokens_all_ranks_t = t(metrics.gen_tokens, torch.long)
    all_ranks_token_count_contributor_ranks_t = t(1, torch.long)
    all_reduce_sum_(prompt_tokens_t)
    all_reduce_sum_(gen_tokens_t)
    all_reduce_sum_(token_count_contributor_ranks_t)
    all_reduce_sum_(prompt_tokens_ep_rank0_t)
    all_reduce_sum_(gen_tokens_ep_rank0_t)
    all_reduce_sum_(ep_rank0_token_count_contributor_ranks_t)
    all_reduce_sum_(prompt_tokens_all_ranks_t)
    all_reduce_sum_(gen_tokens_all_ranks_t)
    all_reduce_sum_(all_ranks_token_count_contributor_ranks_t)

    # ---------------- basic traffic ----------------
    comm = t(metrics.comm_bytes_activation, torch.long)
    comm_remote = t(getattr(metrics, "comm_bytes_activation_remote", 0), torch.long)
    comm_intra_node = t(getattr(metrics, "comm_bytes_activation_intra_node", 0), torch.long)
    comm_inter_node = t(getattr(metrics, "comm_bytes_activation_inter_node", 0), torch.long)
    weaknet_intra_items = t(getattr(metrics, "weaknet_intra_node_item_count", 0), torch.long)
    weaknet_inter_items = t(getattr(metrics, "weaknet_inter_node_item_count", 0), torch.long)
    weaknet_intra_transfers = t(getattr(metrics, "weaknet_intra_node_transfer_count", 0), torch.long)
    weaknet_inter_transfers = t(getattr(metrics, "weaknet_inter_node_transfer_count", 0), torch.long)
    weaknet_packet_sumsq = t(getattr(metrics, "weaknet_inter_node_packet_item_sumsq", 0), torch.long)
    weaknet_packet_max_items = t(getattr(metrics, "weaknet_inter_node_packet_max_items", 0), torch.long)
    weaknet_packet_le1 = t(getattr(metrics, "weaknet_inter_node_packet_le1_count", 0), torch.long)
    weaknet_packet_le2 = t(getattr(metrics, "weaknet_inter_node_packet_le2_count", 0), torch.long)
    weaknet_packet_le4 = t(getattr(metrics, "weaknet_inter_node_packet_le4_count", 0), torch.long)
    weaknet_packet_le8 = t(getattr(metrics, "weaknet_inter_node_packet_le8_count", 0), torch.long)
    weaknet_packet_gt8 = t(getattr(metrics, "weaknet_inter_node_packet_gt8_count", 0), torch.long)
    weaknet_small_packets = t(getattr(metrics, "weaknet_inter_node_small_packet_count", 0), torch.long)
    weaknet_small_packet_items = t(getattr(metrics, "weaknet_inter_node_small_packet_item_count", 0), torch.long)
    weaknet_inter_sim_ms = t(getattr(metrics, "weaknet_inter_node_sim_ms", 0.0), torch.float64)
    weaknet_inter_sim_ms_rankmax = t(getattr(metrics, "weaknet_inter_node_sim_ms", 0.0), torch.float64)
    kvb = t(metrics.comm_bytes_kv, torch.long)
    skvs = t(metrics.skvs_accounted_bytes, torch.long)
    rank_grouped_pair_entries = t(getattr(metrics, "rank_grouped_pair_entries", 0), torch.long)
    rank_grouped_packet_entries = t(getattr(metrics, "rank_grouped_packet_entries", 0), torch.long)
    rank_grouped_dispatch_calls = t(getattr(metrics, "rank_grouped_dispatch_calls", 0), torch.long)
    node_grouped_pair_entries = t(getattr(metrics, "node_grouped_pair_entries", 0), torch.long)
    node_grouped_node_packet_entries = t(getattr(metrics, "node_grouped_node_packet_entries", 0), torch.long)
    node_grouped_rank_packet_entries = t(getattr(metrics, "node_grouped_rank_packet_entries", 0), torch.long)
    node_grouped_dispatch_calls = t(getattr(metrics, "node_grouped_dispatch_calls", 0), torch.long)
    node_grouped_pack_ms = t(getattr(metrics, "node_grouped_pack_ms", 0.0), torch.float64)
    node_grouped_metadata_ms = t(getattr(metrics, "node_grouped_metadata_ms", 0.0), torch.float64)
    node_grouped_intra_fanout_ms = t(getattr(metrics, "node_grouped_intra_fanout_ms", 0.0), torch.float64)
    node_grouped_inter_send_ms = t(getattr(metrics, "node_grouped_inter_send_ms", 0.0), torch.float64)
    node_grouped_inter_send_weaknet_sleep_ms = t(getattr(metrics, "node_grouped_inter_send_weaknet_sleep_ms", 0.0), torch.float64)
    node_grouped_inter_send_stage1_weaknet_sleep_ms = t(getattr(metrics, "node_grouped_inter_send_stage1_weaknet_sleep_ms", 0.0), torch.float64)
    node_grouped_inter_send_stage2_weaknet_sleep_ms = t(getattr(metrics, "node_grouped_inter_send_stage2_weaknet_sleep_ms", 0.0), torch.float64)
    node_grouped_inter_send_stage1_collective_ms = t(getattr(metrics, "node_grouped_inter_send_stage1_collective_ms", 0.0), torch.float64)
    node_grouped_inter_send_stage2_collective_ms = t(getattr(metrics, "node_grouped_inter_send_stage2_collective_ms", 0.0), torch.float64)
    node_grouped_metadata_exchange_ms = t(getattr(metrics, "node_grouped_metadata_exchange_ms", 0.0), torch.float64)
    node_grouped_gather_ms = t(getattr(metrics, "node_grouped_gather_ms", 0.0), torch.float64)
    node_grouped_total_overhead_ms = t(getattr(metrics, "node_grouped_total_overhead_ms", 0.0), torch.float64)
    node_grouped_same_node_packet_entries = t(getattr(metrics, "node_grouped_same_node_packet_entries", 0), torch.long)
    node_grouped_cross_node_packet_entries = t(getattr(metrics, "node_grouped_cross_node_packet_entries", 0), torch.long)
    node_grouped_same_node_bypass_bytes = t(getattr(metrics, "node_grouped_same_node_bypass_bytes", 0), torch.long)
    node_grouped_stage2_local_rank_packet_entries = t(getattr(metrics, "node_grouped_stage2_local_rank_packet_entries", 0), torch.long)
    node_grouped_stage2_remote_rank_packet_entries = t(getattr(metrics, "node_grouped_stage2_remote_rank_packet_entries", 0), torch.long)
    node_grouped_stage2_local_bypass_bytes = t(getattr(metrics, "node_grouped_stage2_local_bypass_bytes", 0), torch.long)
    node_grouped_metadata_collective_count = t(getattr(metrics, "node_grouped_metadata_collective_count", 0), torch.long)
    node_grouped_metadata_bytes = t(getattr(metrics, "node_grouped_metadata_bytes", 0), torch.long)
    node_grouped_counts_build_ms = t(getattr(metrics, "node_grouped_counts_build_ms", 0.0), torch.float64)
    node_grouped_splits_build_ms = t(getattr(metrics, "node_grouped_splits_build_ms", 0.0), torch.float64)
    node_grouped_reverse_map_build_ms = t(getattr(metrics, "node_grouped_reverse_map_build_ms", 0.0), torch.float64)
    node_grouped_rank_to_node_gather_ms = t(getattr(metrics, "node_grouped_rank_to_node_gather_ms", 0.0), torch.float64)
    node_grouped_node_to_token_restore_ms = t(getattr(metrics, "node_grouped_node_to_token_restore_ms", 0.0), torch.float64)
    node_grouped_final_combine_ms = t(getattr(metrics, "node_grouped_final_combine_ms", 0.0), torch.float64)
    node_grouped_combine_intermediate_buffer_bytes = t(getattr(metrics, "node_grouped_combine_intermediate_buffer_bytes", 0), torch.long)
    node_grouped_allocation_ms = t(getattr(metrics, "node_grouped_allocation_ms", 0.0), torch.float64)
    node_grouped_allocation_count = t(getattr(metrics, "node_grouped_allocation_count", 0), torch.long)
    node_grouped_allocation_bytes = t(getattr(metrics, "node_grouped_allocation_bytes", 0), torch.long)
    node_grouped_workspace_reuse_hit = t(getattr(metrics, "node_grouped_workspace_reuse_hit", 0), torch.long)
    node_grouped_enabled_layers = t(getattr(metrics, "node_grouped_enabled_layers", 0), torch.long)
    node_grouped_fallback_layers = t(getattr(metrics, "node_grouped_fallback_layers", 0), torch.long)
    node_grouped_estimated_saving_ms = t(getattr(metrics, "node_grouped_estimated_saving_ms", 0.0), torch.float64)

    h2d_total = t(metrics.expert_weight_h2d_bytes, torch.long)
    h2d_prefetch = t(metrics.expert_weight_h2d_prefetch_bytes, torch.long)
    h2d_demand = t(metrics.expert_weight_h2d_demand_bytes, torch.long)
    oracle_stage_h2d = t(getattr(metrics, "oracle_stage_h2d_bytes", 0), torch.long)
    oracle_stage_event_count = t(getattr(metrics, "oracle_stage_event_count", 0), torch.long)
    oracle_stage_ms = t(getattr(metrics, "oracle_stage_ms", 0.0), torch.float64)
    oracle_stage_prefill_ms = t(getattr(metrics, "oracle_stage_prefill_ms", 0.0), torch.float64)
    oracle_stage_decode_ms = t(getattr(metrics, "oracle_stage_decode_ms", 0.0), torch.float64)
    moe_router_ms = t(getattr(metrics, "moe_router_ms", 0.0), torch.float64)
    moe_expert_ms = t(getattr(metrics, "moe_expert_ms", 0.0), torch.float64)
    moe_shared_ms = t(getattr(metrics, "moe_shared_ms", 0.0), torch.float64)
    moe_router_ms_rankmax = t(getattr(metrics, "moe_router_ms", 0.0), torch.float64)
    moe_expert_ms_rankmax = t(getattr(metrics, "moe_expert_ms", 0.0), torch.float64)
    moe_shared_ms_rankmax = t(getattr(metrics, "moe_shared_ms", 0.0), torch.float64)
    moe_timing_calls = t(getattr(metrics, "moe_timing_calls", 0), torch.long)
    rs_mlp_ms = t(getattr(metrics, "rs_mlp_ms", 0.0), torch.float64)
    rs_mlp_ms_rankmax = t(getattr(metrics, "rs_mlp_ms", 0.0), torch.float64)
    rs_mlp_calls = t(getattr(metrics, "rs_mlp_calls", 0), torch.long)
    rs_next_layer_pred_calls = t(getattr(metrics, "rs_next_layer_pred_calls", 0), torch.long)
    demand_h2d_exposed_ms = t(getattr(metrics, "demand_h2d_exposed_ms", 0.0), torch.float64)
    demand_h2d_exposed_prefill_ms = t(getattr(metrics, "demand_h2d_exposed_prefill_ms", 0.0), torch.float64)
    demand_h2d_exposed_decode_ms = t(getattr(metrics, "demand_h2d_exposed_decode_ms", 0.0), torch.float64)
    sim_demand_h2d_exposed_ms = t(getattr(metrics, "sim_demand_h2d_exposed_ms", 0.0), torch.float64)
    sim_demand_h2d_exposed_prefill_ms = t(getattr(metrics, "sim_demand_h2d_exposed_prefill_ms", 0.0), torch.float64)
    sim_demand_h2d_exposed_decode_ms = t(getattr(metrics, "sim_demand_h2d_exposed_decode_ms", 0.0), torch.float64)
    sim_demand_h2d_exposed_ms_rankmax = t(getattr(metrics, "sim_demand_h2d_exposed_ms", 0.0), torch.float64)
    sim_demand_h2d_exposed_decode_ms_rankmax = t(getattr(metrics, "sim_demand_h2d_exposed_decode_ms", 0.0), torch.float64)
    sim_h2d_extra_sleep_ms = t(getattr(metrics, "sim_h2d_extra_sleep_ms", 0.0), torch.float64)
    demand_h2d_exposed_ms_rankmax = t(getattr(metrics, "demand_h2d_exposed_ms", 0.0), torch.float64)
    demand_h2d_exposed_prefill_ms_rankmax = t(getattr(metrics, "demand_h2d_exposed_prefill_ms", 0.0), torch.float64)
    demand_h2d_exposed_decode_ms_rankmax = t(getattr(metrics, "demand_h2d_exposed_decode_ms", 0.0), torch.float64)
    demand_h2d_event_count = t(getattr(metrics, "demand_h2d_event_count", 0), torch.long)
    prefetch_h2d_copy_ms = t(getattr(metrics, "prefetch_h2d_copy_ms", 0.0), torch.float64)
    prefetch_h2d_event_count = t(getattr(metrics, "prefetch_h2d_event_count", 0), torch.long)
    prefetch_wait_ms = t(getattr(metrics, "prefetch_wait_ms", 0.0), torch.float64)
    prefetch_wait_event_count = t(getattr(metrics, "prefetch_wait_event_count", 0), torch.long)
    prefetch_oom_skip_count = t(getattr(metrics, "prefetch_oom_skip_count", 0), torch.long)
    plain_ep_executor_calls = t(getattr(metrics, "plain_ep_executor_calls", 0), torch.long)
    epipe_executor_calls = t(getattr(metrics, "epipe_executor_calls", 0), torch.long)
    epipe_executor_attempt_count = t(getattr(metrics, "epipe_executor_attempt_count", 0), torch.long)
    sim_expert_exec_ms = t(getattr(metrics, "sim_expert_exec_ms", 0.0), torch.float64)
    sim_expert_exec_ms_rankmax = t(getattr(metrics, "sim_expert_exec_ms", 0.0), torch.float64)
    sim_compute_extra_sleep_ms = t(getattr(metrics, "sim_compute_extra_sleep_ms", 0.0), torch.float64)

    # ---------------- hit/miss counters ----------------
    prefetch_hit = t(metrics.prefetch_hit_count, torch.long)
    prefetch_wasted = t(metrics.prefetch_wasted_count, torch.long)
    cache_hit = t(metrics.cache_hit_count, torch.long)
    cache_miss = t(metrics.cache_miss_count, torch.long)
    rs_prefetch_eval_events = t(getattr(metrics, "rs_prefetch_eval_events", 0), torch.long)
    rs_prefetch_eval_pred_total = t(getattr(metrics, "rs_prefetch_eval_pred_total", 0), torch.long)
    rs_prefetch_eval_actual_total = t(getattr(metrics, "rs_prefetch_eval_actual_total", 0), torch.long)
    rs_prefetch_eval_overlap_total = t(getattr(metrics, "rs_prefetch_eval_overlap_total", 0), torch.long)
    prefetch_budget_zero_count = t(getattr(metrics, "prefetch_budget_zero_count", 0), torch.long)
    prefetch_budget_positive_count = t(getattr(metrics, "prefetch_budget_positive_count", 0), torch.long)
    prefetch_admitted_count = t(getattr(metrics, "prefetch_admitted_count", 0), torch.long)
    rs_pred_candidate_count = t(getattr(metrics, "rs_pred_candidate_count", 0), torch.long)
    rs_pred_local_candidate_count = t(getattr(metrics, "rs_pred_local_candidate_count", 0), torch.long)
    rs_pred_nonlocal_candidate_count = t(getattr(metrics, "rs_pred_nonlocal_candidate_count", 0), torch.long)
    rs_candidate_filter_in_count = t(getattr(metrics, "rs_candidate_filter_in_count", 0), torch.long)
    rs_candidate_filter_out_count = t(getattr(metrics, "rs_candidate_filter_out_count", 0), torch.long)
    rs_candidate_filter_out_cached_count = t(getattr(metrics, "rs_candidate_filter_out_cached_count", 0), torch.long)
    rs_candidate_filter_out_low_score_count = t(getattr(metrics, "rs_candidate_filter_out_low_score_count", 0), torch.long)
    rs_candidate_survived_count = t(getattr(metrics, "rs_candidate_survived_count", 0), torch.long)
    prefetch_issued_count = t(getattr(metrics, "prefetch_issued_count", 0), torch.long)
    prefetch_completed_count = t(getattr(metrics, "prefetch_completed_count", 0), torch.long)
    prefetch_complete_before_use_count = t(getattr(metrics, "prefetch_complete_before_use_count", 0), torch.long)
    prefetch_rejected_active_replace = t(getattr(metrics, "prefetch_rejected_active_replace", 0), torch.long)
    prefetch_rejected_margin = t(getattr(metrics, "prefetch_rejected_margin", 0), torch.long)
    prefetch_rejected_cache_full = t(getattr(metrics, "prefetch_rejected_cache_full", 0), torch.long)
    prefetch_rejected_low_score = t(getattr(metrics, "prefetch_rejected_low_score", 0), torch.long)
    prefill_scp_transient_load_count = t(getattr(metrics, "prefill_scp_transient_load_count", 0), torch.long)
    prefill_scp_transient_hit_count = t(getattr(metrics, "prefill_scp_transient_hit_count", 0), torch.long)
    prefill_scp_transient_promote_count = t(getattr(metrics, "prefill_scp_transient_promote_count", 0), torch.long)
    prefill_scp_transient_evict_count = t(getattr(metrics, "prefill_scp_transient_evict_count", 0), torch.long)
    prefill_scp_transient_resident_at_decode_start = t(getattr(metrics, "prefill_scp_transient_resident_at_decode_start", 0), torch.long)
    prefill_scp_transient_demote_count = t(getattr(metrics, "prefill_scp_transient_demote_count", 0), torch.long)
    prefill_scp_transient_resident_at_decode_end = t(getattr(metrics, "prefill_scp_transient_resident_at_decode_end", 0), torch.long)
    resolver_decode_total_queries = t(getattr(metrics, "resolver_decode_total_queries", 0), torch.long)
    resolver_decode_rr_owner_count = t(getattr(metrics, "resolver_decode_rr_owner_count", 0), torch.long)
    resolver_decode_scp_replica_selected_count = t(getattr(metrics, "resolver_decode_scp_replica_selected_count", 0), torch.long)
    resolver_decode_rr_fallback_count = t(getattr(metrics, "resolver_decode_rr_fallback_count", 0), torch.long)
    resolver_decode_replica_reject_not_ready_count = t(getattr(metrics, "resolver_decode_replica_reject_not_ready_count", 0), torch.long)
    resolver_decode_replica_reject_no_comm_gain_count = t(getattr(metrics, "resolver_decode_replica_reject_no_comm_gain_count", 0), torch.long)
    resolver_decode_replica_cache_ready_count = t(getattr(metrics, "resolver_decode_replica_cache_ready_count", 0), torch.long)
    resolver_decode_replica_prefetch_ready_count = t(getattr(metrics, "resolver_decode_replica_prefetch_ready_count", 0), torch.long)
    resolver_decode_route_cost_before = t(getattr(metrics, "resolver_decode_route_cost_before", 0.0), torch.float64)
    resolver_decode_route_cost_after = t(getattr(metrics, "resolver_decode_route_cost_after", 0.0), torch.float64)
    scp_prefill_route_diag_items = t(getattr(metrics, "scp_prefill_route_diag_items", 0), torch.long)
    scp_prefill_route_diag_replica_items = t(getattr(metrics, "scp_prefill_route_diag_replica_items", 0), torch.long)
    scp_prefill_route_diag_cross_items_before = t(getattr(metrics, "scp_prefill_route_diag_cross_items_before", 0), torch.long)
    scp_prefill_route_diag_cross_items_after = t(getattr(metrics, "scp_prefill_route_diag_cross_items_after", 0), torch.long)
    scp_prefill_route_diag_cross_events_before = t(getattr(metrics, "scp_prefill_route_diag_cross_events_before", 0), torch.long)
    scp_prefill_route_diag_cross_events_after = t(getattr(metrics, "scp_prefill_route_diag_cross_events_after", 0), torch.long)
    scp_prefill_route_diag_cost_before = t(getattr(metrics, "scp_prefill_route_diag_cost_before", 0.0), torch.float64)
    scp_prefill_route_diag_cost_after = t(getattr(metrics, "scp_prefill_route_diag_cost_after", 0.0), torch.float64)
    scp_decode_route_diag_items = t(getattr(metrics, "scp_decode_route_diag_items", 0), torch.long)
    scp_decode_route_diag_replica_items = t(getattr(metrics, "scp_decode_route_diag_replica_items", 0), torch.long)
    scp_decode_route_diag_cross_items_before = t(getattr(metrics, "scp_decode_route_diag_cross_items_before", 0), torch.long)
    scp_decode_route_diag_cross_items_after = t(getattr(metrics, "scp_decode_route_diag_cross_items_after", 0), torch.long)
    scp_decode_route_diag_cross_events_before = t(getattr(metrics, "scp_decode_route_diag_cross_events_before", 0), torch.long)
    scp_decode_route_diag_cross_events_after = t(getattr(metrics, "scp_decode_route_diag_cross_events_after", 0), torch.long)
    scp_decode_route_diag_cost_before = t(getattr(metrics, "scp_decode_route_diag_cost_before", 0.0), torch.float64)
    scp_decode_route_diag_cost_after = t(getattr(metrics, "scp_decode_route_diag_cost_after", 0.0), torch.float64)
    scp_table_local_replica_route_count = t(getattr(metrics, "scp_table_local_replica_route_count", 0), torch.long)
    scp_table_rr_fallback_count = t(getattr(metrics, "scp_table_rr_fallback_count", 0), torch.long)
    scp_table_cross_node_route_count = t(getattr(metrics, "scp_table_cross_node_route_count", 0), torch.long)
    scp_table_same_node_route_count = t(getattr(metrics, "scp_table_same_node_route_count", 0), torch.long)
    scp_route_cost_before_sum = t(getattr(metrics, "scp_route_cost_before_sum", 0.0), torch.float64)
    scp_route_cost_after_sum = t(getattr(metrics, "scp_route_cost_after_sum", 0.0), torch.float64)
    scp_lookup_calls = t(getattr(metrics, "scp_lookup_calls", 0), torch.long)
    scp_lookup_overhead_ms = t(getattr(metrics, "scp_lookup_overhead_ms", 0.0), torch.float64)
    rs_prefetch_target_resolver_match_count = t(getattr(metrics, "rs_prefetch_target_resolver_match_count", 0), torch.long)
    rs_prefetch_target_resolver_mismatch_count = t(getattr(metrics, "rs_prefetch_target_resolver_mismatch_count", 0), torch.long)
    rs_prefetch_would_be_wrong_without_resolver_count = t(getattr(metrics, "rs_prefetch_would_be_wrong_without_resolver_count", 0), torch.long)
    rs_scp_target_mismatch_count = t(getattr(metrics, "rs_scp_target_mismatch_count", 0), torch.long)
    ttl_prevented_evictions = t(getattr(metrics, "ttl_prevented_evictions", 0), torch.long)
    cache_eviction_fallback_all_protected = t(getattr(metrics, "cache_eviction_fallback_all_protected", 0), torch.long)
    warmup_loaded_experts = t(getattr(metrics, "warmup_loaded_experts", 0), torch.long)
    warmup_skipped_resident = t(getattr(metrics, "warmup_skipped_resident", 0), torch.long)
    warmup_h2d = t(getattr(metrics, "warmup_h2d_bytes", 0), torch.long)
    warmup_time_ms = t(getattr(metrics, "warmup_time_ms", 0.0), torch.float64)
    transition_lookup_calls = t(getattr(metrics, "transition_lookup_calls", 0), torch.long)
    transition_lookup_overhead_ms = t(getattr(metrics, "transition_lookup_overhead_ms", 0.0), torch.float64)
    transition_prefetch_admitted = t(getattr(metrics, "transition_prefetch_admitted", 0), torch.long)
    transition_prefetch_hit = t(getattr(metrics, "transition_prefetch_hit", 0), torch.long)
    transition_prefetch_late = t(getattr(metrics, "transition_prefetch_late", 0), torch.long)
    transition_prefetch_wasted = t(getattr(metrics, "transition_prefetch_wasted", 0), torch.long)
    transition_safe_evict_denied = t(getattr(metrics, "transition_safe_evict_denied", 0), torch.long)
    transition_no_replace_denied = t(getattr(metrics, "transition_no_replace_denied", 0), torch.long)
    epipe_float_fields = (
        "epipe_demand_h2d_ms",
        "epipe_wait_ms",
        "epipe_hidden_h2d_ms_est",
        "sampled_demand_h2d_exposed_decode_ms",
        "sampled_prefetch_wait_ms",
        "sampled_epipe_hidden_h2d_ms",
        "sampled_epipe_wait_ms",
        "sampled_epipe_control_overhead_ms",
    )
    epipe_fields = (
        "epipe_shadow_count",
        "epipe_enabled_count",
        "epipe_fallback_count",
        "epipe_fallback_no_ready",
        "epipe_fallback_no_missing",
        "epipe_fallback_no_miss_trigger",
        "epipe_fallback_no_ready_lookahead",
        "epipe_fallback_low_ready_items",
        "epipe_fallback_prefill_disabled",
        "epipe_fallback_no_memory_headroom",
        "epipe_fallback_memory",
        "epipe_fallback_cache_full_no_temp",
        "epipe_fallback_exception",
        "epipe_decode_rs_miss_trigger_count",
        "epipe_decode_rs_skipped_no_miss_count",
        "epipe_prefill_miss_trigger_count",
        "epipe_prefill_shadow_opportunity_count",
        "epipe_rs_buffered_calls",
        "epipe_rs_buffer_flush_count",
        "epipe_rs_buffered_ready_group_count",
        "epipe_rs_buffered_ready_item_count",
        "epipe_rs_residual_scan_calls",
        "epipe_rs_residual_scan_ready_group_count",
        "epipe_rs_residual_scan_ready_item_count",
        "epipe_rs_residual_scan_residual_count",
        "epipe_rs_residual_scan_late_count",
        "epipe_rs_residual_scan_demand_count",
        "epipe_rs_residual_scan_no_ready_count",
        "epipe_rs_trigger_count",
        "epipe_rs_late_prefetch_trigger_count",
        "epipe_rs_demand_miss_trigger_count",
        "epipe_rs_late_prefetch_overlap_count",
        "epipe_rs_demand_stage_count",
        "epipe_rs_demand_stage_disabled_by_gate_count",
        "epipe_rs_demand_stage_force_count",
        "epipe_prefill_executor_calls",
        "epipe_prefill_stage_success_count",
        "epipe_decode_no_rs_executor_calls",
        "epipe_decode_no_rs_stage_success_count",
        "epipe_decode_rs_executor_calls",
        "epipe_decode_rs_stage_success_count",
        "epipe_prefill_enabled_count",
        "epipe_decode_no_rs_enabled_count",
        "epipe_decode_rs_late_overlap_count",
        "epipe_decode_rs_demand_stage_count",
        "epipe_rs_gate_skip_count",
        "epipe_rs_gate_reenable_count",
        "epipe_rs_buffer_too_small_count",
        "epipe_lookahead_ready_group_count",
        "epipe_lookahead_ready_item_count",
        "epipe_staged_expert_count",
        "epipe_staged_h2d_bytes",
        "epipe_deferred_lru_commit_count",
        "epipe_prefill_transient_commit_count",
        "epipe_prefill_transient_demote_count",
        "epipe_prefill_transient_promote_count",
        "epipe_done_group_skip_count",
        "epipe_ready_expert_count",
        "epipe_missing_expert_count",
        "epipe_ready_item_count",
        "epipe_missing_item_count",
        "epipe_cache_free_slot_count",
        "epipe_cache_full_staged_count",
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
        "epipe_no_evict_count",
        "epipe_missing_h2d_bytes_est",
        "epipe_demand_h2d_bytes",
        "epipe_staged_missing_count",
        "epipe_prefill_multistage_shadow_count",
        "epipe_prefill_multistage_ready_expert_count",
        "epipe_prefill_multistage_missing_expert_count",
        "epipe_prefill_multistage_ready_item_count",
        "epipe_prefill_multistage_missing_item_count",
        "epipe_prefill_multistage_top_missing_item_count_sum",
        "epipe_prefill_multistage_est_expert_h2d_bytes",
        "epipe_prefill_multistage_est_expert_h2d_mb",
        "epipe_prefill_multistage_est_ready_compute_items",
        "epipe_prefill_multistage_est_hidden_h2d_items_n1",
        "epipe_prefill_multistage_est_hidden_h2d_items_n2",
        "epipe_prefill_multistage_est_hidden_h2d_items_n4",
        *epipe_float_fields,
    )
    epipe_tensors = {
        name: t(
            getattr(metrics, name, 0.0 if name in epipe_float_fields else 0),
            torch.float64 if name in epipe_float_fields else torch.long,
        )
        for name in epipe_fields
    }

    # Keep a rank-local snapshot before all-reduce mutates tensors. This is the
    # structured view used by bottleneck attribution: total bytes can be summed,
    # but latency often follows the slowest rank.
    local_layers = list((metrics.per_layer or {}).values())
    local_comm_ms = sum(float(getattr(lp, "comm_ms", 0.0)) for lp in local_layers)
    local_expert_exec_ms = sum(float(getattr(lp, "expert_exec_ms", 0.0)) for lp in local_layers)
    local_grouping_ms = sum(float(getattr(lp, "grouping_overhead_ms", 0.0)) for lp in local_layers)
    local_runtime_overhead_ms = sum(float(getattr(lp, "runtime_overhead_ms", 0.0)) for lp in local_layers)

    rank_local_breakdown = {
        "rank": int(env.rank),
        "local_rank": int(env.local_rank),
        "logical_node": int(_parse_sim_node_map(env.world)[env.rank]) if env.rank < len(_parse_sim_node_map(env.world)) else -1,
        "activation_comm_mb": float(getattr(metrics, "comm_bytes_activation", 0)) / 1024**2,
        "activation_remote_mb": float(getattr(metrics, "comm_bytes_activation_remote", 0)) / 1024**2,
        "activation_intra_node_mb": float(getattr(metrics, "comm_bytes_activation_intra_node", 0)) / 1024**2,
        "activation_inter_node_mb": float(getattr(metrics, "comm_bytes_activation_inter_node", 0)) / 1024**2,
        "weaknet_intra_node_item_count": int(getattr(metrics, "weaknet_intra_node_item_count", 0)),
        "weaknet_inter_node_item_count": int(getattr(metrics, "weaknet_inter_node_item_count", 0)),
        "weaknet_intra_node_transfer_count": int(getattr(metrics, "weaknet_intra_node_transfer_count", 0)),
        "weaknet_inter_node_transfer_count": int(getattr(metrics, "weaknet_inter_node_transfer_count", 0)),
        "weaknet_inter_node_packet_item_sumsq": int(getattr(metrics, "weaknet_inter_node_packet_item_sumsq", 0)),
        "weaknet_inter_node_packet_max_items": int(getattr(metrics, "weaknet_inter_node_packet_max_items", 0)),
        "weaknet_inter_node_packet_le1_count": int(getattr(metrics, "weaknet_inter_node_packet_le1_count", 0)),
        "weaknet_inter_node_packet_le2_count": int(getattr(metrics, "weaknet_inter_node_packet_le2_count", 0)),
        "weaknet_inter_node_packet_le4_count": int(getattr(metrics, "weaknet_inter_node_packet_le4_count", 0)),
        "weaknet_inter_node_packet_le8_count": int(getattr(metrics, "weaknet_inter_node_packet_le8_count", 0)),
        "weaknet_inter_node_packet_gt8_count": int(getattr(metrics, "weaknet_inter_node_packet_gt8_count", 0)),
        "weaknet_inter_node_small_packet_count": int(getattr(metrics, "weaknet_inter_node_small_packet_count", 0)),
        "weaknet_inter_node_small_packet_item_count": int(getattr(metrics, "weaknet_inter_node_small_packet_item_count", 0)),
        "weaknet_inter_node_pair_event_counts": dict(getattr(metrics, "weaknet_inter_node_pair_event_counts", {}) or {}),
        "weaknet_inter_node_pair_item_counts": dict(getattr(metrics, "weaknet_inter_node_pair_item_counts", {}) or {}),
        "weaknet_packet_trace": list(getattr(metrics, "weaknet_packet_trace", None) or []),
        "physical_moe_comm_trace": list(getattr(metrics, "physical_moe_comm_trace", None) or []),
        "weaknet_inter_node_sim_ms": float(getattr(metrics, "weaknet_inter_node_sim_ms", 0.0)),
        "node_grouped_pack_ms": float(getattr(metrics, "node_grouped_pack_ms", 0.0)),
        "node_grouped_metadata_ms": float(getattr(metrics, "node_grouped_metadata_ms", 0.0)),
        "node_grouped_intra_fanout_ms": float(getattr(metrics, "node_grouped_intra_fanout_ms", 0.0)),
        "node_grouped_inter_send_ms": float(getattr(metrics, "node_grouped_inter_send_ms", 0.0)),
        "node_grouped_inter_send_weaknet_sleep_ms": float(getattr(metrics, "node_grouped_inter_send_weaknet_sleep_ms", 0.0)),
        "node_grouped_inter_send_stage1_weaknet_sleep_ms": float(getattr(metrics, "node_grouped_inter_send_stage1_weaknet_sleep_ms", 0.0)),
        "node_grouped_inter_send_stage2_weaknet_sleep_ms": float(getattr(metrics, "node_grouped_inter_send_stage2_weaknet_sleep_ms", 0.0)),
        "node_grouped_inter_send_stage1_collective_ms": float(getattr(metrics, "node_grouped_inter_send_stage1_collective_ms", 0.0)),
        "node_grouped_inter_send_stage2_collective_ms": float(getattr(metrics, "node_grouped_inter_send_stage2_collective_ms", 0.0)),
        "node_grouped_metadata_exchange_ms": float(getattr(metrics, "node_grouped_metadata_exchange_ms", 0.0)),
        "node_grouped_gather_ms": float(getattr(metrics, "node_grouped_gather_ms", 0.0)),
        "node_grouped_total_overhead_ms": float(getattr(metrics, "node_grouped_total_overhead_ms", 0.0)),
        "node_grouped_same_node_packet_entries": int(getattr(metrics, "node_grouped_same_node_packet_entries", 0)),
        "node_grouped_cross_node_packet_entries": int(getattr(metrics, "node_grouped_cross_node_packet_entries", 0)),
        "node_grouped_same_node_bypass_mb": float(getattr(metrics, "node_grouped_same_node_bypass_bytes", 0)) / 1024**2,
        "node_grouped_stage2_local_rank_packet_entries": int(getattr(metrics, "node_grouped_stage2_local_rank_packet_entries", 0)),
        "node_grouped_stage2_remote_rank_packet_entries": int(getattr(metrics, "node_grouped_stage2_remote_rank_packet_entries", 0)),
        "node_grouped_stage2_local_bypass_mb": float(getattr(metrics, "node_grouped_stage2_local_bypass_bytes", 0)) / 1024**2,
        "node_grouped_metadata_collective_count": int(getattr(metrics, "node_grouped_metadata_collective_count", 0)),
        "node_grouped_metadata_mb": float(getattr(metrics, "node_grouped_metadata_bytes", 0)) / 1024**2,
        "node_grouped_counts_build_ms": float(getattr(metrics, "node_grouped_counts_build_ms", 0.0)),
        "node_grouped_splits_build_ms": float(getattr(metrics, "node_grouped_splits_build_ms", 0.0)),
        "node_grouped_reverse_map_build_ms": float(getattr(metrics, "node_grouped_reverse_map_build_ms", 0.0)),
        "node_grouped_rank_to_node_gather_ms": float(getattr(metrics, "node_grouped_rank_to_node_gather_ms", 0.0)),
        "node_grouped_node_to_token_restore_ms": float(getattr(metrics, "node_grouped_node_to_token_restore_ms", 0.0)),
        "node_grouped_final_combine_ms": float(getattr(metrics, "node_grouped_final_combine_ms", 0.0)),
        "node_grouped_combine_intermediate_buffer_mb": float(getattr(metrics, "node_grouped_combine_intermediate_buffer_bytes", 0)) / 1024**2,
        "node_grouped_allocation_ms": float(getattr(metrics, "node_grouped_allocation_ms", 0.0)),
        "node_grouped_allocation_count": int(getattr(metrics, "node_grouped_allocation_count", 0)),
        "node_grouped_allocation_mb": float(getattr(metrics, "node_grouped_allocation_bytes", 0)) / 1024**2,
        "node_grouped_workspace_reuse_hit": int(getattr(metrics, "node_grouped_workspace_reuse_hit", 0)),
        "node_grouped_enabled_layers": int(getattr(metrics, "node_grouped_enabled_layers", 0)),
        "node_grouped_fallback_layers": int(getattr(metrics, "node_grouped_fallback_layers", 0)),
        "node_grouped_estimated_saving_ms": float(getattr(metrics, "node_grouped_estimated_saving_ms", 0.0)),
        "expert_h2d_demand_mb": float(getattr(metrics, "expert_weight_h2d_demand_bytes", 0)) / 1024**2,
        "expert_h2d_prefetch_mb": float(getattr(metrics, "expert_weight_h2d_prefetch_bytes", 0)) / 1024**2,
        "demand_h2d_exposed_decode_ms": float(getattr(metrics, "demand_h2d_exposed_decode_ms", 0.0)),
        "demand_h2d_exposed_ms": float(getattr(metrics, "demand_h2d_exposed_ms", 0.0)),
        "sim_demand_h2d_exposed_decode_ms": float(getattr(metrics, "sim_demand_h2d_exposed_decode_ms", 0.0)),
        "sim_demand_h2d_exposed_ms": float(getattr(metrics, "sim_demand_h2d_exposed_ms", 0.0)),
        "sim_h2d_extra_sleep_ms": float(getattr(metrics, "sim_h2d_extra_sleep_ms", 0.0)),
        "demand_h2d_event_count": int(getattr(metrics, "demand_h2d_event_count", 0)),
        "prefetch_h2d_copy_ms": float(getattr(metrics, "prefetch_h2d_copy_ms", 0.0)),
        "prefetch_wait_ms": float(getattr(metrics, "prefetch_wait_ms", 0.0)),
        "plain_ep_executor_calls": int(getattr(metrics, "plain_ep_executor_calls", 0)),
        "epipe_executor_calls": int(getattr(metrics, "epipe_executor_calls", 0)),
        "epipe_executor_attempt_count": int(getattr(metrics, "epipe_executor_attempt_count", 0)),
        "cache_hit_count": int(getattr(metrics, "cache_hit_count", 0)),
        "cache_miss_count": int(getattr(metrics, "cache_miss_count", 0)),
        "prefetch_hit_count": int(getattr(metrics, "prefetch_hit_count", 0)),
        "prefetch_wasted_count": int(getattr(metrics, "prefetch_wasted_count", 0)),
        "scp_table_local_replica_route_count": int(getattr(metrics, "scp_table_local_replica_route_count", 0)),
        "scp_table_rr_fallback_count": int(getattr(metrics, "scp_table_rr_fallback_count", 0)),
        "scp_table_cross_node_route_count": int(getattr(metrics, "scp_table_cross_node_route_count", 0)),
        "scp_table_same_node_route_count": int(getattr(metrics, "scp_table_same_node_route_count", 0)),
        "scp_route_cost_before_sum": float(getattr(metrics, "scp_route_cost_before_sum", 0.0)),
        "scp_route_cost_after_sum": float(getattr(metrics, "scp_route_cost_after_sum", 0.0)),
        "scp_lookup_calls": int(getattr(metrics, "scp_lookup_calls", 0)),
        "scp_lookup_overhead_ms": float(getattr(metrics, "scp_lookup_overhead_ms", 0.0)),
        "rs_scp_target_mismatch_count": int(getattr(metrics, "rs_scp_target_mismatch_count", 0)),
        **{
            name: (
                float(getattr(metrics, name, 0.0))
                if name in epipe_float_fields
                else int(getattr(metrics, name, 0))
            )
            for name in epipe_fields
        },
        "moe_router_ms": float(getattr(metrics, "moe_router_ms", 0.0)),
        "moe_expert_ms": float(getattr(metrics, "moe_expert_ms", 0.0)),
        "moe_shared_ms": float(getattr(metrics, "moe_shared_ms", 0.0)),
        "rs_mlp_ms": float(getattr(metrics, "rs_mlp_ms", 0.0)),
        "comm_ms": float(local_comm_ms),
        "expert_exec_ms": float(local_expert_exec_ms),
        "sim_expert_exec_ms": float(getattr(metrics, "sim_expert_exec_ms", local_expert_exec_ms)),
        "sim_compute_extra_sleep_ms": float(getattr(metrics, "sim_compute_extra_sleep_ms", 0.0)),
        "grouping_overhead_ms": float(local_grouping_ms),
        "runtime_overhead_ms": float(local_runtime_overhead_ms),
    }
    gathered_rank_breakdown = None
    if dist.is_available() and dist.is_initialized():
        gathered_rank_breakdown = [None for _ in range(env.world)] if env.rank == 0 else None
        dist.gather_object(rank_local_breakdown, gathered_rank_breakdown, dst=0)
    elif env.rank == 0:
        gathered_rank_breakdown = [rank_local_breakdown]

    # ---------------- wall / memory ----------------
    prefill_wall = t(metrics.wall_prefill_ms, torch.float64)
    decode_wall = t(metrics.wall_decode_ms, torch.float64)
    ttft_wall = t(metrics.wall_ttft_ms, torch.float64)
    total_wall = t(metrics.wall_total_ms, torch.float64)

    peak_alloc = t(metrics.peak_allocated_mb, torch.float64)
    peak_rsvd = t(metrics.peak_reserved_mb, torch.float64)

    owner_group_count = t(getattr(metrics, "owner_group_count", 0), torch.long)
    owner_changed_samples = t(getattr(metrics, "owner_changed_samples", 0), torch.long)
    owner_empty_ranks = t(getattr(metrics, "owner_empty_ranks_sum", 0), torch.long)
    owner_hist_max_sum = t(getattr(metrics, "owner_hist_max_sum", 0.0), torch.float64)
    owner_hist_min_sum = t(getattr(metrics, "owner_hist_min_sum", 0.0), torch.float64)
    owner_hist_cv_sum = t(getattr(metrics, "owner_hist_cv_sum", 0.0), torch.float64)
    owner_imbalance_ratio_sum = t(getattr(metrics, "owner_imbalance_ratio_sum", 0.0), torch.float64)

    owner_group_reconstruct_ms = t(getattr(metrics, "owner_group_reconstruct_ms", 0.0), torch.float64)
    owner_group_plan_ms = t(getattr(metrics, "owner_group_plan_ms", 0.0), torch.float64)
    owner_group_migrate_ms = t(getattr(metrics, "owner_group_migrate_ms", 0.0), torch.float64)

    owner_attention_ms_sum = t(getattr(metrics, "owner_attention_ms", 0.0), torch.float64)
    owner_attention_ms_max = t(getattr(metrics, "owner_attention_ms", 0.0), torch.float64)
    owner_attention_calls = t(getattr(metrics, "owner_attention_calls", 0), torch.long)

    owner_moe_ms_sum = t(getattr(metrics, "owner_moe_ms", 0.0), torch.float64)
    owner_moe_ms_max = t(getattr(metrics, "owner_moe_ms", 0.0), torch.float64)
    owner_moe_calls = t(getattr(metrics, "owner_moe_calls", 0), torch.long)

    # sum traffic / counters
    all_reduce_sum_(comm)
    all_reduce_sum_(comm_remote)
    all_reduce_sum_(comm_intra_node)
    all_reduce_sum_(comm_inter_node)
    all_reduce_sum_(weaknet_intra_items)
    all_reduce_sum_(weaknet_inter_items)
    all_reduce_sum_(weaknet_intra_transfers)
    all_reduce_sum_(weaknet_inter_transfers)
    all_reduce_sum_(weaknet_packet_sumsq)
    all_reduce_max_(weaknet_packet_max_items)
    all_reduce_sum_(weaknet_packet_le1)
    all_reduce_sum_(weaknet_packet_le2)
    all_reduce_sum_(weaknet_packet_le4)
    all_reduce_sum_(weaknet_packet_le8)
    all_reduce_sum_(weaknet_packet_gt8)
    all_reduce_sum_(weaknet_small_packets)
    all_reduce_sum_(weaknet_small_packet_items)
    all_reduce_sum_(weaknet_inter_sim_ms)
    all_reduce_max_(weaknet_inter_sim_ms_rankmax)
    all_reduce_sum_(kvb)
    all_reduce_sum_(skvs)
    all_reduce_sum_(rank_grouped_pair_entries)
    all_reduce_sum_(rank_grouped_packet_entries)
    all_reduce_sum_(rank_grouped_dispatch_calls)
    all_reduce_sum_(node_grouped_pair_entries)
    all_reduce_sum_(node_grouped_node_packet_entries)
    all_reduce_sum_(node_grouped_rank_packet_entries)
    all_reduce_sum_(node_grouped_dispatch_calls)
    all_reduce_sum_(node_grouped_pack_ms)
    all_reduce_sum_(node_grouped_metadata_ms)
    all_reduce_sum_(node_grouped_intra_fanout_ms)
    all_reduce_sum_(node_grouped_inter_send_ms)
    all_reduce_sum_(node_grouped_inter_send_weaknet_sleep_ms)
    all_reduce_sum_(node_grouped_inter_send_stage1_weaknet_sleep_ms)
    all_reduce_sum_(node_grouped_inter_send_stage2_weaknet_sleep_ms)
    all_reduce_sum_(node_grouped_inter_send_stage1_collective_ms)
    all_reduce_sum_(node_grouped_inter_send_stage2_collective_ms)
    all_reduce_sum_(node_grouped_metadata_exchange_ms)
    all_reduce_sum_(node_grouped_gather_ms)
    all_reduce_sum_(node_grouped_total_overhead_ms)
    all_reduce_sum_(node_grouped_same_node_packet_entries)
    all_reduce_sum_(node_grouped_cross_node_packet_entries)
    all_reduce_sum_(node_grouped_same_node_bypass_bytes)
    all_reduce_sum_(node_grouped_stage2_local_rank_packet_entries)
    all_reduce_sum_(node_grouped_stage2_remote_rank_packet_entries)
    all_reduce_sum_(node_grouped_stage2_local_bypass_bytes)
    all_reduce_sum_(node_grouped_metadata_collective_count)
    all_reduce_sum_(node_grouped_metadata_bytes)
    all_reduce_sum_(node_grouped_counts_build_ms)
    all_reduce_sum_(node_grouped_splits_build_ms)
    all_reduce_sum_(node_grouped_reverse_map_build_ms)
    all_reduce_sum_(node_grouped_rank_to_node_gather_ms)
    all_reduce_sum_(node_grouped_node_to_token_restore_ms)
    all_reduce_sum_(node_grouped_final_combine_ms)
    all_reduce_sum_(node_grouped_combine_intermediate_buffer_bytes)
    all_reduce_sum_(node_grouped_allocation_ms)
    all_reduce_sum_(node_grouped_allocation_count)
    all_reduce_sum_(node_grouped_allocation_bytes)
    all_reduce_sum_(node_grouped_workspace_reuse_hit)
    all_reduce_sum_(node_grouped_enabled_layers)
    all_reduce_sum_(node_grouped_fallback_layers)
    all_reduce_sum_(node_grouped_estimated_saving_ms)

    all_reduce_sum_(h2d_total)
    all_reduce_sum_(h2d_prefetch)
    all_reduce_sum_(h2d_demand)
    all_reduce_sum_(oracle_stage_h2d)
    all_reduce_sum_(oracle_stage_event_count)
    all_reduce_sum_(oracle_stage_ms)
    all_reduce_sum_(oracle_stage_prefill_ms)
    all_reduce_sum_(oracle_stage_decode_ms)
    all_reduce_sum_(moe_router_ms)
    all_reduce_sum_(moe_expert_ms)
    all_reduce_sum_(moe_shared_ms)
    all_reduce_max_(moe_router_ms_rankmax)
    all_reduce_max_(moe_expert_ms_rankmax)
    all_reduce_max_(moe_shared_ms_rankmax)
    all_reduce_sum_(moe_timing_calls)
    all_reduce_sum_(rs_mlp_ms)
    all_reduce_max_(rs_mlp_ms_rankmax)
    all_reduce_sum_(rs_mlp_calls)
    all_reduce_sum_(rs_next_layer_pred_calls)
    all_reduce_sum_(demand_h2d_exposed_ms)
    all_reduce_sum_(demand_h2d_exposed_prefill_ms)
    all_reduce_sum_(demand_h2d_exposed_decode_ms)
    all_reduce_sum_(sim_demand_h2d_exposed_ms)
    all_reduce_sum_(sim_demand_h2d_exposed_prefill_ms)
    all_reduce_sum_(sim_demand_h2d_exposed_decode_ms)
    all_reduce_max_(sim_demand_h2d_exposed_ms_rankmax)
    all_reduce_max_(sim_demand_h2d_exposed_decode_ms_rankmax)
    all_reduce_sum_(sim_h2d_extra_sleep_ms)
    all_reduce_max_(demand_h2d_exposed_ms_rankmax)
    all_reduce_max_(demand_h2d_exposed_prefill_ms_rankmax)
    all_reduce_max_(demand_h2d_exposed_decode_ms_rankmax)
    all_reduce_sum_(demand_h2d_event_count)
    all_reduce_sum_(prefetch_h2d_copy_ms)
    all_reduce_sum_(prefetch_h2d_event_count)
    all_reduce_sum_(prefetch_wait_ms)
    all_reduce_sum_(prefetch_wait_event_count)
    all_reduce_sum_(prefetch_oom_skip_count)
    all_reduce_sum_(plain_ep_executor_calls)
    all_reduce_sum_(epipe_executor_calls)
    all_reduce_sum_(epipe_executor_attempt_count)
    all_reduce_sum_(sim_expert_exec_ms)
    all_reduce_max_(sim_expert_exec_ms_rankmax)
    all_reduce_sum_(sim_compute_extra_sleep_ms)

    all_reduce_sum_(prefetch_hit)
    all_reduce_sum_(prefetch_wasted)
    all_reduce_sum_(cache_hit)
    all_reduce_sum_(cache_miss)
    all_reduce_sum_(rs_prefetch_eval_events)
    all_reduce_sum_(rs_prefetch_eval_pred_total)
    all_reduce_sum_(rs_prefetch_eval_actual_total)
    all_reduce_sum_(rs_prefetch_eval_overlap_total)
    all_reduce_sum_(prefetch_budget_zero_count)
    all_reduce_sum_(prefetch_budget_positive_count)
    all_reduce_sum_(prefetch_admitted_count)
    all_reduce_sum_(rs_pred_candidate_count)
    all_reduce_sum_(rs_pred_local_candidate_count)
    all_reduce_sum_(rs_pred_nonlocal_candidate_count)
    all_reduce_sum_(rs_candidate_filter_in_count)
    all_reduce_sum_(rs_candidate_filter_out_count)
    all_reduce_sum_(rs_candidate_filter_out_cached_count)
    all_reduce_sum_(rs_candidate_filter_out_low_score_count)
    all_reduce_sum_(rs_candidate_survived_count)
    all_reduce_sum_(prefetch_issued_count)
    all_reduce_sum_(prefetch_completed_count)
    all_reduce_sum_(prefetch_complete_before_use_count)
    all_reduce_sum_(prefetch_rejected_active_replace)
    all_reduce_sum_(prefetch_rejected_margin)
    all_reduce_sum_(prefetch_rejected_cache_full)
    all_reduce_sum_(prefetch_rejected_low_score)
    all_reduce_sum_(prefill_scp_transient_load_count)
    all_reduce_sum_(prefill_scp_transient_hit_count)
    all_reduce_sum_(prefill_scp_transient_promote_count)
    all_reduce_sum_(prefill_scp_transient_evict_count)
    all_reduce_sum_(prefill_scp_transient_resident_at_decode_start)
    all_reduce_sum_(prefill_scp_transient_demote_count)
    all_reduce_sum_(prefill_scp_transient_resident_at_decode_end)
    all_reduce_sum_(resolver_decode_total_queries)
    all_reduce_sum_(resolver_decode_rr_owner_count)
    all_reduce_sum_(resolver_decode_scp_replica_selected_count)
    all_reduce_sum_(resolver_decode_rr_fallback_count)
    all_reduce_sum_(resolver_decode_replica_reject_not_ready_count)
    all_reduce_sum_(resolver_decode_replica_reject_no_comm_gain_count)
    all_reduce_sum_(resolver_decode_replica_cache_ready_count)
    all_reduce_sum_(resolver_decode_replica_prefetch_ready_count)
    all_reduce_sum_(resolver_decode_route_cost_before)
    all_reduce_sum_(resolver_decode_route_cost_after)
    all_reduce_sum_(scp_prefill_route_diag_items)
    all_reduce_sum_(scp_prefill_route_diag_replica_items)
    all_reduce_sum_(scp_prefill_route_diag_cross_items_before)
    all_reduce_sum_(scp_prefill_route_diag_cross_items_after)
    all_reduce_sum_(scp_prefill_route_diag_cross_events_before)
    all_reduce_sum_(scp_prefill_route_diag_cross_events_after)
    all_reduce_sum_(scp_prefill_route_diag_cost_before)
    all_reduce_sum_(scp_prefill_route_diag_cost_after)
    all_reduce_sum_(scp_decode_route_diag_items)
    all_reduce_sum_(scp_decode_route_diag_replica_items)
    all_reduce_sum_(scp_decode_route_diag_cross_items_before)
    all_reduce_sum_(scp_decode_route_diag_cross_items_after)
    all_reduce_sum_(scp_decode_route_diag_cross_events_before)
    all_reduce_sum_(scp_decode_route_diag_cross_events_after)
    all_reduce_sum_(scp_decode_route_diag_cost_before)
    all_reduce_sum_(scp_decode_route_diag_cost_after)
    all_reduce_sum_(scp_table_local_replica_route_count)
    all_reduce_sum_(scp_table_rr_fallback_count)
    all_reduce_sum_(scp_table_cross_node_route_count)
    all_reduce_sum_(scp_table_same_node_route_count)
    all_reduce_sum_(scp_route_cost_before_sum)
    all_reduce_sum_(scp_route_cost_after_sum)
    all_reduce_sum_(scp_lookup_calls)
    all_reduce_sum_(scp_lookup_overhead_ms)
    all_reduce_sum_(rs_prefetch_target_resolver_match_count)
    all_reduce_sum_(rs_prefetch_target_resolver_mismatch_count)
    all_reduce_sum_(rs_prefetch_would_be_wrong_without_resolver_count)
    all_reduce_sum_(rs_scp_target_mismatch_count)
    all_reduce_sum_(ttl_prevented_evictions)
    all_reduce_sum_(cache_eviction_fallback_all_protected)
    all_reduce_sum_(warmup_loaded_experts)
    all_reduce_sum_(warmup_skipped_resident)
    all_reduce_sum_(warmup_h2d)
    all_reduce_sum_(warmup_time_ms)
    all_reduce_sum_(transition_lookup_calls)
    all_reduce_sum_(transition_lookup_overhead_ms)
    all_reduce_sum_(transition_prefetch_admitted)
    all_reduce_sum_(transition_prefetch_hit)
    all_reduce_sum_(transition_prefetch_late)
    all_reduce_sum_(transition_prefetch_wasted)
    all_reduce_sum_(transition_safe_evict_denied)
    all_reduce_sum_(transition_no_replace_denied)
    for t_epipe in epipe_tensors.values():
        all_reduce_sum_(t_epipe)

    all_reduce_sum_(owner_group_count)
    all_reduce_sum_(owner_changed_samples)
    all_reduce_sum_(owner_empty_ranks)
    all_reduce_sum_(owner_hist_max_sum)
    all_reduce_sum_(owner_hist_min_sum)
    all_reduce_sum_(owner_hist_cv_sum)
    all_reduce_sum_(owner_imbalance_ratio_sum)

    all_reduce_sum_(owner_group_reconstruct_ms)
    all_reduce_sum_(owner_group_plan_ms)
    all_reduce_sum_(owner_group_migrate_ms)

    all_reduce_sum_(owner_attention_ms_sum)
    all_reduce_max_(owner_attention_ms_max)
    all_reduce_sum_(owner_attention_calls)

    all_reduce_sum_(owner_moe_ms_sum)
    all_reduce_max_(owner_moe_ms_max)
    all_reduce_sum_(owner_moe_calls)

    # max wall / memory
    all_reduce_max_(prefill_wall)
    all_reduce_max_(decode_wall)
    all_reduce_max_(ttft_wall)
    all_reduce_max_(total_wall)

    all_reduce_max_(peak_alloc)
    all_reduce_max_(peak_rsvd)

    # ---------------- per-layer reduce ----------------
    per_layer_records: Dict[int, Dict[str, float]] = {}
    if metrics.per_layer is not None and args.collect_per_layer:
        E = float(num_experts)
        for l, lp in metrics.per_layer.items():
            hist = getattr(lp, "expert_hist", None)
            if hist is None:
                hist = torch.zeros((num_experts,), device=device, dtype=torch.float64)
            else:
                hist = hist.to(device=device, dtype=torch.float64).clone()

            a2a_bytes = torch.tensor([lp.a2a_send_bytes], device=device, dtype=torch.int64)
            a2a_remote_bytes = torch.tensor([getattr(lp, "a2a_remote_bytes", 0)], device=device, dtype=torch.int64)
            a2a_intra_node_bytes = torch.tensor([getattr(lp, "activation_intra_node_bytes", 0)], device=device, dtype=torch.int64)
            a2a_inter_node_bytes = torch.tensor([getattr(lp, "activation_inter_node_bytes", 0)], device=device, dtype=torch.int64)
            weaknet_intra_items_l = torch.tensor([getattr(lp, "weaknet_intra_node_item_count", 0)], device=device, dtype=torch.int64)
            weaknet_inter_items_l = torch.tensor([getattr(lp, "weaknet_inter_node_item_count", 0)], device=device, dtype=torch.int64)
            weaknet_intra_transfers_l = torch.tensor([getattr(lp, "weaknet_intra_node_transfer_count", 0)], device=device, dtype=torch.int64)
            weaknet_inter_transfers_l = torch.tensor([getattr(lp, "weaknet_inter_node_transfer_count", 0)], device=device, dtype=torch.int64)
            weaknet_packet_sumsq_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_item_sumsq", 0)], device=device, dtype=torch.int64)
            weaknet_packet_max_items_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_max_items", 0)], device=device, dtype=torch.int64)
            weaknet_packet_le1_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_le1_count", 0)], device=device, dtype=torch.int64)
            weaknet_packet_le2_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_le2_count", 0)], device=device, dtype=torch.int64)
            weaknet_packet_le4_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_le4_count", 0)], device=device, dtype=torch.int64)
            weaknet_packet_le8_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_le8_count", 0)], device=device, dtype=torch.int64)
            weaknet_packet_gt8_l = torch.tensor([getattr(lp, "weaknet_inter_node_packet_gt8_count", 0)], device=device, dtype=torch.int64)
            weaknet_small_packets_l = torch.tensor([getattr(lp, "weaknet_inter_node_small_packet_count", 0)], device=device, dtype=torch.int64)
            weaknet_small_packet_items_l = torch.tensor([getattr(lp, "weaknet_inter_node_small_packet_item_count", 0)], device=device, dtype=torch.int64)
            weaknet_inter_sim_ms_sum_l = torch.tensor([float(getattr(lp, "weaknet_inter_node_sim_ms", 0.0))], device=device, dtype=torch.float64)
            weaknet_inter_sim_ms_max_l = torch.tensor([float(getattr(lp, "weaknet_inter_node_sim_ms", 0.0))], device=device, dtype=torch.float64)
            h2d_bytes = torch.tensor([lp.h2d_bytes], device=device, dtype=torch.int64)

            local_assign_t = torch.tensor([int(lp.local_assignments)], device=device, dtype=torch.int64)
            remote_assign_t = torch.tensor([int(lp.remote_assignments)], device=device, dtype=torch.int64)
            dispatch_self_t = torch.tensor([int(getattr(lp, "dispatch_self_assignments", 0))], device=device, dtype=torch.int64)
            dispatch_remote_t = torch.tensor([int(getattr(lp, "dispatch_remote_assignments", 0))], device=device, dtype=torch.int64)
            h2d_demand_bytes = torch.tensor([int(getattr(lp, "h2d_demand_bytes", 0))], device=device, dtype=torch.int64)
            h2d_prefetch_bytes = torch.tensor([int(getattr(lp, "h2d_prefetch_bytes", 0))], device=device, dtype=torch.int64)
            cache_hit_t = torch.tensor([int(getattr(lp, "cache_hit_count", 0))], device=device, dtype=torch.int64)
            cache_miss_t = torch.tensor([int(getattr(lp, "cache_miss_count", 0))], device=device, dtype=torch.int64)
            prefetch_hit_t = torch.tensor([int(getattr(lp, "prefetch_hit_count", 0))], device=device, dtype=torch.int64)
            prefetch_wasted_t = torch.tensor([int(getattr(lp, "prefetch_wasted_count", 0))], device=device, dtype=torch.int64)
            prefetch_admitted_t = torch.tensor([int(getattr(lp, "prefetch_admitted_count", 0))], device=device, dtype=torch.int64)
            rs_pred_candidate_t = torch.tensor([int(getattr(lp, "rs_pred_candidate_count", 0))], device=device, dtype=torch.int64)
            rs_pred_local_candidate_t = torch.tensor([int(getattr(lp, "rs_pred_local_candidate_count", 0))], device=device, dtype=torch.int64)
            rs_pred_nonlocal_candidate_t = torch.tensor([int(getattr(lp, "rs_pred_nonlocal_candidate_count", 0))], device=device, dtype=torch.int64)
            rs_candidate_filter_in_t = torch.tensor([int(getattr(lp, "rs_candidate_filter_in_count", 0))], device=device, dtype=torch.int64)
            rs_candidate_filter_out_t = torch.tensor([int(getattr(lp, "rs_candidate_filter_out_count", 0))], device=device, dtype=torch.int64)
            rs_candidate_filter_out_cached_t = torch.tensor([int(getattr(lp, "rs_candidate_filter_out_cached_count", 0))], device=device, dtype=torch.int64)
            rs_candidate_filter_out_low_score_t = torch.tensor([int(getattr(lp, "rs_candidate_filter_out_low_score_count", 0))], device=device, dtype=torch.int64)
            rs_candidate_survived_t = torch.tensor([int(getattr(lp, "rs_candidate_survived_count", 0))], device=device, dtype=torch.int64)
            prefetch_issued_t = torch.tensor([int(getattr(lp, "prefetch_issued_count", 0))], device=device, dtype=torch.int64)
            prefetch_completed_t = torch.tensor([int(getattr(lp, "prefetch_completed_count", 0))], device=device, dtype=torch.int64)
            prefetch_complete_before_use_t = torch.tensor([int(getattr(lp, "prefetch_complete_before_use_count", 0))], device=device, dtype=torch.int64)
            prefetch_rejected_active_replace_t = torch.tensor([int(getattr(lp, "prefetch_rejected_active_replace", 0))], device=device, dtype=torch.int64)
            prefetch_rejected_margin_t = torch.tensor([int(getattr(lp, "prefetch_rejected_margin", 0))], device=device, dtype=torch.int64)
            prefetch_rejected_cache_full_t = torch.tensor([int(getattr(lp, "prefetch_rejected_cache_full", 0))], device=device, dtype=torch.int64)
            prefetch_rejected_low_score_t = torch.tensor([int(getattr(lp, "prefetch_rejected_low_score", 0))], device=device, dtype=torch.int64)
            demand_h2d_ms_sum_t = torch.tensor([float(getattr(lp, "demand_h2d_exposed_ms", 0.0))], device=device, dtype=torch.float64)
            demand_h2d_decode_ms_sum_t = torch.tensor([float(getattr(lp, "demand_h2d_exposed_decode_ms", 0.0))], device=device, dtype=torch.float64)
            sim_demand_h2d_ms_sum_t = torch.tensor([float(getattr(lp, "sim_demand_h2d_exposed_ms", 0.0))], device=device, dtype=torch.float64)
            sim_demand_h2d_decode_ms_sum_t = torch.tensor([float(getattr(lp, "sim_demand_h2d_exposed_decode_ms", 0.0))], device=device, dtype=torch.float64)
            sim_demand_h2d_ms_max_t = torch.tensor([float(getattr(lp, "sim_demand_h2d_exposed_ms", 0.0))], device=device, dtype=torch.float64)
            sim_demand_h2d_decode_ms_max_t = torch.tensor([float(getattr(lp, "sim_demand_h2d_exposed_decode_ms", 0.0))], device=device, dtype=torch.float64)
            sim_h2d_extra_sleep_ms_sum_t = torch.tensor([float(getattr(lp, "sim_h2d_extra_sleep_ms", 0.0))], device=device, dtype=torch.float64)
            demand_h2d_ms_max_t = torch.tensor([float(getattr(lp, "demand_h2d_exposed_ms", 0.0))], device=device, dtype=torch.float64)
            demand_h2d_decode_ms_max_t = torch.tensor([float(getattr(lp, "demand_h2d_exposed_decode_ms", 0.0))], device=device, dtype=torch.float64)
            demand_h2d_events_t = torch.tensor([int(getattr(lp, "demand_h2d_event_count", 0))], device=device, dtype=torch.int64)
            prefetch_h2d_ms_sum_t = torch.tensor([float(getattr(lp, "prefetch_h2d_copy_ms", 0.0))], device=device, dtype=torch.float64)
            prefetch_wait_ms_sum_t = torch.tensor([float(getattr(lp, "prefetch_wait_ms", 0.0))], device=device, dtype=torch.float64)
            comm_ms_sum_t = torch.tensor([float(getattr(lp, "comm_ms", 0.0))], device=device, dtype=torch.float64)
            comm_ms_max_t = torch.tensor([float(getattr(lp, "comm_ms", 0.0))], device=device, dtype=torch.float64)
            expert_exec_ms_sum_t = torch.tensor([float(getattr(lp, "expert_exec_ms", 0.0))], device=device, dtype=torch.float64)
            expert_exec_ms_max_t = torch.tensor([float(getattr(lp, "expert_exec_ms", 0.0))], device=device, dtype=torch.float64)
            sim_expert_exec_ms_sum_t = torch.tensor([float(getattr(lp, "sim_expert_exec_ms", 0.0))], device=device, dtype=torch.float64)
            sim_expert_exec_ms_max_t = torch.tensor([float(getattr(lp, "sim_expert_exec_ms", 0.0))], device=device, dtype=torch.float64)
            sim_compute_extra_sleep_ms_sum_t = torch.tensor([float(getattr(lp, "sim_compute_extra_sleep_ms", 0.0))], device=device, dtype=torch.float64)
            grouping_ms_sum_t = torch.tensor([float(getattr(lp, "grouping_overhead_ms", 0.0))], device=device, dtype=torch.float64)
            grouping_ms_max_t = torch.tensor([float(getattr(lp, "grouping_overhead_ms", 0.0))], device=device, dtype=torch.float64)
            rs_mlp_ms_sum_t = torch.tensor([float(getattr(lp, "rs_mlp_ms", 0.0))], device=device, dtype=torch.float64)
            rs_mlp_ms_max_t = torch.tensor([float(getattr(lp, "rs_mlp_ms", 0.0))], device=device, dtype=torch.float64)
            runtime_overhead_ms_sum_t = torch.tensor([float(getattr(lp, "runtime_overhead_ms", 0.0))], device=device, dtype=torch.float64)
            runtime_overhead_ms_max_t = torch.tensor([float(getattr(lp, "runtime_overhead_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_pack_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_pack_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_metadata_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_metadata_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_intra_fanout_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_intra_fanout_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_inter_send_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_inter_send_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_inter_send_weaknet_sleep_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_inter_send_weaknet_sleep_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_inter_send_stage1_weaknet_sleep_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_inter_send_stage1_weaknet_sleep_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_inter_send_stage2_weaknet_sleep_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_inter_send_stage2_weaknet_sleep_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_inter_send_stage1_collective_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_inter_send_stage1_collective_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_inter_send_stage2_collective_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_inter_send_stage2_collective_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_metadata_exchange_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_metadata_exchange_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_gather_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_gather_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_total_overhead_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_total_overhead_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_total_overhead_ms_max_t = torch.tensor([float(getattr(lp, "node_grouped_total_overhead_ms", 0.0))], device=device, dtype=torch.float64)
            node_grouped_enabled_count_t = torch.tensor([int(getattr(lp, "node_grouped_enabled_count", 0))], device=device, dtype=torch.int64)
            node_grouped_fallback_count_t = torch.tensor([int(getattr(lp, "node_grouped_fallback_count", 0))], device=device, dtype=torch.int64)
            node_grouped_estimated_saving_ms_sum_t = torch.tensor([float(getattr(lp, "node_grouped_estimated_saving_ms", 0.0))], device=device, dtype=torch.float64)

            all_reduce_sum_(hist)
            all_reduce_sum_(a2a_bytes)
            all_reduce_sum_(a2a_remote_bytes)
            all_reduce_sum_(a2a_intra_node_bytes)
            all_reduce_sum_(a2a_inter_node_bytes)
            all_reduce_sum_(weaknet_intra_items_l)
            all_reduce_sum_(weaknet_inter_items_l)
            all_reduce_sum_(weaknet_intra_transfers_l)
            all_reduce_sum_(weaknet_inter_transfers_l)
            all_reduce_sum_(weaknet_packet_sumsq_l)
            all_reduce_max_(weaknet_packet_max_items_l)
            all_reduce_sum_(weaknet_packet_le1_l)
            all_reduce_sum_(weaknet_packet_le2_l)
            all_reduce_sum_(weaknet_packet_le4_l)
            all_reduce_sum_(weaknet_packet_le8_l)
            all_reduce_sum_(weaknet_packet_gt8_l)
            all_reduce_sum_(weaknet_small_packets_l)
            all_reduce_sum_(weaknet_small_packet_items_l)
            all_reduce_sum_(weaknet_inter_sim_ms_sum_l)
            all_reduce_max_(weaknet_inter_sim_ms_max_l)
            all_reduce_sum_(h2d_bytes)
            all_reduce_sum_(h2d_demand_bytes)
            all_reduce_sum_(h2d_prefetch_bytes)
            all_reduce_sum_(local_assign_t)
            all_reduce_sum_(remote_assign_t)
            all_reduce_sum_(dispatch_self_t)
            all_reduce_sum_(dispatch_remote_t)
            all_reduce_sum_(cache_hit_t)
            all_reduce_sum_(cache_miss_t)
            all_reduce_sum_(prefetch_hit_t)
            all_reduce_sum_(prefetch_wasted_t)
            all_reduce_sum_(prefetch_admitted_t)
            all_reduce_sum_(rs_pred_candidate_t)
            all_reduce_sum_(rs_pred_local_candidate_t)
            all_reduce_sum_(rs_pred_nonlocal_candidate_t)
            all_reduce_sum_(rs_candidate_filter_in_t)
            all_reduce_sum_(rs_candidate_filter_out_t)
            all_reduce_sum_(rs_candidate_filter_out_cached_t)
            all_reduce_sum_(rs_candidate_filter_out_low_score_t)
            all_reduce_sum_(rs_candidate_survived_t)
            all_reduce_sum_(prefetch_issued_t)
            all_reduce_sum_(prefetch_completed_t)
            all_reduce_sum_(prefetch_complete_before_use_t)
            all_reduce_sum_(prefetch_rejected_active_replace_t)
            all_reduce_sum_(prefetch_rejected_margin_t)
            all_reduce_sum_(prefetch_rejected_cache_full_t)
            all_reduce_sum_(prefetch_rejected_low_score_t)
            all_reduce_sum_(demand_h2d_ms_sum_t)
            all_reduce_sum_(demand_h2d_decode_ms_sum_t)
            all_reduce_sum_(sim_demand_h2d_ms_sum_t)
            all_reduce_sum_(sim_demand_h2d_decode_ms_sum_t)
            all_reduce_max_(sim_demand_h2d_ms_max_t)
            all_reduce_max_(sim_demand_h2d_decode_ms_max_t)
            all_reduce_sum_(sim_h2d_extra_sleep_ms_sum_t)
            all_reduce_max_(demand_h2d_ms_max_t)
            all_reduce_max_(demand_h2d_decode_ms_max_t)
            all_reduce_sum_(demand_h2d_events_t)
            all_reduce_sum_(prefetch_h2d_ms_sum_t)
            all_reduce_sum_(prefetch_wait_ms_sum_t)
            all_reduce_sum_(comm_ms_sum_t)
            all_reduce_max_(comm_ms_max_t)
            all_reduce_sum_(expert_exec_ms_sum_t)
            all_reduce_max_(expert_exec_ms_max_t)
            all_reduce_sum_(sim_expert_exec_ms_sum_t)
            all_reduce_max_(sim_expert_exec_ms_max_t)
            all_reduce_sum_(sim_compute_extra_sleep_ms_sum_t)
            all_reduce_sum_(grouping_ms_sum_t)
            all_reduce_max_(grouping_ms_max_t)
            all_reduce_sum_(rs_mlp_ms_sum_t)
            all_reduce_max_(rs_mlp_ms_max_t)
            all_reduce_sum_(runtime_overhead_ms_sum_t)
            all_reduce_max_(runtime_overhead_ms_max_t)
            all_reduce_sum_(node_grouped_pack_ms_sum_t)
            all_reduce_sum_(node_grouped_metadata_ms_sum_t)
            all_reduce_sum_(node_grouped_intra_fanout_ms_sum_t)
            all_reduce_sum_(node_grouped_inter_send_ms_sum_t)
            all_reduce_sum_(node_grouped_inter_send_weaknet_sleep_ms_sum_t)
            all_reduce_sum_(node_grouped_inter_send_stage1_weaknet_sleep_ms_sum_t)
            all_reduce_sum_(node_grouped_inter_send_stage2_weaknet_sleep_ms_sum_t)
            all_reduce_sum_(node_grouped_inter_send_stage1_collective_ms_sum_t)
            all_reduce_sum_(node_grouped_inter_send_stage2_collective_ms_sum_t)
            all_reduce_sum_(node_grouped_metadata_exchange_ms_sum_t)
            all_reduce_sum_(node_grouped_gather_ms_sum_t)
            all_reduce_sum_(node_grouped_total_overhead_ms_sum_t)
            all_reduce_max_(node_grouped_total_overhead_ms_max_t)
            all_reduce_sum_(node_grouped_enabled_count_t)
            all_reduce_sum_(node_grouped_fallback_count_t)
            all_reduce_sum_(node_grouped_estimated_saving_ms_sum_t)

            local_assign_i = int(local_assign_t.item())
            remote_assign_i = int(remote_assign_t.item())

            mean = float(hist.mean().item())
            mx = float(hist.max().item())
            var = float(((hist - mean) ** 2).mean().item())
            cv = (math.sqrt(var) / mean) if mean > 0 else 0.0

            total_assign_i = local_assign_i + remote_assign_i
            local_ratio = (local_assign_i / total_assign_i) if total_assign_i > 0 else 0.0
            remote_ratio = (remote_assign_i / total_assign_i) if total_assign_i > 0 else 0.0
            dispatch_self_i = int(dispatch_self_t.item())
            dispatch_remote_i = int(dispatch_remote_t.item())
            dispatch_total_i = dispatch_self_i + dispatch_remote_i
            dispatch_remote_ratio = (dispatch_remote_i / dispatch_total_i) if dispatch_total_i > 0 else 0.0

            placement_hist = []
            try:
                blk = SparseContext.layer_blocks.get(int(l), None)
                if blk is not None:
                    placement_hist = torch.bincount(
                        blk.expert_to_rank.detach().to("cpu", dtype=torch.long),
                        minlength=int(ep_size),
                    ).tolist()
            except Exception:
                placement_hist = []

            per_layer_records[int(l)] = {
                "placement_rank_hist": placement_hist,
                "placement_empty_rank_count": int(sum(1 for x in placement_hist if int(x) == 0)) if placement_hist else 0,
                "expert_mean": mean,
                "expert_max": float(mx),
                "expert_cv": float(cv),
                "a2a_mb": float(a2a_bytes.item()) / 1024**2,
                "a2a_remote_mb": float(a2a_remote_bytes.item()) / 1024**2,
                "activation_intra_node_mb": float(a2a_intra_node_bytes.item()) / 1024**2,
                "activation_inter_node_mb": float(a2a_inter_node_bytes.item()) / 1024**2,
                "weaknet_intra_node_item_count": int(weaknet_intra_items_l.item()),
                "weaknet_inter_node_item_count": int(weaknet_inter_items_l.item()),
                "weaknet_intra_node_transfer_count": int(weaknet_intra_transfers_l.item()),
                "weaknet_inter_node_transfer_count": int(weaknet_inter_transfers_l.item()),
                "weaknet_inter_node_packet_item_sumsq": int(weaknet_packet_sumsq_l.item()),
                "weaknet_inter_node_packet_max_items": int(weaknet_packet_max_items_l.item()),
                "weaknet_inter_node_packet_avg_items": (
                    float(weaknet_inter_items_l.item()) / max(float(weaknet_inter_transfers_l.item()), 1.0)
                ),
                "weaknet_inter_node_packet_le1_count": int(weaknet_packet_le1_l.item()),
                "weaknet_inter_node_packet_le2_count": int(weaknet_packet_le2_l.item()),
                "weaknet_inter_node_packet_le4_count": int(weaknet_packet_le4_l.item()),
                "weaknet_inter_node_packet_le8_count": int(weaknet_packet_le8_l.item()),
                "weaknet_inter_node_packet_gt8_count": int(weaknet_packet_gt8_l.item()),
                "weaknet_inter_node_small_packet_count": int(weaknet_small_packets_l.item()),
                "weaknet_inter_node_small_packet_item_count": int(weaknet_small_packet_items_l.item()),
                "weaknet_inter_node_sim_ms_sum": float(weaknet_inter_sim_ms_sum_l.item()),
                "weaknet_inter_node_sim_ms_rankmax": float(weaknet_inter_sim_ms_max_l.item()),
                "h2d_mb": float(h2d_bytes.item()) / 1024**2,
                "h2d_demand_mb": float(h2d_demand_bytes.item()) / 1024**2,
                "h2d_prefetch_mb": float(h2d_prefetch_bytes.item()) / 1024**2,
                "demand_h2d_exposed_ms_sum": float(demand_h2d_ms_sum_t.item()),
                "demand_h2d_exposed_decode_ms_sum": float(demand_h2d_decode_ms_sum_t.item()),
                "sim_demand_h2d_exposed_ms_sum": float(sim_demand_h2d_ms_sum_t.item()),
                "sim_demand_h2d_exposed_decode_ms_sum": float(sim_demand_h2d_decode_ms_sum_t.item()),
                "sim_demand_h2d_exposed_ms_rankmax": float(sim_demand_h2d_ms_max_t.item()),
                "sim_demand_h2d_exposed_decode_ms_rankmax": float(sim_demand_h2d_decode_ms_max_t.item()),
                "sim_h2d_extra_sleep_ms_sum": float(sim_h2d_extra_sleep_ms_sum_t.item()),
                "demand_h2d_exposed_ms_rankmax": float(demand_h2d_ms_max_t.item()),
                "demand_h2d_exposed_decode_ms_rankmax": float(demand_h2d_decode_ms_max_t.item()),
                "demand_h2d_event_count": int(demand_h2d_events_t.item()),
                "prefetch_h2d_copy_ms_sum": float(prefetch_h2d_ms_sum_t.item()),
                "prefetch_wait_ms_sum": float(prefetch_wait_ms_sum_t.item()),
                "cache_hit_count": int(cache_hit_t.item()),
                "cache_miss_count": int(cache_miss_t.item()),
                "cache_hit_rate": float(cache_hit_t.item()) / max(float(cache_hit_t.item() + cache_miss_t.item()), 1.0),
                "prefetch_hit_count": int(prefetch_hit_t.item()),
                "prefetch_wasted_count": int(prefetch_wasted_t.item()),
                "prefetch_use_rate": float(prefetch_hit_t.item()) / max(float(prefetch_hit_t.item() + prefetch_wasted_t.item()), 1.0),
                "prefetch_admitted_count": int(prefetch_admitted_t.item()),
                "rs_pred_candidate_count": int(rs_pred_candidate_t.item()),
                "rs_pred_local_candidate_count": int(rs_pred_local_candidate_t.item()),
                "rs_pred_nonlocal_candidate_count": int(rs_pred_nonlocal_candidate_t.item()),
                "rs_candidate_filter_in_count": int(rs_candidate_filter_in_t.item()),
                "rs_candidate_filter_out_count": int(rs_candidate_filter_out_t.item()),
                "rs_candidate_filter_out_cached_count": int(rs_candidate_filter_out_cached_t.item()),
                "rs_candidate_filter_out_low_score_count": int(rs_candidate_filter_out_low_score_t.item()),
                "rs_candidate_survived_count": int(rs_candidate_survived_t.item()),
                "rs_candidate_survival_rate": float(rs_candidate_survived_t.item()) / max(float(rs_candidate_filter_in_t.item()), 1.0),
                "prefetch_issued_count": int(prefetch_issued_t.item()),
                "prefetch_completed_count": int(prefetch_completed_t.item()),
                "prefetch_complete_before_use_count": int(prefetch_complete_before_use_t.item()),
                "prefetch_completion_rate": float(prefetch_completed_t.item()) / max(float(prefetch_issued_t.item()), 1.0),
                "prefetch_complete_before_use_rate": float(prefetch_complete_before_use_t.item()) / max(float(prefetch_hit_t.item()), 1.0),
                "prefetch_rejected_active_replace": int(prefetch_rejected_active_replace_t.item()),
                "prefetch_rejected_margin": int(prefetch_rejected_margin_t.item()),
                "prefetch_rejected_cache_full": int(prefetch_rejected_cache_full_t.item()),
                "prefetch_rejected_low_score": int(prefetch_rejected_low_score_t.item()),
                "comm_ms_sum": float(comm_ms_sum_t.item()),
                "comm_ms_rankmax": float(comm_ms_max_t.item()),
                "expert_exec_ms_sum": float(expert_exec_ms_sum_t.item()),
                "expert_exec_ms_rankmax": float(expert_exec_ms_max_t.item()),
                "sim_expert_exec_ms_sum": float(sim_expert_exec_ms_sum_t.item()),
                "sim_expert_exec_ms_rankmax": float(sim_expert_exec_ms_max_t.item()),
                "sim_compute_extra_sleep_ms_sum": float(sim_compute_extra_sleep_ms_sum_t.item()),
                "grouping_overhead_ms_sum": float(grouping_ms_sum_t.item()),
                "grouping_overhead_ms_rankmax": float(grouping_ms_max_t.item()),
                "rs_mlp_ms_sum": float(rs_mlp_ms_sum_t.item()),
                "rs_mlp_ms_rankmax": float(rs_mlp_ms_max_t.item()),
                "runtime_overhead_ms_sum": float(runtime_overhead_ms_sum_t.item()),
                "runtime_overhead_ms_rankmax": float(runtime_overhead_ms_max_t.item()),
                "node_grouped_pack_ms_sum": float(node_grouped_pack_ms_sum_t.item()),
                "node_grouped_metadata_ms_sum": float(node_grouped_metadata_ms_sum_t.item()),
                "node_grouped_intra_fanout_ms_sum": float(node_grouped_intra_fanout_ms_sum_t.item()),
                "node_grouped_inter_send_ms_sum": float(node_grouped_inter_send_ms_sum_t.item()),
                "node_grouped_inter_send_weaknet_sleep_ms_sum": float(node_grouped_inter_send_weaknet_sleep_ms_sum_t.item()),
                "node_grouped_inter_send_stage1_weaknet_sleep_ms_sum": float(node_grouped_inter_send_stage1_weaknet_sleep_ms_sum_t.item()),
                "node_grouped_inter_send_stage2_weaknet_sleep_ms_sum": float(node_grouped_inter_send_stage2_weaknet_sleep_ms_sum_t.item()),
                "node_grouped_inter_send_stage1_non_sleep_ms_sum": max(0.0, float(node_grouped_inter_send_ms_sum_t.item()) - float(node_grouped_inter_send_stage1_weaknet_sleep_ms_sum_t.item())),
                "node_grouped_intra_fanout_stage2_non_sleep_ms_sum": max(0.0, float(node_grouped_intra_fanout_ms_sum_t.item()) - float(node_grouped_inter_send_stage2_weaknet_sleep_ms_sum_t.item())),
                "node_grouped_total_overhead_non_sleep_ms_sum": max(0.0, float(node_grouped_total_overhead_ms_sum_t.item()) - float(node_grouped_inter_send_weaknet_sleep_ms_sum_t.item())),
                "node_grouped_inter_send_stage1_collective_ms_sum": float(node_grouped_inter_send_stage1_collective_ms_sum_t.item()),
                "node_grouped_inter_send_stage2_collective_ms_sum": float(node_grouped_inter_send_stage2_collective_ms_sum_t.item()),
                "node_grouped_metadata_exchange_ms_sum": float(node_grouped_metadata_exchange_ms_sum_t.item()),
                "node_grouped_inter_send_non_sleep_ms_sum": max(0.0, float(node_grouped_inter_send_ms_sum_t.item()) - float(node_grouped_inter_send_stage1_weaknet_sleep_ms_sum_t.item())),
                "node_grouped_gather_ms_sum": float(node_grouped_gather_ms_sum_t.item()),
                "node_grouped_total_overhead_ms_sum": float(node_grouped_total_overhead_ms_sum_t.item()),
                "node_grouped_total_overhead_ms_rankmax": float(node_grouped_total_overhead_ms_max_t.item()),
                "node_grouped_enabled_count": int(node_grouped_enabled_count_t.item()),
                "node_grouped_fallback_count": int(node_grouped_fallback_count_t.item()),
                "node_grouped_estimated_saving_ms_sum": float(node_grouped_estimated_saving_ms_sum_t.item()),
                "dispatch_self_assignments": dispatch_self_i,
                "dispatch_remote_assignments": dispatch_remote_i,
                "dispatch_remote_ratio": float(dispatch_remote_ratio),
                "local_assignments": local_assign_i,
                "remote_assignments": remote_assign_i,
                "local_assignment_ratio": float(local_ratio),
                "remote_assignment_ratio": float(remote_ratio),
            }

    owner_group_count_i = int(owner_group_count.item())
    owner_attention_calls_i = int(owner_attention_calls.item())
    owner_moe_calls_i = int(owner_moe_calls.item())

    owner_local_summary = {
        "owner_group_count": owner_group_count_i,
        "owner_changed_samples": int(owner_changed_samples.item()),
        "owner_group_reconstruct_ms": float(owner_group_reconstruct_ms.item()),
        "owner_group_plan_ms": float(owner_group_plan_ms.item()),
        "owner_group_migrate_ms": float(owner_group_migrate_ms.item()),
        "owner_group_reconstruct_avg_ms": float(owner_group_reconstruct_ms.item()) / max(owner_group_count_i, 1),
        "owner_group_plan_avg_ms": float(owner_group_plan_ms.item()) / max(owner_group_count_i, 1),
        "owner_group_migrate_avg_ms": float(owner_group_migrate_ms.item()) / max(owner_group_count_i, 1),
        "owner_empty_ranks_sum": int(owner_empty_ranks.item()),
        "owner_empty_ranks_avg": float(owner_empty_ranks.item()) / max(owner_group_count_i, 1),
        "owner_hist_max_avg": float(owner_hist_max_sum.item()) / max(owner_group_count_i, 1),
        "owner_hist_min_avg": float(owner_hist_min_sum.item()) / max(owner_group_count_i, 1),
        "owner_hist_cv_avg": float(owner_hist_cv_sum.item()) / max(owner_group_count_i, 1),
        "owner_imbalance_ratio_avg": float(owner_imbalance_ratio_sum.item()) / max(owner_group_count_i, 1),
        "owner_attention_ms_sum": float(owner_attention_ms_sum.item()),
        "owner_attention_ms_rankmax": float(owner_attention_ms_max.item()),
        "owner_attention_calls": owner_attention_calls_i,
        "owner_attention_avg_ms_per_call": float(owner_attention_ms_sum.item()) / max(owner_attention_calls_i, 1),
        "owner_moe_ms_sum": float(owner_moe_ms_sum.item()),
        "owner_moe_ms_rankmax": float(owner_moe_ms_max.item()),
        "owner_moe_calls": owner_moe_calls_i,
        "owner_moe_avg_ms_per_call": float(owner_moe_ms_sum.item()) / max(owner_moe_calls_i, 1),
    }

    # ---------------- skvs debug summary ----------------
    skvs_dbg_local = _collect_skvs_debug_summary()

    def _sum_int_field(name: str) -> int:
        x = torch.tensor([int(skvs_dbg_local.get(name, 0))], device=device, dtype=torch.long)
        all_reduce_sum_(x)
        return int(x.item())

    def _max_int_field(name: str) -> int:
        x = torch.tensor([int(skvs_dbg_local.get(name, 0))], device=device, dtype=torch.long)
        all_reduce_max_(x)
        return int(x.item())

    skvs_dbg = {
        "num_prefill_fallback": _sum_int_field("num_prefill_fallback"),
        "num_decode_fallback": _sum_int_field("num_decode_fallback"),
        "num_ratio1_full": _sum_int_field("num_ratio1_full"),
        "num_layer0_full": _sum_int_field("num_layer0_full"),
        "num_local_sparse": _sum_int_field("num_local_sparse"),
        "num_local_full": _sum_int_field("num_local_full"),
        "num_guidance_refresh": _sum_int_field("num_guidance_refresh"),
        "num_guidance_reuse": _sum_int_field("num_guidance_reuse"),
        "num_guidance_start_suppressed": _sum_int_field("num_guidance_start_suppressed"),
        "num_guidance_interval_suppressed": _sum_int_field("num_guidance_interval_suppressed"),
        "forward_total_tokens": _sum_int_field("forward_total_tokens"),
        "forward_kept_tokens": _sum_int_field("forward_kept_tokens"),
        "forward_dropped_tokens": _sum_int_field("forward_dropped_tokens"),
        "forward_dense_ref_tokens": _sum_int_field("forward_dense_ref_tokens"),
        "forward_dense_kept_tokens": _sum_int_field("forward_dense_kept_tokens"),
        "forward_dense_dropped_tokens": _sum_int_field("forward_dense_dropped_tokens"),
        "forward_dropped_bytes": _sum_int_field("forward_dropped_bytes"),
        "local_selected_cached_tokens": _sum_int_field("local_selected_cached_tokens"),
        "local_selected_protected_tokens": _sum_int_field("local_selected_protected_tokens"),
        "local_selected_unprotected_tokens": _sum_int_field("local_selected_unprotected_tokens"),
        "local_selected_kept_unprotected_tokens": _sum_int_field("local_selected_kept_unprotected_tokens"),
        "last_plan_candidate_total_tokens": _sum_int_field("last_plan_candidate_total_tokens"),
        "last_plan_candidate_kept_tokens": _sum_int_field("last_plan_candidate_kept_tokens"),
        "last_plan_candidate_dropped_tokens": _sum_int_field("last_plan_candidate_dropped_tokens"),
        "last_plan_dense_ref_tokens": _sum_int_field("last_plan_dense_ref_tokens"),
        "last_plan_dense_kept_tokens": _sum_int_field("last_plan_dense_kept_tokens"),
        "last_plan_dense_dropped_tokens": _sum_int_field("last_plan_dense_dropped_tokens"),
        "global_keep_kv_len": _max_int_field("global_keep_kv_len"),
        "global_keep_force_dense": bool(_max_int_field("global_keep_force_dense")),
        "last_guidance_kv_len": _max_int_field("last_guidance_kv_len"),
        "last_guidance_refreshed": bool(_max_int_field("last_guidance_refreshed")),
        "decode_step_idx": _max_int_field("decode_step_idx"),
        "migration_selected_kv_bytes": _sum_int_field("migration_selected_kv_bytes"),
        "migration_full_kv_bytes": _sum_int_field("migration_full_kv_bytes"),
        "migration_tensor_payload_bytes": _sum_int_field("migration_tensor_payload_bytes"),
        "migration_tensor_wire_bytes": _sum_int_field("migration_tensor_wire_bytes"),
        "migration_impl_tensor_count": _sum_int_field("migration_impl_tensor_count"),
        "migration_impl_object_count": _sum_int_field("migration_impl_object_count"),
        "effective_mode_last": str(skvs_dbg_local.get("effective_mode_last", "unknown")),
    }
    for name in sorted(k for k in skvs_dbg_local.keys() if str(k).startswith("fallback_")):
        skvs_dbg[name] = _sum_int_field(name)
    skvs_dbg["avg_keep_ratio"] = float(skvs_dbg["forward_kept_tokens"]) / max(skvs_dbg["forward_total_tokens"], 1)
    skvs_dbg["avg_drop_ratio"] = float(skvs_dbg["forward_dropped_tokens"]) / max(skvs_dbg["forward_total_tokens"], 1)
    skvs_dbg["dense_equiv_keep_ratio"] = float(skvs_dbg["forward_dense_kept_tokens"]) / max(skvs_dbg["forward_dense_ref_tokens"], 1)
    skvs_dbg["dense_equiv_drop_ratio"] = float(skvs_dbg["forward_dense_dropped_tokens"]) / max(skvs_dbg["forward_dense_ref_tokens"], 1)
    skvs_dbg["last_plan_keep_ratio"] = float(skvs_dbg["last_plan_candidate_kept_tokens"]) / max(skvs_dbg["last_plan_candidate_total_tokens"], 1)
    skvs_dbg["last_plan_drop_ratio"] = float(skvs_dbg["last_plan_candidate_dropped_tokens"]) / max(skvs_dbg["last_plan_candidate_total_tokens"], 1)
    skvs_dbg["last_plan_dense_keep_ratio"] = float(skvs_dbg["last_plan_dense_kept_tokens"]) / max(skvs_dbg["last_plan_dense_ref_tokens"], 1)
    skvs_dbg["last_plan_dense_drop_ratio"] = float(skvs_dbg["last_plan_dense_dropped_tokens"]) / max(skvs_dbg["last_plan_dense_ref_tokens"], 1)
    skvs_dbg["migration_selected_kv_mb"] = float(skvs_dbg["migration_selected_kv_bytes"]) / 1024**2
    skvs_dbg["migration_full_kv_mb"] = float(skvs_dbg["migration_full_kv_bytes"]) / 1024**2
    skvs_dbg["migration_saved_ratio"] = (
        1.0 - float(skvs_dbg["migration_selected_kv_bytes"]) / float(skvs_dbg["migration_full_kv_bytes"])
        if int(skvs_dbg["migration_full_kv_bytes"]) > 0
        else 0.0
    )
    lu = int(skvs_dbg["local_selected_unprotected_tokens"])
    lku = int(skvs_dbg["local_selected_kept_unprotected_tokens"])
    skvs_dbg["local_selected_effective_ratio"] = float(lku) / max(lu, 1) if lu > 0 else 0.0
    nodepatch_dbg = _reduce_nodepatch_summary(device)
    task_local_local_meta = dict(getattr(SparseContext, "task_local_metadata", {}) or {})
    gathered_task_local_meta = None
    if dist.is_available() and dist.is_initialized():
        gathered_task_local_meta = [None for _ in range(env.world)] if env.rank == 0 else None
        dist.gather_object(task_local_local_meta, gathered_task_local_meta, dst=0)
    elif env.rank == 0:
        gathered_task_local_meta = [task_local_local_meta]

    lightweight_stats = _gather_lightweight_stats(env)
    if env.rank != 0:
        return None

    os.makedirs(args.out_dir, exist_ok=True)
    step_summary = metrics.decode_step_stats.summary(prefix="decode_step_")

    prefetch_hit_val = int(prefetch_hit.item())
    prefetch_wasted_val = int(prefetch_wasted.item())
    cache_hit_val = int(cache_hit.item())
    cache_miss_val = int(cache_miss.item())
    epipe_extra_cache_hit_val = int(epipe_tensors["epipe_prefill_extra_cache_hit_count"].item())
    main_cache_hit_val = max(0, cache_hit_val - epipe_extra_cache_hit_val)
    main_cache_miss_val = cache_miss_val + epipe_extra_cache_hit_val

    resident_access_total = cache_hit_val + cache_miss_val
    effective_service_hit_count = cache_hit_val + prefetch_hit_val
    effective_service_total = cache_hit_val + prefetch_hit_val + cache_miss_val
    prefetch_total_tagged = prefetch_hit_val + prefetch_wasted_val

    resident_hit_rate = (cache_hit_val / resident_access_total) if resident_access_total > 0 else 0.0
    main_cache_hit_rate = (main_cache_hit_val / resident_access_total) if resident_access_total > 0 else 0.0
    extra_cache_service_hit_rate = (
        epipe_extra_cache_hit_val / resident_access_total
        if resident_access_total > 0 else 0.0
    )
    main_cache_effective_service_hit_count = main_cache_hit_val + prefetch_hit_val
    effective_service_hit_rate = (
        effective_service_hit_count / effective_service_total
        if effective_service_total > 0 else 0.0
    )
    main_cache_effective_service_hit_rate = (
        main_cache_effective_service_hit_count / effective_service_total
        if effective_service_total > 0 else 0.0
    )
    prefetch_use_rate = (prefetch_hit_val / prefetch_total_tagged) if prefetch_total_tagged > 0 else 0.0
    rs_prefetch_pred_total_val = int(rs_prefetch_eval_pred_total.item())
    rs_prefetch_actual_total_val = int(rs_prefetch_eval_actual_total.item())
    rs_prefetch_overlap_total_val = int(rs_prefetch_eval_overlap_total.item())
    rs_prefetch_precision = (
        rs_prefetch_overlap_total_val / rs_prefetch_pred_total_val
        if rs_prefetch_pred_total_val > 0 else 0.0
    )
    rs_prefetch_recall = (
        rs_prefetch_overlap_total_val / rs_prefetch_actual_total_val
        if rs_prefetch_actual_total_val > 0 else 0.0
    )
    prefetch_admitted_total_val = int(prefetch_admitted_count.item())
    prefetch_admission_upper_recall = (
        prefetch_admitted_total_val / rs_prefetch_actual_total_val
        if rs_prefetch_actual_total_val > 0 else 0.0
    )
    prefetch_recall_gap_to_admission_upper = max(
        0.0,
        float(prefetch_admission_upper_recall) - float(rs_prefetch_recall),
    )

    prompt_tokens_global = int(prompt_tokens_t.item())
    gen_tokens_global = int(gen_tokens_t.item())
    token_count_contributor_rank_count = int(token_count_contributor_ranks_t.item())
    prompt_tokens_ep_rank0 = int(prompt_tokens_ep_rank0_t.item())
    gen_tokens_ep_rank0 = int(gen_tokens_ep_rank0_t.item())
    ep_rank0_token_count_contributor_rank_count = int(ep_rank0_token_count_contributor_ranks_t.item())
    prompt_tokens_all_ranks = int(prompt_tokens_all_ranks_t.item())
    gen_tokens_all_ranks = int(gen_tokens_all_ranks_t.item())
    all_ranks_token_count_contributor_rank_count = int(all_ranks_token_count_contributor_ranks_t.item())

    def _decode_tokens_for(gen_tokens: int, contributor_ranks: int) -> int:
        # Exclude the first generated token per active request stream from
        # decode-throughput accounting. For all-rank service throughput this is
        # batch * contributing_ranks; for EP rank0-only it is just batch.
        return max(int(gen_tokens) - int(args.batch) * max(int(contributor_ranks), 1), 0)

    def _tok_s(prompt_tokens: int, gen_tokens: int, wall_ms: float) -> float:
        return (int(prompt_tokens) + int(gen_tokens)) / max(float(wall_ms) / 1000.0, 1e-9)

    def _output_tok_s(gen_tokens: int, wall_ms: float) -> float:
        return int(gen_tokens) / max(float(wall_ms) / 1000.0, 1e-9)

    def _decode_tok_s(gen_tokens: int, contributor_ranks: int, wall_ms: float) -> float:
        return _decode_tokens_for(int(gen_tokens), int(contributor_ranks)) / max(float(wall_ms) / 1000.0, 1e-9)

    placement_rank_histograms: Dict[str, List[int]] = {}
    placement_empty_rank_layers: List[int] = []
    for l, blk in SparseContext.layer_blocks.items():
        try:
            hist = torch.bincount(
                blk.expert_to_rank.detach().to("cpu", dtype=torch.long),
                minlength=int(ep_size),
            ).tolist()
            placement_rank_histograms[str(int(l))] = [int(x) for x in hist]
            if any(int(x) == 0 for x in hist):
                placement_empty_rank_layers.append(int(l))
        except Exception:
            continue

    sim_config = _sim_config_summary(env.world)
    per_node_breakdown = _build_per_node_breakdown(gathered_rank_breakdown or [], sim_config["sim_node_map"])
    weaknet_pair_event_counts = _merge_counter_dicts(gathered_rank_breakdown or [], "weaknet_inter_node_pair_event_counts")
    weaknet_pair_item_counts = _merge_counter_dicts(gathered_rank_breakdown or [], "weaknet_inter_node_pair_item_counts")
    weaknet_packet_trace = _collect_packet_trace(gathered_rank_breakdown or [])
    physical_moe_comm_trace = _collect_physical_moe_comm_trace(gathered_rank_breakdown or [])
    physical_moe_comm_summary = _physical_moe_comm_summary(physical_moe_comm_trace)
    task_local_summary = _counts_task_local_by_rank_node(gathered_task_local_meta or [])

    node_grouped_mode = str(os.environ.get("SPARSE_NODE_GROUPED_FASTPATH_MODE", ""))
    node_grouped_prefill_mode = str(os.environ.get("SPARSE_NODE_GROUPED_FASTPATH_PREFILL_MODE", ""))
    node_grouped_decode_mode = str(os.environ.get("SPARSE_NODE_GROUPED_FASTPATH_DECODE_MODE", ""))
    mode_global_flags = _node_grouped_mode_flags(node_grouped_mode)
    mode_prefill_flags = _node_grouped_mode_flags(node_grouped_prefill_mode or node_grouped_mode)
    mode_decode_flags = _node_grouped_mode_flags(node_grouped_decode_mode or node_grouped_mode)

    raw_tensor = int(_env_flag("SPARSE_NODE_GROUPED_TENSOR_FASTPATH", "0"))
    raw_tensor_prefill = int(_env_flag("SPARSE_NODE_GROUPED_TENSOR_FASTPATH_PREFILL", "0"))
    raw_tensor_decode = int(_env_flag("SPARSE_NODE_GROUPED_TENSOR_FASTPATH_DECODE", "0"))
    raw_triton = int(_env_flag("SPARSE_NODE_GROUPED_TRITON_PACK", "0"))
    raw_triton_prefill = int(_env_flag("SPARSE_NODE_GROUPED_TRITON_PACK_PREFILL", "0"))
    raw_triton_decode = int(_env_flag("SPARSE_NODE_GROUPED_TRITON_PACK_DECODE", "0"))
    raw_same_node = int(_env_flag("SPARSE_NODE_GROUPED_SAME_NODE_FASTPATH", "0"))
    raw_same_node_prefill = int(_env_flag("SPARSE_NODE_GROUPED_SAME_NODE_FASTPATH_PREFILL", "0"))
    raw_same_node_decode = int(_env_flag("SPARSE_NODE_GROUPED_SAME_NODE_FASTPATH_DECODE", "0"))
    raw_stage2_local = int(_env_flag("SPARSE_NODE_GROUPED_STAGE2_LOCAL_RANK_FASTPATH", "0"))
    raw_stage2_local_prefill = int(_env_flag("SPARSE_NODE_GROUPED_STAGE2_LOCAL_RANK_FASTPATH_PREFILL", "0"))
    raw_stage2_local_decode = int(_env_flag("SPARSE_NODE_GROUPED_STAGE2_LOCAL_RANK_FASTPATH_DECODE", "0"))
    raw_pack_meta = int(_env_flag("SPARSE_NODE_GROUPED_PACK_META", "0"))
    raw_pack_meta_prefill = int(_env_flag("SPARSE_NODE_GROUPED_PACK_META_PREFILL", "0"))
    raw_pack_meta_decode = int(_env_flag("SPARSE_NODE_GROUPED_PACK_META_DECODE", "0"))
    raw_identity_combine = int(_env_flag("SPARSE_NODE_GROUPED_IDENTITY_COMBINE_FASTPATH", "0"))
    raw_identity_combine_prefill = int(_env_flag("SPARSE_NODE_GROUPED_IDENTITY_COMBINE_FASTPATH_PREFILL", "0"))
    raw_identity_combine_decode = int(_env_flag("SPARSE_NODE_GROUPED_IDENTITY_COMBINE_FASTPATH_DECODE", "0"))

    plain_ep_exec_mode = str(os.environ.get("SPARSE_PLAIN_EP_EXEC_MODE", "")).strip().lower().replace("-", "_")
    if plain_ep_exec_mode in ("sorted", "segment", "current", "optimized"):
        plain_ep_exec_mode = "sorted_segment"
    elif plain_ep_exec_mode in ("unique", "historical", "history", "original", "legacy_unique"):
        plain_ep_exec_mode = "unique_mask"
    elif plain_ep_exec_mode in ("full", "legacy", "legacy_mask", "legacy_full"):
        plain_ep_exec_mode = "full_mask"
    elif not plain_ep_exec_mode:
        plain_ep_exec_mode = "full_mask" if _env_flag("SPARSE_PLAIN_EP_LEGACY_EXEC", "0") else "unique_mask"
    required_plain_ep_exec_mode = str(os.environ.get("SPARSE_REQUIRE_PLAIN_EP_EXEC_MODE", "")).strip().lower().replace("-", "_")
    if required_plain_ep_exec_mode:
        if required_plain_ep_exec_mode in ("sorted", "segment", "current", "optimized"):
            required_plain_ep_exec_mode = "sorted_segment"
        elif required_plain_ep_exec_mode in ("unique", "historical", "history", "original", "legacy_unique"):
            required_plain_ep_exec_mode = "unique_mask"
        elif required_plain_ep_exec_mode in ("full", "legacy", "legacy_mask", "legacy_full"):
            required_plain_ep_exec_mode = "full_mask"
        if plain_ep_exec_mode != required_plain_ep_exec_mode:
            raise RuntimeError(
                "Plain EP executor mismatch: "
                f"required SPARSE_REQUIRE_PLAIN_EP_EXEC_MODE={required_plain_ep_exec_mode}, "
                f"but effective SPARSE_PLAIN_EP_EXEC_MODE={plain_ep_exec_mode}. "
                "This run is not comparable to the plain-executor baseline."
            )

    eff_tensor = int(bool(raw_tensor or mode_global_flags["tensor"]))
    eff_tensor_prefill = int(bool(raw_tensor or raw_tensor_prefill or mode_prefill_flags["tensor"]))
    eff_tensor_decode = int(bool(raw_tensor or raw_tensor_decode or mode_decode_flags["tensor"]))
    eff_triton = int(bool(raw_triton or mode_global_flags["triton"]))
    eff_triton_prefill = int(bool(raw_triton or raw_triton_prefill or mode_prefill_flags["triton"]))
    eff_triton_decode = int(bool(raw_triton or raw_triton_decode or mode_decode_flags["triton"]))
    eff_same_node = int(bool(raw_same_node or mode_global_flags["same_node"]))
    eff_same_node_prefill = int(bool(raw_same_node or raw_same_node_prefill or mode_prefill_flags["same_node"]))
    eff_same_node_decode = int(bool(raw_same_node or raw_same_node_decode or mode_decode_flags["same_node"]))
    eff_stage2_local = int(bool(raw_stage2_local or mode_global_flags["stage2_local"]))
    eff_stage2_local_prefill = int(bool(raw_stage2_local or raw_stage2_local_prefill or mode_prefill_flags["stage2_local"]))
    eff_stage2_local_decode = int(bool(raw_stage2_local or raw_stage2_local_decode or mode_decode_flags["stage2_local"]))
    eff_pack_meta = int(bool(raw_pack_meta or mode_global_flags["pack_meta"]))
    eff_pack_meta_prefill = int(bool(raw_pack_meta or raw_pack_meta_prefill or mode_prefill_flags["pack_meta"]))
    eff_pack_meta_decode = int(bool(raw_pack_meta or raw_pack_meta_decode or mode_decode_flags["pack_meta"]))
    eff_identity_combine = int(bool(raw_identity_combine or mode_global_flags["identity_combine"]))
    eff_identity_combine_prefill = int(bool(raw_identity_combine or raw_identity_combine_prefill or mode_prefill_flags["identity_combine"]))
    eff_identity_combine_decode = int(bool(raw_identity_combine or raw_identity_combine_decode or mode_decode_flags["identity_combine"]))

    ers_cfg = ers_config_from_env()

    record = {
        "run_tag": args.run_tag,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "world_size": env.world,
        **sim_config,

        "model_path": args.model_path,
        "batch": args.batch,
        "dataset_path": str(args.dataset),
        "dataset_skip_valid": int(getattr(args, "dataset_skip_valid", 0)),
        "dataset_skip_valid_per_rank": str(os.environ.get("DATASET_SKIP_VALID_PER_RANK", os.environ.get("SPARSE_DATASET_SKIP_VALID_PER_RANK", ""))),
        "dataset_skip_valid_rank_stride": str(os.environ.get("DATASET_SKIP_VALID_RANK_STRIDE", os.environ.get("SPARSE_DATASET_SKIP_VALID_RANK_STRIDE", ""))),
        "task_local_enabled": int(bool(_task_manifest_path_from_env())),
        "task_manifest_path": str(_task_manifest_path_from_env()),
        "token_count_mode_requested": str(token_count_mode_requested or "auto"),
        "token_count_mode": str(token_count_mode),
        "token_count_contributor_rank_count": int(token_count_contributor_rank_count),
        "token_count_ep_rank0_contributor_rank_count": int(ep_rank0_token_count_contributor_rank_count),
        "token_count_all_ranks_contributor_rank_count": int(all_ranks_token_count_contributor_rank_count),
        "task_split": str(_task_local_split_from_env("test")),
        "sampled_task_counts_by_rank": task_local_summary.get("sampled_task_counts_by_rank", {}),
        "sampled_task_counts_by_node": task_local_summary.get("sampled_task_counts_by_node", {}),
        "max_new": args.max_new,
        "dtype": args.dtype,
        "torch_version": str(torch.__version__),
        "python_executable": str(sys.executable),
        "torchdynamo_disable": int(_env_flag("TORCHDYNAMO_DISABLE", "0")),
        "plain_ep_legacy_exec": int(plain_ep_exec_mode in ("unique_mask", "full_mask")),
        "plain_ep_exec_mode": plain_ep_exec_mode,

        "placement_mode": args.placement_mode,
        "scp_ep_enabled": int(str(args.placement_mode) == "scp_ep"),
        "ep_fused_weight_dispatch": int(str(os.environ.get("SPARSE_EP_FUSED_WEIGHT_DISPATCH", "0")).strip().lower() not in ("", "0", "false", "no", "off")),
        "routing_mode": args.routing_mode,
        "expert_source": args.expert_source,
        "scp_apply_prefill": int(str(os.environ.get("SPARSE_SCP_APPLY_PREFILL", "1")).strip().lower() not in ("", "0", "false", "no", "off")),
        "scp_apply_decode": int(str(os.environ.get("SPARSE_SCP_APPLY_DECODE", "0")).strip().lower() not in ("", "0", "false", "no", "off")),
        "decode_cache_aware_scp_enabled": int(str(os.environ.get("SPARSE_DECODE_CACHE_AWARE_SCP", "0")).strip().lower() not in ("", "0", "false", "no", "off")),
        "scp_clear_cache_before_decode": int(str(os.environ.get("SPARSE_SCP_CLEAR_CACHE_BEFORE_DECODE", "0")).strip().lower() in ("1", "true", "yes", "on")),
        "reset_runtime_state_each_repeat": int(str(os.environ.get("SPARSE_RESET_RUNTIME_STATE_EACH_REPEAT", "1")).strip().lower() not in ("", "0", "false", "no", "off")),
        "clear_expert_cache_each_repeat": int(str(os.environ.get("SPARSE_CLEAR_EXPERT_CACHE_EACH_REPEAT", "0")).strip().lower() in ("1", "true", "yes", "on")),
        "demote_hot_cache_each_repeat": int(str(os.environ.get("SPARSE_DEMOTE_HOT_CACHE_EACH_REPEAT", "0")).strip().lower() in ("1", "true", "yes", "on")),
        "planner_mode": args.planner_mode,
        "lightweight_stats_by_rank": lightweight_stats,
        "runtime_mode": runtime_mode,
        "ep_size": int(ep_size),
        "dp_size": int(dp_size),
        "prefetch_top_m": args.prefetch_top_m,
        "rs_prefetch_pred_topk": int(os.environ.get("SPARSE_RS_PREFETCH_PRED_TOPK", "0") or 0),
        "rs_prefetch_top_m_by_rank": str(os.environ.get("SPARSE_RS_PREFETCH_TOP_M_BY_RANK", "")),
        "rs_prefetch_pred_topk_by_rank": str(os.environ.get("SPARSE_RS_PREFETCH_PRED_TOPK_BY_RANK", "")),
        "rs_cache_safe_min_score_by_rank": str(os.environ.get("SPARSE_RS_CACHE_SAFE_MIN_SCORE_BY_RANK", "")),
        "rs_cache_safe_replace_advantage_by_rank": str(os.environ.get("SPARSE_RS_CACHE_SAFE_REPLACE_ADVANTAGE_BY_RANK", "")),
        "rs_cache_safe_replace_margin_by_rank": str(os.environ.get("SPARSE_RS_CACHE_SAFE_REPLACE_MARGIN_BY_RANK", "")),
        "rs_prefetch_recent_protect_steps_by_rank": str(os.environ.get("SPARSE_RS_PREFETCH_RECENT_PROTECT_STEPS_BY_RANK", "")),
        "rs_prefetch_prefill": int(str(os.environ.get("SPARSE_RS_PREFETCH_PREFILL", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "rank_grouped_dispatch": int(str(os.environ.get("SPARSE_RANK_GROUPED_DISPATCH", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "rank_grouped_prefill": int(str(os.environ.get("SPARSE_RANK_GROUPED_PREFILL", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "node_grouped_dispatch": int(str(os.environ.get("SPARSE_NODE_GROUPED_DISPATCH", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "node_grouped_prefill": int(str(os.environ.get("SPARSE_NODE_GROUPED_PREFILL", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "node_grouped_adaptive": int(str(os.environ.get("SPARSE_NODE_GROUPED_LAYER_ADAPTIVE", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "node_grouped_min_saving_ms": float(os.environ.get("SPARSE_NODE_GROUPED_MIN_SAVING_MS", "0") or 0),
        "node_grouped_overhead_margin": float(os.environ.get("SPARSE_NODE_GROUPED_OVERHEAD_MARGIN", "1.0") or 1.0),
        "node_grouped_est_overhead_ms": float(os.environ.get("SPARSE_NODE_GROUPED_EST_OVERHEAD_MS", "0") or 0),
        "node_grouped_layer_policy_path": str(os.environ.get("SPARSE_NODE_GROUPED_LAYER_POLICY_PATH", "")),
        "node_grouped_layer_policy_default": str(os.environ.get("SPARSE_NODE_GROUPED_LAYER_POLICY_DEFAULT", "adaptive")),
        "node_grouped_fastpath_mode": node_grouped_mode,
        "node_grouped_fastpath_prefill_mode": node_grouped_prefill_mode,
        "node_grouped_fastpath_decode_mode": node_grouped_decode_mode,
        "node_grouped_force_layers": str(os.environ.get("SPARSE_NODE_GROUPED_FORCE_LAYERS", "")),
        "node_grouped_disable_layers": str(os.environ.get("SPARSE_NODE_GROUPED_DISABLE_LAYERS", "")),
        "node_grouped_tensor_fastpath": eff_tensor,
        "node_grouped_tensor_fastpath_prefill": eff_tensor_prefill,
        "node_grouped_tensor_fastpath_decode": eff_tensor_decode,
        "node_grouped_triton_pack": eff_triton,
        "node_grouped_triton_pack_prefill": eff_triton_prefill,
        "node_grouped_triton_pack_decode": eff_triton_decode,
        "node_grouped_triton_strict": int(_env_flag("SPARSE_NODE_GROUPED_TRITON_STRICT", "0")),
        "node_grouped_same_node_fastpath": eff_same_node,
        "node_grouped_same_node_fastpath_prefill": eff_same_node_prefill,
        "node_grouped_same_node_fastpath_decode": eff_same_node_decode,
        "node_grouped_stage2_local_rank_fastpath": eff_stage2_local,
        "node_grouped_stage2_local_rank_fastpath_prefill": eff_stage2_local_prefill,
        "node_grouped_stage2_local_rank_fastpath_decode": eff_stage2_local_decode,
        "node_grouped_pack_meta": eff_pack_meta,
        "node_grouped_pack_meta_prefill": eff_pack_meta_prefill,
        "node_grouped_pack_meta_decode": eff_pack_meta_decode,
        "node_grouped_pack_meta_float32": int(_env_flag("SPARSE_NODE_GROUPED_PACK_META_FLOAT32", "1")),
        "node_grouped_pack_meta_float32_prefill": int(_env_flag("SPARSE_NODE_GROUPED_PACK_META_FLOAT32_PREFILL", "0")),
        "node_grouped_pack_meta_float32_decode": int(_env_flag("SPARSE_NODE_GROUPED_PACK_META_FLOAT32_DECODE", "0")),
        "node_grouped_identity_combine_fastpath": eff_identity_combine,
        "node_grouped_identity_combine_fastpath_prefill": eff_identity_combine_prefill,
        "node_grouped_identity_combine_fastpath_decode": eff_identity_combine_decode,
        "node_grouped_skip_dummy_a2a": int(_env_flag("SPARSE_NODE_GROUPED_SKIP_DUMMY_A2A", "0")),
        "node_grouped_tensor_fastpath_raw_env": raw_tensor,
        "node_grouped_tensor_fastpath_prefill_raw_env": raw_tensor_prefill,
        "node_grouped_tensor_fastpath_decode_raw_env": raw_tensor_decode,
        "node_grouped_triton_pack_raw_env": raw_triton,
        "node_grouped_triton_pack_prefill_raw_env": raw_triton_prefill,
        "node_grouped_triton_pack_decode_raw_env": raw_triton_decode,
        "node_grouped_same_node_fastpath_raw_env": raw_same_node,
        "node_grouped_same_node_fastpath_prefill_raw_env": raw_same_node_prefill,
        "node_grouped_same_node_fastpath_decode_raw_env": raw_same_node_decode,
        "node_grouped_stage2_local_rank_fastpath_raw_env": raw_stage2_local,
        "node_grouped_stage2_local_rank_fastpath_prefill_raw_env": raw_stage2_local_prefill,
        "node_grouped_stage2_local_rank_fastpath_decode_raw_env": raw_stage2_local_decode,
        "node_grouped_pack_meta_raw_env": raw_pack_meta,
        "node_grouped_pack_meta_prefill_raw_env": raw_pack_meta_prefill,
        "node_grouped_pack_meta_decode_raw_env": raw_pack_meta_decode,
        "node_grouped_identity_combine_fastpath_raw_env": raw_identity_combine,
        "node_grouped_identity_combine_fastpath_prefill_raw_env": raw_identity_combine_prefill,
        "node_grouped_identity_combine_fastpath_decode_raw_env": raw_identity_combine_decode,
        "node_grouped_tensor_fastpath_mode": str(os.environ.get("SPARSE_NODE_GROUPED_TENSOR_FASTPATH_MODE", "")),
        **ers_cfg.to_summary(),
        # Backward-compatible summary fields for existing analysis scripts.
        # These now mirror the canonical ERS config, while SPARSE_NODEPATCH_* is
        # treated as a deprecated alias by utils.runtime.config.
        "nodepatch_enabled": int(ers_cfg.enabled),
        "nodepatch_phase": str(ers_cfg.phase),
        "nodepatch_topm": int(ers_cfg.top_m),
        "nodepatch_margin": float(ers_cfg.margin),
        "nodepatch_margin_kind": str(ers_cfg.margin_kind),
        "nodepatch_max_replace": int(ers_cfg.max_replace),
        "nodepatch_keep_top1": int(ers_cfg.keep_top1),
        "nodepatch_only_cross_node": int(ers_cfg.only_cross_node),
        "nodepatch_fastpath": int(ers_cfg.fastpath),
        "nodepatch_lean_repair": int(_env_flag("SPARSE_ERS_LEAN_REPAIR", os.environ.get("SPARSE_NODEPATCH_LEAN_REPAIR", "0"))),
        "nodepatch_lean_keep_topk": str(os.environ.get("SPARSE_ERS_LEAN_KEEP_TOPK", os.environ.get("SPARSE_NODEPATCH_LEAN_KEEP_TOPK", ""))),
        "nodepatch_lean_candidate_scope": str(os.environ.get("SPARSE_ERS_LEAN_CANDIDATE_SCOPE", os.environ.get("SPARSE_NODEPATCH_LEAN_CANDIDATE_SCOPE", "owner_local"))),
        "nodepatch_lean_resident_nearest": int(_env_flag("SPARSE_ERS_LEAN_RESIDENT_NEAREST", os.environ.get("SPARSE_NODEPATCH_LEAN_RESIDENT_NEAREST", "0"))),
        "nodepatch_lean_require_improvement": int(_env_flag("SPARSE_ERS_LEAN_REQUIRE_IMPROVEMENT", os.environ.get("SPARSE_NODEPATCH_LEAN_REQUIRE_IMPROVEMENT", "1"))),
        "nodepatch_lean_stats": int(_env_flag("SPARSE_ERS_LEAN_STATS", os.environ.get("SPARSE_NODEPATCH_LEAN_STATS", "1"))),
        "nodepatch_collect_hist": int(ers_cfg.collect_hist),
        "nodepatch_disable_margin": int(ers_cfg.disable_margin),
        "nodepatch_event_aware": int(ers_cfg.event_aware),
        "nodepatch_event_aggressive": int(ers_cfg.event_aggressive),
        "nodepatch_event_only_empty_packet": int(ers_cfg.event_only_empty_packet),
        "nodepatch_event_require_local_candidate": int(ers_cfg.event_require_local_candidate),
        "nodepatch_event_packet_max_items": int(ers_cfg.event_packet_max_items),
        "nodepatch_packet_emptying_first": int(ers_cfg.packet_emptying_first),
        "nodepatch_cache_aware": int(ers_cfg.cache_aware),
        "nodepatch_cache_require_owner_local": int(ers_cfg.cache_require_owner_local),
        "nodepatch_require_added_resident": int(ers_cfg.require_added_resident),
        "nodepatch_legacy_alias_deprecated": 1,
        "prefetch_extra_capacity": int(os.environ.get("SPARSE_PREFETCH_EXTRA_CAPACITY", "0") or 0),
        "prefetch_reserved_env": int(os.environ.get("SPARSE_PREFETCH_RESERVED", "0") or 0),
        "prefetch_extra_capacity_effective": (
            int(os.environ.get("SPARSE_PREFETCH_EXTRA_CAPACITY", "0") or 0)
            if args.planner_mode == "rs_prefetch" else 0
        ),
        "prefetch_reserved_effective": (
            int(os.environ.get("SPARSE_PREFETCH_RESERVED", "0") or 0)
            if args.planner_mode == "rs_prefetch" else 0
        ),
        "prefetch_allow_active_replace": int(os.environ.get("SPARSE_PREFETCH_ALLOW_ACTIVE_REPLACE", "1") or 1),
        "prefetch_min_free_mb": float(os.environ.get("SPARSE_PREFETCH_MIN_FREE_MB", "0") or 0),
        "prefetch_min_candidate_score": float(os.environ.get("SPARSE_PREFETCH_MIN_CAND_SCORE", "0.02") or 0.02),
        "prefetch_replace_margin": float(os.environ.get("SPARSE_PREFETCH_REPLACE_MARGIN", "1.05") or 1.05),
        "prefetch_replace_advantage": float(os.environ.get("SPARSE_PREFETCH_REPLACE_ADVANTAGE", "0.02") or 0.02),
        "global_expert_cache": int(str(os.environ.get("SPARSE_GLOBAL_EXPERT_CACHE", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "global_expert_cache_capacity": int(os.environ.get("SPARSE_GLOBAL_EXPERT_CACHE_CAPACITY", "0") or 0),
        "cache_value_score_scale": float(os.environ.get("SPARSE_CACHE_VALUE_SCORE_SCALE", "4096.0") or 4096.0),
        "cache_value_min_candidate_score": float(os.environ.get("SPARSE_CACHE_VALUE_MIN_CAND_SCORE", "0.0") or 0.0),
        "cache_value_replace_advantage": float(os.environ.get("SPARSE_CACHE_VALUE_REPLACE_ADVANTAGE", "0.02") or 0.02),
        "oracle_async_prefetch": int(str(os.environ.get("SPARSE_ORACLE_ASYNC_PREFETCH", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "oracle_stage_current": str(os.environ.get("SPARSE_ORACLE_STAGE_CURRENT", "")),
        "oracle_prefetch_lead_layers": int(os.environ.get("SPARSE_ORACLE_PREFETCH_LEAD_LAYERS", "1") or 1),
        "oracle_prefetch_window_layers": int(os.environ.get("SPARSE_ORACLE_PREFETCH_WINDOW_LAYERS", "1") or 1),
        "oracle_prefetch_top_m": int(os.environ.get("SPARSE_ORACLE_PREFETCH_TOP_M", str(args.prefetch_top_m)) or args.prefetch_top_m),
        "oracle_prefetch_recurring_only": int(str(os.environ.get("SPARSE_ORACLE_PREFETCH_RECURRING_ONLY", "1")).strip().lower() not in ("0", "false", "no", "off")),
        "rs_dir": args.rs_dir,
        "rs_require_complete": bool(args.rs_require_complete),

        "expert_offload": args.expert_offload,
        "expert_cache": args.expert_cache,
        "expert_cache_by_rank": str(os.environ.get("SPARSE_EXPERT_CACHE_BY_RANK", os.environ.get("EXPERT_CACHE_BY_RANK", ""))),
        "expert_cache_by_node": str(os.environ.get("SPARSE_EXPERT_CACHE_BY_NODE", "")),
        "cache_policy": args.cache_policy,
        "scp_path": args.scp_path,
        "scp_ep_placement": str(getattr(args, "scp_ep_placement", "") or os.environ.get("SPARSE_SCP_EP_PLACEMENT", "")),
        "scp_quota_repair": int(str(os.environ.get("SPARSE_SCP_QUOTA_REPAIR", "0")).strip().lower() in ("1", "true", "yes", "on")),
        "placement_rank_histograms": placement_rank_histograms,
        "placement_empty_rank_layers": placement_empty_rank_layers,
        "placement_empty_rank_layer_count": len(placement_empty_rank_layers),

        "skvs_enabled": bool(args.enable_skvs),
        "skvs_sink": args.skvs_sink,
        "skvs_window": args.skvs_window,
        "skvs_ratio": args.skvs_ratio,
        "skvs_min_kv_len": int(args.skvs_min_kv_len),
        "skvs_min_sparse_drop_ratio": float(args.skvs_min_sparse_drop_ratio),
        "skvs_guidance_every_k": int(args.skvs_guidance_every_k),
        "skvs_guidance_start_kv_len": int(args.skvs_guidance_start_kv_len),
        "owner_group_size": int(os.environ.get("SPARSE_OWNER_GROUP_SIZE", "0") or 0),
        "skvs_migration_impl": str(os.environ.get("SPARSE_SKVS_MIGRATION_IMPL", "tensor")),
        "skvs_owner_local": int(str(os.environ.get("SPARSE_SKVS_OWNER_LOCAL", "1")).strip().lower() not in ("0", "false", "no", "off")),
        "skvs_disable_migration": int(str(os.environ.get("SPARSE_SKVS_DISABLE_MIGRATION", "0")).strip().lower() in ("1", "true", "yes", "on")),
        "skvs_static_owner_no_migration": int(str(os.environ.get("SPARSE_SKVS_STATIC_OWNER_NO_MIGRATION", "1")).strip().lower() in ("1", "true", "yes", "on")),
        "skvs_clear_dense_cache": int(str(os.environ.get("SPARSE_SKVS_CLEAR_DENSE_CACHE", "1")).strip().lower() not in ("0", "false", "no", "off")),
        "skvs_compact_every_k": int(os.environ.get("SPARSE_SKVS_COMPACT_EVERY_K", "1") or 1),
        "skvs_start_kv_len": int(os.environ.get("SPARSE_SKVS_START_KV_LEN", "0") or 0),
        "skvs_block_size": int(os.environ.get("SPARSE_SKVS_BLOCK_SIZE", "1") or 1),
        "skvs_apply_layer_start": int(os.environ.get("SPARSE_SKVS_APPLY_LAYER_START", "1") or 1),
        "skvs_apply_layer_end": int(os.environ.get("SPARSE_SKVS_APPLY_LAYER_END", "999999") or 999999),

        "prompt_len": args.prompt_len,
        "prompt_tokens": prompt_tokens_global,
        "gen_tokens": gen_tokens_global,
        "prompt_tokens_ep_rank0": prompt_tokens_ep_rank0,
        "gen_tokens_ep_rank0": gen_tokens_ep_rank0,
        "prompt_tokens_all_ranks": prompt_tokens_all_ranks,
        "gen_tokens_all_ranks": gen_tokens_all_ranks,

        "wall_total_ms": float(total_wall.item()),
        "wall_prefill_ms": float(prefill_wall.item()),
        "wall_decode_ms": float(decode_wall.item()),
        "wall_ttft_ms": float(ttft_wall.item()),

        "decode_initial_tokens_excluded": int(args.batch) * max(int(token_count_contributor_rank_count), 1),
        "decode_initial_tokens_excluded_ep_rank0": int(args.batch) * max(int(ep_rank0_token_count_contributor_rank_count), 1),
        "decode_initial_tokens_excluded_all_ranks": int(args.batch) * max(int(all_ranks_token_count_contributor_rank_count), 1),
        "decode_tokens_counted": _decode_tokens_for(gen_tokens_global, token_count_contributor_rank_count),
        "decode_tokens_counted_ep_rank0": _decode_tokens_for(gen_tokens_ep_rank0, ep_rank0_token_count_contributor_rank_count),
        "decode_tokens_counted_all_ranks": _decode_tokens_for(gen_tokens_all_ranks, all_ranks_token_count_contributor_rank_count),
        "e2e_total_tok_s": _tok_s(prompt_tokens_global, gen_tokens_global, float(total_wall.item())),
        "e2e_output_tok_s": _output_tok_s(gen_tokens_global, float(total_wall.item())),
        "throughput_tok_s": _tok_s(prompt_tokens_global, gen_tokens_global, float(total_wall.item())),
        "decode_throughput_tok_s": _decode_tok_s(gen_tokens_global, token_count_contributor_rank_count, float(decode_wall.item())),
        "e2e_total_tok_s_ep_rank0": _tok_s(prompt_tokens_ep_rank0, gen_tokens_ep_rank0, float(total_wall.item())),
        "e2e_output_tok_s_ep_rank0": _output_tok_s(gen_tokens_ep_rank0, float(total_wall.item())),
        "throughput_tok_s_ep_rank0": _tok_s(prompt_tokens_ep_rank0, gen_tokens_ep_rank0, float(total_wall.item())),
        "decode_throughput_tok_s_ep_rank0": _decode_tok_s(gen_tokens_ep_rank0, ep_rank0_token_count_contributor_rank_count, float(decode_wall.item())),
        "e2e_total_tok_s_all_ranks": _tok_s(prompt_tokens_all_ranks, gen_tokens_all_ranks, float(total_wall.item())),
        "e2e_output_tok_s_all_ranks": _output_tok_s(gen_tokens_all_ranks, float(total_wall.item())),
        "throughput_tok_s_all_ranks": _tok_s(prompt_tokens_all_ranks, gen_tokens_all_ranks, float(total_wall.item())),
        "decode_throughput_tok_s_all_ranks": _decode_tok_s(gen_tokens_all_ranks, all_ranks_token_count_contributor_rank_count, float(decode_wall.item())),
        "throughput_tok_s_semantics": "Deprecated alias for e2e_total_tok_s: (prompt_tokens + gen_tokens) / wall_total_s. Use e2e_output_tok_s for serving-style output-token throughput.",
        "decode_throughput_tok_s_semantics": "Decode throughput excludes the first generated token per active request stream and divides by wall_decode_s.",

        # eval_core metrics 口径
        "comm_mb_activation": float(comm.item()) / 1024**2,
        "comm_mb_activation_remote": float(comm_remote.item()) / 1024**2,
        "comm_mb_activation_intra_node": float(comm_intra_node.item()) / 1024**2,
        "comm_mb_activation_inter_node": float(comm_inter_node.item()) / 1024**2,
        "weaknet_intra_node_item_count": int(weaknet_intra_items.item()),
        "weaknet_inter_node_item_count": int(weaknet_inter_items.item()),
        "weaknet_intra_node_transfer_count": int(weaknet_intra_transfers.item()),
        "weaknet_inter_node_transfer_count": int(weaknet_inter_transfers.item()),
        "weaknet_inter_node_packet_item_sumsq": int(weaknet_packet_sumsq.item()),
        "weaknet_inter_node_packet_max_items": int(weaknet_packet_max_items.item()),
        "weaknet_inter_node_packet_avg_items": (
            float(weaknet_inter_items.item()) / max(float(weaknet_inter_transfers.item()), 1.0)
        ),
        "weaknet_inter_node_packet_le1_count": int(weaknet_packet_le1.item()),
        "weaknet_inter_node_packet_le2_count": int(weaknet_packet_le2.item()),
        "weaknet_inter_node_packet_le4_count": int(weaknet_packet_le4.item()),
        "weaknet_inter_node_packet_le8_count": int(weaknet_packet_le8.item()),
        "weaknet_inter_node_packet_gt8_count": int(weaknet_packet_gt8.item()),
        "weaknet_inter_node_small_packet_threshold": int(os.environ.get("SPARSE_WEAKNET_SMALL_PACKET_THRESHOLD", "4") or 4),
        "weaknet_inter_node_small_packet_count": int(weaknet_small_packets.item()),
        "weaknet_inter_node_small_packet_item_count": int(weaknet_small_packet_items.item()),
        "weaknet_inter_node_small_packet_ratio": (
            float(weaknet_small_packets.item()) / max(float(weaknet_inter_transfers.item()), 1.0)
        ),
        "weaknet_inter_node_pair_event_counts": weaknet_pair_event_counts,
        "weaknet_inter_node_pair_item_counts": weaknet_pair_item_counts,
        "weaknet_packet_trace_enabled": int(_env_flag("SPARSE_WEAKNET_PACKET_TRACE", "0")),
        "weaknet_packet_trace": weaknet_packet_trace,
        "physical_moe_comm_trace_enabled": int(_env_flag("SPARSE_PHYSICAL_MOE_COMM_TRACE", "0")),
        "physical_moe_comm_trace": physical_moe_comm_trace,
        **physical_moe_comm_summary,
        "weaknet_inter_node_sim_ms": float(weaknet_inter_sim_ms.item()),
        "weaknet_inter_node_sim_ms_rankmax": float(weaknet_inter_sim_ms_rankmax.item()),
        "comm_mb_kv": float(kvb.item()) / 1024**2,
        "rank_grouped_pair_entries": int(rank_grouped_pair_entries.item()),
        "rank_grouped_packet_entries": int(rank_grouped_packet_entries.item()),
        "rank_grouped_dispatch_calls": int(rank_grouped_dispatch_calls.item()),
        "rank_grouped_packet_ratio": (
            float(rank_grouped_packet_entries.item()) / max(float(rank_grouped_pair_entries.item()), 1.0)
        ),
        "rank_grouped_theoretical_hidden_saving_ratio": (
            1.0 - float(rank_grouped_packet_entries.item()) / max(float(rank_grouped_pair_entries.item()), 1.0)
            if int(rank_grouped_pair_entries.item()) > 0 else 0.0
        ),
        "node_grouped_pair_entries": int(node_grouped_pair_entries.item()),
        "node_grouped_node_packet_entries": int(node_grouped_node_packet_entries.item()),
        "node_grouped_rank_packet_entries": int(node_grouped_rank_packet_entries.item()),
        "node_grouped_dispatch_calls": int(node_grouped_dispatch_calls.item()),
        "node_grouped_pack_ms": float(node_grouped_pack_ms.item()),
        "node_grouped_metadata_ms": float(node_grouped_metadata_ms.item()),
        "node_grouped_intra_fanout_ms": float(node_grouped_intra_fanout_ms.item()),
        "node_grouped_inter_send_ms": float(node_grouped_inter_send_ms.item()),
        "node_grouped_inter_send_weaknet_sleep_ms": float(node_grouped_inter_send_weaknet_sleep_ms.item()),
        "node_grouped_inter_send_stage1_weaknet_sleep_ms": float(node_grouped_inter_send_stage1_weaknet_sleep_ms.item()),
        "node_grouped_inter_send_stage2_weaknet_sleep_ms": float(node_grouped_inter_send_stage2_weaknet_sleep_ms.item()),
        "node_grouped_inter_send_stage1_non_sleep_ms": max(0.0, float(node_grouped_inter_send_ms.item()) - float(node_grouped_inter_send_stage1_weaknet_sleep_ms.item())),
        "node_grouped_intra_fanout_stage2_non_sleep_ms": max(0.0, float(node_grouped_intra_fanout_ms.item()) - float(node_grouped_inter_send_stage2_weaknet_sleep_ms.item())),
        "node_grouped_total_overhead_non_sleep_ms": max(0.0, float(node_grouped_total_overhead_ms.item()) - float(node_grouped_inter_send_weaknet_sleep_ms.item())),
        "node_grouped_inter_send_stage1_collective_ms": float(node_grouped_inter_send_stage1_collective_ms.item()),
        "node_grouped_inter_send_stage2_collective_ms": float(node_grouped_inter_send_stage2_collective_ms.item()),
        "node_grouped_metadata_exchange_ms": float(node_grouped_metadata_exchange_ms.item()),
        "node_grouped_inter_send_non_sleep_ms": max(0.0, float(node_grouped_inter_send_ms.item()) - float(node_grouped_inter_send_stage1_weaknet_sleep_ms.item())),
        "node_grouped_gather_ms": float(node_grouped_gather_ms.item()),
        "node_grouped_total_overhead_ms": float(node_grouped_total_overhead_ms.item()),
        "node_grouped_same_node_packet_entries": int(node_grouped_same_node_packet_entries.item()),
        "node_grouped_cross_node_packet_entries": int(node_grouped_cross_node_packet_entries.item()),
        "node_grouped_same_node_packet_fraction": (
            float(node_grouped_same_node_packet_entries.item())
            / max(float(node_grouped_same_node_packet_entries.item() + node_grouped_cross_node_packet_entries.item()), 1.0)
        ),
        "node_grouped_same_node_bypass_mb": float(node_grouped_same_node_bypass_bytes.item()) / 1024**2,
        "node_grouped_stage2_local_rank_packet_entries": int(node_grouped_stage2_local_rank_packet_entries.item()),
        "node_grouped_stage2_remote_rank_packet_entries": int(node_grouped_stage2_remote_rank_packet_entries.item()),
        "node_grouped_stage2_local_rank_packet_fraction": (
            float(node_grouped_stage2_local_rank_packet_entries.item())
            / max(float(node_grouped_stage2_local_rank_packet_entries.item() + node_grouped_stage2_remote_rank_packet_entries.item()), 1.0)
        ),
        "node_grouped_stage2_local_bypass_mb": float(node_grouped_stage2_local_bypass_bytes.item()) / 1024**2,
        "node_grouped_metadata_collective_count": int(node_grouped_metadata_collective_count.item()),
        "node_grouped_metadata_mb": float(node_grouped_metadata_bytes.item()) / 1024**2,
        "node_grouped_counts_build_ms": float(node_grouped_counts_build_ms.item()),
        "node_grouped_splits_build_ms": float(node_grouped_splits_build_ms.item()),
        "node_grouped_reverse_map_build_ms": float(node_grouped_reverse_map_build_ms.item()),
        "node_grouped_rank_to_node_gather_ms": float(node_grouped_rank_to_node_gather_ms.item()),
        "node_grouped_node_to_token_restore_ms": float(node_grouped_node_to_token_restore_ms.item()),
        "node_grouped_final_combine_ms": float(node_grouped_final_combine_ms.item()),
        "node_grouped_combine_intermediate_buffer_mb": float(node_grouped_combine_intermediate_buffer_bytes.item()) / 1024**2,
        "node_grouped_allocation_ms": float(node_grouped_allocation_ms.item()),
        "node_grouped_allocation_count": int(node_grouped_allocation_count.item()),
        "node_grouped_allocation_mb": float(node_grouped_allocation_bytes.item()) / 1024**2,
        "node_grouped_workspace_reuse_hit": int(node_grouped_workspace_reuse_hit.item()),
        "node_grouped_enabled_layers": int(node_grouped_enabled_layers.item()),
        "node_grouped_fallback_layers": int(node_grouped_fallback_layers.item()),
        "node_grouped_estimated_saving_ms": float(node_grouped_estimated_saving_ms.item()),
        "node_grouped_node_packet_ratio": (
            float(node_grouped_node_packet_entries.item()) / max(float(node_grouped_pair_entries.item()), 1.0)
        ),
        "node_grouped_rank_packet_ratio": (
            float(node_grouped_rank_packet_entries.item()) / max(float(node_grouped_pair_entries.item()), 1.0)
        ),
        "node_grouped_theoretical_node_hidden_saving_ratio": (
            1.0 - float(node_grouped_node_packet_entries.item()) / max(float(node_grouped_pair_entries.item()), 1.0)
            if int(node_grouped_pair_entries.item()) > 0 else 0.0
        ),
        "nodepatch_debug": nodepatch_dbg,
        "nodepatch_enabled_calls": float(nodepatch_dbg.get("enabled_calls", 0.0)),
        "nodepatch_tokens_seen": float(nodepatch_dbg.get("tokens_seen", 0.0)),
        "nodepatch_cross_node_tokens": float(nodepatch_dbg.get("cross_node_tokens", 0.0)),
        "nodepatch_repaired_tokens": float(nodepatch_dbg.get("repaired_tokens", 0.0)),
        "nodepatch_replacement_count": float(nodepatch_dbg.get("replacement_count", 0.0)),
        "nodepatch_repairable_token_fraction": float(nodepatch_dbg.get("repairable_token_fraction", 0.0)),
        "nodepatch_repairable_cross_node_fraction": float(nodepatch_dbg.get("repairable_cross_node_fraction", 0.0)),
        "nodepatch_transfer_event_reduction_pct": float(nodepatch_dbg.get("transfer_event_reduction_pct", 0.0)),
        "nodepatch_cross_node_item_reduction_pct": float(nodepatch_dbg.get("cross_node_item_reduction_pct", 0.0)),
        "nodepatch_avg_unique_nodes_before": float(nodepatch_dbg.get("avg_unique_nodes_before", 0.0)),
        "nodepatch_avg_unique_nodes_after": float(nodepatch_dbg.get("avg_unique_nodes_after", 0.0)),
        "nodepatch_avg_utility_loss_logit": float(nodepatch_dbg.get("avg_utility_loss_logit", 0.0)),
        "nodepatch_avg_utility_loss_prob": float(nodepatch_dbg.get("avg_utility_loss_prob", 0.0)),
        "nodepatch_avg_max_margin_logit": float(nodepatch_dbg.get("avg_max_margin_logit", 0.0)),
        "nodepatch_avg_max_margin_prob": float(nodepatch_dbg.get("avg_max_margin_prob", 0.0)),
        "nodepatch_max_margin_logit": float(nodepatch_dbg.get("max_margin_logit_max", 0.0)),
        "nodepatch_max_margin_prob": float(nodepatch_dbg.get("max_margin_prob_max", 0.0)),
        "nodepatch_runtime_packet_events_before": float(nodepatch_dbg.get("runtime_packet_events_before", 0.0)),
        "nodepatch_runtime_packet_events_after": float(nodepatch_dbg.get("runtime_packet_events_after", 0.0)),
        "nodepatch_runtime_packet_events_eliminated": float(nodepatch_dbg.get("runtime_packet_events_eliminated", 0.0)),
        "nodepatch_event_aware_skipped_repairs": float(nodepatch_dbg.get("event_aware_skipped_repairs", 0.0)),
        "nodepatch_cache_aware_skipped_repairs": float(nodepatch_dbg.get("cache_aware_skipped_repairs", 0.0)),
        "nodepatch_cache_aware_filtered_candidates": float(nodepatch_dbg.get("cache_aware_filtered_candidates", 0.0)),
        "nodepatch_packet_emptying_candidate_packets": float(nodepatch_dbg.get("packet_emptying_candidate_packets", 0.0)),
        "nodepatch_packet_emptying_feasible_packets": float(nodepatch_dbg.get("packet_emptying_feasible_packets", 0.0)),
        "nodepatch_packet_emptying_rejected_packets": float(nodepatch_dbg.get("packet_emptying_rejected_packets", 0.0)),
        "nodepatch_route_cost_before": float(nodepatch_dbg.get("route_cost_before_sum", 0.0)),
        "nodepatch_route_cost_after": float(nodepatch_dbg.get("route_cost_after_sum", 0.0)),
        "nodepatch_route_cost_gain": float(nodepatch_dbg.get("route_cost_gain_sum", 0.0)),
        "nodepatch_route_cost_reduction_pct": float(nodepatch_dbg.get("route_cost_reduction_pct", 0.0)),
        "nodepatch_avg_route_cost_gain_per_replacement": float(nodepatch_dbg.get("avg_route_cost_gain_per_replacement", 0.0)),
        "nodepatch_avg_replacement_rank_gap": float(nodepatch_dbg.get("avg_replacement_rank_gap", 0.0)),
        "nodepatch_rank_gap_max": float(nodepatch_dbg.get("rank_gap_max", 0.0)),
        "nodepatch_avg_old_prob": float(nodepatch_dbg.get("avg_old_prob", 0.0)),
        "nodepatch_avg_new_prob": float(nodepatch_dbg.get("avg_new_prob", 0.0)),
        "nodepatch_avg_prob_delta": float(nodepatch_dbg.get("avg_prob_delta", 0.0)),
        "nodepatch_candidate_not_prepared_reject_count": float(nodepatch_dbg.get("candidate_not_prepared_reject_count", 0.0)),
        "nodepatch_candidate_no_comm_gain_reject_count": float(nodepatch_dbg.get("candidate_no_comm_gain_reject_count", 0.0)),
        "nodepatch_candidate_scope_reject_count": float(nodepatch_dbg.get("candidate_scope_reject_count", 0.0)),
        "nodepatch_candidate_duplicate_reject_count": float(nodepatch_dbg.get("candidate_duplicate_reject_count", 0.0)),
        "nodepatch_candidate_rank_gap_reject_count": float(nodepatch_dbg.get("candidate_rank_gap_reject_count", 0.0)),
        "nodepatch_candidate_cache_reject_count": float(nodepatch_dbg.get("candidate_cache_reject_count", 0.0)),
        "nodepatch_candidate_total_reject_count": float(nodepatch_dbg.get("candidate_total_reject_count", 0.0)),
        "nodepatch_no_replace_position_count": float(nodepatch_dbg.get("no_replace_position_count", 0.0)),
        "prefill_scp_temporary_load_count": int(prefill_scp_transient_load_count.item()),
        "prefill_scp_temporary_hit_count": int(prefill_scp_transient_hit_count.item()),
        "prefill_scp_temporary_promote_count": int(prefill_scp_transient_promote_count.item()),
        "prefill_scp_temporary_evict_count": int(prefill_scp_transient_evict_count.item()),
        "prefill_scp_temporary_resident_at_decode_start": int(prefill_scp_transient_resident_at_decode_start.item()),
        "prefill_scp_temporary_demote_count": int(prefill_scp_transient_demote_count.item()),
        "prefill_scp_temporary_resident_at_decode_end": int(prefill_scp_transient_resident_at_decode_end.item()),
        "prefill_scp_temporary_decode_reuse_rate": (
            float(prefill_scp_transient_hit_count.item()) / max(float(prefill_scp_transient_resident_at_decode_start.item()), 1.0)
        ),
        "resolver_decode_total_queries": int(resolver_decode_total_queries.item()),
        "resolver_decode_rr_owner_count": int(resolver_decode_rr_owner_count.item()),
        "resolver_decode_scp_replica_selected_count": int(resolver_decode_scp_replica_selected_count.item()),
        "resolver_decode_rr_fallback_count": int(resolver_decode_rr_fallback_count.item()),
        "resolver_decode_replica_reject_not_ready_count": int(resolver_decode_replica_reject_not_ready_count.item()),
        "resolver_decode_replica_reject_no_comm_gain_count": int(resolver_decode_replica_reject_no_comm_gain_count.item()),
        "resolver_decode_replica_cache_ready_count": int(resolver_decode_replica_cache_ready_count.item()),
        "resolver_decode_replica_prefetch_ready_count": int(resolver_decode_replica_prefetch_ready_count.item()),
        "resolver_decode_route_cost_before": float(resolver_decode_route_cost_before.item()),
        "resolver_decode_route_cost_after": float(resolver_decode_route_cost_after.item()),
        "resolver_decode_route_cost_gain": float(resolver_decode_route_cost_before.item() - resolver_decode_route_cost_after.item()),
        "scp_phase_route_diag_enabled": int(os.environ.get("SPARSE_SCP_PHASE_ROUTE_DIAG", "0").strip().lower() not in ("0", "false", "no", "off")),
        "scp_prefill_route_diag_items": int(scp_prefill_route_diag_items.item()),
        "scp_prefill_route_diag_replica_items": int(scp_prefill_route_diag_replica_items.item()),
        "scp_prefill_route_diag_replica_item_rate": (
            float(scp_prefill_route_diag_replica_items.item()) / max(float(scp_prefill_route_diag_items.item()), 1.0)
        ),
        "scp_prefill_route_diag_cross_items_before": int(scp_prefill_route_diag_cross_items_before.item()),
        "scp_prefill_route_diag_cross_items_after": int(scp_prefill_route_diag_cross_items_after.item()),
        "scp_prefill_route_diag_cross_item_reduction_pct": (
            100.0
            * float(scp_prefill_route_diag_cross_items_before.item() - scp_prefill_route_diag_cross_items_after.item())
            / max(float(scp_prefill_route_diag_cross_items_before.item()), 1.0)
        ),
        "scp_prefill_route_diag_cross_events_before": int(scp_prefill_route_diag_cross_events_before.item()),
        "scp_prefill_route_diag_cross_events_after": int(scp_prefill_route_diag_cross_events_after.item()),
        "scp_prefill_route_diag_cross_event_reduction_pct": (
            100.0
            * float(scp_prefill_route_diag_cross_events_before.item() - scp_prefill_route_diag_cross_events_after.item())
            / max(float(scp_prefill_route_diag_cross_events_before.item()), 1.0)
        ),
        "scp_prefill_route_diag_cost_before": float(scp_prefill_route_diag_cost_before.item()),
        "scp_prefill_route_diag_cost_after": float(scp_prefill_route_diag_cost_after.item()),
        "scp_prefill_route_diag_cost_reduction_pct": (
            100.0
            * float(scp_prefill_route_diag_cost_before.item() - scp_prefill_route_diag_cost_after.item())
            / max(float(scp_prefill_route_diag_cost_before.item()), 1.0)
        ),
        "scp_decode_route_diag_items": int(scp_decode_route_diag_items.item()),
        "scp_decode_route_diag_replica_items": int(scp_decode_route_diag_replica_items.item()),
        "scp_decode_route_diag_replica_item_rate": (
            float(scp_decode_route_diag_replica_items.item()) / max(float(scp_decode_route_diag_items.item()), 1.0)
        ),
        "scp_decode_route_diag_cross_items_before": int(scp_decode_route_diag_cross_items_before.item()),
        "scp_decode_route_diag_cross_items_after": int(scp_decode_route_diag_cross_items_after.item()),
        "scp_decode_route_diag_cross_item_reduction_pct": (
            100.0
            * float(scp_decode_route_diag_cross_items_before.item() - scp_decode_route_diag_cross_items_after.item())
            / max(float(scp_decode_route_diag_cross_items_before.item()), 1.0)
        ),
        "scp_decode_route_diag_cross_events_before": int(scp_decode_route_diag_cross_events_before.item()),
        "scp_decode_route_diag_cross_events_after": int(scp_decode_route_diag_cross_events_after.item()),
        "scp_decode_route_diag_cross_event_reduction_pct": (
            100.0
            * float(scp_decode_route_diag_cross_events_before.item() - scp_decode_route_diag_cross_events_after.item())
            / max(float(scp_decode_route_diag_cross_events_before.item()), 1.0)
        ),
        "scp_decode_route_diag_cost_before": float(scp_decode_route_diag_cost_before.item()),
        "scp_decode_route_diag_cost_after": float(scp_decode_route_diag_cost_after.item()),
        "scp_decode_route_diag_cost_reduction_pct": (
            100.0
            * float(scp_decode_route_diag_cost_before.item() - scp_decode_route_diag_cost_after.item())
            / max(float(scp_decode_route_diag_cost_before.item()), 1.0)
        ),
        "rs_prefetch_target_resolver_match_count": int(rs_prefetch_target_resolver_match_count.item()),
        "rs_prefetch_target_resolver_mismatch_count": int(rs_prefetch_target_resolver_mismatch_count.item()),
        "rs_prefetch_would_be_wrong_without_resolver_count": int(rs_prefetch_would_be_wrong_without_resolver_count.item()),
        "rs_scp_target_mismatch_count": int(rs_scp_target_mismatch_count.item()),
        "scp_table_local_replica_route_count": int(scp_table_local_replica_route_count.item()),
        "scp_table_rr_fallback_count": int(scp_table_rr_fallback_count.item()),
        "scp_table_cross_node_route_count": int(scp_table_cross_node_route_count.item()),
        "scp_table_same_node_route_count": int(scp_table_same_node_route_count.item()),
        "scp_route_cost_before_sum": float(scp_route_cost_before_sum.item()),
        "scp_route_cost_after_sum": float(scp_route_cost_after_sum.item()),
        "scp_lookup_calls": int(scp_lookup_calls.item()),
        "scp_lookup_overhead_ms": float(scp_lookup_overhead_ms.item()),
        "scp_route_cost_reduction_pct": (
            100.0
            * float(scp_route_cost_before_sum.item() - scp_route_cost_after_sum.item())
            / max(float(scp_route_cost_before_sum.item()), 1.0)
        ),
        "valid_comparison": bool(int(rs_scp_target_mismatch_count.item()) == 0),
        "expert_h2d_mb": float(h2d_total.item()) / 1024**2,
        "expert_h2d_prefetch_mb": float(h2d_prefetch.item()) / 1024**2,
        "expert_h2d_demand_mb": float(h2d_demand.item()) / 1024**2,
        "oracle_stage_h2d_mb": float(oracle_stage_h2d.item()) / 1024**2,
        "oracle_stage_event_count": int(oracle_stage_event_count.item()),
        "oracle_stage_ms": float(oracle_stage_ms.item()),
        "oracle_stage_prefill_ms": float(oracle_stage_prefill_ms.item()),
        "oracle_stage_decode_ms": float(oracle_stage_decode_ms.item()),
        "expert_timing_enabled": int(str(os.environ.get("SPARSE_EXPERT_TIMING", "0")).strip().lower() not in ("0", "false", "no", "off")),
        "moe_router_ms": float(moe_router_ms.item()),
        "moe_expert_ms": float(moe_expert_ms.item()),
        "moe_shared_ms": float(moe_shared_ms.item()),
        "moe_router_ms_rankmax": float(moe_router_ms_rankmax.item()),
        "moe_expert_ms_rankmax": float(moe_expert_ms_rankmax.item()),
        "moe_shared_ms_rankmax": float(moe_shared_ms_rankmax.item()),
        "moe_timing_calls": int(moe_timing_calls.item()),
        "moe_router_avg_ms": float(moe_router_ms.item()) / max(int(moe_timing_calls.item()) // 3, 1),
        "moe_expert_avg_ms": float(moe_expert_ms.item()) / max(int(moe_timing_calls.item()) // 3, 1),
        "moe_shared_avg_ms": float(moe_shared_ms.item()) / max(int(moe_timing_calls.item()) // 3, 1),
        "moe_router_avg_ms_rankmax": float(moe_router_ms_rankmax.item()) / max(int(moe_timing_calls.item()) // (3 * max(int(env.world), 1)), 1),
        "moe_expert_avg_ms_rankmax": float(moe_expert_ms_rankmax.item()) / max(int(moe_timing_calls.item()) // (3 * max(int(env.world), 1)), 1),
        "moe_shared_avg_ms_rankmax": float(moe_shared_ms_rankmax.item()) / max(int(moe_timing_calls.item()) // (3 * max(int(env.world), 1)), 1),
        "rs_mlp_ms": float(rs_mlp_ms.item()),
        "rs_mlp_ms_rankmax": float(rs_mlp_ms_rankmax.item()),
        "rs_mlp_calls": int(rs_mlp_calls.item()),
        "rs_next_layer_pred_calls": int(rs_next_layer_pred_calls.item()),
        "rs_mlp_avg_ms": float(rs_mlp_ms.item()) / max(int(rs_mlp_calls.item()), 1),
        "rs_mlp_avg_ms_rankmax": float(rs_mlp_ms_rankmax.item()) / max(int(rs_mlp_calls.item()) // max(int(env.world), 1), 1),
        "demand_h2d_exposed_ms": float(demand_h2d_exposed_ms.item()),
        "demand_h2d_exposed_prefill_ms": float(demand_h2d_exposed_prefill_ms.item()),
        "demand_h2d_exposed_decode_ms": float(demand_h2d_exposed_decode_ms.item()),
        "sim_demand_h2d_exposed_ms": float(sim_demand_h2d_exposed_ms.item()),
        "sim_demand_h2d_exposed_prefill_ms": float(sim_demand_h2d_exposed_prefill_ms.item()),
        "sim_demand_h2d_exposed_decode_ms": float(sim_demand_h2d_exposed_decode_ms.item()),
        "demand_h2d_exposed_ms_rankmax": float(demand_h2d_exposed_ms_rankmax.item()),
        "demand_h2d_exposed_prefill_ms_rankmax": float(demand_h2d_exposed_prefill_ms_rankmax.item()),
        "demand_h2d_exposed_decode_ms_rankmax": float(demand_h2d_exposed_decode_ms_rankmax.item()),
        "sim_demand_h2d_exposed_ms_rankmax": float(sim_demand_h2d_exposed_ms_rankmax.item()),
        "sim_demand_h2d_exposed_decode_ms_rankmax": float(sim_demand_h2d_exposed_decode_ms_rankmax.item()),
        "sim_h2d_extra_sleep_ms": float(sim_h2d_extra_sleep_ms.item()),
        "demand_h2d_event_count": int(demand_h2d_event_count.item()),
        "prefetch_h2d_copy_ms": float(prefetch_h2d_copy_ms.item()),
        "prefetch_h2d_event_count": int(prefetch_h2d_event_count.item()),
        "prefetch_wait_ms": float(prefetch_wait_ms.item()),
        "prefetch_wait_event_count": int(prefetch_wait_event_count.item()),
        "prefetch_oom_skip_count": int(prefetch_oom_skip_count.item()),
        "plain_ep_executor_calls": int(plain_ep_executor_calls.item()),
        "epipe_executor_calls": int(epipe_executor_calls.item()),
        "epipe_executor_attempt_count": int(epipe_executor_attempt_count.item()),
        "expert_exec_impl": (
            "epipe_or_mixed"
            if int(epipe_executor_calls.item()) > 0
            else (
                (
                    "plain_ep_legacy_full_mask"
                    if plain_ep_exec_mode == "full_mask"
                    else "plain_ep_legacy_unique_mask"
                )
                if int(plain_ep_executor_calls.item()) > 0
                and plain_ep_exec_mode in ("unique_mask", "full_mask")
                else ("plain_ep" if int(plain_ep_executor_calls.item()) > 0 else "unknown")
            )
        ),
        "sim_expert_exec_ms": float(sim_expert_exec_ms.item()),
        "sim_expert_exec_ms_rankmax": float(sim_expert_exec_ms_rankmax.item()),
        "sim_compute_extra_sleep_ms": float(sim_compute_extra_sleep_ms.item()),
        "sum_h2d_exposed_ms": float(demand_h2d_exposed_ms.item()),
        "max_rank_h2d_exposed_ms": float(demand_h2d_exposed_ms_rankmax.item()),
        "sum_h2d_exposed_decode_ms": float(demand_h2d_exposed_decode_ms.item()),
        "max_rank_h2d_exposed_decode_ms": float(demand_h2d_exposed_decode_ms_rankmax.item()),
        "sum_sim_h2d_exposed_ms": float(sim_demand_h2d_exposed_ms.item()),
        "max_rank_sim_h2d_exposed_ms": float(sim_demand_h2d_exposed_ms_rankmax.item()),
        "sum_sim_h2d_exposed_decode_ms": float(sim_demand_h2d_exposed_decode_ms.item()),
        "max_rank_sim_h2d_exposed_decode_ms": float(sim_demand_h2d_exposed_decode_ms_rankmax.item()),
        "sum_comm_ms": sum(float((r or {}).get("comm_ms", 0.0)) for r in (gathered_rank_breakdown or [])),
        "max_rank_comm_ms": max([float((r or {}).get("comm_ms", 0.0)) for r in (gathered_rank_breakdown or [])] or [0.0]),
        "sum_comm_inter_node_sim_ms": float(weaknet_inter_sim_ms.item()),
        "max_rank_comm_inter_node_sim_ms": float(weaknet_inter_sim_ms_rankmax.item()),
        "sum_compute_ms": float(moe_expert_ms.item()),
        "max_rank_compute_ms": float(moe_expert_ms_rankmax.item()),
        "sum_sim_expert_exec_ms": float(sim_expert_exec_ms.item()),
        "max_rank_sim_expert_exec_ms": float(sim_expert_exec_ms_rankmax.item()),
        "sum_runtime_overhead_ms": sum(float((r or {}).get("runtime_overhead_ms", 0.0)) for r in (gathered_rank_breakdown or [])),
        "max_rank_runtime_overhead_ms": max([float((r or {}).get("runtime_overhead_ms", 0.0)) for r in (gathered_rank_breakdown or [])] or [0.0]),
        "per_rank_breakdown": gathered_rank_breakdown or [],
        "per_node_breakdown": per_node_breakdown,
        "skvs_accounted_mb": float(skvs.item()) / 1024**2,

        "activation_comm_semantics": "eval_core metrics only; excludes verbose debug prints",
        "activation_remote_comm_semantics": "logical off-rank dispatch+gather payload; excludes count exchange and self-rank traffic",
        "activation_inter_node_semantics": "logical cross-node dispatch+gather payload under SPARSE_SIM_NODE_MAP; excludes self traffic and count exchange",
        "physical_moe_comm_trace_semantics": "Opt-in attribution trace. Each sample aligns one MoE-layer execution across ranks and reports the maximum and mean of locally measured NCCL communication time; it is disabled in performance runs.",
        "weaknet_sim_semantics": "simulated inter-node time = per-nonzero-peer latency + payload/BW for logical cross-node payload; weaknet sleep affects wall time only when SPARSE_SIM_ENABLE_WEAKNET=1",
        "weaknet_packet_semantics": "packet metrics count non-empty logical source-rank to destination-rank transfers after the active dispatch packing path; pair breakdown keys are logical src_node->dst_node. Optional weaknet_packet_trace is enabled by SPARSE_WEAKNET_PACKET_TRACE=1.",
        "ep_fused_weight_dispatch_semantics": "0 means standard EP baseline uses conventional separate hidden/id/router-weight dispatch A2A calls; 1 enables the fused hidden+weight dispatch optimization explicitly.",
        "weaknet_packet_bucket_semantics": "packet bucket counts are disjoint by inter-node item count: le1=1, le2=2, le4=3-4, le8=5-8, gt8=>8.",
        "rs_admission_breakdown_semantics": "RS counters trace prediction candidates through local-owner filtering, cache/score filtering, prefetch plan issue, cache admission, and completion-before-use. RS never changes router output in planner-only rs_prefetch mode.",
        "node_grouped_inter_send_breakdown_semantics": "node_grouped_inter_send_ms is wall instrumentation around the Stage-1 node send block; Stage-2 send/fanout timing is included in node_grouped_intra_fanout_ms. node_grouped_inter_send_stage1_weaknet_sleep_ms and stage2_weaknet_sleep_ms split injected simulated sleeps; node_grouped_inter_send_weaknet_sleep_ms is their sum. stage1/stage2 collective fields are payload A2A timing proxies, not physical NIC counters. node_grouped_inter_send_stage1_non_sleep_ms subtracts Stage-1 weaknet sleep from the Stage-1 block. node_grouped_intra_fanout_stage2_non_sleep_ms subtracts Stage-2 weaknet sleep from fanout. node_grouped_total_overhead_non_sleep_ms subtracts all injected weaknet sleep from node_grouped_total_overhead_ms.",
        "kv_comm_semantics": "eval_core metrics only; excludes ad-hoc debug-only accounting",
        "skvs_accounted_semantics": "drop/accounting from skvs path only",

        "prefetch_hit_count": prefetch_hit_val,
        "prefetch_wasted_count": prefetch_wasted_val,
        "cache_hit_count": cache_hit_val,
        "cache_miss_count": cache_miss_val,
        "cache_hit_count_including_extra_cache": cache_hit_val,
        "main_cache_hit_count": main_cache_hit_val,
        "main_cache_miss_count": main_cache_miss_val,
        "main_cache_lookup_total": resident_access_total,
        "main_cache_hit_rate": main_cache_hit_rate,
        "main_cache_effective_service_hit_count": main_cache_effective_service_hit_count,
        "main_cache_effective_service_hit_rate": main_cache_effective_service_hit_rate,
        "extra_cache_service_hit_count": epipe_extra_cache_hit_val,
        "extra_cache_service_hit_rate": extra_cache_service_hit_rate,
        "cache_hit_semantics": "cache_hit_count/cache_hit_rate include prefill extra-cache hits for backward compatibility; use main_cache_hit_rate to exclude EPIPE extra-cache service hits.",
        "rs_prefetch_eval_events": int(rs_prefetch_eval_events.item()),
        "rs_prefetch_eval_pred_total": rs_prefetch_pred_total_val,
        "rs_prefetch_eval_actual_total": rs_prefetch_actual_total_val,
        "rs_prefetch_eval_overlap_total": rs_prefetch_overlap_total_val,
        "rs_prefetch_eval_precision": rs_prefetch_precision,
        "rs_prefetch_eval_recall": rs_prefetch_recall,
        "prefetch_budget_zero_count": int(prefetch_budget_zero_count.item()),
        "prefetch_budget_positive_count": int(prefetch_budget_positive_count.item()),
        "prefetch_admitted_count": prefetch_admitted_total_val,
        "rs_pred_candidate_count": int(rs_pred_candidate_count.item()),
        "rs_pred_local_candidate_count": int(rs_pred_local_candidate_count.item()),
        "rs_pred_nonlocal_candidate_count": int(rs_pred_nonlocal_candidate_count.item()),
        "rs_pred_local_candidate_fraction": (
            float(rs_pred_local_candidate_count.item()) / max(float(rs_pred_candidate_count.item()), 1.0)
        ),
        "rs_candidate_filter_in_count": int(rs_candidate_filter_in_count.item()),
        "rs_candidate_filter_out_count": int(rs_candidate_filter_out_count.item()),
        "rs_candidate_filter_out_cached_count": int(rs_candidate_filter_out_cached_count.item()),
        "rs_candidate_filter_out_low_score_count": int(rs_candidate_filter_out_low_score_count.item()),
        "rs_candidate_survived_count": int(rs_candidate_survived_count.item()),
        "rs_candidate_survival_rate": (
            float(rs_candidate_survived_count.item()) / max(float(rs_candidate_filter_in_count.item()), 1.0)
        ),
        "prefetch_issued_count": int(prefetch_issued_count.item()),
        "prefetch_completed_count": int(prefetch_completed_count.item()),
        "prefetch_complete_before_use_count": int(prefetch_complete_before_use_count.item()),
        "prefetch_completion_rate": (
            float(prefetch_completed_count.item()) / max(float(prefetch_issued_count.item()), 1.0)
        ),
        "prefetch_complete_before_use_rate": (
            float(prefetch_complete_before_use_count.item()) / max(float(prefetch_hit_val), 1.0)
        ),
        "prefetch_rejected_active_replace": int(prefetch_rejected_active_replace.item()),
        "prefetch_rejected_margin": int(prefetch_rejected_margin.item()),
        "prefetch_rejected_cache_full": int(prefetch_rejected_cache_full.item()),
        "prefetch_rejected_low_score": int(prefetch_rejected_low_score.item()),
        "scp_transient_prefill_cache_enabled": int(os.environ.get("SPARSE_SCP_TRANSIENT_PREFILL_CACHE", "0").strip().lower() not in ("0", "false", "no", "off")),
        "prefill_scp_transient_load_count": int(prefill_scp_transient_load_count.item()),
        "prefill_scp_transient_hit_count": int(prefill_scp_transient_hit_count.item()),
        "prefill_scp_transient_promote_count": int(prefill_scp_transient_promote_count.item()),
        "prefill_scp_transient_evict_count": int(prefill_scp_transient_evict_count.item()),
        "prefill_scp_transient_decode_reuse_rate": (
            float(prefill_scp_transient_hit_count.item()) / max(float(prefill_scp_transient_load_count.item()), 1.0)
        ),
        "prefetch_admission_rate": (
            float(prefetch_admitted_total_val) / max(float(rs_candidate_survived_count.item()), 1.0)
        ),
        "cache_ttl_steps": int(os.environ.get("SPARSE_CACHE_TTL_STEPS", "0") or 0),
        "ttl_prevented_evictions": int(ttl_prevented_evictions.item()),
        "cache_eviction_fallback_all_protected": int(cache_eviction_fallback_all_protected.item()),
        "prefill_decode_warmup": int(os.environ.get("SPARSE_PREFILL_DECODE_WARMUP", "0").strip().lower() not in ("0", "false", "no", "off")),
        "warmup_topk_per_layer": int(os.environ.get("SPARSE_WARMUP_TOPK_PER_LAYER", "1") or 1),
        "warmup_source": str(os.environ.get("SPARSE_WARMUP_SOURCE", "freq")),
        "warmup_loaded_experts": int(warmup_loaded_experts.item()),
        "warmup_skipped_resident": int(warmup_skipped_resident.item()),
        "warmup_h2d_mb": float(warmup_h2d.item()) / 1024**2,
        "warmup_time_ms": float(warmup_time_ms.item()),
        "transition_prefetch": int(os.environ.get("SPARSE_TRANSITION_PREFETCH", "0").strip().lower() not in ("0", "false", "no", "off")),
        "transition_table_path": str(os.environ.get("SPARSE_TRANSITION_TABLE_PATH", "")),
        "transition_topk": int(os.environ.get("SPARSE_TRANSITION_TOPK", "1") or 1),
        "transition_no_replace": int(os.environ.get("SPARSE_TRANSITION_NO_REPLACE", "1").strip().lower() not in ("0", "false", "no", "off")),
        "transition_lookup_calls": int(transition_lookup_calls.item()),
        "transition_lookup_overhead_ms": float(transition_lookup_overhead_ms.item()),
        "transition_prefetch_admitted": int(transition_prefetch_admitted.item()),
        "transition_prefetch_hit": int(transition_prefetch_hit.item()),
        "transition_prefetch_late": int(transition_prefetch_late.item()),
        "transition_prefetch_wasted": int(transition_prefetch_wasted.item()),
        "transition_safe_evict_denied": int(transition_safe_evict_denied.item()),
        "transition_no_replace_denied": int(transition_no_replace_denied.item()),
        "epipe_enable": int(os.environ.get("SPARSE_EPIPE_ENABLE", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_shadow": int(os.environ.get("SPARSE_EPIPE_SHADOW", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_mode": str(os.environ.get("SPARSE_EPIPE_MODE", "upfront_scan")),
        "prefill_expert_math": str(os.environ.get("SPARSE_PREFILL_EXPERT_MATH", "native")).strip().lower(),
        "epipe_decode_rs_mode": str(os.environ.get("SPARSE_EPIPE_DECODE_RS_MODE", os.environ.get("SPARSE_EPIPE_DECODE_WITH_RS_MODE", "late_only"))),
        "epipe_decode_with_rs_mode": str(os.environ.get("SPARSE_EPIPE_DECODE_WITH_RS_MODE", os.environ.get("SPARSE_EPIPE_DECODE_RS_MODE", "late_only"))),
        "epipe_debug_counters": int(os.environ.get("SPARSE_EPIPE_DEBUG_COUNTERS", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_timing_sample_rate": float(os.environ.get("SPARSE_EPIPE_TIMING_SAMPLE_RATE", "0.0") or 0.0),
        "prefetch_wait_timing_sample_rate": float(os.environ.get("SPARSE_PREFETCH_WAIT_TIMING_SAMPLE_RATE", "0.0") or 0.0),
        "demand_h2d_timing_sample_rate": float(os.environ.get("SPARSE_DEMAND_H2D_TIMING_SAMPLE_RATE", "0.0") or 0.0),
        "epipe_lookahead_experts": int(os.environ.get("SPARSE_EPIPE_LOOKAHEAD_EXPERTS", "4") or 4),
        "epipe_lookahead_min_ready_items": int(os.environ.get("SPARSE_EPIPE_LOOKAHEAD_MIN_READY_ITEMS", "64") or 64),
        "epipe_deferred_lru_commit": int(os.environ.get("SPARSE_EPIPE_DEFERRED_LRU_COMMIT", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_decode_rs_commit": int(os.environ.get("SPARSE_EPIPE_DECODE_RS_COMMIT", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_decode_no_rs_commit": int(os.environ.get("SPARSE_EPIPE_DECODE_NO_RS_COMMIT", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_enable": int(os.environ.get("SPARSE_EPIPE_PREFILL_ENABLE", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_shadow": int(os.environ.get("SPARSE_EPIPE_PREFILL_SHADOW", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_policy": str(os.environ.get("SPARSE_EPIPE_PREFILL_POLICY", "deferred_lru_commit")),
        "epipe_prefill_commit": int(os.environ.get("SPARSE_EPIPE_PREFILL_COMMIT", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_mark_transient": int(os.environ.get("SPARSE_EPIPE_PREFILL_MARK_TRANSIENT", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_extra_cache_slots": int(os.environ.get("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_SLOTS", "0") or 0),
        "epipe_prefill_extra_cache_decode_read": int(os.environ.get("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_DECODE_READ", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_extra_cache_clear_at_decode": int(os.environ.get("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_CLEAR_AT_DECODE", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_extra_cache_persist_requests": int(os.environ.get("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_PERSIST_REQUESTS", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_prefill_extra_cache_min_free_mb": float(os.environ.get("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_MIN_FREE_MB", os.environ.get("SPARSE_EPIPE_PREFILL_MIN_FREE_MB", "1024")) or 0.0),
        "epipe_prefill_extra_cache_max_mb": float(os.environ.get("SPARSE_EPIPE_PREFILL_EXTRA_CACHE_MAX_MB", "128") or 0.0),
        "epipe_rs_handle_late_prefetch": int(os.environ.get("SPARSE_EPIPE_RS_HANDLE_LATE_PREFETCH", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_rs_handle_demand_miss": int(os.environ.get("SPARSE_EPIPE_RS_HANDLE_DEMAND_MISS", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_rs_force_demand_stage": int(os.environ.get("SPARSE_EPIPE_RS_FORCE_DEMAND_STAGE", "0").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_rs_demand_commit": int(os.environ.get("SPARSE_EPIPE_RS_DEMAND_COMMIT", "1").strip().lower() not in ("0", "false", "no", "off")),
        "epipe_rs_demand_stage_threshold_ms": float(os.environ.get("SPARSE_EPIPE_RS_DEMAND_STAGE_THRESHOLD_MS", "0.0") or 0.0),
        **{
            name: (
                float(epipe_tensors[name].item())
                if name in epipe_float_fields
                else int(epipe_tensors[name].item())
            )
            for name in epipe_fields
        },
        "epipe_demand_h2d_mb": float(epipe_tensors["epipe_demand_h2d_bytes"].item()) / 1024**2,
        "epipe_staged_h2d_mb": float(epipe_tensors["epipe_staged_h2d_bytes"].item()) / 1024**2,
        "epipe_prefill_extra_cache_h2d_mb": float(epipe_tensors["epipe_prefill_extra_cache_h2d_bytes"].item()) / 1024**2,
        "epipe_missing_h2d_est_mb": float(epipe_tensors["epipe_missing_h2d_bytes_est"].item()) / 1024**2,
        "epipe_prefill_multistage_est_expert_h2d_mb": (
            float(epipe_tensors["epipe_prefill_multistage_est_expert_h2d_bytes"].item()) / 1024**2
        ),
        "total_demand_h2d_mb_including_epipe": float(h2d_demand.item()) / 1024**2,
        "prefetch_admission_upper_recall": prefetch_admission_upper_recall,
        "prefetch_recall_gap_to_admission_upper": prefetch_recall_gap_to_admission_upper,
        "effective_service_hit_count": effective_service_hit_count,
        "resident_access_total": resident_access_total,
        "effective_service_total": effective_service_total,
        # backward-compatible alias: old cache_hit_rate actually means resident hit rate
        "cache_hit_rate": resident_hit_rate,
        "resident_hit_rate": resident_hit_rate,
        "effective_service_hit_rate": effective_service_hit_rate,
        "prefetch_use_rate": prefetch_use_rate,

        "peak_allocated_mb": float(peak_alloc.item()),
        "peak_reserved_mb": float(peak_rsvd.item()),

        **owner_local_summary,
        "skvs_debug": skvs_dbg,
        **step_summary,
    }

    if args.collect_per_layer:
        record["per_layer"] = per_layer_records

    record["repeat_index"] = int(repeat_idx)
    return record


def aggregate_repeat_records(records: List[Dict[str, Any]], args) -> Dict[str, Any]:
    if not records:
        raise RuntimeError("No repeat records to aggregate.")

    base = dict(records[0])
    summary = {k: v for k, v in base.items() if k not in ("repeat_index",)}

    numeric_keys = [
        "wall_total_ms", "wall_prefill_ms", "wall_decode_ms", "wall_ttft_ms",
        "e2e_total_tok_s", "e2e_output_tok_s",
        "throughput_tok_s", "decode_throughput_tok_s",
        "comm_mb_activation", "comm_mb_activation_remote", "comm_mb_kv", "skvs_accounted_mb",
        "sim_weaknet_enabled", "sim_heterogeneity_enabled", "sim_heterogeneity_sleep",
        "sim_inter_node_bw_raw", "sim_inter_node_bw_mb_s", "sim_inter_node_lat_ms",
        "comm_mb_activation_intra_node", "comm_mb_activation_inter_node",
        "weaknet_intra_node_item_count", "weaknet_inter_node_item_count",
        "weaknet_intra_node_transfer_count", "weaknet_inter_node_transfer_count",
        "weaknet_inter_node_packet_item_sumsq", "weaknet_inter_node_packet_max_items",
        "weaknet_inter_node_packet_avg_items", "weaknet_inter_node_packet_le1_count",
        "weaknet_inter_node_packet_le2_count", "weaknet_inter_node_packet_le4_count",
        "weaknet_inter_node_packet_le8_count", "weaknet_inter_node_packet_gt8_count",
        "weaknet_inter_node_small_packet_threshold", "weaknet_inter_node_small_packet_count",
        "weaknet_inter_node_small_packet_item_count", "weaknet_inter_node_small_packet_ratio",
        "weaknet_packet_trace_enabled", "physical_moe_comm_trace_enabled",
        "physical_moe_comm_sample_count",
        "physical_moe_comm_tail_p50_ms", "physical_moe_comm_tail_p95_ms",
        "physical_moe_comm_tail_p99_ms", "physical_moe_comm_rankmean_p50_ms",
        "physical_moe_comm_rankmax_over_mean_p50",
        "physical_moe_comm_rankmax_over_mean_p95",
        "weaknet_inter_node_sim_ms", "weaknet_inter_node_sim_ms_rankmax",
        "rank_grouped_pair_entries", "rank_grouped_packet_entries", "rank_grouped_dispatch_calls",
        "rank_grouped_packet_ratio", "rank_grouped_theoretical_hidden_saving_ratio",
        "node_grouped_pair_entries", "node_grouped_node_packet_entries", "node_grouped_rank_packet_entries",
        "node_grouped_dispatch_calls", "node_grouped_node_packet_ratio", "node_grouped_rank_packet_ratio",
        "node_grouped_theoretical_node_hidden_saving_ratio",
        "node_grouped_adaptive", "node_grouped_min_saving_ms", "node_grouped_overhead_margin",
        "node_grouped_est_overhead_ms", "node_grouped_layer_policy_path", "node_grouped_layer_policy_default",
        "node_grouped_tensor_fastpath", "node_grouped_tensor_fastpath_prefill",
        "node_grouped_tensor_fastpath_decode", "node_grouped_triton_pack",
        "node_grouped_triton_pack_prefill", "node_grouped_triton_pack_decode",
        "node_grouped_triton_strict", "node_grouped_same_node_fastpath",
        "node_grouped_same_node_fastpath_prefill", "node_grouped_same_node_fastpath_decode",
        "node_grouped_stage2_local_rank_fastpath", "node_grouped_pack_meta",
        "node_grouped_stage2_local_rank_fastpath_prefill", "node_grouped_stage2_local_rank_fastpath_decode",
        "node_grouped_pack_meta_prefill", "node_grouped_pack_meta_decode",
        "node_grouped_pack_meta_float32", "node_grouped_pack_meta_float32_prefill",
        "node_grouped_pack_meta_float32_decode", "node_grouped_identity_combine_fastpath",
        "node_grouped_identity_combine_fastpath_prefill", "node_grouped_identity_combine_fastpath_decode",
        "node_grouped_skip_dummy_a2a",
        "node_grouped_pack_ms", "node_grouped_metadata_ms", "node_grouped_intra_fanout_ms",
        "node_grouped_inter_send_ms", "node_grouped_inter_send_weaknet_sleep_ms", "node_grouped_inter_send_stage1_weaknet_sleep_ms", "node_grouped_inter_send_stage2_weaknet_sleep_ms", "node_grouped_inter_send_stage1_non_sleep_ms", "node_grouped_intra_fanout_stage2_non_sleep_ms", "node_grouped_total_overhead_non_sleep_ms", "node_grouped_inter_send_stage1_collective_ms", "node_grouped_inter_send_stage2_collective_ms", "node_grouped_metadata_exchange_ms", "node_grouped_inter_send_non_sleep_ms", "node_grouped_gather_ms", "node_grouped_total_overhead_ms",
        "node_grouped_same_node_packet_entries", "node_grouped_cross_node_packet_entries",
        "node_grouped_same_node_packet_fraction", "node_grouped_same_node_bypass_mb",
        "node_grouped_stage2_local_rank_packet_entries", "node_grouped_stage2_remote_rank_packet_entries",
        "node_grouped_stage2_local_rank_packet_fraction", "node_grouped_stage2_local_bypass_mb",
        "node_grouped_metadata_collective_count", "node_grouped_metadata_mb",
        "node_grouped_counts_build_ms", "node_grouped_splits_build_ms", "node_grouped_reverse_map_build_ms",
        "node_grouped_rank_to_node_gather_ms", "node_grouped_node_to_token_restore_ms",
        "node_grouped_final_combine_ms", "node_grouped_combine_intermediate_buffer_mb",
        "node_grouped_allocation_ms", "node_grouped_allocation_count", "node_grouped_allocation_mb",
        "node_grouped_workspace_reuse_hit",
        "node_grouped_enabled_layers", "node_grouped_fallback_layers", "node_grouped_estimated_saving_ms",
        "ers_enabled", "ers_mode", "ers_phase", "ers_topm", "ers_margin", "ers_max_replace", "ers_cache_aware", "ers_legacy_nodepatch_alias_used", "ers_runtime_semantics",
        "nodepatch_enabled", "nodepatch_topm", "nodepatch_margin", "nodepatch_max_replace",
        "nodepatch_keep_top1", "nodepatch_only_cross_node",
        "nodepatch_lean_repair", "nodepatch_lean_candidate_scope", "nodepatch_lean_resident_nearest",
        "nodepatch_enabled_calls", "nodepatch_tokens_seen", "nodepatch_cross_node_tokens",
        "nodepatch_repaired_tokens", "nodepatch_replacement_count",
        "nodepatch_repairable_token_fraction", "nodepatch_repairable_cross_node_fraction",
        "nodepatch_transfer_event_reduction_pct", "nodepatch_cross_node_item_reduction_pct",
        "nodepatch_avg_unique_nodes_before", "nodepatch_avg_unique_nodes_after",
        "nodepatch_avg_utility_loss_logit", "nodepatch_avg_utility_loss_prob",
        "nodepatch_avg_max_margin_logit", "nodepatch_avg_max_margin_prob",
        "nodepatch_max_margin_logit", "nodepatch_max_margin_prob",
        "nodepatch_event_aware", "nodepatch_event_aggressive", "nodepatch_event_only_empty_packet",
        "nodepatch_event_require_local_candidate", "nodepatch_event_packet_max_items",
        "nodepatch_packet_emptying_first", "nodepatch_cache_aware",
        "nodepatch_cache_require_owner_local", "nodepatch_require_added_resident",
        "nodepatch_runtime_packet_events_before", "nodepatch_runtime_packet_events_after",
        "nodepatch_runtime_packet_events_eliminated", "nodepatch_event_aware_skipped_repairs",
        "nodepatch_cache_aware_skipped_repairs", "nodepatch_cache_aware_filtered_candidates",
        "nodepatch_packet_emptying_candidate_packets", "nodepatch_packet_emptying_feasible_packets",
        "nodepatch_packet_emptying_rejected_packets",
        "nodepatch_route_cost_before", "nodepatch_route_cost_after", "nodepatch_route_cost_gain",
        "nodepatch_route_cost_reduction_pct", "nodepatch_avg_route_cost_gain_per_replacement",
        "nodepatch_avg_replacement_rank_gap", "nodepatch_rank_gap_max",
        "nodepatch_avg_old_prob", "nodepatch_avg_new_prob", "nodepatch_avg_prob_delta",
        "nodepatch_candidate_not_prepared_reject_count", "nodepatch_candidate_no_comm_gain_reject_count",
        "nodepatch_candidate_scope_reject_count", "nodepatch_candidate_duplicate_reject_count",
        "nodepatch_candidate_rank_gap_reject_count", "nodepatch_candidate_cache_reject_count",
        "nodepatch_candidate_total_reject_count", "nodepatch_no_replace_position_count",
        "prefill_scp_temporary_load_count", "prefill_scp_temporary_hit_count",
        "prefill_scp_temporary_promote_count", "prefill_scp_temporary_evict_count",
        "prefill_scp_temporary_resident_at_decode_start", "prefill_scp_temporary_demote_count",
        "prefill_scp_temporary_resident_at_decode_end", "prefill_scp_temporary_decode_reuse_rate",
        "resolver_decode_total_queries", "resolver_decode_rr_owner_count",
        "resolver_decode_scp_replica_selected_count", "resolver_decode_rr_fallback_count",
        "resolver_decode_replica_reject_not_ready_count", "resolver_decode_replica_reject_no_comm_gain_count",
        "resolver_decode_replica_cache_ready_count", "resolver_decode_replica_prefetch_ready_count",
        "resolver_decode_route_cost_before", "resolver_decode_route_cost_after", "resolver_decode_route_cost_gain",
        "scp_phase_route_diag_enabled",
        "scp_prefill_route_diag_items", "scp_prefill_route_diag_replica_items",
        "scp_prefill_route_diag_replica_item_rate",
        "scp_prefill_route_diag_cross_items_before", "scp_prefill_route_diag_cross_items_after",
        "scp_prefill_route_diag_cross_item_reduction_pct",
        "scp_prefill_route_diag_cross_events_before", "scp_prefill_route_diag_cross_events_after",
        "scp_prefill_route_diag_cross_event_reduction_pct",
        "scp_prefill_route_diag_cost_before", "scp_prefill_route_diag_cost_after",
        "scp_prefill_route_diag_cost_reduction_pct",
        "scp_decode_route_diag_items", "scp_decode_route_diag_replica_items",
        "scp_decode_route_diag_replica_item_rate",
        "scp_decode_route_diag_cross_items_before", "scp_decode_route_diag_cross_items_after",
        "scp_decode_route_diag_cross_item_reduction_pct",
        "scp_decode_route_diag_cross_events_before", "scp_decode_route_diag_cross_events_after",
        "scp_decode_route_diag_cross_event_reduction_pct",
        "scp_decode_route_diag_cost_before", "scp_decode_route_diag_cost_after",
        "scp_decode_route_diag_cost_reduction_pct",
        "rs_prefetch_target_resolver_match_count", "rs_prefetch_target_resolver_mismatch_count",
        "rs_prefetch_would_be_wrong_without_resolver_count",
        "scp_table_local_replica_route_count", "scp_table_rr_fallback_count",
        "scp_table_cross_node_route_count", "scp_table_same_node_route_count",
        "scp_route_cost_before_sum", "scp_route_cost_after_sum",
        "scp_lookup_calls", "scp_lookup_overhead_ms",
        "scp_route_cost_reduction_pct", "valid_comparison",
        "rs_scp_target_mismatch_count",
        "expert_h2d_mb", "expert_h2d_prefetch_mb", "expert_h2d_demand_mb",
        "oracle_stage_h2d_mb", "oracle_stage_event_count", "oracle_stage_ms",
        "oracle_stage_prefill_ms", "oracle_stage_decode_ms",
        "moe_router_ms", "moe_expert_ms", "moe_shared_ms", "moe_timing_calls",
        "moe_router_avg_ms", "moe_expert_avg_ms", "moe_shared_avg_ms",
        "moe_router_ms_rankmax", "moe_expert_ms_rankmax", "moe_shared_ms_rankmax",
        "moe_router_avg_ms_rankmax", "moe_expert_avg_ms_rankmax", "moe_shared_avg_ms_rankmax",
        "rs_mlp_ms", "rs_mlp_ms_rankmax", "rs_mlp_calls", "rs_next_layer_pred_calls",
        "rs_mlp_avg_ms", "rs_mlp_avg_ms_rankmax",
        "demand_h2d_exposed_ms", "demand_h2d_exposed_prefill_ms", "demand_h2d_exposed_decode_ms",
        "sim_demand_h2d_exposed_ms", "sim_demand_h2d_exposed_prefill_ms", "sim_demand_h2d_exposed_decode_ms",
        "sim_demand_h2d_exposed_ms_rankmax", "sim_demand_h2d_exposed_decode_ms_rankmax", "sim_h2d_extra_sleep_ms",
        "demand_h2d_exposed_ms_rankmax", "demand_h2d_exposed_prefill_ms_rankmax", "demand_h2d_exposed_decode_ms_rankmax",
        "demand_h2d_event_count", "prefetch_h2d_copy_ms", "prefetch_h2d_event_count",
        "prefetch_wait_ms", "prefetch_wait_event_count", "prefetch_oom_skip_count",
        "plain_ep_executor_calls", "epipe_executor_calls", "epipe_executor_attempt_count",
        "sum_h2d_exposed_ms", "max_rank_h2d_exposed_ms",
        "sum_h2d_exposed_decode_ms", "max_rank_h2d_exposed_decode_ms",
        "sum_sim_h2d_exposed_ms", "max_rank_sim_h2d_exposed_ms",
        "sum_sim_h2d_exposed_decode_ms", "max_rank_sim_h2d_exposed_decode_ms",
        "sum_comm_ms", "max_rank_comm_ms", "sum_comm_inter_node_sim_ms", "max_rank_comm_inter_node_sim_ms",
        "sum_compute_ms", "max_rank_compute_ms",
        "sim_expert_exec_ms", "sim_expert_exec_ms_rankmax", "sim_compute_extra_sleep_ms",
        "sum_sim_expert_exec_ms", "max_rank_sim_expert_exec_ms",
        "sum_runtime_overhead_ms", "max_rank_runtime_overhead_ms",
        "prefetch_hit_count", "prefetch_wasted_count", "cache_hit_count", "cache_miss_count",
        "rs_prefetch_eval_events", "rs_prefetch_eval_pred_total", "rs_prefetch_eval_actual_total",
        "rs_prefetch_eval_overlap_total", "prefetch_budget_zero_count", "prefetch_budget_positive_count",
        "prefetch_admitted_count", "rs_pred_candidate_count", "rs_pred_local_candidate_count", "rs_pred_nonlocal_candidate_count",
        "rs_pred_local_candidate_fraction", "rs_candidate_filter_in_count", "rs_candidate_filter_out_count",
        "rs_candidate_filter_out_cached_count", "rs_candidate_filter_out_low_score_count", "rs_candidate_survived_count",
        "rs_candidate_survival_rate", "prefetch_issued_count", "prefetch_completed_count",
        "prefetch_complete_before_use_count", "prefetch_completion_rate", "prefetch_complete_before_use_rate",
        "prefetch_rejected_active_replace", "prefetch_rejected_margin", "prefetch_rejected_cache_full",
        "prefetch_rejected_low_score", "prefetch_admission_rate", "rs_prefetch_eval_precision", "rs_prefetch_eval_recall",
        "prefetch_admission_upper_recall", "prefetch_recall_gap_to_admission_upper",
        "cache_ttl_steps",
        "ttl_prevented_evictions", "cache_eviction_fallback_all_protected",
        "prefill_decode_warmup", "warmup_topk_per_layer", "warmup_loaded_experts",
        "warmup_skipped_resident", "warmup_h2d_mb", "warmup_time_ms",
        "transition_prefetch", "transition_topk", "transition_no_replace",
        "transition_lookup_calls", "transition_lookup_overhead_ms",
        "transition_prefetch_admitted", "transition_prefetch_hit", "transition_prefetch_late",
        "transition_prefetch_wasted", "transition_safe_evict_denied", "transition_no_replace_denied",
        "epipe_enable", "epipe_shadow", "epipe_mode", "epipe_decode_rs_mode", "epipe_decode_with_rs_mode",
        "epipe_shadow_count", "epipe_enabled_count",
        "epipe_fallback_count", "epipe_fallback_no_ready", "epipe_fallback_no_missing",
        "epipe_fallback_no_miss_trigger", "epipe_fallback_no_ready_lookahead",
        "epipe_fallback_low_ready_items",
        "epipe_fallback_prefill_disabled", "epipe_fallback_no_memory_headroom",
        "epipe_fallback_memory", "epipe_fallback_cache_full_no_temp", "epipe_fallback_exception",
        "epipe_decode_rs_miss_trigger_count", "epipe_decode_rs_skipped_no_miss_count",
        "epipe_prefill_miss_trigger_count", "epipe_prefill_shadow_opportunity_count",
        "epipe_rs_buffered_calls", "epipe_rs_buffer_flush_count",
        "epipe_rs_buffered_ready_group_count", "epipe_rs_buffered_ready_item_count",
        "epipe_rs_residual_scan_calls", "epipe_rs_residual_scan_ready_group_count",
        "epipe_rs_residual_scan_ready_item_count", "epipe_rs_residual_scan_residual_count",
        "epipe_rs_residual_scan_late_count", "epipe_rs_residual_scan_demand_count",
        "epipe_rs_residual_scan_no_ready_count",
        "epipe_rs_trigger_count", "epipe_rs_late_prefetch_trigger_count",
        "epipe_rs_demand_miss_trigger_count", "epipe_rs_late_prefetch_overlap_count",
        "epipe_rs_demand_stage_count", "epipe_rs_demand_stage_disabled_by_gate_count",
        "epipe_rs_demand_stage_force_count",
        "epipe_prefill_executor_calls", "epipe_prefill_stage_success_count",
        "epipe_decode_no_rs_executor_calls", "epipe_decode_no_rs_stage_success_count",
        "epipe_decode_rs_executor_calls", "epipe_decode_rs_stage_success_count",
        "epipe_prefill_enabled_count",
        "epipe_decode_no_rs_enabled_count", "epipe_decode_rs_late_overlap_count",
        "epipe_decode_rs_demand_stage_count", "epipe_rs_gate_skip_count",
        "epipe_rs_gate_reenable_count", "epipe_rs_buffer_too_small_count",
        "epipe_lookahead_ready_group_count", "epipe_lookahead_ready_item_count",
        "epipe_staged_expert_count", "epipe_staged_h2d_mb",
        "epipe_deferred_lru_commit_count", "epipe_prefill_transient_commit_count",
        "epipe_prefill_transient_demote_count", "epipe_prefill_transient_promote_count",
        "epipe_done_group_skip_count",
        "epipe_ready_expert_count", "epipe_missing_expert_count",
        "epipe_ready_item_count", "epipe_missing_item_count",
        "epipe_cache_free_slot_count", "epipe_cache_full_staged_count",
        "epipe_transient_payload_count", "epipe_prefill_transient_no_commit_count",
        "epipe_prefill_admission_bypass_count", "epipe_cache_insert_count",
        "epipe_prefill_cache_insert_count", "epipe_no_evict_count",
        "epipe_prefill_extra_cache_slots", "epipe_prefill_extra_cache_decode_read",
        "epipe_prefill_extra_cache_clear_at_decode", "epipe_prefill_extra_cache_persist_requests",
        "epipe_prefill_extra_cache_min_free_mb", "epipe_prefill_extra_cache_max_mb",
        "epipe_prefill_extra_cache_insert_count", "epipe_prefill_extra_cache_hit_count",
        "epipe_prefill_extra_cache_evict_count", "epipe_prefill_extra_cache_no_room_count",
        "epipe_prefill_extra_cache_guard_skip_count",
        "epipe_prefill_extra_cache_h2d_bytes", "epipe_prefill_extra_cache_h2d_mb",
        "epipe_prefill_extra_cache_resident_count",
        "epipe_prefill_extra_cache_resident_at_decode_start",
        "epipe_prefill_extra_cache_resident_at_decode_end",
        "epipe_missing_h2d_bytes_est", "epipe_missing_h2d_est_mb",
        "epipe_demand_h2d_bytes", "epipe_demand_h2d_mb", "epipe_staged_missing_count",
        "epipe_prefill_multistage_shadow_count",
        "epipe_prefill_multistage_ready_expert_count",
        "epipe_prefill_multistage_missing_expert_count",
        "epipe_prefill_multistage_ready_item_count",
        "epipe_prefill_multistage_missing_item_count",
        "epipe_prefill_multistage_top_missing_item_count_sum",
        "epipe_prefill_multistage_est_expert_h2d_bytes",
        "epipe_prefill_multistage_est_expert_h2d_mb",
        "epipe_prefill_multistage_est_ready_compute_items",
        "epipe_prefill_multistage_est_hidden_h2d_items_n1",
        "epipe_prefill_multistage_est_hidden_h2d_items_n2",
        "epipe_prefill_multistage_est_hidden_h2d_items_n4",
        "epipe_demand_h2d_ms", "epipe_wait_ms", "epipe_hidden_h2d_ms_est",
        "sampled_demand_h2d_exposed_decode_ms", "sampled_prefetch_wait_ms",
        "sampled_epipe_hidden_h2d_ms", "sampled_epipe_wait_ms",
        "sampled_epipe_control_overhead_ms",
        "total_demand_h2d_mb_including_epipe",
        "effective_service_hit_count", "resident_access_total", "effective_service_total",
        "cache_hit_count_including_extra_cache",
        "main_cache_hit_count", "main_cache_miss_count", "main_cache_lookup_total",
        "main_cache_hit_rate", "main_cache_effective_service_hit_count",
        "main_cache_effective_service_hit_rate",
        "extra_cache_service_hit_count", "extra_cache_service_hit_rate",
        "cache_hit_rate", "resident_hit_rate", "effective_service_hit_rate", "prefetch_use_rate",
        "peak_allocated_mb", "peak_reserved_mb",
        "owner_group_count", "owner_changed_samples",
        "owner_group_reconstruct_ms", "owner_group_plan_ms", "owner_group_migrate_ms",
        "owner_group_reconstruct_avg_ms", "owner_group_plan_avg_ms", "owner_group_migrate_avg_ms",
        "owner_empty_ranks_sum", "owner_empty_ranks_avg",
        "owner_hist_max_avg", "owner_hist_min_avg", "owner_hist_cv_avg", "owner_imbalance_ratio_avg",
        "owner_attention_ms_sum", "owner_attention_ms_rankmax", "owner_attention_calls", "owner_attention_avg_ms_per_call",
        "owner_moe_ms_sum", "owner_moe_ms_rankmax", "owner_moe_calls", "owner_moe_avg_ms_per_call",
        "decode_step_avg_ms", "decode_step_p50_ms", "decode_step_p95_ms", "decode_step_p99_ms", "decode_step_max_ms",
    ]

    def _coerce_numeric_summary_value(v: Any) -> float:
        if v is None:
            return 0.0
        if isinstance(v, str) and not v.strip():
            return 0.0
        return float(v)

    for k in numeric_keys:
        try:
            vals = [_coerce_numeric_summary_value(r.get(k, 0.0)) for r in records]
        except (TypeError, ValueError):
            # Keep non-numeric config/debug fields from killing a completed
            # multi-GPU run if they accidentally enter numeric_keys.
            for r in reversed(records):
                if k in r:
                    summary[k] = r.get(k)
                    break
            continue
        summary[k] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=0))

    # aggregate skvs_debug numerics if present
    if isinstance(base.get("skvs_debug"), dict):
        skvs_summary = {}
        sk_keys = set()
        for r in records:
            sk_keys.update(r.get("skvs_debug", {}).keys())
        for k in sorted(sk_keys):
            vals = [r.get("skvs_debug", {}).get(k, None) for r in records]
            if all(isinstance(v, (int, float)) for v in vals if v is not None):
                fvals = [float(v) for v in vals]
                skvs_summary[k] = float(np.mean(fvals))
                skvs_summary[f"{k}_std"] = float(np.std(fvals, ddof=0))
            else:
                skvs_summary[k] = vals[-1]
        summary["skvs_debug"] = skvs_summary
        for k, v in skvs_summary.items():
            if k not in summary:
                summary[k] = v

    for counter_field in (
        "weaknet_inter_node_pair_event_counts",
        "weaknet_inter_node_pair_item_counts",
    ):
        keys = sorted({
            str(k)
            for r in records
            for k in ((r.get(counter_field) or {}).keys() if isinstance(r.get(counter_field), dict) else [])
        })
        counter_summary: Dict[str, float] = {}
        counter_std: Dict[str, float] = {}
        for k in keys:
            vals = [float((r.get(counter_field) or {}).get(k, 0.0)) for r in records]
            counter_summary[k] = float(np.mean(vals))
            counter_std[k] = float(np.std(vals, ddof=0))
        summary[counter_field] = counter_summary
        summary[f"{counter_field}_std"] = counter_std

    if any(isinstance(r.get("weaknet_packet_trace"), list) and r.get("weaknet_packet_trace") for r in records):
        trace_limit = int(os.environ.get("SPARSE_WEAKNET_PACKET_TRACE_LIMIT", "20000") or 20000)
        merged_trace: List[Dict[str, Any]] = []
        for r in records:
            merged_trace.extend(x for x in (r.get("weaknet_packet_trace") or []) if isinstance(x, dict))
            if trace_limit > 0 and len(merged_trace) >= trace_limit:
                merged_trace = merged_trace[:trace_limit]
                break
        summary["weaknet_packet_trace"] = merged_trace

    if any(isinstance(r.get("physical_moe_comm_trace"), list) and r.get("physical_moe_comm_trace") for r in records):
        trace_limit = int(os.environ.get("SPARSE_PHYSICAL_MOE_COMM_TRACE_LIMIT", "50000") or 50000)
        merged_trace: List[Dict[str, Any]] = []
        for repeat_index, record in enumerate(records):
            for row in record.get("physical_moe_comm_trace") or []:
                if not isinstance(row, dict):
                    continue
                merged_trace.append({
                    **row,
                    "repeat_index": int(record.get("repeat_index", repeat_index)),
                })
                if trace_limit > 0 and len(merged_trace) >= trace_limit:
                    break
            if trace_limit > 0 and len(merged_trace) >= trace_limit:
                break
        summary["physical_moe_comm_trace"] = merged_trace
        summary.update(_physical_moe_comm_summary(merged_trace))

    if isinstance(base.get("per_layer"), dict):
        per_layer_summary: Dict[str, Dict[str, Any]] = {}
        layer_keys = set()
        for r in records:
            layer_keys.update((r.get("per_layer") or {}).keys())
        for layer_key in sorted(layer_keys, key=lambda x: int(x)):
            layer_records = [(r.get("per_layer") or {}).get(layer_key, {}) for r in records]
            out: Dict[str, Any] = {}
            metric_keys = set()
            for lr in layer_records:
                metric_keys.update(lr.keys())
            for mk in sorted(metric_keys):
                vals = [lr.get(mk, None) for lr in layer_records]
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if len(numeric_vals) == len(vals) and len(vals) > 0:
                    fvals = [float(v) for v in vals]
                    out[mk] = float(np.mean(fvals))
                    out[f"{mk}_std"] = float(np.std(fvals, ddof=0))
                else:
                    out[mk] = vals[-1]
            per_layer_summary[str(layer_key)] = out
        summary["per_layer"] = per_layer_summary

    def _aggregate_indexed_dict_list(field: str, id_key: str) -> Dict[str, Dict[str, Any]]:
        all_items = []
        for r in records:
            src = r.get(field) or {}
            if isinstance(src, dict):
                all_items.extend(x for x in src.values() if isinstance(x, dict))
            elif isinstance(src, list):
                all_items.extend(x for x in src if isinstance(x, dict))
        ids = sorted(
            {str(item.get(id_key)) for item in all_items if item.get(id_key) is not None},
            key=lambda x: int(float(x)),
        )
        out_all: Dict[str, Dict[str, Any]] = {}
        for item_id in ids:
            items = []
            for r in records:
                src = r.get(field) or {}
                if isinstance(src, dict):
                    item = src.get(str(item_id), src.get(item_id, {}))
                elif isinstance(src, list):
                    item = next((x for x in src if isinstance(x, dict) and str(x.get(id_key)) == str(item_id)), {})
                else:
                    item = {}
                if isinstance(item, dict):
                    items.append(item)
            metric_keys = set()
            for item in items:
                metric_keys.update(item.keys())
            out: Dict[str, Any] = {}
            for mk in sorted(metric_keys):
                vals = [item.get(mk, None) for item in items]
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if len(numeric_vals) == len(vals) and len(vals) > 0:
                    fvals = [float(v) for v in vals]
                    out[mk] = float(np.mean(fvals))
                    out[f"{mk}_std"] = float(np.std(fvals, ddof=0))
                else:
                    out[mk] = vals[-1]
            out_all[str(item_id)] = out
        return out_all

    if any("per_rank_breakdown" in r for r in records):
        by_rank = _aggregate_indexed_dict_list("per_rank_breakdown", "rank")
        summary["per_rank_breakdown_by_rank"] = by_rank
        summary["per_rank_breakdown"] = [by_rank[k] for k in sorted(by_rank.keys(), key=lambda x: int(float(x)))]
    if any("per_node_breakdown" in r for r in records):
        summary["per_node_breakdown"] = _aggregate_indexed_dict_list("per_node_breakdown", "node")

    def _repeat_slope(key: str) -> float:
        vals = []
        for r in records:
            v = r.get(key, None)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals), dtype=np.float64)
        y = np.asarray(vals, dtype=np.float64)
        x = x - x.mean()
        denom = float((x * x).sum())
        if denom <= 0.0:
            return 0.0
        return float(((x * (y - y.mean())).sum()) / denom)

    summary["num_repeats"] = len(records)
    summary["repeat_aggregate"] = "mean_std"
    summary["repeat_timestamps"] = [r.get("timestamp", "") for r in records]
    summary["repeat_indices"] = [int(r.get("repeat_index", i)) for i, r in enumerate(records)]
    summary["repeat_decode_tok_s_slope"] = _repeat_slope("decode_throughput_tok_s")
    summary["repeat_throughput_tok_s_slope"] = _repeat_slope("throughput_tok_s")
    summary["repeat_e2e_output_tok_s_slope"] = _repeat_slope("e2e_output_tok_s")
    summary["repeat_e2e_total_tok_s_slope"] = _repeat_slope("e2e_total_tok_s")
    summary["repeat_cache_hit_rate_slope"] = _repeat_slope("cache_hit_rate")
    summary["repeat_h2d_demand_mb_slope"] = _repeat_slope("expert_h2d_demand_mb")
    summary["repeat_prefetch_issued_slope"] = _repeat_slope("prefetch_issued_count")
    summary["repeat_prefetch_rejected_margin_slope"] = _repeat_slope("prefetch_rejected_margin")
    summary["repeat_drift_warning"] = int(
        summary["repeat_decode_tok_s_slope"] < -0.05
        and summary["repeat_h2d_demand_mb_slope"] > 0.0
    )
    summary["raw_repeats"] = records
    summary["lightweight_stats_by_repeat"] = [r.get("lightweight_stats_by_rank", []) for r in records]
    return summary


def save_aggregated_perf(summary: Dict[str, Any], args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = args.run_tag or "default"

    summary_light = dict(summary)
    summary_light.pop("raw_repeats", None)

    # Main result file: one clean aggregated json after all repeats.
    out_json = os.path.join(args.out_dir, f"{tag}_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_light, f, indent=2, ensure_ascii=False)

    # Timestamped archive copy.
    archive_json = os.path.join(args.out_dir, f"run_{ts}_{tag}_summary.json")
    with open(archive_json, "w", encoding="utf-8") as f:
        json.dump(summary_light, f, indent=2, ensure_ascii=False)

    # Optional raw repeat details kept separately for debugging.
    raw_json = os.path.join(args.out_dir, f"run_{ts}_{tag}_raw_repeats.json")
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(summary.get("raw_repeats", []), f, indent=2, ensure_ascii=False)

    out_jsonl = os.path.join(args.out_dir, "runs.jsonl")
    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_light, ensure_ascii=False) + "\n")

    print("\n===== PERF RUN SUMMARY (rank0, aggregated over repeats) =====")
    keys = [
        "routing_mode", "planner_mode", "runtime_mode", "placement_mode", "expert_source",
        "prefetch_top_m", "expert_cache", "expert_cache_by_rank", "num_repeats",
        "rs_prefetch_prefill",
        "owner_group_size", "skvs_migration_impl", "skvs_owner_local", "skvs_disable_migration",
        "wall_total_ms", "wall_total_ms_std",
        "wall_ttft_ms", "wall_ttft_ms_std",
        "wall_prefill_ms", "wall_prefill_ms_std",
        "wall_decode_ms", "wall_decode_ms_std",
        "decode_step_avg_ms", "decode_step_avg_ms_std",
        "decode_step_p50_ms", "decode_step_p50_ms_std",
        "decode_step_p95_ms", "decode_step_p95_ms_std",
        "decode_step_p99_ms", "decode_step_p99_ms_std",
        "e2e_output_tok_s", "e2e_output_tok_s_std",
        "e2e_total_tok_s", "e2e_total_tok_s_std",
        "throughput_tok_s", "throughput_tok_s_std",
        "decode_throughput_tok_s", "decode_throughput_tok_s_std",
        "comm_mb_activation", "comm_mb_activation_remote", "comm_mb_kv", "skvs_accounted_mb",
        "sim_weaknet_enabled", "sim_node_map", "sim_inter_node_bw_mb_s", "sim_inter_node_lat_ms",
        "comm_mb_activation_intra_node", "comm_mb_activation_inter_node",
        "weaknet_inter_node_item_count", "weaknet_inter_node_transfer_count",
        "weaknet_inter_node_packet_avg_items", "weaknet_inter_node_packet_max_items",
        "weaknet_inter_node_packet_le1_count", "weaknet_inter_node_packet_le2_count",
        "weaknet_inter_node_packet_le4_count", "weaknet_inter_node_packet_le8_count",
        "weaknet_inter_node_packet_gt8_count", "weaknet_inter_node_small_packet_threshold",
        "weaknet_inter_node_small_packet_count", "weaknet_inter_node_small_packet_ratio",
        "weaknet_inter_node_sim_ms", "weaknet_inter_node_sim_ms_rankmax",
        "rank_grouped_dispatch", "rank_grouped_prefill",
        "rank_grouped_pair_entries", "rank_grouped_packet_entries", "rank_grouped_dispatch_calls",
        "rank_grouped_packet_ratio", "rank_grouped_theoretical_hidden_saving_ratio",
        "node_grouped_dispatch", "node_grouped_prefill",
        "node_grouped_adaptive", "node_grouped_min_saving_ms", "node_grouped_overhead_margin",
        "node_grouped_est_overhead_ms", "node_grouped_layer_policy_path", "node_grouped_layer_policy_default",
        "node_grouped_fastpath_mode",
        "node_grouped_fastpath_prefill_mode", "node_grouped_fastpath_decode_mode",
        "node_grouped_triton_pack", "node_grouped_triton_pack_prefill",
        "node_grouped_triton_pack_decode", "node_grouped_triton_strict",
        "node_grouped_same_node_fastpath", "node_grouped_same_node_fastpath_prefill",
        "node_grouped_same_node_fastpath_decode", "node_grouped_stage2_local_rank_fastpath",
        "node_grouped_stage2_local_rank_fastpath_prefill", "node_grouped_stage2_local_rank_fastpath_decode",
        "node_grouped_pack_meta", "node_grouped_pack_meta_prefill", "node_grouped_pack_meta_decode",
        "node_grouped_pack_meta_float32", "node_grouped_pack_meta_float32_prefill",
        "node_grouped_pack_meta_float32_decode", "node_grouped_identity_combine_fastpath",
        "node_grouped_identity_combine_fastpath_prefill", "node_grouped_identity_combine_fastpath_decode",
        "node_grouped_skip_dummy_a2a",
        "node_grouped_pair_entries", "node_grouped_node_packet_entries", "node_grouped_rank_packet_entries",
        "node_grouped_dispatch_calls", "node_grouped_node_packet_ratio", "node_grouped_rank_packet_ratio",
        "node_grouped_theoretical_node_hidden_saving_ratio",
        "node_grouped_pack_ms", "node_grouped_metadata_ms", "node_grouped_intra_fanout_ms",
        "node_grouped_inter_send_ms", "node_grouped_inter_send_weaknet_sleep_ms", "node_grouped_inter_send_stage1_weaknet_sleep_ms", "node_grouped_inter_send_stage2_weaknet_sleep_ms", "node_grouped_inter_send_stage1_non_sleep_ms", "node_grouped_intra_fanout_stage2_non_sleep_ms", "node_grouped_total_overhead_non_sleep_ms", "node_grouped_inter_send_stage1_collective_ms", "node_grouped_inter_send_stage2_collective_ms", "node_grouped_metadata_exchange_ms", "node_grouped_inter_send_non_sleep_ms", "node_grouped_gather_ms", "node_grouped_total_overhead_ms",
        "node_grouped_same_node_packet_entries", "node_grouped_cross_node_packet_entries",
        "node_grouped_same_node_packet_fraction", "node_grouped_same_node_bypass_mb",
        "node_grouped_stage2_local_rank_packet_entries", "node_grouped_stage2_remote_rank_packet_entries",
        "node_grouped_stage2_local_rank_packet_fraction", "node_grouped_stage2_local_bypass_mb",
        "node_grouped_metadata_collective_count", "node_grouped_metadata_mb",
        "node_grouped_counts_build_ms", "node_grouped_splits_build_ms", "node_grouped_reverse_map_build_ms",
        "node_grouped_rank_to_node_gather_ms", "node_grouped_node_to_token_restore_ms",
        "node_grouped_final_combine_ms", "node_grouped_combine_intermediate_buffer_mb",
        "node_grouped_allocation_ms", "node_grouped_allocation_count", "node_grouped_allocation_mb",
        "node_grouped_workspace_reuse_hit",
        "node_grouped_enabled_layers", "node_grouped_fallback_layers", "node_grouped_estimated_saving_ms",
        "ers_enabled", "ers_mode", "ers_phase", "ers_topm", "ers_margin", "ers_max_replace", "ers_cache_aware", "ers_legacy_nodepatch_alias_used", "ers_runtime_semantics",
        "nodepatch_enabled", "nodepatch_phase", "nodepatch_topm", "nodepatch_margin", "nodepatch_max_replace",
        "nodepatch_fastpath", "nodepatch_disable_margin",
        "nodepatch_lean_repair", "nodepatch_lean_candidate_scope", "nodepatch_lean_resident_nearest",
        "nodepatch_repaired_tokens", "nodepatch_repairable_cross_node_fraction",
        "nodepatch_transfer_event_reduction_pct", "nodepatch_cross_node_item_reduction_pct",
        "nodepatch_avg_unique_nodes_before", "nodepatch_avg_unique_nodes_after",
        "nodepatch_avg_utility_loss_logit", "nodepatch_avg_utility_loss_prob",
        "nodepatch_event_aware", "nodepatch_event_aggressive", "nodepatch_event_only_empty_packet",
        "nodepatch_event_require_local_candidate", "nodepatch_event_packet_max_items",
        "nodepatch_packet_emptying_first", "nodepatch_cache_aware",
        "nodepatch_cache_require_owner_local", "nodepatch_require_added_resident",
        "nodepatch_runtime_packet_events_before", "nodepatch_runtime_packet_events_after",
        "nodepatch_runtime_packet_events_eliminated", "nodepatch_event_aware_skipped_repairs",
        "nodepatch_cache_aware_skipped_repairs", "nodepatch_cache_aware_filtered_candidates",
        "nodepatch_packet_emptying_candidate_packets", "nodepatch_packet_emptying_feasible_packets",
        "nodepatch_packet_emptying_rejected_packets",
        "nodepatch_route_cost_before", "nodepatch_route_cost_after", "nodepatch_route_cost_gain",
        "nodepatch_route_cost_reduction_pct", "nodepatch_avg_route_cost_gain_per_replacement",
        "nodepatch_avg_replacement_rank_gap", "nodepatch_rank_gap_max",
        "nodepatch_avg_old_prob", "nodepatch_avg_new_prob", "nodepatch_avg_prob_delta",
        "nodepatch_candidate_not_prepared_reject_count", "nodepatch_candidate_no_comm_gain_reject_count",
        "nodepatch_candidate_scope_reject_count", "nodepatch_candidate_duplicate_reject_count",
        "nodepatch_candidate_rank_gap_reject_count", "nodepatch_candidate_cache_reject_count",
        "nodepatch_candidate_total_reject_count", "nodepatch_no_replace_position_count",
        "prefill_scp_temporary_load_count", "prefill_scp_temporary_hit_count",
        "prefill_scp_temporary_promote_count", "prefill_scp_temporary_evict_count",
        "prefill_scp_temporary_resident_at_decode_start", "prefill_scp_temporary_demote_count",
        "prefill_scp_temporary_resident_at_decode_end", "prefill_scp_temporary_decode_reuse_rate",
        "resolver_decode_total_queries", "resolver_decode_rr_owner_count",
        "resolver_decode_scp_replica_selected_count", "resolver_decode_rr_fallback_count",
        "resolver_decode_replica_reject_not_ready_count", "resolver_decode_replica_reject_no_comm_gain_count",
        "resolver_decode_replica_cache_ready_count", "resolver_decode_replica_prefetch_ready_count",
        "resolver_decode_route_cost_before", "resolver_decode_route_cost_after", "resolver_decode_route_cost_gain",
        "scp_phase_route_diag_enabled",
        "scp_prefill_route_diag_items", "scp_prefill_route_diag_replica_items",
        "scp_prefill_route_diag_replica_item_rate",
        "scp_prefill_route_diag_cross_items_before", "scp_prefill_route_diag_cross_items_after",
        "scp_prefill_route_diag_cross_item_reduction_pct",
        "scp_prefill_route_diag_cross_events_before", "scp_prefill_route_diag_cross_events_after",
        "scp_prefill_route_diag_cross_event_reduction_pct",
        "scp_prefill_route_diag_cost_before", "scp_prefill_route_diag_cost_after",
        "scp_prefill_route_diag_cost_reduction_pct",
        "scp_decode_route_diag_items", "scp_decode_route_diag_replica_items",
        "scp_decode_route_diag_replica_item_rate",
        "scp_decode_route_diag_cross_items_before", "scp_decode_route_diag_cross_items_after",
        "scp_decode_route_diag_cross_item_reduction_pct",
        "scp_decode_route_diag_cross_events_before", "scp_decode_route_diag_cross_events_after",
        "scp_decode_route_diag_cross_event_reduction_pct",
        "scp_decode_route_diag_cost_before", "scp_decode_route_diag_cost_after",
        "scp_decode_route_diag_cost_reduction_pct",
        "rs_prefetch_target_resolver_match_count", "rs_prefetch_target_resolver_mismatch_count",
        "rs_prefetch_would_be_wrong_without_resolver_count",
        "scp_table_local_replica_route_count", "scp_table_rr_fallback_count",
        "scp_table_cross_node_route_count", "scp_table_same_node_route_count",
        "scp_route_cost_before_sum", "scp_route_cost_after_sum",
        "scp_lookup_calls", "scp_lookup_overhead_ms",
        "scp_route_cost_reduction_pct", "valid_comparison",
        "rs_scp_target_mismatch_count",
        "expert_h2d_mb", "expert_h2d_prefetch_mb", "expert_h2d_demand_mb",
        "epipe_enable", "epipe_shadow", "epipe_mode", "epipe_decode_rs_mode", "epipe_decode_with_rs_mode",
        "epipe_shadow_count", "epipe_enabled_count",
        "epipe_fallback_count", "epipe_fallback_no_ready", "epipe_fallback_no_missing",
        "epipe_fallback_no_miss_trigger", "epipe_fallback_no_ready_lookahead",
        "epipe_fallback_low_ready_items",
        "epipe_fallback_prefill_disabled", "epipe_fallback_no_memory_headroom",
        "epipe_fallback_memory", "epipe_fallback_cache_full_no_temp", "epipe_fallback_exception",
        "epipe_decode_rs_miss_trigger_count", "epipe_decode_rs_skipped_no_miss_count",
        "epipe_prefill_miss_trigger_count", "epipe_prefill_shadow_opportunity_count",
        "epipe_rs_buffered_calls", "epipe_rs_buffer_flush_count",
        "epipe_rs_buffered_ready_group_count", "epipe_rs_buffered_ready_item_count",
        "epipe_rs_residual_scan_calls", "epipe_rs_residual_scan_ready_group_count",
        "epipe_rs_residual_scan_ready_item_count", "epipe_rs_residual_scan_residual_count",
        "epipe_rs_residual_scan_late_count", "epipe_rs_residual_scan_demand_count",
        "epipe_rs_residual_scan_no_ready_count",
        "epipe_rs_trigger_count", "epipe_rs_late_prefetch_trigger_count",
        "epipe_rs_demand_miss_trigger_count", "epipe_rs_late_prefetch_overlap_count",
        "epipe_rs_demand_stage_count", "epipe_rs_demand_stage_disabled_by_gate_count",
        "epipe_rs_demand_stage_force_count", "epipe_prefill_enabled_count",
        "epipe_decode_no_rs_enabled_count", "epipe_decode_rs_late_overlap_count",
        "epipe_decode_rs_demand_stage_count", "epipe_rs_gate_skip_count",
        "epipe_rs_gate_reenable_count", "epipe_rs_buffer_too_small_count",
        "epipe_lookahead_ready_group_count", "epipe_lookahead_ready_item_count",
        "epipe_staged_expert_count", "epipe_staged_h2d_mb",
        "epipe_deferred_lru_commit_count", "epipe_prefill_transient_commit_count",
        "epipe_prefill_transient_demote_count", "epipe_prefill_transient_promote_count",
        "epipe_done_group_skip_count",
        "epipe_ready_expert_count", "epipe_missing_expert_count",
        "epipe_ready_item_count", "epipe_missing_item_count",
        "epipe_cache_free_slot_count", "epipe_cache_full_staged_count",
        "epipe_transient_payload_count", "epipe_cache_insert_count", "epipe_no_evict_count",
        "epipe_missing_h2d_est_mb", "epipe_demand_h2d_mb", "epipe_staged_missing_count",
        "epipe_demand_h2d_ms", "epipe_wait_ms", "epipe_hidden_h2d_ms_est",
        "sampled_demand_h2d_exposed_decode_ms", "sampled_prefetch_wait_ms",
        "sampled_epipe_hidden_h2d_ms", "sampled_epipe_wait_ms",
        "sampled_epipe_control_overhead_ms",
        "total_demand_h2d_mb_including_epipe",
        "oracle_stage_h2d_mb", "oracle_stage_event_count", "oracle_stage_ms", "oracle_stage_decode_ms",
        "expert_timing_enabled", "moe_router_ms", "moe_expert_ms", "moe_shared_ms",
        "moe_router_avg_ms", "moe_expert_avg_ms", "moe_shared_avg_ms",
        "moe_router_ms_rankmax", "moe_expert_ms_rankmax", "moe_shared_ms_rankmax",
        "moe_router_avg_ms_rankmax", "moe_expert_avg_ms_rankmax", "moe_shared_avg_ms_rankmax",
        "rs_mlp_ms", "rs_mlp_ms_rankmax", "rs_mlp_calls", "rs_next_layer_pred_calls",
        "rs_mlp_avg_ms", "rs_mlp_avg_ms_rankmax",
        "demand_h2d_exposed_ms", "demand_h2d_exposed_decode_ms",
        "demand_h2d_exposed_ms_rankmax", "demand_h2d_exposed_decode_ms_rankmax", "demand_h2d_event_count",
        "sim_demand_h2d_exposed_ms", "sim_demand_h2d_exposed_decode_ms",
        "sim_demand_h2d_exposed_ms_rankmax", "sim_demand_h2d_exposed_decode_ms_rankmax",
        "sim_h2d_extra_sleep_ms", "sim_expert_exec_ms", "sim_expert_exec_ms_rankmax", "sim_compute_extra_sleep_ms",
        "prefetch_h2d_copy_ms", "prefetch_h2d_event_count", "prefetch_wait_ms", "prefetch_wait_event_count",
        "plain_ep_executor_calls", "epipe_executor_calls", "epipe_executor_attempt_count",
        "cache_hit_rate", "cache_hit_rate_std",
        "resident_hit_rate", "resident_hit_rate_std",
        "effective_service_hit_rate", "effective_service_hit_rate_std",
        "main_cache_hit_rate", "main_cache_hit_rate_std",
        "main_cache_effective_service_hit_rate", "main_cache_effective_service_hit_rate_std",
        "extra_cache_service_hit_rate", "extra_cache_service_hit_rate_std",
        "main_cache_hit_count", "main_cache_miss_count", "extra_cache_service_hit_count",
        "prefetch_use_rate", "prefetch_use_rate_std",
        "rs_pred_candidate_count", "rs_pred_local_candidate_count", "rs_pred_nonlocal_candidate_count",
        "rs_pred_local_candidate_fraction", "rs_candidate_filter_in_count", "rs_candidate_filter_out_count",
        "rs_candidate_filter_out_cached_count", "rs_candidate_filter_out_low_score_count", "rs_candidate_survived_count",
        "rs_candidate_survival_rate", "prefetch_issued_count", "prefetch_completed_count",
        "prefetch_complete_before_use_count", "prefetch_completion_rate", "prefetch_complete_before_use_rate",
        "prefetch_rejected_active_replace", "prefetch_rejected_margin", "prefetch_rejected_cache_full",
        "prefetch_rejected_low_score", "prefetch_admission_rate",
        "prefetch_admission_upper_recall", "prefetch_admission_upper_recall_std",
        "prefetch_recall_gap_to_admission_upper", "prefetch_recall_gap_to_admission_upper_std",
        "ttl_prevented_evictions", "cache_eviction_fallback_all_protected",
        "prefill_decode_warmup", "warmup_loaded_experts", "warmup_skipped_resident",
        "warmup_h2d_mb", "warmup_time_ms",
        "transition_prefetch", "transition_lookup_calls", "transition_lookup_overhead_ms",
        "transition_prefetch_admitted", "transition_prefetch_hit",
        "transition_safe_evict_denied", "transition_no_replace_denied",
        "epipe_enable", "epipe_shadow", "epipe_mode", "epipe_decode_with_rs_mode",
        "epipe_shadow_count", "epipe_enabled_count",
        "epipe_fallback_count", "epipe_ready_item_count", "epipe_missing_item_count",
        "epipe_cache_full_staged_count", "epipe_transient_payload_count",
        "epipe_cache_insert_count", "epipe_decode_rs_miss_trigger_count",
        "epipe_decode_rs_skipped_no_miss_count", "epipe_lookahead_ready_item_count",
        "epipe_staged_h2d_mb", "epipe_demand_h2d_mb", "epipe_wait_ms",
        "peak_allocated_mb", "peak_reserved_mb",
        "owner_group_count", "owner_group_reconstruct_avg_ms", "owner_group_plan_avg_ms", "owner_group_migrate_avg_ms",
        "owner_empty_ranks_avg", "owner_hist_cv_avg", "owner_imbalance_ratio_avg",
        "owner_attention_ms_rankmax", "owner_moe_ms_rankmax",
    ]
    for k in keys:
        if k in summary:
            print(f"{k}: {summary[k]}")
    if "skvs_debug" in summary:
        print("skvs_debug:", summary["skvs_debug"])
    print("===================================\n")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    import faulthandler
    _assert_runtime_alignment()
    faulthandler.enable()

    ap = argparse.ArgumentParser()

    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument(
        "--scp-path",
        type=str,
        default="",
    )
    ap.add_argument(
        "--rs-dir",
        type=str,
        default="",
    )

    ap.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog. " * 16)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--use-dataset-prompts", action="store_true", default=True)

    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    ap.add_argument("--routing-mode", type=str, default="router", choices=["rs", "router"])
    ap.add_argument("--rs-require-complete", dest="rs_require_complete", action="store_true")
    ap.add_argument("--no-rs-require-complete", dest="rs_require_complete", action="store_false")
    ap.set_defaults(rs_require_complete=True)
    ap.add_argument("--log-routing-stats", action="store_true")

    ap.add_argument("--enable-skvs", action="store_true")
    ap.add_argument("--skvs-sink", type=int, default=4)
    ap.add_argument("--skvs-window", type=int, default=16)
    ap.add_argument("--skvs-ratio", type=float, default=0.6)
    ap.add_argument("--skvs-min-kv-len", type=int, default=0)
    ap.add_argument("--skvs-min-sparse-drop-ratio", type=float, default=0.0)
    ap.add_argument("--skvs-guidance-every-k", type=int, default=24)
    ap.add_argument("--skvs-guidance-start-kv-len", type=int, default=0)
    ap.add_argument("--skvs-allow-prefill-sparse", action="store_true")

    ap.add_argument("--expert-offload", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--expert-cache", type=int, default=10)
    ap.add_argument("--cache-policy", type=str, default="lru", choices=["lru", "lfu", "tiered"])

    ap.add_argument("--debug-a2a-checks", action="store_true")
    ap.add_argument("--log-per-layer", action="store_true")
    ap.add_argument("--collect-per-layer", action="store_true")
    ap.add_argument("--measure-ttft", action="store_true")
    ap.add_argument("--save-output-tokens", action="store_true",
                    help="Save per-rank input/generated tokens after timing for correctness auditing.")

    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "eval_perf"))
    ap.add_argument("--run-tag", type=str, default="")

    ap.add_argument("--sdpa-flash", action="store_true")
    ap.add_argument("--sdpa-mem-efficient", action="store_true")
    ap.add_argument("--sdpa-math", action="store_true")

    ap.add_argument("--planner-mode", type=str, default="none", choices=["none", "rs_prefetch"])
    ap.add_argument("--expert-source", type=str, default="router", choices=["router", "rs"])
    ap.add_argument("--prefetch-top-m", type=int, default=4)
    ap.add_argument("--rs-group", type=int, default=3)
    ap.add_argument("--rs-debug", action="store_true")
    ap.add_argument("--oracle-mode", type=str, default="none", choices=["none", "record", "replay"])
    ap.add_argument("--oracle-trace-dir", type=str, default="")
    ap.add_argument("--oracle-ignore-cache-capacity", action="store_true")
    ap.add_argument("--lm-head-device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-scan-samples", type=int, default=2000)
    ap.add_argument("--dataset-skip-valid", type=int, default=int(os.environ.get("DATASET_SKIP_VALID", "0") or 0))
    ap.add_argument("--placement-mode", type=str, default=os.environ.get("SPARSE_PLACEMENT", "round_robin"), choices=["scp", "scp_ep", "round_robin", "weighted_random_balanced"])
    ap.add_argument("--scp-ep-placement", type=str, default=os.environ.get("SPARSE_SCP_EP_PLACEMENT", ""))
    ap.add_argument(
        "--runtime-mode",
        type=str,
        default="baseline",
        choices=["baseline"],
    )
    ap.add_argument("--ep-size", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SPARSE_SKVS_VERBOSE", "0")
    os.environ.setdefault("SPARSE_RUNTIME_DEBUG", "0")
    os.environ.setdefault("SPARSE_GROUPSTART_LOG", "0")
    os.environ.setdefault("SPARSE_GROUPSTART_LOG_ALL_RANKS", "0")
    os.environ.setdefault("SPARSE_PHASE_TRACE", "0")

    if not (args.sdpa_flash or args.sdpa_mem_efficient or args.sdpa_math):
        set_sdpa_mode(flash=False, mem_efficient=False, math_sdp=True)
    else:
        set_sdpa_mode(
            flash=args.sdpa_flash,
            mem_efficient=args.sdpa_mem_efficient,
            math_sdp=args.sdpa_math,
        )

    env = init_dist()

    if int(args.ep_size) <= 0:
        args.ep_size = env.world
    if env.world % int(args.ep_size) != 0:
        raise RuntimeError(f"world ({env.world}) must be divisible by ep_size ({args.ep_size})")
    if args.batch < (env.world // int(args.ep_size)):
        raise RuntimeError(
            f"global batch ({args.batch}) must be >= dp_size ({env.world // int(args.ep_size)}) for baseline"
        )
    args.runtime_mode = "baseline"

    metrics = Metrics()
    metrics.decode_step_stats = RunningStats()
    metrics.per_layer = {} if args.collect_per_layer else None

    cfg = CoreBuildConfig(
        model_path=args.model_path,
        scp_path=args.scp_path,
        rs_dir=args.rs_dir,
        dtype=args.dtype,
        expert_offload=args.expert_offload,
        expert_cache=args.expert_cache,
        cache_policy=str(args.cache_policy),
        enable_skvs=args.enable_skvs,
        skvs_sink=args.skvs_sink,
        skvs_window=args.skvs_window,
        skvs_ratio=args.skvs_ratio,
        skvs_min_kv_len=int(args.skvs_min_kv_len),
        skvs_min_sparse_drop_ratio=float(args.skvs_min_sparse_drop_ratio),
        skvs_guidance_every_k=int(args.skvs_guidance_every_k),
        skvs_guidance_start_kv_len=int(args.skvs_guidance_start_kv_len),
        skvs_allow_prefill_sparse=bool(args.skvs_allow_prefill_sparse),
        prompt_len=int(args.prompt_len),
        max_new_tokens=int(args.max_new),
        routing_mode=args.routing_mode,
        expert_source=str(args.expert_source),
        rs_require_complete=bool(args.rs_require_complete),
        rs_prob_to_logit_eps=1e-9,
        debug_a2a_checks=bool(args.debug_a2a_checks),
        collect_per_layer=bool(args.collect_per_layer),
        log_per_layer=bool(args.log_per_layer),
        log_routing_stats=bool(args.log_routing_stats),
        rs_group=int(args.rs_group),
        rs_debug=bool(args.rs_debug),
        planner_mode=str(args.planner_mode),
        prefetch_top_m=int(args.prefetch_top_m),
        oracle_mode=str(args.oracle_mode),
        oracle_trace_dir=str(args.oracle_trace_dir),
        oracle_ignore_cache_capacity=bool(args.oracle_ignore_cache_capacity),
        lm_head_device=str(args.lm_head_device),
        placement_mode=args.placement_mode,
        scp_ep_path=str(args.scp_ep_placement),
        runtime_mode=str(args.runtime_mode),
        ep_size=int(args.ep_size),
    )

    bundle = build_model_and_tokenizer(cfg, env, metrics=metrics)
    if hasattr(bundle, "runtime_mode") and str(bundle.runtime_mode) != str(args.runtime_mode):
        raise RuntimeError(
            f"Runtime mode mismatch: args.runtime_mode={args.runtime_mode}, bundle.runtime_mode={bundle.runtime_mode}. "
            f"This usually means stale eval_core.py / perf.py mismatch."
        )
    barrier(env)

    task_manifest_path = _task_manifest_path_from_env()
    task_local_metadata: Dict[str, Any] = {}
    use_dataset_prompts = True if args.dataset else bool(args.use_dataset_prompts)

    if task_manifest_path:
        prompts, task_local_metadata = _sample_task_local_prompts(
            manifest_path=Path(task_manifest_path).expanduser(),
            tokenizer=bundle.tokenizer,
            rank=int(env.rank),
            world=int(env.world),
            batch=int(args.batch),
            prompt_len=int(args.prompt_len),
            max_scan_samples=int(args.max_scan_samples),
            split=_task_local_split_from_env("test"),
            seed=int(args.seed),
            request_base=int(os.environ.get("SPARSE_TASK_REQUEST_BASE", os.environ.get("TASK_REQUEST_BASE", "0")) or 0),
        )
        task_local_metadata["profile_batch"] = int(args.batch)
        task_local_metadata["profile_prompt_len"] = int(args.prompt_len)
        task_local_metadata["profile_max_new"] = int(args.max_new)
        SparseContext.set_task_local_metadata(task_local_metadata)
    elif use_dataset_prompts:
        dataset_skip_valid = _rank_specific_dataset_skip(
            int(args.dataset_skip_valid),
            int(env.rank),
            int(env.world),
        )
        prompts, request_rows = _load_prompts_from_jsonl_gz(
            path=args.dataset,
            tokenizer=bundle.tokenizer,
            batch=args.batch,
            prompt_len=args.prompt_len,
            max_scan_samples=args.max_scan_samples,
            skip_valid_prompts=dataset_skip_valid,
            return_metadata=True,
        )
        source_node = 0
        try:
            raw_map = str(os.environ.get("SPARSE_LOGICAL_NODE_MAP", os.environ.get("SPARSE_SIM_NODE_MAP", "")) or "")
            vals = [int(x.strip()) for x in raw_map.split(",") if x.strip()]
            source_node = int(vals[int(env.rank)]) if int(env.rank) < len(vals) else int(env.rank)
        except Exception:
            source_node = int(env.rank)
        request_base = int(os.environ.get("SPARSE_PROFILE_ROUTING_REQUEST_ID", os.environ.get("DATASET_SKIP_VALID", "0")) or 0)
        request_meta = {
            "request_profile_enabled": True,
            "task_local_enabled": False,
            "profile_batch": int(args.batch),
            "profile_prompt_len": int(args.prompt_len),
            "profile_max_new": int(args.max_new),
            "profile_split": str(os.environ.get("SPARSE_PROFILE_SPLIT", "")),
            "source_rank": int(env.rank),
            "source_node": int(source_node),
            "rank_to_node": [int(x.strip()) for x in str(os.environ.get("SPARSE_LOGICAL_NODE_MAP", os.environ.get("SPARSE_SIM_NODE_MAP", "")) or "").split(",") if x.strip()],
            "requests": [
                {
                    "task_id": "ordinary",
                    "task_family": "ordinary",
                    "source_node": int(source_node),
                    "source_rank": int(env.rank),
                    "request_global_id": int(request_base) * 1_000_000_000 + int(env.rank) * 1_000_000 + int(i),
                    "split": str(os.environ.get("SPARSE_PROFILE_SPLIT", "")),
                    "dataset_path": str(row.get("dataset_path", args.dataset)),
                    "dataset_index": int(row.get("dataset_index", i)),
                }
                for i, row in enumerate(request_rows)
            ],
        }
        SparseContext.set_task_local_metadata(request_meta)
    else:
        prompts = [args.prompt] * args.batch
        SparseContext.set_task_local_metadata({})

    enc = bundle.tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(env.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(env.device)

    ep_size = int(args.ep_size) if int(args.ep_size) > 0 else env.world
    dp_size = env.world // ep_size
    dp_rank = env.rank // ep_size
    input_ids, attn_mask = _slice_local_dp_batch(
        input_ids,
        attn_mask,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )

    if attn_mask is not None:
        prompt_tokens_one = int(attn_mask.sum().item())
    else:
        prompt_tokens_one = int(input_ids.numel())

    # -----------------------------------------------------------------------------
    # warmup: 走和正式一致的 greedy_decode_profiled 路径，但不用正式 metrics 记账
    # -----------------------------------------------------------------------------
    for _ in range(args.warmup):
        _clear_all_runtime_state_for_perf()
        SparseContext.reset_nodepatch_stats()

        warm_metrics = Metrics()
        warm_metrics.decode_step_stats = RunningStats()
        warm_metrics.per_layer = None
        warm_metrics.prompt_tokens = prompt_tokens_one

        warm_out_ids = greedy_decode_profiled(
            model=bundle.model,
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=min(8, args.max_new),
            env=env,
            metrics=warm_metrics,
            measure_ttft=False,
        )
        del warm_out_ids
        del warm_metrics
        gc.collect()
        torch.cuda.empty_cache()

        _clear_all_runtime_state_for_perf()
        SparseContext.reset_nodepatch_stats()
        barrier(env)

    # -----------------------------------------------------------------------------
    # formal repeats
    # -----------------------------------------------------------------------------
    SparseContext.reset_routing_profile()
    repeat_records: List[Dict[str, Any]] = []

    for rep_idx in range(args.repeat):
        _clear_all_runtime_state_for_perf()
        SparseContext.reset_nodepatch_stats()
        metrics.reset_for_repeat()
        metrics.prompt_tokens = prompt_tokens_one

        barrier(env)

        torch.cuda.synchronize(env.device)
        with wall_timer_s() as wt_total:
            out_ids = greedy_decode_profiled(
                model=bundle.model,
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new,
                env=env,
                metrics=metrics,
                measure_ttft=args.measure_ttft,
            )
            torch.cuda.synchronize(env.device)

        metrics.wall_total_ms = wt_total() * 1000.0
        metrics.gen_tokens = int((out_ids.shape[1] - input_ids.shape[1]) * out_ids.shape[0])
        if args.save_output_tokens:
            token_dir = Path(args.out_dir) / "output_tokens"
            token_dir.mkdir(parents=True, exist_ok=True)
            (token_dir / f"rank{env.rank}.repeat{rep_idx}.json").write_text(
                json.dumps({"rank": env.rank, "repeat": rep_idx,
                            "prompt_len": int(input_ids.shape[1]),
                            "input_ids": input_ids.detach().cpu().tolist(),
                            "output_ids": out_ids.detach().cpu().tolist()}), encoding="utf-8")
        del out_ids
        gc.collect()
        torch.cuda.empty_cache()

        # eval_core.greedy_decode_profiled() already finalizes last-group prefetch accounting.
        # Only clear transient state here to avoid double-counting wasted prefetches.
        _finalize_all_prefetch_wasted()

        barrier(env)
        rec = reduce_and_build_perf_record(metrics, args, env, num_experts=bundle.num_experts, repeat_idx=rep_idx)
        if env.rank == 0 and rec is not None:
            repeat_records.append(rec)
            print(
                f"[repeat {rep_idx + 1}/{args.repeat}] "
                f"throughput_tok_s={rec.get('throughput_tok_s', 0.0):.4f}, "
                f"decode_throughput_tok_s={rec.get('decode_throughput_tok_s', 0.0):.4f}, "
                f"wall_total_ms={rec.get('wall_total_ms', 0.0):.2f}"
            )

    barrier(env)
    if env.rank == 0:
        summary = aggregate_repeat_records(repeat_records, args)
        save_aggregated_perf(summary, args)

    barrier(env)
    if args.log_routing_stats:
        print_routing_stats_if_enabled(env, bundle, tag=args.run_tag)

    barrier(env)
    profile_path = SparseContext.flush_routing_profile(env.rank)
    if profile_path:
        print(f"[SCP-EP profile] rank={env.rank} wrote {profile_path}", flush=True)
    barrier(env)
    destroy_dist()


if __name__ == "__main__":
    main()
