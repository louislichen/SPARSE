# scripts/eval_perf.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist

import sys
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

# -----------------------------------------------------------------------------
def reduce_and_save_perf(metrics: Metrics, args, env: DistEnv, num_experts: int) -> None:
    device = env.device

    def t(x, dtype):
        return torch.tensor([x], device=device, dtype=dtype)

    comm = t(metrics.comm_bytes_activation, torch.long)
    h2d = t(metrics.expert_weight_h2d_bytes, torch.long)
    kvb = t(metrics.comm_bytes_kv, torch.long)
    skvs = t(metrics.skvs_accounted_bytes, torch.long)

    prefill_wall = t(metrics.wall_prefill_ms, torch.float64)
    decode_wall = t(metrics.wall_decode_ms, torch.float64)
    ttft_wall = t(metrics.wall_ttft_ms, torch.float64)
    total_wall = t(metrics.wall_total_ms, torch.float64)

    peak_alloc = t(metrics.peak_allocated_mb, torch.float64)
    peak_rsvd = t(metrics.peak_reserved_mb, torch.float64)

    all_reduce_sum_(comm)
    all_reduce_sum_(h2d)
    all_reduce_sum_(kvb)
    all_reduce_sum_(skvs)

    all_reduce_max_(prefill_wall)
    all_reduce_max_(decode_wall)
    all_reduce_max_(ttft_wall)
    all_reduce_max_(total_wall)

    all_reduce_max_(peak_alloc)
    all_reduce_max_(peak_rsvd)

    per_layer_records: Dict[int, Dict[str, float]] = {}
    if metrics.per_layer is not None and args.collect_per_layer:
        E = float(num_experts)
        for l, lp in metrics.per_layer.items():
            token_sum = torch.tensor([lp.expert_sum], device=device, dtype=torch.float64)
            sumsq = torch.tensor([lp.expert_sumsq], device=device, dtype=torch.float64)
            mx = torch.tensor([lp.expert_max], device=device, dtype=torch.float64)
            a2a_bytes = torch.tensor([lp.a2a_send_bytes], device=device, dtype=torch.int64)
            h2d_bytes = torch.tensor([lp.h2d_bytes], device=device, dtype=torch.int64)

            all_reduce_sum_(token_sum)
            all_reduce_sum_(sumsq)
            all_reduce_max_(mx)
            all_reduce_sum_(a2a_bytes)
            all_reduce_sum_(h2d_bytes)

            mean = float(token_sum.item()) / max(E, 1.0)
            ex2 = float(sumsq.item()) / max(E, 1.0)
            var = max(ex2 - mean * mean, 0.0)
            cv = (math.sqrt(var) / mean) if mean > 0 else 0.0

            per_layer_records[int(l)] = {
                "expert_mean": mean,
                "expert_max": float(mx.item()),
                "expert_cv": float(cv),
                "a2a_mb": float(a2a_bytes.item()) / 1024**2,
                "h2d_mb": float(h2d_bytes.item()) / 1024**2,
            }

    if env.rank != 0:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    step_summary = metrics.decode_step_stats.summary(prefix="decode_step_")

    record = {
        "run_tag": args.run_tag,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "world_size": env.world,

        "model_path": args.model_path,
        "batch": args.batch,
        "max_new": args.max_new,
        "dtype": args.dtype,

        "routing_mode": args.routing_mode,
        "rs_dir": args.rs_dir,
        "rs_require_complete": bool(args.rs_require_complete),

        "expert_offload": args.expert_offload,
        "expert_cache": args.expert_cache,

        "scp_path": args.scp_path,

        "skvs_enabled": bool(args.enable_skvs),
        "skvs_sink": args.skvs_sink,
        "skvs_window": args.skvs_window,
        "skvs_ratio": args.skvs_ratio,

        "prompt_tokens": metrics.prompt_tokens,
        "gen_tokens": metrics.gen_tokens,

        "wall_total_ms": float(total_wall.item()),
        "wall_prefill_ms": float(prefill_wall.item()),
        "wall_decode_ms": float(decode_wall.item()),
        "wall_ttft_ms": float(ttft_wall.item()),

        "throughput_tok_s": (metrics.prompt_tokens + metrics.gen_tokens) / max(float(total_wall.item()) / 1000.0, 1e-9),
        "decode_throughput_tok_s": metrics.gen_tokens / max(float(decode_wall.item()) / 1000.0, 1e-9),

        "comm_mb_activation": float(comm.item()) / 1024**2,
        "comm_mb_kv": float(kvb.item()) / 1024**2,
        "expert_h2d_mb": float(h2d.item()) / 1024**2,
        "skvs_accounted_mb": float(skvs.item()) / 1024**2,

        "peak_allocated_mb": float(peak_alloc.item()),
        "peak_reserved_mb": float(peak_rsvd.item()),

        **step_summary,
    }

    if args.collect_per_layer:
        record["per_layer"] = per_layer_records

    print("\n===== PERF RUN SUMMARY (rank0) =====")
    keys = [
        "routing_mode",
        "wall_total_ms", "wall_ttft_ms", "wall_prefill_ms", "wall_decode_ms",
        "decode_step_avg_ms", "decode_step_p95_ms", "decode_step_p99_ms",
        "throughput_tok_s", "decode_throughput_tok_s",
        "comm_mb_activation", "expert_h2d_mb",
        "peak_allocated_mb", "peak_reserved_mb",
    ]
    for k in keys:
        if k in record:
            print(f"{k}: {record[k]}")
    print("===================================\n")

    out_json = os.path.join(args.out_dir, f"run_{record['timestamp']}_{args.run_tag or 'default'}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    out_jsonl = os.path.join(args.out_dir, "runs.jsonl")
    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    import faulthandler
    faulthandler.enable()

    ap = argparse.ArgumentParser()

    ap.add_argument("--model-path", type=str, default="/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
    ap.add_argument("--scp-path", type=str, required=False,
                    default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/scp_placement.json")
    ap.add_argument("--rs-dir", type=str, required=False,
                    default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/mlp_trainer")

    ap.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    # routing
    ap.add_argument("--routing-mode", type=str, default="rs", choices=["rs", "router"])
    ap.add_argument("--rs-require-complete", action="store_true", help="default True; refuse to run with missing rs weights")
    ap.set_defaults(rs_require_complete=True)
    ap.add_argument("--log-routing-stats", action="store_true", help="print per-layer coverage/entropy after run (rank0)")

    # SKVS
    ap.add_argument("--enable-skvs", action="store_true")
    ap.add_argument("--skvs-sink", type=int, default=4)
    ap.add_argument("--skvs-window", type=int, default=32)
    ap.add_argument("--skvs-ratio", type=float, default=0.6)

    # Experts residency
    ap.add_argument("--expert-offload", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--expert-cache", type=int, default=0)

    # perf/debug toggles
    ap.add_argument("--debug-a2a-checks", action="store_true")
    ap.add_argument("--log-per-layer", action="store_true")
    ap.add_argument("--collect-per-layer", action="store_true")
    ap.add_argument("--measure-ttft", action="store_true")

    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "eval_perf"))
    ap.add_argument("--run-tag", type=str, default="")

    # SDPA mode
    ap.add_argument("--sdpa-flash", action="store_true")
    ap.add_argument("--sdpa-mem-efficient", action="store_true")
    ap.add_argument("--sdpa-math", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if not (args.sdpa_flash or args.sdpa_mem_efficient or args.sdpa_math):
        set_sdpa_mode(flash=False, mem_efficient=False, math_sdp=True)
    else:
        set_sdpa_mode(flash=args.sdpa_flash, mem_efficient=args.sdpa_mem_efficient, math_sdp=args.sdpa_math)

    env = init_dist()

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
        enable_skvs=args.enable_skvs,
        skvs_sink=args.skvs_sink,
        skvs_window=args.skvs_window,
        skvs_ratio=args.skvs_ratio,
        routing_mode=args.routing_mode,
        rs_require_complete=bool(args.rs_require_complete),
        debug_a2a_checks=args.debug_a2a_checks,
        collect_per_layer=args.collect_per_layer,
        log_per_layer=args.log_per_layer,
        log_routing_stats=bool(args.log_routing_stats),
    )

    bundle = build_model_and_tokenizer(cfg, env, metrics=metrics)

    barrier(env)

    prompts = [args.prompt] * args.batch
    enc = bundle.tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(env.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(env.device)

    metrics.prompt_tokens = int(input_ids.numel())

    # Warmup
    for _ in range(args.warmup):
        SparseContext.reset()
        _ = bundle.model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=min(8, args.max_new),
            use_cache=True,
            do_sample=False,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
        barrier(env)

    for _ in range(args.repeat):
        SparseContext.reset()
        metrics.reset_for_repeat()

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

        barrier(env)
        reduce_and_save_perf(metrics, args, env, num_experts=bundle.num_experts)

    # print routing stats AFTER all repeats (won't pollute timed region)
    barrier(env)
    if args.log_routing_stats:
        print_routing_stats_if_enabled(env, bundle, tag=args.run_tag)

    barrier(env)
    destroy_dist()


if __name__ == "__main__":
    main()