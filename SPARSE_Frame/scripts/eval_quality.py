# scripts/eval_quality.py
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.eval_core import (
    CoreBuildConfig,
    SparseContext,
    all_reduce_sum_,
    barrier,
    bcast_bool_from_rank0,
    build_model_and_tokenizer,
    destroy_dist,
    init_dist,
    print_routing_stats_if_enabled,
    set_sdpa_mode,
)


def _clear_all_prefetch_runtime_state() -> None:
    """
    质量评测不关心 prefetch wasted / cache 行为统计，但要避免残留的
    pending prefetch 状态污染后续 forward 的运行态。
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


def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@torch.no_grad()
def compute_nll_stats(
    model,
    input_ids: torch.Tensor,          # [B, T]
    attention_mask: torch.Tensor,     # [B, T]
    valid_sample_mask: torch.Tensor,  # [B] int64 {0,1}
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits  # [B, T, V]

    input_ids = input_ids.to(logits.device)
    attention_mask = attention_mask.to(logits.device)
    valid_sample_mask = valid_sample_mask.to(logits.device)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].to(dtype=torch.float32)

    logp = F.log_softmax(shift_logits, dim=-1)
    tgt = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    nll_pos = -(tgt * shift_mask)
    nll_sum_per_sample = nll_pos.sum(dim=1, dtype=torch.float64)
    tok_cnt_per_sample = shift_mask.sum(dim=1).to(dtype=torch.int64)

    valid = (valid_sample_mask.to(dtype=torch.int64) == 1) & (tok_cnt_per_sample > 0)

    # token-average
    nll_sum_tok = nll_sum_per_sample[valid].sum(dtype=torch.float64)
    tok_cnt = tok_cnt_per_sample[valid].sum(dtype=torch.int64)

    # sample-average
    mean_nll = torch.zeros((input_ids.shape[0],), device=input_ids.device, dtype=torch.float64)
    mean_nll[valid] = nll_sum_per_sample[valid] / tok_cnt_per_sample[valid].to(dtype=torch.float64)
    nll_sum_samp = mean_nll.sum(dtype=torch.float64)
    samp_cnt = valid.to(dtype=torch.int64).sum(dtype=torch.int64)

    return nll_sum_tok, tok_cnt, nll_sum_samp, samp_cnt


@torch.no_grad()
def eval_ppl_sync_streaming(
    model,
    tokenizer,
    env,
    dataset_path: str,
    batch: int,
    max_len: int,
    sample_size: int,
) -> Tuple[float, int, float, int]:
    world = env.world
    rank = env.rank
    device = env.device
    global_batch = world * batch

    it = stream_jsonl_gz(dataset_path) if rank == 0 else None
    seen = 0

    nll_sum_tok_local = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt_local = torch.zeros((), device=device, dtype=torch.int64)
    nll_sum_samp_local = torch.zeros((), device=device, dtype=torch.float64)
    samp_cnt_local = torch.zeros((), device=device, dtype=torch.int64)

    while True:
        has_more = bcast_bool_from_rank0(seen < sample_size if rank == 0 else False, env)
        if not has_more:
            break

        if rank == 0:
            batch_obj: List[Optional[str]] = []
            for _ in range(global_batch):
                if seen >= sample_size:
                    batch_obj.append(None)
                    continue
                ex = next(it, None)
                batch_obj.append(ex.get("text", "") if isinstance(ex, dict) else None)
                seen += 1
        else:
            batch_obj = [None] * global_batch

        obj_list = [batch_obj]
        dist.broadcast_object_list(obj_list, src=0)
        batch_obj = obj_list[0]

        s = rank * batch
        e = s + batch
        local_texts_raw = batch_obj[s:e]

        valid_sample_mask = torch.tensor(
            [0 if (t is None) else 1 for t in local_texts_raw],
            device=device,
            dtype=torch.int64,
        )
        local_texts = [("" if (t is None) else str(t)) for t in local_texts_raw]

        enc = tokenizer(
            local_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64, device=device)
        else:
            attention_mask = attention_mask.to(device)

        nll_tok, ntok, nll_samp, nsamp = compute_nll_stats(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            valid_sample_mask=valid_sample_mask,
        )

        nll_sum_tok_local += nll_tok
        tok_cnt_local += ntok
        nll_sum_samp_local += nll_samp
        samp_cnt_local += nsamp

        barrier(env)

    return (
        float(nll_sum_tok_local.item()),
        int(tok_cnt_local.item()),
        float(nll_sum_samp_local.item()),
        int(samp_cnt_local.item()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model-path",
        type=str,
        default="/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
    )
    ap.add_argument(
        "--scp-path",
        type=str,
        default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/scp_placement.json",
    )
    ap.add_argument(
        "--rs-dir",
        type=str,
        default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/mlp_trainer/Qwen1.5MoE_allData",
    )

    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--expert-offload", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--expert-cache", type=int, default=0)

    ap.add_argument("--routing-mode", type=str, default="router", choices=["router", "rs"])
    ap.add_argument("--rs-require-complete", action="store_true")
    ap.set_defaults(rs_require_complete=True)
    ap.add_argument("--log-routing-stats", action="store_true")

    ap.add_argument("--enable-skvs", action="store_true")
    ap.add_argument("--skvs-sink", type=int, default=4)
    ap.add_argument("--skvs-window", type=int, default=32)
    ap.add_argument("--skvs-ratio", type=float, default=0.6)

    ap.add_argument("--task", type=str, choices=["ppl"], required=True)
    ap.add_argument(
        "--dataset",
        type=str,
        default="/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz",
    )
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sample-size", type=int, default=128)

    ap.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "eval_quality"))
    ap.add_argument("--run-tag", type=str, default="")

    ap.add_argument("--sdpa-flash", action="store_true")
    ap.add_argument("--sdpa-mem-efficient", action="store_true")
    ap.add_argument("--sdpa-math", action="store_true")
    ap.add_argument("--rs-group", type=int, default=3)
    ap.add_argument("--rs-debug", action="store_true")

    # planner / fallback
    ap.add_argument("--planner-mode", type=str, default="none", choices=["none", "rs_prefetch"])
    ap.add_argument("--prefetch-top-m", type=int, default=1)
    ap.add_argument("--rs-fallback-top1-min", type=float, default=0.0)
    ap.add_argument("--rs-fallback-entropy-max", type=float, default=0.0)

    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if not (args.sdpa_flash or args.sdpa_mem_efficient or args.sdpa_math):
        set_sdpa_mode(flash=False, mem_efficient=False, math_sdp=True)
    else:
        set_sdpa_mode(
            flash=args.sdpa_flash,
            mem_efficient=args.sdpa_mem_efficient,
            math_sdp=args.sdpa_math,
        )

    env = init_dist()

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
        rs_prob_to_logit_eps=1e-9,
        log_routing_stats=bool(args.log_routing_stats),
        rs_group=args.rs_group,
        rs_debug=bool(args.rs_debug),
        planner_mode=args.planner_mode,
        prefetch_top_m=args.prefetch_top_m,
        rs_fallback_top1_min=args.rs_fallback_top1_min,
        rs_fallback_entropy_max=args.rs_fallback_entropy_max,
    )

    bundle = build_model_and_tokenizer(cfg, env, metrics=None)
    barrier(env)

    # 清理可能残留的 prefetch 运行态，避免污染本次质量评测
    SparseContext.reset()
    _clear_all_prefetch_runtime_state()

    t0 = time.time()

    nll_tok_local, tok_local, nll_samp_local, samp_local = eval_ppl_sync_streaming(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        env=env,
        dataset_path=args.dataset,
        batch=args.batch,
        max_len=args.max_len,
        sample_size=args.sample_size,
    )

    nll_tok = torch.tensor([nll_tok_local], device=env.device, dtype=torch.float64)
    tok_cnt = torch.tensor([tok_local], device=env.device, dtype=torch.int64)
    nll_samp = torch.tensor([nll_samp_local], device=env.device, dtype=torch.float64)
    samp_cnt = torch.tensor([samp_local], device=env.device, dtype=torch.int64)

    all_reduce_sum_(nll_tok)
    all_reduce_sum_(tok_cnt)
    all_reduce_sum_(nll_samp)
    all_reduce_sum_(samp_cnt)

    if env.rank == 0:
        tok_sum = int(tok_cnt.item())
        samp_sum = int(samp_cnt.item())

        nll_per_tok = float(nll_tok.item()) / max(tok_sum, 1)
        ppl_tok = math.exp(min(nll_per_tok, 50.0))

        nll_per_sample = float(nll_samp.item()) / max(samp_sum, 1)
        ppl_sample = math.exp(min(nll_per_sample, 50.0))
        wall_s = time.time() - t0

        print("\n===== QUALITY RUN SUMMARY (rank0) =====")
        print(f"routing_mode: {args.routing_mode}")
        print(f"planner_mode: {args.planner_mode}")
        print(f"prefetch_top_m: {args.prefetch_top_m}")
        print(f"task: {args.task}")
        print(f"dataset: {args.dataset}")
        print(f"sample_size: {args.sample_size}")
        print(f"tok_sum: {tok_sum}")
        print(f"num_valid_samples: {samp_sum}")
        print(f"nll_per_tok: {nll_per_tok}")
        print(f"ppl_tok: {ppl_tok}")
        print(f"nll_per_sample: {nll_per_sample}")
        print(f"ppl_sample: {ppl_sample}")
        print(f"wall_s: {wall_s}")
        print("======================================\n")

        result_dict = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_tag": args.run_tag,

            # experiment identity
            "routing_mode": args.routing_mode,
            "planner_mode": args.planner_mode,
            "prefetch_top_m": args.prefetch_top_m,
            "rs_dir": args.rs_dir,
            "rs_require_complete": bool(args.rs_require_complete),
            "rs_fallback_top1_min": args.rs_fallback_top1_min,
            "rs_fallback_entropy_max": args.rs_fallback_entropy_max,

            "task": args.task,
            "model_path": args.model_path,
            "dataset": args.dataset,
            "max_len": args.max_len,
            "batch_size": args.batch,
            "sample_size": args.sample_size,

            "tok_sum": tok_sum,
            "num_valid_samples": samp_sum,

            "nll_per_tok": round(nll_per_tok, 6),
            "ppl_tok": round(ppl_tok, 6),
            "nll_per_sample": round(nll_per_sample, 6),
            "ppl_sample": round(ppl_sample, 6),

            "wall_time_seconds": round(wall_s, 4),

            "expert_offload": args.expert_offload,
            "expert_cache": args.expert_cache,
            "dtype": args.dtype,

            "skvs_enabled": bool(args.enable_skvs),
            "skvs_sink": args.skvs_sink,
            "skvs_window": args.skvs_window,
            "skvs_ratio": args.skvs_ratio,
        }

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "quality_result.json"

        all_results: List[Dict[str, Any]] = []
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                    if isinstance(all_results, dict):
                        all_results = [all_results]
            except Exception as e:
                print(f"读取原有JSON文件失败，将创建新文件: {e}")
                all_results = []

        all_results.append(result_dict)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        print(f"\n结果已写入JSON文件：{json_path}")

    barrier(env)
    if args.log_routing_stats:
        print_routing_stats_if_enabled(env, bundle, tag=args.run_tag)

    barrier(env)
    destroy_dist()


if __name__ == "__main__":
    main()