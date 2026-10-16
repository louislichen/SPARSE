"""Collect independent SCP prefill shards while loading model weights once.

This is a routing-data collector, not a timing benchmark. Defaults preserve the
batch, prompt, stride and same-input-per-rank behavior of profile_scp.sh.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--shard-stride", type=int, default=4)
    parser.add_argument("--rank-stride", type=int, default=0)
    parser.add_argument("--expert-cache", type=int, default=8)
    args = parser.parse_args()
    if min(args.shards, args.batch, args.prompt_len, args.shard_stride) <= 0 or args.rank_stride < 0:
        parser.error("invalid profiling dimensions")
    os.environ.update(SPARSE_PROFILE_ROUTING="1", SPARSE_PROFILE_STAGE="prefill",
        SPARSE_LIGHT_ENABLE="0", SPARSE_SCP_APPLY_PREFILL="0", SPARSE_SCP_APPLY_DECODE="0",
        SPARSE_EPIPE_ENABLE="0", SPARSE_ERS_ENABLE="0", SPARSE_NODEPATCH_ENABLE="0",
        SPARSE_SIM_ENABLE_WEAKNET="0", SPARSE_LOGICAL_WEAKNET_ENABLE="0",
        SPARSE_SIM_ENABLE_HETEROGENEITY="0", SPARSE_NODE_GROUPED_DISPATCH="0",
        SPARSE_RANK_GROUPED_DISPATCH="0", SPARSE_EP_FUSED_WEIGHT_DISPATCH="0",
        SPARSE_PREFILL_LAST_LOGITS_ONLY="1", SPARSE_PROFILE_SPLIT=args.split)
    import torch
    import torch.distributed as dist
    from utils.eval_core import (CoreBuildConfig, Metrics, SparseContext,
        build_model_and_tokenizer, greedy_decode_profiled, init_dist, destroy_dist)
    from scripts.perf import _clear_all_runtime_state_for_perf

    env = init_dist()
    try:
        cfg = CoreBuildConfig(model_path=args.model_path, scp_path="", rs_dir="",
            placement_mode="round_robin", planner_mode="none", ep_size=env.world,
            expert_cache=args.expert_cache, prompt_len=args.prompt_len, max_new_tokens=1)
        bundle = build_model_and_tokenizer(cfg, env, metrics=Metrics())
        opener = gzip.open if args.dataset.suffix == ".gz" else open
        prompts = []
        needed = (args.shards - 1) * args.shard_stride + (env.world - 1) * args.rank_stride + args.batch
        with opener(args.dataset, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                ids = row.get("token_ids")
                if ids is None:
                    ids = bundle.tokenizer.encode(row["text"], add_special_tokens=False)
                if len(ids) < args.prompt_len:
                    continue
                prompts.append((ids[:args.prompt_len], row.get("request_id", str(len(prompts)))))
                if len(prompts) >= needed:
                    break
        if len(prompts) < needed:
            raise ValueError(f"need {needed} eligible documents; found {len(prompts)}; no padding or duplicate fallback")
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = {"format": "scp_stream_profile_v1", "args": vars(args), "rank": env.rank,
            "world": env.world, "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "timings_are_serving_measurements": False, "cold_cache_each_shard": True,
            "rank_inputs": "identical" if args.rank_stride == 0 else "offset by rank_stride",
            "logical_node_map": os.environ.get("SPARSE_LOGICAL_NODE_MAP", ""),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in ("scripts/profile_scp_stream.py", "utils/eval_core.py")}, "requests": []}
        for shard in range(args.shards):
            start = shard * args.shard_stride + env.rank * args.rank_stride
            selected = prompts[start:start + args.batch]
            os.environ.update(SPARSE_PROFILE_ROUTING_SHARD_ID=str(shard),
                SPARSE_PROFILE_ROUTING_REQUEST_ID=str(start),
                SPARSE_PROFILE_ROUTING_OUT=str(args.out / f"shard_{shard}"))
            _clear_all_runtime_state_for_perf()
            SparseContext.reset_runtime_state_for_new_request(clear_expert_cache=True)
            SparseContext.reset_routing_profile()
            SparseContext.set_task_local_metadata({})
            ids = torch.tensor([entry[0] for entry in selected], dtype=torch.long, device=env.device)
            dist.barrier()
            with torch.inference_mode():
                greedy_decode_profiled(bundle.model, ids, torch.ones_like(ids), 1, env, Metrics(), False)
            torch.cuda.synchronize()
            SparseContext.flush_routing_profile(env.rank)
            manifest["requests"].append({"shard": shard, "request_ids": [entry[1] for entry in selected]})
            (args.out / f"manifest.rank{env.rank}.json").write_text(json.dumps(manifest, indent=2, default=str))
            if env.rank == 0:
                print(json.dumps({"status": "shard_complete", "split": args.split, "shard": shard,
                    "total_shards": args.shards}), flush=True)
            del ids
            dist.barrier()
    finally:
        destroy_dist()


if __name__ == "__main__":
    main()
