"""CUDA/NCCL correctness smoke; generated RS weights are untrained fixtures.

Prepare once, then launch ``run`` with torchrun. Optional --model-path reuses
an existing Qwen2-MoE checkpoint. This is not a predictor-quality benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from utils.eval_core import (
    CoreBuildConfig, DistributedSparseMoeBlock, Metrics, RoutingScoutMLP,
    SparseContext, build_model_and_tokenizer, destroy_dist, init_dist,
)


def prepare(args):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261014)
    if args.model_path:
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        if config.model_type != "qwen2_moe":
            raise ValueError("this smoke fixture currently supports Qwen2-MoE only")
        model_path = str(Path(args.model_path).resolve())
    else:
        config = Qwen2MoeConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            moe_intermediate_size=32, shared_expert_intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
            num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
            max_position_embeddings=512, pad_token_id=0, bos_token_id=1,
            eos_token_id=2, norm_topk_prob=True,
        )
        model_dir = directory / "model"
        if model_dir.exists():
            raise FileExistsError(f"refusing to overwrite fixture {model_dir}")
        Qwen2MoeForCausalLM(config).save_pretrained(model_dir)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t3"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="t0",
                                bos_token="t1", eos_token="t2", unk_token="t3").save_pretrained(model_dir)
        model_path = str(model_dir)
    rs_dir = directory / "rs"
    rs_dir.mkdir(exist_ok=True)
    for layer in range(1, config.num_hidden_layers):
        module = RoutingScoutMLP(config.hidden_size, config.num_experts, hidden_dim=32).eval()
        torch.save(dict(model_type="independent_light_logits_rs", input_dim=config.hidden_size,
                        num_experts=config.num_experts, hidden_dim=32, feature_mode="hidden",
                        source_layer=layer - 1, state_dict=module.state_dict(),
                        provenance="untrained functional smoke fixture"),
                   rs_dir / f"rs_l{layer}_inL{layer-1}.pth")
    world = args.world_size
    placement = dict(format="replicated_scp_ep_v2",
                     replicas={str(l): {str(e): list(range(world)) for e in range(config.num_experts)}
                               for l in range(config.num_hidden_layers)},
                     rank_to_node={str(r): 0 for r in range(world)},
                     replica_choice_cost_matrix=[[0 if s == d else 1 for d in range(world)] for s in range(world)])
    (directory / "placement.json").write_text(json.dumps(placement, indent=2))
    # Illustrative costs exercise all control paths; no performance conclusion
    # can be drawn without an independent calibration and trained RS weights.
    costs = dict(schema_version=1, world_size=world, calibrated=False,
                 rank_defaults=dict(h2d_bytes_per_ms=8_000_000, h2d_latency_ms=.04,
                                    expert_ms=.025, item_ms=.003, next_attention_ms=.1,
                                    split_overhead_ms=.01),
                 link_latency_ms=[[0 if s == d else .02 for d in range(world)] for s in range(world)],
                 link_bandwidth_bytes_per_ms=[[8_000_000] * world for _ in range(world)])
    (directory / "costs.json").write_text(json.dumps(costs, indent=2))
    (directory / "fixture.json").write_text(json.dumps(dict(model_path=model_path, world_size=world,
                                               pretrained=bool(args.model_path), seed=20261014), indent=2))
    print(f"Prepared functional fixture: {directory}")


def configure(directory, method):
    methods = {"independent", "joint", "joint-fixed-placement", "joint-no-epipe-feedback",
               "joint-reference", "joint-incremental", "joint-candidates", "joint-cache",
               "joint-combined", "joint-fast"}
    if method not in methods:
        raise ValueError(f"unknown method {method}")
    cached = method in ("joint-cache", "joint-combined", "joint-fast")
    combined = method in ("joint-combined", "joint-fast")
    fast = method == "joint-fast"
    settings = dict(
        SPARSE_SCP_APPLY_PREFILL="1", SPARSE_SCP_APPLY_DECODE="0",
        SPARSE_SCP_TRANSIENT_PREFILL_CACHE="1", SPARSE_PLAIN_EP_EXEC_MODE="sorted_segment",
        SPARSE_RS_PREFETCH_POLICY="cache_safe", SPARSE_RS_PREFETCH_PRED_TOPK="4",
        SPARSE_EPIPE_ENABLE="1", SPARSE_EPIPE_DECODE_ONLY="1", SPARSE_EPIPE_PREFILL_ENABLE="0",
        SPARSE_EPIPE_MODE="upfront_scan", SPARSE_EPIPE_DECODE_RS_MODE="residual_scan",
        SPARSE_EPIPE_RS_HANDLE_LATE_PREFETCH="1", SPARSE_EPIPE_RS_HANDLE_DEMAND_MISS="1",
        SPARSE_EPIPE_RS_FORCE_DEMAND_STAGE="1", SPARSE_EPIPE_NO_EVICT="1",
        SPARSE_EPIPE_TRANSIENT_STAGING="1", SPARSE_EPIPE_MAX_STAGED_EXPERTS="1",
        SPARSE_EPIPE_MIN_FREE_MB="384", SPARSE_PREFETCH_MIN_FREE_MB="32",
        SPARSE_EP_FUSED_WEIGHT_DISPATCH="0", SPARSE_RS_TWO_STAGE="0",
        SPARSE_GLOBAL_EXPERT_CACHE="0", SPARSE_ERS_ENABLE="0", SPARSE_NODEPATCH_ENABLE="0",
        SPARSE_NODE_GROUPED_DISPATCH="0", SPARSE_RANK_GROUPED_DISPATCH="0",
        SPARSE_LOCAL_OWNER_BATCH="0", SPARSE_TRANSITION_PREFETCH="0",
        SPARSE_SCP_EP_PLACEMENT=str(directory / "placement.json"),
        SPARSE_JOINT_COST_PROFILE=str(directory / "costs.json"),
        SPARSE_JOINT_ENABLE=str(int(method != "independent")),
        SPARSE_JOINT_PLACEMENT=str(int(method != "joint-fixed-placement")),
        SPARSE_JOINT_EPIPE_FEEDBACK=str(int(method != "joint-no-epipe-feedback")),
        SPARSE_JOINT_EVALUATOR="reference" if method == "joint-reference" else "incremental",
        SPARSE_JOINT_CACHE_STATIC=str(int(cached)),
        SPARSE_JOINT_COMBINE_EXCHANGE=str(int(combined)),
        SPARSE_JOINT_PREDICTION_CANDIDATES_ONLY=str(int(method in ("joint-candidates", "joint-fast"))),
        SPARSE_JOINT_ROUTE_PASSES="1" if fast else "2",
        SPARSE_JOINT_ROUTE_ITEMS="8" if fast else "16",
        SPARSE_JOINT_PREFETCH_TRIALS="8" if fast else "32",
        SPARSE_JOINT_MAX_EVALUATIONS="24" if fast else "0",
        SPARSE_HF_ATTN_IMPLEMENTATION="eager",
    )
    os.environ.update(settings)


@torch.inference_mode()
def forward_sequence(model, tokens, prompt_len, sparse=False):
    past, logits, elapsed = None, [], []
    for step in range(tokens.shape[1] - prompt_len + 1):
        length = prompt_len + step
        ids = tokens[:, :length] if step == 0 else tokens[:, length-1:length]
        if sparse:
            SparseContext.current_phase = "prefill" if step == 0 else "decode"
            SparseContext.begin_runtime_call(step)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(input_ids=ids, attention_mask=torch.ones_like(tokens[:, :length]),
                       past_key_values=past, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        logits.append(output.logits[:, -1].float().cpu())
        past = output.past_key_values
        if sparse and step == 0:
            DistributedSparseMoeBlock.demote_prefill_scp_transients_for_decode(None)
    return torch.stack(logits), elapsed


def release(model):
    torch.cuda.synchronize()
    SparseContext.initialize(0)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    directory = args.directory.resolve()
    fixture = json.loads((directory / "fixture.json").read_text())
    env = init_dist()
    try:
        if env.world != fixture["world_size"]:
            raise ValueError("fixture world_size differs from torchrun world size")
        torch.manual_seed(fixture["seed"] + env.rank)
        config = AutoConfig.from_pretrained(fixture["model_path"], local_files_only=True)
        tokens = torch.randint(4, config.vocab_size, (args.batch, args.prompt_len + args.steps), device=env.device)
        model_dir = Path(fixture["model_path"])
        signature = dict(fixture=fixture, batch=args.batch, prompt_len=args.prompt_len,
                         steps=args.steps, torch=str(torch.__version__),
                         transformers=transformers.__version__, dtype="float16",
                         config_sha256=hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                         weights=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                                  for p in sorted(model_dir.glob("*.safetensors"))])
        reference_file = directory / f"native-reference-rank{env.rank}.pt"
        if args.reuse_reference:
            cached = torch.load(reference_file, map_location="cpu", weights_only=True)
            if cached["signature"] != signature:
                raise ValueError("reference cache signature changed; rerun without --reuse-reference")
            expected = cached["logits"]
            tokens = cached["tokens"].to(env.device)
        else:
            reference = AutoModelForCausalLM.from_pretrained(
                fixture["model_path"], torch_dtype=torch.float16, attn_implementation="eager",
                local_files_only=True, low_cpu_mem_usage=True).to(env.device).eval()
            expected, _ = forward_sequence(reference, tokens, args.prompt_len)
            torch.save(dict(signature=signature, logits=expected, tokens=tokens.cpu()), reference_file)
            release(reference)
            del reference
            gc.collect(); torch.cuda.empty_cache()
        records, independent_logits = [], None
        for method in args.methods.split(","):
            configure(directory, method)
            metrics = Metrics()
            torch.cuda.reset_peak_memory_stats(env.device)
            bundle = build_model_and_tokenizer(CoreBuildConfig(
                model_path=fixture["model_path"], scp_path=str(directory / "placement.json"),
                rs_dir=str(directory / "rs"), expert_cache=args.expert_cache, cache_policy="lru",
                routing_mode="router", expert_source="router", placement_mode="scp_ep",
                planner_mode="rs_prefetch", prefetch_top_m=2, ep_size=env.world), env, metrics)
            for _ in range(args.warmup):
                SparseContext.reset_runtime_state_for_new_request(clear_expert_cache=True)
                forward_sequence(bundle.model, tokens, args.prompt_len, sparse=True)
            runtime = SparseContext.joint_runtime
            repeats, joint_repeats = [], []
            max_diff, matches = 0., True
            for _ in range(args.repeat):
                SparseContext.reset_runtime_state_for_new_request(clear_expert_cache=True)
                actual, elapsed = forward_sequence(bundle.model, tokens, args.prompt_len, sparse=True)
                diff = actual - expected
                numerical = dict(max_abs_logit_diff=float(diff.abs().max()),
                                 relative_rms_per_step=(diff.square().mean(dim=(1, 2)).sqrt() /
                                                       expected.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-12)).tolist(),
                                 greedy_tokens_match=bool(torch.equal(actual.argmax(-1), expected.argmax(-1))))
                print(json.dumps(dict(method=method, rank=env.rank, numerical=numerical)), flush=True)
                torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                max_diff = max(max_diff, float((actual - expected).abs().max()))
                matches &= bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))
                repeats.append(dict(prefill_ms=elapsed[0], decode_ms=elapsed[1:]))
                if runtime is not None:
                    if runtime.stats["execution_plans"] != args.steps * config.num_hidden_layers:
                        raise AssertionError("joint execution hook did not run on every decode layer")
                    if runtime.stats["prediction_plans"] != args.steps * (config.num_hidden_layers - 1):
                        raise AssertionError("joint prediction hook did not run at each adjacent layer")
                    joint_repeats.append(runtime.summary())
            if not matches:
                raise AssertionError("greedy tokens differ from native Transformers")
            independent_comparison = None
            if method == "independent":
                independent_logits = actual.clone()
            elif independent_logits is not None:
                independent_comparison = dict(
                    max_abs_logit_diff=float((actual - independent_logits).abs().max()),
                    greedy_tokens_match=bool(torch.equal(actual.argmax(-1), independent_logits.argmax(-1))))
                torch.testing.assert_close(actual, independent_logits, atol=args.atol, rtol=args.rtol)
            record = dict(method=method, rank=env.rank, max_abs_logit_diff=max_diff,
                          greedy_tokens_match=matches, repeats=repeats,
                          numerical=numerical,
                          comparison_to_independent=independent_comparison,
                          joint_stats=None if runtime is None else runtime.summary(),
                          joint_stats_by_repeat=joint_repeats,
                          epipe={k: v for k, v in vars(metrics).items() if k.startswith("epipe") and isinstance(v, (int, float))},
                          peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20)
            records.append(record)
            print(json.dumps(dict(status="PASS", method=method, rank=env.rank, max_abs_logit_diff=max_diff)), flush=True)
            release(bundle.model)
            del bundle
            gc.collect(); torch.cuda.empty_cache()
        gathered = [None] * env.world
        dist.all_gather_object(gathered, records)
        if env.rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            report = dict(purpose="functional smoke, not scientific speed/accuracy evidence",
                          fixture=fixture, torch=str(torch.__version__), transformers=transformers.__version__, cuda=torch.version.cuda,
                          source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [ROOT / "utils/eval_core.py", ROOT / "utils/joint_runtime.py",
                                                   ROOT / "utils/joint_scheduler.py", Path(__file__)]},
                          gpu=torch.cuda.get_device_name(), args=vars(args), ranks=gathered)
            (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print(f"Smoke passed; report: {args.out_dir / 'report.json'}", flush=True)
    finally:
        destroy_dist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--directory", type=Path, required=True)
    prep.add_argument("--world-size", type=int, required=True)
    prep.add_argument("--model-path")
    launch = subparsers.add_parser("run")
    launch.add_argument("--directory", type=Path, required=True)
    launch.add_argument("--out-dir", type=Path, required=True)
    launch.add_argument("--methods", default="independent,joint,joint-fixed-placement,joint-no-epipe-feedback")
    launch.add_argument("--batch", type=int, default=2)
    launch.add_argument("--prompt-len", type=int, default=12)
    launch.add_argument("--steps", type=int, default=8)
    launch.add_argument("--expert-cache", type=int, default=4)
    launch.add_argument("--warmup", type=int, default=1)
    launch.add_argument("--repeat", type=int, default=2)
    launch.add_argument("--atol", type=float, default=.005)
    launch.add_argument("--rtol", type=float, default=.005)
    launch.add_argument("--reuse-reference", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare" and args.world_size < 1:
        parser.error("world-size must be positive")
    if args.command == "run" and min(args.batch, args.prompt_len, args.steps, args.repeat) < 1:
        parser.error("batch/prompt-len/steps/repeat must be positive")
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
