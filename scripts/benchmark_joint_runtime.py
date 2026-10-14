"""Matched scheduler diagnostics using one loaded model and cached native outputs.

All methods share weights, router, RS, placement, cache and requests. Reference
and optimized planners are switched between requests, with one cold-cache
warmup per method and a balanced method order across measured rounds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
import transformers
from scripts.gpu_smoke import configure, forward_sequence
from utils.eval_core import CoreBuildConfig, Metrics, SparseContext, build_model_and_tokenizer, init_dist, destroy_dist
from utils.joint_runtime import JointRuntime


def reference(directory, fixture, args, env):
    path = directory / f"native-reference-rank{env.rank}.pt"
    data = torch.load(path, map_location='cpu', weights_only=True)
    model = Path(fixture['model_path'])
    signature = dict(fixture=fixture, batch=args.batch, prompt_len=args.prompt_len,
                     steps=args.steps, torch=str(torch.__version__),
                     transformers=transformers.__version__, dtype='float16',
                     config_sha256=hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
                     weights=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                              for p in sorted(model.glob('*.safetensors'))])
    if data['signature'] != signature:
        raise ValueError('native reference parameters changed; regenerate with gpu_smoke.py run')
    return data['tokens'].to(env.device), data['logits']


def encode_trace(record):
    return {key: asdict(value) if key == 'demand' else [asdict(r) for r in value] if key == 'ranks'
            else value for key, value in record.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--methods', default='independent,joint-reference,joint-incremental,joint-candidates,joint-cache,joint-combined,joint-fast')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--prompt-len', type=int, default=16)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--expert-cache', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--atol', type=float, default=.03)
    parser.add_argument('--rtol', type=float, default=.005)
    parser.add_argument('--capture-planner', action='store_true')
    args = parser.parse_args()
    if min(args.batch, args.prompt_len, args.steps, args.repeat) < 1 or args.warmup < 0:
        parser.error('invalid request/repeat parameters')
    directory = args.directory.resolve()
    fixture = json.loads((directory / 'fixture.json').read_text())
    methods = args.methods.split(',')
    if len(methods) != len(set(methods)): parser.error('methods must be unique')
    env = init_dist()
    try:
        if env.world != fixture['world_size']: raise ValueError('fixture EP size mismatch')
        tokens, expected = reference(directory, fixture, args, env)
        configure(directory, 'joint-reference')
        metrics = Metrics()
        bundle = build_model_and_tokenizer(CoreBuildConfig(
            model_path=fixture['model_path'], scp_path=str(directory / 'placement.json'),
            rs_dir=str(directory / 'rs'), expert_cache=args.expert_cache, cache_policy='lru',
            routing_mode='router', expert_source='router', placement_mode='scp_ep',
            planner_mode='rs_prefetch', prefetch_top_m=2, ep_size=env.world), env, metrics)
        first_block = next(iter(SparseContext.layer_blocks.values()))
        runtimes = {}
        for method in methods:
            configure(directory, method)
            runtime = None if method == 'independent' else JointRuntime.from_env(SparseContext, first_block.comm_world)
            if runtime is not None: runtime.verify(first_block)
            runtimes[method] = runtime

        def select(method):
            configure(directory, method)
            SparseContext.joint_runtime = runtimes[method]
            SparseContext.reset_runtime_state_for_new_request(clear_expert_cache=True)
            metrics.reset_for_repeat()
            torch.cuda.synchronize(env.device)
            dist.barrier(device_ids=[env.local_rank])

        # Warm each method independently. There are no checkpoint reloads or
        # asymmetric cache-history advantages between timed methods.
        for method in methods:
            for _ in range(args.warmup):
                select(method)
                actual, _ = forward_sequence(bundle.model, tokens, args.prompt_len, sparse=True)
                torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)

        if args.capture_planner and 'joint-reference' in methods:
            select('joint-reference')
            runtime = runtimes['joint-reference']
            runtime.capture = []
            forward_sequence(bundle.model, tokens, args.prompt_len, sparse=True)
            captured, runtime.capture = runtime.capture, None
            if env.rank == 0:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                (args.out_dir / 'planner-inputs.jsonl').write_text(
                    ''.join(json.dumps(encode_trace(record)) + '\n' for record in captured))

        records = {method: [] for method in methods}
        orders = []
        canonical, reference_logits = None, None
        for repeat in range(args.repeat):
            if repeat == 0:
                order = sorted(methods, key=lambda m: 0 if m == 'independent' else 1 if m == 'joint-reference' else 2)
            else:
                order = list(methods)
                random.Random(20261014 + repeat).shuffle(order)
            orders.append(order)
            for method in order:
                select(method)
                torch.cuda.reset_peak_memory_stats(env.device)
                actual, elapsed = forward_sequence(bundle.model, tokens, args.prompt_len, sparse=True)
                torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
                    raise AssertionError(f'{method} native next-token argmax differs')
                if method == 'independent' and canonical is None: canonical = actual.clone()
                if method == 'joint-reference' and reference_logits is None: reference_logits = actual.clone()
                if canonical is not None:
                    torch.testing.assert_close(actual, canonical, atol=args.atol, rtol=args.rtol)
                    if not torch.equal(actual.argmax(-1), canonical.argmax(-1)):
                        raise AssertionError('next-token argmax differs from independent policy')
                runtime = runtimes[method]
                stats = None if runtime is None else runtime.summary()
                if stats is not None:
                    if stats['execution_plans'] != args.steps * bundle.num_layers:
                        raise AssertionError('missing execution plans')
                    if stats['prediction_plans'] != args.steps * (bundle.num_layers-1):
                        raise AssertionError('missing prediction plans')
                    expected_calls = args.steps * (bundle.num_layers if runtime.combine_exchange else 2*bundle.num_layers-1)
                    if stats['exchange_calls'] != expected_calls:
                        raise AssertionError('metadata collective count mismatch')
                record = dict(repeat=repeat, method=method, rank=env.rank,
                              prefill_ms=elapsed[0], decode_ms=elapsed[1:], joint_stats=stats,
                              max_native_logit_diff=float((actual-expected).abs().max()),
                              max_independent_logit_diff=None if canonical is None else float((actual-canonical).abs().max()),
                              max_reference_logit_diff=None if reference_logits is None else float((actual-reference_logits).abs().max()),
                              next_tokens_match=True, peak_cuda_mb=torch.cuda.max_memory_allocated(env.device)/2**20,
                              prefetch_issued=metrics.prefetch_issued_count,
                              prefetch_completed=metrics.prefetch_completed_count)
                records[method].append(record)
                if env.rank == 0:
                    print(json.dumps(dict(status='PASS', method=method, repeat=repeat,
                                          decode_mean_ms=sum(elapsed[1:])/args.steps)), flush=True)
        gathered = [None] * env.world
        dist.all_gather_object(gathered, records)
        if env.rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            report = dict(purpose='matched control-overhead diagnostics; untrained RS and illustrative costs',
                          fixture=fixture, args=vars(args), method_orders=orders, ranks=gathered,
                          torch=str(torch.__version__), transformers=transformers.__version__,
                          gpu=torch.cuda.get_device_name(),
                          source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                         ['utils/eval_core.py','utils/joint_scheduler.py','utils/joint_runtime.py',
                                          'scripts/gpu_smoke.py','scripts/benchmark_joint_runtime.py']})
            (args.out_dir/'report.json').write_text(json.dumps(report,indent=2,default=str))
            print(f'Wrote {args.out_dir / "report.json"}',flush=True)
    finally:
        destroy_dist()


if __name__ == '__main__':
    main()
