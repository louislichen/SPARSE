"""Matched local-policy diagnostics with one model load and cached native logits.

The fixture/native references can be prepared with the preserved research smoke
script. All requests use the same weights, input tokens, RS, SCP and executor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
import transformers
from utils.eval_core import (CoreBuildConfig, Metrics, SparseContext,
    DistributedSparseMoeBlock, build_model_and_tokenizer, init_dist, destroy_dist,
    install_lightweight_policies)
from utils.lightweight_policy import CostProfile


@torch.inference_mode()
def sequence(model, tokens, prompt_len, cold_decode=False):
    past, logits, elapsed = None, [], []
    for step in range(tokens.shape[1] - prompt_len + 1):
        length = prompt_len + step
        SparseContext.current_phase = 'prefill' if step == 0 else 'decode'
        SparseContext.begin_runtime_call(step)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(input_ids=tokens[:, :length] if step == 0 else tokens[:, length-1:length],
                    attention_mask=torch.ones_like(tokens[:, :length]), past_key_values=past,
                    use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        logits.append(out.logits[:, -1].float().cpu())
        past = out.past_key_values
        if step == 0:
            DistributedSparseMoeBlock.demote_prefill_scp_transients_for_decode()
            if cold_decode: DistributedSparseMoeBlock.clear_all_expert_caches()
    SparseContext.finalize_prefetch_all()
    return torch.stack(logits), elapsed


def configure(directory):
    os.environ.update(
        SPARSE_LIGHT_ENABLE='0', SPARSE_SCP_APPLY_PREFILL='1', SPARSE_SCP_APPLY_DECODE='0',
        SPARSE_SCP_TRANSIENT_PREFILL_CACHE='1', SPARSE_PLAIN_EP_EXEC_MODE='sorted_segment',
        SPARSE_RS_PREFETCH_POLICY='cache_safe', SPARSE_RS_PREFETCH_PRED_TOPK='4',
        SPARSE_EPIPE_ENABLE='1', SPARSE_EPIPE_DECODE_ONLY='1', SPARSE_EPIPE_PREFILL_ENABLE='0',
        SPARSE_EPIPE_MODE='upfront_scan', SPARSE_EPIPE_DECODE_RS_MODE='residual_scan',
        SPARSE_EPIPE_RS_HANDLE_LATE_PREFETCH='1', SPARSE_EPIPE_RS_HANDLE_DEMAND_MISS='1',
        SPARSE_EPIPE_RS_FORCE_DEMAND_STAGE='1', SPARSE_EPIPE_NO_EVICT='1',
        SPARSE_EPIPE_TRANSIENT_STAGING='1', SPARSE_EPIPE_MAX_STAGED_EXPERTS='1',
        SPARSE_EPIPE_MIN_FREE_MB='384', SPARSE_PREFETCH_MIN_FREE_MB='32',
        SPARSE_EP_FUSED_WEIGHT_DISPATCH='0', SPARSE_RS_TWO_STAGE='0',
        SPARSE_GLOBAL_EXPERT_CACHE='0', SPARSE_ERS_ENABLE='0', SPARSE_NODEPATCH_ENABLE='0',
        SPARSE_NODE_GROUPED_DISPATCH='0', SPARSE_RANK_GROUPED_DISPATCH='0',
        SPARSE_LOCAL_OWNER_BATCH='0', SPARSE_TRANSITION_PREFETCH='0',
        SPARSE_DECODE_CACHE_AWARE_SCP_REMOTE_SNAPSHOT='0',
        SPARSE_HF_ATTN_IMPLEMENTATION='eager', SPARSE_SCP_EP_PLACEMENT=str(directory/'placement.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--methods', default='independent,prefetch,ready-route,no-feedback,coupled')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--prompt-len', type=int, default=16)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--expert-cache', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--atol', type=float, default=.03)
    parser.add_argument('--rtol', type=float, default=.005)
    parser.add_argument('--cold-decode', action='store_true')
    args = parser.parse_args()
    if min(args.batch, args.prompt_len, args.steps, args.repeat) < 1: parser.error('invalid dimensions')
    methods = args.methods.split(',')
    flags = {'independent':None, 'prefetch':(True,False,False), 'ready-route':(False,True,False),
             'no-feedback':(True,True,False), 'coupled':(True,True,True)}
    if len(set(methods)) != len(methods) or any(m not in flags for m in methods): parser.error('invalid methods')
    env = init_dist()
    try:
        directory = args.directory.resolve()
        fixture = json.loads((directory/'fixture.json').read_text())
        if env.world != fixture['world_size']: raise ValueError('fixture EP mismatch')
        cached = torch.load(directory/f'native-reference-rank{env.rank}.pt', map_location='cpu', weights_only=True)
        model_path = Path(fixture['model_path'])
        signature = dict(fixture=fixture, batch=args.batch, prompt_len=args.prompt_len, steps=args.steps,
                         torch=str(torch.__version__), transformers=transformers.__version__, dtype='float16',
                         config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
                         weights=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(model_path.glob('*.safetensors'))])
        if cached['signature'] != signature: raise ValueError('native reference signature mismatch')
        tokens, expected = cached['tokens'].to(env.device), cached['logits']
        configure(directory)
        metrics = Metrics()
        bundle = build_model_and_tokenizer(CoreBuildConfig(model_path=fixture['model_path'],
            scp_path=str(directory/'placement.json'), rs_dir=str(directory/'rs'),
            expert_cache=args.expert_cache, cache_policy='lru', routing_mode='router', expert_source='router',
            placement_mode='scp_ep', planner_mode='rs_prefetch', prefetch_top_m=2, ep_size=env.world), env, metrics)
        profile = CostProfile(directory/'costs.json', env.world)
        blocks = list(SparseContext.layer_blocks.values())
        policies = {}
        for method in methods:
            if flags[method] is None: policies[method] = [None] * len(blocks)
            else:
                p,r,f = flags[method]
                install_lightweight_policies(profile, prefetch=p, route=r, feedback=f)
                policies[method] = [block._light_policy for block in blocks]
        def select(method):
            for block, policy in zip(blocks, policies[method]): block._light_policy = policy
            SparseContext.reset_runtime_state_for_new_request(clear_expert_cache=True)
            metrics.reset_for_repeat()
            torch.cuda.synchronize(env.device)
            dist.barrier(device_ids=[env.local_rank])
        for method in methods:
            select(method)
            actual,_ = sequence(bundle.model,tokens,args.prompt_len,args.cold_decode)
            torch.testing.assert_close(actual,expected,atol=args.atol,rtol=args.rtol)
        records,orders,canonical = {m:[] for m in methods},[],None
        for repeat in range(args.repeat):
            order = sorted(methods,key=lambda m:m!='independent') if repeat==0 else list(methods)
            if repeat: random.Random(20261014+repeat).shuffle(order)
            orders.append(order)
            for method in order:
                select(method)
                actual,elapsed = sequence(bundle.model,tokens,args.prompt_len,args.cold_decode)
                torch.testing.assert_close(actual,expected,atol=args.atol,rtol=args.rtol)
                if not torch.equal(actual.argmax(-1),expected.argmax(-1)): raise AssertionError('native argmax differs')
                if method=='independent' and canonical is None: canonical=actual.clone()
                if canonical is not None:
                    torch.testing.assert_close(actual,canonical,atol=args.atol,rtol=args.rtol)
                    if not torch.equal(actual.argmax(-1),canonical.argmax(-1)): raise AssertionError('independent argmax differs')
                policy_stats=[p.summary() for p in policies[method] if p is not None]
                traffic = {k:getattr(metrics,k,0) for k in ('comm_bytes_activation','comm_bytes_activation_inter_node',
                    'expert_weight_h2d_bytes','expert_weight_h2d_prefetch_bytes','expert_weight_h2d_demand_bytes',
                    'cache_hit_count','cache_miss_count','epipe_rs_trigger_count')}
                records[method].append(dict(repeat=repeat,decode_ms=elapsed[1:],prefill_ms=elapsed[0],
                    max_native_logit_diff=float((actual-expected).abs().max()),next_tokens_match=True,
                    policies=policy_stats,prefetch_issued=metrics.prefetch_issued_count,
                    prefetch_completed=metrics.prefetch_completed_count,traffic=traffic))
                if env.rank==0: print(json.dumps(dict(status='PASS',method=method,repeat=repeat,
                    decode_mean_ms=sum(elapsed[1:])/args.steps)),flush=True)
        all_records=[None]*env.world
        dist.all_gather_object(all_records,records)
        if env.rank==0:
            args.out_dir.mkdir(parents=True,exist_ok=True)
            report=dict(purpose='local-policy engineering diagnostic; untrained RS, illustrative costs',
                args=vars(args),fixture=fixture,ranks=all_records,orders=orders,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    ('utils/eval_core.py','utils/lightweight_policy.py','scripts/benchmark_lightweight.py')})
            (args.out_dir/'report.json').write_text(json.dumps(report,indent=2,default=str))
    finally: destroy_dist()


if __name__=='__main__': main()
