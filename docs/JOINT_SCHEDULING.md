# Experimental joint scheduling

The independent implementation is preserved at commit `175da8d`, tag
`sparse-independent-v1`, and branch `baseline/independent-scheduling`.
Development lives on `research/joint-scheduling`; `main` remains the preserved
implementation. This branch contains SPARSE changes only. Comparison-system
baselines remain in the separate local experiment workspace.

The prototype shares a decision model between execution placement, prefetch
admission, and EPIPE residual selection. No paper results or reference tables
have been changed. CUDA functional smoke checks have passed; calibrated
performance and multi-node validation are pending.

## Algorithm

`utils/joint_scheduler.py` uses a probability-weighted layer-latency proxy:

```
max(dispatch send/receive cost)
  + max(rank compute + exposed loading + copy queue - EPIPE overlap)
  + max(combine send/receive cost)
```

This is not an estimator of E[max latency], an optimal solver, or a guarantee
about measured wall time. Correlated activations, collective network contention,
copy-engine concurrency and GEMM interference are not fully modeled.

1. Placement starts at RR owners. A simultaneous readiness-based proposal handles
   tied misses across ranks; bounded coordinate passes refine it using the full
   proxy. Destinations must already have the host weights: only existing SCP
   feasible locations and RR owners qualify. No new host replicas or cross-node
   weight migrations are created.
2. Prefetch candidates are the union of the existing per-source prediction top-K
   sets. The per-rank RS budget is an upper bound and admissions may be zero.
   Each candidate/victim trial updates byte headroom, residency, serial-copy
   completion, and candidate execution location, then evaluates the full proxy.
   Eviction can introduce a new miss and reduce ready work. Protected and
   unfinished residents cannot be evicted. Partial prefetch may help without
   finishing before use. Positive proxy improvement must exceed action cost and
   the minimum-gain threshold. Accepted actions trigger placement refinement.
3. EPIPE chooses one actual late/absent received group for which
   `min(ready compute, remaining transfer) - split cost > 0`. Existing memory,
   staging and correctness guards remain authoritative; rejected splits use
   ordinary execution. This same overlap estimate affects prefetch admission.

Candidate shortlisting uses a transfer/reload heuristic; acceptance uses the full
proxy. Defaults bound search to `JOINT_ROUTE_PASSES=2`, `JOINT_ROUTE_ITEMS=16`,
and `JOINT_PREFETCH_TRIALS=32`. Limits trade decision quality for host overhead.
There is no approximation or measured no-regression guarantee.

## Runtime and correctness

`utils/joint_runtime.py` performs one fixed-width numeric all-gather at each
eligible early-RS decode boundary and another after the actual router at each
decode MoE layer. Prediction planning prepares each rank's share of a common
plan. Actual planning refreshes cache state and demand before dispatch, correcting
predictions and speculative-load failures. Empty sources and zero local budgets
still participate. Resolvers and EPIPE never communicate from data-dependent branches.

Router-selected IDs and weights are untouched. Ordinary EP dispatch, expert
arithmetic, return and accumulation are reused. RS evaluation uses the actual
globally assigned demand on the execution rank. Prefill follows the preserved
SCP policy. Joint decode can select existing host-feasible locations even though
the old `SPARSE_SCP_APPLY_DECODE` flag remains off. Plans are scoped to the
decode step and cleared between requests/repeats.

Initial scope: adjacent-layer probability RS checkpoints, original router,
CPU offload, layer-local LRU, ordinary EP, and residual-scan EPIPE. Group RS,
two-stage refiners, oracle, owner-local execution, routing repair, SKVS, grouped
dispatch and global caches are excluded. Startup checks cost-profile and planner
configuration equality within each EP group. All ranks must enable the same method
and execute the same layer sequence.

## Costs and probability calibration

Copy `configs/joint_cost_profile.example.json` to an ignored path such as
`configs/local/joint_cost_profile.json`, then set `JOINT_COST_PROFILE` in `.env`.
The example contains illustrative numbers and explicitly declares `calibrated=false`.
Use the same file content on all ranks; world size must match EP. Summaries retain
the SHA256 and declared calibration status, which is not independent validation.

Times are ms and bandwidths are bytes/ms. Measure per-rank H2D bandwidth/startup,
expert fixed/per-token compute, next attention, EPIPE split overhead, and directed
activation-link bandwidth/latency. `rank_defaults` supplies common values;
`rank_overrides` maps rank IDs to overrides. `layers` maps layer IDs to
`{defaults, rank_overrides}`. Optional action-cost and eviction reuse-risk penalties
are explicit profile fields.

The preparation horizon currently includes estimated source MoE compute and next
attention, omitting additional source stalls. Unfinished events are queried without
synchronizing and retain a full-transfer remaining-time proxy. Queues assume serial
copies; pending-job estimates can be conservative. GPU traces must establish model error.

Uniform multi-positive CE labels motivate clipped `router_k * q` as an activation
probability proxy; raw softmax scores are not independent activation probabilities.
Optional `activation_calibration[layer]` holds held-out `[proxy, observed_probability]`
knots with increasing x values covering [0,1]. Other RS training targets require
separately fitted empirical calibration. The runtime records which mode is used;
source/destination probabilities use conservative union bounds rather than token
independence assumptions.

## Matched controls

Use identical SCP locations, RS checkpoints, requests, token budgets, persistent
cache capacities and staging headroom:

```bash
METHODS='sparse sparse-joint joint-fixed-placement joint-no-epipe-feedback' \
WORKLOADS='b4' REQUEST_GROUPS='0 1 2 3' RUN_ID=joint-policy-v1 \
  bash scripts/run/launch_two_node.sh
```

| Method | Decode policy |
|---|---|
| `sparse` | Preserved independent policy; joint runtime forced off |
| `sparse-joint` | Joint placement, admission and EPIPE feedback |
| `joint-fixed-placement` | Joint admission/feedback with RR decode mapping |
| `joint-no-epipe-feedback` | Placement/admission uses serial-loading estimates; EPIPE keeps original selection |

All use the same sorted-segment executor. The default matrix remains `sparse`.
Joint launches require the cost profile, supplied SCP locations, and adjacent-layer
RS checkpoints. Existing SCP-regeneration limitations still apply.

## Evidence and remaining work

`joint_stats_by_rank` and `joint_stats_by_repeat` retain prediction/execution/residual
control host times, exchange/planning subcomponent times, metadata bytes, planned
admissions, route changes, residual choices, evaluation counts and profile identity.
Host times include blocking metadata transfers. Subcomponent times overlap the
total-control fields and must not be added again. Planned bytes are estimates;
existing demand/prefetch H2D counters record actual issued traffic.

CPU checks cover eviction/overlap conflicts, partial prefetch, bottleneck priority,
memory/protection guards, legal placement and determinism. Two-process Gloo checks
exercise the actual early-RS hook and resolver, unequal budgets, an empty source,
state reset, and EP dispatch/return with the actual modulelist/fused arithmetic.
IDs/weights remain unchanged and outputs match direct weighted-expert evaluation.

CUDA functional checks also passed on one H100 (EP=1) and two RTX 4090 GPUs
(EP=2, NCCL) with a three-layer Qwen2-MoE fixture. Independent scheduling,
joint scheduling, fixed placement and disabled EPIPE feedback all matched
native Transformers logits within FP16 tolerance and gave identical next-token
argmax values under the same cached input sequence. The two-GPU run exercised
replica route changes and residual EPIPE choices. EP=1 does not test replica
placement or distributed EPIPE. These checks use untrained RS fixtures and
illustrative costs, and establish correctness rather than a performance gain.

Reproduce on an idle pair of GPUs with Transformers 4.44.2 and CUDA PyTorch:

```bash
SMOKE_DIR=results/local/gpu-smoke
python scripts/gpu_smoke.py prepare --directory "$SMOKE_DIR" --world-size 2
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/gpu_smoke.py run \
  --directory "$SMOKE_DIR" --out-dir results/local/gpu-smoke-report
```

Use a fresh directory for each preparation. `prepare --model-path MODEL_DIR`
reuses an existing Qwen2-MoE model without downloading weights; it still creates
untrained predictor fixtures. The report records each rank, repeat, control
counters and numerical differences. Warmup and each measured repeat start with
request metadata reset and cold expert caches. The smoke bypasses the frozen
four-rank paper launcher; it does not change that evaluation protocol.

The deployment checks exposed two shared runtime issues: cache construction
passed a missing `hot` field, and ordinary EP combined returned contributions
with a reduction that rounded differently from native Qwen's sequential
expert-ID accumulation. Both are fixed in this research branch and shared by
its independent/joint controls. The preserved tag and `main` remain unchanged.
GPU materialization now checks mandatory resident tensors directly instead of
rejecting valid models below a fixed one-GB allocation threshold.

This is a host-side Python prototype with synchronous state exchange. Calibrate
costs, verify CUDA events/cache lifetimes, measure control and metadata overhead,
sweep search bounds, and compare total H2D, cache pressure, exposed waits, output
throughput and tail latency before claiming acceleration. Test prediction errors,
topology changes and regressions. CPU proxy improvements establish neither real
speedup nor novelty against neighboring systems.
