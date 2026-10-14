# Lightweight local coupling on the independent runtime

This branch starts from the preserved independent commit `175da8d`, rather than
the synchronous global-planner branches. The original tag, `main` and both global
joint research branches remain available. Comparison systems, model weights,
predictor fixtures and raw measurements are not included in the public release.

## Execution and scope

The policy reuses existing asynchronous RS admission, cache state, EP dispatch
and residual-scan EPIPE. It adds no token/layer metadata collective, explicit
device synchronization or device-to-host transfer. Rank-local startup caches
expert weight sizes and platform costs. There is no full-plan search, global
queue snapshot or cross-step global plan.

Only source-local, already-ready host-feasible SCP replicas with lower existing
topology cost than the RR owner may replace the RR execution destination.
Unknown remote readiness and pending local copies fall back to the original RR
owner. The route lookup table is rebuilt only when eligible ready residents
change. It scans the bounded local cache, not every source/expert combination.
Small table updates still incur ordinary host-to-device enqueue work.

RS continues to use its original sparse Top-K candidates and per-rank budget.
Candidate scores are discounted only when the estimated loading could already
be covered by recently observed ready EPIPE work. Resident retention scores stay
in their original units, preserving existing protection and cache-safe replacement
checks. The preparation-window estimate includes current expert compute and next
attention. Missing window estimates retain the original score. This is a local
heuristic, not an optimal joint cost model or a measured gain guarantee.

The existing residual scan supplies ready group/item counts without an additional
scan. EPIPE chooses the candidate with greatest estimated coverable transfer,
rejecting splits whose estimated overlap does not exceed split overhead. Equal
estimates retain the executor's original late-before-demand ordering. These same
ready-work estimates feed the next forecast admission. Original copy streams,
CUDA event lifetime, headroom and fallback guards remain authoritative. Router
expert IDs and combine weights do not change.

Scope is original-router, adjacent-layer probability RS, CPU expert offload,
layer-local LRU and ordinary EP. Remote queue balancing and remote ready-replica
selection require a different state-sharing design and are not implemented here.
Last-step readiness is only a proxy for upcoming ready work. Overestimating it
can suppress useful prefetch; calibration and prediction-error tests remain needed.

## Serving and matched controls

The policy defaults off. Enable it on every EP rank with the same measured
cost profile, independently of the old global planner:

```bash
export SPARSE_LIGHT_ENABLE=1
export SPARSE_LIGHT_COST_PROFILE=/path/to/local-costs.json
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/perf.py \
  --model-path /path/to/model --dataset /path/to/requests.jsonl.gz \
  --rs-dir /path/to/adjacent-rs --placement-mode scp_ep \
  --scp-ep-placement /path/to/placement.json --ep-size 2 \
  --expert-offload cpu --cache-policy lru --planner-mode rs_prefetch
```

The example profile is `configs/lightweight_cost_profile.example.json`; its
world size must match EP. Its costs are illustrative and declare `calibrated=false`.
Times use ms and bandwidth uses bytes/ms. Optional per-rank/layer overrides follow
the fields in `rank_defaults`. Local replica feasibility and relative route cost
come from the existing SCP topology; profile link fields are not used.

`SPARSE_LIGHT_PREFETCH`, `SPARSE_LIGHT_ROUTE` and `SPARSE_LIGHT_FEEDBACK` default
to `1` when enabled and can be independently disabled. Request reset clears
policy feedback, counters and lookup state. `perf.py` exports each rank/layer's
policy counters per repeat. The statistics collective occurs after request
timing and before nonzero ranks return; it is not a serving-layer collective.
Formal launcher methods and the default four-rank paper matrix remain unchanged.

`scripts/benchmark_lightweight.py` supplies five controls around one loaded model:

| Control | Policy |
| --- | --- |
| independent | Original independent decisions with shared correctness fixes |
| prefetch | Candidate-value adapter only, without EPIPE feedback |
| ready-route | Ready-local-replica route lookup only |
| no-feedback | Candidate adapter and route lookup, original EPIPE ordering |
| coupled | Candidate adapter, ready route and EPIPE feedback/selection |

It requires a matching cached native reference and the fixture layout prepared
by `gpu_smoke.py` on the preserved `research/joint-optimization` branch. Reference
signatures verify framework versions, model config and weight-file metadata, and
input dimensions. No checkpoint is trained by these functional fixtures.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/benchmark_lightweight.py \
  --directory results/local/gpu-smoke --out-dir results/local/local-policy \
  --batch 2 --prompt-len 12 --steps 8 --expert-cache 4 --repeat 5 --atol .005
```

Each method gets one warmup and starts every measured request with expert cache
and request metadata reset. The first round uses a fixed order; remaining rounds
use deterministic shuffled orders shared by ranks. Logits and next-token argmax
are checked for each measured request. `--cold-decode` additionally clears the
expert cache at the prefill/decode boundary; report it separately as a diagnostic.

## Current evidence

Tested serving core is `f28c880`. Later commits add serving component switches,
result export and report fields without changing the measured local decision
functions. Shared fixes include `CacheEntry.hot`, native expert-ID accumulation,
the request's prefill-demotion reset and tensor-based CUDA materialization checks.
Every control uses these same fixes; the original snapshot is untouched.

Five-request results below use per-step maximum rank time, averaged within each
request, then mean/sample SD across requests. All outliers are retained. These are
diagnostics with untrained predictors and uncalibrated costs, not paper results.

| Control | H100 tiny EP=1, ms/step | RTX 4090 tiny EP=2, ms/step | RTX 4090 Qwen EP=2, ms/step |
| --- | ---: | ---: | ---: |
| independent | 6.38 ± 0.41 | 9.93 ± 0.39 | 111.84 ± 15.51 |
| prefetch | 6.50 ± 0.49 | 9.97 ± 0.42 | 105.76 ± 0.86 |
| ready-route | 6.35 ± 0.15 | 10.09 ± 0.56 | 106.66 ± 1.77 |
| no-feedback | 6.30 ± 0.01 | 9.94 ± 0.23 | 107.06 ± 4.33 |
| coupled | 6.30 ± 0.01 | 10.12 ± 0.26 | 112.26 ± 13.70 |

The tiny model is three-layer Qwen2-MoE, eight experts/top-2, batch two per rank,
prompt 12 and eight cached decode steps. Qwen1.5-MoE-A2.7B is 24 layers, 60
experts/top-4, batch one per rank, prompt 16 and three cached decode steps. Both
use FP16. H100 uses PyTorch 2.7.1/CUDA 12.8; 4090 uses PyTorch 2.4.0/CUDA 12.1;
both use Transformers 4.44.2. The servers are tested independently.

Qwen's coupled/independent median request means are 105.86/105.31 ms per step.
The instrumented policy host intervals average about 0.92 ms/token, versus about
78 ms/token for the earlier bounded global prototype in a separate run. These
intervals do not measure all added GPU work or resolver time and cannot substitute
for end-to-end latency. The matched results establish removal of the very large
global-planner overhead, but their variability and short request length do not
establish acceleration. The server also experienced memory/I/O pressure during
weight loading; loading was outside the timed requests.

All five controls matched native and independent logits within FP16 tolerance
and had identical next-token argmax. Maximum native differences were zero on H100
tiny, 3.8147e-6 on 4090 tiny and 0.0234375 on Qwen. Real Qwen uses explicit
`atol=0.03, rtol=0.005`; tiny checks use `atol=rtol=0.005`.

Qwen now retains the original 21/19 per-rank prefetch issues per request, unlike
the previous global planner's zero. All five controls' recorded H2D totals are
identical in this short case. Its initial reports retain total logical activation
bytes and inter-node bytes (zero on this same-node run), not detailed remote
rank traffic. Later report fields add that detail. No communication or H2D
reduction is claimed from the initial table.

CPU checks exercise candidate/retention coupling, reset, residual tie handling,
and actual two-rank EP dispatch/return with local replicas, an empty source and
changing pending-event readiness. Received expert arithmetic matches direct
weighted expert evaluation. The full release suite contains 17 checks.

Low control overhead is the supported result so far. Demonstrating a gain still
requires trained RS, measured overlap windows/split costs, longer representative
requests and physical communication-bound configurations, with matched cache,
H2D, wire traffic, exposed waits and tail-latency measurements.
