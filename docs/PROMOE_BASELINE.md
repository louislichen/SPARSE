# ProMoE-style distributed baseline

This baseline is a new implementation of the mechanisms in
[ProMoE, arXiv:2410.22134v3](https://arxiv.org/html/2410.22134v3), sections 4--6,
adapted to the existing expert-parallel serving harness. The inspected
[authors' repository](https://github.com/promoe-opensource/promoe) revision is
`022f8b22dc11c5fdfbbcee831f6628c14bccf6dd`.
No upstream source is vendored. Use the name **ProMoE-style (Python EP)** in
experiment reports, rather than claiming a reproduction of the original C++
system or its published performance.

## Implemented mechanisms

| Mechanism | Implementation |
|---|---|
| Learned prediction | Independent two-linear-layer ReLU MLP per source/target pair, executed by a CPU worker |
| Intermediate inputs | Pre-attention decoder input captured by a forward pre-hook, identical to the profiling input |
| Stride lookahead | Same-forward-pass source layer `s` predicts `s + distance`; default distance 2 |
| Chunked fetching | One projection tensor per chunk: three chunks for module-list experts, two for fused gate/up experts |
| Early preemption | Precise received expert IDs cancel incorrect speculative jobs for that layer and promote required jobs |
| Priority | One shared per-rank copy worker processes precise demand before speculative work at each chunk boundary |
| Reordered inference | Fully copied experts first, then partially copied experts, then experts not yet started |
| Cache | Independent per-rank, per-MoE-layer LRU; predictions and demand both touch recency |
| Correctness | Cache leases and CUDA compute completion events protect weights still used by kernels |

The original router, top-k experts, combine weights, FP16 expert parameters,
round-robin ownership, attention, and EP dispatch/combine remain the harness
operations. After EP dispatch, the receiving rank has an exact list of its
required experts; this is where precise preemption and reordering occur.
No SPARSE RS score threshold, replacement margin, SCP placement, or EPIPE gate
is used. The sorted-segment expert arithmetic is shared with the matched
reactive control.

Prediction selects the model's top-k experts **per token**, takes their union,
then keeps only experts owned by this rank. Ownership filtering is required
by EP; it is not SPARSE's score-based admission. No extra two-expert prefetch
cap is applied. The persistent cache capacity is the same explicit expert
count used by the other methods. Expert readiness is published only after all
chunks complete. Prefill demand may exceed cache capacity; consumed experts
become evictable only when their GPU computation has finished.

## Preparation and execution

Add the `PROMOE_*` fields from `.env.example` to the node's configuration. Set
the actual model/data paths and `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`,
`EP_SIZE`, visible GPUs, and rendezvous address after selecting the platform.
The launcher accepts one or more GPUs/nodes; it requires `EP_SIZE` to equal
the total process count. Multi-node runs must be started on every node with
the same request shape, dataset offset, method, and run ID.

Prepare a profile on a training subset disjoint from runtime requests:

```bash
bash scripts/run/profile_promoe.sh
bash scripts/run/train_promoe.sh
```

The profiler reuses the trace schema, with `feature_source=decoder_input`.
SPARSE's default gate-input profiles are incompatible and are rejected during
ProMoE-style training. The trainer pairs aligned source/target records at the
configured distance and saves separately labelled predictors. Runtime requires
every eligible pair, checks source/target dimensions and feature provenance,
and refuses RS checkpoints or silent predictor fallback.

Inspect and run the same request configuration for both methods:

```bash
DRY_RUN=1 bash scripts/run/run_promoe.sh promoe-style
DRY_RUN=1 bash scripts/run/run_promoe.sh rr-matched

RUN_ID=promoe-check bash scripts/run/run_promoe.sh promoe-style
RUN_ID=promoe-check bash scripts/run/run_promoe.sh rr-matched
```

`BATCH_OVERRIDE`, `PROMPT_LEN_OVERRIDE`, `MAX_NEW_OVERRIDE`,
`DATASET_SKIP_OVERRIDE`, `WARMUP`, `REPEAT`, and `CACHE_BY_RANK` configure both
paths. Use the same values. `rr-matched` uses reactive caching and the shared
token grouping, expert arithmetic, scatter, and accumulation without ProMoE
prediction/reordering; it is distinct from
the frozen paper's `rr` executor setting. Keep that distinction in summaries.
`REQUEST_GROUP` selects a group offset (default stride 32), and output paths
include the full request shape and group to prevent different cases overwriting
each other. The resulting case files are accepted by the existing throughput
aggregation script. Gains relative to the historical `rr` remain blank when
only the distinct `rr-matched` control is present.

## Metrics and validation status

`perf.py` records `predictive_baseline=promoe-style-python-ep`, distance, and
per-rank counters. Repeat summaries also retain `promoe_stats_by_repeat`.
Counters include CPU prediction time, precise/predicted experts, experts fully
ready at precise declaration, evictions, acquired-weight wait time, copied
chunks, cache capacity/occupancy, and transferred bytes.
Copies already in flight retain their issue-time classification if later
promoted: total expert H2D is **demand bytes + speculative prefetch bytes**.
The same bytes also feed the harness's expert H2D counters. Readiness counters
in `promoe_stats_by_rank` are the baseline's cache measures; SPARSE-specific RS
useful-prefetch counters do not describe this independent implementation.
Generic cache hits for this baseline count weights fully ready when precise
demands are declared; pending copies count as misses. Use that definition when
comparing against another method's cache-hit counters.
Queue drain is included in each measured forward, preventing background work
from escaping the measured request or contaminating the next repeat. Cache
and counter reset happen before measured repeats.

CPU checks cover numerical equivalence for module-list and fused experts,
unchanged routing IDs/weights, demands exceeding a one-slot cache, priority
preemption at a chunk boundary, cancellation of wrong predictions, a simulated
compute-completion eviction fence, worker error propagation, input hooks,
request reset, synthetic stride training/checkpoint loading, and launcher
construction for different EP sizes. **No real-model CUDA or distributed GPU
performance experiment has been run for this baseline.**

## Adaptation limits to resolve before a paper comparison

- The authors' system uses C++ workers and contiguous preallocated GPU caches.
  This version uses Python workers and per-expert PyTorch allocations. Its
  control/allocation overhead must be measured and cannot be assumed equal.
- The copy worker waits for each chunk's CUDA completion before scheduling the
  next chunk. This makes preemption real but may add host scheduling latency;
  a GPU timeline and effective bandwidth check are required.
- Distance is in Transformer-layer indices; pairs involving a dense layer are
  skipped. The first eligible target layers have no same-step prediction.
  Cross-token wraparound and a configurable multi-layer prediction window are
  not implemented.
- The predictor width, uniform top-k cross-entropy target, AdamW settings, and
  token-level 80/10/10 split are this repository's explicit training recipe,
  not recovered original ProMoE training settings. Runtime requests must remain
  disjoint from the entire profiling/training pool.
- Speculative prediction defaults to decode only. Precise fetching/reordering
  operate in prefill and decode. `PROMOE_PREDICT_PREFILL=1` enables speculative
  prefill prediction with the same decoder-input feature, but decode-only
  training does not establish its accuracy. Use appropriate training data and
  report the phase policy before making prefill claims.
- CUDA overlap, numerical tolerance, stream-safe eviction, CPU thread budget,
  peak VRAM/host memory, model coverage, and multi-node behavior need validation
  on the eventual platform. The missing baseline review concern remains open
  until matched experiments and these fidelity checks are completed.
