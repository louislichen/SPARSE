# Joint planner optimization and diagnostics

The implementation is on `research/joint-optimization`, based on the preserved
`research/joint-scheduling` prototype. `main`, `baseline/independent-scheduling`
and tag `sparse-independent-v1` remain unchanged. These experiments use SPARSE
controls only; comparison-system code and raw measurements remain outside the
public release.

## Implemented changes

The incremental evaluator caches per-rank demand, readiness, loading and link
totals. A routing trial updates the affected ranks and edges. EPIPE overlap is
tracked with the two largest activation probabilities for each transfer-time
group. Uniform-size experts make these overlap queries independent of expert
count; variable-size experts retain the same cost proxy. Previously computed
scores are reused instead of evaluating identical states twice. The reference
evaluator remains available for paired decision checks.

Optional candidate filtering limits forecast placement search to the union of
RS top-K candidates. The full proxy still includes the forecast probability
tail at its RR owners. Actual-router placement searches all actual demand.
Optional deterministic evaluation limits bound each planning invocation and
return a legal feasible route when exhausted. These options change search scope
and can change decisions; they do not guarantee the same plan as the reference.

Optional static caching reuses expert byte sizes, validated rank costs and
metadata buffers. Predictor validity travels in the shared metadata instead of
requiring separate scalar CUDA-to-host checks. Optional combined exchange stages
the next-layer forecast and gathers it with current-layer actual demand. This
reduces physical collectives from `2*L-1` to `L` per decode token, while retaining
`2*L-1` logical payloads. Payload volume is essentially unchanged for equal expert
counts; different counts require padding.

Combined exchange requires adjacent, hidden-only RS features. It delays forecast
admission until the current router boundary. Actual planning conservatively adds
the newly planned target prefetch bytes and queue time to its earlier snapshot;
it does not credit victim evictions or refresh completion events at that point.
Runtime memory, staging and copy-event guards remain authoritative. Both the
decision timing and queue estimate can differ from separate exchanges.

A shared request-reset correction clears the prefill-demotion guard before each
request. Previously the first warmup could prevent demotion on later requests.
All measured controls use this correction. Earlier single-run timings and the
new repeated protocol should not be compared as a matched pair.

## Controls and configuration

`scripts/benchmark_joint_runtime.py` switches policies around one loaded model,
using the same RS checkpoints, SCP locations, router, expert executor and cached
input tokens. Each policy gets one warmup; every request clears expert cache and
request state. Five measured rounds use one fixed order followed by deterministic
shuffled orders shared by every rank. Orders are shuffled, not statistically
balanced. Native reference signatures check model-file metadata, framework
versions and input parameters. Logits and next-token argmax are checked on every
measured request.

| Diagnostic name | Change from the full reference search |
| --- | --- |
| `independent` | Joint planner disabled; preserved independent policy with shared runtime fixes |
| `joint-reference` | Full evaluator, separate metadata collectives |
| `joint-incremental` | Incremental routing evaluator and score reuse |
| `joint-candidates` | Incremental evaluator plus forecast candidate filtering |
| `joint-cache` | Incremental evaluator plus static caching and shared validity marker |
| `joint-combined` | Incremental evaluator, caching and combined exchange |
| `joint-fast` | All options; one route pass, eight route items, eight prefetch trials, 24 evaluations per plan |

The diagnostic names are local benchmark controls; the serving CLI continues to
use `sparse-joint` and its existing component controls. Serving options are:

| Environment variable | Default |
| --- | --- |
| `SPARSE_JOINT_EVALUATOR` | `incremental` (`reference` selects the original evaluator) |
| `SPARSE_JOINT_CACHE_STATIC` | `0` |
| `SPARSE_JOINT_COMBINE_EXCHANGE` | `0` |
| `SPARSE_JOINT_PREDICTION_CANDIDATES_ONLY` | `0` |
| `SPARSE_JOINT_ROUTE_PASSES` | `2` |
| `SPARSE_JOINT_ROUTE_ITEMS` | `16` |
| `SPARSE_JOINT_PREFETCH_TRIALS` | `32` |
| `SPARSE_JOINT_MAX_EVALUATIONS` | `0` (unlimited; otherwise at least three) |

Set the same configuration on every EP rank. Startup verifies configuration,
cost-profile identity and predictor layout before serving.

## Measured engineering results

The tested core is commit `b8b90a9`. The following values are mean and sample
standard deviation across five repeated requests. For each decode step, take
the maximum rank time, then average steps within the request. They are
control-overhead diagnostics with untrained RS fixtures, illustrative uncalibrated
costs and fixed synthetic input tokens; they do not establish trained-predictor
quality, paper throughput, heterogeneous-cluster performance or novelty.

| Control | H100 tiny EP=1, ms/step | RTX 4090 tiny EP=2, ms/step | RTX 4090 pretrained Qwen EP=2, ms/step |
| --- | ---: | ---: | ---: |
| independent | 6.11 ± 0.02 | 10.39 ± 0.16 | 107.31 ± 0.76 |
| joint-reference | 9.04 ± 0.02 | 19.54 ± 0.43 | 503.76 ± 1.49 |
| joint-incremental | 8.99 ± 0.02 | 19.83 ± 0.40 | 450.84 ± 3.31 |
| joint-candidates | 8.99 ± 0.03 | 19.54 ± 0.37 | 392.85 ± 2.61 |
| joint-cache | 8.69 ± 0.03 | 18.89 ± 0.33 | 465.85 ± 65.26 |
| joint-combined | 8.38 ± 0.03 | 18.61 ± 0.53 | 445.65 ± 34.31 |
| joint-fast | 8.12 ± 0.07 | 16.02 ± 0.53 | 222.58 ± 1.77 |

The tiny Qwen2-MoE fixture has three layers/eight experts/top-2, batch two per
rank, prompt 12 and eight cached decode steps. The pretrained Qwen1.5-MoE-A2.7B
has 24 layers/60 experts/top-4, batch one per rank, prompt 16 and three cached
decode steps. Both use FP16 and CPU-offloaded experts. H100 uses PyTorch 2.7.1
with CUDA 12.8; 4090 uses PyTorch 2.4.0 with CUDA 12.1. Both use Transformers
4.44.2. These runs do not combine the two machines into one EP group.

Qwen's optimized combination reduces reference latency by 55.8%, but remains
2.07 times slower than the independent control. Its planner time averages about
78 ms per token across ranks, versus 336 ms for the reference. Metadata
collectives fall from 47 to 24 per token; measured exchange time includes
blocking peer waits and device-to-host copies, not just network wire latency.
The two high-variance controls are reported with all repeats retained.

No joint Qwen prefetch was admitted in this short diagnostic, whereas the
independent control issued prefetches. Therefore these results test route changes,
residual choices and planning overhead, without establishing a benefit from joint
prefetch. The cost-profile and probability calibration remain unresolved inputs.

All seven controls passed numerical and next-token checks on both tiny runs and
the pretrained model. Maximum native logit difference was zero on H100 tiny,
3.8147e-6 on 4090 tiny, and 0.0234375 on pretrained Qwen. The real-model check
uses `atol=0.03, rtol=0.005` for the expert-batching FP16 rounding described in
[joint scheduling](JOINT_SCHEDULING.md); tiny checks keep `atol=rtol=0.005`.

Replaying 141 actual Qwen planning inputs through both CPU evaluators for ten
paired rounds produced identical routes, admissions and before/after proxy
values within the comparison tolerance. Mean planning time fell from 7.079 to
5.915 ms per invocation (16.4%). The mixed execution/forecast median improved
only 1.04 times; synthetic four-rank dense-forecast cases show larger gains and
should not substitute for actual-input measurements. Profiling these actual
inputs shows complete prefetch/victim evaluations still dominate the remaining
Python planner cost. Limiting search contributes much more than routing
incremental evaluation alone in this workload.

CPU/Gloo checks cover randomized trial/commit equivalence, complete plan
equivalence, variable expert sizes, memory/protection constraints, deterministic
budgets, saturated probability unions, zero budgets, empty sources and bad-score
rejection on both ranks. CUDA performance-CLI integration also passed two repeats
with combined exchange and per-rank prefetch budgets 0/2: all ranks participated,
the zero-budget rank admitted nothing, and the other admitted/issued four
prefetches per request. Its exported cycle accounting observed three completions;
this count is not proof that every issued transfer was consumed.

A separate cold-decode-cache CUDA check compared fixed-token logits for four
controls with budgets 0/2 and EPIPE feedback disabled. The optimized combination
issued and observed completion of two prefetches on the nonzero-budget rank in
each of two repeats; the zero-budget rank issued none. All outputs and next-token
argmax matched native within tiny-model tolerance (maximum difference 3.8147e-6).
This check includes untimed end-of-request event accounting and is separate from
the seven-control latency table.

## Reproduce the diagnostics

Prepare a fixture and populate its native reference with `gpu_smoke.py`, using
the same input dimensions and framework as the later benchmark. The benchmark
does not generate a missing or mismatched native reference automatically.

```bash
SMOKE_DIR=results/local/gpu-smoke
python scripts/gpu_smoke.py prepare --directory "$SMOKE_DIR" --world-size 2
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/gpu_smoke.py run \
  --directory "$SMOKE_DIR" --out-dir results/local/native-check \
  --methods independent --batch 2 --prompt-len 12 --steps 8 --expert-cache 4

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/benchmark_joint_runtime.py \
  --directory "$SMOKE_DIR" --out-dir results/local/joint-runtime \
  --batch 2 --prompt-len 12 --steps 8 --expert-cache 4 \
  --repeat 5 --warmup 1 --atol .005 --capture-planner

python scripts/benchmark_joint_planner.py \
  --trace results/local/joint-runtime/planner-inputs.jsonl \
  --out results/local/joint-planner-replay.json --rounds 10
```

`--capture-planner` records a separate untimed reference request. It does not add
trace serialization to measured requests. Without `--trace`, the planner script
uses deterministic synthetic cases. Raw reports include per-rank repeats,
method orders, counters and source hashes; they remain ignored by Git.

Further work should first reduce complete prefetch/victim evaluation cost, then
calibrate the model and use trained RS on longer, representative requests. Measure
actual exposed H2D waits and memory pressure as well as planner time. Current
measurements do not support claiming acceleration over independent SPARSE.
