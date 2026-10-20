# Verified evidence for the original SPARSE architecture

This bundle contains scalar measurements and provenance for writing, not model
weights, datasets, generated text, private baseline implementations or machine
configuration. The release is `sparse-original-stable-20261020`: serving snapshot
`e768424`, promoted by `dce871c`. The later global coordination mechanism is not
part of that release or the original-system rows below.

## Read the version relationship before using a number

| Dataset | What was measured | Relationship to the release | Coverage |
|---|---|---|---|
| A | RS ON versus bracketed RS OFF on SCP420+EPIPE | The measured RS adapter and staging sources exactly match `e768424`; the measurement used a private harness, not an identical public checkout | Calibration g1, b4/p512/n128, warmup 1, repeat 1 |
| B | Original SCP+RS+EPIPE and six mechanism-adapted baselines | Original architecture, joint disabled, in a later frozen private serving tree `49b79a8`; **not an exact-tree benchmark of `e768424`** | Runtime g2, b4/p512/n128, warmup 1, repeat 3 |
| C | Earlier six-baseline-only matrix | Earlier numerical/harness protocol; no complete SPARSE row | b1/b2/b4, four groups each, warmup 1, repeat 3 |

Do not merge these datasets into one performance table. In particular, C cannot
supply b1/b2 rows for B's original SPARSE comparison, and A's calibration gain
cannot be presented as a held-out final-runtime result. No complete b1/b2/b4
same-protocol baseline-plus-original-SPARSE table is supplied here.

The existing `results/reference/` contains a separate historical paper-data
collection. This bundle neither replaces it nor certifies it as a measurement
of the newly published checkout.

## A. Release-aligned RS calibration evidence

The actual two-node EP4 deployment uses two RTX 4090s and two RTX 3090s,
Qwen1.5-MoE-A2.7B, FP16, PyTorch 2.2.0+cu121. Each rank has an LRU cache of eight
experts and the same four-slot RS scratch budget for ON/OFF controls. RS uses
top-8 predictions, at most two copies, and `empty_steps=0`. SCP is used for
prefill; decode retains RR ownership. EPIPE is enabled.

The calibrated placement has **420 extra CPU expert copies**, a changed-constraint
SCP configuration; it must not be described as the strict zero-replica SCP case.
CPU expert inventory is not GPU-resident replication.

| Comparison against same-run bracketed RS OFF | Observed value |
|---|---:|
| End-to-end throughput change | +2.1760% |
| Service-decode throughput change | +2.8679% |
| Total demand + prefetch H2D change | +0.8892% |
| OFF control drift | 0.8323% |
| Full own-rank input/output checks | 16/16 |

All source, output and counter checks passed. This diagnostic passed its
**original 1.02x total-H2D screen**, not a strict zero-increase H2D criterion.
Independent repeated confirmation was not completed. No confidence interval or
universal positive RS benefit follows from this single calibration group.

`A_rs_calibration_cases.csv` contains all four cases, including RR and both OFF
brackets. Its gain uses the reciprocal of the mean OFF service time, matching
the original diagnostic. It is the incremental effect of RS on SCP420+EPIPE,
not the full system's improvement over other baselines.

## B. Matched original-architecture comparison: b4, runtime g2 only

All rows use the same two-node/four-GPU resource configuration, input group,
fixed64 prefill and per-token native decode (`native_decode_granularity=1`).
Warmup uses eight generated tokens; measured requests generate 128 tokens with
batch four and prompt length 512. Cache reset, CPU expert inventory and model/RS
weights are fixed across methods. LoD has cache zero by definition; the other
methods have cache eight. Original SPARSE uses the same four-slot RS pool and
two-copy budget; its global coordination mechanism is disabled.

The CPU inventory is a constant superset with 561 extra copies, while the SCP
placement itself has 420 extra copies. Baselines share that inventory to keep
the compared execution environment constant; this is not an official-engine
memory-footprint comparison.

| Method | E2E output tok/s | Service-decode tok/s | Total weight H2D GiB |
|---|---:|---:|---:|
| RR-LRU | 8.0460 | 10.5914 | 454.9746 |
| LoD | 6.2910 | 7.7173 | 955.7271 |
| EPLB-like, r96 | 8.0468 | 10.6678 | 454.7974 |
| Prism/DanceMoE-style, r32 | 8.0618 | 10.5454 | 455.4902 |
| Static-hot, k8 | 8.0659 | 10.6076 | 448.6743 |
| MoE-Infinity-style, k1 | 7.4722 | 9.6159 | 472.8765 |
| Original SPARSE, joint OFF | 10.9230 | 15.6041 | 406.3608 |

At **this input group**, original SPARSE has +35.7575% observed E2E throughput
against RR and +35.4216% against the strongest measured baseline, static-hot.
These are full-system comparisons, not gains attributable to a new module.
Only three repeats of one input group are available here; these percentages
must not be described as an all-workload or multi-group average.

Baseline names denote private **mechanism adaptations in the common runtime**,
not official EPLB, Prism, DanceMoE or MoE-Infinity engine deployments.

- `B_original_controls_summary.csv`: seven method summaries. Baselines average
  three repeats; SPARSE averages three BEFORE and three AFTER measurements.
- `B_original_controls_repeats.csv`: all 24 original/control measurements, with
  separate BEFORE/AFTER rows. No experimental joint-policy measurements included.
- `B_original_controls_rank_repeats.csv`: 96 scalar rank records. Their complete
  inputs/outputs were checked against the corresponding same-rank RR.

The source group contains nine configurations, including an experimental
`coupled` configuration. This export excludes it. The complete 108-case matrix
was not finished; only this complete nine-configuration input group was audited.

## C. Earlier baseline-only coverage

`C_earlier_baseline_only_summary.csv` contains 18 aggregates: six methods for
each of b1/p1024/n32, b2/p512/n64 and b4/p512/n128. There are four groups and
three measured repeats per group: 72 cases and 864 independently checked full
rank-output records in total. Standard deviations are across input-group means;
paired gains are averaged per-group ratios.

This earlier fixed64/batched-native protocol differs from B's per-token-native
decode. The file intentionally has **no SPARSE row**. Filling that gap with B,
with a calibration result or with an older ablation run would create an invalid
comparison.

## Metric and interpretation rules

- E2E output throughput counts generated tokens only. The old
  `(prompt + generated) / time` throughput alias is not used.
- EP ranks repeat inputs. Throughput counts one copy (`ep_rank0_only`), never
  four times the number of independent requests.
- Service decode is `wall_total - wall_prefill`, including request finishing
  and statistics costs. Raw greedy-loop decode is exported separately where
  available. Do not silently substitute one metric for the other.
- B takes the maximum rank time per request; H2D sums all four ranks. It then
  averages measured repeats. SPARSE's BEFORE/AFTER brackets remain separable.
- Total H2D includes both demand and prefetch transfers. Reducing demand-only
  H2D is not a claim that total H2D fell.
- Inter-/intra-node activation byte counters measure application payload, not
  NIC wire bytes. They exclude transport framing and are not a bandwidth probe.
- All admitted outputs equal same-group, same-rank RR. This does not establish
  exact equality to native Hugging Face execution, which is a separate check.
- These heterogeneous GPUs do not authenticate the old 2080Ti platform's
  absolute paper throughput. No new batch scan, scalability claim, official
  baseline-engine reproduction or new global-mechanism benefit is established.

## Provenance and writing assets

`provenance.json` records configuration, source relationship, coverage and
source-report SHA256 values. `source_hashes.json` records safe logical raw-file
identifiers and SHA256 values without publishing their document/token contents.
`manifest.json` authenticates the published bundle. Raw archives, model and RS
weights, document/token arrays, server logs, hostnames, UUIDs and infrastructure
paths remain private.

`paper_tables.tex` contains editable tables with their scope in the captions.
Use these as draft material only; keep the scope statements when adapting them
to a manuscript. A and B answer different experimental questions and must not
be merged into a single measured gain.
