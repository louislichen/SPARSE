# Reproducibility Guide

## Frozen Paper Protocol

- Two physical nodes, two GPUs per node, four distributed ranks, and EP=4.
- FP16 model execution with CPU expert offload.
- Homogeneous cache capacity: `8,8,8,8` experts per rank per MoE layer.
- RS prefetch budget: M=2; predictor candidate depth: K=4.
- RS admission thresholds per rank: minimum score 0.04, replacement advantage
  0.08, replacement margin 1.20, and one-step recent-expert protection.
- SCP budget: 96 additional feasible locations, no more than two locations per
  layer-expert pair, and 256 candidates pre-screened per greedy round.
- One warm-up and three measured repeats for each request group.
- Four non-overlapping request groups separated by 32 eligible records.
- Main C4 requests: `b1/p1024/n32`, `b2/p512/n64`, and `b4/p512/n128`.
- Main links: the measured native physical link and symmetric physical
  400 Mbit/s shaping.
- Simulated weak links and simulated compute heterogeneity are disabled.

The two nodes must execute identical method, workload, and request-group order.
The default order in `run_matrix.sh` is deterministic.

## Model-Specific Settings

For Qwen1.5-MoE-A2.7B, set:

```bash
MODEL_KEY=qwen
MODEL_NUM_LAYERS=24
MODEL_NUM_EXPERTS=60
```

For DeepSeek-V2-Lite-Chat, set:

```bash
MODEL_KEY=deepseek
MODEL_NUM_LAYERS=27
MODEL_NUM_EXPERTS=64
```

Each model requires its own routing profile, RS checkpoints, and SCP placement.
Do not reuse Qwen profiles, checkpoints, or placements for DeepSeek or vice versa.

`MODEL_NUM_LAYERS=27` counts Transformer layers for DeepSeek. Its first layer is
dense; profiling, training, and validation use the manifest's `moe_layer_indices`
to identify its 26 MoE layers.

## Request Subsets

Create three mutually exclusive C4 subsets:

1. profile-train for RS training and SCP optimization;
2. profile-validation for offline SCP admission checks;
3. runtime-evaluation for all measured serving rows.

LongBench uses a fourth subset. The C4-trained offline files remain frozen during
LongBench evaluation. Set `DATASET_PATH_OVERRIDE=$LONGBENCH_RUNTIME` when
launching a held-out runtime matrix.

## Offline Preparation Flow

### RS

`profile_rs.sh` records each MoE layer's router input hidden state and the next
layer's router targets. `train_rs.sh` trains one lightweight predictor per
adjacent layer pair and writes `train_config.json`, per-layer checkpoints, and
validation metrics.

### SCP

`profile_scp.sh` collects prefill routing traces from round-robin execution on
the physical four-rank topology. Node 0 gathers rank-local files and merges
train and validation profiles. `solve_scp.sh` selects the replicated SCP objective.
The current release is missing `scripts/scp_cache_replay.py`, which this objective
requires. Full SCP regeneration is pending restoration of that helper. Runtime
runs with supplied placements do not invoke it. Ordinary C4 profile merging does
not require the optional TaskLocal extension.

For full profile regeneration, expect many hours because 320 distributed shards
load and execute the model. Runtime-only evaluation with supplied offline files is
substantially shorter and is the preferred review workflow.

## Runtime Matrices

Full SPARSE runtime:

```bash
METHODS='sparse'
WORKLOADS='b1 b2 b4'
REQUEST_GROUPS='0 1 2 3'
```

Component ablation:

```bash
METHODS='rr scp rs epipe scp-rs rs-epipe sparse'
WORKLOADS='b4'
REQUEST_GROUPS='0 1 2 3'
```

Executor-matched control:

```bash
METHODS='executor-no-epipe executor-with-epipe'
WORKLOADS='b4'
REQUEST_GROUPS='0 1 2 3'
```

Run commands are written to `case_config.json`; compact metrics are written to
`<method>_summary.json`. Raw repeat and log files remain local and should not be
committed.

## Fairness Checks

Before accepting a row, verify:

- `routing_mode=router` and `expert_source=router`;
- identical model and dataset paths on the two compared methods;
- identical `dataset_skip_valid`, batch, prompt, and output budgets;
- identical cache capacity for all cache-enabled methods;
- `sim_weaknet_enabled=0` and `sim_heterogeneity_enabled=0`;
- identical token-count mode and world size;
- no missing distributed rank and no NCCL timeout.

Runtime aggregation uses `e2e_output_tok_s` and its matching standard deviation.
The legacy `throughput_tok_s` includes prompt tokens; the aggregator rejects
summaries that only contain that field. Compact reference CSVs are retained
without numerical edits. Their measurement provenance and uncertainty require
the original per-run summaries; replotting these tables does not audit the
underlying measurements.

An unavailable standard deviation is left blank; a single request group does
not imply zero uncertainty. Total expert-weight H2D is demand plus prefetch
bytes. EPIPE staged-demand bytes are already included in demand and must not be
added a second time. Demand-only reductions do not establish total-byte savings.

`PREFETCH_TOP_M` and `PREDICTION_TOP_K` propagate to every rank. Optional
`PREFETCH_TOP_M_BY_RANK` and `PREDICTION_TOP_K_BY_RANK` override individual rank
budgets; effective vectors are saved in `case_config.json`.

RS profile paths resolve relative to the profile directory, with default
subdirectories used for missing or empty manifest paths. The training launcher
validates chunks before starting DDP.

RS training loads the same paired records on each rank and shards training only
through `DistributedSampler`. Evaluation partitions records without padding;
residual-model logits pass through DDP's forward call. CPU tests check record
coverage and compare two-rank gradients with a global-batch reference. These
fixes require retraining and GPU measurements before making new claims about
prediction quality or throughput.
