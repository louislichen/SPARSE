# Running and Checking the Code

## What the Code Supports

1. SPARSE runs without changing router-selected experts or combine weights.
2. The main two-node matrix compares SPARSE and controlled baselines under
   matched model, requests, network, cache capacity, and token accounting.
3. The component matrix isolates SCP, RS, EPIPE, and their combinations.
4. Runtime counters expose demand H2D traffic, cache hits, prefetch usefulness,
   completion-before-use, and topology-weighted route cost.
5. The reference plotting workflow regenerates four equal-canvas Qwen figures
   from compact source data.

## Suggested Reviewer Workflow

### A. Functional, CPU-only check (approximately 1 minute)

```bash
bash scripts/validate_release.sh
bash scripts/run/reproduce.sh reference-figures
```

Expected output: `[validate] PASS` and four PDF/PNG pairs under
`results/figures/`. The four PNG files must have identical pixel dimensions.

### B. Environment and two-node smoke check

Configure `.env` on both nodes, then run:

```bash
python scripts/check_env.py --env-file .env --strict

METHODS='rr sparse' WORKLOADS='b1' REQUEST_GROUPS='0' REPEAT=1 WARMUP=0 \
RUN_ID=sparse-smoke \
  bash scripts/run/launch_two_node.sh
```

Expected output: one RR and one SPARSE summary on node 0, no NCCL timeout, and
`sim_weaknet_enabled=0` in both summaries.

### C. Main runtime reproduction

Use the frozen RS checkpoints and SCP placement, then run the main or ablation
matrix from the README.
The full duration depends strongly on GPU generation and host-to-device expert
loading; reserve an overnight window for the complete 84-case main matrix.

## Expected Files

```text
results/local/runtime/<RUN_ID>/.../<method>/case_config.json
results/local/runtime/<RUN_ID>/.../<method>/<method>_summary.json
results/local/runtime/<RUN_ID>/queue/node0.complete
results/local/runtime/<RUN_ID>/queue/node1.complete
results/local/summary/runtime_runs.csv
results/local/summary/runtime_aggregate.csv
```

## Reusability

The runtime accepts any Transformers-compatible MoE model supported by the
frozen model patches, but a new model requires model-specific RS and SCP
offline files. The physical topology JSON and cache vector are explicit inputs, so
the same workflow can be adapted to a different two-node cluster without using
simulated delays.
