# SPARSE

This repository contains the implementation and compact evaluation material for
SPARSE, a physical-readiness-aware Mixture-of-Experts inference system. SPARSE
combines co-activation-aware expert placement (SCP), next-layer expert
prediction and prefetching (RS), and readiness-aware residual scheduling
(EPIPE) without changing router-selected experts or combine weights.

The original implementation remains at tag `sparse-independent-v1`, branch
`baseline/independent-scheduling` and `main` (`175da8d`). This
`research/lightweight-coupling` branch starts from that version and adds optional
local readiness decisions within its existing asynchronous runtime. It does not
use the synchronous global planner from the separate joint research branches.
See [lightweight coupling](docs/LIGHTWEIGHT_COUPLING.md) for scope and validation.

The release is intentionally whitelist-based. It includes the code paths used
by the paper, deterministic launchers, offline-file preparation scripts, compact source
tables, and fixed-canvas plotting code. It does not include model weights,
datasets, RS checkpoints, SCP profiles, raw repeats, server logs, or private
cluster configuration.

## SPARSE Evaluation Paths

The public runtime includes SPARSE, its component ablations, and the shared
round-robin reference needed for those controls. Comparison-system implementations
are kept in a separate local experiment workspace and are not part of this release.

| CLI name | Paper meaning |
|---|---|
| `rr` | Round-robin ownership with reactive LRU caching |
| `scp` | SCP only |
| `rs` | RS only |
| `epipe` | EPIPE execution path only |
| `scp-rs` | SCP and RS |
| `rs-epipe` | RS and EPIPE |
| `sparse` | Full SPARSE |

`executor-no-epipe` and `executor-with-epipe` reproduce the executor-matched
EPIPE control. Deprecated routing-repair, sparse-attention, simulated weak-link,
and simulated-heterogeneity experiments are not exposed by the public
launchers and are forced off in every formal command.

## Repository Layout

```text
configs/                 Compact physical-topology inputs
docs/                    Reproduction and code-review guides
results/reference/       Compact source data behind paper tables and figures
scripts/perf.py          Distributed serving benchmark entry point
scripts/run/              Two-node runtime and offline preparation workflows
scripts/analysis/         Compact result aggregation
scripts/figures/          Fixed-canvas paper plot generation
tests/                    CPU-only deterministic tests
utils/                    Frozen serving, cache, predictor, and executor core
```

## 1. Validate the Release

The release validator checks shell/Python syntax, deterministic data selection,
SCP helper behavior, launcher dry runs, private infrastructure strings, large
files, and accidentally committed model, data, or log files.

```bash
bash scripts/validate_release.sh
```

If `python3` is not the prepared environment interpreter, set it explicitly,
for example `PYTHON_BIN="$CONDA_PREFIX/bin/python" bash scripts/validate_release.sh`.

## 2. Create the Runtime Environment

The tested software stack uses Python 3.10, PyTorch 2.2.0 with CUDA 12.1,
Transformers 4.44.2, and NumPy 1.26.4.

```bash
bash scripts/setup_env.sh sparse-aaai
conda activate sparse-aaai
cp .env.example .env
```

Edit `.env` independently on both nodes. Node-local model, data, Python, and
result paths may differ. `MASTER_ADDR`, `MASTER_PORT`, the experiment `RUN_ID`,
and the method/workload order must match. Set `NODE_RANK=0` on the first node
and `NODE_RANK=1` on the second.

```bash
python scripts/check_env.py --env-file .env
```

Use `--strict` after the SCP placement and RS checkpoints have been installed.

## 3. Prepare Models and Data

Download Qwen1.5-MoE-A2.7B or DeepSeek-V2-Lite-Chat from their official model
repositories and point `MODEL_PATH` at the local snapshot. No model file is
redistributed here.

The deterministic subset builder accepts C4, LongBench, and LMSYS-style JSON or
JSONL sources. For example:

```bash
python scripts/data/build_paper_dataset_subset.py \
  --dataset c4 \
  --source /path/to/c4/source \
  --output /path/to/c4_runtime_eval.jsonl.gz \
  --num-samples 1024 \
  --seed 42 \
  --selection reservoir
```

Use disjoint source ranges or `--skip-eligible` values for profile-train,
profile-validation, and runtime subsets. The paper never trains or solves an
offline placement or predictor on runtime requests.

## 4. Install or Regenerate Offline Files

For a runtime-only reproduction, place the model-specific frozen RS checkpoint
directory and SCP placement JSON at `RS_DIR` and `SCP_PATH` on both nodes. This
is the recommended reproduction path.

SCP regeneration uses the reconstructed `scripts/scp_cache_replay.py` helper.
It checks grouped communication and reactive LRU loading proxies; these estimates
must still be validated with timed serving. See [reproduction details](docs/REPRODUCIBILITY.md)
for its limits and the metric and training corrections.

To regenerate RS:

```bash
bash scripts/run/profile_rs.sh
bash scripts/run/train_rs.sh
```

To regenerate the distributed SCP profile, run the same command on both nodes.
The node-0 process gathers rank files from node 1 before merging:

```bash
RUN_COMMAND='bash scripts/run/profile_scp.sh all' \
  RUN_ID=scp-profile-v1 \
  bash scripts/run/launch_two_node.sh

bash scripts/run/solve_scp.sh
```

The full paper profile uses 256 train shards and 64 validation shards. For a
functional check, set `SCP_PROFILE_TRAIN_SHARDS=4` and
`SCP_PROFILE_VALID_SHARDS=2` on both nodes.

## 5. Configure the Physical Link

Run the selected action on both nodes. `limit` uses Linux `tc` and requires
sudo; `native` removes the root shaper.

```bash
bash scripts/run/configure_network.sh native
RATE_MBIT=400 bash scripts/run/configure_network.sh limit
bash scripts/run/configure_network.sh show
```

The formal launchers never inject simulated communication or compute delays.

## 6. Run the Two-Node Matrix

The default matrix evaluates full SPARSE over three request shapes and four
non-overlapping input groups. Each configuration
uses one warm-up and three measured repeats.

```bash
RUN_ID=qwen-physical-400mbps-main-r3 \
  bash scripts/run/launch_two_node.sh
```

For the component ablation:

```bash
METHODS='rr scp rs epipe scp-rs rs-epipe sparse' \
RUN_ID=qwen-physical-400mbps-ablation-r3 \
  bash scripts/run/launch_two_node.sh
```

For the executor-matched EPIPE control:

```bash
METHODS='executor-no-epipe executor-with-epipe' \
WORKLOADS='b4' \
RUN_ID=qwen-physical-400mbps-executor-control-r3 \
  bash scripts/run/launch_two_node.sh
```

Progress is visible in tmux and under:

```text
results/local/runtime/<RUN_ID>/queue/node0.log
results/local/runtime/<RUN_ID>/queue/node1.log
```

A node is complete when its queue directory contains `node0.complete` or
`node1.complete`. Do not independently skip cases on one node: both distributed
ranks must execute the same matrix in the same order.

## 7. Aggregate Results and Regenerate Figures

```bash
RESULTS_ROOT=$PWD/results/local \
  bash scripts/run/reproduce.sh summarize

bash scripts/run/reproduce.sh reference-figures
```

The aggregator first averages repeats already stored in each benchmark summary,
then reports the mean and sample standard deviation across non-overlapping input
groups. The four generated Qwen plots share exactly the same canvas size and
retain white padding rather than changing aspect ratio.

## Result Semantics

- End-to-end throughput is generated tokens divided by total prefill plus
  decode time.
- Decode throughput excludes the first generated token per request and divides
  the remaining generated tokens by decode time.
- Gains are computed only against a result with the same model, requests,
  network, cache budget, and token accounting.
- The router remains authoritative. RS changes cache preparation; SCP changes
  feasible expert location; EPIPE changes only post-routing execution order.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the frozen protocol
and [docs/RUNNING_THE_CODE.md](docs/RUNNING_THE_CODE.md) for expected
reviewer workflows and outputs. [docs/CODE_SCOPE.md](docs/CODE_SCOPE.md)
documents the whitelist and the small compatibility boundary retained by the
frozen runtime core.
