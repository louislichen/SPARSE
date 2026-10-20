# Public Code Scope

The public repository is built from an explicit whitelist rather than by cleaning the
full research workspace in place. Public launchers expose only the methods
listed in the README and force simulated networking, simulated hardware
heterogeneity, routing repair, and sparse-attention experiments off.

The repository contains SPARSE implementation code and its evaluation utilities.
Comparison-system implementations, predictor checkpoints, training entry points,
placement solvers, and dedicated runtime policies for ProMoE-style, EPLB-like,
Prism/DanceMoE-style, static-hot, and MoE-Infinity-style experiments are kept in
a separate local workspace. The shared EP/LRU execution path and SPARSE component
ablations remain available. Reference tables and plotting code are SPARSE
evaluation artifacts, not comparison-system implementations.

`utils/eval_core.py` contains the SPARSE serving core used for the reported physical
experiments. The release promotes `sparse-verified-20261019-v1` (`e768424`)
without changing serving, training, solver or launcher code. The later global
SCP/EPIPE coordination implementation is excluded; its experimental branches
and private measurements are not merged into this release.
Two compatibility modules remain importable because the frozen
core resolves their symbols at module load time. They are not reachable through
the public launchers. Removing those imports without repeating the full GPU
regression matrix would produce a cleaner-looking but behaviorally different
release, so the code favors exact implementation fidelity over an unsafe
post-hoc rewrite.

The following research-workspace material is deliberately absent:

- legacy launchers and parameter sweeps;
- logical weak-network and simulated heterogeneity drivers;
- comparison-system implementations and launchers;
- vLLM/Ray experiments and old system-baseline wrappers;
- routing-repair and NodePatch experiment drivers;
- SKVS experiment drivers;
- model weights, datasets, checkpoints, profiles, traces, and raw logs;
- machine-specific YAML files, SSH aliases, IP addresses, usernames, and paths.

`scripts/validate_release.sh` checks shell syntax, Python compilation, available
CPU tests, private infrastructure strings, and accidentally included binary
files. It does not verify CUDA execution or the complete offline pipeline.
Training tests require PyTorch and otherwise report skips. The reconstructed
SCP replay's proxy semantics and limitations are documented in `REPRODUCIBILITY.md`.
