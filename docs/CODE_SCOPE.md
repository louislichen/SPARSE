# Public Code Scope

The public repository is built from an explicit whitelist rather than by cleaning the
full research workspace in place. Public launchers expose only the methods
listed in the README and force simulated networking, simulated hardware
heterogeneity, routing repair, and sparse-attention experiments off.

`utils/eval_core.py` is the frozen serving core used for the reported physical
experiments. Two compatibility modules remain importable because the frozen
core resolves their symbols at module load time. They are not reachable through
the public launchers. Removing those imports without repeating the full GPU
regression matrix would produce a cleaner-looking but behaviorally different
release, so the code favors exact implementation fidelity over an unsafe
post-hoc rewrite.

The following research-workspace material is deliberately absent:

- legacy launchers and parameter sweeps;
- logical weak-network and simulated heterogeneity drivers;
- vLLM/Ray experiments and old system-baseline wrappers;
- routing-repair and NodePatch experiment drivers;
- SKVS experiment drivers;
- model weights, datasets, checkpoints, profiles, traces, and raw logs;
- machine-specific YAML files, SSH aliases, IP addresses, usernames, and paths.

`scripts/validate_release.sh` checks shell syntax, Python compilation, available
CPU tests, private infrastructure strings, and accidentally included binary
files. It does not verify CUDA execution or the complete offline pipeline.
Training tests require PyTorch and otherwise report skips. The missing SCP
cache-replay dependency is documented in `REPRODUCIBILITY.md`.

`utils/promoe.py` and the `*_promoe.sh` workflows are a new, independent
ProMoE-style baseline. The serving core has default-off integration hooks;
these do not imply new historical experimental results. Its mechanism mapping,
CPU validation, and remaining CUDA/fidelity checks are documented in
[`PROMOE_BASELINE.md`](PROMOE_BASELINE.md).
