# Original SPARSE: b1 and b2 system measurements

SPARSE only, four input groups for each workload, one eight-token warmup and three full measured repeats. All96 SPARSE rank outputs and64 warmup prefixes strictly match32 current own-rank RR references. RR is executed once per group only for correctness; other baselines and b4 are not remeasured.

**Code relationship:** Published e768424 original architecture through existing later frozen private baseline/managed-RS harness; not an identical public checkout benchmark.

|Workload|E2E output tok/s (mean +/- group SD)|Service decode tok/s|Total H2D GiB|
|---|---:|---:|---:|
|b1|2.4331 +/- 0.0253|5.7831|50.2117|
|b2|6.1615 +/- 0.2828|9.6710|99.2659|

`summary.csv`: two workload summaries. `groups.csv`: eight group means. `repeats.csv`:24 measured SPARSE requests. All measured results retained; no runtime selection. Service decode includes request drain/statistics (total minus prefill for each rank, then max). H2D includes demand plus prefetch, summed across four ranks. Duplicate EP inputs count once, not four times. Activation payload is not NIC wire bytes.

Resources are two RTX4090 and two RTX3090 GPUs, with different exact devices from the earlier b4 sample. Do not pool timings or present the old b4 and these rows as a unified-resource baseline comparison. Existing scoped b4 and historical baseline evidence is in [verified_original_20261020](../verified_original_20261020/README.md). These rows do not establish a new repeated baseline speedup. SCP420 is a replicated CPU placement, not strict zero-replica SCP; shared inventory is561 extra CPU copies, not GPU residency. Native Hugging Face numerical equivalence is a separate question. Heterogeneous WHU throughput is not an absolute reproduction of the old2080Ti platform.

Raw document/token arrays, model weights, RS checkpoints, logs, host paths and UUIDs remain private. Source code is unchanged by this publication.
