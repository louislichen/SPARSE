# Reference Result Data

These CSV files are compact, audited sources for the paper's figures and tables.
They contain aggregate throughput or mechanism counters only. Raw repeats,
distributed logs, routing traces, model paths, dataset paths, and host metadata
are intentionally excluded.

- `source_policy_normalized.csv`: normalized Qwen policy comparison.
- `source_policy_normalized_slices.csv`: per-input-group normalized values.
- `source_main_absolute_throughput.csv`: homogeneous absolute throughput.
- `source_qwen_tc400_b4_ablation.csv`: component ablation.
- `source_mechanism_evidence.csv`: cache, H2D, prefetch, and route-cost evidence.
- `source_heldout_absolute_throughput.csv`: held-out LongBench systems workload.
- `source_output_length_sensitivity.csv`: generated-length sensitivity.
- `source_heterogeneous_boundary.csv`: compact heterogeneous deployment result.

The formal aggregation order is: average measured repeats for each fixed input
group, then average across non-overlapping groups. Standard deviations in the
main plots are across input groups.
