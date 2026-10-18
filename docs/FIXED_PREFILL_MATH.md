# Optional fixed-tile prefill expert arithmetic

Set `SPARSE_PREFILL_EXPERT_MATH=fixed64` on every rank to use a fixed reduction
tile for the prefill expert MLP. The default `native` path retains the existing
PyTorch implementation. Decode retains its existing implementation in both modes.
The option applies equally to demand, staged EPIPE and cached expert payloads.

Changing placement can change an expert GEMM's row count, causing native FP16
rounding differences that propagate through later routing decisions. The optional
Triton kernel keeps its reduction loop, tile dimensions and FP16 intermediate
boundaries fixed when rows are regrouped. It requires CUDA FP16 contiguous
two-dimensional inputs/weights and a matching Triton installation. Unsupported
inputs or missing dependencies fail explicitly; there is no silent native fallback.

This defines a separate numerical version. Full greedy outputs can differ from
native PyTorch/Hugging Face outputs. Use the same setting for every comparator,
check full inputs and generated tokens against the same-rank same-version RR
control, and keep native model quality checks separate. Do not merge measurements
from the two numerical versions or infer unchanged quality from RR parity.

The benchmark records `prefill_expert_math` in raw repeat records and summaries.
Existing numerical-version measurement artifacts remain immutable; this source
integration is a new version and is not substituted into running experiments.
Private placements, hardware information, calibration data and baseline policies
are not part of this module.
