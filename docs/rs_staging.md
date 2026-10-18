# Bounded next-layer RS staging

`utils.rs_overlap.BoundedRS` is an opt-in serving adapter. It keeps speculative
weights in a fixed pool, separate from each layer's resident demand cache. The
existing expert arithmetic, router and expert-copy event wait are reused.

The source layer submits prediction from its attention/residual/post-norm MoE
input. The source MoE and the following attention continue. At entry to the
target MoE, the adapter closes publication and exposes any already-issued
payloads to its native cache lookup and EPIPE classification. Only a real access
waits on its matching copy event. Prediction that has not submitted a copy by
this deadline is discarded; it cannot submit an unseen duplicate later.

Speculation never evicts, pins or reorders a resident expert. After the target
forward has submitted all its expert consumers, successful staged accesses are
admitted in their actual access order through the native demand-cache policy.
The evicted resident allocation becomes the next scratch payload; no additional
weight transfer is needed. A free cache entry needs a replacement allocation.
Pool reuse waits for the consumer event, which also protects the evicted
payload's earlier uses. An unused slot waits for its copy to finish. This avoids
skipping future demand-cache hits merely because a weight arrived by prefetch.
Admission is delayed until layer exit, so exact baseline LRU order is not claimed.

Use a fresh service version and install before constructing measured requests:

```python
pipeline = BoundedRS(eval_core, slots=4, empty_steps=16)
pipeline.install()
# After model / CPU-offload weights / RS bank have loaded, outside request timing:
pipeline.prepare_graphs(batch=batch, dtype=torch.float16, k=8)

# After the ordinary strict request reset, inside the service wall timer:
pipeline.enabled = use_rs
pipeline.begin(request_id)
try:
    outputs = original_greedy_decode(...)
finally:
    statistics = pipeline.finish()

# After the last request has drained:
pipeline.restore()
```

Reserve the same pool for ON and OFF controls. Keep `begin` and `finish` inside
end-to-end timing; `finish` drains remaining speculative work. Prepare every
supported batch/top-k before requests. Pair this adapter with batched accuracy
statistics when the service's old `_record_prefetch_eval_actual` hook reads GPU
indices on every layer. Statistics must also finish before the wall timer closes.

Current constraints are explicit: hidden-only adjacent-layer RS, original
router execution, pinned CPU expert weights of a uniform shape/dtype,
layer-local LRU, and fixed RR ownership during decode. SCP prefill is allowed;
decode SCP, a global cache, transition prefetch and MoE-Infinity are rejected.

Copy publication uses a brief lock around preallocated asynchronous submission.
It never waits for predictor completion or GPU completion. This lock, Python
submission, CUDA execution, shared memory bandwidth and copy engines still have
costs. No worst-case zero-overhead or positive end-to-end gain is assumed.

Run the CPU ownership/native-demand regressions with:

```sh
python3 -m unittest tests.test_rs_staging
```

Private deployment experiments and weights are intentionally outside this repo.
