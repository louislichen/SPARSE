# Reusable RoutingScout checkpoints

RS predicts the next layer's original router experts from the current layer's
hidden state. Offline profiling uses the native model, without a scheduling
policy. A frozen bank can therefore be shared by independent scheduling,
prefetch admission, replica routing and coupled scheduling controls. Every
control must use the same checkpoint hashes, requests, cache budget and placement.
Hardware-specific transfer/compute cost calibration remains separate from RS.

Reuse requires the same base-model weights, tokenizer, hidden features, expert
numbering and prediction horizon. A new base model or feature/target definition
requires another bank. Holding the architecture fixed, changing GPU, topology,
cache budget or scheduling policy does not itself require RS retraining.

## C4 pilot data

The official source is [allenai/c4, English](https://huggingface.co/datasets/allenai/c4/tree/main/en).
For a small reproducible first bank, download `en/c4-train.00000-of-01024.json.gz`
at revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. Its SHA256 is
`8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9`.
Keep data and checkpoints outside the source release. C4 is distributed under
ODC-By; retain the source attribution with derived data.

```bash
python scripts/data/prepare_rs_c4.py \
  --source /path/to/c4-train.00000-of-01024.json.gz \
  --source-revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 \
  --model-path /path/to/model --out-dir /path/to/c4-pools
```

The default selects 1,280 eligible documents, deduplicated by text, URL and the
first 544 model tokens, requiring at least 640 tokens. It takes the first eligible
documents in one shard, shuffles with seed 42, and allocates 1,024 to RS profiling,
128 to calibration and 128 to runtime evaluation. This is a pilot subset, not a
random sample of all C4. The output records source revision/hash, line indices,
document IDs, split membership and deterministic compressed-file hashes.

## Profile, train and freeze

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_next_layer_rs.py \
  --model-path /path/to/model --dataset /path/to/c4-pools/profile.jsonl.gz \
  --out-dir /path/to/profile-v1 --num-samples 1024 \
  --prompt-len 512 --decode-steps 32 --decode-token-source dataset \
  --profile-batch-size 4 --save-interval-records 2048 --device-map cuda:0 \
  --attn-implementation eager --local-files-only --seed 42
python scripts/validate_next_layer_rs_profile.py \
  --profile-base-dir /path/to/profile-v1 --check-load
CUDA_VISIBLE_DEVICES=0 python scripts/train_next_layer_rs.py \
  --profile-base-dir /path/to/profile-v1 --output-dir /path/to/rs-v1 \
  --lead-layers 1 --model-arch residual --hidden-dim 512 --num-hidden-layers 2 \
  --dropout .05 --batch-size 512 --epochs 40 --lr 8e-4 --weight-decay 1e-4 \
  --early-stop-patience 5 --eval-ks 1,2,4,8,12 --feature-mode hidden \
  --target-mode uniform_multihot --loss-mode ce --split-unit request --seed 42
python scripts/freeze_rs_bank.py --rs-dir /path/to/rs-v1 --model-path /path/to/model
```

Native profiling produces 32,768 token records per MoE layer in this setup.
Dataset continuation is teacher forced. Free-generation accuracy and longer
prompts are separate distribution checks before paper-scale serving claims.

The trainer defaults to request-level 80/10/10 splits. All decode steps from a
document stay together, and every layer uses the same split. With 1,024 profiling
requests, the groups contain 819/102/103 requests. Initialization seeds may vary
by layer; partition seeds do not. Early stopping and checkpoint selection use
validation loss, then the best checkpoint is evaluated once on the test split.
`--split-unit row` is available for legacy diagnostics, but should not be used
for a reusable experimental bank. Existing checkpoints/traces require a fresh
versioned output directory rather than an implicit overwrite.

Checkpoint payloads record source/target layers, feature/label metadata, split
hash and model/data provenance. `freeze_rs_bank.py` checks completeness, disjoint
requests, base-model fingerprints and strict serving-loader compatibility. It
also checks finite normalized probabilities and writes every checkpoint hash
to `bank_manifest.json`; it rejects subsequent changes to a frozen bank.
Model-weight identity must additionally be retained in the experiment's model
file hashes: the profiler records weight sizes/mtimes rather than hashing every
large weight file at startup.

Use the frozen directory as `--rs-dir` for each scheduling strategy. Preserve
the bank unchanged while tuning scheduling on the separate calibration pool;
report final policy results on the untouched runtime pool. Predictor recall
alone does not establish end-to-end acceleration.
