"""Prepare disjoint C4 document pools for RS training and policy evaluation.

This pilot selects the first eligible documents from a pinned English C4 shard,
then shuffles them with a fixed seed. It does not represent a full-C4 sample.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile-requests", type=int, default=1024)
    parser.add_argument("--calibration-requests", type=int, default=128)
    parser.add_argument("--runtime-requests", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=640)
    parser.add_argument("--prefix-tokens", type=int, default=544)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = dict(profile=args.profile_requests, calibration=args.calibration_requests,
                  runtime=args.runtime_requests)
    if min(*counts.values(), args.min_tokens, args.prefix_tokens) < 1:
        parser.error("counts and token lengths must be positive")
    if args.prefix_tokens > args.min_tokens:
        parser.error("prefix length must not exceed minimum document length")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("use a fresh output directory")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                              trust_remote_code=True)
    pool, documents, prefixes, urls = [], set(), set(), set()
    scanned = 0
    with gzip.open(args.source, "rt", encoding="utf-8") as handle:
        for scanned, line in enumerate(handle, 1):
            row = json.loads(line)
            text = row.get("text", "").strip()
            if not text:
                continue
            document = hashlib.sha256(text.encode()).hexdigest()
            url = row.get("url", "")
            if document in documents or (url and url in urls):
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < args.min_tokens:
                continue
            prefix = hashlib.sha256(json.dumps(tokens[:args.prefix_tokens]).encode()).hexdigest()
            if prefix in prefixes:
                continue
            documents.add(document)
            prefixes.add(prefix)
            if url:
                urls.add(url)
            pool.append(dict(request_id=document, text=text, source_index=scanned - 1,
                             source="allenai/c4:en", prefix_sha256=prefix,
                             model_tokens=len(tokens)))
            if len(pool) % 256 == 0:
                print(f"eligible={len(pool)} scanned={scanned}", flush=True)
            if len(pool) == sum(counts.values()):
                break
    if len(pool) != sum(counts.values()):
        raise ValueError(f"only {len(pool)} eligible unique documents in source shard")
    random.Random(args.seed).shuffle(pool)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subsets, hashes, offset = {}, {}, 0
    for name, count in counts.items():
        rows = pool[offset:offset + count]
        offset += count
        subsets[name] = rows
        path = args.out_dir / f"{name}.jsonl.gz"
        # Fix gzip header time/name so identical inputs produce identical files.
        with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            for row in rows:
                handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
        hashes[name] = file_sha256(path)
    manifest = dict(dataset="allenai/c4", config="en", source=str(args.source),
                    source_revision=args.source_revision, source_sha256=file_sha256(args.source),
                    selection="first eligible unique documents, shuffled by seed; pilot subset",
                    model=args.model_path, seed=args.seed, scanned=scanned, counts=counts,
                    sha256=hashes, min_model_tokens=args.min_tokens,
                    prefix_dedup_tokens=args.prefix_tokens,
                    deduplication=["document text", "nonempty URL", "token prefix"],
                    split_ids={name: [row["request_id"] for row in rows]
                               for name, rows in subsets.items()})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ("dataset", "scanned", "counts", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
