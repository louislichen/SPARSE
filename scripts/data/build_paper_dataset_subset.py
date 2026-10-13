#!/usr/bin/env python3
"""Build deterministic, normalized JSONL subsets for SPARSE experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("reading parquet requires pyarrow") from exc
        table = pq.read_table(path)
        for row in table.to_pylist():
            if isinstance(row, dict):
                yield row
        return

    if path.suffix == ".json" and not path.name.endswith(".jsonl"):
        with open_text(path) as handle:
            obj = json.load(handle)
        rows = obj if isinstance(obj, list) else obj.get("data", [])
        for row in rows:
            if isinstance(row, dict):
                yield row
        return

    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def source_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file()
                and (
                    item.suffix in {".json", ".jsonl", ".parquet"}
                    or item.name.endswith(".jsonl.gz")
                    or item.name.endswith(".json.gz")
                )
            )
        elif path.is_file():
            files.append(path)
    return sorted(set(files))


def message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = str(message.get("role", message.get("from", ""))).strip()
        prefix = f"{role}: " if role else ""
        parts.append(prefix + content.strip())
    return "\n".join(parts).strip()


def normalize_text(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "longbench":
        context = row.get("context", "")
        question = row.get("question", row.get("input", ""))
        choices = [
            row.get(f"choice_{label}")
            for label in ("A", "B", "C", "D")
            if isinstance(row.get(f"choice_{label}"), str)
        ]
        parts = [part.strip() for part in (context, question) if isinstance(part, str) and part.strip()]
        if choices:
            parts.append("\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)))
        return "\n\n".join(parts).strip()

    for key in ("text", "prompt", "input", "query", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("messages", "conversation", "conversations"):
        text = message_text(row.get(key))
        if text:
            return text
    return ""


def load_tokenizer(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("--tokenizer requires transformers") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def token_count(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def stable_id(dataset: str, source: Path, index: int) -> str:
    raw = f"{dataset}\0{source.as_posix()}\0{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def longbench_stratum(source: Path, row: dict[str, Any]) -> str:
    for key in ("task", "task_name", "dataset_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    name = source.name
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".json", ".parquet"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if re.fullmatch(r"(?:train|test|validation)-\d+-of-\d+", name):
        parent = source.parent.name.strip()
        if parent and parent not in {"data", "default"}:
            return parent
    return name or "unknown"


def stratified_sample(
    strata: dict[str, list[dict[str, Any]]],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered: dict[str, list[dict[str, Any]]] = {}
    for key, rows in strata.items():
        ordered[key] = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['request_id']}".encode("utf-8")
            ).hexdigest(),
        )

    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in ordered}
    keys = sorted(ordered)
    while len(selected) < num_samples:
        made_progress = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(ordered[key]):
                continue
            selected.append(ordered[key][offset])
            offsets[key] = offset + 1
            made_progress = True
            if len(selected) >= num_samples:
                break
        if not made_progress:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("c4", "lmsys_chat_1m", "longbench"), required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eligible", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("reservoir", "sequential", "stratified"),
        default="reservoir",
    )
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--language", default="")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.selection == "stratified" and args.dataset != "longbench":
        raise SystemExit("--selection stratified is currently defined only for longbench")
    files = source_files(args.source)
    if not files:
        raise SystemExit("no supported source files found")

    tokenizer = load_tokenizer(args.tokenizer)
    rng = random.Random(args.seed)
    reservoir: list[dict[str, Any]] = []
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible = 0
    skipped = 0
    scanned = 0

    done = False
    for source in files:
        for index, row in enumerate(iter_json_rows(source)):
            scanned += 1
            if args.language:
                language = str(row.get("language", row.get("lang", ""))).lower()
                if language and args.language.lower() not in language:
                    continue
            text = normalize_text(args.dataset, row)
            if len(text) < args.min_chars:
                continue
            if args.max_chars > 0 and len(text) > args.max_chars:
                text = text[: args.max_chars]
            tokens = token_count(text, tokenizer)
            if args.min_tokens > 0 and tokens < args.min_tokens:
                continue
            if args.max_tokens > 0 and tokens > args.max_tokens:
                continue
            if skipped < args.skip_eligible:
                skipped += 1
                continue

            normalized = {
                "request_id": stable_id(args.dataset, source, index),
                "dataset": args.dataset,
                "text": text,
                "source_file": source.name,
                "source_index": index,
            }
            for key in ("domain", "sub_domain", "difficulty", "length"):
                if key in row and isinstance(row[key], (str, int, float, bool)):
                    normalized[key] = row[key]

            eligible += 1
            if args.selection == "stratified":
                stratum = longbench_stratum(source, row)
                normalized["stratum"] = stratum
                strata[stratum].append(normalized)
            elif args.selection == "sequential":
                reservoir.append(normalized)
                if len(reservoir) >= args.num_samples:
                    done = True
                    break
            elif len(reservoir) < args.num_samples:
                reservoir.append(normalized)
            else:
                replacement = rng.randrange(eligible)
                if replacement < args.num_samples:
                    reservoir[replacement] = normalized
        if done:
            break

    if args.selection == "stratified":
        reservoir = stratified_sample(strata, args.num_samples, args.seed)
    if len(reservoir) < args.num_samples:
        raise SystemExit(
            f"only {len(reservoir)} eligible rows found; requested {args.num_samples}"
        )
    reservoir.sort(key=lambda row: row["request_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(args.output).endswith(".gz") else open
    with opener(args.output, "wt", encoding="utf-8") as handle:
        for row in reservoir:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "sparse_normalized_dataset_subset_v1",
        "dataset": args.dataset,
        "output": str(args.output),
        "num_samples": len(reservoir),
        "seed": args.seed,
        "selection": args.selection,
        "skip_eligible": args.skip_eligible,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "tokenizer": args.tokenizer,
        "language": args.language,
        "source_files": [str(path) for path in files],
        "scanned_rows": scanned,
        "eligible_rows": eligible,
        "selected_stratum_counts": (
            dict(sorted(Counter(row["stratum"] for row in reservoir).items()))
            if args.selection == "stratified"
            else {}
        ),
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[dataset-subset] wrote {args.output}")
    print(f"[dataset-subset] wrote {manifest_path}")


if __name__ == "__main__":
    main()
