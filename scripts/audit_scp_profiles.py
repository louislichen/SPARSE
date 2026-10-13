#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_KEYS = {
    "A_prefill",
    "A_decode",
    "trace_stage",
    "trace_source_rank",
    "trace_expert",
    "trace_count",
    "metadata_json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_manifest(path: Path, expected_rows: int) -> tuple[dict[str, Any], set[str]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    request_ids: list[str] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = row.get("request_id")
            if request_id is None or str(request_id).strip() == "":
                raise ValueError(f"{path}:{line_number} has no request_id")
            request_ids.append(str(request_id))
    request_id_set = set(request_ids)
    if len(request_id_set) != len(request_ids):
        raise ValueError(f"{path} contains duplicate request_id values")
    if expected_rows > 0 and len(request_ids) != expected_rows:
        raise ValueError(f"{path} rows={len(request_ids)} expected={expected_rows}")
    metadata = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(request_ids),
        "unique_request_ids": len(request_id_set),
        "request_id_fingerprint": json_fingerprint(sorted(request_id_set)),
    }
    return metadata, request_id_set


def load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = data.get("metadata_json")
    if raw is None:
        return {}
    return json.loads(str(raw.item()))


def model_manifest(model_path: Path, hash_weights: bool) -> dict[str, Any]:
    metadata_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    metadata_files = []
    for name in metadata_names:
        path = model_path / name
        if path.is_file():
            metadata_files.append(
                {
                    "name": name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    weight_paths = sorted(
        {
            path
            for pattern in ("*.safetensors", "*.bin")
            for path in model_path.glob(pattern)
            if path.is_file()
        }
    )
    weights = []
    for path in weight_paths:
        entry = {
            "name": path.name,
            "size": path.stat().st_size,
        }
        if hash_weights:
            entry["sha256"] = sha256_file(path)
        weights.append(entry)

    if not metadata_files:
        raise FileNotFoundError(f"no model metadata files found under {model_path}")
    if not weights:
        raise FileNotFoundError(f"no model weight files found under {model_path}")

    content = {
        "metadata_files": metadata_files,
        "weights": weights,
        "weights_fully_hashed": bool(hash_weights),
    }
    return {
        "path": str(model_path.resolve()),
        **content,
        "content_fingerprint": json_fingerprint(content),
    }


def record_node(args: argparse.Namespace) -> None:
    train = Path(args.train_dataset).expanduser()
    valid = Path(args.valid_dataset).expanduser()
    model = Path(args.model_path).expanduser()
    for label, path in (("train dataset", train), ("valid dataset", valid), ("model", model)):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")

    train_metadata, train_request_ids = dataset_manifest(train, args.expected_train_rows)
    valid_metadata, valid_request_ids = dataset_manifest(valid, args.expected_valid_rows)
    datasets = {"train": train_metadata, "valid": valid_metadata}
    if datasets["train"]["sha256"] == datasets["valid"]["sha256"]:
        raise ValueError("training and validation dataset files are identical")
    overlap = train_request_ids.intersection(valid_request_ids)
    if overlap:
        raise ValueError(f"training and validation request IDs overlap: {sorted(overlap)[:8]}")

    protocol = {
        "world_size": args.world_size,
        "nnodes": args.nnodes,
        "nproc_per_node": args.nproc_per_node,
        "ep_size": args.ep_size,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_new": args.max_new,
        "stage": args.stage,
        "train_shards": args.train_shards,
        "valid_shards": args.valid_shards,
        "seed": args.seed,
        "simulated_weaknet": False,
    }
    payload = {
        "schema": "sparse_scp_profile_node_provenance_v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "host_key": args.host_key,
        "node_rank": args.node_rank,
        "datasets": datasets,
        "model": model_manifest(model, args.hash_model_weights),
        "protocol": protocol,
        "collection_invocation": {
            "train_shard_start": args.train_shard_start,
            "valid_shard_start": args.valid_shard_start,
        },
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[scp-audit] wrote node provenance: {output}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_node_provenance(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 2:
        raise ValueError(f"expected exactly two node provenance files, got {len(paths)}")
    reports = [load_json(path) for path in paths]
    if {int(report["node_rank"]) for report in reports} != {0, 1}:
        raise ValueError("node provenance must contain node_rank 0 and 1")

    first, second = reports
    mismatches = []
    for split in ("train", "valid"):
        if first["datasets"][split]["sha256"] != second["datasets"][split]["sha256"]:
            mismatches.append(f"{split} dataset sha256")
        if (
            first["datasets"][split]["request_id_fingerprint"]
            != second["datasets"][split]["request_id_fingerprint"]
        ):
            mismatches.append(f"{split} request ID fingerprint")
    if first["model"]["content_fingerprint"] != second["model"]["content_fingerprint"]:
        mismatches.append("model content fingerprint")
    if first["protocol"] != second["protocol"]:
        mismatches.append("profile protocol")
    if mismatches:
        raise ValueError("node provenance mismatch: " + ", ".join(mismatches))
    if first["datasets"]["train"]["sha256"] == first["datasets"]["valid"]["sha256"]:
        raise ValueError("training and validation datasets are not independent files")

    return {
        "node_reports": [str(path) for path in paths],
        "train_dataset_sha256": first["datasets"]["train"]["sha256"],
        "valid_dataset_sha256": first["datasets"]["valid"]["sha256"],
        "model_content_fingerprint": first["model"]["content_fingerprint"],
        "protocol": first["protocol"],
    }


def validate_rank_profile(
    path: Path,
    expected_rank: int,
    expected_split: str,
    expected_batch: int,
    world_size: int,
    num_layers: int,
    num_experts: int,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty rank profile: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(PROFILE_KEYS.difference(data.files))
        if missing:
            raise ValueError(f"{path} misses keys: {missing}")
        prefill = np.asarray(data["A_prefill"], dtype=np.int64)
        decode = np.asarray(data["A_decode"], dtype=np.int64)
        expected_shape = (num_layers, world_size, num_experts)
        if prefill.shape != expected_shape or decode.shape != expected_shape:
            raise ValueError(
                f"{path} shape mismatch: prefill={prefill.shape} decode={decode.shape} "
                f"expected={expected_shape}"
            )
        metadata = load_metadata(data)
        if int(metadata.get("rank", -1)) != expected_rank:
            raise ValueError(f"{path} metadata rank={metadata.get('rank')} expected={expected_rank}")
        if int(metadata.get("world_size", -1)) != world_size:
            raise ValueError(f"{path} world_size={metadata.get('world_size')} expected={world_size}")
        if str(metadata.get("profile_split", "")) != expected_split:
            raise ValueError(
                f"{path} profile_split={metadata.get('profile_split')} expected={expected_split}"
            )
        requests = metadata.get("requests", [])
        if not isinstance(requests, list) or len(requests) != expected_batch:
            raise ValueError(f"{path} request metadata count={len(requests)} expected={expected_batch}")
        dataset_indices = tuple(int(request["dataset_index"]) for request in requests)
        if len(set(dataset_indices)) != expected_batch:
            raise ValueError(f"{path} contains duplicate dataset indices")
        stages = np.asarray(data["trace_stage"], dtype=np.int64)
        counts = np.asarray(data["trace_count"], dtype=np.int64)
        trace_prefill = int(counts[stages == 0].sum())
        trace_decode = int(counts[stages == 1].sum())
        if trace_prefill != int(prefill.sum()) or trace_decode != int(decode.sum()):
            raise ValueError(f"{path} ordered trace does not match aggregate arrays")
        return {
            "prefill_assignments": int(prefill.sum()),
            "decode_assignments": int(decode.sum()),
            "trace_events": int(counts.size),
            "dataset_indices": dataset_indices,
        }


def validate_split(
    root: Path,
    split: str,
    shard_count: int,
    batch: int,
    world_size: int,
    num_layers: int,
    num_experts: int,
) -> dict[str, int]:
    split_root = root / f"{split}_shards"
    expected_paths = {
        (split_root / f"shard_{shard}" / f"profile.rank{rank}.npz").resolve()
        for shard in range(shard_count)
        for rank in range(world_size)
    }
    actual_paths = {path.resolve() for path in split_root.glob("**/profile.rank*.npz")}
    missing_paths = sorted(str(path) for path in expected_paths.difference(actual_paths))
    extra_paths = sorted(str(path) for path in actual_paths.difference(expected_paths))
    if missing_paths or extra_paths:
        raise ValueError(
            f"{split} shard file set mismatch: missing={missing_paths[:8]} "
            f"extra={extra_paths[:8]}"
        )
    totals = {
        "rank_files": 0,
        "prefill_assignments": 0,
        "decode_assignments": 0,
        "trace_events": 0,
        "sampled_requests": 0,
    }
    sampled_dataset_indices: set[int] = set()
    for shard in range(shard_count):
        shard_dir = split_root / f"shard_{shard}"
        shard_dataset_indices: tuple[int, ...] | None = None
        for rank in range(world_size):
            stats = validate_rank_profile(
                shard_dir / f"profile.rank{rank}.npz",
                rank,
                split,
                batch,
                world_size,
                num_layers,
                num_experts,
            )
            current_indices = tuple(stats.pop("dataset_indices"))
            if shard_dataset_indices is None:
                shard_dataset_indices = current_indices
            elif current_indices != shard_dataset_indices:
                raise ValueError(
                    f"{split} shard {shard} sampled different dataset indices across ranks"
                )
            totals["rank_files"] += 1
            for key, value in stats.items():
                totals[key] += value
        assert shard_dataset_indices is not None
        overlap = sampled_dataset_indices.intersection(shard_dataset_indices)
        if overlap:
            raise ValueError(f"{split} profile repeats dataset indices: {sorted(overlap)}")
        sampled_dataset_indices.update(shard_dataset_indices)
    totals["sampled_requests"] = len(sampled_dataset_indices)
    expected_requests = shard_count * batch
    if totals["sampled_requests"] != expected_requests:
        raise ValueError(
            f"{split} sampled requests={totals['sampled_requests']} expected={expected_requests}"
        )
    return totals


def validate_shards(args: argparse.Namespace) -> None:
    provenance = compare_node_provenance([Path(path).expanduser() for path in args.node_provenance])
    root = Path(args.profile_root).expanduser()
    train = validate_split(
        root,
        "train",
        args.train_shards,
        args.batch,
        args.world_size,
        args.num_layers,
        args.num_experts,
    )
    valid = validate_split(
        root,
        "valid",
        args.valid_shards,
        args.batch,
        args.world_size,
        args.num_layers,
        args.num_experts,
    )
    report = {
        "schema": "sparse_scp_profile_shard_audit_v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "profile_root": str(root.resolve()),
        "provenance": provenance,
        "train": train,
        "valid": valid,
        "passed": True,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[scp-audit] shard audit passed: train_files={train['rank_files']} "
        f"valid_files={valid['rank_files']} report={output}"
    )


def validate_grouped_file(
    path: Path,
    expected_rank_files: int,
    stage: str,
    world_size: int,
    num_layers: int,
    num_experts: int,
) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(PROFILE_KEYS.difference(data.files))
        if missing:
            raise ValueError(f"{path} misses keys: {missing}")
        prefill = np.asarray(data["A_prefill"], dtype=np.int64)
        decode = np.asarray(data["A_decode"], dtype=np.int64)
        expected_shape = (num_layers, world_size, num_experts)
        if prefill.shape != expected_shape or decode.shape != expected_shape:
            raise ValueError(f"{path} grouped shape mismatch")
        stages = np.asarray(data["trace_stage"], dtype=np.int64)
        counts = np.asarray(data["trace_count"], dtype=np.int64)
        metadata = load_metadata(data)
        if int(metadata.get("rank_count", -1)) != expected_rank_files:
            raise ValueError(
                f"{path} merged rank_count={metadata.get('rank_count')} "
                f"expected={expected_rank_files}"
            )
        trace_prefill = int(counts[stages == 0].sum())
        trace_decode = int(counts[stages == 1].sum())
        if trace_prefill != int(prefill.sum()) or trace_decode != int(decode.sum()):
            raise ValueError(f"{path} grouped ordered trace does not match aggregate arrays")
        if stage == "prefill" and (trace_prefill <= 0 or trace_decode != 0):
            raise ValueError(
                f"{path} expected prefill-only trace, got prefill={trace_prefill} decode={trace_decode}"
            )
        if stage == "decode" and (trace_decode <= 0 or trace_prefill != 0):
            raise ValueError(
                f"{path} expected decode-only trace, got prefill={trace_prefill} decode={trace_decode}"
            )
        if stage == "both" and (trace_prefill <= 0 or trace_decode <= 0):
            raise ValueError(f"{path} contains no positive prefill/decode assignments")
        return {
            "rank_files": expected_rank_files,
            "prefill_assignments": trace_prefill,
            "decode_assignments": trace_decode,
            "trace_events": int(counts.size),
        }


def validate_grouped(args: argparse.Namespace) -> None:
    train = validate_grouped_file(
        Path(args.train_profile).expanduser(),
        args.train_shards * args.world_size,
        args.stage,
        args.world_size,
        args.num_layers,
        args.num_experts,
    )
    valid = validate_grouped_file(
        Path(args.valid_profile).expanduser(),
        args.valid_shards * args.world_size,
        args.stage,
        args.world_size,
        args.num_layers,
        args.num_experts,
    )
    report = {
        "schema": "sparse_scp_grouped_profile_audit_v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "train_profile": str(Path(args.train_profile).expanduser().resolve()),
        "valid_profile": str(Path(args.valid_profile).expanduser().resolve()),
        "train": train,
        "valid": valid,
        "passed": True,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[scp-audit] grouped profile audit passed: {output}")


def validate_topology(args: argparse.Namespace) -> None:
    path = Path(args.topology).expanduser()
    topology = load_json(path)
    if topology.get("schema") != "sparse_measured_dual_node_topology_v1":
        raise ValueError(f"unexpected topology schema: {topology.get('schema')}")
    if int(topology.get("world_size", -1)) != args.world_size:
        raise ValueError(f"topology world_size={topology.get('world_size')} expected={args.world_size}")
    rank_to_node = {str(key): int(value) for key, value in topology.get("rank_to_node", {}).items()}
    expected = {str(rank): int(node) for rank, node in enumerate(args.rank_to_node)}
    if rank_to_node != expected:
        raise ValueError(f"rank_to_node={rank_to_node} expected={expected}")
    model = topology.get("bandwidth_model", {})
    if not bool(model.get("enabled")) or not bool(model.get("measured")):
        raise ValueError("topology bandwidth model is not marked as measured")
    for key in (
        "same_node_bw_MBps",
        "cross_node_bw_MBps",
        "same_node_latency_ms",
        "cross_node_latency_ms",
    ):
        value = float(model.get(key, float("nan")))
        if not math.isfinite(value) or value < 0 or ("bw_" in key and value <= 0):
            raise ValueError(f"invalid topology value {key}={value}")
    measurement_source = Path(str(topology.get("measurement_source", ""))).expanduser()
    if not measurement_source.is_file():
        raise FileNotFoundError(f"missing topology measurement source: {measurement_source}")
    smoke = load_json(measurement_source)
    if smoke.get("schema") != "sparse_distributed_smoke_v1":
        raise ValueError(f"unexpected smoke schema: {smoke.get('schema')}")
    if int(smoke.get("world_size", -1)) != args.world_size:
        raise ValueError("smoke world size does not match topology")
    identities = smoke.get("rank_identities", [])
    if len(identities) != args.world_size:
        raise ValueError(f"smoke rank identities={len(identities)} expected={args.world_size}")
    host_counts: dict[str, int] = {}
    for identity in identities:
        hostname = str(identity.get("hostname", ""))
        host_counts[hostname] = host_counts.get(hostname, 0) + 1
    if sorted(host_counts.values()) != [2, 2]:
        raise ValueError(f"smoke is not a physical 2x2 deployment: {host_counts}")
    correctness = smoke.get("correctness", {})
    if not bool(correctness.get("all_reduce")) or not bool(correctness.get("broadcast")):
        raise ValueError("smoke collective correctness checks did not pass")
    if not smoke.get("pairwise"):
        raise ValueError("smoke report has no physical pairwise measurements")
    fits = smoke.get("fits", {})
    for topology_key, fit_group, fit_key in (
        ("same_node_bw_MBps", "same_node", "bandwidth_MB_s"),
        ("cross_node_bw_MBps", "cross_node", "bandwidth_MB_s"),
        ("same_node_latency_ms", "same_node", "latency_ms"),
        ("cross_node_latency_ms", "cross_node", "latency_ms"),
    ):
        topology_value = float(model[topology_key])
        smoke_value = float(fits.get(fit_group, {}).get(fit_key, float("nan")))
        if not math.isclose(topology_value, smoke_value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"topology {topology_key}={topology_value} does not match smoke fit={smoke_value}"
            )
    report = {
        "schema": "sparse_scp_topology_audit_v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "topology": str(path.resolve()),
        "topology_sha256": sha256_file(path),
        "measurement_source": str(measurement_source.resolve()),
        "measurement_source_sha256": sha256_file(measurement_source),
        "physical_hosts": host_counts,
        "world_size": args.world_size,
        "rank_to_node": expected,
        "bandwidth_model": model,
        "passed": True,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[scp-audit] physical topology audit passed: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit physical 2x2 SCP profile files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record-node")
    record.add_argument("--output", required=True)
    record.add_argument("--host-key", required=True)
    record.add_argument("--node-rank", type=int, required=True)
    record.add_argument("--train-dataset", required=True)
    record.add_argument("--valid-dataset", required=True)
    record.add_argument("--expected-train-rows", type=int, default=0)
    record.add_argument("--expected-valid-rows", type=int, default=0)
    record.add_argument("--model-path", required=True)
    record.add_argument("--world-size", type=int, required=True)
    record.add_argument("--nnodes", type=int, required=True)
    record.add_argument("--nproc-per-node", type=int, required=True)
    record.add_argument("--ep-size", type=int, required=True)
    record.add_argument("--batch", type=int, required=True)
    record.add_argument("--prompt-len", type=int, required=True)
    record.add_argument("--max-new", type=int, required=True)
    record.add_argument("--stage", choices=("prefill", "decode", "both"), required=True)
    record.add_argument("--train-shards", type=int, required=True)
    record.add_argument("--valid-shards", type=int, required=True)
    record.add_argument("--train-shard-start", type=int, default=0)
    record.add_argument("--valid-shard-start", type=int, default=0)
    record.add_argument("--seed", type=int, default=1)
    record.add_argument("--hash-model-weights", action="store_true")
    record.set_defaults(func=record_node)

    shards = subparsers.add_parser("validate-shards")
    shards.add_argument("--profile-root", required=True)
    shards.add_argument("--node-provenance", action="append", required=True)
    shards.add_argument("--output", required=True)
    shards.add_argument("--train-shards", type=int, required=True)
    shards.add_argument("--valid-shards", type=int, required=True)
    shards.add_argument("--batch", type=int, required=True)
    shards.add_argument("--world-size", type=int, required=True)
    shards.add_argument("--num-layers", type=int, required=True)
    shards.add_argument("--num-experts", type=int, required=True)
    shards.set_defaults(func=validate_shards)

    grouped = subparsers.add_parser("validate-grouped")
    grouped.add_argument("--train-profile", required=True)
    grouped.add_argument("--valid-profile", required=True)
    grouped.add_argument("--output", required=True)
    grouped.add_argument("--train-shards", type=int, required=True)
    grouped.add_argument("--valid-shards", type=int, required=True)
    grouped.add_argument("--stage", choices=("prefill", "decode", "both"), required=True)
    grouped.add_argument("--world-size", type=int, required=True)
    grouped.add_argument("--num-layers", type=int, required=True)
    grouped.add_argument("--num-experts", type=int, required=True)
    grouped.set_defaults(func=validate_grouped)

    topology = subparsers.add_parser("validate-topology")
    topology.add_argument("--topology", required=True)
    topology.add_argument("--output", required=True)
    topology.add_argument("--world-size", type=int, required=True)
    topology.add_argument("--rank-to-node", type=int, nargs="+", required=True)
    topology.set_defaults(func=validate_topology)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
