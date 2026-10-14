"""Validate a complete trained RS bank through the serving loader and freeze hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from utils.eval_core import _load_runtime_rs_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rs-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads((args.model_path / "config.json").read_text())
    training = json.loads((args.rs_dir / "train_config.json").read_text())
    profile = training["manifest"]
    if training["split_unit"] != "request":
        raise ValueError("a reusable bank must use disjoint request splits")
    split = json.loads((args.rs_dir / "split.json").read_text())
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    groups = [set(split["request_ids"][name]) for name in ("train", "validation", "test")]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("request leakage between train, validation and test")
    provenance = profile["provenance"]
    for name, expected in provenance["model_files"].items():
        if hashlib.sha256((args.model_path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"base-model fingerprint mismatch: {name}")
    layers = set(profile["moe_layer_indices"])
    expected = {(source, source + 1) for source in layers if source + 1 in layers}
    paths = sorted(args.rs_dir.glob("rs_l*_inL*.pth"))
    found, hashes = set(), {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        pair = (payload["source_layer"], payload["target_layer"])
        if pair in found or pair not in expected:
            raise ValueError(f"unexpected or duplicate layer pair: {pair}")
        found.add(pair)
        if (payload["input_dim"] != config["hidden_size"] or
                payload["num_experts"] != profile["num_experts_per_layer"] or
                payload["feature_mode"] != "hidden" or payload["lead_layers"] != 1):
            raise ValueError(f"checkpoint feature/target mismatch: {path.name}")
        if (payload["split_sha256"] != split_hash or
                payload["profile_manifest_sha256"] != training["profile_manifest_sha256"]):
            raise ValueError(f"checkpoint provenance mismatch: {path.name}")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != expected:
        raise ValueError(f"missing trained pairs: {sorted(expected - found)}")
    device = torch.device(args.device)
    bank, task, layout, sources, missing = _load_runtime_rs_bank(
        rs_path=args.rs_dir, num_layers=config["num_hidden_layers"],
        hidden_dim=config["hidden_size"], num_experts=profile["num_experts_per_layer"],
        device=device, require_complete=True)
    if missing or task != "router_prob" or layout != "next_layer":
        raise ValueError("serving loader did not resolve a complete next-layer probability bank")
    with torch.inference_mode():
        for source, target in sorted(expected):
            if sources[target] != source:
                raise ValueError(f"runtime source mapping mismatch: L{target}")
            probabilities = bank[target](torch.zeros(2, config["hidden_size"], device=device))
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError(f"invalid probabilities: L{target}")
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, device=device))
    manifest = dict(schema_version=1, task=task, layout=layout, feature_mode="hidden", lead_layers=1,
                    trained_pairs=sorted(expected), checkpoint_sha256=hashes,
                    split_sha256=split_hash, model_files=provenance["model_files"],
                    profile_manifest_sha256=training["profile_manifest_sha256"])
    output = args.rs_dir / "bank_manifest.json"
    if output.exists() and json.loads(output.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("frozen bank changed; use a new versioned directory")
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(status="PASS", trained_predictors=len(expected), manifest=str(output))))


if __name__ == "__main__":
    main()
