from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List

def _load_manifest(base: Path) -> Dict[str, object]:
    path = base / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dir(base: Path, raw: str, fallback_name: str) -> Path:
    p = Path(raw)
    if raw and not p.is_absolute() and (base / p).exists():
        return base / p
    if raw and p.exists():
        return p
    fallback = base / fallback_name
    if fallback.exists():
        return fallback
    return p if raw else fallback


def _sorted_files(pattern: str) -> List[Path]:
    files = [Path(x) for x in glob.glob(pattern)]

    def chunk_id(p: Path) -> int:
        m = re.search(r"_chunk_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    return sorted(files, key=chunk_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate next-layer RS profile chunks before full training.")
    ap.add_argument("--profile-base-dir", required=True)
    ap.add_argument("--check-load", action="store_true", help="Load every chunk with torch.load and check tensor shapes.")
    args = ap.parse_args()

    base = Path(args.profile_base_dir)
    manifest = _load_manifest(base)
    rs_dir = _resolve_dir(base, str(manifest.get("rs_chunk_dir", "")), "rs_decode_chunks")
    trace_dir = _resolve_dir(base, str(manifest.get("trace_chunk_dir", "")), "trace_chunks")
    summary_pt = _resolve_dir(base, str(manifest.get("summary_pt", "")), "summary/decode_profile_summary.pt")

    num_layers = int(manifest["num_layers"])
    num_experts = int(manifest["num_experts_per_layer"])
    top_k = int(manifest["top_k"])
    hidden_size = int(manifest["hidden_size"])
    total_records = int(manifest["total_decode_records"])

    print(f"[manifest] base={base}")
    print(f"[manifest] rs_dir={rs_dir} exists={rs_dir.exists()}")
    print(f"[manifest] trace_dir={trace_dir} exists={trace_dir.exists()}")
    print(f"[manifest] summary_pt={summary_pt} exists={summary_pt.exists()}")
    print(
        f"[manifest] layers={num_layers} experts={num_experts} top_k={top_k} "
        f"hidden={hidden_size} total_records={total_records}"
    )

    torch_mod = None
    if args.check_load:
        import torch as torch_mod

    errors: List[str] = []
    for layer in manifest.get("moe_layer_indices", range(num_layers)):
        input_files = _sorted_files(str(rs_dir / f"rs_inputs_l{layer}_chunk_*.pt"))
        label_files = _sorted_files(str(rs_dir / f"rs_labels_l{layer}_chunk_*.pt"))
        if len(input_files) == 0:
            errors.append(f"L{layer}: missing input chunks")
        if len(label_files) == 0:
            errors.append(f"L{layer}: missing label chunks")
        if len(input_files) != len(label_files):
            errors.append(f"L{layer}: input chunk count {len(input_files)} != label chunk count {len(label_files)}")

        for p in input_files + label_files:
            if not p.exists() or p.stat().st_size <= 0:
                errors.append(f"empty file: {p}")

        if args.check_load and input_files and label_files:
            layer_n = 0
            for x_fp, y_fp in zip(input_files, label_files):
                assert torch_mod is not None
                x = torch_mod.load(x_fp, map_location="cpu")
                y = torch_mod.load(y_fp, map_location="cpu")
                idx = y["indices"]
                weights = y["weights"]
                if tuple(x.shape[1:]) != (hidden_size,):
                    errors.append(f"{x_fp.name}: input shape {tuple(x.shape)} expected [N,{hidden_size}]")
                if tuple(idx.shape[1:]) != (top_k,):
                    errors.append(f"{y_fp.name}: indices shape {tuple(idx.shape)} expected [N,{top_k}]")
                if tuple(weights.shape) != tuple(idx.shape):
                    errors.append(f"{y_fp.name}: weights shape {tuple(weights.shape)} != indices {tuple(idx.shape)}")
                if int(y.get("num_experts", num_experts)) != num_experts:
                    errors.append(f"{y_fp.name}: num_experts mismatch")
                if int(y.get("top_k", top_k)) != top_k:
                    errors.append(f"{y_fp.name}: top_k mismatch")
                if x.shape[0] != idx.shape[0]:
                    errors.append(f"L{layer} chunk {x_fp.name}: input N {x.shape[0]} != label N {idx.shape[0]}")
                layer_n += int(x.shape[0])
            if layer_n != total_records:
                errors.append(f"L{layer}: total records {layer_n} != manifest {total_records}")

    trace_files = _sorted_files(str(trace_dir / "decode_trace_chunk_*.pt"))
    if len(trace_files) == 0:
        errors.append("missing trace chunks")
    for p in trace_files:
        if not p.exists() or p.stat().st_size <= 0:
            errors.append(f"empty trace file: {p}")
    if args.check_load:
        trace_n = 0
        for p in trace_files:
            assert torch_mod is not None
            t = torch_mod.load(p, map_location="cpu")
            li = t["layer_indices"]
            lw = t["layer_weights"]
            if tuple(li.shape[1:]) != (num_layers, top_k):
                errors.append(f"{p.name}: layer_indices shape {tuple(li.shape)} expected [N,{num_layers},{top_k}]")
            if tuple(lw.shape) != tuple(li.shape):
                errors.append(f"{p.name}: layer_weights shape {tuple(lw.shape)} != indices {tuple(li.shape)}")
            trace_n += int(li.shape[0])
        if trace_n != total_records:
            errors.append(f"trace total records {trace_n} != manifest {total_records}")

    if errors:
        print("[FAIL]")
        for e in errors[:200]:
            print(f"  - {e}")
        if len(errors) > 200:
            print(f"  ... {len(errors) - 200} more")
        raise SystemExit(2)

    print("[OK] profile structure is valid")
    if not args.check_load:
        print("[hint] rerun with --check-load on the training server to validate tensor shapes with torch.load")


if __name__ == "__main__":
    main()
