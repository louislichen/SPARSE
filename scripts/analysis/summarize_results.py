#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted(root.rglob("case_config.json")):
        config = load_json(config_path)
        method = str(config["method"])
        summary_path = config_path.parent / f"{method}_summary.json"
        if not summary_path.is_file():
            continue
        summary = load_json(summary_path)
        rows.append(
            {
                **config,
                "e2e_tok_s": float(summary["throughput_tok_s"]),
                "decode_tok_s": float(summary["decode_throughput_tok_s"]),
                "e2e_repeat_sd": float(summary.get("throughput_tok_s_std", 0.0)),
                "decode_repeat_sd": float(summary.get("decode_throughput_tok_s_std", 0.0)),
                "summary": str(summary_path.relative_to(root)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model"]), str(row["network"]), str(row["workload"]), str(row["method"]))
        grouped[key].append(row)

    baseline: dict[tuple[str, str, str], tuple[float, float]] = {}
    for (model, network, workload, method), group in grouped.items():
        if method == "rr":
            baseline[(model, network, workload)] = (
                statistics.mean(float(row["e2e_tok_s"]) for row in group),
                statistics.mean(float(row["decode_tok_s"]) for row in group),
            )

    output: list[dict[str, Any]] = []
    for (model, network, workload, method), group in sorted(grouped.items()):
        e2e = [float(row["e2e_tok_s"]) for row in group]
        decode = [float(row["decode_tok_s"]) for row in group]
        e2e_mean = statistics.mean(e2e)
        decode_mean = statistics.mean(decode)
        base = baseline.get((model, network, workload))
        output.append(
            {
                "model": model,
                "network": network,
                "workload": workload,
                "method": method,
                "request_groups": len(group),
                "e2e_tok_s_mean": e2e_mean,
                "e2e_tok_s_group_sd": statistics.stdev(e2e) if len(e2e) > 1 else 0.0,
                "decode_tok_s_mean": decode_mean,
                "decode_tok_s_group_sd": statistics.stdev(decode) if len(decode) > 1 else 0.0,
                "e2e_gain_vs_rr_pct": ((e2e_mean / base[0]) - 1.0) * 100.0 if base else "",
                "decode_gain_vs_rr_pct": ((decode_mean / base[1]) - 1.0) * 100.0 if base else "",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compact SPARSE runtime summaries.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = find_rows(args.root)
    if not rows:
        raise SystemExit(f"no completed case summaries found below {args.root}")
    aggregates = aggregate(rows)
    write_csv(args.out / "runtime_runs.csv", rows)
    write_csv(args.out / "runtime_aggregate.csv", aggregates)
    print(f"[summary] runs={len(rows)} aggregates={len(aggregates)} out={args.out}")


if __name__ == "__main__":
    main()
