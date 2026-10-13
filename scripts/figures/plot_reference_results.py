#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SETTING_NAMES = {
    "Qwen, native link (0.88 Gbit/s)": "native",
    "Qwen, 400 Mbit/s": "400mbps",
}
WORKLOAD_ORDER = [
    "b1/p1024/n32 long prefill",
    "b2/p512/n64 balanced",
    "b4/p512/n128 decode heavy",
]
WORKLOAD_LABELS = ["b1/p1024/n32", "b2/p512/n64", "b4/p512/n128"]
METHOD_ORDER = [
    "RR-LRU",
    "LoD/RR no-cache",
    "EPLB-like",
    "Prism/DanceMoE static",
    "Static-hot cache",
    "MoE-Infinity-style",
    "SPARSE",
]
METHOD_LABELS = {
    "RR-LRU": "RR-LRU",
    "LoD/RR no-cache": "LoD",
    "EPLB-like": "EPLB",
    "Prism/DanceMoE static": "Prism",
    "Static-hot cache": "Static-hot",
    "MoE-Infinity-style": "MoE-Inf.",
    "SPARSE": "SPARSE",
}
COLORS = ["#D7DEE8", "#F3D8C7", "#CFE4D5", "#E7D9EF", "#F6E6AE", "#BFDCE5", "#E79796"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_one(rows: list[dict[str, str]], setting: str, metric: str, out_dir: Path) -> None:
    values: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        if row["setting"] == setting and row["metric"] == metric:
            values[(row["workload"], row["method"])] = (
                float(row["normalized_mean"]),
                float(row["normalized_slice_sd"]),
            )

    present = [method for method in METHOD_ORDER if any((workload, method) in values for workload in WORKLOAD_ORDER)]
    x = np.arange(len(WORKLOAD_ORDER), dtype=float)
    group_width = 0.90
    width = group_width / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=300)
    for index, method in enumerate(present):
        offset = -group_width / 2 + width / 2 + index * width
        means = [values.get((workload, method), (np.nan, 0.0))[0] for workload in WORKLOAD_ORDER]
        errors = [values.get((workload, method), (np.nan, 0.0))[1] for workload in WORKLOAD_ORDER]
        ax.bar(
            x + offset,
            means,
            width=width,
            color=COLORS[METHOD_ORDER.index(method)],
            edgecolor="#4B5563",
            linewidth=0.45,
            yerr=errors,
            error_kw={"elinewidth": 0.55, "capsize": 1.5, "capthick": 0.55},
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8, zorder=2)
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x, WORKLOAD_LABELS)
    ax.set_ylabel("Throughput normalized to RR-LRU")
    ax.set_ylim(0.35, max(1.65, ax.get_ylim()[1]))
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=4,
        frameon=False,
        fontsize=7.1,
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.19, top=0.79)

    stem = f"fig_qwen_{SETTING_NAMES[setting]}_{metric.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(out_dir / f"{stem}.png", facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the four fixed-canvas paper reference plots.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.input)
    for setting in SETTING_NAMES:
        for metric in ("E2E", "Decode"):
            plot_one(rows, setting, metric, args.out_dir)
    print(f"[figures] wrote four fixed-canvas PDF/PNG pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
