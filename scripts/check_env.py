#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
    values = dict(os.environ)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = os.path.expandvars(value)
        os.environ[key.strip()] = values[key.strip()]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a node-local SPARSE environment.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--strict", action="store_true", help="Require model, data, RS checkpoints, and SCP placement files.")
    args = parser.parse_args()

    env = load_env(args.env_file.resolve())
    required = [
        "PYTHON_BIN",
        "CUDA_VISIBLE_DEVICES",
        "NODE_RANK",
        "MASTER_ADDR",
        "NCCL_SOCKET_IFNAME",
        "MODEL_PATH",
        "C4_RUNTIME",
        "RESULTS_ROOT",
    ]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SystemExit(f"missing required settings: {', '.join(missing)}")

    import numpy as np
    import torch
    import transformers

    visible = [item for item in env["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
    report = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "configured_visible_devices": visible,
        "nccl_available": torch.distributed.is_nccl_available(),
    }
    if torch.cuda.is_available():
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(json.dumps(report, indent=2))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if torch.cuda.device_count() != len(visible):
        raise SystemExit("CUDA_VISIBLE_DEVICES and torch.cuda.device_count() disagree")
    if not torch.distributed.is_nccl_available():
        raise SystemExit("PyTorch NCCL support is unavailable")

    paths = ["MODEL_PATH", "C4_RUNTIME"]
    if args.strict:
        paths.extend(["SCP_PATH", "RS_DIR", "TOPOLOGY_PATH"])
    absent = [f"{key}={env.get(key, '')}" for key in paths if not Path(env.get(key, "")).exists()]
    if absent:
        raise SystemExit("missing paths:\n  " + "\n  ".join(absent))
    print("[env-check] PASS")


if __name__ == "__main__":
    main()
