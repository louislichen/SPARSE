from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _to_path(p: Optional[str]) -> Optional[Path]:
    return Path(p).expanduser().resolve() if p else None


@dataclass(frozen=True)
class Paths:
    model_path: Path
    data_path: Optional[Path]
    base_dir: Path

    profile_dir: Path
    rs_weights_dir: Path
    scp_path: Path
    metrics_dir: Path


@dataclass(frozen=True)
class SKVSConfig:
    h2o_sink: int = 4
    h2o_window: int = 32
    l0_guidance_ratio: float = 0.6


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Central configuration with environment-variable overrides.

    Recommended env vars:
      SPARSE_MODEL_PATH: local HF model dir or repo id
      SPARSE_DATA_PATH:  path to a json/jsonl/json.gz dataset (optional for online eval)
      SPARSE_BASE_DIR:   base output directory

      SKVS_H2O_SINK, SKVS_H2O_WINDOW, SKVS_L0_GUIDANCE_RATIO
    """
    paths: Paths
    skvs: SKVSConfig

    @staticmethod
    def from_env(
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "RuntimeConfig":
        mp = model_path or _env("SPARSE_MODEL_PATH")
        if not mp:
            raise ValueError("MODEL_PATH is required. Provide --model-path or set SPARSE_MODEL_PATH.")

        dp = data_path if data_path is not None else _env("SPARSE_DATA_PATH", None)
        bd = base_dir or _env("SPARSE_BASE_DIR", "./results")

        model_p = _to_path(mp) if os.path.exists(os.path.expanduser(mp)) else Path(mp)  # allow repo id
        data_p = _to_path(dp) if dp else None
        base_p = _to_path(bd) or Path("./results").resolve()

        profile_dir = base_p / "profile_data"
        rs_weights_dir = base_p / "mlp_trainer"
        scp_path = base_p / "scp_placement.json"
        metrics_dir = base_p / "training_metrics"

        skvs = SKVSConfig(
            h2o_sink=int(_env("SKVS_H2O_SINK", "4")),
            h2o_window=int(_env("SKVS_H2O_WINDOW", "32")),
            l0_guidance_ratio=float(_env("SKVS_L0_GUIDANCE_RATIO", "0.6")),
        )

        if overrides:
            if "skvs" in overrides:
                sk = overrides["skvs"]
                skvs = SKVSConfig(
                    h2o_sink=int(sk.get("h2o_sink", skvs.h2o_sink)),
                    h2o_window=int(sk.get("h2o_window", skvs.h2o_window)),
                    l0_guidance_ratio=float(sk.get("l0_guidance_ratio", skvs.l0_guidance_ratio)),
                )

        paths = Paths(
            model_path=model_p if isinstance(model_p, Path) else Path(str(model_p)),
            data_path=data_p,
            base_dir=base_p,
            profile_dir=profile_dir,
            rs_weights_dir=rs_weights_dir,
            scp_path=scp_path,
            metrics_dir=metrics_dir,
        )
        return RuntimeConfig(paths=paths, skvs=skvs)
