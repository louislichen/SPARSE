"""Lightweight HCLM critical-path profiler.

This module is independent from SPARE strategy code.  It records sampled decode
steps and layer-level component timings only when explicitly enabled with
``SPARSE_HCLM_CP_PROFILE=1``.  Modeled communication fields are estimates from
logical counters; they are not physical NIC measurements.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default)) or default))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return float(default)


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(start_ns: int, end_ns: int) -> float:
    return max(0.0, float(int(end_ns) - int(start_ns)) / 1_000_000.0)


class HCLMCriticalPathProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        out_dir: str,
        rank: int,
        world_size: int,
        sample_start: int,
        sample_steps: int,
        record_layers: bool = True,
        record_jsonl: bool = True,
        record_modeled_comm: bool = True,
        record_lite: bool = False,
        bw_mbps: float = 125.0,
        lat_ms: float = 2.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.sample_start = int(sample_start)
        self.sample_steps = max(0, int(sample_steps))
        self.record_layers = bool(record_layers)
        self.record_jsonl = bool(record_jsonl)
        self.record_modeled_comm = bool(record_modeled_comm)
        self.record_lite = bool(record_lite)
        self.bw_mbps = float(bw_mbps)
        self.lat_ms = float(lat_ms)
        self._step_file = None
        self._layer_file = None
        self._lite_file = None
        self._active_steps: Dict[int, Dict[str, Any]] = {}
        self._lite_steps: Dict[int, Dict[str, Any]] = {}
        self._layers: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if self.enabled and self.record_jsonl:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if not self.record_lite:
                self._step_file = (self.out_dir / f"cp_step_rank_rank{self.rank}.jsonl").open("w", encoding="utf-8")
                self._layer_file = (self.out_dir / f"cp_layer_rank_rank{self.rank}.jsonl").open("w", encoding="utf-8")
            else:
                self._lite_file = (self.out_dir / f"cp_trace_minimal_rank{self.rank}.jsonl").open("w", encoding="utf-8")
            if self.rank == 0:
                meta = {
                    "schema": "hclm_cp_lite_v1" if self.record_lite else "hclm_cp_v1",
                    "sample_start": self.sample_start,
                    "sample_steps": self.sample_steps,
                    "record_layers": self.record_layers,
                    "record_lite": self.record_lite,
                    "record_modeled_comm": self.record_modeled_comm,
                    "bw_mbps": self.bw_mbps,
                    "lat_ms": self.lat_ms,
                    "semantics": "CPU-side sampled critical-path trace; modeled communication fields are estimates from logical counters, not physical NIC measurements.",
                }
                (self.out_dir / "hclm_cp_profile_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def should_sample(self, decode_step: int) -> bool:
        if not self.enabled:
            return False
        step = int(decode_step)
        return self.sample_start <= step < self.sample_start + self.sample_steps

    def start_step(self, step_id: int, batch: int, seq_len: int) -> None:
        if not self.should_sample(step_id):
            return
        start_ns = now_ns()
        self._active_steps[int(step_id)] = {
            "event_type": "step",
            "rank": self.rank,
            "world_size": self.world_size,
            "step_id": int(step_id),
            "batch": int(batch),
            "seq_len": int(seq_len),
            "start_ns": int(start_ns),
            "end_ns": None,
            "duration_ms": None,
            "sampled": True,
        }
        if self.record_lite:
            self._lite_steps[int(step_id)] = self._empty_lite_record(step_id, batch, seq_len, start_ns)

    def end_step(self, step_id: int, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.should_sample(step_id):
            return
        step_id = int(step_id)
        rec = self._active_steps.pop(step_id, None)
        if rec is None:
            return
        end_ns = now_ns()
        rec["end_ns"] = int(end_ns)
        rec["duration_ms"] = ns_to_ms(int(rec["start_ns"]), int(end_ns))
        if extra:
            rec.update(extra)
        self._write_step(rec)
        if self.record_lite:
            lite = self._lite_steps.pop(step_id, None)
            if lite is not None:
                lite["step_end_ns"] = int(end_ns)
                lite["step_ms"] = ns_to_ms(int(lite["step_start_ns"]), int(end_ns))
                self._write_lite(lite)
        self._flush_step_layers(step_id)

    def start_layer(self, step_id: int, layer_id: int) -> None:
        if not (self.record_layers and self.should_sample(step_id)):
            return
        rec = self._layers.setdefault((int(step_id), int(layer_id)), self._empty_layer_record(step_id, layer_id))
        rec["start_ns"] = int(now_ns())

    def end_layer(self, step_id: int, layer_id: int, extra: Optional[Dict[str, Any]] = None) -> None:
        if not (self.record_layers and self.should_sample(step_id)):
            return
        rec = self._layers.setdefault((int(step_id), int(layer_id)), self._empty_layer_record(step_id, layer_id))
        end_ns = now_ns()
        rec["end_ns"] = int(end_ns)
        if rec.get("start_ns") is not None:
            rec["duration_ms"] = ns_to_ms(int(rec["start_ns"]), int(end_ns))
        if extra:
            self.record_layer_summary(step_id, layer_id, extra)

    def record_component(
        self,
        step_id: int,
        layer_id: int,
        name: str,
        start_ns: int,
        end_ns: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.should_sample(step_id):
            return
        duration = ns_to_ms(start_ns, end_ns)
        field = self._component_field(name)
        if self.record_layers:
            rec = self._layers.setdefault((int(step_id), int(layer_id)), self._empty_layer_record(step_id, layer_id))
            if field:
                cur = rec.get(field)
                rec[field] = float(duration) if cur is None else float(cur) + float(duration)
        if self.record_lite:
            self._record_lite_component(int(step_id), str(name), int(start_ns), int(end_ns))
        if extra:
            self.record_layer_summary(step_id, layer_id, extra)

    def record_layer_summary(self, step_id: int, layer_id: int, metrics: Dict[str, Any]) -> None:
        if not self.should_sample(step_id):
            return
        if self.record_layers:
            rec = self._layers.setdefault((int(step_id), int(layer_id)), self._empty_layer_record(step_id, layer_id))
            for k, v in metrics.items():
                if v is None:
                    continue
                if isinstance(v, (int, float)) and rec.get(k) is not None:
                    rec[k] = float(rec[k]) + float(v)
                else:
                    rec[k] = v
        if self.record_lite:
            self._record_lite_summary(int(step_id), metrics)

    def flush(self) -> None:
        if self._step_file is not None:
            self._step_file.flush()
        if self._layer_file is not None:
            self._layer_file.flush()
        if self._lite_file is not None:
            self._lite_file.flush()

    def close(self) -> None:
        self.flush()
        if self._step_file is not None:
            self._step_file.close()
            self._step_file = None
        if self._layer_file is not None:
            self._layer_file.close()
            self._layer_file = None
        if self._lite_file is not None:
            self._lite_file.close()
            self._lite_file = None

    def _flush_step_layers(self, step_id: int) -> None:
        keys = sorted(k for k in self._layers.keys() if int(k[0]) == int(step_id))
        for key in keys:
            self._write_layer(self._layers.pop(key))

    def _write_step(self, rec: Dict[str, Any]) -> None:
        if self._step_file is not None:
            self._step_file.write(json.dumps(rec, sort_keys=True) + "\n")

    def _write_layer(self, rec: Dict[str, Any]) -> None:
        if self._layer_file is not None:
            self._layer_file.write(json.dumps(rec, sort_keys=True) + "\n")

    def _write_lite(self, rec: Dict[str, Any]) -> None:
        if self._lite_file is not None:
            self._lite_file.write(json.dumps(rec, sort_keys=True) + "\n")

    def _empty_lite_record(self, step_id: int, batch: int, seq_len: int, start_ns: int) -> Dict[str, Any]:
        return {
            "case": self.out_dir.name,
            "stage": "decode",
            "step_id": int(step_id),
            "rank": self.rank,
            "step_start_ns": int(start_ns),
            "step_end_ns": None,
            "step_ms": None,
            "router_start_ns": None,
            "router_end_ns": None,
            "dispatch_start_ns": None,
            "dispatch_end_ns": None,
            "h2d_wait_start_ns": None,
            "h2d_wait_end_ns": None,
            "expert_compute_start_ns": None,
            "expert_compute_end_ns": None,
            "gather_start_ns": None,
            "gather_end_ns": None,
            "sync_wait_start_ns": None,
            "sync_wait_end_ns": None,
            "h2d_copy_ms": None,
            "modeled_dispatch_comm_ms": None,
            "_batch": int(batch),
            "_seq_len": int(seq_len),
        }

    def _record_lite_component(self, step_id: int, name: str, start_ns: int, end_ns: int) -> None:
        rec = self._lite_steps.get(int(step_id))
        if rec is None:
            return
        mapping = {
            "router": ("router_start_ns", "router_end_ns"),
            "dispatch": ("dispatch_start_ns", "dispatch_end_ns"),
            "dispatch_comm": ("dispatch_start_ns", "dispatch_end_ns"),
            "h2d_exposed_wait": ("h2d_wait_start_ns", "h2d_wait_end_ns"),
            "expert_compute": ("expert_compute_start_ns", "expert_compute_end_ns"),
            "gather": ("gather_start_ns", "gather_end_ns"),
            "gather_comm": ("gather_start_ns", "gather_end_ns"),
        }
        pair = mapping.get(str(name))
        if pair is None:
            return
        s_key, e_key = pair
        cur_s = rec.get(s_key)
        cur_e = rec.get(e_key)
        rec[s_key] = int(start_ns) if cur_s is None else min(int(cur_s), int(start_ns))
        rec[e_key] = int(end_ns) if cur_e is None else max(int(cur_e), int(end_ns))

    def _record_lite_summary(self, step_id: int, metrics: Dict[str, Any]) -> None:
        rec = self._lite_steps.get(int(step_id))
        if rec is None:
            return
        for src_key, dst_key in (
            ("h2d_copy_ms", "h2d_copy_ms"),
            ("demand_h2d_wait_ms", "h2d_copy_ms"),
            ("modeled_dispatch_comm_ms", "modeled_dispatch_comm_ms"),
        ):
            v = metrics.get(src_key)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            cur = rec.get(dst_key)
            rec[dst_key] = fv if cur is None else float(cur) + fv

    def _empty_layer_record(self, step_id: int, layer_id: int) -> Dict[str, Any]:
        return {
            "event_type": "layer",
            "rank": self.rank,
            "world_size": self.world_size,
            "step_id": int(step_id),
            "layer_id": int(layer_id),
            "start_ns": None,
            "end_ns": None,
            "duration_ms": None,
            "router_ms": None,
            "dispatch_ms": None,
            "modeled_dispatch_comm_ms": None,
            "modeled_dispatch_event_ms": None,
            "modeled_dispatch_byte_ms": None,
            "h2d_copy_ms": None,
            "h2d_exposed_wait_ms": None,
            "demand_h2d_wait_ms": None,
            "demand_h2d_bytes": None,
            "demand_h2d_expert_count": None,
            "expert_stage_total_ms": None,
            "expert_compute_ms": None,
            "shared_expert_ms": None,
            "gather_ms": None,
            "modeled_gather_comm_ms": None,
            "control_overhead_ms": None,
            "other_ms": None,
            "inter_node_events": None,
            "inter_node_bytes": None,
            "local_expert_count": None,
            "local_token_count": None,
            "notes": None,
        }

    @staticmethod
    def _component_field(name: str) -> Optional[str]:
        return {
            "router": "router_ms",
            "dispatch": "dispatch_ms",
            "dispatch_comm": "dispatch_ms",
            "h2d_copy": "h2d_copy_ms",
            "demand_h2d_wait": "h2d_copy_ms",
            "h2d_exposed_wait": "h2d_exposed_wait_ms",
            "expert_stage_total": "expert_stage_total_ms",
            "expert_compute": "expert_compute_ms",
            "shared": "shared_expert_ms",
            "shared_expert": "shared_expert_ms",
            "gather": "gather_ms",
            "gather_comm": "gather_ms",
            "control": "control_overhead_ms",
            "control_overhead": "control_overhead_ms",
            "other": "other_ms",
        }.get(str(name))


class NoOpHCLMCriticalPathProfiler(HCLMCriticalPathProfiler):
    def __init__(self) -> None:
        self.enabled = False
        self.out_dir = Path("")
        self.rank = 0
        self.world_size = 1
        self.sample_start = 0
        self.sample_steps = 0
        self.record_layers = False
        self.record_jsonl = False
        self.record_modeled_comm = False
        self.record_lite = False
        self.bw_mbps = 0.0
        self.lat_ms = 0.0

    def should_sample(self, decode_step: int) -> bool:
        return False

    def start_step(self, step_id: int, batch: int, seq_len: int) -> None:
        return

    def end_step(self, step_id: int, extra: Optional[Dict[str, Any]] = None) -> None:
        return

    def start_layer(self, step_id: int, layer_id: int) -> None:
        return

    def end_layer(self, step_id: int, layer_id: int, extra: Optional[Dict[str, Any]] = None) -> None:
        return

    def record_component(self, step_id: int, layer_id: int, name: str, start_ns: int, end_ns: int, extra: Optional[Dict[str, Any]] = None) -> None:
        return

    def record_layer_summary(self, step_id: int, layer_id: int, metrics: Dict[str, Any]) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


_GLOBAL_PROFILER: HCLMCriticalPathProfiler = NoOpHCLMCriticalPathProfiler()


def create_profiler_from_env(rank: int, world_size: int) -> HCLMCriticalPathProfiler:
    if not _env_flag("SPARSE_HCLM_CP_PROFILE", "0"):
        return NoOpHCLMCriticalPathProfiler()
    return HCLMCriticalPathProfiler(
        enabled=True,
        out_dir=os.environ.get("SPARSE_HCLM_CP_OUT_DIR", "results/lod_hclm_cp/raw"),
        rank=int(rank),
        world_size=int(world_size),
        sample_start=_env_int("SPARSE_HCLM_CP_SAMPLE_START", 8),
        sample_steps=_env_int("SPARSE_HCLM_CP_SAMPLE_STEPS", 16),
        record_layers=_env_flag("SPARSE_HCLM_CP_RECORD_LAYERS", "1"),
        record_jsonl=_env_flag("SPARSE_HCLM_CP_RECORD_JSONL", "1"),
        record_modeled_comm=_env_flag("SPARSE_HCLM_CP_RECORD_MODELED_COMM", "1"),
        record_lite=_env_flag("SPARSE_HCLM_CP_LITE", "0"),
        bw_mbps=_env_float("SPARSE_HCLM_CP_BW_MBPS", 125.0),
        lat_ms=_env_float("SPARSE_HCLM_CP_LAT_MS", 2.0),
    )


def set_global_profiler(profiler: HCLMCriticalPathProfiler) -> None:
    global _GLOBAL_PROFILER
    _GLOBAL_PROFILER = profiler


def get_global_profiler() -> HCLMCriticalPathProfiler:
    return _GLOBAL_PROFILER


def modeled_comm_ms(events: float, bytes_count: float, *, bw_mbps: float, lat_ms: float) -> Dict[str, float]:
    event_ms = max(float(events), 0.0) * max(float(lat_ms), 0.0)
    byte_ms = max(float(bytes_count), 0.0) / max(float(bw_mbps), 1e-12) / 1_000_000.0 * 1000.0
    return {
        "modeled_dispatch_event_ms": event_ms,
        "modeled_dispatch_byte_ms": byte_ms,
        "modeled_dispatch_comm_ms": event_ms + byte_ms,
    }
