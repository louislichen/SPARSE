"""Independent ProMoE-style baseline, adapted to rank-local EP ownership.

Mechanisms follow arXiv:2410.22134v3, sections 4--6. This is new Python code,
not the authors' C++ implementation. See docs/PROMOE_BASELINE.md for fidelity
and evaluation limits. CUDA copies are serialized one parameter chunk at a
time so precise work can preempt speculative work at a chunk boundary.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import nn


class ProMoEPredictor(nn.Module):
    """Two linear layers; hidden width is an explicit adaptation setting."""

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, num_experts))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(-1, x.shape[-1]))

    def forward(self, x: torch.Tensor, *, return_logits: bool = False) -> torch.Tensor:
        logits = self.forward_logits(x)
        return logits if return_logits else logits.softmax(-1)


def load_predictors(directory: str, layers: Iterable[int], hidden_size: int,
                    num_experts: int, distance: int) -> Dict[int, Tuple[int, nn.Module]]:
    """Require the complete, explicitly labelled same-step stride bank."""
    if distance < 1:
        raise ValueError("ProMoE prediction distance must be positive")
    layer_set = set(layers)
    pairs = [(s, s + distance) for s in sorted(layer_set) if s + distance in layer_set]
    if not pairs:
        raise ValueError("No MoE layer pairs for the requested prediction distance")
    bank = {}
    for source, target in pairs:
        path = Path(directory) / f"rs_l{target}_inL{source}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ProMoE-style predictor: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"model_type": "promoe_style_mlp", "source_layer": source,
                    "target_layer": target, "input_dim": hidden_size,
                    "num_experts": num_experts, "feature_mode": "hidden", "feature_source": "decoder_input"}
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"{path.name}: {key} must be {value!r}, got {payload.get(key)!r}")
        model = ProMoEPredictor(hidden_size, num_experts, int(payload["hidden_dim"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        bank[source] = (target, model.float().eval())
    return bank


@dataclass
class _Entry:
    payload: Tuple[torch.Tensor, ...]
    ready_chunks: int = 0
    leased: bool = False
    last_use: Any = None


@dataclass
class _Layer:
    host: Dict[int, Tuple[torch.Tensor, ...]]
    capacity: int
    entries: OrderedDict = field(default_factory=OrderedDict)
    required: set = field(default_factory=set)
    precise_step: Optional[int] = None


@dataclass
class _Job:
    layer: int
    expert: int
    step: int
    precise: bool
    cancelled: bool = False


class ChunkedPrefetcher:
    """One copy worker per rank, strict per-layer LRU with compute leases.

    CPU mode executes the same queue/cache logic for deterministic tests.
    A compute lease and CUDA completion event prevent eviction of weights that
    kernels still read. Neither precise nor speculative jobs exceed capacity.
    """

    def __init__(self, device: torch.device, on_copy: Optional[Callable] = None):
        self.device = torch.device(device)
        self.on_copy = on_copy
        self.layers: Dict[int, _Layer] = {}
        self.stats: Dict[str, float] = {}
        self.cv = threading.Condition()
        self.high, self.low = deque(), deque()
        self.jobs: Dict[Tuple[int, int], _Job] = {}
        self.active: Optional[_Job] = None
        self.step = -1
        self.error: Optional[BaseException] = None
        self.closed = False
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.worker = threading.Thread(target=self._run, name="promoe-copy", daemon=True)
        self.worker.start()

    def _count(self, key: str, value: float = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + value

    def _check(self) -> None:
        if self.error is not None:
            raise RuntimeError("ProMoE copy worker failed") from self.error
        if self.closed:
            raise RuntimeError("ProMoE prefetcher is closed")

    def register(self, layer: int, host: Dict[int, Tuple[torch.Tensor, ...]], capacity: int) -> None:
        if capacity < 1:
            raise ValueError("ProMoE-style requires at least one cache slot per rank/MoE layer")
        with self.cv:
            self._check()
            if layer in self.layers:
                raise ValueError(f"Duplicate layer {layer}")
            self.layers[layer] = _Layer(host, capacity)

    def begin_step(self, step: int) -> None:
        with self.cv:
            self._check()
            if self.step == step:
                return
            self.step = step
            self.high.clear()
            self.low.clear()
            for job in self.jobs.values():
                job.cancelled = True
            self.jobs.clear()
            for layer in self.layers.values():
                layer.required.clear()
                layer.precise_step = None
            self.cv.notify_all()

    def _enqueue(self, layer_id: int, eid: int, step: int, precise: bool) -> None:
        layer = self.layers[layer_id]
        if eid not in layer.host:
            raise ValueError(f"Non-owned expert {eid} for layer {layer_id}")
        entry = layer.entries.get(eid)
        if entry is not None:
            layer.entries.move_to_end(eid)
            if entry.ready_chunks == len(entry.payload):
                return
        key = (layer_id, eid)
        job = self.jobs.get(key)
        if job is not None:
            if precise and not job.precise:
                job.precise = True
                self.high.append(job)
            return
        job = _Job(layer_id, eid, step, precise)
        self.jobs[key] = job
        (self.high if precise else self.low).append(job)

    def predicted(self, layer_id: int, experts: Iterable[int], step: int) -> bool:
        with self.cv:
            self._check()
            layer = self.layers[layer_id]
            if step != self.step or layer.precise_step == step:
                self._count("stale_prediction_batches")
                return False
            ids = list(dict.fromkeys(int(e) for e in experts))
            self._count("predicted_experts", len(ids))
            for eid in ids:
                self._enqueue(layer_id, eid, step, False)
            self.cv.notify_all()
            return True

    def precise(self, layer_id: int, experts: Iterable[int], step: int) -> list:
        with self.cv:
            self._check()
            if step != self.step:
                raise ValueError("Precise demand belongs to a different step")
            layer = self.layers[layer_id]
            ids = list(dict.fromkeys(int(e) for e in experts))
            if any(e not in layer.host for e in ids):
                raise ValueError("Precise demand contains non-owned experts")
            layer.precise_step = step
            layer.required = set(ids)
            # In-flight CUDA copy cannot be cancelled; stop after that chunk.
            for key, job in list(self.jobs.items()):
                if job.layer == layer_id and not job.precise and job.expert not in layer.required:
                    job.cancelled = True
                    del self.jobs[key]
                    self._count("cancelled_speculative_experts")
            def readiness(eid):
                entry = layer.entries.get(eid)
                if entry is None:
                    return (0, 0)
                return (2 if entry.ready_chunks == len(entry.payload) else 1, entry.ready_chunks)
            # Fully cached, partially copied, then not started. Stable ties.
            ids.sort(key=readiness, reverse=True)
            self._count("precise_experts", len(ids))
            self._count("ready_at_precise", sum(
                readiness(e)[0] == 2 for e in ids))
            for eid in ids:
                self._enqueue(layer_id, eid, step, True)
            self.cv.notify_all()
            return ids

    def _entry(self, job: _Job) -> Optional[_Entry]:
        layer = self.layers[job.layer]
        entry = layer.entries.get(job.expert)
        if entry is not None:
            return entry
        if len(layer.entries) >= layer.capacity:
            for eid, candidate in list(layer.entries.items()):
                if candidate.leased or eid in layer.required:
                    continue
                if candidate.last_use is not None and not candidate.last_use.query():
                    continue
                # Another job must not still own a partially copied allocation.
                if (job.layer, eid) in self.jobs:
                    continue
                del layer.entries[eid]
                # Release the victim before allocating the replacement. Keeping
                # this local reference would add one live expert above budget.
                del candidate
                self._count("evictions")
                break
            else:
                return None
        host = layer.host[job.expert]
        entry = _Entry(tuple(torch.empty_like(t, device=self.device) for t in host))
        if self.stream is not None:
            for tensor in entry.payload:
                tensor.record_stream(self.stream)
        layer.entries[job.expert] = entry
        self._count("allocated_experts")
        self.stats["peak_resident_experts_per_layer"] = max(
            self.stats.get("peak_resident_experts_per_layer", 0), len(layer.entries))
        return entry

    def _next(self):
        # Inspect at most one pass; blocked precise work may coexist with
        # speculative jobs for a later layer that has available slots.
        for queue in (self.high, self.low):
            for _ in range(len(queue)):
                job = queue.popleft()
                if job.cancelled or job.step != self.step:
                    continue
                if queue is self.low and job.precise:
                    continue
                if self.jobs.get((job.layer, job.expert)) is not job:
                    continue
                entry = self._entry(job)
                if entry is not None:
                    return job, entry
                queue.append(job)
        return None

    def _copy_chunk(self, job: _Job, entry: _Entry, chunk: int) -> int:
        source = self.layers[job.layer].host[job.expert][chunk]
        if self.stream is None:
            entry.payload[chunk].copy_(source)
        else:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                entry.payload[chunk].copy_(source, non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.stream)
            # Do not submit subsequent low-priority chunks into CUDA's FIFO:
            # host queue preemption would otherwise have no effect.
            done.synchronize()
        return source.numel() * source.element_size()

    def _run(self) -> None:
        try:
            with torch.inference_mode():
                while True:
                    with self.cv:
                        if self.closed:
                            return
                        task = self._next()
                        if task is None:
                            self.cv.wait(0.002)
                            continue
                        job, entry = task
                        self.active = job
                        chunk = entry.ready_chunks
                        # Snapshot priority: a promoted in-flight chunk retains
                        # the classification under which it was issued.
                        speculative = not job.precise
                    nbytes = self._copy_chunk(job, entry, chunk)
                    if self.on_copy is not None:
                        self.on_copy(job.layer, nbytes, speculative)
                    with self.cv:
                        entry.ready_chunks += 1
                        self._count("prefetch_bytes" if speculative else "demand_bytes", nbytes)
                        self._count("copied_chunks")
                        self.active = None
                        if self.jobs.get((job.layer, job.expert)) is job:
                            if job.cancelled or entry.ready_chunks == len(entry.payload):
                                del self.jobs[(job.layer, job.expert)]
                            else:
                                (self.high if job.precise else self.low).appendleft(job)
                        self.cv.notify_all()
                    # The next allocation must not retain the previous entry
                    # through worker-local references after it is evicted.
                    del entry, task
        except BaseException as exc:
            with self.cv:
                self.error = exc
                self.active = None
                self.cv.notify_all()

    def acquire(self, layer_id: int, eid: int, timeout: float = 60) -> Tuple[torch.Tensor, ...]:
        start = time.monotonic()
        with self.cv:
            while True:
                self._check()
                layer = self.layers[layer_id]
                if eid not in layer.required:
                    raise ValueError("acquire requires a preceding precise declaration")
                entry = layer.entries.get(eid)
                if entry is not None and entry.ready_chunks == len(entry.payload):
                    if entry.leased:
                        raise ValueError("Expert already acquired")
                    entry.leased = True
                    layer.entries.move_to_end(eid)
                    self._count("acquire_wait_ms", (time.monotonic() - start) * 1000)
                    if self.device.type == "cuda":
                        stream = torch.cuda.current_stream(self.device)
                        for tensor in entry.payload:
                            tensor.record_stream(stream)
                    return entry.payload
                if time.monotonic() - start >= timeout:
                    raise TimeoutError(f"Timed out loading layer={layer_id} expert={eid}")
                self.cv.wait(0.01)

    def release(self, layer_id: int, eid: int) -> None:
        with self.cv:
            self._check()
            layer = self.layers[layer_id]
            entry = layer.entries[eid]
            if not entry.leased:
                raise ValueError("release without acquire")
            if self.device.type == "cuda":
                entry.last_use = torch.cuda.Event()
                entry.last_use.record(torch.cuda.current_stream(self.device))
            entry.leased = False
            layer.required.discard(eid)
            self.cv.notify_all()

    def drain(self, timeout: float = 60) -> None:
        start = time.monotonic()
        with self.cv:
            while self.jobs or self.active:
                self._check()
                if time.monotonic() - start >= timeout:
                    raise TimeoutError("ProMoE queue did not drain")
                self.cv.wait(0.01)
            self._check()

    def reset(self, clear_cache: bool) -> None:
        self.drain()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with self.cv:
            self.step = -1
            self.stats.clear()
            for layer in self.layers.values():
                layer.required.clear()
                layer.precise_step = None
                if clear_cache:
                    layer.entries.clear()

    def snapshot(self) -> dict:
        with self.cv:
            self._check()
            return dict(self.stats) | {"resident_experts_by_layer": {
                str(l): len(v.entries) for l, v in self.layers.items()},
                "cache_capacity_by_layer": {str(l): v.capacity for l, v in self.layers.items()}}

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.worker.join(timeout=60)
        if self.worker.is_alive():
            raise TimeoutError("ProMoE copy worker did not stop")


class ProMoERuntime:
    """CPU prediction worker and copy worker shared across local MoE layers."""

    def __init__(self, blocks: dict, checkpoint_dir: str, distance: int = 2,
                 predict_prefill: bool = False):
        first = next(iter(blocks.values()))
        self.blocks = blocks
        self.distance, self.predict_prefill = distance, predict_prefill
        self.bank = load_predictors(checkpoint_dir, blocks, first.hidden_size,
                                    first.num_experts, distance)
        self.prefetcher = ChunkedPrefetcher(first.device, self._on_copy)
        try:
            for layer, block in blocks.items():
                if block.expert_offload != "cpu":
                    raise ValueError("ProMoE-style requires CPU expert offload")
                if block.experts_kind == "fused":
                    host = {e: (block.local_gate_up_cpu[e], block.local_down_cpu[e])
                            for e in block.local_expert_ids}
                else:
                    host = {e: (block.local_gate_cpu[e], block.local_up_cpu[e], block.local_down_cpu2[e])
                            for e in block.local_expert_ids}
                self.prefetcher.register(layer, host, block._base_expert_cache_capacity)
        except BaseException:
            self.prefetcher.close()
            raise
        self.cv = threading.Condition()
        self.pending = OrderedDict()
        self.active = False
        self.generation = 0
        self.error = None
        self.closed = False
        self.prediction_ms = 0.0
        self.prediction_batches = 0
        self.hooks = []
        self.prediction_thread = threading.Thread(target=self._predict, name="promoe-predict", daemon=True)
        self.prediction_thread.start()

    def _on_copy(self, layer: int, nbytes: int, speculative: bool) -> None:
        self.blocks[layer]._record_h2d_bytes(nbytes, is_prefetch=speculative)

    def observe(self, source: int, x: torch.Tensor, step: int, phase: str) -> None:
        self.prefetcher.begin_step(step)
        if source not in self.bank or (phase != "decode" and not self.predict_prefill):
            return
        with self.cv:
            if self.error is not None:
                raise RuntimeError("ProMoE predictor worker failed") from self.error
            generation = self.generation
        if x.device.type == "cuda":
            cpu = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
            cpu.copy_(x.detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(x.device))
        else:
            cpu, event = x.detach().clone(), None
        with self.cv:
            # Latest input per source bounds backlog under slow CPU prediction.
            self.pending[source] = (cpu, event, step, generation)
            self.cv.notify_all()

    def _predict(self) -> None:
        try:
            with torch.inference_mode():
                while True:
                    with self.cv:
                        while not self.pending and not self.closed:
                            self.cv.wait()
                        if self.closed:
                            return
                        source, (cpu, event, step, generation) = self.pending.popitem(last=False)
                        self.active = True
                    if event is not None:
                        event.synchronize()
                    start = time.monotonic()
                    target, model = self.bank[source]
                    scores = model(cpu.float())
                    block = self.blocks[target]
                    # Match the model's top-k per token, then take the union.
                    predicted = scores.topk(min(block.top_k, scores.shape[-1]), dim=-1).indices.flatten().tolist()
                    owned = self.prefetcher.layers[target].host
                    ids = [e for e in dict.fromkeys(predicted) if e in owned]
                    with self.cv:
                        if generation == self.generation:
                            self.prediction_ms += (time.monotonic() - start) * 1000
                            self.prediction_batches += 1
                            self.prefetcher.predicted(target, ids, step)
                        self.active = False
                        self.cv.notify_all()
        except BaseException as exc:
            with self.cv:
                self.error = exc
                self.active = False
                self.cv.notify_all()

    def drain(self) -> None:
        start = time.monotonic()
        with self.cv:
            while self.pending or self.active:
                if self.error is not None:
                    raise RuntimeError("ProMoE predictor worker failed") from self.error
                if time.monotonic() - start > 60:
                    raise TimeoutError("ProMoE prediction queue did not drain")
                self.cv.wait(0.01)
            if self.error is not None:
                raise RuntimeError("ProMoE predictor worker failed") from self.error
        self.prefetcher.drain()

    def reset(self, clear_cache: bool) -> None:
        self.drain()
        with self.cv:
            self.generation += 1
            self.prediction_ms = 0.0
            self.prediction_batches = 0
        self.prefetcher.reset(clear_cache)

    def summary(self) -> dict:
        return self.prefetcher.snapshot() | {"prediction_cpu_ms": self.prediction_ms,
            "prediction_batches": self.prediction_batches, "prediction_distance": self.distance,
            "predict_prefill": self.predict_prefill, "implementation": "promoe-style-python-ep"}

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.prediction_thread.join(timeout=60)
        self.prefetcher.close()
        if self.prediction_thread.is_alive():
            raise TimeoutError("ProMoE prediction worker did not stop")


def register_prediction_hooks(layers, runtime: ProMoERuntime, get_step_phase: Callable) -> None:
    """Observe pre-attention decoder inputs, matching the profile feature source."""
    def make_hook(source):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise ValueError("Decoder input hidden states are missing")
            step, phase = get_step_phase()
            runtime.observe(source, hidden.reshape(-1, hidden.shape[-1]), step, phase)
        return hook
    for source in runtime.bank:
        runtime.hooks.append(layers[source].register_forward_pre_hook(make_hook(source), with_kwargs=True))
