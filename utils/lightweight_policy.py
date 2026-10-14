"""Local readiness policy for the preserved asynchronous SPARSE runtime.

No distributed communication, device-to-host transfer or full-plan search lives
here. Costs are explicit estimates; correctness stays with the original executor.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time


@dataclass(frozen=True)
class Costs:
    bandwidth: float
    latency: float
    expert: float
    item: float
    attention: float
    split: float

    @classmethod
    def from_dict(cls, values):
        keys = ('h2d_bytes_per_ms', 'h2d_latency_ms', 'expert_ms', 'item_ms',
                'next_attention_ms', 'split_overhead_ms')
        numbers = [float(values[key]) for key in keys]
        if any(not math.isfinite(n) or n < 0 for n in numbers) or numbers[0] <= 0:
            raise ValueError('local policy costs must be finite/nonnegative, with positive H2D bandwidth')
        return cls(*numbers)

    def compute(self, groups, items):
        return self.expert * groups + self.item * items


class CostProfile:
    def __init__(self, path, world):
        raw = Path(path).read_bytes()
        self.data = json.loads(raw)
        if (self.data.get('schema_version') != 1 or self.data.get('world_size') != world
                or not isinstance(self.data.get('calibrated'), bool)):
            raise ValueError('local cost profile schema/world/calibration declaration mismatch')
        self.digest = hashlib.sha256(raw).hexdigest()
        self.cache = {}

    def cost(self, rank, layer):
        key = (rank, layer)
        if key not in self.cache:
            values = dict(self.data['rank_defaults'])
            values.update(self.data.get('rank_overrides', {}).get(str(rank), {}))
            layer_values = self.data.get('layers', {}).get(str(layer), {})
            values.update(layer_values.get('defaults', {}))
            values.update(layer_values.get('rank_overrides', {}).get(str(rank), {}))
            self.cache[key] = Costs.from_dict(values)
        return self.cache[key]


class LocalPolicy:
    def __init__(self, profile, rank, layer, expert_bytes, local_candidates, *,
                 prefetch=True, route=True, feedback=True):
        self.profile, self.cost = profile, profile.cost(rank, layer)
        self.rank, self.layer = rank, layer
        self.transfer = {eid: self.cost.latency + size / self.cost.bandwidth
                         for eid, size in expert_bytes.items()}
        self.local_candidates = frozenset(local_candidates)
        self.prefetch, self.route, self.feedback = prefetch, route, feedback
        self.reset()

    def reset(self):
        self.ready_ms = 0.
        self.window_ms = 0.
        self.route_key, self.route_table = None, None
        self.stats = dict(prefetch_calls=0, candidate_scores=0, feedback_updates=0,
                          route_calls=0, route_table_rebuilds=0, residual_calls=0,
                          prefetch_host_ms=0., route_host_ms=0., residual_host_ms=0.)

    def scores(self, values, residents):
        if not self.prefetch:
            return values
        start = time.perf_counter()
        out = {}
        cover = self.ready_ms if self.feedback else 0.
        for eid, score in values.items():
            # Keep resident retention scores in their existing units. Only
            # missing candidates are discounted by already coverable loading.
            if eid in residents:
                out[eid] = score
                continue
            transfer = self.transfer[eid]
            window = min(self.window_ms, transfer)
            uncovered = max(0., transfer - cover)
            gain = min(window, uncovered)
            # Normalization preserves the existing cache-safe score thresholds.
            # A missing estimate keeps the original score, rather than disabling RS.
            out[eid] = score * gain / window if window > 0 else score
        self.stats['prefetch_calls'] += 1
        self.stats['candidate_scores'] += len(values)
        self.stats['prefetch_host_ms'] += (time.perf_counter() - start) * 1000
        return out

    def observe_ready(self, groups, items):
        if self.feedback:
            self.ready_ms = self.cost.compute(groups, items)
            self.stats['feedback_updates'] += 1

    def residual(self, ready_groups, ready_items, candidates):
        start = time.perf_counter()
        ready = self.cost.compute(ready_groups, ready_items)
        best, benefit = None, self.cost.split
        # candidates are (segment index, expert ID, kind), in the executor's
        # original late-before-demand order. Equal estimates preserve that order.
        for index, eid, kind in candidates:
            overlap = min(ready, self.transfer[eid])
            if overlap > benefit:
                best, benefit = (index, kind), overlap
        self.stats['residual_calls'] += 1
        self.stats['residual_host_ms'] += (time.perf_counter() - start) * 1000
        return best

    def summary(self):
        return {**self.stats, 'profile_sha256': self.profile.digest,
                'calibrated': self.profile.data['calibrated'],
                'prefetch': self.prefetch, 'route': self.route, 'feedback': self.feedback,
                'metadata_collectives': 0, 'device_to_host_copies': 0}
