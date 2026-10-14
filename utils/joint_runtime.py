"""Collective-safe runtime adapter for the experimental joint scheduler.

Exactly one fixed-width tensor all-gather is used per prediction boundary and
one per actual decode-layer boundary. Resolvers and EPIPE never communicate.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist

from utils.joint_scheduler import (
    Demand, ExpertState, JointPlanner, PlannerConfig, RankState, choose_residual,
    nonnegative,
)


def flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


class CostProfile:
    REQUIRED = ("h2d_bytes_per_ms", "h2d_latency_ms", "expert_ms", "item_ms",
                "next_attention_ms", "split_overhead_ms")

    def __init__(self, data, world, raw=None, *, cache_static=False):
        self.cache_static, self._rank_costs = cache_static, {}
        if data.get("schema_version") != 1 or data.get("world_size") != world:
            raise ValueError("joint cost profile schema/world_size mismatch")
        if not isinstance(data.get("calibrated"), bool):
            raise ValueError("joint cost profile must explicitly declare calibrated true/false")
        self.data, self.world = data, world
        self.digest = hashlib.sha256(raw or json.dumps(data, sort_keys=True).encode()).hexdigest()
        self.latency = data["link_latency_ms"]
        self.bandwidth = data["link_bandwidth_bytes_per_ms"]
        for matrix, name in ((self.latency, "latency"), (self.bandwidth, "bandwidth")):
            if len(matrix) != world or any(len(row) != world for row in matrix):
                raise ValueError(f"joint link {name} matrix must match EP world")
            for src, row in enumerate(matrix):
                for dst, value in enumerate(row):
                    if nonnegative(value, name) == 0 and name == "bandwidth" and src != dst:
                        raise ValueError("remote bandwidth must be positive")
        for layer in {0, *(int(key) for key in data.get("layers", {}))}:
            for rank in range(world):
                self.rank_cost(rank, layer)
        for knots in data.get("activation_calibration", {}).values():
            self.validate_knots(knots)

    @classmethod
    def load(cls, path, world, *, cache_static=False):
        path = Path(path).expanduser()
        raw = path.read_bytes()
        return cls(json.loads(raw), world, raw, cache_static=cache_static)

    def rank_cost(self, rank, layer):
        key = (rank, layer)
        if self.cache_static and key in self._rank_costs:
            return self._rank_costs[key]
        cost = dict(self.data["rank_defaults"])
        cost.update(self.data.get("rank_overrides", {}).get(str(rank), {}))
        layer_cost = self.data.get("layers", {}).get(str(layer), {})
        cost.update(layer_cost.get("defaults", {}))
        cost.update(layer_cost.get("rank_overrides", {}).get(str(rank), {}))
        for name in self.REQUIRED:
            nonnegative(cost[name], name)
        if cost["h2d_bytes_per_ms"] <= 0:
            raise ValueError("H2D bandwidth must be positive")
        if self.cache_static:
            self._rank_costs[key] = cost
        return cost

    @staticmethod
    def validate_knots(knots):
        if len(knots) < 2:
            raise ValueError("activation calibration requires at least two knots")
        previous = -1.0
        for x, y in knots:
            if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1 or x <= previous:
                raise ValueError("invalid activation calibration knots")
            previous = x
        if knots[0][0] != 0 or knots[-1][0] != 1:
            raise ValueError("activation calibration must cover [0, 1]")

    def activation_probability(self, scores, top_k, layer, *, validate=True):
        # A uniform-multihot CE target gives q ~= P(expert selected)/router_k.
        # Clipped k*q is a proxy; optional held-out empirical knots calibrate it.
        if validate and (scores.ndim != 2 or not torch.isfinite(scores).all() or bool((scores < 0).any()) or bool((scores > 1).any())):
            raise ValueError("joint prediction must be a finite token/expert probability matrix")
        p = (scores * int(top_k)).clamp(0, 1)
        knots = self.data.get("activation_calibration", {}).get(str(layer))
        if knots is None:
            return p
        out = torch.empty_like(p)
        for i, ((lo, a), (hi, b)) in enumerate(zip(knots, knots[1:])):
            mask = (p >= lo) & ((p <= hi) if i == len(knots) - 2 else (p < hi))
            out[mask] = a + (p[mask] - lo) * ((b - a) / (hi - lo))
        return out


class JointRuntime:
    # Per-expert columns; final row stores cache capacity/headroom and copy queue.
    WIDTH = 10

    def __init__(self, context, profile: CostProfile, config: PlannerConfig, *, cache_static=False, combine_exchange=False):
        self.context, self.profile, self.config = context, profile, config
        self.planner = JointPlanner(config)
        self.cache_static, self._expert_bytes = cache_static, {}
        self.combine_exchange, self._exchange_buffers = combine_exchange, {}
        self.staged_predictions = {}
        self.capture = None
        self.routes: Dict[int, tuple] = {}
        self.forecasts: Dict[int, object] = {}
        self.actual_demands: Dict[int, tuple] = {}
        self.reset()

    @classmethod
    def from_env(cls, context, world):
        path = os.environ.get("SPARSE_JOINT_COST_PROFILE", "").strip()
        if not path:
            raise ValueError("joint scheduler requires SPARSE_JOINT_COST_PROFILE")
        cache_static = flag("SPARSE_JOINT_CACHE_STATIC", "0")
        profile = CostProfile.load(path, world, cache_static=cache_static)
        config = PlannerConfig(
            placement=flag("SPARSE_JOINT_PLACEMENT"),
            epipe_feedback=flag("SPARSE_JOINT_EPIPE_FEEDBACK"),
            route_passes=int(os.environ.get("SPARSE_JOINT_ROUTE_PASSES", "2")),
            min_gain_ms=float(os.environ.get("SPARSE_JOINT_MIN_GAIN_MS", "0.01")),
            action_overhead_ms=float(profile.data.get("action_overhead_ms", 0.0)),
            eviction_risk_weight=float(profile.data.get("eviction_risk_weight", 0.0)),
            route_items=int(os.environ.get("SPARSE_JOINT_ROUTE_ITEMS", "16")),
            prefetch_trials=int(os.environ.get("SPARSE_JOINT_PREFETCH_TRIALS", "32")),
            evaluator=os.environ.get("SPARSE_JOINT_EVALUATOR", "incremental"),
            prediction_candidates_only=flag("SPARSE_JOINT_PREDICTION_CANDIDATES_ONLY", "0"),
            max_evaluations=int(os.environ.get("SPARSE_JOINT_MAX_EVALUATIONS", "0")),
        )
        return cls(context, profile, config, cache_static=cache_static,
                   combine_exchange=flag("SPARSE_JOINT_COMBINE_EXCHANGE", "0"))

    def reset(self):
        self.routes.clear()
        self.forecasts.clear()
        self.actual_demands.clear()
        self.staged_predictions.clear()
        self.stats = dict(prediction_plans=0, execution_plans=0, exchange_calls=0,
                          exchange_bytes=0, exchange_host_ms=0.0, planning_host_ms=0.0,
                          prediction_control_host_ms=0.0, execution_control_host_ms=0.0,
                          residual_control_host_ms=0.0,
                          planned_prefetches=0, planned_prefetch_bytes=0,
                          modeled_gain_ms=0.0, route_item_changes=0,
                          epipe_selected=0, epipe_rejected=0, cost_evaluations=0)
        self.stats.update(full_cost_evaluations=0, incremental_cost_evaluations=0)
        self.stats.update(exchange_payloads=0, combined_boundaries=0)

    def _step(self):
        return int(self.context.runtime_call_id)

    def _snapshot(self, block, window_ms=0.0, budget=0):
        cost = self.profile.rank_cost(int(block.comm_rank), int(block.layer_idx))
        experts = []
        resident_ids = set(int(e) for e in block._cache.resident_ids())
        protected = block._prefetch_protect_ids()
        for eid in range(block.num_experts):
            available = eid in block._local_eid_set
            key = (id(block), eid)
            if not available:
                nbytes = 0
            elif self.cache_static and key in self._expert_bytes:
                nbytes = self._expert_bytes[key]
            else:
                nbytes = int(block._epipe_estimate_expert_nbytes(eid))
                if self.cache_static:
                    self._expert_bytes[key] = nbytes
            resident = eid in resident_ids
            ready = resident and block._local_prefetch_ready(eid)
            pending = 0.0 if ready or not resident else cost["h2d_latency_ms"] + nbytes / cost["h2d_bytes_per_ms"]
            # Unfinished events retain a full-transfer upper proxy; query only,
            # never synchronize the device to estimate remaining copy time.
            last = block._last_used_decode_step.get(eid)
            reuse_risk = 0.0 if last is None else 1.0 / (1.0 + max(0, self._step() - int(last)))
            experts.append(ExpertState(nbytes, available, resident, ready, pending,
                                       eid in protected or (resident and not ready), reuse_risk))
        free = 2**50
        if block.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(block.device)
        queue = 0.0
        for other in self.context.layer_blocks.values():
            if other is block:
                continue
            other_cost = self.profile.rank_cost(int(other.comm_rank), int(other.layer_idx))
            for eid, event in other._prefetch_events.items():
                if not event.query():
                    queue += other_cost["h2d_latency_ms"] + other._epipe_estimate_expert_nbytes(eid) / other_cost["h2d_bytes_per_ms"]
        return RankState(tuple(experts), int(block._cache.capacity), int(free),
                         cost["h2d_bytes_per_ms"], cost["h2d_latency_ms"],
                         cost["expert_ms"], cost["item_ms"], window_ms, queue,
                         int(block._epipe_min_free_mb * 1024**2),
                         int(float(os.environ.get("SPARSE_PREFETCH_MIN_FREE_MB", "32")) * 1024**2),
                         cost["split_overhead_ms"], budget)

    def _exchange(self, block, counts, probabilities, state, candidates=None, validity=None):
        return self._exchange_many([(block, counts, probabilities, state, candidates, validity)])[0]

    def _exchange_many(self, items):
        block = items[0][0]
        n = max(item[0].num_experts for item in items)
        shape = (len(items), n + 1, self.WIDTH)
        key = (block.comm_world, str(block.device), shape)
        # Counts, P(any use), weight bytes, availability, residency, readiness,
        # remaining-time proxy, protection and retention risk are fixed-width.
        if self.cache_static and key in self._exchange_buffers:
            data, gathered = self._exchange_buffers[key]
            data.zero_()
        else:
            data = torch.zeros(shape, dtype=torch.float64, device=block.device)
            gathered = [torch.empty_like(data) for _ in range(block.comm_world)]
            if self.cache_static:
                self._exchange_buffers[key] = (data, gathered)
        for i, (target, counts, probabilities, state, candidates, validity) in enumerate(items):
            size = target.num_experts
            data[i, :size, 0] = counts
            data[i, :size, 1] = probabilities
            if candidates is not None:
                data[i, :size, 9] = candidates
            rows = [[e.nbytes, e.available, e.resident, e.ready, e.pending_ms, e.protected, e.reuse_risk]
                    for e in state.experts]
            data[i, :size, 2:9] = torch.tensor(rows, dtype=torch.float64, device=block.device)
            data[i, size, :7] = torch.tensor([state.capacity, state.free_bytes, state.queue_ms,
                                            state.window_ms, state.prefetch_budget, state.staging_margin_bytes,
                                            state.prefetch_margin_bytes], dtype=torch.float64, device=block.device)
            data[i, size, 8] = 1. if validity is None else validity
        start = time.perf_counter()
        if dist.is_initialized():
            dist.all_gather(gathered, data, group=block.comm_group)
        elif block.comm_world == 1:
            gathered = [data]
        else:
            raise RuntimeError("joint state exchange needs an initialized EP process group")
        # Host planning is explicit and measured, including device-to-host copy.
        values = torch.stack(gathered).cpu().tolist()
        self.stats["exchange_calls"] += 1
        self.stats["exchange_payloads"] += len(items)
        self.stats["exchange_bytes"] += data.numel() * data.element_size() * max(0, block.comm_world - 1)
        self.stats["exchange_host_ms"] += (time.perf_counter() - start) * 1000
        return [self._unpack(item[0], [rank[i] for rank in values]) for i, item in enumerate(items)]

    def _unpack(self, block, values):
        n = block.num_experts
        ranks, count_rows, probability_rows = [], [], []
        for rank, value in enumerate(values):
            row = value[n]
            if row[8] != 1.:
                raise ValueError("joint predictor must contain finite probabilities in [0, 1]")
            cost = self.profile.rank_cost(rank, block.layer_idx)
            states = tuple(ExpertState(int(e[2]), bool(e[3]), bool(e[4]), bool(e[5]),
                                       float(e[6]), bool(e[7]), float(e[8])) for e in value[:n])
            ranks.append(RankState(states, int(row[0]), int(row[1]), cost["h2d_bytes_per_ms"],
                                   cost["h2d_latency_ms"], cost["expert_ms"], cost["item_ms"],
                                   row[3], row[2], int(row[5]), int(row[6]),
                                   cost["split_overhead_ms"], int(row[4])))
            count_rows.append(tuple(e[0] for e in value[:n]))
            probability_rows.append(tuple(e[1] for e in value[:n]))
        eligible = tuple(any(value[e][9] > 0 for value in values) for e in range(n))
        return Demand(tuple(count_rows), tuple(probability_rows), eligible), tuple(ranks)

    def _plan(self, block, demand, ranks, prefetch):
        if self.capture is not None:
            self.capture.append(dict(demand=demand, ranks=ranks, owners=tuple(block.rr_expert_to_rank_cpu),
                                     link_latency=self.profile.latency, link_bandwidth=self.profile.bandwidth,
                                     dispatch_item_bytes=block.hidden_size * torch.empty((), dtype=block.dtype).element_size() + 8 + torch.empty((), dtype=block.dtype).element_size(),
                                     combine_item_bytes=block.hidden_size * torch.empty((), dtype=block.dtype).element_size(),
                                     prefetch=prefetch, config=asdict(self.config)))
        start = time.perf_counter()
        # Exact wire item sizes for ordinary EP: hidden + expert id + weight;
        # return sends hidden states. Original router weights are untouched.
        hidden_bytes = int(block.hidden_size) * torch.empty((), dtype=block.dtype).element_size()
        plan = self.planner.plan(demand, ranks, block.rr_expert_to_rank_cpu,
                                 self.profile.latency, self.profile.bandwidth,
                                 hidden_bytes + 8 + torch.empty((), dtype=block.dtype).element_size(),
                                 hidden_bytes, prefetch=prefetch)
        self.stats["planning_host_ms"] += (time.perf_counter() - start) * 1000
        self.stats["cost_evaluations"] += plan.evaluations
        self.stats["full_cost_evaluations"] += plan.full_evaluations
        self.stats["incremental_cost_evaluations"] += plan.incremental_evaluations
        return plan

    def predict(self, source, target, scores, batch_tokens):
        control_start = time.perf_counter()
        if tuple(scores.shape) != (batch_tokens, target.num_experts):
            raise ValueError("joint predictor shape does not match target experts/source tokens")
        target._begin_group_prefetch_cycle(target.layer_idx, target.layer_idx + 1,
            cycle_key=("joint", self._step(), source.layer_idx, target.layer_idx))
        scores = scores.detach().float()
        validity = None
        if self.cache_static:
            # Transmit the validity marker with metadata so all ranks reject
            # bad scores together, without three scalar CUDA->CPU checks.
            validity = torch.isfinite(scores).all() & (scores >= 0).all() & (scores <= 1).all()
        probability = self.profile.activation_probability(scores, target.top_k, target.layer_idx,
                                                          validate=not self.cache_static)
        if not self.cache_static and not torch.isfinite(probability).all():
            raise ValueError("nonfinite joint predictor probabilities")
        counts = probability.sum(dim=0)
        union = counts.clamp(0, 1)
        candidates = torch.zeros(target.num_experts, dtype=torch.float64, device=target.device)
        k = min(target.num_experts, target._rs_prefetch_predict_topk())
        candidates[torch.topk(scores, k=k, dim=-1).indices.reshape(-1)] = 1
        # Include the cost of predicting/dispatching the source layer only as a
        # provisional horizon; calibrated next-attention time supplies the rest.
        cost = self.profile.rank_cost(source.comm_rank, source.layer_idx)
        window = cost["expert_ms"] * source.top_k + cost["item_ms"] * batch_tokens * source.top_k
        window += self.profile.rank_cost(target.comm_rank, target.layer_idx)["next_attention_ms"]
        budget = target._effective_rs_prefetch_top_m()
        if self.combine_exchange:
            if source.layer_idx in self.staged_predictions:
                raise RuntimeError("duplicate joint prediction at one layer boundary")
            self.staged_predictions[source.layer_idx] = (self._step(), target, counts, union, candidates,
                                                         validity, window, budget)
            self.stats["prediction_control_host_ms"] += (time.perf_counter() - control_start) * 1000
            return
        state = self._snapshot(target, window, budget)
        demand, ranks = self._exchange(target, counts, union, state, candidates, validity)
        self._finish_prediction(target, demand, ranks, state)
        self.stats["prediction_control_host_ms"] += (time.perf_counter() - control_start) * 1000

    def _finish_prediction(self, target, demand, ranks, state):
        plan = self._plan(target, demand, ranks, prefetch=True)
        self.forecasts[target.layer_idx] = (self._step(), plan)
        self.stats["prediction_plans"] += 1
        self.stats["modeled_gain_ms"] += max(0.0, plan.before.total_ms - plan.after.total_ms)
        local = [a for a in plan.admissions if a.rank == target.comm_rank]
        self.stats["planned_prefetches"] += len(local)
        self.stats["planned_prefetch_bytes"] += sum(state.experts[a.expert].nbytes for a in local)
        target._admit_prefetch_plan([(a.expert, a.victim) for a in local], group_end=target.layer_idx + 1)
        return plan

    def verify(self, block):
        features = tuple(getattr(module, "rs_feature_mode", "hidden") for module in self.context.rs_bank)
        if self.combine_exchange and any(mode != "hidden" for mode in features):
            raise ValueError("combined exchange currently requires hidden-only adjacent RS features")
        signature = (self.profile.digest, asdict(self.config), self.cache_static, self.combine_exchange, features,
                     self.context.rs_bank_layout,
                     tuple(self.context.rs_bank_source_layers),
                     tuple(sorted(self.context.layer_blocks)), block.num_experts)
        signatures = [None] * block.comm_world
        if dist.is_initialized():
            dist.all_gather_object(signatures, signature, group=block.comm_group)
        else:
            signatures = [signature]
        if any(value != signature for value in signatures):
            raise ValueError("joint config, cost profile or RS layout differs across EP ranks")

    def route_for(self, block, source, expert, *, prefetch=False):
        if prefetch:
            step, plan = self.forecasts.get(block.layer_idx, (-1, None))
            return None if step != self._step() or plan is None else int(plan.routes[source][expert])
        step, table = self.routes.get(block.layer_idx, (-1, None))
        return None if step != self._step() or table is None else int(table[source, expert].item())

    def prepare_execution(self, block, indices):
        control_start = time.perf_counter()
        counts = torch.bincount(indices.reshape(-1).long(), minlength=block.num_experts).double()
        state = self._snapshot(block)
        staged = self.staged_predictions.pop(block.layer_idx, None)
        prediction_ms = 0.
        if staged is None:
            demand, ranks = self._exchange(block, counts, (counts > 0).double(), state)
        else:
            step, target, forecast_counts, probabilities, candidates, validity, window, budget = staged
            if step != self._step():
                raise RuntimeError("stale staged joint prediction")
            target_state = self._snapshot(target, window, budget)
            actual, forecast = self._exchange_many([
                (block, counts, (counts > 0).double(), state, None, None),
                (target, forecast_counts, probabilities, target_state, candidates, validity)])
            demand, ranks = actual
            prediction_start = time.perf_counter()
            predicted = self._finish_prediction(target, forecast[0], forecast[1], target_state)
            # The actual snapshot preceded new target-layer prefetches. Account
            # for their copy queue and bytes conservatively before dispatch.
            added_bytes, added_ms = [0] * block.comm_world, [0.] * block.comm_world
            for action in predicted.admissions:
                expert = forecast[1][action.rank].experts[action.expert]
                added_bytes[action.rank] += expert.nbytes
                added_ms[action.rank] += forecast[1][action.rank].transfer_ms(action.expert)
            ranks = tuple(replace(rank, free_bytes=max(0, rank.free_bytes - added_bytes[r]),
                                  queue_ms=rank.queue_ms + added_ms[r]) for r, rank in enumerate(ranks))
            prediction_ms = (time.perf_counter() - prediction_start) * 1000
            self.stats["prediction_control_host_ms"] += prediction_ms
            self.stats["combined_boundaries"] += 1
        plan = self._plan(block, demand, ranks, prefetch=False)
        table = torch.tensor(plan.routes, dtype=torch.long, device=block.device)
        self.routes[block.layer_idx] = (self._step(), table)
        needed = {e for src in range(block.comm_world) for e in range(block.num_experts)
                  if demand.counts[src][e] > 0 and plan.routes[src][e] == block.comm_rank}
        self.actual_demands[block.layer_idx] = (self._step(), needed)
        self.stats["execution_plans"] += 1
        row = table[block.comm_rank]
        owners = torch.tensor(block.rr_expert_to_rank_cpu, device=block.device)
        self.stats["route_item_changes"] += int(counts[row != owners].sum().item())
        self.stats["execution_control_host_ms"] += (time.perf_counter() - control_start) * 1000 - prediction_ms

    def actual_requested_ids(self, block):
        step, ids = self.actual_demands.get(block.layer_idx, (-1, set()))
        if step != self._step():
            raise RuntimeError("joint actual demand missing for current decode step")
        return set(ids)

    def destinations(self, block, expert_ids):
        step, table = self.routes.get(block.layer_idx, (-1, None))
        if step != self._step() or table is None:
            raise RuntimeError("joint execution route missing at synchronized layer boundary")
        return table[block.comm_rank][expert_ids]

    def residual(self, block, segments, ready_indices, late_indices, demand_indices):
        control_start = time.perf_counter()
        state = self._snapshot(block)
        ready = {int(segments[i][0]): int(segments[i][2] - segments[i][1]) for i in ready_indices}
        candidates = {int(segments[i][0]): "late" for i in late_indices}
        candidates.update({int(segments[i][0]): "demand" for i in demand_indices})
        selected = choose_residual(state, ready, candidates)
        self.stats["epipe_selected" if selected is not None else "epipe_rejected"] += 1
        self.stats["residual_control_host_ms"] += (time.perf_counter() - control_start) * 1000
        return selected

    def summary(self):
        return {**self.stats, "profile_sha256": self.profile.digest,
                "cache_static": self.cache_static, "combine_exchange": self.combine_exchange,
                "cost_profile_calibrated": self.profile.data["calibrated"],
                "activation_probability_mode": "empirical_knots" if self.profile.data.get("activation_calibration") else "k_times_softmax_proxy",
                "planner_config": asdict(self.config), "model_semantics": "probability-weighted layer latency proxy"}
