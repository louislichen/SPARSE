"""Deterministic joint placement, prefetch and residual-wait planner.

All times are milliseconds. This is a bounded greedy planner over a calibrated
cost *proxy*, not an optimal scheduler or an estimate of E[max latency]. Runtime
collectives and expert arithmetic belong to joint_runtime/eval_core.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple


def nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class ExpertState:
    nbytes: int
    available: bool
    resident: bool = False
    ready: bool = False
    pending_ms: float = 0.0
    protected: bool = False
    reuse_risk: float = 0.0


@dataclass(frozen=True)
class RankState:
    experts: Tuple[ExpertState, ...]
    capacity: int
    free_bytes: int
    h2d_bytes_per_ms: float
    h2d_latency_ms: float
    expert_ms: float
    item_ms: float
    window_ms: float = 0.0
    queue_ms: float = 0.0
    staging_margin_bytes: int = 0
    prefetch_margin_bytes: int = 0
    split_overhead_ms: float = 0.0
    prefetch_budget: int = 0

    def transfer_ms(self, eid: int) -> float:
        return self.h2d_latency_ms + self.experts[eid].nbytes / self.h2d_bytes_per_ms

    def compute_ms(self, count: float, probability: float = 1.0) -> float:
        return self.expert_ms * probability + self.item_ms * count


@dataclass(frozen=True)
class Demand:
    # Expected token/expert item count and probability of any use, by source.
    counts: Tuple[Tuple[float, ...], ...]
    probabilities: Tuple[Tuple[float, ...], ...]
    prefetch_candidates: Optional[Tuple[bool, ...]] = None


@dataclass(frozen=True)
class Evaluation:
    total_ms: float
    rank_ms: Tuple[float, ...]
    hidden_ms: Tuple[float, ...]
    dispatch_ms: float
    combine_ms: float


@dataclass(frozen=True)
class Admission:
    rank: int
    expert: int
    victim: Optional[int]
    gain_ms: float
    finish_ms: float


@dataclass(frozen=True)
class JointPlan:
    routes: Tuple[Tuple[int, ...], ...]
    admissions: Tuple[Admission, ...]
    before: Evaluation
    after: Evaluation
    evaluations: int
    full_evaluations: int = 0
    incremental_evaluations: int = 0


@dataclass(frozen=True)
class PlannerConfig:
    placement: bool = True
    epipe_feedback: bool = True
    route_passes: int = 2
    min_gain_ms: float = 0.01
    action_overhead_ms: float = 0.0
    eviction_risk_weight: float = 0.0
    route_items: int = 16
    prefetch_trials: int = 32
    evaluator: str = "incremental"
    prediction_candidates_only: bool = False
    max_evaluations: int = 0

    def __post_init__(self):
        if self.evaluator not in ("reference", "incremental"):
            raise ValueError("evaluator must be reference or incremental")
        if self.route_passes < 0:
            raise ValueError("route_passes must be nonnegative")
        if self.max_evaluations < 0 or (self.max_evaluations and self.max_evaluations < 3):
            raise ValueError("max_evaluations must be zero or at least three")
        if self.route_items < 1 or self.prefetch_trials < 1:
            raise ValueError("route_items and prefetch_trials must be positive")
        for name in ("min_gain_ms", "action_overhead_ms", "eviction_risk_weight"):
            nonnegative(getattr(self, name), name)


class JointPlanner:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.evaluations = 0
        self.full_evaluations = 0
        self.incremental_evaluations = 0

    def validate(self, demand, ranks, owners, link_latency, link_bandwidth):
        world, experts = len(ranks), len(owners)
        if world < 1 or experts < 1:
            raise ValueError("nonempty ranks and experts are required")
        if len(demand.counts) != world or len(demand.probabilities) != world:
            raise ValueError("demand must have one row per source rank")
        if demand.prefetch_candidates is not None and len(demand.prefetch_candidates) != experts:
            raise ValueError("prefetch candidate mask width must equal expert count")
        for counts, probs in zip(demand.counts, demand.probabilities):
            if len(counts) != experts or len(probs) != experts:
                raise ValueError("demand width must equal expert count")
            for count, prob in zip(counts, probs):
                nonnegative(count, "demand count")
                if not math.isfinite(prob) or not 0 <= prob <= 1:
                    raise ValueError("activation probabilities must lie in [0, 1]")
                if count > 0 and prob <= 0:
                    raise ValueError("positive demand needs positive activation probability")
        for eid, owner in enumerate(owners):
            if not 0 <= owner < world or not ranks[owner].experts[eid].available:
                raise ValueError("each expert needs an available authoritative owner")
        for rank in ranks:
            if len(rank.experts) != experts or rank.capacity < 0 or rank.free_bytes < 0 or rank.prefetch_budget < 0:
                raise ValueError("invalid rank cache or expert dimensions")
            if sum(e.resident for e in rank.experts) > rank.capacity:
                raise ValueError("resident count exceeds cache capacity")
            if nonnegative(rank.h2d_bytes_per_ms, "H2D bandwidth") == 0:
                raise ValueError("H2D bandwidth must be positive")
            for name in ("h2d_latency_ms", "expert_ms", "item_ms", "window_ms", "queue_ms", "split_overhead_ms"):
                nonnegative(getattr(rank, name), name)
            if rank.staging_margin_bytes < 0 or rank.prefetch_margin_bytes < 0:
                raise ValueError("staging margin must be nonnegative")
            for state in rank.experts:
                if state.nbytes < 0 or (state.available and state.nbytes <= 0):
                    raise ValueError("available expert needs positive weight bytes")
                if state.ready and not state.resident:
                    raise ValueError("ready expert must be resident")
                if state.resident and not state.available:
                    raise ValueError("resident expert must be available")
                nonnegative(state.pending_ms, "pending time")
                if not 0 <= state.reuse_risk <= 1:
                    raise ValueError("reuse risk must lie in [0, 1]")
        for matrix, name in ((link_latency, "link latency"), (link_bandwidth, "link bandwidth")):
            if len(matrix) != world or any(len(row) != world for row in matrix):
                raise ValueError(f"{name} must be a square rank matrix")
            for src, row in enumerate(matrix):
                for dst, value in enumerate(row):
                    nonnegative(value, name)
                    if name == "link bandwidth" and src != dst and value == 0:
                        raise ValueError("remote link bandwidth must be positive")

    def evaluate(self, demand, ranks, routes, link_latency, link_bandwidth,
                 dispatch_item_bytes, combine_item_bytes):
        self.evaluations += 1
        self.full_evaluations += 1
        world, experts = len(ranks), len(ranks[0].experts)
        counts = [[0.0] * experts for _ in ranks]
        probs = [[0.0] * experts for _ in ranks]
        edge_counts = [[0.0] * world for _ in ranks]
        for src in range(world):
            for eid in range(experts):
                count = demand.counts[src][eid]
                if count <= 0:
                    continue
                dst = routes[src][eid]
                if not ranks[dst].experts[eid].available:
                    raise ValueError("route selected an unavailable expert")
                counts[dst][eid] += count
                # Conservative union bound; no token-independence assumption.
                probs[dst][eid] = min(1.0, probs[dst][eid] + demand.probabilities[src][eid])
                if src != dst:
                    edge_counts[src][dst] += count
        send_d, recv_d, send_c, recv_c = ([0.0] * world for _ in range(4))
        for src in range(world):
            for dst in range(world):
                count = edge_counts[src][dst]
                if count <= 0:
                    continue
                dispatch = link_latency[src][dst] + count * dispatch_item_bytes / link_bandwidth[src][dst]
                combine = link_latency[dst][src] + count * combine_item_bytes / link_bandwidth[dst][src]
                send_d[src] += dispatch; recv_d[dst] += dispatch
                send_c[dst] += combine; recv_c[src] += combine
        rank_ms, hidden = [], []
        for r, state in enumerate(ranks):
            compute, ready_compute, load, absent_probability = 0.0, 0.0, 0.0, 0.0
            overlap_candidates = []
            for eid, count in enumerate(counts[r]):
                if count <= 0:
                    continue
                p = probs[r][eid]
                expert = state.experts[eid]
                compute += state.compute_ms(count, p)
                if expert.ready or (expert.resident and expert.pending_ms <= state.window_ms):
                    ready_compute += state.compute_ms(count, p)
                else:
                    transfer = max(0.0, expert.pending_ms - state.window_ms) if expert.resident else state.transfer_ms(eid)
                    load += p * transfer
                    if not expert.resident:
                        absent_probability = min(1.0, absent_probability + p)
                    if expert.resident or state.free_bytes >= expert.nbytes + state.staging_margin_bytes:
                        overlap_candidates.append((p, transfer))
            queue_wait = max(0.0, state.queue_ms - state.window_ms) * absent_probability
            benefit = 0.0
            if self.config.epipe_feedback and ready_compute > 0:
                for p, transfer in overlap_candidates:
                    candidate = p * min(ready_compute, transfer) - state.split_overhead_ms
                    benefit = max(benefit, candidate)
            benefit = max(0.0, min(benefit, load))
            rank_ms.append(compute + load + queue_wait - benefit)
            hidden.append(benefit)
        dispatch = max(send_d + recv_d, default=0.0)
        combine = max(send_c + recv_c, default=0.0)
        return Evaluation(dispatch + max(rank_ms, default=0.0) + combine,
                          tuple(rank_ms), tuple(hidden), dispatch, combine)

    def _budget_exhausted(self):
        return self.config.max_evaluations > 0 and self.evaluations >= self.config.max_evaluations

    def _route(self, demand, ranks, routes, evaluate, link_latency, link_bandwidth,
               dispatch_item_bytes, combine_item_bytes, current=None, candidates=None):
        if current is None:
            current = evaluate(ranks, routes)
        if not self.config.placement or self._budget_exhausted():
            return routes, current
        # A one-item move can worsen the current maximum when several ranks
        # have tied misses. Evaluate a simultaneous readiness-based proposal
        # first, then refine it with bounded coordinate moves.
        seed = [list(row) for row in routes]
        for src in range(len(ranks)):
            for eid, count in enumerate(demand.counts[src]):
                if count <= 0 or (candidates is not None and not candidates[eid]):
                    continue
                p = demand.probabilities[src][eid]
                def local_cost(dst):
                    rank, expert = ranks[dst], ranks[dst].experts[eid]
                    ready = expert.ready or (expert.resident and expert.pending_ms <= rank.window_ms)
                    load = 0.0 if ready else (max(0.0, expert.pending_ms - rank.window_ms)
                                             if expert.resident else rank.transfer_ms(eid) + max(0.0, rank.queue_ms - rank.window_ms))
                    if src == dst:
                        network = 0.0
                    else:
                        network = (link_latency[src][dst] + count * dispatch_item_bytes / link_bandwidth[src][dst]
                                   + link_latency[dst][src] + count * combine_item_bytes / link_bandwidth[dst][src])
                    return p * load + rank.compute_ms(count, p) + network
                available = [dst for dst, rank in enumerate(ranks) if rank.experts[eid].available]
                seed[src][eid] = min(available, key=lambda dst: (local_cost(dst), dst != routes[src][eid], dst))
        result = evaluate(ranks, seed)
        if result.total_ms + self.config.min_gain_ms < current.total_ms:
            routes, current = seed, result
        delta = None
        if self.config.evaluator == "incremental" and len(ranks) > 1:
            delta = _RouteEvaluation(demand, ranks, routes, self.config.epipe_feedback,
                                     link_latency, link_bandwidth, dispatch_item_bytes, combine_item_bytes)
        order = sorted(((demand.counts[s][e], s, e) for s in range(len(ranks))
                        for e in range(len(ranks[0].experts)) if demand.counts[s][e] > 0
                        and (candidates is None or candidates[e])),
                       key=lambda x: (-x[0], x[1], x[2]))[:self.config.route_items]
        for _ in range(self.config.route_passes):
            changed = False
            for _, src, eid in order:
                best_dst, best = routes[src][eid], current
                for dst, rank in enumerate(ranks):
                    if self._budget_exhausted():
                        break
                    if dst == routes[src][eid] or not rank.experts[eid].available:
                        continue
                    if delta is None:
                        trial = [list(row) for row in routes]
                        trial[src][eid] = dst
                        result = evaluate(ranks, trial)
                    else:
                        result = delta.move(src, eid, dst)
                        self.evaluations += 1
                        self.incremental_evaluations += 1
                    if result.total_ms + self.config.min_gain_ms < best.total_ms:
                        best_dst, best = dst, result
                if best_dst != routes[src][eid]:
                    if delta is not None:
                        delta.move(src, eid, best_dst, commit=True)
                    routes[src][eid] = best_dst
                    current, changed = best, True
            if not changed:
                break
        return routes, current

    def plan(self, demand: Demand, ranks: Sequence[RankState], owners: Sequence[int],
             link_latency: Sequence[Sequence[float]], link_bandwidth: Sequence[Sequence[float]],
             dispatch_item_bytes: int, combine_item_bytes: int, *, prefetch: bool = True) -> JointPlan:
        self.validate(demand, ranks, owners, link_latency, link_bandwidth)
        if dispatch_item_bytes <= 0 or combine_item_bytes <= 0:
            raise ValueError("activation payload bytes must be positive")
        self.evaluations = 0
        self.full_evaluations = 0
        self.incremental_evaluations = 0
        evaluate = lambda states, assignment: self.evaluate(
            demand, states, assignment, link_latency, link_bandwidth, dispatch_item_bytes, combine_item_bytes)
        states = list(ranks)
        routes = [list(owners) for _ in ranks]
        before = evaluate(states, routes)
        route_candidates = demand.prefetch_candidates if prefetch and self.config.prediction_candidates_only else None
        route = lambda states, assignment, current=None: self._route(
            demand, states, assignment, evaluate, link_latency, link_bandwidth,
            dispatch_item_bytes, combine_item_bytes, current, route_candidates)
        reuse_score = self.config.evaluator == "incremental" or self.config.max_evaluations > 0
        routes, current = route(states, routes, before if reuse_score else None)
        admissions = []
        cursor = [rank.queue_ms for rank in ranks]
        issued = [0] * len(ranks)
        total_budget = sum(rank.prefetch_budget for rank in ranks) if prefetch else 0
        for _ in range(total_budget):
            if self._budget_exhausted():
                break
            best = None
            best_gain = self.config.min_gain_ms
            choices = []
            for dst, rank in enumerate(states):
                if issued[dst] >= rank.prefetch_budget:
                    continue
                resident = sum(e.resident for e in rank.experts)
                for eid, expert in enumerate(rank.experts):
                    if demand.prefetch_candidates is not None and not demand.prefetch_candidates[eid]:
                        continue
                    if not expert.available or expert.resident or not any(row[eid] > 0 for row in demand.counts):
                        continue
                    victims = [None] if resident < rank.capacity else [
                        v for v, old in enumerate(rank.experts) if old.resident and not old.protected]
                    for victim in victims:
                        old = rank.experts[victim] if victim is not None else None
                        free = rank.free_bytes + (old.nbytes if old is not None else 0) - expert.nbytes
                        if free < rank.prefetch_margin_bytes:
                            continue
                        penalty = self.config.action_overhead_ms
                        if old is not None:
                            penalty += self.config.eviction_risk_weight * old.reuse_risk * rank.transfer_ms(victim)
                        probability = min(1.0, sum(row[eid] for row in demand.probabilities))
                        upper = probability * rank.transfer_ms(eid) - penalty
                        if victim is not None:
                            victim_probability = min(1.0, sum(row[victim] for row in demand.probabilities))
                            upper -= victim_probability * rank.transfer_ms(victim)
                        choices.append((-upper, dst, eid, -1 if victim is None else victim, free, penalty))
            # Shortlisting bounds planning work; acceptance always uses the
            # complete placement/eviction/overlap proxy, never this heuristic.
            for _, dst, eid, victim_key, free, penalty in sorted(choices)[:self.config.prefetch_trials]:
                if self._budget_exhausted():
                    break
                rank = states[dst]
                victim = None if victim_key == -1 else victim_key
                finish = cursor[dst] + rank.transfer_ms(eid)
                remaining = max(0.0, finish - rank.window_ms)
                experts = list(rank.experts)
                if victim is not None:
                    experts[victim] = replace(experts[victim], resident=False, ready=False, pending_ms=0.0)
                experts[eid] = replace(experts[eid], resident=True, ready=(remaining == 0),
                                       pending_ms=finish, protected=True)
                trial_states = list(states)
                trial_states[dst] = replace(rank, experts=tuple(experts), free_bytes=free, queue_ms=finish)
                trial_routes = [list(row) for row in routes]
                result = evaluate(trial_states, trial_routes)
                if self.config.placement and not self._budget_exhausted():
                    # Try moving this expert's complete demand together; all
                    # other placements stay fixed during candidate scoring.
                    together = [list(row) for row in routes]
                    for src in range(len(ranks)):
                        if demand.counts[src][eid] > 0:
                            together[src][eid] = dst
                    alternative = evaluate(trial_states, together)
                    if alternative.total_ms < result.total_ms:
                        trial_routes, result = together, alternative
                gain = current.total_ms - result.total_ms - penalty
                if gain > best_gain:
                    best_gain = gain
                    best = (dst, eid, victim, finish, trial_states, trial_routes, penalty, result)
            if best is None:
                break
            previous = current.total_ms
            dst, eid, victim, finish, states, routes, penalty, result = best
            routes, current = route(states, routes, result if reuse_score else None)
            cursor[dst], issued[dst] = finish, issued[dst] + 1
            admissions.append(Admission(dst, eid, victim, previous - current.total_ms - penalty, finish))
        return JointPlan(tuple(tuple(row) for row in routes), tuple(admissions),
                         before, current, self.evaluations,
                         self.full_evaluations, self.incremental_evaluations)


class _RankEvaluation:
    """Cached rank totals and top two probabilities per transfer-time group.

    EPIPE's overlap proxy is max(p * min(ready_compute, transfer)). Grouping
    equal transfer times makes a one-expert routing trial independent of E for
    uniform-size experts. Variable-size experts retain the exact same proxy.
    """
    def __init__(self, state, counts, probs, feedback):
        self.state, self.counts, self.probs, self.feedback = state, counts, probs, feedback
        self.ready, self.transfer, self.group = [], [], []
        self.members, self.tops = {}, {}
        for eid, expert in enumerate(state.experts):
            ready = expert.ready or (expert.resident and expert.pending_ms <= state.window_ms)
            transfer = (max(0., expert.pending_ms - state.window_ms)
                        if expert.resident else state.transfer_ms(eid))
            feasible = expert.resident or state.free_bytes >= expert.nbytes + state.staging_margin_bytes
            group = transfer if expert.available and not ready and feasible else None
            self.ready.append(ready)
            self.transfer.append(transfer)
            self.group.append(group)
            if group is not None:
                self.members.setdefault(group, []).append(eid)
        self.parts = [self.contribution(eid, n, p) for eid, (n, p) in enumerate(zip(counts, probs))]
        self.totals = [sum(part[i] for part in self.parts) for i in range(4)]
        self.active = sum(n > 0 for n in counts)
        for group in self.members:
            self.refresh(group)
        self.queue = max(0., state.queue_ms - state.window_ms)

    def contribution(self, eid, count, probability):
        if count <= 0:
            return (0., 0., 0., 0.)
        compute = self.state.expert_ms * probability + self.state.item_ms * count
        if self.ready[eid]:
            return (compute, compute, 0., 0.)
        return (compute, 0., probability * self.transfer[eid],
                0. if self.state.experts[eid].resident else probability)

    def refresh(self, group):
        first, second, first_id = 0., 0., -1
        for eid in self.members[group]:
            p = self.probs[eid] if self.counts[eid] > 0 else 0.
            if p > first:
                second, first, first_id = first, p, eid
            elif p > second:
                second = p
        self.tops[group] = (first, second, first_id)

    def cost(self, eid=None, count=0., probability=0., commit=False):
        totals = self.totals
        if eid is not None:
            part = self.contribution(eid, count, probability)
            totals = [x + new - old for x, new, old in zip(totals, part, self.parts[eid])]
            if self.active - int(self.counts[eid] > 0) + int(count > 0) == 0:
                totals = [0., 0., 0., 0.]
        compute, ready, load, absent = totals
        hidden = 0.
        if self.feedback and ready > 0:
            changed_group = None if eid is None else self.group[eid]
            for group, (first, second, first_id) in self.tops.items():
                if group == changed_group:
                    p = max(second if first_id == eid else first, probability if count > 0 else 0.)
                else:
                    p = first
                hidden = max(hidden, p * min(ready, group) - self.state.split_overhead_ms)
            hidden = max(0., min(hidden, load))
        result = compute + load + self.queue * min(1., max(0., absent)) - hidden
        if commit and eid is not None:
            self.active += int(count > 0) - int(self.counts[eid] > 0)
            self.parts[eid], self.counts[eid], self.probs[eid] = part, count, probability
            self.totals = totals
            group = self.group[eid]
            if group is not None:
                self.refresh(group)
        return result, hidden


class _RouteEvaluation:
    """Evaluate a source/expert move using only affected rank and link totals."""
    def __init__(self, demand, ranks, routes, feedback, latency, bandwidth, dispatch_bytes, combine_bytes):
        self.demand, self.ranks = demand, ranks
        self.routes = [list(row) for row in routes]
        self.latency, self.bandwidth = latency, bandwidth
        self.dispatch_bytes, self.combine_bytes = dispatch_bytes, combine_bytes
        self.world, self.experts = len(ranks), len(ranks[0].experts)
        counts = [[0.] * self.experts for _ in ranks]
        probs = [[0.] * self.experts for _ in ranks]
        self.edges = [[0.] * self.world for _ in ranks]
        self.edge_ids = [[set() for _ in ranks] for _ in ranks]
        for src in range(self.world):
            for eid, n in enumerate(demand.counts[src]):
                if n <= 0:
                    continue
                dst = routes[src][eid]
                counts[dst][eid] += n
                probs[dst][eid] = min(1., probs[dst][eid] + demand.probabilities[src][eid])
                if src != dst:
                    self.edges[src][dst] += n
                    self.edge_ids[src][dst].add(eid)
        self.rank_costs = [_RankEvaluation(r, n, p, feedback) for r, n, p in zip(ranks, counts, probs)]
        self.rank_ms, self.hidden = [], []
        for rank in self.rank_costs:
            cost, hidden = rank.cost()
            self.rank_ms.append(cost)
            self.hidden.append(hidden)
        self.send_d, self.recv_d, self.send_c, self.recv_c = ([0.] * self.world for _ in range(4))
        for src in range(self.world):
            for dst in range(self.world):
                d, c = self.edge_cost(src, dst, self.edges[src][dst])
                self.send_d[src] += d; self.recv_d[dst] += d
                self.send_c[dst] += c; self.recv_c[src] += c

    def edge_cost(self, src, dst, count):
        if src == dst or count <= 0:
            return 0., 0.
        return (self.latency[src][dst] + count * self.dispatch_bytes / self.bandwidth[src][dst],
                self.latency[dst][src] + count * self.combine_bytes / self.bandwidth[dst][src])

    def aggregate(self, eid, rank, moved_src, moved_dst):
        count, probability = 0., 0.
        for src in range(self.world):
            dst = moved_dst if src == moved_src else self.routes[src][eid]
            n = self.demand.counts[src][eid]
            if dst == rank and n > 0:
                count += n
                probability = min(1., probability + self.demand.probabilities[src][eid])
        return count, probability

    def move(self, src, eid, dst, *, commit=False):
        previous = self.routes[src][eid]
        if previous == dst:
            dispatch, combine = max(self.send_d + self.recv_d), max(self.send_c + self.recv_c)
            return Evaluation(dispatch + max(self.rank_ms) + combine, tuple(self.rank_ms),
                              tuple(self.hidden), dispatch, combine)
        if not self.ranks[dst].experts[eid].available:
            raise ValueError("route selected an unavailable expert")
        rank_ms, hidden = list(self.rank_ms), list(self.hidden)
        for rank in (previous, dst):
            n, p = self.aggregate(eid, rank, src, dst)
            rank_ms[rank], hidden[rank] = self.rank_costs[rank].cost(eid, n, p, commit=commit)
        send_d, recv_d, send_c, recv_c = (list(x) for x in (self.send_d, self.recv_d, self.send_c, self.recv_c))
        n = self.demand.counts[src][eid]
        if n > 0:
            for rank, sign in ((previous, -1), (dst, 1)):
                if src == rank:
                    continue
                old_count = self.edges[src][rank]
                new_count = old_count + sign * n
                if sign < 0 and len(self.edge_ids[src][rank]) == 1:
                    new_count = 0.
                old_d, old_c = self.edge_cost(src, rank, old_count)
                new_d, new_c = self.edge_cost(src, rank, new_count)
                send_d[src] += new_d - old_d; recv_d[rank] += new_d - old_d
                send_c[rank] += new_c - old_c; recv_c[src] += new_c - old_c
                if commit:
                    self.edges[src][rank] = new_count
                    if sign < 0:
                        self.edge_ids[src][rank].remove(eid)
                    else:
                        self.edge_ids[src][rank].add(eid)
        dispatch, combine = max(send_d + recv_d), max(send_c + recv_c)
        if commit:
            self.routes[src][eid] = dst
            self.rank_ms, self.hidden = rank_ms, hidden
            self.send_d, self.recv_d, self.send_c, self.recv_c = send_d, recv_d, send_c, recv_c
        return Evaluation(dispatch + max(rank_ms) + combine, tuple(rank_ms), tuple(hidden), dispatch, combine)


def choose_residual(rank: RankState, ready_counts: Mapping[int, int],
                    candidates: Mapping[int, str]) -> Optional[int]:
    """Choose one actual late/demand group whose hidden wait exceeds split cost."""
    ready_ms = sum(rank.compute_ms(count) for count in ready_counts.values())
    best, gain = None, 0.0
    for eid, kind in sorted(candidates.items()):
        expert = rank.experts[eid]
        if not expert.available or kind not in ("late", "demand"):
            continue
        if kind == "demand" and rank.free_bytes < expert.nbytes + rank.staging_margin_bytes:
            continue
        wait = expert.pending_ms if kind == "late" else rank.queue_ms + rank.transfer_ms(eid)
        benefit = min(ready_ms, wait) - rank.split_overhead_ms
        if benefit > gain:
            best, gain = eid, benefit
    return best
