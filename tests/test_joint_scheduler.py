from __future__ import annotations

from dataclasses import replace
import random
import unittest

from utils.joint_scheduler import (
    Demand, ExpertState, JointPlanner, PlannerConfig, RankState, choose_residual,
)


def rank(experts, **kw):
    defaults = dict(capacity=len(experts), free_bytes=10000, h2d_bytes_per_ms=1,
                    h2d_latency_ms=0, expert_ms=0, item_ms=1, window_ms=5,
                    prefetch_budget=1)
    defaults.update(kw)
    return RankState(tuple(experts), **defaults)


def run(demand, ranks, owners, **kw):
    config = kw.pop('config', PlannerConfig(placement=False))
    n = len(ranks)
    latency = kw.pop('latency', [[0.0] * n for _ in ranks])
    bandwidth = [[1000000.0] * n for _ in ranks]
    return JointPlanner(config).plan(demand, ranks, owners, latency, bandwidth, 16, 12, **kw)


class JointSchedulerTests(unittest.TestCase):
    def test_epipe_feedback_avoids_redundant_prefetch(self):
        r = rank([ExpertState(5, True, True, True), ExpertState(5, True)], capacity=2)
        d = Demand(((10., 1.),), ((1., 1.),))
        feedback = run(d, [r], [0, 0])
        separate = run(d, [r], [0, 0], config=PlannerConfig(placement=False, epipe_feedback=False))
        self.assertEqual(feedback.admissions, ())
        self.assertEqual(separate.admissions[0].expert, 1)
        self.assertEqual(feedback.after.total_ms, separate.after.total_ms)

    def test_eviction_recomputes_ready_work_and_new_miss(self):
        r = rank([ExpertState(10, True, True, True), ExpertState(5, True)], capacity=1)
        d = Demand(((10., 1.),), ((1., 1.),))
        plan = run(d, [r], [0, 0])
        self.assertEqual(plan.admissions, ())
        self.assertEqual(plan.after.total_ms, 11)

    def test_partial_prefetch_may_help_without_meeting_deadline(self):
        r = rank([ExpertState(10, True, True, True), ExpertState(10, True)], window_ms=4)
        d = Demand(((2., 1.),), ((1., 1.),))
        plan = run(d, [r], [0, 0])
        self.assertGreater(plan.admissions[0].finish_ms, r.window_ms)
        self.assertEqual(plan.before.total_ms - plan.after.total_ms, 4)
        self.assertGreater(plan.after.total_ms, 3)

    def test_budget_prioritizes_slowest_rank_over_probability(self):
        absent = ExpertState(0, False)
        ranks = [rank([ExpertState(1, True), absent], window_ms=10),
                 rank([absent, ExpertState(10, True)], window_ms=10)]
        d = Demand(((.9, 0.), (0., .4)), ((.9, 0.), (0., .4)))
        plan = run(d, ranks, [0, 1], config=PlannerConfig(placement=False, epipe_feedback=False))
        self.assertEqual((plan.admissions[0].rank, plan.admissions[0].expert), (1, 1))

    def test_placement_uses_cached_legal_replica(self):
        ranks = [rank([ExpertState(10, True)], prefetch_budget=0),
                 rank([ExpertState(10, True, True, True)], prefetch_budget=0)]
        d = Demand(((1.,), (0.,)), ((1.,), (0.,)))
        cfg = PlannerConfig(placement=True)
        plan = run(d, ranks, [0], config=cfg, prefetch=False)
        self.assertEqual(plan.routes[0][0], 1)
        slow_network = run(d, ranks, [0], config=cfg, latency=[[0, 10], [10, 0]], prefetch=False)
        self.assertEqual(slow_network.routes[0][0], 0)

    def test_unavailable_replica_cannot_receive_or_prefetch(self):
        ranks = [rank([ExpertState(10, True)]), rank([ExpertState(0, False)])]
        d = Demand(((1.,), (0.,)), ((1.,), (0.,)))
        plan = run(d, ranks, [0], config=PlannerConfig(placement=True))
        self.assertEqual(plan.routes[0][0], 0)
        self.assertTrue(all(a.rank == 0 for a in plan.admissions))

    def test_simultaneous_remap_escapes_tied_rank_misses(self):
        ranks = [rank([ExpertState(10, True), ExpertState(10, True, True, True)], prefetch_budget=0),
                 rank([ExpertState(10, True, True, True), ExpertState(10, True)], prefetch_budget=0)]
        demand = Demand(((1., 1.), (1., 1.)), ((1., 1.), (1., 1.)))
        plan = run(demand, ranks, [0, 1], config=PlannerConfig(placement=True), prefetch=False)
        self.assertEqual(plan.routes, ((1, 0), (1, 0)))
        self.assertLess(plan.after.total_ms, plan.before.total_ms)

    def test_memory_and_protection_guards(self):
        d = Demand(((0., 1.),), ((0., 1.),))
        protected = rank([ExpertState(10, True, True, True, protected=True), ExpertState(10, True)], capacity=1)
        self.assertEqual(run(d, [protected], [0, 0]).admissions, ())
        empty = rank([ExpertState(10, True), ExpertState(10, True)], free_bytes=5)
        self.assertEqual(run(d, [empty], [0, 0]).admissions, ())

    def test_empty_demand_and_zero_cache_budget(self):
        r = rank([ExpertState(10, True)], capacity=0, prefetch_budget=0)
        self.assertEqual(run(Demand(((0.,),), ((0.,),)), [r], [0]).after.total_ms, 0)
        self.assertEqual(run(Demand(((1.,),), ((1.,),)), [r], [0]).admissions, ())

    def test_candidate_mask_preserves_predictor_depth(self):
        r = rank([ExpertState(10, True), ExpertState(20, True)], prefetch_budget=2, window_ms=30)
        d = Demand(((1., 1.),), ((1., 1.),), (True, False))
        plan = run(d, [r], [0, 0], config=PlannerConfig(placement=False, epipe_feedback=False))
        self.assertEqual([a.expert for a in plan.admissions], [0])

    def test_residual_choice_accounts_for_split_cost(self):
        r = rank([ExpertState(5, True, True, True),
                  ExpertState(5, True, True, False, 1),
                  ExpertState(5, True, True, False, 5)], split_overhead_ms=.5)
        self.assertEqual(choose_residual(r, {0: 3}, {1: 'late', 2: 'late'}), 2)
        self.assertIsNone(choose_residual(replace(r, split_overhead_ms=4), {0: 3}, {2: 'late'}))
        self.assertIsNone(choose_residual(r, {}, {2: 'late'}))

    def test_invalid_costs_and_cache_state_fail(self):
        d = Demand(((1.,),), ((1.,),))
        r = rank([ExpertState(10, True)])
        for broken in [replace(r, h2d_bytes_per_ms=0), replace(r, item_ms=float('nan')),
                       replace(r, experts=(ExpertState(10, True, False, True),))]:
            with self.assertRaises(ValueError): run(d, [broken], [0])

    def test_deterministic_feasible_nonincreasing_plans(self):
        for seed in range(30):
            rng = random.Random(seed)
            states = [ExpertState(rng.randrange(2, 15), True, e < 2, e < 2) for e in range(4)]
            r = rank(states, capacity=2, window_ms=rng.randrange(2, 10), prefetch_budget=2)
            counts = tuple(rng.randrange(1, 8) for _ in states)
            d = Demand((counts,), ((1.,) * 4,))
            a, b = run(d, [r], [0] * 4), run(d, [r], [0] * 4)
            self.assertEqual(a, b)
            self.assertLessEqual(a.after.total_ms, a.before.total_ms)
            self.assertLessEqual(len(a.admissions), 2)
            self.assertTrue(all(x.gain_ms > 0 for x in a.admissions))


if __name__ == '__main__':
    unittest.main()
