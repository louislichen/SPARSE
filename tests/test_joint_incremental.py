from dataclasses import replace
import random
import unittest

from utils.joint_scheduler import (
    Demand, ExpertState, JointPlanner, PlannerConfig, RankState, _RouteEvaluation,
)


def case(seed, world=None, experts=None):
    rng = random.Random(seed)
    world = world or rng.randrange(1, 5)
    experts = experts or rng.randrange(2, 16)
    owners = [e % world for e in range(experts)]
    ranks = []
    for r in range(world):
        states = []
        window = rng.random() * 5
        for e in range(experts):
            available = r == owners[e] or rng.random() < .6
            resident = available and rng.random() < .4
            ready = resident and rng.random() < .5
            states.append(ExpertState(rng.randrange(5, 100) if available else 0,
                                      available, resident, ready,
                                      0 if ready else rng.random() * 10,
                                      resident and rng.random() < .4, rng.random()))
        ranks.append(RankState(tuple(states), sum(e.resident for e in states) + rng.randrange(3),
                               rng.randrange(0, 200), rng.random() * 20 + 1, rng.random(),
                               rng.random(), rng.random(), window, rng.random() * 6,
                               rng.randrange(50), rng.randrange(50), rng.random(), rng.randrange(3)))
    counts, probs = [], []
    for src in range(world):
        row = tuple(rng.random() * 10 if rng.random() < .7 else 0 for _ in range(experts))
        counts.append(row)
        probs.append(tuple(rng.random() * .99 + .01 if n > 0 else rng.random() for n in row))
    demand = Demand(tuple(counts), tuple(probs), tuple(rng.random() < .7 for _ in owners))
    latency = [[0 if s == d else rng.random() for d in range(world)] for s in range(world)]
    bandwidth = [[rng.random() * 5000 + 1 for d in range(world)] for s in range(world)]
    return demand, ranks, owners, latency, bandwidth


class IncrementalCostTests(unittest.TestCase):
    def assert_evaluation(self, actual, expected):
        self.assertAlmostEqual(actual.total_ms, expected.total_ms, places=9)
        self.assertAlmostEqual(actual.dispatch_ms, expected.dispatch_ms, places=9)
        self.assertAlmostEqual(actual.combine_ms, expected.combine_ms, places=9)
        for a, b in zip(actual.rank_ms + actual.hidden_ms, expected.rank_ms + expected.hidden_ms):
            self.assertAlmostEqual(a, b, places=9)

    def test_trials_and_commits_match_full_evaluation(self):
        for seed in range(100):
            demand, ranks, owners, latency, bandwidth = case(seed)
            for feedback in (False, True):
                config = PlannerConfig(epipe_feedback=feedback, evaluator="reference")
                planner = JointPlanner(config)
                routes = [list(owners) for _ in ranks]
                delta = _RouteEvaluation(demand, ranks, routes, feedback, latency, bandwidth, 32, 24)
                rng = random.Random(seed)
                for step in range(30):
                    src, eid = rng.randrange(len(ranks)), rng.randrange(len(owners))
                    destinations = [r for r, rank in enumerate(ranks) if rank.experts[eid].available]
                    dst = rng.choice(destinations)
                    trial = [list(row) for row in routes]
                    trial[src][eid] = dst
                    expected = planner.evaluate(demand, ranks, trial, latency, bandwidth, 32, 24)
                    self.assert_evaluation(delta.move(src, eid, dst), expected)
                    if step % 3 == 0:
                        self.assert_evaluation(delta.move(src, eid, dst, commit=True), expected)
                        routes = trial

    def test_complete_plans_preserve_reference_decisions(self):
        for seed in range(150):
            demand, ranks, owners, latency, bandwidth = case(seed)
            config = PlannerConfig(evaluator="reference", epipe_feedback=seed % 2 == 0)
            reference = JointPlanner(config).plan(demand, ranks, owners, latency, bandwidth, 32, 24)
            optimized = JointPlanner(replace(config, evaluator="incremental")).plan(
                demand, ranks, owners, latency, bandwidth, 32, 24)
            self.assertEqual(optimized.routes, reference.routes)
            self.assertLessEqual(optimized.evaluations, reference.evaluations)
            self.assertEqual([(a.rank, a.expert, a.victim) for a in optimized.admissions],
                             [(a.rank, a.expert, a.victim) for a in reference.admissions])
            self.assert_evaluation(optimized.before, reference.before)
            self.assert_evaluation(optimized.after, reference.after)

    def test_budget_and_prediction_mask_preserve_legal_nonincreasing_proxy(self):
        for seed in range(50):
            demand, ranks, owners, latency, bandwidth = case(seed, world=4, experts=60)
            config = PlannerConfig(prediction_candidates_only=True, max_evaluations=12)
            plan = JointPlanner(config).plan(demand, ranks, owners, latency, bandwidth, 32, 24)
            self.assertLessEqual(plan.evaluations, 12)
            self.assertLessEqual(plan.after.total_ms, plan.before.total_ms + 1e-9)
            for src, row in enumerate(plan.routes):
                for eid, dst in enumerate(row):
                    self.assertTrue(ranks[dst].experts[eid].available)
                    if not demand.prefetch_candidates[eid]:
                        self.assertEqual(dst, owners[eid])
            self.assertTrue(all(a.gain_ms > config.min_gain_ms for a in plan.admissions))

    def test_saturated_union_last_remote_edge_and_empty_rank(self):
        demand = Demand(((1., 2., 0.), (2., 0., 0.)), ((.8, .9, .3), (.7, .2, .4)))
        states = (ExpertState(10, True, True, True), ExpertState(10, True), ExpertState(20, True))
        ranks = [RankState(states, 3, 100, 1, 0, .3, .5) for _ in range(2)]
        routes = [[1, 1, 0], [1, 0, 0]]
        latency, bandwidth = [[0, 1], [1, 0]], [[100] * 2 for _ in ranks]
        planner = JointPlanner(PlannerConfig())
        delta = _RouteEvaluation(demand, ranks, routes, True, latency, bandwidth, 32, 24)
        for src, eid, dst in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)]:
            routes[src][eid] = dst
            self.assert_evaluation(delta.move(src, eid, dst, commit=True),
                                   planner.evaluate(demand, ranks, routes, latency, bandwidth, 32, 24))


if __name__ == '__main__':
    unittest.main()
