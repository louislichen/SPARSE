"""Paired CPU planner timings with unchanged hypotheses and decision checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.joint_scheduler import Demand, ExpertState, JointPlanner, PlannerConfig, RankState


def synthetic(world, experts, prediction, seed):
    rng = random.Random(seed)
    states = []
    for rank in range(world):
        resident = set(rng.sample(range(experts), min(8, experts)))
        entries = tuple(ExpertState(3 * 2048 * 1408 * 2, True, e in resident, e in resident,
                                   protected=e in resident) for e in range(experts))
        states.append(RankState(entries, min(8, experts), 16 * 1024**3, 8_000_000, .04,
                                .025, .003, .21 if prediction else 0, prefetch_budget=2))
    counts, probs, candidates = [], [], set()
    for src in range(world):
        if prediction:
            scores = [rng.random() + .1 for _ in range(experts)]
            row = tuple(min(1., 4 * s / sum(scores)) for s in scores)
        else:
            chosen = set(rng.sample(range(experts), min(4, experts)))
            row = tuple(float(e in chosen) for e in range(experts))
        candidates.update(sorted(range(experts), key=lambda e: -row[e])[:4])
        counts.append(row); probs.append(row)
    return dict(demand=Demand(tuple(counts), tuple(probs), tuple(e in candidates for e in range(experts))),
                ranks=tuple(states), owners=tuple(e % world for e in range(experts)),
                link_latency=tuple(tuple(0 if s == d else .02 for d in range(world)) for s in range(world)),
                link_bandwidth=tuple((8_000_000,) * world for _ in range(world)),
                dispatch_item_bytes=4106, combine_item_bytes=4096, prefetch=prediction)


def from_trace(data):
    case = dict(data)
    case['demand'] = Demand(**case['demand'])
    case['ranks'] = tuple(RankState(**{**r, 'experts': tuple(ExpertState(**e) for e in r['experts'])})
                          for r in case['ranks'])
    config = case.pop('config', {})
    return case, config


def check(reference, candidate):
    if reference.routes != candidate.routes:
        raise AssertionError('incremental routes differ from reference')
    a = [(x.rank, x.expert, x.victim) for x in reference.admissions]
    b = [(x.rank, x.expert, x.victim) for x in candidate.admissions]
    if a != b:
        raise AssertionError('incremental admissions differ from reference')
    for left, right in [(reference.before.total_ms, candidate.before.total_ms),
                        (reference.after.total_ms, candidate.after.total_ms)]:
        if abs(left - right) > 1e-9 * max(1., abs(left)):
            raise AssertionError('incremental objective differs from reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--cases', type=int, default=12)
    args = parser.parse_args()
    if min(args.rounds, args.cases) < 1: parser.error('rounds/cases must be positive')
    if args.trace:
        groups = {'recorded': [from_trace(json.loads(line)) for line in args.trace.read_text().splitlines() if line.strip()]}
    else:
        groups = {f'r{r}_e{e}_{phase}': [(synthetic(r, e, phase == "forecast", s), {}) for s in range(args.cases)]
                  for r in (2, 4) for e in (8, 60) for phase in ('forecast', 'execution')}
    report = dict(purpose='paired planner microbenchmark; no GPU or predictor-quality claim',
                  source_sha256=hashlib.sha256((ROOT / 'utils/joint_scheduler.py').read_bytes()).hexdigest(), groups={})
    for name, samples in groups.items():
        timings = {'reference': [], 'incremental': []}
        counters = {key: [] for key in timings}
        planners = [(case, {mode: JointPlanner(replace(PlannerConfig(**config), evaluator=mode))
                             for mode in timings}) for case, config in samples]
        for case, pair in planners:
            check(pair['reference'].plan(**case), pair['incremental'].plan(**case))
        for round_idx in range(args.rounds):
            for case, pair in planners:
                results = {}
                # Alternate execution order to avoid a consistent warm-cache bias.
                for mode in (('reference', 'incremental') if round_idx % 2 == 0 else ('incremental', 'reference')):
                    start = time.perf_counter_ns()
                    results[mode] = pair[mode].plan(**case)
                    timings[mode].append((time.perf_counter_ns() - start) / 1e6)
                    plan = results[mode]
                    counters[mode].append((plan.evaluations, plan.full_evaluations, plan.incremental_evaluations))
                check(results['reference'], results['incremental'])
        result = {mode: dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                             p95_ms=sorted(values)[min(len(values)-1, int(len(values)*.95))],
                             mean_evaluations=statistics.mean(c[0] for c in counters[mode]),
                             mean_full_evaluations=statistics.mean(c[1] for c in counters[mode]),
                             mean_incremental_evaluations=statistics.mean(c[2] for c in counters[mode]))
                  for mode, values in timings.items()}
        result['decisions_match'] = True
        result['median_speedup'] = result['reference']['median_ms'] / max(result['incremental']['median_ms'], 1e-12)
        report['groups'][name] = result
        print(name, json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
