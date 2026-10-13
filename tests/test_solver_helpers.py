from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scp_solver", ROOT / "scripts/solve_replicated_scp_ep.py"
)
assert SPEC is not None and SPEC.loader is not None
SOLVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOLVER
SPEC.loader.exec_module(SOLVER)


class SolverHelperTest(unittest.TestCase):
    def test_round_robin_owners(self) -> None:
        placement = SOLVER.round_robin_replicas(layers=2, experts=6, world=4)
        self.assertEqual(placement["0"]["0"], [0])
        self.assertEqual(placement["0"]["5"], [1])
        self.assertEqual(placement["1"]["3"], [3])

    def test_rank_capacity_vector(self) -> None:
        parsed = SOLVER._parse_int_capacity_vector(
            "8,8,5,5", world=4, default=8, name="cache"
        )
        self.assertEqual(parsed, (8, 8, 5, 5))


if __name__ == "__main__":
    unittest.main()
