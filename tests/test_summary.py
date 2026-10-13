from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "result_summary", ROOT / "scripts/analysis/summarize_results.py"
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


class ResultSummaryTest(unittest.TestCase):
    def test_gain_uses_matched_rr_mean(self) -> None:
        rows = []
        for group, rr, sparse in ((0, 10.0, 12.0), (1, 14.0, 16.0)):
            common = {"model": "qwen", "network": "physical", "workload": "b4", "request_group": group}
            rows.append({**common, "method": "rr", "e2e_tok_s": rr, "decode_tok_s": rr * 2})
            rows.append({**common, "method": "sparse", "e2e_tok_s": sparse, "decode_tok_s": sparse * 2})
        aggregate = SUMMARY.aggregate(rows)
        sparse_row = next(row for row in aggregate if row["method"] == "sparse")
        self.assertAlmostEqual(sparse_row["e2e_gain_vs_rr_pct"], (14.0 / 12.0 - 1.0) * 100.0)


if __name__ == "__main__":
    unittest.main()
