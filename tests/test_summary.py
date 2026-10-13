from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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
    def test_e2e_excludes_prompt_tokens_and_uses_matching_sd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case_config.json").write_text(json.dumps({"method": "rr"}))
            (root / "rr_summary.json").write_text(json.dumps({
                "e2e_output_tok_s": 8.0,
                "e2e_output_tok_s_std": 0.25,
                "throughput_tok_s": 40.0,
                "throughput_tok_s_std": 1.25,
                "decode_throughput_tok_s": 16.0,
            }))
            row, = SUMMARY.find_rows(root)
            self.assertEqual(row["e2e_tok_s"], 8.0)
            self.assertEqual(row["e2e_repeat_sd"], 0.25)
            self.assertEqual(row["decode_repeat_sd"], "")

    def test_legacy_input_inclusive_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case_config.json").write_text(json.dumps({"method": "rr"}))
            (root / "rr_summary.json").write_text(json.dumps({
                "throughput_tok_s": 40.0,
                "decode_throughput_tok_s": 16.0,
            }))
            with self.assertRaisesRegex(ValueError, "includes prompt tokens"):
                SUMMARY.find_rows(root)

    def test_gain_uses_matched_rr_mean(self) -> None:
        rows = []
        for group, rr, sparse in ((0, 10.0, 12.0), (1, 14.0, 16.0)):
            common = {"model": "qwen", "network": "physical", "workload": "b4", "request_group": group}
            rows.append({**common, "method": "rr", "e2e_tok_s": rr, "decode_tok_s": rr * 2})
            rows.append({**common, "method": "sparse", "e2e_tok_s": sparse, "decode_tok_s": sparse * 2})
        aggregate = SUMMARY.aggregate(rows)
        sparse_row = next(row for row in aggregate if row["method"] == "sparse")
        self.assertAlmostEqual(sparse_row["e2e_gain_vs_rr_pct"], (14.0 / 12.0 - 1.0) * 100.0)

    def test_single_group_does_not_claim_zero_uncertainty(self) -> None:
        row, = SUMMARY.aggregate([{"model": "qwen", "network": "physical", "workload": "b4",
                                   "method": "rr", "e2e_tok_s": 8.0, "decode_tok_s": 16.0}])
        self.assertEqual(row["e2e_tok_s_group_sd"], "")
        self.assertEqual(row["decode_tok_s_group_sd"], "")


if __name__ == "__main__":
    unittest.main()
