from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetSubsetTest(unittest.TestCase):
    def test_seeded_subset_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            source.write_text(
                "".join(json.dumps({"text": f"sample {index}"}) + "\n" for index in range(20)),
                encoding="utf-8",
            )
            outputs = [tmp_path / "subset_a.jsonl.gz", tmp_path / "subset_b.jsonl.gz"]
            for output in outputs:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/data/build_paper_dataset_subset.py"),
                        "--dataset",
                        "c4",
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--num-samples",
                        "8",
                        "--seed",
                        "42",
                    ],
                    check=True,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            with gzip.open(outputs[0], "rt", encoding="utf-8") as left:
                left_text = left.read()
            with gzip.open(outputs[1], "rt", encoding="utf-8") as right:
                right_text = right.read()
            self.assertEqual(left_text, right_text)


if __name__ == "__main__":
    unittest.main()
