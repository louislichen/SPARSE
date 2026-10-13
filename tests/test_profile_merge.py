from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts import scp_ep_merge_profiles as MERGE


class ProfileMergeTest(unittest.TestCase):
    def test_ordinary_profiles_merge_without_task_local_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = np.zeros((1, 2, 3), dtype=np.int64)
            for rank in range(2):
                counts = np.zeros_like(expected)
                counts[0, rank, rank] = rank + 2
                expected += counts
                np.savez(root / f"profile.rank{rank}.npz", A_prefill=counts,
                         A_decode=np.zeros_like(counts),
                         metadata_json=np.array(json.dumps({"rank": rank, "world_size": 2})))
            output = root / "merged.npz"
            MERGE.merge_profiles(root, output)
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["A_prefill"], expected)
                self.assertEqual(json.loads(str(data["metadata_json"].item()))["total_prefill_assignments"], 5)


if __name__ == "__main__":
    unittest.main()
