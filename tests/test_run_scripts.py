from __future__ import annotations

import subprocess
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunScriptTest(unittest.TestCase):
    def test_formal_case_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / "test.env"
            env_file.write_text(
                "\n".join(
                    [
                        f"PROJECT_ROOT={ROOT}",
                        "PYTHON_BIN=/opt/conda/envs/sparse/bin/python",
                        "CUDA_VISIBLE_DEVICES=0,1",
                        "NODE_RANK=0",
                        "NNODES=2",
                        "NPROC_PER_NODE=2",
                        "EP_SIZE=4",
                        "MASTER_ADDR=127.0.0.1",
                        "MASTER_PORT=29671",
                        "NCCL_SOCKET_IFNAME=lo",
                        "MODEL_KEY=qwen",
                        "MODEL_PATH=/models/qwen",
                        "C4_RUNTIME=/datasets/c4.jsonl.gz",
                        f"RESULTS_ROOT={tmp_path / 'results'}",
                        "NETWORK_LABEL=test",
                        "SCP_PATH=/offline/scp.json",
                        "RS_DIR=/offline/rs",
                        "REMOTE_SSH=review-node",
                        f"REMOTE_REPO={ROOT}",
                        "REMOTE_ENV_FILE=.env",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["bash", str(ROOT / "scripts/run/run_case.sh"), "sparse", "b4", "0"],
                check=True,
                cwd=ROOT,
                env={"PATH": "/usr/bin:/bin", "ENV_FILE": str(env_file), "DRY_RUN": "1",
                     "PREFETCH_TOP_M": "1", "PREDICTION_TOP_K": "8"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertIn("--batch 4", completed.stdout)
            self.assertIn("--max-new 128", completed.stdout)
            self.assertIn("SPARSE_SIM_ENABLE_WEAKNET=0", completed.stdout)
            self.assertIn("SPARSE_RS_PREFETCH_TOP_M_BY_RANK=1\\,1\\,1\\,1", completed.stdout)
            self.assertIn("SPARSE_RS_PREFETCH_PRED_TOPK_BY_RANK=8\\,8\\,8\\,8", completed.stdout)
            config_path, = (tmp_path / "results").rglob("case_config.json")
            config = json.loads(config_path.read_text())
            self.assertEqual(config["prefetch_top_m_by_rank"], "1,1,1,1")
            self.assertEqual(config["prediction_top_k_by_rank"], "8,8,8,8")

            launched = subprocess.run(
                ["bash", str(ROOT / "scripts/run/launch_two_node.sh")],
                check=True,
                cwd=ROOT,
                env={
                    "PATH": "/usr/bin:/bin",
                    "ENV_FILE": str(env_file),
                    "DRY_RUN": "1",
                    "METHODS": "rr sparse",
                    "WORKLOADS": "b1",
                    "REQUEST_GROUPS": "0",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertIn("METHODS='rr sparse'", launched.stdout)
            self.assertIn("WORKLOADS='b1'", launched.stdout)


if __name__ == "__main__":
    unittest.main()
