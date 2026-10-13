from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from scripts import train_next_layer_rs as TRAIN


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        model = TRAIN.RoutingScoutResidualMLP(4, 3, hidden_dim=8, num_hidden_layers=1, dropout=0.0)
        reference = copy.deepcopy(model)
        distributed = DDP(model)
        optimizer = torch.optim.SGD(distributed.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        y = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3).float()
        for _ in range(2):
            optimizer.zero_grad()
            probs, _ = TRAIN._model_probs_logits(distributed, x[rank * 2:(rank + 1) * 2])
            TRAIN._multi_positive_ce(probs, y[rank * 2:(rank + 1) * 2]).backward()
            optimizer.step()
            reference_optimizer.zero_grad()
            ref_probs, _ = TRAIN._model_probs_logits(reference, x)
            TRAIN._multi_positive_ce(ref_probs, y).backward()
            reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


@unittest.skipIf(torch is None, "PyTorch is required for RS training checks")
class RSTrainingTest(unittest.TestCase):
    def test_records_are_sharded_once_with_equal_training_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for chunk, count in enumerate((4, 8, 12)):
                x = torch.arange(count * 4).reshape(count, 4).float() + chunk * 100
                torch.save(x, root / f"rs_inputs_l0_chunk_{chunk}.pt")
                torch.save({"indices": torch.zeros(count, 1).long(),
                            "weights": torch.ones(count, 1), "num_experts": 3, "top_k": 1},
                           root / f"rs_labels_l1_chunk_{chunk}.pt")
            datasets = []
            loaders = []
            for rank in range(2):
                x, y, indices = TRAIN._load_pair_dataset(
                    root, 0, 1, rank, 2, fp32_inputs=True, use_router_weights=False,
                    target_mode="uniform_multihot", distill_temperature=1.0, feature_mode="hidden")
                self.assertEqual(len(x), 24)
                datasets.append(x)
                dataset = TRAIN._split_dataset(x, y, indices, seed=42)[0]
                loaders.append(TRAIN._make_loader(dataset, 4, rank, 2, shuffle=True))
            torch.testing.assert_close(*datasets)
            self.assertEqual(len(loaders[0]), len(loaders[1]))
            coverage = set(loaders[0].sampler) | set(loaders[1].sampler)
            self.assertEqual(coverage, set(range(len(loaders[0].dataset))))

    def test_evaluation_does_not_pad_or_duplicate_records(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(5))
        samples = [list(TRAIN._make_loader(dataset, 2, r, 2, shuffle=False).sampler) for r in range(2)]
        self.assertEqual(sorted(samples[0] + samples[1]), list(range(5)))
        self.assertFalse(set(samples[0]) & set(samples[1]))

    def test_training_and_validation_skip_dense_transformer_layer(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            chunks = profile / "rs_decode_chunks"
            chunks.mkdir(parents=True)
            (profile / "manifest.json").write_text(json.dumps({
                "num_layers": 3, "moe_layer_indices": [1, 2], "num_experts_per_layer": 3,
                "top_k": 1, "hidden_size": 4, "total_decode_records": 20}))
            for layer in (1, 2):
                torch.save(torch.arange(80).reshape(20, 4).float() / 80,
                           chunks / f"rs_inputs_l{layer}_chunk_0.pt")
                torch.save({"indices": torch.zeros(20, 1).long(), "weights": torch.ones(20, 1),
                            "num_experts": 3, "top_k": 1}, chunks / f"rs_labels_l{layer}_chunk_0.pt")
            traces = profile / "trace_chunks"
            traces.mkdir()
            indices = torch.zeros(20, 3, 1).long()
            indices[:, 0] = -1
            torch.save({"layer_indices": indices, "layer_weights": torch.ones(20, 3, 1)},
                       traces / "decode_trace_chunk_0.pt")
            summary = profile / "summary"
            summary.mkdir()
            torch.save({}, summary / "decode_profile_summary.pt")
            subprocess.run([sys.executable, "scripts/validate_next_layer_rs_profile.py",
                            "--profile-base-dir", str(profile), "--check-load"],
                           cwd=root, check=True, capture_output=True, timeout=60)
            output = Path(tmp) / "trained"
            subprocess.run([sys.executable, "scripts/train_next_layer_rs.py", "--profile-base-dir", str(profile),
                            "--output-dir", str(output), "--epochs", "1", "--batch-size", "4",
                            "--hidden-dim", "8", "--model-arch", "residual", "--num-hidden-layers", "1",
                            "--target-mode", "uniform_multihot", "--eval-ks", "1,2"],
                           cwd=root, check=True, capture_output=True, timeout=60)
            self.assertEqual([p.name for p in output.glob("*.pth")], ["rs_l2_inL1.pth"])

    @unittest.skipUnless(torch is not None and dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_two_rank_gradients_match_global_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_ddp_worker, args=((Path(tmp) / "rendezvous").as_uri(),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
