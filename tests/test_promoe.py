from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import weakref
from unittest import mock
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from utils.promoe import (ChunkedPrefetcher, ProMoEPredictor, ProMoERuntime,
                              load_predictors, register_prediction_hooks)
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def _host(experts=4):
    return {e: tuple(torch.full((2, 2), e + c / 10) for c in range(3)) for e in range(experts)}


def _adapter_class(context):
    """Load the actual executor methods without optional Transformers imports."""
    source = ast.parse((ROOT / "utils/eval_core.py").read_text())
    block = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "DistributedSparseMoeBlock")
    names = {"_execute_promoe_recv_groups", "_execute_local_recv_groups", "_plain_ep_compute_segment",
             "_get_local_module_weights", "_get_local_fused", "_experts_world1_exact"}
    block.body = [n for n in block.body if isinstance(n, ast.FunctionDef) and n.name in names]
    block.bases = []
    module = ast.Module(body=[source.body[0], block], type_ignores=[])
    namespace = {"torch": torch, "F": F, "SparseContext": context,
                 "_env_flag": lambda key, default: key == "SPARSE_PROMOE_EXECUTOR_CONTROL"}
    exec(compile(ast.fix_missing_locations(module), "eval_core_executor", "exec"), namespace)
    return namespace["DistributedSparseMoeBlock"]


@unittest.skipIf(torch is None, "PyTorch is required for ProMoE checks")
class ProMoETest(unittest.TestCase):
    def test_reordering_and_capacity_under_more_demands_than_slots(self):
        engine = ChunkedPrefetcher(torch.device("cpu"))
        host = _host()
        try:
            engine.register(0, host, capacity=1)
            engine.begin_step(0)
            engine.predicted(0, [2], 0)
            engine.drain()
            ids = engine.precise(0, [0, 2, 1, 3], 0)
            self.assertEqual(ids[0], 2)
            for eid in ids:
                payload = engine.acquire(0, eid, timeout=5)
                for actual, expected in zip(payload, host[eid]):
                    torch.testing.assert_close(actual, expected)
                engine.release(0, eid)
            engine.drain()
            self.assertEqual(engine.snapshot()["peak_resident_experts_per_layer"], 1)
            self.assertFalse(engine.predicted(0, [1], 0))
            self.assertFalse(engine.predicted(0, [1], -1))
        finally:
            engine.close()

    def test_precise_preempts_speculative_at_one_chunk_boundary(self):
        entered, proceed = threading.Event(), threading.Event()
        copied = []
        class Controlled(ChunkedPrefetcher):
            def _copy_chunk(self, job, entry, chunk):
                copied.append((job.layer, job.expert, chunk))
                if job.layer == 1 and chunk == 0:
                    entered.set()
                    if not proceed.wait(5):
                        raise TimeoutError("Test chunk barrier")
                return super()._copy_chunk(job, entry, chunk)
        engine = Controlled(torch.device("cpu"))
        try:
            engine.register(0, _host(), 2)
            engine.register(1, _host(), 2)
            engine.begin_step(0)
            engine.predicted(1, [1], 0)
            self.assertTrue(entered.wait(5))
            engine.precise(0, [2], 0)
            proceed.set()
            engine.acquire(0, 2, timeout=5)
            engine.release(0, 2)
            engine.drain()
            self.assertEqual(copied[:4], [(1, 1, 0), (0, 2, 0), (0, 2, 1), (0, 2, 2)])
            self.assertEqual(engine.snapshot()["demand_bytes"], 48)
            self.assertEqual(engine.snapshot()["prefetch_bytes"], 48)
        finally:
            proceed.set()
            engine.close()

    def test_early_preemption_cancels_wrong_expert_after_current_chunk(self):
        entered, proceed = threading.Event(), threading.Event()
        copied = []
        class Controlled(ChunkedPrefetcher):
            def _copy_chunk(self, job, entry, chunk):
                copied.append((job.expert, chunk))
                if job.expert == 1 and chunk == 0:
                    entered.set()
                    if not proceed.wait(5):
                        raise TimeoutError("Test chunk barrier")
                return super()._copy_chunk(job, entry, chunk)
        engine = Controlled(torch.device("cpu"))
        try:
            engine.register(0, _host(), 1)
            engine.begin_step(0)
            engine.predicted(0, [1], 0)
            self.assertTrue(entered.wait(5))
            engine.precise(0, [2], 0)
            proceed.set()
            engine.acquire(0, 2, timeout=5)
            engine.release(0, 2)
            engine.drain()
            self.assertEqual(copied, [(1, 0), (2, 0), (2, 1), (2, 2)])
            self.assertEqual(engine.snapshot()["prefetch_bytes"], 16)
        finally:
            proceed.set()
            engine.close()

    def test_empty_precise_demand_rejects_late_prediction(self):
        engine = ChunkedPrefetcher(torch.device("cpu"))
        try:
            engine.register(0, _host(), 1)
            engine.begin_step(0)
            self.assertEqual(engine.precise(0, [], 0), [])
            self.assertFalse(engine.predicted(0, [0], 0))
            with self.assertRaises(ValueError):
                engine.acquire(0, 0, timeout=1)
            engine.reset(clear_cache=True)
            self.assertEqual(engine.snapshot()["resident_experts_by_layer"], {"0": 0})
        finally:
            engine.close()

    def test_copy_worker_failure_reaches_foreground(self):
        class Broken(ChunkedPrefetcher):
            def _copy_chunk(self, job, entry, chunk):
                raise RuntimeError("Synthetic copy failure")
        engine = Broken(torch.device("cpu"))
        try:
            engine.register(0, _host(), 1)
            engine.begin_step(0)
            engine.precise(0, [0], 0)
            with self.assertRaisesRegex(RuntimeError, "copy worker failed"):
                engine.acquire(0, 0, timeout=5)
        finally:
            engine.close()

    def test_compute_completion_fence_prevents_early_eviction(self):
        engine = ChunkedPrefetcher(torch.device("cpu"))
        completed = threading.Event()
        fence = types.SimpleNamespace(query=completed.is_set)
        try:
            engine.register(0, _host(), 1)
            engine.begin_step(0)
            engine.precise(0, [0, 1], 0)
            engine.acquire(0, 0, timeout=5)
            # Hold the scheduler lock while replacing CPU's immediate release
            # with a simulated GPU kernel completion fence.
            with engine.cv:
                engine.release(0, 0)
                engine.layers[0].entries[0].last_use = fence
            with self.assertRaises(TimeoutError):
                engine.acquire(0, 1, timeout=0.02)
            completed.set()
            with engine.cv:
                engine.cv.notify_all()
            engine.acquire(0, 1, timeout=5)
            engine.release(0, 1)
            engine.drain()
        finally:
            completed.set()
            engine.close()

    def test_victim_tensors_are_released_before_replacement_allocation(self):
        engine = ChunkedPrefetcher(torch.device("cpu"))
        try:
            engine.register(0, _host(), 1)
            engine.begin_step(0)
            engine.precise(0, [0], 0)
            engine.acquire(0, 0, timeout=5)
            engine.release(0, 0)
            engine.drain()
            victim = weakref.ref(engine.layers[0].entries[0].payload[0])
            original = torch.empty_like
            references_at_allocation = []
            def observed_allocate(*args, **kwargs):
                references_at_allocation.append(victim() is not None)
                return original(*args, **kwargs)
            with mock.patch("utils.promoe.torch.empty_like", side_effect=observed_allocate):
                engine.precise(0, [1], 0)
                engine.acquire(0, 1, timeout=5)
                engine.release(0, 1)
                engine.drain()
            self.assertEqual(references_at_allocation, [False, False, False])
        finally:
            engine.close()

    def test_decoder_hook_and_async_cpu_prediction_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = ProMoEPredictor(4, 3, 8)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.net[-1].bias[0] = 10
            torch.save({"model_type": "promoe_style_mlp", "source_layer": 1, "target_layer": 3,
                "input_dim": 4, "num_experts": 3, "hidden_dim": 8, "feature_mode": "hidden",
                "feature_source": "decoder_input", "state_dict": model.state_dict()},
                Path(tmp) / "rs_l3_inL1.pth")
            copies = []
            blocks = {}
            for layer in (1, 2, 3):
                host = _host(3)
                blocks[layer] = types.SimpleNamespace(device=torch.device("cpu"), hidden_size=4,
                    num_experts=3, top_k=1, expert_offload="cpu", experts_kind="modulelist",
                    local_expert_ids=list(host), _base_expert_cache_capacity=1,
                    local_gate_cpu={e: v[0] for e, v in host.items()},
                    local_up_cpu={e: v[1] for e, v in host.items()},
                    local_down_cpu2={e: v[2] for e, v in host.items()},
                    _record_h2d_bytes=lambda n, is_prefetch: copies.append((n, is_prefetch)))
            runtime = ProMoERuntime(blocks, tmp)
            class Decoder(torch.nn.Module):
                def forward(self, hidden_states):
                    return hidden_states + 100
            decoders = torch.nn.ModuleList([Decoder() for _ in range(4)])
            step_phase = [0, "prefill"]
            register_prediction_hooks(decoders, runtime, lambda: tuple(step_phase))
            x = torch.ones(2, 1, 4)
            try:
                torch.testing.assert_close(decoders[1](hidden_states=x), x + 100)
                runtime.drain()
                self.assertEqual(runtime.summary()["prediction_batches"], 0)
                step_phase[:] = [1, "decode"]
                decoders[1](x)
                runtime.drain()
                self.assertEqual(runtime.summary()["prediction_batches"], 1)
                self.assertEqual(runtime.prefetcher.precise(3, [2, 0], 1), [0, 2])
                for eid in (0, 2):
                    runtime.prefetcher.acquire(3, eid, timeout=5)
                    runtime.prefetcher.release(3, eid)
                runtime.drain()
                self.assertEqual(sum(n for n, speculative in copies if speculative), 48)
                self.assertEqual(sum(n for n, speculative in copies if not speculative), 48)
                runtime.reset(clear_cache=True)
                self.assertEqual(runtime.summary()["prediction_batches"], 0)
                self.assertEqual(runtime.summary()["resident_experts_by_layer"], {"1": 0, "2": 0, "3": 0})
                runtime.observe(1, torch.rand(2, 7), 2, "decode")
                with self.assertRaisesRegex(RuntimeError, "predictor worker failed"):
                    runtime.drain()
            finally:
                runtime.close()
            self.assertFalse(decoders[1]._forward_pre_hooks)

    def test_real_executor_preserves_router_weights_with_single_slot(self):
        torch.manual_seed(9)
        for fused in (False, True):
            host = {e: (torch.randn(3, 4), torch.randn(3, 4), torch.randn(4, 3)) for e in range(4)}
            if fused:
                host = {e: (torch.cat((v[0], v[1]), dim=0), v[2]) for e, v in host.items()}
            engine = ChunkedPrefetcher(torch.device("cpu"))
            try:
                engine.register(0, host, 1)
                engine.begin_step(0)
                context = types.SimpleNamespace(promoe_runtime=types.SimpleNamespace(prefetcher=engine), runtime_call_id=0)
                block = _adapter_class(context)()
                block.layer_idx, block.planner_mode = 0, "promoe"
                block.metrics = None
                block.experts_kind, block.expert_offload = "fused" if fused else "modulelist", "cpu"
                block.act_fn = F.silu
                x = torch.randn(5, 4)
                indices = torch.tensor([[0, 2], [1, 3], [3, 0], [2, 1], [0, 1]])
                weights = torch.rand(5, 2)
                original_idx, original_w = indices.clone(), weights.clone()
                expected = torch.zeros_like(x)
                for token in range(5):
                    for k in range(2):
                        parts = host[int(indices[token, k])]
                        if fused:
                            gate, up = F.linear(x[token], parts[0]).chunk(2)
                            value = F.linear(F.silu(gate) * up, parts[1])
                        else:
                            value = F.linear(F.silu(F.linear(x[token], parts[0])) * F.linear(x[token], parts[1]), parts[2])
                        expected[token] += value * weights[token, k]
                actual = block._experts_world1_exact(x, indices, weights)
                torch.testing.assert_close(actual, expected)
                # The distributed receive executor uses the very same method.
                recv = block._execute_local_recv_groups(x, indices[:, 0], weights[:, 0])
                reference = torch.stack([expected[t] * 0 for t in range(5)])
                for t in range(5):
                    parts = host[int(indices[t, 0])]
                    if fused:
                        gate, up = F.linear(x[t], parts[0]).chunk(2)
                        reference[t] = F.linear(F.silu(gate) * up, parts[1]) * weights[t, 0]
                    else:
                        reference[t] = F.linear(F.silu(F.linear(x[t], parts[0])) * F.linear(x[t], parts[1]), parts[2]) * weights[t, 0]
                torch.testing.assert_close(recv, reference)
                torch.testing.assert_close(indices, original_idx)
                torch.testing.assert_close(weights, original_w)
                self.assertEqual(engine.snapshot()["peak_resident_experts_per_layer"], 1)
                # Control shares token grouping, scatter and accumulation,
                # changing only fetching and scheduling policy.
                block.planner_mode = "none"
                block._get_local_fused = lambda eid: host[eid]
                block._get_local_module_weights = lambda eid: host[eid]
                control = block._experts_world1_exact(x, indices, weights)
                torch.testing.assert_close(control, actual)
            finally:
                engine.close()

    def test_train_distance_two_load_cpu_checkpoint_and_reject_rs(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            chunks = profile / "rs_decode_chunks"
            chunks.mkdir(parents=True)
            (profile / "manifest.json").write_text(json.dumps({
                "num_layers": 4, "moe_layer_indices": [1, 2, 3], "num_experts_per_layer": 3,
                "feature_source": "decoder_input"}))
            for layer in (1, 2, 3):
                torch.save(torch.rand(30, 4), chunks / f"rs_inputs_l{layer}_chunk_0.pt")
                torch.save({"indices": torch.zeros(30, 1).long(), "weights": torch.ones(30, 1),
                            "num_experts": 3, "top_k": 1}, chunks / f"rs_labels_l{layer}_chunk_0.pt")
            output = Path(tmp) / "trained"
            completed = subprocess.run([sys.executable, "scripts/train_next_layer_rs.py",
                "--profile-base-dir", str(profile), "--output-dir", str(output), "--lead-layers", "2",
                "--model-arch", "promoe", "--hidden-dim", "8", "--epochs", "1", "--batch-size", "8",
                "--target-mode", "uniform_multihot", "--eval-ks", "1,2"], cwd=ROOT,
                env=os.environ | {"PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            bank = load_predictors(str(output), [1, 2, 3], 4, 3, 2)
            self.assertEqual(set(bank), {1})
            self.assertEqual(bank[1][0], 3)
            self.assertEqual(bank[1][1](torch.rand(2, 4)).shape, (2, 3))
            path = output / "rs_l3_inL1.pth"
            payload = torch.load(path, weights_only=True)
            payload["model_type"] = "next_layer_residual_logits_rs"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "model_type"):
                load_predictors(str(output), [1, 2, 3], 4, 3, 2)
            with self.assertRaises(FileNotFoundError):
                load_predictors(str(output), [0, 1, 2, 3], 4, 3, 2)


class ProMoELauncherTest(unittest.TestCase):
    def test_one_or_three_gpu_topology_and_matched_rr(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "test.env"
            env_file.write_text(f"PYTHON_BIN=/runtime/python\nMODEL_PATH=/models/moe\nC4_RUNTIME=/data/eval.jsonl\n"
                                f"RESULTS_ROOT={tmp}/results\nPROMOE_DIR=/offline/promoe\n")
            for count in (1, 3):
                for method, planner in (("promoe-style", "promoe"), ("rr-matched", "none")):
                    completed = subprocess.run(["bash", "scripts/run/run_promoe.sh", method], cwd=ROOT,
                        env={"PATH": "/usr/bin:/bin", "ENV_FILE": str(env_file), "DRY_RUN": "1",
                             "NNODES": "1", "NPROC_PER_NODE": str(count)},
                        capture_output=True, text=True, check=True)
                    self.assertIn(f"--ep-size {count}", completed.stdout)
                    self.assertIn(f"--planner-mode {planner}", completed.stdout)
                    self.assertIn("SPARSE_EPIPE_ENABLE=0", completed.stdout)
                    self.assertIn("SPARSE_PLAIN_EP_EXEC_MODE=sorted_segment", completed.stdout)
                    self.assertIn(f"SPARSE_PROMOE_EXECUTOR_CONTROL={int(method == 'rr-matched')}", completed.stdout)


if __name__ == "__main__":
    unittest.main()
