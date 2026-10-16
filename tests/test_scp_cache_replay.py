import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scp_cache_replay as replay
import solve_replicated_scp_ep as solver


def profile(rows, layers=2, world=4, experts=4):
    # shard, stage, call/step, layer, source, expert, count
    data = np.asarray(rows, dtype=np.int64)
    out = {key: data[:, col].copy() for col, key in enumerate((
        "trace_shard_id", "trace_stage", "trace_call_id", "trace_layer",
        "trace_source_rank", "trace_expert", "trace_count"))}
    out["trace_step"] = out["trace_call_id"].copy()
    out["trace_request_id"] = out["trace_shard_id"] * 1000000 + out["trace_source_rank"]
    out["metadata"] = {"world_size": world, "task_local_enabled": False}
    for stage, key in enumerate(("A_prefill", "A_decode")):
        out[key] = np.zeros((layers, world, experts), dtype=np.int64)
        mask = out["trace_stage"] == stage
        np.add.at(out[key], tuple(out[k][mask] for k in (
            "trace_layer", "trace_source_rank", "trace_expert")), out["trace_count"][mask])
    return out


TOPOLOGY = {"world_size": 4, "rank_to_node": {"0": 0, "1": 0, "2": 1, "3": 1},
    "bandwidth_model": {"enabled": True, "same_node_bw_MBps": 10000,
        "cross_node_bw_MBps": 50, "cross_node_latency_ms": .4},
    "expert_weight_bytes": 1000000, "h2d_bw_MBps_by_rank": [1000] * 4}
RR = {"replicas": solver.round_robin_replicas(2, 4, 4)}


class CacheReplayTest(unittest.TestCase):
    def test_collective_groups_include_all_sources(self):
        p = profile([(0, 0, 0, 0, 0, 2, 2), (0, 0, 0, 0, 1, 3, 3),
                     (0, 0, 0, 1, 0, 2, 2), (1, 0, 0, 0, 0, 2, 2)])
        groups = replay.grouped_trace_indices(p)
        self.assertEqual([len(g) for g in groups], [2, 1, 1])
        expected = .4 + 3 * 8202 / 50 / 1000
        result = replay.evaluate_grouped_comm_tail(p, RR, TOPOLOGY)
        self.assertAlmostEqual(result["J_comm_ms"], expected + 2 * (.4 + 2 * 8202 / 50 / 1000))
        self.assertEqual(result["active_cross_node_events"], 4)

    def test_lru_coalesces_sources_and_keeps_layer_caches_separate(self):
        p = profile([(0, 1, 1, 0, 0, 2, 2), (0, 1, 1, 0, 1, 2, 3),
                     (0, 1, 2, 0, 0, 2, 7), (0, 1, 2, 1, 0, 2, 11),
                     (1, 1, 1, 0, 0, 2, 13)])
        result = replay.evaluate_profile_placement_with_cache_replay(p, RR, TOPOLOGY, stage="decode")
        self.assertEqual(result["demand_expert_loads"], 3)
        self.assertEqual(result["resident_hit_items"], 7)
        self.assertEqual(result["total_items"], 36)
        self.assertAlmostEqual(result["rankmax_h2d_ms"], 3.)
        self.assertTrue(result["h2d_timing_calibrated"])

    def test_replay_matches_solver_grouped_tail_for_replicas_and_rr_ties(self):
        p = profile([(0, 0, 0, 0, 0, 2, 12), (0, 0, 0, 0, 1, 3, 8),
                     (0, 0, 0, 0, 2, 0, 5), (1, 0, 0, 0, 3, 1, 9)])
        model = solver.build_comm_model(TOPOLOGY, {})
        replicas = solver.round_robin_replicas(2, 4, 4)
        replicas["0"]["2"].append(0)
        replicas["0"]["1"].append(3)
        cfg = SimpleNamespace(stage="prefill", lambda_event=0.)
        state = solver._make_grouped_state(trace_profile=p, topology=TOPOLOGY,
            replicas=replicas, model=model, cfg=cfg, layers=2, experts=4)
        placement = solver._base_placement_obj(replicas=replicas, model=model,
            topology=TOPOLOGY, layers=2, experts=4, objective="test")
        result = replay.evaluate_grouped_comm_tail(p, placement, TOPOLOGY)
        self.assertAlmostEqual(result["J_comm_ms"], solver._state_J_comm(state))
        np.testing.assert_allclose(result["per_source_total_comm_ms"], state["source_comm"].sum(axis=0))

    def test_prefill_can_warm_decode_but_is_not_counted_as_decode(self):
        p = profile([(0, 0, 0, 0, 0, 2, 20), (0, 1, 1, 0, 0, 2, 2)])
        cold = replay.evaluate_profile_placement_with_cache_replay(p, RR, TOPOLOGY, stage="decode")
        warm = replay.evaluate_profile_placement_with_cache_replay(p, RR, TOPOLOGY,
            stage="decode", warm_cache=True, warmup_placement=RR)
        self.assertEqual(cold["demand_expert_loads"], 1)
        self.assertEqual(warm["demand_expert_loads"], 0)
        self.assertEqual(warm["resident_hit_items"], 2)
        self.assertEqual(warm["total_items"], 2)

    def test_lru_eviction_and_zero_capacity(self):
        p = profile([(0, 1, step, 0, 1, expert, 1) for step, expert in
                     enumerate((0, 1, 0), 1)])
        placement = {"dest_rank_table_decode": {"0": {"0": [0, 0, 0, 0]}}}
        result = replay.evaluate_profile_placement_with_cache_replay(p, placement, TOPOLOGY,
            stage="decode", cache_capacity=1)
        self.assertEqual(result["demand_expert_loads"], 3)
        empty = replay.evaluate_profile_placement_with_cache_replay(p, placement, TOPOLOGY,
            stage="decode", cache_capacity=0)
        self.assertEqual(empty["resident_hit_items"], 0)

    def test_generic_packet_replay_uses_topology_without_task_labels(self):
        p = profile([(0, 0, 0, 0, 1, 2, 3), (0, 0, 0, 0, 0, 2, 2)])
        p["trace_source_node"] = p["trace_source_rank"].copy()  # legacy merge default
        model = solver.build_comm_model(TOPOLOGY, {})
        table = {"0": {"0": [0, 1, 2, 3], "1": [0, 1, 2, 3]}}
        result = solver._runtime_packet_metrics_from_table(trace_profile=p,
            dest_table=table, model=model, stage="prefill", layers=2, experts=4)
        self.assertEqual(result["cross_events"], 2)
        self.assertEqual(result["per_direction_items"], {"node0->node1": 5})
        with self.assertRaises(RuntimeError):
            solver._require_task_local_profile(p)

    def test_file_validation_and_full_generic_solver(self):
        p = profile([(s, 0, 0, 0, 0, 2, 20) for s in range(2)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.npz"
            payload = {k: v for k, v in p.items() if k != "metadata"}
            payload["metadata_json"] = np.array(json.dumps(p["metadata"]))
            np.savez(path, **payload)
            loaded = replay.load_profile_with_trace(path)
            np.testing.assert_array_equal(loaded["A_prefill"], p["A_prefill"])
            topology = Path(tmp) / "topology.json"
            topology.write_text(json.dumps(TOPOLOGY))
            output = Path(tmp) / "placement.json"
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/solve_replicated_scp_ep.py"),
                "--train-profile", str(path), "--valid-profile", str(path), "--topology", str(topology),
                "--out", str(output), "--stage", "prefill", "--objective", "tail_headroom_scp_v2",
                "--global-extra-replicas", "1", "--expert-cache-capacity-by-rank", "8,8,8,8",
                "--require-valid-positive"], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads(output.read_text())
            self.assertIn("offline_metrics", result)
            # An invalid replay input must abort instead of disabling guards.
            bad_topology = dict(TOPOLOGY, h2d_bw_MBps_by_rank=[0] * 4)
            topology.write_text(json.dumps(bad_topology))
            bad = subprocess.run(proc.args, text=True, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("invalid h2d_bw_MBps_by_rank", bad.stderr)
            payload["A_prefill"] = payload["A_prefill"] + 1
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "aggregate demand disagree"):
                replay.load_profile_with_trace(path)


if __name__ == "__main__":
    unittest.main()
