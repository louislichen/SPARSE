from __future__ import annotations

import ast
import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as F
    from utils.joint_runtime import CostProfile, JointRuntime
except ImportError:
    torch = None

from utils.joint_scheduler import PlannerConfig

ROOT = Path(__file__).resolve().parents[1]


def profile_data(world=2):
    return dict(schema_version=1, world_size=world, calibrated=False,
                rank_defaults=dict(h2d_bytes_per_ms=6.4, h2d_latency_ms=0,
                                   expert_ms=.2, item_ms=.1, next_attention_ms=1,
                                   split_overhead_ms=.01),
                link_latency_ms=[[0 if s == d else .01 for d in range(world)] for s in range(world)],
                link_bandwidth_bytes_per_ms=[[1000000] * world for _ in range(world)])


def core_methods(context):
    tree = ast.parse((ROOT / 'utils/eval_core.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DistributedSparseMoeBlock')
    names = {'_resolve_dest_for_flat_experts', '_plain_ep_compute_segment', '_compute_and_schedule_next_layer_rs'}
    nodes = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                        type_ignores=[])
    namespace = dict(torch=torch, F=F, SparseContext=context, _env_flag=lambda *args: False)
    exec(compile(ast.fix_missing_locations(module), '<actual-serving-methods>', 'exec'), namespace)
    return {n.name: namespace[n.name] for n in nodes}


class TinyBlock:
    def __init__(self, rank, layer, resident, budget):
        self.comm_rank, self.comm_world, self.comm_group = rank, 2, dist.group.WORLD
        self.layer_idx, self.num_layers, self.num_experts, self.top_k = layer, 2, 2, 2
        self.hidden_size, self.dtype, self.device = 3, torch.float32, torch.device('cpu')
        self._local_eid_set = {0, 1}
        self.rr_expert_to_rank_cpu = [0, 1]
        self.ids = set(resident)
        self._cache = SimpleNamespace(capacity=2, resident_ids=lambda: sorted(self.ids))
        self._prefetch_events, self._last_used_decode_step = {}, {}
        self._epipe_min_free_mb, self.budget = 0, budget
        self.admitted = []
        self.planner_mode, self.expert_source, self.exec_routing_mode = 'rs_prefetch', 'router', 'router'
        self.metrics = SimpleNamespace(rs_next_layer_pred_calls=0)
        self.expert_offload, self.experts_kind, self.norm_topk_prob = 'cpu', 'modulelist', True
        self.act_fn = F.silu
        torch.manual_seed(19)
        self.weights = [(torch.randn(4, 3), torch.randn(4, 3), torch.randn(3, 4)) for _ in range(2)]
    def _current_stage(self): return 'decode'
    def _prefetch_protect_ids(self): return set()
    def _epipe_estimate_expert_nbytes(self, eid): return 64
    def _local_prefetch_ready(self, eid): return eid in self.ids
    def _effective_rs_prefetch_top_m(self): return self.budget
    def _rs_prefetch_predict_topk(self): return 2
    def _begin_group_prefetch_cycle(self, *args, **kw): pass
    def _run_rs_mlp_timed(self, module, x): return module(x)
    def _admit_prefetch_plan(self, plan, **kw):
        for eid, victim in plan:
            if victim is not None: self.ids.discard(victim)
            self.ids.add(eid)
        self.admitted.extend(plan)
    def _get_local_module_weights(self, eid): return self.weights[eid]
    def _get_local_fused(self, eid): return torch.cat(self.weights[eid][:2]), self.weights[eid][2]


def distributed_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        source = TinyBlock(rank, 0, [1 - rank], 1)
        target = TinyBlock(rank, 1, [], 0 if rank == 0 else 1)
        context = SimpleNamespace(runtime_call_id=1, current_phase='decode',
                                  layer_blocks={0: source, 1: target},
                                  rs_bank_layout='next_layer', rs_bank_task='router_prob',
                                  rs_bank_source_layers=[-1, 0],
                                  rs_bank=[None, lambda x: torch.softmax(x[:, :2], dim=-1)])
        runtime = JointRuntime(context, CostProfile(profile_data(), 2), PlannerConfig())
        context.joint_runtime = runtime
        runtime.verify(source)
        methods = core_methods(context)
        # Call the actual early-RS hook with unequal budgets and one empty rank.
        features = torch.tensor([[2., -1., .5]]) if rank == 0 else torch.empty((0, 3))
        methods['_compute_and_schedule_next_layer_rs'](source, features, 1)
        assert runtime.stats['prediction_plans'] == 1
        assert source.metrics.rs_next_layer_pred_calls == 1
        assert len(target.admitted) <= target.budget
        for kind in ('modulelist', 'fused'):
            source.experts_kind = kind
            for empty_rank in (None, 1):
                indices = torch.tensor([[0, 1]], dtype=torch.long) if rank != empty_rank else torch.empty((0, 2), dtype=torch.long)
                weights = torch.tensor([[.3, .7]]) if indices.numel() else torch.empty((0, 2))
                hidden = torch.tensor([[1., 2., 3.]]) + rank if indices.numel() else torch.empty((0, 3))
                original_ids, original_weights = indices.clone(), weights.clone()
                runtime.prepare_execution(source, indices)
                destinations = methods['_resolve_dest_for_flat_experts'](source, indices.reshape(-1))
                routes = [None, None]
                dist.all_gather_object(routes, runtime.routes[0][1].tolist())
                assert routes[0] == routes[1]
                if rank != empty_rank:
                    assert destinations.tolist() == [1, 0]  # cached replicas
                order = torch.argsort(destinations)
                send_counts = torch.bincount(destinations, minlength=2).long()
                recv_counts = torch.empty_like(send_counts)
                dist.all_to_all_single(recv_counts, send_counts)
                sends, receives = send_counts.tolist(), recv_counts.tolist()
                expanded = hidden.repeat_interleave(2, dim=0)
                send_hidden, send_ids = expanded[order].contiguous(), indices.reshape(-1)[order].contiguous()
                send_weights = weights.reshape(-1)[order].contiguous()
                recv_hidden = torch.empty((sum(receives), 3))
                recv_ids = torch.empty(sum(receives), dtype=torch.long)
                recv_weights = torch.empty(sum(receives))
                for out, inp in ((recv_hidden, send_hidden), (recv_ids, send_ids), (recv_weights, send_weights)):
                    dist.all_to_all_single(out, inp, output_split_sizes=receives, input_split_sizes=sends)
                result = torch.empty_like(recv_hidden)
                for eid in torch.unique(recv_ids).tolist():
                    assert eid in source._local_eid_set
                    mask = recv_ids == eid
                    y, _ = methods['_plain_ep_compute_segment'](
                        source, eid=eid, cur=recv_hidden[mask], cur_w=recv_weights[mask, None],
                        eid_is_local=True, allow_nonlocal=False)
                    result[mask] = y
                returned = torch.empty_like(send_hidden)
                dist.all_to_all_single(returned, result, output_split_sizes=sends, input_split_sizes=receives)
                restored = torch.empty_like(expanded)
                restored[order] = returned
                actual = restored.reshape(len(hidden), 2, 3).sum(dim=1)
                expected = torch.zeros_like(hidden)
                for token in range(len(hidden)):
                    for slot in range(2):
                        eid = indices[token, slot]
                        gw, uw, dw = source.weights[eid]
                        y = F.linear(F.silu(F.linear(hidden[token], gw)) * F.linear(hidden[token], uw), dw)
                        expected[token] += y * weights[token, slot]
                torch.testing.assert_close(actual, expected)
                assert torch.equal(original_ids, indices) and torch.equal(original_weights, weights)
        assert runtime.stats['execution_plans'] == 4 and runtime.stats['exchange_calls'] == 5
        runtime.reset()
        assert not runtime.routes and not runtime.forecasts and runtime.stats['exchange_calls'] == 0
        with unittest.TestCase().assertRaises(RuntimeError): runtime.destinations(source, torch.tensor([0]))
    finally:
        dist.destroy_process_group()


@unittest.skipIf(torch is None, 'PyTorch required for joint runtime checks')
class JointRuntimeTests(unittest.TestCase):
    def test_cost_profile_and_activation_calibration(self):
        data = profile_data()
        profile = CostProfile(data, 2)
        scores = torch.tensor([[.1, .9]])
        torch.testing.assert_close(profile.activation_probability(scores, 2, 1), torch.tensor([[.2, 1.]]))
        data['activation_calibration'] = {'1': [[0, 0], [.5, .25], [1, 1]]}
        torch.testing.assert_close(CostProfile(data, 2).activation_probability(scores, 2, 1), torch.tensor([[.1, 1.]]))
        for broken in [dict(data, world_size=4), dict(data, calibrated=None),
                       dict(data, link_bandwidth_bytes_per_ms=[[1, 0], [1, 1]])]:
            with self.assertRaises(ValueError): CostProfile(broken, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'costs.json'
            path.write_text(json.dumps(data))
            self.assertEqual(CostProfile.load(path, 2).digest, CostProfile.load(path, 2).digest)

    def test_two_rank_exchange_empty_source_and_router_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = (Path(tmp) / 'gloo').as_uri()
            ctx = mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=False)
            deadline = time.monotonic() + 45
            while not ctx.join(timeout=1):
                if time.monotonic() > deadline:
                    for process in ctx.processes: process.terminate()
                    self.fail('joint collective/numerical test exceeded 45 seconds')


if __name__ == '__main__':
    unittest.main()
