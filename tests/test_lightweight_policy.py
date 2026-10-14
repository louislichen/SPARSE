import ast
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from utils.lightweight_policy import CostProfile, LocalPolicy
try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as F
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def profile(directory):
    path = Path(directory)/'costs.json'
    path.write_text(json.dumps(dict(schema_version=1,world_size=2,calibrated=False,
        rank_defaults=dict(h2d_bytes_per_ms=10,h2d_latency_ms=0,expert_ms=.2,item_ms=.1,
                           next_attention_ms=1,split_overhead_ms=.01))))
    return CostProfile(path,2)


class PolicyTests(unittest.TestCase):
    def test_prefetch_uses_cover_without_discounting_resident_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=LocalPolicy(profile(directory),0,1,{0:10,1:100,2:10},())
            policy.window_ms=2
            policy.observe_ready(2,8)  # 1.2 ms already available for overlap
            values=policy.scores({0:.5,1:.5,2:.9},{2})
            self.assertEqual(values[0],0)
            self.assertEqual(values[1],.5)
            self.assertEqual(values[2],.9)
            self.assertEqual(policy.stats['candidate_scores'],3)
            policy.reset()
            self.assertEqual(policy.ready_ms,0)
            self.assertEqual(policy.stats['candidate_scores'],0)

    def test_residual_prefers_coverable_transfer_and_preserves_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=LocalPolicy(profile(directory),0,1,{0:1,1:100,2:100},())
            choices=[(0,0,'late'),(1,1,'late'),(2,2,'demand')]
            self.assertEqual(policy.residual(1,1,choices),(1,'late'))
            self.assertIsNone(policy.residual(0,0,choices))
            policy.window_ms=0
            self.assertEqual(policy.scores({1:.7},set()),{1:.7})

    def test_hot_policy_source_has_no_device_or_distributed_operations(self):
        tree=ast.parse((ROOT/'utils/lightweight_policy.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(name and ('torch' in name or 'distributed' in name) for name in imports))


def methods(context):
    tree=ast.parse((ROOT/'utils/eval_core.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DistributedSparseMoeBlock')
    names={'_light_destinations','_resolve_dest_for_flat_experts','_plain_ep_compute_segment'}
    nodes=[n for n in block.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace=dict(torch=torch,F=F,time=time,SparseContext=context)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<actual-local-policy-serving>', 'exec'),namespace)
    return {n.name:namespace[n.name] for n in nodes}


class Block:
    def __init__(self,rank,profile):
        self.comm_rank,self.comm_world,self.comm_group=rank,2,dist.group.WORLD
        self.device,self.dtype=torch.device('cpu'),torch.float32
        self.rr_expert_to_rank=torch.tensor([0,1])
        self.rr_expert_to_rank_cpu=[0,1]
        self.residents={1-rank}
        self.pending=False
        self._cache=SimpleNamespace(resident_ids=lambda:sorted(self.residents))
        self._light_policy=LocalPolicy(profile,rank,0,{0:64,1:64},{1-rank})
        self.placement_mode,self.experts_kind,self.expert_offload='scp_ep','modulelist','cpu'
        self.hidden_size,self.act_fn=3,F.silu
        torch.manual_seed(19)
        self.weights=[(torch.randn(4,3),torch.randn(4,3),torch.randn(3,4)) for _ in range(2)]
    def _current_stage(self):return 'decode'
    def _local_prefetch_ready(self,eid):return eid in self.residents and not self.pending
    def _get_local_module_weights(self,eid):return self.weights[eid]


def worker(rank,rendezvous,directory):
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        block=Block(rank,CostProfile(Path(directory)/'costs.json',2)); context=SimpleNamespace(current_phase='decode')
        core=methods(context)
        block._light_destinations=lambda ids:core['_light_destinations'](block,ids)
        for empty_rank in (None,1):
            ids=torch.tensor([[0,1]]) if rank!=empty_rank else torch.empty((0,2),dtype=torch.long)
            weights=torch.tensor([[.3,.7]]) if ids.numel() else torch.empty((0,2))
            hidden=torch.tensor([[1.,2.,3.]])+rank if ids.numel() else torch.empty((0,3))
            destinations=core['_resolve_dest_for_flat_experts'](block,ids.reshape(-1))
            if ids.numel():assert destinations.tolist()==[rank,rank]
            order=torch.argsort(destinations)
            send_counts=torch.bincount(destinations,minlength=2).long(); recv_counts=torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts,send_counts)
            sends,receives=send_counts.tolist(),recv_counts.tolist()
            send_hidden=hidden.repeat_interleave(2,dim=0)[order]
            recv_hidden=torch.empty((sum(receives),3)); recv_ids=torch.empty(sum(receives),dtype=torch.long)
            recv_weights=torch.empty(sum(receives))
            for out,inp in ((recv_hidden,send_hidden),(recv_ids,ids.reshape(-1)[order]),(recv_weights,weights.reshape(-1)[order])):
                dist.all_to_all_single(out,inp,output_split_sizes=receives,input_split_sizes=sends)
            computed=torch.empty_like(recv_hidden)
            for eid in torch.unique(recv_ids).tolist():
                mask=recv_ids==eid
                y,_=core['_plain_ep_compute_segment'](block,eid=eid,cur=recv_hidden[mask],cur_w=recv_weights[mask,None],eid_is_local=True,allow_nonlocal=False)
                computed[mask]=y
            returned=torch.empty_like(send_hidden)
            dist.all_to_all_single(returned,computed,output_split_sizes=sends,input_split_sizes=receives)
            restored=torch.empty_like(returned); restored[order]=returned
            actual=restored.reshape(len(hidden),2,3).sum(dim=1)
            expected=torch.zeros_like(hidden)
            for token in range(len(hidden)):
                for slot in range(2):
                    gw,uw,dw=block.weights[ids[token,slot]]
                    expected[token]+=F.linear(F.silu(F.linear(hidden[token],gw))*F.linear(hidden[token],uw),dw)*weights[token,slot]
            torch.testing.assert_close(actual,expected)
        # Event readiness changes invalidate the route-table cache without a
        # distributed exchange or copying routed IDs to the host.
        block.pending=True
        assert core['_light_destinations'](block,torch.tensor([0,1])).tolist()==[0,1]
        block.pending=False
        assert core['_light_destinations'](block,torch.tensor([0,1])).tolist()==[rank,rank]
        tree=ast.parse((ROOT/'scripts/perf.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gather_lightweight_stats')
        namespace=dict(dist=dist,SparseContext=SimpleNamespace(layer_blocks={0:block}))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-perf-stats>','exec'),namespace)
        summaries=namespace[node.name](SimpleNamespace(world=2))
        assert len(summaries)==2 and all(len(v)==1 for v in summaries)
    finally:dist.destroy_process_group()


@unittest.skipIf(torch is None,'PyTorch required')
class IntegrationTests(unittest.TestCase):
    def test_two_rank_local_replica_empty_source_and_pending_event(self):
        with tempfile.TemporaryDirectory() as directory:
            profile(directory)
            ctx=mp.spawn(worker,args=((Path(directory)/'gloo').as_uri(),directory),nprocs=2,join=False)
            deadline=time.monotonic()+45
            while not ctx.join(timeout=1):
                if time.monotonic()>deadline:
                    for process in ctx.processes:process.terminate()
                    self.fail('local policy collective/numerical test exceeded 45 seconds')


if __name__=='__main__':unittest.main()
