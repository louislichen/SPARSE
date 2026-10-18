"""Exercise production cache and demand methods without loading a model."""
import ast
from collections import OrderedDict
from dataclasses import dataclass
import itertools
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

from utils.rs_staging import DemandCacheView, TargetJob, Transfer, select_candidates
from utils.rs_overlap import BoundedRS


ROOT = Path(__file__).resolve().parents[1]


def production_methods():
    source = ROOT / 'utils/eval_core.py'
    tree = ast.parse(source.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ('CacheEntry', 'ExpertWeightCache')]
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'DistributedSparseMoeBlock')
    names = ('_cache_contains_no_touch', '_get_local_module_weights', '_maybe_wait_prefetch')
    methods = [node for node in block.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                         names=[ast.alias(name='annotations')], level=0), *classes, *methods], type_ignores=[])
    waits = []
    namespace = dict(OrderedDict=OrderedDict, dataclass=dataclass,
                     torch=NS(cuda=NS(current_stream=lambda _: NS(wait_event=waits.append))),
                     _env_flag=lambda *args: False)
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace, waits


class Event:
    def __init__(self, complete):
        self.complete = complete
    def query(self):
        return self.complete
    def synchronize(self):
        raise AssertionError('Demand must not CPU-synchronize an expert copy')


class StagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.methods, cls.waits = production_methods()
    def setUp(self):
        self.waits.clear()
        self.cache = self.methods['ExpertWeightCache'](2)
        self.cache.put(0, 'w0')
        self.cache.put(4, 'w4')
        self.view = DemandCacheView(self.cache)

    def transfer(self, eid, complete=True):
        return Transfer(eid, 'w%d' % eid, Event(complete), object(), 16)

    def test_wrong_prefetch_never_evicts_or_changes_lru(self):
        self.view.staged = {8: self.transfer(8)}
        self.assertEqual(self.cache.resident_ids(), [0, 4])
        self.assertEqual(self.view.get(0), 'w0')
        self.assertEqual(self.cache.resident_ids(), [4, 0])
        self.assertFalse(self.view.staged[8].used)
        self.assertEqual(self.view.capacity, 2)

    def test_ready_stage_is_visible_to_real_epipe_classification(self):
        self.view.staged = {8: self.transfer(8)}
        block = NS(_cache=self.view, _epipe_prefill_extra_cache_contains=lambda _: False)
        self.assertTrue(self.methods['_cache_contains_no_touch'](block, 8))
        self.assertEqual(self.cache.resident_ids(), [0, 4])

    def test_inflight_stage_reuses_exact_payload_without_second_copy(self):
        transfer = self.transfer(8, complete=False)
        self.view.staged = {8: transfer}
        hits = []
        block = NS(oracle_mode='none', _cache=self.view, metrics=NS(), device='cuda:0',
                   prefetch_stream=object(), _prefetch_events={8: transfer.event},
                   _prefetched_ids_this_group={8}, _prefetch_used_ids_this_group=set(),
                   _transition_prefetched_ids=set(),
                   _maybe_upgrade_prefill_scp_transient_on_decode=lambda _: None,
                   _record_cache_hit=lambda **kwargs: hits.append(kwargs),
                   _record_rs_counter=lambda *args: None,
                   _epipe_prefill_extra_cache_get=lambda *args, **kwargs: self.fail('Second load path'))
        block._maybe_wait_prefetch = lambda eid: self.methods['_maybe_wait_prefetch'](block, eid)
        result = self.methods['_get_local_module_weights'](block, 8)
        self.assertEqual(result, transfer.payload)
        self.assertEqual(self.waits, [transfer.event])
        self.assertTrue(transfer.used)
        self.assertEqual(hits, [dict(is_prefetch_hit=True)])
        self.assertEqual(self.cache.resident_ids(), [0, 4])

    def test_native_hit_has_precedence_over_staging(self):
        transfer = self.transfer(0)
        transfer.payload = 'wrong-shadow'
        self.view.staged = {0: transfer}
        self.assertEqual(self.view.get(0), 'w0')
        self.assertFalse(transfer.used)

    def test_future_not_joined_when_target_closes(self):
        job = TargetJob((1, 1, 2), [])
        class Future:
            def result(self, *args, **kwargs):
                raise AssertionError('No prediction join')
        job.future = Future()
        self.assertEqual(job.close(), {})
        self.assertFalse(job.publish(8, lambda: self.fail('Late duplicate H2D')))

    def test_close_and_submission_have_single_owner(self):
        entered, continue_submit = threading.Event(), threading.Event()
        closed = threading.Event()
        job = TargetJob((1, 1, 2), [])
        def submit():
            entered.set()
            self.assertTrue(continue_submit.wait(2))
            return self.transfer(8, False)
        worker = threading.Thread(target=lambda: job.publish(8, submit))
        worker.start()
        self.assertTrue(entered.wait(2))
        result = []
        def close():
            result.append(job.close())
            closed.set()
        consumer = threading.Thread(target=close)
        consumer.start()
        self.assertFalse(closed.is_set())
        continue_submit.set()
        worker.join(2); consumer.join(2)
        self.assertTrue(closed.is_set())
        self.assertEqual(set(result[0]), {8})
        self.assertFalse(job.publish(4, lambda: self.fail('Target already executing')))

    def test_no_wrong_forecast_changes_any_resident_access_sequence(self):
        for sequence in itertools.product((0, 4), repeat=6):
            baseline = self.methods['ExpertWeightCache'](2)
            candidate = self.methods['ExpertWeightCache'](2)
            for eid in (0, 4):
                baseline.put(eid, str(eid)); candidate.put(eid, str(eid))
            view = DemandCacheView(candidate)
            for eid in sequence:
                view.staged = {8: self.transfer(8)}
                self.assertEqual(view.get(eid), baseline.get(eid))
                self.assertEqual(view.resident_ids(), baseline.resident_ids())
                self.assertFalse(view.staged[8].used)

    def test_selection_excludes_residents_and_wrong_owner(self):
        result = select_candidates([0, 8, 8, 9, 12], [.9, .1, .1, 1., .15],
            allowed={0, 8, 12}, resident={0}, minimum=.2, budget=2)
        self.assertEqual(result, [8])

    def test_actual_use_defers_cache_update_and_recycles_victim_storage(self):
        def payload(name):
            return (NS(name=name, shape=(3, 2), dtype='float16'),)
        resident = self.methods['ExpertWeightCache'](2)
        old0, old4, new8, new12 = (payload(n) for n in ('old0', 'old4', 'new8', 'new12'))
        resident.put(0, old0); resident.put(4, old4)
        view = DemandCacheView(resident)
        for eid, value in ((8, new8), (12, new12)):
            view.staged[eid] = Transfer(eid, value, Event(True), {'payload': value}, 12)
        access = []
        owner = NS(views={0: view}, stats=dict(retained=0, recycled_payloads=0, allocated_replacements=0))
        block = NS(layer_idx=0, _on_demand_access=access.append,
                   _cache_put_guarded=lambda eid, value, **kw: resident.put(eid, value),
                   _should_prefer_hot=lambda *args: False)
        # Issuance/usage cannot evict a resident that this target still needs.
        self.assertIs(view.get(12), new12)
        self.assertIs(view.get(8), new8)
        self.assertIs(view.get(12), new12)
        self.assertEqual(resident.resident_ids(), [0, 4])
        self.assertEqual([x.expert for x in view.used_in_order], [12, 8])
        # Target exit admits only actual accesses, in their first-use order.
        for transfer in view.used_in_order:
            BoundedRS._retain_used(owner, block, transfer)
        self.assertEqual(resident.resident_ids(), [12, 8])
        self.assertEqual(access, [12, 8])
        self.assertIs(view.staged[12].slot['payload'], old0)
        self.assertIs(view.staged[8].slot['payload'], old4)
        self.assertIs(resident.peek(12), new12)
        self.assertEqual(owner.stats, dict(retained=2, recycled_payloads=2, allocated_replacements=0))

    def test_cold_true_demand_retention_allocates_only_replacement_payload(self):
        tensor = NS(shape=(3, 2), dtype='float16')
        resident = self.methods['ExpertWeightCache'](2)
        view = DemandCacheView(resident)
        transfer = Transfer(8, (tensor,), Event(True), {'payload': (tensor,)}, 12)
        view.staged[8] = transfer
        owner = NS(views={0: view}, stats=dict(retained=0, recycled_payloads=0, allocated_replacements=0),
                   torch=NS(empty_like=lambda x: NS(shape=x.shape, dtype=x.dtype)))
        block = NS(layer_idx=0, _on_demand_access=lambda _: None,
                   _cache_put_guarded=lambda eid, value, **kw: resident.put(eid, value),
                   _should_prefer_hot=lambda *args: False)
        view.get(8)
        BoundedRS._retain_used(owner, block, transfer)
        self.assertIs(resident.peek(8)[0], tensor)
        self.assertIsNot(transfer.slot['payload'][0], tensor)
        self.assertEqual(owner.stats['allocated_replacements'], 1)

    def test_real_pipeline_has_no_source_finally_future_join(self):
        # Structural guard supplements actual lifecycle tests: source return
        # may release buffers, but may not synchronously consume prediction.
        source = (ROOT/'utils/rs_overlap.py').read_text()
        tree = ast.parse(source)
        forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                       and node.name == 'forward')
        calls = [node.func.attr for node in ast.walk(forward) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('result', calls)
        self.assertNotIn('synchronize', calls)


if __name__ == '__main__':
    unittest.main()
