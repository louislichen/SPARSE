"""Exercise the production lifecycle methods without importing CUDA/Torch."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_boundary_methods(context):
    source = Path(__file__).resolve().parents[1] / 'utils' / 'eval_core.py'
    tree = ast.parse(source.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'DistributedSparseMoeBlock')
    names = {'demote_prefill_scp_transients_for_decode',
             'record_prefill_scp_transient_decode_end'}
    block.body = [n for n in block.body if isinstance(n, ast.FunctionDef) and n.name in names]
    block.bases = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                         names=[ast.alias(name='annotations')], level=0), block], type_ignores=[])
    namespace = {'SparseContext': context}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace['DistributedSparseMoeBlock']


class Cache:
    def __init__(self, ids):
        self.ids = list(ids)
        self.calls = 0

    def prefill_scp_transient_ids(self):
        return list(self.ids)

    def demote_prefill_scp_transients(self):
        self.calls += 1
        return len(self.ids)


def metrics():
    return SimpleNamespace(prefill_scp_transient_resident_at_decode_start=0,
                           prefill_scp_transient_demote_count=0,
                           epipe_prefill_transient_demote_count=0,
                           prefill_scp_transient_resident_at_decode_end=0)


class PrefillCacheBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.caches = [Cache([3, 7]), Cache([11])]
        self.context = SimpleNamespace(layer_blocks={i: SimpleNamespace(_cache=c)
                                       for i, c in enumerate(self.caches)},
                                       prefill_scp_transient_demoted_for_decode=False)
        self.block = load_boundary_methods(self.context)

    def test_demotes_registered_layers_once_per_request(self):
        first = metrics()
        self.block.demote_prefill_scp_transients_for_decode(first)
        self.block.demote_prefill_scp_transients_for_decode(first)
        self.assertEqual([c.calls for c in self.caches], [1, 1])
        self.assertEqual(first.prefill_scp_transient_resident_at_decode_start, 3)
        self.assertEqual(first.prefill_scp_transient_demote_count, 3)
        self.assertEqual(first.epipe_prefill_transient_demote_count, 3)
        self.context.prefill_scp_transient_demoted_for_decode = False
        second = metrics()
        self.block.demote_prefill_scp_transients_for_decode(second)
        self.assertEqual([c.calls for c in self.caches], [2, 2])
        self.assertEqual(second.prefill_scp_transient_demote_count, 3)

    def test_stale_block_class_state_does_not_override_request_context(self):
        self.block.prefill_scp_transient_demoted_for_decode = True
        self.block.layer_blocks = {}  # The wrong registry used by the old code.
        self.block.demote_prefill_scp_transients_for_decode()
        self.assertEqual([c.calls for c in self.caches], [1, 1])
        self.assertTrue(self.context.prefill_scp_transient_demoted_for_decode)

    def test_end_accounting_reads_current_registered_caches(self):
        result = metrics()
        self.caches[0].ids.clear()
        self.block.record_prefill_scp_transient_decode_end(result)
        self.assertEqual(result.prefill_scp_transient_resident_at_decode_end, 1)
        self.block.record_prefill_scp_transient_decode_end(None)


if __name__ == '__main__':
    unittest.main()
