"""Exercise actual serving cache classes without importing CUDA dependencies."""
import ast
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest

try:
    import torch
except ImportError:
    torch = None


def cache_classes():
    source = Path(__file__).resolve().parents[1] / "utils/eval_core.py"
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.ClassDef) and node.name in
             {"CacheEntry", "ExpertWeightCache", "TieredExpertWeightCache"}]
    namespace = dict(dataclass=dataclass, OrderedDict=OrderedDict, Any=Any, Dict=Dict,
                     List=List, Optional=Optional)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class ExpertCacheTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch required for reduced-precision combine check")
    def test_combine_preserves_native_fp16_addition_order(self):
        source = Path(__file__).resolve().parents[1] / "utils/eval_core.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "combine_expert_outputs")
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        combine = namespace["combine_expert_outputs"]
        parts = torch.tensor([[[-2048.], [1.], [2048.]]], dtype=torch.float16)
        ids = torch.tensor([[2, 1, 0]])
        # Native order is 2048 + 1 - 2048, with rounding after each addition.
        self.assertEqual(combine(parts, ids).item(), 0)
        self.assertEqual(parts.sum(dim=1).item(), 1)
        self.assertEqual(combine(parts[:, :0], ids[:, :0]).shape, (1, 1))
        self.assertEqual(combine(parts[:0], ids[:0]).shape, (0, 1))

    def test_demand_prefetch_and_transient_eviction(self):
        classes = cache_classes()
        cache = classes["ExpertWeightCache"](2)
        cache.put(0, "ordinary")
        cache.put(1, "transient", prefill_scp_transient=True)
        self.assertEqual(cache.put(2, "replacement"), 1)
        self.assertTrue(cache.last_evicted_was_prefill_scp_transient())
        self.assertTrue(cache.admit_prefetch(3, "future", preferred_victim=0))
        self.assertEqual(set(cache.resident_ids()), {2, 3})
        self.assertNotIn(3, cache.evictable_prefetch_ids())
        cache.unpin_all()
        self.assertIn(3, cache.evictable_prefetch_ids())
        cache.clear()
        self.assertTrue(cache.admit_prefetch(4, "free-slot"))
        self.assertEqual(cache.get(4), "free-slot")
        tiered = classes["TieredExpertWeightCache"](3, hot_capacity=1, prefetch_reserved=1)
        tiered.put(0, "hot", prefer_hot=True)
        tiered.put(1, "demand")
        self.assertTrue(tiered.admit_prefetch(2, "future"))
        self.assertEqual(set(tiered.resident_ids()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
