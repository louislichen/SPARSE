"""Check the actual optional dispatch without importing Torch, CUDA or Triton."""
import ast
import hashlib
import os
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def block_tree():
    return next(x for x in ast.parse((ROOT/'utils/eval_core.py').read_text()).body
                if isinstance(x,ast.ClassDef)and x.name=='DistributedSparseMoeBlock')


def forward():
    fn=next(x for x in block_tree().body if isinstance(x,ast.FunctionDef)and x.name=='_module_expert_forward')
    namespace={'F':SimpleNamespace(linear=lambda *args:2)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'dispatch','exec'),namespace)
    return namespace['_module_expert_forward']


class FixedPrefillMath(unittest.TestCase):
    def test_fixed64_dispatches_only_prefill_with_exact_arguments(self):
        calls=[];module=ModuleType('utils.kernels.fixed_tile_expert')
        module.expert=lambda *args,**kw:calls.append((args,kw))or'fixed-result'
        fn=forward();inputs=SimpleNamespace(shape=(3,2));weights=(object(),object(),object())
        block=SimpleNamespace(_prefill_expert_math='fixed64',_current_stage=lambda:'prefill',
            _native_decode_granularity=False,act_fn=lambda x:x)
        with patch.dict(sys.modules,{'utils.kernels.fixed_tile_expert':module}):
            self.assertEqual(fn(block,inputs,*weights),'fixed-result')
            self.assertEqual(calls,[((inputs,*weights),{'row_tile':64})])
            block._current_stage=lambda:'decode'
            self.assertEqual(fn(block,inputs,*weights),2)
            self.assertEqual(len(calls),1)
    def test_native_default_does_not_import_optional_kernel(self):
        module=ModuleType('utils.kernels.fixed_tile_expert')
        module.expert=lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Unexpected optional kernel'))
        block=SimpleNamespace(_current_stage=lambda:'prefill',_native_decode_granularity=False,act_fn=lambda x:x)
        with patch.dict(sys.modules,{'utils.kernels.fixed_tile_expert':module}):
            self.assertEqual(forward()(block,SimpleNamespace(shape=(1,2)),None,None,None),2)
    def test_constructor_normalizes_and_rejects_unknown_math_version(self):
        init=next(x for x in block_tree().body if isinstance(x,ast.FunctionDef)and x.name=='__init__')
        index=next(i for i,x in enumerate(init.body)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Attribute)and t.attr=='_prefill_expert_math'for t in x.targets))
        fn=ast.FunctionDef(name='configure',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),
                           body=init.body[index:index+2],decorator_list=[])
        ns={'os':os};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'configure','exec'),ns)
        with patch.dict(os.environ,{},clear=True):
            block=SimpleNamespace();ns['configure'](block);self.assertEqual(block._prefill_expert_math,'native')
            os.environ['SPARSE_PREFILL_EXPERT_MATH']=' FIXED64 ';ns['configure'](block);self.assertEqual(block._prefill_expert_math,'fixed64')
            os.environ['SPARSE_PREFILL_EXPERT_MATH']='unknown'
            with self.assertRaises(ValueError):ns['configure'](block)
    def test_kernel_arithmetic_matches_the_frozen_measured_implementation(self):
        source=(ROOT/'utils/kernels/fixed_tile_expert.py').read_text()
        lines=source.splitlines()
        functions=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)]
        segments=['\n'.join(lines[min([x.lineno,*[d.lineno for d in x.decorator_list]])-1:x.end_lineno])for x in functions]
        digest=hashlib.sha256('\n\n'.join(segments).encode()).hexdigest()
        self.assertEqual(digest,'ea14dc27e3f8a8ac18d08ab05e0b10e5f16de787191cae8e6528d4cd59c97d58')


if __name__=='__main__':unittest.main()
