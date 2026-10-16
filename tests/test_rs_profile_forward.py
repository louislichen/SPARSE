from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    from scripts import profile_next_layer_rs as PROFILE
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class RSProfileForwardTest(unittest.TestCase):
    def setUp(self):
        self.grad_enabled = torch.is_grad_enabled()
        self.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = Qwen2MoeConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            moe_intermediate_size=8, shared_expert_intermediate_size=8,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
            max_position_embeddings=32, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        self.model = Qwen2MoeForCausalLM(config).eval()

    def tearDown(self):
        torch.set_grad_enabled(self.grad_enabled)
        torch.set_num_threads(self.thread_count)

    def collect(self, root, source, *, legacy=False):
        args = SimpleNamespace(
            out_dir=str(root / "profile"), seed=42, model_path=str(root),
            dataset=str(root / "data.jsonl"), flash_attn_import_stub=False,
            local_files_only=True, device_map="cpu", attn_implementation="eager",
            save_router_logits=True, decode_token_source=source,
            prompt_len=4, decode_steps=3,
        )
        model = copy.deepcopy(self.model)
        calls = []
        model.lm_head.register_forward_hook(lambda *_: calls.append(1))
        tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
        with patch.object(PROFILE.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
             patch.object(PROFILE.AutoModelForCausalLM, "from_pretrained", return_value=model):
            profiler = PROFILE.NextLayerRSProfiler(args)
        batch = [torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                 torch.tensor([7, 6, 5, 4, 3, 2, 1])]
        if legacy:
            with patch.object(PROFILE, "_profiling_model", return_value=model):
                profiler._profile_batch(10, batch)
        else:
            profiler._profile_batch(10, batch)
        return profiler, calls

    def test_dataset_skips_lm_head_with_identical_native_routing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Both collectors read the same dataset fingerprint and native weights.
            for name in ("legacy", "optimized"):
                (root / name).mkdir()
                (root / name / "data.jsonl").write_text('{}\n')
            legacy, legacy_calls = self.collect(root / "legacy", "dataset", legacy=True)
            optimized, optimized_calls = self.collect(root / "optimized", "dataset")
        self.assertEqual(len(legacy_calls), 4)
        self.assertEqual(optimized_calls, [])
        self.assertEqual(optimized.total_records, 6)
        for layer in range(2):
            for field in ("inputs", "indices", "weights", "logits"):
                self.assertEqual(len(getattr(legacy.layer_buffers[layer], field)),
                                 len(getattr(optimized.layer_buffers[layer], field)))
                for expected, actual in zip(getattr(legacy.layer_buffers[layer], field),
                                            getattr(optimized.layer_buffers[layer], field)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(legacy.trace_buffer.sample_ids, optimized.trace_buffer.sample_ids)
        self.assertEqual(legacy.trace_buffer.decode_steps, optimized.trace_buffer.decode_steps)
        for field in ("layer_indices", "layer_weights"):
            torch.testing.assert_close(torch.stack(getattr(legacy.trace_buffer, field)),
                                       torch.stack(getattr(optimized.trace_buffer, field)), rtol=0, atol=0)

    def test_generated_tokens_still_use_native_lm_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.jsonl").write_text('{}\n')
            profiler, calls = self.collect(root, "generated")
        self.assertEqual(len(calls), 4)
        self.assertEqual(profiler.total_records, 6)


if __name__ == "__main__":
    unittest.main()
