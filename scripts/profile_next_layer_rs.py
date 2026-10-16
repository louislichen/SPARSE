from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import random
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _install_flash_attn_import_stub() -> bool:
    """Allow DeepSeek remote code to import flash_attn when eager attention is used."""
    if "flash_attn" in sys.modules:
        return False
    try:
        if importlib.util.find_spec("flash_attn") is not None:
            return False
    except Exception:
        pass

    def _unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "flash_attn import stub was called during RS profiling. "
            "Use --attn-implementation eager or install flash-attn."
        )

    def _missing_attr(_name: str) -> Any:
        return _unavailable

    flash_mod = types.ModuleType("flash_attn")
    flash_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_mod.flash_attn_func = _unavailable
    flash_mod.flash_attn_varlen_func = _unavailable
    flash_mod.flash_attn_qkvpacked_func = _unavailable
    flash_mod.flash_attn_varlen_qkvpacked_func = _unavailable
    flash_mod.flash_attn_with_kvcache = _unavailable
    flash_mod.__getattr__ = _missing_attr

    interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    interface_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    interface_mod.flash_attn_func = _unavailable
    interface_mod.flash_attn_varlen_func = _unavailable
    interface_mod.flash_attn_qkvpacked_func = _unavailable
    interface_mod.flash_attn_varlen_qkvpacked_func = _unavailable
    interface_mod.flash_attn_with_kvcache = _unavailable
    interface_mod.__getattr__ = _missing_attr

    bert_padding_mod = types.ModuleType("flash_attn.bert_padding")
    bert_padding_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding_mod.index_first_axis = _unavailable
    bert_padding_mod.pad_input = _unavailable
    bert_padding_mod.unpad_input = _unavailable
    bert_padding_mod.__getattr__ = _missing_attr

    layers_mod = types.ModuleType("flash_attn.layers")
    layers_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers", loader=None)
    layers_mod.__path__ = []
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn.layers.rotary", loader=None)
    rotary_mod.apply_rotary_emb_func = _unavailable
    rotary_mod.__getattr__ = _missing_attr

    sys.modules["flash_attn"] = flash_mod
    sys.modules["flash_attn.flash_attn_interface"] = interface_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding_mod
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod
    return True


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _profiling_model(model, decode_token_source: str):
    # Dataset tokens supply the next input, so the LM head has no consumer.
    # Keep the native decoder, KV cache and gate hooks identical to generation.
    return model.model if decode_token_source == "dataset" else model


def _torch_save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    if (not tmp.exists()) or tmp.stat().st_size <= 0:
        raise RuntimeError(f"torch.save produced empty temp file: {tmp}")
    os.replace(tmp, path)
    if (not path.exists()) or path.stat().st_size <= 0:
        raise RuntimeError(f"saved file is empty: {path}")


def _open_jsonl_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def _prompt_text_from_dataset_row(ex: Any) -> str:
    if not isinstance(ex, dict):
        return ""
    for key in ("text", "prompt", "input", "query", "instruction"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    messages = ex.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                content = msg["content"].strip()
                if content:
                    parts.append(content)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _read_dataset_token_samples(
    path: str,
    tokenizer,
    num_samples: int,
    min_len: int,
    *,
    max_scan_samples: int = 0,
) -> List[torch.Tensor]:
    samples: List[torch.Tensor] = []
    scanned = 0
    with _open_jsonl_maybe_gz(Path(path)) as f:
        for line in f:
            scanned += 1
            if int(max_scan_samples) > 0 and scanned > int(max_scan_samples):
                break
            if len(samples) >= int(num_samples):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = _prompt_text_from_dataset_row(obj)
            if not text:
                continue
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                continue
            if len(ids) < int(min_len):
                continue
            samples.append(torch.tensor(ids[:min_len], dtype=torch.long))
    return samples


def _read_c4_jsonl_gz(path: str, tokenizer, num_samples: int, total_len: int) -> List[torch.Tensor]:
    return _read_dataset_token_samples(path, tokenizer, num_samples, total_len)


def _to_cpu_half(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", dtype=torch.float16).contiguous()


def _to_cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", dtype=torch.float32).contiguous()


def _to_cpu_int16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", dtype=torch.int16).contiguous()


@dataclass
class LayerBuffer:
    inputs: List[torch.Tensor] = field(default_factory=list)
    indices: List[torch.Tensor] = field(default_factory=list)
    weights: List[torch.Tensor] = field(default_factory=list)
    logits: List[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.inputs.clear()
        self.indices.clear()
        self.weights.clear()
        self.logits.clear()


@dataclass
class TraceBuffer:
    sample_ids: List[int] = field(default_factory=list)
    decode_steps: List[int] = field(default_factory=list)
    layer_indices: List[torch.Tensor] = field(default_factory=list)
    layer_weights: List[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.sample_ids.clear()
        self.decode_steps.clear()
        self.layer_indices.clear()
        self.layer_weights.clear()


class NextLayerRSProfiler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        if (self.out_dir/'manifest.json').exists() or any(self.out_dir.glob('rs_decode_chunks/*.pt')):
            raise ValueError('profile output already contains traces; use a fresh versioned directory')
        self.chunk_dir = self.out_dir / "rs_decode_chunks"
        self.trace_dir = self.out_dir / "trace_chunks"
        self.summary_dir = self.out_dir / "summary"
        _ensure_dir(self.chunk_dir)
        _ensure_dir(self.trace_dir)
        _ensure_dir(self.summary_dir)

        torch.manual_seed(int(args.seed))
        random.seed(int(args.seed))

        print(f"[Load] tokenizer/model: {args.model_path}", flush=True)
        if bool(args.flash_attn_import_stub) or (
            "deepseek" in str(args.model_path).lower() and str(args.attn_implementation).strip() in ("", "eager", "sdpa")
        ):
            _install_flash_attn_import_stub()

        hf_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": bool(args.local_files_only),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, **hf_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        max_memory = None
        if args.device_map == "auto":
            max_memory = {i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())}
            max_memory["cpu"] = args.max_memory_cpu

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=args.device_map,
            max_memory=max_memory,
            torch_dtype=torch.float16,
            attn_implementation=args.attn_implementation,
            **hf_kwargs,
        )
        self.model.config.output_router_logits = False
        self.model.eval()
        torch.set_grad_enabled(False)

        cfg = self.model.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_experts = int(
            getattr(
                cfg,
                "num_experts",
                getattr(cfg, "n_routed_experts", getattr(cfg, "num_routed_experts", 0)),
            )
        )
        if self.num_experts <= 0:
            raise RuntimeError("Cannot infer number of routed experts from model config.")
        self.top_k = int(getattr(cfg, "num_experts_per_tok", 4))
        self.hidden_size = int(cfg.hidden_size)
        self.first_device = next(self.model.parameters()).device
        self.moe_layer_indices: List[int] = []
        self.moe_layer_set: set[int] = set()

        print(
            f"[Config] layers={self.num_layers} experts/layer={self.num_experts} "
            f"top_k={self.top_k} hidden={self.hidden_size} first_device={self.first_device}",
            flush=True,
        )

        self.capture_enabled = False
        self.current_capture: Dict[int, Dict[str, torch.Tensor]] = {}
        self.debug_shapes: Dict[int, Dict[str, object]] = {}
        self.layer_buffers = {l: LayerBuffer() for l in range(self.num_layers)}
        self.trace_buffer = TraceBuffer()
        self.flush_index = 0
        self.records_in_buffer = 0
        self.total_records = 0

        self.layer_expert_freq = torch.zeros((self.num_layers, self.num_experts), dtype=torch.int64)
        self.layer_weight_mass = torch.zeros((self.num_layers, self.num_experts), dtype=torch.float64)
        self.layer_active_experts_sum = torch.zeros((self.num_layers,), dtype=torch.float64)
        self.layer_entropy_sum = torch.zeros((self.num_layers,), dtype=torch.float64)
        self.layer_record_count = torch.zeros((self.num_layers,), dtype=torch.int64)

        self._register_gate_hooks()
        model_dir = Path(args.model_path)
        fingerprint_files = ('config.json', 'model.safetensors.index.json', 'tokenizer_config.json', 'tokenizer.json')
        self.provenance = dict(
            model_files={name:hashlib.sha256((model_dir/name).read_bytes()).hexdigest()
                         for name in fingerprint_files if (model_dir/name).is_file()},
            weight_file_metadata=[(p.name,p.stat().st_size,p.stat().st_mtime_ns)
                                  for p in sorted(model_dir.glob('*.safetensors'))],
            dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
            input_feature='native MoE gate input hidden; FP16 capture, FP32 predictor input',
            label='same-token next-layer native router top-k',
            forward_model='native decoder without LM head' if args.decode_token_source == 'dataset' else 'native causal LM',
            torch=str(torch.__version__), seed=int(args.seed),
        )

    def _register_gate_hooks(self) -> None:
        hook_count = 0
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            experts = getattr(mlp, "experts", None)
            if gate is None or experts is None:
                continue
            gate.register_forward_hook(self._make_gate_hook(layer_idx, mlp))
            self.moe_layer_indices.append(int(layer_idx))
            self.moe_layer_set.add(int(layer_idx))
            hook_count += 1
        if hook_count <= 0:
            raise RuntimeError("Registered no MoE gate hooks. Check model structure.")
        print(
            f"[Hook] registered {hook_count} MoE gate hooks layers={self.moe_layer_indices}",
            flush=True,
        )

    def _make_gate_hook(self, layer_idx: int, mlp_module):
        def hook_fn(module, hook_args, output):
            if not self.capture_enabled:
                return
            hidden_in = hook_args[0] if len(hook_args) > 0 else None
            if hidden_in is None or output is None:
                return

            if hidden_in.dim() == 3:
                hidden_last = hidden_in[:, -1, :]
            elif hidden_in.dim() == 2:
                hidden_last = hidden_in
            else:
                raise RuntimeError(f"Unexpected gate hidden shape at layer {layer_idx}: {tuple(hidden_in.shape)}")

            logits_last: Optional[torch.Tensor] = None
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                topi_raw, topw_raw = output[0], output[1]
                if not torch.is_tensor(topi_raw) or not torch.is_tensor(topw_raw):
                    return
                if topi_raw.dim() == 3:
                    topi = topi_raw[:, -1, :].reshape(-1, topi_raw.shape[-1]).to(torch.long)
                elif topi_raw.dim() == 2:
                    topi = topi_raw.reshape(-1, topi_raw.shape[-1]).to(torch.long)
                else:
                    raise RuntimeError(f"Unexpected gate topk index shape at layer {layer_idx}: {tuple(topi_raw.shape)}")
                if topw_raw.dim() == 3:
                    topw = topw_raw[:, -1, :].reshape(-1, topw_raw.shape[-1]).float()
                elif topw_raw.dim() == 2:
                    topw = topw_raw.reshape(-1, topw_raw.shape[-1]).float()
                else:
                    raise RuntimeError(f"Unexpected gate topk weight shape at layer {layer_idx}: {tuple(topw_raw.shape)}")
                if int(topi.shape[-1]) != int(self.top_k):
                    self.top_k = int(topi.shape[-1])
            else:
                router_logits = output
                if not torch.is_tensor(router_logits):
                    return
                if router_logits.dim() == 3:
                    logits_last = router_logits[:, -1, :].reshape(-1, self.num_experts)
                elif router_logits.dim() == 2:
                    logits_last = router_logits.reshape(-1, self.num_experts)
                else:
                    raise RuntimeError(f"Unexpected gate output shape at layer {layer_idx}: {tuple(router_logits.shape)}")
                probs = F.softmax(logits_last.float(), dim=-1)
                topw, topi = torch.topk(probs, k=self.top_k, dim=-1)
                if bool(getattr(mlp_module, "norm_topk_prob", True)):
                    topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if hidden_last.shape[0] != topi.shape[0]:
                raise RuntimeError(
                    f"Batch mismatch at layer {layer_idx}: hidden={tuple(hidden_last.shape)} topk={tuple(topi.shape)}"
                )

            if layer_idx not in self.debug_shapes:
                self.debug_shapes[layer_idx] = {
                    "hidden_in_shape": tuple(hidden_in.shape),
                    "gate_output_type": type(output).__name__,
                    "gate_output_shape": tuple(output.shape) if torch.is_tensor(output) else [tuple(x.shape) if torch.is_tensor(x) else str(type(x)) for x in output],
                    "stored_hidden_shape": tuple(hidden_last.shape),
                    "stored_indices_shape": tuple(topi.shape),
                    "stored_weights_shape": tuple(topw.shape),
                }

            self.current_capture[layer_idx] = {
                "hidden": _to_cpu_half(hidden_last),
                "indices": _to_cpu_int16(topi),
                "weights": _to_cpu_float(topw),
                "logits": _to_cpu_float(logits_last) if self.args.save_router_logits and logits_last is not None else None,
            }

        return hook_fn

    def profile(self) -> None:
        decode_source = str(self.args.decode_token_source)
        min_len = int(self.args.prompt_len + self.args.decode_steps) if decode_source == "dataset" else int(self.args.prompt_len)
        samples = _read_dataset_token_samples(
            self.args.dataset,
            self.tokenizer,
            num_samples=int(self.args.num_samples),
            min_len=min_len,
            max_scan_samples=int(self.args.max_scan_samples),
        )
        if len(samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.args.dataset} with token length >= {min_len}")
        print(
            f"[Data] samples={len(samples)} prompt_len={self.args.prompt_len} "
            f"decode_steps={self.args.decode_steps} decode_token_source={decode_source} "
            f"profile_batch_size={self.args.profile_batch_size}",
            flush=True,
        )

        t0 = time.perf_counter()
        for start in tqdm(range(0, len(samples), int(self.args.profile_batch_size)), desc="profile batches"):
            batch = samples[start:start + int(self.args.profile_batch_size)]
            self._profile_batch(start, batch)
            if self.records_in_buffer >= int(self.args.save_interval_records):
                self.flush()
        self.flush(force=True)
        self.save_summary(time.perf_counter() - t0)

    def _profile_batch(self, sample_start: int, batch: List[torch.Tensor]) -> None:
        prompt_len = int(self.args.prompt_len)
        decode_steps = int(self.args.decode_steps)
        ids = torch.stack(batch, dim=0)
        prompt_ids = ids[:, :prompt_len].to(self.first_device)
        target_ids = None
        if str(self.args.decode_token_source) == "dataset":
            target_ids = ids[:, prompt_len:prompt_len + decode_steps].to(self.first_device)
        batch_size = int(prompt_ids.shape[0])
        forward_model = _profiling_model(self.model, str(self.args.decode_token_source))

        self.capture_enabled = False
        with torch.no_grad():
            prefill = forward_model(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids, device=prompt_ids.device),
                use_cache=True,
                return_dict=True,
            )
        past_key_values = prefill.past_key_values
        generated_input = None
        if target_ids is None:
            generated_input = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        del prefill

        for step in range(decode_steps):
            self.current_capture = {}
            if target_ids is not None:
                cur_input = target_ids[:, step].reshape(batch_size, 1)
            else:
                cur_input = generated_input.reshape(batch_size, 1)
            cur_attn_mask = torch.ones(
                (batch_size, prompt_len + step + 1),
                dtype=torch.long,
                device=self.first_device,
            )

            self.capture_enabled = True
            with torch.no_grad():
                out = forward_model(
                    input_ids=cur_input,
                    attention_mask=cur_attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            self.capture_enabled = False
            past_key_values = out.past_key_values
            if target_ids is None:
                generated_input = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            del out

            missing_layers = [l for l in self.moe_layer_indices if l not in self.current_capture]
            if missing_layers:
                raise RuntimeError(
                    f"Incomplete capture at sample_start={sample_start} step={step}: "
                    f"missing MoE layers {missing_layers[:16]}"
                )

            layer_indices = torch.full((batch_size, self.num_layers, self.top_k), -1, dtype=torch.int16)
            layer_weights = torch.zeros((batch_size, self.num_layers, self.top_k), dtype=torch.float32)
            for l in self.moe_layer_indices:
                cap = self.current_capture[l]
                h = cap["hidden"]
                idx = cap["indices"]
                w = cap["weights"]
                if h.shape != (batch_size, self.hidden_size):
                    raise RuntimeError(f"Layer {l} hidden shape {tuple(h.shape)} != {(batch_size, self.hidden_size)}")
                if idx.shape != (batch_size, self.top_k):
                    raise RuntimeError(f"Layer {l} indices shape {tuple(idx.shape)} != {(batch_size, self.top_k)}")

                self.layer_buffers[l].inputs.append(h)
                self.layer_buffers[l].indices.append(idx)
                self.layer_buffers[l].weights.append(w)
                if self.args.save_router_logits:
                    if cap["logits"] is not None:
                        self.layer_buffers[l].logits.append(cap["logits"])

                layer_indices[:, l, :] = idx
                layer_weights[:, l, :] = w

                idx64 = idx.to(torch.long)
                w64 = w.to(torch.float64)
                for b in range(batch_size):
                    self.layer_expert_freq[l, idx64[b]] += 1
                    self.layer_weight_mass[l].scatter_add_(0, idx64[b], w64[b])
                    self.layer_active_experts_sum[l] += float(torch.unique(idx64[b]).numel())
                    self.layer_entropy_sum[l] += float((-(w64[b] * w64[b].clamp_min(1e-12).log())).sum().item())
                    self.layer_record_count[l] += 1

            for b in range(batch_size):
                self.trace_buffer.sample_ids.append(int(sample_start + b))
                self.trace_buffer.decode_steps.append(int(step))
                self.trace_buffer.layer_indices.append(layer_indices[b].clone())
                self.trace_buffer.layer_weights.append(layer_weights[b].clone())

            self.records_in_buffer += batch_size
            self.total_records += batch_size

        del prompt_ids, target_ids, past_key_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def flush(self, force: bool = False) -> None:
        if self.records_in_buffer == 0:
            return
        if (not force) and self.records_in_buffer < int(self.args.save_interval_records):
            return
        self.flush_index += 1
        chunk_id = self.flush_index
        print(f"[Flush] chunk={chunk_id:05d} records={self.records_in_buffer}", flush=True)

        for l in range(self.num_layers):
            buf = self.layer_buffers[l]
            if len(buf.inputs) == 0:
                continue
            inputs = torch.cat(buf.inputs, dim=0).contiguous()
            indices = torch.cat(buf.indices, dim=0).contiguous()
            weights = torch.cat(buf.weights, dim=0).contiguous()
            _torch_save_atomic(inputs, self.chunk_dir / f"rs_inputs_l{l}_chunk_{chunk_id:05d}.pt")
            label_obj = {
                "indices": indices,
                "weights": weights,
                "layer_idx": int(l),
                "num_experts": int(self.num_experts),
                "top_k": int(self.top_k),
                "hidden_size": int(self.hidden_size),
                "label_semantics": "true_router_topk_for_same_layer_decode_token",
            }
            if self.args.save_router_logits and len(buf.logits) > 0:
                label_obj["logits"] = torch.cat(buf.logits, dim=0).contiguous()
            _torch_save_atomic(label_obj, self.chunk_dir / f"rs_labels_l{l}_chunk_{chunk_id:05d}.pt")
            buf.clear()

        trace_obj = {
            "sample_ids": torch.tensor(self.trace_buffer.sample_ids, dtype=torch.int64),
            "decode_steps": torch.tensor(self.trace_buffer.decode_steps, dtype=torch.int32),
            "layer_indices": torch.stack(self.trace_buffer.layer_indices, dim=0).contiguous(),
            "layer_weights": torch.stack(self.trace_buffer.layer_weights, dim=0).contiguous(),
            "num_layers": int(self.num_layers),
            "num_experts_per_layer": int(self.num_experts),
            "top_k": int(self.top_k),
            "moe_layer_indices": list(self.moe_layer_indices),
            "trace_semantics": "one record per decoded token; layer_indices[n,l,k] is true router top-k expert",
        }
        _torch_save_atomic(trace_obj, self.trace_dir / f"decode_trace_chunk_{chunk_id:05d}.pt")
        self.trace_buffer.clear()
        self.records_in_buffer = 0

    def save_summary(self, elapsed_s: float) -> None:
        avg_active = self.layer_active_experts_sum / self.layer_record_count.clamp_min(1).to(torch.float64)
        avg_entropy = self.layer_entropy_sum / self.layer_record_count.clamp_min(1).to(torch.float64)
        summary = {
            "model_path": self.args.model_path,
            "dataset": self.args.dataset,
            "num_layers": int(self.num_layers),
            "num_experts_per_layer": int(self.num_experts),
            "top_k": int(self.top_k),
            "hidden_size": int(self.hidden_size),
            "prompt_len": int(self.args.prompt_len),
            "decode_steps": int(self.args.decode_steps),
            "decode_token_source": str(self.args.decode_token_source),
            "num_profiled_samples": int(self.args.num_samples),
            "num_valid_samples": int(self.total_records // max(int(self.args.decode_steps), 1)),
            "profile_batch_size": int(self.args.profile_batch_size),
            "total_decode_records": int(self.total_records),
            "save_router_logits": bool(self.args.save_router_logits),
            "schema": "next_layer_rs_v1",
            "moe_layer_indices": list(self.moe_layer_indices),
            "rs_chunk_dir": str(self.chunk_dir),
            "trace_chunk_dir": str(self.trace_dir),
            "layer_expert_freq": self.layer_expert_freq,
            "layer_weight_mass": self.layer_weight_mass,
            "layer_active_experts_avg": avg_active,
            "layer_entropy_avg": avg_entropy,
            "debug_hook_shapes": self.debug_shapes,
            "elapsed_s": float(elapsed_s),
        }
        pt_path = self.summary_dir / "decode_profile_summary.pt"
        json_path = self.summary_dir / "decode_profile_summary.json"
        _torch_save_atomic(summary, pt_path)

        json_summary = dict(summary)
        for k in ["layer_expert_freq", "layer_weight_mass", "layer_active_experts_avg", "layer_entropy_avg"]:
            v = json_summary[k]
            json_summary[k] = v.tolist() if torch.is_tensor(v) else v
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)

        manifest = {
            "provenance": self.provenance,
            "schema": "next_layer_rs_v1",
            "rs_chunk_dir": str(self.chunk_dir),
            "trace_chunk_dir": str(self.trace_dir),
            "summary_pt": str(pt_path),
            "summary_json": str(json_path),
            "num_layers": int(self.num_layers),
            "num_experts_per_layer": int(self.num_experts),
            "top_k": int(self.top_k),
            "hidden_size": int(self.hidden_size),
            "prompt_len": int(self.args.prompt_len),
            "decode_steps": int(self.args.decode_steps),
            "decode_token_source": str(self.args.decode_token_source),
            "total_decode_records": int(self.total_records),
            "moe_layer_indices": list(self.moe_layer_indices),
            "input_file_semantics": "rs_inputs_l{l}: gate input hidden for layer l, shape [N,H], fp16",
            "label_file_semantics": "rs_labels_l{l}: true router top-k for layer l, indices [N,K], weights [N,K]",
            "next_layer_training_semantics": "train source layer s inputs against target layer s+lead labels",
        }
        with open(self.out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[Save] manifest={self.out_dir / 'manifest.json'}", flush=True)
        print(f"[Save] summary={json_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile decode hidden/router traces for pure next-layer RS training.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True, help="JSONL/JSONL.GZ path with text/prompt/input/query/instruction/messages")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--decode-token-source", type=str, default="dataset", choices=["dataset", "generated"])
    ap.add_argument("--profile-batch-size", type=int, default=1)
    ap.add_argument("--max-scan-samples", type=int, default=0)
    ap.add_argument("--save-interval-records", type=int, default=4096)
    ap.add_argument("--save-router-logits", action="store_true")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--max-memory-per-gpu", type=str, default="9GiB")
    ap.add_argument("--max-memory-cpu", type=str, default="80GiB")
    ap.add_argument("--attn-implementation", type=str, default="eager")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--flash-attn-import-stub", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    profiler = NextLayerRSProfiler(args)
    profiler.profile()


if __name__ == "__main__":
    main()
