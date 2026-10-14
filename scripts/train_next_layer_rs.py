from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class RoutingScoutMLP(nn.Module):
    """Runtime-compatible RS module: output is a probability distribution over experts."""

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        return self.mlp(x)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RoutingScoutResidualMLP(nn.Module):
    """Stronger runtime-compatible RS module: output is still expert probabilities."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(max(0, int(num_hidden_layers)))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_experts)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        h = self.input(x)
        h = self.blocks(h)
        return self.head(self.out_norm(h))

    def forward(self, x: torch.Tensor, *, return_logits: bool = False) -> torch.Tensor:
        logits = self.forward_logits(x)
        return logits if return_logits else torch.softmax(logits, dim=-1)


def _ddp_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_dist() -> Tuple[int, int, int, torch.device]:
    if _ddp_enabled():
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, torch.device(f"cuda:{local_rank}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def _cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_reduce_sum_float(v: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(v)
    t = torch.tensor([float(v)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_reduce_sum_int(v: int, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(v)
    t = torch.tensor([int(v)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _sorted_chunk_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)

    def chunk_id(path: str) -> int:
        m = re.search(r"_chunk_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(files, key=chunk_id)


def _load_manifest(profile_base_dir: str) -> Dict[str, object]:
    path = Path(profile_base_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {profile_base_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_profile_subdir(profile_base_dir: str, manifest_value: object, fallback_name: str) -> Path:
    base = Path(profile_base_dir)
    raw = str(manifest_value or "")
    p = Path(raw)
    if raw and not p.is_absolute() and (base / p).exists():
        return base / p
    if raw and p.exists():
        return p
    fallback = base / fallback_name
    if fallback.exists():
        return fallback
    return p if raw else fallback


def _paired_files(rs_dir: Path, source_layer: int, target_layer: int) -> List[Tuple[str, str]]:
    src_files = _sorted_chunk_files(str(rs_dir / f"rs_inputs_l{source_layer}_chunk_*.pt"))
    tgt_files = _sorted_chunk_files(str(rs_dir / f"rs_labels_l{target_layer}_chunk_*.pt"))

    def ids(files: List[str]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for fp in files:
            m = re.search(r"_chunk_(\d+)\.pt$", os.path.basename(fp))
            if m:
                out[int(m.group(1))] = fp
        return out

    src_by_id = ids(src_files)
    tgt_by_id = ids(tgt_files)
    common = sorted(set(src_by_id) & set(tgt_by_id))
    return [(src_by_id[i], tgt_by_id[i]) for i in common]


def _load_pair_dataset(
    rs_dir: Path,
    source_layer: int,
    target_layer: int,
    rank: int,
    world: int,
    *,
    fp32_inputs: bool,
    use_router_weights: bool,
    target_mode: str,
    distill_temperature: float,
    feature_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = _paired_files(rs_dir, source_layer, target_layer)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No paired chunks for source L{source_layer} -> target L{target_layer} in {rs_dir}")
    # All ranks must split the same records. DistributedSampler below is the
    # only training shard; sharding files here would discard records twice and
    # could give DDP ranks different numbers of backward calls.
    src_label_by_chunk: Dict[int, str] = {}
    if feature_mode == "hidden_router_topk":
        src_label_files = _sorted_chunk_files(str(rs_dir / f"rs_labels_l{source_layer}_chunk_*.pt"))
        for fp in src_label_files:
            m = re.search(r"_chunk_(\d+)\.pt$", os.path.basename(fp))
            if m:
                src_label_by_chunk[int(m.group(1))] = fp

    xs: List[torch.Tensor] = []
    target_dists: List[torch.Tensor] = []
    target_indices: List[torch.Tensor] = []
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    for x_fp, y_fp in pairs:
        x = torch.load(x_fp, map_location="cpu")
        y = torch.load(y_fp, map_location="cpu")
        indices = y["indices"].long()
        weights = y["weights"].float()
        e = int(y.get("num_experts", int(indices.max().item()) + 1))
        k = int(y.get("top_k", indices.shape[1]))
        num_experts = e if num_experts is None else num_experts
        top_k = k if top_k is None else top_k
        if e != num_experts:
            raise RuntimeError(f"num_experts mismatch in {y_fp}: {e} != {num_experts}")
        if k != top_k:
            raise RuntimeError(f"top_k mismatch in {y_fp}: {k} != {top_k}")

        if target_mode == "logits":
            if "logits" not in y:
                raise KeyError(
                    f"{y_fp} does not contain full router logits. "
                    "Re-run profiling with --save-router-logits or use --target-mode uniform_multihot/router_weights."
                )
            logits = y["logits"].float()
            if tuple(logits.shape) != (indices.shape[0], e):
                raise RuntimeError(f"{y_fp}: logits shape {tuple(logits.shape)} expected [{indices.shape[0]},{e}]")
            dist_target = torch.softmax(logits / max(float(distill_temperature), 1e-6), dim=-1)
        else:
            dist_target = torch.zeros((indices.shape[0], e), dtype=torch.float32)
        if target_mode == "router_weights" or (target_mode == "legacy" and use_router_weights):
            w = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            dist_target.scatter_add_(1, indices, w)
        elif target_mode == "uniform_multihot" or target_mode == "legacy":
            ones = torch.ones_like(weights, dtype=torch.float32) / float(max(k, 1))
            dist_target.scatter_add_(1, indices, ones)
        elif target_mode != "logits":
            raise ValueError(f"Unknown target_mode={target_mode}")

        if feature_mode == "hidden_router_topk":
            m = re.search(r"_chunk_(\d+)\.pt$", os.path.basename(x_fp))
            chunk_id = int(m.group(1)) if m else -1
            src_fp = src_label_by_chunk.get(chunk_id)
            if src_fp is None:
                raise FileNotFoundError(f"Missing source label chunk for feature_mode={feature_mode}: L{source_layer} chunk={chunk_id}")
            src_y = torch.load(src_fp, map_location="cpu")
            src_idx = src_y["indices"].long()
            src_w = src_y["weights"].float()
            if tuple(src_idx.shape) != tuple(indices.shape):
                raise RuntimeError(f"{src_fp}: source indices shape {tuple(src_idx.shape)} != target {tuple(indices.shape)}")
            src_w = src_w / src_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            src_dist = torch.zeros((src_idx.shape[0], e), dtype=torch.float32)
            src_dist.scatter_add_(1, src_idx, src_w)
            x = torch.cat([x.float(), src_dist], dim=-1).contiguous()
        elif feature_mode != "hidden":
            raise ValueError(f"Unknown feature_mode={feature_mode}")

        xs.append(x.float() if fp32_inputs else x.half())
        target_dists.append(dist_target)
        target_indices.append(indices)

    X = torch.cat(xs, dim=0).contiguous()
    Y = torch.cat(target_dists, dim=0).contiguous()
    I = torch.cat(target_indices, dim=0).contiguous()
    return X, Y, I


def _split_indices(n: int, seed: int, groups: Optional[torch.Tensor] = None):
    g = torch.Generator().manual_seed(int(seed))
    if groups is None:
        perm = torch.randperm(n, generator=g)
        train_n, val_n = int(.8 * n), int(.1 * n)
        return perm[:train_n], perm[train_n:train_n + val_n], perm[train_n + val_n:]
    if groups.ndim != 1 or groups.numel() != n:
        raise ValueError('sample IDs must align with every profile record')
    unique = torch.unique(groups, sorted=True)
    if len(unique) < 10:
        raise ValueError('request-disjoint splitting requires at least ten profiled requests')
    order = unique[torch.randperm(len(unique), generator=g)]
    train_n, val_n = int(.8 * len(unique)), int(.1 * len(unique))
    partitions = (order[:train_n], order[train_n:train_n + val_n], order[train_n + val_n:])
    return tuple(torch.where(torch.isin(groups, part))[0] for part in partitions)


def _profile_sample_ids(base, manifest):
    trace_dir = _resolve_profile_subdir(base, manifest.get('trace_chunk_dir', ''), 'trace_chunks')
    files = _sorted_chunk_files(str(trace_dir / 'decode_trace_chunk_*.pt'))
    if not files:
        raise ValueError('request-disjoint training needs decode traces with sample_ids')
    rows = []
    for filename in files:
        trace = torch.load(filename, map_location='cpu', weights_only=True)
        ids = trace.get('sample_ids')
        if ids is None or ids.ndim != 1 or len(ids) != len(trace['layer_indices']):
            raise ValueError(f'{filename}: missing or misaligned sample_ids; legacy profiles require explicit --split-unit row')
        rows.append(ids.long())
    ids = torch.cat(rows)
    if len(ids) != int(manifest['total_decode_records']):
        raise ValueError('trace record count differs from manifest')
    return ids


def _split_dataset(x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, seed: int,
                   groups: Optional[torch.Tensor] = None) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    n = int(x.shape[0])
    train_idx, val_idx, test_idx = _split_indices(n, seed, groups)
    return (
        TensorDataset(x[train_idx], y[train_idx], idx[train_idx]),
        TensorDataset(x[val_idx], y[val_idx], idx[val_idx]),
        TensorDataset(x[test_idx], y[test_idx], idx[test_idx]),
    )


def _make_loader(dataset: TensorDataset, batch_size: int, rank: int, world: int, shuffle: bool) -> DataLoader:
    sampler = None
    if world > 1:
        # Training pads to equal backward counts. Evaluation uses the raw model
        # and may have unequal counts, so do not duplicate validation records.
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if shuffle else range(rank, len(dataset), world)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def _multi_positive_ce(pred_probs: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    return -(target_dist * pred_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()


def _model_probs_logits(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_model = model.module if isinstance(model, DDP) else model
    if hasattr(raw_model, "forward_logits"):
        # Call through DDP.forward so its reducer prepares gradient reduction.
        logits = model(x, return_logits=True)
        return torch.softmax(logits, dim=-1), logits
    probs = model(x)
    return probs, torch.log(probs.clamp_min(1e-8))


def _ranking_margin_loss(logits: torch.Tensor, true_idx: torch.Tensor, margin: float, neg_k: int) -> torch.Tensor:
    if float(margin) <= 0.0:
        return logits.new_zeros(())
    bsz, num_experts = logits.shape
    pos_mask = torch.zeros((bsz, num_experts), device=logits.device, dtype=torch.bool)
    pos_mask.scatter_(1, true_idx.to(device=logits.device, dtype=torch.long), True)
    weakest_pos = logits.masked_fill(~pos_mask, float("inf")).amin(dim=1)
    neg_logits = logits.masked_fill(pos_mask, float("-inf"))
    k = max(1, min(int(neg_k), max(1, int(num_experts - true_idx.shape[1]))))
    hard_neg = torch.topk(neg_logits, k=k, dim=-1).values
    return F.relu(float(margin) + hard_neg - weakest_pos.unsqueeze(1)).mean()


def _rs_loss(
    probs: torch.Tensor,
    logits: torch.Tensor,
    target_dist: torch.Tensor,
    true_idx: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    ce = _multi_positive_ce(probs, target_dist)
    mode = str(args.loss_mode)
    if mode == "ce":
        loss = ce
    else:
        # For dense distillation targets, every expert has non-zero softmax
        # probability, so (target_dist > 0) would incorrectly mark all experts
        # as positive. Runtime prefetch cares about the router top-k set, so the
        # binary/ranking auxiliary losses must use the true top-k indices.
        binary_target = torch.zeros_like(logits)
        binary_target.scatter_(1, true_idx.to(device=logits.device, dtype=torch.long), 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, binary_target)
        if mode == "bce":
            loss = bce
        elif mode == "ce_bce":
            loss = ce + float(args.bce_weight) * bce
        else:
            raise ValueError(f"Unknown loss_mode={mode}")
    if float(args.rank_weight) > 0.0:
        loss = loss + float(args.rank_weight) * _ranking_margin_loss(
            logits,
            true_idx,
            margin=float(args.rank_margin),
            neg_k=int(args.rank_neg_k),
        )
    return loss


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, ks: List[int]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct_by_k = {k: 0.0 for k in ks}
    pred_by_k = {k: 0.0 for k in ks}
    true_by_k = {k: 0.0 for k in ks}
    prefix_correct_by_k = {k: 0.0 for k in ks}
    prefix_total_by_k = {k: 0.0 for k in ks}
    exact4 = 0.0
    for xb, yb, ib in loader:
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
        ib = ib.to(device=device, dtype=torch.long, non_blocking=True)
        probs, _ = _model_probs_logits(model, xb)
        loss = _multi_positive_ce(probs, yb)
        bs = int(xb.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs

        true_set = ib
        true_k = int(true_set.shape[1])
        for k in ks:
            pred = torch.topk(probs, k=min(k, probs.shape[1]), dim=-1).indices
            match = (pred.unsqueeze(2) == true_set.unsqueeze(1)).any(dim=2)
            num_correct = match.sum(dim=1).float()
            correct_by_k[k] += float(num_correct.sum().item())
            pred_by_k[k] += float(bs * min(k, probs.shape[1]))
            true_by_k[k] += float(bs * true_k)
            prefix_k = min(int(k), int(true_k), int(probs.shape[1]))
            if prefix_k > 0:
                pred_prefix = pred[:, :prefix_k]
                true_prefix = true_set[:, :prefix_k]
                prefix_match = (pred_prefix.unsqueeze(2) == true_prefix.unsqueeze(1)).any(dim=2)
                prefix_correct_by_k[k] += float(prefix_match.sum().item())
                prefix_total_by_k[k] += float(bs * prefix_k)
        pred4 = torch.topk(probs, k=min(true_k, probs.shape[1]), dim=-1).indices
        match4 = (pred4.unsqueeze(2) == true_set.unsqueeze(1)).any(dim=2).sum(dim=1)
        exact4 += float((match4 == true_k).sum().item())

    out: Dict[str, float] = {"loss": total_loss, "n": float(total_n), "exact_topk_match": exact4}
    return out | {
        f"correct@{k}": correct_by_k[k] for k in ks
    } | {
        f"pred@{k}": pred_by_k[k] for k in ks
    } | {
        f"true@{k}": true_by_k[k] for k in ks
    } | {
        f"prefix_correct@{k}": prefix_correct_by_k[k] for k in ks
    } | {
        f"prefix_total@{k}": prefix_total_by_k[k] for k in ks
    }


def _reduce_eval(raw: Dict[str, float], device: torch.device, ks: List[int]) -> Dict[str, float]:
    n = _all_reduce_sum_float(raw["n"], device)
    loss_sum = _all_reduce_sum_float(raw["loss"], device)
    exact = _all_reduce_sum_float(raw["exact_topk_match"], device)
    out = {
        "loss": loss_sum / max(n, 1.0),
        "samples": n,
        "exact_topk_match": exact / max(n, 1.0),
    }
    for k in ks:
        c = _all_reduce_sum_float(raw[f"correct@{k}"], device)
        p = _all_reduce_sum_float(raw[f"pred@{k}"], device)
        t = _all_reduce_sum_float(raw[f"true@{k}"], device)
        out[f"precision@{k}"] = c / max(p, 1.0)
        out[f"recall@{k}"] = c / max(t, 1.0)
        pc = _all_reduce_sum_float(raw[f"prefix_correct@{k}"], device)
        pt = _all_reduce_sum_float(raw[f"prefix_total@{k}"], device)
        out[f"prefix_overlap@{k}"] = pc / max(pt, 1.0)
        out[f"old_recall@{k}"] = out[f"prefix_overlap@{k}"]
    return out


def _train_one_pair(
    source_layer: int,
    target_layer: int,
    rs_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    pair_seed = int(args.seed) + source_layer * 1009 + target_layer
    torch.manual_seed(pair_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(pair_seed)
    X, Y, I = _load_pair_dataset(
        rs_dir,
        source_layer,
        target_layer,
        rank,
        world,
        fp32_inputs=bool(args.cpu_fp32),
        use_router_weights=bool(args.use_router_weights),
        target_mode=str(args.target_mode),
        distill_temperature=float(args.distill_temperature),
        feature_mode=str(args.feature_mode),
    )
    has_data = int(X.shape[0]) > 0
    ok = _all_reduce_sum_int(1 if has_data else 0, device) == world
    if not ok:
        if rank == 0:
            print(f"[Skip] L{source_layer}->L{target_layer}: some rank has no data", flush=True)
        return None

    # Use one request split for every layer. Pair-specific seeds initialize
    # networks, not the train/validation/test partition.
    train_set, val_set, test_set = _split_dataset(X, Y, I, seed=int(args.seed), groups=args.sample_ids)
    train_loader = _make_loader(train_set, int(args.batch_size), rank, world, shuffle=True)
    val_loader = _make_loader(val_set, int(args.batch_size), rank, world, shuffle=False)
    test_loader = _make_loader(test_set, int(args.batch_size), rank, world, shuffle=False)

    input_dim = int(X.shape[1])
    num_experts = int(Y.shape[1])
    if str(args.model_arch) == "residual":
        model = RoutingScoutResidualMLP(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=int(args.hidden_dim),
            num_hidden_layers=int(args.num_hidden_layers),
            dropout=float(args.dropout),
        ).to(device)
    else:
        model = RoutingScoutMLP(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
        ).to(device)
    train_model: nn.Module = DDP(model, device_ids=[device.index]) if world > 1 and device.type == "cuda" else model
    opt = torch.optim.AdamW(train_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    ks = [int(x) for x in str(args.eval_ks).split(",") if str(x).strip()]
    best_val = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    patience_left = int(args.early_stop_patience)

    if rank == 0:
        print(
            f"[Train] L{source_layer}->L{target_layer} n/rank={len(X)} "
            f"H={input_dim} E={num_experts} arch={args.model_arch} hidden={args.hidden_dim} "
            f"layers={args.num_hidden_layers} loss={args.loss_mode}",
            flush=True,
        )

    for epoch in range(int(args.epochs)):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb, ib in train_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            ib = ib.to(device=device, dtype=torch.long, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            probs, logits = _model_probs_logits(train_model, xb)
            loss = _rs_loss(probs, logits, yb, ib, args)
            loss.backward()
            opt.step()
            bs = int(xb.shape[0])
            total_loss += float(loss.item()) * bs
            total_n += bs

        raw_val = _evaluate(model, val_loader, device, ks)
        val = _reduce_eval(raw_val, device, ks)
        train_loss = _all_reduce_sum_float(total_loss, device) / max(_all_reduce_sum_float(total_n, device), 1.0)
        if rank == 0:
            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.5f} val_loss={val['loss']:.5f} "
                + " ".join(f"R@{k}={val[f'recall@{k}']:.4f}" for k in ks),
                flush=True,
            )
        if val["loss"] < best_val:
            best_val = float(val["loss"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(args.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    raw_test = _evaluate(model, test_loader, device, ks)
    test = _reduce_eval(raw_test, device, ks)

    ckpt_path = output_dir / f"rs_l{target_layer}_inL{source_layer}.pth"
    if rank == 0:
        payload = {
            "model_type": "next_layer_residual_logits_rs" if str(args.model_arch) == "residual" else "independent_light_logits_rs",
            "task": "router_prob",
            "state_dict": model.state_dict(),
            "input_dim": int(input_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_hidden_layers": int(args.num_hidden_layers),
            "num_experts": int(num_experts),
            "source_layer": int(source_layer),
            "target_layer": int(target_layer),
            "lead_layers": int(args.lead_layers),
            "top_k": int(I.shape[1]),
            "label_type": "next_layer_true_router_topk_multihot",
            "feature_mode": str(args.feature_mode),
            "target_distribution": str(args.target_mode),
            "loss_mode": str(args.loss_mode),
            "bce_weight": float(args.bce_weight),
            "rank_weight": float(args.rank_weight),
            "rank_margin": float(args.rank_margin),
            "rank_neg_k": int(args.rank_neg_k),
            "distill_temperature": float(args.distill_temperature),
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
            "metrics": test,
            "split_sha256": args.split_sha256,
            "profile_manifest_sha256": args.profile_manifest_sha256,
            "profile_provenance": args.profile_provenance,
        }
        torch.save(payload, ckpt_path)

    row: Dict[str, object] = {
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "lead_layers": int(args.lead_layers),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "ckpt": str(ckpt_path),
    }
    row.update({k: float(v) for k, v in test.items() if isinstance(v, (int, float))})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Train pure next-layer RoutingScout predictors from decode traces.")
    ap.add_argument("--profile-base-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lead-layers", type=int, default=1)
    ap.add_argument("--model-arch", type=str, default="mlp", choices=["mlp", "residual"])
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-hidden-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--eval-ks", type=str, default="1,2,4,8,12")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-unit", choices=["request", "row"], default="request",
                    help="Keep all decode steps of a request together; row is a legacy diagnostic only.")
    ap.add_argument("--cpu-fp32", action="store_true")
    ap.add_argument(
        "--feature-mode",
        type=str,
        default="hidden",
        choices=["hidden", "hidden_router_topk"],
        help=(
            "Input feature. hidden uses only gate input hidden. hidden_router_topk appends "
            "the source layer's true router top-k distribution from the existing profile; "
            "runtime can reproduce this because the source router is already computed."
        ),
    )
    ap.add_argument("--use-router-weights", action="store_true", help="Use router top-k weights instead of uniform multi-hot target.")
    ap.add_argument(
        "--target-mode",
        type=str,
        default="legacy",
        choices=["legacy", "uniform_multihot", "router_weights", "logits"],
        help=(
            "Training target. legacy preserves --use-router-weights behavior; "
            "logits requires profile chunks generated with --save-router-logits."
        ),
    )
    ap.add_argument("--distill-temperature", type=float, default=1.0)
    ap.add_argument("--loss-mode", type=str, default="ce", choices=["ce", "bce", "ce_bce"])
    ap.add_argument("--bce-weight", type=float, default=0.25)
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--rank-margin", type=float, default=0.5)
    ap.add_argument("--rank-neg-k", type=int, default=8)
    ap.add_argument("--start-source-layer", type=int, default=0)
    ap.add_argument("--end-source-layer", type=int, default=-1, help="Exclusive; default num_layers-lead.")
    args = ap.parse_args()

    rank, world, _, device = _setup_dist()
    output_dir = Path(args.output_dir)
    if list(output_dir.glob('rs_l*_inL*.pth')):
        raise ValueError('checkpoint output already contains weights; use a fresh versioned directory')
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(args.profile_base_dir)
    args.sample_ids = _profile_sample_ids(args.profile_base_dir, manifest) if args.split_unit == 'request' else None
    split_rows = _split_indices(int(manifest['total_decode_records']), args.seed, args.sample_ids)
    split_data = dict(unit=args.split_unit, seed=args.seed,
                      rows={name: values.tolist() for name, values in zip(('train','validation','test'), split_rows)})
    if args.sample_ids is not None:
        split_data['request_ids'] = {name: torch.unique(args.sample_ids[values]).tolist()
                                    for name, values in zip(('train','validation','test'), split_rows)}
    args.split_sha256 = hashlib.sha256(json.dumps(split_data, sort_keys=True).encode()).hexdigest()
    args.profile_manifest_sha256 = hashlib.sha256((Path(args.profile_base_dir)/'manifest.json').read_bytes()).hexdigest()
    args.profile_provenance = manifest.get('provenance', {})
    rs_dir = _resolve_profile_subdir(args.profile_base_dir, manifest.get("rs_chunk_dir", ""), "rs_decode_chunks")
    num_layers = int(manifest["num_layers"])
    lead = int(args.lead_layers)
    end_src = int(args.end_source_layer) if int(args.end_source_layer) >= 0 else num_layers - lead
    moe_layers = set(int(l) for l in manifest.get("moe_layer_indices", range(num_layers)))
    source_layers = [
        l for l in range(int(args.start_source_layer), int(end_src))
        if l in moe_layers and l + lead in moe_layers
    ]
    if not source_layers:
        raise ValueError("No adjacent MoE-layer pairs in the requested range")

    if rank == 0:
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
            config = {key:value for key,value in vars(args).items() if key != 'sample_ids'}
            json.dump(config | {"manifest": manifest}, f, indent=2, ensure_ascii=False)
        (output_dir/'split.json').write_text(json.dumps(split_data, indent=2))

    rows: List[Dict[str, object]] = []
    for source_layer in source_layers:
        target_layer = int(source_layer + lead)
        if target_layer >= num_layers:
            continue
        row = _train_one_pair(source_layer, target_layer, rs_dir, output_dir, args, rank, world, device)
        _barrier()
        if rank == 0 and row is not None:
            rows.append(row)
            with open(output_dir / "next_layer_rs_metrics.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            csv_path = output_dir / "next_layer_rs_metrics.csv"
            fieldnames = sorted({k for r in rows for k in r.keys()})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

    _cleanup_dist()


if __name__ == "__main__":
    main()
