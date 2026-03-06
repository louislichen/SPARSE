import sys
import os
import glob
import csv
import gc
import re
import argparse
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class LayerIndependentMLP(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)


class RelativeWeightedMSE(nn.Module):
    def __init__(self, tail_epsilon: float = 0.01):
        super().__init__()
        self.tail_epsilon = tail_epsilon

    def forward(self, pred_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        pred_probs = F.softmax(pred_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        max_values, _ = target_probs.max(dim=1, keepdim=True)
        loss_weights = target_probs / (max_values + 1e-8)
        weights = loss_weights + self.tail_epsilon
        squared_diff = (pred_probs - target_probs) ** 2
        return (weights.detach() * squared_diff).sum(dim=1).mean()


def compute_recall_at_ks(pred_logits: torch.Tensor, true_indices: torch.Tensor, ks: List[int]) -> Dict[int, float]:
    kmax = max(ks)
    _, pred_topk = torch.topk(pred_logits, kmax, dim=-1)
    out = {}
    for k in ks:
        true_topk = true_indices[:, :k]
        pred_k = pred_topk[:, :k]
        matches = (pred_k.unsqueeze(2) == true_topk.unsqueeze(1))
        num_correct = matches.any(dim=2).sum(dim=1).float()
        out[k] = torch.mean(num_correct / k).item()
    return out


def compute_regression_metrics(pred_logits: torch.Tensor, true_logits: torch.Tensor) -> Tuple[float, float]:
    mse = nn.functional.mse_loss(pred_logits, true_logits).item()
    mae = nn.functional.l1_loss(pred_logits, true_logits).item()
    return mse, mae


def release_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get_sorted_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    if not files:
        return []

    def extract_step(fn: str) -> int:
        m = re.search(r"_chunk_(\d+)\.pt$", fn)
        return int(m.group(1)) if m else -1

    return sorted(files, key=extract_step)


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()


def all_reduce_sum(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def all_reduce_mean(value: float, device: torch.device) -> float:
    return all_reduce_sum(value, device) / ddp_world_size()


def ddp_all_true(flag: bool, device: torch.device) -> bool:
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def shard_files_nonempty(files, rank, world_size):
    n = len(files)
    if n == 0:
        return []
    if n >= world_size:
        return [f for i, f in enumerate(files) if (i % world_size) == rank]
    # n < world_size：复制分配，避免 rank2/3 没数据
    return [files[rank % n]]


def load_source_data_chunks_sharded(data_dir: str, layer_idx: int, rank: int, world_size: int, fp32: bool) -> Optional[torch.Tensor]:
    pattern = os.path.join(data_dir, f"rs_inputs_l{layer_idx}_chunk_*.pt")
    files = get_sorted_files(pattern)
    if not files:
        return None
    files = shard_files_nonempty(files, rank, world_size)
    if not files:
        return None

    tensors = []
    for f in files:
        t = torch.load(f, map_location="cpu")  # tensor
        # 尽量保持 fp16 降低 CPU 内存
        tensors.append(t.float() if fp32 else t.half())
    return torch.cat(tensors, dim=0)


def load_target_data_chunks_sharded(data_dir: str, layer_idx: int, rank: int, world_size: int, fp32: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pattern = os.path.join(data_dir, f"rs_labels_l{layer_idx}_chunk_*.pt")
    files = get_sorted_files(pattern)
    if not files:
        return None, None
    files = shard_files_nonempty(files, rank, world_size)
    if not files:
        return None, None

    logits_list, indices_list = [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        logits = d["logits"]
        idx = d["indices"]
        logits_list.append(logits.float() if fp32 else logits.half())
        indices_list.append(idx.long())
    return torch.cat(logits_list, dim=0), torch.cat(indices_list, dim=0)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def train_single_layer(layer_idx: int, X_source: torch.Tensor, source_layer: int, device: torch.device, args):
    rank = ddp_rank()
    world_size = ddp_world_size()
    is_main = (rank == 0)

    # 读取 labels（分片）
    Y_logits, Y_indices = load_target_data_chunks_sharded(args.data_dir, layer_idx, rank, world_size, fp32=args.cpu_fp32)
    has_y = (Y_logits is not None) and (Y_indices is not None)
    ok = ddp_all_true(has_y, device)
    if not ok:
        if is_main:
            print(f"[Skip-All] layer {layer_idx}: some rank failed to load labels")
        ddp_barrier()
        return None, None

    min_len = min(len(X_source), len(Y_logits))
    X_curr = X_source[:min_len]
    Y_logits = Y_logits[:min_len]
    Y_indices = Y_indices[:min_len]

    input_dim = X_curr.shape[1]
    num_experts = Y_logits.shape[1]

    if is_main:
        print(f"\n========== Train layer {layer_idx} (input L{source_layer}) ==========")
        print(f"Per-rank tokens: {min_len} | H={input_dim} | E={num_experts}")

    dataset = TensorDataset(X_curr, Y_logits, Y_indices)
    train_sz = int(0.7 * len(dataset))
    val_sz = int(0.1 * len(dataset))
    test_sz = len(dataset) - train_sz - val_sz

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_sz, val_sz, test_sz], generator=g)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.per_gpu_batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.per_gpu_batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

    model = LayerIndependentMLP(input_dim, num_experts, hidden_dim=1024).to(device)
    model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_patience = max(1, args.early_stop_patience // 2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience)
    criterion = RelativeWeightedMSE(tail_epsilon=args.tail_epsilon).to(device)

    save_path = os.path.join(args.output_dir, f"rs_l{layer_idx}_inL{source_layer}.pth")

    best_val = float("inf")
    early_stop_cnt = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        tr_sum, tr_n = 0.0, 0
        for bx, by_logits, _ in train_loader:
            bx = bx.to(device, non_blocking=True).float()  # 输入保持 fp32（更稳）
            by_logits = by_logits.to(device, non_blocking=True).float()

            # 统一用 fp32 做 loss（更稳）
            pred = model(bx)
            tgt = by_logits

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()

            tr_sum += loss.item()
            tr_n += 1

        local_train = tr_sum / max(1, tr_n)
        global_train = all_reduce_mean(local_train, device)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for bx, by_logits, _ in val_loader:
                bx = bx.to(device, non_blocking=True).float()
                by_logits = by_logits.to(device, non_blocking=True).float()
                pred = model(bx)
                tgt = by_logits
                loss = criterion(pred, tgt)
                va_sum += loss.item()
                va_n += 1

        local_val = va_sum / max(1, va_n)
        global_val = all_reduce_mean(local_val, device)
        scheduler.step(global_val)

        if global_val < best_val:
            best_val = global_val
            early_stop_cnt = 0
            if is_main:
                torch.save(model.module.state_dict(), save_path)
        else:
            early_stop_cnt += 1

        history["train_loss"].append(global_train)
        history["val_loss"].append(global_val)

        if is_main and (epoch + 1) % 5 == 0:
            print(f"Ep{epoch+1}: Train={global_train:.4f} Val={global_val:.4f} LR={optimizer.param_groups[0]['lr']:.2e}")

        stop_flag = 1 if early_stop_cnt >= args.early_stop_patience else 0
        stop_flag = int(all_reduce_sum(stop_flag, device) > 0)
        if stop_flag:
            if is_main:
                print(f"Early stop at ep{epoch+1}, best_val={best_val:.4f}")
            break

    ddp_barrier()

    # load best and test
    state = torch.load(save_path, map_location="cpu")
    model.module.load_state_dict(state)
    model.eval()

    recall_acc = {k: 0.0 for k in args.top_k_list}
    mse_sum, mae_sum, batches = 0.0, 0.0, 0
    with torch.no_grad():
        for bx, by_logits, by_indices in test_loader:
            bx = bx.to(device, non_blocking=True).float()
            by_logits = by_logits.to(device, non_blocking=True).float()
            by_indices = by_indices.to(device, non_blocking=True)

            pred = model(bx)
            tgt = by_logits
            
            r = compute_recall_at_ks(pred, by_indices, args.top_k_list)
            for k in args.top_k_list:
                recall_acc[k] += r[k]
            mse, mae = compute_regression_metrics(pred, tgt)
            mse_sum += mse
            mae_sum += mae
            batches += 1

    gbatches = all_reduce_sum(float(batches), device)
    final_recall = {k: all_reduce_sum(recall_acc[k], device) / max(1.0, gbatches) for k in args.top_k_list}
    final_mse = all_reduce_sum(mse_sum, device) / max(1.0, gbatches)
    final_mae = all_reduce_sum(mae_sum, device) / max(1.0, gbatches)

    metrics = {
        "layer": layer_idx,
        "source_layer": source_layer,
        "recall_2": final_recall.get(2, float("nan")),
        "recall_3": final_recall.get(3, float("nan")),
        "recall_4": final_recall.get(4, float("nan")),
        "mse": final_mse,
        "mae": final_mae,
        "val_loss": best_val,
        "ckpt": save_path,
    }
    if is_main:
        print(f"Test: R2={metrics['recall_2']:.4f} R3={metrics['recall_3']:.4f} R4={metrics['recall_4']:.4f} MSE={final_mse:.6f}")
    return history, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/profile_data/Qwen1.5MoE_allData/rs_chunks")
    ap.add_argument("--output-dir", default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/mlp_trainer/Qwen1.5MoE_allData")
    ap.add_argument("--per-gpu-batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=0)  # 先用0排障
    ap.add_argument("--pin-memory", action="store_true", default=False)
    ap.add_argument("--top-k-list", type=str, default="2,3,4")
    ap.add_argument("--tail-epsilon", type=float, default=0.01)
    ap.add_argument("--cpu-fp32", action="store_true", default=False,
                    help="force load inputs/logits as fp32 on CPU (default fp16 to save RAM)")
    ap.add_argument("--start-layer", type=int, default=0, help="resume training from this layer index (inclusive)")
    args = ap.parse_args()
    args.top_k_list = [int(k) for k in args.top_k_list.split(",")]

    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
        csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Layer", "SourceLayer", "Recall@2", "Recall@3", "Recall@4", "MSE", "MAE", "Best_Val_Loss", "CkptPath"])

    ddp_barrier()

    all_label_files = glob.glob(os.path.join(args.data_dir, "rs_labels_l*_chunk_*.pt"))
    layers = sorted({int(re.search(r"rs_labels_l(\d+)_chunk", os.path.basename(f)).group(1)) for f in all_label_files})
    groups: Dict[int, List[int]] = {}
    for l in layers:
        s = (l // args.group_size) * args.group_size
        groups.setdefault(s, []).append(l)
    group_keys = sorted(groups.keys())
    if is_main:
        print(f"[DDP] world_size={world_size} local_rank={local_rank} device={device}")
        print(f"Detected {len(layers)} layers: {layers}")

    input_cache: Dict[int, torch.Tensor] = {}
    all_histories, all_metrics = {}, {}

    start_layer = getattr(args, "start_layer", 0)

    for source_layer in group_keys:
        # 如果整个组都在 start_layer 前，可以跳过
        if max(groups[source_layer]) < start_layer:
            continue
        release_gpu_memory()
        X_source = load_source_data_chunks_sharded(args.data_dir, source_layer, rank, world_size, fp32=args.cpu_fp32)
        has_x = X_source is not None
        ok = ddp_all_true(has_x, device)
        if not ok:
            if is_main:
                print(f"[Skip-All] source_layer {source_layer}: some rank missing inputs")
            ddp_barrier()
            continue

        if is_main:
            print(f"[ShardCheck] source_layer {source_layer}: rank0 tokens={len(X_source)}")

        for layer_idx in groups[source_layer]:
            if layer_idx < start_layer:
                continue

            if is_main:
                print(f"[Map] layer {layer_idx} <- inputs from layer {source_layer}")

            hist, metrics = train_single_layer(layer_idx, X_source, source_layer, device, args)
            ddp_barrier()

        if metrics is None:
            continue

        if is_main:
            all_histories[layer_idx] = hist
            all_metrics[layer_idx] = metrics
            csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    metrics["layer"], metrics["source_layer"],
                    f"{metrics['recall_2']:.4f}", f"{metrics['recall_3']:.4f}", f"{metrics['recall_4']:.4f}",
                    f"{metrics['mse']:.6f}", f"{metrics['mae']:.6f}", f"{metrics['val_loss']:.4f}",
                    metrics["ckpt"]
                ])

    del X_source
    gc.collect()
    release_gpu_memory()
    ddp_barrier()

    if is_main and all_histories:
        plt.figure(figsize=(15, 12))
        plot_layers = sorted(all_histories.keys())
        plot_indices = [l for l in plot_layers if (l % 4 == 0) or (l == plot_layers[0]) or (l == plot_layers[-1])]

        plt.subplot(2, 1, 1)
        for l in plot_indices:
            plt.plot(all_histories[l]["val_loss"], label=f"L{l} Val Loss")
        plt.title("Validation Loss Curve (Selected Layers)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        recall_2 = [all_metrics[l]["recall_2"] for l in plot_indices]
        recall_3 = [all_metrics[l]["recall_3"] for l in plot_indices]
        recall_4 = [all_metrics[l]["recall_4"] for l in plot_indices]

        x = np.arange(len(plot_indices))
        width = 0.25
        plt.bar(x - width, recall_2, width, label="Recall@2")
        plt.bar(x, recall_3, width, label="Recall@3")
        plt.bar(x + width, recall_4, width, label="Recall@4")
        plt.title("Recall Metrics (Selected Layers)")
        plt.xlabel("Layer Index")
        plt.ylabel("Recall Score")
        plt.xticks(x, plot_indices)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = os.path.join(args.output_dir, "plots", "training_summary.png")
        plt.savefig(out_png)
        print(f"Done. CSV: {os.path.join(args.output_dir, 'metrics_summary.csv')}, Plot: {out_png}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()