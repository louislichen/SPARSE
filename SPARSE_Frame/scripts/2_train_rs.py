import sys
import os
import glob
import csv
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import re
from torch.utils.data import TensorDataset, DataLoader, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

DATA_DIR = "/home/ztl/wyf/SPARSE_MOE/results/profile_data"
OUTPUT_DIR = "/home/ztl/wyf/SPARSE_MOE/results/mlp_trainer"

PER_GPU_BATCH_SIZE = 1024       
GRADIENT_ACCUM_STEPS = 1    
EPOCHS = 100                     
LEARNING_RATE = 1e-3         
WEIGHT_DECAY = 1e-4
TOP_K_LIST = [2, 3, 4]
EARLY_STOP_PATIENCE = 10  

NUM_WORKERS = 4    
PIN_MEMORY = True 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


# ================= 模型定义 =================
class LayerIndependentMLP(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)

class RelativeWeightedMSE(nn.Module):
    def __init__(self, tail_epsilon=0.01):
        super().__init__()
        self.tail_epsilon = tail_epsilon

    def forward(self, pred, target):
        max_values, _ = target.max(dim=1, keepdim=True)
        loss_weights = target / (max_values + 1e-8) 
        weights = loss_weights + self.tail_epsilon
        squared_diff = (pred - target) ** 2
        loss = (weights.detach() * squared_diff).sum(dim=1).mean()
        return loss

# ================= Metrics =================
def compute_recall_at_ks(pred_logits, true_indices, ks=TOP_K_LIST):
    _, pred_top4_indices = torch.topk(pred_logits, 4, dim=-1)
    recall_results = {}
    for k in ks:
        true_topk_indices = true_indices[:, :k]
        pred_expanded = pred_top4_indices.unsqueeze(2)
        true_expanded = true_topk_indices.unsqueeze(1)
        matches = (pred_expanded == true_expanded)
        num_correct = matches.any(dim=2).sum(dim=1).float()
        avg_recall = torch.mean(num_correct / k).item()
        recall_results[k] = avg_recall
    return recall_results

def compute_regression_metrics(pred_logits, true_logits):
    mse = nn.functional.mse_loss(pred_logits, true_logits).item()
    mae = nn.functional.l1_loss(pred_logits, true_logits).item()
    return mse, mae

def release_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# ================= 数据加载辅助函数 =================
def get_sorted_files(pattern):
    """
    获取匹配的文件并按 Chunk ID 排序 (4999 < 9999)
    """
    files = glob.glob(pattern)
    if not files: return []
    
    def extract_step(filename):
        match = re.search(r'_chunk_(\d+)\.pt$', filename)
        return int(match.group(1)) if match else -1
        
    return sorted(files, key=extract_step)

def load_source_data_chunks():
    """加载 Layer 0 的 Input (所有 chunks)"""
    print(f"Loading Source Inputs (Layer 0)...")
    pattern = os.path.join(DATA_DIR, "rs_inputs_l0_chunk_*.pt")
    files = get_sorted_files(pattern)
    
    if not files:
        raise FileNotFoundError(f"No input chunks found in {DATA_DIR}!")
    
    tensors = []
    for f in files:
        try:
            t = torch.load(f, map_location="cpu")
            tensors.append(t.float())
        except Exception as e:
            print(f"Warning: Error loading {f}: {e}")
            
    if not tensors: return None
    return torch.cat(tensors, dim=0)

def load_target_data_chunks(layer_idx):
    """加载指定 Layer 的 Labels (所有 chunks)"""
    pattern = os.path.join(DATA_DIR, f"rs_labels_l{layer_idx}_chunk_*.pt")
    files = get_sorted_files(pattern)
    
    if not files:
        return None, None
    
    logits_list = []
    indices_list = []
    
    for f in files:
        try:
            d = torch.load(f, map_location="cpu")
            logits_list.append(d['logits'].float())
            indices_list.append(d['indices'].long())
        except Exception as e:
            print(f"Warning: Error loading {f}: {e}")
            
    if not logits_list: return None, None
    
    return torch.cat(logits_list, dim=0), torch.cat(indices_list, dim=0)

# ================= 训练流程 =================
def train_single_layer(layer_idx, X, device, num_gpus):
    print(f"\n{'='*10} Processing Layer {layer_idx} {'='*10}")
    
    Y_logits, Y_indices = load_target_data_chunks(layer_idx)
    if Y_logits is None:
        print(f"Layer {layer_idx} data missing, skip...")
        return None

    min_len = min(len(X), len(Y_logits))
    X_curr = X[:min_len]
    Y_logits = Y_logits[:min_len]
    Y_indices = Y_indices[:min_len]

    input_dim = X_curr.shape[1]
    num_experts = Y_logits.shape[1] 
    print(f"Data Loaded: {min_len} samples | Input Dim: {input_dim} | Experts: {num_experts}")

    dataset = TensorDataset(X_curr, Y_logits, Y_indices)
    
    train_sz = int(0.7 * len(dataset))
    val_sz = int(0.1 * len(dataset))
    test_sz = len(dataset) - train_sz - val_sz
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_sz, val_sz, test_sz], 
        generator=torch.Generator().manual_seed(42)
    )

    total_batch_size = PER_GPU_BATCH_SIZE * num_gpus * GRADIENT_ACCUM_STEPS
    
    train_loader = DataLoader(train_ds, batch_size=total_batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=total_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=total_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = LayerIndependentMLP(input_dim, num_experts, hidden_dim=1024).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = RelativeWeightedMSE(tail_epsilon=0.01).to(device)

    best_val_loss = float('inf')
    early_stop_cnt = 0
    history = {"train_loss": [], "val_loss": []}
    
    print(f"Start Training Layer {layer_idx}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by_logits, _ in train_loader:
            bx, by_logits = bx.to(device, non_blocking=True), by_logits.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by_logits)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by_logits, _ in val_loader:
                bx, by_logits = bx.to(device, non_blocking=True), by_logits.to(device, non_blocking=True)
                pred = model(bx)
                loss = criterion(pred, by_logits)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            save_path = os.path.join(OUTPUT_DIR, f"rs_l{layer_idx}.pth")
            if num_gpus > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= EARLY_STOP_PATIENCE:  # 使用配置的 Patience
                print(f"Layer {layer_idx} Early stopping at epoch {epoch}")
                break
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1}: T_Loss={avg_train_loss:.4f}, V_Loss={avg_val_loss:.4f}")

    # Testing
    print(f"Testing Layer {layer_idx}...")
    save_path = os.path.join(OUTPUT_DIR, f"rs_l{layer_idx}.pth")
    if num_gpus > 1:
        model.module.load_state_dict(torch.load(save_path))
    else:
        model.load_state_dict(torch.load(save_path))
    model.eval()

    recall_accum = {k: 0.0 for k in TOP_K_LIST}
    test_mse = 0.0
    test_mae = 0.0
    
    with torch.no_grad():
        for bx, by_logits, by_indices in test_loader:
            bx = bx.to(device, non_blocking=True)
            by_logits = by_logits.to(device, non_blocking=True)
            by_indices = by_indices.to(device, non_blocking=True)

            pred_logits = model(bx)
            recall_dict = compute_recall_at_ks(pred_logits, by_indices, ks=TOP_K_LIST)
            for k in TOP_K_LIST:
                recall_accum[k] += recall_dict[k]
            
            mse, mae = compute_regression_metrics(pred_logits, by_logits)
            test_mse += mse
            test_mae += mae
    
    final_recall = {k: recall_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mse = test_mse / len(test_loader)
    final_mae = test_mae / len(test_loader)

    print(f"Layer {layer_idx} Result: Recall@Top2={final_recall[2]:.4f}")
    
    return history, {
        "layer": layer_idx,
        "recall_2": final_recall[2],
        "recall_3": final_recall[3],
        "recall_4": final_recall[4],
        "mse": final_mse,
        "mae": final_mae,
        "val_loss": best_val_loss
    }

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found {num_gpus} GPUs. Running Serial Training per Layer.")

    # 1. 加载 Inputs
    X = load_source_data_chunks()
    if X is None:
        print("Error: Failed to load source inputs.")
        return

    # 2. 自动检测层
    all_label_files = glob.glob(os.path.join(DATA_DIR, "rs_labels_l*_chunk_*.pt"))
    
    detected_layers = set()
    for f in all_label_files:
        filename = os.path.basename(f)
        match = re.search(r'rs_labels_l(\d+)_chunk', filename)
        if match:
            detected_layers.add(int(match.group(1)))
            
    detected_layers = sorted(list(detected_layers))
    
    if not detected_layers:
        print("CRITICAL: No layer data detected! Check DATA_DIR path and file naming.")
        return
        
    print(f"Detected {len(detected_layers)} layers to train: {detected_layers}")
    
    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Layer", 
            "Recall@2", "Recall@3", "Recall@4",
            "MSE", "MAE", "Best_Val_Loss"
        ])
    
    all_histories = {}
    all_metrics = {}

    for layer_idx in detected_layers:
        release_gpu_memory()
        
        try:
            hist, metrics = train_single_layer(layer_idx, X, device, num_gpus)
            
            if metrics is None: continue
            
            all_histories[layer_idx] = hist
            all_metrics[layer_idx] = metrics
            
            # 实时写入 CSV
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics["layer"],
                    f"{metrics['recall_2']:.4f}", f"{metrics['recall_3']:.4f}", f"{metrics['recall_4']:.4f}",
                    f"{metrics['mse']:.6f}", f"{metrics['mae']:.6f}", f"{metrics['val_loss']:.4f}"
                ])
                
        except Exception as e:
            print(f"Error training layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 绘图
    if all_histories:
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 1, 1)
        plot_indices = [l for l in all_histories.keys() if l % 4 == 0 or l == detected_layers[0] or l == detected_layers[-1]]
        for l in plot_indices:
            plt.plot(all_histories[l]["val_loss"], label=f"L{l} Val Loss")
        plt.title("Validation Loss Curve (Selected Layers)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        if plot_indices:
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
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "training_summary.png"))
        print(f"\n All training complete. Metrics saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()