import os
import csv

import gc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split


DATA_DIR = "/home/ztl/wyf/IC_Exp/pg_moe_experiments_full_data"
OUTPUT_DIR = "/home/ztl/cl/LLM_Partition/results/mlp_trainer"
SOURCE_LAYER = 0
TARGET_LAYERS = list(range(5, 10))

PER_GPU_BATCH_SIZE = 512
GRADIENT_ACCUM_STEPS = 1    
USE_MIXED_PRECISION = False  
EPOCHS = 100                  
LEARNING_RATE = 1e-3         
WEIGHT_DECAY = 1e-4
NUM_EXPERTS = 60            
TOP_K_LIST = [2, 3, 4]

NUM_WORKERS = 0    
PIN_MEMORY = False 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


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
    

class DistillationLoss(nn.Module):
    def __init__(self, temp=1.0, alpha=0.8):
        super().__init__()
        self.temp = temp        
        self.alpha = alpha      
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, pred_logits, true_logits):
        pred_log_prob = nn.functional.log_softmax(pred_logits / self.temp, dim=-1)
        true_prob = nn.functional.softmax(true_logits / self.temp, dim=-1)
        
        loss_kl = self.kl_div(pred_log_prob, true_prob) * (self.temp ** 2)
        
        loss_mse = self.mse(pred_logits, true_logits)
        
        return self.alpha * loss_kl + (1 - self.alpha) * loss_mse


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
    

def compute_recall_at_ks(pred_logits, true_indices, ks=TOP_K_LIST):
    _, pred_top4_indices = torch.topk(pred_logits, 4, dim=-1)  # [batch, 4]
    
    recall_results = {}
    for k in ks:
        true_topk_indices = true_indices[:, :k] # [batch, k]
        
        pred_expanded = pred_top4_indices.unsqueeze(2) # [batch,4,1]
        true_expanded = true_topk_indices.unsqueeze(1) # [batch,1,k]
        matches = (pred_expanded == true_expanded) # [batch,4,k]
        
        num_correct = matches.any(dim=2).sum(dim=1).float() # [batch]
        avg_recall = torch.mean(num_correct / k).item()
        recall_results[k] = avg_recall
    
    return recall_results


def compute_mass_coverage_at_ks(pred_logits, true_logits, ks=TOP_K_LIST):
    true_logits_sum = true_logits.sum(dim=-1, keepdim=True) # [batch, 1]
    true_logits_sum = torch.clamp(true_logits_sum, min=1e-8)
    
    mass_results = {"ideal": {}, "pred": {}}
    for k in ks:
        _, true_topk_indices = torch.topk(true_logits, k, dim=-1) # [batch, k]
        _, pred_topk_indices = torch.topk(pred_logits, k, dim=-1) # [batch, k]
        
        true_topk_values = torch.gather(true_logits, 1, true_topk_indices)  # [batch, k]
        pred_topk_values = torch.gather(true_logits, 1, pred_topk_indices)  # [batch, k]
        
        ideal_mass = (true_topk_values.sum(dim=-1) / true_logits_sum.squeeze(-1)).mean().item()
        pred_mass = (pred_topk_values.sum(dim=-1) / true_logits_sum.squeeze(-1)).mean().item()
        
        mass_results["ideal"][k] = ideal_mass
        mass_results["pred"][k] = pred_mass
    
    return mass_results


def compute_regression_metrics(pred_logits, true_logits):
    mse = nn.functional.mse_loss(pred_logits, true_logits).item()
    mae = nn.functional.l1_loss(pred_logits, true_logits).item()
    return mse, mae


def release_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def load_source_data():
    print(f"Loading Source Layer {SOURCE_LAYER} (Input Features)...")
    s_path = os.path.join(DATA_DIR, f"layer_{SOURCE_LAYER}_data.pt")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Data not found: {s_path}")
    
    s_data = torch.load(s_path, map_location="cpu")
    X = s_data['router_inputs'].float()

    return X


def load_target_layer_data(layer_idx, num_samples):
    t_path = os.path.join(DATA_DIR, f"layer_{layer_idx}_data.pt")
    if not os.path.exists(t_path):
        return None, None
    
    t_data = torch.load(t_path, map_location="cpu")
    if t_data['router_logits'].shape[0] != num_samples:
        print(f"Layer {layer_idx} mismatches, skip...")
        return None, None
        
    return t_data['router_logits'].float(), t_data['router_indices'].long()


def train_single_layer(layer_idx, X, device, num_gpus):
    print(f"{'='*10} Processing Layer {layer_idx} {'='*10}")
    
    Y_logits, Y_indices = load_target_layer_data(layer_idx, len(X))
    if Y_logits is None:
        print(f"Layer {layer_idx} invalid, skip...")
        return None

    input_dim = X.shape[1] 
    print(f"Model Input Dim: {input_dim}, Output Dim: {NUM_EXPERTS}")

    dataset = TensorDataset(X, Y_logits, Y_indices)
    train_sz = int(0.7 * len(X))
    val_sz = int(0.1 * len(X))
    test_sz = len(X) - train_sz - val_sz
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_sz, val_sz, test_sz], 
        generator=torch.Generator().manual_seed(42)
    )

    total_batch_size = PER_GPU_BATCH_SIZE * num_gpus * GRADIENT_ACCUM_STEPS
    
    train_loader = DataLoader(train_ds, batch_size=total_batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=total_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=total_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = LayerIndependentMLP(input_dim, NUM_EXPERTS, hidden_dim=1024).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = RelativeWeightedMSE(tail_epsilon=0.01).to(device)
    # criterion = DistillationLoss(temp=2.0, alpha=0.7).to(device)

    best_val_loss = float('inf')
    early_stop_cnt = 0
    history = {"train_loss": [], "val_loss": [], "mse": [], "recall": []}
    
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
            save_path = os.path.join(OUTPUT_DIR, f"best_model_layer_{layer_idx}.pth")
            if num_gpus > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= 5:
                print(f"Layer {layer_idx} Early stopping at epoch {epoch}")
                break
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1}: T_Loss={avg_train_loss:.4f}, V_Loss={avg_val_loss:.4f}")

    print(f"Testing Layer {layer_idx}...")
    save_path = os.path.join(OUTPUT_DIR, f"best_model_layer_{layer_idx}.pth")
    if num_gpus > 1:
        model.module.load_state_dict(torch.load(save_path))
    else:
        model.load_state_dict(torch.load(save_path))
    model.eval()

    recall_accum = {k: 0.0 for k in TOP_K_LIST}
    mass_ideal_accum = {k: 0.0 for k in TOP_K_LIST}
    mass_pred_accum = {k: 0.0 for k in TOP_K_LIST}
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
            
            mass_dict = compute_mass_coverage_at_ks(pred_logits, by_logits, ks=TOP_K_LIST)
            for k in TOP_K_LIST:
                mass_ideal_accum[k] += mass_dict["ideal"][k]
                mass_pred_accum[k] += mass_dict["pred"][k]
            
            mse, mae = compute_regression_metrics(pred_logits, by_logits)
            test_mse += mse
            test_mae += mae
    
    final_recall = {k: recall_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mass_ideal = {k: mass_ideal_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mass_pred = {k: mass_pred_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mse = test_mse / len(test_loader)
    final_mae = test_mae / len(test_loader)

    print(f"Layer {layer_idx} Result:")
    print(f"Recall: Top2={final_recall[2]:.4f}, Top3={final_recall[3]:.4f}, Top4={final_recall[4]:.4f}")
    print(f"Ideal_Mass: Top2={final_mass_ideal[2]:.4f}, Top3={final_mass_ideal[3]:.4f}, Top4={final_mass_ideal[4]:.4f}")
    print(f"Pred_Mass: Top2={final_mass_pred[2]:.4f}, Top3={final_mass_pred[3]:.4f}, Top4={final_mass_pred[4]:.4f}")
    print(f"Regression: MSE={final_mse:.6f}, MAE={final_mae:.6f}")
    
    return history, {
        "layer": layer_idx,
        # Recall@2/3/4
        "recall_2": final_recall[2],
        "recall_3": final_recall[3],
        "recall_4": final_recall[4],
        # Ideal_Mass@2/3/4
        "ideal_mass_2": final_mass_ideal[2],
        "ideal_mass_3": final_mass_ideal[3],
        "ideal_mass_4": final_mass_ideal[4],
        # Pred_Mass@2/3/4
        "pred_mass_2": final_mass_pred[2],
        "pred_mass_3": final_mass_pred[3],
        "pred_mass_4": final_mass_pred[4],

        "mse": final_mse,
        "mae": final_mae,
        "val_loss": best_val_loss
    }

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found {num_gpus} GPUs. Running Serial Training per Layer.")

    X = load_source_data()
    
    csv_path = os.path.join(OUTPUT_DIR, "serial_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Layer", 
            # Recall@2/3/4
            "Recall@2", "Recall@3", "Recall@4",
            # Ideal_Mass@2/3/4
            "Ideal_Mass@2", "Ideal_Mass@3", "Ideal_Mass@4",
            # Pred_Mass@2/3/4
            "Pred_Mass@2", "Pred_Mass@3", "Pred_Mass@4",

            "MSE", "MAE", "Best_Val_Loss"
        ])
    
    all_histories = {}
    all_metrics = {}

    for layer_idx in TARGET_LAYERS:
        release_gpu_memory()
        
        try:
            hist, metrics = train_single_layer(layer_idx, X, device, num_gpus)
            
            if metrics is None: continue
            
            all_histories[layer_idx] = hist
            all_metrics[layer_idx] = metrics
            
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics["layer"],
                    # Recall@2/3/4
                    f"{metrics['recall_2']:.4f}", f"{metrics['recall_3']:.4f}", f"{metrics['recall_4']:.4f}",
                    # Ideal_Mass@2/3/4
                    f"{metrics['ideal_mass_2']:.4f}", f"{metrics['ideal_mass_3']:.4f}", f"{metrics['ideal_mass_4']:.4f}",
                    # Pred_Mass@2/3/4
                    f"{metrics['pred_mass_2']:.4f}", f"{metrics['pred_mass_3']:.4f}", f"{metrics['pred_mass_4']:.4f}",
                    
                    f"{metrics['mse']:.6f}", f"{metrics['mae']:.6f}", f"{metrics['val_loss']:.4f}"
                ])
                
        except Exception as e:
            print(f"Error training layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_histories:
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 1, 1)
        for l, hist in all_histories.items():
            if l % 5 == 0 or l == 22: 
                plt.plot(hist["val_loss"], label=f"L{l} Val Loss")
        plt.title("Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plot_layers = [l for l in all_metrics.keys() if l % 5 == 0 or l == 22]
        
        if plot_layers:
            recall_2 = [all_metrics[l]["recall_2"] for l in plot_layers]
            recall_3 = [all_metrics[l]["recall_3"] for l in plot_layers]
            recall_4 = [all_metrics[l]["recall_4"] for l in plot_layers]
            
            x = np.arange(len(plot_layers))
            width = 0.25
            plt.bar(x - width, recall_2, width, label="Recall@2")
            plt.bar(x, recall_3, width, label="Recall@3")
            plt.bar(x + width, recall_4, width, label="Recall@4")
            
            plt.title("Recall@2/3/4 Comparison (Selected Layers)")
            plt.xlabel("Layer")
            plt.ylabel("Recall")
            plt.xticks(x, plot_layers)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "serial_training_metrics.png"))
        print(f"Training Complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()