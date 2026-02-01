import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import csv
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import time

# ================= 4卡1080Ti专属配置 =================
DATA_DIR = "./pg_moe_experiments_full_data"
OUTPUT_DIR = "./pg_moe_results_simple_mlp_kl"
SOURCE_LAYER = 0              # 源层（输入特征来源）
TARGET_LAYERS = list(range(0, 23))  # 目标层

# 1080Ti显存适配
PER_GPU_BATCH_SIZE = 512    # 单卡批次
GRADIENT_ACCUM_STEPS = 1    
USE_MIXED_PRECISION = False  
EPOCHS = 50                  
LEARNING_RATE = 1e-3         
WEIGHT_DECAY = 1e-4
NUM_EXPERTS = 60            
TOP_K_LIST = [2, 3, 4]       # 新增：要评估的Top-K列表（2/3/4）

# 多卡稳定策略
NUM_WORKERS = 0    
PIN_MEMORY = False 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

# ================= 模型定义 (保持不变) =================
class LayerIndependentMLP(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(-1, x.shape[-1])
        return self.mlp(x)

class DistillationLoss(nn.Module):
    def __init__(self, temp=1.0, alpha=0.8):
        super().__init__()
        self.temp = temp        # 温度系数，软化分布
        self.alpha = alpha      # 权重平衡
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss() # 或者 nn.HuberLoss()

    def forward(self, pred_logits, true_logits):
        # 1. KL 散度部分 (学习分布形状/排名)
        pred_log_prob = nn.functional.log_softmax(pred_logits / self.temp, dim=-1)
        true_prob = nn.functional.softmax(true_logits / self.temp, dim=-1)
        
        loss_kl = self.kl_div(pred_log_prob, true_prob) * (self.temp ** 2)
        
        # 2. 回归部分 (学习 Logits 的绝对量级，为了后续阈值判断)
        loss_mse = self.mse(pred_logits, true_logits)
        
        # 3. 组合
        return self.alpha * loss_kl + (1 - self.alpha) * loss_mse

# ================= 指标函数 (核心修改：支持多K值) =================
def compute_recall_at_ks(pred_logits, true_indices, ks=TOP_K_LIST):
    """
    批量计算Recall@2/3/4：预测的Top-4专家中覆盖真实Top-K的比例
    pred_logits: [batch, num_experts] → 模型预测得分
    true_indices: [batch, 4] → 真实的Top-4专家索引（取前2/3位作为Top-2/3）
    ks: 要计算的K值列表（[2,3,4]）
    return: dict → {2: recall@2, 3: recall@3, 4: recall@4}
    """
    batch_size = pred_logits.size(0)
    # 1. 获取预测的Top-4索引（因为要评估预测Top-4覆盖真实Top-2/3/4）
    _, pred_top4_indices = torch.topk(pred_logits, 4, dim=-1)  # [batch, 4]
    
    recall_results = {}
    for k in ks:
        # 2. 取真实的Top-K索引（从true_indices的前k位取）
        true_topk_indices = true_indices[:, :k]  # [batch, k]
        
        # 3. 向量化计算匹配数（无for循环，速度快）
        pred_expanded = pred_top4_indices.unsqueeze(2)  # [batch,4,1]
        true_expanded = true_topk_indices.unsqueeze(1) # [batch,1,k]
        matches = (pred_expanded == true_expanded)      # [batch,4,k]
        
        # 4. 统计每个样本匹配的数量
        num_correct = matches.any(dim=2).sum(dim=1).float()  # [batch]
        # 5. 计算平均Recall（匹配数 / k）
        avg_recall = torch.mean(num_correct / k).item()
        recall_results[k] = avg_recall
    
    return recall_results

def compute_mass_coverage_at_ks(pred_logits, true_logits, ks=TOP_K_LIST):
    """
    批量计算Mass@2/3/4（基于原始logits，无归一化）
    return: dict → {
        "ideal": {2: mass@2, 3: mass@3, 4: mass@4},  # 真实Top-K的得分占比
        "pred": {2: mass@2, 3: mass@3, 4: mass@4}     # 预测Top-K的真实得分占比
    }
    """
    # 1. 计算每个样本的所有专家得分总和（分母，避免除0）
    true_logits_sum = true_logits.sum(dim=-1, keepdim=True)  # [batch, 1]
    true_logits_sum = torch.clamp(true_logits_sum, min=1e-8)
    
    mass_results = {"ideal": {}, "pred": {}}
    for k in ks:
        # 2. 获取真实Top-K和预测Top-K的索引
        _, true_topk_indices = torch.topk(true_logits, k, dim=-1)    # [batch, k]
        _, pred_topk_indices = torch.topk(pred_logits, k, dim=-1)    # [batch, k]
        
        # 3. 收集Top-K的原始得分
        true_topk_values = torch.gather(true_logits, 1, true_topk_indices)  # [batch, k]
        pred_topk_values = torch.gather(true_logits, 1, pred_topk_indices)   # [batch, k]
        
        # 4. 计算得分占比（Mass）
        ideal_mass = (true_topk_values.sum(dim=-1) / true_logits_sum.squeeze(-1)).mean().item()
        pred_mass = (pred_topk_values.sum(dim=-1) / true_logits_sum.squeeze(-1)).mean().item()
        
        mass_results["ideal"][k] = ideal_mass
        mass_results["pred"][k] = pred_mass
    
    return mass_results

def compute_regression_metrics(pred_logits, true_logits):
    """保持不变：计算MSE/MAE"""
    mse = nn.functional.mse_loss(pred_logits, true_logits).item()
    mae = nn.functional.l1_loss(pred_logits, true_logits).item()
    return mse, mae

# ================= 显存释放函数 =================
def release_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# ================= 数据加载：拆分为源加载和单层目标加载 =================
def load_source_data():
    """保持不变"""
    print(f"📦 Loading Source Layer {SOURCE_LAYER} (Input Features)...")
    s_path = os.path.join(DATA_DIR, f"layer_{SOURCE_LAYER}_data.pt")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"源层数据不存在：{s_path}")
    s_data = torch.load(s_path, map_location="cpu")
    X = s_data['router_inputs'].float()
    print(f"   Input shape: {X.shape}")
    return X

def load_target_layer_data(layer_idx, num_samples):
    """保持不变"""
    t_path = os.path.join(DATA_DIR, f"layer_{layer_idx}_data.pt")
    if not os.path.exists(t_path):
        return None, None
    
    t_data = torch.load(t_path, map_location="cpu")
    if t_data['router_logits'].shape[0] != num_samples:
        print(f"⚠️ Layer {layer_idx} 样本数不匹配，跳过")
        return None, None
        
    return t_data['router_logits'].float(), t_data['router_indices'].long()

# ================= 单层训练函数 (修改测试阶段的指标计算) =================
def train_single_layer(layer_idx, X, device, num_gpus):
    print(f"\n{'='*20} Processing Layer {layer_idx} {'='*20}")
    
    # 1. 加载该层的目标数据
    Y_logits, Y_indices = load_target_layer_data(layer_idx, len(X))
    if Y_logits is None:
        print(f"❌ Layer {layer_idx} 数据无效，跳过。")
        return None
    # 2. 动态获取输入维度
    input_dim = X.shape[1] 
    print(f"   ⚙️ Model Input Dim: {input_dim}, Output Dim: {NUM_EXPERTS}")
    # 3. 构造Dataset和Loader
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

    # 4. 初始化模型
    model = LayerIndependentMLP(input_dim, NUM_EXPERTS, hidden_dim=1024).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = DistillationLoss(temp=2.0, alpha=0.7).to(device)

    # 5. 训练循环
    best_val_loss = float('inf')
    early_stop_cnt = 0
    history = {"train_loss": [], "val_loss": [], "mse": [], "recall": []}
    
    print(f"🚀 Start Training Layer {layer_idx}...")
    
    for epoch in range(EPOCHS):
        # --- Train ---
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

        # --- Val ---
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

        # --- Early Stop & Save ---
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
                print(f"⏹️  Layer {layer_idx} Early stopping at epoch {epoch}")
                break
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Ep {epoch+1}: T_Loss={avg_train_loss:.4f}, V_Loss={avg_val_loss:.4f}")

    # 6. 测试阶段 (核心修改：计算多K值指标)
    print(f"🧪 Testing Layer {layer_idx}...")
    save_path = os.path.join(OUTPUT_DIR, f"best_model_layer_{layer_idx}.pth")
    if num_gpus > 1:
        model.module.load_state_dict(torch.load(save_path))
    else:
        model.load_state_dict(torch.load(save_path))
    model.eval()

    # 初始化多K值指标累加器
    recall_accum = {k: 0.0 for k in TOP_K_LIST}  # Recall@2/3/4
    mass_ideal_accum = {k: 0.0 for k in TOP_K_LIST}  # Ideal_Mass@2/3/4
    mass_pred_accum = {k: 0.0 for k in TOP_K_LIST}   # Pred_Mass@2/3/4
    test_mse = 0.0
    test_mae = 0.0
    
    with torch.no_grad():
        for bx, by_logits, by_indices in test_loader:
            bx = bx.to(device, non_blocking=True)
            by_logits = by_logits.to(device, non_blocking=True)
            by_indices = by_indices.to(device, non_blocking=True)

            pred_logits = model(bx)
            
            # 计算多K值Recall（预测Top-4覆盖真实Top-2/3/4）
            recall_dict = compute_recall_at_ks(pred_logits, by_indices, ks=TOP_K_LIST)
            for k in TOP_K_LIST:
                recall_accum[k] += recall_dict[k]
            
            # 计算多K值Mass Coverage（原始logits得分占比）
            mass_dict = compute_mass_coverage_at_ks(pred_logits, by_logits, ks=TOP_K_LIST)
            for k in TOP_K_LIST:
                mass_ideal_accum[k] += mass_dict["ideal"][k]
                mass_pred_accum[k] += mass_dict["pred"][k]
            
            # 计算MSE/MAE
            mse, mae = compute_regression_metrics(pred_logits, by_logits)
            test_mse += mse
            test_mae += mae
    
    # 计算平均指标
    final_recall = {k: recall_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mass_ideal = {k: mass_ideal_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mass_pred = {k: mass_pred_accum[k]/len(test_loader) for k in TOP_K_LIST}
    final_mse = test_mse / len(test_loader)
    final_mae = test_mae / len(test_loader)

    # 打印多K值结果（清晰展示Top-2/3/4）
    print(f"✅ Layer {layer_idx} Result:")
    print(f"   Recall → Top2={final_recall[2]:.4f}, Top3={final_recall[3]:.4f}, Top4={final_recall[4]:.4f}")
    print(f"   Ideal_Mass → Top2={final_mass_ideal[2]:.4f}, Top3={final_mass_ideal[3]:.4f}, Top4={final_mass_ideal[4]:.4f}")
    print(f"   Pred_Mass → Top2={final_mass_pred[2]:.4f}, Top3={final_mass_pred[3]:.4f}, Top4={final_mass_pred[4]:.4f}")
    print(f"   Regression → MSE={final_mse:.6f}, MAE={final_mae:.6f}")
    
    # 返回包含多K值的结果
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
        # 原有指标
        "mse": final_mse,
        "mae": final_mae,
        "val_loss": best_val_loss
    }

# ================= 主程序 (修改CSV表头和写入逻辑) =================
def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Found {num_gpus} GPUs. Running Serial Training per Layer.")

    # 1. 加载公共输入 X
    X = load_source_data()
    
    # 2. 初始化CSV结果文件（新增多K值表头）
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
            # 原有指标
            "MSE", "MAE", "Best_Val_Loss"
        ])
    
    all_histories = {}

    # 3. 串行循环每一层
    for layer_idx in TARGET_LAYERS:
        release_gpu_memory()
        
        try:
            hist, metrics = train_single_layer(layer_idx, X, device, num_gpus)
            
            if metrics is None: continue
            
            all_histories[layer_idx] = hist
            
            # 写入多K值指标到CSV
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
                    # 原有指标
                    f"{metrics['mse']:.6f}", f"{metrics['mae']:.6f}", f"{metrics['val_loss']:.4f}"
                ])
                
        except Exception as e:
            print(f"⚠️ Error training layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. 画图（可选：新增多K值Recall对比图）
    print("\n📊 Generating Plots...")
    if all_histories:
        # 子图1：验证损失
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 1, 1)
        for l, hist in all_histories.items():
            if l % 5 == 0 or l == 22: 
                plt.plot(hist["val_loss"], label=f"L{l} Val Loss")
        plt.title("Validation Loss Curve (Selected Layers)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：关键层的Recall@2/3/4对比
        plt.subplot(2, 1, 2)
        plot_layers = [l for l in all_histories.keys() if l % 5 == 0 or l == 22]
        if plot_layers:
            # 提取Recall数据
            recall_2 = [metrics["recall_2"] for l, metrics in all_histories.items() if l in plot_layers]
            recall_3 = [metrics["recall_3"] for l, metrics in all_histories.items() if l in plot_layers]
            recall_4 = [metrics["recall_4"] for l, metrics in all_histories.items() if l in plot_layers]
            
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
        print(f"✅ Training Complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()