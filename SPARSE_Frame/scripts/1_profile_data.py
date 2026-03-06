import sys
import os
import torch
import json
import gzip
import gc
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""
Script Name: 1_profile_data.py
Description: 
    Profiles a pre-trained MoE model to capture data for offline RS & SCP.
    It collects:
    1. Layer-0 Inputs: Post-attention hidden states for training the Routing Scout (RS).
    2. Router Outputs: Logits and selected expert indices for distillation targets.
    3. Co-activation Matrix: Global expert-expert correlation stats for SCP graph partitioning.
"""

def main():
    ap = argparse.ArgumentParser(description="Profile MoE model for RS & SCP data collection")
    ap.add_argument("--model-path", type=str, default="/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
    ap.add_argument("--dataset", type=str, default="/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz")
    ap.add_argument("--base-dir", type=str, default="/home/ztl/wyf/SPARSE/SPARSE_Frame/results/profile_data/Qwen1.5MoE_allData")
    ap.add_argument("--num-samples", type=int, default=8000)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--save-interval", type=int, default=4000)
    args = ap.parse_args()

    MODEL_PATH = args.model_path       
    DATA_PATH = args.dataset          
    BASE_DIR = args.base_dir        
    NUM_SAMPLES = args.num_samples   
    MAX_SEQ_LEN = args.max_seq_len  
    SAVE_INTERVAL = args.save_interval 
    DATA_SAVE_DIR = os.path.join(BASE_DIR, "rs_chunks")

    os.makedirs(BASE_DIR, exist_ok=True)      
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)  

    print(f"Loading Qwen from {MODEL_PATH}...")  
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)  
        # ========== 核心修改1：开启output_router_logits ==========
        MEMORY_MAP = {0: "9GiB", 1: "9GiB", 2: "9GiB", 3: "9GiB", "cpu": "80GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,  
            device_map="auto",           
            max_memory=MEMORY_MAP,
            torch_dtype=torch.float16,   
            attn_implementation="eager",
            trust_remote_code=True,
            output_router_logits=True  # 关键：启用router_logits输出
        )
        model.eval()
        torch.set_grad_enabled(False)
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    num_layers = model.config.num_hidden_layers
    num_experts_per_layer = model.config.num_experts
    top_k = getattr(model.config, "num_experts_per_tok", 4) 
    hidden_size = model.config.hidden_size
    
    print(f"Model Config: {num_layers} Layers, {num_experts_per_layer} Experts/Layer, Top-{top_k}, Hidden Size: {hidden_size}")
    
    # rs_inputs_l0 = []  # Layer 0 Post-attention hidden-state
    rs_inputs = {i: [] for i in range(num_layers)}  # 改为字典，存储每层的post-attention输入

    rs_labels = {i: {"logits": [], "indices": [], "weights": []} for i in range(num_layers)}
    total_experts = num_layers * num_experts_per_layer
    global_co_act_matrix = torch.zeros(total_experts, total_experts, dtype=torch.float32)

    def get_router_hook(layer_idx):
        def hook_fn(module, args, output):
            """
            Hook函数解析MLP层输出：
            - args[0]: MLP输入 → Post-attention hidden-state [Batch, SeqLen, HiddenSize]
            - output: MLP返回值 → (final_hidden_states, router_logits)
            """
            try:
                if isinstance(output, tuple):
                    final_hidden_states, router_logits = output
                else:
                    final_hidden_states = output
                    router_logits = None
                
                if router_logits is None:
                    print(f"Warning: Layer {layer_idx} MLP未返回router_logits!")
                    return
                
                # ========== 1. 收集Post-attention hidden-state（Layer 0） ==========
                batch_size, seq_len, _ = args[0].shape
                
                post_attn_hidden = args[0].reshape(-1, hidden_size).detach().cpu().half()
                rs_inputs[layer_idx].append(post_attn_hidden)
                if layer_idx == 0 and len(rs_inputs[layer_idx]) == 1:
                    print(f"\nLayer 0 Post-attention hidden-state形状: {args[0].shape}")
                    print(f"Layer 0 Router logits形状: {router_logits.shape}")
                
                # ========== 2. 计算选中专家的索引和权重（和源码一致） ==========
                # router_logits形状：[Batch*SeqLen, num_experts]
                routing_weights = F.softmax(router_logits.float(), dim=1)  # 所有专家概率
                routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # Top-k
                
                # 归一化Top-k权重（和源码一致）
                if getattr(module, "norm_topk_prob", True):
                    routing_weights_topk /= routing_weights_topk.sum(dim=-1, keepdim=True)
                
                # ========== 3. 格式化并保存数据 ==========
                # Logits：[Batch*SeqLen, num_experts]
                logits_flat = router_logits.reshape(-1, num_experts_per_layer).detach().cpu().float()
                # Indices：[Batch*SeqLen, top_k]
                indices_flat = selected_experts.reshape(-1, top_k).detach().cpu().to(torch.int16)
                # Weights：[Batch*SeqLen, top_k]
                weights_flat = routing_weights_topk.reshape(-1, top_k).detach().cpu().float()
                
                rs_labels[layer_idx]["logits"].append(logits_flat)
                rs_labels[layer_idx]["indices"].append(indices_flat)
                rs_labels[layer_idx]["weights"].append(weights_flat)
                
                # Debug：第一层打印示例数据
                if layer_idx == 0 and len(rs_labels[0]["logits"]) == 1:
                    print(f"Layer 0 第一个token前5个专家logits: {logits_flat[0, :5].tolist()}")
                    print(f"Layer 0 第一个token选中专家索引: {indices_flat[0, :].tolist()}")
                    print(f"Layer 0 第一个token选中专家权重: {weights_flat[0, :].tolist()}")
                    
            except Exception as e:
                print(f"Hook函数报错 (Layer {layer_idx}): {e}")
                import traceback
                traceback.print_exc()
                
        return hook_fn

    print("\nHooking MoE MLP Layers...")
    hook_count = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, model.model.layers[0].mlp.__class__):
            layer.mlp.register_forward_hook(get_router_hook(i))
            hook_count += 1
    
    if hook_count == 0:
        print("CRITICAL: No MoE MLP layers hooked! Check model structure.")
        return
    print(f"Hooked {hook_count} MLP layers.")

    def data_gen():
        count = 0
        if not os.path.exists(DATA_PATH): 
            print(f"Data file not found: {DATA_PATH}") 
            return
            
        with gzip.open(DATA_PATH, 'rt', encoding='utf-8') as f: 
            for line in f:
                if count >= NUM_SAMPLES: break 
                try:
                    text = json.loads(line).get('text', '')
                    if len(text) > 30:
                        enc = tokenizer(
                            text, 
                            truncation=True, 
                            max_length=MAX_SEQ_LEN, 
                            return_tensors='pt',
                            padding="max_length"  # 确保长度一致
                        )
                        yield enc
                        count += 1
                except Exception as e:
                    continue

    print(f"\nSTART Profiling {NUM_SAMPLES} samples...") 
    first_device = next(model.parameters()).device
    count = 0
    buffer_steps = 0
    
    try:
        with torch.no_grad():
            for enc in tqdm(data_gen(), total=NUM_SAMPLES):
                # 数据移到设备
                input_ids = enc.input_ids.to(first_device)
                attention_mask = enc.attention_mask.to(first_device)
                
                # 前向传播（触发Hook）
                model(input_ids=input_ids, attention_mask=attention_mask)
                
                buffer_steps += 1
                count += 1
                
                # 定期保存数据
                if buffer_steps >= SAVE_INTERVAL:  
                    print(f"\n[Buffer] Flushing at step {count}...")
                    
                    if len(rs_labels[0]["indices"]) == 0:
                        print("Warning: No data captured in this buffer!")
                    else:
                        # 更新共活矩阵
                        _update_co_act_matrix(global_co_act_matrix, rs_labels, num_experts_per_layer, num_layers, top_k)
                        # 保存分块数据
                        _save_chunk(count, rs_inputs, rs_labels, DATA_SAVE_DIR)  
                    
                    for layer_idx in rs_inputs:
                        rs_inputs[layer_idx].clear()
                    for k in rs_labels: 
                        rs_labels[k]["logits"].clear()
                        rs_labels[k]["indices"].clear()
                        rs_labels[k]["weights"].clear()
                    buffer_steps = 0
                    gc.collect()
                
    except KeyboardInterrupt:
        print("\nInterrupted! Saving collected data...")
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()

    if buffer_steps > 0 and len(rs_labels[0]["indices"]) > 0:
        _update_co_act_matrix(global_co_act_matrix, rs_labels, num_experts_per_layer, num_layers, top_k)
        _save_chunk(count, rs_inputs, rs_labels, DATA_SAVE_DIR) 
    
    non_zeros = torch.count_nonzero(global_co_act_matrix).item()
    print(f"\nGlobal Co-activation Matrix Stats:")
    print(f" - Shape: {global_co_act_matrix.shape}")
    print(f" - Non-zero elements: {non_zeros}")
    print(f" - Total elements: {global_co_act_matrix.numel()}")
    
    if non_zeros == 0:
        print("ERROR: Co-activation matrix is empty! Profiling failed.")
    else:
        print(f"Saving Global Co-activation Matrix to {os.path.join(BASE_DIR, 'co_activation.pt')}")
        torch.save(global_co_act_matrix, os.path.join(BASE_DIR, "co_activation.pt")) 
        print(f"Profiling Done.")
        print(f" - Training chunks saved to {DATA_SAVE_DIR}")
        print(f" - Co-activation matrix saved to {os.path.join(BASE_DIR, 'co_activation.pt')}")


def _update_co_act_matrix(matrix, labels_dict, experts_per_layer, num_layers, top_k):
    """
    优化的共活矩阵计算逻辑：
    - 输入：labels_dict 包含每层的选中专家索引
    - 逻辑：每个token在所有层选中的专家两两共现，矩阵对应位置+1
    """
    try:
        # 收集所有层的选中专家索引
        all_layer_indices = []
        min_len = min([len(labels_dict[l]['indices']) for l in range(num_layers)])
        if min_len == 0: 
            print("Warning: No indices data for co-activation matrix!")
            return

        for l in range(num_layers):
            # 拼接当前层所有样本的索引 → [Total_Tokens, TopK]
            layer_ind_list = labels_dict[l]['indices'][:min_len]
            if len(layer_ind_list) == 0:
                continue
            layer_ind = torch.cat(layer_ind_list, dim=0).long()
            
            # 转换为全局专家ID（层号*每层专家数 + 本地索引）
            global_ids = layer_ind + (l * experts_per_layer)
            all_layer_indices.append(global_ids)
        
        # 拼接所有层 → [Total_Tokens, Layers, TopK] → 重塑为 [Total_Tokens, Layers*TopK]
        token_expert_path = torch.stack(all_layer_indices, dim=1).reshape(-1, num_layers * top_k)
        
        # 计算共活矩阵（分批处理避免内存溢出）
        N = token_expert_path.shape[0]
        batch_size = 8192 if N > 8192 else N
        
        for i in range(0, N, batch_size):
            batch_path = token_expert_path[i:i+batch_size] 
            # 转为Multi-hot向量 [Batch, Total_Experts]
            multi_hot = torch.zeros(batch_path.shape[0], matrix.shape[0], dtype=torch.float32)
            multi_hot.scatter_(1, batch_path, 1.0)
            # 累加共活：A^T @ A
            matrix += torch.matmul(multi_hot.T, multi_hot)
            
    except Exception as e:
        print(f"Error computing co-activation matrix: {e}")
        import traceback
        traceback.print_exc()


def _save_chunk(step_idx, inputs, labels_dict, data_save_dir):
    """保存分块数据，包含所有层Post-attention hidden-state、logits、选中专家索引"""
    try:
        # 1. 保存所有Layer的Post-attention hidden-state（修改：遍历所有层）
        # 原代码：if len(inputs) > 0:
        for layer_idx in inputs:
            layer_inputs = inputs[layer_idx]
            if len(layer_inputs) == 0:
                continue
            inputs_cat = torch.cat(layer_inputs, dim=0)
            chunk_path = os.path.join(data_save_dir, f"rs_inputs_l{layer_idx}_chunk_{step_idx}.pt")  
            torch.save(inputs_cat, chunk_path)
            print(f" - Saved Layer {layer_idx} inputs: {chunk_path} (shape: {inputs_cat.shape})")
        
        # 2. 保存每层的Logits和选中专家索引（原有逻辑不变）
        for layer_idx, data in labels_dict.items():
            if len(data["logits"]) == 0:
                continue
            logits_cat = torch.cat(data["logits"], dim=0)
            indices_cat = torch.cat(data["indices"], dim=0)
            weights_cat = torch.cat(data["weights"], dim=0)
            
            save_dict = {
                "logits": logits_cat,          # [Total_Tokens, num_experts]
                "indices": indices_cat,        # [Total_Tokens, top_k]
                "weights": weights_cat,        # [Total_Tokens, top_k]
                "num_experts": logits_cat.shape[-1],
                "top_k": indices_cat.shape[-1]
            }
            chunk_path = os.path.join(data_save_dir, f"rs_labels_l{layer_idx}_chunk_{step_idx}.pt") 
            torch.save(save_dict, chunk_path)
            print(f" - Saved Layer {layer_idx} labels: {chunk_path} (logits shape: {logits_cat.shape}, indices shape: {indices_cat.shape})")
            
    except Exception as e:
        print(f"Error saving chunk {step_idx}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()