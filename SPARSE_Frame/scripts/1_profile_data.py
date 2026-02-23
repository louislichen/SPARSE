import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import torch
import json
import gzip
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import Config 

"""
Script Name: 1_profile_data.py
Description: 
    Profiles a pre-trained MoE model to capture data for offline RS & SCP.
    It collects:
    1. Layer-0 Inputs: Post-attention hidden states for training the Routing Scout (RS).
    2. Router Outputs: Logits and selected expert indices for distillation targets.
    3. Co-activation Matrix: Global expert-expert correlation stats for SCP graph partitioning.
"""

NUM_SAMPLES = 20000       # 采样数量，确保足够覆盖共活模式
MAX_SEQ_LEN = 512        
SAVE_INTERVAL = 5000      

def main():
    os.makedirs(Config.DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.BASE_DIR, exist_ok=True)

    print(f"Loading Qwen from {Config.MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_PATH, 
            device_map="auto",           
            torch_dtype=torch.float16,   
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    num_layers = model.config.num_hidden_layers
    num_experts_per_layer = model.config.num_experts
    top_k = getattr(model.config, "num_experts_per_tok", 4) 
    
    print(f"Model Config: {num_layers} Layers, {num_experts_per_layer} Experts/Layer, Top-{top_k}")
    
    rs_inputs_l0 = []
    rs_labels = {i: {"logits": [], "indices": []} for i in range(num_layers)}
    total_experts = num_layers * num_experts_per_layer
    global_co_act_matrix = torch.zeros(total_experts, total_experts, dtype=torch.float32)

    def get_router_hook(layer_idx):
        def hook_fn(module, args, output):
            # args[0]: [Batch, Seq, Hidden]
            # output: tuple(logits, weights, indices)
            if layer_idx == 0:
                inp = args[0].reshape(-1, args[0].size(-1)).detach().cpu().half()
                rs_inputs_l0.append(inp)
            logits = output[0].reshape(-1, output[0].size(-1)).detach().cpu().float()
            indices = output[2].reshape(-1, output[2].size(-1)).detach().cpu().to(torch.int16)
            
            rs_labels[layer_idx]["logits"].append(logits)
            rs_labels[layer_idx]["indices"].append(indices)
            
        return hook_fn

    print("Hooking MoE Gates...")
    hook_count = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            layer.mlp.gate.register_forward_hook(get_router_hook(i))
            hook_count += 1
    
    if hook_count == 0:
        print("CRITICAL: No MoE Gates hooked! Check model structure names.")
        return
    print(f"Hooked {hook_count} layers.")

    def data_gen():
        count = 0
        if not os.path.exists(Config.DATA_PATH):
            print(f"Data file not found: {Config.DATA_PATH}")
            return
            
        with gzip.open(Config.DATA_PATH, 'rt', encoding='utf-8') as f:
            for line in f:
                if count >= NUM_SAMPLES: break
                try:
                    text = json.loads(line).get('text', '')
                    if len(text) > 30:
                        enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
                        yield enc.input_ids
                        count += 1
                except: continue

    print(f"START Profiling {NUM_SAMPLES} samples...")
    first_device = next(model.parameters()).device
    
    count = 0
    buffer_steps = 0
    
    try:
        with torch.no_grad():
            for input_ids in tqdm(data_gen(), total=NUM_SAMPLES):
                input_ids = input_ids.to(first_device)
                model(input_ids)
                buffer_steps += 1
                
                if buffer_steps >= SAVE_INTERVAL:
                    print(f"\n[Buffer] Flushing at step {count}...")
                    
                    if len(rs_labels[0]["indices"]) == 0:
                        print("Warning: No data captured in this buffer! Check Hook logic.")
                    else:
                        _update_co_act_matrix(global_co_act_matrix, rs_labels, num_experts_per_layer, num_layers)
                        _save_chunk(count, rs_inputs_l0, rs_labels)
                    
                    rs_inputs_l0.clear()
                    for k in rs_labels: 
                        rs_labels[k]["logits"].clear()
                        rs_labels[k]["indices"].clear()
                    buffer_steps = 0
                    gc.collect()
                
                count += 1
                
    except KeyboardInterrupt:
        print("\nInterrupted! Saving collected data...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    if buffer_steps > 0 and len(rs_labels[0]["indices"]) > 0:
        _update_co_act_matrix(global_co_act_matrix, rs_labels, num_experts_per_layer, num_layers)
        _save_chunk(count, rs_inputs_l0, rs_labels)
    
    non_zeros = torch.count_nonzero(global_co_act_matrix).item()
    print(f"Global Matrix Non-Zeros: {non_zeros}")
    
    if non_zeros == 0:
        print("ERROR: Matrix is still empty! Profiling failed.")
    else:
        print("Saving Global Co-activation Matrix...")
        torch.save(global_co_act_matrix, os.path.join(Config.BASE_DIR, "co_activation.pt"))
        print(f"Profiling Done. \n - Matrix saved to {os.path.join(Config.BASE_DIR, 'co_activation.pt')} \n - Training chunks in {Config.DATA_SAVE_DIR}")


def _update_co_act_matrix(matrix, labels_dict, experts_per_layer, num_layers):
    """
    计算 Token 维度的跨层共活。
    逻辑：如果 Token T 在 Layer 1 选中专家 A，在 Layer 2 选中专家 B，则 Matrix[A, B] += 1
    """
    try:
        all_layer_indices = []
        
        min_len = min([len(labels_dict[l]['indices']) for l in range(num_layers)])
        if min_len == 0: return

        for l in range(num_layers):
            layer_ind_list = labels_dict[l]['indices'][:min_len]
            layer_ind = torch.cat(layer_ind_list, dim=0).long() # [Total_Tokens, TopK]
            
            global_ids = layer_ind + (l * experts_per_layer)
            all_layer_indices.append(global_ids)
        
        # Stack -> [Total_Tokens, Layers, TopK]
        # View -> [Total_Tokens, Layers * TopK] 
        token_expert_path = torch.stack(all_layer_indices, dim=1).view(all_layer_indices[0].shape[0], -1)
        
        N = token_expert_path.shape[0]
        batch_size = 4096 
        
        for i in range(0, N, batch_size):
            batch_path = token_expert_path[i:i+batch_size] 
            
            # 转为 Multi-hot 向量 [B, Total_Experts]
            multi_hot = torch.zeros(batch_path.shape[0], matrix.shape[0])
            multi_hot.scatter_(1, batch_path, 1.0)
            
            # 累加共活: A^T @ A
            # A[t, i]=1 且 A[t, j]=1 => Result[i, j] += 1
            matrix += torch.matmul(multi_hot.T, multi_hot)
            
    except Exception as e:
        print(f"Error computing co-activation: {e}")
        import traceback
        traceback.print_exc()

def _save_chunk(step_idx, inputs, labels_dict):
    """保存分块数据，供后续 RS 训练使用"""
    # 1. 保存 Inputs (Layer 0 Post-Attention)
    if len(inputs) > 0:
        chunk_path = os.path.join(Config.DATA_SAVE_DIR, f"rs_inputs_l0_chunk_{step_idx}.pt")
        torch.save(torch.cat(inputs, dim=0), chunk_path)
    
    # 2. 保存 Labels (Logits 用于蒸馏，Indices 用于 Recall 计算)
    for layer_idx, data in labels_dict.items():
        if len(data["logits"]) > 0:
            save_dict = {
                "logits": torch.cat(data["logits"], dim=0),
                "indices": torch.cat(data["indices"], dim=0) 
            }
            chunk_path = os.path.join(Config.DATA_SAVE_DIR, f"rs_labels_l{layer_idx}_chunk_{step_idx}.pt")
            torch.save(save_dict, chunk_path)

if __name__ == "__main__":
    main()