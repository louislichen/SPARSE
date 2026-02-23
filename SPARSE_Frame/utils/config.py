import torch
import os

class Config:
    MODEL_PATH = "/home/ztl/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
    DATA_PATH = "/home/ztl/wyf/ICMoE/datasets/c4_local/c4-train.00000-of-01024.json.gz"
    BASE_DIR = "/home/ztl/wyf/SPARSE_MOE/results"
    
    DATA_SAVE_DIR = os.path.join(BASE_DIR, "profile_data")
    RS_WEIGHTS_DIR = os.path.join(BASE_DIR, "mlp_trainer")
    SCP_PATH = os.path.join(BASE_DIR, "scp_placement.json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "training_metrics")
       
    # SKVS 
    H2O_SINK = 4
    H2O_WINDOW = 32
    L0_GUIDANCE_RATIO = 0.2