import sys
import os
import torch
import json
import time
import numpy as np
import math
import random
from tqdm import tqdm
from transformers import AutoConfig

# 添加上级目录以导入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import Config

class HardwareProfiler:
    """真实物理环境 Profiling (维持原样，这是正确的物理测量方式)"""
    def __init__(self):
        self.num_devices = torch.cuda.device_count()
        print(f"[Profiler] Detected {self.num_devices} GPUs.")

    def _measure_compute_power(self, device_id):
        size = 4096
        a = torch.randn(size, size, device=device_id, dtype=torch.float16)
        b = torch.randn(size, size, device=device_id, dtype=torch.float16)
        for _ in range(5): _ = torch.matmul(a, b)
        torch.cuda.synchronize(device_id)
        start = time.time()
        iterations = 10
        for _ in range(iterations): _ = torch.matmul(a, b)
        torch.cuda.synchronize(device_id)
        end = time.time()
        avg_time = (end - start) / iterations
        return (2 * size**3) / (avg_time + 1e-8)

    def get_node_constraints(self):
        print("[Profiler] Profiling Node Constraints (Memory M_k & Compute C_k)...")
        nodes = []
        MEM_SAFETY_RATIO = 0.8 # 留出20%给系统和KV Cache
        for i in range(self.num_devices):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory
            compute_score = self._measure_compute_power(i)
            nodes.append({
                'id': i,
                'memory_limit': int(total_mem * MEM_SAFETY_RATIO),
                'compute_limit': compute_score,
                'name': props.name
            })
            print(f"    [Node {i}] Mem Limit: {nodes[-1]['memory_limit']/1024**3:.2f}GB, Compute Score: {compute_score/1e12:.2f} TFLOPS")
        return nodes

    def get_communication_cost_matrix(self):
        print("[Profiler] Profiling P2P Communication Cost (D_k,l)...")
        cost_matrix = np.zeros((self.num_devices, self.num_devices))
        data_size_mb = 100 
        for i in range(self.num_devices):
            for j in range(self.num_devices):
                if i == j:
                    cost_matrix[i][j] = 0.0
                    continue
                try:
                    src_tensor = torch.randn(data_size_mb * 1024 * 1024 // 4, device=f"cuda:{i}", dtype=torch.float32)
                    torch.cuda.synchronize(i)
                    torch.cuda.synchronize(j)
                    start = time.time()
                    dst_tensor = src_tensor.to(f"cuda:{j}")
                    torch.cuda.synchronize(j)
                    end = time.time()
                    cost_matrix[i][j] = end - start
                    del src_tensor, dst_tensor
                    torch.cuda.empty_cache()
                except Exception:
                    cost_matrix[i][j] = 10.0
        max_cost = cost_matrix.max()
        if max_cost > 0: cost_matrix = cost_matrix / max_cost
        print("[Profiler] Communication Cost Matrix D_k,l:")
        print(np.round(cost_matrix, 4))
        return cost_matrix

class SCPSolver:
    """
    重构后的 Semantic Co-activation Planner
    实现了 Affinity-Driven (亲和力驱动) 的图划分算法
    """
    def __init__(self, matrix_path, model_config_path, nodes, dist_matrix):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.dist_matrix = dist_matrix
        
        print(f"[SCP] Loading Co-activation Matrix...")
        self.p_matrix = torch.load(matrix_path).float().numpy()
        self.num_experts = self.p_matrix.shape[0]
        
        self.expert_sizes, self.expert_loads = self._profile_experts(model_config_path)
        
        # 将对角线清零（我们只关心跨专家共激活，自身激活度已经记入 Load）
        np.fill_diagonal(self.p_matrix, 0.0)
        # 归一化，防止数值溢出
        if self.p_matrix.max() > 0:
            self.p_matrix /= self.p_matrix.max()

    def _profile_experts(self, config_path):
        print("[SCP] Profiling Expert Attributes...")
        try:
            config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
            hidden_size = config.hidden_size
            intermediate_size = config.intermediate_size
            expert_param_count = 3 * hidden_size * intermediate_size
            expert_size_bytes = expert_param_count * 2 # FP16
            
            m_i = [expert_size_bytes] * self.num_experts
            # 使用对角线作为真实计算负载
            diag = np.diag(torch.load(os.path.join(Config.BASE_DIR, "co_activation.pt")).float().numpy())
            l_i = diag.tolist()
            return m_i, l_i
        except Exception:
            print("[Warning] Using fallback expert properties.")
            return [100 * 1024**2] * self.num_experts, [1.0] * self.num_experts

    def _objective_function(self, placement):
        """目标函数：通信代价 (论文公式)"""
        nodes_mapped = np.array(placement)
        rows, cols = np.nonzero(self.p_matrix)
        mask = rows < cols # 优化：对称矩阵只算一半
        rows, cols = rows[mask], cols[mask]
        
        p_vals = self.p_matrix[rows, cols]
        n_i = nodes_mapped[rows]
        n_j = nodes_mapped[cols]
        d_vals = self.dist_matrix[n_i, n_j]
        
        return np.sum(p_vals * d_vals)

    def _check_constraints(self, placement):
        """严格检查物理约束"""
        node_mem = np.zeros(self.num_nodes)
        node_load = np.zeros(self.num_nodes)
        
        for exp_id, node_id in enumerate(placement):
            node_mem[node_id] += self.expert_sizes[exp_id]
            node_load[node_id] += self.expert_loads[exp_id]
            
        # 显存硬约束
        for k in range(self.num_nodes):
            if node_mem[k] > self.nodes[k]['memory_limit']:
                return False
        
        # 算力负载约束 (允许一定的不均衡，以换取极低的通信代价)
        total_load = sum(self.expert_loads)
        total_compute = sum(n['compute_limit'] for n in self.nodes) + 1e-6
        for k in range(self.num_nodes):
            target_load = total_load * (self.nodes[k]['compute_limit'] / total_compute)
            # 容忍度设为 1.8，优先保证通信聚类
            if node_load[k] > target_load * 1.8: 
                return False
        return True

    def _affinity_initialization(self):
        """
        🚀 创新点 1：亲和力驱动的贪心初始化
        不再顺序分配！而是让共激活度高的专家抱团。
        """
        print("[SCP] Running Affinity-Aware Initialization...")
        placement = [-1] * self.num_experts
        unassigned = set(range(self.num_experts))
        
        # 计算每个节点当前的显存
        node_mem = [0] * self.num_nodes
        
        # 按照专家热度排序，优先分配最热的专家
        sorted_experts = np.argsort(self.expert_loads)[::-1]
        
        for exp in sorted_experts:
            if placement[exp] != -1: continue
            
            # 计算将 exp 放入各节点的“引力” (亲和力)
            affinities = np.zeros(self.num_nodes)
            for node_id in range(self.num_nodes):
                # 遍历已经在该节点上的专家，计算总共激活值
                assigned_to_node = [i for i in range(self.num_experts) if placement[i] == node_id]
                if assigned_to_node:
                    affinities[node_id] = np.sum(self.p_matrix[exp, assigned_to_node])
                else:
                    affinities[node_id] = 0.001 # 给予空节点一点点小权重，防止一直扎堆
            
            # 按照亲和力从大到小尝试放置
            best_nodes = np.argsort(affinities)[::-1]
            placed = False
            for node_id in best_nodes:
                if node_mem[node_id] + self.expert_sizes[exp] <= self.nodes[node_id]['memory_limit']:
                    placement[exp] = node_id
                    node_mem[node_id] += self.expert_sizes[exp]
                    unassigned.remove(exp)
                    placed = True
                    break
            
            # 如果因为约束导致所有节点都放不下 (极其罕见)，强行塞入显存最宽裕的
            if not placed:
                safest_node = np.argmin(node_mem)
                placement[exp] = safest_node
                node_mem[safest_node] += self.expert_sizes[exp]
                unassigned.remove(exp)
                
        return placement

    def solve(self, steps=5000):
        """
        🚀 创新点 2：引力驱动模拟退火 (Gravity-Guided SA)
        取代盲目的随机交换，极大地提高了寻优效率。
        """
        # 1. 智能初始化
        placement = self._affinity_initialization()
        current_cost = self._objective_function(placement)
        best_placement = placement[:]
        best_cost = current_cost
        
        print(f"[SCP] Initial Cost after Affinity Grouping: {current_cost:.4f}")
        
        T = 1.0
        alpha = 0.999
        
        pbar = tqdm(range(steps), desc="Optimizing Placements")
        for step in pbar:
            # --- 智能交换提案 (Smart Proposal) ---
            # 随机挑选一个专家 A
            e1 = random.randint(0, self.num_experts - 1)
            n1 = placement[e1]
            
            # 看看哪个节点对专家 A 的“吸引力”最大
            pulls = np.zeros(self.num_nodes)
            for j in range(self.num_experts):
                if self.p_matrix[e1, j] > 0:
                    pulls[placement[j]] += self.p_matrix[e1, j]
            
            # 如果其他节点的引力大于当前节点，尝试移过去
            best_n2 = np.argmax(pulls)
            
            if best_n2 == n1 or pulls[best_n2] == 0:
                # 已经处于最优位置，或者缺乏关联，执行一次常规的随机扰动跳出局部最优
                e2 = random.randint(0, self.num_experts - 1)
            else:
                # n2 对 A 吸引力大，但 n2 可能满了。我们从 n2 中找一个最想离开的专家 B 出来交换
                candidates_in_n2 = [i for i in range(self.num_experts) if placement[i] == best_n2]
                if not candidates_in_n2: continue
                
                # 找在 n2 中待得最不舒服的（在 n2 朋友少，但在 n1 朋友多的）
                e2 = min(candidates_in_n2, key=lambda x: np.sum(self.p_matrix[x, candidates_in_n2]) - np.sum(self.p_matrix[x, [i for i in range(self.num_experts) if placement[i] == n1]]))

            n2 = placement[e2]
            if n1 == n2: continue

            # 尝试交换
            placement[e1] = n2
            placement[e2] = n1
            
            # 评估
            if self._check_constraints(placement):
                new_cost = self._objective_function(placement)
                delta = new_cost - current_cost
                
                # Metropolis 接受准则
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_placement = placement[:]
                else:
                    # 拒绝，回滚
                    placement[e1] = n1
                    placement[e2] = n2
            else:
                # 违反约束，回滚
                placement[e1] = n1
                placement[e2] = n2
            
            T = max(T * alpha, 1e-5)
            if step % 50 == 0:
                pbar.set_description(f"Min Cost: {best_cost:.4f} | Temp: {T:.4f}")
                
        return best_placement

def main():
    matrix_path = os.path.join(Config.BASE_DIR, "co_activation.pt")
    model_config_path = Config.MODEL_PATH 
    
    if not os.path.exists(matrix_path):
        print(f"Error: Co-activation matrix not found at {matrix_path}")
        return

    # 1. 物理环境测绘
    profiler = HardwareProfiler()
    nodes = profiler.get_node_constraints() 
    dist_matrix = profiler.get_communication_cost_matrix() 
    
    # 2. SCP 求解
    solver = SCPSolver(matrix_path, model_config_path, nodes, dist_matrix)
    
    start_time = time.time()
    # 5000 步的智能搜索足以收敛，因为我们的初始化已经非常优秀了
    final_placement = solver.solve(steps=5000) 
    end_time = time.time()
    
    print(f"\n[SCP] Done in {end_time - start_time:.2f}s")
    
    # 3. 分析与导出
    placement_dict = {int(e): int(n) for e, n in enumerate(final_placement)}
    
    node_counts = [0] * len(nodes)
    for n in final_placement: node_counts[n] += 1
    
    print(f"=====================================")
    print(f"📊 Final Distribution: {node_counts}")
    print(f"🧠 Note: Experts are now clustered by Co-activation affinity, NOT sequence!")
    print(f"=====================================")
    
    save_path = Config.SCP_PATH
    with open(save_path, 'w') as f:
        json.dump(placement_dict, f, indent=4)
    print(f"[SCP] Placement plan saved to {save_path}")

if __name__ == "__main__":
    main()