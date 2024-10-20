import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.mlp import MLPBase
from harl.models.base.simple_layers import CrossAttention, GAT, MLP_improve, MultiVeh_GAT, TrajectoryDecoder
from einops import rearrange, repeat
import time

class CACC_rep(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(CACC_rep, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        self.hist_length = args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.para_env = args['n_rollout_threads']

        # 初始化网络模块
        self.gat_surround = GAT(nfeat=26, nhid=32, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.mlp_surround_lane = MLPBase(args, [2 * 10])
        self.mlp_local_combined = MLPBase(args, [11 + 2 + 64 * 2])

        # 定义模板结构，使用float类型的张量来节省内存
        self.example_extend_info = {
            'road_structure': torch.zeros(8, dtype=torch.float32),               # 0
            'next_node': torch.zeros(3, dtype=torch.float32),                    # 1
            'self_stats': torch.zeros(12, dtype=torch.float32),                  # 2
            'self_hist_stats': torch.zeros(1, 26, dtype=torch.float32),          # 3
            'surround_2_stats': torch.zeros(2, 26, dtype=torch.float32),         # 4
            'surround_4_stats': torch.zeros(4, 26, dtype=torch.float32),         # 5
            'surround_6_stats': torch.zeros(6, 26, dtype=torch.float32),         # 6
            'surround_lane_stats': torch.zeros(2, 10, dtype=torch.float32),      # 7
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return (reconstructed['road_structure'], reconstructed['next_node'],
                reconstructed['self_stats'], reconstructed['self_hist_stats'],
                reconstructed['surround_2_stats'],
                reconstructed['surround_4_stats'],
                reconstructed['surround_6_stats'],
                reconstructed['surround_lane_stats'])

    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        info_current = self.reconstruct_info(obs)
        road_static_info = torch.cat((info_current[0], info_current[1]), dim=1)
        ego_last_action = info_current[2][:, 10:12]
        flow_info = info_current[7]
        flow_embedding = self.mlp_surround_lane(flow_info.view(flow_info.size(0), -1))
        ego_hist = info_current[3]
        surround_hist_2 = info_current[4]
        veh_hist = torch.cat((ego_hist, surround_hist_2), dim=1)

        # 使用预先分配的邻接矩阵，减少每次计算时的内存开销
        if not hasattr(self, 'adj') or self.adj.size(0) != batch_size:
            self.adj = torch.ones(batch_size, 3, 3, device=obs.device)
            self.adj[:, 1:, 1:] = 0

        ego_relation = self.gat_surround(veh_hist, self.adj)
        local_combined_embedding = torch.cat((road_static_info, ego_last_action, flow_embedding, ego_relation), dim=1)
        local_combined_embedding = self.mlp_local_combined(local_combined_embedding)

        return local_combined_embedding

    def reconstruct_obs_batch(self, obs_batch, template_structure):
        device = obs_batch.device  # 获取 obs_batch 的设备

        # 计算每个张量的大小，并使用 torch.split 进行拆分
        sizes = [tensor.numel() for tensor in template_structure.values()]
        split_tensors = torch.split(obs_batch, sizes, dim=1)

        # 重新构造观测数据
        reconstructed_batch = {
            key: split_tensor.view(obs_batch.size(0), *template_structure[key].shape).to(device)
            for key, split_tensor in zip(template_structure.keys(), split_tensors)
        }

        return reconstructed_batch