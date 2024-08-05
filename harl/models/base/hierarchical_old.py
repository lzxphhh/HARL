import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical, Normal

import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.mlp import MLPBase


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (batch_size, num_nodes, in_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, num_nodes, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # shape: (batch_size, num_nodes, num_nodes, 2*out_features)
        return a_input


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=1):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            for _ in range(nheads)
        ])

        # Output linear layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        # import time
        # start_time = time.time()
        x = F.dropout(x, self.dropout, training=self.training)
        # Concatenate the outputs of the multi-head attentions
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply the output layer; aggregate with mean as we want a graph-level output
        x = F.elu(self.out_att(x, adj))
        x = torch.mean(x, dim=1)  # Average pooling over nodes to get the graph-level output
        # end_time = time.time()
        # print('Time taken for GAT layer: ', end_time - start_time)
        return x


class DynamicFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(DynamicFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, last_state):
        # x: (num_envs, num_vehs, size_traj)
        num_envs, size_traj = x.size()
        # Reshape x to (num_envs * num_vehs, -1, 3) for GRU processing
        x = x.reshape(num_envs, -1, 3)  # Each vehicle's state is (x, y, v)
        output, h_n = self.gru(x)  # h_n: (num_layers, num_envs * num_vehs, hidden_size)
        last_state = h_n
        h_n = h_n[-1]  # Get the last layer's hidden state: (num_envs * num_vehs, hidden_size)
        h_n = h_n.reshape(num_envs, -1)  # Reshape to (num_envs, num_vehs, hidden_size)
        dynamic_features = self.fc(h_n)  # Apply fully connected layer: (num_envs, num_vehs, output_size)
        return dynamic_features, last_state


class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(TrajectoryDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoded_features, future_steps):
        # Initialize hidden and cell state for LSTM
        h0 = torch.zeros(self.lstm.num_layers, encoded_features.size(0), self.lstm.hidden_size).to(
            encoded_features.device)
        c0 = torch.zeros(self.lstm.num_layers, encoded_features.size(0), self.lstm.hidden_size).to(
            encoded_features.device)

        # Repeat the encoded features for each future step
        repeated_features = encoded_features.unsqueeze(1).repeat(1, future_steps, 1)

        # Forward pass through LSTM
        out, _ = self.lstm(repeated_features, (h0, c0))

        # Forward pass through the fully connected layer
        pre_pos = self.fc(out)
        out = out.reshape(out.size(0), -1)  # Reshape to (batch_size, 1, :)
        pre_pos = pre_pos.view(pre_pos.size(0), -1)  # Reshape to (batch_size, 1, :)

        return out, pre_pos

# Example dictionary with structure provided by the user
example_dict = {
    'hdv_stats': torch.zeros(12, 13),
    'cav_stats': torch.zeros(12, 13),
    'all_lane_stats': torch.zeros(18, 6),
    'bottle_neck_position': torch.zeros(2),
    'self_stats': torch.zeros(1, 13)
} # ITSC version

class Hierarchical_state_rep(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(Hierarchical_state_rep, self).__init__()
        # Load the environment arguments
        env_args = yaml.load(open('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/configs/envs_cfgs/bottleneck.yaml', 'r'),
                         Loader=yaml.FullLoader)
        self.env_args = copy.deepcopy(env_args)
        train_args = yaml.load(open('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/configs/algos_cfgs/mappo.yaml', 'r'), Loader=yaml.FullLoader)
        self.train_args = copy.deepcopy(train_args)
        self.max_num_HDVs = env_args['max_num_HDVs']
        self.max_num_CAVs = env_args['max_num_CAVs']
        self.num_HDVs = env_args['num_HDVs']
        self.num_CAVs = env_args['num_CAVs']
        self.hist_length = env_args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        self.one_step_obs_dim = int(obs_dim/(self.hist_length+1))
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        self.mlp_all_lane = MLPBase(args, [18*6]) # ITSC version
        self.lane_linear1 = nn.Linear(18*6, 128)
        self.lane_linear2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()

        self.mlp_surround_lane = MLPBase(args, [3*6])
        self.mlp_bottle_neck = MLPBase(args, [2])
        self.mlp_road_structure = MLPBase(args, [10])

        self.mlp_combined = MLPBase(args, [64*2+64+10+2+2]) # self.num_HDVs*6+self.num_CAVs*6+
        self.combine_linear1 = nn.Linear(64*2+64+10+2+2, 128)
        self.combine_linear2 = nn.Linear(128, 61)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.run_step = 0

        # history feature embedding
        feature_size = 3
        GRU_hidden_size = 64
        dyn_size = 32
        self.num_env = train_args['train']['n_rollout_threads']
        num_layers = 1
        self.hist_encoder = DynamicFeatureExtractor(feature_size, GRU_hidden_size, dyn_size)
        self.last_hdv_state = torch.zeros(num_layers, self.num_env * self.num_HDVs, GRU_hidden_size, device='cuda')
        self.last_cav_state = torch.zeros(num_layers, self.num_env * self.num_CAVs, GRU_hidden_size, device='cuda')
        self.last_ego_state = torch.zeros(num_layers, self.num_env * 1, GRU_hidden_size, device='cuda')

        input_dim = 32  # Example input dimension
        hidden_dim = 16  # Example hidden dimension
        output_dim = 2  # Output dimension (e.g., x and y coordinates)
        num_layers = 2  # Number of LSTM layers
        self.pre_model = TrajectoryDecoder(input_dim, hidden_dim, output_dim, num_layers)

        nfeat_sur = 6  # Number of input features
        nhid_sur = 64
        nclass = 64  # Number of output features
        nheads = 3  # Number of attention heads
        self.gat_HDV_surround = GAT(nfeat=nfeat_sur, nhid=nhid_sur, nclass=nclass, dropout=0.6, alpha=0.2,
                                    nheads=nheads)
        self.gat_CAV_surround = GAT(nfeat=nfeat_sur, nhid=nhid_sur, nclass=nclass, dropout=0.6, alpha=0.2,
                                    nheads=nheads)
        self.gat_HDV = GAT(nfeat=15, nhid=nhid_sur, nclass=nclass, dropout=0.6, alpha=0.2,nheads=nheads)
        self.gat_CAV = GAT(nfeat=15, nhid=nhid_sur, nclass=nclass, dropout=0.6, alpha=0.2,nheads=nheads)

        self.example_extend_history = {
            'history_5': torch.zeros(self.one_step_obs_dim),
            'history_4': torch.zeros(self.one_step_obs_dim),
            'history_3': torch.zeros(self.one_step_obs_dim),
            'history_2': torch.zeros(self.one_step_obs_dim),
            'history_1': torch.zeros(self.one_step_obs_dim),
            'current': torch.zeros(self.one_step_obs_dim)
        }
        self.example_extend_info = {
            # 'bottle_neck_position': torch.zeros(2),
            'hdv_stats': torch.zeros(self.max_num_HDVs, 6),  # 0
            'cav_stats': torch.zeros(self.max_num_CAVs, 6),  # 1
            'all_lane_stats': torch.zeros(18, 6),  # 2
            'bottle_neck_position': torch.zeros(2),  # 3
            'road_structure': torch.zeros(10),  # 4
            'road_end': torch.zeros(2),  # 5
            'target': torch.zeros(2),  # 6
            'self_stats': torch.zeros(1, 13),  # 7
            'distance_bott': torch.zeros(2),  # 8
            'distance_end': torch.zeros(2),  # 9
            'executed_action': torch.zeros(1, 2),  # 10
            'generation_action': torch.zeros(1, 1),  # 11
            'surround_hdv_stats': torch.zeros(6, 6),  # 12
            'surround_cav_stats': torch.zeros(6, 6),  # 13
            'surround_lane_stats': torch.zeros(3, 6),  # 14
        }

    def reconstruct_history(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_history)
        return reconstructed['history_5'], reconstructed['history_4'], reconstructed['history_3'], \
            reconstructed['history_2'], reconstructed['history_1'], \
            reconstructed['current']

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['road_structure'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['executed_action'], reconstructed['generation_action'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['surround_lane_stats']

    def reconstruct_ITSC(self, obs):
        # obs: (n_rollout_thread, obs_dim)

        # Loop through the observations and reconstruct the dictionary
        reconstructed = self.reconstruct_obs_batch(obs, example_dict)

        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['self_stats']
    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        # hdv, cav, all_lane, bottle_neck, self_stats = self.reconstruct(obs)
        history_5, history_4, history_3, history_2, history_1, current = self.reconstruct_history(obs)

        hdv_stats_hist_5, cav_stats_hist_5, all_lane_stats_5, _, _, _, _, self_stats_hist_5, _, _, exe_action_hist_5, gen_action_hist_5, surround_hdv_hist_5, surround_cav_hist_5, surround_lane_hist_5 = self.reconstruct_info(history_5)
        hdv_stats_hist_4, cav_stats_hist_4, all_lane_stats_4, _, _, _, _, self_stats_hist_4, _, _, exe_action_hist_4, gen_action_hist_4, surround_hdv_hist_4, surround_cav_hist_4, surround_lane_hist_4 = self.reconstruct_info(
            history_4)
        hdv_stats_hist_3, cav_stats_hist_3, all_lane_stats_3, _, _, _, _, self_stats_hist_3, _, _, exe_action_hist_3, gen_action_hist_3, surround_hdv_hist_3, surround_cav_hist_3, surround_lane_hist_3 = self.reconstruct_info(
            history_3)
        hdv_stats_hist_2, cav_stats_hist_2, all_lane_stats_2, _, _, _, _, self_stats_hist_2, _, _, exe_action_hist_2, gen_action_hist_2, surround_hdv_hist_2, surround_cav_hist_2, surround_lane_hist_2 = self.reconstruct_info(
            history_2)
        hdv_stats_hist_1, cav_stats_hist_1, all_lane_stats_1, _, _, _, _, self_stats_hist_1, _, _, exe_action_hist_1, gen_action_hist_1, surround_hdv_hist_1, surround_cav_hist_1, surround_lane_hist_1 = self.reconstruct_info(
            history_1)
        hdv_stats_current, cav_stats_current, all_lane_stats_0, bottle_neck_0, road_structure_0, _, target_0, self_stats_current, _, _, exe_action_current, gen_action_current, surround_hdv_current, surround_cav_current, surround_lane_current = self.reconstruct_info(
            current)

        # The adjacency matrix needs to be a binary tensor indicating the edges
        # Here, it is assumed that each self_state is connected to all hdv_stats.
        self.hdv_adj = torch.ones(batch_size, self.max_num_HDVs+1, self.max_num_HDVs+1)
        self.hdv_adj[:, 1:, 1:] = 0  # No connections between hdv_stats themselves
        self.cav_adj = torch.ones(batch_size, self.max_num_CAVs+1, self.max_num_CAVs+1)
        self.cav_adj[:, 1:, 1:] = 0  # No connections between cav_stats themselves

        ego_stats_current = self_stats_current[:, :, :3]
        hdv_stats_current = hdv_stats_current[:, :, :3]
        cav_stats_current = cav_stats_current[:, :, :3]

        ############################## self_state & HDV ##############################
        # Concatenate the focal node to the rest of the nodes
        combined_states = torch.cat((ego_stats_current, hdv_stats_current), dim=1)
        HDV_relation = self.gat_HDV(combined_states, self.hdv_adj.to(combined_states.device))

        ############################## self_state & CAV ##############################
        combined_states = torch.cat((ego_stats_current, cav_stats_current), dim=1)
        CAV_relation = self.gat_CAV(combined_states, self.cav_adj.to(combined_states.device))

        ############################## all_lane ##############################
        all_lane_embedding = self.mlp_all_lane(all_lane_stats_0.view(all_lane_stats_0.size(0), -1))

        ############################## bottle_neck ##############################
        bottle_neck_embedding = self.mlp_bottle_neck(bottle_neck_0)

        # Concatenate all the embeddings
        combined_embedding = torch.cat((HDV_relation, CAV_relation, all_lane_embedding, bottle_neck_embedding), dim=1)
        combined_embedding = self.mlp_combined(combined_embedding)

        return combined_embedding

    def reconstruct_obs_batch(self, obs_batch, template_structure):
        device = obs_batch.device  # Get the device of obs_batch

        # Initialize the reconstructed_batch with the same structure as template_structure
        reconstructed_batch = {
            key: torch.empty((obs_batch.size(0),) + tensor.shape, device=device)
            for key, tensor in template_structure.items()
        }

        # Compute the cumulative sizes of each tensor in the template structure
        sizes = [tensor.numel() for tensor in template_structure.values()]
        cumulative_sizes = torch.cumsum(torch.tensor(sizes), dim=0)
        indices = [0] + cumulative_sizes.tolist()[:-1]

        # Split obs_batch into chunks based on the cumulative sizes
        split_tensors = torch.split(obs_batch, sizes, dim=1)

        # Assign the split tensors to the appropriate keys in the reconstructed_batch
        for key, split_tensor in zip(template_structure.keys(), split_tensors):
            reconstructed_batch[key] = split_tensor.view((-1,) + template_structure[key].shape)

        return reconstructed_batch