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
        x = F.dropout(x, self.dropout, training=self.training)
        # Concatenate the outputs of the multi-head attentions
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply the output layer; aggregate with mean as we want a graph-level output
        x = F.elu(self.out_att(x, adj))
        x = torch.mean(x, dim=1)  # Average pooling over nodes to get the graph-level output
        return x


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
        self.num_HDVs = env_args['num_HDVs']
        self.num_CAVs = env_args['num_CAVs']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        self.one_step_obs_dim = int(obs_dim/5)
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        # Assuming the number of heads is 1 for simplicity
        # nfeat = 13  # Number of input features-ITSC version
        nfeat_sur = 3*5  # Number of input features
        nfeat_int = 64  # Number of input features
        nhid = 16  # Number of hidden units per attention head
        nclass = 64  # Number of output features
        nheads = 3  # Number of attention heads

        self.gat_HDV_interaction = GAT(nfeat=nfeat_int, nhid=nhid, nclass=nclass, dropout=0.6, alpha=0.2, nheads=nheads)
        self.gat_CAV_interaction = GAT(nfeat=nfeat_int, nhid=nhid, nclass=nclass, dropout=0.6, alpha=0.2, nheads=nheads)
        self.gat_HDV_surround = GAT(nfeat=nfeat_sur, nhid=nhid, nclass=nclass, dropout=0.6, alpha=0.2, nheads=nheads)
        self.gat_CAV_surround = GAT(nfeat=nfeat_sur, nhid=nhid, nclass=nclass, dropout=0.6, alpha=0.2, nheads=nheads)

        self.mlp_all_lane = MLPBase(args, [18*6]) # ITSC version
        self.mlp_surround_lane = MLPBase(args, [3*6])
        self.mlp_bottle_neck = MLPBase(args, [2])
        self.mlp_road_structure = MLPBase(args, [10])
        self.mlp_combined = MLPBase(args, [256])

        # history feature embedding
        input_size = 13*5
        input_embedding_size = 32  #32
        encoder_size = 64  #64
        dyn_embedding_size = 64  #64

        self.input_embedding = nn.Linear(input_size, input_embedding_size)
        self.history_encoder = nn.GRU(input_embedding_size, encoder_size, 1, batch_first=True)
        self.dyn_embedding = nn.Linear(encoder_size, dyn_embedding_size)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.example_extend_history = {
            'history_4': torch.zeros(self.one_step_obs_dim),
            'history_3': torch.zeros(self.one_step_obs_dim),
            'history_2': torch.zeros(self.one_step_obs_dim),
            'history_1': torch.zeros(self.one_step_obs_dim),
            'current': torch.zeros(self.one_step_obs_dim)
        }
        self.example_extend_info = {
            # 'bottle_neck_position': torch.zeros(2),
            'hdv_stats': torch.zeros(self.num_HDVs, 13),
            'cav_stats': torch.zeros(self.num_CAVs, 13),
            'all_lane_stats': torch.zeros(18, 6),
            'bottle_neck_position': torch.zeros(2),
            'road_structure': torch.zeros(10),
            'self_stats': torch.zeros(1, 13),
            'surround_hdv_stats': torch.zeros(6, 4),
            'surround_cav_stats': torch.zeros(6, 4),
            'ego_lane_stats': torch.zeros(1, 6),
            'left_lane_stats': torch.zeros(1, 6),
            'right_lane_stats': torch.zeros(1, 6)
        }

    def reconstruct_history(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_history)
        return reconstructed['history_4'], reconstructed['history_3'], \
            reconstructed['history_2'], reconstructed['history_1'], \
            reconstructed['current']
    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['road_structure'], reconstructed['self_stats'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['ego_lane_stats'], reconstructed['left_lane_stats'], reconstructed['right_lane_stats']

    def reconstruct_ITSC(self, obs):
        # obs: (n_rollout_thread, obs_dim)

        # Loop through the observations and reconstruct the dictionary
        reconstructed = self.reconstruct_obs_batch(obs, example_dict)

        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['self_stats']
    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        # bottle_neck, self_stats, surround_hdv, surround_cav, lanes = self.reconstruct(obs)
        history_4, history_3, history_2, history_1, current = self.reconstruct_history(obs)
        hdv_stats_hist_4, cav_stats_hist_4, all_lane_stats_4, bottle_neck_4, road_structure_4, self_stats_hist_4, surround_hdv_hist_4, surround_cav_hist_4, ego_lane_hist_4, left_lane_hist_4, right_lane_hist_4 = self.reconstruct_info(history_4)
        hdv_stats_hist_3, cav_stats_hist_3, all_lane_stats_3, bottle_neck_3, road_structure_3, self_stats_hist_3, surround_hdv_hist_3, surround_cav_hist_3, ego_lane_hist_3, left_lane_hist_3, right_lane_hist_3 = self.reconstruct_info(history_3)
        hdv_stats_hist_2, cav_stats_hist_2, all_lane_stats_2, bottle_neck_2, road_structure_2, self_stats_hist_2, surround_hdv_hist_2, surround_cav_hist_2, ego_lane_hist_2, left_lane_hist_2, right_lane_hist_2 = self.reconstruct_info(history_2)
        hdv_stats_hist_1, cav_stats_hist_1, all_lane_stats_1, bottle_neck_1, road_structure_1, self_stats_hist_1, surround_hdv_hist_1, surround_cav_hist_1, ego_lane_hist_1, left_lane_hist_1, right_lane_hist_1 = self.reconstruct_info(history_1)
        hdv_stats_current, cav_stats_current, all_lane_stats_0, bottle_neck_0, road_structure_0, self_stats_current, surround_hdv_current, surround_cav_current, ego_lane_current, left_lane_current, right_lane_current = self.reconstruct_info(current)
        # The adjacency matrix needs to be a binary tensor indicating the edges
        # Here, it is assumed that each self_state is connected to all hdv_stats and cav_stats.
        self.adj_cav = torch.ones(batch_size, self.num_CAVs+1, self.num_CAVs+1)
        self.adj_cav[:, 1:, 1:] = 0  # No connections between veh themselves
        self.adj_hdv = torch.ones(batch_size, self.num_HDVs+1, self.num_HDVs+1)
        self.adj_hdv[:, 1:, 1:] = 0

        self.adj_surround = torch.ones(batch_size, 7, 7)
        self.adj_surround[:, 1:, 1:] = 0  # No connections between veh themselves

        ############################## self_state & surrounding HDVs ##############################
        # Concatenate the focal node to the rest of the nodes
        # ego_stats = torch.zeros(batch_size, 1, 3, device='cuda')
        ego_stats_hist_4 = self_stats_hist_4[:, :, :3]
        ego_stats_hist_3 = self_stats_hist_3[:, :, :3]
        ego_stats_hist_2 = self_stats_hist_2[:, :, :3]
        ego_stats_hist_1 = self_stats_hist_1[:, :, :3]
        ego_stats_current = self_stats_current[:, :, :3]

        # history feature
        history_ego_stats = torch.cat((ego_stats_hist_4, ego_stats_hist_3, ego_stats_hist_2, ego_stats_hist_1, ego_stats_current), dim=2)
        history_surround_hdv = torch.cat((surround_hdv_hist_4, surround_hdv_hist_3, surround_hdv_hist_2, surround_hdv_hist_1, surround_hdv_current), dim=2)
        history_surround_cav = torch.cat((surround_cav_hist_4, surround_cav_hist_3, surround_cav_hist_2, surround_cav_hist_1, surround_cav_current), dim=2)
        history_self_stats = torch.cat((self_stats_hist_4, self_stats_hist_3, self_stats_hist_2, self_stats_hist_1, self_stats_current), dim=2)
        history_hdv_stats = torch.cat((hdv_stats_hist_4, hdv_stats_hist_3, hdv_stats_hist_2, hdv_stats_hist_1, hdv_stats_current), dim=2)
        history_cav_stats = torch.cat((cav_stats_hist_4, cav_stats_hist_3, cav_stats_hist_2, cav_stats_hist_1, cav_stats_current), dim=2)

        hist_self_enc, _ = self.history_encoder(self.leaky_relu(self.input_embedding(history_self_stats)))
        hist_self_embedding = self.leaky_relu(hist_self_enc)
        hist_hdv_enc, _ = self.history_encoder(self.leaky_relu(self.input_embedding(history_hdv_stats)))
        hist_hdv_embedding = self.leaky_relu(hist_hdv_enc)
        hist_cav_enc, _ = self.history_encoder(self.leaky_relu(self.input_embedding(history_cav_stats)))
        hist_cav_embedding = self.leaky_relu(hist_cav_enc)

        combined_self2hdv = torch.cat((hist_self_embedding, hist_hdv_embedding), dim=1)
        self2hdv_relation = self.gat_HDV_interaction(combined_self2hdv, self.adj_hdv.to(combined_self2hdv.device))

        ############################## self_state & surrounding CAVs ##############################
        combined_self2cav = torch.cat((hist_self_embedding, hist_cav_embedding), dim=1)
        self2cav_relation = self.gat_CAV_interaction(combined_self2cav, self.adj_cav.to(combined_self2cav.device))

        ############################## ego&left&right_lanes ##############################
        hist_all_lanes = torch.cat((all_lane_stats_4, all_lane_stats_3, all_lane_stats_2, all_lane_stats_1, all_lane_stats_0), dim=2)
        all_lanes_embedding = self.mlp_all_lane(all_lane_stats_0.view(all_lane_stats_0.size(0), -1))

        ############################## bottle_neck ##############################
        # bottle_neck_embedding = self.mlp_bottle_neck(bottle_neck_0)
        road_embedding = self.mlp_road_structure(road_structure_0)

        # Concatenate all the embeddings
        combined_embedding = torch.cat((self2hdv_relation, self2cav_relation, all_lanes_embedding, road_embedding), dim=1)
        # combined_embedding = torch.cat((HDV_relation, CAV_relation, lanes_embedding, road_embedding), dim=1)
        combined_embedding = self.mlp_combined(combined_embedding)

        return combined_embedding
    def forward_ITSCversion(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        hdv, cav, all_lane, bottle_neck, self_stats = self.reconstruct(obs)
        # The adjacency matrix needs to be a binary tensor indicating the edges
        # Here, it is assumed that each self_state is connected to all hdv_stats.
        self.adj = torch.ones(batch_size, 13, 13)
        self.adj[:, 1:, 1:] = 0  # No connections between hdv_stats themselves

        ############################## self_state & HDV ##############################
        # Concatenate the focal node to the rest of the nodes
        combined_states = torch.cat((self_stats, hdv), dim=1)
        HDV_relation = self.gat_HDV(combined_states, self.adj.to(combined_states.device))

        ############################## self_state & CAV ##############################
        combined_states = torch.cat((self_stats, cav), dim=1)
        CAV_relation = self.gat_CAV(combined_states, self.adj.to(combined_states.device))

        ############################## all_lane ##############################
        all_lane_embedding = self.mlp_all_lane(all_lane.view(all_lane.size(0), -1))

        ############################## bottle_neck ##############################
        bottle_neck_embedding = self.mlp_bottle_neck(bottle_neck)

        # Concatenate all the embeddings
        combined_embedding = torch.cat((HDV_relation, CAV_relation, all_lane_embedding, bottle_neck_embedding), dim=1)
        combined_embedding = self.mlp_combined(combined_embedding)

        return combined_embedding

    # Function to reconstruct the observation tensor into the example dictionary structure
    def reconstruct_obs_batch(self, obs_batch, template_structure):
        device = obs_batch.device  # Get the device of obs_batch
        reconstructed_batch = {}
        for key, tensor in template_structure.items():
            shape = (obs_batch.size(0),) + tensor.shape  # Add the batch dimension to the shape
            reconstructed_batch[key] = torch.empty(shape, device=device)  # Initialize an empty tensor with the shape
        # Loop over each rollout thread
        for i in range(obs_batch.size(0)):
            index = 0
            # Loop over each key in the template structure
            for key, tensor in template_structure.items():
                num_elements = tensor.numel()  # Get the number of elements in the tensor
                extracted_data = obs_batch[i, index:index + num_elements].to(device)  # Extract the data for this key
                reconstructed_batch[key][i] = extracted_data.view(tensor.shape)  # Reshape and assign the data
                index += num_elements  # Move the index
        return reconstructed_batch
