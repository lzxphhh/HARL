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

class Prediction_rep(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(Prediction_rep, self).__init__()
        # Load the environment arguments
        env_args = yaml.load(
            open('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/configs/envs_cfgs/bottleneck.yaml', 'r'),
            Loader=yaml.FullLoader)
        self.env_args = copy.deepcopy(env_args)
        train_args = yaml.load(
            open('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/configs/algos_cfgs/mappo.yaml', 'r'),
            Loader=yaml.FullLoader)
        self.train_args = copy.deepcopy(train_args)
        self.max_num_HDVs = env_args['max_num_HDVs']
        self.max_num_CAVs = env_args['max_num_CAVs']
        self.num_HDVs = env_args['num_HDVs']
        self.num_CAVs = env_args['num_CAVs']
        self.hist_length = env_args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        self.one_step_obs_dim = int(obs_dim / (self.hist_length + 1))
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        self.gat_HDV_1s = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.6, alpha=0.2, nheads=2)
        self.gat_CAV_1s = GAT(nfeat=6, nhid=16, nclass=64, dropout=0.6, alpha=0.2, nheads=2)
        self.gat_HDV_5s = GAT(nfeat=32, nhid=64, nclass=64, dropout=0.6, alpha=0.2, nheads=2)

        self.mlp_all_lane = MLPBase(args, [18*6]) # ITSC version
        self.mlp_combined = MLPBase(args, [64*3+10+3])

        # history feature embedding
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.self_embedding = nn.Linear(3*5, 64)
        self.dyn_embedding = nn.Linear(64, 32)

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
            'hdv_stats': torch.zeros(self.max_num_HDVs, 6),
            'cav_stats': torch.zeros(self.max_num_CAVs, 6),
            'all_lane_stats': torch.zeros(18, 6),
            'bottle_neck_position': torch.zeros(2),
            # 'road_structure': torch.zeros(10),
            'road_end': torch.zeros(2),
            'target': torch.zeros(2),
            'self_stats': torch.zeros(1, 13),
            'distance_bott': torch.zeros(2),
            'distance_end': torch.zeros(2),
            'executed_action': torch.zeros(1, 2),
            'generation_action': torch.zeros(1, 1),
            'surround_hdv_stats': torch.zeros(6, 6),
            'surround_cav_stats': torch.zeros(6, 6),
            'ego_lane_stats': torch.zeros(1, 6),
            'left_lane_stats': torch.zeros(1, 6),
            'right_lane_stats': torch.zeros(1, 6)
        }

    def reconstruct_history(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_history)
        return reconstructed['history_5'], reconstructed['history_4'], reconstructed['history_3'], \
            reconstructed['history_2'], reconstructed['history_1'], \
            reconstructed['current']

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['executed_action'], reconstructed['generation_action'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['ego_lane_stats'], reconstructed['left_lane_stats'], reconstructed['right_lane_stats']

    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        history_5, history_4, history_3, history_2, history_1, current = self.reconstruct_history(obs)

        hdv_stats_hist_5, cav_stats_hist_5, all_lane_stats_5, _, _, _, self_stats_hist_5, _, _, exe_action_hist_5, gen_action_hist_5, surround_hdv_hist_5, surround_cav_hist_5, ego_lane_hist_5, left_lane_hist_5, right_lane_hist_5 = self.reconstruct_info(
            history_5)
        hdv_stats_hist_4, cav_stats_hist_4, all_lane_stats_4, _, _, _, self_stats_hist_4, _, _, exe_action_hist_4, gen_action_hist_4, surround_hdv_hist_4, surround_cav_hist_4, ego_lane_hist_4, left_lane_hist_4, right_lane_hist_4 = self.reconstruct_info(
            history_4)
        hdv_stats_hist_3, cav_stats_hist_3, all_lane_stats_3, _, _, _, self_stats_hist_3, _, _, exe_action_hist_3, gen_action_hist_3, surround_hdv_hist_3, surround_cav_hist_3, ego_lane_hist_3, left_lane_hist_3, right_lane_hist_3 = self.reconstruct_info(
            history_3)
        hdv_stats_hist_2, cav_stats_hist_2, all_lane_stats_2, _, _, _, self_stats_hist_2, _, _, exe_action_hist_2, gen_action_hist_2, surround_hdv_hist_2, surround_cav_hist_2, ego_lane_hist_2, left_lane_hist_2, right_lane_hist_2 = self.reconstruct_info(
            history_2)
        hdv_stats_hist_1, cav_stats_hist_1, all_lane_stats_1, _, _, _, self_stats_hist_1, _, _, exe_action_hist_1, gen_action_hist_1, surround_hdv_hist_1, surround_cav_hist_1, ego_lane_hist_1, left_lane_hist_1, right_lane_hist_1 = self.reconstruct_info(
            history_1)
        hdv_stats_current, cav_stats_current, all_lane_stats_0, bottle_neck_0, road_end_0, target_0, self_stats_current, distance_bott_0, distance_end_0, exe_action_current, gen_action_current, surround_hdv_current, surround_cav_current, ego_lane_current, left_lane_current, right_lane_current = self.reconstruct_info(
            current)
        # The adjacency matrix needs to be a binary tensor indicating the edges
        # Here, it is assumed that each self_state is connected to all hdv_stats and cav_stats.
        self.adj_cav = torch.ones(batch_size, self.max_num_CAVs+1, self.max_num_CAVs+1)
        self.adj_cav[:, 1:, 1:] = 0  # No connections between veh themselves
        self.adj_hdv = torch.ones(batch_size, self.max_num_HDVs+1, self.max_num_HDVs+1)
        self.adj_hdv[:, 1:, 1:] = 0

        ############################## self_state & surrounding HDVs ##############################
        # Concatenate the focal node to the rest of the nodes
        ego_stats_hist_4 = self_stats_hist_4[:, :, :3]
        ego_stats_hist_3 = self_stats_hist_3[:, :, :3]
        ego_stats_hist_2 = self_stats_hist_2[:, :, :3]
        ego_stats_hist_1 = self_stats_hist_1[:, :, :3]
        ego_stats_current = self_stats_current[:, :, :3]
        hdv_motion_hist_4 = hdv_stats_hist_4[:, :, :3]
        hdv_motion_hist_3 = hdv_stats_hist_3[:, :, :3]
        hdv_motion_hist_2 = hdv_stats_hist_2[:, :, :3]
        hdv_motion_hist_1 = hdv_stats_hist_1[:, :, :3]
        hdv_motion_current = hdv_stats_current[:, :, :3]
        # cav_motion_current = cav_stats_current[:, :, :3]

        ego_stats_action = torch.cat((self_stats_current[:, :, :3], gen_action_current, exe_action_current), dim=2)
        cav_motion_current = cav_stats_current
        # self historical embedding
        history_ego_stats = torch.cat((ego_stats_hist_4, ego_stats_hist_3, ego_stats_hist_2, ego_stats_hist_1, ego_stats_current), dim=2)
        # _, hist_ego_embedding = self.self_encoder(self.leaky_relu(self.self_embedding(history_ego_stats)))
        hist_ego_embedding = self.dyn_embedding(self.leaky_relu(self.self_embedding(history_ego_stats)))
        # HDV historical embedding
        history_hdv_stats = torch.cat((hdv_motion_hist_4, hdv_motion_hist_3, hdv_motion_hist_2, hdv_motion_hist_1, hdv_motion_current), dim=2)
        hist_hdv_embedding = self.dyn_embedding(self.leaky_relu(self.self_embedding(history_hdv_stats)))
        # historical state - GAT
        combined_ego2hdv_hist = torch.cat((hist_ego_embedding, hist_hdv_embedding), dim=1)
        ego2hdv_relation_hist = self.gat_HDV_5s(combined_ego2hdv_hist, self.adj_hdv.to(combined_ego2hdv_hist.device))

        # hist_ego_embedding = hist_ego_embedding.squeeze(1)
        # current state - GAT
        combined_ego2cav_current = torch.cat((ego_stats_action, cav_motion_current), dim=1)
        ego2cav_relation_current = self.gat_CAV_1s(combined_ego2cav_current, self.adj_cav.to(combined_ego2cav_current.device))

        # GAT feature
        hdv_relation = ego2hdv_relation_hist
        cav_relation = ego2cav_relation_current

        # hist_hdv_enc = self.leaky_relu(self.input_embedding(hist_hdv_relation))
        # hist_hdv_embedding = self.dyn_embedding(self.leaky_relu(hist_hdv_enc))
        # hist_cav_enc = self.leaky_relu(self.input_embedding(hist_cav_relation))
        # hist_cav_embedding = self.dyn_embedding(self.leaky_relu(hist_cav_enc))

        # hist_hdv_embedding = self.dyn_embedding(self.leaky_relu(self.input_embedding(hist_hdv_relation)))
        # hist_cav_embedding = self.dyn_embedding(self.leaky_relu(self.input_embedding(hist_cav_relation)))

        ############################## ego&left&right_lanes ##############################
        all_lanes_embedding = self.mlp_all_lane(all_lane_stats_0.view(all_lane_stats_0.size(0), -1))

        ############################## bottle_neck ##############################
        road_embedding = torch.cat((bottle_neck_0, distance_bott_0, road_end_0, distance_end_0, target_0), dim=1)

        # Concatenate all the embeddings
        exe_action = exe_action_current.view(exe_action_current.size(0), -1)
        gen_action = gen_action_current.view(gen_action_current.size(0), -1)
        combined_embedding = torch.cat((hdv_relation, cav_relation, all_lanes_embedding, road_embedding, exe_action, gen_action), dim=1)
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