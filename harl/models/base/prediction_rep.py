import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.mlp import MLPBase
from harl.models.base.simple_layers import GAT

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

        self.gat_HDV_1s = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.6, alpha=0.2, nheads=3)
        self.gat_CAV_1s = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.6, alpha=0.2, nheads=3)
        self.gat_CAV_1a = GAT(nfeat=6, nhid=64, nclass=64, dropout=0.6, alpha=0.2, nheads=3)

        self.gat_HDV_5s = GAT(nfeat=64, nhid=64, nclass=64, dropout=0.6, alpha=0.2, nheads=3)

        self.mlp_all_lane = MLPBase(args, [18*6]) # ITSC version
        self.mlp_combined = MLPBase(args, [64*4+10+3])

        # history feature embedding
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.self_embedding = nn.Linear(3*5, 128)
        self.dyn_embedding = nn.Linear(128, 64)

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
            'all_lane_stats': torch.zeros(18, 6),            # 2
            'bottle_neck_position': torch.zeros(2),          # 3
            # 'road_structure': torch.zeros(10),
            'road_end': torch.zeros(2),                      # 4
            'target': torch.zeros(2),                        # 5
            'self_stats': torch.zeros(1, 13),                # 6
            'distance_bott': torch.zeros(2),                 # 7
            'distance_end': torch.zeros(2),                  # 8
            'executed_action': torch.zeros(1, 2),            # 9
            'generation_action': torch.zeros(1, 1),          # 10
            'surround_hdv_stats': torch.zeros(6, 6),         # 11
            'surround_cav_stats': torch.zeros(6, 6),         # 12
            'ego_lane_stats': torch.zeros(1, 6),             # 13
            'left_lane_stats': torch.zeros(1, 6),            # 14
            'right_lane_stats': torch.zeros(1, 6)            # 15
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

        # info_hist_5 = self.reconstruct_info(history_5)
        info_hist_4 = self.reconstruct_info(history_4)
        info_hist_3 = self.reconstruct_info(history_3)
        info_hist_2 = self.reconstruct_info(history_2)
        info_hist_1 = self.reconstruct_info(history_1)
        info_current = self.reconstruct_info(current)

        self.adj_cav = torch.ones(batch_size, self.max_num_CAVs + 1, self.max_num_CAVs + 1)
        self.adj_cav[:, 1:, 1:] = 0
        self.adj_hdv = torch.ones(batch_size, self.max_num_HDVs + 1, self.max_num_HDVs + 1)
        self.adj_hdv[:, 1:, 1:] = 0

        ############################## self_state & surrounding HDVs ##############################
        # Concatenate the focal node to the rest of the nodes
        ego_stats_hist = torch.cat((info_hist_4[6][:, :, :3], info_hist_3[6][:, :, :3], info_hist_2[6][:, :, :3],
                          info_hist_1[6][:, :, :3], info_current[6][:, :, :3]), dim=2)
        # hdv_motion_hist = torch.cat((info_hist_4[0][:, :, :3], info_hist_3[0][:, :, :3], info_hist_2[0][:, :, :3],
        #                     info_hist_1[0][:, :, :3], info_current[0][:, :, :3]), dim=2)

        ego_motion_current = torch.cat((info_current[6][:, :, :3], info_current[10], info_current[9]), dim=2)
        cav_motion_current = info_current[1]
        ego_stats_current = info_current[6][:, :, :3]
        cav_stats_current = info_current[1][:, :, :3]
        hdv_stats_current = info_current[0][:, :, :3]
        # self historical embedding
        hist_ego_embedding = self.leaky_relu(self.self_embedding(ego_stats_hist))
        hist_ego_feature = hist_ego_embedding.squeeze(1)
        hist_ego_feature = self.dyn_embedding(self.leaky_relu(hist_ego_feature))

        # HDV historical embedding
        # hist_hdv_embedding = self.leaky_relu(self.dyn_embedding(self.leaky_relu(self.self_embedding(hdv_motion_hist))))
        # historical state - GAT
        # combined_ego2hdv_hist = torch.cat((hist_ego_embedding, hist_hdv_embedding), dim=1)
        # ego2hdv_relation_hist = self.gat_HDV_5s(combined_ego2hdv_hist, self.adj_hdv.to(combined_ego2hdv_hist.device))

        # hist_ego_embedding = hist_ego_embedding.squeeze(1)
        # current state - GAT
        combined_ego2hdv_current = torch.cat((ego_stats_current, hdv_stats_current), dim=1)
        ego2hdv_relation_current = self.gat_HDV_1s(combined_ego2hdv_current, self.adj_hdv.to(combined_ego2hdv_current.device))
        combined_ego2cav_current = torch.cat((ego_stats_current, cav_stats_current), dim=1)
        ego2cav_relation_current = self.gat_CAV_1s(combined_ego2cav_current, self.adj_cav.to(combined_ego2cav_current.device))
        # combined_ego2cav_current_a = torch.cat((ego_motion_current, cav_motion_current), dim=1)
        # ego2cav_relation_current_a = self.gat_CAV_1a(combined_ego2cav_current_a, self.adj_cav.to(combined_ego2cav_current_a.device))

        # GAT feature
        hdv_relation = ego2hdv_relation_current
        cav_relation = ego2cav_relation_current
        ego_feature = hist_ego_feature

        # hist_hdv_enc = self.leaky_relu(self.input_embedding(hist_hdv_relation))
        # hist_hdv_embedding = self.dyn_embedding(self.leaky_relu(hist_hdv_enc))
        # hist_cav_enc = self.leaky_relu(self.input_embedding(hist_cav_relation))
        # hist_cav_embedding = self.dyn_embedding(self.leaky_relu(hist_cav_enc))

        # hist_hdv_embedding = self.dyn_embedding(self.leaky_relu(self.input_embedding(hist_hdv_relation)))
        # hist_cav_embedding = self.dyn_embedding(self.leaky_relu(self.input_embedding(hist_cav_relation)))

        ############################## ego&left&right_lanes ##############################
        all_lane_stats = info_current[2]
        all_lanes_embedding = self.mlp_all_lane(all_lane_stats.view(all_lane_stats.size(0), -1))

        ############################## bottle_neck ##############################
        # bottle_neck_0, distance_bott_0, road_end_0, distance_end_0, target_0
        # 03-'bottle_neck_position': torch.zeros(2),
        # 07-'distance_bott': torch.zeros(2),
        # 04-'road_end': torch.zeros(2),
        # 08-'distance_end': torch.zeros(2),
        # 05-'target': torch.zeros(2),
        road_embedding = torch.cat((info_current[3], info_current[7], info_current[4], info_current[8], info_current[5]), dim=1)

        # Concatenate all the embeddings
        exe_action = info_current[9].view(info_current[9].size(0), -1)
        gen_action = info_current[10].view(info_current[10].size(0), -1)
        combined_embedding = torch.cat((ego_feature, hdv_relation, cav_relation, all_lanes_embedding, road_embedding, gen_action, exe_action), dim=1)
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