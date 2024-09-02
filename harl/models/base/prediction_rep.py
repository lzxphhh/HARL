import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.mlp import MLPBase
from harl.models.base.simple_layers import CrossAttention, GAT, MLP_improve, MultiVeh_GAT, TrajectoryDecoder
from einops import rearrange, repeat
import time

class Prediction_rep(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(Prediction_rep, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        self.hist_length = args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        self.one_step_obs_dim = int(obs_dim / (self.hist_length + 1))
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.para_env = args['n_rollout_threads']

        self.gat_CAV_1a = GAT(nfeat=6, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.gat_HDV_5s = GAT(nfeat=3*5, nhid=32, nclass=64, dropout=0.1, alpha=0.2, nheads=1)

        self.mlp_surround_lane = MLPBase(args, [3 * 6])
        self.mlp_local_combined = MLPBase(args, [10+64*3])

        # prediction encoder & decoder
        self.gat_all_vehs = MultiVeh_GAT(nfeat=20, nhid=32, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.mlp_enc_combined = MLPBase(args, [10+64+64])
        self.Pre_decoder = TrajectoryDecoder(input_dim=64, hidden_dim=64, output_dim=3)
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids

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
            'hdv_stats': torch.zeros(self.max_num_HDVs, 7),  # 0
            'cav_stats': torch.zeros(self.max_num_CAVs, 7),  # 1
            'all_lane_stats': torch.zeros(18, 6),            # 2
            'bottle_neck_position': torch.zeros(2),          # 3
            'road_structure': torch.zeros(10),               # 4
            'road_end': torch.zeros(2),                      # 5
            'target': torch.zeros(2),                        # 6
            'self_stats': torch.zeros(1, 13),                # 7
            'distance_bott': torch.zeros(2),                 # 8
            'distance_end': torch.zeros(2),                  # 9
            'network_action': torch.zeros(1, 3),             # 10
            'actual_action': torch.zeros(1, 3),              # 11
            'surround_hdv_stats': torch.zeros(6, 6),         # 12
            'surround_cav_stats': torch.zeros(6, 6),         # 13
            'expand_surround_stats': torch.zeros(10, 20),    # 14
            'surround_relation_graph': torch.zeros(10, 10),  # 15
            'surround_IDs': torch.zeros(10),                 # 16
            'surround_lane_stats': torch.zeros(3, 6),        # 17
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
            reconstructed['network_action'], reconstructed['actual_action'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['expand_surround_stats'], reconstructed['surround_relation_graph'], \
            reconstructed['surround_IDs'], reconstructed['surround_lane_stats']

    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        start = time.time()
        history_5, history_4, history_3, history_2, history_1, current = self.reconstruct_history(obs)

        # info_hist_5 = self.reconstruct_info(history_5)
        info_hist_4 = self.reconstruct_info(history_4)
        info_hist_3 = self.reconstruct_info(history_3)
        info_hist_2 = self.reconstruct_info(history_2)
        info_hist_1 = self.reconstruct_info(history_1)
        info_current = self.reconstruct_info(current)

        self.adj_surround_cav = torch.ones(batch_size, 6+1, 6+1)
        self.adj_surround_cav[:, 1:, 1:] = 0
        self.adj_surround_hdv = torch.ones(batch_size, 6+1, 6+1)
        self.adj_surround_hdv[:, 1:, 1:] = 0

        ################################################## trajectory-aware interaction encoder ######################################################
        # bottle_neck_0, distance_bott_0, road_end_0, distance_end_0, target_0
        # 03-'bottle_neck_position': torch.zeros(2),
        # 08-'distance_bott': torch.zeros(2),
        # 05-'road_end': torch.zeros(2),
        # 09-'distance_end': torch.zeros(2),
        # 06-'target': torch.zeros(2),
        local_road_info = torch.cat((info_current[8], info_current[5][:, :1], info_current[9], info_current[6]), dim=1)
        local_surround_cav_motion = info_current[13]
        local_ego_motion = torch.cat((info_current[7][:, :, :3], info_current[11]), dim=2)
        local_surround_lane_stats = info_current[17]

        ego_hist = torch.cat((info_hist_4[7][:, :, :3], info_hist_3[7][:, :, :3], info_hist_2[7][:, :, :3],
                              info_hist_1[7][:, :, :3], info_current[7][:, :, :3]), dim=2)
        hdv_hist = torch.cat((info_hist_4[12][:, :, :3], info_hist_3[12][:, :, :3], info_hist_2[12][:, :, :3],
                              info_hist_1[12][:, :, :3], info_current[12][:, :, :3]), dim=2)
        combined_ego2hdv_hist = torch.cat((ego_hist, hdv_hist), dim=1)
        ego2hdv_relation_hist = self.gat_HDV_5s(combined_ego2hdv_hist, self.adj_surround_hdv.to(combined_ego2hdv_hist.device))
        combined_ego2cav_current_a = torch.cat((local_ego_motion, local_surround_cav_motion), dim=1)
        ego2cav_relation_current_a = self.gat_CAV_1a(combined_ego2cav_current_a, self.adj_surround_cav.to(combined_ego2cav_current_a.device))
        local_cav_relation = ego2cav_relation_current_a
        local_hdv_relation = ego2hdv_relation_hist
        local_lane_embedding = self.mlp_surround_lane(local_surround_lane_stats.view(local_surround_lane_stats.size(0), -1))
        exe_action = info_current[11].view(info_current[11].size(0), -1)
        local_combined_embedding = torch.cat((local_road_info, exe_action, local_cav_relation, local_hdv_relation, local_lane_embedding), dim=1)
        local_combined_embedding = self.mlp_local_combined(local_combined_embedding)

        ################################################## prediction encoder & decoder ######################################################

        self.adj_surround_vehs = info_current[15].to(obs.device)
        enc_veh_hist = info_current[14].to(obs.device)
        enc_veh_relation = self.gat_all_vehs(enc_veh_hist, self.adj_surround_vehs)
        other_info = torch.cat((local_road_info, exe_action, local_lane_embedding), dim=1)
        other_info_expanded = other_info.unsqueeze(1).repeat(1, 10, 1)
        enc_combined_embedding = torch.cat((other_info_expanded, enc_veh_relation), dim=2)
        enc_combined_embedding = self.mlp_enc_combined(enc_combined_embedding)

        # Predict future states
        future_states = self.Pre_decoder(enc_combined_embedding.view(batch_size * 10, 64), future_steps=3)

        # Reshape to [batch_size, num_veh, future_steps, state_dim]
        future_states = future_states.view(batch_size, 10, 3, 3)

        # Change the relative state to absolute state
        ego_position_x = info_current[7][:, :, 0] * 700
        ego_position_y = info_current[7][:, :, 1]
        ego_speed = info_current[7][:, :, 2] * 15

        # Broadcasting and vectorized operations
        absolute_future_states = torch.zeros(batch_size, 10, 3, 3, device=obs.device)
        absolute_future_states[:, :, :, 0] = future_states[:, :, :, 0] + ego_position_x[:, 0, None, None]
        absolute_future_states[:, :, :, 1] = future_states[:, :, :, 1] + ego_position_y[:, 0, None, None]
        absolute_future_states[:, :, :, 2] = ego_speed[:, 0, None, None] - future_states[:, :, :, 2]

        # Initialize prediction_output and prediction_groundtruth
        prediction_output = {veh_id: torch.zeros(batch_size, 3, 3, device=obs.device) for veh_id in self.veh_ids}
        prediction_groundtruth = {veh_id: torch.zeros(batch_size, 3, device=obs.device) for veh_id in self.veh_ids}

        # Process ego_id
        if batch_size == self.para_env:
            ego_id = self.CAV_ids[int(info_current[16][0, 0] - 200)] if info_current[16][0, 0] >= 200 else 0
        else:
            ego_id = [
                self.CAV_ids[int(info_current[16][i, 0] - 200)] if info_current[16][i, 0] >= 200 else None
                for i in range(batch_size)
            ]

        # Process prediction_output using vectorized operations
        cav_mask = (info_current[16] >= 200) & (info_current[16] < 300)
        hdv_mask = (info_current[16] >= 100) & (info_current[16] < 200)

        for i in range(10):
            cav_indices = cav_mask[:, i].nonzero(as_tuple=True)[0]
            hdv_indices = hdv_mask[:, i].nonzero(as_tuple=True)[0]

            if cav_indices.numel() > 0:
                cav_ids = info_current[16][cav_indices, i] - 200
                for cav_id in cav_ids.unique():
                    cav_key = self.CAV_ids[int(cav_id)]
                    selected_indices = cav_indices[info_current[16][cav_indices, i] == (200 + cav_id)]
                    prediction_output[cav_key][selected_indices, :, :] = absolute_future_states[selected_indices, i, :,
                                                                         :]

            if hdv_indices.numel() > 0:
                hdv_ids = info_current[16][hdv_indices, i] - 100
                for hdv_id in hdv_ids.unique():
                    hdv_key = self.HDV_ids[int(hdv_id)]
                    selected_indices = hdv_indices[info_current[16][hdv_indices, i] == (100 + hdv_id)]
                    prediction_output[hdv_key][selected_indices, :, :] = absolute_future_states[selected_indices, i, :,
                                                                         :]

        # Process prediction_groundtruth using vectorized operations
        for veh_id in self.veh_ids:
            idx = int(veh_id[4:])
            if veh_id.startswith('CAV'):
                current_state = info_current[1][:, idx, :3]
            else:
                current_state = info_current[0][:, idx, :3]

            current_state[:, 0] = current_state[:, 0] * 700
            current_state[:, 2] = current_state[:, 2] * 15
            prediction_groundtruth[veh_id] = current_state
        end = time.time()
        print('time:', end - start)
        return local_combined_embedding, prediction_output, ego_id, prediction_groundtruth

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