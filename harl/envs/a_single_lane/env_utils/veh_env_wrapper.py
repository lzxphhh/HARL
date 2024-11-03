import time
from typing import Any, SupportsFloat, Tuple, Dict, List
import random
import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from loguru import logger
import math
import csv
import os

from .generate_scene_straight import generate_scenario
from .wrapper_utils import (
    analyze_traffic,
    compute_ego_vehicle_features,
    compute_base_ego_vehicle_features,
    compute_hierarchical_ego_vehicle_features,
    compute_centralized_vehicle_features,
    compute_centralized_vehicle_features_hierarchical_version,
    check_collisions_based_pos,
    check_collisions
)
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

# 获得全局路径
path_convert = get_abs_path(__file__)
# 设置日志 -- tshub自带的给环境的
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level='ERROR')

GAP_THRESHOLD = 1.5
WARN_GAP_THRESHOLD = 3.0


class VehEnvWrapper(gym.Wrapper):
    """Vehicle Env Wrapper for vehicle info
    """

    def __init__(self, env: Env,
                 name_scenario: str,  # 场景的名称
                 max_num_CAVs: int,  # 最大的 CAV 数量
                 max_num_HDVs: int,  # 最大的 HDV 数量
                 CAV_penetration: float,  # HDV 的数量
                 num_CAVs: int,  # CAV 的数量
                 num_HDVs: int,  # HDV 的数量
                 lane_max_num_vehs: int, # 每条车道车辆最大数量
                 edge_ids: List[str],  # 路网中所有路段的 id
                 edge_lane_num: Dict[str, int],  # 每个 edge 的车道数
                 calc_features_lane_ids: List[str],  # 需要统计特征的 lane id
                 node_positions: Dict[str, float],  # node的坐标
                 filepath: str,  # 日志文件的路径
                 delta_t: int,  # 动作之间的间隔时间
                 warmup_steps: int,  # reset 的时候仿真的步数, 确保 ego vehicle 可以全部出现
                 use_gui: bool,  # 是否使用 GUI
                 aggressive: float,  # aggressive 的概率
                 cautious: float,  # cautious 的概率
                 normal: float,  # normal 的概率
                 strategy: str, # MARL 的策略- feature extraction
                 use_hist_info: bool,  # 是否使用历史信息
                 hist_length: int,  # 历史信息的长度
                 ) -> None:
        super().__init__(env)
        self.name_scenario = name_scenario
        self.max_num_CAVs = max_num_CAVs
        self.max_num_HDVs = max_num_HDVs
        self.CAV_penetration = CAV_penetration
        self.num_CAVs = num_CAVs
        self.num_HDVs = num_HDVs
        self.lane_max_num_vehs = lane_max_num_vehs

        # random generate self.num_CAVs CAVs from range (4, 6)
        # self.num_CAVs = random.choice([4, 5, 6])
        # self.CAV_penetration = random.choice([0.4, 0.5, 0.6])

        self.edge_ids = edge_ids
        self.edge_lane_num = edge_lane_num
        self.calc_features_lane_ids = calc_features_lane_ids  # 需要统计特征的 lane id
        self.node_positions = node_positions  # bottle neck 的坐标
        self.warmup_steps = warmup_steps
        self.use_gui = use_gui
        self.delta_t = delta_t
        self.max_num_seconds = self.num_seconds
        self.aggressive = aggressive
        self.cautious = cautious
        self.normal = normal
        self.strategy = strategy
        self.use_hist_info = use_hist_info
        self.hist_length = hist_length

        self.ego_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.veh_ids = self.ego_ids + [f'HDV_{i}' for i in range(self.num_HDVs)] + ['Leader']

        # 记录当前速度
        self.current_speed = {key: 0 for key in self.ego_ids}
        # 记录当前的lane
        self.current_lane = {key: 0 for key in self.ego_ids}

        self.action_pointer = {
            0: 0,  # 保持车速
            1: 1,  # 速度+1
            2: 2,  # 速度+2
            3: 3,  # 速度+3
            4: -1,  # 速度-1
            5: -2,  # 速度-2
            6: -3,  # 速度-3
        }

        self.congestion_level = 0  # 初始是不堵车的
        self.vehicles_info = {}  # 记录仿真内车辆的 (初始 lane index, travel time)
        self.agent_mask = {ego_id: True for ego_id in self.ego_ids}  # RL控制的车辆是否在路网上

        self.total_timesteps = 0  # 记录总的时间步数
        # #######
        # Writer
        # #######
        logger.info(f'RL: Log Path, {filepath}')
        self.t_start = time.time()
        # self.results_writer = ResultsWriter(
        #     filepath,
        #     header={"t_start": self.t_start},
        # )
        self.rewards_writer = list()
        if self.strategy == 'base':
            # road_structure: four nodes' positions - 4*2 (Ring road) / one node position - 2 (Straight road)
            # next_node_pos: pos_x, pos_y , dist_next_node - 3
            # self_stats: type, pos_x, pos_y, speed, acceleration, heading, (lane_one_hot) - 6/10
            # vehicles_state (front + rear): type, pos_x, pos_y, speed, acceleration, heading - 2*6
            # ego_lane_statistics: num_veh, lane_length, density, mean_speed, mean_acceleration, CAV_penetration - 6 (Ring road)
            # flow_statistics: num_veh, mean_speed, mean_acceleration, num_CAV, CAV_penetration - 5 (Straight road)
            self.self_obs_size = 2 + 1 + 6 + 2 * 6 + 5  # (straight lane)
            # road_structure: four nodes' positions - 4*2 (Ring road) / one node position - 2 (Straight road)
            # all_CAVs_state: pos_x, pos_y, speed, acceleration, heading - max_num_CAVs*5
            # all_lane_statistics: num_veh, lane_length, density, mean_speed, mean_acceleration, CAV_penetration - 4*6 (Ring road)
            # flow_statistics: num_veh, mean_speed, mean_acceleration, num_CAV, CAV_penetration - 5 (Straight road)
            self.shared_obs_size = 2 + self.max_num_CAVs * 5 + 5  # + 4 * 6 (Straight road)
        elif self.strategy == 'iMARL':
            # road_structure: four nodes' positions - 4*2 (Ring road) / one node position - 2 (Straight road)
            # next_node_pos: pos_x, pos_y , dist_next_node - 3
            # self_stats: pos_x, pos_y, speed, acceleration, heading, lane_one_hot, last_actor_action - 12 + 2
            # self_hist: pos_x, pos_y, speed, acceleration, heading - 1+5*self.hist_length
            # vehicles_2_state (front + rear): type, pos_x, pos_y, speed, acceleration, heading - 2*(1+5*self.hist_length)
            # vehicles_4_state (2 front + 2 rear): type, pos_x, pos_y, speed, acceleration, heading - 4*(1+5*self.hist_length)
            # vehicles_6_state (3 front + 3 rear): type, pos_x, pos_y, speed, acceleration, heading - 6*(1+5*self.hist_length)
            # ego_lane_statistics: start, end, num_veh, lane_length, density, mean_speed, mean_acceleration, CAV_penetration - 10
            # next_lane_statistics: start, end, num_veh, lane_length, density, mean_speed, mean_acceleration, CAV_penetration - 10
            self.self_obs_size = (2 + 1 + 8 + (1+5*self.hist_length)
                                  + (2 * (1+5*self.hist_length))
                                  + (4 * (1+5*self.hist_length))
                                  + (6 * (1+5*self.hist_length))
                                  + 5)
                                  # + 2 * 10)
            # road_structure: four nodes' positions - 4*2 (Ring road) / one node position - 2 (Straight road)
            # all_HDVs_state: pos_x, pos_y, speed, acceleration, heading - max_num_HDVs*9
            # all_CAVs_state: pos_x, pos_y, speed, acceleration, heading - max_num_CAVs*5
            # all_lane_statistics - start, end, num_veh, lane_length, density, mean_speed, mean_acceleration, CAV_penetration - 4*10 (Ring road)
            # all_lane_distribution: each lane-veh_info: type, pos_x, pos_y, speed, acceleration, heading - 4 * (max_num_vehs_lane)*6 (Ring road)
            # flow_statistics: num_veh, mean_speed, mean_acceleration, num_CAV, CAV_penetration - 5*hist_length (Straight road)
            # veh_distribution: veh_info: type, pos_x, pos_y, speed, acceleration, heading - max_num_vehs_lane*6*hist_length (Straight road)
            self.shared_obs_size = 2 + (self.max_num_HDVs+1)*5 + self.max_num_CAVs*5 + 5*self.hist_length + self.lane_max_num_vehs*6*self.hist_length
        self.vehicles_hist = {}
        # self.lanes_hist = {}
        self.flow_hist = {}
        self.veh_distribution_hist = {}
        if self.use_hist_info:
            self.obs_size = self.self_obs_size
            for i in range(self.hist_length):
                self.vehicles_hist[f'hist_{i+1}'] = {veh_id: [0.0]*5 for veh_id in self.veh_ids}
                # self.lanes_hist[f'hist_{i+1}'] = {lane_id: np.zeros(26) for lane_id in self.calc_features_lane_ids}
                self.flow_hist[f'hist_{i+1}'] = [0.0]*5
                self.veh_distribution_hist[f'hist_{i+1}'] = np.zeros(self.lane_max_num_vehs * 6)
        else:
            self.obs_size = self.self_obs_size
        self.surround_vehicle_2 = {ego_id: {} for ego_id in self.ego_ids}
        self.surround_vehicle_4 = {ego_id: {} for ego_id in self.ego_ids}
        self.surround_vehicle_6 = {ego_id: {} for ego_id in self.ego_ids}
        self.required_surroundings = ['front', 'back']
        self.TTC_assessment = {ego_id: {key: 100 for key in self.required_surroundings} for ego_id in self.ego_ids}
        self.change_action_mark = {ego_id: [] for ego_id in self.ego_ids}
        self.safety_before = {ego_id: [] for ego_id in self.ego_ids}
        self.safety_after = {ego_id: [] for ego_id in self.ego_ids}

        self.actor_action = {ego_id: [] for ego_id in self.ego_ids}
        self.lowercontroller_action = {ego_id: [] for ego_id in self.ego_ids}
        # now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # self.save_csv_dir = "/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/HIAHR_DIU_MAPPO/" + "action_improve/" + now_time
        # os.makedirs(self.save_csv_dir)

    # #####################
    # Obs and Action Space
    # #####################
    @property
    def action_space(self):
        """定义连续的动作空间，加速度范围为 [-3, 3]"""
        return {_ego_id: gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32) for _ego_id in
                self.ego_ids}

    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,)
        )
        return {_ego_id: obs_space for _ego_id in self.ego_ids}

    @property
    def share_observation_space(self):
        share_obs_space = gym.spaces.Box(
            low = -np.inf, high = np.inf, shape = (self.shared_obs_size,)
        )
        return {_ego_id: share_obs_space for _ego_id in self.ego_ids}

    # ##################
    # Tools for observations
    # ##################
    def append_surrounding(self, state):
        surrounding_vehicles_2 = {}
        sorrounding_vehicles_4 = {}
        sorrounding_vehicles_6 = {}
        """
                    ^ y (+)
                    |
                    |
        x(-) <------------> x (+)
                    | 
                    |
                    v y (-)

        [surround_vehicle_id, relative x, relative y, relative speed]
        """
        for vehicle_id in state['vehicle'].keys():
            # 对于所有RL控制的车辆
            if vehicle_id in self.ego_ids:
                if self.use_gui:
                    import traci as traci
                else:
                    import libsumo as traci
                surrounding_vehicle_2 = {}
                sorrounding_vehicle_4 = {}
                surrounding_vehicle_6 = {}
                ego_speed = state['vehicle'][vehicle_id]['speed']
                ego_long_pos = state['vehicle'][vehicle_id]['position'][0]
                ego_accel = state['vehicle'][vehicle_id]['acceleration']
                ego_heading = state['vehicle'][vehicle_id]['heading']

                # 在当前车道上的前车
                front_vehicle = traci.vehicle.getLeader(vehicle_id, 100)
                if front_vehicle not in [None, ()] and front_vehicle[0] != '':  # 有可能是空的
                    front_vehicle_lane = traci.vehicle.getLaneID(front_vehicle[0])
                    front_vehicle_lane_index = int(front_vehicle_lane.split('_')[-1])
                    front_vehicle_road_id = front_vehicle_lane.split('_')[0]

                    ego_lane = state['vehicle'][vehicle_id]['lane_id']
                    ego_lane_index = int(ego_lane.split('_')[-1])
                    ego_road_id = ego_lane.split('_')[0]
                    if front_vehicle_lane_index != ego_lane_index:
                        pass
                    else:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = ego_speed - traci.vehicle.getSpeed(front_vehicle[0])
                        long_dist = traci.vehicle.getPosition(front_vehicle[0])[0] - ego_long_pos - 5
                        DRAC = relative_speed ** 2 / long_dist if relative_speed > 0 else 0
                        TTC = long_dist / relative_speed if relative_speed > 0 else 100
                        relative_accel = ego_accel - traci.vehicle.getAcceleration(front_vehicle[0])
                        relative_heading = traci.vehicle.getAngle(front_vehicle[0]) - ego_heading
                        if front_vehicle[0][:3] == 'HDV' or front_vehicle[0] == 'Leader':
                            veh_type = 0
                        else:
                            veh_type = 1
                        surrounding_vehicle_2['front'] = (front_vehicle[0], veh_type, long_dist, 0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                        sorrounding_vehicle_4['front'] = (front_vehicle[0], veh_type, long_dist, 0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                        surrounding_vehicle_6['front'] = (front_vehicle[0], veh_type, long_dist, 0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                        front_vehicle_expand = traci.vehicle.getLeader(front_vehicle[0], 100)
                        if front_vehicle_expand not in [None, ()] and front_vehicle_expand[0] != '':
                            relative_speed = ego_speed - traci.vehicle.getSpeed(front_vehicle_expand[0])
                            long_dist = traci.vehicle.getPosition(front_vehicle_expand[0])[0] - ego_long_pos - 5
                            TTC = long_dist / relative_speed if relative_speed > 0 else 100
                            DRAC = relative_speed ** 2 / long_dist if relative_speed > 0 else 0
                            relative_accel = ego_accel - traci.vehicle.getAcceleration(front_vehicle_expand[0])
                            relative_heading = traci.vehicle.getAngle(front_vehicle_expand[0]) - ego_heading
                            if front_vehicle_expand[0][:3] == 'HDV' or front_vehicle_expand[0] == 'Leader':
                                expand_veh_type = 0
                            else:
                                expand_veh_type = 1
                            sorrounding_vehicle_4['front_expand_0'] = (front_vehicle_expand[0], expand_veh_type, long_dist,
                                                                        0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                            surrounding_vehicle_6['front_expand_0'] = (front_vehicle_expand[0], expand_veh_type, long_dist,
                                                                        0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                            front_vehicle_expand_exp = traci.vehicle.getLeader(front_vehicle_expand[0], 100)
                            if front_vehicle_expand_exp not in [None, ()] and front_vehicle_expand_exp[0] != '':
                                relative_speed = ego_speed - traci.vehicle.getSpeed(front_vehicle_expand_exp[0])
                                long_dist = traci.vehicle.getPosition(front_vehicle_expand_exp[0])[0] - ego_long_pos - 5
                                TTC = long_dist / relative_speed if relative_speed > 0 else 100
                                DRAC = relative_speed ** 2 / long_dist if relative_speed > 0 else 0
                                relative_accel = ego_accel - traci.vehicle.getAcceleration(front_vehicle_expand_exp[0])
                                relative_heading = traci.vehicle.getAngle(front_vehicle_expand_exp[0]) - ego_heading
                                if front_vehicle_expand_exp[0][:3] == 'HDV' or front_vehicle_expand_exp[0] == 'Leader':
                                    expand_veh_type = 0
                                else:
                                    expand_veh_type = 1
                                surrounding_vehicle_6['front_expand_1'] = (front_vehicle_expand_exp[0], expand_veh_type, long_dist,
                                                                          0, relative_speed, relative_accel, relative_heading, TTC, DRAC)

                # 在当前车道上的后车
                back_vehicle = traci.vehicle.getFollower(vehicle_id, 100)
                if back_vehicle not in [None, ()] and back_vehicle[0] != '':  # 有可能是空的
                    back_vehicle_lane = traci.vehicle.getLaneID(back_vehicle[0])
                    back_vehicle_lane_index = int(back_vehicle_lane.split('_')[-1])
                    back_vehicle_road_id = back_vehicle_lane.split('_')[0]

                    ego_lane = state['vehicle'][vehicle_id]['lane_id']
                    ego_lane_index = int(ego_lane.split('_')[-1])
                    ego_road_id = ego_lane.split('_')[0]

                    if back_vehicle_lane_index != ego_lane_index:
                        # if back_vehicle_lane != ego_lane:
                        pass
                    else:
                        # 相对速度 - ego车的速度 - 后车的速度的差值
                        relative_speed = ego_speed - traci.vehicle.getSpeed(back_vehicle[0])
                        long_dist = ego_long_pos - traci.vehicle.getPosition(back_vehicle[0])[0] - 5
                        TTC = long_dist / abs(relative_speed) if relative_speed < 0 else 100
                        DRAC = relative_speed ** 2 / long_dist if relative_speed < 0 else 0
                        relative_accel = ego_accel - traci.vehicle.getAcceleration(back_vehicle[0])
                        relative_heading = traci.vehicle.getAngle(back_vehicle[0]) - ego_heading
                        if back_vehicle[0][:3] == 'HDV' or back_vehicle[0] == 'Leader':
                            veh_type = 0
                            if back_vehicle[0] == 'Leader':
                                long_dist = 0
                                DRAC = 10
                        elif back_vehicle[0][:3] == 'CAV':
                            veh_type = 1
                        else:
                            raise ValueError('Unknown vehicle type')
                        surrounding_vehicle_2['back'] = (back_vehicle[0], veh_type, -(long_dist), 0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                        sorrounding_vehicle_4['back'] = (back_vehicle[0], veh_type, -(long_dist), 0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                        surrounding_vehicle_6['back'] = (back_vehicle[0], veh_type, -(long_dist), 0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                        back_vehicle_expand = traci.vehicle.getFollower(back_vehicle[0], 100)
                        if back_vehicle_expand not in [None, ()] and back_vehicle_expand[0] != '':
                            relative_speed = ego_speed - traci.vehicle.getSpeed(back_vehicle_expand[0])
                            long_dist = ego_long_pos - traci.vehicle.getPosition(back_vehicle_expand[0])[0] - 5
                            TTC = long_dist / abs(relative_speed) if relative_speed < 0 else 100
                            DRAC = relative_speed ** 2 / long_dist if relative_speed < 0 else 0
                            relative_accel = ego_accel - traci.vehicle.getAcceleration(back_vehicle_expand[0])
                            relative_heading = traci.vehicle.getAngle(back_vehicle_expand[0]) - ego_heading
                            if back_vehicle_expand[0][:3] == 'HDV':
                                expand_veh_type = 0
                            elif back_vehicle_expand[0][:3] == 'CAV':
                                expand_veh_type = 1
                            else:
                                raise ValueError('Unknown vehicle type')
                            sorrounding_vehicle_4['back_expand_0'] = (back_vehicle_expand[0], expand_veh_type, -(long_dist),
                                                                    0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                            surrounding_vehicle_6['back_expand_0'] = (back_vehicle_expand[0], expand_veh_type, -(long_dist),
                                                                    0, relative_speed, relative_accel, relative_heading, TTC, DRAC)
                            back_vehicle_expand_exp = traci.vehicle.getFollower(back_vehicle_expand[0], 100)
                            if back_vehicle_expand_exp not in [None, ()] and back_vehicle_expand_exp[0] != '':
                                relative_speed = ego_speed - traci.vehicle.getSpeed(back_vehicle_expand_exp[0])
                                long_dist = ego_long_pos - traci.vehicle.getPosition(back_vehicle_expand_exp[0])[0] - 5
                                TTC = long_dist / abs(relative_speed) if relative_speed < 0 else 100
                                DRAC = relative_speed ** 2 / long_dist if relative_speed < 0 else 0
                                relative_accel = ego_accel - traci.vehicle.getAcceleration(back_vehicle_expand_exp[0])
                                relative_heading = traci.vehicle.getAngle(back_vehicle_expand_exp[0]) - ego_heading
                                if back_vehicle_expand_exp[0][:3] == 'HDV':
                                    expand_veh_type = 0
                                elif back_vehicle_expand_exp[0][:3] == 'CAV':
                                    expand_veh_type = 1
                                else:
                                    raise ValueError('Unknown vehicle type')
                                surrounding_vehicle_6['back_expand_1'] = (back_vehicle_expand_exp[0], expand_veh_type, -(long_dist),
                                                                          0, relative_speed, relative_accel, relative_heading, TTC, DRAC)

                surrounding_vehicles_2[vehicle_id] = surrounding_vehicle_2
                sorrounding_vehicles_4[vehicle_id] = sorrounding_vehicle_4
                sorrounding_vehicles_6[vehicle_id] = surrounding_vehicle_6

                pass
        for vehicle_id in surrounding_vehicles_2.keys():
            state['vehicle'][vehicle_id]['surround_2'] = surrounding_vehicles_2[vehicle_id]
            state['vehicle'][vehicle_id]['surround_4'] = sorrounding_vehicles_4[vehicle_id]
            state['vehicle'][vehicle_id]['surround_6'] = sorrounding_vehicles_6[vehicle_id]
            self.surround_vehicle_2[vehicle_id] = surrounding_vehicles_2[vehicle_id]
            self.surround_vehicle_4[vehicle_id] = sorrounding_vehicles_4[vehicle_id]
            self.surround_vehicle_6[vehicle_id] = sorrounding_vehicles_6[vehicle_id]

        return state

    # ##################
    # Tools for actions
    # ##################
    def __init_actions(self, raw_state):
        """初始化所有车辆(CAV+HDV)的速度:
        1. 所有车辆的速度保持不变, (0, -1) --> 0 表示不换道, -1 表示速度不变 [(-1, -1)表示HDV不接受RL输出]
        """
        self.actions = dict()
        for _veh_id, veh_info in raw_state['vehicle'].items():
            self.actions[_veh_id] = (-1, -1)

    def __update_actions(self, raw_action):
        """更新 ego 车辆的速度
        """
        delay_acc = 0.25
        K_acc = 0.75
        IDM_a = 2
        IDM_b = 4
        IDM_v0 = 20
        IDM_T = 1.5
        IDM_s0 = 2
        self.actual_actions = {ego_id: [] for ego_id in self.ego_ids}
        for vehicle_id in self.surround_vehicle_2.keys(): # self.surround_vehicle_expand.keys():
            for surround_key in self.required_surroundings:
                if surround_key not in self.surround_vehicle_2[vehicle_id]:
                    self.surround_vehicle_2[vehicle_id][surround_key] = 0
        leader_speed = np.sin(self.total_timesteps / 5) * 3 + 10
        self.actions['Leader'] = (0, leader_speed)

        for _veh_id in raw_action:
            if _veh_id in self.actions:  # 只更新 ego vehicle 的速度
                # IDM model
                ego_x = self.vehicles_info[_veh_id][3] if self.vehicles_info != {} else 0
                ego_v = self.vehicles_info[_veh_id][5] if self.vehicles_info != {} else 0
                if self.surround_vehicle_2[_veh_id] != {}:
                    front_1 = self.surround_vehicle_2[_veh_id]['front']
                    delta_v = front_1[4]
                    delta_s = front_1[2]
                else:
                    delta_v = 0
                    delta_s = 0
                s_star = IDM_s0 + max(0, ego_v * IDM_T + ego_v * delta_v / (2 * np.sqrt(IDM_a * IDM_b)))
                a_IDM = IDM_a * (1 - (ego_v / IDM_v0) ** 4 - (s_star / delta_s) ** 2)
                # 不换道，只需要更新速度
                control_input = raw_action[_veh_id] + a_IDM
                # current_acceleration = self.vehicles_info[_veh_id][6] if self.vehicles_info != {} else 0
                # time_step = self.vehicles_info[_veh_id][0] + 1 if self.vehicles_info != {} else 1
                # random_number = random.randint(1, 10)
                # distrubance_acc = 2 * np.sin(0.25 * time_step)
                # delta_acceleration = -1/delay_acc * current_acceleration + K_acc / delay_acc * control_input + distrubance_acc
                actual_acceleration = control_input # current_acceleration + delta_acceleration
                self.actual_actions[_veh_id].append([raw_action[_veh_id], control_input, actual_acceleration])
                speed_command = min(20, max(0, self.current_speed[_veh_id] + actual_acceleration))
                self.actions[_veh_id] = (0, speed_command)

                self.actor_action[_veh_id].append([raw_action[_veh_id], speed_command])

        return self.actions

    # ##########################
    # State and Reward Wrappers
    # ##########################
    def state_wrapper(self, state):
        """对原始信息的 state 进行处理, 分别得到:
        - 车道的信息
        - ego vehicle 的属性
        """
        state = state['vehicle'].copy()  # 只需要分析车辆的信息

        # 计算车辆和地图的数据
        lane_statistics, ego_statistics, reward_statistics, hdv_statistics = analyze_traffic(
            state=state, lane_ids=self.calc_features_lane_ids, max_veh_num=self.lane_max_num_vehs
        )
        start_time = time.time()

        # 计算每个 ego vehicle 的 state 拼接为向量
        if self.strategy == 'base':
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = compute_base_ego_vehicle_features(
                self,
                hdv_statistics=hdv_statistics,
                lane_statistics=lane_statistics,
                ego_statistics=ego_statistics,
                unique_edges=self.edge_ids,
                edge_lane_num=self.edge_lane_num,
                node_positions=self.node_positions,
                ego_ids=self.ego_ids,
            )
        elif self.strategy == 'iMARL':
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = compute_hierarchical_ego_vehicle_features(
                self,
                hdv_statistics=hdv_statistics,
                lane_statistics=lane_statistics,
                ego_statistics=ego_statistics,
                unique_edges=self.edge_ids,
                edge_lane_num=self.edge_lane_num,
                node_positions=self.node_positions,
                ego_ids=self.ego_ids,
            )
        # end_time = time.time()
        # print(f'Time for feature extraction: {end_time - start_time} second')

        return feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, lane_statistics, ego_statistics, reward_statistics

    def reward_wrapper(self, lane_statistics, ego_statistics, reward_statistics) -> float:
        """
        根据 ego vehicle 的状态信息和 reward 的统计信息计算 reward
        我希望：
            global reward
                1. 所有车CAV+HDV都能够尽可能时间短的到达终点
                2. 所有CAV的平均速度尽可能的接近最快速度
                3.

            special reward (near bottleneck area)
                1. 在通过bottleneck的时候尽量全速通过
                2. CAV车辆尽可能的减速
                2. CAV车辆尽可能的保持距离


            local reward
                1. 每个单独的CAV尽可能不要和其他车辆碰撞
                    a. 警告距离
                    b. 碰撞距离
                2. 每个单独的CAV尽可能的保持最快速度
                3. 每个单独的CAV离开路网的奖励

        """
        max_speed = 20  # ego vehicle 的最快的速度
        max_acceleration = 3
        TTC_warning_threshold = 1.0
        TTC_collision_threshold = 0.5
        K_TTC = 20 / (TTC_warning_threshold - TTC_collision_threshold)
        T_exp = 1
        L_exp = 2

        # 先把reward_statistics中所有的车辆的信息都全局记录下来self.vehicles_info
        for veh_id, (road_id, distance, position_x, position_y, speed, acceleration, heading, waiting_time,
                     accumulated_waiting_time) in reward_statistics.items():
            self.vehicles_info[veh_id] = [
                self.vehicles_info.get(veh_id, [0, None])[0] + 1,  # travel time
                road_id,
                distance,
                position_x,
                position_y,
                speed,
                acceleration,
                heading,
                waiting_time,
                accumulated_waiting_time
            ]
        # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
        if len(self.out_of_road) > 0:
            for veh_id in self.out_of_road:
                if veh_id in self.vehicles_info:
                    del self.vehicles_info[veh_id]

        # ######################### 开始计算reward  # #########################
        inidividual_rew_ego = {key: {} for key in list(set(self.ego_ids) - set(self.out_of_road))}

        # ######################## 初始化 for group reward ########################
        all_ego_vehicle_speed = []  # CAV车辆的平均速度 - 使用target speed
        all_ego_vehicle_mean_speed = []  # CAV车辆的累积平均速度 - 使用速度/时间
        all_ego_vehicle_acceleration = []  # CAV车辆的平均加速度
        all_ego_vehicle_accumulated_waiting_time = []  # # CAV车辆的累积平均等待时间
        all_ego_vehicle_waiting_time = []  # CAV车辆的等待时间

        all_vehicle_speed = []  # CAV和HDV车辆的平均速度 - 使用target speed
        all_vehicle_mean_speed = []  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
        all_vehicle_acceleration = []  # CAV和HDV车辆的平均加速度
        all_vehicle_accumulated_waiting_time = []  # CAV和HDV车辆的累积平均等待时间
        all_vehicle_waiting_time = []  # CAV和HDV车辆的等待时间

        for veh_id, (veh_travel_time, road_id, distance, position_x,
                     position_y, speed, acceleration, heading,
                     waiting_time, accumulated_waiting_time) in list(self.vehicles_info.items()):

            # CAV和HDV车辆的
            all_vehicle_speed.append(speed)
            all_vehicle_mean_speed.append(distance / veh_travel_time)
            all_vehicle_acceleration.append(abs(acceleration))
            all_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
            all_vehicle_waiting_time.append(waiting_time)

            # 把CAV单独取出来
            if veh_id in self.ego_ids:

                # for group reward 计算CAV车辆的累积平均速度
                all_ego_vehicle_speed.append(speed)
                all_ego_vehicle_mean_speed.append(distance / veh_travel_time)
                all_ego_vehicle_acceleration.append(abs(acceleration))
                all_ego_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
                all_ego_vehicle_waiting_time.append(waiting_time)

                # ######################## for individual reward ########################
                # # CAV车辆的累积平均速度越靠近最大速度，reward越高 - [0, 5]
                # individual_speed_r = -abs(distance / veh_travel_time - max_speed) / max_speed * 5 + 5
                # inidividual_rew_ego[veh_id] += 1 * individual_speed_r

                # CAV车辆的target速度越靠近最大速度，reward越高 - [0, 1]
                individual_speed_r_simple = -abs(speed - max_speed) / max_speed * 1 + 1
                inidividual_rew_ego[veh_id]['efficiency'] = individual_speed_r_simple
                # CAV车辆的加速度绝对值越小，reward越高 - [-1, 0]
                # individual_acceleration_r = -(acceleration ** 2) * 0.5
                individual_acceleration_r = -((acceleration/max_acceleration) ** 2) # + 1
                inidividual_rew_ego[veh_id]['comfort'] = individual_acceleration_r

                # 警告距离和碰撞距离, stability-related values
                if 'front' not in ego_statistics[veh_id][6].keys():
                    TTC = 100
                    s_act = L_exp
                    v_act = 0
                    delta_v = 0
                else:
                    front_info = ego_statistics[veh_id][6]['front']
                    TTC = front_info[7]
                    s_act = front_info[2]
                    v_act = ego_statistics[veh_id][1]
                    delta_v = -front_info[4]
                # stability reward
                delta_s = s_act - (v_act * T_exp + L_exp)
                stability_matric = np.array([delta_s, delta_v])
                stability_weight = np.array([[1, 0], [0, 0.5]])
                e_ss = stability_matric @ stability_weight @ stability_matric.T
                inidividual_rew_ego[veh_id]['stability'] = np.exp(-e_ss)

                if TTC_collision_threshold < TTC <= TTC_warning_threshold:
                    safety_r = (1 / np.exp(-(K_TTC * (TTC - (TTC_collision_threshold + TTC_warning_threshold)/2)))) - 1
                elif TTC <= TTC_collision_threshold:
                    safety_r = -1
                else:
                    safety_r = 0
                inidividual_rew_ego[veh_id]['safety'] = safety_r  # [-1, 0]
        all_rew_ego = inidividual_rew_ego.copy()
        rew_safety = {key: inidividual_rew_ego[key]['safety'] for key in inidividual_rew_ego}
        rew_stability = {key: inidividual_rew_ego[key]['stability'] for key in inidividual_rew_ego}
        rew_efficiency = {key: inidividual_rew_ego[key]['efficiency'] for key in inidividual_rew_ego}
        rew_comfort = {key: inidividual_rew_ego[key]['comfort'] for key in inidividual_rew_ego}

        # 计算全局reward
        all_ego_vehicle_speed = np.mean(all_ego_vehicle_speed)  # CAV车辆的平均速度 - 使用target speed
        # all_ego_mean_speed = np.mean(all_ego_vehicle_mean_speed)  # CAV车辆的累积平均速度 - 使用速度/时间
        all_ego_vehicle_acceleration = np.mean(all_ego_vehicle_acceleration)  # CAV车辆的平均加速度

        all_vehicle_speed = np.mean(all_vehicle_speed)  # CAV和HDV车辆的平均速度 - 使用target speed
        # all_vehicle_mean_speed = np.mean(all_vehicle_mean_speed)  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
        all_vehicle_acceleration = np.mean(all_vehicle_acceleration)  # CAV和HDV车辆的平均加速度
        global_ego_speed_r = -abs(all_ego_vehicle_speed - max_speed) / max_speed * 1 + 1  # [0, 5]
        # global_ego_mean_speed_r = -abs(all_ego_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_ego_acceleration_r = -abs(all_ego_vehicle_acceleration) + 6  # [0, 6]
        global_ego_acceleration_r = -((all_ego_vehicle_acceleration/max_acceleration) ** 2) # + 1

        # global_all_speed_r = -abs(all_vehicle_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_all_mean_speed_r = -abs(all_vehicle_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_all_acceleration_r = -abs(all_vehicle_acceleration) + 6   # [0, 6]
        for veh_id in all_rew_ego.keys():
            all_rew_ego[veh_id]['global_efficiency'] = global_ego_speed_r
            all_rew_ego[veh_id]['global_comfort'] = global_ego_acceleration_r

        time_penalty = 0
        weight = {'efficiency': 1, 'comfort': 1, 'safety': 1, 'stability': 1}

        rewards = {key: weight['safety'] * inidividual_rew_ego[key]['safety'] \
                        + weight['stability'] * inidividual_rew_ego[key]['stability'] \
                        + weight['efficiency'] * (1-self.CAV_penetration) * inidividual_rew_ego[key]['efficiency'] \
                        + weight['comfort'] * (1-self.CAV_penetration) * inidividual_rew_ego[key]['comfort'] \
                        # + time_penalty_ego[key] \
                        + weight['efficiency'] * self.CAV_penetration * all_rew_ego[key]['global_efficiency'] \
                        # + global_ego_mean_speed_r \
                        + weight['comfort'] * self.CAV_penetration * all_rew_ego[key]['global_comfort'] \
                        # + global_ego_waiting_time_r \
                        # + global_ego_accumulated_waiting_time_r \
                        # + global_all_speed_r \
                        # + global_all_mean_speed_r \
                        # + global_all_acceleration_r \
                        + time_penalty
                   for key in inidividual_rew_ego}

        return rewards, all_vehicle_speed, all_vehicle_acceleration, rew_safety, rew_stability, rew_efficiency, rew_comfort
    # ############
    # Collision
    # #############

    def check_collisions(self, init_state):

        ################# 碰撞检查 ###########################################
        # 简单版本 - 根据车头的两两位置计算是否碰撞
        collisions_head_vehs, collisions_head_info = check_collisions_based_pos(init_state['vehicle'],
                                                                                gap_threshold=GAP_THRESHOLD)

        # print('point to point collision:', collisions_head_vehs, collisions_head_info)

        # 稍微复杂的版本 - 根据neighbour位置计算是否碰撞
        collisions_neigh_vehs, warn_neigh_vehs, collisions_neigh_info = check_collisions(init_state['vehicle'],
                                                                                         self.ego_ids,
                                                                                         gap_threshold=GAP_THRESHOLD,
                                                                                         gap_warn_collision=WARN_GAP_THRESHOLD
                                                                                         # 给reward的警告距离
                                                                                         )
        # print('neighbour collision:', collisions_neigh_vehs, collisions__neigh_info)

        collisions_for_reward = {
            'collision': collisions_head_vehs + collisions_neigh_vehs,
            'warn': warn_neigh_vehs,
            'info': collisions_neigh_info + collisions_head_info
        }

        self.warn_ego_ids = {}
        self.coll_ego_ids = {}

        for key, value in collisions_for_reward.items():
            if key == 'warn' and len(value) != 0:
                for element in collisions_for_reward['info']:
                    if 'warn' in element:
                        if not element['CAV_key'] in self.warn_ego_ids:
                            self.warn_ego_ids.update({element['CAV_key']: [element['distance']]})
                        else:
                            # append the distance
                            self.warn_ego_ids[element['CAV_key']].append(element['distance'])

            if key == 'collision' and len(value) != 0:
                for element in collisions_for_reward['info']:
                    if 'collision' in element:
                        if not element['CAV_key'] in self.coll_ego_ids:
                            self.coll_ego_ids.update({element['CAV_key']: [element['distance']]})
                        else:
                            self.coll_ego_ids[element['CAV_key']].append(element['distance'])

    # ############
    # reset & step
    # #############

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化
        """
        # 初始化超参数
        # bottleneck 处的拥堵程度 # TODO: 根据lane statastics来计算
        self.congestion_level = 0
        # 记录仿真内所有车辆的信息 - 在reward wrapper中更新
        self.vehicles_info = {}
        # 记录行驶出路网的车辆
        self.out_of_road = []
        # 假设这些车初始化都在路网上 活着
        self.agent_mask = {ego_id: True for ego_id in self.ego_ids}
        self.current_speed = {key: 10 for key in self.ego_ids}

        # 初始化环境
        init_state = self.env.reset()
        # 生成车流
        generate_scenario(aggressive=self.aggressive,
                          cautious=self.cautious,
                          normal=self.normal,
                          use_gui=self.use_gui, sce_name=self.name_scenario,
                          CAV_num=self.num_CAVs, HDV_num=self.num_HDVs, CAV_penetration=self.CAV_penetration,
                          distribution="uniform")  # generate_scene_MTF.py - "random" or "uniform" distribution

        # if 0 <= self.total_timesteps < 1000000:
        #     assert self.num_CAVs == 5
        #     assert self.CAV_penetration == 0.5
        #     # 生成车流
        #     generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario,
        #                       CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
        #                       distribution="uniform")  # generate_scene_MTF.py - "random" or "uniform" distribution
        #
        # elif 1000000 <= self.total_timesteps < 2000000:
        #     self.num_CAVs = 5
        #     self.CAV_penetration = 0.3
        #     generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario,
        #                       CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
        #                       distribution="uniform")
        # elif 2000000 <= self.total_timesteps <= 3000000:
        #     self.num_CAVs = 5
        #     self.CAV_penetration = 0.1
        #     generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario,
        #                       CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
        #                       distribution="uniform")

        # 初始化车辆的速度
        self.__init_actions(raw_state=init_state)

        # 对于warmup step = 0也适用
        for _ in range(self.warmup_steps + 1):
            init_state, _, _, _, _ = super().step(self.actions)
            init_state = self.append_surrounding(init_state)

            # 检查是否有碰撞
            collisions_vehs, warn_vehs, collision_infos = check_collisions(init_state['vehicle'],
                                                                           self.ego_ids,
                                                                           gap_threshold=GAP_THRESHOLD,
                                                                           gap_warn_collision=WARN_GAP_THRESHOLD)
            # reset 时不应该有碰撞
            assert len(collisions_vehs) == 0, f'Collision with {collisions_vehs} at reset!!! Regenerate the flow'
            assert len(warn_vehs) == 0, f'Warning with {warn_vehs} at reset!!! Regenerate the flow'

            # 对 state 进行处理
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, lane_statistics, _, _ = self.state_wrapper(state=init_state)
            # shared_feature_vectors = compute_centralized_vehicle_features(lane_statistics,
            #                                                               feature_vectors,
            #                                                               self.bottle_neck_positions)
            actor_features, actor_features_flatten, shared_features, shared_features_flatten = compute_centralized_vehicle_features_hierarchical_version(
                self.obs_size, self.shared_obs_size,
                lane_statistics,
                feature_vectors_current, feature_vectors_current_flatten,
                feature_vectors, feature_vectors_flatten, self.ego_ids)
            self.__init_actions(raw_state=init_state)

        return feature_vectors_flatten, shared_features_flatten, {'step_time': self.warmup_steps + 1}

    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        """
        self.total_timesteps += 1

        # 已经死了的车辆不控制 - 从 action 中删除
        for ego_id, ego_live in self.agent_mask.items():
            if not ego_live:
                del action[ego_id]

        # 更新 action
        action = self.__update_actions(raw_action=action).copy()
        # 记录render车辆轨迹信息
        # if self.vehicles_info:
        #     for veh_id in self.vehicles_info.keys():
        #         if veh_id in self.safety_after.keys():
        #             veh_info = [self.vehicles_info[veh_id][0], self.vehicles_info[veh_id][4], self.vehicles_info[veh_id][5],
        #                         self.vehicles_info[veh_id][3], self.safety_after[veh_id]]
        #         else:
        #             veh_info = [self.vehicles_info[veh_id][0], self.vehicles_info[veh_id][4], self.vehicles_info[veh_id][5],
        #                         self.vehicles_info[veh_id][3], 0]
        #         csv_path = self.csv_dir + '/' + veh_id + '_run_info.csv'
        #         with open(csv_path, 'a', newline='') as csvfile:
        #             writer = csv.writer(csvfile)
        #             writer.writerow(veh_info)
        # else:
        #     for veh_id in action.keys():
        #         if self.use_gui:
        #             import traci as traci
        #         else:
        #             import libsumo as traci
        #         if veh_id == 'CAV_0':
        #             now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        #             self.csv_dir = self.save_csv_dir + '/' + now_time
        #             os.makedirs(self.csv_dir)
        #         csv_path = self.csv_dir + '/' + veh_id + '_run_info.csv'
        #         veh_info = [0, traci.vehicle.getPosition(veh_id)[0], traci.vehicle.getPosition(veh_id)[1],
        #                     traci.vehicle.getSpeed(veh_id), 0]
        #         with open(csv_path, 'a', newline='') as csvfile:
        #             writer = csv.writer(csvfile)
        #             writer.writerow(veh_info)

        # 在环境里走一步
        init_state, rewards, truncated, _dones, infos = super().step(action)
        init_state = self.append_surrounding(init_state)
        self.current_speed = {key: init_state['vehicle'][key]['speed'] if key in init_state['vehicle'] else 0 for key in
                              self.ego_ids}
        self.current_lane = {key: init_state['vehicle'][key]['lane_id'] if key in init_state['vehicle'] else 0 for key
                             in self.ego_ids}

        ################# 碰撞检查 ###########################################
        self.check_collisions(init_state)
        ####################################################################

        # 对 state 进行处理 (feature_vectors的长度是没有行驶出CAV的数量)
        feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, lane_statistics, ego_statistics, reward_statistics = self.state_wrapper(
            state=init_state)

        # 处理离开路网的车辆 agent_mask 和 out_of_road
        for _ego_id in self.ego_ids:
            if _ego_id not in feature_vectors:
                assert _ego_id not in ego_statistics, f'ego vehicle {_ego_id} should not be in ego_statistics'
                assert _ego_id not in reward_statistics, f'ego vehicle {_ego_id} should not be in reward_statistics'
                self.agent_mask[_ego_id] = False  # agent 离开了路网, mask 设置为 False
                if _ego_id not in self.out_of_road:
                    self.out_of_road.append(_ego_id)

        # 初始化车辆的速度
        self.__init_actions(raw_state=init_state)

        # 处理 dones 和 infos
        if len(self.coll_ego_ids) == 0 and len(feature_vectors) > 0:  # 还有车在路上 且还没有碰撞发生
            # 计算此时的reward （这里的reward只有还在路网上的车的reward）
            rewards, mean_speeds, mean_accelerations, rew_safety, rew_stability, rew_efficiency, rew_comfort = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)

        elif len(self.coll_ego_ids) > 0 and len(feature_vectors) > 0:  # 还有车在路上 但有车辆碰撞
            # 计算此时的reward
            rewards, mean_speeds, mean_accelerations, rew_safety, rew_stability, rew_efficiency, rew_comfort = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)  # 更新 veh info
            for collid_ego_id in self.coll_ego_ids:
                infos['collision'].append(collid_ego_id)
                self.agent_mask[collid_ego_id] = False
            infos['done_reason'] = 'collision'

        else:  # 所有RL车离开的时候, 就结束
            assert len(feature_vectors) == 0, f'All RL vehicles should leave the environment'
            infos['done_reason'] = 'all CAV vehicles leave the environment'
            # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
            init_state, rew, truncated, _d, _ = super().step(self.actions)
            init_state = self.append_surrounding(init_state)
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, lane_statistics, ego_statistics, reward_statistics = self.state_wrapper(
                state=init_state)
            self.__init_actions(raw_state=init_state)

            while len(reward_statistics) > 0:
                init_state, rew, truncated, _d, _ = super().step(self.actions)
                init_state = self.append_surrounding(init_state)
                feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, lane_statistics, ego_statistics, reward_statistics = self.state_wrapper(
                    state=init_state)
                self.__init_actions(raw_state=init_state)
                # rewards = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)  # 更新 veh info

            infos['out_of_road'] = self.ego_ids
            assert set(self.out_of_road) == set(self.ego_ids), f'All RL vehicles should leave the environment'
            rewards = {key: 20.0 for key in self.ego_ids}
            for out_of_road_ego_id in self.out_of_road:
                self.agent_mask[out_of_road_ego_id] = False
                feature_vectors_flatten[out_of_road_ego_id] = np.zeros(self.obs_size)

        # 处理以下reward
        if len(self.out_of_road) > 0 and len(feature_vectors) > 0:
            for out_of_road_ego_id in self.out_of_road:
                rewards.update({out_of_road_ego_id: 0.0})  # 离开路网之后 reward 也是 0  # TODO: 注意一下dead mask MARL
                if out_of_road_ego_id not in infos['out_of_road']:
                    infos['out_of_road'].append(out_of_road_ego_id)
                self.agent_mask[out_of_road_ego_id] = False
                feature_vectors_flatten[out_of_road_ego_id] = np.zeros(self.obs_size)

        # 获取shared_feature_vectors
        # shared_feature_vectors = compute_centralized_vehicle_features(lane_statistics,
        #                                                               feature_vectors,
        #                                                               self.bottle_neck_positions)
        actor_features, actor_features_flatten, shared_features, shared_features_flatten = compute_centralized_vehicle_features_hierarchical_version(
            self.obs_size,
            self.shared_obs_size,
            lane_statistics,
            feature_vectors_current,
            feature_vectors_current_flatten,
            feature_vectors,
            feature_vectors_flatten,
            self.ego_ids)
        # 处理以下 infos
        if len(self.warn_ego_ids) > 0:
            infos['warning'].append(self.warn_ego_ids)

        # 处理以下done
        dones = {}
        for _ego_id in self.ego_ids:
            dones[_ego_id] = not self.agent_mask[_ego_id]

        # 只要有一个车辆碰撞，就结束所有车辆的仿真
        if len(self.coll_ego_ids) > 0:
            for ego_id in self.ego_ids:
                dones[ego_id] = True

        # # 只要有一个车辆碰撞，不要结束所有车辆的仿真
        # if len(self.coll_ego_ids) > 0:
        #     for ego_id in self.coll_ego_ids:
        #         dones[ego_id] = True

        # 超出时间 结束仿真
        if infos['step_time'] >= self.max_num_seconds:
            for ego_id in self.ego_ids:
                dones[ego_id] = True
                infos['done_reason'] = 'time out'

        # # For DEBUG render
        # infos['done'] = dones.copy()
        # infos['reward'] = rewards.copy()
        # print(infos)
        # debug = []
        # for key, value in feature_vectors.items():
        #     debug.append([key, value[0] * 15])
        # print(debug)

        # TODO: 完成时间越短，reward越高 - [0, 5]

        # # check if all element in feature_vectors_flatten have same length
        # if len(shared_features_flatten) != 5:
        #     print('break')
        # for value in shared_features_flatten.values():
        #     if len(value) != 253 or (not isinstance(value, np.ndarray)):
        #         print('break')

        return feature_vectors_flatten, shared_features_flatten, rewards, mean_speeds, mean_accelerations, rew_safety, rew_stability, rew_efficiency, rew_comfort, dones.copy(), dones.copy(), infos

    def close(self) -> None:
        return super().close()

