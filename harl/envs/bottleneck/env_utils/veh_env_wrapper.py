import time
from typing import Any, SupportsFloat, Tuple, Dict, List
import random
import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from loguru import logger
import math

# from .generate_scene import generate_scenario
# from .generate_scene_MTF import generate_scenario
from .generate_scene_fix_num_CAV import generate_scenario
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

# TODO: 在MARL actor中可以考虑分开encode - 地图信息 - 单车信息 - 所有车的信息 （借鉴MAPPO对state的处理

# TODO: sumo 底层 允许换道碰撞 + 判断

# TODO: 我明天需要确认一下我的evaluation指标是什么

# TODO: 如何在时序上或者空间上让他知道后面的bottleneck需要减速慢行  - 距离bottleneck的距离作为一个factor
# TODO： 如何在时序或者空间上提高bottleneck区域的重要性

# TODO: 在E3全速前进 - rule based

# TODO: reward design is related to the CAV penetration rate, if the CAV penetration rate is higher, the weight of the global reward is higher???

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
                 edge_ids: List[str],  # 路网中所有路段的 id
                 edge_lane_num: Dict[str, int],  # 每个 edge 的车道数
                 calc_features_lane_ids: List[str],  # 需要统计特征的 lane id
                 bottle_necks: List[str],  # 路网中 bottleneck
                 bottle_neck_positions: Tuple[float],  # bottle neck 的坐标
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

        # random generate self.num_CAVs CAVs from range (4, 6)
        # self.num_CAVs = random.choice([4, 5, 6])
        # self.CAV_penetration = random.choice([0.4, 0.5, 0.6])

        self.edge_ids = edge_ids
        self.edge_lane_num = edge_lane_num
        self.bottle_necks = bottle_necks
        self.calc_features_lane_ids = calc_features_lane_ids  # 需要统计特征的 lane id
        self.bottle_neck_positions = bottle_neck_positions  # bottle neck 的坐标
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

        # 记录当前速度
        self.current_speed = {key: 0 for key in self.ego_ids}
        # 记录当前的lane
        self.current_lane = {key: 0 for key in self.ego_ids}

        self.action_pointer = {
            0: (0, 0),  # 不换道
            1: (1, 0),  # 左换道
            2: (2, 0),  # 右换道
            3: (0, 0),  # 加速度+2
            4: (0, 0),  # 减速-2
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
        # global part: road structure(10) + bottleneck(2) + all CAVs(max*6) + all HDVs(max*6) + all lanes(18*6)
        # individual part: target(2) + self stats(13) + action(3) + surround CAVs(6*6) + surround HDVs(6*6) + surround lanes(3*6)
        hier_obs_size = 10 + 2 + 2 + self.max_num_CAVs*6 + self.max_num_HDVs*6 + 18*6 \
                        + 2 + 4 + 13 + 6*6 + 6*6 + 3*6 +3
        # road structure + self stats + surround vehs + surround lanes
        # base_obs_size = 3 + 2 + 2 + 3 * 6
        base_obs_size = 2 + 3 + 2 + 2 + 3 * 6 + 2 + 2
        # base_obs_size = 10 + 13 + 3 * 6 + 3 * 6
        # FP-target position（根据当前位置、静态地图、车道特征筛选） + self stats + surround vehs
        # base_obs_size = 2 + 3 + 3 * 6
        if self.strategy == 'hierarchical':
            self.self_obs_size = hier_obs_size
        elif self.strategy == 'base':
            self.self_obs_size = base_obs_size
        elif self.strategy == 'attention':
            self.self_obs_size = hier_obs_size
        elif self.strategy == 'prediction':
            self.self_obs_size = hier_obs_size
        elif self.strategy == 'cross_aware':
            self.self_obs_size = hier_obs_size
        self.hist_info = {}
        self.action_generation = {}
        self.action_execution = {}
        # FP-global part: road structure(10) +all CAVs(max*6) + all HDVs(max*6) + all lanes(18*6)
        # FP-individual part: target(2) + self stats(13) + surround vehs(6*3)
        # self.shared_obs_size = 10 + 2 + 2 + self.max_num_CAVs*6 + self.max_num_HDVs*6 + 18*6  # EP mode
                               # + 2 + 13 + 6*3
        self.shared_obs_size = hier_obs_size  # EP mode (FP extended)
        # self.shared_obs_size = 3 * self.max_num_CAVs + 18 * 3
        # EP-road structure + cav stats
        # self.shared_obs_size = 2 + 2 + 3 * self.max_num_CAVs + 18 * 3  ####baseline
        if self.use_hist_info:
            self.obs_size = self.self_obs_size * (self.hist_length+1)
            for i in range(self.hist_length):
                self.hist_info[f'hist_{i+1}'] = {ego_id: np.zeros(self.self_obs_size) for ego_id in self.ego_ids}
                self.action_generation[f'hist_{i+1}'] = {ego_id: 0 for ego_id in self.ego_ids}
                self.action_execution[f'hist_{i+1}'] = {ego_id: (0, 0.0) for ego_id in self.ego_ids}
        else:
            self.obs_size = self.self_obs_size
        self.surround_vehicle = {ego_id: {} for ego_id in self.ego_ids}
        self.required_surroundings = ['front', 'back', 'left_followers', 'right_followers', 'left_leaders', 'right_leaders']
        self.safety_before = {ego_id: [] for ego_id in self.ego_ids}
        self.safety_after = {ego_id: [] for ego_id in self.ego_ids}

    # #####################
    # Obs and Action Space
    # #####################
    @property
    def action_space(self):
        """直接控制 ego vehicle 的速度
        """
        return {_ego_id: gym.spaces.Discrete(5) for _ego_id in self.ego_ids}

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
        surrounding_vehicles = {}
        sorrounding_vehicles_expand = {}
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
                surrounding_vehicle = {}
                sorrounding_vehicle_expand = {}

                modes_follow = {
                    'left_followers': 0b000,  # Left and followers
                    'right_followers': 0b001,  # Left and leaders
                }
                # ego车的左右车道的后面的车辆
                for key, mode in modes_follow.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    for n in neighbors:
                        if key == 'left_followers':
                            lateral_dist = 3.2
                        else:
                            lateral_dist = -3.2
                        # 相对速度 - ego车的速度 - 后车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                        if n[0][:3] == 'HDV':
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是HDV）
                            # 3.2是lane的宽度，1.5是HDV的minGap
                            long_dist = n[1] + 1.5
                            surrounding_vehicle[key] = (n[0], -(long_dist), lateral_dist, relative_speed)
                            sorrounding_vehicle_expand[key] = (n[0], -(long_dist), lateral_dist, relative_speed)
                        elif n[0][:3] == 'CAV':
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是CAV）
                            # 3.2是lane的宽度，1.0是CAV的minGap
                            long_dist = n[1] + 1.0
                            surrounding_vehicle[key] = (n[0], -(long_dist), lateral_dist, relative_speed)
                            sorrounding_vehicle_expand[key] = (n[0], -(long_dist), lateral_dist, relative_speed)
                        else:
                            raise ValueError('Unknown vehicle type')
                        neighbors_expand = traci.vehicle.getFollower(n[0])
                        if neighbors_expand not in [None, ()] and neighbors_expand[0] != '':
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(neighbors_expand[0])
                            if neighbors_expand[0][:3] == 'HDV':
                                sorrounding_vehicle_expand[key+'_expand'] = (neighbors_expand[0], -(neighbors_expand[1] + 1.5 + 5 + long_dist),
                                                                             lateral_dist, relative_speed)
                            elif neighbors_expand[0][:3] == 'CAV':
                                sorrounding_vehicle_expand[key+'_expand'] = (neighbors_expand[0], -(neighbors_expand[1] + 1.0 + 5 + long_dist),
                                                                             lateral_dist, relative_speed)
                            else:
                                raise ValueError('Unknown vehicle type')

                modes_lead = {
                    'left_leaders': 0b010,  # Right and followers
                    'right_leaders': 0b011  # Right and leaders
                }
                # ego车的左右车道的前面的车辆
                for key, mode in modes_lead.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    if key == 'left_leaders':
                        lateral_dist = 3.2
                    else:
                        lateral_dist = -3.2
                    for n in neighbors:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                        long_dist = n[1] + 1.0
                        surrounding_vehicle[key] = (n[0], long_dist, lateral_dist, relative_speed)
                        sorrounding_vehicle_expand[key] = (n[0], long_dist, lateral_dist, relative_speed)
                        # traci得到的距离是ego到前车尾部减去minGap的距离，所以要加上minGap（此时后车是ego）
                        # 3.2是lane的宽度，1.0是CAV的minGap
                        neighbors_expand = traci.vehicle.getLeader(n[0])
                        if neighbors_expand not in [None, ()] and neighbors_expand[0] != '':
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(neighbors_expand[0])
                            if n[0][:3] == 'HDV':
                                sorrounding_vehicle_expand[key + '_expand'] = (neighbors_expand[0], neighbors_expand[1] + 1.5 + 5 + long_dist,
                                                                               lateral_dist, relative_speed)
                            elif n[0][:3] == 'CAV':
                                sorrounding_vehicle_expand[key + '_expand'] = (neighbors_expand[0], neighbors_expand[1] + 1.0 + 5 + long_dist,
                                                                               lateral_dist, relative_speed)

                # 在当前车道上的前车
                front_vehicle = traci.vehicle.getLeader(vehicle_id)
                if front_vehicle not in [None, ()] and front_vehicle[0] != '':  # 有可能是空的
                    front_vehicle_lane = traci.vehicle.getLaneID(front_vehicle[0])
                    front_vehicle_lane_index = int(front_vehicle_lane.split('_')[-1])
                    front_vehicle_road_id = front_vehicle_lane.split('_')[0]

                    ego_lane = state['vehicle'][vehicle_id]['lane_id']
                    ego_lane_index = int(ego_lane.split('_')[-1])
                    ego_road_id = ego_lane.split('_')[0]
                    # if front_vehicle_road_id[:3] == ':J3' or ego_road_id[:3] == ':J3':
                    #     print('debug')
                    if front_vehicle_lane_index != ego_lane_index:
                        pass
                    else:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(front_vehicle[0])
                        long_dist = front_vehicle[1] + 1.0
                        surrounding_vehicle['front'] = (front_vehicle[0], long_dist, 0, relative_speed)
                        sorrounding_vehicle_expand['front'] = (front_vehicle[0], long_dist, 0, relative_speed)
                        front_vehicle_expand = traci.vehicle.getLeader(front_vehicle[0])
                        # traci得到的距离是ego到前车尾部减去minGap的距离，所以要加上minGap（此时后车是ego）
                        if front_vehicle[0][:3] == 'HDV':
                            if front_vehicle_expand not in [None, ()] and front_vehicle_expand[0] != '':
                                relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(front_vehicle_expand[0])
                                sorrounding_vehicle_expand['front_expand'] = (front_vehicle_expand[0], front_vehicle_expand[1] + 1.5 + 5 + long_dist,
                                                                              0, relative_speed)
                        elif front_vehicle[0][:3] == 'CAV':
                            if front_vehicle_expand not in [None, ()] and front_vehicle_expand[0] != '':
                                relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(front_vehicle_expand[0])
                                sorrounding_vehicle_expand['front_expand'] = (front_vehicle_expand[0], front_vehicle_expand[1] + 1.0 + 5 + long_dist,
                                                                              0, relative_speed)
                        else:
                            raise ValueError('Unknown vehicle type')

                # 在当前车道上的后车
                back_vehicle = traci.vehicle.getFollower(vehicle_id)
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
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(back_vehicle[0])
                        if back_vehicle[0][:3] == 'HDV':
                            long_dist = back_vehicle[1] + 1.5
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是HDV）
                            surrounding_vehicle['back'] = (back_vehicle[0], -(long_dist), 0, relative_speed)
                            sorrounding_vehicle_expand['back'] = (back_vehicle[0], -(long_dist), 0, relative_speed)
                        elif back_vehicle[0][:3] == 'CAV':
                            long_dist = back_vehicle[1] + 1.0
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是CAV）
                            surrounding_vehicle['back'] = (back_vehicle[0], -(long_dist), 0, relative_speed)
                            sorrounding_vehicle_expand['back'] = (back_vehicle[0], -(long_dist), 0, relative_speed)
                        else:
                            raise ValueError('Unknown vehicle type')
                        back_vehicle_expand = traci.vehicle.getFollower(back_vehicle[0])
                        if back_vehicle_expand not in [None, ()] and back_vehicle_expand[0] != '':
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(back_vehicle_expand[0])
                            if back_vehicle_expand[0][:3] == 'HDV':
                                sorrounding_vehicle_expand['back_expand'] = (back_vehicle_expand[0], -(back_vehicle_expand[1] + 1.5 + 5 + long_dist),
                                                                             0, relative_speed)
                            elif back_vehicle_expand[0][:3] == 'CAV':
                                sorrounding_vehicle_expand['back_expand'] = (back_vehicle_expand[0], -(back_vehicle_expand[1] + 1.0 + 5 + long_dist),
                                                                             0, relative_speed)
                            else:
                                raise ValueError('Unknown vehicle type')

                surrounding_vehicles[vehicle_id] = surrounding_vehicle
                sorrounding_vehicles_expand[vehicle_id] = sorrounding_vehicle_expand

                pass
        for vehicle_id in surrounding_vehicles.keys():
            state['vehicle'][vehicle_id]['surround'] = surrounding_vehicles[vehicle_id]
            state['vehicle'][vehicle_id]['surround_expand'] = sorrounding_vehicles_expand[vehicle_id]
            self.surround_vehicle[vehicle_id] = surrounding_vehicles[vehicle_id]

        return state

    def append_surrounding_ITSCversion(self, state):
        surrounding_vehicles = {}
        """ append_surrounding(ITSC version)
        ^ x (-)
        |
        |
        ------> y (+)
        | 
        |
        v x (-)

        [surround_vehicle_id, relative x, relative y, relative speed]
        """
        for vehicle_id in state['vehicle'].keys():
            # 对于所有RL控制的车辆
            if vehicle_id in self.ego_ids:
                if self.use_gui:
                    import traci as traci
                else:
                    import libsumo as traci
                surrounding_vehicle = {}

                modes_follow = {
                    'left_followers': 0b000,  # Left and followers
                    'right_followers': 0b001,  # Left and leaders
                }
                # ego车的左右车道的后面的车辆
                for key, mode in modes_follow.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    for n in neighbors:
                        if n[0][:3] == 'HDV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是HDV）
                            # 3.2是lane的宽度，1.5是HDV的minGap
                            surrounding_vehicle[key] = (n[0], -3.2, -(n[1] + 1.5), relative_speed)
                        elif n[0][:3] == 'CAV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是CAV）
                            # 3.2是lane的宽度，1.0是CAV的minGap
                            surrounding_vehicle[key] = (n[0], -3.2, -(n[1] + 1.0), relative_speed)
                        else:
                            raise ValueError('Unknown vehicle type')

                modes_lead = {
                    'left_leaders': 0b010,  # Right and followers
                    'right_leaders': 0b011  # Right and leaders
                }
                # ego车的左右车道的前面的车辆
                for key, mode in modes_lead.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    for n in neighbors:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(n[0])
                        # traci得到的距离是ego到前车尾部减去minGap的距离，所以要加上minGap（此时后车是ego）
                        # 3.2是lane的宽度，1.0是CAV的minGap
                        surrounding_vehicle[key] = (n[0], -3.2, n[1] + 1.0, relative_speed)

                # 在当前车道上的前车
                front_vehicle = traci.vehicle.getLeader(vehicle_id)
                if front_vehicle not in [None, ()] and front_vehicle[0] != '':  # 有可能是空的
                    front_vehicle_lane = traci.vehicle.getLaneID(front_vehicle[0])
                    front_vehicle_lane_index = int(front_vehicle_lane.split('_')[-1])
                    front_vehicle_road_id = front_vehicle_lane.split('_')[0]

                    ego_lane = state['vehicle'][vehicle_id]['lane_id']
                    ego_lane_index = int(ego_lane.split('_')[-1])
                    ego_road_id = ego_lane.split('_')[0]
                    # if front_vehicle_road_id[:3] == ':J3' or ego_road_id[:3] == ':J3':
                    #     print('debug')
                    if front_vehicle_lane_index != ego_lane_index:
                        pass
                    else:
                        # 相对速度 - ego车的速度 - 前车的速度的差值
                        relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(front_vehicle[0])
                        # traci得到的距离是ego到前车尾部减去minGap的距离，所以要加上minGap（此时后车是ego）
                        surrounding_vehicle['front'] = (front_vehicle[0], 0, front_vehicle[1] + 1.0, relative_speed)

                # 在当前车道上的后车
                back_vehicle = traci.vehicle.getFollower(vehicle_id)
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
                        if back_vehicle[0][:3] == 'HDV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(
                                back_vehicle[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是HDV）
                            surrounding_vehicle['back'] = (back_vehicle[0], 0, -(back_vehicle[1] + 1.5), relative_speed)
                        elif back_vehicle[0][:3] == 'CAV':
                            # 相对速度 - ego车的速度 - 后车的速度的差值
                            relative_speed = traci.vehicle.getSpeed(vehicle_id) - traci.vehicle.getSpeed(
                                back_vehicle[0])
                            # traci得到的距离是ego到后车头部减去minGap的距离，所以要加上minGap（此时后车是CAV）
                            surrounding_vehicle['back'] = (back_vehicle[0], 0, -(back_vehicle[1] + 1.0), relative_speed)

                surrounding_vehicles[vehicle_id] = surrounding_vehicle

                pass
        for vehicle_id in surrounding_vehicles.keys():
            state['vehicle'][vehicle_id]['surround'] = surrounding_vehicles[vehicle_id]

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
        for vehicle_id in self.surround_vehicle.keys():
            for key in self.required_surroundings:
                if key not in self.surround_vehicle[vehicle_id]:
                    self.surround_vehicle[vehicle_id][key] = 0

        for _veh_id in raw_action:
            if _veh_id in self.actions:  # 只更新 ego vehicle 的速度
                # 只有换道动作，不需要更新速度
                if raw_action[_veh_id] in range(0, 3):
                    self.action_command = (self.action_pointer[raw_action[_veh_id]][0], self.current_speed[_veh_id])

                elif raw_action[_veh_id] in range(3, 5):
                    if raw_action[_veh_id] == 3:
                        # the speed should not exceed 15
                        self.action_command = (0, min(15, self.current_speed[_veh_id] + self.delta_t * 2))
                    else:
                        # the speed should not be negative
                        self.action_command = (0, max(0, self.current_speed[_veh_id] - self.delta_t * 2))
                else:
                    raise ValueError(f'Action {raw_action[_veh_id]} is not in the range of 0-5')

                self.actions[_veh_id] = self.lower_controller(_veh_id, self.action_command, 2)
                # self.actions[_veh_id] = self.action_command

        return self.actions
    def lower_controller_0712(self, _veh_id, action, acc_improve):
        LC_th = [2, 5]
        keep_th = [1.5, 3, 20]
        bottleneck_th = [1, 3, 10]
        bottleneck_point = self.bottle_neck_positions[0] - 20
        bottleneck_start = self.bottle_neck_positions[0] - 150
        bottleneck_end = bottleneck_point
        LC_improve = 0
        if self.vehicles_info:
            if action[0] != 0:
                if action[0] == 1:
                    leader_dis = self.surround_vehicle[_veh_id]['left_leaders'][1] if self.surround_vehicle[_veh_id]['left_leaders'] else 100
                    leader_v = self.surround_vehicle[_veh_id]['left_leaders'][3] if self.surround_vehicle[_veh_id]['left_leaders'] else 1
                    follower_dis = self.surround_vehicle[_veh_id]['left_followers'][1] if self.surround_vehicle[_veh_id]['left_followers'] else -100
                    follower_v = self.surround_vehicle[_veh_id]['left_followers'][3] if self.surround_vehicle[_veh_id]['left_followers'] else -1
                    leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                    if leader_TTC < 0 and leader_dis > 0:
                        leader_TTC = 100
                    elif leader_TTC > 0 and leader_dis < 0:
                        leader_TTC = -leader_TTC
                    follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                    if follower_TTC < 0 and follower_dis < 0:
                        follower_TTC = 100
                    elif follower_TTC > 0 and follower_dis > 0:
                        follower_TTC = -follower_TTC
                else:
                    leader_dis = self.surround_vehicle[_veh_id]['right_leaders'][1] if self.surround_vehicle[_veh_id]['right_leaders'] else 100
                    leader_v = self.surround_vehicle[_veh_id]['right_leaders'][3] if self.surround_vehicle[_veh_id]['right_leaders'] else 1
                    follower_dis = self.surround_vehicle[_veh_id]['right_followers'][1] if self.surround_vehicle[_veh_id]['right_followers'] else -100
                    follower_v = self.surround_vehicle[_veh_id]['right_followers'][3] if self.surround_vehicle[_veh_id]['right_followers'] else -1
                    leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                    if leader_TTC < 0 and leader_dis > 0:
                        leader_TTC = 100
                    elif leader_TTC > 0 and leader_dis < 0:
                        leader_TTC = -leader_TTC
                    follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                    if follower_TTC < 0 and follower_dis < 0:
                        follower_TTC = 100
                    elif follower_TTC > 0 and follower_dis > 0:
                        follower_TTC = -follower_TTC
                if leader_TTC <= LC_th[0] or follower_TTC <= LC_th[0]:
                    if self.vehicles_info[_veh_id][4] > bottleneck_start and self.vehicles_info[_veh_id][4] < bottleneck_end:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * acc_improve))
                        LC_improve = 1
                    else:
                        action = (0, self.current_speed[_veh_id])
                elif leader_TTC <= LC_th[1]:
                    if follower_dis < -2:
                        action = (action[0], max(0, self.current_speed[_veh_id]-self.delta_t*max(acc_improve, leader_v/leader_TTC+1)))
                    else:
                        action = (0, self.current_speed[_veh_id])
                elif follower_TTC <= LC_th[1]:
                    action = (action[0], min(15, self.current_speed[_veh_id] - self.delta_t * follower_v))
                else:
                    action = action
            if action[0] == 0:
                if self.vehicles_info[_veh_id][4] > bottleneck_point and abs(self.vehicles_info[_veh_id][5]) > 2:
                    if self.vehicles_info[_veh_id][5] < -2:
                        if self.surround_vehicle[_veh_id]['front']:
                            leader_dis = self.surround_vehicle[_veh_id]['front'][1]
                            leader_v = self.surround_vehicle[_veh_id]['front'][3]
                            follower_dis = -100
                            follower_v = -1
                        else:
                            if self.surround_vehicle[_veh_id]['left_leaders']:
                                leader_dis = self.surround_vehicle[_veh_id]['left_leaders'][1]
                                leader_v = self.surround_vehicle[_veh_id]['left_leaders'][3]
                            else:
                                leader_dis = 100
                                leader_v = 1
                            if self.surround_vehicle[_veh_id]['left_followers']:
                                follower_dis = self.surround_vehicle[_veh_id]['left_followers'][1]
                                follower_v = self.surround_vehicle[_veh_id]['left_followers'][3]
                            else:
                                follower_dis = -100
                                follower_v = -1
                        leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                        leader_speed = self.current_speed[_veh_id] - self.delta_t * leader_v
                        leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                        follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                        if leader_TTC < 0 and leader_dis > 5 and leader_speed > 10:
                            leader_TTC = 100
                        elif leader_TTC > 0 and leader_dis < 0:
                            leader_TTC = -leader_TTC
                        follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                        if follower_TTC < 0 and follower_dis < -10:
                            follower_TTC = 100
                        elif follower_TTC > 0 and follower_dis > 0:
                            follower_TTC = -follower_TTC
                    else:
                        if self.surround_vehicle[_veh_id]['front']:
                            leader_dis = self.surround_vehicle[_veh_id]['front'][1]
                            leader_v = self.surround_vehicle[_veh_id]['front'][3]
                            follower_dis = 100
                            follower_v = 1
                        else:
                            if self.surround_vehicle[_veh_id]['right_leaders']:
                                leader_dis = self.surround_vehicle[_veh_id]['right_leaders'][1]
                                leader_v = self.surround_vehicle[_veh_id]['right_leaders'][3]
                            else:
                                leader_dis = 100
                                leader_v = 1
                            if self.surround_vehicle[_veh_id]['right_followers']:
                                follower_dis = self.surround_vehicle[_veh_id]['right_followers'][1]
                                follower_v = self.surround_vehicle[_veh_id]['right_followers'][3]
                            else:
                                follower_dis = -100
                                follower_v = -1
                        leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                        leader_speed = self.current_speed[_veh_id] - self.delta_t * leader_v
                        if leader_TTC < 0 and leader_dis > 5 and leader_speed > 10:
                            leader_TTC = 100
                        elif leader_TTC > 0 and leader_dis < 0:
                            leader_TTC = -leader_TTC
                        follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                        if follower_TTC < 0 and follower_dis > 10:
                            follower_TTC = 100
                        elif follower_TTC > 0 and follower_dis < 0:
                            follower_TTC = -follower_TTC
                    if leader_TTC <= bottleneck_th[0] or follower_TTC <= bottleneck_th[0]:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * 7))
                    elif leader_TTC <= bottleneck_th[1]:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * 3))
                    else:
                        action = action
                else:
                    leader_dis = self.surround_vehicle[_veh_id]['front'][1] if self.surround_vehicle[_veh_id]['front'] else 100
                    leader_v = self.surround_vehicle[_veh_id]['front'][3] if self.surround_vehicle[_veh_id]['front'] else 1
                    leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                    follower_TTC = 100
                    if leader_TTC < 0 and leader_dis > 0:
                        leader_TTC = 100
                    elif leader_TTC > 0 and leader_dis < 0:
                        leader_TTC = -leader_TTC
                    if leader_TTC < keep_th[0]:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(leader_v/leader_TTC+2, acc_improve)))
                    elif leader_TTC < keep_th[1]:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(leader_v/leader_TTC, acc_improve)))
                    elif leader_TTC < keep_th[2]:
                        if leader_dis > 10:
                            if LC_improve:
                                action = action
                            else:
                                action = (0, self.current_speed[_veh_id]) if action[1] > self.current_speed[_veh_id] else action
                        else:
                            action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(leader_v/leader_TTC, acc_improve)))
                    else:
                        action = action

            self.safety_before[_veh_id].append(max(1 / leader_TTC, 1 / follower_TTC))

        return action

    def lower_controller_0714(self, _veh_id, action, acc_improve):
        ### 下层控制器--基于安全评估的控制微调
        bottleneck_point = self.bottle_neck_positions[0] - 20
        bottleneck_th = [2, 5, 10]
        if action[0] == 1:
            if self.surround_vehicle[_veh_id]['left_leaders']:
                relative_distance = self.surround_vehicle[_veh_id]['left_leaders'][1]
                relative_speed = self.surround_vehicle[_veh_id]['left_leaders'][3]
                if relative_distance <= 1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance <= 5:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            action = (1, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed)))
                elif relative_distance <= 15:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (1, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed)))
            if self.surround_vehicle[_veh_id]['left_followers']:
                relative_distance = self.surround_vehicle[_veh_id]['left_followers'][1]
                relative_speed = self.surround_vehicle[_veh_id]['left_followers'][3]
                if relative_distance >= -1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -5:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -15:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
        if action[0] == 2:
            if self.surround_vehicle[_veh_id]['right_leaders']:
                relative_distance = self.surround_vehicle[_veh_id]['right_leaders'][1]
                relative_speed = self.surround_vehicle[_veh_id]['right_leaders'][3]
                if relative_distance <= 1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance <= 5:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            action = (2, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed)))
                elif relative_distance <= 15:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (2, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed)))
            if self.surround_vehicle[_veh_id]['right_followers']:
                relative_distance = self.surround_vehicle[_veh_id]['right_followers'][1]
                relative_speed = self.surround_vehicle[_veh_id]['right_followers'][3]
                if relative_distance >= -1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -5:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -15:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
        if action[0] == 0:
            if self.surround_vehicle[_veh_id]['front']:
                relative_distance = self.surround_vehicle[_veh_id]['front'][1]
                relative_speed = self.surround_vehicle[_veh_id]['front'][3]
                if relative_distance <= 8:
                    if relative_speed > 0:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed + 2)))
                    elif self.vehicles_info[_veh_id][4] > bottleneck_point and relative_speed > -2:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * acc_improve))
                    elif action[1] > self.current_speed[_veh_id]:
                        action = (0, self.current_speed[_veh_id])
                elif relative_distance <= 15:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 3:
                            action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed)))
                        elif time_collision <= 5:
                            if action[1] > self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
            else:
                if self.vehicles_info:
                    if self.vehicles_info[_veh_id][4] > bottleneck_point and abs(self.vehicles_info[_veh_id][5]) > 2:
                        if self.vehicles_info[_veh_id][5] < -2:
                            if self.surround_vehicle[_veh_id]['front']:
                                leader_dis = self.surround_vehicle[_veh_id]['front'][1]
                                leader_v = self.surround_vehicle[_veh_id]['front'][3]
                                follower_dis = -100
                                follower_v = -1
                            else:
                                if self.surround_vehicle[_veh_id]['left_leaders']:
                                    leader_dis = self.surround_vehicle[_veh_id]['left_leaders'][1]
                                    leader_v = self.surround_vehicle[_veh_id]['left_leaders'][3]
                                else:
                                    leader_dis = 100
                                    leader_v = 1
                                if self.surround_vehicle[_veh_id]['left_followers']:
                                    follower_dis = self.surround_vehicle[_veh_id]['left_followers'][1]
                                    follower_v = self.surround_vehicle[_veh_id]['left_followers'][3]
                                else:
                                    follower_dis = -100
                                    follower_v = -1
                            leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                            leader_speed = self.current_speed[_veh_id] - self.delta_t * leader_v
                            if leader_TTC < 0 and leader_dis > 5 and leader_speed > 10:
                                leader_TTC = 100
                            elif leader_TTC > 0 and leader_dis < 0:
                                leader_TTC = -leader_TTC
                            follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                            if follower_TTC < 0 and follower_dis < -10:
                                follower_TTC = 100
                            elif follower_TTC > 0 and follower_dis > 0:
                                follower_TTC = -follower_TTC
                        else:
                            if self.surround_vehicle[_veh_id]['front']:
                                leader_dis = self.surround_vehicle[_veh_id]['front'][1]
                                leader_v = self.surround_vehicle[_veh_id]['front'][3]
                                follower_dis = 100
                                follower_v = 1
                            else:
                                if self.surround_vehicle[_veh_id]['right_leaders']:
                                    leader_dis = self.surround_vehicle[_veh_id]['right_leaders'][1]
                                    leader_v = self.surround_vehicle[_veh_id]['right_leaders'][3]
                                else:
                                    leader_dis = 100
                                    leader_v = 1
                                if self.surround_vehicle[_veh_id]['right_followers']:
                                    follower_dis = self.surround_vehicle[_veh_id]['right_followers'][1]
                                    follower_v = self.surround_vehicle[_veh_id]['right_followers'][3]
                                else:
                                    follower_dis = -100
                                    follower_v = -1
                            leader_TTC = leader_dis / leader_v if leader_v != 0 else abs(leader_dis)
                            leader_speed = self.current_speed[_veh_id] - self.delta_t * leader_v
                            if leader_TTC < 0 and leader_dis > 5 and leader_speed > 10:
                                leader_TTC = 100
                            elif leader_TTC > 0 and leader_dis < 0:
                                leader_TTC = -leader_TTC
                            follower_TTC = follower_dis / follower_v if follower_v != 0 else abs(follower_dis)
                            if follower_TTC < 0 and follower_dis > 10:
                                follower_TTC = 100
                            elif follower_TTC > 0 and follower_dis < 0:
                                follower_TTC = -follower_TTC
                        if leader_TTC <= bottleneck_th[0] or follower_TTC <= bottleneck_th[0]:
                            action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * 7))
                        elif leader_TTC <= bottleneck_th[1]:
                            action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * 3))
                        else:
                            action = (0, self.current_speed[_veh_id])
        return action

    def lower_controller(self, _veh_id, action, acc_improve):
        ### 下层控制器--基于安全评估的控制微调
        if action[0] == 1:
            if self.surround_vehicle[_veh_id]['left_leaders']:
                relative_distance = self.surround_vehicle[_veh_id]['left_leaders'][1]
                relative_speed = self.surround_vehicle[_veh_id]['left_leaders'][3]
                if relative_distance <= 1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance <= 5:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            action = (1, max(0, self.current_speed[_veh_id]-self.delta_t*max(acc_improve, relative_speed)))
                elif relative_distance <= 15:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (1, max(0, self.current_speed[_veh_id]-self.delta_t*max(acc_improve, relative_speed)))
            if self.surround_vehicle[_veh_id]['left_followers']:
                relative_distance = self.surround_vehicle[_veh_id]['left_followers'][1]
                relative_speed = self.surround_vehicle[_veh_id]['left_followers'][3]
                if relative_distance >= -1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -5:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -15:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
        if action[0] == 2:
            if self.surround_vehicle[_veh_id]['right_leaders']:
                relative_distance = self.surround_vehicle[_veh_id]['right_leaders'][1]
                relative_speed = self.surround_vehicle[_veh_id]['right_leaders'][3]
                if relative_distance <= 1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance <= 5:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            action = (2, max(0, self.current_speed[_veh_id]-self.delta_t*max(acc_improve, relative_speed)))
                elif relative_distance <= 15:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (2, max(0, self.current_speed[_veh_id]-self.delta_t*max(acc_improve, relative_speed)))
            if self.surround_vehicle[_veh_id]['right_followers']:
                relative_distance = self.surround_vehicle[_veh_id]['right_followers'][1]
                relative_speed = self.surround_vehicle[_veh_id]['right_followers'][3]
                if relative_distance >= -1:
                    action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -5:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            action = (0, self.current_speed[_veh_id])
                        elif time_collision <= 5:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
                elif relative_distance >= -15:
                    if relative_speed < 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 2:
                            if action[1] != self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
        if action[0] == 0:
            if self.surround_vehicle[_veh_id]['front']:
                relative_distance = self.surround_vehicle[_veh_id]['front'][1]
                relative_speed = self.surround_vehicle[_veh_id]['front'][3]
                if relative_distance <= 5:
                    if relative_speed > 0:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed+2)))
                elif relative_distance <= 8:
                    if relative_speed > 0:
                        action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed+2)))
                    else:
                        if action[1] > self.current_speed[_veh_id]:
                            action = (0, self.current_speed[_veh_id])
                elif relative_distance <= 15:
                    if relative_speed > 0:
                        time_collision = relative_distance / relative_speed
                        if time_collision <= 3:
                            action = (0, max(0, self.current_speed[_veh_id] - self.delta_t * max(acc_improve, relative_speed)))
                        elif time_collision <= 5:
                            if action[1] > self.current_speed[_veh_id]:
                                action = (0, self.current_speed[_veh_id])
                    # elif relative_speed < 0:
                    #     action = (0, min(15, self.current_speed[_veh_id] + self.delta_t * acc_improve))
                # else:
                #     if relative_speed <= 0:
                #         action = (0, min(15, self.current_speed[_veh_id] + self.delta_t * acc_improve))
            # else:
            #     action = (0, min(15, self.current_speed[_veh_id] + self.delta_t * acc_improve))
        return action

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
            state=state, lane_ids=self.calc_features_lane_ids
        )

        # 计算每个 ego vehicle 的 state 拼接为向量
        if self.strategy == 'base':
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = compute_base_ego_vehicle_features(
                self,
                hdv_statistics=hdv_statistics,
                lane_statistics=lane_statistics,
                ego_statistics=ego_statistics,
                unique_edges=self.edge_ids,
                edge_lane_num=self.edge_lane_num,
                bottle_neck_positions=self.bottle_neck_positions,
                ego_ids=self.ego_ids,
            )
        elif self.strategy == 'hierarchical' or self.strategy == 'prediction' or self.strategy == 'cross_aware':
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = compute_hierarchical_ego_vehicle_features(
                self,
                hdv_statistics=hdv_statistics,
                lane_statistics=lane_statistics,
                ego_statistics=ego_statistics,
                unique_edges=self.edge_ids,
                edge_lane_num=self.edge_lane_num,
                bottle_neck_positions=self.bottle_neck_positions,
                ego_ids=self.ego_ids,
            )

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
        max_speed = 15  # ego vehicle 的最快的速度

        # 先把reward_statistics中所有的车辆的信息都全局记录下来self.vehicles_info
        for veh_id, (road_id, distance, speed, position_x, position_y, waiting_time, accumulated_waiting_time) in reward_statistics.items():
            self.vehicles_info[veh_id] = [
                self.vehicles_info.get(veh_id, [0, None])[0] + 1,  # travel time
                road_id,
                distance,
                speed,
                position_x,
                position_y,
                waiting_time,
                accumulated_waiting_time
            ]
        # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
        if len(self.out_of_road) > 0:
            for veh_id in self.out_of_road:
                if veh_id in self.vehicles_info:
                    del self.vehicles_info[veh_id]

        # ######################### 开始计算reward  # #########################
        inidividual_rew_ego = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}
        range_reward_ego = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}
        bottleneck_reward_ego = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}
        time_penalty_ego = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}
        is_in_bottleneck = {key: 0 for key in list(set(self.ego_ids) - set(self.out_of_road))}

        # ######################## 初始化 for group reward ########################
        all_ego_vehicle_speed = []  # CAV车辆的平均速度 - 使用target speed
        all_ego_vehicle_mean_speed = []  # CAV车辆的累积平均速度 - 使用速度/时间
        all_ego_vehicle_accumulated_waiting_time = []  # # CAV车辆的累积平均等待时间
        all_ego_vehicle_waiting_time = []  # CAV车辆的等待时间
        all_ego_vehicle_position_x = []  # CAV车辆的位置

        all_vehicle_speed = []  # CAV和HDV车辆的平均速度 - 使用target speed
        all_vehicle_mean_speed = []  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
        all_vehicle_accumulated_waiting_time = []  # CAV和HDV车辆的累积平均等待时间
        all_vehicle_waiting_time = []  # CAV和HDV车辆的等待时间
        all_vehicle_position_x = []  # CAV和HDV车辆的位置

        range_vehicle_speed = []  # bottleneck+ 区域的车辆的速度
        range_vehicle_waiting_time = []  # bottleneck+ 区域的车辆的等待时间
        range_vehicle_accumulated_waiting_time = []  # bottleneck+ 区域的车辆的累积等待时间
        range_ego_vehicle_speed = []  # bottleneck+ 区域的CAV车辆的速度
        range_ego_vehicle_waiting_time = []  # bottleneck+ 区域的CAV车辆的等待时间
        range_ego_vehicle_accumulated_waiting_time = []  # bottleneck+ 区域的CAV车辆的累积等待时间

        for veh_id, (veh_travel_time, road_id, distance, speed, position_x,
                     position_y, waiting_time, accumulated_waiting_time) in list(self.vehicles_info.items()):

            # CAV和HDV车辆的
            all_vehicle_speed.append(speed)
            all_vehicle_mean_speed.append(distance / veh_travel_time)
            all_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
            all_vehicle_waiting_time.append(waiting_time)
            all_vehicle_position_x.append(position_x)
            if road_id in self.bottle_necks + ['E2', 'E3']:
                range_vehicle_speed.append(speed)
                range_vehicle_waiting_time.append(waiting_time)
                range_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)

            # 把CAV单独取出来
            if veh_id in self.ego_ids:

                # for group reward 计算CAV车辆的累积平均速度
                all_ego_vehicle_speed.append(speed)
                all_ego_vehicle_mean_speed.append(distance / veh_travel_time)
                all_ego_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
                all_ego_vehicle_waiting_time.append(waiting_time)
                all_ego_vehicle_position_x.append(position_x)
                if road_id in self.bottle_necks + ['E2', 'E3']:
                    range_ego_vehicle_speed.append(speed)
                    range_ego_vehicle_waiting_time.append(waiting_time)
                    range_ego_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)

                # ######################## for individual reward ########################
                # # CAV车辆的累积平均速度越靠近最大速度，reward越高 - [0, 5]
                # individual_speed_r = -abs(distance / veh_travel_time - max_speed) / max_speed * 5 + 5
                # inidividual_rew_ego[veh_id] += 1 * individual_speed_r

                # CAV车辆的target速度越靠近最大速度，reward越高 - [0, 5]
                individual_speed_r_simple = (-abs(speed - max_speed) / max_speed * 5 + 5) * 1
                inidividual_rew_ego[veh_id] += individual_speed_r_simple

                # # CAV车辆的等待时间越短，reward越高 - (-infty,0]
                # individual_accumulated_waiting_time_r = -accumulated_waiting_time
                # inidividual_rew_ego[veh_id] += individual_accumulated_waiting_time_r
                # individual_waiting_time_r = -waiting_time
                # inidividual_rew_ego[veh_id] += individual_waiting_time_r * 1

                ## reward is higher when the vehicle is closer to the end of the road
                # individual_position_r = position_x / self.bottle_neck_positions[0]
                # inidividual_rew_ego[veh_id] += individual_position_r * 1

                # 警告距离和碰撞距离
                if veh_id in self.warn_ego_ids.keys():
                    individual_warn_r = 0
                    for dis in self.warn_ego_ids[veh_id]:
                        # CAV车辆的警告距离越远，reward越高 - [0, 5]
                        individual_warn_r += -(WARN_GAP_THRESHOLD - dis) / (
                                WARN_GAP_THRESHOLD - GAP_THRESHOLD) * 10  # [-10, 0]
                    inidividual_rew_ego[veh_id] += individual_warn_r * 0.05

                if veh_id in self.coll_ego_ids.keys():
                    individual_coll_r = 0
                    for dis in self.coll_ego_ids[veh_id]:
                        # CAV车辆的碰撞距离越远，reward越高 - [0, 5]
                        individual_coll_r += -(GAP_THRESHOLD - dis) / GAP_THRESHOLD * 20 - 10  # [-30, -10]
                    inidividual_rew_ego[veh_id] += individual_coll_r * 0.05

                # 计算局部地区的reward
                if road_id in self.bottle_necks + ['E3', 'E2', 'E1', 'E0']:
                    # 快速穿过bottleneck区域 和最后一个lane
                    individual_botte_neck_r = -abs(speed - max_speed) / max_speed * 5 + 5
                    range_reward_ego[veh_id] = individual_botte_neck_r
                else:
                    range_reward_ego[veh_id] = 0

                if road_id in self.bottle_necks+['E2']:
                    is_in_bottleneck[veh_id] = 1
                else:
                    is_in_bottleneck[veh_id] = 0

                if inidividual_rew_ego[veh_id] == individual_speed_r_simple:
                    time_penalty_ego[veh_id] = ((self.num_CAVs-1)+self.CAV_penetration) * 10 * (1 / (1 + np.exp(5-speed)) - 1)
                    # time_penalty_ego[veh_id] = 100 * (math.tanh(speed) - 1)
                else:
                    time_penalty_ego[veh_id] = 0

        # 计算全局reward
        all_ego_vehicle_speed = np.mean(all_ego_vehicle_speed)  # CAV车辆的平均速度 - 使用target speed
        # all_ego_mean_speed = np.mean(all_ego_vehicle_mean_speed)  # CAV车辆的累积平均速度 - 使用速度/时间
        # all_ego_vehicle_accumulated_waiting_time = np.mean(all_ego_vehicle_accumulated_waiting_time)  # CAV车辆的累积平均等待时间
        # all_ego_vehicle_waiting_time = np.mean(all_ego_vehicle_waiting_time)  # CAV车辆的mean等待时间
        # all_ego_vehicle_position_x = np.mean(all_ego_vehicle_position_x)  # CAV车辆的位置

        # all_vehicle_speed = np.mean(all_vehicle_speed)  # CAV和HDV车辆的平均速度 - 使用target speed
        # all_vehicle_mean_speed = np.mean(all_vehicle_mean_speed)  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
        # all_vehicle_accumulated_waiting_time = np.mean(all_vehicle_accumulated_waiting_time)  # CAV和HDV车辆的累积平均等待时间
        # all_vehicle_waiting_time = np.mean(all_vehicle_waiting_time)  # CAV和HDV车辆的mean等待时间
        # all_vehicle_position_x = np.mean(all_vehicle_position_x)  # CAV和HDV车辆的位置

        global_ego_speed_r = -abs(all_ego_vehicle_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_ego_mean_speed_r = -abs(all_ego_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_ego_accumulated_waiting_time_r = -all_ego_vehicle_accumulated_waiting_time   # (-infty,0]
        # global_ego_waiting_time_r = -all_ego_vehicle_waiting_time   # (-infty,0]
        # global_ego_position_x_r = all_ego_vehicle_position_x / self.bottle_neck_positions[0]

        # global_all_speed_r = -abs(all_vehicle_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_all_mean_speed_r = -abs(all_vehicle_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_all_accumulated_waiting_time_r = -all_vehicle_accumulated_waiting_time   # [0, 5]
        # global_all_waiting_time_r = -all_vehicle_waiting_time   # (-infty,0]
        # global_all_position_x_r = all_vehicle_position_x / self.bottle_neck_positions[0]

        range_vehicle_speed = np.mean(range_vehicle_speed) if range_vehicle_speed != [] else 0  # bottleneck+ 区域的车辆的速度
        # range_vehicle_waiting_time = np.mean(range_vehicle_waiting_time) if range_vehicle_waiting_time != [] else 0  # bottleneck+ 区域的车辆的等待时间
        # range_vehicle_accumulated_waiting_time = np.mean(range_vehicle_accumulated_waiting_time) if range_vehicle_accumulated_waiting_time != [] else 0  # bottleneck+ 区域的车辆的累积等待时间
        # range_ego_vehicle_speed = np.mean(range_ego_vehicle_speed) if range_ego_vehicle_speed != [] else 0  # bottleneck+ 区域的CAV车辆的速度
        # range_ego_vehicle_waiting_time = np.mean(range_ego_vehicle_waiting_time) if range_ego_vehicle_waiting_time != [] else 0  # bottleneck+ 区域的CAV车辆的等待时间
        # range_ego_vehicle_accumulated_waiting_time = np.mean(range_ego_vehicle_accumulated_waiting_time) if range_ego_vehicle_accumulated_waiting_time != [] else 0

        for veh_id in inidividual_rew_ego.keys():
            if veh_id not in self.vehicles_info.keys():
                print("Error: veh_id not in vehicles_info")
            road_id = self.vehicles_info[veh_id][1]
            if road_id in self.bottle_necks + ['E2', 'E3']:
                range_vehs_speed_r = -abs(range_vehicle_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
                # range_vehs_waiting_time_r = -range_vehicle_waiting_time  # (-infty,0]
                # range_vehs_accumulated_waiting_time_r = -range_vehicle_accumulated_waiting_time  # (-infty,0]
                # range_ego_speed_r = -abs(range_ego_vehicle_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
                # range_ego_waiting_time_r = -range_ego_vehicle_waiting_time  # (-infty,0]
                # range_ego_accumulated_waiting_time_r = -range_ego_vehicle_accumulated_waiting_time  # (-infty,0]
            else:
                range_vehs_speed_r = 0
                # range_vehs_waiting_time_r = 0
                # range_vehs_accumulated_waiting_time_r = 0
                # range_ego_speed_r = 0
                # range_ego_waiting_time_r = 0
                # range_ego_accumulated_waiting_time_r = 0
            bottleneck_reward_ego[veh_id] += range_vehs_speed_r
                                            # + range_ego_speed_r
                                            # + range_ego_waiting_time_r \
                                            # + range_ego_accumulated_waiting_time_r
                                            # + range_vehs_speed_r \
                                            # + range_vehs_waiting_time_r \
                                            # + range_vehs_accumulated_waiting_time_r

        time_penalty = -1

        # TODO： lane_statistics  在E2的等待时间
        # TODO: 完成时间越短，reward越高 - [0, 5]

        # rewards = {key: inidividual_rew_ego[key] + range_reward_ego[key] + global_speed_r + global_waiting_time_r for
        #            key in inidividual_rew_ego}
        # rewards = {key: inidividual_rew_ego[key] + range_reward_ego[key] + global_speed_r + global_waiting_time_r + \
        #               global_ego_speed_r + global_ego_waiting_time_r for key in inidividual_rew_ego}

        rewards = {key: (1-self.CAV_penetration) * inidividual_rew_ego[key] \
                        # + range_reward_ego[key] \
                        # + is_in_bottleneck[key] * bottleneck_reward_ego[key] \
                        + time_penalty_ego[key] \
                        + self.CAV_penetration * global_ego_speed_r \
                        # + global_ego_mean_speed_r \
                        # + global_ego_waiting_time_r \
                        # + global_ego_accumulated_waiting_time_r \
                        # + global_ego_position_x_r \
                        # + global_all_speed_r \
                        # + global_all_mean_speed_r \
                        # + global_all_waiting_time_r \
                        # + global_all_accumulated_waiting_time_r \
                        # + global_all_position_x_r \
                        + time_penalty
                   for key in inidividual_rew_ego}

        return rewards

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
                          CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
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
            else:
                if self.use_hist_info:
                    for i in range(self.hist_length-1):
                        self.action_generation[f'hist_{self.hist_length - i}'][ego_id] = \
                        self.action_generation[f'hist_{self.hist_length - i - 1}'][ego_id]
                    self.action_generation['hist_1'][ego_id] = action[ego_id]

        # 更新 action
        action = self.__update_actions(raw_action=action).copy()
        for ego_id, ego_live in self.agent_mask.items():
            if ego_live:
                if self.use_hist_info:
                    for i in range(self.hist_length-1):
                        self.action_execution[f'hist_{self.hist_length - i}'][ego_id] = \
                        self.action_execution[f'hist_{self.hist_length - i - 1}'][ego_id]
                    self.action_execution['hist_1'][ego_id] = action[ego_id]
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
            rewards = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)

        elif len(self.coll_ego_ids) > 0 and len(feature_vectors) > 0:  # 还有车在路上 但有车辆碰撞
            # 计算此时的reward
            rewards = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)  # 更新 veh info
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
                rewards.update({out_of_road_ego_id: 20.0})  # 离开路网之后 reward 也是 0  # TODO: 注意一下dead mask MARL
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

        return feature_vectors_flatten, shared_features_flatten, rewards, dones.copy(), dones.copy(), infos

    def close(self) -> None:
        return super().close()

    # TODO: MARL最好知道上一刻的动作
