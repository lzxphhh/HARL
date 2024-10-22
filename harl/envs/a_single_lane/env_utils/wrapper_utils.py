import itertools
import math
from typing import List, Dict, Tuple, Union
import copy

import numpy as np
import collections.abc
from itertools import chain
import time

LANE_LENGTHS = {
    'E0': 500,
    'E1': 2000,
    # 'E2': 100,
    # 'E3': 100,
}

state_divisors = np.array([2000, 3, 20, 3, 360, 6, 20])
# Lane_start = {
#     'E0': (0, 0),
#     'E1': (100, 0),
#     'E2': (100, 100),
#     'E3': (0, 100),
# }
# Lane_end = {
#     'E0': (100, 0),
#     'E1': (100, 100),
#     'E2': (0, 100),
#     'E3': (0, 0),
# }
Lane_start = {
    'E0': (-500, 0),
    'E1': (0, 0),
}
Lane_end = {
    'E0': (0, 0),
    'E1': (2000, 0),
}

def analyze_traffic(state, lane_ids, max_veh_num):
    """
    输入：当前所有车的state  &&   需要计算feature的lane id
    输出：
    1. lane_statistics: 每一个 lane 的统计信息 - Dict, key: lane_id, value:
        - vehicle_count: 当前车道的车辆数 1
        - lane_density: 当前车道的车辆密度 1
        - lane_length: 这个车道的长度 1
        - speeds: 在这个车道上车的速度 3 (mean, max, min)
        - waiting_times: 一旦车辆开始行驶，等待时间清零 3  (mean, max, min)
        - accumulated_waiting_times: 车的累积等待时间 3 (mean, max, min)

    2. ego_statistics: ego vehicle 的统计信息
        - speed: vehicle当前车速
        - position: vehicle当前位置
        - heading: vehicle当前朝向
        - road_id: vehicle当前所在道路的 ID
        - lane_index: vehicle当前所在车道的 index
        - surroundings: vehicle周围车辆的相对位置

    3. reward_statistics: Dict, key: vehicle_id, value:
        - 这个车行驶道路的 ID. eg: 'E0'
        - 累积行驶距离
        - 当前车速
        - position[0]: 1
        - position[1]: 1
        - waiting_time: 1

    """

    # 初始化每一个 lane 的统计信息
    lane_statistics = {
        lane_id: {
            'vehicle_count': 0,  # 当前车道的车辆数
            'lane_density': 0,  # 当前车道的车辆密度
            'speeds': [],  # 在这个车道上车的速度
            'accelerations': [], # 在这个车道上车的加速度
            'waiting_times': [],  # 一旦车辆开始行驶，等待时间清零
            'accumulated_waiting_times': [],  # 车的累积等待时间
            # 'crash_assessment': 0,  # 这个车道碰撞了多少次
            'lane_length': 0,  # 这个车道的长度
            'lane_start': (0, 0),  # 这个车道的起点
            'lane_end': (0, 0),  # 这个车道的终点
            'lane_num_CAV': 0,  # 这个车道上的CAV数量
            'lane_CAV_penetration': 0,  # 这个车道上的CAV占比
            'vehicles_state': [],  # 这个车道上车辆的位置
        } for lane_id in lane_ids
    }
    # 初始化 ego vehicle 的统计信息
    ego_statistics = {}
    # 初始化 hdv vehicle 的统计信息
    hdv_statistics = {}
    # 初始化 reward_statistics, 用于记录每一辆车所在的 edge, 后面用于统计车辆的 travel time
    reward_statistics = {}

    # 先收录所有车的信息 - （HDV + CAV）
    for vehicle_id, vehicle in state.items():
        lane_id = vehicle['lane_id']  # 这个车所在车道的 ID. eg: 'E0_2'
        lane_index = vehicle['lane_index']
        if lane_id[:3] in [':J1']:
            lane_id = f'E1_{int(lane_index)}'
        elif lane_id[:3] in [':J2']:
            lane_id = f'E1_{int(lane_index)}'
        # elif lane_id[:3] in [':J3']:
        #     lane_id = f'E3_{int(lane_index)}'
        # elif lane_id[:3] in [':J0']:
        #     lane_id = f'E0_{int(lane_index)}'

        road_id = vehicle['road_id']  # 这个车行驶道路的 ID. eg: 'E0'
        if road_id[:3] in [':J1']:
            road_id = 'E1'
        elif road_id[:3] in [':J2']:
            road_id = 'E1'
        # elif road_id[:3] in [':J3']:
        #     road_id = 'E3'
        # elif road_id[:3] in [':J0']:
        #     road_id = 'E0'

        lane_index = vehicle['lane_index']  # 这个车所在车道的 index eg: 2
        speed = vehicle['speed']  # vehicle当前车速
        position = vehicle['position']  # vehicle当前位置
        heading = vehicle['heading']  # vehicle当前朝向
        acceleration = vehicle['acceleration']

        leader_info = vehicle['leader']  # vehicle前车的信息
        distance = vehicle['distance']  # 这个车的累积行驶距离
        waiting_time = vehicle['waiting_time']  # vehicle的等待时间
        accumulated_waiting_time = vehicle['accumulated_waiting_time']  # vehicle的累积等待时间
        lane_position = vehicle['lane_position']  # vehicle 前保险杠到车道起点的距离

        # 感兴趣的lane数据统计
        if lane_id in lane_ids:
            # reward计算需要的信息
            # 记录每一个 vehicle 的 (edge id, distance, speed), 用于统计 travel time 从而计算平均速度
            reward_statistics[vehicle_id] = [road_id, distance, speed, acceleration,
                                             position[0], position[1],
                                             waiting_time, accumulated_waiting_time]

        # ego车需要的信息
        if vehicle['vehicle_type'] == 'ego':
            # TODO: when will the state['leader'] be not None?
            surroundings = vehicle['surround']
            surroundings_expand_4 = vehicle['surround_expand_4']
            surroundings_expand_6 = vehicle['surround_expand_6']

            ego_statistics[vehicle_id] = [position, speed, acceleration,
                                          heading, road_id, lane_index,
                                          surroundings,
                                          surroundings_expand_4,
                                          surroundings_expand_6
                                          ]
        # HDV车需要的信息
        else:
            hdv_statistics[vehicle_id] = [position, speed, acceleration,
                                          heading, road_id, lane_index]

        # 根据lane_id把以下信息写进lane_statistics
        if lane_id in lane_statistics:
            stats = lane_statistics[lane_id]
            stats['lane_length'] = LANE_LENGTHS[road_id],  # 这个车道的长度
            stats['lane_start'] = Lane_start[road_id],  # 这个车道的起点
            stats['lane_end'] = Lane_end[road_id],  # 这个车道的终点
            stats['vehicle_count'] += 1
            if vehicle['vehicle_type'] == 'ego':
                stats['lane_num_CAV'] += 1
                vehicle_type = 1
            else:
                vehicle_type = 0
            stats['speeds'].append(vehicle['speed'])
            stats['accelerations'].append(vehicle['acceleration'])
            if vehicle['acceleration'] == [] or vehicle['acceleration'] == None:
                print('acceleration is empty')
            stats['accumulated_waiting_times'].append(vehicle['accumulated_waiting_time'])
            stats['waiting_times'].append(vehicle['waiting_time'])
            stats['lane_density'] = stats['vehicle_count'] / LANE_LENGTHS[road_id]
            stats['vehicles_state'].append([vehicle_type, vehicle['position'][0]/state_divisors[0], vehicle['position'][1]/state_divisors[1],
                                            vehicle['speed']/state_divisors[2], vehicle['acceleration']/state_divisors[3], vehicle['heading']/state_divisors[4]])

    # lane_statistics计算
    for lane_id, stats in lane_statistics.items():
        speeds = np.array(stats['speeds'])
        accelerations = np.array(stats['accelerations'])

        veh_stats = np.array(stats['vehicles_state'])
        if len(veh_stats) == 0:
            vehs_stats = np.zeros((max_veh_num, 6))
        elif len(veh_stats) < max_veh_num:
            vehs_stats = np.pad(veh_stats, ((0, max_veh_num - len(veh_stats)), (0, 0)), 'constant', constant_values=0)
        else:
            vehs_stats = veh_stats[:max_veh_num]
        # vehs_stats = vehs_stats.flatten()

        waiting_times = np.array(stats['waiting_times'])
        accumulated_waiting_times = np.array(stats['accumulated_waiting_times'])
        vehicle_count = stats['vehicle_count']
        lane_length = stats['lane_length']
        lane_start = stats['lane_start']
        lane_end = stats['lane_end']
        lane_density = stats['lane_density']
        lane_num_CAV = stats['lane_num_CAV']
        lane_CAV_penetration = lane_num_CAV / vehicle_count if vehicle_count > 0 else 0

        if vehicle_count > 0:  # 当这个车道上有车的时候
            lane_statistics[lane_id] = [
                vehicle_count,
                lane_density,
                lane_length[0],
                np.mean(speeds), np.max(speeds), np.min(speeds),
                np.mean(accelerations), np.max(accelerations), np.min(accelerations),
                np.mean(waiting_times), np.max(waiting_times), np.min(waiting_times),
                np.mean(accumulated_waiting_times), np.max(accumulated_waiting_times), np.min(accumulated_waiting_times),
                lane_num_CAV, lane_CAV_penetration,
                lane_start, lane_end,
                vehs_stats
            ]
        else:
            # lane_statistics[lane_id] = [0] * 12
            lane_statistics[lane_id] = [0] * 17 # add lane_num_CAV, lane_CAV_penetration

            # 将两个数组添加到列表中
            lane_statistics[lane_id].append(lane_start)
            lane_statistics[lane_id].append(lane_end)
            lane_statistics[lane_id].append(vehs_stats)
    # lane_statistics转换成dict
    lane_statistics = {lane_id: stats for lane_id, stats in lane_statistics.items()}

    return lane_statistics, ego_statistics, reward_statistics, hdv_statistics

def check_collisions_based_pos(vehicles, gap_threshold: float):
    """输出距离过小的车辆的 ID, 直接根据 pos 来进行计算是否碰撞 (比较简单)

    Args:
        vehicles: 包含车辆部分的位置信息
        gap_threshold (float): 最小的距离限制
    """
    collisions = []
    collisions_info = []

    _distance = {}  # 记录每辆车之间的距离
    for (id1, v1), (id2, v2) in itertools.combinations(vehicles.items(), 2):
        dist = math.sqrt(
            (v1['position'][0] - v2['position'][0]) ** 2 \
            + (v1['position'][1] - v2['position'][1]) ** 2
        )
        _distance[f'{id1}-{id2}'] = dist
        if dist < gap_threshold:
            collisions.append((id1, id2))
            collisions_info.append({'collision': True,
                                    'CAV_key': id1,
                                    'surround_key': id2,
                                    'distance': dist,
                                    })

    return collisions, collisions_info

def check_collisions(vehicles, ego_ids, gap_threshold: float, gap_warn_collision: float):
    ego_collision = []
    ego_warn = []

    info = []

    # 把ego的state专门拿出来
    def filter_vehicles(vehicles, ego_ids):
        # Using dictionary comprehension to create a new dictionary
        # by iterating over all key-value pairs in the original dictionary
        # and including only those whose keys are in ego_ids
        filtered_vehicles = {key: value for key, value in vehicles.items() if key in ego_ids}
        return filtered_vehicles

    filtered_ego_vehicles = filter_vehicles(vehicles, ego_ids)

    for ego_key, ego_value in filtered_ego_vehicles.items():
        for surround_direction, content in filtered_ego_vehicles[ego_key]['surround'].items():
            c_info = None
            w_info = None

            # print(ego_key, 'is surrounded by:', content[0], 'with direction', surround_direction,
            # 'at distance', content[1])
            # TODO: 同一个车道和不同车道的车辆的warn gap应该是不一样！！！！11
            distance = math.sqrt(content[1] ** 2 + content[2] ** 2)
            # print('distance:', distance)
            if distance < gap_threshold:
                # print(ego_key, 'collision with', content[0])
                ego_collision.append((ego_key, content[0]))
                c_info = {'collision': True,
                          'CAV_key': ego_key,
                          'surround_key': content[0],
                          'distance': distance,
                          'relative_speed': content[3],
                          }

            elif gap_threshold <= distance < gap_warn_collision:
                ego_warn.append((ego_key, content[0]))
                w_info = {'warn': True,
                          'CAV_key': ego_key,
                          'surround_key': content[0],
                          'distance': distance,
                          'relative_speed': content[3],
                          }
            if c_info:
                info.append(c_info)
            elif w_info:
                info.append(w_info)

    return ego_collision, ego_warn, info

def check_prefix(a: str, B: List[str]) -> bool:
    """检查 B 中元素是否有以 a 开头的

    Args:
        a (str): lane_id
        B (List[str]): bottle_neck ids

    Returns:
        bool: 返回 lane_id 是否是 bottleneck
    """
    return any(a.startswith(prefix) for prefix in B)

def count_bottleneck_vehicles(lane_statistics, bottle_necks) -> int:
    """
    统计 bottleneck 处车辆的个数
    """
    veh_num = 0
    for lane_id, lane_info in lane_statistics.items():
        if check_prefix(lane_id, bottle_necks):
            veh_num += lane_info[0]  # lane_statistics的第一个是vehicle_count
    return veh_num

def calculate_congestion(vehicles: int, length: float, num_lane: int, ratio: float = 1) -> float:
    """计算 bottle neck 的占有率, 我们假设一辆车算上车间距是 10m, 那么一段路的。占有率是
        占有率 = 车辆数/(车道长度*车道数/10)
    于是可以根据占有率计算拥堵系数为:
        拥堵程度 = min(占有率, 1)

    Args:
        vehicles (int): 在 bottle neck 处车辆的数量
        length (float): bottle neck 的长度, 单位是 m
        num_lane (int): bottle neck 的车道数

    Returns:
        float: 拥堵系数 in (0,1)
    """
    capacity_used = ratio * vehicles / (length * num_lane / 10)  # 占有率
    congestion_level = min(capacity_used, 1)  # Ensuring congestion level does not exceed 100%
    return congestion_level

def calculate_speed(congestion_level: float, speed: int) -> float:
    """根据拥堵程度来计算车辆的速度

    Args:
        congestion_level (float): 拥堵的程度, 通过 calculate_congestion 计算得到
        speed (int): 车辆当前的速度

    Returns:
        float: 车辆新的速度
    """
    if congestion_level > 0.2:
        speed = speed * (1 - congestion_level)
        speed = max(speed, 1)
    else:
        speed = -1  # 不控制速度
    return speed

def one_hot_encode(value, unique_values):
    """Create an array with zeros and set the corresponding index to 1
    """
    one_hot = np.zeros(len(unique_values))
    index = unique_values.index(value)
    one_hot[index] = 1
    return one_hot.tolist()

def euclidean_distance(point1, point2):
    # Convert points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the Euclidean distance
    distance = np.linalg.norm(point1 - point2)
    return distance

def compute_ego_vehicle_features(
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        bottle_neck_positions: Tuple[float]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        bottle_neck_positions (Tuple[float]): bottle neck 的坐标
    """
    feature_vectors = {}

    for ego_id, ego_info in ego_statistics.items():
        # ############################## ego_statistics 的信息 ##############################
        speed, position, heading, road_id, lane_index, surroundings = ego_info

        # 速度归一化  - 1
        normalized_speed = speed / 20.0

        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / 700
        normalized_position_y = position_y

        # 转向归一化 - 1
        normalized_heading = heading / 360

        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # One-hot encode lane_index - 4
        lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        if len(lane_index_one_hot) < 4:
            lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))

        # ############################## 周车信息 ##############################
        # 提取surrounding的信息 -18
        surround = []
        for index, (_, statistics) in enumerate(surroundings.items()):
            relat_x, relat_y, relat_v = statistics[1:4]
            surround.append([relat_x, relat_y, relat_v])  # relat_x, relat_y, relat_v
        flat_surround = [item for sublist in surround for item in sublist]
        # 如果周围车辆的信息不足6*3个, 补0 对齐到最多数量的周车信息
        if len(flat_surround) < 18:
            flat_surround += [0] * (18 - len(flat_surround))

        # ############################## 当前车道信息 ##############################
        # ego_id当前所在的lane
        ego_lane_id = f'{road_id}_{lane_index}'
        # 每个obs只需要一个lane的信息，不需要所有lane的信息， shared obs可以拿到所有lane的信息
        ego_lane_stats = lane_statistics[ego_lane_id]

        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)
        ego_lane_stats = ego_lane_stats[:4] + ego_lane_stats[6:7] + ego_lane_stats[9:10]

        # ############################## bottle_neck 的信息 ##############################
        # 车辆距离bottle_neck
        bottle_neck_position_x = bottle_neck_positions[0] / 700
        bottle_neck_position_y = bottle_neck_positions[1]

        distance = euclidean_distance(position, bottle_neck_positions)
        normalized_distance = distance / 700

        # ############################## 合并所有 ##############################
        # feature_vector = [normalized_speed, normalized_position_x, normalized_position_y, normalized_heading,
        #                   bottle_neck_position_x, bottle_neck_position_y, normalized_distance] + \
        #                   road_id_one_hot + lane_index_one_hot + flat_surround + all_lane_stats

        feature_vector = [normalized_speed, normalized_position_x, normalized_position_y, normalized_heading,
                          bottle_neck_position_x, bottle_neck_position_y, normalized_distance] + \
                         road_id_one_hot + lane_index_one_hot + flat_surround + ego_lane_stats

        # Assign the feature vector to the corresponding ego vehicle
        feature_vectors[ego_id] = [float(item) for item in feature_vector]

    # 保证每一个 ego vehicle 的特征长度一致
    assert all(len(feature_vector) == 40 for feature_vector in feature_vectors.values())

    # # Create a new dictionary to hold the updated feature lists
    # updated_feature_vectors = {}
    #
    # # Iterate over each key-value pair in the dictionary
    # for cav, features in feature_vectors.items():
    #     # Start with the original list of features
    #     new_features = features.copy()
    #     # Iterate over all other CAVs
    #     for other_cav, other_features in feature_vectors.items():
    #         if other_cav != cav:
    #             # Append the first four elements of the other CAV's features
    #             new_features.extend(other_features[:4])
    #     # Update the dictionary with the new list
    #     updated_feature_vectors[cav] = new_features
    #
    # for updated_feature_vector in updated_feature_vectors.values():
    #     if not len(updated_feature_vector) == 56:
    #         updated_feature_vector += [0] * (56 - len(updated_feature_vector))
    #
    # # assert all(len(updated_feature_vector) == 56 for updated_feature_vector in updated_feature_vectors.values())

    return feature_vectors

def get_target(
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的临近目标点（x_target, y_target）

        Args:
            ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
            lane_statistics (Dict[str, List[float]]): 路网的信息
    """
    target_points = {}
    for ego_id, ego_info in ego_statistics.items():
        if ego_info[0][0] < 400:
            target_points[ego_id] = [400/700, ego_info[0][1]/3.2]
        elif ego_info[0][0] < 496:
            target_points[ego_id] = [496/700, 1.6/3.2] if ego_info[0][1] > 0 else [496/700, -1.6/3.2]
        else:
            target_points[ego_id] = [1, 1.6/3.2] if ego_info[0][1] > 0 else [1, -1.6/3.2]
    return target_points

def compute_base_ego_vehicle_features(
        self,
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        node_positions: Dict[str, List[int]],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        node_positions (Tuple[float]): 每个道路节点的坐标
    output:
    """
    # ############################## 所有HDV的信息 ############################## 13
    hdv_stats = {}
    for hdv_id, hdv_info in hdv_statistics.items():
        position, speed, acceleration, heading, road_id, lane_index = hdv_info
        # 速度归一化  - 1
        normalized_speed = speed / state_divisors[2]
        # 加速度归一化  - 1
        normalized_acceleration = acceleration / state_divisors[3]
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / state_divisors[0]
        normalized_position_y = position_y / state_divisors[1]
        # 转向归一化 - 1
        normalized_heading = heading / state_divisors[4]
        # One-hot encode road_id - 5
        # road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        # lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        # if len(lane_index_one_hot) < 4:
        #     lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        hdv_stats[hdv_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading]  # + road_id_one_hot
    # convert to 2D array (12 * 13)  - 12 is max number of HDVs
    hdv_stats = np.array(list(hdv_stats.values()))
    if 0 < hdv_stats.shape[0] <= self.max_num_HDVs:
        # add 0 to make sure the shape is (12, 13)
        hdv_stats = np.vstack([hdv_stats, np.zeros((self.max_num_HDVs - hdv_stats.shape[0], 5))])
    elif hdv_stats.shape[0] == 0:
        hdv_stats = np.zeros((self.max_num_HDVs, 5))
    # ############################## 所有CAV的信息 ############################## 13
    cav_stats = {}
    ego_stats = {}
    global_cav = {}
    next_node = {}
    ego_lane_stats = {}
    surround_vehs_stats = {key:[] for key in ego_ids}
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        position, speed, acceleration, heading, road_id, lane_index, surroundings, surroundings_expand_4, surroundings_expand_6 = ego_info
        # 速度归一化  - 1
        normalized_speed = speed / state_divisors[2]
        # 加速度归一化  - 1
        normalized_acceleration = acceleration / state_divisors[3]
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / state_divisors[0]
        normalized_position_y = position_y / state_divisors[1]
        # # 转向归一化 - 1
        # if heading < 0:
        #     heading += 360
        # elif heading > 360:
        #     heading -= 360
        normalized_heading = heading / state_divisors[4]
        # One-hot encode road_id - 5
        # road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # if position_y < 5 and 90 <= heading < 180:
        #     next_node[ego_id] = [node_positions['J1'][0]/state_divisors[0], node_positions['J1'][1]/state_divisors[1]]
        #     dis_next_node = 1 - normalized_position_x
        # elif position_x > 95 and 0 <= heading <= 90:
        #     next_node[ego_id] = [node_positions['J2'][0]/state_divisors[0], node_positions['J2'][1]/state_divisors[1]]
        #     dis_next_node = 1 - normalized_position_y
        # elif position_y > 95 and 270 <= heading < 360:
        #     next_node[ego_id] = [node_positions['J3'][0]/state_divisors[0], node_positions['J3'][1]/state_divisors[1]]
        #     dis_next_node = normalized_position_x
        # else:
        #     next_node[ego_id] = [node_positions['J0'][0]/state_divisors[0], node_positions['J0'][1]/state_divisors[1]]
        #     dis_next_node = normalized_position_y
        # next_node[ego_id].append(dis_next_node)

        next_node[ego_id] = [1, 0]
        dis_next_node = 1 - normalized_position_x
        next_node[ego_id].append(dis_next_node)
        # ############################## 周车信息 ############################## 18
        # 提取surrounding的信息 -12
        for index, (_, statistics) in enumerate(surroundings.items()):
            veh_type, relat_x, relat_y, relat_v, veh_acc, veh_heading = statistics[1:7]
            surround_vehs_stats[ego_id].append([veh_type, relat_x/state_divisors[0], relat_y/state_divisors[1], relat_v/state_divisors[2],
                                                veh_acc/state_divisors[3], veh_heading/state_divisors[4]])

        cav_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading]
        ego_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading]  # + road_id_one_hot
        # global_cav[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed]
    # convert to 2D array (5 * 5)
    cav_stats = np.array(list(cav_stats.values()))
    if 0 < cav_stats.shape[0] <= self.max_num_CAVs:
        # add 0 to make sure the shape is (12, 13)
        cav_stats = np.vstack([cav_stats, np.zeros((self.max_num_CAVs - cav_stats.shape[0], 5))])
    elif cav_stats.shape[0] == 0:
        cav_stats = np.zeros((self.max_num_CAVs, 5))

    if len(ego_stats) != len(ego_ids):
        for ego_id in ego_ids:
            if ego_id not in ego_stats:
                ego_stats[ego_id] = [0.0] * 5

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}
    all_lane_state_simple = {}

    # Iterate over all possible lanes to get their statistics
    # for lane_id, lane_info in lane_statistics.items():
    #     # - vehicle_count: 当前车道的车辆数 1
    #     # - lane_density: 当前车道的车辆密度 1
    #     # - lane_length: 这个车道的长度 1
    #     # - speeds: 在这个车道上车的速度 1 (mean)
    #     # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
    #     # - accumulated_waiting_times: 车的累积等待时间 1 (mean)--delete
    #     # - lane_CAV_penetration: 这个车道上的CAV占比 1
    #
    #     # all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[9:10]
    #     lane_info[0] = lane_info[0] / self.lane_max_num_vehs
    #     lane_info[2] = lane_info[2] / state_divisors[0]
    #     lane_info[3] = lane_info[3] / 20
    #     all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[16:17]
    # for ego_id, ego_info in ego_statistics.items():
    #     # ############################## 自己车的信息 ############################## 13
    #     position, speed, acceleration, heading, road_id, lane_index, surroundings, surroundings_expand_4, surroundings_expand_6 = ego_info
    #     ego_lane = f'{road_id}_{lane_index}'
    #     ego_lane_stats[ego_id] = all_lane_stats[ego_lane]
    #
    # # convert to 2D array (18 * 6)
    # all_lane_stats = np.array(list(all_lane_stats.values()))

    feature_vector = {}
    # feature_vector['road_structure'] = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    feature_vector['road_structure'] = np.array([1, 0])
    # A function to flatten a dictionary structure into 1D array
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        size_obs = np.size(np.array(flat_list))
        return np.array(flat_list)
    feature_vectors_current = {}
    shared_feature_vectors_current = {}
    flat_surround_vehs = {key: [] for key in next_node.keys()}
    for ego_id in ego_statistics.keys():
        feature_vectors_current[ego_id] = feature_vector.copy()
        feature_vectors_current[ego_id]['next_node'] = next_node[ego_id]
        feature_vectors_current[ego_id]['self_stats'] = [1] + ego_stats[ego_id]
        flat_surround_vehs[ego_id] = [item for sublist in surround_vehs_stats[ego_id] for item in sublist]
        if len(flat_surround_vehs[ego_id]) < 12:
            flat_surround_vehs[ego_id] += [0] * (12 - len(flat_surround_vehs[ego_id]))
        feature_vectors_current[ego_id]['surround_vehs_stats'] = flat_surround_vehs[ego_id]
        # feature_vectors_current[ego_id]['ego_lane_stats'] = ego_lane_stats[ego_id]

        shared_feature_vectors_current[ego_id] = feature_vector.copy()
        shared_feature_vectors_current[ego_id]['cav_stats'] = cav_stats
        # shared_feature_vectors_current[ego_id]['lane_stats'] = all_lane_stats

    # Flatten the dictionary structure
    # feature_vectors_current_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
    #                            feature_vectors_current.items()}
    feature_vectors = {key: {} for key in ego_statistics.keys()}

    for ego_id, feature_vector_current in feature_vectors_current.items():
        feature_vectors[ego_id] = feature_vector_current
    feature_vectors_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               feature_vectors.items()}

    shared_feature_flatten = {ego_id: flatten_to_1d(shared_feature_vector) for ego_id, shared_feature_vector in
                                shared_feature_vectors_current.items()}
    return shared_feature_vectors_current, shared_feature_flatten, feature_vectors, feature_vectors_flatten

def compute_hierarchical_ego_vehicle_features(
        self,
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        node_positions: Dict[str, List[int]],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    """计算每一个 ego vehicle 的特征

    Args:
        ego_statistics (Dict[str, List[Union[float, str, Tuple[int]]]]): ego vehicle 的信息
        lane_statistics (Dict[str, List[float]]): 路网的信息
        unique_edges (List[str]): 所有考虑的 edge
        edge_lane_num (Dict[str, int]): 每一个 edge 对应的车道
        node_positions (Tuple[float]): 每个道路节点的坐标
    output:
    """
    hdv_stats = {}
    hdv_hist = {}

    cav_stats = {}
    cav_hist = {}
    ego_stats = {}
    global_cav = {}
    next_node = {}
    ego_lane_stats = {}
    next_lane_stats = {}
    surround_2_vehs_stats = {key: [] for key in ego_ids}
    surround_4_vehs_stats = {key: [] for key in ego_ids}
    surround_6_vehs_stats = {key: [] for key in ego_ids}
    ego_hist_motion = {key: [] for key in ego_ids}
    surround_2_key = {'front', 'back'}
    surround_4_key = {'front', 'front_expand_0', 'back', 'back_expand_0'}
    surround_6_key = {'front', 'front_expand_0', 'front_expand_1', 'back', 'back_expand_0', 'back_expand_1'}
    surround_2_stats = {key: {surround_key: [0.0] * (1 + self.hist_length * 5) for surround_key in surround_2_key} for
                        key
                        in ego_ids}
    surround_4_stats = {key: {surround_key: [0.0] * (1 + self.hist_length * 5) for surround_key in surround_4_key} for
                        key
                        in ego_ids}
    surround_6_stats = {key: {surround_key: [0.0] * (1 + self.hist_length * 5) for surround_key in surround_6_key} for
                        key
                        in ego_ids}

    def flatten_to_1d(data_dict):
        # List comprehension for faster concatenation
        flat_list = [item.flatten() if isinstance(item, np.ndarray) else item for item in data_dict.values()]
        # Flatten and concatenate only once
        flat_array = np.concatenate([np.ravel(item) for item in flat_list])
        return flat_array

    def flatten_list(nested_list):
        flat_list = []
        stack = [nested_list]

        while stack:
            current = stack.pop()
            if isinstance(current, collections.abc.Iterable) and not isinstance(current, (str, bytes)):
                stack.extend(current)
            else:
                flat_list.append(current)

        return flat_list
    # ############################## 所有HDV的信息 ##############################
    def normalize_vehicle_info(position, speed, acceleration, heading, state_divisors):
        """归一化位置信息、速度、加速度和方向"""
        pos_x, pos_y = position
        div_pos_x, div_pos_y, div_speed, div_acc, div_heading = state_divisors[:5]

        # Use direct division and list creation for efficiency
        return [
            pos_x / div_pos_x,
            pos_y / div_pos_y,
            speed / div_speed,
            acceleration / div_acc,
            heading / div_heading  # Assuming heading is in [0, 360]
        ]

    def process_vehicle_history(vehicle_id, position, speed, acceleration, heading, state_divisors, vehicle_hist,
                                vehicles_hist, hist_length, use_hist_info):
        """处理车辆的历史信息"""
        if use_hist_info:
            # Loop optimization: Use a range based on precomputed values
            for i in range(hist_length - 1, 0, -1):
                # Minimize the use of deepcopy by reusing data where possible
                vehicles_hist[f'hist_{i + 1}'][vehicle_id] = vehicles_hist[f'hist_{i}'][vehicle_id].copy()

                # Normalize the history state and extend the vehicle_hist
                # hist_state = np.array(vehicles_hist[f'hist_{i + 1}'][vehicle_id][:5]) / state_divisors[:5]
                # vehicle_hist[vehicle_id].extend(hist_state.tolist())

            # Update the most recent history (hist_1)
            vehicles_hist['hist_1'][vehicle_id] = [position[0], position[1], speed, acceleration, heading]

            # Normalize the new state and extend the vehicle history
            # hist_state = np.array(vehicles_hist['hist_1'][vehicle_id][:5]) / state_divisors[:5]
            # vehicle_hist[vehicle_id].extend(hist_state.tolist())

    def initialize_missing_data(vehicle_stats, vehicle_hist, max_num_vehicles, hist_length, prefix):
        """初始化缺失的车辆数据"""
        default_stats = [0.0] * 5  # Precompute the default stats list once
        # default_hist = [0.0] * (5 * hist_length)  # Precompute the default history list once

        for i in range(max_num_vehicles):
            vehicle_key = f'{prefix}_{i}'

            if vehicle_key not in vehicle_stats:
                vehicle_stats[
                    vehicle_key] = default_stats.copy()  # Use precomputed default list and copy it to avoid aliasing
                # vehicle_hist[
                #     vehicle_key] = default_hist.copy()  # Copy default history to avoid referencing the same list
    start = time.time()
    # Loop through hdv_statistics and process each vehicle
    for hdv_id, hdv_info in hdv_statistics.items():
        position, speed, acceleration, heading, road_id, lane_index = hdv_info

        # Normalize HDV data
        normalized_info = normalize_vehicle_info(position, speed, acceleration, heading, state_divisors)

        # One-hot encode road_id
        # road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # Save HDV state
        hdv_stats[hdv_id] = normalized_info  # + road_id_one_hot

        # Initialize HDV history data and process it
        hdv_hist[hdv_id] = []
        process_vehicle_history(
            hdv_id, position, speed, acceleration, heading, state_divisors, hdv_hist,
            self.vehicles_hist, self.hist_length, self.use_hist_info
        )

    # Initialize missing HDV data in a batch operation to avoid repetitive checks
    initialize_missing_data(hdv_stats, hdv_hist, self.max_num_HDVs, self.hist_length, 'HDV')

    # Convert hdv_stats and hdv_hist to NumPy arrays for faster computation
    hdv_stats_array = np.array(list(hdv_stats.values()))
    # hdv_hist_array = np.array(list(hdv_hist.values()))

    # 处理CAV信息
    # Precompute the zero action to avoid recomputation
    zero_action = [0, 0]

    for ego_id, ego_info in ego_statistics.items():
        # Unpack the ego information
        position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
        position_x, position_y = position

        # Normalize vehicle information
        normalized_info = normalize_vehicle_info(position, speed, acceleration, heading, state_divisors)
        # road_id_one_hot = one_hot_encode(road_id, unique_edges)

        # Store normalized CAV state
        cav_stats[ego_id] = normalized_info  # + road_id_one_hot

        # Calculate the last action (handle empty list with fallback)
        last_action = zero_action
        if self.actor_action[ego_id] != []:
            last_action = flatten_list(self.actor_action[ego_id][-1])

        ego_last_action = [last_action[0] / state_divisors[5], last_action[1] / state_divisors[6]]

        # Combine all stats for the ego vehicle
        ego_stats[ego_id] = normalized_info + ego_last_action  # + road_id_one_hot

        # Process vehicle history
        cav_hist[ego_id] = []
        process_vehicle_history(
            ego_id, position, speed, acceleration, heading, state_divisors, cav_hist,
            self.vehicles_hist, self.hist_length, self.use_hist_info
        )
        # Determine the next node and compute distance
        # if position_y < 5 and 90 <= heading < 180:
        #     node_key = 'J1'
        #     dis_next_node = 1 - normalized_info[0]
        # elif position_x > 95 and 0 <= heading <= 90:
        #     node_key = 'J2'
        #     dis_next_node = 1 - normalized_info[1]
        # elif position_y > 95 and 270 <= heading < 360:
        #     node_key = 'J3'
        #     dis_next_node = normalized_info[0]
        # else:
        #     node_key = 'J0'
        #     dis_next_node = normalized_info[1]
        #
        # next_node_pos = [node_positions[node_key][0] / state_divisors[0],
        #                  node_positions[node_key][1] / state_divisors[1]]
        # next_node[ego_id] = next_node_pos + [dis_next_node]

        next_node_pos = [1, 0]
        next_node[ego_id] = next_node_pos + [1 - normalized_info[0]]

    # Initialize missing CAV data
    initialize_missing_data(cav_stats, cav_hist, self.max_num_CAVs, self.hist_length, 'CAV')

    # Convert cav_stats and cav_hist to NumPy arrays for efficient processing
    cav_stats_array = np.array(list(cav_stats.values()))
    # cav_hist_array = np.array(list(cav_hist.values()))

        # ############################## 周车信息 ##############################
    def calculate_relative_motion(target_hist, position_x, position_y, speed, acceleration, heading, state_divisors):
        """Calculate the relative motion between current vehicle and target."""
        if target_hist == [0.0] * 5:
            return [0.0] * 5

        relative_hist = [
            (target_hist[0] - position_x),
            (target_hist[1] - position_y),
            (speed - target_hist[2]),
            (acceleration - target_hist[3]),
            (target_hist[4] - heading)
        ]

        # Normalize the relative motion using state divisors
        return [rm / sd for rm, sd in zip(relative_hist, state_divisors[:5])]

    def process_surrounding_vehicles(ego_id, surroundings, surround_vehs_stats, surround_stats, position_x, position_y,
                                     speed, acceleration, heading, state_divisors, vehicles_hist):
        """Process information from surrounding vehicles."""
        # Preallocate list for surrounding stats
        surround_vehs_stats[ego_id] = []
        surround_stats[ego_id] = {}

        for idx, statistics in surroundings.items():
            surround_ID, veh_type, relat_x, relat_y, relat_v, veh_acc, veh_heading = statistics[:7]

            # Append normalized surrounding vehicle statistics
            surround_vehs_stats[ego_id].append([
                veh_type, relat_x / state_divisors[0], relat_y / state_divisors[1],
                          relat_v / state_divisors[2], veh_acc / state_divisors[3], veh_heading / state_divisors[4]
            ])

            # Fetch historical data in one go for the surrounding vehicle
            hist_data = [
                vehicles_hist[f'hist_{i}'][surround_ID] for i in range(5, 0, -1)
            ]

            # Initialize relative motion list with the vehicle type
            relative_motion = [veh_type]

            # Calculate relative motion for each historical step
            for target_hist in hist_data:
                relative_motion.extend(
                    calculate_relative_motion(target_hist, position_x, position_y, speed, acceleration, heading,
                                              state_divisors)
                )

            # Store relative motion history in surround_stats
            surround_stats[ego_id][idx] = relative_motion

    def process_ego_hist_motion(ego_id, vehicles_hist, position_x, position_y, speed, acceleration, heading,
                                state_divisors, ego_hist_motion):
        """计算ego vehicle的历史运动信息"""

        # Retrieve historical data in a single list comprehension
        hist_data = [vehicles_hist[f'hist_{i}'][ego_id] for i in range(5, 0, -1)]

        # Use list comprehension to compute the relative motion for each time step
        relative_hist_motion = [
            val for target_hist in hist_data
            for val in
            calculate_relative_motion(target_hist, position_x, position_y, speed, acceleration, heading, state_divisors)
        ]

        # Assign the computed relative motion to the ego vehicle's history motion
        ego_hist_motion[ego_id] = relative_hist_motion

    for ego_id, ego_info in ego_statistics.items():
        # Unpack ego vehicle information
        position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
        position_x, position_y = position

        # Calculate ego's historical motion information
        process_ego_hist_motion(ego_id, self.vehicles_hist, position_x, position_y, speed, acceleration, heading,
                                state_divisors, ego_hist_motion)

        # Process surroundings for 2-nearest vehicles
        process_surrounding_vehicles(ego_id, surroundings_2, surround_2_vehs_stats, surround_2_stats, position_x,
                                     position_y, speed, acceleration, heading, state_divisors, self.vehicles_hist)

        # Process surroundings_4
        process_surrounding_vehicles(ego_id, surroundings_4, surround_4_vehs_stats, surround_4_stats, position_x,
                                     position_y, speed, acceleration, heading, state_divisors, self.vehicles_hist)

        # Process surroundings_6
        process_surrounding_vehicles(ego_id, surroundings_6, surround_6_vehs_stats, surround_6_stats, position_x,
                                     position_y, speed, acceleration, heading, state_divisors, self.vehicles_hist)

    if len(ego_stats) != len(ego_ids):
        # Precompute the default list once
        default_ego_stats = [0.0] * 7   # 11

        # Use setdefault to add missing ego_ids with the default value
        for ego_id in ego_ids:
            ego_stats.setdefault(ego_id, default_ego_stats.copy())

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}
    all_lane_distribution = {}
    all_lane_state_simple = {}

    # Iterate over all possible lanes to get their statistics
    def process_lane_info(lane_statistics, lane_max_num_vehs, state_divisors):
        """处理车道信息"""
        all_lane_stats = {}
        all_lane_distribution = {}

        # Precompute some normalization factors for efficiency
        max_veh_norm = 1.0 / lane_max_num_vehs
        length_norm = 1.0 / state_divisors[0]
        speed_norm = 1.0 / 20.0
        position_norm = state_divisors[:2]  # For elements 17 and 18

        for lane_id, lane_info in lane_statistics.items():
            # Normalize lane statistics
            lane_info[0] *= max_veh_norm  # Normalize number of vehicles
            lane_info[2] *= length_norm  # Normalize lane length
            lane_info[3] *= speed_norm  # Normalize speed
            lane_info[17] /= position_norm  # Normalize certain data
            lane_info[18] /= position_norm  # Normalize certain data

            # Select required lane information
            selected_lane_info = lane_info[17:19] + lane_info[:4] + lane_info[6:7] + lane_info[16:17]

            # Flatten if necessary
            if isinstance(selected_lane_info[0], list) or isinstance(selected_lane_info[1], list):
                selected_lane_info = flatten_list(selected_lane_info)

            # Store each lane's processed data
            all_lane_stats[lane_id] = selected_lane_info
            all_lane_distribution[lane_id] = lane_info[19]  # Lane distribution info

        return all_lane_stats, all_lane_distribution

    def process_ego_info(ego_statistics, all_lane_stats):
        """处理每辆车的信息及其车道信息"""
        ego_lane_stats = {}
        next_lane_stats = {}

        # Precompute default lane stats for missing lanes
        default_lane_stats = [0.0] * len(next(iter(all_lane_stats.values())))

        for ego_id, ego_info in ego_statistics.items():
            position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
            ego_lane = f'{road_id}_{lane_index}'

            # Get ego vehicle's current lane stats or default if missing
            ego_lane_stats[ego_id] = all_lane_stats.get(ego_lane, default_lane_stats.copy())

            # Determine the next lane based on the ego vehicle's current lane
            next_lane = {
                'E0_0': 'E1_0',
                'E1_0': 'E0_0',
                'E2_0': 'E3_0'
            }.get(ego_lane, 'E0_0')  # Default to 'E0_0'

            # Get next lane stats or default if missing
            next_lane_stats[ego_id] = all_lane_stats.get(next_lane, default_lane_stats.copy())

        return ego_lane_stats, next_lane_stats

    # 执行主流程
    # all_lane_stats, all_lane_distribution = process_lane_info(lane_statistics, self.lane_max_num_vehs, state_divisors)
    # ego_lane_stats, next_lane_stats = process_ego_info(ego_statistics, all_lane_stats)

    # Efficiently convert to arrays without unnecessary flattening
    # all_lane_stats = np.array(flatten_list(list(all_lane_stats.values())))
    # all_lane_distribution = np.array(flatten_list(list(all_lane_distribution.values())))
    # end = time.time()
    # print(f'hdv+cav+surround+lane time: {end - start}')

    def flatten_and_pad(surround_stats, pad_length):
        """展平嵌套列表并填充到指定长度"""
        flat = [item for sublist in surround_stats.values() for item in sublist]
        if len(flat) < pad_length:
            flat += [0] * (pad_length - len(flat))
        else:
            flat = flat[:pad_length]
        return flat

    # 定义 feature_vector
    # feature_vector = {'road_structure': np.array([0, 0, 1, 0, 1, 1, 0, 1])}
    feature_vector = {'road_structure': np.array([1, 0])}

    # 初始化字典
    feature_vectors_current = {}
    shared_feature_vectors_current = {}

    # 预计算填充长度
    pad_len_2 = 2 * (1 + self.hist_length * 5)
    pad_len_4 = 4 * (1 + self.hist_length * 5)
    pad_len_6 = 6 * (1 + self.hist_length * 5)

    # 处理每个 ego_id 的信息
    for ego_id, ego_info in ego_statistics.items():
        position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
        position_x, position_y = position

        # 复制 feature_vector
        fv_current = feature_vector.copy()

        # 分配 'next_node'
        fv_current['next_node'] = next_node[ego_id]

        # 分配 'self_stats' 和 'self_hist_stats'
        fv_current['self_stats'] = [1] + ego_stats[ego_id]
        fv_current['self_hist_stats'] = [1] + ego_hist_motion[ego_id]

        # 处理 surround_2_vehs_stats
        surround_2_padded = flatten_and_pad(surround_2_stats[ego_id], pad_len_2)
        fv_current['surround_2_vehs_stats'] = surround_2_padded

        # # 处理 surround_4_vehs_stats
        surround_4_padded = flatten_and_pad(surround_4_stats[ego_id], pad_len_4)
        fv_current['surround_4_vehs_stats'] = surround_4_padded
        #
        # # 处理 surround_6_vehs_stats
        surround_6_padded = flatten_and_pad(surround_6_stats[ego_id], pad_len_6)
        fv_current['surround_6_vehs_stats'] = surround_6_padded

        # 分配车道统计信息
        # fv_current['ego_lane_stats'] = flatten_list(ego_lane_stats[ego_id])
        # fv_current['next_lane_stats'] = flatten_list(next_lane_stats[ego_id])

        # 将当前 ego_id 的 feature_vector 存入 feature_vectors_current
        feature_vectors_current[ego_id] = fv_current

        # 处理 shared_feature_vectors_current
        shared_fv_current = feature_vector.copy()
        shared_fv_current['hdv_stats'] = hdv_stats_array
        shared_fv_current['cav_stats'] = cav_stats_array
        # shared_fv_current['lane_stats'] = all_lane_stats
        # shared_fv_current['lane_distribution'] = all_lane_distribution

        # 将共享特征存入 shared_feature_vectors_current
        shared_feature_vectors_current[ego_id] = shared_fv_current

    # 初始化 feature_vectors 为与 ego_statistics 相同的键，值为空字典
    feature_vectors = {key: {} for key in ego_statistics.keys()}

    # 将 feature_vectors_current 的内容复制到 feature_vectors
    for ego_id, feature_vector_current in feature_vectors_current.items():
        feature_vectors[ego_id] = feature_vector_current

    # 将 feature_vectors 和 shared_feature_vectors_current 展平
    feature_vectors_flatten = {
        ego_id: flatten_to_1d(feature_vector)
        for ego_id, feature_vector in feature_vectors.items()
    }

    shared_feature_flatten = {
        ego_id: flatten_to_1d(shared_feature_vector)
        for ego_id, shared_feature_vector in shared_feature_vectors_current.items()
    }
    return shared_feature_vectors_current, shared_feature_flatten, feature_vectors, feature_vectors_flatten

def compute_centralized_vehicle_features(lane_statistics, feature_vectors, bottle_neck_positions):
    shared_features = {}

    # ############################## 所有车的速度 位置 转向信息 ##############################
    all_vehicle = []
    for _, ego_feature in feature_vectors.items():
        all_vehicle += ego_feature[:4]

    # ############################## lane_statistics 的信息 ##############################
    # Initialize a list to hold all lane statistics
    all_lane_stats = []

    # Iterate over all possible lanes to get their statistics
    for _, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)

        all_lane_stats += lane_info[:4] + lane_info[6:7] + lane_info[9:10]

    # ############################## bottleneck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    for ego_id in feature_vectors.keys():
        shared_features[ego_id] = [bottle_neck_position_x, bottle_neck_position_y] + all_vehicle + all_lane_stats

    # assert all(len(shared_feature) == 130 for shared_feature in shared_features.values())

    return shared_features

def compute_centralized_vehicle_features_hierarchical_version(
        obs_size, shared_obs_size, lane_statistics,
        feature_vectors_current, feature_vectors_current_flatten,
        feature_vectors, feature_vectors_flatten, ego_ids):
    shared_features = {}
    actor_features = {}

    for ego_id in feature_vectors.keys():
        actor_features[ego_id] = feature_vectors[ego_id].copy() # shared_features--actor / critic , can there output different obs to actor and critic?
    for ego_id in feature_vectors_current.keys():
        shared_features[ego_id] = feature_vectors_current[ego_id].copy()
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        return np.array(flat_list)

    shared_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               shared_features.items()}
    actor_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               actor_features.items()}
    if len(shared_features_flatten) != len(ego_ids):
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in shared_features_flatten:
                shared_features_flatten[ego_id] = np.zeros(shared_obs_size)
    if len(actor_features_flatten) != len(ego_ids):
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in actor_features_flatten:
                actor_features_flatten[ego_id] = np.zeros(obs_size)
    if len(shared_features_flatten) != len(ego_ids):
        print("Error: len(shared_features_flatten) != len(ego_ids)")
    if len(feature_vectors_flatten) != len(ego_ids):
        print("Error: len(feature_vectors_flatten) != len(ego_ids)")
    return actor_features, actor_features_flatten, shared_features, shared_features_flatten
