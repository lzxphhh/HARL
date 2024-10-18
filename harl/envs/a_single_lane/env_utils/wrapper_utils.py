import itertools
import math
from typing import List, Dict, Tuple, Union
import copy

import numpy as np
import collections.abc

LANE_LENGTHS = {
    'E0': 200,
    'E1': 200,
    'E2': 200,
    'E3': 200,
}

state_divisors = np.array([200, 200, 20, 6, 360, 6, 20])
Lane_start = {
    'E0': (0, 0),
    'E1': (200, 0),
    'E2': (200, 200),
    'E3': (0, 200),
}
Lane_end = {
    'E0': (200, 0),
    'E1': (200, 200),
    'E2': (0, 200),
    'E3': (0, 0),
}

def analyze_traffic(state, lane_ids):
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
            lane_id = f'E2_{int(lane_index)}'
        elif lane_id[:3] in [':J3']:
            lane_id = f'E3_{int(lane_index)}'
        elif lane_id[:3] in [':J0']:
            lane_id = f'E0_{int(lane_index)}'

        road_id = vehicle['road_id']  # 这个车行驶道路的 ID. eg: 'E0'
        if road_id[:3] in [':J1']:
            road_id = 'E1'
        elif road_id[:3] in [':J2']:
            road_id = 'E2'
        elif road_id[:3] in [':J3']:
            road_id = 'E3'
        elif road_id[:3] in [':J0']:
            road_id = 'E0'

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
            vehs_stats = np.zeros((15, 6))
        elif len(veh_stats) < 15:
            vehs_stats = np.pad(veh_stats, ((0, 15 - len(veh_stats)), (0, 0)), 'constant', constant_values=0)
        else:
            vehs_stats = veh_stats[:15]
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
        normalized_speed = speed / 15.0

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
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        # lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        # if len(lane_index_one_hot) < 4:
        #     lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        hdv_stats[hdv_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading] + road_id_one_hot
    # convert to 2D array (12 * 13)  - 12 is max number of HDVs
    hdv_stats = np.array(list(hdv_stats.values()))
    if 0 < hdv_stats.shape[0] <= self.max_num_HDVs:
        # add 0 to make sure the shape is (12, 13)
        hdv_stats = np.vstack([hdv_stats, np.zeros((self.max_num_HDVs - hdv_stats.shape[0], 9))])
    elif hdv_stats.shape[0] == 0:
        hdv_stats = np.zeros((self.max_num_HDVs, 9))
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
        # 转向归一化 - 1
        if heading < 0:
            heading += 360
        elif heading > 360:
            heading -= 360
        normalized_heading = heading / state_divisors[4]
        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        if position_y < 5 and 90 <= heading < 180:
            next_node[ego_id] = [node_positions['J1'][0]/state_divisors[0], node_positions['J1'][1]/state_divisors[1]]
            dis_next_node = 1 - normalized_position_x
        elif position_x > 195 and 0 <= heading <= 90:
            next_node[ego_id] = [node_positions['J2'][0]/state_divisors[0], node_positions['J2'][1]/state_divisors[1]]
            dis_next_node = 1 - normalized_position_y
        elif position_y > 195 and 270 <= heading < 360:
            next_node[ego_id] = [node_positions['J3'][0]/state_divisors[0], node_positions['J3'][1]/state_divisors[1]]
            dis_next_node = normalized_position_x
        else:
            next_node[ego_id] = [node_positions['J0'][0]/state_divisors[0], node_positions['J0'][1]/state_divisors[1]]
            dis_next_node = normalized_position_y
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
                             normalized_heading] + road_id_one_hot
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
                ego_stats[ego_id] = [0.0] * 9

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}
    all_lane_state_simple = {}

    # Iterate over all possible lanes to get their statistics
    for lane_id, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean)--delete
        # - lane_CAV_penetration: 这个车道上的CAV占比 1

        # all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[9:10]
        lane_info[0] = lane_info[0] / self.lane_max_num_vehs
        lane_info[2] = lane_info[2] / state_divisors[0]
        lane_info[3] = lane_info[3] / 20
        all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[16:17]
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        position, speed, acceleration, heading, road_id, lane_index, surroundings, surroundings_expand_4, surroundings_expand_6 = ego_info
        ego_lane = f'{road_id}_{lane_index}'
        ego_lane_stats[ego_id] = all_lane_stats[ego_lane]

    # convert to 2D array (18 * 6)
    all_lane_stats = np.array(list(all_lane_stats.values()))

    feature_vector = {}
    # feature_vector['road_structure'] = np.array([0, 0, bottle_neck_position_x, bottle_neck_position_y, 4,
    #                                              bottle_neck_position_x, bottle_neck_position_y, 1, 0, 2])
    feature_vector['road_structure'] = np.array([0, 0, 1, 0, 1, 1, 0, 1])
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
        feature_vectors_current[ego_id]['ego_lane_stats'] = ego_lane_stats[ego_id]

        shared_feature_vectors_current[ego_id] = feature_vector.copy()
        shared_feature_vectors_current[ego_id]['cav_stats'] = cav_stats
        shared_feature_vectors_current[ego_id]['lane_stats'] = all_lane_stats

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
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        size_obs = np.size(np.array(flat_list))
        return np.array(flat_list)

    def flatten_list(nested_list):
        flat_list = []
        for item in nested_list:
            # 检查是否为可迭代的对象（排除字符串和字节类型）
            if isinstance(item, collections.abc.Iterable) and not isinstance(item, (str, bytes)):
                flat_list.extend(flatten_list(item))  # 递归展平
            else:
                flat_list.append(item)  # 非可迭代对象直接添加
        return flat_list
    # ############################## 所有HDV的信息 ############################## 13
    hdv_stats = {}
    hdv_hist = {}
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
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # One-hot encode lane_index - 4
        # lane_index_one_hot = one_hot_encode(lane_index, list(range(edge_lane_num.get(road_id, 0))))
        # # 如果车道数不足4个, 补0 对齐到最多数量的lane num
        # if len(lane_index_one_hot) < 4:
        #     lane_index_one_hot += [0] * (4 - len(lane_index_one_hot))
        hdv_stats[hdv_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading] + road_id_one_hot
        hdv_hist[hdv_id] = []
        if self.use_hist_info:
            for i in range(self.hist_length - 1):
                self.vehicles_hist[f'hist_{self.hist_length - i}'][hdv_id] = copy.deepcopy(
                    self.vehicles_hist[f'hist_{self.hist_length - i - 1}'][hdv_id])
                hist_state = copy.deepcopy(self.vehicles_hist[f'hist_{self.hist_length - i}'][hdv_id][:5])
                hist_state /= state_divisors[:5]
                hdv_hist[hdv_id].extend(hist_state)
            self.vehicles_hist['hist_1'][hdv_id] = [position_x, position_y, speed, acceleration, heading]
            hist_state = copy.deepcopy(self.vehicles_hist['hist_1'][hdv_id][:5])
            hist_state /= state_divisors[:5]
            hdv_hist[hdv_id].extend(hist_state)
    # convert to 2D array (12 * 13)  - 12 is max number of HDVs
        # convert to 2D array (24 * 3)  - 24 is max number of HDVs
    for i in range(self.max_num_HDVs):
        if 'HDV_' + str(i) not in hdv_stats:
            hdv_stats['HDV_' + str(i)] = [0.0] * 9
            hdv_hist['HDV_' + str(i)] = [0.0] * (5 * self.hist_length)
    # hdv_stats = dict(sorted(hdv_stats.items(), key=lambda x: int(x[0].split('_')[1])))
    # hdv_hist = dict(sorted(hdv_hist.items(), key=lambda x: int(x[0].split('_')[1])))
    hdv_stats = np.array(list(hdv_stats.values()))
    hdv_hist = np.array(list(hdv_hist.values()))
    # hdv_stats = np.array(list(hdv_stats.values()))
    # if 0 < hdv_stats.shape[0] <= self.max_num_HDVs:
    #     # add 0 to make sure the shape is (12, 13)
    #     hdv_stats = np.vstack([hdv_stats, np.zeros((self.max_num_HDVs - hdv_stats.shape[0], 9))])
    # elif hdv_stats.shape[0] == 0:
    #     hdv_stats = np.zeros((self.max_num_HDVs, 9))
    # ############################## 所有CAV的信息 ############################## 13
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
    surround_2_stats = {key: {surround_key: [0.0] * (1 + self.hist_length * 5) for surround_key in surround_2_key} for key
                        in ego_ids}
    surround_4_stats = {key: {surround_key: [0.0] * (1 + self.hist_length * 5) for surround_key in surround_4_key} for key
                        in ego_ids}
    surround_6_stats = {key: {surround_key: [0.0] * (1 + self.hist_length * 5) for surround_key in surround_6_key} for key
                        in ego_ids}
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
        # 速度归一化  - 1
        normalized_speed = speed / state_divisors[2]
        # 加速度归一化  - 1
        normalized_acceleration = acceleration / state_divisors[3]
        # 位置归一化  - 2
        position_x, position_y = position
        normalized_position_x = position_x / state_divisors[0]
        normalized_position_y = position_y / state_divisors[1]
        # 转向归一化 - 1
        if heading < 0:
            heading += 360
        elif heading > 360:
            heading -= 360
        normalized_heading = heading / state_divisors[4]
        # One-hot encode road_id - 5
        road_id_one_hot = one_hot_encode(road_id, unique_edges)
        # ############################## 自己车的运动信息 ############################## 20
        ego_hist_4 = self.vehicles_hist['hist_4'][ego_id]
        ego_hist_3 = self.vehicles_hist['hist_3'][ego_id]
        ego_hist_2 = self.vehicles_hist['hist_2'][ego_id]
        ego_hist_1 = self.vehicles_hist['hist_1'][ego_id]

        relative_hist_4 = [(ego_hist_4[0] - position_x), (ego_hist_4[1] - position_y), (speed - ego_hist_4[2]),
                           (acceleration - ego_hist_4[3]), (ego_hist_4[4] - heading)] \
            if ego_hist_4 != [0.0]*5 else [0.0]*5
        relative_hist_4 = [rm / sd for rm, sd in zip(relative_hist_4, state_divisors[:5])]
        relative_hist_3 = [(ego_hist_3[0] - position_x), (ego_hist_3[1] - position_y), (speed - ego_hist_3[2]),
                           (acceleration - ego_hist_3[3]), (ego_hist_3[4] - heading)] \
            if ego_hist_3 != [0.0]*5 else [0.0]*5
        relative_hist_3 = [rm / sd for rm, sd in zip(relative_hist_3, state_divisors[:5])]
        relative_hist_2 = [(ego_hist_2[0] - position_x), (ego_hist_2[1] - position_y), (speed - ego_hist_2[2]),
                           (acceleration - ego_hist_2[3]), (ego_hist_2[4] - heading)] \
            if ego_hist_2 != [0.0]*5 else [0.0]*5
        relative_hist_2 = [rm / sd for rm, sd in zip(relative_hist_2, state_divisors[:5])]
        relative_hist_1 = [(ego_hist_1[0] - position_x), (ego_hist_1[1] - position_y), (speed - ego_hist_1[2]),
                           (acceleration - ego_hist_1[3]), (ego_hist_1[4] - heading)] \
            if ego_hist_1 != [0.0]*5 else [0.0]*5
        relative_hist_1 = [rm / sd for rm, sd in zip(relative_hist_1, state_divisors[:5])]
        relative_current = [0.0]*5
        ego_hist_motion[ego_id] = relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + relative_current
        ego_last_action = flatten_list(self.actor_action[ego_id]) if self.actor_action[ego_id] != [] else [0, 0]
        ego_last_action = [ego_last_action[0] / state_divisors[5], ego_last_action[1] / state_divisors[6]]

        cav_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading]
        ego_stats[ego_id] = [normalized_position_x, normalized_position_y, normalized_speed, normalized_acceleration,
                             normalized_heading] + road_id_one_hot + ego_last_action
        cav_hist[ego_id] = []
        if self.use_hist_info:
            for i in range(self.hist_length - 1):
                self.vehicles_hist[f'hist_{self.hist_length - i}'][ego_id] = copy.deepcopy(
                    self.vehicles_hist[f'hist_{self.hist_length - i - 1}'][ego_id])
                hist_state = copy.deepcopy(self.vehicles_hist[f'hist_{self.hist_length - i}'][ego_id][:5])
                hist_state /= state_divisors[:5]
                cav_hist[ego_id].extend(hist_state)
            self.vehicles_hist['hist_1'][ego_id] = [position_x, position_y, speed, acceleration, heading]
            hist_state = copy.deepcopy(self.vehicles_hist['hist_1'][ego_id][:5])
            hist_state /= state_divisors[:5]
            cav_hist[ego_id].extend(hist_state)

        if position_y < 5 and 90 <= heading < 180:
            next_node[ego_id] = [node_positions['J1'][0] / state_divisors[0],
                                 node_positions['J1'][1] / state_divisors[1]]
            dis_next_node = 1 - normalized_position_x
        elif position_x > 195 and 0 <= heading <= 90:
            next_node[ego_id] = [node_positions['J2'][0] / state_divisors[0],
                                 node_positions['J2'][1] / state_divisors[1]]
            dis_next_node = 1 - normalized_position_y
        elif position_y > 195 and 270 <= heading < 360:
            next_node[ego_id] = [node_positions['J3'][0] / state_divisors[0],
                                 node_positions['J3'][1] / state_divisors[1]]
            dis_next_node = normalized_position_x
        else:
            next_node[ego_id] = [node_positions['J0'][0] / state_divisors[0],
                                 node_positions['J0'][1] / state_divisors[1]]
            dis_next_node = normalized_position_y
        next_node[ego_id].append(dis_next_node)
    for i in range(self.max_num_CAVs):
        if 'CAV_' + str(i) not in cav_stats:
            cav_stats['CAV_' + str(i)] = [0.0] * 5
            cav_hist['CAV_' + str(i)] = [0.0] * (5 * self.hist_length)
    # cav_stats = dict(sorted(cav_stats.items(), key=lambda x: int(x[0].split('_')[1])))
    # cav_hist = dict(sorted(cav_hist.items(), key=lambda x: int(x[0].split('_')[1])))
    cav_stats = np.array(list(cav_stats.values()))
    cav_hist = np.array(list(cav_hist.values()))
        # ############################## 周车信息 ##############################
    for ego_id, ego_info in ego_statistics.items():
        position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
        position_x, position_y = position
        for index, (_, statistics) in enumerate(surroundings_2.items()):
            surround_ID, veh_type, relat_x, relat_y, relat_v, veh_acc, veh_heading = statistics[:7]
            surround_2_vehs_stats[ego_id].append(
                [veh_type, relat_x / state_divisors[0], relat_y / state_divisors[1], relat_v / state_divisors[2],
                 veh_acc / state_divisors[3], veh_heading / state_divisors[4]])
            target_hist_4 = self.vehicles_hist['hist_5'][surround_ID]
            target_hist_3 = self.vehicles_hist['hist_4'][surround_ID]
            target_hist_2 = self.vehicles_hist['hist_3'][surround_ID]
            target_hist_1 = self.vehicles_hist['hist_2'][surround_ID]
            target_hist_0 = self.vehicles_hist['hist_1'][surround_ID]
            relative_hist_4 = [(target_hist_4[0] - position_x), (target_hist_4[1] - position_y),
                               (speed - target_hist_4[2]), (acceleration - target_hist_4[3]), (target_hist_4[4] - heading)] \
                            if target_hist_4 != [0.0]*5 else [0.0]*5
            relative_hist_4 = [rm / sd for rm, sd in zip(relative_hist_4, state_divisors[:5])]
            relative_hist_3 = [(target_hist_3[0] - position_x), (target_hist_3[1] - position_y),
                               (speed - target_hist_3[2]), (acceleration - target_hist_3[3]), (target_hist_3[4] - heading)] \
                            if target_hist_3 != [0.0]*5 else [0.0]*5
            relative_hist_3 = [rm / sd for rm, sd in zip(relative_hist_3, state_divisors[:5])]
            relative_hist_2 = [(target_hist_2[0] - position_x), (target_hist_2[1] - position_y),
                               (speed - target_hist_2[2]), (acceleration - target_hist_2[3]), (target_hist_2[4] - heading)] \
                            if target_hist_2 != [0.0]*5 else [0.0]*5
            relative_hist_2 = [rm / sd for rm, sd in zip(relative_hist_2, state_divisors[:5])]
            relative_hist_1 = [(target_hist_1[0] - position_x), (target_hist_1[1] - position_y),
                               (speed - target_hist_1[2]), (acceleration - target_hist_1[3]), (target_hist_1[4] - heading)] \
                            if target_hist_1 != [0.0]*5 else [0.0]*5
            relative_hist_1 = [rm / sd for rm, sd in zip(relative_hist_1, state_divisors[:5])]
            relative_hist_0 = [(target_hist_0[0] - position_x), (target_hist_0[1] - position_y),
                               (speed - target_hist_0[2]), (acceleration - target_hist_0[3]), (target_hist_0[4] - heading)] \
                            if target_hist_0 != [0.0]*5 else [0.0]*5
            relative_hist_0 = [rm / sd for rm, sd in zip(relative_hist_0, state_divisors[:5])]
            relative_motion = [veh_type] + relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + relative_hist_0
            surround_2_stats[ego_id][_] = relative_motion
        for index, (_, statistics) in enumerate(surroundings_4.items()):
            surround_ID, veh_type, relat_x, relat_y, relat_v, veh_acc, veh_heading = statistics[:7]
            surround_4_vehs_stats[ego_id].append(
                [veh_type, relat_x / state_divisors[0], relat_y / state_divisors[1], relat_v / state_divisors[2],
                 veh_acc / state_divisors[3], veh_heading / state_divisors[4]])
            target_hist_4 = self.vehicles_hist['hist_5'][surround_ID]
            target_hist_3 = self.vehicles_hist['hist_4'][surround_ID]
            target_hist_2 = self.vehicles_hist['hist_3'][surround_ID]
            target_hist_1 = self.vehicles_hist['hist_2'][surround_ID]
            target_hist_0 = self.vehicles_hist['hist_1'][surround_ID]
            relative_hist_4 = [(target_hist_4[0] - position_x), (target_hist_4[1] - position_y),
                               (speed - target_hist_4[2]), (acceleration - target_hist_4[3]), (target_hist_4[4] - heading)] \
                            if target_hist_4 != [0.0]*5 else [0.0]*5
            relative_hist_4 = [rm / sd for rm, sd in zip(relative_hist_4, state_divisors[:5])]
            relative_hist_3 = [(target_hist_3[0] - position_x), (target_hist_3[1] - position_y),
                               (speed - target_hist_3[2]), (acceleration - target_hist_3[3]), (target_hist_3[4] - heading)] \
                            if target_hist_3 != [0.0]*5 else [0.0]*5
            relative_hist_3 = [rm / sd for rm, sd in zip(relative_hist_3, state_divisors[:5])]
            relative_hist_2 = [(target_hist_2[0] - position_x), (target_hist_2[1] - position_y),
                               (speed - target_hist_2[2]), (acceleration - target_hist_2[3]), (target_hist_2[4] - heading)] \
                            if target_hist_2 != [0.0]*5 else [0.0]*5
            relative_hist_2 = [rm / sd for rm, sd in zip(relative_hist_2, state_divisors[:5])]
            relative_hist_1 = [(target_hist_1[0] - position_x), (target_hist_1[1] - position_y),
                               (speed - target_hist_1[2]), (acceleration - target_hist_1[3]), (target_hist_1[4] - heading)] \
                            if target_hist_1 != [0.0]*5 else [0.0]*5
            relative_hist_1 = [rm / sd for rm, sd in zip(relative_hist_1, state_divisors[:5])]
            relative_hist_0 = [(target_hist_0[0] - position_x), (target_hist_0[1] - position_y),
                               (speed - target_hist_0[2]), (acceleration - target_hist_0[3]), (target_hist_0[4] - heading)] \
                            if target_hist_0 != [0.0]*5 else [0.0]*5
            relative_hist_0 = [rm / sd for rm, sd in zip(relative_hist_0, state_divisors[:5])]
            relative_motion = [veh_type] + relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + relative_hist_0
            surround_4_stats[ego_id][_] = relative_motion
        for index, (_, statistics) in enumerate(surroundings_6.items()):
            surround_ID, veh_type, relat_x, relat_y, relat_v, veh_acc, veh_heading = statistics[:7]
            surround_6_vehs_stats[ego_id].append(
                [veh_type, relat_x / state_divisors[0], relat_y / state_divisors[1], relat_v / state_divisors[2],
                 veh_acc / state_divisors[3], veh_heading / state_divisors[4]])
            target_hist_4 = self.vehicles_hist['hist_5'][surround_ID]
            target_hist_3 = self.vehicles_hist['hist_4'][surround_ID]
            target_hist_2 = self.vehicles_hist['hist_3'][surround_ID]
            target_hist_1 = self.vehicles_hist['hist_2'][surround_ID]
            target_hist_0 = self.vehicles_hist['hist_1'][surround_ID]
            relative_hist_4 = [(target_hist_4[0] - position_x), (target_hist_4[1] - position_y),
                               (speed - target_hist_4[2]), (acceleration - target_hist_4[3]), (target_hist_4[4] - heading)] \
                            if target_hist_4 != [0.0]*5 else [0.0]*5
            relative_hist_4 = [rm / sd for rm, sd in zip(relative_hist_4, state_divisors[:5])]
            relative_hist_3 = [(target_hist_3[0] - position_x), (target_hist_3[1] - position_y),
                               (speed - target_hist_3[2]), (acceleration - target_hist_3[3]), (target_hist_3[4] - heading)] \
                            if target_hist_3 != [0.0]*5 else [0.0]*5
            relative_hist_3 = [rm / sd for rm, sd in zip(relative_hist_3, state_divisors[:5])]
            relative_hist_2 = [(target_hist_2[0] - position_x), (target_hist_2[1] - position_y),
                               (speed - target_hist_2[2]), (acceleration - target_hist_2[3]), (target_hist_2[4] - heading)] \
                            if target_hist_2 != [0.0]*5 else [0.0]*5
            relative_hist_2 = [rm / sd for rm, sd in zip(relative_hist_2, state_divisors[:5])]
            relative_hist_1 = [(target_hist_1[0] - position_x), (target_hist_1[1] - position_y),
                               (speed - target_hist_1[2]), (acceleration - target_hist_1[3]), (target_hist_1[4] - heading)] \
                            if target_hist_1 != [0.0]*5 else [0.0]*5
            relative_hist_1 = [rm / sd for rm, sd in zip(relative_hist_1, state_divisors[:5])]
            relative_hist_0 = [(target_hist_0[0] - position_x), (target_hist_0[1] - position_y),
                               (speed - target_hist_0[2]), (acceleration - target_hist_0[3]), (target_hist_0[4] - heading)] \
                            if target_hist_0 != [0.0]*5 else [0.0]*5
            relative_hist_0 = [rm / sd for rm, sd in zip(relative_hist_0, state_divisors[:5])]
            relative_motion = [veh_type] + relative_hist_4 + relative_hist_3 + relative_hist_2 + relative_hist_1 + relative_hist_0
            surround_6_stats[ego_id][_] = relative_motion

    if len(ego_stats) != len(ego_ids):
        for ego_id in ego_ids:
            if ego_id not in ego_stats:
                ego_stats[ego_id] = [0.0] * 11

    # ############################## lane_statistics 的信息 ############################## 18
    # Initialize a list to hold all lane statistics
    all_lane_stats = {}
    all_lane_distribution = {}
    all_lane_state_simple = {}

    # Iterate over all possible lanes to get their statistics
    for lane_id, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean)--delete
        # - lane_CAV_penetration: 这个车道上的CAV占比 1

        # all_lane_stats[lane_id] = lane_info[:4] + lane_info[6:7] + lane_info[9:10]
        lane_info[0] = lane_info[0] / self.lane_max_num_vehs
        lane_info[2] = lane_info[2] / state_divisors[0]
        lane_info[3] = lane_info[3] / 20
        lane_info[17] = lane_info[17] / state_divisors[0:2]
        lane_info[18] = lane_info[18] / state_divisors[0:2]
        selected_lane_info = lane_info[17:18] + lane_info[18:19] + lane_info[:4] + lane_info[6:7] + lane_info[16:17]
        selected_lane_info = flatten_list(selected_lane_info)
        all_lane_stats[lane_id] = selected_lane_info
        all_lane_distribution[lane_id] = lane_info[19]
    for ego_id, ego_info in ego_statistics.items():
        # ############################## 自己车的信息 ############################## 13
        position, speed, acceleration, heading, road_id, lane_index, surroundings_2, surroundings_4, surroundings_6 = ego_info
        ego_lane = f'{road_id}_{lane_index}'
        ego_lane_stats[ego_id] = all_lane_stats[ego_lane]
        if ego_lane == 'E0_0':
            next_lane = 'E1_0'
        elif ego_lane == 'E1_0':
            next_lane = 'E0_0'
        elif ego_lane == 'E2_0':
            next_lane = 'E3_0'
        else:
            next_lane = 'E0_0'
        next_lane_stats[ego_id] = all_lane_stats[next_lane]

    # convert to 2D array (18 * 6)
    all_lane_stats = flatten_list(list(all_lane_stats.values()))
    all_lane_distribution = flatten_list(list(all_lane_distribution.values()))

    feature_vector = {}
    # feature_vector['road_structure'] = np.array([0, 0, bottle_neck_position_x, bottle_neck_position_y, 4,
    #                                              bottle_neck_position_x, bottle_neck_position_y, 1, 0, 2])
    feature_vector['road_structure'] = np.array([0, 0, 1, 0, 1, 1, 0, 1])

    feature_vectors_current = {}
    shared_feature_vectors_current = {}
    flat_surround_2_vehs = {key: [] for key in next_node.keys()}
    flat_surround_4_vehs = {key: [] for key in next_node.keys()}
    flat_surround_6_vehs = {key: [] for key in next_node.keys()}
    for ego_id in ego_statistics.keys():
        feature_vectors_current[ego_id] = feature_vector.copy()
        feature_vectors_current[ego_id]['next_node'] = next_node[ego_id]
        feature_vectors_current[ego_id]['self_stats'] = [1] + ego_stats[ego_id]
        feature_vectors_current[ego_id]['self_hist_stats'] = [1] + ego_hist_motion[ego_id]
        flat_surround_2_vehs[ego_id] = [item for sublist in surround_2_stats[ego_id].values() for item in sublist]
        if len(flat_surround_2_vehs[ego_id]) < 2*(1 + self.hist_length * 5):
            flat_surround_2_vehs[ego_id] += [0] * (2*(1 + self.hist_length * 5) - len(flat_surround_2_vehs[ego_id]))
        feature_vectors_current[ego_id]['surround_2_vehs_stats'] = flat_surround_2_vehs[ego_id]
        flat_surround_4_vehs[ego_id] = [item for sublist in surround_4_stats[ego_id].values() for item in sublist]
        if len(flat_surround_4_vehs[ego_id]) < 4*(1 + self.hist_length * 5):
            flat_surround_4_vehs[ego_id] += [0] * (4*(1 + self.hist_length * 5) - len(flat_surround_4_vehs[ego_id]))
        feature_vectors_current[ego_id]['surround_4_vehs_stats'] = flat_surround_4_vehs[ego_id]
        flat_surround_6_vehs[ego_id] = [item for sublist in surround_6_stats[ego_id].values() for item in sublist]
        if len(flat_surround_6_vehs[ego_id]) < 6*(1 + self.hist_length * 5):
            flat_surround_6_vehs[ego_id] += [0] * (6*(1 + self.hist_length * 5) - len(flat_surround_6_vehs[ego_id]))
        feature_vectors_current[ego_id]['surround_6_vehs_stats'] = flat_surround_6_vehs[ego_id]
        feature_vectors_current[ego_id]['ego_lane_stats'] = ego_lane_stats[ego_id]
        feature_vectors_current[ego_id]['next_lane_stats'] = next_lane_stats[ego_id]

        shared_feature_vectors_current[ego_id] = feature_vector.copy()
        shared_feature_vectors_current[ego_id]['hdv_stats'] = hdv_stats
        shared_feature_vectors_current[ego_id]['cav_stats'] = cav_stats
        shared_feature_vectors_current[ego_id]['lane_stats'] = all_lane_stats
        shared_feature_vectors_current[ego_id]['lane_distribution'] = all_lane_distribution

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
