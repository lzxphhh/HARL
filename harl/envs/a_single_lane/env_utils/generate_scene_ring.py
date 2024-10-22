import numpy as np
import traci
import libsumo as ls
import random

def generate_scenario(
        aggressive, cautious, normal,
        use_gui: bool, sce_name: str, CAV_num: int, HDV_num: int, CAV_penetration: float, distribution: str):
    """
    Mixed Traffic Flow (MTF) scenario generation: v_0 = 10 m/s, v_max = 20 m/s
    use_gui: false for libsumo, true for traci
    sce_name: scenario name, e.g., "Env_Bottleneck"
    CAV_num: number of CAVs
    CAV_penetration: CAV penetration rate, only 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 can be used
    - the number of vehicles in the scenario is determined by veh_num=CAV_num/CAV_penetration
    - the number of HDVs is determined by HDV_num=veh_num-CAV_num
    - 3 types of HDVs are randomly generated, and the parameters of each type of HDVs are defined in .rou.xml
    --- HDV_0: aggressive, 0.2 probability
    --- HDV_1: cautious, 0.2 probability
    --- HDV_2: normal, 0.6 probability
    distribution: "random" or "uniform"
    """
    veh_num = CAV_num + HDV_num
    lane_veh_num = veh_num / 4
    veh_gap = 90 / lane_veh_num - 5
    start_speed = (veh_gap - 2) / 2
    # generate HDVs with different driving behaviors
    random_numbers_HDV = [random.random() for _ in range(HDV_num)]
    random_HDVs = []
    for i in range(HDV_num):
        if random_numbers_HDV[i] < aggressive:
            random_HDVs.append(0)
        elif random_numbers_HDV[i] < aggressive + cautious:
            random_HDVs.append(1)
        else:
            random_HDVs.append(2)
    random_veh_type = [random.random() for _ in range(veh_num)]
    random_veh_distribution = []
    veh_route = []
    for i in range(veh_num):
        if random_veh_type[i] < CAV_penetration:
            random_veh_distribution.append(1)
        else:
            random_veh_distribution.append(0)
        if i < lane_veh_num:
            veh_route.append(0)
        elif i < 2 * lane_veh_num:
            veh_route.append(1)
        elif i < 3 * lane_veh_num:
            veh_route.append(2)
        else:
            veh_route.append(3)

    if use_gui:
        scene_change = traci.vehicle
    else:
        scene_change = ls.vehicle

    # random - CAVs are randomly distributed
    if distribution == "random":
        i_CAV = 0
        i_HDV = 0
        for i_veh in range(veh_num):
            if random_veh_distribution[i_veh] == 1 and i_CAV < CAV_num:
                scene_change.add(
                    vehID=f'CAV_{i_CAV}',
                    typeID='ego',
                    routeID=f'route_{int(veh_route[i_veh])}',
                    depart="now",
                    departPos="random",
                    departLane="random",
                    departSpeed=f'{start_speed}',
                )
                i_CAV += 1
            else:
                scene_change.add(
                    vehID=f'HDV_{i_HDV}',
                    typeID=f'HDV_{int(random_HDVs[i_veh])}',
                    routeID=f'route_{int(veh_route[i_veh])}',
                    depart="now",
                    departPos="random",
                    departLane="random",
                    departSpeed=f'{start_speed}',
                )
                i_HDV += 1

    # uniform - CAVs are uniformly distributed
    else:
        i_CAV = 0
        i_HDV = 0
        if CAV_penetration == 0.0:
            distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif CAV_penetration == 0.1:
            distribution = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif CAV_penetration == 0.2:
            distribution = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        elif CAV_penetration == 0.3:
            distribution = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        elif CAV_penetration == 0.4:
            distribution = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        elif CAV_penetration == 0.5:
            distribution = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        elif CAV_penetration == 0.6:
            distribution = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        elif CAV_penetration == 0.7:
            distribution = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]
        elif CAV_penetration == 0.8:
            distribution = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        elif CAV_penetration == 0.9:
            distribution = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        else:
            distribution = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for i_all in range(veh_num):
            pos_change = -3+3*random.random()
            if distribution[i_all % 10] == 1 and i_CAV < CAV_num:
                veh_type = 'ego'
                veh_id = f'CAV_{i_CAV}'
                i_CAV += 1
            else:
                veh_type = f'HDV_{int(random_HDVs[i_HDV])}'
                veh_id = f'HDV_{i_HDV}'
                i_HDV += 1
            E0_start = 5
            E1_start = 5
            E2_start = 5
            E3_start = 5
            if i_all < lane_veh_num:
                route_id = 'route_0'
                if i_all == 0:
                    depart_pos = E0_start + 5
                else:
                    depart_pos = depart_pos + veh_gap + 5
            elif i_all < 2 * lane_veh_num:
                route_id = 'route_1'
                if i_all == lane_veh_num:
                    depart_pos = E1_start + 5
                else:
                    depart_pos = depart_pos + veh_gap + 5
            elif i_all < 3 * lane_veh_num:
                route_id = 'route_2'
                if i_all == 2 * lane_veh_num:
                    depart_pos = E2_start + 5
                else:
                    depart_pos = depart_pos + veh_gap + 5
            else:
                route_id = 'route_3'
                if i_all == 3 * lane_veh_num:
                    depart_pos = E3_start + 5
                else:
                    depart_pos = depart_pos + veh_gap + 5
            scene_change.add(
                vehID=veh_id,
                typeID=veh_type,
                routeID=route_id,
                depart="now",
                departPos=float(depart_pos),
                departLane="random",
                departSpeed=f'{start_speed}',
            )