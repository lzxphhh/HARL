import numpy as np

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.check_folder import check_folder
from tshub.utils.init_log import set_logger
from tshub.tshub_env.tshub_env import TshubEnvironment
from harl.envs.a_single_lane.env_utils.generate_scene_straight import generate_scenario
import random
import math
import time
import os
import csv
import traci

if __name__ == '__main__':
    path_convert = get_abs_path(__file__)
    set_logger(path_convert('./'))

    sumo_cfg = path_convert("env_utils/Straight_SUMO_files/scenario.sumocfg")
    net_file = path_convert("env_utils/Straight_SUMO_files/veh.net.xml")  # "bottleneck.net.xml"
    # 创建存储图像的文件夹
    image_save_folder = path_convert('./global/')
    check_folder(image_save_folder)
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    csv_save_folder = path_convert('./motion/' + now_time + '/')
    check_folder(csv_save_folder)

    # build a memory block for the surrounding_info
    observed_vehicles = []

    # 初始化环境
    tshub_env = TshubEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        is_map_builder_initialized=True,
        is_vehicle_builder_initialized=True,
        is_aircraft_builder_initialized=False,
        is_traffic_light_builder_initialized=False,
        # vehicle builder
        vehicle_action_type='lane_continuous_speed',  # 'lane_continuous_speed' 'lane'
        use_gui=True, num_seconds=100
    )

    # 开始仿真
    obs = tshub_env.reset()
    aggressive = 0.2
    cautious = 0.2
    normal = 0.6
    CAV_num = 0
    HDV_num = 20
    CAV_penetration = 0
    CAVs_id = []
    for i in range(CAV_num):
        CAVs_id.append(f'CAV_{i}')
    HDVs_id = []
    for i in range(HDV_num):
        HDVs_id.append(f'HDV_{i}')

    generate_scenario(
        aggressive=aggressive, cautious=cautious, normal=normal,
        use_gui=True, sce_name="Env_SingleLane", CAV_num=CAV_num, HDV_num=HDV_num, CAV_penetration=CAV_penetration, distribution="uniform")
    done = False
    last_acceleration = {}
    now_acceleration = {}
    while not done:
        actions = {'vehicle': {veh_id: (-1, -1) for veh_id in obs['vehicle'].keys()}}
        if obs['vehicle']:
            step = info['step_time']
            speed_command = np.sin(step / 5) * 3 + 10
            actions['vehicle']['Leader'] = (0, speed_command)

        obs, reward, info, done = tshub_env.step(actions=actions)

        # 记录render车辆轨迹信息
        if obs['vehicle']:
            DRAC = {veh_id: 0 for veh_id in obs['vehicle'].keys()}
            if last_acceleration == {} and now_acceleration == {}:
                last_acceleration = {veh_id: 0 for veh_id in obs['vehicle'].keys()}
                now_acceleration = {veh_id: 0 for veh_id in obs['vehicle'].keys()}
            else:
                now_acceleration = {veh_id: obs['vehicle'][veh_id]['acceleration'] for veh_id in
                                    obs['vehicle'].keys()}
            delta_acc = {veh_id: now_acceleration[veh_id] - last_acceleration[veh_id] for veh_id in
                         obs['vehicle'].keys()}
            last_acceleration = now_acceleration
            for veh_id in obs['vehicle'].keys():
                front_vehicle = traci.vehicle.getLeader(veh_id)
                if front_vehicle not in [None, ()] and front_vehicle[0] != '':
                    front_id = front_vehicle[0]
                    front_speed = traci.vehicle.getSpeed(front_id)
                    front_position = traci.vehicle.getPosition(front_id)[0]
                    ego_speed = obs['vehicle'][veh_id]['speed']
                    ego_position = obs['vehicle'][veh_id]['position'][0]
                    gap = front_position - ego_position - 5
                    delta_v = ego_speed - front_speed
                    if delta_v > 0:
                        DRAC_i = delta_v ** 2 / gap
                    else:
                        DRAC_i = 0
                else:
                    DRAC_i = 0
                DRAC[veh_id] = DRAC_i

                veh_info = [int(info["step_time"]),
                            obs['vehicle'][veh_id]['position'][0],
                            # obs['vehicle'][veh_id]['position'][1],
                            obs['vehicle'][veh_id]['speed'],
                            obs['vehicle'][veh_id]['acceleration'],
                            delta_acc[veh_id],
                            DRAC[veh_id]]
                csv_path = csv_save_folder + '/' + veh_id + '_run_info.csv'
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(veh_info)
        fig = tshub_env.render(
            mode='sumo_gui',
            save_folder=image_save_folder,
            focus_id='HDV_0', focus_type='vehicle', focus_distance=200,
        )

        step_time = int(info["step_time"])
        logger.info(f"SIM: {step_time}")
        time.sleep(0.1)

    tshub_env._close_simulation()