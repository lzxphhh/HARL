import time

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from veh_env import VehEnvironment
from veh_env_wrapper import VehEnvWrapper

# 获得全局路径
path_convert = get_abs_path(__file__)
# 设置日志 -- tshub自带的给环境的
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level='INFO')

if __name__ == '__main__':
    # base env
    sumo_cfg = path_convert("SUMO_files/scenario.sumocfg")
    num_seconds = 1500  # 秒
    vehicle_action_type = 'lane_continuous_speed'
    use_gui = True
    trip_info = None

    # for veh wrapper
    scene_name = "Env_SingleLane"
    max_num_CAVs = 32
    max_num_HDVs = 32
    penetration_CAV = 0.1  # only 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 are valid
    num_CAVs = 4  # num_CAVs/penetration_CAV should be an integer multiple of 10, except for penetration_CAV=1
    num_HDVs = 16
    lane_max_num_vehs = 15
    warmup_steps = 0
    ego_ids = ['CAV_0', 'CAV_1', 'CAV_2',
               'CAV_3',
               # 'CAV_4',
               # 'CAV_5', 'CAV_6', 'CAV_7',
               # 'CAV_8', 'CAV_9',
               ]  # the number of ids should be equal to num_CAVs
    edge_ids = ['E0', 'E1', 'E2', 'E3', ]
    edge_lane_num = {'E0': 1,
                     'E1': 1,
                     'E2': 1,
                     'E3': 1,
                     }  # 每一个 edge 对应的车道数
    node_positions = [(0, 0), (200, 0), (200, 200), (0,200)]
    calc_features_lane_ids = ['E0_0',
                              'E1_0',
                              'E2_0',
                              'E3_0',
                              ]  # 计算对应的 lane 的信息
    log_path = path_convert('./log/check_veh_env')
    delta_t = 1.0
    hdv_aggressive = 0.2
    hdv_cautious = 0.2
    hdv_normal = 0.6

    ac_env = VehEnvironment(
        sumo_cfg=sumo_cfg,
        num_seconds=num_seconds,
        vehicle_action_type=vehicle_action_type,
        use_gui=use_gui,
        trip_info=trip_info,
    )

    ac_env_wrapper = VehEnvWrapper(
        env=ac_env,
        name_scenario=scene_name,
        max_num_CAVs=max_num_CAVs,  # 最大的 CAV 数量
        max_num_HDVs=max_num_HDVs,  # 最大的 HDV 数量
        CAV_penetration=penetration_CAV,
        num_CAVs=num_CAVs,
        num_HDVs=num_HDVs,
        lane_max_num_vehs=lane_max_num_vehs,  # 每条车道车辆最大数量
        warmup_steps=warmup_steps,
        ego_ids=ego_ids,
        edge_ids=edge_ids,
        edge_lane_num=edge_lane_num,
        node_positions=node_positions,
        calc_features_lane_ids=calc_features_lane_ids,
        filepath=log_path,
        use_gui=use_gui,
        delta_t=delta_t,
        aggressiv=hdv_aggressive,  # aggressive 的概率
        cautious=hdv_cautious,  # cautious 的概率
        normal=hdv_normal,  # normal 的概率
        strategy='base',  # MARL 的策略- feature extraction
        use_hist_info=False,  # 是否使用历史信息
        hist_length=0,  # 历史信息的长度
    )

    for constant_speed in range(0, 5):  # 测试不同的速度
        """
        在simple中
        0： 不换道
        1： 左换道
        2： 右换道
        3： 加速
        4： 减速
        """
        logger.info(f'SIM: TEST Constant Speed : {constant_speed}')
        done = False
        ac_env_wrapper.reset()
        while not done:
            # 获取环境中所有车辆的ID
            # 为每个车辆生成一个动作
            action = {ego_id: constant_speed for ego_id in ac_env_wrapper.ego_ids}
            states, shared_state, rewards, truncated, dones, infos = ac_env_wrapper.step(action=action)
            done = all([dones[_ego_id] for _ego_id in ac_env_wrapper.ego_ids])
            logger.info(f'SIM: Applied action: {ac_env_wrapper.action_command}')
            logger.info(f'SIM: Speed: {ac_env_wrapper.current_speed}')
            logger.info(f'SIM: Lane: {ac_env_wrapper.current_lane}')
            logger.info(f'SIM: Reward: {rewards}')
            logger.info(f'SIM: Info: {infos}')
            logger.info(f'SIM: Warn Vehicle: {ac_env_wrapper.warn_ego_ids}')
            logger.info(f'SIM: Collision Vehicle: {ac_env_wrapper.coll_ego_ids}')
            time.sleep(0.1)
            if done:
                print('done')

    ac_env_wrapper.close()
