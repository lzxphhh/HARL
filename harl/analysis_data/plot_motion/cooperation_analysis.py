import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d, CubicSpline
from numpy.polynomial.polynomial import Polynomial

def custom_formatter(x, pos):
    return f'{int(x)}'
def load_vehicle_data(directory):
    csv_files = glob.glob(f'{directory}/*.csv')
    vehicle_data = {}
    lane_data = {0: [], 1: [], 2: [], 3: []}
    for file in csv_files:
        vehicle_id = file.split('/')[-1].split('.')[0][:6]  # Assuming filename is the vehicle ID
        data = pd.read_csv(file, header=None, names=['time', 'position_x', 'position_y', 'speed', 'safety_level'])
        data['acceleration'] = 0
        data.loc[1:, 'acceleration'] = data['speed'].diff().fillna(0)
        vehicle_data[vehicle_id] = data
        lane_changes = []
        current_lane = None

        for idx, row in data.iterrows():
            position_y = row['position_y']

            if 3 < position_y < 6:
                lane = 3
            elif 0 < position_y < 3:
                lane = 2
            elif -3 < position_y < 0:
                lane = 1
            elif -6 < position_y < -3:
                lane = 0
            else:
                continue  # 忽略不在车道范围内的数据

            if current_lane is None:
                current_lane = lane
                lane_changes.append([lane, idx, idx])
            elif lane != current_lane:
                lane_changes[-1][2] = idx - 1
                lane_changes.append([lane, idx, idx])
                current_lane = lane
            else:
                lane_changes[-1][2] = idx

        # 将轨迹添加到对应车道
        for change in lane_changes:
            lane, start_idx, end_idx = change
            segment = data.iloc[start_idx:end_idx + 1].copy()
            segment['vehicle_id'] = vehicle_id
            segment['lane_id'] = lane
            lane_data[lane].append(segment)
    return vehicle_data, lane_data

def smooth_data(data, window_size=10, poly_order=3):
    return savgol_filter(data, window_size, poly_order)

def interpolate_data_B(x, y, num_points=10):
    # 去除相邻重复点
    unique_x = []
    unique_y = []
    for i in range(len(x)-1):
        if x[i] != x[i + 1]:
            unique_x.append(x[i])
            unique_y.append(y[i])

    unique_x = np.array(unique_x)
    unique_y = np.array(unique_y)

    # 进行Cubic Spline插值
    cs = CubicSpline(unique_x, unique_y)

    # 生成插值曲线的x坐标
    x_interp = np.linspace(unique_x.min(), unique_x.max(), num_points)
    # 计算插值曲线的y坐标
    y_interp = cs(x_interp)
    return x_interp, y_interp
def interpolate_data(x, y, num_points=10):
    # 去除相邻重复点，仅保留最后一个点的值
    filtered_x = [x[0]]
    filtered_y = [y[0]]
    for i in range(1, len(x)):
        if x[i] != x[i - 1]:
            filtered_x.append(x[i])
            filtered_y.append(y[i])
        else:
            filtered_x[-1] = x[i]
            filtered_y[-1] = y[i]

    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)

    # 插值生成平滑曲线
    interp_x = []
    interp_y = []

    for i in range(len(filtered_x) - 1):
        # 使用前1, 当前, 后2共4个点生成多项式曲线
        start = max(0, i - 1)
        end = min(len(filtered_x), i + 3)

        segment_x = filtered_x[start:end]
        segment_y = filtered_y[start:end]

        if len(segment_x) < 4:  # 如果点不足4个，则扩展至足够数量
            continue

        # 使用多项式拟合
        poly = Polynomial.fit(segment_x, segment_y, 3)

        # 生成当前点到下一个点之间的插值点
        x_new = np.linspace(filtered_x[i], filtered_x[i + 1], num_points)
        y_new = poly(x_new)

        interp_x.extend(x_new)
        interp_y.extend(y_new)

    # 添加最后一个点
    interp_x.append(filtered_x[-1])
    interp_y.append(filtered_y[-1])
    return interp_x, interp_y

# Function to plot speed and safety level curves with smoothing
def plot_vehicle_motion(vehicle_data, vehicle_ids):
    color_index = {
                   # 'CAV_00': 'blue', 'CAV_01': 'blue', 'CAV_02': 'blue', 'CAV_03': 'blue', 'CAV_04': 'blue', 'CAV_05': 'blue',
                   'CAV_00': 'darkred', 'CAV_01': 'darkred', 'CAV_02': 'darkred', 'CAV_03': 'darkred', 'CAV_04': 'darkred', 'CAV_05': 'darkred',
                   # 'HDV_00': 'black', 'HDV_01': 'black', 'HDV_02': 'black', 'HDV_03': 'black', 'HDV_04': 'black',
                   # 'HDV_05': 'black', 'HDV_06': 'black', 'HDV_07': 'black', 'HDV_08':'black', 'HDV_09': 'black',
                   # 'HDV_10': 'black', 'HDV_11': 'black', 'HDV_12': 'black', 'HDV_13': 'black', 'HDV_14': 'black'
                   'HDV_00': 'gray', 'HDV_01': 'gray', 'HDV_02': 'gray', 'HDV_03': 'gray', 'HDV_04': 'gray',
                   'HDV_05': 'gray', 'HDV_06': 'gray', 'HDV_07': 'gray', 'HDV_08': 'gray', 'HDV_09': 'gray',
                   'HDV_10': 'gray', 'HDV_11': 'gray', 'HDV_12': 'gray', 'HDV_13': 'gray', 'HDV_14': 'gray'
    }

    save_folder = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/figure_output/cooperation_analysis/' + 'HIAHR_DIU_MAPPO/'  # HIAHR_DIU_MAPPO, HIAHR_MAPPO, SUMO
    plt.figure(figsize=(20, 10))
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        plt.plot(data['position_x'], data['position_y'], alpha=0.9, color=color_index[vehicle_id], linewidth=3)
    plt.axhline(y=4.8, color='purple', linestyle='--', linewidth=3)
    plt.axhline(y=1.6, color='purple', linestyle='--', linewidth=3)
    plt.axhline(y=-1.6, color='purple', linestyle='--', linewidth=3)
    plt.axhline(y=-4.8, color='purple', linestyle='--', linewidth=3)
    plt.xlabel('x (m)', fontsize=42)
    plt.ylabel('y (m)', fontsize=42)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xticks([200, 300, 400, 500, 600, 700], fontsize=34)
    plt.yticks([-6, -4, -2, 0, 2, 4, 6], fontsize=34)
    plt.xlim(190, 700)
    save_name = save_folder + 'x-y.png'
    plt.savefig(save_name)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(28, 8))
    # for vehicle_id in vehicle_ids:
    #     if vehicle_id not in vehicle_data:
    #         print(f"No data found for {vehicle_id}")
    #         continue
    #     data = vehicle_data[vehicle_id]
    #     smoothed_speed = smooth_data(data['speed'], window_size=5, poly_order=3)
    #     # plt.plot(data['time'], data['speed'], alpha=0.5, color=color_index[vehicle_id], linewidth=3)
    #     plt.plot(data['time'], smoothed_speed, alpha=0.8, color=color_index[vehicle_id], linewidth=5)
    # plt.xlabel('t (s)', fontsize=40)
    # plt.ylabel('v (m/s)', fontsize=40)
    # # plt.legend(fontsize=14)
    # plt.grid(True)
    # # Set custom x-axis ticks and labels
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    # plt.xticks([0, 10, 20, 30, 40, 50], fontsize=40)
    # plt.yticks([0, 3, 6, 9, 12, 15], fontsize=40)
    # plt.xlim(0, 40)
    # plt.ylim(0, 17)
    # save_name = save_folder + 't-v.png'
    # plt.savefig(save_name)
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(20, 10))
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        smoothed_speed = smooth_data(data['speed'], window_size=10, poly_order=5)
        plt.plot(data['position_x'], smoothed_speed, alpha=0.9, color=color_index[vehicle_id], linestyle='-', linewidth=3)

        # x_smooth, y_smooth = interpolate_data(data['position_x'], data['speed'], num_points=100)
        # x_smooth, y_smooth = interpolate_data_B(data['position_x'], data['speed'], num_points=1000)
        # plt.plot(x_smooth, y_smooth, alpha=0.9, color=color_index[vehicle_id], linestyle='-', linewidth=3)

    plt.xlabel('x (m)', fontsize=42)
    plt.ylabel('v (m/s)', fontsize=42)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xticks([300, 400, 500, 600, 700], fontsize=34)
    plt.yticks([0, 3, 6, 9, 12, 15], fontsize=34)
    plt.xlim(290, 600)
    plt.ylim(0, 17)
    save_name = save_folder + 'x-v.png'
    plt.savefig(save_name)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 10))
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        plt.plot(data['position_x'], data['speed'], alpha=0.9, color=color_index[vehicle_id], linestyle='-', linewidth=3)
    plt.xlabel('x (m)', fontsize=42)
    plt.ylabel('v (m/s)', fontsize=42)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xticks([300, 400, 500, 600, 700], fontsize=34)
    plt.yticks([0, 3, 6, 9, 12, 15], fontsize=34)
    plt.xlim(290, 600)
    plt.ylim(0, 17)
    save_name = save_folder + 'x-v-origin.png'
    plt.savefig(save_name)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        plt.plot(data['time'], data['position_x'], color=color_index[vehicle_id], linestyle='-',
                 linewidth=3)
    # plt.xlabel('x (m)', fontsize=42)
    # plt.ylabel('v (m/s)', fontsize=42)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    # plt.xticks([300, 400, 500, 600, 700], fontsize=34)
    # plt.yticks([0, 3, 6, 9, 12, 15], fontsize=34)
    plt.xlim(15, 40)
    plt.ylim(440, 520)
    save_name = save_folder + 't-x-large.png'
    plt.savefig(save_name)
    plt.tight_layout()
    plt.show()

def plot_lane_data(lane_data):
    color_index = {
                   # 'CAV_00': 'blue', 'CAV_01': 'blue', 'CAV_02': 'blue', 'CAV_03': 'blue', 'CAV_04': 'blue', 'CAV_05': 'blue',
                   'CAV_00': 'darkred', 'CAV_01': 'darkred', 'CAV_02': 'darkred', 'CAV_03': 'darkred', 'CAV_04': 'darkred', 'CAV_05': 'darkred',
                   # 'HDV_00': 'black', 'HDV_01': 'black', 'HDV_02': 'black', 'HDV_03': 'black', 'HDV_04': 'black',
                   # 'HDV_05': 'black', 'HDV_06': 'black', 'HDV_07': 'black', 'HDV_08': 'black', 'HDV_09': 'black',
                   # 'HDV_10': 'black', 'HDV_11': 'black', 'HDV_12': 'black', 'HDV_13': 'black', 'HDV_14': 'black'
                   'HDV_00': 'gray', 'HDV_01': 'gray', 'HDV_02': 'gray', 'HDV_03': 'gray', 'HDV_04': 'gray',
                   'HDV_05': 'gray', 'HDV_06': 'gray', 'HDV_07': 'gray', 'HDV_08': 'gray', 'HDV_09': 'gray',
                   'HDV_10': 'gray', 'HDV_11': 'gray', 'HDV_12': 'gray', 'HDV_13': 'gray', 'HDV_14': 'gray'
    }

    save_folder = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/figure_output/cooperation_analysis/' + 'HIAHR_DIU_MAPPO/'  # HIAHR_DIU_MAPPO, HIAHR_MAPPO, SUMO
    # 创建子图
    fig, axs = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
    lane_titles = ['Lane 0', 'Lane 1', 'Lane 2', 'Lane 3']

    for lane in range(4):
        for segment in lane_data[lane]:
            vehicle_id = segment['vehicle_id'].iloc[0]
            axs[lane].plot(segment['time'], segment['position_x'], color=color_index[vehicle_id])
            # 标注起始点和终止点
            start_point = segment.iloc[0]
            end_point = segment.iloc[-1]
            axs[lane].scatter(start_point['time'], start_point['position_x'], color=color_index[vehicle_id], edgecolor='black', zorder=3)
            axs[lane].scatter(end_point['time'], end_point['position_x'], color=color_index[vehicle_id], edgecolor='black', zorder=3)

        axs[lane].set_title(lane_titles[lane], fontsize=24)
        axs[lane].set_xlabel('Time (s)', fontsize=24)
        axs[lane].set_ylabel('Position (m)', fontsize=24)
        axs[3 - lane].set_xticks([0, 10, 20, 30, 40, 50])
        axs[3 - lane].set_yticks([200, 300, 400, 500, 600, 700])
        axs[lane].tick_params(axis='both', which='major', labelsize=20)
        axs[3 - lane].set_xlim(-1, 50)
        axs[3 - lane].set_ylim(160, 700)
        # axs[3 - lane].legend()
    save_name = save_folder + 'lane-motion.png'
    plt.savefig(save_name)
    plt.tight_layout()
    plt.show()

def plot_3D_motion(vehicle_data, vehicle_ids):
    color_index = {
                   # 'CAV_00': 'blue', 'CAV_01': 'blue', 'CAV_02': 'blue', 'CAV_03': 'blue', 'CAV_04': 'blue', 'CAV_05': 'blue',
                   'CAV_00': 'darkred', 'CAV_01': 'darkred', 'CAV_02': 'darkred', 'CAV_03': 'darkred', 'CAV_04': 'darkred', 'CAV_05': 'darkred',
                   # 'HDV_00': 'black', 'HDV_01': 'black', 'HDV_02': 'black', 'HDV_03': 'black', 'HDV_04': 'black',
                   # 'HDV_05': 'black', 'HDV_06': 'black', 'HDV_07': 'black', 'HDV_08': 'black', 'HDV_09': 'black',
                   # 'HDV_10': 'black', 'HDV_11': 'black', 'HDV_12': 'black', 'HDV_13': 'black', 'HDV_14': 'black'
                   'HDV_00': 'gray', 'HDV_01': 'gray', 'HDV_02': 'gray', 'HDV_03': 'gray', 'HDV_04': 'gray',
                   'HDV_05': 'gray', 'HDV_06': 'gray', 'HDV_07': 'gray', 'HDV_08': 'gray', 'HDV_09': 'gray',
                   'HDV_10': 'gray', 'HDV_11': 'gray', 'HDV_12': 'gray', 'HDV_13': 'gray', 'HDV_14': 'gray'
    }

    save_folder = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/figure_output/cooperation_analysis/' + 'HIAHR_DIU_MAPPO/'  # HIAHR_DIU_MAPPO, HIAHR_MAPPO, SUMO
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        ax.plot(data['position_x'], data['position_y'], data['speed'], alpha=0.9, color=color_index[vehicle_id], linewidth=1)
    # 设置标签和标题
    ax.set_xlabel('Position X (m)')
    ax.set_ylabel('Position Y (m)')
    ax.set_zlabel('Speed (m/s)')
    ax.set_title('3D Trajectory of Vehicles')

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show()
# Main execution
if __name__ == "__main__":
    # Directory containing CSV files
    directory = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/cooperation_analysis/HIAHR_DIU_MAPPO_15_2024-08-04-18-17-00'  # Update this path

    # Load all vehicle data
    vehicle_data, lane_data = load_vehicle_data(directory)

    # Plot curves for specific vehicles
    veh_list = ['CAV_00', 'CAV_01', 'CAV_02', 'CAV_03', 'CAV_04', 'CAV_05',
                'HDV_00', 'HDV_01', 'HDV_02', 'HDV_03', 'HDV_04',
                'HDV_05', 'HDV_06', 'HDV_07', 'HDV_08', 'HDV_09',
                'HDV_10', 'HDV_11', 'HDV_12', 'HDV_13', 'HDV_14']
    plot_vehicle_motion(vehicle_data, veh_list)
    plot_lane_data(lane_data)

    # plot_3D_motion(vehicle_data, veh_list)
