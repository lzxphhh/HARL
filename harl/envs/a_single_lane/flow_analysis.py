import pandas as pd
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_vehicle_data(directory):
    csv_files = glob.glob(f'{directory}/*.csv')
    vehicle_data = {}
    for file in csv_files:
        # 使用正则表达式提取文件名中的车辆ID
        match = re.search(r'(HDV_\d+|CAV_\d+|Leader)', os.path.basename(file))
        if match:
            vehicle_id = match.group(0)  # 提取匹配的车辆ID
            data = pd.read_csv(file, header=None, names=['time', 'position_x', 'speed', 'acceleration', 'delta_acc', 'DRAC'])
            vehicle_data[vehicle_id] = data
    return vehicle_data


def compute_overall_average_speed(vehicle_data):
    total_speed = 0
    total_acceleration = 0
    total_delta_acc = 0
    total_count = 0
    for vehicle_id, data in vehicle_data.items():
        total_speed += data['speed'].sum()
        total_acceleration += data['acceleration'].abs().sum()
        total_delta_acc += data['delta_acc'].abs().sum()
        total_count += data['speed'].count()

    overall_average_speed = total_speed / total_count if total_count > 0 else 0
    overall_average_acceleration = total_acceleration / total_count if total_count > 0 else 0
    overall_average_delta_acc = total_delta_acc / total_count if total_count > 0 else 0
    return overall_average_speed, overall_average_acceleration, overall_average_delta_acc

def sort_vehicle_ids_by_position(vehicle_data):
    """
    Sort vehicles so that Leader is first, followed by all other vehicles (CAV and HDV) sorted by position_x
    at the first time step in descending order, regardless of type.
    """
    # Leader comes first
    leader_ids = ['Leader']

    # Gather all other vehicles (CAV and HDV together)
    other_vehicles = [vid for vid in vehicle_data.keys() if vid != 'Leader']

    # Sort CAV and HDV by position_x at the first timestamp in descending order
    vehicles_sorted_by_position = sorted(other_vehicles, key=lambda vid: vehicle_data[vid]['position_x'].iloc[0],
                                         reverse=True)

    # Combine Leader first, then all other vehicles sorted by position
    sorted_vehicle_ids = leader_ids + vehicles_sorted_by_position

    # Assign new numeric IDs starting from 0
    new_numeric_ids = {vid: idx for idx, vid in enumerate(sorted_vehicle_ids)}

    return new_numeric_ids

def plot_vehicle_trajectories_surface(vehicle_data, sorted_vehicle_ids, save_path):
    # 获取最大时间步长
    max_time = max(data['time'].max() for data in vehicle_data.values())

    # 创建网格
    vehicle_ids = list(sorted_vehicle_ids.keys())  # Sorted vehicle IDs
    X, Y = np.meshgrid(range(int(max_time) + 1), range(len(vehicle_ids)))
    Z = np.zeros_like(X, dtype=float)

    # 填充 Z（速度）值
    for vehicle_id, data in vehicle_data.items():
        vid_num = sorted_vehicle_ids[vehicle_id]
        for i, row in data.iterrows():
            time = int(row['time'])
            speed = row['speed']
            Z[vid_num, time] = speed

    # Mask invalid data (where Z is 0)
    Z_masked = np.ma.masked_where(Z == 0, Z)

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D曲面图 (only valid data, masked invalid Z values)
    ax.plot_surface(X, Y, Z_masked, cmap='viridis', edgecolor='none', alpha=0.4)  # Adjust alpha for transparency

    # 绘制除 Leader 之外的 3D 曲线
    for vehicle_id, vid_num in sorted_vehicle_ids.items():
        if vehicle_id == 'Leader':
            continue  # Skip Leader for now, will plot it last

        # Mask invalid data (where Z is 0)
        valid_mask = Z[vid_num, :] != 0
        valid_X = X[vid_num, :][valid_mask]
        valid_Y = Y[vid_num, :][valid_mask]
        valid_Z = Z[vid_num, :][valid_mask]

        # Select color based on vehicle type
        color = 'b' if 'CAV' in vehicle_id else 'gray'

        # Plot only valid data points as 3D line plot
        ax.plot(valid_X, valid_Y, valid_Z, color=color, label=vehicle_id, linewidth=2)

    # 绘制 Leader 曲线 (最后绘制)
    leader_id = 'Leader'
    vid_num = sorted_vehicle_ids[leader_id]
    valid_mask = Z[vid_num, :] != 0
    valid_X = X[vid_num, :][valid_mask]
    valid_Y = Y[vid_num, :][valid_mask]
    valid_Z = Z[vid_num, :][valid_mask]

    # Plot Leader's curve in red
    ax.plot(valid_X, valid_Y, valid_Z, color='r', label='Leader', linewidth=2)

    # 设置轴标签
    ax.set_xlabel('Time')
    ax.set_ylabel('Vehicle ID')
    ax.set_zlabel('Speed')
    ax.set_zlim(5, 15)  # 设置速度范围

    # 保存图片
    plt.savefig(save_path)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Directory containing CSV files
    directory = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/envs/a_single_lane/motion/2024-10-22-15-12-41'  # Update this path

    # Load all vehicle data
    vehicle_data = load_vehicle_data(directory)

    # Compute overall average speed and acceleration
    overall_average_speed, overall_average_acceleration, overall_average_delta_acc = compute_overall_average_speed(vehicle_data)

    # Save the result to a CSV file
    mean_speeds_file = os.path.join(directory, 'analysis', 'mean_speed.csv')
    os.makedirs(os.path.dirname(mean_speeds_file), exist_ok=True)

    with open(mean_speeds_file, 'w') as f:
        f.write(f'Overall_Average_Speed,{overall_average_speed}\n')
        f.write(f'Overall_Average_Acceleration,{overall_average_acceleration}\n')
        f.write(f'Overall_Average_Delta_Acceleration,{overall_average_delta_acc}\n')

    print(f"Average speed is {overall_average_speed},\n"
          f"average acceleration is {overall_average_acceleration},\n"
          f"average delta acceleration is {overall_average_delta_acc}.\n")

    # Sort vehicle IDs by position_x and plot the 3D surface of vehicle speeds
    sorted_vehicle_ids = sort_vehicle_ids_by_position(vehicle_data)
    save_path = directory + '/analysis/speed_distribution.png'
    plot_vehicle_trajectories_surface(vehicle_data, sorted_vehicle_ids, save_path)
