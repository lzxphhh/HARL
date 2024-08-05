import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter

def custom_formatter(x, pos):
    return f'{int(x)}'
def load_vehicle_data(directory):
    csv_files = glob.glob(f'{directory}/*.csv')
    vehicle_data = {}
    for file in csv_files:
        vehicle_id = file.split('/')[-1].split('.')[0][:5]  # Assuming filename is the vehicle ID
        data = pd.read_csv(file, header=None, names=['time', 'position', 'speed', 'safety_level'])
        # data = pd.read_csv(file, header=None, names=['time', 'position_x', 'position_y', 'speed', 'safety_level'])
        # Set safety_level to 20 if it is greater than 20
        data['safety_level'] = data['safety_level'].apply(lambda x: 1/x if x != 0 else x)
        vehicle_data[vehicle_id] = data
    return vehicle_data

def smooth_data(data, window_size=10, poly_order=3):
    return savgol_filter(data, window_size, poly_order)

# Function to plot speed and safety level curves with smoothing
def plot_vehicle_curves(vehicle_data, vehicle_ids):
    color_index = {'CAV_0': 'blue', 'CAV_1': 'orange', 'CAV_2': 'green', 'CAV_3': 'red', 'CAV_4': 'purple',
                   'CAV_5': 'brown'}
    # color_index = {'CAV_00': 'black', 'CAV_01': 'black', 'CAV_02': 'black', 'CAV_03': 'black', 'CAV_04': 'black', 'CAV_05': 'black',
    #                'CAV_06': 'black', 'CAV_07': 'black', 'CAV_08': 'black', 'CAV_09': 'black'}
    plt.figure(figsize=(15, 5))
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        label_id = vehicle_id
        # label_id = vehicle_id[:3] + '_' + vehicle_id[5]
        smoothed_speed = smooth_data(data['speed'], window_size=10, poly_order=3)
        # plt.plot(data['time'], smoothed_speed, label=label_id, alpha=0.5, color=color_index[vehicle_id], linewidth=2)
        plt.plot(data['time'], smoothed_speed, alpha=0.5, color=color_index[vehicle_id], linewidth=1)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Speed (m/s)', fontsize=18)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], fontsize=18)
    plt.yticks([0, 5, 10, 15, 20], fontsize=18)
    plt.savefig(
        '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/figure_output/25_veh/HIAHR_DIU_veh_v.png')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    safety_file = directory + '/analysis/safety.csv'
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        label_id = vehicle_id
        # label_id = vehicle_id[:3] + '_' + vehicle_id[5]
        smoothed_safety_level = smooth_data(data['safety_level'], window_size=5, poly_order=0)

        # plt.plot(data['time'], smoothed_safety_level, label=label_id, alpha=0.5, color=color_index[vehicle_id], linewidth=2)
        plt.plot(data['time'], smoothed_safety_level, alpha=0.5, color=color_index[vehicle_id], linewidth=1)
        with open(safety_file, 'a', newline='') as f:
            f.write('%s:%s\n' % (label_id, smoothed_safety_level))
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Safety Level', fontsize=18)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], fontsize=18)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=18)
    plt.ylim(0, 0.5)
    plt.savefig(
        '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/figure_output/25_veh/HIAHR_DIU_veh_s.png')
    plt.tight_layout()
    plt.show()

def plot_vehicle_motion(vehicle_data, vehicle_ids):
    color_index = {'CAV_0': 'red', 'CAV_1': 'red', 'CAV_2': 'red', 'CAV_3': 'red', 'CAV_4': 'red', 'CAV_5': 'red',
                   'HDV_0': 'blue', 'HDV_1': 'blue', 'HDV_2': 'blue', 'HDV_3': 'blue', 'HDV_4': 'blue', 'HDV_5': 'blue',
                   'HDV_6': 'blue', 'HDV_7': 'blue', 'HDV_8': 'blue'}
    # color_index = {'HDV_00': 'blue', 'HDV_01': 'blue', 'HDV_02': 'blue', 'HDV_03': 'blue', 'HDV_04': 'blue', 'HDV_05': 'blue',
    #                'HDV_06': 'blue', 'HDV_07': 'blue', 'HDV_08': 'blue', 'HDV_09': 'blue', 'HDV_10': 'blue', 'HDV_11': 'blue',
    #                'HDV_12': 'blue', 'HDV_13': 'blue', 'HDV_14': 'blue'}
    plt.figure(figsize=(21, 7))
    for vehicle_id in vehicle_ids:
        if vehicle_id not in vehicle_data:
            print(f"No data found for {vehicle_id}")
            continue
        data = vehicle_data[vehicle_id]
        # label_id = vehicle_id[:3] + '_' + vehicle_id[5]
        plt.plot(data['position_x'], data['position_y'], alpha=0.5, color=color_index[vehicle_id], linewidth=3)
    plt.xlabel('x (m)', fontsize=24)
    plt.ylabel('y (m)', fontsize=24)
    # plt.legend(fontsize=14)
    plt.grid(True)
    # Set custom x-axis ticks and labels
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xticks([200, 300, 400, 500, 600, 700], fontsize=24)
    plt.yticks([-6, -4, -2, 0, 2, 4, 6], fontsize=24)
    plt.xlim(200, 600)
    plt.savefig(
        '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/figure_output/vehicle_motion/HIAHR_DIU_MAPPO_motion.png')
    plt.tight_layout()
    plt.show()

def compute_average_speed(vehicle_data, ranges):
    average_speeds = {f'{pos_range[0]}-{pos_range[1]}': [] for pos_range in ranges}
    for vehicle_id, data in vehicle_data.items():
        for pos_range in ranges:
            range_str = f'{pos_range[0]}-{pos_range[1]}'
            mask = (data['position'] >= pos_range[0]) & (data['position'] < pos_range[1])
            # mask = (data['position_x'] >= pos_range[0]) & (data['position_x'] < pos_range[1])
            average_speed = data[mask]['speed'].mean()
            average_speeds[range_str].append(average_speed)
    return average_speeds


# Function to plot box plot of average speeds
def plot_average_speed_boxplot(average_speeds):
    # Prepare data for plotting
    plot_data = []
    for zone, speeds in average_speeds.items():
        for speed in speeds:
            plot_data.append({'zone': zone, 'speed': speed})

    df = pd.DataFrame(plot_data)
    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='zone', y='speed', data=df)
    plt.title('Average Speed in Different Zones')
    plt.xlabel('Zone')
    plt.ylabel('Average Speed')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Directory containing CSV files
    directory = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/results_analysis/HIAHR_DIU_MAPPO/15_vehs/00_44_2024-07-23-15-37-04'  # Update this path

    # Load all vehicle data
    vehicle_data = load_vehicle_data(directory)

    # Define position ranges
    position_ranges = [(0, 400), (400, 500), (500, 700)]

    # Compute average speeds
    average_speeds = compute_average_speed(vehicle_data, position_ranges)

    # mean_speeds_file = directory + '/analysis/mean_speeds.csv'
    # with open(mean_speeds_file, 'w') as f:
    #     for key, value in average_speeds.items():
    #         f.write('%s:%s\n' % (key, value))

    # # Print average speeds for each vehicle
    # for vehicle_id, averages in average_speeds.items():
    #     print(f'Vehicle {vehicle_id}: {averages}')
    # Print average speeds for each position range
    # for range_str, averages in average_speeds.items():
    #     print(f'Position Range {range_str}: {averages}')
    # plot_average_speed_boxplot(average_speeds)

    # Plot curves for specific vehicles
    CAV_list = ['CAV_0', 'CAV_1', 'CAV_2', 'CAV_3', 'CAV_4', 'CAV_5']
    # CAV_list = ['CAV_00', 'CAV_01', 'CAV_02', 'CAV_03', 'CAV_04', 'CAV_05', 'CAV_06', 'CAV_07', 'CAV_08', 'CAV_09']  # Replace with actual vehicle IDs
    plot_vehicle_curves(vehicle_data, CAV_list)
    # veh_list = ['CAV_0', 'CAV_1', 'CAV_2', 'CAV_3', 'CAV_4', 'CAV_5',
    #             'HDV_0', 'HDV_1', 'HDV_2', 'HDV_3', 'HDV_4', 'HDV_5', 'HDV_6', 'HDV_7', 'HDV_8']
    # veh_list = ['HDV_00', 'HDV_01', 'HDV_02', 'HDV_03', 'HDV_04', 'HDV_05', 'HDV_06', 'HDV_07', 'HDV_08', 'HDV_09',
    #             'HDV_10', 'HDV_11', 'HDV_12', 'HDV_13', 'HDV_14']
    # plot_vehicle_motion(vehicle_data, veh_list)
