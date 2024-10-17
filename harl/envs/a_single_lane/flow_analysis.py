import pandas as pd
import glob
import os
import re


def load_vehicle_data(directory):
    csv_files = glob.glob(f'{directory}/*.csv')
    vehicle_data = {}
    for file in csv_files:
        # 使用正则表达式提取文件名中的车辆ID
        match = re.search(r'(HDV_\d+|CAV_\d+)', os.path.basename(file))
        if match:
            vehicle_id = match.group(0)  # 提取匹配的车辆ID
            data = pd.read_csv(file, header=None, names=['time', 'position_x', 'position_y', 'speed', 'acceleration'])
            vehicle_data[vehicle_id] = data
    return vehicle_data


def compute_overall_average_speed(vehicle_data):
    total_speed = 0
    total_acceleration = 0
    total_count = 0
    for vehicle_id, data in vehicle_data.items():
        total_speed += data['speed'].sum()
        total_acceleration += data['acceleration'].abs().sum()
        total_count += data['speed'].count()

    overall_average_speed = total_speed / total_count if total_count > 0 else 0
    overall_average_acceleration = total_acceleration / total_count if total_count > 0 else 0
    return overall_average_speed, overall_average_acceleration


# Main execution
if __name__ == "__main__":
    # Directory containing CSV files
    directory = '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/envs/a_single_lane/motion/2024-10-17-15-19-26'  # Update this path

    # Load all vehicle data
    vehicle_data = load_vehicle_data(directory)

    # Compute overall average speed
    overall_average_speed, overall_average_acceleration = compute_overall_average_speed(vehicle_data)

    # Save the result to a CSV file
    mean_speeds_file = os.path.join(directory, 'analysis', 'mean_speed.csv')
    os.makedirs(os.path.dirname(mean_speeds_file), exist_ok=True)

    with open(mean_speeds_file, 'w') as f:
        f.write(f'Overall_Average_Speed,{overall_average_speed}\n')
        f.write(f'Overall_Average_Acceleration,{overall_average_acceleration}\n')

    print(f"Average speed is {overall_average_speed}, average acceleration is {overall_average_acceleration}")


