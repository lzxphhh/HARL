import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.ticker import FuncFormatter

def custom_formatter(x, pos):
    return f'{int(x)}'
def extract_scalar_data(event_path, scalar_name='Value'):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    w_times, step_nums, values = zip(*[(e.wall_time, e.step, e.value) for e in ea.Scalars(scalar_name)])
    return pd.DataFrame({'Step': step_nums, 'Value': values})

def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_training_curve_with_variance(event_paths, scalar_name='Value', label_name='algo_name', line_color='blue', window_size=10, p_alpha=0.2):
    all_data = [extract_scalar_data(event_path, scalar_name) for event_path in event_paths]

    combined_data = pd.concat(all_data)
    grouped = combined_data.groupby('Step')
    mean_values = grouped['Value'].mean()
    max_values = grouped['Value'].max()
    min_values = grouped['Value'].min()
    variance_values = grouped['Value'].var()
    std_values = grouped['Value'].std()

    smoothed_mean = smooth_data(mean_values, window_size)
    smoothed_variance = smooth_data(variance_values, window_size)
    smoothed_max = smooth_data(max_values, window_size)
    smoothed_min = smooth_data(min_values, window_size)
    smoothed_std = smooth_data(std_values, window_size)

    steps = mean_values.index

    plt.plot(steps, smoothed_mean, label=label_name, color=line_color, linewidth=3)
    # plt.fill_between(steps, smoothed_mean - smoothed_variance, smoothed_mean + smoothed_variance, color=line_color, alpha=p_alpha)
    # plt.fill_between(steps, smoothed_min, smoothed_max, color=line_color, alpha=p_alpha)
    # plt.fill_between(steps, smoothed_mean - 0.65*std_values, smoothed_mean + 0.65*std_values, color=line_color, alpha=p_alpha)
    plt.fill_between(steps, smoothed_mean-0.65*smoothed_std, smoothed_mean+0.65*smoothed_std, color=line_color, alpha=p_alpha)

# event_paths = ['events.out_1_path',
#                'events.out_2_path',
#                'events.out_3_path']  # Adjust with actual paths
# scalar_name='average_collision_rate' or 'average_episode_length' or 'train_episode_rewards'
# y_label='Mean Collision Rate' or 'Mean Episode Length' or 'Mean Episode Rewards'
# label_name='HIAHR_DIU_MAPPO'  or 'HIAHR_MAPPO' or 'MAPPO' # Adjust with actual algo names
# plot_training_curve_with_variance(event_paths, scalar_name='average_collision_rate', label_name='HIAHR_DIU_MAPPO', y_label='Mean Collision Rate', line_color='blue', window_size=10, p_alpha=0.2)

# Usage
# collision rate
plt.figure(figsize=(21, 10)) # long figure
# plt.figure(figsize=(10, 10)) # square figure
event_paths = ['/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-25-17-14-45/logs/events.out.tfevents.1721898885.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-27-09-03-11/logs/events.out.tfevents.1722042191.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-28-09-09-59/logs/events.out.tfevents.1722128999.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-26-12-20-03/logs/events.out.tfevents.1721967603.liuzhengxuan-RESCUER-Y720-15IKB',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-28-17-23-31/logs/events.out.tfevents.1722158611.liuzhengxuan-RESCUER-Y720-15IKB']  # Adjust with actual paths
plot_training_curve_with_variance(event_paths, scalar_name='average_collision_rate', label_name='w/o PSAF', line_color='blue', window_size=5, p_alpha=0.2)

event_paths = ['/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_1/logs/events.out.tfevents.1720678303.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_2/logs/events.out.tfevents.1721393148.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_3/logs/events.out.tfevents.1721455097.BugMakers']  # Adjust with actual paths
plot_training_curve_with_variance(event_paths, scalar_name='average_collision_rate', label_name='w/ PSAF', line_color='red', window_size=5, p_alpha=0.2)

y_label='Mean Collision Rate'
plt.xlabel('No. of training steps', fontsize=36, color='black')
plt.ylabel(y_label, fontsize=36, color='black')
# plt.title(f'Training Curve with Mean and Variance for {scalar_name.capitalize()}')
plt.legend(fontsize=28)
plt.grid(True)
# Set custom x-axis ticks and labels
plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
plt.xticks([0, 1000000, 2000000, 3000000], fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(0, 3000000)
plt.savefig('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/plot_figure/figure_output/safety_CR_long.png')
# plt.savefig('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/plot_figure/figure_output/safety_CR_square.png')
plt.show()

# episode length
plt.figure(figsize=(21, 10)) # long figure
# plt.figure(figsize=(10, 10)) # square figure
event_paths = ['/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-25-17-14-45/logs/events.out.tfevents.1721898885.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-27-09-03-11/logs/events.out.tfevents.1722042191.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-28-09-09-59/logs/events.out.tfevents.1722128999.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-26-12-20-03/logs/events.out.tfevents.1721967603.liuzhengxuan-RESCUER-Y720-15IKB',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-28-17-23-31/logs/events.out.tfevents.1722158611.liuzhengxuan-RESCUER-Y720-15IKB']  # Adjust with actual paths
plot_training_curve_with_variance(event_paths, scalar_name='average_episode_length', label_name='w/o PSAF', line_color='blue', window_size=5, p_alpha=0.2)

event_paths = ['/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_1/logs/events.out.tfevents.1720678303.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_2/logs/events.out.tfevents.1721393148.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_3/logs/events.out.tfevents.1721455097.BugMakers']  # Adjust with actual paths
plot_training_curve_with_variance(event_paths, scalar_name='average_episode_length', label_name='w/ PSAF', line_color='red', window_size=5, p_alpha=0.2)

y_label='Mean Episode Length'
plt.xlabel('No. of training steps', fontsize=36, color='black')
plt.ylabel(y_label, fontsize=36, color='black')
# plt.title(f'Training Curve with Mean and Variance for {scalar_name.capitalize()}')
plt.legend(fontsize=28)
plt.grid(True)
# Set custom x-axis ticks and labels
plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
plt.xticks([0, 1000000, 2000000, 3000000], fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(0, 3000000)
plt.savefig('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/plot_figure/figure_output/safety_EL_long.png')
# plt.savefig('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/plot_figure/figure_output/safety_EL_square.png')
plt.show()

# episode rewards
plt.figure(figsize=(21, 10)) # long figure
# plt.figure(figsize=(10, 10)) # square figure
event_paths = ['/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-25-17-14-45/logs/events.out.tfevents.1721898885.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-27-09-03-11/logs/events.out.tfevents.1722042191.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-28-09-09-59/logs/events.out.tfevents.1722128999.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-26-12-20-03/logs/events.out.tfevents.1721967603.liuzhengxuan-RESCUER-Y720-15IKB',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/bottleneck/mappo/0725_HIAHR_DIU_mappo_no_lwcontroller/seed-00042-2024-07-28-17-23-31/logs/events.out.tfevents.1722158611.liuzhengxuan-RESCUER-Y720-15IKB']  # Adjust with actual paths
plot_training_curve_with_variance(event_paths, scalar_name='train_episode_rewards', label_name='w/o PSAF', line_color='blue', window_size=5, p_alpha=0.2)

event_paths = ['/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_1/logs/events.out.tfevents.1720678303.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_2/logs/events.out.tfevents.1721393148.BugMakers',
               '/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/improve_HIAHR_DIU/mappo/mappo_3/logs/events.out.tfevents.1721455097.BugMakers']  # Adjust with actual paths
plot_training_curve_with_variance(event_paths, scalar_name='train_episode_rewards', label_name='w/ PSAF', line_color='red', window_size=5, p_alpha=0.2)

y_label='Mean Episode Rewards'
plt.xlabel('No. of training steps', fontsize=36, color='black')
plt.ylabel(y_label, fontsize=36, color='black')
# plt.title(f'Training Curve with Mean and Variance for {scalar_name.capitalize()}')
plt.legend(fontsize=28)
plt.grid(True)
# Set custom x-axis ticks and labels
plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
plt.xticks([0, 1000000, 2000000, 3000000], fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(0, 3000000)
plt.savefig('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/plot_figure/figure_output/safety_ER_long.png')
# plt.savefig('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/examples/results/bottleneck/plot_figure/figure_output/safety_ER_square.png')
plt.show()