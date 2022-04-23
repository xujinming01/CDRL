import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_baselines3.common.monitor import get_monitor_files
from stable_baselines3.common.monitor import LoadMonitorResultsError


def plotter(logs_dir: str = f"tmp/sac_stone/goal"):  # Make sure of using right path
    sns.set()

    results, n_runs = load_results(logs_dir)  # load all log files to one DataFrame
    # smooth the values
    len_episodes = len(results) / n_runs
    smoothed_values = np.array([])
    window = 5000  # MP-DQN smooth the curve with 5000 episodes.
    for run in range(n_runs):
        loc_start = run * len_episodes
        loc_end = loc_start + len_episodes - 1
        values = results.loc[loc_start:loc_end, "r"].values
        smoothed_values = np.append(smoothed_values, smooth(values, window))

    # set x axis
    episodes = list(np.arange(len_episodes)) * n_runs

    ax = sns.lineplot(x=episodes, y=smoothed_values, ci="sd")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    # plt.title("Platform")
    # ax.set_xlim(0, 80000)
    # ax.set_ylim(0, 1)

    # Don't put '/' after LOG_DIRECTORY to save in correct path.
    plt.savefig(f"{logs_dir}.svg")  # vector graph
    plt.savefig(f"{logs_dir}.png", dpi=1000)  # bitmap
    plt.show()


def smooth(values, window=1):
    """
    Smooth the curve by doing a moving average.
        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
    """

    values = np.asarray(values)
    y = np.ones(window)
    z = np.ones(len(values))

    return np.convolve(values, y, 'same') / np.convolve(z, y, 'same')


def load_results(path: str):
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: data_frame: the logged data
    :return: n_runs: the number of monitor.csv
    """
    n_runs = 0
    monitor_files = []
    for root, dirs, files in os.walk(path):
        monitor_file = get_monitor_files(root)
        if monitor_file:
            monitor_files.append(monitor_file[0])
            n_runs += 1

    # monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *monitor.csv found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pd.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pd.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame, n_runs


if __name__ == '__main__':
    plotter()
