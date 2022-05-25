import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_baselines3.common.monitor import get_monitor_files
from stable_baselines3.common.monitor import LoadMonitorResultsError


def plotter(env: str = "platform", window=2000):
    log = f"log/{env}"
    # logs_dir = f"log/{env}/{algo}_stone"
    x_axis = "Episodes"
    df = ts2xy(log, x=x_axis, window=window)

    # plot the curve
    plt.style.use("seaborn-whitegrid")  # useful in presentation
    # plt.style.use("seaborn-paper")  # useful in paper
    ax = sns.lineplot(data=df, x="x", y="reward", hue="algo", ci="sd")
    plt.xlabel(x_axis)
    plt.ylabel("Rewards")
    plt.legend()  # no need to display legend title
    # plt.title("Platform")
    # ax.set_xlim(0, 80000)
    # ax.set_ylim(0, 1)

    plt.tight_layout(pad=0.3)
    # Don't put '/' after LOG_DIRECTORY to save in correct path.
    plt.savefig(f"{log}_{x_axis}.pdf")  # vector graph
    plt.savefig(f"{log}_{x_axis}.png", dpi=1000)  # bitmap
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
    for root, dirs, files in os.walk(path):  # load data and count files
        monitor_file = get_monitor_files(root)
        if monitor_file:
            monitor_files.append(monitor_file[0])
            n_runs += 1

    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(
            f"No monitor files of the form *monitor.csv found in {path}"
        )
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


def ts2xy(log, x="Episodes", window=5000):
    """Decompose the log data to x ans ys.

    :param log: log directory
    :param x: x_axis label, can be 'Timesteps', 'Episodes', 'Walltime_hrs'.
    :param window: smooth window. MP-DQN smooth the curve with 5000 episodes.
    """

    logs_dir = [d for d in os.listdir(log)
                if os.path.isdir(os.path.join(log, d))]

    y_smoothed = np.array([])
    timesteps = np.array([])
    episodes = np.array([])
    walltime = np.array([])
    algo_labels = np.array([])

    for log_dir in logs_dir:
        # load all log files to one DataFrame
        results, n_runs = load_results(os.path.join(log, log_dir))

        len_episodes = len(results) // n_runs

        for run in range(n_runs):
            loc_start = run * len_episodes
            loc_end = loc_start + len_episodes - 1
            values = results.loc[loc_start:loc_end, "r"].values

            y_smoothed = np.append(y_smoothed, smooth(values, window))

            timesteps = np.append(
                timesteps,
                np.cumsum(results.loc[loc_start:loc_end, "l"].values))
            episodes = np.append(
                episodes,
                np.arange(len_episodes))
            walltime = np.append(
                walltime,
                results.loc[loc_start:loc_end, "t"].values / 3600.0)

            algo_labels = np.append(algo_labels, [f"{log_dir}"] * len_episodes)

    if x == "Timesteps":
        x_axis = timesteps
    elif x == "Episodes":
        x_axis = episodes
    elif x == "Walltime_hrs":
        x_axis = walltime
    else:
        raise NotImplementedError("Please give the correct x_axis label.")

    df = pd.DataFrame({"x": x_axis,
                       "reward": y_smoothed,
                       "algo": algo_labels})
    return df


if __name__ == '__main__':
    plotter()
