import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stable_baselines3.common.results_plotter import load_results, ts2xy


def main():
    log_dir = "log/"
    results = load_results(log_dir)  # load all log files to one DataFrame

    # Count the number of log files.
    n_runs = 0
    log_suffix = "monitor.csv"
    for file in os.listdir(log_dir):
        if file[-len(log_suffix):] == log_suffix:
            n_runs += 1

    # smooth the values
    len_episodes = len(results) / n_runs
    smoothed_values = np.array([])
    smooth_window = 1000
    for run in range(n_runs):
        loc_start = run * len_episodes
        loc_end = loc_start + len_episodes - 1
        values = results.loc[loc_start:loc_end, "r"].values
        smoothed_values = np.append(smoothed_values,
                                    moving_average(values, smooth_window))

    # set x axis
    episodes = list(np.arange(smooth_window - 1, len_episodes)) * n_runs

    ax = sns.lineplot(x=episodes, y=smoothed_values, ci="sd")
    ax.set_xlim(0, 80000)
    # ax.set_ylim(0, 1)
    plt.show()


def moving_average(values, window):
    """Smooth values by doing a moving average.

    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """

    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """plot the results.

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    x, y = ts2xy(load_results(log_folder), 'episodes')
    y = moving_average(y, window=5)
    x = x[len(x) - len(y):]  # Truncate x

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(f"1e8.png")
    plt.show()


if __name__ == '__main__':
    main()
