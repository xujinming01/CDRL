import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results


def main():
    sns.set()

    log_dir = "log/"
    results = load_results(log_dir)  # load all log files to one DataFrame

    # Count the number of all *.monitor.csv files.
    n_runs = 0
    log_suffix = "monitor.csv"
    for file in os.listdir(log_dir):
        if file[-len(log_suffix):] == log_suffix:
            n_runs += 1

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
    plt.title("Platform")
    # ax.set_xlim(0, 80000)
    # ax.set_ylim(0, 1)
    # plt.savefig(f"1e8.png")
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


if __name__ == '__main__':
    main()
