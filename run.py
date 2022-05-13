import os

import fire
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import gym_soccer
import gym_goal
import gym_platform
from config import ALGORITHMS
from config import ENVIRONMENTS


def run_stone(
        algo: str = "ddpg",
        env: str = "platform",
        n_runs: int = 5,
        max_timesteps: int = 2_000_000,
        max_episodes: int = 80_000,
        n_eval_episodes: int = 100,
        eval_freq: int = 20_000,
):
    # make sure the path is correct
    logs_dir = f"log/{env}/{algo}_stone"

    # kill_soccer_server()
    env_v0 = ENVIRONMENTS[env]
    algo = ALGORITHMS[algo]

    for i in range(n_runs):
        log_dir = os.path.join(logs_dir, f"{i}")
        os.makedirs(log_dir, exist_ok=True)

        env = Monitor(gym.make(env_v0), log_dir)  # Save log to *.monitor.csv
        env = DummyVecEnv([lambda: env])
        model = algo(env, max_timesteps, max_episodes)

        model.learn(log_dir)


if __name__ == '__main__':
    fire.Fire(run_stone)
    # run_stone()
