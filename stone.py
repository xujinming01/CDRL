"""Directly outputting discrete actions.
Output weight of each discrete action, then choose action with max weight.
Based on https://arxiv.org/abs/1511.04143
Hausknecht M, Stone P. Deep reinforcement learning in parameterized action space[J]. 2015."""

import os
import time

import fire
import gym
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.logger import make_output_format
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import gym_soccer
import gym_goal
import gym_platform

ALGORITHMS = {"ddpg": DDPG,
              "ppo": PPO,
              "sac": SAC}
ENVIRONMENTS = {"goal": "Goal-v0",
                "platform": "Platform-v0"}


def run(algo: str = "sac",
        env: str = "goal",
        n_runs: int = 5,
        max_timesteps: int = 2_000_000,
        max_episodes: int = 80_000,
        n_eval_episodes: int = 100,
        eval_freq: int = 20_000,
        ):
    logs_dir = f"log/{algo}_stone/{env}"  # Make sure of using right path
    algo = ALGORITHMS[algo]
    env = ENVIRONMENTS[env]

    for i in range(n_runs):
        log_dir = os.path.join(logs_dir, f"{i}")
        os.makedirs(log_dir, exist_ok=True)
        learn_single_run(log_dir, algo, env, max_timesteps, max_episodes,
                         n_eval_episodes, eval_freq)


def learn_single_run(log_dir, algo, env, max_timesteps, max_episodes,
                     n_eval_episodes, eval_freq):
    """Set parameters for a single training."""
    env_eval = gym.make(env)  # Separate evaluation env
    env = Monitor(gym.make(env), log_dir)  # Save log to *.monitor.csv
    env = DummyVecEnv([lambda: env])

    model = algo(
        policy='MultiInputPolicy',
        env=env,
        # gamma=0.99,
        # learning_rate=0.001,
        # buffer_size=1_000_000,
        # batch_size=256,
        # verbose=1,
        # action_noise=action_noise,
        # tensorboard_log="tensorboard_outputs/",
        # device="cpu",
    )

    # set up logger
    output_format = [make_output_format("csv", log_dir)]
    logger = Logger(folder=log_dir, output_formats=output_format)
    model.set_logger(logger)

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes, verbose=1)
    callback_eval = EvalCallback(env_eval,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 n_eval_episodes=n_eval_episodes,
                                 eval_freq=eval_freq)
    callback = CallbackList([callback_max_episodes, callback_eval])

    start = time.time()  # Time the training
    model.learn(
        total_timesteps=max_timesteps,
        callback=callback,
    )
    print(f"Training time in seconds: {int(time.time() - start)}")


if __name__ == '__main__':
    fire.Fire(run)
