"""Directly outputting discrete actions.
Output weight of each discrete action, then choose action with max weight.
Based on https://arxiv.org/abs/1511.04143
Hausknecht M, Stone P. Deep reinforcement learning in parameterized action space[J]. 2015."""

import os
import time

from gym.spaces import Dict
from stable_baselines3 import DDPG, PPO, SAC, TD3
# from stable_baselines3.common.callbacks import CallbackList
# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.logger import make_output_format

from utils import kill_soccer_server


class Stone(object):
    def __init__(self, env, n_timesteps, max_episodes):
        self.n_timesteps = n_timesteps
        self.max_episodes = max_episodes
        self.model = None

        # if no Dict obs, no need for `isinstance`.
        # if isinstance(self.env.observation_space, Dict):
        #     self.policy = 'MultiInputPolicy'
        # else:
        #     self.policy = 'MlpPolicy'
        self.policy = 'MlpPolicy'

    def learn(self, log_dir):
        """Set parameters for a single run."""

        model = self.model
        # set up logger(optional)
        output_format = [make_output_format("csv", log_dir)]
        logger = Logger(folder=log_dir, output_formats=output_format)
        model.set_logger(logger)

        callback_max_episodes = StopTrainingOnMaxEpisodes(self.max_episodes,
                                                          verbose=1)

        start = time.time()  # Time the training
        model.learn(total_timesteps=self.n_timesteps,
                    callback=callback_max_episodes)
        model.save(os.path.join(log_dir, "model"))
        print(f"Training time in seconds: {int(time.time() - start)}")

    def load(self, path):
        return self.model.load(path=path)


class DDPGStone(Stone):
    def __init__(self, env=None, n_timesteps=0, max_episodes=0):
        super(DDPGStone, self).__init__(env, n_timesteps, max_episodes)
        self.model = DDPG(
            policy=self.policy,
            env=env,
            gamma=0.9,
            learning_rate=0.001,
            buffer_size=10000,
            batch_size=32,
            verbose=1,
            # action_noise=action_noise,
            # tensorboard_log="tensorboard_outputs/",
            # device="cpu",
        )


class PPOStone(Stone):
    def __init__(self, env=None, n_timesteps=0, max_episodes=0):
        super(PPOStone, self).__init__(env, n_timesteps, max_episodes)
        self.model = PPO(
            policy=self.policy,
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


class SACStone(Stone):
    def __init__(self, env=None, n_timesteps=0, max_episodes=0):
        super(SACStone, self).__init__(env, n_timesteps, max_episodes)
        self.model = SAC(
            policy=self.policy,
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
