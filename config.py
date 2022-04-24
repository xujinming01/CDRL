"""This file contains some constant configurations"""

from stable_baselines3 import DDPG, PPO, SAC

ALGORITHMS = {"ddpg": DDPG,
              "ppo": PPO,
              "sac": SAC}
ENVIRONMENTS = {"goal": "Goal-v0",
                "platform": "Platform-v0"}
