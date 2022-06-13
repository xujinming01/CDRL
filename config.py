"""This file contains some constant configurations"""

from stone import DDPGStone, SACStone, PPOStone

ENVIRONMENTS = {"goal": "Goal-v0",
                "platform": "Platform-v0",
                "hfo": "SoccerScoreGoal-v0",
                }

ALGORITHMS = {"ddpg": DDPGStone,
              "ppo": PPOStone,
              "sac": SACStone,
              }
