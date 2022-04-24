import os

import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

import gym_platform
import gym_goal
import gym_soccer
from config import ALGORITHMS, ENVIRONMENTS


def main(algo: str = 'ddpg',
         env: str = 'goal'):
    log_dir = f"log/{algo}_stone/{env}/"  # Make sure of using right path
    algo = ALGORITHMS[algo]
    eval_env = gym.make(ENVIRONMENTS[env])
    evaluate(log_dir, algo, eval_env)


def evaluate(log_dir, algo, eval_env):
    mean_reward_each = []
    std_each = []
    best_reward = [0, 0]
    n_eval_episodes = 1000
    for root, dirs, files in os.walk(log_dir, topdown=False):
        for file in files:
            if file.endswith("best_model.zip"):
                model = algo.load(f"{root}/{file[:-4]}")
                mean, std = evaluate_policy(model, eval_env,
                                            n_eval_episodes=n_eval_episodes,
                                            deterministic=True)
                # print(f"mean_reward = {mean:.3f} +/- {std:.3f}")
                mean_reward_each.append(mean)
                std_each.append(std)
                if mean > best_reward[0] or (
                        mean == best_reward[0] and std > best_reward[1]):
                    best_reward[0] = mean
                    best_reward[1] = std
                    print(f"the BEST mean_reward = {mean:.3f} +/- {std:.3f}")
                    print(f"the model path is: {root}/{file}")
    print(f"The average reward of {len(std_each)} models with evaluation for "
          f"{n_eval_episodes} episodes is {np.mean(mean_reward_each):.3f} "
          f"+/- {np.mean(std_each):.3f}")


if __name__ == '__main__':
    main()
