import os

import gym
import numpy as np
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy

import gym_platform
import gym_goal
import gym_soccer
from config import ALGORITHMS, ENVIRONMENTS


def main(algo: str = 'ddpg',
         env: str = 'platform'):
    # log_dir = f"log/{algo}_stone/{env}/"  # Make sure of using right path
    log_dir = f"log/{env}/{algo}_stone"
    algo = ALGORITHMS[algo]
    eval_env = gym.make(ENVIRONMENTS[env])
    evaluate(log_dir, algo, eval_env)


def evaluate(log_dir, algo, eval_env):
    mean_reward_single = []
    std_single = []
    best_reward = [0, 0]
    n_eval_episodes = 1000
    for root, dirs, files in os.walk(log_dir, topdown=False):
        for file in files:
            if file.endswith("model.zip"):
                model = algo(eval_env)  # Must Instantiate
                model = model.load(f"{root}/{file[:-4]}")
                mean, std = evaluate_policy(model, eval_env,
                                            n_eval_episodes=n_eval_episodes,
                                            deterministic=True)
                # print(f"mean_reward = {mean:.3f} +/- {std:.3f}")
                mean_reward_single.append(round(mean, 4))
                std_single.append(round(std, 4))
                if mean > best_reward[0] or (
                        mean == best_reward[0] and std > best_reward[1]):
                    best_reward[0] = mean
                    best_reward[1] = std
                    print(f"the BEST mean_reward = {mean:.3f} +/- {std:.3f}")
                    print(f"the model path is: {root}/{file}")

    # TODO(Jinming): Rewrite redundant code below.
    print(f"The average reward of {len(std_single)} models with evaluation for "
          f"{n_eval_episodes} episodes is {np.mean(mean_reward_single):.3f} "
          f"+/- {np.std(std_single):.3f}")
    reward = pd.DataFrame({"mean": mean_reward_single,
                           f"std": std_single})
    # the last row is average reward and stand deviation
    avg_reward = pd.DataFrame({"mean": [np.mean(mean_reward_single)],
                               f"std": [np.std(std_single)]})
    df = pd.concat([reward, avg_reward])
    df.to_csv(log_dir + "_evaluation.csv", index=False)


if __name__ == '__main__':
    main()
