import os

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import gym_platform


def main():
    log_dir = "log/"
    eval_env = gym.make("Platform-v0")
    evaluate(log_dir, eval_env)


def evaluate(log_dir, eval_env):
    best_reward = [0, 0]
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file == "best_model.zip":
                model = SAC.load(f"{root}/{file[:-4]}")
                mean, std = evaluate_policy(model, eval_env,
                                            n_eval_episodes=1000,
                                            deterministic=True)
                print(f"mean_reward = {mean:.3f} +/- {std:.3f}")
                if mean > best_reward[0] or (
                        mean == best_reward[0] and std > best_reward[1]):
                    best_reward[0] = mean
                    best_reward[1] = std
                    print(f"the BEST mean_reward = {mean:.3f} +/- {std:.3f}")
                    print(f"the model path is: {root}/{file}")


if __name__ == '__main__':
    main()
