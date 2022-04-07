"""Testing for directly outputting discrete actions."""

import os
import timeit

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.logger import make_output_format
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# import gym_soccer
# import gym_goal
import gym_platform


def main():
    log_directory = "log/"
    os.makedirs(log_directory, exist_ok=True)

    n_runs = 5
    for i in range(n_runs):
        log_dir = log_directory + f"{i + 2}"
        learn(log_dir)


def learn(log_dir):
    # Save log to *.monitor.csv
    env = Monitor(gym.make('Platform-v0'), log_dir)
    env = DummyVecEnv([lambda: env])  # SAC only support VecEnv

    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions),
    #                                  sigma=float(0.02) * np.ones(n_actions))

    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        # gamma=0.99,
        # learning_rate=0.001,
        # buffer_size=1_000_000,
        # batch_size=256,
        # verbose=1,
        # action_noise=action_noise,
        # tensorboard_log="tensorboard_outputs/"
    )
    # set up logger
    output_format = [make_output_format("csv", log_dir)]
    logger = Logger(folder=log_dir, output_formats=output_format)
    model.set_logger(logger)

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=80000,
                                                      verbose=1)
    # Separate evaluation env
    env_eval = gym.make('Platform-v0')
    callback_eval = EvalCallback(env_eval,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 n_eval_episodes=100,
                                 eval_freq=20000)
    callback = CallbackList([callback_max_episodes, callback_eval])

    start = timeit.default_timer()  # Time the training
    model.learn(
        total_timesteps=1_000_000,
        callback=callback
    )
    stop = timeit.default_timer()

    print(f"\nTraining time in seconds: {int(stop - start)}")


if __name__ == '__main__':
    main()
