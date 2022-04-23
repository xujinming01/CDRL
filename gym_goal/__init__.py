from gym.envs.registration import register

register(
    id='Goal-v0',
    entry_point='gym_goal.envs:GoalEnv',
    # max_episode_steps=100,  # no need this, has a limit on self.max_time
)
