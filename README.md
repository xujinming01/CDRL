# CDRL
This repository includes several reinforcement learning algorithms
for parameterised action space MDPs:

1. PA-DDPG [[Hausknecht & Stone 2016]](https://arxiv.org/abs/1511.04143)
<!-- 2. PA-SAC -->

All implementations of algorithms are based on
Stable-baselines3 (https://github.com/DLR-RM/stable-baselines3). The `plot.py`
borrows some code from Spinning Up (https://github.com/openai/spinningup)
## Domains
Experiment scripts are provided to run each algorithm
on the following domains with parameterised actions:

- Platform (https://github.com/cycraig/gym-platform)
- Robot Soccer Goal (https://github.com/cycraig/gym-goal)
- Half Field Offense (https://github.com/cycraig/gym-soccer)
 
