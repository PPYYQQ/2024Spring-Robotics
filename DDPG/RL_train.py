import gym
from gym import spaces
import numpy as np
from utils_RL import return_para, is_add_car, DDPG, Car, CarEnv
# import wandb

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Robotics"
# )

env = CarEnv()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]

ddpg_agent = DDPG(state_dim, action_dim, min_action, max_action, device)

print("Training Begin!")
# 训练
num_episodes = 100
max_reward = -1000000000
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for t in range(env.total_frames):
        action = ddpg_agent.select_action(state)
        # print(action)
        # print(state)
        # input()
        next_state, reward, done, _ = env.step(action)
        ddpg_agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        ddpg_agent.train()
        if done:
            break
    if episode_reward > max_reward:
        max_reward = episode_reward
        torch.save(ddpg_agent.actor.state_dict(), f"Actor_best_{episode_reward}.pth")
        torch.save(ddpg_agent.critic.state_dict(), f"Critic_best_{episode_reward}.pth")

    print(f"Episode {episode+1}, Reward: {episode_reward}")

print("Training completed")