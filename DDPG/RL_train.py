import gym
from gym import spaces
import numpy as np
from utils_RL import return_para, is_add_car, DDPG, Car, CarEnv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
# import wandb

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Robotics"
# )
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--frames1",type=int, default=100, help="How many frames to add a car from the probability")
parser.add_argument("--frames2",type=int, default=40, help="How many frames to add a car from the probability")
parser.add_argument("--prob1",type=float, default=1, help="The probability to add a car")
parser.add_argument("--prob2",type=float, default=1, help="The probability to add a car")
parser.add_argument("--prefix", type=str, default="net", help="The prefix of the model name")

args = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(args.seed)

env = CarEnv(args.frames1, args.frames2, args.prob1, args.prob2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]

ddpg_agent = DDPG(state_dim, action_dim, min_action, max_action, device)

print("Training Begin!")
# 训练
num_episodes = 1500
max_reward = -1000000000
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for t in range(env.total_frames):
        action = ddpg_agent.select_action(state)
        # print(action)
        # print(state)
        # input()
        action = (action + np.random.normal(0, 0.04, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
        next_state, reward, done, _ = env.step(action)
        ddpg_agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        ddpg_agent.train()
        if done:
            break
    if episode_reward > max_reward:
        print(f"The best reward is updated: {max_reward} -> {episode_reward}, Saving models...")
        max_reward = episode_reward
        torch.save(ddpg_agent.actor.state_dict(), f"Actor_best_{args.frames1}_{args.frames2}_{args.prob1}_{args.prob2}_{args.prefix}.pth")
        torch.save(ddpg_agent.critic.state_dict(), f"Critic_best_{args.frames1}_{args.frames2}_{args.prob1}_{args.prob2}_{args.prefix}.pth")
    if episode % 20 == 0:
        env.render()

    print(f"Episode {episode+1}, Reward: {episode_reward}",flush=True)

print("Training completed")