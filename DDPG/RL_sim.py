import pygame
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

from utils_RL import DDPG, CarEnv

# 初始化pygame
pygame.init()

# 设置屏幕大小
screen = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption("十字路口模拟")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (169, 169, 169)


from utils_RL import is_add_car, Car, return_para

road_y, road_x, road_width, FPS, total_frames = return_para()

cars = []

# 主循环
frame = 0
frame_interval = 0
delay = 0
flow = 0

add_car_fail = 0

env = CarEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddpg_agent = DDPG(state_dim, action_dim, min_action, max_action, device)
ddpg_agent.actor.load_state_dict(torch.load("Actor_best_-64000.pth", map_location=torch.device('cpu')))
ddpg_agent.critic.load_state_dict(torch.load("Critic_best_-64000.pth", map_location=torch.device('cpu')))
ddpg_agent.actor_target.load_state_dict(ddpg_agent.actor.state_dict())
ddpg_agent.critic_target.load_state_dict(ddpg_agent.critic.state_dict())
print("Models loaded")

state = env.reset()

pygame.draw.rect(screen, GRAY, (0, road_y, 1000, road_width))
pygame.draw.rect(screen, GRAY, (road_x, 0, road_width, 1000))

for frame in range(1, env.total_frames+1):
    action = ddpg_agent.select_action(state)

    # print(state)
    # print(action)
    # input()

    next_state, reward, done, _ = env.step(action)
    ddpg_agent.replay_buffer.push(state, action, reward, next_state, done)
    # print(state)
    # print(action)
    # input()

    state = next_state
    

    # accelerations = action
    cars = env.cars
    
    delay += len(cars)
    if flow < env.count / frame:
        flow = env.count / frame

    if frame % 200 == 0:
        print(f"""
    Frame  : {frame}
    Delay  : {delay/frame}
    FLow   : {env.count/frame}
  Max Flow : {flow}
  Add Fail : {env.add_car_fail}
 Collision : {env.col_count}
   Reward  : {reward}
""")

    screen.fill(WHITE)
    # 绘制道路
    pygame.draw.rect(screen, GRAY, (0, road_y, 1000, road_width))
    pygame.draw.rect(screen, GRAY, (road_x, 0, road_width, 1000))
    # 车辆
    for car in cars:
        car.draw(screen)

    pygame.display.flip()
    pygame.time.Clock().tick(FPS)


pygame.quit()
sys.exit()
