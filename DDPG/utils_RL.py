import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

#设置torch的种子



WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (169, 169, 169)
BLUE = (173, 216, 230)
SILVER = (192, 192, 192)
LRED =  (255, 182, 193)

def return_para():
    road_y = 450
    road_x = 450
    road_width = 100

    FPS = 60
    
    total_frames = 1000

    return road_y, road_x, road_width, FPS, total_frames


road_y, road_x, road_width, _, _ = return_para()


def is_add_car(current_frame, interval_frame, p):
    if current_frame % interval_frame==0:
        if random.randint(0,99)/100 < p:
            return True
    return False 

# 定义车辆类
class Car:
    def __init__(self, x, y, direction, speed=0):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed  # 初始速度为0
        self.max_speed = 5  # 最大速度

        # TODO：这里最大加速度和自由减速度相等，会不会使得车没办法加速啊

        self.max_acceleration = 0.1  # 加速度
        self.max_deceleration = 0.3  # 刹车减速度
        self.free_deceleration = 0.1  # 自由减速度
        self.collision = None
        self.avai = True

    def check_collision(self, cars):
        for other in cars:
            if other == self:
                continue
            if self.direction == 1 and other.direction == 1:
                if abs(self.x - other.x) < 30 and abs(self.y - other.y) < 30:
                    return other
            elif self.direction == -1 and other.direction == -1:
                if abs(self.y - other.y) < 30 and abs(self.x - other.x) < 30:
                    return other
            elif self.direction != other.direction:
                if abs(self.x - other.x) < 30 and abs(self.y - other.y) < 30:
                    return other
        return None

    def update(self, acceleration):

        if self.direction == 1:
            self.speed += acceleration
            self.speed = max(self.speed, 0)
            self.speed = min(self.speed, self.max_speed)
            self.x += self.speed
            return self.speed

        elif self.direction == -1:
            self.speed += acceleration
            self.speed = max(self.speed, 0)
            self.speed = min(self.speed, self.max_speed)
            self.y += self.speed
            return self.speed
        
        # TODO：这个地方是不是忘记调整车的加速度了，好像只有速度变化了，加速度没动？

        

    def draw(self, screen):
        if self.direction == -1:
            # 画车身 (一个矩形)
            car_body = pygame.Rect(self.x, self.y, 20, 30)
            pygame.draw.rect(screen, BLUE, car_body)

            # 画车轮 (四个小矩形)
            wheel_width = 5
            wheel_height = 10
            pygame.draw.rect(screen, BLACK, (self.x - 2, self.y, wheel_width, wheel_height))  # 左上轮
            pygame.draw.rect(screen, BLACK, (self.x - 2, self.y + 20, wheel_width, wheel_height))  # 左下轮
            pygame.draw.rect(screen, BLACK, (self.x + 17, self.y, wheel_width, wheel_height))  # 右上轮
            pygame.draw.rect(screen, BLACK, (self.x + 17, self.y + 20, wheel_width, wheel_height))  # 右下轮
        elif self.direction == 1:
            # 画车身 (一个矩形)
            car_body = pygame.Rect(self.x, self.y, 30, 20)
            pygame.draw.rect(screen, LRED, car_body)

            # 画车轮 (四个小矩形)
            wheel_width = 10
            wheel_height = 5
            pygame.draw.rect(screen, BLACK, (self.x, self.y - 2, wheel_width, wheel_height))  # 左上轮
            pygame.draw.rect(screen, BLACK, (self.x + 20, self.y - 2, wheel_width, wheel_height))  # 右上轮
            pygame.draw.rect(screen, BLACK, (self.x, self.y + 17, wheel_width, wheel_height))  # 左下轮
            pygame.draw.rect(screen, BLACK, (self.x + 20, self.y + 17, wheel_width, wheel_height))  # 右下轮


import gym
from gym import spaces
import numpy as np
class CarEnv(gym.Env):
    def __init__(self,frame1=100,frame2=40,prob1=1,prob2=1):
        super(CarEnv, self).__init__()
        self.road_y, self.road_x, self.road_width, self.FPS, self.total_frames = return_para()
        
        self.cars = []
        self.frame = 0
        self.add_car_fail = 0
        self.count = 0
        self.col_count = 0
        self.frame_freq1 = frame1
        self.frame_freq2 = frame2
        
        # 定义动作空间和状态空间
        self.max_cars = 16  # 最大车辆数量
        self.action_space = spaces.Box(low=-0.3, high=0.1, shape=(self.max_cars,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.max_cars * 4,), dtype=np.float32)

    def reset(self):
        self.cars = []
        self.frame = 0
        self.add_car_fail = 0
        self.count = 0
        self.col_count = 0
        return self._get_state()

    def step(self, action):
        self.frame += 1
        reward = 0
        # 根据动作调整加速度
        for i, car in enumerate(self.cars):
            car_speed = int(car.update(action[i]))
            if car_speed == 0:
                reward -= 10
            reward += car_speed* 3
            

        fail_count = 0
        # 添加新车
        if len(self.cars) < self.max_cars:
            if is_add_car(self.frame, self.frame_freq1, 1):
                t_car = Car(0, self.road_y + (self.road_width - 30) / 2, 1, 3)
                t_car.collision = t_car.check_collision(self.cars)
                if t_car.collision is None:
                    self.cars.append(t_car)
                else:
                    self.add_car_fail += 1
                    fail_count += 1

            if is_add_car(self.frame, self.frame_freq2, 1):
                t_car = Car(self.road_x + (self.road_width - 30) / 2, 0, -1, 3)
                t_car.collision = t_car.check_collision(self.cars)
                if t_car.collision is None:
                    self.cars.append(t_car)
                else:
                    self.add_car_fail += 1
                    fail_count += 1

        # 计算奖励
        # reward = -len(self.cars)  # 简单的奖励函数，车辆越少越好
        # TODO: 这里的reward
        reward -= (fail_count * 500)
        for car in self.cars:
            # reward += car.speed * 5
            car.collision = car.check_collision(self.cars)
            if car.collision != None:
                reward -= 1000
                car.avai = False
                car.collision.avai = False
                self.col_count += 1
        # 移除越界、碰撞车辆
        for car in self.cars:
            if not(car.x <= 1000 and car.y <= 1000):
                self.count += 1
                reward += 500
        self.cars = [car for car in self.cars if car.x <= 1000 and car.y <= 1000 and ((car.collision == None) or (car.collision.avai))]
        

        # 判断是否结束
        done = self.frame >= self.total_frames
        
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        # print 当前的frame，状态
        print(f"Frame: {self.frame}")
        print(f"State: {self._get_state()}")
        print(f"Car count: {len(self.cars)}")
        print(f"Collision count: {self.col_count}")
        print(f"Add car fail count: {self.add_car_fail}")
        print(f"Flow: {self.count / self.frame}")
        print("Cars:")
        for car in self.cars:
            print(f"Car: {car.x}, {car.y}, {car.speed}, {car.direction}")
        print("")

    def _get_state(self):
        state = []
        for car in self.cars:
            state.extend([car.x, car.y, car.speed, car.direction])
        # 用0填充未使用的部分
        while len(state) < self.max_cars * 4:
            state.extend([0, 0, 0, 0])
        return np.array(state, dtype=np.float32)



class GaussianNoise:
    def __init__(self, action_dim, sigma=0.1):
        self.action_dim = action_dim
        self.sigma = sigma

    def sample(self):
        return np.random.normal(0, self.sigma, size=self.action_dim)

# 定义 Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def affine(x, a, b):
    return ((b-a)/2) * x + (a+b)/2

# 定义 Actor 和 Critic 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        # self.l5 = nn.Linear(400, 100)
        # self.l6 = nn.Linear(100, 40)
        self.l7 = nn.Linear(256, action_dim)
        self.min_action = min_action
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        # x = torch.relu(self.l5(x))
        # x = torch.relu(self.l6(x))
        x = torch.tanh(self.l7(x))
        x = affine(x, self.min_action, self.max_action)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 400)
        self.l5 = nn.Linear(400, 32)
        # self.l6 = nn.Linear(100, 40)
        self.l7 = nn.Linear(32, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        # x = self.l3(x)
        # x = self.l4(x)
        # x = self.l5(x)
        x = torch.relu(self.l5(x))
        x = self.l7(x)
        return x

# 定义 DDPG 算法
class DDPG:
    def __init__(self, state_dim, action_dim, min_action, max_action, device):
        self.device = device

        self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, min_action, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.min_action = min_action
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = 1000
        self.gamma = 0.99
        self.tau = 0.002
        
        # self.update_frequency = 20
     # 使用高斯噪声或Ornstein-Uhlenbeck噪声
        # self.noise = GaussianNoise(action_dim, sigma=0.1)
        # self.noise = OrnsteinUhlenbeckNoise(action_dim, mu=0, theta=0.15, sigma=0.2)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        # noise = self.noise.sample()
        return action.clip(self.min_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Assuming state, action, reward, next_state, and done are lists of numpy.ndarrays
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward).reshape(-1, 1)
        next_state = np.array(next_state)
        done = np.array(done).reshape(-1, 1)

        # Now convert these numpy arrays to torch tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * self.gamma * target_q).detach()

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__ == '__main__':
    print('OK')
