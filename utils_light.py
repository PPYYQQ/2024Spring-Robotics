# 定义颜色

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (169, 169, 169)

def return_para():
    road_y = 450
    road_x = 450
    road_width = 100

    ns_interval = 400
    ew_interval = 200
    yellow_interval = 10

    FPS = 60

    return road_y, road_x, road_width, ns_interval, ew_interval, yellow_interval, FPS


road_y, road_x, road_width, ns_interval, ew_interval, yellow_interval, _ = return_para()

import pygame
import random
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
        self.acceleration = 0.1  # 加速度
        self.brake_deceleration = 2  # 刹车减速度
        self.free_deceleration = 0.1  # 自由减速度
        self.safe_distance = 70  # 安全距离

    def check_collision(self, other):
        if self.direction == 'horizontal' and other.direction == 'horizontal':
            if abs(self.x - other.x) < 30 and abs(self.y - other.y) < 30:
                return True
        elif self.direction == 'vertical' and other.direction == 'vertical':
            if abs(self.y - other.y) < 30 and abs(self.x - other.x) < 30:
                return True
        elif self.direction != other.direction:
            if abs(self.x - other.x) < 30 and abs(self.y - other.y) < 30:
                return True
        return False

    def check_front_car(self, cars):
        for car in cars:
            if car != self:
                if self.direction == 'horizontal' and car.direction == 'horizontal':
                    if car.x > self.x and abs(car.x - self.x) < self.safe_distance:
                        return car
                elif self.direction == 'vertical' and car.direction == 'vertical':
                    if car.y > self.y and abs(car.y - self.y) < self.safe_distance:
                        return car
                # elif self.direction == 'horizontal' and car.direction == 'vertical':
                #     if abs(self.x - car.x) < 10 and abs(self.y - car.y) < 10:
                #         return car

        return None

    def update(self, traffic_light_ew, traffic_light_ns, cars):
        front_car = self.check_front_car(cars)
        if self.direction == 'horizontal':
            # 判断和前车的距离
            if front_car:
                self.speed -= self.brake_deceleration
            elif self.x < road_x - road_width or self.x > road_x + road_width:
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            elif traffic_light_ew.color == GREEN or (self.x > road_x -10):
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            else:
                self.speed -= self.brake_deceleration

            self.speed = max(self.speed, 0)
            self.x += self.speed

        elif self.direction == 'vertical':
            if front_car:
                self.speed -= self.brake_deceleration
            elif self.y < road_y - road_width or self.y > road_y + road_width:
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            elif traffic_light_ns.color == GREEN or (self.y > road_y-10):
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            else:
                self.speed -= self.brake_deceleration

            self.speed = max(self.speed, 0)
            self.y += self.speed

        # # 检测碰撞并及时刹车
        # for car in cars:
        #     if car != self and self.check_collision(car):
        #         if self.speed > 0:
        #             self.speed -= self.brake_deceleration
        #         break

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, 30, 30))

# 定义红绿灯类
class TrafficLight:
    def __init__(self, x, y, direction, interval, initial):
        self.x = x
        self.y = y
        self.direction = direction
        self.interval = interval
        self.color = initial
        self.yellow_count = yellow_interval
        if initial == GREEN:
            self.countdown = interval
        elif initial == RED:
            self.countdown = 0

    def update(self, other, *yellow_count):
        if self.countdown > 1:
            self.countdown = self.countdown - 1
        elif self.countdown == 1:
            if self.yellow_count==0:
                self.yellow_count = yellow_interval
                self.countdown = 0
                self.color = RED
                other.countdown = other.interval
                other.color = GREEN
            else:
                self.color = YELLOW
                self.yellow_count -= 1



    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, 50, 150))
        if self.color == RED:
            pygame.draw.circle(screen, RED, (self.x + 25, self.y + 25), 20)
            pygame.draw.circle(screen, BLACK, (self.x + 25, self.y + 75), 20)
            pygame.draw.circle(screen, BLACK, (self.x + 25, self.y + 125), 20)
        elif self.color == GREEN:
            pygame.draw.circle(screen, BLACK, (self.x + 25, self.y + 25), 20)
            pygame.draw.circle(screen, BLACK, (self.x + 25, self.y + 75), 20)
            pygame.draw.circle(screen, GREEN, (self.x + 25, self.y + 125), 20)
        elif self.color == YELLOW:
            pygame.draw.circle(screen, BLACK, (self.x + 25, self.y + 25), 20)
            pygame.draw.circle(screen, YELLOW, (self.x + 25, self.y + 75), 20)
            pygame.draw.circle(screen, BLACK, (self.x + 25, self.y + 125), 20)

