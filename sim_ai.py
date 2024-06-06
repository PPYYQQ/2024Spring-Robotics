import pygame
import sys
import time

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


from utils_ai import is_add_car, Car, TrafficLight, return_para

road_y, road_x, road_width, ns_interval, ew_interval, yellow_interval, FPS = return_para()

# 创建红绿灯和车辆
traffic_light_ew = TrafficLight(0, road_y + road_width + 10, 'horizontal', ew_interval, GREEN)  # 东西向红绿灯
traffic_light_ns = TrafficLight(road_x + road_width + 10, 0, 'vertical', ns_interval, RED)    # 南北向红绿灯
# cars = [Car(0, road_y, 'horizontal'), Car(100, road_y, 'horizontal'), Car(road_x, 0, 'vertical'), Car(road_x, 100, 'vertical')]
cars = []

# 主循环
frame = 0
frame_interval = 0
delay = 0
count = 0
flow = 0

car_info = {'horizontal':{'p':0,'np':0},
            'vertical':{'p':0,'np':0},}

running = True
while running:
    time.sleep(frame_interval)
    frame += 1
    if is_add_car(frame, 60, 1):
        t_car = Car(0, road_y +(road_width-30)/2, 'horizontal', 5)
        car_info['horizontal']['np'] += 1
        cars.append(t_car)
    if is_add_car(frame + 15, 30, 1):
        t_car = Car(road_x +(road_width-30)/2, 0, 'vertical', 5)
        car_info['vertical']['np'] += 1
        cars.append(t_car)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # 绘制道路
    pygame.draw.rect(screen, GRAY, (0, road_y, 1000, road_width))
    pygame.draw.rect(screen, GRAY, (road_x, 0, road_width, 1000))

    # 更新红绿灯和车辆
    traffic_light_ew.update(traffic_light_ns)
    traffic_light_ns.update(traffic_light_ew)

    for car in cars:
        car.update(traffic_light_ew, traffic_light_ns,cars=cars)
        if car.x > 1000 or car.y > 1000:
            car_info[car.direction]['np'] -= 1
            car_info[car.direction]['p']  += 1
            count += 1
            cars.remove(car)
    delay += len(cars)
    if flow < count / frame:
        flow = count / frame
    
    if frame % 50 ==0:
        print(f"""
    Frame  : {frame}
    Delay  : {delay/frame}
    FLow   : {count/frame}
  Max Flow : {flow}
      H    : {car_info['horizontal']['p']} / {car_info['horizontal']['np'] + car_info['horizontal']['p']}
      V    : {car_info['vertical']['p']} / {car_info['vertical']['np'] + car_info['vertical']['p']}
""")
            

    # 绘制红绿灯和车辆
    # traffic_light_ew.draw(screen)
    # traffic_light_ns.draw(screen)
    for car in cars:
        car.draw(screen)

    pygame.display.flip()
    pygame.time.Clock().tick(FPS)

pygame.quit()
sys.exit()
