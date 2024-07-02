# 2024 Spring Robotics Project
The code is for our team project of 2024 Spring-Robtics.
We designed a virtual traffic scenario and proposed metrics to describe the efficiency of transportation. Based on this, we utilized reinforcement learning and trained a centralized policy network using the DDPG algorithm to regulate the acceleration of all vehicles in the scenario, thereby controlling their speed. Compared to the traditional method of traffic control through traffic lights, our proposed method effectively alleviated congestion at intersections and improved the efficiency of passage.
## 1. Installation
1. create a virtual environment and activate it

```bash
conda create -n robotics python=3.10
conda activate robotics
```

2. install the required packages

```bash
pip install -r requirements.txt
```

## 2. Usage

1. Begin training

```bash
cd DDPG
python RL_train.py
```
2. Simulation
   We provide our weights in the parent folder, you can use different weight files by modifying the path in the `RL_sim.py`
```
python RL_sim.py
```
