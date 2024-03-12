import os
from itertools import count

import gymnasium as gym
import torch

from utils import load_agent

from Agent import Agent
from DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('MountainCar-v0', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, 'video')
observation, info = env.reset()
observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

agent_path = 'saved agents/agent.pkl'
if os.path.isfile(agent_path):
    agent = load_agent(agent_path)
else:
    exit("agent.pkl not found. Train the model first.")

for t in count():
    action = agent.select_action(observation)
    next_observation, reward, terminated, truncated, _ = env.step(action.item())

    if terminated or truncated:
        print("Episodes: ", t+1)
        break

env.close()
