import os
from itertools import count

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from Agent import Agent
import torch

from ReplayMemory import ReplayMemory
from utils import load_agent, save_agent, plot_durations


def train(will_load_agent: bool):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_episodes = 2000
    else:
        device = torch.device("cpu")
        num_episodes = 30

    agent_path = 'saved agents/agent.pkl'
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder='video', episode_trigger=lambda x: x % (num_episodes // 4) == 0)

    if os.path.isfile(agent_path) and will_load_agent:
        print("Loading Saved Agent.", end=' ')
        agent = load_agent(agent_path)
        print("Saved Agent Loaded.\n")
    else:
        print("Initiating New Agent.", end=' ')
        agent = Agent(env, ReplayMemory(100000), device)
        print("New Agent Initiated.\n")

    episode_durations = []
    for episode in tqdm(range(num_episodes)):
        observation, info = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_observation = None
                reward = torch.tensor([100], device=device)
            else:
                next_observation = torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(observation, action, next_observation, reward)

            observation = next_observation

            agent.optimize_model()
            agent.soft_update()

            if done:
                episode_durations.append(t + 1)

                if episode % (num_episodes // 4) == 0:
                    print("Episode ", episode, ": ", np.average(episode_durations[-(num_episodes // 4):]))

                break

    env.close()
    plot_durations(episode_durations, show_result=True)
    # save_agent(agent, agent_path)
    print('Complete')


if __name__ == "__main__":
    train(False)
