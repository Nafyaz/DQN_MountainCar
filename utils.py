import pickle
import matplotlib.pyplot as plt
import torch


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    durations_t_mean = [durations_t[max(d-20, 0):d].mean() for d in range(durations_t.size(dim=0))]
    plt.plot(durations_t_mean)

    if show_result:
        plt.show()


def save_agent(agent, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as file:
        pickle.dump(agent, file)


def load_agent(file_name):
    with open(file_name, "rb") as file:
        agent = pickle.load(file)
    return agent
