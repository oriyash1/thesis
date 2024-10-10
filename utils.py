import torch
import numpy as np


def save_model(agent, filename):
    """
    Save the model of the agent to a file.

    :param agent: The agent whose model needs to be saved.
    :param filename: The name of the file where the model will be saved.
    """
    torch.save(agent.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(agent, filename):
    """
    Load the model of the agent from a file.

    :param agent: The agent whose model needs to be loaded.
    :param filename: The name of the file from where the model will be loaded.
    """
    agent.network.load_state_dict(torch.load(filename))
    agent.network.eval()
    print(f"Model loaded from {filename}")




