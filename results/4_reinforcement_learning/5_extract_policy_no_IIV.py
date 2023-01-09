import os
import subprocess

import numpy as np
import pandas as pd
import torch

from model import DQN


def load_model():
    """
    Loads and returns the pretrained DQN model.
    """
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/models/dqn_model_no_IIV.pickle'

    model = DQN(1024)
    model.load_state_dict(torch.load(directory + filename))
    model.eval()

    # Set scale of state
    target = 2.5
    model.set_inr_scale(mean=target, std=3*target)
    model.set_age_scale(mean=71, std=1)

    return model


def get_policy(model, n, device):
    """
    Returns the doses predicted by the DQN model based on the current states.
    """
    # Create states
    states = np.empty((n, 5))
    states[:, 0] = np.linspace(0.5, 5, n)
    states[:, 1] = 1  # VKORC1 GG
    states[:, 2] = 1  # CYP *1*1
    states[:, 3] = 0
    states[:, 4] = 71

    # Predict dose
    states = torch.tensor(
        data=states, dtype=torch.float32, device=device)
    doses = model.predict_dose(states)
    doses = np.array(doses)
    states = np.array(states)

    return states, doses


def save_policy(states, doses, filename):
    """
    Saves policy to a csv file.
    """
    data = pd.DataFrame({
        'INR': list(states[:, 0]),
        'VKORC1 G alleles': list(states[:, 1]),
        'CYP2C9 1 alleles': list(states[:, 2]),
        'CYP2C9 2 alleles': list(states[:, 3]),
        'Age': list(states[:, 4]),
        'Dose': list(doses),
    })

    # Save file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data.to_csv(directory + filename, index=False)


if __name__ == '__main__':
    n = 1000
    filename = \
        '/4_reinforcement_learning' \
        + '/policy_no_IIV.csv'

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    states, doses = get_policy(model, n, device)
    save_policy(states, doses, filename)
