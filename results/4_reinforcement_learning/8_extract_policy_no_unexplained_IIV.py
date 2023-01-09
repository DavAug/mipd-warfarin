import os

import numpy as np
import pandas as pd
import torch

from model import DQN


def load_model():
    """
    Loads and returns the pretrained DQN model.
    """
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/models/dqn_model_best_no_unexplained_IIV.pickle'

    model = DQN(width=1024)
    model.load_state_dict(torch.load(directory + filename))
    model.eval()

    # Set scale of state
    target = 2.5
    model.set_inr_scale(mean=target, std=3*target)
    model.set_age_scale(mean=51, std=15 * 3)

    return model


def get_policy(model, n, device):
    """
    Returns the doses predicted by the DQN model based on the current states.
    """
    s = []
    d = []
    for vkorc in [0, 1, 2]:
        g = 1
        if vkorc == 1:
            g = 0.5
        elif vkorc == 2:
            g = 0

        # Create states
        states = np.empty((n, 5))
        states[:, 0] = np.linspace(0.5, 5, n)
        states[:, 1] = g
        states[:, 2] = 1  # CYP *1*1
        states[:, 3] = 0
        states[:, 4] = 71

        # Predict dose
        states = torch.tensor(
            data=states, dtype=torch.float32, device=device)
        doses = model.predict_dose(states)
        doses = np.array(doses)
        states = np.array(states)

        s.append(states)
        d.append(doses)

    return s, d


def save_policy(states, doses, filename):
    """
    Saves policy to a csv file.
    """
    data = pd.DataFrame(columns=[
        'INR', 'VKORC1 G alleles', 'VKORC1 A alleles', 'CYP2C9 1 alleles',
        'CYP2C9 2 alleles', 'CYP2C9 3 alleles', 'Age', 'Dose'])

    for idx, s in enumerate(states):
        data = pd.concat((data, pd.DataFrame({
            'INR': list(s[:, 0]),
            'VKORC1 G alleles': list(s[:, 1]),
            'CYP2C9 1 alleles': list(s[:, 2]),
            'CYP2C9 2 alleles': list(s[:, 3]),
            'Age': list(s[:,4]),
            'Dose': list(doses[idx]),
        })))

    # Save file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data.to_csv(directory + filename, index=False)


if __name__ == '__main__':
    n = 1000
    filename = \
        '/4_reinforcement_learning' \
        + '/policy_no_unexplained_IIV.csv'

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    states, doses = get_policy(model, n, device)
    save_policy(states, doses, filename)
