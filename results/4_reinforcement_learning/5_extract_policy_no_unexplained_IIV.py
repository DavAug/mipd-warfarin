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
    filename = '/models/dqn_model_no_unexplained_IIV.pickle'

    model = DQN(width=256)
    model.load_state_dict(torch.load(directory + filename))
    model.eval()

    # Set scale of state
    target = 2.5
    model.set_inr_scale(mean=target, std=target)
    model.set_age_scale(mean=51, std=15)

    return model


def get_policy(model, n, device):
    """
    Returns the doses predicted by the DQN model based on the current states.
    """
    s = []
    d = []
    covariates = [
        [1, 0.5, 0.5, 50],    # GG, *1*2, 71
        [0.5, 0, 0.5, 46],    # GA, *2*3, 46
        [0, 0.5, 0, 51],      # AA, *1*3, 51
    ]
    for cov in covariates:
        # Create states
        states = np.empty((n, 5))
        states[:, 0] = np.linspace(1, 4, n)
        states[:, 1] = cov[0]
        states[:, 2] = cov[1]
        states[:, 3] = cov[2]
        states[:, 4] = cov[3]

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
        'INR', 'VKORC1 G alleles', 'CYP2C9 1 alleles',
        'CYP2C9 2 alleles', 'Age', 'Dose'])

    for idx, s in enumerate(states):
        data = pd.concat((data, pd.DataFrame({
            'INR': list(s[:, 0]),
            'VKORC1 G alleles': list(s[:, 1]),
            'CYP2C9 1 alleles': list(s[:, 2]),
            'CYP2C9 2 alleles': list(s[:, 3]),
            'Age': list(s[:, 4]),
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
