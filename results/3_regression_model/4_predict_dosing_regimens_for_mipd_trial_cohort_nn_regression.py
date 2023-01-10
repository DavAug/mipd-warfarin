import os

import numpy as np
import pandas as pd
import torch

from model import MaintenanceDoseNetwork


def load_data(target, device):
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + '/data/mipd_trial_cohort.csv')

    # Reshape data
    ids, data = reshape_data(data, target)

    data = torch.tensor(data, dtype=torch.float32, device=device)

    return ids, data


def reshape_data(df, target):
    """
    Reshapes data into [INR, covariates].
    """
    # Reshape data
    ids = df.ID.dropna().unique()
    data = np.zeros(shape=(len(ids), 5))
    for idx, _id in enumerate(ids):
        # Get info from dataframe
        temp = df[df.ID == _id]
        vkorc1 = temp.VKORC1.values[0]
        cyp = temp.CYP2C9.values[0]
        age = temp.Age.values[0]

        # Fill container
        data[idx, 0] = target

        # We implement genetic factors by counting allele variants
        if vkorc1 == 0:
            data[idx, 1] = 1
        elif vkorc1 == 1:
            data[idx, 1] = 0.5
        if cyp == 0:
            data[idx, 2] = 1
        elif cyp == 1:
            data[idx, 2] = 0.5
            data[idx, 3] = 0.5
        elif cyp == 2:
            data[idx, 2] = 0.5
        elif cyp == 3:
            data[idx, 3] = 1
        elif cyp == 4:
            data[idx, 3] = 0.5

        # Add age
        data[idx, 4] = age

    return ids, data

def load_model():
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/model/deep_regression_best.pickle'
    model = MaintenanceDoseNetwork(1024)
    model.load_state_dict(torch.load(directory + filename))
    model.eval()

    # Set scale of state
    target = 2.5
    model.set_inr_scale(mean=target, std=3*target)
    model.set_age_scale(mean=51, std=3 * 15)

    return model

def predict_maintenance_dose(data, model):
    doses = model.predict_dose(data)[:, 0]

    return np.array(doses)

def save_results(ids, doses):
    df = pd.DataFrame({
        'ID': ids,
        'Maintenance dose': doses
    })

    # Save dataframe to csv
    directory = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(
        directory +
        '/mipd_trial_predicted_dosing_regimens_deep_regression.csv',
        index=False)


if __name__ == '__main__':
    target = 2.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, data = load_data(target, device)
    model = load_model()
    model.to(device)
    doses = predict_maintenance_dose(data, model)
    save_results(ids, doses)
