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


def get_cohort():
    """
    Returns the IDs and the covariates of individuals in the cohort.
    """
    # Import cohort data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = '/data/mipd_trial_cohort.csv'
    data = pd.read_csv(directory + filename)

    # Format data
    ids = data.ID.values
    covariates = data[['VKORC1', 'CYP2C9' ,'Age']].values

    # Replace covariates
    # (in 'only noise' trial all individuals have the same covariates)
    covariates[:, 0] = 0
    covariates[:, 1] = 0
    covariates[:, 2] = 71

    return ids, covariates


def generate_measurements(day, filename):
    """
    Runs the Wajima-Hartmann's QSP model to generate TDM data for the MIPD
    cohort for that day.
    """
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_obs = day + 1

    print('Generating TDM data for day: ', day)
    subprocess.Popen([
        'python',
        directory +
        '/1_systems_pharmacology_model/8_simulate_tdm_data_no_IIV.py',
        '--number',
        str(n_obs),
        '--filename',
        filename
    ]).wait()
    print('TDM data generated')


def get_states(day, ids, covs, filename):
    """
    Loads TDM data of the day from file for all individuals in MIPD cohort and
    returns the 'state' of the patients defined by the INR measurement and the
    covariates.
    """
    # Import file with dosing regimens
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df = pd.read_csv(directory + filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Invalid day. The TDM file from the '
            'previous days cannot be found. This means that either the file '
            './mipd_trial_predicted_dosing_regimens.csv '
            'has been '
            'removed or the script was not executed from day 0.')

    # Get data
    n_obs = day + 1
    df = df[df['Number of observations'] == n_obs]
    states = np.zeros(shape=(len(ids), 5))
    for idx, _id in enumerate(ids):
        temp = df[df.ID == _id]
        states[idx, 0] = temp['INR'].values[0]
        states[idx, 1:] = format_covariates(covs[idx])

    return states


def format_covariates(covariates):
    """
    Formats the covariates into the format used by the DQN model.

    Input is VKORC1, CYP2C9, Age with VKORC assuming 0, 1 or 2 and CYP2C9
    0, 1, 2, 3, 4, 5.

    Output is VKORC1 G alleles, VKORC1 A alleles, CYP *1 alleles, CYP *2
    alleles, CYP *3 alleles, Age.
    """
    covs = np.zeros(4)
    vkorc1, cyp, age = covariates
    if vkorc1 == 0:
        covs[0] = 1
    elif vkorc1 == 1:
        covs[0] = 0.5
    if cyp == 0:
        covs[1] = 1
    elif cyp == 1:
        covs[1] = 0.5
        covs[2] = 0.5
    elif cyp == 2:
        covs[1] = 0.5
    elif cyp == 3:
        covs[2] = 1
    elif cyp == 4:
        covs[2] = 0.5

    covs[3] = age

    return covs


def predict_doses(model, states, device):
    """
    Returns the doses predicted by the DQN model based on the current states.
    """
    # Predict dose
    states = torch.tensor(
        data=states, dtype=torch.float32, device=device)
    doses = model.predict_dose(states)
    doses = np.array(doses)

    return doses


def save_regimen(doses, ids, day, filename):
    """
    Saves dosing regimen to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + filename)

    # Split data into n_obs = n and rest
    n_obs = day + 1
    mask = data['Number of observations'] != n_obs
    rest = data[mask]
    data = data[~mask]

    # Add data to file
    for idx, patient_id in enumerate(ids):
        mask = data.ID == patient_id
        data.loc[mask, 'Dose %d in mg' % n_obs] = doses[idx]
        if n_obs > 1:
            # Get previous doses (just for convenience)
            mask2 = \
                (rest['Number of observations'] == (n_obs-1)) \
                & (rest.ID == patient_id)
            temp = rest[mask2]
            for day in range(1, n_obs):
                data.loc[mask, 'Dose %d in mg' % day] = \
                    temp['Dose %d in mg' % day].values[0]

    # Merge new data with rest
    data = pd.concat((rest, data), ignore_index=True)

    # Save file
    data.to_csv(directory + filename, index=False)


if __name__ == '__main__':
    days = 19
    f_meas = \
        '/4_reinforcement_learning' \
        + '/mipd_trial_predicted_dosing_regimens_no_IIV.csv'

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ids, covs = get_cohort()
    for day in range(days):
        generate_measurements(day, f_meas)
        states = get_states(day, ids, covs, f_meas)
        doses = predict_doses(model, states, device)
        save_regimen(doses, ids, day, f_meas)
