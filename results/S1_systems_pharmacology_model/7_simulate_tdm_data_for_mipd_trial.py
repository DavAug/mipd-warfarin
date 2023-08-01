import argparse
import os

import chi
import myokit
import numpy as np
import pandas as pd

from model import define_wajima_model


def get_regimen(patient, n, delays, filename):
    """
    Imports the patient's dosing regimen.
    """
    if n == 1:
        # This is the first TDM measurement of the patient, which is taken
        # before any drug is administered. So we return an empty dosing
        # regimen
        return myokit.Protocol()

    # Import file with dosing regimens
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df = pd.read_csv(directory + filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Invalid number of observations. File does not exist. Try '
            'executing the script with number = 1.')

    # Get dosing regimen from dataframe
    mask = (df['Number of observations'] == (n-1)) & (df.ID == patient.ID)
    doses = df[mask][['Dose %d in mg' % (d+1) for d in range(n-1)]].values[0]

    # Define dosing regimen
    regimen = define_dosing_regimen(doses, delays[:n])

    return regimen


def define_dosing_regimen(doses, delays, cal_time=100*24):
    """
    Returns a dosing regimen with delayed administration times.
    """
    duration = 0.01
    regimen = myokit.Protocol()
    for day, dose in enumerate(doses):
        if dose == 0:
            continue
        regimen.add(myokit.ProtocolEvent(
            level=dose/duration,
            start=cal_time+day*24+delays[day],
            duration=duration))

    return regimen


def generate_measurement(model, patient, n, delays, vk_input, rng):
    """
    Generates TDM data for individual.
    """
    # Get parameters of indvidual
    parameters = np.array(patient[model.parameters()].values)
    sigma_log = np.array([patient['Sigma log']])

    # Simulate measurements
    # NOTE nth measurement occurs on (n-1)th day. And we let the system
    # equilibrate for 100 days.
    cal_time = 100
    times = np.array([0 ,(n - 1) * 24 + delays[n-1]]) + cal_time * 24
    inr = model.simulate(
        parameters=parameters, times=times, vk_input=vk_input[:n])[:, -1]

    # Add noise
    error_model = chi.LogNormalErrorModel()
    measurement = error_model.sample(
        parameters=sigma_log, model_output=inr, seed=rng)[0, 0]

    return measurement


def get_vk_consumption(days, nids, seed):
    """
    Returns deviations of the vk consumption from the mean consumption drawn
    from a normal distribution of shape (days, n_ids).
    """
    rng = np.random.default_rng(seed)
    vk_input = rng.normal(loc=1, scale=0.1, size=(days, nids))

    return vk_input


def save_results(ids, measurements, n, filename):
    """
    Saves measurements to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        data = pd.read_csv(directory + filename)
    except FileNotFoundError:
        data = pd.DataFrame(columns=[
            'ID', 'Number of observations', 'INR', 'Dose 1 in mg',
            'Dose 2 in mg', 'Dose 3 in mg',
            'Dose 4 in mg', 'Dose 5 in mg', 'Dose 6 in mg', 'Dose 7 in mg',
            'Dose 8 in mg', 'Dose 9 in mg', 'Dose 10 in mg', 'Dose 11 in mg',
            'Dose 12 in mg', 'Dose 13 in mg', 'Dose 14 in mg', 'Dose 15 in mg',
            'Dose 16 in mg', 'Dose 17 in mg', 'Dose 18 in mg', 'Dose 19 in mg'
            ])

    # Split data into data with n observations and the rest
    mask = data['Number of observations'] != n
    rest = data[mask]
    data = data[~mask]

    # Add data to file
    for idx, patient_id in enumerate(ids):
        if patient_id in data.ID.unique():
            mask = data.ID == patient_id
            data.loc[mask, 'INR'] = measurements[idx]
        else:
            df = {
                'ID': [patient_id],
                'Number of observations': [n],
                'INR': [measurements[idx]]
            }
            data = pd.concat((data, pd.DataFrame(df)), ignore_index=True)

    # Merge dataframes again
    data = pd.concat((rest, data), ignore_index=True)

    # Save file
    data.to_csv(directory + filename, index=False)


if __name__ == '__main__':
    # Set up day parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int)
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    if not args.number:
        raise ValueError('Invalid number.')
    n = args.number
    if not args.filename:
        raise ValueError('Invalid filename.')
    filename = args.filename

    # Define number of days for simulation
    days = 19
    if n > days:
        raise ValueError(
            'Invalid number. The script is set to 19 trial days. Therefore '
            'n cannot exceed 19.')

    # Load patient data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + '/data/mipd_trial_cohort.csv')

    # Define model
    model, _ = define_wajima_model(patient=True, inr_test=True)

    # Define IOV and EV
    ids = data.ID.unique()
    rng = np.random.default_rng(seed=4)
    delays = rng.exponential(scale=0.5, size=(days, len(ids)))
    vk_input = get_vk_consumption(days, len(ids), seed=14)

    # Simulate measurements
    rng = np.random.default_rng(seed=len(ids)*n)
    measurements = np.empty(shape=len(ids))
    for idp, patient in data.iterrows():
        d = delays[:, idp]
        vk = vk_input[:, idp]
        r = get_regimen(patient, n, d, filename)
        model.set_dosing_regimen(r)
        measurements[idp] = generate_measurement(model, patient, n, d, vk, rng)

    save_results(ids, measurements, n, filename)
