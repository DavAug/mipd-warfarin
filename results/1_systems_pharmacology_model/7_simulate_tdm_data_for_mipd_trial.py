import argparse
import os

import chi
import myokit
import numpy as np
import pandas as pd

from model import define_wajima_model


def define_model():
    """
    Defines Wajima's model of the coagulation network.
    """
    model, _ = define_wajima_model(patient=True, inr_test=True)
    error_model = chi.LogNormalErrorModel()
    model = chi.PredictiveModel(model, error_model)

    return model


def get_regimen(n):
    """
    Imports the patient's dosing regimen.
    """
    if n == 1:
        # This is the first TDM  measurement of the patient, which is taken
        # before any drug is administered. So we return an empty dosing
        # regimen
        return myokit.Protocol()

    # Import file with dosing regimens
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = \
        '/2_semi_mechanistic_model' \
        + '/mipd_trial_predicted_dosing_regimens_bayesian_optimisation.csv'
    try:
        df = pd.read_csv(directory + filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Invalid number of observations. File does not exist. Try '
            'executing the script with number = 1.')

    # Get dosing regimen from dataframe
    doses = df[df['Number of observations'] == n][[
        'Dose %d in mg' % (d+1) for d in range(n)]].values[0]

    # Define dosing regimen
    cal_time = 100
    duration = 0.01
    dose_rates = doses / duration
    regimen = myokit.Protocol()
    for idx, dr in enumerate(dose_rates):
        if dr == 0:
            continue
        regimen.add(myokit.ProtocolEvent(
            level=dr,
            start=(cal_time + idx) * 24,
            duration=duration))

    return regimen


def generate_measurement(model, patient, n):
    """
    Generates TDM data for individual.
    """
    # Get patient ID (use this as seed for RNG)
    seed = int(patient.ID)

    # Get parameters of indvidual
    parameters = np.array(patient[model.get_parameter_names()].values)

    # Simulate measurements
    # NOTE nth  measurement occurs on (n-1)th day. And we let the system
    # equilibrate for 100 days.
    cal_time = 100
    times = [(cal_time + n - 1) * 24]
    measurement = model.sample(
        parameters=parameters, times=times, seed=seed, return_df=False
    )[0, 0, 0]

    return measurement


def save_results(ids, measurements, n):
    """
    Saves measurements to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = \
        '/2_semi_mechanistic_model' \
        + '/mipd_trial_predicted_dosing_regimens.csv'
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
    args = parser.parse_args()

    if not args.number:
        raise ValueError('Invalid number.')
    n = args.number

    # Load patient data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + '/data/mipd_trial_cohort.csv')

    # Define model
    model = define_model()

    # Simulate measurements
    ids = data.ID.unique()
    measurements = np.empty(shape=len(ids))
    for idp, patient in data.iterrows():
        r = get_regimen(n)
        model.set_dosing_regimen(r)
        measurements[idp] = generate_measurement(model, patient, n)

    save_results(ids, measurements, n)
