import os

import chi
import myokit
import numpy as np
import pandas as pd

from model import define_wajima_model


def define_model():
    # Define QSP model
    mechanistic_model, parameters_df = define_wajima_model(
        patient=True, inr_test=True)

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Get data-generating parameters from dataframe
    parameters = np.array([
        parameters_df[parameters_df.Parameter == p].Value.values[0]
        for p in mechanistic_model.parameters()])

    return mechanistic_model, error_model, parameters


def get_initial_dosing_regimen(covariates, offset):
    """
    Implements Hamberg et al's (2011) dosing table based on the VKORC1
    genotype, the CYP2C9 genotype and age.

    All values are rounded to combinations of commercialised tablets.
    """
    v, _, c, a, _ = covariates

    if c == 0:
        if v == 0:
            if a < 60:
                dose = 8.5
            elif a < 80:
                dose = 7.5
            else:
                dose = 6.5
        elif v == 1:
            if a < 60:
                dose = 6
            elif a < 80:
                dose = 5.5
            else:
                dose = 5
        elif v == 2:
            if a < 60:
                dose = 4.0
            elif a < 80:
                dose = 3.5
            else:
                dose = 3
    elif c == 1:
        if v == 0:
            if a < 60:
                dose = 6.5
            elif a < 80:
                dose = 5.5
            else:
                dose = 5
        elif v == 1:
            if a < 60:
                dose = 4.5
            elif a < 80:
                dose = 4
            else:
                dose = 3.5
        elif v == 2:
            if a < 60:
                dose = 3.0
            elif a < 80:
                dose = 2.5
            else:
                dose = 2.5
    elif c == 2:
        if v == 0:
            if a < 60:
                dose = 5.5
            elif a < 80:
                dose = 4.5
            else:
                dose = 4
        elif v == 1:
            if a < 60:
                dose = 4
            elif a < 80:
                dose = 3.5
            else:
                dose = 3
        elif v == 2:
            if a < 60:
                dose = 2.5
            elif a < 80:
                dose = 2
            else:
                dose = 2
    elif c == 3:
        if v == 0:
            if a < 60:
                dose = 4.5
            elif a < 80:
                dose = 4
            else:
                dose = 3.5
        elif v == 1:
            if a < 60:
                dose = 3
            elif a < 80:
                dose = 3
            else:
                dose = 2.5
        elif v == 2:
            if a < 60:
                dose = 2
            elif a < 80:
                dose = 2
            else:
                dose = 2
    elif c == 4:
        if v == 0:
            if a < 60:
                dose = 3
            elif a < 80:
                dose = 3
            else:
                dose = 2.5
        elif v == 1:
            if a < 60:
                dose = 2.5
            elif a < 80:
                dose = 2
            else:
                dose = 2
        elif v == 2:
            if a < 60:
                dose = 2
            elif a < 80:
                dose = 1
            else:
                dose = 1
    elif c == 5:
        if v == 0:
            if a < 60:
                dose = 2
            elif a < 80:
                dose = 2
            else:
                dose = 2
        elif v == 1:
            if a < 60:
                dose = 2
            elif a < 80:
                dose = 1
            else:
                dose = 1
        elif v == 2:
            if a < 60:
                dose = 1
            elif a < 80:
                dose = 1
            else:
                dose = 1

    # Define regimen for the first 3 days
    duration = 0.01
    dose_rate = dose / duration
    dose_rates = []
    regimen = myokit.Protocol()
    for day in range(3):
        time = (offset+day) * 24
        regimen.add(myokit.ProtocolEvent(
            level=dose_rate, start=time, duration=duration, period=0))
        dose_rates.append(dose_rate)

    return regimen, dose_rates


def adjust_dosing_regimen(dose_rates, latest_inr, days, offset):
    # Naively adjust dose by fraction to target INR, if INR is outside
    # therapeutic range
    duration = 0.01
    dose_rate = dose_rates[-1]
    target = 2.5
    if (latest_inr < 2) or (latest_inr > 3):
        dose_rate = dose_rate * target / latest_inr

        # Make sure that dose can be taken with conventional tablets
        dose = dose_rate * duration
        if dose < 0.5:
            dose = 0
        elif dose < 1.5:
            dose = 1
        elif dose < 2.25:
            dose = 2
        else:
            dose = np.round(2 * dose) / 2

        if dose > 30:
            dose = 30
        dose_rate = dose / duration

    # Reconstruct already administered dose events and add dose events until
    # the next measurement
    new_regimen = myokit.Protocol()
    for idx, dr in enumerate(dose_rates):
        new_regimen.add(myokit.ProtocolEvent(
            level=dr, start=(offset+idx)*24, duration=duration,
            period=0))
    n_doses = len(dose_rates)
    for day in range(days):
        new_regimen.add(myokit.ProtocolEvent(
            level=dose_rate, duration=duration, period=0,
            start=(offset+n_doses+day)*24))
        dose_rates.append(dose_rate)

    return new_regimen, dose_rates


def generate_data(
        mechanistic_model, error_model, parameters, nids):
    # Define measurement times in h since dose
    # NOTE: Initial simulation time ensures that the patient's coagulation
    # network is in steady state
    warmup = 100
    times = np.array([
        1, 2, 3, 5, 7, 13, 20, 27, 34, 41, 48, 55]) * 24
    sim_times = times + warmup * 24

    # Perform trial for each patient separately and adjust dosing regimen
    # if INR is outside therapeutic window
    data = pd.DataFrame(columns=[
        'ID', 'Time', 'Observable', 'Value', 'Duration', 'Dose'])
    for idc in range(nids):
        # Define model parameters
        psi = list(parameters) + [0.1]

        # Simulate treatment response
        inrs = []
        cov = [0, 0, 0, 71, 0]
        regimen, dose_rates = get_initial_dosing_regimen(cov, warmup)
        mechanistic_model.set_dosing_regimen(regimen)
        for idt, time in enumerate(sim_times):
            # Simulate QSP model
            t = [warmup * 24, time]
            inr = mechanistic_model.simulate(
                parameters=psi[:-1], times=t,
                vk_input=None)[0, 1]

            # Sample measurement
            inr = error_model.sample(
                psi[-1:], model_output=[inr], seed=1000+idc+1000*idt)[0, 0]
            inrs.append(inr)

            if (idt < 2) or idt > 9:
                # Dose remains unchanged
                continue
            elif idt < 9:
                # Adjust next dose based on INR response
                days_to_next_meas = int(
                    (sim_times[idt+1] - sim_times[idt]) // 24)
                regimen, dose_rates = adjust_dosing_regimen(
                    dose_rates, inr, days_to_next_meas, warmup)
                mechanistic_model.set_dosing_regimen(regimen)
            else:
                # Change dose for the last time
                days_to_next_meas = int(
                    (sim_times[-1] - sim_times[idt]) // 24)
                regimen, dose_rates = adjust_dosing_regimen(
                    dose_rates, inr, days_to_next_meas, warmup)
                mechanistic_model.set_dosing_regimen(regimen)

        # Store results
        n_doses = len(dose_rates)
        doses = np.array(dose_rates) * 0.01
        n_times = len(times)
        dose_times = list(np.arange(n_doses) * 24)
        df = pd.DataFrame({
            'ID': [idc] * (n_times + 3 + n_doses),
            'Time': list(times) + [np.nan] * 3 + dose_times,
            'Observable': [
                'INR'] * n_times + ['CYP2C9', 'Age', 'VKORC1'] \
                + [np.nan] * n_doses,
            'Value': inrs + list(cov[2:]) + [np.nan] * n_doses,
            'Dose': [
                np.nan] * n_times + [np.nan] * 3 + list(doses),
            'Duration': [
                np.nan] * n_times + [np.nan] * 3 + [0.01] * n_doses
        })
        data = pd.concat((data, df), ignore_index=True)

    return data


if __name__ == '__main__':
    n = 1000
    mm, em, p = define_model()
    d = generate_data(mm, em, p, n)

    # Save data to .csv
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d.to_csv(directory + '/data/S1_maintenance_distribution_only_noise.csv')
