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


def define_demographics(n):
    """
    The frequencies of the alleles as well as age ranges are modelled
    after Hamberg et al (2011).
    """
    n_cov = 3  # VKORC1, CYP2C9, Age
    covariates = np.zeros(shape=(n, n_cov))
    covariates[:, 0] = 0
    covariates[:, 1] = 0
    covariates[:, 2] = 71

    return covariates


def get_initial_dosing_regimen(offset, delta_t):
    """
    Defines a simple loading dose sequence of 10mg, 7.5mg, 5mg.
    """
    # Define regimen for the first 3 days
    doses = np.array([10, 7.5, 5])
    duration = 0.01
    dose_rate = doses / duration
    dose_rates = []
    regimen = myokit.Protocol()
    for day in range(3):
        time = (offset+day) * 24 + delta_t[day]
        regimen.add(myokit.ProtocolEvent(
            level=dose_rate[day], start=time, duration=duration, period=0))
        dose_rates.append(dose_rate[day])

    return regimen, dose_rates


def adjust_dosing_regimen(dose_rates, latest_inr, days, offset, delta_t):
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
            level=dr, start=(offset+idx)*24+delta_t[idx], duration=duration,
            period=0))
    n_doses = len(dose_rates)
    for day in range(days):
        new_regimen.add(myokit.ProtocolEvent(
            level=dose_rate, duration=duration, period=0,
            start=(offset+n_doses+day)*24+delta_t[n_doses+day]))
        dose_rates.append(dose_rate)

    return new_regimen, dose_rates


def generate_data(
        mechanistic_model, error_model, parameters, covariates):
    # Define measurement times in h since dose
    # NOTE: Initial simulation time ensures that the patient's coagulation
    # network is in steady state
    warmup = 100
    days = 56
    times = np.array([1, 2, 3, 5, 7, 13, 20, 27, 34, 41, 48, 55]) * 24
    sim_times = times + warmup * 24
    mean_delay = 0.5
    vk_intake_std = 0.1
    sigma_log = [0.1]

    # Perform trial for each patient separately and adjust dosing regimen
    # if INR is outside therapeutic window
    data = pd.DataFrame(columns=[
        'ID', 'Time', 'Observable', 'Value', 'Duration', 'Dose'])
    for idc, cov in enumerate(covariates):
        # Sample patient parameters
        psi = parameters

        # Sample delays
        np.random.seed(idc+1)
        delta_t = np.random.exponential(scale=mean_delay, size=days)

        # Sample vitamin K intake
        vk_input = np.random.normal(loc=1, scale=vk_intake_std, size=days)

        # Simulate treatment response
        inrs = []
        regimen, dose_rates = get_initial_dosing_regimen(warmup, delta_t)
        mechanistic_model.set_dosing_regimen(regimen)
        for idt, time in enumerate(sim_times):
            # Simulate QSP model
            idx = int(time // 24 - warmup)
            t = [warmup * 24, time + delta_t[idx]]
            inr = mechanistic_model.simulate(
                parameters=psi, times=t,
                vk_input=vk_input[:idx+1])[0, 1]

            # Sample measurement
            inr = error_model.sample(
                sigma_log, model_output=[inr], seed=1000+idc+1000*idt)[0, 0]
            inrs.append(inr)

            if (idt < 2) or idt > 9:
                # Dose remains unchanged
                continue
            elif idt < 9:
                # Adjust next dose based on INR response
                days_to_next_meas = int(
                    (sim_times[idt+1] - sim_times[idt]) // 24)
                regimen, dose_rates = adjust_dosing_regimen(
                    dose_rates, inr, days_to_next_meas, warmup, delta_t)
                mechanistic_model.set_dosing_regimen(regimen)
            else:
                # Change dose for the last time
                days_to_next_meas = int(
                    (sim_times[-1] - sim_times[idt]) // 24)
                regimen, dose_rates = adjust_dosing_regimen(
                    dose_rates, inr, days_to_next_meas, warmup, delta_t)
                mechanistic_model.set_dosing_regimen(regimen)

        # Store results
        n_meas = len(times)
        doses = np.array(dose_rates) * 0.01
        n_doses = len(doses)
        dose_times = np.arange(n_doses) * 24
        df = pd.DataFrame({
            'ID': [idc] * (n_meas + 3 + n_doses),
            'Time': list(times) + [np.nan] * 3 + list(dose_times),
            'Observable': [
                'INR'] * n_meas + ['VKORC1', 'CYP2C9', 'Age']
                + [np.nan] * n_doses,
            'Value': inrs + list(cov) + [np.nan] * n_doses,
            'Dose': [np.nan] * (n_meas + 3) + list(doses),
            'Duration': [np.nan] * (n_meas + 3) + [0.01] * n_doses
        })
        data = pd.concat((data, df), ignore_index=True)

    return data


if __name__ == '__main__':
    n = 1000
    mm, em, p = define_model()
    c = define_demographics(n)
    d = generate_data(mm, em, p, c)

    # Save data to .csv
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d.to_csv(directory + '/data/S5_maintenance_distribution_no_IIV.csv')
