import os

import chi
import myokit
import numpy as np
import pandas as pd

from model import define_wajima_model, define_hartmann_population_model


def define_model():
    # Define QSP model
    mechanistic_model, _ = define_wajima_model(patient=True, inr_test=True)

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Define population model
    population_model, parameters_df = define_hartmann_population_model()

    # Compose model
    model = chi.PredictiveModel(mechanistic_model, error_model)
    model = chi.PopulationPredictiveModel(model, population_model)

    # Get data-generating parameters from dataframe
    parameters = np.array([
        parameters_df[parameters_df.Parameter == p].Value.values[0]
        for p in model.get_parameter_names()])

    return model, parameters


def define_demographics(n):
    """
    The frequencies of the alleles as well as age ranges are modelled
    after Hamberg et al (2011).
    """
    seed = 102
    n_cov = 5  # (1, 2, 5) all refer to the VKORC1 genotype
    covariates = np.zeros(shape=(n, n_cov))

    n_cyp2p9_33 = int(np.ceil(0.006 * n))
    covariates[:n_cyp2p9_33, 2] += 1
    n_cyp2p9_23 = int(np.ceil(0.012 * n))
    covariates[:n_cyp2p9_33+n_cyp2p9_23, 2] += 1
    n_cyp2p9_22 = int(np.ceil(0.014 * n))
    covariates[:n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22, 2] += 1
    n_cyp2p9_13 = int(np.ceil(0.123 * n))
    covariates[:n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22+n_cyp2p9_13, 2] += 1
    n_cyp2p9_12 = int(np.ceil(0.184 * n))
    covariates[
        :n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22+n_cyp2p9_13+n_cyp2p9_12, 2] += 1

    typical_age = 65
    np.random.seed(seed)
    covariates[:, 3] = np.random.lognormal(
        mean=np.log(typical_age), sigma=0.1, size=n)

    n_vkorc1_AA = int(np.ceil(0.15 * n))
    covariates[:n_vkorc1_AA, [0, 1, 4]] += 1
    n_vkorc1_GA = int(np.ceil(0.485 * n))
    covariates[:n_vkorc1_AA+n_vkorc1_GA, [0, 1, 4]] += 1

    # Shuffle CYP-VKORC pairs
    indices = np.random.choice(np.arange(n), replace=False, size=n)
    covariates[:, 2] = covariates[indices, 2]

    return covariates


def get_initial_dosing_regimen(covariates, offset):
    """
    Implements Hamberg et al's (2011) dosing table based on the VKORC1
    genotype, the CYP2C9 genotype and age.
    """
    v, _, c, a, _ = covariates

    if c == 0:
        if v == 0:
            if a < 60:
                dose = 8.5
            elif a < 80:
                dose = 7.6
            else:
                dose = 6.7
        elif v == 1:
            if a < 60:
                dose = 6.2
            elif a < 80:
                dose = 5.6
            else:
                dose = 4.9
        elif v == 2:
            if a < 60:
                dose = 4.0
            elif a < 80:
                dose = 3.6
            else:
                dose = 3.2
    elif c == 1:
        if v == 0:
            if a < 60:
                dose = 6.4
            elif a < 80:
                dose = 5.7
            else:
                dose = 5.1
        elif v == 1:
            if a < 60:
                dose = 4.7
            elif a < 80:
                dose = 4.2
            else:
                dose = 3.7
        elif v == 2:
            if a < 60:
                dose = 3.0
            elif a < 80:
                dose = 2.7
            else:
                dose = 2.4
    elif c == 2:
        if v == 0:
            if a < 60:
                dose = 5.3
            elif a < 80:
                dose = 4.7
            else:
                dose = 4.2
        elif v == 1:
            if a < 60:
                dose = 3.9
            elif a < 80:
                dose = 3.5
            else:
                dose = 3.1
        elif v == 2:
            if a < 60:
                dose = 2.5
            elif a < 80:
                dose = 2.2
            else:
                dose = 2.0
    elif c == 3:
        if v == 0:
            if a < 60:
                dose = 4.3
            elif a < 80:
                dose = 3.8
            else:
                dose = 3.4
        elif v == 1:
            if a < 60:
                dose = 3.1
            elif a < 80:
                dose = 2.8
            else:
                dose = 2.5
        elif v == 2:
            if a < 60:
                dose = 2.0
            elif a < 80:
                dose = 1.8
            else:
                dose = 1.6
    elif c == 4:
        if v == 0:
            if a < 60:
                dose = 3.2
            elif a < 80:
                dose = 2.8
            else:
                dose = 2.5
        elif v == 1:
            if a < 60:
                dose = 2.3
            elif a < 80:
                dose = 2.1
            else:
                dose = 1.8
        elif v == 2:
            if a < 60:
                dose = 1.5
            elif a < 80:
                dose = 1.3
            else:
                dose = 1.2
    elif c == 5:
        if v == 0:
            if a < 60:
                dose = 2.1
            elif a < 80:
                dose = 1.8
            else:
                dose = 1.6
        elif v == 1:
            if a < 60:
                dose = 1.5
            elif a < 80:
                dose = 1.4
            else:
                dose = 1.2
        elif v == 2:
            if a < 60:
                dose = 1.0
            elif a < 80:
                dose = 0.9
            else:
                dose = 0.8

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
        if dose < 1.5:
            # Smallest pill
            dose = 1
        elif dose < 2.25:
            dose = 2
        else:
            # From now on, any dose can be delivered in 0.5 steps
            dose = np.round(dose * 2) / 2
        dose_rate = dose / duration

    # Add a dose event until the next check
    new_regimen = myokit.Protocol()
    for idx, dr in enumerate(dose_rates):
        new_regimen.add(myokit.ProtocolEvent(
            level=dr, start=(offset+idx)*24, duration=duration, period=0))
    n_doses = len(dose_rates)
    for day in range(days):
        new_regimen.add(myokit.ProtocolEvent(
            level=dose_rate, start=(offset+n_doses+day)*24, duration=duration,
            period=0))
        dose_rates.append(dose_rate)

    return new_regimen, dose_rates


def generate_data(model, parameters, covariates):
    # Define measurement times in h since dose
    # NOTE: Initial simulation time ensures that the patient's coagulation
    # network is in steady state
    warmup = 100
    times = np.array([1, 2, 3, 5, 7, 13, 20, 27, 34, 41, 48, 55]) * 24
    sim_times = times + warmup * 24

    # Perform trial for each patient separately and adjust dosing regimen
    # if INR is outside therapeutic window
    seed = 12345
    data = pd.DataFrame(columns=[
        'ID', 'Time', 'Observable', 'Value', 'Duration', 'Dose'])
    for idc, cov in enumerate(covariates):
        regimen, dose_rates = get_initial_dosing_regimen(cov, warmup)
        model.set_dosing_regimen(regimen)
        for idt in range(2, len(times)-1):
            latest_inr = model.sample(
                parameters=parameters, times=sim_times[idt:idt+1],
                covariates=cov, seed=seed, n_samples=1, return_df=False
            )[0, 0, 0]

            # Adjust next dose based on INR response
            days = (times[idt+1] - times[idt]) // 24
            regimen, dose_rates = adjust_dosing_regimen(
                dose_rates, latest_inr, days, warmup)
            model.set_dosing_regimen(regimen)

        # Get final INR readouts and dosing regime
        d = model.sample(
            parameters=parameters, times=sim_times[-1:], covariates=cov,
            seed=seed, n_samples=1, include_regimen=True)
        d.ID = idc
        d.Time -= warmup * 24
        data = pd.concat((data, d), ignore_index=True)
        seed += 1

    return data


if __name__ == '__main__':
    n = 1000
    m, p = define_model()
    c = define_demographics(n)
    d = generate_data(m, p, c)

    # Save data to .csv
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d.to_csv(directory + '/data/trial_phase_III.csv')
