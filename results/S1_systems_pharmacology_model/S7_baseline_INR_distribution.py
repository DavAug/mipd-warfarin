import os

import chi
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

    return mechanistic_model, error_model, population_model, parameters


def define_demographics(n):
    """
    The frequencies of the alleles as well as age ranges are modelled
    after Hamberg et al (2011).
    """
    seed = 456
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


def generate_data(
        mechanistic_model, error_model, population_model, parameters,
        covariates):
    # Define measurement times in h since dose
    # NOTE: Initial simulation time ensures that the patient's coagulation
    # network is in steady state
    warmup = 100
    days = 21
    times = np.array([1, 2, 3, 5, 7, 13, 20]) * 24
    sim_times = times + warmup * 24
    mean_delay = 0.5
    vk_intake_std = 0.1

    # Perform trial for each patient separately and adjust dosing regimen
    # if INR is outside therapeutic window
    seed = 12345
    data = pd.DataFrame(columns=[
        'ID', 'Time', 'Observable', 'Value', 'Duration', 'Dose'])
    for idc, cov in enumerate(covariates):
        # Sample patient parameters
        psi = population_model.sample(
            parameters, seed=seed+idc, covariates=cov)[0]

        # Sample delays
        np.random.seed(idc+1)
        delta_t = np.random.exponential(scale=mean_delay, size=days)

        # Sample vitamin K intake
        vk_input = np.random.normal(loc=1, scale=vk_intake_std, size=days)

        # Simulate treatment response
        inrs = []
        for idt, time in enumerate(sim_times):
            # Simulate QSP model
            idx = int(time // 24 - warmup)
            t = [warmup * 24, time + delta_t[idx]]
            inr = mechanistic_model.simulate(
                parameters=psi[:-1], times=t,
                vk_input=vk_input[:idx+1])[0, 1]

            # Sample measurement
            inr = error_model.sample(
                psi[-1:], model_output=[inr], seed=1000+idc+1000*idt)[0, 0]
            inrs.append(inr)

        # Store results
        df = pd.DataFrame({
            'ID': [idc] * 30,
            'Time': list(times) + [np.nan] * 3 + list(range(days-1)),
            'Observable': [
                'INR'] * 7 + ['CYP2C9', 'Age', 'VKORC1'] + [np.nan] * 20,
            'Value': inrs + list(cov[2:]) + [np.nan] * 20,
        })
        data = pd.concat((data, df), ignore_index=True)

    return data


if __name__ == '__main__':
    n = 1000
    mm, em, pm, p = define_model()
    c = define_demographics(n)
    d = generate_data(mm, em, pm, p, c)

    # Save data to .csv
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d.to_csv(directory + '/data/S7_baseline_INR_distribution.csv')
