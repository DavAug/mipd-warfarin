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
    parameter_names = model.get_parameter_names()
    population_model.set_dim_names(parameter_names)

    model = chi.PopulationPredictiveModel(model, population_model)

    # Get data-generating parameters from dataframe
    parameters = np.array([
        parameters_df[parameters_df.Parameter == p].Value.values[0]
        for p in model.get_parameter_names()])

    return population_model, parameters


def define_demographics(n):
    """
    The frequencies of the alleles as well as age ranges are modelled
    after Hamberg et al (2011).
    """
    seed = 42
    n_cov = 5  # (1, 2, 5) all refer to the VKORC1 genotype
    covariates = np.zeros(shape=(n, n_cov))

    n_cyp2p9_33 = int(np.ceil(0.01 * n))
    covariates[:n_cyp2p9_33, 2] += 1
    n_cyp2p9_23 = int(np.ceil(0.02 * n))
    covariates[:n_cyp2p9_33+n_cyp2p9_23, 2] += 1
    n_cyp2p9_22 = int(np.ceil(0.03 * n))
    covariates[:n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22, 2] += 1
    n_cyp2p9_13 = int(np.ceil(0.12 * n))
    covariates[:n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22+n_cyp2p9_13, 2] += 1
    n_cyp2p9_12 = int(np.ceil(0.20 * n))
    covariates[
        :n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22+n_cyp2p9_13+n_cyp2p9_12, 2] += 1

    typical_age = 50
    np.random.seed(seed)
    covariates[:, 3] = np.random.lognormal(
        mean=np.log(typical_age), sigma=0.15, size=n)

    n_vkorc1_AA = int(np.ceil(0.20 * n))
    covariates[:n_vkorc1_AA, [0, 1, 4]] += 1
    n_vkorc1_GA = int(np.ceil(0.45 * n))
    covariates[:n_vkorc1_AA+n_vkorc1_GA, [0, 1, 4]] += 1

    # Shuffle CYP-VKORC pairs
    indices = np.random.choice(np.arange(n), replace=False, size=n)
    covariates[:, 2] = covariates[indices, 2]

    return covariates


def sample_cohort(model, parameters, covariates):
    # Sample parameters of individuals
    seed = 765
    n_samples = len(covariates)
    parameters = model.sample(
        parameters=parameters,
        n_samples=n_samples,
        covariates=covariates,
        seed=seed)

    # Format data as dataframe
    df = {
        'ID': np.arange(1, n_samples+1),
        'CYP2C9': covariates[:, 2],
        'Age': covariates[:, 3],
        'VKORC1': covariates[:, 4]}

    parameter_names = model.get_dim_names()
    for idn, name in enumerate(parameter_names):
        df[name] = parameters[:, idn]

    data = pd.DataFrame(df)

    return data


if __name__ == '__main__':
    n = 1000
    m, p = define_model()
    c = define_demographics(n)
    d = sample_cohort(m, p, c)

    # Save data to .csv
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d.to_csv(directory + '/data/mipd_trial_cohort.csv')
