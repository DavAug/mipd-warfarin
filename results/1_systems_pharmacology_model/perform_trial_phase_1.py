import os
from xml.etree.ElementInclude import include

import chi
import numpy as np
from traitlets import directional_link

from model import define_wajima_model, define_hartmann_population_model


def define_model():
    # Define QSP model
    mechanistic_model, _ = define_wajima_model()
    mechanistic_model.set_outputs(['central_warfarin.warfarin_concentration'])

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
    seed = 1234
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

    typical_age = 68
    np.random.seed(seed)
    covariates[:, 3] = np.random.lognormal(
        mean=np.log(typical_age), sigma=0.1, size=n)

    # (VKORC does not affect PK, but we just synthethise them for completeness)
    n_vkorc1_AA = int(np.ceil(0.15 * n))
    covariates[:n_vkorc1_AA, [0, 1, 4]] += 1
    n_vkorc1_GA = int(np.ceil(0.485 * n))
    covariates[:n_vkorc1_AA+n_vkorc1_GA, [0, 3, 4]] += 1

    # Shuffle CYP-VKORC pairs
    indices = np.random.choice(np.arange(n), replace=False, size=n)
    covariates[:, 2] = covariates[indices, 2]

    return covariates


def generate_data(model, parameters, covariates):
    # Define measurement times in h since dose
    times = np.array([10, 35, 60])

    # Define dosing regimen (single dose 10 mg)
    model.set_dosing_regimen(dose=10, start=0)

    # Sample measurements
    n = len(covariates)
    seed = 67
    data = model.sample(
        parameters=parameters, times=times, covariates=covariates, seed=seed,
        n_samples=n, include_regimen=True)

    return data


if __name__ == '__main__':
    n = 60
    m, p = define_model()
    c = define_demographics(n)
    d = generate_data(m, p, c)

    # Save data to .csv
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d.to_csv(directory + '/data/trial_phase_I.csv')
