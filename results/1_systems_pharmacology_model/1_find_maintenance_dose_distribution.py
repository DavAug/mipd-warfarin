import os

import numpy as np
import pandas as pd

from model import (
    define_wajima_model,
    define_hartmann_population_model,
    find_dose)

def define_patients(n_ids):
    # Define covariates
    # Frequency of alleles matches dataset in publication
    n_cov = 5  # NOTE cov 1, 2 and 5 reference the VKORC genotype
    covariates = np.zeros(shape=(n_ids, n_cov))

    n_cyp2p9_33 = int(np.ceil(0.006 * n_ids))
    covariates[:n_cyp2p9_33, 2] += 1
    n_cyp2p9_23 = int(np.ceil(0.012 * n_ids))
    covariates[:n_cyp2p9_33+n_cyp2p9_23, 2] += 1
    n_cyp2p9_22 = int(np.ceil(0.014 * n_ids))
    covariates[:n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22, 2] += 1
    n_cyp2p9_13 = int(np.ceil(0.123 * n_ids))
    covariates[:n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22+n_cyp2p9_13, 2] += 1
    n_cyp2p9_12 = int(np.ceil(0.184 * n_ids))
    covariates[
        :n_cyp2p9_33+n_cyp2p9_23+n_cyp2p9_22+n_cyp2p9_13+n_cyp2p9_12, 2] += 1

    typical_age = 68
    np.random.seed(12)
    covariates[:, 3] = np.random.lognormal(
        mean=np.log(typical_age), sigma=0.1, size=n_ids)

    n_vkorc1_AA = int(np.ceil(0.15 * n_ids))
    covariates[:n_vkorc1_AA, [0, 1, 4]] += 1
    n_vkorc1_GA = int(np.ceil(0.485 * n_ids))
    covariates[:n_vkorc1_AA+n_vkorc1_GA, [0, 1, 4]] += 1

    # Shuffle covariates
    patient_cov = np.copy(covariates)
    indices = np.random.choice(
        np.arange(n_ids), replace=False, size=n_ids)
    patient_cov[:, 2] = covariates[indices, 2]

    # Sample individual-level parameters
    seed = 123
    population_model, pop_parameters_df = define_hartmann_population_model()
    parameters = np.array([
        pop_parameters_df[pop_parameters_df.Parameter == p].Value.values[0]
        for p in population_model.get_parameter_names()
    ])
    patients = population_model.sample(
        parameters, n_samples=n_ids, seed=seed, covariates=patient_cov)

    return patients, patient_cov


if __name__ == '__main__':
    # Define model and patients
    n_ids = 1000
    model, _ = define_wajima_model(patient=True, inr_test=True)
    patients, covariates = define_patients(n_ids)

    # Find doses
    target = 2.5
    doses = find_dose(model, patients[:, :-1], target=target)

    # Save results
    directory = os.path.dirname(os.path.abspath(__file__))
    data = {
        param: patients[:, idp]
        for idp, param in enumerate(model.parameters())}
    data['ID'] = np.arange(n_ids) + 1
    data['Maintenance dose'] = doses
    data['Target INR'] = [target] * n_ids
    data['VKORC1'] = covariates[:, 0]
    data['CYP2C9'] = covariates[:, 2]
    data['Age'] = covariates[:, 3]
    pd.DataFrame(data).to_csv(directory + '/maintenance_dose_distribution.csv')
