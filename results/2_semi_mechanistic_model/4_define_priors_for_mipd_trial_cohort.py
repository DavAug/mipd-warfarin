import os

import numpy as np
import pandas as pd
import xarray as xr

from model import define_hamberg_model, define_hamberg_population_model


def load_cohort():
    # Load dataframe
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + '/data/mipd_trial_cohort.csv')

    # Reshape into covariate matrix [VKORC, CYP, Age, VKORC]
    ids = data.ID.dropna().unique()
    covariates = np.empty(shape=(len(ids), 4))
    for idx, _id in enumerate(ids):
        temp = data[data.ID == _id]
        covariates[idx, 0] = temp['VKORC1'].values
        covariates[idx, 1] = temp['CYP2C9'].values
        covariates[idx, 2] = temp['Age'].values
        covariates[idx, 3] = temp['VKORC1'].values

    return ids, covariates


def load_posteriors():
    # Import posteriors from trial II and III
    directory = os.path.dirname(os.path.abspath(__file__))
    posterior2 = xr.load_dataset(
        directory + '/posteriors/posterior_trial_phase_II.nc')
    posterior3 = xr.load_dataset(
        directory + '/posteriors/posterior_trial_phase_III.nc')

    # Reshape posteriors into matrix (n_samples, n_parameters)
    # NOTE: Baseline INR and transition rates of chains were not informed by
    # trial III, so we have to get those from trial phase II
    parameter_names = [
        'Log mean myokit.baseline_inr',
        'Log std. myokit.baseline_inr',
        'Rel. baseline INR A',
        'Log mean myokit.elimination_rate',
        'Log std. myokit.elimination_rate',
        'Rel. elimination rate shift *2*2',
        'Rel. elimination rate shift *3*3',
        'Rel. elimination rate shift with age',
        'Log mean myokit.half_maximal_effect_concentration',
        'Log std. myokit.half_maximal_effect_concentration',
        'Rel. EC50 shift AA',
        'Pooled myokit.transition_rate_chain_1',
        'Pooled myokit.transition_rate_chain_2',
        'Log mean myokit.volume',
        'Log std. myokit.volume',
        'Pooled Sigma log'
    ]
    parameters = np.vstack([
        posterior2[parameter_names[0]].values.flatten(),
        posterior2[parameter_names[1]].values.flatten(),
        posterior2[parameter_names[2]].values.flatten(),
        posterior3[parameter_names[3]].values.flatten(),
        posterior2[parameter_names[4]].values.flatten(),
        posterior3[parameter_names[5]].values.flatten(),
        posterior3[parameter_names[6]].values.flatten(),
        posterior3[parameter_names[7]].values.flatten(),
        posterior3[parameter_names[8]].values.flatten(),
        posterior2[parameter_names[9]].values.flatten(),
        posterior3[parameter_names[10]].values.flatten(),
        posterior2[parameter_names[11]].values.flatten(),
        posterior2[parameter_names[12]].values.flatten(),
        posterior3[parameter_names[13]].values.flatten(),
        posterior2[parameter_names[14]].values.flatten(),
        posterior2[parameter_names[15]].values.flatten()]).T

    return parameters


def derive_priors(model, covariates, parameters):
    """
    Derives the prior distribution for an individual based on the covariates
    and the posterior of the population parameters.
    """
    # Sample from prior distribution
    n_samples = len(parameters)
    n_ids = len(covariates)
    n_dim = model.n_dim()
    prior_samples = np.empty(shape=(n_samples, n_ids, n_dim))
    for ids, sample in enumerate(parameters):
        prior_samples[ids] = model.sample(
            parameters=sample, covariates=covariates, n_samples=n_ids,
            seed=ids)

    # Compute mean and std of prior samples
    means = np.mean(prior_samples, axis=0)
    stds = np.std(prior_samples, axis=0, ddof=1)

    return means, stds


def save_prior_to_file(ids, pr):
    # Import existing file
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/mipd_trial_prior_distributions.csv'
    try:
        df = pd.read_csv(directory + filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            'ID', 'Number of observations', 'Parameter', 'Mean', 'Std.'])

    # Reshape prior means and std to dataframe
    means, stds = pr
    m, _ = define_hamberg_model(baseline_inr=None)
    parameter_names = list(m.parameters()) + ['Sigma log']
    if 0 in df['Number of observations'].unique():
        for idp, name in enumerate(parameter_names):
            mask1 = \
                (df['Number of observations'] == 0) & (df.Parameter == name)
            for idx, _id in enumerate(ids):
                mask2 = df.ID == _id
                mask = mask1 & mask2
                df.loc[mask, 'Mean'] = means[idx, idp]
                df.loc[mask, 'Std.'] = means[idx, idp]
    else:
        for idp, name in enumerate(parameter_names):
            df = pd.concat(
                (df, pd.DataFrame({
                    'ID': ids,
                    'Number of observations': 0,
                    'Parameter': name,
                    'Mean': means[:, idp],
                    'Std.': stds[:, idp]})),
                ignore_index=True)

    # Save priors to file
    df.to_csv(directory + filename, index=False)


if __name__ == '__main__':
    m = define_hamberg_population_model(
        centered=True, conc=False, fixed_y0=False)
    ids, c = load_cohort()
    p = load_posteriors()
    pr = derive_priors(m, c, p)
    save_prior_to_file(ids, pr)
