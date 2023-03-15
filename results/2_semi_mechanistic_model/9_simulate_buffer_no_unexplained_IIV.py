import argparse
import os

import chi
import myokit
import numpy as np
import xarray as xr

from model import define_hamberg_model, define_hamberg_population_model


def generate_measurements(
        model, parameters, policy, n_measurements, time_scale, epsilon):
    """
    Generates TDM data for individual.
    """
    measurements = np.empty((n_measurements, 2))
    days = n_measurements * time_scale
    times = np.arange(0, days, time_scale) * 24
    for idt, time in enumerate(times):
        # Measure response
        try:
            measurements[idt, 0] = model.sample(
                parameters, [time], return_df=False)
        except TypeError:
            # Simulation error ocurred, so we will treat this as being a very
            # large INR
            measurements[idt, 0] = 30

        # Choose dose according to epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            dose = np.random.uniform(0, 30)
            if dose < 0.5:
                dose = 0
            elif dose < 1.5:
                dose = 1
            else:
                dose = np.round(2 * dose) / 2
        else:
            mask = policy[:, 0] >= measurements[idt, 0]
            try:
                dose = policy[mask][0, 1]
            except IndexError:
                # Measurement is outside predefined range, so choose dose
                # according to largest INR.
                dose = policy[-1, 1]

        measurements[idt, 1] = dose

        regimen = define_dosing_regimen(measurements[:idt+1, 1], time_scale)
        model.set_dosing_regimen(regimen)

    return measurements


def define_dosing_regimen(doses, time_scale):
    """
    Returns a dosing regimen with delayed administration times.
    """
    duration = 0.01
    doses = np.broadcast_to(
        doses[:, np.newaxis], (len(doses), time_scale)).flatten()
    regimen = myokit.Protocol()
    for day, dose in enumerate(doses):
        if dose == 0:
            continue
        regimen.add(myokit.ProtocolEvent(
            level=dose/duration,
            start=day*24,
            duration=duration))

    return regimen


if __name__ == '__main__':
    # Set up day parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    if not args.filename:
        raise ValueError('Invalid filename.')
    filename = args.filename

    # Define epsilon
    epsilon = 0.05

    # Define simulation parameters
    n_measurements = 30
    time_scale = 1  # (in days)

    # Load patient data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    covariates = np.load(
        directory + '/4_reinforcement_learning/covariates.temp.npy')

    # Load policy
    policy = np.load(
        directory + '/4_reinforcement_learning/policy.temp.npy')

    # Define model
    model, _ = define_hamberg_model(baseline_inr=None)
    model.set_outputs(['myokit.inr'])
    model = chi.PredictiveModel(model, chi.LogNormalErrorModel())
    pop_model = define_hamberg_population_model(
        centered=True, inr=True, conc=False, fixed_y0=False)
    posterior2 = xr.load_dataset(
        directory +
        '/2_semi_mechanistic_model/posteriors/posterior_trial_phase_II.nc')
    posterior3 = xr.load_dataset(
        directory
        + '/2_semi_mechanistic_model/posteriors/posterior_trial_phase_III.nc')
    pop_parameters = np.vstack([
        posterior2['Log mean myokit.baseline_inr'].values.flatten(),
        posterior2['Log std. myokit.baseline_inr'].values.flatten() * 1E-3,
        posterior2['Rel. baseline INR A'].values.flatten(),
        posterior3['Log mean myokit.elimination_rate'].values.flatten(),
        posterior2['Log std. myokit.elimination_rate'].values.flatten() * 1E-3,
        posterior3['Rel. elimination rate shift *2*2'].values.flatten(),
        posterior3['Rel. elimination rate shift *3*3'].values.flatten(),
        posterior3['Rel. elimination rate shift with age'].values.flatten(),
        posterior3[
            'Log mean myokit.half_maximal_effect_concentration'
        ].values.flatten(),
        posterior2[
            'Log std. myokit.half_maximal_effect_concentration'
        ].values.flatten() * 1E-3,
        posterior3['Rel. EC50 shift AA'].values.flatten(),
        posterior2['Pooled myokit.transition_rate_chain_1'].values.flatten(),
        posterior2['Pooled myokit.transition_rate_chain_2'].values.flatten(),
        posterior3['Log mean myokit.volume'].values.flatten(),
        posterior2['Log std. myokit.volume'].values.flatten() * 1E-3,
        posterior2['Pooled Sigma log'].values.flatten()]).T
    n_samples = len(pop_parameters)

    # Simulate treatment response data
    n_ids = len(covariates)
    measurements = np.empty(shape=(n_ids, n_measurements, 2))
    for idc, cov in enumerate(covariates):
        # Sample parameters for individual
        posterior_sample = pop_parameters[np.random.choice(n_samples)]
        parameters = pop_model.sample(posterior_sample, covariates=cov)[0]
        measurements[idc] = generate_measurements(
            model, parameters, policy[idc], n_measurements, time_scale,
            epsilon)

    np.save(filename, measurements)
