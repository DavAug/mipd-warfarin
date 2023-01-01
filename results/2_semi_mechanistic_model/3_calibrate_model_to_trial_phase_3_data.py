import os

import chi
import pandas as pd
import pints

from model import (
    define_steady_state_hamberg_model,
    define_steady_state_hamberg_population_model
)


def define_log_posterior():
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measurements_df = pd.read_csv(
        directory + '/data/trial_phase_III.csv')

    # Keep only steady state measurement
    mask = \
        (measurements_df.Observable != 'INR') | (
        (measurements_df.Observable == 'INR') & (measurements_df.Time == 1320))
    measurements_df = measurements_df[mask]

    # Define hierarchical log-posterior
    mechanistic_model,_ = define_steady_state_hamberg_model()
    error_model = chi.LogNormalErrorModel()
    population_model = define_steady_state_hamberg_population_model(
        centered=False)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(-3.738, 0.027),     # Mean log ke
        pints.GaussianLogPrior(0.116, 0.02),       # Sigma log ke
        pints.GaussianLogPrior(0.571, 0.04),       # Rel. shift ke CYP29P *2
        pints.GaussianLogPrior(0.823, 0.04),       # Rel. shift ke CYP29P *3
        pints.GaussianLogPrior(0.00157, 0.0027),   # Rel. shift ke Age
        pints.GaussianLogPrior(1.471, 0.055),      # Mean log EC50
        pints.GaussianLogPrior(0.208, 0.028),      # Sigma log EC50
        pints.GaussianLogPrior(0.532, 0.041),      # Rel. shift EC50 VKORC1 A
        pints.GaussianLogPrior(2.662, 0.020),      # Mean log volume
        pints.LogNormalLogPrior(-2.31, 0.16),      # Sigma log volume
    )
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df, output_observable_dict={
        'myokit.inr': 'INR'})
    problem.fix_parameters({
        'Log mean myokit.baseline_inr': 0.073,
        'Log std. myokit.baseline_inr': 0.170,
        'Rel. baseline INR A': 1.043,
        'Pooled Sigma log': 0.105
    })
    problem.set_log_prior(log_prior)

    return problem.get_log_posterior()


def run_inference(log_posterior):
    seed = 5
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(True)
    controller.set_sampler(pints.NoUTurnMCMC)

    n_iterations = 1500
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save results
    warmup = 500
    thinning = 1
    directory = os.path.dirname(os.path.abspath(__file__))
    posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning)
    ).to_netcdf(
        directory +
        '/posteriors/posterior_trial_phase_III.nc'
    )


if __name__ == '__main__':
    lp = define_log_posterior()
    run_inference(lp)
