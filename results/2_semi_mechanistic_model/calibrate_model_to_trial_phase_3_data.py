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

    # Define hierarchical log-posterior
    mechanistic_model,_ = define_steady_state_hamberg_model()
    error_model = chi.LogNormalErrorModel()
    population_model = define_steady_state_hamberg_population_model(
        centered=False)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(-3.629, 0.024),     # Mean log clearance
        pints.GaussianLogPrior(0.096, 0.018),      # Sigma log clearance
        pints.GaussianLogPrior(0.524, 0.042),      # Rel. shift CYP29P *2
        pints.GaussianLogPrior(0.709, 0.056),      # Rel. shift CYP29P *3
        pints.GaussianLogPrior(0.00173, 0.00274),  # Rel. shift Age
        pints.GaussianLogPrior(1.286, 0.051),      # Mean log EC50
        pints.GaussianLogPrior(0.308, 0.028),      # Sigma log EC50
        pints.GaussianLogPrior(0.821, 0.018),      # Rel. shift EC50 VKORC1 A
        pints.GaussianLogPrior(2.625, 0.020),      # Mean log volume
        pints.GaussianLogPrior(0.098, 0.015),      # Sigma log volume
        pints.GaussianLogPrior(0.113, 0.004)       # Sigma log INR
    )
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df, output_observable_dict={
        'myokit.inr': 'INR'})
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
