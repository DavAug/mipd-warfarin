import os

import chi
import numpy as np
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
    # NOTE: We reduce uncertainty in baseline INR parameters during inference
    # by a factor sqrt(10) balance steady state versus dynamics. We only
    # measure 160 individuals over time, but measure 1000 indviduals at steady
    # state.
    # This leads to overfitting the steady state, at the cost of fitting the
    # early dynamics.
    mechanistic_model,_ = define_steady_state_hamberg_model()
    error_model = chi.LogNormalErrorModel()
    population_model = define_steady_state_hamberg_population_model(
        centered=False)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(0.348, 0.033),      # Mean log y0 G
        pints.GaussianLogPrior(0.187, 0.00015),    # Std. log y0
        pints.GaussianLogPrior(1.772, 0.11),       # y0 A / y0 G
        pints.GaussianLogPrior(-3.670, 0.029),     # Mean log ke
        pints.GaussianLogPrior(0.105, 0.02),       # Sigma log ke
        pints.GaussianLogPrior(0.535, 0.052),      # Rel. shift ke CYP29P *2
        pints.GaussianLogPrior(0.74, 0.056),       # Rel. shift ke CYP29P *3
        pints.GaussianLogPrior(0.0063, 0.0028),    # Rel. shift ke Age
        pints.GaussianLogPrior(0.832, 0.17),       # Mean log EC50
        pints.GaussianLogPrior(0.14, 0.041),       # Sigma log EC50
        pints.BetaLogPrior(4, 10),                 # Rel. shift EC50 VKORC1 A
        pints.GaussianLogPrior(2.66, 0.023),       # Mean log volume
        pints.GaussianLogPrior(0.0958, 0.015),     # Sigma log volume
        pints.GaussianLogPrior(0.153, 0.0052)      # Sigma log INR
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
