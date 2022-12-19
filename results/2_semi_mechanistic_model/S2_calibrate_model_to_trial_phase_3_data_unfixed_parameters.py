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
        pints.GaussianLogPrior(0.265, 0.032),      # Mean log baseline INR G
        pints.GaussianLogPrior(0.172, 0.016),      # Std. log baseline INR
        pints.GaussianLogPrior(1.251, 0.07),       # Mean log baseline INR A
        pints.GaussianLogPrior(-3.658, 0.029),     # Mean log k_e
        pints.GaussianLogPrior(0.096, 0.02),       # Sigma log k_e
        pints.GaussianLogPrior(0.479, 0.05),       # Rel. shift k_e CYP29P *2
        pints.GaussianLogPrior(0.74, 0.056),       # Rel. shift k_e CYP29P *3
        pints.GaussianLogPrior(0.0063, 0.0028),    # Rel. shift k_e Age
        pints.GaussianLogPrior(0.755, 0.14),       # Mean log EC50
        pints.LogNormalLogPrior(-2.117, 0.31),     # Sigma log EC50
        pints.GaussianLogPrior(0.563, 0.066),      # Rel. shift EC50 VKORC1 A
        pints.GaussianLogPrior(2.66, 0.021),       # Mean log volume
        pints.GaussianLogPrior(0.0949, 0.015),     # Sigma log volume
        pints.GaussianLogPrior(0.185, 0.0052)      # Sigma log INR
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
        '/posteriors/posterior_trial_phase_III_unfixed.nc'
    )


if __name__ == '__main__':
    lp = define_log_posterior()
    run_inference(lp)
