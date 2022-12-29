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
    )
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df, output_observable_dict={
        'myokit.inr': 'INR'})
    problem.fix_parameters({
        'Log mean myokit.baseline_inr': 0.265,
        'Log std. myokit.baseline_inr': 0.172,
        'Rel. baseline INR A': 1.251,
        'Pooled Sigma log': 0.185
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
