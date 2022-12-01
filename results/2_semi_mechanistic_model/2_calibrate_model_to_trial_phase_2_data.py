import os

import chi
import pandas as pd
import pints

from model import define_hamberg_model, define_hamberg_population_model


def define_log_posterior():
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measurements_df = pd.read_csv(
        directory + '/data/trial_phase_II.csv')

    # Define hierarchical log-posterior
    mechanistic_model,_ = define_hamberg_model(baseline_inr=None)
    mechanistic_model.set_outputs(['myokit.inr'])
    error_model = chi.LogNormalErrorModel()
    population_model = define_hamberg_population_model(
        centered=False, inr=True, conc=False, fixed_y0=False)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(0, 0.5),            # Mean log basline INR
        pints.LogNormalLogPrior(-1, 2),            # Std. log basline INR
        pints.GaussianLogPrior(0, 0.5),            # Mean log shift with A
        pints.GaussianLogPrior(-3.682, 0.028),     # Mean log clearance
        pints.GaussianLogPrior(0.119, 0.020),      # Sigma log clearance
        pints.GaussianLogPrior(0.565, 0.063),      # Rel. shift clearance *2
        pints.BetaLogPrior(30, 6),                 # Rel. shift clearance *3
        pints.GaussianLogPrior(0.00546, 0.0031),   # Rel. shift clearance Age
        pints.GaussianLogPrior(1.41, 0.5),         # Mean log EC50
        pints.LogNormalLogPrior(-1, 0.5),          # Sigma log EC50
        pints.UniformLogPrior(0, 1),               # Rel. shift EC50 VKORC1 A
        pints.GaussianLogPrior(0.2, 0.05),         # Transition rate chain 1
        pints.GaussianLogPrior(0.02, 0.005),       # Transition rate chain 2
        pints.GaussianLogPrior(2.662, 0.020),      # Mean log volume
        pints.LogNormalLogPrior(-2.31, 0.16),      # Sigma log volume
        pints.LogNormalLogPrior(-1, 0.3)           # Sigma log INR
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
    controller.set_transform(pints.ComposedTransformation(
        pints.IdentityTransformation(n_parameters=100 * 4 + 6),
        pints.LogitTransformation(n_parameters=1),
        pints.IdentityTransformation(n_parameters=3),
        pints.LogitTransformation(n_parameters=1),
        pints.IdentityTransformation(n_parameters=5)
    ))

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
        '/posteriors/posterior_trial_phase_II.nc'
    )


if __name__ == '__main__':
    lp = define_log_posterior()
    run_inference(lp)
