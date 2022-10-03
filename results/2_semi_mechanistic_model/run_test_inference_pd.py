import os

import chi
import pandas as pd
import pints

from model import define_hamberg_model, define_hamberg_population_model


def define_log_posterior():
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measurements_df = pd.read_csv(
        directory + '/data/synthetic_hamberg_model_data.csv')

    # Define hierarchical log-posterior
    mechanistic_model,_ = define_hamberg_model()
    error_models = [chi.LogNormalErrorModel(), chi.LogNormalErrorModel()]
    population_model = define_hamberg_population_model(centered=False)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(-3, 0.1),       # Mean log clearance
        pints.LogNormalLogPrior(-1, 0.3),      # Sigma log clearance
        pints.UniformLogPrior(0, 1),           # Rel. shift clearance CYP29P *2
        pints.UniformLogPrior(0, 1),           # Rel. shift clearance CYP29P *3
        pints.GaussianLogPrior(0, 0.01),       # Rel. shift clearance Age
        pints.GaussianLogPrior(1.41, 0.5),     # Mean log EC50
        pints.LogNormalLogPrior(0.1, 0.3),     # Sigma log EC50
        pints.UniformLogPrior(0, 1),           # Rel. shift EC50 VKORC1 A
        pints.GaussianLogPrior(0.2, 0.05),     # Pooled rate chain 1
        pints.GaussianLogPrior(0.02, 0.005),   # Pooled rate chain 2
        pints.GaussianLogPrior(2.7, 0.1),      # Mean log volume
        pints.LogNormalLogPrior(-1, 0.3),      # Sigma log volume
        pints.LogNormalLogPrior(-1, 0.3),      # Sigma log drug conc.
        pints.LogNormalLogPrior(-1, 0.3)       # Sigma log INR
    )
    problem = chi.ProblemModellingController(mechanistic_model, error_models)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df)
    problem.set_log_prior(log_prior)

    return problem.get_log_posterior()


def run_inference(log_posterior):
    seed = 2
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(True)
    controller.set_sampler(pints.NoUTurnMCMC)
    controller.set_transform(pints.ComposedTransformation(
        pints.IdentityTransformation(n_parameters=100 * 3 + 2),
        pints.LogitTransformation(n_parameters=2),
        pints.IdentityTransformation(n_parameters=3),
        pints.LogitTransformation(n_parameters=1),
        pints.IdentityTransformation(n_parameters=6)
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
        '/posteriors/0_test_inference.nc'
    )


if __name__ == '__main__':
    lp = define_log_posterior()
    run_inference(lp)