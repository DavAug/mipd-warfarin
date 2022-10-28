import os
import pickle
import warnings

import arviz as az
import chi
import myokit
import numpy as np
import pandas as pd
import pints

from model import define_hamberg_model, define_hamberg_population_model


def define_model():
    """
    Returns Hamberg et al's model, as well as the covariates and the parameters
    of the chosen individual.
    """
    # Define model
    model, df = define_hamberg_model()
    model.set_outputs(['myokit.inr'])

    # Import existing file
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/illustration_of_dosing_regimen_distribution_parameters.pickle'
    with open(directory + filename, 'rb') as f:
        psis = pickle.load(f)
        f.close()

    # Choose an individual
    covariates = [0, 70, 2]  # CYP2C9, Age, VKORC1
    _id = 13
    parameters = psis[covariates[2], _id]

    # Fix all parameters but the EC50
    model = chi.ReducedMechanisticModel(model)
    for idn, name in enumerate(model.parameters()):
        if name == 'myokit.half_maximal_effect_concentration':
            continue
        model.fix_parameters({name: parameters[idn]})

    # Get EC50 of individual and measurement noise parameter
    sigma = df[df.Parameter == 'Pooled INR Sigma log'].Value.values[0]
    parameters = [parameters[1], sigma]

    return model, parameters, covariates


def get_prior(cov):
    """
    Returns the prior distribution of the model parameters based on knowledge
    about covariates of the patient.
    """
    # Define population model
    _, df = define_hamberg_model()
    model = define_hamberg_population_model()
    pop_parameters = np.array([
        df[df.Parameter == p].Value.values[0]
        for p in model.get_parameter_names()])

    # Sample prior from population model
    seed = 12
    n_samples = 1000
    prior_samples = model.sample(
        parameters=pop_parameters, covariates=cov, n_samples=n_samples,
        seed=seed)

    # Compute mean and std of prior samples
    mean = np.mean(np.log(prior_samples[:, 1]))
    std = np.std(np.log(prior_samples[:, 1]), ddof=1)
    sigma = np.mean(prior_samples[:, -1])

    return mean, std, sigma


def get_regimen(day):
    """
    Returns the latest regimen, before measuring the next INR value.

    The measurements of the INR are taken before warfarin is administered.
    So for the measurement on day 0 no warfarin is administered, on day 1 one
    dose is administered, etc.
    """
    if day == 0:
        return myokit.Protocol(), []

    # Get regimen from dataframe
    directory = os.path.dirname(os.path.abspath(__file__))
    try:
        df = pd.read_csv(
            directory +
            '/illustration_bayesian_dosing_regimen_optimisation.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            'Invalid dosing regimen. The dosing regimen file from the '
            'previous days cannot be found. This means that either the file '
            './illustration_bayesian_dosing_regimen_optimisation.csv has been '
            'removed or the script was not executed from day 0.')

    # Get doses up to current day
    doses = df[df['Number of observations'] == day][[
        'Dose %d in mg' % (d+1) for d in range(day)]].values[0]

    # Define dosing regimen
    duration = 0.01
    dose_rates = doses / duration
    regimen = myokit.Protocol()
    for idx, dr in enumerate(dose_rates):
        if dr == 0:
            continue
        regimen.add(myokit.ProtocolEvent(
            level=dr,
            start=idx*24,
            duration=duration))

    return regimen, list(doses)


def get_tdm_data(model, error_model, psi, day):
    """
    Samples INR measurements from the individual.
    """
    # Define measurement model
    model = chi.PredictiveModel(
        mechanistic_model=model, error_models=error_model)

    # Define measurement times in hours
    times = np.arange(0, day+1) * 24

    # Sample measurements
    seed = 678
    meas = model.sample(
        parameters=psi, times=times, seed=seed, return_df=False)[0, :, 0]

    return meas, times


def get_posterior(model, error_model, meas, times, pr):
    """
    Returns the posterior distribution inferred from the TDM data.
    """
    # Define log-posterior
    mean, std, sigma = pr
    log_likelihood = chi.LogLikelihood(model, error_model, meas, times)
    log_likelihood.fix_parameters({'Sigma log': sigma})
    prior = pints.LogNormalLogPrior(log_mean=mean, scale=std)
    log_posterior = chi.LogPosterior(log_likelihood, prior)

    # Infer posterior
    warmup = 1000
    n_iterations = 2000
    thinning = 1
    controller = chi.SamplingController(log_posterior)
    controller.set_n_runs(3)
    controller.set_sampler(pints.HaarioBardenetACMC)
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Compute Rhat of chains
    posterior_samples = posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning))
    rhat = az.rhat(posterior_samples)[
        'myokit.half_maximal_effect_concentration'].values

    # Compute mean and std of posterior
    mean = posterior_samples['myokit.half_maximal_effect_concentration'].sel(
        draw=slice(warmup, n_iterations)).mean().values
    std = posterior_samples[
        'myokit.half_maximal_effect_concentration'].sel(
        draw=slice(warmup, n_iterations)).std().values

    return mean, std, rhat


def define_objective_function(model, posterior, init_doses):
    """
    Defines a function that returns the squared distance of the patient's INR
    from 2.5 within the first 19 days of warfarin treatment for a given
    schedule of daily warfarin doses.
    """
    mean_ec50, _, _ = posterior
    objective_function = SquaredINRDistance(
        model, [mean_ec50], target=2.5, days=19, res=0.1,
        init_doses=init_doses)

    return objective_function

def find_dosing_regimen(objective_function):
    """
    Finds the dosing regimen that minimises the squared distance to the target
    INR.
    """
    n_parameters = objective_function.n_parameters()
    controller = pints.OptimisationController(
        objective_function,
        x0=np.ones(n_parameters) * 0.1,
        transformation=pints.LogitTransformation(n_parameters=n_parameters),
        method=pints.CMAES if n_parameters > 1 else pints.NelderMead)
    controller.set_parallel(True)

    p, _ = controller.run()

    # Format doses to daily doses for the first 19 days
    # NOTE: Doses are increased by a factor 30, enabling the logit-transform,
    # which needs to be reversed here.
    doses = list(p * 30)
    doses = doses[:-1] + [doses[-1]] * 13
    doses = [objective_function._convert_to_tablets(d) for d in doses]

    return doses

def save_results(latest_inr, day, posterior, regimen):
    """
    Saves results to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/illustration_bayesian_dosing_regimen_optimisation.csv'
    try:
        data = pd.read_csv(directory + filename)
    except FileNotFoundError:
        data = pd.DataFrame(columns=[
            'ID', 'Number of observations', 'Dose 1 in mg', 'Dose 2 in mg',
            'Dose 3 in mg', 'Dose 4 in mg', 'Dose 5 in mg', 'Dose 6 in mg',
            'Dose 7 in mg', 'Dose 8 in mg', 'Dose 9 in mg', 'Dose 10 in mg',
            'Dose 11 in mg', 'Dose 12 in mg', 'Dose 13 in mg', 'Dose 14 in mg',
            'Dose 15 in mg', 'Dose 16 in mg', 'Dose 17 in mg', 'Dose 18 in mg',
            'Dose 19 in mg', 'Mean', 'Std.', 'Rhat', 'INR'
            ])

    # Add data to file
    n_obs = day + 1
    if n_obs in data['Number of observations'].unique():
        mask = data['Number of observations'] == n_obs
        for idd, dose in enumerate(regimen):
            data.loc[mask, 'Dose %d in mg' % (idd+1)] = dose
        data.loc[mask, 'Mean'] = posterior[0]
        data.loc[mask, 'Std.'] = posterior[1]
        data.loc[mask, 'Rhat'] = posterior[2]
        data.loc[mask, 'INR'] = latest_inr
    else:
        df = {
            'Dose %d in mg' % (idd+1): [dose]
            for idd, dose in enumerate(regimen)}
        df['ID'] = [13]
        df['Number of observations'] = [n_obs]
        df['Mean'] = posterior[0]
        df['Std.'] = posterior[1]
        df['Rhat'] = posterior[2]
        df['INR'] = latest_inr
        data = pd.concat((data, pd.DataFrame(df)), ignore_index=True)

    # Save file
    data.to_csv(directory + filename, index=False)


class SquaredINRDistance(pints.ErrorMeasure):
    """
    Defines a pints.ErrorMeasure based on the squared distance between a
    patient's INR and a target INR.

    :param model: Model of the patient's INR response.
    :param target: Target INR at all times.
    :param days: Number of treatment days during which the squared distance is
        computed.
    :param res: Time steps in which the INR is evaluated against the target
        in days.
    """
    def __init__(
            self, model, parameters, target=2.5, days=19, res=0.1,
            init_doses=None):
        super(SquaredINRDistance, self).__init__()
        if model.n_parameters() != len(parameters):
            raise ValueError('Invalid model or parameters.')
        if init_doses is not None:
            init_doses = [float(d) for d in init_doses]
        else:
            init_doses = []
        if len(init_doses) >= 7:
            # We only optimise the initial 7 doses
            raise ValueError('Invalid init_doses.')
        self._model = model
        self._parameters = parameters
        self._target = target
        self._n_init_doses = len(init_doses)
        self._doses = np.array(
            init_doses + [0] * (days - self._n_init_doses))
        self._duration = 0.01
        self._n_doses = 7 - self._n_init_doses

        # Construct simulation times in hours
        self._times = np.arange(0, days, res) * 24

    def __call__(self, parameters):
        # Simulate INRs for given dosing regimen
        parameters = 30 * np.array(parameters)  # Doses are scaled for optim.
        regimen = self._define_regimen(parameters)
        self._model.set_dosing_regimen(regimen)
        try:
            inrs = self._model.simulate(
                parameters=self._parameters, times=self._times)
        except (myokit.SimulationError, Exception) as e:  # pragma: no cover
            warnings.warn(
                'An error occured while solving the mechanistic model: \n'
                + str(e) + '.\n A score of -infinity is returned.',
                RuntimeWarning)
            return np.infty

        # Compute normalised squared distance to target
        squared_distance = np.mean((inrs - self._target)**2)

        return squared_distance

    def _convert_to_tablets(self, dose):
        """
        The model accepts any dose values, but in practice we can only
        administer tablets.

        Available tablets are:

        - 1 mg
        - 2 mg
        - 2.5 mg
        - 3 mg
        - 4 mg
        - 5 mg
        - 6 mg
        - 7.5 mg
        - 10 mg

        As a result, we define the following conversion:
            1. If dose is < 0.5: 0mg
            2. If dose is >= 0.5 and < 1.5: 1mg
            3. If dose is >= 1.5 and < 2.25: 2mg
            4. If dose is >= 2.25 and < 2.75: 2.5mg
            5. Remaining doses are rounded to next half mg dose.
        """
        if dose < 0.5:
            return 0
        elif dose < 1.5:
            return 1
        elif dose < 2.25:
            return 2
        else:
            return np.round(2 * dose) / 2

    def _define_regimen(self, doses):
        if len(doses) != self._n_doses:
            raise ValueError('Invalid parameters.')
        doses = [self._convert_to_tablets(d) for d in doses]
        self._doses[self._n_init_doses:6] = np.array(doses[:-1])
        self._doses[6:] = doses[-1]
        dose_rates = self._doses / self._duration

        regimen = myokit.Protocol()
        for idx, dr in enumerate(dose_rates):
            if dr == 0:
                continue
            regimen.add(myokit.ProtocolEvent(
                level=dr,
                start=idx*24,
                duration=self._duration))

        return regimen

    def n_parameters(self):
        return self._n_doses


if __name__ == '__main__':
    days = 7
    m, psi, cov = define_model()
    em = chi.LogNormalErrorModel()
    pr = get_prior(cov)
    for day in range(days):
        r, d = get_regimen(day)
        m.set_dosing_regimen(r)
        meas, times = get_tdm_data(m, em, psi, day)
        post = get_posterior(m, em, meas, times, pr)
        o = define_objective_function(m, post, d)
        d += find_dosing_regimen(o)
        save_results(meas[-1], day, post, d)
