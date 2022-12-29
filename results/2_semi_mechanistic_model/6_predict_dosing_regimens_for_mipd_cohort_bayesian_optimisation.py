import os
import subprocess
import warnings

import arviz as az
import chi
import myokit
import numpy as np
import pandas as pd
import pints

from model import define_hamberg_model


def get_priors(filename):
    """
    Returns a dataframe with the means and standard deviations of the priors.
    """
    # Load priors
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(
        directory + filename)

    # Only keep prior (i.e. distributions after 0 observations)
    n_obs = 0
    data = data[data['Number of observations'] == n_obs]

    return data


def generate_measurements(day, filename):
    """
    Runs the Wajima-Hartmann's QSP model to generate TDM data for the MIPD
    cohort for that day.
    """
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_obs = day + 1

    print('Generating TDM data for day: ', day)
    subprocess.Popen([
        'python',
        directory +
        '/1_systems_pharmacology_model/7_simulate_tdm_data_for_mipd_trial.py',
        '--number',
        str(n_obs),
        '--filename',
        filename
    ]).wait()
    print('TDM data generated')


def get_tdm_data(day, ids, filename):
    """
    Loads TDM data up to the day from file for all individuals in MIPD cohort.
    """
    # Import file with dosing regimens
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df = pd.read_csv(directory + filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Invalid day. The TDM file from the '
            'previous days cannot be found. This means that either the file '
            './mipd_trial_predicted_dosing_regimens.csv '
            'has been '
            'removed or the script was not executed from day 0.')

    # Get data
    n_obs = day + 1
    times = np.arange(0, n_obs) * 24
    measurements = np.zeros(shape=(len(ids), n_obs))
    for idx, _id in enumerate(ids):
        temp = df[df.ID == _id]
        for n in range(1, n_obs+1):
            measurements[idx, n-1] = temp[
                temp['Number of observations'] == n]['INR'].values

    return measurements, times


def get_regimen(day, patient_id, filename):
    """
    Returns the latest regimen, before measuring the next INR value.

    The measurements of the INR are taken before warfarin is administered.
    So for the measurement on day 0 no warfarin is administered, on day 1 one
    dose is administered, etc.
    """
    if day == 0:
        # No dose has been administered when the first measurement was taken.
        return myokit.Protocol(), []

    # Get regimen from dataframe
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(directory + filename)

    # Filter dataframe for individual
    df = df[df.ID == patient_id]

    # Get doses up to current day
    # NOTE: For inference of posterior, we use regimen applied when
    # measurements were collected. Because we measure before administering
    # the next dose, we administer the regimen with n_obs = day measurements
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


def get_posterior(patient_id, model, error_model, meas, times, df_prior):
    """
    Returns the posterior distribution inferred from the TDM data.
    """
    # Define prior
    means = []
    stds = []
    df_prior = df_prior[df_prior.ID == patient_id]
    parameters = list(model.parameters()) + ['Sigma log']
    for parameter in parameters:
        mask = df_prior.Parameter == parameter
        means += [df_prior[mask]['Mean'].values]
        stds += [df_prior[mask]['Std.'].values]
    log_prior = pints.ComposedLogPrior(*[
        pints.GaussianLogPrior(mean=m, sd=stds[idp])
        for idp, m in enumerate(means)])

    # Define log-posterior
    model.set_outputs(['myokit.inr'])
    log_likelihood = chi.LogLikelihood(model, error_model, meas, times)
    log_posterior = chi.LogPosterior(log_likelihood, log_prior)

    # Infer posterior
    warmup = 5000
    n_iterations = 10000
    thinning = 1
    controller = chi.SamplingController(log_posterior, seed=1)
    controller.set_n_runs(3)
    controller.set_sampler(pints.HaarioBardenetACMC)
    controller._initial_params[:, -1] = 0.2  # Avoids infinities
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Compute Rhat of chains
    posterior_samples = posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning))
    rhats = np.array([
        az.rhat(posterior_samples)[p].values for p in parameters])

    # Compute mean and std of posterior
    means = np.array([
        posterior_samples[p].mean().values for p in parameters])
    stds = np.array([
        posterior_samples[p].std().values for p in parameters])

    return means, stds, rhats, parameters


def define_objective_function(model, posterior, init_doses, window):
    """
    Defines a function that returns the squared distance of the patient's INR
    from 2.5 within the first 19 days of warfarin treatment for a given
    schedule of daily warfarin doses.
    """
    means, _, _, _ = posterior
    model.set_outputs(['myokit.inr'])
    objective_function = SquaredINRDistance(
        model, means[:-1], init_doses=init_doses, window=window)

    return objective_function


def find_dosing_regimen(objective_function, day, window):
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
    doses = doses[:-1] + [doses[-1]] * (19 - day - window - 1)
    doses = [objective_function._convert_to_tablets(d) for d in doses]

    return doses


def save_posterior(posterior, patient_id, day, filename):
    """
    Saves posterior to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + filename)

    # Split data into n_obs = n and rest
    n_obs = day + 1
    mask = data['Number of observations'] != n_obs
    rest = data[mask]
    data = data[~mask]

    # Add data to file
    means, stds, rhats, names = posterior
    if patient_id in data.ID.unique():
        mask1 = data.ID == patient_id
        for idn, name in enumerate(names):
            mask2 = data.Parameter == name
            data.loc[mask1 & mask2, 'Mean'] = means[idn]
            data.loc[mask1 & mask2, 'Std.'] = stds[idn]
            data.loc[mask1 & mask2, 'Rhat'] = rhats[idn]
    else:
        df = {
            'ID': [patient_id] * len(names),
            'Number of observations': [n_obs] * len(names),
            'Parameter': names,
            'Mean': means,
            'Std.': stds,
            'Rhat': rhats
        }
        data = pd.concat((data, pd.DataFrame(df)), ignore_index=True)

    # Merge new data with rest
    data = pd.concat((rest, data), ignore_index=True)

    # Save file
    data.to_csv(directory + filename, index=False)


def save_regimen(doses, patient_id, day, filename):
    """
    Saves dosing regimen to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + filename)

    # Split data into n_obs = n and rest
    n_obs = day + 1
    mask = data['Number of observations'] != n_obs
    rest = data[mask]
    data = data[~mask]

    # Add data to file
    if patient_id in data.ID.unique():
        mask = data.ID == patient_id
        for idd, dose in enumerate(doses):
            data.loc[mask, 'Dose %d in mg' % (idd+1)] = dose
    else:
        df = {
            'Dose %d in mg' % (idd+1): [dose]
            for idd, dose in enumerate(doses)}
        df['ID'] = [patient_id]
        df['Number of observations'] = [n_obs]
        data = pd.concat((data, pd.DataFrame(df)), ignore_index=True)

    # Merge new data with rest
    data = pd.concat((rest, data), ignore_index=True)

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
            self, model, parameters, target=2.5, days=50, res=0.1,
            init_doses=None, window=7):
        super(SquaredINRDistance, self).__init__()
        if model.n_parameters() != len(parameters):
            raise ValueError('Invalid model or parameters.')
        if init_doses is not None:
            init_doses = [float(d) for d in init_doses]
        else:
            init_doses = []
        if len(init_doses) >= 19:
            # We only optimise 19 trial days
            raise ValueError('Invalid init_doses.')
        self._model = model
        self._parameters = parameters
        self._target = target
        self._n_init_doses = len(init_doses)
        self._doses = np.array(
            init_doses + [0] * (days - self._n_init_doses))
        self._duration = 0.01
        free_doses = 19 - self._n_init_doses
        self._n_doses = window if free_doses > window else free_doses

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
        split = self._n_init_doses + self._n_doses - 1
        self._doses[self._n_init_doses:split] = np.array(doses[:-1])
        self._doses[split:] = doses[-1]
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
    days = 19
    window = 7
    f_meas = \
        '/2_semi_mechanistic_model' \
        + '/mipd_trial_predicted_dosing_regimens.csv'
    f_post = '/2_semi_mechanistic_model' \
        + '/mipd_trial_prior_distributions.csv'

    m, _ = define_hamberg_model(baseline_inr=None)
    em = chi.LogNormalErrorModel()
    df_pr = get_priors(f_post)
    ids = df_pr.ID.unique()
    for day in range(days):
        generate_measurements(day, f_meas)
        meas, times = get_tdm_data(day, ids, f_meas)
        for idx, _id in enumerate(ids):
            r, d = get_regimen(day, _id, f_meas)
            m.set_dosing_regimen(r)
            post = get_posterior(_id, m, em, meas[idx], times, df_pr)
            o = define_objective_function(m, post, d, window)
            d += find_dosing_regimen(o, day, window)
            save_posterior(post, _id, day, f_post)
            save_regimen(d, _id, day, f_meas)
