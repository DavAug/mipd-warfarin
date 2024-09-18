import os
import pickle
import warnings

import myokit
import numpy as np
import pandas as pd
import pints

from model import define_hamberg_model


def generate_individuals(day):
    """
    Returns a numpy array with model parameters of shape
    (n_ids, n_parameters).
    """
    # Import parameters of individual
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/illustration_of_dosing_regimen_distribution_parameters.pickle'
    with open(directory + filename, 'rb') as f:
        psis = pickle.load(f)
        f.close()
    vkorc1 = 2
    _id = 13
    parameters = psis[vkorc1, _id]

    # Broadcast to (n_ids, n_parameters)
    n_ids = 1000
    psis = np.zeros(shape=(n_ids, len(parameters)))
    psis[:] += parameters[np.newaxis, :]

    # Sample EC50 according to posterior
    seed = 4
    df = pd.read_csv(
        directory + '/illustration_bayesian_dosing_regimen_optimisation.csv')
    n_obs = day + 1
    temp = df[df['Number of observations'] == n_obs]
    mean = temp['Mean'].values[0]
    std = temp['Std.'].values[0]
    np.random.seed(seed)
    if day > 1:
        psis[:, 1] = np.random.normal(loc=mean, scale=std, size=n_ids)
    else:
        psis[:, 1] = np.random.lognormal(
            mean=np.log(mean), sigma=std/mean, size=n_ids)

    return psis


def get_regimen(day):
    """
    Returns the latest regimen, before measuring the next INR value.

    The measurements of the INR are taken before warfarin is administered.
    So for the measurement on day 0 no warfarin is administered, on day 1 one
    dose is administered, etc.
    """
    if day == 0:
        return []

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

    return list(doses)


def define_objective_function(parameters, model, init_doses):
    """
    Defines a function that returns the squared distance of the patient's INR
    from 2.5 within the first 19 days of warfarin treatment for a given
    schedule of daily warfarin doses.
    """
    model.set_outputs(['global.inr'])
    objective_function = SquaredINRDistance(
        model, parameters, target=2.5, days=19, res=0.1,
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

def save_results(doses, day, ids):
    """
    Saves dosing regimen to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/illustration_bayesian_dosing_regimen_distribution.csv'
    try:
        data = pd.read_csv(directory + filename)
    except FileNotFoundError:
        data = pd.DataFrame(columns=[
            'Day', 'Sample', 'Dose 1 in mg', 'Dose 2 in mg',
            'Dose 3 in mg', 'Dose 4 in mg', 'Dose 5 in mg', 'Dose 6 in mg',
            'Dose 7 in mg', 'Dose 8 in mg', 'Dose 9 in mg', 'Dose 10 in mg',
            'Dose 11 in mg', 'Dose 12 in mg', 'Dose 13 in mg', 'Dose 14 in mg',
            'Dose 15 in mg', 'Dose 16 in mg', 'Dose 17 in mg', 'Dose 18 in mg',
            'Dose 19 in mg'
            ])

    # Split data into data from day and rest
    mask = data.Day == day
    rest = data[~mask]
    data = data[mask]

    # Add data to file
    if ids in data.Sample.unique():
        mask = data.Sample == ids
        for idd, dose in enumerate(doses):
            data.loc[mask, 'Dose %d in mg' % (idd+1)] = dose
    else:
        df = {
            'Dose %d in mg' % (idd+1): [dose]
            for idd, dose in enumerate(doses)}
        df['Day'] = [day]
        df['Sample'] = [ids]
        data = pd.concat((data, pd.DataFrame(df)), ignore_index=True)

    # Merge dataframes again
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
    m, _ = define_hamberg_model()
    for day in range(days):
        psis = generate_individuals(day)
        for ids, sample in enumerate(psis):
            d = get_regimen(day)
            o = define_objective_function(sample, m, d)
            d += find_dosing_regimen(o)
            save_results(d, day, ids)
