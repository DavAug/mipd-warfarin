import os
import warnings

import myokit
import numpy as np
import pandas as pd
import pints

from model import define_hamberg_model


def load_priors():
    """
    Returns a dataframe with the means and standard deviations of the priors.
    """
    # Load priors
    directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(directory + '/mipd_trial_prior_distributions.csv')

    # Only keep the priors for zero observations
    # (i.e. the priors only based on the covariates)
    data = data[data['Number of observations'] == 0]

    return data


def define_objective_function(patient_id, data, model):
    """
    Defines a function that returns the squared distance of the patient's INR
    from 2.5 within the first 19 days of warfarin treatment for a given
    schedule of daily warfarin doses.
    """
    data = data[data.ID == patient_id]
    parameters = [
        data[data.Parameter == n].Mean.values[0] for n in model.parameters()]
    model.set_outputs(['myokit.inr'])
    objective_function = SquaredINRDistance(
        model, parameters, target=2.5, days=19, res=0.1)

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
        method=pints.CMAES)
    controller.set_parallel(True)

    p, _ = controller.run()

    # Format doses to daily doses for the first 19 days
    # NOTE: Doses are increased by a factor 30, enabling the logit-transform,
    # which needs to be reversed here.
    doses = list(p * 30)
    doses = doses[:6] + [doses[6]] * 13
    doses = [objective_function._convert_to_tablets(d) for d in doses]

    return doses

def save_results(patient_id, regimen):
    """
    Saves dosing regimen to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/mipd_trial_predicted_dosing_regimens.csv'
    try:
        data = pd.read_csv(directory + filename)
    except FileNotFoundError:
        data = pd.DataFrame(columns=[
            'ID', 'Number of observations', 'Dose 1 in mg', 'Dose 2 in mg',
            'Dose 3 in mg', 'Dose 4 in mg', 'Dose 5 in mg', 'Dose 6 in mg',
            'Dose 7 in mg', 'Dose 8 in mg', 'Dose 9 in mg', 'Dose 10 in mg',
            'Dose 11 in mg', 'Dose 12 in mg', 'Dose 13 in mg', 'Dose 14 in mg',
            'Dose 15 in mg', 'Dose 16 in mg', 'Dose 17 in mg', 'Dose 18 in mg',
            'Dose 19 in mg'
            ])

    # Split data into data with 0 observations and the rest
    mask = data['Number of observations'] != 0
    rest = data[mask]
    data = data[~mask]

    # Add data to file
    if patient_id in data.ID.unique():
        mask = data.ID == patient_id
        for idd, dose in enumerate(regimen):
            data.loc[mask, 'Dose %d in mg' % (idd+1)] = dose
    else:
        df = {
            'Dose %d in mg' % (idd+1): [dose]
            for idd, dose in enumerate(regimen)}
        df['ID'] = [patient_id]
        df['Number of observations'] = [0]
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
    def __init__(self, model, parameters, target=2.5, days=19, res=0.1):
        super(SquaredINRDistance, self).__init__()
        if model.n_parameters() != len(parameters):
            raise ValueError('Invalid model or parameters.')
        self._model = model
        self._parameters = parameters
        self._target = target
        self._doses = [0] * days
        self._duration = 0.01
        self._n_doses = 7  # After one week the maintenance dose is used

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
        doses = doses[:6] + [doses[6]] * (len(self._doses)-6)
        dose_rates = np.array(doses) / self._duration

        regimen = myokit.Protocol()
        for idx, dr in enumerate(dose_rates):
            if dr == 0:
                continue
            regimen.add(myokit.ProtocolEvent(
                level=dr,
                start=idx*24,
                duration=self._duration))
        self._doses = doses

        return regimen

    def n_parameters(self):
        return self._n_doses


if __name__ == '__main__':
    data = load_priors()
    m, _ = define_hamberg_model(baseline_inr=None)
    ids = data.ID.dropna().unique()
    for _id in ids:
        o = define_objective_function(_id, data, m)
        r = find_dosing_regimen(o)
        save_results(_id, r)
