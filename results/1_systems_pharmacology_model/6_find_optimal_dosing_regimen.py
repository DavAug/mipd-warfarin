import enum
import os
import warnings

import chi
import myokit
import numpy as np
import pandas as pd
import pints

from model import define_wajima_model


def define_model():
    """
    Defines Wajima's model of the coagulation network.
    """
    model, _ = define_wajima_model(patient=True, inr_test=True)
    model = chi.ReducedMechanisticModel(model)

    return model

def define_objective_function(patient, model):
    """
    Defines a function that returns the squared distance of the patient's INR
    from 2.5 within the first 19 days of warfarin treatment for a given
    schedule of daily warfarin doses.
    """
    model.fix_parameters({n: patient[n] for n in model.parameters()})
    obective_function = SquaredINRDistance(model, target=2.5, days=19, res=0.1)

    return obective_function

def find_dosing_regimen(objective_function):
    """
    Finds the dosing regimen that minimises the squared distance to the target
    INR.
    """
    n_parameters = objective_function.n_parameters()
    controller = pints.OptimisationController(
        objective_function,
        x0=np.ones(n_parameters),
        transformation=pints.LogTransformation(n_parameters=n_parameters),
        method=pints.CMAES)
    controller.set_parallel(True)

    p, _ = controller.run()

    # Format doses to daily doses for the first 19 days
    doses = list(p)
    doses = doses[:6] + [doses[6]] * 13

    return doses

def save_results(patient, regimen):
    """
    Saves dosing regimen to a csv file.
    """
    # Import existing file
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/mipd_trial_optimal_dosing_regimens.csv'
    try:
        data = pd.read_csv(directory + filename)
    except FileNotFoundError:
        data = pd.DataFrame(columns=[
            'ID', 'Dose 1 in mg', 'Dose 2 in mg', 'Dose 3 in mg',
            'Dose 4 in mg', 'Dose 5 in mg', 'Dose 6 in mg', 'Dose 7 in mg',
            'Dose 8 in mg', 'Dose 9 in mg', 'Dose 10 in mg', 'Dose 11 in mg',
            'Dose 12 in mg', 'Dose 13 in mg', 'Dose 14 in mg', 'Dose 15 in mg',
            'Dose 16 in mg', 'Dose 17 in mg', 'Dose 18 in mg', 'Dose 19 in mg'
            ])

    # Add data to file
    if patient['ID'] in data.ID.unique():
        for idd, dose in enumerate(regimen):
            data['Dose %d in mg' % (idd+1)] = dose
    else:
        df = {
            'Dose %d in mg' % (idd+1): [dose]
            for idd, dose in enumerate(regimen)}
        df['ID'] = [patient['ID']]
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
    def __init__(self, model, target=2.5, days=19, res=0.1):
        super(SquaredINRDistance, self).__init__()
        self._model = model
        self._target = target
        self._doses = [0] * days
        self._duration = 0.01
        self._n_doses = 7  # After one week the maintenance dose is used

        # Construct simulation times in hours
        # NOTE: 100 days calibration, so patient can reach equilibrium,
        # before start of treatment
        self._cal_time = 100 * 24
        self._times = self._cal_time + np.arange(0, days, res) * 24

    def __call__(self, parameters):
        # Simulate INRs for given dosing regimen
        regimen = self._define_regimen(parameters)
        self._model.set_dosing_regimen(regimen)
        try:
            inrs = self._model.simulate(parameters=[], times=self._times)
        except (myokit.SimulationError, Exception) as e:  # pragma: no cover
            warnings.warn(
                'An error occured while solving the mechanistic model: \n'
                + str(e) + '.\n A score of -infinity is returned.',
                RuntimeWarning)
            return np.infty

        # Compute normalised squared distance to target
        squared_distance = np.mean((inrs - self._target)**2)

        # Round squared distance to 3 digits for faster convergence of
        # optimisation
        squared_distance = np.round(squared_distance, decimals=3)

        return squared_distance

    def _define_regimen(self, doses):
        if len(doses) != self._n_doses:
            raise ValueError('Invalid parameters.')
        doses = list(doses[:6]) + [doses[6]] * 13
        dose_rates = np.array(doses) / self._duration

        regimen = myokit.Protocol()
        for idx, dr in enumerate(dose_rates):
            regimen.add(myokit.ProtocolEvent(
                level=dr,
                start=self._cal_time+idx*24,
                duration=self._duration))
        self._doses = doses

        return regimen

    def n_parameters(self):
        return self._n_doses


if __name__ == '__main__':
    # Load patient data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + '/data/mipd_trial_cohort.csv')

    # Define model
    model = define_model()

    # Find optimal dosing strategy
    for _, patient in data.iterrows():
        o = define_objective_function(patient, model)
        r = find_dosing_regimen(o)
        save_results(patient, r)
