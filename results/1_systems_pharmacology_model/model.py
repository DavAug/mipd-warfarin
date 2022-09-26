import os

import chi
import numpy as np
import pandas as pd
from scipy import interpolate


def define_wajima_model(patient=True, inr_test=False):
    """
    Returns Wajima's QSP model of the coagulation network.

    The model can be returned in 3 variants: 1. in its `plain' form, modelling
    only the treatment response of the patient, ``patient=True`` and
    ``inr_test=False``; 2. in its INR test form, only modelling the INR test
    for a given blood sample, ``patient=False`` and ``inr_test=True``; and 3.
    in a combined form, where it models the treatment response of a patient
    over time and simultaneously performs the INR test on blood samples,
    ``patient=True`` and ``inr_test=True``.

    Reference
    ---------
    .. Wajima T, Isbister GK, Duffull SB. A comprehensive model for the humoral
        coagulation network in humans. Clin Pharmacol Ther. 2009
        Sep;86(3):290-8.

    :returns: Wajima model, typical parameter values
    :rtype: Tuple[chi.MechanisticModel, pandas.DataFrame]
    """
    directory = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if patient and not inr_test:
        model_file = '/models/wajima_coagulation_model.xml'
        model = chi.PKPDModel(directory + model_file)
        model.set_administration(
            compartment='central_warfarin', amount_var='warfarin_amount',
            direct=False)
    elif not patient and inr_test:
        model_file = '/models/wajima_inr_test_model.xml'
        model = chi.SBMLModel(directory + model_file)
        model.set_outputs(['central.fibrin_integral_concentration'])
        model = chi.ReducedMechanisticModel(model)
        model.fix_parameters({'central.fibrin_integral_amount': 0})
    elif patient and inr_test:
        model = WajimaWarfarinINRResponseModel()
    else:
        raise ValueError(
            'Invalid patient and inr_test. At least one input has to be True.')

    parameter_file = '/models/wajima_coagulation_model_parameters.csv'
    parameters = pd.read_csv(directory + parameter_file)

    return (model, parameters)


# TODO:
class WajimaWarfarinINRResponseModel(chi.MechanisticModel):
    """
    Defines Wajima's model of the INR response to warfarin treatment.

    This model is composed of two models: 1. A model that models the patient;
    and 2. a model that models the INR test. The first model is Wajima's model
    of the blood coagulation network. The second model is a reduced version of
    the same model, keeping only the relevant pathways for the INR test. In
    addition, the second model simulates the AUC of fibrin, which is used to
    determine the INR. In particular, the model first simulates the patients
    treatment response using the first model, and takes blood samples at all
    simulation times. These blood samples are then used for the INR test in the
    second model to determine the INR over time.

    If no standard prothrombin time is provided the prothrombin time is not
    normalised.

    .. note:
        For efficiency, we remove the ability to change the mode of
        administration. Warfarin is always administered indirectly through a
        dose compartment to the central compartment.

    Reference
    ---------
    .. Hamberg AK, Wadelius M, Lindh JD, Dahl ML, Padrini R, Deloukas P,
        Rane A, Jonsson EN. A pharmacometric model describing the relationship
        between warfarin dose and INR response with respect to variations in
        CYP2C9, VKORC1, and age. Clin Pharmacol Ther. 2010 Jun;87(6):727-34.

    :param standard_pt: Standard PT time that normalises the INR.
    :type standard_pt: float, optional
    :param inr_test_duration: Time after which prothrombin time is determined.
        If the fibrin integral is below 1500 nMs after this time, the
        prothrombin time is linearly extrapolated.
    :type inr_test_duration: float, optional

    Extends :class:`chi.PharmacokineticModel`.
    """
    def __init__(self, standard_pt=None, inr_test_duration=70):
        # Check inputs
        standard_pt = float(standard_pt) if standard_pt else 1
        if standard_pt <= 0:
            raise ValueError(
                'Invalid standard_pt. Prothrombin times have to '
                'greater than 0.')
        inr_test_duration = float(inr_test_duration)
        if inr_test_duration <= 0:
            raise ValueError(
                'Invalid inr_test_duration. The INR test duration has to be '
                'greater than 0.')

        # Define coagulation network and INR test model
        self._network_model, _ = define_wajima_model(
            patient=True, inr_test=False)
        self._inr_test_model, _ = define_wajima_model(
            patient=False, inr_test=True)

        # Define INR test hyperparameters
        self._fibrin_auc_threshold = 1500  # nMs
        self._fibrin_auc_times = np.linspace(
            start=0, stop=float(inr_test_duration), num=100)
        self._delta_time = \
            self._fibrin_auc_times[1] - self._fibrin_auc_times[0]
        self._standard_pt = standard_pt

        # Prepare masks for INR test
        self._masks = self._prepare_masks()

    def _perform_prothrombin_time_test(self, parameters):
        """
        Returns the prothrombin time of the blood sample.

        The prothrombin time is determined by the time point when the fibrin
        time integral reaches a threshold value (1500 nMs). The duration
        of the test is fixed because myokit has no stopping criterion for
        solving ODEs. If the threshold is reached within this time, the PT is
        determined via interpolation; if the threshold is not reached, the PT
        is determined via naïve linear extrapolation.
        """
        fibrin_auc = self._inr_test_model.simulate(
            parameters=parameters, times=self._fibrin_auc_times)[0]

        if fibrin_auc[-1] < self._fibrin_auc_threshold:
            # We are not integrating long enough, so let's linearly
            # extrapolate.
            slope = (fibrin_auc[-1] - fibrin_auc[-2]) / self._delta_time
            return self._fibrin_auc_times[-1] + (
                self._fibrin_auc_threshold - fibrin_auc[-1]) / slope

        # Interpolate prothrombin time.
        time_of_auc = interpolate.interp1d(fibrin_auc, self._fibrin_auc_times)
        return time_of_auc(self._fibrin_auc_threshold)

    def _prepare_masks(self):
        """
        Prepares convenience mask for:

        1. Subsampling the network model parameters to get the parameters of
            the INR test model
        2. Getting the states of the network model
        3. Getting the volume of the central compartment and the tissue factor
            amount
        4. All time related variables of the network model
        """
        # 1. subsampling mask
        subsampling_mask = np.zeros(
            self._inr_test_model.n_parameters(), dtype=int)
        network_model_parameters = np.array(self._network_model.parameters())
        for idp, parameter in enumerate(self._inr_test_model.parameters()):
            index = np.where(network_model_parameters == parameter)[0][0]
            subsampling_mask[idp] = index

        # 2. States of the network model
        n_states = self._network_model._n_states

        # 3. Volume and tissue factor mask
        tf_indices = np.zeros(2, dtype=int)
        tf_indices[0] = np.where(
            network_model_parameters == 'central.size')[0][0]
        tf_indices[1] = np.where(
            network_model_parameters == 'central.tissue_factor_amount')[0][0]

        # 4. Time related variables
        rate_indices = []
        for idp, name in enumerate(network_model_parameters):
            if 'rate' in name:
                rate_indices.append(idp)

        return subsampling_mask, n_states, tf_indices, rate_indices

    def compute_prothrombin_time(
            self, blood_sample, parameters, set_standard_pt=False):
        """
        Returns the blood sample's prothrombin time in seconds.

        If ``set_standard_pt`` is set to ``True`` the prothrombin
        time of the blood sample is set as the standard prothrombin time for
        the INR test.

        The prothrombin time is computed by diluting the sample 1:2, setting
        the thrombomodulin concentration to zero, adding 300 nM of tissue
        factor and determining the time when the fibrin integral over time
        reaches a threshold of 1500 nMs.

        :param blood_sample: The states of the WajimaWarfarinINRResponseModel
            which determine the blood sample.
        :type blood_sample: numpy.ndarray of shape (n_states,)
        :param parameters: The input parameters of the
            WajimaWarfarinINRResponseModel associated to the blood sample.
        :type parameters: numpy.ndarray of shape (n_parameters,)
        :param set_standard_pt: A boolean flag which indicates
            whether the prothrombin time of the sample should be set as the
            standard prothrombin time for future INR tests.
        :type set_standard_pt: bool, optional
        """
        # Test parameters
        test_parameters = np.empty(len(parameters))

        # Dilute blood samples 1:2
        subsampling_mask, n_states, tf_indices, rate_indices = self._masks
        test_parameters[:n_states] = blood_sample / 3

        # Insert the rest of the model parameters
        test_parameters[n_states:] = parameters[n_states:]

        # Probe samples with 300nM tissue factor
        test_parameters[tf_indices[1]] = 300 * parameters[tf_indices[0]]

        # Change time units from h to seconds
        test_parameters[rate_indices] /= 3600

        # Perform INR test with blood samples
        # (parameters need to be masked, because the INR test model is a subset
        # of the coagulation network model)
        pt = self._perform_prothrombin_time_test(
            parameters=test_parameters[subsampling_mask])

        if set_standard_pt and (pt > 0):
            # PT > 0 is really just a safety check, so INR tests don't break
            # if for some reason a bad PT is returned.
            self._standard_pt = pt

        return pt

    def dosing_regimen(self):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`myokit.Protocol`. If the protocol has not been set, ``None`` is
        returned.
        """
        return self._network_model.dosing_regimen()

    def get_standard_prothrombin_time(self):
        """
        Returns the standard prothrombin time that is used to normalise the INR
        values.
        """
        return self._standard_pt

    def get_test_duration(self):
        """
        Return the INR test duration in seconds.
        """
        return self._fibrin_auc_times[-1]

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the INR.
        """
        return self._inr_test_model.n_outputs()

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._network_model.n_parameters()

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return ['INR']

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return self._network_model.parameters()

    def set_dosing_regimen(
            self, dose, start=0, duration=0.01, period=None, num=None):
        """
        Sets the dosing regimen with which the compound is administered.

        The route of administration can be set with :meth:`set_administration`.
        However, the type of administration, e.g. bolus injection or infusion,
        may be controlled with the duration input.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model an infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the dose is administered once. To apply multiple doses
        provide a dose administration period.

        Parameters
        ----------
        dose
            The amount of the compound that is injected at each administration.
        start
            Start time of the treatment.
        duration
            Duration of dose administration. For a bolus injection, a dose
            duration of 1% of the time unit should suffice. By default the
            duration is set to 0.01 (bolus).
        period
            Periodicity at which doses are administered. If ``None`` the dose
            is administered only once.
        num
            Number of administered doses. If ``None`` and the periodicity of
            the administration is not ``None``, doses are administered
            indefinitely.
        """
        self._network_model.set_dosing_regimen(
            dose, start, duration, period, num)

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        :param names: A dictionary that maps the current parameter names to new
            names.
        :type names: dict[str, str]
        """
        self._network_model.set_parameter_names(names)

    def set_test_duration(self, duration):
        """
        Sets the duration of the INR test.

        :param duration: Duration of test in seconds.
        :type duration: float.
        """
        duration = float(duration)
        if duration <= 0:
            raise ValueError(
                'Invalid duration. The INR test duration has to be '
                'greater than 0.')

        # Define INR test hyperparameters
        self._fibrin_auc_threshold = 1500  # nMs
        self._fibrin_auc_times = np.linspace(
            start=0, stop=float(duration), num=100)
        self._delta_time = \
            self._fibrin_auc_times[1] - self._fibrin_auc_times[0]

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs (and optionally
        the sensitivites) for the specified parameters and times.

        The model outputs are returned as a 2 dimensional NumPy array of shape
        (n_outputs, n_times). If sensitivities are enabled, a tuple is returned
        with the NumPy array of the model outputs and a NumPy array of the
        sensitivities of shape (n_times, n_outputs, n_parameters).

        .. note:
            Calling this method `locks' the dosing regimen, such that it cannot
            be changed. This is done to speed up the inference. To unlock the
            dosing regimen execute the ``lock_regimen(locked=False)`` method
            of the simulator proptery, i.e.
            ``model.simulator.lock_regimen(locked=False)``.

        :param parameters: An array-like object with values for the model
            parameters.
        :type parameters: list, numpy.ndarray
        :param times: An array-like object with time points at which the output
            values are returned.
        :type times: list, numpy.ndarray
        """
        # Simulate the coagulation network, i.e. the patient
        blood_samples = self._network_model.simulate(
            parameters=parameters, times=times)

        ## Perform INR test
        # Dilute blood samples 1:2
        subsampling_mask, n_states, tf_indices, rate_indices = self._masks
        blood_samples /= 3

        # Probe samples with 300nM tissue factor
        volume = parameters[tf_indices[0]]
        blood_samples[tf_indices[1]] = 300 * volume

        # Change time units from h to seconds
        test_parameters = np.empty(len(parameters))
        test_parameters[:] = parameters
        test_parameters[rate_indices] /= 3600

        # Perform INR test with blood samples
        pt = np.empty(shape=(1, len(times)))
        for ids, blood_sample in enumerate(blood_samples.T):
            test_parameters[:n_states] = blood_sample
            pt[0, ids] = self._perform_prothrombin_time_test(
                parameters=test_parameters[subsampling_mask])

        return pt / self._standard_pt

    def time_unit(self):
        """
        Returns the model's unit of time.
        """
        return self._network_model.time_unit()
