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


def define_hartmann_population_model():
    """
    Defines Hartmann et al's population model for Wajima et al's coagulation
    network model.

    The model assumes that the production rates associated with
    coagulation factors II, V, VII, IX, X, XI, XII and XIII as well as
    the production rates of PC and PS display normally distributed IIV.
    TODO: Covariates.
    The remaining parameters are assumed to display no IIV.
    """
    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.PooledModel(n_dim=190),
        chi.GaussianModel(n_dim=3),
        chi.PooledModel(n_dim=2),
        chi.GaussianModel(n_dim=1),
        chi.PooledModel(n_dim=2),
        chi.GaussianModel(n_dim=2),
        chi.PooledModel(n_dim=1),
        chi.GaussianModel(n_dim=4),
        chi.PooledModel(n_dim=4)
    ])
    population_model.set_dim_names([
        'central.activated_coagulation_factor_ii_amount',
        'central.activated_coagulation_factor_ix_amount',
        'central.activated_coagulation_factor_v_amount',
        'central.activated_coagulation_factor_vii_amount',
        'central.activated_coagulation_factor_viii_amount',
        'central.activated_coagulation_factor_x_amount',
        'central.activated_coagulation_factor_xi_amount',
        'central.activated_coagulation_factor_xii_amount',
        'central.activated_coagulation_factor_xiii_amount',
        'central.activated_protein_c_amount',
        'central.aii_at_complex_amount',
        'central.aii_atiii_heparin_complex_amount',
        'central.aii_tmod_complex_amount',
        'central.aix_atiii_heparin_complex_amount',
        'central.aix_aviii_complex_amount',
        'central.apc_ps_complex_amount',
        'central.atiii_heparin_complex_amount',
        'central.av_ax_complex_amount',
        'central.avii_tf_ax_tfpi_complex_amount',
        'central.avii_tf_complex_amount',
        'central.ax_atiii_heparin_complex_complex_amount',
        'central.ax_tfpi_complex_amount',
        'central.coagulation_factor_ii_amount',
        'central.coagulation_factor_ix_amount',
        'central.coagulation_factor_v_amount',
        'central.coagulation_factor_vii_amount',
        'central.coagulation_factor_viii_amount',
        'central.coagulation_factor_x_amount',
        'central.coagulation_factor_xi_amount',
        'central.coagulation_factor_xii_amount',
        'central.coagulation_factor_xiii_amount',
        'central.contact_system_activator_amount',
        'central.cross_linked_fibrin_amount',
        'central.d_dimer_amount',
        'central.fibrin_amount',
        'central.fibrin_degradation_product_amount',
        'central.fibrinogen_amount',
        'central.kallikrein_amount',
        'central.plasmin_amount',
        'central.plasminogen_amount',
        'central.prekallikrein_amount',
        'central.protein_c_amount',
        'central.protein_s_amount',
        'central.thrombomodulin_amount',
        'central.tissue_factor_amount',
        'central.tissue_factor_pathway_inhibitor_amount',
        'central.vii_tf_complex_amount',
        'central.vitamin_k_amount',
        'central.vitamin_k_epoxide_amount',
        'central.vitamin_k_hydroquinone_amount',
        'central_warfarin.warfarin_amount',
        'dose.drug_amount',
        'peripheral_vitamin_k.vitamin_k_peripheral_amount',
        'central.size',
        'central_warfarin.size',
        'dose.absorption_rate',
        'myokit.complex_formation_rate_aii_atiii_heparin_complex',
        'myokit.complex_formation_rate_aii_tmod',
        'myokit.complex_formation_rate_aix_atiii_heparin_complex',
        'myokit.complex_formation_rate_aix_aviii',
        'myokit.complex_formation_rate_apc_ps',
        'myokit.complex_formation_rate_av_ax',
        'myokit.complex_formation_rate_avii_tf',
        'myokit.complex_formation_rate_avii_tf_ax_tfpi',
        'myokit.complex_formation_rate_ax_atiii_heparin_complex',
        'myokit.complex_formation_rate_ax_tfpi',
        'myokit.complex_formation_rate_vii_tf',
        'myokit.conversion_rate_vk_vkh2',
        'myokit.conversion_rate_vkh2_vko',
        'myokit.conversion_rate_vko_vk',
        'myokit.degradation_rate_aii_tmod_complex',
        'myokit.degradation_rate_aix_aviii_complex',
        'myokit.degradation_rate_apc_ps_complex',
        'myokit.degradation_rate_av_ax_complex',
        'myokit.degradation_rate_avii_tf_ax_tfpi_complex',
        'myokit.degradation_rate_avii_tf_complex',
        'myokit.degradation_rate_ax_tfpi_complex',
        'myokit.degradation_rate_ca',
        'myokit.degradation_rate_d_dimer',
        'myokit.degradation_rate_fdp',
        'myokit.degradation_rate_fibrin',
        'myokit.degradation_rate_fibrinogen',
        'myokit.degradation_rate_ii',
        'myokit.degradation_rate_ii_activated',
        'myokit.degradation_rate_ix',
        'myokit.degradation_rate_ix_activated',
        'myokit.degradation_rate_kallikrein',
        'myokit.degradation_rate_pc',
        'myokit.degradation_rate_pc_activated',
        'myokit.degradation_rate_plasmin',
        'myokit.degradation_rate_plasminogen',
        'myokit.degradation_rate_prekallikrein',
        'myokit.degradation_rate_ps',
        'myokit.degradation_rate_tat',
        'myokit.degradation_rate_tfpi',
        'myokit.degradation_rate_tissue_factor',
        'myokit.degradation_rate_tmod',
        'myokit.degradation_rate_v',
        'myokit.degradation_rate_v_activated',
        'myokit.degradation_rate_vii',
        'myokit.degradation_rate_vii_activated',
        'myokit.degradation_rate_vii_tf_complex',
        'myokit.degradation_rate_viii',
        'myokit.degradation_rate_viii_activated',
        'myokit.degradation_rate_x',
        'myokit.degradation_rate_x_activated',
        'myokit.degradation_rate_xf',
        'myokit.degradation_rate_xi',
        'myokit.degradation_rate_xi_activated',
        'myokit.degradation_rate_xii',
        'myokit.degradation_rate_xii_activated',
        'myokit.degradation_rate_xiii',
        'myokit.degradation_rate_xiii_activated',
        'myokit.elimination_rate_atiii_heparin_complex',
        'myokit.elimination_rate_vk',
        'myokit.elimination_rate_warfarin',
        'myokit.half_maximal_effect_concentration_aii_tmod_complex_pc',
        'myokit.half_maximal_effect_concentration_aix_aviii_complex_x',
        'myokit.half_maximal_effect_concentration_aix_vii',
        'myokit.half_maximal_effect_concentration_aix_x',
        'myokit.half_maximal_effect_concentration_apc_ps_complex_av',
        'myokit.half_maximal_effect_concentration'
        '_apc_ps_complex_av_ax_complex',
        'myokit.half_maximal_effect_concentration_apc_ps_complex_aviii',
        'myokit.half_maximal_effect_concentration_apc_ps_complex_plasminogen',
        'myokit.half_maximal_effect_concentration_apc_ps_complex_xf',
        'myokit.half_maximal_effect_concentration_av_ax_complex_ii',
        'myokit.half_maximal_effect_concentration_avii_tf_complex_ix',
        'myokit.half_maximal_effect_concentration_avii_tf_complex_vii',
        'myokit.half_maximal_effect_concentration_avii_tf_complex_x',
        'myokit.half_maximal_effect_concentration_avii_x',
        'myokit.half_maximal_effect_concentration_ax_ii',
        'myokit.half_maximal_effect_concentration_ax_vii',
        'myokit.half_maximal_effect_concentration_ax_vii_tf_complex',
        'myokit.half_maximal_effect_concentration_axi_ix',
        'myokit.half_maximal_effect_concentration_axii_prekallikrein',
        'myokit.half_maximal_effect_concentration_axii_xi',
        'myokit.half_maximal_effect_concentration_axiii_fibrin',
        'myokit.half_maximal_effect_concentration_ca_xii',
        'myokit.half_maximal_effect_concentration_fibrin_plasminogen',
        'myokit.half_maximal_effect_concentration_kallikrein_xii',
        'myokit.half_maximal_effect_concentration_plasmin_fibrin',
        'myokit.half_maximal_effect_concentration_plasmin_fibrinogen',
        'myokit.half_maximal_effect_concentration_plasmin_xf',
        'myokit.half_maximal_effect_concentration_tf_vii_tf_complex',
        'myokit.half_maximal_effect_concentration_thrombin_fibrinogen',
        'myokit.half_maximal_effect_concentration_thrombin_plasminogen',
        'myokit.half_maximal_effect_concentration_thrombin_v',
        'myokit.half_maximal_effect_concentration_thrombin_vii',
        'myokit.half_maximal_effect_concentration_thrombin_viii',
        'myokit.half_maximal_effect_concentration_thrombin_xi',
        'myokit.half_maximal_effect_concentration_thrombin_xiii',
        'myokit.half_maximal_effect_concentration_warfarin',
        'myokit.input_rate_vk',
        'myokit.maximal_activation_rate_aii_tmod_complex_pc',
        'myokit.maximal_activation_rate_aix_aviii_complex_x',
        'myokit.maximal_activation_rate_aix_vii',
        'myokit.maximal_activation_rate_aix_x',
        'myokit.maximal_activation_rate_av_ax_complex_ii',
        'myokit.maximal_activation_rate_avii_tf_complex_ix',
        'myokit.maximal_activation_rate_avii_tf_complex_vii',
        'myokit.maximal_activation_rate_avii_tf_complex_x',
        'myokit.maximal_activation_rate_avii_x',
        'myokit.maximal_activation_rate_ax_ii',
        'myokit.maximal_activation_rate_ax_vii',
        'myokit.maximal_activation_rate_ax_vii_tf_complex',
        'myokit.maximal_activation_rate_axi_ix',
        'myokit.maximal_activation_rate_axii_xi',
        'myokit.maximal_activation_rate_ca_xii',
        'myokit.maximal_activation_rate_kallikrein_xii',
        'myokit.maximal_activation_rate_tf_vii_tf_complex',
        'myokit.maximal_activation_rate_thrombin_v',
        'myokit.maximal_activation_rate_thrombin_vii',
        'myokit.maximal_activation_rate_thrombin_viii',
        'myokit.maximal_activation_rate_thrombin_xi',
        'myokit.maximal_activation_rate_thrombin_xiii',
        'myokit.maximal_conversion_rate_apc_ps_complex_plasminogen',
        'myokit.maximal_conversion_rate_axii_prekallikrein',
        'myokit.maximal_conversion_rate_axiii_fibrin',
        'myokit.maximal_conversion_rate_fibrin_plasminogen',
        'myokit.maximal_conversion_rate_thrombin_fibrinogen',
        'myokit.maximal_conversion_rate_thrombin_plasminogen',
        'myokit.maximal_degradation_rate_apc_ps_complex_av',
        'myokit.maximal_degradation_rate_apc_ps_complex_av_ax_complex',
        'myokit.maximal_degradation_rate_apc_ps_complex_aviii',
        'myokit.maximal_degradation_rate_apc_ps_complex_xf',
        'myokit.maximal_degradation_rate_plasmin_fibrin',
        'myokit.maximal_degradation_rate_plasmin_fibrinogen',
        'myokit.maximal_degradation_rate_plasmin_xf',
        'myokit.maximal_inhibitory_effect_warfarin',
        'myokit.production_rate_fibrinogen',
        'myokit.production_rate_ii',
        'myokit.production_rate_ix',
        'myokit.production_rate_pc',
        'myokit.production_rate_plasminogen',
        'myokit.production_rate_prekallikrein',
        'myokit.production_rate_ps',
        'myokit.production_rate_tfpi',
        'myokit.production_rate_tmod',
        'myokit.production_rate_v',
        'myokit.production_rate_vii',
        'myokit.production_rate_viii',
        'myokit.production_rate_x',
        'myokit.production_rate_xi',
        'myokit.production_rate_xii',
        'myokit.production_rate_xiii',
        'myokit.transition_rate_vk_central_to_peripheral',
        'myokit.transition_rate_vk_peripheral_to_central',
        'peripheral_vitamin_k.size',
        'Sigma log'])

    # Import parameters
    directory = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parameter_file = '/models/hartmann_coagulation_model_parameters.csv'
    parameters = pd.read_csv(directory + parameter_file)

    return (population_model, parameters)


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
