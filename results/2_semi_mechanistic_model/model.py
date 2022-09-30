import copy
import os

import chi
import myokit
import myokit.formats.sbml as sbml
import numpy as np
import pandas as pd


def define_hamberg_model():
    """
    Returns Hamberg's semi-mechanistic model of the INR response to warfarin
    treatment.

    Reference
    ---------
    .. Hamberg AK, Wadelius M, Lindh JD, Dahl ML, Padrini R, Deloukas P,
        Rane A, Jonsson EN. A pharmacometric model describing the relationship
        between warfarin dose and INR response with respect to variations in
        CYP2C9, VKORC1, and age. Clin Pharmacol Ther. 2010 Jun;87(6):727-34.

    :returns: Hamberg model, typical parameter values
    :rtype: Tuple[chi.MechanisticModel, pandas.DataFrame]
    """
    # Define model
    model = HambergModel()

    # Fix parameters that are not inferred
    model = chi.ReducedMechanisticModel(model)

    # Fixing initial amounts to small values avoids infinities of lognormal
    # error and does not significantly influence model predictions
    model.fix_parameters({
        'myokit.amount_dose_compartment': 0.001,
        'myokit.amount_central_compartment': 0.001})

    # Fix parameters that are not inferred to Hamberg et al's values
    model.fix_parameters({
        'myokit.delay_compartment_1_chain_1': 1,
        'myokit.delay_compartment_2_chain_1': 1,
        'myokit.delay_compartment_1_chain_2': 1,
        'myokit.delay_compartment_2_chain_2': 1,
        'myokit.relative_change_cf1': 1,
        'myokit.relative_change_cf2': 1,
        'myokit.gamma': 1.15,
        'myokit.absorption_rate': 2,
        'myokit.baseline_inr': 1,
        'myokit.maximal_effect': 1,
        'myokit.maximal_inr_shift': 20
    })

    # Fix initial conditions of sensitivities
    sens_0 = [
        'damount_central_compartment_delimination_rate',
        'ddelay_compartment_1_chain_1_delimination_rate',
        'ddelay_compartment_1_chain_1_dhalf_maximal_effect_concentration',
        'ddelay_compartment_1_chain_1_dtransition_rate_chain_1',
        'ddelay_compartment_1_chain_1_dvolume',
        'ddelay_compartment_1_chain_2_delimination_rate',
        'ddelay_compartment_1_chain_2_dhalf_maximal_effect_concentration',
        'ddelay_compartment_1_chain_2_dtransition_rate_chain_2',
        'ddelay_compartment_1_chain_2_dvolume',
        'ddelay_compartment_2_chain_1_delimination_rate',
        'ddelay_compartment_2_chain_1_dhalf_maximal_effect_concentration',
        'ddelay_compartment_2_chain_1_dtransition_rate_chain_1',
        'ddelay_compartment_2_chain_1_dvolume',
        'ddelay_compartment_2_chain_2_delimination_rate',
        'ddelay_compartment_2_chain_2_dhalf_maximal_effect_concentration',
        'ddelay_compartment_2_chain_2_dtransition_rate_chain_2',
        'ddelay_compartment_2_chain_2_dvolume',
        'drelative_change_cf1_delimination_rate',
        'drelative_change_cf1_dhalf_maximal_effect_concentration',
        'drelative_change_cf1_dtransition_rate_chain_1',
        'drelative_change_cf1_dvolume',
        'drelative_change_cf2_delimination_rate',
        'drelative_change_cf2_dhalf_maximal_effect_concentration',
        'drelative_change_cf2_dtransition_rate_chain_2',
        'drelative_change_cf2_dvolume']
    model.fix_parameters({'myokit.' + p: 0 for p in sens_0})

    # Import model parameters
    directory = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parameter_file = '/models/hamberg_warfarin_inr_model_parameters.csv'
    parameters = pd.read_csv(directory + parameter_file)

    return (model, parameters)


def define_hamberg_population_model(centered=True):
    """
    Returns Hamberg's population model for the semi-mechanistic model of the
    INR response to warfarin treatment.
    """
    # Define covariate model for the elimination rate
    # (Hamberg model for clearance is effectively a model for the elimination
    # rate, since the effective volume of distribution is independent of
    # covariates)
    elim_rate_cov_model = chi.CovariatePopulationModel(
        population_model=chi.LogNormalModel(
            dim_names=['Elimination rate'], centered=centered),
        covariate_model=HambergEliminationRateCovariateModel()
    )
    ec50_cov_model = chi.CovariatePopulationModel(
        population_model=chi.LogNormalModel(
            dim_names=['EC50'], centered=centered),
        covariate_model=HambergEC50CovariateModel()
    )
    population_model = chi.ComposedPopulationModel([
        elim_rate_cov_model,
        ec50_cov_model,
        chi.PooledModel(n_dim=2, dim_names=[
            'Transition rate chain 1',
            'Transition rate chain 2']),
        chi.LogNormalModel(
            dim_names=['Volume of distribution'], centered=centered),
        chi.PooledModel(n_dim=2, dim_names=[
            'Drug conc. Sigma log',
            'INR Sigma log'])
    ])
    population_model.set_dim_names([
        'Elimination rate',
        'EC50',
        'Transition rate chain 1',
        'Transition rate chain 2',
        'Volume of distribution',
        'Drug conc. Sigma log',
        'INR Sigma log'
    ])

    return population_model


class HambergModel(chi.MechanisticModel):
    """
    Implements Hamberg's semi-mechanistic model of the INR response to warfarin
    treatment.

    .. note::
        An out-of-the-box implementation of the model can be instantiated using
        the SBML file in /models/hamberg_warfarin_inr_model.xml and
        chi.PKPDModel(model_file). However, for the inference of the model,
        we found that implementing the sensitivities manually improves the
        performance which requires wrapping the model in this way.

    Reference
    ---------
    .. Hamberg AK, Wadelius M, Lindh JD, Dahl ML, Padrini R, Deloukas P,
        Rane A, Jonsson EN. A pharmacometric model describing the relationship
        between warfarin dose and INR response with respect to variations in
        CYP2C9, VKORC1, and age. Clin Pharmacol Ther. 2010 Jun;87(6):727-34.
    """
    def __init__(self):
        # Import model
        directory = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_file = \
            '/models/hamberg_warfarin_inr_model_with_sensitivities.xml'
        self._model = sbml.SBMLImporter().model(directory + model_file)

        # Bind dose rate to pacing protocol
        dose_rate = self._model.get('myokit.dose_rate')
        dose_rate.set_binding('pace')
        self._dosing_regimen = None

        # Set default number and names of states, parameters and outputs.
        self._set_number_and_names()

        # Create simulator without sensitivities
        self._simulator = myokit.Simulation(self._model)
        self._has_sensitivities = False

        # Define sensitivities of model
        self._n_sens_per_output = 5
        dinr = 'myokit.dinr'
        dconc = 'myokit.dconcentration_central_compartment'
        self._sensitivities = [
            dconc + '_delimination_rate',
            dconc + '_dhalf_maximal_effect_concentration',
            dconc + '_dtransition_rate_chain_1',
            dconc + '_dtransition_rate_chain_2',
            dconc + '_dvolume',
            dinr + '_delimination_rate',
            dinr + '_dhalf_maximal_effect_concentration',
            dinr + '_dtransition_rate_chain_1',
            dinr + '_dtransition_rate_chain_2',
            dinr + '_dvolume']

    def _set_const(self, parameters):
        """
        Sets values of constant model parameters.
        """
        for id_var, var in enumerate(self._const_names):
            self._simulator.set_constant(var, float(parameters[id_var]))

    def _set_state(self, parameters):
        """
        Sets initial values of states.
        """
        parameters = np.array(parameters)
        parameters = parameters[self._original_order]
        self._simulator.set_state(parameters)

    def _set_number_and_names(self):
        """
        Sets the number of states, parameters and outputs, as well as their
        names. If the model is ``None`` the self._model is taken.
        """
        # Get the number of states and parameters
        self._n_states = self._model.count_states()
        n_const = self._model.count_variables(const=True)

        # Get constant variable names and state names
        names = [var.qname() for var in self._model.states()]
        self._state_names = sorted(names)

        const_names = []
        for var in self._model.variables(const=True):
            # Sometimes constants are derived from parameters
            if not var.is_literal():
                n_const -= 1
                continue
            const_names.append(var.qname())
        self._const_names = sorted(const_names)
        self._n_parameters = self._n_states + n_const

        # Remember original order of state names for simulation
        order_after_sort = np.argsort(names)
        self._original_order = np.argsort(order_after_sort)

        # Set default parameter names
        self._parameter_names = self._state_names + self._const_names

        # Set default outputs
        self._allowed_outputs = [
            'myokit.concentration_central_compartment', 'myokit.inr']
        self._output_names = [
            'myokit.concentration_central_compartment', 'myokit.inr']
        self._n_outputs = 2
        self._sensitivity_index = [0, 1]

        # Create references of displayed parameter and output names to
        # orginal myokit names (defaults to identity map)
        # (Key: myokit name, value: displayed name)
        self._parameter_name_map = dict(
            zip(self._parameter_names, self._parameter_names))
        self._output_name_map = dict(
            zip(self._output_names, self._output_names))

    def copy(self):
        """
        Returns a deep copy of the mechanistic model.

        .. note::
            Copying the model resets the sensitivity settings.
        """
        # Copy model manually and get protocol
        m = self._model.clone()
        s = self._simulator
        myokit_model = m.clone()
        self._model = None
        self._simulator = None

        # Copy the mechanistic model
        model = copy.deepcopy(self)

        # Replace myokit model by safe copy and create simulator
        self._model = m
        self._simulator = s
        self._simulator.set_protocol(self._dosing_regimen)
        model._model = myokit_model
        model._simulator = myokit.Simulation(myokit_model)
        model._simulator.set_protocol(self._dosing_regimen)

        return model

    def dosing_regimen(self):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`myokit.Protocol`. If the protocol has not been set, ``None`` is
        returned.
        """
        return self._dosing_regimen

    def enable_sensitivities(self, enabled, *arg, **kwargs):
        """
        Enables the computation of the model output sensitivities to the model
        parameters if set to ``True``.

        The sensitivities are computed using the forward sensitivities method,
        where an ODE for each sensitivity is derived. The sensitivities are
        returned together with the solution to the orginal system of ODEs when
        simulating the mechanistic model :meth:`simulate`.

        The optional parameter names argument can be used to set which
        sensitivities are computed. By default the sensitivities to all
        parameters are computed.

        :param enabled: A boolean flag which enables (``True``) / disables
            (``False``) the computation of sensitivities.
        :type enabled: bool
        """
        # Sensitivities are always simulated for this model. This switch only
        # determines whether or not they are returned.
        enabled = bool(enabled)
        self._has_sensitivities = enabled

    def has_sensitivities(self):
        """
        Returns a boolean indicating whether sensitivities have been enabled.
        """
        return self._has_sensitivities

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._n_parameters

    def outputs(self):
        """
        Returns the output names of the model.
        """
        # Get user specified output names
        output_names = [
            self._output_name_map[name] for name in self._output_names]
        return output_names

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        # Get user specified parameter names
        parameter_names = [
            self._parameter_name_map[name] for name in self._parameter_names]

        return parameter_names

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.

        The outputs can be set to any quantifiable variable name of the
        :class:`myokit.Model`, e.g. `compartment.variable`.

        .. note::
            Setting outputs resets the sensitivity settings (by default
            sensitivities are disabled.)

        :param outputs:
            A list of output names.
        :type outputs: list[str]
        """
        outputs = list(outputs)
        for output in outputs:
            if output not in self._allowed_outputs:
                raise ValueError(
                    'Invalid outputs. The models outputs can be '
                    '<myokit.inr> and/or '
                    '<myokit.concentration_central_compartment>')

        # Remember outputs
        self._output_names = outputs
        self._n_outputs = len(outputs)

        # Disable sensitivities
        indices = []
        for n in range(len(self._allowed_outputs)):
            if self._allowed_outputs[n] in self._output_names:
                indices.append(n)

        self._sensitivity_index = indices

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs (and optionally
        the sensitivites) for the specified parameters and times.

        The model outputs are returned as a 2 dimensional NumPy array of shape
        ``(n_outputs, n_times)``. If sensitivities are enabled, a tuple is
        returned with the NumPy array of the model outputs and a NumPy array of
        the sensitivities of shape ``(n_times, n_outputs, n_parameters)``.

        :param parameters: An array-like object with values for the model
            parameters.
        :type parameters: list, numpy.ndarray
        :param times: An array-like object with time points at which the output
            values are returned.
        :type times: list, numpy.ndarray

        :rtype: np.ndarray of shape (n_outputs, n_times) or
            (n_times, n_outputs, n_parameters)
        """
        # Reset simulation
        self._simulator.reset()

        # Set initial conditions
        self._set_state(parameters[:self._n_states])

        # Set constant model parameters
        self._set_const(parameters[self._n_states:])

        # Simulate
        if not self._has_sensitivities:
            output = self._simulator.run(
                times[-1] + 1, log=self._output_names, log_times=times)
            output = np.array([output[name] for name in self._output_names])

            return output

        output = self._simulator.run(
            times[-1] + 1, log=self._output_names + self._sensitivities,
            log_times=times)

        n_times = len(times)
        sensitivities = np.array(
            [output[name] for name in self._sensitivities])
        output = np.array([output[name] for name in self._output_names])

        # Reshape sensitivities (and only return sensitivities of set outputs)
        sensitivities = sensitivities.reshape(
            2, self._n_sens_per_output, n_times)
        sensitivities = np.moveaxis(
            sensitivities, source=(0, 1, 2), destination=(1, 2, 0))
        sensitivities = sensitivities[:, self._sensitivity_index, :]

        return output, sensitivities

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

        :param dose: The amount of the compound that is injected at each
            administration, or a myokit.Protocol instance that defines the
            dosing regimen.
        :type dose: float or myokit.Protocol
        :param start: Start time of the treatment. By default the
            administration starts at t=0.
        :type start: float, optional
        :param duration: Duration of dose administration. By default the
            duration is set to 0.01 of the time unit (bolus).
        :type duration: float, optional
        :param period: Periodicity at which doses are administered. If ``None``
            the dose is administered only once.
        :type period: float, optional
        :param num: Number of administered doses. If ``None`` and the
            periodicity of the administration is not ``None``, doses are
            administered indefinitely.
        :type num: int, optional
        """
        if num is None:
            # Myokits default is zero, i.e. infinitely many doses
            num = 0

        if period is None:
            # If period is not provided, we administer a single dose
            # Myokits defaults are 0s for that.
            period = 0
            num = 0

        if isinstance(dose, myokit.Protocol):
            self._simulator.set_protocol(dose)
            self._dosing_regimen = dose
            return None

        # Translate dose to dose rate
        dose_rate = dose / duration

        # Set dosing regimen
        dosing_regimen = myokit.pacing.blocktrain(
            period=period, duration=duration, offset=start, level=dose_rate,
            limit=num)
        self._simulator.set_protocol(dosing_regimen)
        self._dosing_regimen = dosing_regimen


    def supports_dosing(self):
        """
        Returns a boolean whether dose administration with
        :meth:`PKPDModel.set_dosing_regimen` is supported by the model.
        """
        return True


class HambergEliminationRateCovariateModel(chi.CovariateModel):
    r"""
    Implements Hamberg's covariate model of the elimination rate.

    In this model the typical elimination rate is assumed to be a function of
    the age and the CYP2C9 genotype

    .. math::
        k_e = (k_{a_1} + k_{a_2}) (1 - tanh(r_{age}(Age - 71))),

    where :math:`k_e` denotes the elimination rate, and
    :math:`k_{a_1}` and :math:`__{a_2}` the elimination rate contributions
    from the CYP2C9 alleles. :math:`r_{age}` denotes the change of the
    clearance with the age of the patient. Note that the tanh is a modification
    of Hamberg's model that avoids negative clearances.

    Hamberg's original model defines the covariate model for the clearance
    and not for the elimination rate. However, the elimination rate and
    clearance are proportional to each other

    .. math::
        k_e = \frac{1}{v}Cl,

    where :math:`v` is the volume of distribution. In Hamberg's model the
    volume of distribution is modelled to be independent of the covariates and
    the clearance.
    As a result, the covariate model describes also the variation of the
    elimination rate.

    The first covariate encodes the CYP2C9 genotype, and the second covariate
    the genotype the age. In particular, the CYP2C9 variants are encoded as
    follows:

    0: 'CYP2C9 variant *1/*1'
    1: 'CYP2C9 variant *1/*2'
    2: 'CYP2C9 variant *1/*3'
    3: 'CYP2C9 variant *2/*2'
    4: 'CYP2C9 variant *2/*3'
    5: 'CYP2C9 variant *3/*3'

    The age of the patients are expected to be in years.

    The parameters of the model are the relative decrease of the elimination
    rate from *1/*1 to *2/2, the relative decrease from *1/*1 to *3/*3 and the
    change of the elimination rate with age, :math:`r_{age}`. The first two
    parameters are defined in the interval [0, 1], and :math:`r_{age}` is
    defined over all positive numbers.

    .. note::
        This model is meant to be used together with a lognormal population
        model where the location parameter is the logarithm of the typical
        population value. The model is therefore implement under the assumption
        that the logarithm of the typical population value is provided.

    Extends :class:`CovariateModel`.
    """
    def __init__(self):
        n_cov = 2
        cov_names = ['CYP2C9', 'Age']
        super(HambergEliminationRateCovariateModel, self).__init__(
            n_cov, cov_names)

        # Set number of parameters (shift *2, shift *3, shift age)
        # Note the clearance for *1/*1 is implemented as the baseline
        self._n_parameters = 3
        self._parameter_names = [
            'Rel. elimination rate shift *2/*2',
            'Rel. elimination rate shift *3/*3',
            'Rel. elimination rate shift with age']

    def compute_population_parameters(
            self, parameters, pop_parameters, covariates):
        """
        Returns the transformed population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_pop_params_per_dim, n_dim)``
        """
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, self._n_parameters)
        relative_shift_22, relative_shift_33, age_change = parameters[0]

        # Compute population parameters
        n_pop, n_dim = pop_parameters.shape
        if n_dim > 1:
            raise ValueError(
                'Invalid pop_parameters. The model is only defined for 1 '
                'dimensional population models.')

        n_ids = len(covariates)
        vartheta = np.zeros((n_ids, n_pop, n_dim))
        vartheta += pop_parameters[np.newaxis, ...]

        # Compute individual parameters
        cyp2c9 = covariates[:, 0]
        age = covariates[:, 1] - 71

        # CYP2C9 variant *1/*1
        # Implemented as baseline

        # CYP2C9 variant *1/*2
        mask = cyp2c9 == 1
        vartheta[mask, 0] += np.log(1 - relative_shift_22 / 2)

        # CYP2C9 variant *1/*3
        mask = cyp2c9 == 2
        vartheta[mask, 0] += np.log(1 - relative_shift_33 / 2)

        # CYP2C9 variant *2/*2
        mask = cyp2c9 == 3
        vartheta[mask, 0] += np.log(1 - relative_shift_22)

        # CYP2C9 variant *2/*3
        mask = cyp2c9 == 4
        vartheta[mask, 0] += np.log(
            1 - (relative_shift_22 + relative_shift_33) / 2)

        # CYP2C9 variant *3/*3
        mask = cyp2c9 == 5
        vartheta[mask, 0] += np.log(1 - relative_shift_33)

        # Age
        vartheta[:, 0] += \
            np.log(1 - np.tanh(age[:, np.newaxis] * age_change))

        return vartheta

    def compute_sensitivities(
            self, parameters, pop_parameters, covariates, dlogp_dvartheta):
        """
        Returns the sensitivities of the likelihood with respect to
        the model parameters and the population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_selected, n_cov)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param dlogp_dvartheta: Unflattened sensitivities of the population
            model to the transformed parameters.
        :type dlogp_dvartheta: np.ndarray of shape
            ``(n_ids, n_param_per_dim, n_dim)``
        :rtype: Tuple[np.ndarray of shape ``(n_pop_params,)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, self._n_parameters)
        relative_shift_22, relative_shift_33, age_change = parameters[0]

        # Compute sensitivities
        n_pop, n_dim = pop_parameters.shape
        n_pop = n_pop * n_dim
        dpop = np.sum(dlogp_dvartheta, axis=0).flatten()

        # Compute derivates of mu
        n_ids = len(covariates)
        cyp2c9 = covariates[:, 0]
        age = covariates[:, 1] - 71
        dmu = np.zeros(shape=(n_ids, self._n_parameters))

        # CYP2C9 variant *1/*1
        # Implemented as baseline

        # CYP2C9 variant *1/*2
        mask = cyp2c9 == 1
        dmu[mask, 0] -= 1 / (1 - relative_shift_22 / 2) / 2

        # CYP2C9 variant *1/*3
        mask = cyp2c9 == 2
        dmu[mask, 1] -= 1 / (1 - relative_shift_33 / 2) / 2

        # CYP2C9 variant *2/*2
        mask = cyp2c9 == 3
        dmu[mask, 0] -= 1 / (1 - relative_shift_22)

        # CYP2C9 variant *2/*3
        mask = cyp2c9 == 4
        dmu[mask, 0] -= \
            1 / (1 - (relative_shift_22 + relative_shift_33) / 2) / 2
        dmu[mask, 1] -= \
            1 / (1 - (relative_shift_22 + relative_shift_33) / 2) / 2

        # CYP2C9 variant *3/*3
        mask = cyp2c9 == 5
        dmu[mask, 1] -= 1 / (1 - relative_shift_33)

        # Age
        dmu[:, 2] -= \
            age / (1 - np.tanh(age * age_change)) / (1 + (age * age_change)**2)

        dparams = np.sum(
            dlogp_dvartheta[:, 0, 0, np.newaxis] * dmu, axis=0)

        return dpop, dparams

    def get_parameter_names(self):
        """
        Returns the names of the model parameters.
        """
        return super(
            HambergEliminationRateCovariateModel, self).get_parameter_names(
                exclude_cov_names=True)

    def set_parameter_names(self, names=None, mask_names=False):
        """
        Sets the names of the model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        # This is just a dummy method. The names of the parameters are fixed
        # for this class
        pass

    def set_population_parameters(self, indices):
        """
        This is a dummy method. The modified population parameter of this
        model is always the first parameter.

        :param indices: A list of parameter indices
            [param index per dim, dim index].
        :type indices: List[List[int]]
        """
        self._pidx = np.array([0])
        self._didx = np.array([0])

        # Update number of parameters and parameters names
        self._n_selected = 1


class HambergEC50CovariateModel(chi.CovariateModel):
    r"""
    Implements Hamberg's covariate model of the EC50.

    In this model the typical clearance is assumed to be a function of
    the VKORC1 genotype

    .. math::
        EC50 = EC50_{a_1} + EC50_{a_2},

    where :math:`EC50` denotes the half maximal effect concentration of
    warfarin, and :math:`EC50_{a_1}` and :math:`EC50_{a_2}` the EC50
    contributions from the VKORC1 alleles.

    The covariate encodes the VKORC1 genotype. In particular, the VKORC1
    variants are encoded as follows:

    0: 'VKORC1 variant GG'
    1: 'VKORC1 variant GA'
    2: 'VKORC1 variant AA'

    The parameter of the model is the relative decrease of the EC50 from
    G/G to A/A, which assumes values between 0 and 1.

    .. note::
        This model is meant to be used together with a lognormal population
        model where the location parameter is the logarithm of the typical
        population value. The model is therefore implement under the assumption
        that the logarithm of the typical population value is provided.

    Extends :class:`CovariateModel`.
    """
    def __init__(self):
        n_cov = 1
        cov_names = ['VKORC1']
        super(HambergEC50CovariateModel, self).__init__(n_cov, cov_names)

        # Set number of parameters (shift A)
        # Note the EC50 for G/G is implemented as the baseline
        self._n_parameters = 1
        self._parameter_names = ['Rel. EC50 shift AA']

    def compute_population_parameters(
            self, parameters, pop_parameters, covariates):
        """
        Returns the transformed population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_pop_params_per_dim, n_dim)``
        """
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, self._n_parameters)
        relative_shift = parameters[0, 0]

        # Compute population parameters
        n_pop, n_dim = pop_parameters.shape
        if n_dim > 1:
            raise ValueError(
                'Invalid pop_parameters. The model is only defined for 1 '
                'dimensional population models.')

        n_ids = len(covariates)
        vartheta = np.zeros((n_ids, n_pop, n_dim))
        vartheta += pop_parameters[np.newaxis, ...]

        # Compute individual parameters
        vkorc1 = covariates[:, 0]

        # VKORC1 variant G/G
        # Implemented as baseline

        # VKORC1 variant G/A
        mask = vkorc1 == 1
        vartheta[mask, 0] += np.log(1 - relative_shift / 2)

        # VKORC1 variant A/A
        mask = vkorc1 == 2
        vartheta[mask, 0] += np.log(1 - relative_shift)

        return vartheta

    def compute_sensitivities(
            self, parameters, pop_parameters, covariates, dlogp_dvartheta):
        """
        Returns the sensitivities of the likelihood with respect to
        the model parameters and the population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_selected, n_cov)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param dlogp_dvartheta: Unflattened sensitivities of the population
            model to the transformed parameters.
        :type dlogp_dvartheta: np.ndarray of shape
            ``(n_ids, n_param_per_dim, n_dim)``
        :rtype: Tuple[np.ndarray of shape ``(n_pop_params,)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, self._n_parameters)
        relative_shift = parameters[0, 0]

        # Compute sensitivities
        n_pop, n_dim = pop_parameters.shape
        n_pop = n_pop * n_dim
        dpop = np.sum(dlogp_dvartheta, axis=0).flatten()

        # Compute derivates of mu
        n_ids = len(covariates)
        vkorc1 = covariates[:, 0]
        dmu = np.zeros(shape=(n_ids, self._n_parameters))

        # VKORC1 variant G/G
        # Implemented as baseline

        # VKORC1 variant G/A
        mask = vkorc1 == 1
        dmu[mask, 0] -= 1 / (1 - relative_shift / 2) / 2

        # VKORC1 variant G/A
        mask = vkorc1 == 2
        dmu[mask, 0] -= 1 / (1 - relative_shift)

        dparams = np.sum(
            dlogp_dvartheta[:, 0, 0, np.newaxis] * dmu, axis=0)

        return dpop, dparams

    def get_parameter_names(self):
        """
        Returns the names of the model parameters.
        """
        return super(HambergEC50CovariateModel, self).get_parameter_names(
            exclude_cov_names=True)

    def set_parameter_names(self, names=None, mask_names=False):
        """
        Sets the names of the model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        # This is just a dummy method. The names of the parameters are fixed
        # for this class
        pass

    def set_population_parameters(self, indices):
        """
        This is a dummy method. The modified population parameter of this
        model is always the first parameter.

        :param indices: A list of parameter indices
            [param index per dim, dim index].
        :type indices: List[List[int]]
        """
        self._pidx = np.array([0])
        self._didx = np.array([0])

        # Update number of parameters and parameters names
        self._n_selected = 1
