import os

import chi
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
    directory = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_file = '/models/hamberg_warfarin_inr_model.xml'
    parameter_file = '/models/hamberg_warfarin_inr_model_parameters.csv'

    # Define model
    model = chi.PKPDModel(directory + model_file)
    model.set_administration(
        compartment='central', amount_var='s_warfarin_amount', direct=False)
    model = chi.ReducedMechanisticModel(model)
    model.fix_parameters({
        'central.s_warfarin_amount': 0.001,
        'dose.drug_amount': 0.001,
        'myokit.delay_compartment_1_chain_1': 1,
        'myokit.delay_compartment_2_chain_1': 1,
        'myokit.delay_compartment_1_chain_2': 1,
        'myokit.delay_compartment_2_chain_2': 1,
        'myokit.relative_change_cf1': 1,
        'myokit.relative_change_cf2': 1,
        'myokit.gamma': 1.15,
        'dose.absorption_rate': 2,
        'myokit.baseline_inr': 1,
        'myokit.maximal_effect': 1,
        'myokit.maximal_inr_shift': 20
    })
    model.set_outputs(
        ['myokit.inr', 'central.s_warfarin_concentration'])

    # Import model parameters
    parameters = pd.read_csv(directory + parameter_file)

    return (model, parameters)


def define_hamberg_population_model(centered=True):
    """
    Returns Hamberg's population model for the semi-mechanistic model of the
    INR response to warfarin treatment.
    """
    # Define covariate model for clearance
    clearance_cov_model = chi.CovariatePopulationModel(
        population_model=chi.LogNormalModel(
            dim_names=['Clearance'], centered=centered),
        covariate_model=HambergClearanceCovariateModel()
    )
    ec50_cov_model = chi.CovariatePopulationModel(
        population_model=chi.LogNormalModel(
            dim_names=['EC50'], centered=centered),
        covariate_model=HambergEC50CovariateModel()
    )
    population_model = chi.ComposedPopulationModel([
        chi.LogNormalModel(
            dim_names=['Volume of distribution'], centered=centered),
        clearance_cov_model,
        ec50_cov_model,
        chi.PooledModel(n_dim=4, dim_names=[
            'Transition rate chain 1',
            'Transition rate chain 2',
            'Drug conc. Sigma log',
            'INR Sigma log'])
    ])
    population_model.set_dim_names([
        'Volume of distribution',
        'Clearance',
        'EC50',
        'Transition rate chain 1',
        'Transition rate chain 2',
        'Drug conc. Sigma log',
        'INR Sigma log'
    ])

    return population_model


class HambergClearanceCovariateModel(chi.CovariateModel):
    r"""
    Implements Hamberg's covariate model of the clearance.

    In this model the typical clearance is assumes to be a function of
    the age and the CYP2C9 genotype

    .. math::
        CL = (CL_{a_1} + CL_{a_2}) (1 - tanh(r_{age}(Age - 71))),

    where :math:`CL` denotes the clearance, and
    :math:`CL_{a_1}` and :math:`CL_{a_2}` the clearance contributions from the
    CYP2C9 alleles. :math:`r_{age}` denotes the change of the clearance with the
    age of the patient. Note that the tanh is a modification of Hamberg's model
    that avoids negative clearances.

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

    The parameters of the model are the relative decrease of the clearance from
    *1/*1 to *2/2, the relative decrease from *1/*1 to *3/*3 and the change of
    the clearance with age, :math:`r_{age}`. The first two parameters are
    defined in the interval [0, 1], and :math:`r_{age}` is defined over all
    positive numbers.

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
        super(HambergClearanceCovariateModel, self).__init__(n_cov, cov_names)

        # Set number of parameters (shift *2, shift *3, shift age)
        # Note the clearance for *1/*1 is implemented as the baseline
        self._n_parameters = 3
        self._parameter_names = [
            'Rel. clearance shift *2/*2',
            'Rel. clearance shift *3/*3',
            'Rel. clearance shift with age']

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
        return super(HambergClearanceCovariateModel, self).get_parameter_names(
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
