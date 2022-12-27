import os

import chi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import myokit
import numpy as np


class QLearningDosePolicy(object):
    r"""
    Implements a Q-learning dose recommender for warfarin dosing.

    Doses are predicted based on the current state of the patient, defined
    by the INR measurement, using an action-value function Q(s, a), where s
    denotes the state and a the action (which dose amount to administer).
    Q(s, a) quantifies the expected return (see below). Once Q(s, a) is
    estimated, the dosing policy is to take the action that maximises Q

    .. math::
        \pi (a | s) = 1 for a = \argmax Q(s, a) and 0 else.

    The action-value function, Q(s, a), is learned iteratively

    .. math::
        Q_{k+1}(s | a) =
            (1 - \alpha) Q_{k}(s | a)
            + \alpha (R(s') + \gamma \max _{a'} Q_{k}(s', a')),

    where s' is the state after taking action a' in state s. \alpha is the
    learning rate and \gamma is the discount factor.

    Here, we discretise the state space and the action space to be able to
    model Q as a matrix.

        .. math:: R(s) = -(2.5 - s) ^ 2.
    """
    def __init__(self, inr_min=0.6, inr_max=5, inr_step=0.2):
        if inr_min <= 0:
            raise ValueError('inr_min is invalid.')
        if inr_max <= inr_min:
            raise ValueError('inr_max is invalid.')
        if (inr_max - inr_min) < inr_step:
            raise ValueError('inr_step is invalid.')

        # Initialise Q
        self._states = np.arange(inr_min, inr_max+inr_step, inr_step)
        self._doses = np.array([
            0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
            10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,
            16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5,
            23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29,
            29.5, 30
        ])
        n_states = len(self._states)
        n_actions = len(self._doses)
        self._q = np.zeros(shape=(n_states, n_actions))

        # Set q for largest INR to choosing 0 mg. This avoids randomly
        # choosing doses when INR get too large and therefore were never
        # observed.
        self._q[-1, 0] = 100

        # Book-keeping
        self._iterations_trained = 0

    def estimate_expected_return(self, state):
        """
        Returns the expected return using the current Q estimate.
        """
        state_index = self.get_state_index(state)

        return self._q[state_index]

    def get_action_index(self, action):
        """
        Returns index of state.
        """
        if (action < self._doses[0]) or (action > self._doses[-1]):
            raise ValueError('action is invalid.')

        for index, dose in enumerate(self._doses):
            if action <= dose:
                return index

    def get_state_index(self, state):
        """
        Returns index of state.
        """
        # Get state index
        try:
            index = np.where(self._states == state)[0][0]
        except IndexError:
            # State is outside range. Assume boundary state
            index = 0 if state < self._states[0] \
                else len(self._states) - 1

        return index

    def predict_action(self, state, epsilon=0, seed=None):
        """
        Predicts dose based on the current state according to an epsilon-greedy
        policy.

        Outside training, epsilon should be set to zero.
        """
        if (epsilon == 0) or (np.random.unform(0, 1) > epsilon):
            action_index = np.argmax(self.estimate_expected_return(state))

            return self._doses[action_index]

        # Randomly choose action
        rng = np.random.default_rng(seed)

        return rng.choice(self._doses)

    def update_action_value_function(
            self, state, action, next_state, reward, alpha=0.8, gamma=0.9):
        r"""
        Uses Bellmann's equation to update the action-value function given
        a state, the next state when a given action was taken and the reward
        for being in the next state.

        The action-value function is updated using
        .. math::
            Q_{k+1}(s | a) =
                (1 - \alpha) Q_{k}(s | a)
                + \alpha (R(s') + \gamma \max _{a'} Q_{k}(s', a')),

        where s' is the state after taking action a' in state s. \alpha is the
        learning rate and \gamma is the discount factor.
        """
        state_index = self.get_state_index(state)
        action_index = self.get_action_index(action)

        # Get maximum expected return being in the next state
        max_q_next_state = np.max(self.estimate_expected_return(next_state))

        self._q[state_index, action_index] = \
            (1 - alpha) * self._q[state_index, action_index] \
            + alpha * (reward + gamma * max_q_next_state)

        # Update training iteration
        self._iterations_trained += 1


def define_model():
    """
    Returns the data-generating model and its parameters.
    """
    # TODO: First we just use Hamberg et al's model with fixed parameters.
    # Define model
    directory = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_file = '/models/hamberg_warfarin_inr_model.xml'
    model = chi.PKPDModel(directory + model_file)
    model.set_administration(
        compartment='central', amount_var='s_warfarin_amount', direct=False)
    model.set_outputs(['myokit.inr'])
    # model.set_outputs(['central.s_warfarin_concentration'])

    # Fix parameters that are not inferred
    model = chi.ReducedMechanisticModel(model)

    # Fixing initial amounts to small values avoids infinities of lognormal
    # error and does not significantly influence model predictions
    model.fix_parameters({
        'central.s_warfarin_amount': 0.001,
        'dose.drug_amount': 0.001})

    # Fix parameters that are not inferred to Hamberg et al's values
    model.fix_parameters({
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

    # Define predictive model
    model = chi.PredictiveModel(model, chi.LogNormalErrorModel())

    # Define parameters
    parameters = [
        14,
        0.3,
        4,
        0.1,
        0.02,
        0.01  # TODO: Artificially low noise
    ]

    return model, parameters


def generate_data(
        model, parameters, target=2.5, epsilon=0.1, n_episodes=1000,
        n_days=55):
    """
    Generates data from Hamberg et al's model.

    The data is a numpy array of shape (n_data, 4), where the second column
    contains the state, the taken action, the asumed state after taking that
    action, and the reward for being in that state.
    """
    # Define measurement days
    times = np.arange(n_days) * 24

    data = np.empty((n_episodes, n_days, 4))
    rng = np.random.default_rng(seed=0)
    for idd in range(n_episodes):
        doses = []

        # Simulate treatment response
        for idt, time in enumerate(times):
            # Simulate model
            inr = model.sample(
                parameters=parameters, times=[time], seed=rng,
                return_df=False)[0, 0, 0]

            # Round INR to steps of 0.2 to reduce state space
            inr = np.round(inr * 5) / 5

            # Adjust dosing regimen
            regimen, doses = adjust_dosing_regimen(
                doses, inr, target, epsilon=epsilon, seed=rng)
            model.set_dosing_regimen(regimen)

            # Compute reward
            reward = 100 - (target - inr)**2

            # Store data (state, action, next state, reward)
            data[idd, idt, 0] = inr
            data[idd, idt, 1] = doses[-1]
            if idt > 0:
                data[idd, idt-1, 2] = inr
                data[idd, idt-1, 3] = reward

    # The last day has no next state and reward measurement. So we can throw
    # this column out
    data = data[:, :-1, :]
    data = data.reshape((n_episodes * (n_days-1), 4))

    # Split into train and test set
    rng.shuffle(data, axis=0)
    split = int(len(data) * 0.9)
    train = data[:split]
    test = data[split:]

    return train, test


def adjust_dosing_regimen(
        doses, latest_inr, target, epsilon=0.1, seed=None):

    # With probability epsilon, choose a dose randomly
    rng = np.random.default_rng(seed)
    if (len(doses) == 0) or (rng.uniform(0, 1) < epsilon) or (latest_inr == 0):
        # Define possible actions (doses)
        allowed_doses = np.array([
            0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
            10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,
            16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5,
            23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29,
            29.5, 30
        ])
        dose = rng.choice(allowed_doses)
    else:
        # Naively adjust dose by fraction to target INR, if INR is outside
        # therapeutic range
        dose = doses[-1]
        dose = dose * target / latest_inr

        # Make sure that dose can be taken with conventional tablets
        if dose < 0.5:
            dose = 0
        elif dose < 1.5:
            dose = 1
        elif dose < 2.25:
            dose = 2
        else:
            dose = np.round(2 * dose) / 2

        if dose > 30:
            dose = 30

    # Reconstruct already administered dose events and add new dose
    doses.append(dose)
    duration = 0.01
    dose_rates = np.array(doses) / duration
    new_regimen = myokit.Protocol()
    for idx, dr in enumerate(dose_rates):
        new_regimen.add(myokit.ProtocolEvent(
            level=dr, start=idx*24, duration=duration, period=0))

    return new_regimen, doses


def train_q_learning_algorithm(
        data_train, data_test, n_epochs=10, gamma=0.9):
    """
    Returns a trained Q-Learning algorithm.

    Uses the model to generate data.
    """
    alpha = 0.99
    rng = np.random.default_rng(12)
    td_errors = np.zeros(shape=(n_epochs, 3))

    q_model = QLearningDosePolicy()
    for epoch in range(n_epochs):
        td_train, td_test = evaluate(q_model, data_train, data_test, gamma)
        td_errors[epoch, 0] = epoch
        td_errors[epoch, 1] = td_train
        td_errors[epoch, 2] = td_test

        lr = alpha / (1 + epoch)
        q_model = train_epoch(q_model, data_train, lr, gamma, rng)

    # Plot training results
    plot_progress(td_errors)

    return q_model


def train_epoch(q_model, data, alpha, gamma, rng):
    # Shuffle data
    rng.shuffle(data)
    for sample in data:
        state, action, next_state, reward = sample
        q_model.update_action_value_function(
            state, action, next_state, reward, alpha=alpha, gamma=gamma)

    return q_model


def evaluate(q_model, data_train, data_test, gamma):
    """
    Evaluates Q-Learning model by computing the temporal difference error.

    delta(0) = R(s') + \gamma \max _{a'} Q_{k}(s', a') - Q_{k}(s | a)
    """
    # Get train error
    delta_train = []
    for sample in data_train:
        state, action, next_state, reward = sample
        action_index = q_model.get_action_index(action)
        delta_train.append(np.abs(
            reward
            + gamma * np.max(q_model.estimate_expected_return(next_state))
            - q_model.estimate_expected_return(state)[action_index]
        ))
    delta_train = np.mean(delta_train)

    # Get test error
    delta_test = []
    for sample in data_test:
        state, action, next_state, reward = sample
        action_index = q_model.get_action_index(action)
        delta_test.append(np.abs(
            reward
            + gamma * np.max(q_model.estimate_expected_return(next_state))
            - q_model.estimate_expected_return(state)[action_index]
        ))
    delta_test = np.mean(delta_test)

    return delta_train, delta_test


def plot_progress(td_errors):
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 700 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(1, 1)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    # Plot data
    axes[0].plot(
        td_errors[:, 0], td_errors[:, 1], marker='x', color='black',
        label='train')
    axes[0].plot(
        td_errors[:, 0], td_errors[:, 2], marker='x', color='blue',
        label='test')

    # Labelling
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('TD error')
    plt.legend()

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/S1_temporal_difference_error.pdf', bbox_inches='tight')
    plt.savefig(
        directory + '/S1_temporal_difference_error.tif', bbox_inches='tight')


def define_dosing_regimen(doses):
    """
    Returns a myokit.Protocol.
    """
    duration = 0.01
    dose_rates = np.array(doses) / duration
    regimen = myokit.Protocol()
    for idx, dr in enumerate(dose_rates):
        if dr == 0:
            continue
        regimen.add(myokit.ProtocolEvent(
            level=dr,
            start=idx*24,
            duration=duration))

    return regimen


def check_performance(q_model, model, parameters, target):
    """
    Uses trained policy to predict doses.
    """
    rng = np.random.default_rng(1)
    times = np.arange(30) * 24
    response = np.empty(len(times))

    doses = []
    # Simulate treatment response
    for idt, time in enumerate(times):
        # Simulate model
        inr = model.sample(
            parameters=parameters, times=[time], seed=rng,
            return_df=False)[0, 0, 0]

        # Round INR to steps of 0.2 to reduce state space
        inr = np.round(inr * 5) / 5

        # Predict dose
        dose = q_model.predict_action(inr)
        doses.append(dose)
        regimen = define_dosing_regimen(doses)
        model.set_dosing_regimen(regimen)

        response[idt] = inr

    plot_response(times, response, doses, target)


def plot_response(times, response, doses, target):
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 700 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(2, 1)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    # Plot data
    axes[0].plot(
        times / 24, response, marker='x', color='black')
    axes[1].plot(
        times / 24, doses, marker='x', color='black')
    axes[0].axhline(target, linestyle='dashed', color='blue')

    # Labelling
    axes[1].set_xlabel('Time in days')
    axes[0].set_ylabel('INR measurements')
    axes[1].set_ylabel('Doses in mg')
    axes[1].set_ylim([-1, 31])

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/S2_response.pdf', bbox_inches='tight')
    plt.savefig(
        directory + '/S2_response.tif', bbox_inches='tight')



if __name__ == '__main__':
    target = 2.5
    model, parameters = define_model()
    data_train, data_test = generate_data(
        model, parameters, target=target, epsilon=0.05, n_episodes=200)
    q_model = train_q_learning_algorithm(
        data_train, data_test, n_epochs=10, gamma=0.5)
    check_performance(q_model, model, parameters, target)