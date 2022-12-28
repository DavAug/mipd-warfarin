import os

import chi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import myokit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Defines deep Q Learning network.
    """
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        # Define action bins
        self._doses = np.array([
            0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
            10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,
            16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5,
            23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29,
            29.5, 30
        ])

        # Define dummies to reverse standardisation of INR values
        self._mean_inr = None
        self._mean_std = None

    def forward(self, state):
        """
        Returns an estimate of the action-value function, Q(a | s).

        Assumes that the state has been appropriately scaled.

        State is of shape (batch, 2), where the rows encode the INR
        measurement, the CYP2C9 genotype.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        q = self.layer3(x)

        return q

    def get_action_index(self, action):
        """
        Returns index of state.
        """
        if (action < self._doses[0]) or (action > self._doses[-1]):
            raise ValueError('action is invalid.')

        for index, dose in enumerate(self._doses):
            if action <= dose:
                return index

    def set_inr_scale(self, mean, std):
        """
        Defines the scale of INR values.

        Is used to translate INR values into inputs to the network.
        """
        self._mean_inr = mean
        self._std_inr = std

    def predict_dose(self, state):
        """
        Returns the predicted dose.

        Assumes that state is not rescaled.
        """
        if self._mean_inr is None:
            raise ValueError('INR scaling has not been set.')

        # Scale state
        state[:, 0] = (state[:, 0] - self._mean_inr) / self._std_inr

        with torch.no_grad():
            q = self.forward(state)
            dose = self._doses[q.max(1)[1]]

        return dose


def define_model():
    """
    Returns the data-generating model and its parameters.
    """
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

    # Define population model
    ke_cov_model = chi.CovariatePopulationModel(
        chi.LogNormalModel(), chi.LinearCovariateModel(n_cov=1))
    ke_cov_model.set_population_parameters([(0, 0)])
    population_model = chi.ComposedPopulationModel([
        chi.LogNormalModel(n_dim=1),
        ke_cov_model,
        chi.LogNormalModel(n_dim=1),
        chi.PooledModel(n_dim=3)
    ])

    # Define predictive model
    model = chi.PredictiveModel(model, chi.LogNormalErrorModel())
    model = chi.PopulationPredictiveModel(model, population_model)

    # Define parameters
    parameters = [
        2.660,
        0.071,
        -3.716,
        0.054,
        2,
        1.411,
        0.240,
        0.1,
        0.02,
        0.1
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
    covariates = np.zeros(n_episodes)
    covariates[n_episodes // 2:] = 1

    data = np.empty((n_episodes, n_days, 5))
    rng = np.random.default_rng(seed=0)
    for idd in range(n_episodes):
        doses = []

        # Simulate treatment response
        for idt in range(n_days):
            # Simulate model
            # NOTE: Important to simulate all time points and to keep the seed
            # the same, so the same individual is samples and the measurements
            # also stay the same
            time = times[:idt+1]
            inr = model.sample(
                parameters=parameters, times=time, seed=idd,
                covariates=covariates[idd:idd+1], return_df=False)[0, -1, 0]

            # Adjust dosing regimen
            regimen, doses = adjust_dosing_regimen(
                doses, inr, target, epsilon=epsilon, seed=rng)
            model.set_dosing_regimen(regimen)

            # Compute reward
            reward = -(target - inr)**2
            if inr < 1.5:
                reward = -10

            # Store data (state, action, next state, reward)
            data[idd, idt, 0] = inr
            data[idd, idt, 1] = doses[-1]
            if idt > 0:
                data[idd, idt-1, 2] = inr
                data[idd, idt-1, 3] = reward

        # Remember covariate
        data[idd, :, 4] = covariates[idd]

    # The last day has no next state and reward measurement. So we can throw
    # this column out
    data = data[:, :-1, :]
    data = data.reshape((n_episodes * (n_days-1), 5))

    return data


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


def format_data(data, policy_net, target_net, target, rng, device):
    """
    Normalise data and split into train and test set.

    inr: needs to be standardised: inr <- (inr - target) / target.
    action: Replace doses by bin number.
    reward: Normalise rewards
    """
    # Normalise INR values
    mean = np.mean(data[:, 0])
    std = np.std(data[:, 0], ddof=1)
    data[:, 0] = (data[:, 0] - target) / target
    data[:, 2] = (data[:, 2] - target) / target

    # Make networks remember this scale
    policy_net.set_inr_scale(target, target)
    target_net.set_inr_scale(target, target)

    # Normalise reward
    mean = np.mean(data[:, 3])
    std = np.std(data[:, 3], ddof=1)
    data[:, 3] = (data[:, 3] - mean) / std

    # Enumerate doses
    indices = [policy_net.get_action_index(d) for d in data[:, 1]]
    data[:, 1] = np.array(indices)

    # Shuffle, format as tensor and split into train and test
    rng.shuffle(data, axis=0)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    split = int(len(data) * 0.9)
    train = data[:split]
    test = data[split:]

    return train, test


def train_q_learning_algorithm(
        data_train, data_test, batch_size, device, n_epochs=10, gamma=0.9,
        model=None, parameters=None, target=None, scheduler=None):
    """
    Returns a trained Q-Learning algorithm.
    """
    performance = np.zeros((n_epochs, 4))
    best = [np.inf, 0, None, None]  # TD test error, epoch, response, model
    for epoch in range(n_epochs):
        # Perform one step of the optimization (on the policy network)
        train_loss = train_epoch(
            data_train, policy_net, target_net, batch_size, optimiser, gamma,
            tau, rng, device)
        performance[epoch, 1] = train_loss

        # Evaluate on test set and simulation
        test_loss, mse, best = evaluate(
            data_test, policy_net, target_net, batch_size, device, gamma,
            model, parameters, target, best, epoch)
        performance[epoch, 2] = test_loss
        performance[epoch, 3] = mse

        # Decay learning rate
        if scheduler is not None:
            scheduler.step(test_loss)

    performance[:, 0] = np.arange(n_epochs) + 1
    best_epoch = best[1]
    best_response = best[2]
    plot_progress(performance, best_epoch)
    plot_response(best_response, target)

    best_model_parameters = best[3]

    return best_model_parameters


def train_epoch(
        data, policy_net, target_net, batch_size, optimiser, gamma, tau, rng,
        device):
    # Shuffle data
    avg_loss = 0
    n_data = len(data)
    indices = np.arange(n_data)
    rng.shuffle(indices)
    data = data[indices]

    for idb in range(n_data // batch_size):
        batch = data[idb * batch_size: (idb + 1) * batch_size]
        loss = train_batch(
            batch, policy_net, target_net, optimiser, gamma, device)
        avg_loss = (avg_loss * idb + loss) / (idb + 1)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = \
                policy_net_state_dict[key] * tau \
                + target_net_state_dict[key] * (1 - tau)
        target_net.load_state_dict(target_net_state_dict)

    return avg_loss


def train_batch(batch, policy_net, target_net, optimiser, gamma, device):
    state_batch = batch[:, [0, 4]]
    action_batch = batch[:, 1:2].type(torch.int64)
    next_state_batch = batch[:, [2, 4]]
    reward_batch = batch[:, 3:4]

    # Estimate Q(a | s) using policy net (continuously updated)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s') = max_a Q(a | s') for next state using target net
    # (slowly updated)
    next_state_values = torch.zeros(state_action_values.shape, device=device)
    with torch.no_grad():
        next_state_values[:, 0] = target_net(next_state_batch).max(1)[0]

    # Estimate Q(a | s) from reward R(s') and max_a Q(a | s')
    expected_state_action_values = gamma * next_state_values + reward_batch

    # Quantify TD error using Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values, expected_state_action_values)

    # Optimize the model
    optimiser.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimiser.step()

    return loss


def evaluate(
        data, policy_net, target_net, batch_size, device, gamma, model,
        parameters, target, best, epoch):
    """
    Evaluates Q-Learning model by computing the temporal difference error.

    delta(0) = R(s') + \gamma \max _{a'} Q_{k}(s', a') - Q_{k}(s | a)
    """
    loss = evaluate_test_set(
        data, policy_net, target_net, batch_size, gamma, device)

    # Compute MSE to target for a simulation run
    mse, response = get_mse_to_target_of_simulation_run(
        policy_net, model, parameters, target, device)

    if loss < best[0]:
        # This model performs better on the test set than in the previous
        # iterations
        best[0] = loss
        best[1] = epoch
        best[2] = response
        best[3] = policy_net.state_dict()

    return loss, mse, best

def evaluate_test_set(data, policy_net, target_net, batch_size, gamma, device):
    n_data = len(data)
    avg_loss = 0
    for idb in range(n_data // batch_size):
        batch = data[idb * batch_size: (idb + 1) * batch_size]
        state_batch = batch[:, [0, 4]]
        action_batch = batch[:, 1:2].type(torch.int64)
        next_state_batch = batch[:, [2, 4]]
        reward_batch = batch[:, 3:4]

        with torch.no_grad():
            state_action_values = \
                policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(
                state_action_values.shape, device=device)
            next_state_values[:, 0] = target_net(next_state_batch).max(1)[0]
            expected_state_action_values = \
                gamma * next_state_values + reward_batch

            # Quantify TD error using Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

        avg_loss = (avg_loss * idb + loss) / (idb + 1)

    return loss


def get_mse_to_target_of_simulation_run(
        policy_net, model, parameters, target, device):

    n_ids = 6
    times = np.arange(30) * 24
    response = np.empty((len(times), 4, n_ids))
    response[:, 0, 0] = times

    seed = 3
    for rep in range(n_ids):
        cov = rep % 2
        doses = []
        # Simulate treatment response
        for idt in range(30):
            time = times[:idt+1]

            # Simulate model
            inr = model.sample(
                parameters=parameters, times=time, return_df=False,
                seed=seed+rep, covariates=[cov])[0, -1, 0]

            # Predict dose
            state = torch.tensor(
                data=[[inr, cov]], dtype=torch.float32, device=device)
            dose = policy_net.predict_dose(state)
            dose = float(dose)
            doses.append(dose)
            regimen = define_dosing_regimen(doses)
            model.set_dosing_regimen(regimen)

            response[idt, 1, rep] = inr
            response[idt, 2, rep] = dose

        response[:, 3, rep] = cov

    # Compute MSE to target
    mse = np.mean((target - response[:, 1])**2)

    return mse, response


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


def plot_progress(performance, best_epoch):
    # We use enumerate to count epochs (starting from 0), so we are one epoch
    # behind
    best_epoch += 1

    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 1000 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(2, 1)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    axes[0].sharex(axes[1])

    # Plot data
    axes[0].plot(
        performance[:, 0], performance[:, 1], marker='x', color='black',
        label='train')
    axes[0].plot(
        performance[:, 0], performance[:, 2], marker='x', color='blue',
        label='test')
    axes[0].axvline(
        best_epoch, linestyle='dashed', color='blue', label='best test')
    axes[1].plot(
        performance[:, 0], performance[:, 3], marker='x', color='black',
        label='simulation')

    # Labelling
    axes[0].set_ylabel('TD error')
    axes[1].set_ylabel('MSE of INR to target')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].legend()

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/S7_dqn_training_iiv.pdf', bbox_inches='tight')
    plt.savefig(
        directory + '/S7_dqn_training_iiv.tif', bbox_inches='tight')


def plot_response(response, target):
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 1000 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(2, 1)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    # Plot data
    _, _, n_ids = response.shape
    colors = ['black', 'blue']
    cov = ['Cov. 1', 'Cov. 2']
    axes[0].axhline(target, linestyle='dashed', color='black', label='target')
    for idr in range(n_ids):
        label = None
        if idr in [0, 1]:
            label = cov[int(response[0, 3, idr])]
        axes[0].plot(
            response[:, 0, 0] / 24, response[:, 1, idr], marker='x',
            label=label,
            color=colors[int(response[0, 3, idr])])
        axes[1].plot(
            response[:, 0, 0] / 24, response[:, 2, idr], marker='x',
            color=colors[int(response[0, 3, idr])])

    # Labelling
    axes[0].set_ylabel('INR')
    axes[1].set_ylabel('Dose in mg')
    axes[1].set_xlabel('Time in days')
    axes[1].set_ylim([-1, 31])
    axes[0].legend()

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/S8_dqn_best_response_iiv.pdf',
        bbox_inches='tight')
    plt.savefig(
        directory + '/S8_dqn_best_response_iiv.tif',
        bbox_inches='tight')


if __name__ == '__main__':
    batch_size = 128
    gamma = 0.5
    tau = 0.005
    lr = 1e-5

    rng = np.random.default_rng(35)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define networks
    # NOTE: DQN uses 2 copies of the same network: Both networks are used
    # to predict the state-action value, but one (policy net) is trained
    # continuously and used to estimate Q(a | s). The other one is used to
    # estimate Q(a | s') and is only gradually updated each
    # batch using a weighted average of the two networks' weights. The network
    # can then be trained by minimising the temporal difference error
    # R(s') + Q(a | s') - Q(a | s). Predicting Q(a | s') with a more slowly
    # updated network stabilises the training.
    n_states = 2
    n_actions = len([
        0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
        10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,
        16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5,
        23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29,
        29.5, 30])
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Define optimiser
    optimiser = optim.Adam(policy_net.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 'min', patience=3)

    target = 2.5
    model, parameters = define_model()
    data = generate_data(
        model, parameters, target=target, epsilon=0.1, n_episodes=1500)
    train, test = format_data(
        data, policy_net, target_net, target, rng, device)
    q_model = train_q_learning_algorithm(
        train, test, batch_size, n_epochs=30, gamma=gamma, device=device,
        model=model, parameters=parameters, target=target, scheduler=scheduler)

    # Save model
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/models/dqn_hamberg_model_iiv.pickle'
    torch.save(q_model, directory + filename)
