import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


def load_data(target):
    """
    Returns the clinical phase II and III data.

    Data is formatted into state, action, next state, reward tuples.

    Actions where the next state was not recorded are disregarded. The reward
    is defined by the negative mean squared error of the INR to the target.
    """
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_1 = pd.read_csv(directory + '/data/trial_phase_II.csv')
    data_2 = pd.read_csv(directory + '/data/trial_phase_III.csv')

    data_1 = reshape_data(data_1, target)
    data_2 = reshape_data(data_2, target)

    return data_2, data_1


def reshape_data(df, target):
    """
    Reshapes data into state, action, next state, reward tuples.

    We don't always have a measurement of the INR at the next time step (24h).
    In those cases we can nevertheless update the policy network by attributing
    the reward associated with the next available measurement equally to all
    intermediate steps. I.e. if we start in state s_0 and peform action a_0,
    and we only know the state s_t with reward r_t, we associate the reward
    r_t / t with the action a_0.

    In the q learning algorithm the q update then becomes

        q(a_0 | s_0) <- q(a_0 | s_0) + sum_(k=1)^(t-1)gamma^k r_t / t
                        + g^t max_a q(a | s_t)
    """
    # Reshape phase II data
    ids = df.ID.dropna().unique()
    n_states = len(df[
        (df.ID == ids[0]) & (df.Observable == 'INR')].Value.dropna())
    data = np.zeros(shape=(len(ids), n_states, 11))
    for idx, _id in enumerate(ids):
        # Get info from dataframe
        temp = df[df.ID == _id]
        times = temp[temp.Observable == 'INR'].Time.values
        states = temp[temp.Observable == 'INR'].Value.values
        actions = [
            temp[temp.Time == t].Dose.dropna().values[0] for t in times[:-1]]
        vkorc1 = temp[temp.Observable == 'VKORC1'].Value.values
        cyp = temp[temp.Observable == 'CYP2C9'].Value.values
        age = temp[temp.Observable == 'Age'].Value.values

        # Get time steps between states
        steps = np.array([
            times[idt+1] - times[idt] for idt in range(len(times)-1)]) // 24

        # Compute reward
        # We penalise too large INRs (>3) linearly in the distance to 3.
        rewards = np.empty(shape=states.shape)
        mask = (states + 0.5) < target
        rewards[mask] = 0  # INR is too small
        mask = np.abs(states - target) <= 0.5
        rewards[mask] = 1  # INR is exactly right
        mask = (states - 0.5) > target
        rewards[mask] = target - 0.5 - states[mask]  # INR is too large

        # Fill container
        data[idx, :, 0] = states                    # State
        data[idx, :-1, 1] = np.array(actions)       # Action
        data[idx, :-1, 2] = states[1:]              # Next state
        data[idx, :-1, 3] = rewards[1:]             # Rewards
        data[idx, :-1, 4] = steps                   # Time between states

        # We implement genetic factors by counting allele variants
        if vkorc1 == 0:
            data[idx, :, 5] = 1
        elif vkorc1 == 1:
            data[idx, :, 5] = 0.5
            data[idx, :, 6] = 0.5
        else:
            data[idx, :, 6] = 1
        if cyp == 0:
            data[idx, :, 7] = 1
        elif cyp == 1:
            data[idx, :, 7] = 0.5
            data[idx, :, 8] = 0.5
        elif cyp == 2:
            data[idx, :, 7] = 0.5
            data[idx, :, 9] = 0.5
        elif cyp == 3:
            data[idx, :, 8] = 1
        elif cyp == 4:
            data[idx, :, 8] = 0.5
            data[idx, :, 9] = 0.5
        else:
            data[idx, :, 9] = 1

        # Add age
        data[idx, :, 10] = age

    # Remove last column of dataset, because the next state / reward is unknown
    data = data[:, :-1]
    data = data.reshape((len(ids) * (n_states - 1), 11))

    return data


def format_data(train, test, policy_net, target_net, target, device):
    """
    Normalise data and split into train and test set.

    inr: needs to be standardised: inr <- (inr - target) / target.
    action: Replace doses by bin number.
    reward: Normalise rewards
    """
    # Normalise INR values
    train[:, 0] = (train[:, 0] - target) / (3 * target)
    train[:, 2] = (train[:, 2] - target) / (3 * target)
    test[:, 0] = (test[:, 0] - target) / (3 * target)
    test[:, 2] = (test[:, 2] - target) / (3 * target)

    # Make networks remember this scale
    policy_net.set_inr_scale(target, 3 * target)
    target_net.set_inr_scale(target, 3 * target)

    # Normalise age
    mean = 51
    std = 15
    train[:, -1] = (train[:, -1] - mean) / (3 * std)
    test[:, -1] = (test[:, -1] - mean) / (3 * std)

    # Make networks remember this scale
    policy_net.set_age_scale(mean, 3 * std)
    target_net.set_age_scale(mean, 3 * std)

    # Enumerate doses
    indices = [policy_net.get_action_index(d) for d in train[:, 1]]
    train[:, 1] = np.array(indices)
    indices = [policy_net.get_action_index(d) for d in test[:, 1]]
    test[:, 1] = np.array(indices)

    # Normalise rewards
    mean = np.mean(train[:,3])
    std = np.std(train[:,3], ddof=1)
    train[:, 3] = (train[:, 3] - mean) / (3 * std)
    test[:, 3] = (test[:, 3] - mean) / (3 * std)

    # Convert to torch tensor
    train = torch.tensor(train, dtype=torch.float32, device=device)
    test = torch.tensor(test, dtype=torch.float32, device=device)

    return train, test


def train_q_learning_algorithm(
        data_train, data_test, batch_size, device, n_epochs=10, gamma=0.9,
        scheduler=None):
    """
    Returns a trained Q-Learning algorithm.
    """
    performance = np.zeros((n_epochs, 3))
    best = [np.inf, 0, None]  # TD test error, epoch, model
    for epoch in range(n_epochs):
        # Perform one step of the optimization (on the policy network)
        train_loss = train_epoch(
            data_train, policy_net, target_net, batch_size, optimiser, gamma,
            tau, rng, device)
        performance[epoch, 1] = train_loss

        # Evaluate on test set
        test_loss, best = evaluate(
            data_test, policy_net, target_net, batch_size, device, gamma,
            best, epoch)
        performance[epoch, 2] = test_loss

        # Decay learning rate
        if scheduler is not None:
            scheduler.step(test_loss)

    performance[:, 0] = np.arange(n_epochs) + 1
    best_epoch = best[1]
    plot_progress(performance, best_epoch)

    best_model_parameters = best[2]
    latest_model_parameters = policy_net.state_dict()

    return best_model_parameters, latest_model_parameters


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
    state_batch = batch[:, [0, 5, 6, 7, 8, 9, 10]]
    action_batch = batch[:, 1:2].type(torch.int64)
    next_state_batch = batch[:, [2, 5, 6, 7, 8, 9, 10]]
    reward_batch = batch[:, 3:4]
    steps = batch[:, 4:5]

    # Estimate Q(a | s) using policy net (continuously updated)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s') = max_a Q(a | s') for next state using target net
    # (slowly updated)
    next_state_values = torch.zeros(state_action_values.shape, device=device)
    with torch.no_grad():
        next_state_values[:, 0] = target_net(next_state_batch).max(1)[0]

    # Estimate Q(a | s) from reward R(s') and max_a Q(a | s')
    expected_state_action_values = torch.zeros(
        state_action_values.shape, device=device)
    for s in steps.unique():
        mask = steps == s
        weighted_gamma = torch.sum(gamma ** torch.arange(start=0, end=s))
        expected_state_action_values[mask] = \
            reward_batch[mask] / s * weighted_gamma \
            + gamma ** s * next_state_values[mask]

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
        data, policy_net, target_net, batch_size, device, gamma, best, epoch):
    """
    Evaluates Q-Learning model by computing the temporal difference error.

    delta(0) = R(s') + \gamma \max _{a'} Q_{k}(s', a') - Q_{k}(s | a)
    """
    loss = evaluate_test_set(
        data, policy_net, target_net, batch_size, gamma, device)

    if loss < best[0]:
        # This model performs better on the test set than in the previous
        # iterations
        best[0] = loss
        best[1] = epoch
        best[2] = policy_net.state_dict()

    return loss, best

def evaluate_test_set(data, policy_net, target_net, batch_size, gamma, device):
    n_data = len(data)
    avg_loss = 0
    for idb in range(n_data // batch_size):
        batch = data[idb * batch_size: (idb + 1) * batch_size]
        state_batch = batch[:, [0, 5, 6, 7, 8, 9, 10]]
        action_batch = batch[:, 1:2].type(torch.int64)
        next_state_batch = batch[:, [2, 5, 6, 7, 8, 9, 10]]
        reward_batch = batch[:, 3:4]
        steps = batch[:, 4:5]

        with torch.no_grad():
            state_action_values = \
                policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(
                state_action_values.shape, device=device)
            next_state_values[:, 0] = target_net(next_state_batch).max(1)[0]
            expected_state_action_values = torch.zeros(
                state_action_values.shape, device=device)
            for s in steps.unique():
                mask = steps == s
                weighted_gamma = torch.sum(
                    gamma ** torch.arange(start=0, end=s))
                expected_state_action_values[mask] = \
                    reward_batch[mask] / s * weighted_gamma \
                    + gamma ** s * next_state_values[mask]

            # Quantify TD error using Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

        avg_loss = (avg_loss * idb + loss) / (idb + 1)

    return loss


def plot_progress(performance, best_epoch):
    # We use enumerate to count epochs (starting from 0), so we are one epoch
    # behind
    best_epoch += 1

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
        performance[:, 0], performance[:, 1], color='black',
        label='train')
    axes[0].plot(
        performance[:, 0], performance[:, 2], color='blue',
        label='test')
    axes[0].axvline(
        best_epoch, linestyle='dashed', color='blue', label='best test')

    # Labelling
    axes[0].set_ylabel('TD error')
    axes[0].set_xlabel('Epoch')
    axes[0].set_xscale('log')
    axes[0].legend()

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/1_dqn_training.pdf', bbox_inches='tight')
    plt.savefig(
        directory + '/1_dqn_training.tif', bbox_inches='tight')


if __name__ == '__main__':
    batch_size = 128
    gamma = 0.9
    tau = 0.005
    lr = 1e-4

    rng = np.random.default_rng(10)
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
    policy_net = DQN(1024).to(device)
    target_net = DQN(1024).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Define optimiser
    optimiser = optim.Adam(policy_net.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 'min', patience=10, min_lr=1e-7, factor=0.5)

    target = 2.5
    train, test = load_data(target=target)
    train, test = format_data(
        train, test, policy_net, target_net, target, device)
    best, latest = train_q_learning_algorithm(
        train, test, batch_size, n_epochs=100, gamma=gamma, device=device,
        scheduler=scheduler)

    # Save model
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/models/dqn_model_best.pickle'
    torch.save(best, directory + filename)
    filename = '/models/dqn_model_latest.pickle'
    torch.save(latest, directory + filename)

    # TODO: Use G alleses and 1 and 2 alleles as genetic features. A and 3 are
    # not necessary.
