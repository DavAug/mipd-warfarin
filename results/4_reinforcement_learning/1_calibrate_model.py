import os
import subprocess

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


def train_dqn_model():
    """
    Returns a trained DQN model.

    The DQN model is trained on data simulated by a PKPD model. During
    the n_epochs of training the data is updated n_buffer_refresh times.

    The data contains the treatment response measurement of n_buffer
    simulated individuals treated according to the epsilon-greedy policy of
    the current DQN model.
    """
    buffer = None
    performance = np.zeros((N_EPOCHS, 3))
    n_epochs_per_buffer = N_EPOCHS // N_BUFFER_REFRESH
    for epoch in range(N_EPOCHS):
        # Alternate policy and target network every epoch (double Q learning)
        policy_net = [DQN_MODEL_1, DQN_MODEL_2][epoch % 2]
        target_net = [DQN_MODEL_1, DQN_MODEL_2][(epoch + 1) % 2]
        optimiser = [optimiser_1, optimiser_2][epoch % 2]

        # Refresh buffer
        if (epoch % n_epochs_per_buffer) == 0:
            buffer = generate_data(policy_net, buffer)

            # Keep track of average reward
            performance[epoch:epoch+n_epochs_per_buffer, 2] = \
                torch.mean(buffer[:, 3]).numpy()

            print('Epoch %d: %f Avg. reward' % (epoch, performance[epoch, 2]))

        # Perform one step of the optimisation (on the policy network)
        train_loss = train_epoch(
            buffer, policy_net, target_net, optimiser, rng)
        performance[epoch, 1] = train_loss

        print('Epoch %d: %f Avg. loss' % (epoch, performance[epoch, 1]))

    performance[:, 0] = np.arange(N_EPOCHS) + 1
    plot_progress(performance)

    return policy_net.state_dict(), performance


def generate_data(dqn_model, buffer):
    """
    Returns treatment response of n_ids individuals treated according to
    the current policy.
    """
    # In the first iteration, sample full buffer
    n_ids = N_IDS_PER_REFRESH
    if buffer is None:
        n_ids = BUFFER_SIZE

    # Sample covariates (VKORC1, CYP2C9, Age, VKORC1)
    # NOTE: To make training more efficient, we sample the genotypes uniformly,
    # thereby avoiding the class imbalance that otherwise occurs in clinical
    # practice.
    covariates = np.empty((n_ids, 4))
    covariates[:, 0] = np.random.choice([0, 1, 2], size=n_ids, replace=True)
    covariates[:, 3] = covariates[:, 0]
    covariates[:, 1] = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=n_ids, replace=True)
    covariates[:, 2] = np.random.lognormal(
        mean=np.log(51), sigma=0.15, size=n_ids)

    # Predict policies for individuals
    # NOTE: We do this in advance for efficiency.
    policy, formatted_covs = get_policy(dqn_model, covariates)
    print(policy[0])

    # Simulate treatment response
    data = simulate_buffer(covariates, policy)
    data = format_data(data, formatted_covs, dqn_model)

    # Return samples
    if buffer is None:
        return data

    # Refresh buffer with new samples
    n_buffer = len(buffer)
    indices = np.arange(n_buffer)
    rng.shuffle(indices)
    buffer = buffer[indices]
    n_data = len(data)
    buffer[:n_data] = data

    return buffer


def get_policy(dqn_model, covariates):
    """
    Returns policy for each individual.
    """
    inrs = np.arange(0, 5, 0.1)
    covariates = format_covariates(covariates)

    states = torch.empty((len(inrs), 1 + covariates.shape[1]))
    states[:, 0] = torch.Tensor(inrs)
    policy = np.empty((len(covariates), len(inrs), 2))
    for idx, cov in enumerate(covariates):
        policy[idx, :, 0] = inrs
        states[:, 1:] = torch.Tensor(cov)[None, :].expand(
            len(inrs), covariates.shape[1])
        policy[idx, :, 1] = dqn_model.predict_dose(states)

    return policy, covariates


def format_covariates(covariates):
    """
    Returns a (n_ids, n_cov) torch.Tensor with the number of G VKORC1 alleles,
    the number of *1 CYP2C9 allese, the number of *2 CYP2C9 alleles and the
    age.
    """
    cov = np.zeros((len(covariates), 4))
    vkorc1 = covariates[:, 0]
    mask = vkorc1 == 0
    cov[mask, 0] = 1
    mask = vkorc1 == 1
    cov[mask, 0] = 0.5

    cyp = covariates[:, 1]
    mask = cyp == 0
    cov[mask, 1] = 1
    mask = cyp == 1
    cov[mask, 1] = 0.5
    cov[mask, 2] = 0.5
    mask = cyp == 2
    cov[mask, 1] = 0.5
    mask = cyp == 3
    cov[mask, 2] = 1
    mask = cyp == 4
    cov[mask, 2] = 0.5

    cov[:, 3] = covariates[:, 2]

    return cov


def simulate_buffer(covariates, policy):
    """
    Simulates treatement responses.
    """
    # Temporaily save covariates and policy to disk
    directory = os.path.dirname(os.path.abspath(__file__))
    np.save(directory + '/covariates.temp.npy', covariates)
    np.save(directory + '/policy.temp.npy', policy)

    # Simulate treatment response
    print('Refreshing buffer... ')
    filename = directory + '/buffer.temp.npy'
    subprocess.Popen([
        'python',
        os.path.dirname(directory) +
        '/2_semi_mechanistic_model/8_simulate_buffer.py',
        '--filename',
        filename
    ]).wait()
    print('Buffer refreshed.')

    # Load buffer
    data = np.load(filename)

    # Delete temporary files
    subprocess.Popen(['rm', directory + '/covariates.temp.npy']).wait()
    subprocess.Popen(['rm', directory + '/policy.temp.npy']).wait()
    subprocess.Popen(['rm', filename]).wait()

    return data


def format_data(data, covariates, dqn_model):
    """
    Formats the buffer.
    """
    # Hyperparameters
    target_inr = 2.5

    # Reshape phase II data
    n_ids, n_days, _ = data.shape
    formatted_data = np.empty(shape=(n_ids, n_days, 8))

    # Compute reward
    # We penalise too large INRs (>3) linearly in the distance to 3.
    inrs = data[:, :, 0]
    doses = data[:, :, 1]
    # rewards = np.empty(shape=inrs.shape)
    # mask = (inrs + 0.5) < target_inr
    # rewards[mask] = 0  # INR is too small
    # mask = np.abs(inrs - target_inr) <= 0.5
    # rewards[mask] = 1  # INR is exactly right
    # mask = (inrs - 0.5) > target_inr
    # rewards[mask] = target_inr - 0.5 - inrs[mask]  # INR is too large
    rewards = -(inrs - target_inr)**2 / target_inr**2

    # Fill container
    formatted_data[:, :, 0] = inrs                         # State
    formatted_data[:, :, 1] = doses                        # Action
    formatted_data[:, :-1, 2] = inrs[:, 1:]                # Next state
    formatted_data[:, :-1, 3] = rewards[:, 1:]             # Rewards

    for idc, cov in enumerate(covariates):
        formatted_data[idc, :, 4] = cov[0]
        formatted_data[idc, :, 5] = cov[1]
        formatted_data[idc, :, 6] = cov[2]
        formatted_data[idc, :, 7] = cov[3]

    # Remove last column of dataset, because the next state / reward is unknown
    data = formatted_data[:, :-1].reshape((n_ids * (n_days - 1), 8))

    # Normalise data
    data[:, 0] = (data[:, 0] - dqn_model._mean_inr) / dqn_model._std_inr
    data[:, 1] = np.array([dqn_model.get_action_index(d) for d in data[:, 1]])
    data[:, 2] = (data[:, 2] - dqn_model._mean_inr) / dqn_model._std_inr
    data[:, 7] = (data[:, 7] - dqn_model._mean_age) / dqn_model._std_age

    return torch.tensor(data, dtype=torch.float32, device=DEVICE)


def train_epoch(data, policy_net, target_net, optimiser, rng):
    # Shuffle data
    avg_loss = 0
    n_data = len(data)
    indices = np.arange(n_data)
    rng.shuffle(indices)
    data = data[indices]

    for idb in range(n_data // BATCH_SIZE):
        batch = data[idb * BATCH_SIZE: (idb + 1) * BATCH_SIZE]
        loss = train_batch(batch, policy_net, target_net, optimiser)
        avg_loss = (avg_loss * idb + loss) / (idb + 1)

    return avg_loss


def train_batch(batch, policy_net, target_net, optimiser):
    state_batch = batch[:, [0, 4, 5, 6, 7]]
    action_batch = batch[:, 1:2].type(torch.int64)
    next_state_batch = batch[:, [2, 4, 5, 6, 7]]
    reward_batch = batch[:, 3:4]

    # Estimate Q(a | s) using policy net (continuously updated)
    action_values = policy_net(state_batch).gather(1, action_batch)

    # Estimate argmax _a' Q(a' | s') using policy net (continuously updated)
    next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)

    # Select Q value estimated by target net using max action from policy net
    # (Double Q learning)
    with torch.no_grad():
        next_action_values = target_net(next_state_batch).gather(
            1, next_actions)

    # Estimate Q(a | s) from reward R(s') and max_a Q(a | s')
    target_action_values = reward_batch + GAMMA * next_action_values

    # Quantify TD error using Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(action_values, target_action_values)

    # Optimize the model
    optimiser.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimiser.step()

    return loss


def plot_progress(performance):
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 1000 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(2, 1, hspace=0.05)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    # Plot data
    axes[0].plot(
        performance[::2, 0], performance[::2, 1], color='black',
        label='DQN 1', linewidth=3)
    axes[0].plot(
        performance[1::2, 0], performance[1::2, 1],
        label='DQN 2', linewidth=3, linestyle='--')
    axes[1].plot(
        performance[:, 0], performance[:, 2] * 2.5**2, color='black',
        linewidth=3)

    # Labelling
    axes[0].set_ylabel('TD error')
    axes[0].set_xticklabels([], visible=False)
    axes[1].set_ylabel('Average reward')
    axes[1].set_xlabel('Epoch')

    axes[0].legend()

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/1_dqn_training.pdf', bbox_inches='tight')
    plt.savefig(
        directory + '/1_dqn_training.tif', bbox_inches='tight')


if __name__ == '__main__':
    BATCH_SIZE = 128
    GAMMA = 0.9
    LR = 1e-4
    N_EPOCHS = 1500
    N_BUFFER_REFRESH = 75
    N_IDS_PER_REFRESH = 100
    BUFFER_SIZE = 1000  # in IDs

    rng = np.random.default_rng(35)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define networks
    DQN_MODEL_1 = DQN(256).to(DEVICE)
    DQN_MODEL_2 = DQN(256).to(DEVICE)
    DQN_MODEL_1.set_inr_scale(2.5, 2.5)
    DQN_MODEL_2.set_inr_scale(2.5, 2.5)
    DQN_MODEL_1.set_age_scale(51, 15)
    DQN_MODEL_2.set_age_scale(51, 15)

    # Define optimisers
    torch.manual_seed(1)
    optimiser_1 = optim.Adam(DQN_MODEL_1.parameters(), lr=LR, amsgrad=False)
    optimiser_2 = optim.Adam(DQN_MODEL_2.parameters(), lr=LR, amsgrad=False)

    model, performance = train_dqn_model()

    # Save model
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/models/dqn_model.pickle'
    torch.save(model, directory + filename)
    filename = '/models/dqn_model_performance.npy'
    np.save(directory + filename, performance)
