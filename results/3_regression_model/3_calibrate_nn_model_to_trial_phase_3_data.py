import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model import MaintenanceDoseNetwork


def load_data():
    """
    Returns the clinical phase III data.

    Data is formatted into state, action, next state, reward tuples.

    Actions where the next state was not recorded are disregarded. The reward
    is defined by the negative mean squared error of the INR to the target.
    """
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(directory + '/data/trial_phase_III.csv')
    data = reshape_data(data)

    # Shuffle, format as tensor and split into train and test
    rng = np.random.default_rng(1234)
    rng.shuffle(data, axis=0)
    split = int(len(data) * 0.9)
    train = data[:split]
    test = data[split:]

    return train, test


def reshape_data(df):
    """
    Reshapes data into [INR, covariates, maintenance dose].
    """
    # Keep only final measurement of INR
    df = df[(df.Observable != 'INR') | (df.Time == 1320)]

    # Reshape data
    ids = df.ID.dropna().unique()
    data = np.zeros(shape=(len(ids), 6))
    for idx, _id in enumerate(ids):
        # Get info from dataframe
        temp = df[df.ID == _id]
        inr = temp[temp.Observable == 'INR'].Value.values
        vkorc1 = temp[temp.Observable == 'VKORC1'].Value.values
        cyp = temp[temp.Observable == 'CYP2C9'].Value.values
        age = temp[temp.Observable == 'Age'].Value.values
        dose = temp.Dose.dropna().values[-1]

        # Fill container
        data[idx, 0] = inr
        data[idx, -1] = dose

        # We implement genetic factors by counting allele variants
        if vkorc1 == 0:
            data[idx, 1] = 1
        elif vkorc1 == 1:
            data[idx, 1] = 0.5
        if cyp == 0:
            data[idx, 2] = 1
        elif cyp == 1:
            data[idx, 2] = 0.5
            data[idx, 3] = 0.5
        elif cyp == 2:
            data[idx, 2] = 0.5
        elif cyp == 3:
            data[idx, 3] = 1
        elif cyp == 4:
            data[idx, 3] = 0.5

        # Add age
        data[idx, 4] = age

    return data


def format_data(train, test, model, target, device):
    """
    Normalise data and split into train and test set.

    inr: needs to be standardised: inr <- (inr - target) / target.
    action: Replace doses by bin number.
    reward: Normalise rewards
    """
    # Normalise INR values
    train[:, 0] = (train[:, 0] - target) / (3 * target)
    test[:, 0] = (test[:, 0] - target) / (3 * target)

    # Make networks remember this scale
    model.set_inr_scale(target, 3 * target)

    # Normalise age
    mean = 51
    std = 15
    train[:, -2] = (train[:, -2] - mean) / (3 * std)
    test[:, -2] = (test[:, -2] - mean) / (3 * std)

    # Make networks remember this scale
    model.set_age_scale(mean, 3 * std)

    # Normalise dose (min 0 and max 30)
    train[:, -1] = train[:, -1] / 30
    test[:, -1] = test[:, -1] / 30

    train = torch.tensor(train, dtype=torch.float32, device=device)
    test = torch.tensor(test, dtype=torch.float32, device=device)

    return train, test


def train_algorithm(
        data_train, data_test, batch_size, n_epochs=10,
        scheduler=None):
    """
    Returns a trained regression model.
    """
    rng = np.random.default_rng(67)
    performance = np.zeros((n_epochs, 3))
    best = [np.inf, 0, None]  # MSE error, epoch, model
    for epoch in range(n_epochs):
        # Perform one step of the optimization (on the policy network)
        train_loss = train_epoch(
            data_train, model, batch_size, optimiser, rng)
        performance[epoch, 1] = train_loss

        # Evaluate on test set
        test_loss, best = evaluate(data_test, model, best, epoch)
        performance[epoch, 2] = test_loss

        # Decay learning rate
        if scheduler is not None:
            scheduler.step(test_loss)

    performance[:, 0] = np.arange(n_epochs) + 1
    best_epoch = best[1]
    plot_progress(performance, best_epoch)

    best_model_parameters = best[2]
    latest_model_parameters = model.state_dict()

    return best_model_parameters, latest_model_parameters


def train_epoch(data, model, batch_size, optimiser, rng):
    # Shuffle data
    avg_loss = 0
    n_data = len(data)
    indices = np.arange(n_data)
    rng.shuffle(indices)
    data = data[indices]

    for idb in range(n_data // batch_size):
        batch = data[idb * batch_size: (idb + 1) * batch_size]
        loss = train_batch(batch, model, optimiser)
        avg_loss = (avg_loss * idb + loss) / (idb + 1)

    return avg_loss


def train_batch(batch, model, optimiser):
    inbut_batch = batch[:, :5]
    target_batch = batch[:, [5]]

    # Predict dose
    doses = model(inbut_batch)

    # Quantify TD error using Huber loss
    criterion = nn.MSELoss()
    loss = criterion(doses, target_batch)

    # Optimize the model
    optimiser.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
    optimiser.step()

    return loss


def evaluate(data, model, best, epoch):
    loss = evaluate_test_set(data, model)

    if loss < best[0]:
        # This model performs better on the test set than in the previous
        # iterations
        best[0] = loss
        best[1] = epoch
        best[2] = model.state_dict()

    return loss, best


def evaluate_test_set(data, model):

    inbut_batch = data[:, :5]
    target_batch = data[:, [5]]

    with torch.no_grad():
        # Predict dose
        doses = model(inbut_batch)

        # Quantify TD error using Huber loss
        criterion = nn.MSELoss()
        loss = criterion(doses, target_batch)

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
    axes[0].set_ylabel('Mean squared error')
    axes[0].set_xlabel('Epoch')
    axes[0].set_xscale('log')
    axes[0].legend()

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(
        directory + '/1_network_training.pdf', bbox_inches='tight')
    plt.savefig(
        directory + '/1_network_training.tif', bbox_inches='tight')


if __name__ == '__main__':
    batch_size = 128
    lr = 1e-4

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
    model = MaintenanceDoseNetwork(width=1024).to(device)

    # Define optimiser
    optimiser = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 'min', patience=10, min_lr=1e-7, factor=0.5)

    target = 2.5
    train, test = load_data()
    train, test = format_data(
        train, test, model, target, device)
    best, latest = train_algorithm(
        train, test, batch_size, n_epochs=1000, scheduler=scheduler)

    # Save model
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = '/model/deep_regression_best.pickle'
    torch.save(best, directory + filename)
    filename = '/model/deep_regression_latest.pickle'
    torch.save(latest, directory + filename)
