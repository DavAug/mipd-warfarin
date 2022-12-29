import pickle
import os
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("error")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline


def prepare_data():
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measurements_df = pd.read_csv(
        directory + '/data/trial_phase_III.csv')

    # Keep only steady state measurement
    mask = \
        (measurements_df.Observable != 'INR') | (
        (measurements_df.Observable == 'INR') & (measurements_df.Time == 1320))
    measurements_df = measurements_df[mask]

    # Reshape data into [INR, CYP *1/*1, CYP *1/*2, CYP *1/*3, CYP *2/*2,
    # CYP *2/*3, CYP *3/*3, VKORC GG, VKORC GA, VKORC AA, Age,
    # Maintenance dose]
    # NOTE: CYP and VKORC are implemented using a 1-hot encoding
    ids = measurements_df.ID.dropna().unique()
    data = np.zeros(shape=(len(ids), 12))
    for idx, _id in enumerate(ids):
        temp = measurements_df[measurements_df.ID == _id]
        data[idx, 0] = temp[temp.Observable == 'INR'].Value.values[0]
        cyp = temp[temp.Observable == 'CYP2C9'].Value.values[0]
        data[idx, int(cyp + 1)] = 1
        vkorc = temp[temp.Observable == 'VKORC1'].Value.values[0]
        data[idx, int(vkorc + 7)] = 1
        data[idx, 10] = temp[temp.Observable == 'Age'].Value.values[0]
        data[idx, 11] = temp.Dose.dropna().values[-1]

    return data

def tune_hyperparameters(data):
    # Split training set into train and validation set for hypertuning
    np.random.seed(6)
    n_ids = len(data)
    indices = np.random.permutation(n_ids)
    split_index = int(n_ids * 0.8)
    train_ids = indices[:split_index]
    val_ids = indices[split_index:]
    x_train = data[train_ids, :-1]
    y_train = data[train_ids, -1]
    x_val = data[val_ids, :-1]
    y_val = data[val_ids, -1]

    # Find hyperparameters
    lambdas = [10**power for power in np.arange(-3, 2, 0.1)]
    degree = 4

    # Compute MSE for models
    mses = []
    for l in lambdas:
        # Define regressors and basis expansion
        lasso = make_pipeline(
            StandardScaler(), PolynomialFeatures(degree), Lasso(alpha=l))

        # Fit model
        try:
            lasso.fit(x_train, y_train)
        except ConvergenceWarning:
            # The lambda is chosen too small for LASSO to converge
            mses.append(np.inf)
            continue

        # Evaluate model
        mse = mean_squared_error(lasso.predict(x_val), y_val)
        mses.append(mse)

        print('For lambda=%f: MSE=%f\n' % (l, mse))

    plot_cross_validation(lambdas, mses)

    # Pick lambda with smallest MSE
    l = lambdas[np.argmin(mses)]
    print('Chosen lambda: ', l)

    return l, degree


def fit_model_to_data(data, l, d):
    lasso = make_pipeline(
        StandardScaler(), PolynomialFeatures(d), Lasso(alpha=l))
    x = data[:, :-1]
    y = data[:, -1]
    lasso.fit(x, y)

    # Plot results
    ypred = lasso.predict(x)
    plot_residuals(y, ypred)

    return lasso


def plot_cross_validation(lambdas, mse):
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
    axes[0].plot(lambdas, mse, marker='x', color='black')

    # Labelling
    axes[0].set_xlabel('Lambda')
    axes[0].set_ylabel('MSE')
    axes[0].set_xscale('log')

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(directory + '/S1_cross_validation.pdf', bbox_inches='tight')
    plt.savefig(directory + '/S1_cross_validation.tif', bbox_inches='tight')


def plot_residuals(y, ypred):
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

    axes[0].plot(
        [0, 1], [0, 1], transform=axes[0].transAxes, color='blue')

    # Plot data
    axes[0].scatter(ypred, y, marker='o', color='black')

    # Labelling
    axes[0].set_xlabel('Individual predictions')
    axes[0].set_ylabel('Maintenance dose in mg')
    axes[0].set_ylim([-1, 25])
    axes[0].set_xlim([-1, 25])

    # Save figure
    directory = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(directory + '/S2_residuals.pdf', bbox_inches='tight')
    plt.savefig(directory + '/S2_residuals.tif', bbox_inches='tight')


if __name__ == '__main__':
    data = prepare_data()
    l, d = tune_hyperparameters(data)
    model = fit_model_to_data(data, l, d)

    # Safe final model
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(directory + '/model/linear_regression_model.pickle', 'wb') as f:
        pickle.dump(model, f)
        f.close()
