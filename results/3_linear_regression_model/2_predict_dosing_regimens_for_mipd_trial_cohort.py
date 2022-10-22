import pickle
import os

import numpy as np
import pandas as pd


def load_data():
    # Import data
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measurements_df = pd.read_csv(directory + '/data/mipd_trial_cohort.csv')

    # Reshape data into [INR, CYP *1/*1, CYP *1/*2, CYP *1/*3, CYP *2/*2,
    # CYP *2/*3, CYP *3/*3, VKORC GG, VKORC GA, VKORC AA, Age]
    # NOTE: CYP and VKORC are implemented using a 1-hot encoding
    ids = measurements_df.ID.dropna().unique()
    data = np.zeros(shape=(len(ids), 11))
    for idx, _id in enumerate(ids):
        data[idx, 0] = 2.5  # Target INR is fixed for all subjects
        temp = measurements_df[measurements_df.ID == _id]
        cyp = temp['CYP2C9']
        data[idx, int(cyp + 1)] = 1
        vkorc = temp['VKORC1']
        data[idx, int(vkorc + 7)] = 1
        data[idx, 10] = temp['Age']

    return ids, data

def load_model():
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(directory + '/model/linear_regression_model.pickle', 'rb') as f:
        model = pickle.load(f)
        f.close()

    return model

def predict_maintenance_dose(data, model):
    log_doses = model.predict(data)
    return np.exp(log_doses)

def save_results(ids, doses):
    df = pd.DataFrame({
        'ID': ids,
        'Maintenance dose': doses
    })

    # Save dataframe to csv
    directory = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(
        directory + '/mipd_trial_predicted_dosing_regimens.csv', index=False)


if __name__ == '__main__':
    ids, data = load_data()
    model = load_model()
    doses = predict_maintenance_dose(data, model)
    save_results(ids, doses)
