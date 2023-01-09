import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Defines deep Q Learning network.
    """
    def __init__(self, width=128):
        super(DQN, self).__init__()
        # Define action bins
        self._doses = np.array([
            0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
            10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,
            16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5,
            23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29,
            29.5, 30
        ])

        # States are: INR, VKORC1 G alleles,
        # CYP2C9 *1 alleles, CYP2C9 *2 alleles, Age
        n_states = 5
        n_actions = len(self._doses)
        self.layer1 = nn.Linear(n_states, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.layer4 = nn.Linear(width, n_actions)

        # Define dummies to reverse standardisation of INR values
        self._mean_inr = None
        self._std_inr = None
        self._mean_age = None
        self._std_age = None

    def forward(self, state):
        """
        Returns an estimate of the action-value function, Q(a | s).

        Assumes that the state has been appropriately scaled.

        State is of shape (batch, 2), where the rows encode the INR
        measurement, the CYP2C9 genotype.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        q = self.layer4(x)

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

    def set_age_scale(self, mean, std):
        """
        Defines the scale of age values.

        Is used to translate age values into inputs to the network.
        """
        self._mean_age = mean
        self._std_age = std

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
        input = torch.clone(state)
        input[:, 0] = (input[:, 0] - self._mean_inr) / self._std_inr
        input[:, -1] = (input[:, -1] - self._mean_age) / self._std_age

        with torch.no_grad():
            q = self.forward(input)
            dose = self._doses[q.max(1)[1]]

        return dose
