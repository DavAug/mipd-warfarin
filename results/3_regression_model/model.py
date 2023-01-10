import torch
import torch.nn as nn
import torch.nn.functional as F


class MaintenanceDoseNetwork(nn.Module):
    """
    Defines a neural network that predicts the maintenance warfarin dose
    based on the target INR and the covariates.
    """
    def __init__(self, width=128):
        super(MaintenanceDoseNetwork, self).__init__()
        # States are: INR, VKORC1 G alleles,
        # CYP2C9 *1 alleles, CYP2C9 *2 alleles, Age
        n_states = 5
        self.layer1 = nn.Linear(n_states, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.layer4 = nn.Linear(width, 1)

        # Define dummies to reverse standardisation of INR values
        self._mean_inr = None
        self._std_inr = None
        self._mean_age = None
        self._std_age = None

    def _format_dose(self, output):
        """
        Returns the warfarin dose.

        The network returns a number between 0 and 1. Here we translate this
        number into warfarin tablets between 0mg and 30 mg.
        """
        dose = output * 30
        mask = dose < 0.5
        dose[mask] = 0
        mask = (dose >= 0.5) & (dose < 1.25)
        dose[mask] = 1
        mask = dose >= 1.25
        dose[mask] = torch.round(dose[mask] * 2) / 2

        return dose

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
        q = torch.sigmoid(self.layer4(x))

        return q

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
        state[:, 0] = (state[:, 0] - self._mean_inr) / self._std_inr
        state[:, -1] = (state[:, -1] - self._mean_age) / self._std_age

        with torch.no_grad():
            q = self.forward(state)

        dose = self._format_dose(q)

        return dose
