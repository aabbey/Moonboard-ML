import torch
from torchmetrics import Accuracy
from torch import nn


class DeepSet(nn.Module):

    r"""
    mlp( sum( mlp( each one hot encoded hold in boulder)))
    """
    def __init__(self, feature_in, output_shape):
        super().__init__()

        self.feature_mlp_out_shape = feature_in/2

        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_in, feature_in*4),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(feature_in*4, int(feature_in/2)),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )

        self.set_mlp = nn.Sequential(
            nn.Linear(int(feature_in/2), feature_in*2),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(feature_in*2, output_shape)
        )

    def forward(self, set_of_holds):
        """
        :param set_of_holds: (batch size, 28, 200)
        :return: output
        """
        features_out = self.feature_mlp(set_of_holds)
        features_output = torch.sum(features_out, dim=1)

        return self.set_mlp(features_output)
