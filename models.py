import torch
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class OneChannelCNNModel(nn.Module):
    def __init__(self, hidden, out_shape):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 4 * 2, out_shape)
        )

    def forward(self, x):
        x = self.conv_block_2(
            self.conv_block_1(x)
        )
        return self.classifier(x)


class MultiChannelCNN(nn.Module):
    def __init__(self, in_channels, hidden, out_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 4 * 2, out_shape)
        )

    def forward(self, x):
        x = self.conv_block_2(
            self.conv_block_1(x)
        )
        return self.classifier(x)


class OneHotChannelCNN(nn.Module):
    def __init__(self, in_channels, hidden, out_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 4 * 2, out_shape)
        )

    def forward(self, x):
        return self.classifier(
            self.conv_block_2(
                self.conv_block_1(x)
            )
        )

