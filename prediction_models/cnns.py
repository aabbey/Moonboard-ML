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


class OneChannelCNNMSE(nn.Module):
    def __init__(self, hidden):
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
            nn.Linear(hidden * 4 * 2, 1)
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


class OneHotChannelCNN2(nn.Module):
    def __init__(self, in_channels, hidden, out_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=5, stride=1, padding='same'),  # (-1, 128, 18, 11)
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden*2, kernel_size=5, stride=1, padding='same'),  # (-1, 256, 18, 11)
            nn.BatchNorm2d(hidden*2),
            nn.LeakyReLU(),
            nn.Conv2d(hidden*2, hidden*4, kernel_size=4, stride=1, padding=1),  # (-1, 512, 17, 10)
            nn.BatchNorm2d(hidden * 4),
            nn.LeakyReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden*4, hidden*6, kernel_size=3, stride=2, padding=2),  # (-1, 768, 10, 6)
            nn.BatchNorm2d(hidden * 6),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # (-1, 1028, 5, 3)
            nn.Conv2d(hidden*6, hidden*8, kernel_size=(3, 2), stride=(2, 1), padding=0),  # (-1, 1024, 2, 2)
            nn.BatchNorm2d(hidden * 8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)  # (32, 2048, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(hidden * 8, out_shape)
        )

    def forward(self, x):
        return self.classifier(
            self.conv_block_2(
                self.conv_block_1(x)
            )
        )