import json
import sys
import torch
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import models
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(33)
MANUAL_HOLD_EMBEDDINGS = torch.rand(2, 18, 11)

RANDOM_STATE = 33
PROBLEMS_PATH = '/home/alex/PycharmProjects/Moonboard-ML/problems.json'


def pull_in_data(size=59787, encoding='single'):
    with open(PROBLEMS_PATH) as p:
        d = json.load(p)  # dict of total, data

    boulder_data = d['data']  # list of dicts, info for all problems in dataset
    data_df = pd.DataFrame(boulder_data)  # dataframe for all the info about the boulders

    grades = []
    for grade in data_df["grade"]:
        if grade not in grades:
            grades.append(grade)
    grades = sorted(grades)

    grades_tensor = torch.tensor(data_df['grade'][:size].apply(lambda x: grades.index(x)), dtype=torch.long)

    if encoding == 'single':
        grid_encoded_data = []
        for prob in d['data'][:size]:
            layout = torch.zeros(18, 11, dtype=int)
            for hold in prob['moves']:
                x, y = hold['description'][0], hold['description'][1]
                x, y = ord(x.upper()) - 65, int(y) - 1
                layout[17 - y, x] = 1
            grid_encoded_data.append(layout)

        grid_encoded_data = torch.stack(grid_encoded_data).type(
            'torch.FloatTensor').unsqueeze(dim=1)  # tensor shape (59787, 1, 18, 11) num_samples by moonboard grid shape

    else:
        grid_encoded_data = torch.zeros(size, 2, 18, 11)
        for i, prob in enumerate(d['data'][:size]):
            for hold in prob['moves']:
                x, y = hold['description'][0], hold['description'][1]
                x, y = ord(x.upper()) - 65, int(y) - 1
                grid_encoded_data[i, :, x, y] = MANUAL_HOLD_EMBEDDINGS[:, x, y]

    return grid_encoded_data, grades_tensor, grades


def create_datasets(X_train, X_test, y_train, y_test):

    class CustomDataset(Dataset):
        def __init__(self, imgs, labels):
            self.labels = labels
            self.imgs = imgs

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            label = self.labels[idx]
            imgs = self.imgs[idx]
            return imgs, label

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=32)
    return train_dataloader, test_dataloader
