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


def create_one_hot_per_hold():
    grid = torch.zeros(198, 18, 11)
    for vec in range(198):
        one_hot_embedding = torch.zeros(198)
        one_hot_embedding[vec] = 1.
        x = vec % 11
        y = int(np.floor(vec / 11))
        grid[:, y, x] = one_hot_embedding
    return grid


def create_hold_embedding_vectors(hold_quality, hold_direction):
    normalized = nn.functional.normalize(hold_direction, dim=0)
    return normalized * hold_quality


def pull_in_data(size=59787):
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

    grid_encoded_data = torch.zeros(size, 1, 18, 11)
    for i, prob in enumerate(d['data'][:size]):
        for hold in prob['moves']:
            x, y = hold['description'][0], hold['description'][1:]
            x, y = ord(x.upper()) - 65, int(y) - 1
            grid_encoded_data[i, 0, 17-y, x] = 1.  # tensor shape (59787, 1, 18, 11) num_samples by moonboard grid shape

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
