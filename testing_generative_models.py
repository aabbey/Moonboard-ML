import torch
import torchvision
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import generative_models
import predictions_models
from hold_embeddings import hold_quality, hold_angles
import pre_process
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path

SAVED_MODELS_PATH = Path("saved_models")
RANDOM_STATE = 33
HOLD_EMBEDDINGS = pre_process.create_one_hot_per_hold()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = pre_process.BATCH_SIZE
VAR_WEIGHT = 0.02


def show_tensor_images(image_tensor, num_images=5, size=(1, 18, 11)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


if __name__ == "__main__":
    set_encoded_data, grades_tensor, grades = pre_process.pull_in_data()  # tensor shape (60000, 1, 18, 11)

    X_train, X_test, y_train, y_test = train_test_split(set_encoded_data,
                                                        grades_tensor,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = X_train.to(DEVICE), X_test.to(DEVICE), y_train.to(DEVICE), y_test.to(DEVICE)

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)
    train_dataloaders, test_dataloaders = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = generative_models.VAEConv(64, 64).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODELS_PATH / 'vae_conv_1'))

    loss_fn = generative_models.VAELoss(recon_weight=5)

    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloaders:
            one_hot_grades = torch.nn.functional.one_hot(y, 14)
            encoded, z_mean, z_log_var, decoded = model(X, one_hot_grades, VAR_WEIGHT)
            print(loss_fn(decoded, X, z_log_var, z_mean))
            print(encoded[:5])
            dist = torch.sqrt(torch.sum(torch.square(encoded - encoded[0]), 1))
            mins = torch.argsort(dist)
            print(dist)
            print(mins)
            show_tensor_images(torch.cat((X[:5], decoded[:5], X[mins[:5]])), num_images=15)
