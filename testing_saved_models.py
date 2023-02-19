import torch
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import models
import pre_process
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from pathlib import Path

SAVED_MODELS_PATH = Path("saved_models")
RANDOM_STATE = 33

if __name__ == "__main__":
    grid_encoded_data, grades_tensor, grades = pre_process.pull_in_data()

    X_train, X_test, y_train, y_test = train_test_split(grid_encoded_data,
                                                        grades_tensor,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE)

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)
    train_dataloaders, test_dataloaders = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = models.OneChannelCNNModel(10, len(grades))
    model.load_state_dict(torch.load(SAVED_MODELS_PATH / 'one_channel_cnn'))

    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy("multiclass", num_classes=len(grades))

    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X,y in test_dataloaders:
            y_logits = model(X)
            y_pred = y_logits.argmax(dim=1)
            y_preds.append(y_pred)
    y_preds = torch.cat(y_preds)

    confmat = ConfusionMatrix(num_classes=len(grades), task='multiclass')
    confmat_tensor = confmat(preds=y_preds,
                             target=y_test)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=grades,  # turn the row and column labels into class names
        figsize=(10, 7))

    FIG_SAVE_PATH = Path('/home/alex/PycharmProjects/Moonboard-ML/figures')
    fig.savefig(FIG_SAVE_PATH / 'one_channel_cnn_conf_mat.png')
    plt.show()