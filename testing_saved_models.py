import torch
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import models
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


def predict(model_output):
    b = model_output[:, -1]
    layer = torch.arange(len(grades)).unsqueeze(0).to(DEVICE)
    y = b * layer.repeat(len(b), 1).T
    b2 = torch.square(b)
    i = b2 - y
    k = 1. / (3 + torch.abs(i))
    modified_preds = torch.softmax(model_output[:, :-1], dim=1) + k.T
    return modified_preds.argmax(dim=1)


if __name__ == "__main__":
    set_encoded_data, grades_tensor, grades = pre_process.pull_in_data(set_encoded=True)

    X_train, X_test, y_train, y_test = train_test_split(set_encoded_data,
                                                        grades_tensor,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = X_train.to(DEVICE), X_test.to(DEVICE), y_train.to(DEVICE), y_test.to(DEVICE)

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)
    train_dataloaders, test_dataloaders = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = models.DeepSet(200, len(grades)+1).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODELS_PATH / 'deep_set'))

    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy("multiclass", num_classes=len(grades)).to(DEVICE)

    y_preds = []
    # acc_av = 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloaders:
            y_logits = model(X)
            y_pred = predict(y_logits)
            y_preds.append(y_pred)
            # acc_av += acc_fn(y_pred, y)
        # acc_av /= len(test_dataloaders)
    y_preds = torch.cat(y_preds)
    # print(acc_av)

    confmat = ConfusionMatrix(num_classes=len(grades), task='multiclass')
    confmat_tensor = confmat(preds=y_preds.to('cpu'),
                             target=y_test.to('cpu'))

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=grades,  # turn the row and column labels into class names
        figsize=(10, 7))

    FIG_SAVE_PATH = Path('figures')
    fig.savefig(FIG_SAVE_PATH / 'deep_set_conf_mat.png')
    plt.show()