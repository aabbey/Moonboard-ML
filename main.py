import json
import sys
import torch
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import predictions_models
import pre_process
from hold_embeddings import hold_quality, hold_angles
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 33


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())

    import sys

    sys.exit()

    HOLD_EMBEDDINGS = pre_process.create_one_hot_per_hold()

    df = pre_process.pull_in_data(300, True)
    print(df[df['userGrade'] != df['grade']].to_string())

    X_train, X_test, y_train, y_test = train_test_split(grid_encoded_data,
                                                        grades_tensor,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE)

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)

    train_dataloader, test_dataloader = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = models.OneChannelCNNModel(10, len(grades))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    acc_fn = Accuracy("multiclass", num_classes=len(grades))

    one_epoch_test()

    epochs = 30
    for epoch in range(epochs):
        print('Epoch ', epoch, '\n')
        train_loss_av = 0
        model.train()
        for X, y in train_dataloader:
            preds = model(X)
            loss = loss_fn(preds, y)
            train_loss_av += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_av /= len(train_dataloader)
        test_loss_av = 0
        test_acc_av = 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                test_preds = model(X)
                loss = loss_fn(test_preds, y)
                test_loss_av += loss
                test_acc_av += acc_fn(test_preds.argmax(dim=1), y)
            test_loss_av /= len(test_dataloader)
            test_acc_av /= len(test_dataloader)

        print(f"Train loss : {train_loss_av} | Test loss : {test_loss_av} | Test accuracy : {test_acc_av}")

    SAVE_PATH = Path('saved_models')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'one_channel_cnn'

    torch.save(model.state_dict(), f=SAVE_PATH / MODEL_NAME)

