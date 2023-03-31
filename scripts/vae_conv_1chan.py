import json
import sys
import torch
import torchvision
from tqdm import tqdm
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import generative_models
import predictions_models
import pre_process
from hold_embeddings import hold_quality, hold_angles
from pathlib import Path
from kl_div_loss import KLLoss
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Create a writer with all default settings
WRITER = SummaryWriter()

RANDOM_STATE = 33
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = pre_process.BATCH_SIZE
VAR_WEIGHT = 0.02  # variance to add to latent var to make z


def show_tensor_images(image_tensor, num_images=5, size=(1, 18, 11)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def training_loop(epoch):
    train_loss_av = 0
    model.train()
    for X, y in train_dataloader:
        one_hot_grades = torch.nn.functional.one_hot(y, 14)
        encoded, z_mean, z_log_var, decoded = model(X, one_hot_grades, var_weight=VAR_WEIGHT)

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), axis=1)
        kl_div = kl_div.mean()


        recon_loss = mse_loss(decoded, X)
        loss = 5 * recon_loss + kl_div

        train_loss_av += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_av /= len(train_dataloader)
    return train_loss_av


if __name__ == "__main__":
    set_encoded_data, grades_tensor, grades = pre_process.pull_in_data()  # tensor shape (60000, 1, 18, 11)

    X_train, X_test, y_train, y_test = train_test_split(set_encoded_data,
                                                        grades_tensor,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = X_train.to(DEVICE), X_test.to(DEVICE), y_train.to(DEVICE), y_test.to(DEVICE)

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)

    train_dataloader, test_dataloader = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = generative_models.VAEConv(64, 64).to(DEVICE)

    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # WRITER.add_graph(model=model, input_to_model=torch.randn(BATCH_SIZE, 1, 18, 11).to(DEVICE))

    epochs = 300
    for epoch in range(epochs):
        if epoch % 1 == 0:
            print('Epoch ', epoch, '\n')

        train_loss_av = training_loop(epoch)

        """WRITER.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss_av},
                           global_step=epoch)
        WRITER.close()"""

        if epoch % 1 == 0:
            print(f"Train loss : {train_loss_av}")
        if epoch in [5, 20, 50, 150, 299]:
            model.eval()
            with torch.inference_mode():
                X, y = next(iter(train_dataloader))
                one_hot_grades = torch.nn.functional.one_hot(y, 14)
                _, _, _, decoded = model(X, one_hot_grades, VAR_WEIGHT)
                show_tensor_images(torch.cat((X[:5], decoded[:5])), num_images=10)

    SAVE_PATH = Path('../saved_models')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'vae_conv_1'

    torch.save(model.state_dict(), f=SAVE_PATH / MODEL_NAME)