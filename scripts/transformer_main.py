import sys
import torch
from generative_models.tranformer_decoder import TransformerDecoder
from generative_models import tranformer_decoder
from tqdm import tqdm
from torchmetrics import Accuracy
from torch import nn
from sklearn.model_selection import train_test_split
import pre_process
from pathlib import Path
from kl_div_loss import KLLoss
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Create a writer with all default settings
# WRITER = SummaryWriter()

RANDOM_STATE = 33
TEST_SPLIT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = pre_process.BATCH_SIZE


def training_loop():
    train_loss_av = 0
    model.train()
    for X in train_dataloader:
        logits = model(X)
        loss = cr_loss()

        train_loss_av += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_av /= len(train_dataloader)
    return train_loss_av


if __name__ == "__main__":
    max_num_holds = 28

    X_train, X_test, grades_tensor, grades = pre_process.pull_in_data(size=100, encoding='transformer')  # tensor shape (488287, 202) for train, (85992, 202) for test

    X_train, X_test = X_train.to(DEVICE), X_test.to(DEVICE)

    train_dataset, test_dataset = pre_process.no_label_datasets(X_train, X_test)

    # TODO : create y data

    train_dataloader, test_dataloader = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = TransformerDecoder(128, 3, 202).to(DEVICE)

    cr_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    '''WRITER.add_graph(model=model,
                     # Pass in an example input
                     input_to_model=torch.randn(BATCH_SIZE, 28, 200).to(DEVICE))'''

    epochs = 100
    for epoch in range(epochs):
        if epoch % 5 == 0:
            print('Epoch ', epoch, '\n')

        train_loss_av = training_loop()

        test_loss_av, test_acc_av, test_obo_acc_av = testing_loop()

        if epoch % 5 == 0:
            print(f"Train loss : {train_loss_av} | Test loss : {test_loss_av}")

    SAVE_PATH = Path('../saved_models')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'transformer_decoder'

    torch.save(model.state_dict(), f=SAVE_PATH / MODEL_NAME)

