import sys
import torch
import matplotlib.pyplot as plt
import torchvision

from generative_models.tranformer_decoder import TransformerDecoder
from generative_models import tranformer_decoder
from generative_models.tranformer_decoder import Transformer
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = pre_process.BATCH_SIZE


@torch.no_grad()
def convert_tensor_to_grid_coordinates(tensor):
    index = torch.argmax(tensor)
    x = index % 18
    y = index // 18
    return x, y


@torch.no_grad()
def make_boulder(grade=2.):
    grid = torch.zeros((18, 11))
    context = torch.zeros(1, 1, 202).to(DEVICE)
    context[0, 0, 0] = grade
    end = False
    while True:
        hold_token, logits, end = next_hold(context)
        # print(logits.softmax(dim=1))
        # print(logits[:, -1, 2:-2].softmax(dim=1))
        if end:
            break
        x, y = convert_tensor_to_grid_coordinates(hold_token)
        grid[x, y] = 1
        next_token = torch.zeros(1, 1, 202, device=DEVICE)
        next_token[0, 0, 2:-2] = hold_token
        context = torch.cat((context, next_token), dim=1)  # (1, loop number, 202)

        show_tensor_images(torch.stack((context.sum(dim=1)[:, 2:-2].view(18, 11).flip(0),
                                        logits[:, -1, 2:-2].softmax(dim=1).view(18, 11).flip(0)),
                                       dim=0).unsqueeze(1))


@torch.no_grad()
def show_tensor_images(image_tensor, num_images=2, size=(1, 18, 11)):
    # image_tensor is (2, 1, 18, 11) for 10 images. (2 by 1) 1 is nrow
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


@torch.no_grad()
def next_hold(context):
    end = False
    logits = model(context)
    if logits[:, -1, :].argmax(dim=1).item() == 1 or logits[:, -1, :].argmax(dim=1).item() == 200:
        logits[1, -1, :] = float('-inf')
        logits[200, -1, :] = float('-inf')
    if logits[:, -1, :].argmax(dim=1).item() == 201 or logits.shape[1] > 13:
        end = True
        return None, logits, end
    hold_token = torch.zeros(198, device=DEVICE)
    hold_token[logits[:, -1, :].argmax(dim=1)-2] = 1.
    return hold_token, logits, end


@torch.no_grad()
def testing_loop():
    test_loss_av = 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            logits = model(X)
            loss = cr_loss(logits, y)
            test_loss_av += loss

        test_loss_av /= len(test_dataloader)
    return test_loss_av


def training_loop():
    train_loss_av = 0
    model.train()
    for X, y in train_dataloader:
        logits = model(X)
        loss = cr_loss(logits, y)

        train_loss_av += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_av /= len(train_dataloader)
    return train_loss_av


if __name__ == "__main__":
    max_num_holds = 28

    X_train, X_test, grades_tensor, grades = pre_process.pull_in_data(encoding='transformer')  # tensor shape (50385, 14, 202) for train, (8903, 14, 202) for test

    X_train, X_test = X_train.to(DEVICE), X_test.to(DEVICE)

    # create targets by offsetting X
    y_train = torch.zeros_like(X_train)
    y_train[:, -1, -1] = 1.
    y_train[:, :-1] = X_train[:, 1:]
    y_test = torch.zeros_like(X_test)
    y_test[:, -1, -1] = 1.
    y_test[:, :-1] = X_test[:, 1:]

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)

    train_dataloader, test_dataloader = pre_process.create_dataloaders(train_dataset, test_dataset, batch_size=16)

    #model = TransformerDecoder(128, 3, 202).to(DEVICE)
    model = Transformer(feature_size=202, head_size=64, block_size=14).to(DEVICE)

    cr_loss = nn.CrossEntropyLoss()
    my_cr_loss = tranformer_decoder.MyCrLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    '''WRITER.add_graph(model=model,
                     # Pass in an example input
                     input_to_model=torch.randn(BATCH_SIZE, 28, 200).to(DEVICE))'''

    epochs = 100
    for epoch in range(epochs):
        if epoch % 1 == 0:
            print('Epoch ', epoch, '\n')

        train_loss_av = training_loop()

        if epoch % 5 == 0:
            test_loss_av = testing_loop()
            print(f"Train loss : {train_loss_av} | Test loss : {test_loss_av}")
            make_boulder(2)

    SAVE_PATH = Path('../saved_models')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'transformer_decoder'

    torch.save(model.state_dict(), f=SAVE_PATH / MODEL_NAME)

