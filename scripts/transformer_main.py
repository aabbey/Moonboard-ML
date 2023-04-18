import torch
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = pre_process.BATCH_SIZE


def training_loop(double_loss=False):
    train_loss_av = 0
    model.train()
    for X, y in train_dataloader:
        preds = model(X)
        if double_loss:
            loss_cr = cr_loss(preds[:, :-1], y)
            loss_mse = mse_loss(preds[:, -1], y.float())
            loss = loss_mse * 0.2 + loss_cr * 0.8
        else:
            loss = kl_loss(preds, y)

        train_loss_av += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_av /= len(train_dataloader)
    return train_loss_av


if __name__ == "__main__":
    max_num_holds = 28

    set_encoded_data, grades_tensor, grades = pre_process.pull_in_data(encoding='transformer')  # tensor shape (60000, 28, 200)

    X_train, X_test, y_train, y_test = train_test_split(set_encoded_data,
                                                        grades_tensor,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = X_train.to(DEVICE), X_test.to(DEVICE), y_train.to(DEVICE), y_test.to(DEVICE)

    # change y to work with kl div
    print(y_test[1])
    y_train, y_test = grade_to_dist(y_train, len(grades)).squeeze(), grade_to_dist(y_test, len(grades)).squeeze()
    print(y_test[1])

    train_dataset, test_dataset = pre_process.create_datasets(X_train, X_test, y_train, y_test)

    train_dataloader, test_dataloader = pre_process.create_dataloaders(train_dataset, test_dataset)

    model = predictions_models.DeepSet(200, len(grades)).to(DEVICE)

    cr_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    kl_loss = KLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    acc_fn = Accuracy("multiclass", num_classes=len(grades)).to(DEVICE)

    '''WRITER.add_graph(model=model,
                     # Pass in an example input
                     input_to_model=torch.randn(BATCH_SIZE, 28, 200).to(DEVICE))'''

    epochs = 100
    for epoch in range(epochs):
        if epoch % 5 == 0:
            print('Epoch ', epoch, '\n')

        train_loss_av = training_loop()

        test_loss_av, test_acc_av, test_obo_acc_av = testing_loop()

        WRITER.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss_av,
                                            "test_loss": test_loss_av},
                           global_step=epoch)

        # Add accuracy results to SummaryWriter
        WRITER.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={"test_acc": test_acc_av},
                           global_step=epoch)

        WRITER.add_scalars(main_tag="Off by one Accuracy",
                           tag_scalar_dict={"test_obo_acc": test_obo_acc_av},
                           global_step=epoch)

        WRITER.close()

        if epoch % 5 == 0:
            print(f"Train loss : {train_loss_av} | Test loss : {test_loss_av} | Test accuracy : {test_acc_av} | Test obo accuracy : {test_obo_acc_av}")

    SAVE_PATH = Path('../saved_models')
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'deep_set_kl_loss'

    torch.save(model.state_dict(), f=SAVE_PATH / MODEL_NAME)

