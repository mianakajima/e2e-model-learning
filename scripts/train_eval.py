import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model_setup import *

import matplotlib.pyplot as plt

def train_RMS_model(X_train, y_train, model, num_epochs):
    """
    TODO: make this more readable
    Trains the neural network (RMSE only portion).
    :param X_train: Torch tensor of training data features
    :param y_train: Torch tensor of training data targets
    :param model: Torch NN model
    :param num_epochs: Number of epochs model should train for
    :return: trained model, loss history, accuracy history
    """
    # set seed
    # TODO: Not sure if this is working - results look different when running
    torch.manual_seed(seed)

    td = TensorDataset(X_train, y_train)
    dataloader = DataLoader(td, batch_size= batch_size, shuffle = True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Initialize outputs
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs

    for epoch in range(num_epochs):
        for X_batch, y_batch in dataloader:
            pred = model(X_batch)
            mse = loss_fn(pred, y_batch)
            rmse_loss = torch.sqrt(mse)
            rmse_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # log loss history
            loss_hist[epoch] += rmse_loss.item()
            # calculate % predicted / actual
            is_correct = torch.div(pred, y_batch)
            accuracy_hist[epoch] += is_correct.mean().item()
        # get average over dataset
        loss_hist[epoch] /= len(dataloader)
        accuracy_hist[epoch] /= len(dataloader)

    return model, loss_hist, accuracy_hist

def plot_loss_accuracy(loss_hist, accuracy_hist):

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_hist, lw=3)
    ax.set_title("Loss (RMSE)", size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy_hist, lw=3)
    ax.set_title("Accuracy (Predicted/Actual)", size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

def evaluate_model_by_hour(X_test, y_test, model, scaler):
    """
    Returns the RMSE and accuracy by hour of day.
    :param X_test:
    :param y_test:
    :param model:
    :param scaler: sklearn scaler
    :return:
    """

    # normalize X_test and convert to tensors
    X_test_norm = scaler.transform(X_test)
    X_test_norm = torch.from_numpy(X_test_norm).float()
    y_test = torch.from_numpy(y_test.values).float()

    pred = model(X_test_norm)

    # compute rmse and accuracy for every hour
    rmse = [0] * 24
    accuracy = [0] * 24

    for hour in range(24):
        # get predictions and actuals for the hour
        pred_for_hour = pred[:, hour]
        actual_for_hour = y_test[:, hour]
        rmse[hour] = torch.sqrt(torch.mean(torch.square(pred_for_hour - actual_for_hour))).item()
        accuracy[hour] = torch.mean(torch.div(pred_for_hour, actual_for_hour)).item()

    return rmse, accuracy