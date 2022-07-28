import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model_setup import *
from models import stochastic_opt_model

import matplotlib.pyplot as plt

def train_RMS_model(X_train, y_train, model, num_epochs, opt_weights):
    """
    TODO: make this more readable
    Trains the neural network (RMSE only portion).
    :param X_train: Torch tensor of training data features
    :param y_train: Torch tensor of training data targets
    :param model: Torch NN model
    :param num_epochs: Number of epochs model should train for
    :param opt_weights: Weights for optimization (task loss)
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
    task_loss_hist = [0] * num_epochs

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for X_batch, y_batch in dataloader:
            pred, _ = model(X_batch, y_batch)
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
            task_loss_hist[epoch] += task_loss(pred, y_batch, opt_weights).item()

        # get average over dataset
        loss_hist[epoch] /= len(dataloader)
        accuracy_hist[epoch] /= len(dataloader)
        task_loss_hist[epoch] /= len(dataloader)

    model.eval()
    pred, _ = model(X_train, y_train)
    model.set_sigma(pred, y_train)

    return model, loss_hist, accuracy_hist, task_loss_hist

def task_loss(pred, y_actual, opt_weights, per_hour = False):
    """ Returns generation cost of prediction (average per day).
    :param per_hour: whether to average per hour (True) or by day (False)
    """
    under_gen = opt_weights['gamma_under'] * torch.maximum(y_actual - pred, torch.tensor(0).repeat(24))
    over_gen = opt_weights['gamma_over'] * torch.maximum(pred - y_actual, torch.tensor(0).repeat(24))
    error_cost = 0.5*(torch.square(y_actual - pred))
    # average cost per day
    if not per_hour:
        total_cost = (under_gen + over_gen + error_cost).sum(1).mean()
    else:
        total_cost = (under_gen + over_gen + error_cost).mean(0)

    return total_cost


def train_optimization_model(X_train, y_train, rms_model, number_epochs, opt_weights):
    """
    TODO: make this more readable
    Trains the neural network (RMSE only portion).
    :param X_train: Torch tensor of training data features
    :param y_train: Torch tensor of training data targets
    :param rms_model: Torch NN model
    :param number_epochs: Number of epochs model should train for
    :param opt_weights: dictionary with the following elements
            - c_ramp
            - gamma_over
            - gamma_under
    :return: trained model, loss history, accuracy history
    """
    # set seed
    # TODO: Not sure if this is working - results look different when running
    torch.manual_seed(seed)
    model = stochastic_opt_model(opt_weights)
    td = TensorDataset(X_train, y_train)
    dataloader = DataLoader(td, batch_size = batch_size, shuffle = True)

    optimizer = torch.optim.Adam(rms_model.parameters(), lr = lr_opt)

    # Initialize outputs
    loss_hist = [0] * number_epochs
    accuracy_hist = [0] * number_epochs
    task_loss_hist = [0] * number_epochs

    for epoch in range(number_epochs):
        print("Epoch: ", epoch)
        for X_batch, y_batch in dataloader:
            # Train initial model
            rms_model.train()
            mu, sigma = rms_model(X_batch, y_batch)
            pred = model(mu, mu, sigma)
            loss = task_loss(pred, y_batch, opt_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # log loss history
            loss_hist[epoch] += loss.item()
            # calculate % predicted / actual
            is_correct = torch.div(pred, y_batch)
            accuracy_hist[epoch] += is_correct.mean().item()
            task_loss_hist[epoch] += task_loss(pred, y_batch, opt_weights).item()
        # get average over dataset
        loss_hist[epoch] /= len(dataloader)
        accuracy_hist[epoch] /= len(dataloader)
        task_loss_hist[epoch] /= len(dataloader)

    return model, loss_hist, accuracy_hist, task_loss_hist


def plot_loss_accuracy(loss_hist, accuracy_hist, task_loss_hist, save_dir, xlabel, show_plot = False):
    """Plot RMSE, accuracy and task loss"""
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(loss_hist, lw=3)
    ax.set_title("Loss (RMSE)", size=15)
    ax.set_xlabel(xlabel, size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(accuracy_hist, lw=3)
    ax.set_title("Accuracy (Predicted/Actual)", size=15)
    ax.set_xlabel(xlabel, size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(task_loss_hist, lw=3)
    ax.set_title("Task Loss", size=15)
    ax.set_xlabel(xlabel, size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # save figure
    plt.savefig(save_dir)

    if show_plot:
        plt.show()



def evaluate_model_by_hour(X_test, y_test, eval_model, scaler, model_type, opt_weights, rms_model = None):
    """
    Returns the RMSE and accuracy by hour of day.
    :param X_test:
    :param y_test:
    :param eval_model: model you would like to evaluate
    :param scaler: sklearn scaler
    :param model_type: Either "RMS" or "opt"
    :param opt_weights: Weights for task loss
    :param rms_model: Only needs to be supplied when evaluating the optimization model
    :return:
    """
    assert model_type in ["RMS", "opt"]

    # normalize X_test and convert to tensors
    X_test_norm = scaler.transform(X_test)
    X_test_norm = torch.from_numpy(X_test_norm).float()
    y_test = torch.from_numpy(y_test.values).float()

    if(model_type == "RMS"):
        eval_model.eval()
        pred, _ = eval_model(X_test_norm, y_test)
    if(model_type == "opt"):
        rms_model.eval()
        mu, sigma = rms_model(X_test_norm, y_test)
        pred = eval_model(mu, mu, sigma)

    # compute rmse and accuracy for every hour
    rmse = [0] * 24
    accuracy = [0] * 24


    for hour in range(24):
        # get predictions and actuals for the hour
        pred_for_hour = pred[:, hour]
        actual_for_hour = y_test[:, hour]
        rmse[hour] = torch.sqrt(torch.mean(torch.square(pred_for_hour - actual_for_hour))).item()
        accuracy[hour] = torch.mean(torch.div(pred_for_hour, actual_for_hour)).item()

    task_loss_avg = task_loss(pred, y_test, opt_weights, per_hour = True).detach().numpy()

    return rmse, accuracy, task_loss_avg