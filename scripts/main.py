# Code outline:
# - Assume that data is already in cleaned format for now (may create a data cleaning function later)
# - Main function should:
#   - Run the data through neural net
#   - Then, take the neural net and run through stochastic optimization

# TODO: Make code structure more readable?
# TODO: Think of way to save model outputs in a good/systematic way - want to compare different model outputs.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch

# neural net model
from models import RMS_model
from model_setup import *

from train_eval import train_RMS_model
from train_eval import plot_loss_accuracy
from train_eval import evaluate_model_by_hour
from train_eval import train_optimization_model

# function run time
import time

# Read in cleaned data
X = pd.read_csv('../data/processed_data/pjm_load_data_2008-11_with_features.csv')
y = pd.read_csv('../data/processed_data/load_day.csv')

# Drop the dates
X_no_date = X.drop('date', axis=1)
y_no_date = y.drop('date', axis=1)

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_no_date, y_no_date, test_size=test_size, random_state=seed)
# Normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

# Convert to tensor
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train.values).float()

# First load forecasting RMS error only model
input_size = X_train_norm.shape[1]
output_size = 24 # number of hours in a day
rms_only_model = RMS_model(input_size, hidden_sizes, output_size)
rms_only_model, loss_hist, acc_hist = train_RMS_model(X_train_norm, y_train, rms_only_model, num_epochs)


plot_loss_accuracy(loss_hist, acc_hist, '../figures/RMSE_loss.png')
rmse_hour, accuracy_hour = evaluate_model_by_hour(X_test, y_test, rms_only_model, scaler, "RMS")
plot_loss_accuracy(rmse_hour, accuracy_hour, '../figures/RMSE_loss_by_hour.png')


# Now, use the stochastic model
opt_weights = {'c_ramp': 0.4,
               'gamma_under': 50.0,
               'gamma_over': 0.5}

optimization_start = time.time()
stochastic_model, stochastic_loss, stochastic_acc = train_optimization_model(X_train_norm, y_train, rms_only_model, num_epochs,
                                                                             opt_weights)
optimization_end = time.time()
optimization_total = optimization_end - optimization_start
print("Optimization training run time: ", optimization_total)

plot_loss_accuracy(stochastic_loss, stochastic_acc, '../figures/stochastic_loss.png')

opt_eval_start = time.time()
rmse_opt_hour, accuracy_opt_hour = evaluate_model_by_hour(X_test, y_test, stochastic_model, scaler, "opt", rms_only_model)
opt_eval_end = time.time()
opt_eval_total = opt_eval_end - opt_eval_start
print("Optimization evaluation run time:", opt_eval_total)

plot_loss_accuracy(rmse_opt_hour, accuracy_opt_hour, '../figures/stochastic_loss_by_hour.png')

