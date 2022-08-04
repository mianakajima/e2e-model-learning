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

# set seed
torch.manual_seed(seed)

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

emissions_schedule = [0.43, 0.42, 0.40, 0.40, 0.41, 0.41, 0.41, 0.39, 0.34, 0.32, 0.32, 0.32,
                        0.32, 0.33, 0.34, 0.36, 0.40, 0.43, 0.46, 0.48, 0.48, 0.47, 0.46, 0.44]

# Weights for task loss
opt_weights = {'c_ramp': 0.4,
               'gamma_under': 50.0,
               'gamma_over': 0.5}


# First load forecasting RMS error only model
input_size = X_train_norm.shape[1]
output_size = 24 # number of hours in a day
rms_only_model = RMS_model(input_size, hidden_sizes, output_size)
rms_only_model, loss_hist, acc_hist, task_loss_hist = train_RMS_model(X_train_norm, y_train, rms_only_model, num_epochs, opt_weights)

torch.save(rms_only_model.state_dict(), '../results/rms_only_model.pt')

plot_loss_accuracy(loss_hist, acc_hist, task_loss_hist, '../figures/RMSE_loss.png', "Epochs")
rmse_hour, accuracy_hour, task_loss_hour = evaluate_model_by_hour(X_test, y_test, rms_only_model, scaler, "RMS", opt_weights)
plot_loss_accuracy(rmse_hour, accuracy_hour, task_loss_hour, '../figures/RMSE_loss_by_hour.png', "Hour")


# Now, use the task loss/stochastic optimization model
optimization_start = time.time()
stochastic_model, stochastic_loss, stochastic_acc, stochastic_task = train_optimization_model(X_train_norm, y_train, rms_only_model, num_epochs_opt,
                                                                             opt_weights)
optimization_end = time.time()
optimization_total = optimization_end - optimization_start
torch.save(stochastic_model.state_dict(), '../results/optimization_model.pt')
torch.save(rms_only_model.state_dict(), '../results/task_loss_model.pt')

plot_loss_accuracy(stochastic_loss, stochastic_acc, stochastic_task, '../figures/stochastic_loss.png', "Epochs")

opt_eval_start = time.time()
rmse_opt_hour, accuracy_opt_hour, task_opt_hour = evaluate_model_by_hour(X_test, y_test, stochastic_model, scaler, "opt", opt_weights, rms_only_model)
opt_eval_end = time.time()
opt_eval_total = opt_eval_end - opt_eval_start


plot_loss_accuracy(rmse_opt_hour, accuracy_opt_hour, task_opt_hour, '../figures/stochastic_loss_by_hour.png', "Hour")

# Log the final results
models = ['RMSE NN', 'Task Loss']
rmse = [loss_hist[-1], stochastic_loss[-1]]
accuracy = [acc_hist[-1], stochastic_acc[-1]]
task_loss = [task_loss_hist[-1], stochastic_task[-1]]
results = pd.DataFrame({'Model': models,
                        'RMSE': rmse,
                        'Accuracy': accuracy,
                        'Task Loss': task_loss})
results.to_csv('../results/metrics.csv')

# Save hourly results
hourly_results = pd.DataFrame({'Hour': range(24),
                               'RMSEnn_rmse': rmse_hour,
                               'RMSEnn_acc': accuracy_hour,
                               'RMSEnn_taskloss': task_loss_hour,
                               'Taskloss_rmse': rmse_opt_hour,
                               'Taskloss_acc': accuracy_opt_hour,
                               'Taskloss_taskloss': task_opt_hour})
hourly_results.to_csv('../results/hourly_results.csv')

# Print run time for reference
print("Optimization training run time: ", optimization_total)
print("Optimization evaluation run time:", opt_eval_total)