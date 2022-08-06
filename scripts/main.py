import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

import torch
# neural net model
from models import RMS_model
from model_setup import *

from train_eval import train_RMS_model
from train_eval import plot_loss_accuracy
from train_eval import evaluate_model_by_hour
from train_eval import train_optimization_model
from train_eval import convert_pred_tensor_to_pd

# function run time
import time

# set seed
torch.manual_seed(seed)

# Read in cleaned data
X = pd.read_csv('../data/processed_data/pjm_load_data_2008-11_with_features.csv')
y = pd.read_csv('../data/processed_data/load_day.csv')

X['date'] = pd.to_datetime(X['date'], format = "%Y-%m-%d %H:%M:%S%z", utc = True)
y['date'] = pd.to_datetime(y['date'], format = "%Y-%m-%d %H:%M:%S%z", utc = True)

# Split into train and test datasets - consider last year to be test set
X_train = X[X.date.dt.year != 2011]
X_test = X[X.date.dt.year == 2011]
y_train = y[y.date.dt.year != 2011]
y_test = y[y.date.dt.year == 2011]

# Keep test dates
y_test_dates = y_test['date'].to_list()

# Drop the dates
X_train.drop('date', axis = 1, inplace= True)
X_test.drop('date', axis = 1, inplace = True)
y_train.drop('date', axis = 1, inplace = True)
y_test.drop('date', axis = 1, inplace = True)

# Normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

# Convert to tensor
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train.values).float()

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
mse_hour, accuracy_hour, task_loss_hour, rmse_pred = evaluate_model_by_hour(X_test, y_test, rms_only_model, scaler, "RMS", opt_weights)
plot_loss_accuracy(np.sqrt(mse_hour), accuracy_hour, task_loss_hour, '../figures/RMSE_loss_by_hour.png', "Hour")

# Save Predictions
rmse_pred = convert_pred_tensor_to_pd(rmse_pred, y_test_dates)
rmse_pred.to_csv('../results/RMSE_predictions.csv')

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
mse_opt_hour, accuracy_opt_hour, task_opt_hour, pred_opt = evaluate_model_by_hour(X_test, y_test, stochastic_model, scaler, "opt", opt_weights, rms_only_model)
opt_eval_end = time.time()
opt_eval_total = opt_eval_end - opt_eval_start


plot_loss_accuracy(np.sqrt(mse_opt_hour), accuracy_opt_hour, task_opt_hour, '../figures/stochastic_loss_by_hour.png', "Hour")

# Save predictions
pred_opt = convert_pred_tensor_to_pd(pred_opt, y_test_dates)
pred_opt.to_csv('../results/taskloss_predictions.csv')

# Log the final training and test results
models = ['RMSE NN', 'Task Loss']
rmse_train = [loss_hist[-1], stochastic_loss[-1]]
accuracy_train = [acc_hist[-1], stochastic_acc[-1]]
task_loss_train = [task_loss_hist[-1], stochastic_task[-1]]
rmse_test = [np.sqrt(np.mean(mse_hour)), np.sqrt(np.mean(mse_opt_hour))]
accuracy_test = [np.mean(accuracy_hour), np.mean(accuracy_opt_hour)]
task_loss_test = [np.sum(task_loss_hour), np.sum(task_opt_hour)]

results = pd.DataFrame({'Model': models,
                        'Training RMSE': rmse_train,
                        'Training Accuracy': accuracy_train,
                        'Training Task Loss': task_loss_train,
                        'Test RMSE': rmse_test,
                        'Test Accuracy': accuracy_test,
                        'Test Task Loss': task_loss_test})

results.to_csv('../results/metrics.csv')

# Save hourly results
hourly_results = pd.DataFrame({'Hour': range(24),
                               'RMSEnn_rmse': np.sqrt(mse_hour),
                               'RMSEnn_acc': accuracy_hour,
                               'RMSEnn_taskloss': task_loss_hour,
                               'Taskloss_rmse': np.sqrt(mse_opt_hour),
                               'Taskloss_acc': accuracy_opt_hour,
                               'Taskloss_taskloss': task_opt_hour})
hourly_results.to_csv('../results/hourly_results.csv')

# Print run time for reference
print("Optimization training run time: ", optimization_total)
print("Optimization evaluation run time:", opt_eval_total)


# Experimental runs with different weights for optimization

# Equal weights for task loss
opt_weights_0 = {'c_ramp': 0.4,
               'gamma_under': 25.0,
               'gamma_over': 25.0}

# Overgenerating penalty by hour to simulate penalty for overgenerating when grid is less clean
emissions_schedule = np.array([0.43, 0.42, 0.40, 0.40, 0.41, 0.41, 0.41, 0.39, 0.34, 0.32, 0.32, 0.32,
                        0.32, 0.33, 0.34, 0.36, 0.40, 0.43, 0.46, 0.48, 0.48, 0.47, 0.46, 0.44])

# Try to make gamma equal on average

opt_weights_1 = {'c_ramp': 0.4,
               'gamma_under': 25.0,
               'gamma_over': torch.tensor(25.0/np.mean(emissions_schedule) * emissions_schedule).float()}

# min max scale to increase difference between low emission and high emission times.
emissions_schedule = (emissions_schedule - np.min(emissions_schedule))/(np.max(emissions_schedule) - np.min(emissions_schedule))

opt_weights_2 = {'c_ramp': 0.4,
               'gamma_under': 25.0,
               'gamma_over': torch.tensor(25.0/np.mean(emissions_schedule) * emissions_schedule).float()}


weights_list = [opt_weights_0, opt_weights_1, opt_weights_2]

for i, weight in enumerate(weights_list):

    # Load in prior trained RMSE moodel
    rms_only_model = RMS_model(input_size, hidden_sizes, output_size)
    rms_only_model.load_state_dict(torch.load('../results/rms_only_model.pt'))

    # Train optimization model
    stochastic_model, stochastic_loss, stochastic_acc, stochastic_task = train_optimization_model(X_train_norm, y_train, rms_only_model, num_epochs_opt,
                                                                                 weight)

    plot_loss_accuracy(stochastic_loss, stochastic_acc, stochastic_task, f'../figures/stochastic_loss_exp_{i}.png', "Epochs")

    mse_opt_hour, accuracy_opt_hour, task_opt_hour, pred_opt = evaluate_model_by_hour(X_test, y_test, stochastic_model, scaler, "opt", weight, rms_only_model)

    plot_loss_accuracy(np.sqrt(mse_opt_hour), accuracy_opt_hour, task_opt_hour, f'../figures/stochastic_loss_by_hour_exp_{i}.png', "Hour")

    # Save predictions
    pred_opt = convert_pred_tensor_to_pd(pred_opt, y_test_dates)
    pred_opt.to_csv(f'../results/taskloss_predictions_exp_{i}.csv')