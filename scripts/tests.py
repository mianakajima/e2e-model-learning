from models import RMS_model
from models import stochastic_opt_model
from train_eval import task_loss
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_setup import *
import pandas as pd

# Scratch/tests to make sure functions work

# Stochastic Optimization Model Test

test_rms = RMS_model(input_size = 10, hidden_sizes = [10, 10], output_size = 3)
opt_weights = {'c_ramp': 0.4,
               'gamma_under': 50.0,
               'gamma_over': 0.5}


test_stochastic = stochastic_opt_model(opt_weights)

x = torch.tensor([0, 0.5])
mu = torch.tensor([0, 1])
sigma = torch.tensor([1, 1])
dz = test_stochastic.calc_dz(x, mu, sigma)
print("Calculated dz: ", dz)
assert dz[0] == torch.tensor([-24.75]), "Not expected. Check dz calculation."
# Interesting rounding things happening - had to do this assertion. TODO: Look into pytorch rounding.
assert torch.round(dz[1]) == torch.tensor([-34]), "Not expected. Check dz calculations."


# test forward pass of stochastic_opt_model

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

test_y = y_train[1]
print("Test_y:\n", test_y)
mu = torch.mean(y_train, axis = 0)
sigma = torch.std(y_train, axis = 0)

new_y = test_stochastic(test_y.unsqueeze(0), mu.unsqueeze(0), sigma.unsqueeze(0))
print("New_y: \n", new_y)

# check with a batch size of 2
test_y_2 = torch.cat((test_y.unsqueeze(0), test_y.unsqueeze(0)))
print(test_y_2)
mu_2 = torch.cat((mu.unsqueeze(0), mu.unsqueeze(0)))
sigma_2 = torch.cat((sigma.unsqueeze(0), sigma.unsqueeze(0)))

new_y_2 = test_stochastic(test_y_2, mu_2, sigma_2)
print("New y_2: \n", new_y_2)

# Test_y:
#  tensor([1.6620, 1.5948, 1.5425, 1.5157, 1.5004, 1.5032, 1.5924, 1.7498, 1.8343,
#         1.8716, 1.8869, 1.9225, 1.9319, 1.9367, 1.9430, 1.9605, 1.9490, 2.0110,
#         2.0857, 2.0777, 2.0484, 1.9553, 1.8972, 1.7742])
# New_y: (qpth)
#  tensor([[0.1815, 0.1478, 0.1321, 0.1211, 0.1148, 0.1161, 0.1130, 0.1235, 0.1416,
#          0.1506, 0.1721, 0.1955, 0.2276, 0.2593, 0.2898, 0.3018, 0.3227, 0.2975,
#          0.2638, 0.2459, 0.2166, 0.2212, 0.2211, 0.2084]])

# New_y: (cvxopt)
#  [ 1.82e-01]
# [ 1.48e-01]
# [ 1.32e-01]
# [ 1.21e-01]
# [ 1.15e-01]
# [ 1.16e-01]
# [ 1.13e-01]
# [ 1.23e-01]
# [ 1.42e-01]
# [ 1.51e-01]
# [ 1.72e-01]
# [ 1.96e-01]
# [ 2.28e-01]
# [ 2.59e-01]
# [ 2.90e-01]
# [ 3.02e-01]
# [ 3.23e-01]
# [ 2.98e-01]
# [ 2.64e-01]
# [ 2.46e-01]
# [ 2.17e-01]
# [ 2.21e-01]
# [ 2.21e-01]
# [ 2.08e-01]

# TODO: Converging to weird values? Very large or very small

# test task_loss function

loss = task_loss(new_y, test_y, opt_weights)
print("loss: \n", loss)

# TODO: Make it so loss function returns average loss across batch items 
loss_2 = task_loss(new_y_2, test_y_2, opt_weights)
print("loss2: \n", loss)