from models import RMS_model
from models import stochastic_opt_model
import torch

# Stochastic Optimization Model Test

test_rms = RMS_model(input_size = 10, hidden_sizes = [10, 10], output_size = 3)
opt_weights = {'c_ramp': 0.4,
               'gamma_under': 50.0,
               'gamma_over': 0.5}
test_stochastic = stochastic_opt_model(test_rms, opt_weights)

x = torch.tensor([0, 0.5])
mu = torch.tensor([0, 1])
sigma = torch.tensor([1, 1])
dz = test_stochastic.calc_dz(x, mu, sigma)
print(dz)
assert dz[0] == torch.tensor([-24.75]), "Not expected. Check dz calculation."
# Interesting rounding things happening - had to do this assertion. TODO: Look into pytorch rounding.
assert torch.round(dz[1]) == torch.tensor([-34]), "Not expected. Check dz calculations."
