import torch
import torch.nn as nn
from torch.distributions.normal import Normal

#TODO: Class Constants?
## RMS Neural Network
class RMS_model(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size=24):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)
        return x

class stochastic_opt_model(nn.Module):

    def __init__(self, rms_model, opt_weights):
        """
        :param rms_model:
        :param opt_weights: dictionary with the following elements
            - c_ramp
            - gamma_over
            - gamma_under
        """
        super().__init__()
        self.parameters = rms_model.parameters
        self.opt_weights = opt_weights

    def calc_dz(self, x, mu, sigma):
        """
        :param mu:
        :param sigma:
        :return: d(alpha)/dz
        """
        norm_distribution = Normal(mu, sigma)
        cdf = norm_distribution.cdf(x)
        dz = cdf*(self.opt_weights['gamma_over'] + self.opt_weights['gamma_under']) - self.opt_weights['gamma_under']
        return dz

    def calc_dz2(self, x, mu, sigma):
        norm_distribution = Normal(mu, sigma)
        pdf = norm_distribution.pdf(x)
        dz2 = pdf * (self.opt_weights['gamma_over'] + self.opt_weights['gamma_under'])
        return dz2

    def forward(self, x):
        return x
        # sol = solvers.qp(P, q, G, h)
        # return sol['x']