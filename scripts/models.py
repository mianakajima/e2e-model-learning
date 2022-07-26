import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from qpth.qp import QPFunction

import cvxopt
from cvxopt import matrix
from cvxopt import solvers

#TODO: Class Constants?
## RMS Neural Network
class RMS_model(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size=24):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], output_size)

    def calculate_sigma(self, pred, actual):
        """
        Return RMSE of errors
        """
        sigma = (pred - actual).square().mean(axis = 0).sqrt()
        return sigma.expand(actual.shape[0], 24)

    def forward(self, x, y_actual):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)

        sigma = self.calculate_sigma(x, y_actual)
        return x, sigma



class stochastic_opt_model(nn.Module):

    def __init__(self, opt_weights):
        """
        :param n: Number of hours in batch
        :param opt_weights: dictionary with the following elements
            - c_ramp
            - gamma_over
            - gamma_under
        """
        super().__init__()
        self.opt_weights = opt_weights
        # create G and h (constraints for optimization problem).
        negative_identity = torch.diag(torch.tensor([-1]).repeat(24))
        identity_offset = torch.diag(torch.tensor([1]).repeat(23), 1)
        added_matrix = negative_identity + identity_offset
        self.G = added_matrix[:23, :].float()  # remove last row

        self.h = torch.tensor([self.opt_weights['c_ramp']]).repeat(23).float()
        self.e = torch.Tensor().float()

    def calc_dz(self, x, mu, sigma):
        """
        :param mu:
        :param sigma:
        :return: d(alpha)/dz
        """
        norm_distribution = Normal(mu, sigma)
        cdf = norm_distribution.cdf(x)
        dz = cdf*(self.opt_weights['gamma_over'] + self.opt_weights['gamma_under']) - self.opt_weights['gamma_under']
        return torch.squeeze(dz)

    def calc_dz2(self, x, mu, sigma):
        norm_distribution = Normal(mu, sigma)
        pdf = norm_distribution.log_prob(x).exp()
        dz2 = pdf * (self.opt_weights['gamma_over'] + self.opt_weights['gamma_under'])
        return torch.squeeze(dz2)

    def solve_QP(self, x, mu, sigma):
        dz2 = self.calc_dz2(x, mu, sigma)
        Q = torch.diag(dz2 + 1)

        dz = self.calc_dz(x, mu, sigma)
        p = dz - mu
        solver = QPFunction()
        return torch.squeeze(solver(Q, p, self.G, self.h, self.e, self.e))

    def forward(self, x, mu, sigma):

        max_iteration = 10
        diff = 1e-10

        #Number of batches
        n_batch = x.shape[0]
        all_outputs = torch.Tensor()

        # Pause on gradient calculations while iterating
        x_no_grad = x.detach()
        mu_no_grad = mu.detach()
        sigma_no_grad = sigma.detach()

        for batch in range(n_batch):

            mu_batch = mu_no_grad[batch]
            sigma_batch = sigma_no_grad[batch]
            x_batch = x_no_grad[batch]

            for i in range(max_iteration):
                #print(i)
                # Calculate P, q  needed for optimization
                old_x = x_batch
                x_no_grad = self.solve_QP(x_batch, mu_batch, sigma_batch)
                if(torch.linalg.norm(old_x - x) < diff):
                    break

             # Gradient producing iteration

            x = torch.unsqueeze(self.solve_QP(x[0], mu[0], sigma[0]), 0)
            all_outputs = torch.cat((all_outputs, x))

        return(all_outputs)
