import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from qpth.qp import QPFunction
from torch.nn.parameter import Parameter


## RMS Neural Network
class RMS_model(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_sizes[1], output_size)
        )

        self.sigma = Parameter(torch.tensor([1]).repeat(24).float())

    def set_sigma(self, pred, actual):
        """
        Return RMSE of errors
        """
        sigma = (pred - actual).square().mean(axis = 0).sqrt()
        self.sigma.data = sigma

    def forward(self, x, y_actual):
        x = self.net(x)
        return x, self.sigma.expand(x.shape[0], 24)


## Define model that solves optimization problem
class stochastic_opt_model(nn.Module):

    def __init__(self, opt_weights):
        """
        :param opt_weights: dictionary
            Weights for optimization (task loss). Should have keys 'c_ramp', 'gamma_under', and 'gamma_over'
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
        Calculate first derivative of optimization problem.
        :param mu: torch tensor
        :param sigma: torch tensor
        :return:
        torch tensor
            d(alpha)/dz
        """
        norm_distribution = Normal(mu, sigma)
        cdf = norm_distribution.cdf(x)
        dz = cdf*(self.opt_weights['gamma_over'] + self.opt_weights['gamma_under']) - self.opt_weights['gamma_under']
        return torch.squeeze(dz)

    def calc_dz2(self, x, mu, sigma):
        """
        Calculate first derivative of optimization problem.
        :param mu: torch tensor
        :param sigma: torch tensor
        :return:
        torch tensor
            d(alpha)/dz
        """
        norm_distribution = Normal(mu, sigma)
        pdf = norm_distribution.log_prob(x).exp()
        dz2 = pdf * (self.opt_weights['gamma_over'] + self.opt_weights['gamma_under'])
        return torch.squeeze(dz2)

    def solve_QP(self, x, mu, sigma):
        """
        Solve quadratic program: find x that minimizes 0.5*(x.T)Qx + (P.T)x with constraint Gx <= h
        :param x: torch tensor
            Estimate of solution to QP
        :param mu: torch tensor
        :param sigma: torch tensor
        :return:
        """
        dz2 = self.calc_dz2(x, mu, sigma)
        Q = torch.diag(dz2 + 1)

        dz = self.calc_dz(x, mu, sigma)
        p = dz + x - mu
        solver = QPFunction(verbose=False)
        return torch.squeeze(solver(Q, p, self.G, self.h, self.e, self.e))

    def forward(self, x, mu, sigma):

        max_iteration = 10
        diff = 0.00001

        #Number of batches
        n_batch = x.shape[0]
        all_outputs = torch.Tensor()

        # Pause on gradient calculations while iterating
        x_no_grad = x.detach()
        mu_no_grad = mu.detach()
        sigma_no_grad = sigma.detach()

        # Solve QP for each (X, y) pair
        for batch in range(n_batch):

            mu_batch = mu_no_grad[batch]
            sigma_batch = sigma_no_grad[batch]
            x_batch = x_no_grad[batch]

            # Iterate until convergence or max_iterations reached
            for i in range(max_iteration):
                # Calculate P, q  needed for optimization
                old_x = x_batch
                delta_x = self.solve_QP(old_x, mu_batch, sigma_batch)
                difference = torch.linalg.norm(delta_x)
                print(f'Iteration {i} delta x: {difference}')
                x_batch = old_x + delta_x
                if(difference < diff):
                    break

            # Gradient producing iteration
            batch_solution = torch.unsqueeze(x_batch + self.solve_QP(x_batch, mu[batch], sigma[batch]), 0)
            all_outputs = torch.cat((all_outputs, batch_solution))

        return(all_outputs)
