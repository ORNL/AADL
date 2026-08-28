#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:30:19 2020

@author: 7ml
"""

import torch

###############################################################################


class Paraboloid(torch.nn.Module):
    def __init__(self, dim, condition_number=1.e3, device='cpu'):
        super().__init__()

        self.device = device

        # Scale the singular values so the quadratic loss Hessian has the
        # requested condition number (the Hessian contains A.T @ A).
        generator = torch.Generator().manual_seed(0)
        A, _ = torch.linalg.qr(torch.rand(dim, dim, generator=generator))
        for i in range(dim):
            A[i, :] *= condition_number**(i / (2 * max(dim - 1, 1)))
        self.register_buffer('A', A)

        self.weight = torch.nn.Parameter(torch.rand(dim, generator=generator))

    def forward(self, x):
        return torch.mv(self.A, self.weight)

    def get_model(self):
        return self

    def get_weight(self):
        return self.weight

    def get_device(self):
        return self.device



class Rosenbrock(torch.nn.Module):
    def __init__(self, dim, device='cpu', initial_guess=None):
        super().__init__()

        self.device = device

        if initial_guess is None:
            self.weight = torch.nn.Parameter(torch.rand(dim))
        else:
            initial_guess = torch.tensor(initial_guess, dtype=torch.float).view(-1)
            assert initial_guess.numel()==dim, "initial_guess has wrong dimension, need "+str(dim)+", got "+str(initial_guess.numel())
            self.weight = torch.nn.Parameter(initial_guess)

    def forward(self, x):
        f = 0
        for i in range(self.weight.numel()-1):
            f = f + 100 * (self.weight[i+1]-self.weight[i]**2)**2 + (1-self.weight[i])**2
        return f

    def get_model(self):
        return self

    def get_weight(self):
        return self.weight

    def get_device(self):
        return self.device
