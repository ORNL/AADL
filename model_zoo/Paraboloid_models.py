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

        # A is orthogonal matrix
        A, _ = torch.qr( torch.rand(dim,dim) )
        for i in range(dim):
            A[i,:] *= condition_number**(i/dim)
        self.register_buffer('A', A)

        self.weight = torch.nn.Parameter(torch.rand(dim))

    def forward(self, x):
        return torch.mv(self.A, self.weight)

    def get_model(self):
        return self

    def get_weight(self):
        return self.weight

    def get_device(self):
        return self.device
