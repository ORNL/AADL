#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:19:08 2020

@author: 7ml
"""

import torch

###############################################################################

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, use_bias: bool=True, device='cpu'):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, use_bias)

        self.model = torch.nn.Sequential(self.linear)

        self.device = torch.device(device)

    def forward(self, x):
        out = self.model(x)
        return out

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device