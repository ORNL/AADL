#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:52:48 2020

@author: 7ml
"""

import numpy as np
from numpy import linalg as LA


def anderson(X, reg=0):
    # Anderson Acceleration
    # Take a matrix X of iterates, where X[:,i] is the difference between the {i+1}th and the ith iterations of the
    # fixed-point operation
    #   x_i = g(x_{i-1})
    #   
    #   r_i = x_{i+1} - x_i
    #   X[:,i] = r_i
    #
    # reg is the regularization parameter used for solving the system
    #   (F'F + reg I)z = r_{i+1}
    # where F is the matrix of differences in the residual, i.e. R[:,i] = r_{i+1}-r_{i}

    # Recovers parameters, ensure X is a matrix
    (d, k) = np.shape(X)
    k = k - 1
    X = np.asmatrix(X)  # check if necessary

    # Compute the matrix of residuals
    DX = np.diff(X)
    DR = np.diff(DX)

    projected_residual = np.matmul(DR.T, DX[:,k-1])
    DX = DX[:,:-1]

    # "Square" the matrix, and normalize it
    RR = np.matmul(np.transpose(DR), DR)

    # Solve (R'R + lambda I)z = 1
    (extr, c) = anderson_precomputed(DX, RR, projected_residual, reg)

    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    return extr, c


def anderson_precomputed(DX, RR, residual, reg=0):
    # Regularized Nonlinear Acceleration, with RR precomputed
    # Same than rna, but RR is computed only once

    # Recovers parameters
    (d, k) = DX.shape

    # Solve (R'R + lambda I)z = 1
    reg_I = reg * np.eye(k)

    # In case of singular matrix, we solve using least squares instead
    try:
        z = np.linalg.solve(RR + reg_I, residual)
    except LA.linalg.LinAlgError:
        z = np.linalg.lstsq(RR+reg_I, residual, -1)
        z = z[0]

    # Recover weights c, where sum(c) = 1
    if np.abs(np.sum(z)) < 1e-10:
        z = np.ones((k,1))

    alpha = np.asmatrix(z / np.sum(z))

    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    extr = np.matmul(DX, alpha)
    return np.array(extr), alpha


