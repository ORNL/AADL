#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:52:48 2020

@author: 7ml
"""

import numpy as np
from numpy import linalg as LA


def anderson(X, reg=0):
    # Regularized Nonlinear Acceleration
    # Take a matrix X of iterates, where X[:,i] is the ith iteration of the
    # fixed-point operation
    #   x_i = g(x_{i-1})
    #
    # reg is the regularization parameter used for solving the system
    #   (R'R + reg I)z = 1
    # where R is the matrix of residuals, i.e. R[:,i] = x_{i+1}-x_{i}

    # Recovers parameters, ensure X is a matrix
    (d, k) = np.shape(X)
    k = k - 1
    X = np.asmatrix(X)  # check if necessary

    # Compute the matrix of residuals
    R = np.diff(X)

    # "Square" the matrix, and normalize it
    RR = np.matmul(np.transpose(R), R)

    # Solve (R'R + lambda I)z = 1
    (extr, c) = anderson_precomputed(X, RR, reg)

    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    return extr, c


def anderson_precomputed(X, RR, reg=0):
    # Regularized Nonlinear Acceleration, with RR precomputed
    # Same than rna, but RR is computed only once

    # Recovers parameters
    (d, k) = X.shape
    k = k - 1

    # Solve (R'R + lambda I)z = 1
    reg_I = reg * np.eye(k)

    # In case of singular matrix, we solve using least squares instead
    try:
        z = np.linalg.solve(RR + reg_I, np.ones(k))
    except LA.linalg.LinAlgError:
        z = np.linalg.lstsq(RR+reg_I, np.ones(k), -1);
        z = z[0]

    # Recover weights c, where sum(c) = 1
    if np.abs(np.sum(z)) < 1e-10:
        z = np.ones(k)

    c = np.asmatrix(z / np.sum(z)).T

    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    extr = np.dot(X[:, 1:k + 1], c[:, 0])
    return np.array(extr), c


