#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:52:48 2020

@author: 7ml
"""

import math
import numpy as np


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

    projected_residual = DX[:, k - 1]
    DX = DX[:, :-1]

    # Solve (R'R + lambda I)z = 1
    (extr, c) = anderson_precomputed(DX, DR, projected_residual, reg)

    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    return extr, c


def anderson_precomputed(DX, DR, residual, reg=0):
    # Regularized Nonlinear Acceleration, with RR precomputed
    # Same than rna, but RR is computed only once

    # Recovers parameters
    (d, k) = DX.shape
   
    RR = np.matmul(np.transpose(DR), DR)
    
    if math.sqrt(np.linalg.cond(RR, 'fro')) < 1e5:

        # In case of singular matrix, we solve using least squares instead
        q, r = np.linalg.qr(DR)   
        
        new_residual = np.matmul(np.transpose(q), residual)      
        z = np.linalg.lstsq(r, new_residual, reg)
        z = z[0]
        
        # Recover weights c, where sum(c) = 1
        if np.abs(np.sum(z)) < 1e-10:
            z = np.ones((k, 1))
    
        alpha = np.asmatrix(z / np.sum(z))
        
    else:
        
        alpha  = np.zeros((DX.shape[1],1))
        
    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    extr = np.matmul(DX, alpha)
    return np.array(extr), alpha