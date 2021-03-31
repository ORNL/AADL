#!/usr/bin/env python3

import torch

def anderson_lstsq(X, relaxation=1.0):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim==2, "X must be a matrix"

    # Compute residuals
    DX =  X[:,1:] -  X[:,:-1] # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR = DX[:,1:] - DX[:,:-1] # DR[:,i] = DX[:,i+1] - DX[:,i] = X[:,i+2] - 2*X[:,i+1] + X[:,i]

    # # use QR factorization
    # q, r = torch.qr(DR)
    # gamma, _ = torch.triangular_solve( (q.t()@DX[:,-1]).unsqueeze(1), r )
    # gamma = gamma.squeeze(1)

    # solve unconstrained least-squares problem
    gamma, _ = torch.lstsq( DX[:,-1].unsqueeze(1), DR )
    gamma = gamma.squeeze(1)[:DR.size(1)]

    # compute acceleration
    extr = X[:,-2] + DX[:,-1] - (DX[:,:-1]+DR)@gamma

    if relaxation!=1:
        assert relaxation>0, "relaxation must be positive"
        # compute solution of the contraint optimization problem s.t. gamma = X[:,1:]@alpha
        alpha = torch.zeros(gamma.numel()+1).to(DX.device)
        alpha[0]    = gamma[0]
        alpha[1:-1] = gamma[1:] - gamma[:-1]
        alpha[-1]   = 1 - gamma[-1]
        extr = relaxation*extr + (1-relaxation)*X[:,:-1]@alpha

    return extr


def anderson_ne(X, relaxation=1.0):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim==2, "X must be a matrix"

    # Compute residuals
    DX =  X[:,1:] -  X[:,:-1] # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR = DX[:,1:] - DX[:,:-1] # DR[:,i] = DX[:,i+1] - DX[:,i] = X[:,i+2] - 2*X[:,i+1] + X[:,i]

    # # use QR factorization
    # q, r = torch.qr(DR)
    # gamma, _ = torch.triangular_solve( (q.t()@DX[:,-1]).unsqueeze(1), r )
    # gamma = gamma.squeeze(1)

    # solve unconstrained least-squares problem
    
    RR = DR.t()@DR
    projected_residual = DR.t()@DX[:,-1].unsqueeze(1)
    
    gamma, _ = torch.solve( projected_residual, RR )
    gamma = gamma.squeeze(1)[:DR.size(1)]

    # compute acceleration
    extr = X[:,-2] + DX[:,-1] - (DX[:,:-1]+DR)@gamma

    if relaxation!=1:
        assert relaxation>0, "relaxation must be positive"
        # compute solution of the contraint optimization problem s.t. gamma = X[:,1:]@alpha
        alpha = torch.zeros(gamma.numel()+1).to(DX.device)
        alpha[0]    = gamma[0]
        alpha[1:-1] = gamma[1:] - gamma[:-1]
        alpha[-1]   = 1 - gamma[-1]
        extr = relaxation*extr + (1-relaxation)*X[:,:-1]@alpha

    return extr