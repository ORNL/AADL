import numpy as np
from numpy import linalg as LA


def determine_aggressive(X):
    R = np.diff(np.asmatrix(X))
    RR = np.matmul(np.transpose(R), R)
    RR = RR / LA.norm(RR, 2)
    (eigenval, eigenvec) = np.linalg.eig(RR)
    return np.amin(eigenval)


def rna(X, reg=0):
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
    RR = RR / LA.norm(RR, 2)

    # Solve (R'R + lambda I)z = 1
    (extr, c) = rna_precomputed(X, RR, reg)

    # Compute the extrapolation / weigthed mean  "sum_i c_i x_i", and return
    return extr, c


def min_eignevalRR(X):
    # Recovers parameters, ensure X is a matrix
    (d, k) = np.shape(X)
    k = k - 1
    X = np.asmatrix(X)

    # Compute the matrix of residuals
    R = np.diff(X)

    # "Square" the matrix, and normalize it
    RR = np.matmul(np.transpose(R), R)
    RR = RR / LA.norm(RR, 2)
    eigenvalues = LA.eigvalsh(RR)
    return np.amin(eigenvalues)


def rna_precomputed(X, RR, reg=0):
    # Regularized Nonlinear Acceleration, with RR precomputed
    # Same than rna, but RR is computed only once

    # Recovers parameters
    (d, k) = X.shape
    k = k - 1

    # RR is already computed, we do not need this step anymore

    # # Compute the matrix of residuals
    # R = np.diff(X);

    # # "Square" the matrix, and normalize it
    # RR = np.dot(np.transpose(R),R);
    # normRR = LA.norm(RR,2);
    # RR = RR/normRR;

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


def grid_search(logmin, logmax, num_grid_points, objective_functional, eigenval_offset=0):
    # Perform a logarithmic grid search between [10^logmin,10^logmax] using
    # k points. Return the best value found in the grid, i.e. the minimum value
    # of objective_functional. In other words, it returns the value satisfying
    #   argmin_{val in logspace(logmin,logmax)} objective_functional(val)

    # always test 0
    lambda_grid = np.append([1e-16], np.logspace(logmin, logmax, k)) - eigenval_offset;
    num_grid_points = num_grid_points + 1

    # pre-allocation
    vec = np.zeros(num_grid_points)

    # test all values in the grid
    for idx in range(lambda_grid.size):
        vec[idx] = objective_functional(lambda_grid[idx])

    # get the best value in the grid and return it
    idx = np.argmin(vec)
    return lambda_grid[idx]


def approx_line_search(obj_fun, x0, step):
    # Perform an approximate line search; i.e. find a good value t for the
    # problem
    #   min_t f(x0+t*step)

    # Define the anonymous function f(t), returning obj_fun(x0+t*step) for
    # fixed x0 and step
    d = len(x0);
    x0 = np.reshape(x0, (d, 1))
    step = np.reshape(step, (d, 1))
    objval_step = lambda t: obj_fun(x0 + t * step)

    # We multiply the value of t at each iteration, then stop when the function
    # value increases.
    oldval = objval_step(1)
    newval = objval_step(2)
    t = 2
    while newval < oldval:
        t = 2 * t
        oldval = newval
        newval = objval_step(t)

    return x0 + t * step / 2, t / 2


def adaptive_rna(X, obj_fun, lagrange=[-15, 1], eigenval_offset=0):
    # Adaptive regularized nonlinear acceleration
    #
    # Perform an adaptive search for lambda and the stepsize for the rna
    # algorithm. It automatically finds a good value of lambda wy using a grid
    # search, then find a good step size by an approximate line-search.
    #
    # X is the matrix of iterates, and obj_fun is the value of the objective
    # function that we want to minimize.

    d, k = X.shape
    k = k - 1

    if k == 0:
        return X, 1, 0
    # Precompute the residual matrix
    R = np.diff(X)
    RR = np.matmul(np.transpose(R), R)
    RR = RR / LA.norm(RR, 2)

    if lagrange is not None:
        # anonymous function, return the objective value of x_extrapolated(lambda)
        obj_extr = lambda lambda_val: obj_fun(rna_precomputed(X, RR, lambda_val)[0])

        lambda_opt = grid_search(lagrange[0], lagrange[1], k, obj_extr, eigenval_offset=eigenval_offset)

        # Retrieve the best extrapolated point found in the grisd search
        x_extr, c = rna_precomputed(X, RR, lambda_opt)
    else:
        x_extr, c = rna_precomputed(X, RR, 0)

    # Perform an approximate line-search
    step = x_extr[:, 0] - X[:, 0]
    (x_extr, t) = approx_line_search(obj_fun, X[:, 0], step)

    # Return the extraplated point, the coefficients of the weigthed mean and
    # the step size.
    x_extr = np.reshape(x_extr, (d, 1))
    return x_extr, c
