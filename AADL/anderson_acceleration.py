import torch


def _compute_differences(X):
    # DX[:,i] =  X[:,i+1] -  X[:,i]
    # DR[:,i] = DX[:,i+1] - DX[:,i] = X[:,i+2] - 2*X[:,i+1] + X[:,i]
    DX = X[:, 1:] - X[:, :-1]
    DR = DX[:, 1:] - DX[:, :-1]
    return DX, DR


def _apply_relaxation(extr, X, DX, gamma, relaxation):
    if relaxation == 1:
        return extr
    assert relaxation > 0, "relaxation must be positive"
    # solution of the constrained optimization problem s.t. gamma = X[:,1:]@alpha
    alpha = torch.zeros(gamma.numel() + 1, device=DX.device, dtype=DX.dtype)
    alpha[0] = gamma[0]
    alpha[1:-1] = gamma[1:] - gamma[:-1]
    alpha[-1] = 1 - gamma[-1]
    return relaxation * extr + (1 - relaxation) * X[:, :-1] @ alpha


def anderson_qr_factorization(X, relaxation=1.0, regularization=0.0):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim == 2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    DX, DR = _compute_differences(X)

    # Solve min_gamma || A gamma - b ||_2 explicitly via QR + triangular solve.
    # For tall-skinny A (numel >> history) this is measurably faster and more
    # stable than torch.linalg.lstsq, which dispatches to a generic SVD/LU
    # driver on most builds.
    if regularization == 0.0:
        A = DR
        b = DX[:, -1]
    else:
        # Augmented system for Tikhonov regularization.
        sqrt_reg = torch.sqrt(torch.tensor(regularization, device=DR.device, dtype=DR.dtype))
        eye = torch.eye(DR.size(1), device=DR.device, dtype=DR.dtype)
        zero_pad = torch.zeros(DR.size(1), device=DR.device, dtype=DR.dtype)
        A = torch.cat((DR, sqrt_reg * eye), dim=0)
        b = torch.cat((DX[:, -1], zero_pad), dim=0)

    Q, R = torch.linalg.qr(A, mode='reduced')
    gamma = torch.linalg.solve_triangular(
        R, (Q.transpose(-2, -1) @ b).unsqueeze(-1), upper=True,
    ).squeeze(-1)

    extr = X[:, -2] + DX[:, -1] - (DX[:, :-1] + DR) @ gamma
    return _apply_relaxation(extr, X, DX, gamma, relaxation)


def anderson_normal_equation(X, relaxation=1.0, regularization=0.0):
    # Anderson Acceleration via the normal equations
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim == 2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    DX, DR = _compute_differences(X)

    RR = DR.t() @ DR
    if regularization != 0.0:
        RR = RR + regularization * torch.eye(DR.size(1), device=DR.device, dtype=DR.dtype)

    projected_residual = DR.t() @ DX[:, -1].unsqueeze(1)
    gamma = torch.linalg.solve(RR, projected_residual).view(-1)

    extr = X[:, -2] + DX[:, -1] - (DX[:, :-1] + DR) @ gamma
    return _apply_relaxation(extr, X, DX, gamma, relaxation)


_ACCELERATIONS = {
    "anderson": anderson_qr_factorization,
    "anderson_normal_equation": anderson_normal_equation,
}


def get_acceleration(acc_type):
    """Look up an acceleration kernel by name. Raises ValueError on unknown name."""
    try:
        return _ACCELERATIONS[acc_type]
    except KeyError:
        raise ValueError(
            f"Unknown acceleration type {acc_type!r}; "
            f"expected one of {sorted(_ACCELERATIONS)}"
        )
