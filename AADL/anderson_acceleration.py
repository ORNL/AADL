import torch


def _resolve_dtype(dtype):
    """Normalize a dtype specifier to a ``torch.dtype`` (or ``None``).

    Accepts ``None`` (keep the input dtype), a ``torch.dtype``, or a string
    naming one (e.g. ``"float64"``, ``"float32"``, ``"bfloat16"``).
    """
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(f"Unsupported dtype specifier: {dtype!r}")


def _reraise_linalg_dtype_error(err, tensor):
    """Turn a backend ``NotImplementedError`` from an unsupported mixing dtype
    into an actionable message. Re-raises unrelated errors unchanged.
    """
    raise NotImplementedError(
        f"Anderson mixing solve is not supported for dtype {tensor.dtype} on "
        f"device '{tensor.device.type}' by this torch/LAPACK build "
        f"(original error: {str(err).splitlines()[0]}). "
        f"Use mixing_dtype=float32 or float64."
    ) from err


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


def anderson_qr_factorization(X, relaxation=1.0, regularization=0.0, dtype=None):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]
    #
    # ``dtype`` optionally selects the floating-point precision used to compute
    # the mixing vector (e.g. float32 for speed, float64 for accuracy). The
    # returned extrapolation is always cast back to X's original dtype.

    assert X.ndim == 2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    orig_dtype = X.dtype
    compute_dtype = _resolve_dtype(dtype)
    if compute_dtype is not None and compute_dtype != orig_dtype:
        X = X.to(compute_dtype)

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

    try:
        Q, R = torch.linalg.qr(A, mode='reduced')
        gamma = torch.linalg.solve_triangular(
            R, (Q.transpose(-2, -1) @ b).unsqueeze(-1), upper=True,
        ).squeeze(-1)
    except NotImplementedError as err:
        _reraise_linalg_dtype_error(err, A)

    extr = X[:, -2] + DX[:, -1] - (DX[:, :-1] + DR) @ gamma
    extr = _apply_relaxation(extr, X, DX, gamma, relaxation)
    return extr.to(orig_dtype) if extr.dtype != orig_dtype else extr


def anderson_normal_equation(X, relaxation=1.0, regularization=0.0, dtype=None):
    # Anderson Acceleration via the normal equations
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]
    #
    # ``dtype`` optionally selects the floating-point precision used to compute
    # the mixing vector (e.g. float32 for speed, float64 for accuracy). The
    # returned extrapolation is always cast back to X's original dtype.

    assert X.ndim == 2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    orig_dtype = X.dtype
    compute_dtype = _resolve_dtype(dtype)
    if compute_dtype is not None and compute_dtype != orig_dtype:
        X = X.to(compute_dtype)

    DX, DR = _compute_differences(X)

    RR = DR.t() @ DR
    if regularization != 0.0:
        RR = RR + regularization * torch.eye(DR.size(1), device=DR.device, dtype=DR.dtype)

    projected_residual = DR.t() @ DX[:, -1].unsqueeze(1)
    try:
        gamma = torch.linalg.solve(RR, projected_residual).view(-1)
    except NotImplementedError as err:
        _reraise_linalg_dtype_error(err, RR)

    extr = X[:, -2] + DX[:, -1] - (DX[:, :-1] + DR) @ gamma
    extr = _apply_relaxation(extr, X, DX, gamma, relaxation)
    return extr.to(orig_dtype) if extr.dtype != orig_dtype else extr


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
