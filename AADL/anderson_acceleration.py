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


def _sketched_system(DR, b, row_indices):
    """Return the rows used to estimate the Anderson mixing coefficients.

    The full ``DR`` and ``b`` are deliberately retained by the caller for the
    final extrapolation.  Sketching therefore changes only the inexpensive
    mixing coefficients, never the dimensionality of the returned iterate.
    """
    if row_indices is None:
        return DR, b
    if (not isinstance(row_indices, torch.Tensor)
            or row_indices.ndim != 1
            or row_indices.dtype != torch.long):
        raise ValueError("row_indices must be a one-dimensional torch.long tensor")
    if row_indices.numel() == 0:
        raise ValueError("row_indices must not be empty")
    row_indices = row_indices.to(device=DR.device)
    if int(row_indices.min()) < 0 or int(row_indices.max()) >= DR.size(0):
        raise ValueError("row_indices contains an out-of-range row")
    return DR.index_select(0, row_indices), b.index_select(0, row_indices)


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


def _equilibrate_columns(A):
    """Scale each column of ``A`` to unit L2 norm.

    Returns ``(A_scaled, scale)`` where ``scale[j] = ||A[:, j]||``. Zero-norm
    columns are left unscaled to avoid division by zero. Solving the scaled
    least-squares problem and dividing the solution by ``scale`` recovers the
    original solution while improving the conditioning of the solve.
    """
    scale = A.norm(dim=0)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return A / safe, safe


def _num_oldest_to_drop(A, cond_threshold):
    """Walker-Ni filtering: how many oldest (leftmost) columns of ``A`` to drop
    so the remaining matrix has 2-norm condition number <= ``cond_threshold``.

    Works on the small ``k x k`` Gram matrix ``G = A^T A`` (k = history depth),
    using ``cond(A[:, d:]) = sqrt(cond(G[d:, d:]))``. Returns 0 when filtering
    is disabled (``cond_threshold`` falsy / <= 0) or unnecessary.
    """
    if not cond_threshold or cond_threshold <= 0:
        return 0
    k = A.size(1)
    if k <= 1:
        return 0
    G = A.t() @ A
    eps = torch.finfo(G.dtype).eps
    for d in range(k - 1):
        ev = torch.linalg.eigvalsh(G[d:, d:])
        lo = ev[0].clamp_min(eps)
        if torch.sqrt(ev[-1] / lo) <= cond_threshold:
            return d
    return k - 1  # keep at least one column


def _iterative_refine(y, DR_high, b_high, scale_high, reg, steps, correction_solve):
    """Mixed-precision iterative refinement of the scaled LS variable ``y``.

    ``y`` solves ``min ||A_s y - b||^2 + reg ||y||^2`` with ``A_s`` the
    unit-column-scaled ``DR``. Each step forms the (regularized) normal-equation
    gradient residual in high precision and applies the supplied low-precision
    ``correction_solve`` (an approximate inverse of ``A_s^T A_s + reg I``),
    recovering high-precision accuracy at ``O(numel * k)`` cost per step.

    A monotone guard rejects any step that does not decrease the gradient-residual
    norm and stops early. This keeps refinement safe on very ill-conditioned
    systems (where the reduced-precision factor is a poor preconditioner and the
    plain iteration could diverge): the result is never worse than the input.
    Returns the refined ``y`` in ``DR_high.dtype``.
    """
    y = y.to(DR_high.dtype)
    A_s = DR_high / scale_high

    def _grad(v):
        return A_s.t() @ (b_high - A_s @ v) - reg * v

    g = _grad(y)
    gnorm = g.norm()
    for _ in range(steps):
        y_new = y + correction_solve(g)
        g_new = _grad(y_new)
        gnorm_new = g_new.norm()
        if not torch.isfinite(gnorm_new) or gnorm_new >= gnorm:
            break
        y, g, gnorm = y_new, g_new, gnorm_new
    return y



def _anderson_extrapolate(X, DX, DR, b, gamma, n_drop, relaxation):
    """Assemble the Anderson extrapolation from a mixing vector ``gamma``.

    ``gamma`` corresponds to columns ``[n_drop:]`` of ``DR`` (and of
    ``DX[:, :-1]``) after column filtering. Computed in ``X``'s dtype.
    """
    DXsub = DX[:, n_drop:-1]
    DR_k = DR[:, n_drop:]
    extr = X[:, -2] + b - (DXsub + DR_k) @ gamma
    return _apply_relaxation(extr, X[:, n_drop:], DX, gamma, relaxation)


def anderson_qr_factorization(X, relaxation=1.0, regularization=0.0, dtype=None,
                              equilibrate=True, filter_condition=0.0,
                              refinement_steps=0, row_indices=None):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]
    #
    # ``dtype``            precision used to compute the mixing vector.
    # ``equilibrate``      unit-scale DR columns before the solve (#5).
    # ``filter_condition`` >0 drops oldest columns until cond(DR) <= it (#4).
    # ``refinement_steps`` >0 does mixed-precision iterative refinement (#6).
    # The extrapolation is always assembled in X's original dtype.

    assert X.ndim == 2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    orig_dtype = X.dtype
    compute_dtype = _resolve_dtype(dtype)
    downcast = compute_dtype is not None and compute_dtype != orig_dtype

    # Differences in the original (high) precision; reused for the extrapolation
    # and the refinement residual.
    DX, DR = _compute_differences(X)
    b = DX[:, -1]

    # Matrix actually factorized (optionally in reduced precision).
    DR_s, b_s = _sketched_system(DR, b, row_indices)
    DR_c = DR_s.to(compute_dtype) if downcast else DR_s

    # #4 Walker-Ni column filtering.
    n_drop = _num_oldest_to_drop(DR_c, filter_condition)
    DR_c = DR_c[:, n_drop:]

    # #5 Column equilibration.
    if equilibrate:
        A_s, scale = _equilibrate_columns(DR_c)
    else:
        A_s = DR_c
        scale = torch.ones(DR_c.size(1), device=DR_c.device, dtype=DR_c.dtype)
    b_c = b_s.to(A_s.dtype)

    # Solve min_y || A_s y - b ||_2 (+ Tikhonov) via QR + triangular solve.
    # For tall-skinny A_s (numel >> history) this is measurably faster and more
    # stable than torch.linalg.lstsq, which dispatches to a generic SVD/LU
    # driver on most builds.
    if regularization == 0.0:
        A = A_s
        rhs = b_c
    else:
        sqrt_reg = torch.sqrt(torch.tensor(regularization, device=A_s.device, dtype=A_s.dtype))
        eye = torch.eye(A_s.size(1), device=A_s.device, dtype=A_s.dtype)
        zero_pad = torch.zeros(A_s.size(1), device=A_s.device, dtype=A_s.dtype)
        A = torch.cat((A_s, sqrt_reg * eye), dim=0)
        rhs = torch.cat((b_c, zero_pad))

    try:
        Q, R = torch.linalg.qr(A, mode='reduced')
        y = torch.linalg.solve_triangular(
            R, (Q.transpose(-2, -1) @ rhs).unsqueeze(-1), upper=True,
        ).squeeze(-1)
    except NotImplementedError as err:
        _reraise_linalg_dtype_error(err, A)

    # #6 Mixed-precision iterative refinement. R^T R = A_s^T A_s + reg I, so R
    # provides the correction operator for the regularized normal equations.
    if refinement_steps > 0:
        R_h = R.to(orig_dtype)

        def _corr(g):
            z = torch.linalg.solve_triangular(
                R_h.transpose(-2, -1), g.unsqueeze(-1), upper=False)
            dy = torch.linalg.solve_triangular(R_h, z, upper=True)
            return dy.squeeze(-1)

        scale_h = scale.to(orig_dtype)
        y = _iterative_refine(y, DR_s[:, n_drop:], b_s, scale_h,
                              regularization, refinement_steps, _corr)
        gamma = (y / scale_h).to(orig_dtype)
    else:
        gamma = (y / scale).to(orig_dtype)

    return _anderson_extrapolate(X, DX, DR, b, gamma, n_drop, relaxation)


def anderson_normal_equation(X, relaxation=1.0, regularization=0.0, dtype=None,
                             equilibrate=True, filter_condition=0.0,
                             refinement_steps=0, row_indices=None):
    # Anderson Acceleration via the normal equations
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]
    #
    # See ``anderson_qr_factorization`` for the shared options
    # (dtype / equilibrate / filter_condition / refinement_steps).

    assert X.ndim == 2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    orig_dtype = X.dtype
    compute_dtype = _resolve_dtype(dtype)
    downcast = compute_dtype is not None and compute_dtype != orig_dtype

    DX, DR = _compute_differences(X)
    b = DX[:, -1]

    DR_s, b_s = _sketched_system(DR, b, row_indices)
    DR_c = DR_s.to(compute_dtype) if downcast else DR_s

    # #4 Walker-Ni column filtering.
    n_drop = _num_oldest_to_drop(DR_c, filter_condition)
    DR_c = DR_c[:, n_drop:]

    # #5 Column equilibration.
    if equilibrate:
        A_s, scale = _equilibrate_columns(DR_c)
    else:
        A_s = DR_c
        scale = torch.ones(DR_c.size(1), device=DR_c.device, dtype=DR_c.dtype)
    b_c = b_s.to(A_s.dtype)

    RR = A_s.t() @ A_s
    if regularization != 0.0:
        RR = RR + regularization * torch.eye(A_s.size(1), device=A_s.device, dtype=A_s.dtype)

    projected_residual = A_s.t() @ b_c.unsqueeze(1)
    use_chol = False
    try:
        # RR is symmetric positive (semi-)definite. When it is positive definite
        # (always so for regularization > 0) a Cholesky factorization exploits
        # that structure and is ~2x faster and more stable than the general LU
        # driver used by torch.linalg.solve. cholesky_ex reports failure via an
        # info flag instead of raising, so we can cheaply fall back to LU when
        # RR is only semidefinite (e.g. a rank-deficient history at reg == 0).
        L, info = torch.linalg.cholesky_ex(RR)
        if int(info) == 0:
            use_chol = True
            y = torch.cholesky_solve(projected_residual, L).view(-1)
        else:
            y = torch.linalg.solve(RR, projected_residual).view(-1)
    except NotImplementedError as err:
        _reraise_linalg_dtype_error(err, RR)

    # #6 Mixed-precision iterative refinement via the (Cholesky/LU) factor of RR.
    if refinement_steps > 0:
        if use_chol:
            L_h = L.to(orig_dtype)

            def _corr(g):
                return torch.cholesky_solve(g.unsqueeze(1), L_h).view(-1)
        else:
            RR_h = RR.to(orig_dtype)

            def _corr(g):
                return torch.linalg.solve(RR_h, g.unsqueeze(1)).view(-1)

        scale_h = scale.to(orig_dtype)
        y = _iterative_refine(y, DR_s[:, n_drop:], b_s, scale_h,
                              regularization, refinement_steps, _corr)
        gamma = (y / scale_h).to(orig_dtype)
    else:
        gamma = (y / scale).to(orig_dtype)

    return _anderson_extrapolate(X, DX, DR, b, gamma, n_drop, relaxation)


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
