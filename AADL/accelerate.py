import math

import torch

from AADL.utils import (
    buffer_row_to_parameters_,
    parameters_to_buffer_row_,
    parameters_to_vector_device,
    vector_to_parameters,
)

from collections import deque
from types import MethodType

import AADL.anderson_acceleration as anderson


def _ensure_buffer(self, state, params):
    """Lazily allocate the (capacity, numel) ring buffer for one param group."""
    if state['buf'] is not None:
        return
    numel = sum(p.numel() for p in params)
    dtype = params[0].dtype
    state['buf'] = torch.empty(
        self.acc_history_depth, numel,
        device=self.history_device, dtype=dtype,
    )


def _last_row(state, capacity):
    """Return the row of ``state['buf']`` holding the most-recent stored vector."""
    return state['buf'][(state['count'] - 1) % capacity]


def _history_chrono(state, capacity, compute_device):
    """Return chronologically-ordered iterates as a (numel, n) tensor on
    ``compute_device``, or ``None`` if fewer than 3 entries are available.

    One contiguous allocation regardless of history depth (replaces a
    per-column ``torch.stack`` over a deque of independent tensors).
    """
    count = state['count']
    n = min(count, capacity)
    if n < 3:
        return None
    buf = state['buf']
    if count <= capacity:
        rows = buf[:n]
    else:
        idx = count % capacity
        rows = torch.cat((buf[idx:], buf[:idx]), dim=0)
    # Move once, then transpose; .contiguous() does the single (numel, n) alloc.
    return rows.to(device=compute_device).t().contiguous()


def _store_current_params(self):
    """Append current parameters to the ring buffer if it's time to.

    Optimizations vs. the previous deque-based version:
      * O(1) allocation per step (writes into a pre-allocated buffer row)
      * single fused ``torch._foreach_copy_`` kernel instead of per-tensor copies
      * skips entirely during warmup if the stored value would be evicted
        before the first acceleration call
    """
    # #1: skip during warmup when the stored value would be evicted before
    # acceleration is enabled.
    capacity = self.acc_history_depth
    if (self.acc_call_counter + capacity * self.acc_store_each_nth
            <= self.acc_wait_iterations):
        return

    self.acc_store_counter += 1
    if self.acc_store_counter < self.acc_store_each_nth:
        return
    self.acc_store_counter = 0

    for group, state in zip(self.param_groups, self.acc_param_hist):
        params = group['params']
        _ensure_buffer(self, state, params)
        col_idx = state['count'] % capacity
        row = state['buf'][col_idx]
        parameters_to_buffer_row_(params, row)
        state['count'] += 1


def _moving_average_step(self):
    """Replace current parameters with a moving average of the recent history."""
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        group_hist.append(parameters_to_vector_device(group['params'], self.history_device))

    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)
        average = torch.mean(X, dim=1)
        std = torch.std(X, dim=1, correction=0)
        # Use magnitudes and a finite denominator: a zero or negative mean
        # should not make the relative-variation test invalid.
        scale = average.abs().amax().clamp_min(torch.finfo(average.dtype).eps)
        if std.amax() / scale > 0.1:
            vector_to_parameters(average, group['params'])


def _safeguard_accept(self, closure, base_loss):
    """Decide whether to accept the accelerated step.

    ``base_loss`` must be the loss of the *un-accelerated* iterate that the step
    would revert to (i.e. the plain optimizer step), so that acceptance is
    consistent with the fallback: the accelerated step is kept only when it is
    strictly better than not accelerating.

    Returns (accept: bool, acc_loss). When closure is None, the step is
    always accepted (no information available to compare).
    """
    if closure is None or not getattr(self, "acc_safeguard", True):
        return True, base_loss

    acc_loss = closure()
    return acc_loss < base_loss, acc_loss


@torch.no_grad()
def _unified_step(self, closure=None):
    """Apply the underlying optimizer step and optional local acceleration."""
    if self.acc_average_pre_step:
        # moving-average sweep before the underlying optimizer step
        _moving_average_step(self)

    orig_loss = self.orig_step(closure)
    self.acc_last_plain = None

    _store_current_params(self)

    self.acc_call_counter += 1
    ready = (
        self.acc_call_counter > self.acc_wait_iterations
        and self.acc_call_counter % self.acc_frequency == 0
    )
    if not ready:
        return orig_loss

    accel_fn = anderson.get_acceleration(self.acc_type)
    capacity = self.acc_history_depth

    # Baseline for the safeguard: the loss at the plain (un-accelerated) step,
    # which is exactly the iterate we revert to if the candidate is rejected.
    # Evaluating it here (params are still the plain step) keeps acceptance and
    # fallback consistent. Costs one extra forward eval per acceleration cycle.
    safeguard_closure = closure if self.acc_safeguard else None
    base_loss = safeguard_closure() if safeguard_closure is not None else orig_loss
    candidates = []
    for group, state in zip(self.param_groups, self.acc_param_hist):
        X = _history_chrono(state, capacity, self.compute_device)
        if X is None:
            continue
        acc_param = accel_fn(
            X, self.acc_relaxation, self.acc_reg, self.acc_dtype,
            equilibrate=self.acc_equilibrate,
            filter_condition=self.acc_filter_condition,
            refinement_steps=self.acc_refinement_steps,
        )
        candidates.append((group, state, acc_param))

    if not candidates:
        return base_loss

    # Retain the local plain iterate for an optional native model-averaging
    # boundary comparison. Only parameters participating in this backward pass
    # are included, matching PyTorch's model-averaging filter.
    self.acc_last_plain = [
        (param, param.detach().clone())
        for group, _, _ in candidates
        for param in group["params"]
        if param.grad is not None
    ]

    # Apply every group before evaluating the candidate. Acceptance must be an
    # optimizer-wide transaction; evaluating groups one at a time makes the
    # result depend on parameter-group ordering and can leave a hybrid state.
    for group, _, acc_param in candidates:
        vector_to_parameters(acc_param, group['params'])

    accepted, acc_loss = _safeguard_accept(self, safeguard_closure, base_loss)

    for group, state, acc_param in candidates:
        last_row = _last_row(state, capacity)
        if accepted:
            last_row.copy_(acc_param)
        else:
            buffer_row_to_parameters_(last_row, group['params'])

    return acc_loss if accepted else base_loss


# ---------------------------------------------------------------------------
# Backwards-compatible aliases so external code importing the old function
# names keeps working.
accelerated_step = _unified_step
averaged_accelerated_step = _unified_step


@torch.no_grad()
def averaged_step(self, closure=None):
    self.orig_step(closure)
    _moving_average_step(self)


def accelerate(
    optimizer,
    acceleration_type: str = "identity",
    relaxation: float = 0.1,
    wait_iterations: int = 1,
    history_depth: int = 15,
    store_each_nth: int = 1,
    frequency: int = 1,
    reg_acc: float = 0.0,
    average: bool = False,
    history_device: str = "cpu",
    compute_device: str = "cpu",
    mixing_dtype=None,
    equilibrate: bool = True,
    filter_condition: float = 0.0,
    refinement_steps: int = 0,
    safeguard: bool = True,
):
    """Wrap ``optimizer.step`` to apply Anderson-type acceleration.

    The wrapped ``step`` first delegates to the underlying optimizer, stores
    the resulting parameter vector in a per-group ring buffer, and -- once
    enough history has accumulated -- replaces the parameters with an
    Anderson-accelerated extrapolation.  When a ``closure`` is supplied the
    accelerated step is safeguarded by re-evaluating the loss and reverting
    to the plain optimizer step whenever the accelerated iterate is not
    strictly better than that plain step.

    Notes on ``closure``:
        Passing a ``closure`` enables the safeguard above but costs extra
        forward passes on each acceleration cycle: one to evaluate the plain
        step (the comparison baseline / revert target) plus one per candidate
        extrapolation.  To amortize this, increase ``frequency`` so
        acceleration (and the extra forwards) is attempted only every Nth
        optimizer step.  When ``closure`` is ``None`` no safeguard is
        performed and the accelerated step is always accepted.

    Parameters
    ----------
    acceleration_type : {"identity", "anderson", "anderson_normal_equation"}
    relaxation : float in (0, 1]
        Convex mixing weight between the Anderson extrapolation and the
        constrained-LS combination of past iterates.
    history_depth : int
        Capacity of the ring buffer of past iterates.
    store_each_nth, frequency : int
        Cadence of buffer writes / acceleration attempts.
    reg_acc : float >= 0
        Tikhonov regularization for the inner least-squares solve.
    mixing_dtype : None | torch.dtype | str
        Floating-point precision used to compute the Anderson mixing vector
        (e.g. ``torch.float32`` or ``"float64"``). ``None`` keeps the
        parameter dtype. Lower precision speeds up the least-squares solve;
        the extrapolated parameters are always cast back to their original
        dtype before being written to the model.
    equilibrate : bool
        Scale the columns of the difference matrix to unit norm before the
        least-squares solve (improves conditioning; on by default).
    filter_condition : float
        If > 0, drop the oldest history columns until the 2-norm condition
        number of the least-squares matrix falls below this threshold
        (Walker-Ni filtering). 0 disables filtering.
    refinement_steps : int
        Number of mixed-precision iterative-refinement steps applied to the
        mixing vector (residual formed in the parameter dtype, correction via
        the reduced-precision factor). Useful together with a low
        ``mixing_dtype`` to recover accuracy cheaply. 0 disables refinement.
    safeguard : bool
        When true, a supplied closure is used to accept the optimizer-wide
        accelerated candidate only if it improves on the plain optimizer step.
    """
    if hasattr(optimizer, "acc_type"):
        raise ValueError(
            "optimizer is already wrapped by AADL; call remove_acceleration() "
            "before wrapping it again"
        )

    def _positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    _positive_int("history_depth", history_depth)
    _positive_int("store_each_nth", store_each_nth)
    _positive_int("frequency", frequency)
    if (isinstance(wait_iterations, bool)
            or not isinstance(wait_iterations, int)
            or wait_iterations < 0):
        raise ValueError("wait_iterations must be a non-negative integer")
    if (not isinstance(relaxation, (int, float))
            or isinstance(relaxation, bool)
            or not math.isfinite(relaxation)
            or not 0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")
    if not isinstance(reg_acc, (int, float)) or not math.isfinite(reg_acc) or reg_acc < 0.0:
        raise ValueError("reg_acc must be non-negative")
    if (not isinstance(filter_condition, (int, float))
            or not math.isfinite(filter_condition)
            or filter_condition < 0.0):
        raise ValueError("filter_condition must be non-negative")
    if (isinstance(refinement_steps, bool)
            or not isinstance(refinement_steps, int)
            or refinement_steps < 0):
        raise ValueError("refinement_steps must be a non-negative integer")
    # validate acceleration type early
    acc_type = acceleration_type.lower()
    if acc_type != "identity":
        anderson.get_acceleration(acc_type)  # raises ValueError if unknown

    # acceleration options
    optimizer.acc_type            = acc_type
    optimizer.acc_wait_iterations = wait_iterations
    optimizer.acc_relaxation      = relaxation
    optimizer.acc_history_depth   = history_depth
    optimizer.acc_store_each_nth  = store_each_nth
    optimizer.acc_frequency       = frequency
    optimizer.acc_reg             = reg_acc
    optimizer.acc_dtype           = anderson._resolve_dtype(mixing_dtype)
    optimizer.acc_equilibrate     = equilibrate
    optimizer.acc_filter_condition = filter_condition
    optimizer.acc_refinement_steps = refinement_steps
    optimizer.acc_safeguard       = safeguard
    optimizer.acc_average_pre_step = average and acc_type != "identity"

    # acceleration history: ring buffer per param group, lazily allocated
    # on first store so we can pick up dtype/numel from the actual params.
    optimizer.acc_param_hist = [{'buf': None, 'count': 0} for _ in optimizer.param_groups]
    # avg_param_hist is only used by the (rare) moving-average path; keep
    # the simple deque representation.
    optimizer.avg_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]

    optimizer.acc_call_counter  = 0
    optimizer.acc_store_counter = 0
    optimizer.acc_last_plain = None

    optimizer.history_device = history_device
    optimizer.compute_device = compute_device

    # redefine step of the optimizer
    optimizer.orig_step = optimizer.step

    if acc_type != "identity":
        optimizer.step = MethodType(_unified_step, optimizer)
    elif average:
        optimizer.step = MethodType(averaged_step, optimizer)
    # else: leave step unchanged (identity, no averaging => no-op wrapper)

    return optimizer


_ACC_ATTRS = (
    "acc_type", "acc_wait_iterations", "acc_relaxation", "acc_history_depth",
    "acc_frequency", "acc_store_each_nth", "acc_reg",
    "acc_dtype",
    "acc_equilibrate", "acc_filter_condition", "acc_refinement_steps",
    "acc_safeguard",
    "acc_average_pre_step",
    "acc_param_hist", "avg_param_hist",
    "acc_call_counter", "acc_store_counter",
    "acc_last_plain",
    "orig_step", "history_device", "compute_device",
)


def reset_acceleration_history(optimizer):
    """Discard history after an external operation changes model parameters.

    Call this after native PyTorch model averaging (for example,
    ``PeriodicModelAverager.average_parameters``) so subsequent Anderson
    extrapolations do not mix pre- and post-consensus trajectories.
    """
    if not hasattr(optimizer, "acc_param_hist"):
        raise ValueError("optimizer is not wrapped by AADL")
    for state in optimizer.acc_param_hist:
        state["buf"] = None
        state["count"] = 0
    for history in optimizer.avg_param_hist:
        history.clear()
    optimizer.acc_store_counter = 0
    return optimizer


def remove_acceleration(optimizer):
    if not hasattr(optimizer, 'acc_type'):
        return

    optimizer.step = optimizer.orig_step
    for attr in _ACC_ATTRS:
        if hasattr(optimizer, attr):
            delattr(optimizer, attr)

    return optimizer
