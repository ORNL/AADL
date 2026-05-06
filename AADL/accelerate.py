import os

import torch

from AADL.utils import parameters_to_vector_device, vector_to_parameters

from collections import deque
from types import MethodType

import AADL.anderson_acceleration as anderson


def _dist_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def _dist_local_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def _store_current_params(self):
    """Append current parameters to the acceleration history if it's time to."""
    self.acc_store_counter += 1
    if self.acc_store_counter >= self.acc_store_each_nth:
        self.acc_store_counter = 0
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(parameters_to_vector_device(group['params'], self.history_device))


def _moving_average_step(self):
    """Replace current parameters with a moving average of the recent history."""
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        group_hist.append(parameters_to_vector_device(group['params'], self.history_device))

    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)
        average = torch.mean(X, dim=1)
        std = torch.std(X, dim=1)
        if torch.max(std) / torch.max(average) > 0.1:
            vector_to_parameters(average, group['params'])


def _maybe_sync_acc_param(self, acc_param):
    """All-reduce-mean acc_param across ranks if distributed and time to sync."""
    if not self.acc_distributed:
        return acc_param
    world_size = _dist_world_size()
    if world_size <= 1:
        return acc_param
    self.acc_sync_counter += 1
    if self.acc_sync_counter % self.acc_sync_frequency == 0:
        self.acc_sync_counter = 0
        torch.distributed.all_reduce(acc_param, op=torch.distributed.ReduceOp.SUM)
        acc_param = acc_param / world_size
    return acc_param


def _safeguard_accept(self, closure, orig_loss):
    """Decide whether to accept the accelerated step.

    Returns (accept: bool, acc_loss). When closure is None, the step is
    always accepted (no information available to compare). In distributed
    mode, ranks vote and accept when the fraction agreeing exceeds
    ``acc_vote_threshold``.
    """
    if closure is None:
        return True, orig_loss

    acc_loss = closure()
    if not self.acc_distributed or _dist_world_size() <= 1:
        return acc_loss < orig_loss, acc_loss

    acc_vote = (acc_loss < orig_loss).float()
    torch.distributed.all_reduce(acc_vote, op=torch.distributed.ReduceOp.SUM)
    acc_vote = acc_vote / _dist_world_size()
    return acc_vote.item() > self.acc_vote_threshold, acc_loss


def _debug_log_divergence(self, group_hist, acc_param, closure_used, accepted):
    if not self.acc_debug:
        return
    if not (self.acc_distributed and _dist_world_size() > 1):
        return
    rank = _dist_local_rank()
    world_size = _dist_world_size()
    history_list = [torch.zeros_like(group_hist[-1]) for _ in range(world_size)] if rank == 0 else None
    acc_param_list = [torch.zeros_like(acc_param) for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather(group_hist[-1], gather_list=history_list, dst=0)
    torch.distributed.gather(acc_param, gather_list=acc_param_list, dst=0)
    if rank == 0:
        diff_history = sum((h - history_list[0]) for h in history_list)
        diff_param = sum((p - acc_param_list[0]) for p in acc_param_list)
        print(
            f"rel_history diff: {diff_history.abs().max().item() / history_list[0].abs().max().item():.2e}, "
            f"rel_acc_diff: {diff_param.abs().max().item() / acc_param_list[0].abs().max().item():.2e}, "
            f"accepted: {accepted}"
        )


@torch.no_grad()
def _unified_step(self, closure=None):
    """Single step implementation covering plain, distributed, and averaged variants."""
    if self.acc_average_pre_step:
        # moving-average sweep before the underlying optimizer step
        _moving_average_step(self)

    orig_loss = self.orig_step(closure)

    _store_current_params(self)

    self.acc_call_counter += 1
    ready = (
        self.acc_call_counter > self.acc_wait_iterations
        and self.acc_call_counter % self.acc_frequency == 0
    )
    if not ready:
        return orig_loss

    final_loss = orig_loss
    accel_fn = anderson.get_acceleration(self.acc_type)

    for group, group_hist in zip(self.param_groups, self.acc_param_hist):
        if len(group_hist) < 3:
            continue

        X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)
        acc_param = accel_fn(X, self.acc_relaxation, self.acc_reg)

        acc_param = _maybe_sync_acc_param(self, acc_param)

        # apply candidate acceleration
        vector_to_parameters(acc_param, group['params'])

        accepted, acc_loss = _safeguard_accept(self, closure, orig_loss)

        if accepted:
            # replace last history entry with the accepted accelerated point
            group_hist.pop()
            group_hist.append(
                acc_param.detach().to(
                    device=self.history_device, memory_format=torch.contiguous_format
                )
            )
            final_loss = acc_loss
        else:
            # revert to the non-accelerated parameters
            vector_to_parameters(group_hist[-1], group['params'])

        _debug_log_divergence(self, group_hist, acc_param, closure is not None, accepted)

    return final_loss


# ---------------------------------------------------------------------------
# Backwards-compatible aliases so external code importing the old function
# names keeps working.
accelerated_step = _unified_step
distributed_accelerated_step = _unified_step
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
    distributed: bool = False,
    sync_frequency: int = 1,
    vote_threshold: float = 0.9,
    debug: bool = False,
):
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
    optimizer.acc_sync_frequency  = sync_frequency
    optimizer.acc_reg             = reg_acc
    optimizer.acc_distributed     = distributed
    optimizer.acc_vote_threshold  = vote_threshold
    optimizer.acc_debug           = debug
    optimizer.acc_average_pre_step = average and acc_type != "identity"

    # acceleration history
    optimizer.acc_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]
    optimizer.avg_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]

    optimizer.acc_call_counter  = 0
    optimizer.acc_store_counter = 0
    optimizer.acc_sync_counter  = 0

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


def distributed_accelerate(optimizer, **kwargs):
    return accelerate(optimizer, **kwargs, distributed=True)


_ACC_ATTRS = (
    "acc_type", "acc_wait_iterations", "acc_relaxation", "acc_history_depth",
    "acc_frequency", "acc_sync_frequency", "acc_store_each_nth", "acc_reg",
    "acc_distributed", "acc_vote_threshold", "acc_debug", "acc_average_pre_step",
    "acc_param_hist", "avg_param_hist",
    "acc_call_counter", "acc_store_counter", "acc_sync_counter",
    "orig_step", "history_device", "compute_device",
)


def remove_acceleration(optimizer):
    if not hasattr(optimizer, 'acc_type'):
        return

    optimizer.step = optimizer.orig_step
    for attr in _ACC_ATTRS:
        if hasattr(optimizer, attr):
            delattr(optimizer, attr)

    return optimizer
