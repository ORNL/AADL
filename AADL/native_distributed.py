"""Thin adapters for PyTorch-native distributed training components."""

import torch

from torch.distributed.algorithms.model_averaging.averagers import (
    PeriodicModelAverager,
)
from torch.distributed.algorithms.model_averaging.utils import average_parameters

from AADL.accelerate import reset_acceleration_history
from AADL.distributed import accept_candidate, validate_acceptance_policy


class HistoryResetPeriodicModelAverager(PeriodicModelAverager):
    """Native ``PeriodicModelAverager`` that invalidates stale AADL history.

    All communication is implemented by PyTorch.  The only added behavior is
    clearing Anderson history after PyTorch changes parameters by global model
    averaging.
    """

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(optimizer, "acc_param_hist"):
            raise ValueError("optimizer must be wrapped by AADL first")
        self.aadl_optimizer = optimizer

    def average_parameters(self, params):
        self.last_did_average = (
            self.step >= self.warmup_steps
            and (self.step - self.warmup_steps) % self.period == 0
        )
        result = super().average_parameters(params)
        if self.last_did_average:
            reset_acceleration_history(self.aadl_optimizer)
        return result


def average_and_accept(
    optimizer,
    averager,
    closure,
    *,
    policy="vote",
    vote_threshold=0.5,
    loss_weight=1.0,
):
    """Run native averaging and select the global plain or Anderson iterate.

    Local plain parameters saved by AADL and the current local Anderson
    parameters are averaged using PyTorch's native model-averaging code. Every
    rank then evaluates the same two global parameter vectors, and AADL reduces
    only the scalar acceptance statistic. ``closure`` is evaluated under
    ``torch.no_grad()`` so the comparison does not trigger extra DDP gradient
    synchronization.

    Configure the optimizer with ``safeguard=False`` when this function owns
    acceptance at averaging boundaries.
    """
    policy = validate_acceptance_policy(policy)
    if policy == "local":
        raise ValueError("average_and_accept requires 'vote' or 'mean_loss'")
    if not isinstance(averager, HistoryResetPeriodicModelAverager):
        raise TypeError("averager must be a HistoryResetPeriodicModelAverager")
    if averager.aadl_optimizer is not optimizer:
        raise ValueError("averager and optimizer do not refer to the same optimizer")

    averager.average_parameters(optimizer.param_groups)
    if not averager.last_did_average or optimizer.acc_last_plain is None:
        return None

    tracked = optimizer.acc_last_plain
    candidate = [(param, param.detach().clone()) for param, _ in tracked]
    plain = [snapshot for _, snapshot in tracked]

    # Use PyTorch's native flattened parameter-averaging implementation for
    # the plain branch as well as for the candidate branch above.
    average_parameters(iter(plain), averager.process_group)

    with torch.no_grad():
        for (param, _), snapshot in zip(tracked, plain):
            param.copy_(snapshot)
    with torch.no_grad():
        baseline_loss = closure()

    with torch.no_grad():
        for param, snapshot in candidate:
            param.copy_(snapshot)
    with torch.no_grad():
        candidate_loss = closure()

    accepted = accept_candidate(
        candidate_loss,
        baseline_loss,
        policy=policy,
        vote_threshold=vote_threshold,
        loss_weight=loss_weight,
        process_group=averager.process_group,
    )
    if not accepted:
        with torch.no_grad():
            for (param, _), snapshot in zip(tracked, plain):
                param.copy_(snapshot)
    optimizer.acc_last_plain = None
    return accepted, candidate_loss, baseline_loss
