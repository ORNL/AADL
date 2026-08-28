"""Distributed acceptance policies for Anderson candidates.

PyTorch owns gradient and parameter synchronization.  This module only
reduces scalar loss statistics that are specific to deciding whether an
Anderson candidate should be accepted.
"""

import math

import torch
import torch.distributed as dist


ACCEPTANCE_POLICIES = ("local", "vote", "mean_loss")


def validate_acceptance_policy(policy):
    if not isinstance(policy, str):
        raise ValueError(f"acceptance_policy must be one of {ACCEPTANCE_POLICIES}")
    policy = policy.lower()
    if policy not in ACCEPTANCE_POLICIES:
        raise ValueError(f"acceptance_policy must be one of {ACCEPTANCE_POLICIES}")
    return policy


def accept_candidate(
    candidate_loss,
    baseline_loss,
    *,
    policy="local",
    vote_threshold=0.5,
    loss_weight=1.0,
    process_group=None,
):
    """Return a common accept/reject decision for an Anderson candidate.

    ``vote`` accepts when at least ``vote_threshold`` of ranks improve.
    ``mean_loss`` accepts when the sample-weighted global loss difference is
    negative.  ``local`` performs no communication.

    For a global interpretation, all ranks must evaluate the same baseline and
    candidate parameters.  If ranks evaluate local candidates, the result only
    summarizes improvement of those rank-local candidates.
    """
    policy = validate_acceptance_policy(policy)
    if (not isinstance(vote_threshold, (int, float))
            or isinstance(vote_threshold, bool)
            or not math.isfinite(vote_threshold)
            or not 0.0 <= vote_threshold <= 1.0):
        raise ValueError("vote_threshold must be in [0, 1]")
    if (not isinstance(loss_weight, (int, float))
            or isinstance(loss_weight, bool)
            or not math.isfinite(loss_weight)
            or loss_weight <= 0.0):
        raise ValueError("loss_weight must be positive")
    improved = candidate_loss < baseline_loss
    if policy == "local":
        return bool(improved)

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"acceptance_policy={policy!r} requires an initialized "
            "torch.distributed process group"
        )

    reference = candidate_loss if torch.is_tensor(candidate_loss) else baseline_loss
    if not torch.is_tensor(reference):
        reference = torch.tensor(float(candidate_loss))
    device = reference.device

    if policy == "vote":
        statistic = torch.tensor(float(improved), device=device, dtype=torch.float64)
        dist.all_reduce(statistic, op=dist.ReduceOp.SUM, group=process_group)
        fraction = statistic / dist.get_world_size(process_group)
        return bool(fraction.item() >= vote_threshold)

    delta = torch.as_tensor(candidate_loss, device=device, dtype=torch.float64)
    delta = delta - torch.as_tensor(baseline_loss, device=device, dtype=torch.float64)
    weight = torch.as_tensor(loss_weight, device=device, dtype=torch.float64)
    totals = torch.stack((delta.detach() * weight, weight))
    dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
    if totals[1].item() <= 0:
        raise ValueError("the globally reduced loss_weight must be positive")
    return bool(totals[0].item() < 0.0)
