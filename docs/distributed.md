# Distributed Anderson acceleration

AADL delegates distributed training mechanics to PyTorch. It does not maintain
its own gradient synchronization, parameter all-reduce schedule, process-group
lifecycle, or LocalSGD/FedAvg implementation.

## Responsibility boundary

PyTorch owns:

- `DistributedDataParallel` and gradient buckets;
- `post_localSGD_hook` and local/subgroup gradient communication;
- `PeriodicModelAverager` and global parameter averaging;
- process groups, backends, rank membership, and collective ordering.

AADL owns:

- local Anderson histories and mixing coefficients;
- construction of local plain and Anderson branches;
- invalidation of history after native model averaging;
- scalar acceptance policies at global averaging boundaries.

Coordinate sketching remains rank-local, just like each rank's Anderson
history and coefficients. At a model-averaging boundary, the existing vote or
sample-weighted mean-loss policy evaluates the resulting globally averaged
candidate. Configure a fixed sketch with `safeguard=False` for this mode;
adaptive sketch retries currently use the local closure safeguard and are not
replayed after a global boundary rejection.

## Execution sequence

During a normal local step:

1. DDP/Post-LocalSGD performs the configured gradient communication.
2. The underlying optimizer applies its update.
3. AADL stores the plain parameters and constructs a local Anderson candidate.
4. Between global averaging boundaries, ranks continue their local trajectories.

At a global averaging boundary, `average_and_accept` performs:

1. PyTorch-native averaging of the local Anderson candidates.
2. PyTorch-native averaging of the saved local plain updates.
3. Loss-only evaluation of both shared global vectors on every rank.
4. Scalar reduction using the selected acceptance policy.
5. A common accept/reject decision on all ranks.
6. Anderson-history reset, because global averaging changes the trajectory.

Every rank must call `average_and_accept` in the same order. A rank must not
skip a boundary call independently, or collective execution can deadlock.

## Native PyTorch setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState,
    post_localSGD_hook,
)

import AADL

dist.init_process_group(backend="nccl")
model = DDP(module.to(local_rank), device_ids=[local_rank])

warmup_steps = 100
period = 4

state = PostLocalSGDState(
    process_group=None,
    subgroup=None,
    start_localSGD_iter=warmup_steps,
)
model.register_comm_hook(state, post_localSGD_hook)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
AADL.accelerate(
    optimizer,
    acceleration_type="anderson",
    safeguard=False,
)

averager = AADL.HistoryResetPeriodicModelAverager(
    optimizer,
    period=period,
    warmup_steps=warmup_steps,
)
```

The native `PostLocalSGDOptimizer` wrapper does not accept an optimizer closure.
AADL's boundary comparison needs a loss-evaluation closure, so the documented
integration calls the native averager explicitly after `optimizer.step`.

```python
def closure():
    output = model(inputs)
    loss = loss_fn(output, targets)
    if torch.is_grad_enabled():
        optimizer.zero_grad()
        loss.backward()
    return loss

optimizer.step(closure)
result = AADL.average_and_accept(
    optimizer,
    averager,
    closure,
    policy="vote",
    vote_threshold=0.5,
    loss_weight=targets.size(0),
)
```

The closure is used with gradients by the optimizer step. AADL invokes the same
closure under `torch.no_grad()` for boundary loss comparisons, so it must guard
backward work with `torch.is_grad_enabled()` as shown above.

## Acceptance policies

### Vote

Each rank compares the shared candidate and baseline on its local data. The
candidate is accepted when

```text
improving_ranks / world_size >= vote_threshold
```

This policy favors broad benefit across ranks and can be appropriate for
heterogeneous data. `vote_threshold` must be in `[0, 1]`.

### Sample-weighted mean loss

Each rank contributes its loss difference and a positive `loss_weight`. AADL
accepts when

```text
sum_r loss_weight_r * (candidate_loss_r - baseline_loss_r) < 0
```

Use the local sample count as `loss_weight` to approximate the global empirical
objective when ranks process different batch sizes.

## History lifecycle

`HistoryResetPeriodicModelAverager` subclasses PyTorch's
`PeriodicModelAverager`. It delegates the collective unchanged and adds only a
call to `reset_acceleration_history` after an actual averaging boundary.

If another native PyTorch component changes parameters outside this averager,
call:

```python
AADL.reset_acceleration_history(optimizer)
```

Keeping pre-consensus history after parameters are replaced would mix iterates
from different local trajectories in the next Anderson solve.

## Communication cost

Between boundaries, AADL adds no distributed communication. At a boundary,
PyTorch averages the candidate and baseline parameter branches, and AADL
performs one small scalar reduction for the acceptance decision. The two
loss-only evaluations do not run backward or synchronize gradients.
