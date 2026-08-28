# Anderson Accelerated Deep Learning (AADL)

AADL adds Anderson acceleration to existing PyTorch optimizers. It stores a
bounded history of parameter iterates, solves a small least-squares problem,
and optionally replaces a normal optimizer update with an extrapolated one.

AADL supports QR and normal-equation Anderson kernels, safeguards, mixed
precision, conditioning controls, moving-average smoothing, multiple optimizer
parameter groups, and PyTorch-native Post-LocalSGD integration.

## Requirements
Python 3.11 or greater\
PyTorch (`torch>=2.13`) and NumPy (`numpy>=2.0`)

These minimum versions are enforced by the package metadata and
`requirements.txt`.

## Installation

The quickest way to get a working environment is the provided helper script,
which creates a local virtual environment (`.venv`), installs the dependencies,
and installs AADL in editable mode:

```bash
./setup_venv.sh                 # uses python3 by default
PYTHON=python3.11 ./setup_venv.sh   # pick a specific interpreter
./setup_venv.sh --recreate      # delete an existing .venv first
```

Then activate it:

```bash
source .venv/bin/activate
```

### Manual installation

If you prefer to manage your own environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt   # core dependencies
python -m pip install -e .                  # install AADL (editable)
```

The `examples/` demos need extra packages (torchvision, pandas, scikit-learn,
scikit-image, opencv-python, docopt, pyyaml). Install them with:

```bash
python -m pip install -r requirements-examples.txt
```

### Running tests

```bash
python -m unittest discover -s tests -t . -v          # fast suite
RUN_SLOW_TESTS=1 python -m unittest discover -s tests -t . -v   # full suite
```

The slow suite contains numerical convergence experiments whose results can be
sensitive to optimizer and PyTorch version changes. The fast suite contains the
API, kernel, safeguard, and distributed-policy regression tests.

## Architecture

AADL has three distinct responsibilities:

- `AADL.accelerate` wraps a local PyTorch optimizer and computes Anderson
  candidates. It does not synchronize gradients or model parameters.
- PyTorch DDP/Post-LocalSGD owns gradient communication, process groups, and
  periodic model averaging.
- AADL's distributed acceptance layer compares globally averaged plain and
  Anderson branches and reduces only scalar loss statistics.

This separation avoids maintaining a second implementation of DDP, LocalSGD,
or FedAvg-style parameter averaging inside AADL.

## Usage

```python
import torch
import torch.nn
import torch.optim
import AADL

model = torch.nn.Linear(8, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

AADL.accelerate(
    optimizer,
    acceleration_type="anderson",
    relaxation=0.5,
    wait_iterations=0,
    history_depth=10,
    store_each_nth=1,
    frequency=1,
    reg_acc=1e-8,
    safeguard=True,
)

def closure():
    with torch.enable_grad():
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
    return loss

loss = optimizer.step(closure)
```

### Acceleration implementations

- `anderson`: QR-factorization kernel and the recommended default.
- `anderson_normal_equation`: normal-equation kernel, which may be faster but
  is more sensitive to ill-conditioned histories.
- `identity`: disables Anderson acceleration; with `average=True`, it retains
  only moving-average behavior.

### Main options

- `relaxation`: mixing weight in `(0, 1]`.
- `wait_iterations`: ordinary optimizer steps before acceleration begins.
- `history_depth`: capacity of the FIFO/ring history.
- `store_each_nth`: cadence for storing parameter iterates.
- `frequency`: cadence for attempting Anderson acceleration.
- `reg_acc`: non-negative Tikhonov regularization.
- `safeguard`: compare the candidate against the post-optimizer plain step.
  This requires a closure; without one, the candidate is accepted.
- `average`: enable stochastic-history moving-average smoothing.
- `history_device` and `compute_device`: independently place stored history and
  the small Anderson solve.
- `mixing_dtype`: `None`, a `torch.dtype`, or a dtype string such as
  `"float32"` or `"float64"`.
- `equilibrate`: unit-scale difference-matrix columns before solving.
- `filter_condition`: drop oldest columns until the requested condition bound
  is met; `0` disables filtering.
- `refinement_steps`: mixed-precision iterative-refinement iterations; `0`
  disables refinement.

All size and cadence arguments are validated. Calling `accelerate` twice on the
same optimizer raises an error; call `AADL.remove_acceleration(optimizer)`
before changing its configuration.

## Distributed training

AADL composes with PyTorch's native Post-LocalSGD hook and model averager:

```python
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState,
    post_localSGD_hook,
)
from AADL import HistoryResetPeriodicModelAverager, average_and_accept

state = PostLocalSGDState(
    process_group=None,
    subgroup=None,
    start_localSGD_iter=100,
)
ddp_model.register_comm_hook(state, post_localSGD_hook)

local_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
AADL.accelerate(
    local_optimizer,
    acceleration_type="anderson",
    safeguard=False,  # acceptance is decided globally below
)

averager = HistoryResetPeriodicModelAverager(
    local_optimizer, period=4, warmup_steps=100,
)
# In the training loop, use a closure for the two global loss evaluations:
local_optimizer.step(closure)
average_and_accept(
    local_optimizer,
    averager,
    closure,
    policy="vote",            # or "mean_loss"
    vote_threshold=0.5,
    loss_weight=local_batch_size,
)
```

`average_and_accept` returns `None` between averaging boundaries. At a boundary
it returns `(accepted, candidate_loss, baseline_loss)` and leaves every rank on
the same selected global parameters.

Available global policies are:

- `vote`: accept when at least `vote_threshold` of ranks report a lower local
  loss for the shared global Anderson candidate.
- `mean_loss`: accept when the sample-weighted global loss difference is
  negative. Set `loss_weight` to the rank's local sample count.

Both branches are averaged through PyTorch's native model-averaging utilities.
Loss-only closure evaluations run under `torch.no_grad()`, so the closure must
guard backward work with `torch.is_grad_enabled()` to avoid extra DDP gradient
synchronization. See [Distributed training](docs/distributed.md) for the
execution sequence, policy semantics, and integration requirements.

## Reference models

Reusable reference models are available through stable package imports:

```python
from AADL.models import MLP
from AADL.models.vision import create_model, list_models

network = create_model("resnet18", num_classes=10)
print(list_models())
```

Numerical models used only by the test suite live in `tests.fixtures`. The
top-level `model_zoo` modules remain as source-tree compatibility shims for
legacy examples; new code should import from `AADL.models`.

## Public lifecycle helpers

- `AADL.reset_acceleration_history(optimizer)`: clear history after any
  external parameter-changing operation.
- `AADL.remove_acceleration(optimizer)`: restore the original optimizer step
  and remove AADL state.
- `AADL.accept_candidate(...)`: low-level scalar policy reducer for advanced
  integrations that already provide comparable candidate and baseline losses.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html)

### Software citation
M. Lupo Pasini, V. Reshniak, and M. K. Stoyanov. AADL: Anderson Accelerated Deep Learning. Computer Software. https://github.com/ORNL/AADL.git. 06 Sep. 2021. Web. doi:10.11578/dc.20210723.1. Copyright ID#: 81927550 


### Publications
M. Lupo Pasini, J. Yin, V. Reshniak and M. K. Stoyanov, "Anderson Acceleration for Distributed Training of Deep Learning Models," SoutheastCon 2022, 2022, pp. 289-295, doi: 10.1109/SoutheastCon48659.2022.9763953.
