# Anderson Accelerated Deep Learning (AADL)

AADL is a Python package that implements the Anderson acceleration to speed-up the training of deep learning (DL) models using the PyTorch library.\
AA is an extrapolation technique that can accelerate fixed-point iterations such those arising from the iterative training of DL models. However, large volume of data are typically processed in sequential random batches which introduces stochastic oscillations in the fixed-point iteration that hinders AA acceleration. AADL implements a moving average that reduces the oscillations and results in a smoother sequence of gradient descent updates which enables the use of AA. AADL uses a criterion to automatically decide if the moving average is needed by monitoring if the relative standard deviation between consecutive stochastic gradient updates exceeds a tolerance defined by the user.

## Requirements
Python 3.8 or greater\
PyTorch (`torch>=2.1`) and NumPy

> `torch>=2.1` is required because the acceleration kernels use
> `torch.linalg.solve_triangular` and `torch._foreach_copy_`.

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

### Running the tests

```bash
python -m unittest discover -s tests -t . -v          # fast suite
RUN_SLOW_TESTS=1 python -m unittest discover -s tests -t . -v   # full suite
```

## Usage

```python
import torch
import torch.nn
import torch.optim
import AADL

# Creation of the DL model (neural network)
class model(torch.nn.Module):
	...

# Definition of the stochastic optimizer used to train the model
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov = True)

# Parameters for Anderson acceleration
relaxation = 0.5
wait_iterations = 0
history_depth = 10
store_each_nth = 10
frequency = store_each_nth
reg_acc = 0.0
safeguard = True
average = True

# Over-writing of the torch.optim.step() method 
AADL.accelerate(
    optimizer,
    acceleration_type="anderson",
    relaxation=relaxation,
    wait_iterations=wait_iterations,
    history_depth=history_depth,
    store_each_nth=store_each_nth,
    frequency=frequency,
    reg_acc=reg_acc,
    average=average,
    safeguard=safeguard,
)

```

## Meaning of hyperparameters
```relaxation```: Float. Linear mixing parameter between a standard gradient descent update and the Anderson update\
```wait_iterations```: Integer. Number of initial gradient descent updates to wait before starting the Anderson scheme\
```history_depth```: Integer. Number of gradient updates used to compute the Anderson mixing. The history is updated with a first-in-first-out policy\
```store_each_nth```: Integer. Number of gradient updates to skip between two vector updates consecutively stored in the history window\
```frequency```: Integer. Number of gradient updates to skip between two consecutive Anderson steps\
```reg_acc```: Float. Tikhonov regularization factor used to stabilize the least-squares problem solved to compute the Anderson mixing vector\
```mixing_dtype```: ``None``, ``torch.dtype``, or string (e.g. ``"float32"``, ``"float64"``). Floating-point precision at which the Anderson mixing vector is computed. ``None`` keeps the parameter dtype; a lower precision speeds up the least-squares solve while the extrapolated parameters are always cast back to their original dtype. Portable choices are ``float32`` and ``float64``; ``float16``/``bfloat16`` are not supported by ``torch.linalg`` on CPU (and only conditionally on GPU) and raise a clear error if requested there\
```equilibrate```: Boolean (default ``True``). Scales the columns of the difference matrix to unit norm before the least-squares solve, which improves conditioning. It is an exact change of variables for the unregularized, full-rank problem (the mixing vector is unchanged up to round-off) and is generally beneficial\
```filter_condition```: Float (default ``0.0``, disabled). If greater than ``0``, oldest history columns are dropped (Walker-Ni filtering) until the 2-norm condition number of the least-squares matrix falls below this threshold. The condition number is estimated cheaply from the small Gram matrix. Use it to stabilize the solve when the history becomes nearly rank-deficient\
```refinement_steps```: Integer (default ``0``, disabled). Number of mixed-precision iterative-refinement steps applied to the mixing vector: the residual is formed in the parameter precision while the correction reuses the reduced-precision factor, recovering accuracy cheaply when ``mixing_dtype`` is lower than the parameter dtype. A monotone guard rejects any non-improving step, so refinement never diverges on ill-conditioned systems (it becomes a no-op instead)\
```safeguard```: Boolean. If set to True, the Anderson step is kept only when it strictly reduces the loss relative to the plain optimizer step it would otherwise revert to (the comparison uses the post-step loss, so a candidate that is worse than not accelerating is rejected). Non-finite (``NaN``/``Inf``) candidates from a degenerate solve are also rejected. Requires passing a ``closure`` to ``optimizer.step``; with no closure the safeguard is disabled and the accelerated step is always accepted\
```average```: Boolean. If set to True, a movign average is applied to the history window before computing the Anderson step\ 


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html)

### Software citation
M. Lupo Pasini, V. Reshniak, and M. K. Stoyanov. AADL: Anderson Accelerated Deep Learning. Computer Software. https://github.com/ORNL/AADL.git. 06 Sep. 2021. Web. doi:10.11578/dc.20210723.1. Copyright ID#: 81927550 


### Publications
M. Lupo Pasini, J. Yin, V. Reshniak and M. K. Stoyanov, "Anderson Acceleration for Distributed Training of Deep Learning Models," SoutheastCon 2022, 2022, pp. 289-295, doi: 10.1109/SoutheastCon48659.2022.9763953.
