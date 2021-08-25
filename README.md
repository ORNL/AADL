# Anderson Accelerated Deep Learning (AADL)

AADL is a Python package that implements the Anderson acceleration to speed-up the training of deep learning (DL) models using the PyTorch library.\
AA is an extrapolation technique that can accelerate fixed-point iterations such those arising from the iterative trainingof DL models. However, large volume of data are typically processed in sequential random batches which introduces stochastic oscillations in the fixed-point iteration that hinders AA acceleration. AADL implements a moving average that reduces the oscillations and results in a smoother sequence of gradient descent updates which enables the use of AA. AADL uses a criterion to automatically decide if the moving average is needed by monitoring if the relative standard deviation be-tween consecutive stochastic gradient updates exceeds a tolerance defined by the user.

## Requirements
Python 3.5 or greater\
PyTorch (any version works)

## Installation

AADL comes with a ```python setuptools``` install script:

```python
python3 setup.py install
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
AADL.accelerate(optimizer_anderson, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html)

### Citations
"AADL: Anderson Accelerated Deep Learning", Copyright ID#: 81927550 
https://doi.org/10.11578/dc.20210723.1
