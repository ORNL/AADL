# Anderson Accelerated Deep Learning (AADL)

AADL is a Python package that implements the Anderson acceleration to speed-up the training of deep learning (DL) models implemented using the PyTorch library.

## Requirements
Python 3.5 or greater\
PyTorch (any version works)

## Installation

python3 setup.py install

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
