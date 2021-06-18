from __future__ import print_function
from itertools import count
from copy import deepcopy

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

validation_classic_loss_history = []
validation_anderson_loss_history = []
validation_average_loss_history = []


torch.manual_seed(0)

colors = ["tab:blue", "tab:orange", "tab:green"]

plt.rcParams.update({'font.size': 16})

def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=300):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


# Define model
fc = torch.nn.Linear(W_target.size(0), 1)
fc_anderson = deepcopy(fc)
fc_average = deepcopy(fc)

lr = 1e-4
max_iters = 500
optim = torch.optim.SGD(fc.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

for idx in range(max_iters):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    optim.zero_grad()

    # Forward pass
    output = F.mse_loss(fc(batch_x), batch_y)
    loss = output.item()
    validation_classic_loss_history.append(loss)

    # Backward pass
    output.backward()
    optim.step()

    # Stop criterion
    if loss < 1e-4:
        break

print('Loss: {:.6f} after {} batches'.format(loss, idx))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))


import AADL as accelerate


# Parameters for Anderson acceleration
lr = 1e-2
relaxation = 0.1
wait_iterations = 1
history_depth = 5
store_each_nth = 3
frequency = store_each_nth
reg_acc = 0
average = False
optimizer_anderson= torch.optim.SGD(fc_anderson.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
accelerate.accelerate(optimizer_anderson, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average)

for idx in range(max_iters):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    optimizer_anderson.zero_grad()

    # Forward pass
    output = F.mse_loss(fc_anderson(batch_x), batch_y)
    loss = output.item()
    validation_anderson_loss_history.append(loss)
    
    def closure():
        if torch.is_grad_enabled():
            optimizer_anderson.zero_grad()
            output = fc_anderson(batch_x)
            loss = F.smooth_l1_loss(output, batch_y)
            if loss.requires_grad:
                loss.backward()
        return loss    

    # Backward pass
    output.backward()
    optimizer_anderson.step(closure)

    # Stop criterion
    if loss < 1e-4:
        break

print('Loss: {:.6f} after {} batches'.format(loss, idx))
print('==> Learned function:\t' + poly_desc(fc_anderson.weight.view(-1), fc_anderson.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))

average = True

optimizer_average= torch.optim.SGD(fc_average.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
accelerate.accelerate(optimizer_average, "anderson", relaxation, wait_iterations, history_depth, store_each_nth, frequency, reg_acc, average)

for idx in range(max_iters):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    optimizer_anderson.zero_grad()

    # Forward pass
    output = F.mse_loss(fc_average(batch_x), batch_y)
    loss = output.item()
    validation_average_loss_history.append(loss)
    
    def closure():
        if torch.is_grad_enabled():
            optimizer_average.zero_grad()
            output = fc_average(batch_x)
            loss = F.smooth_l1_loss(output, batch_y)
            if loss.requires_grad:
                loss.backward()
        return loss    

    # Backward pass
    output.backward()
    optimizer_average.step(closure)

    # Stop criterion
    if loss < 1e-4:
        break

print('Loss: {:.6f} after {} batches'.format(loss, idx))
print('==> Learned function:\t' + poly_desc(fc_average.weight.view(-1), fc_average.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))

epochs1 = range(1, len(validation_classic_loss_history) + 1)
epochs2 = range(1, len(validation_anderson_loss_history) + 1)
epochs3 = range(1, len(validation_average_loss_history) + 1)

plt.plot(epochs1,validation_classic_loss_history,color=colors[0],linestyle='-', linewidth=4)
plt.plot(epochs2,validation_anderson_loss_history,color=colors[1],linestyle='-', linewidth=4)
plt.plot(epochs3,validation_average_loss_history,color=colors[2],linestyle='-', linewidth=4)
 
plt.yscale('log')
plt.title('Polynomial fitting')
plt.xlabel('Epochs')
plt.ylabel('Validation Measn Squared Error')
plt.legend(["NSGD", "NSGD + AA", "NSDG + Average + AA"])
plt.tight_layout()
plt.draw()
plt.savefig('validation_loss_plot')
