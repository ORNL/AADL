import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import torch

sys.path.append("./utils")
sys.path.append("./modules")
from modules.NN_models import MLP, CNN2D
from modules.optimizers import FixedPointIteration, RNA_Acceleration
from utils.dataloaders import graduate_admission_data, mnist_data, cifar10_data

plt.rcParams.update({'font.size': 16})

# Import data
input_dim, output_dim, dataset = graduate_admission_data()
#input_dim, output_dim, dataset = mnist_data()
#input_dim, output_dim, dataset = cifar10_data()

print("Finished importing data")

# Setting for the neural network
num_neurons_list = [10, 10]
use_bias = True

# Parameters for generic optimizer
max_iterations = 50
learning_rate = 1e-2
threshold = 1e-7
batch_size = 50
weight_decay = 0.0

# Parameters for RNA
wait_iterations = 50
window_depth = 10
frequency = 5
store_each = 1
reg_acc = 0.0

# Generate dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size)

# Define deep learning model
model_fixedpoint = MLP(input_dim, output_dim, num_neurons_list, use_bias, 'relu')
#model_fixedpoint = CNN2D(input_dim, output_dim, num_neurons_list, use_bias, 'relu', True)
#model_fixedpoint.set_coefficients_to_random()

model_acceleration = deepcopy(model_fixedpoint)

optimizer_classis = FixedPointIteration(dataloader, learning_rate, weight_decay)

# Import neural network in optimizer
optimizer_classis.import_model(model_fixedpoint)
optimizer_classis.set_loss_function('mse')
optimizer_classis.set_optimizer('adam')
training_classis_loss_history = optimizer_classis.train(max_iterations, threshold, batch_size)

# Set up optimizer
optimizer_anderson = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth, frequency, reg_acc, store_each)

# Import neural network in optimizer
optimizer_anderson.import_model(model_acceleration)
optimizer_anderson.set_loss_function('mse')
optimizer_anderson.set_optimizer('adam')
training_anderson_loss_history = optimizer_anderson.train(max_iterations, threshold, batch_size)

epochs1 = range(1, len(training_classis_loss_history) + 1)
epochs2 = range(1, len(training_anderson_loss_history) + 1)
plt.plot(epochs1, training_classis_loss_history, label='training loss - Fixed Point')
plt.plot(epochs2, training_anderson_loss_history, label='training loss - RNA')
plt.yscale('log')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.draw()
plt.savefig('loss_plot')
