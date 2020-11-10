import sys
from copy import deepcopy

import matplotlib.pyplot as plt

sys.path.append("./utils")
sys.path.append("./modules")
from modules.NN_models import *
from modules.optimizers import *
from utils.dataloaders import *

plt.rcParams.update({'font.size': 16})

# Import data
#input_dim, output_dim, dataset = graduate_admission_data()
#input_dim, output_dim, dataset = mnist_data()
input_dim, output_dim, dataset = cifar10_data()

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
#model_fixedpoint = MLP(input_dim, output_dim, num_neurons_list, use_bias, 'relu')
model_fixedpoint = CNN2D(input_dim, output_dim, num_neurons_list, use_bias, 'relu', True)
#model_fixedpoint.set_coefficients_to_random()

model_acceleration = deepcopy(model_fixedpoint)

# Set up optimizer
optimizer1 = FixedPointIteration(dataloader, learning_rate, weight_decay)

# Import neural network in optimizer
optimizer1.import_model(model_fixedpoint)
optimizer1.set_loss_function('ce')
optimizer1.set_optimizer('adam')
training1_loss_history = optimizer1.train(max_iterations, threshold, batch_size)

# Set up optimizer
optimizer2 = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth, frequency, reg_acc, store_each)

# Import neural network in optimizer
optimizer2.import_model(model_acceleration)
optimizer2.set_loss_function('ce')
optimizer2.set_optimizer('adam')
training2_loss_history = optimizer2.train(max_iterations, threshold, batch_size)

epochs1 = range(1, len(training1_loss_history) + 1)
epochs2 = range(1, len(training2_loss_history) + 1)
plt.plot(epochs1, training1_loss_history, label='training loss - Fixed Point')
plt.plot(epochs2, training2_loss_history, label='training loss - RNA')
plt.yscale('log')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.draw()
plt.savefig('loss_plot')
