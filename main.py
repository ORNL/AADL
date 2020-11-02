from NN_models import *
from optimizers import *
import matplotlib.pyplot as plt
import sys

sys.path.append("./utils")
from dataloaders import *

plt.rcParams.update({'font.size': 16})

# Import data
#input_dim, output_dim, dataset = graduate_admission_data()
input_dim, output_dim, dataset = mnist_data()

# Generate dataloader
dataloader = torch.utils.data.DataLoader(dataset)

print("Finished importing data")

# Setting for the neural network
num_neurons_list = [10, 10]
use_bias = True

# Parameters for optimizer
window_depth = 10
frequency = 5
max_iterations = 5000
learning_rate = 1e-2
threshold = 1e-7
batch_size = 100
weight_decay = 1e-5

# Define deep learning model
#model = MLP(input_dim, output_dim, num_neurons_list, use_bias, 'relu')
model = CNN2D(input_dim, output_dim, num_neurons_list, use_bias, 'relu', True)
#model.set_coefficients_to_random()

# Set up optimizer
optimizer = FixedPointIteration(dataloader, learning_rate, weight_decay)

# Import neural network in optimizer
optimizer.import_model(model)
optimizer.set_loss_function('ce')
optimizer.set_optimizer('adam')

training_loss_history = optimizer.train(max_iterations, threshold, batch_size)

epochs = range(1, len(training_loss_history) + 1)
plt.plot(epochs, training_loss_history, 'b', label='training loss')
plt.yscale('log')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.draw()
plt.savefig('loss_plot')
